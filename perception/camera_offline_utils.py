import time
import threading
import numpy as np
import cv2
import os

from camera_utils import CameraBackend

class OfflineCameraBackend(CameraBackend):
    """Camera backend that plays back an offline recording (.mp4 + .npz).

    Note: For GUI visualization in headless environments, use camera_visualizer:
        from camera_visualizer import create_visualizer
        viz = create_visualizer()  # Auto-detects headless and uses Viser if needed
        viz.show("Window Name", frame)
        key = viz.wait_key(1)
        viz.cleanup()
    """

    def __init__(self, recording_path: str, loop: bool = True):
        """Initialize OfflineCameraBackend.

        Args:
            recording_path: Path to recording (without extension, or with .mp4 or .npz).
            loop: Whether to loop the video playback when it reaches the end.
        """
        # Clean up path to get base name without extension
        if recording_path.endswith('.mp4') or recording_path.endswith('.npz'):
            self.base_path = recording_path[:-4]
        else:
            self.base_path = recording_path

        self.vid_path = f"{self.base_path}.mp4"
        self.meta_path = f"{self.base_path}.npz"

        if not os.path.exists(self.vid_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Recording not found at {self.base_path} (.mp4 or .npz missing)")

        super().__init__(self.base_path)

        # Load metadata
        try:
            self.metadata = np.load(self.meta_path, allow_pickle=True)
            self.K = self.metadata['K']
            self.fps = float(self.metadata['fps'])
            self.resolution = self.metadata['resolution']  # [width, height]
            self.recorded_tag_size = float(self.metadata['tag_size'])
            
            # Extract tag_sizes dict correctly depending on how it was saved
            if 'tag_sizes' in self.metadata:
                try: # If dict was saved inside object array
                   self.recorded_tag_sizes = self.metadata['tag_sizes'].item() 
                except: # If it's a bare value or other format
                   self.recorded_tag_sizes = self.metadata['tag_sizes']
            else:
                self.recorded_tag_sizes = {}
                
            self.is_undistorted = bool(self.metadata['undistorted'])
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {self.meta_path}: {e}")

        # Playback configuration
        self.loop = loop
        self.playback_speed = 1.0
        self.paused = False
        
        # We'll calculate sleep time dynamically based on playback speed
        self._base_sleep_time = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

        # Video capture object
        self.cap = None
        
        # Last read frame to hold when paused
        self._last_gray = None
        self._last_bgr = None

    def initialize(self, width: int = None, height: int = None, framerate: int = None) -> tuple:
        """Initialize video capture and buffers. Ignores input args, uses recorded settings."""
        def _init():
            try:
                self.cap = cv2.VideoCapture(self.vid_path)
                if not self.cap.isOpened():
                    print(f"Failed to open video: {self.vid_path}")
                    return

                # Read first frame to verify dimensions
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Failed to read first frame of: {self.vid_path}")
                    return
                
                # Reset to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Setup internal buffers
                actual_height, actual_width = frame.shape[:2]
                
                # Print warning if resolution doesn't match metadata (shouldn't happen with mp4v but good check)
                if list(self.resolution) != [actual_width, actual_height]:
                    print(f"Warning: Video resolution {actual_width}x{actual_height} differs from metadata {self.resolution[0]}x{self.resolution[1]}")
                
                self.gray_buf_internal = np.empty((actual_height, actual_width), dtype=np.uint8)
                self.frame_buf_internal = np.empty((actual_height, actual_width, 3), dtype=np.uint8)
                
                print(f"Offline backend initialized: {actual_width}x{actual_height} @ {self.fps:.1f}fps")
            except Exception as e:
                 print(f"Offline backend init failed: {e}")
            finally:
                self.init_complete.set()

        threading.Thread(target=_init, daemon=True).start()
        
        # We return the actual recorded resolution
        act_width, act_height = self.resolution
        return (act_width, act_height)

    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics"""
        self.init_complete.wait(timeout=4)
        return self.K

    def capture_frame(self, undistort: bool = True) -> bool:
        """Capture next frame from video file.
        
        Args:
            undistort: Ignored. Uses whatever state the video was recorded in.
        """
        if not self.init_complete.is_set():
            return False

        if self.paused and self._last_gray is not None:
            # Re-yield the same frame buffers
            np.copyto(self.gray_buf_internal, self._last_gray)
            np.copyto(self.frame_buf_internal, self._last_bgr)
            self.frame_ready.set()
            return True

        ret, frame = self.cap.read()
        
        if not ret:
            # End of video reached
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret: # Shouldn't happen unless file is corrupted
                    return False
            else:
                return False

        # Convert to BGR array for frame buffer (mp4v might load as BGR even if recorded grayscale)
        if len(frame.shape) == 2 or frame.shape[2] == 1:
             gray_frame = frame.reshape(frame.shape[0], frame.shape[1])
             bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        else:
             bgr_frame = frame
             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        np.copyto(self.gray_buf_internal, gray_frame)
        np.copyto(self.frame_buf_internal, bgr_frame)
        
        # Cache for pausing
        self._last_gray = self.gray_buf_internal.copy()
        self._last_bgr = self.frame_buf_internal.copy()

        # Signal that new frame is available
        self.frame_ready.set()
        return True

    def capture_frame_continuously(self):
        """Start continuous playback in background thread (non-blocking)"""
        if self.should_capture_frame_continuously:
            return
            
        def _capture_loop():
            self.init_complete.wait(timeout=5.0)

            while self.should_capture_frame_continuously:
                start_time = time.perf_counter()
                
                if not self.capture_frame():
                     # End of video reached (and not looping)
                     print("End of video reached.")
                     self.should_capture_frame_continuously = False
                     break
                
                # Calculate sleep time based on playback speed and processing time
                sleep_target = self._base_sleep_time / self.playback_speed
                elapsed = time.perf_counter() - start_time
                sleep_amount = sleep_target - elapsed
                
                if sleep_amount > 0:
                    time.sleep(sleep_amount)
                    
        self.should_capture_frame_continuously = True
        threading.Thread(target=_capture_loop, daemon=True).start()

    def get_gray_buffer(self) -> np.ndarray:
        return self.gray_buf_internal

    def get_frame_buffer(self) -> np.ndarray:
        return self.frame_buf_internal

    def cleanup(self):
        """Release video resources"""
        self.should_capture_frame_continuously = False
        if self.cap is not None:
            self.cap.release()

    # --- Playback Controls ---
    def set_playback_speed(self, speed: float):
        """Set the playback multiplier (e.g., 0.5 for half speed, 2.0 for double)"""
        if speed <= 0:
            self.paused = True
        else:
            self.playback_speed = float(speed)
            self.paused = False

    def toggle_pause(self):
        """Toggle play/pause state"""
        self.paused = not self.paused
        return self.paused
        
    def step_forward(self):
         """Step one frame forward (forces pause)"""
         self.paused = False
         self.capture_frame()
         self.paused = True
