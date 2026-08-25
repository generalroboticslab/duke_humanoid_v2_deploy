# sudo apt install v4l-utils
import cv2
import numpy as np
from abc import ABC, abstractmethod
from camera_visualizer import create_visualizer
import yaml
import os
import time
import subprocess
import threading

# fps helper class
class FPS:
    __slots__ = ('l','f','a','i','v')
    def __init__(s,i=0.5): s.l=time.perf_counter();s.f=0;s.a=0;s.i=i;s.v=0.0
    def tick(s):
        n=time.perf_counter();d=n-s.l;s.l=n;s.f+=1;s.a+=d
        if s.a>=s.i:s.v=s.f/s.a;s.f=s.a=0
        return d
    @property
    def fps(s): return s.v

# ============================================================================
# CAMERA BACKENDS - Strategy Pattern for different camera types
# ============================================================================

class CameraBackend(ABC):
    """Abstract interface for camera backends"""

    def __init__(self, device):
        """Initialize common camera backend components

        Args:
            device: Camera identifier (device path, serial number, etc.)
        """
        # Frame synchronization (Event has built-in lock + flag)
        self.frame_ready = threading.Event()

        # Camera identifier
        self.device = device

        # Frame buffers (allocated during initialize())
        self.gray_buf_internal = None
        self.frame_buf_internal = None

        # Async initialization synchronization
        self.init_complete = threading.Event()

        # Continuous capture control
        self.should_capture_frame_continuously = False

    @abstractmethod
    def initialize(self, width: int, height: int, framerate: int) -> tuple:
        """Initialize camera hardware

        Returns:
            tuple: (actual_width, actual_height)
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics

        Returns:
            np.ndarray: K_matrix - 3x3 camera intrinsic matrix
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        pass

    @abstractmethod
    def capture_frame(self, undistort: bool = True) -> bool:
        """Capture frame from camera and update internal buffers

        Args:
            undistort: If True and calibration available, apply undistortion to frames (default: True)
                      Ignored for cameras with factory calibration (e.g., RealSense)

        Returns:
            bool: True if frame was captured successfully
        """
        pass
    
    @abstractmethod
    def capture_frame_continuously(self):
        """Start camera capture in background thread (non-blocking)"""
        pass

    @abstractmethod
    def get_gray_buffer(self) -> np.ndarray:
        """Return internal gray buffer"""
        raise NotImplementedError

    @abstractmethod
    def get_frame_buffer(self) -> np.ndarray:
        """Return internal color buffer (BGR)"""
        raise NotImplementedError

    def wait_for_frame(self, timeout: float = None) -> bool:
        """Block until a new frame is available

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if frame ready, False if timeout
        """
        if self.frame_ready.wait(timeout):
            self.frame_ready.clear()  # Consume the signal
            return True
        return False

    @property
    @abstractmethod
    def cleanup(self):
        """Release camera resources"""
        pass


class USBCameraBackend(CameraBackend):
    """USB camera backend implementation"""

    def __init__(self, device, profile=None, autofocus=True, focus=None, passthrough=False):
        """Initialize USB camera backend

        Args:
            device: Device path (e.g., '/dev/video0') or index (e.g., 0)
            profile: Optional calibration dict from load_calibration()
                                 containing 'cam_params', 'camera_matrix', 'dist_coeffs', etc.
                                 If None, intrinsics will be estimated from resolution.
            autofocus: Enable continuous autofocus (default True). Set False for manual focus.
            focus: Manual focus value 0-100 (camera-dependent units). Only applied when autofocus=False.
                   0 = infinity (far), 100 = closest focus, 30 ≈ mid-far range
            passthrough: Keep the camera's own MJPEG compressed instead of decoding it.
                   These webcams already hand V4L2 a JPEG; the normal path decodes it to
                   BGR and a streaming server then re-encodes it, which costs CPU and
                   loses a generation of quality for no gain. In this mode
                   get_frame_buffer()/get_gray_buffer() stay EMPTY — only
                   get_encoded_frame() is valid — so it suits a pure publisher, not a
                   consumer that needs pixels. Falls back to decoding automatically if
                   the camera's buffers do not contain usable JPEG.
        """
        super().__init__(device)

        # Calibration profile
        if isinstance(profile, str):
            self.profile = self.load_calibration(profile)
        elif isinstance(profile, dict):
            self.profile = profile
        else:
            self.profile = None

        # Focus control
        self.autofocus = autofocus
        self.focus = focus

        # Pass-through state
        self.passthrough = passthrough
        self._raw_buf = None       # V4L2 hands back a 1xN uint8 buffer in this mode
        self._encoded_frame = None  # newest camera JPEG, trimmed to SOI..EOI

        # USB-specific variables
        self.cap = None
        self.undistort_temp_buf = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        self.K = None
        self.K_undistorted = None
        self.frame_counter = 0
        self.frame_sleep_time = None

    def _setup_undistortion(self, actual_width: int, actual_height: int):
        """Setup undistortion maps if calibration profile is available"""
        if self.profile is None:
            return
        dist_coeffs = np.array(self.profile['dist_coeffs'], dtype=np.float64)

        self.K_undistorted, _ = cv2.getOptimalNewCameraMatrix(
            self.K, dist_coeffs, (actual_width, actual_height), 1, (actual_width, actual_height)
        )

        self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
            self.K, dist_coeffs, None, self.K_undistorted, (actual_width, actual_height), cv2.CV_16SC2
        )
        print(f"USB Camera: Undistortion maps initialized")

    def initialize(self, width: int, height: int, framerate: int) -> tuple:
        """Initialize USB camera in background thread"""
        def _init():
            # CAP_V4L2 must be explicit: with autodetect, OpenCV picks FFMPEG for a
            # /dev/video* path, and the FFMPEG backend silently ignores FOURCC/WIDTH/
            # HEIGHT, opening at its own default (640x480) with an empty fourcc.
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"Failed to open USB camera: {self.device}")
                self.init_complete.set()
                return

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, framerate)
            # The V4L2 default of 4 buffered frames is 133 ms of latency nobody can see
            # from the outside: read() hands back the OLDEST queued frame, so after any
            # hiccup the stream stays a third of a second behind forever — a consumer
            # running at exactly the capture rate never catches up. Depth 1 drops a
            # frame instead of accumulating lag. Measured: 4 stale frames -> 1.
            # (CAP_PROP_BUFFERSIZE still reads back as 4 afterwards; the behaviour does
            # change, so trust the probe, not the getter.)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
            if not self.autofocus and self.focus is not None:
                self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)

            aw, ah = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_framerate = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"USB Camera: Requested {width}x{height}@{framerate}fps, Got {aw}x{ah}@{actual_framerate}fps")

            if self.passthrough and not self._enable_passthrough(aw, ah):
                # Verification failed — the camera's raw buffers are not usable JPEG.
                # Reopen in the ordinary decoded mode rather than publish garbage.
                print(f"USB Camera {self.device}: pass-through unavailable, decoding instead")
                self.passthrough = False
                self.cap.release()
                self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, framerate)

            self.undistort_temp_buf = np.empty((ah, aw, 3), dtype=np.uint8)
            self.gray_buf_internal = np.empty((ah, aw), dtype=np.uint8)
            self.frame_buf_internal = np.empty((ah, aw, 3), dtype=np.uint8)
            self.K_undistorted = None
            self.frame_sleep_time = 1.0 / (actual_framerate * 4)
            self.K = self._get_intrinsics_internal(aw, ah)
            self._setup_undistortion(aw, ah)
            self.init_complete.set()

        threading.Thread(target=_init, daemon=True).start()
        return (width, height)

    def get_intrinsics(self):
        self.init_complete.wait(timeout=4)  # Wait for initialization to complete
        return self.K_undistorted if self.K_undistorted is not None else self.K
    
    def _get_intrinsics_internal(self, width: int, height: int) -> np.ndarray:
        """Get or estimate camera intrinsics"""
        if self.profile is not None:
            fx, fy, cx, cy = self.profile["cam_params"]
            if "image_size" in self.profile:
                cw, ch = self.profile["image_size"]
                if width != cw or height != ch:
                    sx, sy = width / cw, height / ch
                    fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
                    if abs((width/height) - (cw/ch)) / (cw/ch) > 0.01:
                        print("\033[93mWARNING: Aspect ratio changed - may indicate cropping!\033[0m")
                    print(f"USB Camera: Scaled intrinsics {cw}x{ch} → {width}x{height} (scale: {sx:.3f}, {sy:.3f})")
            print(f"USB Camera: Using intrinsics fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        else:
            fx = fy = width * 1.2
            cx, cy = width / 2.0, height / 2.0
            print(f"USB Camera: Using estimated intrinsics fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print("\033[93mWARNING: Using estimated camera intrinsics. Calibrate your camera!\033[0m")

        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _enable_passthrough(self, width: int, height: int) -> bool:
        """Switch the capture to raw mode and prove the buffers really are JPEG.

        CAP_PROP_CONVERT_RGB=0 stops OpenCV decoding, but what arrives then is whatever
        the driver's buffer holds. The first buffer after the switch is routinely a
        partial frame, so several are checked and one must decode at the expected size.
        """
        if not self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
            return False
        for _ in range(12):
            ok, raw = self.cap.read()
            if not ok:
                continue
            payload = self._trim_jpeg(raw)
            if payload is None:
                continue
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is not None and img.shape[:2] == (height, width):
                print(f"USB Camera {self.device}: pass-through enabled "
                      f"({len(payload)/1024:.0f} KB/frame from the camera, no transcode)")
                return True
        return False

    @staticmethod
    def _trim_jpeg(raw: np.ndarray):
        """Extract SOI..EOI from a V4L2 buffer, or None if it holds no complete JPEG.

        The driver returns a fixed-capacity buffer with the frame at the front and
        trailing padding, so the end has to be found rather than assumed.
        """
        b = raw.tobytes() if raw.flags['C_CONTIGUOUS'] else np.ascontiguousarray(raw).tobytes()
        soi = b.find(b'\xff\xd8\xff')
        if soi < 0:
            return None
        eoi = b.rfind(b'\xff\xd9')
        return b[soi:eoi + 2] if eoi > soi else None

    def get_encoded_frame(self):
        """Newest camera JPEG in pass-through mode, else None."""
        return self._encoded_frame

    def capture_frame(self, undistort: bool = True) -> bool:
        """Capture frame from USB camera, update frame buffer and gray buffer"""
        if not self.init_complete.is_set():
            return False
        if self.cap is None or not self.cap.isOpened():
            return False

        if self.passthrough:
            # No decode, no colour conversion — the camera's own JPEG goes straight out.
            ok, raw = self.cap.read()
            if not ok:
                return False
            payload = self._trim_jpeg(raw)
            if payload is None:
                return False   # torn buffer; the next frame is usually fine
            self._encoded_frame = payload
            self.frame_ready.set()
            return True

        if undistort and self.undistort_map1 is not None:
            if not self.cap.read(self.undistort_temp_buf):
                return False
            cv2.remap(self.undistort_temp_buf, self.undistort_map1, self.undistort_map2, cv2.INTER_LINEAR, dst=self.frame_buf_internal)
        else:
            if not self.cap.read(self.frame_buf_internal):
                return False
        cv2.cvtColor(self.frame_buf_internal, cv2.COLOR_BGR2GRAY, dst=self.gray_buf_internal)

        # Signal that new frame is available
        self.frame_ready.set()

        return True
    
    def capture_frame_continuously(self):
        """Start continuous frame capture in background thread (non-blocking)"""
        if self.should_capture_frame_continuously:
            return
        def _capture_loop():
            # Wait for initialization to complete
            self.init_complete.wait(timeout=5.0)

            while self.should_capture_frame_continuously:
                if not self.capture_frame():
                    time.sleep(self.frame_sleep_time)
                    
        self.should_capture_frame_continuously = True
        threading.Thread(target=_capture_loop, daemon=True).start()

    def cleanup(self):
        """Release USB camera"""
        self.should_capture_frame_continuously = False
        if self.cap is not None:
            self.cap.release()


    def get_gray_buffer(self) -> np.ndarray:
        """Return internal gray buffer"""
        return self.gray_buf_internal

    def get_frame_buffer(self) -> np.ndarray:
        """Return internal color buffer (BGR)"""
        return self.frame_buf_internal
    
    @staticmethod
    def _build_calibration_data(camera_matrix, dist_coeffs, rms_error, actual_width, actual_height, **extras):
        """Build calibration data dictionary with common fields"""
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.flatten().tolist() if dist_coeffs.ndim > 1 else dist_coeffs.tolist(),
            'cam_params': [float(fx), float(fy), float(cx), float(cy)],
            'rms_error': float(rms_error),
            'image_size': [actual_width, actual_height]
        }
        data.update(extras)
        return data

    @staticmethod
    def show_undistortion_preview(camera_device, width, height, camera_matrix, dist_coeffs,
                                    actual_width, actual_height, rms_error, framerate=30):
        """Show live undistorted preview after calibration

        Args:
            camera_device: Device path or index
            width: Requested camera width
            height: Requested camera height
            camera_matrix: Calibrated camera matrix
            dist_coeffs: Calibrated distortion coefficients
            actual_width: Actual camera width
            actual_height: Actual camera height
            rms_error: RMS reprojection error
            framerate: Camera framerate (default: 30)
        """
        print("\n" + "=" * 60)
        print("LIVE UNDISTORTION PREVIEW\n" + "=" * 60)
        print("Showing calibrated (undistorted) camera view\nPress 'q' to finish\n" + "=" * 60)

        calib_profile = {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs,
                        'cam_params': [camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]],
                        'image_size': [actual_width, actual_height]}

        backend = USBCameraBackend(device=camera_device, profile=calib_profile)
        backend.initialize(width, height, framerate)
        viz = create_visualizer()
        fps = FPS()
        try:
            while True:
                if not backend.capture_frame(undistort=True):
                    continue
                frame = backend.get_frame_buffer()
                fps.tick()
                viz.add_text(frame, "CALIBRATED VIEW (Undistorted)", (10, 30), 0.8, (0, 255, 0), 2)
                viz.add_text(frame, f"RMS Error: {rms_error:.4f} px", (10, 60), 0.6, (255, 255, 0), 2)
                viz.add_text(frame, f"FPS: {fps.fps:.1f}", (10, 90), 0.6, (255, 255, 255), 2)
                viz.add_text(frame, "Press 'q' to finish", (10, 120), 0.6, (255, 255, 255), 1)
                viz.show("Calibration Result", frame)
                if viz.wait_key(1) == ord('q'):
                    break
        finally:
            backend.cleanup()
            viz.cleanup("Calibration Result")

    @staticmethod
    def calibrate_camera(camera_device, width=1280, height=720, chessboard_size=(9, 6),
                        square_size=0.025, num_images=20, visualize=True, framerate=30):
        """Calibrate USB camera using chessboard pattern

        Args:
            camera_device: Device path (e.g., '/dev/video0') or index (e.g., 0)
            width: Camera resolution width
            height: Camera resolution height
            chessboard_size: Inner corners (cols, rows) - default 9x6
            square_size: Physical size of chessboard square in meters (default 25mm)
            num_images: Minimum number of calibration images to capture
            visualize: Show live preview with corner detection
            framerate: Camera framerate (default: 30)

        Returns:
            dict: Calibration results containing:
                - camera_matrix: 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
                - dist_coeffs: Distortion coefficients [k1,k2,p1,p2,k3]
                - cam_params: [fx, fy, cx, cy] (compatible with tag_detection)
                - rms_error: RMS reprojection error
                - image_size: [width, height]
                - chessboard_size: [cols, rows]
                - square_size: Physical square size in meters
        """
        print("=" * 60)
        print("USB CAMERA CALIBRATION")
        print("=" * 60)
        print(f"Camera: {camera_device}")
        print(f"Resolution: {width}x{height}")
        print(f"Chessboard: {chessboard_size[0]}x{chessboard_size[1]} inner corners")
        print(f"Square size: {square_size*1000:.1f}mm")
        print(f"Target images: {num_images}")
        print("=" * 60)

        # Initialize camera using class (reuses existing initialization logic)
        backend = USBCameraBackend(device=camera_device)
        actual_width, actual_height = backend.initialize(width, height, framerate)

        # Prepare object points (3D points in real world space)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        captured_count = 0

        print("\nInstructions:")
        print("  - Move chessboard to different positions and angles")
        print("  - Press SPACE to capture when corners are detected")
        print("  - Press 'q' to quit early (min 10 images required)")
        print("  - Ensure good lighting and focus")
        print("-" * 60)

        viz = None
        if visualize:
            viz = create_visualizer()

        try:
            while captured_count < num_images:
                # Use class method for frame capture
                if not backend.capture_frame(undistort=False):
                    continue

                frame = backend.get_frame_buffer()
                gray = backend.get_gray_buffer()

                # Use SB (Structured Board) method - most robust and fastest
                ret_corners, corners = cv2.findChessboardCornersSB(
                    gray, chessboard_size, 0
                )

                # Draw visualization
                display = frame.copy()
                if ret_corners:
                    # Refine corner locations
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    # Draw corners
                    cv2.drawChessboardCorners(display, chessboard_size, corners_refined, ret_corners)

                    # Add status text
                    status_text = "DETECTED - Press SPACE"
                    color = (0, 255, 0)  # Green
                    viz.add_text(display, status_text, (10, 30), 0.7, color, 2) if viz else None
                else:
                    viz.add_text(display, "NOT DETECTED", (10, 30), 0.7, (0, 0, 255), 2) if viz else None
                    # Add troubleshooting hints
                    viz.add_text(display, f"Looking for {chessboard_size[0]}x{chessboard_size[1]} inner corners",
                               (10, 110), 0.5, (255, 255, 255), 1) if viz else None
                    viz.add_text(display, "Wrong size? Try swapping: --cols {} --rows {}".format(chessboard_size[1], chessboard_size[0]),
                               (10, 135), 0.5, (255, 200, 100), 1) if viz else None
                    viz.add_text(display, "Press 'd' to save debug image",
                               (10, 160), 0.5, (255, 255, 0), 1) if viz else None

                viz.add_text(display, f"Captured: {captured_count}/{num_images}",
                           (10, 70), 0.7, (255, 255, 0), 2) if viz else None

                if visualize:
                    viz.show("Camera Calibration", display)

                key = viz.wait_key(1) if viz else -1

                if key == ord(' '):
                    if ret_corners:
                        # Capture with detected corners
                        objpoints.append(objp)
                        imgpoints.append(corners_refined)
                        captured_count += 1
                        print(f"  [{captured_count}/{num_images}] Image captured!")
                    else:
                        print("  Cannot capture - pattern not detected.")
                        continue

                    # Flash effect
                    if visualize:
                        flash = np.ones_like(frame) * 255
                        viz.show("Camera Calibration", flash)
                        viz.wait_key(100)

                elif key == ord('d'):
                    # Save debug image
                    import time as time_module
                    timestamp = int(time_module.time())
                    debug_filename = f"debug_chessboard_{timestamp}.png"
                    cv2.imwrite(debug_filename, frame)
                    print(f"  Debug image saved: {debug_filename}")
                    print(f"    Pattern size: {chessboard_size[0]}x{chessboard_size[1]} inner corners")
                    print(f"    Image shape: {frame.shape}")
                    print(f"    Mean brightness: {gray.mean():.1f}")

                elif key == ord('q'):
                    if captured_count >= 10:
                        print(f"\nEarly exit with {captured_count} images")
                        break
                    else:
                        print(f"\nNeed at least 10 images (currently {captured_count})")
        finally:
            # Ensure cleanup happens even if user interrupts
            backend.cleanup()
            if viz:
                viz.cleanup("Camera Calibration")

        print("-" * 60)
        print(f"Calibrating with {len(objpoints)} images...")

        # Calibrate camera (retval is the RMS reprojection error)
        rms_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (actual_width, actual_height), None, None
        )

        # Extract camera parameters
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        print("=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)
        print(f"RMS Reprojection Error: {rms_error:.4f} pixels")
        print(f"Camera Matrix (K):")
        print(f"  fx = {fx:.2f}  fy = {fy:.2f}")
        print(f"  cx = {cx:.2f}  cy = {cy:.2f}")
        print(f"Distortion Coefficients:")
        print(f"  k1={dist_coeffs[0][0]:.6f}  k2={dist_coeffs[0][1]:.6f}")
        print(f"  p1={dist_coeffs[0][2]:.6f}  p2={dist_coeffs[0][3]:.6f}")
        print(f"  k3={dist_coeffs[0][4]:.6f}")
        print("=" * 60)

        # Package results
        calibration_data = USBCameraBackend._build_calibration_data(
            camera_matrix, dist_coeffs, rms_error, actual_width, actual_height,
            chessboard_size=list(chessboard_size),
            square_size=float(square_size),
            calibration_method='chessboard'
        )

        # Show live undistorted preview
        if visualize:
            USBCameraBackend.show_undistortion_preview(
                camera_device, width, height, camera_matrix, dist_coeffs,
                actual_width, actual_height, rms_error
            )

        return calibration_data

    @staticmethod
    def calibrate_camera_charuco(camera_device, width=1280, height=720, squares_x=9, squares_y=6,
                                 square_size=0.025, marker_size=None, aruco_dict=cv2.aruco.DICT_4X4_100,
                                 num_images=20, visualize=True, framerate=30):
        """Calibrate USB camera using ChArUco board pattern

        ChArUco (Chessboard + ArUco) boards provide superior calibration accuracy:
        - Partial occlusions handled gracefully
        - More robust corner detection
        - Each corner uniquely identified
        - Better for complex camera distortions

        Args:
            camera_device: Device path (e.g., '/dev/video0') or index (e.g., 0)
            width: Camera resolution width
            height: Camera resolution height
            squares_x: Number of squares horizontally (e.g., 9)
            squares_y: Number of squares vertically (e.g., 6)
            square_size: Physical size of chessboard square in meters (default 25mm)
            marker_size: Physical size of ArUco marker in meters (default 0.75 * square_size)
            aruco_dict: ArUco dictionary to use (default: DICT_4X4_50)
            num_images: Minimum number of calibration images to capture
            visualize: Show live preview with corner detection
            framerate: Camera framerate (default: 30)

        Returns:
            dict: Calibration results (same format as calibrate_camera)
        """
        # Default marker size is 75% of square size
        if marker_size is None:
            marker_size = square_size * 0.75

        print("=" * 60)
        print("USB CAMERA CALIBRATION (ChArUco)")
        print("=" * 60)
        print(f"Camera: {camera_device}")
        print(f"Resolution: {width}x{height}")
        print(f"Board: {squares_x}x{squares_y} squares")
        print(f"Square size: {square_size*1000:.1f}mm")
        print(f"Marker size: {marker_size*1000:.1f}mm")
        print(f"Target images: {num_images}")
        print("=" * 60)

        # Create ChArUco board
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, aruco_dictionary)
        charuco_detector = cv2.aruco.CharucoDetector(board)

        # Print detection parameters
        dict_names = {
            cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
            cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
            cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
            cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
        }
        dict_name = dict_names.get(aruco_dict, f"Custom({aruco_dict})")
        print(f"ArUco dictionary: {dict_name}")
        print(f"Expected corners: {(squares_x-1) * (squares_y-1)} ({squares_x-1}x{squares_y-1})")

        # Initialize camera using class (reuses existing initialization logic)
        backend = USBCameraBackend(device=camera_device)
        actual_width, actual_height = backend.initialize(width, height, framerate)

        # Arrays to store detected corners
        all_charuco_corners = []
        all_charuco_ids = []

        captured_count = 0

        print("\nInstructions:")
        print("  - Move ChArUco board to different positions and angles")
        print("  - Press SPACE to capture when board is detected")
        print("  - Press 'q' to quit early (min 10 images required)")
        print("  - Ensure good lighting and focus")
        print("-" * 60)

        viz = None
        if visualize:
            viz = create_visualizer()

        try:
            while captured_count < num_images:
                # Use class method for frame capture
                if not backend.capture_frame(undistort=False):
                    continue

                frame = backend.get_frame_buffer()
                gray = backend.get_gray_buffer()

                # Detect ChArUco board
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

                # Draw visualization
                display = frame.copy()
                detected = charuco_corners is not None and len(charuco_corners) > 3

                if detected:
                    # Draw detected corners and markers
                    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
                    if marker_corners is not None:
                        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

                    # Add status text
                    status_text = f"DETECTED ({len(charuco_corners)} corners) - Press SPACE"
                    color = (0, 255, 0)  # Green
                    viz.add_text(display, status_text, (10, 30), 0.7, color, 2) if viz else None
                else:
                    viz.add_text(display, "NOT DETECTED", (10, 30), 0.7, (0, 0, 255), 2) if viz else None
                    # Add troubleshooting hints
                    viz.add_text(display, f"Looking for {squares_x}x{squares_y} ChArUco board",
                               (10, 110), 0.5, (255, 255, 255), 1) if viz else None

                    # Show marker detection status
                    if marker_ids is not None and len(marker_ids) > 0:
                        viz.add_text(display, f"Markers: {len(marker_ids)} detected (corners: {len(charuco_corners) if charuco_corners is not None else 0})",
                                   (10, 135), 0.5, (255, 255, 0), 1) if viz else None
                    else:
                        viz.add_text(display, "Markers: 0 detected - check lighting/distance",
                                   (10, 135), 0.5, (255, 200, 100), 1) if viz else None

                viz.add_text(display, f"Captured: {captured_count}/{num_images}",
                           (10, 70), 0.7, (255, 255, 0), 2) if viz else None

                if visualize:
                    viz.show("ChArUco Calibration", display)

                key = viz.wait_key(1) if viz else -1

                if key == ord(' '):
                    if detected:
                        # Capture corners
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        captured_count += 1
                        print(f"  [{captured_count}/{num_images}] Image captured ({len(charuco_corners)} corners)!")
                    else:
                        print("  Cannot capture - board not detected.")
                        continue

                    # Flash effect
                    if visualize:
                        flash = np.ones_like(frame) * 255
                        viz.show("ChArUco Calibration", flash)
                        viz.wait_key(100)

                elif key == ord('q'):
                    if captured_count >= 10:
                        print(f"\nEarly exit with {captured_count} images")
                        break
                    else:
                        print(f"\nNeed at least 10 images (currently {captured_count})")
        finally:
            # Ensure cleanup happens even if user interrupts
            backend.cleanup()
            if viz:
                viz.cleanup("ChArUco Calibration")

        print("-" * 60)
        print(f"Calibrating with {len(all_charuco_corners)} images...")

        # Prepare object points and image points for calibration
        # Get the 3D coordinates of chessboard corners from the board
        board_corners_3d = board.getChessboardCorners()

        objpoints = []
        imgpoints = []

        for i in range(len(all_charuco_corners)):
            # Get the 3D object points for this image's detected corners
            corner_ids = all_charuco_ids[i].flatten()
            obj_pts = board_corners_3d[corner_ids]

            objpoints.append(obj_pts)
            imgpoints.append(all_charuco_corners[i])

        # Calibrate camera using standard calibrateCamera
        rms_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (actual_width, actual_height), None, None
        )

        if camera_matrix is None:
            raise RuntimeError("ChArUco calibration failed!")

        print(f"Calibration complete!")
        print(f"  RMS reprojection error: {rms_error:.4f} pixels")
        print(f"  Camera matrix (fx, fy, cx, cy): [{camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f}, {camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f}]")

        calibration_data = USBCameraBackend._build_calibration_data(
            camera_matrix, dist_coeffs, rms_error, actual_width, actual_height,
            board_size=[squares_x, squares_y],
            square_size=float(square_size),
            marker_size=float(marker_size),
            calibration_method='charuco'
        )

        # Show live undistorted preview
        if visualize:
            USBCameraBackend.show_undistortion_preview(
                camera_device, width, height, camera_matrix, dist_coeffs,
                actual_width, actual_height, rms_error
            )

        return calibration_data

    @staticmethod
    def save_calibration(calibration_data, filepath):
        """Save calibration data to YAML file

        Args:
            calibration_data: Calibration dictionary from calibrate_camera()
            filepath: Path to output YAML file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False)

        print(f"Calibration saved to: {filepath}")

    @staticmethod
    def load_calibration(filepath, verbose=True):
        """Load calibration data from YAML file

        Args:
            filepath: Path to YAML calibration file

        Returns:
            dict: Calibration data containing camera_matrix, dist_coeffs, cam_params, etc.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        with open(filepath, 'r') as f:
            calibration_data = yaml.safe_load(f)
        if verbose:
            print(f"Calibration loaded from: {filepath}")
            print(f"  Image size: {calibration_data['image_size']}")
            print(f"  cam_params: {calibration_data['cam_params']}")
            print(f"  RMS error: {calibration_data['rms_error']:.4f} pixels")
        return calibration_data

    @staticmethod
    def find_camera_by_name(camera_name, verbose=False):
        """Find camera device by name (e.g., 'OBSBOT Meet 2' -> '/dev/video18')

        Args:
            camera_name: Name or partial name of the camera to find
            verbose: If True, show list_cameras() output when camera not found

        Returns:
            Device path (e.g., '/dev/video18') or None if not found
        """
        try:
            if verbose:
                cameras = USBCameraBackend.list_cameras()
            else:
                cameras = USBCameraBackend._get_camera_list()

            # Search in both v4l2 and udev names
            for camera in cameras:
                if camera_name.lower() in camera['name'].lower() or \
                   (camera.get('udev_name') and camera_name.lower() in camera['udev_name'].lower()):
                    if camera['capture_devices']:
                        return camera['capture_devices'][0]

            # Camera not found
            if verbose:
                print(f"Camera '{camera_name}' not found.")
            return None

        except Exception as e:
            if verbose:
                print(f"Error finding camera: {e}")
            return None

    @staticmethod
    def _get_camera_list():
        """Get list of cameras without printing"""
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                            capture_output=True, text=True, check=True)
        cameras = []
        current_camera, current_devices = None, []

        # Parse v4l2-ctl output
        for line in result.stdout.strip().split('\n'):
            if line and not line[0].isspace():
                if current_camera:
                    cameras.append({'name': current_camera, 'devices': current_devices.copy(), 'capture_devices': []})
                current_camera, current_devices = line.strip(), []
            elif line.strip().startswith('/dev/video'):
                current_devices.append(line.strip().split(':')[0])

        if current_camera:
            cameras.append({'name': current_camera, 'devices': current_devices.copy(), 'capture_devices': []})

        for camera in cameras:
            camera['udev_name'] = None
            if camera['devices']:
                try:
                    r = subprocess.run(['udevadm', 'info', '--query=property', f'--name={camera["devices"][0]}'], capture_output=True, text=True, timeout=1)
                    for line in r.stdout.split('\n'):
                        if line.startswith('ID_MODEL='):
                            camera['udev_name'] = line.split('=', 1)[1]
                            break
                except:
                    pass
            for device in camera['devices']:
                try:
                    if 'Video Capture' in subprocess.run(['v4l2-ctl', f'--device={device}', '--list-formats-ext'], capture_output=True, text=True, timeout=1).stdout:
                        camera['capture_devices'].append(device)
                except:
                    pass
        return cameras

    @staticmethod
    def list_cameras():
        """List all available cameras with their video devices"""
        try:
            cameras = USBCameraBackend._get_camera_list()

            # Print results
            print("Available cameras:")
            print("=" * 60)
            for camera in cameras:
                camera['v4l2_name'] = camera['name']
                name_line = f"{camera['name']} | {camera['udev_name']}" if camera['udev_name'] else camera['name']
                print(f"\n{name_line}")
                print(f"  All devices: {', '.join(camera['devices'])}")
                print(f"  Capture devices: {', '.join(camera['capture_devices']) if camera['capture_devices'] else 'None'}")
            print("=" * 60)

            return cameras

        except FileNotFoundError:
            print("Error: v4l2-ctl not found. Please install v4l-utils package.")
            return []
        except Exception as e:
            print(f"Unexpected error listing cameras: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    USBCameraBackend.list_cameras()
    camera_name = 'OBSBOT Meet 2'
    camera_device = USBCameraBackend.find_camera_by_name(camera_name, verbose=True)
    print(f"Found camera device: {camera_device}")

    # Visualize the camera device (auto-detects headless environment)
    if camera_device:
        backend = USBCameraBackend(device=camera_device, autofocus=False, focus=20)
        backend.initialize(1280, 720, 30)
        viz = create_visualizer()
        try:
            while True:
                if backend.capture_frame():
                    viz.show("Camera", backend.get_frame_buffer())
                if viz.wait_key(1) == ord('q'):
                    break
        finally:
            backend.cleanup()
            viz.cleanup()
