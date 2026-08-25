"""Camera visualization abstraction - automatically handles headless environments

Provides unified interface for displaying camera frames using either:
- OpenCV GUI (when X11 display available)
- Viser web GUI (headless/remote access)

Zero overhead when GUI is not used. Minimal overhead when streaming.
"""

import math
import os
import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict
import threading


class BaseCameraVisualizer(ABC):
    """Abstract base for camera visualization backends"""

    @abstractmethod
    def show(self, name: str, frame: np.ndarray):
        """Display a frame in a named window

        Args:
            name: Window name
            frame: BGR or grayscale image
        """
        pass

    @abstractmethod
    def wait_key(self, delay_ms: int = 1) -> int:
        """Wait for keyboard input

        Args:
            delay_ms: Delay in milliseconds

        Returns:
            ASCII code of pressed key, or -1 if no key pressed
        """
        pass

    @abstractmethod
    def cleanup(self, window_name: Optional[str] = None):
        """Clean up resources

        Args:
            window_name: Specific window to destroy, or None for all
        """
        pass

    @abstractmethod
    def add_text(self, frame: np.ndarray, text: str, pos: tuple,
                 font_scale: float = 0.7, color: tuple = (255, 255, 255),
                 thickness: int = 2) -> np.ndarray:
        """Add text overlay to frame (in-place or copy depending on backend)

        Args:
            frame: Input image
            text: Text to display
            pos: (x, y) position
            font_scale: Font size
            color: BGR color tuple
            thickness: Line thickness

        Returns:
            Frame with text added (may be same object or copy)
        """
        pass


class OpenCVVisualizer(BaseCameraVisualizer):
    """OpenCV-based visualization (requires X11 display)"""

    def __init__(self):
        self.windows_created = set()

    def show(self, name: str, frame: np.ndarray):
        """Show frame using cv2.imshow (zero overhead)"""
        if name not in self.windows_created:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            self.windows_created.add(name)
        cv2.imshow(name, frame)

    def wait_key(self, delay_ms: int = 1) -> int:
        """Direct passthrough to cv2.waitKey"""
        return cv2.waitKey(delay_ms) & 0xFF

    def cleanup(self, window_name: Optional[str] = None):
        """Destroy OpenCV windows"""
        if window_name:
            cv2.destroyWindow(window_name)
            self.windows_created.discard(window_name)
        else:
            cv2.destroyAllWindows()
            self.windows_created.clear()

    def add_text(self, frame: np.ndarray, text: str, pos: tuple,
                 font_scale: float = 0.7, color: tuple = (255, 255, 255),
                 thickness: int = 2) -> np.ndarray:
        """Add text using cv2.putText (in-place)"""
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness)
        return frame


class ViserVisualizer(BaseCameraVisualizer):
    """Viser web-based visualization (headless-compatible)

    Features:
    - Lazy initialization (server only starts on first show())
    - Optional frame throttling (disabled by default - no frame skipping)
    - JPEG compression for efficient streaming
    - Reusable image handles (no per-frame allocation)
    - GUI buttons for keyboard simulation
    """

    FOV = 0.5
    """Vertical field of view (radians) used for the default camera framing."""

    VIEWPORT_ASPECT = 1.5
    """Assumed browser viewport aspect, used to pick a distance that fills it.

    Deliberately squarer than 16:9: a browser window is usually narrower than its
    monitor once side panels are subtracted, and guessing too wide crops the image
    horizontally — which is worse than a little unused space above and below.
    """

    def __init__(self, port: int = 8080, max_fps: Optional[float] = None, server=None,
                 jpeg_quality: int = 80, threaded_encode: bool = True):
        """Initialize Viser visualizer.

        Args:
            port: Web server port (ignored if server is provided)
            max_fps: Maximum frame update rate (None = no throttling, display all frames)
                    Set to a value (e.g., 15.0) to reduce overhead at cost of frame drops
            server: Existing viser.ViserServer to reuse (avoids starting a second server)
            jpeg_quality: Quality of the composite JPEG sent to the browser.
            threaded_encode: Encode the composite on a worker thread so show() returns
                    immediately. cv2.imencode releases the GIL, so it genuinely overlaps
                    with the caller building the next frame.
        """
        self.port = port
        self.max_fps = max_fps
        self.jpeg_quality = jpeg_quality
        self.frame_interval = 1.0 / max_fps if max_fps else 0

        # Depth-1 handoff to the encode worker: newest wins, older composites are
        # dropped. An unbounded queue here would turn a throughput win into a latency
        # disaster, showing the browser frames from seconds ago.
        self.threaded_encode = threaded_encode
        self._enc_pending = None
        self._enc_cv = threading.Condition()
        self._enc_thread = None
        self._enc_stop = False
        # The caller reuses its canvas between frames, so the worker cannot borrow it.
        # Two scratch buffers, alternated, give the worker one to read while the next
        # composite is copied into the other.
        self._enc_scratch = []
        self._enc_slot = 0

        # Set True only by a caller whose frame is ALREADY bottom-up, pixel rows and
        # all, so show() skips its own [::-1]. The flip is needed because
        # scene.add_image maps image row 0 to the bottom of the plane. Note that
        # merely placing tiles bottom-up is not enough — each tile's own rows must be
        # reversed too, or every tile renders vertically mirrored.
        self.flip_in_source = False

        self.image_handles: Dict[str, any] = {}  # window_name -> ImageHandle
        self.image_shapes: Dict[str, tuple] = {}  # window_name -> (h, w)
        self.last_update_time: Dict[str, float] = {}  # window_name -> timestamp

        # Keyboard simulation via GUI buttons
        self.key_pressed = -1
        self.key_lock = threading.Lock()

        # Camera distance that makes the image fill the viewport. Recomputed
        # whenever the image aspect changes (e.g. the camera grid gains a row).
        self._fit_distance = 4.0

        # Use provided server immediately, or defer creation until first show()
        self.server = server
        if self.server is not None:
            self._setup_server()

    def _setup_server(self):
        """Register GUI buttons and camera callbacks on self.server."""
        quit_btn = self.server.gui.add_button("Quit")
        capture_btn = self.server.gui.add_button("Capture")

        @quit_btn.on_click
        def _on_quit(_):
            with self.key_lock:
                self.key_pressed = ord('q')

        @capture_btn.on_click
        def _on_capture(_):
            with self.key_lock:
                self.key_pressed = ord(' ')

    def _ensure_server(self):
        """Lazy server initialization on first use (only when no server was injected)."""
        if self.server is None:
            import viser
            self.server = viser.ViserServer(port=self.port)
            throttle_msg = f" (throttled to {self.max_fps}fps)" if self.max_fps else ""
            print(f"[Viser] Web viewer: http://localhost:{self.server.get_port()}{throttle_msg}")

            @self.server.on_client_connect
            def _on_connect(client):
                # Face the image straight-on: camera at +z, looking toward origin.
                # 180° about X, not Y. Viser uses OpenCV camera axes (look=+Z, up=-Y),
                # so both quaternions aim at the origin, but the Y-rotation also negates
                # right and up — a 180° roll that renders the image upside-down/mirrored.
                client.camera.position = (0.0, 0.0, self._fit_distance)
                client.camera.wxyz = (0.0, 1.0, 0.0, 0.0)
                client.camera.fov = self.FOV

                @client.camera.on_update
                def _on_camera_update(cam):
                    print(f"[Viser] position={np.round(cam.position,3).tolist()} "
                          f"wxyz={np.round(cam.wxyz,3).tolist()} fov={cam.fov:.3f}")

            self._setup_server()

    def show(self, name: str, frame: np.ndarray):
        """Show frame as a 3D scene image, preserving aspect ratio.

        Uses scene.add_image with render dimensions matching the image aspect ratio
        so Viser never stretches the content regardless of viewport size.
        """
        self._ensure_server()

        # Frame throttling (only if max_fps is set)
        if self.max_fps is not None:
            current_time = time.perf_counter()
            if name in self.last_update_time:
                elapsed = current_time - self.last_update_time[name]
                if elapsed < self.frame_interval:
                    return  # Skip this frame
            self.last_update_time[name] = current_time

        # Viser's ImageHandle.image setter fancy-indexes RGB->BGR (a full ~8 MB
        # gather) and then cv2.imencode's the result. We already hold BGR — exactly
        # what the encoder wants — so encoding here and handing Viser the bytes skips
        # both our BGR->RGB conversion and Viser's RGB->BGR undo of it. Measured on a
        # 1440x1080 composite: 10.06 ms via handle.image, 2.62 ms this way.
        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        h, w = frame_bgr.shape[:2]
        shape = (h, w)
        if name not in self.image_handles or self.image_shapes.get(name) != shape:
            # Remove stale handle if shape changed (grid grew/shrank)
            if name in self.image_handles and hasattr(self.image_handles[name], 'remove'):
                self.image_handles[name].remove()
            # render_width/height in world units — ratio encodes aspect ratio.
            # The [::-1] is still needed: scene.add_image maps image row 0 to the
            # BOTTOM of the plane (WebGL texture origin is bottom-left), so without it
            # text and overlays render upside-down. Verified through a headless
            # browser via client.get_render().
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.image_handles[name] = self.server.scene.add_image(
                name, rgb if self.flip_in_source else rgb[::-1],
                render_width=w / h, render_height=1.0,
                format='jpeg', jpeg_quality=self.jpeg_quality,
            )
            self.image_shapes[name] = shape

            # Pull the camera in so the image fills the viewport rather than
            # floating in the middle of it. The image spans render_height=1.0
            # world units vertically and w/h horizontally; a camera with vertical
            # FOV sees 2*d*tan(FOV/2) of height at the image plane.
            half_extent = max(1.0, (w / h) / self.VIEWPORT_ASPECT) / 2.0
            self._fit_distance = half_extent / math.tan(self.FOV / 2) * 1.15
            for client in self.server.get_clients().values():
                client.camera.position = (0.0, 0.0, self._fit_distance)
                client.camera.wxyz = (0.0, 1.0, 0.0, 0.0)
                client.camera.fov = self.FOV
        else:
            handle = self.image_handles[name]
            # _data/_format are declared ImageProps fields, so assignment goes through
            # Viser's normal props_setattr -> _queue_update path and does reach the
            # browser. It is underscore-prefixed and outside their stability promise,
            # hence the fallback to the public setter.
            if not hasattr(handle, '_data'):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                handle.image = rgb if self.flip_in_source else rgb[::-1]
            elif self.threaded_encode:
                self._submit_encode(name, frame_bgr)
            else:
                self._encode_into_handle(name, frame_bgr)

    def _encode_into_handle(self, name: str, frame_bgr: np.ndarray):
        """JPEG-encode a BGR composite and push the bytes to its Viser handle."""
        ok, buf = cv2.imencode(
            '.jpg', frame_bgr if self.flip_in_source else frame_bgr[::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            self.image_handles[name]._data = buf.tobytes()
            # Viser windows outgoing messages at 1/60 s and only re-checks on its flush
            # event, so a frame pushed just after a window closes waits the full 16.7 ms.
            self.server.flush()

    def _submit_encode(self, name: str, frame_bgr: np.ndarray):
        """Hand the composite to the encode worker, replacing any frame still queued."""
        if self._enc_thread is None:
            self._enc_thread = threading.Thread(target=self._encode_worker, daemon=True)
            self._enc_thread.start()

        # Copy into a scratch buffer the caller does not own, alternating slots so the
        # worker is never reading the buffer we are writing.
        slot = self._enc_slot
        if len(self._enc_scratch) < 2:
            self._enc_scratch.append(np.empty_like(frame_bgr))
            slot = len(self._enc_scratch) - 1
        elif self._enc_scratch[slot].shape != frame_bgr.shape:
            self._enc_scratch = [np.empty_like(frame_bgr), np.empty_like(frame_bgr)]
            slot = 0
        np.copyto(self._enc_scratch[slot], frame_bgr)
        self._enc_slot = 1 - slot if len(self._enc_scratch) == 2 else 0

        with self._enc_cv:
            self._enc_pending = (name, self._enc_scratch[slot])
            self._enc_cv.notify()

    def _encode_worker(self):
        while True:
            with self._enc_cv:
                while self._enc_pending is None and not self._enc_stop:
                    self._enc_cv.wait()
                if self._enc_stop:
                    return
                name, arr = self._enc_pending
                self._enc_pending = None   # anything newer replaces this before we wake
            try:
                self._encode_into_handle(name, arr)
            except Exception as e:
                print(f"[Viser] composite encode failed: {e}")

    def wait_key(self, delay_ms: int = 1) -> int:
        """Simulate cv2.waitKey() by checking GUI button state"""
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # Check if any button was pressed
        with self.key_lock:
            key = self.key_pressed
            self.key_pressed = -1  # Consume the key
            return key

    def cleanup(self, window_name: Optional[str] = None):
        """Clean up Viser resources"""
        with self._enc_cv:
            self._enc_stop = True
            self._enc_cv.notify_all()
        if self._enc_thread is not None:
            self._enc_thread.join(timeout=2)
            self._enc_thread = None
        for handle in self.image_handles.values():
            if hasattr(handle, 'remove'):
                handle.remove()
        self.image_handles.clear()
        self.image_shapes.clear()
        self.last_update_time.clear()

    def add_text(self, frame: np.ndarray, text: str, pos: tuple,
                 font_scale: float = 0.7, color: tuple = (255, 255, 255),
                 thickness: int = 2) -> np.ndarray:
        """Add text using cv2.putText (in-place, before JPEG encoding)"""
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness)
        return frame


class CameraVisualizer:
    """Factory for creating appropriate visualizer based on environment"""

    @staticmethod
    def is_headless() -> bool:
        """Check if running in headless environment (no X11 display)"""
        display = os.environ.get('DISPLAY', '')
        if not display:
            return True

        # Try importing display-dependent libraries
        try:
            # Quick check - if DISPLAY is set, assume it works
            # More thorough check would be to actually test cv2.namedWindow,
            # but that's slower and may have side effects
            return False
        except:
            return True

    @staticmethod
    def create(headless: Optional[bool] = None, **kwargs) -> BaseCameraVisualizer:
        """Create appropriate visualizer for current environment.

        Args:
            headless: Force headless mode (None = auto-detect)
            **kwargs: Forwarded to ViserVisualizer (port, max_fps, server).
                      'server' allows reusing an existing viser.ViserServer.

        Returns:
            OpenCVVisualizer or ViserVisualizer
        """
        if headless is None:
            headless = CameraVisualizer.is_headless()

        if headless:
            print("[CameraVisualizer] Headless environment detected - using Viser web GUI")
            return ViserVisualizer(**kwargs)
        else:
            print("[CameraVisualizer] Display available - using OpenCV GUI")
            return OpenCVVisualizer()


# Convenience function for simple use cases
def create_visualizer(headless: Optional[bool] = None, **kwargs) -> BaseCameraVisualizer:
    """Convenience function - see CameraVisualizer.create()"""
    return CameraVisualizer.create(headless=headless, **kwargs)


if __name__ == "__main__":
    # Test visualization
    print("Testing camera visualizer...")

    viz = create_visualizer()

    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        for i in range(100):
            # Update frame
            frame[:] = (i * 2) % 256
            viz.add_text(frame, f"Frame {i}", (10, 30), color=(255, 255, 0))
            viz.add_text(frame, "Press 'q' to quit", (10, 60),
                        font_scale=0.5, color=(255, 255, 255), thickness=1)

            # Show frame
            viz.show("Test Window", frame)

            # Check for quit
            key = viz.wait_key(30)
            if key == ord('q'):
                print("Quit requested")
                break
    finally:
        viz.cleanup()
        print("Test complete")
