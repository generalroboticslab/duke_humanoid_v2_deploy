import cv2
import numpy as np
from camera_utils import CameraBackend
from camera_visualizer import create_visualizer
import pyrealsense2 as rs
import threading
import time

INHERIT_EXPOSURE = object()
"""Sentinel for color_exposure: use whatever `exposure` is set to."""


class RealSenseCameraBackend(CameraBackend):
    """RealSense camera backend implementation"""

    def __init__(self, device, ir_stream_flag=True, ir_camera_index=1, color_stream_flag=False ,depth_stream_flag=False, exposure=100, color_exposure=INHERIT_EXPOSURE, white_balance=None, compute_gray_in_capture=True):
        """Initialize RealSense camera backend

        Args:
            device: RealSense camera serial number
            ir_stream_flag: Use infrared stream (grayscale) instead of color
            ir_camera_index: Which IR camera to use (1=left, 2=right)
            color_stream_flag: Enable color stream
            depth_stream_flag: Enable depth stream
            exposure: Manual exposure value (µs) to reduce motion blurriness. Set to None to use auto-exposure.
                      Typical indoor color values: 3000–8000 µs. IR default 100 µs is intentionally short.
            color_exposure: Exposure for the *color* sensor, when it needs to differ from the IR one.
                            The IR imager wants a very short exposure for sharp tags, which leaves color
                            nearly black; the same value that exposes color correctly smears IR. Defaults
                            to inheriting `exposure`. None = auto-exposure for color only.
            white_balance: Manual white balance (K), e.g. 4600 for warm/daylight. None = auto white balance.
                           Only applies to color sensor. Greenish tint usually means AWB stuck on wrong preset.
            compute_gray_in_capture: If True, derive gray from color every frame when IR is disabled.
                                    Set False for pure color streaming to save CPU.
        """
        super().__init__(device)

        # RealSense-specific configuration
        self.ir_stream_flag = ir_stream_flag
        self.ir_camera_index = ir_camera_index
        self.color_stream_flag = color_stream_flag
        self.depth_stream_flag = depth_stream_flag
        self.exposure = exposure
        self.color_exposure = exposure if color_exposure is INHERIT_EXPOSURE else color_exposure
        self.white_balance = white_balance
        self.compute_gray_in_capture = compute_gray_in_capture

        # RealSense-specific variables
        self.pipeline = None
        self.profile = None
        self.align = None
        self.frame_sleep_time = 0.01  # overwritten after init with 1/(fps*4)

    @property
    def stream_type(self):
        """Get RealSense stream type based on configuration"""
        return rs.stream.infrared if self.ir_stream_flag else rs.stream.color

    def _device_handle(self):
        """Look this camera up in a fresh context by serial (pipeline not yet started)."""
        for d in rs.context().query_devices():
            if d.get_info(rs.camera_info.serial_number) == str(self.device):
                return d
        return None

    def _usb_type(self):
        d = self._device_handle()
        if d is not None and d.supports(rs.camera_info.usb_type_descriptor):
            return d.get_info(rs.camera_info.usb_type_descriptor)
        return '?'

    def _best_supported_fps(self, width, height, ceiling):
        """Highest framerate <= ceiling that *every* enabled stream offers at width x height.

        Intersecting across streams matters: colour and infrared can advertise different
        rate tables, and pipeline.start() fails unless the single requested rate is in both.
        """
        d = self._device_handle()
        if d is None:
            return None
        wanted = []
        if self.depth_stream_flag:
            wanted.append(rs.stream.depth)
        if self.ir_stream_flag:
            wanted.append(rs.stream.infrared)
        if self.color_stream_flag:
            wanted.append(rs.stream.color)

        common = None
        for stream in wanted:
            rates = set()
            for sensor in d.sensors:
                for p in sensor.get_stream_profiles():
                    v = p.as_video_stream_profile()
                    if (v and p.stream_type() == stream
                            and v.width() == width and v.height() == height
                            and p.fps() <= ceiling):
                        rates.add(p.fps())
            common = rates if common is None else (common & rates)
        return max(common) if common else None

    def initialize(self, width: int, height: int, framerate: int) -> tuple:
        """Initialize RealSense pipeline in background thread"""
        def _init():
            try:
                # A USB 2.1 link cannot carry colour and infrared together. The nasty
                # part is that pipeline.start() *succeeds* — it is frame delivery that
                # never happens, so the camera looks healthy while publishing nothing.
                # Give up infrared rather than the whole camera.
                if (self.ir_stream_flag and self.color_stream_flag
                        and self._usb_type().startswith('2')):
                    print(f"[{self.device}] usb=2.1 link: dropping infrared, "
                          f"colour+IR together never delivers frames on USB 2")
                    self.ir_stream_flag = False

                def build_config(fps):
                    config = rs.config()
                    config.enable_device(self.device)
                    if self.depth_stream_flag:
                        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                    if self.ir_stream_flag:
                        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
                    if self.color_stream_flag:
                        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    return config

                self.pipeline = rs.pipeline()
                actual_fps = framerate
                try:
                    self.profile = self.pipeline.start(build_config(framerate))
                except RuntimeError:
                    # A camera that negotiated a USB 2.1 link advertises a much smaller
                    # mode table — a D435 tops out at 15 fps there — and librealsense
                    # answers the 30 fps request with a flat "Couldn't resolve requests".
                    # Drop to the fastest rate this link actually offers rather than
                    # losing the camera entirely.
                    actual_fps = self._best_supported_fps(width, height, framerate)
                    if actual_fps is None or actual_fps == framerate:
                        raise
                    print(f"[{self.device}] {framerate}fps unavailable "
                          f"(usb={self._usb_type()}), falling back to {actual_fps}fps")
                    self.profile = self.pipeline.start(build_config(actual_fps))
                framerate_actual = actual_fps
                print(f"[{self.device}] RealSense started: {width}x{height}@{framerate_actual}fps")

                # Allocate buffers immediately after pipeline.start() so that any
                # subsequent sensor-option failure leaves them in a valid state.
                intr = self.profile.get_stream(self.stream_type).as_video_stream_profile().get_intrinsics()
                self.gray_buf_internal = np.empty((intr.height, intr.width), dtype=np.uint8)
                if self.color_stream_flag:
                    self.frame_buf_internal = np.empty((intr.height, intr.width, 3), dtype=np.uint8)
                else:
                    self.frame_buf_internal = self.gray_buf_internal
                self.frame_sleep_time = 1.0 / (framerate_actual * 4)

                # sensor set_option() called immediately after start(); warm-up frames cause hangs on D435I fw 5.17.x
                try:
                    if self.ir_stream_flag and self.exposure is not None:
                        ir_sensor = self.profile.get_device().first_depth_sensor()
                        ir_sensor.set_option(rs.option.enable_auto_exposure, 0)
                        ir_sensor.set_option(rs.option.exposure, self.exposure)
                        print(f"[{self.device}] IR exposure set to {self.exposure} µs")
                    if self.color_stream_flag:
                        # Note: D405 emulates color via SW conversion; first_color_sensor() will fail on it.
                        color_sensor = self.profile.get_device().first_color_sensor()
                        color_sensor.set_option(rs.option.enable_auto_exposure, 0 if self.color_exposure is not None else 1)
                        if self.color_exposure is not None:
                            color_sensor.set_option(rs.option.exposure, self.color_exposure)
                        else:
                            # Auto-exposure is allowed to *lower the framerate* to buy light, and in a
                            # dim room it will happily settle on a ~500 ms integration — 2 fps. Priority 0
                            # caps the integration at one frame period, so we keep the requested framerate.
                            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
                        color_sensor.set_option(rs.option.enable_auto_white_balance, 0 if self.white_balance is not None else 1)
                        if self.white_balance is not None:
                            color_sensor.set_option(rs.option.white_balance, self.white_balance)
                        print(f"[{self.device}] Color: exposure={'manual '+str(self.color_exposure)+' µs' if self.color_exposure is not None else 'auto (fps-priority)'}, "
                              f"WB={'manual '+str(self.white_balance)+' K' if self.white_balance is not None else 'auto'}")
                except Exception as e:
                    print(f"[{self.device}] Warning: sensor option config skipped ({e}); "
                          f"D405 color stream is SW-converted from IR — no physical color sensor to configure")

                if self.depth_stream_flag:
                    self.align = rs.align(self.stream_type)
                    depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
                    primary_profile = self.profile.get_stream(self.stream_type).as_video_stream_profile()
                    extr = depth_profile.get_extrinsics_to(primary_profile)
                    print(f"[{self.device}] Depth→{'IR' if self.ir_stream_flag else 'Color'} extrinsics:")
                    print(f"  rotation:    {extr.rotation}")
                    print(f"  translation: {extr.translation}")

            except Exception as e:
                print(f"[{self.device}] RealSense init failed: {e}")
            finally:
                self.init_complete.set()

        threading.Thread(target=_init, daemon=True).start()
        return (width, height)

    def get_intrinsics(self) -> np.ndarray:
        """Extract intrinsics from RealSense profile"""
        self.init_complete.wait(timeout=4)
        intr = self.profile.get_stream(self.stream_type).as_video_stream_profile().get_intrinsics()
        return np.array([[intr.fx, 0.0, intr.ppx],
                        [0.0, intr.fy, intr.ppy],
                        [0.0, 0.0, 1.0]], dtype=np.float64)

    def capture_frame(self, undistort: bool = True) -> bool:
        """Capture frame from RealSense camera

        Args:
            undistort: Ignored for RealSense (has factory calibration)
        """
        if not self.init_complete.is_set():
            return False
        if self.profile is None:  # init completed but failed (e.g. pipeline.start() threw)
            return False

        # The whole read is guarded, not just wait_for_frames: a camera unplugged
        # mid-frame also throws from get_infrared_frame()/get_color_frame(), and a
        # caller polling capture_frame() wants False, not an exception up the stack.
        try:
            frames = self.pipeline.wait_for_frames(2000)  # generous timeout; 30fps = 33ms/frame
            if self.depth_stream_flag:
                frames = self.align.process(frames)
            if self.ir_stream_flag:
                np.copyto(self.gray_buf_internal, frames.get_infrared_frame(self.ir_camera_index).get_data())
            if self.color_stream_flag:
                np.copyto(self.frame_buf_internal, frames.get_color_frame().get_data())
                if not self.ir_stream_flag and self.compute_gray_in_capture:
                    cv2.cvtColor(self.frame_buf_internal, cv2.COLOR_BGR2GRAY, dst=self.gray_buf_internal)
        except Exception:
            return False

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

    def get_gray_buffer(self) -> np.ndarray:
        """Return internal gray buffer"""
        return self.gray_buf_internal

    def get_frame_buffer(self) -> np.ndarray:
        """Return internal color buffer (BGR)"""
        return self.frame_buf_internal

    def cleanup(self):
        """Stop RealSense pipeline"""
        self.should_capture_frame_continuously = False
        if self.pipeline is not None and self.profile is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                # Stopping a pipeline whose device has already been yanked throws.
                # Nothing to recover — the device is gone — but the caller is usually
                # a supervisor tearing down one camera among many, so it must not die.
                print(f"[{self.device}] pipeline.stop() failed on a departed device: {e}")
            self.profile = None
            self.pipeline = None

    @staticmethod
    def list_devices():
        """List all connected RealSense devices.

        Returns:
            list: List of dicts with keys 'name', 'serial', 'firmware'
        """
        devices = [
            {'name': d.get_info(rs.camera_info.name),
             'serial': d.get_info(rs.camera_info.serial_number),
             'firmware': d.get_info(rs.camera_info.firmware_version)}
            for d in rs.context().query_devices()
        ]
        if not devices:
            print("No RealSense devices found.")
        else:
            print("Connected RealSense devices:")
            for i, d in enumerate(devices):
                print(f"[{i}] {d['name']} | Serial: {d['serial']} | Firmware: {d['firmware']}")
        return devices

def make_camera_grid(frames: list) -> np.ndarray:
    """Tile a list of same-shape BGR frames into a compact grid image.

    Layout: ceil(sqrt(N)) rows × ceil(N/rows) cols.
    Examples: 1→1×1, 2→1×2, 3→2×2, 4→2×2, 5→2×3, 6→2×3, 7→3×3 …
    Missing cells in the last row are filled with black.

    Args:
        frames: Non-empty list of BGR (or grayscale) numpy arrays with identical shape.

    Returns:
        Single composite BGR image.
    """
    import math
    n = len(frames)
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3

    # Pad to rows*cols with black placeholders
    black = np.zeros((h, w, 3) if is_color else (h, w), dtype=np.uint8)
    padded = frames + [black] * (rows * cols - n)

    # Build grid row by row
    grid_rows = []
    for r in range(rows):
        row_frames = padded[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_frames))
    return np.vstack(grid_rows)


if __name__ == "__main__":
    # Example usage: view all connected RealSense cameras in a grid.
    # Works in both headless (Viser web GUI) and X11 (OpenCV) environments.
    devices = RealSenseCameraBackend.list_devices()

    if not devices:
        print("No devices to visualize.")
        exit(1)

    # Initialize all available cameras.
    # exposure=None → auto-exposure, white_balance=None → auto-AWB
    # This is the correct baseline for diagnosing per-unit color differences.
    backends = []
    for dev in devices:
        b = RealSenseCameraBackend(dev['serial'], ir_stream_flag=False, color_stream_flag=True, exposure=None)
        b.initialize(1280, 720, 30)
        backends.append(b)

    viz = create_visualizer(headless=True)

    print("Waiting for camera init...")
    for b in backends:
        if not b.init_complete.wait(timeout=15):
            print(f"ERROR: Camera {b.device} init timed out.")
        elif b.profile is None:
            print(f"ERROR: Camera {b.device} init failed. Check device logs above.")
        else:
            print(f"Camera {b.device} ready.")

    active = [b for b in backends if b.profile is not None]
    if not active:
        print("No cameras initialized successfully.")
        for b in backends:
            b.cleanup()
        exit(1)

    n = len(active)
    import math
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    print(f"Displaying {n} camera(s) in a {rows}×{cols} grid.")

    # Each camera captures in its own background thread so all pipelines run
    # in parallel. wait_for_frames() is blocking (~33 ms at 30 fps), so serial
    # polling would cap throughput to 1/(N*33ms). With one thread per camera
    # each pipeline runs at full rate independently.
    for b in active:
        b.capture_frame_continuously()

    try:
        last_frames = [None] * n
        while True:
            for i, b in enumerate(active):
                if b.frame_ready.is_set():
                    last_frames[i] = b.get_frame_buffer().copy()
                    b.frame_ready.clear()
            ready = [f for f in last_frames if f is not None]
            if ready:
                grid = make_camera_grid(ready)
                viz.show("RealSense", grid)
            if viz.wait_key(1) == ord('q'):
                break
    finally:
        for b in backends:
            b.cleanup()
        viz.cleanup()
