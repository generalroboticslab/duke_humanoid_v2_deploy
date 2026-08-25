#!/usr/bin/env python3
import vs_site as _site
import cv2
import numpy as np
import time
import threading
import ctypes
import os
import sys
from typing import List, Dict, Optional, Tuple
import viser

from pupil_apriltags import Detector as _OriginalDetector
from pupil_apriltags.bindings import (
    Detection,
    _ApriltagDetection,
    _ApriltagDetectionInfo,
    _ApriltagPose,
    _ZArray,
    _matd_get_array,
    zarray_get,
)
from camera_utils import USBCameraBackend, CameraBackend
from camera_realsense_utils import RealSenseCameraBackend
from common.rotation_utils import matrix_to_quat, quat_to_matrix

# ============================================================================
# PATCHED DETECTOR - Support for multiple tag sizes
# ============================================================================

class Detector(_OriginalDetector):
    """Patched Detector with per-tag size support via tag_sizes parameter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Profiling stats (public, readable by AprilTagDetector)
        self.time_core_detect = 0.0    # apriltag_detector_detect() call
        self.time_data_extract = 0.0   # Extracting detection data
        self.time_pose_compute = 0.0   # Actual estimate_tag_pose() calls
        self.time_stderr_ops = 0.0     # stderr suppression overhead
        self.time_cleanup = 0.0        # Memory cleanup

    def detect(
        self,
        img: np.ndarray,
        estimate_tag_pose: bool = False,
        camera_params: Optional[Tuple[float, float, float, float]] = None,
        tag_size: Optional[float] = None,
        tag_sizes: Optional[Dict[int, float]] = None,
    ) -> List[Detection]:
        """Detect tags with per-tag size support.

        Args:
            img: Grayscale uint8 image
            estimate_tag_pose: Enable 3D pose estimation
            camera_params: [fx, fy, cx, cy]
            tag_size: Default size in meters
            tag_sizes: {tag_id: size} overrides (optional)
        """
        assert len(img.shape) == 2, "Image must be grayscale"
        assert img.dtype == np.uint8, "Image must be uint8"

        c_img = self._convert_image(img)
        return_info = []

        # PROFILE: Core detection
        t0_core = time.perf_counter()
        self.libc.apriltag_detector_detect.restype = ctypes.POINTER(_ZArray)
        detections = self.libc.apriltag_detector_detect(self.tag_detector_ptr, c_img)
        self.time_core_detect = time.perf_counter() - t0_core
        apriltag = ctypes.POINTER(_ApriltagDetection)()

        # PROFILE: stderr suppression
        t0_stderr = time.perf_counter()
        if estimate_tag_pose and detections.contents.size > 0:
            self.libc.estimate_tag_pose.restype = ctypes.c_double
            stderr_fd = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
        self.time_stderr_ops = time.perf_counter() - t0_stderr

        # Accumulators for per-tag timing
        total_data_extract = 0.0
        total_pose_compute = 0.0

        for i in range(0, detections.contents.size):
            # PROFILE: Data extraction
            t0_extract = time.perf_counter()
            zarray_get(detections, i, ctypes.byref(apriltag))
            tag = apriltag.contents

            # Extract basic detection info
            homography = _matd_get_array(tag.H).copy()
            center = np.ctypeslib.as_array(tag.c, shape=(2,)).copy()
            corners = np.ctypeslib.as_array(tag.p, shape=(4, 2)).copy()

            detection = Detection()
            detection.tag_family = ctypes.string_at(tag.family.contents.name)
            detection.tag_id = tag.id
            detection.hamming = tag.hamming
            detection.decision_margin = tag.decision_margin
            detection.homography = homography
            detection.center = center
            detection.corners = corners
            total_data_extract += time.perf_counter() - t0_extract

            # PROFILE: Pose computation
            if estimate_tag_pose:
                t0_pose = time.perf_counter()
                if camera_params is None:
                    raise Exception("camera_params required for pose estimation")

                # Select size for this tag
                if tag_sizes is not None and tag.id in tag_sizes:
                    current_tag_size = tag_sizes[tag.id]
                elif tag_size is not None:
                    current_tag_size = tag_size
                else:
                    raise Exception(
                        f"No size specified for tag {tag.id}. "
                        f"Provide tag_size or add {tag.id} to tag_sizes."
                    )

                camera_fx, camera_fy, camera_cx, camera_cy = camera_params

                info = _ApriltagDetectionInfo(
                    det=apriltag,
                    tagsize=current_tag_size,  # Per-tag size!
                    fx=camera_fx,
                    fy=camera_fy,
                    cx=camera_cx,
                    cy=camera_cy,
                )
                pose = _ApriltagPose()

                err = self.libc.estimate_tag_pose(ctypes.byref(info), ctypes.byref(pose))

                detection.pose_R = _matd_get_array(pose.R).copy()
                detection.pose_t = _matd_get_array(pose.t).copy()
                detection.pose_err = err
                total_pose_compute += time.perf_counter() - t0_pose

            return_info.append(detection)

        # Restore stderr after all pose estimations
        if estimate_tag_pose and detections.contents.size > 0:
            t0_stderr2 = time.perf_counter()
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
            self.time_stderr_ops += time.perf_counter() - t0_stderr2

        self.time_data_extract = total_data_extract
        self.time_pose_compute = total_pose_compute

        # PROFILE: Cleanup
        t0_cleanup = time.perf_counter()
        self.libc.image_u8_destroy.restype = None
        self.libc.image_u8_destroy(c_img)
        self.libc.apriltag_detections_destroy.restype = None
        self.libc.apriltag_detections_destroy(detections)
        self.time_cleanup = time.perf_counter() - t0_cleanup

        return return_info


# ============================================================================
# TAG VISER RENDERER
# ============================================================================

# Viser visualization modes for detected tags.
# BODY:       tag belongs to a rigid body → green wireframe only (body renders its own axes+label)
# STANDALONE: detector is sole visualizer → full coordinate axes + ID label + wireframe
_TAG_VIS_BODY       = dict(axes_visible=False, label_visible=False)
_TAG_VIS_STANDALONE = dict(axes_visible=True,  label_visible=True)

# Axes size for standalone tag visualization (metres)
_TAG_AXES_LEN    = 0.03
_TAG_AXES_RADIUS = 0.0005


class TagViserRenderer:
    """Renders detected AprilTags in a Viser 3D scene.

    Separates 3D rendering from AprilTagDetector (which handles detection only).
    Uses a pool of reusable Viser scene nodes to avoid per-frame allocations.

    Each pool slot has four nodes under a shared path prefix:
      ``{prefix}``        — anchor frame (position/rotation only, no visible axes)
      ``{prefix}/axes``   — coordinate axes child, toggled visible per-tag type
      ``{prefix}/visual`` — green wireframe outline, always visible when tag is active
      ``{prefix}/label``  — ID label, toggled visible per-tag type

    Visualization adapts based on whether the tag belongs to a rigid body:
      - **Body-bound** (tag_id in known_tag_ids): wireframe only.  TaggedRigidBody already
        renders axes and label for these tags; the outline confirms the tag is live.
      - **Standalone**: full coordinate axes + ID label + wireframe.

    Args:
        server:        Viser server to render into.
        camera_name:   Unique scene-graph prefix for this camera, e.g. ``"/camera_<id>"``.
        tag_size:      Tag (black area) size in metres — sets wireframe padding.
        known_tag_ids: Set of tag IDs claimed by rigid bodies (pass ``TagRegistry.tag_ids``
                       when using the full tracking pipeline). None → all tags are standalone.
    """

    def __init__(self, server: viser.ViserServer, camera_name: str,
                 tag_size: float = 0.030, known_tag_ids: Optional[set] = None):
        self._server = server
        self._name   = camera_name
        self._known  = known_tag_ids or set()

        # Handle pool: reuse Viser nodes across frames instead of creating new ones.
        self._active: Dict[int, dict] = {}  # tag_id → {'anchor', 'axes', 'label'}
        self._pool:   List[dict]      = []  # retired handles ready for reuse
        self._pool_n  = 0                   # monotonic counter for unique node names

        # Camera body (root node in world space; updated every frame by update())
        # Create frame without axes (axes_length=0 makes it invisible as coordinate frame)
        self._cam_frame = server.scene.add_frame(
            camera_name, axes_length=0.0, axes_radius=0.0
        )
        # Add coordinate axes as separate child
        server.scene.add_frame(
            f"{camera_name}/axes",
            axes_length=0.05, axes_radius=0.001
        )
        # Add camera body box
        server.scene.add_box(
            f"{camera_name}/body",
            dimensions=(0.03, 0.03, 0.01),
            color=(0.2, 0.2, 0.2)
        )

        # Pre-compute wireframe geometry once: orange square outline in tag XY plane,
        # padded 6 mm beyond the tag border. Shape (4, 2, 3): 4 segments × 2 pts × XYZ.
        s = tag_size / 2 + 0.006
        corners = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
        )
        self._wf_pts    = np.stack([corners, corners[[1, 2, 3, 0]]], axis=1)  # (4,2,3)
        self._wf_colors = np.full((4, 2, 3), [255, 165, 0], dtype=np.uint8)

    def update(self, tags: list, camera_transform: np.ndarray):
        """Update the 3D scene for all currently detected tags.

        Call once per frame, after ``AprilTagDetector.update_detection()``.

        Args:
            tags:             ``AprilTagDetector.tags`` — list of pupil_apriltags Detection objects.
            camera_transform: ``AprilTagDetector.camera_transform`` — 4×4 camera-to-world SE(3) matrix.
        """
        # Sync camera node to current world pose
        self._cam_frame.position = camera_transform[:3, 3]
        self._cam_frame.wxyz     = matrix_to_quat(camera_transform[:3, :3], wxyz=True)

        current_ids = {tag.tag_id for tag in tags}

        # Update or activate a handle for each detected tag (hot path: just 2 property sets)
        for tag in tags:
            if tag.tag_id not in self._active:
                self._activate(tag.tag_id)
            h = self._active[tag.tag_id]
            h['anchor'].position = tag.pose_t.flatten()
            h['anchor'].wxyz     = matrix_to_quat(tag.pose_R, wxyz=True)

        # Return disappeared tags to pool (renders them invisible)
        for tid in set(self._active) - current_ids:
            h = self._active.pop(tid)
            h['anchor'].visible = False
            self._pool.append(h)

    def _activate(self, tag_id: int):
        """Assign a pool handle to tag_id, creating a new one if the pool is empty."""
        vis = _TAG_VIS_BODY if tag_id in self._known else _TAG_VIS_STANDALONE

        if self._pool:
            # Reuse an existing handle: just flip visibility and update label text
            h = self._pool.pop()
            h['anchor'].visible = True
            h['axes'].visible   = vis['axes_visible']
            h['label'].visible  = vis['label_visible']
            if vis['label_visible']:
                h['label'].text = str(tag_id)  # only update when visible (avoids redundant WebSocket msg)
        else:
            # Pool exhausted — create a new set of Viser nodes
            name = f"{self._name}/tag_pool/{self._pool_n}"
            self._pool_n += 1

            # Anchor: position/rotation only — axes_length=0 keeps it invisible as a frame
            anchor = self._server.scene.add_frame(
                name, axes_length=0.0, axes_radius=0.0
            )
            # Axes: separate child at real size, hidden or shown based on tag type
            axes = self._server.scene.add_frame(
                f"{name}/axes",
                axes_length=_TAG_AXES_LEN, axes_radius=_TAG_AXES_RADIUS,
                visible=vis['axes_visible']
            )
            # Wireframe: always visible when tag is active (independent of axes/label)
            self._server.scene.add_line_segments(
                f"{name}/visual",
                points=self._wf_pts, colors=self._wf_colors, line_width=8.0
            )
            # Label: shown or hidden based on tag type
            label = self._server.scene.add_label(
                f"{name}/label",
                text=str(tag_id),
                visible=vis['label_visible']
            )
            h = {'anchor': anchor, 'axes': axes, 'label': label}

        self._active[tag_id] = h

    def remove(self):
        """Remove all Viser nodes owned by this renderer (camera frame + all pool nodes)."""
        try:
            self._cam_frame.remove()  # removes /body child too
            for h in list(self._active.values()) + self._pool:
                h['anchor'].remove()   # removes /axes, /visual, /label children
        except Exception:
            pass


# ============================================================================
# APRILTAG DETECTOR
# ============================================================================

class AprilTagDetector:
    """AprilTag detection with camera backend separation.

    Handles raw AprilTag detection and 2D OpenCV visualization only.
    For 3D Viser visualization, create a TagViserRenderer and call
    ``renderer.update(detector.tags, detector.camera_transform)`` each frame.

    The camera backend must be initialized before passing to this detector.
    Supports both RealSense and USB camera backends.
    """

    # ---------- CONFIG ----------
    tag_size = 0.030                    # Default tag size in meters
    AXIS_LEN = 0.015                    # Coordinate axes length (1.5 cm)
    FPS_SMOOTHING = 0.9                 # Exponential moving average weight for FPS calculation

    # Color constants (BGR format)
    COLOR_GREEN = (0, 255, 0)     # Y-axis
    COLOR_CYAN = (255, 255, 0)    # Tag text
    COLOR_RED = (0, 0, 255)       # Tag center and X-axis
    COLOR_BLUE = (255, 0, 0)      # Z-axis
    COLOR_ORANGE = (0, 165, 255)  # Tag ID
    COLOR_YELLOW = (0, 255, 255)  # FPS text
    # ----------------------------

    def __init__(self, camera_backend: CameraBackend, tag_size: float = tag_size,
                 tag_sizes: Optional[Dict[int, float]] = None):
        """Initialize the AprilTag detector.

        For 3D Viser visualization, create a TagViserRenderer separately and call
        renderer.update(detector.tags, detector.camera_transform) each frame.

        Args:
            camera_backend: Pre-initialized camera backend (RealSenseCameraBackend or USBCameraBackend).
            tag_size:       Default tag size in metres.
            tag_sizes:      Optional per-tag size overrides: {tag_id: size_m}.

        Example::

            backend = RealSenseCameraBackend(device=(_site.CAMERA_SERIALS[0] if _site.CAMERA_SERIALS else ""), ...)
            backend.initialize(1280, 720, 30)
            detector = AprilTagDetector(backend, tag_size=0.030)
        """
        self.tag_size   = tag_size
        self.tag_sizes  = tag_sizes or {}
        self.camera_backend = camera_backend
        self._init_done = threading.Event()

        # Initialised in background thread (_init_detector)
        self.K           = None
        self.cam_params  = None
        self.detector    = None
        self.axes        = None
        self.vis_buf     = None
        self.gray_buffer = None

        # FPS (prev_time=None until first frame avoids inflated first reading)
        self.prev_time = None
        self.fps       = 0.0

        # Benchmarking (detailed profiling)
        self.detect_time = 0.0
        self.detect_time_core = 0.0  # Core detection only
        self.detect_time_pose = 0.0  # Pose estimation only
        self.draw_time   = 0.0

        # Public outputs, updated each frame
        self.color_image = None  # Latest colour frame
        self.tags        = []    # Latest detected tags

        # Threading
        self._thread  = None
        self._running = False

        # camera_transform: 4×4 camera-to-world SE(3) matrix.
        # Updated by set_camera_pose(); read by estimate_pose() and TagViserRenderer.update().
        self.camera_transform = np.eye(4)

        threading.Thread(target=self._init_detector, daemon=True).start()

    def _init_detector(self):
        """Initialize camera intrinsics and AprilTag detector.

        Runs in a background thread. Signals self._init_done when complete so that
        update_detection() can begin safely.
        """
        self.K = self.camera_backend.get_intrinsics()
        if self.K is None:
            return
        self.cam_params = [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]]

        self.detector = Detector(families='tag36h11', nthreads=4, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=1)

        # Precompute coordinate axes for 3D pose visualization
        self.axes = np.array([[0.0, 0.0, 0.0], [self.AXIS_LEN, 0.0, 0.0],
                                [0.0, self.AXIS_LEN, 0.0], [0.0, 0.0, self.AXIS_LEN]], dtype=np.float32)

        # Cache gray buffer reference.
        # IMPORTANT: valid only if the backend updates this buffer in-place each frame.
        # If the backend allocates a new array per frame, this reference becomes stale.
        self.gray_buffer = self.camera_backend.get_gray_buffer()
        actual_height, actual_width = self.gray_buffer.shape

        self.vis_buf = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)

        self._init_done.set()

    def _draw_detection(self, color_image, tag):
        """Draw single tag detection with pose axes
        +X → right
        +Y → down
        +Z → forward (away from the camera, into the scene)
        """
        # Draw tag boundary
        corners = np.rint(tag.corners).astype(np.int32, copy=False)
        cv2.polylines(color_image, [corners], True, self.COLOR_ORANGE, 1)

        # Draw tag center
        cx, cy = int(tag.center[0]), int(tag.center[1])
        cv2.circle(color_image, (cx, cy), 4, self.COLOR_RED, -1)

        # Project coordinate axes via direct matrix math (skips Rodrigues + projectPoints overhead)
        R = tag.pose_R
        t = tag.pose_t.reshape(3, 1)
        pts_cam = R @ self.axes.T + t              # (3, 4) — axes in camera frame
        pts_2d = self.K @ pts_cam                   # (3, 4) — project
        pts = (pts_2d[:2] / pts_2d[2:]).T.astype(int)  # (4, 2) — normalize

        # Draw coordinate axes
        o, xp, yp, zp = map(tuple, pts)
        cv2.line(color_image, o, xp, self.COLOR_RED, 2)      # X-axis
        cv2.line(color_image, o, yp, self.COLOR_GREEN, 2)    # Y-axis
        cv2.line(color_image, o, zp, self.COLOR_BLUE, 2)     # Z-axis

        cv2.putText(color_image, f"ID {tag.tag_id}",
                    (corners[0][0], corners[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1)

    def set_camera_pose(self, pos, quat_xyzw):
        """Update the camera-to-world transform matrix.

        Args:
            pos:       [x, y, z] camera position in world coordinates.
            quat_xyzw: [x, y, z, w] camera orientation (scipy/PyBullet convention).

        Note:
            Pure matrix update — no Viser side effect. The TagViserRenderer reads
            camera_transform and syncs the Viser camera node inside its update() call.
        """
        self.camera_transform[:3, 3]   = pos
        self.camera_transform[:3, :3]  = quat_to_matrix(quat_xyzw, wxyz=False)

    def update_detection(self):
        """Wait for new frame and process it (consumer pattern)"""
        # Skip if background init has not completed yet
        if not self._init_done.is_set():
            return False

        # Wait for new frame to be available (producer signals this)
        if not self.camera_backend.wait_for_frame(timeout=1.0):
            return False  # Timeout

        # Get buffers from backend (already captured by producer)
        self.color_image = self.camera_backend.get_frame_buffer()
        self.gray_buffer = self.camera_backend.get_gray_buffer()

        # Detect AprilTags with timing
        t0 = time.perf_counter()
        self.tags = self.detector.detect(self.gray_buffer, estimate_tag_pose=True,
                                         camera_params=self.cam_params, tag_size=self.tag_size,
                                         tag_sizes=self.tag_sizes)
        t1 = time.perf_counter()
        self.detect_time = t1 - t0

        # Update FPS — skip first frame (prev_time is None until first call)
        if self.prev_time is not None:
            dt = t1 - self.prev_time
            self.fps = self.FPS_SMOOTHING * self.fps + (1.0 - self.FPS_SMOOTHING) / dt
        self.prev_time = t1

        return True

    def update_2d_visualization(self):
        """update color image with detections drawn on it and copy to self.vis_buf"""
        if self.color_image is None:
            return

        # Time the drawing operations
        t0 = time.perf_counter()

        # Convert grayscale to BGR if needed (check image shape, not camera type)
        if len(self.color_image.shape) == 2:
            # Grayscale image - convert to BGR (already writes to vis_buf, no copy needed)
            cv2.cvtColor(self.color_image, cv2.COLOR_GRAY2BGR, dst=self.vis_buf)
        else:
            # Already BGR - use copyto for in-place copy (faster than assignment)
            # Note: This is needed because draw_detection() modifies the image in-place
            # and we want to preserve the original frame buffer
            np.copyto(self.vis_buf, self.color_image)

        # Draw detections
        for tag in self.tags:
            self._draw_detection(self.vis_buf, tag)

        # Add FPS text
        cv2.putText(self.vis_buf, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_YELLOW, 2)

        self.draw_time = time.perf_counter() - t0
        return self.vis_buf

    def update_detection_continuously(self):
        """Start detection in background thread (non-blocking)

        Uses producer-consumer pattern:
        - Camera backend continuously captures frames (producer)
        - Detection thread waits for frames and processes them (consumer)
        """
        # Start camera backend's continuous capture (producer thread)
        self.camera_backend.capture_frame_continuously()

        # Start detection loop (consumer thread)
        self._running = True
        def _loop():
            while self._running:
                # update_detection() waits for frame signal and processes
                self.update_detection()
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background detection thread and release camera resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.camera_backend.cleanup()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


class CombineGridBuffered:
    """
    Fast grid combiner using pre-allocated buffer reuse (5-10x faster than hstack/vstack).

    Key optimization: Reuses single output buffer instead of allocating new arrays each call.
    Buffer only reallocates if image dimensions change.

    Usage: combiner = CombineGridBuffered(); combined = combiner(vis_list, columns=2)
    """
    def __init__(self):
        self._buffer = None  # Reused output buffer
        self._cache_key = None  # (height, width, n_images) - triggers reallocation if changed

    def __call__(self, vis_list: List[np.ndarray], columns: int = 2) -> np.ndarray | None:
        """Combine images into grid. Assumes all images have same dimensions."""
        vis_list = [img for img in vis_list if img is not None]
        if not vis_list:
            return None

        # Compute output dimensions
        h, w = vis_list[0].shape[:2]
        n_rows = (len(vis_list) + columns - 1) // columns
        cache_key = (n_rows * h, columns * w, len(vis_list))

        # Reallocate buffer if dimensions changed, else just clear it
        if self._cache_key != cache_key:
            self._buffer = np.zeros((cache_key[0], cache_key[1], 3), dtype=np.uint8)
            self._cache_key = cache_key
        else:
            self._buffer.fill(0)  # Fast memset vs reallocation

        # Copy each image directly into buffer (no intermediate arrays)
        for idx, img in enumerate(vis_list):
            row, col = divmod(idx, columns)
            y, x = row * h, col * w
            self._buffer[y:y+h, x:x+w] = img

        return self._buffer


if __name__ == "__main__":
    # ========== AVAILABLE CAMERAS ==========
    # To find USB cameras on your system, run these commands:
    #   ls -la /dev/video*
    #   v4l2-ctl --list-devices
    #
    # RealSense Cameras: (example shape -- run the command above on your rig)
    # [0] Intel RealSense D405  | Serial: <12 digits> | Firmware: x.y.z.w
    #
    # USB Cameras:
    # [4] W4DS USB Camera | Device: /dev/video0 | Resolution: 1920x1080@60fps (MJPEG), 640x480@30fps (YUYV)
    # ======================================

    # ========== SINGLE CAMERA EXAMPLE ==========
    # Example: USB camera with calibration
    # backend = USBCameraBackend(
    #     device='/dev/video0',
    #     profile='camera_calibration.yaml'  # Optional calibration file
    # )
    # backend.initialize(width=1280, height=720, framerate=30)
    # detector = AprilTagDetector(
    #     camera_backend=backend,
    #     tag_size=0.030,
    # )
    # renderer = TagViserRenderer(viser.ViserServer(), f"/camera_{id(backend)}", tag_size=0.030)
    # backend.capture_frame_continuously()
    # try:
    #     while True:
    #         if detector.update_detection():
    #             detector.update_2d_visualization()
    #             renderer.update(detector.tags, detector.camera_transform)
    # except KeyboardInterrupt:
    #     detector.stop()
    #     renderer.remove()
    # ========================================

    # ========== MULTI-CAMERA EXAMPLE ==========
    # Enable/disable individual cameras
    ENABLE_DETECTOR1 = 0   # vs_site.CAMERA_SERIALS[1 - 1]
    ENABLE_DETECTOR2 = 0   # vs_site.CAMERA_SERIALS[2 - 1]
    ENABLE_DETECTOR3 = 0   # vs_site.CAMERA_SERIALS[3 - 1]
    ENABLE_DETECTOR4 = 0   # vs_site.CAMERA_SERIALS[4 - 1]
    ENABLE_DETECTOR5 = 1   # USB Camera

    # Camera settings
    RESOLUTION = (1280, 720)
    FPS = 30
    USE_IR_STREAM = False  # Use IR (grayscale) for RealSense (more efficient)

    # Display settings
    USE_CV2_DISPLAY = True  # True: OpenCV window (requires X11), False: Viser-only (headless-friendly)

    # Create Viser server for 3D visualization
    print("=" * 60)
    enabled_count = sum([ENABLE_DETECTOR1, ENABLE_DETECTOR2, ENABLE_DETECTOR3, ENABLE_DETECTOR4, ENABLE_DETECTOR5])
    if enabled_count > 1:
        print(f"MULTI-CAMERA MODE ({enabled_count} cameras)")
    else:
        print("SINGLE CAMERA MODE")
    print("=" * 60)

    viser_server = viser.ViserServer()

    # Setup CV2 window only if enabled
    if USE_CV2_DISPLAY:
        cv2.namedWindow("Multi-Camera View", cv2.WINDOW_NORMAL)
        print("• Display: OpenCV window + Viser web GUI")
    else:
        print("• Display: Viser web GUI only (headless mode)")
        print("• Access at: http://localhost:8080")

    if enabled_count > 1:
        print("• Viser GUI: 3D tag visualization + 2D camera feeds")
        print("• All camera feeds available in web interface")
        print("=" * 60)

    # Create buffered grid combiner for efficient visualization
    grid_combiner = CombineGridBuffered()

    # Define camera configurations
    # Serials come from vs_site (env / vs_site_local.py); "" lets the backend
    # pick whatever RealSense is attached, which is right on any other rig.
    _sn = list(_site.CAMERA_SERIALS) + [""] * 4
    camera_configs = [
        (ENABLE_DETECTOR1, "realsense", _sn[0], USE_IR_STREAM),
        (ENABLE_DETECTOR2, "realsense", _sn[1], USE_IR_STREAM),
        (ENABLE_DETECTOR3, "realsense", _sn[2], False),
        (ENABLE_DETECTOR4, "realsense", _sn[3], False),
        (ENABLE_DETECTOR5, "usb", "OBSBOT Meet 2", None),
    ]

    # Create and initialize backends
    backends = []
    for enabled, cam_type, identifier, use_ir in camera_configs:
        if not enabled:
            backends.append(None)
            continue

        if cam_type == "realsense":
            backend = RealSenseCameraBackend(device=identifier, ir_stream_flag=use_ir, color_stream_flag=not use_ir)
        else:  # usb
            device = USBCameraBackend.find_camera_by_name(identifier)
            backend = USBCameraBackend(device=device, profile='camera_calibration.yaml')

        backend.initialize(*RESOLUTION, FPS)
        backends.append(backend)

    tag_size = 0.030  # Default tag size (30mm)
    tag_sizes = {100: 0.1}  # Example: tag ID 100 is 100mm

    # Create detectors and renderers from backends
    detectors = []
    renderers = []
    viser_image_handles = []  # Store Viser GUI image handles
    camera_names = ["Camera 1 (D435I)", "Camera 2 (D405)", "Camera 3 (D435I)", "Camera 4 (D435)", "Camera 5 (USB)"]

    for idx, backend in enumerate(backends):
        if backend:
            detector = AprilTagDetector(camera_backend=backend, tag_size=tag_size, tag_sizes=tag_sizes)
            renderer = TagViserRenderer(viser_server, f"/camera_{id(backend)}", tag_size)
            detector.update_detection_continuously()
            detectors.append(detector)
            renderers.append(renderer)

            # Add Viser 3D scene image as child of camera frame
            # Calculate image aspect ratio
            aspect_ratio = RESOLUTION[0] / RESOLUTION[1]  # width / height
            render_height = 0.3  # 0.3 meters tall in camera-local space
            render_width = render_height * aspect_ratio

            # Create image as child of camera frame (positioned in front of camera)
            # Camera frame: +Z forward, +X right, +Y down
            camera_frame_name = f"/camera_{id(backend)}"
            viser_img = viser_server.scene.add_image(
                f"{camera_frame_name}/feed",
                np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8),
                render_width=render_width,
                render_height=render_height,
                position=(0.0, 0.0, 0.4),  # 0.4m in front of camera
                format='jpeg',
                jpeg_quality=85
            )
            viser_image_handles.append(viser_img)
        else:
            detectors.append(None)
            renderers.append(None)
            viser_image_handles.append(None)

    try:
        # Print stats every 1 second for enabled cameras
        t1 = time.perf_counter()
        while True:
            # Update visualization buffers for all detectors
            for detector in detectors:
                if detector:
                    detector.update_2d_visualization()

            # Update Viser GUI image panels (convert BGR to RGB)
            for detector, viser_img in zip(detectors, viser_image_handles):
                if detector and viser_img and detector.vis_buf is not None:
                    rgb_frame = cv2.cvtColor(detector.vis_buf, cv2.COLOR_BGR2RGB)
                    viser_img.image = rgb_frame  # Use .image property, not .value!

            # CV2 display (optional, only if enabled)
            if USE_CV2_DISPLAY:
                vis_buffers = [detector.vis_buf if detector else None for detector in detectors]
                n_active = sum(1 for d in detectors if d is not None)
                combined = grid_combiner(vis_buffers, columns=1 if n_active <= 1 else 2)
                if combined is not None:
                    cv2.imshow("Multi-Camera View", combined)
                cv2.waitKey(1)

            # Update 3D visualization
            for detector, renderer in zip(detectors, renderers):
                if detector and renderer:
                    renderer.update(detector.tags, detector.camera_transform)
            if viser_server is not None:
                viser_server.flush()

            current = time.perf_counter()
            if current - t1 >= 1.0:
                for i, detector in enumerate(detectors):
                    if detector:
                        # Detailed profiling breakdown
                        core = detector.detector.time_core_detect * 1000
                        extract = detector.detector.time_data_extract * 1000
                        pose = detector.detector.time_pose_compute * 1000
                        stderr = detector.detector.time_stderr_ops * 1000
                        cleanup = detector.detector.time_cleanup * 1000
                        total = detector.detect_time * 1000
                        draw = detector.draw_time * 1000
                        print(f"{camera_names[i]} - FPS:{detector.fps:.1f} "
                              f"Detect:{total:.1f}ms (Core:{core:.1f}ms Extract:{extract:.1f}ms Pose:{pose:.1f}ms Stderr:{stderr:.1f}ms) "
                              f"Draw:{draw:.1f}ms Tags:{len(detector.tags)}")
                print("-" * 60)
                t1 = current
    except KeyboardInterrupt:
        pass
    finally:
        # Stop enabled detectors
        for detector, renderer in zip(detectors, renderers):
            if detector:
                detector.stop()
            if renderer:
                renderer.remove()

        # Clean up OpenCV window (only if it was created)
        if USE_CV2_DISPLAY:
            cv2.destroyWindow("Multi-Camera View")
