"""Reusable single-camera AprilTag rigid-body detector.

Phase-1 module for humanoid_monitor: owns one streaming camera + AprilTag detector,
produces latest snapshot (annotated frame + per-body poses) with latest-value
semantics.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

_common_pkg = sys.modules.get("common")
if _common_pkg is not None and hasattr(_common_pkg, "__path__"):
    _vs_common = os.path.join(os.path.dirname(__file__), "common")
    if _vs_common not in list(_common_pkg.__path__):
        _common_pkg.__path__.append(_vs_common)

from camera_streaming_utils import StreamingCameraBackend
from camera_tag_detection import AprilTagDetector
from tagged_bodies import ALL_CONFIGS
from tagged_rigid_body import TagRegistry, TaggedRigidBody


# T_ROBOT_HEAD_CAM = np.array(
#     [[0.0, -0.7071, 0.7071, 0.06684],
#      [-1.0, 0.0, 0.0, 0.0175],
#      [0.0, -0.7071, -0.7071, 0.42003],
#      [0.0, 0.0, 0.0, 1.0]],
#     dtype=float,
# )

# # Chest (front) camera: lens points forward (+X robot), mounted at z≈0.365 m.
# T_ROBOT_CHEST_CAM = np.array(
#     [[0.0,  0.0,  1.0, 0.07275],
#      [-1.0, 0.0,  0.0, 0.03250],
#      [0.0, -1.0,  0.0, 0.36500],
#      [0.0,  0.0,  0.0, 1.0]],
#     dtype=float,
# )

#   T_ROBOT_HEAD_CAM (port 5555):                                                                                          
T_ROBOT_HEAD_CAM = np.array( # SHORT HEAD
    [[0.0, -0.7071, 0.7071, 0.08237],   # was 0.06684                                                                  
    [-1.0, 0.0, 0.0, 0.03300],          # was 0.01750                                                                 
    [0.0, -0.7071, -0.7071, 0.42164],   # was 0.42003
    [0.0, 0.0, 0.0, 1.0]], dtype=float)                                                                                                                    
                        
                        
T_ROBOT_HEAD_CAM = np.array( # LONG HEAD
    [[0.0, -0.8, 0.6, 0.02377433243],
     [-1.0, 0.0, 0.0, 0.0175],
     [0.0, -0.6,-0.8, 0.56719387374],
     [0.0,  0.0, 0.0, 1.0]],
    dtype=float,
)

                        
                                                                                                                        
#   T_ROBOT_CHEST_CAM (port 5556):                                                                                         
T_ROBOT_CHEST_CAM = np.array(
    [[0.0,  0.0,  1.0, 0.07271],   # was 0.07275                                                                       
    [-1.0, 0.0,  0.0, 0.03295],   # was 0.03250                                                                     
    [0.0, -1.0,  0.0, 0.36520],   # was 0.36500                                                                       
    [0.0,  0.0,  0.0, 1.0]], dtype=float) 


CAMERA_TRANSFORMS = {
    "head": T_ROBOT_HEAD_CAM,
    "chest": T_ROBOT_CHEST_CAM,
}


@dataclass(frozen=True)
class BodySpec:
    key: str
    body_name: str
    frame_name: str | None
    cfg: object


@dataclass
class SingleCameraBodyPose:
    key: str
    body_name: str
    frame_name: str | None
    pos: np.ndarray
    quat_wxyz: np.ndarray
    n_inliers: int
    camera_port: int
    capture_stamp: float


@dataclass
class CameraDetectionSnapshot:
    frame_id: int
    camera_port: int
    capture_stamp: float
    detect_hz: float
    vis_bgr: np.ndarray | None
    body_poses: dict[str, SingleCameraBodyPose]
    target_poses: dict[str, SingleCameraBodyPose]


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def expand_body_spec(spec: str) -> list[BodySpec]:
    """Expand a body spec string to one or more BodySpec instances.

    "body_name"        → expands to all frames if cfg.frames is non-empty,
                         else one body-center BodySpec (frame_name=None)
    "body_name:frame"  → single BodySpec for that exact frame (validated)
    """
    parts = spec.split(":", 1)
    body_name = parts[0]
    frame_name = parts[1] if len(parts) == 2 else None

    cfg = next((c for c in ALL_CONFIGS if c.name == body_name), None)
    if cfg is None:
        raise ValueError(f"Unknown body '{body_name}'. Available: {[c.name for c in ALL_CONFIGS]}")

    if frame_name is not None:
        if frame_name not in cfg.frames:
            raise ValueError(
                f"Body '{body_name}' has no frame '{frame_name}'. "
                f"Available: {list(cfg.frames) or '(none)'}"
            )
        return [BodySpec(key=spec, body_name=body_name, frame_name=frame_name, cfg=cfg)]

    if cfg.frames:
        return [
            BodySpec(key=f"{body_name}:{fname}", body_name=body_name, frame_name=fname, cfg=cfg)
            for fname in cfg.frames
        ]
    return [BodySpec(key=body_name, body_name=body_name, frame_name=None, cfg=cfg)]


class CameraTagDetector:
    """Single-camera rigid-body detector with latest snapshot slot."""

    def __init__(
        self,
        camera_host: str,
        camera_port: int = 5556,
        camera_transform: np.ndarray = T_ROBOT_HEAD_CAM,
        body_specs: list[str] | None = None,
        quit_flag: threading.Event | None = None,
        enable_vis: bool = False,
    ):
        self.camera_host = camera_host
        self.camera_port = camera_port
        self.camera_transform = np.asarray(camera_transform, dtype=float)
        self.quit_flag = quit_flag
        self.enable_vis = enable_vis

        if body_specs is None:
            body_specs = [
                "holder_6tag_pipette_tip",
                "holder_6tag_bottle_42mm",
            ]

        self.specs: list[BodySpec] = [spec for s in body_specs for spec in expand_body_spec(s)]
        self._unique_cfgs = list({spec.cfg.name: spec.cfg for spec in self.specs}.values())

        self._registry = TagRegistry(self._unique_cfgs)
        self._bodies = {cfg.name: TaggedRigidBody(cfg, server=None) for cfg in self._unique_cfgs}

        self._backend: StreamingCameraBackend | None = None
        self._detector: AprilTagDetector | None = None

        self._lock = threading.Lock()
        self.latest: CameraDetectionSnapshot | None = None
        self._frame_id = 0

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        self._backend = StreamingCameraBackend(self.camera_host, self.camera_port)
        self._backend.initialize(1280, 720, 30)
        self._backend.capture_frame_continuously()

        self._detector = AprilTagDetector(
            camera_backend=self._backend,
            tag_sizes=self._registry.tag_sizes,
        )

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera_tag_detector")
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._detector is not None:
            self._detector.stop()
        elif self._backend is not None:
            self._backend.cleanup()

    def _loop(self):
        while self._running and (self.quit_flag is None or not self.quit_flag.is_set()):
            detector = self._detector
            backend = self._backend
            if detector is None or backend is None:
                time.sleep(0.01)
                continue

            if not detector.update_detection():
                time.sleep(0.005)  # prevent spin-loop GIL starvation when no camera frames arrive
                continue

            if self.enable_vis:
                detector.update_2d_visualization()
            grouped = self._registry.detect_bodies(detector.tags)

            body_poses: dict[str, SingleCameraBodyPose] = {}
            target_poses: dict[str, SingleCameraBodyPose] = {}

            capture_stamp = float(getattr(backend, "last_frame_recv_stamp", 0.0) or 0.0)
            if capture_stamp <= 0.0:
                capture_stamp = time.monotonic()

            estimates: dict[str, tuple] = {}
            for cfg in self._unique_cfgs:
                det_tags = grouped.get(cfg, [])
                if not det_tags:
                    continue
                est = self._bodies[cfg.name].estimate_pose(
                    det_tags,
                    self.camera_transform,
                    detector.K,
                )
                if est is None:
                    continue
                estimates[cfg.name] = (est, int(np.count_nonzero(est.inliers)))

                body_quat = _rot_to_quat_wxyz(est.R_world)
                body_poses[cfg.name] = SingleCameraBodyPose(
                    key=cfg.name,
                    body_name=cfg.name,
                    frame_name=None,
                    pos=est.t_world.copy(),
                    quat_wxyz=body_quat,
                    n_inliers=int(np.count_nonzero(est.inliers)),
                    camera_port=self.camera_port,
                    capture_stamp=capture_stamp,
                )

            for spec in self.specs:
                est_pack = estimates.get(spec.body_name)
                if est_pack is None:
                    continue
                est, n_inliers = est_pack
                if spec.frame_name is None:
                    pos = est.t_world
                    R = est.R_world
                else:
                    idx = spec.cfg.frame_name_to_idx[spec.frame_name]
                    pos = est.R_world @ spec.cfg.frame_pos_arr[idx] + est.t_world
                    R = est.R_world @ spec.cfg.frame_rot_arr[idx]

                target_poses[spec.key] = SingleCameraBodyPose(
                    key=spec.key,
                    body_name=spec.body_name,
                    frame_name=spec.frame_name,
                    pos=np.asarray(pos, dtype=float).copy(),
                    quat_wxyz=_rot_to_quat_wxyz(R),
                    n_inliers=n_inliers,
                    camera_port=self.camera_port,
                    capture_stamp=capture_stamp,
                )

            snapshot = CameraDetectionSnapshot(
                frame_id=self._frame_id,
                camera_port=self.camera_port,
                capture_stamp=capture_stamp,
                detect_hz=float(detector.fps),
                vis_bgr=detector.vis_buf.copy() if self.enable_vis and detector.vis_buf is not None else None,
                body_poses=body_poses,
                target_poses=target_poses,
            )
            self._frame_id += 1

            with self._lock:
                self.latest = snapshot

    def get_latest(self) -> CameraDetectionSnapshot | None:
        with self._lock:
            return self.latest


@dataclasses.dataclass
class Args:
    camera_host: str = "127.0.0.1"
    camera_port: int = 5556
    transform: str = "head"
    bodies: list[str] = dataclasses.field(default_factory=lambda: [
        "holder_6tag_pipette_tip",
        "holder_6tag_bottle_42mm",
    ])
    print_hz: float = 5.0


def main():
    import tyro

    args = tyro.cli(Args)
    if args.transform not in CAMERA_TRANSFORMS:
        raise ValueError(f"Unknown transform '{args.transform}'. Available: {list(CAMERA_TRANSFORMS)}")

    quit_flag = threading.Event()
    detector = CameraTagDetector(
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        camera_transform=CAMERA_TRANSFORMS[args.transform],
        body_specs=args.bodies,
        quit_flag=quit_flag,
    )
    detector.start()

    period = 1.0 / max(args.print_hz, 0.5)
    last_frame_id = -1
    try:
        while True:
            snap = detector.get_latest()
            if snap is None or snap.frame_id == last_frame_id:
                time.sleep(period)
                continue
            last_frame_id = snap.frame_id
            age_ms = (time.monotonic() - snap.capture_stamp) * 1000.0
            if not snap.target_poses:
                print(f"[{snap.camera_port}] {snap.detect_hz:5.1f} Hz  NO_TAG  stamp_age={age_ms:5.1f}ms")
            for key, pose in sorted(snap.target_poses.items()):
                pos = pose.pos
                print(
                    f"[{snap.camera_port}] {snap.detect_hz:5.1f} Hz  {key:28s} "
                    f"pos=[{pos[0]:+0.3f} {pos[1]:+0.3f} {pos[2]:+0.3f}] "
                    f"inliers={pose.n_inliers:2d} stamp_age={age_ms:5.1f}ms"
                )
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        quit_flag.set()
        detector.stop()


if __name__ == "__main__":
    main()
