"""Unified humanoid monitor.

Subscribes to robot telemetry (port 9870), runs multi-camera AprilTag rigid-body
detection, shows everything in viser, and publishes detected target poses on the
detection socket (default ipc:///tmp/humanoid_detections.sock, --detection-pub-url)
for downstream consumers (the operator, the reach tool, humanoid_visual_manip_debug_gui.py).

Inputs:
  tcp://<robot>:9870   — telemetry (joint_*, IMU, ee_poses, stamp)
  tcp://<robot>:5555   — LEFT head-gimbal camera frames (default)
  tcp://<robot>:5556   — RIGHT head-gimbal camera frames (default)
  (both cameras ride the head gimbal side by side — robot.xml cam_base_left /
  cam_base_right; the port<->side binding is set by the streamer's --devices)

Outputs:
  detection socket     — MonitorDetectionPacket (detected rigid-body poses in body frame);
                         ipc:///tmp/humanoid_detections.sock by default (--detection-pub-url)
                         Published each time any camera's detection frame advances.
                         Payload: {stamp, frame_id_by_port, targets: {key: MonitorTargetPose}}

Architecture — three independent threads:

  _viser_thread (20 Hz)
    → robot pose/detection updates via HumanoidViserViz (single-threaded, no races)
    → all viser scene updates wrapped in server.atomic() to eliminate WS message bursts
    → scene.flush() only when detection frames changed
    → publishes MonitorDetectionPacket on the detection socket after each flush

  _cam_bg_loop (event-driven, ≤25 Hz)
    → wakes on new camera frame (no polling lag)
    → per-viewport padded array cache → set_background_image
    → rate-limits to 25 Hz to protect browser websocket queue

  _cam_capture (camera framerate)
    → MultiStreamingCameraClient captures, hstack, optional resize
    → stores RGB frame; signals _cam_bg_loop via Event

  _monitor_telem (event-driven, ≤20 Hz)
    → NNGSubscriber recv; updates ChannelStats on new packet


                                                                               
python humanoid_monitor.py --hide-links ee_left ee_right wrist_3_L wrist_3_R  
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tyro

from ipc.publisher import NNGPublisher, NNGSubscriber

# visual_servoing has same-named modules (common/, tagged_bodies/, etc.)
from humanoid_site import VISUAL_SERVOING_ROOT as _VS_ROOT  # noqa: E402
import humanoid_site as _site  # noqa: E402
sys.path.insert(0, str(_VS_ROOT))
from camera_tag_detector import CameraTagDetector, CAMERA_TRANSFORMS  # noqa: E402
from tagged_bodies import ALL_CONFIGS  # noqa: E402
from tagged_rigid_body import BatchedScene  # noqa: E402

# humanoid_utils lives in the script's directory (already on sys.path via cwd)
from humanoid_model import MJCF_MODEL_PATH  # noqa: E402
from humanoid_utils import HumanoidViserViz, get_joint_info_from_mjcf  # noqa: E402

TELEMETRY_PORT = _site.TELEMETRY_PORT
DETECTION_IPC_URL = "ipc:///tmp/humanoid_detections.sock"
RATE_WINDOW = 1.0
_ANSI_CLEAR = "\033[2J\033[H"
DETECTION_RETENTION_S = 0.5  # publisher-side sighting retention. A scan-transit
                             # decode often lives in 1-2 camera frames (33-66 ms)
                             # and then has to survive TWO latest-only samplers:
                             # this 20 Hz packet build ships only the newest
                             # snapshot per camera, and the operator's poll is
                             # latest-packet-wins — end-to-end survival of a
                             # single-frame sighting was a coin flip, which is
                             # why cameras kept sweeping past cubes the viser
                             # view (which retains 0.5 s for display) was
                             # visibly drawing (hardware 2026-08-03). Retaining
                             # each key's newest pose for 0.5 s puts it in EVERY
                             # packet of that window; capture_stamp still carries
                             # the true age for any consumer that cares.

# Tag cube bodies → MJCF mounting links for hand-eye calibration recording.
# Keys: visual_servoing body names; values: MJCF body names where cubes are mounted.
_HANDEYE_OBJECTS: dict[str, str] = {
    "tag_cube_0": "end_effector_L",
    "tag_cube_1": "end_effector_R",
}


TELEMETRY_FIELDS = [
    "base_lin_vel", "base_ang_vel", "projected_gravity",
    "joint_pos", "joint_vel", "joint_effort", "joint_effort_ref",
    "actions", "command", "imu_quat_wxyz", "vicon_pos",
    "wb_pos", "wb_quat_wxyz",
]


@dataclasses.dataclass
class MonitorTargetPose:
    """One tagged body as the monitor saw it, in the ROBOT BASE frame.

    Fields: `key` is the visual_servoing body name; `pos`/`quat_wxyz` are the
    fused pose; `n_inliers` is the PnP inlier count backing it; `camera_port`
    is the camera that MEASURED this sample (5555 = left/front datum, 5556 =
    right/rear); `capture_stamp` is when the frame was captured, not when it
    arrived. Consumers age evidence on the capture stamp -- arrival age hides
    pipeline latency and would let a stale pose pass a freshness gate.
    """
    key: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    n_inliers: int
    camera_port: int
    capture_stamp: float


@dataclasses.dataclass
class MonitorDetectionPacket:
    """One publish on the detection IPC socket: everything seen at one instant.

    `targets` is keyed by body name; `frame_id_by_port` lets a consumer tell a
    genuinely new frame from a repeat of the last one per camera; `ee` carries
    the measured and commanded end-effector poses when telemetry is available,
    and is `{}` when it is not. Consumers must treat a missing key as "not
    seen this tick", never as "gone".
    """
    stamp: float
    frame_id_by_port: dict[int, int]
    targets: dict[str, MonitorTargetPose]
    ee: dict  # {"left": {"pos_actual": [...], "quat_actual": [...], "pos_target": [...], "quat_target": [...]}, ...} or {}


def build_retained_targets(retained: dict, snapshots, now: float,
                           retention_s: float = DETECTION_RETENTION_S) -> dict:
    """Fold camera snapshots' target poses into `retained` (mutated in place)
    and return the packet view: every key whose newest decode is younger than
    `retention_s`. Newest capture_stamp wins across cameras — the old
    last-camera-wins overwrite let a stale grazing view clobber the tracking
    camera's fresher fit. A frozen camera cannot pin a ghost: entries are
    pruned on capture AGE, so a stale snapshot re-folding the same pose still
    expires on schedule. `now` and capture_stamp share the monotonic clock."""
    for snap in snapshots:
        for key, pose in snap.target_poses.items():
            cur = retained.get(key)
            if cur is None or pose.capture_stamp >= cur.capture_stamp:
                retained[key] = pose
    for key in [k for k, p in retained.items()
                if now - p.capture_stamp > retention_s]:
        del retained[key]
    return dict(retained)


class GimbalCameraFK:
    """T_base_link←camera for cameras riding the head gimbal, from live joint angles.

    The monitor already receives the full 31-joint state on port 9870 and already
    has the robot MJCF; this class plugs the gimbal angles into that model and
    reads out the camera site pose — replacing the hard-coded static extrinsic,
    which is wrong whenever the gimbal is away from neutral. The MJCF rgb sites
    (cam_left_rgb / cam_right_rgb) already use the OpenCV camera convention
    (z forward, x right, y down), so the site pose IS the camera extrinsic.

    Thread model: base_to_camera() mutates one shared MjData — call it from a
    single thread (_monitor_telem). It returns a fresh array per call; assigning
    it to detector.camera_transform is an atomic reference swap picked up by the
    detector loop on its next frame.
    """

    def __init__(self, mounts: dict[int, str]):
        import mujoco
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        self._data = mujoco.MjData(self._model)
        self._data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]  # identity root → poses in base_link frame
        self.n_joints = self._model.nq - 7
        self.site_ids: dict[int, int] = {}
        for port, site in mounts.items():
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site)
            if sid < 0:
                raise ValueError(f"--tag-mounts: site '{site}' not found in {MJCF_MODEL_PATH}")
            self.site_ids[port] = sid
        self._short_jpos_warned = False

    def base_to_camera(self, port: int, joint_pos) -> np.ndarray:
        """4x4 T_base_link←camera at the given joint vector (None → all joints zero)."""
        self._data.qpos[7:] = 0.0  # no stale angles from previous calls (esp. beyond a short vector)
        if joint_pos is not None:
            if len(joint_pos) < self.n_joints and not self._short_jpos_warned:
                self._short_jpos_warned = True
                print(f"[gimbal] WARNING: telemetry has {len(joint_pos)} joints, model has "
                      f"{self.n_joints} — gimbal joints stuck at 0, camera transform will be "
                      f"wrong whenever the gimbal moves")
            k = min(len(joint_pos), self.n_joints)
            self._data.qpos[7:7 + k] = joint_pos[:k]
        self._mujoco.mj_kinematics(self._model, self._data)
        sid = self.site_ids[port]
        T = np.eye(4)
        T[:3, :3] = self._data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self._data.site_xpos[sid]
        return T


class TelemetryChannelStats:
    """Rolling arrival-rate and staleness for one subscribed channel.

    Rate is measured over a fixed RATE_WINDOW rather than smoothed, so a
    channel that dies reads 0 Hz within one window instead of decaying
    gently -- the panel is there to make a dead publisher obvious.
    """
    def __init__(self) -> None:
        self.msg_count = 0
        self.last_recv_t: float | None = None
        self.last_data: dict | None = None
        self._timestamps: deque[float] = deque()

    def update(self, data: dict) -> None:
        """Record one received message and prune the rate window."""
        self.msg_count += 1
        now = time.monotonic()
        self.last_recv_t = now
        self.last_data = data
        self._timestamps.append(now)
        # Bounded prune of timestamps older than RATE_WINDOW
        while self._timestamps and now - self._timestamps[0] > RATE_WINDOW:
            self._timestamps.popleft()

    @property
    def hz(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        duration = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / max(duration, 1e-6)

    def staleness(self) -> float | None:
        """Seconds since the last message, or None if nothing ever arrived.

        None and 0.0 mean opposite things: never-connected vs just-heard.
        """
        return None if self.last_recv_t is None else time.monotonic() - self.last_recv_t


def _format_array(arr: Any, fmt: str = ".3f") -> str:
    arr = np.asarray(arr).ravel()
    values = arr if arr.size <= 6 else arr[:4]
    body = ", ".join(f"{v:{fmt}}" for v in values)
    suffix = f" … ({arr.size})" if arr.size > 6 else ""
    return f"[{body}{suffix}]"


def _render_terminal_stats(
    telemetry_stats: TelemetryChannelStats,
    camera_tag_detectors: dict[int, CameraTagDetector],
    fields_filter: list[str] | None,
) -> str:
    lines = ["=== humanoid_monitor ==="]
    stale = telemetry_stats.staleness()
    stale_str = f"{stale*1000:.1f} ms ago" if stale is not None else "never"
    lines.append(
        f"\n[Port {TELEMETRY_PORT}] msgs={telemetry_stats.msg_count} "
        f"rate={telemetry_stats.hz:.1f} Hz last={stale_str}"
    )

    if telemetry_stats.last_data is None:
        lines.append("  no telemetry data received yet")
    else:
        for field in TELEMETRY_FIELDS:
            if fields_filter and field not in fields_filter:
                continue
            val = telemetry_stats.last_data.get(field)
            if val is None:
                lines.append(f"  {field:25s} None")
            else:
                lines.append(f"  {field:25s} last={_format_array(val)}")

    if not camera_tag_detectors:
        lines.append("\n[Detection] disabled")
    else:
        for detector in camera_tag_detectors.values():
            snap = detector.get_latest()
            if snap is None:
                lines.append(f"\n[Detection] cam={detector.camera_port} waiting for frames")
                continue
            age_ms = (time.monotonic() - snap.capture_stamp) * 1000.0
            lines.append(
                f"\n[Detection] cam={detector.camera_port} frame={snap.frame_id} "
                f"rate={snap.detect_hz:.1f} Hz age={age_ms:.1f} ms"
            )
            if not snap.target_poses:
                lines.append("  NO_TAG")
            for key, pose in snap.target_poses.items():
                lines.append(f"  {key:28s} pos={_format_array(pose.pos)} inliers={pose.n_inliers}")

    return "\n".join(lines)


def _pad_to_aspect(frame: np.ndarray, vp_w: int, vp_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target_aspect = vp_w / vp_h
    frame_aspect = w / h
    if frame_aspect > target_aspect:
        # frame wider than viewport: add black bars top and bottom
        new_h = int(w / target_aspect)
        pad = (new_h - h) // 2
        return np.pad(frame, ((pad, new_h - h - pad), (0, 0), (0, 0)))
    # frame taller than viewport: add black bars left and right
    new_w = int(h * target_aspect)
    pad = (new_w - w) // 2
    return np.pad(frame, ((0, 0), (pad, new_w - w - pad), (0, 0)))


def _viser_thread(
    telemetry_stats: TelemetryChannelStats,
    camera_tag_detectors: dict[int, CameraTagDetector],
    args: "Args",
    quit_event: threading.Event,
    detection_publisher: NNGPublisher | None,
):
    """The monitor's render / detect / publish loop — owns the viser server.

    Sets up the scene (robot mannequin, tagged bodies, target frames, cuRobo
    route-preview clouds), spawns the camera threads (_cam_capture ->
    _cam_bg_loop, which also draws the route overlay into each camera image),
    then ticks at 20 Hz: apply the newest telemetry to the mannequin, drain
    the route-preview socket, pull each camera's latest detection snapshot
    into the 3D view, and — whenever anything changed — publish one
    MonitorDetectionPacket (publisher-side retention, see
    build_retained_targets) and flush the batched scene. Runs until
    quit_event is set."""
    import viser
    from camera_streaming_utils import MultiStreamingCameraClient

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print(f"[monitor] viser at http://0.0.0.0:{server.get_port()}")

    joint_names, _, _ = get_joint_info_from_mjcf(MJCF_MODEL_PATH)
    # enable_internal_loop=False: this thread drives viser updates directly,
    # eliminating concurrent handle writes from HumanoidViserViz's own 50 Hz loop.
    viser_viz = HumanoidViserViz(
        mjcf_path=MJCF_MODEL_PATH,
        joint_names=joint_names,
        show_robot=True,
        server=server,
        enable_internal_loop=False,
        hidden_links=set(args.hide_links),
    )

    batched_scene = BatchedScene(server, retention_time_s=1.0, prefix="/robot/base_link")
    active_body_names = {spec.body_name for detector in camera_tag_detectors.values() for spec in detector.specs}
    # 3D cube-body rendering is viz-only and pulls an optional tag-image dependency
    # (moms_apriltag). A missing dep must NOT kill this thread — it also owns the
    # detection publisher. Skip un-renderable bodies; detection/publishing continue.
    viser_body_handles: dict[str, Any] = {}
    for cfg in ALL_CONFIGS:
        if cfg.name not in active_body_names:
            continue
        try:
            viser_body_handles[cfg.name] = batched_scene.add(cfg)
        except Exception as e:
            print(f"[monitor] 3D body '{cfg.name}' not rendered ({type(e).__name__}: {e}); "
                  f"detection unaffected")

    target_frame_handles: dict[str, Any] = {
        spec.key: server.scene.add_frame(
            f"/robot/base_link/targets/{spec.key}",
            axes_length=0.05, axes_radius=0.003, visible=False,
        )
        for detector in camera_tag_detectors.values()
        for spec in detector.specs
    }

    # ---- cuRobo route preview -------------------------------------------------
    # humanoid_curobo_reach publishes its gated route's EE path (base frame)
    # on an IPC socket; we draw it TWICE: projected into each camera's live
    # image (the first-person view) and as a point cloud under
    # /robot/base_link (rides the 3D mannequin). Passive: nothing here ever
    # publishes back, and an absent/naive publisher costs nothing.
    route_sub = NNGSubscriber(args.route_preview_url)
    route_sub.start()
    route_state: dict = {"paths": {}, "cands": {}}   # written by the 20 Hz loop,
                                                     # read by the capture thread
    route_sub_id = None
    _overlay_mounts = {p: s for p, s in args.tag_mounts.items()
                       if args.camera_port <= p < args.camera_port + args.n_cameras}
    # Separate FK instance: base_to_camera mutates its scratch MjData, and the
    # tag-detection sync loop already owns gimbal_fk's — no sharing across threads.
    fk_overlay = GimbalCameraFK(_overlay_mounts) if _overlay_mounts else None
    route_cloud_handles: dict[str, Any] = {}

    def _draw_route_overlay(frame_bgr: np.ndarray, port: int) -> None:
        """Project the planned EE path + grasp candidates into this camera's
        live image: green polyline = route, red dot = final EE point, yellow
        dots = goalset candidates. Extrinsic is live gimbal FK, intrinsics are
        the tag detector's own K — the same pair the detections trust."""
        paths = route_state["paths"]
        if not paths or fk_overlay is None or port not in _overlay_mounts:
            return
        det = camera_tag_detectors.get(port)
        K = getattr(getattr(det, "_detector", None), "K", None)
        tel = telemetry_stats.last_data or {}
        jpos = tel.get("joint_pos")
        if K is None or jpos is None:
            return
        T = fk_overlay.base_to_camera(port, jpos)        # camera pose in base
        Rm, t = T[:3, :3], T[:3, 3]
        Km = np.asarray(K, dtype=float)
        h, w = frame_bgr.shape[:2]

        def project(pts) -> np.ndarray:
            P = np.asarray(pts, dtype=float).reshape(-1, 3)
            cam = (P - t) @ Rm                           # rowwise R.T @ (p - t)
            keep = cam[:, 2] > 0.03                      # in front of the lens
            if not np.any(keep):
                return np.zeros((0, 2), dtype=int)
            uv = cam[keep] @ Km.T
            uv = uv[:, :2] / uv[:, 2:3]
            inside = ((uv[:, 0] > -w) & (uv[:, 0] < 2 * w)
                      & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h))
            return uv[inside].astype(int)

        for pts in paths.values():
            pix = project(pts)
            if len(pix) >= 2:
                cv2.polylines(frame_bgr, [pix.reshape(-1, 1, 2)],
                              False, (40, 220, 40), 2, cv2.LINE_AA)
            if len(pix):
                cv2.circle(frame_bgr, tuple(pix[-1]), 7, (0, 0, 255), -1)
        for cds in route_state["cands"].values():
            for p in project(cds):
                cv2.circle(frame_bgr, tuple(p), 3, (0, 255, 255), -1)

    # RGB numpy frame written by _cam_capture, read by _cam_bg_loop (GIL-safe).
    latest_frame: np.ndarray | None = None
    latest_frame_id: int = -1
    new_frame_event = threading.Event()

    def _cam_bg_loop():
        """Send camera background to viser clients. Event-driven, rate-limited to 25 Hz.
        Per-viewport padded array cache avoids redundant _pad_to_aspect work when multiple
        clients share the same viewport size."""
        _MIN_INTERVAL = 1.0 / 25.0
        # (frame_id, vp_w, vp_h) -> padded ndarray: shared across clients with same viewport
        viewport_cache: dict[tuple, np.ndarray] = {}
        client_last_sent: dict = {}  # client_id -> (frame_id, vp_w, vp_h)
        last_send_t = 0.0
        while not quit_event.is_set():
            new_frame_event.wait(timeout=0.1)
            new_frame_event.clear()
            now = time.monotonic()
            if now - last_send_t < _MIN_INTERVAL:
                continue
            last_send_t = now
            frame, fid = latest_frame, latest_frame_id
            if frame is None:
                continue
            active = server.get_clients()
            client_last_sent = {k: v for k, v in client_last_sent.items() if k in active}
            viewport_cache = {k: v for k, v in viewport_cache.items() if k[0] == fid}
            for cid, client in active.items():
                vp_w, vp_h = client.camera.image_width, client.camera.image_height
                if vp_w <= 0 or vp_h <= 0:
                    continue
                key = (fid, vp_w, vp_h)
                if client_last_sent.get(cid) == key:
                    continue
                if key not in viewport_cache:
                    viewport_cache[key] = _pad_to_aspect(frame, vp_w, vp_h)
                client.scene.set_background_image(viewport_cache[key], format="jpeg",
                                                  jpeg_quality=75)
                client_last_sent[cid] = key

    threading.Thread(target=_cam_bg_loop, daemon=True, name="cam_bg").start()

    if args.camera:
        def _cam_capture():
            nonlocal latest_frame, latest_frame_id
            cam_client = MultiStreamingCameraClient(
                args.robot_ip, args.n_cameras, args.camera_port)
            cam_client.initialize_all()
            if not cam_client.wait_all(timeout=6.0):
                print(f"[camera] ERROR — not all {args.n_cameras} cameras connected within 6s")
                return
            counter = 0
            try:
                while not quit_event.is_set():
                    cam_client.capture_all_parallel(timeout=0.1)
                    frames = []
                    for idx, b in enumerate(cam_client):
                        f = b.get_frame_buffer()
                        if f is None:
                            continue
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f
                        if route_state["paths"]:
                            f = f.copy()          # never draw on the client's buffer
                            try:
                                _draw_route_overlay(f, args.camera_port + idx)
                            except Exception:     # noqa: BLE001 — overlay must never kill capture
                                pass
                        frames.append(f)
                    if not frames:
                        continue
                    bgr = np.hstack(frames) if len(frames) > 1 else frames[0]
                    if args.camera_max_width > 0 and bgr.shape[1] > args.camera_max_width:
                        scale = args.camera_max_width / bgr.shape[1]
                        bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_LINEAR)
                    counter += 1
                    latest_frame = bgr[:, :, ::-1]
                    latest_frame_id = counter
                    new_frame_event.set()
            finally:
                cam_client.cleanup()

        threading.Thread(target=_cam_capture, daemon=True).start()

    # 20 Hz: fewer atomic batch sends per second vs 30 Hz, reduces browser render load.
    _FRAME_PERIOD = 1.0 / 20.0
    deadline = time.monotonic() + _FRAME_PERIOD
    last_data_count = -1
    last_det_frame_id: dict[int, int] = {port: -1 for port in camera_tag_detectors}
    # Hide target frames only after unseen by all cameras for this long.
    # Prevents flicker from single missed detections and multi-camera last-writer-wins conflicts.
    _TARGET_RETENTION_S = 0.5
    target_last_seen: dict[str, float] = {}
    retained_targets: dict = {}   # publisher-side retention (build_retained_targets)

    while not quit_event.is_set():
        flushed = False

        if telemetry_stats.last_data is not None and telemetry_stats.msg_count != last_data_count:
            last_data_count = telemetry_stats.msg_count
            viser_viz.update_from_packet(dict(telemetry_stats.last_data))
            flushed = True

        if route_sub.data_id != route_sub_id:
            route_sub_id = route_sub.data_id
            d = route_sub.data or {}
            if d.get("goals_only"):
                # Live MPC aim, one magenta dot per tracked hand. The green
                # route is a plan-time snapshot riding base_link and cannot
                # show lock; THESE dots are where the tracker is going right
                # now — nailed on the cube means locked.
                pts = list((d.get("goals") or {}).values())
                if pts:
                    garr = np.asarray(pts, dtype=np.float32)
                    route_cloud_handles["live"] = server.scene.add_point_cloud(
                        "/robot/base_link/route_live_goals", points=garr,
                        colors=np.tile(np.array([[255, 60, 200]],
                                                dtype=np.uint8),
                                       (len(garr), 1)),
                        point_size=0.016)
                    route_cloud_handles["live"].visible = True
                flushed = True
            else:
                route_state["paths"] = d.get("paths") or {}
                route_state["cands"] = d.get("cands") or {}
                pts = [p for path in route_state["paths"].values() for p in path]
                if pts:
                    arr = np.asarray(pts, dtype=np.float32)
                    route_cloud_handles["path"] = server.scene.add_point_cloud(
                        "/robot/base_link/route_preview", points=arr,
                        colors=np.tile(np.array([[40, 220, 40]], dtype=np.uint8),
                                       (len(arr), 1)),
                        point_size=0.006)
                    route_cloud_handles["path"].visible = True
                    cds = [p for c in route_state["cands"].values() for p in c]
                    if cds:
                        carr = np.asarray(cds, dtype=np.float32)
                        route_cloud_handles["cands"] = server.scene.add_point_cloud(
                            "/robot/base_link/route_cands", points=carr,
                            colors=np.tile(np.array([[255, 210, 0]],
                                                    dtype=np.uint8),
                                           (len(carr), 1)),
                            point_size=0.010)
                        route_cloud_handles["cands"].visible = True
                    print(f"[route] preview ({d.get('goal')}): {len(arr)} EE "
                          f"waypoints"
                          + (f", {len(cds)} grasp candidates" if cds else ""))
                else:
                    for hnd in route_cloud_handles.values():
                        hnd.visible = False
                flushed = True

        # atomic(): batch all handle updates into one WS window.
        # Without this, 56 individual SceneNodeUpdateMessages arrive at the browser
        # as separate WS frames → burst-then-repaint → discrete jumps.
        with server.atomic():
            viser_viz.apply_if_changed()

        for port, detector in camera_tag_detectors.items():
            snap = detector.get_latest()
            if snap is None or snap.frame_id == last_det_frame_id[port]:
                continue
            last_det_frame_id[port] = snap.frame_id
            flushed = True

            for name, pose in snap.body_poses.items():
                if (body := viser_body_handles.get(name)):
                    body.set_pose(position=pose.pos.tolist(), wxyz=pose.quat_wxyz.tolist())

            now = time.monotonic()
            for key, handle in target_frame_handles.items():
                pose = snap.target_poses.get(key)
                if pose is not None:
                    target_last_seen[key] = now
                    handle.visible = True
                    handle.position = pose.pos
                    handle.wxyz = pose.quat_wxyz
                elif now - target_last_seen.get(key, 0.0) > _TARGET_RETENTION_S:
                    handle.visible = False

        if flushed:
            if detection_publisher is not None:
                frame_id_by_port: dict[int, int] = {}
                snaps = []
                for port2, detector2 in camera_tag_detectors.items():
                    snap2 = detector2.get_latest()
                    if snap2 is None:
                        continue
                    frame_id_by_port[port2] = snap2.frame_id
                    snaps.append(snap2)
                targets: dict[str, MonitorTargetPose] = {
                    key2: MonitorTargetPose(
                        key=key2,
                        pos=tuple(pose2.pos.tolist()),
                        quat_wxyz=tuple(pose2.quat_wxyz.tolist()),
                        n_inliers=int(pose2.n_inliers),
                        camera_port=int(pose2.camera_port),
                        capture_stamp=float(pose2.capture_stamp),
                    )
                    for key2, pose2 in build_retained_targets(
                        retained_targets, snaps, time.monotonic()).items()
                }
                ee_data = {}
                if telemetry_stats.last_data is not None:
                    ee_data = telemetry_stats.last_data.get("ee") or {}
                detection_packet = MonitorDetectionPacket(
                    stamp=time.monotonic(),
                    frame_id_by_port=frame_id_by_port,
                    targets=targets,
                    ee=ee_data,
                )
                detection_publisher.publish(dataclasses.asdict(detection_packet))
            batched_scene.flush()

        deadline += _FRAME_PERIOD
        now = time.monotonic()
        sleep_s = deadline - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            deadline += _FRAME_PERIOD * int(-sleep_s / _FRAME_PERIOD)


@dataclasses.dataclass
class Args:
    """Unified humanoid monitor: telemetry + AprilTag rigid-body detection in a
    viser 3D view, publishing detected target poses for the operator and the
    reach tool; also the hand-eye calibration recorder."""
    # --- Robot connection ---
    robot_ip: str = _site.ROBOT_IP          # robot's IP address
    # --- Terminal display ---
    refresh_hz: float = 10.0                  # terminal stats print rate (Hz); ignored if not print_stats
    fields: list[str] | None = None           # telemetry fields to show; None = all
    print_stats: bool = False                  # print telemetry stats to terminal
    # --- Viser 3D GUI ---
    gui: bool = True                           # enable viser 3D visualization at :8080
    # --- AprilTag detection ---
    detect_tags: bool = True                   # run AprilTag rigid-body detection on camera streams
    publish_detections: bool = True            # publish MonitorDetectionPacket on detection_pub_url
    detection_pub_url: str = DETECTION_IPC_URL # NNG pub socket URL for downstream consumers
    route_preview_url: str = "ipc:///tmp/humanoid_route_preview.sock"
    # cuRobo route preview subscription (humanoid_curobo_reach publishes its
    # gated route's EE path here; drawn into the camera view + 3D scene)
    tag_camera_host: str = ""                  # camera host override; defaults to robot_ip if empty
    tag_camera_ports: list[int] = dataclasses.field(default_factory=lambda: [5555, 5556])
    # maps port → static transform name (CAMERA_TRANSFORMS in camera_tag_detector.py).
    # Legacy fixed-camera extrinsics; unused when tag_mounts covers the port — and the
    # default tag_mounts covers both ports, so the "head"/"chest" names here are inert
    # unless a port is removed from tag_mounts.
    tag_transforms: dict[int, str] = dataclasses.field(default_factory=lambda: {5555: "head", 5556: "chest"})
    # maps port → MJCF camera site the camera rides (tyro syntax: --tag-mounts 5555 cam_left_rgb 5556 cam_right_rgb).
    # For these ports the extrinsic is recomputed from live gimbal FK on every telemetry
    # packet. Default: both physical cameras ride the head gimbal, side by side —
    # confirmed mapping left = 5555 (cam_left_rgb), right = 5556 (cam_right_rgb);
    # the streamer's --devices order is what binds a physical camera to a port.
    tag_mounts: dict[int, str] = dataclasses.field(
        default_factory=lambda: {5555: "cam_left_rgb", 5556: "cam_right_rgb"})
    bodies: list[str] = dataclasses.field(default_factory=lambda: [
        "tag_cube_0",
        "tag_cube_1",
    ])                                         # visual_servoing body names to detect and track
    # --- Camera display in viser ---
    camera: bool = True                        # show stitched camera feed as viser background
    camera_port: int = 5555                    # first camera port for display stream
    n_cameras: int = 2                         # number of cameras to hstack for display
    camera_max_width: int = 0               # max pixel width of stitched display frame; 0 = no resize
    # --- Robot visualization ---

    # MJCF body names to hide in 3D view; e.g. ["ee_left", "ee_right", "wrist_3_L", "wrist_3_R"]
    hide_links: list[str] = dataclasses.field(default_factory=list)
    # --- Hand-eye calibration recording ---
    record_handeye: bool = False               # record tag+joint data for hand-eye calibration; saves calib_YYYYMMDD_HHMMSS_port<N>.npz
    handeye_objects: dict = dataclasses.field(default_factory=lambda: dict(_HANDEYE_OBJECTS))
    # maps visual_servoing body name → MJCF mounting link; edit _HANDEYE_OBJECTS above to change defaults
    handeye_delay: float = 10.0               # countdown (s) before recording starts; use to get into position
    handeye_min_inliers: int = 2              # min simultaneous tag faces visible for a valid detection (≥2 recommended; lower to 1 only if cube ROM is large)
    handeye_yaml: str = ""                    # path to calibrated camera_handeye.yaml; overrides default approximate transforms if calibration_valid=true


def _load_handeye_yaml_transform(yaml_path: str, cam_port: int,
                                  default_transform: "np.ndarray") -> "np.ndarray":
    """Load refined camera transform from handeye YAML. Returns default on any failure."""
    import yaml
    try:
        with open(yaml_path) as f:
            hy = yaml.safe_load(f)
    except Exception as e:
        print(f"[handeye] WARNING: cannot read {yaml_path}: {e}. Using default transform.")
        return default_transform

    if not hy.get("calibration_valid", False):
        reason = hy.get("failure_reason", "unknown")
        print(f"[handeye] WARNING: {yaml_path} has calibration_valid=false ({reason}). "
              f"Using default transform.")
        return default_transform

    cam_entry = hy.get("cameras", {}).get(str(cam_port))
    if cam_entry is None:
        print(f"[handeye] WARNING: port {cam_port} not in {yaml_path}. Using default transform.")
        return default_transform

    T = np.array(cam_entry["T_base_link_to_camera"], dtype=np.float64)
    print(f"[handeye] Loaded refined transform for port {cam_port} from {yaml_path}")
    return T


def _record_handeye_thread(
    detector: "CameraTagDetector",
    cam_port: int,
    telemetry_deque: deque,
    telemetry_lock: threading.Lock,
    handeye_objects: dict,
    min_inliers: int,
    joint_velocity_threshold: float,
    recording_quit_event: threading.Event,
    accumulator: dict,
):
    """Per-camera recording thread for hand-eye calibration data.

    Polls detector for new frames, interpolates joint_pos from telemetry, computes
    arm-joint velocity weights, and appends to accumulator.
    """
    obj_names = list(handeye_objects.keys())
    M = len(obj_names)

    frame_q: list = []
    frame_time: list = []
    frame_valid_telem: list = []
    frame_w_speed: list = []
    frame_interp_span_ms: list = []
    obj_pos: list = [[] for _ in range(M)]
    obj_quat: list = [[] for _ in range(M)]
    obj_n_inliers: list = [[] for _ in range(M)]
    obj_valid: list = [[] for _ in range(M)]

    last_frame_id = -1
    prev_joint_pos = None
    prev_capture_stamp = None

    while not recording_quit_event.is_set():
        snap = detector.get_latest()
        if snap is None or snap.frame_id == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = snap.frame_id

        with telemetry_lock:
            buf_copy = list(telemetry_deque)
        if not buf_copy:
            time.sleep(0.005)
            continue

        t_cam = snap.capture_stamp
        times = np.array([x[0] for x in buf_copy])
        i0 = int(np.searchsorted(times, t_cam)) - 1
        i1 = i0 + 1

        telem_valid = False
        span_s = 0.0
        if 0 <= i0 < i1 <= len(buf_copy) - 1:
            span_s = times[i1] - times[i0]
            if span_s < 0.100:
                alpha = (t_cam - times[i0]) / span_s
                synced_joint_pos = (1 - alpha) * buf_copy[i0][1] + alpha * buf_copy[i1][1]
                telem_valid = True
            else:
                synced_joint_pos = buf_copy[i0][1]
                span_s = 0.0
        else:
            nearest = min(buf_copy, key=lambda x: abs(x[0] - t_cam))
            synced_joint_pos = nearest[1]
            telem_valid = abs(nearest[0] - t_cam) < 0.020

        # Arm joint velocity (indices 13:27 = L-arm + R-arm in MJCF order), rad/s
        if prev_joint_pos is not None:
            joint_velocity_approx = (synced_joint_pos - prev_joint_pos) / max(t_cam - prev_capture_stamp, 1e-6)
            w_speed = 1.0 / (1.0 + (np.max(np.abs(joint_velocity_approx[13:27])) / joint_velocity_threshold) ** 2)
        else:
            joint_velocity_approx = np.zeros_like(synced_joint_pos)
            w_speed = 1.0
        hard_gate = w_speed >= 0.1
        prev_joint_pos, prev_capture_stamp = synced_joint_pos, t_cam

        frame_q.append(synced_joint_pos.copy())
        frame_time.append(t_cam)
        frame_valid_telem.append(telem_valid)
        frame_w_speed.append(float(w_speed))
        frame_interp_span_ms.append(float(span_s * 1000.0))

        for m, body_name in enumerate(obj_names):
            pose = snap.body_poses.get(body_name)
            if pose is not None:
                obj_pos[m].append(pose.pos.copy())
                obj_quat[m].append(pose.quat_wxyz.copy())
                obj_n_inliers[m].append(int(pose.n_inliers))
                obj_valid[m].append(
                    bool(telem_valid and hard_gate and pose.n_inliers >= min_inliers)
                )
            else:
                obj_pos[m].append([np.nan, np.nan, np.nan])
                obj_quat[m].append([np.nan, np.nan, np.nan, np.nan])
                obj_n_inliers[m].append(0)
                obj_valid[m].append(False)

    accumulator["frame_q"] = frame_q
    accumulator["frame_time"] = frame_time
    accumulator["frame_valid_telem"] = frame_valid_telem
    accumulator["frame_w_speed"] = frame_w_speed
    accumulator["frame_interp_span_ms"] = frame_interp_span_ms
    accumulator["obj_pos"] = obj_pos
    accumulator["obj_quat"] = obj_quat
    accumulator["obj_n_inliers"] = obj_n_inliers
    accumulator["obj_valid"] = obj_valid


def _save_handeye_npz(
    cam_port: int,
    cam_transform: "np.ndarray",
    detector: "CameraTagDetector",
    handeye_objects: dict,
    accumulator: dict,
    out_dir: Path,
    prefix: str,
):
    """Save accumulated hand-eye recording to .npz file."""
    obj_names = list(handeye_objects.keys())
    obj_links = list(handeye_objects.values())
    M = len(obj_names)

    frame_q = accumulator["frame_q"]
    frame_time = accumulator["frame_time"]
    obj_pos = accumulator["obj_pos"]
    obj_quat = accumulator["obj_quat"]
    obj_n_inliers = accumulator["obj_n_inliers"]
    obj_valid = accumulator["obj_valid"]

    if not frame_q:
        print(f"[handeye] WARNING: no frames recorded for port {cam_port}. Skipping save.")
        return

    N = len(frame_q)
    out_path = out_dir / f"{prefix}_port{cam_port}.npz"
    np.savez(
        out_path,
        q_pos=np.array(frame_q, dtype=np.float64),                          # (N,27)
        time=np.array(frame_time, dtype=np.float64),                         # (N,)
        valid_telem=np.array(accumulator["frame_valid_telem"], dtype=bool),  # (N,)
        w_speed=np.array(accumulator["frame_w_speed"], dtype=np.float32),    # (N,)
        interp_span_ms=np.array(accumulator["frame_interp_span_ms"], dtype=np.float32),  # (N,)
        cam_port=cam_port,
        cam_K=detector._detector.K,                                           # (3,3)
        cam_transform_approx=cam_transform,                                   # (4,4)
        obj_names=np.array(obj_names),                                        # (M,)
        obj_links=np.array(obj_links),                                        # (M,)
        obj_pos=np.array(
            [obj_pos[m] for m in range(M)], dtype=np.float64
        ).transpose(1, 0, 2),            # (N,M,3)
        obj_quat_wxyz=np.array(
            [obj_quat[m] for m in range(M)], dtype=np.float64
        ).transpose(1, 0, 2),            # (N,M,4)
        obj_n_inliers=np.array(
            [obj_n_inliers[m] for m in range(M)], dtype=np.int32
        ).T,                             # (N,M)
        obj_valid=np.array(
            [obj_valid[m] for m in range(M)], dtype=bool
        ).T,                             # (N,M)
    )
    valid_count = sum(any(obj_valid[m][i] for m in range(M)) for i in range(N))
    print(f"[handeye] Saved {N} frames ({valid_count} with ≥1 valid detection) → {out_path}")


def main():
    args = tyro.cli(Args)

    telemetry_stats = TelemetryChannelStats()
    telemetry_subscriber = NNGSubscriber(f"tcp://{args.robot_ip}:{TELEMETRY_PORT}")
    telemetry_subscriber.start()

    quit_event = threading.Event()

    # Telemetry ring buffer for hand-eye recording (5 s @ 20 Hz).
    # Written by _monitor_telem; read by _record_handeye_thread.
    telemetry_deque: deque = deque(maxlen=100)
    telemetry_lock = threading.Lock()

    detection_publisher: NNGPublisher | None = None
    if args.publish_detections:
        if args.detection_pub_url.startswith("ipc://"):
            Path(args.detection_pub_url[len("ipc://"):]).unlink(missing_ok=True)
        detection_publisher = NNGPublisher(args.detection_pub_url)
        print(f"[detection] publishing detected targets on {args.detection_pub_url}")

    camera_tag_detectors: dict[int, CameraTagDetector] = {}
    cam_transforms: dict[int, np.ndarray] = {}  # exact transform used per port (gimbal: init pose)
    gimbal_fk: GimbalCameraFK | None = None
    gimbal_freeze = threading.Event()  # set during --record-handeye (see below)
    active_mounts: dict[int, str] = {}
    if args.detect_tags and args.tag_mounts:
        # Only mounts for ports actually being started matter; extra DEFAULT entries
        # (e.g. --tag-camera-ports 5556 with the two-port default mounts) are dropped
        # with a note instead of crashing the whole monitor.
        active_mounts = {p: s for p, s in args.tag_mounts.items() if p in args.tag_camera_ports}
        dropped = set(args.tag_mounts) - set(active_mounts)
        if dropped:
            print(f"[gimbal] note: ignoring --tag-mounts for inactive ports {sorted(dropped)}")
        if active_mounts:
            gimbal_fk = GimbalCameraFK(active_mounts)
        if args.handeye_yaml and active_mounts:
            print(f"[handeye] WARNING: --handeye-yaml is only applied to fixed (base_link) "
                  f"cameras; gimbal-mounted ports {sorted(active_mounts)} use nominal MJCF FK")
    if args.detect_tags:
        camera_host = args.tag_camera_host or args.robot_ip
        for port in args.tag_camera_ports:
            try:
                if port in active_mounts:
                    # Gimbal-mounted camera: extrinsic = live FK, refreshed per
                    # telemetry packet in _monitor_telem. Init at all-joints-zero.
                    cam_transform = gimbal_fk.base_to_camera(port, None)
                else:
                    # Legacy fixed camera: static hard-coded transform.
                    if port not in args.tag_transforms:
                        raise ValueError(
                            f"Missing tag transform mapping for port {port}; "
                            f"provide --tag-transforms or --tag-mounts for all --tag-camera-ports"
                        )
                    transform_name = args.tag_transforms[port]
                    if transform_name not in CAMERA_TRANSFORMS:
                        raise ValueError(
                            f"Unknown transform '{transform_name}' for port {port}, "
                            f"expected one of: {list(CAMERA_TRANSFORMS)}"
                        )
                    cam_transform = CAMERA_TRANSFORMS[transform_name].copy()
                    if args.handeye_yaml:
                        cam_transform = _load_handeye_yaml_transform(
                            args.handeye_yaml, port, cam_transform
                        )
                detector = CameraTagDetector(
                    camera_host=camera_host,
                    camera_port=port,
                    camera_transform=cam_transform,
                    body_specs=args.bodies,
                    quit_flag=quit_event,
                    enable_vis=False,
                )
                detector.start()
                camera_tag_detectors[port] = detector
                cam_transforms[port] = cam_transform
            except Exception as e:
                print(f"[detection] failed to start camera {port}: {e}")
                continue

    if args.detect_tags and not camera_tag_detectors:
        print("[detection] disabled after startup failures; robot visualization continues.")

    if args.gui or args.camera:
        threading.Thread(
            target=_viser_thread,
            args=(telemetry_stats, camera_tag_detectors, args, quit_event, detection_publisher),
            daemon=True
        ).start()

    def _monitor_telem():
        last_id = -1
        while not quit_event.is_set():
            if telemetry_subscriber.data_id != last_id and telemetry_subscriber.data is not None:
                last_id = telemetry_subscriber.data_id
                telemetry_stats.update(telemetry_subscriber.data)
                # Feed ring buffer for hand-eye recording (arm joints only needed, but store all)
                jpos = telemetry_subscriber.data.get("joint_pos")
                if jpos is not None:
                    jarr = np.asarray(jpos, dtype=np.float64)
                    with telemetry_lock:
                        telemetry_deque.append((time.monotonic(), jarr.copy()))
                    # Gimbal-mounted cameras: refresh the detector extrinsic from
                    # live FK (atomic ref swap; detector reads it next frame).
                    # Frozen during --record-handeye: the hand-eye solver assumes ONE
                    # static transform per session (it inverts cam_transform_approx),
                    # so live swaps would silently corrupt the recording.
                    if gimbal_fk is not None and not gimbal_freeze.is_set():
                        for gport in gimbal_fk.site_ids:
                            det = camera_tag_detectors.get(gport)
                            if det is not None:
                                det.camera_transform = gimbal_fk.base_to_camera(gport, jarr)
            quit_event.wait(timeout=0.05)

    threading.Thread(target=_monitor_telem, daemon=True).start()

    # -- Hand-eye recording mode -----------------------------------------------
    if args.record_handeye:
        import datetime
        handeye_prefix = datetime.datetime.now().strftime("calib_%Y%m%d_%H%M%S")
        if not camera_tag_detectors:
            print("[handeye] ERROR: no detectors running — cannot record. Exiting.")
            quit_event.set()
            telemetry_subscriber.stop()
            return

        _DQ_THRESH = 0.5  # rad/s arm joint inf-norm threshold for velocity weighting

        # Add handeye object bodies to detector body_specs if not already tracked
        handeye_body_names = set(args.handeye_objects.keys())
        for port, detector in camera_tag_detectors.items():
            existing = {spec.body_name for spec in detector.specs}
            missing = handeye_body_names - existing
            if missing:
                print(f"[handeye] WARNING: port {port} detectors do not track {missing}. "
                      f"Add them to --bodies. Detected body names: {existing}")

        # Delay start with countdown so operator can get into position
        if args.handeye_delay > 0:
            print(f"[handeye] Starting in {args.handeye_delay:.0f}s — get into position. Press Ctrl+C to abort.")
            try:
                for remaining in range(int(args.handeye_delay), 0, -1):
                    sys.stdout.write(f"\r[handeye] Starting in {remaining}s...  ")
                    sys.stdout.flush()
                    time.sleep(1.0)
                sys.stdout.write("\r[handeye] Recording — press Ctrl+C to stop and save.\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                print("\n[handeye] Aborted before recording started.")
                quit_event.set()
                telemetry_subscriber.stop()
                for detector in camera_tag_detectors.values():
                    detector.stop()
                if detection_publisher is not None:
                    detection_publisher.close()
                return

        # Hand-eye recording needs ONE static extrinsic per session: freeze the
        # gimbal FK refresh and pin detectors back to the transform that will be
        # saved as cam_transform_approx, so record-time == solve-time exactly.
        if gimbal_fk is not None:
            gimbal_freeze.set()
            time.sleep(0.3)  # drain detector frames computed with a pre-freeze transform
            for gport in gimbal_fk.site_ids:
                det = camera_tag_detectors.get(gport)
                if det is not None:
                    det.camera_transform = cam_transforms[gport]
            print("[handeye] gimbal FK refresh FROZEN for recording (static extrinsic); "
                  "keep the gimbal at neutral during the session")

        recording_quit_event = threading.Event()  # separate flag so recording stops but telem keeps running
        handeye_recordings: dict[int, dict] = {port: {} for port in camera_tag_detectors}
        recording_threads = []
        for port, detector in camera_tag_detectors.items():
            t = threading.Thread(
                target=_record_handeye_thread,
                kwargs=dict(
                    detector=detector,
                    cam_port=port,
                    telemetry_deque=telemetry_deque,
                    telemetry_lock=telemetry_lock,
                    handeye_objects=args.handeye_objects,
                    min_inliers=args.handeye_min_inliers,
                    joint_velocity_threshold=_DQ_THRESH,
                    recording_quit_event=recording_quit_event,
                    accumulator=handeye_recordings[port],
                ),
                daemon=True,
                name=f"handeye_rec_{port}",
            )
            recording_threads.append((t, port))
            t.start()
        print(f"[handeye] Recording on ports {list(camera_tag_detectors)} "
              f"→ calibration/{handeye_prefix}_port*.npz  (Ctrl+C to stop and save)")
        try:
            while True:
                time.sleep(1.0)
                counts = []
                for port in camera_tag_detectors:
                    acc = handeye_recordings[port]
                    n = len(acc.get("frame_q", []))
                    obj_valid = acc.get("obj_valid")  # list of M lists
                    if obj_valid and n > 0:
                        n_valid = sum(any(obj_valid[m][i] for m in range(len(obj_valid))) for i in range(n))
                    else:
                        n_valid = 0
                    counts.append(f"port{port}: {n}fr {n_valid}valid")
                sys.stdout.write("\r[handeye] " + "  |  ".join(counts) + "   ")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n[handeye] Stopping — saving data...")

        recording_quit_event.set()
        for t, port in recording_threads:
            t.join(timeout=5.0)
            if t.is_alive():
                print(f"[handeye] WARNING: recording thread for port {port} did not stop — skipping save.")
                handeye_recordings[port]["skip"] = True

        out_dir = Path(__file__).resolve().parent / "calibration"
        out_dir.mkdir(exist_ok=True)
        for port, detector in camera_tag_detectors.items():
            if not handeye_recordings[port].get("skip") and "frame_q" in handeye_recordings[port]:
                _save_handeye_npz(
                    cam_port=port,
                    cam_transform=cam_transforms[port],
                    detector=detector,
                    handeye_objects=args.handeye_objects,
                    accumulator=handeye_recordings[port],
                    out_dir=out_dir,
                    prefix=handeye_prefix,
                )
        print("[handeye] Recording complete.")

        telemetry_subscriber.stop()
        for detector in camera_tag_detectors.values():
            detector.stop()
        if detection_publisher is not None:
            detection_publisher.close()
        return
    # -- End hand-eye recording mode -------------------------------------------

    refresh_period = 1.0 / max(args.refresh_hz, 0.5)
    try:
        while True:
            if args.print_stats:
                sys.stdout.write(_ANSI_CLEAR + _render_terminal_stats(telemetry_stats, camera_tag_detectors, args.fields) + "\n")
                sys.stdout.flush()
            time.sleep(refresh_period)
    except KeyboardInterrupt:
        pass
    finally:
        quit_event.set()
        telemetry_subscriber.stop()
        for detector in camera_tag_detectors.values():
            detector.stop()
        if detection_publisher is not None:
            detection_publisher.close()


if __name__ == "__main__":
    main()
