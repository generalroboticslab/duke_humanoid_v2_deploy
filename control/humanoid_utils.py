"""Shared helpers: the async colour logger (`make_async_logger`), `CircularBuffer`,
MJCF joint-info loading (`get_joint_info_from_mjcf`), numpy transform helpers
and the viser robot view (`HumanoidViserViz`). Imports torch, mujoco, viser
and trimesh at module level.
"""
import logging
import logging.handlers
import os
import queue as _queue
import threading
import time
from typing import NamedTuple

import torch
import numpy as np
import viser
import mujoco
import trimesh

# ---------------------------------------------------------------------------
# Async colored logger
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {
    logging.DEBUG:    "\033[37m",    # grey
    logging.WARNING:  "\033[93m",    # yellow
    logging.ERROR:    "\033[91m",    # red
    logging.CRITICAL: "\033[91;1m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Prepend ANSI color based on level; reset after."""
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{_RESET}" if color else msg


def make_async_logger(name: str, fmt: str = "%(message)s") -> logging.Logger:
    """Non-blocking logger: log calls enqueue in O(1); a daemon thread writes to stderr.

    Usage:
        log = make_async_logger("my_module")
        log.warning("joint %s hit limit %.3f", name, val)  # never blocks control loop

    Shutdown (optional, for clean flush):
        log._listener.stop()
    """
    log = logging.getLogger(name)
    if log.handlers:
        return log  # idempotent — safe to call multiple times

    q = _queue.SimpleQueue()                                # lock-free enqueue
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter(fmt))
    listener = logging.handlers.QueueListener(q, handler)  # daemon thread
    listener.start()

    log.addHandler(logging.handlers.QueueHandler(q))
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log._listener = listener  # type: ignore[attr-defined]  # expose for shutdown
    return log


class CircularBuffer:
    """Circular buffer for observation history."""

    def __init__(self, max_len: int, batch_size: int, obs_dim: int, device: torch.device):
        self._max_len = max_len
        self._batch_size = batch_size
        self._pointer = 0
        self._initialized = False
        self._buffer = torch.zeros(batch_size, max_len, obs_dim, device=device)

    def reset(self):
        self._initialized = False
        self._pointer = 0
        self._buffer.zero_()

    def append(self, data: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """Append data to buffer and return history (oldest to newest)."""
        # Ensure data has batch dimension
        if data.ndim == 1:
            data = data.unsqueeze(0)

        if not self._initialized:
            # Backfill all slots with initial observation
            self._buffer[:] = data.unsqueeze(1)
            self._initialized = True
        else:
            self._buffer[:, self._pointer] = data
            self._pointer = (self._pointer + 1) % self._max_len

        # Roll to get oldest-to-newest order
        history = torch.roll(self._buffer, -self._pointer, dims=1)
        return history.reshape(self._batch_size, -1) if flatten else history


def get_joint_info_from_mjcf(mjcf_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract actuated joint names and limits from MJCF model (excludes freejoint).

    Returns:
        joint_names: List of joint names
        joint_lower: Array of lower joint limits (radians)
        joint_upper: Array of upper joint limits (radians)
    """
    import xml.etree.ElementTree as ET
    # 2026-08-22: fail with the remedy, not a bare [Errno 2]. Every process
    # that loads the deploy model (humanoid_base at import, the monitor, the
    # debug receiver) comes through here first, so this is the one place a
    # mis-pointed checkout can be named for what it is.
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(
            f"deploy model not found at {mjcf_path} — the control stack reads "
            f"it from legged_env_v2 (mj_envs/deploy/runs/<task>/robot.xml); "
            f"set HUMANOID_LEGGED_ENV_ROOT in the environment, or "
            f"LEGGED_ENV_ROOT in control/site_local.py, to that checkout "
            f"(README 'Install').")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    # Find all <joint> elements (not <freejoint>) and extract names and limits
    joint_names = []
    joint_lower = []
    joint_upper = []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name:
            # Skip the floating base. mjlab/MjSpec emits it as <joint type="free"/>
            # (a rangeless joint) rather than <freejoint>; both mean the 6-DOF root
            # and must not appear in the actuated-joint list.
            if joint.get("type") == "free":
                continue
            range_str = joint.get("range")
            if range_str is None:
                continue
            joint_names.append(name)
            # Parse range attribute: "lower upper"
            lower, upper = map(float, range_str.split())
            joint_lower.append(lower)
            joint_upper.append(upper)
    return joint_names, np.array(joint_lower, dtype=np.float32), np.array(joint_upper, dtype=np.float32)


def invert_transform_np(T):
    """invert transform (numpy)"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def make_transform_np(pos, quat_wxyz):
    """Build 4x4 SE(3) transform from xyz position and wxyz quaternion."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def transform_to_pos_quat_np(T):
    """Extract xyz position and wxyz quaternion from a 4x4 SE(3) transform (Shepperd's method)."""
    pos = T[:3, 3].copy()
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return pos, np.array([w, x, y, z])


def rot_mats_to_quats_wxyz_np(R: np.ndarray) -> np.ndarray:
    """Vectorized 3x3 rotation matrices -> quaternions (wxyz), shape (N,3,3) -> (N,4)."""
    R = np.asarray(R)
    n = R.shape[0]

    m00 = R[:, 0, 0]
    m11 = R[:, 1, 1]
    m22 = R[:, 2, 2]
    trace = m00 + m11 + m22

    q = np.empty((n, 4), dtype=R.dtype)

    mask_trace = trace > 0.0
    if np.any(mask_trace):
        s = 2.0 * np.sqrt(trace[mask_trace] + 1.0)
        q[mask_trace, 0] = 0.25 * s
        q[mask_trace, 1] = (R[mask_trace, 2, 1] - R[mask_trace, 1, 2]) / s
        q[mask_trace, 2] = (R[mask_trace, 0, 2] - R[mask_trace, 2, 0]) / s
        q[mask_trace, 3] = (R[mask_trace, 1, 0] - R[mask_trace, 0, 1]) / s

    mask_x = (~mask_trace) & (m00 > m11) & (m00 > m22)
    if np.any(mask_x):
        s = 2.0 * np.sqrt(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x])
        q[mask_x, 0] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / s
        q[mask_x, 3] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / s

    mask_y = (~mask_trace) & (~mask_x) & (m11 > m22)
    if np.any(mask_y):
        s = 2.0 * np.sqrt(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y])
        q[mask_y, 0] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) / s
        q[mask_y, 1] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / s

    mask_z = (~mask_trace) & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = 2.0 * np.sqrt(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z])
        q[mask_z, 0] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) / s
        q[mask_z, 1] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / s
        q[mask_z, 2] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / s
        q[mask_z, 3] = 0.25 * s

    return q



class _VizState(NamedTuple):
    """Snapshot passed to the background viz thread each control step."""
    vicon_pos:     np.ndarray | None  # (7,) pos+quat of first Vicon marker
    imu_quat_wxyz: np.ndarray         # (4,) wxyz
    wb_pos:        np.ndarray | None  # (3,) world-frame base position
    wb_quat_wxyz:  np.ndarray | None  # (4,) wxyz
    joint_pos:     np.ndarray | None  # (N_joints,)


VizState = _VizState


def viz_state_from_packet(packet: dict) -> VizState:
    """Convert a wire packet from humanoid_real_env viz publisher into VizState."""
    _a = lambda x: None if x is None else np.asarray(x)
    return VizState(
        vicon_pos=_a(packet.get("vicon_pos")),
        imu_quat_wxyz=np.asarray(packet["imu_quat_wxyz"]),
        wb_pos=_a(packet.get("wb_pos")),
        wb_quat_wxyz=_a(packet.get("wb_quat_wxyz")),
        joint_pos=_a(packet.get("joint_pos")),
    )


# Camera mount transforms in base_link frame (FLU convention).
# Keep names aligned with humanoid_monitor semantics.
T_BASE_HEAD_CAM = np.array(
    [[0.0, -0.7071, 0.7071, 0.06684],
     [-1.0, 0.0, 0.0, 0.0175],
     [0.0, -0.7071, -0.7071, 0.42003],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)

T_BASE_CHEST_CAM = np.array(
    [[0.0, 0.0, 1.0, 0.07275],
     [-1.0, 0.0, 0.0, 0.03250],
     [0.0, -1.0, 0.0, 0.36500],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)

CAMERA_FRAME_SPECS = (
    ("camera_head", T_BASE_HEAD_CAM),
    ("camera_chest", T_BASE_CHEST_CAM),
)


class HumanoidViserViz:
    """Live viser visualization for HumanoidRealEnv: Vicon pose, IMU orientation,
    and optionally a calibrated base_link frame with full robot mesh.

    Usage::
        viz = HumanoidViserViz(mjcf_path, joint_names, show_robot=True)
        viz.update(vicon_result=..., imu_quat_wxyz=..., wb_pos=..., wb_quat_wxyz=..., joint_pos=...)

    hidden_links: set of names to hide at startup. Two namespaces:
      - EE axis frames:  "ee_left", "ee_right"  (FK XYZ-arrow overlays, NOT MJCF bodies)
      - MJCF body meshes: exact body name from MJCF, e.g. "wrist_3_L", "wrist_3_R"
      Applied synchronously for EE frames; applied per-body inside _create_body_meshes (bg thread).
      Unknown names silently ignored.
    """

    def __init__(self, mjcf_path: str, joint_names: list,
                 show_robot: bool = False,
                 host: str = "0.0.0.0", port: int = 8080,
                 server=None,
                 enable_internal_loop: bool = True,
                 hidden_links: set[str] | None = None):
        if server is None:
            server = viser.ViserServer(host=host, port=port)
        self._server = server

        server.scene.add_grid("/ground", width=4, height=4, cell_size=0.25, plane="xy", position=(0.0, 0.0, -0.58))

        self._vicon_frame = server.scene.add_frame("/vicon", axes_length=0.15, axes_radius=0.001)
        # IMU frame is top-level (not child of vicon) so rotation stays independent
        self._imu_frame = server.scene.add_frame("/imu", axes_length=0.15, axes_radius=0.001)

        self._base_link_frame = None
        self._robot_frame = None
        self._ee_frames: list = []          # [left_handle, right_handle], populated when show_robot=True
        self._ee_ids: np.ndarray | None = None  # shape (2,) body ids, set after model loads
        self._mj_model = None
        self._mj_data = None
        self._joint_names = joint_names
        self._mesh_handles_by_body = {}  # body_id -> viser batched mesh handle
        self._mesh_items: list[tuple[int, object]] = []
        self._body_ids_arr = np.array([], dtype=np.int32)
        self._site_handles: list = []
        self._mujoco = None
        # True only after _load_mjcf_bg fully completes (model + data + mesh handles ready).
        # Guards _apply_state against the race where _mj_model is set but _mj_data is still None.
        self._mj_ready: bool = False
        self._hidden_links: set[str] = hidden_links or set()
        self._log = make_async_logger("humanoid_viz")
        self._joint_dim_warned = False  # warn once on telemetry/model joint-count mismatch

        if show_robot:
            self._robot_frame = server.scene.add_frame("/robot", show_axes=False)
            self._base_link_frame = server.scene.add_frame(
                "/robot/base_link", axes_length=0.15, axes_radius=0.001)

            # Camera frames are rigidly attached to /robot/base_link.
            for frame_name, T_base_cam in CAMERA_FRAME_SPECS:
                handle = server.scene.add_frame(
                    f"/robot/base_link/{frame_name}",
                    axes_length=0.05,
                    axes_radius=0.001,
                )
                cam_pos, cam_quat_wxyz = transform_to_pos_quat_np(T_base_cam)
                handle.position = cam_pos
                handle.wxyz = cam_quat_wxyz

            # End-effector frames — updated from FK once model loads (robot-local)
            self._ee_frames = [
                server.scene.add_frame("/robot/ee_left",  axes_length=0.10, axes_radius=0.001),
                server.scene.add_frame("/robot/ee_right", axes_length=0.10, axes_radius=0.001),
            ]
            for _name, _frame in (("ee_left", self._ee_frames[0]), ("ee_right", self._ee_frames[1])):
                if _name in self._hidden_links:
                    _frame.visible = False

            # Load MJCF model in background so robot init isn't blocked

            self._mujoco = mujoco
            _log = make_async_logger("humanoid_viz")
            def _load_mjcf_bg():
                self._mj_model = self._mujoco.MjModel.from_xml_path(mjcf_path)
                self._mj_data = self._mujoco.MjData(self._mj_model)
                # Initialize to default pose (all joints at 0)
                self._mujoco.mj_kinematics(self._mj_model, self._mj_data)
                # Create merged mesh handles per body
                self._create_body_meshes(server)
                # Resolve end-effector body IDs (-1 means not found in model)
                ids = [self._mujoco.mj_name2id(self._mj_model, self._mujoco.mjtObj.mjOBJ_BODY, n)
                       for n in ("end_effector_L", "end_effector_R")]
                if all(i >= 0 for i in ids):
                    self._ee_ids = np.array(ids, dtype=np.int32)
                else:
                    _log.warning(f"[VIZ] EE bodies not found {ids}; EE frames disabled")
                self._mj_ready = True  # must be set after _create_body_meshes so _mesh_items is populated
                _log.info(f"[VIZ] MJCF ready: {len(joint_names)} joints, {len(self._mesh_handles_by_body)} bodies")
            threading.Thread(target=_load_mjcf_bg, daemon=True).start()
            _log.info("[VIZ] MJCF loading in background...")

        # Shared state for background viz thread; None means no data yet
        self._viz_state: _VizState | None = None
        self._viz_hz = 50.0
        # Version counter: bumped on every submit_state(); lets callers skip redundant apply.
        self._viz_state_version: int = 0
        self._applied_version: int = -1
        # FK dirty-check: skip mj_kinematics when joint_pos bytes unchanged
        self._last_joint_bytes: bytes | None = None

        if enable_internal_loop:
            threading.Thread(target=self._viz_loop, daemon=True, name="humanoid_viz").start()

    def _create_body_meshes(self, server) -> None:
        """Create merged mesh handles for each body (mjlab-style approach)."""
        
        
        try:
            import viser.transforms as vtf
        except ImportError:
            # Older viser versions
            import viser as vtf

        _log = make_async_logger("humanoid_viz")

        # Group visual geoms by body
        body_geoms = {}
        for geom_id in range(self._mj_model.ngeom):
            geom_group = self._mj_model.geom_group[geom_id]
            # In this MJCF: group 2 = visual, group 3 = collision
            if geom_group != 2:
                continue

            body_id = self._mj_model.geom_bodyid[geom_id]
            if body_id not in body_geoms:
                body_geoms[body_id] = []
            body_geoms[body_id].append(geom_id)

        # For each body, merge all its geoms into one trimesh
        for body_id, geom_ids in body_geoms.items():
            body_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

            try:
                # Merge all geoms for this body
                merged_mesh = self._merge_geoms(geom_ids)

                # Debug: Check mesh validity
                if len(merged_mesh.vertices) == 0:
                    _log.warning(f"[VIZ] Body {body_name} has empty merged mesh, skipping")
                    continue

                # Get initial body pose from MuJoCo
                body_pos = self._mj_data.xpos[body_id]
                body_mat = self._mj_data.xmat[body_id].reshape(3, 3)
                body_quat = vtf.SO3.from_matrix(body_mat).wxyz

                # Create mesh handle with initial pose
                handle = server.scene.add_mesh_trimesh(
                    f"/robot/{body_name}",
                    merged_mesh,
                    position=tuple(body_pos),
                    wxyz=tuple(body_quat),
                )
                if body_name in self._hidden_links:
                    handle.visible = False

                self._mesh_handles_by_body[body_id] = handle
                # _log.info(f"[VIZ] Created mesh for body {body_name}: {len(geom_ids)} geoms, {len(merged_mesh.vertices)} verts, pos={body_pos}, quat={body_quat}")
            except Exception as e:
                _log.error(f"[VIZ] Failed to create mesh for body {body_name}: {e}")

        self._mesh_items = list(self._mesh_handles_by_body.items())
        if self._mesh_items:
            self._body_ids_arr = np.fromiter((body_id for body_id, _ in self._mesh_items), dtype=np.int32)

        for site_id in range(self._mj_model.nsite):
            site_name = (mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, site_id)
                         or f"site_{site_id}")
            body_id = int(self._mj_model.site_bodyid[site_id])
            body_name = (mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                         or f"body_{body_id}")
            self._site_handles.append(server.scene.add_frame(
                f"/robot/{body_name}/site_{site_name}",
                axes_length=0.02,
                axes_radius=0.001,
                position=tuple(self._mj_model.site_pos[site_id]),
                wxyz=tuple(self._mj_model.site_quat[site_id]),
            ))

    def _load_mujoco_mesh(self, geom_id: int) -> "trimesh.Trimesh":
        """Load a MuJoCo mesh geom as a plain geometry (vertices + faces only).

        Textures and UV data are intentionally skipped: _merge_geoms always overwrites
        the visual with a flat ColorVisuals, so loading textures here is pure waste
        (PIL image creation, vertex duplication, PBRMaterial allocation).
        """
        mesh_id = self._mj_model.geom_dataid[geom_id]
        if mesh_id < 0:
            return None

        vert_start = int(self._mj_model.mesh_vertadr[mesh_id])
        vert_num   = int(self._mj_model.mesh_vertnum[mesh_id])
        face_start = int(self._mj_model.mesh_faceadr[mesh_id])
        face_num   = int(self._mj_model.mesh_facenum[mesh_id])

        vertices = self._mj_model.mesh_vert[vert_start:vert_start + vert_num]
        faces    = self._mj_model.mesh_face[face_start:face_start + face_num]
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def _merge_geoms(self, geom_ids: list[int]) -> "trimesh.Trimesh":
        """Merge multiple geoms into a single trimesh (mjlab-style).

        Each geom's local pose is baked into the mesh vertices.
        """
        
        
        import viser.transforms as vtf

        meshes = []
        for geom_id in geom_ids:
            geom_type = self._mj_model.geom_type[geom_id]
            geom_mesh = None

            # Create primitive or load mesh
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                size = self._mj_model.geom_size[geom_id]
                geom_mesh = trimesh.creation.box(extents=size * 2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = self._mj_model.geom_size[geom_id][0]
                geom_mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = self._mj_model.geom_size[geom_id][0]
                height = self._mj_model.geom_size[geom_id][1] * 2
                geom_mesh = trimesh.creation.capsule(radius=radius, height=height)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                radius = self._mj_model.geom_size[geom_id][0]
                height = self._mj_model.geom_size[geom_id][1] * 2
                geom_mesh = trimesh.creation.cylinder(radius=radius, height=height)
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Use mjlab-style mesh loading with proper texture support
                geom_mesh = self._load_mujoco_mesh(geom_id)

            if geom_mesh is None:
                continue

            # Apply geom's local transform to the mesh vertices
            pos = self._mj_model.geom_pos[geom_id]
            quat = self._mj_model.geom_quat[geom_id]  # (w,x,y,z)
            transform = np.eye(4)
            transform[:3, :3] = vtf.SO3(quat).as_matrix()
            transform[:3, 3] = pos
            geom_mesh.apply_transform(transform)

            # Apply color
            rgba = self._mj_model.geom_rgba[geom_id]
            geom_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=geom_mesh,
                vertex_colors=(rgba * 255).astype(np.uint8)
            )

            meshes.append(geom_mesh)

        if len(meshes) == 0:
            # Return empty mesh
            return trimesh.Trimesh()
        elif len(meshes) == 1:
            return meshes[0]
        else:
            return trimesh.util.concatenate(meshes)

    def _viz_loop(self) -> None:
        """Background thread: apply latest state to viser. Never blocks caller."""
        dt = 1.0 / self._viz_hz
        while True:
            t0 = time.monotonic()
            self.apply_if_changed()
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _apply_state(self, s: _VizState) -> None:
        """Do the actual viser scene updates (called from background thread)."""
        if s.vicon_pos is not None:
            self._vicon_frame.position = s.vicon_pos[:3]
            self._vicon_frame.wxyz     = s.vicon_pos[3:7]
            self._imu_frame.position   = s.vicon_pos[:3]  # position-only attachment
        self._imu_frame.wxyz = s.imu_quat_wxyz
        if self._base_link_frame is not None and s.wb_pos is not None:
            self._base_link_frame.position = s.wb_pos
            self._base_link_frame.wxyz     = s.wb_quat_wxyz

        # Camera frames are children of /robot/base_link with fixed local extrinsics.
        # No per-tick world update needed here.
        # Update MuJoCo model joints whenever available; use vicon pose or fall back to IMU
        if self._mj_ready and s.joint_pos is not None:
            if s.wb_pos is not None:
                self._robot_frame.position = s.wb_pos
                self._robot_frame.wxyz     = s.wb_quat_wxyz
            else:
                self._robot_frame.wxyz = s.imu_quat_wxyz

            # Only re-solve FK and update mesh handles when joint positions actually changed.
            joint_bytes = s.joint_pos.tobytes()
            if joint_bytes != self._last_joint_bytes:
                self._last_joint_bytes = joint_bytes

                # Update MuJoCo joint positions (skip freejoint at qpos[0:7]).
                # Tolerate a joint-count mismatch (e.g. 27-joint telemetry from an
                # older recording vs the 31-joint cam-grafted model, or vice versa)
                # by copying the common prefix — the first 27 joints are verified
                # identical between humanoid_v21.xml and the deploy robot.xml.
                qpos_offset = 7  # freejoint = 3 pos + 4 quat
                n_model = len(self._joint_names)
                n_recv = len(s.joint_pos)
                if n_recv != n_model and not self._joint_dim_warned:
                    self._joint_dim_warned = True
                    self._log.warning(
                        f"[VIZ] telemetry joint_pos has {n_recv} joints but model has "
                        f"{n_model}; updating the common prefix only")
                k = min(n_recv, n_model)
                self._mj_data.qpos[qpos_offset:qpos_offset + k] = s.joint_pos[:k]

                # Keep freejoint identity so FK outputs stay robot-local.
                # /robot already carries world pose; non-identity here would double-transform.
                self._mj_data.qpos[0:3] = [0, 0, 0]  # position
                self._mj_data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w,x,y,z)

                # Only compute forward kinematics (not full physics step - faster)
                self._mujoco.mj_kinematics(self._mj_model, self._mj_data)

                if self._mesh_items:
                    body_pos_all = self._mj_data.xpos[self._body_ids_arr]
                    body_mat_all = self._mj_data.xmat[self._body_ids_arr].reshape(-1, 3, 3)
                    body_quat_all = rot_mats_to_quats_wxyz_np(body_mat_all)

                    for i, (_, handle) in enumerate(self._mesh_items):
                        handle.position = body_pos_all[i]
                        handle.wxyz = body_quat_all[i]

                if self._ee_ids is not None:
                    ee_quats = rot_mats_to_quats_wxyz_np(
                        self._mj_data.xmat[self._ee_ids].reshape(2, 3, 3))
                    for frame, bid, q in zip(self._ee_frames, self._ee_ids, ee_quats):
                        frame.position = self._mj_data.xpos[bid]
                        frame.wxyz = q

    def submit_state(self, state: VizState) -> None:
        """Non-blocking latest-value-wins state submission for viz thread."""
        self._viz_state = state
        self._viz_state_version += 1

    def apply_if_changed(self) -> bool:
        """Apply latest state only if it changed since last apply.

        Designed for callers that drive their own render loop (enable_internal_loop=False).
        Safe to call from a single external thread; never call concurrently with _viz_loop.

        Returns True if _apply_state() was called.
        """
        if self._viz_state is None or self._viz_state_version == self._applied_version:
            return False
        self._applied_version = self._viz_state_version
        self._apply_state(self._viz_state)
        return True

    def update_from_packet(self, packet: dict) -> None:
        """Parse a published viz packet and submit it to the renderer."""
        self.submit_state(viz_state_from_packet(packet))

    def update(self, *, vicon_result=None, imu_quat_wxyz,
               wb_pos=None, wb_quat_wxyz=None, joint_pos=None) -> None:
        """Non-blocking: stash a copy of the current state for the viz thread."""
        _a = lambda x: None if x is None else np.array(x)
        self.submit_state(VizState(
            vicon_pos     = np.array(vicon_result[0][0]) if vicon_result is not None else None,  # (N_markers,7)[0]
            imu_quat_wxyz = np.array(imu_quat_wxyz),
            wb_pos        = _a(wb_pos),
            wb_quat_wxyz  = _a(wb_quat_wxyz),
            joint_pos     = _a(joint_pos),
        ))
