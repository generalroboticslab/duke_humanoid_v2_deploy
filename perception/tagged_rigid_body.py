"""Tagged rigid body: config, tag placement, rendering, and pose estimation bridge."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union, Tuple, Any
import numpy as np
import trimesh
import viser
import cv2
import time
from scipy.spatial.transform import Rotation
from common.rotation_utils import matrix_to_quat

TAG_SIZE     = 0.03   # m — black tag area
BORDER_SIZE  = 0.005  # m — white quiet zone on each side
TAG_FAMILY   = "tag36h11"

# Rendering defaults (metres)
TAG_AXES_LEN    = 0.015
TAG_AXES_RADIUS = 0.0005
BODY_AXES_LEN    = 0.04
BODY_AXES_RADIUS = 0.001
FRAME_AXES_LEN    = 0.02
FRAME_AXES_RADIUS = 0.0005
BODY_AXES_3D = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])  # for 2D image projection (scale independent of BODY_AXES_LEN)

_CACHE = Path(__file__).parent / ".tag_cache"
_MESH_CACHE: Dict[Tuple, trimesh.Trimesh] = {}  # (path, mtime, scale, alpha) → Trimesh


def _load_body_mesh(path: str, scale: float, alpha: int) -> trimesh.Trimesh:
    """Load, scale, and alpha-blend a mesh file. Cached by (path, mtime, scale, alpha)
    so changes on disk are picked up automatically."""
    key = (path, os.path.getmtime(path), scale, alpha)
    if key not in _MESH_CACHE:
        mesh = trimesh.load(path, force="mesh")
        mesh.apply_scale(scale)
        if alpha < 255:
            # to_color() normalises any visual type (TextureVisuals, etc.) into plain vertex colours
            color_vis = mesh.visual.to_color()
            color_vis.mesh = mesh  # trimesh bug: to_color() drops the mesh reference, leaving count=None
            color_vis.vertex_colors[:, 3] = alpha
            mesh.visual = color_vis
            # Viser/GLB requires a PBR material with alphaMode="BLEND" to render transparency;
            # trimesh needs this exact chain: ColorVisuals → TextureVisuals → PBRMaterial
            mesh.visual = mesh.visual.to_texture()
            mesh.visual.material = mesh.visual.material.to_pbr()
            mesh.visual.material.alphaMode = "BLEND"
        _MESH_CACHE[key] = mesh
    return _MESH_CACHE[key]


# ── Helpers ──────────────────────────────────────────────────────────

def make_tag_image(tid: int, family: str = TAG_FAMILY, size: float = TAG_SIZE, border: float = BORDER_SIZE) -> np.ndarray:
    """Upscaled RGB tag image (cached). Delete .tag_cache/ to regenerate."""
    ratio_key = f"_b{round(border/size*1000)}" if border != BORDER_SIZE or size != TAG_SIZE else ""
    path = _CACHE / family / f"{tid}{ratio_key}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return np.load(path)
    from moms_apriltag import TagGenerator2
    raw = TagGenerator2(family).generate(tid)
    up  = np.kron(raw, np.ones((16, 16), dtype=raw.dtype))
    pad = max(1, round(up.shape[0] * border / size))
    img = np.stack([np.pad(up, pad, constant_values=up.max())] * 3, axis=-1)
    # Write tag ID onto the bottom padding strip
    label = str(tid)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, pad * 0.045, max(1, pad // 8)
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    tx = (img.shape[1] - tw) // 2
    ty = img.shape[0] - (pad - th) // 2
    cv2.putText(img, label, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    np.save(path, img)
    return img

# ── Config dataclasses ───────────────────────────────────────────────

@dataclass(eq=False)
class TagConfig:
    """Per-tag config. None fields inherit from the body-level TaggedRigidBodyConfig."""
    id: int
    pos: Union[np.ndarray, List[float], tuple]
    rot: Optional[Union[np.ndarray, List[List[float]]]] = None  # None → resolved by body config
    size: Optional[float] = None
    border_size: Optional[float] = None
    family: Optional[str] = None

    def __post_init__(self):
        """Ensure array inputs are converted to numpy arrays."""
        self.pos = np.asarray(self.pos, dtype=np.float64)
        if self.rot is not None:
            self.rot = np.asarray(self.rot, dtype=np.float64)

class ResolvedTag(NamedTuple):
    size: float
    border: float
    family: str

@dataclass
class FrameConfig:
    """A named point/orientation fixed in the body frame.

    Use this to mark any point of interest on a tracked object — a tool tip,
    a grasp point, a connector socket — so you can query its world-frame pose
    at runtime via TaggedRigidBody.get_frame_pose("name").

    Args:
        pos: [x, y, z] offset from the body origin, in metres.
             Measured in the body's own coordinate frame.
        rot: 3x3 rotation matrix giving the frame's orientation relative to
             the body. None means "same orientation as the body" (identity).

    Example::

        frames={
            "tcp":   FrameConfig(pos=[0, 0, 0.12]),           # 12 cm above origin, same orientation
            "grasp": FrameConfig(pos=[0.02, 0, 0.05], rot=R), # offset + custom orientation
        }
    """
    pos: Union[np.ndarray, List[float], tuple]
    rot: Optional[Union[np.ndarray, List[List[float]]]] = None  # None → same orientation as body

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64)
        if self.rot is not None:
            self.rot = np.asarray(self.rot, dtype=np.float64)


@dataclass
class MeshConfig:
    """Configuration for one mesh part of a rigid body.

    A body can have multiple named meshes — useful for multi-part assemblies
    (e.g. a tool with a handle and a tip, or a holder with an attached pipette).

    Args:
        path:  Path to the mesh file (OBJ, STL, PLY, …).
        scale: Scale factor applied to mesh vertices. Default 1e-3 converts mm → m.
        alpha: Opacity 0–255. Values < 255 enable transparency (BLEND mode).
        pos:   XYZ offset from the body origin, in metres (body frame).
        rot:   3×3 rotation matrix for the mesh's orientation in the body frame.
               None means "same orientation as the body" (identity).

    Example::

        meshes={
            "body": MeshConfig("holder.obj"),
            "tip":  MeshConfig("tip.obj", pos=[0, 0, 0.12], alpha=200),
        }
    """
    path: str
    scale: float = 1e-3
    alpha: int = 255
    pos: Union[np.ndarray, List[float], tuple] = field(default_factory=lambda: np.zeros(3))
    rot: Optional[Union[np.ndarray, List[List[float]]]] = None  # None → identity

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.rot = np.asarray(self.rot, dtype=np.float64) if self.rot is not None else np.eye(3)

    @property
    def key(self) -> str:
        """Unique batch identifier — bodies sharing the same key share one GPU draw call."""
        return f"mesh_{Path(self.path).stem}_{self.scale}_{self.alpha}"

    def to_entry(self) -> 'MeshEntry':
        """Load (or retrieve from cache) the mesh and return a render-ready MeshEntry."""
        return MeshEntry(self.key, _load_body_mesh(self.path, self.scale, self.alpha), self.pos, self.rot)


class MeshEntry(NamedTuple):
    """Resolved mesh geometry with its batch key and local pose offset.

    Returned by TaggedRigidBodyConfig.get_mesh_entries(). Used internally by
    BatchedScene to create _MeshGroups and by TaggedRigidBody to render meshes.
    """
    key:  str               # unique batch key — bodies sharing the same key share one GPU draw call
    mesh: trimesh.Trimesh   # geometry
    pos:  np.ndarray        # offset from body origin (metres, body frame)
    rot:  np.ndarray        # orientation in body frame (3×3 matrix)


class PoseEstimate(NamedTuple):
    """Estimated pose of a rigid body from a single camera frame.

    Naming convention: ``R_cam`` / ``t_cam`` = body expressed in camera frame;
    ``R_world`` / ``t_world`` = body expressed in world frame.
    """
    R_cam: np.ndarray    # body's 3×3 rotation expressed in camera coordinates (from solvePnP)
    t_cam: np.ndarray    # body's position (metres) in camera frame
    R_world: np.ndarray  # body's 3×3 rotation expressed in world coordinates
    t_world: np.ndarray  # body's position (metres) in world frame
    inliers: np.ndarray  # boolean mask [n_tags] — True if all 4 corners of that tag were inliers

class BodyPose(NamedTuple):
    """Pose of a body, tag, or named frame in world coordinates."""
    pos: np.ndarray  # shape (3,)
    R:   np.ndarray  # shape (3, 3)

@dataclass(eq=False)
class TaggedRigidBodyConfig:
    """Pure-data configuration for a tagged rigid body (serializable, no runtime state).

    To add named frames (queryable points on the object), pass a ``frames`` dict::

        cfg = TaggedRigidBodyConfig(
            name="holder",
            tags={...},
            frames={
                "tcp": FrameConfig(pos=[0, 0, 0.12]),   # 12 cm above origin
            }
        )
        # After detection starts:
        body.get_frame_pose("tcp")  # → BodyPose(pos, R) in world frame

    To attach meshes, pass a ``meshes`` dict::

        cfg = TaggedRigidBodyConfig(
            name="tool",
            tags={...},
            meshes={
                "body": MeshConfig("tool_body.obj"),
                "tip":  MeshConfig("tool_tip.obj", pos=[0, 0, 0.12], alpha=200),
            }
        )

    Extension point: override ``get_mesh_entries()`` in a subclass to provide
    procedurally-generated meshes (see ChamferedCubeConfig for an example).

    Extension point: pass ``body_class=MyTaggedRigidBody`` (a subclass of
    TaggedRigidBody) to override the default renderer.
    BatchedScene.add() will instantiate that class instead of TaggedRigidBody.
    """
    tags: Dict[str, TagConfig]
    name: str = "body"
    tag_size: float = TAG_SIZE
    tag_border_size: float = BORDER_SIZE
    tag_family: str = TAG_FAMILY
    # Per-body RANSAC pose-solve settings. Defaults reproduce the previously
    # hardcoded behaviour, so existing bodies are unchanged; only bodies that opt
    # in (e.g. hand-stuck tables with mm-level sticker error) loosen them.
    ransac_reproj_error: float = 3.0     # px; solvePnPRansac inlier threshold
    ransac_iterations: int = 100         # solvePnPRansac iterationsCount (cv2 default)
    ransac_min_inlier_tags: int = 1      # min inlier tags to accept a pose; gate only bites when > 1
    # Planar-body pose solve. Coplanar tags (e.g. all flat on a table top) give a
    # two-fold PnP ambiguity that RANSAC/ITERATIVE flips between frame-to-frame. When
    # True, solve with IPPE (the planar-target solver) via solvePnPGeneric, which returns
    # BOTH branches, and keep the one whose +Z (outward face normal) faces the camera.
    # Off by default, so non-planar bodies (cubes/tripods) keep the RANSAC path unchanged.
    planar_solve: bool = False
    meshes: Dict[str, MeshConfig] = field(default_factory=dict)
    frames: Dict[str, FrameConfig] = field(default_factory=dict)  # see FrameConfig for usage
    body_class: Optional[type] = field(default=None, repr=False)  # override with a TaggedRigidBody subclass for custom rendering

    def get_mesh_entries(self) -> Dict[str, 'MeshEntry']:
        """Return all mesh geometries for this body, keyed by name.

        Override in subclasses to provide procedurally-generated meshes::

            def get_mesh_entries(self):
                mesh = generate_my_mesh(self.param)
                return {"body": MeshEntry(key=f"my_mesh_{self.param}", mesh=mesh,
                                          pos=np.zeros(3), rot=np.eye(3))}
        """
        return {name: m.to_entry() for name, m in self.meshes.items()}

    def __post_init__(self):
        """Pre-compute per-tag transforms for fast pose estimation."""
        tag_list = list(self.tags.values())
        n = len(tag_list)

        self.tag_ids = np.array([t.id for t in tag_list], dtype=np.int32)
        self.id_to_idx = {int(tid): i for i, tid in enumerate(self.tag_ids)}

        # Pre-compute tag properties
        self.resolved = [self._resolve_tag_props(t) for t in tag_list]
        self.size_arr = np.array([r.size for r in self.resolved], dtype=np.float64)
        
        # Per-tag pose arrays
        self.pos_arr = np.array([t.pos for t in tag_list], dtype=np.float64)
        self.rot_arr = np.array([self.resolve_tag_rot(t) for t in tag_list], dtype=np.float64)

        # Pre-compute the 4 local 3D corners for each tag based on its size and configuration
        # AprilTag corner order: bottom-left, bottom-right, top-right, top-left
        self.tag_obj_pts = np.zeros((n, 4, 3), dtype=np.float64)
        for i in range(n):
            s = self.size_arr[i] / 2.0
            # Base corners in tag frame (X right, Y down => bottom-left is (-s, s), etc.)
            corners_tag = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
            # Transform to body frame
            self.tag_obj_pts[i] = corners_tag @ self.rot_arr[i].T + self.pos_arr[i]

        # Pre-pack user-defined frames into arrays so get_frame_pose() is a
        # single matrix multiply at query time rather than a Python loop.
        frame_list = list(self.frames.values())
        self.frame_name_to_idx = {name: i for i, name in enumerate(self.frames.keys())}
        self.frame_pos_arr = np.array([f.pos for f in frame_list], dtype=np.float64).reshape(-1, 3)
        self.frame_rot_arr = np.array(
            [f.rot if f.rot is not None else np.eye(3) for f in frame_list],
            dtype=np.float64
        ).reshape(-1, 3, 3)

    def _resolve_tag_props(self, tag: TagConfig) -> ResolvedTag:
        """Resolve size/border/family with body defaults filling None values."""
        return ResolvedTag(
            size   = tag.size        if tag.size        is not None else self.tag_size,
            border = tag.border_size if tag.border_size is not None else self.tag_border_size,
            family = tag.family      if tag.family      is not None else self.tag_family,
        )

    def resolve_tag_rot(self, tag: TagConfig) -> np.ndarray:
        """Return tag base rotation; identity if unspecified. Subclasses can override."""
        return tag.rot if tag.rot is not None else np.eye(3)



class TagRegistry:
    """Fast tag->body lookup. Initialized from a list or dict of configs."""
    def __init__(self, configs: Union[List[TaggedRigidBodyConfig], Dict[str, TaggedRigidBodyConfig]]):
        if isinstance(configs, dict):
            configs = list(configs.values())
        self.configs = list(configs)
        self._lookup = {}
        self._tag_sizes = {}

        for cfg in self.configs:
            for i, tid in enumerate(cfg.tag_ids):
                tid_int = int(tid)
                if tid_int in self._lookup:
                    other_cfg = self._lookup[tid_int][0]
                    raise ValueError(f"Tag ID {tid_int} in '{cfg.name}' conflicts with '{other_cfg.name}'")
                self._lookup[tid_int] = (cfg, i)
                self._tag_sizes[tid_int] = cfg.size_arr[i]

    @property
    def tag_sizes(self) -> Dict[int, float]:
        """Pre-built {tag_id: size} for AprilTagDetector."""
        return self._tag_sizes

    @property
    def tag_ids(self) -> set:
        """Set of all tag IDs claimed by any registered rigid body."""
        return set(self._lookup.keys())

    def detect_bodies(self, detected_tags: List[Any]) -> Dict[TaggedRigidBodyConfig, List[Any]]:
        """Group detected tags by body config."""
        bodies = {}
        for tag in detected_tags:
            entry = self._lookup.get(tag.tag_id)
            if entry:
                bodies.setdefault(entry[0], []).append(tag)
        return bodies



# ── Body classes ─────────────────────────────────────────────────────

@dataclass
class PoseRecord:
    """Stores the latest position, rotation, and timestamp for a single tracked object."""
    pos: np.ndarray
    wxyz: np.ndarray
    last_seen: float

def _auto_lod(mesh: trimesh.Trimesh):
    """Compute LOD level from vertex count.

    Returns a Viser LOD spec: a tuple of (screen_size_threshold, simplification_ratio) pairs,
    or "off" when the mesh is already simple enough not to need decimation.
    The 2.0 threshold means LOD kicks in when the object occupies > 2 normalised screen units.
    """
    ratio = 1000.0 / max(mesh.vertices.shape[0], 1)
    return ((2.0, ratio),) if ratio < 0.5 else "off"

class _MeshGroup:
    """Manages a single Viser BatchedGlbHandle for a specific 3D mesh geometry.
    Instead of sending one mesh per object to the browser, this batches all instances
    of the same shape into a single GPU draw call for significant performance gains.
    """
    def __init__(self, server: viser.ViserServer, name: str, mesh: trimesh.Trimesh, prefix: str = ""):
        mesh_path = f"/{prefix}/shared_meshes/{name}" if prefix else f"/shared_meshes/{name}"
        # We start with empty arrays. They are resized dynamically during `flush()`.
        self._handle = server.scene.add_batched_meshes_trimesh(
            mesh_path, mesh,
            batched_wxyzs=np.zeros((0, 4), dtype=np.float32),
            batched_positions=np.zeros((0, 3), dtype=np.float32),
            lod=_auto_lod(mesh)
        )
        # Dictionary mapping unique body IDs to their latest pose.
        # This allows O(1) updates when an object moves.
        self._active_bodies: Dict[int, PoseRecord] = {}

    def update_pose(self, body_id: int, pos: np.ndarray, wxyz: np.ndarray):
        """Called every frame by individual bodies to update their local pose cache."""
        if body_id in self._active_bodies:
            record = self._active_bodies[body_id]
            record.pos = pos
            record.wxyz = wxyz
            record.last_seen = time.perf_counter()
        else:
            self._active_bodies[body_id] = PoseRecord(pos, wxyz, time.perf_counter())

    def flush(self, now: float, retention_time_s: float):
        """Pushes active poses to Viser. Handles SLAM object flickering by
        keeping objects visible briefly (retention_time_s) after detection loss."""
        active_pos = []
        active_wxyz = []
        expired_ids = []

        for body_id, record in self._active_bodies.items():
            if now - record.last_seen > retention_time_s:
                expired_ids.append(body_id) # TTL expired, object lost
            else:
                active_pos.append(record.pos)
                active_wxyz.append(record.wxyz)

        # Clean up stale objects from memory
        for body_id in expired_ids:
            del self._active_bodies[body_id]

        if not active_pos:
            self._handle.visible = False
            return

        # Send one single payload over WebSocket to move all objects of this type instantly
        self._handle.batched_positions = np.array(active_pos, dtype=np.float32)
        self._handle.batched_wxyzs = np.array(active_wxyz, dtype=np.float32)
        self._handle.visible = True


class _MeshSlot(NamedTuple):
    """Ties one named mesh to its batch-rendering group and local pose offset.

    Created by BatchedScene.add() — one slot per entry in cfg.get_mesh_entries().
    The local_pos/local_rot offsets are applied to the body pose at each update
    so each mesh part can sit at the correct position/orientation on the body.
    """
    group:     _MeshGroup   # batch manager for this mesh geometry
    local_pos: np.ndarray   # mesh offset from body origin (metres, body frame)
    local_rot: np.ndarray   # mesh orientation in body frame (3×3 matrix)


class BatchedScene:
    """Centralized manager for rendering many dynamic rigid bodies.
    Instead of drawing 500 individual meshes, this class groups bodies by their geometric shape
    (e.g. all 30mm cubes) and draws them using Instanced Rendering via BatchedGlbHandle.
    """
    def __init__(self, server: viser.ViserServer, retention_time_s: float = 1.0, prefix: str = ""):
        self.server = server
        self.retention_time_s = retention_time_s
        self.prefix = prefix.strip("/")
        self._mesh_groups: Dict[str, _MeshGroup] = {}
        # Every body added via .add() is tracked here so flush() can hide its
        # scene node when it hasn't been detected for more than retention_time_s.
        self._bodies: List['TaggedRigidBody'] = []

    def add(self, cfg: TaggedRigidBodyConfig) -> 'TaggedRigidBody':
        # Build one _MeshSlot per mesh, lazily creating _MeshGroups on first encounter.
        slots: Dict[str, _MeshSlot] = {}
        for mesh_name, entry in cfg.get_mesh_entries().items():
            if entry.key not in self._mesh_groups:
                self._mesh_groups[entry.key] = _MeshGroup(self.server, entry.key, entry.mesh, prefix=self.prefix)
            slots[mesh_name] = _MeshSlot(self._mesh_groups[entry.key], entry.pos, entry.rot)

        body_cls = cfg.body_class or TaggedRigidBody
        body = body_cls(cfg, self.server, mesh_slots=slots, prefix=self.prefix)
        body.render()
        self._bodies.append(body)
        return body

    def flush(self):
        now = time.perf_counter()
        for body in self._bodies:
            body._flush_visibility(now, self.retention_time_s)
        for group in self._mesh_groups.values():
            group.flush(now, self.retention_time_s)

class TaggedRigidBody:
    """Base class for a rigid body with AprilTags. Handles tag rendering and provides
    pre-computed transforms for pose estimation."""

    def __init__(self, cfg: TaggedRigidBodyConfig, server: viser.ViserServer,
                 mesh_slots: Dict[str, _MeshSlot] = {}, prefix: str = ""):
        self.cfg = cfg
        self.server = server
        self.prefix = prefix.strip("/")
        self.root_path = f"/{self.prefix}/{self.cfg.name}" if self.prefix else self.cfg.name

        # If provided, each slot ties one named mesh to its BatchedScene group.
        # An empty dict means this body was created outside BatchedScene and will
        # render its own individual mesh nodes instead.
        self._mesh_slots = mesh_slots
        self._body_id = id(self)  # Ensure unique ID for batching even if config is shared
        self._root_handle: Optional[viser.FrameHandle] = None

        # Temporal smoothing state
        self._filtered_t: Optional[np.ndarray] = None
        self._filtered_R: Optional[np.ndarray] = None

        # Visibility tracking: stamp updated by update_pose()/set_pose().
        # _root_visible caches the last value sent to viser so we only emit a
        # WebSocket message when visibility actually changes.
        self._last_seen: float = -float('inf')
        self._root_visible: bool = True  # viser nodes start visible by default

    def _flush_visibility(self, now: float, retention_time_s: float):
        """Hide/show this body's scene node. Only writes to viser when the state
        changes to avoid sending redundant WebSocket messages every frame.

        All child nodes (tag frames, image quads, named frames) inherit visibility
        via the Three.js scene-graph parent chain — no per-child visibility calls
        are needed. Child nodes must NOT have their own explicit visible=False set
        at creation, or they will stay hidden even when this root is shown.
        See _render_tags() docstring for the full rationale.
        """
        visible = (now - self._last_seen) <= retention_time_s
        if visible != self._root_visible:
            self._root_handle.visible = visible
            self._root_visible = visible

    def _update_mesh_slots(self):
        """Push current body pose (+ per-mesh offsets) to all batch groups."""
        for slot in self._mesh_slots.values():
            world_pos = self._filtered_R @ slot.local_pos + self._filtered_t
            world_R   = self._filtered_R @ slot.local_rot
            slot.group.update_pose(self._body_id, world_pos, matrix_to_quat(world_R, wxyz=True))

    def set_pose(self, position: Union[np.ndarray, List[float], tuple] = (0, 0, 0),
                 wxyz: Union[np.ndarray, List[float], tuple] = (1, 0, 0, 0)):
        """Set the global pose of the rigid body."""
        if self._root_handle is None and self.server is not None:
            self.render()
        self._filtered_t = np.asarray(position, dtype=float)
        wxyz_arr = np.asarray(wxyz, dtype=float)
        # scipy uses [x, y, z, w]; our convention is [w, x, y, z]
        self._filtered_R = Rotation.from_quat([*wxyz_arr[1:], wxyz_arr[0]]).as_matrix()
        self._last_seen = time.perf_counter()

        if self._root_handle is not None:
            self._root_handle.position = self._filtered_t
            self._root_handle.wxyz = wxyz_arr

        if self._mesh_slots:
            self._update_mesh_slots()

    def update_pose(self, est: PoseEstimate, smoothing_alpha: float = 1.0):
        """Update the global pose of the rigid body from a PoseEstimate.

        Args:
            est: The PoseEstimate object containing world coordinates.
            smoothing_alpha: Float in (0.0, 1.0]. 1.0 means no smoothing.
                             Lower values explicitly freeze jitter at the cost of latency.
        """
        if self._root_handle is None and self.server is not None:
            self.render()
        if not np.all(np.isfinite(est.R_world)) or not np.all(np.isfinite(est.t_world)):
            return

        if smoothing_alpha >= 1.0 or self._filtered_t is None:
            self._filtered_t = est.t_world
            self._filtered_R = est.R_world
        else:
            # EMA for translation vector
            self._filtered_t = (1.0 - smoothing_alpha) * self._filtered_t + smoothing_alpha * est.t_world

            # Exponential-map interpolation for rotation: R_new = R0 * (R0⁻¹ * R1)^alpha
            # This is equivalent to Slerp but expressed via the rotation vector (axis-angle).
            R0 = Rotation.from_matrix(self._filtered_R)
            R1 = Rotation.from_matrix(est.R_world)

            # Decompose relative rotation into an axis-angle, scale the angle, then reapply
            rot_diff = R0.inv() * R1
            rotvec = rot_diff.as_rotvec()
            self._filtered_R = (R0 * Rotation.from_rotvec(smoothing_alpha * rotvec)).as_matrix()

        self._last_seen = time.perf_counter()

        if self._root_handle is not None:
            self._root_handle.position = self._filtered_t
            quat_wxyz = matrix_to_quat(self._filtered_R, wxyz=True)
            self._root_handle.wxyz = quat_wxyz

        if self._mesh_slots:
            self._update_mesh_slots()

    # ── Pose queries ──────────────────────────────────────────────────

    def get_pose(self) -> Optional[BodyPose]:
        """Current body-origin pose in world frame.
        Returns None before the first detection arrives."""
        if self._filtered_t is None:
            return None
        return BodyPose(pos=self._filtered_t.copy(), R=self._filtered_R.copy())

    def get_tag_pose(self, tag_id: int) -> Optional[BodyPose]:
        """World-frame pose of an individual tag (by its AprilTag integer ID).
        Returns None before the first detection. Raises KeyError for IDs not in this config."""
        if self._filtered_t is None:
            return None
        idx = self.cfg.id_to_idx.get(tag_id)
        if idx is None:
            raise KeyError(f"Tag ID {tag_id} not found in '{self.cfg.name}'")
        return BodyPose(
            pos=self._filtered_R @ self.cfg.pos_arr[idx] + self._filtered_t,
            R=self._filtered_R @ self.cfg.rot_arr[idx],
        )

    def get_frame_pose(self, frame_name: str) -> Optional[BodyPose]:
        """World-frame pose of a named frame defined in cfg.frames.
        Returns None before the first detection. Raises KeyError if the name is not in cfg.frames."""
        if self._filtered_t is None:
            return None
        idx = self.cfg.frame_name_to_idx.get(frame_name)
        if idx is None:
            raise KeyError(f"Frame '{frame_name}' not found in '{self.cfg.name}'")
        return BodyPose(
            pos=self._filtered_R @ self.cfg.frame_pos_arr[idx] + self._filtered_t,
            R=self._filtered_R @ self.cfg.frame_rot_arr[idx],
        )

    def draw_on_image(self, est: PoseEstimate, K: np.ndarray, image: np.ndarray):
        """Project body axes + label onto OpenCV image using camera-frame pose."""
        axes_cam = BODY_AXES_3D @ est.R_cam.T + est.t_cam
        axes_2d = axes_cam @ K.T

        # Avoid division by zero
        Z = axes_2d[:, 2:3]
        Z[Z == 0] = 1e-6
        axes_2d = (axes_2d[:, :2] / Z).astype(int)

        # Prevent OpenCV cv2.line segfault if coordinates are wildly off-screen or non-finite
        if not np.all(np.isfinite(axes_2d)) or np.any(np.abs(axes_2d) > 10000):
            return

        o = tuple(axes_2d[0])
        cv2.line(image, o, tuple(axes_2d[1]), (0, 0, 255), 2)   # X red
        cv2.line(image, o, tuple(axes_2d[2]), (0, 255, 0), 2)   # Y green
        cv2.line(image, o, tuple(axes_2d[3]), (255, 0, 0), 2)   # Z blue
        cv2.putText(image, self.cfg.name, o,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # ── Pose estimation ──────────────────────────────────────────────

    def estimate_pose(self, detected_tags: List[Any], cam_transform: np.ndarray,
                      K: np.ndarray, use_ransac: bool = True, ransac_threshold: Optional[float] = None) -> Optional[PoseEstimate]:
        """Estimate rigid body pose from detected AprilTags via solvePnP.

        Args:
            detected_tags: pupil_apriltags Detection objects belonging to THIS body.
            cam_transform: 4×4 camera-to-world SE(3) matrix (np.eye(4) for camera-frame only).
            K: 3×3 camera intrinsic matrix.
            use_ransac: Use RANSAC to reject outlier corners.
            ransac_threshold: Reprojection error threshold (pixels). None (default) →
                use this body's ``cfg.ransac_reproj_error``. Set RANSAC_DEBUG=1 in the
                environment to print per-tag residuals and inlier/outlier verdicts.

        Returns:
            PoseEstimate if successful, None otherwise.
        """
        cfg = self.cfg
        n_tags = len(detected_tags)
        if n_tags == 0:
            return None

        # Build index array for fast lookups into config arrays
        tag_indices = [cfg.id_to_idx[tag.tag_id] for tag in detected_tags]

        # Stack 2D image points and 3D object points
        img_pts = np.vstack([tag.corners for tag in detected_tags]).astype(np.float64)
        obj_pts = np.vstack([cfg.tag_obj_pts[idx] for idx in tag_indices])

        # Per-body RANSAC settings (defaults reproduce the old hardcoded behaviour;
        # only opted-in bodies loosen them). Caller may still override the threshold.
        reproj_err = ransac_threshold if ransac_threshold is not None else cfg.ransac_reproj_error

        # Planar bodies (coplanar tags, e.g. a table top): RANSAC/ITERATIVE flips between
        # the two ambiguous branches. Use IPPE — the planar-target solver — via
        # solvePnPGeneric to get BOTH branches, then keep the one whose +Z (outward face
        # normal) faces the camera (its z-component in the camera frame is < 0). Per-tag
        # inliers are by reprojection residual so the min-inlier-tags floor still applies.
        if cfg.planar_solve:
            try:
                n_sol, rvecs, tvecs, rerrs = cv2.solvePnPGeneric(
                    obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:
                return None  # degenerate/edge-on frame: IPPE can't solve this plane
            if n_sol < 1:
                return None
            best = None
            for i in range(n_sol):
                Ri = cv2.Rodrigues(rvecs[i])[0]
                if Ri[2, 2] < 0:  # outward normal points back toward the camera
                    e = float(rerrs[i][0]) if rerrs is not None else 0.0
                    if best is None or e < best[0]:
                        best = (e, rvecs[i], tvecs[i])
            if best is None:
                return None  # no camera-facing branch (degenerate/edge-on view)
            rvec, tvec, success = best[1], best[2], True
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
            resid = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).reshape(n_tags, 4).mean(axis=1)
            inlier_mask = resid < reproj_err
        # ITERATIVE = Levenberg-Marquardt reprojection error minimization.
        # RANSAC needs ≥2 full tags (8 corners); single-tag bodies use plain solvePnP.
        elif use_ransac and len(img_pts) >= 8:
            success, rvec, tvec, inliers_pts = cv2.solvePnPRansac(
                obj_pts, img_pts, K, None,
                iterationsCount=cfg.ransac_iterations,
                reprojectionError=reproj_err,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success and inliers_pts is not None:
                mask = np.zeros(len(img_pts), dtype=bool)
                mask[inliers_pts.flatten()] = True
                inlier_mask = mask.reshape(n_tags, 4).all(axis=1)
            else:
                inlier_mask = np.zeros(n_tags, dtype=bool)
        else:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            inlier_mask = np.ones(n_tags, dtype=bool)

        if not success:
            return None

        # Optional per-solve diagnostics: RANSAC_DEBUG=1 prints each tag's mean
        # reprojection residual (px) and inlier/outlier verdict for THIS body — so a
        # rejected/jittery body can be diagnosed live without another code round.
        if os.environ.get("RANSAC_DEBUG"):
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
            resid = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).reshape(n_tags, 4).mean(axis=1)
            ids = [t.tag_id for t in detected_tags]
            parts = [f"id{ids[i]}={resid[i]:.1f}px[{'IN' if inlier_mask[i] else 'OUT'}]" for i in range(n_tags)]
            print(f"[RANSAC {cfg.name}] thr={reproj_err:.1f}px iters={cfg.ransac_iterations} "
                  f"inliers={int(inlier_mask.sum())}/{n_tags}  " + " ".join(parts))

        # Inlier-tag floor: only bites when the body sets it > 1 (e.g. the tables need
        # ≥2 tags to agree before the pose is trusted). Default 1 = no extra gate,
        # so every existing body is unchanged.
        if cfg.ransac_min_inlier_tags > 1 and int(inlier_mask.sum()) < cfg.ransac_min_inlier_tags:
            return None

        # Body pose in camera coordinates
        R_body_in_cam = cv2.Rodrigues(rvec)[0]
        t_body_in_cam = tvec.flatten()

        # Chain: body-in-cam → cam-in-world = body-in-world
        R_cam_in_world = cam_transform[:3, :3]
        t_cam_in_world = cam_transform[:3, 3]
        R_body_in_world = R_cam_in_world @ R_body_in_cam
        t_body_in_world = R_cam_in_world @ t_body_in_cam + t_cam_in_world

        if not np.all(np.isfinite(R_body_in_world)) or not np.all(np.isfinite(t_body_in_world)):
            print(f"Warning: estimate_pose produced NaNs for {cfg.name}!")
            return None

        return PoseEstimate(
            R_cam=R_body_in_cam, t_cam=t_body_in_cam,
            R_world=R_body_in_world, t_world=t_body_in_world,
            inliers=inlier_mask,
        )

    def render(self):
        # Create root frame
        self._root_handle = self.server.scene.add_frame(
            self.root_path,
            axes_length=0.0,  # Hide axes of the root frame by default, body axes rendered separately
            axes_radius=0.0,
        )
        self._render_body()
        self._render_tags()
        self._render_frames()

    def _render_body(self):
        """Render body axes and mesh parts.

        When managed by a BatchedScene (mesh_slots is populated), mesh rendering is
        delegated to the shared _MeshGroup instances — no individual nodes are created.
        Otherwise each mesh is added as its own Viser node under the body's root frame.
        """
        name = self.root_path
        self.server.scene.add_frame(f"{name}/axes", axes_length=BODY_AXES_LEN, axes_radius=BODY_AXES_RADIUS)

        if self._mesh_slots:
            return  # BatchedScene handles mesh rendering

        for mesh_name, entry in self.cfg.get_mesh_entries().items():
            # Each mesh gets its own frame so pos/rot offsets are applied via scene hierarchy
            frame = f"{name}/{mesh_name}"
            self.server.scene.add_frame(
                frame, position=entry.pos, wxyz=matrix_to_quat(entry.rot, wxyz=True),
                axes_length=0.0, axes_radius=0.0,
            )
            self.server.scene.add_mesh_trimesh(f"{frame}/visual", entry.mesh)

    def _render_tags(self):
        """Render each tag as a positioned frame + image quad.

        Visibility design: tag image quads default to visible=True so that
        _flush_visibility() hiding/showing the body root handle is the sole
        controller of their display via the Three.js scene-graph parent chain.

        Do NOT pass visible=False at creation and do NOT add add_label() nodes here.
        Both patterns cause white-rectangle rendering artifacts in Viser:
        - explicit visible=False on a child node is NOT cleared when the parent is
          later shown, so the quad stays permanently hidden.
        - add_label() nodes use Viser's BatchedLabelManager which ignores
          handle.visible=False; label background quads (InstancedMesh) remain
          rendered even after the Python-side visible=False message is sent,
          producing white rectangles over the camera background image.
        """
        for i, (tag_name, tag) in enumerate(self.cfg.tags.items()):
            tid = tag.id
            r = self.cfg.resolved[i]
            rot = self.cfg.rot_arr[i]
            pos = self.cfg.pos_arr[i]
            vis_size = r.size + 2 * r.border
            name = f"{self.root_path}/tag_{tid}"

            self.server.scene.add_frame(
                name, position=pos, wxyz=matrix_to_quat(rot, wxyz=True),
                axes_length=TAG_AXES_LEN, axes_radius=TAG_AXES_RADIUS,
            )
            self.server.scene.add_image(
                f"{name}/visual",
                make_tag_image(tid, r.family, r.size, r.border),
                vis_size, vis_size,
                cast_shadow=False,
                receive_shadow=False,
            )

    def _render_frames(self):
        """Render each named frame (e.g. tool tip) as a small positioned frame."""
        for frame_name, i in self.cfg.frame_name_to_idx.items():
            pos = self.cfg.frame_pos_arr[i]
            rot = self.cfg.frame_rot_arr[i]
            name = f"{self.root_path}/frame_{frame_name}"
            self.server.scene.add_frame(
                name, position=pos, wxyz=matrix_to_quat(rot, wxyz=True),
                axes_length=FRAME_AXES_LEN, axes_radius=FRAME_AXES_RADIUS,
            )



# ── Verification and Entrypoint Helpers ──────────────────────────────

def _run_tracking_loop(
    detector,
    registry: 'TagRegistry',
    bodies: Dict['TaggedRigidBodyConfig', 'TaggedRigidBody'],
    scene: 'BatchedScene',
    tag_renderer=None,
    window_name: str = "Tracking",
    smoothing_alpha: float = 1.0,
    viser_server=None,
    headless: Optional[bool] = None,
    silhouette_renderer=None,
):
    """Shared detect-estimate-render loop. Runs until 'q' is pressed or Ctrl-C.

    Args:
        detector:             AprilTagDetector instance (already capturing frames).
        registry:             TagRegistry mapping tag IDs to body configs.
        bodies:               dict of config → TaggedRigidBody renderer, as returned by BatchedScene.add().
        scene:                BatchedScene to flush after each frame.
        tag_renderer:         Optional TagViserRenderer for 3D tag overlays. If provided,
                              renderer.update() is called with the latest tags each frame.
        window_name:          OpenCV window title.
        smoothing_alpha:      passed to update_pose(); 1.0 = no smoothing, 0.0 = frozen.
        viser_server:         Existing viser.ViserServer to reuse for headless display.
                              When provided and headless, the camera image streams on the
                              same server as the 3D scene — no second server is started.
        headless:             Force headless (True → Viser web) or windowed (False → OpenCV).
                              None = auto-detect from $DISPLAY.
        silhouette_renderer:  Optional MeshSilhouetteRenderer. When provided, projects each
                              body's mesh wireframe onto the 2D camera image each frame.
    """
    from camera_visualizer import create_visualizer
    viz = create_visualizer(headless=headless, server=viser_server)
    # Show placeholder immediately so the window appears before the first frame arrives.
    # OpenCV also requires waitKey() every iteration to pump its GUI event loop.
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    viz.show(window_name, placeholder)
    try:
        while True:
            if not detector.update_detection():
                if viz.wait_key(1) == ord('q'):
                    break
                continue
            vis = detector.update_2d_visualization()
            if tag_renderer:
                tag_renderer.update(detector.tags, detector.camera_transform)
            active_bodies = []
            for cfg, config_tags in registry.detect_bodies(detector.tags).items():
                body = bodies[cfg]
                est = body.estimate_pose(config_tags, detector.camera_transform, detector.K)
                if est is not None:
                    body.update_pose(est, smoothing_alpha=smoothing_alpha)
                    active_bodies.append((body, est))
            # Wireframe drawn before axes/labels so text is never occluded
            if silhouette_renderer is not None and active_bodies:
                try:
                    silhouette_renderer.draw_wireframes(
                        vis, active_bodies, detector.K, detector.camera_transform
                    )
                except Exception as _sil_exc:
                    import traceback; traceback.print_exc()
                    silhouette_renderer = None  # disable after first error
            for body, est in active_bodies:
                body.draw_on_image(est, detector.K, vis)
            scene.flush()
            viz.show(window_name, vis)
            if viz.wait_key(1) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        viz.cleanup()


def _find_first_camera():
    """Find and initialize the first available camera."""
    from camera_utils import USBCameraBackend

    try:
        from camera_realsense_utils import RealSenseCameraBackend

        # Check if any realsense found via pyrealsense2
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                # Use the serial number of the first device
                serial = devices[0].get_info(rs.camera_info.serial_number)
                backend = RealSenseCameraBackend(device=serial, ir_stream_flag=False, color_stream_flag=True)
                backend.initialize(1280, 720, 30)
                return backend
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback to general USB devices (could still be realsense without pyrealsense2)
    cameras = USBCameraBackend._get_camera_list()
    for cam in cameras:
        if cam['capture_devices']:
            backend = USBCameraBackend(device=cam['capture_devices'][0])
            backend.initialize(1280, 720, 30)
            return backend

    raise RuntimeError("No camera found")

def _find_camera(camera_index):
    """Find and initialize the first available camera."""
    from camera_utils import USBCameraBackend

    try:
        from camera_realsense_utils import RealSenseCameraBackend

        # Check if any realsense found via pyrealsense2
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                # Use the serial number of the first device
                serial = devices[camera_index].get_info(rs.camera_info.serial_number)
                backend = RealSenseCameraBackend(device=serial, ir_stream_flag=False, color_stream_flag=True)
                backend.initialize(1280, 720, 30)
                return backend
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback to general USB devices (could still be realsense without pyrealsense2)
    cameras = USBCameraBackend._get_camera_list()
    for cam in cameras:
        if cam['capture_devices']:
            backend = USBCameraBackend(device=cam['capture_devices'][camera_index])
            backend.initialize(1280, 720, 30)
            return backend

    raise RuntimeError("No camera found")

def verify_bodies(cfgs: Union[List[TaggedRigidBodyConfig], TaggedRigidBodyConfig], camera=False):
    """Verify tag placement. Opens viser, and optionally camera.

    Args:
        cfgs: single config or list of configs
        camera: if True, opens camera and updates pose from detections
    """
    if isinstance(cfgs, TaggedRigidBodyConfig):
        cfgs = [cfgs]

    server = viser.ViserServer()

    # Create body renderers
    scene = BatchedScene(server, retention_time_s=1.0)
    bodies = {c: scene.add(c) for c in cfgs}

    # Initialize at origin so batched meshes are visible immediately
    for body in bodies.values():
        body.set_pose()
    scene.flush()

    if not camera:
        print(f"Verify {len(cfgs)} bodies at http://localhost:8080")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Done")
    else:
        # Camera mode: detect tags and update poses
        from camera_tag_detection import AprilTagDetector
        registry = TagRegistry(cfgs)
        backend = _find_first_camera()
        detector = AprilTagDetector(
            camera_backend=backend,
            tag_size=cfgs[0].tag_size, # default size
            tag_sizes=registry.tag_sizes,
            viser_server=server,
        )
        # Start camera capture in background; detection runs in this (main) thread below.
        # Do NOT call update_detection_continuously() here — that would start a second
        # consumer thread racing with the main loop's update_detection() calls.
        backend.capture_frame_continuously()

        print(f"Verify {len(cfgs)} bodies at http://localhost:8080 or the OpenCV window")
        _run_tracking_loop(detector, registry, bodies, scene, window_name="Verify")
