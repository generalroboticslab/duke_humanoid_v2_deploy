"""Hand-eye calibration solver for humanoid cameras.

Loads synchronized (joint_pos, tag_pose) recordings produced by humanoid_monitor.py
--record-handeye and solves for refined camera extrinsics (T_base_link_to_camera) and
tagged-cube mounting offsets (T_obj_link_to_object) via joint nonlinear WLS.

Usage:
    python humanoid_handeye_calibration.py \\
        --logs calib_port5555.npz calib_port5556.npz \\
        --output camera_handeye.yaml

Transform convention: T_A_B transforms points FROM B INTO A (p_A = T_A_B @ p_B_hom).
Unknowns:
  X_c = T_{base_link←camera}  (6 DOF per camera)
  Y_j = T_{obj_link←object}   (6 DOF per tagged cube, shared across cameras)
Forward model: T_{cam←obj}^pred = inv(X_c) @ F_j(q) @ Y_j
  where F_j(q) = T_{base_link←wrist} from FK at joint angles q.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from humanoid_model import MJCF_MODEL_PATH as MJCF_XML_PATH

# Pre-gimbal recordings have 27 joints; the 4 cam-gimbal joints are appended at
# the end of the model's joint order, so zero-padding is exact for arm FK.
_LEGACY_N_JOINTS = 27

# Observability warning thresholds
_OBS_MIN_ROT_DEG = 30.0
_OBS_MIN_TRANS_M = 0.05
_OBS_MIN_CUBE_DEG = 20.0

# Huber loss scale: activates at ||r|| ≈ 0.01 (~10 mm or ~0.57°)
_HUBER_F_SCALE = 0.01

# Initial guess for tag cube center offset from mounting link origin (m).
# Optimizer refines to true value; just needs to be in the right ballpark.
_OBJ_INIT_OFFSET_M: list[float] = [0.0618, 0.0, 0.0]

_ARM_JOINT_LIMITS = None  # lazily loaded from MJCF in _load_arm_joint_limits()
_ARM_JOINT_QPOS_SLICE = None  # (start, stop) in model qpos

def _load_arm_joint_limits(obj_links: list[str]) -> tuple[NDArray, tuple[int, int]]:
    """Discover arm joint limits by walking the kinematic chain from obj_links to root.

    Dynamically finds all hinge joints in the FK path for each obj_link, sorts them
    by qpos address, and returns limits array + qpos slice range.

    Returns:
        limits: (N_joints, 2) array of [min, max] per joint
        qpos_slice: (start, stop) for the arm joint range in model qpos
    """
    global _ARM_JOINT_LIMITS, _ARM_JOINT_QPOS_SLICE
    if _ARM_JOINT_LIMITS is not None and _ARM_JOINT_QPOS_SLICE is not None:
        return _ARM_JOINT_LIMITS, _ARM_JOINT_QPOS_SLICE

    model = mujoco.MjModel.from_xml_path(MJCF_XML_PATH)
    data = mujoco.MjData(model)

    # Collect all hinge joints in the FK chain for each obj_link
    chain_joints: set[int] = set()
    for link_name in obj_links:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            raise ValueError(f"Body '{link_name}' not found in MJCF {MJCF_XML_PATH}")
        # Walk from body up to root (body 0 = world), collecting each body's own joints
        current = body_id
        while current > 0:
            jnt_start = model.body_jntadr[current]
            jnt_count = model.body_jntnum[current]
            for j in range(jnt_start, jnt_start + jnt_count):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    chain_joints.add(j)
            current = model.body_parentid[current]

    # Sort by qpos address to get consistent ordering
    sorted_joints = sorted(chain_joints, key=lambda j: model.jnt_qposadr[j])
    limits = np.array([model.jnt_range[j] for j in sorted_joints])

    # Determine qpos slice in the model
    qpos_addrs = [model.jnt_qposadr[j] for j in sorted_joints]
    qpos_start = min(qpos_addrs)
    qpos_stop = max(qpos_addrs) + 1  # exclusive end

    _ARM_JOINT_LIMITS = limits
    _ARM_JOINT_QPOS_SLICE = (int(qpos_start), int(qpos_stop))
    return limits, (qpos_start, qpos_stop)

# Warm-start check: warn if max rotation residual after linear stage exceeds this
_WARMUP_ROT_WARN_DEG = 15.0


# ---------------------------------------------------------------------------
# Geometry helpers (adapted from mj_envs/deploy/calibrate_vicon_robot.py)
# ---------------------------------------------------------------------------

def _batch_mat_to_rotvec(R: NDArray) -> NDArray:
    """(N,3,3) rotation matrices → (N,3) rotation vectors."""
    return Rotation.from_matrix(R).as_rotvec()


def mat_to_quat_xyzw(R: NDArray) -> NDArray:
    """(3,3) or (N,3,3) rotation matrix → xyzw quaternion."""
    return Rotation.from_matrix(R).as_quat(scalar_first=False)


def quat_wxyz_to_xyzw(q: NDArray) -> NDArray:
    """wxyz (MuJoCo) → xyzw. Works for (4,) or (N,4)."""
    return np.array([q[1], q[2], q[3], q[0]]) if q.ndim == 1 else q[:, [1, 2, 3, 0]]


def make_transform(rot: NDArray, pos: NDArray) -> NDArray:
    """Build 4×4 rigid transform(s). rot: (3,3)|(N,3,3), pos: (3,)|(N,3)."""
    single = rot.ndim == 2
    rot, pos = (rot[np.newaxis], pos[np.newaxis]) if single else (rot, pos)
    T = np.broadcast_to(np.eye(4), (len(rot), 4, 4)).copy()
    T[:, :3, :3] = rot
    T[:, :3, 3] = pos
    return T[0] if single else T


def invert_transform(T: NDArray) -> NDArray:
    """Invert rigid transform(s). T: (4,4) or (N,4,4). Uses R^T, -R^T @ t."""
    single = T.ndim == 2
    T = T[np.newaxis] if single else T
    Rt = T[:, :3, :3].transpose(0, 2, 1)
    T_inv = np.broadcast_to(np.eye(4), T.shape).copy()
    T_inv[:, :3, :3] = Rt
    T_inv[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, T[:, :3, 3])
    return T_inv[0] if single else T_inv


def params_to_transform(p: NDArray) -> NDArray:
    """6-vector [tx,ty,tz, rx,ry,rz] → 4×4 SE(3). Identity ↔ zeros(6)."""
    return make_transform(Rotation.from_rotvec(p[3:6]).as_matrix(), p[:3])


def quat_angle_diff_deg(q1_xyzw: NDArray, q2_xyzw: NDArray) -> float:
    """Angular difference in degrees between two xyzw quaternions."""
    return float(np.degrees(
        (Rotation.from_quat(q1_xyzw, scalar_first=False).inv()
         * Rotation.from_quat(q2_xyzw, scalar_first=False)).magnitude()
    ))


# ---------------------------------------------------------------------------
# FK solver
# ---------------------------------------------------------------------------

class CameraFKSolver:
    """Forward kinematics solver backed by a MuJoCo model.

    Loads the MJCF directly from XML; does not depend on humanoid_v21_constants.
    Assumes identity root (base_link at origin), so xpos/xmat give poses in
    base_link frame without any additional inversion.
    """

    def __init__(self, obj_links: list[str]):
        self._model = mujoco.MjModel.from_xml_path(MJCF_XML_PATH)
        self._data = mujoco.MjData(self._model)
        self.n_joints = self._model.nq - 7  # actuated joints after the freejoint
        self._obj_ids: dict[str, int] = {}
        for name in obj_links:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found in MJCF {MJCF_XML_PATH}")
            self._obj_ids[name] = bid

    def compute_fk_batch(self, q_joints_batch: NDArray, obj_link: str) -> NDArray:
        """Return F_j(q) for all frames: (N,4,4) wrist pose in base_link frame.

        Args:
            q_joints_batch: (N,27) actuated joint angles in MJCF order.
            obj_link: body name (e.g. 'wrist_3_L').
        """
        N = len(q_joints_batch)
        out = np.empty((N, 4, 4))
        out[:, 3, :] = [0, 0, 0, 1]
        bid = self._obj_ids[obj_link]
        # Identity root: pos=[0,0,0], quat=[w,x,y,z]=[1,0,0,0]
        self._data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        for i, q in enumerate(q_joints_batch):
            self._data.qpos[7:] = q
            mujoco.mj_kinematics(self._model, self._data)
            out[i, :3, :3] = self._data.xmat[bid].reshape(3, 3)
            out[i, :3, 3] = self._data.xpos[bid]
        return out


# ---------------------------------------------------------------------------
# Multi-log reconciliation
# ---------------------------------------------------------------------------

def _reconcile_logs(logs: list[dict]) -> list[str]:
    """Canonicalize obj_names ordering; validate obj_link consistency across logs.

    Reorders per-frame arrays in-place so all logs share the same obj_names order.
    Returns canonical_names list.
    """
    canonical_names = [str(n) for n in logs[0]["obj_names"]]
    canonical_links = {
        n: str(logs[0]["obj_links"][i]) for i, n in enumerate(canonical_names)
    }
    for log in logs[1:]:
        log_names = [str(n) for n in log["obj_names"]]
        if set(log_names) != set(canonical_names):
            raise ValueError(f"obj_names mismatch: {canonical_names} vs {log_names}")
        for n in canonical_names:
            link = str(log["obj_links"][log_names.index(n)])
            if link != canonical_links[n]:
                raise ValueError(
                    f"obj_link mismatch for '{n}': {canonical_links[n]} vs {link}"
                )
        reorder = [log_names.index(n) for n in canonical_names]
        if reorder != list(range(len(canonical_names))):
            log["obj_pos"] = log["obj_pos"][:, reorder, :]
            log["obj_quat_wxyz"] = log["obj_quat_wxyz"][:, reorder, :]
            log["obj_n_inliers"] = log["obj_n_inliers"][:, reorder]
            log["obj_valid"] = log["obj_valid"][:, reorder]
    return canonical_names


# ---------------------------------------------------------------------------
# Per-camera data preparation
# ---------------------------------------------------------------------------

def _prepare_camera_data(
    log: dict,
    fk: CameraFKSolver,
    subsample: int,
    dq_thresh: float,
    v_link_thresh: float,
) -> dict:
    """Preprocess one npz log into a camera_data dict ready for CameraHandEyeCalibrator.

    Processing order (invariant — do not reorder):
      1. Subsample
      2. Compute T_FK per object
      3. Compute w_joint (from saved w_speed)
      4. Compute w_link per object (FK-derived wrist Cartesian speed)
      5. Compute valid (FIRST — required before inv_T_meas fill)
      6. Fill inv_T_meas[invalid] = eye(4)  (NaN safety)
      7. Compute weights_sqrt

    Returns dict with keys: T_FK, inv_T_meas, weights_sqrt, obj_names, obj_links.
    """
    # Validate required keys and shapes
    required = ["q_pos", "time", "valid_telem", "w_speed", "obj_pos", "obj_quat_wxyz",
                "obj_n_inliers", "obj_valid", "obj_names", "obj_links",
                "cam_transform_approx", "cam_port"]
    for k in required:
        if k not in log:
            raise KeyError(f"npz missing key '{k}'")
    n_joints = fk.n_joints
    if log["q_pos"].shape[1] == _LEGACY_N_JOINTS and n_joints > _LEGACY_N_JOINTS:
        pad = np.zeros((len(log["q_pos"]), n_joints - _LEGACY_N_JOINTS))
        log["q_pos"] = np.concatenate([log["q_pos"], pad], axis=1)
        print(f"[handeye] note: legacy {_LEGACY_N_JOINTS}-joint recording zero-padded "
              f"to {n_joints} joints (cam gimbal at 0)")
    if log["q_pos"].shape[1] != n_joints:
        raise ValueError(f"q_pos.shape[1] = {log['q_pos'].shape[1]}, expected {n_joints}")

    obj_names = [str(n) for n in log["obj_names"]]
    obj_links_list = [str(n) for n in log["obj_links"]]
    M = len(obj_names)

    if log["obj_pos"].shape[1:] != (M, 3):
        raise ValueError(f"obj_pos shape {log['obj_pos'].shape}, expected (N,{M},3)")

    # Step 1: subsample
    N_raw = len(log["q_pos"])
    idx = np.arange(0, N_raw, subsample)
    q_pos = log["q_pos"][idx]
    time_arr = log["time"][idx]
    valid_telem_arr = log["valid_telem"][idx].astype(bool)
    w_speed_arr = log["w_speed"][idx].astype(np.float64)
    obj_pos_arr = log["obj_pos"][idx]          # (N,M,3)
    obj_quat_arr = log["obj_quat_wxyz"][idx]   # (N,M,4)
    n_inliers_arr = log["obj_n_inliers"][idx]  # (N,M)
    obj_valid_arr = log["obj_valid"][idx]      # (N,M) bool
    N = len(q_pos)

    # Hard-gate: reject frames where any arm joint exceeds MJCF limits
    limits, qpos_slice = _load_arm_joint_limits(obj_links_list)
    # qpos_slice is in MODEL qpos coords (0-33); q_pos is already the 27-element slice [7:].
    # Map model qpos index to q_pos index: q_pos[i] = model_qpos[qpos_slice[0] + i]
    arm_offset = qpos_slice[0] - 7  # model qpos start minus root (7)
    arm_q = q_pos[:, arm_offset:arm_offset + limits.shape[0]]  # (N, N_joints)
    in_limits = (
        (arm_q >= limits[:, 0]).all(axis=1) &
        (arm_q <= limits[:, 1]).all(axis=1)
    )
    valid_telem_arr &= in_limits

    C_approx = np.asarray(log["cam_transform_approx"], dtype=np.float64)
    inv_C_approx = invert_transform(C_approx)

    # Step 2: FK per object
    T_FK: list[NDArray] = []
    for m, link in enumerate(obj_links_list):
        T_FK.append(fk.compute_fk_batch(q_pos, link))  # (N,4,4)

    # Steps 3-7: per-object processing
    inv_T_meas: list[NDArray] = []
    weights_sqrt: list[NDArray] = []
    valid: NDArray = np.zeros((N, M), dtype=bool)

    for m in range(M):
        # Step 3: joint velocity weight (saved at record time, arm joints only)
        w_joint = w_speed_arr.copy()

        # Step 4: Cartesian link velocity (per object — wrist_3_L vs wrist_3_R differ)
        dt = np.maximum(
            np.diff(time_arr, prepend=time_arr[:1] - 0.05), 1e-6
        )  # (N,)
        v_lnk = (
            np.diff(T_FK[m][:, :3, 3], axis=0, prepend=T_FK[m][:1, :3, 3]) / dt[:, None]
        )  # (N,3) m/s
        w_link_m = 1.0 / (1.0 + (np.linalg.norm(v_lnk, axis=1) / v_link_thresh) ** 2)

        # Step 5: valid — MUST be computed before inv_T_meas fill
        # Hard-gate: both velocity gates must pass independently
        hard_gate = (w_joint >= 0.1) & (w_link_m >= 0.1)
        valid_m = obj_valid_arr[:, m] & valid_telem_arr & hard_gate
        valid[:, m] = valid_m
        if (n_obj_v := int(obj_valid_arr[:, m].sum())) > 0:
            print(f"[handeye debug] {obj_names[m]} ({obj_links_list[m]}): "
                  f"obj_valid={n_obj_v}/{N}  telem={int(valid_telem_arr.sum())}/{N}  "
                  f"wjoint={(obj_valid_arr[:, m] & valid_telem_arr & (w_joint >= 0.1)).sum()}  "
                  f"wlink={(obj_valid_arr[:, m] & valid_telem_arr & (w_link_m >= 0.1)).sum()}  "
                  f"final={int(valid_m.sum())}  "
                  f"w_link_median={np.median(w_link_m[obj_valid_arr[:, m]]):.3f}")

        # Step 6: inv_T_meas — fill identity for invalid frames (NaN safety)
        # obj_pos[invalid] = NaN; 0*NaN = NaN in IEEE 754; eye(4) default prevents propagation
        inv_T_meas_m = np.tile(np.eye(4), (N, 1, 1))
        for i in np.where(valid_m)[0]:
            R = Rotation.from_quat(quat_wxyz_to_xyzw(obj_quat_arr[i, m])).as_matrix()
            T_meas_world = make_transform(R, obj_pos_arr[i, m])
            T_meas_cam = inv_C_approx @ T_meas_world
            inv_T_meas_m[i] = invert_transform(T_meas_cam)
        inv_T_meas.append(inv_T_meas_m)

        # Step 7: weights_sqrt = sqrt(w_combined * valid) / sqrt(N_valid_cm)
        w_inlier = np.clip(n_inliers_arr[:, m], 0, 5) / 5.0
        w_combined = w_joint * w_link_m * w_inlier
        N_valid_cm = int(valid_m.sum())
        norm = np.sqrt(N_valid_cm) if N_valid_cm > 0 else 1.0
        ws = np.sqrt(w_combined) * valid_m.astype(np.float64) / norm
        weights_sqrt.append(ws)

        # Observability checks (warn only; don't fail calibration)
        _check_observability(
            T_FK[m], inv_T_meas_m, valid_m, obj_names[m], obj_links_list[m], C_approx
        )

    return {
        "T_FK": T_FK,            # list of M arrays, each (N,4,4)
        "inv_T_meas": inv_T_meas,  # list of M arrays, each (N,4,4)
        "weights_sqrt": weights_sqrt,  # list of M arrays, each (N,)
        "obj_names": obj_names,
        "obj_links": obj_links_list,
        "cam_port": int(log["cam_port"]),
        "N": N,
    }


def _check_observability(
    T_FK_m: NDArray,
    inv_T_meas_m: NDArray,
    valid_m: NDArray,
    obj_name: str,
    obj_link: str,
    C_approx: NDArray,
) -> None:
    """Warn if FK motion or cube axis angular diversity is insufficient."""
    if valid_m.sum() < 2:
        print(f"[handeye] WARNING: {obj_link} ({obj_name}): fewer than 2 valid frames — "
              f"observability not checkable")
        return

    T_FK_valid = T_FK_m[valid_m]  # (Nv,4,4)
    wrist_pos = T_FK_valid[:, :3, 3]
    trans_span = float(np.linalg.norm(wrist_pos.max(0) - wrist_pos.min(0)))
    R0 = T_FK_valid[0, :3, :3]
    rot_span_deg = float(max(
        np.degrees(np.arccos(np.clip(
            np.trace(R0.T @ T_FK_valid[i, :3, :3]) / 2.0 - 0.5, -1.0, 1.0
        )))
        for i in range(len(T_FK_valid))
    ))
    if rot_span_deg < _OBS_MIN_ROT_DEG:
        print(f"[handeye] WARNING: {obj_link}: FK rotation span {rot_span_deg:.1f}° < "
              f"{_OBS_MIN_ROT_DEG}° — increase arm ROM")
    if trans_span < _OBS_MIN_TRANS_M:
        print(f"[handeye] WARNING: {obj_link}: FK translation span {trans_span*1000:.0f} mm < "
              f"{_OBS_MIN_TRANS_M*1000:.0f} mm — increase arm ROM")

    # Cube z-axis angular span in camera frame (checks optical-axis rotation observability)
    # inv_T_meas_m[valid_m, :3, :3] = R_obj_cam; R_cam_obj = R_obj_cam.T
    R_obj_cam = inv_T_meas_m[valid_m, :3, :3]  # (Nv,3,3) = inv(T_cam_obj)[:3,:3]
    R_cam_obj = R_obj_cam.transpose(0, 2, 1)   # (Nv,3,3)
    z_cam = R_cam_obj[:, :, 2]  # (Nv,3) object z-axis in camera frame
    z_mean = z_cam.mean(0)
    z_norm = np.linalg.norm(z_mean)
    if z_norm < 1e-8:
        return
    z_mean /= z_norm
    cube_span_deg = float(np.degrees(
        np.max(np.arccos(np.clip(np.abs(z_cam @ z_mean), 0.0, 1.0)))
    ))
    if cube_span_deg < _OBS_MIN_CUBE_DEG:
        print(f"[handeye] WARNING: {obj_link}: cube z-axis angular span {cube_span_deg:.1f}° < "
              f"{_OBS_MIN_CUBE_DEG}° — add wrist roll to make cube rotation observable")


# ---------------------------------------------------------------------------
# Joint multi-camera calibrator
# ---------------------------------------------------------------------------

class CameraHandEyeCalibrator:
    """Joint multi-camera nonlinear WLS hand-eye calibrator.

    Optimizes X_c (camera extrinsics, one per camera) and Y_j (cube mounting
    offsets, shared across cameras) simultaneously. Y_j coupling prevents
    divergence that occurs with sequential per-camera solves.

    Residual: r_ij = [t_err; rotvec_err] ∈ ℝ⁶ per (camera, frame, object).
    Weighted by sqrt(w_joint * w_link * w_inlier * valid) / sqrt(N_valid_cm).
    Two-stage solve: linear warm-start → Huber (f_scale=0.01).
    """

    def __init__(self, camera_data: list[dict]):
        self.C = len(camera_data)
        self.obj_names: list[str] = camera_data[0]["obj_names"]
        self.M = len(self.obj_names)
        self._cdata = camera_data

        # Validate all array lengths match within each camera
        for c, cd in enumerate(camera_data):
            N_c = len(cd["T_FK"][0])
            for m in range(self.M):
                assert len(cd["T_FK"][m]) == N_c, \
                    f"cam {c} T_FK[{m}] length {len(cd['T_FK'][m])} ≠ {N_c}"
                assert len(cd["inv_T_meas"][m]) == N_c, \
                    f"cam {c} inv_T_meas[{m}] length mismatch"
                assert len(cd["weights_sqrt"][m]) == N_c, \
                    f"cam {c} weights_sqrt[{m}] length mismatch"

        # Validate obj_names consistent across cameras
        for c, cd in enumerate(camera_data):
            if cd["obj_names"] != self.obj_names:
                raise ValueError(
                    f"camera_data[{c}].obj_names {cd['obj_names']} ≠ {self.obj_names}"
                )

    def _unpack_params(self, params: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        X_cs = [params_to_transform(params[6 * c: 6 * c + 6]) for c in range(self.C)]
        Y_js = [
            params_to_transform(params[6 * self.C + 6 * m: 6 * self.C + 6 * m + 6])
            for m in range(self.M)
        ]
        return X_cs, Y_js

    def _residuals(self, params: NDArray, weighted: bool = True) -> NDArray:
        """Compute stacked residuals [t_err; rotvec_err] for all (camera, object) pairs.

        Vectorized over N frames via einsum. Invalid frames have weights_sqrt=0 → r*0=0.
        """
        X_cs, Y_js = self._unpack_params(params)
        all_r: list[NDArray] = []
        for c, cd in enumerate(self._cdata):
            inv_Xc = invert_transform(X_cs[c])  # (4,4)
            for m in range(self.M):
                # T_pred[i] = inv(X_c) @ F_j(q_i) @ Y_j  — fully vectorized
                T_pred = np.einsum("ij,njk,kl->nil", inv_Xc, cd["T_FK"][m], Y_js[m])
                T_err = np.matmul(cd["inv_T_meas"][m], T_pred)  # (N,4,4)
                t_err = T_err[:, :3, 3]                          # (N,3)
                r_err = _batch_mat_to_rotvec(T_err[:, :3, :3])  # (N,3)
                r = np.concatenate([t_err, r_err], axis=1)      # (N,6)
                if weighted:
                    r *= cd["weights_sqrt"][m][:, None]
                all_r.append(r)
        return np.concatenate(all_r).ravel()

    def _jac_sparsity(self):
        """Jacobian sparsity pattern as csr_matrix.

        Param vector: [X_c0(6)|X_c1(6)|...|Y_0(6)|Y_1(6)|...]
          X_ci cols [6c:6c+6]: nonzero only for camera i rows.
          Y_mj cols [6C+6m:6C+6m+6]: nonzero for ALL cameras' rows for object m.
        """
        N_cs = [len(cd["T_FK"][0]) for cd in self._cdata]
        total_rows = sum(N_c * self.M * 6 for N_c in N_cs)
        n_params = 6 * self.C + 6 * self.M
        S = lil_matrix((total_rows, n_params), dtype=bool)
        row = 0
        for c, N_c in enumerate(N_cs):
            for m in range(self.M):
                block_rows = N_c * 6
                S[row: row + block_rows, 6 * c: 6 * c + 6] = True
                S[row: row + block_rows, 6 * self.C + 6 * m: 6 * self.C + 6 * m + 6] = True
                row += block_rows
        assert S.shape == (total_rows, n_params), \
            f"Jac sparsity shape {S.shape} ≠ ({total_rows}, {n_params})"
        return S.tocsr()

    def solve(
        self,
        X_c_inits: list[NDArray],
        Y_j_inits: dict[str, NDArray],
    ) -> tuple[list[NDArray], list[NDArray], dict]:
        """Two-stage solve: linear warm-start → Huber.

        Never skip the linear warm-start — it ensures R_err is small before
        Huber, validating the decoupled SO(3)×ℝ³ residual approximation.

        Args:
            X_c_inits: list of (4,4) initial camera extrinsics, one per camera.
            Y_j_inits: dict mapping obj_name → (4,4) initial cube mounting offset.

        Returns:
            X_cs_final: list of (4,4) refined camera extrinsics.
            Y_js_final: list of (4,4) refined cube offsets (same order as obj_names).
            metrics: dict with residual statistics (valid frames only).
        """
        def T_to_params(T: NDArray) -> NDArray:
            return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])

        p0 = np.concatenate(
            [T_to_params(X_c_inits[c]) for c in range(self.C)] +
            [T_to_params(Y_j_inits[name]) for name in self.obj_names]
        )

        sparsity = self._jac_sparsity()

        # Stage 1: linear warm-start (fast convergence to correct basin)
        print("[handeye] Stage 1: linear warm-start...")
        r_warm = least_squares(
            self._residuals, p0,
            jac_sparsity=sparsity, method="trf", loss="linear",
            verbose=0,
        )
        print(f"[handeye]   cost={r_warm.cost:.4g}  nfev={r_warm.nfev}")

        # Check rotation residuals on valid frames before entering Huber
        X_cs_w, Y_js_w = self._unpack_params(r_warm.x)
        rot_norms: list[NDArray] = []
        for c_i, cd_i in enumerate(self._cdata):
            inv_Xc_w = invert_transform(X_cs_w[c_i])
            for m_i in range(self.M):
                T_pred_w = np.einsum(
                    "ij,njk,kl->nil", inv_Xc_w, cd_i["T_FK"][m_i], Y_js_w[m_i]
                )
                T_err_w = np.matmul(cd_i["inv_T_meas"][m_i], T_pred_w)
                r_rot_w = _batch_mat_to_rotvec(T_err_w[:, :3, :3])  # (N,3) rad
                vm = cd_i["weights_sqrt"][m_i] > 0                   # valid mask
                if vm.any():
                    rot_norms.append(np.linalg.norm(r_rot_w[vm], axis=1))
        if rot_norms:
            max_rot_deg = float(np.degrees(np.max(np.concatenate(rot_norms))))
            print(f"[handeye]   max rotation residual after warm-start: {max_rot_deg:.1f}°")
            if max_rot_deg > _WARMUP_ROT_WARN_DEG:
                print(
                    f"[handeye] WARNING: max rotation residual {max_rot_deg:.1f}° > "
                    f"{_WARMUP_ROT_WARN_DEG}° threshold. "
                    f"Check obj_rot_inits — cube may be physically twisted relative to "
                    f"wrist MJCF frame. Proceeding to Huber but result may be biased."
                )
        else:
            max_rot_deg = 0.0

        # Stage 2: Huber (robust to outliers)
        print("[handeye] Stage 2: Huber refinement...")
        r_final = least_squares(
            self._residuals, r_warm.x,
            jac_sparsity=sparsity, method="trf",
            loss="huber", f_scale=_HUBER_F_SCALE,
            verbose=0,
        )
        print(f"[handeye]   cost={r_final.cost:.4g}  nfev={r_final.nfev}")

        X_cs_final, Y_js_final = self._unpack_params(r_final.x)
        metrics = self._compute_metrics(X_cs_final, Y_js_final)
        metrics["max_rot_residual_after_warmup_deg"] = max_rot_deg
        return X_cs_final, Y_js_final, metrics

    def _compute_metrics(
        self, X_cs: list[NDArray], Y_js: list[NDArray]
    ) -> dict:
        """Residual stats on valid frames only (weights_sqrt > 0)."""
        trans_errs: list[NDArray] = []
        rot_errs: list[NDArray] = []
        per_obj: dict[str, dict] = {}

        for c, cd in enumerate(self._cdata):
            inv_Xc = invert_transform(X_cs[c])
            for m, name in enumerate(self.obj_names):
                T_pred = np.einsum("ij,njk,kl->nil", inv_Xc, cd["T_FK"][m], Y_js[m])
                T_err = np.matmul(cd["inv_T_meas"][m], T_pred)
                vm = cd["weights_sqrt"][m] > 0
                t_mm = np.linalg.norm(T_err[vm, :3, 3], axis=1) * 1000.0
                r_deg = np.degrees(
                    np.linalg.norm(_batch_mat_to_rotvec(T_err[vm, :3, :3]), axis=1)
                )
                trans_errs.append(t_mm)
                rot_errs.append(r_deg)
                if name not in per_obj:
                    per_obj[name] = {"t_mm": [], "n_valid": 0}
                per_obj[name]["t_mm"].extend(t_mm.tolist())
                per_obj[name]["n_valid"] += int(vm.sum())

        all_t = np.concatenate(trans_errs) if trans_errs else np.array([0.0])
        all_r = np.concatenate(rot_errs) if rot_errs else np.array([0.0])
        n_valid_total = sum(
            int((cd["weights_sqrt"][0] > 0).sum()) for cd in self._cdata
        )
        return {
            "mean_trans_mm": float(all_t.mean()),
            "max_trans_mm": float(all_t.max()),
            "mean_rot_deg": float(all_r.mean()),
            "max_rot_deg": float(all_r.max()),
            "n_valid_total": n_valid_total,
            "per_object": {
                n: {
                    "mean_trans_mm": float(np.mean(v["t_mm"])) if v["t_mm"] else 0.0,
                    "n_valid": v["n_valid"],
                }
                for n, v in per_obj.items()
            },
        }


# ---------------------------------------------------------------------------
# CLI dataclass and main
# ---------------------------------------------------------------------------

_CALIB_DIR = Path(__file__).resolve().parent / "calibration"


def _diagnose(logs: list[dict]) -> None:
    """Print per-camera, per-object gate breakdown from raw NPZ data."""
    for log in logs:
        port = log["cam_port"]
        obj_names = list(log["obj_names"])
        obj_links = list(log["obj_links"])
        N = int(np.asarray(log["obj_valid"]).shape[0])
        valid_telem = np.asarray(log["valid_telem"])
        w_speed = np.asarray(log["w_speed"])
        speed_gate = w_speed >= 0.1
        print(f"\n[diagnose] port {port}  ({N} frames)")
        for m, (name, link) in enumerate(zip(obj_names, obj_links)):
            n_inliers = np.asarray(log["obj_n_inliers"])[:, m]
            detected = n_inliers > 0
            sufficient = n_inliers >= 2  # min_inliers default
            obj_valid = np.asarray(log["obj_valid"])[:, m]
            print(f"  {name} → {link}")
            print(f"    raw detections (n_inliers>0): {int(detected.sum()):4d}/{N}")
            print(f"    sufficient inliers (>=2):     {int(sufficient.sum()):4d}/{N}")
            print(f"    telem valid:                  {int(valid_telem.sum()):4d}/{N}")
            print(f"    speed gate (w>=0.1):          {int(speed_gate.sum()):4d}/{N}")
            print(f"    all gates → obj_valid:        {int(obj_valid.sum()):4d}/{N}")


def _find_latest_logs() -> list[str]:
    """Auto-discover npz files from the most recent recording session in calibration/.

    Groups files by timestamp prefix (calib_YYYYMMDD_HHMMSS), returns all files
    from the newest group. Raises if calibration/ is empty or missing.
    """
    npz_files = sorted(_CALIB_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {_CALIB_DIR}. "
            f"Run: python humanoid_monitor.py --record-handeye"
        )
    # Group by prefix (everything before _port)
    groups: dict[str, list[Path]] = {}
    for p in npz_files:
        prefix = p.stem.rsplit("_port", 1)[0]
        groups.setdefault(prefix, []).append(p)
    latest_prefix = sorted(groups)[-1]
    chosen = [str(p) for p in sorted(groups[latest_prefix])]
    print(f"[handeye] Auto-selected logs (prefix '{latest_prefix}'): {[Path(p).name for p in chosen]}")
    return chosen


@dataclasses.dataclass
class CalibArgs:
    logs: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    # defaults to most recent recording in calibration/; pass explicitly to override
    obj_inits: dict = dataclasses.field(default_factory=lambda: {
        "tag_cube_0": list(_OBJ_INIT_OFFSET_M),  # m, end_effector_L body frame
        "tag_cube_1": list(_OBJ_INIT_OFFSET_M),  # m, end_effector_R body frame
    })
    obj_rot_inits: dict = dataclasses.field(default_factory=dict)
    # e.g. {"tag_cube_0": [0.0, 0.0, 0.3]} — rotvec [rad] if cube is physically twisted
    subsample: int = 5
    dq_thresh: float = 0.5   # rad/s arm joint (dq[13:27]) inf-norm; half-weight at threshold
    v_link_thresh: float = 0.3  # m/s wrist Cartesian speed; half-weight at threshold
    output: str = str(_CALIB_DIR / "camera_handeye.yaml")
    visualize: bool = False
    diagnose: bool = True  # print per-camera per-object gate stats before calibration


def main(args: CalibArgs) -> None:
    # 1. Load logs — auto-discover if not provided
    log_paths = list(args.logs) if args.logs else _find_latest_logs()
    logs: list[dict] = []
    for p in log_paths:
        raw = dict(np.load(p, allow_pickle=True))
        # Convert NpzFile scalars to Python types for mutable indexing
        for k in list(raw.keys()):
            v = raw[k]
            if v.ndim == 0:
                raw[k] = v.item()
        logs.append(raw)
    if not logs:
        raise ValueError("No logs provided")

    if args.diagnose:
        _diagnose(logs)

    # 2. Reconcile obj_names across logs
    canonical_names = _reconcile_logs(logs)

    # 3. FK solver (shared — all logs use same MJCF)
    obj_links = [
        str(logs[0]["obj_links"][list(logs[0]["obj_names"]).index(n)])
        for n in canonical_names
    ]
    fk = CameraFKSolver(obj_links=list(set(obj_links)))  # deduplicate for FK init

    # 4. Preprocess each log → camera_data dict
    camera_data: list[dict] = []
    cam_ports: list[int] = []
    cam_transforms_approx: list[NDArray] = []
    for log in logs:
        cd = _prepare_camera_data(log, fk, args.subsample, args.dq_thresh, args.v_link_thresh)
        camera_data.append(cd)
        cam_ports.append(int(log["cam_port"]))
        cam_transforms_approx.append(np.asarray(log["cam_transform_approx"], dtype=np.float64))

    # 5. Build init transforms
    X_c_inits: list[NDArray] = list(cam_transforms_approx)  # approximate extrinsics
    Y_j_inits: dict[str, NDArray] = {}
    for name in canonical_names:
        t = np.array(args.obj_inits.get(name, [0.0618, 0.0, 0.0]), dtype=np.float64)
        rotvec = np.array(args.obj_rot_inits.get(name, [0.0, 0.0, 0.0]), dtype=np.float64)
        Y_j_inits[name] = make_transform(Rotation.from_rotvec(rotvec).as_matrix(), t)

    # 6. Solve
    calibrator = CameraHandEyeCalibrator(camera_data)
    X_cs, Y_js, metrics = calibrator.solve(X_c_inits, Y_j_inits)

    # 7. Yj rotation from init
    for i, name in enumerate(canonical_names):
        R_init = Y_j_inits[name][:3, :3]
        R_final = Y_js[i][:3, :3]
        q_init = mat_to_quat_xyzw(R_init)
        q_final = mat_to_quat_xyzw(R_final)
        metrics["per_object"][name]["Yj_rot_deg_from_init"] = quat_angle_diff_deg(q_init, q_final)

    # 8. Determine calibration_valid
    valid_flag = True
    reason = ""
    if metrics["n_valid_total"] < 50:
        valid_flag = False
        reason = f"n_valid={metrics['n_valid_total']}<50"
    elif metrics["max_trans_mm"] > 20:
        valid_flag = False
        reason = f"max_trans={metrics['max_trans_mm']:.1f}mm>20"
    elif metrics["max_rot_deg"] > 5:
        valid_flag = False
        reason = f"max_rot={metrics['max_rot_deg']:.1f}°>5"
    for name, pm in metrics["per_object"].items():
        if pm["n_valid"] < 20:
            valid_flag = False
            reason = f"{name} n_valid={pm['n_valid']}<20"

    # 9. Write YAML (always — even on failure)
    out: dict[str, Any] = {
        "calibration_valid": bool(valid_flag),
        "failure_reason": reason,
        "cameras": {
            str(cam_ports[c]): {
                "T_base_link_to_camera": X_cs[c].tolist(),
                "cam_position_m": X_cs[c][:3, 3].tolist(),
                "cam_quaternion_xyzw": mat_to_quat_xyzw(X_cs[c][:3, :3]).tolist(),
            }
            for c in range(len(logs))
        },
        "objects": {
            name: {
                "obj_link": obj_links[i],
                "T_obj_link_to_object": Y_js[i].tolist(),
                "obj_position_m": Y_js[i][:3, 3].tolist(),
                "obj_quaternion_xyzw": mat_to_quat_xyzw(Y_js[i][:3, :3]).tolist(),
            }
            for i, name in enumerate(canonical_names)
        },
        "metrics": {
            "mean_trans_mm": metrics["mean_trans_mm"],
            "max_trans_mm": metrics["max_trans_mm"],
            "mean_rot_deg": metrics["mean_rot_deg"],
            "max_rot_deg": metrics["max_rot_deg"],
            "n_valid_total": metrics["n_valid_total"],
            "max_rot_residual_after_warmup_deg": metrics["max_rot_residual_after_warmup_deg"],
        },
        "per_object_metrics": {
            name: {
                "mean_trans_mm": pm["mean_trans_mm"],
                "n_valid": pm["n_valid"],
                "Yj_rot_deg_from_init": pm["Yj_rot_deg_from_init"],
            }
            for name, pm in metrics["per_object"].items()
        },
    }

    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    status = "VALID" if valid_flag else f"INVALID ({reason})"
    print(f"[handeye] {status}")
    print(f"[handeye] residuals: mean={metrics['mean_trans_mm']:.1f}mm / "
          f"max={metrics['max_trans_mm']:.1f}mm | "
          f"rot mean={metrics['mean_rot_deg']:.2f}° max={metrics['max_rot_deg']:.2f}°")

    # Print approx vs refined comparison per camera
    print("[handeye] camera correction summary:")
    for c, (port, T_approx) in enumerate(zip(cam_ports, cam_transforms_approx)):
        pos_approx = T_approx[:3, 3] * 1000
        q_approx = mat_to_quat_xyzw(T_approx[:3, :3])
        pos_refined = X_cs[c][:3, 3] * 1000
        q_refined = mat_to_quat_xyzw(X_cs[c][:3, :3])
        dpos = pos_refined - pos_approx
        drot_deg = float(np.degrees(
            (Rotation.from_quat(q_approx, scalar_first=False).inv()
             * Rotation.from_quat(q_refined, scalar_first=False)).magnitude()
        ))
        print(f"  port {port}:"
              f"  approx={np.round(pos_approx, 1)} mm"
              f"  refined={np.round(pos_refined, 1)} mm"
              f"  Δpos={np.round(dpos, 1)} mm |Δpos|={np.linalg.norm(dpos):.1f} mm"
              f"  Δrot={drot_deg:.2f}°")

    print(f"[handeye] Written to {args.output}")

    if args.visualize:
        print("[handeye] --visualize not yet implemented")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(CalibArgs))
