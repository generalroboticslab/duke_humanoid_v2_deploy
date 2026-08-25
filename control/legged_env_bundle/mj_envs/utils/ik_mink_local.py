"""Local Mink-compatible primitives used by ik_mink facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import numpy as np
import qpsolvers

# Pre-allocated constant arrays — read-only; never mutate these in-place.
_EYE3 = np.eye(3, dtype=np.float64)
_EYE6 = np.eye(6, dtype=np.float64)

class MinkError(Exception):
    """Base class for Mink-compatible exceptions."""


class TaskDefinitionError(MinkError):
    """Raised when task construction receives invalid arguments."""


class TargetNotSet(MinkError):
    """Raised when compute_* runs before set_target."""

    def __init__(self, task_name: str):
        super().__init__(f"{task_name} target not set")


class InvalidFrame(MinkError):
    """Raised when an unsupported frame_type is requested."""


class InvalidTarget(MinkError):
    """Raised when a target has an invalid shape or value."""


class NoSolutionFound(MinkError):
    """Raised when QP solver fails to find a solution."""


@dataclass(frozen=True)
class _Constraint:
    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None

    @property
    def inactive(self) -> bool:
        return self.G is None and self.h is None


@dataclass(frozen=True)
class _Objective:
    H: np.ndarray
    c: np.ndarray


def _skew(v: np.ndarray) -> np.ndarray:
    wx, wy, wz = v
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=v.dtype)


def _so3_log(wxyz: np.ndarray) -> np.ndarray:
    q = np.array(wxyz, dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    w, v = q[0], q[1:]
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return (2.0 * np.arctan2(norm, w) / norm) * v


def _so3_ljacinv(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    t2 = theta * theta
    if theta < 1e-10:
        beta = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
    else:
        beta = (1.0 / t2) * (1.0 - (theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta)))))
    ljacinv = beta * (np.outer(omega, omega) - t2 * np.eye(3, dtype=np.float64))
    ljacinv -= 0.5 * _skew(omega)
    ljacinv[0, 0] += 1.0
    ljacinv[1, 1] += 1.0
    ljacinv[2, 2] += 1.0
    return ljacinv


def _se3_Q(c: np.ndarray) -> np.ndarray:
    """Q matrix for the SE3 left Jacobian block (Chirikjian Eqn. 180)."""
    theta = float(np.linalg.norm(c[3:]))
    t2 = theta * theta
    A = 0.5
    if t2 < 1e-10:
        B = 1.0 / 6.0 + t2 / 120.0
        C = -1.0 / 24.0 + t2 / 720.0
        D = -1.0 / 60.0
    else:
        t4 = t2 * t2
        s = np.sin(theta)
        co = np.cos(theta)
        B = (theta - s) / (t2 * theta)
        C = (1.0 - 0.5 * t2 - co) / t4
        D = (2.0 * theta - 3.0 * s + theta * co) / (2.0 * t4 * theta)
    V = _skew(c[:3])
    W = _skew(c[3:])
    VW = V @ W
    WV = VW.T  # = W @ V because skew matrices are antisymmetric
    WVW = WV @ W
    VWW = VW @ W
    return A * V + B * (WV + VW + WVW) - C * (VWW - VWW.T - 3.0 * WVW) + D * (WVW @ W + W @ WVW)


def _se3_log(wxyz_xyz: np.ndarray) -> np.ndarray:
    """SE3 log map: returns 6-vector [v, omega] body twist."""
    omega = _so3_log(wxyz_xyz[:4])
    theta = float(np.linalg.norm(omega))
    t2 = theta * theta
    skew_omega = _skew(omega)
    skew_omega2 = skew_omega @ skew_omega
    if t2 < 1e-10:
        vinv = np.eye(3, dtype=np.float64) - 0.5 * skew_omega + skew_omega2 / 12.0
    else:
        half_theta = 0.5 * theta
        vinv = (
            np.eye(3, dtype=np.float64)
            - 0.5 * skew_omega
            + (1.0 - 0.5 * theta * np.cos(half_theta) / np.sin(half_theta)) / t2 * skew_omega2
        )
    tangent = np.empty(6, dtype=np.float64)
    tangent[:3] = vinv @ wxyz_xyz[4:]
    tangent[3:] = omega
    return tangent


def _se3_ljacinv(tangent: np.ndarray) -> np.ndarray:
    """6×6 SE3 left Jacobian inverse."""
    omega = tangent[3:]
    theta_sq = float(np.dot(omega, omega))
    if theta_sq < 1e-10:
        return _EYE6  # read-only; callers only use result in matrix multiply
    Q = _se3_Q(tangent)
    ljacinv_so3 = _so3_ljacinv(omega)
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = ljacinv_so3
    out[:3, 3:] = -ljacinv_so3 @ Q @ ljacinv_so3
    out[3:, 3:] = ljacinv_so3
    return out


@dataclass(frozen=True)
class SE3:
    """SE3 Lie group element (wxyz quaternion + xyz translation).

    Tangent convention: [v_x, v_y, v_z, omega_x, omega_y, omega_z].
    Matches the mink.lie.SE3 API used by FrameTask.
    """

    wxyz_xyz: np.ndarray

    def __post_init__(self) -> None:
        if self.wxyz_xyz.shape != (7,):
            raise ValueError(f"Expected wxyz_xyz shape (7,), got {self.wxyz_xyz.shape}")

    def copy(self) -> "SE3":
        return SE3(np.array(self.wxyz_xyz, dtype=np.float64, copy=True))

    def multiply(self, other: "SE3") -> "SE3":
        out = np.empty(7, dtype=np.float64)
        mujoco.mju_mulQuat(out[:4], self.wxyz_xyz[:4], other.wxyz_xyz[:4])
        mujoco.mju_rotVecQuat(out[4:], other.wxyz_xyz[4:], self.wxyz_xyz[:4])
        out[4:] += self.wxyz_xyz[4:]
        return SE3(out)

    def __matmul__(self, other: "SE3") -> "SE3":
        return self.multiply(other)

    def inverse(self) -> "SE3":
        out = np.empty(7, dtype=np.float64)
        mujoco.mju_negQuat(out[:4], self.wxyz_xyz[:4])
        mujoco.mju_rotVecQuat(out[4:], -self.wxyz_xyz[4:], out[:4])
        return SE3(out)

    def log(self) -> np.ndarray:
        return _se3_log(self.wxyz_xyz)

    def jlog(self) -> np.ndarray:
        """Right Jacobian inverse of log(self): rjacinv(log) = ljacinv(-log)."""
        return _se3_ljacinv(-self.log())

    def minus(self, other: "SE3") -> np.ndarray:
        """Right-minus: (other.inverse() @ self).log() — body-frame twist to other."""
        return (other.inverse() @ self).log()

    def rminus(self, other: "SE3") -> np.ndarray:
        """Alias for minus (official Mink naming)."""
        return self.minus(other)

    def adjoint(self) -> np.ndarray:
        """6x6 adjoint matrix mapping twists by this transform."""
        R = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R, self.wxyz_xyz[:4])
        R = R.reshape(3, 3)
        t_skew = _skew(self.wxyz_xyz[4:])
        adj = np.zeros((6, 6), dtype=np.float64)
        adj[:3, :3] = R
        adj[:3, 3:] = t_skew @ R
        adj[3:, 3:] = R
        return adj


class Configuration:
    """MuJoCo configuration wrapper matching Mink API used by this file."""

    def __init__(self, model: mujoco.MjModel, q: Optional[np.ndarray] = None):
        self.model = model
        self.data = mujoco.MjData(model)
        self._jac_cache: dict = {}
        self._xfm_cache: dict = {}
        self._eye_nv = np.eye(model.nv, dtype=np.float64)
        self.update(q=q)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos

    @property
    def nv(self) -> int:
        return self.model.nv

    def update(self, q: Optional[np.ndarray] = None) -> None:
        if q is not None:
            self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        if self.model.neq > 0:
            mujoco.mj_makeConstraint(self.model, self.data)
        self._jac_cache.clear()
        self._xfm_cache.clear()

    def check_limits(self, safety_break: bool = False, tol: float = 1e-6) -> None:
        for jnt in range(self.model.njnt):
            if self.model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if not self.model.jnt_limited[jnt]:
                continue
            qaddr = self.model.jnt_qposadr[jnt]
            qval = self.q[qaddr]
            qmin, qmax = self.model.jnt_range[jnt]
            if qval < qmin - tol or qval > qmax + tol:
                if safety_break:
                    name = self.model.joint(jnt).name
                    raise ValueError(
                        f"Joint {jnt} ({name}) violates limits {qmin} <= {qval} <= {qmax}"
                    )

    def get_transform_frame_to_world(self, frame_name: str, frame_type: str) -> SE3:
        if frame_type != "body":
            raise ValueError(f"Unsupported frame_type '{frame_type}', expected 'body'")
        cached = self._xfm_cache.get(frame_name)
        if cached is not None:
            return cached
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if bid == -1:
            raise ValueError(f"Body '{frame_name}' not found")
        wxyz_xyz = np.empty(7, dtype=np.float64)
        wxyz_xyz[:4] = self.data.xquat[bid]
        wxyz_xyz[4:] = self.data.xpos[bid]
        result = SE3(wxyz_xyz)
        self._xfm_cache[frame_name] = result
        return result

    def get_transform(
        self,
        source_name: str,
        source_type: str,
        dest_name: str,
        dest_type: str,
    ) -> SE3:
        """Pose of `source` expressed in `dest` frame (body frames only here)."""
        cache_key = (source_name, dest_name)
        cached = self._xfm_cache.get(cache_key)
        if cached is not None:
            return cached
        T_src_w = self.get_transform_frame_to_world(source_name, source_type)
        T_dest_w = self.get_transform_frame_to_world(dest_name, dest_type)
        result = T_dest_w.inverse() @ T_src_w
        self._xfm_cache[cache_key] = result
        return result

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> np.ndarray:
        if frame_type != "body":
            raise ValueError(f"Unsupported frame_type '{frame_type}', expected 'body'")
        cached = self._jac_cache.get(frame_name)
        if cached is not None:
            return cached
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if bid == -1:
            raise ValueError(f"Body '{frame_name}' not found")
        jac_pos = np.empty((3, self.model.nv), dtype=np.float64)
        jac_rot = np.empty((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, bid)
        jac = np.empty((6, self.model.nv), dtype=np.float64)
        jac[:3] = jac_pos
        jac[3:] = jac_rot
        # Convert world-aligned Jacobian to body frame via adjoint of R_fw.
        R_wf = self.data.xmat[bid].reshape(3, 3)
        R_fw = R_wf.T
        jac_body = np.empty_like(jac)
        jac_body[:3] = R_fw @ jac[:3]
        jac_body[3:] = R_fw @ jac[3:]
        self._jac_cache[frame_name] = jac_body
        return jac_body


class _BaseTask:
    def compute_qp_objective(self, configuration: Configuration) -> _Objective:
        raise NotImplementedError


class _Task(_BaseTask):
    def __init__(self, cost: np.ndarray, gain: float = 1.0, lm_damping: float = 0.0):
        if not 0.0 <= gain <= 1.0:
            raise ValueError("gain must be in [0, 1]")
        if lm_damping < 0.0:
            raise ValueError("lm_damping must be >= 0")
        self.cost = cost
        self.gain = gain
        self.lm_damping = lm_damping
        self._wj_work: Optional[np.ndarray] = None
        self._we_work: Optional[np.ndarray] = None
        self._h_work: Optional[np.ndarray] = None
        self._c_work: Optional[np.ndarray] = None

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        raise NotImplementedError

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        raise NotImplementedError

    def _assemble_qp(self, error: np.ndarray, jacobian: np.ndarray, eye_nv: np.ndarray) -> _Objective:
        rows, nv = jacobian.shape
        if self._wj_work is None or self._wj_work.shape != (rows, nv):
            self._wj_work = np.empty((rows, nv), dtype=np.float64)
            self._h_work = np.empty((nv, nv), dtype=np.float64)
            self._c_work = np.empty(nv, dtype=np.float64)
        if self._we_work is None or self._we_work.shape != (rows,):
            self._we_work = np.empty(rows, dtype=np.float64)

        WJ = self._wj_work
        We = self._we_work
        H = self._h_work
        c = self._c_work

        np.multiply(error, -self.gain, out=We)
        np.multiply(We, self.cost, out=We)
        np.multiply(jacobian, self.cost[:, None], out=WJ)

        mu = self.lm_damping * float(We @ We)
        np.matmul(WJ.T, WJ, out=H)
        if mu > 0.0:
            H += mu * eye_nv
        np.matmul(We, WJ, out=c)
        np.negative(c, out=c)
        return _Objective(H=H, c=c)

    def compute_qp_objective(self, configuration: Configuration) -> _Objective:
        return self._assemble_qp(
            self.compute_error(configuration),
            self.compute_jacobian(configuration),
            configuration._eye_nv,
        )


class FrameTask(_Task):
    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        position_cost,
        orientation_cost,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(6, dtype=np.float64), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.transform_target_to_world: Optional[SE3] = None
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost) -> None:
        position_cost = np.atleast_1d(position_cost).astype(np.float64)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise ValueError(f"position_cost must have shape (1,) or (3,), got {position_cost.shape}")
        if np.any(position_cost < 0.0):
            raise ValueError("position_cost must be >= 0")
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost) -> None:
        orientation_cost = np.atleast_1d(orientation_cost).astype(np.float64)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise ValueError(
                f"orientation_cost must have shape (1,) or (3,), got {orientation_cost.shape}"
            )
        if np.any(orientation_cost < 0.0):
            raise ValueError("orientation_cost must be >= 0")
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_world: SE3) -> None:
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self.set_target(configuration.get_transform_frame_to_world(self.frame_name, self.frame_type))

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_world is None:
            raise ValueError("FrameTask target not set")
        T_frame = configuration.get_transform_frame_to_world(self.frame_name, self.frame_type)
        return self.transform_target_to_world.minus(T_frame)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_world is None:
            raise ValueError("FrameTask target not set")
        jac = configuration.get_frame_jacobian(self.frame_name, self.frame_type)
        T_frame = configuration.get_transform_frame_to_world(self.frame_name, self.frame_type)
        T_tb = self.transform_target_to_world.inverse() @ T_frame
        return -T_tb.jlog() @ jac


class RelativeFrameTask(_Task):
    """Regulate `frame` pose relative to `root` frame (body frames only).

    Mirrors official `mink.RelativeFrameTask`. Target is stored in the root frame,
    so the task is invariant to root motion (what a base_link-native EE target needs).
    """

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        root_name: str,
        root_type: str,
        position_cost,
        orientation_cost,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(6, dtype=np.float64), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.root_name = root_name
        self.root_type = root_type
        self.transform_target_to_root: Optional[SE3] = None
        self._transform_target_to_root_inv: Optional[SE3] = None
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost) -> None:
        position_cost = np.atleast_1d(position_cost).astype(np.float64)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"position_cost must have shape (1,) or (3,), got {position_cost.shape}"
            )
        if np.any(position_cost < 0.0):
            raise TaskDefinitionError("position_cost must be >= 0")
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost) -> None:
        orientation_cost = np.atleast_1d(orientation_cost).astype(np.float64)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"orientation_cost must have shape (1,) or (3,), got {orientation_cost.shape}"
            )
        if np.any(orientation_cost < 0.0):
            raise TaskDefinitionError("orientation_cost must be >= 0")
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_root: SE3) -> None:
        self.transform_target_to_root = transform_target_to_root.copy()
        self._transform_target_to_root_inv = self.transform_target_to_root.inverse()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self.set_target(
            configuration.get_transform(
                self.frame_name, self.frame_type, self.root_name, self.root_type
            )
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self._transform_target_to_root_inv is None:
            raise TargetNotSet(self.__class__.__name__)
        T_fr = configuration.get_transform(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        return (self._transform_target_to_root_inv @ T_fr).log()

    def _compute_error_jacobian(self, configuration: Configuration) -> tuple[np.ndarray, np.ndarray]:
        if self._transform_target_to_root_inv is None:
            raise TargetNotSet(self.__class__.__name__)
        J_frame = configuration.get_frame_jacobian(self.frame_name, self.frame_type)
        J_root = configuration.get_frame_jacobian(self.root_name, self.root_type)
        T_fr = configuration.get_transform(
            self.frame_name, self.frame_type, self.root_name, self.root_type
        )
        T_ft = self._transform_target_to_root_inv @ T_fr
        err = T_ft.log()
        jac = T_ft.jlog() @ (J_frame - T_fr.inverse().adjoint() @ J_root)
        return err, jac

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        _, jac = self._compute_error_jacobian(configuration)
        return jac

    def compute_qp_objective(self, configuration: Configuration) -> _Objective:
        err, jac = self._compute_error_jacobian(configuration)
        return self._assemble_qp(err, jac, configuration._eye_nv)


class PostureTask(_Task):
    def __init__(self, model: mujoco.MjModel, cost, gain: float = 1.0, lm_damping: float = 0.0):
        self.model = model
        self.nq = model.nq
        self.target_q: Optional[np.ndarray] = None
        super().__init__(cost=np.zeros(model.nv, dtype=np.float64), gain=gain, lm_damping=lm_damping)
        self.set_cost(cost)
        self._free_v_ids = []
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                vadr = model.jnt_dofadr[j]
                self._free_v_ids.extend(range(vadr, vadr + 6))
        # Pre-allocated identity Jacobian — returned read-only from compute_jacobian.
        self._J_eye = np.eye(model.nv, dtype=np.float64)
        if self._free_v_ids:
            self._J_eye[:, self._free_v_ids] = 0.0

    def set_cost(self, cost) -> None:
        cost = np.atleast_1d(cost).astype(np.float64)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.model.nv):
            raise ValueError(f"PostureTask cost must have shape (1,) or ({self.model.nv},), got {cost.shape}")
        if np.any(cost < 0.0):
            raise ValueError("PostureTask cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target_q) -> None:
        target_q = np.atleast_1d(target_q).astype(np.float64)
        if target_q.shape != (self.nq,):
            raise ValueError(f"Expected target posture shape ({self.nq},), got {target_q.shape}")
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.target_q is None:
            raise ValueError("PostureTask target not set")
        qvel = np.empty(configuration.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            m=configuration.model,
            qvel=qvel,
            dt=1.0,
            qpos1=self.target_q,
            qpos2=configuration.q,
        )
        if self._free_v_ids:
            qvel[self._free_v_ids] = 0.0
        return qvel

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        return self._J_eye


class DofFreezingTask(_Task):
    def __init__(self, model: mujoco.MjModel, dof_indices: list[int], gain: float = 1.0):
        if not dof_indices:
            raise ValueError("DofFreezingTask requires at least one DOF index")
        for idx in dof_indices:
            if idx < 0 or idx >= model.nv:
                raise ValueError(f"DOF index {idx} out of range [0, {model.nv})")
        if len(dof_indices) != len(set(dof_indices)):
            raise ValueError(f"Duplicate DOF indices found: {dof_indices}")
        self.dof_indices = sorted(dof_indices)
        super().__init__(cost=np.ones(len(self.dof_indices), dtype=np.float64), gain=gain, lm_damping=0.0)
        self._error = np.zeros(len(self.dof_indices), dtype=np.float64)
        self._jacobian = np.zeros((len(self.dof_indices), model.nv), dtype=np.float64)
        for i, dof_idx in enumerate(self.dof_indices):
            self._jacobian[i, dof_idx] = 1.0

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        return self._error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        return self._jacobian


class _Limit:
    def compute_qp_inequalities(self, configuration: Configuration, dt: float) -> _Constraint:
        raise NotImplementedError


class ConfigurationLimit(_Limit):
    def __init__(
        self,
        model: mujoco.MjModel,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        if not 0.0 < gain <= 1.0:
            raise ValueError("ConfigurationLimit gain must be in (0, 1]")
        lower = np.full(model.nq, -mujoco.mjMAXVAL, dtype=np.float64)
        upper = np.full(model.nq, mujoco.mjMAXVAL, dtype=np.float64)
        index_list: list[int] = []
        for jnt in range(model.njnt):
            jtype = model.jnt_type[jnt]
            if jtype == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[jnt]:
                continue
            padr = model.jnt_qposadr[jnt]
            lower[padr] = model.jnt_range[jnt, 0] + min_distance_from_limits
            upper[padr] = model.jnt_range[jnt, 1] - min_distance_from_limits
            vadr = model.jnt_dofadr[jnt]
            index_list.append(vadr)
        self.model = model
        self.gain = gain
        self.lower = lower
        self.upper = upper
        self.indices = np.array(index_list, dtype=np.int64)
        self.projection_matrix = np.eye(model.nv, dtype=np.float64)[self.indices] if len(index_list) > 0 else None
        self._G = (
            np.vstack([self.projection_matrix, -self.projection_matrix])
            if self.projection_matrix is not None
            else None
        )
        if len(self.indices) > 0:
            self._h_work = np.empty(2 * len(self.indices), dtype=np.float64)
            self._dq_max_work = np.empty((self.model.nv,), dtype=np.float64)
            self._dq_min_work = np.empty((self.model.nv,), dtype=np.float64)
        else:
            self._h_work = None
            self._dq_max_work = None
            self._dq_min_work = None

    def compute_qp_inequalities(self, configuration: Configuration, dt: float) -> _Constraint:
        del dt
        if self.projection_matrix is None or self._G is None:
            return _Constraint()
        dq_max = self._dq_max_work
        dq_min = self._dq_min_work
        mujoco.mj_differentiatePos(self.model, dq_max, 1.0, configuration.q, self.upper)
        mujoco.mj_differentiatePos(self.model, dq_min, 1.0, self.lower, configuration.q)
        h = self._h_work
        n = len(self.indices)
        idx = self.indices
        np.multiply(dq_max[idx], self.gain, out=h[:n])
        np.multiply(dq_min[idx], self.gain, out=h[n:])
        return _Constraint(G=self._G, h=h)


class VelocityLimit(_Limit):
    def __init__(self, model: mujoco.MjModel, velocities=None):
        if velocities is None:
            velocities = {}
        index_list: list[int] = []
        limit_list: list[float] = []
        for joint_name, max_vel in velocities.items():
            jid = model.joint(joint_name).id
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Free joint {joint_name} not supported")
            vadr = model.jnt_dofadr[jid]
            max_vel = np.atleast_1d(max_vel).astype(np.float64)
            if max_vel.shape != (1,):
                raise ValueError(f"Joint {joint_name} velocity limit shape must be (1,), got {max_vel.shape}")
            index_list.append(vadr)
            limit_list.append(float(max_vel[0]))
        self.indices = np.array(index_list, dtype=np.int64)
        self.limit = np.array(limit_list, dtype=np.float64)
        self.projection_matrix = np.eye(model.nv, dtype=np.float64)[self.indices] if len(index_list) > 0 else None
        self._G = (
            np.vstack([self.projection_matrix, -self.projection_matrix])
            if self.projection_matrix is not None
            else None
        )
        self._h_work = np.empty(2 * len(self.indices), dtype=np.float64) if len(self.indices) > 0 else None
        self._dt_limit_work = np.empty(len(self.indices), dtype=np.float64) if len(self.indices) > 0 else None

    def compute_qp_inequalities(self, configuration: Configuration, dt: float) -> _Constraint:
        del configuration
        if self._G is None:
            return _Constraint()
        h = self._h_work
        dt_limit = self._dt_limit_work
        np.multiply(self.limit, dt, out=dt_limit)
        h[: len(self.indices)] = dt_limit
        h[len(self.indices) :] = dt_limit
        return _Constraint(G=self._G, h=h)


class CollisionAvoidanceLimit(_Limit):
    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)
        # Pre-allocated working buffers — filled in-place each call.
        self._G_work = np.zeros((self.max_num_contacts, model.nv), dtype=np.float64)
        self._h_work = np.empty(self.max_num_contacts, dtype=np.float64)
        self._cdd_tol = collision_detection_distance * (1.0 - 1e-9)

    def _homogenize_geom_id_list(self, geom_list) -> list[int]:
        out: list[int] = []
        for g in geom_list:
            out.append(int(g) if isinstance(g, int) else int(self.model.geom(g).id))
        return out

    def _construct_geom_id_pairs(self, geom_pairs):
        # No contype/conaffinity re-check here (unlike upstream mink): geom_pairs is already
        # an explicitly hand-curated caller list (see HUMANOID_V21_MINK_COLLISION_PAIRS), not
        # an auto-discovered "avoid everything" set. Re-applying a physics-contact affinity
        # gate on top silently drops any pair involving a physics-inert geom (contype=0,
        # conaffinity=0 -- e.g. the gripper's *_ikproxy* proxies), even though the caller
        # explicitly asked for it to be monitored. Confirmed upstream mink has the same gate
        # (mink/limits/collision_avoidance_limit.py:_is_pass_contype_conaffinity_check) and
        # would silently drop the same pairs -- not fixable there since it's a third-party
        # package, and giving ikproxy geoms real contype/conaffinity bits isn't safe either:
        # BatchedMinkIK runs on the live sim model, so nonzero bits would also make these
        # oversized proxy capsules generate real (wrong) contact forces during simulation.
        pairs = []
        for pair in geom_pairs:
            ga = list(set(self._homogenize_geom_id_list(pair[0])))
            gb = list(set(self._homogenize_geom_id_list(pair[1])))
            for a in ga:
                for b in gb:
                    if a == b:
                        continue
                    ba = self.model.geom_bodyid[a]
                    bb = self.model.geom_bodyid[b]
                    if self.model.body_weldid[ba] == self.model.body_weldid[bb]:
                        continue
                    pairs.append((min(a, b), max(a, b)))
        return list(set(pairs))

    def compute_qp_inequalities(self, configuration: Configuration, dt: float) -> _Constraint:
        if self.max_num_contacts == 0:
            return _Constraint()
        # Reuse pre-allocated buffers; reset in-place.
        G = self._G_work
        h = self._h_work
        G[:] = 0.0
        h[:] = np.inf
        fromto = np.empty(6, dtype=np.float64)
        for idx, (g1, g2) in enumerate(self.geom_id_pairs):
            dist = mujoco.mj_geomDistance(
                self.model,
                configuration.data,
                g1,
                g2,
                self.collision_detection_distance,
                fromto,
            )
            # Fast scalar comparison — np.isclose on a scalar has huge Python overhead
            # (triggers full broadcasting machinery).
            if dist >= self._cdd_tol:
                continue
            p1 = fromto[:3]
            p2 = fromto[3:]
            normal = p2 - p1
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                continue
            normal /= normal_norm
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            jac1 = np.empty((3, self.model.nv), dtype=np.float64)
            jac2 = np.empty((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(self.model, configuration.data, jac1, None, p1, b1)
            mujoco.mj_jac(self.model, configuration.data, jac2, None, p2, b2)
            jac_n = normal @ (jac2 - jac1)
            if dist > self.minimum_distance_from_collisions:
                gap = dist - self.minimum_distance_from_collisions
                h[idx] = self.gain * gap / dt + self.bound_relaxation
            else:
                h[idx] = self.bound_relaxation
            sign = -1.0 if dist >= 0.0 else 1.0
            G[idx] = sign * jac_n
        return _Constraint(G=G, h=h)


def _get_qp_objective_workspace(configuration: Configuration) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    nv = configuration.model.nv
    ws = getattr(configuration, "_qp_objective_ws", None)
    if ws is None or ws["H"].shape != (nv, nv):
        ws = {
            "H": np.empty((nv, nv), dtype=np.float64),
            "c": np.empty(nv, dtype=np.float64),
            "diag": np.diag_indices(nv),
        }
        setattr(configuration, "_qp_objective_ws", ws)
    return ws["H"], ws["c"], ws["diag"]


def _compute_qp_objective(configuration: Configuration, tasks: Sequence[_BaseTask], damping: float) -> _Objective:
    H, c, diag = _get_qp_objective_workspace(configuration)
    if not tasks:
        H.fill(0.0)
        c.fill(0.0)
        if damping != 0.0:
            H[diag] = damping
        return _Objective(H=H, c=c)

    first = tasks[0].compute_qp_objective(configuration)
    np.copyto(H, first.H)
    np.copyto(c, first.c)

    for task in tasks[1:]:
        obj = task.compute_qp_objective(configuration)
        H += obj.H
        c += obj.c

    if damping != 0.0:
        H[diag] += damping
    return _Objective(H=H, c=c)


def _compute_qp_inequalities(
    configuration: Configuration,
    limits: Optional[Sequence[_Limit]],
    dt: float,
):
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]
    if len(limits) == 1:
        ineq = limits[0].compute_qp_inequalities(configuration, dt)
        if ineq.inactive:
            return None, None
        return ineq.G, ineq.h

    active_ineqs = []
    total_rows = 0
    for limit in limits:
        ineq = limit.compute_qp_inequalities(configuration, dt)
        if not ineq.inactive:
            active_ineqs.append(ineq)
            total_rows += ineq.G.shape[0]

    if not active_ineqs:
        return None, None

    nv = configuration.model.nv
    G = np.empty((total_rows, nv), dtype=np.float64)
    h = np.empty(total_rows, dtype=np.float64)
    row = 0
    for ineq in active_ineqs:
        n = ineq.G.shape[0]
        G[row : row + n, :] = ineq.G
        h[row : row + n] = ineq.h
        row += n
    return G, h


def _compute_qp_equalities(
    configuration: Configuration,
    constraints: Optional[Sequence[_Task]],
):
    if not constraints:
        return None, None
    if len(constraints) == 1:
        task = constraints[0]
        if isinstance(task, DofFreezingTask) and task.gain == 1.0:
            return task.compute_jacobian(configuration), task.compute_error(configuration)
        return task.compute_jacobian(configuration), -task.gain * task.compute_error(configuration)

    rows = []
    total_rows = 0
    for task in constraints:
        J = task.compute_jacobian(configuration)
        feedback = -task.gain * task.compute_error(configuration)
        rows.append((J, feedback))
        total_rows += J.shape[0]

    nv = configuration.model.nv
    A = np.empty((total_rows, nv), dtype=np.float64)
    b = np.empty(total_rows, dtype=np.float64)
    row = 0
    for J, feedback in rows:
        n = J.shape[0]
        A[row : row + n, :] = J
        b[row : row + n] = feedback
        row += n
    return A, b


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[_BaseTask],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[_Limit]] = None,
    constraints: Optional[Sequence[_Task]] = None,
    **kwargs,
) -> np.ndarray:
    # Only check limits when safety_break=True; the no-op path (safety_break=False)
    # was the #2 profiler hotspot (0.028s / 200 solves) — skipping it is safe since
    # BatchedMinkIK already hard-clamps arm joints after every solve.
    if safety_break:
        configuration.check_limits(safety_break=True)
    obj = _compute_qp_objective(configuration, tasks, damping)
    G, h = _compute_qp_inequalities(configuration, limits, dt)
    A, b = _compute_qp_equalities(configuration, constraints)
    delta_q = qpsolvers.solve_qp(obj.H, obj.c, G, h, A, b, solver=solver, **kwargs)
    if delta_q is None:
        raise NoSolutionFound(f"QP solver {solver} failed to find a solution.")
    return delta_q / dt


__all__ = [
    "MinkError",
    "TaskDefinitionError",
    "TargetNotSet",
    "InvalidFrame",
    "InvalidTarget",
    "NoSolutionFound",
    "SE3",
    "Configuration",
    "FrameTask",
    "RelativeFrameTask",
    "PostureTask",
    "DofFreezingTask",
    "ConfigurationLimit",
    "VelocityLimit",
    "CollisionAvoidanceLimit",
    "solve_ik",
]
