"""Hardware abstraction layer for the Duke Humanoid V2.

Provides HumanoidBase: motor + IMU + MuJoCo gravity compensation, no policy config needed.
Use this directly from standalone scripts (arm teach/replay, calibration, etc.).
HumanoidRealEnv inherits from this and adds the policy/obs/IK/Vicon stack.

Design decisions:
  - No YAML config read. All parameters are explicit constructor args or module-level constants.
  - torch throughout (not numpy) so callers can do tensor math without per-call conversion.
  - Module-level URDF_JOINT_NAMES parsed once on import (~50 ms); shared across all instances.
  - control_freq defaults to 200 Hz (low-level CAN rate); HumanoidRealEnv overrides after super().
  - prepare() takes explicit target_pos/kp/kd so the base class stays config-agnostic.
"""

import time
import numpy as np
import torch
import mujoco

from hardware_bindings.motor.py_motor import CanMotorController, FakeMotorController, R00, R01, R02, R03, R04, R05, R06
from hardware_bindings.imu.py_imu import IMU, FakeIMU
from humanoid_config import motor_setup
from humanoid_utils import get_joint_info_from_mjcf, make_async_logger

_log = make_async_logger("humanoid_base")

# ---------------------------------------------------------------------------
# Module-level constants (moved from humanoid_real_env.py)
# Parsed once on import; shared by all importers — do not move into __init__.
# ---------------------------------------------------------------------------

# Operator-set, paired with motor_setup (humanoid_config.py). Must yield the same
# joint count+order as motor_setup. Shared with the workstation viewers and the
# hand-eye solver via humanoid_model.py — edit the path THERE, not here.
# Re-exported for existing importers (humanoid_real_env.py etc.).
from humanoid_model import MJCF_MODEL_PATH

# Per-motor torque correction ratios (empirically calibrated).
# R03 runs slightly hot; scale down to match commanded torque.
MOTOR_TORQUE_CORRECTION_RATIO = {
    R00: 1.0,
    R01: 1.0,
    R02: 1.0,
    R03: 0.9,   # calibrated lower torque — measured output exceeds command
    R04: 0.95,
    R05: 1.0,
    R06: 1.0,
}


URDF_JOINT_NAMES, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS = get_joint_info_from_mjcf(MJCF_MODEL_PATH)

# Joint group slices (based on motor_setup_dict ordering in humanoid_config.py)
_MOTOR_LEG_SLICE = slice(0, 13)   # waist (0) + left leg (1-6) + right leg (7-12)
_MOTOR_ARM_SLICE = slice(13, 27)  # left arm (13-19) + right arm (20-26)
_MOTOR_CAMERA_SLICE = slice(27, 31)  # cam yaw/pitch left (27-28) + right (29-30); see cam_* in motor_setup_dict

# enable_motor mode -> motor slices to enable. "true"/"false" handled separately.
_MOTOR_ENABLE_SLICES = {
    "leg":    (_MOTOR_LEG_SLICE,),
    "arm":    (_MOTOR_ARM_SLICE,),
    "camera": (_MOTOR_CAMERA_SLICE,),
    # bench reach: policy-driven gaze + IK arms with the legs limp (hanging robot)
    "arm_camera": (_MOTOR_ARM_SLICE, _MOTOR_CAMERA_SLICE),
}


class HumanoidBase:
    """Motor + IMU + MuJoCo gravity compensation — no policy config required.

    Initialization order (invariant — see plan/arm_teach_replay.md):
      motor create → torque_correction_ratio → IMU → MuJoCo → gravity attrs
      → gravity comp warmup → IMU validate → motor enable/start

    Attributes set here and available to subclasses:
      device, torque_limit, use_fake, n_joints
      control_freq, dt  (default 200 Hz; HumanoidRealEnv overrides after super())
      motor, imu
      torque_correction_ratio  (torch, cuda)
      mj_model, mj_data, root_body_id
      gravity_comp_scale, gravity_torque_raw, gravity_torque_applied
      gravity_compensation_enabled
    """

    def __init__(
        self,
        torque_limit: float = 0.1,
        use_fake: bool = False,
        gravity_compensation_enabled: bool = False,
        add_right_ee: bool = False,
        enable_motor: str | bool = True, # "true"/True=all joints, "leg"=legs+waist only, "arm"=arms only, "camera"=cam yaw/pitch only, "false"/False=none (passive)
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torque_limit = torque_limit
        self.use_fake = use_fake
        self.n_joints = len(URDF_JOINT_NAMES)
        if isinstance(enable_motor, bool):
            enable_motor = "true" if enable_motor else "false"
        self.enable_motor = enable_motor  # "true" | "false" | "leg" | "arm" | "camera"

        # Default to low-level CAN rate; HumanoidRealEnv overrides from config after super()
        self.control_freq = 200
        self.dt = 1.0 / self.control_freq

        # --- Motor ---
        if use_fake:
            self.motor = FakeMotorController(motor_setup)
            _log.info("[BASE] Using FAKE motor controller")
        else:
            self.motor = CanMotorController(motor_setup)

        self.torque_correction_ratio = torch.tensor(
            [MOTOR_TORQUE_CORRECTION_RATIO[t] for t in self.motor.motor_types],
            dtype=torch.float32, device=self.device,
        )

        # --- IMU ---
        if use_fake:
            self.imu = FakeIMU()
            _log.info("[BASE] Using FAKE IMU")
        else:
            self.imu = IMU()

        # --- MuJoCo (gravity compensation) ---
        if add_right_ee:
            # Build spec with EE cube so RNE accounts for its mass.
            _spec = mujoco.MjSpec.from_file(MJCF_MODEL_PATH)
            _geom = _spec.body("wrist_3_R").add_geom()
            _geom.name = "ee_cube"
            _geom.type = mujoco.mjtGeom.mjGEOM_BOX
            _geom.size = [0.02, 0.02, 0.02]
            _geom.pos = [0.04, 0.0, 0.0]
            _geom.mass = 0.4
            _geom.contype = _geom.conaffinity = 1
            self.mj_model = _spec.compile()
            _log.info("[BASE] EE cube attached to wrist_3_R (0.04 m / 0.4 kg) — gravity comp updated")
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Root body: the body with the freejoint (used for xpos/subtree_com lookups)
        self.root_body_id = next(
            i for i in range(self.mj_model.nbody)
            if self.mj_model.body_jntnum[i] > 0
            and self.mj_model.jnt_type[self.mj_model.body_jntadr[i]] == 0  # free joint
        )
        self.body_name_to_id = {
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i): i
            for i in range(self.mj_model.nbody)
        }

        # --- Gravity compensation state ---
        self.gravity_comp_scale    = 1.0
        self.gravity_torque_raw    = torch.zeros(self.n_joints, device=self.device)
        self.gravity_torque_applied = torch.zeros(self.n_joints, device=self.device)

        self.set_gravity_compensation(gravity_compensation_enabled)
        self.compute_gravity_compensation()  # warm up MuJoCo data structures
        _log.info(f"[BASE] Gravity comp ready: {self.mj_model.nq} DoF, scale={self.gravity_comp_scale}")

        # --- IMU validation (after MuJoCo warmup so IMU has had time to start) ---
        if not use_fake:
            if not hasattr(self.imu, 'counter') or self.imu.counter == 0:
                raise RuntimeError("[BASE] IMU not responding — check connection!")
            gravity_norm = np.linalg.norm(self.imu.transformed_gravity_vec)
            if abs(gravity_norm - 1) > 0.1:
                raise RuntimeError(
                    f"[BASE] IMU gravity norm {gravity_norm:.3f} far from 1.0 — check IMU calibration!"
                )
            _log.info(f"[BASE] IMU OK: gravity norm={gravity_norm:.3f}, counter={self.imu.counter}")

        # --- Motor start sequence ---
        _log.info("[BASE] Initializing hardware...")
        self.motor.loc_kp[:] = 0
        self.motor.spd_kp[:] = 10  # damping only until prepare() sets real gains
        self.motor.mech_torque_ref[:] = 0
        
        if self.enable_motor != "false":
            self.motor.enable(self._build_motor_enable_mask())

        self.motor.set_max_torque_ratio(torque_limit)
        self.motor.start_motion_control_continuously()

        # Tighten recv timeout after CAN thread is running (startup needs the long default)
        time.sleep(0.1)
        self.motor.recv_timeout_ms = 200

        _log.info(f"[BASE] Ready: {self.n_joints} joints, {self.control_freq} Hz, torque={torque_limit*100:.0f}%")

    # -------------------------------------------------------------------------
    # Kinematics and Gravity compensation
    # -------------------------------------------------------------------------

    def update_kinematics(self, update_joint_vel: bool = False) -> None:
        """Sync qpos/qvel from sensors and run the minimal MuJoCo kinematic chain.

        Always runs: mj_kinematics → mj_comPos → mj_comVel.
        After any call, get_body_pose() and get_body_velocity() are valid.

        Args:
            use_encoder_vel: If True, qvel[6:] is set from motor encoders (freejoint qvel[:6]=0).
                             If False, all qvel=0 (Coriolis-free; used by gravity compensation).
        """
        self.mj_data.qpos[3:7] = self.imu.transformed_quat_wxyz
        self.mj_data.qpos[7:] = self.motor.mech_pos
        if update_joint_vel:
            self.mj_data.qvel[:6] = 0.0
            self.mj_data.qvel[6:] = self.motor.mech_vel
        else:
            self.mj_data.qvel[:] = 0.0

        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)
        mujoco.mj_comVel(self.mj_model, self.mj_data)


    def get_body_pose(self, body: str | int) -> tuple[np.ndarray, np.ndarray]:
        """World-frame pose from last _update_mj_kinematics() call.

        Returns:
            pos:  shape (3,) position in metres  — view into mj_data, copy if holding.
            quat: shape (4,) orientation wxyz    — view into mj_data, copy if holding.
        """
        bid = self.body_name_to_id[body] if isinstance(body, str) else body
        return self.mj_data.xpos[bid], self.mj_data.xquat[bid]

    def get_body_velocity(self, body: str | int) -> tuple[np.ndarray, np.ndarray]:
        """World-frame velocity from last _update_mj_kinematics() call.

        Velocity is zero if mj_kinematics was last called with use_encoder_vel=False
        (gravity-comp path), which is correct for that context.
        mj_data.cvel convention: [:3]=angular, [3:]=linear.

        Returns:
            vel_lin: shape (3,) linear velocity in m/s.
            vel_ang: shape (3,) angular velocity in rad/s.
        """
        bid = self.body_name_to_id[body] if isinstance(body, str) else body
        return self.mj_data.cvel[bid, 3:], self.mj_data.cvel[bid, :3]

    def compute_gravity_compensation(self) -> torch.Tensor:
        """Compute gravity compensation torques for current joint positions.

        Uses MuJoCo inverse dynamics (RNE with zero velocity) to find the
        torques needed to counteract gravity at the current configuration.
        IMU orientation is used so the gravity direction is correct when tilted.

        Returns:
            torch.Tensor: shape (n_joints,) gravity torques in Nm, on self.device.
        """
        self.update_kinematics(update_joint_vel=False)
        mujoco.mj_rne(self.mj_model, self.mj_data, 0, self.mj_data.qfrc_bias)

        # qfrc_bias = gravity + Coriolis (Coriolis == 0 since qvel=0)
        # First 6 elements are freejoint forces/torques — skip them
        torque_np = self.mj_data.qfrc_bias[6:].copy() * self.gravity_comp_scale
        return torch.from_numpy(torque_np).float().to(self.device)

    def set_gravity_compensation(self, enabled: bool):
        """Enable or disable gravity compensation feedforward."""
        self.gravity_compensation_enabled = enabled
        if not enabled:
            self.motor.mech_torque_ref[:] = 0
        log = _log.warning if enabled else _log.info
        log(f"[BASE] Gravity compensation: {'enabled' if enabled else 'disabled'}")

    # -------------------------------------------------------------------------
    # Prepare (move to initial pose)
    # -------------------------------------------------------------------------

    def prepare(
        self,
        target_pos: torch.Tensor,
        kp: np.ndarray,
        kd: np.ndarray,
        duration: float = 2.0,
    ):
        """Smoothly move all joints from current position to target_pos over duration seconds.

        Args:
            target_pos: Desired joint positions, shape (n_joints,), on any device.
            kp:         Position gains, shape (n_joints,), numpy array.
            kd:         Velocity gains, shape (n_joints,), numpy array.
            duration:   Ramp time in seconds.

        Uses self.control_freq and self.dt, so subclasses that override those
        (e.g. HumanoidRealEnv after reading config) get the correct timing.
        Interpolation: smoothstep s = 3t² - 2t³ (zero velocity at endpoints).
        """
        if self.use_fake or self.enable_motor == "false":
            _log.info("[BASE] Skipping prepare() since motor is disabled or fake")
            return

        _log.info(f"[BASE] Moving to initial pose over {duration}s...")

        current_pos = torch.tensor(self.motor.mech_pos, device=self.device)
        target_pos  = target_pos.to(self.device)

        # Clamp target to joint limits — prevents driving into hard stops if config has bad values
        lower = torch.tensor(JOINT_LOWER_LIMITS, dtype=torch.float32, device=self.device)
        upper = torch.tensor(JOINT_UPPER_LIMITS, dtype=torch.float32, device=self.device)
        target_pos = torch.clamp(target_pos, lower, upper)

        self.motor.loc_kp[:] = kp
        self.motor.spd_kp[:] = kd
        if self.enable_motor != "false":
            self.motor.enable(self._build_motor_enable_mask())

        # TODO(H5): gravity compensation is not applied during prepare(), so heavy-limbed
        # robots may sag, overshoot, or jerk at ramp end. If needed, call
        # compute_gravity_compensation() here and write to motor.mech_torque_ref each step.
        num_steps = int(duration * self.control_freq)
        for i in range(num_steps + 1):
            t = i / num_steps
            s = 3 * t**2 - 2 * t**3          # smoothstep: zero velocity at t=0 and t=1
            pos = current_pos + s * (target_pos - current_pos)
            self.motor.mech_pos_ref[:] = pos.cpu().numpy()
            time.sleep(self.dt)
            if i % 50 == 0:
                _log.info(f"  {i}/{num_steps} ({100*t:.0f}%)")

        _log.info("[BASE] Initial pose reached")

    def _build_motor_enable_mask(self) -> np.ndarray:
        """Return a bool array of length n_joints: True = send Enable CAN frame, False = leave in Reset.

        "true" enables every joint. "leg"/"arm"/"camera" enable ONLY that group
        (additive: start all-False, turn on the requested slice) so cameras stay
        passive under "leg"/"arm" and vice-versa. "false" never reaches here
        (callers guard on enable_motor != "false").
        """
        if self.enable_motor == "true":
            return np.ones(self.n_joints, dtype=bool)
        mask = np.zeros(self.n_joints, dtype=bool)
        for sl in _MOTOR_ENABLE_SLICES[self.enable_motor]:
            mask[sl] = True
        return mask

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def shutdown(self):
        """Wind down motors to damping-only and shut down IMU.

        Subclasses should call super().shutdown() after their own cleanup
        (e.g. HumanoidRealEnv saves recording and stops the controller first).
        """
        _log.info("[BASE] Shutting down hardware...")
        self.motor.spd_kp[:] = 5
        self.motor.loc_kp[:] = 0
        self.motor.mech_torque_ref[:] = 0
        _log.info("[BASE] Waiting for motors to stop...")
        time.sleep(2)
        self.motor.spd_kp[:] = 1
        self.motor.loc_kp[:] = 0
        self.imu.shutdown()
        _log.info("[BASE] Done")
