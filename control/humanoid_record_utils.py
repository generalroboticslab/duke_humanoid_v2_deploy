"""`HumanoidRecorder` — the humanoid_real_env `--record` writer: copies per-tick
tensors host-side (`_snap`), marks keyframes and pickles the stacked recording
on save.
"""
import numpy as np
import time
import pickle
from datetime import datetime
from pathlib import Path
from humanoid_utils import make_async_logger

_log = make_async_logger("humanoid_recorder")


def _snap(x, size=None) -> np.ndarray:
    """Host-side COPY of a tensor (or array); NaN vector of `size` when absent.

    Required, not an optimization to remove: the robot has no GPU, so the env's
    tensors live on CPU. There `.cpu()` returns self and `.numpy()` returns a
    memory-SHARING view, so storing the result aliases the caller's persistent
    sensor buffer (humanoid_real_env._sensor_cuda slices). The controller
    overwrites that buffer every tick, which collapses every recorded row to the
    final value at save time.

    Accepts a plain ndarray (gravity-torque fallbacks) and None (optional fields
    such as the arm/gaze reference, which only exist for policies that observe
    them). None yields NaN rather than zeros: every row must keep the same shape
    so save-time stacking works, and a zero would read as a real setpoint.
    """
    if x is None:
        return np.full(size, np.nan, dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x, copy=True)

class HumanoidRecorder:
    """Handles logging telemetry and saving recordings for the Humanoid environment."""
    
    def __init__(self, record_path=None, enabled=True):
        self.enabled = enabled
        self.record_path = record_path
        self._record_data = []  # list of per-step dicts
        self._record_t0 = None
        
        if self.enabled:
            _log.info(f"[RECORDER] Enabled -> {self.record_path or 'auto'}")

    def record_step(self, env, action, target_pos):
        """Capture one timestep of data for simulation replay.
        
        Args:
            env: The HumanoidRealEnv instance (for accessing state).
            action: The action tensor applied this step.
            target_pos: The target joint position tensor.
        """
        if not self.enabled:
            return

        if self._record_t0 is None:
            self._record_t0 = time.time()
        
        t = time.time() - self._record_t0
        
        # Every tensor field goes through _snap: the stored array must not alias the
        # env's live sensor buffer (see _snap docstring).
        record_data = {
            'time': t,
            'q_pos': _snap(env.joint_pos),
            'q_vel': _snap(env.joint_vel),
            'q_tau': _snap(env.joint_torque),
            'tau_grav_raw': _snap(getattr(env, 'gravity_torque_raw', np.zeros(env.n_joints))),
            'tau_grav_applied': _snap(getattr(env, 'gravity_torque_applied', np.zeros(env.n_joints))),
            'cmd_pos': _snap(target_pos),
            'actions': _snap(action),
            'base_ang_vel': _snap(env.base_ang_vel),
            'projected_gravity': _snap(env.projected_gravity),
            'imu_quat_wxyz': _snap(env.base_quat_wxyz),
            'imu_timestamp': env.imu_timestamp,
            'command': _snap(env.command),
            'keyframe': False,
            # --- debug fields (added 2026-08-12) ---------------------------------
            # The eight actor observation terms must ALL be recoverable offline, so
            # the checkpoint can be replayed on the recorded input and its actions
            # diffed against 'actions'. A match localises a fault to the physics; a
            # mismatch localises it to obs assembly / normalisation. Six terms were
            # already covered by the fields above; these are the missing two.
            'target_arm_joint_pos': _snap(getattr(env, 'current_target_arm_pos', None),
                                          len(getattr(env, 'arm_motor_indices', ()))),
            'target_camera_joint_pos': _snap(getattr(env, 'gaze_reference', None),
                                             len(getattr(env, '_cam_motor_indices', ()))),
            # Estimator output. Noisy and only approximate, but it is what the policy
            # consumed, so tracking error cannot be reconstructed offline without it.
            'base_lin_vel': _snap(env.base_lin_vel),
            # Commanded vs measured torque: clipping at the limit is indistinguishable
            # from a stiff policy in 'q_tau' alone, and reads as shaking either way.
            'joint_effort_ref': _snap(env.joint_effort_ref),
            # Gait clock, when the policy carries one — tells whether it advanced.
            'gait_phase': _snap(getattr(env, 'gait_phase_sensor', None), 4),
            # Waist-pin blend in [0,1]: a DEPLOY-ONLY override of the policy's waist
            # target that has no counterpart in training, active exactly in the
            # zero-command regime under investigation. Invisible without this field.
            'waist_pin_alpha': float(getattr(getattr(env, '_waist_pin', None), 'alpha', float('nan'))),
            # Winding temperature: thermal derate silently removes available torque
            # over a long run, which looks like a policy regression.
            'motor_temp': np.asarray(getattr(getattr(env, 'motor', None), 'temperature',
                                             np.full(env.n_joints, np.nan)), dtype=np.float32).copy(),
        }

        # Record vicon objects
        if env.vicon is not None:
            for prefix in env.vicon.objects:
                result = env.vicon.last_results.get(prefix)
                if result is not None:
                    positions, frame_num, latency, ts = result
                    record_data[f'{prefix}_pos'] = positions[0, :3].copy()
                    record_data[f'{prefix}_quat_wxyz'] = positions[0, 3:7].copy()
                    record_data[f'{prefix}_frame'] = frame_num
                    record_data[f'{prefix}_latency'] = latency
                    record_data[f'{prefix}_timestamp'] = ts
                else:
                    record_data[f'{prefix}_pos'] = np.zeros(3)
                    record_data[f'{prefix}_quat_wxyz'] = np.array([1.0, 0.0, 0.0, 0.0])
                    record_data[f'{prefix}_frame'] = 0
                    record_data[f'{prefix}_latency'] = 0.0
                    record_data[f'{prefix}_timestamp'] = 0.0
        
        self._record_data.append(record_data)

    def mark_keyframe(self):
        """Mark the most recent recorded step as a keyframe."""
        if self.enabled and self._record_data:
            self._record_data[-1]['keyframe'] = True
            _log.info(f">>> [KEYFRAME] marked at step {len(self._record_data)}, t={self._record_data[-1]['time']:.2f}s")

    def save_recording(self, env, path=None):
        """Save recorded data to pickle file for simulation replay.
        
        Args:
            env: The HumanoidRealEnv instance (for accessing metadata).
            path: Optional override for save path.
        """
        if not self.enabled or not self._record_data:
            if self.enabled:
                _log.info("[RECORDER] No recorded data to save")
            return None
            
        path = path or self.record_path
        if path is None:
            Path("logs").mkdir(exist_ok=True)
            path = f"logs/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        else:
            Path(path).parent.mkdir(exist_ok=True)

        N = len(self._record_data)
        # Stack per-step lists into arrays
        out = {k: np.array([s[k] for s in self._record_data]) for k in self._record_data[0]}
        
        # Add static metadata from env
        out['kp'] = env.motor_kp.copy()
        out['kd'] = env.motor_kd.copy()
        out['action_scale'] = env.action_scale.cpu().numpy()
        out['default_joint_pos'] = env.default_joint_pos.cpu().numpy()
        # We need to reach into the environment for some constants or imports
        from humanoid_base import URDF_JOINT_NAMES
        out['joint_names'] = list(URDF_JOINT_NAMES)
        out['control_freq'] = env.control_freq
        out['grav_comp_scale'] = env.gravity_comp_scale
        # Provenance. Reading a run's identity off anything but the artefact itself
        # has produced backwards A/B verdicts before; 'config' carries the observation
        # structure and per-term history lengths needed to rebuild the policy input.
        out['torque_limit'] = env.torque_limit
        out['config_path'] = getattr(env, 'config_path', None)
        out['config'] = env.config

        with open(path, 'wb') as f:
            pickle.dump(out, f)
        _log.info(f"[RECORDER] Saved {N} steps to {path}")
        return path
