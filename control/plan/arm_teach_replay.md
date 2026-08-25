# Arm Teach-and-Replay + HumanoidBase Modularization

> Historical plan (see [`README.md`](README.md) in this directory). The
> `HumanoidBase` extraction it proposes shipped (`control/humanoid_base.py`);
> the teach/replay scripts it describes (`humanoid_arm_teach_replay`, later
> `humanoid_arm_teach_replay_ref.py` and `humanoid_arm_replay_stream.py`) and
> the `trajectories.py` helper they imported are **not in this release** — the
> helper was never part of the tree, and the scripts were removed in the
> 2026-08-22 runtime pass (`PROVENANCE.md`). Only the torque-free keyframe
> recorder `humanoid_record_replay.py` remains.

## Motivation

`humanoid_real_env.py` (1294 lines) cannot be imported without loading a policy YAML, publishing to UDP, or initializing Vicon — even when all you want is raw motor + IMU access. This blocks writing standalone hardware utilities. The immediate need is an arm teaching tool (gravity-comp + keyframe record) and a replay tool (smooth trajectory from keyframes). Both need motor + IMU + MuJoCo gravity compensation but nothing from the policy stack.

**Rejected: standalone arm script with duplicated hardware init.** Would work short-term but creates a second place to maintain the motor/IMU/MuJoCo initialization sequence, which has nontrivial ordering constraints. Duplication of `compute_gravity_compensation()` is the same risk.

**Rejected: optional imports in `HumanoidRealEnv`.** Does not reduce cognitive load or coupling; the class is still 1294 lines.

**Chosen:** Extract `HumanoidBase` (hardware layer, no config) from `HumanoidRealEnv`. `humanoid_arm_teach_replay` uses `HumanoidBase` directly. `HumanoidRealEnv` becomes a thin subclass with no logic change.

---

## Assumptions

- Motor low-level CAN thread runs at **200 Hz** (`start_motion_control_continuously`). Python control loop also runs at 200 Hz. Policy runs at a lower rate (50 Hz, configured in YAML). These are separate cadences.
- `motor.mech_pos_ref`, `motor.mech_torque_ref`, `motor.loc_kp`, `motor.spd_kp` are numpy arrays read by the CAN thread each cycle. Writes from Python are not synchronized but safe (C-layer ring buffer).
- Arm joints are indices **13–26** in `motor_setup` and `URDF_JOINT_NAMES` (verified in `humanoid_config.py`).
- **The arm script is used with the robot hanging from a harness or lying on a surface — legs are NOT weight-bearing.** Legs/waist are damping-only (`loc_kp=0`); the robot relies on external support, not stiff leg control.
- `imu.transformed_quat_wxyz` is always available once IMU init passes.
- CUDA is available. `torch.device("cuda")` throughout.

---

## Out of Scope

- `HumanoidRealEnv.step()`, `get_observation()`, `reset()`, `process_input()`, all IK/Vicon methods — **untouched**. Refactor is structural only.
- No Vicon, IK, or navigation in `HumanoidBase` or arm script.
- No UDP publishing in `HumanoidBase` or arm script.
- Joint limits are enforced in **both modes** for safety (see Safety section).
- No arm `prepare()` — arm script does not ramp to a default arm pose; takes arms from wherever they are.

---

## Design Invariants

Violations are silent bugs.

1. **`HumanoidBase.__init__` must not read any YAML or config file.** All parameters are explicit constructor args or module-level constants. Breaking this re-introduces the import-time coupling the refactor is eliminating.

2. **`compute_gravity_compensation()` must be called every control tick before writing `mech_torque_ref`.** Calling it less often makes the feedforward stale as the arm moves; gravity torque changes continuously with joint angle.

3. **`HumanoidBase` must not be destroyed before `imu.shutdown()` is called.** The IMU runs a background thread; abandoning it causes thread-leak and serial port hold.

4. **`URDF_JOINT_NAMES` is parsed at module level in `humanoid_base.py`, once on first import.** Do not move into `__init__` (re-parses per instance, ~50 ms each). All importers share the same list object.

5. **`HumanoidRealEnv` overrides `self.control_freq` and `self.dt` after `super().__init__()`.** `prepare()` reads these at call-time, not init-time, so the override is in effect before `prepare()` runs. Do not cache `self.dt` inside `HumanoidBase.__init__`.

6. **`torque_correction_ratio` in `HumanoidRealEnv` is rebuilt as a torch tensor after `super().__init__()`.** The rebuild is intentional — it re-derives from `self.motor.motor_types`, which is authoritative post-hardware-init. The base also builds it for its own use.

7. **`prepare()` clamps `target_pos` to `JOINT_LOWER_LIMITS`/`JOINT_UPPER_LIMITS` before ramping.** Without this clamp, a misconfigured YAML `default_joint_pos` would drive joints into hard stops at full PD gain.

---

## Safety

### Joint-limit enforcement

- **`prepare()` (base):** `target_pos` is clamped to `[JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS]` with a warning before any ramp begins.
- **Record mode:** arm `mech_pos` (operator-set, read-only) is checked at ~1 Hz; warnings are logged if any joint is outside limits. No clamping (operator drives the arm; the robot cannot enforce it).
- **Replay mode:** every `step_pos` is clamped to `[ARM_LOWER, ARM_UPPER]` before writing `mech_pos_ref`. Violations logged once by pre-check before hardware init.

### IMU sanity check at startup

After IMU init, `gravity_norm = |imu.transformed_gravity_vec|` is asserted to be within 1.5 m/s² of 9.81. A bad IMU (not yet converged, miscalibrated, or disconnected) would produce a wrong gravity compensation direction, potentially applying large unexpected torques. Startup is aborted if the check fails.

### Exception safety

Both control loops (`run_record`, `run_replay`) wrap the `while True` body in `try/except Exception`. Any unhandled exception calls `_shutdown()`, which winds motors down to damping-only and shuts down the IMU cleanly.

### Palindrome loop (no transition jerk)

With `--loop`, keyframes are extended into a palindrome: `[kf0, kf1, …, kfN-1, kfN-2, …, kf1]`. The trajectory ends at `kf0`, matching the start, so the loop boundary is seamless. Without this, the arm would jump from `kfN-1` to `kf0` instantaneously at each loop iteration.

### Leg/waist: damping-only

Legs and waist are set to `loc_kp=0, spd_kp=LEG_KD` throughout. The arm script is designed for use with the robot externally supported (harness or surface). Setting `loc_kp=0` ensures the legs do not fight gravity or resist manual repositioning.

---

## Dependency Chain

```
humanoid_config.motor_setup                  (exists)
humanoid_utils.get_joint_info_from_mjcf      (exists)
legged_env_v2/asset/create/humanoid_v21.xml (exists)
trajectories.generate_smooth_curve           (exists)
keyboard_gamepad_controller.Controller       (exists)

Step 1 → CREATE humanoid_base.py
    provides: HumanoidBase
              MJCF_MODEL_PATH,
              MOTOR_TORQUE_CORRECTION_RATIO
              URDF_JOINT_NAMES, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS

Step 2 → MODIFY humanoid_real_env.py
    requires: humanoid_base.py (Step 1)
    provides: HumanoidRealEnv (unchanged external interface)

Step 3 → CREATE humanoid_arm_teach_replay
    requires: humanoid_base.py (Step 1)
              trajectories.py, keyboard_gamepad_controller.py
```

---

## File 1: `humanoid_base.py`

### Module-level (runs on import)

```python
MJCF_MODEL_PATH = "<legged_env_v2>/asset/create/humanoid_v21.xml"

MOTOR_TORQUE_CORRECTION_RATIO = {
    R00: 1.0, R01: 1.0, R02: 1.0,
    R03: 0.9,   # calibrated lower torque
    R04: 0.95, R05: 1.0, R06: 1.0,
}

URDF_JOINT_NAMES, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS = get_joint_info_from_mjcf(MJCF_MODEL_PATH)
```

**Why at module level:** Parsed once (~50 ms), shared across all instances and importers. Repeating per-instance penalizes multi-instance scenarios and makes constants unavailable before instantiation.

### `HumanoidBase.__init__(torque_limit=0.1, use_fake=False, gravity_compensation_enabled=False)`

Initialization order (order matters):

1. `self.device`, `self.torque_limit`, `self.use_fake`, `self.n_joints`
2. `self.control_freq = 200`; `self.dt = 1/200` ← default; `HumanoidRealEnv` overrides after `super()`
3. Motor: `CanMotorController(motor_setup)` or `FakeMotorController`
4. `self.torque_correction_ratio` from `motor.motor_types`
5. IMU: `IMU()` or `FakeIMU()`
6. MuJoCo: model + data + `root_body_id`
7. Gravity comp state; `set_gravity_compensation()`; warmup call to `compute_gravity_compensation()`
8. IMU validation: assert `counter != 0`; assert `|gravity_norm - 9.81| < 1.5`
9. Motor start: `loc_kp[:]=0`, `spd_kp[:]=10`, `mech_torque_ref[:]=0`, `enable()`, `set_max_torque_ratio()`, `start_motion_control_continuously()`, `sleep(0.1)`, `recv_timeout_ms=200`

### Methods

**`compute_gravity_compensation() -> torch.Tensor`**
MuJoCo RNE with zero velocity. Sets `qpos[3:7]` from IMU quaternion; `qpos[7:]` from motor positions. Returns `qfrc_bias[6:] * gravity_comp_scale` as torch tensor on `self.device`.

*Why torch:* Arm script and `HumanoidRealEnv.step()` both multiply by `torque_correction_ratio` (torch). A numpy return would require per-call conversion.

**`prepare(target_pos, kp, kd, duration=2.0)`**
Smoothstep ramp `s = 3t² - 2t³` from current position to `target_pos` (clamped to joint limits) over `duration` seconds. Uses `self.control_freq` and `self.dt` so a subclass override of those is in effect at call-time.

*Why explicit args:* `HumanoidBase` has no concept of a "default pose" or "configured gains." The caller always knows what to use.

**`shutdown()`**
Sets `spd_kp=1`, `loc_kp=0`, `mech_torque_ref=0`, sleeps 2 s (allows motors to coast to rest), then `imu.shutdown()`.

---

## File 2: `humanoid_real_env.py`

### Key changes

- `class HumanoidRealEnv(HumanoidBase):`
- Hardware init block replaced by `super().__init__(torque_limit, use_fake, gravity_compensation_enabled)`
- `control_freq` and `dt` overridden from config after `super()`
- `torque_correction_ratio` rebuilt post-super (authoritative from `motor.motor_types`)
- `compute_gravity_compensation` and `set_gravity_compensation` deleted (inherited)
- `prepare()` → thin wrapper: `super().prepare(self.default_joint_pos, self.motor_kp, self.motor_kd, duration)`
- `shutdown()` → override: save recording → `ctrl.stop()` → `super().shutdown()` → flush log

### Nothing else changes

`step()`, `get_observation()`, `reset()`, `process_input()`, `_record_step()`, `mark_keyframe()`, `save_recording()`, all IK methods, all Vicon methods.

---

## File 3: `humanoid_arm_teach_replay`

### Interface

```
python humanoid_arm_teach_replay --record [--output PATH] [--torque_limit 0.3]
python humanoid_arm_teach_replay --replay PATH [--seg_time 2.0] [--loop] [--torque_limit 0.5]
```

### Constants

```python
CONTROL_FREQ = 200
DT = 1.0 / CONTROL_FREQ
LEG_SLICE = slice(0, 13)     # waist (0) + left leg (1-6) + right leg (7-12)
ARM_SLICE = slice(13, 27)    # left arm (13-19) + right arm (20-26)
ARM_NAMES = URDF_JOINT_NAMES[13:27]

LEG_KD               = 2.0         # damping-only for legs/waist (loc_kp = 0)
ARM_KP_REPLAY, ARM_KD_REPLAY = 10.0, 2.0  # PD tracking during replay
ARM_KD_RECORD        = 1.5         # damping-only during teach (loc_kp = 0)
```

*Why legs damping-only:* Arm teaching requires external support (harness). `loc_kp=0` means legs do not fight gravity or resist manual repositioning. `spd_kp=LEG_KD` provides enough damping to avoid oscillation.

*Why ARM_KD_RECORD=1.5:* Enough damping to prevent oscillation when backdriven, low enough not to resist manual movement noticeably. Slightly below 2.0 because there is no `loc_kp` fighting the operator.

### Replay mode — palindrome loop

With `--loop`, the keyframe sequence is extended to a palindrome: `[kf0, …, kfN-1, kfN-2, …, kf1]`. The trajectory ends at `kf0`, so the loop boundary is seamless (no position jump).

*Why not use recorded timestamps:* Recorded dwell time varies with operator speed; fixed `seg_time` makes replay timing predictable and easy to tune.

### Keyframe file format

```python
{
    "joint_names": list(ARM_NAMES),     # 14 strings — verified on load
    "keyframes":   np.ndarray(N, 14),   # float32, radians
    "timestamps":  np.ndarray(N,),      # float64, seconds from t0
}
```

---

## Usage Guide

### Prerequisites

1. Robot is hanging from a harness **or** lying safely on a surface — legs are **not** weight-bearing.
2. All CAN buses connected. IMU connected and calibrated (gravity norm 9.81 ± 1.5 m/s²).
3. Run as high-priority process to maintain 200 Hz:
   ```
   sudo chrt -f 80 python humanoid_arm_teach_replay ...
   ```

### Record a motion

```bash
# Step 1: low torque limit for first trial (extra safety)
sudo chrt -f 80 python humanoid_arm_teach_replay --record --torque_limit 0.3

# Controls:
#   SPACE / gamepad B   → capture keyframe (prints joint positions)
#   S     / gamepad X   → save to file now (non-blocking, continues recording)
#   Q/ESC / gamepad A   → save and exit
```

The arm is gravity-compensated and freely backdriveable. Move it to each desired pose and press SPACE. The script auto-saves on exit. Output file: `arm_keyframes_YYYYMMDD_HHMMSS.pkl`.

### Replay a motion

```bash
# Single pass, slow (3 s per segment) for first verification
sudo chrt -f 80 python humanoid_arm_teach_replay --replay arm_keyframes_*.pkl --seg_time 3.0

# Loop continuously at normal speed
sudo chrt -f 80 python humanoid_arm_teach_replay --replay arm_keyframes_*.pkl --seg_time 2.0 --loop

# ESC / Q / gamepad A to stop at any time
```

---

## Safe Evaluation Checklist

### Before first run

- [ ] Robot on harness or lying flat — legs not weight-bearing
- [ ] Torque limit set low: `--torque_limit 0.3` for record, `--torque_limit 0.4` for first replay
- [ ] Arms at a neutral pose (not near joint limits) before launching
- [ ] Confirm IMU is connected — script will abort with "gravity norm" error if not

### First record session

1. Launch with `--torque_limit 0.3`
2. Verify arms feel light and backdriveable immediately after startup
3. Move arms slowly through a small range (< 30°) to confirm gravity comp is reasonable
4. Capture 2–3 keyframes, check the printed joint values look correct (sign, magnitude)
5. Press ESC; verify pkl file is created

### First replay session

1. Use `--seg_time 3.0` (slow) first
2. Watch the first 2–3 steps — arms should move smoothly with no jerk
3. If motion looks correct at 3 s, try `--seg_time 2.0`
4. Only enable `--loop` after verifying a single pass is safe
5. Keep hand near keyboard to press ESC

### Things to watch for

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| Arm jerks immediately at startup | Gravity comp direction wrong | Check IMU mounting/orientation |
| Arm oscillates during teach | `ARM_KD_RECORD` too low | Increase `ARM_KD_RECORD` in code |
| Arms track poorly during replay | `ARM_KP_REPLAY` too low | Increase `ARM_KP_REPLAY` (currently 10) |
| Motor fault / CAN error | Overcurrent, bad connection | Lower `torque_limit`, check cables |
| "IMU gravity norm" error at startup | IMU not ready or disconnected | Wait a moment, reconnect, relaunch |
| Joint limit warnings during teach | Arm moved too far | Back off from the limit before capturing keyframe |

---

## Verification

| Test | Command | Pass condition |
|------|---------|----------------|
| Fast import | `python -c "from humanoid_base import HumanoidBase"` | No YAML read, no policy machinery loaded |
| Record | `python humanoid_arm_teach_replay --record --torque_limit 0.3` | Arms backdriveable, SPACE prints table + appends, ESC saves pkl |
| Replay (single) | `python humanoid_arm_teach_replay --replay *.pkl --seg_time 3.0` | Smooth arm motion, no jerk at segment boundaries |
| Replay (loop) | `python humanoid_arm_teach_replay --replay *.pkl --loop` | Seamless loop, no jerk at loop boundary |
| Policy deploy | `python humanoid_real_env.py --task HumanoidVelocityStanding --torque_limit 0.8` | Identical behavior to pre-refactor |
| prepare() | Called from policy deploy main loop | Uses config kp/kd/default_joint_pos, all joints clamped to limits, smooth ramp |
