# Motor Velocity Safety — Smooth Barrier in C++ Control Loop

## Motivation

No velocity safety exists on the Duke Humanoid V2. If a motor runs away (policy bug, IK divergence,
gravity pulling an unsupported arm, bad gains), it spins until the firmware `torque_limit` cap slows
acceleration — but by then velocity is already dangerous. Joint position limits in `step()` catch
out-of-range positions but have no effect on velocity.

---

## System Context

### Threading Model

```
Python thread (100 Hz)           C++ CAN thread (200 Hz)
    step()                           motion_control_once()
      ↓ writes:                        ↓ reads:
    motor.mech_pos_ref[:]              mech_pos_ref(id)
    motor.mech_torque_ref[:]           mech_torque_ref(id)
    motor.loc_kp[:]                    loc_kp(id)
    motor.spd_kp[:]                    spd_kp(id)
                                       mech_vel(id)   ← written by CAN receiver thread
```

No mutex around Eigen array reads/writes. This is by design — all existing code operates with
this "acceptable race" for performance. Any new safety mechanism must be consistent with this pattern.

### Control Law (Operation Mode 0)

Each CAN frame encodes 5 fields as uint16: `torque_ff, pos_ref, vel_ff, kp, kd`.  
Motor firmware applies: `tau = kp*(pos_ref - pos) + kd*(vel_ff - vel) + torque_ff`

- `vel_ff` (`mech_vel_ref`) is always 0 in normal operation → `kd*(0 - vel) = -kd*vel` (damping/braking)
- `torque_ff` (`mech_torque_ref`) is set per-step by IK inverse dynamics or gravity compensation
- `pos_ref` (`mech_pos_ref`) is set per-step by policy + IK

### `motion_control_once()` (motor/motor.hpp, line 929)

Runs in a dedicated C++ thread at 200 Hz (5ms period). For each motor in `round_robin_motor_order`:
1. Reads `mech_torque_ref(id)`, `mech_pos_ref(id)`, `mech_vel_ref(id)`, `loc_kp(id)`, `spd_kp(id)`
2. NaN-guards each value (fallback to 0 or `mech_pos`)
3. Encodes to uint16 via `float_to_uint16(value, spec_min, spec_max)`
4. Builds `CanMessage` and stages for batch send

### Firmware Safety Features (What Exists)

- **`torque_limit` (0x700B)**: caps peak torque in ALL modes. Set via `set_max_torque_ratio()`.
  Limits acceleration but NOT velocity — a motor at high velocity with torque_limit=0 still coasts.
- **`spd_limit` (0x7017)**: velocity cap in **position mode (run_mode=1) ONLY**.
  Does absolutely nothing in operation mode (run_mode=0). Cannot be used here.

### Python Safety Features (What Exists)

- Joint position limits: `step()` clamps `target_pos` to `[JOINT_LOWER, JOINT_UPPER]` before writing `mech_pos_ref`.
- Action clipping: policy action is clipped before being added to `target_pos`.
- `torque_limit` ratio: set at init via `motor.set_max_torque_ratio(torque_limit)`.

### What Is Missing

No check on actual `mech_vel`. Two acceleration sources are unconstrained:
1. `kp*(pos_ref - pos)`: if policy commands far target or IK diverges, large position error → large torque → velocity buildup
2. `torque_ff` (`mech_torque_ref`): IK inverse dynamics + gravity comp written per-step with no magnitude constraint beyond the firmware torque cap

---

## Constraints

1. **Must work in operation mode (run_mode=0).** Cannot switch to position mode.
2. **Cannot use firmware `spd_limit`** — only applies to position mode.
3. **No new mutex** around Eigen array accesses — consistent with existing codebase.
4. **C++ rebuild required** for changes to `motor.hpp` and `my_ext.cpp`.
5. **`vel_limit` must be set before `start_motion_control_continuously()`** — the C++ thread reads `vel_limit` from the first call; if it reads before Python sets values, it sees the safe default (1e6 = barrier inactive).
6. **`FakeMotorController` always returns `mech_vel=0`** — barrier must be a no-op on fake hardware.
7. **n_joints = 27** (fixed by robot configuration; `humanoid_config.py` `motor_setup_dict`).

---

## Assumptions

- `mech_vel` is the motor encoder velocity in rad/s, updated by the CAN receiver thread each status frame. It is the ground truth for motor velocity.
- `mech_vel_ref` is always 0 in normal operation (feedforward speed = 0). The kd term then provides pure damping: `kd*(0 - vel) = -kd*vel`.
- Motor spec `vel_max` values (from `motor.hpp`): R00=33, R01=44, R02=44, R03=20, R04=15, R05=50, R06=50 rad/s. These are firmware encoding limits, NOT safe operational velocities.
- Walking gait peak velocities are well below spec limits. R04 knee peaks at ~8-10 rad/s (spec: 15 rad/s). Arms during normal use peak at ~3-5 rad/s (spec: 20-50 rad/s).
- `kd` (spd_kp) is set to a nonzero value before and during normal operation, so the damping term `-kd*vel` provides meaningful braking when the barrier zeroes the acceleration terms.
- The 10ms lag for Python to detect `vel_estop_triggered` is acceptable — C++ barrier already clamps the CAN frame to pure damping in that window, so the motor is decelerating, not accelerating.
- URDF joint ordering matches `motor_setup_dict` insertion order in `humanoid_config.py`.

---

## Rejected Approaches

### Python-side brake zone (bang-bang)
Check `|mech_vel| > threshold` in `step()`, then zero `mech_pos_ref` delta and `mech_torque_ref` for
violating joints. **Rejected**: 10ms reaction lag. In those 10ms, `motion_control_once()` sends 2 CAN
frames with the pre-brake command values before Python can intervene. Motor may overshoot.

### Python-side torque_ff clamp only
Clamp `mech_torque_ref` to `±MAX_TORQUE * torque_limit`. **Rejected**: leaves `kp*(pos_ref - pos)`
unconstrained. A large position error from a policy glitch still drives high velocity.

### Both together in Python (torque clamp + brake zone)
Still has the 10ms lag problem. Clamp is applied at 100 Hz; barrier at 200 Hz catches it sooner.
Also: Python writes to `mech_pos_ref` and `mech_torque_ref` sequentially, then the C++ thread
reads them atomically (same encode pass). Python can clamp, but C++ may have already encoded the
unclamped value in the cycle before the Python clamp fires.

### Firmware spd_limit
`spd_limit` (0x7017) only works in position mode (run_mode=1). **Rejected**: operator cannot switch.

---

## Solution: Smooth Velocity Barrier in `motion_control_once()`

Insert a proportional velocity barrier inside the per-motor encode loop of `motion_control_once()`.
For each motor, before encoding:

1. Compute `barrier_scale ∈ [0, 1]` from current `mech_vel` vs `vel_limit[id]`
2. Apply to position error and torque feedforward via **local variables** (no writes to shared arrays)
3. Encode the modified values into the CAN frame

```
barrier_scale = clamp(1 - max(0, |vel|/vel_limit - 0.7) / 0.3, 0, 1)
```

- `|vel| < 0.7 * vel_limit` → `barrier_scale = 1.0` → no change
- `0.7 * vel_limit ≤ |vel| < vel_limit` → `barrier_scale` ramps linearly 1 → 0
- `|vel| ≥ vel_limit` → `barrier_scale = 0.0` → pos_ref = pos, torq_ff = 0; only `kd*(-vel)` braking

```cpp
float pos_ref_eff = mech_pos(id) + barrier_scale * (mech_pos_ref(id) - mech_pos(id));
float torq_eff    = barrier_scale * mech_torque_ref(id);
// kd unchanged → -kd*vel always brakes
```

When `|vel| ≥ vel_limit`, set `vel_estop_triggered = true`. Python detects this in `step()` and
calls `motor.disable(True)` + raises `RuntimeError`.

**Why this works:**
- Fires at 200 Hz — same rate as CAN frames. No lag.
- Uses local vars — no shared state mutation, no mutex.
- Both acceleration terms (pos error + torque_ff) are scaled together — no unclamped residual.
- kd damping is preserved at full strength — motor actively brakes as barrier_scale → 0.
- Linear ramp — smooth deceleration, not a hard wall. Motor "feels resistance" as it approaches limit.

---

## Implementation

**Files changed:** `humanoid_config.py`, `motor/motor.hpp`, `motor/my_ext.cpp`, `humanoid_base.py`, `humanoid_real_env.py`

---

### 1. `humanoid_config.py` — Add `VEL_LIMIT`

Add `import numpy as np` if not present. After `motor_setup = list(motor_setup_dict.values())`:

```python
# Per-joint velocity limit for the C++ velocity barrier (rad/s).
# Indexed by URDF joint order = motor_setup_dict insertion order (27 joints).
# These are OPERATIONAL limits — set conservatively below spec vel_max.
# Tune after logging peak velocities during actual use.
#
# Motor spec vel_max (motor.hpp): R00=33, R01=44, R02=44, R03=20, R04=15, R05=50, R06=50 rad/s
VEL_LIMIT = np.array([
    # #0   waist           (R03, spec 20)
    5.0,
    # #1-6  left leg: hip1, hip2, hip3 (R03), knee (R04), ankle1 (R03), ankle2 (R06)
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    # #7-12 right leg: hip1, hip2, hip3 (R03), knee (R04), ankle1 (R03), ankle2 (R06)
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    # #13-19 left arm: shoulder1 (R03), shoulder2 (R06), shoulder3 (R02),
    #                  elbow (R02), wrist1 (R02), wrist2 (R00), wrist3 (R05)
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0,
    # #20-26 right arm: shoulder1 (R03), shoulder2 (R06), shoulder3 (R02),
    #                   elbow (R02), wrist1 (R02), wrist2 (R00), wrist3 (R05)
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0,
], dtype=np.float32)  # shape (27,)
```

Values above are **placeholders**. Tune before deploying — see Tuning section below.

---

### 2. `motor/motor.hpp` — New members

**After `mech_torque_ref` in the member declarations (~line 444):**

```cpp
Eigen::Matrix<float, Eigen::Dynamic, 1> vel_limit;  // rad/s per-motor; 1e6 = barrier inactive
bool vel_estop_triggered = false;
```

**In constructor initializer list** (after `mech_torque_ref(can_ids.size())`):
```cpp
vel_limit(can_ids.size()),
```

**In constructor body** (after `mech_torque_ref.setZero()`, ~line 457):
```cpp
vel_limit.setConstant(1e6f);  // barrier inactive until Python sets real values
```

**In `motion_control_once()` for-loop** (after the `chk` lambda definition, before line 945):

```cpp
// Velocity barrier — proportional scale on pos error and torque_ff as vel approaches limit.
// kd (velocity damping) is NOT scaled: -kd*vel braking remains full at all speeds.
// Local variables only — no writes to shared Eigen arrays, no new mutex needed.
constexpr float kVelBarrierAlpha = 0.7f;  // barrier onset: fraction of vel_limit
float vel_abs = std::abs(mech_vel(id));
float barrier_scale = 1.0f;
if (vel_limit(id) < 1e5f) {  // < 1e5 means Python set a real limit (default 1e6 = inactive)
    float vel_ratio = vel_abs / vel_limit(id);
    barrier_scale = std::clamp(
        1.0f - std::max(0.0f, vel_ratio - kVelBarrierAlpha) / (1.0f - kVelBarrierAlpha),
        0.0f, 1.0f);
    if (vel_abs >= vel_limit(id))
        vel_estop_triggered = true;
}
float pos_ref_eff = mech_pos(id) + barrier_scale * (mech_pos_ref(id) - mech_pos(id));
float torq_eff    = barrier_scale * mech_torque_ref(id);
```

**Replace the two existing `chk()` calls on lines 945-946:**
```cpp
// Before:
u_int16_t mech_torque_u16 = float_to_uint16(chk(mech_torque_ref(id), 0.0f,         "torq"), ...);
u_int16_t mech_pos_u16    = float_to_uint16(chk(mech_pos_ref(id),    mech_pos(id), "pos"),  ...);

// After:
u_int16_t mech_torque_u16 = float_to_uint16(chk(torq_eff,    0.0f,         "torq"), ...);
u_int16_t mech_pos_u16    = float_to_uint16(chk(pos_ref_eff, mech_pos(id), "pos"),  ...);
```

Lines for `mech_vel_ref`, `loc_kp`, `spd_kp` (947-949) are **unchanged**.

---

### 3. `motor/my_ext.cpp` — Expose via nanobind

After `.def_rw("mech_torque_ref", ...)` (~line 189):

```cpp
.def_rw("vel_limit", &CanMotorController::vel_limit)
.def_rw("vel_estop_triggered", &CanMotorController::vel_estop_triggered)
```

---

### 4. `humanoid_base.py` — Accept `vel_limit` arg

**Constructor signature** — add after `enable_motor`:
```python
def __init__(
    self,
    torque_limit: float = 0.1,
    use_fake: bool = False,
    gravity_compensation_enabled: bool = False,
    add_right_ee: bool = False,
    enable_motor: str | bool = True,
    vel_limit: np.ndarray | None = None,  # rad/s per joint, shape (n_joints,); None = barrier off
):
```

**Before `self.motor.start_motion_control_continuously()`** (line ~169):
```python
if vel_limit is not None:
    assert vel_limit.shape == (self.n_joints,), \
        f"vel_limit shape {vel_limit.shape} != ({self.n_joints},)"
    self.motor.vel_limit[:] = vel_limit
    _log.info(f"[BASE] Velocity barrier active (rad/s): {vel_limit.tolist()}")
else:
    _log.info("[BASE] Velocity barrier disabled")
```

---

### 5. `humanoid_real_env.py` — Wire up VEL_LIMIT + e-stop check

**Add to imports** (with other humanoid_config imports):
```python
from humanoid_config import VEL_LIMIT
```

**In `HumanoidRealEnv.__init__`** — add `vel_limit=VEL_LIMIT` to the `super().__init__()` call.

**At the very start of `step()`**, before any motor writes:
```python
if self.motor.vel_estop_triggered:
    _log.error("[SAFETY] Motor velocity e-stop triggered — disabling")
    self.motor.vel_estop_triggered = False
    self.motor.disable(True)
    raise RuntimeError("[SAFETY] Motor velocity limit exceeded — emergency stop")
```

---

### 6. Build

```bash
cd <control>
cmake --build build
```

---

## Design Invariants

Violations are silent bugs.

1. **`vel_limit` must be written before `start_motion_control_continuously()`.**
   The C++ thread reads `vel_limit(id)` from the first `motion_control_once()` call. If Python sets
   it after the thread starts, there is a window where the old value (1e6 = inactive) is read —
   harmless, but the barrier is not active. Placement before `start_motion_control_continuously()`
   in `humanoid_base.__init__` ensures this.

2. **Barrier uses local variables only — never modifies `mech_pos_ref` or `mech_torque_ref` member arrays.**
   The C++ thread and Python thread share these arrays without a mutex. Writing from C++ while Python
   reads would violate the existing "one writer" assumption and could cause Python to read back the
   barrier-modified value (a stale `pos` snapshot) rather than the intended reference.

3. **`barrier_scale` is applied to the position ERROR, not the reference directly.**
   `pos_ref_eff = pos + barrier_scale * (pos_ref - pos)`. When `barrier_scale=0`: `pos_ref_eff = pos`
   → zero position error → zero kp torque. If instead we zeroed `pos_ref` directly, we'd command the
   motor toward angle=0, which is wrong.

4. **`kd` (spd_kp) must be nonzero for braking to work at barrier_scale=0.**
   If `spd_kp[id] = 0` (e.g., prepare() hasn't run yet), the barrier provides no deceleration —
   only zero acceleration. The motor would coast at constant velocity. Ensure `spd_kp` is set before
   the barrier can activate.

5. **`vel_estop_triggered` is a plain bool, not atomic.**
   This is consistent with all other shared state in the codebase (Eigen arrays accessed without
   mutex). Stale reads are acceptable: worst case Python sees `false` for one extra step (10ms),
   during which the barrier already clamps the CAN frame to pure damping. The motor is decelerating,
   not accelerating, in that window.

---

## Potential Issues

### 1. Barrier onset too early — interferes with normal gait

**Problem:** If `VEL_LIMIT` is set too low, the barrier activates during normal walking.
Barrier onset is at `0.7 * vel_limit`. For R04 knee: `VEL_LIMIT[4] = 10.0 rad/s` →
onset at 7 rad/s. If walking knee peaks at 8 rad/s, the barrier fires and damps the knee
mid-swing → limping gait.

**Detection:** Log `self.motor.mech_vel` over a walking episode. Identify peak per joint.
Ensure peak < 70% of `VEL_LIMIT[joint]`.

**Fix:** Raise the relevant `VEL_LIMIT` entry. Or lower `kVelBarrierAlpha` from 0.7 to a
higher value (e.g., 0.8) to start the ramp later.

### 2. kd too low to brake before e-stop triggers

**Problem:** At `|vel| = vel_limit`, barrier_scale=0 and only `kd*(-vel)` brakes.
If `kd` is small and `vel_limit` is set high, deceleration may be slow. The motor may not
decelerate fast enough to stay below `vel_limit` in steady state — it oscillates around the limit
threshold, repeatedly setting `vel_estop_triggered`.

**Detection:** On real hardware, observe `mech_vel` when barrier is active. If it keeps hitting
`vel_limit` despite braking, kd is insufficient.

**Fix:** Increase `spd_kp` (kd) for affected joints, or lower `VEL_LIMIT` so barrier has more
headroom to decelerate before the hard limit.

### 3. `vel_estop_triggered` never cleared on FakeMotorController

**Non-issue:** `FakeMotorController` never runs `motion_control_once()`, so `vel_estop_triggered`
is never set. No change needed.

### 4. `mech_vel` latency from CAN receiver

**Problem:** `mech_vel(id)` in `motion_control_once()` is the value from the last received
status frame. CAN round-trip time is ~2ms; if the receive thread is behind, `mech_vel` may be
stale by 1-2 frames (5-10ms equivalent). The barrier reads a velocity from slightly in the past.

**Impact:** Barrier activates slightly late (motor is already past the onset point). At 200 Hz
with ~2ms latency, the error is ~0.2% of the period — negligible for safety purposes.

### 5. `pos_ref_eff` snapshot drift across C++ cycles

**Problem:** `pos_ref_eff = mech_pos(id) + barrier_scale * (mech_pos_ref(id) - mech_pos(id))`
is computed fresh each `motion_control_once()` call. `mech_pos(id)` advances each cycle.
If Python hasn't written a new `mech_pos_ref` yet (10ms cadence), `mech_pos_ref` is frozen
from the last Python step while `mech_pos` advances. This means the position error
`(mech_pos_ref - mech_pos)` grows more negative (if motor moving forward) — slightly more
braking from the kp term even above barrier onset. This is safe and actually helpful.

### 6. Wrist joints during fast arm replay

**Problem:** `VEL_LIMIT` placeholder values (3 rad/s for wrists) may be too low for arm
replay trajectories with fast segments.

**Fix:** Log peak wrist velocity during intended fast motions. Set `VEL_LIMIT` for wrist joints
above that peak, with headroom. 3 rad/s is a conservative starting point; raise to 5-8 rad/s
if replay requires it.

### 7. `HumanoidBase` used without `HumanoidRealEnv` (arm scripts)

`HumanoidBase.__init__` accepts `vel_limit=None` (barrier disabled). Arm teach/replay scripts
that use `HumanoidBase` directly will not have the barrier active by default. If needed, pass
`vel_limit=VEL_LIMIT` explicitly. For arm-only scripts, a subset of `VEL_LIMIT` (arm joints
only) could be passed if leg safety is not required.

---

## Tuning Guide

Before first run:
1. Run with `--use-fake` and verify no compile errors, no `vel_estop_triggered` fires.
2. Run on real robot with `vel_limit=None` (disabled) and log `motor.mech_vel` over representative
   motions (standing, walking, arm gestures). Save per-joint max.
3. Set `VEL_LIMIT[joint] = 1.5 * observed_max[joint]` as a starting point (50% headroom).
4. Verify barrier onset (70% of VEL_LIMIT) is above observed max: `0.7 * VEL_LIMIT[joint] > observed_max[joint]`.
5. Run again with barrier enabled. Verify no barrier interference and `vel_estop_triggered` never fires.

---

## Verification

1. **Build**: `cmake --build build` — no compile errors or warnings on new members.
2. **Fake motor**: `--use-fake` — `vel_estop_triggered` never set, barrier no-op.
3. **Unit check**: set `VEL_LIMIT` to a very low value (e.g., 0.1 rad/s), manually nudge a joint —
   verify `vel_estop_triggered` is set and e-stop fires immediately.
4. **Real hardware**: drive a joint toward `VEL_LIMIT` slowly — observe `mech_vel` tapering (logged
   CAN pos commands approach `mech_pos`) before e-stop fires.
5. **Normal gait**: log `max(|mech_vel[joint]|)` over a walking episode. Confirm all joints below
   70% of their `VEL_LIMIT`. Adjust limits if any joint triggers during normal operation.
6. **`self_check()`**: runs `getAllParams()` which reads motor state — unaffected by new members.
