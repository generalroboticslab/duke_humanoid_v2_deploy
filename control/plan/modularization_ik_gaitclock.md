# Modularization: IKController + GaitPhaseClock

## Motivation

`humanoid_real_env.py` still has two large embedded subsystems that make the file hard to follow:

- **IK subsystem** (~250 lines): background thread, buffers, inv-dyn — logically separate from the env loop
- **GaitPhaseClock** (~50 lines scattered across `__init__`, `reset()`, `get_observation()`): simple clock but hard to read when split across three methods

Extracting these achieves the readability goal with behavioral equivalence.

---

## Part 1: IKController (`humanoid_ik_controller.py`)

### What moves

| Current location | What |
|---|---|
| `_setup_ik()` | init: thread, buffers, all CUDA tensors |
| `_ik_get_physics_qpos()` | build qpos from encoder + IMU |
| `_ik_get_actual_ee_state()` | read EE pos/quat from mj_data |
| `_ik_worker_loop()` | background solve thread |
| `_ik_snap_orientation_targets()` | seed targets from FK on reset |
| `_setup_inv_dyn()` | scratch MjData + dof address map |
| `_compute_arm_inv_dyn_cpu()` | RNEA torque feedforward |

### Proposed interface

```python
class IKController:
    def __init__(self, mj_model, arm_joint_names: dict, dt: float, device: torch.device):
        # arm_joint_names = {"left": [...], "right": [...]}
        # sets up ik_solver, thread, all buffers
        ...

    def reset(self, base_quat: torch.Tensor, motor_pos: np.ndarray) -> None:
        # snaps EE targets from FK at current pose (was _ik_snap_orientation_targets)
        ...

    def post_inputs(self, qpos: torch.Tensor, ee_pos_targets, ee_quat_targets,
                    actual_pos, actual_quat) -> None:
        # writes shared input buffers + triggers worker
        ...

    def read_output(self) -> torch.Tensor:
        # returns clamped joint solution (1-step latency)
        ...

    def read_dyn(self) -> tuple[np.ndarray, np.ndarray]:
        # returns (dq, ddq) for inv-dyn feedforward
        ...

    def compute_inv_dyn(self, mj_data, dq, ddq) -> np.ndarray:
        # RNEA torque (was _compute_arm_inv_dyn_cpu)
        ...

    def shutdown(self) -> None:
        ...

    # Exposed for env to write ee targets from arm stream:
    ee_pos_targets: torch.Tensor   # (1, 2, 3)
    ee_quat_targets: torch.Tensor  # (1, 2, 4)

    # Exposed for env to sync current_target_arm_pos after IK step:
    arm_urdf_idx: torch.Tensor     # URDF indices of IK joints
    in_arm_idx: torch.Tensor       # positions within current_target_arm_pos
```

`HumanoidRealEnv` then holds `self.ik: IKController | None`. The `step()` IK block shrinks from ~40 lines to ~10.

### Critical invariants that must be preserved

1. **Post-then-read ordering**: `post_inputs()` must be called BEFORE `read_output()` in the same step. The read returns the result from the *previous* step's solve. Swapping these would apply a stale-by-2 result.

2. **`current_target_arm_pos` sync** (line 1074 current): after reading IK output, env must do:
   ```python
   self.current_target_arm_pos[self.ik.in_arm_idx] = ik_out
   ```
   This must stay in `HumanoidRealEnv.step()`, not move into `IKController`. It keeps the rate-limiter seeded at the true arm position so timeout ramp starts correctly.

3. **`_ik_solver_lock` scope**: `ik_solver.solve()` is called from both the worker thread and `reset()` (via `_ik_snap_orientation_targets`). Both must acquire the same lock. Moving the lock into `IKController` handles this cleanly — `reset()` becomes an `IKController` method that acquires its own lock.

4. **`mj_data` ownership**: `_ik_get_actual_ee_state()` reads `mj_data.xpos/xquat` which `update_kinematics()` populates. The env must call `update_kinematics()` *before* calling `post_inputs()`. This ordering is currently implicit — must be documented or enforced.

5. **inv-dyn scratch `MjData`**: uses `mj_data.qpos/qvel` populated by `update_kinematics()`. `compute_inv_dyn(mj_data, ...)` takes mj_data as argument (already the case) — no hidden dependency.

### Risks

- **Thread timing regression**: any change to when `_ik_trigger.set()` is called relative to `read_output()` changes latency by one step. Test by logging IK output vs encoder and confirming 1-step lag, not 0 or 2.
- **Lock deadlock**: if `reset()` and the worker thread both try to acquire `_ik_solver_lock` simultaneously. Current code avoids this because `reset()` stops the trigger before calling snap — must replicate this in `IKController.reset()`.
- **No unit tests**: behavioral equivalence can only be verified on hardware. Recommended: run a fixed arm replay trajectory, log `target_pos[arm_joints]` before and after refactor, compare traces.

### Test protocol

1. Run `humanoid_arm_replay_stream.py` with a recorded pkl (that script was removed on 2026-08-22 — it imported an unshipped `trajectories` module; any fixed joint-space sender on 9874, e.g. `humanoid_joint_monkey_hw.py`, serves the same purpose)
2. Log `motor.mech_pos_ref[arm_motor_indices]` at 100 Hz for 30 s
3. Refactor, repeat
4. Compare traces: max per-joint deviation should be < 1e-5 rad (floating point only)

---

## Part 2: GaitPhaseClock (inline class in `humanoid_real_env.py`)

### What moves

State currently scattered:
- `__init__`: `_gait_period_buf`, `_gait_phase_buf`, `_phase_updated_step`, `_gait_snap_window`, thresholds
- `reset()`: zero phase, randomize period, reset `_phase_updated_step`
- `get_observation()`: tick + snap logic, build `[sin, cos, sin, cos]` tensor

### Proposed interface

```python
class GaitPhaseClock:
    def __init__(self, period_range, lin_thresh, yaw_thresh, step_dt, device): ...
    def reset(self) -> None: ...
    def step(self, command: torch.Tensor) -> torch.Tensor:
        # command: [vx, vy, wz]
        # returns: [sin(2π·φ_L), cos(2π·φ_L), sin(2π·φ_R), cos(2π·φ_R)]
        ...
```

`get_observation()` replaces the 20-line block with:
```python
if self._gait_clock is not None:
    self.obs_dict["gait_phase"] = self._gait_clock.step(self.command)
```

### Critical invariant

`_phase_updated_step` guard prevents double-stepping when `get_observation()` is called multiple times per control step (e.g. during reset). The `GaitPhaseClock.step()` must track whether it already stepped this control tick. Simplest: add an internal `_last_step_id` counter incremented by the caller, or rely on the fact that `get_observation()` is only called once per `step()` + once in `reset()` (where phase is reset anyway).

### Risk

Low. The snap logic (lines 1254-1256 current) is the only subtlety:
```python
snap = (~is_moving) & (new_phase < snap_window) & (old_phase > 1 - snap_window)
```
Snaps phase to 0 at the stride boundary when decelerating. Easy to mis-transcribe the `old_phase` reference if using `new_phase` by mistake. Write a 5-line unit test before and after.

---

## Execution order

1. `GaitPhaseClock` first (lower risk, no hardware test needed)
2. `IKController` second (requires hardware verification)
3. Do **not** combine both in one PR — isolate failures

## File layout after both extractions

```
humanoid_real_env.py        ~1000 lines  (was ~1580)
humanoid_ik_controller.py     ~270 lines  (new)
humanoid_vicon.py               ~90 lines  (done)
```
