## Context
Need live visibility of IK tracking quality in `humanoid_monitor.py`. Current viser scene shows robot/detections/camera feed but not numeric error between IK target and actual end-effector pose, making teleop tuning harder.

## Why this approach
Use existing telemetry stream (`9870`) as source of truth.
- `humanoid_monitor.py` already ingests this stream and owns viser updates in one thread (`_viser_thread`) with batched `server.atomic()` updates.
- Actual EE pose already published as `ee_poses` in base/body frame.
- IK targets exist in env (`ee_pos_targets`, `ee_quat_targets`) but are not yet on telemetry.

Rejected alternatives:
- Pull target from debug GUI packets: not canonical; other producers can drive IK.
- Add new target socket: unnecessary complexity and sync risk.

## Implementation plan
### 1) Publish IK targets on telemetry
**File:** `control/humanoid_real_env.py`
- In the existing publish block (same area as `ee_poses`), add `ee_targets`.
- Schema mirrors `ee_poses`:
  - `{"end_effector_L": {"pos": [x,y,z], "quat_wxyz": [w,x,y,z]}, "end_effector_R": ...}`
- Populate from `self.ee_pos_targets` / `self.ee_quat_targets` when IK active.
- If IK inactive/unavailable, send `ee_targets=None`.

### 2) Compute target-vs-actual position error in monitor
**File:** `control/humanoid_monitor.py`
- At new-telemetry boundary (`stats.msg_count != last_data_count`), read `ee_poses` + `ee_targets`.
- Compute per-arm position error:
  - `err_vec = actual_pos - target_pos`
  - `err_norm_m = ||err_vec||`
- Keep only key intersection (e.g., `end_effector_L`, `end_effector_R`); missing data => mark `n/a`.

### 3) Display error in viser
**File:** `control/humanoid_monitor.py`
- In `_viser_thread`, create persistent readouts in viser GUI (preferred `server.gui` text/markdown handles):
  - Left EE error (mm)
  - Right EE error (mm)
  - Max/mean error (mm)
  - target status (`present` / `n/a`)
- Update these readouts in the existing update loop, aligned with atomic batch cadence.
- Fallback if GUI text mutability unsupported: scene labels near EE frames.

## Assumptions
- IK targets and `ee_poses` both in base/body FLU frame.
- Units meters.
- Quaternion order `wxyz`.

## Design invariants
1. Never compare mixed frames (world vs base).
2. Metric for this task = position-only norm.
3. Missing target never shows stale numeric value (`n/a` instead).
4. No new rendering thread; `_viser_thread` remains sole writer of viser handles.

## Dependency chain
1. Extend env telemetry (`ee_targets`).
2. Consume + compute in monitor.
3. Add viser readout handles and updates.
4. Runtime validation.

## Scope boundaries
In scope:
- Telemetry extension (`ee_targets`) on existing channel.
- Viser display of IK position error.

Out of scope:
- Orientation error metrics.
- IK solver/control changes.
- New network endpoints.

## Verification
1. Run env + monitor with IK enabled.
2. Confirm telemetry includes both `ee_poses` and `ee_targets`.
3. Verify viser error panel appears and updates each fresh packet.
4. Apply known target step (e.g., +5 cm via arm debug GUI); verify error jump then convergence.
5. Clear/stop targets; verify display switches to `n/a`.
6. Confirm monitor render cadence unchanged/no visible websocket stutter regression.
