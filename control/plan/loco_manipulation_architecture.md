# Locomotion + Manipulation Integration Plan

## At-a-Glance Summary

**Goal:** Integrate RL-based bipedal locomotion with visual manipulation on a 27-DOF humanoid robot — while keeping the strict-timing joint control loop isolated from perception and network jitter.

**Robot:** DukeHumanoidV2 — 27 actuated joints over CAN:
- Joints 0–12: waist + left/right legs (13 joints, RL policy controlled)
- Joints 13–26: left/right arms (14 joints, 7 per arm: shoulder×3, elbow, wrist×3)

**Control stack:** RL policy runs at ~100 Hz; motor control loop at 200 Hz. Policy outputs position deltas; `humanoid_real_env.py` is the sole writer of motor position references.

**Status:** Core arm stream path is implemented. Supervisor FSM, visual manipulation, and end-effector service are not yet built. This document covers both what exists and what is planned.

---

## System Overview

### Hardware

| Segment | Joints | Motor buses | Notes |
|---------|--------|-------------|-------|
| Waist | 1 | can22 | R03 |
| Left leg | 6 (hip×3, knee, ankle×2) | can24 | R03/R04/R06 |
| Right leg | 6 (hip×3, knee, ankle×2) | can23 | R03/R04/R06 |
| Left arm | 7 (shoulder×3, elbow, wrist×3) | can20/22 | R02/R03/R05/R06/R00 |
| Right arm | 7 (shoulder×3, elbow, wrist×3) | can21/22 | R02/R03/R05/R06/R00 |

Motors are PD-controlled with optional torque feedforward. Joint indices match URDF ordering (see `humanoid_config.py`).

### Process Topology

```
[Operator workstation]          [Robot onboard]         [Jetson / vision]
  gamepad.py ──────────9871──►  humanoid_real_env.py ◄──9873── visual_navigator.py
  ssh_keyboard.py ─────9871──►       │                  ◄──9874── visual_manipulator.py
  humanoid_arm_                      │ 9870──► humanoid_viser_viz.py
    replay_stream.py ──9874──►       │ 9875──► (EE state subscribers)
                                     │
                              [CAN buses → motors]
```

All inter-process channels use **pynng Pub0/Sub0 over TCP**. Publisher binds; subscriber connects. Latest-value-wins semantics (`recv_buffer_size=1` + manual drain).

### Port Table

| Port | Direction | Binder | Connector | Payload |
|------|-----------|--------|-----------|---------|
| 9870 | Robot → workstation/viz | Robot | Workstation / `humanoid_viser_viz.py` | telemetry + viz (joint\_pos, imu\_quat\_wxyz, wb\_pos/quat, vicon\_pos) |
| 9871 | Operator → Robot | Gamepad/keyboard | Robot | `cmd`/`command_xy` |
| 9873 | Jetson → Robot | `visual_navigator.py` | Robot | `nav_cmd` (velocity intent) |
| 9874 | Sender → Robot | Arm sender | Robot | `arm_targets` (see schema note) |
| 9875 | Robot → subscribers | Robot | EE state consumers | `ee_poses` (wrist FK poses in base FLU) |
| 5555+ | Jetson → all | Camera streamer | Subscribers | camera frames |

### Control Loop Structure (`humanoid_real_env.step()`)

Each tick at 200 Hz:
1. Clip + filter policy action (EMA low-pass)
2. Read latest arm targets from stream (`_update_arm_targets()`)
3. Build full `target_pos` for all 27 joints
4. Apply joint limit clamping and anti-windup filter reset
5. Apply arm integrator (gravity model error correction, arm joints only)
6. Optionally overwrite arm targets via IK (Cartesian mode)
7. Write `motor.mech_pos_ref[:]` — **single writer, all joints**
8. Apply gravity compensation torque feedforward — **arm joints only** (legs handled by RL policy)

---

## Current Implementation Status

### Implemented

- **Transport layer** (`control/ipc/publisher.py`, `control/common/` until 2026-08-22): `NNGPublisher` / `NNGSubscriber` over pynng Pub0/Sub0 TCP. All channels migrated from raw UDP (2026-03-30). **Note:** `NNGPublisher` encodes asynchronously in a background thread — live numpy buffers (IMU, motor) must be `.copy()`-ed before calling `publish()`, otherwise the control loop overwrites them before encode.
- **Viz data merged into port 9870** (2026-03-31): separate `viz_publisher` on port 9872 removed; viz fields (`imu_quat_wxyz`, `wb_pos`, `wb_quat_wxyz`, `vicon_pos`) added to `self.publisher` payload. `humanoid_viser_viz.py` now subscribes to port 9870.
- **Arm stream receive** (`humanoid_real_env._update_arm_targets()`): ingests joint-space arm targets from port 9874, with freshness gating (0.5 s timeout) and limit clamping. Falls back to default pose on timeout.
- **Arm-only gravity compensation** (`humanoid_real_env.step()`): computes full-body gravity torques but applies feedforward to arm joints only — legs are omitted because the RL policy implicitly handles them and adding feedforward would interfere.
- **Arm position integrator** (`humanoid_real_env.step()`): anti-windup PI in position space (`KI=0.5`, `limit=0.15 rad`) corrects residual steady-state error from gravity model mismatch. Active only when gravity comp is on and IK is not in use.
- **Teach-and-replay** (`humanoid_arm_replay_stream.py` — removed from the tree on 2026-08-22, it imported an unshipped `trajectories` module; `PROVENANCE.md`): gravity-compensated keyframe recording (hardware-direct) and stream-mode replay (publishes joint targets to env via port 9874).
- **Nav command receive** (`humanoid_real_env`): subscribes to port 9873, applies freshness gating (1.0 s timeout).
- **IK mode** (`humanoid_real_env`): Cartesian arm target mode; overwrites arm joint targets from IK solution each tick.

### Not Yet Built

- Supervisor state machine (no global mode, no transition logic)
- Liveness/heartbeat signals from producers
- End-effector service (gripper/tool control)
- Visual manipulation integration (`visual_manipulator.py` → env arm_targets path)
- Mode-aware command arbitration

---

## Key Design Decisions (rationale for reviewers)

### Single-writer invariant
`humanoid_real_env.step()` is the sole writer of `motor.mech_pos_ref[:]`. All other processes are intent producers only. Eliminates command contention without explicit locking.

### Gravity comp: arm joints only
The RL locomotion policy was trained without torque feedforward on leg joints — it learned to compensate implicitly. Applying gravity feedforward to legs during RL operation would fight the policy. Gravity comp is therefore applied to arm joints only when `--grav-comp` is enabled. Full-body gravity comp (for non-RL use) lives in `HumanoidBase` directly (the pattern of `humanoid_arm_teach_replay_ref.py`, a script no longer in the tree — removed 2026-08-22, see `PROVENANCE.md`).

### Arm integrator in env, not in stream script
An anti-windup PI integrator corrects the steady-state arm position error caused by imperfect gravity models. It lives in the env (not the stream script) because: (a) actual joint positions are available locally with zero latency, (b) it compensates for a robot hardware property, not a trajectory property, (c) all arm target sources benefit automatically.

### pynng Pub0/Sub0 throughout
All channels use the same transport library (pynng), same encoding (msgpack), same binding convention (publisher binds). Eliminates broadcast/unicast inconsistency of the prior UDP setup and enables multi-subscriber fanout without configuration changes.

---

## Open Design Questions (unresolved)

### arm_targets schema conflict (Port 9874) — **RESOLVED (2026-03-30)**
**Option A chosen** (dual schema): `_update_arm_targets()` detects schema by key:
- `{"arm_targets": {"joint_pos": [...]}}` → apply directly after rate-limiting (existing replay stream behavior)
- `{"arm_targets": {"ee_pos": [[x,y,z],[x,y,z]], "ee_quat": [[w,x,y,z],[w,x,y,z]]}}` → update `ee_pos_targets`/`ee_quat_targets`; IK runs in `step()` (requires `--use-ik`)

Senders (as of this plan): `humanoid_arm_replay_stream.py` continues joint-space (that script has since been removed, 2026-08-22; today's joint-space senders are the operator, `humanoid_joint_monkey_hw.py` and `humanoid_arm_hold_pose.py`); `gampad_gui_control.py` (--robot-ip) and future `visual_manipulator.py` use Cartesian schema.

### Grasp target adaptability
Current replay stream is open-loop and robot-relative — works only if the grasp target is at a predictable pose relative to the robot body. Visual servoing (closed-loop, camera-guided) is the long-term path but is not yet integrated.

---

## Motivation

Current setup already separates concerns across processes:
- `visual_navigator.py` publishes locomotion intent (`nav_cmd`)
- `visual_manipulator.py` publishes manipulation intent (`arm_targets` pose-level goals)
- `camera_streaming_utils.py` provides camera transport

The missing piece is orchestration and timing-domain separation at the robot control boundary.

Primary requirement: **robot joint control timing is strict and highest priority**. End-effector servos are more timing-forgiving and may vary by tool.

## Motivation

Current setup already separates concerns across processes:
- `visual_navigator.py` publishes locomotion intent (`nav_cmd`)
- `visual_manipulator.py` publishes manipulation intent (`arm_targets` pose-level goals)
- `camera_streaming_utils.py` provides camera transport

The missing piece is orchestration and timing-domain separation at the robot control boundary.

Primary requirement: **robot joint control timing is strict and highest priority**. End-effector servos are more timing-forgiving and may vary by tool.

---

## Chosen Architecture (and rejected alternatives)

### Chosen
Use a **ROS-like pub/sub architecture pattern** (without ROS runtime), plus a **hierarchical state machine**:
1. Independent producer processes (vision/navigation/manipulation)
2. One strict-timing robot control core as single actuator authority
3. A supervisor state machine coordinating modes and health
4. A decoupled end-effector execution service

### Rejected for now: monolithic single-process merge
- Would couple camera/perception jitter into control timing.
- Harder to scale to multiple end-effectors.

### Rejected for now: no global supervisor
- Leads to scattered local if/timeout logic and inconsistent behavior.
- Makes failure handling and transition guarantees unclear.

---

## Assumptions

1. `visual_navigator.py` and `visual_manipulator.py` stay as separate producers.
2. `arm_targets` remains pose-level (position + orientation + stamp), not joint commands. ⚠️ **Currently violated**: `humanoid_arm_replay_stream.py` sends joint-space targets on the same port (that script was removed on 2026-08-22, but the per-arm modal wire it motivated still carries joint-space targets from the operator and the joint monkey). See Open Design Questions above.
3. Joint control loop remains the only writer of robot joint references.
4. Network transport is best-effort; stale/missing packets are expected and must be handled explicitly.
5. End-effector hardware can change (different servo counts, IDs, and action vocabularies).

---

## Design Invariants (must hold)

1. **Single writer invariant**: exactly one component writes joint references each control tick.
2. **Timing isolation invariant**: no camera/network/end-effector blocking path inside strict joint loop.
3. **Freshness gating invariant**: stale intent is never treated as fresh control input.
4. **Safety-first invariant**: safety constraints override navigation/manipulation objectives.
5. **Explicit mode invariant**: behavior changes only via defined supervisor state transitions.

---

## Logical Components

## 1) Perception/Intent Producers (async)
- Navigation producer: emits body velocity intent.
- Manipulation producer: emits arm target poses and optional end-effector action intents.
- Camera transport service: provides image stream substrate.

## 2) Control Core (strict timing)
- Fixed-rate loop, deterministic execution.
- Consumes latest valid nav/manip intents.
- Performs arbitration and outputs final joint references.
- Applies limits/safety/watchdogs each tick.

## 3) End-Effector Service (timing-forgiving)
- Receives discrete/open-loop action intents (open/close/grab/custom).
- Uses tool profile mapping to concrete servo commands.
- Runs independently from strict loop; reports ack/status.

## 4) Supervisor State Machine (global orchestrator)
- Owns global mode and transition logic.
- Evaluates channel health/freshness/operator override/safety events.
- Commands enable/hold/degrade behavior across subsystems.

---

## Timing-Domain Plan

### Domain A: Strict control domain (highest priority)
- Locomotion + arm joint command fusion
- Deterministic cadence
- Latest-snapshot input model

### Domain B: Soft control domain (lower priority)
- End-effector servo execution
- Event-driven or slower cadence
- Retry/timeout tolerant

Rule: **Domain B must never delay Domain A**.

---

## State-Machine Plan (high-level)

Global states:
- `LOCO_ONLY`
- `LOCO_PLUS_MANIP`
- `MANIP_STATIONARY` (optional stabilization mode)
- `SAFE_HOLD`
- `RECOVERY`

Transition drivers:
- Target freshness and confidence
- Channel heartbeat/health
- Operator commands/override
- Safety faults/limit events

State-machine structure:
- Global supervisor FSM (authoritative)
- Local per-process FSMs (role-specific behavior)

---

## Data Semantics (logical contract)

1. **Continuous commands** (e.g., nav velocities, arm target stream):
   - Latest-value-wins semantics
   - Freshness timeout required

2. **Discrete actions** (e.g., end-effector open/close/tool action):
   - Queued semantics with action IDs
   - Acknowledge/fail/timeout status

3. **Health/heartbeat**:
   - Each producer emits liveness signal
   - Supervisor degrades mode if liveness/freshness violated

---

## Command-Arbitration Logic (logical)

Priority order each control tick:
1. Safety constraints and fault handling
2. Stabilizing locomotion constraints
3. Manipulation objective application to arm joints
4. Optional optimization/smoothing

If manip target invalid/stale:
- Hold or decay to safe arm behavior (policy choice)
- Do not destabilize locomotion loop

---

## End-Effector Scalability Plan

Introduce **tool profiles** as the abstraction boundary:
- Servo topology (count, IDs, limits)
- Action vocabulary (open/close/grab/custom)
- Timing parameters and completion criteria

Control core emits abstract tool action; EE service resolves via active profile.

---

## Dependency Chain (implementation ordering)

1. ~~Define contract-level interfaces (message semantics, freshness, state events)~~ **Done** — pynng transport + freshness gating + arm stream schema in place
2. Resolve arm_targets schema (joint-space vs pose-space) — prerequisite for visual manipulation integration
3. Define supervisor FSM and transition table
4. Integrate arbitration in control core loop
5. Integrate visual manipulation (`visual_manipulator.py` → env arm_targets path, with IK)
6. Integrate end-effector service with tool profiles
7. Add observability (health/status/event logs)
8. Validate degraded/failure scenarios

---

## Scope Boundary

### In scope (this plan)
- Logical architecture
- Timing priorities
- Orchestration model
- High-level state machine structure

### Out of scope (for this stage)
- Concrete code edits/API signatures
- Exact timeout numeric tuning
- Full IK redesign details
- Deployment tooling/process wrappers

---

## Risks and Mitigations (logical)

1. **Mode flapping under intermittent vision**
   - Mitigate with hysteresis/cooldown and explicit recovery state

2. **Hidden command contention**
   - Mitigate with single-writer invariant and centralized arbitration point

3. **Perception latency spikes affecting behavior**
   - Mitigate with freshness gating and degraded fallback policies

4. **Tool-specific servo behavior divergence**
   - Mitigate with tool profiles and adapter isolation

---

## Acceptance Criteria (high-level)

1. Joint loop timing remains stable under camera/manip load.
2. Loss of manip/nav data causes controlled degradation, not instability.
3. End-effector actions continue without blocking strict loop.
4. Mode transitions are explicit, observable, and reproducible.
5. New end-effector can be added by profile/adapter without changing control-core logic.

---

## Pre-Implementation Checklist (answer before coding)

1. **Control-loop ownership**
   - Which process/thread is the single writer for robot joint references?
   - What is the exact control frequency target and acceptable jitter budget?

2. **Freshness and timeout policy**
   - What timeout defines stale `nav_cmd`?
   - What timeout defines stale `arm_targets`?
   - What is fallback behavior per channel when stale (hold, zero, safe pose, mode drop)?

3. **Mode/state definition**
   - Which global states are mandatory in v1 (`LOCO_ONLY`, `LOCO_PLUS_MANIP`, `SAFE_HOLD`, `RECOVERY`)?
   - What events trigger each transition?
   - What hysteresis/cooldown prevents transition flapping?

4. **Arbitration policy details**
   - In `LOCO_PLUS_MANIP`, what has priority if locomotion and manipulation objectives conflict?
   - Which joints are considered manipulable vs protected for stability?

5. **Manipulation target semantics**
   - How are `arm_targets` keys mapped to arms/tasks?
   - Is orientation always required, or optional per task?
   - What confidence/visibility rule is needed before applying targets?

6. **IK and safety boundaries**
   - What workspace bounds and joint limits are enforced before IK and after IK?
   - What behavior is required when IK is infeasible or unstable?

7. **End-effector abstraction**
   - What is the v1 tool profile schema (servo IDs, count, limits, action set)?
   - Which actions are discrete queue-based vs continuous?
   - What ack/timeout semantics are required for each action?

8. **Transport and message contract**
   - What exact message envelope fields are required (timestamp, sequence_id, source, mode)?
   - Latest-value-wins vs queued behavior per topic?
   - What heartbeat/liveness signals are mandatory?

9. **Operator control and override**
   - Which operator commands can force mode changes immediately?
   - What manual override has highest priority during unsafe behavior?

10. **Validation and observability**
   - What minimum logs/metrics are required to debug timing and mode transitions?
   - What degraded scenario tests are required before enabling full loco+manip operation?
   - What constitutes a go/no-go threshold for first integrated trials?

11. **Rollout guardrails**
   - Which features are disabled by default for first integration runs?
   - What staged rollout order is required (stand → slow walk → full task)?

12. **Failure ownership**
   - Which component is responsible for declaring `SAFE_HOLD`?
   - Who is responsible for recovery exit conditions and re-entry to active mode?

Completion rule: implementation starts only after all items above have explicit answers recorded in this plan (or a linked contract document).
