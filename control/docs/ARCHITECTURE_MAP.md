# Duke Humanoid V2 — Architecture Map

A reader's map of the released tree: what runs where, which module owns what,
and which tests cover it. Scope is `control/` plus the bundled `perception/`
modules the runtime imports; `legged_env_v2` (training and deploy export) and
cuRobo are read-only interface providers and appear only as boundaries.
Companions: the repository `README.md` (install, expected test results),
`docs/OPERATIONS.md` (the runbook), `docs/SETUP.md` (building the C++ bindings,
CAN, udev), `perception/README.md` and `PROVENANCE.md` at the repository root
(what was removed, and why). Line counts were measured on this tree with `wc -l`.

---

## 1. Runtime picture

Eight processes in a grasp session. Seven run on the robot computer; the cuRobo
plan/MPC server runs on a separate CUDA machine. All inter-process traffic is
pynng (nng): TCP between processes and machines, IPC sockets for detections and
the gripper service.

```
                   GPU machine                          robot computer
        +---------------------------+     +--------------------------------------+
  T5    | curobo_plan_server.py     |     |                                      |
        |   plan-0 / MPC solver     |<===>| humanoid_curobo_reach.py       (T6)  |
        |   TCP :9880  (msgpack)    |     |   mission driver, gates, verdicts    |
        +---------------------------+     |        |  in-process                 |
                                          |   humanoid_curobo_client.py          |
                                          |        (PlanClient, StepGate)        |
        camera rig                        |                                      |
   +--------------------+                 |   humanoid_auto_operator.py    (--)  |
   | perception/        |  tcp :5555 L    |     + auto_operator/ package         |
   | camera_streaming   |  tcp :5556 R    |       pure decision core:            |
   | _utils.py     (T1) |---------------->|       tick(det, telem, now)          |
   +--------------------+                 |            -> (nav, packet)          |
                                          |                                      |
   +--------------------+                 |   humanoid_monitor.py          (T4)  |
   | humanoid_end_      |  ipc ee_request |     AprilTag detect + viser UI :8080 |
   | effector_service   |<----------------|     ipc:///tmp/humanoid_detections   |
   | .py           (T2) |  ipc ee_status  |                                      |
   +--------------------+  (via real_env) |   humanoid_real_env.py         (T3)  |
            |                             |     50 Hz policy, obs assembly,      |
      FEETECH serial                      |     watchdog, safety gates           |
      (grippers)                          |        |  nng :9870 telemetry        |
                                          |        |  nng :9873 nav command      |
                                          |        |  nng :9874 arm + ee_action  |
                                          |        v                             |
                                          |   hardware_bindings (C++/nanobind)   |
                                          |     CanMotorController  244 Hz CAN   |
                                          |     can9, can21..can25               |
                                          +--------------------------------------+
   T0 humanoid_setup_can.py          once, after ANY power cycle
   T7 humanoid_mission_recorder.py   -> one MP4 per camera per run
```

### Port / channel map

The setting name in parentheses is the `humanoid_site` constant that declares
the number (section 2). Since 2026-08-22 the runtime reads those settings on
one side of each channel: `humanoid_real_env` opens its three sockets from
`TELEMETRY_PORT` / `NAV_COMMAND_PORT` / `ARM_COMMAND_PORT`, the monitor's
telemetry subscriber reads `TELEMETRY_PORT`, the mission recorder reads
`ARM_COMMAND_PORT` (its gaze subscriber) and `CAMERA_BASE_PORT` (its two
camera ports), and seven UDP bench publishers (`humanoid_config`,
`humanoid_camera_test`, `humanoid_camera_point`, `humanoid_pose_finder`,
`humanoid_static_stand`, `humanoid_test_motor`, `move_grip`) read
`TELEMETRY_PORT`. The peers — the operator, the reach tool, the dual-follow
demo, the joint monkey, the gamepad / GUI senders, the forensics and check
tools, the hand-eye tools' `PORT_TO_SITE` — still bind or dial the literals
(`humanoid_site.py` lists them under each setting), so a non-default port in
`site_local.py` moves real_env's socket away from its peers: treat the numbers
below as fixed unless you change every call site together.

| Channel | Transport | Direction | Payload |
|---|---|---|---|
| `9870` (`TELEMETRY_PORT`; read by real_env's publisher, the monitor and seven bench tools; the other subscribers dial the literal) | nng pub/sub | real_env -> monitor, operator / reach, tools | joint state, IMU / projected gravity, EE poses, per-side gripper state and `ee_alive`, stamp |
| `9873` (`NAV_COMMAND_PORT`; read by real_env's subscriber; every publisher binds the literal) | nng | operator or gamepad -> real_env | `{"nav_cmd": [vx, vy, wz]}`, base FLU frame |
| `9874` (`ARM_COMMAND_PORT`; read by real_env's subscriber and the recorder; every binder uses the literal) | nng | operator **or** reach tool (single binder: only one of them runs) -> real_env | `gaze_targets`, `arm_targets` (joint **or** Cartesian, per-arm modal), `ee_action` |
| `ipc:///tmp/ee_request.sock`, `ipc:///tmp/ee_status.sock` | nng IPC | real_env (`--ee-service`) <-> EE service | relayed `ee_action`; gripper status and heartbeat |
| `9880` (`PLAN_SERVER`) | nng Rep0, msgpack | reach client -> plan server | cuRobo plan / MPC session |
| `5555` / `5556` (`CAMERA_BASE_PORT` + index; read by the recorder; the monitor, the operator and the calibration tools use the literals, and the streaming server's own `--base-port` defaults to 5555) | nng pub/sub, JPEG | cameras -> monitor, recorder, calibration tools | 5555 = `cam_left_rgb` (LEFT camera on the head gimbal), 5556 = `cam_right_rgb` (RIGHT); `PORT_TO_SITE` in the hand-eye tools, `GAZE_ORDER` in the operator. Which physical camera gets which port is decided by the streaming server's `--devices` order — on the command line, or from `vs_site.CAMERA_SERIALS` as its default (OPERATIONS.md, T1) |
| `ipc:///tmp/humanoid_detections.sock` (`DETECTION_IPC_URL`) | nng IPC | monitor -> operator / reach | detected tagged-body poses |
| `ipc:///tmp/humanoid_route_preview.sock` (`ROUTE_PREVIEW_URL`) | nng IPC | reach -> monitor | planned route for the viser preview |
| `8080` / `8090` | HTTP (viser / web) | monitor UI (the EE service's manual GUI defaults to 8080 too) / `humanoid_joint_monkey_hw.py`, `humanoid_camera_view.py` | operator console / bench tools |

### Timing budget

| Loop | Rate | Budget | Owner |
|---|---|---|---|
| CAN servo | 244 Hz per motor | — | C++ `hardware_bindings/motor` |
| Policy / control | 50 Hz (`control_freq` in the deploy `env_config.yaml`) | 20 ms per tick | `humanoid_real_env.py` |
| Operator tick | 20 Hz (`OP_RATE_HZ`; `DT = 1.0 / OP_RATE_HZ`) | 50 ms | `AutoOperator.tick` |
| Gate scoring | throttled, `GATE_EVAL_PERIOD_S = 0.1` per target | — | `_gate_eval` (INC-5) |
| Reach tool publish | 20 Hz (`PUB_HZ`, `PUB_DT`) | — | `humanoid_curobo_reach.py` |
| Gripper state poll | 200 Hz | — | `humanoid_end_effector_service.py` |

INC-5 records that un-throttled CPU-torch gate scoring (17-37 ms/call, 2-3x per
tick) alone blew the 50 ms operator tick; the throttle is load-bearing.

---

## 2. Configuration layer

No path, host address or camera serial is a literal anywhere in the tree. Two
small modules resolve every installation-specific value in the same order: an
environment variable, then an optional gitignored local file, then a portable
default.

| Module | Env prefix | Local file | Resolves |
|---|---|---|---|
| `control/humanoid_site.py` | `HUMANOID_*` | `control/site_local.py` (template `site_local.example.py`) | `REPO_ROOT`, `LEGGED_ENV_ROOT`, `VISUAL_SERVOING_ROOT`, `DEPLOY_TASK` (-> `DEPLOY_RUN_DIR`, `MJCF_MODEL_PATH`; also `humanoid_real_env --task`'s default), `RECORDINGS_DIR` (default `--out` / `--out-dir` of the mission recorder, the arm forensics recorder and `humanoid_camera_view`), `PLAN_SERVER`, `ROBOT_IP`, `WORKSTATION_IP`, `VICON_IP`, `TELEMETRY_PORT` / `NAV_COMMAND_PORT` / `ARM_COMMAND_PORT` / `CAMERA_BASE_PORT` (read by real_env, the monitor and the recorder — one side of each channel; see "Port / channel map"), `CAMERA_SERIALS` (declared; no reader in `control/` — the perception twin is the one that is read), `HAND_USB_SERIAL_LEFT` / `HAND_USB_SERIAL_RIGHT` (the gripper service's `/dev/serial/by-id` fallback) |
| `perception/vs_site.py` | `VS_*` | `perception/vs_site_local.py` (template `vs_site_local.example.py`) | `REPO_ROOT`, `LEGGED_ENV_ROOT`, `CONTROL_ROOT`, `CAMERA_SERIALS` (read by `camera_tag_detection.py` and, when non-empty, by the streaming server as its `--devices` default), and the paths the other-robot scripts used |

- `CONTROL_ROOT` is self-locating (`__file__`), never configured.
- `VISUAL_SERVOING_ROOT` defaults to the bundled `perception/` directory when it
  exists, else `REPO_ROOT/visual_servoing`. `humanoid_monitor.py` inserts it on
  `sys.path` at import time; every other consumer (`humanoid_curobo_reach.py`'s
  `from tagged_bodies import ALL_CONFIGS`, the operator) relies on that single
  insert — a second insert would conflict.
- `LEGGED_ENV_ROOT` is put on `sys.path` by the modules that import runtime
  code from `legged_env_v2`: `humanoid_real_env.py` (at import),
  `curobo_plan_server.py` (at import, before argparse — a missing checkout
  fails there with an `ImportError` naming the remedies), `gampad_gui_control.py`
  (at import), and the operator only inside `main()` when `--use-gate` is
  given (since 2026-08-22; importing the operator no longer puts
  `legged_env_v2`'s top-level names on the path of every test and tool).
- Network defaults are loopback, so a missing setting fails by refusing to
  connect. Cameras are pinned to ports by the streaming server's device order:
  `--devices <left-serial> <right-serial>` on the command line, or — since
  2026-08-22 — `vs_site.CAMERA_SERIALS` (`perception/vs_site_local.py` /
  `VS_CAMERA_SERIALS`), which `perception/camera_streaming_utils.py
  multi-server` uses as the `--devices` default when it is non-empty (an
  explicit `--devices` still wins; a bare `--devices` returns to discovery).
  With neither, it takes the attached RealSense devices in USB enumeration
  order, and a left/right swap is silent because every downstream binding is
  by port (OPERATIONS.md, T1). `vs_site.CAMERA_SERIALS` is also read by the
  standalone detector `perception/camera_tag_detection.py`;
  `humanoid_site.CAMERA_SERIALS` mirrors it and has no reader in `control/`.
- Control constants are **not** site configuration: torque limits, gate
  thresholds, timing budgets and postures stay beside the code that reasons
  about them, with their dated forensic comments.

**Known seam, resolved on the control side (2026-08-22): three packages named
`common`.** The bare package name `common` used to exist three times in the
tree — `control/common` (the nng publisher / subscriber, `sleep_test.cpp`),
`control/hardware_bindings/common` (the publisher twin, see
`hardware_bindings/README.md`) and `perception/common` (`rotation_utils`) —
and which one a plain `import common` resolved to depended
on the process: scripts run from `control/` got `control/common` from the
current directory, but `humanoid_monitor.py` inserts `VISUAL_SERVOING_ROOT`
(`perception/`) at `sys.path[0]`, so in the monitor and in everything that
imports it — the operator, the reach tool, the joint monkey and their tests —
`common` was `perception/common`. The control-side package is now
`control/ipc`, a name nothing else on any `sys.path` provides, and every
control-side consumer is a plain `from ipc.publisher import ...`; the eight
file-path loaders (`spec_from_file_location` in the monitor, the operator, the
joint monkey, the gimbal zero check, the wiggle watch, the mass check, the arm
forensics recorder and the stage-walk test), the `sys.modules["common"]` purges
in `tests/test_ee_zeroing.py`, `test_gaze_failsafe.py`, `test_grasp_action_id.py`
and `test_end_effector_service.py`, and the mission recorder's detour through
the `hardware_bindings.common` twin went with the rename (the in-code comments
that blamed an upstream `common` package went with them; the shadow was always
`perception/common`). Two packages named `common` remain and are distinct by
full path: `perception/common`, reached as a bare `common` only inside
perception-side processes (`perception/camera_tag_detector.py` still appends
its own directory to whatever `common.__path__` is cached — perception-side,
unchanged), and `control/hardware_bindings/common`, imported as
`hardware_bindings.common` (`imu/py_imu.py` keeps a bare `common.publisher`
fallback for direct runs, which puts `hardware_bindings/` itself on `sys.path`
first). No control-side module imports either of them under the bare name.

---

## 3. Module map

### 3.1 Decision core — `humanoid_auto_operator.py` (≈2330 lines) + `auto_operator/` (≈4360)

Entry point and constant home. `AutoOperator.tick(detections, telemetry, now)
-> (nav, packet)` is pure: no I/O. That purity is the seam the S2-S6 refactor
hangs from (`auto_operator_refactor_report.md`; ADRs in `auto_operator_adr.md`;
the incident register in `auto_operator_incidents.md`).

| Module | Lines | Responsibility | Tests that reference it directly |
|---|---|---|---|
| `independent.py` | ≈1120 | symmetric per-arm task scheduler (`ArmTask` lifecycle) and the journey legs | `test_independent_arms.py` (89 tests, I/O-free: imports only `auto_operator`), `test_journey_audit_0807.py`, `test_journey_curobo.py` |
| `arms.py` | 1011 | per-arm hold / grasp / lift / carry; `held_update` keeps its exact branch order (ADR-5) | `test_independent_arms.py` |
| `mission.py` | 522 | SEARCH / GO / REACH / HOLD / PARK ticks, arrival gating, latching | through the operator suites (section 6) |
| `arbitration.py` | 458 | wire ownership, packet builder, `attach_ee_action` repeat burst (ADR-2, INC-3) | `test_independent_arms.py`, `test_grasp_action_id.py` |
| `servo.py` | 303 | visual-servo bias estimator (ADR-3) | `test_drift_abort_evidence.py`, `test_cube_yaw.py`, `test_dual_follow_demo.py` |
| `models.py` | 266 | typed state + dict shims (ADR-7): `ArmTask`, `GripperBurst`, `HoldState`, `SecondaryReach`, `Sighting`, `TrackMotion` | `test_cube_yaw.py`, `test_independent_arms.py`, `test_sighting_evidence.py` |
| `gaze.py` | 213 | aim / scan / trim / slew; `scan_hold` (08-12: a free eye no longer serpentines during a journey fetch) | `test_scan_hold.py`, `test_first_seer_lock.py`, `test_journey_audit_0807.py` |
| `secondary.py` | 153 | dual-parallel secondary reach | through the operator suites |
| `perception.py` | 118 | cube tracks, camera claims, ingest (`quat_wxyz` rides with each sighting since 08-13) | `test_sighting_evidence.py`, `test_first_seer_lock.py`, `test_cube_yaw.py` |
| `staging.py` | 101 | joint-space track primitives (ADR-1) | through the operator suites |
| `safety.py` | 84 | pure predicates: target_safe / sector / bearing / tilt | `test_independent_arms.py` |

**ADR-6 constraint (load-bearing):** extracted modules resolve operator
constants live via `_K(op) = sys.modules[type(op).__module__]` rather than
importing the operator. Any refactor that replaces this with a plain import
creates a second module instance and silently decouples monkeypatched constants
from the code under test. The test suite and the reach tool's `GazeRig` (which
duck-types `AutoOperator` to run `gaze` and `perception` verbatim) depend on it.

### 3.2 Robot runtime — `humanoid_real_env.py` (≈3260 lines)

The 50 Hz loop: telemetry assembly, observation construction, policy inference,
the 07-25 arm motor-dropout watchdog, torque limits, IK engage/disengage
(`mj_envs.utils.ik_mink.BatchedAnalyticalIK`), `--pin-waist-standing`,
`--obs-frame-fix` (08-12, off by default), the `ee_action` relay to the gripper
service, and the recorder (`humanoid_record_utils.py`, ≈180 lines). Imports
`humanoid_base.py` (358) -> `hardware_bindings.motor.py_motor`
(`CanMotorController`) and `hardware_bindings.imu`. Covered by
`test_waist_pin.py`, `test_obs_frame_fix.py`, `test_lw2_mirror.py`,
`test_gaze_failsafe.py`; `test_model_derived_constants.py` guards the operator
constants that are FK products of the loaded model.

### 3.3 Planning — `humanoid_curobo_reach.py` (≈7400 lines, largest file)

Mission driver for the standing dual-arm grasp: startup gates, world model
(exactly one of `--tables` / `--table` / `--no-table-world` is required), plan-0
route, streamed execution, the real-time MPC session (`--mpc`, default on), the
retreat verdict system (`RETREAT_TRIGGERS`: T01-T14 runtime, S01-S05 setup,
raised as `Abort`), the table z prior (`sane_tables`, `TABLE_Z_PRIOR_CAP_M`) and
hand floor (`HAND_FLOOR_*`), `--journey` walk legs (run on hardware, the least
mature mode — see `docs/OPERATIONS.md` section 5) and `--decoupled-arms`
(not run on hardware). Pairs with:

- `humanoid_curobo_client.py` (≈880): `PlanClient`, `StepGate`, `verify_route`
  with gates A-F (`GATE_A_FK_TOL_M`, `GATE_B_START_TOL_RAD`, `GATE_C_FLOOR_M`,
  `GATE_D_LIMIT_MARGIN_RAD`, `GATE_E_HOME_BRANCH_TOL_RAD`, `GATE_F_WINDUP_RAD`),
  the units seam (fold / seed, with the left wrist 2 sign flip),
  `merge_decoupled_bundles`. Its rule: cuRobo's model says "collision-free";
  only our own model may say "safe to execute".
- `curobo_plan_server.py` (≈1000): runs on the GPU machine and imports
  `mj_envs.tasks.visual_manipulation.curobo` from `legged_env_v2`, found
  through `humanoid_site.LEGGED_ENV_ROOT` or `PYTHONPATH` (a missing checkout
  fails at import — `--help` included — with an `ImportError` naming the
  remedies; the launch and restart lines are `docs/OPERATIONS.md` section 1),
  one warm session behind pynng Rep0 on 9880; plan-0 plus the MPC session;
  `MPC_FLOOR_FINGER_DROP_M` (08-12), `MPC_REANCHOR_RAD`. The client's
  `PlanServerDeadError` and the reach tool's "server unreachable" refusal
  print that restart recipe.
- `humanoid_plan_server_probe.py` (176): drives real MPC sessions against the
  server with a simulated executor; all three cases must PASS before the robot
  is touched.

**Paired constants — never change one alone:** client `MPC_LEAD_SCALE` (reach)
and `GATE_F_WINDUP_RAD` (client) against the server's `MPC_REANCHOR_RAD`. When
they drift apart the track degenerates into a sawtooth — the server re-anchors
every tick and the hand makes no progress (T13). Hardware 2026-08-05: a server
re-anchor threshold of 0.10 rad against the client's 0.152 rad lead. The probe
prints that lead (`MPC_LEAD_SCALE * max_rate * PUB_DT`) next to its verdict;
compare it with the server you deploy before a run — a `governed` PARK while
`perfect` converges is the signature. `docs/OPERATIONS.md` section 4 lists the
tuned constants.

### 3.4 Perception bridge — `humanoid_monitor.py` (≈1170 lines)

AprilTag detection through `perception/camera_tag_detector.py`, multi-camera
fusion (`BatchedScene`), the viser UI on 8080, detection publishing on
`DETECTION_IPC_URL`. Owns `GimbalCameraFK`, which the operator and the reach
tool import. It performs the one `sys.path` insert of `VISUAL_SERVOING_ROOT`
(section 2), which puts `perception/` at `sys.path[0]` — the reason the
control-side publisher package is named `ipc` rather than `common` (the "three
packages named `common`" seam in section 2).
Covered by `test_detection_retention.py`.

### 3.5 End effector — `humanoid_end_effector_service.py` (≈1420 lines)

Owns both FEETECH serial grippers through `hardware_bindings.ft_servo`
(`FtServo`): a 200 Hz state-poll thread, per-side independent actions,
`zero_gripper` calibration, the load-watching `hand_grab` close, a heartbeat;
consumes `ee_action` relayed by real_env over the IPC sockets. The INC-3
sender-side mechanisms live in the operator: `EE_ACTION_REPEAT_TICKS = 30`,
queue rotation with same-side supersession, a burst yields only after 5 sends
(`attach_ee_action`), `ee_alive` gating (`EE_ALIVE_WAIT_S`). Covered by
`test_end_effector_service.py`, `test_ee_zeroing.py`, `test_ee_shutdown.py`
(subprocess-tested Ctrl+C contract), `test_grasp_action_id.py`; these need the
compiled `ft_servo_ext` extension (see the README's Tests section).

### 3.6 Model source of truth — `humanoid_model.py` (≈40 lines)

Exports one constant, `MJCF_MODEL_PATH`, resolved by `humanoid_site` from
`LEGGED_ENV_ROOT` + `DEPLOY_MODEL_TASK` (the calibrated model's run directory —
deliberately NOT `DEPLOY_TASK`, which selects the checkpoint) and re-exported as
a `str`. Every tool (reach, monitor, gates, FK, calibration) loads that one MJCF
so joint ordering and hand-eye FK cannot disagree. The deploy-export and
checkpoint facts (the calibrated v159bMixedArmsCam model; the released Cosine
seed-1 checkpoint; the 2026-08-06 rejection of the `...BankFlatDecoupled`
checkpoint) are in `docs/design-notes/deploy-model-and-checkpoint.md`.

### 3.7 Native bindings, third-party code, other tools

| Component | Role | Status |
|---|---|---|
| `hardware_bindings/motor/` | CAN motor controller (`motor_bindings`, nanobind) + `py_motor.py` | ours; source inlined from the internal submodule (`PROVENANCE.md`) |
| `hardware_bindings/imu/` | IMU serial reader (`imu_nanobind`) + `py_imu.py` | ours |
| `hardware_bindings/ft_servo/` | FEETECH servo SDK (MIT, `LICENSE.FTServo`, `NOTICE.md`) + our `ft_servo_ext` binding | redistributed with attribution and the one local fix |
| `patches/` | local changes to the pinned `st_servo` component, as patches | standalone servo tooling only, not the live grasp path |

Compiled extensions (`*.so`) are gitignored build artefacts; `docs/SETUP.md`
builds them. Of the 69 top-level modules most are hand-run operator, bench and
calibration tools — section 7 indexes every one; the ones the tests lean on
are `humanoid_joint_monkey_hw.py` (≈710; hanging-only arm choreography,
`POWERON_JOINTS`), `humanoid_dual_follow_demo.py` (≈980),
`humanoid_jacobian_tracker.py` (≈400) and `humanoid_mission_recorder.py`
(≈200). Hand-eye calibration tools are `humanoid_handeye_*.py`; the
record / solve / apply workflow is `docs/SETUP.md` section 5 (`CALIBRATION.md`
at `control/` only explains why no captured data ships).

---

## 4. The `perception/` bundle

Exactly the import closure of the lab's `visual_servoing` package that the
control stack uses (`perception/README.md`). The control side imports five
modules — `camera_streaming_utils`, `camera_tag_detector`,
`camera_offline_utils`, `tagged_bodies`, `tagged_rigid_body` — and those pull
in `camera_tag_detection`, `camera_realsense_utils`, `camera_utils`,
`camera_visualizer`, `common/rotation_utils.py`, `vs_site` and the
`tagged_bodies/` registry with its `asset/` tag images.

The registry is the reason the bundle exists: a tagged body is authored once
and both sides read the same fields. `humanoid_curobo_reach.py` builds
`TABLE_CONFIGS` from every registered body that has a `slab_cuboid`, so the
obstacle the planner avoids and the slab the monitor draws are one set of
numbers. Tag **roll** is recorded per physical instance — a 90 deg error swaps
length for width in the collision model.

---

## 5. Boundaries with `legged_env_v2` and cuRobo

`legged_env_v2` is not public at the time of release. The control stack
consumes the interfaces below; every row except the last two is served by
`control/legged_env_bundle/` when no checkout is present (the repository README,
"Install"). The stack consumes:

| Interface | Path / symbol | Consumer |
|---|---|---|
| deploy MJCF | `<LEGGED_ENV_ROOT>/mj_envs/deploy/runs/<DEPLOY_MODEL_TASK>/robot.xml` (+ meshes under `<LEGGED_ENV_ROOT>/asset/`) | `humanoid_model.MJCF_MODEL_PATH` -> everything |
| policy weights | `.../<DEPLOY_TASK>/policy_deployed.pt` | `humanoid_real_env.py` |
| deploy env config | `.../<DEPLOY_TASK>/env_config.yaml` | obs scaling, default pose, action scale, `control_freq` |
| `mj_envs` package | `mj_envs.utils.ik_mink.BatchedAnalyticalIK`, `mj_envs.utils.torch_math_utils` | `humanoid_real_env.py` |
| learned arrival gate | `tasks.visual_manipulation.moving_policy.ReachabilityGate` | `humanoid_auto_operator.py` (geometric floor when absent) |
| Vicon calibration | `mj_envs/deploy/vicon_calibration.yaml` (`VICON_CALIBRATION_PATH`) | `humanoid_real_env.py` when Vicon is enabled |
| cuRobo + `mj_envs.tasks.visual_manipulation.curobo` | GPU machine only | `curobo_plan_server.py` |

`control/legged_env_bundle/` is the upstream subset in the upstream layout
(`mj_envs/deploy/runs/<task>/`, `asset/create/meshes/`, `asset/duke_v2/`,
`mj_envs/utils/`, `mj_envs/tasks/visual_manipulation/`, `mj_envs/asset_zoo/`),
with its own README and a `MANIFEST.md5` of every file. `humanoid_site.
LEGGED_ENV_ROOT` resolves to it when `REPO_ROOT/legged_env_v2` is absent, so the
same relative `meshdir` inside `robot.xml` works in both layouts. The only
in-tree direct reader is `test_gaze_failsafe.py` (the released checkpoint's
`env_config.yaml`). `_audit_cube_0.03.xml` is the cube-audit scene for offline
gate checks. Do not clean up the `robot.xml.*` and `env_config.yaml.*`
variants; each was the rollback path for a hardware calibration step.

**Two directories, resolved separately:** `DEPLOY_TASK`
(`...BankFlatDecoupledCosine`, the released checkpoint: weights + env config)
and `DEPLOY_MODEL_TASK` (`...v159bMixedArmsCam`, the calibrated robot model). The
model directory's name is historical — it held the deployed weights (v159b, then
v2_best) until the 2026-08-24 checkpoint decision and ships no weights now.
Documented in `humanoid_site.py`, `legged_env_bundle/README.md` and the design note.

---

## 6. Test topology

42 files under `control/tests/`, 635 `def test_` definitions, plain `unittest`
(not pytest), run from `control/`:

```
python -m unittest discover -s tests
```

The runner collects all 635 with the C++ extensions built (599 without:
six modules error at import until the extensions exist); the README's Tests
section says which, and which tests skip when cuRobo or the deploy model is
absent.
Largest suites: `test_mpc_track.py` (106 KB, 66 tests), `test_independent_arms.py`
(90 KB, 89), `test_journey_audit_0807.py` (40 KB, 44), `test_journey_curobo.py`
(40 KB, 42), `test_curobo_table_scene.py` (28 KB, 39), `test_curobo_bridge.py`
(31 KB, 31). Per-module coverage for the runtime, gripper, monitor and recorder
is in section 3; the planning and operator suites, by what they import:

| Module | Suites that import it |
|---|---|
| `humanoid_curobo_reach.py` | `test_mpc_track`, `test_journey_curobo`, `test_journey_audit_0807`, `test_journey_align`, `test_curobo_table_scene`, `test_curobo_envelope`, `test_dual_grasp`, `test_retreat_verdict`, `test_table_z_prior`, `test_table_latch_and_hold`, `test_reach_hover`, `test_zero_gimbals`, `test_zero_gripper_guard`, `test_grasp_candidate_sets`, `test_grasp_action_id`, `test_solve_keepalive`, `test_speculative_solve`, `test_plan_until_gated`, `test_arm_hold_during_waits`, `test_chest_home_ramp`, `test_drift_abort_evidence`, `test_decoupled_arms`, `test_cube_yaw`, `test_sighting_evidence` |
| `humanoid_curobo_client.py` | `test_curobo_bridge`, `test_decoupled_arms`, `test_plan_until_gated`, `test_retract_home`, `test_home_pose_margins`, `test_jacobian_tracker`, `test_solve_keepalive`, `test_speculative_solve`, `test_arm_hold_during_waits`, `test_mpc_track`, `test_journey_audit_0807` |
| `humanoid_auto_operator.py` + `auto_operator/` | `test_independent_arms`, `test_journey_curobo`, `test_journey_audit_0807`, `test_journey_align`, `test_first_seer_lock`, `test_scan_hold`, `test_sighting_evidence`, `test_cube_yaw`, `test_dual_follow_demo`, `test_model_derived_constants`, `test_curobo_envelope`, `test_drift_abort_evidence`, `test_chest_home_ramp`, `test_retract_home`, `test_home_pose_margins` |
| `humanoid_joint_monkey_hw.py`, `humanoid_jacobian_tracker.py`, `humanoid_dual_follow_demo.py`, `humanoid_mission_recorder.py` | `test_home_pose_margins` / `test_curobo_bridge` / `test_model_derived_constants`; `test_jacobian_tracker`; `test_dual_follow_demo`; `test_mission_recorder` |

The behavioural suite and differential replay harness that the
`auto_operator_*` documents cite lived outside the internal repository and are
not part of this release; the release-time replay result (13/13 scenarios
bit-identical against the pre-cleanup snapshot) is recorded in `PROVENANCE.md`.

---

## 7. Tool index

All 69 top-level `control/*.py` modules, one line each, grouped by how they
are used. Everything runs from `control/` with the project environment active
unless a line says otherwise. Bench scripts marked **no flags: runs on
invocation** have no argument parsing: `python x.py` runs their routine at
once — `python x.py --help` included, the flag is ignored — opening sockets
or enabling motors, so run them only on a bench with the robot hanging and
nothing else on the buses. Since 2026-08-22 each of them keeps that routine
under a `main()` guard, so importing the module is inert. The T-numbers are
the `docs/OPERATIONS.md` ladder.

**Bring-up ladder** (T1, the camera streaming server, is
`perception/camera_streaming_utils.py`, not under `control/`)

| Module | Purpose | When |
|---|---|---|
| `humanoid_setup_can.py` | bring up `can9`, `can21`–`can25` at 1 Mbit/s (`--interfaces <name> ...` for another set, e.g. the single-motor bench bus), USB-reset stragglers via the udev rules | T0, after every power cycle |
| `humanoid_end_effector_service.py` | owns both FEETECH grippers over the IPC request/status sockets, 200 Hz poll | T2 |
| `humanoid_real_env.py` | the 50 Hz policy loop: observation, policy, watchdog, IK, telemetry (`--task` defaults to `humanoid_site.DEPLOY_TASK`) | T3 |
| `humanoid_monitor.py` | AprilTag detection + fusion, viser UI on 8080, detection IPC | T4 |
| `curobo_plan_server.py` | cuRobo plan-0 + MPC server on 9880 | T5, GPU machine; finds `legged_env_v2` through `humanoid_site.LEGGED_ENV_ROOT` or `PYTHONPATH` |
| `humanoid_plan_server_probe.py` | three PASS/FAIL MPC cases against the server, no robot | before T6, after any server change |
| `humanoid_curobo_reach.py` | standing dual-arm reach-and-grasp: gates, stream, MPC, verdicts | T6 |
| `humanoid_mission_recorder.py` | both camera streams to MP4, one file per camera per run (`--out`, default `humanoid_site.RECORDINGS_DIR`) | T7, optional |

**Autonomous operator and scripted missions** (each binds 9874 — one at a time)

| Module | Purpose | When |
|---|---|---|
| `humanoid_auto_operator.py` | scan → lock → walk → stop → reach → hold operator; pure `tick` core | walk + reach mission; `--gaze-only` for camera tracking |
| `humanoid_dual_follow_demo.py` | stationed arms track cubes live, no grasp | tracking demo |
| `humanoid_joint_monkey_hw.py` | replay the sim joint-monkey arm choreography (hanging only) | photo / video sessions |
| `humanoid_arm_hold_pose.py` | park the arms at a named posture and hold | between runs |
| `humanoid_record_replay.py` | torque-free arm keyframe recording (`self_check` only, no torque ever applied) | teach step (the replay tools that consumed its keyframes were removed on 2026-08-22 — they imported an unshipped `trajectories` module; `PROVENANCE.md`) |
| `humanoid_stage_walk_test.py` | one-shot hardware validation of the joint-space staging chain | hardware validation |
| `humanoid_nav_step_test.py` | one commanded velocity step, measured | first hardware walk |
| `ee_demo.py` | minimal remote hand open/close via 9874 | gripper smoke |
| `move_grip.py` | open the right gripper, position the arm by hand, close (pure-Python servo driver) | bench grip test |

**Teleop, viewers, GUIs**

| Module | Purpose | When |
|---|---|---|
| `gamepad.py` | Bluetooth gamepad → `nav_cmd` on 9873 | manual driving — **no flags: runs on invocation** (binds 9873, starts pygame; import inert) |
| `gampad_gui_control.py` | DearPyGui panel → nav 9873 + arm EE 9874 ("gampad" sic) | manual driving / arms; imports `ipc.publisher.DataPublisher` (no upstream checkout needed since 2026-08-24); needs `dearpygui` (optional — exits with a one-line hint when absent) |
| `keyboard_gamepad_controller.py` | keyboard / joystick controller (sshkeyboard + pygame) | manual driving |
| `ssh_keyboard.py` | sshkeyboard → velocity over UDP 9871 (nothing in real_env subscribes) | legacy — **no flags: runs on invocation** (import inert) |
| `humanoid_recv_debug.py` | print / inspect real_env telemetry | debugging 9870 |
| `humanoid_viser_viz.py` | standalone viser robot view from telemetry | viewer |
| `humanoid_visual_manip_debug_gui.py` | send arm IK targets + `ee_action` on 9874 | manual visual-manipulation debugging; needs `dearpygui` (optional — exits with a one-line hint when absent) |
| `humanoid_pose_finder.py` | interactive joint nudging, save named postures | authoring postures |
| `humanoid_camera_view.py` | live-view / record both head cameras from the streams (web UI 8090) | during missions |
| `camera_viewer.py` | laptop-side UDP JPEG viewer for `frame_streamer` | legacy bench viewing |
| `frame_streamer.py` | UDP JPEG sender used by `humanoid_camera_point` | legacy bench |

**Calibration and model maintenance** (hand-eye workflow: `docs/SETUP.md` section 5)

| Module | Purpose | When |
|---|---|---|
| `humanoid_handeye_calibration.py` | solve camera extrinsics + cube offsets from a monitor recording | SETUP section 5, step 2 |
| `humanoid_handeye_gripper_record.py` | record a hand-eye session using the gripper-base tags | alternative protocol |
| `humanoid_handeye_gripper_solve.py` | solve a gripper-tag session (wrap → filter → solve → clean) | alternative protocol |
| `humanoid_handeye_crosscheck.py` | cross-validate the four-combination solutions | after solving |
| `humanoid_handeye_camera_offset.py` | turn a static solution into a camera-site correction (gimbal era) | after solving |
| `calibrate_cam_gear.py` | measure the motor → camera gear ratio from encoder + tag bearing (energises head motors after parsing args) | gimbal commissioning |
| `humanoid_gimbal_zero_check.py` | is the gimbal encoder zero still calibrated? read-only, ~3 s | before sessions |
| `verify_base_frame.py` | verify / solve the `mech_pos` → model-joint sign convention for `GimbalCameraFK` | gimbal commissioning |
| `humanoid_set_zero.py` | write the current position as each motor's zero and set the ±π range — DESTRUCTIVE, asks for "yes" | mechanical zeroing only |
| `humanoid_set_current_limit.py` | set motor current limits by motor type | commissioning |
| `humanoid_mass_check.py` | compare deploy-model arm masses with the real robot | after a hardware change |
| `rebuild_deploy_model.py` | rebuild the deploy `robot.xml` from the upstream sim source, re-applying the local layers | after an upstream export |

**Bench, first power-on, forensics** (robot hanging, no policy)

| Module | Purpose | When |
|---|---|---|
| `humanoid_test_motor.py` | enable all motors at 5 % torque with low gains, publish telemetry | first power-on smoke — **no flags: runs on invocation** (import inert) |
| `humanoid_static_stand.py` | hand-scripted stand / pose sequence at 30 % torque | pre-policy bench — **no flags: runs on invocation** (import inert) |
| `humanoid_camera_test.py` | low-torque sweep of the four gimbal motors | gimbal wiring / sign check — **no flags: runs on invocation** (import inert) |
| `humanoid_camera_point.py` | drive the gimbal motors while streaming RealSense frames over UDP | early pointing bench — **no flags: runs on invocation** (import inert) |
| `humanoid_profile_motor_latency.py` | per-motor CAN round-trip latency stats (avg / std / p99) | CAN / adapter diagnosis |
| `humanoid_motor_temps.py` | read-only motor temperature probe (real_env stopped) | thermal check |
| `humanoid_dropout_probe.py` | read motor RAM post-incident without disturbing the evidence | after a dropout |
| `humanoid_wiggle_watch.py` | live motor-dropout alarm for the harness wiggle test | harness retest |
| `humanoid_arm_forensics.py` | record everything the right arm does (9870 + 9874) | incident replay |
| `humanoid_grip_slip_test.py` | bench slip test for the grip-hold torque ceilings | gripper tuning |
| `humanoid_record_log.py` | chirp / dwell excitation logs | system identification |
| `record_vicon_imu.py` | IMU + Vicon at 200 Hz to a pickle (needs `pyvicon`, optional — exits with a one-line hint when absent) | IMU / Vicon calibration |
| `torque_sensor_test.py` | read an ASCII torque sensor on `/dev/ttyACM0`, publish UDP 9870 | sensor bench — **no flags: runs on invocation** (import inert) |
| `humanoid_curobo_mpc_smoke.py` | MPC session smoke: start / step / converge / track, no robot | server bring-up |

(The GL40 single-motor bench bus is brought up with
`humanoid_setup_can.py --interfaces can26`; the separate `gl_setup_can.py`
copy was removed on 2026-08-22.)

**FEETECH servo tooling** (pure Python, no compiled extension)

| Module | Purpose | When |
|---|---|---|
| `ft_servo_python_only.py` | standalone HLS-series servo driver | bench |
| `scanner.py` | is the HLS servo connected? scan IDs | bench |
| `disable_servo.py` | torque-off servo 19 on `/dev/ttyACM1` | bench one-off — **no flags: runs on invocation** (import inert) |

**Libraries** (imported, not run)

| Module | Purpose |
|---|---|
| `humanoid_base.py` | hardware abstraction over the motor + IMU bindings |
| `humanoid_config.py` | motor table: joint → (CAN id, bus, type) in URDF order; `motor_setup`; its `__main__` is a go-to-zero helper |
| `humanoid_model.py` | `MJCF_MODEL_PATH`, single source of truth |
| `humanoid_site.py` | installation values (env → `site_local.py` → default) |
| `site_local.example.py` | template for `site_local.py` |
| `humanoid_utils.py` | async colour logger, `CircularBuffer`, MJCF joint info, transform helpers, `HumanoidViserViz` |
| `humanoid_record_utils.py` | the real_env `--record` writer (`_snap`, pickle) |
| `humanoid_gait_utils.py` | `GaitPhaseClock` for the policy observation |
| `humanoid_vicon.py` | Vicon interface for `HumanoidRealEnv` |
| `humanoid_curobo_client.py` | `PlanClient` / `StepGate` / `verify_route` gates A–F, units seam |
| `humanoid_jacobian_tracker.py` | DLS Jacobian drift-servo along a plan-0 route |
| `__init__.py` | package marker |
