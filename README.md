# Duke Humanoid V2 — control stack

The onboard control software for the Duke Humanoid V2: a 31-DoF, 36 kg bipedal
humanoid with two 7-DoF arms, parallel grippers and two independently actuated
yaw-pitch RGB-D camera gimbals. This repository holds everything that runs **on
the robot** — the 50 Hz policy loop, the perception bridge, the cuRobo planning
client, the gripper service and the autonomous operator that drives a standing
dual-arm cube grasp.

Policy *training* lives in
[`duke_humanoid_v2_simulation`](https://github.com/generalroboticslab/duke_humanoid_v2_simulation);
this one deploys the exported result. Both are submodules of
[**duke_humanoid_v2**](https://github.com/generalroboticslab/duke_humanoid_v2),
the project entry point, which has the hardware specifications and results.

> **Safety.** This code moves a 36 kg machine with people beside it. Every
> control constant, wire-protocol field, timing budget and safety gate here has
> a hardware run behind it, and most carry the incident that produced them in a
> comment. [`control/docs/auto_operator_incidents.md`](control/docs/auto_operator_incidents.md)
> lists the failures these mechanisms exist to prevent, each with the
> "simplification" that would bring it back. Read
> [`control/docs/OPERATIONS.md`](control/docs/OPERATIONS.md) before running anything.

## What is in here

| Area | Entry point | Notes |
|---|---|---|
| Robot loop | `control/humanoid_real_env.py` | 50 Hz policy, observation assembly, watchdog, torque limits |
| Autonomous operator | `control/humanoid_auto_operator.py` | pure decision core: `tick(detections, telemetry, now) -> (nav, packet)` |
| Grasp mission | `control/humanoid_curobo_reach.py` | gates, retreat verdicts, MPC session management |
| Planning client / server | `control/humanoid_curobo_client.py`, `control/curobo_plan_server.py` | cuRobo, over TCP :9880 |
| Perception | `control/humanoid_monitor.py` | AprilTag detection, multi-camera fusion, viser UI |
| Camera / tag library | [`perception/`](perception/) | streaming, tag detection, the tagged-body registry |
| Grippers | `control/humanoid_end_effector_service.py` | ST/FT serial servo hands |
| Motor bindings | `control/hardware_bindings/` | C++/nanobind CAN layer, 244 Hz |

A fuller map — module responsibilities, the process/port diagram, the timing
budget and which tests cover what — is in
[`control/docs/ARCHITECTURE_MAP.md`](control/docs/ARCHITECTURE_MAP.md).

## System diagram

```
        GPU machine                          robot computer
   +--------------------+          +-----------------------------------+
   | curobo_plan_server |<========>| humanoid_curobo_reach   (grasp)   |
   |   TCP :9880        |          | humanoid_auto_operator  (decide)  |
   +--------------------+          | humanoid_monitor        (see)     |
                                   | humanoid_real_env       (act)     |
   cameras --tcp :5555/5556------->|      |  nng :9870 telemetry       |
   (left / right, head gimbal)     |      |  nng :9873 nav command     |
   gripper service <-ipc ee_request|      |  nng :9874 arm targets,    |
   (--ee-service)  --ipc ee_status>|      |    ee_action, gaze targets |
                                   |      v                            |
                                   | hardware_bindings -> CAN 244 Hz   |
                                   +-----------------------------------+
```

The gripper service never sees port 9874: `ee_action` arrives there from the
operator or the reach tool and `humanoid_real_env --ee-service` relays it over
`ipc:///tmp/ee_request.sock`, taking gripper status back on
`ipc:///tmp/ee_status.sock`.

## Hardware this targets

- Duke Humanoid V2: 27 body joints + 4 camera-gimbal joints = 31 actuated.
- Six CAN buses (`can9`, `can21`–`can25`) driving the body motors.
- Two Intel RealSense D436 RGB-D cameras side by side on the head gimbal — left on stream
  port 5555, right on 5556 — streamed as JPEG over TCP. There is no chest camera.
- FEETECH serial-bus servo grippers, one per arm.
- SYD Dynamics TransducerM TM171 IMU (serial, EasyProfile protocol); no foot
  force sensors.

Nothing here assumes a GPU **on the robot**. The plan server does need CUDA, and
is expected to run on a different machine.

## Install

Requires **Python 3.12** — the codebase uses PEP 701 f-strings and will not
parse on 3.11 or earlier. Target platform: Linux x86_64, glibc >= 2.28 (the
pinned wheels are `manylinux_2_28`).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Install `torch` first, from the index that matches your hardware. A plain
`pip install torch==2.9.1` on Linux resolves to the CUDA build and pulls
~3 GB of `nvidia-*` wheels onto a robot computer that has no NVIDIA GPU; the
CPU wheel above runs everything in this repository (the robot itself uses the
`+rocm6.3` build — see the comment block in `requirements.txt`).

`requirements.txt` pins the versions the robot actually runs. Two dependencies
are deliberately absent because they are checkouts, not PyPI packages:

| Dependency | What it supplies | Availability |
|---|---|---|
| `legged_env_v2` (`mj_envs`) | the deploy run directories (policy weights, `env_config.yaml`, the calibrated `robot.xml` **and its meshes**) and the CPU-side Python the robot loop imports: `mj_envs.utils.ik_mink` (IK for `--use-ik` and the arm stream; needs `qpsolvers` + `daqp`, pinned in `requirements.txt` for that reason), `mj_envs.utils.torch_math_utils`, `tasks.visual_manipulation.moving_policy.ReachabilityGate` (operator `--use-gate`) — **all bundled in `control/legged_env_bundle/`, so the robot side needs no checkout**. The plan server's cuRobo planner and scene (`mj_envs.tasks.visual_manipulation.curobo`) are not bundled | the upstream training/deploy repository (General Robotics Lab; clone name `legged_env_dev`). **Not public at the time of this release**; needed only for the plan server (GPU machine) and `rebuild_deploy_model.py`. |
| [cuRobo](https://github.com/NVlabs/curobo) | the plan/MPC solver behind `control/curobo_plan_server.py` | NVIDIA licence; GPU machine only; pin `8e734f3` (`v0.8.0-42`) |

**What runs without a `legged_env_v2` checkout:** everything on the robot —
`humanoid_site.LEGGED_ENV_ROOT` falls back to `control/legged_env_bundle/`, an
upstream-shaped subset (two run directories, 55 mesh/texture files, the two
CPU-side Python packages, the Vicon calibration; `MANIFEST.md5` lists every
file) — plus, as before, the operator decision core
(`humanoid_auto_operator.py` + `auto_operator/`), the perception bundle
(`perception/`: streaming server, standalone tag detector, tagged-body
registry), the gripper service, and every test that does not load the MJCF.
**What does not:** `humanoid_real_env.py` (T3), `humanoid_monitor.py` (T4,
loads the MJCF for the viser scene), `humanoid_curobo_reach.py` (T6),
`humanoid_plan_server_probe.py`, the plan server, and the MJCF-backed tests
(they skip, each saying why). The symptom of a missing or mis-pointed checkout
is, at start-up,

```
FileNotFoundError: deploy model not found at …/legged_env_v2/mj_envs/deploy/runs/<task>/robot.xml — the control stack reads it from legged_env_v2 (mj_envs/deploy/runs/<task>/robot.xml); set HUMANOID_LEGGED_ENV_ROOT in the environment, or LEGGED_ENV_ROOT in control/site_local.py, to that checkout (README 'Install').
```

from `humanoid_real_env.py` and everything else that goes through
`humanoid_base`, or MuJoCo's `ValueError: ParseXML: Error opening file
'…/robot.xml'` from the tools that load the MJCF directly (the monitor, the
reach tool, the probe — the first two only once they build their scene, so
their `--help` still prints). The plan server fails at import with an
`ImportError` naming the same remedies. Set `HUMANOID_LEGGED_ENV_ROOT` in the
environment, or `LEGGED_ENV_ROOT` in `control/site_local.py`, to the checkout
that holds `mj_envs/deploy/runs/<task>/` — only needed for the plan server and
the model-rebuild tool. Without either setting, `control/legged_env_bundle/`
is used: byte-exact copies of the two run directories the robot resolves (the
released checkpoint and the calibrated robot model) together with the meshes
the model references and the Python the robot loop imports, laid out exactly
as the upstream tree is, so nothing else needs configuring.

Then tell the stack where things are:

```bash
cp control/site_local.example.py control/site_local.py   # and edit
```

The perception modules ship in this repository at `perception/`, so nothing
needs configuring to find them. `control/humanoid_site.py` picks that bundled
directory up automatically.

`control/humanoid_site.py` resolves every installation-specific value — repo
paths, the plan-server address, the robot and workstation addresses, the
ports, camera serials, the gripper boards' USB serials — from a `HUMANOID_*`
environment variable, then `site_local.py`, then a portable default, so the
stack runs from any directory, on any machine, as any user. **No path, host or
serial number is hard-coded on that run path.** One default still carries a
rig's identity: `HAND_USB_SERIAL_LEFT` / `HAND_USB_SERIAL_RIGHT` (the gripper
driver boards, the fallback the gripper service uses when the udev symlinks are
absent) default to one rig's boards — set your own in `site_local.py` or
`HUMANOID_HAND_USB_SERIAL_*` (`control/docs/SETUP.md`, section 4). The known
exceptions sit beside the run path: the `/dev/ttyACM*` defaults of two bench
one-offs (`control/disable_servo.py`, `control/torque_sensor_test.py`), and the
`$env{HOME}/repo/…` toolchain paths in the CMake presets
(`control/CMakePresets.json`; the plain `cmake -S control -B control/build …`
build in SETUP does not read them).

For building the C++ motor bindings and setting up CAN, udev and the servo
ports, see [`control/docs/SETUP.md`](control/docs/SETUP.md).

## Running it

The terminal-by-terminal bring-up ladder — and the reason each step is ordered
where it is — is [`control/docs/OPERATIONS.md`](control/docs/OPERATIONS.md).
The short version, once built and configured:

```
T0  python humanoid_setup_can.py                # after ANY power cycle
T1  camera_streaming_utils.py multi-server ...  # ../perception/
T2  python humanoid_end_effector_service.py     # grippers
T3  python humanoid_real_env.py --task <deploy-task> ...   # stand it 2 min
T4  python humanoid_monitor.py ...              # detections + UI on :8080
T5  the plan server, on the GPU machine
T6  python humanoid_curobo_reach.py --cubes ... --execute
```

## Tests

The suite is offline: no robot, no cameras, no GPU.

```bash
cd control && python -m unittest discover -s tests
```

With the C++ extensions built (see [`control/docs/SETUP.md`](control/docs/SETUP.md))
and `legged_env_v2` reachable, the whole suite is expected to pass. Measured on
a clean checkout and a fresh venv built from `requirements.txt` plus a reachable
`legged_env_v2`:

```
Ran 635 tests — OK
```

The tests assert the contract the robot runs as of 2026-08-13: the table is a
height sensor (z prior) rather than a collision body in the MPC world, and the
reverse-leg drift compensation is cancelled.

Two things show up on the way there and are not regressions:

- **Fresh clone, extensions not yet built.** The modules that reach a compiled
  binding error at import time rather than run: `test_ee_zeroing` and
  `test_end_effector_service` (via `humanoid_end_effector_service` ->
  `hardware_bindings.ft_servo`), and `test_gaze_failsafe`, `test_lw2_mirror`,
  `test_obs_frame_fix` and `test_waist_pin` (via `humanoid_real_env` ->
  `humanoid_base` -> `hardware_bindings.motor`). `test_ee_shutdown` spawns the
  gripper service as a subprocess and fails for the same reason;
  `test_grasp_action_id` guards its import and skips. Build the extensions and
  all of them run.
- **Skips** when cuRobo is absent or the deploy model is unreachable. Each skip
  message says why.

Without `legged_env_v2` reachable, many more tests skip; that is expected and
each skip message says why.

## Repository layout

```
control/                   the robot software
  *.py                     entry points, bench tools and the runtime — every
                           top-level module is indexed in docs/ARCHITECTURE_MAP.md
  humanoid_site.py         installation-specific values, all of them
  auto_operator/           the operator's extracted subsystems
  hardware_bindings/       C++/nanobind CAN + IMU bindings (two vendor SDKs
                           redistributed inside, each with a NOTICE.md)
  legged_env_bundle/       the upstream subset the robot side needs (run dirs,
                           meshes, CPU-side Python), upstream layout + MANIFEST.md5
  tests/                   42 files, plain unittest
  docs/
    ARCHITECTURE_MAP.md    modules, ports, timing, test coverage, tool index
    OPERATIONS.md          the runbook
    SETUP.md               building and wiring
    CODING_GUIDELINES.md   the conventions this code follows
    auto_operator_*.md     ADRs, incident register, safety contract
    design-notes/          value ledgers and evaluations, with their dates
perception/                camera streaming, tag detection, tagged bodies
patches/                   local changes to pinned third-party components
```

## Provenance and licence

This is a fresh-history public release of an internal research repository. See
`PROVENANCE.md` for the source commits and for the third-party components —
pinned by URL and commit, or, for the two vendor SDKs under
`control/hardware_bindings/`, redistributed with a `NOTICE.md` beside them.

Licence: **Apache-2.0** — see [`LICENSE`](LICENSE), matching the simulation
repository. Third-party components keep their own licences (see
`PROVENANCE.md`).
