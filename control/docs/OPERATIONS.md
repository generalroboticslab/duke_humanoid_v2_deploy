# Duke Humanoid V2 — operator runbook

How to bring up the standing dual-arm cuRobo/MPC cube-grasping stack, in order,
and what each step is guarding against. Originally written 2026-08-05 as an
operator handoff; the machine-specific values it used to name now live in
`control/site_local.py` (see `site_local.example.py`).

> **This drives a 36 kg humanoid with people beside it.** The order below is not
> a suggestion: T0 before anything, T3 stood for two minutes before T6, and a
> gates-only dry run before `--execute` whenever something changed.

## 0. Machines and roles

Three roles, which may or may not be three boxes:

| role | what runs there | configured by |
|---|---|---|
| **robot computer** | every process below except the plan server. No CUDA needed. | — |
| **GPU machine** | the cuRobo plan/MPC server, port 9880. Needs CUDA. | `PLAN_SERVER` |
| **operator console** | any laptop with ssh; the monitor UI is a browser page | `ROBOT_IP` |

The monitor UI is served on port 8080 by T4. From a laptop, tunnel it:
`ssh -L 18081:localhost:8080 <user>@<robot-host>` then open `localhost:18081`.

**The `taskset` core lists in section 2 are this rig's partition, not a
requirement.** This robot computer has 24 logical CPUs, and the ladder gives
each pinned process a disjoint set of them: the camera server 3 (`10-12`), the
gripper service 1 (`22`), real_env 5 (`0-4`), the monitor 9 (`13-21`), the
recorder 2 (`8-9`); T6 is unpinned on purpose, for the reason its heading
gives. On another machine re-derive the sets from your own `nproc` and keep the
rule rather than the numbers: real_env gets cores nobody else touches, the
camera server and the monitor get separate sets, and T6 stays unpinned.

Copying the literals onto a smaller machine fails two ways. A set entirely
above the last CPU — T2's `-c 22` on anything with fewer than 23 — makes
`taskset` exit 1 with `failed to set pid N's affinity: Invalid argument`, and
the process never starts, which reads as a robot fault rather than a core
count. A set that only partly overlaps fails silently instead: `-c 13-21` on a
16-thread box succeeds and pins the monitor to 3 cores instead of the 9 its
budget assumes, so detections just degrade.

## 1. The plan server (GPU machine)

Start it on the GPU machine, from a checkout of this repository, in an
environment with cuRobo (commit `8e734f3`) installed and `legged_env_v2`
reachable:

```bash
cd <repo> && HUMANOID_LEGGED_ENV_ROOT=<legged_env_v2> python control/curobo_plan_server.py --port 9880
```

Flags: `--port` (default 9880, the port `PLAN_SERVER` points at),
`--hand-z-floor` / `--no-hand-z-floor`, `--hand-floor-weight`. The server
imports `mj_envs.tasks.visual_manipulation.curobo` from `legged_env_v2` at
module level, before argument parsing, and it puts
`humanoid_site.LEGGED_ENV_ROOT` at the FRONT of `sys.path` itself. That is why
the environment variable above is the setting that works: `LEGGED_ENV_ROOT`
defaults to the bundled `control/legged_env_bundle/` whenever that directory
exists — it always does in a release checkout — and the bundle carries the
robot side only, no planner, so a `legged_env_v2` reachable through
`PYTHONPATH` is shadowed by it and the import fails anyway. Set
`HUMANOID_LEGGED_ENV_ROOT` in the environment as above, or `LEGGED_ENV_ROOT`
in `control/site_local.py`; either one moves what goes on the front of
`sys.path`. A missing or mis-pointed checkout fails at import, `--help`
included, with an `ImportError` naming the remedies — its third suggestion,
`PYTHONPATH`, is the one the bundle shadows. `legged_env_v2` is not public at
the time of release (see the repository README, "Install").

Then verify from the robot computer BEFORE touching the robot:

```bash
cd <repo>/control && python humanoid_plan_server_probe.py
```

Three cases must print PASS. The probe drives real MPC sessions with a
simulated executor and no robot; read its docstring for what a FAIL means.

**Restarting it over ssh** — the recipe the reach tool and the plan client
print when the server is unreachable or its CUDA context has died
(`PlanServerDeadError`):

```bash
ssh <gpu-host> 'pkill -f "plan_serve[r]"'
ssh <gpu-host> 'cd <repo> && HUMANOID_LEGGED_ENV_ROOT=<legged_env_v2> setsid nohup python control/curobo_plan_server.py --port 9880 > plan_server.log 2>&1 </dev/null & exit 0'
```

then wait ~40 s for `listening` in `plan_server.log` (the CUDA warm-up) and
re-run the probe. Keep stop and start as **separate ssh calls**: a `pkill`
pattern spelled out in the same command line matches that very command line and
kills the shell before it can relaunch — this has left a stale server running
behind a log that read perfectly healthy. Do not use `ssh -f ... &` either; the
child dies with the session. If the server answers its ping with
`gpu_ok=False`, the CUDA context is faulted on the GPU machine; the reach tool
prints the `nvidia_uvm` recovery for that case.

First solve per server life re-pays ~3.5 s of CUDA graph capture. That is
normal, not a hang.

## 2. Bring-up: the terminal ladder

Every terminal starts in `<repo>/control` with the project environment active.
The plan server (T5) is the one exception: it runs on the GPU machine, section 1.
The `taskset` core lists below are this rig's 24-CPU partition; on any other
machine re-derive them first (section 0).

### T0 — after ANY robot power cycle (mandatory)

```bash
python humanoid_setup_can.py
# verify: all six ERROR-ACTIVE
for c in can9 can21 can22 can23 can24 can25; do
  echo -n "$c: "; ip -details link show $c | grep -o "can state [A-Z-]*" | head -1
done
```

Any `[RECV] error frame` in T3, or a bus stuck ERROR-WARNING → power-cycle
again; if it recurs, inspect the harness before running.

### T1 — cameras

The camera server ships in this repository at
`perception/camera_streaming_utils.py`; the path below is relative to
`control/`, where every terminal in this ladder starts.

Both RealSense cameras sit side by side on the head gimbal. Port order IS
stream order: `--base-port` (5555) is the LEFT camera, the next port (5556) is
the RIGHT one. Downstream code identifies cameras by port, so keep the order
stable across restarts.

The RealSense host packages — the udev rules that make the camera nodes openable
by a non-root user, and the `rs-*` command-line tools — are not installed by the
`pyrealsense2` wheel in `requirements.txt`; they come from Intel's
`librealsense2-udev-rules` and `librealsense2-utils`, the latter supplying
`rs-enumerate-devices` (SETUP.md, "Cameras"). Without them the server cannot open
the cameras as a non-root user and `rs-enumerate-devices` below does not exist.

```bash
taskset -c 10-12 python ../perception/camera_streaming_utils.py \
  multi-server --backend realsense --base-port 5555 --quality 80 \
  --devices <left-serial> <right-serial>
```

The `--devices` order is what pins which camera gets which port. Set it on the
command line as above, or once per rig as `CAMERA_SERIALS` in
`perception/vs_site_local.py` (or `VS_CAMERA_SERIALS`): when that is
non-empty the server uses it as the `--devices` default (since 2026-08-22; an
explicit `--devices` still wins, and a bare `--devices` with no values returns
to auto-discovery). The `humanoid_site` copy of `CAMERA_SERIALS` is a mirror
with no reader — set the perception one. With neither, the server takes the
RealSense devices in USB enumeration order, which can change across replugs and
power cycles — and a left/right swap is SILENT, because everything downstream
(`--tag-mounts`, `PORT_TO_SITE`, the gaze slots) binds by port, not by serial:
tags land in the wrong camera frame and the gaze drives the wrong gimbal while
every log line looks healthy. List the serials with `rs-enumerate-devices -s`.

If a camera drops to USB2 after a power cycle ("Couldn't resolve requests"),
replug or move ports until `lsusb -t` shows 5000M.

### T2 — gripper service

```bash
taskset -c 22 python humanoid_end_effector_service.py
```

Wait for both `Connected left/right hand`.

### T3 — real_env (the robot)

```bash
taskset -c 0-4 python -u humanoid_real_env.py \
  --task <deploy-task> \
  --torque-limit 0.8 --enable-motor true --no-use-ik --grav-comp --ee-service \
  --arm-sender-ip 127.0.0.1 --high-level-controller-ip 127.0.0.1 2>&1 | tee /tmp/realenv.log
```

**Stand it two minutes watching for `██ WATCHDOG ██` before anything else.**

`--task` is the run directory name under `<legged_env_v2>/mj_envs/deploy/runs/`
that holds the policy and its `env_config.yaml`, i.e. `humanoid_site.DEPLOY_TASK`
— which is also its default since 2026-08-22, so the flag names the released
checkpoint explicitly rather than choosing it. The robot MODEL does not follow
the flag: `humanoid_site.DEPLOY_MODEL_TASK` names the run directory whose
`robot.xml` carries the hardware calibration, whichever checkpoint runs. See
[`design-notes/deploy-model-and-checkpoint.md`](design-notes/deploy-model-and-checkpoint.md)
for which export is deployed and which available checkpoint must NOT be adopted
for standing work.

### T4 — monitor / detections

```bash
taskset -c 13-21 python -u humanoid_monitor.py \
  --robot-ip 127.0.0.1 --tag-camera-host 127.0.0.1 \
  --bodies grasp_cube_60mm_a grasp_cube_60mm_b gripper_L_base gripper_R_base 2>&1 | tee /tmp/monitor.log
```

The first line MUST include `[detection] publishing detected targets on ipc://...`.
If detections silently die: `journalctl -k | grep segfault` (libapriltag has
crashed this process before); restart T4.

### T5 — plan server (GPU machine)

Section 1: start `curobo_plan_server.py` on the GPU machine, then run
`humanoid_plan_server_probe.py` from this directory and read three PASS lines
before T6. Re-run the probe after any server restart or server-side change.

### T6 — the grasp (NO taskset: pinning it starves its own control loop)

```bash
python -u humanoid_curobo_reach.py \
  --cubes grasp_cube_60mm_a grasp_cube_60mm_b --no-table-world --chest-home \
  --clearance-floor-mm 25 --execute 2>&1 | tee /tmp/reach.log
```

Drop `--execute` for a gates-only dry run first if anything changed.

### T7 — mission video recorder (optional)

```bash
taskset -c 8-9 python humanoid_mission_recorder.py     # MP4s under --out, default humanoid_site.RECORDINGS_DIR (~/recordings)
```

Optional, and its place in the ladder is loose: it is one more subscriber on
the camera streams (`CAMERA_BASE_PORT` +0/+1, 5555/5556) and on the arm/gaze
stream (`ARM_COMMAND_PORT`, 9874), and it waits for T6's first gaze packet
before it starts a file, so it can be started any time after T1 — before or
after T6. `--out` defaults to `humanoid_site.RECORDINGS_DIR` (`site_local.py` /
`HUMANOID_RECORDINGS_DIR`, default `~/recordings`); pass it to move the MP4s
for one run.

## 3. Placement doctrine (hardware-derived, do not guess)

- Cubes rotated ~45° so TWO tag faces show. One-face evidence is the #1 cause
  of off-centre grasps and mid-track freezes.
- Front-left cube: **|y| ≥ 0.15 m**. Midline-hugging placements summon the
  wind-up IK family — gate F rejects it, you just burn plan retries.
- Cube z ≥ 0.09 m; MPC-validated radial band ~0.45-0.55 m.
- Legs hang STRAIGHT when hanging the robot. Bent legs → torso tilt → geometry
  corruption; three sessions died to this.

## 4. When it retreats: the verdict system

Every abort prints a `[verdict]` banner naming a coded trigger (T01-T14 runtime,
S01-S05 setup), the culprit (arm/joint/cube/geom pair), and the run ends with a
tally histogram. The code table is `RETREAT_TRIGGERS` in
`humanoid_curobo_reach.py`. Rough triage:

| trigger | first suspect |
|---|---|
| T12 / T03 (executor / follow) recurring | wiring or torque |
| T07 / T13 (envelope / no-progress) | perception — tag faces, placement |
| T10 (clearance standoff) | placement too close to the torso |

Any single trigger retreats BOTH arms together — by design.

Key tuned constants live in `humanoid_curobo_reach.py` and
`humanoid_curobo_client.py`, each with its forensic comment:
`MPC_LEAD_SCALE`, `GATE_F_WINDUP_RAD`, `MPC_EXEC_MIN_SPEED_RAD_S`,
`STREAM_FOLLOW_LAG_RAD`, and server-side `MPC_REANCHOR_RAD`.

> **The client lead clamp and the server re-anchor threshold are a PAIR.** Never
> change one without checking the other. On 2026-08-05 a server value of 0.10
> silently confiscated the client's 0.152 lead → sawtooth, zero progress.

## 5. Maturity of the optional modes

- `--decoupled-arms` (plan-0 per-arm solving; the merged route is still
  C-gated) has not been run on hardware.
- `--journey` (walk→grasp→carry legs) HAS been run on hardware — complete
  two-cube missions succeeded in August 2026 — but it is the least mature
  mode: the arrival/coast constants (`JOURNEY_*` in `humanoid_auto_operator.py`)
  are rig-specific and were tuned leg by leg, 30 mm tags are undecodable past
  ~2 m, and walk-in scenarios succeed far less often than the standing grasp.
  Expect to re-tune before relying on it.

## 6. Running the offline suites

No hardware required:

```bash
cd control && python -m unittest discover -s tests
```

See the repository README for the expected result, including the modules that
error at import until the C++ extensions are built, and which tests skip when
cuRobo or the deploy model is absent.
