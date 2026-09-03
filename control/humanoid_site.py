"""Site configuration: every value that changes when this stack moves machines.

Purpose
-------
The control stack was written on, and for, one physical installation. Absolute
paths, lab IP addresses and camera serial numbers sat as literals across ~35
call sites, so the code only ran from one directory, on one machine, as one
user. This module is the single place those values are resolved.

Resolution order (first hit wins), per value:

    1. environment variable          -- per-run override, nothing to edit
    2. `site_local.py` beside this file  -- per-installation, gitignored
    3. the default below             -- portable, derived, no lab identity

Nothing here is a control constant. Tuned physical values -- torque limits,
gate thresholds, timing budgets, staging postures -- stay next to the code that
reasons about them, where their forensic comments are. If a value changes what
the robot DOES rather than where it finds a file or whom it talks to, it does
not belong in this file.

Design decisions
----------------
* **Defaults are derived, not literal.** `REPO_ROOT` defaults to
  `~/repo`, which on the robot computer resolves to the exact path the old
  literals named -- so behaviour there is unchanged -- while carrying no
  username. The old literal `<home>/...` was actively wrong for any other
  user: it silently read one user's checkout while running as another.
* **`CONTROL_ROOT` is self-locating** (`__file__`), never configured. A module
  that has to be told where it already is invites the two answers to disagree.
* **Network defaults are loopback.** A service you have not been told how to
  reach is running next to you, not on someone else's lab machine. The real
  address for an installation belongs in `site_local.py` or the environment.
  Rejected: shipping the lab's address as the default -- it is exactly the
  detail that must not leave the building, and a stale address fails silently
  by connecting to the wrong host rather than loudly by refusing.
* Rejected: a YAML/TOML config file. It would need a parser imported before
  `sys.path` is even arranged (`humanoid_monitor` inserts a path at import
  time), and the values are read by module-level code in a dozen files.
  Plain Python with environment overrides has no bootstrap problem.

Assumptions
-----------
`legged_env_v2` and `visual_servoing` sit side by side under `REPO_ROOT` unless
individually overridden. Mesh paths inside the deploy `robot.xml` are relative
to that file's own directory, so the model loads from wherever the tree is --
a full `legged_env_v2` checkout, or `control/legged_env_bundle/`, which mirrors
that layout so the same relative `meshdir` lands on the bundle's own
`asset/create/meshes` (see its README).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# optional per-installation overrides
# --------------------------------------------------------------------------
try:                                    # pragma: no cover - presence varies
    import site_local as _local         # type: ignore
except ImportError:
    _local = None


def _resolve(name: str, default):
    """Resolve one setting: environment, then `site_local.py`, then `default`."""
    env = os.environ.get(f"HUMANOID_{name}")
    if env is not None:
        return env
    if _local is not None and hasattr(_local, name):
        return getattr(_local, name)
    return default


def _path(name: str, default) -> Path:
    return Path(_resolve(name, default)).expanduser()


# --------------------------------------------------------------------------
# filesystem
# --------------------------------------------------------------------------
CONTROL_ROOT: Path = Path(__file__).resolve().parent
"""This package's own directory. Never configured -- it is discovered."""

REPO_ROOT: Path = _path("REPO_ROOT", Path.home() / "repo")
"""Parent directory holding the sibling checkouts (legged_env_v2, ...)."""

_BUNDLED_LEGGED_ENV = CONTROL_ROOT / "legged_env_bundle"

LEGGED_ENV_ROOT: Path = _path(
    "LEGGED_ENV_ROOT",
    _BUNDLED_LEGGED_ENV if _BUNDLED_LEGGED_ENV.is_dir() else REPO_ROOT / "legged_env_v2")
"""The upstream training/deploy tree, or the bundled subset of it.

Supplies the deploy run directories (policy weights, env_config, the calibrated
robot model and its meshes) and the two CPU-side Python packages the robot loop
imports (`mj_envs.utils.ik_mink`, `tasks.visual_manipulation.moving_policy`).

Two layouts, and the default picks whichever is present:

* `control/legged_env_bundle/`: the paths the on-robot stack reads, byte-exact and
  checked by its MANIFEST.md5 (see its README). A release checkout therefore runs
  the robot side with no upstream checkout at all;
* otherwise a full `legged_env_v2` checkout beside this repository
  (`REPO_ROOT/legged_env_v2`) — required for the cuRobo plan server (its planner is
  upstream code with CUDA dependencies) and for `rebuild_deploy_model.py`, neither of
  which the bundle carries.

The bundle is checked first for the same reason VISUAL_SERVOING_ROOT checks its own:
a release checkout must not silently bind to an unrelated sibling that happens to be
named `legged_env_v2`. Measured 2026-08-28 on a clean host, where a stale sibling
checkout without the shipped run directory was preferred over the bundle that had it,
and 27 tests failed with `deploy model not found` while the correct model sat unused
in `control/legged_env_bundle/`.

The plan server needs the upstream tree either way and fails at import naming the
remedy, so set HUMANOID_LEGGED_ENV_ROOT (or `site_local.py`) on the GPU machine; an
explicit setting overrides both layouts.
"""

_BUNDLED_PERCEPTION = CONTROL_ROOT.parent / "perception"

VISUAL_SERVOING_ROOT: Path = _path(
    "VISUAL_SERVOING_ROOT",
    _BUNDLED_PERCEPTION if _BUNDLED_PERCEPTION.is_dir() else REPO_ROOT / "visual_servoing")
"""Perception sources: camera streaming, tag detection, tagged-body registry.

Two layouts are supported, and the default picks whichever is present:

* the public release bundles them at `<repo>/perception/`, so a reader clones
  one repository and nothing needs configuring;
* the internal tree keeps `visual_servoing` as a sibling CHECKOUT, because it
  is a lab-shared library with its own life, so the fallback is
  `REPO_ROOT/visual_servoing`.

An explicit HUMANOID_VISUAL_SERVOING_ROOT or `site_local.py` entry overrides
both. The bundled directory is checked first so that a release checkout can
never silently bind to some unrelated sibling of the same name.
"""

RECORDINGS_DIR: Path = _path("RECORDINGS_DIR", Path.home() / "recordings")
"""Default output directory for the recording tools. Read as the default of
`humanoid_mission_recorder.py --out` (the mission MP4s; since 2026-08-22),
`humanoid_arm_forensics.py --out-dir` and `humanoid_camera_view.py --out-dir`.
Never inside the source tree."""

DEPLOY_TASK: str = str(_resolve("DEPLOY_TASK", "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine"))
"""Run directory under `mj_envs/deploy/runs/` holding the deployed policy
(`policy_deployed.pt`) and its `env_config.yaml`.

The released checkpoint (decided 2026-08-24): the cosine-LR retrain, seed 1 —
upstream commit `e5ecf0d`, weights md5 `e83cf4b72073928832241c8542d921a3` —
the checkpoint that carried the two-cube walk-grasp-carry journey on hardware
on 2026-08-12. The directory that used to be the default,
`...v159bMixedArmsCam`, now serves only as the CALIBRATED ROBOT MODEL (see
`DEPLOY_MODEL_TASK`); its weights are not part of this release.
"""

DEPLOY_MODEL_TASK: str = str(_resolve("DEPLOY_MODEL_TASK", "HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam"))
"""Run directory whose `robot.xml` is the hardware-calibrated deploy model
(written by `rebuild_deploy_model.py`: wrist_1 `ref` ±π/2, `left_wrist_2` axis
turned to the motor's direction, the hand-eye-calibrated `cam_left_rgb` site).
The model is the robot, not the checkpoint, so it is resolved SEPARATELY from
`DEPLOY_TASK`: a run directory's own `robot.xml` is the bare training export
and lacks that calibration — loading it would break the "encoder 0 == model
qpos 0" invariant the wrist fold and the hand-eye FK rely on.
"""

DEPLOY_RUNS_DIR: Path = LEGGED_ENV_ROOT / "mj_envs" / "deploy" / "runs"
DEPLOY_RUN_DIR: Path = DEPLOY_RUNS_DIR / DEPLOY_TASK
MJCF_MODEL_PATH: Path = DEPLOY_RUNS_DIR / DEPLOY_MODEL_TASK / "robot.xml"
VICON_CALIBRATION_PATH: Path = LEGGED_ENV_ROOT / "mj_envs" / "deploy" / "vicon_calibration.yaml"

# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
PLAN_SERVER: str = str(_resolve("PLAN_SERVER", "tcp://127.0.0.1:9880"))
"""cuRobo plan/MPC server endpoint. Runs on a CUDA machine, not the robot computer."""

ROBOT_IP: str = str(_resolve("ROBOT_IP", "127.0.0.1"))
"""Host running `humanoid_real_env.py`, as seen by workstation-side tools."""

WORKSTATION_IP: str = str(_resolve("WORKSTATION_IP", "127.0.0.1"))
"""Workstation that subscribes to telemetry and publishes nav / arm commands.

The robot PUSHES telemetry here, so a stale value sends the robot's state to
whoever now holds that address. Loopback by default for exactly that reason.
"""

VICON_IP: str = str(_resolve("VICON_IP", "127.0.0.1"))
"""Vicon tracker host. Only read when Vicon is explicitly enabled."""

TELEMETRY_PORT: int = int(_resolve("TELEMETRY_PORT", 9870))
"""real_env -> workstation telemetry (nng pub/sub).

Read by `humanoid_real_env` (the publisher bind, since 2026-08-22),
`humanoid_monitor` (its subscriber) and the UDP bench publishers
`humanoid_config`, `humanoid_camera_test`, `humanoid_camera_point`,
`humanoid_pose_finder`, `humanoid_static_stand`, `humanoid_test_motor` and
`move_grip`. The remaining subscribers still dial the literal 9870 -- the
operator, the reach tool (`TelemetryView`, shared by the dual-follow demo),
`humanoid_joint_monkey_hw`, `humanoid_arm_hold_pose`, `humanoid_stage_walk_test`,
`humanoid_arm_forensics`, `humanoid_gimbal_zero_check`, `humanoid_mass_check`,
`humanoid_wiggle_watch`, `humanoid_nav_step_test`, `humanoid_recv_debug` and
`humanoid_viser_viz --port` -- so a non-default value here moves real_env's
socket away from them; change it only together with those call sites.
"""

NAV_COMMAND_PORT: int = int(_resolve("NAV_COMMAND_PORT", 9873))
"""High-level controller -> real_env: the base velocity command (vx, vy, wz).

Read by `humanoid_real_env` only (its subscriber, since 2026-08-22). Every
publisher still binds the literal 9873 -- the operator, the reach tool
(`--journey --execute`), `gamepad.py`, `gampad_gui_control.py`,
`humanoid_nav_step_test` -- so change it only together with them.
"""

ARM_COMMAND_PORT: int = int(_resolve("ARM_COMMAND_PORT", 9874))
"""Operator / reach tool -> real_env: arm targets, gaze targets and ee_action.

Read by `humanoid_real_env` (its subscriber) and `humanoid_mission_recorder`
(its gaze-stream subscriber), both since 2026-08-22. Every binder still uses
the literal 9874 -- the operator, the reach tool, `humanoid_dual_follow_demo`,
`humanoid_joint_monkey_hw`, `humanoid_arm_hold_pose`, `humanoid_stage_walk_test`,
`ee_demo`, `gampad_gui_control`, `humanoid_visual_manip_debug_gui --arm-port`
-- as does the `humanoid_arm_forensics` subscriber; change it only together
with them.
"""

# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------
def _serials(default: str) -> tuple[str, ...]:
    raw = _resolve("CAMERA_SERIALS", default)
    if isinstance(raw, (list, tuple)):
        return tuple(str(s) for s in raw)
    return tuple(s for s in str(raw).replace(",", " ").split() if s)


CAMERA_SERIALS: tuple[str, ...] = _serials("")
"""RealSense serials, in stream order: index 0 -> port 5555 (left), index 1 -> 5556 (right).

Empty by default: serial numbers identify one physical rig and cannot be
guessed. The perception tools read the twin `perception/vs_site.CAMERA_SERIALS`
(env `VS_CAMERA_SERIALS` / `perception/vs_site_local.py`): it is read by
`perception/camera_tag_detection.py` and, when non-empty, by
`camera_streaming_utils.py multi-server` as the `--devices` default (since
2026-08-22; an explicit `--devices` still wins) -- so set THAT one to pin the
camera order; left empty, the order is USB enumeration. This control-side
copy is declared for the same reader contract but nothing in `control/` reads
it.
"""

CAMERA_BASE_PORT: int = int(_resolve("CAMERA_BASE_PORT", 5555))
"""First camera stream port (left = +0, right = +1).

Read by `humanoid_mission_recorder` only (`CAM_PORTS`, since 2026-08-22). The
other consumers still use the literals 5555/5556 -- `humanoid_monitor`
(`--tag-camera-ports`, `--tag-mounts`, `--camera-port` defaults), the operator
(`CAM_PORTS` / `CAM_SITES` / `GAZE_ORDER`), `humanoid_joint_monkey_hw`,
`humanoid_gimbal_zero_check`, the hand-eye tools (`PORT_TO_SITE`),
`calibrate_cam_gear`, `verify_base_frame`, `humanoid_camera_view --base-port`,
`humanoid_recv_debug --camera-port` -- and the streaming server's own
`--base-port` default is 5555, so treat the number as fixed."""

# --------------------------------------------------------------------------
# gripper driver boards
# --------------------------------------------------------------------------
HAND_USB_SERIAL_LEFT: str = str(_resolve("HAND_USB_SERIAL_LEFT", "5B3D045039"))
"""USB serial of the LEFT hand's CH340 gripper driver board, as udev reports it
(`ID_SERIAL_SHORT`) and as it appears in
`/dev/serial/by-id/usb-1a86_USB_Single_Serial_<SERIAL>-if00`.

Read by `humanoid_end_effector_service.py` (`HAND_USB_SERIAL`): when the
`/dev/ttyACMservo{Left,Right}` udev symlink is absent the service falls back to
the by-id entry carrying this serial, so a stale or missing udev rule no longer
fails the open (the 2026-07-31 right-gripper failure). The defaults are the two
boards of ONE rig, kept so that rig's behaviour is unchanged; any other
installation must set its own -- find them as docs/SETUP.md section 4 says:
`udevadm info --name=/dev/ttyACM<N> | grep ID_SERIAL_SHORT`. An empty value
disables the by-id fallback for that side (the configured port is then used
as-is).
"""

HAND_USB_SERIAL_RIGHT: str = str(_resolve("HAND_USB_SERIAL_RIGHT", "5B61033903"))
"""Same, for the RIGHT hand's driver board. See HAND_USB_SERIAL_LEFT."""
