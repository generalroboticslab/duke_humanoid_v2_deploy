# Provenance

This repository is a **fresh-history public release**. It was assembled from two
internal research repositories by copying cleaned file trees; none of the
internal git history is published, and none is recoverable from here.

The internal history was withheld deliberately. It contains lab-internal IP
addresses, absolute home-directory paths, ~1 GB of recorded robot runs, and
collaborators' internal commit messages — none of which belong in a public
repository, and none of which can be removed from history after the fact.

## Sources

| Directory here | Internal repository | Branch | Commit |
|---|---|---|---|
| `control/` | DukeHumanoidV2 | `release/public-cleanup` | `e730d885f361efadfa9b4fe410e70ce45a18078f` |
| `control/hardware_bindings/` | hardware_bindings (a submodule internally; source inlined here) | `release/public-cleanup` | `9e1d8983d1402b68dd157fdd7dee196736b21d47` |
| `perception/` | visual_servoing (subset — see below) | `release/public-cleanup` | `fca8d186954cf87c28ad6f56ad6e17743384c2b8` |

All three were snapshotted before any cleanup began: DukeHumanoidV2 at
`eb6e917` (`f92d149` plus 26 paths of hardware-validated work folded into that
snapshot commit), hardware_bindings at `244d408`, visual_servoing at `2f9576b`.

### Why one repository, when internally there are two

Internally, `visual_servoing` is a lab-shared perception library used by other
robots. From outside, this is one project: the control stack cannot run without
those modules, and requiring a second clone plus a path variable buys the reader
nothing. This release is a snapshot, not a mirror, so preserving the internal
split would have been fidelity to our filing system rather than to the work.

`perception/` therefore contains **the import closure** the control stack
uses — 10 modules and the tag assets they load — **plus the fabrication
assets** under `perception/asset/`: the printable tag sheets with their
generators, the tag-cube CAD and the tripod base. Nothing imports those; they
are there so the physical tagged objects can be reproduced to the geometry the
registry describes. The rest of `visual_servoing` targets other robots (Unitree
G1, UR5e, Pinocchio/mink experiments) and, with ~146 MB of Unitree's G1 meshes,
is not part of this project and is not redistributed here.

## What was removed, and why

| Removed | Reason |
|---|---|
| `control/recordings/`, `control/logs*/` | ~1.1 GB of recorded runs; regenerate with `humanoid_mission_recorder.py` |
| `control/calibration/` | rig-specific hand-eye captures — see `control/CALIBRATION.md` |
| `control/reference/` | a vendor motor manual; the manufacturer's document to distribute |
| `control/ft_servo/` | an unused divergent fork of the FEETECH SDK that `hardware_bindings/ft_servo/` already carries, with attribution |
| `control/st_servo/` | third-party; pinned below and patched, not vendored |
| `argus_*`, `modular_drone_*`, `tof_a010/` | code for two other robots |
| `perception/common/urdf_utils.py` | unreachable from the closure; pulled open3d, yourdfpy, networkx and pycollada with it |
| build artefacts, `__pycache__`, `.mypy_cache`, editor and local tooling state | not source |
| internal audit and machine-topology documents | lab-internal by construction |

## The released checkpoint

Decided 2026-08-24, with the training side's rule that only the checkpoint the
team is satisfied with is published: the release ships **one** policy, the
cosine-LR retrain of the `...BankFlatDecoupled` family, **seed 1** (upstream
`e5ecf0d`, weights md5 `e83cf4b72073928832241c8542d921a3`) — the checkpoint
that carried the two-cube walk → grasp → carry journey on hardware on
2026-08-12. It is `humanoid_site.DEPLOY_TASK`'s default and is snapshotted under
`control/legged_env_bundle/`. The robot model is resolved separately
(`humanoid_site.DEPLOY_MODEL_TASK` → the calibrated `robot.xml` of the
`...v159bMixedArmsCam` directory), because a run directory's own `robot.xml` is
an uncalibrated training export. The earlier weights that lived in that
directory (v159b, then v2_best) are not part of this release. The record of
the decision and of the refused checkpoints is
`control/docs/design-notes/deploy-model-and-checkpoint.md`.

## The bundled upstream subset (`control/legged_env_bundle/`)

Decided 2026-08-24 so the robot side runs from this repository alone: the part
of `legged_env_v2` the on-robot stack reads is copied here in the upstream
layout — the two deploy run directories, the 55 mesh/texture files the
calibrated `robot.xml` references, the Vicon calibration, and two CPU-side
Python packages (`mj_envs.utils.ik_mink` and friends, 4 modules; the
`ReachabilityGate` closure, 5 modules + two TorchScript scorers). 86
files, 63.9 MB; every file's md5 is in its `MANIFEST.md5`. Source:
upstream commit `6a486fb` (2026-08-11) as checked out on the robot on
2026-08-24. The Python is the training repository's code (author: the
upstream maintainer); it is redistributed here with the project owners'
agreement under this repository's Apache-2.0 licence, unmodified — the same
licence the training repository itself carries. Not bundled: the
cuRobo planner package and the assets `rebuild_deploy_model.py` needs (see the
bundle README).

## Configuration, not identity

No path, host address or camera serial number is hard-coded on the run path.
`control/humanoid_site.py` and `perception/vs_site.py` resolve every
installation-specific value from an environment variable, then an optional
gitignored `*_local.py`, then a portable default. Network defaults are
loopback, so a missing setting fails by refusing to connect rather than by
quietly reaching some other machine. `CAMERA_SERIALS` defaults to empty; the
copy that is read is `perception/vs_site.CAMERA_SERIALS` — by
`perception/camera_tag_detection.py` and, when it is non-empty, by the
streaming server as its `--devices` default (an explicit `--devices` still
wins; with neither, the server takes whatever is attached, in USB enumeration
order). The gripper driver boards' USB serials (`HAND_USB_SERIAL_LEFT` /
`HAND_USB_SERIAL_RIGHT` in `control/humanoid_site.py`, the fallback the gripper
service uses when the udev symlinks are absent) are the one setting whose
default still names a particular rig's hardware; `control/docs/SETUP.md`,
section 4, says how to find and set your own.

The known exceptions sit beside that path and are documented where they live:
the `/dev/ttyACM*` defaults of two bench one-offs (`control/disable_servo.py`,
`control/torque_sensor_test.py`), and the `$env{HOME}/repo/…` toolchain paths
in the CMake presets (`control/CMakePresets.json`), which the plain
`cmake -S control -B control/build …` build in `control/docs/SETUP.md` does
not read.

## Third-party components

Referenced by URL and commit, and not vendored — with two exceptions: the two
vendor SDKs that the C++ extensions compile against are redistributed under
`control/hardware_bindings/`, each with a `NOTICE.md` beside it (the last two
rows below). Local changes, where they exist, are patches under `patches/` —
see `patches/README.md`.

| Component | Upstream | Pinned commit | Licence |
|---|---|---|---|
| st_servo | https://github.com/boxiXia/st_servo | `30c65ca` | Apache-2.0 |
| FTServo_Python | https://gitee.com/ftservo/FTServo_Python | `a203373` | MIT |
| FTServo_Linux | https://gitee.com/ftservo/FTServo_Linux | `064a6db` | MIT |
| orca_core | https://github.com/orcahand/orca_core | `9d03675` | see upstream |
| cuRobo | https://github.com/NVlabs/curobo | `8e734f3` (`v0.8.0-42`) | NVIDIA licence — never copied |
| FEETECH SDK (FTServo_Linux sources) | as above; redistributed under `control/hardware_bindings/ft_servo/` | `064a6db` | MIT, with attribution — `ft_servo/NOTICE.md` |
| SYD Dynamics EasyProfile | SYD Dynamics ApS, EasyProfile SDK for the TransducerM TM3xx IMU (generated by the vendor's IMU Assistant; no upstream URL in the sources); redistributed under `control/hardware_bindings/imu/EasyProfile/` (8 files) | version 1.1.9 (R), Jul 27 2016, per the headers | BSD-2-Clause, (c) 2017 SYD Dynamics ApS, with attribution — `imu/EasyProfile/NOTICE.md` |

The FEETECH SDK sources under `control/hardware_bindings/ft_servo/` ARE
redistributed, under their MIT licence, with attribution and a record of the one
local bug fix — see `control/hardware_bindings/ft_servo/NOTICE.md`. The SYD
Dynamics EasyProfile sources under `control/hardware_bindings/imu/EasyProfile/`
(the TM3xx IMU protocol library that the built `imu_nanobind` extension links)
are likewise redistributed, unmodified as received, under their BSD-2-Clause
licence — see `control/hardware_bindings/imu/EasyProfile/NOTICE.md`, which also
carries the licence text for binary distributions of the extension.

## Licence

**Apache-2.0**, copyright holder General Robotics Lab, Duke University — see
`LICENSE`. Chosen to match `duke_humanoid_v2_simulation`, so the two halves of
the project ship under one licence and the training-repository code
redistributed in the deploy bundle above rides on its own terms rather than a
second set. Third-party components keep their own licences: the FEETECH
SDK sources redistributed under `control/hardware_bindings/ft_servo/` stay
under their MIT licence with attribution (`NOTICE.md` there), the SYD Dynamics
EasyProfile sources under `control/hardware_bindings/imu/EasyProfile/` stay
under their BSD-2-Clause licence with attribution (`NOTICE.md` there), and the
pinned components in the table above stay under theirs.

## Verification at release time

- Offline test suite from a fresh clone and a fresh virtualenv built from
  `requirements.txt` alone: passes completely once the C++ extensions are
  built and `legged_env_v2` is reachable. The numbers are in the README's
  Tests section.
- Differential replay of the autonomous operator against the pre-cleanup
  snapshot: 13/13 scenarios bit-identical.
- No remote is configured in this repository, and nothing has been pushed.

### Release polish (2026-08-21)

- The tagged-body meshes (`perception/tagged_bodies/**/*.obj`,
  `perception/asset/**/*.obj`) are back in the index. A blanket `*.obj`
  ignore rule, meant for C object files, had silently excluded them, so a
  fresh clone had no mesh for the monitor's gripper bodies.
- A CIFS symlink stub (`control/visual_servoing`, 68 bytes of `IntxLNK` data
  pointing at an internal path, not a symlink) was removed.
- Seven stale tests were updated to the 2026-08-13 contract: the table is a
  height sensor (z prior), and the reverse-leg drift compensation is
  cancelled.
- `control/docs/SETUP.md` and `control/docs/ARCHITECTURE_MAP.md` were
  rewritten for the released tree.
- 2026-08-22: personal-name attributions in comments and docs were replaced
  by role words ("upstream" for the training/simulation side; the dated patch
  markers keep their dates). The pinned third-party repository URLs are
  unchanged. The `LICENSE` file was added (MIT at the time; relicensed to
  Apache-2.0 on 2026-08-25 to match the simulation repository).

### Release polish (2026-08-22, runtime pass)

A second pass, after the owner approved changes that touch code. Removed:

- `control/humanoid_arm_replay_stream.py`, `control/humanoid_arm_teach_replay_ref.py`,
  `control/humanoid_end_effector_change_demo.py` — the three arm teach/replay
  demos. Each imported a `trajectories` module that exists neither in this
  tree nor in the internal one, so none of them could run as released. The
  design history stays in `control/plan/arm_teach_replay.md`.
- `control/gl_setup_can.py` — a copy of `humanoid_setup_can.py` with the
  interface list set to the single-motor bench bus; the behaviour is now
  `python humanoid_setup_can.py --interfaces can26`.
- `control/common/buffer.py` and `perception/common/profiling_utils.py` — no
  importer anywhere in the tree. (`perception/` still holds the 10-module
  import closure named above; `profiling_utils` was never part of it.)
- the `LegOdometry` design study inside `control/humanoid_curobo_reach.py`
  (`leg_enc`, `LegOdometry`, `LEG_JOINTS`, `FOOT_SITES`) — never wired in,
  no caller.
- the stray 2-byte root `__init__.py` — nothing imports the checkout as a
  package.

Renamed: `control/common` → `control/ipc` (the nng publisher / subscriber and
`sleep_test.cpp`), which ends the clash of three packages named `common` on
the control side; `control/docs/ARCHITECTURE_MAP.md`, section 2, records
what went with it. The rest of the pass is behaviour-preserving except where
a document here says otherwise: the C++ build drops the unused TBB dependency
and makes the extensions relocatable (`control/docs/SETUP.md`, section 2);
`humanoid_site` settings that were declared but unread are now read (the
ports by `humanoid_real_env`, the monitor and the mission recorder,
`RECORDINGS_DIR` by the recorder, the gripper-board serials by the gripper
service, `vs_site.CAMERA_SERIALS` by the streaming server); eight bench
scripts that used to act at import now act only when invoked; the plan
server, `humanoid_real_env` and the reach tool say where `legged_env_v2` is
looked for when it is missing.
