# Patches against third-party components

This repository does **not** vendor the components below. Each is pinned to an
upstream URL and commit; where the robot needs a local change, that change is a
patch here rather than a fork. (The two vendor SDKs that ARE redistributed —
the FEETECH sources under `control/hardware_bindings/ft_servo/` and the SYD
Dynamics EasyProfile sources under `control/hardware_bindings/imu/EasyProfile/`
— each carry their own `NOTICE.md`; see `PROVENANCE.md`.)

Apply with `git apply` from inside a fresh checkout of the component at the
pinned commit.

## st_servo

Upstream: `https://github.com/boxiXia/st_servo` — commit `30c65ca`

| Patch | What it does |
|---|---|
| `st_servo/0001-st_servo-local-changes.patch` | +90/-1 in `st_servo.py` |
| `st_servo/0002-add-local-helper-scripts.patch` | adds `st_hand.py`, `st_servos_scan.py`, `example_hls3915.py`, `__init__.py` |

`example_hls3915.py` imports `orca_core` (below); the other three do not.

**Scope note.** The gripper service that actually runs on the robot
(`control/humanoid_end_effector_service.py`) drives the hands through
`hardware_bindings.ft_servo`, our own C++ binding — not through `st_servo`.
Nothing in this repository imports `st_servo` any more: the one script that
did, an end-effector change demo, was removed in the 2026-08-22 runtime pass
(`PROVENANCE.md`) because it also depended on an unshipped `trajectories`
module. These patches matter only for the standalone servo tooling that the
second patch adds to an `st_servo` checkout, not for the live grasp path.

Not included: 644→755 file-mode changes on two files, and the deletion of four
upstream reference documents (a protocol PDF, a register table and two notes)
from the local working copy. Neither is a source change; both are artefacts of
how the tree was copied off the robot.

## Components referenced, not modified

| Component | Upstream | Pinned commit | Used for |
|---|---|---|---|
| FTServo_Python | `https://gitee.com/ftservo/FTServo_Python` | `a203373` | FEETECH Python SDK reference |
| FTServo_Linux | `https://gitee.com/ftservo/FTServo_Linux` | `064a6db` | FEETECH C++ SDK reference |
| orca_core | `https://github.com/orcahand/orca_core` | `9d03675` | only by `example_hls3915.py` |
| cuRobo | `https://github.com/NVlabs/curobo` | `8e734f3` (`v0.8.0-42`) | plan server, on the GPU machine |

The three FEETECH/orca checkouts carried **no** local source change: every file
git reported as modified was a 644→755 mode difference from copying the tree
off the robot over CIFS — 192 files, zero content. cuRobo is under NVIDIA's
licence and is never copied.
