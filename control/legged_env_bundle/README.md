# legged_env_bundle — the upstream subset the robot side needs

`legged_env_v2` is the training/deploy repository (the RL and simulation side of
this project, not public at the time of release). The on-robot stack reads a
small, fixed part of it. This directory is that part — 87 files,
63.9 MB — laid out exactly as the upstream tree is, so that
`humanoid_site.LEGGED_ENV_ROOT` can simply point here when no checkout of
`legged_env_v2` sits beside this repository (that is its default; an explicit
`HUMANOID_LEGGED_ENV_ROOT` or `site_local.py` entry overrides it). Every file
is listed with its md5 in `MANIFEST.md5`; they are byte-exact copies from
upstream commit `6a486fb` (2026-08-11) as checked out on the robot on
2026-08-24, with `env_config.yaml` carrying the robot's arm-home block.

| Path | What it is | Role |
|---|---|---|
| `mj_envs/deploy/runs/…BankFlatDecoupledCosine/` | `policy_deployed.pt` (seed 1), `env_config.yaml` | **the released checkpoint** (`humanoid_site.DEPLOY_TASK`) |
| `mj_envs/deploy/runs/…v159bMixedArmsCam/` | `robot.xml` + calibration history, `env_config.yaml` + history, `_audit_cube_0.03.xml` | **the calibrated robot model** (`humanoid_site.DEPLOY_MODEL_TASK`); no weights |
| `asset/create/meshes/`, `asset/duke_v2/…` | 55 mesh and tag-texture files (57 MB + the two old-wrist meshes) | everything `robot.xml` and its history variants reference through `meshdir="../../../../asset/create/meshes"` |
| `mj_envs/utils/` (`ik_mink`, `ik_mink_local`, `torch_math_utils`, `humanoid_v21_collision_exclusions`) | 4 modules, pure Python + CPU libraries | imported by `humanoid_real_env.py` (`--use-ik`, the arm stream) |
| `mj_envs/tasks/visual_manipulation/` (`moving_policy`, `control_vec`), `mj_envs/tasks/camera_perception.py`, `mj_envs/asset_zoo/grasp_planning.py`, `mj_envs/utils/reachability.py` + `mj_envs/asset_zoo/cache/*.pt` | `ReachabilityGate`: 5 modules + two TorchScript scorers (1.7 MB), torch on CPU | imported by `humanoid_auto_operator.py` (`--use-gate`, on by default) |
| `mj_envs/deploy/vicon_calibration.yaml` | Vicon-to-robot calibration | `humanoid_real_env.py --vicon` |

**Not bundled, by design:** the cuRobo planner and scene
(`mj_envs.tasks.visual_manipulation.curobo`, ~15 k lines, needs NVIDIA cuRobo,
mjlab and a CUDA GPU) — the plan server (`curobo_plan_server.py`) runs on the
GPU machine from a full `legged_env_v2` checkout; and `humanoid_v21_constants`
+ the full asset tree that `rebuild_deploy_model.py` (a maintenance tool)
needs. The `ImportError` those two raise names the checkout they want.

## The checkpoint

`policy_deployed.pt` in the Cosine directory is the cosine-LR retrain of the
`...BankFlatDecoupled` family, **seed 1** — upstream commit `e5ecf0d`, staged
2026-08-11, the checkpoint that carried the two-cube walk → grasp → carry
journey on hardware on 2026-08-12. Released by decision of 2026-08-24; the
record, including the checkpoints that were refused, is
`control/docs/design-notes/deploy-model-and-checkpoint.md`.

Its `env_config.yaml` is upstream's verbatim except the arm `default_pose`
block, which is ours (the 2026-07-31 "sim home == cuRobo planning home"
alignment; see the comment block inside). Interface: obs 120, act 31, 39
tensors, `has_velocity_estimator: true`, camera `action_scale` 0.44.

The Cosine run directory's own `robot.xml` is deliberately NOT snapshotted: it
is the bare training export (no wrist_1 `ref`, `left_wrist_2` axis `-1 0 0`,
nominal camera site). The robot runs on the calibrated model below.

Known behaviour, so nobody rediscovers it: at zero velocity command, while
one arm reaches forward with grasp-phase observations streaming, the base
turns slowly about yaw (~0.02–0.03 rad/s) and stops when those streams reset.
A property of the checkpoint, not of the control code.

## The model

`robot.xml` in the v159b directory is the LIVE deploy model: the new-wrist
(2026-07-28 redesign) export with the hardware joint-zero calibration folded
in by `rebuild_deploy_model.py` — wrist_1 `ref` ±π/2, `left_wrist_2` axis
turned to the motor's direction, the 2026-07-12 hand-eye-calibrated
`cam_left_rgb` site. Every tool (reach, monitor, gates, FK, calibration) loads
this one file. The directory's name is historical: it held the deployed
weights (v159b from 07-19, v2_best from 07-31) until the 2026-08-24 checkpoint
decision; those weights are not part of this release.

| File | What it is |
|---|---|
| `robot.xml` | live deploy model (new wrist, 2026-07-28, calibrated) |
| `robot.xml.prev` | one rebuild before the live file (2026-07-28 17:41): wrist_1 `ref` already in, `left_wrist_2_joint` axis not yet flipped, rack meshes without `content_type`, older inertia export |
| `robot.xml.precalib` | new-wrist model with the NOMINAL `cam_left_rgb` site (before the 2026-07-12 hand-eye correction was re-applied) and no wrist_1 `ref` |
| `robot.xml.before_wrist1_ref` | new-wrist model with the calibrated `cam_left_rgb` site, before the wrist_1 `ref` (±π/2) was added |
| `robot.xml.oldwrist_backup_0728` | old-wrist model (2026-07-19 export, calibrated camera site), backed up at the 2026-07-28 wrist swap — **byte-identical to `robot_OLDWRIST.xml`** |
| `robot.xml.precalib.oldwrist_backup_0728` | old-wrist model with the nominal `cam_left_rgb` site |
| `robot_OLDWRIST.xml` | twin of `robot.xml.oldwrist_backup_0728` (same md5) |
| `env_config.yaml` | the config the v2_best weights ran with — **byte-identical to `env_config.yaml.simhome_0731`**; its arm `default_pose` is the same 2026-07-31 alignment the released checkpoint uses |
| `env_config.yaml.simhome_0731` | twin of `env_config.yaml` (same md5) |
| `env_config.yaml.chesthome_0730` | the 2026-07-30 chest-home arm default, superseded the next day |
| `_audit_cube_0.03.xml` | cube-audit scene for offline gate tests |

The twins cost nothing in the repository (git stores one blob). Do NOT "clean
up" the backup variants; every one of them was the rollback path for a hardware
calibration step. Verify with `md5sum *`.

## How the code reaches it

`humanoid_site` resolves two run directories under
`<LEGGED_ENV_ROOT>/mj_envs/deploy/runs/`:

- `DEPLOY_TASK` → `policy_deployed.pt` and `env_config.yaml`
  (`humanoid_real_env --task` defaults to it);
- `DEPLOY_MODEL_TASK` → `robot.xml` (`humanoid_model.MJCF_MODEL_PATH`).

They are separate on purpose: the model is the robot, the checkpoint is a
choice, and a run directory's own `robot.xml` is an uncalibrated export. With
`LEGGED_ENV_ROOT` pointing here, `robot.xml`'s relative `meshdir` lands on
`asset/create/meshes/` in this directory, and `humanoid_real_env.py` /
`humanoid_auto_operator.py` find `mj_envs.utils.ik_mink` and
`tasks.visual_manipulation.moving_policy` here through the same
`sys.path.insert(0, LEGGED_ENV_ROOT)` they use against a full checkout. The
release test suite passes with `HUMANOID_REPO_ROOT` pointed at an empty
directory, which is how this bundle was verified. The only in-tree reader that
names this directory directly is `control/tests/test_gaze_failsafe.py`.

## Other run directories named in `humanoid_real_env --task`

The `--task` choices in `humanoid_real_env.py` list the run directories that
were staged beside these in the upstream `deploy/runs` tree. None of the
others is part of this snapshot; the ledger is kept so the code can carry
one-line tags. Dates are staging dates; "upstream" is the training repository.

- `HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam` (2026-07-19) — the
  calibrated-model directory above. Held v159b weights, then v2_best (upstream
  `4364efa`, 2026-07-31) until 2026-08-24.
- `...v2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled` (2026-08-06,
  upstream `c163845`) — same net (35 tensors, obs 120, act 31) and joint
  definitions as v2_best; training run `2026-08-05_00-34-46 model_0015000`.
  **REJECTED 2026-08-06: cannot hold a stand** (`std_min` binds at cmd 0,
  +55 % joint power).
- `...BankFlatDecoupledCosine` (2026-08-11, upstream `e5ecf0d`) — the
  released checkpoint above (seed 1). Seed 0 of the same run waist-tilts
  continuously on the hang and was rejected.
- `...BankFlatDecoupledMuonLr1e3CosineAll` (2026-08-12, upstream `64168a6`) —
  Muon optimiser (lr 1e-3) + CosineAll retrain of the Cosine family. UNTESTED
  ON HARDWARE.
- `...BankFlatDecoupledCosineAR` (2026-08-13, upstream `c0b6b18`) — the "AR"
  retrain of the Cosine family, two seeds; interface identical to the released
  Cosine (drop-in). UNTESTED ON HARDWARE.
- `v2_best` (2026-07-31, upstream `4364efa`) — the run-directory export of
  the weights that lived in the v159b directory from 07-31 to 08-24; its own
  `robot.xml` is uncalibrated.
