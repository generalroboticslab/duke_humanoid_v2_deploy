# Deploy model and checkpoint choice

Migrated verbatim from `humanoid_model.py` during the public-release cleanup.
The model path itself now resolves through `humanoid_site`; this is the record
of WHICH export is deployed and WHY one available checkpoint is refused.

See also `control/legged_env_bundle/README.md` (byte-exact copies of the two run
directories the robot resolves, with md5s) and `docs/OPERATIONS.md` T3.

```
# 2026-07-19: v159bMixedArmsCam deploy (upstream master ede1f33). The old notags
# workaround is RESOLVED — this export's tag assets load cleanly. The deploy
# copy has (a) asset paths re-rooted for this directory (meshdir ../../../../),
# (b) the 07-12 hand-eye-calibrated cam_left_rgb site ported from the v83
# deploy (nominal original kept as robot.xml.precalib). Kinematics verified
# IDENTICAL to v83 (joint order + side-home/station FK to 0.0 mm), so every
# baked posture and clearance audit carries over. NOTE the NEW power-on arm
# pose (env_config default_pose changed) — see POWERON_JOINTS in the operator.
# v2_best weights (upstream 4364efa, 2026-07-31), under every successful grasp
# through 08-05. The directory name still says v159b — it has not held v159b
# weights since 07-31; see control/legged_env_bundle/README.md.
#
# DO NOT ADOPT upstream's c163845 "...BankFlatDecoupled" checkpoint for standing
# work. Tried 2026-08-06 and MEASURED, not guessed:
#
#   run 1  c163845 + its camera action_scale (0.44/0.44)   -> cannot hold a stand
#   run 2  c163845 + OUR camera action_scale (0.3/0.2)     -> cannot hold a stand
#
# Run 2 is the whole point: it isolates the variable. The camera authority is
# exonerated; the checkpoint itself is the cause. That matches its design —
# its only change is std_z=0.3 / std_min=0.1 on track_linear_velocity, and
# with std_rel=0.5 that XY floor binds for ||cmd_xy|| < 0.2, which INCLUDES
# exact zero. At cmd 0 there is no walk-vs-stand tie to break, so the
# sharpening only penalises the irreducible gait ripple and the policy strains
# against physics it cannot beat. Upstream's own measurement: cmd-0 joint_power
# 9.38 vs 6.05 (+55%). It was promoted for the low-command dead zone on a
# single seed — a walking benefit this mission never collects.
#
# The fix is written but never trained: sibling class
# ...BankFlatDecoupledStand sets std_stand=0.3, which makes the standing
# population bit-identical to this formula. ASK UPSTREAM FOR THAT RUN before
# anyone tries this lineage again.
```

## 2026-08-24: the released checkpoint is the cosine-LR retrain, seed 1

Decision (the robot owner, with the training side's rule that only the
checkpoint the team is satisfied with is published): the release ships ONE
policy — `...v2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine`,
seed 1 (upstream `e5ecf0d`, staged 2026-08-11, weights md5
`e83cf4b72073928832241c8542d921a3`). It is the checkpoint that carried the
two-cube walk → grasp → carry journey on hardware on 2026-08-12 (69 s, both
tables registered). It is now `humanoid_site.DEPLOY_TASK`'s default.

Two consequences in the code:

1. **The model is decoupled from the checkpoint.** `MJCF_MODEL_PATH` now comes
   from `humanoid_site.DEPLOY_MODEL_TASK` (default: the `...v159bMixedArmsCam`
   directory), not from `DEPLOY_TASK`. That directory's `robot.xml` is the
   hardware-calibrated model written by `rebuild_deploy_model.py` — wrist_1
   `ref` ±π/2, `left_wrist_2` axis turned to the motor's direction, the
   hand-eye-calibrated `cam_left_rgb` site. The Cosine run directory's own
   `robot.xml` is the bare training export and has none of that; loading it
   would break the "encoder 0 == model qpos 0" invariant every wrist fold and
   FK depends on. Internally this was never a risk because the model path was
   a hard-coded constant; the site layer made it follow the task, which this
   entry corrects.
2. **The v159b directory ships no weights.** Its history stays as the record
   above (v159b weights 07-19 → v2_best 07-31 → superseded 08-24); the
   `policy_deployed.pt*` files were dropped from the snapshot.

Known behaviour of the released checkpoint, recorded so nobody rediscovers
it: at zero velocity command, while one arm reaches forward with grasp-phase
observations streaming (arm targets + aimed cameras), the base turns slowly
about yaw (~0.02–0.03 rad/s, below the mission's stillness gate) and stops
the moment those streams reset. It is a property of this checkpoint
generation, not of the control code (the command stays zero; the policy is
the waist's only writer). Seed 0 of the same run tilts continuously on the
hang and was rejected.
