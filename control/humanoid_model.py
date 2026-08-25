"""Single source of truth for the robot MJCF model path.

The robot-side control stack (humanoid_base.py / humanoid_real_env.py) and the
workstation-side viewers and calibration (humanoid_monitor.py,
humanoid_viser_viz.py, humanoid_recv_debug.py, humanoid_handeye_calibration.py)
must all load the SAME model: the deploy-exported robot.xml with the actuated
head-camera gimbal (4 joints; 27 body + 4 cam = 31 total) and welded parallel
grippers. A mismatch breaks telemetry parsing (joint_pos length) and hand-eye FK.

The path is task-specific (regenerated per policy export) and is resolved by
`humanoid_site` from LEGGED_ENV_ROOT + DEPLOY_MODEL_TASK — point those at your own
checkout rather than editing this file. For a no-cam robot (27 motors) the
static XML is `<legged_env>/asset/create/humanoid_v21.xml`.

Mesh paths inside robot.xml are relative to the XML's own directory, so the
model loads from wherever the tree lives -- inside a full legged_env_v2
checkout (robot.xml's meshdir is relative to that tree; the bundled
legged_env_bundle copy alone does not load).
"""

# The deployed export is v159bMixedArmsCam (upstream master ede1f33): tag assets
# load cleanly, asset paths re-rooted for this directory, and the 07-12
# hand-eye-calibrated cam_left_rgb site ported from the v83 deploy. Kinematics
# verified IDENTICAL to v83 (joint order + side-home/station FK to 0.0 mm), so
# every baked posture and clearance audit carries over. NOTE the power-on arm
# pose changed with it (env_config default_pose) -- see POWERON_JOINTS.
#
# DO NOT ADOPT upstream's c163845 '...BankFlatDecoupled' checkpoint for standing
# work: it cannot hold a stand, MEASURED on 2026-08-06 with the camera
# authority isolated and exonerated. The fix exists as a sibling class but was
# never trained -- ask upstream for that run first.
#
# Both findings in full, with the numbers and the run table:
#     docs/design-notes/deploy-model-and-checkpoint.md

from humanoid_site import MJCF_MODEL_PATH as _MJCF_MODEL_PATH

MJCF_MODEL_PATH = str(_MJCF_MODEL_PATH)   # str: callers pass it to mujoco/os APIs
