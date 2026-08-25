"""Per-installation settings. Copy to `site_local.py` and edit; that name is gitignored.

`humanoid_site` reads these only when the matching `HUMANOID_<NAME>` environment
variable is unset, so this file is the convenient way to pin an installation and
the environment stays available for one-off overrides.

Every name here is optional -- omit one and `humanoid_site`'s portable default
applies. Only the values that are true for YOUR machines belong in this file.
"""

# Parent directory holding the sibling checkouts (legged_env_v2, visual_servoing).
# REPO_ROOT = "/home/<user>/repo"

# Individual overrides, if the checkouts are not siblings:
# LEGGED_ENV_ROOT = "/data/legged_env_v2"
# VISUAL_SERVOING_ROOT = "/data/visual_servoing"

# Run directory under mj_envs/deploy/runs/ that holds the deployed weights and
# env_config.yaml (the released checkpoint), and the one whose robot.xml is the
# hardware-calibrated model. They are resolved separately; see humanoid_site.py.
# DEPLOY_TASK = "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine"
# DEPLOY_MODEL_TASK = "HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam"

# cuRobo plan/MPC server. Needs a CUDA GPU, so it is normally NOT the robot's
# onboard computer. Point this at the machine running the server.
# PLAN_SERVER = "tcp://<gpu-host>:9880"

# Host running humanoid_real_env.py, as seen from workstation-side tools.
# ROBOT_IP = "<robot-host>"

# Vicon tracker host; only read when Vicon is explicitly enabled.
# VICON_IP = "<vicon-host>"

# RealSense serials in stream order: index 0 -> port 5555, index 1 -> 5556.
# Leave unset to let the streaming server enumerate whatever is attached.
# NOTE: the perception tools (camera_tag_detection.py and the streaming
# server's --devices default) read the twin in perception/vs_site_local.py --
# set that one to pin the camera order.
# CAMERA_SERIALS = ("<serial-a>", "<serial-b>")

# USB serials (udevadm info --name=/dev/ttyACM<N> | grep ID_SERIAL_SHORT) of the
# two CH340 gripper driver boards. The EE service falls back to
# /dev/serial/by-id/*<serial>* when the udev symlink is missing. The shipped
# defaults are one rig's boards -- set your own (SETUP.md section 4).
# HAND_USB_SERIAL_LEFT = "<left-board-serial>"
# HAND_USB_SERIAL_RIGHT = "<right-board-serial>"

# Where recordings and mission MP4s are written. Keep this OUT of the source tree.
# RECORDINGS_DIR = "/data/recordings"
