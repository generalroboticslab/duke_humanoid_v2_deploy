"""Per-installation settings for visual_servoing.

Copy to `vs_site_local.py` (gitignored) and set only what is true for your lab.
Any name omitted falls back to the portable default in `vs_site`. Names marked
"unused in this tree" are kept for the lab's other robots' scripts; nothing
here reads them.
"""
# REPO_ROOT = "/home/<user>/repo"
# CONTROL_ROOT = "/home/<user>/repo/DukeHumanoidV2/control"      # unused in this tree
# LEGGED_ENV_ROOT = "/home/<user>/repo/legged_env_v2"
# LEGGED_ENV_DEV_ROOT = "/home/<user>/repo/legged_env_dev"        # unused in this tree
# MINK_ROOT = "/home/<user>/repo/mink/src"                        # unused in this tree
# MJLAB_ROOT = "/home/<user>/repo/mjlab/src"                      # unused in this tree

# UR5E_IP = "<ur5e-controller-host>"                              # unused in this tree

# RealSense serials for this rig, in stream order (left 5555, right 5556).
# Read by camera_tag_detection.py and, when non-empty, by the streaming server
# (camera_streaming_utils.py multi-server) as its --devices default.
# CAMERA_SERIALS = ("<serial-a>", "<serial-b>")
