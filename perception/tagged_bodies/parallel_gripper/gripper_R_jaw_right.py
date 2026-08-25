"""Instance: gripper_R_jaw_right — Duke v2 parallel gripper, hand R, right jaw.
Tags (tag36h11, 20 mm): top(+Z)=92, bottom(-Z)=93. Layout mirrors
legged_env_dev tag_layout.py HAND_TAGS."""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.parallel_gripper import make_jaw
from tagged_rigid_body import verify_bodies

cfg = make_jaw(name="gripper_R_jaw_right", jaw="right", top_id=92, bottom_id=93)

if __name__ == "__main__":
    verify_bodies(cfg)
