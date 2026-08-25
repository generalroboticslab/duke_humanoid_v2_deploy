"""Instance: gripper_R_base — Duke v2 parallel gripper, hand R, base + tag holders.
Pad tags (tag36h11, 20 mm): top_a=94 top_b=95 (holder on base top face),
bottom_a=96 bottom_b=97 (mirrored holder on the bottom face). Layout mirrors
legged_env_dev tag_layout.py HAND_TAGS / base_* SLOTS."""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.parallel_gripper import make_base
from tagged_rigid_body import verify_bodies

cfg = make_base(name="gripper_R_base", top_a=94, top_b=95, bottom_a=96, bottom_b=97)

if __name__ == "__main__":
    verify_bodies(cfg)
