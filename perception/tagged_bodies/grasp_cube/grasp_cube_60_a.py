"""Instance: grasp_cube_60mm_a — 60 mm AprilTag cube, tag36h11 ids 507-512.

The first of the two 60 mm cubes (paired with grasp_cube_60mm_b, ids 525-530).
Uses the default FACE_FRAMES [-r, u, -n] tag orientation.
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube(
    name="grasp_cube_60mm_a",
    size_m=0.060,
    face_ids={"top": 507, "bottom": 508, "front": 509, "back": 510, "left": 511, "right": 512},
)

if __name__ == "__main__":
    verify_bodies(cfg)
