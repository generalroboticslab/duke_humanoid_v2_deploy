"""Instance: grasp_cube_60mm_c — a 60 mm AprilTag cube with 50 mm tags, ids 543-548.

Distinct from grasp_cube_60mm_a (507-512) and _b (525-530), all left untouched.
Uses EXPLICIT per-face orientation (make_grasp_cube_explicit), and OVERRIDES the
default marker geometry: tag_size = 0.050 (black-to-black), tag_border_size =
0.005 (per-side quiet zone) -> 50 + 2*5 = 60 mm fills the face exactly.

Per face (cube body frame X=width Y=length Z=height, -Y=front); tag_y = z x x,
z = -n (into the face):
    front  -Y 543 tag_x=+X   back  +Y 545 tag_x=-Z
    top    +Z 547 tag_x=+Y   bottom -Z 546 tag_x=+X
    right  +X 548 tag_x=+Y   left  -X 544 tag_x=+Z
Every face's tag_x differs (+X/-Z/+Y/+X/+Y/+Z) with no unifying rule — this is
intentional and per-face verified; do not flip any face about its normal to make
them "look consistent".
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube_explicit
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube_explicit(
    name="grasp_cube_60mm_c",
    size_m=0.060,
    tag_size=0.050,
    tag_border_size=0.005,
    face_specs={
        "front":  {"n": (0, -1, 0), "id": 543, "tag_x": (1, 0, 0)},
        "back":   {"n": (0, 1, 0),  "id": 545, "tag_x": (0, 0, -1)},
        "top":    {"n": (0, 0, 1),  "id": 547, "tag_x": (0, 1, 0)},
        "bottom": {"n": (0, 0, -1), "id": 546, "tag_x": (1, 0, 0)},
        "right":  {"n": (1, 0, 0),  "id": 548, "tag_x": (0, 1, 0)},
        "left":   {"n": (-1, 0, 0), "id": 544, "tag_x": (0, 0, 1)},
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
