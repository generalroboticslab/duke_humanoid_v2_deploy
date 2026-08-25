"""Instance: grasp_cube_60mm_d — a 60 mm AprilTag cube with 50 mm tags, ids 549-554.

Distinct from grasp_cube_60mm_a (507-512), _b (525-530) and _c (543-548), all
left untouched. Uses EXPLICIT per-face orientation (make_grasp_cube_explicit) and
the same marker override as _c: tag_size = 0.050 (black-to-black),
tag_border_size = 0.005 (per-side) -> 50 + 2*5 = 60 mm fills the face.

Per face (cube body frame X=width Y=length Z=height, -Y=front); tag_y = z x x,
z = -n (into the face):
    front  -Y 552 tag_x=+X   back  +Y 550 tag_x=-Z
    top    +Z 553 tag_x=+Y   bottom -Z 554 tag_x=+X
    right  +X 551 tag_x=-Y   left  -X 549 tag_x=-Y

Intentional (do NOT "correct"):
  - front/back/top/bottom have the SAME orientation as _c's 543/545/547/546
    (only the ids differ). Not a missed edit.
  - right 551 and left 549 both use tag_x = -Y (same direction). This is the
    OPPOSITE of _b (which was +Y / -Y). Also intentional.
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube_explicit
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube_explicit(
    name="grasp_cube_60mm_d",
    size_m=0.060,
    tag_size=0.050,
    tag_border_size=0.005,
    face_specs={
        "front":  {"n": (0, -1, 0), "id": 552, "tag_x": (1, 0, 0)},
        "back":   {"n": (0, 1, 0),  "id": 550, "tag_x": (0, 0, -1)},
        "top":    {"n": (0, 0, 1),  "id": 553, "tag_x": (0, 1, 0)},
        "bottom": {"n": (0, 0, -1), "id": 554, "tag_x": (1, 0, 0)},
        "right":  {"n": (1, 0, 0),  "id": 551, "tag_x": (0, -1, 0)},
        "left":   {"n": (-1, 0, 0), "id": 549, "tag_x": (0, -1, 0)},
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
