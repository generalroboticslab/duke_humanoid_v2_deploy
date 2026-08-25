"""Instance: grasp_cube_60mm_b — a SECOND 60 mm AprilTag cube, tag36h11 ids 525-530.

A distinct cube from ``grasp_cube_60mm_a`` (ids 507-512), which is left untouched.
Its printed tags use an EXPLICIT per-face orientation that is 180 deg (about the
face normal) from the default grasp_cube [-r, u, -n] convention, so it is built
with ``make_grasp_cube_explicit`` rather than the default factory.

Per face (cube body frame: X=width, Y=length, Z=height; -Y = front):
    front  -Y  525   tag x=+X   (tag y=-Z, z=+Y)
    back   +Y  530   tag x=+X   (tag y=+Z, z=-Y)
    top    +Z  526   tag x=+X   (tag y=-Y, z=-Z)
    bottom -Z  527   tag x=+X   (tag y=+Y, z=+Z)
    right  +X  528   tag x=+Y   (tag y=-Z, z=-X)
    left   -X  529   tag x=-Y   (tag y=-Z, z=+X)
front/back/top/bottom all pin tag x = +X in the body frame (NOT flipped per face);
right/left pin tag x to the viewer's right when facing that face (+Y / -Y). tag z
is into the face; tag y = z x x. User-specified and verified against the print.
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube_explicit
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube_explicit(
    name="grasp_cube_60mm_b",
    size_m=0.060,
    face_specs={
        "front":  {"n": (0, -1, 0), "id": 525, "tag_x": (1, 0, 0)},
        "back":   {"n": (0, 1, 0),  "id": 530, "tag_x": (1, 0, 0)},
        "top":    {"n": (0, 0, 1),  "id": 526, "tag_x": (1, 0, 0)},
        "bottom": {"n": (0, 0, -1), "id": 527, "tag_x": (1, 0, 0)},
        "right":  {"n": (1, 0, 0),  "id": 528, "tag_x": (0, 1, 0)},
        "left":   {"n": (-1, 0, 0), "id": 529, "tag_x": (0, -1, 0)},
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
