"""Instance: grasp_cube_60mm_e — a 60 mm cube with 50 mm tags on all SIX faces, ids 300-305.

Distinct from _a/_b/_c/_d (all untouched). Uses make_grasp_cube_explicit; marker
override tag_size=0.050 (black-to-black), tag_border_size=0.005 (per-side) ->
50 + 2*5 = 60 mm fills the face. (The bottom -Z tag, id 305, was added after the
cube was first registered with five faces — the physical cube's bottom is now
tagged too.)

Per face (cube body frame X=width Y=length Z=height, -Y=front); tag_y = z x x,
z = -n (into the face):
    front  -Y 303 tag_x=-Z   back  +Y 304 tag_x=+Z
    top    +Z 300 tag_x=-Y   bottom -Z 305 tag_x=-Y
    right  +X 301 tag_x=-Y   left  -X 302 tag_x=-Y
Intentional (do NOT "correct"): right/left both use tag_x=-Y (same direction,
opposite of _b, same as _d); front/back/top all yield tag_y=-X (a natural
consequence of the convention). The BOTTOM tag uses tag_x=-Y (tag_y=+X) —
180 deg about the face normal from the +Y this file first carried, and different
from the tag_x=+X that every other cube's bottom uses (_a/_b/_c/_d/_40mm/_80mm).
Set by physical inspection of the printed cube; NOT a typo.
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube_explicit
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube_explicit(
    name="grasp_cube_60mm_e",
    size_m=0.060,
    tag_size=0.050,
    tag_border_size=0.005,
    face_specs={
        "front": {"n": (0, -1, 0), "id": 303, "tag_x": (0, 0, -1)},
        "back":  {"n": (0, 1, 0),  "id": 304, "tag_x": (0, 0, 1)},
        "top":   {"n": (0, 0, 1),  "id": 300, "tag_x": (0, -1, 0)},
        "bottom": {"n": (0, 0, -1), "id": 305, "tag_x": (0, -1, 0)},
        "right": {"n": (1, 0, 0),  "id": 301, "tag_x": (0, -1, 0)},
        "left":  {"n": (-1, 0, 0), "id": 302, "tag_x": (0, -1, 0)},
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
