"""Instance: grasp_cube_60mm_f — a 60 mm cube with 50 mm tags on all SIX faces, ids 306-311.

Same as grasp_cube_60mm_e in every way except the tag ids. Distinct from
_a/_b/_c/_d/_e (all untouched). make_grasp_cube_explicit; tag_size=0.050
(black-to-black), tag_border_size=0.005 (per-side) -> 50 + 2*5 = 60 mm fills the
face. (The bottom -Z tag, id 311, was added after the cube was first registered
with five faces — the physical cube's bottom is now tagged too.)

Per face (cube body frame X=width Y=length Z=height, -Y=front); tag_y = z x x,
z = -n (into the face):
    front  -Y 309 tag_x=-Z   back  +Y 310 tag_x=+Z
    top    +Z 306 tag_x=-Y   bottom -Z 311 tag_x=-Y
    right  +X 307 tag_x=-Y   left  -X 308 tag_x=-Y
Intentional (do NOT "correct"): the six orientations are IDENTICAL to _e's, only
the ids differ (303->309, 304->310, 300->306, 301->307, 302->308, 305->311);
right/left both tag_x=-Y; the BOTTOM tag uses tag_x=-Y (tag_y=+X) — 180 deg about
the normal from the +Y this file first carried, and different from the tag_x=+X
every other cube's bottom uses. Set by physical inspection of the printed cube;
NOT a typo.
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube_explicit
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube_explicit(
    name="grasp_cube_60mm_f",
    size_m=0.060,
    tag_size=0.050,
    tag_border_size=0.005,
    face_specs={
        "front": {"n": (0, -1, 0), "id": 309, "tag_x": (0, 0, -1)},
        "back":  {"n": (0, 1, 0),  "id": 310, "tag_x": (0, 0, 1)},
        "top":   {"n": (0, 0, 1),  "id": 306, "tag_x": (0, -1, 0)},
        "bottom": {"n": (0, 0, -1), "id": 311, "tag_x": (0, -1, 0)},
        "right": {"n": (1, 0, 0),  "id": 307, "tag_x": (0, -1, 0)},
        "left":  {"n": (-1, 0, 0), "id": 308, "tag_x": (0, -1, 0)},
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
