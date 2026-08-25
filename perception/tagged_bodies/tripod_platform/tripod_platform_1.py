"""Instance: tripod_platform_v8 — the tall stand-mounted rack, tag36h11 ids 127-129.

Same fan geometry as ``tripod_platform`` (the v6 rack) but on a taller mast: the
three tags sit 30 mm higher and 40 mm further out. Distinct ids, so the two racks
can be in frame at the same time.

Face->id assignment, read off the printed rack. A viewer FACING the tags sees,
left to right: 127 128 129. Face names are in the raw STEP frame; the BODY frame
is the STEP frame rotated 180 deg about Z (see package docstring), so in body
coordinates the viewer looks along +X and viewer-left is the +Y side:
    right  (STEP -Y = body +Y side) -> 127  (viewer's left)
    center                          -> 128
    left   (STEP +Y = body -Y side) -> 129  (viewer's right)
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.tripod_platform import make_tripod_platform
from tagged_rigid_body import verify_bodies

cfg = make_tripod_platform(
    name="tripod_platform_v8",
    face_ids={"right": 127, "center": 128, "left": 129},
    layout="v8",
)

if __name__ == "__main__":
    verify_bodies(cfg)
