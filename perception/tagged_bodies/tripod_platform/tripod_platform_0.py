"""Instance: tripod_platform — 3-tag fan rack, tag36h11 ids 67-69.

Face->id assignment, confirmed on the printed rack. A viewer FACING the tags
sees, left to right: 67 68 69. Face names are in the raw STEP frame; the BODY
frame is the STEP frame rotated 180 deg about Z (see package docstring), so in
body coordinates the viewer looks along +X and viewer-left is the +Y side:
    right  (STEP -Y = body +Y side) -> 67   (viewer's left)
    center                          -> 68
    left   (STEP +Y = body -Y side) -> 69   (viewer's right)
"""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.tripod_platform import make_tripod_platform
from tagged_rigid_body import verify_bodies

cfg = make_tripod_platform(
    name="tripod_platform",
    face_ids={"right": 67, "center": 68, "left": 69},
    layout="v6",
)

if __name__ == "__main__":
    verify_bodies(cfg)
