"""Instance: grasp_cube_40mm — 40 mm AprilTag cube, tag36h11 ids 501-506."""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube(
    name="grasp_cube_40mm",
    size_m=0.040,
    face_ids={"top": 501, "bottom": 502, "front": 503, "back": 504, "left": 505, "right": 506},
)

if __name__ == "__main__":
    verify_bodies(cfg)
