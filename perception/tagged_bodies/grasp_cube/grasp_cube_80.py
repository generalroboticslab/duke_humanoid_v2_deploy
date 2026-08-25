"""Instance: grasp_cube_80mm — 80 mm AprilTag cube, tag36h11 ids 513-518."""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_bodies.grasp_cube import make_grasp_cube
from tagged_rigid_body import verify_bodies

cfg = make_grasp_cube(
    name="grasp_cube_80mm",
    size_m=0.080,
    face_ids={"top": 513, "bottom": 514, "front": 515, "back": 516, "left": 517, "right": 518},
)

if __name__ == "__main__":
    verify_bodies(cfg)
