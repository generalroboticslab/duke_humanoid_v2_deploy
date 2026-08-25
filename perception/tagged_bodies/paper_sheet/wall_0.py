"""Auto-generated combined wall config for wall_0"""
import os, sys
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TagConfig, verify_bodies
from tagged_bodies.paper_sheet import PaperSheetConfig

cfg = PaperSheetConfig(
    name="wall_0",
    width_m=4.3180,
    height_m=0.2794,
    tag_size=0.1800,
    tags={
        "tag_401": TagConfig(id=401, pos=[-2.051050, -0.002500, -0.000300]),
        "tag_402": TagConfig(id=402, pos=[-1.835150, -0.002500, -0.000300]),
        "tag_403": TagConfig(id=403, pos=[-1.619250, -0.002500, -0.000300]),
        "tag_404": TagConfig(id=404, pos=[-1.403350, -0.002500, -0.000300]),
        "tag_405": TagConfig(id=405, pos=[-1.187450, -0.002500, -0.000300]),
        "tag_406": TagConfig(id=406, pos=[-0.971550, -0.002500, -0.000300]),
        "tag_407": TagConfig(id=407, pos=[-0.755650, -0.002500, -0.000300]),
        "tag_408": TagConfig(id=408, pos=[-0.539750, -0.002500, -0.000300]),
        "tag_409": TagConfig(id=409, pos=[-0.323850, -0.002500, -0.000300]),
        "tag_410": TagConfig(id=410, pos=[-0.107950, -0.002500, -0.000300]),
        "tag_411": TagConfig(id=411, pos=[0.107950, -0.002500, -0.000300]),
        "tag_412": TagConfig(id=412, pos=[0.323850, -0.002500, -0.000300]),
        "tag_413": TagConfig(id=413, pos=[0.539750, -0.002500, -0.000300]),
        "tag_414": TagConfig(id=414, pos=[0.755650, -0.002500, -0.000300]),
        "tag_415": TagConfig(id=415, pos=[0.971550, -0.002500, -0.000300]),
        "tag_416": TagConfig(id=416, pos=[1.187450, -0.002500, -0.000300]),
        "tag_417": TagConfig(id=417, pos=[1.403350, -0.002500, -0.000300]),
        "tag_418": TagConfig(id=418, pos=[1.619250, -0.002500, -0.000300]),
        "tag_419": TagConfig(id=419, pos=[1.835150, -0.002500, -0.000300]),
        "tag_420": TagConfig(id=420, pos=[2.051050, -0.002500, -0.000300]),
    }
)

if __name__ == "__main__":
    verify_bodies(cfg)
