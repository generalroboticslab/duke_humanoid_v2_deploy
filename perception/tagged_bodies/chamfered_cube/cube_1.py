"""80mm cube, tags 17-21. Verify: python cube_1.py (from its directory)"""
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TagConfig, verify_bodies
from tagged_bodies.chamfered_cube import ChamferedCubeConfig


H = 0.04  # 80mm / 2

cfg = ChamferedCubeConfig(name="cube_1", cube_size=0.08, tags={
    "top":   TagConfig(id=17, pos=[0, 0, H]),
    "front": TagConfig(id=18, pos=[0, H, 0]),
    "back":  TagConfig(id=19, pos=[0,-H, 0]),
    "right": TagConfig(id=20, pos=[H, 0, 0]),
    "left":  TagConfig(id=21, pos=[-H, 0, 0]),
})

if __name__ == "__main__":
    verify_bodies(cfg)
