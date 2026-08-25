"""80mm cube, tags 12-16. Verify: python cube_0.py (from its directory)"""
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TagConfig, verify_bodies
from tagged_bodies.chamfered_cube import ChamferedCubeConfig


H = 0.04  # 80mm / 2

cfg = ChamferedCubeConfig(name="cube_0", cube_size=0.08, tags={
    "top":   TagConfig(id=12, pos=[0, 0, H]),
    "front": TagConfig(id=13, pos=[0, H, 0]),
    "back":  TagConfig(id=14, pos=[0,-H, 0]),
    "right": TagConfig(id=15, pos=[H, 0, 0]),
    "left":  TagConfig(id=16, pos=[-H, 0, 0]),
})

if __name__ == "__main__":
    verify_bodies(cfg)
