"""Pipette holder 4-tag instance 0. Verify: python holder_0.py"""
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TagConfig, verify_bodies, FrameConfig
from tagged_bodies.pipette_holder_4tag import PipetteHolder4TagConfig

# Which way each tag faces (rotation matrices).
# You shouldn't need to change these unless the holder geometry changes.
LEFT_ROT = [
    [ 0.8,  0.0,  0.6],
    [-0.6,  0.0,  0.8],
    [ 0.0, -1.0,  0.0]
]  # faces left ~37°

RIGHT_ROT = [
    [ 0.8,  0.0, -0.6],
    [ 0.6,  0.0,  0.8],
    [ 0.0, -1.0,  0.0]
]  # faces right ~37°

# Physical tag positions [X, Y, Z] in meters from the body center.
cfg = PipetteHolder4TagConfig(
    name="pipette_holder_4tag_0",
    tags={
        "left_top":  TagConfig(id=22, pos=(-0.034, -0.006,  0.075), rot=LEFT_ROT),
        "left_low":  TagConfig(id=23, pos=(-0.034, -0.006,  0.025), rot=LEFT_ROT),
        "right_low": TagConfig(id=24, pos=( 0.034, -0.006,  0.025), rot=RIGHT_ROT),
        "right_top": TagConfig(id=25, pos=( 0.034, -0.006,  0.075), rot=RIGHT_ROT),
    },
    frames={
        "tip": FrameConfig(pos=(0.0, 0.0, -0.089))
    }
    
)

if __name__ == "__main__":
    verify_bodies(cfg)
