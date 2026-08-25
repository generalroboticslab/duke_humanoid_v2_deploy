"""Holder 6-tag instance 0. Verify: python holder_6tag_pipette_tip.py"""
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import verify_bodies, FrameConfig, MeshConfig
from tagged_bodies.holder_6tag import Holder6TagConfig
from tagged_bodies.holder_6tag.hexgon_pattern import flower_pattern
from pathlib import Path
import numpy as np


tag_configs = Holder6TagConfig.make_tag_configs(start_id=56)

frame_xy = flower_pattern(r=14.625e-3)
frame_xyz = np.zeros((frame_xy.shape[0], 3))
frame_xyz[:, :2] = frame_xy
frame_xyz[:, 2] = 0.07

# 3. Create the final tracking object
cfg = Holder6TagConfig(
    name="holder_6tag_pipette_tip",
    tags=tag_configs,
    frames={
        f"{k}": FrameConfig(pos=pos)
        for k, pos in enumerate(frame_xyz)
    },
    meshes={
        "top": MeshConfig(str(Path(__file__).parent / "pipette_tips_holder_100-1000ul.obj"), pos=[0, 0, -0.01], scale=1e-3, alpha=180),
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
