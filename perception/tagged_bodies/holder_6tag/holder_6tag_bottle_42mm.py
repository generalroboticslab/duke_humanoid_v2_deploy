"""Holder 6-tag instance 1. Verify: python holder_6tag_bottle_42mm.py"""
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import verify_bodies, FrameConfig, MeshConfig
from tagged_bodies.holder_6tag import Holder6TagConfig
from pathlib import Path


tag_configs = Holder6TagConfig.make_tag_configs(start_id=0)

# 3. Create the final tracking object
cfg = Holder6TagConfig(
    name="holder_6tag_bottle_42mm",
    tags=tag_configs,
    frames={
        f"0_prepare": FrameConfig(pos=[0, 0, 0.16], rot=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        f"1_grasp": FrameConfig(pos=[0, 0, 0.12], rot=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        **{f"3_test_{k}{j}{i}": FrameConfig(
                pos=[x, y, z],
                rot=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            )
            for k, z in enumerate([0.1, 0.05, 0.00])
            for j, y in enumerate([-0.2, -0.25, -0.3])
            for i, x in enumerate([0, 0.05, -0.05])},
    },
    meshes={
        "top": MeshConfig(str(Path(__file__).parent / "holder_6tag_bottle_42mm_top.obj"), pos=[0, 0, 0], scale=1e-3, alpha=180),
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
