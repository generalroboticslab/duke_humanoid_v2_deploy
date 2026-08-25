"""Holder 6-tag body type. Instances auto-discovered from sibling .py files."""
import importlib
import pkgutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TaggedRigidBodyConfig, TagConfig, MeshConfig



# The tilted rotation matrix shared by all tags on the holder
TILT_ROTATION = [
    [ 1.0,  0.0,  0.0],
    [ 0.0, -0.8,  0.6],
    [ 0.0, -0.6, -0.8]
]

# 1. Define the intuitive physical layout of your tags [X, Y, Z]
TAG_POSITION = {
    "left_top":   (-0.08,  0.05,  0.025),
    "left_low":   (-0.08, -0.01,  0.000),
    "mid_left":   (-0.03, -0.07, -0.010),
    "mid_right":  ( 0.03, -0.07, -0.010),
    "right_low":  ( 0.08, -0.01,  0.000),
    "right_top":  ( 0.08,  0.05,  0.025),
}

TAG_POSITION_TALL = {
    "left_top":   (-0.08,  0.05,  0.025),
    "left_low":   (-0.08, -0.01,  0.000),
    "mid_left":   (-0.03, -0.07, -0.010),
    "mid_right":  ( 0.03, -0.07, -0.010),
    "right_low":  ( 0.08, -0.01,  0.000),
    "right_top":  ( 0.08,  0.05,  0.025),
}


@dataclass(eq=False)
class Holder6TagConfig(TaggedRigidBodyConfig):
    """Config for a 6-tag holder rigid body."""
    name: str = "holder_6tag"
    meshes: Dict[str, MeshConfig] = field(default_factory=dict)
    tag_size: float = 0.03    # default 3 cm tag width

    def __post_init__(self):
        # Merge: start from default body mesh, then overlay any caller-supplied meshes
        default = {"body": MeshConfig(str(Path(__file__).parent / "holder_6tag.obj"), scale=1e-3, alpha=180)}
        self.meshes = {**default, **self.meshes}
        super().__post_init__()

    @staticmethod
    def make_tag_configs(start_id: int) -> dict:
        """Return a dict mapping position name -> TagConfig with sequential IDs starting at start_id."""
        return {
            name: TagConfig(id=start_id + i, pos=pos, rot=TILT_ROTATION)
            for i, (name, pos) in enumerate(TAG_POSITION.items())
        }


# Auto-discover instances from sibling .py files
ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, 'cfg'):
        ALL_CONFIGS.append(mod.cfg)
