"""Pipette holder 4-tag body type. Instances auto-discovered from sibling .py files."""
import importlib
import pkgutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TaggedRigidBodyConfig, MeshConfig

@dataclass(eq=False)
class PipetteHolder4TagConfig(TaggedRigidBodyConfig):
    """Config for a 4-tag pipette holder rigid body."""
    name: str = "pipette_holder_4tag"
    meshes: Dict[str, MeshConfig] = field(default_factory=lambda: {
        "body1227": MeshConfig(str(Path(__file__).parent / "pipette_holder_4_tag.obj"), scale=1e-3, alpha=180)
    })
    tag_size: float = 0.03    # default 3 cm tag width

ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, 'cfg'):
        ALL_CONFIGS.append(mod.cfg)
