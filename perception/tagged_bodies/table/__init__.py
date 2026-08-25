"""Rectangular work table carrying a row of tags on its top surface.

The two lab tables (``lab_table_a`` / ``lab_table_b``) are HAND-MAINTAINED
instance files; this package only defines the shared ``TableConfig`` and the one
tag roll they use. (They were originally emitted by
``asset/tag_creation/print_table_tags.py``, which is now stale — its frame and
layout assumptions no longer match this file; see the banner in that script.)

BODY FRAME (origin = centre of the table's TOP SURFACE), aligned with the
grasp_cube convention:

    +x  along the LENGTH (the 1200 mm long edge)
    +y  along the WIDTH  (the 600 mm short edge)
    +z  UP, out of the top surface

Right-handed. The slab hangs BELOW the origin: modelled as a mesh centred half a
thickness below the top, i.e. at z = -thickness/2, spanning z in [-thickness, 0].
Putting the origin on the top surface makes the one number the manipulation stack
cares about — the working-surface height — the body's own z, no offset to remember.

A tag lies flat on the top surface FACING UP, so its own z-axis (into the marked
surface) points DOWN = -z. The single roll both tables use,
``TAG_FLAT_ON_TOP_X_ALONG_Y``, puts the tag's +x along body +y (the width) and
its z = -z: columns [tag_x=+y, tag_y=+x, tag_z=-z], det +1.

TAG ROLL is still the one thing that cannot be guessed from a sentence — a flat
tag can be stuck at any of four 90-degree rolls, and a 90-degree error swaps
length for width in the collision model (safety-relevant on a 1200 x 600 slab).
Both physical tables were stuck to match this constant and verified on the D435.
"""
import importlib
import os
import pkgutil
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import trimesh

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import MeshEntry, TaggedRigidBodyConfig

# The single tag roll both tables use, in the table body frame (columns = tag
# x, y, z). The tag lies flat on the top surface FACING UP, so its z-axis (into
# the marked surface) points -z; its +x runs along body +y (the 600 mm width).
# Columns: tag_x=+y, tag_y=+x, tag_z=-z. det +1.
TAG_FLAT_ON_TOP_X_ALONG_Y = [[0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0]]

TAG_ROLLS = {"flat_on_top_x_along_y": TAG_FLAT_ON_TOP_X_ALONG_Y}


@dataclass(eq=False)
class TableConfig(TaggedRigidBodyConfig):
    """A rectangular table slab; tags lie flat on the top surface."""
    name: str = "table"
    length_m: float = 1.200        # along body x (the 1200 mm long edge)
    width_m: float = 0.600         # along body y (the 600 mm short edge)
    thickness_m: float = 0.030     # slab thickness (hangs below the origin, body -z)
    tag_roll: str = "flat_on_top_x_along_y"   # key into TAG_ROLLS; recorded for provenance

    def get_mesh_entries(self) -> Dict[str, MeshEntry]:
        """Slab centred half a thickness below the top surface (origin); +z is up."""
        mesh = trimesh.creation.box([self.length_m, self.width_m, self.thickness_m])
        mesh.visual.vertex_colors = [205, 200, 190, 255]
        return {"body": MeshEntry(
            key=f"table_mesh_{self.name}",
            mesh=mesh,
            pos=np.array([0.0, 0.0, -self.thickness_m / 2.0]),
            rot=np.eye(3),
        )}

    def slab_cuboid(self) -> dict:
        """The collision box a motion planner should be handed, in body frame:
        ``{"dims": [x, y, z], "pos": [x, y, z]}`` in metres. Kept here rather
        than in the planner so the obstacle and the visual mesh can never
        disagree — both read these same fields."""
        return {"dims": [self.length_m, self.width_m, self.thickness_m],
                "pos": [0.0, 0.0, -self.thickness_m / 2.0]}


# Auto-discover instances from sibling .py files.
ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, 'cfg'):
        ALL_CONFIGS.append(mod.cfg)
