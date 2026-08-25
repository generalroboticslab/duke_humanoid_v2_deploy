"""AprilTag 3D-printable cube body type. Instances auto-discovered from sibling .py files.

Physical geometry matches the 3MF generator (not included in this repository):
  40mm core, 2mm face slabs on top + 4 sides (no bottom slab), 5 tagged faces.
  Body origin: cube center = CORE_SIZE_MM/2 above physical base.
"""
import importlib
import pkgutil
import sys
import os
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Dict

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TaggedRigidBodyConfig, MeshEntry, _CACHE

# --- Physical constants (mm) — mirror the 3MF generator (not included in this repository) ---
CORE_SIZE_MM      = 40.0  # inner cube side
FACE_THICKNESS_MM =  2.0  # each face slab thickness
MARKER_SIZE_MM    = 30.0  # ArUco pattern area (full 8×8 grid)
BORDER_SIZE_MM    =  5.0  # quiet zone around marker

# --- Derived geometry (metres) ---
# Distance from body-center to the outer face of any tagged face
OUTER_FACE_DIST_M  = (CORE_SIZE_MM / 2 + FACE_THICKNESS_MM) / 1000

TAG_SIZE_M         = MARKER_SIZE_MM    / 1000
TAG_BORDER_SIZE_M  = BORDER_SIZE_MM    / 1000

# Full outer dimensions for the visualization mesh box
OUTER_WIDTH_M      = (CORE_SIZE_MM + 2 * FACE_THICKNESS_MM) / 1000  # XY
OUTER_HEIGHT_M     = (CORE_SIZE_MM + FACE_THICKNESS_MM)     / 1000  # Z (no bottom face)

# Body origin is CORE_SIZE_MM/2 above the physical base.
# Mesh box spans Z: −CORE_SIZE_MM/2 .. +(CORE_SIZE_MM/2 + FACE_THICKNESS_MM).
# → centroid sits FACE_THICKNESS_MM/2 above body origin.
MESH_Z_OFFSET_M    = FACE_THICKNESS_MM / 2 / 1000

# Pull each face in by this amount to prevent Z-fighting with tag image quads.
# Tag positions are unchanged (physically correct); only the visual mesh shrinks.
_RENDER_FACE_INSET_M = 0.0001  # 0.1 mm per face


def _tag_cube_mesh() -> trimesh.Trimesh:
    """Generate a box mesh representing the outer envelope of the tag cube, cached to disk."""
    render_width  = OUTER_WIDTH_M  - 2 * _RENDER_FACE_INSET_M
    render_height = OUTER_HEIGHT_M - 2 * _RENDER_FACE_INSET_M
    cache_path = _CACHE / (
        f"tag_cube_core{round(CORE_SIZE_MM*1e3)}um"
        f"_face{round(FACE_THICKNESS_MM*1e3)}um"
        f"_inset{round(_RENDER_FACE_INSET_M*1e6)}nm.npz"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        d = np.load(cache_path)
        flat_verts, flat_tris = d["verts"], d["tris"]
    else:
        box = trimesh.creation.box([render_width, render_width, render_height])
        # Flatten to unindexed triangles so npz round-trips cleanly
        flat_verts = box.vertices[box.faces].reshape(-1, 3).astype(np.float32)
        flat_tris  = np.arange(len(flat_verts), dtype=np.uint32).reshape(-1, 3)
        np.savez(cache_path, verts=flat_verts, tris=flat_tris)

    mesh = trimesh.Trimesh(vertices=flat_verts, faces=flat_tris, process=False)
    mesh.visual.vertex_colors = [80, 130, 230, 255]
    mesh.visual = mesh.visual.to_texture()
    mesh.visual.material = mesh.visual.material.to_pbr()
    return mesh


@dataclass(eq=False)
class TagCubeConfig(TaggedRigidBodyConfig):
    """Config for the 3D-printed AprilTag cube (tag36h11, 40mm core, 2mm face slabs)."""
    name: str = "tag_cube"

    def get_mesh_entries(self) -> Dict[str, MeshEntry]:
        mesh = _tag_cube_mesh()
        return {"body": MeshEntry(
            key=(f"tag_cube_core{round(CORE_SIZE_MM*1e3)}um"
                 f"_face{round(FACE_THICKNESS_MM*1e3)}um"
                 f"_inset{round(_RENDER_FACE_INSET_M*1e6)}nm"),
            mesh=mesh,
            pos=np.array([0.0, 0.0, MESH_Z_OFFSET_M]),
            rot=np.eye(3),
        )}


ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, 'cfg'):
        ALL_CONFIGS.append(mod.cfg)
