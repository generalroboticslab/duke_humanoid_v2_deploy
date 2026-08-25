"""Chamfered cube body type. Instances auto-discovered from sibling .py files."""
import importlib
import pkgutil
import numpy as np
import trimesh
import sys
import os
from typing import Dict
from dataclasses import dataclass

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TaggedRigidBodyConfig, TaggedRigidBody, TagConfig, MeshEntry, _CACHE

def _chamfered_cube_mesh(size: float, chamfer: float) -> trimesh.Trimesh:
    """Generate a chamfered cube mesh, cached to disk."""
    path = _CACHE / f"cube_s{round(size*1e6)}um_ch{round(chamfer*1e6)}um.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        d = np.load(path)
        flat_verts, flat_tris = d["verts"], d["tris"]
    else:
        from itertools import product
        from scipy.spatial import ConvexHull
        a, b  = size / 2, size / 2 - chamfer
        signs = np.array(list(product((-1, 1), repeat=3)), dtype=np.float32)
        verts = np.vstack([signs * [a, b, b], signs * [b, a, b], signs * [b, b, a]])
        hull = ConvexHull(verts)
        tris = hull.simplices.copy()
        normals = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                         verts[tris[:, 2]] - verts[tris[:, 0]])
        # Fix winding
        flip = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
        tris[flip] = tris[flip][:, [0, 2, 1]]
        flat_verts = verts[tris].reshape(-1, 3)
        flat_tris = np.arange(len(flat_verts), dtype=np.uint32).reshape(-1, 3)
        np.savez(path, verts=flat_verts, tris=flat_tris)

    mesh = trimesh.Trimesh(vertices=flat_verts, faces=flat_tris, process=False)
    mesh.visual.vertex_colors = [80, 130, 230, 255]  # Blue-ish, fully opaque for correct depth occlusion
    mesh.visual = mesh.visual.to_texture()
    mesh.visual.material = mesh.visual.material.to_pbr()
    return mesh

@dataclass(eq=False)
class ChamferedCubeConfig(TaggedRigidBodyConfig):
    """Config for a chamfered-cube rigid body."""
    name: str = "cube"
    cube_size: float = 0.08
    chamfer: float = 0.019

    def get_mesh_entries(self) -> Dict[str, MeshEntry]:
        """Generate chamfered cube mesh procedurally (no file required)."""
        # Pull each face in by 0.1 mm to prevent Z-fighting with the tag image
        # quads that sit exactly on the cube surface. Tag config positions are
        # unchanged (physically correct); only the visual mesh is shrunken.
        render_size = self.cube_size - 0.0002  # 0.1 mm per face
        mesh = _chamfered_cube_mesh(render_size, self.chamfer)
        return {"body": MeshEntry(
            key=f"chamfered_cube_{round(render_size*1e6)}um_{round(self.chamfer*1e6)}um",
            mesh=mesh, pos=np.zeros(3), rot=np.eye(3),
        )}

    def resolve_tag_rot(self, tag: TagConfig) -> np.ndarray:
        """Derive rotation from face normal (= normalized tag position) when rot is None."""
        if tag.rot is not None:
            return tag.rot
        n = tag.pos / np.linalg.norm(tag.pos)
        return self._face_rot(n)

    @staticmethod
    def _face_rot(n: np.ndarray) -> np.ndarray:
        """Rotation: X=right, Y=down, Z=into-face (AprilTag convention: z away from camera)."""
        up = np.array([0., 1., 0.]) if abs(n[2]) > 0.5 else np.array([0., 0., 1.])
        return np.column_stack([np.cross(up, n), -up, -n])


ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, 'cfg'):
        ALL_CONFIGS.append(mod.cfg)
