"""Standalone 3D-printed AprilTag cubes (6 faces), for a gripper to grasp.

Distinct from ``tag_cube/`` (the 40 mm, 5-face parallel-gripper *wrist* cubes):
these are SOLID cubes of arbitrary size with a tag on all 6 faces, produced by
the TaggedCube pipeline (flush dual-material inlay). Instances are auto-discovered
from sibling ``.py`` files that expose a module-level ``cfg``.

Frame convention (mirrors TaggedCube config.py FACE_FRAMES): body axes X=width,
Y=length, Z=height; cube centred at the body origin. Each face's tag pose is
converted into the tagged_rigid_body ``rot`` convention (tag X=face-left, Y=up,
Z=into-face) via columns ``[-r, u, -n]`` (see build_cube_tags) — matching how the
printed tags actually decode.
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
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_rigid_body import TaggedRigidBodyConfig, TagConfig, MeshEntry, _CACHE

# Marker geometry shared by all instances (metres); matches the TaggedCube pipeline
# (tag36h11, 30 mm marker + 5 mm quiet-zone border).
MARKER_SIZE_M = 0.030
BORDER_SIZE_M = 0.005

# Per face: (outward normal n, image-up u) in the cube body frame.
FACE_FRAMES = {
    "right":  (np.array([1.0, 0.0, 0.0]),  np.array([0.0, 0.0, 1.0])),
    "left":   (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "back":   (np.array([0.0, 1.0, 0.0]),  np.array([0.0, 0.0, 1.0])),
    "front":  (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "top":    (np.array([0.0, 0.0, 1.0]),  np.array([0.0, -1.0, 0.0])),
    "bottom": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
}


def _face_axes(n, up):
    """Orthonormal (right, up, normal) for a face; right x up == normal."""
    n = n / np.linalg.norm(n)
    r = np.cross(up, n)
    r = r / np.linalg.norm(r)
    u = np.cross(n, r)
    return r, u, n


def build_cube_tags(size_m: float, face_ids: Dict[str, int]) -> Dict[str, TagConfig]:
    """Build the per-face TagConfig dict for a solid cube of side ``size_m``.

    pos  = (size/2) * outward_normal  (flush inlay → the tag plane is the face surface).
    rot  columns = [tag_right, tag_down, tag_into_face] = [-r, u, -n]
         (empirically confirmed against the printed tags + pupil_apriltags decode,
          and by physical inspection: the printed tag reads 180° in-plane from the
          textbook [r,-u,-n] — tag X = -r (the face's left), tag Y = +u (up),
          tag Z = -n (into the face)).
    """
    half = size_m / 2.0
    tags: Dict[str, TagConfig] = {}
    for face, tid in face_ids.items():
        if face not in FACE_FRAMES:
            raise ValueError(f"unknown face '{face}'; choose from {list(FACE_FRAMES)}")
        n0, up0 = FACE_FRAMES[face]
        r, u, n = _face_axes(n0, up0)
        rot = np.column_stack([-r, u, -n])
        assert abs(np.linalg.det(rot) - 1.0) < 1e-9, f"{face} rot is not a proper rotation"
        tags[face] = TagConfig(id=int(tid), pos=(n * half).tolist(), rot=rot)
    return tags


def build_cube_tags_explicit(size_m: float, face_specs: Dict[str, dict]) -> Dict[str, TagConfig]:
    """Build per-face TagConfigs from EXPLICIT per-face tag orientations.

    Use this (instead of ``build_cube_tags``) for a printed cube whose tags do
    NOT follow the default FACE_FRAMES ``[-r, u, -n]`` convention. Each face spec
    is ``{"n": outward_normal, "id": tag_id, "tag_x": tag_+X_axis}`` with ``n``
    and ``tag_x`` in the CUBE BODY frame. The tag +Z points INTO the face
    (``-n``) and the tag +Y is derived as ``z x x`` — so the caller only pins the
    normal and the in-plane +X, exactly as specified per face.

    pos = (size/2) * n  (flush inlay → the tag plane is the face surface).
    """
    half = size_m / 2.0
    tags: Dict[str, TagConfig] = {}
    for face, spec in face_specs.items():
        if face not in FACE_FRAMES:
            raise ValueError(f"unknown face '{face}'; choose from {list(FACE_FRAMES)}")
        n = np.asarray(spec["n"], float); n = n / np.linalg.norm(n)
        tx = np.asarray(spec["tag_x"], float); tx = tx / np.linalg.norm(tx)
        if abs(float(tx @ n)) > 1e-9:
            raise ValueError(f"{face}: tag_x {spec['tag_x']} is not in the face plane (must be perp to n {spec['n']})")
        tz = -n                          # tag +Z into the face
        ty = np.cross(tz, tx)            # tag +Y = z x x
        rot = np.column_stack([tx, ty, tz])
        assert abs(np.linalg.det(rot) - 1.0) < 1e-9, f"{face} rot is not a proper rotation"
        tags[face] = TagConfig(id=int(spec["id"]), pos=(n * half).tolist(), rot=rot)
    return tags


def _cube_mesh(size_m: float):
    """A box mesh of side ``size_m`` (the visualization envelope), cached to disk."""
    key = f"grasp_cube_{round(size_m * 1e6)}um"
    cache_path = _CACHE / (key + ".npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    inset = 0.0001  # 0.1 mm per face, avoids z-fighting with tag image quads
    if cache_path.exists():
        d = np.load(cache_path)
        fv, ft = d["verts"], d["tris"]
    else:
        box = trimesh.creation.box([size_m - 2 * inset] * 3)
        fv = box.vertices[box.faces].reshape(-1, 3).astype(np.float32)
        ft = np.arange(len(fv), dtype=np.uint32).reshape(-1, 3)
        np.savez(cache_path, verts=fv, tris=ft)
    mesh = trimesh.Trimesh(vertices=fv, faces=ft, process=False)
    mesh.visual.vertex_colors = [200, 200, 205, 255]
    mesh.visual = mesh.visual.to_texture()
    mesh.visual.material = mesh.visual.material.to_pbr()
    return mesh, key


@dataclass(eq=False)
class GraspCubeConfig(TaggedRigidBodyConfig):
    """A solid AprilTag cube of side ``size_m`` with a tag on all 6 faces."""
    name: str = "grasp_cube"
    size_m: float = 0.05

    def get_mesh_entries(self) -> Dict[str, MeshEntry]:
        mesh, key = _cube_mesh(self.size_m)
        return {"body": MeshEntry(key=key, mesh=mesh, pos=np.zeros(3), rot=np.eye(3))}


def make_grasp_cube(name: str, size_m: float, face_ids: Dict[str, int]) -> GraspCubeConfig:
    """Factory: build a GraspCubeConfig from a size and a face->id mapping
    (default FACE_FRAMES [-r, u, -n] orientation)."""
    return GraspCubeConfig(
        tags=build_cube_tags(size_m, face_ids),
        name=name,
        size_m=size_m,
        tag_size=MARKER_SIZE_M,
        tag_border_size=BORDER_SIZE_M,
    )


def make_grasp_cube_explicit(name: str, size_m: float, face_specs: Dict[str, dict],
                             tag_size: float = MARKER_SIZE_M,
                             tag_border_size: float = BORDER_SIZE_M) -> GraspCubeConfig:
    """Factory: build a GraspCubeConfig with EXPLICIT per-face tag orientations
    (see ``build_cube_tags_explicit``). For cubes whose tags do not follow the
    default convention.

    tag_size:        BLACK-square edge length in metres — black-to-black, the
                     detector's tag_size (the only quantity that sets the metric
                     translation scale; solvePnP object points span +-tag_size/2,
                     and the pupil_apriltags pose uses the same edge). Defaults to
                     MARKER_SIZE_M (0.030) so existing callers are unchanged.
    tag_border_size: white quiet-zone width on EACH side (per-side, NOT the sum of
                     both sides), measured outward from the black edge to the
                     sticker/face edge -> full face = tag_size + 2*tag_border_size.
                     Visualization only; does not affect pose. Defaults to
                     BORDER_SIZE_M (0.005).
    """
    return GraspCubeConfig(
        tags=build_cube_tags_explicit(size_m, face_specs),
        name=name,
        size_m=size_m,
        tag_size=tag_size,
        tag_border_size=tag_border_size,
    )


ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, "cfg"):
        ALL_CONFIGS.append(mod.cfg)
