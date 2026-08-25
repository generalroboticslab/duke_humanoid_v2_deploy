"""Tripod platform (rack) body type: 3 AprilTags on a fan of vertical faces.

Two physical racks share this body type. Their tag fans are geometrically
IDENTICAL — three vertical 44x44 mm pads fanned around +X at 0 / +-36.87 deg
(exact 3-4-5 normals), each carrying a centred 30 mm tag — and differ only by a
rigid translation: the v8 fan sits 40 mm further -X and 30 mm higher than the v6
fan. Everything else (mast, base) differs. Pick one via ``LAYOUTS`` / the
``layout`` argument; each layout brings its own visualization mesh.

Geometry source of truth (Fusion STEP exports, mm):
    v6 -> ``asset/tripod_base/tripod_platform_v6.step``   short one-piece rack
    v8 -> ``asset/tripod_base/tripod_platform_v8.step``   tall stand-mounted rack
(v6 supersedes an earlier v7 export this package was first built from: v7
modelled a small separate 40x40 mm base puck, v6 the real one-piece base that
extends to x=+40. Their tag faces are bit-identical, so v6 changed no tag pose.)

The body frame shares the Fusion origin but is ROTATED 180 deg about Z (X and Y
negated, Z unchanged — user-requested lab convention). LAYOUTS stores the raw
STEP-frame data; ``R_BODY_STEP`` maps it (and the visualization mesh, tessellated
from the same STEP) into the body frame, so everything stays consistent by
construction.

Tags are dual-material printed (flush inlay, baked in Fusion — NOT the
TaggedCube pipeline). Decode convention, CONFIRMED by physical scan of both
printed racks (detector axes: red/X = viewer's right, green/Y = down,
blue/Z = into the face): rot columns = [r, -u, -n] — the textbook AprilTag
orientation. Note this is 180 deg in-plane from grasp_cube's [-r, u, -n]; the
two fabrication pipelines bake the bitmap differently, so each body type carries
its own verified convention.

Instances are auto-discovered from sibling ``.py`` files exposing ``cfg``.
"""
import importlib
import os
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tagged_rigid_body import TaggedRigidBodyConfig, TagConfig, MeshConfig

MARKER_SIZE_M = 0.030
BORDER_SIZE_M = 0.005

# Body frame = STEP/Fusion frame rotated 180 deg about Z. This matrix maps
# STEP-frame coordinates/vectors into the body frame (it is its own inverse).
R_BODY_STEP = np.diag([-1.0, -1.0, 1.0])

# Exact pad data extracted from each STEP (mm, RAW STEP frame — converted to the
# body frame via R_BODY_STEP in build_tripod_tags):
# tag centre = pad centre; n = outward normal; up = tag image-top direction.
LAYOUTS: Dict[str, Dict[str, dict]] = {
    "v6": {
        "center": {"pos_mm": (-62.0, 0.0, 120.0),    "n": (1.0, 0.0, 0.0),  "up": (0.0, 0.0, 1.0)},
        "left":   {"pos_mm": (-75.2, 39.6, 120.0),   "n": (0.8, 0.6, 0.0),  "up": (0.0, 0.0, 1.0)},
        "right":  {"pos_mm": (-75.2, -39.6, 120.0),  "n": (0.8, -0.6, 0.0), "up": (0.0, 0.0, 1.0)},
    },
    # v6 fan translated by (-40, 0, +30) mm — verified pad-for-pad against the STEP.
    # NOTE the v8 "left" pad: that CAD face blends 0.529 mm into the neighbouring
    # surface, so its face CENTROID reads (-115.518, 40.023, 150). The 44x44 pad
    # itself is centred at (-115.2, 39.6, 150) — the exact mirror of "right" — and
    # that (bbox) centre is what belongs here.
    "v8": {
        "center": {"pos_mm": (-102.0, 0.0, 150.0),   "n": (1.0, 0.0, 0.0),  "up": (0.0, 0.0, 1.0)},
        "left":   {"pos_mm": (-115.2, 39.6, 150.0),  "n": (0.8, 0.6, 0.0),  "up": (0.0, 0.0, 1.0)},
        "right":  {"pos_mm": (-115.2, -39.6, 150.0), "n": (0.8, -0.6, 0.0), "up": (0.0, 0.0, 1.0)},
    },
}

# Visualization mesh per layout (tessellated from that layout's STEP, raw STEP frame).
MESHES: Dict[str, str] = {
    "v6": "tripod_platform.obj",
    "v8": "tripod_platform_v8.obj",
}


def _face_axes(n, up):
    """Orthonormal (right, up, normal) for a face; right x up == normal."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    r = np.cross(up, n); r = r / np.linalg.norm(r)
    u = np.cross(n, r)
    return r, u, n


def build_tripod_tags(face_ids: Dict[str, int], layout: str = "v6") -> Dict[str, TagConfig]:
    """Per-face TagConfig dict. pos in metres; rot = [r, -u, -n] (scan-confirmed)."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout '{layout}'; choose from {list(LAYOUTS)}")
    faces = LAYOUTS[layout]
    tags: Dict[str, TagConfig] = {}
    for face, tid in face_ids.items():
        if face not in faces:
            raise ValueError(f"unknown face '{face}'; choose from {list(faces)}")
        spec = faces[face]
        r, u, n = _face_axes(spec["n"], spec["up"])
        rot = R_BODY_STEP @ np.column_stack([r, -u, -n])
        assert abs(np.linalg.det(rot) - 1.0) < 1e-9, f"{face} rot is not a proper rotation"
        pos = (R_BODY_STEP @ (np.asarray(spec["pos_mm"], float) * 1e-3)).tolist()
        tags[face] = TagConfig(id=int(tid), pos=pos, rot=rot)
    return tags


@dataclass(eq=False)
class TripodPlatformConfig(TaggedRigidBodyConfig):
    """Tripod platform rack with 3 fan tags."""
    name: str = "tripod_platform"
    layout: str = "v6"

    def __post_init__(self):
        # OBJ vertices are in the raw STEP frame -> rotate into the body frame.
        default = {"body": MeshConfig(
            str(Path(__file__).parent / MESHES[self.layout]),
            rot=R_BODY_STEP, scale=1e-3, alpha=180)}
        self.meshes = {**default, **self.meshes}
        super().__post_init__()


def make_tripod_platform(name: str, face_ids: Dict[str, int],
                         layout: str = "v6") -> TripodPlatformConfig:
    """Factory: build a TripodPlatformConfig from a face->id mapping and a layout."""
    return TripodPlatformConfig(
        tags=build_tripod_tags(face_ids, layout),
        name=name,
        layout=layout,
        tag_size=MARKER_SIZE_M,
        tag_border_size=BORDER_SIZE_M,
    )


ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, "cfg"):
        ALL_CONFIGS.append(mod.cfg)
