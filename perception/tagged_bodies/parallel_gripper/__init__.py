"""Duke v2 parallel-jaw gripper JAWS: four 2-tag rigid bodies (2 hands x 2 jaws).

The gripper is NOT one rigid body: each hand has a base (no tags) and two jaws
(``left_rack`` / ``right_rack``) that slide along +-Y, 1:1 coupled. The tag
plates ride the JAWS (one on the +Z/top face, one on the -Z/bottom face of each
jaw), so each JAW is registered as its own TaggedRigidBody with 2 back-to-back
tags — a camera sees at most one of them at a time. Hand centre ~= SE(3)
midpoint of its two jaw poses (the coupling is symmetric); gripper opening
~= the change in their Y separation.

Geometry source of truth (first-hand, mirrored below — keep in sync):
    legged_env_dev/asset/duke_v2/parallel_gripper/tag_layout.py  (SLOTS, HAND_TAGS)
Cross-checked against the CAD STEP (mini_gripper_old/ParallelGripper0710/
ParallelGripper0710.step, the full re-export that added the base tag holders/pads
and the usb_c_protector): every 20x20 mm plate/pad outer face sits at its SLOT
centre to ±0.005 mm (jaw plates z=+10/-30, base pads z=+15.5/-35.5, normals +-Z).

Body frame: the gripper BASE frame at the authored CAD pose (jaw slide
q = ref = 0.05, half-open). Both jaw frames coincide with the base frame at
that pose; each estimated pose is that frame riding its jaw. The rack meshes
are exported in the same frame (mm; scale 1e-3).

Decode convention (pupil_apriltags frame expressed in the jaw frame), derived
from make_tags.write_quad UV math + TAG_ROT=180 + the cv2.aruco ->
pupil_apriltags 180 deg in-plane decode offset (the same offset measured
empirically on grasp_cube):
    +Z (top) plate:    tagX=+X  tagY=-Y  tagZ=-Z   -> ROT_TOP = diag(1,-1,-1)
    -Z (bottom) plate: tagX=+X  tagY=+Y  tagZ=+Z   -> ROT_BOTTOM = identity
i.e. "detected tag +X = +gripper X (robot forward)" per tag_layout.py. The
live axis scan is the final arbiter — adjust ROT_* here if the camera disagrees.

Tag size: the PHYSICAL pads carry the 10x10 tag36h11 layout (8x8 pattern +
1-cell white border) filling the 20x20 mm plate -> black-to-black = 16 mm
(tag_size 0.016), 2 mm quiet zone inside the pad. (The sim decals in
legged_env_dev make_tags.py stretch the bare 8x8 to the full 20 mm — sim/real
differ; the registry follows the REAL pads.) Instances are auto-discovered
from sibling ``.py`` files exposing a module-level ``cfg``.
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

MARKER_SIZE_M = 0.016    # tag36h11 BLACK square edge, physically measured: the 10x10
                         # layout (8x8 pattern + 1-cell white border) fills the 20 mm
                         # pad -> cell 2 mm, black-to-black 16 mm
BORDER_SIZE_M = 0.002    # the 1-cell (2 mm) white quiet zone inside the pad

# Plate OUTER-face centres (mm, gripper base frame at the authored pose).
# Mirror of tag_layout.SLOTS; verified against the STEP (see module docstring).
SLOTS = {
    "left_top":     {"pos_mm": (43.45, -45.04,  10.0), "face": "+Z"},
    "left_bottom":  {"pos_mm": (44.61, -44.99, -30.0), "face": "-Z"},
    "right_top":    {"pos_mm": (44.61,  58.96,  10.0), "face": "+Z"},
    "right_bottom": {"pos_mm": (43.45,  59.01, -30.0), "face": "-Z"},
}

# BASE tag-holder pad OUTER-face centres (mm, same frame) — the CAD rev added a
# tag-holder plate flush on the base TOP face and its mirrored twin on the BOTTOM
# face (mirror plane z=-10, the gripper's z symmetry mid-plane); each holder has
# two 20x20x2 mm pocket-seated pads. Mirror of tag_layout.SLOTS base_* entries.
# CAD ground truth from the ParallelGripper0710.step full re-export (probe_step pad
# solid centres, ±0.001 mm; holders + pads + usb_c_protector are baked into base.obj).
BASE_SLOTS = {
    "top_a":    {"pos_mm": (-4.743, -5.00,  15.5), "face": "+Z"},
    "top_b":    {"pos_mm": (-4.743, 17.00,  15.5), "face": "+Z"},
    "bottom_a": {"pos_mm": (-4.743, -5.00, -35.5), "face": "-Z"},
    "bottom_b": {"pos_mm": (-4.743, 17.00, -35.5), "face": "-Z"},
}

ROT_TOP    = np.diag([1.0, -1.0, -1.0])  # +Z plate: tagX=+X, tagY=-Y, tagZ=-Z
ROT_BOTTOM = np.eye(3)                   # -Z plate: tagX=+X, tagY=+Y, tagZ=+Z


def build_jaw_tags(jaw: str, top_id: int, bottom_id: int) -> Dict[str, TagConfig]:
    """TagConfigs for one jaw ('left'|'right'): its top (+Z) and bottom (-Z) plates."""
    if jaw not in ("left", "right"):
        raise ValueError(f"jaw must be 'left' or 'right', got '{jaw}'")
    top, bottom = SLOTS[f"{jaw}_top"], SLOTS[f"{jaw}_bottom"]
    return {
        "top":    TagConfig(id=int(top_id),
                            pos=(np.asarray(top["pos_mm"], float) * 1e-3).tolist(),
                            rot=ROT_TOP.copy()),
        "bottom": TagConfig(id=int(bottom_id),
                            pos=(np.asarray(bottom["pos_mm"], float) * 1e-3).tolist(),
                            rot=ROT_BOTTOM.copy()),
    }


@dataclass(eq=False)
class ParallelGripperJawConfig(TaggedRigidBodyConfig):
    """One jaw (rack) of the Duke v2 parallel gripper, carrying 2 back-to-back tags."""
    name: str = "gripper_jaw"
    jaw: str = "left"  # which rack mesh to show: 'left' or 'right'

    def __post_init__(self):
        default = {"body": MeshConfig(
            str(Path(__file__).parent / f"{self.jaw}_rack.obj"), scale=1e-3, alpha=180)}
        self.meshes = {**default, **self.meshes}
        super().__post_init__()


def make_jaw(name: str, jaw: str, top_id: int, bottom_id: int) -> ParallelGripperJawConfig:
    """Factory: one jaw body from its side and its two tag ids (top, bottom)."""
    return ParallelGripperJawConfig(
        tags=build_jaw_tags(jaw, top_id, bottom_id),
        name=name,
        jaw=jaw,
        tag_size=MARKER_SIZE_M,
        tag_border_size=BORDER_SIZE_M,
    )


@dataclass(eq=False)
class ParallelGripperBaseConfig(TaggedRigidBodyConfig):
    """The gripper BASE (+ its two rigidly attached tag-holder plates) as one
    rigid body carrying 4 pad tags (2 up, 2 down). Unlike the jaws, the base
    does not slide: its pose IS the hand pose, so this body gives the camera a
    direct 'hand centre' measurement (the jaws still track separately for the
    opening width)."""
    name: str = "gripper_base"

    def __post_init__(self):
        # base.obj (from ParallelGripper0710.step) already contains the two tag-holder
        # plates, their 4 pads and the usb-c protector — one mesh covers the whole body.
        default = {"body": MeshConfig(
            str(Path(__file__).parent / "base.obj"), scale=1e-3, alpha=180)}
        self.meshes = {**default, **self.meshes}
        super().__post_init__()


def make_base(name: str, top_a: int, top_b: int, bottom_a: int, bottom_b: int) -> ParallelGripperBaseConfig:
    """Factory: the base+holders body from its four pad tag ids.

    Same decode convention as the jaws (make_tags.write_quad + TAG_ROT=180):
    +Z pads use ROT_TOP, -Z pads ROT_BOTTOM -> detected tag +X = +gripper X.
    """
    ids = {"top_a": top_a, "top_b": top_b, "bottom_a": bottom_a, "bottom_b": bottom_b}
    tags = {
        slot: TagConfig(id=int(tid),
                        pos=(np.asarray(BASE_SLOTS[slot]["pos_mm"], float) * 1e-3).tolist(),
                        rot=(ROT_TOP if BASE_SLOTS[slot]["face"] == "+Z" else ROT_BOTTOM).copy())
        for slot, tid in ids.items()
    }
    return ParallelGripperBaseConfig(
        tags=tags,
        name=name,
        tag_size=MARKER_SIZE_M,
        tag_border_size=BORDER_SIZE_M,
    )


ALL_CONFIGS = []
for mod_info in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f".{mod_info.name}", __name__)
    if hasattr(mod, "cfg"):
        ALL_CONFIGS.append(mod.cfg)
