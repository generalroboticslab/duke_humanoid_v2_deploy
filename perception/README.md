# perception/

Camera streaming, AprilTag detection and the tagged-body registry that the
control stack imports. These modules come from the lab's `visual_servoing`
package; what is here is the import closure the humanoid uses plus the
fabrication assets under `asset/`, not the whole library.

| Module | Role |
|---|---|
| `camera_streaming_utils.py` | multi-camera JPEG server/client over TCP |
| `camera_tag_detector.py`, `camera_tag_detection.py` | AprilTag detection and per-camera transforms |
| `camera_realsense_utils.py`, `camera_utils.py`, `camera_visualizer.py` | RealSense backends and viewers |
| `camera_offline_utils.py` | replay a recorded take through the same pipeline |
| `tagged_bodies/` | the registry of physical tagged objects — cubes, grippers, tables |
| `tagged_rigid_body.py` | batched scene of those bodies, and their geometry |
| `common/` | rotation helpers shared by the above |
| `vs_site.py` | camera serials and paths, resolved like `control/humanoid_site.py` |
| `asset/` | **fabrication assets, not imports**: the printable tag sheets and their generators (`tag_creation/`, PDFs + `print_*.py`), the tag-cube CAD for building the physical tag cube (`tag_cube_creation/`: `.obj`/`.mtl` and the wrist-interface `.step`, plus `export_cube_obj.py`), and the tripod base (`tripod_base/*.step`) |

The bundle also ships the asset generators under `asset/` because the tagged
objects are physical: a reader who wants to reproduce the grasp needs to print
the tag sheets and fabricate the tag cube and tripod base to the same geometry
the registry describes. Nothing in the control stack imports them; the
generators need `matplotlib` and `moms-apriltag` (both pinned) and, for the
3-D cube OBJ export, `cadquery` (optional — see `../requirements.txt`).

Not included: the `visual_servoing` scripts targeting other robots (Unitree G1,
UR5e, Pinocchio/mink experiments) and Unitree's ~146 MB of G1 meshes. They are
unrelated to this project.

Dependencies: `../requirements.txt` (one pinned set for the whole tree; the
`perception/` block there lists the modules only this directory needs).

## The tagged-body registry

A tagged body is authored **once**, here, and both the perception side and the
planner side read the same fields, so they cannot disagree about an object's
geometry. A table's slab dimensions are used by the detector to place it and by
the motion planner to avoid it, from one definition.

Tag *roll* is the one thing that cannot be guessed from a description: a flat
tag can be stuck at any of four 90° rolls, and a 90° error swaps length for
width in the collision model. Each body records the roll its physical instance
was actually built with.

## Configuration

Camera serials and paths resolve through `vs_site.py`: a `VS_*` environment
variable, then an optional gitignored `vs_site_local.py`, then a portable
default. `CAMERA_SERIALS` is **empty** by default — a serial names one physical
camera. It is read by `camera_tag_detection.py` and, when it is non-empty, by
the streaming server (`camera_streaming_utils.py multi-server`) as the default
of `--devices <left> <right>` (since 2026-08-22), so a rig can pin its
camera -> port order once; an explicit `--devices` still wins, and a bare
`--devices` with no values returns to auto-discovery. With neither, the server
takes whatever is attached, in USB enumeration order, and a left/right swap is
silent because everything downstream binds by port.

Calibration values are deliberately not site configuration. Tag sizes, board
layouts, intrinsics and tagged-body geometry are measured facts about physical
objects and live with the bodies they describe.
