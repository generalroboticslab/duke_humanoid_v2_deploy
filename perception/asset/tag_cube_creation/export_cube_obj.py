"""Export tag cubes as separate white/black OBJ files per cube.

Generates (4 files total):
  tag_cube_0_white.obj  — left wrist white face slabs
  tag_cube_0_black.obj  — left wrist black core + pixels + wrist interface
  tag_cube_1_white.obj  — right wrist white face slabs
  tag_cube_1_black.obj  — right wrist black core + pixels + wrist interface

In MJCF use two geoms per body with explicit rgba:
  <geom type="mesh" mesh="tag_cube_0_white" rgba="0.95 0.95 0.95 1"/>
  <geom type="mesh" mesh="tag_cube_0_black" rgba="0.04 0.04 0.04 1"/>

Usage:
  python export_cube_obj.py [--out-dir DIR]

Geometry (mm):
  Wrist interface: Z=[-12, 0]
  Core:            Z=[0, 40], X/Y=[-20, 20]
  Cube center:     Z=+20 (body frame origin in visual_servoing convention)
  mesh_scale=0.001 converts mm→m in MJCF.
"""

import argparse
import pathlib
import struct
import tempfile

import cadquery as cq
import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent

# tag_cube_0: left wrist  (matches tagged_bodies/tag_cube/tag_cube_0.py)
# tag_cube_1: right wrist (matches tagged_bodies/tag_cube/tag_cube_1.py)
CUBES = {
    "tag_cube_0": {"top": 586, "front": 585, "back": 584, "left": 583, "right": 582},
    "tag_cube_1": {"top": 581, "front": 580, "back": 579, "left": 578, "right": 577},
}

ARUCO_DICT = {
    "DICT_4X4_50":          cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":         cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":         cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000":        cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":          cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":         cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":         cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000":        cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":          cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":         cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":         cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000":        cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":          cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":         cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":         cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000":        cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL":  cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5":   cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9":   cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10":  cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11":  cv2.aruco.DICT_APRILTAG_36h11,
}

DICT_TYPE     = "DICT_APRILTAG_36h11"
CORE_SIZE     = 40.0  # inner cube side (mm); must equal MARKER_SIZE + 2*BORDER
TAG_THICKNESS = 2.0   # face slab thickness (mm)
BLACK_HEIGHT  = 2     # black pixel extrusion depth on outer face (mm)
MARKER_SIZE   = 30.0  # ArUco pattern area (mm)
BORDER        = 5.0   # quiet zone beyond pattern on each side (mm)

assert MARKER_SIZE + 2 * BORDER == CORE_SIZE, "MARKER_SIZE + 2*BORDER must equal CORE_SIZE"

_OUTER = CORE_SIZE / 2 + TAG_THICKNESS   # 22.0 — outer corner XY position
_TOP_Z = CORE_SIZE + TAG_THICKNESS       # 42.0 — top face Z


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _occupancy_grid(marker_id: int):
    """Return 0=black, 1=white cell grid for the marker."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[DICT_TYPE])
    # 36h11: 6x6 data + 1-cell mandatory border → 8x8 image
    img = np.array(aruco_dict.generateImageMarker(marker_id, 8))
    return np.where(img == 255, 1, 0)


def _face_slab(marker_id: int):
    """
    Build a CORE_SIZE × CORE_SIZE × TAG_THICKNESS slab in XY, centered at origin.
    Tag pattern is on the +Z face (outer face before placement).
    Returns (white_slab, black_slab | None).
    """
    grid = _occupancy_grid(marker_id)
    n = grid.shape[0]        # 8 for 36h11
    sq = MARKER_SIZE / n     # cell side in mm
    half_m = MARKER_SIZE / 2

    black_pts = []
    for i in range(n):
        for j in range(n):
            if grid[i, j] == 0:
                x = j * sq - half_m + sq / 2
                y = (n - 1 - i) * sq - half_m + sq / 2
                black_pts.append((x, y))

    solid = cq.Workplane("XY").rect(CORE_SIZE, CORE_SIZE).extrude(TAG_THICKNESS)

    if black_pts:
        pixels = (
            cq.Workplane("XY")
            .workplane(offset=TAG_THICKNESS - BLACK_HEIGHT)
            .pushPoints(black_pts)
            .rect(sq, sq)
            .extrude(BLACK_HEIGHT)
        )
        return solid.clean(), pixels.clean()

    return solid.clean(), None


def _place(slab, axis: str, deg: float, translate: tuple):
    """Rotate slab around a cardinal axis then translate."""
    ax_vec = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return slab.rotate((0, 0, 0), ax_vec, deg).translate(translate)


def _edge_strips():
    """
    8 white filler strips that close the open seams left by the 40x40 face slabs.

    4 vertical edges (2x2x40mm columns at each vertical corner of the cube):
      Front-Left:  X=[-22,-20], Y=[-22,-20], Z=[0,40]
      Front-Right: X=[ 20, 22], Y=[-22,-20], Z=[0,40]
      Back-Left:   X=[-22,-20], Y=[ 20, 22], Z=[0,40]
      Back-Right:  X=[ 20, 22], Y=[ 20, 22], Z=[0,40]

    4 top horizontal edges (2mm tall strips along the top perimeter, Z=[40,42]):
      Front/Back: 44mm wide (X=[-22,22]) to also cap the 4 top corners
      Left/Right:  40mm deep (Y=[-20,20]), corners already covered above
    """
    t    = TAG_THICKNESS   # 2
    half = CORE_SIZE / 2   # 20
    c    = half + t / 2    # 21  (center of a 2mm strip butted against ±20)

    pieces = []

    for sx in (-1, 1):
        for sy in (-1, 1):
            pieces.append(
                cq.Workplane("XY")
                .center(sx * c, sy * c)
                .rect(t, t)
                .extrude(CORE_SIZE)
            )

    for sy in (-1, 1):
        pieces.append(
            cq.Workplane("XY")
            .workplane(offset=CORE_SIZE)
            .center(0, sy * c)
            .rect(CORE_SIZE + 2 * t, t)
            .extrude(t)
        )
    for sx in (-1, 1):
        pieces.append(
            cq.Workplane("XY")
            .workplane(offset=CORE_SIZE)
            .center(sx * c, 0)
            .rect(t, CORE_SIZE)
            .extrude(t)
        )

    result = pieces[0]
    for p in pieces[1:]:
        result = result.union(p, clean=False)
    return result.clean()


class _TopEdgeSel(cq.selectors.Selector):
    """Selects horizontal edges that lie in the Z=_TOP_Z plane."""
    def filter(self, objs):
        return [e for e in objs
                if abs(e.BoundingBox().zmin - _TOP_Z) < 0.1
                and abs(e.BoundingBox().zmax - _TOP_Z) < 0.1]


class _VertOuterSel(cq.selectors.Selector):
    """Selects Z-parallel edges at the 4 outer corner positions X=±_OUTER, Y=±_OUTER."""
    def filter(self, objs):
        result = []
        for e in objs:
            t = e.tangentAt(0.5)
            if abs(abs(t.z) - 1.0) > 0.01:
                continue
            bb = e.BoundingBox()
            cx = (bb.xmin + bb.xmax) / 2
            cy = (bb.ymin + bb.ymax) / 2
            if abs(abs(cx) - _OUTER) < 0.5 and abs(abs(cy) - _OUTER) < 0.5:
                result.append(e)
        return result


def build_cube(tag_ids: dict):
    """
    Assemble the AprilTag cube.

    tag_ids: mapping of face name → AprilTag marker ID.

    Slab XY convention before placement (centered at origin):
      X=[-20,20], Y=[-20,20], Z=[0,2]  outer (tag) face at Z=2

    After rotation + translation each slab lands at:
      top:   Z=[40,42]              (no rotation, translate Z+40)
      front: Y=[-22,-20], Z=[0,40]  (+90° X-axis  → translate (0,-20,+20))
      back:  Y=[ 20, 22], Z=[0,40]  (-90° X-axis  → translate (0,+20,+20))
      right: X=[ 20, 22], Z=[0,40]  (+90° Y-axis  → translate (+20,0,+20))
      left:  X=[-22,-20], Z=[0,40]  (-90° Y-axis  → translate (-20,0,+20))
    """
    half = CORE_SIZE / 2  # 20.0

    # Tag orientation per face (world-frame unit vectors after rotation):
    #   TagConfig.rot cols = [tag_right, -tag_up, -outward]
    #                        (AprilTag: X=right, Y=down, Z=into-face)
    #
    #   face   axis  deg   tag_right  tag_up  outward
    #   top    z      0    +X         +Y      +Z
    #   front  x    +90    +X         +Z      -Y
    #   back   x    -90    +X         -Z      +Y
    #   right  y    +90    -Z         +Y      +X
    #   left   y    -90    +Z         +Y      -X
    placements = {
        "top":   ("z",   0,  (0,     0,     CORE_SIZE)),
        "front": ("x",  90,  (0,    -half,  half)),
        "back":  ("x", -90,  (0,     half,  half)),
        "right": ("y",  90,  (half,  0,     half)),
        "left":  ("y", -90,  (-half, 0,     half)),
    }

    core = cq.Workplane("XY").rect(CORE_SIZE, CORE_SIZE).extrude(CORE_SIZE)

    wrist_path = str(HERE / "v2_wrist_interface.step")
    wrist = cq.importers.importStep(wrist_path)

    shell = None
    pixel_vols = []

    for face, (axis, deg, xlate) in placements.items():
        solid_slab, pixels = _face_slab(tag_ids[face])
        s_placed = _place(solid_slab, axis, deg, xlate)
        shell = s_placed if shell is None else shell.union(s_placed, clean=False)
        if pixels is not None:
            pixel_vols.append(_place(pixels, axis, deg, xlate))

    shell = shell.union(_edge_strips(), clean=False).clean()
    shell = shell.edges(_TopEdgeSel()).chamfer(1.0)
    shell = shell.edges(_VertOuterSel()).chamfer(1.0)

    white_body = shell
    for pv in pixel_vols:
        white_body = white_body.cut(pv, clean=False)
    white_body = white_body.clean()

    black_body = core.union(wrist, clean=False)
    for pv in pixel_vols:
        black_body = black_body.union(pv, clean=False)
    black_body = black_body.clean()

    return white_body, black_body


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

_STL_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("v0",     "<f4", (3,)),
    ("v1",     "<f4", (3,)),
    ("v2",     "<f4", (3,)),
    ("attr",   "<u2"),
])


def _load_stl(path: str) -> np.ndarray:
    """Binary STL → (N, 3, 3) float32 vertex positions."""
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        data = np.frombuffer(f.read(), dtype=_STL_DTYPE)
    assert len(data) == n, f"STL triangle count mismatch: header={n} data={len(data)}"
    return np.stack([data["v0"], data["v1"], data["v2"]], axis=1)  # (N, 3, 3)


def _write_obj(obj_path: pathlib.Path, tris: np.ndarray) -> None:
    """Write single-material OBJ (color set via geom rgba in MJCF)."""
    with open(obj_path, "w") as f:
        for tri in tris:
            for v in tri:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for i in range(len(tris)):
            b = i * 3 + 1
            f.write(f"f {b} {b+1} {b+2}\n")
    print(f"  {obj_path.name}: {len(tris)} tris")


def export_cube(name: str, tag_ids: dict, out_dir: pathlib.Path) -> None:
    print(f"Building {name} (top={tag_ids['top']} front={tag_ids['front']} ...)...")
    white_body, black_body = build_cube(tag_ids)

    with tempfile.TemporaryDirectory() as tmp:
        wp = f"{tmp}/white.stl"
        bp = f"{tmp}/black.stl"
        white_body.val().exportStl(wp)
        black_body.val().exportStl(bp)
        white_tris = _load_stl(wp)
        black_tris = _load_stl(bp)

    _write_obj(out_dir / f"{name}_white.obj", white_tris)
    _write_obj(out_dir / f"{name}_black.obj", black_tris)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _default_out = str(HERE.parents[2] / "legged_env_v2/assets/create/meshes/tag_cube")
    parser.add_argument("--out-dir", default=_default_out,
                        help="Output directory (default: legged_env_v2/assets/create/meshes/tag_cube)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, tag_ids in CUBES.items():
        export_cube(name, tag_ids, out_dir)

    # Remove stale single-file OBJ and MTL from previous versions
    for name in CUBES:
        for stale in (f"{name}.obj", f"{name}.mtl"):
            p = out_dir / stale
            if p.exists():
                p.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
