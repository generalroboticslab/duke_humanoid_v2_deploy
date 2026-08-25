"""Regression checks for the grasp_cube family (pure Python, no pytest needed).

Run directly:
    python tagged_bodies/grasp_cube/verify_grasp_cube.py
prints PASS/FAIL per check and exits nonzero on any failure. The checks are also
written as ``test_*()`` functions, so if pytest is ever installed it collects them
as-is -- but pytest is NOT a dependency and is never imported here.

WHY THIS FILE EXISTS
--------------------
grasp_cube_60mm_e / _f carry 50 mm tags (not the usual 30 mm) on all six faces,
and their BOTTOM tag deliberately uses tag_x = -Y -- different from the tag_x = +X
that EVERY other cube's bottom uses, and 180 deg about the face normal from the
+Y this cube first carried (before the printed cube was re-checked). That value
looks like a typo but is not: it was set by physical inspection.
``test_ef_bottom_tag_x_is_minus_y`` locks it so a future "cleanup" cannot silently
flip it to +X (or back to +Y).

``test_ef_solvepnp_edges_are_50mm`` locks that the 50 mm actually reaches the pose
geometry (the solvePnP object points), not merely the config field -- that gap is
exactly where a rotation-fine / translation-x0.6 scale bug would hide.

All expected values below are hard-coded literals (not read from a snapshot), so
this file is a true fixed baseline and does not drift with the code it guards.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tagged_bodies
from tagged_rigid_body import TagRegistry

_CFGS = {c.name: c for c in tagged_bodies.ALL_CONFIGS}
_REG = TagRegistry(tagged_bodies.ALL_CONFIGS)

FACES6 = {"front", "back", "top", "bottom", "right", "left"}
BORDER = 0.005

# Six-faced 50 mm cubes added/completed this round: expected face -> id.
EF_IDS = {
    "grasp_cube_60mm_e": {"front": 303, "back": 304, "top": 300, "bottom": 305, "right": 301, "left": 302},
    "grasp_cube_60mm_f": {"front": 309, "back": 310, "top": 306, "bottom": 311, "right": 307, "left": 308},
}

# Legacy six-faced cubes that must stay untouched: name -> expected tag_size.
# Their bottom tag_x is +X (the pre-existing convention) and must not change.
LEGACY_TAG_SIZE = {
    "grasp_cube_60mm_a": 0.030,
    "grasp_cube_60mm_b": 0.030,
    "grasp_cube_60mm_c": 0.050,
    "grasp_cube_60mm_d": 0.050,
    "grasp_cube_40mm": 0.030,
    "grasp_cube_80mm": 0.030,
}


def _bottom_tag_x(cfg):
    return np.asarray(cfg.tags["bottom"].rot, float)[:, 0]


def _solvepnp_edges(cfg):
    op = cfg.tag_obj_pts  # (n, 4, 3) corners in the body frame
    return [float(np.linalg.norm(op[k, (j + 1) % 4] - op[k, j])) for k in range(len(op)) for j in range(4)]


def test_ef_have_six_faces():
    for name, ids in EF_IDS.items():
        c = _CFGS[name]
        assert len(c.tags) == 6, f"{name}: expected 6 tags, got {len(c.tags)}"
        assert set(c.tags.keys()) == FACES6, f"{name}: face keys {set(c.tags.keys())} != {FACES6}"
        assert {f: c.tags[f].id for f in ids} == ids, f"{name}: face->id mapping changed"


def test_ef_marker_geometry_is_50mm():
    for name in EF_IDS:
        c = _CFGS[name]
        assert c.tag_size == 0.050, f"{name}: cfg.tag_size {c.tag_size} != 0.050"
        assert c.tag_border_size == BORDER, f"{name}: cfg.tag_border_size {c.tag_border_size} != {BORDER}"
        for t in c.tags.values():
            assert abs(_REG.tag_sizes[t.id] - 0.050) < 1e-12, f"{name}: registry tag_size for id {t.id} != 0.050"


def test_ef_solvepnp_edges_are_50mm():
    # HARD guard: 50 mm reached the pose geometry, not just the config field.
    for name in EF_IDS:
        edges = _solvepnp_edges(_CFGS[name])
        assert np.allclose(edges, 0.050, atol=1e-9), f"{name}: a solvePnP tag edge != 0.050 m (translation-scale bug)"


def test_ef_bottom_tag_x_is_minus_y():
    # DELIBERATE anti-intuitive value: e/f bottom uses tag_x = -Y, unlike every
    # other cube's bottom (+X), and 180 deg from the +Y this file first carried.
    # Set by physical inspection of the printed cube -- do NOT "fix" it to +X/+Y.
    for name in EF_IDS:
        tx = _bottom_tag_x(_CFGS[name])
        assert np.allclose(tx, [0.0, -1.0, 0.0], atol=1e-12), f"{name}: bottom tag_x {tx} != -Y (0,-1,0)"


def test_legacy_cubes_unchanged():
    for name, sz in LEGACY_TAG_SIZE.items():
        c = _CFGS[name]
        assert c.tag_size == sz, f"{name}: tag_size {c.tag_size} != {sz}"
        assert c.tag_border_size == BORDER, f"{name}: tag_border_size {c.tag_border_size} != {BORDER}"
        tx = _bottom_tag_x(c)
        assert np.allclose(tx, [1.0, 0.0, 0.0], atol=1e-12), f"{name}: bottom tag_x {tx} != +X (must stay +X)"
        for t in c.tags.values():
            assert abs(_REG.tag_sizes[t.id] - sz) < 1e-12, f"{name}: registry tag_size for id {t.id} != {sz}"


def test_registry_loads_without_id_collision():
    seen = {}
    for c in tagged_bodies.ALL_CONFIGS:
        for t in c.tags.values():
            assert t.id not in seen, f"tag id {t.id} in {c.name} collides with {seen[t.id]}"
            seen[t.id] = c.name


def _all_tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


def main():
    failures = 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print("ALL PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
