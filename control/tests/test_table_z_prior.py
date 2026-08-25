"""Pins for the 08-12 table-resting z prior (build_scene / table_resting_z).

The incident it encodes: front grasps run on 1-face cube fits whose vertical
is off by tens of mm either way — grasping high/empty and pinning the
tip-guard floor above low targets (F_windup storms). A cube over a seen, sane
slab now takes its grasp z from physics: fitted top surface at the cube's own
xy + half the cube. Pinned here: the plane math (flat and tilted), the
footprint gate, the near-vertical junk-fit refusal, and the cap semantics.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_curobo_reach import (  # noqa: E402
    HAND_FLOOR_TARGET_GAP_M, HAND_FLOOR_TIP_MARGIN_M,
    HAND_FLOOR_XY_TILT_CAP_M, TABLE_Z_PRIOR_CAP_M, TABLE_Z_PRIOR_XY_SLACK_M,
    _table_surfaces, hand_floor_for_tables, table_resting_z)

DIMS = [1.2, 0.6, 0.03]
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def yaw_quat(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def pitch_quat(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0])


class TestTableRestingZ(unittest.TestCase):
    def test_flat_table_returns_surface_z(self):
        z = table_resting_z((0.1, -0.1), [0.5, 0.0, 0.045], IDENT, DIMS)
        self.assertAlmostEqual(z, 0.045, places=9)

    def test_yawed_table_footprint_rotates_with_it(self):
        # 90-deg yaw swaps the slab's long/short axes in the base frame
        q = yaw_quat(90.0)
        self.assertIsNotNone(table_resting_z((0.5, 0.55), [0.5, 0.0, 0.02], q, DIMS))
        self.assertIsNone(table_resting_z((0.9, 0.0), [0.5, 0.0, 0.02], q, DIMS))

    def test_outside_footprint_is_none(self):
        edge = DIMS[0] / 2 + TABLE_Z_PRIOR_XY_SLACK_M
        self.assertIsNone(
            table_resting_z((edge + 0.02, 0.0), [0.0, 0.0, 0.0], IDENT, DIMS))
        self.assertIsNotNone(
            table_resting_z((edge - 0.02, 0.0), [0.0, 0.0, 0.0], IDENT, DIMS))

    def test_tilted_fit_evaluates_plane_at_cube_xy(self):
        """A 5-deg pitched slab: surface z at x != center differs from the
        center z by dx*tan(5 deg) — the prior must follow the plane."""
        q = pitch_quat(5.0)
        z_c = table_resting_z((0.5, 0.0), [0.5, 0.0, 0.10], q, DIMS)
        z_off = table_resting_z((0.8, 0.0), [0.5, 0.0, 0.10], q, DIMS)
        self.assertAlmostEqual(z_c, 0.10, places=9)
        self.assertAlmostEqual(z_off - z_c, -0.3 * np.tan(np.radians(5.0)),
                               places=6)

    def test_near_vertical_fit_refused(self):
        self.assertIsNone(
            table_resting_z((0.0, 0.0), [0.0, 0.0, 0.0], pitch_quat(80.0), DIMS))

    def test_cap_constant_covers_the_measured_bias_band(self):
        """Tonight's measured 1-face vertical errors were 12-29 mm — the cap
        must admit them (override fires) while refusing a cube a full height
        above the slab (stacked/held)."""
        self.assertGreater(TABLE_Z_PRIOR_CAP_M, 0.03)
        self.assertLess(TABLE_Z_PRIOR_CAP_M, 0.09)


class _TableRig:
    """table_pose fake: (top-surface pos, quat_wxyz, pose spread deg)."""

    def __init__(self, poses):
        self.poses = poses

    def table_pose(self, name, now):
        return self.poses.get(name)


class TestFloorPriorConvention(unittest.TestCase):
    """#2 patch (08-13, the leg-2 refusal): the floor's tip guard must speak
    the SAME plane-at-xy surface as the z prior. The incident: a fit tilted
    3-4 deg put the slab-CENTER top 17.8 mm above the plane at the cube's xy;
    tip guard = center-top + 10 mm landed 13 mm ABOVE the -15 mm grasp line
    and all 30 solves went INFEASIBLE."""

    def _leg2_rig(self):
        # ~3.6 deg pitch: at 0.27 m from the center the plane drops ~17 mm,
        # today's measured mismatch, well inside the slab footprint.
        return _TableRig({"lab_table_b": ([0.635, 0.0, 0.0736],
                                          pitch_quat(3.6), 0.4)})

    def test_tip_guard_follows_the_plane_at_the_target_xy(self):
        rig = self._leg2_rig()
        xy = (0.905, 0.0)                       # 0.27 m downhill of the center
        surf = _table_surfaces(rig, ("lab_table_b",), xy)[0]
        self.assertAlmostEqual(surf, 0.0736 - 0.27 * np.tan(np.radians(3.6)),
                               places=3)
        # The floor must clamp to THIS surface + the tip guard — not to the
        # slab-CENTRE top + 10 mm, which is what vetoed the reach. The target
        # is placed exactly where the target cap wants the floor AT the
        # surface, so the tip guard is the binding constraint whatever the
        # grasp-line dial happens to be (it moved 0 -> +8 -> +5 mm in one
        # evening, and hard-coding around it made this pin lie).
        target_z = surf + HAND_FLOOR_TARGET_GAP_M
        floor = hand_floor_for_tables(rig, ("lab_table_b",),
                                      target_z=target_z, target_xy=xy)
        self.assertAlmostEqual(floor, surf + HAND_FLOOR_TIP_MARGIN_M,
                               places=9)
        self.assertLess(floor, target_z)
        # ...and with the cube high enough that the cap binds instead, the
        # floor sits exactly HAND_FLOOR_TARGET_GAP_M under the grasp point:
        # the gap is defined relative to the GRASP POINT, so it survives
        # every re-dial of GRASP_ANCHOR_ABOVE_M.
        high_z = surf + 0.030
        floor_hi = hand_floor_for_tables(rig, ("lab_table_b",),
                                         target_z=high_z, target_xy=xy)
        self.assertAlmostEqual(floor_hi, high_z - HAND_FLOOR_TARGET_GAP_M,
                               places=9)
        self.assertGreaterEqual(floor_hi, surf + HAND_FLOOR_TIP_MARGIN_M)

    def test_without_target_xy_the_center_top_still_rules(self):
        rig = self._leg2_rig()
        surf = _table_surfaces(rig, ("lab_table_b",), None)[0]
        self.assertAlmostEqual(surf, 0.0736, places=9)

    def test_target_off_the_footprint_falls_back_to_the_center_top(self):
        rig = self._leg2_rig()
        surf = _table_surfaces(rig, ("lab_table_b",), (5.0, 5.0))[0]
        self.assertAlmostEqual(surf, 0.0736, places=9)

    def test_a_wobbly_fit_extrapolation_is_clamped(self):
        # 12-deg tilt over 0.5 m extrapolates >100 mm below the center top —
        # that is the fit wobbling, not a bench; trust the center to the cap
        rig = _TableRig({"lab_table_b": ([0.5, 0.0, 0.10],
                                         pitch_quat(12.0), 0.4)})
        surf = _table_surfaces(rig, ("lab_table_b",), (1.0, 0.0))[0]
        self.assertAlmostEqual(surf, 0.10 - HAND_FLOOR_XY_TILT_CAP_M,
                               places=9)

    def test_highest_surface_still_wins_across_tables(self):
        rig = _TableRig({"lab_table_a": ([-1.43, 0.0, -0.025], IDENT, 0.4),
                         "lab_table_b": ([0.635, 0.0, 0.0736], IDENT, 0.4)})
        floor = hand_floor_for_tables(
            rig, ("lab_table_a", "lab_table_b"), target_z=0.30,
            target_xy=(0.5, 0.0))
        # far below the target: plain surface + jaw-sphere margin
        self.assertAlmostEqual(floor, 0.0736 + 0.023, places=9)


if __name__ == "__main__":
    unittest.main()
