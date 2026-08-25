"""Pins for the 08-12 left_wrist_2 deviation mirror (--lw2-mirror).

The finding it encodes: seed1 front grasps tilt clockwise with the LEFT arm
and not the RIGHT — and left_wrist_2 is the stack's only left-only frame
asymmetry (encoder sign inverted vs the training frame). The mirror corrects
ONLY the deviation: obs = 2*default - enc for position and target, vel
negated, action residual flipped. Contract pinned here: at the default the
policy-facing row is BIT-IDENTICAL to the raw one (the property whose absence
killed --obs-frame-fix on the hang), and exactly three columns ever move.
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_real_env import (  # noqa: E402
    URDF_JOINT_NAMES, _ARM_PATTERN, wrist_defold_columns)

TERM_ORDER = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
              "actions", "command", "target_arm_joint_pos",
              "target_camera_joint_pos"]
TERM_DIMS = [3, 3, 31, 31, 31, 3, 14, 4]
LW2_DEFAULT = -1.4546          # Cosine chest home, encoder frame


def slices():
    out, s = [], 0
    for d in TERM_DIMS:
        out.append(slice(s, s + d))
        s += d
    return out


ARM_NAMES = [n for n in URDF_JOINT_NAMES if re.search(_ARM_PATTERN, n)]
_, _, NEG = wrist_defold_columns(TERM_ORDER, slices(), ARM_NAMES)
POS_COL, VEL_COL, TARGET_COL = NEG      # [jpos, jvel, target] for left_wrist_2


def apply_mirror(row):
    out = row.copy()
    for c in (POS_COL, TARGET_COL):
        out[c] = 2.0 * LW2_DEFAULT - out[c]
    out[VEL_COL] = -out[VEL_COL]
    return out


class TestLw2Mirror(unittest.TestCase):
    def test_static_default_is_bit_identical(self):
        """The property --obs-frame-fix lacked: at rest, nothing changes."""
        row = np.random.default_rng(0).normal(0, 0.3, 120)
        row[POS_COL] = LW2_DEFAULT
        row[TARGET_COL] = LW2_DEFAULT
        row[VEL_COL] = 0.0
        np.testing.assert_array_equal(apply_mirror(row), row)

    def test_deviation_is_mirrored(self):
        row = np.zeros(120)
        row[POS_COL] = LW2_DEFAULT + 0.5       # wrist folded 0.5 rad one way
        row[TARGET_COL] = LW2_DEFAULT - 0.2
        row[VEL_COL] = 0.3
        out = apply_mirror(row)
        self.assertAlmostEqual(out[POS_COL], LW2_DEFAULT - 0.5, places=9)
        self.assertAlmostEqual(out[TARGET_COL], LW2_DEFAULT + 0.2, places=9)
        self.assertAlmostEqual(out[VEL_COL], -0.3, places=9)

    def test_only_three_columns_touched(self):
        row = np.random.default_rng(1).normal(0, 0.5, 120)
        out = apply_mirror(row)
        moved = set(np.nonzero(out != row)[0].tolist())
        self.assertEqual(moved, {POS_COL, VEL_COL, TARGET_COL})

    def test_mirror_is_involution(self):
        """Applying twice restores the original — no drift on re-application."""
        row = np.random.default_rng(2).normal(0, 0.5, 120)
        np.testing.assert_allclose(apply_mirror(apply_mirror(row)), row,
                                   atol=1e-12)

    def test_columns_are_the_left_wrist_2_slots(self):
        self.assertEqual(POS_COL, 6 + URDF_JOINT_NAMES.index("left_wrist_2_joint"))
        self.assertEqual(VEL_COL, 37 + URDF_JOINT_NAMES.index("left_wrist_2_joint"))
        self.assertEqual(TARGET_COL, 102 + ARM_NAMES.index("left_wrist_2_joint"))


if __name__ == "__main__":
    unittest.main()
