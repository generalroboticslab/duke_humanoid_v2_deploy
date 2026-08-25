"""Pins for the 08-12 wrist observation frame de-fold (--obs-frame-fix).

The seam: encoders speak the rebuilt deploy model's frame (wrist_1 carries
-+pi/2 refs, left_wrist_2's axis is flipped), the policy was trained on the
training model's frame. wrist_defold_columns() computes which columns of the
combined 120-dim history row convert an encoder-frame row into a
training-frame row; these tests pin the column arithmetic and the numeric
transform against the known chest-home correspondence
(enc left_wrist_1 -1.2403 == train +0.3305, enc left_wrist_2 -1.4546 ==
train +1.4546) so a silent obs-layout change breaks a test, not a robot.
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_real_env import (  # noqa: E402
    URDF_JOINT_NAMES, _ARM_PATTERN, _WRIST_HALF_PI, wrist_defold_columns)

# the deployed policies' actor obs layout (boot log: 8 terms, D_total=120)
TERM_ORDER = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
              "actions", "command", "target_arm_joint_pos",
              "target_camera_joint_pos"]
TERM_DIMS = [3, 3, 31, 31, 31, 3, 14, 4]


def slices():
    out, s = [], 0
    for d in TERM_DIMS:
        out.append(slice(s, s + d))
        s += d
    return out


ARM_NAMES = [n for n in URDF_JOINT_NAMES if re.search(_ARM_PATTERN, n)]


class TestDefoldColumns(unittest.TestCase):
    def test_column_indices(self):
        plus, minus, neg = wrist_defold_columns(TERM_ORDER, slices(), ARM_NAMES)
        jp, jv, ta = 6, 37, 102          # term starts in the 120-dim row
        self.assertEqual(sorted(plus), sorted([jp + 17, ta + 4]))    # left_wrist_1
        self.assertEqual(sorted(minus), sorted([jp + 24, ta + 11]))  # right_wrist_1
        self.assertEqual(sorted(neg),
                         sorted([jp + 18, jv + 18, ta + 5]))         # left_wrist_2
        # disjoint: no column gets two corrections
        self.assertEqual(len(set(plus) | set(minus) | set(neg)),
                         len(plus) + len(minus) + len(neg))

    def test_arm_order_matches_wire_layout(self):
        # target_arm_joint_pos columns follow the _ARM_PATTERN filter order:
        # left arm 0-6, right arm 7-13, wrists at 4/5 and 11/12
        self.assertEqual(ARM_NAMES[4], "left_wrist_1_joint")
        self.assertEqual(ARM_NAMES[5], "left_wrist_2_joint")
        self.assertEqual(ARM_NAMES[11], "right_wrist_1_joint")
        self.assertEqual(ARM_NAMES[12], "right_wrist_2_joint")

    def test_numeric_transform_chest_home(self):
        """Encoder chest-home wrists de-fold to the known training values."""
        plus, minus, neg = wrist_defold_columns(TERM_ORDER, slices(), ARM_NAMES)
        row = np.zeros(120, dtype=np.float64)
        jp = 6
        row[jp + 17] = -1.2403           # enc left_wrist_1 (chest home)
        row[jp + 24] = +1.2403           # enc right_wrist_1
        row[jp + 18] = -1.4546           # enc left_wrist_2
        row[jp + 25] = +1.4546           # enc right_wrist_2 (no seam — untouched)
        row[37 + 18] = 0.30              # a left_wrist_2 velocity
        before = row.copy()
        row[plus] += _WRIST_HALF_PI
        row[minus] -= _WRIST_HALF_PI
        row[neg] *= -1.0
        self.assertAlmostEqual(row[jp + 17], +0.3305, places=4)
        self.assertAlmostEqual(row[jp + 24], -0.3305, places=4)
        self.assertAlmostEqual(row[jp + 18], +1.4546, places=4)
        self.assertAlmostEqual(row[jp + 25], +1.4546, places=4)   # unchanged
        self.assertAlmostEqual(row[37 + 18], -0.30, places=6)     # vel sign flips
        # every column outside the three sets is untouched
        touched = set(plus) | set(minus) | set(neg)
        for i in range(120):
            if i not in touched:
                self.assertEqual(row[i], before[i], f"column {i} moved")

    def test_wrist1_velocity_untouched(self):
        """Constant-offset corrections have zero derivative: no wrist_1 vel col."""
        _, _, neg = wrist_defold_columns(TERM_ORDER, slices(), ARM_NAMES)
        plus, minus, _ = wrist_defold_columns(TERM_ORDER, slices(), ARM_NAMES)
        jv = 37
        for col in list(plus) + list(minus):
            self.assertFalse(jv <= col < jv + 31,
                             "offset correction applied to a velocity column")


if __name__ == "__main__":
    unittest.main()
