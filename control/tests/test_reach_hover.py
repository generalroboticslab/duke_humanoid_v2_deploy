"""Hover endpoint (user 2026-08-03: 'end every route 3 cm above the cube,
then descend and grasp'). The offset lives ONLY in the solver-facing copies;
the true targets object — which feeds MPC goals, drift watches and the leash
anchor — must pass through untouched, or the tracker would descend onto a
point 3 cm in the air."""
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"


@unittest.skipIf(R is None, _SKIP)
class ReachHoverTests(unittest.TestCase):

    def test_solver_copies_are_lifted_and_originals_untouched(self):
        targets = {"a": np.array([0.35, 0.20, 0.05]),
                   "b": np.array([0.30, -0.25, 0.09])}
        lifted = R.hover_targets(targets, 0.03)
        for k in targets:
            np.testing.assert_allclose(lifted[k][:2], targets[k][:2],
                                       err_msg="hover moved the target in XY")
            self.assertAlmostEqual(float(lifted[k][2] - targets[k][2]), 0.03,
                                   places=9)
        self.assertAlmostEqual(float(targets["a"][2]), 0.05, places=9,
                               msg="the TRUE target was mutated — the MPC "
                                   "would descend onto a hover in the air")

    def test_zero_hover_is_identity(self):
        targets = {"a": np.array([0.35, 0.20, 0.05])}
        self.assertIs(R.hover_targets(targets, 0.0), targets)

    def test_the_default_is_15mm(self):
        self.assertAlmostEqual(R.Args.reach_hover_mm, 15.0, places=9,
                               msg="the user's 'every run' spec is the default "
                                   "(30 mm trimmed to 15 mm, 2026-08-03)")


if __name__ == "__main__":
    unittest.main()
