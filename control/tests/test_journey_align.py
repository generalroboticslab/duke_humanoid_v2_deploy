"""Pin the journey's offset-pursuit aim law (journey_align_error).

History (2026-08-08): a decouple-then-walk align phase (rotate in place, then
walk straight) was implemented and review-hardened, then the --wz bench proved
this checkpoint cannot turn in place (0.2 rad/s = torso wind-up, feet planted;
0.4 = near-fall with dragging feet), and upstream ruled: CURVED WALKING ONLY, no
in-place rotation. The phase was removed. What survives is the law extraction:
journey_align_error is the single source both of the walk's steering and of
any future alignment logic, and these tests pin it.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanoid_curobo_reach as R  # noqa: E402
import humanoid_auto_operator as OP  # noqa: E402


class AlignLawTests(unittest.TestCase):

    def test_align_error_is_the_walks_own_steering_law(self):
        """Differential grid: journey_nav's wz must equal
        clip(K_STEER * journey_align_error) everywhere — one law, two users."""
        for x in (-1.2, -0.6, 0.5, 0.9, 1.4):
            for y in (-0.4, -0.1, 0.0, 0.2, 0.45):
                for arm in ("left", "right"):
                    pos = np.array([x, y, 0.05])
                    leg_dir = R.journey_leg_dir(pos)
                    vx, wz = R.journey_nav(pos, arm, leg_dir)
                    err = R.journey_align_error(pos, arm, leg_dir)
                    if abs(err) > OP.JOURNEY_FACE_TOL_RAD:
                        want = math.copysign(
                            float(np.clip(OP.K_STEER * abs(err),
                                          OP.JOURNEY_WZ_UNFACED_FLOOR,
                                          OP.JOURNEY_WZ_UNFACED_MAX)), err)
                    else:
                        want = float(np.clip(OP.K_STEER * err,
                                             -OP.WZ_MAX, OP.WZ_MAX))
                    self.assertAlmostEqual(wz, want, places=9,
                                           msg=f"pos={pos} arm={arm}")

    def test_zero_error_when_the_cube_sits_on_the_aim_line(self):
        pos = np.array([1.0, OP.JOURNEY_AIM_OFFSET_M, 0.0])
        self.assertAlmostEqual(
            R.journey_align_error(pos, "left", +1.0), 0.0, places=9)
        # Reverse legs have their OWN offset (user 08-10) AND a drift
        # compensation: the equilibrium sits DRIFT_COMP to the right of the
        # parking spot, so the backward gait's measured leftward settle
        # drift lands the cube on the spot itself.
        pos = np.array([-1.0, -OP.JOURNEY_AIM_OFFSET_BACK_M
                        - OP.JOURNEY_REVERSE_DRIFT_COMP_M, 0.0])
        self.assertAlmostEqual(
            R.journey_align_error(pos, "right", -1.0), 0.0, places=9)

    def test_reverse_aim_carries_no_drift_compensation(self):
        """08-13 night (user): the reverse right-bias is CANCELLED, comp 0.
        Ledger: the 08-10 full-length reverse legs all settled 0.15-0.21 m
        LEFT of their aim (both arms, three offsets) and paid for it here
        (0.15 -> 0.08 -> 0 on 08-12 -> 0.05 on the 08-13 morning); then the
        bias pushed a left-side cube (y +0.37 at survey) onto the MIDLINE
        (settled y -0.001) on a leg with no drift, and BOTH 08-13 reverse
        legs showed no drift at all -- so the compensation only steered aim
        error in. Current contract: reverse equilibria sit exactly at
        -/+ JOURNEY_AIM_OFFSET_BACK_M (right / left arm), forward equilibria
        at -/+ JOURNEY_AIM_OFFSET_M, nothing biased. The expected equilibria
        are still computed from the constants so a documented re-introduction
        (leftward misses of +0.15..0.21 on full-length legs) only has to
        revisit the explicit comp == 0 assertion, not the law."""
        comp = OP.JOURNEY_REVERSE_DRIFT_COMP_M
        self.assertEqual(comp, 0.0,
                         "reverse drift comp is back -- the 08-13 night "
                         "cancellation is the documented decision; re-pin "
                         "the ledger before re-introducing a bias")
        # Reverse equilibria: the parking spot itself, both arms. Computed
        # from the constants (offset minus comp) and ALSO pinned to the
        # unbiased spot, so a silent comp != 0 shows up twice.
        eq_r = -OP.JOURNEY_AIM_OFFSET_BACK_M - comp
        eq_l = OP.JOURNEY_AIM_OFFSET_BACK_M - comp
        self.assertAlmostEqual(eq_r, -OP.JOURNEY_AIM_OFFSET_BACK_M, places=12)
        self.assertAlmostEqual(eq_l, OP.JOURNEY_AIM_OFFSET_BACK_M, places=12)
        self.assertAlmostEqual(R.journey_align_error(
            np.array([-1.0, eq_r, 0.0]), "right", -1.0), 0.0, places=9)
        self.assertAlmostEqual(R.journey_align_error(
            np.array([-1.0, eq_l, 0.0]), "left", -1.0), 0.0, places=9)
        # Both arms are treated alike: a rear cube displaced the same
        # distance off its own aim line reads the same |error| either side.
        for d in (0.10, 0.25):
            self.assertAlmostEqual(
                R.journey_align_error(
                    np.array([-1.0, eq_r - d, 0.0]), "right", -1.0),
                -R.journey_align_error(
                    np.array([-1.0, eq_l + d, 0.0]), "left", -1.0),
                places=9, msg=f"d={d}")
        # forward equilibria unchanged and carry NO compensation either
        self.assertAlmostEqual(R.journey_align_error(
            np.array([1.0, OP.JOURNEY_AIM_OFFSET_M, 0.0]), "left", 1.0),
            0.0, places=9)
        self.assertAlmostEqual(R.journey_align_error(
            np.array([1.0, -OP.JOURNEY_AIM_OFFSET_M, 0.0]), "right", 1.0),
            0.0, places=9)

    def test_each_direction_uses_its_own_offset(self):
        """Guard the split itself: a reverse leg steered with the FORWARD
        offset would re-open the over-curved backward walk the 08-10 change
        exists to close."""
        if OP.JOURNEY_AIM_OFFSET_M == OP.JOURNEY_AIM_OFFSET_BACK_M:
            self.skipTest("offsets currently equal — the split is unobservable")
        pos = np.array([-1.0, -OP.JOURNEY_AIM_OFFSET_M, 0.0])
        self.assertNotAlmostEqual(
            R.journey_align_error(pos, "right", -1.0), 0.0, places=3,
            msg="a reverse leg is steering on the forward offset")

    def test_reverse_error_is_measured_against_the_backward_axis(self):
        """A rear cube slightly off the AIM LINE must read as a SMALL error
        against the backward axis — without the pi-wrap it would read ~pi and
        the steering would command the forbidden turn-around. Offset-relative
        so the test survives aim-offset experiments."""
        pos = np.array([-1.0, -OP.JOURNEY_AIM_OFFSET_BACK_M
                        - OP.JOURNEY_REVERSE_DRIFT_COMP_M - 0.25, 0.0])
        err = R.journey_align_error(pos, "right", -1.0)
        self.assertLess(abs(err), math.radians(15.0))

    def test_unfaced_steering_holds_the_yaw_floor(self):
        """Sim parity (MoverCfg.turn_cruise, 08-08): outside the face
        tolerance the yaw command is FLOORED, never the old proportional
        crawl — sub-floor yaw is the measured dead band on both sim and
        hardware. Inside, the validated WZ_MAX trim rules."""
        # 20 deg off the aim line: old law gave clip(0.6*0.35)=0.2; the arc
        # law must give at least the floor.
        pos = np.array([1.0, OP.JOURNEY_AIM_OFFSET_M
                        + math.tan(math.radians(20.0)), 0.0])
        vx, wz = R.journey_nav(pos, "left", +1.0)
        self.assertGreaterEqual(abs(wz), OP.JOURNEY_WZ_UNFACED_FLOOR - 1e-9)
        self.assertLessEqual(abs(wz), OP.JOURNEY_WZ_UNFACED_MAX + 1e-9)
        self.assertGreater(wz, 0.0)                 # toward the aim line
        self.assertAlmostEqual(vx, OP.CRUISE_VX)    # arcs AT CRUISE, no pause
        # 3 deg off: faced — the trim cap rules and the floor must NOT apply.
        pos = np.array([1.0, OP.JOURNEY_AIM_OFFSET_M
                        + math.tan(math.radians(3.0)), 0.0])
        _, wz = R.journey_nav(pos, "left", +1.0)
        self.assertLess(abs(wz), OP.JOURNEY_WZ_UNFACED_FLOOR)
        self.assertLessEqual(abs(wz), OP.WZ_MAX + 1e-9)

    def test_the_arc_law_is_direction_symmetric(self):
        """The fully mirrored geometry (rear cube, mirrored side, reversed
        travel) must produce the IDENTICAL command — fore-aft symmetry is the
        journey's design, and the wrapped error comes out equal, not
        negated."""
        # Each direction measured against ITS OWN equilibrium (offsets and
        # the reverse drift comp differ since 08-10): geometry equally
        # displaced from its own aim line still commands identically.
        fwd = np.array([1.0, OP.JOURNEY_AIM_OFFSET_M + 0.4, 0.0])
        rev = np.array([-1.0, -OP.JOURNEY_AIM_OFFSET_BACK_M
                        - OP.JOURNEY_REVERSE_DRIFT_COMP_M - 0.4, 0.0])
        ef = R.journey_align_error(fwd, "left", +1.0)
        er = R.journey_align_error(rev, "right", -1.0)
        self.assertAlmostEqual(ef, er, places=9)
        vf, wf = R.journey_nav(fwd, "left", +1.0)
        vr, wr = R.journey_nav(rev, "right", -1.0)
        self.assertAlmostEqual(wf, wr, places=9)
        self.assertAlmostEqual(vf, -vr, places=9)   # travel reverses, yaw not

    def test_the_walk_loop_slews_its_yaw_command(self):
        """Sim MAX_ACCEL parity, second half of the port: the published wz
        must be rate-limited. The first arc run went out unramped and the
        yaw floor turned the face boundary into a growing slalom (measured
        leg 2, 08-08 night: wz -0.35 -> +0.35 -> -0.50)."""
        import inspect
        src = inspect.getsource(R.run_visit)
        self.assertIn("JOURNEY_WZ_SLEW", src,
                      "run_visit publishes the raw law wz — the bang-bang "
                      "slalom is back")
        self.assertGreater(OP.JOURNEY_WZ_SLEW, 0.0)

    def test_no_in_place_rotation_anywhere_in_the_mission(self):
        """Upstream ruling 2026-08-08: curved walking only. The removed align phase must
        stay removed — no mission code may command wz with vx == 0 (the walk
        loop's blind-stop (0,0) is a STOP, not a turn)."""
        self.assertFalse(hasattr(R, "journey_align"),
                         "journey_align is back — in-place rotation is "
                         "forbidden on this robot (upstream ruling 08-08)")
        self.assertFalse(hasattr(OP, "JOURNEY_ALIGN_ENABLED"),
                         "the align gate constant is back")


if __name__ == "__main__":
    unittest.main()
