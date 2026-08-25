"""The target envelope, and the reason it now reports instead of a bare refusal.

The envelope itself is unchanged and must stay that way — these bounds are what
keep a reach inside the arm's measured workspace. What changed is the OPERATOR's
experience of hitting it: a cube seen out of reach used to end the run, so
fixing a 5 cm placement error meant relaunching a five-process stack. Now the
tool waits, and prints which correction to make.

So the tests below pin two separate things: the bounds themselves (a silent
widening here would let a reach leave the workspace), and the actionability of
each message (a "refused" with no direction is a second run to find out why).
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"


@unittest.skipIf(R is None, _SKIP)
class EnvelopeTests(unittest.TestCase):

    def test_a_target_in_the_middle_of_the_envelope_is_accepted(self):
        self.assertIsNone(R.target_unreachable_why(np.array([0.38, 0.22, 0.09])))

    def test_each_bound_is_reported_with_the_direction_to_move(self):
        """Every message must name the correction, not just the violation."""
        cases = [
            (np.array([0.62, 0.31, 0.08]), "CLOSER"),      # r beyond reach
            (np.array([0.10, 0.05, 0.00]), "AWAY"),        # inside the keep-out
            # z cases derive from the live dials (08-11: the ceiling moved
            # 0.30 -> 0.50 by user order and a hardcoded 0.45 became legal)
            (np.array([0.35, 0.20, OP.TARGET_Z_MAX_M + 0.05]), "LOWER"),
            (np.array([0.35, 0.20, OP.TARGET_Z_MIN_M - 0.20]), "HIGHER"),
        ]
        for pos, direction in cases:
            with self.subTest(pos=pos.tolist()):
                why = R.target_unreachable_why(pos)
                self.assertIsNotNone(why)
                self.assertIn(direction, why)

    def test_the_bounds_are_exactly_the_operator_constants(self):
        """No second copy of the workspace. If the operator's envelope moves,
        this tool must move with it — a divergence would be invisible until a
        reach left the measured workspace."""
        eps = 1e-6
        r = OP.TARGET_MAX_RADIUS_M
        self.assertIsNone(R.target_unreachable_why(np.array([r - eps, 0.0, 0.0])))
        self.assertIsNotNone(R.target_unreachable_why(np.array([r + 1e-3, 0.0, 0.0])))
        lo = OP.TARGET_MIN_RADIUS_M
        self.assertIsNone(R.target_unreachable_why(np.array([lo + eps, 0.0, 0.0])))
        self.assertIsNotNone(R.target_unreachable_why(np.array([lo - 1e-3, 0.0, 0.0])))
        for z, inside in ((OP.TARGET_Z_MAX_M - eps, True),
                          (OP.TARGET_Z_MAX_M + 1e-3, False),
                          (OP.TARGET_Z_MIN_M + eps, True),
                          (OP.TARGET_Z_MIN_M - 1e-3, False)):
            with self.subTest(z=z):
                why = R.target_unreachable_why(np.array([0.35, 0.0, z]))
                self.assertEqual(why is None, inside)

    def test_a_non_finite_position_is_refused_before_any_arithmetic(self):
        """NaN reaches the envelope check from a dropped detection; comparisons
        against it are all False, so an unguarded check would ACCEPT it."""
        for bad in (np.array([np.nan, 0.2, 0.1]), np.array([0.3, np.inf, 0.1])):
            with self.subTest(pos=bad.tolist()):
                self.assertIsNotNone(R.target_unreachable_why(bad))

    def test_the_overshoot_is_quantified(self):
        """'move it 9 cm closer' beats 'out of envelope' by exactly one run.
        Anchored to the constant so envelope dials don't break the pin."""
        why = R.target_unreachable_why(
            np.array([OP.TARGET_MAX_RADIUS_M + 0.09, 0.0, 0.0]))
        self.assertIn("9 cm", why)


def _tel(q_left, q_right, gravity=(0.0, 0.0, -1.0)):
    jp = np.zeros(31)
    jp[13:20] = q_left
    jp[20:27] = q_right
    return {"projected_gravity": list(gravity), "joint_pos": jp.tolist()}


@unittest.skipIf(R is None, _SKIP)
class PostureGateTests(unittest.TestCase):
    """The other gate that fires before anything is planned.

    It exists because reaching from an unknown posture crosses configuration
    basins, and it must keep firing. But its dominant real-world cause is not an
    unknown posture at all — it is real_env's arm-silence failsafe still crawling
    the arms home at 0.025 rad/s after the previous tool exited, a ~30 s journey
    from the raised chest home. So the gate reports the DISTANCE as well as the
    verdict, which is what lets the caller wait it out.
    """

    def setUp(self):
        self.left = np.asarray(OP.POWERON_JOINTS, float)
        self.right = np.asarray(OP.mirror_arm(OP.POWERON_JOINTS, "right"), float)

    def test_a_known_home_passes_with_zero_distance(self):
        why, err = R.posture_gate(_tel(self.left, self.right))
        self.assertIsNone(why)
        self.assertAlmostEqual(err, 0.0, places=9)

    def test_every_accepted_home_is_accepted(self):
        """Because a tool that only knew POWERON would refuse an arm parked at
        a station the operator legitimately left it at.

        ENUMERATED FROM THE GATE, not hand-listed. The 07-31 review found this
        test claiming "all four" while posture_gate held six — the migration's
        chest_home_0730 entry could be deleted with the whole suite still
        green, and that entry is the most operationally load-bearing line of
        the change (the robot is physically parked at the 07-30 posture through
        the migration, so losing it refuses the first run after a failsafe
        crawl the operator already waited out). Reading the gate's own dict
        means a future home cannot be added without being covered here."""
        homes = R.posture_gate_homes()
        self.assertGreaterEqual(len(homes), 6, "the accepted-home set shrank")
        for name, posture in homes.items():
            with self.subTest(home=name):
                q = np.asarray(posture, float)
                why, _ = R.posture_gate(_tel(q, OP.mirror_arm(posture, "right")))
                self.assertIsNone(why, f"{name} is listed but refused: {why}")

    def test_the_legacy_0730_chest_posture_is_accepted(self):
        """Named explicitly as well as enumerated, because this one has a
        deadline: it exists so the FIRST post-migration run is not refused
        while the arms still sit where the pre-migration session left them.
        It is 53 deg from every other accepted home."""
        q = np.asarray(OP.POWERON_JOINTS_LEGACY_0730, float)
        why, err = R.posture_gate(
            _tel(q, OP.mirror_arm(OP.POWERON_JOINTS_LEGACY_0730, "right")))
        self.assertIsNone(why, f"the 07-30 chest posture is refused: {why}")
        self.assertAlmostEqual(err, 0.0, places=9)
        self.assertGreater(
            float(np.max(np.abs(q - np.asarray(OP.POWERON_JOINTS, float)))),
            OP.FRONT_HOME_TOL_RAD,
            "the 07-30 posture is now within tolerance of the sim home — this "
            "entry is redundant and the migration note should say so")

    def test_an_off_home_arm_is_refused_with_its_distance_and_nearest_home(self):
        """The message must name the nearest home, because '0.4 rad from
        chest_home' is a crawl to wait out while '1.7 rad from side_home' is an
        arm somewhere it should not be, and those need different responses.

        Shoulder_1, not the elbow: 0.61 rad on the ELBOW lands near the chest
        home. And since 2026-07-30 the label is "chest_home" rather than
        "poweron" — those are now the SAME posture (the power-on pose was
        raised to the working one), so both names are correct and the dict
        simply reports the first. chest_home is the better of the two to print:
        it is the home the operator commands.
        """
        off = self.left.copy()
        off[0] += 0.61
        why, err = R.posture_gate(_tel(off, self.right))
        self.assertIsNotNone(why)
        self.assertIn("35 deg", why)
        self.assertIn("nearest: chest_home", why)
        self.assertAlmostEqual(err, 0.61, places=6)

    def test_the_raised_chest_home_is_recognised(self):
        """The fix for the refusal seen on hardware 2026-07-30. An operator run
        with --chest-home parks the arms here; without recognition the next tool
        refuses until the failsafe has crawled them 0.754 rad back to POWERON —
        30 s spent undoing the pose the previous run just set up."""
        chest = np.asarray(OP.CHEST_HOME_JOINTS, float)
        why, err = R.posture_gate(
            _tel(chest, OP.mirror_arm(OP.CHEST_HOME_JOINTS, "right")))
        self.assertIsNone(why)
        self.assertAlmostEqual(err, 0.0, places=9)

    def test_a_few_seconds_of_failsafe_crawl_still_counts_as_the_chest_home(self):
        """Ctrl+C, then type the next command: the crawl runs at 0.025 rad/s the
        whole time, so recognition has to survive a few seconds of it or the fix
        is theoretical."""
        drifted = np.asarray(OP.CHEST_HOME_JOINTS, float).copy()
        drifted[3] -= 0.025 * 8.0                       # 8 s of crawl
        why, _ = R.posture_gate(
            _tel(drifted, OP.mirror_arm(OP.CHEST_HOME_JOINTS, "right")))
        self.assertIsNone(why)

    def test_the_tolerance_boundary_is_the_operator_constant(self):
        eps = 1e-6
        for delta, passes in ((OP.FRONT_HOME_TOL_RAD - eps, True),
                              (OP.FRONT_HOME_TOL_RAD + 1e-3, False)):
            with self.subTest(delta=delta):
                q = self.left.copy()
                q[3] += delta
                why, _ = R.posture_gate(_tel(q, self.right))
                self.assertEqual(why is None, passes)

    def test_the_worst_arm_is_the_one_reported(self):
        """Both arms are checked; naming the wrong one sends the operator to the
        wrong side of the robot."""
        bad = np.asarray(OP.mirror_arm(OP.POWERON_JOINTS, "right"), float)
        bad[3] += 0.8
        why, _ = R.posture_gate(_tel(self.left, bad))
        self.assertIn("right arm", why)

    def test_missing_telemetry_is_refused_not_treated_as_home(self):
        why, err = R.posture_gate({"projected_gravity": [0, 0, -1]})
        self.assertIsNotNone(why)
        self.assertEqual(err, float("inf"))


if __name__ == "__main__":
    unittest.main()
