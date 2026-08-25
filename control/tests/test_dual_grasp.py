"""Two cameras, two cubes, two hands, one mission.

WHAT CHANGED. The tool was single-target from end to end: the selection loop
broke on the first reachable cube, build_scene sent one target, do_grasp closed
one hand, the drift monitor watched one cube and the retract excluded one. The
layers UNDER that were already dual-ready and always had been — Gate A iterates
bundle.reaches, streaming iterates active_sides, and a two-entry assignment is
upstream's original server design — so this is the mission layer catching up.

THE RULE THAT DECIDES THE REST: half a dual grasp is not a state this tool
models. The retract plans for what the reach took, so a mission that started on
one cube and found the other unreachable would leave a hand closed on nothing
and a scene that disagrees with the hardware. Everything below follows from
refusing to enter that state: wait for ALL cubes, refuse if any is lost between
selection and scene-build, and fail the grasp if any hand comes back empty.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

A, B = "grasp_cube_40mm", "grasp_cube_60mm"
POS = {A: np.array([0.40, 0.22, 0.06]), B: np.array([0.40, -0.20, 0.08])}


class _Args:
    def __init__(self, cubes=(A, B)):
        self.tables = ()
        self.table = ()
        self.no_table_world = True
        self.cubes = tuple(cubes)


class _Rig:
    def __init__(self, seen=(A, B)):
        self.seen = set(seen)

    def table_pose(self, key, now):
        return None

    def median(self, key, now):
        return POS[key].copy() if key in self.seen else None

    def latest(self, key, now):
        return self.median(key, now)


@unittest.skipIf(R is None, _SKIP)
class SceneTests(unittest.TestCase):

    def test_both_cubes_are_targets_and_both_are_obstacles(self):
        """The upstream contract keeps the targets in the obstacle set — cuRobo's
        goalset handles the approach — so a two-cube reach must plan around the
        cube the OTHER hand is going for, not just around clutter."""
        scene, targets = R.build_scene(_Args(), _Rig(), (A, B))
        self.assertEqual(set(targets), {A, B})
        self.assertLessEqual({A, B}, set(scene))

    def test_each_cube_gets_its_own_size(self):
        """40 and 60 mm in one scene: a shared constant would put one obstacle
        10 mm wrong in every direction."""
        scene, _ = R.build_scene(_Args(), _Rig(), (A, B))
        self.assertAlmostEqual(scene[A]["dims"][0], 0.040, places=9)
        self.assertAlmostEqual(scene[B]["dims"][0], 0.060, places=9)

    def test_a_target_lost_between_selection_and_the_scene_refuses(self):
        """Dropping it would send the planner ONE target and quietly turn a
        two-arm mission into a one-arm one — which then reads as a successful
        run that grasped half of what was asked for."""
        with self.assertRaises(R.TargetLost) as ctx:
            R.build_scene(_Args(), _Rig(seen=(A,)), (A, B))
        self.assertEqual(ctx.exception.name, B)

    def test_the_retract_drops_every_grasped_cube(self):
        """Both ride in hands now; their last SEEN positions are where they
        used to be, and planning around those would route the arms around
        ghosts."""
        scene, targets = R.build_scene(_Args(), _Rig(), (), exclude=(A, B))
        self.assertEqual(targets, {})
        self.assertNotIn(A, scene)
        self.assertNotIn(B, scene)

    def test_a_bare_string_is_rejected_rather_than_iterated(self):
        """A str would iterate as characters and silently build a scene with no
        cubes in it — the exact shape of bug that looks like it worked."""
        for bad in (dict(target_keys=A), dict(exclude=A)):
            with self.subTest(**bad):
                kw = {"target_keys": (A,), "exclude": ()} | bad
                with self.assertRaises(TypeError):
                    R.build_scene(_Args(), _Rig(), **kw)


@unittest.skipIf(R is None, _SKIP)
class AssignmentTests(unittest.TestCase):

    def test_two_cubes_offer_both_pairings(self):
        """The whole search space is two entries, so it is enumerated rather
        than guessed at. A cube can be unreachable-with-margin for the arm on
        its side and easy for the other."""
        got = R.forced_assignments((A, B))
        self.assertEqual(got, [{"L": A, "R": B}, {"L": B, "R": A}])

    def test_one_cube_still_offers_each_arm(self):
        self.assertEqual(R.forced_assignments((A,)), [{"L": A}, {"R": A}])

    def test_every_pairing_uses_each_arm_once_and_each_cube_once(self):
        for pairing in R.forced_assignments((A, B)):
            self.assertEqual(sorted(pairing), ["L", "R"])
            self.assertEqual(sorted(pairing.values()), sorted([A, B]))


@unittest.skipIf(R is None, _SKIP)
class ArgTests(unittest.TestCase):

    def test_one_or_two_cubes_are_accepted(self):
        for cubes in ((A,), (A, B)):
            with self.subTest(cubes=cubes):
                self.assertIsNone(R._check_world(_Args(cubes=cubes)))

    def test_three_cubes_are_refused_because_there_are_two_arms(self):
        why = R._check_world(_Args(cubes=(A, B, "grasp_cube_80mm"))) or ""
        self.assertIn("1 or 2", why)

    def test_no_cubes_at_all_is_refused(self):
        self.assertIn("1 or 2", R._check_world(_Args(cubes=())) or "")

    def test_the_same_cube_twice_is_refused(self):
        """The assignment map is keyed by SIDE, so a repeat would not error —
        one of the two would silently vanish and the mission would look like it
        had planned for both."""
        self.assertIn("repeat", R._check_world(_Args(cubes=(A, A))) or "")


class _Tel:
    def fresh(self):
        return {"joint_pos": [0.0] * 31, "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, False], "ee": self.ee}

    ee: dict = {}


class _Det:
    def poll(self, rig):
        pass


class _HoldRig:
    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out


class _Pub:
    def __init__(self, tel, verdicts):
        self.tel = tel
        self.verdicts = verdicts     # {side: True/False} applied once commanded
        self.sent = []
        self.actions = []

    def publish(self, packet):
        self.sent.append(packet)
        act = packet.get("ee_action")
        if act:
            self.actions.append((act["side"], act["action_id"]))
            side = act["side"]
            if side in self.verdicts:
                self.tel.ee = dict(self.tel.ee)
                self.tel.ee[side] = {"grasp_action_id": act["action_id"],
                                     "grasp_detected": self.verdicts[side]}


class _Bundle:
    goal = "reach"
    active_sides = ("left", "right")

    def __init__(self):
        self.joint_names = tuple(R.ARM_JOINTS_L) + tuple(R.ARM_JOINTS_R)
        self.q_enc = np.zeros((2, len(self.joint_names)))


@unittest.skipIf(R is None, _SKIP)
class DoGraspTests(unittest.TestCase):

    def _run(self, verdicts):
        tel = _Tel()
        tel.ee = {}
        pub = _Pub(tel, verdicts)
        why, empty = R.do_grasp(_Bundle(), pub, tel, _Det(), _HoldRig(),
                                ("left", "right"), 0.3)
        self.last_empty = empty
        return why, pub

    def test_both_hands_close_and_both_verdicts_are_required(self):
        why, pub = self._run({"left": True, "right": True})
        self.assertIsNone(why, f"a clean dual grasp was refused: {why}")
        self.assertEqual({s for s, _ in pub.actions}, {"left", "right"})

    def test_the_two_commands_interleave_rather_than_running_in_sequence(self):
        """A 9874 packet carries at most ONE ee_action, so they cannot share a
        tick. Blocking on the left for its whole repeat window would start the
        right ~1.5 s later and leave the left un-repeated while it ran."""
        _, pub = self._run({"left": True, "right": True})
        first_two = [s for s, _ in pub.actions[:2]]
        self.assertEqual(set(first_two), {"left", "right"},
                         f"the sides did not interleave: {pub.actions[:4]}")

    def test_each_side_carries_its_own_id(self):
        """The verdict is correlated BY id; two hands answering with the same
        id is a correlation that proves nothing."""
        _, pub = self._run({"left": True, "right": True})
        ids = {s: i for s, i in pub.actions}
        self.assertNotEqual(ids["left"], ids["right"])

    def test_an_empty_hand_is_reported_by_name_and_not_confused_with_a_fault(self):
        """do_grasp REPORTS which hands came up empty; it no longer decides what
        that means. The 07-30 audit found the old conflation: EMPTY was mapped
        straight to FATAL, and a mission with an EMPTY hand parked forever in
        refuse_holding while the cube sat on the table still being seen. Whether
        an empty pinch is retryable depends on whether the OTHER hand is loaded,
        which is the caller's question."""
        why, _ = self._run({"left": True, "right": False})
        self.assertIsNotNone(why)
        self.assertIn("right", why)
        self.assertIn("EMPTY", why)
        self.assertEqual(self.last_empty, ("right",))

    def test_both_hands_empty_is_reported_as_both(self):
        """The retryable case: nothing is held, so the class-1 contract applies
        and the caller may reopen and try again."""
        self._run({"left": False, "right": False})
        self.assertEqual(self.last_empty, ("left", "right"))

    def test_an_indeterminate_result_reports_no_empty_sides(self):
        """A lost verdict must NEVER look retryable: we do not know what the
        fingers hold, and reopening a maybe-held cube drops it."""
        tel = _Tel()
        tel.ee = {}
        pub = _Pub(tel, {})
        keep, R.GRASP_RESULT_TIMEOUT_S = R.GRASP_RESULT_TIMEOUT_S, 0.4
        try:
            why, empty = R.do_grasp(_Bundle(), pub, tel, _Det(), _HoldRig(),
                                    ("left", "right"), 0.3)
        finally:
            R.GRASP_RESULT_TIMEOUT_S = keep
        self.assertIn("FAIL CLOSED", why)
        self.assertEqual(empty, (), "an indeterminate result must not be "
                                    "offered to the retry path")

    def test_a_verdict_for_the_wrong_id_is_not_accepted(self):
        """A stale marker from a previous run answering for this one is the
        fail-open the id contract exists to prevent."""
        tel = _Tel()
        tel.ee = {"left": {"grasp_action_id": 1, "grasp_detected": True},
                  "right": {"grasp_action_id": 1, "grasp_detected": True}}
        pub = _Pub(tel, {})                       # never posts a matching id
        keep, R.GRASP_RESULT_TIMEOUT_S = R.GRASP_RESULT_TIMEOUT_S, 0.4
        try:
            why, _ = R.do_grasp(_Bundle(), pub, tel, _Det(), _HoldRig(),
                                ("left", "right"), 0.3)
        finally:
            R.GRASP_RESULT_TIMEOUT_S = keep
        self.assertIsNotNone(why)
        self.assertIn("FAIL CLOSED", why)

    def test_the_timeout_names_the_hand_that_never_answered(self):
        """One side answering and the other not is the case worth naming — it
        sends the operator to the right side of the robot."""
        tel = _Tel()
        tel.ee = {}
        pub = _Pub(tel, {"left": True})
        keep, R.GRASP_RESULT_TIMEOUT_S = R.GRASP_RESULT_TIMEOUT_S, 0.4
        try:
            why, _ = R.do_grasp(_Bundle(), pub, tel, _Det(), _HoldRig(),
                                ("left", "right"), 0.3)
        finally:
            R.GRASP_RESULT_TIMEOUT_S = keep
        self.assertIn("right", why or "")
        self.assertNotIn("left", (why or "").split("in ")[0])


if __name__ == "__main__":
    unittest.main()
