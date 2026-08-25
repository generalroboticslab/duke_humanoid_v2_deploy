"""Two failures the retract could not survive, and both were self-inflicted.

WHY THIS EXISTS. After the first successful hardware grasp (2026-07-30) the
retract still had two ways to end badly, and the user found them by asking what
happens when a tag is occluded mid-mission.

1. THE TABLE HAS TO BE RE-SEEN. build_scene runs again for the retract and
   refused if a declared table was not visible right then — at the one moment it
   is hardest to see, with the arm extended into the workspace and the gaze
   locked on the cube. StaticPoseTrack's window is 2 s and its own first line is
   "a body that does not move". So the mission died because our own arm blocked
   the view of a table we had measured, certified at 1.2 deg spread, and planned
   against seconds earlier. The 07-30 run survived it by luck: the tags happened
   to stay in frame.

   The guarantee that refusal protects is real — a lost table must not silently
   become an EMPTY world — and the latch keeps it. Until 2026-08-13 that meant
   the slab stayed in the scene as a hard obstacle; since the user's order that
   day ("THE TABLE IS A HEIGHT SENSOR, NOT AN OBSTACLE — IN THE PLANNER TOO")
   a registered table never enters the cuboids at all, and what the remembered
   pose keeps serving is the TABLE-RESTING Z PRIOR (grasp z = surface + half
   cube, rig._z_prior_keys naming the targets that carry it). Only the
   measurement is old, and every use says so.

2. "ARM HOLDS AT GRASP" DID NOT HOLD. Those messages ended in refuse(), which
   returns immediately: publishing stops, the arm-silence failsafe takes the
   arm, and it crawls home still gripping the cube. The text promised a
   stationary arm to an operator who was about to walk up to it.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

TABLE = "lab_table_a"
CUBE = "grasp_cube_60mm"
TOP = np.array([0.675, 0.077, 0.051])
Q_FLAT = np.array([1.0, 0.0, 0.0, 0.0])


class _Args:
    def __init__(self):
        self.tables = (TABLE,)
        self.table = ()
        self.cubes = (CUBE,)
        self.no_table_world = False


class _Rig:
    """Table visible or not, on demand; the cube is always there."""

    def __init__(self, table_visible=True, spread=1.2):
        self.table_visible = table_visible
        self.spread = spread

    def table_pose(self, key, now):
        if not self.table_visible:
            return None
        return TOP.copy(), Q_FLAT.copy(), self.spread

    def median(self, key, now):
        return np.array([0.403, 0.103, 0.083])

    def latest(self, key, now):
        return self.median(key, now)


@unittest.skipIf(R is None, _SKIP)
class TableLatchTests(unittest.TestCase):

    def test_a_table_seen_once_survives_going_out_of_view(self):
        """THE fix. Reach sees it, retract does not, and the table is still
        SERVED — from the pose we measured and certified. Until 2026-08-13
        "served" meant a slab in the retract scene; since the table became a
        height sensor (no registered table ever enters the cuboids) it means
        the remembered pose keeps feeding the z prior, so the pin is: the
        seen build writes the latch and corrects the 60 mm cube's grasp z to
        surface + 30 mm; the blind RETRACT build (cube in hand, excluded) does
        not refuse and says the measurement is old; and a blind build with
        the cube still a target lands the SAME corrected z from memory."""
        latch: dict = {}
        rig_seen = _Rig(True)
        with contextlib.redirect_stdout(io.StringIO()):
            seen, t_seen = R.build_scene(_Args(), rig_seen, (CUBE,),
                                         latched=latch)
        self.assertNotIn(TABLE, seen, "the table re-entered the planner world")
        self.assertIn(TABLE, latch, "the seen table was not latched")
        resting_z = TOP[2] + 0.030                 # 60 mm cube, centre
        self.assertEqual(rig_seen._z_prior_keys, {CUBE})
        self.assertAlmostEqual(float(t_seen[CUBE][2]), resting_z, places=9)

        rig_blind = _Rig(False)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            blind, _ = R.build_scene(_Args(), rig_blind, (CUBE,),
                                     exclude=(CUBE,), latched=latch)
        self.assertNotIn(TABLE, blind)
        self.assertIn(TABLE, latch, "the retract must not consume the latch")
        self.assertIn("not visible right now", out.getvalue())
        self.assertIn("only the measurement is old", out.getvalue())
        self.assertIn("z prior", out.getvalue(),
                      "the fallback must say what the old pose still does")

        with contextlib.redirect_stdout(io.StringIO()):
            blind2, t_blind = R.build_scene(_Args(), rig_blind, (CUBE,),
                                            latched=latch)
        self.assertEqual(rig_blind._z_prior_keys, {CUBE},
                         "the remembered pose stopped feeding the z prior")
        self.assertAlmostEqual(float(t_blind[CUBE][2]), resting_z, places=9)
        self.assertAlmostEqual(blind2[CUBE]["pose"][2], seen[CUBE]["pose"][2],
                               places=9, msg="the cube obstacle z drifted "
                                             "between the seen and blind builds")

    def test_the_age_of_a_remembered_pose_is_printed(self):
        """A silent fallback is how a stale world becomes invisible. The
        operator has to be able to see that the number is old."""
        latch = {TABLE: (TOP.copy(), Q_FLAT.copy(), 1.2, 0.0)}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.build_scene(_Args(), _Rig(False), (CUBE,), latched=latch)
        self.assertRegex(out.getvalue(), r"measured \d+\.\d+s ago")

    def test_a_table_never_seen_at_all_still_refuses(self):
        """The latch must not turn 'we never had a world' into silence — that
        is the false declaration --no-table-world exists to make impossible."""
        with self.assertRaises(R.TableNotSeen):
            R.build_scene(_Args(), _Rig(False), (CUBE,), latched={})

    def test_without_a_latch_the_old_refusal_is_unchanged(self):
        """Every other caller — the tests above included — must see exactly the
        behaviour it always saw when it passes no cache."""
        with self.assertRaises(R.TableNotSeen):
            R.build_scene(_Args(), _Rig(False), (CUBE,))

    def test_a_visible_table_is_always_preferred_over_the_memory(self):
        """The latch is a fallback, not a cache to be served from. A table that
        HAS been re-measured must be used, or a base that shifted would be
        planned against yesterday's geometry. Until 2026-08-13 "used" showed
        as the slab centre following the live top (top - 0.015 under the
        08-11 re-frame); since the table is a height sensor it shows as the
        z prior following the LIVE surface — 40 mm higher than the memory —
        and the latch being overwritten with the new measurement."""
        moved = TOP + np.array([0.0, 0.0, 0.04])
        latch = {TABLE: (TOP.copy(), Q_FLAT.copy(), 1.2, 0.0)}

        class _Moved(_Rig):
            def table_pose(self, key, now):
                return moved.copy(), Q_FLAT.copy(), 1.2

        rig = _Moved(True)
        with contextlib.redirect_stdout(io.StringIO()):
            scene, targets = R.build_scene(_Args(), rig, (CUBE,),
                                           latched=latch)
        self.assertNotIn(TABLE, scene)
        self.assertEqual(rig._z_prior_keys, {CUBE})
        # 60 mm cube resting on the LIVE surface, not the remembered one.
        self.assertAlmostEqual(float(targets[CUBE][2]), moved[2] + 0.030,
                               places=9)
        self.assertNotAlmostEqual(float(targets[CUBE][2]), TOP[2] + 0.030,
                                  places=6, msg="the prior came from memory")
        self.assertAlmostEqual(latch[TABLE][0][2], moved[2], places=9)

    def test_a_junk_live_fit_is_refused_rather_than_silently_latched(self):
        """A visible-but-wobbling table must still raise: falling back would be
        defensible, but silently ACCEPTING junk geometry is the 78.5-deg
        incident, and the latch must not become a way to launder it."""
        with self.assertRaises(R.TablePoseUnstable):
            R.build_scene(_Args(), _Rig(True, spread=78.5), (CUBE,), latched={})


class _Tel:
    def __init__(self, fault=False):
        self.fault = fault
        self.calls = 0

    def fresh(self):
        self.calls += 1
        return {"joint_pos": [0.1] * 31, "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, self.fault]}


class _Det:
    def poll(self, rig):
        pass


class _HoldRig:
    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out


class _Pub:
    def __init__(self, interrupt_after=None):
        self.sent = []
        self.interrupt_after = interrupt_after

    def publish(self, packet):
        self.sent.append(packet)
        if self.interrupt_after and len(self.sent) >= self.interrupt_after:
            raise KeyboardInterrupt


@unittest.skipIf(R is None, _SKIP)
class RefuseHoldingTests(unittest.TestCase):

    def test_it_keeps_publishing_instead_of_going_silent(self):
        """THE fix. refuse() returned at once and the failsafe crawled the arm
        home still gripping; the message said the arm was holding."""
        pub = _Pub(interrupt_after=10)
        with contextlib.redirect_stdout(io.StringIO()):
            code = R.refuse_holding("no retract route", 5, pub, _Tel(), _Det(),
                                    _HoldRig(), 0.3)
        self.assertEqual(code, 5)
        self.assertGreaterEqual(len(pub.sent), 10)
        for packet in pub.sent:
            self.assertIn("arm_targets", packet)

    def test_the_held_posture_is_a_snapshot_of_the_measurement(self):
        pub = _Pub(interrupt_after=6)
        with contextlib.redirect_stdout(io.StringIO()):
            R.refuse_holding("x", 5, pub, _Tel(), _Det(), _HoldRig(), 0.3)
        first = pub.sent[0]["arm_targets"]["left"]["joint_pos"]
        self.assertEqual(first, [0.1] * len(R.ARM_JOINTS_L))
        for packet in pub.sent[1:]:
            self.assertEqual(packet["arm_targets"]["left"]["joint_pos"], first)

    def test_an_arm_fault_ends_the_hold(self):
        """A hold that ignores a fault is not a hold."""
        pub = _Pub()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = R.refuse_holding("x", 5, pub, _Tel(fault=True), _Det(),
                                    _HoldRig(), 0.3)
        self.assertEqual(code, 5)
        self.assertIn("ARM FAULT", out.getvalue())

    def test_it_says_the_hand_is_still_closed(self):
        """The operator walking up to the robot needs to know the gripper is
        loaded before they reach into it."""
        pub = _Pub(interrupt_after=3)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.refuse_holding("x", 5, pub, _Tel(), _Det(), _HoldRig(), 0.3)
        said = out.getvalue()
        self.assertIn("still closed on the cube", said)
        self.assertIn("Ctrl+C", said)

    def test_no_encoders_means_it_says_so_rather_than_pretending(self):
        """Nothing to snapshot, so nothing to hold. Claiming a hold here would
        be the same lie in a new place."""
        class _NoEnc:
            def fresh(self):
                return {"projected_gravity": [0.0, 0.0, -1.0]}

        pub = _Pub()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = R.refuse_holding("x", 5, pub, _NoEnc(), _Det(), _HoldRig(),
                                    0.3)
        self.assertEqual(code, 5)
        self.assertEqual(pub.sent, [])
        self.assertIn("no encoders to hold at", out.getvalue())


if __name__ == "__main__":
    unittest.main()
