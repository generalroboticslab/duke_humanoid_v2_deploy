"""Every retreat must say WHICH rule fired and WHO tripped it.

WHY THIS EXISTS (user 2026-08-05). After the false-conviction fixes (the
executor dither acquittal, drift demoted to a warning) the robot still retreats
sometimes, and the log said only what happened, in one sentence, at 20 Hz —
which is not something a fix can be aimed at. Now every abort path in
stream_route/mpc_track returns a CODED verdict (T01-T14 runtime, S01-S05
session setup) carrying the culprit — the arm, the joint, the cube, the geom
pair — and the retreat prints it as a banner BEFORE the arms walk back.

The AST guard below is the part that keeps this true: a new abort path added to
either function without a code would silently reopen the hole this closes, and
no runtime test would notice, because an uncoded reason is still a perfectly
good string.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

CUBE = "grasp_cube_60mm"
HERE = np.array([0.403, 0.103, 0.083])


# --------------------------------------------------------------------------
# stream_route fixtures (the same shape test_drift_abort_evidence uses)
# --------------------------------------------------------------------------
class _Rig:
    def __init__(self):
        self.cubes = {CUBE: None}
        self._pos = HERE

    def latest(self, key, now):
        return self._pos

    def median(self, key, now):
        return self._pos

    def last_inliers(self, key):
        return 3

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out


class _Det:
    def poll(self, rig):
        pass


class _Tel:
    """Returns the same telemetry every tick; None means the link went stale."""

    def __init__(self, item):
        self.item = item

    def fresh(self):
        return self.item


def _telemetry(fault=(False, False), enc=None):
    """`enc` is {joint: rad}; _arm_enc reads the arms out of joint_pos[13:27]."""
    jp = [0.0] * 31
    for i, name in enumerate(R.ARM_JOINTS_L):
        jp[13 + i] = float((enc or {}).get(name, 0.0))
    for i, name in enumerate(R.ARM_JOINTS_R):
        jp[20 + i] = float((enc or {}).get(name, 0.0))
    return {"joint_pos": jp, "projected_gravity": [0.0, 0.0, -1.0],
            "arm_fault": list(fault)}


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)


class _Bundle:
    goal = "reach"
    active_sides = ("left",)
    horizon = 2

    def __init__(self, q=None, duration_s=0.4):
        self.duration_s = duration_s
        self.joint_names = tuple(R.ARM_JOINTS_L) + tuple(R.ARM_JOINTS_R)
        self.controlled_joints = tuple(R.ARM_JOINTS_L)
        self.q_enc = np.zeros((2, len(self.joint_names))) if q is None else q

    def q_at(self, t):
        return self.q_enc[0]


def _stream(tel):
    return R.stream_route(_Bundle(), _Pub(), tel, _Det(), _Rig(),
                          {CUBE: HERE}, 0.3)


@unittest.skipIf(R is None, _SKIP)
class RegistryTests(unittest.TestCase):

    def test_the_runtime_triggers_are_all_registered(self):
        """T01-T14 are the operator's own enumeration of the retreat rules;
        the table must stay complete, or a banner prints a code with no
        meaning next to it."""
        for i in range(1, 15):
            self.assertIn(f"T{i:02d}", R.RETREAT_TRIGGERS)
        for i in range(1, 6):
            self.assertIn(f"S{i:02d}", R.RETREAT_TRIGGERS)

    def test_trigger_names_are_unique(self):
        names = [n for n, _ in R.RETREAT_TRIGGERS.values()]
        self.assertEqual(len(names), len(set(names)),
                         "two rules share a name — greps become ambiguous")

    def test_an_unregistered_code_is_refused(self):
        with self.assertRaises(AssertionError):
            R.Abort("T99", "nowhere", "nobody", "typo")

    def test_the_verdict_is_still_a_plain_reason_string(self):
        """Every caller composes, prints and substring-matches on it. The
        code rides in the text so it survives into the round-level line."""
        a = R.Abort("T12", "MPC track", "right arm / right_elbow_joint",
                    "right arm is NOT executing")
        self.assertIsInstance(a, str)
        self.assertIn("right arm is NOT executing", a)
        self.assertIn("T12", a)
        self.assertIn("MPC-EXEC", a)
        self.assertIn("still holding", f"{a} — still holding")


@unittest.skipIf(R is None, _SKIP)
class CodedPathTests(unittest.TestCase):
    """No abort path may leave either function uncoded. Checked statically,
    because the only way to notice at runtime is to hit that path on hardware
    — which is exactly the moment the code was needed."""

    def _uncoded_returns(self, func):
        """Every `return "..."` / `return f"..."` (or such a value as the first
        element of a returned tuple) is a reason string that reached the
        operator without a trigger code."""
        bad = []
        for node in ast.walk(ast.parse(inspect.getsource(func))):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            value = node.value
            if isinstance(value, ast.Tuple) and value.elts:
                value = value.elts[0]                   # (why, played_s)
            if isinstance(value, ast.JoinedStr) or (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                bad.append(ast.unparse(node))
        return bad

    def test_stream_route_returns_only_coded_aborts(self):
        self.assertEqual(self._uncoded_returns(R.stream_route), [])

    def test_mpc_track_returns_only_coded_aborts(self):
        self.assertEqual(self._uncoded_returns(R.mpc_track), [])


@unittest.skipIf(R is None, _SKIP)
class StreamVerdictTests(unittest.TestCase):

    def test_a_stale_link_is_T01(self):
        why, _ = _stream(_Tel(None))
        self.assertEqual(why.code, "T01")
        self.assertEqual(why.culprit, "telemetry link")

    def test_an_arm_fault_names_the_faulted_SIDE(self):
        """The whole point of the culprit field: 'ARM FAULT' alone sends an
        operator to both harnesses; the right arm's can21 is the known-bad
        one and the log must say when it is the one that dropped."""
        why, _ = _stream(_Tel(_telemetry(fault=(False, True))))
        self.assertEqual(why.code, "T02")
        self.assertIn("right", why.culprit)
        self.assertNotIn("left", why.culprit)

    def test_the_follow_watchdog_names_the_LAGGING_JOINT(self):
        # The route sits at zero while the measured arm is parked 2.0 rad away
        # at one joint: far past STREAM_FOLLOW_LAG_RAD, held every tick.
        lag_joint = R.ARM_JOINTS_L[1]
        tel = _Tel(_telemetry(enc={lag_joint: 2.0}))
        # Long enough to outlast STREAM_FOLLOW_LAG_TICKS: a two-waypoint
        # 0.4 s route ends before the watchdog can convict.
        why, _ = R.stream_route(_Bundle(duration_s=3.0), _Pub(), tel, _Det(),
                                _Rig(), {CUBE: HERE}, 0.3)
        self.assertEqual(why.code, "T03")
        self.assertIn(lag_joint, why.culprit)
        self.assertIn("left arm", why.culprit)


@unittest.skipIf(R is None, _SKIP)
class BannerTests(unittest.TestCase):

    def setUp(self):
        R.RETREAT_TALLY.clear()

    def tearDown(self):
        R.RETREAT_TALLY.clear()

    def _banner(self, why, note=""):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.print_retreat_verdict(why, note)
        return out.getvalue()

    def test_the_banner_carries_code_rule_culprit_and_detail(self):
        said = self._banner(R.Abort(
            "T12", "MPC track", "right arm / right_elbow_joint",
            "right arm is NOT executing: moved 0.31 of 2.10 rad"))
        self.assertIn("T12", said)
        self.assertIn("MPC-EXEC", said)
        self.assertIn("right arm / right_elbow_joint", said)
        self.assertIn("moved 0.31 of 2.10 rad", said)
        self.assertIn("MPC track", said)

    def test_the_banner_states_that_BOTH_arms_retreat(self):
        """The user's standing question ('does one side's trigger retreat
        both?') is answered in the log itself, every single time."""
        said = self._banner(R.Abort("T07", "MPC track", "left arm / cube_a",
                                    "left the envelope"))
        self.assertIn("BOTH arms", said)

    def test_an_uncoded_reason_still_prints_and_is_marked(self):
        said = self._banner("some path nobody coded")
        self.assertIn("UNCODED", said)
        self.assertIn("some path nobody coded", said)

    def test_repeats_are_counted_and_summarised(self):
        for _ in range(3):
            self._banner(R.Abort("T12", "MPC track", "right arm / j", "x"))
        self._banner(R.Abort("T07", "MPC track", "left arm / c", "y"))
        self.assertEqual(R.RETREAT_TALLY, {"T12": 3, "T07": 1})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.print_retreat_tally()
        said = out.getvalue()
        self.assertIn("T12 MPC-EXEC", said)
        self.assertIn("x3", said)
        self.assertIn("4 total", said)

    def test_a_clean_run_prints_no_tally(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.print_retreat_tally()
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
