"""No wait may publish without arm content — gaze does not hold the arms.

WHY THIS EXISTS. Hardware 2026-07-31. A dry run printed all four gates green,
drew both green paths in the monitor, and then the arms lifted on their own and
drifted. The --execute run that followed refused its own posture gate:

    [gate] WAITING — left arm 23 deg from every known home (nearest: front)
           — reaching from an unknown posture crosses configuration basins

Nothing had commanded that motion. The cause is one line of contract, in
real_env's _update_arm_targets: arm freshness tracks ARM content ONLY —

    if data is not None and any(k in data for k in ("arm_targets", "kp", "kd")):

— so a 9874 packet carrying only gaze_targets leaves the arm timer expired.
_ARM_CMD_TIMEOUT is 0.5 s and _ARM_RETURN_RATE is 0.025 rad/s, so every wait
longer than half a second that publishes gaze alone hands the arms to the
failsafe for its whole duration. 23 deg is 0.4 rad, which is 16 s of crawl: the
6 s noise-floor watch plus the time it takes an operator to read the message
above a BLOCKING input().

pump_during fixed exactly this for the solve on 07-30 and its docstring spells
the mechanism out. The waits kept the bug because each one had a comment saying
the quiet part out loud — "gaze-only: arm timer untouched" — and read as
deliberate. It was not: nothing in this tool wants the failsafe to own the arms
mid-mission.

Six sites were silent, in order of exposure:

    the standing scan          DET_WAIT_S        120 s   crawls past default_pose
    the journey survey         DET_WAIT_S        120 s   same
    the journey re-acquire     JOURNEY_ACQUIRE_S  40 s   56 deg
    the table wait             TABLE_WAIT_S       20 s   28 deg
    zero_grippers              GRIPPER_ZERO_S     12 s   16 deg
    the dry-run watch + wait   OBSERVE_S + input  UNBOUNDED

The 120 s ones are the worst and the least visible: --chest-home is ordered
BEFORE the scan precisely so the reach departs from the posture it was planned
from, and 120 s of crawl is more than enough to undo the raise completely. A
scan that ended early left the arms frozen partway there, in a posture matching
no home at all — which is the refusal above.

These tests pin the PROPERTY (every packet carries arm content, and the hold is
a snapshot) rather than any one loop, plus a structural check that the shape
which caused it cannot come back.
"""

from __future__ import annotations

import ast
import pathlib
import time
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R
    from humanoid_curobo_client import ARM_JOINTS_ALL

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

ENC = None if R is None else {n: 0.1 * i for i, n in enumerate(ARM_JOINTS_ALL)}
SRC = pathlib.Path(__file__).resolve().parent.parent / "humanoid_curobo_reach.py"


class _Tel:
    def __init__(self, ee=None):
        self.ee = ee if ee is not None else {
            "left": {"ee_alive": True}, "right": {}}

    def fresh(self):
        return {"joint_pos": [0.1 * i for i in range(31)],
                "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, False], "ee": self.ee}


class _Det:
    def __init__(self):
        self.polls = 0

    def poll(self, rig):
        self.polls += 1


class _Rig:
    """Passes the packet through and stamps gaze, like the real one."""

    def __init__(self):
        # zero_grippers parks both eyes for the zeroing (08-10 FSM order)
        # through the retire set, and restores it on exit.
        self._cam_retired = set()

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0, 0.0, 0.0, 0.0]
        return out


class _Pub:
    def __init__(self):
        self.sent = []
        self.stamps = []

    def publish(self, packet):
        self.sent.append(packet)
        self.stamps.append(time.monotonic())


class _Stdin:
    """Readable only after `ready_after` seconds, like an operator pressing
    Enter partway through the wait."""

    def __init__(self, ready_after):
        self.t0 = time.monotonic()
        self.ready_after = ready_after

    def ready(self):
        return time.monotonic() - self.t0 >= self.ready_after

    def readline(self):
        return "\n"


class _Select:
    def select(self, rlist, _w, _x, _timeout):
        return ([f for f in rlist if f.ready()], [], [])


def _patch_stdin(test, ready_after):
    """Point hold_until_enter's select/stdin at the fake, restore afterwards."""
    keep_sel, keep_sys = R.select, R.sys

    class _Sys:
        stdin = _Stdin(ready_after)

    R.select, R.sys = _Select(), _Sys()
    test.addCleanup(lambda: setattr(R, "select", keep_sel))
    test.addCleanup(lambda: setattr(R, "sys", keep_sys))


@unittest.skipIf(R is None, _SKIP)
class HoldUntilEnterTests(unittest.TestCase):

    def setUp(self):
        self.pub, self.tel, self.det, self.rig = _Pub(), _Tel(), _Det(), _Rig()

    def _run(self, ready_after=0.8, enc=ENC):
        _patch_stdin(self, ready_after)
        R.hold_until_enter(self.pub, self.tel, self.det, self.rig, enc, 0.3)

    def test_every_packet_carries_arm_targets(self):
        """THE test. The old code called input() and published NOTHING; a
        keepalive of gaze alone would look just as busy here and still let the
        arms crawl on hardware, so this asserts the ARM key specifically."""
        self._run()
        self.assertTrue(self.pub.sent)
        for packet in self.pub.sent:
            self.assertIn("arm_targets", packet)
            self.assertEqual(set(packet["arm_targets"]), {"left", "right"})

    def test_it_never_goes_silent_longer_than_the_failsafe_timeout(self):
        self._run()
        gaps = np.diff(np.asarray(self.pub.stamps))
        self.assertTrue(gaps.size, "only one packet in the whole wait")
        self.assertLess(float(gaps.max()), 0.5,
                        f"silent for {gaps.max():.2f}s — _ARM_CMD_TIMEOUT "
                        f"is 0.5s")

    def test_the_wait_actually_lasts_until_enter(self):
        """A hold that returns immediately would pass the two tests above and
        still leave the failsafe the whole time the operator reads the screen.
        The point of the fix is that the WAIT is held, not that a packet or two
        gets sent."""
        t0 = time.monotonic()
        self._run(ready_after=0.8)
        self.assertGreater(time.monotonic() - t0, 0.7)
        self.assertGreater(len(self.pub.sent), 0.7 / R.PUB_DT * 0.6)

    def test_the_posture_is_a_snapshot_not_a_follower(self):
        """Re-reading the encoders each tick would let gravity sag walk the arm
        downhill one packet at a time, every packet looking like a no-op."""
        self._run()
        first = self.pub.sent[0]["arm_targets"]["left"]["joint_pos"]
        for packet in self.pub.sent[1:]:
            self.assertEqual(packet["arm_targets"]["left"]["joint_pos"], first)

    def test_the_posture_is_the_one_it_was_handed(self):
        self._run()
        slot = self.pub.sent[0]["arm_targets"]["right"]["joint_pos"]
        self.assertEqual(slot, [ENC[n] for n in R.ARM_JOINTS_R])

    def test_detections_keep_flowing(self):
        """The route preview and the cube history stay live while the operator
        looks at them; a frozen history is what made a stationary cube read as
        drift on 07-30."""
        self._run()
        self.assertGreater(self.det.polls, 5)


@unittest.skipIf(R is None, _SKIP)
class ZeroGrippersHoldTests(unittest.TestCase):
    """Calibration happens BEFORE the chest-home raise, so a crawl here starts
    the mission from a posture nobody chose."""

    def setUp(self):
        self.pub, self.tel, self.rig = _Pub(), _Tel(), _Rig()
        # BOTH constants, and the repeat one is not optional. The ee_action
        # burst runs EE_ACTION_REPEAT_TICKS x 2 sides x PUB_DT = 3 s at the
        # real value, so shrinking only GRIPPER_ZERO_S left the `while
        # monotonic() - t0 < GRIPPER_ZERO_S` body never entered — the test
        # passed against the reverted bug because it never reached the loop it
        # was written to cover.
        keep_s, keep_n = R.GRIPPER_ZERO_S, R.OP.EE_ACTION_REPEAT_TICKS
        R.GRIPPER_ZERO_S, R.OP.EE_ACTION_REPEAT_TICKS = 0.6, 2
        self.addCleanup(lambda: setattr(R, "GRIPPER_ZERO_S", keep_s))
        self.addCleanup(
            lambda: setattr(R.OP, "EE_ACTION_REPEAT_TICKS", keep_n))

    def test_the_wait_loop_is_actually_reached(self):
        """Guards the guard: if the burst above ever outgrows GRIPPER_ZERO_S
        again, the two tests below stop testing anything and say nothing."""
        R.zero_grippers(self.pub, self.tel, self.rig, _Det(), 0.3)
        bursts = sum(1 for p in self.pub.sent if "ee_action" in p)
        self.assertGreater(len(self.pub.sent) - bursts, 4,
                           "the GRIPPER_ZERO_S wait loop never ran")

    def test_every_packet_carries_arm_targets(self):
        why = R.zero_grippers(self.pub, self.tel, self.rig, _Det(), 0.3)
        self.assertIsNone(why, f"zeroing refused: {why}")
        self.assertTrue(self.pub.sent)
        for packet in self.pub.sent:
            self.assertIn("arm_targets", packet)

    def test_the_mission_waits_for_the_zeroing_to_finish(self):
        """FSM order (user 08-10): zero_grippers returns only after BOTH
        sides report `zeroing` False in telemetry — the survey scan can only
        start after the calibration is done. The eyes are parked (retired)
        for the duration and the prior retire set is restored on exit."""
        tel = _Tel(ee={"left": {"ee_alive": True, "zeroing": False},
                       "right": {"zeroing": False}})
        rig = _Rig()
        rig._cam_retired.add(9999)       # a pre-retired port must survive
        why = R.zero_grippers(self.pub, tel, rig, _Det(), 0.3)
        self.assertIsNone(why, f"zeroing refused: {why}")
        self.assertEqual(rig._cam_retired, {9999},
                         "the zeroing park was not restored on exit")

    def test_a_calibration_that_kills_the_service_refuses_the_mission(self):
        """08-10 night: an unpowered servo rail answered all-zero reads, the
        'calibration' finished in 2 s (zeroing False) and the gate printed
        ZEROED over jaws that never moved. A failed calibration marks the
        service unhealthy (ee_alive False) — finished is not succeeded."""
        class _DiesAfterZero(_Tel):
            def fresh(self):
                d = super().fresh()
                d["ee"] = {"left": {"ee_alive": True, "zeroing": False},
                           "right": {"zeroing": False}}
                # alive long enough to pass the entry wait, dead by the time
                # the completion check runs (the flags already read False)
                if time.monotonic() - self._t0 > 0.5:
                    d["ee"]["left"]["ee_alive"] = False
                return d

            def __init__(self):
                super().__init__(ee={"left": {"ee_alive": True,
                                              "zeroing": False},
                                     "right": {"zeroing": False}})
                self._t0 = time.monotonic()

        rig = _Rig()
        why = R.zero_grippers(self.pub, _DiesAfterZero(), rig, _Det(), 0.3)
        self.assertIsNotNone(why, "a dead service still printed ZEROED")
        self.assertIn("FAILED", why)
        self.assertEqual(rig._cam_retired, set(),
                         "the failure path did not restore the retire set")

    def test_a_stuck_calibration_refuses_the_mission(self):
        """`zeroing` True past the budget is a stuck calibration — refuse
        loudly instead of surveying with an uncalibrated hand."""
        keep = R.GRIPPER_ZERO_DONE_S
        R.GRIPPER_ZERO_DONE_S = 1.2
        self.addCleanup(lambda: setattr(R, "GRIPPER_ZERO_DONE_S", keep))
        tel = _Tel(ee={"left": {"ee_alive": True, "zeroing": True},
                       "right": {"zeroing": True}})
        rig = _Rig()
        why = R.zero_grippers(self.pub, tel, rig, _Det(), 0.3)
        self.assertIsNotNone(why)
        self.assertIn("still in flight", why)
        self.assertEqual(rig._cam_retired, set(),
                         "the refusal path did not restore the retire set")

    def test_the_zero_commands_still_go_out(self):
        """The hold must ride ALONGSIDE the ee_action, not replace it — a
        packet carries at most one ee_action and dropping it would silently
        skip the calibration this whole function exists for."""
        R.zero_grippers(self.pub, self.tel, self.rig, _Det(), 0.3)
        sides = {p["ee_action"]["side"] for p in self.pub.sent
                 if "ee_action" in p}
        self.assertEqual(sides, {"left", "right"})

    def test_it_refuses_rather_than_crawling_when_encoders_are_missing(self):
        class _NoEnc(_Tel):
            def fresh(self):
                d = super().fresh()
                d["joint_pos"] = [0.0] * 4
                return d

        why = R.zero_grippers(self.pub, _NoEnc(), self.rig, _Det(), 0.3)
        self.assertIsNotNone(why)
        self.assertIn("encoders", why)


@unittest.skipIf(R is None, _SKIP)
class JourneyTuckPacketTests(unittest.TestCase):
    """The walk's arm content. It used to evaporate exactly when it converged."""

    class _WalkRig:
        _journey_walk_pose = True

        def __init__(self, jpos, held=()):
            self._jpos = jpos
            self._held = set(held)

    def _tucked_jpos(self):
        jp = [0.0] * 31
        for i, arm in enumerate(("left", "right")):
            tgt = R.OP.mirror_arm(R.OP.POWERON_JOINTS, arm)
            jp[13 + 7 * i:20 + 7 * i] = [float(v) for v in tgt]
        return jp

    def test_arms_already_at_the_tuck_are_still_commanded(self):
        """THE regression. Dropping converged arms made the packet None the
        moment the last one arrived, so the failsafe took the arms right after
        they reached the tuck and crawled them for the rest of the walk."""
        pkt = R.journey_tuck_packet(self._WalkRig(self._tucked_jpos()))
        self.assertIsNotNone(pkt, "packet went None once the arms were tucked")
        self.assertEqual(set(pkt["arm_targets"]), {"left", "right"})

    def test_the_commanded_target_is_the_tuck_not_the_measurement(self):
        """Re-sending a FIXED posture is safe; re-sending the encoders is the
        sag-follower bug hold_packet's docstring warns about."""
        jp = self._tucked_jpos()
        jp[13] += 0.05                                   # a little sag
        pkt = R.journey_tuck_packet(self._WalkRig(jp))
        want = [float(v) for v in R.OP.mirror_arm(R.OP.POWERON_JOINTS, "left")]
        self.assertEqual(pkt["arm_targets"]["left"]["joint_pos"], want)

    def test_a_held_arm_is_still_left_out(self):
        """Unchanged by this fix and called out on purpose: a held arm rides
        with its cube and must not be yanked to the tuck. It therefore gets NO
        arm content from this packet, which is a known gap in the journey path
        (never hardware-run) — a loaded arm walking beside a tucked one has
        only the other arm's slot keeping the timer alive."""
        pkt = R.journey_tuck_packet(
            self._WalkRig(self._tucked_jpos(), held=("left",)))
        self.assertEqual(set(pkt["arm_targets"]), {"right"})


@unittest.skipIf(R is None, _SKIP)
class NoSilentPublishTests(unittest.TestCase):
    """The shape that caused it, pinned structurally.

    Every site was `rig.tick(telemetry)` — one argument, no packet, therefore
    no arm content. The behavioural tests above cover the helpers, but the
    worst two sites (the 120 s scan and the 120 s survey) live inside main()
    and journey_mission() behind a plan server, a monitor and a robot. Reading
    the source is the only thing that covers them without that stack, and it
    covers every future one for free.
    """

    def test_no_tick_call_omits_the_packet(self):
        tree = ast.parse(SRC.read_text())
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "tick" \
                    and len(node.args) < 2 and not node.keywords:
                bad.append(node.lineno)
        self.assertEqual(bad, [], f"gaze-only publish(es) at line(s) {bad} — "
                                 f"a .tick() with no packet carries no arm "
                                 f"content, so real_env's 0.5 s arm-silence "
                                 f"failsafe owns the arms for that whole loop")


if __name__ == "__main__":
    unittest.main()
