"""Pin the measured-close zeroing contract (2026-08-10).

User: "sometimes the gripper does not close to the zero." The old calibration
commanded past the pinch, slept a blind 1.5 s and read ONCE — a gripper still
travelling (long stroke from wide open) or stiction-stalled at read time
recorded a short `close`, shifting the whole calibrated frame (seen live:
close=4839 vs a true ~5997, a 6000-count frame displaced by 1158).

Contract now, no hardware needed (fake hand with a stall ceiling):
  * "closed" = the position STOPPED MOVING (consecutive still reads), so a
    still-travelling gripper can never be read short;
  * a second back-off-and-pinch pass breaks static friction, and the DEEPER
    stall wins;
  * dead serial still fails the calibration instead of inventing a frame.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest

_CONTROL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CONTROL)

import humanoid_end_effector_service as EES  # noqa: E402


class _FakeHand:
    """Physical model: position follows the commanded target at a fixed step
    per read but can never pass the current stall ceiling — the pinch (or the
    debris). A backoff target below the ceiling is followed freely. Load reads
    like the HLS does: pushing against the ceiling rides the torque limit,
    sitting AT the commanded target reads idle. Each fresh ZERO_CLOSE_TARGET
    write starts a new pinch pass and may break loose to the next ceiling."""

    def __init__(self, *stalls: int, step: int = 900):
        self.pos = -1000
        self.target = None
        self.step = step
        self.stalls = list(stalls)
        self.pinch_i = -1
        self.targets: list[int] = []

    def _stall(self) -> int:
        return self.stalls[min(max(self.pinch_i, 0), len(self.stalls) - 1)]

    def set_positions(self, ids, targets, *a):
        self.target = targets[0]
        self.targets.append(targets[0])
        if targets[0] == EES.ZERO_CLOSE_TARGET:
            self.pinch_i += 1
        return True

    def get_positions(self, ids):
        limit = min(self.target, self._stall()) if self.target > self.pos \
            else self.target
        if self.pos < limit:
            self.pos = min(self.pos + self.step, limit)
        elif self.pos > limit:
            self.pos = max(self.pos - self.step, limit)
        return [self.pos]

    def get_load(self, sid):
        # Pushing toward an unreached inward target = stalled on the pinch,
        # duty rides the ZERO_TORQUE limit; parked at the target = idle.
        return 140 if (self.target is not None and self.target > self.pos) \
            else 10


class _DeadHand:
    def set_positions(self, ids, targets, *a):
        return True

    def get_positions(self, ids):
        return []

    def get_load(self, sid):
        return None


class _ZeroHand:
    """An UNPOWERED servo rail: the USB adapter enumerates and 'answers', but
    every read is zero — position 0, load 0 (the live 08-10 signature)."""

    def set_positions(self, ids, targets, *a):
        return True

    def get_positions(self, ids):
        return [0]

    def get_load(self, sid):
        return 0


def _svc(hand) -> EES.EndEffectorService:
    svc = object.__new__(EES.EndEffectorService)
    svc.hands = {"left": hand}
    svc.servo_ids = {"left": [1]}
    svc.healthy = {"left": True}
    return svc


class ZeroCalibrationTests(unittest.TestCase):

    def setUp(self):
        keep = (EES.ZERO_POLL_S, EES.ZERO_BACKOFF_S, EES.ZERO_TRAVEL_BUDGET_S)
        EES.ZERO_POLL_S, EES.ZERO_BACKOFF_S, EES.ZERO_TRAVEL_BUDGET_S = \
            0.01, 0.01, 2.0
        self.addCleanup(lambda: setattr(EES, "ZERO_POLL_S", keep[0]))
        self.addCleanup(lambda: setattr(EES, "ZERO_BACKOFF_S", keep[1]))
        self.addCleanup(lambda: setattr(EES, "ZERO_TRAVEL_BUDGET_S", keep[2]))

    def test_a_travelling_gripper_is_never_read_short(self):
        """The 6000-count stroke takes many polls — the old fixed-sleep read
        landed mid-travel. Closed must be the STALL value, exactly."""
        hand = _FakeHand(5996)
        got = asyncio.run(_svc(hand).run_zero_calibration("left"))
        self.assertEqual(got, 5996)
        # and the double-tap really happened: two inward pinches + a back-off
        self.assertEqual(
            hand.targets.count(EES.ZERO_CLOSE_TARGET), 2,
            "the second confirming pinch is gone")
        self.assertIn(5996 - EES.ZERO_BACKOFF_COUNTS, hand.targets,
                      "the back-off between pinches is gone")

    def test_stiction_on_the_first_pinch_is_beaten_by_the_second(self):
        """First pass stalls 1200 counts short (static friction); the second
        pass reaches the true pinch. The deeper stall wins."""
        hand = _FakeHand(4800, 5990, 5990)
        got = asyncio.run(_svc(hand).run_zero_calibration("left"))
        self.assertEqual(got, 5990,
                         "the early stiction stall was taken as the zero")

    def test_disagreeing_taps_earn_a_third_opinion(self):
        """Two stalls 1190 counts apart mean at least one lied; the deeper of
        two liars is still a guess. The suspicious case — and only it — pays
        for one more pinch, and the deepest of the three wins."""
        hand = _FakeHand(4800, 5990, 5992)
        got = asyncio.run(_svc(hand).run_zero_calibration("left"))
        self.assertEqual(got, 5992, "the third tap's deeper stall was ignored")
        self.assertEqual(hand.targets.count(EES.ZERO_CLOSE_TARGET), 3,
                         "the disagreement did not trigger a third pinch")

    def test_agreeing_taps_do_not_pay_for_a_third(self):
        """The clean case keeps its time budget: two agreeing stalls, exactly
        two pinches."""
        hand = _FakeHand(5996)
        asyncio.run(_svc(hand).run_zero_calibration("left"))
        self.assertEqual(hand.targets.count(EES.ZERO_CLOSE_TARGET), 2,
                         "an agreeing pair still paid for a third tap")

    def test_a_frame_shifted_past_the_close_target_is_chased_to_the_pinch(self):
        """The replaced board's hazard: the true pinch sits BEYOND
        ZERO_CLOSE_TARGET, so the servo arrives at its target in free space —
        perfectly still, jaws open. Still + at-target + idle load must chase
        deeper until the load proves a real stall."""
        hand = _FakeHand(EES.ZERO_CLOSE_TARGET + 600)
        got = asyncio.run(_svc(hand).run_zero_calibration("left"))
        self.assertEqual(got, EES.ZERO_CLOSE_TARGET + 600,
                         "a free-space arrival at the close target was "
                         "recorded as the zero — the jaws never touched")
        self.assertTrue(any(t > EES.ZERO_CLOSE_TARGET for t in hand.targets),
                        "the chase never drove past the shallow target")

    def test_dead_serial_fails_the_calibration(self):
        got = asyncio.run(_svc(_DeadHand()).run_zero_calibration("left"))
        self.assertIsNone(got, "a dead serial line invented a zero frame")

    def test_an_unpowered_rail_answering_zeros_fails_loudly(self):
        """THE 08-10 night incident ("It's not zeroing at all!"): with the
        gripper servo rail unpowered, every read answers position 0 / load 0.
        Constant zeros pass the stillness test, both taps agree at 0, and the
        old code succeeded with close=0/open=-6000 while the jaws never
        moved. Still + far from target + no push = not executing = FAIL."""
        got = asyncio.run(_svc(_ZeroHand()).run_zero_calibration("left"))
        self.assertIsNone(got, "an unpowered rail's all-zero answers were "
                               "accepted as a zero frame (close=0)")


if __name__ == "__main__":
    unittest.main()
