"""The zeroing guard weighs a stale grasp marker against the servo's own load.

WHY THIS EXISTS (2026-08-01, hardware, the user's words, translated: "every
power cycle ... it very probably refuses, and that is unacceptable"). The
guard used to refuse on the mere PRESENCE of a
prior grasp marker. The marker is cleared only by hand_open — and a successful
mission's DESIGNED ending is "hold the cube until Ctrl+C", which never sends
one. So every grasp poisoned the next run into a REFUSED. The refusal that
triggered this fix carried grasp_action_id=1785606424056956162, which decodes
to a hand_grab from two minutes earlier in the same bench session: not stale
power-cycle state, the tool's own previous run.

The hazard is real — zeroing pinches then opens, dropping anything held — so
the marker is now weighed against physical evidence: heartbeat servo_load
(|load|, 0..1023). Squeezing a cube is an active servo: sustained load. An
empty hand at rest reads noise. The three verdicts, each pinned below:

    marker + load > EMPTY_GRIP_LOAD_MAX   -> refuse (really holding something)
    marker + load <= EMPTY_GRIP_LOAD_MAX  -> explain and calibrate
    marker + no load reading in time      -> refuse (fail CLOSED, say why)

And throughout the evidence wait the arms are HELD — the no-silent-waits rule
(test_arm_hold_during_waits) applies to this new wait like every other.
"""

from __future__ import annotations

import time
import unittest

try:
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"


def _jpos():
    return [0.1 * i for i in range(31)]


class _Tel:
    """Telemetry with a configurable left-gripper ee block."""

    def __init__(self, marker=True, load=None, load_after=None,
                 alive=True, alive_after=None):
        self.load = load
        self.load_after = load_after          # (seconds, value): late heartbeat
        self.marker = marker
        self.alive = alive
        self.alive_after = alive_after        # seconds until alive flips True
        self.t0 = time.monotonic()

    def fresh(self):
        load = self.load
        if self.load_after and time.monotonic() - self.t0 >= self.load_after[0]:
            load = self.load_after[1]
        alive = self.alive
        if self.alive_after is not None and \
                time.monotonic() - self.t0 >= self.alive_after:
            alive = True
        left = {"ee_alive": alive, "servo_load": load}
        if self.marker:
            left["grasp_action_id"] = 1785606424056956162
        return {"joint_pos": _jpos(),
                "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, False],
                "ee": {"left": left, "right": {"servo_load": 0.0}}}


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)


class _Rig:
    def __init__(self):
        # zero_grippers parks both eyes at the origin for the calibration
        # (retire + restore in its finally) — the rig must carry the set.
        self._cam_retired = set()

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out


class _Det:
    def poll(self, rig):
        pass


def _run(tel):
    keep = (R.GRIPPER_ZERO_S, R.EE_LOAD_EVIDENCE_S, R.EE_ALIVE_WAIT_S,
            R.OP.EE_ACTION_REPEAT_TICKS)
    R.GRIPPER_ZERO_S, R.EE_LOAD_EVIDENCE_S, R.EE_ALIVE_WAIT_S = 0.2, 0.6, 0.6
    R.OP.EE_ACTION_REPEAT_TICKS = 1
    pub = _Pub()
    try:
        why = R.zero_grippers(pub, tel, _Rig(), _Det(), 0.3)
    finally:
        (R.GRIPPER_ZERO_S, R.EE_LOAD_EVIDENCE_S, R.EE_ALIVE_WAIT_S,
         R.OP.EE_ACTION_REPEAT_TICKS) = keep
    return why, pub


@unittest.skipIf(R is None, _SKIP)
class ZeroGuardVerdictTests(unittest.TestCase):

    def test_marker_with_real_load_still_refuses(self):
        """The original hazard, intact: an actively squeezing hand is never
        zeroed, marker or not — a hold at reduced torque reads hundreds."""
        why, _ = _run(_Tel(marker=True, load=350.0))
        self.assertIsNotNone(why)
        self.assertIn("squeezing", why)

    def test_marker_with_idle_load_calibrates(self):
        """THE fix. A stale marker on a hand exerting no grip force must not
        end the mission — this exact combination (verdict None, id set, load
        idle) is what every post-grasp restart looks like."""
        why, pub = _run(_Tel(marker=True, load=12.0))
        self.assertIsNone(why, f"refused a verifiably empty hand: {why}")
        sides = {p["ee_action"]["side"] for p in pub.sent if "ee_action" in p}
        self.assertEqual(sides, {"left", "right"}, "zeroing never went out")

    def test_marker_with_no_load_reading_fails_closed(self):
        """No evidence is not permission: a service whose heartbeat never
        delivers a load reading gets a refusal, not a shrug."""
        why, _ = _run(_Tel(marker=True, load=None))
        self.assertIsNotNone(why)
        self.assertIn("no servo load reading", why)

    def test_a_late_heartbeat_is_waited_for(self):
        """The poll alternates sides on a seconds cadence; a reading that
        arrives during the evidence window must be used, not missed."""
        why, _ = _run(_Tel(marker=True, load=None, load_after=(0.3, 8.0)))
        self.assertIsNone(why, f"refused despite the load arriving: {why}")

    def test_no_marker_needs_no_evidence(self):
        why, _ = _run(_Tel(marker=False, load=None))
        self.assertIsNone(why, f"clean gripper refused: {why}")

    def test_the_evidence_wait_holds_the_arms(self):
        """The new wait obeys the no-silent-waits rule: every packet it
        publishes carries arm content."""
        _, pub = _run(_Tel(marker=True, load=None, load_after=(0.3, 8.0)))
        self.assertTrue(pub.sent)
        for pkt in pub.sent:
            self.assertIn("arm_targets", pkt)


@unittest.skipIf(R is None, _SKIP)
class EeAliveRaceTests(unittest.TestCase):
    """ee_alive is heartbeat-derived; a service restart needs a redial plus one
    1 Hz beat. Sampling it once refused freshly-restarted stacks (2026-08-01:
    restart T2, re-run, REFUSED — cured by literally running it again)."""

    def test_a_late_heartbeat_is_waited_for_not_refused(self):
        why, _ = _run(_Tel(marker=False, alive=False, alive_after=0.3))
        self.assertIsNone(why, f"refused during the heartbeat race: {why}")

    def test_a_chain_that_stays_dead_still_refuses(self):
        why, _ = _run(_Tel(marker=False, alive=False))
        self.assertIsNotNone(why)
        self.assertIn("still not alive", why)

    def test_the_alive_wait_holds_the_arms(self):
        _, pub = _run(_Tel(marker=False, alive=False, alive_after=0.4))
        self.assertTrue(pub.sent)
        for pkt in pub.sent:
            self.assertIn("arm_targets", pkt)


if __name__ == "__main__":
    unittest.main()
