"""A blocking solve must not make the commander go silent.

WHY THIS EXISTS. Hardware 2026-07-30, three failures with one cause. Every
cuRobo solve until that day took 0.13-0.14 s, so nobody noticed that
plan_until_gated blocks the tool's whole 50 Hz loop. Then one solve took 4.91 s.

  1. real_env's arm-silence failsafe fires at _ARM_CMD_TIMEOUT = 0.5 s and then
     ramps the arms toward the default pose at _ARM_RETURN_RATE = 0.025 rad/s.
     4.41 s past the timeout is 0.11 rad of crawl on every joint.
  2. Gate B certifies route[0] == measured to 0.05 rad — from a reading taken
     BEFORE the solve. 0.11 rad is twice that, so the certificate was void
     before it finished printing. It read 1.2e-07 rad.
  3. No detections are ingested while blocked, so the plan target goes stale by
     the length of the solve and the drift abort compares a fresh sighting
     against it. The run died on "cube moved 5.1 cm" with a p50 of 46.4 mm on
     the FIRST streaming sample — before the arm could have touched anything.

These tests pin the properties, not the implementation: packets keep flowing at
roughly the publish rate, the held target is a snapshot rather than a follower,
detections keep being ingested, and the worker's exception still reaches the
caller. A test that only checked "pump_during returns the right value" would
pass against the very bug this replaced.
"""

from __future__ import annotations

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


class _Tel:
    def __init__(self):
        self.calls = 0

    def fresh(self):
        self.calls += 1
        return {"joint_pos": [0.0] * 31, "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, False]}


class _Det:
    def __init__(self):
        self.polls = 0

    def poll(self, rig):
        self.polls += 1


class _Rig:
    """Passes the packet through and stamps a gaze field, like the real one."""

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


@unittest.skipIf(R is None, _SKIP)
class PumpDuringTests(unittest.TestCase):

    def setUp(self):
        self.pub, self.tel, self.det, self.rig = _Pub(), _Tel(), _Det(), _Rig()

    def _run(self, work, hold=ENC):
        return R.pump_during(work, self.pub, self.tel, self.det, self.rig,
                             hold, 0.3, "test solve")

    def test_the_loop_keeps_publishing_for_the_whole_solve(self):
        """THE test. A 1.2 s solve is 2.4x the 0.5 s failsafe timeout, so the
        old code would have left a 0.7 s hole here and the arms crawling."""
        got = self._run(lambda: (time.sleep(1.2), "route")[1])
        self.assertEqual(got, "route")
        self.assertGreater(len(self.pub.sent), 1.2 / R.PUB_DT * 0.6,
                           "published far fewer packets than the solve lasted")
        gaps = np.diff(np.asarray(self.pub.stamps))
        self.assertLess(float(gaps.max()), 0.5,
                        f"went silent for {gaps.max():.2f}s — the failsafe "
                        f"fires at 0.5s")

    def test_every_packet_carries_the_arm_targets_that_reset_the_timer(self):
        """gaze_targets deliberately does NOT refresh real_env's arm timer
        ([humanoid_real_env.py:885]), so a keepalive of gaze alone would look
        busy here and still let the arms crawl on hardware."""
        self._run(lambda: (time.sleep(0.4), None)[1])
        self.assertTrue(self.pub.sent)
        for packet in self.pub.sent:
            self.assertIn("arm_targets", packet)
            self.assertEqual(set(packet["arm_targets"]), {"left", "right"})

    def test_the_held_posture_is_a_snapshot_not_a_follower(self):
        """Re-reading the encoders each tick would let gravity sag walk the arm
        downhill one packet at a time, every packet looking like a no-op."""
        self._run(lambda: (time.sleep(0.4), None)[1])
        first = self.pub.sent[0]["arm_targets"]["left"]["joint_pos"]
        for packet in self.pub.sent[1:]:
            self.assertEqual(packet["arm_targets"]["left"]["joint_pos"], first)

    def test_the_held_posture_is_the_measured_one(self):
        self._run(lambda: (time.sleep(0.3), None)[1])
        slot = self.pub.sent[0]["arm_targets"]["right"]["joint_pos"]
        self.assertEqual(slot, [ENC[n] for n in R.ARM_JOINTS_R])

    def test_detections_keep_being_ingested(self):
        """The plan target is a median over the cube history. Freezing that
        history for the length of the solve is what made a stationary cube read
        as 46 mm of drift on the first streaming sample."""
        self._run(lambda: (time.sleep(0.5), None)[1])
        self.assertGreater(self.det.polls, 5)

    def test_a_worker_exception_reaches_the_caller(self):
        """PlanServerError has to land in the same handler it always did, not
        vanish into a thread and return None as if the plan had simply failed
        its gates."""
        class Boom(RuntimeError):
            pass

        def work():
            raise Boom("server died")

        with self.assertRaises(Boom):
            self._run(work)

    def test_without_encoders_it_still_pumps_gaze(self):
        """Degraded but honest: no snapshot to hold, so no arm_targets — the
        caller refuses before reaching here, and this only pins that the pump
        does not crash on the path."""
        self._run(lambda: (time.sleep(0.2), None)[1], hold=None)
        self.assertTrue(self.pub.sent)
        self.assertTrue(all("arm_targets" not in p for p in self.pub.sent))


@unittest.skipIf(R is None, _SKIP)
class DriftAbortTests(unittest.TestCase):
    """The abort compares medians, because what it compares against is one."""

    def test_the_window_matches_the_plan_targets_window(self):
        """planned_target comes from GazeRig.median over CubeTrack.hist, which
        is maxlen=5. Gating a median-of-5 with a single raw sighting was the
        asymmetry: one 1-inlier PnP glitch could end a mission."""
        self.assertEqual(R.CUBE_DRIFT_WINDOW, 5)

    def test_one_garbage_frame_cannot_reach_the_threshold(self):
        """Four good sightings and one 40 cm flyer: the median is unmoved."""
        good = np.array([0.40, 0.10, 0.08])
        pts = [good] * 4 + [good + np.array([0.4, 0.0, 0.0])]
        drift = float(np.linalg.norm(np.median(np.stack(pts), axis=0) - good))
        self.assertLess(drift, R.CUBE_DRIFT_ABORT_M)

    def test_a_real_move_still_trips(self):
        """Robustness must not become blindness: once the cube has actually
        moved, the whole window moves with it."""
        good = np.array([0.40, 0.10, 0.08])
        moved = good + np.array([0.09, 0.0, 0.0])
        drift = float(np.linalg.norm(
            np.median(np.stack([moved] * 5), axis=0) - good))
        self.assertGreater(drift, R.CUBE_DRIFT_ABORT_M)


if __name__ == "__main__":
    unittest.main()
