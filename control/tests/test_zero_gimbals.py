"""Forced camera zeroing before the sweep (user 2026-08-03: 'every run must
zero the cameras first, then start turning'). Pins: the phase slews the gaze
toward zero (never a raw jump — the robot applies gaze_reference raw), holds
the arms in every packet, exits when the measured gimbals reach zero, resets
the sweep clock, and a far-from-zero start converges instead of timing out."""
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"


class _Tel:
    """Gimbals follow the last published gaze_targets exactly; arms static."""

    def __init__(self, gimbals):
        self.gimbals = list(gimbals)

    def fresh(self):
        jp = [0.0] * 31
        jp[27:31] = self.gimbals
        return {"joint_pos": jp, "arm_fault": [False, False]}


class _Pub:
    def __init__(self, tel):
        self.tel = tel
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)
        g = (packet or {}).get("gaze_targets")
        if g is not None:
            self.tel.gimbals = [float(x) for x in g]


class _Det:
    def poll(self, rig):
        pass


@unittest.skipIf(R is None, _SKIP)
class ZeroGimbalsTests(unittest.TestCase):

    def _run(self, start):
        tel = _Tel(start)
        pub = _Pub(tel)
        rig = R.GazeRig.__new__(R.GazeRig)
        rig.gaze = np.zeros(4)
        rig._gaze_seeded = False
        rig._scan_phase = 99.0                  # stale sweep clock from a
        why = R.zero_gimbals(pub, tel, rig, _Det(), 0.3)   # previous run
        return why, pub, rig, tel

    def test_a_far_gimbal_is_slewed_to_zero_and_the_sweep_clock_resets(self):
        why, pub, rig, tel = self._run([2.5, -0.8, -2.5, 0.7])
        self.assertIsNone(why, f"zeroing refused: {why}")
        self.assertLess(max(abs(v) for v in tel.gimbals),
                        R.GIMBAL_ZERO_TOL_RAD, "cameras never reached zero")
        self.assertEqual(rig._scan_phase, 0.0, "sweep clock was not reset")
        # sender-side slew: no single packet may jump more than one slew step
        step_cap = R.GAZE_SLEW_RATE * R.DT + 1e-6
        prev = None
        for pkt in pub.sent:
            g = pkt["gaze_targets"]
            if prev is not None:
                worst = max(abs(a - b) for a, b in zip(g, prev))
                self.assertLessEqual(worst, step_cap,
                                     "a gaze packet jumped past the slew "
                                     "cap — that is a raw slam on hardware")
            prev = g

    def test_every_packet_carries_the_arm_hold(self):
        why, pub, _, _ = self._run([1.0, 0.0, -1.0, 0.0])
        self.assertIsNone(why)
        self.assertTrue(pub.sent)
        for k, pkt in enumerate(pub.sent):
            self.assertIn("arm_targets", pkt,
                          f"packet {k} is gaze-only — the 0.5 s arm-silence "
                          f"failsafe would own the arms through this phase")

    def test_already_at_zero_costs_one_look(self):
        why, pub, rig, _ = self._run([0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(why)
        self.assertEqual(pub.sent, [], "no motion was needed, none published")
        self.assertEqual(rig._scan_phase, 0.0)


if __name__ == "__main__":
    unittest.main()
