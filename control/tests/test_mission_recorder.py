"""MissionWindow trigger pins (user 2026-08-03: record both cameras to MP4
per run, 'from zeroing to the post-grasp camera re-park'). The window must
start on the FIRST gaze packet (T6 publishes from its camera-zeroing phase),
stop only after motion-then-park held 3 s (a run that never moved must not
stop on its initial zeros), stop on 9874 silence, and re-arm for the next
run."""
import unittest

try:
    from humanoid_mission_recorder import (
        MissionWindow,
        REC_PARK_HOLD_S,
        REC_SILENT_S,
    )

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    MissionWindow = None
    _SKIP = f"recorder unavailable ({type(exc).__name__}: {exc})"

ZERO = [0.0, 0.0, 0.0, 0.0]
SWEPT = [1.2, -0.8, 0.4, -0.7]


@unittest.skipIf(MissionWindow is None, _SKIP)
class MissionWindowTests(unittest.TestCase):

    def test_starts_on_first_packet_and_stops_on_park_after_motion(self):
        w = MissionWindow()
        self.assertIsNone(w.feed(0.0, None), "started with no mission at all")
        self.assertEqual(w.feed(1.0, ZERO), "start",
                         "the first gaze packet (zeroing phase) must start it")
        # initial zeros before the sweep must NOT count as the done-park
        t = 1.0
        while t < 1.0 + REC_PARK_HOLD_S + 2.0:
            t += 0.5
            self.assertIsNone(w.feed(t, ZERO),
                              "stopped before the sweep ever moved")
        w.feed(t + 0.5, SWEPT)                     # the sweep is visibly on
        w.feed(t + 1.0, ZERO)                      # parked after the grasp
        self.assertIsNone(w.feed(t + 1.0 + REC_PARK_HOLD_S - 0.5, ZERO))
        self.assertEqual(w.feed(t + 1.0 + REC_PARK_HOLD_S + 0.1, ZERO),
                         "stop", "park held past the bar must stop the run")
        self.assertFalse(w.recording)

    def test_a_wiggle_during_the_park_restarts_the_hold(self):
        w = MissionWindow()
        w.feed(0.0, ZERO)
        w.feed(1.0, SWEPT)
        w.feed(2.0, ZERO)
        w.feed(2.0 + REC_PARK_HOLD_S - 0.5, SWEPT)   # re-aim: not done yet
        self.assertIsNone(w.feed(2.0 + REC_PARK_HOLD_S + 1.0, ZERO),
                          "the hold did not restart after a re-aim")

    def test_silence_stops_and_the_window_rearms(self):
        w = MissionWindow()
        self.assertEqual(w.feed(0.0, ZERO), "start")
        self.assertEqual(w.feed(REC_SILENT_S + 0.1, None), "stop",
                         "9874 silence (T6 gone) must finalize the files")
        self.assertEqual(w.feed(100.0, ZERO), "start",
                         "the window must re-arm for the next run")


if __name__ == "__main__":
    unittest.main()
