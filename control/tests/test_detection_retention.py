"""Publisher-side detection retention (hardware 2026-08-03): a scan-transit
decode lives in 1-2 camera frames and then must survive TWO latest-only
samplers (the monitor's 20 Hz packet build and the operator's latest-packet
poll) — a coin flip that made cameras sweep past cubes the viser view (which
retains 0.5 s for DISPLAY) was visibly drawing. build_retained_targets puts a
key's newest pose in EVERY packet for DETECTION_RETENTION_S, so one decoded
frame anywhere is guaranteed to reach the operator."""
import unittest
from types import SimpleNamespace

try:
    from humanoid_monitor import DETECTION_RETENTION_S, build_retained_targets

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    build_retained_targets = None
    _SKIP = f"monitor unavailable ({type(exc).__name__}: {exc})"


def _pose(key, stamp, port=5555):
    return SimpleNamespace(key=key, capture_stamp=float(stamp),
                           camera_port=port)


def _snap(*poses):
    return SimpleNamespace(target_poses={p.key: p for p in poses})


@unittest.skipIf(build_retained_targets is None, _SKIP)
class DetectionRetentionTests(unittest.TestCase):

    def test_a_single_frame_sighting_ships_in_every_packet_of_the_window(self):
        retained = {}
        out = build_retained_targets(retained, [_snap(_pose("a", 10.0))], 10.0)
        self.assertIn("a", out)
        # the cube decodes in NO further frame — every publish inside the
        # retention window must still carry it, and the first publish after
        # the window must not
        for now in (10.1, 10.2, 10.3, 10.4, 10.0 + DETECTION_RETENTION_S):
            self.assertIn("a", build_retained_targets(retained, [_snap()], now),
                          f"sighting dropped early at now={now}")
        self.assertNotIn("a", build_retained_targets(
            retained, [_snap()], 10.0 + DETECTION_RETENTION_S + 0.01))

    def test_newest_capture_wins_across_cameras_in_either_fold_order(self):
        for order in ([_snap(_pose("a", 10.0, 5555)), _snap(_pose("a", 10.2, 5556))],
                      [_snap(_pose("a", 10.2, 5556)), _snap(_pose("a", 10.0, 5555))]):
            out = build_retained_targets({}, order, 10.2)
            self.assertEqual(out["a"].camera_port, 5556,
                             "an older grazing view clobbered the fresher fit")

    def test_a_redetect_refreshes_the_window(self):
        retained = {}
        build_retained_targets(retained, [_snap(_pose("a", 10.0))], 10.0)
        build_retained_targets(retained, [_snap(_pose("a", 10.4))], 10.4)
        self.assertIn("a", build_retained_targets(retained, [_snap()], 10.8))

    def test_a_frozen_camera_cannot_pin_a_ghost(self):
        # A dead camera's get_latest() returns the SAME stale snapshot forever;
        # re-folding it every publish must not keep the entry alive past its
        # capture age.
        retained = {}
        stale = _snap(_pose("a", 10.0))
        for now in (10.0, 10.2, 10.4, 10.6, 10.8):
            out = build_retained_targets(retained, [stale], now)
        self.assertNotIn("a", out, "a frozen snapshot pinned a ghost sighting")


if __name__ == "__main__":
    unittest.main()
