"""Pins for the 08-12 scan hold (GazeRig.scan_hold / auto_operator.gaze.eye).

The incident: during a journey fetch the other leg's cube has no fresh claim,
so the free eye ran the panoramic serpentine through the whole reach — and a
sweeping camera reference drags the base into a slow rotation through the
checkpoints' trained gaze-follow coupling (both v2best and Cosine; survived a
pinned waist because the LEGS carry it; stopped instantly when the operator's
gaze stream died — the user's Ctrl+C differential).

Contract under test: with scan_hold set, an eye whose cube is claimed-fresh
still AIMS, an eye with nothing to track PARKS at origin instead of scanning;
with scan_hold clear the serpentine behaves exactly as before; retired eyes
park regardless.

The duck op is defined IN THIS MODULE on purpose: gaze._K(op) resolves
constants from type(op).__module__, so the module-level names below are the
constants the law reads.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_operator import gaze as ao_gaze  # noqa: E402

# --- constants gaze._K(op) resolves from this module -------------------------
CAM_PORTS = (5555, 5556)
DET_LOST_RESCAN_S = 3.0
GAZE_STARE_GRACE_S = 3.0
GAZE_ORDER = {5555: (0, 1), 5556: (2, 3)}
DT = 0.05

AIM = (0.7, -0.5)
SCAN = (9.9, 8.8)


class FakeTrack:
    def __init__(self, reached=False, pos=None, camera_port=None,
                 last_camera_port=None, fresh=False):
        self.reached = reached
        self.pos = pos
        self.camera_port = camera_port
        self.last_camera_port = last_camera_port
        self._fresh = fresh

    def fresh(self, now, window):
        return self._fresh


class DuckOp:
    def __init__(self, cubes, scan_hold=False, retired=()):
        self.gaze = np.zeros(4)
        self.cubes = cubes
        self.scan_hold = scan_hold
        self._cam_retired = set(retired)
        self._scan_phase = 0.0

    def _aim_angles(self, port, pos, jpos):
        return AIM

    def _scan_angles(self, port):
        return SCAN


def run_eye(op):
    return ao_gaze.eye(op, np.zeros(31), now=100.0)


def tracked_plus_unclaimed(scan_hold):
    return DuckOp({
        "a": FakeTrack(pos=np.array([0.4, 0.1, 0.0]), camera_port=5555,
                       fresh=True),
        "b": FakeTrack(),                       # unclaimed, never seen
    }, scan_hold=scan_hold)


class TestScanHold(unittest.TestCase):
    def test_free_eye_scans_without_hold(self):
        g = run_eye(tracked_plus_unclaimed(scan_hold=False))
        self.assertEqual((g[0], g[1]), AIM)     # claimed eye aims
        self.assertEqual((g[2], g[3]), SCAN)    # free eye serpentines

    def test_free_eye_parks_under_hold(self):
        g = run_eye(tracked_plus_unclaimed(scan_hold=True))
        self.assertEqual((g[0], g[1]), AIM)     # tracking is untouched
        self.assertEqual((g[2], g[3]), (0.0, 0.0))  # free eye parks

    def test_missing_attr_means_no_hold(self):
        """Operator ducks without the field keep the old serpentine."""
        op = tracked_plus_unclaimed(scan_hold=False)
        del op.scan_hold
        g = run_eye(op)
        self.assertEqual((g[2], g[3]), SCAN)

    def test_all_reached_parks_both_regardless(self):
        for hold in (False, True):
            op = DuckOp({"a": FakeTrack(reached=True),
                         "b": FakeTrack(reached=True)}, scan_hold=hold)
            g = run_eye(op)
            self.assertEqual(tuple(g), (0.0, 0.0, 0.0, 0.0))

    def test_retired_eye_parks_even_when_scanning_allowed(self):
        op = tracked_plus_unclaimed(scan_hold=False)
        op._cam_retired = {5556}
        g = run_eye(op)
        self.assertEqual((g[2], g[3]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
