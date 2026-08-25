"""FIRST SEER LOCKS (user 2026-08-02): the eye that actually saw a cube locks
it immediately; a busy eye's sighting must NOT assign the cube to the other
eye. Hardware symptom that forced the spec: during the serpentine the monitor
visibly drew a decoded cube, yet no camera stopped — the retired cross-claim
("spotted by X, which is taken -> assign the free eye") parked the claim on an
eye that had never decoded the cube, while the seeing eye swept on because the
cube "had an owner". These tests pin the ingest claim rule and the eye()
interplay that breaks that livelock.
"""
import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    from auto_operator import gaze as G
    from auto_operator import perception as P

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    OP = None
    _SKIP = f"operator unavailable ({type(exc).__name__}: {exc})"

PORT_L, PORT_R = (OP.CAM_PORTS if OP else (5555, 5556))


class _Op:
    """Duck-typed operator: exactly the fields ingest()/eye() touch. Claims the
    operator's module so _K(op) resolves the REAL constants (journey-test
    pattern, see test_journey_curobo.py)."""

    def __init__(self, keys):
        self.cubes = {k: OP.CubeTrack(key=k) for k in keys}
        self._cam_retired = set()
        self._grip = {"left": None, "right": None}
        self.gaze = np.zeros(31)
        self._scan_phase = 0.0
        self.aimed, self.scanned = [], []

    def _aim_angles(self, port, target, jpos):
        self.aimed.append(port)
        return 1.0, -1.0

    def _scan_angles(self, port):
        self.scanned.append(port)
        return 2.0, -2.0


_Op.__module__ = "humanoid_auto_operator"


def _det(port, pos=(0.5, 0.0, 0.05), inliers=3):
    return {"pos": list(pos), "camera_port": port, "n_inliers": inliers}


@unittest.skipIf(OP is None, _SKIP)
class FirstSeerLockTests(unittest.TestCase):

    def test_first_sighting_locks_the_seeing_eye(self):
        op = _Op(["a", "b"])
        P.ingest(op, {"b": _det(PORT_R, pos=(-0.4, 0.1, 0.05))}, now=10.0)
        self.assertEqual(op.cubes["b"].camera_port, PORT_R,
                         "the eye that saw the cube must lock it immediately")
        self.assertIsNone(op.cubes["a"].camera_port)

    def test_a_busy_eyes_sighting_does_not_cross_claim_the_other_eye(self):
        op = _Op(["a", "b"])
        P.ingest(op, {"a": _det(PORT_L)}, now=10.0)     # a locks the left eye
        # the BUSY left eye grazes b on its sweep: b must stay unclaimed —
        # assigning the right eye to a position it never decoded is the
        # livelock ("monitor shows the cube, nobody stops").
        P.ingest(op, {"b": _det(PORT_L, pos=(-0.4, 0.1, 0.05))}, now=10.5)
        self.assertIsNone(op.cubes["b"].camera_port,
                          "a busy eye's sighting must not assign the cube "
                          "to an eye that has never seen it")
        # the sighting still feeds GO/REACH: pos/freshness update regardless
        self.assertIsNotNone(op.cubes["b"].pos)
        self.assertEqual(op.cubes["b"].pos_port, PORT_L)
        self.assertTrue(op.cubes["b"].fresh(10.6, 1.0))
        # and the moment the FREE eye sees it itself, it locks
        P.ingest(op, {"b": _det(PORT_R, pos=(-0.4, 0.1, 0.05))}, now=11.0)
        self.assertEqual(op.cubes["b"].camera_port, PORT_R)

    def test_the_free_eye_keeps_scanning_while_the_grazed_cube_is_unclaimed(self):
        op = _Op(["a", "b"])
        now = 10.0
        P.ingest(op, {"a": _det(PORT_L)}, now=now)
        P.ingest(op, {"b": _det(PORT_L, pos=(-0.4, 0.1, 0.05))}, now=now)
        G.eye(op, jpos=None, now=now)
        self.assertIn(PORT_L, op.aimed, "the locked eye must track its cube")
        self.assertIn(PORT_R, op.scanned,
                      "an unclaimed-but-fresh cube must keep the free eye "
                      "SCANNING (not parked, not staring at hearsay)")
        # once the free eye locks b itself, both eyes aim and the scan ends
        P.ingest(op, {"b": _det(PORT_R, pos=(-0.4, 0.1, 0.05))}, now=now + 0.5)
        op.aimed, op.scanned = [], []
        G.eye(op, jpos=None, now=now + 0.5)
        self.assertEqual(sorted(op.aimed), sorted([PORT_L, PORT_R]))
        self.assertEqual(op.scanned, [])

    def test_a_locked_eye_stares_through_the_grace_then_rescans(self):
        """User 2026-08-10 ("fix on the cube as long as you can; if you lose
        track, just scan around again" — supersedes the 08-04 permanent
        stare): within GAZE_STARE_GRACE_S of the last sighting the eye keeps
        aiming at the known position (covers a grasp's hand-occlusion
        window, and a claim release alone must not move it); PAST the grace
        the eye rejoins the serpentine instead of watching a rotted
        body-frame point forever (the 08-10 camera-5556 stare)."""
        op = _Op(["a", "b"])
        now = 10.0
        P.ingest(op, {"a": _det(PORT_L)}, now=now)
        inside = now + OP.GAZE_STARE_GRACE_S - 1.0
        op.aimed, op.scanned = [], []
        G.eye(op, jpos=None, now=inside)
        self.assertIn(PORT_L, op.aimed,
                      "the eye stopped aiming inside the stare grace")
        self.assertNotIn(PORT_L, op.scanned,
                         "the eye rescanned inside the grace")
        # claim release alone (still inside the grace) must not move it
        op.cubes["a"].camera_port = None
        op.aimed, op.scanned = [], []
        G.eye(op, jpos=None, now=inside)
        self.assertIn(PORT_L, op.aimed,
                      "the claim release moved the eye inside the grace")
        # PAST the grace: the stale stare ends and the search resumes
        op.aimed, op.scanned = [], []
        G.eye(op, jpos=None, now=now + OP.GAZE_STARE_GRACE_S + 1.0)
        self.assertNotIn(PORT_L, op.aimed,
                         "the eye still stares at a point stale past the "
                         "grace — the 08-10 frozen-camera bug is back")
        self.assertIn(PORT_L, op.scanned,
                      "the eye did not rejoin the search after the grace")

    def test_a_retired_eye_parks_at_origin_and_never_rejoins(self):
        """User 2026-08-09 (supersedes the 07-26 rejoin rule): a camera whose
        cube was carried home sticks to the origin. Even the strongest pull
        the old rule answered — the OTHER cube pending with NO fresh fix —
        must not move it; the free eye carries the search alone."""
        op = _Op(["a", "b"])
        P.ingest(op, {"a": _det(PORT_L)}, now=10.0)
        op.cubes["a"].reached = True
        op.cubes["a"].held_by = "left"
        op._cam_retired.add(PORT_L)
        stale = 10.0 + OP.DET_LOST_RESCAN_S + 5.0   # b: pending, unfixed
        op.gaze[:] = 0.7
        gaze = G.eye(op, jpos=None, now=stale)
        self.assertNotIn(PORT_L, op.scanned,
                         "the retired eye rejoined the search")
        self.assertNotIn(PORT_L, op.aimed, "the retired eye aimed at a cube")
        ys, ps = OP.GAZE_ORDER[PORT_L]
        self.assertEqual((gaze[ys], gaze[ps]), (0.0, 0.0),
                         "the retired eye is not at the origin")
        self.assertIn(PORT_R, op.scanned,
                      "the free eye must carry the search alone")

    def test_a_retired_port_is_never_reclaimed(self):
        """The claim side of the same 08-09 rule: perception.ingest used to
        un-retire a port on a fresh claim ('actively needed again') — now a
        retired eye's sighting feeds pos/hist but never takes the claim, and
        the free eye locks the cube the moment it sees it itself."""
        op = _Op(["a", "b"])
        op._cam_retired.add(PORT_L)
        P.ingest(op, {"b": _det(PORT_L, pos=(-0.4, 0.1, 0.05))}, now=10.0)
        self.assertIsNone(op.cubes["b"].camera_port,
                          "a retired eye took a claim")
        self.assertIn(PORT_L, op._cam_retired, "ingest un-retired the eye")
        self.assertIsNotNone(op.cubes["b"].pos,
                             "the sighting must still feed GO/REACH")
        P.ingest(op, {"b": _det(PORT_R, pos=(-0.4, 0.1, 0.05))}, now=10.5)
        self.assertEqual(op.cubes["b"].camera_port, PORT_R)

    def test_all_cubes_reached_parks_both_eyes_at_zero(self):
        """User 2026-08-04 (re-stated): after the grasp, back to the origin.
        Every cube reached -> both eyes park at the power-on attitude."""
        op = _Op(["a", "b"])
        P.ingest(op, {"a": _det(PORT_L), "b": _det(PORT_R)}, now=10.0)
        for c in op.cubes.values():
            c.reached = True
        op.gaze[:] = 0.7
        gaze = G.eye(op, jpos=None, now=10.1)
        self.assertEqual(op.scanned, [], "a finished mission kept scanning")
        for port in (PORT_L, PORT_R):
            ys, ps = OP.GAZE_ORDER[port]
            self.assertEqual((gaze[ys], gaze[ps]), (0.0, 0.0),
                             f"camera {port} did not park at zero")


if __name__ == "__main__":
    unittest.main()
