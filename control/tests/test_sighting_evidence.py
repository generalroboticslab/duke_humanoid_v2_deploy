"""What the reach latch's median is actually made of (audit 2026-08-06).

The tracker's whole verdict is one number, ``worst = ||FK(measured encoders) -
goal_pos||``, and ``goal_pos`` is ``GazeRig.median`` over ``CubeTrack.hist``.
Hardware that evening parked ``worst`` at 51 mm with the right gripper
reporting CONTACT: the hand was ON the cube and the model said 51 mm, i.e. the
whole number was goal error. Nine more millimetres and the identical
successful grasp would have printed T13 and retreated.

Two defects in that history made the error far larger than it needed to be,
and both are pinned here:

  * the deque stored no evidence count, so a 1-face 16 mm-tag PnP fit ("tens
    of mm" by this codebase's own measure) counted exactly as much as a
    three-face fit — and with two samples ``np.median`` IS the mean, so one
    bad frame moved the answer by half its error;
  * entries were stamped on ARRIVAL, and the monitor republishes each key's
    newest pose in every packet for ``DETECTION_RETENTION_S``
    (test_detection_retention.py pins that it must), so one decode filled all
    five slots — the median of five identical values is that value, and the
    averaging this deque exists for silently vanished exactly while a
    descending hand was occluding its own cube.
"""
import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R
    from auto_operator import perception as P

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    OP = None
    _SKIP = f"operator unavailable ({type(exc).__name__}: {exc})"

PORT_L, PORT_R = (OP.CAM_PORTS if OP else (5555, 5556))


class _Op:
    """Duck-typed operator: exactly the fields ingest() touches. Claims the
    operator's module so _K(op) resolves the REAL constants (first-seer-lock
    pattern, see test_first_seer_lock.py)."""

    def __init__(self, keys):
        self.cubes = {k: OP.CubeTrack(key=k) for k in keys}
        self._cam_retired = set()
        self._grip = {"left": None, "right": None}


_Op.__module__ = "humanoid_auto_operator"


def _det(pos, inliers, stamp, port=PORT_L):
    return {"pos": list(pos), "camera_port": port, "n_inliers": inliers,
            "capture_stamp": float(stamp)}


@unittest.skipIf(OP is None, _SKIP)
class SightingEvidenceTests(unittest.TestCase):

    # --- the history carries what the median needs to judge it -------------

    def test_a_sighting_records_the_face_count_behind_its_fit(self):
        op = _Op(["a"])
        P.ingest(op, {"a": _det((0.5, 0.0, 0.05), 3, 10.0)}, now=10.0)
        s = op.cubes["a"].hist[-1]
        self.assertEqual(s.faces, 3, "the evidence count was thrown away — the "
                                     "median cannot tell a 3-face fit from a guess")
        self.assertEqual(s.t, 10.0, "stamped on arrival instead of on capture")
        self.assertEqual(s.port, PORT_L)

    # --- retention must not collapse the window ----------------------------

    def test_a_retained_republish_does_not_fill_the_window_with_one_decode(self):
        """The monitor ships the SAME pose in every packet for 0.5 s. Ingesting
        it five times must leave ONE sample, not five — otherwise the median
        degenerates to that single decode while looking like a five-sample
        average."""
        op = _Op(["a"])
        det = _det((0.5, 0.0, 0.05), 1, stamp=10.0)
        for k in range(5):                       # five packets, one capture
            P.ingest(op, {"a": det}, now=10.0 + 0.05 * k)
        self.assertEqual(len(op.cubes["a"].hist), 1,
                         "one decode occupied the whole window")

    def test_distinct_captures_still_accumulate(self):
        op = _Op(["a"])
        for k in range(5):
            P.ingest(op, {"a": _det((0.5, 0.0, 0.05), 2, stamp=10.0 + 0.05 * k)},
                     now=10.0 + 0.05 * k)
        self.assertEqual(len(op.cubes["a"].hist), 5,
                         "the dedup swallowed genuinely new observations")

    # --- the median prefers the best evidence it holds ---------------------

    def test_a_lone_one_face_frame_cannot_drag_a_two_face_median(self):
        """The failure this whole file exists for: two samples, one of them a
        1-face guess 100 mm off. np.median over an even count is the MEAN, so
        the old unfiltered median answered 50 mm wrong — the error lands whole
        in `worst`, which is what decides a retreat."""
        rig = R.GazeRig(["a"])
        rig.ingest({"a": _det((0.50, 0.0, 0.05), 2, stamp=10.00)}, 10.00)
        rig.ingest({"a": _det((0.60, 0.0, 0.05), 1, stamp=10.05)}, 10.05)
        med = rig.median("a", 10.05)
        self.assertIsNotNone(med)
        np.testing.assert_allclose(
            med, [0.50, 0.0, 0.05], atol=1e-9,
            err_msg="the 1-face frame moved the goal — best evidence must win")

    def test_equal_evidence_still_averages(self):
        """Filtering is on evidence, not on recency: two 2-face fits are both
        admissible and the median must still absorb their spread. Dropping to
        'newest wins' would throw away the noise rejection this deque is for."""
        rig = R.GazeRig(["a"])
        rig.ingest({"a": _det((0.50, 0.0, 0.05), 2, stamp=10.00)}, 10.00)
        rig.ingest({"a": _det((0.54, 0.0, 0.05), 2, stamp=10.05)}, 10.05)
        med = rig.median("a", 10.05)
        np.testing.assert_allclose(med, [0.52, 0.0, 0.05], atol=1e-9)

    def test_the_evidence_gates_read_the_median_not_the_newest_packet(self):
        """Audit 2026-08-06 (D7). last_inliers is the face count of the NEWEST
        PACKET; faces_used is the face count of the samples median() actually
        took its median over. With a 3-face fix followed by a 1-face one, the
        median is pure 3-face — and every gate that consumed last_inliers threw
        it away as if it were a guess, printing "1-face update ignored" about a
        value containing zero 1-face samples."""
        rig = R.GazeRig(["a"])
        rig.ingest({"a": _det((0.50, 0.0, 0.05), 3, stamp=10.00)}, 10.00)
        rig.ingest({"a": _det((0.60, 0.0, 0.05), 1, stamp=10.05)}, 10.05)
        self.assertEqual(rig.last_inliers("a"), 1,
                         "the newest packet really is 1-face")
        self.assertEqual(rig.faces_used("a", 10.05), 3,
                         "the median is pure 3-face and must report it")
        np.testing.assert_allclose(rig.median("a", 10.05), [0.50, 0.0, 0.05],
                                   atol=1e-9)

    def test_faces_used_is_zero_when_there_is_no_live_evidence(self):
        rig = R.GazeRig(["a"])
        rig.ingest({"a": _det((0.50, 0.0, 0.05), 2, stamp=10.0)}, 10.0)
        stale = 10.0 + OP.CUBE_LATCH_WINDOW_S + 0.05
        self.assertIsNone(rig.median("a", stale))
        self.assertEqual(rig.faces_used("a", stale), 0)

    def test_the_window_is_measured_on_capture_age_not_arrival_age(self):
        """A decode older than CUBE_LATCH_WINDOW_S must leave the window even
        while retention keeps re-delivering it — otherwise a stale pose is
        laundered into a fresh-looking median every packet."""
        rig = R.GazeRig(["a"])
        rig.ingest({"a": _det((0.50, 0.0, 0.05), 2, stamp=10.0)}, 10.0)
        stale_by = OP.CUBE_LATCH_WINDOW_S + 0.05
        self.assertIsNone(
            rig.median("a", 10.0 + stale_by),
            "a decode past the latch window still answered the tracker")

    # --- the low-evidence stabilizer (user 2026-08-10) ---------------------

    def test_a_one_face_stream_answers_with_its_stable_centre_not_its_slide(self):
        """The e-cube T09 run: 1-face PnP drifted 44-51 mm and the fresh 0.7 s
        median rode the slide, dragging the MPC goal 40 mm off its anchor. With
        only 1-face evidence in the fresh window the median must widen to
        LOWEV_STABLE_WINDOW_S and answer the stream's central position — the
        drifted tail alone must NOT be the answer."""
        rig = R.GazeRig(["a"])
        # 2.4 s of 1-face samples at 10 Hz: y oscillates around 0, then the
        # last 0.6 s slides to +0.06 (the ghost class, inside the leash).
        t0 = 10.0
        stamps = [t0 + 0.1 * k for k in range(25)]
        for k, t in enumerate(stamps):
            y = 0.06 if t > stamps[-1] - 0.6 else (0.005 if k % 2 else -0.005)
            rig.ingest({"a": _det((0.50, y, 0.05), 1, stamp=t)}, t)
        med = rig.median("a", stamps[-1])
        self.assertIsNotNone(med)
        self.assertLess(
            abs(float(med[1])), 0.02,
            f"the median rode the 1-face slide to y={float(med[1]):.3f} — the "
            f"stable window did not widen (deque too shallow, or the "
            f"stabilizer branch is gone)")

    def test_an_old_two_face_decode_outranks_a_fresh_one_face_stream(self):
        """Best evidence wins, extended in time: a 2-face fix from 2 s ago is
        worth more than a whole fresh 1-face stream, and faces_used must say
        so — the goal gates key off THAT count."""
        rig = R.GazeRig(["a"])
        rig.ingest({"a": _det((0.50, 0.0, 0.05), 2, stamp=10.0)}, 10.0)
        for k in range(1, 20):
            t = 10.0 + 0.1 * k
            rig.ingest({"a": _det((0.56, 0.02, 0.05), 1, stamp=t)}, t)
        now = 10.0 + 0.1 * 19
        self.assertEqual(rig.faces_used("a", now), 2)
        np.testing.assert_allclose(
            rig.median("a", now), [0.50, 0.0, 0.05], atol=1e-9,
            err_msg="the 1-face stream outvoted the better-evidenced fix")

    def test_a_walk_era_sample_never_reaches_the_stable_window(self):
        """THE 08-10 night bug, first hardware run of the stabilizer:
        sightings are BASE-FRAME, so a 2-face decode captured mid-walk
        describes the cube relative to a robot position that no longer exists
        (~0.8 m of bias at walk speed) — and 'best evidence wins' handed it
        the whole window, aiming the grasp very far off the live cube. Any
        sample older than the base-motion epoch must be fenced out."""
        rig = R.GazeRig(["a"])
        # mid-walk: a clean 2-face fix, robot still far away
        rig.ingest({"a": _det((1.30, 0.10, 0.05), 2, stamp=10.0)}, 10.0)
        rig._base_moved_t = 10.5          # the walk kept driving until 10.5
        for k in range(6, 16):            # settled: fresh 1-face stream
            t = 10.0 + 0.1 * k
            rig.ingest({"a": _det((0.55, 0.02, 0.05), 1, stamp=t)}, t)
        now = 11.5
        self.assertEqual(rig.faces_used("a", now), 1,
                         "a walk-era 2-face fix outranked the settled stream")
        np.testing.assert_allclose(
            rig.median("a", now), [0.55, 0.02, 0.05], atol=1e-9,
            err_msg="the grasp target came from a frame the walk destroyed")

    def test_the_stabilizer_is_inert_while_the_base_moves(self):
        """While the epoch is being stamped every tick (the walk loop), the
        wide window is empty and the median must fall back to the fresh
        window — the exact pre-stabilizer semantics."""
        rig = R.GazeRig(["a"])
        for k in range(20):
            t = 10.0 + 0.1 * k
            rig.ingest({"a": _det((0.90 - 0.02 * k, 0.0, 0.05), 1, stamp=t)}, t)
        now = 10.0 + 0.1 * 19
        rig._base_moved_t = now           # driving right now
        med = rig.median("a", now)
        self.assertIsNotNone(med, "walking blinded the tracker entirely")
        fresh = [0.90 - 0.02 * k for k in range(20)
                 if now - (10.0 + 0.1 * k) <= OP.CUBE_LATCH_WINDOW_S]
        np.testing.assert_allclose(float(med[0]), np.median(fresh), atol=1e-9,
                                   err_msg="a moving base used the wide window")

    def test_median_evidence_measures_agreement_not_luck(self):
        """08-12 ('a waypoint into empty space'): the target-adoption gate
        reads (spread, n) — a jumping 1-face stream mid-flip must report a
        FAT spread even when its median lands somewhere plausible, and a
        tight stream must report a thin one."""
        rig = R.GazeRig(["a"])
        for k, y in enumerate((0.0, 0.09, -0.01, 0.10)):     # flipping fix
            rig.ingest({"a": _det((0.50, y, 0.05), 1, stamp=10.0 + 0.1 * k)},
                       10.0 + 0.1 * k)
        spread, n = rig.median_evidence("a", 10.3)
        self.assertEqual(n, 4)
        self.assertGreater(spread, R.TARGET_STABLE_SPREAD_M,
                           "a flipping stream passed the stability gate")
        rig2 = R.GazeRig(["a"])
        for k in range(4):                                    # tight fix
            rig2.ingest({"a": _det((0.50, 0.001 * k, 0.05), 2,
                                   stamp=10.0 + 0.1 * k)}, 10.0 + 0.1 * k)
        spread2, n2 = rig2.median_evidence("a", 10.3)
        self.assertEqual(n2, 4)
        self.assertLessEqual(spread2, R.TARGET_STABLE_SPREAD_M,
                             "an agreeing stream was held at the gate")

    def test_the_stabilizer_never_resurrects_a_dark_cube(self):
        """The widening fires only when FRESH samples exist. A cube unseen for
        the whole latch window must stay None — blind_since, the blind-grasp
        bar and the 20 s re-see hold all key off that None."""
        rig = R.GazeRig(["a"])
        for k in range(10):
            t = 10.0 + 0.1 * k
            rig.ingest({"a": _det((0.50, 0.0, 0.05), 1, stamp=t)}, t)
        dark = 10.9 + OP.CUBE_LATCH_WINDOW_S + 0.05
        self.assertLess(dark - 10.9, R.LOWEV_STABLE_WINDOW_S,
                        "test premise: the stable window would still hold "
                        "these samples if the guard were missing")
        self.assertIsNone(rig.median("a", dark),
                          "a dark cube answered from the stable window — "
                          "every blind-grasp path just lost its meaning")


if __name__ == "__main__":
    unittest.main()
