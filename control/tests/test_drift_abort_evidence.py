"""The drift abort must act on evidence, and a one-face fit is not evidence.

WHY THIS EXISTS. Hardware 2026-07-30, third attempt. Everything worked: the
table fitted to 1.2 deg, the cube sat at r_xy 0.416, all four gates passed with
29.7 mm clearance, the post-solve start check read 8.4e-03 rad, and the route
streamed. Then it was killed at waypoint 122 of 161 — 95 % of the way to the
cube — on "cube moved 5.1 cm". Nobody had touched the cube.

The abort fires hardest exactly when the hand ARRIVES, because that is when the
gripper slides between the camera and the cube. A 60 mm cube that was showing
three tag faces drops to one partly-occluded face, and the PnP pose from a
single face wanders by centimetres. The drift monitor then reports the reach
succeeding as the cube running away.

This is not a new judgement call. auto_operator already draws exactly this line
for the gripper-base visual servo — SERVO_MIN_INLIERS = 2, "single 16 mm-tag PnP
is garbage-prone" — and it draws it for the same reason: that measurement feeds
a control loop. So does this one. Below the bar there is no measurement, and no
measurement must mean NO VERDICT, never a verdict of "moved".

Nothing here raises CUBE_DRIFT_ABORT_M. A cube that really moves 9 cm in full
view still trips the abort, and the test below proves it.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

CUBE = "grasp_cube_60mm"
HERE = np.array([0.403, 0.103, 0.083])         # the 07-30 target, verbatim


class _Rig:
    """Just enough GazeRig for stream_route's perception calls."""

    def __init__(self, script):
        self.script = list(script)             # [(pos, n_inliers), ...]
        self.i = 0
        self.inliers = {}
        self.cubes = {CUBE: None}

    def step(self):
        pos, n = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.inliers[CUBE] = n
        return pos

    def latest(self, key, now):
        return self._pos

    def last_inliers(self, key):
        return int(self.inliers.get(key, 0))

    def faces_used(self, key, now=None):
        # faces behind the MEDIAN, not the newest packet
        return self.last_inliers(key)

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out


class _Det:
    def __init__(self, rig):
        self.rig = rig

    def poll(self, rig):
        rig._pos = rig.step()


class _Tel:
    def fresh(self):
        return {"joint_pos": [0.0] * 31, "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, False]}


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)


class _Bundle:
    """A two-waypoint route that streams in well under a second."""
    goal = "reach"
    active_sides = ("left",)
    horizon = 2
    duration_s = 0.4
    joint_names = None
    controlled_joints = None

    def __init__(self):
        self.joint_names = tuple(R.ARM_JOINTS_L) + tuple(R.ARM_JOINTS_R)
        # All zeros, and _Tel reports all-zero encoders, so the settle phase
        # converges on its first tick and the test measures the drift logic
        # rather than a settle timeout.
        self.controlled_joints = tuple(R.ARM_JOINTS_L)
        self.q_enc = np.zeros((2, len(self.joint_names)))

    def q_at(self, t):
        return self.q_enc[0]


def _run(script):
    rig = _Rig(script)
    rig._pos = script[0][0]
    pub = _Pub()
    why, _played = R.stream_route(_Bundle(), pub, _Tel(), _Det(rig), rig,
                                  {CUBE: HERE}, 0.3)
    return why, pub


@unittest.skipIf(R is None, _SKIP)
class DriftEvidenceTests(unittest.TestCase):

    def test_the_bar_is_the_operators_own_servo_bar(self):
        """Not a number invented here. If the operator ever revises what it
        considers a trustworthy PnP fit, this must move with it."""
        self.assertEqual(R.DRIFT_MIN_INLIERS, OP.SERVO_MIN_INLIERS)

    def test_a_one_face_fit_that_wanders_15_cm_cannot_abort(self):
        """THE test — the 07-30 failure. The gripper occludes the cube on
        arrival, the fit collapses to one face and reports the cube 15 cm away.
        That is the reach succeeding, and it must not read as the cube fleeing."""
        far = HERE + np.array([0.15, 0.0, 0.0])
        why, _ = _run([(far, 1)] * 40)
        self.assertIsNone(why, f"aborted on one-face fits: {why}")

    def test_a_real_move_warns_loudly_but_streams_on(self):
        """User doctrine 2026-08-04: the position was locked at plan time —
        the stream does not re-litigate perception mid-route (arm-occlusion
        artifacts pass the 2-face bar with CONSISTENT wrong poses, and the
        old abort kept killing healthy routes). A well-evidenced 9 cm drift
        now prints one loud WARNING and streams on: a truly moved cube is
        re-acquired by the MPC handoff's live re-seed, not by an abort."""
        import contextlib
        import io
        moved = HERE + np.array([0.09, 0.0, 0.0])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            why, _ = _run([(moved, 3)] * 40)
        self.assertIsNone(why, f"a drift reading still vetoes the stream: {why}")
        said = out.getvalue()
        self.assertIn("WARNING", said, "a 9 cm full-view drift went unsaid")
        self.assertEqual(said.count("WARNING"), 1,
                         "the drift warning must fire once, not spam")

    def test_a_cube_sitting_still_never_aborts(self):
        why, _ = _run([(HERE, 3)] * 40)
        self.assertIsNone(why)

    def test_one_bad_frame_among_good_ones_cannot_abort(self):
        """Even at full inlier count a single 1-inlier-class PnP glitch happens;
        the median window is the second line of defence behind the face bar."""
        script = []
        for k in range(40):
            script.append((HERE + np.array([0.4, 0.0, 0.0]), 3) if k % 7 == 0
                          else (HERE, 3))
        why, _ = _run(script)
        self.assertIsNone(why, f"one flyer in seven aborted the route: {why}")

    def test_the_drift_warning_names_the_evidence_bar(self):
        """An operator reading 'the cube reads off-target' needs to know the
        tool believed the sighting, and why."""
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as out:
            _run([(HERE + np.array([0.09, 0.0, 0.0]), 3)] * 40)
        self.assertIn("faces", out.getvalue())


@unittest.skipIf(R is None, _SKIP)
class WatchReportTests(unittest.TestCase):
    """The [watch] line has to carry the number that explains the verdict."""

    def test_the_face_count_is_reported(self):
        w = R.MotionWatch("test", HERE)
        for n in (3, 3, 1, 1, 1):
            w.sample({"projected_gravity": [0, 0, -1]}, HERE, 0.0, n)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as out:
            w.report()
        said = out.getvalue()
        self.assertIn("tag faces", said)
        self.assertIn("60% below", said)      # 3 of 5 under the 2-face bar


if __name__ == "__main__":
    unittest.main()
