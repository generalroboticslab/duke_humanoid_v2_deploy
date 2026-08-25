"""Pins for the 08-13 cube-orientation plumbing (Sighting.quat -> GazeRig.cube_yaw).

The incident it encodes: the monitor has always published `quat_wxyz` beside
`pos` for every detected body, and the cube track dropped it — so the scene
cuboid was written axis-aligned and the grasp goalset (generated in the cube's
frame, composed with obj_quat) was rotated by nothing. The operator is told to
yaw cubes 30-45 deg so a second tag face decodes; a 60 mm cube at 45 deg spans
85 mm across its diagonal against a 90 mm jaw opening, so "good for perception"
was quietly "corner-to-corner for the gripper".

Pinned here: the yaw-only reduction, the modulo-90 fold (the cube's 4-fold
vertical symmetry, which the server's candidate set already spans), the medoid's
outlier rejection, and the >=2-face evidence gate that makes a missing or noisy
measurement degrade to exactly the old identity behaviour.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOT importing humanoid_real_env here on purpose: nothing below needs it.
# (2026-08-13 it was also a loader error under discovery — its `common.publisher`
# import could be shadowed by visual_servoing's own top-level `common/` once an
# earlier test module had put that on sys.path; 2026-08-22 the control-side
# package became `ipc`, so that second reason is gone.)
import humanoid_curobo_reach as R  # noqa: E402
from auto_operator.models import Sighting  # noqa: E402


def yaw_quat(deg):
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


class _Cube:
    def __init__(self, hist):
        self.hist = hist

    def confirmed(self, now):
        return True


def _rig(**cubes):
    rig = R.GazeRig.__new__(R.GazeRig)
    rig.cubes = {}
    rig._base_moved_t = -1e9
    now = 100.0
    for name, (degs, faces) in cubes.items():
        rig.cubes[name] = _Cube([
            Sighting(t=now - 0.05, pos=np.zeros(3), port=1, faces=faces,
                     quat=(None if d is None else yaw_quat(d)))
            for d in degs])
    return rig, now


class CubeYawTests(unittest.TestCase):

    def test_a_well_evidenced_yaw_is_measured(self):
        rig, now = _rig(c=((29.0, 30.0, 31.0), 3))
        self.assertAlmostEqual(math.degrees(rig.cube_yaw("c", now)), 30.0,
                               places=6)

    def test_the_fold_is_modulo_90_degrees(self):
        """The server's candidate set already spans yaws 0/90/180/270 about
        the cube +Z (4-fold symmetry), so only the remainder is missing —
        and folding removes the jump when the visible face changes identity
        between decodes."""
        rig, now = _rig(c=((124.0, 125.0, 126.0), 3))
        self.assertAlmostEqual(math.degrees(rig.cube_yaw("c", now)), 35.0,
                               places=6)

    def test_the_fold_does_not_wrap_near_its_own_edge(self):
        rig, now = _rig(c=((89.0, 89.0, 89.0), 3))
        self.assertAlmostEqual(math.degrees(rig.cube_yaw("c", now)), 89.0,
                               places=6)

    def test_one_face_evidence_refuses_to_answer(self):
        """The gate that makes this safe: a wrong yaw ROTATES the approach,
        which the jaws cannot recover from, while no yaw merely reproduces
        today's identity behaviour. Same bar as the 1-face z snap."""
        rig, now = _rig(c=((30.0, 30.0, 30.0), 1))
        self.assertIsNone(rig.cube_yaw("c", now))

    def test_a_publisher_without_quaternions_degrades_silently(self):
        rig, now = _rig(c=((None, None, None), 3))
        self.assertIsNone(rig.cube_yaw("c", now))

    def test_the_medoid_ignores_a_wild_sample(self):
        """StaticPoseTrack chose a medoid over a mean for the tables after a
        hardware window spread 78.5 deg; the same reasoning applies here, and
        a mean of (40, 41, 40.5, 88) would sit ~52 deg — a quarter turn off
        the faces."""
        rig, now = _rig(c=((40.0, 41.0, 40.5, 88.0), 3))
        got = math.degrees(rig.cube_yaw("c", now))
        self.assertLess(abs(got - 40.5), 1.5,
                        f"an outlier dragged the yaw to {got:.1f} deg")

    def test_a_cube_whose_x_axis_points_up_has_no_yaw(self):
        """Degenerate fit: yaw about the vertical is undefined, and inventing
        one would aim the approach at random."""
        h = math.radians(90.0) / 2.0
        pitch = np.array([math.cos(h), 0.0, math.sin(h), 0.0])  # +x -> -z
        rig = R.GazeRig.__new__(R.GazeRig)
        rig.cubes = {"c": _Cube([Sighting(t=99.95, pos=np.zeros(3), port=1,
                                          faces=3, quat=pitch)] * 3)}
        rig._base_moved_t = -1e9
        self.assertIsNone(rig.cube_yaw("c", 100.0))


class SightingSchemaTests(unittest.TestCase):

    def test_quat_is_optional_so_every_old_construction_site_still_works(self):
        s = Sighting(t=1.0, pos=np.zeros(3), port=1, faces=2)
        self.assertIsNone(s.quat)

    def test_the_ingest_validates_the_quaternion_like_it_validates_pos(self):
        """A malformed or non-unit quaternion must never enter a track: it
        would rotate a grasp goalset, not merely offset a position."""
        from auto_operator import perception

        from humanoid_auto_operator import CubeTrack

        class _Op:
            def __init__(self):
                self.cubes = {"k": CubeTrack(key="k")}
                self._cam_retired = set()      # perception.ingest reads it
                self._held = {}
        # perception._K(op) resolves tuning constants from the module the
        # operator CLASS lives in, so a fake declared here would send it
        # looking for SERVO_GRIPPER_KEYS in the test module.
        _Op.__module__ = "humanoid_auto_operator"

        for bad in ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0], "nope",
                    [float("nan"), 0.0, 0.0, 1.0]):
            op = _Op()
            perception.ingest(op, {"k": {"pos": [0.1, 0.2, 0.3], "n_inliers": 3,
                                         "camera_port": 5555, "capture_stamp": 1.0,
                                         "quat_wxyz": bad}}, 1.0)
            hist = op.cubes["k"].hist
            self.assertTrue(hist, f"the sighting itself was dropped for {bad}")
            self.assertIsNone(hist[-1].quat,
                              f"a malformed quaternion {bad} entered the track")

    def test_a_good_quaternion_survives_ingest_normalised(self):
        from auto_operator import perception
        from humanoid_auto_operator import CubeTrack

        class _Op:
            def __init__(self):
                self.cubes = {"k": CubeTrack(key="k")}
                self._cam_retired = set()      # perception.ingest reads it
                self._held = {}
        # perception._K(op) resolves tuning constants from the module the
        # operator CLASS lives in, so a fake declared here would send it
        # looking for SERVO_GRIPPER_KEYS in the test module.
        _Op.__module__ = "humanoid_auto_operator"

        op = _Op()
        q = yaw_quat(30.0) * 3.0                      # deliberately un-normalised
        perception.ingest(op, {"k": {"pos": [0.1, 0.2, 0.3], "n_inliers": 3,
                                     "camera_port": 5555, "capture_stamp": 1.0,
                                     "quat_wxyz": q.tolist()}}, 1.0)
        got = op.cubes["k"].hist[-1].quat
        self.assertIsNotNone(got)
        self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
