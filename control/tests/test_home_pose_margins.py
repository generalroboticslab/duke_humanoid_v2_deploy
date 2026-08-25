"""The two PHYSICAL facts about the home posture, measured on the deploy model.

WHY THIS FILE EXISTS. The 07-31 one-home migration moved the arms' home to the
simulator's HUMANOID_ARM_JOINT_HOME so that sim, plan server, power-on and
chest home are ONE posture (the user's spec). Everything else about that change
is checkable by identity — the constants either equal the sim source or they do
not, and test_model_derived_constants pins that. These two are different: they
are properties of where the new posture puts the ARMS IN SPACE, they decide
whether the robot damages itself or fails to see, and neither can be derived
from the constants by inspection.

  1. CLEARANCE, and its SENSITIVITY. The new home folds the hands at the chest
     (r_xy 0.27, z 0.26, elbow 111 deg) and its static floor is 39.5 mm — the
     right gripper base against the torso. That is comfortably positive, and it
     is also the WHOLE margin: unlike the 07-30 chest home, whose 51 mm floor
     came from a structurally-adjacent pair and did not move with posture, this
     floor is posture-SENSITIVE. Monte-Carlo (uniform sag on all 14 joints,
     worst of 600): +/-3 deg -> 21.5 mm, +/-5 deg -> ~10 mm, +/-8 deg ->
     PENETRATION. SIDE_HOME_EE's own comment argues from hardware that 30 mm
     dies to sag and 47 mm survives, so 39.5 mm is inside the band that
     comment calls decidable-by-hardware-asymmetry.

     These tests do NOT assert the margin is safe — that is a hardware
     question the user owns, recorded in CHEST_HOME_JOINTS' comment and to be
     watched on the first runs. They assert the number has not SILENTLY MOVED.
     A later home edit that quietly halves it is the failure this catches.

  2. SIGHTLINES. Tag detection is the mission's first gate, and the arms park
     inside their own cameras' field of view. Ray-tested against cubes at
     r_xy 0.45 on both sides, the new home leaves all four camera->cube lines
     clear, while the 07-30 chest home put the gripper racks 38-66 mm off-axis
     at 0.56-0.65 m — squarely in the path. This is the migration's clearest
     physical WIN and it should not be given back unnoticed.

Skipped, not failed, without MuJoCo or the deploy model (a gitignored build
artifact — see rebuild_deploy_model.py).
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import mujoco

    import humanoid_auto_operator as OP
    from humanoid_curobo_client import ARM_JOINTS_ALL, _FrameShim
    from humanoid_joint_monkey_hw import preflight
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _MODEL = None
    _SKIP = f"deploy model unavailable ({type(exc).__name__}: {exc})"

STEMS = ("shoulder_1", "shoulder_2", "shoulder_3", "elbow",
         "wrist_1", "wrist_2", "wrist_3")

# Measured 2026-07-31 on the deploy model. Bounds, not equalities: the point is
# to catch a home edit that moves these, not to re-assert MuJoCo's arithmetic.
STATIC_CLEARANCE_M = 0.0395
CLEARANCE_FLOOR_M = 0.030        # below this the 07-24 hardware argument says
                                 # sag decides the outcome — a home that lands
                                 # here needs a hardware decision, not a test
CUBE_R_XY_M = 0.45               # a typical grasp target (the 0.30-0.50 sweet
                                 # spot the operator notes recommend)
SIGHTLINE_CLEAR_M = 0.09         # a geom centre closer than this to the
                                 # camera->cube axis is treated as in the way


def _both_arms(left7) -> np.ndarray:
    return np.concatenate([np.asarray(left7, float),
                           np.asarray(OP.mirror_arm(list(left7), "right"), float)])


@unittest.skipIf(_MODEL is None, _SKIP)
class ClearanceTests(unittest.TestCase):

    def setUp(self):
        self.data = mujoco.MjData(_MODEL)
        self.qadr = [_MODEL.jnt_qposadr[mujoco.mj_name2id(
            _MODEL, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS_ALL]

    def _clearance(self, left7) -> float:
        worst, _where = preflight(_MODEL, self.data,
                                  _FrameShim([_both_arms(left7)]), self.qadr)
        return float(worst)

    def test_the_home_posture_does_not_touch_itself(self):
        """The floor, as a number. Uses the SAME auditor as Gate C, so a home
        the route gate would reject cannot be the home routes start from."""
        got = self._clearance(OP.POWERON_JOINTS)
        self.assertGreater(
            got, CLEARANCE_FLOOR_M,
            f"the home posture's self-clearance is {got*1000:.1f} mm — below "
            f"{CLEARANCE_FLOOR_M*1000:.0f} mm the 07-24 hardware finding says "
            f"gravity sag and encoder offset decide the outcome; this needs a "
            f"hardware decision (re-solve the sim home wider, upstream), not a "
            f"looser test")
        self.assertAlmostEqual(
            got, STATIC_CLEARANCE_M, delta=0.002,
            msg=f"the home's self-clearance moved to {got*1000:.1f} mm from "
                f"the recorded {STATIC_CLEARANCE_M*1000:.1f} mm — if the home "
                f"was edited deliberately, re-run the sag study in "
                f"CHEST_HOME_JOINTS' comment and update this constant WITH the "
                f"new Monte-Carlo numbers")

    def test_the_glide_in_never_dips_below_the_endpoint(self):
        """Every recognised start glides to the home along a joint lerp. The
        07-31 audit found all of them endpoint-bound — the parked home is
        itself the tightest frame — which is what makes ONE recorded number
        cover the whole approach. If a path ever dips below its endpoint that
        reasoning breaks and the corridor needs re-auditing per segment."""
        starts = {
            "side_home": OP.SIDE_HOME_JOINTS,
            "front_station": OP.STAGE_JOINTS["front"],
            "chest_home_0730": OP.POWERON_JOINTS_LEGACY_0730,
            "poweron_v83": OP.POWERON_JOINTS_V83,
        }
        end = np.asarray(OP.POWERON_JOINTS, float)
        for name, start in starts.items():
            with self.subTest(start=name):
                a, b = _both_arms(start), _both_arms(end)
                frames = [a + (b - a) * t for t in np.linspace(0.0, 1.0, 61)]
                worst, where = preflight(_MODEL, self.data,
                                         _FrameShim(frames), self.qadr)
                self.assertGreaterEqual(
                    float(worst), STATIC_CLEARANCE_M - 0.002,
                    f"the {name} -> home glide dips to {worst*1000:.1f} mm at "
                    f"{where}, below the endpoint's own {STATIC_CLEARANCE_M*1000:.1f} "
                    f"mm — this path needs its own audit")

    def test_a_folded_home_is_tighter_than_the_open_one_it_replaced(self):
        """Pins the DIRECTION of the trade the user accepted, so nobody later
        reads the margin as unchanged. The 07-30 chest home sat at 51 mm; this
        one is deliberately tighter in exchange for sim identity."""
        self.assertLess(self._clearance(OP.POWERON_JOINTS),
                        self._clearance(OP.POWERON_JOINTS_LEGACY_0730))


@unittest.skipIf(_MODEL is None, _SKIP)
class SightlineTests(unittest.TestCase):
    """The arms must not park in front of the cameras that find the cubes."""

    def setUp(self):
        self.data = mujoco.MjData(_MODEL)
        self.hand_geoms = []
        for g in range(_MODEL.ngeom):
            n = mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "collision" in n and any(k in n for k in
                                        ("rack", "gripper", "wrist",
                                         "R_base", "L_base")):
                self.hand_geoms.append((g, n))

    def _pose(self, left7) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        for arm in ("left", "right"):
            for v, stem in zip(OP.mirror_arm(list(left7), arm), STEMS):
                jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{arm}_{stem}_joint")
                self.data.qpos[_MODEL.jnt_qposadr[jid]] = float(v)
        mujoco.mj_forward(_MODEL, self.data)

    def _blockers(self, cam: str, target: np.ndarray) -> list[str]:
        sid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE, cam)
        eye = self.data.site_xpos[sid].copy()
        d = target - eye
        length = float(np.linalg.norm(d))
        d = d / length
        out = []
        for g, name in self.hand_geoms:
            p = self.data.geom_xpos[g] - eye
            t = float(np.dot(p, d))
            if 0.02 < t < length and \
                    float(np.linalg.norm(p - t * d)) < SIGHTLINE_CLEAR_M:
                out.append(name)
        return out

    def test_neither_camera_is_blocked_by_a_parked_hand(self):
        """THE test. All four camera->cube lines, both sides. A home that
        blocks its own perception fails the mission at its first gate, and the
        symptom is 'no cube confirmed' — which reads as a lighting or tag
        problem, not as the arms being in the way."""
        self._pose(OP.POWERON_JOINTS)
        for cam in ("cam_left_rgb", "cam_right_rgb"):
            for label, y in (("left_cube", +0.20), ("right_cube", -0.20)):
                with self.subTest(cam=cam, cube=label):
                    blockers = self._blockers(
                        cam, np.array([CUBE_R_XY_M, y, 0.05]))
                    self.assertEqual(
                        blockers, [],
                        f"{cam} -> {label} is blocked by {blockers} at the "
                        f"home posture")

    def test_this_is_a_real_improvement_and_the_check_can_fail(self):
        """Guards the guard. The test above only means something if this
        geometry CAN detect a blocked line — and the 07-30 chest home is a
        posture that genuinely blocked all four. If this ever stops finding
        blockers, the detector has gone blind and the test above is vacuous."""
        self._pose(OP.POWERON_JOINTS_LEGACY_0730)
        blocked = sum(
            1 for cam in ("cam_left_rgb", "cam_right_rgb")
            for y in (+0.20, -0.20)
            if self._blockers(cam, np.array([CUBE_R_XY_M, y, 0.05])))
        self.assertGreater(
            blocked, 0,
            "the 07-30 chest home no longer registers as blocking any "
            "sightline — the detector above cannot fail, so it proves nothing")


if __name__ == "__main__":
    unittest.main()
