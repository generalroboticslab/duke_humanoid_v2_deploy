"""Guards the operator constants that are FK PRODUCTS of the deploy model.

WHY THIS FILE EXISTS SEPARATELY from test_independent_arms.py: that module is
deliberately I/O-free — it mirrors the production constants locally and never
imports humanoid_auto_operator, so the whole state machine can be exercised on
a machine with no model, no MuJoCo and no robot. This module is the opposite by
necessity: its entire job is to compare the committed constants against the
model that is actually loaded, so it must import both.

THE FAILURE CLASS IT CLOSES. Operator constants fall into four groups by how
they depend on the model:

  A. read from the model at runtime (IK, collision pairs, joint limits, gaze FK)
     — migrate for free, nothing to guard;
  B. authored in JOINT space (STAGE_JOINTS, SIDE_HOME_JOINTS, POWERON_JOINTS)
     — survive a geometry change by construction, the arm still forms those
     configurations;
  C. FK PRODUCTS of group B (REST_EE, JOURNEY_WALK_EE, STATION_BEARING_DEG)
     — these go stale the moment the model changes, and they go stale SILENTLY:
     nothing times out, no gate refuses, the staircase still reports "settled";
  D. empirical/hardware (hand-eye, cable mass) — a model swap does not touch them.

Group C is the only one that needs a human to re-derive, and 07-28 proved a
human forgets: the new-wrist migration re-derived REST_EE and JOURNEY_WALK_EE
but left STATION_BEARING_DEG at its old-wrist values (front off by 9.3 deg),
found only by a later code read. This test makes that impossible to repeat.

DELIBERATE OFFSETS — do not "fix" these to match FK:
  SIDE_HOME_EE is NOT an FK product. Its z is deliberately raised 0.10 m above
  FK(SIDE_HOME_JOINTS) (07-24), and is therefore not guarded at all.
  CHEST_HOME_EE stopped being one of these on 07-31: it is now a PLAIN FK
  product of CHEST_HOME_JOINTS (per side, no offset), because the joints come
  first — they are the simulator's home, imported verbatim, and the EE values
  merely record where that posture puts the hands. Guarded as an identity.

ONE GROUP-B CONSTANT IS ALSO PINNED TO AN EXTERNAL SOURCE. POWERON_JOINTS /
CHEST_HOME_JOINTS are joint-space and would survive a geometry change, but
since 07-31 they carry a stronger promise: joint-for-joint identity with the
simulator's HUMANOID_ARM_JOINT_HOME through the units seam. That is the user's
spec ("the power-on position must be the same as the home pose in the
simulator"), it is what makes the plan
server's planning_home, the robot's power-on and the chest home ONE posture,
and it is guarded below against the sim-side source file itself.

Skipped, not failed, when MuJoCo or the deploy model is absent — the deploy
model is a gitignored build artifact (see rebuild_deploy_model.py), so a fresh
clone legitimately has no model to check against.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    import humanoid_auto_operator as OP
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:                      # noqa: BLE001 — any of a dozen causes
    _MODEL = None
    _SKIP = f"deploy model unavailable ({type(exc).__name__}: {exc})"

# Tolerances. Position ones are set an order of magnitude below anything the
# mission reasons about (REACH_OK_M 0.05 m, GRASP_EE_OK_M 0.06 m) so a real
# re-derivation miss trips this long before it perturbs behaviour.
POS_TOL_M = 0.001
BEARING_TOL_DEG = 0.5
# Left-arm joints in the order every stored posture in the operator uses.
_ARM_JOINTS_L = ("left_shoulder_1_joint", "left_shoulder_2_joint",
                 "left_shoulder_3_joint", "left_elbow_joint",
                 "left_wrist_1_joint", "left_wrist_2_joint",
                 "left_wrist_3_joint")
# The left/right shoulder mounts are not perfectly symmetric in the CAD, so a
# perfect joint mirror still lands 1.92 mm off a perfect Cartesian mirror. The
# value is CONSTANT across every posture (verified 07-28) — that constancy is
# the actual invariant, and it is asserted below. This bound just has to sit
# above the fixed offset and far below any kinematic break.
MIRROR_TOL_M = 0.003

ARM_JOINT_STEMS = ("shoulder_1", "shoulder_2", "shoulder_3",
                   "elbow", "wrist_1", "wrist_2", "wrist_3")


def _ee_fk(q7, arm: str) -> np.ndarray:
    """Base-frame EE-site position for a 7-joint posture, in ENCODER units.

    qpos is zeroed and then filled with the posture verbatim rather than being
    seeded from qpos0, because the deploy model carries a `ref` on wrist_1 (see
    rebuild_deploy_model.py layer 3): qpos0 is NOT the zero vector, and every
    constant in the operator is expressed in encoder units, where the model's
    qpos 0 is the hardware's 0. Seeding from qpos0 would silently add 90 deg.
    """
    data = mujoco.MjData(_MODEL)
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0                        # free-joint quaternion w
    side_prefix = "left" if arm == "left" else "right"
    for value, stem in zip(q7, ARM_JOINT_STEMS):
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{side_prefix}_{stem}_joint")
        data.qpos[_MODEL.jnt_qposadr[jid]] = float(value)
    mujoco.mj_forward(_MODEL, data)
    sid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE,
                            f"end_effector_{'L' if arm == 'left' else 'R'}_site")
    return data.site_xpos[sid].copy()


@unittest.skipIf(_MODEL is None, _SKIP)
class ModelDerivedConstantTests(unittest.TestCase):
    """Every group-C constant, checked against the loaded model."""

    def test_station_bearings_match_the_loaded_model(self):
        """STATION_BEARING_DEG drives lift_arc's sweep DIRECTION.

        lift_arc picks direction from sign(station_bearing - current_bearing),
        so a stale station bearing does not merely shorten the clearance arc —
        for a cube whose bearing falls between the stale and true values it
        arcs the wrong way. That is why 0.5 deg, not "close enough".
        """
        for name, posture in OP.STAGE_JOINTS.items():
            with self.subTest(station=name):
                ee = _ee_fk(posture, "left")
                actual = math.degrees(math.atan2(ee[1], ee[0]))
                self.assertAlmostEqual(
                    actual, OP.STATION_BEARING_DEG[name], delta=BEARING_TOL_DEG,
                    msg=(f"STATION_BEARING_DEG['{name}'] is "
                         f"{OP.STATION_BEARING_DEG[name]:.2f} deg but the model "
                         f"says {actual:.2f} deg — re-derive it (the recipe is "
                         f"in the constant's comment)"))

    def test_station_bearings_stay_interior_to_their_sector_bands(self):
        """The property lift_arc relies on: a full sweep cannot leave its sector."""
        bands = {"front": (0.0, OP.SECTOR_FRONT_DEG),
                 "side": (OP.SECTOR_FRONT_DEG, OP.SECTOR_REAR_DEG),
                 "rear": (OP.SECTOR_REAR_DEG, 180.0)}
        for name, (lo, hi) in bands.items():
            with self.subTest(station=name):
                bearing = OP.STATION_BEARING_DEG[name]
                self.assertGreater(bearing, lo)
                self.assertLess(bearing, hi)

    def test_the_walking_tuck_is_fk_of_the_power_on_posture_again(self):
        """RE-UNIFIED 2026-07-31, and the history matters because this test
        has now asserted three different things in three days. Identical until
        07-30; split on 07-30 (power-on raised to the chest, tuck kept at the
        policy's low training default); re-unified on 07-31 by the one-home
        spec — and the split could not have survived it partially, because
        journey_arms_tucked's joint branch checks POWERON_JOINTS while its EE
        fallback checks THIS, and a 0.28 m disagreement means the tuck gate
        either never passes or passes at the wrong pose.

        So the pre-07-30 identity is restored: the walking tuck IS the FK of
        the power-on posture (the sim home). The cost — walking now carries
        the arms high-folded instead of at the low training default — is
        recorded at JOURNEY_WALK_EE and decided by the user, not by this test
        drifting. Guarded so the two cannot separate silently again.
        """
        poweron = _ee_fk(OP.POWERON_JOINTS, "left")
        walk = np.asarray(OP.JOURNEY_WALK_EE, float)
        np.testing.assert_allclose(
            walk, poweron, atol=POS_TOL_M,
            err_msg="JOURNEY_WALK_EE is no longer FK(POWERON_JOINTS) — the "
                    "07-31 one-home unification has come apart; if this is "
                    "deliberate, journey_arms_tucked's EE fallback and the "
                    "arbitration glide must be split from POWERON too")

    def test_the_power_on_posture_matches_the_deployed_env_config(self):
        """THE guard for 2026-07-30. POWERON_JOINTS is a MIRROR of the robot's
        own default_pose; the robot's copy is authoritative and this one exists
        so the tools can reason offline. If they drift, every tool that decides
        "is this arm at a known home?" is reasoning about a pose the robot does
        not hold — and the symptom is a refusal 30 s into a run, not an error
        here.

        Right arm = left negated, per joint, which is the convention every
        stored posture in the operator is authored in.
        """
        import yaml
        cfg_path = (Path(MJCF_MODEL_PATH).parent / "env_config.yaml")
        if not cfg_path.exists():
            self.skipTest(f"deployed env_config not readable at {cfg_path}")
        jp = yaml.safe_load(cfg_path.read_text())["default_pose"]["joint_pos"]
        catch_all = jp.get(".*", 0.0)
        for i, joint in enumerate(_ARM_JOINTS_L):
            with self.subTest(joint=joint):
                self.assertAlmostEqual(
                    jp.get(joint, catch_all), OP.POWERON_JOINTS[i], places=4,
                    msg=f"{joint}: env_config default_pose and POWERON_JOINTS "
                        f"disagree")
                right = joint.replace("left_", "right_")
                self.assertAlmostEqual(
                    jp.get(right, catch_all), -OP.POWERON_JOINTS[i], places=4,
                    msg=f"{right}: the right arm is not the left negated")

    def test_the_chest_home_and_the_power_on_posture_are_one_posture(self):
        """Spelled out twice — POWERON_JOINTS is defined further down the
        module than CHEST_HOME_JOINTS, so the latter cannot reference it — and
        duplication that nothing checks is duplication that rots."""
        np.testing.assert_allclose(
            np.asarray(OP.CHEST_HOME_JOINTS, float),
            np.asarray(OP.POWERON_JOINTS, float), atol=1e-9,
            err_msg="CHEST_HOME_JOINTS and POWERON_JOINTS have drifted apart")

    def test_chest_home_ee_is_fk_of_chest_home_joints_per_side(self):
        """The derivation INVERTED on 07-31: the joints are the source (the
        simulator's home, imported verbatim) and CHEST_HOME_EE records where
        that posture puts the hands — used by tools to RECOGNISE an arm parked
        at the home and by raise_to_chest_home as its Cartesian goal. Drift
        breaks the recognition silently, and the symptom is the 30-second
        failsafe crawl this constant was added to avoid.

        Both sides FK'd independently, because the model's arms are NOT
        perfect mirrors (~2 mm constant CAD offset, see MIRROR_TOL_M): the
        stored values are per-side FK truths, not one side mirrored, so a
        symmetrised edit would fail here by ~2 mm — that is intended.
        """
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                posture = OP.mirror_arm(OP.CHEST_HOME_JOINTS, arm)
                np.testing.assert_allclose(
                    np.asarray(OP.CHEST_HOME_EE[arm], float),
                    _ee_fk(posture, arm), atol=POS_TOL_M,
                    err_msg=f"CHEST_HOME_EE['{arm}'] is no longer "
                            f"FK(CHEST_HOME_JOINTS)")

    def test_chest_home_quat_is_the_fk_orientation_of_the_home(self):
        """CHEST_HOME_QUAT exists because the chest home stopped sharing the
        walking tuck's wrist orientation on 07-31 (they are 19.3 deg apart).
        raise_to_chest_home publishes it as the IK's orientation target; a
        stale value aims the ramp at a pose that is neither home, the same
        class of bug as the 07-21 NEUTRAL-quat finding.

        The right arm must be the left with x and z negated — the convention
        raise_to_chest_home and the arbitration glide both apply — and that is
        checked against the right site's own FK rather than assumed.
        """
        data = mujoco.MjData(_MODEL)
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        for arm in ("left", "right"):
            posture = OP.mirror_arm(OP.CHEST_HOME_JOINTS, arm)
            prefix = arm
            for value, stem in zip(posture, ARM_JOINT_STEMS):
                jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{prefix}_{stem}_joint")
                data.qpos[_MODEL.jnt_qposadr[jid]] = float(value)
        mujoco.mj_forward(_MODEL, data)
        want = {"left": np.asarray(OP.CHEST_HOME_QUAT, float)}
        wl = want["left"]
        want["right"] = np.array([wl[0], -wl[1], wl[2], -wl[3]])
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                sid = mujoco.mj_name2id(
                    _MODEL, mujoco.mjtObj.mjOBJ_SITE,
                    f"end_effector_{'L' if arm == 'left' else 'R'}_site")
                got = np.empty(4)
                mujoco.mju_mat2Quat(got, data.site_xmat[sid].flatten())
                angle = math.degrees(2.0 * math.acos(
                    min(1.0, abs(float(np.dot(got, want[arm]))))))
                self.assertLess(
                    angle, 0.5,
                    f"CHEST_HOME_QUAT[{arm}] is {angle:.2f} deg from the FK "
                    f"orientation of CHEST_HOME_JOINTS")

    def test_chest_home_joints_are_the_simulators_home_through_the_seam(self):
        """THE 07-31 invariant, guarded against the sim's own source file.

        The user's spec is joint-space identity: the robot powers on, plans
        from, and retracts to the SAME posture the simulator calls home
        (HUMANOID_ARM_JOINT_HOME), so the plan server's planning_home, the
        power-on pose and the chest home are one posture and the 07-28
        dynamics validation transfers. This test re-reads the sim constant
        from the local legged_env_v2 clone and re-folds it through the units
        seam (enc = SIGN*cspace + qpos0, the client's own convention) — if
        the sim home is ever re-solved, this goes red and says to re-run the
        migration, instead of the two sides silently working from different
        postures again.

        Parsed with ast rather than imported: the module imports cuRobo.
        """
        import ast as _ast
        from humanoid_site import LEGGED_ENV_ROOT
        src = LEGGED_ENV_ROOT / "mj_envs/tasks/visual_manipulation/curobo/ik_curobo_robot_cfg.py"
        if not src.exists():
            self.skipTest(f"sim source not present at {src}")
        sim: dict[str, float] = {}
        for node in _ast.walk(_ast.parse(src.read_text())):
            if isinstance(node, _ast.Assign) and any(
                    isinstance(t, _ast.Name) and
                    t.id == "_REACH_READY_ARM_JOINT_POS" for t in node.targets):
                sim = {k.value: v.value if not isinstance(v, _ast.UnaryOp)
                       else -v.operand.value
                       for k, v in zip(node.value.keys, node.value.values)}
        self.assertTrue(sim, "_REACH_READY_ARM_JOINT_POS not found in the sim "
                             "source — the constant moved; update this test's "
                             "parser and the migration notes together")
        qpos0 = {}
        for stem in ARM_JOINT_STEMS:
            name = f"left_{stem}_joint"
            jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos0[name] = float(_MODEL.qpos0[_MODEL.jnt_qposadr[jid]])
        for i, stem in enumerate(ARM_JOINT_STEMS):
            name = f"left_{stem}_joint"
            sign = -1.0 if name == "left_wrist_2_joint" else 1.0
            expect = sign * sim[name] + qpos0[name]
            with self.subTest(joint=name):
                self.assertAlmostEqual(
                    OP.CHEST_HOME_JOINTS[i], expect, places=4,
                    msg=f"{name}: CHEST_HOME_JOINTS[{i}] disagrees with the "
                        f"sim home through the seam (sim cspace "
                        f"{sim[name]:+.4f}, qpos0 {qpos0[name]:+.4f})")

    def test_chest_home_is_still_distinct_from_the_homes_it_is_not(self):
        """Until 2026-07-30 this asserted the chest home was far from EVERY
        other home, including power-on — that was the whole reason it existed,
        since --chest-home otherwise left the arms 0.754 rad from anything
        recognised. Power-on has since been RAISED to meet it, so that pair is
        now zero by intent and is guarded by its own test.

        The rest of the claim still stands and still matters: if the chest home
        were within recognition tolerance of the front station, the v83
        power-on or the side home, posture_gate could not tell an arm parked at
        one from an arm parked at another, and the tools would accept a posture
        they were meant to refuse.
        """
        q = np.asarray(OP.CHEST_HOME_JOINTS, float)
        others = {"front": OP.STAGE_JOINTS["front"],
                  "poweron_v83": OP.POWERON_JOINTS_V83,
                  "side_home": OP.SIDE_HOME_JOINTS}
        for name, h in others.items():
            with self.subTest(home=name):
                self.assertGreater(
                    float(np.max(np.abs(q - np.asarray(h, float)))),
                    OP.FRONT_HOME_TOL_RAD,
                    f"the chest home is now indistinguishable from {name}")

    def test_joint_monkey_poweron_matches_operator(self):
        """The hardware joint-monkey duplicates POWERON_JOINTS on purpose (it
        is the recovery/audit path and must run when the operator module cannot
        import) — and the 07-31 consumer survey caught that duplicate TWO
        generations stale: still the pre-07-30 low tuck, so the monkey's own
        startup gate refused every legitimately-homed arm. A duplicate with no
        guard is how it got two generations behind without anyone noticing.
        """
        import humanoid_joint_monkey_hw as JM
        np.testing.assert_allclose(
            np.asarray(JM.POWERON_JOINTS, float),
            np.asarray(OP.POWERON_JOINTS, float), atol=1e-6,
            err_msg="humanoid_joint_monkey_hw.POWERON_JOINTS has drifted from "
                    "the operator's — update the monkey's literal (it is "
                    "duplicated deliberately, see its comment)")

    def test_rest_ee_is_fk_of_the_front_station(self):
        """Both arms — the right one is FK'd, not derived from the left."""
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                posture = OP.mirror_arm(OP.STAGE_JOINTS["front"], arm)
                np.testing.assert_allclose(
                    np.asarray(OP.REST_EE[arm], float), _ee_fk(posture, arm),
                    atol=POS_TOL_M,
                    err_msg=f"REST_EE['{arm}'] is no longer "
                            f"FK(STAGE_JOINTS['front'])")

    def test_arm_mirror_is_a_true_sagittal_mirror_in_the_model(self):
        """ARM_MIRROR_SIGN checked against GEOMETRY, not against itself.

        test_independent_arms.py can only prove mirror_arm is a consistent
        negation; it cannot know whether that negation matches the model. This
        does: mirroring the joints must mirror the EE about the sagittal plane.

        The 07-28 left_wrist_2 axis divergence is exactly what this catches —
        that one was found by hardware gravity torque, after the fact.
        """
        residuals = []
        postures = {**OP.STAGE_JOINTS,
                    "poweron": OP.POWERON_JOINTS,
                    "side_home": OP.SIDE_HOME_JOINTS}
        for name, posture in postures.items():
            with self.subTest(posture=name):
                left = _ee_fk(posture, "left")
                right = _ee_fk(OP.mirror_arm(posture, "right"), "right")
                residual = float(np.linalg.norm(
                    right - left * np.array([1.0, -1.0, 1.0])))
                residuals.append(residual)
                self.assertLess(
                    residual, MIRROR_TOL_M,
                    msg=(f"mirroring '{name}' does not mirror the EE "
                         f"({residual * 1000:.2f} mm) — ARM_MIRROR_SIGN "
                         f"disagrees with the model's joint axes; run "
                         f"humanoid_joint_monkey_hw.py before moving the arms"))
        # A FIXED asymmetry is a CAD mount offset and is benign; one that VARIES
        # with posture is a kinematic disagreement, which is not.
        self.assertLess(
            max(residuals) - min(residuals), 0.0005,
            msg=("the mirror residual varies with posture — that is a "
                 "kinematic break, not a constant mount offset"))


if __name__ == "__main__":
    unittest.main()
