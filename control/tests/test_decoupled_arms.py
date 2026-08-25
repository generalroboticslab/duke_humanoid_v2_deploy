"""Decoupled dual-arm planning (user 2026-08-04: the joint solve's coupled
cost landscape deadlocks some placements; solve each arm ALONE, keep the
SIMULTANEOUS execution). Pins: the merge takes each arm's columns from its
own solve, pads the shorter route at its final posture, unions reaches and
assignment, refuses mismatched time bases — and the merged route is gated by
the real verify_route, where two benign single-arm routes pass and a
torso-crossing left arm still fails the C sweep (the coupled safety check
that replaces the joint solver's internal mutual avoidance)."""
import unittest

import numpy as np

try:
    import mujoco

    import humanoid_auto_operator as OP
    import humanoid_curobo_client as C
    import humanoid_curobo_reach as R
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    C = None
    _SKIP = f"bridge unavailable ({type(exc).__name__}: {exc})"

if C is not None:
    JN = tuple(C.ARM_JOINTS_ALL)
    HOME14 = np.array(
        [float(v) for v in OP.mirror_arm(OP.POWERON_JOINTS, "left")]
        + [float(v) for v in OP.mirror_arm(OP.POWERON_JOINTS, "right")])


def _final_ee(q14, side):
    data = mujoco.MjData(_MODEL)
    data.qpos[:] = _MODEL.qpos0
    for i, j in enumerate(JN):
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, j)
        data.qpos[_MODEL.jnt_qposadr[jid]] = q14[i]
    mujoco.mj_kinematics(_MODEL, data)
    sid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE,
                            f"end_effector_{side}_site")
    return data.site_xpos[sid].copy()


def _bundle(side, q_rows, dt=0.05):
    names = C.ARM_JOINTS_L if side == "L" else C.ARM_JOINTS_R
    q = np.asarray(q_rows, float)
    return C.RouteBundle(
        joint_names=JN, q_enc=q, dt=dt,
        reaches={f"end_effector_{side}_site": {
            "target": _final_ee(q[-1], side),   # A gate: aims where it ends
            "cands": np.zeros((0, 3)), "err": 0.0}},
        controlled_joints=tuple(names),
        assignment={side: f"cube_{side}"}, goal="reach", solve_s=0.1)


def _arm_route(side, delta, horizon):
    """HOME-based route moving ONLY `side`'s shoulder by `delta` (linear),
    with the idle peer meandering (must be discarded by the merge)."""
    rows = []
    col = JN.index(("left_" if side == "L" else "right_")
                   + "shoulder_1_joint")
    peer = JN.index(("right_" if side == "L" else "left_")
                    + "shoulder_1_joint")
    for h in range(horizon):
        q = HOME14.copy()
        q[col] += delta * h / max(1, horizon - 1)
        q[peer] += 0.7 * h / max(1, horizon - 1)   # idle-arm wander: discard
        rows.append(q)
    return _bundle(side, rows)


@unittest.skipIf(C is None, _SKIP)
class MergeTests(unittest.TestCase):

    def test_each_arm_comes_from_its_own_solve_and_padding_holds_the_end(self):
        bl = _arm_route("L", 0.3, 5)
        br = _arm_route("R", -0.3, 3)              # shorter: gets padded
        m = C.merge_decoupled_bundles({"L": bl, "R": br})
        self.assertEqual(m.horizon, 5)
        li = JN.index("left_shoulder_1_joint")
        ri = JN.index("right_shoulder_1_joint")
        self.assertAlmostEqual(float(m.q_enc[-1, li]), HOME14[li] + 0.3,
                               places=9, msg="left column not from L's solve")
        self.assertAlmostEqual(float(m.q_enc[-1, ri]), HOME14[ri] - 0.3,
                               places=9,
                               msg="right column not from R's own solve — "
                                   "the idle-arm wander leaked through")
        np.testing.assert_allclose(
            m.q_enc[3, ri], m.q_enc[4, ri],
            err_msg="the padded tail did not hold the final posture")
        self.assertEqual(set(m.assignment), {"L", "R"})
        self.assertEqual(set(m.reaches),
                         {"end_effector_L_site", "end_effector_R_site"})
        self.assertEqual(len(m.controlled_joints), 14)

    def test_mismatched_time_bases_are_refused(self):
        bl = _arm_route("L", 0.3, 5)
        br = _bundle("R", _arm_route("R", -0.3, 5).q_enc, dt=0.02)
        with self.assertRaises(ValueError):
            C.merge_decoupled_bundles({"L": bl, "R": br})

    def test_the_merged_route_faces_the_real_c_sweep(self):
        """Two benign outward reaches pass; a left arm lerped onto its own
        MIRRORED posture (the torso-crossing fixture, measured -92 mm
        penetration) must still be caught ON THE MERGED ROUTE — this is the
        coupled check that replaces the joint solver's mutual avoidance."""
        ok = C.merge_decoupled_bundles(
            {"L": _arm_route("L", 0.25, 6), "R": _arm_route("R", -0.25, 6)})
        report = C.verify_route(ok, None, model=_MODEL)
        self.assertTrue(report["ok"], report)
        crossed = np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"))
        rows = []
        for h in range(6):
            q = HOME14.copy()
            f = h / 5.0
            q[:7] = q[:7] + (crossed - q[:7]) * f
            for i, j in enumerate(JN):
                jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, j)
                lo, hi = _MODEL.jnt_range[jid]
                q[i] = float(np.clip(q[i], lo + 0.05, hi - 0.05))
            rows.append(q)
        bad = C.merge_decoupled_bundles(
            {"L": _bundle("L", rows), "R": _arm_route("R", -0.25, 6)})
        report = C.verify_route(bad, None, model=_MODEL)
        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["C_clearance"]["ok"],
                         "a torso-crossing merged route passed the C sweep")


@unittest.skipIf(C is None, _SKIP)
class AssignmentTests(unittest.TestCase):

    def test_the_lefter_cube_goes_to_the_left_arm(self):
        pair = R.decoupled_assignment(
            {"a": np.array([0.4, 0.3, 0.05]), "b": np.array([0.4, -0.3, 0.05])})
        self.assertEqual(pair, {"L": "a", "R": "b"})


if __name__ == "__main__":
    unittest.main()
