"""No-CUDA, no-network tests for the remote cuRobo bridge.

WHAT CAN BE TESTED HERE. The GPU solve itself cannot (the robot computer has no CUDA; the
server smoke-runs on the GPU machine instead). Everything that decides whether
a route is ALLOWED TO MOVE THE ROBOT can, and therefore must:

  * the units seam (the silent-90-degree class): pinned model facts, fold
    round-trip, and the geometric fold check Gate A performs;
  * every verify_route gate proven BOTH ways — passes a good route AND fails
    the specific corruption it exists to catch. A gate only proven on green
    routes is decoration;
  * the wire protocol against a fake in-process server (pynng inproc), so a
    schema drift between client and server fails a test, not a hardware run;
  * stretch/interpolation math the streamer trusts blindly.

Ground-truth numbers used below were measured on the 2026-07-28 deploy model:
benign staircase min clearance 51.0 mm; a left-arm lerp onto its own MIRRORED
posture penetrates -92.1 mm (elbow_L through base_link) — the reliable
known-bad route for Gate C.

Like tests/test_model_derived_constants.py this module needs the deploy model
(a gitignored build artifact) and SKIPS, not fails, when it is absent.
"""

from __future__ import annotations

import threading
import time
import unittest
import unittest.mock

import numpy as np

try:
    import mujoco
    import msgpack
    import pynng

    import humanoid_auto_operator as OP
    import humanoid_curobo_client as C
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _MODEL = None
    _SKIP = f"deploy model / deps unavailable ({type(exc).__name__}: {exc})"

FRONT = None if _MODEL is None else np.asarray(OP.STAGE_JOINTS["front"], dtype=float)


def _enc_route(frames_left) -> np.ndarray:
    """(H,14) encoder route: left arm follows `frames_left`, right holds front-mirror."""
    right = np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"), dtype=float)
    return np.stack([np.concatenate([np.asarray(f, dtype=float), right])
                     for f in frames_left])


def _bundle(q_enc: np.ndarray, dt: float = 0.025, goal: str = "reach",
            reaches: dict | None = None) -> C.RouteBundle:
    if reaches is None:
        # Gate A ground truth: the candidate IS the FK of the final waypoint,
        # so an intact seam scores ~0 mm and any fold corruption scores big.
        data = mujoco.MjData(_MODEL)
        data.qpos[:] = _MODEL.qpos0
        for i, name in enumerate(C.ARM_JOINTS_ALL):
            jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[_MODEL.jnt_qposadr[jid]] = q_enc[-1, i]
        mujoco.mj_kinematics(_MODEL, data)
        sid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE, "end_effector_L_site")
        site = data.site_xpos[sid].copy()
        reaches = {"end_effector_L_site":
                   {"target": site, "cands": site.reshape(1, 3), "err": 0.0}}
    return C.RouteBundle(
        joint_names=tuple(C.ARM_JOINTS_ALL), q_enc=q_enc, dt=dt, reaches=reaches,
        controlled_joints=tuple(C.ARM_JOINTS_L), assignment={"L": "cube"},
        goal=goal, solve_s=0.0)


def _measured_from(q_enc_row) -> dict:
    return {n: float(q_enc_row[i]) for i, n in enumerate(C.ARM_JOINTS_ALL)}


@unittest.skipIf(_MODEL is None, _SKIP)
class SeamTests(unittest.TestCase):
    """The units seam, pinned to the model so drift fails HERE first."""

    def test_model_seam_facts_are_what_the_client_hardcodes(self):
        """CSPACE_SIGN and the qpos0 fold encode two specific deploy-model
        layers (wrist_1 ref, left_wrist_2 axis flip). If either layer changes
        — say upstream fixes the axis and the rebuild drops layer 4 — the
        client constants are WRONG and this must fail before a grasp does."""
        q0 = C._qpos0_by_name(_MODEL)
        self.assertAlmostEqual(q0["left_wrist_1_joint"], -np.pi / 2, places=9)
        self.assertAlmostEqual(q0["right_wrist_1_joint"], +np.pi / 2, places=9)
        for name, v in q0.items():
            if "wrist_1" not in name:
                self.assertAlmostEqual(v, 0.0, places=9, msg=name)
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_2_joint")
        np.testing.assert_allclose(_MODEL.jnt_axis[jid], [1.0, 0.0, 0.0],
                                   err_msg="left_wrist_2 axis is no longer the "
                                           "flipped +x — CSPACE_SIGN is stale")
        self.assertEqual(C.CSPACE_SIGN, {"left_wrist_2_joint": -1.0})

    def test_fold_round_trip_is_identity(self):
        rng = np.random.default_rng(7)
        enc = {n: float(v) for n, v in zip(C.ARM_JOINTS_ALL,
                                           rng.uniform(-1.2, 1.2, 14))}
        seed = C.enc_to_cspace_seed(enc, _MODEL)
        row = np.array([[seed[n] for n in C.ARM_JOINTS_ALL]])
        back = C.fold_cspace_to_enc(row, list(C.ARM_JOINTS_ALL), _MODEL)[0]
        np.testing.assert_allclose(back, [enc[n] for n in C.ARM_JOINTS_ALL],
                                   atol=1e-12)

    def test_fold_semantics(self):
        """cspace zero folds to the CAD pose (qpos0); left_wrist_2 flips sign."""
        row = np.zeros((1, 14))
        i_lw2 = C.ARM_JOINTS_ALL.index("left_wrist_2_joint")
        row[0, i_lw2] = 0.5
        enc = C.fold_cspace_to_enc(row, list(C.ARM_JOINTS_ALL), _MODEL)[0]
        q0 = C._qpos0_by_name(_MODEL)
        expect = np.array([q0[n] for n in C.ARM_JOINTS_ALL])
        expect[i_lw2] = -0.5
        np.testing.assert_allclose(enc, expect, atol=1e-12)


@unittest.skipIf(_MODEL is None, _SKIP)
class GateTests(unittest.TestCase):
    """Each gate proven in both directions."""

    def test_benign_staircase_route_passes_all_gates(self):
        side = np.asarray(OP.STAGE_JOINTS["side"], dtype=float)
        frames = [FRONT + (side - FRONT) * t for t in np.linspace(0, 1, 15)]
        b = _bundle(_enc_route(frames))
        report = C.verify_route(b, _measured_from(b.q_enc[0]), model=_MODEL)
        self.assertTrue(report["ok"], report)
        # ground truth from the 07-28 audit of this exact model
        self.assertGreater(report["gates"]["C_clearance"]["worst_mm"], 40.0)
        self.assertLess(report["gates"]["A_fk"]["worst_mm"], 1.0)

    def test_gate_a_catches_a_dropped_qpos0_fold(self):
        """Simulate the exact historical bug: forgetting the wrist_1 ref fold
        leaves the route 90 deg off on wrist_1 — invisible in joint space,
        decimeters in Cartesian. Gate A must see it."""
        b = _bundle(_enc_route([FRONT] * 3))
        broken = b.q_enc.copy()
        i = C.ARM_JOINTS_ALL.index("left_wrist_1_joint")
        broken[:, i] += np.pi / 2                      # the un-applied ref fold
        bad = C.RouteBundle(**{**b.__dict__, "q_enc": broken})
        report = C.verify_route(bad, None, model=_MODEL)
        self.assertFalse(report["ok"])
        self.assertIn("A_fk", report["gates"])
        self.assertFalse(report["gates"]["A_fk"]["ok"])

    def test_gate_b_catches_a_scrambled_seed(self):
        b = _bundle(_enc_route([FRONT] * 3))
        measured = _measured_from(b.q_enc[0])
        measured["left_elbow_joint"] += 0.3            # arm is NOT where route[0] is
        report = C.verify_route(b, measured, model=_MODEL)
        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["B_start"]["ok"])
        self.assertIn("silent-90-deg", report["gates"]["B_start"]["msg"])

    def test_gate_f_catches_the_windup_family(self):
        """User field observation 2026-08-02: reach executions where
        shoulder_1 swings AWAY from the cubes (left more negative / right
        more positive) always end in the regrasp path. Probing the live
        server showed a bimodal solution family: direct (0.00 away) vs
        wind-up (0.97 rad away). Gate F rejects the wind-up family so the
        randomized restart draws again; home routes are exempt."""
        b = _bundle(_enc_route([FRONT] * 5))
        report = C.verify_route(b, None, model=_MODEL, windup_guard=True)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gates"]["F_windup"]["ok"])
        wound = b.q_enc.copy()
        i = C.ARM_JOINTS_ALL.index("left_shoulder_1_joint")
        wound[2, i] -= 0.9                      # mid-route wind-up excursion
        bad = C.RouteBundle(**{**b.__dict__, "q_enc": wound})
        report = C.verify_route(bad, None, model=_MODEL, windup_guard=True)
        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["F_windup"]["ok"])
        self.assertIn("wind-up", report["gates"]["F_windup"]["msg"])
        # guard off (home routes): the same excursion is not judged
        report = C.verify_route(bad, None, model=_MODEL, windup_guard=False)
        self.assertNotIn("F_windup", report["gates"])

    def test_gate_f_rejects_the_shallow_windup_the_old_tolerance_accepted(self):
        """Hardware 2026-08-05, three runs in one session: a boundary gate on
        a dominant solution basin turns randomized restarts into a sampler
        that accepts the SHALLOWEST windup — the rejected draws read
        0.44-0.66 rad, and BOTH accepted routes read 0.2888/0.29987, hugging
        the old 0.3 tolerance from below. Both aborted T12 (the detour left
        the hand ~150 mm out at the MPC handoff); the zero-windup run
        converged in 3.5 s and grasped. A 0.25 rad excursion — comfortably
        accepted before — must now be rejected; the direct family (~0) must
        still pass, or every front reach dies at the plan."""
        b = _bundle(_enc_route([FRONT] * 5))
        wound = b.q_enc.copy()
        i = C.ARM_JOINTS_ALL.index("left_shoulder_1_joint")
        wound[2, i] -= 0.25
        bad = C.RouteBundle(**{**b.__dict__, "q_enc": wound})
        report = C.verify_route(bad, None, model=_MODEL, windup_guard=True)
        self.assertFalse(report["ok"],
                         "the 2026-08-05 T12 family (0.25 rad windup) was "
                         "accepted again")
        self.assertFalse(report["gates"]["F_windup"]["ok"])
        clean = C.verify_route(b, None, model=_MODEL, windup_guard=True)
        self.assertTrue(clean["ok"], clean)
        self.assertLess(C.GATE_F_WINDUP_RAD, 0.288,
                        "tolerance re-loosened past the measured failure "
                        "family — both T12 routes read 0.2888/0.29987")

    def test_the_clearance_sweep_is_skipped_once_a_cheap_gate_failed(self):
        """2026-08-03: the ~1 s clearance sweep ran on every REJECTED attempt
        too — 9 wind-up rejections paid ~9 s of sweeps for routes nobody
        would ever run, which was most of the user's 10 s lock-to-reach
        wait. A failed cheap gate must short-circuit Gate C; a clean route
        must still get the full sweep."""
        b = _bundle(_enc_route([FRONT] * 5))
        wound = b.q_enc.copy()
        i = C.ARM_JOINTS_ALL.index("left_shoulder_1_joint")
        wound[2, i] -= 0.9
        bad = C.RouteBundle(**{**b.__dict__, "q_enc": wound})
        report = C.verify_route(bad, None, model=_MODEL, windup_guard=True)
        self.assertFalse(report["ok"])
        self.assertIn("skipped", report["gates"]["C_clearance"],
                      "the expensive sweep ran on an already-rejected route")
        clean = C.verify_route(b, None, model=_MODEL, windup_guard=True)
        self.assertTrue(clean["ok"], clean)
        self.assertNotIn("skipped", clean["gates"]["C_clearance"])
        self.assertIn("worst_mm", clean["gates"]["C_clearance"])

    def test_gate_e_catches_a_wrong_branch_home_route(self):
        """goal='home' is an EE-pose goal server-side: the solver may return
        the hands home in a DIFFERENT IK branch (hardware 2026-08-01 night:
        left shoulder visibly wrong after a successful carry-home). Gate E
        pins the END posture to home_enc; it only runs when home_enc is
        given, so reach routes are untouched."""
        b = _bundle(_enc_route([FRONT] * 3))
        home = _measured_from(b.q_enc[-1])
        report = C.verify_route(b, None, model=_MODEL, home_enc=home)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gates"]["E_home_branch"]["ok"])
        wrong = dict(home)
        wrong["left_shoulder_1_joint"] += 1.0          # the other branch
        report = C.verify_route(b, None, model=_MODEL, home_enc=wrong)
        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["E_home_branch"]["ok"])
        self.assertIn("IK branch", report["gates"]["E_home_branch"]["msg"])
        # no home_enc -> gate absent -> reach routes unaffected
        report = C.verify_route(b, None, model=_MODEL)
        self.assertNotIn("E_home_branch", report["gates"])

    def test_gate_c_catches_the_torso_crossing_route(self):
        """Left arm lerped onto its own mirrored posture: measured -92.1 mm
        penetration (elbow_L through base_link) on this model.

        The route is CLAMPED inside the model's joint ranges first (the raw
        mirrored posture sits 0.161 rad outside them): since the cheap-gate
        short-circuit (2026-08-03), a route that fails D never reaches the C
        sweep, and this test pins C's own verdict — a deep interior crossing
        survives a 0.05 rad clamp untouched."""
        crossed = np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"))
        frames = [FRONT + (crossed - FRONT) * t for t in np.linspace(0, 1, 20)]
        route = _enc_route(frames)
        for i, j in enumerate(C.ARM_JOINTS_ALL):
            jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = _MODEL.jnt_range[jid]
            route[:, i] = np.clip(route[:, i], lo + 0.05, hi - 0.05)
        b = _bundle(route)                # reaches derive from the CLAMPED end
        report = C.verify_route(b, None, model=_MODEL)
        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["C_clearance"]["ok"])
        self.assertIn("-", report["gates"]["C_clearance"]["msg"])  # negative mm

    def test_gate_d_catches_an_out_of_range_waypoint(self):
        b = _bundle(_enc_route([FRONT] * 3))
        broken = b.q_enc.copy()
        i = C.ARM_JOINTS_ALL.index("left_wrist_2_joint")   # range ±1.6057
        broken[1, i] = 2.5
        bad = C.RouteBundle(**{**b.__dict__, "q_enc": broken})
        report = C.verify_route(bad, None, model=_MODEL)
        self.assertFalse(report["ok"])
        self.assertFalse(report["gates"]["D_limits"]["ok"])


@unittest.skipIf(_MODEL is None, _SKIP)
class StepGateTests(unittest.TestCase):
    """The PER-TICK gate — the real-time path's replacement safety contract.

    verify_route's argument ("the whole route is gated before one packet is
    sent") cannot hold once a command is produced one tick before it executes,
    so StepGate has to carry the same weight configuration-by-configuration.
    Every test below is therefore in the same spirit as GateTests: prove the
    refusal, not just the pass.
    """

    SIDE = None if _MODEL is None else np.asarray(OP.STAGE_JOINTS["side"], dtype=float)
    # The left arm lerped onto its own MIRRORED front posture: -92.1 mm
    # (elbow_L through base_link). Symmetrically, the RIGHT arm at the raw
    # (unmirrored) front posture is -92.1 mm at elbow_R — used below to prove
    # the peer arm is read from the measurement.
    CROSSED = None if _MODEL is None else \
        np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"), dtype=float)

    def setUp(self):
        self.gate = C.StepGate(model=_MODEL)
        self.measured = _measured_from(np.concatenate([FRONT, self.CROSSED]))

    def _cmd(self, q7, joints=C.ARM_JOINTS_L if _MODEL is not None else ()):
        return {j: float(v) for j, v in zip(joints, q7)}

    def test_a_small_tracking_step_passes(self):
        v = self.gate.check(self._cmd(FRONT + 0.01), self.measured, 0.05)
        self.assertTrue(v.ok, v.why)
        self.assertGreater(v.clearance_m, 0.04)
        self.assertAlmostEqual(v.rate_rad_s, 0.2, places=6)

    def test_the_pair_rule_agrees_with_the_route_sweep_exactly(self):
        """THE invariant tying the two gates together: they must not disagree
        about what a collision is. StepGate copies preflight's pair rule instead
        of calling it (the rebuild is the cost), so the copy has to be proven —
        same configuration, same number, bit-for-bit.

        Compared at THREE configurations on purpose, each with a different
        closest pair. A single comparison point is nearly worthless here: at the
        benign posture the minimum is the constant shoulder_2 ~ cam_base
        structural pair, so a pair set missing every hip or base_link entry
        still reports 51.0 mm and agrees. Verified 2026-07-29 by deleting the
        base_link pairs — the one-point version of this test passed.
        """
        from humanoid_joint_monkey_hw import preflight
        half = FRONT + (self.CROSSED - FRONT) * 0.5
        cases = {
            "benign (shoulder_2 ~ cam_base, +51.0 mm)":
                np.concatenate([FRONT, self.CROSSED]),
            "half-crossed (wrist_3_L ~ hip_2_L, -56.1 mm)":
                np.concatenate([half, self.CROSSED]),
            "both arms un-mirrored (elbow_R ~ base_link, -92.1 mm)":
                np.concatenate([FRONT, FRONT]),
        }
        qadr = [_MODEL.jnt_qposadr[mujoco.mj_name2id(
            _MODEL, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in C.ARM_JOINTS_ALL]
        for label, q14 in cases.items():
            with self.subTest(case=label):
                route_min, _ = preflight(_MODEL, mujoco.MjData(_MODEL),
                                         C._FrameShim([q14]), qadr)
                step_min, _ = self.gate._clearance(q14)
                self.assertEqual(step_min, route_min,
                                 "StepGate's cached pair set no longer matches "
                                 "preflight's — the two gates would disagree "
                                 "about safety, which is worse than either "
                                 "being wrong")

    def test_a_nan_command_is_dropped(self):
        cmd = self._cmd(FRONT)
        cmd["left_elbow_joint"] = float("nan")
        self.assertFalse(self.gate.check(cmd, self.measured, 0.05).ok)

    def test_a_limit_violation_is_refused(self):
        q = FRONT.copy()
        q[5] = 2.5                                   # left_wrist_2 range ±1.6057
        v = self.gate.check(self._cmd(q), self.measured, 0.05)
        self.assertFalse(v.ok)
        self.assertIn("limit", v.why)

    def test_a_jump_is_refused_even_though_it_is_geometrically_clear(self):
        """front -> side is 1.641 rad and every configuration between them
        clears 51 mm, so ONLY the rate gate can refuse it. This is the check
        that replaces route gate B: a real-time command must be a correction
        from the measurement, not a teleport."""
        v = self.gate.check(self._cmd(self.SIDE), self.measured, 0.05)
        self.assertFalse(v.ok)
        self.assertIn("rad/s", v.why)
        self.assertGreater(v.rate_rad_s, 30.0)

    def test_a_colliding_command_is_refused(self):
        """Halfway along the crossing lerp: -56.1 mm (wrist_3_L through hip_2_L)
        while every driven joint still has 0.524 rad of limit margin, so the
        clearance sweep is the ONLY gate that can refuse it. The full crossed
        pose would not do — it is 0.161 rad out of range and the limit gate
        would fire first, proving nothing about clearance. dt is deliberately
        huge (10 s) so the rate gate cannot fire either."""
        half = FRONT + (self.CROSSED - FRONT) * 0.5
        v = self.gate.check(self._cmd(half), self.measured, 10.0)
        self.assertFalse(v.ok)
        self.assertIn("clearance", v.why)
        self.assertLess(v.clearance_m, 0.0)          # actual penetration
        self.assertIn("wrist_3_L", v.why)

    def test_the_peer_arm_is_taken_from_the_measurement(self):
        """Command the LEFT arm to hold still; the measurement says the RIGHT
        arm is in its crossing pose (-92.1 mm at elbow_R). A gate that placed
        the peer arm at a plan, or at qpos0, would call this configuration safe
        — and it is a robot with an arm inside its own torso."""
        measured = _measured_from(np.concatenate([FRONT, FRONT]))
        v = self.gate.check(self._cmd(FRONT), measured, 0.05)
        self.assertFalse(v.ok)
        self.assertLess(v.clearance_m, 0.0)
        self.assertIn("_R_", v.why)                  # the RIGHT arm is the culprit

    def test_the_segment_interior_is_sampled_not_just_the_endpoints(self):
        """A 0.05 s step can pass THROUGH an obstacle with both ends clear, so
        the gate samples the segment. No natural example of that exists on this
        model — every station-to-station lerp clears 51 mm and the crossing path
        is monotonically bad — so the MECHANISM is what is pinned here: with
        segment_points=3 the midpoint is evaluated, with 2 it is not.
        """
        seen = []
        orig = C.StepGate._clearance

        def spy(gate, q14):
            seen.append(np.asarray(q14).copy())
            return orig(gate, q14)

        cmd = self._cmd(FRONT + 0.01)
        mid_expected = np.concatenate([FRONT + 0.005, self.CROSSED])
        for points, want in ((3, True), (2, False)):
            seen.clear()
            gate = C.StepGate(model=_MODEL, segment_points=points)
            with unittest.mock.patch.object(C.StepGate, "_clearance", spy):
                gate.check(cmd, self.measured, 0.05)
            self.assertEqual(len(seen), points)
            got = any(np.allclose(q, mid_expected, atol=1e-9) for q in seen)
            self.assertEqual(got, want,
                             f"segment_points={points}: midpoint sampled={got}")

    def test_the_gate_is_fast_enough_for_the_control_loop(self):
        """The regression this guards is specific and silent: preflight rebuilds
        its 1640-pair set per call, which is 7.6 ms for ONE frame while
        mj_kinematics is 0.006 ms. StepGate hoists that into __init__ (measured
        1.5 ms per check, 3 samples). If someone re-introduces the rebuild, the
        real-time path silently misses its tick budget instead of failing — so
        assert the budget. Bound set between the two regimes, not near either.
        """
        cmd = self._cmd(FRONT + 0.01)
        self.gate.check(cmd, self.measured, 0.05)          # warm
        t0 = time.perf_counter()
        for _ in range(20):
            self.gate.check(cmd, self.measured, 0.05)
        per_call_ms = (time.perf_counter() - t0) / 20 * 1e3
        self.assertLess(per_call_ms, 5.0,
                        f"{per_call_ms:.1f} ms/check — a 20 Hz loop has 50 ms "
                        f"for solve + RTT + gate; is the pair set being rebuilt?")


@unittest.skipIf(_MODEL is None, _SKIP)
class StretchTests(unittest.TestCase):
    def test_stretch_caps_the_rate_and_keeps_the_path(self):
        fast = _enc_route([FRONT, FRONT + 0.4, FRONT + 0.8])   # 0.4 rad / 0.025 s = 16 rad/s
        b = _bundle(fast, dt=0.025)
        s = b.stretch(0.3)
        self.assertLessEqual(s.max_joint_rate(), 0.3 * 1.001)
        np.testing.assert_array_equal(s.q_enc, b.q_enc)        # shape untouched
        self.assertAlmostEqual(s.duration_s, b.duration_s * (b.max_joint_rate() / 0.3),
                               places=6)

    def test_stretch_is_a_noop_below_the_cap(self):
        slow = _enc_route([FRONT, FRONT + 0.001])
        b = _bundle(slow, dt=0.025)
        self.assertIs(b.stretch(0.3), b)

    def test_q_at_interpolates_and_clamps(self):
        b = _bundle(_enc_route([FRONT, FRONT + 0.1]), dt=1.0)
        np.testing.assert_allclose(b.q_at(-1.0), b.q_enc[0])
        np.testing.assert_allclose(b.q_at(99.0), b.q_enc[-1])
        np.testing.assert_allclose(b.q_at(0.5), (b.q_enc[0] + b.q_enc[1]) / 2)


@unittest.skipIf(_MODEL is None, _SKIP)
class ProtocolTests(unittest.TestCase):
    """PlanClient against a fake in-process server — schema drift fails here."""

    ADDR = "inproc://curobo-bridge-test"

    def _serve_one(self, reply_builder):
        def run():
            with pynng.Rep0(listen=self.ADDR) as sock:
                sock.recv_timeout = 3000
                req = msgpack.unpackb(sock.recv())
                sock.send(msgpack.packb(reply_builder(req)))
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def _client(self):
        return C.PlanClient(self.ADDR, model=_MODEL)

    def test_plan_round_trip_decodes_and_folds(self):
        seed_echo = {}

        def reply(req):
            seed_echo.update(req["seed_q"])
            h, names = 4, list(C.ARM_JOINTS_ALL)
            return {"kind": "plan0", "gen": req["gen"], "solve_s": 0.5, "err": None,
                    "route": {"joint_names": names,
                              "route_q": np.zeros((h, 14)).tolist(),
                              "interpolation_dt": 0.025,
                              "reaches": {"end_effector_L_site":
                                          {"target": [0.3, 0.2, 0.0],
                                           "cands": [[0.3, 0.2, 0.0]], "err": 0.0}},
                              "controlled_joints": list(C.ARM_JOINTS_L),
                              "base_pos": [0, 0, 0], "base_quat": [1, 0, 0, 0],
                              "assignment": {"L": "cube_a"}}}
        t = self._serve_one(reply)
        cli = self._client()
        try:
            enc = {n: 0.1 for n in C.ARM_JOINTS_ALL}
            b = cli.plan({"table": {"dims": [1, 1, 0.1],
                                    "pose": [0.4, 0, -0.2, 1, 0, 0, 0]}},
                         {"cube_a": [0.35, 0.15, -0.1]},
                         measured_enc=enc, goal="reach", timeout_s=3.0)
        finally:
            cli.close()
            t.join(timeout=3.0)
        # cspace zeros fold to qpos0 (the CAD pose) — the seam ran on decode
        q0 = C._qpos0_by_name(_MODEL)
        np.testing.assert_allclose(
            b.q_enc[0], [q0[n] for n in C.ARM_JOINTS_ALL], atol=1e-12)
        self.assertEqual(b.assignment, {"L": "cube_a"})
        self.assertEqual(b.active_sides, ("left",))
        # and the seed the server saw was cspace: enc 0.1 - qpos0, signed
        self.assertAlmostEqual(seed_echo["left_wrist_1_joint"], 0.1 + np.pi / 2, places=9)
        self.assertAlmostEqual(seed_echo["left_wrist_2_joint"], -0.1, places=9)

    def test_infeasible_returns_none(self):
        t = self._serve_one(lambda req: {"kind": "plan0", "gen": req["gen"],
                                         "solve_s": 1.0, "err": None, "route": None})
        cli = self._client()
        try:
            out = cli.plan({}, {"c": [0.4, 0.1, -0.1]}, timeout_s=3.0)
        finally:
            cli.close()
            t.join(timeout=3.0)
        self.assertIsNone(out)

    def test_server_error_raises(self):
        t = self._serve_one(lambda req: {"kind": "plan0", "gen": req["gen"],
                                         "solve_s": 0.0, "err": "Boom()", "route": None})
        cli = self._client()
        try:
            with self.assertRaises(C.PlanServerError):
                cli.plan({}, {"c": [0.4, 0.1, -0.1]}, timeout_s=3.0)
        finally:
            cli.close()
            t.join(timeout=3.0)

    def test_server_error_surfaces_even_with_a_mismatched_gen(self):
        """The regression this file missed. The server tags an error reply with
        gen=-1 when it could not parse the request; an earlier client drained
        "stale" replies BEFORE checking `err`, which over Req0 meant a second
        illegal recv() — turning a precise server traceback into
        "pynng BadState: Incorrect state". Cost one hardware run."""
        t = self._serve_one(lambda req: {"kind": "plan0", "gen": -1, "solve_s": 0.0,
                                         "err": "ValueError: boom", "route": None})
        cli = self._client()
        try:
            with self.assertRaises(C.PlanServerError) as ctx:
                cli.plan({}, {"c": [0.4, 0.1, -0.1]}, timeout_s=3.0)
            self.assertIn("boom", str(ctx.exception))
        finally:
            cli.close()
            t.join(timeout=3.0)

    def test_a_transport_timeout_is_a_planservererror_not_a_crash(self):
        """Review 2026-08-01: a raw pynng.exceptions.Timeout is NOT a
        PlanServerError, so it skipped every caller's failure handling —
        mpc_track's consecutive-failure counter, plan_reach's refuse path —
        and crashed the mission mid-approach with the arms extended. The wrap
        lives in _rpc, the one seam every RPC shares; a server that swallows
        the request and never replies must surface as PlanServerError."""
        def run():
            with pynng.Rep0(listen=self.ADDR) as sock:
                sock.recv_timeout = 3000
                try:
                    sock.recv()               # swallow the request, no reply
                    time.sleep(0.8)           # outlive the client's timeout
                except pynng.exceptions.NNGException:
                    pass
        t = threading.Thread(target=run, daemon=True)
        t.start()
        cli = self._client()
        try:
            with self.assertRaises(C.PlanServerError) as ctx:
                cli.ping(timeout_s=0.3)
            self.assertIn("transport", str(ctx.exception))
        finally:
            cli.close()
            t.join(timeout=3.0)

    def test_a_dead_cuda_context_is_its_own_error_type(self):
        """A poisoned CUDA context is not a retryable solve failure: the process
        must be restarted, so it gets its own exception type carrying that
        instruction. Retrying it just burns hardware time."""
        tb = ("Traceback (most recent call last):\n"
              "torch.AcceleratorError: CUDA error: unspecified launch failure\n")
        t = self._serve_one(lambda req: {"kind": "plan0", "gen": req["gen"],
                                         "solve_s": 0.0, "err": tb, "route": None})
        cli = self._client()
        try:
            with self.assertRaises(C.PlanServerDeadError) as ctx:
                cli.plan({}, {"c": [0.4, 0.1, -0.1]}, timeout_s=3.0)
            msg = str(ctx.exception)
            self.assertIn("CUDA context is dead", msg)
            # Actionable, not just a diagnosis: it must name the server
            # module to relaunch AND the fact that the restart happens over
            # ssh on the machine running the server, which is not this one.
            self.assertIn("curobo_plan_server.py", msg)
            self.assertIn("ssh", msg)
        finally:
            cli.close()
            t.join(timeout=3.0)
        self.assertTrue(issubclass(C.PlanServerDeadError, C.PlanServerError),
                        "callers catching PlanServerError must still catch this")

    def test_retract_requires_a_seed(self):
        cli = self._client()
        try:
            with self.assertRaises(AssertionError):
                cli.plan({}, {}, goal="home")
        finally:
            cli.close()


if __name__ == "__main__":
    unittest.main()
