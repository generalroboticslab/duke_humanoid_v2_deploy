"""The MPC terminal track: live goals, frozen-on-occlusion, per-tick gated.

WHY THIS EXISTS (2026-08-01, the user's own hardware diagnosis): "the green
line's endpoint does not follow the cube in the monitor". The route is planned
against a 0.7 s median and played open-loop, so torso sway, detection
refinement and any real cube motion all land as grasp error. mpc_track closes
the loop for the FINAL approach only — plan-0 still owns (and gates) the path.

What these tests pin, with a scripted MPC client and a perfect-follower
telemetry rig, against the REAL deploy model and the REAL StepGate:

  * convergence returns the MEASURED posture (the grasp must hold where the
    hand actually is, not the route's endpoint) and always tears the session
    down;
  * a sighting below DRIFT_MIN_INLIERS freezes the goal — the hand occluding
    its own cube on final approach must not send the tracker chasing one-face
    PnP garbage;
  * a goal that leaves the reach envelope aborts;
  * a command the per-tick gate rejects is NEVER published, and enough
    consecutive rejections abort the track;
  * the tracking budget degrades gracefully: inside GRASP_EE_OK_M it grasps
    (open-loop parity), outside it SKIPs;
  * every published packet carries BOTH arms (the no-silent-waits rule).
"""

from __future__ import annotations

import time
import unittest

import numpy as np

try:
    import mujoco

    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R
    from humanoid_curobo_client import ARM_JOINTS_ALL, ARM_JOINTS_L, ARM_JOINTS_R
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _MODEL = None
    _SKIP = f"deploy model unavailable ({type(exc).__name__}: {exc})"

KEY_L, KEY_R = "grasp_cube_60mm", "grasp_cube_40mm"


def _fk_ee(enc: dict) -> dict:
    data = mujoco.MjData(_MODEL)
    data.qpos[:] = _MODEL.qpos0
    for n in ARM_JOINTS_ALL:
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, n)
        data.qpos[_MODEL.jnt_qposadr[jid]] = enc[n]
    mujoco.mj_kinematics(_MODEL, data)
    out = {}
    for s in ("L", "R"):
        sid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE,
                                f"end_effector_{s}_site")
        out[s] = data.site_xpos[sid].copy()
    return out


def _home_enc() -> dict:
    enc = {}
    for n, v in zip(ARM_JOINTS_L, OP.POWERON_JOINTS):
        enc[n] = float(v)
    for n, v in zip(ARM_JOINTS_R, OP.mirror_arm(OP.POWERON_JOINTS, "right")):
        enc[n] = float(v)
    return enc


HOME = None if _MODEL is None else _home_enc()
GOALS = None if _MODEL is None else _fk_ee(HOME)   # cubes exactly at the home EEs


class _Args:
    tables = ()
    table = ()
    no_table_world = True
    cubes = (KEY_L, KEY_R)
    clearance_floor_mm = 10.0
    max_rate = 0.3
    mpc = True
    anchor_trust = True     # mirrors the real default (08-12 night)
    reach_hover_mm = 0.0    # direct mode: the tracking-semantics pins below
                            # predate the staged hover and pin goals == cube;
                            # the staging itself is pinned separately


class _HoverArgs(_Args):
    reach_hover_mm = 30.0


class _NoTrustArgs(_Args):
    """Legacy pins: the pre-08-12 trigger behaviour survives behind
    --no-anchor-trust, and these tests are its regression suite."""
    anchor_trust = False


class _NoTrustHoverArgs(_HoverArgs):
    anchor_trust = False


class _Tel:
    """Arms follow the last published joint command — exactly by default, or
    with a persistent per-joint deficit (`sag`) so tests can tell MEASURED
    from COMMANDED. A perfect follower cannot: command == measurement by
    construction, and the measured-posture contract goes unenforced (review
    2026-08-01)."""

    def __init__(self, enc, sag=None):
        self.enc = dict(enc)
        self.sag = dict(sag or {})
        self.g = [0.0, 0.0, -1.0]              # projected gravity; tests tilt
                                               # the base by mutating this

    def follow(self, slots):
        for arm, names in (("left", ARM_JOINTS_L), ("right", ARM_JOINTS_R)):
            if arm in slots:
                for n, v in zip(names, slots[arm]["joint_pos"]):
                    self.enc[n] = float(v) - self.sag.get(n, 0.0)

    def fresh(self):
        jp = [0.0] * 31
        for i, n in enumerate(ARM_JOINTS_L):
            jp[13 + i] = self.enc[n]
        for i, n in enumerate(ARM_JOINTS_R):
            jp[20 + i] = self.enc[n]
        return {"joint_pos": jp, "projected_gravity": list(self.g),
                "arm_fault": [False, False]}


class _Det:
    def poll(self, rig):
        pass


class _Rig:
    def __init__(self, med, inliers=3):
        # None means NO LIVE SIGHTING, which median() already has a branch for
        # — but np.asarray(None, float) is a 0-d NaN, so the old blanket
        # conversion made that branch unreachable and turned a blind cube into
        # a poisoned one. Blindness is a state these tests must be able to
        # express (MPC_BLIND_GRASP).
        self.med = {k: (None if v is None else np.asarray(v, float))
                    for k, v in med.items()}
        self.inliers = {k: inliers for k in med}

    def median(self, key, now):
        v = self.med.get(key)
        return None if v is None else v.copy()

    def latest(self, key, now):
        return self.median(key, now)

    def last_inliers(self, key):
        return self.inliers.get(key, 0)

    def faces_used(self, key, now=None):
        # faces behind the MEDIAN, not the newest packet
        return self.last_inliers(key)

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out

    def table_pose(self, key, now):
        return None


class _Pub:
    def __init__(self, tel):
        self.tel = tel
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)
        slots = (packet or {}).get("arm_targets") or {}
        if slots:
            self.tel.follow(slots)


class _Client:
    """Scripted MPC: each step moves the commanded enc toward `target_enc`
    by `step_rad`. Records everything the tool sends."""

    def __init__(self, target_enc, step_rad=0.02, fail=None, jump=False):
        self.target = dict(target_enc)
        self.step_rad = step_rad
        self.fail = fail                 # exception instance to raise per step
        self.jump = jump                 # command a rate-violating jump
        self.started = None
        self.stopped = 0
        self.cube_history = []
        self.cur = None

    def mpc_start(self, scene, *, exclude, reaching_sides, measured_enc,
                  cubes_init=None, grasp_index=-1, warm_iters=50):
        self.started = {"sides": list(reaching_sides),
                        "exclude": tuple(exclude),
                        "scene": dict(scene),
                        "cubes_init": {k: np.asarray(v, float)
                                       for k, v in (cubes_init or {}).items()}}
        self.cur = dict(measured_enc)
        return {"warm_s": 0.0, "joint_names": list(ARM_JOINTS_ALL),
                "tracked": list(reaching_sides), "grasp_index": {}}

    def mpc_step(self, measured_enc, cube_pos_by_side):
        self.cube_history.append({k: np.asarray(v, float).copy()
                                  for k, v in cube_pos_by_side.items()})
        if self.fail is not None:
            raise self.fail
        if self.jump:
            return {"enc": {n: measured_enc[n] + 1.0 for n in ARM_JOINTS_ALL},
                    "goals": {}, "solve_s": 0.001}
        out = {}
        for n in ARM_JOINTS_ALL:
            cur = self.cur[n]
            tgt = self.target[n]
            d = np.clip(tgt - cur, -self.step_rad, self.step_rad)
            out[n] = cur + float(d)
        self.cur = dict(out)
        return {"enc": out, "goals": {}, "solve_s": 0.001}

    def mpc_stop(self):
        self.stopped += 1


class _Bundle:
    duration_s = 1.0

    def __init__(self):
        self.assignment = {"L": KEY_L, "R": KEY_R}
        self.joint_names = tuple(ARM_JOINTS_ALL)


class _RouteBundle(_Bundle):
    """A bundle with an actual (linear) route, for the retreat/stream tests."""

    goal = "reach"
    horizon = 2

    def __init__(self, q0, q1, duration_s=1.0):
        super().__init__()
        self.duration_s = float(duration_s)
        self.active_sides = ("left", "right")
        self.controlled_joints = tuple(ARM_JOINTS_ALL)
        self._q0 = np.asarray(q0, float)
        self._q1 = np.asarray(q1, float)
        # The real RouteBundle carries the waypoint array; the settle loop
        # reads q_enc[-1]. Absent here until 2026-08-06 because every test
        # truncated the stream with until_s and never reached the settle.
        self.q_enc = np.stack([self._q0, self._q1])

    def q_at(self, t):
        f = min(1.0, max(0.0, t / self.duration_s))
        return self._q0 + f * (self._q1 - self._q0)


def _ctx(client, tel, rig, args=None, route_pub=None):
    return R.Ctx(args if args is not None else _Args(), _MODEL, tel, _Det(),
                 client, _Pub(tel), route_pub, rig, home_enc=None)


def _run(client, rig, start_off=0.3, max_s=6.0, fail_max=None,
         converged_ticks=3, targets=None, args=None, tel_sag=None,
         route_pub=None):
    tel = _Tel({n: HOME[n] + (start_off if n in ARM_JOINTS_L[:1] else 0.0)
                for n in ARM_JOINTS_ALL}, sag=tel_sag)
    ctx = _ctx(client, tel, rig, args=args, route_pub=route_pub)
    keep = (R.MPC_MAX_S, R.MPC_STEP_FAIL_MAX, R.MPC_CONVERGED_TICKS)
    R.MPC_MAX_S = max_s
    if fail_max is not None:
        R.MPC_STEP_FAIL_MAX = fail_max
    R.MPC_CONVERGED_TICKS = converged_ticks
    if targets is None:
        targets = {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}
    try:
        why, hold = R.mpc_track(ctx, _Bundle(), targets, {})
    finally:
        R.MPC_MAX_S, R.MPC_STEP_FAIL_MAX, R.MPC_CONVERGED_TICKS = keep
    return why, hold, ctx


@unittest.skipIf(_MODEL is None, _SKIP)
class MpcTrackTests(unittest.TestCase):

    def test_converges_and_returns_the_measured_posture(self):
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, ctx = _run(client, rig)
        self.assertIsNone(why, f"track failed: {why}")
        self.assertIsNotNone(hold)
        # THE contract: hold is the MEASURED posture at convergence — exactly
        # what the telemetry reads, not the goal, not the route endpoint. (EE
        # convergence at 2 cm legitimately leaves individual joints ~0.1 rad
        # from the nominal home; asserting against HOME here would encode a
        # false expectation.)
        worst = max(abs(hold[n] - ctx.tel.enc[n]) for n in ARM_JOINTS_ALL)
        self.assertLess(worst, 0.03, "returned hold is not the measured arm")
        self.assertEqual(client.stopped, 1, "session not torn down exactly once")
        self.assertEqual(client.started["sides"], ["L", "R"])

    def test_every_packet_carries_both_arms(self):
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        _, _, ctx = _run(client, rig)
        self.assertTrue(ctx.pub.sent)
        for pkt in ctx.pub.sent:
            self.assertEqual(set(pkt["arm_targets"]), {"left", "right"})

    def test_occlusion_freezes_the_goal(self):
        """Below the evidence bar the goal must NOT follow the median — the
        rig reports the cube 30 cm away with 1 tag face, and every cube pose
        the client is asked to track must stay at the initial position."""
        moved = {KEY_L: GOALS["L"] + np.array([0.3, 0.0, 0.0]),
                 KEY_R: GOALS["R"]}
        rig = _Rig(moved, inliers=1)
        # Initial goals fall back to the PLAN targets because the rig's median
        # is ignored at inliers=1?  No: mpc_track seeds from rig.median if not
        # None regardless of inliers (pre-loop), so seed the rig with the true
        # goals and flip it to the moved position after the first tick instead.
        rig.med = {KEY_L: GOALS["L"].copy(), KEY_R: GOALS["R"].copy()}

        flip = {"done": False}
        orig_median = rig.median

        def median(key, now):
            if not flip["done"]:
                return orig_median(key, now)
            return moved[key].copy()

        rig.median = median
        client = _Client(HOME)
        orig_step = client.mpc_step

        def step(measured_enc, cubes):
            flip["done"] = True
            return orig_step(measured_enc, cubes)

        client.mpc_step = step
        why, hold, _ = _run(client, rig)
        self.assertIsNone(why, f"track failed: {why}")
        for sent in client.cube_history:
            np.testing.assert_allclose(
                sent["L"], GOALS["L"], atol=1e-9,
                err_msg="a 1-inlier sighting moved the tracked goal")

    def test_a_goal_leaving_the_envelope_aborts(self):
        rig = _Rig({KEY_L: GOALS["L"] + np.array([0.6, 0.0, 0.0]),
                    KEY_R: GOALS["R"]}, inliers=3)
        client = _Client(HOME)
        why, hold, _ = _run(client, rig)
        self.assertIsNotNone(why)
        self.assertIn("envelope", why)
        self.assertEqual(client.stopped, 1, "abort skipped the teardown")

    def test_the_envelope_abort_needs_more_than_one_noisy_tick(self):
        """T07 used to fire on a SINGLE tick — alone among the track guards,
        while judging the noisiest signal in the system. Hardware 2026-08-06:
        a 1-face cube estimate drifting 26-100 mm produced six aborts in an
        evening at r_xy 0.600-0.613 against a 0.600 bar, one of them printing
        'move it 0 cm CLOSER'. A lone excursion must now be tolerated; a
        SUSTAINED one must still abort (the previous test pins that)."""
        far = GOALS["L"] + np.array([0.6, 0.0, 0.0])

        class _Blip(_Rig):
            """Out of the envelope for exactly one tick, then back."""

            def __init__(self, med):
                super().__init__(med, inliers=3)
                self.n = 0

            def median(self, key, now):
                if key == KEY_L:
                    self.n += 1
                    return far.copy() if self.n == 3 else GOALS["L"].copy()
                return super().median(key, now)

        client = _Client(HOME)
        why, _hold, _ctx = _run(client, _Blip({KEY_L: GOALS["L"],
                                               KEY_R: GOALS["R"]}), max_s=2.0)
        self.assertNotEqual(
            getattr(why, "code", None), "T07",
            f"one out-of-envelope tick still aborted the mission: {why}")
        self.assertGreaterEqual(
            R.MPC_ENVELOPE_STRIKES, 2,
            "a single-tick envelope abort is what this test exists to prevent")

    def test_an_arm_already_at_the_cube_ignores_the_envelope(self):
        """The grasp-anyway rule T07 never had. Hardware 2026-08-06: both
        hands printed 'at the hover (15 mm above the cube)' and the mission
        retreated two ticks later because the cube's ESTIMATE read 0.4 mm past
        a soft planning limit. T12/T13/T14 all close the gripper when the hand
        is inside GRASP_EE_OK; T07 walked away. The envelope is a precondition
        for PLANNING a reach — once the hand is there, physics has already
        answered the question it asks."""
        # The hardware situation is "hand ON the cube, cube reads unreachable".
        # Building that by moving the cube is self-contradictory (a cube 0.6 m
        # away is 0.6 m from the hand), and this arm cannot physically put its
        # own hand past the 0.6 m bound, so the condition is created by
        # SHRINKING the envelope under a stationary, already-reached target.
        tel = _Tel(dict(HOME))                   # hand exactly at GOALS
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, inliers=3)
        ctx = _ctx(_Client(dict(HOME)), tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, OP.TARGET_MAX_RADIUS_M)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 2.5, 3
        # home EE sits at r_xy ~0.269; 0.20 puts BOTH goals outside.
        OP.TARGET_MAX_RADIUS_M = 0.20
        import contextlib
        import io
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS,
             OP.TARGET_MAX_RADIUS_M) = keep
        said = out.getvalue()
        self.assertNotEqual(
            getattr(why, "code", None), "T07",
            f"T07 aborted with the hand already at the cube: {why}")
        # 08-13: a hand already ON its cube now closes at the handoff before
        # the session (HANDOFF CLOSE); the in-session grasp-anyway wording
        # remains valid for tracks that arrive off the handoff bars.
        self.assertTrue("inside GRASP_EE_OK" in said
                        or "HANDOFF CLOSE" in said,
                        "no close branch ever announced itself")

    def test_gate_rejected_commands_are_not_published_and_abort(self):
        client = _Client(HOME, jump=True)     # every command a 1 rad jump
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, ctx = _run(client, rig, fail_max=5)
        self.assertIsNotNone(why)
        # a 1 rad jump now dies at the SANITY bound, upstream of the gate —
        # same contract (reject, count, abort), earlier tripwire
        self.assertIn("unusable ticks", why)
        for pkt in ctx.pub.sent:              # nothing published ever jumped
            for arm, names in (("left", ARM_JOINTS_L), ("right", ARM_JOINTS_R)):
                jp = pkt["arm_targets"][arm]["joint_pos"]
                for n, v in zip(names, jp):
                    self.assertLess(abs(v - HOME[n]), 0.5,
                                    "a gate-rejected jump reached the wire")
        self.assertEqual(client.stopped, 1)

    def test_transport_failures_abort_after_the_budget(self):
        client = _Client(HOME, fail=R.PlanServerError("link down"))
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, _ = _run(client, rig, fail_max=4)
        self.assertIsNotNone(why)
        self.assertIn("failed", why)
        self.assertEqual(client.stopped, 1)

    def test_budget_inside_grasp_tolerance_grasps_anyway(self):
        """Open-loop parity: converging ~36 mm off is worse than
        MPC_CONVERGED_M but inside GRASP_EE_OK_M — burning the budget there
        must grasp, not give up. The elapsed-time assertion pins that the
        BUDGET branch answered: the previous 0.06 rad bias produced 18 mm —
        inside convergence — and the test never reached this path (review
        2026-08-01)."""
        off = dict(HOME)
        off[ARM_JOINTS_L[3]] += 0.12          # elbow bias -> ~36 mm of EE
        client = _Client(off)                 # tracker converges OFF the goal
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        t0 = time.monotonic()
        why, hold, _ = _run(client, rig, start_off=0.3, max_s=2.5)
        self.assertGreaterEqual(
            time.monotonic() - t0, 2.5,
            "returned before the budget — the convergence branch answered "
            "and this test is vacuous again")
        self.assertIsNone(why, f"gave up inside GRASP_EE_OK_M: {why}")
        self.assertIsNotNone(hold)

    def test_budget_outside_grasp_tolerance_skips(self):
        off = dict(HOME)
        off[ARM_JOINTS_L[0]] += 0.6           # shoulder way off: decimetres
        client = _Client(off)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        # start OUTSIDE the ring and stay there, so this exercises the BUDGET
        # path it names. At the old default (0.3 rad ~= 41 mm) the hand began
        # INSIDE GRASP_EE_OK and the solver then walked it out — which is the
        # 08-13 escape-guard scenario verbatim, and that guard now closes it
        # rather than letting the budget expire (see MPC_ESCAPE_MARGIN_M).
        why, hold, _ = _run(client, rig, start_off=0.6, max_s=1.5)
        self.assertIsNotNone(why)
        self.assertIn("did not converge", why)
        self.assertEqual(why.code, "T14", "the budget abort lost its code")
        self.assertIn("arm", why.culprit)     # names the hand that ran out
        self.assertEqual(client.stopped, 1)

    def test_a_low_evidence_seed_falls_back_to_the_plan_target(self):
        """Hardware 2026-08-01: a cube seen at ONE tag face all stream had a
        garbage median; seeding from it aimed the tracker ~5 cm off while the
        evidence gate froze the goal ON the garbage. Below the bar, the seed
        must be the PLAN target."""
        garbage = {KEY_L: GOALS["L"] + np.array([0.05, 0.0, 0.0]),
                   KEY_R: GOALS["R"]}
        rig = _Rig(garbage, inliers=1)         # a median exists, evidence poor
        client = _Client(HOME)
        why, _, _ = _run(client, rig, max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        np.testing.assert_allclose(
            client.started["cubes_init"]["L"], GOALS["L"], atol=1e-9,
            err_msg="a 1-inlier median seeded the tracker")
        # updates may FOLLOW a 1-face median now (user direction 2026-08-01),
        # but never further than the leash from the plan anchor
        for sent in client.cube_history:
            self.assertLessEqual(
                float(np.linalg.norm(sent["L"] - GOALS["L"])),
                R.MPC_LOWEV_LEASH_M + 1e-9,
                "a 1-face update pulled the goal off its leash")

    def test_a_credible_moving_cube_is_followed_and_judged_live(self):
        """THE feature: a median moving WITH evidence (inliers >= bar) must
        re-aim the tracker, and convergence must be judged against the LIVE
        goal. The mutant that freezes the goal at its seed — the original
        hardware bug, "the green line's endpoint does not follow the cube" —
        converges early against the stale seed and never sends the moved
        position; both assertions catch it (review 2026-08-01: no test moved
        a credible median, so exactly that mutant passed the whole suite)."""
        moved = {KEY_L: GOALS["L"] + np.array([0.03, 0.0, 0.0]),
                 KEY_R: GOALS["R"]}
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})   # inliers=3
        flip = {"done": False}
        orig_median = rig.median

        def median(key, now):
            if flip["done"]:
                return moved[key].copy()
            return orig_median(key, now)

        rig.median = median
        # Slow solver ON PURPOSE: with the MPC_LEAD_SCALE budget the default
        # pace catches the goal mid-slew and converges at ~0.5 s — legitimate
        # behavior, but this test pins that the FULLY moved median reaches the
        # tracker, so the arm must not be able to outrun the 0.05 m/s slew.
        client = _Client(HOME, step_rad=0.01)
        orig_step = client.mpc_step

        def step(measured_enc, cubes):
            flip["done"] = True
            return orig_step(measured_enc, cubes)

        client.mpc_step = step
        t0 = time.monotonic()
        why, hold, _ = _run(client, rig, max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        np.testing.assert_allclose(
            client.cube_history[-1]["L"], moved[KEY_L], atol=1e-9,
            err_msg="a credible moved median never reached the tracker")
        self.assertGreaterEqual(
            time.monotonic() - t0, 2.0,
            "converged early — convergence was judged against the stale seed, "
            "not the LIVE goal 30 mm away")

    def test_a_goal_drifting_out_mid_track_aborts(self):
        """The envelope is enforced per tick on the LIVE goal, not just the
        seed: a credibly-seen cube carried out of reach mid-track must abort,
        not be chased outside the validated volume."""
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})   # inliers=3
        flip = {"done": False}
        orig_median = rig.median

        def median(key, now):
            if flip["done"] and key == KEY_L:
                return GOALS["L"] + np.array([0.6, 0.0, 0.0])
            return orig_median(key, now)

        rig.median = median
        # Slow solver: with the MPC_LEAD_SCALE budget the default pace catches
        # the goal within MPC_CONVERGED_M during the first slew ticks and
        # converges — this test needs the goal to actually make it OUT of the
        # envelope, so the arm must trail the 0.05 m/s slew.
        client = _Client(HOME, step_rad=0.01)
        orig_step = client.mpc_step

        def step(measured_enc, cubes):
            flip["done"] = True
            return orig_step(measured_enc, cubes)

        client.mpc_step = step
        # Legacy pin: the immediate abort survives under --no-anchor-trust;
        # with trust ON the first offense returns the goal to the anchor
        # instead (pinned in the anchor-trust tests).
        why, hold, _ = _run(client, rig, max_s=15.0, args=_NoTrustArgs())
        self.assertIsNotNone(why)
        # With slewed goals the no-progress guard usually wins the race to
        # catch a carried-away cube; the envelope check is the backstop for
        # anything that gets a goal outside in one step. Both abort to the
        # retreat — either reason proves the mid-track protection.
        self.assertTrue("envelope" in why or "no progress" in why,
                        f"unexpected abort reason: {why}")
        self.assertGreaterEqual(len(client.cube_history), 1,
                                "aborted before any step ran — that is the "
                                "seed check, not the mid-track check")
        self.assertEqual(client.stopped, 1)

    def test_out_of_envelope_garbage_does_not_abort_a_frozen_goal(self):
        """The complement: the envelope must judge the FROZEN goal, not the
        raw median. One-face garbage far outside the envelope while the
        credible goal sits inside must not abort the track."""
        garbage = {KEY_L: GOALS["L"] + np.array([0.6, 0.0, 0.0]),
                   KEY_R: GOALS["R"]}
        rig = _Rig(garbage, inliers=1)
        client = _Client(HOME)
        why, hold, _ = _run(client, rig)
        self.assertIsNone(why, f"garbage PnP aborted the track: {why}")
        for sent in client.cube_history:
            np.testing.assert_allclose(sent["L"], GOALS["L"], atol=1e-9)

    def test_a_frozen_goal_is_tilt_compensated_to_stay_on_the_cube(self):
        """User requirement 2026-08-01: the goal is CUBE-anchored. When
        detections freeze, torso lean must not drag the goal with the base:
        the IMU gravity delta re-expresses it into the current base frame
        each tick (rotation about the ankle pivot). A 2 deg forward pitch at
        these reaches is ~32 mm — the dominant grasp error."""
        import math as _m
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        frozen = {"on": False}
        orig_median = rig.median

        def median(key, now):
            # ONLY the left cube goes dark. Freezing both would be a blind
            # tracker, which MPC_BLIND_GRASP now ends on the spot — and the
            # behaviour under test is what happens to a goal that is frozen
            # while the track CONTINUES, which is the real case: one hand
            # occludes its own cube (or the descent freeze rejects its 1-face
            # updates) while the peer still has live evidence.
            return None if (frozen["on"] and key == KEY_L) \
                else orig_median(key, now)

        rig.median = median
        client = _Client(HOME)
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        theta = _m.radians(2.0)
        g2 = [_m.sin(theta), 0.0, -_m.cos(theta)]   # base pitched forward 2 deg
        orig_step = client.mpc_step

        def step(measured_enc, cubes):
            frozen["on"] = True
            tel.g = list(g2)
            return orig_step(measured_enc, cubes)

        client.mpc_step = step
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 2.0, 3
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = keep
        self.assertIsNone(why, f"track failed: {why}")
        # independent expectation: rotation about -y by theta, ankle pivot
        piv = np.array([0.0, 0.0, -R.TILT_COMP_PIVOT_M])
        rot = np.array([[_m.cos(theta), 0.0, -_m.sin(theta)],
                        [0.0, 1.0, 0.0],
                        [_m.sin(theta), 0.0, _m.cos(theta)]])
        np.testing.assert_allclose(rot @ [0.0, 0.0, -1.0], g2, atol=1e-12)
        exp = rot @ (GOALS["L"] - piv) + piv
        np.testing.assert_allclose(
            client.cube_history[-1]["L"], exp, atol=1e-6,
            err_msg="the frozen goal was not re-expressed into the tilted "
                    "base frame")
        self.assertGreater(
            float(np.linalg.norm(client.cube_history[-1]["L"] - GOALS["L"])),
            0.02, "compensation did nothing — the goal rode the sway")

    def test_the_plan_anchor_returns_exactly_after_a_closed_tilt_loop(self):
        """Audit 2026-08-06 (D5): the anchor was rotated IN PLACE by the
        tick-to-tick gravity delta, and a product of minimal rotations is not
        the minimal rotation of the product — a base that sways in 2-D leaves a
        residual yaw about gravity equal to the swept solid angle. Unlike
        goal_pos nothing ever re-measures the anchor, so it walks off the 40 mm
        blind bar for the whole track. Driven around a CLOSED loop, an absolute
        rotation from a stored epoch must return it exactly."""
        import contextlib
        import io
        import math as _m
        th = _m.radians(4.0)
        loop = [(_m.sin(th) * _m.cos(a), _m.sin(th) * _m.sin(a), -_m.cos(th))
                for a in np.linspace(0.0, 2.0 * _m.pi, 24)]
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        # Start OFF the 08-13 handoff-close bars so the session opens, AND
        # outside GRASP_EE_OK so the 08-13 escape guard never arms — this
        # test is about the anchor's rotation invariant and needs all 24
        # tilt steps to run, not an early close.
        tel = _Tel({n: HOME[n] + (0.6 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        seen = {"n": 0, "anchor": []}
        orig_step = client.mpc_step

        def step(measured_enc, cubes):
            tel.g = list(loop[min(seen["n"], len(loop) - 1)])
            seen["n"] += 1
            return orig_step(measured_enc, cubes)

        client.mpc_step = step
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 2.0, 999
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                R.mpc_track(ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = keep
        self.assertGreaterEqual(seen["n"], len(loop),
                                "the tilt loop never completed")
        # The invariant itself, on the pure function, so the failure is legible
        # rather than an integration wobble: composing the per-tick minimal
        # rotations around a closed loop does NOT return the point, while one
        # absolute rotation from the epoch does. Both use the same pivot the
        # tracker uses.
        piv = np.array([0.0, 0.0, -R.TILT_COMP_PIVOT_M])
        p0 = np.asarray(GOALS["L"], float)
        composed = p0.copy()
        prev = np.asarray(loop[0], float)
        for g in loop[1:]:
            rot = R._rot_between(prev, np.asarray(g, float))
            if rot is not None:
                composed = rot @ (composed - piv) + piv
            prev = np.asarray(g, float)
        rot_abs = R._rot_between(np.asarray(loop[0], float),
                                 np.asarray(loop[-1], float))
        absolute = p0.copy() if rot_abs is None else rot_abs @ (p0 - piv) + piv
        np.testing.assert_allclose(
            absolute, p0, atol=1e-9,
            err_msg="the absolute rotation did not close the loop")
        self.assertGreater(
            float(np.linalg.norm(composed - p0)), 1e-4,
            "composing per-tick deltas closed the loop too — the defect this "
            "pin describes would then not exist and the fix is pointless")

    def test_live_goal_markers_stream_to_the_monitor(self):
        """The green route is a plan-time snapshot riding base_link; the
        monitor can only show LOCK through the live-goal markers. They must
        stream during the track, one per tracked hand."""
        class _Rec:
            def __init__(self):
                self.msgs = []

            def publish(self, m):
                self.msgs.append(m)

        rec = _Rec()
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, _ = _run(client, rig, route_pub=rec)
        self.assertIsNone(why, f"track failed: {why}")
        gm = [m for m in rec.msgs if m.get("goals_only")]
        self.assertTrue(gm, "no live-goal markers streamed to the monitor")
        self.assertEqual(set(gm[-1]["goals"]), {"left", "right"})
        np.testing.assert_allclose(gm[-1]["goals"]["left"], GOALS["L"],
                                   atol=1e-6)

    def test_one_face_updates_move_the_goal_on_a_leash(self):
        """User direction 2026-08-01: one tag face is evidence too. A 1-face
        median 3 cm off the plan anchor must be followed (slewed) — that is
        the tilt-drift correction — while the 30 cm ghost stays rejected
        (test_occlusion_freezes_the_goal pins that side)."""
        near = {KEY_L: GOALS["L"] + np.array([0.03, 0.0, 0.0]),
                KEY_R: GOALS["R"]}
        rig = _Rig(near, inliers=1)
        client = _Client(HOME)
        why, hold, _ = _run(client, rig, max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        np.testing.assert_allclose(
            client.cube_history[-1]["L"], near[KEY_L], atol=1e-9,
            err_msg="a leashed 1-face correction was not followed")

    def test_one_face_z_rides_the_table_prior(self):
        """08-13 (user): while the MPC runs, a 1-face median trusts the TABLE
        for its vertical — xy is followed as before, z snaps to the plan
        anchor's z, which build_scene set from the table-resting prior. The
        1-face vertical is the worst number in the pipeline (12-25 mm off,
        measured 08-12); the table is physics."""
        near = {KEY_L: GOALS["L"] + np.array([0.03, 0.0, 0.03]),
                KEY_R: GOALS["R"]}
        rig = _Rig(near, inliers=1)
        rig._z_prior_keys = {KEY_L, KEY_R}
        client = _Client(HOME)
        why, _, _ = _run(client, rig, max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        sent = client.cube_history[-1]["L"]
        # xy is SLEWED toward the 1-face fix (the track may converge before
        # the 50 mm/s slew finishes — direction is the contract, not arrival)
        self.assertGreater(
            float(sent[0]), float(GOALS["L"][0]) + 0.005,
            "the 1-face xy correction was lost to the z snap")
        self.assertAlmostEqual(
            float(sent[2]), float(GOALS["L"][2]), places=9,
            msg="a 1-face z 30 mm off the resting height reached the tracker")

    def test_two_face_z_keeps_its_own_vertical(self):
        """>=2 faces are well-conditioned in z (user 08-13: '2-side PnP ->
        use the cube's height info'): the snap must not fire."""
        moved = {KEY_L: GOALS["L"] + np.array([0.0, 0.0, 0.03]),
                 KEY_R: GOALS["R"]}
        rig = _Rig(moved, inliers=3)
        rig._z_prior_keys = {KEY_L, KEY_R}
        client = _Client(HOME)
        why, _, _ = _run(client, rig, max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        self.assertAlmostEqual(
            float(client.cube_history[-1]["L"][2]),
            float(moved[KEY_L][2]), places=9,
            msg="a 2-face vertical was overridden by the table prior")

    def test_no_detected_table_keeps_the_vision_z(self):
        """08-13 (user): 'if tables are not detected, just use the cube's
        height'. A rig with no prior claim behind the target (build_scene
        stashes an empty set when no sane table was seen) must pass the
        1-face vertical through untouched."""
        near = {KEY_L: GOALS["L"] + np.array([0.0, 0.0, 0.03]),
                KEY_R: GOALS["R"]}
        rig = _Rig(near, inliers=1)     # no _z_prior_keys on the rig at all
        client = _Client(HOME)
        why, _, _ = _run(client, rig, max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        # directional: the goal z must MOVE toward the vision z (a snap
        # would have pinned it at the anchor's z for the whole track)
        self.assertGreater(
            float(client.cube_history[-1]["L"][2]),
            float(GOALS["L"][2]) + 0.005,
            "the z snap fired with no table prior behind the target")

    def test_a_z_far_off_resting_is_not_snapped(self):
        """Past TABLE_Z_PRIOR_CAP_M the cube is NOT resting on the slab
        (lifted, stacked, knocked off) — the vision z is the truth and the
        snap stands down, mirroring build_scene's cap. The offset stays
        inside MPC_LOWEV_LEASH_M so the leash is not what rejects it."""
        off = R.TABLE_Z_PRIOR_CAP_M + 0.02
        self.assertLess(off, R.MPC_LOWEV_LEASH_M)
        high = {KEY_L: GOALS["L"] + np.array([0.0, 0.0, off]),
                KEY_R: GOALS["R"]}
        rig = _Rig(high, inliers=1)
        rig._z_prior_keys = {KEY_L, KEY_R}
        client = _Client(HOME)
        _why, _, _ = _run(client, rig, max_s=2.0)
        self.assertGreater(
            float(client.cube_history[-1]["L"][2]),
            float(GOALS["L"][2]) + 0.005,
            "a beyond-the-cap vertical was snapped onto the table")

    def test_the_mpc_world_carries_no_tables(self):
        """User 08-13: post-plan the table is a HEIGHT SENSOR, not a force
        field. Plan-0 certifies the route against the slab; carrying it into
        the session put its 70 mm collision-activation band under the grasp
        point and the descent equilibrated 13-20 mm high until the T14
        budget closed anyway. The session scene must strip the slab (other
        cubes stay as obstacles)."""
        class _TableArgs(_Args):
            tables = ("lab_table_a",)

        class _SlabRig(_Rig):
            def table_pose(self, key, now):
                return (np.array([0.5, 0.0, 0.05]),
                        np.array([1.0, 0.0, 0.0, 0.0]), 0.3)

        rig = _SlabRig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        client = _Client(HOME)
        why, _, _ = _run(client, rig, args=_TableArgs(), max_s=2.0)
        self.assertIsNone(why, f"track failed: {why}")
        self.assertNotIn("lab_table_a", client.started["scene"],
                         "the slab reached the MPC session as an obstacle")

    def test_single_cube_tracks_one_side_and_freezes_the_other_arm(self):
        """Journey legs are single-cube with the OTHER arm possibly holding an
        earlier cube. Server-side the untracked arm is only soft-pinned in
        task space — its q_next wanders — so the publish must freeze it at the
        handoff snapshot while the tracked arm follows the solver."""
        wander = dict(HOME)
        wander[ARM_JOINTS_R[0]] += 0.1        # solver wanders the untracked arm
        client = _Client(wander)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, ctx = _run(client, rig, targets={KEY_L: GOALS["L"]})
        self.assertIsNone(why, f"single-side track failed: {why}")
        self.assertEqual(client.started["sides"], ["L"])
        self.assertTrue(ctx.pub.sent)
        for pkt in ctx.pub.sent:
            self.assertEqual(set(pkt["arm_targets"]), {"left", "right"})
            for n, v in zip(ARM_JOINTS_R,
                            pkt["arm_targets"]["right"]["joint_pos"]):
                self.assertAlmostEqual(
                    v, HOME[n], places=9,
                    msg="the untracked arm moved — it must stay frozen at "
                        "the handoff snapshot")

    def test_hold_is_the_sagged_measurement_not_the_command(self):
        """Under payload the arm tracks its command with a deficit. The
        returned hold must be where the arm IS — the sagged telemetry — not
        the last command: grasping at the command closes the gripper above
        the cube. A perfect-follower rig cannot see the difference, hence
        the explicit sag (review 2026-08-01).

        The sag must stay BELOW the clamp's lead budget
        (MPC_LEAD_SCALE*max_rate*PUB_DT): the clamp anchors at the MEASUREMENT, so a
        persistent deficit larger than the budget ratchets the published
        target downhill each tick until the sanity bound aborts — by design
        (it is an arm that cannot hold its commanded posture), and one more
        reason --grav-comp is mandatory on hardware."""
        n = ARM_JOINTS_L[3]                    # left elbow
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, ctx = _run(client, rig, tel_sag={n: 0.01})
        self.assertIsNone(why, f"track failed: {why}")
        self.assertAlmostEqual(hold[n], ctx.tel.enc[n], places=9,
                               msg="hold is not the measured (sagged) joint")
        last_cmd = ctx.pub.sent[-1]["arm_targets"]["left"]["joint_pos"][3]
        self.assertGreater(abs(hold[n] - last_cmd), 0.008,
                           "hold tracked the command, not the measurement")

    def test_the_clearance_floor_cli_reaches_the_per_tick_gate(self):
        """--clearance-floor-mm must reach StepGate: an operator raising the
        floor for a cautious session changes what the per-tick gate enforces.
        Only the plumb is pinned here (the recorder swaps the default floor
        back in so the track behaves normally); sweep behavior itself is
        pinned by test_curobo_bridge."""
        seen = {}
        real_gate = R.StepGate

        def recorder(*args, **kw):
            seen.update(kw)
            kw["clearance_floor_m"] = _Args.clearance_floor_mm / 1000.0
            return real_gate(*args, **kw)

        class _A(_Args):
            clearance_floor_mm = 17.0

        R.StepGate = recorder
        try:
            client = _Client(HOME)
            rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
            why, hold, _ = _run(client, rig, args=_A())
            self.assertIsNone(why, f"track failed: {why}")
        finally:
            R.StepGate = real_gate
        self.assertAlmostEqual(seen.get("clearance_floor_m", -1.0), 0.017,
                               msg="the CLI clearance floor never reached "
                                   "the per-tick gate")

    def test_the_session_build_is_pumped_not_silent(self):
        """mpc_start is a multi-second GPU cold start server-side. The arms
        must be HELD through it: an unpumped session build is a silent
        commander, and past 0.5 s the failsafe crawls the extended arms away
        from the handoff posture (review 2026-08-01)."""
        client = _Client(HOME)
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        start_pose = dict(tel.enc)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        box = {}
        orig_start = client.mpc_start

        def slow_start(*args, **kw):
            time.sleep(0.35)
            box["during_warm"] = len(ctx.pub.sent)
            return orig_start(*args, **kw)

        client.mpc_start = slow_start
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 6.0, 3
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = keep
        self.assertIsNone(why, f"track failed: {why}")
        n = box["during_warm"]
        self.assertGreaterEqual(n, 3, "the session build went silent — no "
                                      "hold packets while the server warmed")
        for pkt in ctx.pub.sent[:n]:
            for arm, names in (("left", ARM_JOINTS_L), ("right", ARM_JOINTS_R)):
                for j, v in zip(names, pkt["arm_targets"][arm]["joint_pos"]):
                    self.assertAlmostEqual(v, start_pose[j], places=9,
                                           msg="the warm hold moved the arm")

    def test_a_clearance_standoff_aborts_early(self):
        """Hardware 2026-08-01 run 2: the solver's path threaded 24 mm from
        the torso against a 25 mm floor — 21 identical gate rejections with
        the arm frozen mid-air. A clearance rejection is a standoff, not
        noise; MPC_CLEARANCE_STANDOFF_MAX ticks are proof enough."""
        import types
        real_gate = R.StepGate

        class _StandoffGate:
            def __init__(self, *a, **k):
                pass

            def check(self, cand, enc, dt):
                return types.SimpleNamespace(
                    ok=False, why="clearance 24.3 mm < floor 25 mm at "
                                  "elbow_L_collision ~ base_link_collision1",
                    where=("elbow_L_collision", "base_link_collision1"))

        R.StepGate = _StandoffGate
        try:
            client = _Client(HOME)
            rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
            # start_off 0.6 rad ~= 82 mm EE — OUTSIDE GRASP_EE_OK, so the
            # 08-13 standoff-close rule stands aside and the abort is pinned
            why, hold, _ = _run(client, rig, start_off=0.6)
        finally:
            R.StepGate = real_gate
        self.assertIsNotNone(why)
        self.assertIn("clearance standoff", why)
        self.assertEqual(why.code, "T10", "the standoff abort lost its code")
        self.assertIn("elbow_L_collision", why.culprit,
                      "the verdict must name the geom pair that stood off")
        self.assertLessEqual(len(client.cube_history),
                             R.MPC_CLEARANCE_STANDOFF_MAX + 2,
                             "the standoff was not aborted early")
        self.assertEqual(client.stopped, 1)

    def test_a_clearance_standoff_in_reach_closes_instead_of_retreating(self):
        """User 08-13, after four hardware T10 retreats with the hand ~20 mm
        from the cube ('just let it grasp!'): the MPC's elbow preference
        sits 1-3 cm below our floor in OUR model, so the terminal press ends
        frozen mid-air with the jaws already straddling the cube. A standoff
        INSIDE GRASP_EE_OK must close where it stands (default start_off 0.3
        rad ~= 41 mm, inside 60 mm); only a far hand still retreats — the
        previous test pins that side."""
        import types
        real_gate = R.StepGate

        class _StandoffGate:
            def __init__(self, *a, **k):
                pass

            def check(self, cand, enc, dt):
                return types.SimpleNamespace(
                    ok=False, why="clearance 4.6 mm < floor 5 mm at "
                                  "elbow_R_collision ~ base_link_collision0",
                    where=("elbow_R_collision", "base_link_collision0"))

        R.StepGate = _StandoffGate
        try:
            client = _Client(HOME)
            rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
            why, hold, _ = _run(client, rig)
        finally:
            R.StepGate = real_gate
        self.assertIsNone(why, f"an in-reach standoff still retreated: {why}")
        self.assertIsNotNone(hold, "no grasp hold returned")
        self.assertLessEqual(len(client.cube_history),
                             R.MPC_CLEARANCE_STANDOFF_MAX + 2,
                             "the close was not decided at the standoff")
        self.assertEqual(client.stopped, 1)

    def test_a_stall_inside_grasp_tolerance_grasps_instead_of_retreating(self):
        """User 2026-08-02 ('almost there, then it retracts' — correct
        fury): a stalled arm INSIDE GRASP_EE_OK closes the gripper where it
        stands — the same bet the budget path has always made — instead of
        walking away. 0.3 rad shoulder freeze ~= 41 mm EE, inside 60 mm."""
        client = _Client(HOME)
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        orig_follow = tel.follow
        seen = {"n": 0}

        def wedge(slots):
            seen["n"] += 1
            if seen["n"] <= 3:
                orig_follow(slots)

        tel.follow = wedge
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 8.0, 3
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = keep
        self.assertIsNone(why, f"a graspable stall retreated: {why}")
        self.assertIsNotNone(hold, "no hold posture returned for the grasp")

    def test_solver_dither_near_convergence_is_not_convicted(self):
        """Hardware 2026-08-04: 'moved 0.28 of 1.89 rad (15%)' with the
        binding joint lagging 0.07 rad against a 0.152 rad ceiling — the arm
        was ON its target and the asked-sum was command OSCILLATION near
        convergence. A conviction now also requires the binding lag to reach
        MPC_EXEC_LAG_FRACTION of the clamp lead; a tiny-lag low-ratio window
        must keep tracking and converge."""
        client = _Client(HOME)
        orig_step = client.mpc_step
        flip = {"n": 0}

        def dither(measured_enc, cubes):
            out = orig_step(measured_enc, cubes)
            flip["n"] += 1
            j = ARM_JOINTS_L[0]
            out["enc"][j] = measured_enc[j] + (0.02 if flip["n"] % 2 else -0.02)
            return out

        client.mpc_step = dither
        tel = _Tel({n: HOME[n] + (0.012 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        tel.follow = lambda slots: None       # arm holds still near the goal
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 4.0, 3
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = keep
        self.assertIsNone(why, f"a dithering solver was convicted: {why}")

    def test_a_slow_but_moving_arm_is_not_convicted_as_wedged(self):
        """Hardware 2026-08-05, twice, both arms of the same family: the hand
        151/158 mm out, the solver asking 1.12 rad/s, the arm delivering 0.22
        rad/s at a binding lag 70% of the clamp ceiling — convicted at exactly
        the 20% bar and retreated.

        That ratio is ARCHITECTURE, not a wedge. `cand = enc + clip(raw - enc,
        +-MPC_LEAD_SCALE*max_rate*dt)` caps the commanded lead, the wire
        carries no velocity reference so kd brakes toward standstill, and the
        arm equilibrates wherever kp*lag balances that — while the solver asks
        for the full streaming rate the whole time the hand is far out. So the
        ratio is structurally low EXACTLY when there is the most ground to
        cover, and this guard cannot be the thing that judges it.

        An arm still covering ground belongs to the progress guard and the
        budget behind it — both of which grasp anyway inside GRASP_EE_OK_M
        instead of walking away."""
        client = _Client(HOME, step_rad=0.010)
        tel = _Tel({n: HOME[n] + (1.0 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        # The arm executes at most 0.006 rad per packet -> ~0.12 rad/s, over
        # the 0.10 floor, while the clamp asks up to 0.045 every tick: a ~16%
        # ratio. It starts 1.0 rad out so 3 s at that speed still leaves the
        # hand well outside GRASP_EE_OK — otherwise the budget path grasps
        # anyway and the test proves nothing.
        def governed(slots):
            for arm, names in (("left", ARM_JOINTS_L), ("right", ARM_JOINTS_R)):
                if arm in slots:
                    for n, v in zip(names, slots[arm]["joint_pos"]):
                        d = float(v) - tel.enc[n]
                        tel.enc[n] += float(np.clip(d, -0.006, 0.006))

        tel.follow = governed
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_EXEC_GUARD)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 3.0, 3
        R.MPC_EXEC_GUARD = True     # the SPEED FLOOR must be what saves this,
                                    # not the guard being switched off
        import contextlib
        import io
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_EXEC_GUARD = keep
        self.assertIsNotNone(why, "the 3 s budget should have ended this")
        self.assertNotEqual(getattr(why, "code", None), "T12",
                            f"a moving arm was convicted as wedged: {why}")
        self.assertEqual(why.code, "T14",
                         f"expected the budget to own this verdict: {why}")
        said = out.getvalue()
        self.assertIn("MOVING at", said, "the acquittal went unsaid")
        self.assertIn("NOT wedged", said)

    def test_with_the_guard_off_a_wedge_still_aborts_via_the_progress_guard(self):
        """MPC_EXEC_GUARD off (user 2026-08-05) withholds the T12 VERDICT, not
        the evidence and not the safety. The same frozen arm must still leave
        the track — via MPC_NO_PROGRESS_S, whose question ('is the HAND
        closing on the cube?') a wedged arm fails just as hard — and the line
        the abort would have carried must still be printed, so the runs that
        would have retreated stay countable in the log.

        This is the whole cost of the switch: the retreat comes later, not
        never."""
        client = _Client(HOME, step_rad=0.005)
        tel = _Tel({n: HOME[n] + (0.55 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        orig_follow = tel.follow
        seen = {"n": 0}

        def wedge(slots):
            seen["n"] += 1
            if seen["n"] <= 3:
                orig_follow(slots)

        tel.follow = wedge
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_NO_PROGRESS_S,
                R.MPC_EXEC_GUARD)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 12.0, 3
        R.MPC_NO_PROGRESS_S = 3.0              # keep the test short
        R.MPC_EXEC_GUARD = False               # the shipped default
        import contextlib
        import io
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_NO_PROGRESS_S,
             R.MPC_EXEC_GUARD) = keep
        self.assertIsNotNone(why, "a wedged arm was tracked forever")
        self.assertNotEqual(why.code, "T12", "the guard is supposed to be off")
        # T09 joined the accepted set on 08-12: with anchor trust, the first
        # T13 conviction returns the goal home instead of aborting, and a
        # truly wedged arm then trips the MPC-jump guard (T09) — the
        # guarantee under test is "a wedge cannot loop forever", not which
        # guard files the paperwork.
        self.assertIn(why.code, ("T09", "T13", "T14"),
                      f"a wedge escaped every remaining guard: {why}")
        said = out.getvalue()
        self.assertIn("T12 DISABLED", said,
                      "the withheld verdict went unprinted — the evidence is "
                      "the reason the switch is survivable")
        self.assertIn("NOT executing", said)

    def test_a_wedged_arm_aborts_via_the_executor_gap(self):
        """Forensics 2026-08-01: a wedged arm is asked a clamp step every
        tick and moves nothing. The executor-gap guard names it in ~1 s —
        and OUTSIDE the grasp tolerance (0.55 rad ~= 75 mm) it still aborts
        to the retreat. (Slow solver steps keep the raw command inside the
        sanity bound so the executor guard, not the insane-command abort,
        is what fires.)"""
        client = _Client(HOME, step_rad=0.005)
        tel = _Tel({n: HOME[n] + (0.55 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        orig_follow = tel.follow
        seen = {"n": 0}

        def wedge(slots):                   # the arm freezes after 3 packets
            seen["n"] += 1
            if seen["n"] <= 3:
                orig_follow(slots)

        tel.follow = wedge
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_EXEC_GUARD)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 6.0, 3
        R.MPC_EXEC_GUARD = True          # this test IS the guard's own logic
        t_start = time.monotonic()
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_EXEC_GUARD = keep
        self.assertIsNotNone(why, "a frozen arm was tracked to the budget")
        self.assertIn("NOT executing", why)
        self.assertLess(time.monotonic() - t_start, 4.0,
                        "the executor gap did not beat the no-progress guard")

    def test_the_target_leads_a_lagging_arm_beyond_one_tick(self):
        """Forensics 2026-08-02 (rear-sector 13-20% execution): anchoring the
        target AT the measurement capped motor PD error at ONE tick's step
        while the wire's zero velocity-ref means kd brakes commanded motion,
        so ~1 N*m of harness drag starved the arm. The published target must
        LEAD a lagging arm beyond one tick's step (authority restored, up to
        MPC_LEAD_SCALE ticks), the per-tick gate must ACCEPT that lead
        rather than reject it as a jump, and the eventual stall verdict must
        name the binding joint."""
        client = _Client(HOME, step_rad=0.005)
        tel = _Tel({n: HOME[n] + (0.55 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        tel.follow = lambda slots: None     # the arm never moves at all
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_EXEC_GUARD)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 6.0, 3
        R.MPC_EXEC_GUARD = True        # the verdict asserted below IS T12's
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS, R.MPC_EXEC_GUARD = keep
        self.assertIsNotNone(why)
        self.assertIn("NOT executing", why,
                      "a leading command must be gated and followed, not "
                      f"rejected as a jump (got: {why})")
        self.assertIn(ARM_JOINTS_L[0], why,
                      "the stall verdict must name the binding joint")
        jl = ARM_JOINTS_L.index(ARM_JOINTS_L[0])
        lead_max = max(
            abs(pkt["arm_targets"]["left"]["joint_pos"][jl]
                - tel.enc[ARM_JOINTS_L[0]])
            for pkt in ctx.pub.sent)
        one_tick = _Args.max_rate * 0.05
        self.assertGreater(lead_max, 1.5 * one_tick,
                           "the target never led the frozen arm beyond one "
                           "tick's step — PD authority is still collapsed")
        self.assertLessEqual(lead_max, R.MPC_LEAD_SCALE * _Args.max_rate * 0.25,
                             "the lead exceeded the scaled budget "
                             "(generous dt-jitter allowance)")

    def test_staged_hover_chases_above_then_descends_onto_the_cube(self):
        """User 2026-08-03 + adversarial review F1: the hover must live in
        the TRACKER's goals (the plan-0 endpoint lift never executes past
        the 80% handoff). Phase 1 goals sent to the solver sit exactly
        reach_hover_mm above the live cube; once the hand arrives at the
        hover the side flips to phase 2 and the goals become the cube
        itself; convergence still completes on the TRUE goal."""
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, ctx = _run(client, rig, args=_HoverArgs())
        self.assertIsNone(why, f"staged-hover track failed: {why}")
        first, last = client.cube_history[0], client.cube_history[-1]
        for s in ("L", "R"):
            self.assertAlmostEqual(float(first[s][2]),
                                   float(GOALS[s][2]) + 0.03, places=6,
                                   msg=f"{s}: phase-1 goal is not 30 mm "
                                       f"above the cube")
            np.testing.assert_allclose(first[s][:2], GOALS[s][:2],
                                       err_msg=f"{s}: hover moved the goal "
                                               f"in XY")
            self.assertAlmostEqual(float(last[s][2]), float(GOALS[s][2]),
                                   places=6,
                                   msg=f"{s}: never descended — the gripper "
                                       f"would close 30 mm above the cube")
            # phase 2 is a RAMP down the vertical rail, never a jump: no
            # single tick may drop the commanded z faster than the rate
            zs = np.array([float(h[s][2]) for h in client.cube_history])
            drops = -np.diff(zs)
            # The ramp now advances on the MEASURED tick (a nominal PUB_DT made
            # a 15 mm rail take 0.78 s at the p90 130 ms this loop really runs
            # at), bounded at 2x nominal so a starved tick cannot commit a
            # 6.5 mm z step on exactly the ticks the rail exists to smooth.
            self.assertLessEqual(float(drops.max()),
                                 R.MPC_DESCENT_M_S * 2.0 * R.PUB_DT + 1e-9,
                                 f"{s}: the goal z JUMPED down instead of "
                                 f"riding the descent rail")

    class _CrawlRig(_Rig):
        """1-face median crawling sideways ~0.5 mm/tick — the occlusion-slide
        signature the descent freeze was built against."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.crawl_n = 0

        def median(self, key, now):
            v = super().median(key, now)
            if v is None or key != KEY_L:
                return v
            self.crawl_n += 1
            v[0] += 0.00025 * self.crawl_n
            return v

    def _run_crawl(self):
        client = _Client(HOME)
        rig = self._CrawlRig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]},
                             inliers=1)
        # far start: 0.3 rad puts the EE only ~15 mm out, inside the 35 mm
        # hover-arrival ball — phase 1 would last one tick and prove nothing
        why, hold, ctx = _run(client, rig, args=_HoverArgs(), start_off=0.6,
                              max_s=10.0)
        self.assertIsNone(why, f"crawling-median track failed: {why}")
        last_goal_x = float(client.cube_history[-1]["L"][0])
        final_med_x = float(GOALS["L"][0]) + 0.00025 * rig.crawl_n
        self.assertGreater(last_goal_x, float(GOALS["L"][0]) + 0.003,
                           "phase 1 never followed the live median — that "
                           "is the tracking feature itself")
        return last_goal_x, final_med_x

    def test_the_descent_freeze_is_the_default_again(self):
        """2026-08-09: the 08-08 1-face override is revoked by its own night
        of data (goals dragged 37-105 mm, T13 x5; every successful grasp went
        blind-early-close-on-anchor instead). The freeze is the default."""
        self.assertFalse(R.MPC_DESCENT_1FACE_TRACKS)

    def test_descent_tracks_one_face_updates_with_the_override(self):
        """The 08-08 override mode stays available behind the flag: with
        MPC_DESCENT_1FACE_TRACKS forced True a 1-face median keeps moving the
        goal THROUGH the descent — still slewed and leashed, never frozen."""
        prev = R.MPC_DESCENT_1FACE_TRACKS
        R.MPC_DESCENT_1FACE_TRACKS = True
        try:
            last_goal_x, final_med_x = self._run_crawl()
        finally:
            R.MPC_DESCENT_1FACE_TRACKS = prev
        self.assertLess(final_med_x - last_goal_x, 0.003,
                        "the goal stopped following the 1-face median during "
                        "the descent — the override is not tracking")

    def test_descent_freeze_restored_when_the_override_is_off(self):
        """2026-08-03, 'the left gripper veers on every descent': with the
        override OFF, a crawling 1-face median must be FOLLOWED during
        phase 1 (that is live tracking) and IGNORED during the descent
        (goal xy frozen until >=2-face evidence returns)."""
        prev = R.MPC_DESCENT_1FACE_TRACKS
        R.MPC_DESCENT_1FACE_TRACKS = False
        try:
            last_goal_x, final_med_x = self._run_crawl()
        finally:
            R.MPC_DESCENT_1FACE_TRACKS = prev
        self.assertGreater(final_med_x - last_goal_x, 0.003,
                           "the goal kept chasing the 1-face slide during "
                           "the descent — the veer bug is back")

    def test_descent_reanchors_onto_the_last_two_face_fix(self):
        """2026-08-03, 'grasps left of the cube every time': 1-face PnP
        carries a DIRECTION-CONSTANT lateral bias, phase 1 legitimately
        follows it on the leash, and the top-down descent then lands the
        bias in full. At the descent handover the goal must snap back to
        the freshest >=2-face fix when one exists — and with the descent
        FREEZE (override off) it must STAY there."""
        prev = R.MPC_DESCENT_1FACE_TRACKS
        R.MPC_DESCENT_1FACE_TRACKS = False
        try:
            last_goal_y = self._run_decay_reanchor()
        finally:
            R.MPC_DESCENT_1FACE_TRACKS = prev
        self.assertAlmostEqual(last_goal_y, float(GOALS["L"][1]), places=3,
                               msg="the descent kept the 1-face lateral bias "
                                   "instead of re-anchoring onto the last "
                                   "2-face fix — the left-of-cube grasp is "
                                   "back")

    def test_with_the_override_the_live_median_outranks_the_reanchor(self):
        """2026-08-08 user override: the handover still snaps to the last
        2-face fix, but the descent keeps LIVE-tracking, so continuing
        1-face updates win over the stale snapshot — the goal ends on the
        live median, bias and all. That trade is the override's explicit
        content ('the tag is large enough, almost no flipping'). Behind the
        flag since 08-09 — forced on here."""
        prev = R.MPC_DESCENT_1FACE_TRACKS
        R.MPC_DESCENT_1FACE_TRACKS = True
        try:
            last_goal_y = self._run_decay_reanchor()
        finally:
            R.MPC_DESCENT_1FACE_TRACKS = prev
        self.assertAlmostEqual(last_goal_y, float(GOALS["L"][1]) + 0.025,
                               places=3,
                               msg="the goal did not follow the live 1-face "
                                   "median through the descent — the "
                                   "override is not tracking")

    def _run_decay_reanchor(self) -> float:
        """Solid 2-face fixes first, then 1-face estimates parked 25 mm to
        the LEFT (the systematic single-face bias). Returns the goal y the
        client was last asked to track."""
        state = {"calls": 0}

        class _DecayRig(_Rig):
            def median(self, key, now):
                v = super().median(key, now)
                if v is None or key != KEY_L:
                    return v
                state["calls"] += 1
                if state["calls"] > 6:
                    self.inliers[KEY_L] = 1
                    v[1] += 0.025
                return v

        client = _Client(HOME)
        rig = _DecayRig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, inliers=2)
        # Legacy flag: an anchor-trust conviction mid-scenario would freeze
        # evidence-following and defeat the override under test.
        why, hold, ctx = _run(client, rig, args=_NoTrustHoverArgs(),
                              start_off=0.6, max_s=10.0)
        self.assertIsNone(why, f"re-anchor track failed: {why}")
        return float(client.cube_history[-1]["L"][1])

    def test_a_slow_but_moving_arm_is_not_convicted(self):
        """Hardware 2026-08-01 night, seven straight false aborts: the arms
        executed at ~50% of the commanded rate on final approach (drift
        already down to 7-15 mm) and the integral-form guard convicted them
        exactly when the hand was almost there. Ratio form: 50% execution is
        a slow finish, not a wedge — the track must converge."""
        client = _Client(HOME)
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        orig_follow = tel.follow

        def half_rate(slots):               # executes half of every step
            before = dict(tel.enc)
            orig_follow(slots)
            for n in ARM_JOINTS_ALL:
                tel.enc[n] = before[n] + 0.5 * (tel.enc[n] - before[n])

        tel.follow = half_rate
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_CONVERGED_TICKS)
        R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = 8.0, 3
        try:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_CONVERGED_TICKS = keep
        self.assertIsNone(why, f"a slow finisher was convicted: {why}")

    def test_the_stream_aborts_when_the_arm_stops_following(self):
        """Hardware 2026-08-01 night: the left elbow wedged against the torso
        mid-corridor — encoders live, current gentle, no watchdog fired — and
        the stream played to the end with the arm stuck at mid-reach; the MPC
        then inherited a wedged arm and reported 'no progress, 345 mm'. The
        stream must verify FOLLOWING, not just publish, and back out at the
        wedge."""
        jn = list(ARM_JOINTS_ALL)
        q0 = np.array([HOME[n] for n in jn])
        q1 = q0.copy()
        q1[0] += 2.0                        # far route: the frozen arm's lag
                                            # must clear the 0.8 rad bar
                                            # (raised 2026-08-04) AND sustain
                                            # it 10 ticks before the route ends
        bundle = _RouteBundle(q0, q1, duration_s=1.5)
        tel = _Tel(dict(HOME))
        orig_follow = tel.follow
        seen = {"n": 0}

        def wedge(slots):                   # the arm stops after 5 packets
            seen["n"] += 1
            if seen["n"] <= 5:
                orig_follow(slots)

        tel.follow = wedge
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, played = R.stream_route(bundle, _Pub(tel), tel, _Det(), rig,
                                     None, 0.3)
        self.assertIsNotNone(why, "a non-following arm streamed to the end")
        self.assertIn("not following", why)
        self.assertLess(played, bundle.duration_s,
                        "the abort came only at the end of the route")

    def test_a_following_arm_streams_clean(self):
        jn = list(ARM_JOINTS_ALL)
        q0 = np.array([HOME[n] for n in jn])
        q1 = q0.copy()
        q1[0] += 0.3
        bundle = _RouteBundle(q0, q1, duration_s=0.6)
        tel = _Tel(dict(HOME))
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, played = R.stream_route(bundle, _Pub(tel), tel, _Det(), rig,
                                     None, 0.3, until_s=0.5)
        self.assertIsNone(why, f"clean stream aborted: {why}")

    def test_the_handoff_fraction_is_what_makes_the_stream_settle(self):
        """Hardware 2026-08-06, the T13 x2 session. The arms are driven at
        max_rate by a position loop with NO velocity feedforward, so they can
        only hold a speed by lagging — the lag IS the torque. Measured in that
        run: K = 1.7-3.7 rad/s per rad, i.e. 0.27-0.60 rad of lag to follow the
        stream, worth 116-254 mm of hand error. Handing over at 0.8 passed that
        straight to the tracker (observed 165 and 233 mm) where the ABSOLUTE
        5 mm / 6 s progress guard convicted a hand that was still closing.

        At MPC_HANDOFF_FRACTION == 1.0, `until_s` no longer truncates the
        route, so stream_route falls through to its settle loop and the tracker
        inherits a SETTLED arm instead of the lag. That coupling is invisible —
        it lives in one `end_s < bundle.duration_s` branch — so it is pinned
        here: lowering the fraction silently reintroduces the failure."""
        self.assertGreaterEqual(
            R.MPC_HANDOFF_FRACTION, 1.0,
            "handing over before the end of the route skips the settle loop, "
            "so the tracker inherits the streaming following-error (165-233 mm "
            "on hardware 2026-08-06) instead of a settled arm")

        jn = list(ARM_JOINTS_ALL)
        q0 = np.array([HOME[n] for n in jn])
        q1 = q0.copy()
        q1[0] += 0.3
        bundle = _RouteBundle(q0, q1, duration_s=0.4)

        # A LAGGING arm: it follows, but always trails its streamed target.
        # Truncated handoff must return with that lag intact; the full-route
        # handoff must not return until the lag has been settled out.
        def rig():
            return _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})

        tel_cut = _Tel(dict(HOME), sag={jn[0]: 0.12})
        why, played = R.stream_route(bundle, _Pub(tel_cut), tel_cut, _Det(),
                                     rig(), None, 0.3,
                                     until_s=0.8 * bundle.duration_s)
        self.assertIsNone(why)
        self.assertAlmostEqual(played, 0.8 * bundle.duration_s, places=6)
        self.assertGreater(
            abs(tel_cut.enc[jn[0]] - float(q1[0])), 0.05,
            "fixture is vacuous: the truncated handoff should leave the arm "
            "short of the route end")

        tel_full = _Tel(dict(HOME))          # settles once commanded
        why, played = R.stream_route(bundle, _Pub(tel_full), tel_full, _Det(),
                                     rig(), None, 0.3,
                                     until_s=bundle.duration_s)
        self.assertIsNone(why)
        self.assertAlmostEqual(
            tel_full.enc[jn[0]], float(q1[0]), places=6,
            msg="the full-route handoff returned without settling — the "
                "tracker would inherit the streaming lag")

    def test_retreat_replays_the_measured_trail_in_reverse(self):
        """The retreat walks back along the path the robot ACTUALLY took —
        measured postures reversed, ending at the round's start. It must
        never command a posture the arm did not occupy (hardware 2026-08-01:
        the route-reversal version yanked a wedged arm to the route's 80%
        posture — 'one arm suddenly lifts high' — before winding home)."""
        jn = list(ARM_JOINTS_ALL)
        hist = []
        for off in (0.0, 0.05, 0.10, 0.15, 0.19):   # the measured trail out
            rec = {n: float(HOME[n]) for n in jn}
            rec[jn[0]] = HOME[jn[0]] + off
            hist.append(rec)
        tel = _Tel({n: hist[-1][n] for n in jn})
        ctx = _ctx(_Client(HOME), tel, _Rig({KEY_L: GOALS["L"],
                                             KEY_R: GOALS["R"]}))
        note = R.retreat_after_track_abort(ctx, hist)
        self.assertIsNone(note, f"retreat failed: {note}")
        self.assertTrue(ctx.pub.sent)
        shoulder = [pkt["arm_targets"]["left"]["joint_pos"][0]
                    for pkt in ctx.pub.sent]
        self.assertAlmostEqual(shoulder[0], HOME[jn[0]] + 0.19, places=6,
                               msg="retreat did not start where the arm is")
        self.assertAlmostEqual(shoulder[-1], HOME[jn[0]], places=6,
                               msg="retreat did not end at the trail's start")
        trail_vals = {round(HOME[jn[0]] + off, 9)
                      for off in (0.0, 0.05, 0.10, 0.15, 0.19)}
        for v in shoulder:
            self.assertIn(round(v, 9), trail_vals,
                          "retreat commanded a posture the arm never occupied")
        self.assertAlmostEqual(tel.enc[jn[0]], HOME[jn[0]], places=3,
                               msg="the arm did not settle at the start")

    def test_retreat_is_skipped_when_the_arm_is_off_the_trail(self):
        """Replaying a trail the arm is not actually at would sweep unchecked
        space — a big measured-vs-trail gap must skip the retreat and publish
        nothing."""
        jn = list(ARM_JOINTS_ALL)
        rec = {n: float(HOME[n]) for n in jn}
        tel_enc = dict(rec)
        tel_enc[jn[0]] += 0.6                  # arm is 0.6 rad off the trail
        tel = _Tel(tel_enc)
        ctx = _ctx(_Client(HOME), tel, _Rig({KEY_L: GOALS["L"],
                                             KEY_R: GOALS["R"]}))
        note = R.retreat_after_track_abort(ctx, [rec])
        self.assertIsNotNone(note)
        self.assertIn("skipped", note)
        self.assertFalse(ctx.pub.sent, "a skipped retreat must publish nothing")

    def test_an_overshooting_solver_is_clamped_and_followed_not_rejected(self):
        """Hardware 2026-08-01, round 2: the solver's first commits ran at
        2.08 rad/s against the 0.7 cap and 21 straight REJECTIONS killed the
        round. A sane-but-fast command must be clamped to the lead budget
        and followed — published steps never exceed
        MPC_LEAD_SCALE*max_rate*dt, and the track still converges."""
        client = _Client(HOME, step_rad=0.15)   # 3 rad/s at 20 Hz: fast, sane
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        why, hold, ctx = _run(client, rig)
        self.assertIsNone(why, f"a fast solver killed the track: {why}")
        lim = R.MPC_LEAD_SCALE * _Args.max_rate * 0.06 + 1e-6   # ~20 Hz tick
        prev = None
        for pkt in ctx.pub.sent:
            cur = {}
            for arm, names in (("left", ARM_JOINTS_L), ("right", ARM_JOINTS_R)):
                for n, v in zip(names, pkt["arm_targets"][arm]["joint_pos"]):
                    cur[n] = v
            if prev is not None:
                worst = max(abs(cur[n] - prev[n]) for n in cur)
                self.assertLessEqual(worst, lim * 3,
                                     "published steps exceed the clamp")
            prev = cur


    # -- the tags going dark IS the grasp condition -------------------------

    @staticmethod
    def _blind_after_start(client, rig):
        """Visible at the handoff, dark from the first tick on.

        The handoff rule short-circuits any track whose cubes are ALREADY dark,
        so a rig that starts blind can never reach the IN-LOOP rule. Flipping on
        mpc_start puts the blindness exactly where that rule lives: the session
        opens, then tick 1 finds no median."""
        orig_median, orig_start = rig.median, client.mpc_start
        dark = {"on": False}

        def median(key, now):
            return None if dark["on"] else orig_median(key, now)

        def mpc_start(*a, **kw):
            rep = orig_start(*a, **kw)
            dark["on"] = True
            return rep

        rig.median, client.mpc_start = median, mpc_start

    def _mid_track_blind(self, start_off=0.0, follow=True, max_s=6.0,
                         blind_grasp=True, args=None):
        """Run a track that goes blind once the session is up.

        `start_off` displaces ARM_JOINTS_L[0] from HOME, and the plan anchors
        ARE the HOME end-effector positions — so it is exactly the left hand's
        distance from its anchor. `follow=False` freezes the arm there."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        self._blind_after_start(client, rig)
        tel = _Tel({n: HOME[n] + (start_off if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        if not follow:
            tel.follow = lambda slots: None
        ctx = _ctx(client, tel, rig, args=args)
        keep = (R.MPC_MAX_S, R.MPC_BLIND_GRASP)
        R.MPC_MAX_S, R.MPC_BLIND_GRASP = max_s, blind_grasp
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_BLIND_GRASP = keep
        return why, hold, out.getvalue(), client, ctx

    def test_a_high_hand_inside_the_anchor_bar_does_not_blind_close(self):
        """User 08-10 ("why does it always grasp HIGH?"): every close that
        day carried z +11..+26 mm hiding inside the 3-D anchor bar — the
        hand approaches from above, so the residual's z part is a systematic
        high bias, and the grasp-anchor dial was being turned to compensate
        a closing-logic leak. A hand laterally on its anchor but more than
        MPC_CLOSE_Z_MAX_M HIGH must NOT blind-close; the budget salvage
        (which prints how far out it closed) stays the bounded escape."""
        import contextlib
        import io
        # Find a single-joint perturbation that LIFTS the left EE >= 15 mm
        # while staying inside the 40 mm anchor bar in 3-D — probed on the
        # real deploy-model FK so the test tracks the model, not a guess.
        base = _fk_ee(HOME)["L"]
        found = None
        for jn in ARM_JOINTS_L:
            for step in (0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2):
                enc = dict(HOME)
                enc[jn] += step
                p = _fk_ee(enc)["L"]
                if float(p[2] - base[2]) > 0.015 and \
                        float(np.linalg.norm(p - base)) < 0.035:
                    found = (jn, step)
                    break
            if found:
                break
        self.assertIsNotNone(found, "no single-joint lift found on the model")
        jn, step = found
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        self._blind_after_start(client, rig)
        tel = _Tel({n: HOME[n] + (step if n == jn else 0.0)
                    for n in ARM_JOINTS_ALL})
        tel.follow = lambda slots: None       # frozen arm: the residual stays
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_BLIND_GRASP)
        R.MPC_MAX_S, R.MPC_BLIND_GRASP = 2.0, True
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_BLIND_GRASP = keep
        said = out.getvalue()
        self.assertNotIn("Closing WITHOUT moving", said,
                         "a hand >15 mm HIGH blind-closed anyway — the z "
                         "bar is gone")
        self.assertIn("z bar", said, "the z-bar hold was never announced")

    def test_already_blind_at_the_handoff_never_opens_the_session(self):
        """User 2026-08-06, third time of asking: "It should grasp at [exec]!
        why are you waiting?" A cube nothing has seen since before the settle
        gives the tracker nothing to correct, so building the session at all is
        a 0.6 s GPU cold start with the arms extended followed by an immediate
        admission that there was nothing to solve. Decided BEFORE mpc_start,
        and deliberately with no anchor bar: the arm is still exactly where the
        six-gate route put it, which IS the 07-30 open-loop grasp."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: None, KEY_R: None})
        tel = _Tel({n: HOME[n] + (0.6 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        # Legacy pin: the UNGATED shortcut close survives only under
        # --no-anchor-trust since 08-12 late night — with trust on, a hand
        # this far off the anchor refuses the shortcut and opens the session
        # (pinned separately below).
        ctx = _ctx(client, tel, rig, args=_NoTrustArgs())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            why, hold = R.mpc_track(
                ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        said = out.getvalue()
        self.assertIsNone(why, f"a handoff-blind track aborted: {why}")
        self.assertIsNotNone(hold)
        self.assertIsNone(client.started, "the MPC session was opened anyway")
        self.assertIn("BLIND AT THE HANDOFF", said)
        self.assertIn("plan anchor", said,
                      "the log must still record how good the close was")
        self.assertNotIn("too far", said,
                         "the anchor bar is deliberately absent at the handoff")

    def test_the_handoff_shortcut_refuses_a_hand_off_the_corrected_anchor(self):
        """08-12 late night (user, 'why is it STILL grasping high'): the
        shortcut was the one close path with no z/lateral gate — every blind
        front grasp carried the execution-high bias straight into the jaws.
        A displaced hand now refuses the shortcut and opens the session."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: None, KEY_R: None})
        tel = _Tel({n: HOME[n] + (0.6 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        ctx = _ctx(client, tel, rig)          # default args: anchor trust ON
        keep = R.MPC_MAX_S
        R.MPC_MAX_S = 4.0
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                R.mpc_track(ctx, _Bundle(),
                            {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S = keep
        said = out.getvalue()
        self.assertIn("NOT closing blind", said,
                      "the ungated shortcut close is back")
        self.assertIsNotNone(client.started,
                             "refusing the shortcut must open the session")

    def test_the_handoff_shortcut_requires_a_settle_that_converged(self):
        """Audit 2026-08-06 (D10). The shortcut's justification is that the arm
        is exactly where the six-gate route put it; after a settle TIMEOUT it
        provably is not, and every timed-out round this week handed the tracker
        about 72 mm. Both settle exits returned the same (None, end_s), so the
        caller could not tell — now it can, and a timed-out settle opens the
        session instead of closing blind."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: None, KEY_R: None})
        tel = _Tel(dict(HOME))
        ctx = _ctx(client, tel, rig)              # reach_hover_mm 0: shortcut live
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.mpc_track(ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {},
                        settled=False)
        said = out.getvalue()
        self.assertNotIn("BLIND AT THE HANDOFF", said,
                         "closed blind on a posture the settle never reached")
        self.assertIsNotNone(client.started,
                             "a timed-out settle must open the session")

    def test_the_handoff_shortcut_still_fires_after_a_clean_settle(self):
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: None, KEY_R: None})
        tel = _Tel(dict(HOME))
        ctx = _ctx(client, tel, rig)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.mpc_track(ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {},
                        settled=True)
        # 08-13: an on-anchor handoff closes via the tag-or-no-tag HANDOFF
        # CLOSE (same doctrine, wider z bar); the session still never opens
        self.assertIn("HANDOFF CLOSE", out.getvalue())
        self.assertIsNone(client.started)

    def test_a_blind_tracker_publishes_nothing_before_it_closes(self):
        """User, first hardware run of the rule: "I saw the left arm move a
        little bit all of a sudden, but it should not move because the cube is
        inside of the gripper — when you move, it lost." The first version sat
        after mpc_step, so a blind tracker stepped the solver and published one
        command before closing, nudging a hand already on its cube."""
        why, _hold, said, client, ctx = self._mid_track_blind()
        self.assertIsNone(why)
        # 08-13: a hand already on its anchor never even builds the session —
        # HANDOFF CLOSE fires first, which satisfies this pin's demand
        # (nothing moves, nothing is published) even more strictly
        self.assertIn("HANDOFF CLOSE", said)
        self.assertIsNone(client.started, "the session was built anyway")
        self.assertEqual(len(client.cube_history), 0,
                         "the solver was stepped for a goal nothing measures")
        for pkt in ctx.pub.sent:
            self.assertNotIn("arm_targets", pkt,
                             "a blind tracker moved the arm before closing")

    def test_a_blind_tracker_closes_instead_of_chasing_dead_reckoning(self):
        """User 2026-08-06, after three rounds and zero grasps: 'when you no
        longer see the tags, just skip mpc and grasp it'. A tracker whose every
        cube has gone dark is servoing to IMU dead reckoning — the goal is
        frozen and only the tilt compensation moves it — so there is nothing
        left to correct."""
        why, hold, said, _c, _x = self._mid_track_blind()
        self.assertIsNone(why, f"a blind tracker aborted instead of closing: {why}")
        self.assertIsNotNone(hold, "the grasp posture went missing")
        # 08-13: the on-anchor case now closes at the handoff (HANDOFF
        # CLOSE); the in-session BLIND GRASP path remains pinned by the
        # off-anchor blind tests below
        self.assertIn("HANDOFF CLOSE", said, "the close went unexplained")
        self.assertIn("corrected anchor", said,
                      "the operator needs the distance to the CLEAN plan-time "
                      "measurement to judge whether the close was a good one")

    def test_a_blind_hand_far_from_its_anchor_does_not_close(self):
        """Hardware 2026-08-06, the rule's second run: it closed at left 51 mm
        / right 76 mm from the anchor and BOTH grippers came back EMPTY. The
        goal had drifted 40-55 mm off the anchor on 100%-1-face evidence and
        the arms had faithfully followed it there — "the tags went dark" and
        "the gripper is over the cube" are only the same event when the hand is
        actually AT the cube."""
        why, _hold, said, _c, _x = self._mid_track_blind(
            start_off=0.6, follow=False, max_s=2.0)
        self.assertNotIn("BLIND GRASP", said,
                         "closed on air with the hand nowhere near its cube")
        self.assertIn("too far to call this an occluded grasp", said,
                      "the operator was not told why it kept tracking")
        self.assertIsNotNone(why, "a hand stuck far from its cube never aborted")

    def test_anchor_trust_returns_a_blind_far_hand_home_and_closes(self):
        """08-12 night (user doctrine): blind and far from the anchor no
        longer waits on the frozen goal — the lock walks the goal home and
        the blind-anchor machinery closes there. The whole point: one
        certified-EMPTY pinch beats a retreat."""
        why, _hold, said, _c, _x = self._mid_track_blind(start_off=0.3)
        self.assertIn("ANCHOR TRUST", said,
                      "blind-far did not engage the anchor lock")
        self.assertIn("BLIND GRASP", said,
                      "the locked goal never produced the anchor close")
        self.assertIsNone(why, "anchor trust still ended in an abort")

    def test_a_blind_hand_grasps_once_it_reaches_its_anchor(self):
        """The fall-through has to be a CHANCE, not a veto. With no live
        evidence the frozen goal is the best estimate available, so a blind
        tracker that is still short must be allowed to close the remaining
        distance — and grasp the moment it arrives. (Legacy path: pinned
        under --no-anchor-trust since 08-12.)"""
        _why, _hold, said, _c, _x = self._mid_track_blind(
            start_off=0.3, args=_NoTrustArgs())
        self.assertIn("too far to call this an occluded grasp", said,
                      "it closed immediately instead of tracking in")
        self.assertIn("BLIND GRASP", said,
                      "the hand reached its anchor and was still refused")
        self.assertLess(said.index("too far"), said.index("BLIND GRASP"),
                        "the refusal must come first, then the grasp")

    def test_a_blind_grasp_lands_the_descent_rail_before_it_closes(self):
        """User 2026-08-06 on seeing where the jaws actually bit: "should be the
        center!" While desc_off > 0 the solver is aimed reach_hover_mm ABOVE the
        grasp point, so closing there is cube centre + GRASP_Z_ABOVE_M + hover =
        25 mm high on a cube whose top face is +30 mm — the jaws take the top
        edge, or air. Neither MPC_BLIND_ANCHOR_M nor GRASP_EE_OK_M can catch it:
        both are 3-D distances and a pure z offset hides inside them."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        self._blind_after_start(client, rig)
        tel = _Tel(dict(HOME))                    # standing ON both anchors
        ctx = _ctx(client, tel, rig, args=_HoverArgs())
        keep = R.MPC_MAX_S
        R.MPC_MAX_S = 8.0
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S = keep
        said = out.getvalue()
        self.assertIn("landing the", said,
                      "closed on the hover rail — the jaws bite the top edge")
        self.assertIn("straddle the cube centre", said)
        self.assertIn("BLIND GRASP", said,
                      "it landed the rail but then never closed")
        self.assertLess(said.index("landing the"), said.index("BLIND GRASP"),
                        "the descent must be landed BEFORE the close")
        self.assertIn("descent rail has landed", said,
                      "the close must say the rail was down")
        self.assertIsNone(why)
        self.assertIsNotNone(hold)

    def test_the_handoff_shortcut_defers_when_a_descent_is_owed(self):
        """The handoff shortcut skips the MPC session entirely, so it cannot land
        a descent. With reach_hover_mm > 0 the arm is parked on the hover rail,
        so the shortcut must NOT fire — it falls through, the session opens, and
        the in-loop rule lands the rail first. The shortcut survives only for
        --reach-hover-mm 0, where the route endpoint IS the grasp point."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: None, KEY_R: None})
        tel = _Tel(dict(HOME))
        ctx = _ctx(client, tel, rig, args=_HoverArgs())
        keep = R.MPC_MAX_S
        R.MPC_MAX_S = 8.0
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S = keep
        said = out.getvalue()
        self.assertNotIn("BLIND AT THE HANDOFF", said,
                         "closed 25 mm high by skipping a descent it owed")
        self.assertIsNotNone(client.started,
                             "the session must open so the rail can be landed")
        self.assertIsNone(why)

    def test_the_stall_exits_land_the_rail_before_they_close(self):
        """Audit 2026-08-06: only the convergence exit ever checked the descent
        rail. T12/T13/T14 all closed on the raised hover rail — 25 mm high on a
        60 mm cube — and could not see it, because GRASP_EE_OK_M is a 3-D
        distance and a pure vertical offset hides inside a 60 mm bar. Here the
        arm sits ON its anchor with a 30 mm rail still owed and the tracker
        stalled, so a stall exit is the one that fires."""
        import contextlib
        import io
        # 0.3 rad on shoulder_1 puts the hand ~41 mm out: inside GRASP_EE_OK
        # (60) and outside MPC_CONVERGED_M (20), and 41 mm laterally + a 30 mm
        # hover is 51 mm from the hover point — past MPC_HOVER_ARRIVE_M (35), so
        # descending[] never flips on its own. That is precisely the case
        # rail_down has to force.
        client = _Client(HOME, step_rad=0.0)      # solver commands nothing ->
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})   # the track stalls
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        ctx = _ctx(client, tel, rig, args=_HoverArgs())
        keep = (R.MPC_MAX_S, R.MPC_NO_PROGRESS_S)
        # short enough that the guard fires while the 30 mm ramp
        # (12 ticks at MPC_DESCENT_M_S) is still running
        R.MPC_MAX_S, R.MPC_NO_PROGRESS_S = 12.0, 0.15
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_NO_PROGRESS_S = keep
        said = out.getvalue()
        self.assertIn("descent rail is still owed", said,
                      "a stall exit closed on the raised hover rail")
        self.assertIn("straddle the cube centre", said)
        self.assertIsNone(why, f"landing the rail was convicted as a stall: {why}")
        self.assertIsNotNone(hold)

    def test_landing_the_rail_is_bounded_and_says_how_high_it_closed(self):
        """T12 fires when an arm is STALLED, and a stalled arm may not descend
        either — waiting forever would trade a 15 mm error for a hang. After
        MPC_LAND_MAX_TICKS it closes anyway and prints the height, so the number
        is in the log instead of being silently absorbed by a 3-D bar."""
        import contextlib
        import io
        client = _Client(HOME, step_rad=0.0)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        tel.follow = lambda slots: None           # nothing moves, ever
        ctx = _ctx(client, tel, rig, args=_HoverArgs())
        keep = (R.MPC_MAX_S, R.MPC_NO_PROGRESS_S, R.MPC_LAND_MAX_TICKS)
        # the BUDGET exit re-tests every tick, so a bound of 1 is reached on the
        # second call — the no-progress exit re-arms only once per window
        R.MPC_MAX_S, R.MPC_NO_PROGRESS_S, R.MPC_LAND_MAX_TICKS = 0.2, 99.0, 1
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            (R.MPC_MAX_S, R.MPC_NO_PROGRESS_S,
             R.MPC_LAND_MAX_TICKS) = keep
        said = out.getvalue()
        self.assertIn("could not land the last", said,
                      "it hung waiting for a descent a frozen arm cannot make")
        self.assertIn("mm high", said, "the height it closed at went unreported")

    def test_the_progress_guard_re_baselines_when_the_goal_recedes(self):
        """Audit 2026-08-06 (D4): the old `best_worst` only ever went DOWN and
        was never re-baselined when the GOAL moved, so once the goal receded no
        new all-time best could be set and T13 fired 6 s later however fast the
        hand was closing. Here the goal walks AWAY at 30 mm/s while the hand
        tracks it perfectly — the sliding window must see the last 6 s, not the
        all-time minimum, and must not convict."""
        import contextlib
        import io
        client = _Client(HOME)
        moving = {"k": 0}
        base = np.asarray(GOALS["L"], float)

        def median(key, now):
            if key != KEY_L:
                return np.asarray(GOALS["R"], float).copy()
            moving["k"] += 1
            return base + np.array([0.0, 0.0, 0.0015 * moving["k"]])

        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        rig.median = median
        tel = _Tel(dict(HOME))
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_NO_PROGRESS_S)
        R.MPC_MAX_S, R.MPC_NO_PROGRESS_S = 3.0, 1.0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            R.MPC_MAX_S, R.MPC_NO_PROGRESS_S = keep
        self.assertNotEqual(getattr(why, "code", None), "T13",
                            f"a receding goal was convicted as a stuck hand: {why}")

    def test_the_no_progress_verdict_reports_the_goal_travel(self):
        """The number whose absence made 72 mm -> 178 mm unreconcilable. `worst`
        is |hand - goal| and BOTH ends move; the verdict must say where the goal
        was, not only where the hand ended up."""
        import contextlib
        import io
        client = _Client(HOME, step_rad=0.0)       # solver commands nothing
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        tel = _Tel({n: HOME[n] + (1.0 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        tel.follow = lambda slots: None
        ctx = _ctx(client, tel, rig)
        keep = (R.MPC_MAX_S, R.MPC_NO_PROGRESS_S, R.MPC_EXEC_GUARD)
        R.MPC_MAX_S, R.MPC_NO_PROGRESS_S, R.MPC_EXEC_GUARD = 6.0, 0.5, False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                why, _hold = R.mpc_track(
                    ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        finally:
            (R.MPC_MAX_S, R.MPC_NO_PROGRESS_S,
             R.MPC_EXEC_GUARD) = keep
        self.assertIsNotNone(why)
        self.assertEqual(why.code, "T13", f"expected the progress guard: {why}")
        self.assertIn("its GOAL sat", why,
                      "the verdict still hides the goal end of the subtraction")
        self.assertIn("from the plan anchor", why)

    def test_the_watch_reports_how_far_the_goal_travelled(self):
        """Instrument for the same gap: `cube drift` measures rig.latest()
        against an un-rotated seed, so it is neither the goal nor frame-
        consistent with `worst`. GOAL travel is |goal_pos - plan_anchor|, both
        of which carry the same tilt rotation."""
        import contextlib
        import io
        client = _Client(HOME)
        rig = _Rig({KEY_L: GOALS["L"], KEY_R: GOALS["R"]})
        # off the 08-13 handoff-close bars so the session opens at all
        tel = _Tel({n: HOME[n] + (0.3 if n == ARM_JOINTS_L[0] else 0.0)
                    for n in ARM_JOINTS_ALL})
        ctx = _ctx(client, tel, rig)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.mpc_track(ctx, _Bundle(), {KEY_L: GOALS["L"], KEY_R: GOALS["R"]}, {})
        self.assertIn("GOAL travel from its plan anchor", out.getvalue())

    def test_the_blind_grasp_is_switchable(self):
        """MPC_BLIND_GRASP False restores the pre-2026-08-06 behaviour, where a
        blind track runs on to a stall guard or the budget."""
        _why, _hold, said, _c, _x = self._mid_track_blind(blind_grasp=False)
        self.assertNotIn("BLIND GRASP", said, "the switch did nothing")
        self.assertNotIn("BLIND AT THE HANDOFF", said)



if __name__ == "__main__":
    unittest.main()
