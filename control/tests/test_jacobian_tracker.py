"""The Jacobian drift-servo: does it actually cancel drift, and stay bounded?

WHAT HAS TO BE PROVEN HERE. This tracker is the thing that will be allowed to
deviate from a gated cuRobo route in real time, and the upstream safety argument for it
is explicitly conditional: "collision safety comes from the nominal path already
being collision-free plus SMALL, BOUNDED drift". So the tests split in two:

  * it must WORK — with the object where the plan expected it the command must
    reproduce the planned route, and when the object moves the command must
    follow it (that following IS the feature);
  * it must stay BOUNDED — the two correction caps are load-bearing safety, not
    tuning, so a target yanked far away must NOT produce an unbounded command.

A "the code runs" test would catch neither. The drift-cancellation test in
particular is the one that would have caught an object-frame composition written
backwards, which produces a servo that moves confidently in the wrong direction.

Skipped, not failed, when the deploy model is unavailable.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import mujoco

    import humanoid_auto_operator as OP
    import humanoid_curobo_client as C
    from humanoid_jacobian_tracker import (
        MAX_CORRECTION_FROM_PLANNED_RAD as JT_MAX_FROM_PLANNED,
        JacobianRouteTracker, pose_compose, pose_relative)
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _MODEL = None
    _SKIP = f"deploy model / deps unavailable ({type(exc).__name__}: {exc})"

CUBE = "grasp_cube_40mm"
IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
DT = 0.05                                    # 20 Hz control


def _ee_of(q7_left, arm="left"):
    """EE site position for a left-arm posture, base frame, encoder units."""
    d = mujoco.MjData(_MODEL)
    d.qpos[:] = _MODEL.qpos0
    for v, n in zip(q7_left, C.ARM_JOINTS_L):
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, n)
        d.qpos[_MODEL.jnt_qposadr[jid]] = float(v)
    mujoco.mj_kinematics(_MODEL, d)
    sid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE, "end_effector_L_site")
    return d.site_xpos[sid].copy()


MAX_RATE = 0.3          # what the mission streams at; routes are stretched to it


def _route(n=25, stretch=True):
    """A benign left-arm route: the audited front -> side station lerp, whose
    every configuration clears 51 mm on this model.

    STRETCHED by default because the mission always does: raw, this lerp is
    1.641 rad over 1.2 s = up to 1.37 rad/s, which the per-tick rate gate
    rightly refuses. See test_an_unstretched_route_is_refused_by_the_gate.
    """
    front = np.asarray(OP.STAGE_JOINTS["front"], float)
    side = np.asarray(OP.STAGE_JOINTS["side"], float)
    right = np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"), float)
    q = np.stack([np.concatenate([front + (side - front) * t, right])
                  for t in np.linspace(0, 1, n)])
    bundle = C.RouteBundle(
        joint_names=tuple(C.ARM_JOINTS_ALL), q_enc=q, dt=DT,
        reaches={"end_effector_L_site": {"target": _ee_of(q[-1][:7]),
                                         "cands": _ee_of(q[-1][:7]).reshape(1, 3),
                                         "err": 0.0}},
        controlled_joints=tuple(C.ARM_JOINTS_L), assignment={"L": CUBE},
        goal="reach", solve_s=0.0)
    return bundle.stretch(MAX_RATE) if stretch else bundle


@unittest.skipIf(_MODEL is None, _SKIP)
class PoseAlgebraTests(unittest.TestCase):
    """compose . relative == identity. Everything downstream rests on it, and a
    backwards composition yields a servo that moves confidently the wrong way."""

    def test_compose_inverts_relative(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            frame = (rng.normal(size=3), _unit(rng.normal(size=4)))
            pose = (rng.normal(size=3), _unit(rng.normal(size=4)))
            back = pose_compose(frame, pose_relative(frame, pose))
            np.testing.assert_allclose(back[0], pose[0], atol=1e-12)
            self.assertGreater(abs(float(np.dot(back[1], pose[1]))), 1 - 1e-12)

    def test_a_pure_translation_of_the_frame_translates_the_pose(self):
        frame = (np.zeros(3), IDENTITY_Q.copy())
        rel = pose_relative(frame, (np.array([0.3, 0.1, 0.0]), IDENTITY_Q.copy()))
        moved = pose_compose((np.array([0.05, 0.0, 0.0]), IDENTITY_Q.copy()), rel)
        np.testing.assert_allclose(moved[0], [0.35, 0.1, 0.0], atol=1e-12)


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


@unittest.skipIf(_MODEL is None, _SKIP)
class TrackingTests(unittest.TestCase):

    def setUp(self):
        self.bundle = _route()
        self.tracker = JacobianRouteTracker(model=_MODEL, control_dt=DT)
        self.plan_obj = {CUBE: (np.array([0.35, 0.25, 0.05]), IDENTITY_Q.copy())}
        self.tracker.rebind(self.bundle, self.plan_obj)

    # The arm is simulated as following the previous command exactly. That is the
    # right model for these tests: it isolates the SERVO from tracking lag, and
    # it is how the route is actually consumed (measured -> command -> measured).
    # Sampling the raw waypoint array by tick index instead would be wrong after
    # stretch(), which grows dt while leaving the waypoints alone — the cursor
    # then sits at waypoint k*DT/dt, not waypoint k.
    def _run(self, ticks, live=None, gate=None, tracker=None):
        tracker = tracker or self.tracker
        live = live if live is not None else self.plan_obj
        measured = {n: float(self.bundle.q_enc[0][i])
                    for i, n in enumerate(C.ARM_JOINTS_ALL)}
        cmds, verdicts = [], []
        for _ in range(ticks):
            cmd = tracker.step(live, measured_enc=measured)
            if gate is not None:
                verdicts.append(gate.check(cmd, measured, DT))
            measured = {**measured, **cmd}
            cmds.append(cmd)
        return measured, cmds, verdicts

    def test_an_unmoved_object_reproduces_the_planned_route(self):
        """The null case, and the sharpest one: with the object exactly where the
        plan put it, the command must be the planned route sampled at the cursor.
        Any deviation means the servo is inventing motion the gated route never
        authorised."""
        _, cmds, _ = self._run(20)
        for k, cmd in enumerate(cmds):
            # The command at tick k is for cursor time k*DT — the cursor advances
            # AFTER the solve, so comparing against (k+1)*DT is off by one tick.
            planned = self.bundle.q_at(k * DT)
            col = {n: i for i, n in enumerate(self.bundle.joint_names)}
            worst = max(abs(cmd[j] - float(planned[col[j]])) for j in cmd)
            self.assertLess(worst, 5e-3,
                            f"tick {k}: {worst:.4f} rad off the plan on an "
                            f"unmoved object")

    def test_a_moved_object_moves_the_hand_by_the_same_amount(self):
        """THE feature. Shift the cube 4 cm in +y and the commanded EE must land
        4 cm further along +y than the plan would have put it — this is both 'the
        cube was pushed' and 'the torso leaned', which are the same correction
        because both move the object's pose in the base frame."""
        shift = np.array([0.0, 0.04, 0.0])
        live = {CUBE: (self.plan_obj[CUBE][0] + shift, IDENTITY_Q.copy())}
        n = 20
        _, moved, _ = self._run(n, live=live)
        _, plain, _ = self._run(n, tracker=self._fresh())
        got = _ee_of([moved[-1][j] for j in C.ARM_JOINTS_L])
        base = _ee_of([plain[-1][j] for j in C.ARM_JOINTS_L])
        np.testing.assert_allclose(got - base, shift, atol=3e-3)

    def _fresh(self):
        t = JacobianRouteTracker(model=_MODEL, control_dt=DT)
        t.rebind(self.bundle, self.plan_obj)
        return t

    def test_a_sustained_pull_cannot_walk_the_arm_off_the_planned_route(self):
        """The safety half, and it has to be SUSTAINED to test the right thing.
        A single tick is bounded by the output slew limit no matter what, so a
        one-tick check silently tests the limiter instead of the caps. Held for
        40 ticks, an uncapped servo would accumulate 40 x 0.025 = 1.0 rad away
        from the collision-checked waypoint; the caps are what keep 'no per-tick
        collision term' an argument rather than a hope."""
        live = {CUBE: (self.plan_obj[CUBE][0] + np.array([1.0, -1.0, 0.5]),
                       IDENTITY_Q.copy())}
        n = 40
        measured, cmds, _ = self._run(n, live=live)
        col = {n_: i for i, n_ in enumerate(self.bundle.joint_names)}
        # Assert against the MODULE constant, not the instance attribute: an
        # instance whose cap was widened would otherwise satisfy its own bound
        # trivially, and the test would pass while proving nothing.
        for k, cmd in enumerate(cmds):
            q_nom = self.bundle.q_at(k * DT)
            for j, v in cmd.items():
                self.assertLessEqual(
                    abs(v - float(q_nom[col[j]])),
                    JT_MAX_FROM_PLANNED + 1e-9,
                    f"tick {k}: {j} walked past the cap against the PLANNED "
                    f"waypoint under a sustained pull")

    def test_only_the_controlled_joints_are_commanded(self):
        _, cmds, _ = self._run(1)
        self.assertEqual(set(cmds[0]), set(C.ARM_JOINTS_L))

    def test_the_cursor_advances_and_finishes(self):
        self.assertFalse(self.tracker.done)
        self._run(int(self.bundle.duration_s / DT) + 2)
        self.assertTrue(self.tracker.done)

    def test_every_emitted_command_passes_the_per_tick_gate(self):
        """The servo's output is not trusted on its own — StepGate is what stands
        between it and the motors. On a stretched, benign route with a small
        object shift every tick must pass, or the pairing is unusable."""
        gate = C.StepGate(model=_MODEL)
        live = {CUBE: (self.plan_obj[CUBE][0] + np.array([0.0, 0.02, 0.0]),
                       IDENTITY_Q.copy())}
        _, _, verdicts = self._run(int(self.bundle.duration_s / DT),
                                   live=live, gate=gate)
        bad = [(k, v.why) for k, v in enumerate(verdicts) if not v.ok]
        self.assertFalse(bad, f"gate refused {len(bad)} tick(s): {bad[:3]}")

    def test_an_unstretched_route_is_refused_at_bind_time(self):
        """Caught at bind, not at runtime, because the runtime symptom is silent:
        with the output slew limit in place an over-fast route does not error, it
        LAGS — increasingly, with the correction caps then dragging against a
        waypoint the arm never reached. Missions stretch before streaming; this
        makes forgetting loud."""
        raw = _route(stretch=False)
        t = JacobianRouteTracker(model=_MODEL, control_dt=DT)
        with self.assertRaises(AssertionError) as ctx:
            t.rebind(raw, self.plan_obj)
        self.assertIn("stretch", str(ctx.exception))

    def test_a_jump_in_the_object_is_ramped_in_not_lunged(self):
        """The object moving 2 cm between planning and the first tick used to
        produce 1.58 rad/s on wrist_1 — refused by the gate, which republishes
        the last command, so a lunging servo is a STALLED servo. The correction
        must arrive over several ticks instead."""
        live = {CUBE: (self.plan_obj[CUBE][0] + np.array([0.0, 0.02, 0.0]),
                       IDENTITY_Q.copy())}
        measured = {n: float(self.bundle.q_enc[0][i])
                    for i, n in enumerate(C.ARM_JOINTS_ALL)}
        cmd = self.tracker.step(live, measured_enc=measured)
        worst = max(abs(cmd[j] - measured[j]) for j in cmd) / DT
        self.assertLessEqual(worst, self.tracker.max_output_step / DT + 1e-9)
        self.assertLess(worst, C.STEP_MAX_RATE_RAD_S,
                        "the servo must stay strictly inside the gate's own cap")

    def test_a_frozen_hold_is_a_fixed_base_frame_pose(self):
        """After the jaws close, the object's detection is occluded and — once
        the cube is lifted — meaningless, so the held target must stop following
        it. `hold_step` takes no object at all, and holding must be a fixed
        point: repeated calls on a settled arm must not drift."""
        # Run the route to completion first: hold_step's nominal posture is
        # q_route[-1] (upstream's choice), so freezing mid-route would leave the
        # null-space term pulling toward an end the arm never reached — an
        # inconsistency the real flow never has, since the hold begins where the
        # reach finished.
        measured, _, _ = self._run(int(self.bundle.duration_s / DT) + 2)
        self.tracker.freeze_hold()
        frozen = {f: (p[0].copy(), p[1].copy())
                  for f, p in self.tracker._hold_pose.items()}
        cmds = []
        for _ in range(6):
            cmd = self.tracker.hold_step(measured_enc=measured)
            measured = {**measured, **cmd}
            cmds.append(cmd)
        for j in cmds[-1]:                       # settled: no residual drift
            self.assertAlmostEqual(cmds[-1][j], cmds[-2][j], places=6)
        for f, p in self.tracker._hold_pose.items():
            np.testing.assert_allclose(p[0], frozen[f][0], atol=0.0)


@unittest.skipIf(_MODEL is None, _SKIP)
class BindingTests(unittest.TestCase):
    """Refuse loudly rather than track something undefined."""

    def test_a_missing_plan_time_object_pose_is_refused(self):
        t = JacobianRouteTracker(model=_MODEL, control_dt=DT)
        with self.assertRaises(AssertionError):
            t.rebind(_route(), {})

    def test_stepping_before_binding_is_refused(self):
        t = JacobianRouteTracker(model=_MODEL, control_dt=DT)
        with self.assertRaises(AssertionError):
            t.step({CUBE: (np.zeros(3), IDENTITY_Q.copy())})


if __name__ == "__main__":
    unittest.main()
