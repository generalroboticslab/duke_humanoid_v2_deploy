"""The journey merged onto cuRobo — and the six deadlock classes it must not reopen.

WHAT THIS GUARDS. 2d1f9df closed six livelock/deadlock classes in the operator's
journey after two adversarial review passes. Every one of them was the same
mistake wearing different clothes: A STATE WITH NO EXIT. Porting the journey onto
a tool whose every failure path was `refuse()` is exactly how they come back, so
this file pins the property that replaces them.

  1. Empty-pinch retry (CONTRACT: unlimited while the cube stays servable)
  2. Pre-grab EE-convergence timeout (a hold pinned outside tolerance releases)
  3. Midline band (the steering attractor; natural-sign arm; stall memory)
  4. Latch-fail budget (3 failures skip the cube LOUDLY)
  5. Journey leg 2 (retired eye PARKED at origin permanently since 08-09 —
     the 07-26 rejoin rule is superseded, leg-2 search is one-eyed by user
     order; posture glide merges; backward vx floor; nav aims into the
     serving arm's half-space)
  6. ASSIGN stall escape (zero-reachability releases the arm)

THE STRUCTURAL ANSWER, which is stronger than matching them one for one: the
mission is a FOR loop over a list latched once by the survey, and a leg returns
DONE / SKIP / FATAL. Nothing is re-queued, so no cube can be attempted twice and
the loop cannot fail to terminate. The operator re-queues, and that is precisely
where its livelocks lived.

The steering law is the one piece that could not be imported (it lives inside
navigation_command with the task scheduler this tool replaces), so it gets a
DIFFERENTIAL test against the operator's own function rather than a restatement
of what I believe it does.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R
    from auto_operator import independent as IND

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# The steering law, against the operator's own navigation_command
# ---------------------------------------------------------------------------

class _Task:
    """The fields navigation_command's guards touch: phase (manipulation-active)
    and track (tracks-active). ASSIGNED with no track is the state that reaches
    the steering law."""

    def __init__(self, key, arm):
        self.key, self.arm = key, arm
        self.phase = IND.ArmTaskPhase.ASSIGNED if R else None
        self.track = None


class _Cube:
    def __init__(self, pos):
        self.pos = np.asarray(pos, float)
        self.reached = False
        self.held_by = None
        self.camera_port = 5555
        self.last_camera_port = 5555

    def fresh(self, now, horizon):
        return True

    def confirmed(self, now):
        return True


class _Op:
    """Duck-typed operator: exactly the fields navigation_command touches on the
    path that reaches the steering law, and nothing else."""

    def __init__(self, key, pos, arm):
        self.journey = True
        self.cubes = {key: _Cube(pos)}
        self._survey_order = [key]
        self._arm_tasks = {arm: _Task(key, arm), _peer(arm): None}
        self._held = {}
        self._nav_task_key = None
        self._nav_leg_dir = None
        self._journey_walk_pose = False
        self._journey_serve_t = 0.0
        self._journey_still_since = None
        self._journey_fail_warned = False
        self._jpos = np.zeros(31)
        self._ee_act = {"left": None, "right": None}
        self.active = None
        # journey_arms_tucked passes when the measured arms ARE the tuck
        for i, a in enumerate(("left", "right")):
            self._jpos[13 + 7 * i:20 + 7 * i] = OP.mirror_arm(
                OP.POWERON_JOINTS, a)


# independent.py resolves its constants with _K(op) = sys.modules[
# type(op).__module__], so a duck-typed op must CLAIM the operator's module or
# every JOURNEY_* lookup lands in this test file. (humanoid_curobo_reach solves
# the same problem the other way, by re-exporting the constants into itself.)
_Op.__module__ = "humanoid_auto_operator"
# The operator's OWN wrap, bound in rather than reimplemented: a differential
# test that supplies its own version of a term is comparing my code to my code.
_Op._wrap = staticmethod(OP.AutoOperator._wrap)


def _peer(arm):
    return "right" if arm == "left" else "left"


@unittest.skipIf(R is None, _SKIP)
class SteeringLawTests(unittest.TestCase):
    """THE differential test. journey_nav restates the operator's law because it
    could not be imported; getting it subtly wrong is invisible until the robot
    walks, so every (vx, wz) is compared against navigation_command itself."""

    def test_it_matches_the_operator_over_a_grid_of_cube_positions(self):
        checked = 0
        for x in (0.4, 0.8, 1.2, 2.0):
            for y in (-0.6, -0.2, -0.05, 0.05, 0.2, 0.6):
                for arm in ("left", "right"):
                    with self.subTest(x=x, y=y, arm=arm):
                        op = _Op("cube", (x, y, 0.05), arm)
                        want = IND.navigation_command(op, 1.0)
                        got = R.journey_nav(np.array([x, y, 0.05]), arm,
                                            op._nav_leg_dir)
                        np.testing.assert_allclose(got, want, atol=1e-12)
                        checked += 1
        self.assertEqual(checked, 48)

    def test_a_backward_leg_matches_too(self):
        """The -0.15 floor is its own branch: the gait policy steps in place at
        a commanded -0.1 m/s, so a backward approach stalls short of the arrival
        gate forever at the forward floor (07-26 audit)."""
        for x in (-0.5, -1.0, -2.0):
            with self.subTest(x=x):
                op = _Op("cube", (x, 0.1, 0.05), "left")
                want = IND.navigation_command(op, 1.0)
                self.assertEqual(op._nav_leg_dir, -1.0)
                got = R.journey_nav(np.array([x, 0.1, 0.05]), "left", -1.0)
                np.testing.assert_allclose(got, want, atol=1e-12)

    def test_every_constant_is_read_from_the_operator(self):
        """No retyped numbers: if the operator retunes its walk, this follows.
        The aim law lives in journey_align_error since the 08-08 decouple
        redesign (journey_nav calls it), so the constants are the UNION of
        the two functions' reads."""
        src = (R.journey_nav.__code__.co_names
               + R.journey_align_error.__code__.co_names)
        for name in ("JOURNEY_AIM_OFFSET_M", "JOURNEY_AIM_OFFSET_BACK_M",
                     "JOURNEY_REVERSE_DRIFT_COMP_M",
                     "K_STEER", "WZ_MAX",
                     "JOURNEY_NEAR_VX_BACK", "JOURNEY_NEAR_VX",
                     "JOURNEY_SLOW_K", "JOURNEY_STOP_M", "CRUISE_VX"):
            self.assertIn(name, src, f"{name} is not read from OP")

    @unittest.skipIf(OP.JOURNEY_AIM_OFFSET_M == 0.0,
                     "08-09 plain-pursuit experiment (user): the offset is 0, "
                     "so its purpose is deliberately suspended — this guard "
                     "re-arms the moment a nonzero offset returns")
    def test_the_aim_offset_pushes_the_cube_off_the_midline(self):
        """Class 3. Plain pursuit servoes the cube onto bearing 0 — the centre
        of the band where ownership checks disqualify it. The offset means the
        equilibrium parks it inside the serving arm's half-space."""
        for arm, sign in (("left", +1.0), ("right", -1.0)):
            with self.subTest(arm=arm):
                on_axis = np.array([1.0, 0.0, 0.05])
                _, wz = R.journey_nav(on_axis, arm, 1.0)
                # steering turns AWAY from the arm's side, which walks the cube
                # into that side of the body
                self.assertLess(wz * sign, 0.0)

    def test_the_leg_direction_is_latched_not_recomputed(self):
        """A cube crossing the +/-90 deg bearing line mid-leg would otherwise
        flip the robot's direction of travel."""
        self.assertEqual(R.journey_leg_dir(np.array([1.0, 0.1, 0.0])), 1.0)
        self.assertEqual(R.journey_leg_dir(np.array([-1.0, 0.1, 0.0])), -1.0)

    def test_the_serving_arm_is_the_one_on_the_cubes_side(self):
        """Measured 07-30: the near arm cleared every gate from y -0.30..+0.30
        and crossing past ~0.10 m was INFEASIBLE. Not a preference."""
        self.assertEqual(R.journey_serving_arm(np.array([0.4, 0.2, 0.0])), "left")
        self.assertEqual(R.journey_serving_arm(np.array([0.4, -0.2, 0.0])), "right")


# ---------------------------------------------------------------------------
# The anti-deadlock structure
# ---------------------------------------------------------------------------

@unittest.skipIf(R is None, _SKIP)
class NoDeadlockTests(unittest.TestCase):

    def test_every_wait_in_a_leg_has_a_finite_bound(self):
        """Class 5 and 6 in one property. The operator's journey deadlocked at
        journey_arms_tucked and at a zero-reachability assignment; both were
        states with no exit. Each wait here owns a named timeout, and a missing
        one is a state with no exit again."""
        for name in ("JOURNEY_TUCK_TIMEOUT_S", "JOURNEY_WALK_TIMEOUT_S",
                     "JOURNEY_STILL_TIMEOUT_S", "JOURNEY_ACQUIRE_S"):
            self.assertGreater(getattr(R, name), 0.0)
            self.assertLess(getattr(R, name), 600.0, f"{name} is not a bound")
        self.assertIn(name, R.run_leg.__code__.co_names)

    def test_the_acquire_wait_is_shorter_than_the_standing_scan(self):
        """DET_WAIT_S is 120 s and correct for a one-shot tool that has nothing
        else to do. Standing that long PER LEG is how a mission silently parks
        (class 6), so a leg gets its own, shorter budget."""
        self.assertLess(R.JOURNEY_ACQUIRE_S, R.DET_WAIT_S)

    def test_plan_rounds_are_bounded(self):
        self.assertGreaterEqual(R.JOURNEY_LEG_PLAN_ROUNDS, 2)
        self.assertLessEqual(R.JOURNEY_LEG_PLAN_ROUNDS, 10)

    def test_the_empty_regrasp_window_is_actually_READ_by_the_code(self):
        """THE test that was missing, and its absence is why class 1 regressed.

        The first version of this file asserted only that
        JOURNEY_EMPTY_REGRASP_S was greater than zero — which a constant no line
        of code reads passes perfectly. An adversarial audit found exactly that:
        the constant's own comment promised "an empty pinch reopens and
        re-grabs without limit", run_leg's docstring promised the same, and
        neither existed. A certified-empty pinch went straight to FATAL and the
        mission parked forever in refuse_holding with an EMPTY hand while the
        cube sat on the table still being seen.

        A constant that nothing reads is a comment. This asserts it is wiring.
        """
        self.assertGreater(R.JOURNEY_EMPTY_REGRASP_S, 0.0)
        self.assertIn("JOURNEY_EMPTY_REGRASP_S", R.run_leg.__code__.co_names,
                      "the empty-regrasp window is a dead constant again")
        self.assertIn("empty_retry_s",
                      R.execute_and_retract.__code__.co_varnames)

    def test_the_leg_warm_solve_overlaps_the_walk_and_joins_before_rpc(self):
        """User 08-10 warm start: a throwaway same-shape solve launches
        BEFORE the walk loop (paying cuRobo's CUDA-graph capture and the
        z-floor session rebuild during travel, logged 2.5-3 s at the stop
        point before this), and the plan rounds JOIN the thread before their
        first RPC — the lockstep REQ socket must never have two users."""
        import inspect
        src = inspect.getsource(R.run_leg)
        launch = src.index("ctx.warm_thread = threading.Thread")
        walk = src.index("tucked in")
        join = src.index("hand back the")
        rounds = src.index("JOURNEY_LEG_PLAN_ROUNDS + 1")
        self.assertLess(launch, walk, "the warm-up no longer overlaps the walk")
        self.assertLess(walk, join)
        self.assertLess(join, rounds,
                        "plan rounds may collide with the warm-up on the "
                        "REQ socket")
        # the throwaway must mirror the real solve's session keys — a warm-up
        # with a different goal/floor/assignment warms the WRONG kernels
        body = src[launch - 3000:launch]
        self.assertIn('goal="reach"', body)
        self.assertIn("hand_z_floor=floor0", body)
        self.assertIn("assignment={side: target_key}", body)

    def test_a_leg_can_only_end_in_three_ways(self):
        self.assertEqual({R.DONE, R.SKIP, R.FATAL}, {"done", "skip", "fatal"})

    def test_the_mission_never_re_queues_a_cube(self):
        """THE structural argument, and it is stronger than matching the six
        classes one for one. The operator re-queues skipped cubes — assign,
        stall, cancel, reassign, forever — and that is where its livelocks
        lived. Here the survey latches the order ONCE and the mission is a FOR
        over that list, so every cube is attempted exactly once and the loop
        terminates whatever the world does."""
        src = R.journey_mission.__code__
        self.assertIn("run_leg", src.co_names)
        # a `while` over a mutable pending set is what this must never become
        self.assertIn("order", src.co_varnames)

    def test_a_skipped_leg_does_not_end_the_mission(self):
        """Class 4's lesson: skip LOUDLY and go on. A journey that stops on the
        first unservable cube is a journey that never carries the second."""
        import inspect
        src = inspect.getsource(R.journey_mission)
        self.assertIn("continue", src)
        self.assertIn("SKIPPED", src)

    def test_only_a_held_cube_makes_a_failure_fatal(self):
        """FATAL holds the arms rather than crawling them home, so it must be
        reserved for states a next leg could not survive."""
        import inspect
        src = inspect.getsource(R.journey_mission)
        self.assertIn("refuse_holding", src)


@unittest.skipIf(R is None, _SKIP)
class GateImportTests(unittest.TestCase):
    """The three gates are the operator's own objects, not copies."""

    def test_the_gates_are_the_operators_own_functions(self):
        self.assertIs(R.journey_survey, IND.journey_survey)
        self.assertIs(R.journey_base_still, IND.journey_base_still)
        self.assertIs(R.journey_arms_tucked, IND.journey_arms_tucked)

    def test_the_rig_carries_every_field_those_gates_read(self):
        """Duck-typing that is missing a field fails at the worst moment — mid
        mission, on hardware — so it is checked here instead."""
        rig = R.GazeRig(("grasp_cube_40mm",))
        for f in ("_survey_order", "_survey_warn_t", "_base_vel", "_base_ang",
                  "_journey_lin_seen", "_journey_blind_warned",
                  "_journey_still_since", "_jpos", "_ee_act", "_held",
                  "_cam_retired", "cubes"):
            self.assertTrue(hasattr(rig, f), f"GazeRig lacks {f}")

    def test_the_stillness_gate_runs_against_the_rig(self):
        """v159b publishes base_lin_vel as hard zeros, so the gate must take its
        gyro + fixed-settle fallback rather than passing vacuously."""
        rig = R.GazeRig(("grasp_cube_40mm",))
        rig._base_vel = [0.0, 0.0, 0.0]
        rig._base_ang = [0.0, 0.0, 0.0]
        self.assertFalse(R.journey_base_still(rig, 0.0))       # timer starts
        self.assertTrue(R.journey_base_still(rig, OP.JOURNEY_STILL_S_BLIND + 1))
        self.assertTrue(rig._journey_blind_warned)

    def test_motion_resets_the_stillness_timer(self):
        """Leg 1's stale timer must never satisfy leg 2's arrival check."""
        rig = R.GazeRig(("grasp_cube_40mm",))
        rig._base_vel, rig._base_ang = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        R.journey_base_still(rig, 0.0)
        self.assertIsNotNone(rig._journey_still_since)
        rig._journey_still_since = None                 # what run_leg does while driving
        self.assertFalse(R.journey_base_still(rig, 1e6))


@unittest.skipIf(R is None, _SKIP)
class TuckPacketTests(unittest.TestCase):

    def _rig(self, walk_pose=True, left_at_tuck=False):
        rig = R.GazeRig(("grasp_cube_40mm",))
        rig._journey_walk_pose = walk_pose
        jp = np.zeros(31)
        for i, a in enumerate(("left", "right")):
            jp[13 + 7 * i:20 + 7 * i] = OP.mirror_arm(OP.POWERON_JOINTS, a)
        if not left_at_tuck:
            jp[13] += 0.8
        rig._jpos = jp
        return rig

    def test_an_arm_away_from_the_tuck_is_driven(self):
        pkt = R.journey_tuck_packet(self._rig())
        self.assertIn("left", pkt["arm_targets"])

    def test_an_arm_already_tucked_is_STILL_commanded(self):
        """REVERSED 2026-07-31, and the old rationale was a mis-port.

        This used to assert silence, reasoning that "re-sending a target the
        arm already holds is how the operator's arms oscillated tuck<->home
        forever". That is true of the OPERATOR, whose identical tolerance gate
        (independent.py:771) sits among several arm commanders — SIDE_HOME vs
        POWERON goals plus the Cartesian pin/home machinery — so an arm dropped
        from one payload is still being commanded by another, and re-sending
        would fight them.

        Here journey_tuck_packet is the ONLY arm commander for the whole walk.
        Dropping the last converged arm made the packet None, and None means
        real_env sees no arm content: the 0.5 s silence failsafe then took both
        arms the instant they arrived at the tuck and crawled them toward the
        default pose at 0.025 rad/s for the rest of the leg. Same code, opposite
        consequence. Re-commanding a FIXED posture cannot oscillate — there is
        no second target to alternate with — and it keeps the timer alive."""
        pkt = R.journey_tuck_packet(self._rig(left_at_tuck=True))
        self.assertIsNotNone(pkt, "went silent once the arms reached the tuck")
        self.assertEqual(set(pkt["arm_targets"]), {"left", "right"})

    def test_a_held_arm_is_exempt(self):
        """It rides with its cube; gliding it to the tuck would carry the cube
        through the tuck trajectory. Asserted on the HELD arm's absence, not on
        the packet being None — the free arm is commanded either way, and the
        old assertIsNone only passed because the tolerance gate happened to
        drop the free arm too."""
        rig = self._rig()
        rig._held["left"] = {"key": "grasp_cube_40mm"}
        pkt = R.journey_tuck_packet(rig)
        self.assertNotIn("left", pkt["arm_targets"])
        self.assertIn("right", pkt["arm_targets"])

    def test_nothing_is_commanded_outside_the_walk_pose(self):
        self.assertIsNone(R.journey_tuck_packet(self._rig(walk_pose=False)))

    def test_the_target_is_the_joint_tuck_not_a_cartesian_point(self):
        """07-21: the Cartesian home_glide left the arm up to 62 deg off the
        true tuck posture, so this streams JOINTS."""
        pkt = R.journey_tuck_packet(self._rig())
        self.assertIn("joint_pos", pkt["arm_targets"]["left"])
        np.testing.assert_allclose(
            pkt["arm_targets"]["left"]["joint_pos"],
            OP.mirror_arm(OP.POWERON_JOINTS, "left"), atol=1e-9)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Class 1: the empty-pinch contract, found MISSING by the 07-30 audit
# ---------------------------------------------------------------------------

class _GraspTel:
    def __init__(self):
        self.ee = {}
        self.fault = False

    def fresh(self):
        return {"joint_pos": [0.0] * 31, "projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, self.fault], "ee": self.ee}


class _GraspDet:
    def poll(self, rig):
        pass


class _GraspRig:
    """Reports the cube as seen or not, on demand."""

    def __init__(self, seen=True):
        self.seen = seen

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        out["gaze_targets"] = [0.0] * 4
        return out

    def median(self, key, now):
        return np.array([0.4, 0.2, 0.05]) if self.seen else None

    def latest(self, key, now):
        return self.median(key, now)

    def last_inliers(self, key):
        return 3

    def faces_used(self, key, now=None):
        # faces behind the MEDIAN, not the newest packet
        return self.last_inliers(key)


class _GraspPub:
    """Answers every hand_grab with EMPTY, and records the reopens."""

    def __init__(self, tel):
        self.tel = tel
        self.grabs = 0
        self.opens = 0

    def publish(self, packet):
        act = (packet or {}).get("ee_action")
        if not act:
            return
        if act["command"] == "hand_grab":
            self.grabs += 1
            self.tel.ee = dict(self.tel.ee)
            self.tel.ee[act["side"]] = {"grasp_action_id": act["action_id"],
                                        "grasp_detected": False}
        elif act["command"] == "hand_open":
            self.opens += 1


class _GraspBundle:
    goal = "reach"
    active_sides = ("left",)
    assignment = {"L": "grasp_cube_40mm"}

    def __init__(self):
        self.joint_names = tuple(R.ARM_JOINTS_L) + tuple(R.ARM_JOINTS_R)
        self.q_enc = np.zeros((2, len(self.joint_names)))
        self.horizon = 2
        self.duration_s = 0.0
        self.dt = 0.0
        self.controlled_joints = tuple(R.ARM_JOINTS_L)

    def q_at(self, t):
        return self.q_enc[0]


@unittest.skipIf(R is None, _SKIP)
class EmptyPinchContractTests(unittest.TestCase):
    """Class 1 is a CONTRACT: unlimited re-grab while the cube stays servable.

    Every one of these fails against the code the audit reviewed, where a
    certified-empty pinch was mapped to FATAL and ended the mission in an
    unbounded hold with an empty hand.
    """

    def _run(self, seen=True, window=0.6):
        tel, rig = _GraspTel(), _GraspRig(seen=seen)
        pub = _GraspPub(tel)
        ctx = R.Ctx(_ReachArgs(), None, tel, _GraspDet(), None, pub, None, rig)
        plan = R.Plan(_GraspBundle(), {"grasp_cube_40mm": np.zeros(3)}, None, {})
        keep, R.GRASP_RESULT_TIMEOUT_S = R.GRASP_RESULT_TIMEOUT_S, 5.0
        try:
            # Servability is judged INTERNALLY now, per empty hand, against the
            # cube that hand was assigned — there is no callback to inject.
            out = R.execute_and_retract(ctx, plan, ("grasp_cube_40mm",), {},
                                        empty_retry_s=window)
        finally:
            R.GRASP_RESULT_TIMEOUT_S = keep
        return out, pub

    def test_an_empty_pinch_is_retried_not_fatal(self):
        """THE regression. Nothing is in the hand, so this is the one grasp
        failure a journey can recover from — and the audit found it classified
        as the one that cannot."""
        (outcome, why, grasped, _), pub = self._run()
        self.assertEqual(outcome, R.SKIP, f"empty pinch was {outcome}: {why}")
        self.assertEqual(grasped, ())
        self.assertGreater(pub.grabs, 1, "the empty pinch was never retried")

    def test_the_fingers_are_REOPENED_between_attempts(self):
        """Re-issuing hand_grab on a closed gripper asks the service to close an
        already-closed hand: it stalls at zero travel and certifies empty again
        forever. The retry is only a retry if the fingers go back out."""
        _, pub = self._run()
        self.assertGreater(pub.opens, 0, "hand_open was never sent")

    def test_it_stops_when_the_cube_stops_being_seen(self):
        """'Unlimited while SERVABLE' — an unseen cube is not servable, and a
        retry loop with no such exit is the livelock in a new place."""
        (outcome, why, _, _), pub = self._run(seen=False)
        self.assertEqual(outcome, R.SKIP)
        self.assertIn("no longer seen", why)

    def test_the_window_bounds_the_re_grips_so_the_leg_can_RE_PLAN(self):
        """Class 2's shape. A hand that keeps closing on nothing is in the wrong
        PLACE; repeating the same route forever is the livelock, so the round
        ends and run_leg plans again."""
        (outcome, why, _, _), _ = self._run(window=0.0)
        self.assertEqual(outcome, R.SKIP)
        self.assertIn("re-planning", why)


@unittest.skipIf(R is None, _SKIP)
class HoldingAnEmptyHandTests(unittest.TestCase):

    def test_refuse_holding_releases_when_nothing_is_held(self):
        """Defence in depth. refuse_holding's loop is deliberately unbounded —
        when to let go of a held object is an operator's decision, not a
        timeout's — which makes it the only place that can hold a healthy robot
        forever. Reaching it with an EMPTY hand is never right, and the
        classification upstream should not be the only thing that has to be
        correct for that to hold."""
        class _P:
            sent = []

            def publish(self, packet):
                _P.sent.append(packet)

        code = R.refuse_holding("x", 5, _P(), _GraspTel(), _GraspDet(),
                                _GraspRig(), 0.3, holding=False)
        self.assertEqual(code, 5)
        self.assertEqual(_P.sent, [], "it held an empty hand")


class _ScriptedPub:
    """Answers each side's Nth LOGICAL hand_grab from a per-side script of
    verdicts (True=contact, False=empty). Logical, not per-publish: the wire is
    latest-value-wins and do_grasp re-sends every command up to
    EE_ACTION_REPEAT_TICKS times under ONE action id, exactly like the real
    service sees it — so a new grab is a NEW id, and repeats of the same id
    must neither consume the script nor count as retries. (The first version
    counted publishes and the numbers meant nothing.)"""

    def __init__(self, tel, script):
        self.tel = tel
        self.script = {k: list(v) for k, v in script.items()}
        self.grabs: dict = {}
        self.opens: dict = {}
        self.order: dict = {}      # per side, the finger commands in sequence
        self._seen: dict = {}

    def publish(self, packet):
        act = (packet or {}).get("ee_action")
        if not act:
            return
        side, kind, aid = act["side"], act["command"], act["action_id"]
        if self._seen.get((side, kind)) == aid:
            return                              # repeat of the same burst
        self._seen[(side, kind)] = aid
        self.order.setdefault(side, []).append(kind)
        if kind == "hand_open":
            self.opens[side] = self.opens.get(side, 0) + 1
            return
        if kind != "hand_grab":
            return
        self.grabs[side] = self.grabs.get(side, 0) + 1
        if self.script.get(side):
            verdict = self.script[side].pop(0)
            self.tel.ee = dict(self.tel.ee)
            self.tel.ee[side] = {"grasp_action_id": aid,
                                 "grasp_detected": verdict}


class _DualBundle:
    goal = "reach"
    active_sides = ("left", "right")
    assignment = {"L": "grasp_cube_40mm", "R": "grasp_cube_60mm"}

    def __init__(self):
        self.joint_names = tuple(R.ARM_JOINTS_L) + tuple(R.ARM_JOINTS_R)
        self.q_enc = np.zeros((2, len(self.joint_names)))
        self.horizon = 2
        self.duration_s = 0.0
        self.dt = 0.0
        self.controlled_joints = self.joint_names

    def q_at(self, t):
        return self.q_enc[0]


class _DualRig(_GraspRig):
    def median(self, key, now):
        return np.array([0.4, 0.2, 0.05]) if self.seen else None


@unittest.skipIf(R is None, _SKIP)
class MixedGraspRetryTests(unittest.TestCase):
    """One hand loaded, the other certified empty — the dual-arm case the user
    asked about by name: does the empty hand retry, and does the retry disturb
    the loaded one? The contract: retry goes ONLY to the empty hand; a hand
    that confirmed contact never receives another finger command; when the
    window closes with a cube in one hand, the mission HOLDS (walking away
    from a loaded hand is not an option), and `held` names that hand so the
    hold-vs-release decision upstream cannot get it wrong."""

    TARGETS = {"grasp_cube_40mm": np.zeros(3), "grasp_cube_60mm": np.zeros(3)}

    class _EndOfHarness:
        """The retract needs a live plan server; this harness ends there. A
        PlanServerError is the one exception execute_and_retract already maps
        to FATAL-with-held, which is exactly the boundary the tests read."""

        def plan(self, *a, **kw):
            raise R.PlanServerError("harness boundary — retract not under test")

    def _run(self, script, window=5.0):
        tel, rig = _GraspTel(), _DualRig(seen=True)
        pub = _ScriptedPub(tel, script)
        ctx = R.Ctx(_ReachArgs(), None, tel, _GraspDet(), self._EndOfHarness(),
                    pub, None, rig)
        plan = R.Plan(_DualBundle(), dict(self.TARGETS), None, {})
        keep, R.GRASP_RESULT_TIMEOUT_S = R.GRASP_RESULT_TIMEOUT_S, 5.0
        try:
            out = R.execute_and_retract(ctx, plan, tuple(self.TARGETS), {},
                                        empty_retry_s=window)
        finally:
            R.GRASP_RESULT_TIMEOUT_S = keep
        return out, pub

    def test_only_the_empty_hand_is_retried_and_it_can_succeed(self):
        """Left contacts, right is empty once and contacts on the retry. The
        mission must end DONE... at the retract, which this harness cannot run
        (no plan client) — reaching 'telemetry stale before retract'/FATAL with
        BOTH hands held is the success signal here, because it means the grasp
        phase completed with both cubes in hand."""
        (outcome, why, held, _), pub = self._run(
            {"left": [True], "right": [False, True]})
        self.assertEqual(held, ("left", "right"),
                         f"grasp phase did not confirm both: {why}")
        self.assertEqual(pub.grabs["right"], 2, "the empty hand was not retried")
        self.assertEqual(pub.grabs["left"], 1,
                         "the LOADED hand was sent another grab — that squeezes "
                         "the cube it already holds")
        self.assertEqual(pub.opens.get("left", 0), 0,
                         "the loaded hand's fingers were commanded")
        self.assertGreaterEqual(pub.opens.get("right", 0), 1,
                                "the empty hand was never reopened")

    def test_window_exhausted_with_one_cube_held_is_FATAL_with_that_hand_named(self):
        """The bounded end of the mixed case: right never closes on anything,
        the window runs out, and the ONLY safe end state is holding — with
        `held` naming the loaded hand so refuse_holding(holding=True) fires."""
        (outcome, why, held, _), pub = self._run(
            {"left": [True], "right": [False] * 50}, window=0.4)
        self.assertEqual(outcome, R.FATAL)
        self.assertEqual(held, ("left",))
        self.assertIn("left", why)
        self.assertIn("window exhausted", why)

    def test_both_empty_keeps_retrying_inside_the_window_then_skips(self):
        (outcome, why, held, _), pub = self._run(
            {"left": [False] * 50, "right": [False] * 50}, window=0.4)
        self.assertEqual(outcome, R.SKIP)
        self.assertEqual(held, ())
        self.assertGreater(pub.grabs["left"], 1)
        self.assertGreater(pub.grabs["right"], 1)
        self.assertIn("re-planning", why)

    def test_an_unanswered_hand_counts_as_maybe_holding(self):
        """Left contacts, right never answers: the verdict is lost, not empty.
        NOTHING may be reopened and BOTH count as held — some hands can have
        confirmed contact moments before the other timed out, and the cost of
        guessing wrong is a dropped cube."""
        (outcome, why, held, _), pub = self._run(
            {"left": [True], "right": []})
        self.assertEqual(outcome, R.FATAL)
        self.assertEqual(held, ("left", "right"))
        self.assertIn("FAIL CLOSED", why)
        self.assertEqual(pub.opens, {}, "reopened a hand with an unknown verdict")


@unittest.skipIf(R is None, _SKIP)
class StandingRoundsTests(unittest.TestCase):
    """The standing mission re-plans on SKIP, like a journey leg."""

    def test_the_standing_path_is_wired_to_the_same_round_and_window_constants(self):
        """A dead constant passed review once already (the audit's class-1
        finding), so the wiring itself is what gets asserted: main must read
        both the round bound and the regrasp window."""
        import inspect
        src = inspect.getsource(R.main)
        self.assertIn("JOURNEY_LEG_PLAN_ROUNDS", src)
        self.assertIn("empty_retry_s=JOURNEY_EMPTY_REGRASP_S", src)
        self.assertIn("holding=bool(held)", src)


class _ReachArgs:
    grasp = True
    max_rate = 0.3
    mpc = False          # these tests pin the OPEN-LOOP execute path; the
                         # tracked path has its own suite (test_mpc_track)
    tables = ()
    table = ()
    cubes = ("grasp_cube_40mm",)
    no_table_world = True
    clearance_floor_mm = 10.0
    other_arm_fallback = True
    execute = True
    chest_home = False
    journey = True


# ---------------------------------------------------------------------------
# The 07-31 audit: three ways a cube ends up on the floor (no deadlocks found)
# ---------------------------------------------------------------------------

@unittest.skipIf(R is None, _SKIP)
class SkipLeavesHandsOpenTests(unittest.TestCase):
    """A SKIP hands the next round these grippers, and the next round's FIRST
    act is stream_route — the arm is driven back to a grasp pose certified for
    an OPEN gripper (build_scene keeps the cube a hard obstacle; cuRobo's
    goalset puts it in the open jaw's mouth). Closed fingertips sweep exactly
    the volume the cube occupies. Four independent auditors found this and none
    could refute it; worst case on a journey, the shut fist tucks, walks and
    approaches the NEXT cube.

    Opening is provably safe on this path: every side in `empty` was CERTIFIED
    empty by the service, which only says so after ARRIVING at the calibrated
    close position with no load. It also clears the service's latched
    grasp_detected — hand_open is the only thing that does — without which the
    next invocation's zero_grippers refuses to calibrate at all.
    """

    def _run(self, script, seen=True, window=0.4):
        tel, rig = _GraspTel(), _DualRig(seen=seen)
        pub = _ScriptedPub(tel, script)
        ctx = R.Ctx(_ReachArgs(), None, tel, _GraspDet(),
                    MixedGraspRetryTests._EndOfHarness(), pub, None, rig)
        plan = R.Plan(_DualBundle(),
                      {"grasp_cube_40mm": np.zeros(3),
                       "grasp_cube_60mm": np.zeros(3)}, None, {})
        keep, R.GRASP_RESULT_TIMEOUT_S = R.GRASP_RESULT_TIMEOUT_S, 5.0
        try:
            out = R.execute_and_retract(
                ctx, plan, ("grasp_cube_40mm", "grasp_cube_60mm"), {},
                empty_retry_s=window)
        finally:
            R.GRASP_RESULT_TIMEOUT_S = keep
        return out, pub

    def test_the_window_skip_leaves_both_hands_OPEN(self):
        """THE property is the LAST finger command, not that an open was ever
        sent: the in-loop retries reopen before each re-grab, so counting opens
        passes with or without the fix. What decides whether the next round
        approaches with a fist is which command the hand saw last."""
        (outcome, _, held, _), pub = self._run(
            {"left": [False] * 50, "right": [False] * 50})
        self.assertEqual(outcome, R.SKIP)
        self.assertEqual(held, ())
        for side in ("left", "right"):
            self.assertEqual(
                pub.order[side][-1], "hand_open",
                f"{side} was left PINCHED for the next round's approach "
                f"(sequence {pub.order[side]})")

    def test_the_unseen_cube_skip_also_leaves_them_open(self):
        """Different exit, same hazard: the mission goes on with these hands."""
        (outcome, why, held, _), pub = self._run(
            {"left": [False] * 50, "right": [False] * 50}, seen=False)
        self.assertEqual(outcome, R.SKIP)
        self.assertIn("no longer seen", why)
        for side in ("left", "right"):
            self.assertEqual(pub.order[side][-1], "hand_open",
                             f"{side} left pinched (sequence {pub.order[side]})")

    def test_a_hand_that_HOLDS_is_never_opened_on_the_way_out(self):
        """The mirror image, and the one that drops a cube if it is wrong: when
        anything is held the outcome is FATAL, and NO hand may be opened —
        opening the loaded one drops its cube, and the empty one is about to be
        frozen for inspection anyway."""
        (outcome, _, held, _), pub = self._run(
            {"left": [True], "right": [False] * 50})
        self.assertEqual(outcome, R.FATAL)
        self.assertEqual(held, ("left",))
        self.assertEqual(pub.opens.get("left", 0), 0,
                         "the LOADED hand was opened — that drops the cube")
        self.assertEqual(pub.order["right"][-1], "hand_grab",
                         "the empty peer was opened on a FATAL exit — those "
                         "arms are about to be frozen for inspection, and an "
                         "open command there is motion nobody asked for")


@unittest.skipIf(R is None, _SKIP)
class MissionHomeIsFixedTests(unittest.TestCase):
    """The retract returns to where the MISSION began, not to where the last
    failure left the arm. plan_reach used to re-sample the live encoders every
    call, which is correct exactly once: on round 2 the live reading is the arm
    out at the cube, so 'go home' would have gone to the failed grasp pose —
    silently undoing fcb7d8b, whose entire point was the opposite."""

    def test_plan_reach_prefers_the_caller_s_home_over_the_live_reading(self):
        self.assertIn("home_enc", R.plan_reach.__code__.co_varnames)
        import inspect
        src = inspect.getsource(R.plan_reach)
        self.assertIn("home = home_enc if home_enc is not None else enc", src)
        # `home` must be the 4th positional — the Plan gained a 5th field
        # (gravity, the attitude `targets` was measured at) on 2026-08-06.
        self.assertIn("Plan(bundle, targets, floor, home", src)

    def test_both_round_loops_pass_the_mission_home(self):
        """A caller that forgets it silently gets the old behaviour back."""
        import inspect
        for fn in (R.run_leg, R.main):
            with self.subTest(fn=fn.__name__):
                self.assertIn("ctx.home_enc", inspect.getsource(fn))

    def test_the_mission_home_is_captured_once_after_the_gates(self):
        import inspect
        src = inspect.getsource(R.main)
        self.assertIn("mission_home = _arm_enc", src)
        self.assertIn("home_enc=mission_home", src)
        self.assertEqual(src.count("mission_home = _arm_enc"), 1,
                         "captured more than once is captured per round again")


@unittest.skipIf(R is None, _SKIP)
class JourneyEndsHoldingTests(unittest.TestCase):
    """A journey that carried cubes must not END by going silent: 0.5 s later
    real_env's failsafe starts crawling LOADED arms toward the default pose at
    0.025 rad/s, unattended, at the end of a run that worked. The standing
    mission has always held at PUB_HZ until Ctrl+C."""

    def test_the_success_path_holds_when_anything_is_carried(self):
        import inspect
        src = inspect.getsource(R.journey_mission)
        self.assertIn("if rig._held:", src)
        self.assertIn("refuse_holding", src)
        self.assertIn("holding=True", src)

    def test_an_empty_handed_journey_still_returns(self):
        """Nothing carried -> nothing to hold, and holding an empty hand is the
        frozen-forever-over-nothing failure the audit closed yesterday."""
        import inspect
        src = inspect.getsource(R.journey_mission)
        self.assertIn("return 0", src.split("if rig._held:")[1])
