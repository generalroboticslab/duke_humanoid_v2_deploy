"""The seven must-fix defects from the 2026-08-07 adversarial journey audit.

A 153-agent audit raised 94 candidates and confirmed 16 through three
independent refuters each. Seven of those were ranked must-fix-before-the-first-
hardware-journey, on this cut line: ANYTHING THAT ENDS THE MISSION WITH A CUBE IN
A HAND, WALKS WITH AN EXTENDED LOADED ARM, OR MAKES A LEG'S EXIT CONDITION
UNREACHABLE. This file pins each of the seven.

Six of them share one root cause, and it is an ABSENCE rather than a bug:
``rig._held`` was read in exactly three places — marking a leg's end, exempting
held arms from the walking tuck, and the hold-vs-release decision at mission end
— and NOWHERE in the plan, nav or grasp path. So on leg 2 the tool could
nominate, steer toward, plan onto and finally re-close the hand that was already
carrying leg 1's cube. The EE service certifies that stalled close as CONTACT
(`run_grasp_close` returns True on load), so the failure mode is not a freeze but
a MISSION THAT LIES: "2/2 carried" printed with the second cube still on the
table. That is worse than any deadlock, because it defeats the operator's
supervision rather than asking for it.

The seventh is unrelated and cheaper: real_env latches a joystick override and
silently discards every nav command until LB is pressed, which no autonomous
client could observe.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R
    from auto_operator import independent as IND
    from humanoid_curobo_client import GATE_E_HOME_BRANCH_TOL_RAD

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# 1. The joystick latch is OBSERVABLE, and the walk names it
# ---------------------------------------------------------------------------

@unittest.skipIf(R is None, _SKIP)
class NavPausedIsObservableTests(unittest.TestCase):
    """`_nav_paused` latches on any stick deflection past
    `_NAV_OVERRIDE_THRESHOLD` and ONLY `[NAV_RESUME]` (LB) clears it. While it is
    set real_env feeds the policy the stick, not the socket, so every nav_cmd on
    9873 is dropped. Journey's arrival test is `dist <= JOURNEY_ARRIVE_M` on a
    base that is not moving — mathematically unsatisfiable — so each leg burned
    JOURNEY_WALK_TIMEOUT_S and SKIPped while the log printed the speeds it was
    publishing into a void. A skipped cube is never retried, so one stick nudge
    before launch silently costs the whole mission."""

    def test_real_env_publishes_the_latch(self):
        """The client cannot react to a state it cannot see. This is the entire
        fix on the real_env side: one telemetry key."""
        # Read, not import: real_env pulls in the CAN/nng stack, which is not
        # available in a test env and is not what this pins.
        src = (pathlib.Path(__file__).resolve().parent.parent
               / "humanoid_real_env.py").read_text()
        self.assertIn('"nav_paused": bool(self._nav_paused)', src,
                      "real_env stopped publishing the joystick-override latch")
        self.assertIn('"[NAV_RESUME]"', src,
                      "the only documented way out of the latch is gone")

    def test_the_walk_has_a_bounded_tolerance_for_it(self):
        """A pause is legitimate — someone may have taken the stick on purpose —
        so it must not be instantly fatal. But it must not be infinitely
        tolerated either, or the bound is JOURNEY_WALK_TIMEOUT_S per leg again."""
        self.assertLess(R.JOURNEY_NAV_HANDBACK_S, R.JOURNEY_WALK_TIMEOUT_S,
                        "the handback window must expire BEFORE the walk "
                        "timeout, or naming the latch changes nothing")
        self.assertGreater(R.JOURNEY_NAV_HANDBACK_S, 0.0)

    def test_the_walk_loop_reads_it_and_ends_the_MISSION(self):
        """FATAL, not SKIP. Every remaining leg would fail identically, and a
        skipped cube is never retried — so skipping walks the whole order into
        the ground one 150 s timeout at a time."""
        src = R.run_leg.__code__.co_consts
        flat = " ".join(str(c) for c in src if isinstance(c, str))
        self.assertIn("nav_paused", flat + str(R.run_leg.__code__.co_names))
        self.assertIn("NAV PAUSED", flat,
                      "the walk never tells the operator why it is not moving")


# ---------------------------------------------------------------------------
# 2. + 4. A cube in the hand is never a reason to give up on getting home
# ---------------------------------------------------------------------------

class _TrailTel:
    def __init__(self, enc=None):
        self.enc = enc if enc is not None else [0.0] * 31

    def fresh(self):
        return {"joint_pos": list(self.enc), "arm_fault": [False, False],
                "projected_gravity": [0.0, 0.0, -1.0], "ee": {}}


class _TrailDet:
    def poll(self, rig):
        pass


class _TrailRig:
    def __init__(self):
        self.published = []

    def tick(self, tel, packet=None):
        out = packet if packet is not None else {}
        self.published.append(out)
        return out


class _TrailPub:
    def __init__(self):
        self.sent = []

    def publish(self, packet):
        self.sent.append(packet)


class _TrailArgs:
    """The args execute_and_retract / glide_home_residual read. Mirrors
    test_journey_curobo._ReachArgs; the open-loop path is what these pin."""
    grasp = True
    max_rate = 0.6
    mpc = False
    tables = ()
    table = ()
    cubes = ("cube_b",)
    no_table_world = True
    clearance_floor_mm = 10.0
    other_arm_fallback = True
    execute = True
    chest_home = False
    journey = True


@unittest.skipIf(R is None, _SKIP)
class TrailReturnTests(unittest.TestCase):
    """The retract had FOUR ways to give up with a cube in the hand — the three
    scene exceptions and a plan-server error — and all four returned FATAL 25
    lines above a certified fallback that was already written. The measured trail
    is the corridor the arm physically traversed minutes ago (the reach C-gate
    cleared it whole) and the cubes ride in closed hands, so it is available on
    exactly the failures that were refusing it. Worse, the scene rebuild is
    hardest precisely when it is needed: right after a grasp has put a hand
    between the cameras and the table."""

    def test_every_post_grasp_giveup_routes_through_the_trail(self):
        names = R.execute_and_retract.__code__.co_names
        self.assertIn("_carry_home_on_the_trail", str(
            R.execute_and_retract.__code__.co_consts) + str(names),
            "the retract no longer funnels its give-up paths through the trail")

    def test_the_glide_refuses_a_branch_it_cannot_certify(self):
        """Fix 4 adds `glide_home_residual` to the trail path so a leg cannot end
        with the arm extended over the table and then WALK like that (the next
        leg's tuck exempts held arms). But that glide's whole safety argument is
        Gate E's — 'this is a small SAME-BRANCH offset, so a joint lerp is a
        short straight motion, not a branch crossing' — and no Gate E runs on the
        trail path. So it must measure before it moves."""
        tel = _TrailTel()
        home = {n: 0.0 for n in R.ARM_JOINTS_ALL}
        # A gap far past the same-branch bar: a lerp across this could sweep the
        # gripper through unmodelled space.
        for i in range(13, 20):                      # left arm block
            tel.enc[i] = GATE_E_HOME_BRANCH_TOL_RAD * 3.0
        pub, rig = _TrailPub(), _TrailRig()
        ctx = R.Ctx(_TrailArgs(), None, tel, _TrailDet(), None, pub, None, rig)
        R.glide_home_residual(ctx, ("left",), home, gated_branch=False)
        self.assertEqual(pub.sent, [],
                         "it lerped across an uncertified branch gap")

    def test_a_small_gap_still_glides_on_the_trail_path(self):
        """The guard must not disable the fix it is protecting: inside the bar
        the same-branch argument holds and the residual should still close."""
        tel = _TrailTel()
        home = {n: 0.0 for n in R.ARM_JOINTS_ALL}
        for i in range(13, 20):                      # left arm block
            tel.enc[i] = GATE_E_HOME_BRANCH_TOL_RAD * 0.5
        pub, rig = _TrailPub(), _TrailRig()
        ctx = R.Ctx(_TrailArgs(), None, tel, _TrailDet(), None, pub, None, rig)
        R.glide_home_residual(ctx, ("left",), home, gated_branch=False)
        self.assertGreater(len(pub.sent), 0, "the residual was never closed")

    def test_the_gated_path_is_unchanged(self):
        """Gate E already certified the branch, so that call must not start
        measuring and refusing — it would undo the 2026-08-01 fix."""
        tel = _TrailTel()
        home = {n: 0.0 for n in R.ARM_JOINTS_ALL}
        for i in range(13, 20):                      # left arm block
            tel.enc[i] = GATE_E_HOME_BRANCH_TOL_RAD * 3.0
        pub, rig = _TrailPub(), _TrailRig()
        ctx = R.Ctx(_TrailArgs(), None, tel, _TrailDet(), None, pub, None, rig)
        R.glide_home_residual(ctx, ("left",), home)      # gated_branch=True
        self.assertGreater(len(pub.sent), 0,
                           "the Gate-E-certified glide stopped working")


# ---------------------------------------------------------------------------
# 3. A fault on ONE arm must not release the other, loaded, healthy one
# ---------------------------------------------------------------------------

class _HoldTel:
    """arm_fault is [left, right] and TERMINAL once latched."""

    def __init__(self, faults):
        self.faults = list(faults)
        self.reads = 0

    def fresh(self):
        self.reads += 1
        return {"joint_pos": [0.0] * 31, "arm_fault": list(self.faults),
                "projected_gravity": [0.0, 0.0, -1.0], "ee": {}}


class _HoldRig:
    def tick(self, tel, packet=None):
        return packet if packet is not None else {}


@unittest.skipIf(R is None, _SKIP)
class ArmFaultReleaseTests(unittest.TestCase):
    """`arm_fault` is the dropout watchdog's terminal latch AND the entry
    condition for six of run_leg's FATALs. So the very failure that routes into
    refuse_holding re-read its own latch on tick one, returned, published
    nothing — and real_env's 0.5 s arm-silence failsafe then crawled the OTHER
    arm, healthy and by construction the one holding a cube, to the default pose
    at 0.025 rad/s. A faulted arm is already frozen by the watchdog and cannot be
    helped by this loop; the healthy loaded one is what the loop exists for."""

    def _run(self, faults, held_sides=(), ticks=3):
        tel = _HoldTel(faults)
        sent = []

        class _P:
            def publish(self, packet):
                sent.append(packet)
                if len(sent) >= ticks:
                    raise KeyboardInterrupt

        code = R.refuse_holding("x", 5, _P(), tel, _TrailDet(), _HoldRig(),
                                0.3, holding=True, held_sides=held_sides)
        return code, sent

    def test_every_holding_arm_faulted_releases(self):
        """Nothing this loop publishes can help an arm the watchdog froze."""
        _, sent = self._run([True, True], held_sides=("left", "right"))
        self.assertEqual(sent, [], "it kept publishing at two dead arms")
        _, sent = self._run([False, True], held_sides=("right",))
        self.assertEqual(sent, [],
                         "the only loaded hand was frozen and it held anyway")

    def test_a_fault_on_the_OTHER_arm_keeps_holding_the_loaded_one(self):
        """THE regression. arm_fault is terminal and is the entry condition for
        six of run_leg's FATALs, so the failure that routes here re-read its own
        latch and released the loaded peer to the 0.025 rad/s crawl."""
        for faults, held in (([True, False], ("right",)),
                             ([False, True], ("left",))):
            with self.subTest(faults=faults, held=held):
                _, sent = self._run(faults, held_sides=held)
                self.assertGreater(
                    len(sent), 0,
                    f"arm_fault={faults} released the loaded {held[0]} arm")

    def test_a_partial_fault_with_BOTH_loaded_keeps_holding(self):
        _, sent = self._run([True, False], held_sides=("left", "right"))
        self.assertGreater(len(sent), 0, "one fault released a loaded peer")

    def test_callers_that_name_no_sides_release_on_any_fault(self):
        """Back-compat, and deliberate: without side information the live risk
        is the 07-30 one — holding an EMPTY hand forever."""
        _, sent = self._run([False, True])
        self.assertEqual(sent, [])

    def test_no_fault_holds_as_before(self):
        _, sent = self._run([False, False], held_sides=("left",))
        self.assertGreater(len(sent), 0)


# ---------------------------------------------------------------------------
# 5. One raw gyro sample must not cost the whole stillness window
# ---------------------------------------------------------------------------

class _StillOp:
    """The fields journey_base_still touches. Claims the operator's module so
    `_K(op)` resolves the REAL constants."""

    def __init__(self, ang_z=0.0):
        self._base_vel = None
        self._base_ang = [0.0, 0.0, ang_z]
        self._journey_lin_seen = False
        self._journey_blind_warned = True
        self._journey_still_since = None


_StillOp.__module__ = "humanoid_auto_operator"


@unittest.skipIf(R is None, _SKIP)
class StillnessDebitTests(unittest.TestCase):
    """`base_ang_vel` arrives raw and unfiltered, the bar is 0.10 rad/s, and the
    blind window needs ~125 consecutive good ticks at 50 Hz. Nulling the window
    on any single sample meant one outlier restarted all of it — and window
    expiry is the leg's ONLY measurement-driven SKIP, so a lone spike could cost
    the cube. Debiting keeps the gate's meaning while bounding what an outlier
    costs."""

    def test_a_lone_spike_costs_the_debit_not_the_window(self):
        op = _StillOp()
        IND.journey_base_still(op, 100.0)            # window opens
        self.assertEqual(op._journey_still_since, 100.0)
        op._base_ang = [0.0, 0.0, 10.0]              # one wild sample
        self.assertFalse(IND.journey_base_still(op, 100.5))
        self.assertAlmostEqual(op._journey_still_since,
                               100.0 + OP.JOURNEY_STILL_DEBIT_S, places=9,
                               msg="a single spike discarded the whole window")

    def test_sustained_motion_still_drives_the_credit_to_zero(self):
        """The debit must not turn the gate into a rubber stamp: real motion
        produces bad ticks continuously, and those must still walk the start
        time up to `now` — which is exactly the old reset."""
        op = _StillOp()
        IND.journey_base_still(op, 100.0)
        op._base_ang = [0.0, 0.0, 10.0]
        now = 100.0
        for _ in range(60):                           # ~1.2 s of real motion
            now += 0.02
            self.assertFalse(IND.journey_base_still(op, now))
        self.assertAlmostEqual(op._journey_still_since, now, places=9,
                               msg="sustained motion did not exhaust the credit")

    def test_the_debit_never_runs_the_clock_into_the_future(self):
        op = _StillOp()
        IND.journey_base_still(op, 100.0)
        op._base_ang = [0.0, 0.0, 10.0]
        IND.journey_base_still(op, 100.01)            # debit > elapsed
        self.assertLessEqual(op._journey_still_since, 100.01)

    def test_a_quiet_base_still_latches_after_the_sustain(self):
        """The gate must keep working at all."""
        op = _StillOp()
        self.assertFalse(IND.journey_base_still(op, 100.0))
        self.assertTrue(
            IND.journey_base_still(op, 100.0 + OP.JOURNEY_STILL_S_BLIND + 0.01))


# ---------------------------------------------------------------------------
# 6. + 7. The free-hand rule: nominate, steer, plan and grasp with an EMPTY hand
# ---------------------------------------------------------------------------

@unittest.skipIf(R is None, _SKIP)
class FreeHandTests(unittest.TestCase):

    def test_the_serving_arm_is_still_the_cubes_own_side_when_free(self):
        """Unchanged behaviour on leg 1 and on any leg with both hands empty.
        Crossing the midline past ~0.10 m was measured INFEASIBLE on 07-30, so
        the side is not a preference."""
        for y, want in ((+0.20, "left"), (-0.20, "right"), (0.0, "left")):
            with self.subTest(y=y):
                pos = np.array([0.45, y, 0.05])
                self.assertEqual(R.journey_serving_arm(pos), want)
                self.assertEqual(
                    R.journey_serving_arm(pos, ("left", "right")), want)

    def test_a_cube_DEEP_in_the_full_hands_side_yields_None(self):
        """Beyond CROSS_MIDLINE_REACH_M the 07-30 sweep says there is no route,
        so None — the caller must SKIP loudly rather than hand the planner a
        pairing measured as having no solution."""
        deep = OP.CROSS_MIDLINE_REACH_M * 2.0
        self.assertIsNone(
            R.journey_serving_arm(np.array([0.45, deep, 0.05]), ("right",)))
        self.assertIsNone(
            R.journey_serving_arm(np.array([0.45, -deep, 0.05]), ("left",)))

    def test_a_cube_NEAR_the_midline_is_taken_by_the_free_hand(self):
        """Hardware 2026-08-07. The 07-30 finding is 'INFEASIBLE past ~0.10 m'
        — a BAND. This was implemented as a sign test, so a leg was refused over
        a cube at y=+0.026: 26 mm, a fifth of the way to the bar and inside the
        detector's own noise, with the other hand free. A 100 mm tolerance had
        become a 0 mm one."""
        self.assertEqual(
            R.journey_serving_arm(np.array([0.45, +0.026, 0.05]), ("right",)),
            "right", "a 26 mm crossing still refuses the free hand")
        self.assertEqual(
            R.journey_serving_arm(np.array([0.45, -0.026, 0.05]), ("left",)),
            "left")

    def test_the_band_edge_is_the_measured_number(self):
        """Inclusive at the bar, None just past it — so the constant is the
        contract and not an approximation of one."""
        edge = OP.CROSS_MIDLINE_REACH_M
        self.assertEqual(
            R.journey_serving_arm(np.array([0.45, edge, 0.05]), ("right",)),
            "right")
        self.assertIsNone(
            R.journey_serving_arm(np.array([0.45, edge + 1e-3, 0.05]),
                                  ("right",)))

    def test_the_own_side_still_wins_when_it_is_free(self):
        """The band is a fallback, not a preference: crossing is measurably
        worse than not crossing, so a free own-side hand must always take it."""
        self.assertEqual(
            R.journey_serving_arm(np.array([0.45, +0.02, 0.05]),
                                  ("left", "right")), "left")
        self.assertEqual(
            R.journey_serving_arm(np.array([0.45, -0.02, 0.05]),
                                  ("left", "right")), "right")

    def test_a_cube_on_the_free_hands_side_is_served(self):
        self.assertEqual(
            R.journey_serving_arm(np.array([0.45, 0.20, 0.05]), ("left",)),
            "left")

    def test_forced_assignments_never_offer_a_full_hand(self):
        """The fallback ladder enumerates pairings exhaustively — which, with a
        loaded hand, means half its entries are ways to re-grab the cube already
        held."""
        self.assertEqual(R.forced_assignments(("a",), ("left",)), [{"L": "a"}])
        self.assertEqual(R.forced_assignments(("a",), ("right",)), [{"R": "a"}])
        self.assertEqual(R.forced_assignments(("a",), ()), [])

    def test_forced_assignments_is_unchanged_without_a_restriction(self):
        """Standing missions pass nothing and must keep both pairings."""
        self.assertEqual(R.forced_assignments(("a",)),
                         [{"L": "a"}, {"R": "a"}])
        self.assertEqual(R.forced_assignments(("a", "b")),
                         [{"L": "a", "R": "b"}, {"L": "b", "R": "a"}])

    def test_the_planner_is_given_the_restriction(self):
        """AUTO is the server's own pick and is blind to which hands are free,
        so a restricted caller must FORCE the assignment rather than ask and
        hope."""
        for fn in (R.plan_reach, R.plan_with_reacquire):
            with self.subTest(fn=fn.__name__):
                self.assertIn("only_sides", fn.__code__.co_varnames,
                              f"{fn.__name__} cannot be restricted to free hands")

    def test_the_leg_start_decision_reads_the_FILTERED_fix(self):
        """Hardware 2026-08-07. `cube.pos` is the last ingested detection with
        no freshness test and no evidence filter — one grazing 1-face PnP fit
        sets it, and on a two-cube run the far cube's claim churns every
        DET_UNCLAIM_S so exactly those marginal fits land. A bad y sign then
        chose the serving arm and killed the leg before a step was taken: seen
        live, cube on the robot's LEFT, refused as right-side."""
        self.assertIn("median", R.run_leg.__code__.co_names,
                      "run_leg still decides the serving arm from the raw "
                      "last sighting instead of the filtered window")

    def test_a_full_side_is_STEERED_ACROSS_not_refused_up_front(self):
        """The nav aim offset exists to park the cube in the serving arm's
        half-space, and a leg is 0.6-0.8 m of travel — so a cube on the full
        hand's side is a geometry the BASE can fix. The 07-30 INFEASIBLE
        measurement that motivated the old up-front refusal was taken STANDING,
        arm alone, base fixed."""
        consts = " ".join(str(c) for c in R.run_leg.__code__.co_consts)
        self.assertIn("ACROSS", consts,
                      "run_leg no longer announces the cross-midline approach")
        self.assertIn("after the approach", consts,
                      "the SKIP is no longer taken from the post-arrival fix")

    def test_a_skipped_cube_leaves_the_pending_set(self):
        """ao_gaze.eye keeps a camera aimed at any cube that is not `reached`
        and sweeps the other, re-claiming every DET_UNCLAIM_S — so a skipped
        cube left the eyes churning through the entire final hold, with a cube
        in the hand and nothing left to look for. The park-at-zero branch needs
        EVERY cube reached, so one skip disabled it outright."""
        src = R.journey_mission.__code__
        self.assertIn("reached", src.co_names,
                      "journey_mission never retires a skipped cube, so the "
                      "cameras hunt it forever")

    def test_run_leg_computes_the_free_set_and_passes_it(self):
        self.assertIn("_held", R.run_leg.__code__.co_names,
                      "run_leg still decides the serving arm blind to _held")
        consts = " ".join(str(c) for c in R.run_leg.__code__.co_consts)
        self.assertIn("only_sides", consts,
                      "run_leg does not restrict the planner to free hands")


# ---------------------------------------------------------------------------
# 6 (backstop). Closing a loaded gripper is refused, loudly
# ---------------------------------------------------------------------------

class _ClashBundle:
    """Mirrors test_journey_curobo._GraspBundle, but assigns the NEW cube to the
    LEFT hand — the hand _ClashRig already reports as loaded."""
    goal = "reach"
    active_sides = ("left",)
    assignment = {"L": "cube_b"}

    def __init__(self):
        self.joint_names = tuple(R.ARM_JOINTS_L) + tuple(R.ARM_JOINTS_R)
        self.q_enc = np.zeros((2, len(self.joint_names)))
        self.horizon = 2
        self.duration_s = 0.0
        self.dt = 0.0
        self.controlled_joints = tuple(R.ARM_JOINTS_L)

    def q_at(self, t):
        return self.q_enc[0]

    def stretch(self, rate):
        return self


class _ClashRig(_TrailRig):
    def __init__(self):
        super().__init__()
        self._held = {"left": {"key": "cube_a"}}

    def median(self, key, now):
        return np.array([0.45, 0.2, 0.05])

    def latest(self, key, now):
        return self.median(key, now)

    def last_inliers(self, key):
        return 3

    def faces_used(self, key, now=None):
        return 3


@unittest.skipIf(R is None, _SKIP)
class LoadedGripperBackstopTests(unittest.TestCase):
    """Defence in depth behind `only_sides`. Re-closing a loaded hand runs the
    stepped close AGAIN on the held cube and the service certifies CONTACT on
    load — so without this the mission counts a cube it never picked up. FATAL
    rather than dropping the side: a plan built on the wrong hand is not
    something the next round can improve on, and the hand is loaded, so the
    caller must be told to hold rather than release."""

    def test_it_refuses_before_any_grasp_is_sent(self):
        pub, rig = _TrailPub(), _ClashRig()
        ctx = R.Ctx(_TrailArgs(), None, _TrailTel(), _TrailDet(), None, pub,
                    None, rig)
        plan = R.Plan(_ClashBundle(), {"cube_b": np.zeros(3)}, None, {})
        outcome, why, held, _ = R.execute_and_retract(
            ctx, plan, ("cube_b",), {}, empty_retry_s=0.0)
        self.assertEqual(outcome, R.FATAL)
        self.assertIn("already carrying", why)
        self.assertIn("cube_a", why)
        self.assertEqual(held, ("left",),
                         "the loaded hand must be reported as held, or the "
                         "caller releases a gripper with a cube in it")
        self.assertFalse(any("ee_action" in str(p) for p in pub.sent),
                         "a gripper command was sent at a loaded hand")


# ---------------------------------------------------------------------------
# --fresh-scan: the origin has to mean what it says
# ---------------------------------------------------------------------------

@unittest.skipIf(R is None, _SKIP)
class FreshScanTests(unittest.TestCase):
    """`zero_gimbals` drives the cameras to the origin and, in the SAME loop,
    calls det.poll(rig) every tick. With the monitor left running across a
    relaunch the cubes are still being decoded, so positions and first-seer
    locks land DURING the drive home and eye() aims straight at them — the scan
    banner prints and nothing sweeps.

    That is the 08-03 fast path and it stays the default. But it makes the
    origin a lie, and the operator could not demonstrate see->scan->fix->grasp
    without tearing down all five processes (user, 2026-08-08)."""

    def _rig_with_a_remembered_cube(self):
        rig = R.GazeRig(["grasp_cube_60mm_a"])
        rig.ingest({"grasp_cube_60mm_a": {"pos": [0.45, 0.2, 0.05],
                                          "camera_port": 5555, "n_inliers": 3,
                                          "capture_stamp": 10.0}}, 10.0)
        return rig

    def test_it_clears_every_place_a_position_can_hide(self):
        rig = self._rig_with_a_remembered_cube()
        cube = rig.cubes["grasp_cube_60mm_a"]
        self.assertIsNotNone(cube.pos)
        self.assertTrue(cube.hist)
        R.forget_all_cubes(rig)
        self.assertIsNone(cube.pos, "the position survived")
        self.assertFalse(cube.hist, "the sighting history median() reads survived")
        self.assertIsNone(cube.camera_port, "the claim survived")
        self.assertIsNone(cube.last_camera_port,
                          "last_camera_port survives an unclaim by design, so "
                          "leaving it would re-aim the eye anyway")
        self.assertIsNone(rig.median("grasp_cube_60mm_a", 10.1))

    def test_the_eye_actually_switches_from_aiming_to_scanning(self):
        """The property that matters. Clearing fields is only useful if eye()
        then takes the scan branch — SCAN_PITCH_DOWN on BOTH cameras, where
        before only the claiming eye was aimed."""
        from auto_operator import gaze as G
        rig = self._rig_with_a_remembered_cube()
        rig.gaze = np.zeros(4)
        rig._gaze_seeded = True
        jp = np.zeros(31)
        before = G.eye(rig, jp, 10.05).copy()
        R.forget_all_cubes(rig)
        after = G.eye(rig, jp, 10.10).copy()
        self.assertFalse(np.allclose(before, after), "eye() did not change")
        pitches = [after[p] for _, p in OP.GAZE_ORDER.values()]
        for pitch in pitches:
            self.assertAlmostEqual(abs(float(pitch)), OP.SCAN_PITCH_DOWN,
                                   places=6,
                                   msg="not scanning: pitch is not 47 deg down")

    def test_the_scan_clock_restarts(self):
        """A stale _scan_phase would drop the sweep in mid-lap instead of at the
        settle, which is what SCAN_SETTLE_S exists to prevent."""
        rig = self._rig_with_a_remembered_cube()
        rig._scan_phase = 37.0
        R.forget_all_cubes(rig)
        self.assertEqual(rig._scan_phase, 0.0)

    def test_it_is_opt_in(self):
        """Default OFF: the fast loop (leave the monitor up, relaunch only this
        tool) is what the 08-03 fix bought and it stays."""
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(R.Args)}
        self.assertIn("fresh_scan", f)
        self.assertIs(f["fresh_scan"].default, False)


# ---------------------------------------------------------------------------
# 8. Stop on skip (user 2026-08-08): a leg that did not grasp ends the mission
# ---------------------------------------------------------------------------

class _StopCube:
    def __init__(self):
        self.reached = False


class _StopRig:
    def __init__(self, keys=("cube_a", "cube_b")):
        self._survey_order = list(keys)
        self.cubes = {k: _StopCube() for k in keys}
        self._held = set()


class StopOnSkipTests(unittest.TestCase):
    """2026-08-08, first 0.4 m/s journey: leg 1 failed to plan (overshoot into
    the wind-up basin) and the robot marched off to leg 2, abandoning the
    ungrasped cube. User: 'if the 1st one is not grasped, the humanoid should
    not move to the next one.' Default now = stop at the failed cube; the old
    salvage walk-on lives behind --journey-continue-on-skip."""

    def _run(self, continue_on_skip: bool):
        from types import SimpleNamespace
        calls = {"legs": [], "refuse": []}
        orig = (R.journey_survey, R.run_leg, R.refuse_holding)
        R.journey_survey = lambda rig, now: True
        def _leg(ctx, key, i, n):
            calls["legs"].append(key)
            return R.SKIP, "no gated route"
        R.run_leg = _leg
        def _refuse(msg, code, *a, **kw):
            calls["refuse"].append((msg, code, kw))
            return code
        R.refuse_holding = _refuse
        try:
            rig = _StopRig()
            ctx = SimpleNamespace(
                rig=rig, home_enc={}, pub=None, tel=None, det=None,
                args=R.Args(journey_continue_on_skip=continue_on_skip,
                            no_table_world=True))
            rc = R.journey_mission(ctx)
        finally:
            R.journey_survey, R.run_leg, R.refuse_holding = orig
        return rc, calls, rig

    def test_the_default_stops_at_the_first_failed_grasp(self):
        rc, calls, rig = self._run(continue_on_skip=False)
        self.assertEqual(calls["legs"], ["cube_a"],
                         "leg 2 was attempted after leg 1 failed to grasp — "
                         "the robot walked away from the ungrasped cube")
        self.assertEqual(rc, 5)
        self.assertEqual(len(calls["refuse"]), 1)
        msg, code, kw = calls["refuse"][0]
        self.assertIn("stopping here", msg)
        self.assertIn("cube_b", msg)          # the un-attempted cube is named
        self.assertFalse(kw.get("holding"), "nothing was held, yet the stop "
                                            "claimed a loaded hand")

    def test_the_stop_still_holds_a_cube_from_an_earlier_leg(self):
        """Leg 1 grasps, leg 2 fails: the stop must keep the loaded hand held
        (refuse_holding with holding=True and the side named), never release
        a carried cube to the failsafe."""
        from types import SimpleNamespace
        calls = {"refuse": []}
        orig = (R.journey_survey, R.run_leg, R.refuse_holding)
        R.journey_survey = lambda rig, now: True
        def _leg(ctx, key, i, n):
            if key == "cube_a":
                ctx.rig._held.add("left")
                return R.DONE, "carried"
            return R.SKIP, "no gated route"
        R.run_leg = _leg
        def _refuse(msg, code, *a, **kw):
            calls["refuse"].append((msg, code, kw))
            return code
        R.refuse_holding = _refuse
        try:
            ctx = SimpleNamespace(
                rig=_StopRig(), home_enc={}, pub=None, tel=None, det=None,
                args=R.Args(no_table_world=True))
            rc = R.journey_mission(ctx)
        finally:
            R.journey_survey, R.run_leg, R.refuse_holding = orig
        self.assertEqual(rc, 5)
        msg, code, kw = calls["refuse"][0]
        self.assertTrue(kw.get("holding"))
        self.assertEqual(kw.get("held_sides"), ("left",))

    def test_the_flag_restores_the_walk_on_salvage(self):
        rc, calls, rig = self._run(continue_on_skip=True)
        self.assertEqual(calls["legs"], ["cube_a", "cube_b"],
                         "the flag did not walk on to the remaining cubes")
        self.assertEqual(rc, 0)               # nothing held -> plain exit
        self.assertEqual(calls["refuse"], [])
        self.assertTrue(all(c.reached for c in rig.cubes.values()))

    def test_a_skipped_cube_is_still_retired_before_the_stop(self):
        """The eyes must hear the skip even on the stop path: refuse_holding
        keeps ticking the rig, so an un-retired cube would put one camera
        into the lock/release churn the 08-07 fix removed."""
        rc, calls, rig = self._run(continue_on_skip=False)
        self.assertTrue(rig.cubes["cube_a"].reached)

    def test_the_default_is_stop(self):
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(R.Args)}
        self.assertIn("journey_continue_on_skip", f)
        self.assertIs(f["journey_continue_on_skip"].default, False)


# ---------------------------------------------------------------------------
# 9. One-shot doctrine (user 2026-08-09): walk once, grasp from where you stop
# ---------------------------------------------------------------------------

class OneShotTests(unittest.TestCase):
    """No re-approach, no hover: the leg must win from its single stop. The
    two levers that survived review: plan DIRECT to the grasp point (the 15 mm
    hover was the wrist-corridor thief on close/high landings), and fail a
    bad-height cube in one line instead of thirty solves (height is a stand
    property — no round can fix it)."""

    def test_the_z_band_matches_the_hardware_ledger(self):
        import humanoid_auto_operator as OP
        self.assertLess(OP.JOURNEY_CUBE_Z_MIN, OP.JOURNEY_CUBE_Z_MAX)
        floor = OP.JOURNEY_Z_LEDGER_FLOOR_MM  # the floor the ledger was cut at
        # every measured successful grasp (z -0.00 .. +0.040) must be inside,
        # and since 08-13 (user: the -0.01 bar over-generalized large-radius
        # rear failures; the envelope floor is radius-dependent) so must the
        # solver-judged band down to -0.06 — z=-0.016 was the leg the old bar
        # refused without letting cuRobo speak
        for z in (-0.000, -0.016, -0.021, 0.026, 0.030, 0.040):
            self.assertIsNone(R.journey_z_verdict(z, floor),
                              f"z={z} wrongly refused")
        # the high failure pole and the below-band low pole stay outside,
        # with the right advice
        high = R.journey_z_verdict(0.070, floor)
        self.assertIsNotNone(high)
        self.assertIn("LOWER", high)
        low = R.journey_z_verdict(-0.070, floor)
        self.assertIsNotNone(low)
        self.assertIn("RAISE", low)

    def test_a_relaxed_clearance_floor_unlocks_the_high_cube_only(self):
        """User 08-09: "accept the higher cube". The 20/20 high-pole refusals
        were routes 20-25 mm from the chest missing the 25 mm bar — at a floor
        below the ledger's 25 the verdict is no longer known-lost, so the gate
        must step aside. The LOW pole is cuRobo's own envelope INFEASIBLE and
        must keep binding at ANY floor."""
        import humanoid_auto_operator as OP
        self.assertIsNone(R.journey_z_verdict(0.070, 15.0),
                          "high cube still refused at a 15 mm floor")
        self.assertIsNotNone(R.journey_z_verdict(0.070,
                                                 OP.JOURNEY_Z_LEDGER_FLOOR_MM))
        self.assertIsNotNone(R.journey_z_verdict(-0.070, 15.0),
                             "the floor-blind LOW pole stopped binding")
        # the run_leg call site must actually pass the run's floor
        import inspect
        src = inspect.getsource(R.run_leg)
        self.assertIn("journey_z_verdict(float(pos[2]), a.clearance_floor_mm)",
                      src)

    def test_run_leg_fails_fast_on_a_bad_height(self):
        import inspect
        src = inspect.getsource(R.run_leg)
        self.assertIn("journey_z_verdict(", src,
                      "run_leg no longer checks the cube height before "
                      "burning solver rounds")
        self.assertLess(src.index("journey_z_verdict("),
                        src.index("JOURNEY_LEG_PLAN_ROUNDS + 1"),
                        "the z check must come BEFORE the round loop")
        # Review 08-09 (both lenses, CONFIRMED): a mission-ending verdict must
        # clear the evidence bar every control decision does — a 1-face z
        # reading (tens of mm systematics vs a 65 mm band) may WARN but never
        # SKIP; the plan rounds judge instead.
        self.assertIn("faces_used", src,
                      "the z gate terminates the leg without checking the "
                      "evidence behind the reading — one grazing 1-face "
                      "frame can end the mission with a wrong 'move the "
                      "stand' instruction")
        self.assertIn("DRIFT_MIN_INLIERS", src)

    def test_journey_legs_plan_direct_to_the_grasp_point(self):
        import inspect
        src = inspect.getsource(R.run_leg)
        self.assertIn("a.reach_hover_mm = 0.0", src,
                      "journey legs no longer force hover 0 — the 15 mm "
                      "hover is back to stealing the wrist corridor")
        self.assertIn("a.reach_hover_mm = hover_mm0", src,
                      "the hover is not restored after the leg — the "
                      "standing mission would inherit journey's 0")
        self.assertLess(src.index("a.reach_hover_mm = 0.0"),
                        src.index("JOURNEY_LEG_PLAN_ROUNDS + 1"),
                        "hover must be zeroed before the first plan round")

    def test_the_descent_override_is_revoked(self):
        """08-09: 1-face descent tracking (the 08-08 override) caused five
        T13 goal-chases in one night; every successful grasp went
        blind-early-close-on-anchor. The freeze is the default again (and
        journey legs, planning direct, have no descent at all)."""
        self.assertFalse(R.MPC_DESCENT_1FACE_TRACKS)


if __name__ == "__main__":
    unittest.main()
