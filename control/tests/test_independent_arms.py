from __future__ import annotations

import contextlib
import enum
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from auto_operator import arbitration, arms, independent
from auto_operator.models import (
    ArmTask,
    ArmTaskPhase,
    GraspState,
    HoldState,
    Sighting,
    TrackMotion,
)


# independent._K(op) intentionally resolves constants from the module owning
# the operator class.  These test constants are therefore the fake operator's
# live configuration, just like the production entry module's constants.
MIDLINE_MARGIN_M = 0.03
GEOMETRIC_FLOOR_M = 0.28
DYN_OUTREACH_M = 0.32
GATE_CONSECUTIVE = 5
DET_STALE_S = 0.5
DET_LOST_RESCAN_S = 3.0
DYN_LOST_GRACE_S = 2.5
GRASP_NEAR_M = 0.10
GRASP_NEAR_LOST_S = 4.0
DYN_RETRACK_M = 0.01
CUBE_LATCH_WINDOW_S = 0.7
STAGE_REACH_EE_RATE = 0.12
REACH_OK_M = 0.05
REACH_TIMEOUT_S = 8.0
DT = 0.05
K_STEER = 1.0
WZ_MAX = 0.2
CRUISE_VX = 0.3
SECTOR_REACH_ENABLED = {"front": True, "side": True, "rear": True}
TRACK = ("front", "side", "rear")
GRASP_ZERO_GRACE_S = 10.0
GRASP_SETTLE_S = 1.0
GRASP_EE_OK_M = 0.06
GRASP_RESULT_TIMEOUT_S = 20.0
GRASP_DROP_DIST_M = 0.15
GRASP_DROP_PERSIST_S = 2.0
GRASP_LIFT_UP_M = 0.05
HOME_SETTLE_TOL_M = 0.06
HOME_SETTLE_TIMEOUT_S = 20.0
JOURNEY_NEAR_VX = 0.1
JOURNEY_NEAR_VX_BACK = 0.15  # 07-26: backward-leg floor (policy steps in place at 0.1 reversed)
JOURNEY_SLOW_K = 0.5
JOURNEY_STOP_M = 0.15
JOURNEY_AIM_OFFSET_M = 0.12  # 07-26: steer aim point into the serving arm's half-space
JOURNEY_AIM_OFFSET_BACK_M = 0.05  # 08-10: reverse legs steer on their own (smaller)
                                  # offset; differs from the forward value here so the
                                  # tests exercise the direction split
JOURNEY_REVERSE_DRIFT_COMP_M = 0.15  # 08-10: reverse aim biased right against the
                                  # measured backward-gait leftward settle drift
GRASP_PRECON_RETRY_S = 15.0  # 07-26: pre-grab EE-convergence timeout -> release/re-queue
ASSIGN_STALL_S = 20.0        # 07-26: standing-mission ASSIGNED no-progress release
LATCH_FAIL_SKIP_N = 3        # 07-26: consecutive latch failures before the cube is skipped
JOURNEY_POSTURE_CORRIDOR_RAD = 0.35  # 07-26: joint glide only near the SIDE<->POWERON segment
JOURNEY_STILL_VEL = 0.05
JOURNEY_STILL_ANG = 0.10
JOURNEY_STILL_S = 1.0
JOURNEY_STILL_DEBIT_S = 0.2  # 08-07: one non-still sample DEBITS the sustain
                             # window rather than nulling it (raw gyro, one
                             # outlier used to discard ~50 good ticks)
JOURNEY_WALK_EE = [0.2815, 0.2183, 0.0674]
JOURNEY_TUCK_TOL_M = 0.08
JOURNEY_WALK_QUAT = [0.9885, 0.0949, -0.1138, -0.0309]
JOURNEY_STILL_S_BLIND = 2.5
JOURNEY_TUCK_HOLD_S = 5.0
JOURNEY_ARRIVE_M = 0.34
JOURNEY_POSTURE_TOL_RAD = 0.06
# Arc-into-facing (08-08 sim parity) — production values, so the unfaced yaw
# floor behaves here exactly as it does on the robot.
JOURNEY_FACE_TOL_RAD = 0.14
JOURNEY_WZ_UNFACED_FLOOR = 0.35
JOURNEY_WZ_UNFACED_MAX = 0.5
POWERON_JOINTS = [-0.2, -0.2, 0.0, 1.0, -1.5707963267948966, -1.0, 0.0]
SIDE_HOME_JOINTS = [-0.008, -0.482, -0.006, 0.005, -0.008, -1.080, 0.010]
# 07-28 gimbal-lock-free wrist: right arm is no longer a flat negation of left —
# wrist_2 (index 5) keeps its sign. Mirrors the production module so `_K(op)`
# resolves the same convention here.
ARM_MIRROR_SIGN = np.array([-1., -1., -1., -1., -1., -1., -1.])
PARALLEL_STAIRCASE_CERTIFIED = True   # mirrors production (07-28 re-audit passed)


@contextlib.contextmanager
def _parallel_staircases(certified: bool):
    """Force the parallel-staircase gate for one test.

    Both sides need covering: the CAPABILITY (both arms step in a tick) and the
    GATE (a geometry change that fails the mid-sagittal certificate must be able
    to serialize them again by flipping one constant). The module default
    mirrors production so a silent flip there shows up as a test failure.
    """
    global PARALLEL_STAIRCASE_CERTIFIED
    prev = PARALLEL_STAIRCASE_CERTIFIED
    PARALLEL_STAIRCASE_CERTIFIED = certified
    try:
        yield
    finally:
        PARALLEL_STAIRCASE_CERTIFIED = prev


def parallel_staircases_certified():
    return _parallel_staircases(True)


def parallel_staircases_uncertified():
    return _parallel_staircases(False)


def mirror_arm(q7, arm: str) -> np.ndarray:
    q = np.asarray(q7, dtype=float)
    return q if arm == "left" else q * ARM_MIRROR_SIGN
STAGE_RATE = 0.3
BEARING_JUMP_DEG = 45.0
HOLD_RELEASE_S = 3.0


class Phase(enum.Enum):
    SEARCH = "SEARCH"
    GO = "GO"
    REACH = "REACH"
    HOLD = "HOLD"
    PARK = "PARK"
    DONE = "DONE"


class FakeCube:
    def __init__(self, key: str, pos, now: float = 10.0):
        self.key = key
        self.pos = np.asarray(pos, dtype=float)
        self.pos_port = 5555
        self.first_seen = now - 1.0
        self.last_seen = now
        self.reached = False
        self.held_by = None
        self.assigned_to = None
        self.camera_port = None
        self.last_camera_port = None
        self.hist = deque([Sighting(now, self.pos.copy(), 5555, 2)], maxlen=5)

    def see(self, now: float) -> None:
        """Refresh BOTH clocks the way perception.ingest does.

        The real CubeTrack stamps last_seen on every packet AND appends a
        capture-stamped Sighting; tests that only moved last_seen were
        simulating a state the robot cannot be in, and the latch gate now ages
        on the capture stamp (audit 2026-08-06, D12) so the two must move
        together."""
        self.last_seen = now
        self.hist.append(Sighting(now, self.pos.copy(), 5555, 2))

    def fresh(self, now: float, horizon: float) -> bool:
        return (now - self.last_seen) < horizon

    def confirmed(self, now: float) -> bool:
        # mirrors CubeTrack.confirmed: FIRST SIGHTING COUNTS (user 2026-08-02)
        return self.fresh(now, 1.0)


class FakeOperator:
    def __init__(self, cubes):
        self.cubes = {c.key: c for c in cubes}
        self._arm_tasks = {"left": None, "right": None}
        self._held = {}
        self._reach_tm = None
        self._ind_joint_owners: set = set()   # 07-21 PHASE 2: token is a set
        self._ind_gate_rr = 0
        self._ind_gate_inferred = False
        self._nav_task_key = None
        self._nav_leg_dir = None
        self._vx = 0.0
        self._wz = 0.0
        self._ee_act = {
            "left": np.array([0.3, 0.25, 0.0]),
            "right": np.array([0.3, -0.25, 0.0]),
        }
        self.rest_ee = {k: v.copy() for k, v in self._ee_act.items()}
        self._home_settled = True
        self._home_wait_t0 = None
        self._hands_opened = True
        self._zeros_t = None
        self._ee_chain_dead = False
        self._ee_queue = []
        self._ee_current = None
        self._ee_next_action_id = 1000
        self._grasp_safety_latched = False
        self._grasp_safety_arm = None
        self._grasp_safety_reason = None
        self._grasp_intervention_required = False
        self._grasp_intervention_reason = None
        self._intervention_arm_targets = None
        self.grasp = False
        self.hold_after_reach = True
        self.side_home = True
        self.stage_sectors = False
        self.dynamic_track = False
        self.visual_servo = False
        self.gate = None
        self.active = None
        self.phase = Phase.SEARCH
        self._servo_log_t = -1e9
        self._grip = {}
        self._home_stream = {"left": None, "right": None}
        self._sec = None
        self._jpos = None
        self._cam_retired = set()
        self.journey = False
        self._survey_order = None
        self._survey_warn_t = -1e9
        self._journey_still_since = None
        self._journey_walk_pose = False
        self._journey_serve_t = -1e9
        self._journey_lin_seen = False
        self._journey_blind_warned = False
        self._journey_fail_warned = False
        self._journey_done = False
        self._base_vel = None
        self._base_ang = None
        self.registered = []
        self.packet_positions = None
        self.packet_ok = True
        self.step_calls = []
        self._gate_results = []

    @property
    def _home_idx(self):
        return 0

    def _sector(self, _pos):
        return "front"

    def _standoff_point(self, pos):
        return np.asarray(pos, dtype=float).copy()

    def _staged_target_idx(self, _sector, _pos):
        return 0

    def _target_safe(self, pos):
        # Mirrors the real envelope's radius rule (07-21: the permissive fake
        # masked the journey never-assigns-far-cubes CRITICAL).
        p = np.asarray(pos, dtype=float)
        return bool(np.all(np.isfinite(p))) and float(np.hypot(p[0], p[1])) <= 0.45

    def _sector_left(self, _pos, _sector):
        return False

    def _bearing_deg(self, pos):
        return float(np.degrees(np.arctan2(pos[1], pos[0])))

    def _dyn_reachable(self, _pos, _arm, _hold):
        return True

    def _gate_eval(self, _pos):
        return self._gate_results.pop(0)

    def _wrap(self, angle):
        return angle

    def _new_track(self, arm, cur, target, now, replant=False):
        return TrackMotion(arm=arm, cur=cur, target=target,
                           seq=[target], hop_t0=now)

    def _step_toward(self, tm, _jpos, _now):
        # 07-21: real per-arm shape (staging.staged_joint_payload) — the old
        # legacy 14-joint fake meant the per-arm merge semantics shipped on
        # 07-20 were never actually exercised by the suite.
        self.step_calls.append(tm.arm)
        return {tm.arm: {"joint_pos": [0.0] * 7, "rate": STAGE_RATE}}, "moving"

    def _staged_joint_payload(self, arm, _posture, _frozen=None):
        return {arm: {"joint_pos": [0.0] * 7, "rate": STAGE_RATE}}

    def _track_posture(self, _arm, _idx):
        return np.zeros(7)

    def _live_retrack(self, *_args):
        return None

    def _gentle_approach(self, _stream, target, _actual):
        target = np.asarray(target, dtype=float)
        return target.copy(), target.copy()

    def _register_hold(self, arm, cube, target, sector, bias, now,
                       restream=None):
        self.registered.append((arm, cube.key))
        self._held[arm] = HoldState(
            key=cube.key, target=np.asarray(target), bias=bias,
            sector=sector, sector_idx=0, grasp_t0=now,
            grasp_move_t=now,
            restream=(None if restream is None else
                      np.asarray(restream, dtype=float).copy()))
        cube.held_by = arm

    def _arm_packet(self, reach_pos=None, reach_positions=None):
        self.packet_positions = dict(reach_positions or {})
        return ({"positions": self.packet_positions}
                if self.packet_ok else None)

    def _grip_base_fk(self, _arm, _jpos):
        return None

    @staticmethod
    def _ee_alive(telemetry):
        side = (telemetry.get("ee") or {}).get("left") or {}
        return side.get("ee_alive")

    def _queue_ee_action(self, action):
        return arbitration.enqueue_ee_action(self, action)

    def _grasp_dropped(self, *_args):
        return False

    def _maybe_send_grab(self, *_args):
        return False

    def _enter_carried(self, h, arm, actual):
        h.grasp = "carried"
        h.restream = np.asarray(actual, dtype=float)

    def _cube_servable(self, *_args):
        return False, None

    def _arm_near_front(self, _arm, _jpos):
        return True

    def _home_glide(self, arm):
        return self.rest_ee[arm].tolist()


class IndependentArmTests(unittest.TestCase):
    NOW = 10.0

    def test_full_tick_latches_and_finishes_both_reachable_sides(self):
        left = FakeCube("left_cube", [0.25, 0.12, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.25, -0.12, -0.1], self.NOW)
        op = FakeOperator([left, right])
        telemetry = {"ee": {
            "left": {"pos_actual": left.pos.copy()},
            "right": {"pos_actual": right.pos.copy()},
        }}

        vx, wz, _ = independent.tick_independent(
            op, telemetry, None, self.NOW)

        self.assertEqual((vx, wz), (0.0, 0.0))
        self.assertCountEqual(op.registered,
                              [("left", left.key), ("right", right.key)])
        self.assertEqual(op.phase, Phase.HOLD)

    def test_direct_task_waits_for_finite_ee_seed_without_releasing_peer(self):
        left = FakeCube("left_cube", [0.25, 0.12, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.25, -0.12, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._ee_act["left"] = None
        op._arm_tasks = {
            "left": ArmTask("left", left.key,
                            phase=ArmTaskPhase.WAIT_STOP),
            "right": ArmTask("right", right.key,
                             phase=ArmTaskPhase.WAIT_STOP),
        }

        independent.prepare_tasks(op, self.NOW)

        self.assertEqual(op._arm_tasks["left"].phase,
                         ArmTaskPhase.WAIT_STOP)
        self.assertEqual(op._arm_tasks["right"].phase,
                         ArmTaskPhase.REACHING)
        self.assertIsNone(op._arm_tasks["left"].stream)
        self.assertIsNotNone(op._arm_tasks["right"].stream)

    def test_both_sides_are_assigned_in_the_same_tick(self):
        left = FakeCube("left_cube", [0.38, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.38, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])

        independent.assign_tasks(op, self.NOW)

        self.assertEqual(op._arm_tasks["left"].key, "left_cube")
        self.assertEqual(op._arm_tasks["right"].key, "right_cube")
        self.assertEqual(left.assigned_to, "left")
        self.assertEqual(right.assigned_to, "right")

    def test_gate_debounce_counts_only_fresh_samples_per_arm(self):
        cube = FakeCube("left_cube", [0.38, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op.gate = SimpleNamespace(SCORE_THRESHOLD=0.5)
        task = ArmTask("left", cube.key)
        op._gate_results = [(1.0, "left", True)] + [
            (1.0, "left", False) for _ in range(4)]

        self.assertFalse(independent.reachability_sample(op, task, cube.pos))
        for _ in range(4):
            self.assertFalse(independent.reachability_sample(op, task, cube.pos))
        self.assertEqual(task.reach_hits, 1)

        op._gate_results = [(1.0, "left", True) for _ in range(4)]
        for _ in range(3):
            self.assertFalse(independent.reachability_sample(op, task, cube.pos))
        self.assertTrue(independent.reachability_sample(op, task, cube.pos))

        peer = ArmTask("right", "peer")
        self.assertEqual(peer.reach_hits, 0)

    def test_learned_gate_budget_alternates_one_fresh_arm_per_tick(self):
        left = FakeCube("left_cube", [0.38, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.38, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op.gate = SimpleNamespace(SCORE_THRESHOLD=0.5)
        op._arm_tasks = {
            "left": ArmTask("left", left.key),
            "right": ArmTask("right", right.key),
        }
        op._gate_results = [
            (1.0, "left", True), (1.0, "right", True)]

        independent.prepare_tasks(op, self.NOW)
        self.assertEqual(op._arm_tasks["left"].reach_hits, 1)
        self.assertEqual(op._arm_tasks["right"].reach_hits, 0)
        self.assertEqual(len(op._gate_results), 1)

        independent.prepare_tasks(op, self.NOW + DT)
        self.assertEqual(op._arm_tasks["left"].reach_hits, 1)
        self.assertEqual(op._arm_tasks["right"].reach_hits, 1)
        self.assertEqual(op._gate_results, [])

    def test_both_cartesian_tasks_finish_without_a_primary(self):
        left = FakeCube("left_cube", [0.25, 0.12, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.25, -0.12, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key,
                            phase=ArmTaskPhase.REACHING,
                            target=left.pos.copy(), port=5555),
            "right": ArmTask("right", right.key,
                             phase=ArmTaskPhase.REACHING,
                             target=right.pos.copy(), port=5555),
        }
        telemetry = {"ee": {
            "left": {"pos_actual": left.pos.copy()},
            "right": {"pos_actual": right.pos.copy()},
        }}

        independent.tick_task_motion(op, telemetry, None, self.NOW)

        self.assertCountEqual(op.registered,
                              [("left", left.key), ("right", right.key)])
        self.assertIsNone(op._arm_tasks["left"])
        self.assertIsNone(op._arm_tasks["right"])
        self.assertEqual(set(op._held), {"left", "right"})

    def test_cancel_is_isolated_to_one_arm(self):
        left = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key),
            "right": ArmTask("right", right.key),
        }
        left.assigned_to = "left"
        right.assigned_to = "right"

        independent.cancel_task(op, "left", "synthetic failure")

        self.assertIsNone(op._arm_tasks["left"])
        self.assertIsNotNone(op._arm_tasks["right"])
        self.assertIsNone(left.assigned_to)
        self.assertEqual(right.assigned_to, "right")

    def test_visual_servo_state_is_not_shared_between_arms(self):
        left = ArmTask("left", "left_cube")
        right = ArmTask("right", "right_cube")

        left.servo.bias = np.array([0.01, 0.0, 0.0])
        left.servo.warned.add("synthetic")
        left.servo.accepted = 2

        self.assertIsNone(right.servo.bias)
        self.assertEqual(right.servo.warned, set())
        self.assertEqual(right.servo.accepted, 0)

    def test_joint_token_advances_only_one_task_and_pauses_peer_timer(self):
        left = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key, phase=ArmTaskPhase.STAGING,
                            track=TrackMotion("left", 0, 1, [1])),
            "right": ArmTask("right", right.key, phase=ArmTaskPhase.STAGING,
                             track=TrackMotion("right", 0, 1, [1])),
        }

        with parallel_staircases_certified():
            payload = independent.tick_task_motion(op, {"ee": {}}, None, self.NOW)

        # 07-21 PHASE 2: BOTH staged arms step in the same tick and their
        # per-arm joint slots merge into one packet (the exclusive token is
        # gone; the dual-staircase clearance audit replaced it).
        self.assertEqual(sorted(op.step_calls), ["left", "right"])
        self.assertIsNotNone(payload)
        self.assertEqual(op._arm_tasks["left"].reach_elapsed, 0.0)
        self.assertEqual(op._arm_tasks["right"].reach_elapsed, 0.0)

    def test_detection_loss_never_cancels_a_mid_staging_task(self):
        cube = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        cube.last_seen = self.NOW - DET_LOST_RESCAN_S - 0.1
        op = FakeOperator([cube])
        task = ArmTask(
            "left", cube.key, phase=ArmTaskPhase.STAGING,
            target=cube.pos.copy(),
            track=TrackMotion("left", 0, 1, [1]))
        op._arm_tasks["left"] = task
        cube.assigned_to = "left"

        independent.prepare_tasks(op, self.NOW)

        self.assertIs(op._arm_tasks["left"], task)
        self.assertEqual(cube.assigned_to, "left")

    def test_cartesian_ready_arm_runs_before_peer_starts_staging(self):
        left = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key, phase=ArmTaskPhase.REACHING,
                            target=left.pos.copy(), port=5555),
            "right": ArmTask("right", right.key, phase=ArmTaskPhase.STAGING,
                             target=right.pos.copy(), port=5555,
                             track=TrackMotion("right", 0, 1, [1])),
        }
        telemetry = {"ee": {
            "left": {"pos_actual": np.array([0.2, 0.2, -0.1])},
            "right": {"pos_actual": np.array([0.2, -0.2, -0.1])},
        }}

        independent.tick_task_motion(op, telemetry, None, self.NOW)

        # 07-21 PHASE 2: the peer's staircase steps in the SAME tick as this
        # arm's Cartesian reach — one mixed packet, no serialization.
        self.assertEqual(op.step_calls, ["right"])
        self.assertEqual(op._arm_tasks["left"].reach_elapsed, DT)
        self.assertEqual(op._arm_tasks["right"].reach_elapsed, 0.0)
        self.assertIn("left", op.packet_positions)

    def test_completed_staging_arm_reaches_before_peer_gets_joint_token(self):
        left = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key, phase=ArmTaskPhase.STAGING,
                            target=left.pos.copy(), sector_idx=1, port=5555,
                            track=TrackMotion("left", 0, 1, [1])),
            "right": ArmTask("right", right.key, phase=ArmTaskPhase.STAGING,
                             target=right.pos.copy(), sector_idx=1, port=5555,
                             track=TrackMotion("right", 0, 1, [1])),
        }

        def finish_stage(tm, _jpos, _now):
            op.step_calls.append(tm.arm)
            tm.seq.clear()
            return {"joint_pos": [0.0] * 14}, "done"

        op._step_toward = finish_stage
        telemetry = {"ee": {
            "left": {"pos_actual": np.array([0.2, 0.2, -0.1])},
            "right": {"pos_actual": np.array([0.2, -0.2, -0.1])},
        }}

        with parallel_staircases_certified():
            independent.tick_task_motion(op, telemetry, None, self.NOW)

        # 07-21 PHASE 2: both staircases finish and both hand off to their
        # Cartesian reach in the same tick.
        self.assertEqual(sorted(op.step_calls), ["left", "right"])
        for arm in ("left", "right"):
            self.assertEqual(op._arm_tasks[arm].phase, ArmTaskPhase.REACHING)
            self.assertEqual(op._arm_tasks[arm].reach_elapsed, DT)
            self.assertIsNone(op._arm_tasks[arm].track)

    def test_one_arrival_does_not_reset_peer_reach(self):
        left = FakeCube("left_cube", [0.25, 0.12, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key, phase=ArmTaskPhase.REACHING,
                            target=left.pos.copy(), port=5555),
            "right": ArmTask("right", right.key, phase=ArmTaskPhase.REACHING,
                             target=right.pos.copy(), port=5555),
        }
        telemetry = {"ee": {
            "left": {"pos_actual": left.pos.copy()},
            "right": {"pos_actual": np.array([0.2, -0.2, -0.1])},
        }}

        independent.tick_task_motion(op, telemetry, None, self.NOW)

        self.assertIsNone(op._arm_tasks["left"])
        self.assertIsNotNone(op._arm_tasks["right"])
        self.assertEqual(op._arm_tasks["right"].reach_elapsed, DT)
        self.assertIn("right", op.packet_positions)

    def test_timeout_hands_current_ramp_point_to_hold(self):
        cube = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        task = ArmTask("left", cube.key, phase=ArmTaskPhase.REACHING,
                       target=cube.pos.copy(), port=5555,
                       stream=np.array([0.2, 0.2, -0.1]),
                       reach_elapsed=REACH_TIMEOUT_S)
        op._arm_tasks["left"] = task

        def slow_step(stream, target, _actual):
            delta = np.asarray(target) - np.asarray(stream)
            nxt = np.asarray(stream) + delta / np.linalg.norm(delta) * 0.004
            return nxt, nxt

        op._gentle_approach = slow_step
        telemetry = {"ee": {
            "left": {"pos_actual": np.array([0.2, 0.2, -0.1])}}}

        independent.tick_task_motion(op, telemetry, None, self.NOW)

        self.assertIsNone(op._arm_tasks["left"])
        self.assertIsNotNone(op._held["left"].restream)
        self.assertLess(float(np.linalg.norm(
            op._held["left"].restream - cube.pos)), 0.2)
        self.assertGreater(float(np.linalg.norm(
            op._held["left"].restream - cube.pos)), 0.0)

    def test_packet_builder_merges_two_cartesian_slots(self):
        op = FakeOperator([])
        packet = arbitration.build_arm_packet(
            op, reach_positions={
                "left": np.array([0.3, 0.2, -0.1]),
                "right": np.array([0.3, -0.2, -0.1]),
            })
        self.assertEqual(packet["ee_pos"], [
            [0.3, 0.2, -0.1], [0.3, -0.2, -0.1]])

    def test_packet_builder_combines_hold_and_peer_reach(self):
        cube = FakeCube("left_cube", [0.3, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op._held["left"] = HoldState(
            key=cube.key, target=cube.pos.copy(), bias=None,
            sector="front", sector_idx=0)

        packet = arbitration.build_arm_packet(
            op, reach_positions={
                "right": np.array([0.3, -0.2, -0.1])})

        self.assertEqual(packet["ee_pos"], [
            [0.3, 0.2, -0.1], [0.3, -0.2, -0.1]])

    def test_arbitration_never_mixes_cartesian_with_joint_task(self):
        cube = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op._arm_tasks["left"] = ArmTask(
            "left", cube.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("left", 0, 1, [1]))
        op._ind_joint_owners = {"left"}
        packet = {"arm_targets": {
            "ee_pos": [[0.3, 0.2, -0.1], [0.3, -0.2, -0.1]],
            "ee_quat": [[1.0, 0.0, 0.0, 0.0]] * 2,
        }}

        arbitration.arbitrate(op, packet, None, None)

        self.assertNotIn("arm_targets", packet)

    def test_pending_staircase_does_not_suppress_granted_cartesian_packet(self):
        cube = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op._arm_tasks["left"] = ArmTask(
            "left", "left_reaching", phase=ArmTaskPhase.REACHING,
            target=np.array([0.3, 0.2, -0.1]))
        op._arm_tasks["right"] = ArmTask(
            "right", cube.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("right", 0, 1, [1]))
        packet = {"arm_targets": {
            "ee_pos": [[0.3, 0.2, -0.1], [0.3, -0.2, -0.1]],
            "ee_quat": [[1.0, 0.0, 0.0, 0.0]] * 2,
        }}

        arbitration.arbitrate(op, packet, None, None)

        self.assertIn("arm_targets", packet)

    def test_unpublishable_combined_packet_commits_no_reach_state(self):
        cube = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op.packet_ok = False
        task = ArmTask(
            "left", cube.key, phase=ArmTaskPhase.REACHING,
            target=cube.pos.copy(), port=5555,
            stream=np.array([0.2, 0.2, -0.1]))
        op._arm_tasks["left"] = task
        telemetry = {"ee": {
            "left": {"pos_actual": cube.pos.copy()}}}

        packet = independent.tick_task_motion(
            op, telemetry, None, self.NOW)

        self.assertIsNone(packet)
        np.testing.assert_allclose(task.stream, [0.2, 0.2, -0.1])
        self.assertEqual(task.reach_elapsed, 0.0)
        self.assertEqual(op.registered, [])
        self.assertIs(op._arm_tasks["left"], task)

    def test_staged_carry_can_start_while_peer_only_waits_for_base(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.38, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.grasp = True
        op._held["left"] = HoldState(
            key=held_cube.key, target=held_cube.pos.copy(), bias=None,
            sector="side", sector_idx=1, grasp="grasped")
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.ASSIGNED)

        arms.held_update(op, self.NOW, {"ee": {}}, None)

        self.assertIsNotNone(op._held["left"].track)
        self.assertIsNotNone(op._arm_tasks["right"])

    def test_ready_hold_carry_precedes_unstarted_peer_staircase(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.grasp = True
        op._held["left"] = HoldState(
            key=held_cube.key, target=held_cube.pos.copy(), bias=None,
            sector="side", sector_idx=1, grasp="grasped")
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("right", 0, 1, [1]))

        arms.held_update(op, self.NOW, {"ee": {}}, None)

        self.assertIsNotNone(op._held["left"].track)
        self.assertEqual(op._ind_joint_owners, set())
        self.assertIsNotNone(op._arm_tasks["right"].track)

    def test_held_cartesian_ramp_advances_during_active_task_joint_wire(self):
        # 07-20 per-arm protocol: a peer's joint staircase no longer freezes a
        # holding arm's Cartesian ramp — its slot rides the same mixed packet,
        # so the park ramp keeps advancing toward the target.
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.grasp = True
        start = np.array([0.25, 0.2, -0.1])
        target = np.array([0.35, 0.2, -0.1])
        op._held["left"] = HoldState(
            key=held_cube.key, target=target.copy(),
            bias=None, sector="front", sector_idx=0,
            grasp="carried", restream=start.copy())
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("right", 0, 1, [1]))
        op._ind_joint_owners = {"right"}

        arms.held_update(op, self.NOW, {"ee": {}}, None)

        # The fake _gentle_approach teleports the ramp to its target in one
        # call, so an ADVANCING ramp completes (restream cleared). The old
        # frozen contract left restream exactly at `start`.
        self.assertIsNone(op._held["left"].restream)

    def test_held_cartesian_ramp_freezes_when_peer_slot_is_unpublishable(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.grasp = True
        start = np.array([0.25, 0.2, -0.1])
        op._held["left"] = HoldState(
            key=held_cube.key, target=np.array([0.35, 0.2, -0.1]),
            bias=None, sector="front", sector_idx=0,
            grasp="carried", restream=start.copy())
        held_cube.held_by = "left"
        op._ee_act["right"] = None

        op._ind_cartesian_grant = independent.cartesian_packet_ready(op)
        arms.held_update(op, self.NOW, {"ee": {}}, None)

        self.assertFalse(op._ind_cartesian_grant)
        np.testing.assert_allclose(op._held["left"].restream, start)

    def test_final_hold_cartesian_point_precedes_new_peer_staircase(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.grasp = True
        op._held["left"] = HoldState(
            key=held_cube.key, target=np.array([0.31, 0.2, -0.1]),
            bias=None, sector="front", sector_idx=0,
            grasp="carried", restream=np.array([0.30, 0.2, -0.1]))
        held_cube.held_by = "left"
        op._ind_hold_cartesian_active = \
            independent.hold_cartesian_motion_active(op)

        arms.held_update(op, self.NOW, {"ee": {}}, None)
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("right", 0, 1, [1]))
        packet = independent.tick_task_motion(
            op, {"ee": {}}, None, self.NOW)

        # 07-21 PHASE 2: the hold's final Cartesian point and the peer's new
        # staircase now ride the SAME packet — no serialization, and the hold
        # ramp still completes (restream cleared).
        self.assertIsNone(op._held["left"].restream)
        self.assertIsNotNone(packet)
        self.assertEqual(op._ind_joint_owners, {"right"})
        self.assertEqual(op.step_calls, ["right"])
        self.assertIsNotNone(op._arm_tasks["right"].track)

    def test_new_peer_staircase_waits_one_tick_after_hold_live_retrack(self):
        held_cube = FakeCube("held", [0.35, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.side_home = False
        op._held["left"] = HoldState(
            key=held_cube.key, target=np.array([0.30, 0.2, -0.1]),
            bias=None, sector="front", sector_idx=0, grasp="failed")
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.WAIT_STOP)
        peer_cube.assigned_to = "right"
        op._staged_target_idx = lambda _sector, _pos: 1

        arms.held_update(op, self.NOW, {"ee": {}}, None)
        independent.prepare_tasks(op, self.NOW)
        first_packet = independent.tick_task_motion(
            op, {"ee": {}}, None, self.NOW)

        np.testing.assert_allclose(
            op._held["left"].target, held_cube.pos)
        self.assertEqual(op._arm_tasks["right"].phase,
                         ArmTaskPhase.STAGING)
        self.assertIsNotNone(first_packet)
        self.assertEqual(op.step_calls, [])
        self.assertEqual(op._ind_joint_owners, set())

        second_packet = independent.tick_task_motion(
            op, {"ee": {}}, None, self.NOW + DT)

        # per-arm shape: the staircase side owns its own joint slot
        self.assertIn("joint_pos", second_packet["right"])
        self.assertEqual(op.step_calls, ["right"])

    def test_dead_ee_chain_does_not_block_peer_navigation(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.38, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.grasp = True
        op._ee_chain_dead = True
        op._held["left"] = HoldState(
            key=held_cube.key, target=held_cube.pos.copy(), bias=None,
            sector="front", sector_idx=0, grasp=None)
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.ASSIGNED)

        vx, _ = independent.navigation_command(op, self.NOW)

        self.assertEqual(vx, CRUISE_VX)

    def test_retracted_lost_hold_does_not_block_peer_navigation(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.38, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.grasp = True
        op._held["left"] = HoldState(
            key=held_cube.key, target=held_cube.pos.copy(), bias=None,
            sector="front", sector_idx=op._home_idx,
            home=True, grasp=None)
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.ASSIGNED)

        vx, _ = independent.navigation_command(op, self.NOW)

        self.assertEqual(vx, CRUISE_VX)

    def test_reappearing_hold_waits_for_moving_base_to_stop(self):
        held_cube = FakeCube("held", [0.3, 0.2, -0.1], self.NOW)
        peer_cube = FakeCube("peer", [0.38, -0.2, -0.1], self.NOW)
        op = FakeOperator([held_cube, peer_cube])
        op.independent_arms = True
        op.dynamic_track = True
        op.grasp = True
        op.side_home = False
        op._vx = CRUISE_VX
        op._held["left"] = HoldState(
            key=held_cube.key, target=held_cube.pos.copy(), bias=None,
            sector="front", sector_idx=op._home_idx,
            home=True, grasp=None)
        held_cube.held_by = "left"
        op._arm_tasks["right"] = ArmTask(
            "right", peer_cube.key, phase=ArmTaskPhase.ASSIGNED)

        arms.held_update(op, self.NOW, {
            "ee": {"left": {"pos_actual": op.rest_ee["left"].copy()}}}, None)
        vx, _ = independent.navigation_command(op, self.NOW)

        self.assertTrue(op._held["left"].home)
        self.assertTrue(op._held["left"].resume_pending)
        self.assertEqual(vx, 0.0)

        op._vx = 0.0
        arms.held_update(op, self.NOW + DT, {
            "ee": {"left": {"pos_actual": op.rest_ee["left"].copy()}}}, None)

        self.assertFalse(op._held["left"].home)
        self.assertFalse(op._held["left"].resume_pending)


class GripperFailClosedTests(unittest.TestCase):
    """A perception or IPC verdict may stop motion, but may never release.

    These tests deliberately exercise the operator functions directly.  They
    do not construct an environment, open an IPC socket, or touch servo
    hardware.
    """
    NOW = 10.0

    def _sent_hold(self, reported, *, expected_id=41, reported_id=41):
        cube = FakeCube("held", [0.30, 0.18, -0.10], self.NOW)
        cube.held_by = "left"
        op = FakeOperator([cube])
        op.dynamic_track = True
        op.grasp = True
        hold = HoldState(
            key=cube.key,
            target=cube.pos.copy(),
            bias=None,
            sector="front",
            sector_idx=0,
            grasp=GraspState.SENT,
            grasp_t0=self.NOW - 1.0,
            grasp_base=None,
            grasp_action_id=expected_id,
        )
        op._held["left"] = hold
        telemetry = {"ee": {"left": {
            "pos_actual": cube.pos.copy(),
            "ee_alive": True,
            "grasp_detected": reported,
            "grasp_action_id": reported_id,
        }}}
        return op, cube, hold, telemetry

    def test_indeterminate_result_latches_without_open(self):
        # None verdict = ambiguity (lossy status channel) — still fails closed.
        op, cube, hold, telemetry = self._sent_hold(None)

        arms.held_update(op, self.NOW, telemetry, None)

        self.assertEqual(hold.grasp, GraspState.UNCERTAIN)
        self.assertTrue(op._grasp_safety_latched)
        self.assertEqual(cube.held_by, "left")
        self.assertIs(op._held["left"], hold)
        self.assertNotIn(
            "hand_open", [a.get("command") for a in op._ee_queue])

    def test_certified_empty_pinch_reopens_and_retries_without_limit(self):
        # 07-26 deadlock audit: an id-matched grasp_detected=False from a live
        # chain is the service's CERTIFIED empty pinch — the 07-21 contract
        # (GRASP_MAX_TRIES removed: retry WITHOUT LIMIT) applies, the mission
        # must NOT latch the intervention freeze. 80b997c had revoked this.
        op, cube, hold, telemetry = self._sent_hold(False)

        arms.held_update(op, self.NOW, telemetry, None)

        self.assertIsNone(hold.grasp)              # re-armed, not UNCERTAIN
        self.assertFalse(op._grasp_safety_latched)
        self.assertEqual(cube.held_by, "left")     # hold retained in place
        self.assertIs(op._held["left"], hold)
        self.assertIn(
            "hand_open", [a.get("command") for a in op._ee_queue])
        self.assertEqual(hold.grasp_tries, 1)
        self.assertIsNone(hold.grasp_action_id)
        self.assertEqual(hold.grasp_move_t, self.NOW)  # settle timer restarted

    def test_empty_pinch_with_dead_chain_still_latches(self):
        # The retry carve-out demands a LIVE chain: False + ee_alive=False is
        # ambiguous (chain died mid-burst) and must keep failing closed.
        op, cube, hold, telemetry = self._sent_hold(False)
        telemetry["ee"]["left"]["ee_alive"] = False

        arms.held_update(op, self.NOW, telemetry, None)

        self.assertEqual(hold.grasp, GraspState.UNCERTAIN)
        self.assertTrue(op._grasp_safety_latched)
        self.assertNotIn(
            "hand_open", [a.get("command") for a in op._ee_queue])

    def test_result_for_an_old_action_id_is_ignored(self):
        op, _cube, hold, telemetry = self._sent_hold(
            False, expected_id=42, reported_id=41)

        arms.held_update(op, self.NOW, telemetry, None)

        self.assertEqual(hold.grasp, GraspState.SENT)
        self.assertFalse(op._grasp_safety_latched)
        self.assertEqual(op._ee_queue, [])

    def test_persistent_visual_drop_only_latches_and_keeps_ownership(self):
        cube = FakeCube("held", [0.30, 0.18, -0.10], self.NOW)
        cube.held_by = "left"
        op = FakeOperator([cube])
        hold = HoldState(
            key=cube.key,
            target=cube.pos.copy(),
            bias=None,
            sector="front",
            sector_idx=0,
            grasp=GraspState.CARRIED,
        )
        op._held["left"] = hold
        telemetry = {"ee": {"left": {
            "pos_actual": [0.0, 0.55, 0.0],
        }}}

        self.assertFalse(arms.grasp_dropped(
            op, hold, cube, "left", telemetry, self.NOW))
        self.assertTrue(arms.grasp_dropped(
            op, hold, cube, "left", telemetry,
            self.NOW + GRASP_DROP_PERSIST_S + 0.1))

        self.assertEqual(hold.grasp, GraspState.UNCERTAIN)
        self.assertTrue(op._grasp_safety_latched)
        self.assertEqual(cube.held_by, "left")
        self.assertIs(op._held["left"], hold)
        self.assertNotIn(
            "hand_open", [a.get("command") for a in op._ee_queue])

    def test_grab_repeat_burst_reuses_one_action_id(self):
        cube = FakeCube("held", [0.30, 0.18, -0.10], self.NOW)
        op = FakeOperator([cube])
        op.grasp = True
        hold = HoldState(
            key=cube.key,
            target=cube.pos.copy(),
            bias=None,
            sector="front",
            sector_idx=0,
            grasp_move_t=self.NOW - GRASP_SETTLE_S - 0.1,
        )
        op._held["left"] = hold
        cube.held_by = "left"
        telemetry = {"ee": {"left": {
            "pos_actual": cube.pos.copy(),
            "ee_alive": True,
            "grasp_detected": None,
        }}}

        self.assertTrue(arms.maybe_send_grab(
            op, "left", hold, cube, telemetry, self.NOW))
        burst_ids = []
        for _ in range(3):
            packet = {}
            arbitration.attach_ee_action(op, packet, repeat_ticks=3)
            burst_ids.append(packet["ee_action"]["action_id"])

        self.assertIsNotNone(hold.grasp_action_id)
        self.assertEqual(burst_ids, [hold.grasp_action_id] * 3)
        op._ee_queue.append({"side": "left", "command": "hand_hold"})
        next_packet = {}
        arbitration.attach_ee_action(op, next_packet, repeat_ticks=3)
        next_id = next_packet["ee_action"]["action_id"]
        self.assertNotEqual(next_id, hold.grasp_action_id)

    def test_startup_grasp_telemetry_blocks_zero_and_latches(self):
        op = FakeOperator([])
        telemetry = {"ee": {
            "left": {"grasp_detected": True, "ee_alive": True},
            "right": {"grasp_detected": None},
        }}

        self.assertTrue(arms.startup_grasp_guard(op, telemetry))

        self.assertTrue(op._grasp_safety_latched)
        self.assertEqual(op._grasp_safety_arm, "left")
        self.assertEqual(op._ee_queue, [])


class ParallelStaircaseTests(unittest.TestCase):
    """07-21 PHASE 2: two joint staircases may walk at once. Safety comes from
    the dual-staircase clearance audit (arm-vs-trunk is peer-independent with a
    fixed base; arm-vs-arm >= 239.9 mm by the mid-sagittal certificate), not
    from serialization."""
    NOW = 10.0

    def setUp(self):
        self._cm = parallel_staircases_certified()
        self._cm.__enter__()

    def tearDown(self):
        self._cm.__exit__(None, None, None)

    def _two_staged(self):
        left = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key, phase=ArmTaskPhase.STAGING,
                            target=left.pos.copy(),
                            track=TrackMotion("left", 0, 1, [1])),
            "right": ArmTask("right", right.key, phase=ArmTaskPhase.STAGING,
                             target=right.pos.copy(),
                             track=TrackMotion("right", 0, 1, [1])),
        }
        return op, left, right

    def test_both_task_staircases_step_in_one_tick(self):
        op, _l, _r = self._two_staged()
        per_arm = {}

        def step(tm, _jpos, _now):
            op.step_calls.append(tm.arm)
            return {tm.arm: {"joint_pos": [0.1] * 7, "rate": STAGE_RATE}}, "moving"
        op._step_toward = step

        payload = independent.tick_task_motion(op, {"ee": {}}, None, self.NOW)
        per_arm = payload if payload else {}

        self.assertEqual(sorted(op.step_calls), ["left", "right"])
        self.assertIn("left", per_arm)
        self.assertIn("right", per_arm)
        self.assertIn("joint_pos", per_arm["left"])
        self.assertIn("joint_pos", per_arm["right"])
        self.assertEqual(op._ind_joint_owners, {"left", "right"})

    def test_double_timeout_never_drops_a_peer_slot(self):
        # Both hops time out on the same tick: the first cancel must not
        # short-circuit the peer's payload (the 07-21 audit's merge-and-
        # continue requirement).
        op, _l, _r = self._two_staged()

        def step(tm, _jpos, _now):
            op.step_calls.append(tm.arm)
            tm.timed_out = True
            tm.seq.clear()
            return {tm.arm: {"joint_pos": [0.0] * 7, "rate": STAGE_RATE}}, "moving"
        op._step_toward = step

        payload = independent.tick_task_motion(op, {"ee": {}}, None, self.NOW)

        self.assertEqual(sorted(op.step_calls), ["left", "right"])
        self.assertIn("left", payload)
        self.assertIn("right", payload)
        self.assertIsNone(op._arm_tasks["left"])     # both cancelled
        self.assertIsNone(op._arm_tasks["right"])

    def test_two_held_tracks_merge_and_neither_gets_a_cartesian_slot(self):
        # The PRE-EXISTING leak the audit found: with two held tracks the
        # waiting arm used to receive a Cartesian home-glide mid-staircase.
        left_c = FakeCube("lc", [0.4, 0.2, -0.1], self.NOW)
        right_c = FakeCube("rc", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left_c, right_c])
        op.dynamic_track = True
        op.grasp = True
        for arm, cube in (("left", left_c), ("right", right_c)):
            op._held[arm] = HoldState(
                key=cube.key, target=cube.pos.copy(), bias=None,
                sector="rear", sector_idx=2, grasp="grasped",
                track=TrackMotion(arm, 2, 0, [0], frozen=[0.0] * 7))
            cube.held_by = arm

        def step(tm, _jpos, _now):
            op.step_calls.append(tm.arm)
            return {tm.arm: {"joint_pos": [0.2] * 7, "rate": STAGE_RATE}}, "moving"
        op._step_toward = step

        ret = arms.held_update(op, self.NOW, {"ee": {
            "left": {"pos_actual": [0.2, 0.2, -0.1]},
            "right": {"pos_actual": [0.2, -0.2, -0.1]}}}, None)

        self.assertEqual(sorted(op.step_calls), ["left", "right"])
        self.assertIn("left", ret)
        self.assertIn("right", ret)
        packet = {"arm_targets": {"ee_pos": [[0.0, 0.55, 0.0],
                                             [0.0, -0.55, 0.0]],
                                  "ee_quat": [[1.0, 0.0, 0.0, 0.0]] * 2}}
        arbitration.arbitrate(op, packet, ret, None)
        at = packet["arm_targets"]
        for side in ("left", "right"):
            self.assertIn("joint_pos", at[side])   # no Cartesian leak

    def test_staircase_arm_is_pinned_not_home_glided(self):
        # A task-staircasing arm must never advance a home-glide ramp that its
        # joint slot will overwrite (phantom stream advance).
        cube = FakeCube("c", [0.4, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op._arm_tasks["left"] = ArmTask(
            "left", cube.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("left", 0, 1, [1], frozen=[0.0] * 7))
        pkt = arbitration.build_arm_packet(op)
        np.testing.assert_allclose(pkt["ee_pos"][0], op._ee_act["left"])
        self.assertIsNone(op._home_stream["left"])   # ramp never seeded

    def test_mixed_packet_joint_side_keeps_joint_cartesian_side_merges(self):
        left_c = FakeCube("lc", [0.3, 0.2, -0.1], self.NOW)
        right_c = FakeCube("rc", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left_c, right_c])
        op._arm_tasks["right"] = ArmTask(
            "right", right_c.key, phase=ArmTaskPhase.STAGING,
            track=TrackMotion("right", 0, 1, [1], frozen=[0.0] * 7))
        op._ind_joint_owners = {"right"}
        ret = {"right": {"joint_pos": [0.0] * 7, "rate": STAGE_RATE}}
        packet = {"arm_targets": {"ee_pos": [[0.3, 0.2, -0.1], None],
                                  "ee_quat": [[1.0, 0.0, 0.0, 0.0]] * 2}}
        arbitration.arbitrate(op, packet, ret, None)
        at = packet["arm_targets"]
        self.assertIn("joint_pos", at["right"])
        self.assertIn("ee_pos", at["left"])          # peer keeps Cartesian


class PreconReleaseTests(unittest.TestCase):
    """07-26 review rework: the pre-grab EE-convergence timeout must retreat
    and release through the audited machinery — never re-extend (ping-pong),
    never release in place, never count dead-chain seconds."""
    NOW = 10.0

    def _settled_hold(self):
        cube = FakeCube("c", [0.30, 0.18, -0.10], self.NOW)
        cube.held_by = "left"
        op = FakeOperator([cube])
        op.grasp = True
        op.dynamic_track = True
        h = HoldState(key="c", target=cube.pos.copy(), bias=None,
                      sector="front", sector_idx=0,
                      grasp_move_t=self.NOW - 5.0)
        op._held["left"] = h
        far_act = [0.30, 0.26, -0.10]          # 8 cm off the target
        telemetry = {"ee": {"left": {"pos_actual": far_act,
                                     "ee_alive": True}}}
        return op, cube, h, telemetry

    def test_ee_far_clock_stamps_holds_and_clears(self):
        op, _cube, h, telemetry = self._settled_hold()
        # distance is the only unmet precondition -> stamp, then hold stamp
        self.assertFalse(arms.maybe_send_grab(op, "left", h, op.cubes["c"],
                                              telemetry, self.NOW))
        self.assertEqual(h.ee_far_t0, self.NOW)
        arms.maybe_send_grab(op, "left", h, op.cubes["c"],
                             telemetry, self.NOW + 1.0)
        self.assertEqual(h.ee_far_t0, self.NOW)   # original stamp kept
        # chain outage: why starts with "EE " too — must CLEAR, not retain
        telemetry["ee"]["left"]["ee_alive"] = False
        arms.maybe_send_grab(op, "left", h, op.cubes["c"],
                             telemetry, self.NOW + 2.0)
        self.assertIsNone(h.ee_far_t0)
        # guard path (hold retracting): stale stamp must not survive either
        telemetry["ee"]["left"]["ee_alive"] = True
        op.cubes["c"].see(self.NOW + 3.0)           # keep the sighting fresh
        arms.maybe_send_grab(op, "left", h, op.cubes["c"],
                             telemetry, self.NOW + 3.0)
        self.assertEqual(h.ee_far_t0, self.NOW + 3.0)
        h.home = True
        arms.maybe_send_grab(op, "left", h, op.cubes["c"],
                             telemetry, self.NOW + 4.0)
        self.assertIsNone(h.ee_far_t0)

    def test_direct_timeout_retreats_then_releases_at_rest_only(self):
        op, cube, h, telemetry = self._settled_hold()
        op._ee_act = {"left": cube.pos.copy(),      # arm out at the cube
                      "right": np.array([0.0, -0.55, 0.0])}
        h.ee_far_t0 = self.NOW - GRASP_PRECON_RETRY_S - 1.0
        arms.held_update(op, self.NOW, telemetry, None)
        self.assertTrue(h.precon_release)
        self.assertTrue(h.home, "must retreat, not re-extend")
        self.assertIn("left", op._held, "never release in place")
        self.assertEqual(cube.held_by, "left")
        # cube still fresh & servable next tick: follow must NOT flip home back
        arms.held_update(op, self.NOW + 0.05, telemetry, None)
        self.assertTrue(h.home, "servable follow must not undo the retreat")
        self.assertIn("left", op._held)
        # measured EE back at REST -> now, and only now, release + re-queue
        op._ee_act["left"] = op.rest_ee["left"].copy()
        arms.held_update(op, self.NOW + 0.10, telemetry, None)
        self.assertNotIn("left", op._held)
        self.assertIsNone(cube.held_by)

    def test_staged_timeout_walks_the_retreat_track_to_release(self):
        op, cube, h, telemetry = self._settled_hold()
        op._ee_act = {"left": cube.pos.copy(),
                      "right": np.array([0.0, -0.55, 0.0])}
        h.sector = "rear"
        h.sector_idx = 2                            # staged (home_idx = 0)
        h.ee_far_t0 = self.NOW - GRASP_PRECON_RETRY_S - 1.0
        arms.held_update(op, self.NOW, telemetry, None)
        self.assertTrue(h.precon_release)
        self.assertTrue(h.home)
        self.assertIsNotNone(h.track, "retreat staircase must start")
        self.assertIn("left", op._held)
        # while walking home the follow must not cancel the retreat
        arms.held_update(op, self.NOW + 0.05, telemetry, None)
        self.assertTrue(h.home)
        self.assertIsNotNone(h.track)
        # settled at HOME with the flag forcing not-servable -> release there
        h.track.cur = 0
        h.track.target = 0
        h.track.seq = []
        op._step_toward = lambda _tm, _j, _n: ({}, "done")
        arms.held_update(op, self.NOW + 0.10, telemetry, None)
        self.assertNotIn("left", op._held)
        self.assertIsNone(cube.held_by)


class AssignStallTests(unittest.TestCase):
    """07-27 second review: the ASSIGN_STALL cancel needs a MEMORY. Without
    one, a band cube whose gate crowns the NON-natural arm livelocked: the
    natural arm re-claimed after every 20 s stall while its band priority
    blocked the gate-favored peer forever (reproduced: 90 s, left claims 5x,
    right 0x, cube never serviced, no loud skip)."""
    NOW = 10.0

    def test_band_stall_rotates_to_gate_favored_arm(self):
        cube = FakeCube("band_cube", [0.35, 0.01, -0.1], self.NOW)
        op = FakeOperator([cube])
        op.gate = SimpleNamespace(SCORE_THRESHOLD=0.5)
        independent.assign_tasks(op, self.NOW)
        # band priority: the free natural (left) arm claims first
        self.assertEqual(op._arm_tasks["left"].key, "band_cube")
        self.assertIsNone(op._arm_tasks["right"])
        # the gate persistently crowns RIGHT -> zero progress -> stall cancel
        op._gate_results = [(1.0, "right", True)]
        later = self.NOW + ASSIGN_STALL_S + 1.0
        cube.see(later)
        independent.prepare_tasks(op, later)
        self.assertIsNone(op._arm_tasks["left"])
        self.assertEqual(op._assign_stalls["band_cube"]["left"], 1)
        # next pass: natural priority is void, left steps aside, RIGHT claims
        independent.assign_tasks(op, later)
        self.assertIsNone(op._arm_tasks["left"])
        self.assertEqual(op._arm_tasks["right"].key, "band_cube")
        # the crowned arm makes real progress and the stall record is wiped
        # (base kept moving so the WAIT_STOP block does not fall through to a
        # latch attempt against the long-stale single-sample history)
        op._vx = 0.5
        op._gate_results = [(1.0, "right", True)] * GATE_CONSECUTIVE
        for i in range(GATE_CONSECUTIVE):
            t = later + (i + 1) * DT
            cube.see(t)
            independent.prepare_tasks(op, t)
        self.assertEqual(op._arm_tasks["right"].phase, ArmTaskPhase.WAIT_STOP)
        self.assertNotIn("band_cube", op._assign_stalls)

    def test_every_servable_arm_stalled_n_times_skips_loudly(self):
        # Strictly left-side cube: the peer can never legally serve it, so
        # left keeps retrying (no step-aside to a wrong-side arm) until the
        # per-arm budget is spent, then the cube is skipped LOUDLY.
        cube = FakeCube("far_left", [0.35, 0.2, -0.1], self.NOW)
        op = FakeOperator([cube])
        op.gate = SimpleNamespace(SCORE_THRESHOLD=0.5)
        now = self.NOW
        for _ in range(LATCH_FAIL_SKIP_N):
            independent.assign_tasks(op, now)
            self.assertEqual(op._arm_tasks["left"].key, "far_left")
            now += ASSIGN_STALL_S + 1.0
            cube.see(now)
            op._gate_results = [(0.0, None, True)]
            independent.prepare_tasks(op, now)
            self.assertIsNone(op._arm_tasks["left"])
        self.assertTrue(cube.reached)

    def test_stalled_cube_yields_priority_to_clean_work(self):
        stalled = FakeCube("stalled", [0.30, 0.15, -0.1], self.NOW)
        stalled.first_seen = self.NOW - 5.0     # older: would win the old pick
        clean = FakeCube("clean", [0.30, 0.2, -0.1], self.NOW)
        op = FakeOperator([stalled, clean])
        op._assign_stalls = {"stalled": {"left": 1}}
        independent.assign_tasks(op, self.NOW)
        self.assertEqual(op._arm_tasks["left"].key, "clean")


class WristMirrorTests(unittest.TestCase):
    """07-28 gimbal-lock-free wrist: the right arm is no longer -q_left.

    wrist_2 (index 5) alone keeps its sign, because the redesign gives
    wrist_3_L a 180 deg body quat that wrist_3_R lacks, so the two arms'
    wrist_2 world axes now agree instead of opposing. A regression here is
    SILENT on hardware -- the negated target stays inside wrist_2's +-1.6057
    limit, so the staircase reaches it and reports 'settled' with the gripper
    2*|wrist_2| (64-124 deg) rotated from the audited posture.
    """
    NOW = 10.0

    def test_mirror_is_a_plain_negation_and_is_an_involution(self):
        q = np.arange(1.0, 8.0)                       # 1..7, all distinct
        np.testing.assert_allclose(mirror_arm(q, "left"), q)
        np.testing.assert_allclose(mirror_arm(q, "right"), -q)
        np.testing.assert_allclose(mirror_arm(mirror_arm(q, "right"), "right"), q)

    def test_every_joint_participates_in_the_mirror(self):
        """No joint may silently opt out — that is the 07-28 failure mode.

        Upstream's new wrist turned left_wrist_2's axis to match the right arm's
        without rewiring the motor, which would have made index 5 a +1. It was
        invisible offline (targets stay inside the joint limit, so a staircase
        still reports 'settled') and only surfaced in the hardware gravity-torque
        check. Pin the whole vector so a future model change cannot reintroduce
        it unnoticed; humanoid_joint_monkey_hw re-derives it from the model.
        """
        np.testing.assert_array_equal(ARM_MIRROR_SIGN, np.full(7, -1.0))

    def test_posture_payload_mirrors_with_the_wrist_convention(self):
        op = FakeOperator([])
        op.journey = True
        op._journey_walk_pose = True
        jp = np.zeros(31)
        jp[13:20] = SIDE_HOME_JOINTS
        jp[20:27] = mirror_arm(SIDE_HOME_JOINTS, "right")
        pp = independent.journey_posture_payload(op, jp)
        # Routed through mirror_arm, not a hardcoded -q: the payload has to
        # follow whatever convention the deploy model is actually on.
        np.testing.assert_allclose(pp["right"]["joint_pos"],
                                   mirror_arm(POWERON_JOINTS, "right"))
        np.testing.assert_allclose(pp["left"]["joint_pos"], POWERON_JOINTS)


class StaircaseSerializationTests(unittest.TestCase):
    """07-28: parallel staircases are gated off until the dual-staircase
    clearance audit is re-run on the regenerated deploy model (the 239.9 mm
    mid-sagittal certificate was measured on the OLD wrist; the new one gives
    236.7 mm at the TCP). The CAPABILITY stays in the code behind
    PARALLEL_STAIRCASE_CERTIFIED."""
    NOW = 10.0

    def setUp(self):
        self._cm = parallel_staircases_uncertified()
        self._cm.__enter__()

    def tearDown(self):
        self._cm.__exit__(None, None, None)

    def _two_staged(self):
        left = FakeCube("left_cube", [0.4, 0.2, -0.1], self.NOW)
        right = FakeCube("right_cube", [0.4, -0.2, -0.1], self.NOW)
        op = FakeOperator([left, right])
        op._arm_tasks = {
            "left": ArmTask("left", left.key, phase=ArmTaskPhase.STAGING,
                            track=TrackMotion("left", 0, 1, [1])),
            "right": ArmTask("right", right.key, phase=ArmTaskPhase.STAGING,
                             track=TrackMotion("right", 0, 1, [1])),
        }
        return op

    def test_uncertified_gate_serializes_to_one_arm(self):
        op = self._two_staged()
        independent.tick_task_motion(op, {"ee": {}}, None, self.NOW)
        self.assertEqual(len(op.step_calls), 1)
        self.assertEqual(len(op._ind_joint_owners), 1)

    def test_owner_is_not_preempted_across_ticks(self):
        op = self._two_staged()
        independent.tick_task_motion(op, {"ee": {}}, None, self.NOW)
        first = set(op._ind_joint_owners)
        op.step_calls.clear()
        independent.tick_task_motion(op, {"ee": {}}, None, self.NOW + DT)
        self.assertEqual(set(op._ind_joint_owners), first)
        self.assertEqual(op.step_calls, list(first))


class JourneyTests(unittest.TestCase):
    """07-21 far-cube journey mission: survey both -> nearer first -> steady
    walk (forward/backward, never turning) -> grasp -> other leg -> stand."""
    NOW = 10.0

    def _op(self, near_pos, far_pos):
        near = FakeCube("near", near_pos, self.NOW)
        far = FakeCube("far", far_pos, self.NOW)
        op = FakeOperator([near, far])
        op.journey = True
        return op, near, far

    def _tuck(self, op):
        op._ee_act = {
            "left": np.array(JOURNEY_WALK_EE),
            "right": np.array(JOURNEY_WALK_EE) * np.array([1.0, -1.0, 1.0]),
        }

    def test_survey_gate_blocks_nav_until_both_confirmed(self):
        op, near, far = self._op([0.6, 0.15, -0.1], [-2.0, -0.2, -0.1])
        far.last_seen = self.NOW - 2.0           # sighting stale: unconfirmed
        op._arm_tasks["left"] = ArmTask("left", "near")
        self._tuck(op)
        vx, wz = independent.navigation_command(op, self.NOW)
        self.assertEqual((vx, wz), (0.0, 0.0))
        self.assertIsNone(op._survey_order)
        far.see(self.NOW)                        # both confirmed now
        independent.navigation_command(op, self.NOW)
        self.assertEqual(op._survey_order, ["near", "far"])

    def test_nav_serves_nearer_cube_despite_earlier_far_sighting(self):
        op, near, far = self._op([0.6, 0.15, -0.1], [-2.0, -0.2, -0.1])
        far.first_seen = self.NOW - 5.0          # far seen long before near
        op._arm_tasks["left"] = ArmTask("left", "near")
        op._arm_tasks["right"] = ArmTask("right", "far")
        self._tuck(op)
        vx, _ = independent.navigation_command(op, self.NOW)
        self.assertEqual(op._nav_task_key, "near")
        self.assertGreater(vx, 0.0)              # front cube: walks forward

    def test_backward_leg_for_rear_cube_no_turn_around(self):
        op, near, _far = self._op([-0.9, -0.15, -0.1], [2.2, 0.2, -0.1])
        op._arm_tasks["right"] = ArmTask("right", "near")
        op._arm_tasks["left"] = ArmTask("left", "far")
        self._tuck(op)
        vx, _ = independent.navigation_command(op, self.NOW)
        self.assertEqual(op._nav_task_key, "near")
        self.assertLess(vx, 0.0)                 # rear cube: walks BACKWARD

    def test_speed_schedule_cruise_far_slow_near(self):
        op, near, _far = self._op([2.0, 0.1, -0.1], [-2.5, -0.2, -0.1])
        op._arm_tasks["left"] = ArmTask("left", "near")
        self._tuck(op)
        vx_far, _ = independent.navigation_command(op, self.NOW)
        self.assertAlmostEqual(vx_far, CRUISE_VX, places=6)
        near.pos = np.array([0.4, 0.1, -0.1])    # close in: schedule ramps down
        vx_near, _ = independent.navigation_command(op, self.NOW)
        self.assertLess(vx_near, CRUISE_VX)
        self.assertGreaterEqual(vx_near, JOURNEY_NEAR_VX)

    def test_untucked_arms_gate_walking(self):
        op, _near, _far = self._op([0.8, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op._arm_tasks["left"] = ArmTask("left", "near")
        # arms parked wide at side-home -> no walking until the tuck completes,
        # but the tuck request must be raised
        op._ee_act = {"left": np.array([0.0, 0.55, 0.0]),
                      "right": np.array([0.0, -0.55, 0.0])}
        vx, _ = independent.navigation_command(op, self.NOW)
        self.assertEqual(vx, 0.0)
        self.assertTrue(op._journey_walk_pose)

    def test_base_still_requires_sustained_measurement(self):
        op, _n, _f = self._op([0.8, 0.1, -0.1], [-2.0, -0.2, -0.1])
        op._base_vel = [0.2, 0.0, 0.0]
        self.assertFalse(independent.journey_base_still(op, self.NOW))
        op._base_vel = [0.01, 0.0, 0.0]
        op._base_ang = [0.0, 0.0, 0.01]
        self.assertFalse(independent.journey_base_still(op, self.NOW))  # timer starts
        self.assertFalse(independent.journey_base_still(
            op, self.NOW + JOURNEY_STILL_S * 0.5))
        self.assertTrue(independent.journey_base_still(
            op, self.NOW + JOURNEY_STILL_S + 0.1))
        # Motion DEBITS the credit (08-07 audit) rather than nulling it: the
        # gyro is raw and one outlier used to discard the whole window, which
        # is the leg's only measurement-driven SKIP. One bad sample must cost
        # JOURNEY_STILL_DEBIT_S; SUSTAINED motion must still exhaust it.
        op._base_vel = [0.2, 0.0, 0.0]
        was = op._journey_still_since
        now = self.NOW + JOURNEY_STILL_S + 0.2
        self.assertFalse(independent.journey_base_still(op, now))
        self.assertAlmostEqual(op._journey_still_since,
                               was + JOURNEY_STILL_DEBIT_S, places=9)
        for _ in range(60):                      # ~1.2 s of real motion
            now += 0.02
            self.assertFalse(independent.journey_base_still(op, now))
        self.assertAlmostEqual(op._journey_still_since, now, places=9,
                               msg="sustained motion left credit standing")

    def test_carried_invalidates_pending_survey_position(self):
        op, near, far = self._op([0.6, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op.grasp = True
        h = HoldState(key="near", target=np.array([0.6, 0.15, -0.1]),
                      bias=None, sector="front", sector_idx=0)
        op._held["left"] = h
        near.held_by = "left"
        arms.enter_carried(op, h, "left", np.array([0.25, 0.25, 0.0]))
        self.assertEqual(far.last_seen, -1e9)    # survey position dead
        self.assertEqual(len(far.hist), 0)
        self.assertEqual(near.last_seen, self.NOW)  # the carried cube untouched

    def test_walk_pose_glide_targets_poweron_ee(self):
        op, _n, _f = self._op([0.8, 0.1, -0.1], [-2.0, -0.2, -0.1])
        op._journey_walk_pose = True
        pub = arbitration.home_glide(op, "right", 0.12)
        expect = np.array(JOURNEY_WALK_EE) * np.array([1.0, -1.0, 1.0])
        np.testing.assert_allclose(pub, expect)  # fake glide teleports to tgt
        op._journey_walk_pose = False
        op._home_stream = {"left": None, "right": None}
        pub2 = arbitration.home_glide(op, "right", 0.12)
        np.testing.assert_allclose(pub2, op.rest_ee["right"])

    def test_journey_assigns_far_cube_beyond_envelope(self):
        # The 07-21 CRITICAL: with the (now radius-honest) _target_safe, a 2 m
        # cube must still get a task in journey mode once the survey is done —
        # and must NOT in the standing mission.
        op, near, far = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._survey_order = ["near", "far"]
        independent.assign_tasks(op, self.NOW)
        self.assertIsNotNone(op._arm_tasks["left"])
        self.assertEqual(op._arm_tasks["left"].key, "near")
        op2, _n, _f = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op2.journey = False
        independent.assign_tasks(op2, self.NOW)
        self.assertIsNone(op2._arm_tasks["left"])

    def test_no_assignment_before_survey_completes(self):
        op, near, far = self._op([0.3, 0.15, -0.1], [-2.0, -0.2, -0.1])
        far.last_seen = self.NOW - 2.0           # far stale: survey open
        independent.assign_tasks(op, self.NOW)
        self.assertIsNone(op._arm_tasks["left"])  # even the CLOSE cube waits

    def test_survey_immune_to_held_cubes(self):
        op, near, far = self._op([0.3, 0.15, -0.1], [-0.9, -0.2, -0.1])
        near.held_by = "left"
        near.last_seen = -1e9                    # in-gripper: never confirms
        self.assertTrue(independent.journey_survey(op, self.NOW))

    def test_still_timer_resets_while_driving(self):
        op, _n, _f = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._arm_tasks["left"] = ArmTask("left", "near")
        op._survey_order = ["near", "far"]
        self._tuck(op)
        op._journey_still_since = self.NOW - 99.0   # stale from leg 1
        vx, _ = independent.navigation_command(op, self.NOW)
        self.assertNotEqual(vx, 0.0)
        self.assertIsNone(op._journey_still_since)

    def test_walk_pose_debounced_through_task_flicker(self):
        op, _n, _f = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._arm_tasks["left"] = ArmTask("left", "near")
        op._survey_order = ["near", "far"]
        self._tuck(op)
        independent.navigation_command(op, self.NOW)     # serving: tuck on
        self.assertTrue(op._journey_walk_pose)
        op._arm_tasks["left"] = None                     # flicker cancel
        independent.navigation_command(op, self.NOW + 1.0)
        self.assertTrue(op._journey_walk_pose)           # hysteresis holds
        independent.navigation_command(op, self.NOW + JOURNEY_TUCK_HOLD_S + 1.1)
        self.assertFalse(op._journey_walk_pose)          # released after hold

    def test_failed_hold_stands_and_mission_ends(self):
        op, near, far = self._op([0.4, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op._survey_order = ["near", "far"]
        near.held_by = "left"
        op._held["left"] = HoldState(
            key="near", target=near.pos.copy(), bias=None,
            sector="front", sector_idx=0, grasp="failed")
        op._arm_tasks["right"] = ArmTask("right", "far")
        self._tuck(op)
        vx, _ = independent.navigation_command(op, self.NOW)
        self.assertEqual(vx, 0.0)                        # refuses to walk
        far.held_by = "right"
        op._held["right"] = HoldState(
            key="far", target=far.pos.copy(), bias=None,
            sector="rear", sector_idx=2, grasp="carried")
        op._arm_tasks["right"] = None
        independent.tick_independent(op, {"ee": {}}, None, self.NOW)
        self.assertTrue(op._journey_done)                # failed counts resolved

    def test_blind_stillness_uses_longer_settle(self):
        op, _n, _f = self._op([0.8, 0.1, -0.1], [-2.0, -0.2, -0.1])
        op._base_vel = [0.0, 0.0, 0.0]                   # estimator-less zeros
        op._base_ang = [0.0, 0.0, 0.01]
        self.assertFalse(independent.journey_base_still(op, self.NOW))
        self.assertFalse(independent.journey_base_still(
            op, self.NOW + JOURNEY_STILL_S + 0.1))       # 1 s is NOT enough
        self.assertTrue(independent.journey_base_still(
            op, self.NOW + JOURNEY_STILL_S_BLIND + 0.1))

    def test_tuck_slots_publish_poweron_quat(self):
        op, _n, _f = self._op([2.0, 0.1, -0.1], [-2.3, -0.2, -0.1])
        op._journey_walk_pose = True
        pkt = arbitration.build_arm_packet(op)
        wq = JOURNEY_WALK_QUAT
        np.testing.assert_allclose(pkt["ee_quat"][0], wq)
        np.testing.assert_allclose(
            pkt["ee_quat"][1], [wq[0], -wq[1], wq[2], -wq[3]])
        op._journey_walk_pose = False
        op._home_stream = {"left": None, "right": None}
        pkt2 = arbitration.build_arm_packet(op)
        np.testing.assert_allclose(pkt2["ee_quat"][0], [1.0, 0.0, 0.0, 0.0])

    def test_journey_keeps_creeping_until_arrive_distance(self):
        # Gate-reachable at 0.40 m must NOT stop a journey leg; inside
        # JOURNEY_ARRIVE_M it must. (No gate -> reachable = dist <= 0.32.)
        op, near, _far = self._op([0.31, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op._arm_tasks["left"] = ArmTask("left", "near")
        task = op._arm_tasks["left"]
        near.pos = np.array([0.40, 0.10, -0.1])
        op.gate = SimpleNamespace(SCORE_THRESHOLD=0.5)
        op._gate_results = [(1.0, "left", True)] * 10
        for _ in range(6):
            independent.prepare_tasks(op, self.NOW)
        self.assertEqual(task.phase, ArmTaskPhase.ASSIGNED)   # still creeping
        near.pos = np.array([0.30, 0.10, -0.1])
        op._gate_results = [(1.0, "left", True)] * 10
        independent.prepare_tasks(op, self.NOW)
        self.assertEqual(task.phase, ArmTaskPhase.WAIT_STOP)  # inside 0.34: stop

    @staticmethod
    def _jpos_at(left7, right7):
        jp = np.zeros(31)
        jp[13:20] = left7
        jp[20:27] = right7
        return jp

    def test_posture_glide_streams_tuck_joints_from_side_home(self):
        op, _n, _f = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._journey_walk_pose = True
        jp = self._jpos_at(SIDE_HOME_JOINTS, mirror_arm(SIDE_HOME_JOINTS, "right"))
        pp = independent.journey_posture_payload(op, jp)
        self.assertIsNotNone(pp)
        np.testing.assert_allclose(pp["left"]["joint_pos"], POWERON_JOINTS)
        np.testing.assert_allclose(pp["right"]["joint_pos"],
                                   mirror_arm(POWERON_JOINTS, "right"))
        self.assertEqual(pp["left"]["rate"], STAGE_RATE)

    def test_posture_glide_silent_when_settled(self):
        op, _n, _f = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._journey_walk_pose = True
        jp = self._jpos_at(POWERON_JOINTS, mirror_arm(POWERON_JOINTS, "right"))
        self.assertIsNone(independent.journey_posture_payload(op, jp))

    def test_posture_glide_untucks_to_side_home_on_arrival(self):
        op, _n, _f = self._op([0.3, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._journey_walk_pose = False               # arrival released the tuck
        jp = self._jpos_at(POWERON_JOINTS, mirror_arm(POWERON_JOINTS, "right"))
        pp = independent.journey_posture_payload(op, jp)
        self.assertIsNotNone(pp)
        np.testing.assert_allclose(pp["left"]["joint_pos"], SIDE_HOME_JOINTS)

    def test_leg2_tuck_glide_flows_into_a_nonempty_packet(self):
        # 07-26 deadlock audit (CRITICAL regression): the joint tuck glide was
        # gated on `packet is None`, which never happens under nominal
        # telemetry — so after leg 1 parked the free arm at side-home, the
        # Cartesian home_glide could never satisfy the JOINT-space tuck gate
        # and every journey run deadlocked at the leg-2 walk entry. The
        # posture slots must now ride in the returned packet even when the
        # Cartesian packet is non-None.
        op, near, far = self._op([0.35, 0.15, -0.1], [-2.0, -0.2, -0.1])
        near.held_by = "left"                     # cube 1 carried
        op._held["left"] = HoldState(
            key="near", target=near.pos.copy(), bias=None,
            sector="front", sector_idx=0, grasp=GraspState.CARRIED)
        op._arm_tasks["right"] = ArmTask("right", "far")
        far.assigned_to = "right"
        jp = self._jpos_at(POWERON_JOINTS,        # held arm exempt anyway
                           mirror_arm(SIDE_HOME_JOINTS, "right"))  # free arm at side
        op._jpos = jp          # production tick() wires telemetry joints here;
                               # journey_arms_tucked reads THIS, not the arg
        vx, _wz, packet = independent.tick_independent(
            op, {"ee": {}}, jp, self.NOW)
        self.assertTrue(op._journey_walk_pose)    # tuck requested for leg 2
        self.assertEqual(vx, 0.0)                 # gate still closed this tick
        self.assertIsNotNone(packet, "tuck glide must reach the wire")
        self.assertIn("right", packet)
        np.testing.assert_allclose(packet["right"]["joint_pos"],
                                   mirror_arm(POWERON_JOINTS, "right"))

    def test_posture_glide_skips_staging_and_reaching_arms(self):
        # 07-26: the merge is unconditional now, so the payload itself must
        # never fight an arm whose task streams its own command.
        op, _n, _f = self._op([2.0, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._journey_walk_pose = True
        jp = self._jpos_at(SIDE_HOME_JOINTS, mirror_arm(SIDE_HOME_JOINTS, "right"))
        for phase in (ArmTaskPhase.STAGING, ArmTaskPhase.REACHING):
            with self.subTest(phase=phase):
                task = ArmTask("left", "near")
                task.phase = phase
                op._arm_tasks["left"] = task
                pp = independent.journey_posture_payload(op, jp)
                self.assertIsNotNone(pp)
                self.assertNotIn("left", pp, f"{phase} arm must be excluded")
                self.assertIn("right", pp)
        op._arm_tasks["left"] = None

    def test_posture_glide_defers_to_active_staircase(self):
        op, near, _f = self._op([0.3, 0.15, -0.1], [-2.3, -0.2, -0.1])
        op._journey_walk_pose = True
        op._held["right"] = HoldState(
            key="far", target=np.array([-0.3, -0.2, -0.1]), bias=None,
            sector="rear", sector_idx=2,
            track=TrackMotion("right", 2, 1, [1]))
        jp = self._jpos_at(SIDE_HOME_JOINTS, mirror_arm(SIDE_HOME_JOINTS, "right"))
        self.assertIsNone(independent.journey_posture_payload(op, jp))

    def test_tucked_check_is_joint_based_when_telemetry_present(self):
        op, _n, _f = self._op([0.8, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op._ee_act = {"left": np.array([0.0, 0.55, 0.0]),   # EE says wide...
                      "right": np.array([0.0, -0.55, 0.0])}
        op._jpos = self._jpos_at(POWERON_JOINTS, mirror_arm(POWERON_JOINTS, "right"))
        self.assertTrue(independent.journey_arms_tucked(op))  # ...joints win
        op._jpos = self._jpos_at(SIDE_HOME_JOINTS, mirror_arm(SIDE_HOME_JOINTS, "right"))
        self.assertFalse(independent.journey_arms_tucked(op))

    def test_sync_reach_first_ready_waits_then_both_launch(self):
        # Video-demo sync: left's cube confirmed early, right's cube appears
        # late — left must hold at the latch gate; once right is ready both
        # enter REACHING together.
        left_c = FakeCube("lc", [0.25, 0.12, -0.1], self.NOW)
        right_c = FakeCube("rc", [0.25, -0.12, -0.1], self.NOW)
        right_c.last_seen = self.NOW - 2.0      # sighting stale: unconfirmed
        op = FakeOperator([left_c, right_c])
        op.sync_reach = True
        op._sync_wait_t = -1e9
        independent.assign_tasks(op, self.NOW)
        self.assertIsNotNone(op._arm_tasks["left"])
        self.assertIsNone(op._arm_tasks["right"])   # not confirmed: no task
        independent.prepare_tasks(op, self.NOW)
        self.assertEqual(op._arm_tasks["left"].phase,
                         ArmTaskPhase.WAIT_STOP)    # ready but held at the gate
        right_c.last_seen = self.NOW                # now confirmed
        independent.assign_tasks(op, self.NOW)
        independent.prepare_tasks(op, self.NOW)
        self.assertEqual(op._arm_tasks["left"].phase, ArmTaskPhase.REACHING)
        self.assertEqual(op._arm_tasks["right"].phase, ArmTaskPhase.REACHING)

    def test_sync_reach_second_round_proceeds_solo(self):
        left_c = FakeCube("lc", [0.25, 0.12, -0.1], self.NOW)
        right_c = FakeCube("rc", [0.25, -0.12, -0.1], self.NOW)
        left_c.held_by = "left"                     # first cube already in hand
        op = FakeOperator([left_c, right_c])
        op.sync_reach = True
        op._sync_wait_t = -1e9
        independent.assign_tasks(op, self.NOW)
        independent.prepare_tasks(op, self.NOW)
        self.assertEqual(op._arm_tasks["right"].phase, ArmTaskPhase.REACHING)

    def test_grasp_window_immunity_to_occlusion(self):
        # Hand AT the cube: tag occluded 3 s (beyond the 2.5 s grace, inside
        # the 4 s near-grasp grace) must NOT retract; a far hand must.
        cube = FakeCube("c", [0.32, 0.18, -0.1], self.NOW)
        cube.last_seen = self.NOW - 3.0
        op = FakeOperator([cube])
        op.grasp = True
        op.dynamic_track = True
        op.independent_arms = True
        tgt = np.array([0.32, 0.18, -0.1])
        op._held["left"] = HoldState(key="c", target=tgt.copy(), bias=None,
                                     sector="front", sector_idx=0)
        cube.held_by = "left"
        arms.held_update(op, self.NOW, {"ee": {"left": {"pos_actual":
                                                        tgt.tolist()}}}, None)
        self.assertEqual(op._held["left"].state, "tracking")   # immune
        op._held["left"].state = "tracking"
        op._held["left"].home = False
        arms.held_update(op, self.NOW, {"ee": {"left": {"pos_actual":
                                                        [0.0, 0.55, 0.0]}}}, None)
        self.assertEqual(op._held["left"].state, "retracted")  # far hand: normal

    def test_grasp_window_immunity_to_gate_flicker(self):
        cube = FakeCube("c", [0.32, 0.18, -0.1], self.NOW)   # cube fully visible
        op = FakeOperator([cube])
        op.grasp = True
        op.dynamic_track = True
        op.independent_arms = True
        op._dyn_reachable = lambda *_a: False                # gate says NO
        tgt = np.array([0.32, 0.18, -0.1])
        op._held["left"] = HoldState(key="c", target=tgt.copy(), bias=None,
                                     sector="front", sector_idx=0)
        cube.held_by = "left"
        arms.held_update(op, self.NOW, {"ee": {"left": {"pos_actual":
                                                        tgt.tolist()}}}, None)
        self.assertEqual(op._held["left"].state, "tracking")   # flicker ignored

    def test_survey_phase_keeps_the_chest_tuck(self):
        # 07-21: journey starts tucked (power-on IS the tuck); surveying must
        # not clear the flag — arms would detour to side-home and back.
        op, _near, far = self._op([0.6, 0.15, -0.1], [-2.0, -0.2, -0.1])
        far.last_seen = self.NOW - 2.0           # survey still open
        op._journey_walk_pose = True             # as the real ctor sets it
        independent.navigation_command(op, self.NOW)
        self.assertTrue(op._journey_walk_pose)

    def test_mission_ready_settles_against_tuck_not_side_home(self):
        op, _n, _f = self._op([0.6, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op._home_settled = False
        op._journey_walk_pose = True
        self._tuck(op)                           # arms AT the tuck, far from rest_ee
        self.assertTrue(independent.ensure_mission_ready(op, self.NOW))
        op2, _n2, _f2 = self._op([0.6, 0.15, -0.1], [-2.0, -0.2, -0.1])
        op2._home_settled = False
        op2._journey_walk_pose = True            # tucked expected, but arms wide
        op2._ee_act = {"left": np.array([0.0, 0.55, 0.0]),
                       "right": np.array([0.0, -0.55, 0.0])}
        self.assertFalse(independent.ensure_mission_ready(op2, self.NOW))

    def test_mission_complete_prints_once_and_stands(self):
        op, near, far = self._op([0.6, 0.15, -0.1], [-0.8, -0.2, -0.1])
        for cube, arm in ((near, "left"), (far, "right")):
            cube.held_by = arm
            op._held[arm] = HoldState(
                key=cube.key, target=cube.pos.copy(), bias=None,
                sector="front", sector_idx=0, grasp="carried")
        vx, wz, _packet = independent.tick_independent(
            op, {"ee": {}}, None, self.NOW)
        self.assertTrue(op._journey_done)
        self.assertEqual((vx, wz), (0.0, 0.0))


class ChestHomeTests(unittest.TestCase):
    """07-24 --chest-home (temporary standing session): free arms HOLD the
    power-on chest tuck instead of gliding to side-home, and every cube is
    reached directly from it.

    The load-bearing safety property is the LAST test: chest-home must not wake
    journey_posture_payload. Since the 07-26 deadlock fixes the payload DOES
    carry a STAGING/REACHING exclusion, but membership in it is still an arm's
    control mode — a standing mission has no business flipping free arms into
    joint posture slots, so the `journey` gate remains the contract.
    """
    NOW = 10.0

    SIDE_HOME_EE = {"left": np.array([0.0, 0.55, 0.0]),
                    "right": np.array([0.0, -0.55, 0.0])}

    def _op(self, chest=True):
        cube = FakeCube("c", [0.30, 0.18, -0.05], self.NOW)
        op = FakeOperator([cube])
        op.journey = False
        op._journey_walk_pose = False
        op.chest_home = chest
        # the operator installs the chest tuck AS rest_ee (that is the whole
        # mechanism); the fake mirrors that wiring.
        op.rest_ee = {a: np.array(JOURNEY_WALK_EE)
                      * np.array([1.0, 1.0 if a == "left" else -1.0, 1.0])
                      for a in ("left", "right")} if chest else \
            {a: v.copy() for a, v in self.SIDE_HOME_EE.items()}
        return op

    def _chest(self, arm):
        return np.array(JOURNEY_WALK_EE) * np.array(
            [1.0, 1.0 if arm == "left" else -1.0, 1.0])

    def test_idle_arm_homes_to_the_chest_not_the_side(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                op = self._op()
                pub = arbitration.home_glide(op, arm, STAGE_REACH_EE_RATE)
                np.testing.assert_allclose(pub, self._chest(arm))
                self.assertFalse(np.allclose(pub, self.SIDE_HOME_EE[arm]),
                                 "chest-home must never target the side home")

    def test_retracted_held_arm_homes_to_the_chest(self):
        # 07-24 hardware: a dyn-retract printed "gliding to SIDE home (50 cm)"
        # because only the IDLE glide had been patched. rest_ee is the fix.
        op = self._op()
        op._held["right"] = HoldState(
            key="c", target=np.array([0.3, -0.2, 0.0]), bias=None,
            sector="front", sector_idx=0, home=True)
        pub = arbitration.home_glide(op, "right", STAGE_REACH_EE_RATE)
        np.testing.assert_allclose(pub, self._chest("right"))

    def test_carry_home_targets_the_chest(self):
        # "re-tuck to the chest after a grasp": enter_carried parks the grasped
        # hold at rest_ee
        op = self._op()
        h = HoldState(key="c", target=np.array([0.3, 0.2, 0.0]), bias=None,
                      sector="front", sector_idx=0, grasp=GraspState.GRASPED)
        op._held["left"] = h
        op.cubes["c"].held_by = "left"
        arms.enter_carried(op, h, "left", np.array([0.3, 0.2, 0.0]))
        np.testing.assert_allclose(h.target, self._chest("left"))

    def test_off_flag_still_homes_to_the_side(self):
        op = self._op(chest=False)
        pub = arbitration.home_glide(op, "left", STAGE_REACH_EE_RATE)
        np.testing.assert_allclose(pub, self.SIDE_HOME_EE["left"])

    def test_tuck_slots_publish_the_power_on_orientation(self):
        # NEUTRAL at the tuck point twists the wrists 17.4 deg off the power-on
        # pose (07-21 review) — chest-home must use the FK orientation too.
        op = self._op()
        packet = arbitration.build_arm_packet(op)
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(packet["ee_quat"][0], JOURNEY_WALK_QUAT)
        wq = JOURNEY_WALK_QUAT
        np.testing.assert_allclose(packet["ee_quat"][1],
                                   [wq[0], -wq[1], wq[2], -wq[3]])

    def test_settle_gate_measures_against_the_tuck(self):
        # judged against side-home the gate would never pass and would burn the
        # full HOME_SETTLE_TIMEOUT_S before tasks may begin
        op = self._op()
        op._home_settled = False
        op._ee_act = {a: np.array(JOURNEY_WALK_EE) * np.array(
            [1.0, 1.0 if a == "left" else -1.0, 1.0]) for a in ("left", "right")}
        self.assertTrue(independent.ensure_mission_ready(op, self.NOW))
        self.assertTrue(op._home_settled)

    def test_chest_home_never_wakes_the_joint_posture_payload(self):
        # THE safety contract: journey_posture_payload stays gated on `journey`.
        op = self._op()
        jp = np.zeros(31)
        jp[13:20] = np.asarray(SIDE_HOME_JOINTS)      # far from POWERON
        jp[20:27] = mirror_arm(SIDE_HOME_JOINTS, "right")
        self.assertIsNone(
            independent.journey_posture_payload(op, jp),
            "chest-home must not activate the joint posture payload — the "
            "journey gate is the contract keeping standing missions Cartesian")


if __name__ == "__main__":
    unittest.main()
