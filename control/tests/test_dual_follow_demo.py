"""What the dual-arm sector-follow demo may and may not command.

The demo hands the robot-side mink IK a LIVE Cartesian stream while the arms
sit at STAGE stations, which puts it on the wrong side of two incidents this
codebase already paid for:

  * INC-1, the 07-15 cross-body jump — an arm swept front->rear THROUGH the
    torso because a local IK was asked to cross a basin. The stations exist so
    the solver is seeded in the right basin, and the ONLY thing keeping this
    demo out of that class is the rule that an arm follows nothing outside its
    own sector. Two tests pin it: at the gate (a foreign-sector cube is never
    picked) and mid-follow (one that crosses out ends the follow).
  * the 07-31 chest-home measurement — mink's CollisionAvoidanceLimit refuses
    to steer into folds, so a Cartesian home command landed 86.6 deg off from
    every start. Homes here must therefore be JOINT slots, never Cartesian,
    and the startup path must walk stations that were actually audited:
    power-on->front and power-on->side at >= 39.5 mm (the 07-31 re-run) and
    side->rear at >= 28.9 mm (the 07-17 raised-chain re-validation). A direct
    power-on->rear hop has no audit at all, and startup_seq must never emit
    one.

The rest pin the properties that make it a DEMO rather than a mission: it
never grasps, it never walks, and it never goes silent (real_env's 0.5 s
arm-silence failsafe crawls the arms to default at 0.125 rad/s, which is how
the journey lost its tuck mid-walk).
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import humanoid_auto_operator as OP  # noqa: E402
    import humanoid_dual_follow_demo as D  # noqa: E402

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    OP = D = None
    _SKIP = f"demo unavailable ({type(exc).__name__}: {exc})"


class _Cube:
    """Just the CubeTrack surface the demo reads."""

    def __init__(self, last_seen: float = 0.0):
        self.last_seen = last_seen


class _Rig:
    """Duck-typed GazeRig: median/faces_used (the ENTRY evidence) and
    latest/last_inliers (the LIVE tracking evidence follow_slot reads since
    08-11), all backed by the same canned fix."""

    def __init__(self, fixes: dict):
        # fixes: key -> (pos | None, faces, last_seen)
        self._fixes = fixes
        self.cubes = {k: _Cube(v[2]) for k, v in fixes.items()}

    def median(self, key, now):
        pos = self._fixes[key][0]
        return None if pos is None else np.asarray(pos, float)

    def faces_used(self, key, now):
        return self._fixes[key][1]

    def latest(self, key, now):
        return self.median(key, now)

    def last_inliers(self, key):
        return self._fixes[key][1]


def _arm(home="front", **kw):
    a = D.Arm(arm="left" if home == "front" else "right", home=home)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@unittest.skipIf(D is None, _SKIP)
class StartupStaircaseTests(unittest.TestCase):

    def test_a_rear_home_never_hops_straight_from_power_on(self):
        """The whole point of startup_seq. power-on->rear is a ~1.2 rad
        shoulder sweep with no clearance audit behind it; the audited route is
        power-on->side (39.5 mm, 07-31) then side->rear (28.9 mm, 07-17)."""
        seq = D.startup_seq(D.STATIONS.index("rear"))
        self.assertEqual([D.STATIONS[i] for i in seq], ["side", "rear"])

    def test_directly_audited_homes_are_one_hop(self):
        """front<->POWERON and side<->POWERON were both measured at the same
        39.5 mm floor, so neither needs an intermediate station."""
        for name in ("front", "side"):
            seq = D.startup_seq(D.STATIONS.index(name))
            self.assertEqual([D.STATIONS[i] for i in seq], [name])

    def test_every_hop_is_adjacent_on_the_track(self):
        """side is the middle station precisely so front<->rear must pass
        through it; a sequence that skips a station is an unaudited curve."""
        for home in range(len(D.STATIONS)):
            seq = D.startup_seq(home)
            for lo, hi in zip(seq, seq[1:]):
                self.assertEqual(abs(hi - lo), 1,
                                 msg=f"home={D.STATIONS[home]} seq={seq}")

    def test_the_staircase_settles_one_station_at_a_time(self):
        """Arrival THEN dwell, per hop — a station is only 'settled' after it
        has been held, so every segment starts from the audited configuration
        rather than from a corner being cut at speed."""
        a = _arm(home="rear")
        a.start_homing(D.startup_seq(a.home_idx), now=0.0)
        at_side = {n: float(v) for n, v in zip(
            D.ARM_JOINTS["right"], D.station_posture("right", 1))}
        # sitting exactly on SIDE: arrival latches, but the dwell is not done
        slot, arrived = D.step_home(a, at_side, now=0.0)
        self.assertFalse(arrived)
        self.assertEqual(slot["joint_pos"],
                         [float(v) for v in D.station_posture("right", 1)])
        slot, arrived = D.step_home(a, at_side, now=OP.STAGE_DWELL_S / 2.0)
        self.assertFalse(arrived, "advanced before the dwell completed")
        # dwell over: SIDE pops, REAR becomes the commanded hop, still not home
        slot, arrived = D.step_home(a, at_side, now=OP.STAGE_DWELL_S + 0.01)
        self.assertFalse(arrived)
        self.assertEqual(a.seq, [2])
        slot, _ = D.step_home(a, at_side, now=OP.STAGE_DWELL_S + 0.02)
        self.assertEqual(slot["joint_pos"],
                         [float(v) for v in D.station_posture("right", 2)])

    def test_homes_are_joint_slots_never_cartesian(self):
        """The 07-31 finding: a Cartesian home command cannot reach a folded
        posture (mink's 50 mm CollisionAvoidanceLimit), and it landed 86.6 deg
        off. Every station command must be joints."""
        a = _arm(home="rear")
        a.start_homing(D.startup_seq(a.home_idx), now=0.0)
        enc = {n: 0.0 for n in D.ARM_JOINTS["right"]}
        slot, _ = D.step_home(a, enc, now=0.0)
        self.assertIn("joint_pos", slot)
        self.assertNotIn("ee_pos", slot)
        self.assertEqual(len(slot["joint_pos"]), 7)
        self.assertEqual(slot["rate"], OP.STAGE_RATE)

    def test_the_right_arm_is_mirrored_not_negated_by_hand(self):
        """ARM_MIRROR_SIGN is the only sanctioned mirror: a wrong sign does not
        fail loudly — the target stays inside every limit and the gripper just
        sits rotated."""
        for idx, name in enumerate(D.STATIONS):
            np.testing.assert_allclose(
                D.station_posture("right", idx),
                OP.mirror_arm(OP.STAGE_JOINTS[name], "right"))
            np.testing.assert_allclose(
                D.station_posture("left", idx), OP.STAGE_JOINTS[name])

    def test_a_dry_run_still_walks_the_whole_staircase(self):
        """Nothing is published in a dry run, so the arm can never arrive and
        the encoder test would park the rehearsal on hop 0 forever."""
        a = _arm(home="rear")
        a.start_homing(D.startup_seq(a.home_idx), now=0.0)
        enc = {n: 0.0 for n in D.ARM_JOINTS["right"]}   # nowhere near a station
        t, arrived = 0.0, False
        for _ in range(200):
            t += OP.STAGE_DWELL_S / 2.0
            _, arrived = D.step_home(a, enc, now=t, moves=False)
            if arrived:
                break
        self.assertTrue(arrived, "dry run never reached its station")
        self.assertFalse(D.step_home(a, enc, now=t, moves=True)[1] is False,
                         "an empty sequence must read as arrived")


@unittest.skipIf(D is None, _SKIP)
class StartupCorridorTests(unittest.TestCase):
    """The one motion in this demo that no offline audit covers.

    With real_env --use-ik and nobody on 9874, mink owns the arms and walks
    them out of the power-on fold (measured 86.6 deg, 07-31), so the arms
    routinely start at a posture no whitelist recognises. A whitelist gate just
    refuses the demo forever; a swept certificate answers the same question
    about the actual corridor.
    """

    @staticmethod
    def _poweron_enc():
        return {n: float(v) for arm in D.ARMS
                for n, v in zip(D.ARM_JOINTS[arm],
                                OP.mirror_arm(OP.POWERON_JOINTS, arm))}

    def test_frames_follow_the_receivers_equal_rate_curve(self):
        """NOT a straight line in joint space. real_env clamps each joint
        independently by rate*dt, so the joint with the smaller delta arrives
        first — that per-joint curve is what the 07-17 station audit measured,
        and a lerp would certify a corridor the robot never drives."""
        enc = {n: 0.0 for n in D.ARM_JOINTS_ALL}
        target = D.station_posture("left", 0)
        frames = D.staircase_frames(enc, {"left": [0], "right": []})
        step = OP.STAGE_RATE * D.PUB_DT
        prev = frames[0][:7]
        for f in frames[1:]:
            move = np.abs(f[:7] - prev)
            self.assertTrue(np.all(move <= step + 1e-9),
                            "a joint moved faster than the commanded rate")
            prev = f[:7]
        np.testing.assert_allclose(frames[-1][:7], target,
                                   atol=OP.STAGE_TOL_RAD)
        # the small-delta joints must reach their target strictly earlier than
        # the largest one — the signature of equal-rate, not lerp
        deltas = np.abs(np.asarray(target) - frames[0][:7])
        slow, fast = int(np.argmax(deltas)), int(np.argmin(deltas))

        def arrival(j):
            # STAGE_TOL_RAD is step_home's own arrival test, and the schedule
            # latches its dwell the moment the WORST joint clears it — so the
            # binding joint stops ~0.02 rad short and a tighter probe would
            # never fire on it.
            return next(i for i, f in enumerate(frames)
                        if abs(f[j] - target[j]) <= OP.STAGE_TOL_RAD)
        self.assertLess(arrival(fast), arrival(slow))

    def test_an_idle_arm_contributes_no_motion(self):
        enc = self._poweron_enc()
        frames = D.staircase_frames(enc, {"left": [0], "right": []})
        for f in frames:
            np.testing.assert_allclose(f[7:], frames[0][7:])

    def test_the_schedule_ends_on_both_stations(self):
        enc = self._poweron_enc()
        frames = D.staircase_frames(
            enc, {"left": D.startup_seq(0), "right": D.startup_seq(2)})
        np.testing.assert_allclose(frames[-1][:7], OP.STAGE_JOINTS["front"],
                                   atol=OP.STAGE_TOL_RAD)
        np.testing.assert_allclose(
            frames[-1][7:], OP.mirror_arm(OP.STAGE_JOINTS["rear"], "right"),
            atol=OP.STAGE_TOL_RAD)

    def test_the_power_on_corridor_to_front_and_rear_clears_the_floor(self):
        """REGRESSION PIN with the measured number. Bound by the LEFT arm's own
        power-on -> FRONT hop (L_base_collision ~ base_link_collision1, ~1.6 s
        in) at 27.3 mm — 2.3 mm of margin over the 25 mm floor. The right arm's
        power-on -> side -> rear route never dips below 39.5 mm, the 07-31
        audited POWERON floor. If a model or station change moves this, the
        demo's startup is what changes with it."""
        import mujoco
        from humanoid_model import MJCF_MODEL_PATH
        model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        frames = D.staircase_frames(
            self._poweron_enc(),
            {"left": D.startup_seq(0), "right": D.startup_seq(2)})
        clear_m, where = D.certify(model, frames)
        self.assertIsNone(D.corridor_verdict(clear_m, where),
                          f"power-on startup no longer certifies: "
                          f"{clear_m * 1000:.1f} mm at {where}")
        self.assertAlmostEqual(clear_m * 1000, 27.3, delta=1.0)

    def test_a_corridor_under_the_floor_is_refused(self):
        why = D.corridor_verdict(D.CERT_FLOOR_M - 0.001, "elbow ~ torso")
        self.assertIsNotNone(why)
        self.assertIn("elbow ~ torso", why)
        self.assertIsNone(D.corridor_verdict(D.CERT_FLOOR_M, "x ~ y"))

    def test_the_station_postures_are_recognised_starts(self):
        """The demo parks an arm at REAR by design, so its own second run must
        not be refused — the reach tool's list has no side/rear entry."""
        homes = D.accepted_homes()
        for name in ("front", "side", "rear"):
            self.assertIn(name, homes)
        for arm in D.ARMS:
            enc = {n: float(v) for n, v in zip(
                D.ARM_JOINTS[arm], OP.mirror_arm(OP.STAGE_JOINTS["rear"], arm))}
            far, name = D.nearest_home(enc, arm)
            self.assertEqual(name, "rear")
            self.assertAlmostEqual(far, 0.0, places=9)


@unittest.skipIf(D is None, _SKIP)
class SectorOwnershipTests(unittest.TestCase):
    """INC-1: an arm must never be sent to a target in another basin."""

    def test_a_front_arm_never_picks_a_rear_cube(self):
        rig = _Rig({"c": ([-0.40, -0.05, 0.0], 3, 0.0)})
        self.assertEqual(D.sector_of(np.array([-0.40, -0.05, 0.0])), "rear")
        self.assertIsNone(D.pick_cube(_arm("front"), rig, ("c",), 0.0, 2))

    def test_a_rear_arm_never_picks_a_front_cube(self):
        rig = _Rig({"c": ([0.40, 0.05, 0.0], 3, 0.0)})
        self.assertIsNone(D.pick_cube(_arm("rear"), rig, ("c",), 0.0, 2))

    def test_each_arm_picks_the_cube_in_its_own_sector(self):
        rig = _Rig({"e": ([0.40, 0.10, 0.0], 3, 0.0),
                    "f": ([-0.40, -0.10, 0.0], 3, 0.0)})
        self.assertEqual(D.pick_cube(_arm("front"), rig, ("e", "f"), 0.0, 2)[0],
                         "e")
        self.assertEqual(D.pick_cube(_arm("rear"), rig, ("e", "f"), 0.0, 2)[0],
                         "f")

    def test_sectors_partition_the_circle_so_two_arms_cannot_contend(self):
        """A cube has exactly one sector, so disjoint-station arms can never
        latch the same key — the demo needs no claim protocol."""
        for deg in range(0, 360, 7):
            r = np.radians(deg)
            pos = np.array([0.4 * np.cos(r), 0.4 * np.sin(r), 0.0])
            owners = [h for h in D.STATIONS if D.sector_of(pos) == h]
            self.assertEqual(len(owners), 1, msg=f"deg={deg}")

    def test_a_cube_crossing_out_mid_follow_ends_the_follow(self):
        a = _arm("front", state=D.FOLLOW, key="e",
                 goal=np.array([0.40, 0.10, 0.0]),
                 stream=np.array([0.40, 0.10, 0.0]))
        rig = _Rig({"e": ([-0.40, -0.10, 0.0], 3, 0.0)})   # dragged to the rear
        for _ in range(D.FOLLOW_OUT_TICKS):
            slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        self.assertIsNone(slot)
        # The verdict must name where the cube WENT (rear), not where the
        # frozen goal still sits (front) — a goal stops updating the instant
        # the gate fails, so reporting against it would blame the arm's own
        # sector for the arm not being allowed to serve it.
        self.assertIn("REAR", why)
        self.assertIn("FRONT station", why)

    def test_the_sector_edge_is_debounced_in_both_directions(self):
        """A cube held on a bearing boundary must not flap the arm."""
        a = _arm("front", state=D.FOLLOW, key="e",
                 goal=np.array([0.40, 0.10, 0.0]),
                 stream=np.array([0.40, 0.10, 0.0]))
        rig = _Rig({"e": ([-0.40, -0.10, 0.0], 3, 0.0)})
        for _ in range(D.FOLLOW_OUT_TICKS - 1):
            slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
            self.assertIsNone(why, "left on fewer than the debounce count")
            self.assertIsNotNone(slot)


@unittest.skipIf(D is None, _SKIP)
class FollowRuleTests(unittest.TestCase):

    def test_an_unseen_cube_sends_the_arm_home(self):
        a = _arm("front", state=D.FOLLOW, key="e",
                 goal=np.array([0.40, 0.10, 0.0]),
                 stream=np.array([0.40, 0.10, 0.0]))
        rig = _Rig({"e": (None, 0, -D.FOLLOW_LOST_S - 0.1)})
        slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        self.assertIsNone(slot)
        self.assertIn("unseen", why)

    def test_a_brief_occlusion_does_not_retract(self):
        """DYN_LOST_GRACE_S was raised 1.0 -> 2.5 exactly because every hand
        entering frame yanked a following arm back."""
        a = _arm("front", state=D.FOLLOW, key="e",
                 goal=np.array([0.40, 0.10, 0.0]),
                 stream=np.array([0.40, 0.10, 0.0]))
        rig = _Rig({"e": (None, 0, -1.0)})
        slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        self.assertIsNone(why)
        self.assertIn("ee_pos", slot)

    def test_one_face_evidence_freezes_the_goal_without_retracting(self):
        """Single-tag PnP is not steering data (SERVO_MIN_INLIERS) — but the
        cube IS still being seen, so the honest response is a still hand, not
        a retreat."""
        goal = np.array([0.40, 0.10, 0.0])
        a = _arm("front", state=D.FOLLOW, key="e", goal=goal.copy(),
                 stream=goal.copy())
        rig = _Rig({"e": ([0.30, 0.30, 0.0], 1, 0.0)})
        slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        self.assertIsNone(why)
        np.testing.assert_allclose(a.goal, goal)
        np.testing.assert_allclose(slot["ee_pos"], goal)

    def test_the_published_point_is_ramped_never_jumped(self):
        """07-18: a raw EE goal handed to the IK 'shot the arms out to the
        sides'. One tick may move the published point at most
        FOLLOW_EE_RATE * dt."""
        a = _arm("front", state=D.FOLLOW, key="e",
                 goal=np.array([0.30, 0.10, 0.0]),
                 stream=np.array([0.30, 0.10, 0.0]))
        rig = _Rig({"e": ([0.45, 0.20, 0.10], 3, 0.0)})
        slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        self.assertIsNone(why)
        moved = float(np.linalg.norm(
            np.asarray(slot["ee_pos"]) - np.array([0.30, 0.10, 0.0])))
        self.assertLessEqual(moved, D.FOLLOW_EE_RATE * D.PUB_DT + 1e-9)

    def test_jitter_inside_the_deadband_does_not_re_aim(self):
        goal = np.array([0.40, 0.10, 0.0])
        a = _arm("front", state=D.FOLLOW, key="e", goal=goal.copy(),
                 stream=goal.copy())
        nudge = goal + np.array([D.FOLLOW_DEADBAND_M * 0.5, 0.0, 0.0])
        rig = _Rig({"e": (nudge, 3, 0.0)})
        D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        np.testing.assert_allclose(a.goal, goal)

    def test_an_out_of_envelope_cube_is_never_picked(self):
        """The IK has no self-collision awareness: a cube against the torso or
        at face height would otherwise become a literal IK target."""
        far = ([D.FOLLOW_MAX_RADIUS_M + 0.05, 0.0, 0.0], 3, 0.0)
        near = ([OP.TARGET_MIN_RADIUS_M - 0.05, 0.0, 0.0], 3, 0.0)
        high = ([0.35, 0.05, OP.TARGET_Z_MAX_M + 0.05], 3, 0.0)
        for name, fix in (("far", far), ("near", near), ("high", high)):
            rig = _Rig({"c": fix})
            self.assertIsNone(D.pick_cube(_arm("front"), rig, ("c",), 0.0, 2),
                              msg=name)

    def test_a_one_face_cube_is_never_picked_up_at_the_gate(self):
        rig = _Rig({"c": ([0.40, 0.10, 0.0], 1, 0.0)})
        self.assertIsNone(D.pick_cube(_arm("front"), rig, ("c",), 0.0, 2))
        self.assertIsNotNone(D.pick_cube(_arm("front"), rig, ("c",), 0.0, 1))

    def test_the_lost_grace_is_two_seconds(self):
        """08-11 user: 'when it loses it, hold for 2 seconds and then
        retreat' — and the retreat being re-acquirable is what makes the
        shorter grace cheap."""
        self.assertEqual(D.FOLLOW_LOST_S, 2.0)

    def test_the_tracking_goal_reads_the_newest_fix_not_the_median(self):
        """08-11 user: 'there's always a giant latency.' The 0.7 s median is
        a built-in ~0.35 s lag for a moving cube; follow_slot must read the
        newest packet (latest/last_inliers), with the deadband and the ramp
        absorbing its jitter."""
        src = pathlib.Path(D.__file__).read_text()
        body = src.split("def follow_slot", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("rig.latest(", body,
                      "follow_slot no longer tracks the newest fix")
        self.assertIn("rig.last_inliers(", body)
        pick = src.split("def pick_cube", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("rig.median(", pick,
                      "the ENTRY decision must stay on the median")

    def test_a_retreat_reacquire_enters_follow_from_where_the_arm_is(self):
        """08-11 user: 'if it sees the cube while retreating, don't hesitate,
        just fetch again.' begin_follow seeds both ramps from the MEASURED
        hand, so a mid-retreat launch is the same gentle launch."""
        a = _arm("front", state=D.HOMING, retreating=True, seq=[0])
        tel = {"ee": {"left": {"pos_actual": [0.25, 0.15, 0.05],
                               "quat_actual": [1.0, 0.0, 0.0, 0.0]}}}
        ok = a.begin_follow(tel, "c", np.array([0.40, 0.10, 0.0]))
        self.assertTrue(ok)
        self.assertEqual(a.state, D.FOLLOW)
        self.assertFalse(a.retreating)
        np.testing.assert_allclose(a.stream, [0.25, 0.15, 0.05],
                                   err_msg="the ramp must start at the HAND, "
                                           "not at the cube or the station")

    def test_the_startup_staircase_is_never_reacquirable(self):
        """The certified startup schedule may not be interrupted: the
        re-acquire gate in the main loop keys off `retreating`, which only a
        FOLLOW exit sets and start_homing clears."""
        a = _arm("front")
        a.start_homing([1, 2], 0.0)
        self.assertFalse(a.retreating,
                         "a fresh staircase claims to be a retreat")
        src = pathlib.Path(D.__file__).read_text()
        self.assertIn("if a.retreating:", src,
                      "the mid-retreat re-acquire gate is gone")

    def test_one_face_tracking_is_the_default(self):
        """User 08-11, after the first hardware run: '1 face is ok to track'.
        A hand-held cube hides faces by construction, and the frozen-goal
        rhythm read as broken. The strict 2-face bar stays reachable via
        --follow-min-faces 2."""
        self.assertEqual(D.FOLLOW_MIN_FACES, 1)
        self.assertEqual(D.Args().follow_min_faces, 1)

    def test_the_leave_verdict_names_the_limit_that_fired(self):
        """08-11 hardware: a cube lifted over the z ceiling was reported as
        '0.38 m out, past the reach envelope' — a radius well inside the band.
        The verdict must name the violated limit, not print the radius for
        every envelope failure."""
        high = [0.40, 0.10, D.OP.TARGET_Z_MAX_M + 0.10]     # radius fine, z not
        a = _arm("front", state=D.FOLLOW, key="c",
                 goal=np.array([0.40, 0.10, 0.0]),
                 stream=np.array([0.40, 0.10, 0.0]))
        rig = _Rig({"c": (high, 3, 0.0)})
        why = None
        for _ in range(D.FOLLOW_OUT_TICKS):
            _slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT,
                                       min_faces=2)
        self.assertIsNotNone(why)
        self.assertIn("z=", why, f"the z violation was hidden: {why}")
        self.assertNotIn("past the reach envelope", why,
                         "an in-band radius was blamed for a z violation")


@unittest.skipIf(D is None, _SKIP)
class DemoContractTests(unittest.TestCase):
    """It is a demo: no grasp, no walk, no silence."""

    @staticmethod
    def _code() -> str:
        """The tool's source with its module docstring stripped — the prose
        legitimately NAMES the things the demo refuses to do ("never binds
        9873", "no ee_action"), and a scan that cannot tell a promise from a
        call would either fail on the promise or pass on a real one."""
        with open(D.__file__) as fh:
            return fh.read().split('"""', 2)[2]

    def test_nothing_in_the_tool_can_emit_a_gripper_command(self):
        code = self._code()
        self.assertNotIn("ee_action", code)
        self.assertNotIn("hand_grab", code)

    def test_nothing_in_the_tool_can_bind_the_nav_socket(self):
        code = self._code()
        self.assertNotIn("9873", code)
        self.assertNotIn("nav_cmd", code)
        # ...and exactly one socket IS bound: the arm/gaze publisher.
        self.assertEqual(code.count("NNGPublisher("), 1)
        self.assertIn('NNGPublisher("tcp://*:9874")', code)

    def test_a_parked_arm_still_gets_a_slot_every_tick(self):
        """real_env's arm-silence failsafe fires at 0.5 s and crawls the arms
        to default at 0.125 rad/s. A settled station must keep being
        re-commanded — it is a FIXED posture, so re-sending it cannot walk the
        arm downhill the way republishing live encoders would."""
        a = _arm("front", state=D.HOME)
        slot = D.joint_slot(a.arm, D.station_posture(a.arm, a.home_idx))
        self.assertEqual(slot["joint_pos"],
                         [float(v) for v in OP.STAGE_JOINTS["front"]])

    def test_two_arms_may_not_share_a_station(self):
        """Same station == same sector == both arms latching one cube.

        Tested on the pure validator rather than by spawning the CLI: a fresh
        interpreter has to re-import the entire perception stack
        (humanoid_monitor -> camera_tag_detector), which fails for reasons that
        have nothing to do with this guard and made the check flaky.
        """
        why = D.validate_homes({"left": "front", "right": "front"})
        self.assertIsNotNone(why)
        self.assertIn("same sector", why.lower())
        self.assertIsNotNone(D.validate_homes({"left": "nope", "right": "rear"}))
        self.assertIsNone(D.validate_homes({"left": "front", "right": "rear"}))
        self.assertIsNone(D.validate_homes({"left": "front", "right": "side"}))

    def test_the_follow_quaternion_is_the_one_the_ik_is_given(self):
        """TARGET_MAX_RADIUS_M was measured for the CLEAN reach — EE within
        5 deg of NEUTRAL_QUAT — so the envelope and the commanded orientation
        have to be the same assumption. Since 08-11 the follow RAMPS to that
        quat instead of jumping, but neutral stays the default and the ramp's
        terminus."""
        self.assertEqual(D.cart_slot(np.zeros(3))["ee_quat"],
                         list(OP.NEUTRAL_QUAT))

    def test_the_follow_launch_does_not_whip_the_wrist(self):
        """08-11 hardware ("fetch out like a rocket"): the published quat used
        to jump station-pose -> neutral in ONE packet while the position
        crawled, and mink reoriented the wrist at its full 2.5 rad/s internal
        slew. The first FOLLOW packet must command the hand's own MEASURED
        orientation (zero reorientation), and every later packet may rotate
        at most FOLLOW_ORI_RATE_RAD_S * dt further toward neutral."""
        start = np.array([math.cos(0.6), math.sin(0.6), 0.0, 0.0])  # 68.8 deg
        a = _arm("front", state=D.FOLLOW, key="c",
                 goal=np.array([0.40, 0.10, 0.0]),
                 stream=np.array([0.30, 0.10, 0.0]))
        a.quat_stream = start.copy()
        rig = _Rig({"c": ([0.40, 0.10, 0.0], 3, 0.0)})
        prev = start.copy()
        for _ in range(3):
            slot, why = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT,
                                      min_faces=2)
            self.assertIsNone(why)
            q = np.asarray(slot["ee_quat"])
            step = 2.0 * math.acos(min(1.0, abs(float(np.dot(prev, q)))))
            self.assertLessEqual(
                step, D.FOLLOW_ORI_RATE_RAD_S * D.PUB_DT + 1e-6,
                "the published quat rotated faster than the ramp allows")
            prev = q
        # and the ramp genuinely arrives at neutral, not merely near it
        for _ in range(200):
            slot, _ = D.follow_slot(a, rig, now=0.0, dt=D.PUB_DT, min_faces=2)
        np.testing.assert_allclose(
            np.abs(slot["ee_quat"]), np.abs(np.asarray(OP.NEUTRAL_QUAT)),
            atol=1e-6, err_msg="the orientation ramp never reaches neutral")

    def test_slerp_toward_is_shortest_arc_and_bounded(self):
        qa = np.array([1.0, 0.0, 0.0, 0.0])
        qb_neg = -np.array([math.cos(0.1), math.sin(0.1), 0.0, 0.0])
        out = D.slerp_toward(qa, qb_neg, max_step_rad=1.0)
        self.assertGreater(
            abs(float(np.dot(out, qa))), 0.99,
            "q/-q double cover was not handled — the slerp took the long way")
        big = np.array([math.cos(0.8), 0.0, math.sin(0.8), 0.0])
        out = D.slerp_toward(qa, big, max_step_rad=0.05)
        step = 2.0 * math.acos(min(1.0, abs(float(np.dot(qa, out)))))
        self.assertLessEqual(step, 0.05 + 1e-9)


if __name__ == "__main__":
    unittest.main()
