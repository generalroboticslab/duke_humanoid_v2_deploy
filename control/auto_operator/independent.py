"""Symmetric per-arm task scheduler.

This module replaces the primary/secondary asymmetry when
``AutoOperator.independent_arms`` is enabled.  Left and right each own the
same :class:`ArmTask` lifecycle.  Shared resources are deliberately narrow:

* the base chooses one not-yet-reachable assignment to walk toward;
* every other resource is PER-ARM since the 07-20 wire protocol change and
  the 07-21 PHASE 2 unlock: each side of one packet independently carries a
  joint (staircase) or Cartesian (reach/hold) slot, so two staircases, two
  Cartesian reaches, or one of each all run in the same tick.

Waiting for a shared resource never advances the waiting arm's stream or
timeout.  Phase 2 safety rests on the dual-staircase clearance audit: with a
fixed base an arm's distance to the trunk depends on that arm's 7 joints
only, so a peer staircase adds zero body risk, and the two arms' entire
staging-reachable sets are separated by the mid-sagittal plane (>= 239.9 mm
guaranteed, 289 mm measured minimum across all 16 hop pairs, bare / loaded /
6 deg sag).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from . import arbitration as ao_arbitration
from . import servo as ao_servo
from .models import ArmTask, ArmTaskPhase, ServoState


ARMS = ("left", "right")


def _K(op):
    return sys.modules[type(op).__module__]


def _finite_vec3(value) -> bool:
    if value is None:
        return False
    arr = np.asarray(value, dtype=float)
    return arr.shape == (3,) and bool(np.all(np.isfinite(arr)))


def task_manipulation_active(op) -> bool:
    return any(t is not None and t.phase in (
        ArmTaskPhase.WAIT_STOP, ArmTaskPhase.STAGING,
        ArmTaskPhase.REACHING) for t in op._arm_tasks.values())


def task_tracks_active(op) -> bool:
    return any(t is not None and t.track is not None
               for t in op._arm_tasks.values())


def cartesian_packet_ready(op) -> bool:
    """Pure preflight for the two-slot Cartesian packet.

    ``held_update`` historically mutates lift/park/re-extension ramps before
    ``build_arm_packet`` discovers that the *other* slot has no finite value.
    Independent mode calls this before held_update and freezes those mutations
    unless every slot already has either an explicit owner value or a finite
    measured/ongoing-home fallback. No ramp state is changed here.
    """
    if op._reach_tm is not None or any(
            h["track"] is not None for h in op._held.values()):
        return False
    for arm in ARMS:
        h = op._held.get(arm)
        if h is not None:
            if not h["home"]:
                source = h.get("restream")
                if source is None:
                    source = h["target"] if h["bias"] is None \
                        else h["target"] - h["bias"]
                if not _finite_vec3(source):
                    return False
                continue
            if h["track"] is None and h["sector_idx"] != op._home_idx:
                if not _finite_vec3(op._ee_act.get(arm)):
                    return False
                continue
            if _finite_vec3(op._home_stream.get(arm)) \
                    or _finite_vec3(op._ee_act.get(arm)):
                continue
            return False

        task = op._arm_tasks.get(arm)
        if task is not None and task.phase == ArmTaskPhase.REACHING:
            if not _finite_vec3(task.stream):
                return False
            continue
        if op._sec is not None and op._sec["arm"] == arm \
                and _finite_vec3(op._sec["stream"]):
            continue
        if _finite_vec3(op._home_stream.get(arm)) \
                or _finite_vec3(op._ee_act.get(arm)):
            continue
        return False
    return True


def hold_cartesian_motion_active(op) -> bool:
    """Whether a hold needs Cartesian wire time before a new staircase.

    A stationary hold can be frozen safely inside a joint packet.  A lift,
    re-extension, carried-home park, or direct retract is different: starting
    a new task staircase would suppress its packet while its state machine
    continued to advance.  Give that already-started motion priority.
    """
    for arm, h in op._held.items():
        if h["track"] is not None:
            continue
        if h.get("restream") is not None or h["grasp"] == "lifting":
            return True
        if h["grasp"] == "grasped" and h["sector_idx"] == op._home_idx:
            # The next held_update starts the direct carried-home ramp.
            return True
        if h["home"] and h["sector_idx"] == op._home_idx:
            actual = op._ee_act.get(arm)
            if actual is not None:
                actual = np.asarray(actual, dtype=float)
                if actual.shape == (3,) and np.all(np.isfinite(actual)) and \
                        float(np.linalg.norm(
                            actual - np.asarray(op.rest_ee[arm], dtype=float))) \
                        > _K(op).HOME_SETTLE_TOL_M:
                    return True
    return False


def _arm_for_pos(op, pos) -> str | None:
    y = float(pos[1])
    if abs(y) <= _K(op).MIDLINE_MARGIN_M:
        return None
    return "left" if y > 0.0 else "right"


def _wrong_side(op, pos, arm: str) -> bool:
    """Cube is in the OTHER arm's exclusive half-space (beyond the midline
    margin). 07-26 deadlock audit: ownership checks used to demand
    `_arm_for_pos(...) == arm`, which returns None inside the +/-3 cm midline
    band — and the journey steering law's attractor is exactly bearing 0, so a
    pursued cube was driven INTO the band where no arm could assign, stay
    reachable, or latch (permanent stall). The band is now a shared zone with
    hysteresis: either arm may keep serving a cube there (same pattern the
    hold release rule at arms.py already uses); only a cube clearly across
    the midline disqualifies."""
    y = float(pos[1])
    m = _K(op).MIDLINE_MARGIN_M
    return (y < -m) if arm == "left" else (y > m)


def _route_enabled(op, pos) -> bool:
    sector = op._sector(pos)
    target = op._standoff_point(pos)
    direct = op.side_home and \
        op._staged_target_idx(sector, target) == op._home_idx
    return direct or (_K(op).SECTOR_REACH_ENABLED[sector]
                      and op.hold_after_reach)


def _pending(op):
    return [c for c in op.cubes.values()
            if not c.reached and c.held_by is None
            and getattr(c, "assigned_to", None) is None]


def ensure_mission_ready(op, now: float) -> bool:
    """Preserve the legacy zero/open and measured-home gates."""
    if op.grasp and not op._ee_chain_dead:
        if not op._hands_opened or (
                op._zeros_t is not None
                and (now - op._zeros_t) < _K(op).GRASP_ZERO_GRACE_S):
            if now - getattr(op, "_zero_wait_diag_t", -1e9) > 3.0:
                op._zero_wait_diag_t = now
                print("[grasp] independent arms wait for both grippers to "
                      "finish zeroing OPEN")
            return False

    if op._home_settled:
        return True
    if op._home_wait_t0 is None:
        op._home_wait_t0 = now
    near = True
    for arm in ARMS:
        if arm in op._held:
            continue
        act = op._ee_act.get(arm)
        # Journey pre-walk phases keep the chest tuck (power-on ≈ tuck), so
        # "settled" is judged against the tuck, not side-home — arms never
        # detour to the side before the first stop (07-21 user).
        # (CHEST-HOME needs no case here: op.rest_ee IS the chest tuck there.)
        tgt = _journey_walk_ee(op, arm) \
            if getattr(op, "_journey_walk_pose", False) \
            else np.asarray(op.rest_ee[arm], float)
        if act is None or not np.all(np.isfinite(np.asarray(act, float))) \
                or float(np.linalg.norm(np.asarray(act, float) - tgt)) \
                > _K(op).HOME_SETTLE_TOL_M:
            near = False
            break
    if near:
        op._home_settled = True
        print("[home] both arms settled — independent tasks may begin")
        return True
    if (now - op._home_wait_t0) > _K(op).HOME_SETTLE_TIMEOUT_S:
        op._home_settled = True
        print(f"[home] WARNING: arms did not settle within "
              f"{_K(op).HOME_SETTLE_TIMEOUT_S:.0f}s; proceeding with the "
              "existing fail-open policy")
        return True
    if now - getattr(op, "_home_wait_diag_t", -1e9) > 3.0:
        op._home_wait_diag_t = now
        print("[home] independent tasks wait for both arms to settle")
    return False


def _sync_peer_pending(op, arm: str) -> bool:
    """SYNC-REACH helper: True while any OTHER pending cube's arm is not yet
    latch-ready (unassigned, or its task still ASSIGNED). Held/reached cubes
    never block — the second round proceeds solo."""
    for k, c in op.cubes.items():
        if c.held_by is not None or c.reached:
            continue
        t = next((t for t in op._arm_tasks.values()
                  if t is not None and t.key == k), None)
        if t is None:
            return True
        if t.arm != arm and t.phase == ArmTaskPhase.ASSIGNED:
            return True
    return False


def assign_tasks(op, now: float) -> None:
    """Give each free arm its own same-side cube, without a primary role."""
    # A legacy/held staircase already owns the global joint wire.  Let it land
    # before creating assignments whose base-frame histories would otherwise
    # age while they cannot act.
    if op._reach_tm is not None or any(
            h["track"] is not None for h in op._held.values()):
        return
    for arm in ARMS:
        if op._arm_tasks[arm] is not None or arm in op._held:
            continue
        candidates = []
        peer = "right" if arm == "left" else "left"
        for cube in _pending(op):
            if not cube.confirmed(now) or _wrong_side(op, cube.pos, arm):
                continue
            # 07-27 second review: the ASSIGN_STALL cancel had no memory, so
            # a band cube whose gate persistently crowns the NON-natural arm
            # livelocked — the natural arm re-claimed after every 20 s stall
            # while its priority blocked the gate-favored peer forever. The
            # stall record breaks the cycle two ways: an arm that has stalled
            # on a cube MORE than an available peer steps aside (the peer
            # gets its try), and the natural-priority rule below is void once
            # the natural arm has already stalled on that cube.
            stalls = getattr(op, "_assign_stalls", {}).get(cube.key, {})
            if stalls.get(arm, 0) > stalls.get(peer, 0) \
                    and op._arm_tasks.get(peer) is None \
                    and peer not in op._held \
                    and not _wrong_side(op, cube.pos, peer):
                continue          # the less-stalled peer tries first
            # 07-26 review: MIDLINE-BAND service rules. The shared band killed
            # the dead-zone deadlock, but unconstrained it let (a) the
            # iteration-order arm claim a band cube its gate will never crown
            # and (b) BOTH arms serve band cubes 4 cm apart with converging
            # grippers. Rules: a band cube belongs to its natural-sign arm
            # while that arm is free; and no arm may take a band cube while
            # the peer's current business is also inside the band.
            if abs(float(cube.pos[1])) <= _K(op).MIDLINE_MARGIN_M:
                peer_task = op._arm_tasks.get(peer)
                peer_hold = op._held.get(peer)
                peer_target = None
                if peer_task is not None:
                    pc = op.cubes.get(peer_task.key)
                    peer_target = None if pc is None else pc.pos
                elif peer_hold is not None:
                    peer_target = peer_hold.get("target")
                if peer_target is not None and \
                        abs(float(peer_target[1])) <= _K(op).MIDLINE_MARGIN_M:
                    continue          # never two band services at once
                natural = "left" if float(cube.pos[1]) >= 0.0 else "right"
                if arm != natural and peer_task is None \
                        and peer_hold is None \
                        and stalls.get(natural, 0) == 0:
                    continue          # the free natural arm takes it
            if getattr(op, "journey", False):
                # JOURNEY (07-21 review CRITICAL fix): far cubes are the POINT
                # of the mission — TARGET_MAX_RADIUS_M is a standing-reach
                # rule and rejected every walk target, so the journey never
                # walked at all. Admit any confirmed same-side cube once the
                # SURVEY is complete; _latch_task independently re-validates
                # the envelope on the median at the stop point.
                if op._survey_order is None:
                    continue
            elif not op._target_safe(cube.pos) \
                    or not op._target_safe(op._standoff_point(cube.pos)) \
                    or not _route_enabled(op, cube.pos):
                continue
            candidates.append(cube)
        if not candidates:
            continue
        # Stalled cubes sort LAST so an arm serves its clean work first and a
        # zero-progress cube cannot monopolize the arm's attention.
        _stall_rec = getattr(op, "_assign_stalls", {})
        cube = min(candidates, key=lambda c: (
            sum(_stall_rec.get(c.key, {}).values()),
            c.first_seen, float(np.hypot(c.pos[0], c.pos[1]))))
        cube.assigned_to = arm
        op._arm_tasks[arm] = ArmTask(arm=arm, key=cube.key,
                                      assigned_t=now)
        print(f"[task:{arm}] assigned {cube.key}; waiting for this arm's "
              "reachability verdict")


def cancel_task(op, arm: str, reason: str) -> None:
    """Release only one arm's assignment; the peer is untouched."""
    task = op._arm_tasks.get(arm)
    if task is None:
        return
    cube = op.cubes[task.key]
    if getattr(cube, "assigned_to", None) == arm:
        cube.assigned_to = None
    op._arm_tasks[arm] = None
    op._ind_joint_owners.discard(arm)
    print(f"[task:{arm}] released {task.key}: {reason}")


def reachability_sample(op, task: ArmTask, pos) -> bool:
    """Debounced per-arm reachability; cached gate samples never add hits."""
    if not op._target_safe(pos) or _wrong_side(op, pos, task.arm) \
            or not op._target_safe(op._standoff_point(pos)):
        # ^ standoff validation (07-26 review): _latch_task re-validates the
        #   standoff point; judging reachability without it promoted cubes to
        #   WAIT_STOP that could never latch — the cancel->reassign cycle
        #   juddered the arms side<->tuck next to the cube forever.
        task.reach_hits = 0
        return False
    dist = float(np.hypot(pos[0], pos[1]))
    if dist <= _K(op).GEOMETRIC_FLOOR_M:
        task.reach_hits = _K(op).GATE_CONSECUTIVE
        return True
    if op.gate is None:
        ok = dist <= _K(op).DYN_OUTREACH_M
        task.reach_hits = _K(op).GATE_CONSECUTIVE if ok else 0
        return ok
    score, best, fresh = op._gate_eval(pos)
    if fresh:
        op._ind_gate_inferred = True
    ok = bool(score > op.gate.SCORE_THRESHOLD) and best == task.arm
    if fresh:
        task.reach_hits = task.reach_hits + 1 if ok else 0
    return task.reach_hits >= _K(op).GATE_CONSECUTIVE


def _latch_task(op, task: ArmTask, now: float) -> bool | None:
    """Latch a median target.

    ``None`` means the target is valid but a direct Cartesian leg still lacks
    a finite measured EE seed; remain in WAIT_STOP and retry. ``False`` is a
    real validation failure and releases the assignment.
    """
    cube = op.cubes[task.key]
    window = [(s.pos, s.port) for s in cube.hist
              if (now - s.t) < _K(op).CUBE_LATCH_WINDOW_S]
    if not window:
        return False
    median = np.median(np.stack([p for p, _ in window]), axis=0)
    if _wrong_side(op, median, task.arm) or not op._target_safe(median):
        return False
    target = op._standoff_point(median)
    # Unlike the legacy primary, validate the point that will actually be
    # published after standoff/aim-up as well as the raw cube measurement.
    if not op._target_safe(target) or not _route_enabled(op, median):
        return False
    ports = [p for _, p in window if p is not None]
    port = max(set(ports), key=ports.count) if ports else cube.pos_port
    sector = op._sector(target)
    sector_idx = op._staged_target_idx(sector, target)
    seed = None
    if sector_idx == op._home_idx:
        seed = op._ee_act.get(task.arm)
        if seed is not None:
            seed = np.asarray(seed, dtype=float)
            if seed.shape != (3,) or not np.all(np.isfinite(seed)):
                seed = None
        if seed is None:
            return None
    task.target = target
    task.sector = sector
    task.sector_idx = sector_idx
    task.port = port
    getattr(op, "_latch_fails", {}).pop(task.key, None)  # success resets budget
    task.stream = None if seed is None else seed.copy()
    task.reach_t0 = now
    task.reach_elapsed = 0.0
    task.servo = ServoState(accept_t=now)
    if task.sector_idx != op._home_idx:
        task.track = op._new_track(task.arm, op._home_idx,
                                   task.sector_idx, now,
                                   replant=op.stage_sectors)
        # held_update ran earlier in this tick and may have changed a held
        # peer's Cartesian target without a ramp (live retrack).  Let the
        # combined Cartesian packet publish once before this new track can
        # switch the global wire to joint mode.
        task.joint_not_before = now + _K(op).DT
        task.phase = ArmTaskPhase.STAGING
        print(f"[task:{task.arm}] {task.key} latched; staging "
              f"{ _K(op).TRACK[op._home_idx] }→{task.sector}")
    else:
        task.track = None
        task.phase = ArmTaskPhase.REACHING
        print(f"[task:{task.arm}] {task.key} latched; Cartesian reach starts")
    return True


def prepare_tasks(op, now: float) -> None:
    """Advance both assignments from reachable to a latched arm task."""
    # gate.score() costs 17-37 ms on the deployment CPU. Two fresh inferences
    # in one 50 ms operator tick recreate the measured gaze/control jitter, so
    # alternate first service and permit at most one fresh learned inference
    # per tick. Geometric/no-gate verdicts still evaluate both arms together.
    start = getattr(op, "_ind_gate_rr", 0)
    order = ARMS[start:] + ARMS[:start]
    op._ind_gate_rr = (start + 1) % len(ARMS)
    op._ind_gate_inferred = False
    for arm in order:
        task = op._arm_tasks.get(arm)
        if task is None:
            continue
        cube = op.cubes[task.key]
        # Before latch, a stale body-frame target is released for re-scan.
        # After latch, especially mid-staircase, cancellation would switch the
        # global wire out of joint mode from an unconfirmed posture. Keep the
        # median target and let the normal track/reach timeout or later HOLD
        # lost-target machinery resolve it.
        if task.phase in (ArmTaskPhase.ASSIGNED, ArmTaskPhase.WAIT_STOP) and \
                (now - cube.last_seen) > _K(op).DET_LOST_RESCAN_S:
            cancel_task(op, arm, "target lost before reach")
            continue
        if task.phase == ArmTaskPhase.ASSIGNED:
            if not cube.fresh(now, _K(op).DET_STALE_S):
                continue
            # JOURNEY: the radius envelope is a standing-reach rule — a far
            # cube is exactly what we walk toward. Half-space ownership still
            # applies; the envelope is re-validated at the latch (07-21
            # review: this second check site cancelled every far task the
            # assign-site fix admitted).
            _envelope_ok = op._target_safe(cube.pos) \
                if not getattr(op, "journey", False) else True
            if _wrong_side(op, cube.pos, arm) or not _envelope_ok:
                cancel_task(op, arm, "target left this arm's safe half-space")
                continue
            if op.gate is not None and op._ind_gate_inferred and \
                    float(np.hypot(cube.pos[0], cube.pos[1])) \
                    > _K(op).GEOMETRIC_FLOOR_M:
                continue
            reachable = reachability_sample(op, task, cube.pos)
            # 07-26 review: standing missions only — an assignment that makes
            # zero reachability progress must not pin its arm forever (a band
            # cube whose gate crowns the peer used to stall the claiming arm
            # for the whole mission). Journey is exempt: ASSIGNED lasts a
            # whole walk leg there by design.
            if not reachable and not getattr(op, "journey", False) \
                    and task.reach_hits == 0 \
                    and now - task.assigned_t > _K(op).ASSIGN_STALL_S:
                # 07-27 second review: give the stall a MEMORY (mirrors
                # _latch_fails). assign_tasks reads the record to rotate the
                # cube to the less-stalled arm and to void the band natural-
                # priority; once every arm that can legally serve the cube
                # has stalled LATCH_FAIL_SKIP_N times, the cube is skipped
                # LOUDLY — in a standing mission neither cube nor base moves,
                # so further zero-progress claims can never turn reachable.
                stalls = getattr(op, "_assign_stalls", None)
                if stalls is None:
                    stalls = op._assign_stalls = {}
                rec = stalls.setdefault(task.key, {})
                rec[arm] = rec.get(arm, 0) + 1
                servable = [a for a in ARMS
                            if not _wrong_side(op, cube.pos, a)]
                if servable and all(rec.get(a, 0) >= _K(op).LATCH_FAIL_SKIP_N
                                    for a in servable):
                    cube.reached = True   # out of _pending: nav/gaze stop
                    print(f"[task:{arm}] {task.key} SKIPPED — every arm that "
                          "can serve it stalled "
                          f"{_K(op).LATCH_FAIL_SKIP_N}x with zero "
                          "reachability progress (restart to retry)")
                cancel_task(op, arm, "no reachability progress "
                            f"in {_K(op).ASSIGN_STALL_S:.0f}s — re-queued")
                continue
            if reachable and getattr(op, "journey", False) and \
                    float(np.hypot(cube.pos[0], cube.pos[1])) \
                    > _K(op).JOURNEY_ARRIVE_M:
                # Journey "stop closer" (07-21 user): the gate fires early
                # (0.40+); keep creeping at the schedule floor until the cube
                # is inside JOURNEY_ARRIVE_M, THEN stop. Hits stay latched.
                continue
            if reachable:
                # Real progress: forget past stalls so a cube that BECAME
                # reachable (it was moved) regains normal priority.
                getattr(op, "_assign_stalls", {}).pop(task.key, None)
                task.phase = ArmTaskPhase.WAIT_STOP
                print(f"[task:{arm}] {task.key} is reachable; waiting for base stop")
        if task.phase == ArmTaskPhase.WAIT_STOP:
            # Journey: the tuck-release + stillness settle takes ~6 s, so the
            # 0.5 s staleness demotion would bounce WAIT_STOP<->ASSIGNED under
            # any detection flicker at the stop point — use the 3 s rescan
            # horizon instead (07-21 review hysteresis fix).
            _stale_h = _K(op).DET_LOST_RESCAN_S if getattr(op, "journey", False) \
                else _K(op).DET_STALE_S
            # DECODE AGE, NOT PACKET AGE (audit 2026-08-06, D12). CubeTrack.fresh
            # reads last_seen, stamped on every arriving packet, while the latch
            # window below ages Sightings on their CAPTURE stamps. Since the
            # capture-stamp change those two clocks differ by up to
            # DETECTION_RETENTION_S, opening a band where the cube is "fresh"
            # and the window is EMPTY — and an empty window returns False from
            # _latch_task, which releases the arm assignment and can set
            # cube.reached, abandoning a cube on decode timing alone. Gate and
            # window must age the same thing.
            _hist = cube.hist
            _decoded = float(_hist[-1].t) if _hist else -1e9
            if not (cube.pos is not None and (now - _decoded) < _stale_h):
                task.phase = ArmTaskPhase.ASSIGNED
                task.reach_hits = 0
                continue
            if abs(op._vx) < 0.02 and abs(op._wz) < 0.02:
                if getattr(op, "sync_reach", False) and \
                        _sync_peer_pending(op, arm):
                    # SYNC-REACH (07-21 user, video demo): hold at the latch
                    # gate until every other pending cube's arm is ready too —
                    # then all latch back-to-back and reach in the same packet.
                    if now - op._sync_wait_t > 3.0:
                        op._sync_wait_t = now
                        print(f"[sync] {arm} ready at the latch gate — "
                              "waiting for the other cube's arm")
                    continue
                if getattr(op, "journey", False):
                    # JOURNEY (07-21): the command ramp at zero is not the
                    # robot standing still. Latch only after MEASURED base
                    # stillness sustains, the walking tuck is released, and
                    # the reaching arm has glided back to its side home —
                    # so the median samples a quiet robot and the staircase
                    # starts from its audited basin.
                    op._journey_walk_pose = False
                    if not journey_base_still(op, now):
                        continue
                    act = op._ee_act.get(arm)
                    if act is None or float(np.linalg.norm(
                            np.asarray(act, dtype=float)
                            - np.asarray(op.rest_ee[arm], dtype=float))) \
                            > _K(op).HOME_SETTLE_TOL_M + 0.04:
                        continue
                latched = _latch_task(op, task, now)
                if latched is False:
                    # 07-26 review: latch-failure budget. Without it an
                    # unlatchable cube (median fails the safety/standoff
                    # checks at the stop point) cycles cancel->reassign->
                    # latch-fail forever, physically juddering the arms
                    # side<->tuck next to the cube. After N consecutive
                    # failures the cube is marked unserviceable and skipped
                    # LOUDLY; a successful latch resets its count.
                    fails = getattr(op, "_latch_fails", None)
                    if fails is None:
                        fails = op._latch_fails = {}
                    fails[task.key] = fails.get(task.key, 0) + 1
                    if fails[task.key] >= _K(op).LATCH_FAIL_SKIP_N:
                        cube.reached = True   # out of _pending: nav/gaze stop
                        print(f"[task:{arm}] {task.key} SKIPPED after "
                              f"{fails[task.key]} consecutive latch failures "
                              "— cube unserviceable at this stop point "
                              "(restart to retry)")
                    cancel_task(op, arm, "latch window or final target was unsafe")


def _hold_blocks_base(op) -> bool:
    for arm, h in op._held.items():
        if h["track"] is not None or h.get("restream") is not None:
            return True
        if op.grasp and not op._ee_chain_dead \
                and h["grasp"] not in ("carried", "failed"):
            # A direct/front hold whose cube disappeared before the gripper
            # fired retracts to its measured home.  Once it is physically
            # settled there, keeping the shared base stopped forever serves
            # no safety purpose and prevents the peer arm from pursuing its
            # own assignment.  The hold itself remains live and may re-extend
            # if its cube reappears.
            actual = op._ee_act.get(arm)
            if h.get("resume_pending", False):
                return True
            safely_retracted = h["grasp"] is None and h["home"] \
                and h["sector_idx"] == op._home_idx \
                and _finite_vec3(actual) \
                and float(np.linalg.norm(
                    np.asarray(actual, dtype=float)
                    - np.asarray(op.rest_ee[arm], dtype=float))) \
                <= _K(op).HOME_SETTLE_TOL_M
            if not safely_retracted:
                return True
    return False


def journey_survey(op, now: float) -> bool:
    """JOURNEY gate (07-21): navigation stays locked until EVERY registered
    cube has a CONFIRMED sighting, then the near-first service order is
    latched ONCE from that survey standpoint. Positions rot in the body frame
    as the base moves, so the survey is only trusted for ORDERING — each leg
    re-acquires its cube live before walking."""
    if op._survey_order is not None:
        return True
    # Held/reached cubes are excluded: a cube in the gripper never confirms
    # again (fingers occlude the tags) and must not hold the survey hostage
    # (07-21 review: pre-survey grasp deadlocked the mission forever).
    missing = [k for k, c in op.cubes.items()
               if c.held_by is None and not c.reached and not c.confirmed(now)]
    if missing:
        if now - op._survey_warn_t > 10.0:
            op._survey_warn_t = now
            print(f"[journey] surveying — waiting for {missing} "
                  "(cameras scanning; robot stands)")
        return False
    order = sorted(op.cubes,
                   key=lambda k: float(np.hypot(*op.cubes[k].pos[:2])))
    op._survey_order = order
    dists = {k: float(np.hypot(*op.cubes[k].pos[:2])) for k in order}
    print(f"[journey] survey complete — service order "
          + " -> ".join(f"{k} ({dists[k]:.2f} m)" for k in order))
    return True


def journey_base_still(op, now: float) -> bool:
    """REAL stop check: measured base velocity below the stillness thresholds,
    SUSTAINED (the command ramp reaching zero is not the robot being still —
    inertia and post-stop sway corrupt the latch median).

    07-21 review fixes: (a) the deployed v159b checkpoint has NO velocity
    estimator — real_env publishes base_lin_vel as hard ZEROS, so the lin-vel
    comparison is vacuous. Detect it (lin never nonzero) and fall back to the
    gyro check plus a LONGER fixed settle. (b) the timer is additionally reset
    by navigation whenever the base is commanded to move, so leg 1's stillness
    can never satisfy leg 2's arrival check."""
    K = _K(op)
    lin = op._base_vel
    if not op._journey_lin_seen and lin is not None and len(lin) >= 2 \
            and float(np.hypot(lin[0], lin[1])) > 1e-9:
        op._journey_lin_seen = True
    still = True
    if op._journey_lin_seen and lin is not None and len(lin) >= 2:
        still &= float(np.hypot(lin[0], lin[1])) < K.JOURNEY_STILL_VEL
    elif not op._journey_lin_seen and not op._journey_blind_warned:
        op._journey_blind_warned = True
        print("[journey] WARNING: base_lin_vel reads all zeros (checkpoint has "
              "no velocity estimator) — stillness gate = gyro + "
              f"{K.JOURNEY_STILL_S_BLIND:.1f}s fixed settle")
    if op._base_ang is not None and len(op._base_ang) >= 3:
        still &= abs(float(op._base_ang[2])) < K.JOURNEY_STILL_ANG
    if not still:
        # DEBIT, DO NOT NULL (08-07 audit). base_ang_vel arrives raw and
        # unfiltered, so one outlier sample used to discard the entire sustain
        # window — and window expiry is the leg's only measurement-driven SKIP,
        # i.e. a single spike could cost the cube. Pushing the start forward by
        # a bounded penalty keeps the gate's meaning (genuine motion produces
        # bad ticks continuously and still walks the credit to zero in a few
        # ticks) while a lone spike costs JOURNEY_STILL_DEBIT_S, not everything.
        # min(now, ...) stops the debit running the clock into the future.
        if op._journey_still_since is not None:
            op._journey_still_since = min(
                now, op._journey_still_since + K.JOURNEY_STILL_DEBIT_S)
        return False
    if op._journey_still_since is None:
        op._journey_still_since = now
    need = K.JOURNEY_STILL_S if op._journey_lin_seen else K.JOURNEY_STILL_S_BLIND
    return (now - op._journey_still_since) >= need


def _journey_walk_ee(op, arm: str) -> np.ndarray:
    ee = np.asarray(_K(op).JOURNEY_WALK_EE, dtype=float).copy()
    if arm == "right":
        ee[1] = -ee[1]
    return ee


def journey_arms_tucked(op) -> bool:
    """Free arms measured at the walking tuck — walking starts only after the
    tuck completes so the glide never overlaps gait onset. Held arms ride at
    REST with their cube and are exempt. Primary check is JOINT-space against
    the power-on posture (the tuck IS that posture since the 07-21 joint-glide
    change); EE-distance is the fallback when joint telemetry is absent."""
    jp = getattr(op, "_jpos", None)
    for i, arm in enumerate(("left", "right")):
        if arm in op._held:
            continue
        if jp is not None and len(jp) >= 27:
            seg = np.asarray(jp[13:20] if i == 0 else jp[20:27], dtype=float)
            tgt = _K(op).mirror_arm(_K(op).POWERON_JOINTS, arm)
            if not np.all(np.isfinite(seg)) or \
                    float(np.max(np.abs(seg - tgt))) > 2.5 * _K(op).JOURNEY_POSTURE_TOL_RAD:
                return False
            continue
        act = op._ee_act.get(arm)
        if act is None or not np.all(np.isfinite(np.asarray(act, dtype=float))):
            return False
        if float(np.linalg.norm(np.asarray(act, dtype=float)
                                - _journey_walk_ee(op, arm))) \
                > _K(op).JOURNEY_TUCK_TOL_M:
            return False
    return True


def journey_posture_payload(op, jpos) -> dict | None:
    """JOINT-SPACE side<->tuck glide (07-21 user: the Cartesian home_glide
    'looked weird' — and it was: straight-line EE + IK left the arm up to
    62 deg off the true power-on posture). Streams per-arm joint_pos toward
    the CURRENT posture goal (tuck = POWERON while _journey_walk_pose, else
    SIDE_HOME) for every free arm still away from it; the receiver's per-arm
    rate limiter does the motion, exactly like a staircase hop. Returns None
    when nothing needs to move (arms settled -> Cartesian keep-alive resumes,
    with the per-arm mode-flip reseed guarding the transition) or while any
    real staircase owns joint motion (unaudited combination)."""
    if not getattr(op, "journey", False):
        return None
    if jpos is None or len(jpos) < 27:
        return None
    if op._reach_tm is not None or task_tracks_active(op) or any(
            h["track"] is not None for h in op._held.values()):
        return None
    K = _K(op)
    goal = np.asarray(K.POWERON_JOINTS if op._journey_walk_pose
                      else K.SIDE_HOME_JOINTS, dtype=float)
    payload: dict = {}
    for i, arm in enumerate(("left", "right")):
        if arm in op._held:
            continue
        # 07-26 deadlock audit: an arm mid-STAGING/REACHING streams its own
        # Cartesian/staircase command — a posture joint slot must never fight
        # it (the chest-home docstring lists exactly this missing exclusion).
        # ASSIGNED and WAIT_STOP arms are free to glide.
        task = op._arm_tasks.get(arm)
        if task is not None and task.phase in (ArmTaskPhase.STAGING,
                                               ArmTaskPhase.REACHING):
            continue
        seg = np.asarray(jpos[13:20] if i == 0 else jpos[20:27], dtype=float)
        if not np.all(np.isfinite(seg)):
            continue
        # 07-26 review: CORRIDOR GUARD. The receiver lerps joint targets, so
        # the audited motion is the straight joint-space segment between
        # SIDE_HOME and POWERON — an arm ON that segment (mid-glide residual
        # ~0) may be driven. An arm found anywhere else (freed at an extended
        # reach pose, post-fault, hand-moved) must NOT be joint-lerped from
        # an unaudited pose: leave it to the Cartesian pin/home machinery.
        a = K.mirror_arm(K.SIDE_HOME_JOINTS, arm)
        b = K.mirror_arm(K.POWERON_JOINTS, arm)
        ab = b - a
        t = float(np.clip(np.dot(seg - a, ab) / max(float(np.dot(ab, ab)),
                                                    1e-9), 0.0, 1.0))
        if float(np.max(np.abs(seg - (a + t * ab)))) \
                > K.JOURNEY_POSTURE_CORRIDOR_RAD:
            continue
        tgt = K.mirror_arm(goal, arm)
        if float(np.max(np.abs(seg - tgt))) > K.JOURNEY_POSTURE_TOL_RAD:
            payload[arm] = {"joint_pos": [float(v) for v in tgt],
                            "rate": K.STAGE_RATE}
    return payload or None


def navigation_command(op, now: float) -> tuple[float, float]:
    """Shared base service for assigned tasks that are not reachable yet."""
    if task_manipulation_active(op) or task_tracks_active(op) \
            or _hold_blocks_base(op):
        op._journey_walk_pose = False
        return 0.0, 0.0
    if getattr(op, "journey", False) and not journey_survey(op, now):
        # Survey in progress: KEEP the chest tuck (set at mission start) —
        # arms stand pat instead of detouring to side-home and back (07-21).
        return 0.0, 0.0
    if getattr(op, "journey", False) and \
            any(h["grasp"] == "failed" for h in op._held.values()):
        # A failed-grasp hold is a plain extended hold: walking the next leg
        # would carry an outstretched arm through the world. Stand and shout.
        if not op._journey_fail_warned:
            op._journey_fail_warned = True
            print("[journey] a grasp FAILED — standing here (no further legs); "
                  "clear the gripper / restart to continue")
        op._journey_walk_pose = False
        return 0.0, 0.0
    waiting = [t for t in op._arm_tasks.values()
               if t is not None and t.phase == ArmTaskPhase.ASSIGNED]
    if not waiting:
        op._nav_task_key = None
        op._nav_leg_dir = None
        # Tuck hysteresis (07-21 review): detection flicker at range cancels
        # and reassigns tasks on a ~3 s cycle; dropping the tuck immediately
        # made the arms oscillate side-home<->tuck forever. Hold the tuck for
        # JOURNEY_TUCK_HOLD_S after the last served tick.
        if not getattr(op, "journey", False) or \
                now - op._journey_serve_t > _K(op).JOURNEY_TUCK_HOLD_S:
            op._journey_walk_pose = False
        return 0.0, 0.0
    if getattr(op, "journey", False):
        # JOURNEY: serve strictly in the latched near-first survey order.
        by_key = {t.key: t for t in waiting}
        task = next((by_key[k] for k in op._survey_order if k in by_key), None)
        if task is None:
            if now - op._journey_serve_t > _K(op).JOURNEY_TUCK_HOLD_S:
                op._journey_walk_pose = False
            return 0.0, 0.0
    else:
        task = min(waiting, key=lambda t: (
            op.cubes[t.key].first_seen,
            float(np.hypot(op.cubes[t.key].pos[0],
                           op.cubes[t.key].pos[1]))))
    cube = op.cubes[task.key]
    op.active = cube                  # diagnostics/backward-compatible status
    if not cube.fresh(now, _K(op).DET_STALE_S):
        return 0.0, 0.0
    if op._nav_task_key != task.key:
        op._nav_task_key = task.key
        op._nav_leg_dir = None
    x, y = float(cube.pos[0]), float(cube.pos[1])
    if op._nav_leg_dir is None:
        op._nav_leg_dir = 1.0 if abs(math.atan2(y, x)) <= math.pi / 2 else -1.0
    # 07-26 deadlock audit: steer at an aim point offset INTO the serving
    # arm's half-space. Plain pursuit servoes the cube onto bearing 0 — the
    # exact centre of the midline dead band, where the ownership checks
    # (assign/reachability/latch) used to disqualify it. With the offset the
    # equilibrium parks the cube at y ~= +/-JOURNEY_AIM_OFFSET_M, inside the
    # serving arm's half-space by construction. Arrival distance below still
    # uses the true cube position, not the aim point.
    aim_side = 1.0 if task.arm == "left" else -1.0
    # Per-direction offset (user 08-10): reverse walks straighter at the
    # smaller offset, forward keeps the table-face clearance. Reverse aims
    # JOURNEY_REVERSE_DRIFT_COMP_M right of the parking spot — the backward
    # gait's measured leftward settle drift lands the cube on it.
    if op._nav_leg_dir > 0:
        y_eq = aim_side * _K(op).JOURNEY_AIM_OFFSET_M
    else:
        y_eq = aim_side * _K(op).JOURNEY_AIM_OFFSET_BACK_M \
            - _K(op).JOURNEY_REVERSE_DRIFT_COMP_M
    bearing = math.atan2(y - y_eq, x)
    err = bearing if op._nav_leg_dir > 0 else op._wrap(bearing + math.pi)
    if getattr(op, "journey", False) and \
            abs(err) > _K(op).JOURNEY_FACE_TOL_RAD:
        # ARC INTO THE FACING (sim parity 08-08, MoverCfg.turn_cruise): while
        # unfaced the yaw command is FLOORED, not proportional — yaw only
        # tracks at cruise, and a sub-floor command is the measured dead band
        # (sim 08-02; hardware --wz bench 08-08). The base carves an arc at
        # cruise speed until the aim point sits inside the face tolerance,
        # then the ordinary trim cap below takes over.
        wz = math.copysign(
            float(np.clip(_K(op).K_STEER * abs(err),
                          _K(op).JOURNEY_WZ_UNFACED_FLOOR,
                          _K(op).JOURNEY_WZ_UNFACED_MAX)), err)
    else:
        wz = float(np.clip(_K(op).K_STEER * err,
                           -_K(op).WZ_MAX, _K(op).WZ_MAX))
    vx_mag = _K(op).CRUISE_VX
    if getattr(op, "journey", False):
        # Tuck the free arms at the power-on pose BEFORE gait onset (user
        # decision 07-21: POWERON to walk, side-home on arrival); stand until
        # the tuck completes. The home-glide machinery does the moving.
        op._journey_walk_pose = True
        op._journey_serve_t = now
        if not journey_arms_tucked(op):
            return 0.0, wz * 0.0
        # Distance-scheduled speed: cruise far out, ramp down into the
        # approach so the stop is gentle (arrival is judged by the gate).
        # 07-26 deadlock audit: BACKWARD legs get a higher floor — the gait
        # policy steps in place at a commanded -0.1 m/s, so a backward
        # approach used to stall short of the 0.34 m arrival gate forever.
        K = _K(op)
        dist = float(np.hypot(x, y))
        near_vx = K.JOURNEY_NEAR_VX_BACK if op._nav_leg_dir < 0 \
            else K.JOURNEY_NEAR_VX
        vx_mag = float(np.clip(K.JOURNEY_SLOW_K * (dist - K.JOURNEY_STOP_M),
                               near_vx, K.CRUISE_VX))
        # Driving: the stillness timer can never carry over to the next
        # arrival (07-21 review: leg 1's stale timer let leg 2 latch on the
        # first ramp-zero tick, mid-sway).
        op._journey_still_since = None
    return op._nav_leg_dir * vx_mag, wz


def _choose_joint_owners(op, now: float) -> list:
    """07-21 PHASE 2: EVERY staged arm walks — the exclusive token is gone.

    An arm already walking keeps walking (hops are non-preemptible); a newly
    staged arm joins as soon as its one-publish window (joint_not_before) has
    passed. Safety rests on the dual-staircase clearance audit (arm-vs-trunk
    is peer-independent with a fixed base; arm-vs-arm >= 239.9 mm by the
    mid-sagittal certificate) — not on serialization.

    07-28 WRIST REDESIGN — re-measured, still certified. The separating plane
    survives the new wrist at 204.3 mm nominal (was 239.9) and 139.0 mm under
    the audited +-6 deg sag, so the arms still cannot reach each other at any
    phase. Gated behind PARALLEL_STAIRCASE_CERTIFIED so a future geometry
    change can serialize the arms again by flipping one constant.
    """
    staged = [a for a in ARMS if op._arm_tasks[a] is not None
              and op._arm_tasks[a].track is not None]
    owners = [a for a in staged
              if a in op._ind_joint_owners
              or op._arm_tasks[a].joint_not_before <= now]
    if not _K(op).PARALLEL_STAIRCASE_CERTIFIED and len(owners) > 1:
        # Keep whoever already holds the token (hops are non-preemptible).
        held = [a for a in owners if a in op._ind_joint_owners]
        owners = held[:1] or owners[:1]
    op._ind_joint_owners = set(owners)
    return owners


def _hold_settled_station(op, task: ArmTask, telemetry: dict) -> dict:
    tm = task.track
    frozen = tm["frozen"] if tm is not None and tm["frozen"] is not None else \
        [float(v) for v in op._track_posture(
            "right" if task.arm == "left" else "left", op._home_idx)]
    return op._staged_joint_payload(
        task.arm, op._track_posture(task.arm, task.sector_idx), frozen)


def tick_task_motion(op, telemetry: dict, jpos, now: float) -> dict | None:
    """Advance granted task motion and compose both Cartesian arm slots."""
    # 07-20 PER-ARM PROTOCOL: a held-arm carry/retreat track owns only ITS
    # side of the wire; Cartesian REACHING slots merge alongside it in final
    # arbitration, so tasks keep advancing. The joint TOKEN stays exclusive —
    # while a held track runs, no task staircase may be granted (one joint
    # motion at a time remains the audited contract).
    # 07-21 PHASE 2: every staged arm steps this tick and their per-arm joint
    # slots MERGE into one packet (an arm can never own both a hold track and
    # a task, so slots never collide). A held arm's carry/retreat track no
    # longer blocks task staircases — it owns only its own side.
    joint_payload: dict = {}
    for owner in _choose_joint_owners(op, now):
        task = op._arm_tasks[owner]
        tm = task.track
        payload, status = op._step_toward(tm, jpos, now)
        if payload:
            joint_payload.update(payload)
        if tm["timed_out"] and not tm["seq"]:
            cancel_task(op, owner, "staging hop timed out and returned home")
            continue           # merge-and-continue: never drop a peer's slot
        if status != "done":
            continue
        actual = ((telemetry.get("ee") or {}).get(owner) or {}).get("pos_actual")
        if actual is not None:
            actual = np.asarray(actual, dtype=float)
            if actual.shape != (3,) or not np.all(np.isfinite(actual)):
                actual = None
        if actual is None:
            joint_payload.update(_hold_settled_station(op, task, telemetry))
            continue
        task.track = None
        task.phase = ArmTaskPhase.REACHING
        task.stream = actual.copy()
        task.reach_t0 = now
        task.reach_elapsed = 0.0
        op._ind_joint_owners.discard(owner)
        print(f"[task:{owner}] staging settled; Cartesian reach starts")

    positions: dict[str, np.ndarray] = {}
    proposals = []
    for arm in ARMS:
        task = op._arm_tasks.get(arm)
        if task is None or task.phase != ArmTaskPhase.REACHING:
            continue
        cube = op.cubes[task.key]
        if op.dynamic_track:
            upd = op._live_retrack(cube, arm, task.sector,
                                   task.target, now)
            if upd is not None and op._target_safe(upd):
                task.target = upd
        if op.visual_servo:
            ao_servo.task_servo_step(op, task, telemetry, jpos, now)
        cmd_prev = None if task.servo.cmd_prev is None else \
            task.servo.cmd_prev.copy()
        target = ao_servo.task_servo_target(op, task)
        if not op._target_safe(target):
            # Servo bias is bounded, but a nominal target may sit close enough
            # to an envelope face for that bounded correction to leave it.
            # Keep the independently validated open-loop point; never publish
            # an out-of-envelope correction.
            ao_servo.task_warn_once(
                task, "corrected-envelope",
                "visual-servo correction left the target envelope; using "
                "the validated open-loop target")
            task.servo.bias = np.zeros(3, dtype=float)
            target = np.asarray(task.target, dtype=float).copy()
            task.servo.cmd_prev = target.copy()
        actual = ((telemetry.get("ee") or {}).get(arm) or {}).get("pos_actual")
        if actual is not None:
            actual = np.asarray(actual, dtype=float)
            if actual.shape != (3,) or not np.all(np.isfinite(actual)):
                actual = None
        next_stream, published = op._gentle_approach(
            task.stream, target, actual)
        if published is None:
            task.servo.cmd_prev = cmd_prev
            continue
        positions[arm] = published
        next_elapsed = task.reach_elapsed + _K(op).DT
        done = actual is not None and \
            float(np.linalg.norm(np.asarray(actual) - target)) \
            < _K(op).REACH_OK_M and ao_servo.task_servo_ready(op, task)
        timed_out = next_elapsed > _K(op).REACH_TIMEOUT_S
        proposals.append((arm, task, cube, next_stream, next_elapsed,
                          target, done, timed_out, cmd_prev))

    # `_arm_packet` can reject the *whole* Cartesian packet when either arm
    # lacks a finite hold/idle value.  Commit no stream, timer, or ownership
    # transition until we know the combined two-slot packet is publishable.
    # NOTE: always compose the Cartesian packet, even with no REACHING task —
    # held arms and idle keep-alive slots must keep publishing while a peer
    # staircases (regression caught by the contract tests: returning early
    # silenced a held arm's final ramp point).
    packet = op._arm_packet(reach_positions=positions)
    if packet is None:
        for _arm, task, _cube, _stream, _elapsed, _target, \
                _done, _timed_out, cmd_prev in proposals:
            task.servo.cmd_prev = cmd_prev
        return joint_payload or None

    for arm, task, cube, next_stream, next_elapsed, target, done, \
            timed_out, _cmd_prev in proposals:
        task.stream = next_stream
        task.reach_elapsed = next_elapsed
        if done or timed_out:
            ao_servo.task_servo_report(op, task)
            op._register_hold(arm, cube, task.target, task.sector,
                              task.servo.bias, now,
                              restream=(next_stream if timed_out and
                                        float(np.linalg.norm(
                                            next_stream - target)) > 1e-9
                                        else None))
            cube.assigned_to = None
            op._arm_tasks[arm] = None
            print(f"[task:{arm}] independent reach complete; per-arm "
                  "grasp/lift/carry continues")
    if joint_payload:
        # MIXED packet (07-21 PHASE 2): a staircasing arm's joint slot rides
        # alongside the peer's Cartesian slot. Sides the staircase owns keep
        # their joint slot (a joint side must never also carry a Cartesian
        # target); the rest are carried over in per-arm form.
        mixed = dict(joint_payload)
        for side, slot in ao_arbitration.per_arm_slots(packet).items():
            mixed.setdefault(side, slot)
        return mixed
    return packet


def _derive_phase(op):
    K = _K(op)
    if task_manipulation_active(op) or task_tracks_active(op):
        return K.Phase.REACH
    if any(t is not None for t in op._arm_tasks.values()):
        return K.Phase.GO
    if any(not c.reached and c.held_by is None for c in op.cubes.values()):
        return K.Phase.SEARCH
    return K.Phase.HOLD if op._held else K.Phase.DONE


def tick_independent(op, telemetry: dict, jpos, now: float) \
        -> tuple[float, float, dict | None]:
    """One symmetric dual-arm scheduler tick."""
    if not ensure_mission_ready(op, now):
        op.phase = _K(op).Phase.SEARCH
        return 0.0, 0.0, op._arm_packet(reach_positions={})

    # 07-20 PER-ARM PROTOCOL: a held-arm track no longer halts the scheduler —
    # tasks assign/advance and their Cartesian slots merge alongside the track's
    # joint side in final arbitration. The base still stands still while any
    # joint track is walking (carry stability), and tick_task_motion itself
    # refuses to grant a second staircase.
    held_track_active = any(h["track"] is not None for h in op._held.values())
    assign_tasks(op, now)
    prepare_tasks(op, now)
    manipulating = task_manipulation_active(op) or task_tracks_active(op)
    packet = tick_task_motion(op, telemetry, jpos, now)
    if held_track_active or manipulating or task_manipulation_active(op) \
            or task_tracks_active(op):
        op._journey_walk_pose = False  # arms back to side-home while manipulating
        vx, wz = 0.0, 0.0
    else:
        vx, wz = navigation_command(op, now)
    # 07-21: joint-space tuck/untuck glide — free arms stream their posture
    # goal (tuck while walking is pending, side-home otherwise) as per-arm
    # joint slots.
    # 07-26 deadlock audit (CRITICAL): this used to be gated on
    # `packet is None`, which never happens under nominal telemetry —
    # build_arm_packet fills every slot and returns None only on missing/
    # non-finite telemetry. The joint glide was therefore dead code, the
    # Cartesian home_glide (IK lands up to 62 deg off the power-on posture,
    # see JOURNEY_WALK_QUAT comment) could never satisfy the JOINT-space
    # journey_arms_tucked gate, and EVERY journey run deadlocked at the
    # leg-2 walk entry with nav silently zeroed. The posture payload now
    # merges unconditionally: its per-arm joint slots own their sides
    # (payload skips held/STAGING/REACHING arms), the packet's remaining
    # Cartesian slots ride alongside — the same mixed-packet idiom the
    # staircase path above uses.
    posture = journey_posture_payload(op, jpos)
    if posture is not None:
        if packet is not None:
            for side, slot in ao_arbitration.per_arm_slots(packet).items():
                posture.setdefault(side, slot)
        packet = posture
    # JOURNEY end state: every registered cube resolved -> stand, done.
    # 'failed' counts as resolved (07-21 review: requiring all-carried made a
    # single failed grasp an unfinishable mission) but gets its own message;
    # navigation already refuses to walk with a failed hold.
    if getattr(op, "journey", False) and not op._journey_done and op.cubes and \
            all(h["grasp"] in ("carried", "failed")
                for h in op._held.values()) and \
            len(op._held) == len(op.cubes):
        op._journey_done = True
        n_fail = sum(1 for h in op._held.values() if h["grasp"] == "failed")
        if n_fail:
            print(f"[journey] MISSION ENDED — standing; {n_fail} grasp(s) "
                  "FAILED, the rest carried")
        else:
            print("[journey] MISSION COMPLETE — standing with "
                  f"{len(op.cubes)} cube(s) in hand")
    op.phase = _derive_phase(op)
    return vx, wz, packet
