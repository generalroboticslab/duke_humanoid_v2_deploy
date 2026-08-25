"""Mission phases, arrival gating, latching, reach (S6). SAFE-NAV-*, SAFE-ENVELOPE-* call sites; INC-4/INC-6.

Extraction contract (S6): verbatim operator bodies, self -> op; constants and
Phase resolve LIVE against the owning operator module (importlib multi-load +
test monkeypatch safe). See docs/auto_operator_refactor_plan.md.
"""
from __future__ import annotations

import math
import sys

import numpy as np


def _K(op):
    return sys.modules[type(op).__module__]


def gate_eval(op, pos) -> tuple[float, str | None, bool]:
    """(score, best_arm, fresh) from the learned gate, throttled to
    _K(op).GATE_EVAL_PERIOD_S per 2 cm-quantized target. Between refreshes the
    cached verdict returns with fresh=False — debounce counters must step
    ONLY on fresh samples (they integrate INDEPENDENT stochastic draws;
    stepping on cached repeats would let one lucky sample latch)."""
    import time as _time
    key = (int(round(float(pos[0]) / 0.02)),
           int(round(float(pos[1]) / 0.02)),
           int(round(float(pos[2]) / 0.02)))
    t = _time.monotonic()
    hit = op._gate_cache.get(key)
    if hit is not None and (t - hit[2]) < _K(op).GATE_EVAL_PERIOD_S:
        return hit[0], hit[1], False
    import torch
    score = float(op.gate.score(
        torch.tensor([float(pos[0]), float(pos[1]), float(pos[2])],
                     dtype=torch.float32),
        torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0])))
    best = op.gate.best_arm
    if len(op._gate_cache) > 128:
        op._gate_cache.clear()
    op._gate_cache[key] = (score, best, t)
    return score, best, True


def reachable_now(op, pos, arm: str) -> bool:
    """Instantaneous 'can `arm` reach `pos`' — NO debounce (used only at settled
    stations to decide a turnaround, where per-tick chatter is not a concern;
    the per-tick HOLD follow keeps using the debounced _dyn_reachable)."""
    if not op._target_safe(pos):
        return False
    if op.gate is None:
        return float(np.hypot(pos[0], pos[1])) <= _K(op).DYN_OUTREACH_M
    score, best, _ = op._gate_eval(pos)
    return bool(score > op.gate.SCORE_THRESHOLD) and best == arm


def dyn_reachable(op, pos: np.ndarray, arm: str, h: dict) -> bool:
    """Can `arm` still get to `pos`? Decided by the LEARNED ReachabilityGate
    when available (same scorer that judged arrival in GO), with
    _K(op).GATE_CONSECUTIVE debounce in both directions — the gate samples fresh
    random candidates per call, so single verdicts chatter. Geometric radii are
    only the --no-use-gate fallback (deterministic → no debounce needed).
    Counters and hysteresis live in the hold entry `h`, so each holding arm
    judges its own cube independently (dual-cube)."""
    if not op._target_safe(pos):
        return False
    if op.gate is None:
        d = float(np.hypot(pos[0], pos[1]))
        # hysteresis: once retracted, the cube must come back INSIDE the tighter
        # arrival floor before we reach again
        return d <= (_K(op).GEOMETRIC_FLOOR_M if h["home"] else _K(op).DYN_OUTREACH_M)
    score, best, fresh = op._gate_eval(pos)
    reachable_any = bool(score > op.gate.SCORE_THRESHOLD)
    h["best_arm"] = best if reachable_any else None
    ok = reachable_any and best == arm
    if fresh:                      # counters integrate independent samples only
        if ok:
            h["in_hits"] += 1
            h["out_hits"] = 0
        else:
            h["out_hits"] += 1
            h["in_hits"] = 0
    if h["home"]:
        return h["in_hits"] >= _K(op).GATE_CONSECUTIVE           # need N hits to re-reach
    return h["out_hits"] < _K(op).GATE_CONSECUTIVE               # need N misses to retract


def legs_and_gate(op, now: float) -> tuple[float, float, bool]:
    """One GO tick → (vx_desired, wz_desired, arrived).

    Arrival LATCHES (sim ReachabilityGate semantics: "once latched it never
    un-latches — the robot must not stop-start on the boundary"): the gate
    scorer samples fresh random candidates each call, so re-deriving the
    verdict every tick would chatter right at the envelope edge.
    """
    if op._arrived:
        return 0.0, 0.0, True
    cube = op.active
    if not cube.fresh(now, _K(op).DET_STALE_S):
        return 0.0, 0.0, False                      # blind → stand still, keep tracking
    x, y = float(cube.pos[0]), float(cube.pos[1])
    dist_xy = math.hypot(x, y)

    # arrival gate: learned reachability (both rings) with geometric floor
    arrived = dist_xy <= _K(op).GEOMETRIC_FLOOR_M
    gate_fired = False
    gate_best = None
    if not arrived and op.gate is not None:
        score, gate_best, fresh = op._gate_eval(
            np.array([x, y, float(cube.pos[2])]))
        if fresh:                  # counters integrate independent samples only
            op._gate_hits = op._gate_hits + 1 \
                if score > op.gate.SCORE_THRESHOLD else 0
        gate_fired = op._gate_hits >= _K(op).GATE_CONSECUTIVE
        arrived = gate_fired
    if arrived:
        # Reaching arm: HARD same-side rule (dual-cube collision safety: arms
        # never cross the midline — the IK has no reliable arm-vs-arm
        # avoidance, and the user places one cube per half-space). The gate
        # still decides WHEN we are close enough, no longer WHICH arm; if its
        # side-agnostic winner disagrees we log it and keep the side arm.
        if abs(y) <= _K(op).MIDLINE_MARGIN_M:
            # ON the midline at arrival: the side pick would ride one noisy
            # frame's sign (07-15 audit) — re-queue until it sits clearly.
            print(f"[go] {op.active.key} sits on the midline (|y| ≤ "
                  f"{_K(op).MIDLINE_MARGIN_M} m) — arm choice would be noise; "
                  f"re-queue via SEARCH")
            op.phase = _K(op).Phase.SEARCH
            return 0.0, 0.0, False
        side = "left" if y > 0 else "right"
        if side in op._held:
            # that arm is busy holding the other cube — abort the leg and
            # re-queue (the one-cube-per-side placement contract was broken)
            print(f"[go] {side} arm busy but {op.active.key} sits on its "
                  f"side — re-queue via SEARCH")
            op.phase = _K(op).Phase.SEARCH
            return 0.0, 0.0, False
        if gate_fired and gate_best is not None and gate_best != side:
            print(f"[go] gate favors {gate_best} but same-side rule "
                  f"binds {op.active.key} to the {side} arm")
        op._arrived = True
        op._reach_arm = side
        return 0.0, 0.0, True

    if op._leg_dir is None:                       # direction chosen ONCE per leg (sim semantics)
        op._leg_dir = 1.0 if abs(math.atan2(y, x)) <= math.pi / 2 else -1.0
    bearing = math.atan2(y, x)
    err = bearing if op._leg_dir > 0 else op._wrap(bearing + math.pi)
    wz = float(np.clip(_K(op).K_STEER * err, -_K(op).WZ_MAX, _K(op).WZ_MAX))
    return op._leg_dir * _K(op).CRUISE_VX, wz, False


def tick_search(op, now: float) -> None:
    """SEARCH: start a leg as soon as ANY pursuable cube has a CONFIRMED
    sighting — FIRST SEEN, FIRST GRABBED (user spec): the mission does NOT
    wait for the other cube to be found; the free eye keeps scanning for it
    while this leg runs. Ties (both confirmed in the same streak instant,
    e.g. both already in view) break by distance. Pursuable = the arm of its
    half-space is free (hard same-side rule); a cube whose arm is busy
    holding the other cube waits here. Held cubes are not pending — their
    arms stay on them (overlay) while this machine chases the rest. CONFIRMED
    sightings only: base-frame positions from a previous leg are stale the
    moment the robot has walked. Nothing left to pursue → HOLD (hands keep
    their cubes) or DONE (legacy park mode)."""
    pending = [c for c in op.cubes.values()
               if not c.reached and c.held_by is None]
    if not pending:
        op.phase = _K(op).Phase.HOLD if op._held else _K(op).Phase.DONE
        return
    # GUARD (serialize arms): never start a new reach leg while a held arm is
    # mid-track (retreat / turnaround / re-extend). Reaches (GO/REACH) and
    # joint tracks are MUTUALLY EXCLUSIVE — both drive the global joint stream,
    # so running them together lets one override the other (the 07-15 jump).
    # Let the in-flight track finish first; it will land in HOLD and re-enter.
    if any(h["track"] is not None for h in op._held.values()):
        return
    # GRASP SEQUENCING: with --reach-standoff 0 the jaw centre is driven ONTO
    # the cube centre (aim_up above it) — the fingers must be OPEN (zeroed)
    # before ANY reach, or
    # a closed gripper is rammed into the cube. Hold the first reach until the
    # startup zeroing has been sent AND its grace has elapsed. A dead chain
    # (op._ee_chain_dead) skips this: classic reach-and-hold, gripper inert.
    if op.grasp and not op._ee_chain_dead:
        if not op._hands_opened or (
                op._zeros_t is not None
                and (now - op._zeros_t) < _K(op).GRASP_ZERO_GRACE_S):
            if now - getattr(op, "_zero_wait_diag_t", -1e9) > 3.0:
                op._zero_wait_diag_t = now
                print("[grasp] holding the first reach until the grippers "
                      "finish zeroing OPEN (closed fingers must not be "
                      "driven onto a cube)")
            return
    if not op._home_settled:
        # HOME-FIRST gate (user 07-18): the mission may not reach until BOTH
        # arms have physically settled at the home EE — reaching from the
        # power-on pose mid-glide starts the approach from an unvalidated
        # seed. Measured, not commanded: the glide publishing the target is
        # not the arm being there.
        if op._home_wait_t0 is None:
            op._home_wait_t0 = now
        near = True
        for arm in ("left", "right"):
            if arm in op._held:
                continue
            act = op._ee_act.get(arm)
            if act is None or not np.all(np.isfinite(np.asarray(act, float))) \
                    or float(np.linalg.norm(
                        np.asarray(act, float)
                        - np.asarray(op.rest_ee[arm], float))) \
                    > _K(op).HOME_SETTLE_TOL_M:
                near = False
                break
        if near:
            op._home_settled = True
            print("[home] both arms settled at the home pose — reaching "
                  "may begin")
        elif (now - op._home_wait_t0) > _K(op).HOME_SETTLE_TIMEOUT_S:
            op._home_settled = True
            print(f"[home] WARNING: arms did NOT settle at home within "
                  f"{_K(op).HOME_SETTLE_TIMEOUT_S:.0f}s (EE telemetry gap or an "
                  f"obstructed glide) — proceeding anyway; watch the first "
                  f"reach")
        else:
            if now - getattr(op, "_home_wait_diag_t", -1e9) > 3.0:
                op._home_wait_diag_t = now
                print("[home] holding the first reach until both arms "
                      "settle at the home pose")
            return
    if op._sec is not None:
        return          # dual-parallel approach in flight: no new leg starts
                        # (and with walk enabled, absolutely no base motion
                        # under a latched Cartesian reach target)
    free = [a for a in ("left", "right") if a not in op._held]
    ready = []
    for c in pending:
        if not c.confirmed(now):
            continue
        if abs(float(c.pos[1])) <= _K(op).MIDLINE_MARGIN_M:
            # ON the midline: detection jitter would pick the arm from a coin
            # flip and (rear especially) commit an un-audited cross-back reach
            # (07-15 audit). Wait for the cube to sit clearly in one half-space.
            continue
        if ("left" if c.pos[1] > 0 else "right") not in free:
            continue
        if not op._target_safe(c.pos):
            if f"{c.key}:env" not in op._sector_warned:  # once per cube
                op._sector_warned.add(f"{c.key}:env")
                print(f"[search] {c.key} is outside the target safety envelope "
                      f"(r [{_K(op).TARGET_MIN_RADIUS_M}, {_K(op).TARGET_MAX_RADIUS_M}] m / "
                      f"z [{_K(op).TARGET_Z_MIN_M}, {_K(op).TARGET_Z_MAX_M}] m) at "
                      f"[{c.pos[0]:+.2f},{c.pos[1]:+.2f},{c.pos[2]:+.2f}] — NOT "
                      f"reaching. r>{_K(op).TARGET_MAX_RADIUS_M} or z>0 for a "
                      f"table cube = the TORSO-TILT signature: re-check the "
                      f"hang/stance (RULE #0: legs straight, torso upright — "
                      f"since the envelope was raised to "
                      f"{_K(op).TARGET_MAX_RADIUS_M} m this print no longer "
                      f"fires on the r~0.5 tilt readings it used to catch, so "
                      f"RULE #0 is the only tilt guard left)")
            continue
        sector = op._sector(c.pos)
        # Staged legs need the hold's reverse staircase to come home safely
        # (legacy PARK mode has none → forbid, INCLUDING side-home front-HIGH)
        # and answer to the per-sector SECTOR_REACH_ENABLED kill-switch
        # (hardware roll-out authority). Direct legs (classic front; side-home
        # side/low-front; every leg when side-home runs without stage-sectors)
        # are always allowed. The route is judged on the STANDOFF point — the
        # same value latch_reach_target and register_hold use — so the +2 cm
        # aim_up can never make search and latch disagree (07-18 review: a
        # raw-z judgment opened a 2 cm sliver that bypassed the kill-switch).
        enabled = (op.side_home
                   and op._staged_target_idx(
                       sector, op._standoff_point(c.pos)) == op._home_idx) \
            or (_K(op).SECTOR_REACH_ENABLED[sector] and (
                op.hold_after_reach
                or (sector == "front" and not op.side_home)))
        if not enabled:
            if c.key not in op._sector_warned:      # once per cube
                op._sector_warned.add(c.key)
                print(f"[search] {c.key} sits in the {sector.upper()} sector — "
                      f"reaching there is DISABLED pending the robot-side "
                      f"joint-stream rate knob (see _K(op).SECTOR_REACH_ENABLED)")
            continue
        ready.append(c)
    if not ready:
        return                        # nothing confirmed-and-servable yet: wait
    op.active = min(ready, key=lambda c: (c.first_seen,
                                            float(np.hypot(c.pos[0], c.pos[1]))))
    op._leg_dir, op._gate_hits, op._arrived = None, 0, False
    op.phase = _K(op).Phase.GO


def tick_go(op, now: float) -> tuple[float, float]:
    """GO: walk toward the active cube; on gate-approved arrival (and once the
    ramp has actually stopped the base) latch the reach target. Losing the cube
    rewinds to SEARCH — the camera claim is kept, only the sighting must be
    re-confirmed. Returns (vx_desired, wz_desired)."""
    if (now - op.active.last_seen) > _K(op).DET_LOST_RESCAN_S:
        op.phase = _K(op).Phase.SEARCH
        return 0.0, 0.0
    vx_des, wz_des, arrived = op._legs_and_gate(now)
    if arrived and abs(op._vx) < 0.02:
        op._latch_reach_target(now)
    return vx_des, wz_des


def latch_reach_target(op, now: float) -> None:
    """Freeze this leg's reach target and enter REACH.

    Target = per-axis MEDIAN of the recent sightings (a single 1-inlier PnP
    glitch at the arrival tick cannot poison the leg), pulled back toward the
    robot by the standoff. The servo chain binds to the MODAL measurement port
    of that window (the claim port can differ — the monitor's targets dict is
    last-camera-wins per key)."""
    window = [(s.pos, s.port) for s in op.active.hist
              if (now - s.t) < _K(op).CUBE_LATCH_WINDOW_S]
    if not window:
        # occlusion exactly at the arrival tick: np.median over an empty stack
        # would CRASH the operator mid-mission (07-15 audit) — re-acquire instead
        print(f"[reach] latch window empty for {op.active.key} (occluded at "
              f"arrival) — re-acquiring via SEARCH")
        op.phase = _K(op).Phase.SEARCH
        return
    median = np.median(np.stack([p for p, _ in window]), axis=0)
    side_ok = (median[1] > _K(op).MIDLINE_MARGIN_M) if op._reach_arm == "left" \
        else (median[1] < -_K(op).MIDLINE_MARGIN_M)
    if not side_ok or not op._target_safe(median):
        # the MEDIAN (what the arm will actually chase) must sit clearly on the
        # bound arm's half-space and inside the safety envelope — the arrival
        # frame alone chose the arm, and a lagged median may disagree (07-15
        # audit: rear-midline cross-back / envelope findings)
        print(f"[reach] latched median for {op.active.key} fails the "
              f"{'side' if not side_ok else 'safety-envelope'} check — "
              f"re-acquiring via SEARCH")
        op.phase = _K(op).Phase.SEARCH
        return
    op._reach_target = op._standoff_point(median)
    print(f"[reach] latched {op.active.key} target="
          f"[{op._reach_target[0]:+.3f}, {op._reach_target[1]:+.3f}, "
          f"{op._reach_target[2]:+.3f}]  r_xy="
          f"{float(np.hypot(op._reach_target[0], op._reach_target[1])):.3f} m  "
          f"arm={op._reach_arm}  (envelope cap {_K(op).TARGET_MAX_RADIUS_M} m; "
          f"clean-orientation reach ~0.51 m; validated circle "
          f"{_K(op).GEOMETRIC_FLOOR_M} m)")
    op._reach_t0 = now
    op._reset_leg_state(now)
    ports = [prt for _, prt in window]
    op._reach_port = max(set(ports), key=ports.count)
    # Route into the sector along the bidirectional track: the arm starts at
    # its HOME station (front(0) classically, the SIDE(1) under stage-sectors)
    # and walks out to the sector's station. target==home → zero-length track
    # → straight to Cartesian. _staged_target_idx owns the routing decision:
    # classic unchanged; side-home direct everywhere without stage-sectors;
    # stage-sectors sends high-front and ALL rear via their stations (the
    # settled station re-seeds the robot-side IK in the correct elbow branch
    # — S-A audit 07-18).
    op._reach_sector = op._sector(op._reach_target)
    target_idx = op._staged_target_idx(op._reach_sector, op._reach_target)
    op._reach_tm = op._new_track(op._reach_arm, op._home_idx, target_idx, now,
                                   replant=op.stage_sectors) \
        if target_idx != op._home_idx else None
    # (replant under stage-sectors: nothing PROVES the arm is physically at
    #  home when a leg latches — e.g. a hold released mid-glide — so the
    #  home station is re-planted first; settles instantly when already there)
    if op._reach_tm is not None:
        print(f"[stage] {op._reach_arm} arm tracks "
              f"{_K(op).TRACK[op._home_idx]}→{op._reach_sector} "
              f"(joint stream, {_K(op).STAGE_RATE} rad/s)")
    op.phase = _K(op).Phase.REACH


def tick_reach(op, telemetry: dict, jpos, now: float) -> dict | None:
    """REACH: walk the bidirectional staging track OUT to the cube's sector
    station (front legs have a zero-length track → straight to Cartesian),
    then stream the (servo-corrected) Cartesian target until the EE tracks it
    AND the servo had its say (_servo_ready), or the 8 s timeout. A JOINT
    payload rides the track; None on a telemetry blip (tick() keeps the stream
    silent). The 8 s reach clock + servo give-up clock start only AFTER the
    track settles at the target station; the robot re-seeds its IK from
    encoders on the joint→Cartesian switch (real_env 8335075), which is why
    that switch happens ONLY at a settled target station, never mid-hop. A hop
    the arm cannot reach flips the track's goal to front (0); once it settles
    back at front the leg is abandoned and SEARCH re-acquires (no in-place
    retry — that is HOLD's job)."""
    if op._reach_tm is not None:
        tm = op._reach_tm
        payload, status = op._step_toward(tm, jpos, now)
        if now - getattr(op, "_reach_diag_t", -1e9) > 1.0:
            op._reach_diag_t = now
            sl = slice(13, 20) if op._reach_arm == "left" else slice(20, 27)
            if jpos is not None and len(jpos) >= 31 and tm["seq"]:
                err = float(np.max(np.abs(
                    np.asarray(jpos[sl])
                    - op._track_posture(op._reach_arm, tm["seq"][0]))))
                print(f"[reach-debug] stage={_K(op).TRACK[tm['seq'][0]]} "
                      f"status={status} max_joint_err={err:.4f} rad "
                      f"seq={tm['seq']}")
            else:
                print(f"[reach-debug] status={status} "
                      f"jpos={'OK' if jpos is not None else 'MISSING'} "
                      f"seq={tm['seq']}")
        if tm["timed_out"] and not tm["seq"]:
            print(f"[stage] {op._reach_arm} reach hop unreachable — returned "
                  f"to {_K(op).TRACK[op._home_idx]}, re-acquiring via SEARCH")
            op._reach_tm = None
            op.phase = _K(op).Phase.SEARCH
            return payload
        if status != "done":
            return payload           # en route (payload None on a telemetry blip)
        # Settled at the target station. The joint→Cartesian switch needs the
        # live EE as the ramp seed — if telemetry lacks it THIS tick, keep
        # STREAMING the settled station posture and retry (07-16 review: the
        # old flow cleared _reach_tm then returned None BEFORE the 8 s timeout
        # check — a persistent pos_actual gap became a silent forever-stall
        # with the arm parked at side/rear and the stream dead).
        ee_probe = (telemetry.get("ee") or {}).get(op._reach_arm) or {}
        if ee_probe.get("pos_actual") is None:
            if now - getattr(op, "_missing_ee_diag_t", -1e9) > 1.0:
                op._missing_ee_diag_t = now
                print(f"[reach] WAITING at the {op._reach_sector} station: "
                      f"telemetry ee.{op._reach_arm}.pos_actual is missing — "
                      f"holding the station posture, not switching to Cartesian")
            return op._staged_joint_payload(
                op._reach_arm,
                op._track_posture(op._reach_arm,
                                    _K(op).SECTOR_IDX[op._reach_sector]),
                tm["frozen"] if tm["frozen"] is not None else
                [float(v) for v in op._track_posture(
                    "right" if op._reach_arm == "left" else "left",
                    op._home_idx)])
                # (idle-arm pin: its HOME posture — pinning the FRONT station
                #  under side-home would sweep the resting arm across the body)
        # seed available → switch to Cartesian this tick
        op._reach_t0 = now
        op._reach_tm = None
    if op.dynamic_track:
        upd = op._live_retrack(op.active, op._reach_arm,
                                 op._reach_sector, op._reach_target, now)
        if upd is not None:
            op._reach_target = upd
    if op.visual_servo:
        op._servo_step(telemetry, jpos, now)
    tgt = op._servo_target()
    ee = (telemetry.get("ee") or {}).get(op._reach_arm) or {}
    actual = ee.get("pos_actual")
    # EVERY reach approaches at the gentle rate — front legs included (07-16:
    # the classic direct-target front reach snapped at the robot-side 0.25 m/s
    # cap, which at 0.8 torque looked like a catapult and could knock the cube
    # away at contact with --reach-standoff 0).
    op._reach_stream, pub_tgt = op._gentle_approach(
        op._reach_stream, tgt, actual)
    if pub_tgt is None:
        if now - getattr(op, "_missing_ee_diag_t", -1e9) > 1.0:
            op._missing_ee_diag_t = now
            print(f"[reach] WAITING: no ee.{op._reach_arm}.pos_actual to "
                  f"seed the approach ramp")
        return None                  # no hand position yet → SILENCE this tick
    packet = op._arm_packet(pub_tgt)
    done = actual is not None and \
        float(np.linalg.norm(np.asarray(actual) - tgt)) < _K(op).REACH_OK_M and \
        op._servo_ready(now)
    if done or (now - op._reach_t0) > _K(op).REACH_TIMEOUT_S:
        op._servo_leg_report(now)
        others_pending = any(c is not op.active and not c.reached
                             and c.held_by is None for c in op.cubes.values())
        if op.hold_after_reach:
            # Register the hold: this arm STAYS on the cube (kept in the packet
            # by the _arm_packet overlay, followed/retracted by _held_update)
            # while the mission moves on to the next cube with the OTHER arm.
            # active.reached is left False so the eye keeps tracking. Servo bias
            # frozen at registration.
            op._register_hold(op._reach_arm, op.active,
                                op._reach_target, op._reach_sector,
                                op._servo_bias, now)
            op.phase = _K(op).Phase.SEARCH if others_pending else _K(op).Phase.HOLD
        else:
            op.active.reached = True
            op._park_t0 = now
            op.phase = _K(op).Phase.PARK
    return packet


def tick_hold(op, telemetry: dict, jpos, now: float) -> dict | None:
    """HOLD: terminal phase — every cube is in a hand. Per-arm follow /
    retract / release already ran in _held_update this tick. The SERVO chain
    (single, per-leg state) keeps refining the MOST RECENT leg's hand bias
    exactly as the classic single-cube HOLD did — b̂ converges to ~66% during
    REACH (exits at 3 accepted updates) and finishes here; the refined bias
    is mirrored live into that arm's hold entry. Earlier holds keep the bias
    frozen at their registration (one chain cannot measure two hands).
    Runs until Ctrl+C. NOTE (07-15 contract audit): with --use-ik there is NO
    robot-side retract-on-silence — after Ctrl+C the arms HOLD the last
    published targets. Shutdown protocol: remove the cubes, let the holds
    release and the arms come home, THEN exit."""
    if op.visual_servo and op._reach_arm in op._held:
        h = op._held[op._reach_arm]
        op._reach_target = h["target"]    # dyn-followed target feeds the chain
        op._servo_step(telemetry, jpos, now)
        op._servo_target()                # maintains the settle-gate bookkeeping
        h["bias"] = None if op._servo_bias is None else op._servo_bias.copy()
    return op._arm_packet(None)


def tick_park(op, now: float) -> dict | None:
    """PARK (non-final cubes): stream the rest pose briefly, then SEARCH resolves
    to the next cube or DONE."""
    packet = op._arm_packet(None)
    if (now - op._park_t0) > _K(op).PARK_HOLD_S:
        op.phase = _K(op).Phase.SEARCH
    return packet


def standoff_point(op, pos) -> np.ndarray:
    """`pos` pulled back toward the robot by reach_standoff along the XY ray and
    lifted by aim_up — the point the hand is actually sent to (0 standoff ⇒
    aim_up above the cube centre). Every commanded reach point funnels through
    here, so the settle check and the lift-off waypoints see the same offset."""
    p = np.asarray(pos, dtype=float).copy()
    back = p[:2] / max(np.linalg.norm(p[:2]), 1e-6)
    p[:2] -= back * op.reach_standoff
    p[2] += op.aim_up
    return p

