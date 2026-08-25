"""Per-arm hold / grasp / carry state machinery (refactor stage S5).

Verbatim extractions of the operator's arm-lifecycle methods (self -> op;
constants resolved LIVE from the operator module via sys.modules so that
importlib multi-instances and test monkeypatching keep working). The giant
held_update keeps its exact branch order — the state chart in
docs/auto_operator_adr.md (ADR-5) names each branch; a per-function split is a
future stage with its own gate. SAFE-HELD-*, SAFE-GRIPPER-*, SAFE-DUAL-*
(register/lift), SAFE-SIDEHOME-* (via-home) — see the safety contract.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .models import GraspState, HoldState


def _K(op):
    """The operator MODULE that owns `op` — constants and Phase resolve
    against the exact module instance (importlib load names differ across
    the test suite, the replay harness and production)."""
    return sys.modules[type(op).__module__]


def latch_grasp_uncertain(op, h: dict, arm: str, reason: str,
                          telemetry: dict | None = None) -> None:
    """Fail closed on any ambiguous grasp/release evidence.

    The fingers are deliberately left exactly as commanded: this helper never
    queues ``hand_open`` (nor any other EE action).  The arm target is pinned at
    the measured EE when available, and the operator's global intervention latch
    stops navigation and all task advancement until a human inspects the hand.
    """
    ee = (((telemetry or {}).get("ee") or {}).get(arm) or {})
    act = ee.get("pos_actual", getattr(op, "_ee_act", {}).get(arm))
    if act is not None:
        act_a = np.asarray(act, dtype=float)
        if act_a.shape == (3,) and bool(np.all(np.isfinite(act_a))):
            h["target"] = act_a.copy()
            h["bias"] = None
    h["restream"] = None
    h["home"] = False
    h["grasp"] = GraspState.UNCERTAIN
    h["state"] = "uncertain"
    h["drop_t0"] = None
    op._grasp_safety_latched = True
    op._grasp_safety_arm = arm
    op._grasp_safety_reason = reason
    # Compatibility aliases retained for the in-progress refactor; tick uses
    # the canonical _grasp_safety_latched name below.
    op._grasp_intervention_required = True
    op._grasp_intervention_reason = f"{arm}: {reason}"
    op._intervention_arm_targets = None
    print(f"[grasp] INTERVENTION REQUIRED — {arm} {reason}. "
          "Gripper kept CLOSED; navigation and both arms PAUSED. "
          "Support/inspect the cube and use an explicit manual hand_open only "
          "when safe.")


def startup_grasp_guard(op, telemetry: dict) -> bool:
    """Refuse startup zeroing when either side has any prior grasp marker.

    False/None status transport and the physical fingers are not authoritative
    inverses.  Only the total absence of both result fields is an empty-start
    calibration precondition.
    """
    for arm in ("left", "right"):
        ee = ((telemetry.get("ee") or {}).get(arm) or {})
        marked = ee.get("grasp_detected") is not None \
            or ee.get("grasp_action_id") is not None
        if not marked:
            continue
        op._grasp_safety_latched = True
        op._grasp_safety_arm = arm
        marker = (f"grasp_detected={ee.get('grasp_detected')!r}, "
                  f"grasp_action_id={ee.get('grasp_action_id')!r}")
        op._grasp_safety_reason = f"startup prior grasp marker ({marker})"
        op._grasp_intervention_required = True
        op._grasp_intervention_reason = \
            f"{arm}: startup prior grasp marker ({marker})"
        op._intervention_arm_targets = None
        print("[grasp] INTERVENTION REQUIRED — startup telemetry contains a "
              f"prior grasp marker on {arm} ({marker}). ZEROING REFUSED (zero would "
              "auto-open); navigation and both arms PAUSED. Support/inspect "
              "the payload and use explicit manual hand_open only when safe.")
        return True
    return False


def lift_arc(op, wp1: np.ndarray, arm: str, sector: str) -> list[np.ndarray]:
    """Horizontal-arc waypoints from `wp1` in the direction of `sector`'s
    staging-station bearing: constant radius about the torso axis, constant
    z, and ALWAYS the FULL _K(op).GRASP_LIFT_OUT_M of arc — never capped at the
    station bearing and never skipped (user 07-17: an early stop can leave
    the gripper still over the cube's stand when the retract starts;
    clearance beats alignment, overshooting the station is harmless — the
    carry pulls to the station next anyway). Full sweeps stay inside every
    sector band by construction (worst case ~19 deg vs 60-deg bands);
    points are ~2 cm apart and each is sector/envelope-guarded regardless,
    truncating at the first violation (partial arc beats no arc)."""
    r = float(np.hypot(wp1[0], wp1[1]))
    if r < 1e-6:
        return []
    sgn = 1.0 if arm == "left" else -1.0
    th0 = math.atan2(sgn * float(wp1[1]), float(wp1[0]))  # fold right onto +y
    ths = math.radians(_K(op).STATION_BEARING_DEG[sector])
    direction = math.copysign(1.0, ths - th0) if abs(ths - th0) > 1e-6 else 1.0
    sweep = direction * _K(op).GRASP_LIFT_OUT_M / r
    n = max(1, math.ceil(abs(sweep) * r / 0.02))
    out = []
    for k in range(1, n + 1):
        th = th0 + sweep * k / n
        wp = np.array([r * math.cos(th), sgn * r * math.sin(th), float(wp1[2])])
        if op._sector(wp) != sector or not op._target_safe(wp):
            break
        out.append(wp)
    return out


def maybe_send_grab(op, arm: str, h: dict, cube, telemetry: dict,
                     now: float) -> bool:
    """Settle check + hand_grab dispatch for a registered hold. Returns True
    the tick the grab is sent. Called from the classic hold flow AND from
    the GO/REACH guard (07-17 audit: a hold registered while the OTHER arm
    is still reaching — dual-parallel first-arrival — must grab immediately;
    the gripper rides its own serial bus, never the 14-joint stream, so this
    cannot violate the modal-wire exclusivity Guard 1 protects)."""
    if not (op.grasp and not op._ee_chain_dead and h["grasp"] is None
            and h["track"] is None
            and h["restream"] is None and not h["home"]):
        # 07-26 review: the precondition chain is NOT evaluated on this path —
        # the EE-far persistence clock must not keep aging across retracts,
        # re-extend ramps, staircases or in-flight grabs (a stale stamp fired
        # the 15 s release mid-re-extend toward a perfectly servable cube).
        h["ee_far_t0"] = None
        return False
    # hand settled ON the cube (target unmoved + EE converged) → grab.
    # Every un-met precondition is REPORTED (throttled): a hold that
    # never grabs must say WHY — silent waiting is the failure class
    # the 07-16 audit was commissioned against.
    act = ((telemetry.get("ee") or {}).get(arm) or {}).get("pos_actual")
    why = None
    ee_far = False              # 07-26: explicit flag drives the ee_far_t0 clock
    if op._ee_alive(telemetry) is False:
        why = "EE chain not alive"
    elif op._zeros_t is not None \
            and (now - op._zeros_t) < _K(op).GRASP_ZERO_GRACE_S:
        why = "gripper zeroing still settling (startup grace)"
    elif (now - h["grasp_move_t"]) <= _K(op).GRASP_SETTLE_S:
        why = "target still moving (settle timer restarts on retrack)"
    elif act is None:
        why = "no EE telemetry"
    elif not cube.fresh(now, _K(op).DYN_LOST_GRACE_S):
        why = "cube unseen"
    else:
        tgt_a = np.asarray(h["target"], dtype=float)
        dist = float(np.linalg.norm(np.asarray(act) - tgt_a))
        if dist >= _K(op).GRASP_EE_OK_M:
            ee_far = True
            why = (f"EE {dist*100:.1f} cm from target (needs < "
                   f"{_K(op).GRASP_EE_OK_M*100:.0f} cm) — target="
                   f"[{tgt_a[0]:+.3f},{tgt_a[1]:+.3f},{tgt_a[2]:+.3f}] "
                   f"r_xy={float(np.hypot(tgt_a[0], tgt_a[1])):.3f}m, "
                   f"EE=[{act[0]:+.3f},{act[1]:+.3f},{act[2]:+.3f}]")
            # 07-26 deadlock audit: this precondition can be a steady state
            # (gravity sag / IK residual leaves the measured EE parked past
            # GRASP_EE_OK_M forever) — start the persistence clock. Any other
            # unmet precondition, or the grab actually sending, clears it —
            # by explicit flag, NOT a message-prefix test: "EE chain not
            # alive" also starts with "EE " and must clear, never retain
            # (07-26 review: dead-chain seconds were counting toward the
            # release threshold).
            h["ee_far_t0"] = h.get("ee_far_t0") or now
    if not ee_far:
        h["ee_far_t0"] = None
    if why is None:
        action_id = op._queue_ee_action(
            {"side": arm, "command": "hand_grab"})
        h["grasp"] = "sent"
        h["grasp_t0"] = now
        h["grasp_action_id"] = action_id
        h["grasp_base"] = ((telemetry.get("ee") or {}).get(arm)
                           or {}).get("grasp_detected")
        print(f"[grasp] {arm} gripper closing on {h['key']} "
              f"(load-supervised hand_grab)")
        return True
    if now - h.setdefault("grasp_diag_t", -1e9) > 3.0:
        h["grasp_diag_t"] = now
        print(f"[grasp] {arm} holding {h['key']} but NOT grabbing: {why}")
    return False


def live_retrack(op, cube, arm: str, sector: str, current,
                  now: float) -> np.ndarray | None:
    """IN-FLIGHT RETRACK (07-17): during a Cartesian reach leg the target
    follows the LIVE median of the last _K(op).CUBE_LATCH_WINDOW_S of sightings
    instead of staying frozen at the latch snapshot — the latch is a
    body-frame point, and standing-balance sway / reach-reaction tilt makes
    that frame drift 1-2 cm over a multi-second approach (the
    grabbed-off-centre class). Shared by the primary reach and the
    dual-parallel secondary. Same deadband as the hold follow
    (_K(op).DYN_RETRACK_M) and latch-grade validations: half-space, safety
    envelope, SECTOR-CLAMP with hysteresis (a cube drifting across a basin
    boundary must never slide the target un-staged — the facet-2
    cross-body class). Returns the refreshed standoff point, or None to
    keep the current target (occlusion → empty window → frozen behavior)."""
    if cube is None or not cube.fresh(now, _K(op).DYN_LOST_GRACE_S):
        return None
    window = [s.pos for s in cube.hist
              if (now - s.t) < _K(op).CUBE_LATCH_WINDOW_S]
    if not window:
        return None
    median = np.median(np.stack(window), axis=0)
    side_ok = (median[1] > _K(op).MIDLINE_MARGIN_M) if arm == "left" \
        else (median[1] < -_K(op).MIDLINE_MARGIN_M)
    staged_leg = op._staged_target_idx(sector, current) != op._home_idx
    # (staged legs re-apply the sector clamp under side-home too: the arm sits
    #  in a station-chosen basin — an un-staged cross-boundary slide is the
    #  facet-2 class. Direct legs keep only the bearing-jump rule below.)
    if not side_ok or not op._target_safe(median) \
            or ((not op.side_home or staged_leg)
                and op._sector_left(median, sector)):
        return None
    live_probe = op._standoff_point(median)
    if op.side_home and op.stage_sectors \
            and op._staged_target_idx(sector, live_probe) \
            != op._staged_target_idx(sector, current):
        # routing-class crossing (07-18 review): the live point slid across
        # FRONT_STAGE_Z_M — a DIRECT leg must never be retracked into
        # must-stage territory (nor a staged one out). FREEZE the target;
        # the HOLD machinery re-routes the cube after this leg ends.
        return None
    live = op._standoff_point(median)
    if op.side_home and abs(op._bearing_deg(live)
                              - op._bearing_deg(current)) > _K(op).BEARING_JUMP_DEG:
        return None    # VIA-HOME rule: an in-flight reach never chases a
                       # big bearing jump — the leg finishes/times out at the
                       # old point and the HOLD machinery routes via the side
    if float(np.linalg.norm(live - np.asarray(current, dtype=float))) \
            > _K(op).DYN_RETRACK_M:
        return live
    return None


def register_hold(op, arm: str, cube, target, sector: str, bias,
                   now: float, restream=None) -> None:
    """Create the per-arm hold entry — shared by the primary REACH arrival
    and the dual-parallel secondary. The hold carries the arm's _K(op).TRACK
    STATION (sector_idx): a retreat is a track motion toward front(0), a
    re-extend a track motion back out — both via _step_toward. Grasp
    lifecycle (--grasp): None → "sent" → "lifting" → "grasped" →
    "carried" (home at REST with the cube) | "failed" (plain hold).
    `restream` seeds the hold mid-ramp (a secondary that TIMED OUT short of
    the cube): the classic machinery finishes the gentle approach instead of
    the packet snapping the remaining distance in one step (07-17 audit)."""
    op._held[arm] = HoldState(
        key=cube.key, target=np.asarray(target, dtype=float).copy(),
        bias=None if bias is None else np.asarray(bias, dtype=float).copy(),
        sector=sector,
        sector_idx=op._staged_target_idx(sector, target),
        # (sector_idx == op._home_idx ⇒ DIRECT lifecycle: no track machinery,
        #  in-place Cartesian retract/park. Classic front==0==home reproduces
        #  the old rule; plain side-home routes everything direct as before;
        #  stage-sectors gives high-front/rear holds their true station so
        #  retreats and carries walk the audited staircase — never a raw
        #  Cartesian glide across a basin boundary.)
        restream=None if restream is None else
                 np.asarray(restream, dtype=float).copy(),
        grasp_t0=now, grasp_move_t=now)
    if hasattr(cube, "assigned_to"):
        cube.assigned_to = None
    cube.held_by = arm
    print(f"[hold] {arm} arm holds {cube.key} "
          f"({len(op._held)} arm(s) engaged)")


def enter_carried(op, h: dict, arm: str, ee_actual) -> None:
    """Transition a grasped hold to CARRIED: park ramp toward REST, and
    release the cube's camera claim — a cube riding in the gripper at the
    chest needs no dedicated eye, and keeping the claim starves the other
    cube's search when only one camera survives (07-16 audit)."""
    h["restream"] = np.asarray(ee_actual, dtype=float)
    h["target"] = np.asarray(op.rest_ee[arm], dtype=float)
    h["state"] = "carrying"
    h["grasp"] = "carried"
    cube = op.cubes[h["key"]]
    cube.no_claim = True
    if cube.camera_port is not None:
        print(f"[lock] {h['key']} released from camera {cube.camera_port} "
              f"(carried — no eye needed)")
        cube.camera_port = None
    # Retire this cube's camera (07-20 user spec): its job is done, so it
    # returns to neutral and sits out the scan — the other camera owns any
    # remaining search. PERMANENT since 2026-08-09 (user): no drop, claim or
    # pending-cube pull ever brings a retired eye back; only a mission
    # restart (--fresh-scan) clears the set.
    port = cube.last_camera_port
    if port is not None and port not in op._cam_retired:
        op._cam_retired.add(port)
        print(f"[lock] camera {port} retired to neutral ({h['key']} carried home)")
    if getattr(op, "journey", False):
        # JOURNEY (07-21): body-frame positions recorded before this leg's
        # walking are geometrically dead. Invalidate every still-pending
        # cube so the next leg re-acquires it LIVE (scan machinery) instead
        # of navigating to a rotted survey coordinate.
        for k, c in op.cubes.items():
            if c.held_by is None and not c.reached and c.pos is not None \
                    and c.last_seen > -1e9:
                c.last_seen = -1e9
                c.hist.clear()
                print(f"[journey] {k} survey position invalidated — "
                      "re-acquiring live before the next leg")
    print(f"[grasp] {arm} arm parks at REST with {h['key']} in hand")


def grasp_dropped(op, h: dict, cube, arm: str, telemetry: dict,
                   now: float) -> bool:
    """DROP detection for grasped/carried holds: the cube seen persistently
    far from the gripper means it is on the table, not in the hand (the
    chain has no other feedback — grasp_detected never updates after the
    grab; 07-16 audit: a dropped cube stayed 'held' forever and the mission
    ended 'complete' with an empty gripper). On a confirmed drop: reopen the
    gripper used to be reopened here.  That is not safe: a camera/body-frame
    error can put a still-held cube 15 cm from the EE numerically.  A persistent
    suspicion now latches UNCERTAIN, keeps the fingers closed, and pauses the
    mission for human inspection."""
    if h["track"] is not None:
        return False                       # mid-staircase: fingers occlude, skip
    act = ((telemetry.get("ee") or {}).get(arm) or {}).get("pos_actual")
    seen_far = (act is not None and cube.fresh(now, _K(op).DYN_LOST_GRACE_S)
                and float(np.linalg.norm(
                    np.asarray(cube.pos) - np.asarray(act))) > _K(op).GRASP_DROP_DIST_M)
    if not seen_far:
        h["drop_t0"] = None
        return False
    if h.get("drop_t0") is None:
        h["drop_t0"] = now
        return False
    if (now - h["drop_t0"]) <= _K(op).GRASP_DROP_PERSIST_S:
        return False
    latch_grasp_uncertain(
        op, h, arm,
        f"DROP SUSPECTED for {h['key']}: fresh vision stayed "
        f">{_K(op).GRASP_DROP_DIST_M*100:.0f} cm from the measured EE for "
        f">{_K(op).GRASP_DROP_PERSIST_S:.0f}s", telemetry)
    return True


def cube_servable(op, cube, arm: str, now: float) -> tuple[bool, int | None]:
    """Is `cube` currently servable BY `arm`: fresh, on `arm`'s half-space
    (not across the midline), and reachable now. Returns (ok, sector_idx)."""
    if not cube.fresh(now, _K(op).DYN_LOST_GRACE_S):
        return False, None
    p = cube.pos
    wrong_side = (p[1] < -_K(op).MIDLINE_MARGIN_M) if arm == "left" else (p[1] > _K(op).MIDLINE_MARGIN_M)
    if wrong_side or not op._reachable_now(p, arm):
        return False, None
    return True, _K(op).SECTOR_IDX[op._sector(p)]


def held_update(op, now: float, telemetry: dict, jpos) -> dict | None:
    """Per-arm dynamic tracking for EVERY holding arm, every mission tick.

    One state block per arm (dual-cube: both run side by side): follow a fresh
    servable cube (retrack deadband), retract while it is unseen past the
    grace window or out of reach, and RELEASE the hold — cube back to the
    pending queue, arm freed — when the cube persistently (> _K(op).HOLD_RELEASE_S)
    settles where this arm cannot serve it: the gate crowns the other arm,
    the cube crossed the midline (hard same-side rule), or it left this
    hold's SECTOR past the hysteresis band (a sector change needs a fresh
    staged leg, not an in-place slide). Release from the terminal HOLD phase
    rewinds to SEARCH so the freed cube is re-pursued.

    SIDE/REAR retraction walks the transit chain in reverse (each hop
    physically confirmed via telemetry EE, with a per-hop timeout) and then
    ALWAYS releases: those holds never re-extend in place — a fresh leg with
    proper forward staging is the only safe way back in. FRONT holds keep
    the classic retract/re-extend-in-place behavior. With --no-dynamic-track
    targets stay frozen at their latch values."""
    ret_payload = None
    for arm in list(op._held):
        h = op._held[arm]
        cube = op.cubes[h["key"]]
        if h["grasp"] == "uncertain":
            # Terminal fail-closed state.  Never follow/retract/release/retry:
            # tick() publishes the global two-arm intervention snapshot and
            # leaves the gripper command channel untouched until a human acts.
            continue
        if not op.dynamic_track:
            continue
        # GUARD (serialize arms): while ANOTHER arm is actively reaching
        # (GO/REACH), a held arm may keep FOLLOWING its cube in Cartesian
        # (in-basin, safe) but must NEVER start/advance a JOINT retreat — the
        # global 14-joint packet would override the reaching arm's stream and
        # fling a mid-basin arm across the torso (the 07-15 hardware jump).
        # Retreats/turnarounds/releases run only in HOLD/SEARCH. If the cube
        # is not servable right now, HOLD the last target (do not retract to
        # REST — that itself would command a side/rear arm toward front).
        if (not getattr(op, "independent_arms", False)) and \
                op.phase in (_K(op).Phase.GO, _K(op).Phase.REACH):
            # SECTOR-CLAMP the follow (mirrors the HOLD branch's _sector_left
            # guard below): while another arm reaches, a held arm may follow its
            # cube ONLY within its own sector. Following a cube that has drifted
            # out of h["sector"] would slide h["target"] across the
            # front<->side/rear basin boundary UN-staged; and because the
            # reaching arm freezes this arm at a one-shot encoder snapshot for the
            # whole staircase, at staging-completion the single Cartesian packet
            # would then command an un-staged cross-sector move from a stale basin
            # — the split-verdict torso-strike the 07-15 adversarial audit found
            # that the serialization guards do NOT cover (it is a Cartesian-vs-
            # sector gap, not a joint-vs-Cartesian one). An out-of-sector cube is
            # handled the safe way once the reach ends and we are back in HOLD:
            # the normal _sector_left release re-queues it for a fresh staged leg.
            # A grabbing/grasped/carried arm never follows: the gripper is
            # closing on (or holding) the cube — detections now track the HAND,
            # and chasing them would chase ourselves. The carry home waits here
            # too (Guard 1: no joint track during another arm's reach).
            if h["grasp"] in (None, "failed") and h["track"] is None \
                    and cube.fresh(now, _K(op).DYN_LOST_GRACE_S) \
                    and ((op.side_home and h["sector_idx"] == op._home_idx)
                         or not op._sector_left(cube.pos, h["sector"])):
                # (STAGED holds re-apply the sector clamp even under side-home:
                #  a staircase parked this arm in a specific basin — following
                #  a cross-boundary slide un-staged is the facet-2 class.
                #  Direct side-home holds keep the bearing-jump rule below.)
                ok, _ = op._cube_servable(cube, arm, now)
                if ok:
                    live = op._standoff_point(cube.pos)
                    if op.side_home and abs(
                            op._bearing_deg(live)
                            - op._bearing_deg(h["target"])) > _K(op).BEARING_JUMP_DEG:
                        pass       # VIA-HOME rule: no big-bearing slide while
                                   # another arm reaches; handled in HOLD
                    elif op.side_home and op.stage_sectors \
                            and op._staged_target_idx(h["sector"], live) \
                            != h["sector_idx"]:
                        pass       # routing-class crossing (z slid across
                                   # FRONT_STAGE_Z_M): never dragged across
                                   # un-staged — HOLD re-routes it after the
                                   # other arm's reach ends
                    elif float(np.linalg.norm(live - h["target"])) > _K(op).DYN_RETRACK_M:
                        h["target"] = live
                        h["grasp_move_t"] = now
            # 07-17 dual-parallel audit: a hold registered while the other
            # arm still reaches (secondary first-arrival) must GRAB now, not
            # after that reach ends — the hand_grab rides the gripper's own
            # serial bus, never the 14-joint stream, so Guard 1's modal-wire
            # exclusivity is untouched. Result processing / lift / carry
            # keep their existing deferral (lift+carry need the wire).
            op._maybe_send_grab(arm, h, cube, telemetry, now)
            continue
        # ---- an in-flight track motion (retreat toward front, or a turnaround
        #      back out) — INTERRUPTIBLE: re-decided ONLY at each settled station.
        #      Mid-hop, no new decision is taken and the joint stream is never
        #      swapped for Cartesian (the four self-check rules). ----
        if h["track"] is not None:
            if op._reach_tm is not None:
                continue     # the LEGACY 14-joint reach staircase still owns
                             # the whole wire; this arm waits (independent mode
                             # never sets _reach_tm)
            # 07-21 PHASE 2: two held tracks may walk at once — their per-arm
            # slots MERGE below (an arm owns at most one track, so the keys
            # never collide). This also closes a PRE-EXISTING leak: the waiting
            # arm used to fall through to build_arm_packet and receive a
            # CARTESIAN home-glide slot mid-staircase (an INC-1-class mode flip
            # for that arm).
            tm = h["track"]
            payload, status = op._step_toward(tm, jpos, now)
            if payload is None and status != "done":
                continue     # telemetry blip: tick() silences the arm packet
            if h["grasp"] == "grasped":
                # CARRY home: the cube is IN the hand, so every cube-position
                # decision (re-extend / reverse / release) is meaningless —
                # detections track the hand. Just walk the staircase to front;
                # at 'done' park at REST with the cube (gentle Cartesian ramp).
                if tm.get("timed_out") and \
                        now - h.setdefault("carry_warn_t", -1e9) > 5.0:
                    h["carry_warn_t"] = now
                    print(f"[grasp] WARNING: {arm} carry staircase hop timed "
                          f"out (jam/sag?) — retrying toward front at crawl "
                          f"rate with {h['key']} in hand; watch the arm")
                if status == "done":
                    ee = (telemetry.get("ee") or {}).get(arm) or {}
                    act = ee.get("pos_actual")
                    if act is not None:
                        h["track"] = None
                        h["sector"], h["sector_idx"] = \
                            _K(op).TRACK[op._home_idx], op._home_idx
                        h["home"] = False
                        print(f"[grasp] {arm} arm is home with {h['key']}")
                        op._enter_carried(h, arm, act)
                    continue      # (no EE this tick: stay settled, retry)
                ret_payload = {**(ret_payload or {}), **payload}
                continue
            if status in ("settled", "done"):
                ok, sect = op._cube_servable(cube, arm, now)
                if h.get("precon_release", False):
                    # 07-26: forced not-servable — the retreat must run to the
                    # settled-at-home release, never re-extend or turn around.
                    ok = False
                if ok and sect == tm["cur"]:
                    # settled AT the cube's sector station → re-extend (Cartesian,
                    # gentle 8 cm/s ramp). Needs the live EE to seed the ramp; if
                    # telemetry lacks it this tick, stay settled and retry.
                    ee = (telemetry.get("ee") or {}).get(arm) or {}
                    act = ee.get("pos_actual")
                    if act is not None:
                        h["track"] = None
                        h["sector"], h["sector_idx"] = _K(op).TRACK[sect], sect
                        h["home"], h["wrong_t0"], h["state"] = False, None, "tracking"
                        h["restream"] = np.asarray(act, dtype=float)
                        h["target"] = op._standoff_point(cube.pos)
                        print(f"[hold] {arm} re-extends to {h['key']} "
                              f"(back at the {h['sector']} station)")
                        continue
                elif ok and tm["target"] != sect:
                    # cube back in a different sector → turn around IN PLACE toward
                    # it from this station; ownership kept, nothing re-queued.
                    op._retarget_track(tm, sect, now)
                    print(f"[hold] {arm} reverses toward {_K(op).TRACK[sect]} — "
                          f"{h['key']} back")
                elif not ok and tm["target"] == op._home_idx \
                        and tm["cur"] == op._home_idx and not tm["seq"]:
                    # settled at HOME, still no cube → the only place we release
                    print(f"[hold] {arm} arm releases {h['key']} (retreated to "
                          f"{_K(op).TRACK[op._home_idx]}) — cube re-queued")
                    cube.held_by = None
                    del op._held[arm]
                    if op.phase == _K(op).Phase.HOLD:
                        op.phase = _K(op).Phase.SEARCH
                    continue
                elif not ok and status == "done" and tm["target"] != op._home_idx:
                    # settled at an off-home station (a turnaround's goal) with
                    # no servable cube: head home. Without this the track is an
                    # ABSORBING state — no branch fires, the arm strands at
                    # an outer station and the stream stays silent forever
                    # (07-15 audit). Walking home re-enters the release branch.
                    op._retarget_track(tm, op._home_idx, now)
                    print(f"[hold] {arm} arm heads home — {h['key']} gone at "
                          f"the {_K(op).TRACK[tm['cur']]} station")
            if status != "done":
                ret_payload = {**(ret_payload or {}), **payload}
            continue

        if getattr(op, "independent_arms", False):
            # 07-20 PER-ARM PROTOCOL: a peer's joint staircase no longer freezes
            # this hold — its live Cartesian slot rides the same mixed packet
            # (arbitration attaches it), so follow/retrack/lift ramps advance
            # normally and grabs are never dispatched against a frozen target
            # (the 07-20 empty-pinch forensics, cause #1). The ONLY remaining
            # deferral: the two-slot Cartesian preflight denied the packet
            # (missing finite encoder value) — ramps must not advance behind
            # publish silence or the resumed command jumps.
            cartesian_denied = not getattr(
                op, "_ind_cartesian_grant", True)
            if cartesian_denied:
                # Gripper commands use a separate bus and may still be sent.
                # Pause wall-clock motion/result timers while their state
                # processing is deliberately deferred behind the arm wire or
                # an incomplete two-slot Cartesian packet.
                if h["grasp"] in (None, "failed"):
                    op._maybe_send_grab(arm, h, cube, telemetry, now)
                elif h["grasp"] == "lifting":
                    h["lift_t0"] += _K(op).DT
                elif h["grasp"] == "sent":
                    h["grasp_t0"] += _K(op).DT
                continue
        # ---- grasp lifecycle (--grasp): trigger → result → carry → park.
        #      The grab itself never touches the joint stream (gripper servos
        #      are a separate serial bus), so it may overlap the other arm's
        #      reach; only the CARRY (a normal track) queues behind it. ----
        if op.grasp and h["grasp"] == "carried":
            # home at REST with the cube: finish the gentle park ramp; no
            # following, no releases — terminal per-arm state (unless the
            # cube is seen DROPPED, below).
            if op._grasp_dropped(h, cube, arm, telemetry, now):
                continue
            if h["restream"] is not None:
                h["restream"], _ = op._gentle_approach(
                    h["restream"], np.asarray(h["target"], dtype=float), None)
                if float(np.linalg.norm(
                        h["restream"] - np.asarray(h["target"]))) < 1e-9:
                    h["restream"] = None
            continue
        if op.grasp and h["grasp"] == "lifting":
            # clearance move: gentle ramp through the lift waypoints; the
            # arm slot publishes h["restream"] via _arm_packet. The legacy
            # coordinator pauses it during another GO/REACH; the independent
            # scheduler lets it share Cartesian ticks and pauses it only while
            # a global joint staircase actually owns the wire.
            if op._reach_tm is not None:
                h["lift_t0"] += _K(op).DT
                continue               # 07-21 PHASE 2: only the LEGACY 14-joint
                                       # reach staircase still suppresses our
                                       # Cartesian slot. A peer's per-arm
                                       # staircase does not — arbitration merges
                                       # our slot alongside it, so the ramp may
                                       # advance (it is genuinely published).
            if (now - h["lift_t0"]) > _K(op).GRASP_LIFT_TIMEOUT_S:
                print(f"[grasp] WARNING: {arm} clearance move timed out — "
                      f"carrying home from here")
                h["grasp"] = "grasped"
                h["restream"] = None
                continue
            if h["restream"] is None:
                act = ((telemetry.get("ee") or {}).get(arm)
                       or {}).get("pos_actual")
                if act is not None:
                    h["restream"] = np.asarray(act, dtype=float)
                continue               # no EE seed yet: hold, retry
            tgt_wp = np.asarray(h["target"], dtype=float)
            h["restream"], _ = op._gentle_approach(h["restream"], tgt_wp, None)
            if float(np.linalg.norm(h["restream"] - tgt_wp)) < 1e-9:
                h["lift_wps"].pop(0)
                if h["lift_wps"]:
                    h["target"] = h["lift_wps"][0]   # next waypoint; restream
                                                      # continues from here
                else:
                    h["grasp"] = "grasped"
                    h["restream"] = None
                    print(f"[grasp] {arm} clear of the stand — carrying home")
            continue
        if op.grasp and h["grasp"] == "grasped":
            # waiting to carry home. 07-21 PHASE 2: the carry staircase
            # starts immediately — it owns only its own arm's slot and the
            # dual-staircase clearance audit cleared the geometry.
            if op._grasp_dropped(h, cube, arm, telemetry, now):
                continue
            if h["track"] is None and op._reach_tm is None:
                if h["sector_idx"] != op._home_idx:
                    if op._sec is None:
                        # 07-21 PHASE 2: a carry staircase no longer waits for
                        # peer task motion — the wire is per-arm and the
                        # dual-staircase clearance audit cleared the geometry.
                        # (_sec is the legacy dual-parallel secondary, dead in
                        # independent mode, kept for the classic coordinator.)
                        h["track"] = op._new_track(
                            arm, h["sector_idx"], op._home_idx, now,
                            replant=True)  # carry starts at the ARC END, not
                                           # the station: re-plant it first
                        print(f"[grasp] {arm} arm carries {h['key']} home "
                              f"({h['sector']}→{_K(op).TRACK[op._home_idx]} "
                              f"staircase)")
                else:
                    ee = (telemetry.get("ee") or {}).get(arm) or {}
                    act = ee.get("pos_actual")
                    if act is not None:
                        op._enter_carried(h, arm, act)
            continue
        if op.grasp and h["grasp"] == "sent":
            # gripper is closing: the arm HOLDS STILL (no follow — the fingers
            # occlude the tags, and a mid-grab retract would yank the cube).
            # RESULT POLICY (07-16 audit): only a FRESH result is trusted. A
            # timeout/unchanged/None result means the status packet was lost,
            # the chain died, or real_env restarted — the gripper MAY be
            # holding the cube, so we never hand_open on uncertainty (that
            # used to DROP a successfully grasped cube).
            ee_result = (telemetry.get("ee") or {}).get(arm) or {}
            cur = ee_result.get("grasp_detected")
            alive = op._ee_alive(telemetry)
            # New protocol: a terminal result is accepted only for the stable
            # ID assigned to THIS hand_grab burst.  Presence of the field means
            # the peer supports IDs; None/mismatch is therefore "still waiting",
            # never a reason to consume a stale boolean.  Old deployments omit
            # the key entirely and retain the conservative boolean-edge fallback.
            result_has_id = "grasp_action_id" in ee_result
            result_id = ee_result.get("grasp_action_id")
            expected_id = h.get("grasp_action_id")
            terminal_result = (expected_id is not None
                               and result_id == expected_id) if result_has_id \
                else (cur is not None and cur != h["grasp_base"])
            if terminal_result:
                if cur is True:
                    h["bias"] = None       # servo bias aimed at the cube; the
                                           # carry targets REST — a stale bias
                                           # would park |bias| off and step-jump
                    base = np.asarray(h["target"], dtype=float)
                    wp1 = base + np.array([0.0, 0.0, _K(op).GRASP_LIFT_UP_M])
                    wps = []
                    if op._target_safe(wp1):
                        wps.append(wp1)
                        if not op.side_home \
                                or h["sector_idx"] != op._home_idx:
                            # classic legs AND staged side-home legs keep the
                            # audited clearance arc toward their own station
                            # (07-17 user spec; S-A 07-18: arc + inward
                            # re-plant hop from the arc end verified clean).
                            # DIRECT side-home legs: no arc — the straight
                            # gentle ramp to the SIDE home IS the sweep
                            # (user 07-18 spec).
                            wps += op._lift_arc(wp1, arm, h["sector"])
                    if wps:
                        act = ((telemetry.get("ee") or {}).get(arm)
                               or {}).get("pos_actual")
                        h["grasp"] = "lifting"
                        h["lift_wps"] = wps
                        h["lift_t0"] = now
                        h["target"] = wps[0]
                        h["restream"] = None if act is None else \
                            np.asarray(act, dtype=float)
                        arc_cm = sum(
                            float(np.linalg.norm(b[:2] - a[:2]))
                            for a, b in zip(wps, wps[1:])) * 100.0
                        print(f"[grasp] {arm} gripper HOLDS {h['key']} — "
                              f"lifting clear of the stand "
                              f"(+{_K(op).GRASP_LIFT_UP_M*100:.0f} cm up"
                              + (f", ~{arc_cm:.0f} cm arc toward the "
                                 f"{h['sector']} station"
                                 if len(wps) > 1 else
                                 (" — straight ramp home to SIDE follows"
                                  if op.side_home else
                                  "; arc TRUNCATED by the sector/envelope "
                                  "guard — lifting straight up only")) + ")")
                    else:
                        h["grasp"] = "grasped"
                        print(f"[grasp] {arm} gripper HOLDS {h['key']} — will "
                              f"carry home (lift skipped: envelope)")
                elif cur is False and result_has_id and alive is not False:
                    # 07-26 deadlock audit: a service-CERTIFIED empty pinch
                    # (id-matched terminal result from a live EE chain,
                    # grasp_detected=False = the load-supervised close reached
                    # its calibrated closed position with no load) is the one
                    # verdict that IS proof the fingers are empty — reopening
                    # cannot drop a held cube. This restores the documented
                    # 07-21 contract (GRASP_MAX_TRIES removed: empty pinches
                    # retry WITHOUT LIMIT while the cube is present) that the
                    # 80b997c fail-closed change accidentally revoked — one
                    # off-centre pinch used to freeze the whole mission.
                    # Ambiguous verdicts (None value, dead chain, timeout)
                    # still fail closed below.
                    h["grasp_tries"] += 1
                    op._queue_ee_action({"side": arm, "command": "hand_open"})
                    h["grasp"] = None
                    h["grasp_action_id"] = None
                    h["grasp_t0"] = None
                    h["grasp_move_t"] = now   # restart the settle window so
                    # _maybe_send_grab re-fires only after the hand reopens
                    # and the hold re-settles on the still-present cube.
                    print(f"[grasp] {arm} EMPTY PINCH on {h['key']} "
                          f"(try {h['grasp_tries']}) — reopening and "
                          f"retrying (no limit while the cube is present)")
                else:
                    # A None verdict (or a boolean-edge fallback without IDs)
                    # is NOT proof that the fingers are empty: load telemetry
                    # and status delivery can be wrong while the cube is
                    # physically held.  Never auto-release.
                    h["grasp_tries"] += 1
                    latch_grasp_uncertain(
                        op, h, arm,
                        f"hand_grab action {expected_id!r} returned "
                        f"grasp_detected={cur!r} for {h['key']}", telemetry)
            elif alive is False:
                op._ee_chain_dead = True
                latch_grasp_uncertain(
                    op, h, arm,
                    f"EE service lost during hand_grab action {expected_id!r} "
                    f"for {h['key']}", telemetry)
            elif (now - h["grasp_t0"]) > _K(op).GRASP_RESULT_TIMEOUT_S:
                latch_grasp_uncertain(
                    op, h, arm,
                    f"no matching result for hand_grab action "
                    f"{expected_id!r} within "
                    f"{_K(op).GRASP_RESULT_TIMEOUT_S:.0f}s "
                    f"(value={cur!r}, result_id={result_id!r}, "
                    f"ee_alive={alive!r})", telemetry)
            continue
        if op._maybe_send_grab(arm, h, cube, telemetry, now):
            continue
        # 07-26 deadlock audit (redesigned after review): pre-grab convergence
        # timeout. If the ONLY thing stopping the grab is the EE-distance
        # precondition and it has persisted GRASP_PRECON_RETRY_S, the hold can
        # never grab and used to block the base forever. We do NOT release in
        # place (an extended freed arm is the 07-18 strand/cross-basin class,
        # and journey's posture glide must never joint-lerp from an unaudited
        # pose) and we do NOT just set h["home"] (the servable follow flips it
        # back next tick — review-confirmed ping-pong). Instead latch
        # precon_release: while set, every servability check treats the cube
        # as NOT servable, so the audited not-servable machinery runs — direct
        # holds retract to REST and release once measured-at-home, staged
        # holds walk their retreat staircase and release in the settled-at-
        # home branch ("the only place we release"). The cube re-queues and
        # the reach retries without limit.
        if h["grasp"] is None and not h.get("precon_release") \
                and h.get("ee_far_t0") is not None \
                and op._ee_alive(telemetry) is not False \
                and (now - h["ee_far_t0"]) > _K(op).GRASP_PRECON_RETRY_S:
            h["ee_far_t0"] = None
            h["precon_release"] = True
            print(f"[hold] {arm} EE stuck past grab tolerance for "
                  f"{_K(op).GRASP_PRECON_RETRY_S:.0f}s — retreating and "
                  f"re-queueing {h['key']} (forced not-servable until "
                  "released)")
        # 07-21 (user): GRASP-WINDOW IMMUNITY — with the hand essentially at
        # the cube (standoff 0) the gripper occludes the tag and the learned
        # gate chatters; both used to yank the arm back mid-approach (the
        # halfway-retract limit cycle the user kept seeing). Near the target:
        # the lost-grace stretches and a gate flicker alone cannot retract.
        # Hard geometry (midline / sector / stage-class / bearing jump) stays
        # fully live — those are safety rules, not perception noise.
        _act_now = ((telemetry.get("ee") or {}).get(arm) or {}).get("pos_actual")
        near_grasp = (op.grasp and h["grasp"] is None
                      and _act_now is not None
                      and bool(np.all(np.isfinite(np.asarray(_act_now, float))))
                      and float(np.linalg.norm(
                          np.asarray(_act_now, float)
                          - np.asarray(h["target"], float)))
                      < _K(op).GRASP_NEAR_M)
        seen = cube.fresh(now, _K(op).GRASP_NEAR_LOST_S if near_grasp
                          else _K(op).DYN_LOST_GRACE_S)
        release_why = None
        wrong_side = sector_left = staged_jump = class_left = False
        if seen:
            live = cube.pos.copy()
            wrong_side = (live[1] < -_K(op).MIDLINE_MARGIN_M) if arm == "left" \
                else (live[1] > _K(op).MIDLINE_MARGIN_M)
            staged_hold = h["sector_idx"] != op._home_idx
            # ONE arbiter per hold (07-18 recon): a STAGED hold answers to the
            # sector machinery — sector-exit release → fresh staged leg, track
            # retreats — and NEVER to the via-home Cartesian detour; a DIRECT
            # side-home hold keeps the via-home bearing rule and never starts
            # a track. Without the split the two mechanisms fight over the
            # same events and the arm ping-pongs station↔side-home.
            sector_left = (not op.side_home or staged_hold) \
                and op._sector_left(live, h["sector"])
            # VIA-HOME rule (side-home, user 07-18 hard requirement): a big
            # bearing jump (cube moved front→rear) NEVER slides the target
            # across — the arm retracts to the SIDE home and may re-extend
            # ONLY once the measured EE is physically AT home (the home is
            # the stage). via_home latches so a mid-retract sighting cannot
            # shortcut the detour.
            at_home = (op._ee_act.get(arm) is not None
                       and bool(np.all(np.isfinite(
                           np.asarray(op._ee_act[arm], float))))
                       and float(np.linalg.norm(
                           np.asarray(op._ee_act[arm], float)
                           - np.asarray(op.rest_ee[arm], float))) < 0.10)
            jump = (op.side_home
                    and abs(op._bearing_deg(op._standoff_point(live))
                            - op._bearing_deg(h["target"]))
                    > _K(op).BEARING_JUMP_DEG)
            via_block = op.side_home and not staged_hold \
                and (jump or h.get("via_home", False)) and not at_home
            # STAGED holds obey the same NEVER-CHASE-A-BIG-JUMP hard rule,
            # but their protected route is the STAIRCASE retreat (the
            # one-arbiter rule), not the via-home Cartesian detour: a >45°
            # in-sector jump (the sector clamp fires only past the band+10°
            # hysteresis, so up to ~70° was uncovered — 07-18 review) drops
            # the follow, the hold retracts, and the settled-station
            # machinery re-extends from the right basin.
            staged_jump = staged_hold and jump
            # Routing-class crossing (07-18 review): a followed cube sliding
            # across FRONT_STAGE_Z_M would drag a DIRECT target into
            # must-stage territory (or a staged one out) un-staged — the
            # exact elbow-bifurcation class the flag exists for. Handled as
            # a timed release (like sector_left): the re-queued cube gets a
            # freshly ROUTED leg.
            class_left = op.side_home and op.stage_sectors \
                and op._staged_target_idx(
                    h["sector"], op._standoff_point(live)) != h["sector_idx"]
            servable = not wrong_side and not sector_left and not class_left \
                    and not via_block and not staged_jump \
                    and not h.get("precon_release", False) \
                    and (op._dyn_reachable(live, arm, h) or near_grasp)
            # ^ precon_release (07-26): a hold whose grab preconditions timed
            #   out is forced NOT servable until it has retreated and
            #   released — otherwise this very branch flips h["home"] back
            #   and re-extends to the same unconvergeable target forever.
            # A grasp-pending cube can disappear long enough for this direct
            # hold to retract fully, allowing the peer to use the base.  If it
            # reappears while that shared base is still moving, request a stop
            # first and keep the arm at home; extending into a moving body
            # frame would turn independence into a base/arm race.
            wait_for_base_stop = servable and h["home"] \
                and getattr(op, "independent_arms", False) and op.grasp \
                and h["grasp"] is None \
                and (abs(op._vx) >= 0.02 or abs(op._wz) >= 0.02)
            if wait_for_base_stop:
                h["resume_pending"] = True
                h["wrong_t0"] = None
                h["restream"] = None
            elif servable:
                h["resume_pending"] = False
                if h.get("via_home"):
                    print(f"[hold] {arm} is AT the side home — re-extending "
                          f"to {h['key']} from the stage")
                    h["via_home"] = False
                if h["home"] and h["restream"] is None:
                    # re-extend from a retract: approach at the gentle rate
                    # instead of snapping the full REST→cube jump in one
                    # packet (same catapult/knock-the-cube issue as reach)
                    act_re = ((telemetry.get("ee") or {}).get(arm)
                              or {}).get("pos_actual")
                    if act_re is not None:
                        h["restream"] = np.asarray(act_re, dtype=float)
                h["home"] = False
                h["wrong_t0"] = None
                live = op._standoff_point(live)
                if float(np.linalg.norm(live - h["target"])) > _K(op).DYN_RETRACK_M:
                    h["target"] = live         # cube moved: follow it
                    h["grasp_move_t"] = now    # grasp settle timer restarts
                if h["restream"] is not None:  # finish a re-extend gentle ramp
                    h["restream"], _ = op._gentle_approach(
                        h["restream"], np.asarray(h["target"], dtype=float), None)
                    if float(np.linalg.norm(
                            h["restream"] - np.asarray(h["target"]))) < 1e-9:
                        h["restream"] = None
            else:
                h["resume_pending"] = False
                h["home"] = True               # visible but not servable by this arm
                h["restream"] = None
                if jump and not staged_hold and not h.get("via_home"):
                    h["via_home"] = True
                    print(f"[hold] {arm} target bearing jumped "
                          f">{_K(op).BEARING_JUMP_DEG:.0f}° — routing VIA the side "
                          f"home before re-extending (never across the body)")
                if wrong_side or sector_left or class_left \
                        or h["best_arm"] not in (None, arm):
                    if h["wrong_t0"] is None:
                        h["wrong_t0"] = now
                    elif now - h["wrong_t0"] > _K(op).HOLD_RELEASE_S:
                        release_why = "crossed the midline" if wrong_side else (
                            f"left the {h['sector']} sector" if sector_left
                            else ("crossed the stage-z boundary — re-queued "
                                  "for a re-routed leg" if class_left
                                  else f"gate favors the {h['best_arm']} arm"))
                else:
                    h["wrong_t0"] = None
        else:
            h["resume_pending"] = False
            h["home"] = True                   # unseen past the grace window
            h["wrong_t0"] = None
            h["restream"] = None
        state = "retracted" if h["home"] else "tracking"
        if state != h["state"]:
            why = "" if not h["home"] else (
                " (unseen)" if not seen else
                (" (crossed midline)" if wrong_side else
                 (f" (left {h['sector']} sector)" if sector_left else
                  (" (crossed the stage-z boundary)" if class_left else
                   (" (bearing jumped >45° — staged retreat, re-extend from "
                    "the station)" if staged_jump else
                    (f" (gate favors {h['best_arm']} arm)"
                     if h["best_arm"] not in (None, arm)
                     else " (out of reach)"))))))
            print(f"[dyn] {arm} → {state}{why}")
            h["state"] = state
        # STAGED holds: retraction is a _K(op).TRACK motion toward HOME, now
        # INTERRUPTIBLE (state A re-decides at each settled station). Direct
        # holds (sector_idx == home) keep the classic in-place REST retract.
        # Evaluated EVERY tick, not only on the state edge: under stage-sectors
        # a staged hold CAN coexist with a dual-parallel secondary (the classic
        # by-construction exclusion no longer holds) — if the _sec gate blocks
        # the track start on the edge tick, we must RETRY when the wire frees,
        # never fall through to the Cartesian home glide (cross-basin sweep;
        # build_arm_packet holds the arm in place meanwhile).
        # 07-21 PHASE 2: the retreat staircase starts regardless of peer task
        # motion (per-arm wire + cleared dual-staircase geometry).
        if h["home"] and h["track"] is None \
                and h["sector_idx"] != op._home_idx and op._sec is None:
            h["track"] = op._new_track(arm, h["sector_idx"], op._home_idx, now,
                                         replant=True)
            # (replant: the retreat starts from the drifted cube-follow pose,
            #  never exactly at the station — same rule as the classic inward
            #  re-plant, made explicit so it survives home_idx=1)
            continue
        # 07-26: precon_release DIRECT holds release only once the measured EE
        # is back at REST — never in place with the arm extended (the 07-18
        # strand/cross-basin class; journey's posture glide must never see a
        # freed arm at an unaudited pose). Staged holds release through their
        # settled-at-home track branch above.
        if h.get("precon_release", False) and h["track"] is None \
                and h["sector_idx"] == op._home_idx:
            act_h = op._ee_act.get(arm)
            if act_h is not None \
                    and bool(np.all(np.isfinite(np.asarray(act_h, float)))) \
                    and float(np.linalg.norm(
                        np.asarray(act_h, float)
                        - np.asarray(op.rest_ee[arm], float))) < 0.10:
                print(f"[hold] {arm} arm releases {h['key']} (grab "
                      "preconditions stuck; retreated to rest) — cube "
                      "re-queued")
                cube.held_by = None
                del op._held[arm]
                if op.phase == _K(op).Phase.HOLD:
                    op.phase = _K(op).Phase.SEARCH
            continue
        if release_why:
            if h["sector_idx"] != op._home_idx:
                # STAGED hold: an in-place release would strand the extended
                # arm (off-home idle arms are pinned in place forever and the
                # keep-alive stays suppressed — 07-18 review). Walk home
                # first: the retreat track runs, and the settled-at-home
                # track branch performs the actual release ('the only place
                # we release'). The timer is cleared so this fires once.
                if not h["home"]:
                    print(f"[hold] {arm} arm retreats home before releasing "
                          f"{h['key']} ({release_why}) — re-queues from home")
                    h["home"] = True
                    h["restream"] = None
                h["wrong_t0"] = None
                continue
            print(f"[hold] {arm} arm releases {h['key']} ({release_why}) — "
                  f"cube re-queued")
            cube.held_by = None
            del op._held[arm]
            if op.phase == _K(op).Phase.HOLD:
                op.phase = _K(op).Phase.SEARCH
    return ret_payload
