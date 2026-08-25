"""Command-wire arbitration for the auto-operator (refactor stage S4).

Owns the assembly and prioritization of everything published on 9874:
per-arm Cartesian slots, the joint-staircase suppression, the keep-alive,
and the gripper-action repeat burst. Extraction contract: every function is
the VERBATIM body of the corresponding operator method / tick() block with
`self` -> `op`; cross-calls go through the operator's delegate shells so the
dispatch graph is unchanged. Priority order (SAFE-ARBITRATION-*):

    ret_payload (active joint track)          — the wire is JOINT-owned
  > staircase_active Cartesian suppression    — never flip modes mid-track
  > the phase handler's own packet
  > keep-alive REST/home glide                — only from verified-home basins
  > silence                                   — robot holds its last target

Incident anchors: INC-1 (cross-body override — the mutual exclusion this
module enforces), INC-2 (stale-IK snap — `_arm_engaged` never-silent rule),
INC-6 (catapult — the home glide), INC-7 (glide freeze — the active-stream
posture-gate exemption).
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .models import GripperBurst

NEUTRAL_QUAT = [1.0, 0.0, 0.0, 0.0]


def free_arms_at_tuck(op) -> bool:
    """True when a FREE (unheld) arm rests at the power-on CHEST tuck instead
    of the side home.

    Two modes want it:
      * JOURNEY, while the walk pose is active (arms tuck to walk, then untuck
        to side-home on arrival);
      * ``--chest-home`` (07-24 photo/standing session): the arms NEVER glide to
        side-home — they hold the power-on chest pose until a cube's DIRECT
        Cartesian reach seeds straight from it ("chest -> cube -> grasp").

    Deliberately implemented WITHOUT touching ``journey_posture_payload``: that
    function stays gated on ``journey``, so a standing chest-home mission never
    activates the joint-posture payload at all. Membership of that payload is an
    arm's control MODE, and on this build it has neither the REACHING exclusion
    nor the hysteresis band, so waking it in a standing mission would reproduce
    the reach-vs-posture judder this build is free of.
    """
    return bool(getattr(op, "_journey_walk_pose", False)
                or getattr(op, "chest_home", False))


def home_glide(op, arm: str, ee_rate: float) -> list | None:
    """Gentle ee_rate ramp of an arm's HOME target (idle REST and the
    dyn-retract). Publishing op.rest_ee RAW let the robot-side IK sweep
    the whole distance at its 0.25 m/s cap — an invisible 8 cm hop in the
    front mode, but a violent-looking ~40 cm sweep to the side home
    (hardware 07-18: "arms shot out to the sides at startup"). Seeds from
    the measured EE; None until telemetry provides a seed (caller goes
    silent, robot holds pose). The ramp state resets whenever the arm's
    slot is owned by anything other than home (reach/hold/carry), so every
    fresh homing re-seeds from where the arm actually is."""
    if getattr(op, "_journey_walk_pose", False) and arm not in op._held:
        # JOURNEY walking tuck (07-21): free arms glide to the power-on EE
        # (since 07-31 the sim home — the policy's LOW training default was
        # retired with the one-home unification, see JOURNEY_WALK_EE) instead
        # of the wide side-home while the base is driving; released on arrival
        # and the same glide walks them back to side-home for the reach.
        # (CHEST-HOME needs no branch here: it puts the tuck in op.rest_ee
        # itself, so EVERY homing path — idle glide, held retract, carry —
        # lands at the chest instead of only the idle one.)
        import sys
        _mod = sys.modules[type(op).__module__]
        tgt = np.asarray(_mod.JOURNEY_WALK_EE, dtype=float).copy()
        if arm == "right":
            tgt[1] = -tgt[1]
    else:
        tgt = np.asarray(op.rest_ee[arm], dtype=float)
    seeding = op._home_stream[arm] is None
    op._home_stream[arm], pub = op._gentle_approach(
        op._home_stream[arm], tgt, op._ee_act.get(arm))
    if pub is None:
        return None
    if seeding:
        dist = float(np.linalg.norm(tgt - np.asarray(pub)))
        if dist > 0.03:               # announce real glides, not micro-nudges
            # Name the home by WHERE it actually is, not by the profile flag:
            # side-home's rest_ee can be relocated to the front (07-24 "B"), so a
            # hard-coded "SIDE" would mislead ("gliding to SIDE" while the target
            # is the front chest). y-dominant ⇒ a true side rest; x-dominant ⇒ front.
            home = "FRONT" if abs(tgt[0]) >= abs(tgt[1]) else "SIDE"
            print(f"[home] {arm} arm gliding to {home} home "
                  f"[{tgt[0]:.2f}, {tgt[1]:+.2f}, {tgt[2]:+.2f}] "
                  f"({dist*100:.0f} cm at {ee_rate*100:.0f} cm/s, "
                  f"~{dist/ee_rate:.0f}s)")
    return [float(v) for v in pub]


def build_arm_packet(op, reach_pos: np.ndarray | None = None,
                     reach_positions: dict[str, np.ndarray] | None = None) \
        -> dict | None:
    """arm_targets payload: per-arm slots — every HOLDING arm's live target
    (REST while that arm is dyn-retracted: the deliberate, audited front-basin
    retract), the actively reaching arm's point on top, and REST for idle arms
    ONLY when they verifiably sit in the front basin. An idle arm caught
    off-front (stranded retreat/abort) is held IN PLACE at its measured EE —
    never dragged home in one un-staged Cartesian packet (07-15 audit: the
    phase-handler packets used to bypass the _idle_arms_home guard). Returns
    None when no safe value exists for a slot (caller goes silent; real_env
    holds its last target). Orientation is NEUTRAL (yaw-to-face-the-cube is
    shelved in af97f1f, to return with the grasp work package)."""
    # home_glide mutates its ramp stream. If a later slot proves unpublishable,
    # roll both streams back so no Cartesian state advances behind silence.
    home_before = {
        arm: (None if op._home_stream[arm] is None else
              np.asarray(op._home_stream[arm], dtype=float).copy())
        for arm in ("left", "right")}

    def fail_packet():
        op._home_stream.update(home_before)
        return None

    ee_pos: list = [None, None]
    glided: set = set()
    for arm, h in op._held.items():
        if not h["home"]:
            # mid re-extend: publish the gentle-approach ramp point; else the
            # (bias-corrected) held target.
            if h.get("restream") is not None:
                tgt = h["restream"]
            else:
                tgt = h["target"] if h["bias"] is None else h["target"] - h["bias"]
            ee_pos[0 if arm == "left" else 1] = [float(v) for v in tgt]
        elif h["track"] is not None or h["sector_idx"] != op._home_idx:
            # (a) TRACK IN FLIGHT: this arm is walking a joint staircase — it
            #     must NEVER be handed a Cartesian point (mode flip mid-hop,
            #     INC-1 class). Pin it at its measured EE; arbitration then
            #     overwrites this slot with the arm's joint payload, and if
            #     that payload is missing the pin is the safe fallback.
            #     (07-21 audit: previously fell through to the home glide,
            #     reachable whenever two held tracks coexisted.)
            # (b) STAGED hold retracting, track not started yet: its way home
            #     is the STAIRCASE, never a Cartesian glide across the basin
            #     boundary. Hold in place until the track starts.
            act = op._ee_act.get(arm)
            if act is None or not np.all(np.isfinite(np.asarray(act, dtype=float))):
                return fail_packet()              # no safe value → stream silence
            ee_pos[0 if arm == "left" else 1] = [float(v) for v in act]
        else:
            g = op._home_glide(arm)
            if g is None:
                return fail_packet()              # no seed yet → stream silence
            ee_pos[0 if arm == "left" else 1] = g
            glided.add(arm)
    if reach_pos is not None:
        ee_pos[0 if op._reach_arm == "left" else 1] = [float(v) for v in reach_pos]
    if reach_positions:
        for arm, pos in reach_positions.items():
            if arm not in ("left", "right"):
                raise ValueError(f"unknown arm slot {arm!r}")
            i = 0 if arm == "left" else 1
            if arm in op._held:
                raise RuntimeError(f"{arm} cannot own a reach task and hold together")
            ee_pos[i] = [float(v) for v in pos]
    if op._sec is not None and op._sec["stream"] is not None:
        # dual-parallel secondary: its gentle-approach ramp point owns the
        # free arm's slot in every phase (an unseeded secondary leaves the
        # slot to the idle logic below — the arm has not moved yet)
        j = 0 if op._sec["arm"] == "left" else 1
        if ee_pos[j] is None:
            ee_pos[j] = [float(v) for v in op._sec["stream"]]
    for i, arm in enumerate(("left", "right")):
        if ee_pos[i] is not None:
            continue                              # slot already owned above
        task = getattr(op, "_arm_tasks", {}).get(arm)
        if task is not None and task["track"] is not None:
            # 07-21 PHASE 2: this arm is walking a task staircase — it must
            # not be home-glided (its slot is joint, and advancing a glide
            # ramp that never gets published would leave the stream metres
            # ahead by the time the arm returns to Cartesian). Pin it.
            act = op._ee_act.get(arm)
            if act is None or not np.all(np.isfinite(np.asarray(act, dtype=float))):
                return fail_packet()
            ee_pos[i] = [float(v) for v in act]
            continue
        if op._arm_near_front(arm, op._jpos) \
                or op._home_stream[arm] is not None:
            # An ACTIVE home-glide stream keeps publishing to completion:
            # mid-glide the joints traverse the no-man's-land between the
            # power-on and side-home postures, where the near-home check is
            # False for every reference — gating on it FROZE the glide 8 cm
            # out (hardware 07-18: pos_target stuck at [0.27, 0.29], arms
            # never reached the side). The glide itself is the safety
            # mechanism (seeded from the measured EE, 8 cm/s, straight
            # line); a STRANDED arm (no stream) is still held in place.
            g = op._home_glide(arm)
            if g is None:
                return fail_packet()              # no seed yet → stream silence
            ee_pos[i] = g
            glided.add(arm)
        else:
            act = op._ee_act.get(arm)
            if act is None or not np.all(np.isfinite(np.asarray(act, dtype=float))):
                return fail_packet()              # no safe value → stream silence
            ee_pos[i] = [float(v) for v in act]   # hold the off-front arm in place
    for arm in ("left", "right"):
        if arm not in glided:
            op._home_stream[arm] = None           # re-seed on the next idle spell
    quats = [list(NEUTRAL_QUAT), list(NEUTRAL_QUAT)]
    chest = getattr(op, "chest_home", False)
    if free_arms_at_tuck(op):
        # JOURNEY tuck (07-21 review): the power-on pose's EE orientation is
        # 17.4 deg off identity — publishing NEUTRAL at the tuck point makes
        # the IK twist the wrists away from the pose the tuck is meant to
        # reproduce. Give gliding tuck slots the FK orientation (right arm
        # mirrors by negating x and z).
        import sys
        _mod = sys.modules[type(op).__module__]
        # One orientation for both glide destinations: since 07-31 the walking
        # tuck and the chest home are the SAME posture (the sim home), and
        # JOURNEY_WALK_QUAT is an alias of CHEST_HOME_QUAT in the operator —
        # so chest-home glides and journey tuck glides cannot disagree.
        wq = list(_mod.JOURNEY_WALK_QUAT)
        for i, arm in enumerate(("left", "right")):
            # CHEST-HOME: a HELD arm glides to the same chest tuck (retract and
            # carry-home both target rest_ee), so it needs the tuck orientation
            # too. Journey keeps the free-arm-only rule: there a held arm homes
            # to side-home with its cube, where NEUTRAL is correct.
            if arm in glided and (chest or arm not in op._held):
                quats[i] = wq if arm == "left" else \
                    [wq[0], -wq[1], wq[2], -wq[3]]
    return {"ee_pos": ee_pos, "ee_quat": quats}


def per_arm_slots(at: dict | None) -> dict:
    """Normalize any arm_targets payload to {side: slot} form (07-21 PHASE 2).

    Accepts the legacy two-slot Cartesian form {"ee_pos": [l, r], "ee_quat":
    [...]}, the per-arm form {"left": {...}, "right": {...}}, and anything
    else (returns {} — callers then leave that packet alone). A legacy slot
    whose entry is None is simply absent."""
    if not isinstance(at, dict):
        return {}
    if "ee_pos" in at and isinstance(at.get("ee_pos"), list) \
            and len(at["ee_pos"]) == 2:
        out = {}
        for i, side in enumerate(("left", "right")):
            if at["ee_pos"][i] is not None:
                out[side] = {"ee_pos": at["ee_pos"][i],
                             "ee_quat": at["ee_quat"][i]}
        return out
    return {s: at[s] for s in ("left", "right")
            if isinstance(at.get(s), dict)}


def _joint_sides(at: dict | None) -> set:
    """Sides whose slot in `at` is JOINT content — legacy all-14 counts as both."""
    if at is None:
        return set()
    if "joint_pos" in at:
        return {"left", "right"}
    return {s for s in ("left", "right")
            if isinstance(at.get(s), dict) and "joint_pos" in at[s]}


def held_cart_slot(op, arm: str) -> list | None:
    """The holding arm's live Cartesian point, if it has one (07-20 protocol):
    mid re-extend/lift ramp -> restream, else the (bias-corrected) held target.
    None for non-held / retracted arms — an absent side makes real_env HOLD
    that arm, which is exactly the old frozen-snapshot semantics."""
    h = op._held.get(arm)
    if h is None or h["home"]:
        return None
    tgt = h["restream"] if h.get("restream") is not None else (
        h["target"] if h["bias"] is None else h["target"] - h["bias"])
    return [float(v) for v in tgt]


def intervention_hold_targets(op, jpos) -> dict | None:
    """Return one immutable two-arm emergency hold snapshot.

    A grasp result marked uncertain is a human-intervention state: no scheduler
    is allowed to advance and neither arm may inherit a stale reach/home target.
    Pin both arms at the measured joints from the latch tick.  Reusing the first
    snapshot on every later tick keeps the command mode and target stable.  If
    joint telemetry is unavailable, replay the last packet that was actually
    published rather than inventing a Cartesian orientation.
    """
    snap = getattr(op, "_intervention_arm_targets", None)
    if snap is not None:
        return copy.deepcopy(snap)
    if jpos is not None:
        jp = np.asarray(jpos, dtype=float)
        if jp.shape[0] >= 27 and bool(np.all(np.isfinite(jp[13:27]))):
            snap = {
                "left": {"joint_pos": [float(v) for v in jp[13:20]],
                         "rate": 0.5},
                "right": {"joint_pos": [float(v) for v in jp[20:27]],
                          "rate": 0.5},
            }
    if snap is None:
        snap = getattr(op, "_last_arm_targets", None)
    if snap is None:
        return None
    op._intervention_arm_targets = copy.deepcopy(snap)
    return copy.deepcopy(snap)


def enqueue_ee_action(op, action: dict) -> int:
    """Queue one logical EE action and assign its stable ID exactly once."""
    payload = dict(action)
    if "action_id" not in payload:
        seq = getattr(op, "_ee_next_action_id", None)
        if seq is None:
            seq = time.time_ns()
        payload["action_id"] = int(seq)
        op._ee_next_action_id = int(seq) + 1
    op._ee_queue.append(payload)
    return int(payload["action_id"])


def arbitrate(op, arm_packet: dict, ret_payload: dict | None, jpos) -> None:
    """Packet arbitration (mutates arm_packet in place). A holding arm must
    keep receiving its target in EVERY phase (real_env re-engages IK on the
    last target after 0.5 s of arm-silence). 07-20 PER-ARM PROTOCOL: a joint
    staircase owns only ITS ARM's slot; the peer's live Cartesian slot is
    MERGED alongside instead of frozen out (the old global modal flag forced
    all-or-nothing, and grabs against frozen targets were the empty-pinch
    class). The one invariant kept from the old world: the staircase arm's
    OWN slot must never flip to Cartesian mid-track, so a phase packet with
    no joint content is still suppressed while a staircase is in flight.
    Silence on telemetry blips makes the robot HOLD ITS LAST TARGET (with
    --use-ik there is NO crawl-home failsafe — the 07-15 contract audit found
    that path is dead code): position held, nothing moves."""
    at = arm_packet.get("arm_targets")
    # 07-21 PHASE 2: staircase state is PER-ARM. `staircase_arms` is the set of
    # sides currently walking a joint hop (task tracks and held tracks alike);
    # the legacy 14-joint reach staircase (_reach_tm) still owns both.
    # "Walking" = the hop has actually STARTED (TrackMotion.frozen is the
    # motion-start sentinel set by the first step_toward). A track that is
    # merely pending — created this tick, or waiting out its one-publish
    # window — is still parked at its station, so a Cartesian slot for it is
    # not a mid-hop mode flip and must not be suppressed.
    def _walking(t) -> bool:
        return t is not None and t["track"] is not None \
            and t["track"]["frozen"] is not None
    staircase_arms = {a for a, t in getattr(op, "_arm_tasks", {}).items()
                      if _walking(t)}
    staircase_arms |= {a for a, h in op._held.items() if _walking(h)}
    # A task GRANTED the joint wire this tick counts as walking even before
    # its first step_toward sets `frozen` (production sets both in the same
    # call; this keeps the rule conservative if they ever separate).
    staircase_arms |= {a for a in getattr(op, "_ind_joint_owners", set())
                       if getattr(op, "_arm_tasks", {}).get(a) is not None}
    if op._reach_tm is not None:
        staircase_arms |= {"left", "right"}
    staircase_active = bool(staircase_arms)
    if ret_payload is not None:
        merged = dict(ret_payload)
        if at is not None and "ee_pos" in at:
            # legacy two-slot phase packet: carry the non-staircase side over
            for i, side in enumerate(("left", "right")):
                if side not in merged and at["ee_pos"][i] is not None:
                    merged[side] = {"ee_pos": at["ee_pos"][i],
                                    "ee_quat": at["ee_quat"][i]}
        elif at is not None:
            for side in ("left", "right"):
                if side not in merged and isinstance(at.get(side), dict):
                    merged[side] = at[side]
        arm_packet["arm_targets"] = merged
    elif staircase_active and not (staircase_arms <= _joint_sides(at)):
        # A staircasing side must never receive a Cartesian slot mid-hop
        # (mode flip). Suppress only when THAT side's joint content is
        # missing — a peer's Cartesian slot alone is fine (07-21 PHASE 2:
        # the old test was global, so one side's joint content wrongly
        # licensed the other side's Cartesian).
        arm_packet.pop("arm_targets", None)
    elif at is None:
        # keep-alive: explicit REST beats stream silence — 07-17 (user spec):
        # fires from MISSION START, not only after the first engagement, so
        # idle arms rise from the power-on pose to the home right after the
        # startup checks pass instead of waiting for the first reach. Same
        # safety gate as ever: only when every idle (unheld) arm is verifiably
        # near a home posture or riding an active home glide. An arm left in
        # the side/rear basin (timed-out retreat) must NOT be pulled home
        # across the torso by a Cartesian REST; going silent instead makes
        # real_env hold its current pose.
        keep_alive = op._arm_packet(None) if op._idle_arms_home(jpos) else None
        if keep_alive is not None:
            arm_packet["arm_targets"] = keep_alive
    # 07-20 protocol: a PER-ARM packet with an absent side gets the holding
    # arm's live Cartesian slot attached — holds keep following their cube
    # while the peer's staircase owns the joint side (the fix for grabs firing
    # against frozen targets). An absent side with no held slot stays absent:
    # real_env holds that arm in place.
    at_final = arm_packet.get("arm_targets")
    if at_final is not None and "joint_pos" not in at_final \
            and "ee_pos" not in at_final:
        for side in ("left", "right"):
            if side in at_final:
                continue
            slot = held_cart_slot(op, side)
            if slot is not None:
                at_final[side] = {"ee_pos": slot,
                                  "ee_quat": list(NEUTRAL_QUAT)}


def attach_ee_action(op, arm_packet: dict, repeat_ticks: int) -> None:
    """Gripper command burst (mutates arm_packet in place). Commands ride the
    same 9874 packet (real_env --ee-service forwards "ee_action"). Each
    command is RE-SENT for repeat_ticks packets — the channel is
    latest-value-wins with no ack at BOTH hops (07-16 audit; INC-3). Repeats
    are safe: the service rejects a busy side, open is idempotent. Yielding
    ROTATES the unsent remainder to the back of the queue instead of dropping
    it (07-17: dropping starved the FIRST of the back-to-back startup zeros
    to one 0.25 s burst a reconnecting subscriber never saw = "left gripper
    never opened"); a queued newer command for the same side supersedes the
    remainder instead of resurrecting behind it. ee_action does NOT refresh
    real_env's arm-stream clock."""
    if op._ee_current is not None and op._ee_queue and \
            op._ee_current[2] >= 5:
        cmd, remaining, _ = op._ee_current
        if remaining > 0 and not any(q.get("side") == cmd.get("side")
                                     for q in op._ee_queue):
            op._ee_queue.append({**cmd, "_ticks": remaining})
        op._ee_current = None
    if op._ee_current is None and op._ee_queue:
        nxt = op._ee_queue.pop(0)
        # One logical burst has ONE stable action ID.  The same ID survives
        # queue rotation/retransmission, so the EE service can reject exact
        # replays and the grasp FSM can correlate a terminal result to the
        # command that caused it.  A wall-clock seed avoids collisions when the
        # workstation operator restarts while real_env/service stay alive.
        if "action_id" not in nxt:
            # Compatibility for a legacy caller that appended directly instead
            # of using enqueue_ee_action(). New callers receive the ID at queue
            # time so their FSM can correlate the result before first publish.
            seq = getattr(op, "_ee_next_action_id", None)
            if seq is None:
                seq = time.time_ns()
            nxt["action_id"] = int(seq)
            op._ee_next_action_id = int(seq) + 1
        op._ee_current = GripperBurst(
            {k: v for k, v in nxt.items() if k != "_ticks"},
            int(nxt.pop("_ticks", repeat_ticks)), 0)
        cmd = op._ee_current[0]
        if cmd.get("command") == "hand_grab":
            h = op._held.get(cmd.get("side"))
            if h is not None and h["grasp"] == "sent":
                h["grasp_action_id"] = cmd["action_id"]
    if op._ee_current is not None:
        arm_packet["ee_action"] = dict(op._ee_current[0])
        op._ee_current[1] -= 1
        op._ee_current[2] += 1
        if op._ee_current[1] <= 0:
            op._ee_current = None
