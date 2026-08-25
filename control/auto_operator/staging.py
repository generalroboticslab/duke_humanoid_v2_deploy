"""Joint-space staging-track primitives (S6). SAFE-STAGING-*; INC-1. NEVER replace joint postures with Cartesian equivalents (7-DOF branch escape, audited -33.7 mm wrist-through-hip).

Extraction contract (S6): verbatim operator bodies, self -> op; constants and
Phase resolve LIVE against the owning operator module (importlib multi-load +
test monkeypatch safe). See docs/auto_operator_refactor_plan.md.
"""
from __future__ import annotations

import sys

import numpy as np

from .models import TrackMotion


def _K(op):
    return sys.modules[type(op).__module__]


def staged_joint_payload(op, arm: str, posture, frozen: list) -> dict:
    """PER-ARM joint payload (07-20 protocol, approved upstream): only the staircase
    arm's 7 joints ride the packet; the peer's slot is simply ABSENT — real_env
    holds an absent side's mode and targets, and final arbitration may attach
    the peer's live Cartesian slot alongside (mixed-mode packet). Replaces the
    old all-14 global-schema payload that pinned the peer at a frozen snapshot
    (`frozen` is retained in the signature for call-site compatibility; the
    receiver-side hold has identical semantics to the snapshot pin)."""
    return {arm: {"joint_pos": [float(v) for v in posture],
                  "rate": _K(op).STAGE_RATE}}


def step_toward(op, tm: TrackMotion, jpos, now: float) -> tuple[dict | None, str]:
    """One tick of the shared staging primitive. Walks `tm` one station at a
    time toward tm['target'], commanding the current hop's joint posture (the
    other arm frozen at the motion-start snapshot). Each station: encoder
    arrival (err < _K(op).STAGE_TOL_RAD) THEN a _K(op).STAGE_DWELL_S dwell — the identical
    'arrive-and-stabilize' event in BOTH directions. A hop that overruns its
    crawl-time budget flips the goal to the HOME station (op._home_idx), a
    plain direction change (tm['timed_out'] latches so the caller can react
    on arrival).

    Returns (joint_payload | None, status) where status is:
      'moving'  — still en route to / dwelling at the current hop
      'settled' — just arrived AND finished dwelling at a station (the ONLY
                  moment the caller may change direction / switch to Cartesian)
      'done'    — the track has reached tm['target'] and is settled there
    A None payload means a telemetry blip: caller keeps the stream silent."""
    if not tm["seq"]:
        return None, "done"
    if jpos is None or len(jpos) < 31:
        return None, "moving"                    # blip: silence, hold state
    arm = tm["arm"]
    mv = slice(13, 20) if arm == "left" else slice(20, 27)
    fz = slice(20, 27) if arm == "left" else slice(13, 20)
    if tm["frozen"] is None:                     # motion start: snapshot + budget
        tm["frozen"] = [float(v) for v in jpos[fz]]
        tm["hop_t0"] = now
        tm["hop_budget"] = float(np.max(np.abs(
            jpos[mv] - op._track_posture(arm, tm["seq"][0])))) / _K(op).STAGE_RATE * 1.4 + 5.0
    hop = tm["seq"][0]
    posture = op._track_posture(arm, hop)
    payload = op._staged_joint_payload(arm, posture, tm["frozen"])
    if tm["dwell_until"] is not None:            # dwelling at the hop station
        if now < tm["dwell_until"]:
            return payload, "moving"
        tm["dwell_until"] = None                 # dwell complete: this hop is settled
        tm["cur"] = hop
        tm["seq"].pop(0)
        if tm["seq"]:
            tm["hop_t0"] = now
            tm["hop_budget"] = float(np.max(np.abs(
                jpos[mv] - op._track_posture(arm, tm["seq"][0])))) / _K(op).STAGE_RATE * 1.4 + 5.0
        return payload, "settled"
    err = float(np.max(np.abs(jpos[mv] - posture)))
    if err < _K(op).STAGE_TOL_RAD:
        tm["dwell_until"] = now + _K(op).STAGE_DWELL_S   # arrived → stabilize before advancing
    elif (now - tm["hop_t0"]) > tm["hop_budget"]:
        # unreachable hop: flip the goal to the HOME station (a normal
        # direction change, NOT a special abort path) and rebuild the inward
        # sequence from here. Home = front(0) classically, the SIDE(1) under
        # stage-sectors — flipping a side-home arm toward FRONT on timeout
        # would walk it the wrong way past its rest (07-18 recon).
        hidx = op._home_idx
        tm["timed_out"] = True
        tm["target"] = hidx
        # Re-plant discipline: a timed-out arm is MID-HOP (drifted off every
        # station), so `cur` must be the first hop of the homeward sequence —
        # _track_seq skips it on numerically-outward flips (front(0)→home(1)
        # under stage-sectors) and returns [] for cur==home (a FIRST-hop
        # timeout would then read as an instant false "returned home",
        # 07-15 audit). Classic flips are inward, where this is a no-op.
        seq = op._track_seq(tm["cur"], hidx)
        if not seq or seq[0] != tm["cur"]:
            seq.insert(0, tm["cur"])
        tm["seq"] = seq
        tm["dwell_until"] = None
        tm["hop_t0"] = now
        tm["hop_budget"] = float(np.max(np.abs(
            jpos[mv] - op._track_posture(arm, tm["seq"][0])))) / _K(op).STAGE_RATE * 1.4 + 5.0
    return payload, "moving"

