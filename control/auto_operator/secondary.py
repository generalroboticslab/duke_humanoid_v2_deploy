"""Dual-parallel secondary reach (S6). SAFE-DUAL-*; catapult-abandon + mid-ramp handoff per INC-6.

Extraction contract (S6): verbatim operator bodies, self -> op; constants and
Phase resolve LIVE against the owning operator module (importlib multi-load +
test monkeypatch safe). See docs/auto_operator_refactor_plan.md.
"""
from __future__ import annotations

import sys

import numpy as np

from .models import SecondaryReach


def _K(op):
    return sys.modules[type(op).__module__]


def try_latch_secondary(op, now: float) -> None:
    """DUAL-PARALLEL (front+front ONLY): while the primary reach is in its
    Cartesian leg toward a FRONT cube, latch a second FRONT cube for the
    OTHER arm and fly both approaches at once. The Cartesian packet carries
    independent per-arm slots and the robot IK solves both arms together,
    so there is NO wire-mode conflict — unlike joint staircases, which stay
    strictly exclusive (side/rear legs never parallelize; the 07-15
    modal-wire contract is untouched). Latch rules mirror the primary:
    confirmed sighting, off-midline on the free arm's side, front sector,
    safety envelope, reachable from the current stance, median-of-window
    target. No visual servo on the secondary (bias stays None)."""
    if not op.hold_after_reach:
        return                        # legacy park-and-DONE mode has no hold
                                      # machinery — a secondary hold would
                                      # strand the mission in HOLD forever
                                      # (07-17 audit)
    if op._reach_tm is not None or not (op.side_home
                                          or op._reach_sector == "front"):
        return                        # primary must be wire-free: no staircase
                                      # in flight (a staged primary blocks here
                                      # until its track settles; its Cartesian
                                      # hop then coexists with a DIRECT
                                      # secondary — both pure Cartesian slots)
    if any(h["track"] is not None for h in op._held.values()):
        return                        # a joint stream owns the wire
    arm = "left" if op._reach_arm == "right" else "right"
    if arm in op._held:
        return
    for c in op.cubes.values():
        if c is op.active or c.held_by is not None or c.reached:
            continue
        if not c.confirmed(now) or abs(float(c.pos[1])) <= _K(op).MIDLINE_MARGIN_M:
            continue
        if ("left" if c.pos[1] > 0 else "right") != arm:
            continue
        if op._staged_target_idx(op._sector(c.pos),
                                   op._standoff_point(c.pos)) != op._home_idx \
                or not op._target_safe(c.pos):
            # A secondary is PURE Cartesian by design — it may only take legs
            # the router classifies as DIRECT (classic: front; side-home:
            # side + low-front; stage-sectors: same — high-front/rear need a
            # staircase, which only the primary machinery may run). Judged on
            # the STANDOFF point, the target actually flown, so this verdict
            # matches the sector_idx register_hold will record on arrival.
            continue
        if not op._reachable_now(c.pos, arm):
            continue
        window = [s.pos for s in c.hist
                  if (now - s.t) < _K(op).CUBE_LATCH_WINDOW_S]
        if not window:
            continue
        median = np.median(np.stack(window), axis=0)
        side_ok = (median[1] > _K(op).MIDLINE_MARGIN_M) if arm == "left" \
            else (median[1] < -_K(op).MIDLINE_MARGIN_M)
        if not side_ok or not op._target_safe(median) \
                or op._staged_target_idx(op._sector(median),
                                           op._standoff_point(median)) \
                != op._home_idx:
            continue
        # debounce (07-17 audit): the learned ReachabilityGate samples fresh
        # random candidates per call, so a single _reachable_now verdict
        # chatters — mirror the arrival gate's _K(op).GATE_CONSECUTIVE streak
        # before committing an arm to this cube
        if op._sec_hits is not None and op._sec_hits[0] == c.key:
            op._sec_hits[1] += 1
        else:
            op._sec_hits = [c.key, 1]
        if op._sec_hits[1] < _K(op).GATE_CONSECUTIVE:
            return
        op._sec_hits = None
        op._sec = SecondaryReach(arm=arm, key=c.key,
                                   target=op._standoff_point(median),
                                   t0=now)
        t = op._sec["target"]
        print(f"[reach2] DUAL-PARALLEL: {arm} arm reaches {c.key} "
              f"target=[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]  "
              f"r_xy={float(np.hypot(t[0], t[1])):.3f} m — both arms in flight")
        return
    op._sec_hits = None             # no qualifying candidate: streak broken


def tick_secondary(op, telemetry: dict, now: float) -> None:
    """Advance the dual-parallel secondary approach (pure Cartesian, no
    staircase, no servo). Its ramp point rides the shared packet via the
    _arm_packet slot injection, in EVERY phase. Arrival OR the reach
    timeout registers the hold — the same policy as the primary — after
    which the normal per-arm hold/grasp machinery owns the arm (first
    arrived starts grabbing immediately, no waiting for the other arm)."""
    s = op._sec
    arm = s["arm"]
    if op.dynamic_track:
        upd = op._live_retrack(op.cubes[s["key"]], arm,
                                 op._sector(s["target"]),
                                 s["target"], now)
        # (the secondary's REAL sector, not a hardcoded "front" — under
        #  side-home a SIDE cube is a legal secondary, and "front" would
        #  sector-clamp its retrack against a band it was never in;
        #  classic secondaries are front-only, so this resolves identically)
        if upd is not None:
            s["target"] = upd
    tgt = np.asarray(s["target"], dtype=float)
    actual = ((telemetry.get("ee") or {}).get(arm) or {}).get("pos_actual")
    s["stream"], pub = op._gentle_approach(s["stream"], tgt, actual)
    if pub is None and now - getattr(op, "_sec_diag_t", -1e9) > 1.0:
        op._sec_diag_t = now
        print(f"[reach2] WAITING: no ee.{arm}.pos_actual to seed the "
              f"secondary approach ramp")
    done = actual is not None and \
        float(np.linalg.norm(np.asarray(actual) - tgt)) < _K(op).REACH_OK_M
    if done:
        op._register_hold(arm, op.cubes[s["key"]], tgt,
                            op._sector(tgt) if op.side_home else "front",
                            None, now)
        op._sec = None
    elif (now - s["t0"]) > _K(op).REACH_TIMEOUT_S:
        if s["stream"] is None:
            # never seeded: the arm NEVER MOVED. Registering a hold here
            # would make the next packet publish the full REST→cube target
            # in ONE step (the audited 07-16 catapult class — the primary
            # is immune because it stalls before its timeout check).
            # Abandon; the cube stays pending for a classic serialized leg.
            print(f"[reach2] {arm} secondary reach to {s['key']} never got "
                  f"an EE seed in {_K(op).REACH_TIMEOUT_S:.0f}s — ABANDONED "
                  f"(cube re-queued for a classic leg)")
            op._sec = None
        else:
            # timed out mid-ramp: hand the hold the CURRENT ramp point so
            # the classic machinery finishes the approach gently
            op._register_hold(arm, op.cubes[s["key"]], tgt,
                                op._sector(tgt) if op.side_home
                                else "front",
                                None, now, restream=s["stream"])
            op._sec = None

