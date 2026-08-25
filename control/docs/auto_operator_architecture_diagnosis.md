# Auto-Operator — Architecture Diagnosis (pre-refactor)

Date: 2026-07-18 · Target: `control/humanoid_auto_operator.py` (~2 850 lines)
Scope: diagnosis only — no behavior change. Companion documents:
`auto_operator_safety_contract.md` (inventory), `auto_operator_refactor_plan.md`
(staged plan), `auto_operator_characterization_plan.md` (test foundation).

## 1. What the module is today

A single file containing five architectural layers that grew incident-by-incident:

1. **Perception bookkeeping** — `CubeTrack`, `_ingest`, camera claim/unclaim,
   gripper-tag sightings. Owns freshness, provenance (claim vs measurement
   port), NaN rejection, history for median latching.
2. **Gaze control** — FK-probed camera frames, closed-form aim + fixed-point
   refinement + integral trim, serpentine scan, sender-side slew, encoder
   seeding. Pure w.r.t. mission state except the shared `self.gaze`.
3. **Mission state machine** — `Phase` enum dispatched in `tick()`; handlers
   `_tick_search/_tick_go/_tick_reach/_tick_hold/_tick_park`. Owns target
   selection, arrival gating (learned gate + throttle/debounce), latching.
4. **Arm control** — the giant `_held_update` (per-arm hold lifecycle:
   follow/retract/re-extend/release + the entire grasp lifecycle
   None→sent→lifting→grasped→carried|failed), the staging track primitives
   (`_new_track/_step_toward/_retarget_track`), the dual-parallel secondary
   (`_try_latch_secondary/_tick_secondary`), home glides, via-home routing.
5. **Command assembly** — `_arm_packet` per-arm slot ownership, the packet
   arbitration block in `tick()` (ret_payload > staircase suppression >
   phase packet > keep-alive), the gripper command burst queue.

## 2. Why it is hard to maintain (the honest list)

- **State is a flat attribute soup**: ~45 mutable `self._*` fields plus
  per-hold dicts with 15+ string keys (`h["grasp"]`, `h["via_home"]`,
  `h["restream"]`, …). Nothing enforces which keys exist in which lifecycle
  state; `setdefault`/`get` scattering hides the schema.
- **`_held_update` is a 300-line multi-state handler** interleaving four
  concerns (guard-1 freeze, track stepping, grasp lifecycle, classic
  follow/release) with `continue`-based flow. Every new rule (via-home,
  bearing-jump, _sec gates) lands as another guard inside it.
- **Mode arbitration is implicit**: joint-vs-Cartesian exclusivity is enforced
  by five separate guards (`staircase_active` suppression, one-track token,
  Guard-1, Guard-2, `_sec` gates) whose interplay is documented only in
  comments. Ownership of the wire is never a first-class value.
- **Safety predicates are duplicated at call sites**: `_target_safe`,
  midline checks, sector checks appear inline at latch, follow, retrack,
  secondary latch, lift-arc — correct today, but every new flow must remember
  all of them (each of the 07-17/18 audits found one missed site).
- **Two mission profiles (classic staging vs side-home) share every code path
  via `self.side_home` branches** — 12+ conditional bypasses. Readable now,
  but each new feature must be reasoned for both modes.
- **Testing is behavioral-only**: `~/operator_tests/test_grasp_carry.py`
  (66 asserts) drives full scenarios through a fake rig. Excellent
  characterization; no unit isolation, no differential replay, no invariant
  checking.

## 3. What is GOOD and must not be lost

- The **pure decision core** boundary already exists:
  `tick(detections, telemetry, now) -> (nav, packet)` with zero I/O. This is
  the seam every extraction hangs from.
- **Incident knowledge lives next to the code** (dated, quantified comments).
  The refactor moves this into ADRs/incidents docs WITHOUT deleting specifics.
- The **fake-rig test suite** doubles as the characterization foundation.
- The gaze subsystem and staging primitives are already near-pure functions.

## 4. Target architecture (per the accepted plan)

```
control/auto_operator/            # new package; humanoid_auto_operator.py
    models.py                     # typed state (dataclasses/enums, invariants)
    safety.py                     # pure predicates + refusal reasons
    perception.py                 # tracks, claims, ingest
    gaze.py                       # aim/scan/trim/slew
    staging.py                    # track primitives (verbatim move)
    arms.py                       # per-arm state handlers (hold/grasp/lift/carry)
    secondary.py                  # dual-parallel secondary reach
    arbitration.py                # wire ownership + packet builder
    operator.py                   # AutoOperator façade (same public surface)
control/humanoid_auto_operator.py # stays: CLI + re-export, byte-compatible flags
control/docs/                     # this file + safety contract + ADR/incidents
~/operator_tests/replay/          # differential replay harness + scenarios
```

`humanoid_auto_operator.py` remains the entry point (`python
humanoid_auto_operator.py …` — the hardware recipe never changes); it imports
from the package. The legacy single-file implementation is archived verbatim
(`auto_operator_legacy_20260718.py`) and stays runnable until the differential
harness shows tick-for-tick equivalence on every scenario.

## 5. Execution order (summary; details in the refactor plan)

0. Safety inventory (workflow, 14 lenses) → `auto_operator_safety_contract.md`.
1. Archive legacy snapshot; build the differential replay harness; prove
   identity (legacy vs current = zero divergence) — the harness validates
   itself before it may validate a refactor.
2. Mechanical extraction #1: typed models (enums + dataclasses replacing the
   hold dict / _sec dict / track dict), logic untouched → harness green.
3. Extraction #2: safety predicates module (same call sites, same reasons).
4. Extraction #3: command arbitration (wire-ownership value + packet builder).
5. Extraction #4: per-arm state handlers out of `_held_update`.
6. Extraction #5: mission controller / perception / gaze split.
7. Docs: ADRs + incidents + traceability matrix; observability pass.
Each step: full test suite + differential replay before the next.
