# Auto-Operator — Characterization & Regression Test Plan

Date: 2026-07-18. Two layers: (A) the existing behavioral suite (already 66
asserts — kept, extended), (B) the NEW differential replay harness. Together
they gate every refactor stage.

## A. Existing behavioral suite (`~/operator_tests/test_grasp_carry.py`)

Already-covered incident regressions (kept verbatim; IDs assigned in the
safety contract): grasp+carry side flow incl. lift waypoints; retry-then-fail
reopen policy; dual serialized (Guard-1/one-track); chain-dead degradation;
lost-result KEEP-hold policy; drop detection revert; gripper burst rotation +
budget + 2s-spread + `_ticks` leak guard; dual-parallel (parallel flight, both
slots live, first-arrived grabs, off-switch, side-primary gate, sec-first grab,
unseeded-timeout abandon incl. no-catapult packet check); in-flight retrack +
sector clamp; startup gentle glide (front and side-home) + no-mans-land
non-freeze; via-home routing (physically passes through home); tilt gate
(refuse 30°, pass 5°); gate throttle (1 net call, fresh-only debounce);
side-home basic + dual any-sector with zero joint ticks.

## B. Differential replay harness (`~/operator_tests/replay/`)

`replay_harness.py` runs LEGACY and CURRENT implementations in lockstep on
identical scripted inputs and diffs a per-tick record:

```
nav [vx,vy,wz] · arm_targets(mode: joint|cart|absent, values) · ee_action ·
gaze[4] · phase · per-arm {key, grasp, home, sector_idx, via_home} · sec(arm?)
```

Exactness: discrete fields exact; floats ≤1e-9 (mechanical refactors must be
bit-identical). First divergence is reported with tick index, field, both
values, and the scenario event log around it.

Self-validation (runs before any refactor comparison):
1. identity: legacy vs unmodified current → 0 divergences on all scenarios;
2. mutation smoke: a scratch copy with one perturbed constant MUST diverge.

### Scenario matrix (deterministic; events at fixed tick indices)

| # | name | profile | script |
|---|------|---------|--------|
| 1 | classic_front_grasp | classic, grasp | 1 front cube → reach→grab→lift+arc→carry→REST |
| 2 | classic_dual_serialized | classic | side cube then front cube; staircase + Guard-1 deferrals |
| 3 | classic_loss_return | classic | front hold → cube hidden 3 s → retract → return → re-extend |
| 4 | classic_park_mode | classic, hold_after_reach=False | front cube → PARK→DONE |
| 5 | sidehome_basic | side-home | startup glide → side cube direct reach → grab → lift → side return |
| 6 | sidehome_dual_any | side-home + dual | side + front cubes simultaneously |
| 7 | sidehome_via_home | side-home | front hold → cube jumps to rear → via-home detour |
| 8 | sidehome_glide_nml | side-home | joints forced to no-mans-land mid-glide |
| 9 | drop_detection | side-home | carried → stale cube reappears far → reopen/requeue |
| 10 | chain_dead | any | ee_alive False 15 s → classic degradation |
| 11 | idle_no_detections | both profiles | 400 ticks, glides only, keep-alive stability |
| 12 | tilt_refusal | any | projected_gravity 30° → SystemExit at the same tick |
| 13 | masked_telemetry | side-home + dual | right pos_actual masked → sec ABANDON path |
| 14 | gripper_burst_timing | any | zero pair rotation timeline (exact ee_action per tick) |

Determinism notes: the rig clock advances DT per tick; `_gate_eval` uses
`time.monotonic()` — scenarios run with gate=None (deterministic geometric
fallback) except one gate-throttle scenario that stubs a counting fake gate;
scan-phase gaze is deterministic given the tick count.

## C. New incident tests to ADD during stages (unit-level, from the spec §48)

The behavioral suite covers most end-to-end paths; these become cheap unit
tests once safety predicates/arbitration are extracted (S3-S4): NaN ingest
rejection; non-finite startup fail-closed; no-telemetry silence; envelope
min/max/z refusals incl. the exact refusal strings; midline both arms;
front→rear and rear→front staging passes through side (classic mode);
no-Cartesian-during-staircase; blip-silence during staircase; per-tick ramp
pause when unpublished; servo trust-rule battery (cross-camera, stale stamp,
moving arm/gimbal, step/bias clamps, decay); CLI defaults snapshot; packet
schema snapshot.

## D. Hardware rollout checklist (unchanged from the accepted plan)

offline suites → import/CLI smoke → gaze-only → no-walk standing → single-arm
front → side → rear (classic) → side-home reach → via-home jump test →
grasp-no-carry → lift+carry → loss/return → dual static holds → dual-parallel
→ full dual-cube mission. Safety rig + gamepad soft-stop at every step.
