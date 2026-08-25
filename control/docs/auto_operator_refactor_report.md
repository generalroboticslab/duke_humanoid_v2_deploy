# Auto-Operator Refactor — Final Report (stages S0-S7)

Date: 2026-07-18 · Branch: the operator refactor branch (internal)
Commits: 0897b6d (features) · 40c4e13 (S2+S3+docs) · 9a361bb (S4) ·
a9adc72 (S5) · df1bb97 (S6) · this doc lands with S7.

## A. Files changed

| file | role |
|---|---|
| `humanoid_auto_operator.py` | coordinator: constants, `AutoOperator.__init__`/`tick`, startup gates, delegate shells, CLI `main()` — **entry point unchanged** (2853 → 1244 lines) |
| `auto_operator/models.py` | S2 typed state (HoldState/TrackMotion/SecondaryReach/GripperBurst + dict shims) |
| `auto_operator/safety.py` | S3 pure predicates (target_safe/sector/bearing/tilt) |
| `auto_operator/arbitration.py` | S4 wire ownership: build_arm_packet, arbitrate, attach_ee_action, home_glide |
| `auto_operator/arms.py` | S5 hold/grasp/lift/carry machinery incl. held_update |
| `auto_operator/{gaze,perception,staging,servo,secondary,mission}.py` | S6 subsystem extractions |
| `auto_operator_legacy_20260718.py` | verbatim pre-refactor snapshot (rollback + replay reference) |
| `docs/auto_operator_*.md` | diagnosis, plan, characterization plan, incidents INC-1..9, ADR-1..7, 261-mechanism safety contract w/ traceability, this report |
| `~/operator_tests/replay/replay_harness.py` | differential replay (outside the repo, wipe-safe location) |

## B. Behavior preservation (precise claims)

- Every extraction is the VERBATIM legacy body (`self`→`op`); cross-calls go
  through the retained delegate shells, so the dynamic dispatch graph is
  unchanged. No constant, threshold, timeout, debounce count, packet schema,
  CLI flag/default, print string, or state-transition condition was altered.
- Two exceptions, both representation-only and gated: (1) S2 dict→dataclass
  at 4 construction sites behind access shims; (2) S5/S6 constant references
  now resolve live via `sys.modules[type(op).__module__]` (same objects, same
  values, monkeypatch-compatible — ADR-6).
- Verified equivalence: 13 replay scenarios (classic single/dual/loss/park,
  side-home basic/dual/via-home/no-mans-land, drop, chain-dead, idle, tilt
  refusal, masked telemetry; up to 5000 ticks each) — **zero divergence vs
  the legacy snapshot after every stage**, on a harness that itself passed an
  identity gate and a mutation-detection gate.

## C. Safety traceability

`docs/auto_operator_safety_contract.md`: 261 mechanisms, each with legacy
line refs, constants, fail-closed/timing flags, covering tests, incident
context, and a filled **New location (S6)** field. Incident knowledge is
duplicated durably in `auto_operator_incidents.md` (INC-1..9) and the design
rationale in `auto_operator_adr.md` (ADR-1..7).

## D. Test results

- Behavioral suite `~/operator_tests/test_grasp_carry.py`: **66/66 asserts
  PASS after every stage** (S2, S3, S4, S5, S6).
- Differential replay: **13/13 scenarios equivalent after every stage**;
  harness self-checks: identity PASS, mutation-detection PASS.
- One extraction defect was caught and fixed DURING S6 by the gates
  (a blanket `getattr(self→op` rewrite corrupted the models shim → NameError
  → both gates red → fixed → green), demonstrating the gates work.

## E. Remaining uncertainty (not verifiable offline)

- Real-robot timing (tick jitter with the module indirection — expected
  negligible: delegate shells are one extra call frame; gate throttle
  unchanged). Verify tick time on first hardware run.
- The learned ReachabilityGate path (replay runs gate=None deterministic;
  gate behavior itself untouched and throttle-tested separately).
- real_env / monitor / EE-service integration and the deploy MJCF — no
  changes were made to them, but the first hardware session should follow
  the staged checklist (characterization plan §D).

## F. Hardware rollout

Per `auto_operator_characterization_plan.md` §D: offline suites → CLI smoke →
gaze-only → no-walk standing → single-arm reaches → side-home + via-home →
grasp/carry → dual. Rollback: `git revert` any stage commit, or run the
snapshot directly (`auto_operator_legacy_20260718.py` is self-contained).

## Deferred (explicitly, each needs its own dual-gated stage)

- Per-branch split of `arms.held_update` (ADR-5 — `continue` edges are
  load-bearing); grasp/sector enum conversion (string comparisons today);
  attribute migration off the S2 dict shims; unit-test battery from the
  characterization plan §C; structured event objects for diagnostics.
