# Auto-Operator — Staged Refactoring Plan

Date: 2026-07-18. Governing rule: **every step is mechanical, behavior-preserving,
and gated by (a) the full behavioral suite (`~/operator_tests/test_grasp_carry.py`,
66 asserts) and (b) the differential replay harness (legacy vs refactored,
tick-for-tick).** A step that cannot pass both is reverted, not patched forward.

## Gating infrastructure (Stage 0-1 — before any code moves)

- **S0. Safety inventory** — 14-lens workflow over the file → every mechanism
  gets a `SAFE-<AREA>-NNN` ID in `auto_operator_safety_contract.md` with
  line refs, constants, fail-closed flag, covering tests. The traceability
  matrix starts as this table plus a "new location" column that fills in as
  stages land.
- **S1. Legacy snapshot + differential harness**
  - `control/auto_operator_legacy_20260718.py` = verbatim copy (importable).
  - `~/operator_tests/replay/replay_harness.py`: drives BOTH implementations
    in lockstep through scripted deterministic scenarios (same rig semantics
    as the behavioral suite: perfect Cartesian/joint tracking, injectable
    grasp results, scriptable detections/telemetry incl. masks and tilt).
  - Per-tick comparison record: nav triple; arm_targets presence + schema
    (joint|cartesian) + values; ee_action; gaze vector; phase; per-arm hold
    summary (key, grasp state, home flag, sector_idx, via_home); `_sec`
    presence/arm. Tolerance: exact for discrete fields, ≤1e-9 for floats
    (pure refactors must be bit-identical; documented exception only if a
    numerically identical expression is re-ordered).
  - **Identity gate**: harness must first pass legacy-vs-current with the
    UNMODIFIED file (proves the harness detects nothing spuriously); a
    mutation smoke check (deliberately perturb one constant in a scratch
    copy) must FAIL (proves it detects real divergence).
  - Scenario set (each N ticks, events at fixed tick indices) — see
    `auto_operator_characterization_plan.md` for the full list.

## Extraction stages (each = one commit, suite + replay green)

- **S2. Typed models (`auto_operator/models.py`)** — enums `Phase` (moved),
  `GraspState`, `ArmSide`, `Sector`; dataclasses `HoldState` (replacing the
  15-key hold dict), `TrackMotion`, `SecondaryReach`, `GripperBurst`,
  `CamFrame`. Field-for-field identical semantics; dict-style access shims
  removed only after all readers are typed. Explicit invariant helpers
  (assert-only, used in tests): one cube per arm, one track on the wire,
  side-home ⇒ sector_idx==0, via_home ⇒ side_home, carried ⇒ held cube.
- **S3. Safety module (`safety.py`)** — pure predicates with reason strings:
  `target_safe`, `side_ok`, `sector_of/sector_left`, `bearing_jump`,
  `tilt_deg`, `near_home_posture`. Call sites unchanged in NUMBER and ORDER;
  each predicate documents its SAFE-IDs.
- **S4. Arbitration (`arbitration.py`)** — a `WireOwner` value computed once
  per tick (RET_TRACK | REACH_TRACK | CARTESIAN | SILENCE) + the packet
  builder (today's `_arm_packet` + the tick() arbitration block) with the
  priority order verbatim. The five exclusivity guards keep their sites but
  read the explicit owner.
- **S5. Arm state handlers (`arms.py`)** — `_held_update` split into
  per-state functions (guard1_follow, track_step, grasp_sent, lifting,
  grasped_carry, carried, classic_follow_release), each taking
  (HoldState, inputs) → (HoldState, intents, events). Order of evaluation and
  every `continue` edge preserved exactly (the state chart in
  `docs/auto_operator_adr.md` (ADR-5) is drawn FROM the old code first, then the
  split is checked against it).
- **S6. Mission/perception/gaze split** (`operator.py` façade + `perception.py`
  + `gaze.py` + `mission` handlers). `AutoOperator.__init__` signature,
  attribute names used by tests (`_held`, `_sec`, `phase`, `rest_ee`, …) are
  preserved via properties on the façade until the test suite migrates.
- **S7. Docs + observability** — ADRs/incidents (cross-body override,
  gripper command loss, tilt corruption, gaze slam, glide freeze, via-home),
  traceability matrix completion, event constants for the operationally
  load-bearing prints (text unchanged — operators grep these).

## Rollback

Every stage is a single commit on the operator refactor branch; rollback =
`git revert <stage>` (or reset to the pre-stage tag). The legacy snapshot
stays importable until S7 lands AND a full hardware validation pass (staged
checklist in the characterization plan §Hardware) succeeds; only then may it
move to `docs/archive/`.

## Explicit non-goals (restating the order)

No constant/threshold/timeout/rate changes; no packet or CLI changes; no
behavior redesign; no async; no new runtime dependencies; prints preserved
verbatim where operators rely on them (the diagnostics inventory marks each).
