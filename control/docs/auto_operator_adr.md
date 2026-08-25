# Auto-Operator — Architecture Decision Records

Companion to `auto_operator_incidents.md` (the WHY behind each rule) and
`auto_operator_safety_contract.md` (the WHAT, with SAFE-IDs). Dates preserved.

## ADR-1 · Joint-space sector staging, never Cartesian waypoints (07-14)

Decision: front/side/rear transit uses BAKED JOINT POSTURES walked at a fixed
per-joint rate with encoder-confirmed arrival + dwell; the final cube hop
alone is Cartesian, and only from a settled station (the IK re-seeds from
encoders on the mode switch). Rejected: FK-matched Cartesian transit points —
7-DOF redundancy lets the online QP escape to a different configuration
branch even for pose-identical targets (measured: 163° away on hop 1; the
arbitrary-quat attempt audited at −33.7 mm wrist-through-hip). Now in
`auto_operator/staging.py`; postures + validation notes stay with the
STAGE_JOINTS constants in the operator.

## ADR-2 · Command-wire modal exclusivity (07-15)

Decision: the 9874 arm stream is treated as GLOBALLY MODAL (real_env keeps
one joint-vs-Cartesian flag for both arms; a joint packet carries all 14
joints). Therefore: one joint track at a time; no Cartesian packet while any
staircase is in flight; reaches and tracks mutually exclusive (Guard-1/2 +
the one-track token + the arbitrate() suppression). Now in
`auto_operator/arbitration.py`.

## ADR-3 · Visual servo = bounded bias estimator (07-13)

Decision: b̂ ← EMA of (camera-measured gripper base − FK-predicted), applied
as target−b̂; per-leg reset; same-camera-only; executed+quasi-static frame
gates; step ≤1 cm, ‖b̂‖ ≤8 cm; blocked-and-blind decay. Rejected: driving the
tag origin onto the standoff point (walks the hand 8.6 cm past the validated
contact pose) and any error-integrator form (latency wind-up). Now in
`auto_operator/servo.py`.

## ADR-4 · Side-home profile & the VIA-HOME rule (07-18)

Decision: with the side home ([0, ±0.55, 0], mink-validated) every leg is a
direct Cartesian reach and the staircase machinery is structurally
unreachable (sector_idx≡0); the ONE protected transition — a large bearing
jump — routes through the home with a MEASURED at-home gate (the home IS the
stage). Rejected: keeping three-sector staging under side-home (no benefit,
serialization cost), and direct big-jump chases (INC-8). Now spread across
`arms.py` (via-home), `arbitration.py` (glide), `mission.py` (bypasses).
PARTLY SUPERSEDED by ADR-8 (07-18): "no benefit" was wrong for HIGH targets
(elbow-branch bifurcation, INC-10) — `--stage-sectors` re-enables staging for
high-front/rear legs; via-home remains the rule for DIRECT holds only.

## ADR-5 · Arm state machine (as extracted, S5/S6)

Hold lifecycle branches of `arms.held_update`, in EXACT evaluation order per
arm per tick — this order is behavior, do not reorder:

```
dynamic_track off ─────────────────────────────► (frozen targets)
phase in {GO, REACH} ──► guard-1: sector/jump-clamped follow + grab trigger; STOP
track is not None ─────► staircase step (token-gated); carry-done → carried; STOP
grasp == carried ──────► drop-check + park ramp; STOP
grasp == lifting ──────► timeout/wire-pause/waypoint ramp; STOP
grasp == grasped ──────► drop-check + carry start (sector_idx/_sec-gated); STOP
grasp == sent ─────────► fresh-result policy (lift | retry | failed); STOP
maybe_send_grab fires ─► STOP
else ──────────────────► classic follow / via-home / retract / release
```

The dispatch chain lives in `arms.held_update` verbatim; a per-branch
function split remains open (needs its own dual-gated stage — the `continue`
edges are load-bearing).

## ADR-6 · Live constant resolution in extracted modules (S5/S6)

Extracted functions resolve module constants via
`sys.modules[type(op).__module__]` instead of importing the operator module:
(a) no circular import; (b) the test suite and the replay harness load the
operator under bespoke names via importlib — a plain import would create a
SECOND module instance and silently decouple monkeypatched constants
(`SECTOR_REACH_ENABLED` in tests) from the code under test. Rejected:
constants relocation (tests and hardware notes reference them on the
operator module) and parameter-threading (36 constants across 27 functions).

## ADR-7 · Strangler shims over big-bang typing (S2)

Typed dataclasses replaced the runtime dicts at construction sites only,
with dict-style access shims preserving all ~200 access sites textually.
Rejected: one-shot attribute migration (unreviewably large diff on a
safety-critical file; the shim lets each later key migration be its own
tiny, gated change).

## ADR-8 · Stage-sectors: stations as IK-basin selectors under side-home (07-18)

Problem: the robot-side mink IK is an incremental local tracker. A direct
Cartesian stream from the side home to a HIGH target (z above ~0) hits the
elbow-sign bifurcation: the front solution bends the elbow + while its
fore-aft mirror needs the elbow bent −, and once the first steps push the
elbow to one side the flow can never cross the straight-arm singularity to
the other basin. Hardware 07-18: a chest-height REAR cube solved as a −111°
backward shoulder sweep (elbow above the shoulder, behind the back), 42–64 mm
short — while the mirror solution exists and is stable (seeded offline: 0 mm).

Decision: re-enable the joint staging machinery under side-home behind
`--stage-sectors` (CLI default ON): the settled station re-seeds the IK (and
its PostureTask reference) in the HUMAN-CHOSEN basin — the station posture IS
the basin selector. Routing (user decision A): side and LOW front stay direct
Cartesian (validated flow; dual-parallel untouched); high front and ALL rear
stage via STAGE_JOINTS. Home station moves front(0)→side(1) via `op._home_idx`
(the 12 hard-coded `0`-as-home sites are parameterized); station 1 is re-baked
to SIDE_HOME_JOINTS in `_track_posture` (stage-sectors only — classic chains
untouched); staged holds carry their true sector_idx: track retreats/carries,
sector clamp restored, 8 cm clearance arc restored, via-home rule EXCLUDED
(one arbiter per hold); direct holds keep via-home. Secondaries remain pure
Cartesian: only DIRECT-classified legs, judged on the standoff point.
Rejected: always-stage front (loses dual-parallel simultaneity for the
validated low-cube demo); PostureTask retuning (robot-side file, treats the
symptom); Cartesian detour waypoints (ADR-1's 163° branch escape).
S-A offline audit 07-18: re-baked chains 28.9 mm trunk clearance (the static
shoulder-torso bound), limit margins ≥31°; rear real logged target 0.6 mm from
the station vs 42.4 mm direct; carry arc + inward re-plant clean.
