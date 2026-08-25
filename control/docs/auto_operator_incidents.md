# Auto-Operator — Incident Register

Each entry: what happened on hardware, root cause, the mechanism that now
prevents it, and the forbidden "simplification". Refactors must keep every
mechanism listed here traceable (see the safety contract for SAFE-IDs).

## INC-1 · Cross-body joint/Cartesian override (2026-07-15)

Dual-arm mission: a held arm's JOINT retreat packet overrode the other arm's
Cartesian reach on the shared 14-joint wire → the reach starved, a phantom
hold latched at a far side/rear point, and the joint→Cartesian switch solved
from a stale front warm-start basin — the arm swept THROUGH the torso. User
powered off. Two facets:
(1) interleave — fixed by Guard-1 (`_held_update` GO/REACH freeze) + Guard-2
(`_tick_search` no-latch-while-track) + the one-track token;
(2) un-clamped held-arm follow across a sector boundary — fixed by the
`_sector_left` clamp on the GO/REACH follow.
Forbidden: merging the guards "because they look redundant"; publishing any
Cartesian packet while a staircase owns the wire; letting a held target slide
across a basin boundary un-staged.

## INC-2 · Stale-IK re-engagement snap (2026-07-14)

After a rear staircase retreat, stream silence let real_env re-engage IK on
the LAST Cartesian target (the old cube position) — the arm whipped back to
it. Mechanism: `_arm_engaged` (never go arm-silent after the first packet) +
real_env's rising-edge reseed patch. Forbidden: "cleaning up" the keep-alive,
or assuming silence retracts the arm (with --use-ik there is NO crawl-home —
07-15 contract audit).

## INC-3 · Gripper command loss / left-zero starvation (2026-07-16/17)

The ee_action channel is latest-value-wins at BOTH hops with no acks. One-shot
sends were eaten by real_env stalls; later, the yield-after-5 preemption
DROPPED the unsent remainder, starving the first of the back-to-back startup
zeros (left) to a 0.25 s burst that a mid-reconnect 9874 subscriber never saw
→ "left gripper never opened" (left-only / both / neither variants all
explained). Mechanisms: EE_ACTION_REPEAT_TICKS=30, queue ROTATION with
same-side supersession, min-burst 5 before yield, ee_alive liveness gating,
fresh-result-only policy (never hand_open on timeout/loss — that DROPPED a
gripped cube once), zero grace, latched-grasp_detected NOTE.
Forbidden: single-send "optimization"; opening on uncertainty; skipping the
zero ritual on a latched flag.

## INC-4 · Torso-tilt geometry corruption (2026-07-16, recurred 07-17, 07-18×2)

Robot hung/stood tilted → body-frame geometry chain corrupts end-to-end:
table cubes read z≈+0.2 (chest height) at r_xy≈0.5, the learned gate goes OOD
and waves them through, mink's use_yaw_frame IK diverges from body-frame
commands (EE parks 19-28 cm off; the 07-18 side-home glide could not track).
Mechanisms: TILT_REFUSE/WARN startup gate + continuous 5 s tilt watch
(projected_gravity), TARGET_MAX_RADIUS_M=0.45 envelope ceiling with the
loud TORSO-TILT refusal message, RULE #0 operational (hang straight).
Forbidden: widening the envelope; treating the refusal as noise.

## INC-5 · Gaze init slam & gimbal jitter (2026-07-15/17)

Zeros-init gaze slammed a parked gimbal multi-radian in one step → encoder
seeding gate + sender-side slew (robot applies the reference RAW since
8045371). Separately: the pre-reach camera shudder — CPU-torch gate scoring
(17-37 ms/call, 2-3×/tick in SEARCH/GO) blew the 50 ms tick budget; the
publish jitter excited the hot trim integrator (0.3/step) around a plant
with +100 ms low-pass lag. Mechanisms: `_gate_eval` throttle (10 Hz/target,
debounce steps on FRESH samples only), TRIM_GAIN=0.05.
Forbidden: un-throttling the gate; raising the trim gain; letting debounce
counters step on cached verdicts (one lucky sample would latch arrival).

## INC-6 · Catapult-class single-packet jumps (2026-07-16/18)

Handing the IK a full-distance target in one packet slews at the robot cap
(0.25 m/s) — knocked cubes at contact; at side-home scale (40+ cm) the arms
visibly "shot out". Mechanisms: STAGE_REACH_EE_RATE=0.08 gentle streams for
EVERY approach, re-extend, lift waypoint, carry, and (07-18) the home glide;
the unseeded-secondary ABANDON rule; mid-ramp timeout hands `restream` to the
hold. Forbidden: publishing raw far targets; seeding ramps at the goal.

## INC-7 · Home-glide freeze in the posture no-man's land (2026-07-18)

The side-home startup glide froze 8 cm out: mid-glide joints sit between the
power-on and side-home reference postures, `_arm_near_front` was False for
every home, and the idle gate stopped publishing (live probe: pos_target
stuck at [0.27, 0.29]). Mechanism: an ACTIVE `_home_stream` exempts the arm
from the posture gate (the glide itself is the safety mechanism — seeded from
measured EE, 8 cm/s, straight line); a stranded arm (no stream) is still held
in place. Forbidden: re-tightening the idle gate without the stream exemption.

## INC-8 · Via-home requirement (2026-07-18, user hard rule)

A cube whose detection jumps front→rear must NEVER be chased directly in
side-home mode (no sector staircases exist there to protect the path).
Mechanism: BEARING_JUMP_DEG=45 refusal at all three follow sites + the
`via_home` latch — retract to the side home, re-extend only after the
MEASURED EE is physically at home (±10 cm). Forbidden: shortcutting the
detour on a mid-retract sighting; comparing commanded instead of measured.

## INC-9 · Table-leg strikes by the transit sweep (2026-07-17)

The old side station swept the hand at tabletop height (EE z −0.183) and
clipped the cube stand. Mechanism: RAISED stations (front/rear z +0.18, side
abduction 55° → z +0.012), offline-validated chain (path z min −0.09,
clearance ≥28.9 mm), lift-then-clear before every carry.
Forbidden: reverting station joints without re-running the chain validation.

## INC-10 · Elbow-branch bifurcation on high direct reaches (2026-07-18)

With cubes at chest height (standing pelvis ≈0.8 m put the table cubes at
z≈+0.22), the direct side-home stream to a REAR target drove the arm into the
WRONG IK basin: shoulder pitch −111°, elbow above the shoulder behind the
back, 42–64 mm short of the target — while the correct mirror solution exists
and is stable (offline: 0 mm when seeded). Root cause: the local incremental
IK cannot cross the straight-arm singularity between the elbow-plus and
elbow-minus basins; the branch is chosen irreversibly in the first steps of
the stream (spontaneous symmetry breaking — low targets have elbow≈0 where
the basins coincide, which is why the direct flow was fine all week).
Mechanism: `--stage-sectors` (ADR-8) — high-front/rear legs stage through
their STAGE_JOINTS station; the settled station re-seeds the IK in the
human-chosen basin. Forbidden: direct Cartesian legs to z>0 front or ANY rear
target under side-home; disabling the staged holds' sector clamp; letting a
secondary take a staged-class leg.
