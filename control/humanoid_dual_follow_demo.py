"""Dual-arm sector-follow demo — stationed arms, live cube tracking, no grasp.

WHAT THIS IS. A standing demo: the two arms park at their own audited STAGE
stations (left FRONT, right REAR by default), the two cameras find and lock
one cube each, and whenever a cube enters an arm's sector-and-envelope the
robot-side mink IK is handed a live Cartesian stream so that hand FOLLOWS the
cube around. Take the cube away — or drag it out of reach, or into the other
arm's half of the world — and the arm walks its joint staircase back to its
station and waits there. Nothing is ever grasped and the base never steps.

    startup gates -> station staircase -> [HOME <-> FOLLOW] per arm, forever
    (a retreat is interruptible: see the cube again mid-walk-home and the arm
    fetches again from wherever it is — only the STARTUP staircase is sacred)

WHY A SEPARATE TOOL. The operator already owns every ingredient (perception,
stations, the reachability envelope, the HOLD-phase cube follow), but its
homes are SYMMETRIC: `_home_idx` is one shared station index threaded through
`_track_posture`, `_track_seq`, `staging.step_toward`, `arms.held_update` and
the arbitration packet builder — all of them on the hardware-validated journey
path. Making that asymmetric to get "left front, right rear" would edit the
mission's own spine for a demo. So the demo is its own linear loop and the
operator is untouched. (Same call the cuRobo reach tool made, for the same
reason.)

PROCESS WIRING (the operator's slot — 9874 is single-binder, so the operator
must be OFF): humanoid_real_env.py --use-ik, camera streaming, and
humanoid_monitor.py --bodies grasp_cube_60mm_e grasp_cube_60mm_f. This tool
binds 9874 (arm + gaze) and subscribes 9870 (telemetry) plus the monitor's
detection IPC socket. It NEVER binds 9873, so nav silence keeps the base
standing, and it never sends an `ee_action`, so the EE service is not needed
and no gripper can close.

--use-ik IS MANDATORY. Cartesian slots are the whole demo, and real_env drops
them with a one-line warning when use_ik=False ("Cartesian targets received
but use_ik=False — ignored"): the arms would sit at their stations forever
while this tool cheerfully reported FOLLOW.

START THIS TOOL *BEFORE* real_env, AND HERE IS WHY (hardware 2026-08-10).
--use-ik plus nobody on 9874 is not a neutral idle state. real_env computes
`arm_ik_active = use_ik and (not _arm_stream_active or _arm_cart[...])`, so with
no arm stream the IK owns BOTH arms and its output overwrites everything —
including the 0.125 rad/s arm-silence failsafe ramp, which therefore never runs
under --use-ik. reset() snaps the EE targets to the power-on chest tuck, and
that posture violates mink's own CollisionAvoidanceLimit (a hard 50 mm, against
shoulder_3~wrist_1 at 16.5 mm and wrist_1~wrist_3 at 10.2 mm), so the QP walks
the arm OUT of the fold while chasing the EE pose. Measured 07-31 with the
deployment's own solver: 86.6 deg from CHEST_HOME_JOINTS with wrist_3 pinned on
its limit, from every recognised start. That is the "arms go home and then move
somewhere weird" every power-up. One joint slot ends it, so this tool claims the
arms with a frozen encoder snapshot on the FIRST telemetry packet it sees —
TEL_WAIT_S is long precisely so it can be sitting there waiting when real_env
comes up. Started second, it inherits whatever mink already did.

THE STARTUP CORRIDOR IS CERTIFIED AT RUNTIME, NOT WHITELISTED. Because of the
above, the arms routinely start at a posture no audit covers, and a whitelist
gate ("park it at a known home first") just refuses the demo every time. So the
whole startup schedule — both arms, the receiver's own per-joint equal-rate
curve, dwells included — is swept with `preflight`, joint_monkey's auditor and
the same one the cuRobo Gate C uses, and refused only if it comes within
CERT_FLOOR_M of anything. Station-to-station hops are still the audited 07-17
chain; the certificate is what covers the first hop out of an arbitrary pose.

SAFETY CONTRACT (inherited, not reinvented):
  * the operator's startup gates rerun: fresh telemetry, torso tilt under
    TILT_REFUSE_DEG, both arms parked at a KNOWN posture;
  * EVERY inter-station move is a JOINT stream along the audited TRACK, one
    station at a time, with per-hop arrival (STAGE_TOL_RAD) and a
    STAGE_DWELL_S dwell. Never a Cartesian home command: mink's
    CollisionAvoidanceLimit refuses to steer into folds (the 07-31 chest-home
    measurement landed 86.6 deg off, 2000 IK steps, from every recognised
    start) and the 07-21 journey glide landed 62 deg off its posture;
  * an arm only ever follows a cube in ITS OWN sector. A rear-parked arm
    chasing a front cube is the 07-15 cross-body class (the arm swept
    front->rear THROUGH the torso) and is exactly what the stations exist to
    prevent;
  * the Cartesian stream is RAMPED from the measured hand position at
    STAGE_REACH_EE_RATE — never a raw goal jump (07-18: a raw EE goal handed
    to the IK "shot the arms out to the sides");
  * the startup corridor out of the measured posture carries a runtime
    clearance certificate (above); the station-to-station hops carry the 07-17
    audit.

WHAT RELEASING THE ARMS ACTUALLY DOES, under --use-ik: NOT the 0.125 rad/s
crawl home. Ctrl+C (or any abort — every one of them simply stops publishing)
makes the stream stale, `arm_ik_active` flips True, real_env re-seeds the IK
from encoders to hold the current pose, and mink then pushes out of whatever
fold it is in. The stations themselves clear only ~28.9 mm at the static
shoulder-torso pair (07-17), under mink's 50 mm bar, so expect a parked arm to
drift once this tool lets go. It is slow (0.5 rad/s VelocityLimit) and
collision-free by mink's own constraint, but it is motion, not a stop. To stop
the arm dead you need real_env down or the motors disabled.

GAZE AND PERCEPTION ARE THE OPERATOR'S OWN, imported not re-implemented:
GazeRig (humanoid_curobo_reach) already duck-types AutoOperator for
auto_operator.gaze + auto_operator.perception and runs their bodies verbatim —
the panoramic serpentine scan, FIRST-SEER camera locking (the eye that
actually decoded a cube claims it; the other keeps sweeping), per-cube claim
expiry, NaN drops, and the best-evidence median. "Each camera tracks its own
cube independently" is that machinery, unmodified. Importing it pulls in the
cuRobo plan CLIENT as a side effect; nothing here ever dials a plan server.

THE FACE BAR IS 1 BY DEFAULT (user 2026-08-11: "1 face is ok to track").
The first hardware run spent most of its time at 1 face — a hand-held cube
hides faces by construction — and the frozen-goal rhythm (follow, stick,
follow, stick) read as broken rather than honest. A 1-face fit does drift
tens of mm (44-51 mm measured 08-10), but this demo grasps nothing and the
envelope + sector fences stay on, so the drift costs jitter, not safety.
--follow-min-faces 2 restores the strict bar (SERVO_MIN_INLIERS, the gripper
servo's own rule: "single-tag PnP poses are garbage-prone") for a run where
smoothness matters more than responsiveness. Whatever the bar, sub-bar
evidence FREEZES the goal rather than moving the hand, and does NOT count as
losing the cube: the lost-grace below runs off `last_seen`, so a low-evidence
stream keeps the arm extended and still.

KNOWN LIMITS (deliberate): no grasp, no walking, no visual-servo bias
estimator (there is no grasp to bias-correct), no obstacle model — the arm
follows the cube through free space and the ONLY thing keeping it out of the
torso is the sector rule plus the reach envelope.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time

import mujoco
import numpy as np
import tyro

import humanoid_auto_operator as OP
from auto_operator import safety as ao_safety
from humanoid_curobo_client import ARM_JOINTS_ALL, ARM_JOINTS_L, ARM_JOINTS_R
from humanoid_curobo_reach import (
    DetectionView,
    GazeRig,
    TelemetryView,
    _arm_enc,
    _measured_ee,
    _tilt_deg,
    posture_gate_homes,
)
from humanoid_joint_monkey_hw import preflight
from humanoid_model import MJCF_MODEL_PATH

NNGPublisher = OP.NNGPublisher

PUB_HZ = 20.0
PUB_DT = 1.0 / PUB_HZ

ARMS = ("left", "right")
ARM_JOINTS = {"left": ARM_JOINTS_L, "right": ARM_JOINTS_R}

# A station's NAME is also the sector it serves: STAGE_JOINTS' keys and
# safety.sector_of's return values are the same three words by construction
# (STATION_BEARING_DEG puts each station's own EE bearing inside its sector
# band — 28.8 / 90.1 / 140.5 deg against the 60/120 deg boundaries). That is
# what makes "an arm serves the sector it is parked in" a geometric statement
# rather than a naming coincidence.
STATIONS = OP.TRACK                        # ("front", "side", "rear")

# --- follow behaviour: every value is the operator's own, reused ---------------
FOLLOW_MIN_FACES = 1                       # user 2026-08-11 after the first
                                           # hardware run: "1 face is ok to
                                           # track" — see the module docstring.
                                           # --follow-min-faces 2 restores the
                                           # gripper servo's strict bar
                                           # (SERVO_MIN_INLIERS).
FOLLOW_LOST_S = 2.0                        # 08-11 (user): "when it loses it,
                                           # hold for 2 seconds and then
                                           # retreat". Was DYN_LOST_GRACE_S
                                           # (2.5). Short occlusions still
                                           # keep the arm extended and still;
                                           # and since the retreat is now
                                           # RE-ACQUIRABLE (see the HOMING
                                           # branch), giving up 0.5 s earlier
                                           # costs almost nothing — a cube
                                           # that reappears mid-retreat is
                                           # fetched again from wherever the
                                           # arm is.
FOLLOW_IN_TICKS = OP.GATE_CONSECUTIVE      # 5 consecutive good samples to ENTER
FOLLOW_OUT_TICKS = OP.GATE_CONSECUTIVE     # 5 consecutive bad ones to LEAVE.
                                           # Debounced in BOTH directions so a
                                           # cube held at the envelope edge
                                           # cannot flap the arm in and out.
FOLLOW_DEADBAND_M = OP.DYN_RETRACK_M       # 0.01 — re-aim only once the cube
                                           # has actually moved this far, so
                                           # detection jitter cannot wiggle a
                                           # parked hand.
FOLLOW_EE_RATE = 0.25                      # m/s ramp on the PUBLISHED point.
                                           # 08-11 (user: "giant latency"):
                                           # 0.12 -> 0.25, which is exactly
                                           # mink's own internal target slew
                                           # (max_target_pos_step 0.005 m at
                                           # 50 Hz) — publishing faster than
                                           # this is clipped inside the IK, so
                                           # 0.25 is the ceiling of useful.
                                           # Smoothness is preserved by mink's
                                           # trajectory filter and velocity
                                           # limits; the launch stays gentle
                                           # because the ORIENTATION ramp
                                           # (below) paces it. Never a whole
                                           # station->cube jump in one packet
                                           # (the 07-18 lesson).
FOLLOW_ORI_RATE_RAD_S = 0.5                # 08-11 (user: "fetch out like a
                                           # rocket") — the launch whip was
                                           # the ORIENTATION, not the
                                           # position: the published quat used
                                           # to jump station-pose -> neutral
                                           # in ONE packet, and mink's
                                           # internal ori slew allows 2.5
                                           # rad/s, so the wrist+elbow whipped
                                           # to reorient in ~0.3 s while the
                                           # position crawled at 0.12 m/s.
                                           # Now the published quat SLERPs
                                           # from the measured hand
                                           # orientation to neutral at this
                                           # rate (~29 deg/s, mink's own
                                           # per-joint VelocityLimit), so the
                                           # reorientation finishes on the
                                           # same clock as the reach glide.
                                           # Tracking sensitivity untouched —
                                           # this ramps only the launch.
FOLLOW_MAX_RADIUS_M = OP.DYN_OUTREACH_M    # 0.60, deliberately equal to
                                           # TARGET_MAX_RADIUS_M: the radius at
                                           # which we START following must be
                                           # the radius the arm is allowed to
                                           # work in, or the cube has to be
                                           # dragged closer than the arm can
                                           # actually reach (the 07-24 trigger/
                                           # capability mismatch).

# Per-hop crawl budget, the same shape staging.step_toward uses: the joint
# travel at STAGE_RATE, 40% slack, plus 5 s of fixed overhead for the receiver
# ramp and the dwell. Overrunning it does not abort — there is nowhere safer to
# send an arm than the audited posture it is already streaming toward — it just
# says so once, loudly, because a hop that never arrives is a wedge or a fault.
HOP_BUDGET_SLACK = 1.4
HOP_BUDGET_FIXED_S = 5.0

HOME_TIMEOUT_S = 90.0      # startup staircase budget. Sized for the worst
                           # recognised start (power-on -> side -> rear at
                           # STAGE_RATE, two hops with dwells) with room for a
                           # failsafe crawl still finishing underneath.
TEL_WAIT_S = 180.0         # 2026-08-10: 10 -> 180 so this tool can be started
                           # BEFORE real_env and be waiting when its first
                           # packet lands. That ordering is the whole fix for
                           # the mink drift described in the module docstring:
                           # the arms are claimed within one tick of existing,
                           # so the IK never gets a window to walk them out of
                           # the power-on fold.
STATUS_PERIOD_S = 2.0

# --- runtime corridor certification -------------------------------------------
CERT_FLOOR_M = 0.025       # the same 25 mm floor every cuRobo mission gates its
                           # routes at (--clearance-floor-mm 25). The startup
                           # glide is the ONE motion in this demo that starts
                           # from an arbitrary measured posture, so it is not
                           # covered by the 07-17 station audit and must earn
                           # its own certificate against our MuJoCo model.
CERT_MAX_FRAMES = 4000     # runaway guard on the simulated staircase (~200 s at
                           # PUB_DT); a real schedule is 300-500 frames.


def station_posture(arm: str, idx: int) -> np.ndarray:
    """Joint posture of TRACK station `idx` for `arm`.

    mirror_arm, never a bare `-q`: the deploy model's mirror convention is
    ARM_MIRROR_SIGN and a wrong sign does not fail loudly — the negated target
    stays inside every limit, the stream reports "settled", and the gripper
    simply sits up to 124 deg rotated from the audited posture.
    """
    return OP.mirror_arm(OP.STAGE_JOINTS[STATIONS[idx]], arm)


def startup_seq(home_idx: int) -> list[int]:
    """Stations to visit walking from POWER-ON to `home_idx`, in order.

    Power-on is not a TRACK station, so this cannot come from _track_seq — the
    question is which station the arm may enter the track AT. Answered by the
    07-31 clearance audit (JOURNEY_POSTURE_TOL_RAD): "Same floors measured for
    front<->POWERON ... side<->POWERON interpolation min clearance 39.5 mm",
    so power-on may glide straight to FRONT or to SIDE. It says nothing about
    power-on->REAR, and that interpolation is a 1.2 rad shoulder sweep through
    unaudited space — so a rear-homed arm enters at SIDE and takes the audited
    side->rear hop (>= 28.9 mm, the 07-17 raised-chain re-validation) from
    there. Two audited hops instead of one unaudited one.
    """
    return [home_idx] if home_idx <= STATIONS.index("side") else [1, 2]


def joint_slot(arm: str, posture) -> dict:
    return {"joint_pos": [float(v) for v in posture],
            "rate": float(OP.STAGE_RATE)}


def cart_slot(pos, quat=None) -> dict:
    """Cartesian slot for one side. NEUTRAL_QUAT is what the runtime IK is
    given everywhere else in this codebase, and TARGET_MAX_RADIUS_M was
    measured against it (the clean reach, EE within 5 deg of neutral).
    `quat` overrides it for the ORIENTATION RAMP below — the ramp's whole job
    is to end at NEUTRAL_QUAT."""
    return {"ee_pos": [float(v) for v in pos],
            "ee_quat": [float(v) for v in
                        (OP.NEUTRAL_QUAT if quat is None else quat)]}


def _measured_ee_quat(tel_data: dict, arm: str) -> np.ndarray | None:
    """Measured EE orientation (base frame, wxyz) from telemetry, or None."""
    q = ((tel_data.get("ee") or {}).get(arm) or {}).get("quat_actual")
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) \
            or np.linalg.norm(q) < 1e-6:
        return None
    return q / np.linalg.norm(q)


def slerp_toward(q_from: np.ndarray, q_to, max_step_rad: float) -> np.ndarray:
    """`q_from` rotated toward `q_to` by at most `max_step_rad` (wxyz).

    Shortest arc (hemisphere-aligned), exact arrival when the remaining angle
    is inside the step — the orientation analog of the position ramp above."""
    a = np.asarray(q_from, dtype=float)
    b = np.asarray(q_to, dtype=float)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = 2.0 * math.acos(min(1.0, abs(dot)))
    if angle <= max_step_rad or angle < 1e-9:
        return b.copy()
    t = max_step_rad / angle
    # standard slerp between unit quaternions
    theta = math.acos(dot)
    s = math.sin(theta)
    out = (math.sin((1.0 - t) * theta) / s) * a + (math.sin(t * theta) / s) * b
    return out / np.linalg.norm(out)


def in_envelope(pos) -> bool:
    """The coarse safety envelope every point the arms may be sent to must
    pass: outside the trunk keep-out cylinder, inside the reach radius, inside
    the z band. The IK has no self-collision awareness, so a cube against the
    torso or at face height would otherwise become a literal IK target."""
    return ao_safety.target_safe(pos, OP.TARGET_MIN_RADIUS_M,
                                 FOLLOW_MAX_RADIUS_M,
                                 OP.TARGET_Z_MIN_M, OP.TARGET_Z_MAX_M)


def sector_of(pos) -> str:
    return ao_safety.sector_of(pos, OP.SECTOR_FRONT_DEG, OP.SECTOR_REAR_DEG)


def sector_holds(pos, sector: str) -> bool:
    """Whether a cube being FOLLOWED still belongs to `sector`, with the
    hysteresis margin — a cube wobbling on a bearing boundary must not flap a
    following arm between "mine" and "not mine"."""
    return not ao_safety.sector_left(pos, sector, OP.SECTOR_FRONT_DEG,
                                     OP.SECTOR_REAR_DEG, OP.SECTOR_HYST_DEG)


class _Frames:
    """Duck-typed stand-in for joint_monkey's Trajectory: preflight reads only
    `.frames`. Same shim the cuRobo client uses for its Gate C sweep."""

    def __init__(self, frames):
        self.frames = frames


def staircase_frames(enc: dict, seqs: dict[str, list[int]]) -> list[np.ndarray]:
    """Every 14-joint configuration the arms will ACTUALLY pass through on the
    way to their stations, in ARM_JOINTS_ALL order.

    Reproduces the RECEIVER's own motion law rather than a straight line: each
    joint is stepped independently toward the hop target by STAGE_RATE*dt and
    clamped, exactly as real_env._update_arm_targets does, so joints with the
    smaller deltas arrive first. That per-joint EQUAL-RATE curve is the path the
    07-17 station audit was measured on; a straight-line joint interpolation is
    a different curve through space and would certify a corridor the robot never
    takes. Both arms are simulated together, so the sweep also sees arm-vs-arm
    clearance during simultaneous motion.
    """
    q = {n: float(enc[n]) for n in ARM_JOINTS_ALL}
    todo = {a: list(seqs[a]) for a in ARMS}
    dwell = {a: 0 for a in ARMS}
    dwell_ticks = max(1, int(round(OP.STAGE_DWELL_S / PUB_DT)))
    step = OP.STAGE_RATE * PUB_DT
    frames = [np.array([q[n] for n in ARM_JOINTS_ALL])]
    for _ in range(CERT_MAX_FRAMES):
        if not any(todo[a] for a in ARMS):
            break
        for a in ARMS:
            if not todo[a]:
                continue
            if dwell[a] > 0:
                dwell[a] -= 1
                if dwell[a] == 0:
                    todo[a].pop(0)
                continue
            target = station_posture(a, todo[a][0])
            worst = 0.0
            for i, n in enumerate(ARM_JOINTS[a]):
                q[n] += float(np.clip(float(target[i]) - q[n], -step, step))
                worst = max(worst, abs(float(target[i]) - q[n]))
            if worst < OP.STAGE_TOL_RAD:
                dwell[a] = dwell_ticks
        frames.append(np.array([q[n] for n in ARM_JOINTS_ALL]))
    return frames


def certify(model, frames) -> tuple[float, str]:
    """(min clearance [m], the geom pair that bound it) over `frames`.

    preflight is joint_monkey's own auditor — the same function the cuRobo
    client's Gate C calls — so a corridor certified here is certified by the
    identical measure a planned route has to pass.
    """
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS_ALL]
    return preflight(model, data, _Frames(frames), qadr)


def validate_homes(homes: dict[str, str]) -> str | None:
    """Refusal for an unusable --left-home/--right-home pair, else None.

    Sharing a station is not a preference but a contradiction: a station's name
    IS the sector it serves, so two arms on one station would both latch the
    same cube and reach for it simultaneously.
    """
    for arm, name in homes.items():
        if name not in STATIONS:
            return (f"--{arm}-home {name!r} is not a station "
                    f"({'/'.join(STATIONS)})")
    if homes["left"] == homes["right"]:
        return (f"both arms are homed at {homes['left'].upper()}, so they "
                f"would serve the SAME sector and contend for the same cube — "
                f"give each arm its own station")
    return None


def corridor_verdict(clear_m: float, where: str) -> str | None:
    """Refusal for a startup corridor that does not clear CERT_FLOOR_M, else
    None. Split out from main so the decision is testable without a robot."""
    if clear_m >= CERT_FLOOR_M:
        return None
    return (f"the startup glide to the stations passes within "
            f"{clear_m * 1000:.1f} mm at {where} — under the "
            f"{CERT_FLOOR_M * 1000:.0f} mm floor. The arms are somewhere this "
            f"corridor cannot leave safely; restart real_env so its prepare() "
            f"ramp returns them to the power-on pose, then start THIS TOOL "
            f"FIRST and real_env second.")


def accepted_homes() -> dict[str, list]:
    """Postures the startup gate accepts an arm parked at.

    The reach tool's list plus the SIDE and REAR stations. Those two are absent
    there for a good reason — nothing in the cuRobo path can leave an arm at
    them — but this demo parks an arm at REAR by design, so its own second run
    would be refused by an unextended list.
    """
    homes = dict(posture_gate_homes())
    homes["side"] = OP.STAGE_JOINTS["side"]
    homes["rear"] = OP.STAGE_JOINTS["rear"]
    return homes


def nearest_home(enc: dict, arm: str) -> tuple[float, str]:
    """(distance [rad], name) of the closest known posture for `arm`.

    INFORMATIONAL, not a veto. The reach tool refuses an unrecognised posture
    because "reaching from an unknown posture crosses configuration basins" —
    a sound rule when the corridor cannot be checked. Here it CAN be: `certify`
    sweeps the exact curve the receiver will drive, from the exact posture the
    arm is in, against our own model. That answers the same question with a
    measurement instead of a whitelist, so this is reported and the certificate
    decides. It matters because with --use-ik the arms routinely start OUTSIDE
    every known posture — see the module docstring.
    """
    homes = accepted_homes()
    q = np.array([enc[n] for n in ARM_JOINTS[arm]])
    return min((float(np.max(np.abs(q - OP.mirror_arm(h, arm)))), n)
               for n, h in homes.items())


def startup_gates(tel: dict) -> str | None:
    """Refusals that no certificate can overrule. Tilt only: a tilted torso
    corrupts every base-frame number the demo reads (RULE #0)."""
    tilt = _tilt_deg(tel)
    if tilt is None:
        return "no usable projected_gravity in telemetry"
    if tilt > OP.TILT_REFUSE_DEG:
        return (f"torso {tilt:.1f} deg off vertical (> {OP.TILT_REFUSE_DEG}) — "
                f"RULE #0: legs straight, torso upright")
    if tilt > OP.TILT_WARN_DEG:
        print(f"[gate] WARNING: torso {tilt:.1f} deg off vertical — margins "
              f"are thin")
    return None


HOMING, HOME, FOLLOW = "HOMING", "HOME", "FOLLOW"


@dataclasses.dataclass
class Arm:
    """One arm's whole state. Two of these and a 20 Hz loop is the demo.

    The three states are exhaustive and every transition is bounded: HOMING
    ends when the staircase settles, HOME ends when a cube passes the gate
    FOLLOW_IN_TICKS times, FOLLOW ends when the cube goes unseen or fails the
    gate FOLLOW_OUT_TICKS times. There is no state whose only exit is a human.
    """
    arm: str
    home: str                                   # station name == served sector
    state: str = HOMING
    seq: list[int] = dataclasses.field(default_factory=list)
    dwell_until: float | None = None
    hop_t0: float = 0.0
    hop_budget: float | None = None
    hop_warned: bool = False
    key: str | None = None                      # latched cube while FOLLOWing
    goal: np.ndarray | None = None              # live aim point
    stream: np.ndarray | None = None            # the RAMPED published point
    quat_stream: np.ndarray | None = None       # the RAMPED published quat —
                                                # measured hand orientation at
                                                # entry, slerped to neutral
    reject: np.ndarray | None = None            # last fix that FAILED the gate
    in_hits: int = 0
    out_hits: int = 0
    retreating: bool = False                    # HOMING because a FOLLOW ended
                                                # — such a retreat may be
                                                # interrupted by re-acquiring
                                                # the cube (08-11); the STARTUP
                                                # staircase may not (its
                                                # corridor was certified as a
                                                # whole and mid-staircase
                                                # postures cross basins)

    @property
    def home_idx(self) -> int:
        return STATIONS.index(self.home)

    def start_homing(self, seq: list[int], now: float) -> None:
        self.state = HOMING
        self.seq = list(seq)
        self.dwell_until = None
        self.hop_t0 = now
        self.hop_budget = None
        self.hop_warned = False
        self.key = None
        self.goal = None
        self.stream = None
        self.quat_stream = None
        self.reject = None
        self.in_hits = 0
        self.out_hits = 0
        self.retreating = False

    def begin_follow(self, telemetry: dict, key: str,
                     med: np.ndarray) -> bool:
        """Enter FOLLOW on `key`, seeding both ramps from the MEASURED hand.
        False (state unchanged) when telemetry has no hand pose to ramp from.

        Shared by the HOME entry and the mid-retreat re-acquire so the two
        launches are the same launch: position glides from the real hand at
        FOLLOW_EE_RATE, orientation slerps from the real wrist at
        FOLLOW_ORI_RATE_RAD_S — never a jump, wherever the arm was."""
        hand = _measured_ee(telemetry, self.arm)
        if hand is None:
            return False
        self.state = FOLLOW
        self.key = key
        self.goal = np.asarray(med, float).copy()
        self.stream = hand.copy()
        self.quat_stream = _measured_ee_quat(telemetry, self.arm)
        self.out_hits = 0
        self.retreating = False
        return True


def step_home(a: Arm, enc: dict, now: float, moves: bool = True) -> tuple[dict, bool]:
    """One tick of the joint staircase. Returns (slot, arrived).

    Deliberately the same shape as staging.step_toward — command one station's
    posture, require encoder arrival THEN a dwell before advancing, so every
    segment starts from exactly the configuration the chain was audited at —
    minus its timeout-flips-to-home branch, which is meaningless here: home IS
    the target, so a hop that overruns has nowhere better to go and keeps
    streaming the audited posture while saying so.

    `moves` False is the dry run: nothing is published, so the arm can never
    arrive and the encoder test would park the whole demo on hop 0 forever.
    Arrival is then assumed (the dwell still runs, so the rehearsal keeps the
    real cadence) and everything downstream — sector pick, follow entry, the
    lost/out-of-reach exits — is exercised against LIVE perception.
    """
    if not a.seq:
        return joint_slot(a.arm, station_posture(a.arm, a.home_idx)), True
    hop = a.seq[0]
    posture = station_posture(a.arm, hop)
    slot = joint_slot(a.arm, posture)
    q = np.array([enc[n] for n in ARM_JOINTS[a.arm]])
    if a.hop_budget is None:
        a.hop_budget = (float(np.max(np.abs(q - posture))) / OP.STAGE_RATE
                        * HOP_BUDGET_SLACK + HOP_BUDGET_FIXED_S)
        a.hop_t0 = now
    if a.dwell_until is not None:
        if now < a.dwell_until:
            return slot, False
        a.dwell_until = None
        a.seq.pop(0)
        a.hop_budget = None
        a.hop_warned = False
        print(f"[{a.arm}] settled at the {STATIONS[hop].upper()} station")
        return slot, not a.seq
    err = float(np.max(np.abs(q - posture)))
    if err < OP.STAGE_TOL_RAD or not moves:
        a.dwell_until = now + OP.STAGE_DWELL_S
    elif not a.hop_warned and now - a.hop_t0 > a.hop_budget:
        a.hop_warned = True
        print(f"[{a.arm}] WARNING: the {STATIONS[hop].upper()} hop has not "
              f"arrived in {a.hop_budget:.0f}s (worst joint "
              f"{math.degrees(err):.0f} deg out) — still streaming the audited "
              f"posture; check real_env for arm faults or OVER CURRENT")
    return slot, False


def pick_cube(a: Arm, rig: GazeRig, keys, now: float,
              min_faces: int) -> tuple[str, np.ndarray] | None:
    """The nearest well-evidenced cube inside this arm's sector and envelope.

    Two arms can never contend for one cube: sectors partition the bearing
    circle, so a cube belongs to exactly one of them. Nearest-first only
    matters when the demo runs with more cubes than arms.
    """
    best = None
    for key in keys:
        med = rig.median(key, now)
        if med is None or rig.faces_used(key, now) < min_faces:
            continue
        if not in_envelope(med) or sector_of(med) != a.home:
            continue
        r = float(np.hypot(med[0], med[1]))
        if best is None or r < best[0]:
            best = (r, key, med)
    return None if best is None else (best[1], best[2])


def follow_slot(a: Arm, rig: GazeRig, now: float, dt: float,
                min_faces: int) -> tuple[dict | None, str | None]:
    """One FOLLOW tick. Returns (slot, leave_reason).

    The goal moves only on >= min_faces evidence; the PUBLISHED point then
    ramps toward it at FOLLOW_EE_RATE. Freezing the goal is not the same as
    losing the cube — the lost test below runs off `last_seen`, so a
    hand-occluded cube keeps the arm extended and still instead of retracting
    from a cube that is right there.

    THE GOAL READS THE NEWEST FIX, NOT THE MEDIAN (08-11 user: "there's
    always a giant latency"). rig.median averages CUBE_LATCH_WINDOW_S=0.7 s
    of samples, which for a MOVING cube is a built-in ~0.35 s lag — the right
    trade for aiming a grasp at a stationary target, the wrong one for
    chasing a hand-held cube. The newest packet's jitter is absorbed
    downstream by the 1 cm deadband and the FOLLOW_EE_RATE ramp. pick_cube
    (the ENTRY decision) deliberately stays on the median: launching the arm
    is a considered act, tracking is a live one.
    """
    cube = rig.cubes[a.key]
    med = rig.latest(a.key, now)
    if med is not None and rig.last_inliers(a.key) >= min_faces:
        if in_envelope(med) and sector_holds(med, a.home):
            a.out_hits = 0
            a.reject = None
            if a.goal is None or float(np.linalg.norm(med - a.goal)) \
                    > FOLLOW_DEADBAND_M:
                a.goal = np.asarray(med, float).copy()
        else:
            a.out_hits += 1
            # The REJECTED fix, not the frozen goal. `goal` deliberately stops
            # updating the moment the gate fails, so reporting against it would
            # describe where the cube USED to be — the verdict would name the
            # arm's own sector as the reason the arm may not serve it.
            a.reject = np.asarray(med, float).copy()
    unseen = now - cube.last_seen
    if unseen > FOLLOW_LOST_S:
        return None, f"{a.key} unseen for {unseen:.1f}s"
    if a.out_hits >= FOLLOW_OUT_TICKS:
        bad = a.reject if a.reject is not None else a.goal
        # Name the limit that actually FIRED (08-11 hardware: a cube lifted
        # over the z ceiling was reported as "0.38 m out, past the reach
        # envelope" — a radius well inside the band, sending the operator to
        # doubt the wrong number).
        r = float(np.hypot(bad[0], bad[1]))
        if in_envelope(bad):
            where = f"in the {sector_of(bad).upper()} sector"
        elif r > FOLLOW_MAX_RADIUS_M:
            where = f"{r:.2f} m out, past the reach envelope"
        elif r < OP.TARGET_MIN_RADIUS_M:
            where = f"{r:.2f} m out, inside the trunk keep-out"
        else:
            where = (f"at z={float(bad[2]):+.2f} m, outside the "
                     f"{OP.TARGET_Z_MIN_M:+.2f}..{OP.TARGET_Z_MAX_M:+.2f} band")
        return None, (f"{a.key} is {where}, not servable from the "
                      f"{a.home.upper()} station")
    # Ramp the published point; never hand the IK the whole jump at once.
    delta = a.goal - a.stream
    n = float(np.linalg.norm(delta))
    step = FOLLOW_EE_RATE * dt
    a.stream = a.goal.copy() if n <= step else a.stream + delta * (step / n)
    # Ramp the published ORIENTATION the same way (08-11, the rocket launch):
    # from the measured hand quat at entry toward neutral, never in one jump.
    if a.quat_stream is None:
        a.quat_stream = np.asarray(OP.NEUTRAL_QUAT, dtype=float)
    a.quat_stream = slerp_toward(a.quat_stream, OP.NEUTRAL_QUAT,
                                 FOLLOW_ORI_RATE_RAD_S * dt)
    return cart_slot(a.stream, a.quat_stream), None


@dataclasses.dataclass
class Args:
    robot_ip: str = "127.0.0.1"
    cubes: tuple[str, ...] = ("grasp_cube_60mm_e", "grasp_cube_60mm_f")
    """Cube bodies to track. Must also be in the monitor's --bodies, or they
    are never detected and both arms simply wait at their stations."""
    detection_url: str = OP.DETECTION_IPC_URL
    left_home: str = "front"
    """The left arm's STAGE station — front / side / rear. A station's name is
    also the sector that arm serves, so the default left=front + right=rear is
    the front-and-back demo: drag a cube in front of the robot and the left
    hand follows it, drag one behind and the right hand does. The two arms may
    not share a station (same station == same sector == both arms reaching for
    one cube)."""
    right_home: str = "rear"
    """The right arm's STAGE station — see --left-home."""
    follow_min_faces: int = FOLLOW_MIN_FACES
    """Tag faces required before the follow goal MOVES. 1 by default (user
    08-11: a hand-held cube hides faces by construction, and the frozen-goal
    rhythm read as broken) — accepts single-face PnP and its tens-of-mm
    jitter; no grasp, and the envelope/sector fences stay on. 2 restores the
    gripper servo's strict bar for a smoothness-first run."""
    execute: bool = False
    """PUBLISH ARM COMMANDS. Without it this is a dry run: gaze still goes out
    (so the cameras scan and you can watch the locks land in the monitor), the
    corridor is still certified and every arm decision is printed, but no arm
    slot is ever sent. NOTE that a dry run therefore cannot CLAIM the arms
    either, so under --use-ik mink keeps them and they will drift — the dry run
    shows you the logic, not a still robot. Run it FIRST anyway: the REAR
    station's own note records its 07-17 re-validation as OFFLINE, with
    "hardware re-validation pending: watch the first staircase run", and the
    wrist was redesigned after that."""


def refuse(msg: str) -> int:
    print(f"[demo] REFUSED: {msg}")
    return 2


def main() -> int:
    args = tyro.cli(Args)
    homes = {"left": args.left_home, "right": args.right_home}
    why = validate_homes(homes)
    if why:
        return refuse(why)

    tel = TelemetryView(args.robot_ip)
    det = DetectionView(args.detection_url)
    pub = NNGPublisher("tcp://*:9874")
    rig = GazeRig(args.cubes)
    arms = [Arm(arm=a, home=homes[a]) for a in ARMS]
    time.sleep(0.3)                           # let real_env's dialer attach
    try:
        print(f"[gate] waiting up to {TEL_WAIT_S:.0f}s for telemetry — START "
              f"real_env NOW if it is not up; the arms are claimed on its "
              f"first packet, before mink can walk them out of the power-on "
              f"fold …")
        deadline = time.monotonic() + TEL_WAIT_S
        telemetry = None
        while telemetry is None and time.monotonic() < deadline:
            telemetry = tel.fresh()
            time.sleep(0.05)
        if telemetry is None:
            return refuse(f"no telemetry on 9870 within {TEL_WAIT_S:.0f}s — is "
                          f"humanoid_real_env running?")
        why = startup_gates(telemetry)
        if why:
            return refuse(why)
        enc0 = _arm_enc(telemetry)
        if enc0 is None:
            return refuse("arm joints missing/non-finite in telemetry")
        print("[gate] tilt OK")

        # CLAIM THE ARMS BEFORE ANYTHING ELSE, and hold a SNAPSHOT.
        # With real_env --use-ik and no commander on 9874, `arm_ik_active` is
        # True (`not _arm_stream_active`) and mink owns both arms — it holds the
        # power-on chest tuck as a CARTESIAN target, and that posture violates
        # mink's own 50 mm CollisionAvoidanceLimit (shoulder_3~wrist_1 is
        # 16.5 mm, wrist_1~wrist_3 10.2 mm), so the QP walks the arm out of the
        # fold: measured 86.6 deg away with wrist_3 pinned on its limit. One
        # joint slot ends it — `_arm_cart` goes [False, False] with the stream
        # live, `arm_ik_active` goes False, and the IK lets go.
        # A SNAPSHOT, never a live re-read: republishing the encoders every tick
        # lets gravity sag ratchet the arm downhill a packet at a time.
        hold = {a: joint_slot(a, [enc0[n] for n in ARM_JOINTS[a]]) for a in ARMS}

        def pump(work) -> None:
            """Run `work` on a thread while the frozen hold keeps publishing.
            real_env drops the stream after 0.5 s, so a multi-second synchronous
            certification would hand the arms straight back to mink."""
            th = threading.Thread(target=work, daemon=True)
            th.start()
            while th.is_alive():
                tick = time.monotonic()
                tel_now = tel.fresh()
                if tel_now is not None and args.execute:
                    out = rig.tick(tel_now, {"arm_targets": dict(hold)})
                    if out:
                        pub.publish(out)
                time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))

        for a in arms:
            a.start_homing(startup_seq(a.home_idx), time.monotonic())
            far, name = nearest_home(enc0, a.arm)
            print(f"[{a.arm}] home = {a.home.upper()} station, serving the "
                  f"{a.home.upper()} sector; startup staircase "
                  f"{' -> '.join(STATIONS[i].upper() for i in a.seq)} "
                  f"(starting {math.degrees(far):.0f} deg from {name})")

        # CERTIFY THE WHOLE STARTUP SCHEDULE against our own model before one
        # packet moves. The station-to-station hops are the audited 07-17 chain,
        # but the FIRST hop starts wherever the arms actually are — which, after
        # any mink excursion, is a posture no audit covers. preflight is
        # joint_monkey's auditor, the same one the cuRobo Gate C uses.
        model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        frames = staircase_frames(enc0, {a.arm: list(a.seq) for a in arms})
        cert: dict = {}
        print(f"[cert] sweeping the startup corridor ({len(frames)} frames, "
              f"both arms, receiver's own equal-rate curve) …")
        pump(lambda: cert.update(zip(("m", "where"), certify(model, frames))))
        why = corridor_verdict(cert.get("m", -1.0), str(cert.get("where")))
        if why:
            return refuse(why)
        print(f"[cert] startup corridor clear: {cert['m'] * 1000:.1f} mm at "
              f"{cert['where']} (floor {CERT_FLOOR_M * 1000:.0f})")
        print(f"[demo] cubes {list(args.cubes)}; "
              + ("PUBLISHING arm commands" if args.execute else
                 "DRY RUN — gaze only, no arm slot is sent (add --execute)"))

        t_home0 = time.monotonic()
        last_tick = time.monotonic()
        last_status = -1e9
        homed = False
        while True:
            now = time.monotonic()
            dt = float(np.clip(now - last_tick, PUB_DT, 2.0 * PUB_DT))
            last_tick = now
            telemetry = tel.fresh()
            if telemetry is None:
                return refuse("telemetry went stale — stopping, which hands "
                              "the arms to real_env's failsafe crawl")
            if any(telemetry.get("arm_fault") or [False, False]):
                return refuse("ARM FAULT latched by the dropout watchdog — "
                              "inspect the harness and restart real_env")
            enc = _arm_enc(telemetry)
            det.poll(rig)

            slots: dict[str, dict] = {}
            for a in arms:
                if enc is None:
                    continue                  # telemetry blip: command nothing
                if a.state == FOLLOW:
                    slot, leave = follow_slot(a, rig, now, dt,
                                              args.follow_min_faces)
                    if leave is None:
                        slots[a.arm] = slot
                        continue
                    print(f"[{a.arm}] FOLLOW -> HOME: {leave}")
                    # Re-plant the station: the follow drifted the arm off the
                    # exact posture, and staging's inward doctrine is that such
                    # a move must re-confirm its own station as the first hop.
                    a.start_homing([a.home_idx], now)
                    a.retreating = True
                if a.state == HOMING:
                    # RE-ACQUIRE MID-RETREAT (08-11 user: "if it sees the cube
                    # while retreating, don't hesitate, just fetch again").
                    # Only a retreat may be interrupted: it walks back through
                    # space the arm just followed through, in its own sector.
                    # The STARTUP staircase may not — its corridor was
                    # certified as one schedule, and mid-staircase postures
                    # (a rear arm passing SIDE) are cross-basin territory.
                    if a.retreating:
                        pick = pick_cube(a, rig, args.cubes, now,
                                         args.follow_min_faces)
                        if pick is None:
                            a.in_hits = 0
                        else:
                            key, med = pick
                            a.in_hits = a.in_hits + 1 if key == a.key else 1
                            a.key = key
                            if a.in_hits >= FOLLOW_IN_TICKS \
                                    and a.begin_follow(telemetry, key, med):
                                # First packet = the measured hand pose (zero
                                # motion); follow_slot ramps from next tick.
                                slots[a.arm] = cart_slot(a.stream,
                                                         a.quat_stream)
                                print(f"[{a.arm}] RETREAT -> FOLLOW {key} — "
                                      f"re-acquired at "
                                      f"{np.round(med, 3).tolist()}, no need "
                                      f"to finish going home")
                                continue
                    slot, arrived = step_home(a, enc, now, moves=args.execute)
                    slots[a.arm] = slot
                    if not arrived:
                        continue
                    a.state = HOME
                    a.retreating = False
                if a.state == HOME:
                    slots[a.arm] = joint_slot(
                        a.arm, station_posture(a.arm, a.home_idx))
                    pick = pick_cube(a, rig, args.cubes, now,
                                     args.follow_min_faces)
                    if pick is None:
                        a.in_hits = 0
                        a.key = None
                        continue
                    key, med = pick
                    a.in_hits = a.in_hits + 1 if key == a.key else 1
                    a.key = key
                    if a.in_hits < FOLLOW_IN_TICKS:
                        continue
                    if not a.begin_follow(telemetry, key, med):
                        continue              # no hand pose to ramp FROM
                    print(f"[{a.arm}] HOME -> FOLLOW {key} at "
                          f"{np.round(med, 3).tolist()} "
                          f"(r_xy {np.hypot(med[0], med[1]):.2f} m)")

            if not homed and all(a.state != HOMING for a in arms):
                homed = True
                print(f"[demo] both arms stationed in "
                      f"{time.monotonic() - t_home0:.0f}s — drag a cube into "
                      f"either sector")
            if not homed and args.execute and now - t_home0 > HOME_TIMEOUT_S:
                return refuse(f"the arms did not reach their stations within "
                              f"{HOME_TIMEOUT_S:.0f}s — the JOINT stream was "
                              f"published throughout, so this is a receiver or "
                              f"motor problem, not an IK one")

            # EVERY tick carries arm content, even when both arms are parked and
            # idle: real_env's arm-silence failsafe fires after 0.5 s and crawls
            # the arms to default at 0.125 rad/s. Re-sending a FIXED station
            # posture the arm is already at costs nothing and cannot walk the
            # arm downhill the way republishing live encoders would.
            packet = {"arm_targets": slots} if (slots and args.execute) else None
            out = rig.tick(telemetry, packet)
            if out:
                pub.publish(out)

            if now - last_status > STATUS_PERIOD_S:
                last_status = now
                seen = []
                for k in args.cubes:
                    p = rig.median(k, now)
                    seen.append(f"{k}:—" if p is None else
                                f"{k}@{sector_of(p).upper()}"
                                f"/{rig.faces_used(k, now)}f")
                state = "  ".join(
                    f"{a.arm}:{a.state}"
                    + (f"({a.key})" if a.state == FOLLOW else "")
                    for a in arms)
                print(f"[demo] {state}  |  {', '.join(seen)}")
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("\n[demo] Ctrl+C — publishing stops; real_env's failsafe crawls "
              "the arms to the power-on pose at 0.125 rad/s")
        return 0
    finally:
        det.stop()
        tel.stop()


if __name__ == "__main__":
    raise SystemExit(main())
