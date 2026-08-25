"""cuRobo reach-and-grasp mission driver — remote plan, local gates, local stream.

WHAT THIS IS. The standing dual-arm grasp mission on top of the remote cuRobo
planner (control/curobo_plan_server.py on the GPU machine): one or two cubes
(--cubes, one arm each), a declared world, one planned collision-free plan-0
route per motion — gated on OUR model before a single packet is sent — streamed
to the robot, then a real-time MPC session (--mpc, default on) that tracks each
hand onto the LIVE median of its own cube for the final approach. It
deliberately does NOT touch humanoid_auto_operator.py — the operator (with its
staircase certificates) remains the independent fallback path; this tool
REFUSES and exits rather than degrading into it.

    startup gates -> zero gimbals -> zero grippers -> [--chest-home]
      -> scan (operator serpentine) until EVERY cube is confirmed and reachable
      -> declared tables seen -> build scene
      -> plan(reach, aimed at the --reach-hover-mm hover) -> gates A-F
      -> [dry-run: noise-floor watch, arms held, stop]
      -> stream the gated route to MPC_HANDOFF_FRACTION
      -> MPC track onto the live cube (per-tick StepGate) -> grasp (fail-closed,
         empty-pinch retry) -> plan(home, scene minus the in-hand cubes, seeded
         at measured) -> gates -> stream -> hold the retracted posture, Ctrl+C

Plan + execute run in ROUNDS (JOURNEY_LEG_PLAN_ROUNDS): a SKIP from the execute
phase re-plans from fresh detections, a FATAL holds whatever is in the hand and
exits. Every retreat prints a CODED VERDICT naming the rule and the culprit —
T01-T03 the streamed phase, T04-T14 the MPC track, S01-S05 the session setup
(RETREAT_TRIGGERS) — and the per-run tally is printed at exit.

PROCESS WIRING (the operator's slot, operator NOT running — 9874 single-binder
rule): humanoid_real_env.py --use-ik --ee-service pointing at this machine;
humanoid_end_effector_service.py; camera streaming; humanoid_monitor.py with
--bodies listing the cubes (and tables). On the GPU machine:
control/curobo_plan_server.py (--server; see docs/OPERATIONS.md section 1).
This tool binds 9874 (arm+ee), subscribes 9870 (telemetry) and the monitor's
detection IPC socket, and publishes its gated route for the monitor's preview
overlay. It binds 9873 ONLY on an executing --journey — otherwise nav silence
keeps the base at zero, which is exactly the standing mission.

SAFETY CONTRACT (inherited, not reinvented):
  * every operator startup gate reruns here: telemetry fresh, tilt < refuse,
    both arms near a known home, every target inside the envelope;
  * the route is gated by verify_route — gates A-F: seam FK / start match /
    clearance sweep on OUR model / joint limits / home-posture end / wind-up
    family — before one packet is sent; every MPC tick passes the per-tick
    StepGate (limits / rate / segment clearance on OUR model) before it is
    published;
  * the whole plan-0 route is downloaded before motion starts — mid-motion
    link loss to the GPU machine cannot strand the arm; a dead MPC step is a
    coded retreat, not a frozen command;
  * any abort path simply STOPS PUBLISHING: real_env's 0.5 s arm-silence
    failsafe then crawls the arms to default at 0.125 rad/s — there is no
    "half-executed" state to manage;
  * grasp verdicts are id-matched and fail CLOSED: ambiguous/timeout leaves
    the fingers exactly as commanded and exits loudly (the operator's
    latch_grasp_uncertain contract, minus the mission-freeze — exiting IS the
    freeze here); after a grasp, failures refuse WITHOUT letting go;
  * arm_fault from the dropout watchdog aborts immediately.

GAZE AND PERCEPTION: THE OPERATOR'S OWN, MIGRATED NOT REWRITTEN. The first
hardware dry-run refused with "no cube confirmed" — this tool originally sent
no gaze, the gimbal held its forward stare, and a low front cube sits ~45 deg
below that view. The user's call (07-28): cuRobo is the icing — collision-
aware routes replacing the staircases — while seeing/finding/claiming stays
the PROVEN operator machinery. So GazeRig duck-types AutoOperator for
auto_operator.gaze + auto_operator.perception and runs their bodies verbatim:
the dual-camera panoramic SERPENTINE scan, per-cube camera CLAIMS with expiry
and NaN drops, parallax-corrected aim with the ±pi branch unwrap, sender-side
slew, park-when-done, CubeTrack confirmed/median semantics. Gaze rides the
9874 packets the receiver excludes from the arm-silence timer — so a DRY-RUN
MOVES THE CAMERA GIMBAL (scanning; and nothing else).

WORLD MODEL. Exactly one of --tables / --table / --no-table-world is required;
an empty world has to be stated, never defaulted. --tables names REGISTERED
tables (the perception package — perception/tagged_bodies/table/ here,
historically visual_servoing) and takes their pose from their AprilTags as a
full 6-DoF fit, with the slab geometry read from the registration's own
slab_cuboid() — so the obstacle the planner avoids and the slab the monitor
draws come from one set of numbers and cannot drift apart. A declared table
that is never confirmed REFUSES rather than warning: planning against a world
we claimed had a table in it is the false declaration that --no-table-world
exists to keep honest.

HEIGHTS. Two dials decide where the hand goes vertically. The TABLE Z PRIOR
(default on; --no-table-z-prior): a cube whose xy sits over a seen, sane table
takes its grasp z from the table surface at that xy plus half the cube instead
of the vision vertical, within TABLE_Z_PRIOR_CAP_M of it. The HAND FLOOR
(hand_floor_for_tables / check_hand_floor / HAND_FLOOR_*): the z floor handed
to the plan server with every solve is built from the registered table tops
and the gripper margins, and is clamped to never sit above the target — a
floor above the grasp pose does not degrade the reach, it makes it INFEASIBLE.
GRASP_ANCHOR_ABOVE_M is the client mirror of the server's GRASP_Z_ABOVE_M for
that arithmetic.

OTHER MODES. --journey walks to each cube in turn — survey once, then one leg
per cube in near-first order (tuck -> walk -> arrive -> measured stillness ->
re-acquire -> the same reach/grasp/retract) using the operator's own journey
gates, run verbatim; a failed leg stops the mission unless
--journey-continue-on-skip. --decoupled-arms solves each arm's route as its
own cuRobo problem and gates the MERGED 14-joint route. --anchor-trust
(default on) turns the tracking phase's soft aborts into "return the goal to
the gate-certified plan anchor and close there". --fresh-scan forgets every
cube after the gimbals zero so the scan demonstrably runs. --execute is
required for ANY arm motion; the default is a full dry-run.

KNOWN LIMITS. At most two cubes, one arm each, one planned assignment per
round (the server picks the pairing; both pairings are tried if its pick fails
our gates). A cube moving > CUBE_DRIFT_ABORT_M mid-route WARNS, it does not
abort — the MPC handoff re-seeds from live medians and tracks a genuinely
moved cube. The held cube is not modelled during retract (removed from the
scene as an obstacle, not attached to the hand). Base yaw (gyro integral) and
tilt (gravity) rotate the anchors during the track; base TRANSLATION is not
compensated. --no-mpc restores the pure open-loop route, and --reach-hover-mm
needs the MPC to close the last centimetres. --journey has been run on
hardware (two-cube walk->grasp->carry missions, August 2026) but is the least
mature mode — rig-specific JOURNEY_* constants, 30 mm tags undecodable past
~2 m; --decoupled-arms has not been run on hardware (docs/OPERATIONS.md
section 5).

MotionWatch SIZES the open-loop error budget rather than guessing at it: it
reports the standing noise floor in dry-run and the drift-under-motion during
--execute, so the difference is a measurement before anything closed-loop is
trusted with the arm.
"""

from __future__ import annotations

import dataclasses
import humanoid_site as _site
import math
import select
import sys
import threading
import time
from collections import deque

import mujoco
import numpy as np
import tyro

import humanoid_auto_operator as OP  # constants + mirror_arm only; no AutoOperator
from auto_operator import gaze as ao_gaze
from auto_operator import perception as ao_perception
from auto_operator.independent import (  # the journey's own gates, run verbatim
    journey_arms_tucked,
    journey_base_still,
    journey_survey,
)
from humanoid_curobo_client import (
    ARM_JOINTS_ALL,
    ARM_JOINTS_L,
    ARM_JOINTS_R,
    GATE_B_START_TOL_RAD,
    GATE_E_HOME_BRANCH_TOL_RAD,
    PlanClient,
    PlanServerError,
    STEP_MAX_RATE_RAD_S,
    StepGate,
    merge_decoupled_bundles,
    verify_route,
)
from humanoid_model import MJCF_MODEL_PATH
from humanoid_monitor import GimbalCameraFK

# Registered tagged bodies — the SAME source the monitor detects and draws from,
# which is the whole point: a table's slab geometry is authored once, in the
# perception package (perception/tagged_bodies/, historically visual_servoing),
# and both the perception side and the planner side read those same fields so
# they cannot disagree (TableConfig.slab_cuboid's own docstring makes that
# promise). The perception package lands on sys.path via the humanoid_monitor
# import above; importing it again here would be a second, conflicting insert.
from tagged_bodies import ALL_CONFIGS as _TAGGED_CONFIGS  # noqa: E402

TABLE_CONFIGS = {c.name: c for c in _TAGGED_CONFIGS if hasattr(c, "slab_cuboid")}

# The operator's gaze/perception bodies resolve their constants LIVE via
# _K(op) = sys.modules[type(op).__module__] — the S6 extraction contract. The
# GazeRig shim below lives in THIS module, so these names must exist here,
# verbatim from the operator. Do not tune them here; they have one home.
from humanoid_auto_operator import (  # noqa: F401  (consumed via _K resolution)
    CAM_PORTS,
    CAM_SITES,
    CUBE_LATCH_WINDOW_S,
    DET_LOST_RESCAN_S,
    DET_UNCLAIM_S,
    DT,
    GAZE_ORDER,
    GAZE_STARE_GRACE_S,
    GAZE_SLEW_RATE,
    SCAN_PITCH_DOWN,
    SCAN_SETTLE_S,
    SCAN_STEP_RAD,
    SCAN_YAW_RATE,
    SERVO_GRIPPER_KEYS,
    SERVO_MIN_INLIERS,
    TRIM_GAIN,
    TRIM_MAX,
    CubeTrack,
    # --- read by the imported journey gates, same _K(op) resolution ----------
    # journey_base_still and journey_arms_tucked look their constants up in the
    # module that defines type(op) — which for a GazeRig is THIS module. A name
    # missing here is an AttributeError raised mid-mission, on hardware, inside
    # a gate that had been passing: found exactly that way by a test.
    JOURNEY_POSTURE_TOL_RAD,
    JOURNEY_STILL_ANG,
    JOURNEY_STILL_DEBIT_S,
    JOURNEY_STILL_S,
    JOURNEY_STILL_S_BLIND,
    JOURNEY_STILL_VEL,
    JOURNEY_TUCK_TOL_M,
    JOURNEY_WALK_EE,
    POWERON_JOINTS,
    mirror_arm,
)

NNGPublisher = OP.NNGPublisher
NNGSubscriber = OP.NNGSubscriber

PUB_HZ = 20.0
PUB_DT = 1.0 / PUB_HZ
TEL_STALE_S = 0.5
DET_WAIT_S = 120.0                # the serpentine needs time: half-turn lead-in
                                  # ~6 s + 12.6 s per full row at SCAN_YAW_RATE
SETTLE_TOL_RAD = 0.05
SETTLE_TIMEOUT_S = 10.0
CUBE_DRIFT_ABORT_M = 0.05         # ⚠ NAMED "ABORT", ABORTS NOTHING (audit
                                  # 2026-08-06, I6). Its only consumer is the
                                  # mid-route WARNING at the drift check —
                                  # deliberate since 2026-08-04 ("REPORT, DON'T
                                  # VETO": the position was locked at plan time,
                                  # and arm-occlusion artifacts look exactly
                                  # like a moved cube). The MPC handoff
                                  # re-seeds from live medians and TRACKS a
                                  # genuinely moved cube, so the stream has no
                                  # reason to re-litigate perception. Kept as
                                  # the WARNING threshold; the name is
                                  # historical and the docstring above it lies
                                  # about aborting.
CUBE_DRIFT_WINDOW = 5             # the abort compares a MEDIAN of this many
                                  # recent sightings against the planned target,
                                  # not one frame. The plan target is itself a
                                  # median-of-5 (GazeRig.median), so gating a
                                  # robust number with a raw one was asymmetric:
                                  # a single 1-inlier PnP glitch — the exact
                                  # thing CubeTrack.hist exists to absorb —
                                  # could end a mission. Same window, so a real
                                  # 5 cm move still trips within ~0.1 s.
DRIFT_MIN_INLIERS = OP.SERVO_MIN_INLIERS
                                  # a sighting must be backed by this many tag
                                  # faces before the abort will act on it. NOT a
                                  # new threshold: it is the operator's own bar
                                  # for the gripper-base servo, "single-tag PnP
                                  # is garbage-prone and this feeds a control
                                  # loop", and the drift abort feeds one too.
SLOW_SOLVE_NOTE_S = 1.5           # say something once a solve outruns the
                                  # arm-silence timeout it is being held against

# --- journey: every bound below exists to make a deadlock impossible ----------
# THE RULE, and it is the whole design: a one-shot tool may end on any failure,
# a MULTI-LEG mission may not. Every per-leg failure path here terminates in
# "skip this cube LOUDLY and go to the next"; only an arm fault, telemetry loss
# or a dead plan server is fatal. Nothing retries without a bound, and nothing
# is ever re-queued into the leg loop — the order is latched once by the survey
# and the loop is a FOR over that fixed list, so the mission cannot livelock by
# construction. (The operator re-queues, and 2d1f9df is the record of what that
# cost: six livelock classes, two adversarial review passes.)
JOURNEY_TUCK_TIMEOUT_S = 25.0     # tuck must complete before gait onset; if it
                                  # does not, the leg is skipped rather than
                                  # standing tucked forever (operator class 5:
                                  # "EVERY journey run deadlocked at the
                                  # journey_arms_tucked gate")
JOURNEY_WALK_TIMEOUT_S = 150.0    # a leg that never reaches JOURNEY_ARRIVE_M
JOURNEY_NAV_HANDBACK_S = 10.0     # 08-07 audit: how long the walk tolerates
                                  # real_env's joystick-override latch before
                                  # calling the mission. A pause is legitimate
                                  # (someone grabbed the stick on purpose), so
                                  # it is not instantly fatal; but past this the
                                  # nav channel is dead and EVERY remaining leg
                                  # would time out identically, so it ends the
                                  # mission rather than skipping cube after cube
                                  # (a skipped cube is never retried).
JOURNEY_STILL_TIMEOUT_S = 30.0    # base never settles -> skip, do not stand
JOURNEY_ACQUIRE_S = 40.0          # per-leg re-acquire window at the stop point.
                                  # NOT DET_WAIT_S: 120 s of standing per leg is
                                  # how a mission silently parks (operator class
                                  # 6, the ASSIGN stall escape)
JOURNEY_LEG_PLAN_ROUNDS = 3       # full plan->reach->grasp attempts per leg.
                                  # The operator's contract is UNLIMITED grasp
                                  # retries while the cube stays servable, and
                                  # that is honoured INSIDE a round: an empty
                                  # pinch reopens and re-grabs without limit.
                                  # What is bounded is how many times we re-PLAN
                                  # — because a hand that keeps closing on
                                  # nothing is in the wrong place, and repeating
                                  # the same route is the livelock. Rounds
                                  # exhausted => skip loudly. This is the
                                  # cuRobo-side shape of the operator's
                                  # pre-grab EE-convergence timeout (class 2).
JOURNEY_EMPTY_REGRASP_S = 20.0    # inside one round, keep re-grabbing while the
                                  # cube is still SEEN and this long has not
                                  # passed; the cube going out of view ends the
                                  # round immediately (a certified-empty pinch
                                  # is never a reason to stop trying, but an
                                  # unseen cube is no longer servable)
GRASP_RESULT_TIMEOUT_S = 20.0     # == operator; service's own abort is ~53 s
CUBE_OBSTACLE_Z_PAD_M = 0.02      # the upstream object_cuboids height_pad
ROUTE_PREVIEW_URL = "ipc:///tmp/humanoid_route_preview.sock"
# The monitor subscribes here and draws the gated route's EE path into the
# live camera view (green polyline, red final dot, yellow goalset candidates)
# and as a point cloud under /robot/base_link in viser. Published AFTER the
# gates pass — what you see is exactly what --execute would stream — and
# cleared on exit. A monitor without the feature just never dials: free.
MPC_HANDOFF_FRACTION = 1.0        # stream the gated route to this fraction,
                                  # then hand the last stretch to the tracker.
                                  # The bulk of the path keeps the whole-route
                                  # certificate; the tracker owns only the
                                  # final approach, where the live cube matters
                                  #
                                  # 0.90 -> 1.0 (user, 08-12 late night):
                                  # "100% Curobo" restored for the seed1 run —
                                  # the 0.90 experiment below never got a
                                  # hardware verdict. The blind-handoff
                                  # caveat it targeted is live again: with
                                  # hover 0, an occluded cube at handoff
                                  # closes on the plan anchor, MPC unopened.
                                  #
                                  # 1.0 -> 0.90 (user, 08-12 night): at 1.0 +
                                  # hover 0, a blind handoff closes on the plan
                                  # anchor with the MPC session never opened —
                                  # on 1-face front grasps that anchor carries
                                  # a systematic z bias and the hand grasped
                                  # ~2 cm high all day. At 0.90 the stream
                                  # short-circuits its settle loop, settled
                                  # stays False, the BLIND shortcut therefore
                                  # cannot fire, and the MPC session ALWAYS
                                  # opens — the tracker (with the 08-12 floor
                                  # drop) flies the last stretch on live
                                  # evidence whenever there is any. WATCH: the
                                  # 08-06 lag numbers below are why 0.8 died
                                  # (T13 aborts from a 116-254 mm handoff
                                  # error); MPC_LEAD_SCALE=3 shipped since,
                                  # but a T13 relapse means this goes back up.
                                  #
                                  # 0.8 -> 1.0 (hardware 2026-08-06). At 1.0
                                  # the stream no longer short-circuits its
                                  # settle loop (see the `end_s <
                                  # bundle.duration_s` branch in stream_route),
                                  # so the tracker inherits a SETTLED arm.
                                  # WHY: the arms are driven at max_rate
                                  # 1.0125 rad/s by a position loop with NO
                                  # velocity feedforward, which can only hold
                                  # a speed by lagging — the lag IS the
                                  # torque. Measured in that run's own MPC
                                  # phase, K = 1.7-3.7 rad/s per rad of lag,
                                  # so following the stream needs 0.27-0.60
                                  # rad of lag = 116-254 mm of hand error at
                                  # the handoff posture. Handing over at 0.8
                                  # passed that error straight to the tracker
                                  # (observed 165 and 233 mm), where the
                                  # ABSOLUTE progress guard (5 mm / 6 s)
                                  # convicted a hand that was closing, just
                                  # from too far out — T13 x2, then a third
                                  # round with no gated route at all. A
                                  # correctly tracked route arrives AT the
                                  # hover by the handoff: 26 mm mean, 42 mm
                                  # max over six solves to those same cube
                                  # positions. The 0.8 saving was a fraction
                                  # of a second; it cost whole missions.
MPC_HOVER_ARRIVE_M = 0.035        # hand-to-HOVER-point distance that flips a
                                  # side from phase 1 (chase the point 30 mm
                                  # ABOVE the live cube) to phase 2 (descend
                                  # onto the cube). The 8d17dfd hover-endpoint
                                  # first cut lifted only the plan-0 route —
                                  # which the MPC handoff at 80% never
                                  # executes (adversarial review F1) — so the
                                  # staging lives HERE, in the tracker's own
                                  # goals, where it actually runs: arrive
                                  # above, then a live-corrected vertical
                                  # descent. Slightly wider than
                                  # MPC_CONVERGED_M so arrival is reachable
                                  # while the true-goal convergence check
                                  # (which stays vs the CUBE) cannot fire
                                  # early at the hover (30 mm > 20 mm).
DESC_ANCHOR_FRESH_S = 2.0         # at the descent handover, re-anchor the goal
                                  # to the freshest >=2-face median no older
                                  # than this. Phase 1 legitimately follows
                                  # 1-face updates on the leash, and 1-face
                                  # PnP carries a DIRECTION-CONSTANT lateral
                                  # bias (single-face view ambiguity) — the
                                  # top-down descent then lands that bias in
                                  # full ("grasps left of the cube, every
                                  # time", 2026-08-03). A <=2 s-old 2-face fix
                                  # is the best truth available; without one
                                  # the goal is kept and the operator is told
                                  # to expose a second face.
MPC_DESCENT_M_S = 0.05            # phase-2 descent rate: the effective goal's
                                  # z RAMPS from the hover down to the cube at
                                  # this speed instead of jumping, so the
                                  # commanded path is a strict vertical rail
                                  # (~0.6 s for the 30 mm hover) that the MPC
                                  # tracks closely — "straight down" as
                                  # commanded geometry, not as a hope
MPC_CONVERGED_M = 0.02            # our-FK-to-live-target distance that counts
                                  # as arrived, per hand
MPC_CONVERGED_TICKS = 8           # consecutive in-tolerance ticks before the
                                  # grasp fires (~0.4 s at 20 Hz): one lucky
                                  # sample through detection noise is not
                                  # arrival
MPC_CLOSE_Z_MAX_M = 0.005         # 08-10 (user): 8 -> 5 mm, tighter still.
                                  # ORIGINAL NOTE (user: "why does it always grasp
                                  # HIGH?"): the measured z residual allowed AT
                                  # CLOSE. Every close that day carried z +11
                                  # to +26 mm hiding inside the 3-D bars — the
                                  # hand approaches from ABOVE, so the
                                  # residual's z part is a systematic high
                                  # bias, and the grasp-anchor dial was being
                                  # used to compensate a closing-logic leak.
                                  # Applies to the converge close and the
                                  # blind-anchor close; the budget/no-progress
                                  # salvage closes stay exempt as the bounded
                                  # escape hatch (they print how far out).
MPC_HANDOFF_CLOSE_Z_MAX_M = 0.015  # 08-13 (user: "lat <= 40 mm, z <= 15 mm —
                                  # do NOT open the MPC session, close at
                                  # once"). The handoff-close z bar, WIDER
                                  # than the in-session MPC_CLOSE_Z_MAX_M on
                                  # purpose: the day's forensics showed the
                                  # session doing net damage from exactly
                                  # this posture — its solver's own elbow
                                  # collision model runs away from near+high
                                  # goals (T10 standoffs at -21..+5 mm
                                  # clearance, then a T13 where it dragged
                                  # the cube 95 mm with the open jaws). A
                                  # 60 mm cube's top face is +30 mm, so a
                                  # z +15 close still bites the upper half;
                                  # opening a session to shave those 15 mm
                                  # risks losing the cube entirely.
MPC_BLIND_ANCHOR_M = 0.04         # ...and only when every hand is this close to
                                  # its PLAN ANCHOR. Blindness alone convicted
                                  # nothing: hardware 2026-08-06 closed at 51
                                  # and 76 mm out and both grippers came back
                                  # EMPTY, because the goal had drifted 40-55
                                  # mm off the anchor on 100%-1-face evidence
                                  # and the arms had followed it there. The
                                  # anchor is the scan-time fix, taken with the
                                  # camera parked and nothing occluding it —
                                  # the one number a drifting 1-face estimate
                                  # cannot corrupt. Blind-but-far falls through
                                  # to the existing guards instead of closing.
MPC_BLIND_GRASP = True            # Close the gripper the moment EVERY tracked
                                  # cube goes dark, instead of tracking a goal
                                  # nothing is measuring any more. See the long
                                  # note at the check itself in mpc_track: the
                                  # tracker's whole job is correcting what the
                                  # cube did during the open-loop stream, and a
                                  # blind tracker is servoing to IMU dead
                                  # reckoning. On a top-down approach the tags
                                  # go dark because the gripper is standing on
                                  # them — the descent freeze already says so.
                                  # Set False to restore the old behaviour,
                                  # where a blind track runs to a stall guard
                                  # or the budget.
MPC_MAX_S = 25.0                  # tracking budget; on timeout, within
                                  # GRASP_EE_OK_M still grasps (open-loop
                                  # parity), beyond it SKIPs to a fresh plan
MPC_STEP_FAIL_MAX = 20            # consecutive unusable ticks (insane command,
                                  # gate-rejected, transport error) before the
                                  # track aborts — ~1 s of not moving safely
                                  # at the 20 Hz cadence. Transport-error
                                  # ticks each block up to MPC_STEP_TIMEOUT_S
                                  # first, so a dead server is bounded at
                                  # ~8 s of held arms, not 1 s — still an
                                  # abort, just a slower-counted one
STREAM_FOLLOW_LAG_WARN_RAD = 0.3  # SAY IT, don't abort. The bar below was
                                  # raised 0.12 -> 0.8 to tolerate the can21
                                  # harness, and that silence is what hid the
                                  # 2026-08-06 failure: 0.8 rad is 236-341 mm
                                  # of hand error at the handoff posture, so
                                  # every bit of a mission-killing deficit sat
                                  # UNDER the alarm. This threshold only
                                  # prints (rate-limited), so the abort
                                  # behaviour the harness needs is untouched
                                  # while the operator gets the number back.
STREAM_FOLLOW_LAG_RAD = 0.8       # streamed-target vs measured divergence that
STREAM_FOLLOW_LAG_TICKS = 10      # sustained this many ticks (~0.5 s) means
                                  # the arm is NOT following the stream.
                                  # WHY (hardware 2026-08-01 night): the left
                                  # elbow physically wedged against the torso
                                  # — encoders live, current gentle — and the
                                  # stream played to the end with the arm
                                  # stuck at mid-reach; the MPC inherited it
                                  # as "no progress, 345 mm". This check
                                  # backs out instead.
                                  # 0.12 -> 0.8 (user order 2026-08-04):
                                  # right-elbow harness drag tripped the old
                                  # bar three times (0.31-0.39 rad lags with
                                  # clean walk-backs — drag, not a wedge) and
                                  # cost the rounds. The trade accepted with
                                  # the order: a TRUE wedge now grinds up to
                                  # ~0.8 rad of PD error (drive caps still
                                  # bound the torque, the C gate still
                                  # certified the route's clearance) before
                                  # the stream backs out. Replacing the can21
                                  # harness is what earns this bar back down.
                                  # Normal lag is receiver ramp (~1-3 ticks
                                  # of route speed) + gravity sag (~0.02-0.04
                                  # with --grav-comp): well under the bar.
TILT_COMP_PIVOT_M = 0.70          # ankle pivot depth below base_link. The
                                  # grasp goal is CUBE-anchored (user
                                  # requirement 2026-08-01): a frozen goal is
                                  # a base-frame snapshot, and torso sway
                                  # moves the base under it — 19 mm per deg
                                  # of lean at these reaches, the dominant
                                  # grasp error. While detections are frozen,
                                  # each tick re-expresses the goal into the
                                  # CURRENT base frame from the IMU's gravity
                                  # delta, rotating about the ankle. The
                                  # pivot depth is an estimate; its error
                                  # enters only as (delta-tilt x delta-depth),
                                  # second order.
MPC_GOAL_PREVIEW_S = 0.5          # live-goal marker cadence to the monitor —
                                  # the green route is a plan-time snapshot
                                  # that rides base_link, so it CANNOT show
                                  # whether the tracker is locked; these
                                  # markers are the live aim, drawn where the
                                  # tracker is actually going.
MPC_LOWEV_LEASH_M = 0.12          # ONE tag face is evidence too — on a leash.
                                  # 0.12, not 0.06 (hardware 2026-08-01
                                  # night): the leash is centred on the
                                  # SCAN-TIME anchor, and with 1-face tags
                                  # all day the anchor itself was dirty — the
                                  # live-aim marker showed the hand converged
                                  # 5-8 cm BESIDE the live cube because every
                                  # closer-range (better) 1-face median sat
                                  # just outside the old 6 cm leash and was
                                  # silently rejected. At grasp range the
                                  # cube fills the frame and the live median
                                  # outranks the far-away scan fix; the wide
                                  # leash lets it win while still fencing the
                                  # wild ghost class, with the envelope, the
                                  # slew, the no-progress guard and the
                                  # empty-pinch retry underneath.
                                  # The dominant grasp error today is tilt-
                                  # induced base-frame drift (30-93 mm
                                  # measured 2026-08-01), and a 1-face
                                  # sighting carries that common-mode signal
                                  # exactly like a 2-face one; freezing the
                                  # goal forfeits it. What 1-face PnP CANNOT
                                  # be trusted for is absolute truth (flip
                                  # ambiguity + slide-along-ray: three
                                  # incidents at 5+ cm), so a low-evidence
                                  # update may pull the goal at most this far
                                  # from the PLAN target — the scan-time
                                  # anchor measured before any occlusion.
                                  # Worst case it is wrong: an empty pinch,
                                  # which the grasp-retry contract recovers.
MPC_DESCENT_1FACE_TRACKS = False  # 2026-08-09: the 08-08 override ("the
                                  # descent MUST use it if you can see the
                                  # tags") is REVOKED by its own night of
                                  # data — with 1-face-only evidence the
                                  # tracked goal was dragged 37/61/68/105 mm
                                  # off the anchor and the hand chased it
                                  # into the night's leading abort family
                                  # (T13 x5), while EVERY successful grasp
                                  # took the other path: blind early, close
                                  # on the anchor. Freeze semantics restored:
                                  # during the descent only >=2-face evidence
                                  # moves the goal; 1-face freezes onto the
                                  # scan-time anchor (the one number 1-face
                                  # slide cannot corrupt). True restores the
                                  # override; both behaviors stay pinned in
                                  # test_mpc_track. NOTE: journey legs now
                                  # plan DIRECT (hover 0) and have no descent
                                  # phase at all — this flag only governs the
                                  # standing mission's staged hover.
TARGET_STABLE_SPREAD_M = 0.02     # 08-12 (user: "sometimes it takes an
                                  # unstable estimation as the ground truth
                                  # and plans a waypoint into empty space"):
                                  # the settled fix may only become the plan
                                  # anchor when the samples behind it agree
                                  # within this — a jumping 1-face stream
                                  # mid-flip elects medians no two frames
                                  # agree on, and the anchor then centres the
                                  # 120 mm leash on a ghost.
TARGET_STABLE_MIN_N = 4           # ... and at least this many samples voted.
TARGET_STABLE_EXTRA_S = 2.0       # BOUNDED patience (user: no long standing
                                  # waits): the gate may hold adoption at
                                  # most this long past first sight; then it
                                  # proceeds on the best evidence in hand,
                                  # loudly — a ghost anchor costs a full
                                  # round (~20 s), so 2 s of insurance is a
                                  # net latency WIN, and a fix that is stable
                                  # from the start pays ZERO.
LOWEV_STABLE_WINDOW_S = 3.0       # 08-10 (user, after the e-cube T09 run:
                                  # 1-face fixes drifted 44-51 mm, the goal
                                  # was dragged 40 mm off its anchor and the
                                  # MPC re-solves went unstable): when the
                                  # fresh latch window holds ONLY 1-face
                                  # evidence, the median widens to this many
                                  # seconds of history — the goal settles on
                                  # the stable central position of the noisy
                                  # stream ("the most frequent place the cube
                                  # shows up") instead of chasing each slide.
                                  # A >=2-face decode ANYWHERE in this window
                                  # outranks the 1-face stream entirely (best
                                  # evidence wins, extended in time). The
                                  # fresh window must be non-empty for the
                                  # widening to fire, so a dark cube still
                                  # reads None and every blind-grasp path is
                                  # untouched. Cost, stated: a cube GENUINELY
                                  # nudged during a 1-face-only stream is
                                  # followed ~4x slower; the leash, slew and
                                  # empty-pinch retry own that case.
LOWEV_HIST_N = 96                 # sighting history depth for the reach rig's
                                  # CubeTracks — the operator's deque(maxlen=5)
                                  # holds ~0.2 s at camera rate, which silently
                                  # capped even the 0.7 s fresh median to ~5
                                  # samples and makes a 3 s stable window
                                  # impossible. 96 covers 3 s at ~30 Hz.
                                  # Widened ONLY on GazeRig's own tracks; the
                                  # operator's mission latches keep their own
                                  # class default.
MPC_GOAL_SLEW_M_S = 0.05          # goal motion is slewed for EVERY evidence
                                  # level: a credible 9 cm hop (2026-08-01
                                  # run 2) re-routed the solver violently;
                                  # fed as a 1.8 s glide the same correction
                                  # is a tracking correction, not a new plan.
MPC_EXEC_LAG_FRACTION = 0.6       # a NOT-executing conviction ALSO requires
                                  # the binding joint's lag to reach this
                                  # fraction of the clamp lead
                                  # (MPC_LEAD_SCALE*max_rate*PUB_DT): a truly
                                  # stalled arm sits pinned AT the lead
                                  # ceiling, while an arm tracking a solver
                                  # that DITHERS near convergence shows a
                                  # tiny lag with a huge asked-sum (hardware
                                  # 2026-08-04: 'moved 0.28 of 1.89 rad, 15%'
                                  # with the binding joint lagging 0.07 rad —
                                  # ceiling 0.152 — the ratio was inflated by
                                  # command oscillation, the arm was
                                  # following; the 08-01 integral-form
                                  # false-conviction story, ratio edition)
MPC_EXEC_WINDOW_TICKS = 40        # executor guard, RATIO form. The first
MPC_EXEC_MIN_ASKED_RAD = 0.25     # version integrated (asked - moved) and
MPC_EXEC_MIN_RATIO = 0.20         # convicted the innocent: on final approach
                                  # the target sits a constant hair ahead
                                  # while the arm decelerates into it, so the
                                  # integral grew fastest EXACTLY when the
                                  # hand was almost there — seven straight
                                  # hardware rounds aborted at drift 7-15 mm,
                                  # alternating arms, all at the 0.25
                                  # threshold (2026-08-01 night, the user's
                                  # "reaches out then retreats, it was almost
                                  # there"). What separates a wedge from a
                                  # slow finish is the RATIO: over the last
                                  # ~2 s window, an arm that EXECUTED less
                                  # than 20% of what it was asked (>=0.25 rad
                                  # asked) is stuck; a slow-but-moving arm
                                  # runs at 40-100% and converges.
MPC_EXEC_GUARD = True             # T12 CONVICTION. ON (user 2026-08-05: it
                                  # was switched off for one round and put
                                  # straight back — a wedged arm pushing on
                                  # its obstruction for MPC_NO_PROGRESS_S
                                  # instead of ~1 s is not a trade worth
                                  # making). The switch stays because the
                                  # question it settles is worth one word: a
                                  # run with it off still prints every number
                                  # the abort would have carried, so the
                                  # conviction can be audited without being
                                  # suffered. The false-conviction mode it was
                                  # switched off for — a speed-limited arm
                                  # reading a structurally low ratio — is
                                  # answered properly by the speed floor
                                  # below, which is where that belonged.
MPC_EXEC_MIN_SPEED_RAD_S = 0.10   # ABSOLUTE floor that outranks the ratio: an
                                  # arm covering this much ground is not the
                                  # thing this guard hunts. The ratio's
                                  # denominator is a speed OUR OWN CODE
                                  # forbids — `cand = enc + clip(raw - enc,
                                  # +-MPC_LEAD_SCALE*max_rate*dt)` caps the
                                  # commanded lead at 0.152 rad, there is no
                                  # velocity reference on the wire (it is
                                  # written zero and never updated), so kd
                                  # brakes toward standstill and the arm
                                  # equilibrates wherever kp*lag balances
                                  # gravity, friction and that braking. The
                                  # solver, meanwhile, asks for the full
                                  # streaming rate whenever the hand is far
                                  # out. Hardware 2026-08-05, twice: the hand
                                  # 151/158 mm away, the solver asking 1.12
                                  # rad/s, the arm delivering 0.22 rad/s at a
                                  # lag 70% of the ceiling — a 20% ratio that
                                  # is ARCHITECTURE, not a wedge, convicted at
                                  # exactly the 20% bar. A wedged arm reads
                                  # ~0.00 rad/s and is still convicted; a
                                  # slow-but-closing arm now falls through to
                                  # MPC_NO_PROGRESS_S, which asks the question
                                  # that matters — is the HAND getting closer
                                  # — and to the budget behind it, both of
                                  # which grasp anyway inside GRASP_EE_OK_M.
MPC_TEL_REPEAT_WARN = 0.25        # EVERY execution number above is measured
                                  # through the telemetry stream: `moved` is
                                  # a difference of two encoder reads, and a
                                  # tick that got no new packet contributes
                                  # zero motion no matter what the arm did.
                                  # Hardware 2026-08-05: real_env's 50 Hz loop
                                  # ran late on 20-46% of its ticks (CPU
                                  # contention — the unpinned monitor's 340%
                                  # landing on real_env's five pinned cores),
                                  # the ratio read 18%, and the verdict named
                                  # a joint. The guard cannot tell a starved
                                  # PUBLISHER from a stalled ARM, so when this
                                  # fraction of a window carried no update it
                                  # says so — out loud, and inside the verdict.

MPC_ESCAPE_GOAL_STILL_M = 0.020   # ...and only while the goal sits this close
                                  # to its plan anchor. A goal that has moved
                                  # further than this is a cube that actually
                                  # went somewhere, and then a growing hand
                                  # distance is the goal running away rather
                                  # than the solver walking out — the drift
                                  # and envelope aborts own that case, and
                                  # closing would grasp where the cube WAS.
MPC_ESCAPE_MARGIN_M = 0.010       # 08-13 (user): how far the hand must recede
                                  # from its own best in-ring distance before
                                  # the ESCAPE GUARD closes. Big enough that
                                  # solver dither and the goal's 50 mm/s slew
                                  # cannot trip it (the observed dither band
                                  # is 1-3 mm); small enough to fire long
                                  # before the 60 mm ring is crossed, which
                                  # is what disqualified the grasp-anyway
                                  # branch in the 47 -> 67 mm track.
MPC_NO_PROGRESS_S = 3.0           # a track whose worst hand has not moved
                                  # (08-12 night, user: 6 -> 3 — halve the
                                  # wait. The 6 s caution priced a wrong
                                  # verdict at a full retreat; under ANCHOR
                                  # TRUST a wrong verdict costs one certified-
                                  # EMPTY pinch, so the judge may be quick.)
MPC_PROGRESS_MIN_M = 0.005        # >= 5 mm closer to its goal in 3 s is STUCK
                                  # — a physically blocked arm produces sane,
                                  # gate-passing, useless commands forever
                                  # (fails resets every accepted tick), and
                                  # without this the only exit was the 25 s
                                  # budget, ending in sustained servo pressure
                                  # against the obstruction and possibly a
                                  # blind "grasping anyway" (review
                                  # 2026-08-01). Aborting routes it to the
                                  # retreat instead.
MPC_LAND_MAX_TICKS = 20           # how long a grasp-anyway exit may spend landing
                                  # the descent rail before it closes regardless
                                  # (~1 s at 20 Hz; the 15 mm rail needs 6 ticks
                                  # at MPC_DESCENT_M_S, so this is 3x margin).
                                  # Bounded on purpose: T12 fires when an arm is
                                  # STALLED, and a stalled arm may not descend
                                  # either — waiting forever would trade a 15 mm
                                  # error for a hang.
MPC_ENVELOPE_STRIKES = 12         # consecutive out-of-envelope ticks before T07
                                  # (08-09, user: 8 -> 12 after a boundary cube's
                                  # 1-face noise held 8 straight out-of-envelope
                                  # ticks at r_xy 0.603 vs the then-0.60 bar)
                                  # aborts. It used to fire on ONE tick, alone
                                  # among the track guards — the clearance
                                  # standoff needs 8, step failures 20 —
                                  # while the thing it judges is the noisiest
                                  # signal we have: a 1-face cube estimate
                                  # that drifts 26-100 mm. Hardware
                                  # 2026-08-06: six aborts in an evening, all
                                  # at r_xy 0.600-0.613 against a 0.600 bar,
                                  # one of them printing "move it 0 cm
                                  # CLOSER" — the tool asking for zero
                                  # centimetres because an estimate wobbled
                                  # 0.4 mm past a soft limit.
MPC_CLEARANCE_STANDOFF_MAX = 8    # consecutive CLEARANCE-class gate rejections
                                  # before the track aborts early. A clearance
                                  # rejection is a STANDOFF, not noise: the
                                  # solver's optimum threads below our floor
                                  # and re-solving from the same state keeps
                                  # producing the same path (hardware
                                  # 2026-08-01: 21 identical elbow-torso
                                  # rejections, arm frozen mid-air the whole
                                  # time). 8 ticks (~0.4 s) proves intent;
                                  # waiting the full MPC_STEP_FAIL_MAX just
                                  # delays the retreat.
RETREAT_STEP_MIN_RAD = 0.005      # measured-history dedup: record a posture
                                  # only when some joint moved this much — a
                                  # wedged arm contributes a handful of
                                  # entries instead of 20 per second, and the
                                  # replay stays proportional to actual travel
RETREAT_START_TOL_RAD = 0.35      # retreat only walks a corridor the arm is
                                  # actually IN: if the measured posture is
                                  # further than this from the last published
                                  # target, the recorded footsteps are not
                                  # where the arm is and replaying them blind
                                  # could sweep unchecked space — skip, and
                                  # let the failsafe own the arms instead.
MPC_CMD_SANE_RAD = 0.35           # |q_next - measured| above this on any joint
                                  # is not a tracking correction the clamp can
                                  # domesticate — it is the solver commanding a
                                  # different posture, and following it slowly
                                  # is still following it. Reject outright.
                                  # (Hardware 2026-08-01: legitimate first
                                  # steps after warm-up measured ~0.10 rad; a
                                  # scripted-fault test uses 1.0.)
MPC_LEAD_SCALE = 3.0              # target-lead budget in units of one tick's
                                  # stream step (max_rate*dt). Forensics
                                  # 2026-08-02, rear-sector stalls: anchoring
                                  # the target AT the measurement capped motor
                                  # PD error at ONE tick's step (~1.4 N*m at
                                  # the shoulder) while the MIT frame streams
                                  # velocity-ref = 0, so kd BRAKES commanded
                                  # motion — zero-load tracking tops out near
                                  # 50% of the stream and ~1 N*m of harness
                                  # drag starves it to the observed 13-20%.
                                  # A 3x lead restores ~4 N*m of kp authority
                                  # against drag; the receiver's 0.7 rad/s cap
                                  # still bounds EXECUTED speed, and
                                  # MPC_CMD_SANE_RAD still rejects jumps.
MPC_STREAM_DRIFT_ABORT_M = 0.09   # pre-handoff drift abort when the tracker
                                  # will run: the open-loop 5 cm bound killed
                                  # round 1 on torso-lean drift the terminal
                                  # tracker exists to absorb. 9 cm still
                                  # catches a genuinely displaced cube (whose
                                  # route SHAPE is stale) while sub-9 cm rides
                                  # through to the tracker, which re-aims
                                  # every tick and re-checks the envelope
SERVER_WAIT_S = 90.0              # startup: wait this long for the plan server
                                  # before refusing — a server restart is ~60 s
                                  # of CUDA warm-up and this check runs before
                                  # any motion, so waiting is free
EE_ALIVE_WAIT_S = 10.0            # how long to wait (arms held) for real_env's
                                  # ee_alive to flip True — the EE service
                                  # heartbeats at 1 Hz and a restart needs a
                                  # redial plus one beat; sampling once was a
                                  # race that refused freshly-started stacks
EE_LOAD_EVIDENCE_S = 10.0         # how long to wait (arms held) for a servo
                                  # load reading before deciding a marked
                                  # gripper's fate — the service's thermal poll
                                  # alternates sides every 2 s, so ~4 s/side
EMPTY_GRIP_LOAD_MAX = 60.0        # |load| (0..1023) at or below which a hand is
                                  # verifiably exerting NO grip force. Contact
                                  # detection fires at 600; a static hold at
                                  # reduced torque still reads hundreds; an idle
                                  # servo reads tens. Deliberately far below the
                                  # hold band: a false refuse costs a re-run, a
                                  # false open drops a cube.
CUBE_REACQUIRE_S = 20.0           # a target that vanished between selection and
                                  # the scene build gets this long to be re-seen
                                  # (arms held) before the round is charged —
                                  # tags flicker at range; a 2 s occlusion must
                                  # not cost a five-terminal relaunch
PLAN_RETRIES = 5                  # plan-0 is a randomized-seed search and flips
                                  # INFEASIBLE/route with zero input difference
                                  # (upstream's "bucket 1"; measured live INF,INF,OK
                                  # at ~0.15 s per verdict on the warm graph) —
                                  # so an INFEASIBLE verdict earns retries, and
                                  # only a unanimous run of them is believed
CHEST_HOME_TOL_M = 0.03           # ⚠ NO READER (audit 2026-08-06, I6): the
                                  # chest-home ramp converges in JOINT space
                                  # (SETTLE_TOL_RAD), which is what the
                                  # 2026-07-28 forensics showed the Cartesian
                                  # tolerance could never reach. Retained only
                                  # as documentation of that measurement;
                                  # nothing gates on it, so tuning it does
                                  # nothing.
                                  # measured EE within this of CHEST_HOME_EE counts
                                  # as arrived; mink settles well inside it and a
                                  # tighter bound would just wait on servo droop
CHEST_HOME_WAIT_S = 25.0          # sized for the WORST recognised start, not
                                  # the common one: on a fresh boot the ramp
                                  # measures ~0 and exits at once (power-on ==
                                  # chest home since 07-31); from the 07-30
                                  # legacy chest posture the glide is 53 deg of
                                  # shoulder_1 with a ~56 deg wrist reorient,
                                  # and mink caps at 0.5 rad/s
GIMBAL_ZERO_TOL_RAD = 0.05        # both cameras measured this close to their
                                  # joint zeros counts as "at zero" for the
                                  # forced pre-scan park (user 2026-08-03:
                                  # every run must zero the cameras FIRST,
                                  # then start the sweep — the scan clock
                                  # starts at zero, so the sweep's coverage
                                  # order is deterministic instead of
                                  # depending on where the last run parked)
GIMBAL_ZERO_WAIT_S = 6.0          # worst gimbal excursion is ~pi at the 1.5
                                  # rad/s sender slew ≈ 2.1 s; 6 s covers a
                                  # sluggish joint, and a timeout WARNS and
                                  # proceeds rather than ending a mission
                                  # over a camera park
GRIPPER_ZERO_S = 3.0              # command-settle window only (was 12.0, a
                                  # dead wait that was most of the user's
                                  # lock-to-reach 10 s, 2026-08-03). The
                                  # service runs the calibration as its OWN
                                  # background task per side regardless of
                                  # this loop, and NOTHING downstream needs
                                  # the result until hand_grab — which is
                                  # behind the reach route + MPC converge,
                                  # 15-20 s away. 3 s covers the command
                                  # burst + a settle; the pinch/open finishes
                                  # in parallel while the arms travel.
                                  # 08-10 (user FSM order): the mission then
                                  # WAITS for completion before the survey —
                                  # see the two constants below.
GRIPPER_ZERO_DONE_S = 60.0        # zeroing-completion budget: both sides'
                                  # `zeroing` telemetry flags must read False
                                  # (measured single round ~15-30 s); a stuck
                                  # calibration refuses the mission
GRIPPER_ZERO_FIELD_GRACE_S = 2.0  # if the flags are absent past settle+grace,
                                  # real_env/the service predate the field —
                                  # warn and fall back to the old settle-only
                                  # behaviour instead of blocking a mission on
                                  # a telemetry field that will never appear
POSTURE_WAIT_S = 90.0             # the failsafe crawl from the raised chest home
                                  # was ~30 s at the old 0.025 rad/s (now ~6 s at 0.125); 90 leaves room for a
                                  # further pose without waiting on a parked arm
                                  # forever (which is reported, not just waited on)
OBSERVE_S = 6.0                   # dry-run: sample this long, standing still,
                                  # to separate detection NOISE from motion-
                                  # induced drift (see MotionWatch)
TABLE_POSE_SPREAD_WARN_DEG = 5.0  # a static slab's fitted orientation should
                                  # barely move; more than this means one tag
                                  # face is mis-fitting or the registered
                                  # tag_roll convention is wrong
TABLE_WAIT_S = 20.0               # declared tables must be SEEN before planning
                                  # (a declared-but-unseen table is a false
                                  # world model, so this waits and then refuses)


TABLE_POSE_SPREAD_REFUSE_DEG = 20.0  # above this the fit is junk, not noisy: a
                                     # 1.2 x 0.6 m slab at a junk orientation is
                                     # a WRONG obstacle, and planning against
                                     # geometry that is not there is worse than
                                     # planning against none


class TablePoseUnstable(Exception):
    """A declared table's fitted orientation is too unstable to plan against."""

    def __init__(self, name: str, spread_deg: float):
        super().__init__(
            f"table '{name}' orientation wobbles {spread_deg:.1f} deg inside one "
            f"{TABLE_POSE_SPREAD_REFUSE_DEG:.0f}-deg window — that is not a "
            f"measurement, and a 1.2x0.6 m slab placed at a junk orientation "
            f"would certify the route against geometry that is not there. "
            f"Likely causes: the tag strip is edge-on or partly occluded, the "
            f"table is too far for a 30 mm tag, or only one face is visible. "
            f"Move so the tag strip faces a camera, or drop --tables and use "
            f"--table / --no-table-world if the table is not in the way.")
        self.name = name
        self.spread_deg = spread_deg


class TargetLost(Exception):
    """A cube confirmed during selection had no median left at scene-build time.

    Not a warning. Dropping it would send the planner one target instead of two
    and turn a two-arm mission into a one-arm one, which then reads as a
    successful run that grasped half of what was asked for.
    """

    def __init__(self, name: str):
        super().__init__(
            f"'{name}' was confirmed and reachable during selection but has no "
            f"fresh median now — its tags went out of view between choosing it "
            f"and building the scene. Refusing rather than planning for the "
            f"other cube alone.")
        self.name = name


class TableNotSeen(Exception):
    """A --tables entry was declared but never confirmed by the cameras."""

    def __init__(self, name: str):
        super().__init__(
            f"table '{name}' was declared with --tables but its tags were never "
            f"confirmed. Planning would use a world model that claims a table "
            f"exists where none was measured — refusing. Check: the monitor's "
            f"--bodies lists '{name}'; the tags are lit and unoccluded; the "
            f"gimbal can actually see them.")
        self.name = name


class MotionWatch:
    """Turns "does the torso tilt matter?" into numbers, per run.

    WHY. The plan-0 route is open-loop: computed once, streamed for ~15 s. Two
    things can invalidate it mid-stream and neither is currently measured —
    detection noise, and the fact that a world-fixed cube MOVES IN THE BASE
    FRAME when the torso leans (the detection reaches the base frame through
    gimbal FK on that same leaning torso, so the lean shows up as apparent cube
    motion). Geometry puts the sensitivity near 1.1 mm per 0.001 rad of lean for
    a cube at r 0.4 m with the pelvis ~0.7 m up — i.e. 2 deg would eat most of
    GRASP_EE_OK_M (60 mm) and 3 deg would trip CUBE_DRIFT_ABORT_M (50 mm). That
    is an estimate from the kinematics, NOT a measurement, and the tool already
    computes the drift every tick for the abort check — it just threw it away.

    So: keep it, and report it. Used two ways, and the pair is what makes it
    readable — the standing sample is the noise floor, the streaming sample is
    noise PLUS motion, and the difference is what a live tracker would have to
    correct for.
    """

    def __init__(self, label: str, reference: np.ndarray | None):
        self.label = label
        self.ref = None if reference is None else np.asarray(reference, float).copy()
        self.drift: list[float] = []
        self.tilt: list[float] = []
        self.period_ms: list[float] = []
        self.inliers: list[int] = []
        self.goal_travel: list[float] = []
        self._last_t: float | None = None
        # 08-12 (user): the slow base rotation during fetches needed NUMBERS.
        # Gyro yaw rate is sampled from the same telemetry every tick and
        # integrated over the watch window; the report prints both. Positive
        # wz = counter-clockwise (left); the observed clockwise grind shows
        # as a NEGATIVE yaw drift.
        self.wz: list[float] = []
        self.yaw_rad = 0.0

    def sample(self, tel: dict, live: np.ndarray | None, now: float,
               inliers: int | None = None) -> None:
        """One tick of evidence: loop period, torso tilt and gyro yaw (integrated)
        from `tel`; |live - ref| as cube drift and the backing tag-face count
        when a sighting exists. Missing pieces are skipped, never zero-filled."""
        ang = tel.get("base_ang_vel")
        if ang is not None and len(ang) >= 3 and np.isfinite(ang[2]):
            self.wz.append(float(ang[2]))
            if self._last_t is not None:
                self.yaw_rad += float(ang[2]) * (now - self._last_t)
        if self._last_t is not None:
            self.period_ms.append((now - self._last_t) * 1e3)
        self._last_t = now
        t = _tilt_deg(tel)
        if t is not None:
            self.tilt.append(t)
        if live is not None and inliers is not None:
            # Guarded together on purpose: appending `inliers` unconditionally
            # while drift is appended only when a sighting exists made the face
            # histogram report high counts for cubes nobody could see (audit
            # 2026-08-06).
            self.inliers.append(int(inliers))
        if live is not None and self.ref is not None:
            self.drift.append(float(np.linalg.norm(np.asarray(live, float) - self.ref)))

    def sample_goal(self, goal: np.ndarray | None,
                    anchor: np.ndarray | None) -> None:
        """How far the GOAL itself has travelled from its plan-time anchor.

        THE NUMBER WHOSE ABSENCE MADE A WHOLE NIGHT UNRECONCILABLE. `worst` is
        |hand - goal| and BOTH ends move, but nothing logged the goal end, so a
        T13 reading "hand stuck 178 mm" could not be checked against a handoff
        deficit of 72 mm. The `cube drift` statistic above cannot stand in for
        it: its `live` is rig.latest() — the raw retained pose, re-sampled for
        DET_LOST_RESCAN_S after a cube goes dark — and its `ref` is a seed that
        is never tilt-rotated while goal_pos is rotated every tick, so the two
        are not even in one frame.

        goal_pos and plan_anchor ARE in one frame (the same rotation is applied
        to both every tick), so this difference is honest, and it is sampled
        unconditionally — a frozen goal that the tilt compensation is walking
        away from is exactly the case that needs reporting.
        """
        if goal is None or anchor is None:
            return
        self.goal_travel.append(
            float(np.linalg.norm(np.asarray(goal, float)
                                 - np.asarray(anchor, float))))

    def report(self) -> None:
        """Print the window's summary on one line: cube drift (max/p50/last),
        goal travel from its plan anchor, tag-face histogram against the
        DRIFT_MIN_INLIERS bar, tilt span, base yaw rate + integrated yaw drift,
        and loop period p50/p90/max against PUB_DT. Sections with no samples
        are omitted (drift prints "n/a")."""
        parts = [f"[watch] {self.label}:"]
        if self.drift:
            d = np.asarray(self.drift)
            parts.append(f"cube drift max {d.max()*1000:.1f} mm "
                         f"(p50 {np.percentile(d, 50)*1000:.1f}, "
                         f"last {d[-1]*1000:.1f}, n={d.size})")
        else:
            parts.append("cube drift n/a (no live sighting)")
        if self.goal_travel:
            g = np.asarray(self.goal_travel)
            parts.append(f"GOAL travel from its plan anchor max "
                         f"{g.max()*1000:.1f} mm (p50 {np.percentile(g,50)*1000:.1f}, "
                         f"last {g[-1]*1000:.1f}, leash "
                         f"{MPC_LOWEV_LEASH_M*1000:.0f})")
        if self.inliers:
            # The number that says WHY the drift did what it did. A run whose
            # tag faces collapse as the hand arrives is being occluded by its
            # own gripper, not watching a cube move.
            n = np.asarray(self.inliers)
            parts.append(f"tag faces p50 {np.percentile(n, 50):.0f} "
                         f"(min {n.min()}, "
                         f"{100.0*float((n < DRIFT_MIN_INLIERS).mean()):.0f}% "
                         f"below the {DRIFT_MIN_INLIERS}-face bar)")
        if self.tilt:
            t = np.asarray(self.tilt)
            parts.append(f"tilt {t.min():.2f}-{t.max():.2f} deg "
                         f"(span {t.max()-t.min():.2f})")
        if self.wz:
            w = np.asarray(self.wz)
            parts.append(f"base wz p50 {np.percentile(np.abs(w), 50):.3f} "
                         f"max {w[np.abs(w).argmax()]:+.3f} rad/s, "
                         f"yaw drift {np.degrees(self.yaw_rad):+.1f} deg "
                         f"over the window")
        if self.period_ms:
            p = np.asarray(self.period_ms)
            parts.append(f"loop p50/p90/max {np.percentile(p,50):.0f}/"
                         f"{np.percentile(p,90):.0f}/{p.max():.0f} ms "
                         f"(budget {PUB_DT*1e3:.0f})")
        print("  ".join(parts))


class StaticPoseTrack:
    """Windowed 6-DoF pose of a body that does not move — a table.

    WHY NOT CubeTrack. The operator's CubeTrack is position-only and carries
    machinery a table has no use for (per-camera claims, gaze arbitration,
    not-servable latches). A table needs the opposite: the ORIENTATION, because
    a 1.2 x 0.6 m slab yawed 30 deg has corners a third of a metre away from
    where an axis-aligned box would put them, and it needs no arbitration at
    all because nothing competes to look at it.

    Orientation is averaged with hemisphere alignment (q and -q are the same
    rotation; a naive mean of mixed signs collapses toward zero), and the worst
    angular deviation inside the window is reported so a mis-fit face — the
    failure mode that a wrong tag_roll convention or a single bad tag would
    produce — shows up as a number instead of as a quietly rotated obstacle.
    """

    def __init__(self, key: str, window_s: float = 2.0):
        self.key = key
        self.window_s = window_s
        self.hist: list[tuple[float, np.ndarray, np.ndarray]] = []

    def ingest(self, pos, quat_wxyz, now: float) -> None:
        p = np.asarray(pos, dtype=float)
        q = np.asarray(quat_wxyz, dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(p)) \
                or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
            return                              # NaN drop, same rule as the cubes
        self.hist.append((now, p, q / np.linalg.norm(q)))
        self.hist = [s for s in self.hist if now - s[0] <= self.window_s]

    def pose(self, now: float,
             min_t: float = -1e18) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(pos, quat_wxyz, worst_angle_deg) or None when not confirmed.

        `min_t` fences out samples captured before the base last moved
        (2026-08-10 night): table sightings are BASE-FRAME, so a sample from
        mid-walk describes the slab relative to a robot position that no
        longer exists — the same staleness class the cube stabilizer's
        base-motion fence closes. Callers with a static base pass nothing.

        Position is a per-axis MEDIAN and orientation is the MEDOID — the sample
        whose total angular distance to the others is smallest. Both are chosen
        for the same reason: a minority of bad fits must not move the answer.

        The first version averaged the quaternions, and that was wrong for
        exactly the data it was meant to handle: hardware 2026-07-30 produced a
        window spreading 78.5 deg, where a mean is dragged by whichever samples
        are worst. It also aligned the q/-q hemispheres against the LAST sample,
        so if that one was the outlier the whole set was normalised against
        garbage. The medoid needs no reference and no threshold: it can only ever
        return a pose the sensor actually reported.
        """
        pts = [s for s in self.hist
               if now - s[0] <= self.window_s and s[0] > min_t]
        if len(pts) < 3:
            return None
        pos = np.median(np.stack([p for _, p, _ in pts]), axis=0)
        quats = np.stack([q for _, _, q in pts])
        # |dot| makes this blind to the q/-q double cover, so no hemisphere
        # alignment step is needed at all.
        dots = np.clip(np.abs(quats @ quats.T), 0.0, 1.0)
        angles = 2.0 * np.arccos(dots)                    # (n, n) radians
        quat = quats[int(np.argmin(angles.sum(axis=1)))]
        worst = float(np.degrees(angles[int(np.argmin(angles.sum(axis=1)))].max()))
        return pos, quat.copy(), worst


def table_cuboid(cfg, pos, quat_wxyz) -> dict:
    """A registered table's slab as a cuRobo cuboid in the base frame.

    slab_cuboid() is authored in the TABLE BODY frame, whose origin sits on
    the TOP SURFACE — since perception (historically visual_servoing) 8bb9628
    (08-11 re-frame) with +z
    pointing UP and the slab at z in [-0.030, 0], so the box centre is half a
    thickness along the body's own MINUS z, not at the detected origin. This
    function never assumes which way that is: the offset comes from the
    registration and is rotated as `t + R @ p_body`, correct for any frame
    convention and any table orientation, including a tilted one. Skipping
    that rotation is how you get an obstacle 15 mm out of place with
    everything else looking right.

    Dims stay in body-axis order because the returned quaternion is the body's:
    cuRobo reads `dims` along the cuboid's own frame.
    """
    slab = cfg.slab_cuboid()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, dtype=float))
    centre = np.asarray(pos, dtype=float) + R.reshape(3, 3) @ np.asarray(
        slab["pos"], dtype=float)
    return {"dims": [float(v) for v in slab["dims"]],
            "pose": [float(centre[0]), float(centre[1]), float(centre[2])]
                    + [float(v) for v in quat_wxyz]}


TABLE_Z_PRIOR_CAP_M = 0.06        # a vision z farther than this from "resting
                                  # on the slab" means the cube probably is NOT
                                  # resting there (stacked, held, mis-associated
                                  # table) — keep the measured height
TABLE_Z_PRIOR_XY_SLACK_M = 0.05   # footprint tolerance: tag-fit xy error must
                                  # not disqualify a cube sitting near an edge


def table_resting_z(cube_xy, table_pos, table_quat_wxyz, table_dims):
    """z of the table's TOP surface at the cube's xy, or None if the cube is
    outside the slab footprint (+ slack) or the fit is unusable for a prior.

    08-12 night (user: "still grasping too high"): front grasps run on 1-face
    cube fits whose VERTICAL is worth tens of mm either way — measured tonight
    as centers reading 12-25 mm off the resting height, which grasps high/
    empty AND pins the tip-guard floor above low targets (F_windup storms).
    A cube on a table obeys physics: center = surface + half its size. The
    surface is evaluated at the cube's own xy on the fitted top PLANE (not the
    slab-center z), so a slightly tilted fit stays honest across a 1.2 m slab.
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(table_quat_wxyz, dtype=float))
    R = R.reshape(3, 3)
    p = np.asarray(table_pos, dtype=float)
    d = R.T @ np.array([cube_xy[0] - p[0], cube_xy[1] - p[1], 0.0])
    if abs(d[0]) > table_dims[0] / 2 + TABLE_Z_PRIOR_XY_SLACK_M or \
            abs(d[1]) > table_dims[1] / 2 + TABLE_Z_PRIOR_XY_SLACK_M:
        return None
    n = R @ np.array([0.0, 0.0, 1.0])
    if abs(n[2]) < 0.5:
        # a slab fitted anywhere near vertical is junk for a height prior
        return None
    return float(p[2] - (n[0] * (cube_xy[0] - p[0])
                         + n[1] * (cube_xy[1] - p[1])) / n[2])


@dataclasses.dataclass
class Args:
    """cuRobo reach-and-grasp mission driver: plan on the GPU server, gate on our
    model, stream to the robot — dry-run by default, --execute moves the arms;
    exactly one of --tables / --table / --no-table-world is required."""
    robot_ip: str = "127.0.0.1"
    server: str = _site.PLAN_SERVER
    """cuRobo plan server endpoint. Runs on a CUDA machine, which is normally
    NOT the robot's onboard computer. Start control/curobo_plan_server.py there
    and verify with humanoid_plan_server_probe.py; see docs/OPERATIONS.md section 1."""
    cubes: tuple[str, ...] = ("grasp_cube_40mm",)
    """The GRASP SET — one arm per cube, so one or two names. Every one of them
    must be confirmed AND inside the reach envelope before anything is planned:
    a two-cube mission waits for both rather than starting on whichever appeared
    first, because half a dual grasp is not a state this tool models. They are
    all obstacles too (the upstream contract, targets included — cuRobo's goalset
    handles the approach). Which hand takes which is the server's choice, with
    both pairings tried explicitly if its pick fails our gates."""
    detection_url: str = OP.DETECTION_IPC_URL
    tables: tuple[str, ...] = ()
    """Registered table body names (e.g. lab_table_a lab_table_b). Pose comes
    from their AprilTags — full 6-DoF, not just position — and the slab
    geometry from the registration itself, so nothing is retyped here. Add the
    same names to the monitor's --bodies or they are never detected."""
    table: tuple[float, ...] = ()
    """cx cy cz sx sy sz — ONE axis-aligned table cuboid, base frame, metres.
    The manual escape hatch for a table with no tags (or tags out of view);
    prefer --tables, which reads the registered geometry and the real
    orientation."""
    no_table_world: bool = False
    """Declare the reach volume genuinely obstacle-free (bare-bench parity
    with the validated operator grasp). Exactly one of --tables / --table /
    --no-table-world is REQUIRED: an empty world must be an explicit
    statement, never a default."""
    anchor_trust: bool = True
    """08-12 night (user: "when the cuRobo waypoint is generated it should be
    regarded as reachable"): the tracking phase's SOFT aborts stop meaning
    "retreat" and start meaning "return the goal to the six-gate-certified
    plan anchor and close there". First offense of T07 (envelope strikes),
    T12/T13 (stalls beyond GRASP_EE_OK), or sustained blindness far off the
    anchor engages the anchor lock: live evidence is ignored, the goal slews
    home to the anchor, and the existing converge/blind/z-rail machinery
    closes on it. A wrong anchor costs one certified-EMPTY pinch (~4 s,
    auto-reopen, round retry) instead of a 20-30 s trail retreat. A SECOND
    conviction while already locked aborts for real — the bound that keeps a
    physically stuck arm from looping. Hard faults (ARM FAULT, stale
    telemetry, T14 budget) retreat exactly as before. --no-anchor-trust
    restores the old trigger-happy behaviour."""
    no_table_z_prior: bool = False
    """08-12 night (user: "still grasping too high"): by default, a cube whose
    xy sits over a SEEN, sane table gets its grasp z from PHYSICS — table
    surface at the cube's xy + half the cube — instead of the vision vertical.
    Front grasps run on 1-face fits whose z error is tens of mm in either
    direction; it poisons the anchor (grasps high/empty) AND the tip-guard
    floor (floor lands above a low target -> F_windup / INFEASIBLE storms).
    The prior applies only within TABLE_Z_PRIOR_CAP_M of the vision z, so a
    cube genuinely off the table keeps its measured height. This flag turns
    the prior OFF."""
    chest_home: bool = False
    """Return both arms to the chest home before planning. Since 07-31 the
    chest home IS the power-on posture (the simulator's home), so on a fresh
    boot this measures zero distance and returns at once — it exists for the
    session whose arms were left elsewhere. This is the tool's only Cartesian
    moment — the pose is published as (CHEST_HOME_EE, CHEST_HOME_QUAT) and the
    robot-side IK finds the configuration, exactly as the operator does. CAN
    MOVE THE ARMS even in a dry-run, which is why it is opt-in."""
    execute: bool = False
    """Default is dry-run: full pipeline, zero motion. --execute streams."""
    journey: bool = False
    """Walk to each cube in turn instead of standing still: survey once, then
    one LEG per cube in near-first order (tuck -> walk -> arrive -> measured
    stillness -> re-acquire -> cuRobo reach/grasp/retract). The three gates are
    the operator's own, imported and run verbatim; the steering law is restated
    from its constants and pinned to it by a differential test.

    A leg that fails is SKIPPED LOUDLY and the mission STOPS there (user
    2026-08-08) — it may not retry one cube forever either. See run_leg for
    the bound on every wait, and journey_continue_on_skip for the old
    walk-on salvage behaviour."""
    journey_continue_on_skip: bool = False
    """After a failed leg, walk on and attempt the remaining cubes instead of
    stopping. The pre-08-08 default: it salvages what it can from a multi-cube
    mission, at the price of the robot visibly abandoning an ungrasped cube
    and burying the failure a leg further away. User 2026-08-08, watching leg
    1 fail to plan and the robot march off to leg 2: "if the 1st one is not
    grasped, the humanoid should not move to the next one" — so stopping at
    the failed cube is now the default; the mission ends standing where the
    evidence is, holding whatever earlier legs grasped. Either way a skipped
    leg is never retried in-mission: only a restart retries a cube."""
    grasp: bool = True
    other_arm_fallback: bool = True
    """When no route from the auto assignment clears our gates, ask cuRobo for
    each arm explicitly before giving up. A cube near the midline can be
    unreachable-with-margin for one arm and easy for the other."""
    max_rate: float = 1.0125
    """Joint-rate ceiling for streaming [rad/s]. 1.0125 = 0.675 x1.5 (user
    2026-08-02, third speed bump). The receiver's hard cap was raised
    0.7 -> 1.05 in the same change so this is not silently clipped; the
    StepGate rate ceiling (1.05) stays the outer bound with the same
    3.7% headroom. Every consumer scales with this: route stretch, MPC
    clamp+lead (lead cap 3 x 1.0125 x 0.05 = 0.152 rad, still under
    MPC_CMD_SANE_RAD; shoulder kd braking at full rate costs 4.05 N*m vs
    kp x lead = 6.08 authority), retreat and glide rates; the
    stream-follow bar is rate-aware (see stream_route)."""
    decoupled_arms: bool = False
    """Solve each arm's route as its OWN single-target cuRobo problem, merge
    the two trajectories, and gate the MERGED route with the full local gate
    suite (the C sweep on the merged 14-joint trajectory replaces the joint
    solver's internal mutual avoidance). User 2026-08-04: some placements
    (front-left + rear-right) deadlock the coupled solve's cost landscape;
    decoupling removes the trade-off while KEEPING simultaneous execution —
    both arms stream, grasp and retract together, exactly as before. Falls
    back to the joint solve when the merged route cannot be gated (arms
    genuinely conflicting)."""
    reach_hover_mm: float = 15.0
    """Route ENDPOINT height above each grasp point [mm] (user 2026-08-03:
    'end every route 3 cm above the cube, then descend and grasp'; trimmed
    to 15 mm the same day — the landing gate on convergence is what keeps a
    hover smaller than MPC_CONVERGED_M from closing early). The
    offset is applied ONLY to the positions handed to the solver, so the
    streamed route delivers the hand to a HOVER directly above the cube;
    the MPC tracker — whose goals, watches and leash all keep the TRUE
    grasp point — then closes the last 3 cm as a short, near-vertical,
    live-corrected descent. Requires the MPC (without it the gripper would
    close at the hover); 0 restores direct-to-grasp routes."""
    clearance_floor_mm: float = 10.0
    mpc: bool = True
    """Real-time terminal tracking. The gated route streams to
    MPC_HANDOFF_FRACTION, then a cuRobo MPC session tracks each hand onto the
    LIVE median of its own cube at ~20 Hz until it converges, with every
    commanded tick passing the per-tick StepGate (limits / rate / segment
    clearance on OUR model). This is the answer to the frozen-green-line
    problem: the planned endpoint stops being a snapshot. --no-mpc restores
    the pure open-loop route (the 07-30-validated behaviour)."""
    fresh_scan: bool = False
    """Forget every cube position after the cameras return to the origin, so
    the scan actually runs.

    Off by default because keeping them is the fast iteration loop: leave the
    monitor up, relaunch only this tool, and the eyes go straight back to work.
    That happens because zero_gimbals polls detections in the same loop that
    drives the gimbals home, so locks land before the scan phase is even
    reached.

    Turn it ON to DEMONSTRATE the pipeline. Without it the origin is not a real
    reset — the cameras come back having kept everything they knew, the scan
    banner prints and nothing sweeps, and see->scan->fix->grasp can only be
    shown by tearing down all five processes."""


# ------------------------------------------------------------------------------
# Telemetry / detection helpers
# ------------------------------------------------------------------------------

def _tilt_deg(tel: dict) -> float | None:
    g = np.asarray(tel.get("projected_gravity", ()), dtype=float)
    if g.size != 3 or not np.all(np.isfinite(g)) or np.linalg.norm(g) < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(-g[2] / np.linalg.norm(g), -1.0, 1.0))))


def _arm_enc(tel: dict) -> dict[str, float] | None:
    jp = np.asarray(tel.get("joint_pos", ()), dtype=float)
    if jp.size < 31 or not np.all(np.isfinite(jp[13:27])):
        return None
    out = {}
    for i, name in enumerate(ARM_JOINTS_L):
        out[name] = float(jp[13 + i])
    for i, name in enumerate(ARM_JOINTS_R):
        out[name] = float(jp[20 + i])
    return out


class TelemetryView:
    """Latest-packet view of real_env telemetry (tcp://<robot>:9870) with a
    staleness clock: fresh() returns the newest packet, or None once nothing new
    has arrived for TEL_STALE_S — so every caller fails closed on a dead link."""

    def __init__(self, robot_ip: str):
        self._sub = NNGSubscriber(f"tcp://{robot_ip}:9870")
        self._sub.start()
        self._last_id, self._last_t = None, -1e9

    def fresh(self) -> dict | None:
        if self._sub.data_id != self._last_id:
            self._last_id = self._sub.data_id
            self._last_t = time.monotonic()
        if time.monotonic() - self._last_t > TEL_STALE_S:
            return None                      # stale => every caller fails closed
        return self._sub.data or None

    def stop(self):
        self._sub.stop()


class GazeRig:
    """The operator's dual-camera gaze/perception, MIGRATED not rewritten.

    This shim duck-types exactly the AutoOperator fields that
    auto_operator.gaze and auto_operator.perception touch; the module-level
    constant re-exports above satisfy their live _K(op) resolution. The code
    that runs is the operator's own, verbatim: per-cube camera CLAIMS with
    expiry and NaN drops (perception.ingest), the panoramic SERPENTINE scan
    (half-turn lead-in, four alternating 360-deg rows), parallax-corrected
    aim with the ±pi-cut branch unwrap, sender-side slew, park-when-done.
    CubeTrack is the operator's too, so confirmed()/fresh()/hist median
    semantics are identical to a mission run."""

    def __init__(self, cube_keys, table_keys=()):
        self.fk = GimbalCameraFK({p: CAM_SITES[p] for p in CAM_PORTS})
        self.cubes = {k: CubeTrack(k) for k in cube_keys}
        # Deeper sighting history than the operator's default (maxlen=5): the
        # low-evidence stable median needs LOWEV_STABLE_WINDOW_S of samples to
        # exist at all. Swapped per instance so the operator's own missions
        # keep their class default untouched.
        for t in self.cubes.values():
            t.hist = deque(t.hist, maxlen=LOWEV_HIST_N)
        # Sightings are BASE-FRAME positions, valid only while the base has
        # not moved. The journey's walk and stillness loops stamp this every
        # tick they drive (or are not yet still); the stable window admits
        # only samples captured after it, so a fix measured from a robot
        # position that no longer exists can never outrank the fresh stream
        # (hardware 2026-08-10 night: a mid-walk 2-face decode won the wide
        # window and aimed the grasp very far off the cube).
        self._base_moved_t = -1e9
        # Tables live OUTSIDE self.cubes on purpose: everything the operator's
        # perception does to a member of self.cubes (claims, gaze scheduling,
        # servable latches) is wrong for a static obstacle, and a table in the
        # cube dict would also compete for a camera claim against the cube we
        # actually want to grasp.
        self.tables = {k: StaticPoseTrack(k) for k in table_keys}
        self.inliers: dict[str, int] = {}   # tag faces behind the last sighting
        # SCAN HOLD (08-12): while True, an eye with no claimed-fresh cube
        # PARKS instead of running the serpentine (see auto_operator/gaze.py
        # eye()). The journey engages it from target lock to the next walk —
        # a sweeping free eye during the fetch drags the base into a slow
        # rotation through the checkpoints' trained gaze-follow coupling.
        self.scan_hold = False
        # YAW INTEGRAL (08-12 late night, user: "the body tilts, so the
        # base-frame cube position is no longer where the cube is"). The gyro
        # z integral, accumulated on every tick() from telemetry — the
        # anchor corrector's clock. build_scene stamps _anchor_yaw_mark when
        # targets are measured; mpc_track rotates its anchors by the delta so
        # a blind close aims at where the cube IS in the current base frame,
        # not where it was at fix time. Drift over a 10-20 s fetch is a few
        # tenths of a degree — far under the 5-14 deg it corrects.
        self.yaw_integral = 0.0
        self._yaw_t: float | None = None
        self._anchor_yaw_mark: float | None = None   # None until build_scene
                                                     # stamps a target epoch
        # --- fields the OPERATOR's journey gates read on `op` ------------------
        # journey_survey / journey_base_still / journey_arms_tucked are imported
        # and run VERBATIM (same rule as ao_gaze / ao_perception): duck-type the
        # fields rather than reimplement the gate, so a fix to the operator's
        # stillness or tuck logic reaches this tool without being ported twice.
        self._survey_order: list[str] | None = None
        self._survey_warn_t = -1e9
        self._base_vel = None                # measured base linear velocity
        self._base_ang = None                # measured base angular velocity
        self._journey_lin_seen = False       # v159b publishes hard zeros; the
        self._journey_blind_warned = False   # gate falls back to gyro + a fixed
        self._journey_still_since = None     # settle when it never sees nonzero
        self._jpos = None                    # journey_arms_tucked reads joints
        self._ee_act: dict[str, object] = {"left": None, "right": None}
        self._held: dict[str, dict] = {}     # arm -> {} once it carries a cube
        self.gaze = np.zeros(4)
        self._gaze_seeded = False
        self._scan_phase = 0.0
        self._cam_retired: set[int] = set()
        self._trim = {p: np.zeros(2) for p in CAM_PORTS}
        self._grip: dict = {}
        self._cam_frame = {p: ao_gaze.probe_cam_frame(self, p) for p in CAM_PORTS}

    # --- the operator's own thin wrappers (verbatim delegation) -------------
    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _aim_angles(self, port, target, jpos):
        return ao_gaze.aim_angles(self, port, target, jpos)

    def _scan_angles(self, port):
        return ao_gaze.scan_angles(self, port)

    # --- mission-facing surface ---------------------------------------------
    def ingest(self, targets: dict, now: float) -> None:
        """Feed one monitor packet's `targets` through the operator's own
        perception.ingest (claims, NaN drops, CubeTrack histories), then record
        what the operator discards: the tag-face count behind each cube sighting
        and the 6-DoF pose of each declared table."""
        ao_perception.ingest(self, targets, now)
        # HOW MANY TAG FACES BACKED THE SIGHTING WE JUST ACCEPTED. The operator's
        # ingest takes any cube detection with n_inliers >= 1 and stores only the
        # position, so downstream code cannot tell a three-face fit from a
        # one-face guess. That is fine for gaze — aiming a camera 2 cm wrong
        # costs nothing — but the drift abort is a control decision, and this
        # codebase already draws the line for those: SERVO_MIN_INLIERS = 2,
        # "single-tag PnP poses are garbage-prone and this feeds a control
        # loop". Kept HERE rather than on CubeTrack so the operator's shared
        # tuple layout is untouched.
        for key in self.cubes:
            t = targets.get(key)
            if isinstance(t, dict):
                self.inliers[key] = int(t.get("n_inliers", 0) or 0)
        for key, track in self.tables.items():
            t = targets.get(key)
            if isinstance(t, dict):
                track.ingest(t.get("pos"), t.get("quat_wxyz"), now)

    def last_inliers(self, key: str) -> int:
        return int(self.inliers.get(key, 0))

    def table_pose(self, key: str, now: float):
        track = self.tables.get(key)
        return None if track is None \
            else track.pose(now, min_t=self._base_moved_t)

    def tick(self, tel: dict, packet: dict | None = None) -> dict | None:
        """Run one operator gaze tick and merge gaze_targets into `packet`.
        No packet until seeded from measured gimbal encoders (operator rule:
        zeros-init + a parked gimbal = a multi-radian slam)."""
        jp = np.asarray(tel.get("joint_pos", ()), dtype=float)
        # Journey inputs, captured on EVERY tick — the operator does the same in
        # its own tick() and its gates read them as plain fields. Kept here so
        # they cannot go stale behind a gate that is only polled sometimes.
        self._base_vel = tel.get("base_lin_vel")
        self._base_ang = tel.get("base_ang_vel")
        # yaw integral: every pumped tick, before any early return below —
        # the anchor corrector must keep counting through gaze-less phases
        _now_yaw = time.monotonic()
        if self._yaw_t is not None and self._base_ang is not None \
                and len(self._base_ang) >= 3 and np.isfinite(self._base_ang[2]):
            self.yaw_integral += float(self._base_ang[2]) * \
                min(_now_yaw - self._yaw_t, 0.1)
        self._yaw_t = _now_yaw
        self._jpos = jp if jp.size >= 27 else None
        ee_t = tel.get("ee") or {}
        self._ee_act = {a: (ee_t.get(a) or {}).get("pos_actual")
                        for a in ("left", "right")}
        if jp.size < 31 or not np.all(np.isfinite(jp[27:31])):
            return packet
        if not self._gaze_seeded:
            self.gaze = jp[27:31].astype(float).copy()
            self._gaze_seeded = True
        target = ao_gaze.eye(self, jp, time.monotonic())
        self.gaze = ao_gaze.slew_gaze(self, target)
        out = packet if packet is not None else {}
        out["gaze_targets"] = [float(x) for x in self.gaze]
        return out

    def median(self, key: str, now: float) -> np.ndarray | None:
        """Confirmed-sighting median over the operator's latch window — the
        same median-of-hist the operator's reach latch uses.

        FIRST SIGHTING COUNTS (user 2026-08-02): one fresh frame is enough —
        the median is over whatever the window holds. The old >=3-frame bar
        added ~0.2-0.3 s before a cube could be selected; with first-seer
        locking the camera is already parked on the cube, so later frames
        only refine what live tracking re-reads every tick anyway.

        BEST EVIDENCE WINS (audit 2026-08-06). The window is filtered to the
        highest face count it holds before the median is taken, because these
        samples are NOT interchangeable: a 1-face 16 mm-tag PnP fit is worth
        "tens of mm" by this codebase's own measure, and mixing one into a
        two-sample window moves the answer by half of that (np.median over an
        even count is the mean). The old unfiltered median therefore let a
        single grazing 1-face frame drag the goal a hand's width — the error
        lands whole in `worst`, which is what decides a retreat.

        Deliberately scoped to THIS median. The operator's own latch windows
        (auto_operator/arms.py, mission.py, secondary.py, independent.py) read
        the same, now cleaner, history unchanged: they gate a walk-and-reach
        with its own evidence rules, not a 20 Hz terminal tracker.

        RETURNS THE EVIDENCE IT USED, and callers must gate on THAT (audit
        2026-08-06). The filter above and `rig.last_inliers` answer different
        questions: last_inliers is the face count of the NEWEST PACKET, this is
        the face count of the samples the median was actually taken over. With a
        window of [3-face at t-0.5 s, 1-face at t-0.05 s] the median is a clean
        3-face fix while last_inliers says 1 — so the descent freeze discarded
        it and printed "1-face update ignored" about a value containing zero
        1-face samples, and the leash printed "1-face median N mm off the
        anchor" about a pure 3-face one. On hardware where good frames are rare
        that discards exactly the frames that matter: a 2-face decode among
        1-face decodes was admitted for ~2 ticks instead of the full 0.7 s
        window, a 7x slowdown onto the best evidence in hand.

        `faces_used` below reports that face count; the two are computed from
        the same window so they cannot disagree.
        """
        win = self._best_window(key, now)
        if win is None:
            return None
        return np.median(np.stack([s.pos for s in win]), axis=0)

    def _best_window(self, key: str, now: float) -> list | None:
        """The samples median() takes its median over: the freshest face count
        available inside CUBE_LATCH_WINDOW_S, and only the samples that carry
        it. None when there is no live evidence at all.

        LOW-EVIDENCE STABILIZER (user 2026-08-10, the e-cube T09 run): when the
        fresh window holds ONLY 1-face samples, the window widens to
        LOWEV_STABLE_WINDOW_S so the median is the stable central position of
        the noisy stream rather than its latest slide — 1-face PnP drifted the
        fix 44-51 mm that run, dragged the MPC goal 40 mm and the re-solves
        went unstable (an insane one-tick command on the IDLE arm tripped T09).
        Best evidence still wins inside the widened window: an older >=2-face
        decode outranks the whole 1-face stream. The widening requires fresh
        samples to exist, so a cube gone dark still returns None and the
        blind-grasp machinery keeps its meaning."""
        cube = self.cubes.get(key)
        if cube is None or not cube.confirmed(now):
            return None
        win = [s for s in cube.hist if now - s.t <= CUBE_LATCH_WINDOW_S]
        if not win:
            return None
        best = max(s.faces for s in win)
        if best < DRIFT_MIN_INLIERS:
            # BASE-FRAME STALENESS FENCE (hardware 2026-08-10 night, the very
            # first run of this stabilizer): sightings are base-frame, so a
            # sample captured before the base last moved describes the cube
            # relative to a robot position that NO LONGER EXISTS — at walk
            # speed a 2 s-old fix is ~0.8 m of pure bias, and the "best
            # evidence wins" rule made a mid-walk 2-face decode outrank the
            # whole fresh post-stop stream. Only post-motion samples may
            # enter the wide window; while the base is moving it is empty
            # and the median falls back to the fresh window — the exact
            # pre-stabilizer semantics.
            wide = [s for s in cube.hist
                    if now - s.t <= LOWEV_STABLE_WINDOW_S
                    and s.t > self._base_moved_t]
            if wide:
                best = max(s.faces for s in wide)
                return [s for s in wide if s.faces >= best]
        return [s for s in win if s.faces >= best]

    def faces_used(self, key: str, now: float) -> int:
        """Tag faces behind the value median() would return right now.

        NOT `last_inliers`, which is the face count of the newest PACKET. The
        two answer different questions and the gates were asking the wrong one
        (audit 2026-08-06): with a window of [3-face at t-0.5 s, 1-face at
        t-0.05 s] the median is a clean 3-face fix while last_inliers says 1, so
        the descent freeze discarded it and printed "1-face update ignored"
        about a value containing zero 1-face samples, and the leash printed
        "1-face median N mm off the anchor" about a pure 3-face one. On hardware
        where good frames are rare that throws away exactly the frames that
        matter — a 2-face decode among 1-face decodes was admitted for ~2 ticks
        instead of the full 0.7 s window, a 7x slowdown onto the best evidence
        in hand. 0 when there is no live evidence.
        """
        win = self._best_window(key, now)
        return 0 if not win else int(win[0].faces)

    def cube_yaw(self, key: str, now: float) -> float | None:
        """The cube's YAW about base +z, folded into [0, pi/2), or None.

        2026-08-13 (user: "could we include the orientation into the cuRobo
        planner?"). Three deliberate reductions, each closing a failure mode
        this codebase has already paid for:

        YAW ONLY. The measured quaternion also carries roll and pitch, and
        those are the ill-conditioned components of a planar tag fit (the
        IPPE two-fold ambiguity lives out of plane). We already assert the
        cube rests flat whenever we take its z from the table prior — the
        same assumption says its roll and pitch are the table's, not the
        fit's. Feeding a bad tilt into a grasp goalset would aim the jaws
        off the faces in the one axis the gripper cannot recover from.

        MODULO 90 DEGREES. The grasp set is generated in the cube's frame
        and already spans its 4-fold vertical symmetry (yaws 0/90/180/270
        about cube +Z, each with a flip twin). So the only thing the solver
        is missing is the yaw's remainder, and folding removes the jump when
        the visible face changes identity between decodes.

        MEDOID, NOT MEAN, on the folded circle: the sample whose total
        angular distance to the others is smallest. The same choice
        StaticPoseTrack made for tables after a hardware window spread
        78.5 deg and dragged every mean it touched.

        EVIDENCE-GATED: >= DRIFT_MIN_INLIERS faces, the same bar the 1-face
        z snap and the goal leash use. Below it this returns None and every
        caller falls back to the identity quaternion, i.e. exactly today's
        behaviour — a wrong yaw is worse than no yaw, because it rotates the
        approach instead of merely offsetting it.
        """
        win = self._best_window(key, now)
        if not win or int(win[0].faces) < DRIFT_MIN_INLIERS:
            return None
        angles = []
        for s in win:
            q = getattr(s, "quat", None)
            if q is None:
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
            x_axis = R.reshape(3, 3)[:, 0]        # cube +x in the base frame
            if float(np.linalg.norm(x_axis[:2])) < 1e-6:
                continue                          # cube +x points straight up:
                                                  # its yaw is undefined
            angles.append(math.atan2(float(x_axis[1]), float(x_axis[0]))
                          % (math.pi / 2.0))
        if not angles:
            return None
        a = np.asarray(angles, dtype=float)
        # Circular medoid with period pi/2: distances are taken the short way
        # around the folded circle, so 1 deg and 89 deg are 2 deg apart.
        d = np.abs(a[:, None] - a[None, :]) % (math.pi / 2.0)
        d = np.minimum(d, math.pi / 2.0 - d)
        return float(a[int(np.argmin(d.sum(axis=1)))])

    def latest(self, key: str, now: float) -> np.ndarray | None:
        cube = self.cubes.get(key)
        return cube.pos if cube is not None and cube.fresh(now, DET_LOST_RESCAN_S) \
            else None

    def median_evidence(self, key: str, now: float) -> tuple[float, int] | None:
        """(spread_m, n) of the samples behind median() right now, or None.

        spread = the worst sample's distance from the median — the direct
        measure of whether the fix is an AGREEMENT or a lucky draw from a
        jumping estimate (the 08-12 'waypoint into empty space' class: a
        1-face stream mid-flip elects a median that no two frames agree on).
        Callers gate target ADOPTION on this; tracking gates stay as they
        are."""
        win = self._best_window(key, now)
        if not win:
            return None
        pts = np.stack([s.pos for s in win])
        med = np.median(pts, axis=0)
        return float(np.max(np.linalg.norm(pts - med, axis=1))), len(win)


class DetectionView:
    """Thin socket owner: hands every NEW monitor packet to the rig's operator
    ingest (claims, NaN drops, histories all happen there)."""

    def __init__(self, url: str):
        self._sub = NNGSubscriber(url)
        self._sub.start()
        self._last_id = None

    def poll(self, rig: GazeRig) -> None:
        """If a new detection packet has arrived since the last call, hand its
        `targets` to rig.ingest stamped at time.monotonic(); otherwise no-op."""
        if self._sub.data_id == self._last_id:
            return
        self._last_id = self._sub.data_id
        pkt = self._sub.data or {}
        rig.ingest(pkt.get("targets") or {}, time.monotonic())

    def stop(self):
        self._sub.stop()


# ------------------------------------------------------------------------------
# Gates and scene
# ------------------------------------------------------------------------------

def _startup_gates(tel: dict) -> str | None:
    """Operator startup contract, re-run verbatim. Returns a refusal or None."""
    tilt = _tilt_deg(tel)
    if tilt is None:
        return "no usable projected_gravity in telemetry"
    if tilt > OP.TILT_REFUSE_DEG:
        return (f"torso {tilt:.1f} deg off vertical (> {OP.TILT_REFUSE_DEG}) — "
                f"RULE #0: legs straight, torso upright")
    if tilt > OP.TILT_WARN_DEG:
        print(f"[gate] WARNING: torso {tilt:.1f} deg off vertical — margins are thin")
    why, _ = posture_gate(tel)
    return why


def posture_gate_homes() -> dict[str, list]:
    """The postures posture_gate will accept an arm parked at, by name.

    A FUNCTION so the tests can enumerate it. The 07-31 review found the
    "every accepted home is accepted" test hand-listing four while the gate
    held six — the migration's chest_home_0730 entry could be deleted with the
    whole suite still green. Reading it from here means a home cannot be added
    to the gate without the test covering it.

    chest_home == poweron since 07-31 (both are the sim home); the entry is
    kept so the printed "nearest:" name stays meaningful. chest_home_0730 is
    the MIGRATION case with a deadline: the last pre-migration session parked
    the arms at the 07-30 chest posture, 53 deg from the sim home, and without
    this entry the first post-migration run is REFUSED after the operator has
    already waited out a failsafe crawl. Retire it once no robot can still be
    sitting there.
    """
    return {"front": OP.STAGE_JOINTS["front"],
            "poweron": OP.POWERON_JOINTS,
            "poweron_v83": OP.POWERON_JOINTS_V83,
            "side_home": OP.SIDE_HOME_JOINTS,
            "chest_home": OP.CHEST_HOME_JOINTS,
            "chest_home_0730": OP.POWERON_JOINTS_LEGACY_0730}


def posture_gate(tel: dict) -> tuple[str | None, float]:
    """(refusal or None, worst distance from a known home in rad).

    The distance comes back so the caller can WAIT rather than refuse: the
    overwhelmingly common cause of failing this gate is the arm-silence failsafe
    still crawling home at 0.125 rad/s after the previous tool exited, which is a
    situation that fixes itself in under a minute.
    """
    enc = _arm_enc(tel)
    if enc is None:
        return "arm joints missing/non-finite in telemetry", float("inf")
    accepted = posture_gate_homes()
    worst, worst_arm, nearest = 0.0, "", ""
    for arm, names in (("left", ARM_JOINTS_L), ("right", ARM_JOINTS_R)):
        q = np.array([enc[n] for n in names])
        err, name = min(
            (float(np.max(np.abs(q - OP.mirror_arm(h, arm)))), n)
            for n, h in accepted.items())
        if err > worst:
            worst, worst_arm, nearest = err, arm, name
    if worst > OP.FRONT_HOME_TOL_RAD:
        # Name the nearest home and the distance: "0.42 rad from chest_home" is a
        # crawl to wait out, "1.7 rad from side_home" is an arm somewhere it
        # should not be, and the operator needs to tell those apart.
        return (f"{worst_arm} arm {math.degrees(worst):.0f} deg from every known "
                f"home (nearest: {nearest}) — reaching from an unknown posture "
                f"crosses configuration basins"), worst
    return None, worst


def wait_for_posture(tel: TelemetryView, timeout_s: float = POSTURE_WAIT_S) -> str | None:
    """Wait out the failsafe crawl instead of refusing on the first sample.

    WHY THIS WAITS RATHER THAN REFUSING. The gate itself is not relaxed — the arm
    still has to be at a known home before anything is planned, because reaching
    from an unknown posture crosses configuration basins. What changed is the
    response to the ONE cause that dominates in practice: every previous tool
    leaves the arms wherever it stopped, and real_env's arm-silence failsafe then
    crawls them to default at 0.125 rad/s. From the raised chest home that was a
    30-second journey, during which this gate refuses and the operator retypes
    the command guessing at the timing. Twice, on 2026-07-30.

    Reports whether the distance is SHRINKING, because that is the difference
    between "wait" and "go move the arm yourself": a crawl closes predictably, a
    parked arm never will.
    """
    last, first, warn_t, t0 = None, None, -1e9, time.monotonic()
    stale_since: float | None = None
    while time.monotonic() - t0 < timeout_s:
        sample = tel.fresh()
        if sample is None:
            # SAY SO. Telemetry stopping and the posture not settling look
            # identical from inside this loop — both are "keep waiting" — and on
            # 2026-07-30 that cost several minutes chasing a phantom motor fault
            # when the robot had simply been powered off. A frozen joint reading
            # is not a slow crawl, and the difference is one line of output.
            if stale_since is None:
                stale_since = time.monotonic()
            elif time.monotonic() - stale_since > TEL_STALE_S * 4:
                return (f"telemetry stopped {time.monotonic()-stale_since:.0f}s "
                        f"ago while waiting for the arms to settle — real_env is "
                        f"gone or the robot is powered off. The last posture read "
                        f"is frozen, not converging.")
            time.sleep(0.2)
            continue
        stale_since = None
        if sample is not None:
            why, err = posture_gate(sample)
            if why is None:
                return None
            now = time.monotonic()
            if now - warn_t > 5.0:
                warn_t = now
                # Rate over the whole wait, not between two samples: a
                # sample-to-sample difference is mostly encoder noise, and
                # dividing by it produced ETAs bouncing 46 s / 238 s / 390 s
                # while the arm sat still (hardware 2026-07-30). Averaging from
                # the first sample makes "stalled" look stalled.
                elapsed = now - t0
                closed = (first - err) if first is not None else 0.0
                if elapsed > 2.0 and closed > 0.02:
                    rate = closed / elapsed              # rad/s, averaged
                    trend = (f"crawling home, ~"
                             f"{max((err - OP.FRONT_HOME_TOL_RAD) / rate, 0.0):.0f}s to go")
                elif elapsed > 15.0:
                    trend = ("NOT CONVERGING — the arm has barely moved in "
                             f"{elapsed:.0f}s. The silence failsafe should crawl "
                             f"it at 0.125 rad/s; if it is not, the arm is "
                             f"faulted (check real_env for OVER CURRENT / error "
                             f"frames) or physically blocked")
                else:
                    trend = "waiting for the crawl to start"
                print(f"[gate] WAITING — {why} ({trend})")
            if first is None:
                first = err
            last = err
        time.sleep(0.2)
    why, _ = posture_gate(tel.fresh() or {})
    return why or "posture never settled"


def target_unreachable_why(pos: np.ndarray) -> str | None:
    """Why this target is outside the envelope, or None when it is inside.

    A reason rather than a bool because the operator has to ACT on it — "move it
    closer" and "it is too high" are different corrections, and being told which
    one, with the measured number, is the difference between one adjustment and
    several."""
    if not np.all(np.isfinite(pos)):
        return "position is not finite"
    r = float(np.hypot(pos[0], pos[1]))
    z = float(pos[2])
    if r < OP.TARGET_MIN_RADIUS_M:
        return (f"r_xy {r:.3f} m is inside the {OP.TARGET_MIN_RADIUS_M} m keep-out "
                f"cylinder around the trunk — move it AWAY from the robot")
    if r > OP.TARGET_MAX_RADIUS_M:
        return (f"r_xy {r:.3f} m is beyond the {OP.TARGET_MAX_RADIUS_M} m reach "
                f"envelope — move it {(r - OP.TARGET_MAX_RADIUS_M)*100:.0f} cm "
                f"CLOSER (this tool never walks)")
    if z > OP.TARGET_Z_MAX_M:
        return (f"z {z:+.3f} m is above the {OP.TARGET_Z_MAX_M} m ceiling "
                f"(~chest height) — put it LOWER")
    if z < OP.TARGET_Z_MIN_M:
        return (f"z {z:+.3f} m is below the {OP.TARGET_Z_MIN_M} m floor — put it "
                f"HIGHER")
    return None


def _cube_dims(key: str) -> float:
    for tok in key.split("_"):
        if tok.endswith("mm") and tok[:-2].isdigit():
            return int(tok[:-2]) / 1000.0
    return 0.05


def build_scene(args: Args, rig: GazeRig, target_keys,
                exclude=(), latched: dict | None = None,
                optional_tables: bool = False) -> tuple[dict, dict]:
    """(scene_cuboids, targets) in base frame, positions from the operator's
    cube tracks. Every SEEN cube is an obstacle (the upstream contract, incl. the
    targets themselves — cuRobo's goalset handles the approach); `exclude`
    drops the in-hand cubes for the retract plan.

    `target_keys` and `exclude` are COLLECTIONS since 07-30 (dual-arm): one
    entry per arm. A bare string is rejected rather than iterated, because
    iterating one would silently yield single characters and produce a scene
    with no cubes in it at all.

    `latched` is a caller-owned {table: (pos, quat, worst_deg, stamp)} cache,
    READ when a declared table is not currently visible and WRITTEN whenever one
    is. Pass the same dict to the reach and the retract.

    WHY A STATIC OBJECT GETS REMEMBERED. StaticPoseTrack's own first line is "a
    body that does not move", and its window is 2 s. The retract rebuilds the
    scene at the one moment the table is hardest to see: the arm is extended
    into the workspace and the gaze is locked on the cube. Refusing there costs
    a mission that has already succeeded — the hand is closed on the cube — and
    it refuses because our own arm blocked the view of a table that has not
    moved since we measured and certified it seconds earlier. The 07-30 run got
    through this by luck: the table tags happened to stay in frame.

    The guarantee the refusal protects is still intact. It exists so a lost
    table cannot silently become an EMPTY world. Since 08-13 (user: "THE TABLE
    IS A HEIGHT SENSOR, NOT AN OBSTACLE — IN THE PLANNER TOO", see the block
    in the loop below) a registered table NEVER enters the returned cuboids;
    what a latched pose keeps serving is the TABLE-RESTING Z PRIOR for every
    target over it, which is the opposite of pretending the table was never
    measured. (The hand z-floor reads LIVE sightings plus the target — see
    _table_surfaces / hand_floor_for_tables — not this cache.) What the
    fallback must never do is be quiet about it, so every use prints the age.

    `optional_tables` (2026-08-10 night, journey mode): a JOURNEY leg may
    legitimately stop somewhere its declared table cannot be seen — the rear
    cube's stop faces away from it — and the strict refusal would burn every
    plan round of that leg against a structural fact. With optional_tables a
    declared table that has no usable pose is SKIPPED from the world, loudly;
    the hand z-floor (target-derived) remains the only shield for that leg.
    Standing missions keep the strict contract.

    LATCH INVALIDATION ON BASE MOTION (same night): the latch is a BASE-FRAME
    pose, and the journey WALKS between legs. A slab latched at leg 1's stop
    is WRONG geometry at leg 2's stop — worse than none: when it was still a
    hard obstacle the plan got certified against a slab that was not there,
    and since 08-13 it would plant the z prior on a surface that is not there.
    Any latch entry stamped before the base last moved (rig._base_moved_t) is
    discarded.
    """
    if isinstance(target_keys, str) or isinstance(exclude, str):
        raise TypeError("target_keys/exclude are collections of cube names, "
                        "not a single name — a str would iterate as characters")
    target_keys = tuple(target_keys)
    excluded = set(exclude)
    now = time.monotonic()
    cuboids: dict = {}
    sane_tables: list = []      # (name, top_pos, quat_wxyz, dims) of every
                                # slab accepted below — feeds the z prior
    moved_t = getattr(rig, "_base_moved_t", -1e9)
    for name in args.tables:
        cfg = TABLE_CONFIGS[name]             # existence checked in _check_world
        got = rig.table_pose(name, now)
        if got is not None and latched is not None:
            latched[name] = (*got, now)
        if got is None:
            remembered = None if latched is None else latched.get(name)
            if remembered is not None and remembered[-1] <= moved_t:
                print(f"[scene] {name}'s remembered pose predates the last "
                      f"base motion — a base-frame slab from a stop that no "
                      f"longer exists is WRONG geometry (it would plant the z prior on a "
                      f"surface that is not there). Discarding "
                      f"the latch.")
                latched.pop(name, None)
                remembered = None
            if remembered is None:
                if optional_tables:
                    # Journey leg facing away from its table: proceed without
                    # the slab rather than burning the leg's plan rounds
                    # against a structural fact — but never quietly.
                    print(f"[scene] {name} declared but not seen from THIS "
                          f"stop — planning WITHOUT the slab for this leg "
                          f"(the hand z-floor is the only table shield here)")
                    continue
                # Never seen at all. THIS is the refusal --no-table-world exists
                # to make impossible: planning against a world we said had a
                # table in it and does not.
                raise TableNotSeen(name)
            *got, stamped = remembered
            got = tuple(got)
            print(f"[scene] {name} is not visible right now — using the pose "
                  f"measured {now - stamped:.1f}s ago (a table does not move, "
                  f"and the arm is in the way of its tags). It keeps serving "
                  f"the table-resting z prior; only the measurement is old.")
        pos, quat, worst_deg = got
        box = table_cuboid(cfg, pos, quat)
        c = box["pose"]
        print(f"[scene] {name}: top surface at {np.round(pos, 3)}, slab centre "
              f"{np.round(c[:3], 3)}, dims {box['dims']}, "
              f"yaw {math.degrees(math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))):+.1f} deg, "
              f"pose spread {worst_deg:.1f} deg")
        if worst_deg > TABLE_POSE_SPREAD_REFUSE_DEG:
            # A slab whose fitted orientation swings this far inside one window
            # is not a measurement, and a 1.2 x 0.6 m box placed at a junk
            # orientation is a WRONG obstacle — worse than no obstacle, because
            # the plan is then certified against geometry that is not there.
            # Hardware 2026-07-30: 78.5 deg of wobble, and the same fit put the
            # table surface 18 mm above the cube standing on it.
            if optional_tables:
                print(f"[scene] {name}'s fit wobbles {worst_deg:.1f} deg — a "
                      f"junk slab is worse than none; planning WITHOUT it "
                      f"for this leg")
                continue
            raise TablePoseUnstable(name, worst_deg)
        # THE TABLE IS A HEIGHT SENSOR, NOT AN OBSTACLE — IN THE PLANNER TOO
        # (user 08-13, extending the same order that took it out of the MPC
        # world: "only use it as the indicator of the cube's height").
        #
        # Why a modelled slab makes a resting cube nearly ungraspable: the
        # cube centre sits half a cube above the surface (30 mm for a 60 mm
        # cube), and a centred side grasp needs the LOWER finger rack's
        # collision spheres 20-30 mm BELOW that centre — i.e. level with the
        # wood — while each sphere carries a 23 mm radius. The space the
        # fingers must occupy IS the space the slab forbids, so the solver
        # is asked to satisfy two costs that address the same air and
        # answers INFEASIBLE. Measured 08-13: every leg whose stop could
        # decode the table tags went 30/30 INFEASIBLE (cube +0.0898 over a
        # surface at +0.0614), while every leg that could NOT see them
        # planned on the first attempt from the same geometry.
        #
        # What still protects the table: the hand z-floor, which since the
        # #2 convention patch is derived from THIS fitted surface evaluated
        # at the target's own xy (see _table_surfaces / hand_floor_for_
        # tables) and guards the direction that actually damages hardware —
        # downward, into the wood. `sane_tables` below is what feeds the z
        # prior, and it is deliberately still appended: the table keeps
        # doing the job it is good at (telling us how high the cube is) and
        # stops doing the one it was bad at.
        sane_tables.append((name, np.asarray(pos, float),
                            np.asarray(quat, float), box["dims"]))
        if worst_deg > TABLE_POSE_SPREAD_WARN_DEG:
            print(f"[scene] WARNING: {name}'s orientation wobbles "
                  f"{worst_deg:.1f} deg across the window (> "
                  f"{TABLE_POSE_SPREAD_WARN_DEG:.0f}) — one tag face may be "
                  f"mis-fitting, or tag_roll may be the wrong convention. "
                  f"Check the slab drawn in viser against the real table.")
    if args.table:
        cx, cy, cz, sx, sy, sz = args.table
        cuboids["table"] = {"dims": [sx, sy, sz],
                            "pose": [cx, cy, cz, 1.0, 0.0, 0.0, 0.0]}
    for key in args.cubes:
        if key in excluded:
            continue
        pos = rig.median(key, now) if key in target_keys else rig.latest(key, now)
        if pos is None:
            continue
        e = _cube_dims(key)
        # ORIENTED OBSTACLE BOX (08-13, user). The box used to be written
        # axis-aligned no matter how the cube actually sat, and the operator
        # is told to yaw cubes 30-45 deg so a second tag face decodes: a
        # 60 mm cube at 45 deg spans 85 mm across its diagonal, so the real
        # cube stuck ~12 mm out of the box the planner routed around, on the
        # two axes where the fingers approach. Yaw only, evidence-gated
        # (see GazeRig.cube_yaw); no measurement -> identity, i.e. exactly
        # the old box. The GRASP goalset's orientation is a separate seam
        # that must move on the client AND the server together — see the
        # note in mpc_track — so this is deliberately the obstacle half.
        _cy = rig.cube_yaw(key, now) if hasattr(rig, "cube_yaw") else None
        _cq = [1.0, 0.0, 0.0, 0.0] if _cy is None else \
            [math.cos(_cy / 2.0), 0.0, 0.0, math.sin(_cy / 2.0)]
        cuboids[key] = {"dims": [e, e, e + CUBE_OBSTACLE_Z_PAD_M],
                        "pose": [float(pos[0]), float(pos[1]), float(pos[2])]
                                + _cq}
    targets = {}
    for key in target_keys:
        if key in excluded:
            continue
        pos = rig.median(key, now)
        if pos is None:
            # Confirmed a moment ago in the selection loop and gone by the time
            # the scene is built. Dropping it silently would send the planner
            # ONE target and quietly turn a two-arm mission into a one-arm one,
            # which then looks like a successful run that grasped half of what
            # was asked for.
            raise TargetLost(key)
        targets[key] = pos
    # >>> TABLE-RESTING Z PRIOR (08-12 night, user: "still grasping too high").
    # The vertical of a 1-face cube fit is the worst number in this pipeline;
    # a cube whose xy sits over a SEEN, sane slab gets its grasp z from
    # physics instead. Applied HERE, the single source of the targets dict,
    # so the plan request, the MPC anchors, the close gates and the z-floor
    # cap all inherit the corrected height. The cube obstacle moves with it.
    z_prior_keys: set = set()
    if targets and sane_tables and not getattr(args, "no_table_z_prior", False):
        for key in list(targets):
            p_c = np.asarray(targets[key], float)
            half = _cube_dims(key) / 2.0
            for name, t_pos, t_quat, t_dims in sane_tables:
                z_surf = table_resting_z(p_c[:2], t_pos, t_quat, t_dims)
                if z_surf is None:
                    continue
                prior = z_surf + half
                dz = prior - float(p_c[2])
                if abs(dz) > TABLE_Z_PRIOR_CAP_M:
                    print(f"[scene] z prior: {key} vision z {p_c[2]:+.4f} is "
                          f"{abs(dz) * 1000:.0f} mm from resting on {name} "
                          f"(cap {TABLE_Z_PRIOR_CAP_M * 1000:.0f} mm) — not "
                          f"resting there; keeping the vision z")
                    continue
                targets[key] = np.array([p_c[0], p_c[1], prior])
                z_prior_keys.add(key)
                if key in cuboids:
                    cuboids[key]["pose"][2] = float(prior)
                print(f"[scene] z prior: {key} grasp z {p_c[2]:+.4f} (vision) "
                      f"-> {prior:+.4f} (resting on {name}: surface "
                      f"{z_surf:+.4f} at the cube's xy + {half * 1000:.0f} mm; "
                      f"shift {dz * 1000:+.0f} mm)")
                break
    # Which targets actually carry the prior, for mpc_track's 1-face z snap
    # (08-13, user): stashed EVERY build, so a leg whose tables were not
    # detected clears the previous leg's set — with no seen table the vision
    # z is all there is, and the snap must not fire on a stale claim.
    rig._z_prior_keys = z_prior_keys
    # <<< TABLE-RESTING Z PRIOR
    # ANCHOR-EPOCH YAW MARK (08-12 late night): these targets are base-frame
    # positions measured NOW. Stamp the yaw integral so mpc_track can rotate
    # its anchors by however much the base turns between this measurement and
    # the (possibly blind) close.
    if targets and hasattr(rig, "yaw_integral"):
        rig._anchor_yaw_mark = float(rig.yaw_integral)
    return cuboids, targets


# ------------------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------------------

def _slot(bundle, q: np.ndarray, side: str, rate: float) -> dict:
    names = ARM_JOINTS_L if side == "left" else ARM_JOINTS_R
    col = {n: i for i, n in enumerate(bundle.joint_names)}
    return {"joint_pos": [float(q[col[n]]) for n in names], "rate": float(rate)}


def hold_packet(enc: dict[str, float], rate: float) -> dict:
    """A JOINT-mode "stay exactly where you are" for BOTH arms.

    Snapshot semantics on purpose: the caller freezes one encoder reading and
    republishes THAT, rather than re-reading each tick. Tracking the live
    measurement would let gravity sag walk the arm downhill one packet at a
    time, with every packet individually looking like a no-op.
    """
    return {"arm_targets": {
        "left": {"joint_pos": [float(enc[n]) for n in ARM_JOINTS_L],
                 "rate": float(rate)},
        "right": {"joint_pos": [float(enc[n]) for n in ARM_JOINTS_R],
                  "rate": float(rate)}}}


def pump_during(work, pub, tel: TelemetryView, det: DetectionView, rig: GazeRig,
                hold_enc: dict[str, float] | None, rate: float, label: str):
    """Run `work()` on a worker thread while KEEPING THE 50 Hz LOOP ALIVE.

    WHY THIS EXISTS — hardware 2026-07-30, and it is three failures with one
    cause. plan_until_gated blocks on the plan server. Every solve until that
    day took 0.13-0.14 s so nobody noticed; that run took 4.91 s, and for the
    whole 4.91 s this tool published NOTHING. Consequences, in order of how bad
    they are:

    1. real_env's arm-silence failsafe fires at _ARM_CMD_TIMEOUT = 0.5 s and
       then ramps the arms toward the default pose at _ARM_RETURN_RATE =
       0.025 rad/s. 4.41 s past the timeout is 0.11 rad — 6.3 deg — of crawl on
       every arm joint, while the tool believed the arms were parked.
    2. Gate B certifies "route[0] == measured to 0.05 rad", but its measurement
       is read BEFORE the solve. 0.11 rad of crawl is twice that tolerance, so
       any solve slower than ~2.5 s VOIDS the certificate it just printed.
       B_start read 1.2e-07 rad in that run and meant nothing.
    3. No detections are ingested while blocked, so the plan target — a median
       over the cube history — is stale by the length of the solve, and the
       drift abort then compares a fresh sighting against it. The run aborted on
       "cube moved 5.1 cm" with a p50 of 46.4 mm from the very first streaming
       sample, i.e. before the arm could have touched anything.

    So the fix is not a bigger drift threshold or a longer timeout. It is to
    stop going silent: hold the arms at their measured posture, keep the gaze
    stream flowing, and keep ingesting detections, for exactly as long as the
    solver needs. Retries multiply the exposure — five attempts per arm at
    4.9 s is 25 s of silence — which is why this wraps plan_until_gated as a
    whole rather than one plan() call.

    Exceptions from the worker are re-raised in the caller's thread so
    PlanServerError still reaches the same handler it always did.
    """
    result: list = []
    failure: list = []

    def run() -> None:
        try:
            result.append(work())
        except BaseException as exc:                              # noqa: BLE001
            failure.append(exc)

    worker = threading.Thread(target=run, name="plan", daemon=True)
    t0 = time.monotonic()
    worker.start()
    noted = False
    while worker.is_alive():
        tick = time.monotonic()
        telemetry = tel.fresh()
        det.poll(rig)
        packet = hold_packet(hold_enc, rate) if hold_enc is not None else None
        out = rig.tick(telemetry or {}, packet)
        if out:
            pub.publish(out)
        if not noted and tick - t0 > SLOW_SOLVE_NOTE_S:
            noted = True
            print(f"[plan] {label} is taking > {SLOW_SOLVE_NOTE_S:.0f}s — arms "
                  f"held at their measured posture, detections still flowing")
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    worker.join()
    if failure:
        raise failure[0]
    return result[0]


def hold_until_enter(pub, tel: TelemetryView, det: DetectionView, rig: GazeRig,
                     enc: dict[str, float], rate: float) -> None:
    """Wait for Enter (or Ctrl+C / EOF) WITHOUT letting go of the arms.

    WHY THIS EXISTS — hardware 2026-07-31, and it is the same lesson as
    pump_during one layer up: a blocking call is a silent commander. This one
    was `input()`, which publishes nothing for as long as the operator reads
    the screen. Gaze does not cover for it — real_env's freshness check tracks
    ARM content only ("arm_targets"/"kp"/"kd"), so a stream of gaze_targets
    leaves the arm timer expired — and the failsafe therefore engaged 0.5 s in
    and crawled both arms toward default_pose at 0.025 rad/s for the whole
    wait. The dry run ended with the left arm 23 deg from every known home
    (16 s of crawl), and the --execute run that followed refused its own
    posture gate: "reaching from an unknown posture crosses configuration
    basins". A dry run must leave the robot exactly where it found it.

    Snapshot semantics come from hold_packet: the caller freezes ONE encoder
    reading and republishes it, so gravity sag cannot walk the arm downhill a
    packet at a time.
    """
    try:
        while True:
            tick = time.monotonic()
            telemetry = tel.fresh()
            det.poll(rig)
            out = rig.tick(telemetry or {}, hold_packet(enc, rate))
            if out:
                pub.publish(out)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()          # EOF reads '' — same as before
                return
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        return


def recheck_route_start(bundle, tel: TelemetryView) -> str | None:
    """Gate B against a FRESH encoder read, just before streaming.

    The gate inside plan_until_gated proves route[0] matches the encoders it was
    HANDED, which were sampled before a call that can take seconds. That is a
    statement about the past. Hardware 2026-07-30: a 4.91 s solve let the
    arm-silence failsafe crawl every joint 0.11 rad — twice the tolerance —
    while B_start printed 1.2e-07 rad and passed.

    pump_during now holds the arms through the solve, so this should be
    trivially green. That is the point: it is cheap, and it is the only thing
    that can tell the difference between "held" and "believed to be held".
    """
    telemetry = tel.fresh()
    if telemetry is None:
        return "telemetry went stale between planning and streaming"
    enc = _arm_enc(telemetry)
    if enc is None:
        return "no arm encoders between planning and streaming"
    col = {n: i for i, n in enumerate(bundle.joint_names)}
    worst, worst_j = 0.0, ""
    for j in bundle.controlled_joints:
        err = abs(float(bundle.q_enc[0, col[j]]) - float(enc[j]))
        if err > worst:
            worst, worst_j = err, j
    if worst > GATE_B_START_TOL_RAD:
        return (f"the arms moved {worst:.3f} rad at {worst_j} while the route "
                f"was being solved (tol {GATE_B_START_TOL_RAD}) — route[0] no "
                f"longer matches the hardware, streaming it would be a jump")
    print(f"[check] reach B_start (post-solve): OK  worst_rad={worst:.2e} "
          f"joint={worst_j}")
    return None


# ---------------------------------------------------------------------------
# RETREAT VERDICTS — which rule fired, and who tripped it
# ---------------------------------------------------------------------------
# WHY THIS EXISTS (user 2026-08-05): after the false-conviction fixes the robot
# still retreats sometimes, and a one-line sentence buried in a 20 Hz stream is
# not a diagnosis — "it retreated again" cannot be turned into a fix. Every
# abort path in stream_route/mpc_track now returns a CODED verdict, and the
# retreat prints it as a banner naming the rule AND the culprit (which arm,
# which joint, which cube, which geom pair). Codes are stable: T01-T03 are the
# streamed phase, T04-T14 the MPC track, S01-S05 the pre-track session setup.
# The numbering is the operator's own enumeration of the triggers, so a log
# line maps 1:1 onto the table they hold.
RETREAT_TRIGGERS: dict[str, tuple[str, str]] = {
    # -- streamed phase (plan-0 route playback + settle) ----------------------
    "T01": ("STREAM-STALE", "telemetry stopped arriving"),
    "T02": ("STREAM-FAULT", "the dropout watchdog latched an ARM FAULT"),
    "T03": ("STREAM-FOLLOW", "a joint fell behind its streamed target"),
    # -- MPC track, per tick --------------------------------------------------
    "T04": ("MPC-STALE", "telemetry stopped arriving"),
    "T05": ("MPC-FAULT", "the dropout watchdog latched an ARM FAULT"),
    "T06": ("MPC-ENC", "arm encoders vanished from telemetry"),
    "T07": ("MPC-ENVELOPE", "the tracked cube left the reach envelope"),
    "T08": ("MPC-STEP", "the plan server failed to step"),
    "T09": ("MPC-JUMP", "the solver commanded an insane one-tick jump"),
    "T10": ("MPC-CLEARANCE", "clearance standoff: the solver will not route "
                             "above our floor"),
    "T11": ("MPC-GATE", "the per-tick gate rejected consecutive commands"),
    "T12": ("MPC-EXEC", "an arm executed almost none of what it was asked"),
    "T13": ("MPC-NOPROG", "the worst hand stopped closing on its goal"),
    "T14": ("MPC-BUDGET", "the track ran out of time short of the cube"),
    # -- MPC session setup, before the first tracked tick ---------------------
    "S01": ("MPC-NOSIDE", "no tracked side matches the targets"),
    "S02": ("MPC-ENC0", "no arm encoders at the handoff"),
    "S03": ("MPC-SCENE", "the table scene could not be rebuilt"),
    "S04": ("MPC-START", "the MPC session failed to start"),
    "S05": ("MPC-WARM", "arm encoders vanished during the session warm"),
}

# Counted per process so the end of a run says which rule keeps costing rounds
# — three T12s and one T07 is a wiring problem, the reverse is a perception one.
RETREAT_TALLY: dict[str, int] = {}


class Abort(str):
    """A retreat verdict that names the RULE and the CULPRIT, not just the event.

    It IS the reason string — a str subclass, so every caller that prints,
    composes (`f"{why} — and the cube is gone"`) or substring-matches on it is
    unaffected — and it additionally carries the fields the banner needs. Note
    that composing one into an f-string yields a plain str: print the verdict
    from the value returned by stream_route/mpc_track, before it is wrapped.

    I/O: Abort(code, phase, culprit, detail) -> str reading
    "[T12 MPC-EXEC] <detail>". `code` must be registered in RETREAT_TRIGGERS
    (typo guard); `culprit` is free text naming the arm/joint/cube/geom pair.
    """

    def __new__(cls, code: str, phase: str, culprit: str, detail: str):
        assert code in RETREAT_TRIGGERS, f"unregistered abort code {code!r}"
        self = super().__new__(cls, f"[{code} {RETREAT_TRIGGERS[code][0]}] "
                                    f"{detail}")
        self.code = code
        self.trigger = RETREAT_TRIGGERS[code][0]
        self.rule = RETREAT_TRIGGERS[code][1]
        self.phase = phase
        self.culprit = culprit
        self.detail = detail
        return self


def _arm_of(joint: str) -> str:
    return "left" if joint.startswith("left_") else "right"


def print_retreat_verdict(why, phase_note: str = "") -> None:
    """The banner printed the instant a retreat is decided — BEFORE the arms
    start walking back, because an operator watching them walk has every reason
    to Ctrl+C before any later line appears (2026-08-04, twice).

    An uncoded reason still prints, loudly marked: that means an abort path
    exists that this table does not know about, which is itself a finding.
    """
    coded = isinstance(why, Abort)
    code = why.code if coded else "??"
    RETREAT_TALLY[code] = RETREAT_TALLY.get(code, 0) + 1
    n = RETREAT_TALLY[code]
    bar = "=" * 68
    print(f"\n[verdict] {bar}")
    if coded:
        print(f"[verdict]  RETREAT TRIGGER  {why.code}  {why.trigger}"
              f"   (#{n} this run)")
        print(f"[verdict]   rule    : {why.rule}")
        print(f"[verdict]   phase   : {why.phase}"
              + (f" — {phase_note}" if phase_note else ""))
        print(f"[verdict]   culprit : {why.culprit}")
        print(f"[verdict]   detail  : {why.detail}")
    else:
        print(f"[verdict]  RETREAT TRIGGER  UNCODED  (#{n} this run) — this "
              f"abort path is missing from RETREAT_TRIGGERS")
        print(f"[verdict]   detail  : {why}")
    print(f"[verdict]   scope   : BOTH arms retreat together — the abort is "
          f"track-level")
    print(f"[verdict] {bar}\n")


def print_retreat_tally() -> None:
    """End-of-run histogram. Prints nothing when nothing retreated."""
    if not RETREAT_TALLY:
        return
    total = sum(RETREAT_TALLY.values())
    print(f"\n[verdict] retreat tally for this run ({total} total):")
    for code, n in sorted(RETREAT_TALLY.items(), key=lambda kv: -kv[1]):
        name, rule = RETREAT_TRIGGERS.get(code, ("UNCODED", "unknown path"))
        print(f"[verdict]   {code} {name:<14} x{n}   {rule}")


def stream_route(bundle, pub, tel: TelemetryView, det: DetectionView,
                 rig: GazeRig, planned: dict | None, rate: float,
                 until_s: float | None = None,
                 drift_abort_m: float | None = None,
                 meas_hist: list | None = None,
                 settled: dict | None = None) -> tuple[str | None, float]:
    """Stream the route, then settle on the final waypoint. Returns a refusal
    string on abort (caller stops publishing => silence failsafe), else None.

    `planned` is {cube_key: the target the route was built for}. Each entry gets
    its OWN drift monitor: on a two-arm reach either cube going stale invalidates
    the route, and reporting them together would hide which one moved. Empty (the
    retract) still watches tilt and loop period, which are what say whether the
    open-loop assumption held.

    `until_s` truncates the playback for an MPC handoff: stream to that route
    time and return WITHOUT settling — the tracker takes over immediately, and
    settling on a waypoint the tracker is about to move off would waste the
    seconds the handoff exists to use.

    Returns (why, played_s): why None on success; played_s is how far into the
    route the stream got, which is exactly how far a retreat must reverse.
    The FOLLOW WATCHDOG (see STREAM_FOLLOW_LAG_RAD) aborts when the measured
    arm stops tracking the streamed targets — a wedged arm, a frozen feedback
    stream, an undertorqued joint — instead of playing the route to the end
    against a robot that is not on it.
    """
    sides = bundle.active_sides
    # DID THE SETTLE CONVERGE, OR TIME OUT? Both exits returned (None, end_s),
    # so the caller could not tell — and the blind-at-handoff shortcut then
    # closed the gripper at a posture whose joint error was unbounded, on the
    # claim that the arm was "exactly where the six-gate route put it" (audit
    # 2026-08-06, D10). It usually is; after a timeout it provably is not, and
    # every timed-out round this week left ~72 mm on the table. Written through
    # a caller-supplied dict so the (why, end_s) contract every other caller
    # relies on is untouched.
    if settled is not None:
        settled["ok"] = False
    planned = dict(planned or {})
    drift_abort = CUBE_DRIFT_ABORT_M if drift_abort_m is None else float(drift_abort_m)
    end_s = bundle.duration_s if until_s is None \
        else min(float(until_s), bundle.duration_s)
    t0 = time.monotonic()
    watches = {k: MotionWatch(f"{bundle.goal} route [{k}] (streaming)", p)
               for k, p in planned.items()} or \
              {None: MotionWatch(f"{bundle.goal} route (streaming)", None)}
    recent = {k: deque(maxlen=CUBE_DRIFT_WINDOW) for k in planned}
    drift_warned: set = set()      # one WARNING per cube per stream (see below)
    print(f"[exec] streaming {bundle.horizon} waypoints over "
          f"{bundle.duration_s:.1f}s to {'+'.join(sides)} arm(s)"
          + (f", watching {len(planned)} cube(s)" if planned else ""))
    col = {n: i for i, n in enumerate(bundle.joint_names)}
    lag_ticks = 0
    last_lag_warn = 0.0
    while True:
        tick = time.monotonic()
        t = tick - t0
        telemetry = tel.fresh()
        if telemetry is None:
            return Abort("T01", "streaming the plan-0 route", "telemetry link",
                         "telemetry went stale mid-route"), t
        fault = telemetry.get("arm_fault") or [False, False]
        if any(fault):
            who = "+".join(s for s, f in zip(("left", "right"), fault) if f)
            return Abort("T02", "streaming the plan-0 route", f"{who} arm",
                         f"ARM FAULT latched by the dropout watchdog: "
                         f"{fault} — inspect that arm's harness"), t
        # FOLLOW WATCHDOG: the streamed target is only a command; verify the
        # measured arm is actually on it. A wedged elbow, a frozen feedback
        # stream or an undertorqued joint all look identical here — the
        # divergence grows while everything else stays green — and all of
        # them mean STOP AND BACK OUT, not play the route to the end.
        enc_now = _arm_enc(telemetry)
        if enc_now is not None and meas_hist is not None and (
                not meas_hist or any(
                    abs(enc_now[n] - meas_hist[-1][n]) > RETREAT_STEP_MIN_RAD
                    for n in ARM_JOINTS_ALL)):
            meas_hist.append({n: float(enc_now[n]) for n in ARM_JOINTS_ALL})
        if enc_now is not None:
            q_now = bundle.q_at(t)
            lag_joint = max(bundle.controlled_joints,
                            key=lambda j: abs(enc_now[j] - float(q_now[col[j]])))
            lag = abs(enc_now[lag_joint] - float(q_now[col[lag_joint]]))
            # Rate-aware bar: normal following lag is a few ticks of route
            # speed plus sag, so a faster stream legitimately trails further
            # — the fixed floor alone would false-trip at max_rate 0.45.
            lag_bar = max(STREAM_FOLLOW_LAG_RAD, 4.0 * rate * PUB_DT + 0.05)
            # Between the warn line and the abort bar lies the whole 2026-08-06
            # failure mode: enough lag to wreck the handoff, not enough to
            # trip anything. Print it while it is still happening.
            if STREAM_FOLLOW_LAG_WARN_RAD <= lag <= lag_bar and \
                    tick - last_lag_warn > 2.0:
                last_lag_warn = tick
                print(f"[exec] following lag {lag:.2f} rad at {lag_joint} "
                      f"({t:.1f}s) — under the {lag_bar:.2f} abort bar, but "
                      f"this is what the tracker will inherit at the handoff")
            lag_ticks = lag_ticks + 1 if lag > lag_bar else 0
            if lag_ticks >= STREAM_FOLLOW_LAG_TICKS:
                for w in watches.values():
                    w.report()
                return Abort(
                    "T03", "streaming the plan-0 route",
                    f"{_arm_of(lag_joint)} arm / {lag_joint}",
                    f"the arm is not following the stream: {lag_joint} "
                    f"is {lag:.2f} rad behind its streamed target (bar "
                    f"{lag_bar:.2f}, {lag_ticks} straight ticks at {t:.1f}s) "
                    f"— wedged, frozen feedback, or undertorqued; backing "
                    f"out"), t
        det.poll(rig)
        if None in watches:
            watches[None].sample(telemetry, None, tick)
        for key, ref in planned.items():
            live = rig.latest(key, tick)
            faces = rig.last_inliers(key)
            watches[key].sample(telemetry, live, tick, faces)
            # ONLY SIGHTINGS WITH ENOUGH TAG FACES BEHIND THEM ARE EVIDENCE. The
            # abort fires hardest exactly when the hand arrives, because that is
            # when the gripper slides between the camera and the cube and the
            # fit collapses onto one partly-occluded face. Hardware 2026-07-30:
            # a route that had passed every gate was killed at waypoint 122 of
            # 161 — 95 % of the way there — on a "5.1 cm move" of a cube nobody
            # had touched. SERVO_MIN_INLIERS is this codebase's own answer to
            # the same question for the gripper-base servo: a single-tag PnP
            # pose is not good enough to steer on. Below that bar we simply have
            # no measurement, and no measurement must mean no verdict — never a
            # verdict of "moved".
            if live is not None and faces >= DRIFT_MIN_INLIERS:
                recent[key].append(np.asarray(live, float))
            # The abort compares MEDIANS, because the thing it is compared
            # against is one: the planned target came from GazeRig.median.
            # Gating a median with a single raw sighting meant one 1-inlier PnP
            # glitch could end a mission — precisely the failure CubeTrack.hist
            # exists to absorb.
            if len(recent[key]) == recent[key].maxlen:
                drift = float(np.linalg.norm(
                    np.median(np.stack(recent[key]), axis=0) - ref))
                if drift > drift_abort and key not in drift_warned:
                    # REPORT, DON'T VETO (user doctrine 2026-08-04: the
                    # position was locked at plan time — execute). The abort
                    # this replaces kept killing healthy routes with a
                    # DIRECTION-CORRELATED signature: the shoulder swinging
                    # through the wind-up-adjacent arc puts the forearm into
                    # the camera-cube sight line, and a partially occluded
                    # tag still decodes 2 faces — CONSISTENTLY wrong, so the
                    # 5-median window agreed with itself and "the cube
                    # moved". A truly moved cube costs nothing here: the MPC
                    # handoff re-seeds from live medians and TRACKS it (or
                    # the empty-pinch retry replans), so the stream has no
                    # reason to re-litigate perception mid-route.
                    drift_warned.add(key)
                    print(f"[watch] WARNING: {key} reads {drift*100:.1f} cm "
                          f"off the planned target (5-median, >= "
                          f"{DRIFT_MIN_INLIERS} faces) — arm-occlusion "
                          f"artifacts look exactly like this; streaming ON, "
                          f"the MPC re-seeds from live evidence at handoff")
        q = bundle.q_at(t)
        pub.publish(rig.tick(telemetry,
                    {"arm_targets": {s: _slot(bundle, q, s, rate) for s in sides}}))
        if t >= end_s:
            break
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    for w in watches.values():
        w.report()
    if end_s < bundle.duration_s:
        # THE NUMBER THE LOG NEVER CARRIED (2026-08-06). Whether the tracker
        # can succeed is decided here, by how far the arm is from the route it
        # was supposed to have followed — and until now that was discoverable
        # only six seconds later, phrased as "no progress".
        tel_h = tel.fresh()
        enc_h = _arm_enc(tel_h) if tel_h else None
        if enc_h is not None:
            q_h = bundle.q_at(end_s)
            worst_lag = sorted(
                ((abs(enc_h[j] - float(q_h[col[j]])), j)
                 for j in bundle.controlled_joints), reverse=True)[:3]
            print("[exec] handoff lag (measured vs streamed): "
                  + ", ".join(f"{j} {v:.2f}" for v, j in worst_lag)
                  + f" rad — the tracker starts from here")
        return None, end_s                 # MPC handoff: the tracker owns the rest

    # settle on the final waypoint
    q_final = bundle.q_enc[-1]
    t0 = time.monotonic()
    while True:
        tick = time.monotonic()
        telemetry = tel.fresh()
        if telemetry is None:
            return Abort("T01", "settling on the final waypoint",
                         "telemetry link",
                         "telemetry went stale during settle"), end_s
        fault = telemetry.get("arm_fault") or [False, False]
        if any(fault):
            who = "+".join(s for s, f in zip(("left", "right"), fault) if f)
            return Abort("T02", "settling on the final waypoint", f"{who} arm",
                         f"ARM FAULT during settle: {fault}"), end_s
        enc = _arm_enc(telemetry)
        det.poll(rig)
        pub.publish(rig.tick(telemetry,
                    {"arm_targets": {s: _slot(bundle, q_final, s, rate) for s in sides}}))
        if enc is not None:
            worst = max(abs(enc[j] - float(q_final[col[j]]))
                        for j in bundle.controlled_joints)
            if worst < SETTLE_TOL_RAD:
                print(f"[exec] settled (worst joint err "
                      f"{math.degrees(worst):.1f} deg)")
                if settled is not None:
                    settled["ok"] = True
                return None, end_s
        if time.monotonic() - t0 > SETTLE_TIMEOUT_S:
            print(f"[exec] WARNING: settle timeout ({SETTLE_TIMEOUT_S:.0f}s) — "
                  f"continuing from where the arm is")
            return None, end_s
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))


def _yaw_rot(dyaw: float) -> np.ndarray:
    """Rz(dyaw). The anchor corrector's workhorse (08-12 late night, user):
    when the BASE yaws by dyaw, a world-fixed point appears rotated by -dyaw
    in the base frame — so callers pass the NEGATED integrated base yaw to
    re-express an old-epoch anchor in the current frame. Composes with the
    gravity-delta tilt compensation, which cannot see yaw by construction."""
    c, s = math.cos(dyaw), math.sin(dyaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _lat_vert(a, b) -> str:
    """`|a-b|` as "N mm (lat N, z +N)".

    EVERY DISTANCE IN THIS TOOL IS A 3-D NORM, AND THAT HID A DEFECT FOR WEEKS
    (audit 2026-08-06, I2). MPC_BLIND_ANCHOR_M (40 mm) and GRASP_EE_OK_M
    (60 mm) are 3-D bars, so a hand parked on the un-descended hover rail — a
    pure vertical 15-25 mm offset that grasps the top edge of the cube instead
    of straddling its centre — passed both of them without ever being named.
    A 25 mm z error and a 25 mm xy error are different events and must not
    print identically. The z part is SIGNED: above the target is the failure
    mode, and the sign is what says so.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    return (f"{np.linalg.norm(d)*1000:.0f} mm "
            f"(lat {np.linalg.norm(d[:2])*1000:.0f}, z {d[2]*1000:+.0f})")


def _last_seen(rig, key: str, now: float) -> float:
    """When `key` was last DECODED, on the caller's clock.

    The blind-grasp banner needs the true age of the evidence, and the loop's
    own "when did I notice" counter cannot supply it: a track can begin already
    blind (hardware 2026-08-06 — the cube went dark during the settle, so the
    first tick both noticed and fired, and the banner read 0.0s).

    READS THE CAPTURE STAMP, NOT `last_seen`. CubeTrack.last_seen is stamped on
    every arriving PACKET (auto_operator/perception.py), and the monitor
    republishes each retained pose for DETECTION_RETENTION_S = 0.5 s, so it lags
    the true decode age by up to half a second. Blindness cannot even be
    declared before ~0.7 s of real darkness (GazeRig.median needs a capture
    inside CUBE_LATCH_WINDOW_S), so a banner sourced from last_seen printed
    "for 0.2s" and made a legitimate occluded grasp look like one dropped
    frame — contradicting the rule's own comment (audit 2026-08-06). The
    Sighting stamps in hist ARE capture times and are strictly increasing by
    construction, so hist[-1].t is the decode we actually mean.

    A track with no sightings at all reads as `now`, i.e. age 0 — the
    conservative direction for a print, and it cannot mislead: a cube that was
    never seen cannot reach a blind rule, because goal_pos would have no seed.
    """
    cube = getattr(rig, "cubes", {}).get(key)
    hist = getattr(cube, "hist", None) if cube is not None else None
    if hist:
        return float(hist[-1].t)
    return now


def new_grasp_action_id() -> int:
    """A NON-NEGATIVE INTEGER action id, which is what the wire actually takes.

    real_env's EEServiceClient.send_action validates every id with
    `int(action_id)` and RETURNS WITHOUT PUBLISHING on a ValueError. The warning
    it logs goes to real_env's terminal, not the commander's, so a bad id looks
    from here like a gripper that simply never answered.

    Hardware 2026-07-30: this function used to be `uuid.uuid4().hex[:12]`, the
    run sent `dd3a8235c87b`, `int()` raised, the command was dropped before it
    ever reached the service, and the reach ended in "no id-matched grasp
    verdict in 20s" after a flawless approach — all four gates green, 39.5 mm
    clearance, 1.6 deg settle. Zeroing worked in the same run because it passes
    no id at all and takes the auto-increment branch, which is exactly why the
    fault survived to the last step of the mission.

    time.time_ns() is the operator's own scheme ([:827]): monotonic, unique
    across restarts, and always parseable. Ids repeat within one grasp on
    purpose — the service dedups on last_processed_action_id, which is what
    makes EE_ACTION_REPEAT_TICKS safe.
    """
    return time.time_ns()


def do_grasp(bundle, pub, tel: TelemetryView, det: DetectionView,
             rig: GazeRig, sides, rate: float,
             hold_enc: dict[str, float] | None = None) -> str | None:
    """hand_grab on every named side, with the operator's id-matched,
    FAIL-CLOSED verdict contract. Returns a refusal on anything but a confirmed
    grasp on ALL of them; fingers are NEVER reopened on an ambiguous outcome.

    BOTH HANDS CLOSE AT ONCE, and the wire is what makes that non-trivial: a
    9874 packet carries at most ONE ee_action, so two commands cannot share a
    tick. They are interleaved rather than run in sequence — the same rule
    zero_grippers follows and for the same reason. Blocking on the left for its
    full EE_ACTION_REPEAT_TICKS window would start the right ~1.5 s later and
    leave the left un-repeated while the right runs, and 9874 is
    latest-value-wins at both hops (seen eating commands live, 07-16).

    Each side carries its OWN action id. The service tracks
    last_processed_action_id per side, so sharing one would work — but the
    verdict is correlated BY id, and two hands answering with the same id is a
    correlation that proves nothing.

    RETURNS (why, empty_sides). why is None only when every named hand closed on
    something. empty_sides names the hands that came back CERTIFIED EMPTY, and
    it is the difference between a retryable failure and a fatal one:

      * empty on every hand  -> nothing is held; the operator's class-1 contract
        says reopen and try again, without limit, while the cube stays servable.
      * empty on SOME hands  -> a cube IS in the other hand. Not retryable here:
        reopening would have to be per-side and the retract already plans for
        both, so the caller stops and holds.
      * empty_sides EMPTY but why set -> indeterminate (lost verdict, stale
        telemetry, arm fault). NEVER retryable: we do not know what the fingers
        contain, and reopening a maybe-held cube drops it. Fail closed.
    """
    sides = tuple(sides)
    empty: set[str] = set()
    action_id = {s: new_grasp_action_id() for s in sides}
    # The hold posture during the pinch. Open-loop runs hold the route's final
    # waypoint (where the arm demonstrably is, having just settled there). An
    # MPC run passes hold_enc — the arm converged onto the LIVE cube, which is
    # not the route's endpoint, and holding the endpoint here would yank the
    # hand off the cube in the very tick the fingers close.
    if hold_enc is not None:
        q_final = np.array([float(hold_enc[j]) for j in bundle.joint_names])
    else:
        q_final = bundle.q_enc[-1]
    print(f"[grasp] hand_grab on {'+'.join(sides)} "
          f"({', '.join(f'{s}={action_id[s]}' for s in sides)}) — arms hold still")
    t0 = time.monotonic()
    sent = {s: 0 for s in sides}
    pending = set(sides)
    turn = 0
    # 08-12 (user): the close takes 3-4 s with the arms frozen — exactly where
    # a slowly rotating base carries the jaws off the cube, and the blind path
    # has no MotionWatch here. Integrate gyro yaw over the close and report it.
    _yaw_rad, _yaw_t = 0.0, None
    while True:
        tick = time.monotonic()
        telemetry = tel.fresh()
        if telemetry is None:
            return "telemetry went stale during grasp", ()
        _ang = telemetry.get("base_ang_vel")
        if _ang is not None and len(_ang) >= 3 and np.isfinite(_ang[2]):
            if _yaw_t is not None:
                _yaw_rad += float(_ang[2]) * (tick - _yaw_t)
            _yaw_t = tick
        if any(telemetry.get("arm_fault") or [False, False]):
            return "ARM FAULT during grasp", ()
        packet = {"arm_targets": {s: _slot(bundle, q_final, s, rate)
                                  for s in bundle.active_sides}}
        # One ee_action per packet: round-robin over the sides still owed
        # repeats, so both commands reach the wire within a tick of each other.
        owed = [s for s in sides if sent[s] < OP.EE_ACTION_REPEAT_TICKS]
        if owed:
            s = owed[turn % len(owed)]
            turn += 1
            packet["ee_action"] = {"side": s, "command": "hand_grab",
                                   "action_id": action_id[s]}
            sent[s] += 1
        det.poll(rig)
        pub.publish(rig.tick(telemetry, packet))
        for s in list(pending):
            ee = (telemetry.get("ee") or {}).get(s) or {}
            if ee.get("grasp_action_id") != action_id[s]:
                continue
            if ee.get("grasp_detected") is True:
                print(f"[grasp] {s}: CONTACT confirmed (id matched) after "
                      f"{time.monotonic()-t0:.1f}s")
                pending.discard(s)
            elif ee.get("grasp_detected") is False:
                # COLLECTED, not returned. A certified-empty pinch is the one
                # failure the operator's contract calls RETRYABLE (class 1), so
                # the caller has to be able to tell it apart from an
                # indeterminate result — and to know which HAND it was, because
                # empty-on-one-side-while-the-other-holds is a different
                # situation from both hands empty.
                print(f"[grasp] {s}: EMPTY (certified) after "
                      f"{time.monotonic()-t0:.1f}s")
                empty.add(s)
                pending.discard(s)
        if not pending:
            print(f"[base] yaw drifted {np.degrees(_yaw_rad):+.1f} deg during "
                  f"the {time.monotonic()-t0:.1f}s close (gyro integral; "
                  f"negative = clockwise/right)")
            if empty:
                return (f"{'+'.join(sorted(empty))} gripper closed EMPTY "
                        f"(certified) — nothing to carry on that side"), \
                       tuple(sorted(empty))
            return None, ()
        if time.monotonic() - t0 > GRASP_RESULT_TIMEOUT_S:
            # INDETERMINATE, never retryable: a lost verdict means we do not
            # know what the fingers are holding, and reopening on a maybe-held
            # cube drops it. Fail closed, exactly as the operator does.
            return (f"no id-matched grasp verdict from {'+'.join(sorted(pending))} "
                    f"in {GRASP_RESULT_TIMEOUT_S:.0f}s — FAIL CLOSED: fingers "
                    f"left as commanded, inspect before restart"), ()
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))


def reopen_hands(pub, tel: TelemetryView, det: DetectionView, rig: GazeRig,
                 sides, bundle, rate: float,
                 hold_enc: dict[str, float] | None = None) -> None:
    """hand_open on every named side, interleaved, holding the arm still.

    Needed because the class-1 contract is "reopen and re-grab": re-issuing
    hand_grab on a hand that is already closed asks the service to close an
    already-closed gripper, which stalls at zero travel and certifies empty
    again forever. The fingers have to go back out first.

    Interleaved for the same reason do_grasp is: a 9874 packet carries at most
    one ee_action, and blocking on one side leaves the other un-repeated on a
    latest-value-wins socket.
    """
    sides = tuple(sides)
    if not sides:
        return
    ids = {s: new_grasp_action_id() for s in sides}
    # Same hold rule as do_grasp: an MPC run's arms sit on the LIVE cube, not
    # on the route's final waypoint — reopening must not move them.
    if hold_enc is not None:
        q_final = np.array([float(hold_enc[j]) for j in bundle.joint_names])
    else:
        q_final = bundle.q_enc[-1]
    sent = {s: 0 for s in sides}
    turn = 0
    print(f"[grasp] reopening {'+'.join(sides)} before the next attempt …")
    while any(sent[s] < OP.EE_ACTION_REPEAT_TICKS for s in sides):
        tick = time.monotonic()
        telemetry = tel.fresh()
        if telemetry is None:
            return
        packet = {"arm_targets": {s: _slot(bundle, q_final, s, rate)
                                  for s in bundle.active_sides}}
        owed = [s for s in sides if sent[s] < OP.EE_ACTION_REPEAT_TICKS]
        s = owed[turn % len(owed)]
        turn += 1
        packet["ee_action"] = {"side": s, "command": "hand_open",
                               "action_id": ids[s]}
        sent[s] += 1
        det.poll(rig)
        pub.publish(rig.tick(telemetry, packet))
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))


def _measured_ee(tel_data: dict, arm: str):
    ee = ((tel_data.get("ee") or {}).get(arm) or {}).get("pos_actual")
    if ee is None:
        return None
    p = np.asarray(ee, dtype=float)
    return p if p.shape == (3,) and np.all(np.isfinite(p)) else None


def raise_to_chest_home(pub, tel: TelemetryView, rig: GazeRig,
                        det: DetectionView, spec=None) -> str | None:
    """Return both arms to the chest home before planning anything.

    `det` is polled every tick (2026-08-03): this loop ticks the rig, so the
    cameras are already sweeping — a phase that rotates the eyes but never
    consumes detections is a structural blind window (the user clocked 6-8 s
    of "camera turning, nothing reacts" that was exactly zeroing + this
    raise). Polling here lets first-seer locks land before the scan phase
    even starts.

    Since 07-31 the chest home is the power-on posture (the simulator's home),
    so on a fresh boot this measures zero distance and exits immediately. It
    earns its keep on the session AFTER an abort: the failsafe crawl ends at
    default_pose (the same posture), but a run interrupted mid-crawl or an arm
    moved by hand starts from nowhere in particular, and this puts it back.
    History: born 07-29 as a real ~10 cm lift when the chest home sat above the
    power-on pose; the two have converged twice since (07-30 raised power-on to
    the chest, 07-31 moved both to the sim home).

    JOINT SPACE SINCE 07-31, AND THAT REVERSAL IS THE WHOLE POINT. This used to
    be "the tool's one Cartesian moment": it published (CHEST_HOME_EE,
    CHEST_HOME_QUAT) and let the robot-side mink find the configuration, on the
    reasoning that the joint solution belonged to the solver and not to us.
    That reasoning held only while the chest home WAS an IK product — solved
    offline FROM that 6-DoF target, residual 0.8 mm, hence reachable by
    construction. The one-home migration inverted the derivation: the joints
    are now the simulator's HUMANOID_ARM_JOINT_HOME, imported verbatim, and
    CHEST_HOME_EE is merely their FK. Nothing re-checked attainability, and it
    does not hold.

    Measured with the deployment's own solver (BatchedAnalyticalIK, built
    exactly as humanoid_real_env.py does), 2000 steps from each recognised
    start: the worst-arm EE never comes within CHEST_HOME_TOL_M, and every run
    lands 1.512 rad — 86.6 deg — from CHEST_HOME_JOINTS with wrist_3 pinned on
    its limit. The same harness aimed at the OLD target converges to 1.2 mm.
    Cause: mink's CollisionAvoidanceLimit keeps a hard 50 mm, and at this home
    shoulder_3~wrist_1 measures 16.5 mm and wrist_1~wrist_3 10.2 mm, so the QP
    will not steer INTO the fold. Set collision_min_distance=0 and it converges
    to 0.7 mm. The posture is holdable — starting there, mink holds it to
    4e-4 rad — just not attainable through a Cartesian command.

    So the home is commanded as JOINTS, the exact vector, no solver in the
    loop. This is not a new idea in this codebase: the journey's side<->tuck
    glide was moved off Cartesian on 07-21 for the identical failure
    (auto_operator/independent.py: "straight-line EE + IK left the arm up to
    62 deg off the true power-on posture"). The chest-home path was simply
    never moved with it. CHEST_HOME_EE / CHEST_HOME_QUAT survive as what they
    always were underneath — the FK record used to RECOGNISE an arm parked at
    the home — and are no longer commanded by anything here.

    Returns a refusal, or None once both arms have settled.
    """
    goal = {arm: np.asarray(OP.mirror_arm(OP.CHEST_HOME_JOINTS, arm), dtype=float)
            for arm in ("left", "right")}
    names = {"left": ARM_JOINTS_L, "right": ARM_JOINTS_R}
    print(f"[home] returning to the chest home (JOINT space, the sim home) at "
          f"{OP.STAGE_RATE} rad/s …")
    # The rate is carried IN the packet and the receiver lerps toward the
    # target, so the goal may be published whole — there is no Cartesian jump
    # to ramp away from (the 07-18 "arms shot out to the sides" incident was a
    # raw EE goal handed to an IK that swept it at its own cap).
    t0, warn_t = time.monotonic(), -1e9
    while time.monotonic() - t0 < CHEST_HOME_WAIT_S:
        tick = time.monotonic()
        telemetry = tel.fresh()
        if telemetry is None:
            return "telemetry went stale while returning to the chest home"
        if any(telemetry.get("arm_fault") or [False, False]):
            return "ARM FAULT while returning to the chest home"
        enc = _arm_enc(telemetry)
        if enc is None:
            # No arm encoders yet: stay SILENT rather than command blind.
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
            continue
        slots, worst = {}, 0.0
        for arm in ("left", "right"):
            slots[arm] = {"joint_pos": [float(v) for v in goal[arm]],
                          "rate": float(OP.STAGE_RATE)}
            q = np.array([enc[n] for n in names[arm]])
            worst = max(worst, float(np.max(np.abs(q - goal[arm]))))
        det.poll(rig)
        if spec is not None:
            spec.maybe_launch(time.monotonic())
        pub.publish(rig.tick(telemetry, {"arm_targets": slots}))
        if worst <= OP.JOURNEY_POSTURE_TOL_RAD:
            print(f"[home] both arms at the chest home "
                  f"({math.degrees(worst):.1f} deg, {time.monotonic()-t0:.0f}s)")
            return None
        if tick - warn_t > 5.0 and np.isfinite(worst):
            warn_t = tick
            print(f"[home] returning … worst joint {math.degrees(worst):.1f} deg "
                  f"out (need {math.degrees(OP.JOURNEY_POSTURE_TOL_RAD):.1f})")
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    return (f"arms did not reach the chest home within {CHEST_HOME_WAIT_S:.0f}s — "
            f"the JOINT-space stream was published throughout, so this is a "
            f"receiver or motor problem, not an IK one: check real_env for "
            f"arm faults, OVER CURRENT, or a dropout-watchdog hold")


def zero_gimbals(pub, tel: TelemetryView, rig: GazeRig, det: DetectionView,
                 rate: float, spec=None) -> str | None:
    """Force both cameras to their joint ZEROS before any sweep starts (user
    2026-08-03: 'every run must zero the cameras first, then start turning').

    WHY. The sweep's clock starts at zero, but the gimbals start wherever the
    LAST run parked them — mid-track, mid-serpentine, or at a DONE park — so
    the first seconds of every run covered a different bearing order and the
    time-to-first-lock varied run to run. Parking at zero first makes the
    sweep deterministic: same start, same coverage order, every time.

    The target is published as a sender-side SLEW toward zero (the robot
    applies gaze_reference raw since 8045371 — a raw zero from a far pose
    would be a multi-radian slam), with the arms held and detections polled
    like every other startup phase. eye() is deliberately NOT in the loop: it
    would immediately re-aim/scan. On arrival (or a WARNED timeout) the scan
    phase is reset so the serpentine starts from its own t=0."""
    telemetry = tel.fresh()
    hold_enc = _arm_enc(telemetry) if telemetry else None
    if hold_enc is None:
        return "no arm encoders — cannot hold the arms while zeroing the cameras"
    print("[gaze] parking both cameras at ZERO before the sweep …")
    t0 = time.monotonic()
    zeros = np.zeros(4)
    while time.monotonic() - t0 < GIMBAL_ZERO_WAIT_S:
        tick = time.monotonic()
        telemetry = tel.fresh()
        if telemetry is None:
            return "telemetry went stale while zeroing the cameras"
        det.poll(rig)
        if spec is not None:
            spec.maybe_launch(time.monotonic())
        jp = np.asarray(telemetry.get("joint_pos", ()), dtype=float)
        if jp.size >= 31 and np.all(np.isfinite(jp[27:31])):
            if not rig._gaze_seeded:
                rig.gaze = jp[27:31].astype(float).copy()
                rig._gaze_seeded = True
            if float(np.max(np.abs(jp[27:31]))) < GIMBAL_ZERO_TOL_RAD:
                rig._scan_phase = 0.0          # the sweep clock starts HERE
                print(f"[gaze] cameras at zero "
                      f"({time.monotonic() - t0:.1f}s) — sweep starts fresh")
                return None
            rig.gaze = ao_gaze.slew_gaze(rig, zeros)
            pub.publish({"gaze_targets": [float(x) for x in rig.gaze],
                         **hold_packet(hold_enc, rate)})
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    rig._scan_phase = 0.0
    print(f"[gaze] WARNING: cameras not at zero after "
          f"{GIMBAL_ZERO_WAIT_S:.0f}s — proceeding with the sweep anyway "
          f"(a stuck gimbal joint, or gaze packets not reaching real_env)")
    return None


def zero_grippers(pub, tel: TelemetryView, rig: GazeRig, det: DetectionView,
                  rate: float, spec=None) -> str | None:
    """Calibrate both grippers at startup, or explain why it was skipped.

    `det` is polled in every hold loop (2026-08-03): zeroing takes seconds and
    the rig is ticking — the cameras sweep the whole time. Without polling,
    every cube they crossed was seen-but-not-consumed, and the user clocked
    the result as "6-8 s after the cameras start turning, nothing reacts".
    Polling here means a cube can be first-seer-locked before calibration
    even finishes.

    WHY THIS TOOL HAS TO DO IT ITSELF. `zero_gripper` is a CALIBRATION, not a
    flourish: the service closes slowly to a stall, records that position as
    "closed", and derives "open" from it. Without it, hand_grab's travel is
    referenced to whatever the last session left behind — so a grasp verdict
    from an uncalibrated gripper is not trustworthy, and this tool's whole grasp
    contract is built on trusting that verdict. It lived only in the operator,
    which meant "I want to test cuRobo, so I skipped the operator" silently cost
    the calibration. Asked exactly that on 2026-07-30.

    THE GUARD ASKS A PHYSICAL QUESTION NOW (2026-08-01). It used to refuse on
    the mere PRESENCE of a prior grasp marker — but the marker is cleared only
    by hand_open, and a successful mission's designed ending is "hold the cube
    until Ctrl+C": nobody ever sends hand_open. So every grasp, successful or
    not, poisoned the NEXT run into a REFUSED, which read as "power-cycling
    breaks the tool" (it was never about power at all — the id in the refusal
    decoded to a hand_grab from minutes earlier in the same bench session).

    The hazard the guard exists for is real — zeroing pinches and then opens,
    which drops whatever is held — so the marker is now weighed against the
    servo's own LOAD (heartbeat servo_load, |load| 0..1023). Holding a cube
    means the servo is actively squeezing: sustained load. An empty hand at
    rest reads noise. Marker + load => refuse, exactly as before. Marker + no
    load => the marker is stale history on a verifiably empty hand; say so and
    calibrate. Marker + NO load reading (service just started, poll not in
    yet) => wait briefly for the ~4 s-cadence heartbeat, then fail CLOSED.
    """
    telemetry = tel.fresh()
    if telemetry is None:
        return "no telemetry — cannot check the gripper state before zeroing"

    # HOLD THE ARMS from here on: the evidence wait below and the calibration
    # both outlast _ARM_CMD_TIMEOUT, and the rule is that no wait longer than
    # that may publish without arm content (see hold_until_enter).
    hold_enc = _arm_enc(telemetry)
    if hold_enc is None:
        return "no arm encoders — cannot hold the arms while the grippers zero"

    def _marked(t: dict, side: str) -> bool:
        ee = (t.get("ee") or {}).get(side) or {}
        return ee.get("grasp_detected") is not None or \
            ee.get("grasp_action_id") is not None

    suspects = [s for s in ("left", "right") if _marked(telemetry, side=s)]
    if suspects:
        # Wait (holding) for a load reading on every suspect side — the
        # thermal poll alternates sides every SERVO_TEMP_POLL_SEC, so a fresh
        # service needs a few seconds before servo_load is non-None.
        deadline = time.monotonic() + EE_LOAD_EVIDENCE_S
        while time.monotonic() < deadline:
            tick = time.monotonic()
            telemetry = tel.fresh()
            if telemetry is None:
                return "telemetry went stale while checking the grippers"
            det.poll(rig)
            if spec is not None:
                spec.maybe_launch(time.monotonic())
            out = rig.tick(telemetry, hold_packet(hold_enc, rate))
            if out:
                pub.publish(out)
            if all(((telemetry.get("ee") or {}).get(s) or {})
                   .get("servo_load") is not None for s in suspects):
                break
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
        for side in suspects:
            ee = (telemetry.get("ee") or {}).get(side) or {}
            load = ee.get("servo_load")
            if load is None:
                return (f"{side} gripper carries a prior grasp marker and no "
                        f"servo load reading arrived in {EE_LOAD_EVIDENCE_S:.0f}s "
                        f"to prove the hand empty — NOT zeroing (that pinches "
                        f"and opens, dropping anything held). Is the EE "
                        f"service's heartbeat flowing? A restart of it clears "
                        f"stale markers.")
            if abs(float(load)) > EMPTY_GRIP_LOAD_MAX:
                return (f"{side} gripper carries a prior grasp marker AND its "
                        f"servo shows load {abs(float(load)):.0f} "
                        f"(> {EMPTY_GRIP_LOAD_MAX:.0f}): the hand is actively "
                        f"squeezing SOMETHING — NOT zeroing. Take the object "
                        f"out (send hand_open or open by hand), then re-run.")
            print(f"[grasp] {side}: stale grasp marker "
                  f"(id={ee.get('grasp_action_id')!r}) but servo load "
                  f"{abs(float(load)):.0f} <= {EMPTY_GRIP_LOAD_MAX:.0f} — the "
                  f"hand is verifiably exerting no grip; calibrating. (The "
                  f"marker is only ever cleared by hand_open, and a mission "
                  f"that ends holding-until-Ctrl+C never sends one.)")
    # ee_alive is DERIVED liveness: real_env's client flips it on the EE
    # service's 1 Hz heartbeat, and a service (re)start needs a redial plus one
    # heartbeat before it reads True. Sampling it once was therefore a race —
    # hardware 2026-08-01: restart T2, re-run the mission, and the first
    # telemetry frame still said alive=False -> REFUSED, cured by nothing but
    # running the same command again. Wait it out (holding), refuse only when
    # it STAYS dead.
    deadline = time.monotonic() + EE_ALIVE_WAIT_S
    alive = None
    while time.monotonic() < deadline:
        tick = time.monotonic()
        alive = ((telemetry.get("ee") or {}).get("left") or {}).get("ee_alive")
        if alive is True:
            break
        det.poll(rig)
        if spec is not None:
            spec.maybe_launch(time.monotonic())
        out = rig.tick(telemetry, hold_packet(hold_enc, rate))
        if out:
            pub.publish(out)
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
        telemetry = tel.fresh()
        if telemetry is None:
            return "telemetry went stale while waiting for the EE heartbeat"
    if alive is None:
        return ("telemetry has no ee_alive field after "
                f"{EE_ALIVE_WAIT_S:.0f}s — is real_env running with "
                f"--ee-service? Without it every gripper command is dropped.")
    if alive is False:
        return (f"EE chain still not alive after {EE_ALIVE_WAIT_S:.0f}s of "
                f"heartbeats missed — is humanoid_end_effector_service up and "
                f"connected to both hands?")

    print("[grasp] zeroing grippers (slow pinch -> open, both sides) …")
    t0 = time.monotonic()
    # FSM ORDER (user 2026-08-10): the zeroing FINISHES before the cameras
    # start searching. Both eyes are parked for the duration through the
    # retire mechanism (retired = parked at origin, the 08-09 rule) and the
    # prior set is restored on EVERY exit path, so the survey scan starts
    # only after the completion wait below has passed. The completion signal
    # is the per-side `zeroing` flag the EE heartbeat carries
    # (service -> real_env client -> telemetry "ee" dicts).
    parked0 = set(rig._cam_retired)
    rig._cam_retired.update(OP.CAM_PORTS)
    try:
        # INTERLEAVED, not one side then the other. A packet carries at most
        # one ee_action, so the two commands cannot share a tick — but
        # blocking on the left for its full repeat window would start the
        # right side 1.5 s later and leave the left un-repeated while the
        # right runs. Alternating puts both commands on the wire within one
        # tick of each other and keeps repeating BOTH, which is what
        # EE_ACTION_REPEAT_TICKS is for: 9874 is latest-value-wins at both
        # hops and was seen eating commands live (07-16). The service
        # calibrates each side as an independent background task, so they
        # then run concurrently.
        for _ in range(OP.EE_ACTION_REPEAT_TICKS):
            for side in ("left", "right"):
                telemetry = tel.fresh()
                if telemetry is None:
                    return "telemetry went stale while zeroing"
                det.poll(rig)
                if spec is not None:
                    spec.maybe_launch(time.monotonic())
                pub.publish(rig.tick(telemetry, {
                    "ee_action": {"side": side, "command": "zero_gripper"},
                    **hold_packet(hold_enc, rate)}))
                time.sleep(PUB_DT)
        # Settle: give the service time to ACCEPT (its in-flight flags go
        # True within one poll of the burst; the heartbeat mirrors them at
        # 1 Hz), keeping gaze stream and arm hold alive.
        while time.monotonic() - t0 < GRIPPER_ZERO_S:
            telemetry = tel.fresh()
            det.poll(rig)
            if spec is not None:
                spec.maybe_launch(time.monotonic())
            pkt = rig.tick(telemetry or {}, hold_packet(hold_enc, rate))
            if pkt:
                pub.publish(pkt)
            time.sleep(PUB_DT)
        # Completion wait: both sides must report the calibration finished
        # (zeroing False) before the mission may move on to the survey.
        while True:
            telemetry = tel.fresh()
            if telemetry is None:
                return ("telemetry went stale while waiting for the gripper "
                        "zeroing to finish")
            zst = {s: ((telemetry.get("ee") or {}).get(s) or {}).get("zeroing")
                   for s in ("left", "right")}
            if zst["left"] is False and zst["right"] is False:
                # FINISHED is not SUCCEEDED (08-10 night: an unpowered servo
                # rail "calibrated" to close=0 in 2 s and this gate printed
                # ZEROED over jaws that never moved). A failed calibration
                # marks its side unhealthy, the heartbeat's healthy field
                # goes False, and real_env mirrors that as ee_alive on the
                # LEFT dict (the codebase's one liveness field — the alive
                # wait above reads the same key). Dead here = the zeroing
                # FAILED; say so instead of searching with jaws that never
                # moved.
                if not ((telemetry.get("ee") or {}).get("left") or {}) \
                        .get("ee_alive"):
                    return ("gripper zeroing FAILED — the EE service went "
                            "unhealthy during the calibration (unpowered "
                            "servo rail? dead serial?). Check the T2 "
                            "terminal and the gripper servo power, then "
                            "restart T2.")
                print(f"[grasp] grippers ZEROED ({time.monotonic() - t0:.0f}s)"
                      f" — the cameras may search now")
                break
            if (zst["left"] is None or zst["right"] is None) \
                    and time.monotonic() - t0 > GRIPPER_ZERO_S \
                    + GRIPPER_ZERO_FIELD_GRACE_S:
                print("[grasp] WARNING: telemetry carries no per-side "
                      "`zeroing` state — real_env or the EE service predates "
                      "the zero-before-scan order (restart T2 and T3 with "
                      "the current code); proceeding on the old settle timer")
                break
            if time.monotonic() - t0 > GRIPPER_ZERO_DONE_S:
                return (f"gripper zeroing still in flight after "
                        f"{GRIPPER_ZERO_DONE_S:.0f}s — a stuck calibration; "
                        f"check the EE service terminal")
            det.poll(rig)
            if spec is not None:
                spec.maybe_launch(time.monotonic())
            pkt = rig.tick(telemetry, hold_packet(hold_enc, rate))
            if pkt:
                pub.publish(pkt)
            time.sleep(PUB_DT)
        return None
    finally:
        rig._cam_retired.clear()
        rig._cam_retired.update(parked0)


# ------------------------------------------------------------------------------
# Mission
# ------------------------------------------------------------------------------

def route_preview_payload(bundle, model) -> dict:
    """FK the route's EE path on OUR deploy model (the same FK Gate A trusts)
    into a base-frame polyline per active tool site, decimated to ~150 points."""
    data = mujoco.MjData(model)
    col = {n: i for i, n in enumerate(bundle.joint_names)}
    jadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in bundle.joint_names}
    sids = {frame: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame)
            for frame in bundle.reaches}
    stride = max(1, bundle.horizon // 150)
    rows = list(range(0, bundle.horizon, stride))
    if rows[-1] != bundle.horizon - 1:
        rows.append(bundle.horizon - 1)
    paths = {frame: [] for frame in bundle.reaches}
    for h in rows:
        data.qpos[:] = model.qpos0
        for n in bundle.joint_names:
            data.qpos[jadr[n]] = bundle.q_enc[h, col[n]]
        mujoco.mj_kinematics(model, data)
        for frame, sid in sids.items():
            paths[frame].append([float(x) for x in data.site_xpos[sid]])
    cands = {frame: np.asarray(r["cands"], dtype=float).reshape(-1, 3).tolist()
             for frame, r in bundle.reaches.items()}
    return {"paths": paths, "cands": cands, "goal": bundle.goal,
            "stamp": time.time()}


def journey_nav(pos: np.ndarray, arm: str, leg_dir: float) -> tuple[float, float]:
    """(vx, wz) for one journey leg — the operator's steering law, same constants.

    WHY REWRITTEN RATHER THAN IMPORTED. The three journey GATES are imported and
    run verbatim; this one cannot be, because in the operator it lives inside
    _journey_nav together with op._arm_tasks and ArmTaskPhase — the task
    scheduler this tool exists to replace. So the law is restated, every
    constant is read from OP rather than retyped, and a differential test pins
    (vx, wz) against the operator's own function over a grid of cube positions.
    Getting it subtly wrong is otherwise invisible until the robot walks.

    The two non-obvious terms, both from the 07-26 deadlock audit:

      * the aim point is offset JOURNEY_AIM_OFFSET_M INTO the serving arm's
        half-space. Plain pursuit servoes the cube onto bearing 0, which is the
        centre of the midline dead band where ownership checks disqualify it —
        the robot arrives perfectly aimed at a cube it may not touch. Arrival
        distance below still uses the TRUE cube position, not the aim point.
      * a BACKWARD leg gets a higher speed floor. The gait policy steps in place
        at a commanded -0.1 m/s, so a backward approach stalls short of the
        arrival gate forever at the normal floor.

    `leg_dir` is latched by the caller on the first tick of a leg (+1 forward,
    -1 reverse) and must not be recomputed mid-leg: a cube crossing the +/-90
    deg bearing line would otherwise flip the robot's direction of travel.
    """
    err = journey_align_error(pos, arm, leg_dir)
    if abs(err) > OP.JOURNEY_FACE_TOL_RAD:
        # ARC INTO THE FACING — restated from navigation_command (the
        # differential test pins the two): unfaced yaw is FLOORED at
        # JOURNEY_WZ_UNFACED_FLOOR (sub-floor is the measured dead band),
        # capped at JOURNEY_WZ_UNFACED_MAX; faced yaw keeps the WZ_MAX trim.
        wz = math.copysign(
            float(np.clip(OP.K_STEER * abs(err),
                          OP.JOURNEY_WZ_UNFACED_FLOOR,
                          OP.JOURNEY_WZ_UNFACED_MAX)), err)
    else:
        wz = float(np.clip(OP.K_STEER * err, -OP.WZ_MAX, OP.WZ_MAX))
    x, y = float(pos[0]), float(pos[1])
    near_vx = OP.JOURNEY_NEAR_VX_BACK if leg_dir < 0 else OP.JOURNEY_NEAR_VX
    dist = float(np.hypot(x, y))
    vx_mag = float(np.clip(OP.JOURNEY_SLOW_K * (dist - OP.JOURNEY_STOP_M),
                           near_vx, OP.CRUISE_VX))
    return leg_dir * vx_mag, wz


def _wrap_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def journey_align_error(pos, arm: str, leg_dir: float) -> float:
    """Signed bearing of the aim point off the TRAVEL axis [rad].

    The single source of the offset-pursuit aim law — journey_nav steers on
    this WHILE WALKING (curved walking only, upstream ruling 2026-08-08: in-place
    rotation is forbidden on this robot; the 08-08 --wz bench measured torso
    wind-up at 0.2 rad/s and a near-fall at 0.4). The aim point is the cube
    shifted JOURNEY_AIM_OFFSET_M INTO the serving arm's half-space; a
    reverse leg measures against the BACKWARD axis (bearing wrapped by pi) —
    never a 180 turn-around (the robot walks backward; fore-aft symmetry is
    the design)."""
    x, y = float(pos[0]), float(pos[1])
    aim_side = 1.0 if arm == "left" else -1.0
    # Per-direction offset (user 08-10): reverse walks straighter at the
    # smaller offset, forward keeps the table-face clearance. Reverse aims
    # JOURNEY_REVERSE_DRIFT_COMP_M to the RIGHT of its parking spot: the
    # backward gait's measured leftward settle drift (+0.15..0.21 across all
    # logged reverse legs, both arms) then lands the cube ON the spot.
    if leg_dir > 0:
        y_eq = aim_side * OP.JOURNEY_AIM_OFFSET_M
    else:
        y_eq = aim_side * OP.JOURNEY_AIM_OFFSET_BACK_M \
            - OP.JOURNEY_REVERSE_DRIFT_COMP_M
    bearing = math.atan2(y - y_eq, x)
    return bearing if leg_dir > 0 else _wrap_pi(bearing + math.pi)


def journey_leg_dir(pos: np.ndarray) -> float:
    """+1 to walk forward at the cube, -1 to back toward it. Latched ONCE."""
    return 1.0 if abs(math.atan2(float(pos[1]), float(pos[0]))) <= math.pi / 2 \
        else -1.0


def journey_z_verdict(z: float, clearance_floor_mm: float) -> str | None:
    """Complaint if the cube height is outside the graspable band, else None.

    The 08-08/09 ledger: successful grasps read z in [-0.00, +0.040]; z=+0.07
    folded the wrist against the chest (20/20 clearance refusals within 5 mm
    of the floor) and z=-0.021 pushed the rear reach INFEASIBLE 20/20. Height
    is a property of the STAND, not the stance — no amount of replanning or
    walking fixes it, so the leg fails fast with the physical instruction.

    The HIGH verdict is a fact about the 25 mm clearance floor it was measured
    at: those 20/20 refusals all had routes 20-25 mm from the chest. At a
    floor below JOURNEY_Z_LEDGER_FLOOR_MM the outcome is not pre-decided, so
    the gate steps aside and the solver judges (user 08-09: "accept the
    higher cube"). The LOW verdict is cuRobo's own envelope INFEASIBLE —
    floor-blind, it always binds."""
    if z > OP.JOURNEY_CUBE_Z_MAX \
            and clearance_floor_mm >= OP.JOURNEY_Z_LEDGER_FLOOR_MM:
        return (f"cube reads z={z * 1000:+.0f} mm — "
                f"{(z - OP.JOURNEY_CUBE_Z_MAX) * 1000:.0f}"
                f" mm above the graspable band (max {OP.JOURNEY_CUBE_Z_MAX * 1000:.0f}):"
                f" a high close cube folds the wrist against the chest. LOWER the stand"
                f" (or run with --clearance-floor-mm below"
                f" {OP.JOURNEY_Z_LEDGER_FLOOR_MM:.0f} to attempt it)")
    if z < OP.JOURNEY_CUBE_Z_MIN:
        return (f"cube reads z={z * 1000:+.0f} mm — below the graspable band "
                f"(min {OP.JOURNEY_CUBE_Z_MIN * 1000:.0f}): the reach goes "
                f"INFEASIBLE at the envelope floor. RAISE the stand")
    return None


def journey_serving_arm(pos: np.ndarray, free=None) -> str | None:
    """The arm on the cube's own side, restricted to hands that are FREE.

    Measured 07-30 across y in [-0.30, +0.30]: the near arm cleared every gate,
    and crossing the midline past ~0.10 m was INFEASIBLE — so the side is not a
    preference, it is the only assignment with a route. The nav aim offset then
    keeps the cube on this side through arrival.

    `free` is the set of hands not already carrying a cube (08-07 audit). This
    used to be the bare sign of y, which on leg 2 nominates the hand that is
    already full about half the time — and the nav aim offset then spends the
    entire walk DRIVING the second cube into that full hand's half-space.

    THE 07-30 FINDING IS A BAND, NOT A SIGN (hardware 2026-08-07). Crossing was
    INFEASIBLE *past ~0.10 m*, so inside that band either arm has a route. This
    read `y >= 0` and refused a leg over a cube measured at y=+0.026 — 26 mm, a
    fifth of the way to the bar and inside the detector's own noise — while the
    other hand was free. A 100 mm tolerance had been implemented as a 0 mm one.
    So: prefer the cube's own side; if that hand is full, the free hand may
    still take it while the cube is within CROSS_MIDLINE_REACH_M of the midline.
    Returns None only when the cube is genuinely deep in the full hand's
    half-space — there the caller must SKIP rather than hand the planner a
    pairing the sweep measured as having no solution."""
    y = float(pos[1])
    want = "left" if y >= 0.0 else "right"
    if free is None or want in free:
        return want
    other = "right" if want == "left" else "left"
    if other in free and abs(y) <= OP.CROSS_MIDLINE_REACH_M:
        return other
    return None


def forced_assignments(target_keys, only_sides=None) -> list[dict]:
    """Every explicit side<->cube pairing worth asking for, in order.

    The auto assignment is the server's own choice and is usually right; this
    is the fallback for when it produces nothing OUR gates accept, which is a
    different question from what cuRobo can solve. One cube can be
    unreachable-with-margin for the arm on its side and easy for the other; two
    cubes have exactly two pairings and the cheap one is simply trying the
    other.

    Deliberately exhaustive rather than clever: at these sizes the whole search
    space is two entries, and a heuristic that picked one would be a guess
    dressed as a decision.

    `only_sides` ("left"/"right" names) restricts the search to hands that are
    actually available (08-07 audit). On a journey's second leg one hand is
    already carrying a cube, and every pairing that uses it is not a fallback
    but a way to re-grab the cube already held.
    """
    keys = tuple(target_keys)
    ok = None if only_sides is None else \
        {{"left": "L", "right": "R"}[s] for s in only_sides}
    out: list[dict] = []
    if len(keys) == 1:
        out = [{"L": keys[0]}, {"R": keys[0]}]
    elif len(keys) == 2:
        a, b = keys
        out = [{"L": a, "R": b}, {"L": b, "R": a}]
    if ok is None:
        return out
    return [d for d in out if set(d) <= ok]


def decoupled_assignment(targets: dict) -> dict:
    """{'L': key, 'R': key} — the cube further toward the robot's LEFT
    (greater y) goes to the left arm. Two cubes only; the decoupled path is
    for the dual grasp."""
    keys = sorted(targets, key=lambda k: float(targets[k][1]), reverse=True)
    return {"L": keys[0], "R": keys[-1]}


def plan_decoupled_until_gated(client: PlanClient, scene, targets, *,
                               measured_enc, hand_z_floor, model,
                               max_rate: float, clearance_floor_m: float,
                               retries: int | None = None):
    """Two independent single-arm solves per attempt, merged and gated as one
    simultaneous route. (bundle, report) or (None, last_report) — the caller
    falls back to the joint solve. See merge_decoupled_bundles for why."""
    pair = decoupled_assignment(targets)
    report = None
    n = PLAN_RETRIES if retries is None else retries
    for k in range(n):
        per = {}
        for side, key in pair.items():
            b = client.plan(scene, {key: targets[key]}, measured_enc=measured_enc,
                            assignment={side: key}, goal="reach",
                            hand_z_floor=hand_z_floor)
            if b is None:
                print(f"[plan] decoupled {side}->{key}: INFEASIBLE "
                      f"{k + 1}/{n} — randomized restart …")
                per = None
                break
            per[side] = b
        if per is None:
            continue
        merged = merge_decoupled_bundles(per).stretch(max_rate)
        report = verify_route(merged, measured_enc, model=model,
                              clearance_floor_m=clearance_floor_m,
                              windup_guard=True)
        if report["ok"]:
            if k:
                print(f"[plan] decoupled pair accepted on attempt {k + 1}/{n}")
            return merged, report
        bad = ", ".join(f"{g}: {r['msg']}" for g, r in report["gates"].items()
                        if not r["ok"])
        print(f"[plan] decoupled attempt {k + 1}/{n} REJECTED on the MERGED "
              f"route ({bad}) — randomized restart …")
    return None, report


def plan_until_gated(client: PlanClient, *args, model=None, max_rate: float,
                     clearance_floor_m: float, retries: int | None = None,
                     **kwargs):
    """Re-solve until a route passes OUR gates. Returns (bundle, report) or
    (None, last_report).

    WHY THE GATES ARE INSIDE THE RETRY LOOP. The acceptance test for a plan is
    not "cuRobo returned something", it is "our own model says this is safe to
    execute" — those are different verdicts on different models, which is the
    premise the whole gate design rests on. The first version only retried an
    INFEASIBLE verdict and treated a GATE rejection as final, so one unlucky
    solve ended the mission.

    That is the wrong shape for a randomized-seed search. cuRobo restarts from
    fresh seeds every call and its clearance varies run to run: hardware
    2026-07-30 saw 0.6 mm, then 8.0 mm, then 39.5 mm on comparable targets, at
    0.13 s per solve. Asking again costs a fraction of a second and is exactly
    what the solver is built for; giving up after one sample is not.

    The gates themselves are unchanged and never relaxed — a route still has to
    clear the floor on our mesh model. What changed is that failing to find such
    a route on the first try is no longer failing the mission.
    """
    report = None
    n = PLAN_RETRIES if retries is None else retries
    for k in range(n):
        bundle = client.plan(*args, **kwargs)
        if bundle is None:
            print(f"[plan] INFEASIBLE verdict {k + 1}/{n} — "
                  f"randomized restart …")
            continue
        bundle = bundle.stretch(max_rate)
        report = verify_route(
            bundle, kwargs.get("measured_enc"), model=model,
            clearance_floor_m=clearance_floor_m,
            # Gate E only judges HOME routes: their end must be the home
            # POSTURE (same IK branch), not merely the home hand pose.
            home_enc=kwargs.get("home_enc")
            if kwargs.get("goal") == "home" else None,
            # Gate F only judges REACH routes: reject the wind-up family
            # (user field logs 2026-08-02: shoulder_1 swinging away from
            # the cubes predicts the regrasp path every time).
            windup_guard=kwargs.get("goal", "reach") == "reach")
        if report["ok"]:
            if k:
                print(f"[plan] accepted on attempt {k + 1}/{n}")
            return bundle, report
        bad = ", ".join(f"{g}: {r['msg']}" for g, r in report["gates"].items()
                        if not r["ok"])
        print(f"[plan] attempt {k + 1}/{n} REJECTED by our model "
              f"({bad}) — randomized restart …")
    return None, report


GRASP_ANCHOR_ABOVE_M = 0.005      # CLIENT MIRROR of the deployed server's
                                  # GRASP_Z_ABOVE_M (control/curobo_plan_server.py,
                                  # GRASP_Z_ABOVE_M; +0.005 since 08-13 night,
                                  # user — 8 mm for one hour, then 5. The
                                  # [z] ledger measured physical
                                  # closes ~+10 mm over the commanded line
                                  # all day with CONTACT every time, and a
                                  # raised line buys margin against the
                                  # tip-guard floor and the low-rear-cube
                                  # envelope. PRIOR: 0.0 (08-13 midday),
                                  # -0.015 (08-12, floor collision)): the
                                  # grasp point sits this far above the cube
                                  # centre (negative = below). The client
                                  # never sees the server
                                  # constant, so this mirror exists for
                                  # floor arithmetic that is defined
                                  # relative to the GRASP POINT; a guard pin
                                  # in test_grasp_candidate_sets compares it
                                  # against the sim clone's value whenever
                                  # legged_env_v2 is importable. Change the
                                  # server dial -> change this with it.
HAND_FLOOR_UNDER_GRASP_M = 0.020  # 08-11 (user): "if the camera cannot see
                                  # the tables, just regard that the
                                  # INFINITE table is 2 cm under the
                                  # grasping point." The unseen-table floor
                                  # is defined relative to the GRASP POINT,
                                  # not the cube centre, so it survives
                                  # anchor re-dials. History of the dial:
                                  # 23 mm under centre hovered (13 mm gap to
                                  # a 70 mm-activation 1e6 floor), 40 then
                                  # 30 mm under centre fixed it; this
                                  # re-anchoring keeps the same doctrine.
                                  # If the MPC hover returns, THIS is the
                                  # dial (and then the server's activation
                                  # distance).
HAND_FLOOR_TARGET_GAP_M = HAND_FLOOR_UNDER_GRASP_M - GRASP_ANCHOR_ABOVE_M
                                  # = 0.020 below the cube CENTRE — the form
                                  # the floor code speaks (targets are cube
                                  # centres; the server adds the anchor). On
                                  # REGISTERED, SEEN tables the surface tip
                                  # guard (HAND_FLOOR_TIP_MARGIN_M) still
                                  # bounds the clamp from below.
HAND_FLOOR_MARGIN_M = 0.023       # upstream's measured value: lifts the floor from
                                  # the work surface TOP to the gripper sphere
                                  # radius, so the jaw CENTRES clear it, not
                                  # just the site
HAND_FLOOR_SLACK_M = 0.01         # a floor this far above the target is treated
                                  # as fighting it, not protecting it
HAND_FLOOR_TIP_MARGIN_M = 0.010   # 08-10 night ("the gripper ALWAYS hits the
                                  # table"): hard LOWER bound on the clamped
                                  # floor when a real surface IS seen —
                                  # surface + this. The target cap follows the
                                  # MEASURED cube z, and a low/noisy reading
                                  # (cube f read +0.019 on a ~+0.045 table)
                                  # dragged the floor to 35 mm UNDER the wood;
                                  # with --no-table-world the floor is the
                                  # ONLY table protection, so the fingers
                                  # dove into the slab on every fetch. 10 mm
                                  # ~= the small fingertip-sphere radius (the
                                  # 23 mm HAND_FLOOR_MARGIN_M is the fat JAW
                                  # sphere, which rides much higher on the
                                  # hand) — tips close above the wood, and a
                                  # target reading below surface+10 mm loses
                                  # to the physical table rather than
                                  # licensing a dive. Wrong-fit surfaces
                                  # (measured ABOVE the target) never apply
                                  # this bound: a junk surface must not veto
                                  # the reach.


HAND_FLOOR_XY_TILT_CAP_M = 0.04   # #2 patch (08-13): the at-xy surface may
                                  # differ from the slab-CENTER top by at most
                                  # this much before the extrapolation is
                                  # distrusted. A real bench tilts a few
                                  # degrees at worst (~3.8 deg over the 0.6 m
                                  # half-slab = 40 mm); past that the tilt is
                                  # the FIT wobbling, and the center
                                  # measurement is the safer number.


def _table_surfaces(rig: GazeRig, table_keys, target_xy=None) -> list:
    """Effective work-surface z of every SEEN registered table.

    #2 PATCH (08-13, the leg-2 refusal): ONE CONVENTION, ONE SOURCE. The z
    prior evaluates the fitted top PLANE at the cube's xy (table_resting_z);
    the floor's tip guard used the slab-top at the TABLE'S center. A fit
    tilted 3-4 deg put those 17.8 mm apart at the cube, the tip guard landed
    ABOVE the grasp pose, and 30/30 solves went INFEASIBLE. With target_xy
    given, each table's surface is now the SAME plane-at-xy the prior uses —
    clamped to center-top ± HAND_FLOOR_XY_TILT_CAP_M against wobbly-fit
    extrapolation — and falls back to the center-top when the xy is off that
    slab's footprint (the floor still guards every slab, not just the one
    under the target)."""
    now = time.monotonic()
    out = []
    for n in table_keys:
        got = rig.table_pose(n, now)
        if got is None:
            continue
        pos = np.asarray(got[0], float)
        top = float(pos[2])
        if target_xy is not None:
            dims = TABLE_CONFIGS[n].slab_cuboid()["dims"]
            z_xy = table_resting_z(np.asarray(target_xy, float)[:2], pos,
                                   np.asarray(got[1], float), dims)
            if z_xy is not None:
                top = min(max(z_xy, top - HAND_FLOOR_XY_TILT_CAP_M),
                          top + HAND_FLOOR_XY_TILT_CAP_M)
        out.append(top)
    return out


def hand_floor_for_tables(rig: GazeRig, table_keys,
                          target_z: float | None = None,
                          target_xy=None) -> float | None:
    """The hand z-floor implied by the REGISTERED tables, or None if none are seen.

    Derivable all along: the tables carry their geometry in the registration, the
    tags put the top surface in the base frame, and the only extra term is the
    gripper-sphere radius so the jaw CENTRES clear the surface rather than just
    the tool site. Highest surface wins — a floor below a work surface stops
    guarding against diving under it. Since 08-13 the surface is evaluated AT
    THE TARGET'S XY on the fitted plane (see _table_surfaces) so the tip guard
    and the z prior speak the same number.

    NEVER ABOVE THE TARGET, though, and that clamp is not a nicety. A floor at or
    above the grasp pose is a 1e6-weight cost pushing the gripper away from the
    exact pose being requested, which does not degrade — it makes the reach
    INFEASIBLE outright. Hardware, 2026-07-30: a table fitted 18 mm ABOVE the
    cube standing on it (its orientation was wobbling 78 deg, so the fit was
    junk) produced floor +0.0886 against a target at +0.0483, and the run printed
    its own prediction of INFEASIBLE and then burned five solves proving it.
    A surface measured above an object resting on it is a WRONG surface, and a
    wrong surface must not be allowed to veto the reach.
    """
    tops = _table_surfaces(rig, table_keys, target_xy)
    if not tops:
        # NO WORK SURFACE, so derive the floor from the TARGET instead of
        # returning None. None means "keep whatever the server has", and the
        # server keeps a value across runs — hardware 2026-07-30: a --tables run
        # left +0.0886 behind, the next --no-table-world run reached for a cube
        # at +0.023, and cuRobo contorted around the stale floor until the left
        # elbow came within 0.6 mm of the torso. Gate C caught it, but a gate
        # catching a self-inflicted wound is not a design.
        return None if target_z is None else target_z - HAND_FLOOR_TARGET_GAP_M
    surface = max(tops)
    floor = surface + HAND_FLOOR_MARGIN_M
    if target_z is not None and floor >= target_z - HAND_FLOOR_SLACK_M:
        clamped = target_z - HAND_FLOOR_TARGET_GAP_M
        if surface >= target_z:
            print(f"[scene] NOTE: the table fit puts the work surface at "
                  f"{surface:+.4f}, AT OR ABOVE the grasp target at "
                  f"{target_z:+.4f} — impossible for an object resting on it, "
                  f"so the fit is WRONG. Clamping the hand z-floor to "
                  f"{clamped:+.4f} rather than letting a bad surface make the "
                  f"reach INFEASIBLE. The table is a height sensor only since 08-13 (not a planner obstacle), so this floor is its protection; check "
                  f"the slab drawn in viser.")
        else:
            # NOT a bad fit — this is what a small object on a table looks like.
            # A 60 mm cube puts its centre 30 mm above the surface while the
            # gripper-sphere margin is 23 mm, so the floor lands 7 mm under the
            # grasp pose: inside the slack band, and a 1e6-weight cost that
            # close to the goal fights the reach. Blaming the fit here sends the
            # operator to inspect a slab that is drawn perfectly correctly.
            #
            # TIP GUARD (08-10 night): but the cap follows the MEASURED target,
            # so a low/noisy cube reading dragged the floor tens of mm UNDER
            # this verifiably real surface — and on a --no-table-world run the
            # floor is the only table protection there is. The clamped floor
            # never goes below surface + HAND_FLOOR_TIP_MARGIN_M: the table
            # wins over a target reading that claims to be inside the wood.
            guard = surface + HAND_FLOOR_TIP_MARGIN_M
            bound = clamped < guard
            clamped = max(clamped, guard)
            print(f"[scene] NOTE: the work surface at {surface:+.4f} is "
                  f"{(target_z - surface) * 1000:.0f} mm below the grasp target "
                  f"at {target_z:+.4f} — normal for a small object resting on a "
                  f"table, and NOT a bad fit. Adding the "
                  f"{HAND_FLOOR_MARGIN_M * 1000:.0f} mm gripper-sphere margin "
                  f"would leave the floor within {HAND_FLOOR_SLACK_M * 1000:.0f} "
                  f"mm of the grasp pose, where its 1e6 weight fights the reach, "
                  f"so the floor is clamped to {clamped:+.4f}"
                  + (f" — held at the surface +"
                     f"{HAND_FLOOR_TIP_MARGIN_M * 1000:.0f} mm tip guard, the "
                     f"target cap wanted lower" if bound else "")
                  + ". The table is a height sensor only (08-13), not a planner obstacle; this floor is its protection.")
        return clamped
    return floor


def check_hand_floor(target_z: float, server_floor: float | None,
                     surface_z: float | None = None) -> str | None:
    """Is the server's hand z-floor compatible with the reach we are about to
    ask for? Returns a message to print, or None.

    THE FLOOR IS ON BY DEFAULT AND THE DEFAULT IS AN ASSUMPTION. cuRobo's
    build_curobo_planner applies hand_z_floor=+0.060 m at weight 1e6 unless told
    otherwise, and HUMANOID_CFG's empty planner_kwargs does not turn it off — it
    accepts it. The pelvis (base origin) stands roughly 0.7 m up, so +0.060 m is
    around the height of a 0.75 m lab table: exactly where "put the cube on the
    table" lands. Below that line a 1e6-weight cost fights the reach, and the
    symptom is an INFEASIBLE verdict or a hand hovering above the cube — never a
    message about a floor. That is why this check exists.

    Kept a pure function of three numbers so the branches can be tested without
    a GPU, a server, or a robot.
    """
    if server_floor is None:
        return None                            # explicitly disabled: nothing to check
    if target_z < server_floor + HAND_FLOOR_SLACK_M:
        surf = "" if surface_z is None else \
            f" (work surface top {surface_z:+.4f})"
        return (f"the plan server's hand z-floor is {server_floor:+.4f} m but the "
                f"grasp target is at z={target_z:+.4f}{surf} — the floor is AT OR "
                f"ABOVE the target, so a 1e6-weight cost is pushing the gripper "
                f"spheres away from the very pose we are asking for. Expect "
                f"INFEASIBLE, or a hand that stops short. Restart the server "
                f"with\n        --hand-z-floor "
                f"{(surface_z if surface_z is not None else target_z) + HAND_FLOOR_MARGIN_M:.4f}"
                f"\n    or --no-hand-z-floor for a genuinely floor-free reach.")
    return None


def _report_hand_floor(args: Args, rig: GazeRig, target: np.ndarray,
                       server_floor: float | None) -> None:
    tops = _table_surfaces(rig, args.tables, target[:2])
    why = check_hand_floor(float(target[2]), server_floor,
                           max(tops) if tops else None)
    if why:
        print(f"[scene] WARNING: {why}")


def refuse(msg: str, code: int) -> int:
    """Print the REFUSED banner and return `code` as main()'s exit status. Moves
    nothing: the caller stops publishing and the robot's silence failsafes own
    the arms and cameras from here. For failures with a cube in hand use
    refuse_holding, which keeps commanding the loaded hand."""
    print(f"\n[REFUSED] {msg}\n[REFUSED] no further packets — the robot's "
          f"silence failsafes now own the arms (0.125 rad/s crawl to the chest "
          f"home, grippers stay latched) and the cameras (0.75 rad/s back to "
          f"the origin)")
    return code


def refuse_holding(msg: str, code: int, pub, tel: TelemetryView,
                   det: DetectionView, rig: GazeRig, rate: float,
                   holding: bool = True, held_sides=()) -> int:
    """Refuse WITHOUT letting go — for failures after the cube is in the hand.

    `held_sides` names WHICH hands carry a cube, and it changes what an arm
    fault means (08-07 audit). Without it this loop released on the first fault
    on EITHER arm — but arm_fault is the dropout watchdog's terminal latch and
    the entry condition for six of run_leg's FATALs, so the very failure that
    routes here re-read its own latch on tick one, published nothing, and handed
    the OTHER arm — healthy, and on a journey the one carrying leg 1's cube — to
    real_env's 0.5 s silence failsafe. With the sides known the rule is exact:
    hold while any HELD hand is still healthy; release once every held hand is
    latched, because a frozen arm is beyond this loop's help. Callers that do
    not know (the standing mission, which cannot have a loaded peer left over
    from a previous leg) keep the old any-fault-releases behaviour, since for
    them the competing risk — holding an EMPTY hand forever, the thing the
    07-30 audit added defence in depth against — is the live one.

    These paths used to call refuse(), whose message said "arm HOLDS at grasp;
    Ctrl+C when ready" while the process returned immediately: publishing
    stopped, the arm-silence failsafe took the arm, and it crawled home still
    gripping. The message promised something the code did not do, which is worse
    than either behaviour on its own — an operator who read it would walk over
    expecting a stationary arm.

    Holding is also the right answer, not just the honest one. The failure modes
    here (a table that went out of view, no gated retract route) leave a loaded
    arm extended into the workspace, and WHEN to let go of a held object is an
    operator's decision, not a timeout's. So: keep republishing the measured
    posture at PUB_HZ until Ctrl+C. An arm fault still ends it — a hold that
    ignores a fault is not a hold.
    """
    print(f"\n[REFUSED] {msg}")
    if not holding:
        # DEFENCE IN DEPTH after the 07-30 audit. This loop is deliberately
        # unbounded — when to let go of a held object is an operator's decision,
        # not a timeout's — which makes it the only place in this program that
        # can hold a healthy robot forever. Reaching it with an EMPTY hand is
        # therefore never right, and the classification upstream is not the only
        # thing that should have to be correct for that to hold.
        print("[REFUSED] nothing is held — releasing to the failsafe rather "
              "than holding an empty hand forever")
        return code
    telemetry = tel.fresh()
    enc = _arm_enc(telemetry) if telemetry else None
    if enc is None:
        print("[REFUSED] no encoders to hold at — the failsafe now owns the arms")
        return code
    print(f"[HOLD] the hand is still closed on the cube. Holding this posture at "
          f"{PUB_HZ:.0f} Hz — take the cube, then Ctrl+C.")
    warned_fault = False
    try:
        while True:
            tick = time.monotonic()
            telemetry = tel.fresh()
            faults = list(telemetry.get("arm_fault") or [False, False]) \
                if telemetry is not None else []
            faults = (faults + [False, False])[:2]
            idx = {"left": 0, "right": 1}
            loaded = [s for s in held_sides if s in idx]
            dead = all(faults[idx[s]] for s in loaded) if loaded \
                else any(faults)
            if any(faults) and dead:
                print("[HOLD] ARM FAULT on every holding arm during the hold "
                      "— releasing to the failsafe")
                return code
            if any(faults) and not warned_fault:
                # RELEASE ONLY WHEN EVERY ARM IS LATCHED (08-07 audit). This
                # used to be any(): arm_fault is the dropout watchdog's TERMINAL
                # latch and the entry condition for six FATALs, so the very
                # failure that routes here re-read its own latch on tick one,
                # published nothing, and handed the OTHER arm — healthy, and by
                # construction the one holding a cube — to real_env's 0.5 s
                # silence failsafe, which crawls it to the default pose at
                # 0.025 rad/s with the cube still in the fingers. A faulted arm
                # is already frozen by the watchdog and cannot be helped by this
                # loop; the healthy loaded one is exactly what the loop is for.
                warned_fault = True
                side = "left" if faults[0] else "right"
                print(f"[HOLD] {side} arm is FAULT-latched, but the other arm "
                      f"is healthy and may be holding — continuing to hold it. "
                      f"Take the cube, then Ctrl+C.")
            det.poll(rig)
            out = rig.tick(telemetry or {}, hold_packet(enc, rate))
            if out:
                pub.publish(out)
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\n[HOLD] Ctrl+C — the failsafe now crawls the arm home")
        return code


def _print_report(tag: str, report: dict) -> None:
    for gate, r in report["gates"].items():
        line = " ".join(f"{k}={v}" for k, v in r.items() if k != "ok")
        print(f"[check] {tag} {gate}: {'OK' if r['ok'] else 'FAIL'}  {line}")


def _check_world(args: Args) -> str | None:
    """Exactly one world declaration, and it has to be a real one. Returns a
    complaint or None. Kept separate from main() so a test can exercise every
    branch without a robot."""
    declared = [bool(args.tables), bool(args.table), args.no_table_world]
    if sum(declared) != 1:
        return ("give exactly one of --tables <names>  |  --table cx cy cz sx sy sz"
                "  |  --no-table-world  (an empty world must be explicit)")
    if args.table and len(args.table) != 6:
        return "--table needs exactly 6 numbers: cx cy cz sx sy sz"
    if not 1 <= len(args.cubes) <= 2:
        return (f"--cubes is the grasp set, one arm each, so 1 or 2 names — "
                f"got {len(args.cubes)}: {list(args.cubes)}")
    if len(set(args.cubes)) != len(args.cubes):
        return (f"--cubes has a repeat: {list(args.cubes)}. Two arms cannot be "
                f"assigned the same cube, and the server's assignment map is "
                f"keyed by side, so the duplicate would silently vanish.")
    unknown = [n for n in args.tables if n not in TABLE_CONFIGS]
    if unknown:
        return (f"unknown table(s) {unknown} — registered: "
                f"{sorted(TABLE_CONFIGS)}. Tables are registered in "
                f"visual_servoing/tagged_bodies/table/, one file per table.")
    return None


@dataclasses.dataclass
class Ctx:
    """Everything a reach needs, so the standing path and a journey leg run the
    SAME code. Duplicating the plan/execute/retract sequence per path is how the
    two would drift, and only one of them gets hardware time."""
    args: Args
    model: object
    tel: TelemetryView
    det: DetectionView
    client: PlanClient
    pub: object
    route_pub: object
    rig: GazeRig
    server_floor: object = None
    nav_pub: object = None
    home_enc: object = None      # the posture every retract returns to, fixed
                                 # for the whole mission (see plan_reach)
    spec: object = None          # SpeculativeSolve launched during startup
                                 # (standing missions only; see solve-on-lock)
    warm_thread: object = None   # journey per-leg background warm solve (user
                                 # 08-10): pays cuRobo's cold costs during the
                                 # walk; joined before any main-thread RPC


DONE, SKIP, FATAL = "done", "skip", "fatal"


@dataclasses.dataclass
class Plan:
    """A gated reach and everything the retract needs to follow it.

    home_enc travels here rather than on the bundle because RouteBundle is
    frozen — deliberately, it is a certificate — and because the retract's
    destination is a property of the MISSION (where this run began), not of the
    route.
    """
    bundle: object
    targets: dict
    floor: object
    home_enc: dict
    # Torso attitude at the instant `targets` was medianed. It rides on the
    # PLAN, not the bundle, because the speculative path can adopt an older
    # bundle while the targets are re-measured — binding the epoch to the wrong
    # one would be worse than having none. mpc_track re-derives plan_anchor
    # from this by one absolute rotation, instead of composing per-tick deltas
    # that no measurement ever heals (audit 2026-08-06, D5/D6).
    gravity: object = None


def reacquire_targets(ctx: Ctx, keys, lost_key: str) -> bool:
    """Hold the arms and wait up to CUBE_REACQUIRE_S for every target to have a
    fresh median again. True when they all do; False on timeout or lost input.

    WHY THIS EXISTS (2026-08-01, hardware). A cube's tags flickered out in the
    gap between the selection loop confirming it and build_scene asking for its
    median — a window measured in hundreds of milliseconds — and the mission
    ended with a REFUSED. The median needs >=3 sightings inside
    CUBE_LATCH_WINDOW_S (0.7 s), so even a two-second occlusion (an arm
    crossing a camera, a gimbal saccade, tag glare) empties it transiently.
    That is weather, not failure: stand still, keep the gaze on it, give it
    CUBE_REACQUIRE_S to come back, and only then charge a plan round.

    The arms hold at their MEASURED posture (snapshot semantics, same rule as
    pump_during): this wait must not itself become the silent window that
    hands them to the failsafe crawl.
    """
    telemetry = ctx.tel.fresh()
    enc = _arm_enc(telemetry) if telemetry else None
    if enc is None:
        return False
    print(f"[scene] {lost_key} lost between selection and the scene build — "
          f"holding {CUBE_REACQUIRE_S:.0f}s for it to be re-seen …")
    t0 = time.monotonic()
    while time.monotonic() - t0 < CUBE_REACQUIRE_S:
        tick = time.monotonic()
        telemetry = ctx.tel.fresh()
        if telemetry is None:
            return False
        if any(telemetry.get("arm_fault") or [False, False]):
            return False
        ctx.det.poll(ctx.rig)
        out = ctx.rig.tick(telemetry, hold_packet(enc, ctx.args.max_rate))
        if out:
            ctx.pub.publish(out)
        now = time.monotonic()
        if all(ctx.rig.median(k, now) is not None for k in keys):
            print(f"[scene] {lost_key} re-seen after "
                  f"{now - t0:.1f}s — resuming")
            return True
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    print(f"[scene] {lost_key} still unseen after {CUBE_REACQUIRE_S:.0f}s")
    return False


SPEC_TARGET_STALE_M = 0.06        # adopt the speculative route if every
                                  # target's fresh median is within this of
                                  # the snapshot it was solved against.
                                  # 20 mm (the first bar) discarded a WINNING
                                  # in-zeroing solve on hardware 2026-08-03:
                                  # 1-face PnP noise alone wanders 30-70 mm
                                  # at r_xy ~0.5, so the lottery ticket was
                                  # thrown away almost every time. 60 mm is
                                  # safe because our C gate sweeps ROBOT
                                  # self-collision only (cubes are not in the
                                  # MJCF), a 6 cm cube-cuboid offset is
                                  # negligible for cross-arm avoidance at
                                  # ~0.7 m separation, and terminal accuracy
                                  # is owned by the MPC tracker (live goal
                                  # updates + the 9 cm drift abort)
SPEC_SEED_STALE_RAD = 0.03        # ... and only if the live encoders are this
                                  # close to the solve's seed — inside Gate B's
                                  # 0.05 rad with margin, and
                                  # recheck_route_start re-proves the same
                                  # bound before anything streams
SPEC_PLAN_RETRIES = 12            # the speculation's retry budget. Its time is
                                  # FREE (it burns the zeroing window), and
                                  # deep placements make the wind-up family a
                                  # seed lottery (2026-08-03: 9 F/C rejections
                                  # across 12 solves at r_xy 0.51-0.54) —
                                  # more tickets in the free window, so the
                                  # paid window rarely needs any


class SpeculativeSolve:
    """Solve-on-lock (user 2026-08-03: 'start computing the waypoints the
    instant it locks'). The zeroing/chest-home loops maybe_launch() this at ~20 Hz;
    the moment EVERY target cube has an in-envelope median, it snapshots the
    targets and the measured seed, builds the scene, and runs the whole
    solve-and-gate pipeline (plan_until_gated via the injected solve_fn) in a
    background thread — overlapping the 4-12 s the operator already spends
    watching calibration. plan_reach later ADOPTS the result only if it is
    still true (targets within SPEC_TARGET_STALE_M, seed within
    SPEC_SEED_STALE_RAD); anything stale is discarded and the normal pumped
    solve runs, so a wasted speculation costs only GPU time that was idle.

    THREAD CONTRACT. The PlanClient REQ socket is lockstep: the phases that
    maybe_launch() this object make no RPCs, and plan_reach joins the thread (pumped,
    arms held) before any main-thread RPC — the socket never has two users.
    verify_route builds its own MjData and MjModel is read-only, so gating
    off-thread is safe. A non-transport crash in the thread prints its own
    traceback (threading excepthook) and the mission simply solves normally.

    solve_fn(scene, targets, enc) -> (bundle, report); scene_builder() ->
    (scene, targets) — injected so tests pin the lifecycle without a GPU.

    `warmup` (optional, () -> None) runs on its own thread IMMEDIATELY at
    construction: cuRobo pays a ~3.5 s CUDA-graph capture on the FIRST solve
    of each problem shape per server life (server log 2026-08-03: reach
    gen=1 3.5-3.6 s, every later reach 0.11-0.13 s), and the GPU died three
    times today so every run's first solve paid it again — which read as
    'cuRobo got slow'. A throwaway same-shape solve during zeroing hides
    that cost where nobody waits on it. The `_rpc` lock serializes the
    warmup and the real solve on the lockstep REQ socket."""

    def __init__(self, rig: GazeRig, tel: TelemetryView, cube_keys, args,
                 solve_fn, scene_builder=None, warmup=None):
        self.rig, self.tel = rig, tel
        self.keys = tuple(cube_keys)
        self.solve_fn = solve_fn
        self.scene_builder = scene_builder if scene_builder is not None else \
            (lambda: build_scene(args, rig, self.keys, latched={}))
        self.thread: threading.Thread | None = None
        self.targets: dict | None = None
        self.enc: dict | None = None
        self.result = None
        self.launched_t: float | None = None
        self._rpc = threading.Lock()
        self._warm: threading.Thread | None = None
        if warmup is not None:
            def warm_run():
                try:
                    with self._rpc:
                        warmup()
                except PlanServerError as e:
                    print(f"[plan] solver warm-up failed harmlessly: {e}")
            self._warm = threading.Thread(target=warm_run, daemon=True,
                                          name="solver_warmup")
            self._warm.start()

    def busy(self) -> bool:
        """A thread of ours is alive — the caller must join_all (pumped)
        before any main-thread RPC touches the shared REQ socket."""
        return (self._warm is not None and self._warm.is_alive()) or \
            (self.thread is not None and self.thread.is_alive())

    def join_all(self) -> None:
        if self._warm is not None:
            self._warm.join()
        if self.thread is not None:
            self.thread.join()

    def maybe_launch(self, now: float) -> None:
        """Launch once, the first tick every cube is selected. Cheap when not
        ready (median lookups) and a no-op after launch."""
        if self.thread is not None:
            return
        for k in self.keys:
            med = self.rig.median(k, now)
            if med is None or target_unreachable_why(np.asarray(med)) is not None:
                return
        telemetry = self.tel.fresh()
        enc = _arm_enc(telemetry) if telemetry else None
        if enc is None:
            return
        try:
            scene, targets = self.scene_builder()
        except (TargetLost, TableNotSeen, TablePoseUnstable):
            return                      # flicker this instant; next tick retries
        self.targets = {k: np.asarray(p, float) for k, p in targets.items()}
        self.enc, self.launched_t = dict(enc), now
        print("[plan] speculative solve launched — every cube locked, "
              "overlapping the startup phases")

        def run():
            try:
                with self._rpc:
                    self.result = self.solve_fn(scene, self.targets, self.enc)
            except PlanServerError as e:
                self.result = None
                print(f"[plan] speculative solve failed harmlessly: {e}")

        self.thread = threading.Thread(target=run, daemon=True,
                                       name="speculative_solve")
        self.thread.start()

    def adopt(self, fresh_targets: dict, enc_now: dict):
        """(bundle, report) if the speculation still describes the world, else
        None. Caller must have JOINED the thread (pumped). One-shot."""
        if self.thread is None or self.thread.is_alive():
            return None
        result, self.thread = self.result, None
        if not result or result[0] is None:
            return None
        for k, p in self.targets.items():
            fp = fresh_targets.get(k)
            if fp is None or \
                    float(np.linalg.norm(np.asarray(fp, float) - p)) > SPEC_TARGET_STALE_M:
                print(f"[plan] speculative route discarded: {k} moved past "
                      f"{SPEC_TARGET_STALE_M * 1000:.0f} mm since the solve")
                return None
        worst = max(abs(self.enc[n] - enc_now[n]) for n in ARM_JOINTS_ALL)
        if worst > SPEC_SEED_STALE_RAD:
            print(f"[plan] speculative route discarded: arms {worst:.3f} rad "
                  f"from the solve's seed (bar {SPEC_SEED_STALE_RAD})")
            return None
        print(f"[plan] speculative route ADOPTED — solved during startup, "
              f"{time.monotonic() - self.launched_t:.1f}s ago; zero wait here")
        return result


def hover_targets(targets: dict, hover_m: float) -> dict:
    """Solver-facing copies of the grasp targets, lifted `hover_m` straight
    up. The TRUE targets stay untouched everywhere else — MPC goals, drift
    watches and the leash anchor must keep pointing at the cube itself."""
    if hover_m <= 0.0:
        return targets
    return {k: np.asarray(p, float) + np.array([0.0, 0.0, hover_m])
            for k, p in targets.items()}


def plan_with_reacquire(ctx: Ctx, target_keys, table_latch: dict, home_enc,
                        only_sides=None):
    """plan_reach, with tag flicker handled WITHOUT charging a plan round.

    Hardware 2026-08-01, round 3 of 3: a cube vanished at the scene build, the
    re-acquire wait got it back in 1.0 s — and the round loop then `continue`d,
    which ENDED the mission with "3 plan rounds exhausted" without ever
    planning again. A successful re-acquire meant the same thing as a failure.
    Rounds exist to bound REPEATED PLANNING at a target that never works out;
    weather is not planning, so it retries here, inside one round, bounded to
    one successful re-acquire per call (a target that keeps flickering still
    fails the round rather than looping).
    """
    for attempt in range(2):
        try:
            return plan_reach(ctx, target_keys, table_latch, home_enc,
                              only_sides=only_sides)
        except TargetLost as e:
            if attempt == 1 or not reacquire_targets(ctx, target_keys, e.name):
                return None, (f"{e.name} lost at the scene build and not "
                              f"re-seen within {CUBE_REACQUIRE_S:.0f}s")
    return None, "unreachable"  # loop structure guarantees a return above


def plan_reach(ctx: Ctx, target_keys, table_latch: dict, home_enc=None,
               only_sides=None):
    """(Plan | None, reason). None when no route was gated; `reason` then says
    why, and the caller decides whether that ends a mission (standing) or skips
    a leg (journey).

    `only_sides` restricts the solve to the named hands ("left"/"right"). The
    planner has no idea a gripper is already carrying something — `rig._held`
    was never read anywhere in this path before the 08-07 audit — so the caller
    that knows must say so, or leg 2 can be routed onto the loaded hand.

    `home_enc` is the MISSION's start posture — where the retract must return
    to — and the caller owns it. It used to be re-sampled here from the live
    encoders on every call, which is right exactly once: on round 2 the live
    encoders read wherever round 1's failure left the arm (out at the cube), so
    the retract's "go home" would have gone to the failed grasp pose. That
    silently undid fcb7d8b, whose whole point was that the retract returns to
    where the run BEGAN. Absent, it falls back to the live reading — the
    single-round callers that have no other notion of home.
    """
    a, rig = ctx.args, ctx.rig
    # TargetLost PROPAGATES (2026-08-01). It used to be flattened to a string
    # here, which made "the tag flickered for two seconds" indistinguishable
    # from "no route exists" — and the round loops answered both with the end
    # of the mission. The table failures stay caught: a declared table that
    # cannot be seen is structural, not a flicker.
    try:
        scene, targets = build_scene(a, rig, target_keys, latched=table_latch,
                                     optional_tables=getattr(a, 'journey', False))
    except (TableNotSeen, TablePoseUnstable) as e:
        return None, str(e)
    # The attitude these targets were measured AT. Captured here rather than in
    # mpc_track because the interval between the two — solve, route stream, up
    # to 10 s of settle, a multi-second GPU warm, arms at full stretch — is the
    # most attitude-active stretch of the mission, and at 12-16 mm/deg about
    # 2.5 deg of it fully consumes MPC_BLIND_ANCHOR_M.
    plan_g = _grav_unit(ctx.tel.fresh() or {})
    print(f"[scene] {len(scene)} cuboid(s): {list(scene)}")
    # LOWEST target sets the floor. The clamp exists so a floor at or above a
    # grasp pose cannot veto the reach, and with two cubes at different heights
    # only the lower one can be vetoed — a floor cleared for a 60 mm cube on a
    # riser would sit above a 40 mm cube lying flat.
    low_p = min((np.asarray(p, float) for p in targets.values()),
                key=lambda p: float(p[2]))
    low_z = float(low_p[2])
    floor = hand_floor_for_tables(rig, a.tables, target_z=low_z,
                                  target_xy=low_p[:2])
    if floor is not None:
        print(f"[scene] hand z-floor {floor:+.4f} m from the registered "
              f"table(s) (surface +{HAND_FLOOR_MARGIN_M} sphere margin; cap "
              f"{HAND_FLOOR_TARGET_GAP_M * 1000:.0f} mm under the LOWEST "
              f"target at {low_z:+.4f}, never below the surface "
              f"+{HAND_FLOOR_TIP_MARGIN_M * 1000:.0f} mm tip guard) — sent "
              f"with the plan")
    for p in targets.values():
        _report_hand_floor(a, rig, p,
                           floor if floor is not None else ctx.server_floor)
    telemetry = ctx.tel.fresh()
    enc = _arm_enc(telemetry) if telemetry else None
    if enc is None:
        return None, "no arm encoders — cannot hold the arms while the solver works"
    # SEED from where the arm IS (every round), RETURN to where the mission
    # began (fixed) — two different postures once a round has failed.
    home = home_enc if home_enc is not None else enc
    floor_m = a.clearance_floor_mm / 1000.0
    # HOVER ENDPOINT (user 2026-08-03): the solver aims 3 cm above each grasp
    # point; the MPC tracker owns the descent onto the TRUE target. Only with
    # the MPC — without it the gripper would close at the hover. The floor
    # above was computed from the TRUE targets on purpose: it must stay under
    # the GRASP point or the descent itself would be cost-blocked.
    hover_m = a.reach_hover_mm / 1000.0 if a.mpc else 0.0
    solve_targets = hover_targets(targets, hover_m)
    if hover_m > 0.0:
        print(f"[plan] route endpoints set {a.reach_hover_mm:.0f} mm ABOVE "
              f"each grasp point — the MPC tracker descends the last leg")
    elif a.reach_hover_mm > 0.0:
        print("[plan] NOTE: --reach-hover-mm is ignored without the MPC "
              "(the gripper would close at the hover)")
    # SOLVE-ON-LOCK adoption: a route already solved AND gated during the
    # startup phases skips the 4-10 s wait entirely — if it still describes
    # the world. A stale one falls through to the normal solve without
    # charging anything.
    bundle = report = None
    spec = ctx.spec
    if spec is not None:
        if spec.busy():
            pump_during(spec.join_all, ctx.pub, ctx.tel, ctx.det, rig,
                        enc, a.max_rate, "the speculative solve")
        if spec.thread is not None:
            adopted = spec.adopt(targets, enc)
            if adopted is not None:
                bundle, report = adopted
    try:
        # WRAPPED, never called directly: a blocking solve is a silent commander
        # and 0.5 s of silence starts the failsafe crawl (see pump_during).
        if bundle is None and a.decoupled_arms and len(solve_targets) == 2:
            bundle, report = pump_during(
                lambda: plan_decoupled_until_gated(
                    ctx.client, scene, solve_targets, measured_enc=enc,
                    hand_z_floor=floor, model=ctx.model,
                    max_rate=a.max_rate, clearance_floor_m=floor_m),
                ctx.pub, ctx.tel, ctx.det, rig, enc, a.max_rate,
                "the decoupled solves")
            if bundle is not None:
                print("[plan] DECOUPLED routes merged and gated — "
                      "simultaneous execution, no joint-solve coupling")
            else:
                print("[plan] decoupled pair could not be gated — falling "
                      "back to the JOINT solve")
        if bundle is None:
            # AUTO is the server's own pick and is usually right — but it is
            # blind to which hands are FREE. With a cube already in one hand
            # (journey leg 2) AUTO can hand the new cube to the loaded arm, so
            # when the caller restricts us we force the assignment instead of
            # asking and hoping (08-07 audit).
            auto_assign = None
            if only_sides is not None and len(solve_targets) == 1:
                side = {"left": "L", "right": "R"}[tuple(only_sides)[0]]
                auto_assign = {side: tuple(solve_targets)[0]}
                print(f"[plan] restricted to the free hand(s) "
                      f"{'+'.join(sorted(only_sides))} — forcing "
                      f"{side}->{tuple(solve_targets)[0]}")
            bundle, report = pump_during(
                lambda: plan_until_gated(
                    ctx.client, scene, solve_targets, measured_enc=enc,
                    goal="reach", assignment=auto_assign, hand_z_floor=floor,
                    model=ctx.model,
                    max_rate=a.max_rate, clearance_floor_m=floor_m),
                ctx.pub, ctx.tel, ctx.det, rig, enc, a.max_rate,
                "the reach solve")
        if bundle is None and a.other_arm_fallback:
            for forced in forced_assignments(target_keys, only_sides):
                label = ", ".join(f"{s}->{c}" for s, c in forced.items())
                print(f"[plan] no gated route with the auto assignment — "
                      f"forcing {label} …")
                bundle, report = pump_during(
                    lambda s=forced: plan_until_gated(
                        ctx.client, scene, solve_targets, measured_enc=enc,
                        goal="reach", assignment=s, hand_z_floor=floor,
                        model=ctx.model, max_rate=a.max_rate,
                        clearance_floor_m=floor_m),
                    ctx.pub, ctx.tel, ctx.det, rig, enc, a.max_rate,
                    f"the {label} solve")
                if bundle is not None:
                    break
    except PlanServerError as e:
        return None, f"plan server failed: {e}"
    if bundle is None:
        if report is not None:
            _print_report("reach (last attempt)", report)
        return None, (
            f"no route passed our gates in {PLAN_RETRIES} attempts per arm — "
            f"cuRobo can reach the target but not with the "
            f"{a.clearance_floor_mm:.0f} mm clearance we require"
            + (f" (note: the solver was aimed {a.reach_hover_mm:.0f} mm ABOVE "
               f"each grasp point; a cube near the top of the envelope or the "
               f"arm's reach can be INFEASIBLE at the hover while the grasp "
               f"itself is reachable — bring it closer/lower, or "
               f"--reach-hover-mm 0" if hover_m > 0.0 else ""))
    print(f"[plan] route: {bundle.horizon} waypoints, {bundle.duration_s:.1f}s "
          f"at <= {a.max_rate} rad/s (solve {bundle.solve_s:.2f}s), "
          f"assignment {bundle.assignment}, planned reach_err "
          f"{max(r['err'] for r in bundle.reaches.values()):.4f}")
    _print_report("reach", report)
    why = recheck_route_start(bundle, ctx.tel)
    if why:
        return None, why
    ctx.route_pub.publish(route_preview_payload(bundle, ctx.model))
    print("[route] preview published — green path in the monitor")
    return Plan(bundle, targets, floor, home, plan_g), None


def _grav_unit(telemetry: dict) -> np.ndarray | None:
    """Unit gravity direction in the base frame, or None if unusable."""
    g = telemetry.get("projected_gravity")
    if g is None or len(g) != 3:
        return None
    v = np.asarray(g, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Rotation matrix taking unit vector a to unit vector b (Rodrigues);
    None when they are already aligned (or opposed — a 180 deg base flip is
    not a tilt and must not be 'compensated')."""
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(a, b))
    if s < 1e-9:
        return None
    k = axis / s
    kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    theta = math.atan2(s, c)
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def mpc_track(ctx: Ctx, bundle, targets: dict, table_latch: dict,
              meas_hist: list | None = None, plan_gravity=None,
              settled: bool = True):
    """Track each reaching hand onto the LIVE median of its own cube until it
    measurably arrives. Returns (why, hold_enc): why None on success, and
    hold_enc the measured arm posture at convergence — the posture the grasp
    must hold, which is NOT the route's final waypoint once the cube has been
    followed.

    `meas_hist`, when given, collects the MEASURED arm postures (a
    {joint: rad} dict per entry, both arms, deduped) in order — the path the
    robot ACTUALLY travelled. The caller replays it in reverse to walk back
    out of an aborted track (see retreat_after_track_abort). Measured, not
    commanded, deliberately (hardware 2026-08-01 night): a wedged arm never
    followed its commands, and reversing the COMMANDED route yanked it
    through free space to a route posture it had never reached — the
    "one arm suddenly lifts high" the user watched before every retreat.
    Reversing measured postures traverses only configurations the robot
    physically occupied this round.

    WHY THIS EXISTS (2026-08-01, the user's own diagnosis): the green line's
    endpoint is a snapshot. The route is planned against a 0.7 s median and
    played open-loop; every source of divergence after that instant — torso
    sway moving the base frame, detection refinement, the cube actually being
    nudged — lands as grasp error. This loop closes it: plan-0 still owns the
    path (gated whole, streamed to MPC_HANDOFF_FRACTION), the MPC session owns
    only the final approach, re-aimed at ~20 Hz.

    THE SAFETY CONTRACT CHANGES SHAPE HERE, deliberately: verify_route's
    "whole route gated before one packet" cannot hold for a command produced
    one tick before it executes, so every commanded configuration passes the
    per-tick StepGate (limits / rate cap / measured->commanded segment
    clearance on OUR model) and a rejected command is simply not published —
    the arm keeps its last accepted target and the failure COUNTS. Enough
    consecutive unusable ticks abort the track rather than pushing on.

    EVIDENCE RULES (softened 2026-08-01 at the user's direction): >= 2 tag
    faces moves the goal freely; ONE face moves it too, but only within
    MPC_LOWEV_LEASH_M of the scan-time plan anchor — 1-face PnP carries the
    tilt-drift signal (today's dominant error) yet cannot be trusted off its
    leash (the 5 cm ghost class). Since 2026-08-10 a 1-face-only stream is
    additionally STABILIZED at the median layer: GazeRig._best_window widens
    the median to LOWEV_STABLE_WINDOW_S so the goal sits on the stream's
    stable central position instead of chasing each slide (the e-cube T09
    run: 44-51 mm of 1-face drift dragged the goal 40 mm and the re-solves
    went unstable). All goal motion is slewed at MPC_GOAL_SLEW_M_S so hops
    reach the solver as glides. A goal that leaves the reach envelope aborts.
    """
    a, rig = ctx.args, ctx.rig
    cube_of = {s: c for s, c in bundle.assignment.items() if c in targets}
    tracked = sorted(cube_of)                     # "L"/"R"
    side_name = {"L": "left", "R": "right"}
    if not tracked:
        return Abort("S01", "MPC session setup", "route assignment",
                     f"MPC track: no tracked side matches the targets "
                     f"(assignment {bundle.assignment}, targets "
                     f"{sorted(targets)})"), None

    telemetry = ctx.tel.fresh()
    enc = _arm_enc(telemetry) if telemetry else None
    if enc is None:
        return Abort("S02", "MPC session setup", "telemetry link",
                     "no arm encoders at the MPC handoff"), None

    # 1-face z snap license (08-13): which of THESE anchors carry the
    # table-resting prior. Snapshotted BEFORE the session build_scene below —
    # that rebuild restashes rig._z_prior_keys from whatever tables happen to
    # be decodable at the handoff, and the reaching arm occluding the TABLE
    # tags must not revoke a license the anchors already own (the table has
    # not moved; its tags are merely hidden). The license travels with the
    # `targets` parameter, whose z it describes.
    z_snap_keys = frozenset(getattr(rig, "_z_prior_keys", ()))

    # The MPC scene is the retract's scene: every cube being grasped leaves
    # the obstacle set (it is the goal; as an obstacle it wedges the hand).
    try:
        scene, _ = build_scene(a, rig, (), exclude=tuple(targets),
                               latched=table_latch,
                               optional_tables=getattr(a, 'journey', False))
    except (TableNotSeen, TablePoseUnstable) as e:
        return Abort("S03", "MPC session setup", "table detection",
                     f"MPC track: {e}"), None
    # POST-PLAN, THE TABLE IS A HEIGHT SENSOR, NOT A FORCE FIELD (user
    # 08-13): plan-0 already certified the route against the slab; the MPC's
    # whole job is the last centimetres onto a goal that sits ABOVE the
    # surface by construction (cube centre = surface + half-cube, via the z
    # prior). Carrying the slab into the session put its 70 mm collision-
    # activation band directly under the grasp point, and the descent
    # equilibrated against it 13-20 mm high — the hand dithered there until
    # the T14 budget closed anyway (08-13 run, 25 s burned). The table tags'
    # only post-plan role is the vertical: z prior + 1-face z snap. The
    # server-side hand z-floor stays as the dive-catcher.
    dropped = [n for n in (*getattr(a, "tables", ()), "table")
               if scene.pop(n, None) is not None]
    if dropped:
        print(f"[mpc] {'+'.join(dropped)} removed from the MPC world — "
              f"post-plan the table serves height only (z prior); the slab "
              f"was plan-0's constraint, not the tracker's")

    now = time.monotonic()
    goal_pos = {}
    for s in tracked:
        # The SEED obeys the same evidence bar as every later update — found
        # on hardware 2026-08-01: a cube seen at ONE tag face for the whole
        # stream has a median built from garbage PnP, and seeding from it
        # aimed the tracker ~5 cm off while the evidence gate then (correctly)
        # refused every update, freezing the goal ON the garbage. The plan
        # target came from confirmed selection medians; below the bar it is
        # the better truth.
        med = rig.median(cube_of[s], now)
        if med is not None and rig.faces_used(cube_of[s], now) >= DRIFT_MIN_INLIERS:
            goal_pos[s] = np.asarray(med, float)
        else:
            goal_pos[s] = np.asarray(targets[cube_of[s]], float)
    # The scan-time plan target is the leash anchor for 1-face updates: it was
    # measured before the hand could occlude anything, and 1-face PnP may
    # correct AROUND it (tilt drift) but never abandon it.
    #
    # KEPT AS THE ORIGINAL POINT PLUS ITS EPOCH, and re-derived by ONE ABSOLUTE
    # rotation every tick (audit 2026-08-06, D5/D6). It used to be rotated in
    # place by the tick-to-tick gravity delta, which is wrong twice over:
    #   * a product of minimal rotations is not the minimal rotation of the
    #     product, so a base that sways in 2-D (circling rather than planar)
    #     leaves a residual yaw about gravity equal to the swept solid angle —
    #     17/29/59 mm over a 25 s budget at 0.3/0.5/1.0 Hz for the 4.4 deg span
    #     this robot actually shows;
    #   * unlike goal_pos, plan_anchor is never re-measured, so nothing heals
    #     it. It walks off the 40 mm blind-grasp bar and biases the 120 mm
    #     leash for the whole track.
    # An absolute rotation from a stored epoch has no accumulation term at all:
    # return the base to where it started and the anchor returns exactly.
    anchor0 = {s: np.asarray(targets[cube_of[s]], float) for s in tracked}
    plan_anchor = {s: anchor0[s].copy() for s in tracked}
    # Anchor-epoch yaw (08-12 late night, see _yaw_rot): the delta between the
    # rig's running gyro-z integral and the mark build_scene stamped when these
    # targets were measured. A rig without the fields (tests' fakes) or a mark
    # that never got stamped degrades to zero correction.
    yaw_mark = getattr(rig, "_anchor_yaw_mark", None)
    if yaw_mark is None:
        yaw_mark = getattr(rig, "yaw_integral", 0.0)
    yaw_prev = getattr(rig, "yaw_integral", 0.0)
    # The attitude at which `targets` was MEASURED — inside plan_reach, before
    # the solve, the whole route stream, up to 10 s of settle and the multi-
    # second GPU warm, i.e. across the most attitude-active interval of the
    # mission with the arms at full stretch. Seeding the epoch from the
    # post-warm telemetry instead (what the old code effectively did) discards
    # every degree of that, and at 12-16 mm/deg about 2.5 deg fully consumes
    # MPC_BLIND_ANCHOR_M.
    g_anchor = plan_gravity if plan_gravity is not None else _grav_unit(telemetry)
    # STAGED HOVER (user 2026-08-03): phase 1 chases the point reach_hover_mm
    # ABOVE the live cube, phase 2 (per side, once the hand arrives there)
    # chases the cube itself. Everything else — leash, drift watches,
    # envelope checks, convergence, grasp-anyway — stays on the TRUE goal.
    hover_m = ctx.args.reach_hover_mm / 1000.0
    hover_vec = np.array([0.0, 0.0, hover_m])
    descending = {s: hover_m <= 0.0 for s in tracked}
    desc_off = {s: hover_m for s in tracked}   # z offset still commanded; ramps
                                               # to 0 during phase 2
    last2face: dict = {}      # side -> (monotonic, raw >=2-face median): the
                              # descent handover re-anchors onto this fix
    # Everything the loop needs is built BEFORE the session opens, so the
    # try/finally below can enclose mpc_start itself: an exception (or Ctrl+C)
    # anywhere after the session exists reaches mpc_stop. (Review 2026-08-01:
    # setup living between mpc_start and the old try leaked the session.)
    # The gate's rate check judges the same lead budget the clamp below
    # grants: a command may lead the measurement by MPC_LEAD_SCALE ticks
    # (executed speed is still receiver-capped), so judging it at 1x would
    # reject every deliberately-leading command as a jump.
    gate = StepGate(model=ctx.model,
                    clearance_floor_m=a.clearance_floor_mm / 1000.0,
                    max_rate_rad_s=MPC_LEAD_SCALE * STEP_MAX_RATE_RAD_S)
    data = mujoco.MjData(ctx.model)
    jadr = {n: ctx.model.jnt_qposadr[
        mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ARM_JOINTS_ALL}
    sid = {s: mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE,
                                f"end_effector_{s}_site") for s in tracked}

    def measured_ee(enc_now: dict) -> dict:
        data.qpos[:] = ctx.model.qpos0
        for n in ARM_JOINTS_ALL:
            data.qpos[jadr[n]] = enc_now[n]
        mujoco.mj_kinematics(ctx.model, data)
        return {s: data.site_xpos[sid[s]].copy() for s in tracked}

    def z_ledger(label: str) -> None:
        """THE Z LEDGER (08-13, user): one line per closing hand with every
        height in the chain, printed at the instant a close is decided, so
        the next "grasped high" session starts from data instead of a
        re-derivation. How to read it: vision-vs-anchor is what the tag
        stream claimed (perception); EE-vs-goal is what the arm delivered
        (execution); the table surface is the physics the prior asserted;
        the grasp-line dial says where the JAWS are designed to sit relative
        to the anchor. Base-frame metres, mm for the deltas."""
        t_l = ctx.tel.fresh()
        e_l = _arm_enc(t_l) if t_l else None
        if e_l is None:
            print(f"[z] close[{label}]: no telemetry for the ledger")
            return
        ee_l = measured_ee(e_l)
        now_l = time.monotonic()
        for s in tracked:
            key = cube_of[s]
            med_l = rig.median(key, now_l)
            if med_l is None:
                vis = "vision n/a (blind)"
            else:
                vis = (f"vision {float(np.asarray(med_l, float)[2]):+.4f} "
                       f"({rig.faces_used(key, now_l)}-face)")
            if key in z_snap_keys:
                surf = (f"table surf "
                        f"{plan_anchor[s][2] - _cube_dims(key) / 2.0:+.4f} "
                        f"(prior ON)")
            else:
                surf = "table prior OFF"
            print(f"[z] close[{label}] {key}/{side_name[s]}: "
                  f"EE {ee_l[s][2]:+.4f} | goal {goal_pos[s][2]:+.4f} "
                  f"(dz {(ee_l[s][2] - goal_pos[s][2]) * 1000:+.0f} mm) | "
                  f"anchor {plan_anchor[s][2]:+.4f} | {vis} | {surf} | "
                  f"grasp line {GRASP_ANCHOR_ABOVE_M * 1000:+.0f} mm vs "
                  f"centre")

    # ALREADY BLIND AT THE HANDOFF -> NEVER OPEN THE SESSION (user 2026-08-06,
    # third time of asking: "It should grasp at [exec] !!! why are you
    # waiting?"). The tracker's entire job is correcting what the cube did
    # during the open-loop stream, and it cannot do that for a cube nothing has
    # seen since before the settle. Building the session anyway costs a 0.6 s
    # GPU cold start with the arms extended, then hands a blind tracker an arm
    # it can only move on IMU dead reckoning — the in-loop rule would end it
    # one tick later regardless. Deciding BEFORE mpc_start is the difference
    # between "grasp where the gated route left you" and "build a solver, warm
    # it, and then admit there was nothing to solve".
    #
    # No anchor bar here, unlike the in-loop rule. The distinction is real: at
    # this instant the arm is wherever the six-gate-verified plan-0 route put
    # it, aimed at a scan-time fix taken with the camera parked (this round:
    # planned reach_err 0.0000), and NOTHING has chased a drifting goal yet.
    # That posture is the 07-30 hardware-validated open-loop grasp. Mid-track
    # the same cannot be said, which is why MPC_BLIND_ANCHOR_M guards there.
    # The anchor distances are printed either way, so the log always says how
    # good the close was.
    # ...AND ONLY WHEN NO DESCENT IS STILL OWED. hover_m > 0 means the gated
    # route deliberately ends reach_hover_mm ABOVE the grasp point, so the arm
    # is parked on the hover rail, not on the cube: closing there is
    #   cube centre + GRASP_Z_ABOVE_M (10 mm) + hover (15 mm) = 25 mm high,
    # and a 60 mm cube's top face is at +30 mm — the jaws bite the top 5 mm, or
    # air. The first version of this rule claimed that pose was "the 07-30
    # hardware-validated open-loop grasp"; that was wrong, because the route is
    # lifted by the hover ONLY when --mpc is on, which is exactly when this rule
    # runs (audit 2026-08-06, confirmed). The error is invisible to every
    # tolerance we have: MPC_BLIND_ANCHOR_M and GRASP_EE_OK_M are 3-D distances,
    # so a pure 25 mm z offset passes both without being named.
    #
    # With a descent owed we fall through and open the session after all — the
    # in-loop rule below lands the rail first and closes on the cube. The
    # shortcut survives for the direct-reach configuration (--reach-hover-mm 0),
    # where the route endpoint IS the grasp point and there is nothing to land.
    # ...AND ONLY AFTER A SETTLE THAT ACTUALLY CONVERGED. The shortcut's whole
    # justification is that the arm is exactly where the six-gate route put it;
    # a timed-out settle is the case where it provably is not (audit
    # 2026-08-06, D10). Every timed-out round this week handed the tracker
    # ~72 mm, and closing on that without even opening a session would grasp
    # air with no number to show for it.
    if MPC_BLIND_GRASP and settled and hover_m <= 0.0:
        now0 = time.monotonic()
        blind0 = all(rig.median(cube_of[s], now0) is None for s in tracked)
        ee0 = measured_ee(enc)
        # 08-12 late night (user, "why is it STILL grasping high"): the
        # shortcut used to close UNGATED — the one path nearly every front
        # grasp took carried the execution z-high (+10..+29 mm measured all
        # day) and any base rotation since the fix straight into the jaws.
        # Now: rotate the anchors into the CURRENT base frame (gyro yaw
        # integral since the fix epoch — the tilt comp downstream cannot see
        # yaw) and close only when the hand is laterally on the corrected
        # anchor AND not high. Off either bar -> open the session and let
        # its blind machinery press onto the corrected anchor through the
        # z gate. Legacy ungated close survives under --no-anchor-trust.
        _mark0 = getattr(rig, "_anchor_yaw_mark", None)
        if _mark0 is None:
            _mark0 = getattr(rig, "yaw_integral", 0.0)
        _dyaw0 = getattr(rig, "yaw_integral", 0.0) - _mark0
        _Rz0 = _yaw_rot(-_dyaw0)
        anchor_c = {s: _Rz0 @ plan_anchor[s] for s in tracked}
        lat_off = {s: float(np.linalg.norm((ee0[s] - anchor_c[s])[:2]))
                   for s in tracked}
        z_high0 = max(float(ee0[s][2] - anchor_c[s][2]) for s in tracked)
        # HANDOFF CLOSE, TAG OR NO TAG (user 08-13: "lat <= 40, z <= 15 —
        # do NOT open the MPC session, close at once"). The route is
        # six-gate certified onto the scan anchor and the hand is ON it;
        # every failure today came AFTER this instant, from the session
        # itself — its solver's elbow model runs away from near+high goals
        # (T10 standoffs, then the 95 mm cube drag). The session now exists
        # only for hands that arrive genuinely OFF the anchor.
        if max(lat_off.values()) <= MPC_BLIND_ANCHOR_M and \
                z_high0 <= MPC_HANDOFF_CLOSE_Z_MAX_M:
            print(f"[mpc] HANDOFF CLOSE: the route delivered "
                  + ", ".join(f"{side_name[s]} "
                              f"{_lat_vert(ee0[s], anchor_c[s])}"
                              for s in tracked)
                  + f" from the corrected anchor (bars lat "
                  f"{MPC_BLIND_ANCHOR_M*1000:.0f} / z "
                  f"{MPC_HANDOFF_CLOSE_Z_MAX_M*1000:.0f} mm; base yawed "
                  f"{math.degrees(_dyaw0):+.1f} deg since the fix) — "
                  f"closing NOW, no MPC session.")
            z_ledger("handoff-close")
            return None, dict(enc)
        if blind0 and getattr(a, "anchor_trust", True):
            print(f"[mpc] blind at the handoff but OFF the corrected anchor "
                  f"(lat max {max(lat_off.values())*1000:.0f} mm vs "
                  f"{MPC_BLIND_ANCHOR_M*1000:.0f}, z {z_high0*1000:+.0f} mm vs "
                  f"{MPC_HANDOFF_CLOSE_Z_MAX_M*1000:.0f}; base yawed "
                  f"{math.degrees(_dyaw0):+.1f} deg since the fix) — NOT "
                  f"closing blind; opening the session to press onto the "
                  f"corrected anchor first.")
        elif blind0:
            print(f"[mpc] BLIND AT THE HANDOFF: no tag on "
                  f"{', '.join(cube_of[s] for s in tracked)} for "
                  f"{max(now0 - _last_seen(rig, cube_of[s], now0) for s in tracked):.1f}s"
                  f" — the tracker has nothing to correct and the route ends ON the "
                  f"grasp point (no hover), so the session is never opened. "
                  f"Grasping the gated route's own endpoint ("
                  + ", ".join(f"{side_name[s]} "
                              f"{_lat_vert(ee0[s], anchor_c[s])}"
                              for s in tracked)
                  + f" from the plan anchor; base yawed "
                  f"{math.degrees(_dyaw0):+.1f} deg since the fix, corrected).")
            z_ledger("handoff-shortcut")
            return None, dict(enc)

    watches = {s: MotionWatch(f"mpc track [{cube_of[s]}]",
                              goal_pos[s].copy()) for s in tracked}
    hold = dict(enc)   # frozen handoff posture: the pre-first-command hold AND
                       # the untracked arm's target for the whole track —
                       # snapshot semantics on purpose (see hold_packet)
    last_cmd: dict | None = None
    fails = 0
    cstandoff = 0
    streak = 0
    try:
        # The session build is a blocking multi-second GPU cold start —
        # MPCSolver construction + warm iterations + CUDA-graph capture, fresh
        # on EVERY mpc_start — and it happens with the arms extended at the
        # 80% handoff posture, the worst place to go silent. Pumped for the
        # same reason every plan-0 solve is (see pump_during); found unpumped
        # by review 2026-08-01.
        try:
            rep = pump_during(
                lambda: ctx.client.mpc_start(
                    scene, exclude=tuple(targets), reaching_sides=tracked,
                    measured_enc=enc, cubes_init=goal_pos, grasp_index=-1),
                ctx.pub, ctx.tel, ctx.det, rig, enc, a.max_rate,
                "the MPC session build")
        except PlanServerError as e:
            return Abort("S04", "MPC session setup", "plan server",
                         f"mpc_start failed: {e}"), None
        print(f"[mpc] session up in {rep.get('warm_s', 0.0):.1f}s — tracking "
              f"{'+'.join(side_name[s] for s in tracked)}, candidates "
              f"{rep.get('grasp_index')}")
        # The warm took seconds; the pump held the arms, but "held" is
        # commanded, not measured — re-read the encoders so the first tick's
        # clamp and the hold baseline start from the posture the arm actually
        # kept, not the one the session was seeded with.
        telemetry = ctx.tel.fresh()
        fresh = _arm_enc(telemetry) if telemetry else None
        if fresh is None:
            return Abort("S05", "MPC session setup", "telemetry link",
                         "arm encoders vanished during the MPC warm"), None
        enc = fresh
        hold = dict(enc)
        # HANDOFF DEFICIT — the single number that decides whether this track
        # can converge inside its guards. A correctly tracked route arrives AT
        # the hover: 26 mm mean, 42 mm max over six solves (2026-08-06). The
        # runs that died on T13 started here at 165 and 233 mm, and nothing
        # said so until the progress guard fired six seconds later.
        ee0 = measured_ee(enc)
        print("[mpc] handoff deficit: " + ", ".join(
            f"{side_name[s]} {_lat_vert(ee0[s], goal_pos[s])}"
            for s in tracked)
            + (f" (a tracked route arrives at the hover, "
               f"~{hover_m*1000:.0f} mm)" if hover_m > 0.0 else
               " (a tracked route arrives ON the grasp point — hover 0)"))

        t0 = time.monotonic()
        # SLIDING WINDOW, NOT AN ALL-TIME RATCHET (audit 2026-08-06). The old
        # best_worst only ever went DOWN and was never re-baselined when the
        # GOAL moved, so once the goal receded for any reason — the unslewed
        # descent re-anchor, tilt-compensation residual, a slewed median — no
        # new all-time best could be set and T13 fired 6 s later no matter how
        # fast the hand was closing. The verdict then asserted "hand stuck N mm
        # from its goal for 6s", a claim the guard had never tested. That is why
        # 72 mm -> 178 mm was unreconcilable and why a night of forensics
        # pointed at perception.
        #
        # The window answers the question the name promises: "six seconds ago I
        # was HERE, and I am not MPC_PROGRESS_MIN_M closer now." It re-baselines
        # continuously, so a goal that jumps costs one window, not the track.
        prog_win: deque = deque()
        # THE TICK THIS LOOP ACTUALLY RUNS AT, carried across iterations.
        # PUB_DT is 0.05 s but this tool's own MotionWatch has measured p90 tick
        # times of 130 ms, and the lead clamp is built from the MEASURED dt
        # while every interpreter of that clamp used the nominal one. At 130 ms
        # the real ceiling is 0.395 rad, not the printed 0.152 — so a lag of
        # 0.111 read as 73% of the ceiling when it was 28%, and the
        # "=CEILING, arm NOT executing" tag fired at 31% of the real budget.
        # The T12/T13 differential diagnosis the operator reads was mis-scaled
        # by 2.6x (audit 2026-08-06). The goal slew and the descent ramp had the
        # mirror-image bug: both are m/s rates that were multiplied by the
        # nominal tick, so the goal crept at 19 mm/s instead of 50 and the 15 mm
        # rail took 0.78 s instead of 0.30.
        #
        # Bounded at 2x nominal for the RAMPS: an unbounded dt would commit a
        # 6.5 mm z step on exactly the starved ticks the rail exists to smooth.
        # The CEILINGS take it unbounded — they only interpret a clamp that was
        # already applied with the same number.
        dt_prev = PUB_DT
        # Previous tick's worst hand-to-goal, for the guards that run BEFORE
        # this tick's FK (see T07). Infinity until the first tick measures it,
        # so no guard can excuse itself on evidence that does not exist yet.
        last_worst = float("inf")
        env_strikes = {s: 0 for s in tracked}
        # side -> when its cube last had a live median. Present == blind now.
        blind_since: dict = {}
        g_ref = _grav_unit(telemetry)
        last_goal_pub = 0.0
        leash_note: dict = {}
        land_wait = {"n": 0}
        # >>> ANCHOR TRUST (08-12 night, user doctrine: "when the cuRobo
        # waypoint is generated it should be regarded as reachable"). Once a
        # side is locked, live evidence stops steering it and its goal slews
        # home to the plan anchor — the one point six gates certified. The
        # close then happens through the machinery that already exists
        # (converge bar + z gate, blind-anchor rail). One-way and one-shot:
        # locks never release within a track, and a second conviction of any
        # soft guard while locked falls through to its original abort.
        best_worst = float("inf")   # closest `worst` this track has achieved;
                                    # the ESCAPE GUARD's memory of having
                                    # been inside GRASP_EE_OK
        anchor_lock = {s: False for s in tracked}

        def engage_anchor_trust(code: str, why: str) -> bool:
            """Convert a soft guard's first conviction into an anchor return.

            Returns True when the lock was (newly) engaged — the caller
            withholds its abort. False when anchor trust is off or every side
            is already locked — the caller aborts exactly as it always did,
            which is the bound that keeps a physically stuck arm from
            looping between guards forever."""
            if not getattr(a, "anchor_trust", True) or \
                    all(anchor_lock[s] for s in tracked):
                return False
            for s2 in tracked:
                anchor_lock[s2] = True
            prog_win.clear()          # the return trip IS progress; re-judge fresh
            print(f"[mpc] {code} -> ANCHOR TRUST: {why} — the six-gate route "
                  f"endpoint is reachable by construction, so the goal(s) "
                  f"return to the plan anchor(s) and the close happens there. "
                  f"(A wrong anchor costs one certified-EMPTY pinch, not a "
                  f"retreat; a second conviction aborts for real.)")
            return True
        # <<< ANCHOR TRUST

        def rail_down(who: str) -> bool:
            """May the gripper close now, or is a descent still owed?

            THE JAWS MUST STRADDLE THE CUBE CENTRE, NOT ITS TOP EDGE (user
            2026-08-06). While desc_off > 0 the solver is aimed reach_hover_mm
            ABOVE the grasp point, so closing is
                cube centre + GRASP_Z_ABOVE_M (10 mm) + hover (15 mm) = 25 mm
            high, and a 60 mm cube's top face is at +30 mm. Only the convergence
            exit ever checked this; T12, T13 and T14 all closed on the raised
            rail, and the error is invisible to them because GRASP_EE_OK_M is a
            3-D distance — a pure vertical 25 mm offset passes a 60 mm bar
            without being named (audit 2026-08-06, confirmed).

            descending[] is FORCED rather than waited on: its normal trigger is
            |ee - (goal+hover)| < MPC_HOVER_ARRIVE_M, and a hand that is inside
            GRASP_EE_OK_M can still be outside that in 3-D, so the ramp would
            never start on its own at exactly the moment we want to close.

            Bounded by MPC_LAND_MAX_TICKS because these are the STALL exits: the
            arm that triggered T12 may be unable to descend at all, and hanging
            forever is a worse failure than closing high. When the bound is hit
            it closes anyway and SAYS how high, so the log carries the number.
            """
            if all(desc_off[s] <= 0.0 for s in tracked):
                return True
            owed = max(desc_off.values())
            for s in tracked:
                descending[s] = True
            land_wait["n"] += 1
            if land_wait["n"] > MPC_LAND_MAX_TICKS:
                print(f"[mpc] {who}: could not land the last "
                      f"{owed*1000:.0f} mm of descent rail in "
                      f"{MPC_LAND_MAX_TICKS} ticks — closing anyway, the jaws "
                      f"take the cube {owed*1000:.0f} mm high")
                return True
            if tick - leash_note.get("land", 0.0) > 1.0:
                leash_note["land"] = tick
                print(f"[mpc] {who} would close here, but {owed*1000:.0f} mm of "
                      f"descent rail is still owed — landing it first so the "
                      f"jaws straddle the cube centre, not its top edge")
            return False

        exec_win = {"left": deque(maxlen=MPC_EXEC_WINDOW_TICKS),
                    "right": deque(maxlen=MPC_EXEC_WINDOW_TICKS)}
        # Ticks in the same window that carried NO new telemetry — the
        # denominator of every execution claim (see MPC_TEL_REPEAT_WARN).
        tel_repeat = deque(maxlen=MPC_EXEC_WINDOW_TICKS)
        enc_prev = dict(enc)
        while True:
            tick = time.monotonic()
            telemetry = ctx.tel.fresh()
            if telemetry is None:
                return Abort("T04", "MPC track", "telemetry link",
                             f"telemetry went stale during the MPC track "
                             f"({tick - t0:.1f}s in)"), None
            fault = telemetry.get("arm_fault") or [False, False]
            if any(fault):
                who = "+".join(s for s, f in zip(("left", "right"), fault) if f)
                return Abort("T05", "MPC track", f"{who} arm",
                             f"ARM FAULT during the MPC track: {fault} — "
                             f"inspect that arm's harness"), None
            enc = _arm_enc(telemetry)
            if enc is None:
                return Abort("T06", "MPC track", "telemetry link",
                             "arm encoders vanished during the MPC track"), None
            ctx.det.poll(rig)

            # CUBE-ANCHORED GOALS (user requirement 2026-08-01): the goal and
            # its leash anchor are base-frame numbers, and torso sway moves
            # the base under the world-fixed cube. Re-express both into the
            # CURRENT base frame from the IMU's tick-to-tick gravity delta,
            # rotating about the ankle pivot — so a FROZEN goal stays nailed
            # to the cube instead of riding the sway. A fresh detection
            # (below) simply overwrites with a measurement that already
            # carries the current base attitude.
            g_now = _grav_unit(telemetry)
            piv = np.array([0.0, 0.0, -TILT_COMP_PIVOT_M])
            # goal_pos is INCREMENTAL on purpose: it is overwritten by every
            # fresh median, so it carries no epoch of its own and each update
            # already arrives in the current base frame. The compounding
            # residual is bounded by the interval since the last sighting.
            if g_now is not None and g_ref is not None:
                rot = _rot_between(g_ref, g_now)
                if rot is not None:
                    for s in tracked:
                        goal_pos[s] = rot @ (goal_pos[s] - piv) + piv
            # YAW leg of the same compensation (08-12 late night, user): the
            # gravity delta above is yaw-blind by construction, and the base
            # yawing under a frozen goal was measured at 5-14 deg per fetch —
            # 40-90 mm of the cube sliding through the base frame unseen.
            # Incremental for the goal (like the tilt), absolute-from-epoch
            # for the anchor (below).
            _yaw_now = getattr(rig, "yaw_integral", 0.0)
            _d_inc = _yaw_now - yaw_prev
            if abs(_d_inc) > 1e-9:
                _Rzi = _yaw_rot(-_d_inc)
                for s in tracked:
                    goal_pos[s] = _Rzi @ (goal_pos[s] - piv) + piv
            yaw_prev = _yaw_now
            # plan_anchor is ABSOLUTE from its own epoch — see anchor0 above.
            if g_now is not None and g_anchor is not None:
                rot_a = _rot_between(g_anchor, g_now)
                for s in tracked:
                    plan_anchor[s] = anchor0[s].copy() if rot_a is None \
                        else rot_a @ (anchor0[s] - piv) + piv
            _dyaw_a = _yaw_now - yaw_mark
            if abs(_dyaw_a) > 1e-9:
                _Rza = _yaw_rot(-_dyaw_a)
                for s in tracked:
                    plan_anchor[s] = _Rza @ (plan_anchor[s] - piv) + piv
            if g_now is not None:
                g_ref = g_now

            # live goals: >=2 faces move the goal freely, ONE face moves it on
            # a leash around the plan anchor (see MPC_LOWEV_LEASH_M), and all
            # motion is slewed so a hop reaches the solver as a glide
            for s in tracked:
                key = cube_of[s]
                med = rig.median(key, tick)
                # The faces behind THIS median, not the newest packet's — see
                # GazeRig.faces_used. Every gate below asks "is the value I am
                # about to use well-evidenced?", and last_inliers answers a
                # different question.
                faces_now = rig.faces_used(key, tick)
                if med is None:
                    blind_since.setdefault(s, tick)
                else:
                    blind_since.pop(s, None)
                if med is not None:
                    med = np.asarray(med, float).copy()
                    # 1-FACE Z SNAP (08-13, user): while the MPC runs, a
                    # median backed by ONE tag face keeps its xy but takes
                    # its z from the table-resting prior. The plan anchor's
                    # z IS that prior — build_scene set it, and the tilt/yaw
                    # compensation above keeps it in the current base frame —
                    # so snapping to it needs no table geometry here and
                    # inherits the compensation for free. Two-face medians
                    # are well-conditioned in z and keep their own vertical;
                    # a cube whose target never got the prior (no sane table
                    # detected) keeps its vision z — there is nothing better.
                    # The cap mirrors build_scene's: a z far off the resting
                    # height means the cube is NOT resting there (lifted,
                    # stacked, knocked off) and the vision z is the truth.
                    if faces_now < DRIFT_MIN_INLIERS and key in z_snap_keys:
                        dz_snap = float(med[2]) - float(plan_anchor[s][2])
                        if abs(dz_snap) <= TABLE_Z_PRIOR_CAP_M:
                            if abs(dz_snap) > 0.002 and \
                                    tick - leash_note.get(("zsnap", s),
                                                          0.0) > 2.0:
                                leash_note[("zsnap", s)] = tick
                                print(f"[mpc] {key}: 1-face z {med[2]:+.3f} "
                                      f"-> {plan_anchor[s][2]:+.3f} "
                                      f"(table-resting prior; xy kept)")
                            med[2] = float(plan_anchor[s][2])
                    cand_goal = med.copy()
                    if faces_now >= DRIFT_MIN_INLIERS:
                        last2face[s] = (tick, np.asarray(med, float).copy())
                    if hover_m > 0.0 and descending[s] and \
                            faces_now < DRIFT_MIN_INLIERS:
                        # (hover_m guard: with the staged hover OFF there is
                        # no descent phase — 1-face leash tracking keeps its
                        # 2026-08-01 semantics untouched)
                        if MPC_DESCENT_1FACE_TRACKS:
                            # 2026-08-08 user override: keep tracking through
                            # the descent on single-face evidence instead of
                            # freezing — the leash and the slew below still
                            # fence the flip/ghost class.
                            if tick - leash_note.get(("desc", s), 0.0) > 2.0:
                                leash_note[("desc", s)] = tick
                                print(f"[mpc] {key}: descent tracking on "
                                      f"1-FACE evidence (override) — leash "
                                      f"and slew still apply")
                        else:
                            # DESCENT FREEZE (2026-08-03, "the left gripper
                            # veers every descent"): the descending hand
                            # occludes its own cube, and 1-face PnP slides
                            # tens of mm SIDEWAYS while staying inside the
                            # 120 mm leash — confidently wrong updates dragged
                            # the goal, and the hand followed (A-side logs:
                            # p50 50-159 mm drift, 100% 1-face). During the
                            # descent only >=2-face evidence may move the
                            # goal; a frozen goal still rides the tilt
                            # compensation above.
                            cand_goal = None
                            if tick - leash_note.get(("desc", s), 0.0) > 2.0:
                                leash_note[("desc", s)] = tick
                                print(f"[mpc] {key}: 1-face update ignored "
                                      f"during the descent — goal frozen; "
                                      f"2-face evidence still tracks")
                    off_anchor = float(np.linalg.norm(
                        np.asarray(med, float) - plan_anchor[s]))
                    if cand_goal is not None and \
                            faces_now < DRIFT_MIN_INLIERS and \
                            off_anchor > MPC_LOWEV_LEASH_M:
                        cand_goal = None       # low evidence far off-anchor:
                                               # that is the wild ghost class
                        # NEVER silently (2026-08-01: a whole track's worth
                        # of better-than-anchor medians was rejected without
                        # a single line, and the hand converged beside the
                        # cube with nothing to show why)
                        if tick - leash_note.get(s, 0.0) > 2.0:
                            leash_note[s] = tick
                            print(f"[mpc] {key}: 1-face median "
                                  f"{off_anchor*1000:.0f} mm off the anchor "
                                  f"— leash-rejected (bar "
                                  f"{MPC_LOWEV_LEASH_M*1000:.0f} mm)")
                    if cand_goal is not None and not anchor_lock[s]:
                        delta = cand_goal - goal_pos[s]
                        dist = float(np.linalg.norm(delta))
                        cap = MPC_GOAL_SLEW_M_S * min(dt_prev, 2.0 * PUB_DT)
                        if dist > 1e-12:
                            goal_pos[s] = goal_pos[s] + delta * min(
                                1.0, cap / dist)
                    # ANCHOR TRUST: a locked side ignores evidence above and
                    # walks its goal home at the same slew the evidence had.
                    if anchor_lock[s]:
                        delta = plan_anchor[s] - goal_pos[s]
                        dist = float(np.linalg.norm(delta))
                        cap = MPC_GOAL_SLEW_M_S * min(dt_prev, 2.0 * PUB_DT)
                        if dist > 1e-12:
                            goal_pos[s] = goal_pos[s] + delta * min(
                                1.0, cap / dist)
                watches[s].sample_goal(goal_pos[s], plan_anchor[s])
                watches[s].sample(telemetry, rig.latest(key, tick), tick,
                                  rig.last_inliers(key))
                # T07 IS A PLANNING PRECONDITION, AND THE TRACKER IS PAST IT.
                # "Could I plan a reach to this point?" is the right question
                # before a route exists; during the terminal descent the right
                # question is "am I already there?", and the hand answers it.
                # This guard was the only one that never got the grasp-anyway
                # rule the others carry, and the only one with no debounce, so
                # a single noisy tick from a 1-face estimate retreated a
                # mission with both hands ON the cubes (hardware 2026-08-06).
                # last_worst is the PREVIOUS tick's hand distance — this check
                # runs before this tick's FK — and starts at infinity so the
                # first tick can never grasp on no evidence.
                why = target_unreachable_why(goal_pos[s])
                if why is None:
                    env_strikes[s] = 0
                else:
                    env_strikes[s] += 1
                    if last_worst < OP.GRASP_EE_OK_M:
                        if tick - leash_note.get(("env", s), 0.0) > 2.0:
                            leash_note[("env", s)] = tick
                            print(f"[mpc] {key} reads out of the envelope "
                                  f"({why.split(' — ')[0]}) but the worst hand "
                                  f"is {last_worst*1000:.0f} mm out, inside "
                                  f"GRASP_EE_OK — the arm is already there; "
                                  f"tracking on")
                    elif env_strikes[s] >= MPC_ENVELOPE_STRIKES:
                        # ANCHOR TRUST: live reads walked the GOAL out of the
                        # envelope — the anchor was inside it at plan time, so
                        # return there instead of retreating. Strikes reset
                        # because the locked goal re-enters the envelope as it
                        # slews home; if even the anchor reads unreachable
                        # (the base rotated since the plan), strikes reaccrue
                        # and the second conviction aborts below.
                        if engage_anchor_trust(
                                "T07", f"{key} read out of the envelope for "
                                f"{env_strikes[s]} straight ticks on live "
                                f"evidence ({why.split(' — ')[0]})"):
                            env_strikes[s] = 0
                        else:
                            return Abort("T07", "MPC track",
                                         f"{side_name[s]} arm / {key}",
                                         f"{key} drifted out of the envelope "
                                         f"during the track for "
                                         f"{env_strikes[s]} straight ticks, worst "
                                         f"hand {last_worst*1000:.0f} mm out: "
                                         f"{why}"), None

            # THE TAGS GOING DARK IS THE GRASP CONDITION, NOT A FAILURE
            # (user 2026-08-06). The tracker exists to correct what the cube did
            # during the ~5 s open-loop stream, and it can only do that while it
            # can SEE the cube. Once every tracked cube is dark the goal is dead
            # reckoning off the IMU — frozen, moved only by the tilt
            # compensation — so the tracker is servoing to torso sway. That
            # evening's left cube was blind for an ENTIRE track and the round
            # still burned its whole budget before retreating.
            #
            # On a top-down approach the overwhelmingly likely reason the tags
            # vanish is the gripper standing over them; the descent freeze forty
            # lines up says exactly that ("the descending hand occludes its own
            # cube"). This machine already recognised self-occlusion as the
            # signature of a SUCCESSFUL approach and then did nothing with it.
            #
            # THE CHECK RUNS BEFORE THE SOLVER STEPS, AND THAT PLACEMENT IS THE
            # WHOLE POINT (user, first hardware run of the rule: "I saw the left
            # arm move a little bit all of a sudden, but it should not move
            # because the cube is inside of the gripper — when you move, it
            # lost"). Sitting after mpc_step, it published a command first and
            # nudged a hand that was already on its cube. A blind tracker must
            # not move the arm AT ALL: no step, no publish, close where the
            # stream left it.
            #
            # This is the only POSITIVE exit in the tracker. The other four
            # (T07/T12/T13/T14) are failures being forgiven, and the shape of
            # that list is what said the machine needed a rule that says
            # "conditions are met, go".
            #
            # The age printed is the real one — time since the cube was last
            # SEEN, not since this loop noticed — because a blind track can
            # (and did) begin already blind, where a since-noticed counter
            # reads a meaningless 0.0s. Blindness is ~0.7-1.0 s sustained by
            # construction anyway: GazeRig.median needs a capture inside
            # CUBE_LATCH_WINDOW_S and a sighting inside CubeTrack.confirmed(),
            # so one dropped frame cannot trigger this.
            #
            # BLIND IS NOT ENOUGH — IT MUST ALSO BE STANDING ON THE CUBE
            # (hardware 2026-08-06, the rule's second run). It closed at left
            # 51 mm / right 76 mm from the anchor and both grippers came back
            # EMPTY, because the goal had drifted 40-55 mm off the anchor on
            # 100%-1-face evidence and the arms had faithfully followed it
            # there. "The tags went dark" and "the gripper is over the cube"
            # are only the same event when the hand is actually AT the cube;
            # the arms also cross the camera-cube sight line on their way in,
            # and this log shows both cubes going dark together, which one
            # gripper cannot do.
            #
            # The yardstick is the PLAN ANCHOR, not the tracked goal: the
            # anchor was measured with the camera parked and nothing occluding
            # it, so it is the one number a drifting 1-face estimate cannot
            # corrupt. Gating on the tracked goal would be circular — it is
            # the thing under suspicion.
            #
            # Blind but NOT on the anchor falls through to the machinery that
            # was already there. That is deliberate: with no live evidence the
            # frozen goal IS the best estimate available, and letting the
            # tracker close the remaining distance onto it is exactly right —
            # if the hand then reaches the anchor, the rule above fires and
            # grasps; if it cannot, T13/T14 own it with a readable verdict.
            # The first version denied that chance and closed on air instead.
            if MPC_BLIND_GRASP and len(blind_since) == len(tracked):
                ee_now = measured_ee(enc)
                off = {s: float(np.linalg.norm(ee_now[s] - plan_anchor[s]))
                       for s in tracked}
                dark = max(tick - _last_seen(rig, cube_of[s], tick)
                           for s in tracked)
                where = ", ".join(
                    f"{side_name[s]} {_lat_vert(ee_now[s], plan_anchor[s])}"
                    for s in tracked)
                keys = ", ".join(cube_of[s] for s in tracked)
                if max(off.values()) <= MPC_BLIND_ANCHOR_M and \
                        any(desc_off[s] > 0.0 for s in tracked):
                    # LAND THE RAIL FIRST — the jaws must straddle the cube
                    # CENTRE, not its top edge (user 2026-08-06: "should be the
                    # center!"). While desc_off > 0 the solver is aimed
                    # reach_hover_mm ABOVE the grasp point, so closing here is
                    # cube centre + GRASP_Z_ABOVE_M + hover = 25 mm high on a
                    # cube whose top face is +30 mm. Neither
                    # MPC_BLIND_ANCHOR_M nor GRASP_EE_OK_M can see it: both are
                    # 3-D distances, and a pure z offset hides inside them.
                    #
                    # descending[] is forced because the normal trigger
                    # (|ee - (goal+hover)| < MPC_HOVER_ARRIVE_M) may never fire
                    # on a blind track — the hand can sit inside the anchor bar
                    # and still be 42 mm from the hover point in 3-D. The ramp
                    # is MPC_DESCENT_M_S, so this costs ~6 ticks, and it is the
                    # only motion a blind tracker is allowed: straight down the
                    # vertical rail onto a frozen goal, never a lateral chase.
                    for s in tracked:
                        descending[s] = True
                    if tick - leash_note.get("blindland", 0.0) > 1.0:
                        leash_note["blindland"] = tick
                        print(f"[mpc] blind on {keys} and on the plan anchor "
                              f"({where}) — landing the "
                              f"{max(desc_off.values())*1000:.0f} mm still owed "
                              f"on the descent rail before closing, so the jaws "
                              f"straddle the cube centre and not its top edge.")
                elif max(off.values()) <= MPC_BLIND_ANCHOR_M and \
                        max(float(ee_now[s][2] - plan_anchor[s][2])
                            for s in tracked) > MPC_CLOSE_Z_MAX_M:
                    # Z BAR (user 08-10): laterally on the anchor is not
                    # enough — the 3-D bar was letting closes through 11-26
                    # mm HIGH (the approach is always from above, so the
                    # residual's z part never lands low). Do NOT close; keep
                    # the MPC pressing onto the frozen goal. If the arm
                    # truly cannot get lower, the budget salvage still ends
                    # the track — and says how high it closed.
                    if tick - leash_note.get("blindz", 0.0) > 1.0:
                        leash_note["blindz"] = tick
                        z_hi = max(float(ee_now[s][2] - plan_anchor[s][2])
                                   for s in tracked)
                        print(f"[mpc] blind and laterally on the anchor "
                              f"({where}), but {z_hi*1000:+.0f} mm HIGH "
                              f"(z bar {MPC_CLOSE_Z_MAX_M*1000:.0f} mm) — "
                              f"pressing the descent before closing")
                elif max(off.values()) <= MPC_BLIND_ANCHOR_M:
                    print(f"[mpc] BLIND GRASP: no tag on {keys} for "
                          f"{dark:.1f}s, every hand is on its plan anchor "
                          f"({where}, bar {MPC_BLIND_ANCHOR_M*1000:.0f} mm) and "
                          f"the descent rail has landed — the hand occludes its "
                          f"own cube, so tracking has nothing left to correct. "
                          f"Closing WITHOUT moving.")
                    z_ledger("blind-anchor")
                    return None, dict(enc)
                elif engage_anchor_trust(
                        "BLIND", f"blind on {keys} for {dark:.1f}s with hands "
                        f"{where} from the anchor — a frozen drifted goal has "
                        f"nothing left to offer"):
                    pass                      # goals now slew home; the blind
                                              # anchor rules above will close
                elif tick - leash_note.get("blindfar", 0.0) > 2.0:
                    leash_note["blindfar"] = tick
                    print(f"[mpc] blind on {keys} for {dark:.1f}s but "
                          f"{where} from the plan anchor (bar "
                          f"{MPC_BLIND_ANCHOR_M*1000:.0f} mm) — too far to "
                          f"call this an occluded grasp; tracking the frozen "
                          f"goal on, the progress guard and budget own this.")

            # The monitor's green route is a plan-time snapshot riding
            # base_link — it cannot show lock. These markers are the LIVE aim:
            # nailed on the cube means the tracker is locked.
            # The EFFECTIVE aim: hover point in phase 1, then a z-RAMP down
            # the vertical rail in phase 2 (MPC_DESCENT_M_S) until the cube.
            # Only the solver and the preview see it — every guard keeps the
            # true goal.
            for s in tracked:
                if descending[s] and desc_off[s] > 0.0:
                    desc_off[s] = max(0.0,
                                      desc_off[s] - MPC_DESCENT_M_S
                                      * min(dt_prev, 2.0 * PUB_DT))
            eff_goal = {s: goal_pos[s] + np.array([0.0, 0.0, desc_off[s]])
                        for s in tracked}
            if ctx.route_pub is not None and \
                    tick - last_goal_pub > MPC_GOAL_PREVIEW_S:
                last_goal_pub = tick
                ctx.route_pub.publish({
                    "goals_only": True,
                    "goals": {side_name[s]: [float(x) for x in eff_goal[s]]
                              for s in tracked}})

            try:
                step = ctx.client.mpc_step(enc, eff_goal)
            except PlanServerError as e:
                fails += 1
                if fails > MPC_STEP_FAIL_MAX:
                    return Abort("T08", "MPC track", "plan server",
                                 f"MPC stepping failed {fails}x: {e}"), None
                cmd = last_cmd                  # hold the last accepted target
                step = None
            if step is not None:
                dt = max(PUB_DT, time.monotonic() - tick)
                dt_prev = dt
                raw = step["enc"]
                jump_joint = max(ARM_JOINTS_ALL,
                                 key=lambda n: abs(raw[n] - enc[n]))
                worst_jump = abs(raw[jump_joint] - enc[jump_joint])
                if worst_jump > MPC_CMD_SANE_RAD:
                    # Not a correction the clamp below can domesticate: the
                    # solver is commanding a different posture. Following it
                    # slowly is still following it — reject and count. (The
                    # server's re-anchor guard should re-pace the rollout long
                    # before this trips; if it trips anyway, the joint name is
                    # the first diagnostic.)
                    fails += 1
                    if fails > MPC_STEP_FAIL_MAX:
                        return Abort(
                            "T09", "MPC track",
                            f"{_arm_of(jump_joint)} arm / {jump_joint}",
                            f"MPC commanded {worst_jump:.2f} rad in one tick "
                            f"at {jump_joint} (sane bound "
                            f"{MPC_CMD_SANE_RAD}), {fails} unusable ticks in "
                            f"a row — aborting the track"), None
                    cmd = last_cmd
                else:
                    # CLAMP, don't reject: after warm-up the solver's first
                    # commits legitimately overshoot our streaming rate
                    # (hardware 2026-08-01: 2.08 rad/s where the cap is 0.7 —
                    # 21 straight rejections killed the round). The receiver
                    # already rate-limits to the packet's `rate`; clamping
                    # here keeps the gated segment a superset of what will
                    # actually be executed this tick, instead of gating a leap
                    # that was never going to happen. The clamp anchors at the
                    # MEASUREMENT but allows an MPC_LEAD_SCALE-tick LEAD: at
                    # 1x, a lagging arm's PD error was capped at one tick's
                    # step and kd (velocity-ref is always 0 on the wire)
                    # braked it below the stream — the rear-sector 13-20%
                    # execution collapse (forensics 2026-08-02).
                    lim = MPC_LEAD_SCALE * a.max_rate * dt
                    cand = {n: enc[n] + float(np.clip(raw[n] - enc[n],
                                                      -lim, lim))
                            for n in ARM_JOINTS_ALL}
                    verdict = gate.check(cand, enc, dt)
                    if verdict.ok:
                        fails = 0
                        cstandoff = 0
                        last_cmd = cand
                        cmd = cand
                    else:
                        fails += 1
                        # A clearance rejection is a STANDOFF: the solver's
                        # optimum threads below our floor, and re-solving from
                        # the same state reproduces the same path (hardware
                        # 2026-08-01: 21 identical elbow-torso rejections with
                        # the arm frozen mid-air). Abort early and retreat.
                        cstandoff = cstandoff + 1 \
                            if "clearance" in (verdict.why or "") else 0
                        if cstandoff >= MPC_CLEARANCE_STANDOFF_MAX:
                            pair = " ~ ".join(verdict.where) if verdict.where \
                                else "unknown geom pair"
                            # STANDOFF WITH THE CUBE IN REACH -> CLOSE, DON'T
                            # RETREAT (user 08-13, after four T10 retreats
                            # with the hand ~20 mm out: "just let it
                            # grasp!"). The MPC's elbow preference sits
                            # 1-3 cm below our clearance floor in OUR model —
                            # the two collision models disagree at the elbow
                            # — so the terminal press routinely ends exactly
                            # here: arm frozen mid-air, jaws already
                            # straddling the cube. Every successful close
                            # today happened at 13-25 mm and the jaws open
                            # 90 mm; a wrong close costs one certified-EMPTY
                            # pinch and a re-plan, a retreat costs the whole
                            # round. Beyond GRASP_EE_OK the standoff still
                            # aborts — closing 80 mm out only grabs air.
                            ee_t10 = measured_ee(enc)
                            worst_t10 = max(float(np.linalg.norm(
                                ee_t10[s] - goal_pos[s])) for s in tracked)
                            if worst_t10 < OP.GRASP_EE_OK_M:
                                print(f"[mpc] clearance standoff at {pair} — "
                                      f"but the worst hand is "
                                      f"{worst_t10 * 1000:.0f} mm from its "
                                      f"cube, inside GRASP_EE_OK: grasping "
                                      f"where we are instead of retreating")
                                z_ledger("T10-standoff")
                                return None, dict(enc)
                            return Abort(
                                "T10", "MPC track", pair,
                                f"clearance standoff: {cstandoff} straight "
                                f"gate rejections ({verdict.why}) — the "
                                f"solver will not route above our floor; "
                                f"aborting early to retreat"), None
                        if fails > MPC_STEP_FAIL_MAX:
                            return Abort(
                                "T11", "MPC track", "per-tick StepGate",
                                f"per-tick gate rejected {fails} "
                                f"consecutive commands — last: "
                                f"{verdict.why}"), None
                        cmd = last_cmd

            slots = {}
            for arm, sk, names in (("left", "L", ARM_JOINTS_L),
                                   ("right", "R", ARM_JOINTS_R)):
                # The untracked arm — possibly holding an earlier leg's cube —
                # is NOT the solver's to command: server-side it is only
                # soft-pinned in task space, so its q_next wanders around the
                # pin. It stays frozen at the handoff snapshot. The tracked arm
                # follows the accepted command; before the first acceptance it
                # holds the SAME snapshot, never this tick's live encoders —
                # tracking the measurement lets gravity sag ratchet the arm
                # downhill a packet at a time (see hold_packet).
                src = cmd if (sk in tracked and cmd is not None) else hold
                slots[arm] = {"joint_pos": [float(src[n]) for n in names],
                              "rate": float(a.max_rate)}
            ctx.pub.publish(rig.tick(telemetry, {"arm_targets": slots}))
            # EXECUTOR GUARD, ratio form: over the last ~2 s, an arm that
            # executed under MPC_EXEC_MIN_RATIO of what it was asked is
            # stuck. Ratio, not integral — the integral form convicted a
            # decelerating final approach (target a constant hair ahead,
            # motion slowing into it) exactly when the hand was almost there.
            # A tick whose encoders are bit-identical to the last one got no
            # new packet: the arm is being tracked at 20 Hz and cannot hold
            # still to the last bit while it is being driven.
            tel_repeat.append(enc == enc_prev)
            stalls = sum(tel_repeat)
            if len(tel_repeat) == tel_repeat.maxlen and \
                    stalls > MPC_TEL_REPEAT_WARN * len(tel_repeat) and \
                    tick - leash_note.get("telrep", 0.0) > 2.0:
                leash_note["telrep"] = tick
                print(f"[mpc] WARNING: {stalls}/{len(tel_repeat)} of the last "
                      f"ticks carried NO telemetry update — real_env is "
                      f"publishing late (CPU contention on its pinned cores?). "
                      f"Every execution ratio below is measured through that.")
            stall_why = None
            stall_who = ""
            for arm, sk, names in (("left", "L", ARM_JOINTS_L),
                                   ("right", "R", ARM_JOINTS_R)):
                if sk not in tracked:
                    continue
                src = cmd if cmd is not None else hold
                asked = max(abs(src[n] - enc[n]) for n in names)
                moved = max(abs(enc[n] - enc_prev[n]) for n in names)
                exec_win[arm].append((asked, moved, tick))
                if len(exec_win[arm]) == exec_win[arm].maxlen:
                    asked_sum = sum(a_ for a_, _, _ in exec_win[arm])
                    moved_sum = sum(m_ for _, m_, _ in exec_win[arm])
                    # MEASURED, not nominal: a starved loop stretches the
                    # window, and a rad/s printed against 2.0 s that really
                    # took 5 s is a lie in the verdict (2026-08-05, the tool
                    # pinned to two cores ran p90 130 ms ticks).
                    win_s = max(PUB_DT,
                                exec_win[arm][-1][2] - exec_win[arm][0][2])
                    speed = moved_sum / win_s
                    bind = max(names, key=lambda n: abs(src[n] - enc[n]))
                    lag_bar = MPC_EXEC_LAG_FRACTION * MPC_LEAD_SCALE \
                        * ctx.args.max_rate * dt_prev
                    low_ratio = asked_sum >= MPC_EXEC_MIN_ASKED_RAD and \
                        moved_sum < MPC_EXEC_MIN_RATIO * asked_sum
                    if low_ratio and speed >= MPC_EXEC_MIN_SPEED_RAD_S:
                        # MOVING, just not as fast as it was asked to. This
                        # guard exists to catch a WEDGED arm; a ratio alone
                        # cannot, because the ratio's denominator is a speed
                        # the lead clamp structurally forbids (see
                        # MPC_EXEC_MIN_SPEED_RAD_S). An arm still covering
                        # ground is the progress guard's business, not this
                        # one's: MPC_NO_PROGRESS_S judges whether the HAND is
                        # closing on the cube, which is the question that
                        # actually matters, and the budget backstops it.
                        if tick - leash_note.get(("slow", arm), 0.0) > 2.0:
                            leash_note[("slow", arm)] = tick
                            print(f"[mpc] {arm} arm ratio "
                                  f"{100.0 * moved_sum / asked_sum:.0f}% but "
                                  f"MOVING at {speed:.2f} rad/s (floor "
                                  f"{MPC_EXEC_MIN_SPEED_RAD_S}) — speed-"
                                  f"limited by the lead clamp, NOT wedged; "
                                  f"tracking on, progress guard owns this")
                    elif low_ratio and abs(src[bind] - enc[bind]) < lag_bar:
                        # Low ratio but the arm is ON its target: the solver
                        # is DITHERING near convergence and the asked-sum is
                        # oscillation, not ignored commands (2026-08-04:
                        # 15% ratio with a 0.07 rad binding lag against a
                        # 0.152 ceiling). Say so, keep tracking.
                        if tick - leash_note.get(("dither", arm), 0.0) > 2.0:
                            leash_note[("dither", arm)] = tick
                            print(f"[mpc] {arm} arm ratio "
                                  f"{100.0 * moved_sum / asked_sum:.0f}% but "
                                  f"binding lag "
                                  f"{abs(src[bind] - enc[bind]):.2f} < "
                                  f"{lag_bar:.2f} — solver dither, NOT a "
                                  f"stall; tracking on")
                    elif low_ratio:
                        # DECISION DEFERRED: whether a stalled arm aborts
                        # depends on WHERE it stalled — see the grasp-anyway
                        # branch after `worst` is computed below (hardware
                        # 2026-08-02, the user's fury was correct: 'almost
                        # there then it retracts' — a stall INSIDE the grasp
                        # tolerance must close the gripper, not walk away).
                        # SELF-INTERPRETING (2026-08-05): the bar, the clamp
                        # ceiling the lag is capped by, both speeds in rad/s,
                        # and how many ticks the measurement even had — a
                        # verdict read hours later must not need this file.
                        ceil_ = MPC_LEAD_SCALE * ctx.args.max_rate * dt_prev
                        lag_now = abs(src[bind] - enc[bind])
                        stall_who = f"{arm} arm / {bind}"
                        stall_why = (f"{arm} arm is NOT executing: moved "
                                     f"{moved_sum:.2f} rad of the "
                                     f"{asked_sum:.2f} rad commanded over "
                                     f"the last {win_s:.1f}s "
                                     f"({100.0 * moved_sum / asked_sum:.0f}%, "
                                     f"bar {100.0 * MPC_EXEC_MIN_RATIO:.0f}%; "
                                     f"{speed:.2f} rad/s, under the "
                                     f"{MPC_EXEC_MIN_SPEED_RAD_S} floor, vs "
                                     f"{asked_sum / win_s:.2f} rad/s asked); "
                                     f"binding "
                                     f"joint {bind} lags {lag_now:.2f} rad = "
                                     f"{100.0 * lag_now / ceil_:.0f}% of the "
                                     f"{ceil_:.3f} clamp ceiling, dither "
                                     f"acquittal below {lag_bar:.2f}; "
                                     f"{stalls}/{len(tel_repeat)} ticks in the "
                                     f"window carried no telemetry update")
            enc_prev = dict(enc)
            if meas_hist is not None and (
                    not meas_hist or any(
                        abs(enc[n] - meas_hist[-1][n]) > RETREAT_STEP_MIN_RAD
                        for n in ARM_JOINTS_ALL)):
                meas_hist.append({n: float(enc[n]) for n in ARM_JOINTS_ALL})

            ee = measured_ee(enc)
            for s in tracked:
                if not descending[s] and float(np.linalg.norm(
                        ee[s] - (goal_pos[s] + hover_vec))) < MPC_HOVER_ARRIVE_M:
                    descending[s] = True
                    # DESCENT RE-ANCHOR: phase 1 may have followed a 1-face
                    # drift (direction-constant lateral bias); land on the
                    # freshest 2-face truth instead when one exists.
                    t2 = last2face.get(s)
                    if t2 is not None and tick - t2[0] <= DESC_ANCHOR_FRESH_S:
                        shift = float(np.linalg.norm(t2[1] - goal_pos[s]))
                        goal_pos[s] = t2[1].copy()
                        print(f"[mpc] {side_name[s]} hand at the hover — "
                              f"descending onto the last 2-face fix "
                              f"({tick - t2[0]:.1f}s old, {shift * 1000:.0f} "
                              f"mm from the tracked goal)")
                    else:
                        mode = ("live 1-face tracking stays ON through the "
                                "descent (override)"
                                if MPC_DESCENT_1FACE_TRACKS else
                                "goal frozen for the descent")
                        print(f"[mpc] {side_name[s]} hand at the hover "
                              f"({hover_m * 1000:.0f} mm above the cube) — "
                              f"descending on 1-FACE evidence, {mode}; expose "
                              f"a second tag face for centered grasps")
            worst = max(float(np.linalg.norm(ee[s] - goal_pos[s]))
                        for s in tracked)
            last_worst = worst          # for next tick's pre-FK guards (T07)

            # >>> ESCAPE GUARD (user 08-13, translated: "if the hand has been
            # inside the 60 mm ring at any point this track and the distance
            # then keeps growing — close IMMEDIATELY at the recorded closest
            # point; no 3 s verdict wait, no abort").
            #
            # The failure it closes, measured three times today: the hand
            # ARRIVES inside GRASP_EE_OK, then the solver walks it back out.
            # Not a wedge and not a stall — the arm executes faithfully
            # (verdict lag 0.051-0.102 against a 0.152 ceiling); cuRobo's own
            # self-collision cost simply dominates the goal cost for a
            # near+high target and commands the hand away. The last such
            # track read "47 -> 67 mm in 3s": by the time the 3 s
            # no-progress window ripened, `worst` had crossed 60 mm, so the
            # grasp-anyway branch that would have closed at 47 mm no longer
            # applied and the round aborted to a retreat instead.
            #
            # So: once a side has been inside the ring this track, its own
            # best distance is a PROVEN-REACHABLE pose, and drifting away
            # from it is evidence about the solver, not about the cube.
            # Close at once rather than spend three more seconds getting
            # worse. Bounded three ways — it must have been inside the ring,
            # it must have receded by a real margin (not solver dither), and
            # the close still runs the descent rail like every other exit.
            # ...but ONLY while the GOAL itself is holding still. If the cube
            # genuinely moved — or was carried out of the envelope — then a
            # growing distance is the goal running away, not the solver, and
            # the drift/envelope aborts are the correct answer; closing on a
            # pose the cube has left would grasp air. The real failure this
            # guard exists for had "GOAL sat 0 mm from the plan anchor".
            _goal_still = max(
                float(np.linalg.norm(goal_pos[s] - plan_anchor[s]))
                for s in tracked) <= MPC_ESCAPE_GOAL_STILL_M
            if worst < OP.GRASP_EE_OK_M:
                best_worst = min(best_worst, worst)
            elif _goal_still and best_worst < OP.GRASP_EE_OK_M and \
                    worst > best_worst + MPC_ESCAPE_MARGIN_M:
                if rail_down("escape guard"):
                    print(f"[mpc] ESCAPE GUARD: the hand reached "
                          f"{best_worst * 1000:.0f} mm (inside GRASP_EE_OK) "
                          f"and the solver has since walked it out to "
                          f"{worst * 1000:.0f} mm — closing NOW at the pose "
                          f"we have rather than retreating from a target we "
                          f"already reached")
                    z_ledger("escape-guard")
                    return None, dict(enc)
            # <<< ESCAPE GUARD

            # THE GRASP-ANYWAY RULE OUTRANKS EVERY STALL GUARD (user
            # 2026-08-02: "almost there, then it retracts" — correct fury).
            # The budget path has always closed the gripper at any stall
            # within GRASP_EE_OK_M; a stalled ARM inside that same tolerance
            # is the same situation arriving earlier. The hand cannot get
            # closer — closing where it stands is the designed bet, and the
            # empty-pinch retry underwrites a miss. Only a stall OUTSIDE the
            # tolerance aborts to the retreat.
            # The verdict is withheld, not the evidence: the operator still
            # sees the exact line the abort would have carried, so the runs
            # that would have retreated remain countable in the log.
            if stall_why is not None and not MPC_EXEC_GUARD:
                if tick - leash_note.get("t12off", 0.0) > 2.0:
                    leash_note["t12off"] = tick
                    print(f"[mpc] T12 DISABLED (MPC_EXEC_GUARD=False) — would "
                          f"have retreated: {stall_why}. Tracking on; the "
                          f"{MPC_NO_PROGRESS_S:.0f}s progress guard and the "
                          f"budget still own this track.")
                stall_why = None
            if stall_why is not None:
                if worst < OP.GRASP_EE_OK_M:
                    if rail_down("T12 grasp-anyway"):
                        print(f"[mpc] {stall_why} — but the worst hand is "
                              f"{worst*1000:.0f} mm from its cube, inside "
                              f"GRASP_EE_OK — grasping where we are")
                        z_ledger("T12-stall")
                        return None, dict(enc)
                    stall_why = None          # deferred: re-judged next tick
                if stall_why is not None:
                    # ANCHOR TRUST: an execution stall this far out most often
                    # means the GOAL ran (the hand started ON the anchor at
                    # 100% handoff). Return home first; a stall that repeats
                    # while locked is the genuinely wedged arm and aborts.
                    if engage_anchor_trust(
                            "T12", f"{stall_who} stalled with the worst hand "
                            f"{worst*1000:.0f} mm out, beyond GRASP_EE_OK"):
                        stall_why = None
                    else:
                        return Abort("T12", "MPC track", stall_who,
                                     f"{stall_why} — the worst hand is "
                                     f"{worst*1000:.0f} mm out, beyond GRASP_EE_OK "
                                     f"({OP.GRASP_EE_OK_M*1000:.0f} mm), so this "
                                     f"aborts to retreat"), None
            prog_win.append((tick, worst))
            # Keep exactly ONE sample at or beyond the window edge: popping
            # everything older would leave prog_win[0] just INSIDE the window,
            # and the "a full window has been observed" test below could then
            # never be satisfied on discrete ticks.
            while len(prog_win) > 1 and \
                    tick - prog_win[1][0] >= MPC_NO_PROGRESS_S:
                prog_win.popleft()
            # Judge only once a full window has actually been observed, so a
            # track cannot be convicted on its first second.
            if tick - t0 > MPC_NO_PROGRESS_S \
                    and tick - prog_win[0][0] >= MPC_NO_PROGRESS_S \
                    and prog_win[0][1] - worst < MPC_PROGRESS_MIN_M \
                    and worst >= MPC_CONVERGED_M:
                if worst < OP.GRASP_EE_OK_M:
                    # ANCHOR TRUST first offense (08-12 B): a stall inside the
                    # ring used to close 30-47 mm off-centre right where it
                    # stood — the marginal-grasp factory. Press to the anchor
                    # once for a centred close; a second stall settles for
                    # closing here, exactly as before.
                    if engage_anchor_trust(
                            "T13-near", f"stalled at {worst*1000:.0f} mm "
                            f"inside GRASP_EE_OK — pressing to the anchor for "
                            f"a centred close before settling for here"):
                        t13_defer = True
                        continue
                    if rail_down("T13 grasp-anyway"):
                        print(f"[mpc] no progress for {MPC_NO_PROGRESS_S:.0f}s "
                              f"at {worst*1000:.0f} mm — inside GRASP_EE_OK, "
                              f"grasping where we are")
                        z_ledger("T13-stall")
                        return None, dict(enc)
                    # Landing the rail IS progress, not a stall: restart the
                    # no-progress window and skip the verdict rather than
                    # convicting a hand that is inside GRASP_EE_OK and
                    # descending onto its own cube.
                    prog_win.clear()          # landing the rail IS progress
                    t13_defer = True
                else:
                    t13_defer = False
                # The one-line differential diagnosis — CALIBRATED against the
                # clamp (review 2026-08-01 night caught the first version
                # backwards): cand = enc + clip(raw - enc,
                # ±MPC_LEAD_SCALE*max_rate*dt) mathematically caps this lag
                # AT the ceiling, so a lag sitting AT the ceiling means the
                # solver demanded the full lead every tick and the arm
                # executed NONE of it (wedged / frozen feedback /
                # undertorqued); only a lag well BELOW the ceiling means the
                # arm is where it was told to be and the GOAL is the problem
                # (perception).
                ceil = MPC_LEAD_SCALE * a.max_rate * dt_prev
                lag = {}
                for arm, names in (("left", ARM_JOINTS_L),
                                   ("right", ARM_JOINTS_R)):
                    lag[arm] = float("nan") if last_cmd is None else \
                        max(abs(last_cmd[n] - enc[n]) for n in names)
                tag = {arm: ("=CEILING, arm NOT executing"
                             if lag[arm] >= 0.8 * ceil else "executing")
                       for arm in lag}
                worst_side = max(tracked, key=lambda s: float(
                    np.linalg.norm(ee[s] - goal_pos[s])))
                if not t13_defer:
                    # ANCHOR TRUST: first conviction returns the goal home
                    # (the off-anchor distance in the print is the usual
                    # culprit); a conviction while locked aborts as before.
                    if engage_anchor_trust(
                            "T13", f"no progress in {MPC_NO_PROGRESS_S:.0f}s "
                            f"with the worst hand {worst*1000:.0f} mm out and "
                            f"its goal "
                            f"{np.linalg.norm(goal_pos[worst_side] - plan_anchor[worst_side])*1000:.0f}"
                            f" mm off the anchor"):
                        continue
                    return Abort(
                        "T13", "MPC track",
                        f"{side_name[worst_side]} arm / {cube_of[worst_side]}",
                        f"no progress: {side_name[worst_side]} hand went "
                        f"{prog_win[0][1]*1000:.0f} -> {worst*1000:.0f} mm in "
                        f"{MPC_NO_PROGRESS_S:.0f}s while its GOAL sat "
                        f"{np.linalg.norm(goal_pos[worst_side] - plan_anchor[worst_side])*1000:.0f}"
                        f" mm from the plan anchor (lag vs clamp ceiling "
                        f"{ceil:.3f}: left {lag['left']:.3f} {tag['left']}, "
                        f"right {lag['right']:.3f} {tag['right']}) — "
                        f"aborting to retreat"), None
            # Convergence ALSO requires the descent rail to have landed:
            # while desc_off > 0 the arm is commanded ABOVE the cube, and
            # |ee - true_goal| can dip under the bar mid-ramp — closing there
            # grasps ~15 mm high (caught by the staged-hover pin in sim).
            # AND the measured z residual must clear MPC_CLOSE_Z_MAX_M (user
            # 08-10): the 3-D converge bar hid a systematic +11..26 mm high
            # bias exactly like the blind-anchor bar did.
            z_high = max(float(ee[s][2] - goal_pos[s][2]) for s in tracked)
            if worst < MPC_CONVERGED_M and \
                    z_high <= MPC_CLOSE_Z_MAX_M and \
                    all(desc_off[s] <= 0.0 for s in tracked):
                streak += 1
                if streak >= MPC_CONVERGED_TICKS:
                    print(f"[mpc] converged: worst hand "
                          f"{worst*1000:.0f} mm from its LIVE cube "
                          f"({time.monotonic()-t0:.1f}s)")
                    z_ledger("converged")
                    return None, dict(enc)
            else:
                streak = 0

            if time.monotonic() - t0 > MPC_MAX_S:
                if worst < OP.GRASP_EE_OK_M:
                    if rail_down("T14 budget"):
                        print(f"[mpc] budget spent at {worst*1000:.0f} mm — "
                              f"inside GRASP_EE_OK_M, grasping anyway "
                              f"(open-loop parity)")
                        z_ledger("T14-budget")
                        return None, dict(enc)
                    continue          # landing the rail; re-judge next tick
                budget_side = max(tracked, key=lambda s: float(
                    np.linalg.norm(ee[s] - goal_pos[s])))
                return Abort(
                    "T14", "MPC track",
                    f"{side_name[budget_side]} arm / {cube_of[budget_side]}",
                    f"MPC track did not converge in {MPC_MAX_S:.0f}s "
                    f"(worst hand {worst*1000:.0f} mm out, needs "
                    f"{MPC_CONVERGED_M*1000:.0f} mm)"), None
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    finally:
        # The failure runs are the ones that need sizing: every abort return
        # used to skip the watch reports, so the runs where drift/tilt/inlier
        # numbers would explain the failure printed nothing (review
        # 2026-08-01). Reporting here covers every exit exactly once.
        for w in watches.values():
            w.report()
        # ALWAYS torn down — including Ctrl+C and every return above (a stop
        # with no session open is a server-side no-op, so the failed-start
        # path lands here harmlessly). mpc_stop itself never raises, and its
        # timeout is the short MPC one: this runs unpumped, and a dead server
        # must not buy seconds of arm silence for a best-effort teardown.
        ctx.client.mpc_stop()


def retreat_after_track_abort(ctx: Ctx, history: list) -> str | None:
    """Walk the arms BACK along the path they ACTUALLY travelled.

    `history` is the round's measured-posture log (stream + track, in order,
    deduped — see meas_hist). The retreat replays it in reverse and settles on
    its first entry, the posture the round physically started from.

    MEASURED, NOT COMMANDED — the distinction is a hardware lesson
    (2026-08-01 night, watched by the user on every failure): the first
    version replayed the published footsteps and then the ROUTE backwards
    from the handoff time, silently assuming the arm was ON the route. A
    wedged arm never was: the route-reversal's first command yanked it from
    its wedge to the route's 80% posture — "one arm suddenly lifts high" —
    before both arms wound home. Reversing the measured trail cannot do
    that: every replayed posture is a configuration the robot physically
    occupied this round (collision-free by evidence, not by assumption), a
    wedged arm's trail barely moves it until the early-stream entries walk
    it gently off the obstruction, and the trail ends exactly at the true
    start posture — the clean seed the next round needs (a mid-corridor
    abort posture can sit inside the route-clearance floor, where every
    re-plan is rejected at frame 1).

    Returns None on success, else why the retreat was skipped or cut short.
    Either way the caller proceeds to SKIP; a skipped retreat just means the
    next round may refuse on the seed posture — which is exactly where we
    already were without this function.
    """
    a = ctx.args
    telemetry = ctx.tel.fresh()
    enc = _arm_enc(telemetry) if telemetry else None
    if enc is None or any(telemetry.get("arm_fault") or [False, False]):
        return "no fresh telemetry (or arm fault) — retreat skipped"
    if not history:
        return "no travel recorded — retreat skipped"
    gap = max(abs(enc[n] - history[-1][n]) for n in ARM_JOINTS_ALL)
    if gap > RETREAT_START_TOL_RAD:
        return (f"measured posture is {gap:.2f} rad from the end of the "
                f"recorded trail — not where the arm is; retreat skipped")
    print(f"[retreat] walking back along the measured trail: "
          f"{len(history)} posture(s) in reverse to the round's start")

    def _publish(rec) -> str | None:
        t = ctx.tel.fresh()
        if t is None:
            return "telemetry went stale during the retreat"
        if any(t.get("arm_fault") or [False, False]):
            return "ARM FAULT during the retreat"
        ctx.det.poll(ctx.rig)
        slots = {arm: {"joint_pos": [float(rec[n]) for n in names],
                       "rate": float(a.max_rate)}
                 for arm, names in (("left", ARM_JOINTS_L),
                                    ("right", ARM_JOINTS_R))}
        ctx.pub.publish(ctx.rig.tick(t, {"arm_targets": slots}))
        return None

    for rec in reversed(history):
        tick = time.monotonic()
        why = _publish(rec)
        if why:
            return why
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))

    # Settle on the trail's first posture so the next round's gates see a
    # stationary arm, not one still gliding.
    start = history[0]
    t0 = time.monotonic()
    while time.monotonic() - t0 < SETTLE_TIMEOUT_S:
        tick = time.monotonic()
        why = _publish(start)
        if why:
            return why
        t = ctx.tel.fresh()
        enc = _arm_enc(t) if t else None
        if enc is not None:
            worst = max(abs(enc[n] - start[n]) for n in ARM_JOINTS_ALL)
            if worst < SETTLE_TOL_RAD:
                print(f"[retreat] back at the round's start posture (worst "
                      f"joint {math.degrees(worst):.1f} deg) — clean seed "
                      f"for the next round")
                return None
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    print("[retreat] WARNING: settle timeout — proceeding anyway")
    return None


def glide_home_residual(ctx: Ctx, sides, home_enc: dict,
                        gated_branch: bool = True) -> None:
    """Close the post-retract residual to the EXACT home posture.

    Gate E guarantees the retract route ended in the SAME IK branch as
    home_enc (<= GATE_E_HOME_BRANCH_TOL_RAD), so the remaining gap is a
    small same-branch offset — a rate-limited joint glide across it is a
    short straight motion, not a branch crossing. Only the retract's active
    sides move; a held peer arm keeps its frozen snapshot. Best-effort: any
    fault/stale telemetry just ends the glide where it is.

    `gated_branch=False` says NO Gate E ran on the route that got us here —
    the trail-return path (08-07 audit). The glide is then conditional: it
    MEASURES the residual first and refuses if it is bigger than Gate E would
    have allowed, because the whole safety argument above is "this is a small
    same-branch offset" and nothing on that path establishes it. Refusing
    leaves the arm at the trail's end, which is a posture the C-gate already
    cleared, rather than lerping it blindly across a branch."""
    telemetry = ctx.tel.fresh()
    enc0 = _arm_enc(telemetry) if telemetry else None
    if enc0 is None or not home_enc:
        return
    arm_joints = {"left": ARM_JOINTS_L, "right": ARM_JOINTS_R}
    # Only joints home_enc actually carries. The trail path can be reached with
    # a partial (or empty) home posture — the standing single-round callers pass
    # whatever they have — and a KeyError here would abort a retract that had
    # already succeeded (08-07).
    glide_j = {n for s in sides for n in arm_joints[s] if n in home_enc}
    if not glide_j:
        return
    if not gated_branch:
        gap = max(abs(enc0[n] - home_enc[n]) for n in glide_j)
        if gap > GATE_E_HOME_BRANCH_TOL_RAD:
            print(f"[exec] skipping the home-residual glide: the trail ended "
                  f"{math.degrees(gap):.1f} deg from home, past the "
                  f"{math.degrees(GATE_E_HOME_BRANCH_TOL_RAD):.1f} deg "
                  f"same-branch bar — holding at the trail's cleared posture")
            return
    lim = ctx.args.max_rate * PUB_DT
    t0 = time.monotonic()
    while time.monotonic() - t0 < SETTLE_TIMEOUT_S:
        tick = time.monotonic()
        telemetry = ctx.tel.fresh()
        if telemetry is None or any(telemetry.get("arm_fault") or
                                    [False, False]):
            return
        enc = _arm_enc(telemetry)
        if enc is None:
            return
        worst = max(abs(enc[n] - home_enc[n]) for n in glide_j)
        if worst < SETTLE_TOL_RAD:
            print(f"[exec] home posture exact (worst joint "
                  f"{math.degrees(worst):.1f} deg)")
            return
        ctx.det.poll(ctx.rig)
        rec = {n: (enc[n] + float(np.clip(home_enc[n] - enc[n], -lim, lim))
                   if n in glide_j else float(enc0[n]))
               for n in ARM_JOINTS_ALL}
        slots = {arm: {"joint_pos": [rec[n] for n in names],
                       "rate": float(ctx.args.max_rate)}
                 for arm, names in (("left", ARM_JOINTS_L),
                                    ("right", ARM_JOINTS_R))}
        ctx.pub.publish(ctx.rig.tick(telemetry, {"arm_targets": slots}))
        time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    print("[exec] WARNING: home-residual glide timeout — holding where it is")


def execute_and_retract(ctx: Ctx, plan: Plan, target_keys, table_latch: dict,
                        empty_retry_s: float = 0.0):
    """Stream the reach, grasp, then plan and stream the retract.

    Returns (outcome, reason, held_sides, retract_bundle). held_sides is the
    hands that HOLD — or, on a lost verdict, MAY hold — a cube, and it is what
    the caller's hold-vs-release decision must be based on: refuse_holding with
    an empty hand is a robot frozen forever over nothing, and releasing a
    loaded (or maybe-loaded) one drops a cube. FATAL always comes with the true
    held set; SKIP always comes with an empty one, by construction.

    THE GRASP RETRY CONTRACT (class 1, per side since the dual-arm merge):
    a certified-empty pinch is retried — reopen, re-grab — while ITS cube is
    still seen and the retry window is open, and the retry is sent ONLY to the
    empty hands. A hand that has already confirmed contact is left completely
    alone: its arm keeps being commanded to the same waypoint it is already at
    (frozen in place) and its fingers receive no command at all — re-sending
    hand_grab to a loaded hand would run the stepped close again and squeeze
    the held cube. When the window closes: nothing held anywhere -> SKIP, the
    caller re-plans (a hand that keeps closing on nothing is in the wrong
    PLACE, and repeating the route is the livelock — class 2); something held
    -> FATAL, because walking away from a loaded hand is not an option.
    """
    a = ctx.args
    bundle, targets, floor = plan.bundle, plan.targets, plan.floor
    plan_gravity = plan.gravity
    handoff = bundle.duration_s * MPC_HANDOFF_FRACTION if a.mpc else None
    travel: list = []                # measured postures, stream + track
    settled: dict = {"ok": False}
    why, _played_s = stream_route(
        bundle, ctx.pub, ctx.tel, ctx.det, ctx.rig, targets,
        a.max_rate, until_s=handoff,
        drift_abort_m=MPC_STREAM_DRIFT_ABORT_M if a.mpc else None,
        meas_hist=travel, settled=settled)
    if why:
        # SAY WHY FIRST (2026-08-04): the round-level "aborted:" line only
        # prints after the retreat finishes, and an operator watching the
        # arm walk back has every reason to Ctrl+C before it appears — the
        # verdict then dies unread (it did, twice). Since 2026-08-05 the
        # verdict names the coded trigger and its culprit, so a recurring
        # retreat is diagnosed from the log instead of re-derived.
        print_retreat_verdict(why, f"{_played_s:.1f}s into the route")
        print(f"[exec] stream ABORTED: {why} — retreating, then this round "
              f"re-plans")
        # Nothing has been grasped yet — walk back along the MEASURED trail
        # (a wedged arm must back off the obstruction the way it came, and a
        # mid-corridor posture may sit inside the route clearance floor where
        # every re-plan is rejected at frame 1).
        note = retreat_after_track_abort(ctx, travel)
        if note:
            print(f"[retreat] {note}")
        return SKIP, why, (), None
    grasp_hold: dict | None = None
    if a.mpc:
        why, grasp_hold = mpc_track(ctx, bundle, targets, table_latch,
                                    plan_gravity=plan_gravity,
                                    meas_hist=travel,
                                    settled=bool(settled["ok"]))
        if why:
            print_retreat_verdict(why)
            print(f"[exec] MPC track ABORTED: {why} — retreating, then this "
                  f"round re-plans")
            # Nothing grasped — but do NOT just drop the arms wherever the
            # track died: an abort posture can sit inside the route-clearance
            # floor, and every re-plan from it is then rejected at frame 1
            # (hardware 2026-08-01 burned all three rounds that way). Walk
            # back along the measured trail — stream and track combined —
            # THEN let the round loop re-plan from the round's true start.
            note = retreat_after_track_abort(ctx, travel)
            if note:
                print(f"[retreat] {note}")
            return SKIP, why, (), None
    held: tuple = ()
    if a.grasp:
        cube_of = {{"L": "left", "R": "right"}[s]: c
                   for s, c in bundle.assignment.items() if c in targets}
        # BACKSTOP, not the primary defence (08-07 audit). plan_reach's
        # only_sides is what should keep a loaded hand out of the assignment;
        # this catches the case where it did not — an adopted speculative
        # bundle, a caller that forgot, a future edit. Closing a hand that
        # already holds a cube runs the stepped close AGAIN on the held cube
        # (GRASP_SQUEEZE_TORQUE drives deeper) and the service then certifies
        # CONTACT, so the mission would count a cube it never picked up. FATAL
        # rather than dropping the side: a plan built on the wrong hand is not
        # something the next round can improve on.
        rig_held = getattr(ctx.rig, "_held", {})
        clash = sorted(set(cube_of) & set(rig_held))
        if clash:
            return FATAL, (
                f"the route assigns {'+'.join(clash)} a new cube, but that "
                f"hand is already carrying "
                f"{'+'.join(rig_held[s].get('key', '?') for s in clash)} "
                f"— refusing to close a loaded gripper"), \
                tuple(sorted(rig_held)), None
        confirmed: set = set()
        attempt = tuple(sorted(cube_of))
        deadline = time.monotonic() + max(0.0, empty_retry_s)
        while True:
            why, empty = do_grasp(bundle, ctx.pub, ctx.tel, ctx.det, ctx.rig,
                                  attempt, a.max_rate, hold_enc=grasp_hold)
            if why is None:
                confirmed.update(attempt)
                break
            if not empty:
                # Indeterminate: a verdict never arrived, so every side in this
                # attempt MAY be holding its cube — some can have confirmed
                # contact moments before the other timed out. All of them count
                # as held, because the cost of guessing wrong is a dropped cube.
                return FATAL, why, tuple(sorted(confirmed | set(attempt))), None
            confirmed.update(s for s in attempt if s not in empty)
            unseen = [s for s in empty
                      if ctx.rig.median(cube_of[s], time.monotonic()) is None]
            # A SKIP HANDS THE NEXT ROUND THESE HANDS, so they must not leave
            # this function pinched shut. The caller's next act is plan_reach +
            # stream_route — the arm is driven back to a grasp pose CERTIFIED
            # FOR AN OPEN GRIPPER (build_scene keeps the cube as a hard obstacle
            # and cuRobo's goalset puts it in the open jaw's mouth), so closed
            # fingertips sweep exactly the volume the cube occupies. Worse on a
            # journey: rounds exhausted -> the leg is skipped -> the still-shut
            # fist tucks, walks and approaches the NEXT cube. Four independent
            # auditors found this; none of them could refute it.
            #
            # Opening here is provably safe: every side in `empty` is CERTIFIED
            # EMPTY by the service (run_grasp_close returns False only after
            # ARRIVING at the calibrated close position with no load), so there
            # is nothing to drop. It also clears the service's latched
            # grasp_detected/grasp_action_id — hand_open is the only thing that
            # does — without which the NEXT invocation's zero_grippers refuses
            # to calibrate ("carries a prior grasp marker") and the tool cannot
            # be restarted without a human opening the hand by hand.
            if unseen or time.monotonic() >= deadline:
                if not confirmed:
                    reopen_hands(ctx.pub, ctx.tel, ctx.det, ctx.rig, empty,
                                 bundle, a.max_rate, hold_enc=grasp_hold)
            if unseen:
                # "Unlimited while SERVABLE": an unseen cube is not servable.
                if confirmed:
                    return FATAL, (f"{why} — and the cube for "
                                   f"{'+'.join(unseen)} is no longer seen while "
                                   f"{'+'.join(sorted(confirmed))} holds"), \
                           tuple(sorted(confirmed)), None
                return SKIP, f"{why} — and the cube is no longer seen", (), None
            if time.monotonic() >= deadline:
                if confirmed:
                    return FATAL, (f"{why} — regrasp window exhausted while "
                                   f"{'+'.join(sorted(confirmed))} holds its "
                                   f"cube"), tuple(sorted(confirmed)), None
                return SKIP, f"{why} — re-planning rather than re-gripping", \
                       (), None
            reopen_hands(ctx.pub, ctx.tel, ctx.det, ctx.rig, empty, bundle,
                         a.max_rate, hold_enc=grasp_hold)
            attempt = tuple(sorted(empty))
        held = tuple(sorted(confirmed))
    telemetry = ctx.tel.fresh()
    enc_now = _arm_enc(telemetry) if telemetry else None
    if enc_now is None:
        return FATAL, "telemetry stale before retract", held, None
    def _carry_home_on_the_trail(why_fresh: str):
        """The certified way home when no FRESH route can be had (08-07 audit).

        A CUBE IS IN THE HAND at every call site below, so "give up" is not a
        neutral outcome: FATAL here ends the mission with a loaded arm parked
        over the table, and refuse_holding then stands there until a human
        comes. But a certified corridor always exists — the way we came. The
        measured trail is the path the arm physically traversed minutes ago
        (the reach C-gate cleared it whole), the cubes ride in closed hands,
        and nothing is released.

        This used to be reachable ONLY from `ret is None`. The three scene
        exceptions and the plan-server error returned FATAL 25 lines above it,
        even though the trail was equally available and equally certified —
        and the scene rebuild is hardest exactly when the table is hardest to
        see, i.e. right after a grasp put a hand in front of it."""
        print(f"[retract] {why_fresh} — carrying the cube(s) back along the "
              f"measured trail instead")
        note = retreat_after_track_abort(ctx, travel)
        if note is not None:
            print(f"[retract] trail return failed: {note}")
            return FATAL, f"{why_fresh}, and the trail return failed: {note}", \
                held, None
        # NO GATE E RAN on this path, so the glide measures before it moves.
        glide_home_residual(ctx, bundle.active_sides, plan.home_enc,
                            gated_branch=False)
        return DONE, "retracted along the measured trail", held, None

    try:
        # Every grasped cube leaves the obstacle set: it rides in a hand now and
        # its last SEEN position is where it used to be. (v1 limit: removed, not
        # attached — the retract does not know the hand got bigger.)
        scene_ret, _ = build_scene(a, ctx.rig, (), exclude=target_keys,
                                   latched=table_latch,
                                   optional_tables=getattr(a, 'journey', False))
    except (TableNotSeen, TablePoseUnstable, TargetLost) as e:
        return _carry_home_on_the_trail(f"the retract scene refused ({e})")
    try:
        ret, report = pump_during(
            lambda: plan_until_gated(
                ctx.client, scene_ret, {}, measured_enc=enc_now,
                assignment=bundle.assignment, goal="home", hand_z_floor=floor,
                home_enc=plan.home_enc, model=ctx.model, max_rate=a.max_rate,
                clearance_floor_m=a.clearance_floor_mm / 1000.0),
            ctx.pub, ctx.tel, ctx.det, ctx.rig, enc_now, a.max_rate,
            "the retract solve (arm is holding the cube)")
    except PlanServerError as e:
        return _carry_home_on_the_trail(f"the retract plan failed ({e})")
    if ret is None:
        if report is not None:
            _print_report("home (last attempt)", report)
        # The solver cannot find a FRESH 25 mm-clear way home from this grasp
        # posture (hardware 2026-08-01 night: a deep-side grasp at r=0.55
        # produced retract candidates threading the left elbow 4 mm from —
        # and even -1.6 mm INTO — the torso, five rejections, and the mission
        # ended holding the cubes waiting for a human).
        return _carry_home_on_the_trail(
            f"no gated fresh route home in {PLAN_RETRIES} attempts")
    _print_report("home", report)
    why = recheck_route_start(ret, ctx.tel)
    if why:
        return FATAL, why, held, None
    ctx.route_pub.publish(route_preview_payload(ret, ctx.model))
    why, _ = stream_route(ret, ctx.pub, ctx.tel, ctx.det, ctx.rig, None,
                          a.max_rate)
    if why:
        return FATAL, why, held, None
    # Gate E pinned the branch; this closes the last few degrees so the
    # mission ends at the SAME posture it started from (hardware 2026-08-01
    # night: hands home, left shoulder visibly not).
    glide_home_residual(ctx, ret.active_sides, plan.home_enc)
    return DONE, None, held, ret


def _pump(ctx: Ctx, nav=None, packet=None) -> dict | None:
    """One 50 Hz mission tick: telemetry in, gaze + optional nav/arm out.

    Returns the telemetry, or None when it went stale. EVERY journey wait loop
    goes through here so none of them can accidentally be a silent loop.

    "Not silent" MEANS ARM CONTENT, and `packet` is where it comes from. Gaze
    alone does not count: real_env's freshness check reads ARM keys only
    ("arm_targets"/"kp"/"kd"), so a loop calling this with packet=None streams
    gaze at 20 Hz while the arm-silence failsafe crawls both arms toward the
    default pose at 0.025 rad/s. Two callers did exactly that (2026-07-31) —
    the 120 s survey and the 40 s re-acquire. Every call site now passes either
    a tuck or a hold.
    """
    telemetry = ctx.tel.fresh()
    ctx.det.poll(ctx.rig)
    out = ctx.rig.tick(telemetry or {}, packet)
    if out:
        ctx.pub.publish(out)
    if ctx.nav_pub is not None:
        ctx.nav_pub.publish({"nav_cmd": [float(nav[0]), 0.0, float(nav[1])]
                             if nav else [0.0, 0.0, 0.0]})
    return telemetry


def _fatal_telemetry(telemetry) -> str | None:
    if telemetry is None:
        return "telemetry stopped — real_env is gone or the robot is off"
    if any(telemetry.get("arm_fault") or [False, False]):
        return "ARM FAULT latched by the dropout watchdog"
    return None


def run_leg(ctx: Ctx, key: str, leg_no: int, n_legs: int) -> tuple[str, str]:
    """Walk to one cube and cuRobo the grasp. Returns (DONE|SKIP|FATAL, reason).

    THE SHAPE IS THE POINT. Every wait below is bounded and every bounded wait
    expires into SKIP, never into standing there. The 2d1f9df audit found six
    livelock classes in the operator's journey and all six were the same
    mistake in different clothes: a state with no exit. So:

      tuck   -> JOURNEY_TUCK_TIMEOUT_S   (operator class 5: every journey run
                                          deadlocked at journey_arms_tucked)
      walk   -> JOURNEY_WALK_TIMEOUT_S
      still  -> JOURNEY_STILL_TIMEOUT_S
      acquire-> JOURNEY_ACQUIRE_S        (operator class 6: the ASSIGN stall)
      plan   -> JOURNEY_LEG_PLAN_ROUNDS  (operator class 4: latch-fail budget)
      grasp  -> unlimited re-grabs INSIDE a round while the cube is still seen
                (operator class 1, the CONTRACT), bounded by the rounds above
                (operator class 2, the pre-grab convergence timeout)

    FATAL is reserved for things no leg can recover from and no next leg would
    survive: an arm fault, telemetry loss, or a cube already in the hand when
    something goes wrong (there the arms are HELD, not crawled).
    """
    rig, a = ctx.rig, ctx.args
    cube = rig.cubes[key]
    # FILTERED, NOT RAW (hardware 2026-08-07). This read `cube.pos`, which is
    # the last ingested detection with NO freshness test and NO evidence
    # filter — one grazing 1-face PnP fit sets it, and on a two-cube run the
    # far cube's claim churns every DET_UNCLAIM_S so those marginal fits are
    # exactly what lands. A bad y sign then decided the serving arm and killed
    # the leg before a step was taken: observed live, cube on the robot's LEFT,
    # refused as right-side. median() is the same window with the best-face
    # filter and an age bound, and it was sitting right here unused.
    fix_src = "filtered fix"
    pos0 = rig.median(key, time.monotonic())
    if pos0 is None:
        pos0 = cube.pos
        fix_src = "RAW last sighting"
        if pos0 is not None:
            print(f"[journey]   WARNING: no filtered fix for {key} at leg "
                  f"start — falling back to the last raw sighting")
    if pos0 is None:
        return SKIP, f"{key} has no position at leg start"
    # WHICH HANDS ARE EMPTY (08-07 audit). Nothing in the plan, nav or grasp
    # path read rig._held before this line existed, so leg 2 could nominate,
    # steer toward, plan onto and finally re-close the hand already carrying
    # leg 1's cube — and the EE service certifies that stalled close as CONTACT,
    # so the mission reports a cube it never picked up.
    free = tuple(s for s in ("left", "right") if s not in rig._held)
    if not free:
        return SKIP, (f"both hands are already carrying a cube — nothing left "
                      f"to grasp {key} with")
    # AIM AT A HAND THAT CAN ACTUALLY TAKE IT, THEN WALK (hardware 2026-08-07).
    # This used to SKIP outright when the cube's own side was full. But the
    # nav law's aim offset exists precisely to park the cube in the serving
    # arm's half-space, and a leg is 0.6-0.8 m of travel — so a cube on the
    # full hand's side is a geometry the BASE can fix, not a dead end. Aim at
    # the free hand instead and let the approach do the work; the side is
    # re-decided from a fresh fix after arrival (below), which is the only
    # reading that can be trusted anyway. The 07-30 INFEASIBLE measurement
    # that motivated the old refusal was taken STANDING — arm alone, base
    # fixed — and does not describe what a leg can do.
    arm = journey_serving_arm(pos0, free)
    steering_across = arm is None
    if steering_across:
        arm = free[0]
    leg_dir = journey_leg_dir(pos0)
    print(f"\n[journey] LEG {leg_no}/{n_legs} → {key}, serving arm "
          f"{arm.upper()} (free: {'+'.join(free)}), "
          f"direction {'forward' if leg_dir > 0 else 'reverse'}"
          + ("  [steering it ACROSS the midline into the free hand]"
             if steering_across else ""))
    # The evidence behind the PROVISIONAL choices above (user 2026-08-09: the
    # log must show what decided them). Both are re-decided from the settled
    # fix after arrival — this line is the walk's steering premise, not the
    # grasp commitment.
    print(f"[journey]   decided from the {fix_src}: cube at "
          f"x={float(pos0[0]):+.2f}, y={float(pos0[1]):+.2f} m — "
          f"y {'>=' if float(pos0[1]) >= 0.0 else '<'} 0 nominates the "
          f"{'left' if float(pos0[1]) >= 0.0 else 'right'} arm, "
          f"x {'>=' if leg_dir > 0 else '<'} 0 sets "
          f"{'forward' if leg_dir > 0 else 'reverse'}; the aim parks the "
          f"cube "
          f"{(OP.JOURNEY_AIM_OFFSET_M if leg_dir > 0 else OP.JOURNEY_AIM_OFFSET_BACK_M):.2f}"
          f" m beside the {arm.upper()} shoulder"
          + ("" if leg_dir > 0 else
             f" (aim biased {OP.JOURNEY_REVERSE_DRIFT_COMP_M:.2f} m RIGHT "
             f"against the measured backward-gait left drift)"))

    # -- WARM START (user 08-10). The leg's first real solve used to pay
    # cuRobo's cold costs at the stop point, robot standing, user watching:
    # the once-per-shape-per-server-life CUDA-graph capture plus the
    # hand-z-floor session rebuild (logged "solve 2.53 s / 3.08 s" with the
    # "> 2s" banner on every leg's first solve). A THROWAWAY solve with this
    # leg's shape — its cuboid count, its serving arm's assignment — and its
    # predicted floor (from the pre-walk fix's z: that is the stand height,
    # which survives the walk; x/y are irrelevant to warmth) runs on a
    # background thread DURING the walk. The route is discarded; the warmed
    # kernels and the rebuilt floor session are the product. The REQ socket
    # is lockstep, so the plan rounds below JOIN this thread before their
    # first RPC; the thread is bounded by the client's own plan timeout.
    if ctx.warm_thread is not None and ctx.warm_thread.is_alive():
        ctx.warm_thread.join(timeout=60.0)   # a previous leg's stray warm-up
    warm_enc = _arm_enc(ctx.tel.fresh() or {})
    if warm_enc is not None and ctx.client is not None:
        held_keys = {h.get("key") for h in rig._held.values()}

        def _leg_warm(target_key=key, z=float(pos0[2]),
                      x=0.40 * float(leg_dir),
                      y=(0.25 if arm == "left" else -0.25),
                      side=("L" if arm == "left" else "R"),
                      cuboid_keys=tuple(k for k in a.cubes
                                        if k not in held_keys),
                      enc0=dict(warm_enc)):
            try:
                floor0 = hand_floor_for_tables(rig, a.tables, target_z=z,
                                               target_xy=(x, y))
                ws, spread = {}, 0.0
                for k in cuboid_keys:
                    d = _cube_dims(k)
                    px, py = (x, y) if k == target_key \
                        else (x - 0.20 - spread, -y)
                    spread += 0.15
                    ws[k] = {"dims": [d, d, d + CUBE_OBSTACLE_Z_PAD_M],
                             "pose": [px, py, z, 1.0, 0.0, 0.0, 0.0]}
                t0w = time.monotonic()
                ctx.client.plan(ws, {target_key: np.array([x, y, z])},
                                measured_enc=enc0,
                                assignment={side: target_key}, goal="reach",
                                hand_z_floor=floor0)
                print(f"[plan] leg solver warmed in the background "
                      f"({time.monotonic() - t0w:.1f}s — shape + z-floor, "
                      f"paid during the walk, not at the stop point)")
            except PlanServerError as e:
                print(f"[plan] background warm-up skipped ({e}) — the first "
                      f"real solve pays the cold cost as before")

        ctx.warm_thread = threading.Thread(target=_leg_warm, daemon=True)
        ctx.warm_thread.start()

    if ctx.nav_pub is not None:
        # -- tuck: walking never starts before the arms are at the walk pose --
        # The walk needs the search back: release the scan hold the previous
        # leg's fetch engaged (see GazeRig.scan_hold) so the eyes may sweep
        # for this leg's cube while the base is legitimately in motion.
        rig.scan_hold = False
        rig._journey_walk_pose = True
        t0 = time.monotonic()
        while not journey_arms_tucked(rig):
            telemetry = _pump(ctx, packet=journey_tuck_packet(rig))
            fatal = _fatal_telemetry(telemetry)
            if fatal:
                return FATAL, fatal
            if time.monotonic() - t0 > JOURNEY_TUCK_TIMEOUT_S:
                return SKIP, (f"arms never reached the walking tuck in "
                              f"{JOURNEY_TUCK_TIMEOUT_S:.0f}s")
            time.sleep(PUB_DT)
        print(f"[journey]   tucked in {time.monotonic()-t0:.0f}s — walking")

        # -- walk --------------------------------------------------------------
        # CURVED WALKING ONLY (upstream ruling, 2026-08-08): no in-place rotation, no
        # decoupled turn-then-straight legs — heading changes happen while
        # walking, through the pursuit law's wz. The decision followed the
        # --wz bench that day: from standstill this checkpoint winds its torso
        # at 0.2 rad/s (net ~1 deg) and near-falls with dragging feet at 0.4.
        # An align-in-place phase was implemented, review-hardened and then
        # REMOVED under this rule; humanoid_nav_step_test --wz keeps the
        # measurement harness should a turning-capable checkpoint ever ship.
        t0, say_t = time.monotonic(), -1e9
        paused_since = None
        wz_out = 0.0        # slewed yaw actually published (JOURNEY_WZ_SLEW)
        last_tick = time.monotonic()
        while True:
            now = time.monotonic()
            dt = min(now - last_tick, 0.25)
            last_tick = now
            live = rig.latest(key, now)
            if live is None:
                # Blind: STOP the base rather than dead-reckon. The operator's
                # DET_STALE_S rule — a base-frame target rots as the base moves,
                # so walking on a stale fix walks at a ghost.
                nav = (0.0, 0.0)
            else:
                nav = journey_nav(live, arm, leg_dir)
                # Reverse legs brake worse (both hardware landings overshot
                # into the wind-up basin) — they get the larger arrival radius.
                arrive_m = OP.JOURNEY_ARRIVE_M_BACK if leg_dir < 0 \
                    else OP.JOURNEY_ARRIVE_M
                if float(np.hypot(live[0], live[1])) <= arrive_m:
                    break
                if now - say_t > 3.0:
                    say_t = now
                    print(f"[journey]   walking: dist "
                          f"{float(np.hypot(live[0], live[1])):.2f} m, "
                          f"vx {nav[0]:+.2f}, wz {nav[1]:+.2f}")
            # SLEW THE YAW (sim MAX_ACCEL parity): the law's wz is a desire,
            # the published wz walks toward it at JOURNEY_WZ_SLEW. Without
            # this the unfaced floor is bang-bang at the face boundary — the
            # measured leg-2 slalom (wz -0.35/+0.35/-0.50). vx passes raw.
            step = OP.JOURNEY_WZ_SLEW * dt
            wz_out += float(np.clip(nav[1] - wz_out, -step, step))
            nav = (nav[0], wz_out)
            telemetry = _pump(ctx, nav=nav, packet=journey_tuck_packet(rig))
            fatal = _fatal_telemetry(telemetry)
            if fatal:
                return FATAL, fatal
            # NAME THE JOYSTICK LATCH RATHER THAN TIME OUT AGAINST IT (08-07
            # audit). real_env drops every nav_cmd while _nav_paused is set, and
            # only [NAV_RESUME] (LB) clears it — so a single stick nudge before
            # launch makes this loop's arrival condition unsatisfiable and every
            # leg burns JOURNEY_WALK_TIMEOUT_S while the log prints the speeds it
            # is publishing into a void. A pause is not itself an error (the
            # operator may be taking manual control on purpose), so allow a
            # handback window; past it, this is FATAL rather than SKIP, because
            # the next leg would fail identically and a skipped leg is never
            # retried.
            if telemetry is not None and telemetry.get("nav_paused"):
                if paused_since is None:
                    paused_since = now
                    print("[journey]   NAV PAUSED by the joystick override — "
                          "real_env is ignoring our nav commands. Press LB to "
                          "hand navigation back.")
                elif now - paused_since > JOURNEY_NAV_HANDBACK_S:
                    return FATAL, (
                        f"navigation has been joystick-paused for "
                        f"{JOURNEY_NAV_HANDBACK_S:.0f}s — real_env discards "
                        f"nav_cmd while _nav_paused is latched, so no leg can "
                        f"arrive. Press LB (NAV_RESUME) and restart.")
            elif paused_since is not None:
                paused_since = None
                print("[journey]   navigation handed back — resuming the walk")
            # Driving resets the stillness timer: leg 1's stale timer must never
            # satisfy leg 2's arrival check (operator, 07-21 review). It also
            # stamps the base-motion epoch: every sighting captured up to here
            # is in a frame the walk is destroying (see _best_window's fence).
            rig._journey_still_since = None
            rig._base_moved_t = now
            if now - t0 > JOURNEY_WALK_TIMEOUT_S:
                return SKIP, (f"never reached {OP.JOURNEY_ARRIVE_M:.2f} m of "
                              f"{key} in {JOURNEY_WALK_TIMEOUT_S:.0f}s")
            time.sleep(PUB_DT)
        print(f"[journey]   arrived — waiting for MEASURED base stillness …")

        # -- stillness: the command ramp at zero is not the robot standing still
        t0 = time.monotonic()
        while not journey_base_still(rig, time.monotonic()):
            # Still settling = still moving: keep the base-motion epoch
            # current so the deceleration's sightings (the base covers ~10 cm
            # ramping down from walk speed) stay out of the stable window.
            rig._base_moved_t = time.monotonic()
            telemetry = _pump(ctx, packet=journey_tuck_packet(rig))
            fatal = _fatal_telemetry(telemetry)
            if fatal:
                return FATAL, fatal
            if time.monotonic() - t0 > JOURNEY_STILL_TIMEOUT_S:
                return SKIP, (f"base never settled within "
                              f"{JOURNEY_STILL_TIMEOUT_S:.0f}s of arriving")
            time.sleep(PUB_DT)
        print(f"[journey]   base still — releasing the tuck")
        rig._journey_walk_pose = False
        if a.chest_home:
            why = raise_to_chest_home(ctx.pub, ctx.tel, rig, ctx.det)
            if why:
                return SKIP, f"could not return to the chest home: {why}"

    # -- re-acquire LIVE at the stop point ------------------------------------
    # The survey is trusted for ORDER only: base-frame positions rot as the base
    # moves, so the position that got us here is not the position we grasp.
    t0, say_t = time.monotonic(), -1e9
    t_seen = None
    while True:
        now = time.monotonic()
        pos = rig.median(key, now)
        why = None if pos is None else target_unreachable_why(pos)
        if pos is not None and why is None:
            # STABILITY GATE (08-12, user): a fix may only become the plan
            # anchor when its samples AGREE — a jumping estimate that happens
            # to be reachable used to be adopted on its first appearance and
            # the round then planned a waypoint into empty space. Bounded:
            # a stable fix passes instantly; an unstable one gets at most
            # TARGET_STABLE_EXTRA_S of extra sampling, then we proceed on
            # the best evidence in hand (the rounds and the leash own it).
            t_seen = t_seen or now
            ev = rig.median_evidence(key, now)
            spread, n = ev if ev is not None else (float("inf"), 0)
            if spread <= TARGET_STABLE_SPREAD_M and n >= TARGET_STABLE_MIN_N:
                break
            if now - t_seen > TARGET_STABLE_EXTRA_S:
                print(f"[journey]   {key}: fix still unstable after "
                      f"{TARGET_STABLE_EXTRA_S:.0f}s (spread "
                      f"{spread * 1000:.0f} mm over {n} samples) — "
                      f"proceeding on the best evidence in hand")
                break
            if now - say_t > 1.0:
                say_t = now
                print(f"[journey]   {key}: fix unstable (spread "
                      f"{spread * 1000:.0f} mm over {n} samples) — letting "
                      f"it settle")
        else:
            t_seen = None
        # STAY TUCKED while re-acquiring. This wait is JOURNEY_ACQUIRE_S = 40 s
        # long and used to publish gaze alone, which does not reset the arm
        # freshness timer: 39.5 s of failsafe crawl is 56 deg per joint, undoing
        # the tuck the leg just walked in.
        telemetry = _pump(ctx, packet=journey_tuck_packet(rig))
        fatal = _fatal_telemetry(telemetry)
        if fatal:
            return FATAL, fatal
        if now - t0 > JOURNEY_ACQUIRE_S:
            return SKIP, (f"{key} not confirmed-and-reachable within "
                          f"{JOURNEY_ACQUIRE_S:.0f}s of the stop point"
                          + (f": {why}" if why else " (never re-seen)"))
        if now - say_t > 5.0:
            say_t = now
            print(f"[journey]   re-acquiring {key}"
                  + (f" — {why}" if why else " (scanning)"))
        time.sleep(PUB_DT)
    # TARGET LOCKED — hold the search until the next walk (08-12): from here
    # through plan/stream/MPC/grasp/retract the base must stand still, and a
    # serpentine on the free eye is what ground the base rightward through
    # every front fetch (both checkpoints; the trained gaze-follow coupling
    # recruits the LEGS, so even a pinned waist could not stop it). The
    # tracking eye keeps its claim and stares; any eye with nothing to track
    # parks at origin until the next leg's tuck releases the hold.
    rig.scan_hold = True
    # THE SIDE IS DECIDED HERE, NOT AT LEG START (hardware 2026-08-07). `pos`
    # is filtered AND measured from the stop point, i.e. from the geometry the
    # arm actually has to solve — every earlier reading was taken from
    # somewhere the robot no longer is. Only now can a cube on the full hand's
    # side be called genuinely unreachable, and only now is a SKIP honest.
    settled_arm = journey_serving_arm(pos, free)
    if settled_arm is None:
        held_side = "left" if "left" in rig._held else "right"
        return SKIP, (
            f"{key} is at y={float(pos[1]):+.3f} after the approach — "
            f"{abs(float(pos[1])) * 1000.0:.0f} mm into the {held_side} "
            f"half-space, past the {OP.CROSS_MIDLINE_REACH_M * 1000.0:.0f} mm "
            f"the free hand can cross (07-30 sweep), and the {held_side} hand "
            f"already carries a cube. Place the two cubes on OPPOSITE sides of "
            f"the robot's own midline (+y is its LEFT).")
    if settled_arm != arm:
        print(f"[journey]   the approach moved {key} to y={float(pos[1]):+.3f} "
              f"— serving arm is {settled_arm.upper()}, not {arm.upper()}")
        arm = settled_arm
    else:
        # Silent-confirmation was a log gap (user 2026-08-09): the FINAL arm
        # commitment comes from the settled fix, and the log must say so even
        # when it agrees with the pre-walk nomination.
        print(f"[journey]   serving arm CONFIRMED {arm.upper()} from the "
              f"settled fix (y={float(pos[1]):+.3f}) — the grasp is "
              f"restricted to this hand")
    print(f"[scene] target {key} at {np.round(pos, 3)}  "
          f"r_xy={np.hypot(pos[0], pos[1]):.3f} m")

    # -- z FAST-FAIL (one-shot doctrine, 08-09): height is a property of the
    # stand, not the stance — outside the measured band no round can win, so
    # say the physical fix in one line instead of fighting 30 solves.
    # ONLY on >=2-face evidence (review 08-09, both lenses): a mission-ending
    # verdict must clear the same bar every control decision does
    # (DRIFT_MIN_INLIERS) — 1-face z systematics run tens of mm against a
    # 65 mm band, and telling the operator to move a correctly-placed stand
    # on one grazing frame is worse than letting the solver rounds judge.
    z_why = journey_z_verdict(float(pos[2]), a.clearance_floor_mm)
    if z_why is not None:
        if rig.faces_used(key, time.monotonic()) >= DRIFT_MIN_INLIERS:
            return SKIP, z_why
        print(f"[journey]   WARNING: {z_why} — but the fix is 1-face "
              f"(bar {DRIFT_MIN_INLIERS}), not evidence enough to end the "
              f"leg; letting the plan rounds judge")
    elif float(pos[2]) > OP.JOURNEY_CUBE_Z_MAX:
        print(f"[journey]   cube z={float(pos[2]) * 1000:+.0f} mm sits above "
              f"the floor-{OP.JOURNEY_Z_LEDGER_FLOOR_MM:.0f} band cap, but "
              f"the {a.clearance_floor_mm:.0f} mm clearance floor unlocks "
              f"the wrist-fold corridor — attempting (user 08-09: accept "
              f"the higher cube; the solver judges)")

    # -- plan / reach / grasp, bounded rounds ---------------------------------
    # DIRECT TO THE GRASP POINT (user 2026-08-09, one-shot doctrine: "no more
    # 15 mm hover — straight to the grasp point"): every journey round plans
    # with hover 0. The hover lifts route endpoints 15 mm, which is exactly
    # the margin a close/high landing steals from the wrist-torso corridor —
    # and with it gone the MPC converges on the grasp point itself (the
    # pre-08-03 validated shape) and the descent phase (with its 1-face
    # freeze politics) simply does not exist on journey legs. ctx.args is the
    # single source both the planner and the tracker read, so it is mutated
    # for the leg and ALWAYS restored (the standing mission keeps its hover).
    hover_mm0 = float(a.reach_hover_mm)
    if a.reach_hover_mm != 0.0:
        a.reach_hover_mm = 0.0
        print(f"[journey]   planning DIRECT to the grasp point (hover 0; "
              f"the standing mission's {hover_mm0:.0f} mm hover does not "
              f"apply to journey legs)")
    # The warm-up thread and the plan rounds share one lockstep REQ socket:
    # join it (pumped, arms held at the settled posture) before the first
    # real RPC. Normally the thread died mid-walk and this costs nothing.
    if ctx.warm_thread is not None and ctx.warm_thread.is_alive():
        print("[plan] waiting for the background warm-up to hand back the "
              "plan socket …")
        enc_h = _arm_enc(ctx.tel.fresh() or {})
        while ctx.warm_thread.is_alive():
            telemetry = _pump(ctx, packet=(hold_packet(enc_h, a.max_rate)
                                           if enc_h is not None else None))
            fatal = _fatal_telemetry(telemetry)
            if fatal:
                return FATAL, fatal
            time.sleep(PUB_DT)
    last = "no attempt made"
    try:
        for rnd in range(1, JOURNEY_LEG_PLAN_ROUNDS + 1):
            t_now = ctx.tel.fresh()
            if t_now is not None and any(t_now.get("arm_fault") or [False, False]):
                # The watchdog latch is TERMINAL (hardware 2026-08-01: sh2
                # dropped mid-track and the mission then burned two 20 s
                # re-acquires with a latched arm parked over the table).
                return FATAL, ("ARM FAULT latched by the dropout watchdog — "
                               "inspect the harness and restart real_env")
            table_latch: dict = {}
            # only_sides: the planner cannot see a loaded gripper, so the leg
            # that can must say which hand it is allowed to use (08-07 audit).
            plan, why = plan_with_reacquire(ctx, (key,), table_latch,
                                            ctx.home_enc, only_sides=(arm,))
            if plan is None:
                last = why
                print(f"[journey]   round {rnd}/{JOURNEY_LEG_PLAN_ROUNDS}: {why}")
                continue
            if not a.execute:
                return DONE, "dry-run: route gated, nothing streamed"
            outcome, why, held, _ret = execute_and_retract(
                ctx, plan, (key,), table_latch,
                empty_retry_s=JOURNEY_EMPTY_REGRASP_S)
            # Mark _held for EVERY outcome, not only DONE: on a FATAL these
            # hands hold (or may hold) a cube, and journey_mission decides
            # hold-vs-release from rig._held — an unmarked held hand there
            # would be released with the cube still in it.
            for side in held:
                rig._held[side] = {"key": key}
            if outcome is DONE:
                cube.held_by = held[0] if held else None
                cube.reached = True
                # Retire the eye that was watching it. PERMANENT (user
                # 08-09: "stick to the origin when his mission completed —
                # never join for finding next one"): ao_gaze parks a retired
                # eye at zero unconditionally, so leg 2's search is one-eyed
                # by order (the 07-26 rejoin rule is superseded).
                if cube.last_camera_port is not None:
                    rig._cam_retired.add(cube.last_camera_port)
                return DONE, f"grasped by {'+'.join(held) or 'nobody'}"
            if outcome is FATAL:
                return FATAL, why
            last = why
            print(f"[journey]   round {rnd}/{JOURNEY_LEG_PLAN_ROUNDS} "
                  f"aborted: {why}")
    finally:
        a.reach_hover_mm = hover_mm0
    return SKIP, f"{JOURNEY_LEG_PLAN_ROUNDS} rounds exhausted — last: {last}"


def journey_tuck_packet(rig: GazeRig) -> dict | None:
    """Joint-space glide of the FREE arms to the walking tuck, or None.

    Joint space, not Cartesian: the operator moved this off the Cartesian
    home_glide on 07-21 because straight-line EE plus IK left the arm up to
    62 deg off the true tuck posture. Held arms are exempt — they ride with
    their cube.

    A FREE ARM IS COMMANDED EVERY TICK, even once it is already inside
    JOURNEY_POSTURE_TOL_RAD of the tuck. It used to be dropped from the packet
    at that point, which turned the packet into None the moment the last free
    arm converged — and None publishes no arm content, so real_env's 0.5 s
    silence failsafe took the arms right after they arrived and crawled them at
    0.025 rad/s for the rest of the walk. Re-commanding a target the arm is
    already at costs nothing and keeps the freshness timer alive; the tuck is a
    FIXED posture, not a live measurement, so re-sending it cannot walk the arm
    downhill the way republishing the encoders would (see hold_packet).
    """
    jp = rig._jpos
    if jp is None or len(jp) < 27 or not getattr(rig, "_journey_walk_pose", False):
        return None
    slots = {}
    for i, arm in enumerate(("left", "right")):
        if arm in rig._held:
            continue
        seg = np.asarray(jp[13:20] if i == 0 else jp[20:27], dtype=float)
        tgt = np.asarray(OP.mirror_arm(OP.POWERON_JOINTS, arm), dtype=float)
        if not np.all(np.isfinite(seg)):
            continue
        slots[arm] = {"joint_pos": [float(v) for v in tgt],
                      "rate": OP.STAGE_RATE}
    return {"arm_targets": slots} if slots else None


def forget_all_cubes(rig: GazeRig) -> None:
    """Drop every remembered cube position, so the scan actually runs.

    WHY THIS EXISTS (user, 2026-08-08). `zero_gimbals` drives the cameras to
    the origin and, in the SAME loop, calls `det.poll(rig)` on every tick. With
    the monitor left running across a relaunch the cubes are still being
    decoded, so positions and first-seer locks land DURING the drive to zero —
    and `eye()` then aims straight at them. The scan phase prints its banner and
    is immediately skipped, because `mine` is non-None on the first tick.

    That is deliberate for iteration (the 08-03 fix: "polling here lets
    first-seer locks land before the scan phase even starts", added because the
    operator clocked 6-8 s of "camera turning, nothing reacts"). But it makes
    the reset a lie: the cameras return to the origin having quietly kept
    everything they knew, so the see->scan->fix->grasp sequence CANNOT BE
    DEMONSTRATED without tearing down all five processes.

    So: opt in with --fresh-scan and the origin means what it says. Clears
    everything a position could hide in — pos, the sighting history the median
    reads, the freshness clocks, and the camera claims, since a surviving claim
    would aim an eye at a cube it can no longer justify.
    """
    for cube in rig.cubes.values():
        cube.pos = None
        cube.pos_port = None
        cube.hist.clear()
        cube.last_seen = -1e9
        cube.first_seen = -1e9
        cube.camera_port = None
        cube.last_camera_port = None
    rig._cam_retired.clear()
    rig._scan_phase = 0.0
    print(f"[gaze] --fresh-scan: forgot every cube position — the cameras "
          f"start from the origin knowing nothing, and the scan runs for real")


def journey_mission(ctx: Ctx) -> int:
    """Survey once, then one leg per cube in near-first order.

    A FOR LOOP OVER A LATCHED LIST, and that is the anti-deadlock argument. The
    operator re-queues skipped cubes, which is where its livelocks lived —
    assign, stall, cancel, reassign, forever. Here the survey latches the order
    once, every cube is attempted exactly once, and a leg can only end DONE,
    SKIP or FATAL. The mission therefore terminates whatever the world does.
    """
    rig = ctx.rig
    t0, deadline = time.monotonic(), time.monotonic() + DET_WAIT_S
    while not journey_survey(rig, time.monotonic()):
        # HELD for the survey, for the same reason the standing scan is: this
        # runs for up to DET_WAIT_S = 120 s, which is more than enough failsafe
        # crawl to undo the chest-home raise that happened directly above.
        telemetry = _pump(ctx, packet=hold_packet(ctx.home_enc, ctx.args.max_rate))
        fatal = _fatal_telemetry(telemetry)
        if fatal:
            return refuse(fatal, 4)
        if time.monotonic() > deadline:
            missing = [k for k, c in rig.cubes.items()
                       if not c.confirmed(time.monotonic())]
            return refuse(f"survey never completed in {DET_WAIT_S:.0f}s — "
                          f"{missing} never confirmed. Every cube must be seen "
                          f"from the start point before the first leg: the "
                          f"service order is latched from that one standpoint.",
                          2)
        time.sleep(PUB_DT)
    order = list(rig._survey_order)
    done, skipped = [], []
    for i, key in enumerate(order, 1):
        outcome, why = run_leg(ctx, key, i, len(order))
        if outcome is FATAL:
            print(f"[journey] LEG {i}/{len(order)} FATAL — {why}")
            return refuse_holding(why, 5, ctx.pub, ctx.tel, ctx.det, rig,
                                  ctx.args.max_rate, holding=bool(rig._held),
                                  held_sides=tuple(sorted(rig._held)))
        if outcome is SKIP:
            skipped.append((key, why))
            # A SKIPPED CUBE IS OUT OF THE MISSION, AND THE EYES MUST HEAR IT
            # (hardware 2026-08-07). `reached` was set on DONE only, so a
            # skipped cube stayed "pending" forever: ao_gaze.eye keeps one
            # camera aimed at it and the other sweeping, and the claim expires
            # and is re-taken every DET_UNCLAIM_S. Observed as an endless
            # lock/release churn through the whole final hold, with a cube in
            # the hand and nobody left to look for. The park-at-zero branch
            # requires EVERY cube reached, so one skip disables it entirely.
            # `reached` is the right flag: the loop never re-queues, so
            # skipped and grasped are identical to everything downstream —
            # only a restart retries this cube.
            rig.cubes[key].reached = True
            print(f"[journey] LEG {i}/{len(order)} SKIPPED — {key}: {why}")
            if not ctx.args.journey_continue_on_skip:
                # STOP ON SKIP (user 2026-08-08, after watching leg 1 fail to
                # plan and the robot walk away toward leg 2): a skipped leg
                # means the world at THIS cube disagrees with the plan, and
                # walking on buries that failure 1-2 m of travel away from its
                # evidence. End the mission standing right here — holding
                # whatever earlier legs grasped — so the operator can fix the
                # cube (or the constant) and restart. The 07-27 anti-deadlock
                # argument is untouched: a stop terminates at least as surely
                # as walking on did, and a skipped leg is still never retried
                # in-mission.
                left = list(order[i:])
                return refuse_holding(
                    f"leg {i}/{len(order)} ({key}) did not grasp: {why} — "
                    f"stopping here instead of walking on"
                    + (f" (not attempted: {', '.join(left)})" if left else "")
                    + ". Fix the cube and restart to retry, or pass "
                    "--journey-continue-on-skip for the old salvage "
                    "behaviour.",
                    5, ctx.pub, ctx.tel, ctx.det, rig, ctx.args.max_rate,
                    holding=bool(rig._held),
                    held_sides=tuple(sorted(rig._held)))
            print(f"[journey]   continuing to the next cube "
                  f"(--journey-continue-on-skip; restart to retry this one)")
            continue
        done.append(key)
        print(f"[journey] LEG {i}/{len(order)} DONE — {key}: {why}")
    print(f"\n[journey] mission complete in {time.monotonic()-t0:.0f}s — "
          f"{len(done)}/{len(order)} carried")
    for key, why in skipped:
        print(f"[journey]   SKIPPED {key}: {why}")
    if rig._held:
        # A SUCCESSFUL journey ends with cubes in the hands, and returning here
        # stops every publisher — real_env's 0.5 s arm-silence failsafe would
        # then crawl LOADED arms toward the default pose at 0.025 rad/s, with
        # nobody watching, at the very end of a run that worked. The standing
        # mission has always held its posture at PUB_HZ until Ctrl+C; the
        # journey exited instead. Same ending now, and the message says what
        # is in the hands.
        return refuse_holding(
            f"carried {'+'.join(sorted(rig._held))} — mission complete",
            0, ctx.pub, ctx.tel, ctx.det, rig, ctx.args.max_rate, holding=True,
            held_sides=tuple(sorted(rig._held)))
    return 0


def main() -> int:
    """Entry point. Parses Args, checks the world declaration, then runs the
    mission in the order the module docstring draws: plan-server ping (waits
    SERVER_WAIT_S, refuses on gpu_ok=False), telemetry + startup gates, gimbal
    and gripper zeroing, optional chest home, --journey (journey_mission) or
    the standing scan -> table wait -> plan/execute rounds -> dry-run watch or
    post-retract hold. Returns the process exit code (0 ok, 2-6 refusals);
    the finally block prints the retreat tally and clears the route preview."""
    args = tyro.cli(Args)
    complaint = _check_world(args)
    if complaint:
        print(f"[args] {complaint}")
        return 2

    model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    tel = TelemetryView(args.robot_ip)
    det = DetectionView(args.detection_url)
    client = PlanClient(args.server, model=model)
    # 9874 bound from the start (operator must be OFF): the detection wait
    # needs gaze packets — a forward-staring camera cannot see a low cube.
    pub = NNGPublisher("tcp://*:9874")
    route_pub = NNGPublisher(ROUTE_PREVIEW_URL)
    # 9873 ONLY on an executing journey. A dry-run journey (no --execute)
    # skips the tuck/align/walk/stillness block entirely (run_leg gates it on
    # nav_pub) and goes straight to the standing plan phases — the nav state
    # machine is exercised only when it can actually move the robot.
    nav_pub = NNGPublisher("tcp://*:9873") \
        if (args.journey and args.execute) else None
    rig = GazeRig(args.cubes, table_keys=args.tables)
    time.sleep(0.3)                          # let real_env's dialer attach
    try:
        # -- server reachable -------------------------------------------------
        # WAIT AND RETRY, never insta-refuse. This check runs BEFORE any
        # motion, any gate, any gripper zeroing — waiting here costs nothing
        # but patience, while refusing costs the operator a full relaunch of a
        # five-terminal stack. 2026-08-01: a plan-server restart (CUDA warm-up
        # ~60 s) plus this insta-refuse produced three dead-on-arrival missions
        # in a row and was the largest single source of REFUSED that day.
        pong, t0 = None, time.monotonic()
        while pong is None:
            try:
                pong = client.ping()
            except Exception as e:  # noqa: BLE001
                waited = time.monotonic() - t0
                if waited >= SERVER_WAIT_S:
                    return refuse(
                        f"plan server unreachable at {args.server} for "
                        f"{SERVER_WAIT_S:.0f}s: {e}. If it is restarting, "
                        f"re-run once it prints 'listening' (~40 s after "
                        f"launch); to launch it on the GPU machine, from a "
                        f"checkout of this repository (docs/OPERATIONS.md "
                        f"section 1):\n"
                        f"    ssh <gpu-host> 'pkill -f \"plan_serve[r]\"'\n"
                        f"    ssh <gpu-host> 'cd <repo> && "
                        f"PYTHONPATH=<legged_env_v2>:$PYTHONPATH setsid nohup "
                        f"python control/curobo_plan_server.py --port 9880 "
                        f"> plan_server.log 2>&1 < /dev/null & exit 0'\n"
                        f"(two SEPARATE ssh calls: a pkill pattern spelled "
                        f"out in the same command line matches that command "
                        f"line and kills the shell before it can relaunch)", 2)
                print(f"[server] {args.server} not answering "
                      f"({type(e).__name__}) — retrying, "
                      f"{SERVER_WAIT_S - waited:.0f}s left …")
        if pong.get("gpu_ok") is False:
            # The server process is up but its CUDA context is dead — every
            # solve would hang or fault. No amount of client retrying fixes
            # this; say exactly what does (the 07-28 nvidia_uvm recovery).
            return refuse(
                f"plan server is up but reports gpu_ok=False "
                f"({pong.get('gpu_err', '')!r}) — CUDA is faulted on the GPU "
                f"machine. Recover it there (needs sudo):\n"
                f"    ssh <gpu-host>\n"
                f"    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm\n"
                f"then restart the plan server and re-run.", 2)
        print(f"[server] {args.server} alive (robot={pong.get('robot')}, "
              f"hand_z_floor={pong.get('hand_z_floor')})")

        # -- telemetry + startup gates ---------------------------------------
        print("[gate] waiting for fresh telemetry …")
        deadline = time.monotonic() + 10.0
        telemetry = None
        while telemetry is None and time.monotonic() < deadline:
            telemetry = tel.fresh()
            time.sleep(0.05)
        if telemetry is None:
            return refuse("no telemetry on 9870 — is humanoid_real_env running?", 2)
        why = _startup_gates(telemetry)
        if why and "from every known home" in why:
            # The failsafe crawl case: wait it out rather than making the
            # operator guess when 0.025 rad/s has finished.
            why = wait_for_posture(tel)
        if why:
            return refuse(why, 2)
        print("[gate] tilt/posture OK")

        # SOLVE-ON-LOCK (user 2026-08-03): the zeroing/home loops below tick
        # this — the instant every cube is locked and selected, the reach
        # solve+gates run in the background, overlapping the calibration the
        # operator is watching anyway. Standing missions only (a journey
        # plans per leg after walking). Tables are handled, not excluded
        # (2026-08-03 second pass: the original no-tables gate silently
        # turned BOTH accelerations off for every --tables run): the scene
        # builder raises until a declared table is actually seen, so
        # maybe_launch simply keeps waiting, and the floor is computed
        # inside the solve closure exactly as plan_reach computes it.
        spec = None
        if args.cubes and not args.journey:
            def _warm_reach():
                # Throwaway SAME-SHAPE solve: pays cuRobo's once-per-server
                # ~3.5 s CUDA-graph capture here, during zeroing, instead of
                # on the first real solve. The capture is per problem SHAPE,
                # so the synthetic scene mirrors the mission's cuboid count:
                # one per cube plus one slab per declared table (poses are
                # made up; the kernels are the product, the route is
                # discarded).
                t = tel.fresh()
                enc0 = _arm_enc(t) if t else None
                if enc0 is None:
                    return
                wt = {k: [0.30, y, 0.05] for k, y in
                      zip(args.cubes, (0.25, -0.25))}
                ws = {k: {"dims": [0.06, 0.06, 0.06 + CUBE_OBSTACLE_Z_PAD_M],
                          "pose": [*p, 1.0, 0.0, 0.0, 0.0]}
                      for k, p in wt.items()}
                for name in args.tables:
                    ws[name] = {"dims": [1.2, 0.6, 0.03],
                                "pose": [0.6, 0.0, -0.30, 1.0, 0.0, 0.0, 0.0]}
                t0 = time.monotonic()
                client.plan(ws, {k: np.asarray(p) for k, p in wt.items()},
                            measured_enc=enc0, goal="reach",
                            hand_z_floor=None)
                print(f"[plan] reach solver warmed in the background "
                      f"({time.monotonic() - t0:.1f}s — paid during zeroing, "
                      f"not before your grasp)")

            def _spec_solve(scene, targets, enc):
                # floor from the TRUE targets, solver aimed at the HOVER —
                # identical to plan_reach, or the adopted route would end at
                # a different height than a fresh one. Same decoupled-vs-
                # joint choice too, for the same reason.
                low_p = min((np.asarray(p, float) for p in targets.values()),
                            key=lambda p: float(p[2]))
                low_z = float(low_p[2])
                hover_m = args.reach_hover_mm / 1000.0 if args.mpc else 0.0
                lifted = hover_targets(targets, hover_m)
                floor0 = hand_floor_for_tables(rig, args.tables,
                                               target_z=low_z,
                                               target_xy=low_p[:2])
                if args.decoupled_arms and len(lifted) == 2:
                    bundle, report = plan_decoupled_until_gated(
                        client, scene, lifted, measured_enc=enc,
                        hand_z_floor=floor0, model=model,
                        max_rate=args.max_rate,
                        clearance_floor_m=args.clearance_floor_mm / 1000.0,
                        retries=SPEC_PLAN_RETRIES)
                    if bundle is not None:
                        return bundle, report
                return plan_until_gated(
                    client, scene, lifted,
                    measured_enc=enc, goal="reach", hand_z_floor=floor0,
                    model=model, max_rate=args.max_rate,
                    clearance_floor_m=args.clearance_floor_mm / 1000.0,
                    retries=SPEC_PLAN_RETRIES)

            spec = SpeculativeSolve(rig, tel, tuple(args.cubes), args,
                                    solve_fn=_spec_solve, warmup=_warm_reach)

        # Cameras to ZERO first (user 2026-08-03) — the sweep that follows
        # starts from a deterministic pose, every run.
        why = zero_gimbals(pub, tel, rig, det, args.max_rate, spec=spec)
        if why:
            return refuse(why, 2)

        # Calibrate before anything is planned: a grasp verdict from an
        # uncalibrated gripper is not evidence, and this tool's grasp contract
        # is built entirely on trusting that verdict.
        if args.grasp:
            why = zero_grippers(pub, tel, rig, det, args.max_rate, spec=spec)
            if why:
                return refuse(why, 2)

        # Raise BEFORE the scan, not after: the scan takes up to two minutes and
        # the chest home is where the reach is meant to start from, so lifting
        # afterwards would plan from one posture and depart from another.
        if args.chest_home:
            why = raise_to_chest_home(pub, tel, rig, det, spec=spec)
            if why:
                return refuse(why, 3)

        # THE MISSION'S START POSTURE, captured ONCE — after the posture gate
        # and the optional chest-home lift, so it is one of the four homes
        # posture_gate recognises. Every retract in every round and every leg
        # returns HERE, not to wherever the previous failure left the arm.
        telemetry = tel.fresh() or telemetry
        mission_home = _arm_enc(telemetry)
        if mission_home is None:
            return refuse("no arm encoders — cannot fix the posture every "
                          "retract must return to", 2)
        ctx = Ctx(args, model, tel, det, client, pub, route_pub, rig,
                  server_floor=pong.get("hand_z_floor"), nav_pub=nav_pub,
                  home_enc=mission_home, spec=spec)

        if args.fresh_scan:
            forget_all_cubes(rig)

        if args.journey:
            if not args.execute:
                print("[journey] DRY-RUN: the full state machine runs and every "
                      "nav command is computed, but 9873 is never bound — the "
                      "robot stands. Add --execute to walk.")
            return journey_mission(ctx)

        # -- detections: the operator's serpentine scan drives the cameras ----
        print(f"[scene] scanning for {args.cubes} (operator serpentine, "
              f"up to {DET_WAIT_S:.0f}s) …")
        # WAITS for a REACHABLE cube rather than refusing the first unreachable
        # one. The envelope is not relaxed by a millimetre — what changed is that
        # a cube seen out of reach is a situation you can FIX by moving it,
        # standing still the whole time, instead of a reason to tear down and
        # relaunch a five-process stack. The reason is printed on a throttle so
        # the correction to make is on screen while you make it.
        # EVERY named cube, not the first one that happens to be reachable.
        # --cubes is the GRASP SET (one arm each), so a mission asking for two
        # waits until BOTH are confirmed and BOTH are inside the envelope. Half
        # of a dual grasp is not a state this tool models: the retract plans for
        # what the reach took, and a partial start would leave one hand closed
        # on nothing.
        deadline = time.monotonic() + DET_WAIT_S
        selected: dict[str, np.ndarray] = {}
        last_why: dict[str, str] = {}
        warn_t = -1e9
        # HELD AT mission_home FOR THE WHOLE SCAN. This is the longest wait in
        # the program — DET_WAIT_S is 120 s, and the serpentine routinely uses
        # most of it — and it used to publish gaze alone. 119.5 s past the 0.5 s
        # arm timeout is far more crawl than it takes to reach default_pose, so
        # the failsafe silently UNDID the chest-home raise that happens directly
        # above, and a scan that ended early left the arms frozen mid-crawl in a
        # posture belonging to no home at all. The raise is deliberately ordered
        # before the scan so the reach departs from the posture it was planned
        # from; holding here is what makes that ordering mean anything.
        while time.monotonic() < deadline:
            loop_t = time.monotonic()
            telemetry = tel.fresh()
            out = rig.tick(telemetry or {},
                           hold_packet(mission_home, args.max_rate))
            if out:
                pub.publish(out)
            det.poll(rig)
            now = time.monotonic()
            selected.clear()
            last_why.clear()
            for key in args.cubes:
                pos = rig.median(key, now)
                if pos is None:
                    last_why[key] = f"{key}: not confirmed yet (scanning)"
                    continue
                why = target_unreachable_why(pos)
                if why is None:
                    selected[key] = pos
                else:
                    last_why[key] = f"{key} at {np.round(pos, 3)}: {why}"
            if len(selected) == len(args.cubes):
                break
            if last_why and now - warn_t > 5.0:
                warn_t = now
                for line in last_why.values():
                    print(f"[scene] WAITING — {line}")
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - loop_t)))
        if len(selected) != len(args.cubes):
            return refuse(
                f"not every cube was confirmed and reachable within "
                f"{DET_WAIT_S:.0f}s ({len(selected)}/{len(args.cubes)} ready) — "
                f"{'; '.join(last_why.values()) or 'nothing seen at all'}. "
                f"Check --bodies, tag lighting, and that each cube is within "
                f"~0.7 m", 2)
        target_keys = tuple(args.cubes)
        for key in target_keys:
            p = selected[key]
            print(f"[scene] target {key} at {np.round(p, 3)}  "
                  f"r_xy={np.hypot(p[0], p[1]):.3f} m")

        # -- declared tables must be SEEN before anything is planned ----------
        # The scan above sweeps the whole panorama, so a visible table is
        # normally already confirmed by now; this only covers the case where the
        # cube was found early in the sweep.
        if args.tables:
            deadline = time.monotonic() + TABLE_WAIT_S
            while time.monotonic() < deadline:
                loop_t = time.monotonic()
                telemetry = tel.fresh() or telemetry
                out = rig.tick(telemetry or {},
                               hold_packet(mission_home, args.max_rate))
                if out:
                    pub.publish(out)
                det.poll(rig)
                if all(rig.table_pose(n, time.monotonic()) is not None
                       for n in args.tables):
                    break
                time.sleep(max(0.0, PUB_DT - (time.monotonic() - loop_t)))

        # -- plan + execute in ROUNDS — the same contract as a journey leg ----
        # A SKIP from the execute phase (drift abort mid-route, every hand
        # empty and the regrasp window spent, a cube gone from view) means the
        # world and the route disagree — the answer is a FRESH plan from fresh
        # detections, not the end of the mission. Bounded like a leg: the same
        # JOURNEY_LEG_PLAN_ROUNDS, because the failure mode it guards (planning
        # forever at a target that never works out) is the same.
        # ⚠ SCOPE DIFFERS FROM run_leg, DELIBERATELY UNCHANGED (audit
        # 2026-08-06, I5). run_leg rebuilds this INSIDE its round loop, so each
        # round re-measures the table; here it lives outside, so round 3 plans
        # against a pose measured before round 1 — and a base-frame table pose
        # drifts with torso sway exactly as a cube does. Left as-is on purpose:
        # the latch is also what keeps a table that has gone out of view from
        # killing later rounds, the standing mission runs --no-table-world so
        # nothing is latched at all, and flipping it blind would trade a known
        # staleness for an unknown TableNotSeen. Decide it with a table on the
        # bench, not from a reading of the code.
        table_latch: dict = {}
        last_skip = "no attempt made"
        for rnd in range(1, JOURNEY_LEG_PLAN_ROUNDS + 1):
            t_now = ctx.tel.fresh()
            if t_now is not None and any(t_now.get("arm_fault") or
                                         [False, False]):
                # Terminal by real_env's own contract; further rounds just
                # burn re-acquire waits with a latched arm over the table
                # (hardware 2026-08-01, sh2 recurrence).
                return refuse(
                    "ARM FAULT latched by the dropout watchdog — the latch "
                    "is terminal: inspect the harness, restart real_env, "
                    "then re-run", 6)
            plan, why = plan_with_reacquire(ctx, target_keys, table_latch,
                                            ctx.home_enc)
            if plan is None:
                if why and "lost at the scene build" in why:
                    # A target still unseen after the re-acquire wait: charge
                    # the round and look again — each round buys another
                    # CUBE_REACQUIRE_S of gaze-on-target, which is the only
                    # thing that can bring a flickering tag back.
                    last_skip = why
                    print(f"[mission] round {rnd}/{JOURNEY_LEG_PLAN_ROUNDS}: "
                          f"{why}")
                    continue
                # plan_until_gated has already retried per arm and per pairing;
                # an outer round would re-ask the identical question.
                return refuse(why, 3)
            targets = plan.targets
            if not args.execute:
                break                       # dry-run: one plan, then the watch
            outcome, why, held, ret = execute_and_retract(
                ctx, plan, target_keys, table_latch,
                empty_retry_s=JOURNEY_EMPTY_REGRASP_S)
            if outcome is FATAL:
                return refuse_holding(why, 5, pub, tel, det, rig,
                                      args.max_rate, holding=bool(held))
            if outcome is SKIP:
                last_skip = why
                print(f"[mission] round {rnd}/{JOURNEY_LEG_PLAN_ROUNDS} "
                      f"aborted: {why} — re-planning from fresh detections")
                continue
            # DONE. Mark every held cube reached and its hand held — the same
            # bookkeeping a journey leg does. Without it the gaze logic keeps
            # counting the carried cubes as PENDING, and both cameras spend
            # the entire post-grasp hold serpentine-scanning for cubes that
            # are inside the closed grippers; with it, ao_gaze's own idle
            # branch PARKS the gimbals at neutral — the power-on attitude
            # (user requirement 2026-08-01).
            sidekey = {"left": "L", "right": "R"}
            for side in held:
                rig._held[side] = {"key": plan.bundle.assignment.get(
                    sidekey[side])}
                key = plan.bundle.assignment.get(sidekey[side])
                if key in rig.cubes:
                    rig.cubes[key].held_by = side
                    rig.cubes[key].reached = True
            break                           # DONE
        else:
            return refuse(f"{JOURNEY_LEG_PLAN_ROUNDS} plan rounds exhausted — "
                          f"last: {last_skip}", 4)

        if not args.execute:
            # THE ARMS ARE HELD FOR THE WHOLE DRY RUN. Everything below this
            # point used to publish gaze alone, which does not reset the arm
            # freshness timer, so the failsafe owned the arms from 0.5 s after
            # the green path appeared until the operator pressed Ctrl+C — see
            # hold_until_enter for the hardware failure that found it. Snapshot
            # the posture the route was GATED from, before anything drifts it.
            telemetry = tel.fresh()
            hold_enc = _arm_enc(telemetry) if telemetry else None
            if hold_enc is None:
                return refuse("no arm encoders — cannot hold the arms through "
                              "the dry-run watch", 2)
            # Standing NOISE FLOOR, measured against the very target the plan
            # was built on. Pairs with the streaming sample --execute prints:
            # this is noise alone, that one is noise plus whatever the torso
            # lean adds, and the difference is what a live tracker would have to
            # correct. Costs six seconds and moves nothing.
            watches = {k: MotionWatch(f"standing still [{k}] (noise floor)", p)
                       for k, p in targets.items()}
            t_end = time.monotonic() + OBSERVE_S
            print(f"[watch] sampling {OBSERVE_S:.0f}s standing still — "
                  f"detection noise floor, arms held, no motion")
            while time.monotonic() < t_end:
                loop_t = time.monotonic()
                telemetry = tel.fresh()
                out = rig.tick(telemetry or {},
                               hold_packet(hold_enc, args.max_rate))
                if out:
                    pub.publish(out)
                det.poll(rig)
                for k, w in watches.items():
                    w.sample(telemetry or {}, rig.latest(k, loop_t),
                             loop_t, rig.last_inliers(k))
                time.sleep(max(0.0, PUB_DT - (time.monotonic() - loop_t)))
            for w in watches.values():
                w.report()
            print("\n[dry-run] all gates green — preview stays in the monitor; "
                  f"arms HELD at the gated posture at {PUB_HZ:.0f} Hz. "
                  "Ctrl+C or Enter to exit, rerun with --execute to move")
            hold_until_enter(pub, tel, det, rig, hold_enc, args.max_rate)
            return 0
        # -- hold (reached only via the DONE break above) ---------------------
        print(f"\n[done] {'grasped ' + '+'.join(held) + ' and ' if held else ''}"
              f"retracted — holding posture at {PUB_HZ:.0f} Hz, Ctrl+C to end")
        # A trail-return retract has no bundle (ret is None): hold at the
        # MEASURED posture instead — the trail ended at the round's start.
        hold_meas = None
        if ret is None:
            t_now = tel.fresh()
            hold_meas = _arm_enc(t_now) if t_now else None
            if hold_meas is None:
                return refuse("no encoders to hold after the trail return", 4)
        else:
            q_final = ret.q_enc[-1]
        while True:
            tick = time.monotonic()
            telemetry = tel.fresh()
            if telemetry and any(telemetry.get("arm_fault") or [False, False]):
                return refuse("ARM FAULT during hold", 4)
            det.poll(rig)
            if ret is not None:
                packet = {"arm_targets": {s: _slot(ret, q_final, s,
                                                   args.max_rate)
                                          for s in ret.active_sides}}
            else:
                packet = hold_packet(hold_meas, args.max_rate)
            pub.publish(rig.tick(telemetry or {}, packet))
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C — stopping publishers. The robot's silence "
              "failsafes take over: arms crawl to the chest home at 0.125 "
              "rad/s, cameras return to the origin at 0.75 rad/s. Grippers "
              "do NOT open — the EE service is a separate process and keeps "
              "whatever it was last told.")
        return 0
    finally:
        # The histogram outlives the round loop on purpose: it is the only
        # place a run's retreats are summed, and "which rule kept firing" is
        # what a session's next fix is chosen from.
        print_retreat_tally()
        try:
            route_pub.publish({"paths": {}, "cands": {}, "goal": "clear",
                               "stamp": time.time()})
            time.sleep(0.1)
        except Exception:  # noqa: BLE001 — clearing the overlay is best-effort
            pass
        tel.stop()
        det.stop()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
