"""Auto-operator: "scan → lock → walk → stop → reach → hold/track" on the real robot.

Runs on the same host as humanoid_monitor.py (detections arrive over a local IPC
socket); in the documented bring-up that host is the robot computer. Consumes
monitor detections + robot telemetry, and publishes the same command packets a
human operator would:

    9873  {"nav_cmd": [vx, vy, wz]}                     walking velocity (base FLU frame)
    9874  {"gaze_targets": [yl, pl, yr, pr],            gimbal reference (rad)
           "arm_targets": {...} | None}                 Cartesian reach targets (--use-ik)

Default mission scheduler (``--independent-arms``): left and right each own a
symmetric per-arm task ``ASSIGNED -> WAIT_STOP -> STAGING -> REACHING`` and then
hand off independently to that arm's existing hold/grasp/lift/carry state.  A
same-side cube starts its arm as soon as that arm's reachability debounce fires;
both Cartesian approaches share one packet and advance in the same tick.  The
base and the robot's globally-modal 14-joint/Cartesian arm wire are the only
shared resources: walking serves one still-distant assignment at a time, and
joint staircases remain non-preemptible and serialized.  A ready Cartesian arm
finishes before a peer starts a new staircase.

``--no-independent-arms`` restores the legacy global mission phases:
  SEARCH  scan until any pending cube has a CONFIRMED sighting; choose the
          first-seen pursuable cube (distance breaks ties).
  GO      walk with in-stride steering toward that cube; arrival is judged by
          the learned ReachabilityGate with a geometric floor.
  REACH   latch the reach target (median of recent sightings, pulled back by
          --reach-standoff) and drive the EE site there via the robot's own IK;
          the optional gripper-tag visual servo (see the SERVO_* block) trims the
          camera↔FK hand bias while reaching.
  HOLD    terminal phase: every cube is in a hand — with two cubes BOTH arms stay
          engaged, each on its own cube (legacy sequential reach, concurrent hold: a
          reached cube is never parked; its arm keeps the hold while the mission
          chases the next cube with the other arm). With --dynamic-track (default)
          each holding hand independently FOLLOWS its moving cube, retracts to
          rest while it is out of reach or unseen past a grace window, re-extends
          when it returns, and RELEASES the cube back to the queue when it
          persistently favors the other arm or crosses the body midline (hard
          same-side rule — arms never cross; one cube per half-space by placement).
  PARK    legacy --no-hold-after-reach only: stream the rest pose briefly between
          cubes, then SEARCH the next one (no hand keeps its cube).

Closed-form TRACK aim comes from sim policies.py (_compute_camera) with FK-probed
camera frames — no hardcoded axis conventions survive a model re-export.

--gaze-only: a reduced mode that ONLY tracks the pre-registered cube(s) with the head
cameras — no walking, no reaching, no 9873 traffic (publishes only gaze_targets on 9874).
Gaze runs the SAME per-camera claim/track logic as the full mission (_eye): each cube is
claimed by one camera on first sighting, each camera stares at its own claimed cube and
ignores the rest, and a claim dissolves after DET_UNCLAIM_S unseen so cameras can be
reassigned. (An earlier version pinned the rear camera at rest out of cable-ROM caution;
the ±180° panoramic scan has since been validated on hardware, so both cameras
participate.) Used to bring up / validate the gimbal on the real robot before enabling
the legs, and to test multi-cube camera locking.

Robot-side prerequisites: humanoid_real_env.py launched with --use-ik and with
--high-level-controller-ip / --arm-sender-ip pointing at THIS machine. The monitor must
run with --bodies listing the mission cubes. Nothing else may bind 9873/9874 here.
(--gaze-only needs neither --use-ik nor 9873; only --arm-sender-ip pointing here.)

Safety: outputs hard-clamped (|vx| ≤ 0.3 m/s, vy ≡ 0, |wz| ≤ 0.2 rad/s); stale or frozen
detections zero the walk command; every silence failsafe on the robot side (nav 1 s,
arm 0.5 s, gaze 2 s) stays armed because we publish continuously at OP_RATE_HZ.
A human with the gamepad always wins (stick deflection pauses nav robot-side).
"""
from __future__ import annotations

import copy
import dataclasses
import math
import sys
import time
from collections import deque
from enum import Enum

import numpy as np

from humanoid_site import LEGGED_ENV_ROOT as _LEV_ROOT  # noqa: E402
import humanoid_site as _site  # noqa: E402
# (_LEV_ROOT is put on sys.path only inside main(), for the --use-gate import —
# 08-22: it used to happen here, handing every importer of this module
# upstream's top-level `utils/`, `tasks/`, `test/`, `tools/` as bare names.)

from ipc.publisher import NNGPublisher, NNGSubscriber  # noqa: E402

from humanoid_model import MJCF_MODEL_PATH  # noqa: E402  (single source of truth)
from humanoid_monitor import GimbalCameraFK, DETECTION_IPC_URL  # noqa: E402
# S2 typed state containers (dict-style access preserved via shims — see
# auto_operator/models.py and docs/auto_operator_refactor_plan.md)
from auto_operator.models import (  # noqa: E402
    ArmTask, GripperBurst, HoldState, SecondaryReach, Sighting, TrackMotion)
from auto_operator import safety as ao_safety  # noqa: E402  (S3 predicates)
from auto_operator import arbitration as ao_arbitration  # noqa: E402  (S4)
from auto_operator import arms as ao_arms  # noqa: E402  (S5)
from auto_operator import gaze as ao_gaze  # noqa: E402  (S6)
from auto_operator import perception as ao_perception  # noqa: E402  (S6)
from auto_operator import staging as ao_staging  # noqa: E402  (S6)
from auto_operator import servo as ao_servo  # noqa: E402  (S6)
from auto_operator import secondary as ao_secondary  # noqa: E402  (S6)
from auto_operator import mission as ao_mission  # noqa: E402  (S6)
from auto_operator import independent as ao_independent  # noqa: E402

OP_RATE_HZ = 20.0          # operator tick rate; > all robot-side timeout rates
DT = 1.0 / OP_RATE_HZ

# --- eye (from policies.py DualCubeSearchGaze / _compute_camera) -------------------
SCAN_YAW_RATE = 1.0        # rad/s AVERAGE scan advance, including dwells.
                           # (History: 0.5 -> 1.0 -> 1.3 -> 1.0 -> 0.7 ->
                           # 1.0. The 0.7 dwell-stretch (2026-08-02 night,
                           # ~0.27 s stare) was tried against the 3-4-laps-
                           # to-lock symptom; the user judged on hardware
                           # 2026-08-03 that scan speed was NOT the issue
                           # and asked for 1.0 back. The remaining decode
                           # margin levers are physical: cubes nearer
                           # (r_xy 0.35-0.40) and two faces exposed —
                           # tracking watches show p50 1 face at ~0.8 m
                           # slant even from a stationary camera.)
SCAN_STEP_RAD = 0.35       # step-and-stare quantum: the scan bearing is
                           # quantized to 20-deg steps; the 1.5 rad/s gaze
                           # slew sprints between quanta (~0.23 s) and the
                           # camera then HOLDS the remainder of the quantum
                           # — the sharp frames are what actually decode a
                           # tag and stop the sweep on a visible cube
# SINGLE-ROW scan (user 2026-08-02: "no serpentine — just look down 47°, same
# rotation range"). The pitch never changes; only yaw sweeps. 47° down is where
# floor/table cubes at ~0.4-0.6 m actually sit from the ~0.66 m camera, and the
# retired 30/45/60/15° serpentine spent 3/4 of its cycle staring somewhere
# else. The yaw trajectory is UNCHANGED from the serpentine (half-turn lead-in,
# then full circles alternating direction so each turn unwinds the previous
# one — the joint bounces within ±(180°+|datum|), no cable wind-up).
SCAN_PITCH_DOWN = math.radians(47)
SCAN_SETTLE_S = 0.5        # pitch down FIRST, hold this long, THEN start the
                           # yaw sweep (user 2026-08-04: starting the rotation
                           # while still pitching swept the start sector past
                           # the camera before it could decode anything —
                           # 47 deg down at the 1.5 rad/s slew takes ~0.55 s,
                           # so the dwell covers the pitch travel)
TRIM_GAIN = 0.05           # /step integral trim on (commanded - actual) gimbal joints.
                           # 07-17: 0.3 (~6/s at 20 Hz) was a hot integrator around a
                           # plant that gained ~100 ms of lag (gaze low-pass) + 5%-torque
                           # sluggishness — marginal, and tick jitter (see
                           # GATE_EVAL_PERIOD_S) excited it into the violent periodic
                           # pre-reach camera shudder. 0.05 still cancels a 0.1 rad
                           # steady bias in ~1 s.
TRIM_MAX = 0.35            # rad anti-windup clamp
CAM_PORTS = (5555, 5556)   # left, right (default humanoid_monitor --tag-mounts mapping)
CAM_SITES = {5555: "cam_left_rgb", 5556: "cam_right_rgb"}
GAZE_ORDER = {5555: (0, 1), 5556: (2, 3)}  # port → (yaw, pitch) slots in gaze_targets
GAZE_SLEW_RATE = 1.5       # rad/s cap on the PUBLISHED gaze reference. The robot-side slew
                           # was removed in 8045371 (cam target = gaze_reference + residual,
                           # applied raw), so the SENDER must bound the step: an acquisition
                           # jump (scan→lock) or a ±2π branch change would otherwise reach
                           # the gimbal motors in one control step. 1.5 rad/s mirrors the sim
                           # slew cap and the old robot-side limit.

# --- legs (from moving_policy.py HeuristicMovingPolicy) ----------------------------
CRUISE_VX = 0.4            # m/s, the walk magnitude. 08-08: 0.3 -> 0.4 on the
                           # trainer's word that 0.4 is the policy's minimum
                           # usable walking speed. RAISED TOGETHER WITH THE
                           # FLOORS BELOW, and that coupling is not optional:
                           # the schedule is clip(K*(dist-STOP), NEAR_VX,
                           # CRUISE_VX), and numpy applies the floor first then
                           # the ceiling — so with NEAR_VX 0.4 against a 0.3
                           # ceiling every command would silently come out 0.3
                           # and the change would do nothing at all. Well inside
                           # the trained band (lin_vel_x = +/-1.0).
K_STEER = 1.0              # wz = clamp(K_STEER * bearing_err, ±WZ_MAX).
                           # 08-10 (user): 0.6 -> 1.0 = the sim's RUNNING
                           # value (moving_policy.py K_STEER=1.0; our old 0.6
                           # matched MoverCfg's DEAD default, a porting trap
                           # plus the pre-arc operator gain). With the 0.7
                           # cap this commands yaw into the >0.5-0.6 range
                           # the 08-08 bench called degraded — if the walk
                           # snakes, drop the cap or this gain first.
WZ_MAX = 0.2               # rad/s (the FACED in-stride trim cap; the unfaced
                           # arc uses the floor/cap pair below)

# ---- ARC-INTO-FACING (sim parity, 2026-08-08). The simulator's v2 mover
# (MoverCfg.turn_cruise, measured 08-02 on the sim loco checkpoint) never
# turns in place: it enters WALK on tick 1 and HOLDS A YAW FLOOR while the
# heading error is outside the face tolerance, arcing into the facing at
# cruise — because yaw tracking is a function of FORWARD SPEED (ang_err
# 0.299 rad/s at vx 0/wz 0.6 vs 0.161 at vx 0.4/wz 0.6), and in-place the
# policy just twists on planted feet (authority ratio 0.24 at 87 deg — the
# exact torso-windup our 08-08 --wz bench measured on v159b). Same rule
# upstream gave for hardware: curved walking only. Journey-gated: the standing
# mission and the non-journey operator walk are untouched.
# Hardware numbers are DELIBERATELY below the sim's floor 0.6 / cap 1.0:
# v159b's trained wz range is +/-0.7 (VEL_SCALE_ANGULAR) but nothing above
# a commanded 0.2-while-walking has ever run on hardware — bench the arc
# first (humanoid_nav_step_test --vx 0.4 --wz 0.35) and retune from what it
# measures. At the floor the arc radius is 0.4/0.35 ~= 1.1 m (was >= 2 m
# under the old 0.2 cap).
JOURNEY_FACE_TOL_RAD = 0.14  # ~8 deg, the sim's FACE_TOL_RAD: outside this
                           # the yaw floor holds; inside, the trim cap rules
JOURNEY_WZ_UNFACED_FLOOR = 0.35  # rad/s: never command an unfaced arc below
                           # this — sub-floor yaw is the measured dead band
JOURNEY_WZ_UNFACED_MAX = 0.7     # rad/s: unfaced arc ceiling. 08-10 (user):
                           # 0.5 -> 0.7 = the policy's full trained range
                           # (VEL_SCALE_ANGULAR); sim caps 1.0. WATCH the
                           # first big correction: the 08-08 bench saw yaw
                           # tracking degrade above ~0.5-0.6 while walking —
                           # if the walk snakes or feet drag at a hard turn,
                           # come back to 0.5.
JOURNEY_WZ_SLEW = 1.0      # rad/s^2: cap on the PUBLISHED wz's rate of change
                           # — the second half of the sim port, missed on the
                           # first pass. The sim's mover feeds every channel
                           # through an accel limiter (MoverCfg MAX_ACCEL,
                           # "~0.1 s ramps, never steps"); the first hardware
                           # arc run went out UNRAMPED and the yaw floor
                           # turned the face boundary into bang-bang: leg 2
                           # (reverse) logged wz -0.35 -> +0.35 -> -0.50, a
                           # growing slalom the operator read as "went the
                           # wrong direction". At 1.0 rad/s^2 a full flip
                           # takes 0.7 s and passes through the trim band
                           # instead of slamming across it; the initial
                           # 0 -> 0.35 arc entry still takes only 0.35 s.
                           # vx stays deliberately UNRAMPED — the 0 -> 0.4
                           # step is hardware-validated and unchanged.

# ---- JOURNEY mission (07-21, user spec): survey BOTH cubes first, walk to the
#      NEARER one (forward or backward — the robot never turns around, its
#      symmetric structure makes both faces a "front"), grasp, walk the other
#      way to the second, grasp, stand. All constants journey-gated: the
#      standing mission is untouched. ----
JOURNEY_NEAR_VX = 0.4      # m/s: slowest commanded FORWARD walk. 08-08: 0.2 ->
                           # 0.4, on the trainer's word that 0.4 is the policy's
                           # minimum usable walking speed. EQUAL TO CRUISE_VX, so
                           # clip(K*(dist-STOP), NEAR_VX, CRUISE_VX) is constant
                           # and a journey approach has no taper at all — that is
                           # the intended shape, not an accident.
                           # WHAT THE HARDWARE SAW ON 08-07, at the old 0.2:
                           #   0.20  leans forward, NEVER lifts a foot. Gyro p50
                           #         0.005 rad/s = a standing robot.
                           #   0.25  DEGENERATE. Tiny steps, |wz| peaks 1.67
                           #         rad/s (96 deg/s), operator: "almost fall
                           #         down ... each step is soooooo small".
                           #   0.30  clean gait, |wz| p50 0.176 max 0.692 — a
                           #         LOWER peak than 0.25, which is the tell.
                           # So the band below the usable range is UNSTABLE, not
                           # merely slow, and 0.4 clears it with margin. The
                           # 08-07 2/2 journey ran at 0.2 and survived only
                           # because its legs were long enough that the taper
                           # reached 0.21 with the robot ALREADY walking — the
                           # band bites at gait ONSET, not once moving.
JOURNEY_NEAR_VX_BACK = 0.4  # 08-08: raised with the forward floor. MEASURED
                           # 08-07 that reverse walks cleanly at 0.20 (|wz| max
                           # 0.469, quiet 0.61 s after the stop) — so this is
                           # the one number here that is HIGHER than hardware
                           # requires, taken on the trainer's word rather than
                           # the measurement. Kept a separate constant so the
                           # measured asymmetry can be restored without touching
                           # the forward floor: forward is the hard direction
                           # (the arm home sits 29 mm aft of the training
                           # posture), which INVERTS the 07-26 audit's belief
                           # that reverse needed more.
JOURNEY_SLOW_K = 0.5       # m/s per m: vx = clip(K*(dist-STOP), NEAR_VX, CRUISE_VX)
ASSIGN_STALL_S = 20.0      # 07-26 review: standing missions only — an ASSIGNED
                           # task with zero reachability progress this long
                           # releases its claim so the peer arm (or a fresh
                           # median) can take the cube. Journey is exempt:
                           # its ASSIGNED phase legitimately lasts a whole
                           # walk leg.
LATCH_FAIL_SKIP_N = 3      # 07-26 review: consecutive latch failures on the
                           # same cube before it is marked unserviceable and
                           # skipped LOUDLY — the cancel->reassign->latch-fail
                           # cycle otherwise judders the arms side<->tuck
                           # forever next to an unlatchable cube.
JOURNEY_POSTURE_CORRIDOR_RAD = 0.35  # 07-26 review: the joint tuck/untuck glide
                           # may only drive an arm whose measured joints lie
                           # near the audited SIDE<->POWERON segment (the
                           # receiver lerps along it, so mid-glide residual is
                           # ~0). An arm freed anywhere else (e.g. extended
                           # after a released hold) is left to the Cartesian
                           # pin — joint-lerping it from an unaudited pose
                           # sweeps the gripper through unknown space.
                           # 07-31: segment endpoint moved (POWERON = sim
                           # home) and the audit was RE-RUN, not carried over:
                           # min clearance 39.5 mm (endpoint's own static
                           # pair), arm-arm >=229 mm — see JOURNEY_POSTURE_TOL_RAD.
JOURNEY_AIM_OFFSET_M = 0.05 # 08-13 (user): 0 -> 0.05. A small lateral
                           # parking offset: 5 cm beside the shoulder buys arm
                           # room without recreating the 08-09 table-strike
                           # geometry (that lesson was at 0.10+, when the WALK
                           # was doing the collision avoidance). With the aim
                           # this close to the cube centre, table clearance
                           # comes from PLACEMENT and from the tip guard /
                           # C_clearance gates -- not from this dial.
                           # Full value ledger (08-08..08-13, with the
                           # measurements): docs/design-notes/journey-walk-and-aim.md
JOURNEY_AIM_OFFSET_BACK_M = 0.0 # 08-10 (user): 0.35 -> 0. Reverse legs walk
                           # STRAIGHT at the cube. RISK, stated: a dead-midline
                           # cube sits at the envelope's hardest corner (the
                           # 0.10 and 0.13 runs both died there); the settled-arm
                           # re-decision and CROSS_MIDLINE_REACH_M own the arm
                           # choice in that region.
                           # Full value ledger (08-08..08-13, with the
                           # measurements): docs/design-notes/journey-walk-and-aim.md
JOURNEY_REVERSE_DRIFT_COMP_M = 0.0 # 08-13 night (user): 0.05 -> 0.
                           # The right-bias is CANCELLED: both 08-13 reverse
                           # legs showed no drift, and the bias itself pushed a
                           # left-side cube onto the midline. Re-introduce only
                           # if the leftward settle drift returns (misses of
                           # +0.15..0.21 to the LEFT on full-length reverse
                           # legs); the ledger has the measurements.
                           # Full value ledger (08-08..08-13, with the
                           # measurements): docs/design-notes/journey-walk-and-aim.md
JOURNEY_STOP_M = 0.15      # distance at which the schedule bottoms out (arrival
                           # itself is still judged by the reachability gate)
# In-place rotation is FORBIDDEN on this robot (upstream ruling, 2026-08-08):
# curved walking only. The --wz bench that day measured the reason:
# from standstill the checkpoint winds its torso at 0.2 rad/s (net
# ~1 deg, feet planted) and near-falls with dragging feet at 0.4.
# The JOURNEY_ALIGN_* phase built on the decouple idea was removed
# the same night; humanoid_nav_step_test --wz keeps the harness.
JOURNEY_ARRIVE_M_BACK = 0.50 # 08-13 (user, 7th pass): 0.53 -> 0.50.
                           # Stop closer on reverse legs so the rear-reach
                           # envelope bends lower -- a smaller radius is the one
                           # stance-side lever against a low rear cube. Reverse
                           # coast is LONGER than forward (0.18-0.25 band):
                           # watch for wind-up-basin landings (<0.30).
                           # Full value ledger (08-08..08-13, with the
                           # measurements): docs/design-notes/journey-walk-and-aim.md
JOURNEY_ARRIVE_M = 0.63 # 08-13 (user, 7th pass): 0.58 -> 0.63.
                           # Stop coast + settle refinement eats ~0.20 m off the
                           # live-distance cutoff, so 0.63 aims the landing at
                           # ~0.45 -- the middle of the proven band. The day's
                           # ledger split cleanly on landing radius: 0.40-0.47
                           # carried, 0.36-0.41 failed.
                           # Full value ledger (08-08..08-13, with the
                           # measurements): docs/design-notes/journey-walk-and-aim.md
JOURNEY_CUBE_Z_MIN = -0.06 # m, base frame. 08-13 (user): -0.01 -> -0.06.
                           # The envelope floor is RADIUS-dependent, so a flat
                           # fail-fast bar over-generalized: both low-pole
                           # failures were large-radius REAR reaches, while
                           # z=-0.002 at r=0.42 grasped clean.
                           # Full value ledger (08-08..08-13, with the
                           # measurements): docs/design-notes/journey-walk-and-aim.md
JOURNEY_CUBE_Z_MAX = 0.055  # the 08-08/09 hardware ledger. Every grasp that
                           # SUCCEEDED read z in [-0.00, +0.040]; the two
                           # failure poles sit just outside: z=+0.07 (front,
                           # close+high -> wrist folded against the chest,
                           # 20/20 clearance refusals within 5 mm of the
                           # floor) and z=-0.021 (rear, low -> rear-envelope
                           # INFEASIBLE 20/20). A leg whose settled fix reads
                           # outside this band SKIPs in one line ("raise/
                           # lower the cube") instead of burning ~90 s of
                           # solver rounds that the geometry has already
                           # decided. ONE-SHOT doctrine (user 2026-08-09):
                           # the robot walks once and grasps from where it
                           # stopped — bad ammunition gets named, not fought.
JOURNEY_Z_LEDGER_FLOOR_MM = 25.0
                           # The HIGH pole above is a fact about THIS floor:
                           # the 20/20 refusals at z=+0.07 all missed the
                           # 25 mm clearance bar by <= 5 mm (routes existed at
                           # 20-25 mm from the chest). Run with a clearance
                           # floor BELOW this and the high-side verdict is no
                           # longer known-lost, so journey_z_verdict steps
                           # aside and lets the solver judge (user 08-09:
                           # "accept the higher cube"). The LOW pole is
                           # cuRobo's own envelope INFEASIBLE — floor-blind,
                           # it always binds.
JOURNEY_STILL_VEL = 0.05   # m/s: measured |base_lin_vel| xy below this = still
JOURNEY_STILL_ANG = 0.10   # rad/s: measured |base_ang_vel| z below this = still
JOURNEY_STILL_S = 1.5      # s: stillness must SUSTAIN this long before latch —
                           # the ramp-zero moment still carries inertia/sway, and
                           # the latch median must sample a quiet robot.
                           # 08-12 (user): 1.0 -> 1.5 — buy the settled fix a
                           # quieter robot before the re-locate.
JOURNEY_STILL_DEBIT_S = 0.2  # 08-07 audit: what ONE non-still sample costs the
                           # sustain window, instead of nulling it. base_ang_vel
                           # is raw and unfiltered, the bar is 0.10 rad/s, and
                           # the window needs ~50 consecutive good ticks (125 at
                           # the blind 2.5 s) — so a single spike used to restart
                           # the whole thing, and expiry is the leg's only
                           # measurement-SKIP. Debiting keeps the gate honest
                           # (real motion produces many bad ticks and still
                           # drives the credit to zero within a few) while a lone
                           # outlier costs 0.2 s rather than everything.
JOURNEY_WALK_EE = [0.1061, 0.2476, 0.2588]  # the WALKING tuck (left; right
                           # mirrors y): free arms tuck HERE while walking.
                           # Held arms stay at REST with their cube.
                           #
                           # 07-31: RE-UNIFIED with FK(POWERON_JOINTS) — the
                           # sim home. The 07-30 split (power-on raised, tuck
                           # stays at the policy's low training default) died
                           # with the user's one-home spec, and it could not
                           # survive it partially: journey_arms_tucked's joint
                           # branch checks POWERON_JOINTS while its EE fallback
                           # and the arbitration glide checked THIS — a 0.28 m
                           # disagreement in which the tuck gate either never
                           # passes or passes at the wrong pose. One home means
                           # one tuck, so the pre-07-30 identity is restored
                           # (and guarded again in test_model_derived_constants).
                           #
                           # COST, MEASURED not guessed (07-31 review caught the
                           # first draft of this note claiming "lower yaw
                           # inertia" — the opposite of what the model says).
                           # Full-body Izz about the vertical through the CoM,
                           # legs at the deployed default_pose:
                           #   low tuck (retired)  0.7636 kg m^2
                           #   07-30 chest         0.7943   (+4.0%)
                           #   sim home 07-31      0.8077   (+5.8%)
                           # The HANDS come in (EE r_xy 0.383 -> 0.269) but the
                           # heavy links go OUT: wrist_1_L (0.84 kg) r_xy
                           # 0.213 -> 0.266, elbow_L (0.65 kg) 0.216 -> 0.279,
                           # and those dominate. CoM moves +24.4 mm UP and
                           # 29.0 mm AFT (x +19.1 -> -9.9 mm), i.e. from 58.1%
                           # to 45.8% of the 235 mm foot support measured from
                           # the heel — still inside the polygon, but it is a
                           # permanent ankle-torque bias handed to a locomotion
                           # policy trained at the posture being replaced.
                           # Walking was never hardware-validated in either
                           # configuration; the 07-28 sim dynamics run
                           # validated the policy STANDING here. First hardware
                           # walk decides this.
CHEST_HOME_EE = {"left": [+0.1061, +0.2476, +0.2588],
                 "right": [+0.1063, -0.2457, +0.2588]}
# ^ FK of CHEST_HOME_JOINTS on the deploy model — DERIVED THE OTHER WAY ROUND
#   since 07-31. Until then the EE target came first (walking tuck + a named
#   0.10 m rise) and the joints were solved from it; now the JOINTS come first,
#   because the user's requirement is joint-space identity with the simulator:
#   cuRobo's HUMANOID_ARM_JOINT_HOME (lev2_master ik_curobo_robot_cfg.py) IS
#   the home, and an EE-first derivation could only ever approximate it through
#   IK. The old CHEST_HOME_RISE_M derivation died with that inversion — the sim
#   home is not "the walking tuck raised", it is its own posture: hands pulled
#   in to r_xy 0.27 m (was 0.38) and up to z 0.26 m (was 0.18), elbows 111 deg
#   (was 100). guard: tests/test_model_derived_constants recomputes this FK.
#   Installed as op.rest_ee, so ALL THREE homing paths land here — the idle
#   keep-alive glide, a held arm's dyn-retract, and the carry-home after a
#   grasp. Patching only the idle glide was the 07-24 bug.
#   THE SIDES ARE NOT MIRROR IMAGES (y: +0.2476 vs -0.2457, x differs 0.2 mm):
#   the joints ARE plain mirrors, but the deploy model's two arms are not
#   perfectly symmetric, so the honest FK values differ by ~2 mm. Stored
#   per-side rather than symmetrised — these are recognition/convergence
#   targets, and a 2 mm fudge would eat 2 mm of everyone's tolerance.
#   Orientation is CHEST_HOME_QUAT below, NOT JOURNEY_WALK_QUAT: the two homes
#   (chest vs walking tuck) had the same wrist orientation only while the chest
#   home was derived FROM the tuck, and the sim home broke that coincidence.
CHEST_HOME_JOINTS = [-1.3740, -0.1870, +0.1310, +1.9290, -1.2403, -1.4546, -0.0090]
# ^ THE SIMULATOR'S HOME, verbatim through the units seam — 07-31 (user):
#   "cuRobo's start posture, the power-on position and home must all be the
#   same as the home pose in the simulator" (translated from the operator's
#   Chinese; that sentence is the spec this constant exists to satisfy).
#   Source: _REACH_READY_ARM_JOINT_POS (= HUMANOID_ARM_JOINT_HOME) in
#   lev2_master/mj_envs/tasks/visual_manipulation/curobo/ik_curobo_robot_cfg.py,
#   verified byte-identical on the machine that runs the sim
#   and the local legged_env_v2 clone on 07-31. Conversion is the client's own
#   fold: enc = SIGN*cspace + qpos0, which moves wrist_1 (qpos0 -pi/2:
#   0.3305 -> -1.2403) and folds wrist_2's sign (cspace +1.4546 -> enc
#   -1.4546); everything else passes through. mirror_arm reproduces the right
#   arm exactly (max err 0.0), because the wrist_2 axis flip and the enc-space
#   negation compose back to the sim's own left=+right exception.
#
#   WHY JOINT-SPACE IDENTITY WITH THE SIM IS THE SPEC (07-31): with the same
#   home on both sides of the bridge, the three homes that spent July drifting
#   apart — sim/planner home, robot power-on, chest home — collapse into ONE.
#   The server's planning_home equals this by construction (same source
#   constant), so the home_q patch (fcb7d8b) stops being a correction and
#   becomes a confirmation, plan seeds match the sim's, and the 07-28 dynamics
#   validation (8/10 grasps, RL policy standing AT this posture) transfers.
#
#   Same posture as POWERON_JOINTS, spelled out rather than aliased:
#   POWERON_JOINTS is "what the deployed env_config says", this is "the home
#   the arms work from". A guard test asserts the equality.
#
#   ⚠ MEASURED HARDWARE MARGIN, 07-31 offline (Monte-Carlo, uniform +/-sag on
#   all 14 joints, worst case): static 39.5 mm (R gripper base ~ torso),
#   +/-3 deg -> 21.5 mm, +/-5 deg -> ~10 mm, +/-8 deg -> PENETRATION. The
#   07-30 chest home measured 51 mm and was sag-INSENSITIVE (its floor was a
#   structural pair); this fold's floor is posture-sensitive. Extended-arm
#   sag was measured at +/-14 deg (torque 0.8) but a folded arm carries far
#   less gravity torque, and 07-30 hardware DID park the left arm at exactly
#   this posture (the server's planning_home, pre-fcb7d8b) without contact.
#   Still: the margin is real and thin — first runs, watch the RIGHT gripper
#   base against the torso; if hardware sag at the fold exceeds ~5 deg,
#   re-solving the sim home a few cm wider (upstream, so identity holds) is
#   the fix, not padding constants here.
#
#   ONE THING IT MEASURABLY IMPROVES, checked 07-31 on the model: SIGHTLINES.
#   Ray-testing both head cameras against cubes at r_xy 0.45 on either side,
#   this fold leaves all four lines CLEAR, while the 07-30 chest home put the
#   gripper racks 38-66 mm off-axis at 0.56-0.65 m — i.e. squarely in the
#   camera's path to the cube. Tag detection is the mission's first gate, and
#   the arms parking out of their own cameras' way is worth margin.
#
#   --chest-home is consequently a NO-OP on a freshly powered robot (the ramp
#   measures zero distance and returns at once). It is deliberately kept: it
#   still does the right thing after a session has left the arms elsewhere, and
#   it is what puts them back before a plan is solved.
CHEST_HOME_QUAT = [0.9721, 0.0084, -0.2276, 0.0563]
# ^ FK ORIENTATION of CHEST_HOME_JOINTS (left; right negates x,z — verified
#   against the right site's own FK). Exists because the chest home stopped
#   sharing the walking tuck's orientation (see CHEST_HOME_EE): publishing
#   JOURNEY_WALK_QUAT at the new home would have the IK pull toward a wrist
#   orientation 19.3 deg off the posture the ramp is trying to reach — the
#   same class of bug as the 07-21 NEUTRAL-quat finding, one constant later.
JOURNEY_TUCK_TOL_M = 0.08  # arms must be tucked within this before walking starts
JOURNEY_WALK_QUAT = list(CHEST_HOME_QUAT)  # FK ORIENTATION of the walking tuck
                           # (left; right negates x,z). Publishing NEUTRAL at
                           # the tuck point made the IK twist the wrists 17.4
                           # deg off the pose (07-21 review). Since 07-31 the
                           # tuck IS the power-on/chest/sim home, so this is an
                           # ALIAS of CHEST_HOME_QUAT rather than a second
                           # literal — one orientation, impossible to drift.
                           # (Old value, walking tuck pre-unification:
                           # [0.9885, 0.0949, -0.1138, -0.0309], 19.3 deg away.)
JOURNEY_STILL_S_BLIND = 1.5  # settle when base_lin_vel is hard zeros (v159b has
                           # NO velocity estimator): gyro + a fixed wait.
                           # 08-12 (user): 1.0 -> 1.5, matching
                           # JOURNEY_STILL_S — the same wait whether the
                           # stillness is measured or blind.
                           # 08-10 (user, latency plan item 3): 2.5 -> 1.0 —
                           # the sim's entire stop window is ~0.3 s, and the
                           # gyro gate (JOURNEY_STILL_ANG) still has to agree
                           # for the WHOLE second. WATCH: if settled fixes get
                           # noisier or landings read badly, raise this first.
JOURNEY_TUCK_HOLD_S = 5.0  # walk-tuck hysteresis: hold the tuck this long after
                           # the served task vanishes (detection flicker at range
                           # cancels/reassigns on a 3 s cycle — without this the
                           # arms oscillate side-home<->tuck forever)
JOURNEY_POSTURE_TOL_RAD = 0.06  # 07-21 (user): side<->tuck transitions are now
                           # JOINT-SPACE glides (per-arm joint_pos streams of
                           # POWERON/SIDE_HOME postures at STAGE_RATE) — the old
                           # Cartesian home_glide contorted the arm (IK landed
                           # up to 62 deg off the real power-on pose). Stream
                           # until every joint is within this of the target.
                           # Path RE-AUDITED 07-31 for the sim-home endpoint
                           # (the old 51 mm / >=80 mm numbers died with the old
                           # segment): side<->POWERON interpolation min
                           # clearance 39.5 mm, bound by the ENDPOINT's own
                           # static pair (R gripper base ~ torso — the parked
                           # sim home is itself the tightest frame; the moving
                           # path never dips below it). Arm-arm >=229.0 mm
                           # simultaneous mirrored (07-31 review CORRECTION:
                           # the first draft said 240 mm, which is
                           # shoulder_2_L~shoulder_2_R — a STRUCTURAL constant,
                           # 240.0 mm in every posture. The arm-vs-arm scan had
                           # not been given preflight's hops>2 filter, so the
                           # mounts masked the real pair. True per-segment
                           # minima: side 275.5, legacy-0730 262.5, front
                           # 229.0, v83 229.4 mm — the last two are
                           # L_left_rack4~R_right_rack4, the grippers. This is
                           # the exact masking SIDE_HOME_EE warns about 80
                           # lines up.) Same floors measured for
                           # front<->POWERON, old-chest-home->POWERON,
                           # old-low-tuck->POWERON (the prepare ramp) and
                           # v83->POWERON — every legacy start glides in at
                           # >=39.5 mm.
ACCEL_LIMIT = 3.0          # m/s² per-channel ramp

# --- stop / hand -------------------------------------------------------------------
GATE_CONSECUTIVE = 5       # ReachabilityGate.CONSECUTIVE_HITS equivalent at operator rate
GATE_EVAL_PERIOD_S = 0.1   # learned-gate inference throttle: gate.score() is CPU torch
                           # at 17-37 ms/call (measured 07-17). Unthrottled, SEARCH
                           # scored every pending cube EVERY 20 Hz tick (2-3 calls =
                           # 35-75 ms) and blew the 50 ms budget — the resulting gaze
                           # publish jitter excited the gimbal trim loop into the
                           # pre-reach camera shudder (calm again once REACH started
                           # and the gate calls stopped). Debounce counters step only
                           # on FRESH samples, so GATE_CONSECUTIVE now spans ~0.5 s
                           # wall-clock instead of 0.25 s.
GEOMETRIC_FLOOR_M = 0.28   # unconditional stop distance (sim constant)
REACH_STANDOFF_M = 0.05    # pull the reach point this far back toward the robot (real cube is solid)
AIM_UP_M = 0.015           # lift the reach point this far ABOVE the cube centre.
                           # 07-17: briefly 0.03 (the jaw landed low and bit the cube's
                           # stand); restored to 0 after the transit stations were
                           # raised. 07-18: 0.02 (user) — jaw bit low on hardware then.
                           # 07-20 afternoon: -0.01 ran 3 cm low, grabbed air; brief
                           # return to 0.02; then 0.0 (aim at the centre).
                           # 07-21 night: 0.01 -> 0.02 -> 0.015 (user) — split the
                           # difference. Override per-run via --aim-up.
REACH_OK_M = 0.05          # success: EE within this of the reach point (sim constant)
REACH_TIMEOUT_S = 8.0      # sim used 5 s; real IK EE speed is capped at 0.25 m/s — allow more
PARK_HOLD_S = 3.0          # stream the rest pose this long before switching cube
DET_STALE_S = 0.5          # detection packet older than this → treat as blind, stop walking
DET_LOST_RESCAN_S = 3.0    # target unseen this long during GO → back to SEARCH
GAZE_STARE_GRACE_S = 8.0   # 08-10 (user): "fix on the cube as long as you can;
                           # if you lose track, just scan around again." An eye
                           # bound to a PENDING cube keeps aiming at the last
                           # known position for this long after the sighting
                           # goes stale — long enough to sit still through the
                           # hand-occlusion window of a grasp (blind close
                           # triggers at 1.8 s, closes ~3 s, verdict ~3 s) —
                           # then REJOINS the serpentine to re-find it. This
                           # supersedes the 08-04 "locked = pointed,
                           # permanently" rule, whose stale stare was built
                           # for standing missions; on a walking journey leg
                           # the stored body-frame point rots and a frozen
                           # stare can watch empty space forever (seen live
                           # 08-10, camera 5556 vs cube f).
DET_UNCLAIM_S = 6.0        # cube unseen (by ANY camera) this long → its camera claim
                           # dissolves, so the next sighting may bind a different camera.
                           # Deliberately > DET_LOST_RESCAN_S: the claiming camera first
                           # gets a rescan window to re-find its own cube before the
                           # binding is up for grabs. Any-camera freshness is the right
                           # clock: while the OTHER camera still sees the cube, the pos
                           # stays live and the claiming camera aims at it (spotter
                           # handoff) — releasing the claim then would just re-assign it
                           # back to the only free camera.
NEUTRAL_QUAT = ao_arbitration.NEUTRAL_QUAT   # moved in S4; kept for compat
REST_EE = {"left": [0.3258, 0.1791, 0.2226], "right": [0.3247, -0.1776, 0.2231]}
# ^ 07-28 RE-DERIVED for the new wrist. These are FK of STAGE_JOINTS['front'] on
#   the deploy model — the JOINT posture is unchanged and still clears by
#   149.8 mm, but the redesigned wrist puts its EE 84 mm away from the old
#   values (which the pre-swap model reproduces exactly, confirming provenance).
#   Recomputed offline rather than re-measured on hardware: the model was
#   validated against measured gravity torque on both arms first (0.22 / 0.18
#   N.m), so its FK is the same authority the original values came from.
# ^ RAISED 07-17 to the new (higher) front station EE (mirror-FK'd on the deploy
#   model). NOT the power-on default EE (+0.1001): at startup the idle-arm REST
#   logic glides the arms up the 8 cm from the power-on pose to this rest.
SIDE_HOME_EE = {"left": [0.0, 0.55, 0.10], "right": [0.0, -0.55, 0.10]}
# ^ 07-24 (user): z 0.0 -> 0.10, y unchanged. r goes 0.550 -> 0.5590 (+9 mm), a
#   1.6% step off the mink-validated radius and still 49 mm short of the r=0.608
#   that solved on three runs and FAILED on the fourth into a startup
#   self-collision the same day. Measured with the robot's OWN mink solver
#   (400 steps @20 Hz from the power-on pose, NEUTRAL_QUAT, base upright):
#   0.8 mm residual — the cleanest of every home tried that day.
#
# DO NOT MOVE THIS HOME TO THE FRONT. Tried 07-24 as [0.3207, ±0.2512, 0.18]
# (= REST_EE = FK of STAGE_JOINTS['front']) on the theory that the classic front
# chest home carried its own validation pedigree. It does not, and the right arm
# collided on the first run. Measured afterwards, right-arm-to-body clearance
# along the mission-start glide, EXCLUDING the structurally-adjacent pairs whose
# distance is constant (shoulder_2_R<->base_link sits at 10.1 mm in every posture
# and masks the real minimum):
#     front home  ->  29.7 mm  (shoulder_3_R <-> base_link, tightest at step 0)
#     side  home  ->  47.5 mm  (same pair)
#   Both are positive, so the MODEL predicts no penetration — what closes those
#   30 mm is everything the model lacks: gravity sag (+-14 deg measured on held
#   poses at torque 0.8), encoder zero offset, harness, and on that day a
#   right_shoulder_2 already reporting STALL SIGNATURE. 47 mm survives those;
#   30 mm does not. The model is exactly mirror-symmetric (joint limits mirror,
#   FK mirrors, both arms 15.0 deg off NEUTRAL at the front station), so the
#   left/right asymmetry is entirely physical — which is the point: a margin
#   thin enough for hardware asymmetry to decide the outcome is too thin.
#
# 07-31, home moved to the sim home: the mission-start glide is now
# sim-home -> side-home, a different sweep than the pedigree above measured.
# Re-audited offline on the deploy model (mirrored simultaneous, 241 frames):
# min clearance 39.5 mm — bound by the sim home's own static gripper~torso
# pair, i.e. the path never dips below its start posture — arm-arm >=229 mm.
# See JOURNEY_POSTURE_TOL_RAD for the full 07-31 audit table.
SIDE_HOME_JOINTS = [-0.008, -0.482, -0.006, 0.005, -0.008, -1.080, 0.010]
# ^ FK of the POWER-ON pose (env_config default_pose) on the deploy model: the
#   idle/home hand position IS the power-on / scanning posture (user directive
#   07-14). The old debug-GUI rest [0.15, ±0.20, 0.05] sat 18 cm away and yanked
#   every idle arm into a "horse stance" elbow flare at each joint→Cartesian
#   switch. NEUTRAL_QUAT matches this pose's true orientation (FK quat=identity),
#   so the IK reproduces the power-on joints almost exactly.

# --- dynamic tracking in HOLD (follow a moving cube) --------------------------------
DYN_LOST_GRACE_S = 2.5     # 07-21 (user): 1.0 -> 2.5 — the 1 s window made every
                           # brief occlusion (walking sway, hand entering frame)
                           # yank a following arm back to home; 2.5 s still
                           # retracts on a genuinely removed cube.
GRASP_NEAR_M = 0.10        # 07-21 (user): GRASP-WINDOW IMMUNITY radius — with the
                           # measured EE within this of the target the gripper
                           # OCCLUDES the tag by construction (standoff 0), so:
GRASP_NEAR_LOST_S = 4.0    # ...the lost-grace stretches to this, and a gate
                           # flicker alone cannot retract (hard geometry checks
                           # stay live). Kills the halfway-retract limit cycle.   # cube unseen this long → slowly retract to rest (robot-side
                         # EE slew 0.25 m/s IS the "slowly"); reappearing resumes reach
DYN_OUTREACH_M = 0.60    # FALLBACK ONLY (--no-use-gate): geometric out-of-reach radius
                         # (hysteresis above the 0.28 floor). With the gate active, the
                         # LEARNED reachability model makes this call instead.
                         # 07-24 (user): 0.32 -> 0.45 -> 0.60, TRACKING TARGET_MAX_RADIUS_M
                         # so the EXTEND trigger equals the envelope the arm is allowed
                         # to reach in. Keep the two equal. Measured that day (at the
                         # 0.45 cap of the moment), the learned gate is not
                         # uniformly conservative but DIRECTIONAL (left arm, z=-0.05,
                         # 7-sample pass rate): straight ahead it collapses to 0/7 the
                         # moment r_xy passes 0.30, at 30 deg it holds to 0.38, at 45 deg
                         # to 0.42, and at 60-90 deg it passes everywhere up to the 0.45
                         # cap. So a front cube had to be dragged 15 cm closer than the
                         # arm can actually work — the trigger/capability mismatch the
                         # user hit. --no-use-gate + this radius trades the learned
                         # model's best-arm judgement for ONE isotropic number.
                         # NOTE the rim IS full stretch: 400k-FK measurement (see
                         # TARGET_MAX_RADIUS_M) puts the CLEAN reach — EE within 5 deg of
                         # NEUTRAL_QUAT, which is what the runtime IK is given — at
                         # ~0.51 m, so 0.60 is only reachable with the wrist ~10 deg off.
                         # Expect larger IK residuals and weaker grasp authority there.
                         # Same-side and z limits are unchanged and still fail closed.
DYN_RETRACK_M = 0.01     # follow the live cube once it moved more than this — a
                         # deadband so detection jitter cannot wiggle the arm

# --- dual-cube hold (each hand keeps its own cube) -----------------------------------
CROSS_MIDLINE_REACH_M = 0.15  # how far ACROSS the midline an arm can still serve
                         # a cube, when its own-side arm is unavailable.
                         # 08-10 (user): 0.10 -> 0.15, doctrine restated — the
                         # only things allowed to prevent a grasp are a
                         # collision rejection and the base_link reach
                         # envelope; the recorded band must not pre-refuse
                         # what the solver could judge. The 07-30 sweep put
                         # the soft boundary at ~0.10 (front sector); 0.15
                         # lets the solver rule on the extra 5 cm itself —
                         # expect honest INFEASIBLEs out there.
                         # PRIOR: the 07-30 sweep across y in [-0.30, +0.30]
                         # found the near arm cleared every gate and crossing
                         # was INFEASIBLE PAST ~0.10 m — i.e. the finding is a
                         # BAND, and inside it either arm has a route.
                         # 08-07 hardware: journey_serving_arm implemented that
                         # finding as a SIGN TEST, so a cube measured at
                         # y=+0.026 — 26 mm, a fifth of the way to the bar, and
                         # inside the detector's own noise — was called
                         # "left side" and the leg was refused with the right
                         # hand free. A 100 mm tolerance had become a 0 mm one.
                         # NOT the same as MIDLINE_MARGIN_M below: that guards a
                         # cube ALREADY IN A HAND being dragged across, which is
                         # a collision question. This one is a reach question.
MIDLINE_MARGIN_M = 0.03  # hard same-side guard: a HELD cube dragged past the body
                         # midline by more than this is "not servable by this arm".
                         # Arms never cross the midline (collision safety — the IK has
                         # no reliable arm-vs-arm avoidance; user places one cube per
                         # half-space). The margin is a hysteresis band so detection
                         # jitter at y≈0 cannot flap the verdict.
HOLD_RELEASE_S = 3.0     # persistently-unservable time (wrong arm per the gate, or
                         # across the midline) before a holding arm gives its cube
                         # back to the pending queue for the correct arm to take

# --- target safety envelope (2026-07-15 adversarial audit) ----------------------------
# The IK has NO self-collision awareness and none of the reach math looked at z, so a
# cube placed against the torso or held at face height became a literal IK target
# (confirmed hardware-damage findings: trunk keep-out + z-band both absent). Any cube
# outside this envelope is treated exactly like an out-of-reach one: not servable,
# never latched, never followed.
TARGET_MIN_RADIUS_M = 0.16  # keep-out cylinder around the trunk axis (torso half-width
                            # + gripper body; every validated bench cube sits ≥ 0.20)
TARGET_MAX_RADIUS_M = 0.60  # 08-10 (user): back to 0.60. The 08-09 raise to
                            # 0.68 answered a landing problem (cube settled at
                            # 0.598-0.64, riding the line, T07 aborting on
                            # 3 mm of 1-face noise) — the REAL fix landed the
                            # same night: arrival cutoffs 0.55/0.60 put the
                            # landing band at ~0.32-0.55, clear of the line,
                            # so the envelope returns to the measured value.
                            # The T07 12-tick debounce (was 8) stays as noise
                            # armor. If a leg undershoots and refuses at
                            # 0.60-0.65, that is a landing problem, not an
                            # envelope one.
                            # 07-24 (user): 0.45 -> 0.60. MEASURED THAT DAY on the
                            # deploy model (400k FK samples over the left arm's 7
                            # joints, z in [-0.40, 0.30], EE orientation measured
                            # against identity):
                            #   position only, any orientation ....... 0.755 m
                            #   EE within 15 deg of NEUTRAL .......... 0.650 m
                            #   EE within 10 deg of NEUTRAL .......... 0.619 m
                            #   EE within  5 deg of NEUTRAL .......... 0.514 m
                            # The runtime Cartesian IK is given NEUTRAL_QUAT, so the
                            # CLEAN envelope is the ~0.51 m row; 0.60 m can only be
                            # held by letting the wrist sit ~10 deg off, i.e. position
                            # and orientation can no longer both be satisfied and the
                            # solve saturates with residual.
                            # THIS IS THE MARGINAL BAND THAT ALREADY BIT US: raising
                            # the side home to r=0.608 converged on three runs and
                            # failed on the fourth, producing a startup self-collision
                            # (07-24). Expect degraded accuracy and full-stretch poses
                            # beyond ~0.55; dial back to 0.50-0.55 if the hand misses
                            # or a shoulder starts loading up.
                            # It also RETIRES this constant's second job: r_xy ~ 0.5
                            # readings used to be the tell-tale of torso-tilt geometry
                            # corruption (hang-too-low class, 07-18) and were refused
                            # here. At 0.60 that detector no longer fires — RULE #0
                            # (hang/stand with the torso visibly upright) is now the
                            # only thing standing between a tilted base and a bad reach.
                            # ORIGINAL NOTE (07-18): the arm PHYSICALLY ends ~0.45 m out — targets
                            # beyond it are unreachable AND are the signature of the
                            # torso-tilt geometry corruption (hang-too-low class:
                            # cubes read z=+0.2 chest height at r_xy≈0.5 and the
                            # OOD-fooled learned gate lets the mission reach at
                            # them — observed AGAIN 07-18 14:24). Refuse loudly.
TARGET_Z_MIN_M = -0.40      # below ≈ knee-height reach from the bench hang
TARGET_Z_MAX_M = 0.50       # 08-11 (user, for the dual-follow demo: within
                            # reach + seen should be the ONLY follow gates):
                            # 0.30 -> 0.50 — the band now tops out around the
                            # robot's chin (~1.1 m above ground at stand)
                            # instead of the chest. The face-height caveat was
                            # stated and OVERRIDDEN by the user: mink's follow
                            # path has no self-collision sense, so a cube held
                            # at the new ceiling aims the hand just below the
                            # head/cameras — handlers, keep the cube off the
                            # face. SHARED dial: the standing missions'
                            # target acceptance and the MPC envelope guard
                            # (T07) read the same constant.
FRONT_HOME_TOL_RAD = 0.35   # "arm is at the front/power-on posture" tolerance: real
                            # PD-vs-gravity sag of the HELD pose measured ±14 deg
                            # (07-15); SIDE differs by 85 deg, REAR by 155 deg, so
                            # discrimination is untouched. Shared by the startup gate,
                            # _idle_arms_home and the per-arm REST gate.

# --- grasp + carry-home (--grasp; needs real_env --ee-service + the EE service) -------
# After a hold settles, the gripper closes on the cube (load-supervised hand_grab in
# humanoid_end_effector_service.py, forwarded by real_env from our packet's ee_action
# field); on grasp_detected the arm CARRIES the cube home to front along the normal
# track machinery and parks at REST with the cube in hand (terminal). First grasped
# walks home first — carries are ordinary tracks, so the one-staircase-at-a-time and
# Guard-1/Guard-2 serialization apply unchanged. The grab itself does NOT touch the
# joint stream, so a grab may overlap the other arm's reach; only the walk home queues.
GRASP_SETTLE_S = 1.0         # hand must be ON target and target unmoved this long
GRASP_EE_OK_M = 0.06         # ...with the measured EE within this of the held target
                             # (a timeout-registered hold never converged: never grab air)
GRASP_RESULT_TIMEOUT_S = 20.0  # worst-case hand_grab ≈ 12 s (service constants) + margin
GRASP_PRECON_RETRY_S = 15.0  # 07-26 deadlock audit: a registered hold whose ONLY
                             # unmet grab precondition is the EE-distance check
                             # (measured EE stuck >= GRASP_EE_OK_M from the held
                             # target — gravity sag / IK residual) used to spin
                             # forever and block the base. After this long the
                             # hold releases through the ordinary release path,
                             # which re-queues the cube and retries without limit.
# GRASP_MAX_TRIES removed 07-21 (user): empty pinches retry WITHOUT LIMIT until
# the cube is held — "failed" is reserved for hardware faults (EE chain death /
# result timeout). grasp_tries still counts, for the logs.
# 07-16 grasp-chain audit hardening: the EE command/status path is lossy
# (latest-value-wins, no acks) and its liveness used to be invisible — the
# operator would mislabel a DEAD chain as physical "empty pinches" and could
# hand_open a gripper that was actually holding the cube (dropping it).
EE_ALIVE_WAIT_S = 15.0       # startup wait for telemetry ee_alive before zeroing;
                             # expired+False ⇒ grasping disabled loudly for the run
EE_ACTION_REPEAT_TICKS = 30  # re-send each ee_action this many packets (1.5 s at
                             # 20 Hz): the command channel is latest-value-wins at
                             # BOTH hops (operator→real_env→service) and was seen
                             # eating commands live on 07-16 — 5 repeats (250 ms)
                             # did not cover real_env stalls. Duplicates are safe:
                             # grab/zero re-sends are rejected busy by the service,
                             # open is idempotent.
# Thermal early-warning for the OVER CURRENT class (07-18: left_wrist_2
# latched a firmware over-current fault after sustained side-home load with
# zero temperature visibility). Winding temps ride telemetry "motor_temp"
# (real_env marked patch); Robstride firmware protection trips ~80 C.
# Print-only — no behavior change; gives the human time to rest the arm.
MOTOR_TEMP_WARN_C = 60.0     # warm: normal continuous running is 35-55 C
MOTOR_TEMP_HOT_C = 70.0      # 10 C from the firmware trip — REST the arm now
MOTOR_TEMP_RATE_C_S = 0.5    # sustained rise this fast = STALL signature
                             # (a jammed joint at ~87 W winding heat climbs
                             # degrees per second — fires long before 60 C)
MOTOR_TEMP_RATE_WIN_S = 5.0  # rate is averaged over this window
TILT_WARN_DEG = 8.0          # torso off vertical: warn (standing reaches lean a few deg)
TILT_REFUSE_DEG = 15.0       # STARTUP REFUSED above this — a tilted torso corrupts the
                             # ENTIRE body-frame geometry chain (07-16 + three 07-17/18
                             # recurrences: table cubes read z=+0.2 chest height at
                             # r_xy≈0.5, the OOD learned gate waves them through, and
                             # mink's use_yaw_frame IK diverges from our body-frame
                             # commands so the arms never land where we point). The
                             # operator can now SEE it directly: telemetry
                             # projected_gravity. Fix the HANG/STANCE, don't override.
BEARING_JUMP_DEG = 45.0      # side-home VIA-HOME rule (user 07-18, hard requirement):
                             # an EXTENDED arm never chases a target whose bearing
                             # jumped more than this (e.g. the cube teleporting
                             # front→rear) — it retracts to the SIDE home FIRST and
                             # only re-extends once the measured EE is physically AT
                             # home. The side home (bearing 90°) IS the stage: every
                             # big transition routes through it, nothing crosses the
                             # body directly.
HOME_SETTLE_TOL_M = 0.06     # both arms measured within this of the home EE before
                             # the FIRST reach may start (user 07-18: home first,
                             # THEN grab — never reach from the power-on pose)
HOME_SETTLE_TIMEOUT_S = 20.0 # glide is ~5 s; a persistent shortfall (EE telemetry
                             # gap, obstruction) proceeds with a LOUD warning
                             # instead of stranding the mission forever
GRASP_ZERO_GRACE_S = 10.0    # no hand_grab until this long after the startup zeros
                             # (a grab during ZEROING_IN_PROGRESS is rejected unseen)
GRASP_DROP_DIST_M = 0.15     # cube seen this far from the gripper while "in hand"...
GRASP_DROP_PERSIST_S = 2.0   # ...for this long ⇒ it was DROPPED: reopen, re-queue
# Lift-off clearance (07-17, user): cubes sit on a STAND — walking the carry
# staircase straight from the grasp pose can catch the gripper/cube on it.
# After a confirmed grasp: raise vertically, then slide OUTWARD (away from the
# midline — never toward the other arm), then carry home as usual. The slide
# waypoint must stay inside the hold's own sector (staging invariants) and the
# safety envelope, else the slide is skipped (lift only).
GRASP_LIFT_UP_M = 0.05       # vertical clearance over the stand top
GRASP_LIFT_OUT_M = 0.08      # max clearance-arc length toward the sector station (~8 cm)
GRASP_LIFT_TIMEOUT_S = 8.0   # budget for the whole clearance move (then carry anyway)

# --- staged reach for side/rear objects (pre-defined transit poses) ------------------
# Upstream's concern: mink IK is a LOCAL solver — asked to go from the front rest pose
# straight to a side/rear target, the QP's path crosses the torso basin and the arm
# strikes the body (observed on hardware; the reason rear reaches were banned). Fix:
# classify the latched target by base-frame bearing and route the arm through
# pre-defined transit poses FIRST, so the IK warm start already sits in the safe
# basin before the real reach begins. Retraction walks the same chain in reverse.
SECTOR_FRONT_DEG = 60.0  # |bearing| ≤ this → FRONT: direct reach (behavior unchanged)
SECTOR_REAR_DEG = 120.0  # |bearing| > this → REAR; between the two → SIDE
SECTOR_HYST_DEG = 10.0   # a held cube must move this far past a boundary before the
                         # hold treats it as having left its sector (jitter guard)
STAGE_JOINTS = {         # LEFT-arm transit POSTURES (7 joints, shoulder_1..wrist_3);
    # the right arm is the exact negation (deploy-model mirror convention).
    # Chain + rate hardware-validated 2026-07-14 (humanoid_stage_walk_test.py):
    # zero contact at 0.025/0.1/0.2/0.3 rad/s; offline audit: >=40 deg limit
    # margins, >=49.9 mm clearance on the true equal-rate curves BOTH directions
    # plus 300 random in-envelope poses (scratchpad joint_stage_audit.py).
    "side": [0.0, -0.9599, 0.0, 0.0, 0.0, 0.0, 0.0],
    # ^ user-defined SIDE: PURE CORONAL ABDUCTION, RAISED 07-17 from 25 to 55 deg
    #   (EE z -0.183 -> +0.012): the transit hub used to sweep at tabletop height
    #   and clipped the cube stand's legs. Same axis (shoulder_2 only), bearing
    #   unchanged (90.1 deg), abduction headroom to -180 deg.
    "front": [-0.1943, -0.6844, -0.3430, 1.6406, -0.4615, -0.6902, -0.5096],
    # ^ FRONT: RAISED home/transit station (EE z +0.18). Solved 07-17 at the
    #   THEN power-on EE xy with z+8 cm (max 0.16 rad/joint from the v83-era
    #   power-on). Power-on has since moved twice (chest 07-30, sim home
    #   07-31) and now sits 1.18 rad away at shoulder_1 — the old "still
    #   within FRONT_HOME_TOL_RAD of power-on" adjacency is GONE, and the gate
    #   accepts an arm here because the front station is its OWN entry in
    #   every accepted-homes list, not because it is near power-on. Retreat
    #   chains END here.
    "rear": [0.0951, -0.8619, 0.3088, -1.3323, 1.1987, -0.8343, -0.3777],
    # ^ REAR: the raised front mirrored through the coronal plane — re-solved
    #   07-17 at (-0.322, 0.252, 0.18) on the deploy model (0.1 mm residual,
    #   >=48.7 deg limit margins). front/rear heights match (user spec).
    #   Runtime never solves IK for postures.
    # 07-17 offline re-validation of the RAISED chain: per-joint equal-rate
    # curves front<->side<->rear BOTH directions, right arm parked at the new
    # mirrored front: EE z never below -0.09 (was -0.183), self-clearance
    # >=28.9 mm every step (bounded by the STATIC shoulder-torso pair — the
    # moving links never got closer). Hardware re-validation pending: watch the
    # first staircase run.
}
POWERON_JOINTS = [-1.3740, -0.1870, +0.1310, +1.9290, -1.2403, -1.4546, -0.0090]
# ^ 07-31 (user): THE SIMULATOR'S HOME IS NOW THE POWER-ON POSTURE — the same
#   vector as CHEST_HOME_JOINTS above, which carries the full derivation
#   (source constant, units seam, why joint-space identity with the sim is the
#   spec). This vector is a MIRROR of the deployed env_config default_pose —
#   the robot's copy is authoritative, this one exists so the tools can reason
#   offline, and tests/test_model_derived_constants compares them so they
#   cannot drift.
#
#   History: the low tuck until 07-29, the raised chest posture (
#   [-0.4435, -0.2747, +0.0743, +1.7540, -1.4812, -0.4797, -0.0125]) 07-30,
#   the sim home since 07-31. The 07-30 finding that makes rewriting this SAFE
#   still holds and is re-stated here because it is what licenses every future
#   edit too: in this deployment default_pose is NOT in the observation and NOT
#   the arms'
#   action baseline (joint_pos is absolute, target_arm_joint_pos is absolute,
#   and the arm PD baseline is current_target_arm_pos + policy residual). It
#   drives the startup ramp, the initial arm target, and the failsafe
#   destination. So the working pose became the power-on pose.
#
#   Startup consequence to expect on hardware: prepare()'s 3 s ramp travels
#   ~67 deg at shoulder_1 from the old low tuck (53 deg from the 07-30 chest
#   home) with wrist_2 close behind — the biggest power-on excursion this file
#   has carried. First run after this change: hang the robot with legs straight
#   (RULE #0) and watch the ramp before trusting it standing.
#
#   Startup still ALSO accepts the old v83 power-on (below, kept so a rollback
#   to the v83 task never strands a session) and the raised front.
POWERON_JOINTS_LEGACY_0730 = [-0.4435, -0.2747, +0.0743, +1.7540, -1.4812,
                              -0.4797, -0.0125]
# ^ The 07-30 raised chest posture — power-on AND chest home for exactly one
#   day. Kept as a RECOGNISED legacy home for the same reason V83 below is: the
#   robot is physically parked here right now (last pre-migration session), an
#   env_config rollback would put the failsafe crawl's endpoint back here, and
#   a home the gates no longer recognise means STARTUP REFUSED plus a needless
#   power-cycle, 53 deg from anything accepted. The old-chest-home -> sim-home
#   glide was clearance-audited 07-31 (39.5 mm, endpoint-bound — see
#   JOURNEY_POSTURE_TOL_RAD), so accepting a start here is measured, not hoped.
POWERON_JOINTS_V83 = [-0.1786, -0.6451, -0.4061, 1.4804, -0.4229, -0.5632, -0.5089]
# Right arm = left arm mirrored through the sagittal plane, per joint.
#
# 07-28: this was briefly (-1,-1,-1,-1,-1,+1,-1). The gimbal-lock-free wrist
# arrived upstream with left_wrist_2's axis turned to match the right arm's,
# which breaks the plain negation — but the physical motor was never rewired.
# The hardware gravity-torque check (humanoid_mass_check.py, both arms held at
# the mirrored default_pose) caught it: the left arm was off by 1.15 N.m with
# wrist_2's term sign-inverted while the right matched at 0.18 N.m. Flipping
# that axis back in the deploy model (rebuild_deploy_model.py layer 4) takes the
# left residual to 0.22 N.m and restores this vector to a plain negation, which
# is what every stored posture in this file was authored in.
#
# WHY IT STILL LIVES HERE rather than being inlined as `-q`: a wrong sign does
# NOT fail loudly. Negated targets stay inside wrist_2's +-1.6057 limit, so a
# staircase reaches them, reports "settled", and the gripper simply sits
# 2*|wrist_2| (64-124 deg) rotated from the audited posture — no timeout, no
# gate refusal, no error line. humanoid_joint_monkey_hw.py DERIVES these signs
# from the loaded model's world joint axes and loud-fails on disagreement; run
# it after any upstream model change.
ARM_MIRROR_SIGN = np.array([-1., -1., -1., -1., -1., -1., -1.])

# Parallel dual-arm staircases: both arms may walk their joint tracks at once.
#
# The safety property is a MID-SAGITTAL SEPARATING PLANE, not a magic number:
# every collision geom of the left arm stays at y > 0 across its entire
# staircase, the right arm mirrors it, so the arms cannot touch at any phase
# combination and no serialization is needed. 07-21 measured the margin at
# 239.9 mm on the old wrist; re-measured 07-28 on the redesigned wrist
# (geom bounding spheres, box-path sampling over all six transitions):
#
#     nominal            204.3 mm      (was 239.9)
#     +-3 deg sag/error  173.0 mm
#     +-6 deg sag/error  139.0 mm      <- the audited sag case
#     +-10 deg            90.1 mm
#
# Sag is applied as the worst sign combination over all three shoulder joints,
# and bounding spheres over-estimate each geom, so the true separation is
# larger than these. The plane holds with room; re-enabled.
PARALLEL_STAIRCASE_CERTIFIED = True


def mirror_arm(q7, arm: str) -> np.ndarray:
    """Map a LEFT-arm joint vector onto `arm` (identity for left).

    Every stored arm posture in this module is authored for the LEFT arm; this
    is the ONLY sanctioned way to obtain the right arm's equivalent. Do not
    reintroduce a bare `-q` — see ARM_MIRROR_SIGN.
    """
    q = np.asarray(q7, dtype=float)
    return q if arm == "left" else q * ARM_MIRROR_SIGN
# A single BIDIRECTIONAL TRACK replaces the old forward-queue / reverse-queue pair.
# The staging postures are one linear track; the arm is a bead on it addressed by a
# station INDEX. side is always the middle station, so any front<->rear motion
# physically passes through side. Motion = (cur_idx, target_idx); direction =
# sign(target-cur). One shared primitive (_step_toward) walks it in both directions.
TRACK = ("front", "side", "rear")            # index 0/1/2 → STAGE_JOINTS key
SECTOR_IDX = {"front": 0, "side": 1, "rear": 2}
FRONT_STAGE_Z_M = 0.0    # --stage-sectors: a FRONT cube ABOVE this z stages through
                         # the front station; at/below it the leg stays a direct
                         # Cartesian reach (the hardware-validated low-cube flow, and
                         # what keeps dual-parallel front+front fully simultaneous).
                         # WHY high targets must stage: the direct stream from the
                         # side home hits the elbow-sign bifurcation — the local IK
                         # keeps the elbow positive and swings the shoulder instead
                         # (07-18 hardware: rear chest-height target solved as a
                         # -111 deg backward shoulder sweep, 42-64 mm short). From a
                         # settled station the solver is ALREADY in the right basin:
                         # S-A audit 07-18 — rear real target 0.6 mm (vs 42.4 direct),
                         # front r=0.44 logged target 0.8 mm (vs 23.5 direct).
STATION_BEARING_DEG = {"front": 28.80, "side": 90.08, "rear": 140.53}
# ^ base-frame EE bearing of each STAGE_JOINTS posture (left arm; right mirrors),
#   FK'd offline on the deploy model. The grasp clearance arc sweeps toward the
#   held cube's sector station — each bearing is interior to its sector band, so
#   the sweep cannot leave the sector.
#   RE-DERIVE WHENEVER STAGE_JOINTS OR THE MODEL CHANGES:
#       atan2(EE_y, EE_x) of FK(STAGE_JOINTS[name]) at the `end_effector_L_site`,
#       with qpos in ENCODER units (qpos[7:38] = the posture, NOT qpos0 — see the
#       wrist_1 `ref` note in rebuild_deploy_model.py).
#   07-28: was {38.07, 90.08, 142.00} — those were the OLD-wrist bearings and
#   survived the new-wrist migration unnoticed while REST_EE / JOURNEY_WALK_EE
#   were re-derived (front off by 9.3 deg). Benign in effect but not in kind:
#   lift_arc picks its sweep DIRECTION from sign(station - current), so a front
#   cube bearing between the stale and true values arced the wrong way. It erred
#   outboard (away from the midline), which is why nothing was seen on hardware.
STAGE_RATE = 0.3         # commanded joint rate [rad/s] (user-tuned; needs the real_env
                         # per-packet rate knob ae1ca6c; robot-side hard cap 0.5)
                         # 07-20 (user): 0.2 -> 0.3 at FULL 0.8 torque only — the
                         # 07-19 torso clip at 0.3 is attributed to low-torque sag
                         # (user observed 0.8 runs clean at 0.3). Conditions: torque
                         # 0.8, cubes at r >= 0.26 m (sim: 6-deg sag penetrates at
                         # 0.22); drop back to 0.2 if any loaded carry scrapes.
STAGE_TOL_RAD = 0.03     # per-joint arrival (1.7 deg): looser tolerances CUT CORNERS
                         # at speed and deviate from the audited curve
STAGE_DWELL_S = 0.6      # freeze at each waypoint so every segment starts exactly
                         # from the audited configuration
STAGE_REACH_EE_RATE = 0.12   # m/s: EVERY final Cartesian approach (07-16: front too) is
                             # STREAMED from the current hand position toward the cube
                             # at this speed — handing the IK the full 30 cm jump in
                             # one packet read as "too fast" on hardware (user).
                             # FRONT legs keep the classic direct-target behavior.
                             # 07-20: 0.08 -> 0.12 -> 0.18 -> back to 0.12 (user;
                             # 0.18 was too hot). NOTE: STAGE_RATE was separately
                             # raised to 0.3 the same evening (user decision at
                             # full 0.8 torque — see its own comment block for
                             # the conditions).
# WHY JOINT postures and not Cartesian transit points: 7-DOF redundancy lets the
# online IK escape to a different configuration branch even for FK-matched pose
# targets (measured: 163 deg away on the first hop), and the arbitrary-quat
# Cartesian attempt audited at -33.7 mm wrist-through-hip. Only the joint stream
# reproduces the audited chain. The final cube hop DOES use Cartesian mink — the
# staircase has already parked the arm in the right basin, and real_env 8335075
# re-seeds the solver from encoders on the joint->Cartesian switch.
# SIDE/REAR history: disabled 2026-07-15 after a hardware cross-body jump (arm swept
# front→rear through the torso). Both facets are fixed and offline-verified — see the
# crossbody-jump-fix memo: (1) dual-arm interleave, serialized by GUARD 1 in
# _held_update + GUARD 2 in _tick_search (joint tracks and GO/REACH reaches are
# mutually exclusive; lagged repro scratchpad/repro_crossbody_jump.py: 40 override
# ticks pre-fix → 0 post); (2) the held-arm follow is sector-clamped via _sector_left
# (scratchpad/test_sector_clamp.py). Re-enabled for HARDWARE VALIDATION — validate
# single-arm side/rear FIRST, then dual-cube; keep the safety rig.
SECTOR_REACH_ENABLED = {"front": True, "side": True, "rear": True}

# --- visual servo (gripper-tag closed loop) -----------------------------------------
# The open-loop reach lands with a multi-cm bias: the cube→hand chain crosses the
# camera extrinsics (gimbal mount + hand-calibrated zeros) AND the arm FK (hand-
# calibrated joint zeros), and the encoder loop cannot see either error. During
# REACH/HOLD the camera also sees the gripper's own base tags (16 mm pads), giving a
# direct measurement of the SAME physical frame the model FK also predicts (MJCF body
# L_base/R_base — the tagged-body registry uses the same CAD frame). The servo is a
# BIAS ESTIMATOR, not an error integrator:
#     b̂ ← EMA of (camera-measured base − FK-predicted base)      [camera-world vs FK-world]
#     published target = latched reach target − b̂
# At equilibrium the camera sees the hand exactly where FK believes it is, so the
# validated open-loop geometry (IK site → cube − standoff) is preserved bit-for-bit —
# the servo never introduces a site↔tag-origin offset (rejected alternative: driving
# the tag origin onto the standoff point walks the hand ~8.6 cm past the validated
# contact pose). The estimator form is idempotent per frame (re-processing the same
# frame re-estimates the same b̂), so measurement latency cannot wind it up.
#
# Trust rules (each one earned by an adversarially-confirmed failure mode):
#   · b̂ only from the camera that measured the latched cube fix — common-mode errors
#     cancel only within one chain (the monitor's targets dict is last-camera-wins,
#     so the claim port is NOT that camera; the latch records the measurement port).
#   · b̂ is per-LEG (reset at every latch): the underlying angular errors rotate with
#     gimbal aim and arm pose, so a bias learned at cube #1's bearing miscorrects at
#     cube #2's (rejected alternative: persisting b̂ across legs).
#   · Updates only when the previous command was EXECUTED (EE-FK near the published
#     target) AND the arm and gimbal are quasi-static across camera frames — a frame
#     is captured 50-150 ms before it arrives, so anything moving pollutes b_meas.
#   · ‖Δb̂‖ ≤ 1 cm per accepted frame and ‖b̂‖ ≤ 8 cm: one bad frame moves the hand
#     ≤1 cm, any run of bad frames ≤8 cm from the validated open-loop pose. No
#     absolute workspace box (rejected: it corrupts legitimate high/low targets by
#     clipping the whole target, not the correction).
#   · REACH does not report success until the servo has actually corrected
#     (SERVO_MIN_UPDATES) or provably cannot (tags unusable for SERVO_GIVEUP_S, or
#     the 8 s timeout) — otherwise non-final cubes would exit at open-loop accuracy.
#   · Blocked-and-blind self-heal: in HOLD, if the command stops being executable
#     (obstacle) and no fresh accepted measurement arrives, b̂ decays toward 0 so the
#     hand walks back to the validated open-loop pose instead of pressing forever.
# Requires the monitor to publish the gripper body (--bodies ... gripper_L_base).
SERVO_GRIPPER_KEYS = {"left": "gripper_L_base", "right": "gripper_R_base"}
SERVO_BODIES = {"left": "L_base", "right": "R_base"}  # MJCF bodies of the same CAD frames
SERVO_GAIN = 0.3           # EMA weight per accepted measurement
SERVO_MIN_INLIERS = 2      # single 16 mm-tag PnP is garbage-prone → require ≥2 tag faces
SERVO_FRESH_S = 0.5        # gripper sighting older than this → no update
SERVO_SETTLE_M = 0.03      # ‖EE-FK − last published target‖ below this → command executed
SERVO_STATIC_M = 0.005     # EE-FK moved less than this between camera frames → arm static
SERVO_STATIC_GIMBAL_RAD = 0.01  # per-frame gimbal motion above this → extrinsic lag, skip
SERVO_MEAS_REJECT_M = 0.15 # single measurement offset beyond this = mis-ID/garbage, dropped
SERVO_STEP_M = 0.01        # ‖Δb̂‖ per accepted frame → published target moves ≤1 cm at once
SERVO_BIAS_MAX_M = 0.08    # ‖b̂‖ clamp: real hand bias is a few cm; bounds total deviation
SERVO_MIN_UPDATES = 3      # REACH holds its done-verdict until this many accepted updates
SERVO_GIVEUP_S = 3.0       # no usable gripper sighting this long after arrival → open loop
SERVO_DECAY_MPS = 0.001    # blocked-and-blind b̂ decay rate in HOLD [m/s]
CUBE_LATCH_WINDOW_S = 0.7  # reach target = per-axis median of sightings this recent



class Phase(Enum):
    """Mission phase. Dispatched once per tick; the order below is the flow.

    SEARCH scans for cubes, GO walks, REACH runs the Cartesian approach, HOLD
    keeps the hand on the reached cube (and keeps staring) until Ctrl+C, PARK
    retracts, DONE terminates. HOLD is deliberately terminal-until-interrupted:
    releasing on a timer would drop a gripped cube.
    """
    SEARCH = "SEARCH"
    GO = "GO"
    REACH = "REACH"
    HOLD = "HOLD"    # keep the hand ON the reached cube + keep staring, until Ctrl+C
    PARK = "PARK"
    DONE = "DONE"


@dataclasses.dataclass
class CubeTrack:
    """Per-cube perception state: where it was last seen, by which eye, how well.

    Position is BASE-FRAME, so it rots as the robot moves -- `fresh()` is not
    optional bookkeeping, it is what stops a stale pose being treated as a
    place in the world. `camera_port` is the claiming eye (one cube, one eye);
    `pos_port` is the eye that measured the current sample and can differ,
    because the monitor's targets dict is last-camera-wins when both see it.
    `hist` keeps the last few capture-stamped sightings so the reach latch can
    median over them: a single 1-inlier PnP glitch at the arrival tick must not
    poison a leg.
    """
    key: str                       # detection key, e.g. "grasp_cube_40mm"
    camera_port: int | None = None # claiming camera (assigned on first sighting)
    last_camera_port: int | None = None  # survives claim release — the camera to
                                   # RETIRE when this cube is carried home (07-20
                                   # user spec: a done camera parks, never scans)
    pos: np.ndarray | None = None  # latest base-frame position [m]
    pos_port: int | None = None    # camera that MEASURED pos (≠ claim: the monitor's targets
                                   # dict is last-camera-wins when both cameras see the key)
    last_seen: float = -1e9        # operator clock of last sighting
    first_seen: float = -1e9       # start of the current uninterrupted sighting streak
    reached: bool = False
    held_by: str | None = None     # arm currently keeping its hand on this cube
    assigned_to: str | None = None  # independent pre-hold owner; mutually exclusive
    hist: deque = dataclasses.field(default_factory=lambda: deque(maxlen=5))
    # recent `Sighting`s, capture-stamped and deduplicated — the reach latch medians
    # over these so a single garbage frame (1-inlier PnP glitch) at the arrival tick
    # cannot poison a leg

    def fresh(self, now: float, horizon: float) -> bool:
        """Position usable: seen within `horizon` s. Base-frame positions rot as the
        robot moves, so anything older is geometrically meaningless."""
        return self.pos is not None and (now - self.last_seen) < horizon

    def confirmed(self, now: float) -> bool:
        """THE FIRST SIGHTING COUNTS (user 2026-08-02): confirmed == fresh.

        The retired >=0.3 s streak filtered one-frame flukes captured mid-sweep
        (gimbal moving -> FK-extrinsic skew). It predates first-seer locking:
        the eye now STOPS on the very frame that saw the cube, so the frames
        that follow come from a stationary camera and downstream medians
        self-heal within ~0.2 s anyway — while a skewed first target is still
        caught by the reach envelope, the six route gates, live MPC tracking
        (evidence-gated goal updates) and the drift abort."""
        return self.fresh(now, 1.0)


class AutoOperator:
    """Pure decision core: tick(detections, telemetry, now) → (nav_cmd, arm_packet).

    Separated from I/O so the synthetic-playback test can drive it without sockets.
    All positions are robot base-frame [m]; angles rad; velocities m/s, rad/s.
    """

    def __init__(self, cube_keys: list[str], gate=None, gaze_only: bool = False,
                 hold_after_reach: bool = True, visual_servo: bool = True,
                 reach_standoff: float = REACH_STANDOFF_M, dynamic_track: bool = True,
                 grasp: bool = False, aim_up: float = AIM_UP_M,
                 dual_parallel: bool = False, side_home: bool = False,
                 stage_sectors: bool = False,
                 independent_arms: bool = False,
                 journey: bool = False,
                 sync_reach: bool = False,
                 chest_home: bool = False):
        self.cubes = {k: CubeTrack(key=k) for k in cube_keys}
        self.phase = Phase.SEARCH
        self.grasp = grasp                        # close the gripper on held cubes and
                                                  # carry them home (see the GRASP_*
                                                  # constants block); default OFF so the
                                                  # classic hold behavior is unchanged
        self._ee_queue: list[dict] = []           # pending ee_action payloads (each is
                                                  # re-sent EE_ACTION_REPEAT_TICKS times
                                                  # in the 9874 packet; real_env
                                                  # --ee-service forwards them)
        self._ee_current: GripperBurst | None = None  # active ee_action repeat burst
        self._ee_next_action_id = time.time_ns()      # stable per-burst IDs; seeded
                                                      # across operator restarts
        self._hands_opened = False                # one-shot startup zeroing latch
        self._ee_wait_t0: float | None = None     # started waiting for ee_alive
        self._ee_chain_dead = False               # EE chain confirmed dead/lost: all
                                                  # grasping disabled for the mission
                                                  # (classic reach-and-hold continues)
        self._zeros_t: float | None = None        # when the startup zeros were queued
        self._grasp_safety_latched = False        # canonical global fail-closed latch
        self._grasp_safety_arm = None
        self._grasp_safety_reason = None
        self._grasp_intervention_required = False # terminal fail-closed latch: an
        self._grasp_intervention_reason = None    # ambiguous grasp/drop never opens
        self._intervention_arm_targets = None     # immutable two-arm hold snapshot
        self._last_arm_targets = None             # fallback if latch tick lacks jpos
        self.active: CubeTrack | None = None      # cube of the current GO/REACH/PARK leg
        self._held: dict[str, HoldState] = {}     # arm → hold entry: the hand stays on
                                                  # its reached cube (per-arm dynamic
                                                  # tracking state lives in the entry)
        self._sector_warned: set = set()          # cubes already announced as sitting
                                                  # in a disabled reach sector
        self._startup_posture_ok = False          # first-telemetry guard: a full
                                                  # mission may only begin with BOTH
                                                  # arms at the power-on pose — an
                                                  # operator restarted while an arm is
                                                  # parked side/rear would otherwise
                                                  # command a cross-basin Cartesian
                                                  # reach with no staircase (the demo
                                                  # walk test has the same guard)
        self._arm_engaged = False                 # True once any arm packet was sent:
                                                  # from then on the mission NEVER goes
                                                  # arm-silent — real_env keeps the last
                                                  # Cartesian target forever and re-engages
                                                  # its IK on it after stream silence, so
                                                  # silence after a staircase could snap
                                                  # the arm back to a stale cube target
                                                  # (observed on hardware 07-14)
        self.fk = GimbalCameraFK({p: CAM_SITES[p] for p in CAM_PORTS})
        # Per-camera optical frame PROBED from the model FK — (datum_bearing, yaw_sense,
        # pitch_sense). NEVER hardcode these: the 07-10 re-export silently flipped the
        # model's yaw sense (+yaw was LEFT, is now RIGHT) and inverted every hardcoded
        # aim formula. Probing keeps the aim correct under any future re-export.
        self._cam_frame = {p: self._probe_cam_frame(p) for p in CAM_PORTS}
        self.gate = gate                          # ReachabilityGate; None → geometric floor only
        self.gaze_only = gaze_only                # cameras-track-only: no walk, no reach
        self.hold_after_reach = hold_after_reach  # last cube: HOLD on it forever vs park+DONE
        self.reach_standoff = reach_standoff      # pullback of the reach point toward the robot
        self.aim_up = aim_up                      # lift of the reach point above the cube centre
        self.dual_parallel = dual_parallel        # front+front: both arms reach at once
        self.independent_arms = independent_arms and hold_after_reach
        # JOURNEY mission (07-21): needs the independent scheduler (per-arm
        # tasks own the two legs) — forced off without it.
        self.journey = journey and self.independent_arms
        # 07-21 (user, video demo): SYNC-REACH — the first camera-locked cube's
        # arm holds at the latch gate until the OTHER pending cube's arm is
        # also latch-ready, then both start their Cartesian reach in the same
        # tick (one shared packet, two slots). Cosmetic/synchronization only:
        # every safety gate is unchanged. If the second cube never shows, the
        # first arm waits forever (by design — demo mode, operator watches).
        self.sync_reach = sync_reach and self.independent_arms
        self._sync_wait_t = -1e9                  # throttle for the waiting print
        self._survey_order: list | None = None    # cube keys, near first, latched
                                                  # once at survey completion
        self._survey_warn_t = -1e9                # 10 s throttle for the waiting print
        self._journey_still_since: float | None = None  # measured-stillness timer
        # 07-21 (user): journey starts ALREADY tucked — power-on IS the tuck,
        # so the old mission-start side-home glide was a pointless 8 s round
        # trip (POWERON -> side -> back to chest). Arms hold the chest tuck
        # through zeroing + survey; the FIRST side-home glide happens at the
        # stop point, where reaching actually needs it.
        # (Briefly FALSE between the 07-30 power-on raise and the 07-31
        # one-home unification, while the tuck stayed at the old low default;
        # true again now — power-on, tuck and chest home are all the sim home.)
        self._journey_walk_pose = self.journey    # free arms tucked at JOURNEY_WALK_EE
        self._journey_serve_t = -1e9              # last tick nav served a journey task
        self._journey_lin_seen = False            # base_lin_vel ever nonzero (estimator alive)
        self._journey_blind_warned = False        # one-shot no-estimator warning
        self._journey_fail_warned = False         # one-shot failed-grasp stand warning
        self._journey_done = False                # MISSION COMPLETE printed
        self._base_vel = None                     # latest telemetry base_lin_vel
        self._base_ang = None                     # latest telemetry base_ang_vel
                                                  # symmetric per-arm FSM; legacy primary/
                                                  # secondary path remains available for replay
        self.side_home = side_home                # arms live at the SIDE; no sector
                                                  # staging; carry returns to the side
        # 07-24 (standing session): CHEST-HOME — the arms never
        # glide to side-home. They hold the CHEST tuck (CHEST_HOME_EE +
        # CHEST_HOME_QUAT since 07-31 — the sim home's own FK orientation, no
        # longer the walking tuck's) and every cube is reached DIRECTLY from
        # that chest pose. Since 07-31 the chest tuck IS the power-on posture,
        # so mission start is a zero-distance hold on a fresh boot and a real
        # homing glide only when a previous session moved the arms;
        # _home_settled waits either out.
        # Two deliberate constraints:
        #  * STAGING IS FORCED OFF below. A joint staircase replants from the
        #    SIDE station; with the arm parked at the chest that start posture
        #    is wrong, so a staged leg would sweep from the wrong basin (this
        #    is how the 07-24 run drove an arm into the tripod at the front
        #    station). chest-home therefore serves DIRECT legs only.
        #  * journey_posture_payload is NOT used. It stays gated on `journey`,
        #    so a standing chest-home mission never wakes the joint-posture
        #    payload — on this build that payload has neither the REACHING
        #    exclusion nor the hysteresis band, and membership in it IS an arm's
        #    control mode, so waking it would reproduce the reach-vs-posture
        #    judder this build is free of.
        self.chest_home = chest_home and side_home and self.independent_arms
        self.stage_sectors = stage_sectors and side_home \
            and not self.chest_home
        # ^ --stage-sectors (07-18, user): under SIDE-HOME, front-HIGH and ALL
        #   rear reaches route through their STAGE_JOINTS station first — the
        #   settled station re-seeds the robot-side IK in the human-chosen
        #   configuration basin (elbow-sign branch), which the direct stream
        #   cannot reach (spontaneous symmetry breaking at the bifurcation).
        #   side + low-front legs stay direct Cartesian (validated flow,
        #   keeps dual-parallel simultaneous). Meaningless without side_home
        #   (classic mode already stages every non-front leg).
        self.rest_ee = CHEST_HOME_EE if self.chest_home else \
            (SIDE_HOME_EE if side_home else REST_EE)
        self._sec: SecondaryReach | None = None   # in-flight SECONDARY reach
        self._sec_hits: list | None = None        # [cube_key, streak] latch debounce
        self._arm_tasks: dict[str, ArmTask | None] = {
            "left": None, "right": None}
        # 07-21 PHASE 2 (parallel staircases): the wire is per-arm since the
        # 07-20 protocol change, and the dual-staircase clearance audit closed
        # the safety question — arm-vs-trunk distance is a function of THAT
        # arm's 7 joints only (fixed base), so a peer staircase adds exactly
        # zero body risk; the only new class, arm-vs-arm, is bounded by a
        # mid-sagittal separating-plane certificate at >= 239.9 mm (measured
        # minimum 289 mm across all 16 hop pairs x phase grid, bare / loaded /
        # 6 deg sag). The token therefore becomes a SET: every eligible staged
        # arm may walk at once.
        self._ind_joint_owners: set[str] = set()  # staged tasks walking their staircases
        self._ind_gate_rr = 0                     # one fresh learned inference/tick,
                                                  # alternating arm-first service
        self._ind_gate_inferred = False
        self._ind_cartesian_grant = True          # preflight verdict for the next
                                                  # two-slot Cartesian packet; held
                                                  # ramps freeze when either slot has
                                                  # no finite publishable value
        self._ind_hold_cartesian_active = False   # tick-start latch: a hold's final
                                                  # Cartesian point publishes before a
                                                  # newly-created task takes joint mode
        self._nav_task_key: str | None = None     # assigned cube served by shared base
        self._nav_leg_dir: float | None = None
        self._home_stream: dict = {"left": None, "right": None}  # home-glide ramps
        self._home_settled = False                # both arms reached home once
        self._home_wait_t0: float | None = None   # settle-gate timeout clock
        self.dynamic_track = dynamic_track        # HOLD follows a moving cube / retracts
        self._gate_hits = 0
        self._gate_cache: dict = {}               # quantized pos → (score, best, t)
        self._arrived = False                     # per-leg arrival latch
        self._scan_phase = 0.0                    # serpentine parameter
        self._trim = {p: np.zeros(2) for p in CAM_PORTS}
        self._vx = 0.0                            # ramped output
        self._wz = 0.0
        self._leg_dir: float | None = None        # +1 forward / -1 backward, chosen once per leg
        self._reach_target: np.ndarray | None = None
        self._reach_arm: str = "right"
        self._reach_t0 = 0.0
        self._park_t0 = 0.0
        self.visual_servo = visual_servo          # gripper-tag closed-loop reach correction
        self._grip: dict[str, dict] = {}          # arm → latest gripper-base sighting
        self._servo_log_t = -1e9                  # print throttle (persists across legs)
        self._reset_leg_state(-1e9)               # per-leg servo + dynamic-track state
        # FK body ids for the gripper bases; empty → model predates the grippers, servo off
        import mujoco as _mj
        self._grip_body_id = {}
        if visual_servo:
            for arm, body in SERVO_BODIES.items():
                bid = _mj.mj_name2id(self.fk._model, _mj.mjtObj.mjOBJ_BODY, body)
                if bid >= 0:
                    self._grip_body_id[arm] = bid
            if not self._grip_body_id:
                print(f"[servo] model has no {list(SERVO_BODIES.values())} bodies — "
                      f"visual servo DISABLED (open-loop reach)")
                self.visual_servo = False
        if grasp and reach_standoff > 0.02:
            print(f"[grasp] WARNING: reach-standoff {reach_standoff:.3f} m > 2 cm — "
                  f"the gripper will close {reach_standoff*100:.0f} cm SHORT of the "
                  f"cube centre; grasping expects --reach-standoff 0")
        self.gaze = np.zeros(4)                   # last commanded gimbal reference
        self._cam_retired: set[int] = set()       # cameras whose cube was carried home:
                                                  # they PARK at neutral instead of joining
                                                  # the scan (07-20 user spec; also stops
                                                  # idle-sweep detections polluting the
                                                  # other camera's tracking). PERMANENT
                                                  # since 08-09 (user): never un-retired —
                                                  # not by drop, not by a claim, not by a
                                                  # pending unfixed cube.
        self._gaze_seeded = False                 # gaze publishes only after seeding from
                                                  # the gimbal encoders — a zeros-init first
                                                  # packet used to slam a parked gimbal to 0
                                                  # in one control step (07-15 audit)
        self._jpos = None                         # latest telemetry joints (per tick; used
        self._ee_act = {"left": None, "right": None}  # by _arm_packet's per-arm REST gate)

    # ---------------- eye ----------------
    def _probe_cam_frame(self, port: int) -> tuple[float, float, float]:
        """Moved to auto_operator.gaze.probe_cam_frame (S6)."""
        return ao_gaze.probe_cam_frame(self, port)

    def _aim_angles(self, port: int, target: np.ndarray, jpos) -> tuple[float, float]:
        """Moved to auto_operator.gaze.aim_angles (S6)."""
        return ao_gaze.aim_angles(self, port, target, jpos)

    def _scan_angles(self, port: int) -> tuple[float, float]:
        """Moved to auto_operator.gaze.scan_angles (S6)."""
        return ao_gaze.scan_angles(self, port)

    @staticmethod
    def _wrap(a: float) -> float:
        return ao_safety.wrap(a)

    def _slew_gaze(self, target: np.ndarray) -> np.ndarray:
        """Moved to auto_operator.gaze.slew_gaze (S6)."""
        return ao_gaze.slew_gaze(self, target)

    def _eye(self, jpos, now: float) -> np.ndarray:
        """Moved to auto_operator.gaze.eye (S6)."""
        return ao_gaze.eye(self, jpos, now)

    # ---------------- per-leg state ----------------
    def _reset_leg_state(self, now: float) -> None:
        """Fresh per-leg state — called at construction and at every GO→REACH latch.
        The servo bias is re-learned each leg (its angular error sources rotate with
        gimbal aim and arm pose), and dynamic-track / once-per-leg warning state must
        not leak between legs. Deliberately NOT reset: _servo_log_t (print throttle)
        and _grip (a measurement stream, not leg state)."""
        self._reach_port: int | None = None         # camera that measured the latched target
        self._servo_bias: np.ndarray | None = None  # b̂, the servo's hand-bias estimate
        self._servo_seen = False                    # any usable (same-port) sighting this leg
        self._servo_accepted = 0                    # accepted b̂ updates this leg
        self._servo_stamp: float | None = None      # capture_stamp last processed
        self._servo_cmd_prev: np.ndarray | None = None   # last PUBLISHED target (settle gate)
        self._servo_ee_prev: np.ndarray | None = None    # EE-FK at previous frame (static gate)
        self._servo_gimbal_prev: np.ndarray | None = None  # gimbal joints at previous frame
        self._servo_accept_t = now                  # time of last accepted update (decay logic)
        self._servo_warned: set = set()             # once-per-leg warning keys
        self._reach_tm: TrackMotion | None = None   # REACH's bidirectional track motion
                                                    # (front→sector out; front on timeout).
                                                    # None once the arm has settled at the
                                                    # target station and switched to Cartesian.
        self._reach_stream: np.ndarray | None = None  # streamed Cartesian point (staged legs)
        self._reach_sector = "front"
        # (per-arm dynamic-tracking state — follow/retract/release — lives in the
        #  self._held entries now, one block per holding arm, not per leg)

    # ---------------- perception bookkeeping ----------------
    def _ingest(self, detections: dict, now: float):
        """Moved to auto_operator.perception.ingest (S6)."""
        return ao_perception.ingest(self, detections, now)

    # ---------------- visual servo (gripper tags) ----------------
    def _warn_once(self, key: str, msg: str):
        """Moved to auto_operator.servo.warn_once (S6)."""
        return ao_servo.warn_once(self, key, msg)

    def _grip_base_fk(self, arm: str, jpos) -> np.ndarray | None:
        """Moved to auto_operator.servo.grip_base_fk (S6)."""
        return ao_servo.grip_base_fk(self, arm, jpos)

    def _servo_step(self, telemetry: dict, jpos, now: float):
        """Moved to auto_operator.servo.servo_step (S6)."""
        return ao_servo.servo_step(self, telemetry, jpos, now)

    def _servo_target(self) -> np.ndarray:
        """Moved to auto_operator.servo.servo_target (S6)."""
        return ao_servo.servo_target(self)

    def _servo_ready(self, now: float) -> bool:
        """Moved to auto_operator.servo.servo_ready (S6)."""
        return ao_servo.servo_ready(self, now)

    def _servo_leg_report(self, now: float):
        """Moved to auto_operator.servo.servo_leg_report (S6)."""
        return ao_servo.servo_leg_report(self, now)

    # ---------------- reach sectors (front / side / rear staging) ----------------
    @staticmethod
    def _bearing_deg(pos) -> float:
        return ao_safety.bearing_deg(pos)

    def _sector(self, pos) -> str:
        """FRONT / SIDE / REAR by base-frame bearing (auto_operator.safety)."""
        return ao_safety.sector_of(pos, SECTOR_FRONT_DEG, SECTOR_REAR_DEG)

    def _sector_left(self, pos, sector: str) -> bool:
        """Sector-band hysteresis exit test (auto_operator.safety)."""
        return ao_safety.sector_left(pos, sector, SECTOR_FRONT_DEG,
                                     SECTOR_REAR_DEG, SECTOR_HYST_DEG)

    def _lift_arc(self, wp1: np.ndarray, arm: str, sector: str) -> list[np.ndarray]:
        """Moved to auto_operator.arms.lift_arc (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.lift_arc(self, wp1, arm, sector)

    def _standoff_point(self, pos) -> np.ndarray:
        """Moved to auto_operator.mission.standoff_point (S6)."""
        return ao_mission.standoff_point(self, pos)

    # ---------------- bidirectional staging track ----------------
    @property
    def _home_idx(self) -> int:
        """TRACK station index of the arms' HOME basin. Classic profile: the
        front chest rest (0). stage-sectors side-home: the SIDE (1) — the arms
        live there, so every track starts/ends at station 1 and 'home' and
        'index 0' stop being the same symbol (07-15-era code hard-coded 0)."""
        return 1 if (self.side_home and self.stage_sectors) else 0

    def _staged_target_idx(self, sector: str, pos) -> int:
        """Station a reach/hold of `pos` must stage through; == _home_idx ⇒
        a direct Cartesian leg (no track). Routing (07-18 user decision A):
        classic mode unchanged (front direct, side/rear staged); side-home
        without stage-sectors: everything direct; side-home + stage-sectors:
        side and LOW front (z ≤ FRONT_STAGE_Z_M) direct — high front and ALL
        rear stage through their station (elbow-branch selection, S-A audit)."""
        if self.side_home:
            if not self.stage_sectors or sector == "side":
                return self._home_idx
            if sector == "front" and float(pos[2]) <= FRONT_STAGE_Z_M:
                return self._home_idx
            return SECTOR_IDX[sector]
        return SECTOR_IDX[sector]

    def _track_posture(self, arm: str, idx: int) -> np.ndarray:
        """Joint posture of track station `idx` for `arm` (right = exact negation).
        stage-sectors: station 1 is RE-BAKED to SIDE_HOME_JOINTS — the home IS
        the side station, so tracks arrive exactly where the arm rests (S-A
        audit 07-18: home↔front/rear chains 28.9 mm trunk clearance — the
        static shoulder-torso bound, moving links never closer — limit margins
        ≥31°, both arms; bearings unchanged). Classic chains keep the pure
        STAGE_JOINTS['side'] abduction posture untouched."""
        if idx == 1 and self.side_home and self.stage_sectors:
            return mirror_arm(SIDE_HOME_JOINTS, arm)
        return mirror_arm(STAGE_JOINTS[TRACK[idx]], arm)

    @staticmethod
    def _track_seq(cur: int, target: int) -> list[int]:
        """Station indices to VISIT walking from `cur` to `target`, in order.

        Outward (target>cur): [cur+1 .. target] — the arm is firmly at `cur`
        already (a rest/held station), so step straight out; this reproduces the
        old forward staircase (front(0)->rear(2) = [1,2] = side,rear).
        Inward (target<cur): [cur .. target] — INCLUDES cur as the first hop to
        RE-PLANT it, because an inward motion begins from a Cartesian cube-follow
        pose that has drifted off the exact station posture; this reproduces the
        old retreat re-confirming its terminal station first (rear(2)->front(0) =
        [2,1,0]). target==cur → [] (no motion)."""
        if target > cur:
            return list(range(cur + 1, target + 1))
        if target < cur:
            return list(range(cur, target - 1, -1))
        return []

    def _new_track(self, arm: str, cur: int, target: int, now: float,
                   replant: bool = False) -> TrackMotion:
        """A fresh track-motion state for `arm` from `cur` toward `target`.

        replant=True forces `cur` as the FIRST hop even when _track_seq's
        index-order rule would skip it — required whenever the arm starts
        from a DRIFTED Cartesian pose (hold retreats, carries, and every
        stage-sectors latch). _track_seq keys re-plant on index order
        (inward = target<cur), which coincided with direction-toward-home
        only while home was station 0: under stage-sectors a front(0)→home(1)
        retreat is numerically 'outward' and silently skipped its re-plant —
        the first joint command became SIDE_HOME_JOINTS straight from the
        extended cube-follow/arc-end pose (07-18 review, critical). Classic
        homeward tracks are inward, so replant=True is a no-op there."""
        seq = self._track_seq(cur, target)
        if replant and (not seq or seq[0] != cur):
            seq.insert(0, cur)
        return TrackMotion(arm=arm, cur=cur, target=target,
                           seq=seq, hop_t0=now)

    def _step_toward(self, tm: TrackMotion, jpos, now: float) -> tuple[dict | None, str]:
        """Moved to auto_operator.staging.step_toward (S6)."""
        return ao_staging.step_toward(self, tm, jpos, now)

    def _retarget_track(self, tm: TrackMotion, target: int, now: float) -> None:
        """Change a settled track's goal (direction reversal / new sector). Only
        legal at a settled station — the sequence is rebuilt from tm['cur']."""
        tm["target"] = target
        tm["seq"] = self._track_seq(tm["cur"], target)
        tm["dwell_until"] = None
        tm["hop_t0"] = now
        tm["timed_out"] = False

    # ---------------- dynamic-tracking reachability ----------------
    @staticmethod
    def _target_safe(pos) -> bool:
        """Safety envelope for ANY point the arms may be sent to — full
        incident context lives on auto_operator.safety.target_safe."""
        return ao_safety.target_safe(pos, TARGET_MIN_RADIUS_M,
                                     TARGET_MAX_RADIUS_M,
                                     TARGET_Z_MIN_M, TARGET_Z_MAX_M)
    def _dyn_reachable(self, pos: np.ndarray, arm: str, h: dict) -> bool:
        """Moved to auto_operator.mission.dyn_reachable (S6)."""
        return ao_mission.dyn_reachable(self, pos, arm, h)

    # ---------------- legs / stop ----------------
    def _legs_and_gate(self, now: float) -> tuple[float, float, bool]:
        """Moved to auto_operator.mission.legs_and_gate (S6)."""
        return ao_mission.legs_and_gate(self, now)

    def _ramp(self, current: float, desired: float) -> float:
        step = ACCEL_LIMIT * DT
        return float(current + np.clip(desired - current, -step, step))

    # ---------------- mission phases (one handler each, dispatched by tick) --------
    def _tick_search(self, now: float) -> None:
        """Moved to auto_operator.mission.tick_search (S6)."""
        return ao_mission.tick_search(self, now)

    def _tick_go(self, now: float) -> tuple[float, float]:
        """Moved to auto_operator.mission.tick_go (S6)."""
        return ao_mission.tick_go(self, now)

    def _latch_reach_target(self, now: float) -> None:
        """Moved to auto_operator.mission.latch_reach_target (S6)."""
        return ao_mission.latch_reach_target(self, now)

    def _staged_joint_payload(self, arm: str, posture, frozen: list) -> dict:
        """Moved to auto_operator.staging.staged_joint_payload (S6)."""
        return ao_staging.staged_joint_payload(self, arm, posture, frozen)

    def _gentle_approach(self, stream: np.ndarray | None, tgt: np.ndarray,
                         actual) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Ramp a PUBLISHED Cartesian point from the hand's current position toward
        `tgt` at STAGE_REACH_EE_RATE (the audited-basin cube approach — handing the
        IK the whole staged→cube jump in one packet read as 'too fast'). Returns
        (new_stream, pub_point); pub_point is None when there is no hand position
        yet (caller must stay SILENT rather than seed the ramp at the goal)."""
        if stream is None:
            if actual is None:
                return None, None
            stream = np.asarray(actual, dtype=float)
        delta = tgt - stream
        n = float(np.linalg.norm(delta))
        step = STAGE_REACH_EE_RATE * DT
        stream = tgt.copy() if n <= step else stream + delta * (step / n)
        return stream, stream

    def _tick_reach(self, telemetry: dict, jpos, now: float) -> dict | None:
        """Moved to auto_operator.mission.tick_reach (S6)."""
        return ao_mission.tick_reach(self, telemetry, jpos, now)

    def _maybe_send_grab(self, arm: str, h: dict, cube, telemetry: dict,
                         now: float) -> bool:
        """Moved to auto_operator.arms.maybe_send_grab (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.maybe_send_grab(self, arm, h, cube, telemetry, now)

    def _queue_ee_action(self, action: dict) -> int:
        """Assign one stable ID and queue a logical EE action burst."""
        return ao_arbitration.enqueue_ee_action(self, action)

    def _live_retrack(self, cube, arm: str, sector: str, current,
                      now: float) -> np.ndarray | None:
        """Moved to auto_operator.arms.live_retrack (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.live_retrack(self, cube, arm, sector, current, now)

    def _register_hold(self, arm: str, cube, target, sector: str, bias,
                       now: float, restream=None) -> None:
        """Moved to auto_operator.arms.register_hold (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.register_hold(self, arm, cube, target, sector, bias, now, restream)

    def _try_latch_secondary(self, now: float) -> None:
        """Moved to auto_operator.secondary.try_latch_secondary (S6)."""
        return ao_secondary.try_latch_secondary(self, now)

    def _tick_secondary(self, telemetry: dict, now: float) -> None:
        """Moved to auto_operator.secondary.tick_secondary (S6)."""
        return ao_secondary.tick_secondary(self, telemetry, now)

    def _gate_eval(self, pos) -> tuple[float, str | None, bool]:
        """Moved to auto_operator.mission.gate_eval (S6)."""
        return ao_mission.gate_eval(self, pos)

    def _reachable_now(self, pos, arm: str) -> bool:
        """Moved to auto_operator.mission.reachable_now (S6)."""
        return ao_mission.reachable_now(self, pos, arm)

    @staticmethod
    def _tilt_deg(telemetry: dict) -> float | None:
        """Torso tilt off vertical [deg] (auto_operator.safety.tilt_deg)."""
        return ao_safety.tilt_deg(telemetry)
    @staticmethod
    def _ee_alive(telemetry: dict) -> bool | None:
        """EE-chain liveness from telemetry (real_env liveness patch): True/False,
        or None when the field is absent (an un-patched real_env — caller must
        degrade loudly, never assume alive)."""
        side = (telemetry.get("ee") or {}).get("left") or {}
        return side.get("ee_alive")

    def _enter_carried(self, h: dict, arm: str, ee_actual) -> None:
        """Moved to auto_operator.arms.enter_carried (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.enter_carried(self, h, arm, ee_actual)

    def _grasp_dropped(self, h: dict, cube, arm: str, telemetry: dict,
                       now: float) -> bool:
        """Moved to auto_operator.arms.grasp_dropped (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.grasp_dropped(self, h, cube, arm, telemetry, now)

    def _cube_servable(self, cube, arm: str, now: float) -> tuple[bool, int | None]:
        """Moved to auto_operator.arms.cube_servable (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.cube_servable(self, cube, arm, now)

    def _held_update(self, now: float, telemetry: dict, jpos) -> dict | None:
        """Moved to auto_operator.arms.held_update (S5) — incident
        context and SAFE-IDs live there."""
        return ao_arms.held_update(self, now, telemetry, jpos)

    def _tick_hold(self, telemetry: dict, jpos, now: float) -> dict | None:
        """Moved to auto_operator.mission.tick_hold (S6)."""
        return ao_mission.tick_hold(self, telemetry, jpos, now)

    def _tick_park(self, now: float) -> dict | None:
        """Moved to auto_operator.mission.tick_park (S6)."""
        return ao_mission.tick_park(self, now)

    def _motor_names(self) -> list | None:
        """Motor names in telemetry order (lazy; None off-robot — the import
        pulls the CAN bindings, which may not exist on a workstation)."""
        if not hasattr(self, "_motor_names_cache"):
            try:
                from humanoid_config import motor_setup_dict
                self._motor_names_cache = list(motor_setup_dict.keys())
            except Exception:
                self._motor_names_cache = None
        return self._motor_names_cache

    def _temp_watch(self, telemetry: dict, now: float) -> None:
        """Thermal early-warning (07-18 OVER CURRENT incident): PRINT-ONLY —
        two absolute tiers (60/70 C vs the ~80 C Robstride firmware trip)
        plus a rate-of-rise STALL signature (avg >0.5 C/s over 5 s): a
        jammed joint heats far faster than any posture load, so the rate
        alarm fires minutes before the absolute tiers. No behavior change:
        missing/short telemetry is silently ignored (replay-identical)."""
        temps = telemetry.get("motor_temp")
        if not temps:
            return
        t = np.asarray(temps, dtype=float)
        if t.size == 0 or not np.all(np.isfinite(t)):
            return
        hist = getattr(self, "_temp_hist", None)
        if hist is None:
            hist = self._temp_hist = []
        hist.append((now, t))
        while len(hist) > 2 and now - hist[1][0] >= MOTOR_TEMP_RATE_WIN_S:
            hist.pop(0)                      # keep one sample ≥ the window old
        names = self._motor_names()
        def _nm(i):
            return names[i] if names and i < len(names) else f"motor[{i}]"
        t0, told = hist[0]
        dt = now - t0
        if dt >= MOTOR_TEMP_RATE_WIN_S and t.shape == told.shape:
            rate = (t - told) / dt
            ri = int(np.argmax(rate))
            if float(rate[ri]) >= MOTOR_TEMP_RATE_C_S \
                    and now - getattr(self, "_temp_rate_warn_t", -1e9) > 10.0:
                self._temp_rate_warn_t = now
                print(f"[temp] STALL SIGNATURE: {_nm(ri)} winding rising "
                      f"{rate[ri]:.1f} C/s (now {t[ri]:.0f} C) — a jammed or "
                      f"hard-loaded joint; STOP and inspect before the "
                      f"firmware latches OVER CURRENT")
        hi = int(np.argmax(t))
        if float(t[hi]) >= MOTOR_TEMP_HOT_C:
            if now - getattr(self, "_temp_hot_warn_t", -1e9) > 5.0:
                self._temp_hot_warn_t = now
                print(f"[temp] HOT: {_nm(hi)} at {t[hi]:.0f} C — firmware "
                      f"trips ~80 C: REST this arm now (Ctrl+C or remove the "
                      f"cube; resume below {MOTOR_TEMP_WARN_C - 5:.0f} C)")
        elif float(t[hi]) >= MOTOR_TEMP_WARN_C \
                and now - getattr(self, "_temp_warn_t", -1e9) > 10.0:
            self._temp_warn_t = now
            warm = {_nm(i): round(float(v), 1)
                    for i, v in enumerate(t) if v >= MOTOR_TEMP_WARN_C}
            print(f"[temp] warm windings {warm} (warn {MOTOR_TEMP_WARN_C:.0f} C"
                  f" / rest {MOTOR_TEMP_HOT_C:.0f} C / firmware trip ~80 C)")

    def _arm_near_front(self, arm: str, jpos) -> bool:
        """True iff `arm` VERIFIABLY sits within FRONT_HOME_TOL_RAD of a KNOWN
        home posture — the front station, the power-on default, or (side-home
        mode) the side-home posture. Fails CLOSED on missing/short/non-finite
        telemetry — NaN > x is False, so a plain comparison would pass a
        garbage posture as 'home' (07-15 audit, fail-open class)."""
        if jpos is None or len(jpos) < 31:
            return False
        sl = slice(13, 20) if arm == "left" else slice(20, 27)
        seg = np.asarray(jpos[sl], dtype=float)
        if not np.all(np.isfinite(seg)):
            return False
        homes = [STAGE_JOINTS["front"], POWERON_JOINTS,
                 POWERON_JOINTS_LEGACY_0730, POWERON_JOINTS_V83]
        if self.side_home:
            homes.append(SIDE_HOME_JOINTS)
        return any(float(np.max(np.abs(seg - mirror_arm(h, arm))))
                   <= FRONT_HOME_TOL_RAD for h in homes)

    def _idle_arms_home(self, jpos) -> bool:
        """True iff every arm that is NOT actively holding a cube sits near a
        KNOWN home posture OR is mid-way through an ACTIVE home glide (07-18:
        the glide traverses the no-man's-land between the reference postures —
        suppressing the keep-alive there froze the glide 8 cm out). Publishing
        a Cartesian REST/keep-alive is only safe from a verified home or along
        a glide that STARTED at one — an arm stuck in side/rear (e.g. a
        timed-out retreat, stream None) is still never pulled home in one
        packet."""
        return all((self._arm_near_front(arm, jpos)
                    or self._home_stream[arm] is not None)
                   for arm in ("left", "right") if arm not in self._held
                   and not (self._sec is not None and self._sec["arm"] == arm)
                   and self._arm_tasks.get(arm) is None)

    def _home_glide(self, arm: str) -> list | None:
        """Gentle home-target ramp (auto_operator.arbitration.home_glide)."""
        return ao_arbitration.home_glide(self, arm, STAGE_REACH_EE_RATE)

    def _arm_packet(self, reach_pos: np.ndarray | None = None,
                    reach_positions: dict[str, np.ndarray] | None = None) \
            -> dict | None:
        """Per-arm slot assembly (auto_operator.arbitration.build_arm_packet —
        the slot-ownership and stranded-arm rules live there with their
        incident context)."""
        return ao_arbitration.build_arm_packet(
            self, reach_pos, reach_positions=reach_positions)

    def _grasp_intervention_packet(self, jpos) -> tuple[list, dict]:
        """Freeze navigation/arms and leave the gripper command unchanged."""
        self._vx = 0.0
        self._wz = 0.0
        packet: dict = {}
        hold = ao_arbitration.intervention_hold_targets(self, jpos)
        if hold is not None:
            packet["arm_targets"] = hold
            self._last_arm_targets = hold
            self._arm_engaged = True
        if self._gaze_seeded:
            # Hold the current gaze; do not advance search/tracking state.
            packet["gaze_targets"] = [float(v) for v in self.gaze]
        return [0.0, 0.0, 0.0], packet

    # ---------------- main tick ----------------
    def tick(self, detections: dict, telemetry: dict, now: float) -> tuple[list | None, dict]:
        """→ (nav_cmd [vx,vy,wz] | None, packet_9874 {gaze_targets, arm_targets?}).

        nav is None in gaze-only mode (no 9873 publish); otherwise a 3-vector.
        """
        jpos = telemetry.get("joint_pos")
        jpos = np.asarray(jpos, dtype=float) if jpos is not None else None
        self._jpos = jpos                          # per-arm REST gate reads these
        # JOURNEY (07-21): measured base velocity for the REAL stop-stillness
        # gate (the command ramp reaching zero is not the robot being still).
        self._base_vel = telemetry.get("base_lin_vel")
        self._base_ang = telemetry.get("base_ang_vel")
        ee_t = telemetry.get("ee") or {}
        self._ee_act = {a: (ee_t.get(a) or {}).get("pos_actual") for a in ("left", "right")}
        # SAFETY PATCH (07-25): honour real_env's motor
        # dropout watchdog. When the robot side reports an arm latched into its
        # damped hold ("arm_fault", 07-24 incident class: a shoulder pair went
        # off-bus and came back torqueless), this mission must STOP — the
        # latched arm cannot execute anything, navigation assumes both arms
        # track their sectors, and the peer arm's clearance certificates assume
        # the same. Reuses the existing fail-closed intervention latch verbatim
        # (nav zeroed, both arms pinned, gaze held, grippers untouched).
        _af = telemetry.get("arm_fault")
        if _af and any(_af) and not self.gaze_only \
                and not self._grasp_safety_latched:
            _sides = [s for s, f in zip(("left", "right"), _af) if f]
            self._vx = 0.0
            self._wz = 0.0
            self._grasp_safety_latched = True
            self._grasp_safety_arm = _sides[0]
            self._grasp_safety_reason = (
                f"real_env watchdog latched {'/'.join(_sides)} arm "
                f"(motor dropout — damped hold)")
            self._grasp_intervention_required = True
            self._grasp_intervention_reason = self._grasp_safety_reason
            self._intervention_arm_targets = None
            print(f"[operator] INTERVENTION REQUIRED — {self._grasp_safety_reason}. "
                  "Mission FROZEN: navigation zeroed, both arms held. Inspect "
                  "the arm/harness, then restart real_env and the operator.")
        if not self._gaze_seeded and jpos is not None and len(jpos) >= 31 \
                and bool(np.all(np.isfinite(jpos[27:31]))):
            # Seed the gaze reference from the ACTUAL gimbal encoders before the
            # first publish: zeros-init + a parked gimbal = a multi-radian slam in
            # one step (07-15 audit). Until seeded, no gaze_targets are published
            # (real_env holds its last clamped reference — benign).
            self.gaze = np.asarray(jpos[27:31], dtype=float).copy()
            self._gaze_seeded = True
        if not self._startup_posture_ok and jpos is not None and len(jpos) >= 31 \
                and bool(np.all(np.isfinite(jpos[13:27]))) and not self.gaze_only:
            # (non-finite jpos skips this block entirely → _startup_posture_ok stays
            #  False → the fail-closed branch below: NaN > x is False, so without the
            #  isfinite gate a NaN posture would PASS the check — fail-open.)
            tilt = self._tilt_deg(telemetry)
            if tilt is not None and tilt > TILT_REFUSE_DEG:
                raise SystemExit(
                    f"[operator] STARTUP REFUSED: torso is {tilt:.0f} deg off "
                    f"vertical (limit {TILT_REFUSE_DEG:.0f}). A tilted base "
                    f"corrupts EVERY body-frame target — table cubes read "
                    f"chest-height, the learned gate goes OOD, and the IK "
                    f"diverges from our commands (07-16/18 incident class). "
                    f"Fix the HANG/STANCE (RULE #0: legs straight, torso "
                    f"upright) and restart.")
            if tilt is not None and tilt > TILT_WARN_DEG:
                print(f"[tilt] WARNING: torso {tilt:.1f} deg off vertical at "
                      f"startup — geometry margins are thin; verify the first "
                      f"[reach] latched z is NEGATIVE")
            # TWO known-safe home postures (07-17, raised stations): the power-on
            # default (cold start / fresh real_env prepare) and the RAISED front
            # station (arms parked there after any mission). Accept EITHER within
            # tolerance — gravity sag (~5 deg at 0.8 torque) plus the deliberate
            # 9.2 deg power-on->new-front offset would otherwise eat the margin.
            accepted = [np.asarray(STAGE_JOINTS["front"]),
                        np.asarray(POWERON_JOINTS),
                        np.asarray(POWERON_JOINTS_LEGACY_0730),
                        np.asarray(POWERON_JOINTS_V83)]
            if self.side_home:
                # side-home restart: arms legitimately parked at the SIDE home
                # from the previous run (07-18 gap: the gate refused it and
                # forced a needless power-cycle; mirrors _arm_near_front)
                accepted.append(np.asarray(SIDE_HOME_JOINTS))
            bad = {arm: min(float(np.max(np.abs(jpos[sl] - mirror_arm(h, arm))))
                            for h in accepted)
                   for arm, sl in (("left", slice(13, 20)),
                                   ("right", slice(20, 27)))}
            bad = {a: e for a, e in bad.items() if e > FRONT_HOME_TOL_RAD}
            if bad:
                raise SystemExit(
                    f"[operator] STARTUP REFUSED: arm(s) not at the power-on pose "
                    f"{ {a: f'{np.degrees(e):.0f}deg off' for a, e in bad.items()} } — "
                    f"reaching from an unknown posture can cross configuration basins "
                    f"(no staircase would protect it). Home the arm first: power-cycle, "
                    f"let the failsafe crawl finish, or run humanoid_stage_walk_test.py.")
            self._startup_posture_ok = True
        if now - getattr(self, "_tilt_diag_t", -1e9) > 5.0:
            # continuous tilt watch: the hang/stance can degrade MID-mission
            self._tilt_diag_t = now
            _tilt = self._tilt_deg(telemetry)
            if _tilt is not None and _tilt > TILT_WARN_DEG:
                print(f"[tilt] torso {_tilt:.1f} deg off vertical — body-frame "
                      f"geometry suspect (latched targets may be corrupted)")
        self._temp_watch(telemetry, now)
        self._ingest(detections, now)

        if self.gaze_only:
            # cameras-track-only: never walk, never reach → nav=None so main() does
            # not even publish 9873, and the packet carries ONLY gaze_targets. Gaze
            # itself is the same per-camera claim/track/scan logic as the mission.
            if not self._gaze_seeded:
                return None, {}                      # no encoders yet → publish nothing
            self.gaze = self._slew_gaze(self._eye(jpos, now))
            return None, {"gaze_targets": [float(v) for v in self.gaze]}

        arm_packet: dict = {}
        vx_des = wz_des = 0.0

        # FAIL CLOSED: a full mission may not command anything until the startup
        # posture check has actually PASSED (it needs telemetry to run). No
        # telemetry / an arm parked off-home ⇒ publish gaze only, nav zero, NO arm
        # packet — the robot's arm-silence failsafe holds/crawls. (An operator
        # relaunched against a wrong --robot-ip, or before the arm homed, must
        # never drive a blind Cartesian reach.)
        if not self._startup_posture_ok:
            self._vx = self._ramp(self._vx, 0.0)
            self._wz = self._ramp(self._wz, 0.0)
            if not self._gaze_seeded:
                return [self._vx, 0.0, self._wz], {}
            self.gaze = self._slew_gaze(self._eye(jpos, now))
            return [self._vx, 0.0, self._wz], {"gaze_targets": [float(v) for v in self.gaze]}

        if self.grasp and not self._hands_opened:
            # startup: ZERO both grippers (low-torque pinch → measured close →
            # auto-open). Calibration lives in the EE-service process and dies
            # with it; the service runs WITHOUT --gui (the GUI embeds its own
            # commander and steals the single request socket from real_env) —
            # so the operator re-zeros every mission. GATED on the telemetry
            # ee_alive signal (07-16 audit: fire-and-forget zeros were silently
            # swallowed by a dead chain and the mission ran on fake pinches).
            # PRECONDITION: grippers EMPTY (a held object would calibrate its
            # width as "closed" — sides reporting grasp_detected are skipped).
            alive = self._ee_alive(telemetry)
            if self._ee_wait_t0 is None:
                self._ee_wait_t0 = now
            if ao_arms.startup_grasp_guard(self, telemetry):
                self._hands_opened = True
            elif alive:
                for side in ("left", "right"):
                    self._queue_ee_action(
                        {"side": side, "command": "zero_gripper"})
                self._zeros_t = now
                self._hands_opened = True
                print("[grasp] EE chain alive — zeroing grippers "
                      "(slow pinch → open)")
            elif (now - self._ee_wait_t0) > EE_ALIVE_WAIT_S:
                self._hands_opened = True
                if alive is None:
                    self._grasp_safety_latched = True
                    self._grasp_safety_arm = None
                    self._grasp_safety_reason = \
                        "startup telemetry has no ee_alive field"
                    self._grasp_intervention_required = True
                    self._grasp_intervention_reason = \
                        self._grasp_safety_reason
                    self._intervention_arm_targets = None
                    print("[grasp] INTERVENTION REQUIRED — telemetry has no "
                          "ee_alive field after startup wait. BLIND ZEROING "
                          "REFUSED; navigation and both arms PAUSED pending "
                          "human verification.")
                else:
                    self._ee_chain_dead = True
                    print("[grasp] ============================================")
                    print("[grasp] EE CHAIN DEAD: ee_alive stayed False for "
                          f"{EE_ALIVE_WAIT_S:.0f}s — service down, --ee-service "
                          "off, or socket conflict. GRASPING DISABLED for this "
                          "mission; continuing as classic reach-and-hold.")
                    print("[grasp] ============================================")

        if self._grasp_safety_latched:
            return self._grasp_intervention_packet(jpos)

        # Per-arm hold upkeep runs EVERY mission tick regardless of phase — a
        # holding arm keeps following/retracting while the state machine is off
        # chasing the other cube. Returns a joint payload while an arm walks its
        # reverse staircase (one staircase at a time, REACH staging has priority).
        if self.independent_arms:
            # held_update mutates lift/park/re-extension ramps.  Check that the
            # complete two-arm Cartesian packet is constructible first; if the
            # peer slot has no finite encoder/home fallback, those ramps must
            # remain exactly where the robot last received them.
            self._ind_hold_cartesian_active = \
                ao_independent.hold_cartesian_motion_active(self)
            self._ind_cartesian_grant = \
                ao_independent.cartesian_packet_ready(self)
        ret_payload = self._held_update(now, telemetry, jpos) if self._held else None

        # held_update may have just latched a false/None/timeout/drop suspicion.
        # Stop before either scheduler can advance or attach another EE action.
        if self._grasp_safety_latched:
            return self._grasp_intervention_packet(jpos)

        if self.independent_arms:
            # Two symmetric ArmTask FSMs. Both compute/advance in this tick;
            # Cartesian targets share the packet, while a staged task receives
            # the explicit global joint token without resetting its peer.
            vx_des, wz_des, independent_payload = \
                ao_independent.tick_independent(self, telemetry, jpos, now)
            if independent_payload is not None:
                arm_packet["arm_targets"] = independent_payload
        else:
            # Legacy primary + optional secondary compatibility path.
            if self.dual_parallel:
                if self._sec is None and self.phase == Phase.REACH:
                    self._try_latch_secondary(now)
                if self._sec is not None:
                    self._tick_secondary(telemetry, now)

            if self.phase == Phase.SEARCH:
                self._tick_search(now)
            elif self.phase == Phase.GO:
                vx_des, wz_des = self._tick_go(now)
            elif self.phase == Phase.REACH:
                reach_payload = self._tick_reach(telemetry, jpos, now)
                if reach_payload is not None:
                    arm_packet["arm_targets"] = reach_payload
            elif self.phase == Phase.HOLD:
                hold_payload = self._tick_hold(telemetry, jpos, now)
                if hold_payload is not None:
                    arm_packet["arm_targets"] = hold_payload
            elif self.phase == Phase.PARK:
                park_payload = self._tick_park(now)
                if park_payload is not None:
                    arm_packet["arm_targets"] = park_payload

        # Packet arbitration (auto_operator.arbitration.arbitrate): priority is
        # ret_payload > staircase Cartesian suppression > phase packet >
        # keep-alive > silence. SAFE-ARBITRATION-*.
        ao_arbitration.arbitrate(self, arm_packet, ret_payload, jpos)
        if arm_packet.get("arm_targets") is not None:
            self._arm_engaged = True
            self._last_arm_targets = copy.deepcopy(arm_packet["arm_targets"])

        # Gripper command burst (auto_operator.arbitration.attach_ee_action):
        # lossy-channel repeats + queue rotation. SAFE-GRIPPER-*/INC-3.
        ao_arbitration.attach_ee_action(self, arm_packet, EE_ACTION_REPEAT_TICKS)

        # eye runs in every phase except DONE (then reference holds robot-side)
        if self.phase != Phase.DONE and self._gaze_seeded:
            self.gaze = self._slew_gaze(self._eye(jpos, now))
            arm_packet["gaze_targets"] = [float(v) for v in self.gaze]

        # final safety clamp + ramp
        self._vx = self._ramp(self._vx, float(np.clip(vx_des, -CRUISE_VX, CRUISE_VX)))
        self._wz = self._ramp(self._wz, float(np.clip(wz_des, -WZ_MAX, WZ_MAX)))
        return [self._vx, 0.0, self._wz], arm_packet


def main():
    import tyro

    @dataclasses.dataclass
    class Args:
        robot_ip: str = _site.ROBOT_IP
        cubes: list[str] = dataclasses.field(
            default_factory=lambda: ["grasp_cube_60mm_a", "grasp_cube_60mm_b"])
        detection_url: str = DETECTION_IPC_URL
        use_gate: bool = True     # learned reachability stop; False → geometric floor only
        gaze_only: bool = False   # cameras track the cube(s) only — no walk, no reach, no 9873
        hold_after_reach: bool = True  # final cube: HOLD hand on it + keep staring until Ctrl+C
        """(--no-hold-after-reach restores park-and-DONE on the last cube)"""
        visual_servo: bool = True  # gripper-tag closed loop: estimate the camera↔FK hand
        """bias from the gripper base tags and subtract it from the reach target; needs
        monitor --bodies gripper_L_base [gripper_R_base]. (--no-visual-servo = open loop)"""
        reach_standoff: float = REACH_STANDOFF_M  # pull the reach point this far back
        """toward the robot [m]; 0 = drive the EE site (jaw center) onto the cube centre
        itself — the palm may push the cube"""
        aim_up: float = AIM_UP_M  # lift the reach point this far above the cube centre
        """[m] — the jaw lands low on hardware and can bite the cube's stand; aiming
        high centres the closing jaw on the cube (0 = aim at the centre itself)"""
        dynamic_track: bool = True  # HOLD follows a moving cube; retracts to rest when
        """the cube is out of reach or unseen >1s, re-reaches when it returns.
        (--no-dynamic-track = classic frozen-target hold)"""
        walk: bool = True  # publish real walk velocities during GO.
        """--no-walk hard-zeroes every nav command (STANDING missions: the robot
        balances in place and must never step; cubes must sit inside the arrival
        gate's envelope or GO waits forever). The state machine is unchanged —
        identical to the hanging-bench behavior where legs ignore nav anyway."""
        grasp: bool = True  # close the gripper on each held cube (load-supervised
        """hand_grab) and CARRY it home to front — first grasped walks home first;
        arms park at REST with their cubes (terminal). Needs real_env --ee-service
        + humanoid_end_effector_service.py running, and dynamic tracking (default).
        Without the EE service each grab times out after 20 s and the mission
        degrades to the classic hold. (--no-grasp = reach-and-hold only)"""
        chest_home: bool = False  # 07-24 (standing session): keep the arms at the
        """power-on CHEST tuck the whole mission — NO side-home glide at startup —
        and reach every cube DIRECTLY from that chest pose. FORCES --no-stage-sectors
        (a joint staircase replants from the SIDE station, which is the wrong start
        posture when the arm is parked at the chest). Place cubes where a direct leg
        can serve them. Needs the side-home independent profile (defaults); intended
        with --no-walk."""
        side_home: bool = True  # SIDE-HOME mission profile (07-18): after startup
        """the arms glide to the SIDE home ([0, ±0.55, 0] — IK-validated, the raw
        side station r=0.652 is unreachable under the neutral orientation), EVERY
        cube is a direct Cartesian reach (no front/side/rear staging, no joint
        staircases → both arms fully independent for ANY cube placement), and a
        grasped cube lifts 5 cm then ramps slowly back to the side home.
        (--no-side-home = classic three-sector staging with the front chest home)"""
        stage_sectors: bool = True  # (side-home only) route front-HIGH (z > 0) and ALL
        """rear reaches through their audited STAGE_JOINTS station first: the settled
        station re-seeds the robot-side IK in the correct elbow branch — a direct
        stream to a chest-height rear target swings the whole arm backwards instead
        (07-18 hardware + offline proof; S-A audit: rear real target 0.6 mm from the
        station vs 42.4 mm / -111° shoulder sweep direct). side and LOW-front legs
        stay direct Cartesian (dual-parallel unaffected). Carries from staged legs
        walk the reverse staircase home. (--no-stage-sectors = pure direct side-home)"""
        sync_reach: bool = False  # video-demo sync: the first-ready arm waits at the
        # latch gate until the other cube's arm is ready too, then BOTH reach in
        # the same tick. Standing mission only; needs both cubes placed.
        journey: bool = False  # 07-21 far-cube mission: survey BOTH cubes standing,
        # then walk to the nearer one (forward/backward, never turning around),
        # grasp, walk the other way to the second, grasp, stand. Needs walking
        # (drop --no-walk) and the independent scheduler. Free arms tuck at the
        # power-on pose while walking, return to side-home on arrival.
        independent_arms: bool = True  # symmetric left/right task FSMs: each side owns
        """its target selection, reachability debounce, staging/reach, visual servo and
        handoff into grasp/lift/carry. Cartesian work runs concurrently; the current
        robot wire still serializes joint staircases. --no-independent-arms restores
        the legacy primary + optional secondary coordinator."""
        dual_parallel: bool = True  # LEGACY coordinator only (ignored while
        """--independent-arms is active): front+front both arms reach at once — while
        one arm flies its Cartesian leg toward a FRONT cube, the other arm
        latches a second FRONT cube and approaches in parallel; whichever arrives
        first starts grabbing immediately. Joint staircases (side/rear legs,
        carries) remain strictly serialized — the modal-wire contract is
        untouched. (--no-dual-parallel = classic one-reach-at-a-time)"""

    args = tyro.cli(Args)
    independent_enabled = args.independent_arms and args.hold_after_reach \
        and not args.gaze_only
    if args.independent_arms and not args.hold_after_reach:
        print("[operator] --independent-arms requires hold machinery; "
              "falling back to the legacy park-and-DONE coordinator")
    gate = None
    if args.use_gate and not args.gaze_only:  # gaze-only never walks → no reachability gate
        import os
        # The gate is the ONLY consumer of upstream's tree on sys.path (08-22:
        # moved here from import time — see the note beside _LEV_ROOT above).
        sys.path.insert(0, str(_LEV_ROOT))
        sys.path.insert(0, str(_LEV_ROOT / "mj_envs"))
        os.chdir(str(_LEV_ROOT / "mj_envs"))  # gate loads scorers from relative cache path
        from tasks.visual_manipulation.moving_policy import ReachabilityGate
        gate = ReachabilityGate(device="cpu")

    op = AutoOperator(args.cubes, gate=gate, gaze_only=args.gaze_only,
                      hold_after_reach=args.hold_after_reach, visual_servo=args.visual_servo,
                      reach_standoff=args.reach_standoff, dynamic_track=args.dynamic_track,
                      grasp=args.grasp and not args.gaze_only and args.dynamic_track,
                      aim_up=args.aim_up,
                      dual_parallel=args.dual_parallel
                      and not independent_enabled and not args.gaze_only,
                      side_home=args.side_home and not args.gaze_only,
                      stage_sectors=args.stage_sectors and args.side_home
                      and not args.gaze_only,
                      independent_arms=independent_enabled,
                      journey=args.journey and not args.gaze_only,
                      sync_reach=args.sync_reach and not args.gaze_only,
                      chest_home=args.chest_home and not args.gaze_only)
    if op.journey and not args.walk:
        print("[journey] WARNING: --journey with --no-walk — the robot cannot "
              "reach far cubes standing still; drop --no-walk for this mission")
    det_sub = NNGSubscriber(args.detection_url); det_sub.start()
    tel_sub = NNGSubscriber(f"tcp://{args.robot_ip}:9870"); tel_sub.start()
    nav_pub = None if args.gaze_only else NNGPublisher("tcp://*:9873")
    arm_pub = NNGPublisher("tcp://*:9874")
    mode = "GAZE-ONLY (no walk/reach)" if args.gaze_only else \
        f"full mission, gate={'learned' if gate else 'geometric'}"
    print(f"[operator] mission cubes={args.cubes}  mode={mode}")
    if not args.gaze_only:
        _h = op.rest_ee["left"]
        _where = "FRONT" if abs(_h[0]) >= abs(_h[1]) else "SIDE"
        _hs = f"[{_h[0]:.2f}, ±{abs(_h[1]):.2f}, {_h[2]:+.2f}]"
        print("[operator] profile: "
              + ((f"{_where}-HOME + STAGE-SECTORS (arms live at {_hs}; "
                  "side/low-front direct, high-front & rear via their "
                  "stations; staged carries walk the staircase home)"
                  if op.stage_sectors else
                  f"{_where}-HOME (arms live at {_hs}; direct Cartesian "
                  "reaches, no sector staging; carry returns home)")
                 if op.side_home else
                 "CLASSIC three-sector staging (front chest home)"))
    if not args.walk and not args.gaze_only:
        print("[operator] WALK DISABLED (--no-walk): nav is hard-zeroed — standing "
              "mission; place cubes inside the arrival envelope")
    if op.chest_home:
        print("[operator] CHEST-HOME armed: arms HOLD the power-on chest tuck "
              "(no side-home glide); every cube is reached DIRECTLY from the "
              "chest. Sector staging FORCED OFF — no joint staircases at all.")
    elif args.chest_home and not args.gaze_only:
        print("[operator] NOTE: --chest-home IGNORED — it needs the side-home "
              "independent profile (drop --no-side-home / --no-independent-arms).")

    last_det_id = -1
    last_det_arrival = -1e9
    det_targets: dict = {}
    last_tel_id = -1
    last_tel_arrival = -1e9
    last_phase = None
    last_status = -1e9
    try:
        while True:
            t0 = time.monotonic()
            if det_sub.data_id != last_det_id and det_sub.data is not None:
                last_det_id = det_sub.data_id
                last_det_arrival = time.monotonic()
                det_targets = det_sub.data.get("targets") or {}
            elif time.monotonic() - last_det_arrival > 0.5:
                # Monitor frozen/dead: retaining the last dict would keep re-stamping
                # cube.last_seen and the robot would walk forever on dead perception.
                det_targets = {}
            if tel_sub.data_id != last_tel_id and tel_sub.data is not None:
                last_tel_id = tel_sub.data_id
                last_tel_arrival = time.monotonic()
            # Telemetry freshness gate (07-15 audit): tel_sub.data holds the LAST
            # packet forever, so a frozen/restarted 9870 stream would keep feeding
            # yesterday's encoders to every posture guard (fail-open). Stale ⇒
            # tick() sees no telemetry at all and every guard fails CLOSED.
            telemetry = (tel_sub.data or {}) \
                if (time.monotonic() - last_tel_arrival) <= 0.5 else {}
            now = time.monotonic()
            nav, arm_packet = op.tick(det_targets, telemetry, now)
            if nav is not None and nav_pub is not None:
                if not args.walk:
                    nav = [0.0, 0.0, 0.0]   # standing mission: never step (see Args.walk)
                nav_pub.publish({"nav_cmd": nav})
            if arm_packet:
                arm_pub.publish(arm_packet)
            if args.gaze_only:
                if now - last_status > 2.0:
                    last_status = now
                    st = ", ".join(
                        f"{k}:{'@%.2f,%.2f' % (c.pos[0], c.pos[1]) if c.fresh(now, DET_LOST_RESCAN_S) else 'scan'}"
                        for k, c in op.cubes.items())
                    print(f"[gaze] {st}  gaze={np.round(op.gaze, 2).tolist()}")
            else:
                if op.phase.value != last_phase:
                    last_phase = op.phase.value
                    print(f"[operator] → {last_phase}  cubes={{ {', '.join(f'{k}:{'seen' if c.pos is not None else '—'}' for k, c in op.cubes.items())} }}")
                if op.phase == Phase.DONE:
                    print("[operator] mission complete"); break
            time.sleep(max(0.0, DT - (time.monotonic() - t0)))
    finally:
        if nav_pub is not None:
            nav_pub.publish({"nav_cmd": [0.0, 0.0, 0.0]})
            time.sleep(0.1)
        det_sub.stop(); tel_sub.stop()


if __name__ == "__main__":
    main()
