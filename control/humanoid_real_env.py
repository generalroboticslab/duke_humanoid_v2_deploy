"""Real-robot policy loop — the process that drives the humanoid's motors.

WHAT IT DOES. One control loop at the task's control_freq (50 Hz for the
deployed tasks): read joystick/keyboard, run the exported TorchScript policy
on the assembled observation, then step(): fold the streamed arm / gaze
references into the joint targets, clamp near joint limits, apply the IK
worker's result and the arm integrator, write position + torque feed-forward
to the CAN motors through hardware_bindings (motor + IMU), publish telemetry.
Wire (HumanoidRealEnv's docstring has the tick order and the failsafes):
  9870 PUB telemetry — joint state, IMU, command, `ee` wrist poses, nav_paused
  9873 SUB {"nav_cmd": [vx, vy, wz]} — humanoid_auto_operator.py,
           humanoid_curobo_reach.py (--journey --execute), gampad_gui_control.py
  9874 SUB arm_targets (per-arm joint or Cartesian) + gaze_targets + ee_action —
           humanoid_auto_operator.py / humanoid_curobo_reach.py (bench senders:
           gampad_gui_control.py, humanoid_joint_monkey_hw.py)
Guards: nav silence stops the base, arm silence ramps the arms home, gaze
silence parks the gimbals, the CAN dropout watchdog latches a silent arm into
a damped hold, per-group torque caps and an arm feed-forward ceiling bound
every path. Needs legged_env_v2 (humanoid_site.LEGGED_ENV_ROOT): the deploy
MJCF, mj_envs.utils.ik_mink and torch_math_utils are imported from it.

FLAGS (--help): --task <run dir under <legged_env_v2>/mj_envs/deploy/runs/>,
--torque-limit, --enable-motor true|false|leg|arm|camera|arm_camera, --use-ik,
--grav-comp, --ee-service, --pin-waist-standing, --obs-frame-fix, --record,
--vicon, --profile. Bring-up, identical to docs/OPERATIONS.md T3:

    taskset -c 0-4 python -u humanoid_real_env.py \
      --task <deploy-task> \
      --torque-limit 0.8 --enable-motor true --no-use-ik --grav-comp --ee-service \
      --arm-sender-ip 127.0.0.1 --high-level-controller-ip 127.0.0.1 2>&1 | tee /tmp/realenv.log
"""
import math
import numpy as np
import time
import threading
from datetime import datetime
import re
import yaml
from typing import Literal
import torch
import mujoco

from ipc.publisher import NNGPublisher, NNGSubscriber
from keyboard_gamepad_controller import KeyboardGampadController
from humanoid_utils import CircularBuffer, make_async_logger
from humanoid_record_utils import HumanoidRecorder
from humanoid_gait_utils import GaitPhaseClock
from humanoid_base import (
    HumanoidBase,
    URDF_JOINT_NAMES, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS,
)

_log = make_async_logger("humanoid_real_env")

# Import IK solver (Warp GPU path to match training env behavior)
import sys as _sys
import humanoid_site as _site
from humanoid_site import LEGGED_ENV_ROOT as _LEV_ROOT
if str(_LEV_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_LEV_ROOT))
import mujoco_warp as mjwarp
from mj_envs.utils.ik_mink import BatchedAnalyticalIK
from mj_envs.utils.torch_math_utils import _quat_apply, _quat_conjugate


_DEPLOY_DIR = str(_LEV_ROOT / "mj_envs" / "deploy" / "runs") + "/"

_LEG_PATTERN = r"(waist_joint|.*_hip_.*|.*_knee_.*|.*_ankle_.*)"
# Arm = neither leg nor camera-gimbal. cam_* joints (cam-equipped robot) must be
# excluded or they pollute arm_motor_indices / arm gains / arm integrator.
_ARM_PATTERN = f"^(?!{_LEG_PATTERN}|cam_).*"

# Camera gimbal hold gains (robot-level, mirror CAMERA_MOTORS in legged_env_v2).
# Used when a policy config omits cam actuator entries — e.g. an old non-cam policy
# running on a cam-equipped robot — so the gimbal motors PD-hold their default pose.
# action_scale 0 = no policy delta (cam isn't in such a policy's controlled_joints).
_CAM_HOLD_KP = 5.0
_CAM_HOLD_KD = 0.3

def _get_deploy_paths(task: str) -> tuple[str, str]:
    """Return (policy_path, config_path). Folder = task name."""
    return (
        f"{_DEPLOY_DIR}/{task}/policy_deployed.pt",
        f"{_DEPLOY_DIR}/{task}/env_config.yaml",
    )

# 08-22: the default task is the deployed checkpoint (humanoid_site.DEPLOY_TASK,
# overridable with HUMANOID_DEPLOY_TASK / site_local.py). Until this date it
# named a 2025 run directory that the deploy tree no longer carries, so a bare
# `python humanoid_real_env.py` died on a missing policy_deployed.pt;
# docs/OPERATIONS.md T3 passes --task explicitly either way.
DEFAULT_TASK = _site.DEPLOY_TASK
DEFAULT_POLICY, DEFAULT_CONFIG = _get_deploy_paths(DEFAULT_TASK)



_IMU_STALE_STEPS_MAX = 20   # steps before stale-IMU error (~200 ms @ 100 Hz)
_LIMIT_WARN_STEP_INTERVAL = 100  # steps between per-limit warnings (~1 s @ 100 Hz)
_NAV_CMD_TIMEOUT        = 1.0   # seconds: stop if nav policy goes silent
_NAV_OVERRIDE_THRESHOLD = 0.2   # axis magnitude above stick drift (~0.05)
_ARM_CMD_TIMEOUT        = 0.5   # seconds: ramp arm back to default pose if arm stream goes silent
_ARM_RETURN_RATE        = 0.125   # rad/s: failsafe return-to-default rate on
                                  # arm-silence timeout / explicit clear.
                                  # 0.025 -> 0.125 (user 2026-08-03, "x5 at
                                  # least": a post-grasp Ctrl+C took ~30 s to
                                  # crawl home). Still 8x slower than the
                                  # supervised stream cap; the return is a
                                  # position lerp under the same PD, so
                                  # contact forces stay bounded.
_ARM_DEFAULT_PKT_RATE   = 0.025   # rad/s: rate for joint packets that carry
                                  # NO rate field — kept at the classic crawl
                                  # so rate-less senders (bench/staging tools)
                                  # keep their exact old speed; the x5 above
                                  # applies ONLY to the failsafe return
_GAZE_CMD_TIMEOUT       = 0.5     # seconds: park the gimbals at zero if the
                                  # gaze stream goes silent. Its OWN timer, not
                                  # the arm one: gaze-only keepalives stream all
                                  # through a walk (which is exactly why they are
                                  # excluded from the arm timer), and arm-only
                                  # packets never happen — the two streams stop
                                  # together, but only a gaze packet may hold the
                                  # gimbals off their failsafe.
_GAZE_RETURN_RATE       = 0.75    # rad/s: failsafe return-to-zero rate for the
                                  # cam joints (user 2026-08-08: "the 2 cameras
                                  # need to be reset to the origin as well when I
                                  # control C"). Half the operator's supervised
                                  # GAZE_SLEW_RATE (1.5): a scan can leave yaw at
                                  # ±π, which unwinds in ~4 s. Before this the
                                  # reference was latched forever with no
                                  # timeout, so a killed operator left both eyes
                                  # frozen mid-sweep, staring at a cube that is
                                  # no longer there.
_ARM_KI                 = 1.0   # arm integrator gain [rad / (rad·s)] — corrects steady-state joint error (gravity-comp residual, IK residual)
_ARM_INT_SAT_WARN_TICKS = 50   # consecutive ticks pinned at the clamp before
                               # the integrator says so (1 s at 50 Hz)
_ARM_INTEGRAL_LIMIT     = 0.4  # per-joint integral clamp [rad]. 0.2 -> 0.4
                               # (2026-08-06). MEASURED, not guessed, from
                               # recordings/armfx_0806_214548_left.npz over a
                               # 40 s steady window, where the identity
                               #   pos_ref = cmd_joint + action*0.6 + KI*integ
                               # closes to 0.000 mrad:
                               #   left_shoulder_1  policy residual +0.2520 rad
                               #                    integ  -0.2000 rad, AT THE
                               #                    CLAMP 100% of ticks
                               #                    net offset +0.0520 = 2.89 deg
                               # Every other arm joint sat at 0% clamp and landed
                               # on its streamed target to 0.00 deg; shoulder_1
                               # tracked its REFERENCE to 0.088 deg — better than
                               # any of them. Nothing was stalled. The clamp was
                               # simply smaller than the residual it had to
                               # cancel, and the 2.89 deg leftover is 0.03 deg
                               # over SETTLE_TOL_RAD (2.865 deg), which timed the
                               # settle out EVERY round and handed the tracker a
                               # ~72 mm deficit it then died on (T13).
                               #
                               # THIS IS A BAND-AID AND SHOULD BE SAID SO. The
                               # integrator is being spent cancelling the
                               # locomotion policy's own arm residual, which
                               # real_env adds on top of the cuRobo joint stream
                               # (search action_to_motor_map below) — not on
                               # correcting real gravity/IK error, which is what
                               # it was built for. The real fix is to mask the
                               # arm joints out of the policy residual while the
                               # arm stream owns them; that needs upstream, because
                               # the task is …MixedArms… and the residual may be
                               # load-bearing for balance.
# SAFETY PATCH (07-25): arm feed-forward torque ceiling.
_ARM_TAU_FF_MAX         = 15.0  # per-joint |feed-forward| ceiling [N·m]
# The arm feed-forward had NO magnitude limit: whatever
#     tau = M(q)*ddq + C(q,dq)*dq + g(q)
# came out of the inverse dynamics went straight to the drives, scaled only by
# joint_limit_scale (<=1, and only near a joint limit) and the CLI torque ratio.
# ddq is the IK solver's desired acceleration, so ANY discontinuity in the IK
# solution — a branch flip, a target jump, a reseed — reads as "cross a large
# angle within one 20 ms tick" and the inertia term explodes.
#
# Measured on hardware 07-24, right arm, two separate runs. Normal operation
# sits at |tau_ff| ~ 0.7 median / 4.2 at the 99th percentile, and the left arm
# never exceeded 7.7 across either run. During the two incidents the right arm
# was commanded 292.0 and 260.5 N·m — 60x the 99th percentile — across four
# joints at once (shoulder_1/2/3 + elbow). The drives saturate, but the measured
# torque still reached 42.6 N·m and threw the arm at 10-13 rad/s into the body.
#
# 15 N·m is ~3.5x the 99th percentile and ~2x the largest honest value ever
# recorded, so it cannot interfere with real gravity/inertia compensation, while
# anything above it is by construction a solver fault rather than a physical
# demand. Saturation is logged, never silent: hiding it would hide the IK
# discontinuity that is the actual defect.
#
# NOTE this is a MITIGATION, not the fix. In the 07-24 incidents the spike was a
# CONSEQUENCE — right_shoulder_2/3 (CAN 21/22, both on can21) lost torque first
# and the arm was already falling 0.16 s before the feed-forward blew up. The
# ceiling stops the fall from being amplified into a slam; it does not stop the
# dropout.
# SAFETY PATCH (07-25): arm motor-dropout watchdog.
_WATCHDOG_STALE_TICKS   = 3     # consecutive bit-identical (pos, torque) feedback
                                # rows before an arm motor counts as OFF-BUS.
# WHY: in both 07-24 collisions right_shoulder_2+3 (CAN 21/22) went silent on
# can21 for 0.3-0.4 s, tripped their own CAN-timeout protection (param
# 0x7028=5000 ≈ 0.25 s) into Reset, and came back torqueless with a 13-16 deg
# position jump. The SLAM was not the dropout: it was the controller reacting
# to that jump — inverse dynamics demanded 260-292 N.m and the still-alive
# joints (shoulder_1 measured 34-42.6 N.m, 7-13 rad/s) whipped the half-limp
# linkage into the torso. This watchdog removes the reaction: the moment any
# arm motor's feedback freezes, that whole arm's targets are pinned at the
# MEASURED pose and its feed-forward is zeroed — a damped hold, no chasing.
# THRESHOLD: 3 ticks = 60 ms at 50 Hz. Across all four 07-24 forensic
# recordings (two collision runs, two clean left-arm runs, ~4.7 min total)
# no non-incident freeze ever reached 3 ticks (normal repeat rate 0.4-1.0%
# single ticks); the incidents ran 15/19 ticks. 60 ms also beats the motor's
# own 250 ms self-disable, so the latch is armed well before the arm can fall.
# DETECTION is differential — a joint is stale only while >=2 joints anywhere
# on the robot ARE updating — because bit-identical feedback from a genuinely
# still, unpowered motor is legitimate (FakeMotorController reports constants).
# A GLOBALLY frozen snapshot is treated as a receiver-side condition and the
# watchdog deliberately stays silent on it. NOTE (07-25 review): nothing else
# in this process monitors global feedback freshness either — that gap is
# real and out of this watchdog's scope.
_WATCHDOG_MIN_TICK_S    = 0.012 # ignore catch-up ticks: after a scheduling stall
                                # the rate limiter fires 2-6 ms ticks, shorter
                                # than a motor's ~4-5 ms feedback period, so
                                # frames legitimately have not arrived and
                                # freezes are 20-30x enriched there (measured
                                # across the four 07-24 recordings). Ticks
                                # shorter than this advance NO watchdog state.
_WATCHDOG_STALE_MIN_S   = 0.05  # wall-clock floor on a stale streak: ticks are
                                # cheap under jitter, seconds are not. A latch
                                # needs BOTH >=_WATCHDOG_STALE_TICKS qualified
                                # ticks AND this much elapsed time frozen.
# THE LATCH IS TERMINAL for the run (matches the operator's fail-closed
# intervention philosophy): a recovered motor sits in Reset needing a manual
# enable anyway, and auto-resume against a post-fall pose is exactly the snap
# this patch exists to prevent. Restart real_env after fixing the cause.
_RECORD_WARMUP_STEPS    = 500   # skip first ~10 s of recording (@ 50 Hz)

DEFAULT_TRIGGERS = {
    "kb": {},
    "js": {"A": "[SHUTDOWN]", "X": "[TORQUE_DOWN]", "B": "[TORQUE_UP]", "LB": "[NAV_RESUME]"},
}

# EE frame config — only place to edit when switching body↔site or renaming EEs.
_EE_LEFT_NAME  = "end_effector_L_site"
_EE_RIGHT_NAME = "end_effector_R_site"
_EE_LEFT_TYPE  = "site"   # EEFrameType: "body" | "site"
_EE_RIGHT_TYPE = "site"

_MJ_OBJ_FROM_FRAME = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "site": mujoco.mjtObj.mjOBJ_SITE,
}
_EE_FRAME_TYPE_MAP = {_EE_LEFT_NAME: _EE_LEFT_TYPE, _EE_RIGHT_NAME: _EE_RIGHT_TYPE}
_EE_SIDE           = {_EE_LEFT_NAME: "left",         _EE_RIGHT_NAME: "right"}


# PATCH (08-12): wrist observation frame de-fold.
# The encoders speak the REBUILT deploy model's frame (rebuild_deploy_model.py,
# invariant "encoder 0 == model qpos 0"), which differs from the TRAINING
# model's frame at exactly three joints: left_wrist_1 carries a -pi/2
# reference, right_wrist_1 +pi/2, and left_wrist_2's axis is flipped. The
# policy was trained on training-frame absolute qpos, but this file fed it raw
# encoder values in joint_pos / joint_vel / target_arm_joint_pos — a constant
# corruption at rest (tolerated) that turns DYNAMIC during left-arm reaches:
# offline A/B replay (08-12) showed the encoder-frame wrist stream drives a
# sustained spurious waist command during a left-arm front reach (the
# front-grasp waist yaw) which the de-fold removes.
_WRIST_HALF_PI = np.pi / 2.0


def wrist_defold_columns(term_order, col_slices, arm_joint_names):
    """Column index sets in the combined history row for the wrist de-fold.

    Returns (plus_cols, minus_cols, neg_cols): row columns that need +pi/2,
    -pi/2, and sign negation to express the encoder-frame row in the training
    frame. Velocity columns only appear in neg_cols — the wrist_1 corrections
    are constant offsets with zero derivative. Pure function so tests can pin
    the arithmetic without constructing the env.
    """
    start = {k: s.start for k, s in zip(term_order, col_slices)}
    jp, jv = start["joint_pos"], start["joint_vel"]
    ta = start["target_arm_joint_pos"]
    u = {n: i for i, n in enumerate(URDF_JOINT_NAMES)}
    a = {n: i for i, n in enumerate(arm_joint_names)}
    plus = [jp + u["left_wrist_1_joint"], ta + a["left_wrist_1_joint"]]
    minus = [jp + u["right_wrist_1_joint"], ta + a["right_wrist_1_joint"]]
    neg = [jp + u["left_wrist_2_joint"], jv + u["left_wrist_2_joint"],
           ta + a["left_wrist_2_joint"]]
    return plus, minus, neg


# PATCH (08-12, user order): dynamic standing waist pin.
# During every cmd-0 phase of a mission (aiming, planning, fetching, grasping)
# the Cosine checkpoint slowly grinds the waist rightward — the wrist-obs
# frame seam is the diagnosed driver, and its direct fix (--obs-frame-fix)
# failed hang validation the same day. This is the symptom-level clamp with
# hardware history: the STATIC _PIN_WAIST held standing grasps waist-stable
# through mid-July before walking missions needed the waist back. Dynamic
# version: pin only while the nav command is zero, hand back when walking.
_WAIST_PIN_CMD_EPS = 0.02      # commands are exact zeros when idle (nav sends
                               # 0.0, the gamepad has a 0.2 deadband), so any
                               # |cmd| above this is a real drive request


class WaistPin:
    """Blend authority over the waist target between the policy and a pin.

    update() returns alpha in [0,1]: 0 = policy owns the waist, 1 = fully
    pinned at the default. Pin engages after STILL_DEBOUNCE_S of zero command
    and blends in over ENGAGE_S (no step on a settling robot); any drive
    command starts an immediate RELEASE_S blend-out (no step into the walk —
    the gait ramp is slower than this). Pure, clock-injected, testable."""

    STILL_DEBOUNCE_S = 0.5
    ENGAGE_S = 0.5
    RELEASE_S = 0.3

    def __init__(self):
        self._still_since = None
        self.alpha = 0.0

    def update(self, cmd_active: bool, now: float, dt: float) -> float:
        if cmd_active:
            self._still_since = None
            self.alpha = max(0.0, self.alpha - dt / self.RELEASE_S)
        else:
            if self._still_since is None:
                self._still_since = now
            if now - self._still_since >= self.STILL_DEBOUNCE_S:
                self.alpha = min(1.0, self.alpha + dt / self.ENGAGE_S)
        return self.alpha


class HumanoidRealEnv(HumanoidBase):
    """Policy-side robot environment: obs assembly, reference folding, motor I/O.

    Threads. The main thread owns the control tick (main(): rate.sleep ->
    process_input -> policy forward -> step()). Around it run daemon threads
    that exchange latest-value buffers only: the NNGSubscriber receivers for
    9873/9874, the NNGPublisher sender for 9870, the keyboard/joystick reader
    (KeyboardGampadController), the IK worker (_ik_worker_loop, --use-ik; one
    tick of latency, handshake via _ik_trigger and the _ik_* locks) and the
    CAN sender/receiver threads inside hardware_bindings, which read
    motor.mech_pos_ref / mech_torque_ref asynchronously — hence "pre-mask,
    never write-then-zero" at every feed-forward write site.

    Tick order in step(): clip action -> _update_arm_targets (arm/gaze/ee
    packet, arm-silence ramp, gaze failsafe, ee_action relay) -> target =
    default pose + arm baseline + gaze reference + policy residual -> waist
    pin (_PIN_WAIST / --pin-waist-standing) -> near-limit clamp -> refresh
    joint_pos -> _arm_watchdog_check -> IK overwrite for Cartesian arms ->
    arm integrator -> watchdog pin (the LAST position writer) -> mech_pos_ref
    -> arm torque feed-forward (inverse dynamics under IK, else gravity comp,
    else zero; clamped and watchdog-masked) -> get_observation (publishes
    9870) -> recorder.

    Failsafes. (1) Stream silence: no nav_cmd for _NAV_CMD_TIMEOUT stops the
    base; no arm packet for _ARM_CMD_TIMEOUT ramps both arms to the default
    pose at _ARM_RETURN_RATE; no gaze packet for _GAZE_CMD_TIMEOUT parks the
    gimbals at _GAZE_RETURN_RATE. (2) CAN dropout watchdog (--arm-watchdog,
    default on): a joint whose feedback freezes for _WATCHDOG_STALE_TICKS
    latches its arm into a damped hold — position pinned at the measured
    pose, feed-forward zeroed, integrator cleared — terminal until restart.

    Flags -> behaviour: --use-ik starts the IK worker and lets Cartesian
    arm_targets drive the arms; --grav-comp selects the gravity feed-forward
    branch when IK is not active; --ee-service relays ee_action to the
    gripper service; --enable-motor picks which motor groups are powered
    (true/false/leg/arm/camera/arm_camera); --torque-limit is the global
    ratio (_CAM_TORQUE_RATIO may cap the gimbals below it); --fake swaps in
    the fake motor/IMU drivers; --pin-waist-standing blends the waist to its
    default at cmd 0; --obs-frame-fix / --lw2-mirror change the wrist
    observation/action frame (mutually exclusive, both off by default);
    --record writes a replay pkl; --vicon adds the Vicon base pose to the
    telemetry; --profile prints per-stage timings every 100 steps.
    """
    def __init__(self,
                 config_path=DEFAULT_CONFIG,
                 torque_limit=0.1,
                 use_fake=False,
                 gravity_compensation_enabled=False,
                 add_right_ee=False,
                 enable_motor=True,
                 record=False,
                 record_path=None,
                 triggers: dict = None,
                 use_ik: bool = False,
                 telemetry_ip: str = _site.WORKSTATION_IP,
                 high_level_controller_ip: str = "127.0.0.1",
                 arm_sender_ip: str = "127.0.0.1",
                 enable_ee_service: bool = False,
                 profile: bool = False,
                 vicon: bool = False,
                 arm_watchdog: bool = True,
                 obs_frame_fix: bool = False,
                 pin_waist_standing: bool = False,
                 lw2_mirror: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torque_limit = torque_limit
        # 07-25 watchdog flag — stored before _init_arm_targets() (called from
        # __init__ below) initializes the detector state that reads it.
        self.arm_watchdog = arm_watchdog
        self.obs_frame_fix = obs_frame_fix   # 08-12 wrist de-fold; setup below
                                             # after the model + obs buffers exist
        # 08-12 dynamic standing waist pin (see WaistPin at the module head)
        self.pin_waist_standing = pin_waist_standing
        self._waist_pin = WaistPin()
        self._waist_pin_engaged = False      # transition-print latch
        self.lw2_mirror = lw2_mirror         # 08-12 night; setup below

        # Telemetry publisher: robot binds, remote workstation subscribes
        self.publisher = NNGPublisher(f"tcp://*:{_site.TELEMETRY_PORT}")

        # Navigation command subscriber (from the nav publisher: humanoid_auto_operator.py,
        # humanoid_curobo_reach.py on an executing --journey, or gampad_gui_control.py)
        self.nav_receiver = NNGSubscriber(f"tcp://{high_level_controller_ip}:{_site.NAV_COMMAND_PORT}")
        self.nav_receiver.start()
        self._nav_paused = False    # True = operator has taken manual joystick control
        self._nav_paused_source = None  # "stick" (positioning; a starting nav
                                    # stream clears it) or "button" (explicit LB
                                    # pause; only LB clears it) — 08-11 rework
        self._nav_was_fresh = False # nav-stream freshness edge detector
        self._last_nav_data_id = -1
        self._last_nav_recv_time = 0.0

        # Arm command subscriber (from humanoid_auto_operator.py / humanoid_curobo_reach.py;
        # bench senders: gampad_gui_control.py, humanoid_joint_monkey_hw.py)
        self.arm_receiver = NNGSubscriber(f"tcp://{arm_sender_ip}:{_site.ARM_COMMAND_PORT}")
        self.arm_receiver.start()
        self._last_arm_data_id = -1
        self._last_arm_recv_time = 0.0
        self._arm_stream_is_cartesian = False  # True when stream sends ee_pos/ee_quat (Cartesian schema)
        # PROTOCOL (07-20, approved upstream verbally): PER-ARM
        # modal wire. New packet schema (fully backward compatible — the two
        # legacy schemas above still drive BOTH arms together):
        #     {"arm_targets": {"left":  {"ee_pos":[3], "ee_quat":[4]}      # Cartesian
        #                              | {"joint_pos":[7], "rate": r},     # joint
        #                      "right": ... }}          # either side may be absent
        # An absent side HOLDS its current mode and targets. _arm_cart[i] is the
        # per-arm mode (left=0, right=1); _arm_stream_is_cartesian stays as the
        # both-arms summary for the legacy guards below. Rationale: the global
        # single-mode flag forced the operator to serialize joint staircases and
        # FREEZE the other arm's Cartesian stream (07-20 forensics: grabs fired
        # against frozen targets while the cube drifted — the empty-pinch class).
        self._arm_cart = [False, False]
        self._arm_rates = [_ARM_RETURN_RATE, _ARM_RETURN_RATE]

        # EE servo service client (optional — lazy import avoids StServo hardware libs at startup).
        # The client LISTENS on the EE request socket, so only one commander can
        # exist per machine (humanoid_monitor --ee is another). If someone else
        # owns it, run without EE instead of crashing at startup.
        self.ee = None
        if enable_ee_service:
            import pynng
            from humanoid_end_effector_service import EEServiceClient
            try:
                self.ee = EEServiceClient()
            except pynng.AddressInUse:
                _log.warning("[EE] request socket already owned by another commander "
                             "(humanoid_monitor --ee?) — EE commands disabled")

        # Load config
        self.config_path = config_path   # provenance for the recorder
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.n_joints = len(URDF_JOINT_NAMES)

        # Build default joint positions and action scales from config
        self.default_joint_pos = self._build_default_joint_pos()

        self.action_scale, self.motor_kp, self.motor_kd = self._build_actuator_params()


        # Build controlled joints mapping (if specified in config)
        controlled_joints = self.config["action"].get("controlled_joints", None)
        if controlled_joints is not None:
            self.action_dim = len(controlled_joints)
            self.action_to_motor_map = torch.tensor(
                [URDF_JOINT_NAMES.index(j) for j in controlled_joints],
                dtype=torch.long, device=self.device
            )
            self.policy_action_scale = torch.tensor(
                [self.action_scale[URDF_JOINT_NAMES.index(j)] for j in controlled_joints],
                dtype=torch.float32, device=self.device
            )
            _log.info(f"[ENV] Controlled joints specified: {self.action_dim} joints")
        else:
            self.action_dim = self.n_joints
            self.action_to_motor_map = torch.arange(self.n_joints, dtype=torch.long, device=self.device)
            self.policy_action_scale = self.action_scale
            _log.info(f"[ENV] Full body control: {self.action_dim} joints")

        # Joint limits from MJCF model
        self.joint_lower = torch.tensor(JOINT_LOWER_LIMITS, dtype=torch.float32, device=self.device)
        self.joint_upper = torch.tensor(JOINT_UPPER_LIMITS, dtype=torch.float32, device=self.device)

        _log.info(f"default_joint_pos: {self.default_joint_pos}")
        _log.info(f"action_scale: {self.action_scale}")
        _log.info(f"joint_limits: [{self.joint_lower.min():.2f}, {self.joint_upper.max():.2f}] rad")
        _log.info(f"motor_kp: {self.motor_kp}\nmotor_kd: {self.motor_kd}")

        # State buffers
        self.previous_action = torch.zeros(self.action_dim, device=self.device)
        self.command = torch.zeros(3, device=self.device)
        self._command_cpu = torch.zeros(3)  # CPU mirror — passed to gait clock to avoid D2H sync
        self.base_lin_vel = torch.zeros(3, device=self.device)  # updated by policy or stays zero

        # Sensor state tensors — single contiguous CUDA buffer, individual tensors are views.
        # Eliminates 7 H2D copies/step → 1: all sensor data written via numpy slice + one copy_().
        # Extra 4 slots at end reserved for gait_phase (folded into bundle when present).
        _N = self.n_joints
        self._sensor_cuda = torch.zeros(10 + _N * 4 + 4, dtype=torch.float32, device=self.device)
        self._sensor_np   = np.zeros(10 + _N * 4 + 4, dtype=np.float32)
        _s = 0
        self.base_ang_vel      = self._sensor_cuda[_s:_s+3];  _s += 3
        self.projected_gravity = self._sensor_cuda[_s:_s+3];  _s += 3
        self.base_quat_wxyz    = self._sensor_cuda[_s:_s+4];  _s += 4
        self.joint_pos         = self._sensor_cuda[_s:_s+_N]; _s += _N
        self.joint_vel         = self._sensor_cuda[_s:_s+_N]; _s += _N
        self.joint_torque      = self._sensor_cuda[_s:_s+_N]; _s += _N
        self.joint_effort_ref  = self._sensor_cuda[_s:_s+_N]

        # Action clipping (matches training env)
        # None means no clipping, float value clips to [-value, value]
        self.action_clip_value = self.config["action"].get("clip_actions", None)
        if self.action_clip_value is not None:
            _log.warning(f"[ENV] Action clipping enabled: ±{self.action_clip_value}")

        self.obs_dim = self.config["obs_dim"]
        # Observation group the deployed policy consumes. For L2T runs this is the
        # deployable "student" group, not the privileged "actor" group. Defaults to
        # "actor" so pre-flag configs keep their old behavior.
        self._policy_obs_group = self.config.get("policy_obs_group", "actor")
        # Authoritative velocity-estimator signal. None = key absent (old config) →
        # main loop falls back to hasattr; True/False = exported flag wins.
        self._has_velocity_estimator_cfg = self.config.get("has_velocity_estimator", None)
        # Load observation structure (default to standard if not present for backward compatibility)
        self.obs_structure = self.config["observation"][self._policy_obs_group].keys()  # live view — config must not be mutated after init
        _log.info(f"[ENV] Observation structure ({self._policy_obs_group}): {self.obs_structure}")

        # Define base observation dimensions per term (before history expansion)
        self._obs_term_dims = {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "joint_pos": self.action_dim,
            "joint_vel": self.action_dim,
            "joint_effort": self.action_dim,
            "joint_effort_ref": self.action_dim,
            "actions": self.action_dim,
            "command": 3,
            "gait_phase": 4,
            "target_camera_joint_pos": 4,
        }
        
        # Override with explicit term_dim from the exported YAML if present
        for group_name, group_terms in self.config["observation"].items():
            for term_name, term_cfg in group_terms.items():
                if "term_dim" in term_cfg and term_cfg["term_dim"] is not None:
                    # term_dim is the per-frame dimension (not including history expansion)
                    per_frame_dim = term_cfg["term_dim"]
                    self._obs_term_dims[term_name] = per_frame_dim
                    if term_name in ["joint_pos", "joint_vel"] and per_frame_dim == self.n_joints:
                        _log.info(f"[ENV] Config explicit observe_full_joints=True. Dimension: {per_frame_dim}")

        self._observe_full_joints = self._obs_term_dims.get("joint_pos", self.action_dim) == self.n_joints

        # Create history buffers for terms with history_length > 0
        self._obs_history_buffers = {}
        self._obs_history_config = {}
        obs_policy_config = self.config["observation"][self._policy_obs_group]
        for term_name in self.obs_structure:
            term_cfg = obs_policy_config.get(term_name, {})
            history_length = term_cfg.get("history_length", 0)
            if history_length > 0:
                obs_dim = self._obs_term_dims.get(term_name, 0)
                if obs_dim == 0:
                    _log.warning(f"[WARNING] Unknown obs term dimension for '{term_name}', skipping history")
                    continue
                self._obs_history_buffers[term_name] = CircularBuffer(
                    max_len=history_length,
                    batch_size=1,  # Real robot is single environment
                    obs_dim=obs_dim,
                    device=self.device,
                )
                self._obs_history_config[term_name] = {
                    "flatten": term_cfg.get("flatten_history_dim", True)
                }
                _log.info(f"[ENV] History buffer for '{term_name}': length={history_length}, dim={obs_dim}")

        # Phase C: combined double-buffer — 3 CUDA kernels/step vs 2N+1 individual.
        # Requires: all actor terms have history, all share same history_length.
        _hist_lens = [b._max_len for b in self._obs_history_buffers.values()]
        _all_covered = len(self._obs_history_buffers) == len(list(self.obs_structure))
        if self._obs_history_buffers and _all_covered and len(set(_hist_lens)) == 1:
            _H = _hist_lens[0]
            _hist_keys = list(self._obs_history_buffers.keys())
            _D = sum(self._obs_term_dims[k] for k in _hist_keys)
            self._hist_H          = _H
            self._hist_D          = _D
            self._hist_term_order = _hist_keys
            _s = 0; _slices = []
            for k in _hist_keys:
                d = self._obs_term_dims[k]; _slices.append(slice(_s, _s + d)); _s += d
            self._hist_col_slices = _slices
            # Double buffer: positions [ptr] and [ptr+H] always mirror each other.
            # Slice [ptr:ptr+H] is always contiguous, sorted oldest-to-newest.
            self._hist_combined  = torch.zeros(1, 2 * _H, _D, device=self.device)
            self._hist_cur_frame = torch.zeros(1, _D, device=self.device)
            self._hist_ptr       = 0
            self._hist_init      = False
            _log.info(f"[ENV] Combined history buffer: H={_H}, D_total={_D}, {len(_hist_keys)} terms")
        else:
            self._hist_combined = None

        # Detect sequence (3D) obs mode: any actor term with flatten_history_dim=False + history > 0
        # Used by RMA-CNN / TCN policies that expect (1, H, D_act) input.
        self._obs_is_sequence = any(
            cfg.get("history_length", 0) > 0 and not cfg.get("flatten_history_dim", True)
            for cfg in obs_policy_config.values()
        )
        if self._obs_is_sequence:
            _log.info("[ENV] Sequence obs mode active: get_observation() returns (1, H, D)")

        # Initialize hardware: motor, IMU, MuJoCo, gravity compensation.
        # super().__init__() also sets self.device, self.n_joints, self.torque_correction_ratio.
        super().__init__(
            torque_limit=torque_limit,
            use_fake=use_fake,
            gravity_compensation_enabled=gravity_compensation_enabled,
            add_right_ee=add_right_ee,
            enable_motor=enable_motor,
        )
        # PATCH (07-16): cap the cam gimbal torque right
        # after the base applied the global CLI ratio (see _apply_group_torque_limits).
        self._apply_group_torque_limits()

        # Override control_freq/dt from config (base defaults to 200 Hz low-level rate).
        # prepare() reads these at call-time, so this override is in effect before prepare() runs.
        self.control_freq = self.config["simulation"]["control_freq"]
        self.dt = 1.0 / self.control_freq

        # PATCH (08-12): wrist de-fold setup — see the
        # module-level rationale at wrist_defold_columns. Runs after
        # super().__init__ so both the obs buffers and the MuJoCo model exist.
        if self.obs_frame_fix:
            if self._hist_combined is None:
                raise RuntimeError(
                    "--obs-frame-fix requires the combined history obs path "
                    "(this policy's obs config does not use it)")
            # The loaded deploy model must BE the encoder frame (the rebuilt
            # model with the wrist_1 refs). If these refs ever disappear —
            # e.g. the frames get unified upstream — de-folding would corrupt
            # instead of correct, so refuse loudly rather than double-convert.
            _q0 = {}
            for _n in ("left_wrist_1_joint", "right_wrist_1_joint"):
                _jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, _n)
                _q0[_n] = float(self.mj_model.qpos0[self.mj_model.jnt_qposadr[_jid]])
            if not (abs(_q0["left_wrist_1_joint"] + _WRIST_HALF_PI) < 1e-3
                    and abs(_q0["right_wrist_1_joint"] - _WRIST_HALF_PI) < 1e-3):
                raise RuntimeError(
                    f"--obs-frame-fix: loaded model wrist_1 qpos0 {_q0} does not "
                    f"carry the -+pi/2 encoder refs — the encoder/training frame "
                    f"seam has changed; re-derive the de-fold before using it")
            # same derivation as _init_arm_targets (which runs later in init):
            # target_arm_joint_pos columns follow the _ARM_PATTERN order
            _arm_names = [n for n in URDF_JOINT_NAMES if re.search(_ARM_PATTERN, n)]
            _p, _m, _n2 = wrist_defold_columns(
                self._hist_term_order, self._hist_col_slices, _arm_names)
            self._defold_plus = torch.tensor(_p, dtype=torch.long, device=self.device)
            self._defold_minus = torch.tensor(_m, dtype=torch.long, device=self.device)
            self._defold_neg = torch.tensor(_n2, dtype=torch.long, device=self.device)
            _lw2_motor = URDF_JOINT_NAMES.index("left_wrist_2_joint")
            _hits = (self.action_to_motor_map == _lw2_motor).nonzero().flatten()
            if _hits.numel() != 1:
                raise RuntimeError("--obs-frame-fix: left_wrist_2_joint not in "
                                   "controlled joints — cannot flip its residual")
            self._defold_act_lw2 = int(_hits[0])
            _log.info("[OBS] wrist frame de-fold ON: policy sees TRAINING-frame "
                      "wrists (wrist_1 -+pi/2 refs removed, left_wrist_2 "
                      "unflipped); left_wrist_2 action residual flips at the "
                      "wire boundary. Telemetry and all control paths stay in "
                      "the encoder frame.")

        # PATCH (08-12 night): left_wrist_2 DEVIATION
        # mirror. The user's controlled hardware comparison (seed1, front
        # grasp: RIGHT arm = no tilt, LEFT arm = clockwise grind) isolated the
        # stack's single left-only asymmetry: left_wrist_2 is the one joint
        # whose encoder sign is inverted vs the training frame. The full
        # de-fold (--obs-frame-fix) failed hang validation because it moved
        # the STATIC operating point the checkpoints have adapted to; this
        # variant mirrors ONLY the deviation around the default —
        #   obs_pos/target = 2*default - enc      (static point bit-identical)
        #   obs_vel        = -enc_vel             (pure deviation)
        #   action         = lw2 residual sign-flipped at the wire boundary
        # so at the chest home the policy's inputs are unchanged, and only the
        # MOTION of that wrist reads in the training-correct direction.
        if self.lw2_mirror:
            if self.obs_frame_fix:
                raise RuntimeError("--lw2-mirror and --obs-frame-fix both "
                                   "rewrite left_wrist_2's policy view — "
                                   "pick one")
            if self._hist_combined is None:
                raise RuntimeError("--lw2-mirror requires the combined "
                                   "history obs path")
            _arm_names = [n for n in URDF_JOINT_NAMES if re.search(_ARM_PATTERN, n)]
            _, _, _neg = wrist_defold_columns(
                self._hist_term_order, self._hist_col_slices, _arm_names)
            # neg = [jpos col, jvel col, target col] for left_wrist_2
            self._lw2_pt_cols = torch.tensor([_neg[0], _neg[2]],
                                             dtype=torch.long, device=self.device)
            self._lw2_vel_col = int(_neg[1])
            _lw2_motor = URDF_JOINT_NAMES.index("left_wrist_2_joint")
            self._lw2_default = float(self.default_joint_pos[_lw2_motor])
            _hits = (self.action_to_motor_map == _lw2_motor).nonzero().flatten()
            if _hits.numel() != 1:
                raise RuntimeError("--lw2-mirror: left_wrist_2_joint not in "
                                   "controlled joints")
            self._defold_act_lw2 = int(_hits[0])   # shared with obs_frame_fix's
                                                   # action-flip site
            _log.info(f"[OBS] left_wrist_2 deviation mirror ON around default "
                      f"{self._lw2_default:+.4f} rad — static policy inputs "
                      f"unchanged, wrist motion reads training-direction; "
                      f"lw2 action residual flips at the wire boundary.")

        # Vicon: disabled by default. To activate, instantiate ViconInterface here.
        # See humanoid_vicon.py for setup instructions.
        self.vicon = None  # ViconInterface | None
        if vicon:
            from humanoid_vicon import ViconInterface
            self.vicon = ViconInterface(
                vicon_ip=_site.VICON_IP,
                objects={"vicon": "grl_hb"},
                calibration_path=str(_site.VICON_CALIBRATION_PATH),
            )

        # COM tracking buffers for robot_com_b observation
        self._com_pos_b = torch.zeros(3, device=self.device)
        self._com_vel_b = torch.zeros(3, device=self.device)

        # Pre-alloc obs output buffer (eliminates torch.cat alloc every step)
        if self._obs_is_sequence:
            self._obs_buf: torch.Tensor | None = None  # lazy: shape depends on H, set on first call
        else:
            self._obs_buf = torch.zeros(1, self.obs_dim, device=self.device)

        # Pre-alloc target_pos buffer (eliminates default_joint_pos.clone() every step)
        self._target_pos_buf = self.default_joint_pos.clone()

        # Cached limit-check index slices (joint limits are static; eliminates 6 gather ops/step)
        self._joint_lower_ctrl = self.joint_lower[self.action_to_motor_map].clone()
        self._joint_upper_ctrl = self.joint_upper[self.action_to_motor_map].clone()

        # Waist joint index (observe-only, not controlled)
        if "waist_joint_pos" in self.obs_structure:
            self._waist_idx = next(
                (i for i, n in enumerate(URDF_JOINT_NAMES) if re.search(r'waist', n, re.I)),
                None,
            )
            if self._waist_idx is None:
                raise RuntimeError("[ENV] waist_joint_pos in obs_structure but no waist joint found in URDF")

        if self.vicon is not None:
            self.vicon.verify()

        # EE state: wrist poses in base FLU frame ride the 9870 telemetry as `ee`
        # (humanoid_monitor.py, humanoid_auto_operator.py and humanoid_curobo_reach.py
        # read them); no separate 9875 socket is bound — the 9875 some older
        # design notes still list is historical.
        # FK is free — update_kinematics() already populates mj_data.xpos/xmat each step.
        self.ee_frame_ids = {
            _EE_LEFT_NAME:  mujoco.mj_name2id(self.mj_model, _MJ_OBJ_FROM_FRAME[_EE_LEFT_TYPE],  _EE_LEFT_NAME),
            _EE_RIGHT_NAME: mujoco.mj_name2id(self.mj_model, _MJ_OBJ_FROM_FRAME[_EE_RIGHT_TYPE], _EE_RIGHT_NAME),
        }
        _log.info("[ENV] EE state on the 9870 telemetry stream as `ee`")

        # Recording state for simulation replay
        self.recorder = HumanoidRecorder(record_path=record_path, enabled=record)

        _log.info(f"[ENV] Ready: {self.n_joints} joints, {self.control_freq} Hz, torque={torque_limit*100:.0f}%")
        _log.debug(f"[DEBUG] Motor mech_pos available: {len(self.motor.mech_pos)} values")

        # IMU freshness tracking (counter increments every packet on real hardware)
        self._imu_prev_counter = -1
        self._imu_stale_steps = 0

        # Limit-warning throttle (avoids flooding the log at 100 Hz)
        self._step_count = 0
        self._last_limit_warn_step = -_LIMIT_WARN_STEP_INTERVAL  # fire immediately on first hit
        self.joint_limit_threshold = 0.05  # rad (~3 deg)

        # CPU numpy mirrors for fast limit detection in step() — no H2D/CUDA ops in common case.
        # Copied from the CUDA tensors once at init; joint limits are static after construction.
        self._joint_lower_ctrl_np    = self._joint_lower_ctrl.cpu().numpy().copy()
        self._joint_upper_ctrl_np    = self._joint_upper_ctrl.cpu().numpy().copy()
        self._action_to_motor_map_np = self.action_to_motor_map.cpu().numpy()
        self._joint_limit_threshold_f = self.joint_limit_threshold

        # Velocity scaling for joystick axes.
        # PATCH (07-31): FULL STICK == THE TRAINED
        # COMMAND RANGE (user). These are not preferences — they are the
        # curriculum's final velocity_stages entry for this policy
        # (humanoid_velocity_env_cfg.py, "velocity_stages" step 5000 *
        # num_steps_per_env, which model_0015000 is far past):
        #     lin_vel_x (-1.0, 1.0) m/s     lin_vel_y (-1.0, 1.0) m/s
        #     ang_vel_z (-0.7, 0.7) rad/s
        # so a stick pushed to the stop asks for exactly the fastest command
        # the policy was ever trained to track, and never past it. The
        # ANGULAR scale is 0.7, NOT 1.0 and not the 0.8 it carried before:
        # 0.8 was outside the trained yaw range, i.e. the old default could
        # command a turn rate the policy had never seen.
        # SIDE EFFECT worth knowing: _NAV_OVERRIDE_THRESHOLD (0.2) is compared
        # against the SCALED command, so raising the scale makes the manual
        # takeover more sensitive — at 1.0 a 20 % stick deflection seizes
        # control (and holds it until LB), versus 67 % at scale 0.3.
        self.VEL_SCALE_LINEAR = 1.0   # m/s max   — trained lin_vel_x/y range
        self.VEL_SCALE_ANGULAR = 0.7  # rad/s max — trained ang_vel_z range

        # Controller (keyboard + gamepad)
        t = triggers or DEFAULT_TRIGGERS
        self.ctrl = KeyboardGampadController(kb_triggers=t["kb"], js_triggers=t["js"])
        self.ctrl.start()
        # The banner lists what is ACTUALLY bound, and is written out rather
        # than derived from `t`: t["kb"] is empty, so KeyboardGampadController
        # substitutes its own default map (esc/space) and a derived line would
        # under-report ESC. Keyboard axes are LeftX/LeftY/RightX
        # (KeyboardThread.get_axes) while forward velocity reads RightY, so W/S
        # move an axis this loop never reads — forward/back is gamepad-only.
        # Torque and reset have no keyboard keys either; torque is X/B on the
        # gamepad, and no map binds [RESET] at all.
        _log.info("[CTRL] Keyboard: A/D=strafe  Q/E=turn  ESC=shutdown")
        _log.info("[CTRL] Gamepad: RightY=fwd/back  LeftX=strafe  RightX=turn  "
                  "A=shutdown  X/B=torque-/+  LB=nav pause/resume")

        # Arm target tracking logic (for HumanoidRandArmsAdditiveCtrlTracking)
        # Always init arm targets — needed for arm_receiver stream even when
        # "target_arm_joint_pos" is not in obs_structure (arm replay without policy tracking obs)
        self.current_target_arm_pos = None
        self._init_arm_targets()
        arm_index_set = set(self.arm_motor_indices)
        self._arm_action_mask = torch.tensor(
            [idx.item() in arm_index_set for idx in self.action_to_motor_map],
            dtype=torch.bool,
            device=self.device,
        )
        self._arm_stream_active = False
        self.arm_integrator_enabled = True
        # Arm gain caches as CUDA tensors; single numpy conversion happens at motor write boundary.
        self._arm_kp = torch.as_tensor(
            self.motor_kp[self.arm_motor_indices], dtype=torch.float32, device=self.device
        )
        self._arm_kd = torch.as_tensor(
            self.motor_kd[self.arm_motor_indices], dtype=torch.float32, device=self.device
        )

        self._apply_arm_gains(
            kp=self.motor_kp[self.arm_motor_indices],
            kd=self.motor_kd[self.arm_motor_indices],
        )

        # Camera gaze (cam-equipped robot). The policy is a residual on the gaze
        # reference: cam joint target = gaze_reference + scale*action (mirrors sim
        # JointPosRefResidualAction, joint_ref_command.py:745). gaze_reference holds the
        # latest clamped setpoint commanded over port 9874 ("gaze_targets"); it feeds
        # both the policy obs and the step() baseline, and defaults to look-forward.
        #   gaze_in_obs:         policy observes the reference → allocate gaze_reference + obs.
        #   policy_actuates_cam: gaze_in_obs AND cam joints ∈ controlled_joints → step()
        #                        uses gaze_reference as the motor baseline. Else the cam
        #                        PD-holds default_joint_pos (non-cam policy, action_scale 0).
        self._cam_motor_indices = [i for i, n in enumerate(URDF_JOINT_NAMES) if n.startswith("cam_")]
        self.gaze_in_obs = "target_camera_joint_pos" in self.obs_structure
        _ctrl_set = set(self.action_to_motor_map.tolist())
        self.policy_actuates_cam = self.gaze_in_obs and all(i in _ctrl_set for i in self._cam_motor_indices)
        if self.gaze_in_obs:
            self.gaze_reference = self.default_joint_pos[self._cam_motor_indices].clone()
            # PATCH (07-16): gaze smoothing state — raw
            # 20 Hz operator target; gaze_reference glides toward it per step.
            self._gaze_ref_target = self.gaze_reference.clone()
            # Gaze silence failsafe (2026-08-08). _cam_home is the ORIGIN the
            # operator's zero_gimbals aims at — cam joints carry no default_pose
            # entry, so it is all zeros; taken from default_joint_pos anyway so
            # the failsafe target and the boot reference can never disagree.
            self._cam_home = self.default_joint_pos[self._cam_motor_indices].clone()
            self._last_gaze_recv_time = 0.0
            self._gaze_stream_active = False
            self._cam_joint_lower = self.joint_lower[self._cam_motor_indices]
            self._cam_joint_upper = self.joint_upper[self._cam_motor_indices]
            _log.info(f"[ENV] Gaze reference via 'gaze_targets' "
                      f"({len(self._cam_motor_indices)} cam joints, "
                      f"residual={'on' if self.policy_actuates_cam else 'off'})")

        # IK (Cartesian arm control)
        self.use_ik = use_ik
        if self.use_ik:
            self._setup_ik()

        # Gait phase clock (HumanoidVelocityRMACNNShortEstimatorPhase)
        self.step_dt = self.dt  # real robot: one control period per policy step
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self._episode_buf_cpu = torch.zeros(1, dtype=torch.long)  # CPU mirror — passed to gait clock to avoid D2H sync
        self._has_gait_phase = "gait_phase" in self.obs_structure
        # Gait values are folded into the sensor bundle H2D copy (save 1 kernel launch ~150 µs).
        # _gait_np_offset: byte offset in _sensor_np where 4 gait floats live.
        # gait_phase_sensor: CUDA view into those 4 slots — valid after each obs_h2d_copy.
        if self._has_gait_phase:
            self._gait_np_offset  = 10 + _N * 4
            self.gait_phase_sensor = self._sensor_cuda[self._gait_np_offset:self._gait_np_offset + 4]
        self._gait_clock = None
        if self._has_gait_phase:
            _phase_cfg = self.config.get("observation", {}).get(self._policy_obs_group, {}).get("gait_phase", {})
            phase_params = _phase_cfg.get("params", {})
            period_range = phase_params.get("period_range", (0.55, 0.75))
            lin_thresh = float(phase_params.get("lin_thresh", 0.05))
            yaw_thresh = float(phase_params.get("yaw_thresh", 0.05))
            self._gait_clock = GaitPhaseClock(
                period_range=period_range,
                linear_velocity_threshold=lin_thresh,
                yaw_velocity_threshold=yaw_thresh,
                step_dt=self.step_dt,
                device=self.device
            )
            _log.info(f"[ENV] Gait phase clock: period={period_range}s thresh=(lin={lin_thresh},yaw={yaw_thresh})")

        self._profile_enabled = profile
        self._prof: dict[str, list] = {}  # tag -> [count, sum_us, max_us]

    def _prof_tick(self, tag: str, t0: int) -> None:
        us = (time.perf_counter_ns() - t0) * 1e-3
        s = self._prof.get(tag)
        if s is None:
            self._prof[tag] = [1, us, us]
        else:
            s[0] += 1; s[1] += us; s[2] = max(s[2], us)

    def _prof_report(self) -> None:
        lines = [f"[PROF] step={self._step_count}"]
        for tag, (n, total, mx) in sorted(self._prof.items()):
            lines.append(f"  {tag:22s}: mean={total/n:7.1f} µs  max={mx:7.1f} µs")
        _log.info("\n".join(lines))

    def _init_arm_targets(self):
        """Initialize static arm targets using default pose."""
        # 1. Identify arm joints from URDF_JOINT_NAMES matching _ARM_PATTERN
        # This matches how they are identified in the experiment config via ARM_PATTERN
        self.arm_motor_indices = [i for i, n in enumerate(URDF_JOINT_NAMES) if re.search(_ARM_PATTERN, n)]
        self.arm_joint_names = [URDF_JOINT_NAMES[i] for i in self.arm_motor_indices]
        _log.info(f"[ARM] Tracking enabled for {len(self.arm_motor_indices)} joints: {self.arm_joint_names}")

        # SAFETY PATCH (07-25): arm motor-dropout
        # watchdog state (see _WATCHDOG_STALE_TICKS at the module head for the
        # incident rationale). All numpy on purpose: the check runs in the
        # 50 Hz hot path and 14-element numpy ops are cheap and device-free.
        self._wd_prev = None               # (2,14) last (pos, torque) of arm joints
        self._wd_prev_pos31 = None         # (31,) last full pos — the liveness reference
        self._wd_consec = np.zeros(14, dtype=int)   # per-arm-joint frozen streak
        self._wd_stale_since = np.zeros(14)         # wall time each streak began
        self._wd_last_t = None                      # last QUALIFIED check time
        self._arm_fault = [False, False]   # [left, right] — TERMINAL once set
        self._arm_fault_hold = np.zeros(14, dtype=np.float32)  # pinned pose
        self._wd_condemned = np.zeros(14, dtype=bool)  # ever went stale — its
                                           # hold tracks measured FOREVER (a
                                           # reconnecting motor must meet zero
                                           # position error, never a snap-back)
        self._wd_idx = np.asarray(self.arm_motor_indices)
        self._wd_idx_t = torch.tensor(self.arm_motor_indices,
                                      dtype=torch.long, device=self.device)
        # 07-25b MODE trigger state (per arm): streak of qualified ticks on
        # which any joint of the arm reports firmware mode != Running.
        self._wd_mode_consec = np.zeros(2, dtype=int)
        self._wd_mode_since = np.zeros(2)
        # An UNPOWERED arm (bench profiles: enable_motor camera/leg/false) can
        # legitimately freeze bit-for-bit when it hangs still — a disabled
        # motor reports a constant torque word and a motionless encoder. The
        # watchdog exists to stop an arm being COMMANDED against a dropout;
        # with the arm motors unpowered there is nothing to protect and every
        # latch would be a false positive, so it disarms itself.
        if self.arm_watchdog and str(self.enable_motor) not in (
                "true", "True", "arm", "arm_camera"):
            self.arm_watchdog = False
            _log.info(f"[ARM] dropout watchdog DISARMED: arm motors not "
                      f"enabled (enable_motor={self.enable_motor!r})")
        if self.arm_watchdog:
            _log.info(f"[ARM] dropout watchdog ON: freeze >= "
                      f"{_WATCHDOG_STALE_TICKS} ticks -> damped hold "
                      f"(--no-arm-watchdog to disable)")

        # 2. Set static target to default pose (Home pose)
        # Randomization is disabled for safety on the real robot.
        self.current_target_arm_pos = self.default_joint_pos[self.arm_motor_indices].clone()
        self._arm_integral = torch.zeros(len(self.arm_motor_indices),
                                         dtype=torch.float32, device=self.device)
        self._arm_int_sat_ticks = torch.zeros_like(self._arm_integral)
        self._arm_int_sat_warn_step = -10**9
        self._arm_cmd_buf = torch.zeros(len(self.arm_motor_indices),
                                        dtype=torch.float32, device=self.device)
        # PROTOCOL (07-20): per-arm rate-limit step vector
        # (left 7 | right 7) for the mixed-mode schema.
        self._arm_step_buf = torch.zeros(len(self.arm_motor_indices),
                                         dtype=torch.float32, device=self.device)
        _log.info(f"[ARM] Using static arm target from default pose")

    def _reset_arm_targets(self):
        """Reset arm targets to default pose on episode reset."""
        self.current_target_arm_pos = self.default_joint_pos[self.arm_motor_indices].clone()
        self._arm_integral.zero_()
        self._arm_int_sat_ticks.zero_()


    def _apply_arm_gains(self, kp=None, kd=None):
        """Apply arm gains without restarting the env.

        Supports teach-replay workflow: teach mode sends low kp/kd so the human can
        backdrive the arm; replay mode sends higher kp/kd to track recorded keyframes
        stiffly; restore on exit returns to config defaults. Unspecified values stay
        latched from the previous call.

        _arm_kp / _arm_kd are the authoritative CUDA tensor caches.
        motor.loc_kp / spd_kp are numpy (C extension); conversion happens here, once.
        """
        def _expand(value, current):
            gain = torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten()
            if gain.numel() == 1:
                return torch.full_like(current, gain.item())
            if gain.numel() != len(self.arm_motor_indices):
                raise ValueError(
                    f"Expected scalar or {len(self.arm_motor_indices)} arm gains, got {gain.numel()}"
                )
            return gain

        self._arm_integral.zero_()
        self._arm_int_sat_ticks.zero_()
        if kp is not None:
            self._arm_kp = _expand(kp, self._arm_kp)
            self.motor.loc_kp[self.arm_motor_indices] = self._arm_kp.cpu().numpy()
        if kd is not None:
            self._arm_kd = _expand(kd, self._arm_kd)
            self.motor.spd_kp[self.arm_motor_indices] = self._arm_kd.cpu().numpy()

    # PATCH (07-16): per-group torque limits.
    # set_max_torque_ratio is a single global scalar, but the driver's setParam
    # already takes a PER-MOTOR array — so cap the 4 cam-gimbal motors at their
    # bench-tuned sweet spot (0.04-0.05, 07-14) while arms/legs keep the CLI
    # ratio. At a global 0.8 the gimbal had 20x its sweet-spot authority and
    # executed every reference step + policy-residual twitch as visible jitter.
    _CAM_TORQUE_RATIO = None           # None → cams FOLLOW the CLI ratio (07-17,
                                       # user spec: cams back to 80% with the body.
                                       # The 07-16 jitter that motivated the 0.05
                                       # cap traced to the reference staircase +
                                       # operator tick jitter, both fixed since:
                                       # gaze low-pass, gate throttle, trim gain).
                                       # Set a float (e.g. 0.05) to re-cap.
    _CAM_MOTOR_SLICE = slice(27, 31)   # cam yaw/pitch L, cam yaw/pitch R
    # PATCH (07-17): PIN THE WAIST. During standing grasp
    # missions the policy wiggles the waist (yaw) for heading/balance and the
    # WHOLE upper body — both arms, both cameras, every body-frame target —
    # yaws with it, destabilizing the reach. Yaw contributes little to standing
    # balance, so the waist position target is pinned to the default (0 rad)
    # while the motor stays enabled (PD-held, not passive). Set False to give
    # the waist back to the policy — walking/turning wants it.
    _PIN_WAIST = False                 # 07-19 (user): waist returned to the
                                       # policy — set True to pin it again
                                       # for waist-stable standing grasps
    _WAIST_IDX = 0                     # waist_joint in URDF joint order

    def _apply_group_torque_limits(self):
        """Re-assert per-group torque caps (call after any set_max_torque_ratio)."""
        import numpy as _np
        from hardware_bindings.motor.py_motor import MOTO_PARAM
        ratios = _np.full(self.n_joints, self.torque_limit, dtype=_np.float32)
        cam_ratio = self.torque_limit if self._CAM_TORQUE_RATIO is None \
            else min(self.torque_limit, self._CAM_TORQUE_RATIO)
        ratios[self._CAM_MOTOR_SLICE] = cam_ratio
        self.motor.setParam(MOTO_PARAM.torque_limit.value,
                            self.motor.max_motor_torques * ratios)
        _log.info(f"[TORQUE] per-group limits: body {self.torque_limit*100:.0f}% | "
                  f"cams {cam_ratio*100:.0f}%")

    def _apply_gaze_command(self, values):
        """Clamp one gaze_targets payload to cam ROM and latch it as the gaze reference.

        values: sequence of len(cam joints) angles [rad] in URDF cam-joint order
        (cam_yaw_left, cam_pitch_left, cam_yaw_right, cam_pitch_right). Out-of-limit
        values are clamped to the MJCF joint limits. Wrong shape → warn and drop.
        The cam joint target is gaze_reference + scale*action (residual); this sets gaze_reference.
        """
        if not self.gaze_in_obs:
            if not getattr(self, "_gaze_unsupported_warned", False):
                self._gaze_unsupported_warned = True
                _log.warning("[GAZE] gaze_targets received but policy has no "
                             "target_camera_joint_pos observation — ignored")
            return
        try:
            g = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            if g.shape != self.gaze_reference.shape:
                raise ValueError(f"expected {tuple(self.gaze_reference.shape)} angles, got {tuple(g.shape)}")
            if not torch.isfinite(g).all():
                # A NaN survives clamp and would poison the policy obs and every motor
                # reference (gaze_reference feeds both), unrecoverably — reject the packet.
                raise ValueError(f"non-finite gaze angles: {values}")
            # PATCH (07-16): latch the RAW target; the
            # low-pass in _update_arm_targets glides gaze_reference toward it.
            # Applying the 20 Hz staircase directly made a strong (0.8-ratio)
            # gimbal snap-and-wait at every step — visible jitter + motion blur
            # that degrades tag detection. (Reference smoothing also matches
            # sim, where the reference is continuous.)
            self._gaze_ref_target = torch.clamp(g, self._cam_joint_lower, self._cam_joint_upper)
            # Stamped only on a packet that APPLIED: a sender emitting malformed
            # gaze at 20 Hz would otherwise hold the failsafe off forever while
            # commanding nothing — the exact state the failsafe exists to end.
            self._last_gaze_recv_time = time.time()
        except Exception as e:
            _log.warning(f"[GAZE] Malformed gaze_targets packet: {e}")

    def _gaze_failsafe_step(self, now: float) -> bool:
        """Drive the gaze reference back to the origin while the stream is silent.

        Returns True if the failsafe is engaged this tick (stream stale).

        The gaze reference used to be latched latest-value-wins with NO timeout —
        "a frozen gaze is benign" was true only while the operator owned the run.
        It is not true on exit: Ctrl+C left both eyes frozen mid-sweep, pitched
        47 deg down and yawed wherever the scan stopped, so the robot sat staring
        at a cube it had already picked up and every later launch started from an
        unknown gimbal pose. Symmetric with the arm silence failsafe
        (_ARM_CMD_TIMEOUT / _ARM_RETURN_RATE) and for the same reason: the
        commanded reference must decay to a known posture when nobody is
        commanding it. Covers a crash or a kill -9 as well as a clean Ctrl+C —
        the operator cannot be trusted to park the gimbals on its way out.

        Rate-limits the RAW target (_gaze_ref_target); the 100 ms low-pass above
        then glides gaze_reference after it, so the return is smooth end to end
        and the gimbal never sees a step. A scan can leave yaw several rad from
        zero (cam ROM is +/-4.71 rad, and the sweep deliberately runs off the
        principal branch), so this is a plain joint lerp with no wrapping — the
        long way round is the way the joint actually got there.
        """
        stale = (now - self._last_gaze_recv_time) >= _GAZE_CMD_TIMEOUT
        if stale:
            if self._gaze_stream_active:
                self._gaze_stream_active = False
                _log.info("[GAZE] stream silent — parking the cameras at the "
                          f"origin at {_GAZE_RETURN_RATE} rad/s")
            max_step = _GAZE_RETURN_RATE * self.dt
            self._gaze_ref_target += (self._cam_home - self._gaze_ref_target
                                      ).clamp(-max_step, max_step)
        elif not self._gaze_stream_active:
            self._gaze_stream_active = True
            _log.info("[GAZE] stream live — the operator owns the cameras")
        return stale

    def _ik_mode_reseed(self, reason: str) -> None:
        """PROTOCOL (07-20): one-shot IK re-seed from encoders,
        shared by the legacy both-arms mode switch, the per-arm joint->Cartesian
        edges, and the mixed-mode stream-stale falling edge. Resets the solver's
        warm start to the measured configuration and snaps EE targets to the
        measured pose (position from FK; orientation target IDENTITY — callers
        that stream quats re-assert them within one packet) so the first solve
        after a joint-stream excursion cannot command a cross-branch jump.
        Serialized against the hot IK worker via _ik_solver_lock (07-20 review:
        an unserialized reset raced the in-flight solve and could publish one
        stale-warm-start result after the reseed)."""
        _log.info(f"[ARM] {reason}: re-seeding IK from encoders")
        self.base_quat_wxyz = torch.tensor(
            self.imu.transformed_quat_wxyz, dtype=torch.float32, device=self.device)
        self._ik_input_qpos.copy_(self._ik_get_physics_qpos(), non_blocking=True)
        with self._ik_solver_lock:
            self.ik_solver.reset(torch.tensor([0], dtype=torch.long, device=self.device),
                                 self._ik_input_qpos)
            self._ik_snap_orientation_targets()

    def _update_arm_targets(self):
        """Read latest arm command from arm_receiver and update gains/targets.

        Freshness gate: if no valid packet arrives within _ARM_CMD_TIMEOUT seconds,
        desired target becomes the default pose. Arm gains are latched and do not
        reset on timeout. Latest-value-wins semantics; packet is already the newest
        (NNGSubscriber drains queue on each recv).

        In all cases current_target_arm_pos is stepped toward desired at most
        _ARM_RETURN_RATE rad/s per joint, preventing dangerous jumps on timeout,
        stream restore, or sudden large command changes.

        Expected packet schema (dual schema — Option A)::

            Joint-space:
                {"kp": float | [float x N_arm_joints],
                 "kd": float | [float x N_arm_joints],
                 "arm_targets": {"joint_pos": [float x N_arm_joints]} | None}

            Cartesian (IK) — requires use_ik=True:
                {"arm_targets": {"ee_pos":  [[x,y,z], [x,y,z]],   # left, right in body frame
                                 "ee_quat": [[w,x,y,z], [w,x,y,z]]}}

            Gaze (any packet may carry it, alone or alongside arm keys):
                {"gaze_targets": [yaw_l, pitch_l, yaw_r, pitch_r]}  # rad, cam-joint order

        Joint-space targets are applied directly after rate-limiting.
        Cartesian targets update ee_pos_targets / ee_quat_targets; IK runs in step().
        N_arm_joints matches len(self.arm_motor_indices).
        gaze_targets sets the policy's gaze reference (clamped to cam joint limits);
        held latest-value-wins, no timeout — a frozen gaze is benign.
        """
        rx = self.arm_receiver
        now = time.time()

        if rx.data_id != self._last_arm_data_id:
            self._last_arm_data_id = rx.data_id
            data = rx.data
            # Arm freshness tracks ARM content ONLY. gaze_targets rides the same 9874
            # socket but must NOT refresh the arm timer: a gaze-only keepalive (streamed
            # every tick during scan/walk) would otherwise keep the arm-silence
            # ramp-to-default failsafe from ever firing, latching a stale Cartesian
            # target (IK holds it while walking).
            if data is not None and any(k in data for k in ("arm_targets", "kp", "kd")):
                self._last_arm_recv_time = now

            if self.ee and data is not None and "ee_action" in data:
                # Expected: {"side": "left"|"right", "command": "hand_grab"|...,
                # optional "width_mm"} — grippers are commanded per side.
                act = data["ee_action"]
                if isinstance(act, dict):
                    self.ee.send_action(act.get("side", ""), act.get("command", ""),
                                        act.get("width_mm"),
                                        act.get("torque"),
                                        action_id=act.get("action_id"))
                else:
                    _log.warning(f"[EE] ee_action must be a dict with side+command, "
                                 f"got {act!r} — dropped")
            # SAFETY PATCH (07-16): a gripper command
            # arriving while EE is disabled used to vanish with ZERO logging —
            # the operator's whole grasp mission ran against a void (07-16
            # audit, the incident class). Warn loudly, throttled to 5 s.
            elif data is not None and "ee_action" in data:
                if (now - getattr(self, "_ee_drop_warn_t", 0.0)) > 5.0:
                    self._ee_drop_warn_t = now
                    _log.warning("[EE] operator is sending gripper commands but EE "
                                 "is DISABLED (--ee-service off, or the request "
                                 "socket is owned by another commander) — commands "
                                 "are being DROPPED")
            if data is not None and "gaze_targets" in data:
                self._apply_gaze_command(data["gaze_targets"])

        # PATCH (07-16): gaze low-pass (tau = 100 ms).
        # Turns the 20 Hz reference staircase (and acquisition jumps) into a
        # continuous exponential glide at the 50 Hz step rate — constant-velocity
        # scanning instead of snap-and-wait. ~0.05 rad steady lag at scan speed;
        # the operator's closed-form aim recomputes every tick and absorbs it.
        if self.gaze_in_obs:
            self._gaze_failsafe_step(now)
            _alpha = min(1.0, self.dt / 0.10)
            self.gaze_reference += (self._gaze_ref_target - self.gaze_reference) * _alpha

        arm_fresh = (now - self._last_arm_recv_time) < _ARM_CMD_TIMEOUT
        self._arm_stream_active = arm_fresh
        if not arm_fresh:
            # PROTOCOL (07-20, review CRITICAL fix): stream
            # died during MIXED per-arm mode -> the joint-mode arm's IK targets
            # are stale (solver converged on them all through the staircase) and
            # arm_ik_active STAYS True through this falling edge, so the rising-
            # edge detector in step() never fires and the full both-arm apply
            # would whip the staircase arm back to its pre-staircase target in
            # one 50 Hz step. Reseed ONCE here (carts become equal below, so
            # this cannot refire) — failsafe = hold the measured pose.
            if self.use_ik and self._arm_cart[0] != self._arm_cart[1]:
                self._ik_mode_reseed("stream stale during mixed per-arm mode — "
                                     "failsafe holds the measured pose")
            self._arm_stream_is_cartesian = False
            self._arm_cart = [False, False]                    # PROTOCOL (07-20)
            self._arm_rates = [_ARM_RETURN_RATE, _ARM_RETURN_RATE]
            desired = self.default_joint_pos[self.arm_motor_indices]
            max_step = _ARM_RETURN_RATE * self.dt
            self.current_target_arm_pos += (desired - self.current_target_arm_pos).clamp(-max_step, max_step)
            return

        if rx.data is not None:
            try:
                if "kp" in rx.data or "kd" in rx.data:
                    # 07-25 review #5: while the dropout watchdog holds an arm,
                    # its only remaining support is the PD gains — a teach-mode
                    # low-gain packet arriving mid-fault would silently strip
                    # the damped hold. Gains apply to both arms in one vector,
                    # so refuse the whole packet while any arm is latched.
                    if self._arm_fault[0] or self._arm_fault[1]:
                        if time.time() - getattr(self, "_wd_gain_warn_t", -1e9) > 5.0:
                            self._wd_gain_warn_t = time.time()
                            _log.warning(
                                "[ARM] kp/kd packet REFUSED: watchdog holds "
                                "a faulted arm and its PD gains are the hold")
                    else:
                        self._apply_arm_gains(kp=rx.data.get("kp"), kd=rx.data.get("kd"))
            except Exception as e:
                _log.warning(f"[ARM] Malformed arm gains packet: {e}")

        # Determine desired target; default: hold current (malformed/no-op cases)
        desired = self.current_target_arm_pos
        if rx.data is not None and "arm_targets" in rx.data:
            try:
                payload = rx.data["arm_targets"]
                if payload is None:
                    # Sender explicitly cleared targets (e.g. tracking lost)
                    self._arm_stream_is_cartesian = False
                    self._arm_cart = [False, False]            # PROTOCOL (07-20)
                    self._arm_rates = [_ARM_RETURN_RATE, _ARM_RETURN_RATE]
                    self._arm_stream_rate = _ARM_RETURN_RATE   # clear → failsafe crawl
                    desired = self.default_joint_pos[self.arm_motor_indices]
                elif "ee_pos" in payload and "ee_quat" in payload:
                    # Cartesian schema: update IK targets; IK overwrites arm joints in step().
                    # Requires use_ik=True; ignored with a warning otherwise.
                    if self.use_ik:
                        if not (self._arm_cart[0] and self._arm_cart[1]):
                            # Mode switch (joint/none/mixed -> Cartesian): the
                            # solver's internal ARM state is stale — solve()
                            # deliberately never re-reads arm encoders
                            # (streaming-IK keeps the command trajectory
                            # decoupled from tracking error), so after a
                            # joint-stream excursion (staged side/rear transit)
                            # the warm start still sits where the arm was BEFORE
                            # the excursion and the first solve can command a
                            # cross-branch jump. Re-seed from encoders once,
                            # exactly like the startup/reset snap.
                            self._ik_mode_reseed("stream mode -> Cartesian (both arms)")
                        # (07-20 review): NaN/shape guard — a
                        # poisoned IK target survives every downstream clamp.
                        _p = torch.as_tensor(payload["ee_pos"], dtype=torch.float32)
                        _q = torch.as_tensor(payload["ee_quat"], dtype=torch.float32)
                        if _p.shape != (2, 3) or _q.shape != (2, 4) or \
                                not bool(torch.isfinite(_p).all()) or \
                                not bool(torch.isfinite(_q).all()):
                            _log.warning("[ARM] Malformed Cartesian targets "
                                         f"({tuple(_p.shape)}/{tuple(_q.shape)} or "
                                         "NaN) — packet ignored")
                            return
                        self.ee_pos_targets[0]  = _p.to(self.device)
                        self.ee_quat_targets[0] = _q.to(self.device)
                        if "orientation_cost" in payload:
                            self.ik_solver.set_orientation_cost(payload["orientation_cost"])
                        self._arm_stream_is_cartesian = True
                        self._arm_cart = [True, True]          # PROTOCOL (07-20)
                    else:
                        _log.warning("[ARM] Cartesian targets received but use_ik=False — ignored")
                    return  # skip joint-space rate-limit; IK handles targets in step()
                # PROTOCOL (07-20, approved upstream): PER-ARM
                # schema {"left"|"right": {ee_pos+ee_quat} | {joint_pos[7](+rate)}}.
                # Each side picks its own mode; an absent side holds its current
                # mode and targets (its desired slice stays = current, a no-op
                # under the rate limiter). This is what lets one arm walk a joint
                # staircase while the other keeps its live Cartesian stream.
                elif ("left" in payload) or ("right" in payload):
                    self._arm_cmd_buf.copy_(self.current_target_arm_pos)
                    # PASS 1 (07-20 review fix): validate every present side into
                    # temporaries BEFORE touching any solver/target state. A bad
                    # slot must neither abort the other side nor trigger a reseed
                    # (this method re-parses the LATCHED packet at 50 Hz, so an
                    # unvalidated reseed-then-raise looped the reseed every tick,
                    # pinning the healthy arm at snap targets). NaN anywhere is
                    # rejected — clamp passes NaN through and the rate limiter
                    # makes a poisoned current_target_arm_pos PERMANENT (same
                    # guard pattern as _apply_gaze_command).
                    slots: dict[int, tuple] = {}
                    for i, side in enumerate(("left", "right")):
                        sp = payload.get(side)
                        if sp is None:
                            continue                           # absent side: hold
                        try:
                            if not isinstance(sp, dict):
                                raise ValueError(f"slot is {type(sp).__name__}")
                            if "ee_pos" in sp and "ee_quat" in sp:
                                if not self.use_ik:
                                    raise ValueError("Cartesian but use_ik=False")
                                p = torch.as_tensor(sp["ee_pos"], dtype=torch.float32)
                                q = torch.as_tensor(sp["ee_quat"], dtype=torch.float32)
                                if p.shape != (3,) or q.shape != (4,) or \
                                        not bool(torch.isfinite(p).all()) or \
                                        not bool(torch.isfinite(q).all()):
                                    raise ValueError("bad ee_pos/ee_quat shape or NaN")
                                slots[i] = ("cart", p.to(self.device), q.to(self.device))
                            elif sp.get("joint_pos") is not None:
                                jp = torch.as_tensor(sp["joint_pos"], dtype=torch.float32)
                                rate = float(sp.get("rate", _ARM_DEFAULT_PKT_RATE))
                                if jp.shape != (7,) or \
                                        not bool(torch.isfinite(jp).all()) or \
                                        not math.isfinite(rate):
                                    raise ValueError("bad joint_pos/rate shape or NaN")
                                slots[i] = ("joint", jp.to(self.device),
                                            min(max(rate, 0.0), 1.05))
                            else:
                                raise ValueError(f"unknown keys {list(sp.keys())}")
                        except (ValueError, TypeError) as e:
                            if (now - getattr(self, "_arm_slot_warn_t", 0.0)) > 5.0:
                                self._arm_slot_warn_t = now
                                _log.warning(f"[ARM] Malformed {side} slot ({e}) — held")
                    # PASS 2: at most ONE reseed per packet, BEFORE any slot is
                    # assigned — so the (solver-global) snap can never clobber a
                    # target written from this same packet. The peer's slot, if
                    # absent, is re-asserted by its next packet (<=50 ms).
                    if any(kind == "cart" and not self._arm_cart[i]
                           for i, (kind, *_r) in slots.items()):
                        flip = [("left", "right")[i] for i, (k, *_r) in slots.items()
                                if k == "cart" and not self._arm_cart[i]]
                        self._ik_mode_reseed(f"{'+'.join(flip)} arm -> Cartesian")
                    for i, slot in slots.items():
                        if slot[0] == "cart":
                            self.ee_pos_targets[0, i] = slot[1]
                            self.ee_quat_targets[0, i] = slot[2]
                            self._arm_cart[i] = True
                        else:
                            sl = slice(0, 7) if i == 0 else slice(7, 14)
                            torch.clamp(
                                slot[1],
                                self.joint_lower[self.arm_motor_indices][sl],
                                self.joint_upper[self.arm_motor_indices][sl],
                                out=self._arm_cmd_buf[sl],
                            )
                            self._arm_rates[i] = slot[2]
                            self._arm_cart[i] = False
                    self._arm_stream_is_cartesian = \
                        self._arm_cart[0] and self._arm_cart[1]
                    desired = self._arm_cmd_buf
                    # NOTE: while an arm is joint-mode, the solver's internal copy
                    # of that arm diverges from reality (it never re-reads its
                    # encoders); its IK output is masked off in step(), and the
                    # per-arm reseed above repairs the divergence on that arm's
                    # return to Cartesian. Bounded: staircases may now run in
                    # PARALLEL (07-21 PHASE 2) and half-space separation is
                    # certified by the mid-sagittal audit (>= 239.9 mm), not by
                    # serialization.
                else:
                    self._arm_stream_is_cartesian = False
                    self._arm_cart = [False, False]            # PROTOCOL (07-20)
                    joint_pos = payload.get("joint_pos")
                    _jp14 = None
                    if joint_pos is not None and len(joint_pos) == len(self.arm_motor_indices):
                        _jp14 = torch.as_tensor(joint_pos, dtype=torch.float32)
                        # (07-20 review): NaN guard — clamp
                        # passes NaN and the rate limiter makes it permanent.
                        if not bool(torch.isfinite(_jp14).all()):
                            _log.warning("[ARM] NaN in joint_pos — packet ignored")
                            _jp14 = None
                    if _jp14 is not None:
                        torch.clamp(
                            _jp14,
                            self.joint_lower[self.arm_motor_indices],
                            self.joint_upper[self.arm_motor_indices],
                            out=self._arm_cmd_buf,
                        )
                        desired = self._arm_cmd_buf
                        # Commanded-rate knob (staged-reach transit poses): a joint
                        # packet may carry "rate" [rad/s], hard-capped at 1.05 —
                        # raised from 0.5 to 0.7 to 1.05 (user 2026-08-02, arm
                        # speed x1.5 again) to match the StepGate rate ceiling,
                        # which stays the outer bound on every commanded step. Absent
                        # field → the classic failsafe crawl, so every existing
                        # sender behaves exactly as before. Per-packet: silence and
                        # explicit-clear paths always reset to _ARM_RETURN_RATE.
                        # (07-20): floor at 0 — a negative
                        # rate flips clamp(min>max) into a constant full-step
                        # drift on every joint regardless of target. NaN rate
                        # falls back to the crawl (min/max pass NaN through).
                        _r = float(payload.get("rate", _ARM_DEFAULT_PKT_RATE))
                        if not math.isfinite(_r):
                            _r = _ARM_DEFAULT_PKT_RATE
                        self._arm_stream_rate = min(max(_r, 0.0), 1.05)
                        self._arm_rates = [self._arm_stream_rate] * 2  # PROTOCOL (07-20)
                    # else malformed — hold (desired stays as current)
            except Exception as e:
                _log.warning(f"[ARM] Malformed arm_targets packet: {e}")
        # Rate-limit step: safe for timeout, stream-restore, and sudden command jumps
        # (joint-stream packets may raise the rate via their "rate" field; the
        # stream-silence timeout ramp above keeps the hardcoded failsafe crawl).
        # PROTOCOL (07-20): per-arm step vector — under the
        # mixed schema each side tracks at its own commanded rate.
        self._arm_step_buf[:7].fill_(self._arm_rates[0] * self.dt)
        self._arm_step_buf[7:].fill_(self._arm_rates[1] * self.dt)
        self.current_target_arm_pos += torch.clamp(
            desired - self.current_target_arm_pos,
            -self._arm_step_buf, self._arm_step_buf)

    def _build_default_joint_pos(self) -> torch.Tensor:
        """Build default joint positions in URDF order."""
        joint_pos_config = self.config["default_pose"]["joint_pos"]

        # Start with catch-all default
        default_value = joint_pos_config.get(".*", 0.0)
        default_pos = torch.full((self.n_joints,), default_value, dtype=torch.float32)

        # Override with specific patterns
        for joint_name, value in joint_pos_config.items():
            if joint_name == ".*":
                continue
            for idx, urdf_name in enumerate(URDF_JOINT_NAMES):
                if re.match(f"^{joint_name}$", urdf_name):
                    default_pos[idx] = value

        return default_pos.to(self.device)

    def _build_actuator_params(self) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Build action scale and PD gains in URDF order."""
        scales, kp_list, kd_list = [], [], []
        for joint_name in URDF_JOINT_NAMES:
            actuator = next(
                (a for a in self.config["actuators"] for p in a["joints"] if re.match(f"^{p}$", joint_name)),
                None,
            )
            if actuator:
                scales.append(actuator["action_scale"])
                kp_list.append(actuator["kp"])
                kd_list.append(actuator["kd"])
            elif joint_name.startswith("cam_"):
                # Cam motor present on the robot but absent from this policy's config
                # (non-cam policy). Hold it at default pose with robot-level gains.
                scales.append(0.0)
                kp_list.append(_CAM_HOLD_KP)
                kd_list.append(_CAM_HOLD_KD)
            else:
                raise RuntimeError(f"No actuator config for joint: {joint_name}")
        _log.debug(f"[DEBUG] Built actuator params for {len(scales)} joints")
        return (
            torch.tensor(scales, dtype=torch.float32, device=self.device),
            np.array(kp_list, dtype=np.float32),
            np.array(kd_list, dtype=np.float32),
        )

    def prepare(self, duration=2.0):
        """Move to config default pose with smooth interpolation.

        Thin wrapper: passes config-derived gains and target to HumanoidBase.prepare().
        Called by main() as env.prepare(duration=3.0) — signature unchanged.
        """
        super().prepare(self.default_joint_pos, self.motor_kp, self.motor_kd, duration)

    # =========================================================================
    # IK (Cartesian Arm Control)
    # =========================================================================

    def _setup_ik(self):
        """Initialize IK solver for body-frame Cartesian arm control.

        IK runs in a background daemon thread to avoid blocking the control loop.
        Main thread posts sensor snapshots each step and reads the previous step's result
        (1-step latency). MuJoCo C extension releases the GIL during mj_kinematics, so
        the worker thread runs in true parallel with the main thread.

        Shared state:
          _ik_input_*      written by main under _ik_input_lock, read by worker
          _ik_output       written by worker under _ik_output_lock, read by main
          _ik_solver_lock  serializes worker vs _ik_snap_orientation_targets (both call ik_solver.solve)
        """
        arm_pat = re.compile(r"_(shoulder|elbow|wrist)_")
        all_arm     = [j for j in URDF_JOINT_NAMES if arm_pat.search(j)]
        left_names  = [j for j in all_arm if j.startswith("left_")]
        right_names = [j for j in all_arm if j.startswith("right_")]
        _log.info(f"[IK] Left arm joints ({len(left_names)}): {left_names}")
        _log.info(f"[IK] Right arm joints ({len(right_names)}): {right_names}")

        self.ee_pos_targets  = torch.zeros(1, 2, 3, device=self.device)
        self.ee_quat_targets = torch.zeros(1, 2, 4, device=self.device)
        self.ee_quat_targets[..., 0] = 1.0

        # Warp backend: auto-detected from wp_model (non-None → warp).
        _ik_wp_model = mjwarp.put_model(self.mj_model)
        self.ik_solver = BatchedAnalyticalIK(
            self.mj_model, _ik_wp_model, 1, str(self.device),
            _EE_LEFT_NAME, _EE_RIGHT_NAME, left_names, right_names,
            ee_left_type=_EE_LEFT_TYPE, ee_right_type=_EE_RIGHT_TYPE,
            step_dt=self.dt,
        )
        # Store in [left..., right...] order — matches IK output layout.
        # Used by _setup_inv_dyn() to build MuJoCo DOF address mapping.
        self._ik_arm_joint_names = left_names + right_names
        self._ik_arm_urdf_idx = torch.tensor(
            [URDF_JOINT_NAMES.index(j) for j in left_names + right_names],
            dtype=torch.long, device=self.device,
        )
        self._ik_arm_urdf_idx_np = self._ik_arm_urdf_idx.cpu().numpy()  # cached for hot-path indexing

        # IK limits stay on CUDA; avoid GPU->CPU->GPU round trip on hot path.
        self._ik_joint_lower = self.joint_lower[self._ik_arm_urdf_idx]
        self._ik_joint_upper = self.joint_upper[self._ik_arm_urdf_idx]

        # Map IK joints (_ik_arm_urdf_idx order) → positions in current_target_arm_pos
        # (arm_motor_indices order). Used to keep current_target_arm_pos in sync with
        # ik_out each step so the rate-limiter starts from the true arm position on
        # STOP/timeout rather than snapping from default_joint_pos.
        arm_urdf_to_pos = {v: i for i, v in enumerate(self.arm_motor_indices)}
        self._ik_in_arm_idx = torch.tensor(
            [arm_urdf_to_pos[v.item()] for v in self._ik_arm_urdf_idx.cpu()],
            dtype=torch.long, device=self.device,
        )

        # Reusable staging buffers: avoid per-step tensor allocations in IK hot path.
        self._ik_qpos_cpu = torch.zeros(1, self.mj_model.nq, dtype=torch.float32)
        self._ik_imu_quat_np = np.zeros(4, dtype=np.float32)
        self._ik_motor_pos_np = np.zeros(self.n_joints, dtype=np.float32)
        self._ik_imu_quat_cpu = torch.from_numpy(self._ik_imu_quat_np)
        self._ik_motor_pos_cpu = torch.from_numpy(self._ik_motor_pos_np)
        self._ik_actual_pos_np = np.zeros((1, 2, 3), dtype=np.float32)
        self._ik_actual_quat_np = np.zeros((1, 2, 4), dtype=np.float32)
        self._ik_actual_quat_np[..., 0] = 1.0
        self._ik_site_quat_f64 = np.empty(4, dtype=np.float64)  # mju_mat2Quat requires float64
        self._ik_actual_pos_cpu = torch.from_numpy(self._ik_actual_pos_np)
        self._ik_actual_quat_cpu = torch.from_numpy(self._ik_actual_quat_np)

        # Shared input buffers (main thread writes, worker reads): all on CUDA.
        self._ik_input_lock      = threading.Lock()
        self._ik_input_qpos      = torch.zeros(1, self.mj_model.nq, dtype=torch.float32, device=self.device)
        self._ik_input_ee_pos    = torch.zeros_like(self.ee_pos_targets)
        self._ik_input_ee_quat   = torch.zeros_like(self.ee_quat_targets)
        self._ik_input_act_pos   = torch.zeros(1, 2, 3, dtype=torch.float32, device=self.device)
        self._ik_input_act_quat  = torch.zeros(1, 2, 4, dtype=torch.float32, device=self.device)
        self._ik_input_act_quat[..., 0] = 1.0
        self._ik_input_imu_quat  = torch.zeros(1, 4, dtype=torch.float32)  # CPU; wxyz IMU quat
        self._ik_input_imu_quat[0, 0] = 1.0

        # Worker-local snapshots: copied from shared inputs under lock.
        self._ik_work_qpos = torch.zeros_like(self._ik_input_qpos)
        self._ik_work_ee_pos = torch.zeros_like(self._ik_input_ee_pos)
        self._ik_work_ee_quat = torch.zeros_like(self._ik_input_ee_quat)
        self._ik_work_act_pos = torch.zeros_like(self._ik_input_act_pos)
        self._ik_work_act_quat = torch.zeros_like(self._ik_input_act_quat)
        self._ik_work_imu_quat = torch.zeros(1, 4, dtype=torch.float32)  # CPU; wxyz IMU quat
        self._ik_work_imu_quat[0, 0] = 1.0

        # Output buffers (worker writes, main reads).
        # _ik_output stays on CUDA for direct target_pos assignment.
        # dq/ddq stay on CPU for mj_inverse feedforward path (numpy backend).
        n_arm = len(self._ik_arm_joint_names)
        self._ik_output_lock = threading.Lock()
        self._ik_output = torch.zeros(n_arm, dtype=torch.float32, device=self.device)
        self._ik_out_step = torch.zeros(n_arm, dtype=torch.float32, device=self.device)
        self._ik_out_apply = torch.zeros(n_arm, dtype=torch.float32, device=self.device)
        self._ik_dq_output = torch.zeros(n_arm, dtype=torch.float32)
        self._ik_ddq_output = torch.zeros(n_arm, dtype=torch.float32)
        self._ik_dq_output_np = self._ik_dq_output.numpy()
        self._ik_ddq_output_np = self._ik_ddq_output.numpy()
        self._ik_dq_step_np = np.zeros(n_arm, dtype=np.float32)
        self._ik_ddq_step_np = np.zeros(n_arm, dtype=np.float32)

        # Serializes ik_solver.solve() between worker and _ik_snap_orientation_targets.
        self._ik_solver_lock = threading.Lock()

        self._ik_trigger = threading.Event()
        self._ik_stop    = threading.Event()
        self._ik_thread  = threading.Thread(target=self._ik_worker_loop, daemon=True)
        self._ik_thread.start()
        _log.info("[IK] Background solver thread started")
        self._setup_inv_dyn()

    def _ik_get_physics_qpos(self) -> torch.Tensor:
        """Build (1, nq) CPU tensor for IK FK using reusable staging buffers."""
        self._ik_qpos_cpu.zero_()
        np.copyto(self._ik_imu_quat_np, self.imu.transformed_quat_wxyz)
        np.copyto(self._ik_motor_pos_np, self.motor.mech_pos)
        self._ik_qpos_cpu[0, 3:7].copy_(self._ik_imu_quat_cpu)
        self._ik_qpos_cpu[0, 7:].copy_(self._ik_motor_pos_cpu)
        return self._ik_qpos_cpu

    def _ik_get_actual_ee_state(self):
        """Read actual EE pos/quat from mj_data into reusable CPU staging buffers.

        Output is in root-body frame: R_root.T @ (p_world - p_root), matching
        the frame expected by BatchedMinkIK.solve() actual_ee_pos/quat args.
        """
        lid, rid = self.ik_solver._ee_left_id, self.ik_solver._ee_right_id
        R_base = self.mj_data.xmat[self.root_body_id].reshape(3, 3).T  # R_world_from_base^T
        p_root = self.mj_data.xpos[self.root_body_id]
        l_pos = self.mj_data.site_xpos[lid] if _EE_LEFT_TYPE == "site" else self.mj_data.xpos[lid]
        l_mat = self.mj_data.site_xmat[lid] if _EE_LEFT_TYPE == "site" else self.mj_data.xmat[lid]
        r_pos = self.mj_data.site_xpos[rid] if _EE_RIGHT_TYPE == "site" else self.mj_data.xpos[rid]
        r_mat = self.mj_data.site_xmat[rid] if _EE_RIGHT_TYPE == "site" else self.mj_data.xmat[rid]
        self._ik_actual_pos_np[0, 0] = R_base @ (l_pos - p_root)
        mujoco.mju_mat2Quat(self._ik_site_quat_f64, (R_base @ l_mat.reshape(3, 3)).flatten())
        self._ik_actual_quat_np[0, 0] = self._ik_site_quat_f64
        self._ik_actual_pos_np[0, 1] = R_base @ (r_pos - p_root)
        mujoco.mju_mat2Quat(self._ik_site_quat_f64, (R_base @ r_mat.reshape(3, 3)).flatten())
        self._ik_actual_quat_np[0, 1] = self._ik_site_quat_f64
        return self._ik_actual_pos_cpu, self._ik_actual_quat_cpu  # (1, 2, 3), (1, 2, 4) CPU

    def _ik_worker_loop(self):
        """Background thread: solve IK from latest posted inputs, write clamped result.

        Triggered each control step via _ik_trigger. Reads a snapshot of sensor inputs
        (posted by main thread under _ik_input_lock), runs ik_solver.solve() under
        _ik_solver_lock (shared with _ik_snap_orientation_targets), then writes the
        clamped arm joint solution to _ik_output under _ik_output_lock.

        MuJoCo Warp (mjwarp.kinematics) releases the GIL, so this runs in true parallel
        with the main control thread. Warp backend state is private to this solver instance
        and serialized vs _ik_snap_orientation_targets via _ik_solver_lock.
        """
        while not self._ik_stop.is_set():
            if not self._ik_trigger.wait(timeout=0.01):
                continue
            self._ik_trigger.clear()

            with self._ik_input_lock:
                self._ik_work_qpos.copy_(self._ik_input_qpos)
                self._ik_work_ee_pos.copy_(self._ik_input_ee_pos)
                self._ik_work_ee_quat.copy_(self._ik_input_ee_quat)
                self._ik_work_act_pos.copy_(self._ik_input_act_pos)
                self._ik_work_act_quat.copy_(self._ik_input_act_quat)
                self._ik_work_imu_quat.copy_(self._ik_input_imu_quat)

            with self._ik_solver_lock:
                result = self.ik_solver.solve(
                    self._ik_work_qpos,
                    self._ik_work_ee_pos,
                    self._ik_work_ee_quat,
                    actual_ee_pos=self._ik_work_act_pos,
                    actual_ee_quat=self._ik_work_act_quat,
                    base_quat_wxyz=self._ik_work_imu_quat,
                )
                # Copy dq/ddq while solver lock is held — ik_solver.dq_target/ddq_target are
                # overwritten on each solve(), so reading them outside the lock is a data race.
                self._ik_dq_output.copy_(self.ik_solver.dq_target[0])
                self._ik_ddq_output.copy_(self.ik_solver.ddq_target[0])

            with self._ik_output_lock:
                self._ik_out_step.copy_(result[0])
                self._ik_out_step.clamp_(self._ik_joint_lower, self._ik_joint_upper)
                self._ik_output.copy_(self._ik_out_step)

    def _ik_snap_orientation_targets(self):
        """Seed IK EE targets from current arm pose on reset.

        Runs FK at current encoder positions to capture actual EE pos/quat,
        then seeds ee_pos_targets / ee_quat_targets so IK tracks from the
        current configuration without a jump. Also seeds _ik_output from
        encoder positions so step() has a valid result before the worker fires.
        """
        physics_qpos = self._ik_get_physics_qpos()
        self.ik_solver.reset(torch.tensor([0]), physics_qpos.to(self.device))

        # Run FK manually: reset() copies encoders into q_ik but does not call
        # mj_kinematics, so xpos/xquat are stale until we do it explicitly.
        # Warp backend has no _mj_datas; use a local scratch data for this one-shot FK.
        mj_d0 = mujoco.MjData(self.ik_solver.mj_model)
        mj_d0.qpos[:] = self.ik_solver.q_ik[0].cpu().numpy()
        mujoco.mj_kinematics(self.ik_solver.mj_model, mj_d0)
        lid, rid = self.ik_solver._ee_left_id, self.ik_solver._ee_right_id
        root_quat_inv = _quat_conjugate(self.base_quat_wxyz.cpu())

        # Seed position targets from FK EE positions (world → body frame).
        left_src  = mj_d0.site_xpos[lid] if _EE_LEFT_TYPE  == "site" else mj_d0.xpos[lid]
        right_src = mj_d0.site_xpos[rid] if _EE_RIGHT_TYPE == "site" else mj_d0.xpos[rid]
        left_p_w  = torch.tensor(left_src.copy(),  dtype=torch.float32)
        right_p_w = torch.tensor(right_src.copy(), dtype=torch.float32)
        self.ee_pos_targets[0, 0] = _quat_apply(root_quat_inv, left_p_w)
        self.ee_pos_targets[0, 1] = _quat_apply(root_quat_inv, right_p_w)

        # Seed orientation targets: identity quaternion in body frame
        # (EE aligns with root body orientation at rest, same as sim env)
        self.ee_quat_targets[0, 0] = torch.tensor([1., 0., 0., 0.], dtype=torch.float32)
        self.ee_quat_targets[0, 1] = torch.tensor([1., 0., 0., 0.], dtype=torch.float32)

        # Seed _ik_output from current encoders so step() applies no jump on first call.
        arm_enc = torch.tensor(self.motor.mech_pos, dtype=torch.float32, device=self.device)[self._ik_arm_urdf_idx]
        with self._ik_output_lock:
            self._ik_out_step.copy_(arm_enc)
            self._ik_out_step.clamp_(self._ik_joint_lower, self._ik_joint_upper)
            self._ik_output.copy_(self._ik_out_step)

    def _ik_step(self) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Post sensor snapshot to IK worker; return previous step's result.

        Must be called AFTER update_kinematics() so mj_data.xpos/xquat are fresh.
        Absorbs _ik_get_physics_qpos + _ik_get_actual_ee_state to avoid intermediate
        temporaries. Reads output, dq, ddq under a single _ik_output_lock acquisition
        to prevent torn reads (worker writes all three in one locked block).

        Returns (ik_out_clamped, dq_np, ddq_np) — 1-step latency on ik_out.
        """
        # Build physics qpos into pre-allocated staging buffers (no heap alloc).
        self._ik_qpos_cpu.zero_()
        np.copyto(self._ik_imu_quat_np, self.imu.transformed_quat_wxyz)
        np.copyto(self._ik_motor_pos_np, self.motor.mech_pos)
        self._ik_qpos_cpu[0, 3:7].copy_(self._ik_imu_quat_cpu)
        self._ik_qpos_cpu[0, 7:].copy_(self._ik_motor_pos_cpu)

        # Read actual EE pos/quat from mj_data, transformed to root-body frame.
        lid, rid = self.ik_solver._ee_left_id, self.ik_solver._ee_right_id
        R_base = self.mj_data.xmat[self.root_body_id].reshape(3, 3).T
        p_root = self.mj_data.xpos[self.root_body_id]
        l_pos = self.mj_data.site_xpos[lid] if _EE_LEFT_TYPE == "site" else self.mj_data.xpos[lid]
        l_mat = self.mj_data.site_xmat[lid] if _EE_LEFT_TYPE == "site" else self.mj_data.xmat[lid]
        r_pos = self.mj_data.site_xpos[rid] if _EE_RIGHT_TYPE == "site" else self.mj_data.xpos[rid]
        r_mat = self.mj_data.site_xmat[rid] if _EE_RIGHT_TYPE == "site" else self.mj_data.xmat[rid]
        self._ik_actual_pos_np[0, 0] = R_base @ (l_pos - p_root)
        mujoco.mju_mat2Quat(self._ik_site_quat_f64, (R_base @ l_mat.reshape(3, 3)).flatten())
        self._ik_actual_quat_np[0, 0] = self._ik_site_quat_f64
        self._ik_actual_pos_np[0, 1] = R_base @ (r_pos - p_root)
        mujoco.mju_mat2Quat(self._ik_site_quat_f64, (R_base @ r_mat.reshape(3, 3)).flatten())
        self._ik_actual_quat_np[0, 1] = self._ik_site_quat_f64

        if self.ik_solver.debug:
            desired_pos_body = self.ee_pos_targets.cpu()
            desired_quat_body = self.ee_quat_targets.cpu()
            print(f"[IK] desired_pos_body   L={desired_pos_body[0,0].tolist()}  R={desired_pos_body[0,1].tolist()}")
            print(f"[IK] desired_quat_body  L={desired_quat_body[0,0].tolist()}  R={desired_quat_body[0,1].tolist()}")
            print(f"[IK] actual_pos_world   L={self._ik_actual_pos_cpu[0,0].tolist()}  R={self._ik_actual_pos_cpu[0,1].tolist()}")
            print(f"[IK] actual_quat_world  L={self._ik_actual_quat_cpu[0,0].tolist()}  R={self._ik_actual_quat_cpu[0,1].tolist()}")

        # Post inputs to worker thread for next step's solve.
        with self._ik_input_lock:
            self._ik_input_qpos.copy_(self._ik_qpos_cpu)
            self._ik_input_ee_pos.copy_(self.ee_pos_targets)
            self._ik_input_ee_quat.copy_(self.ee_quat_targets)
            self._ik_input_act_pos.copy_(self._ik_actual_pos_cpu)
            self._ik_input_act_quat.copy_(self._ik_actual_quat_cpu)
            self._ik_input_imu_quat.copy_(self._ik_imu_quat_cpu)
        self._ik_trigger.set()

        # Read previous step's result + dq/ddq in one lock — prevents torn read.
        with self._ik_output_lock:
            self._ik_out_apply.copy_(self._ik_output)
            np.copyto(self._ik_dq_step_np, self._ik_dq_output_np)
            np.copyto(self._ik_ddq_step_np, self._ik_ddq_output_np)

        if self.ik_solver.debug:
            actual_arm_pos = torch.tensor(
                self.motor.mech_pos, dtype=torch.float32, device=self.device
            )[self._ik_arm_urdf_idx]
            print(f"[IK] desired_arm_joints (IK soln, clamped) {self._ik_out_apply.tolist()}")
            print(f"[IK] actual_arm_joints  (encoder)          {actual_arm_pos.tolist()}")

        return self._ik_out_apply, self._ik_dq_step_np, self._ik_ddq_step_np

    def _setup_inv_dyn(self) -> None:
        """Scratch MjData + pre-allocated buffers for per-step arm inverse dynamics (real robot).

        qpos populated from encoder+IMU via update_kinematics() before each call.
        qvel[arm]=dq_target (IK FD, clean); qvel[other]=0 (real hardware always zeroes qvel).
        Output applied to motor.mech_torque_ref[_ik_arm_urdf_idx].

        No vel-FF ctrl offset (unlike sim): real env sets mech_pos_ref directly, not via
        MuJocoPDActuator, so velocity FF is not applicable here.
        """
        self._inv_dyn_scratch = mujoco.MjData(self.mj_model)
        self._arm_dofadr = np.array(
            [self.mj_model.joint(n).dofadr[0] for n in self._ik_arm_joint_names],
            dtype=np.int64,
        )
        n_arm = len(self._arm_dofadr)
        self._qvel_scratch = np.zeros(self.mj_model.nv, dtype=np.float64)
        self._qacc_scratch = np.zeros(self.mj_model.nv, dtype=np.float64)
        self._tau_arm_np   = np.zeros(n_arm, dtype=np.float32)

    def _compute_arm_inv_dyn_cpu(
        self,
        mj_data: mujoco.MjData,
        dq_arm: np.ndarray,    # (2K,) desired arm velocity, IK joint order
        ddq_arm: np.ndarray,   # (2K,) desired arm acceleration, IK joint order
    ) -> np.ndarray:           # (2K,) float32 torques — reused buffer
        """Arm inverse dynamics via mj_inverse (RNEA) on isolated scratch MjData.

        Computes: τ_ff = M(q_actual)*ddq_arm + C(q_actual, dq_arm)*dq_arm + g(q_actual)

        mj_data must be pre-populated (qpos from encoder+IMU, qvel=0) by update_kinematics().
        Writes result into _tau_arm_np (float32). No heap allocation.
        """
        s = self._inv_dyn_scratch
        s.qpos[:] = mj_data.qpos
        # Option A: qvel[arm]=dq_target (desired); qvel[other]=0 (real hw always zeroes qvel).
        self._qvel_scratch[:] = mj_data.qvel
        self._qvel_scratch[self._arm_dofadr] = dq_arm
        s.qvel[:] = self._qvel_scratch
        self._qacc_scratch[:] = 0.0
        self._qacc_scratch[self._arm_dofadr] = ddq_arm
        s.qacc[:] = self._qacc_scratch
        mujoco.mj_inverse(self.mj_model, s)
        self._tau_arm_np[:] = s.qfrc_inverse[self._arm_dofadr]
        return self._tau_arm_np

    def reset(self) -> torch.Tensor:
        """Reset state and return observation."""
        self.previous_action.zero_()
        # self.command.zero_()

        # Reset history buffers
        for buffer in self._obs_history_buffers.values():
            buffer.reset()
        if self._hist_combined is not None:
            self._hist_combined.zero_()
            self._hist_ptr  = 0
            self._hist_init = False

        self.episode_length_buf.zero_()
        self._episode_buf_cpu.zero_()
        if self._has_gait_phase:
            self._gait_clock.reset()

        if self.current_target_arm_pos is not None:
            self._reset_arm_targets()

        if self.use_ik:
            # _ik_snap_orientation_targets needs base_quat_wxyz; seed it from IMU before
            # get_observation() runs (which normally sets it).
            self.base_quat_wxyz = torch.tensor(
                self.imu.transformed_quat_wxyz, dtype=torch.float32, device=self.device
            )
            env_ids = torch.tensor([0], dtype=torch.long, device=self.device)
            self._ik_input_qpos.copy_(self._ik_get_physics_qpos(), non_blocking=True)
            self.ik_solver.reset(env_ids, self._ik_input_qpos)
            self._ik_snap_orientation_targets()

        return self.get_observation()

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Execute action and return observation."""
        _t0_step = time.perf_counter_ns() if self._profile_enabled else 0
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        if action.device != self.device:
            action = action.to(self.device)
        if action.ndim == 2:
            action = action[0]

        # Apply action clipping BEFORE filtering (matches training order)
        # Note: clip value is always None or float (normalized during export)
        if self.action_clip_value is not None:
            action = torch.clamp(action, -self.action_clip_value, self.action_clip_value)
        if self._profile_enabled: self._prof_tick("action_pre", _t0)

        # Update arm targets and poll EE service status (both non-blocking)
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        self._update_arm_targets()
        # SAFETY PATCH (08-12): a LATCHED arm's reference
        # must stay at the fault hold. Without this, the stream-stale failsafe
        # ramp above drags current_target_arm_pos to the default pose while the
        # watchdog pin keeps the wire at the hold — a permanent phantom offset
        # that wound the integrator to the clamp on all 7 right-arm joints and
        # spammed SATURATED for minutes after the 08-12 can21 latch, burying
        # the real diagnosis (ARM FAULT) under a false one (settle timeouts).
        if self.arm_watchdog and (self._arm_fault[0] or self._arm_fault[1]) \
                and self.current_target_arm_pos is not None:
            for _wd_i in (0, 1):
                if self._arm_fault[_wd_i]:
                    _sl = slice(_wd_i * 7, (_wd_i + 1) * 7)
                    self.current_target_arm_pos[_sl] = torch.from_numpy(
                        self._arm_fault_hold[_sl]).to(self.current_target_arm_pos)
        if self._profile_enabled: self._prof_tick("arm_update", _t0)
        if self.ee: self.ee.poll()

        # Compute full target position for all joints
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        self._target_pos_buf.copy_(self.default_joint_pos)
        target_pos = self._target_pos_buf
        if self._profile_enabled: self._prof_tick("clone_target", _t0)
        
        # Arms baseline from current target (additive ctrl); legs from default_joint_pos
        if self.current_target_arm_pos is not None:
            target_pos[self.arm_motor_indices] = self.current_target_arm_pos
        # Cam baseline = gaze_reference when the policy actuates the cam, so the residual
        # below yields gaze_reference + scale*action (matches sim). Otherwise the cam is
        # not actuated here and its baseline stays default_joint_pos.
        if self.policy_actuates_cam:
            target_pos[self._cam_motor_indices] = self.gaze_reference
        # PATCH (08-12): with the obs de-fold ON the
        # policy emits its left_wrist_2 residual in the TRAINING frame; the
        # wire is encoder frame (axis flipped), so that one residual flips
        # sign at the boundary. `action` itself stays untouched — the obs
        # feedback (previous_action) must remain in the policy's own frame.
        if self.obs_frame_fix or self.lw2_mirror:
            _action_wire = action.clone()
            _action_wire[self._defold_act_lw2] = -action[self._defold_act_lw2]
        else:
            _action_wire = action
        target_pos[self.action_to_motor_map] += _action_wire * self.policy_action_scale
        # PATCH (07-17): pin the waist AFTER the policy
        # delta so the override wins (see _PIN_WAIST at the class head).
        if self._PIN_WAIST:
            target_pos[self._WAIST_IDX] = self.default_joint_pos[self._WAIST_IDX]
        # 08-12 dynamic standing pin (user order, see WaistPin): full pin while
        # the nav command is zero, policy authority while driving. Same
        # override site as the static pin so it dominates the policy delta;
        # blended so neither engage nor release ever steps the target.
        elif self.pin_waist_standing:
            _cmd_active = bool((self.command.abs() > _WAIST_PIN_CMD_EPS).any())
            _a = self._waist_pin.update(_cmd_active, time.monotonic(), self.dt)
            if _a > 0.0:
                _w = self._WAIST_IDX
                target_pos[_w] = (1.0 - _a) * target_pos[_w] \
                    + _a * self.default_joint_pos[_w]
            _engaged = _a >= 1.0
            if _engaged != self._waist_pin_engaged:
                self._waist_pin_engaged = _engaged
                _log.info("[WAIST] pinned at default (standing)" if _engaged
                          else "[WAIST] released to the policy (drive command)")

        # Limit detection on CPU: motor.mech_pos is always-available numpy — no H2D/CUDA ops.
        # Common case (no violations): ~12 µs vs ~1218 µs for the CUDA path.
        # Violation case (rare): D2H controlled_targets → numpy clamp → H2D back (~200 µs).
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        self._step_count += 1
        motor_pos_at_ctrl_joints = self.motor.mech_pos[self._action_to_motor_map_np]
        dist_to_lower_np = (motor_pos_at_ctrl_joints - self._joint_lower_ctrl_np) / self._joint_limit_threshold_f
        dist_to_upper_np = (self._joint_upper_ctrl_np - motor_pos_at_ctrl_joints) / self._joint_limit_threshold_f
        near_lower_np = dist_to_lower_np < 1.0
        near_upper_np = dist_to_upper_np < 1.0

        if near_lower_np.any() or near_upper_np.any():
            # Controlled joint targets: D2H to numpy for clamping (uncommon in steady locomotion).
            controlled_targets_np = target_pos[self.action_to_motor_map].cpu().numpy()

            # Warn check: compute clamped flags from pre-clamp values (post-clamp always within limits).
            at_warn_rate_limit = (self._step_count - self._last_limit_warn_step >= _LIMIT_WARN_STEP_INTERVAL)
            if at_warn_rate_limit:
                clamped_lower_np = near_lower_np & (controlled_targets_np < self._joint_lower_ctrl_np)
                clamped_upper_np = near_upper_np & (controlled_targets_np > self._joint_upper_ctrl_np)

            controlled_targets_np = np.where(near_lower_np,
                                             np.maximum(controlled_targets_np, self._joint_lower_ctrl_np),
                                             controlled_targets_np)
            controlled_targets_np = np.where(near_upper_np,
                                             np.minimum(controlled_targets_np, self._joint_upper_ctrl_np),
                                             controlled_targets_np)
            target_pos[self.action_to_motor_map] = torch.from_numpy(controlled_targets_np).to(self.device)

            if at_warn_rate_limit and (clamped_lower_np.any() or clamped_upper_np.any()):
                self._last_limit_warn_step = self._step_count
                for i in range(self.action_dim):
                    motor_idx = self._action_to_motor_map_np[i]
                    if clamped_lower_np[i]:
                        _log.warning(f"[LIMIT] {URDF_JOINT_NAMES[motor_idx]}: "
                                     f"pos={self.motor.mech_pos[motor_idx]:.3f} < lower={self._joint_lower_ctrl_np[i]:.3f}")
                    if clamped_upper_np[i]:
                        _log.warning(f"[LIMIT] {URDF_JOINT_NAMES[motor_idx]}: "
                                     f"pos={self.motor.mech_pos[motor_idx]:.3f} > upper={self._joint_upper_ctrl_np[i]:.3f}")
        if self._profile_enabled: self._prof_tick("limit_check", _t0)

        # Refresh joint_pos CUDA view from current encoder readings.
        # Needed by arm_integrator (arm error) and torque_ff (joint_limit_scale).
        self.joint_pos.copy_(torch.from_numpy(self.motor.mech_pos))
        # SAFETY PATCH (07-25): dropout watchdog — runs
        # on the freshest feedback, BEFORE any of it is consumed as a reference.
        if self.arm_watchdog:
            self._arm_watchdog_check()

        # Arm reference source this step (mutually exclusive):
        #   IK       : Cartesian solution overwrites the policy residual   (use_ik & stale-or-cartesian-cmd)
        #   RESIDUAL : policy delta on current_target_arm_pos              (otherwise)
        # Failsafe (stream stale) is RESIDUAL with the reference already ramped to default in
        # _update_arm_targets(); it is not a separate control path.
        # PROTOCOL (07-20): IK runs when ANY arm is Cartesian
        # (per-arm schema); the joint-mode arm's slots are masked off below.
        arm_ik_active = self.use_ik and (not self._arm_stream_active
                                         or self._arm_cart[0] or self._arm_cart[1])
        # SAFETY PATCH (07-14) — rides on the arm_ik_active
        # mode flag above. RISING-EDGE RESEED:
        # ee_pos_targets keep the last streamed Cartesian target forever; when the
        # IK re-engages after a joint-stream phase / stream silence, resuming on
        # that stale target VIOLENTLY snaps the arm back to it (observed on
        # hardware: after a rear-staircase retreat the arm whipped back to the old
        # cube position). Re-seed solver AND targets from encoders: failsafe
        # semantics = "hold the pose the arm is actually in".
        if arm_ik_active and not getattr(self, "_arm_ik_was_active", True):
            # (07-20: body extracted to _ik_mode_reseed — adds _ik_solver_lock
            # serialization against the in-flight worker solve.)
            self._ik_mode_reseed("IK re-engaging (hold current pose)")
        self._arm_ik_was_active = arm_ik_active

        # IK: overwrite arm joint targets with Cartesian solution.
        # Runs when: (a) local IK mode and no arm stream, or (b) arm stream sends Cartesian targets.
        # 1-step latency: _ik_step posts fresh inputs to the worker and returns the previous
        # step's result. dq/ddq are read in the same lock as ik_out to prevent torn reads.
        if arm_ik_active:
            _t0_ik = time.perf_counter_ns() if self._profile_enabled else 0
            ik_out, ik_dq, ik_ddq = self._ik_step()
            if (self._arm_cart[0] and self._arm_cart[1]) or not self._arm_stream_active:
                target_pos[self._ik_arm_urdf_idx] = ik_out
                self.current_target_arm_pos[self._ik_in_arm_idx] = ik_out  # keep in sync for smooth STOP/timeout ramp
            else:
                # PROTOCOL (07-20): mixed per-arm mode — apply
                # the IK solution ONLY to the Cartesian arm's 7 joints; the
                # joint-mode arm keeps its rate-limited stream targets (already in
                # target_pos via the current_target_arm_pos baseline above).
                for i in range(2):
                    if self._arm_cart[i]:
                        sl = slice(0, 7) if i == 0 else slice(7, 14)
                        target_pos[self._ik_arm_urdf_idx[sl]] = ik_out[sl]
                        self.current_target_arm_pos[self._ik_in_arm_idx[sl]] = ik_out[sl]
            if self._profile_enabled: self._prof_tick("ik_step", _t0_ik)

        # Arm integrator: corrects steady-state joint-position error (gravity-comp residual, IK residual, etc).
        # Runs after IK so it corrects IK residual when IK is active (current_target_arm_pos synced above).
        # Skipped when stream sends Cartesian targets without IK — current_target_arm_pos would be in Cartesian
        # coords (invalid as joint reference) and IK is not running to convert it.
        # Guard is a distinct condition from arm_ik_active: it asks "is current_target_arm_pos a
        # valid joint reference?" — false only when a Cartesian target is held with IK off.
        if (self.arm_integrator_enabled
                and self.current_target_arm_pos is not None
                and not (self._arm_stream_is_cartesian and not self.use_ik)):
            arm_pos = self.joint_pos[self.arm_motor_indices]
            error = self.current_target_arm_pos - arm_pos
            self._arm_integral = torch.clamp(
                self._arm_integral + error * self.dt,
                -_ARM_INTEGRAL_LIMIT, _ARM_INTEGRAL_LIMIT,
            )
            # 08-12: a latched arm accumulates NO integral. The latch zeroed it
            # once, but a held arm still carries a small steady PD sag, so left
            # running it re-winds to the clamp in seconds and the SATURATED
            # warning cries wolf about an arm that is deliberately not tracking.
            # (The reference re-pin after _update_arm_targets kills the big
            # phantom error; this kills the residual one. The wire never saw
            # either — the watchdog pin is the last writer — the harm was the
            # false diagnosis.)
            if self.arm_watchdog and (self._arm_fault[0] or self._arm_fault[1]):
                for _wd_i in (0, 1):
                    if self._arm_fault[_wd_i]:
                        self._arm_integral[_wd_i * 7:(_wd_i + 1) * 7] = 0.0
            # SATURATION IS A DIAGNOSIS, AND IT USED TO BE SILENT. The 08-06
            # settle-timeout hunt cost three sessions because left_shoulder_1's
            # integral sat pinned at the clamp for 67% of an 80 s recording and
            # emitted nothing: a saturated integrator means the arm is holding a
            # setpoint it can never reach, and every downstream symptom (settle
            # timeout, handoff deficit, T13) is that offset wearing a different
            # hat. It matters MORE now that the clamp is 0.4 rad, because a real
            # stall can hide ~180 mm of command offset behind it.
            self._arm_int_sat_ticks = torch.where(
                self._arm_integral.abs() >= _ARM_INTEGRAL_LIMIT - 1e-6,
                self._arm_int_sat_ticks + 1,
                torch.zeros_like(self._arm_int_sat_ticks))
            if self._step_count - self._arm_int_sat_warn_step >= \
                    _LIMIT_WARN_STEP_INTERVAL:
                pinned = (self._arm_int_sat_ticks >= _ARM_INT_SAT_WARN_TICKS)
                if bool(pinned.any()):
                    self._arm_int_sat_warn_step = self._step_count
                    who = ", ".join(
                        f"{self.arm_joint_names[i]} {float(self._arm_integral[i]):+.3f}"
                        for i in range(len(self.arm_motor_indices)) if bool(pinned[i]))
                    _log.warning(
                        f"[ARM] integrator SATURATED at +-{_ARM_INTEGRAL_LIMIT} "
                        f"rad for >= {_ARM_INT_SAT_WARN_TICKS} ticks: {who} — "
                        f"the arm is holding a setpoint it cannot reach, so the "
                        f"streamed target and the wire reference have a "
                        f"permanent offset. Expect settle timeouts.")
            target_pos[self.arm_motor_indices] = torch.clamp(
                target_pos[self.arm_motor_indices] + _ARM_KI * self._arm_integral,
                self.joint_lower[self.arm_motor_indices],
                self.joint_upper[self.arm_motor_indices],
            )

        # SAFETY PATCH (07-25): dropout watchdog pin —
        # deliberately the LAST position writer before mech_pos_ref.
        if self.arm_watchdog and (self._arm_fault[0] or self._arm_fault[1]):
            self._arm_watchdog_pin(target_pos)

        if self._profile_enabled:
            _t0 = time.perf_counter_ns()
            # PATCH (08-11): CUDA-guard — the profile
            # path was written on the GPU machine; on the CPU-only robot computer
            # torch.cuda.synchronize() raises ("No HIP GPUs are available")
            # and --profile crashed the run at the first step. On CPU the
            # .cpu().numpy() below is already synchronous, so the tag keeps
            # its meaning: the device-drain + readback stall.
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # drain GPU: stall includes policy inference + all prior CUDA ops
            self.motor.mech_pos_ref[:] = target_pos.cpu().numpy()
            self._prof_tick("gpu_sync", _t0)
        else:
            self.motor.mech_pos_ref[:] = target_pos.cpu().numpy()

        # Arm torque feedforward — legs handled by RL policy; no full-body FF.
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        dist_to_lower = (self.joint_pos - self.joint_lower) / self.joint_limit_threshold
        dist_to_upper = (self.joint_upper - self.joint_pos) / self.joint_limit_threshold
        joint_limit_scale = torch.clamp(torch.min(dist_to_lower, dist_to_upper), 0, 1)
        if arm_ik_active:
            # ik_dq, ik_ddq guaranteed defined: gated by the same arm_ik_active flag as the _ik_step() call above.
            self.update_kinematics(update_joint_vel=False)
            tau_arm = self._compute_arm_inv_dyn_cpu(self.mj_data, ik_dq, ik_ddq)
            # Clamp BEFORE the scalings: the ceiling is about the physical torque
            # the inverse dynamics is asking for, not about what survives the
            # limit/CLI derating (both <= 1, so clamping after would let a
            # near-limit joint pass a larger raw demand).
            tau_arm = self._clamp_arm_tau_ff(tau_arm, "inverse dynamics")
            joint_limit_scale_arm = joint_limit_scale[self._ik_arm_urdf_idx].cpu().numpy()
            # 07-25 review #4: mask a latched arm's FF BEFORE it reaches
            # mech_torque_ref. The CAN sender threads read that buffer
            # asynchronously, so "write nonzero, zero it later in the tick"
            # leaves a race window where a latched arm's joints could be sent
            # a live feed-forward. The value must simply never be nonzero.
            _ff_arm = (tau_arm * joint_limit_scale_arm
                       * self.torque_correction_ratio[self._ik_arm_urdf_idx].cpu().numpy())
            self._arm_watchdog_mask_ff(_ff_arm)
            self.motor.mech_torque_ref[self._ik_arm_urdf_idx_np] = _ff_arm
        elif self.gravity_compensation_enabled:
            self.gravity_torque_raw = self.compute_gravity_compensation()
            self.gravity_torque_applied = self.gravity_torque_raw * joint_limit_scale * self.torque_correction_ratio
            # Same ceiling on the gravity branch. It has never been seen to spike,
            # but it is fed by the same model/encoder state, so leaving one of the
            # two feed-forward paths unbounded would just move the hole.
            _ff_arm = self._clamp_arm_tau_ff(
                self.gravity_torque_applied[self.arm_motor_indices].cpu().numpy(),
                "gravity compensation")
            self._arm_watchdog_mask_ff(_ff_arm)   # review #4: pre-mask, no race
            self.motor.mech_torque_ref[self.arm_motor_indices] = _ff_arm
        else:
            # (07-21 PHASE 2 review): with BOTH arms in
            # joint mode (two parallel staircases) arm_ik_active is False, and
            # without grav-comp NOTHING wrote or cleared the arm torque
            # feed-forward — the last IK tick's tau_arm stayed latched as a
            # constant bias for the whole staircase. ZERO it: joint-stream
            # tracking is pure PD, no feed-forward.
            self.motor.mech_torque_ref[self.arm_motor_indices] = 0.0

        # SAFETY PATCH (07-25): dropout watchdog — a
        # latched arm gets NO feed-forward at all (pure PD+D damped hold).
        # Runs after all three FF branches so it dominates whichever wrote.
        if self.arm_watchdog and (self._arm_fault[0] or self._arm_fault[1]):
            for _wd_i in (0, 1):
                if self._arm_fault[_wd_i]:
                    self.motor.mech_torque_ref[
                        self._wd_idx[_wd_i * 7:(_wd_i + 1) * 7]] = 0.0

        if self._profile_enabled: self._prof_tick("torque_ff", _t0)
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        self.previous_action.copy_(action)
        if self._profile_enabled: self._prof_tick("action_clone", _t0)
        self.episode_length_buf += 1
        self._episode_buf_cpu += 1
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        obs = self.get_observation()
        if self._profile_enabled: self._prof_tick("get_obs_total", _t0)

        if self.recorder.enabled and self.episode_length_buf >= _RECORD_WARMUP_STEPS:
            self.recorder.record_step(self, action, target_pos)

        if self._profile_enabled:
            self._prof_tick("step_total", _t0_step)
            if self._step_count % 100 == 0:
                self._prof_report()
        return obs

    # SAFETY PATCH (07-25): arm feed-forward ceiling.
    def _clamp_arm_tau_ff(self, tau: np.ndarray, source: str) -> np.ndarray:
        """Bound the arm feed-forward at _ARM_TAU_FF_MAX and SAY SO when it bites.

        In-place on the caller's reused buffer — both call sites hand over a
        scratch array they do not read afterwards, and the inverse-dynamics path
        runs every control tick, so allocating here would add garbage to the hot
        loop for no benefit.

        The warning is the point as much as the clamp. A silent ceiling would
        turn the 07-24 failure into an invisible one: the arm would merely
        behave oddly instead of visibly slamming, and the IK discontinuity that
        produced a 292 N·m demand would never be chased. Throttled to one line
        per second per source so a sustained fault cannot flood the console and
        push the control loop late.
        """
        peak = float(np.max(np.abs(tau))) if tau.size else 0.0
        if peak <= _ARM_TAU_FF_MAX:
            return tau
        np.clip(tau, -_ARM_TAU_FF_MAX, _ARM_TAU_FF_MAX, out=tau)
        now = time.time()
        if now - getattr(self, "_tau_ff_warn_t", -1e9) > 1.0:
            self._tau_ff_warn_t = now
            _log.warning(
                f"[ARM] feed-forward CLAMPED: {source} asked for "
                f"{peak:.1f} N·m, ceiling {_ARM_TAU_FF_MAX:.0f} N·m. Normal peak "
                f"is ~4 N·m — a demand this size is an IK/model discontinuity, "
                f"not a physical load. Check the arm forensics recording around "
                f"this moment.")
        return tau

    # SAFETY PATCH (07-25): arm motor-dropout watchdog.
    def _arm_watchdog_check(self, now: float | None = None) -> None:
        """Detect off-bus arm motors from frozen feedback; latch a damped hold.

        Runs every control tick right after fresh motor feedback lands. A joint
        is STALE when its (mech_pos, mech_torque) pair is bit-identical to the
        previous QUALIFIED tick — the receiver threads overwrite those buffers
        ~5x per control tick, so a live motor virtually always differs
        (measured repeat rate 0.4-1.0% single ticks). Copies the buffers before
        comparing: they are shared with the receiver threads, and comparing a
        live buffer against a stored VIEW of itself would read eternally frozen.

        `now` exists for offline replay (feed recorded timestamps); production
        callers omit it.

        Latch semantics (per arm, terminal — rationale at _WATCHDOG_STALE_TICKS):
          * hold pose = measured joints at latch; live joints stay pinned there
            (PD + damping, feed-forward zeroed at the write sites);
          * every joint that ever went stale ("condemned") has its hold track
            its own reported position permanently, so a motor that reconnects
            in ANY state meets zero position error — never a snap;
          * the arm integrator is cleared so it cannot wind up against a limp
            motor and unload into it on reconnection.
        """
        if now is None:
            now = time.monotonic()
        pos31 = np.array(self.motor.mech_pos, dtype=np.float32, copy=True)
        tq31 = np.array(self.motor.mech_torque, dtype=np.float32, copy=True)
        cur = np.stack([pos31[self._wd_idx], tq31[self._wd_idx]])
        if self._wd_prev is None:
            self._wd_prev, self._wd_prev_pos31, self._wd_last_t = cur, pos31, now
            return
        # QUALIFIED-TICK RULE (07-25 review findings #1/#3): watchdog state
        # advances ONLY on a tick that is (a) at least _WATCHDOG_MIN_TICK_S
        # after the previous qualified tick — catch-up ticks after a scheduling
        # stall are shorter than a motor's own feedback period, so their
        # freezes are meaningless — and (b) differentially live (>=2 joints
        # anywhere updated vs the last qualified snapshot). On an unqualified
        # tick NOTHING moves: counters do not climb toward a false latch
        # during a global receiver stall, and they are not cleared either, so
        # a real dropout in progress just pauses and resumes. Both false-latch
        # mechanisms the review demonstrated (streaks accrued behind a closed
        # gate + per-bus staggered drain on resume; near-miss on a 6.3 ms
        # catch-up tick) die at this rule.
        if now - self._wd_last_t < _WATCHDOG_MIN_TICK_S:
            return
        if int(np.count_nonzero(pos31 != self._wd_prev_pos31)) < 2:
            return
        frozen = np.all(cur == self._wd_prev, axis=0)
        starting = frozen & (self._wd_consec == 0)
        self._wd_stale_since[starting] = now
        self._wd_consec = np.where(frozen, self._wd_consec + 1, 0)
        for i, side in enumerate(("LEFT", "RIGHT")):
            sl = slice(i * 7, (i + 1) * 7)
            ripe = (self._wd_consec[sl] >= _WATCHDOG_STALE_TICKS) & \
                   (now - self._wd_stale_since[sl] >= _WATCHDOG_STALE_MIN_S)
            if self._arm_fault[i] or not np.any(ripe):
                continue
            self._arm_fault[i] = True
            self._arm_fault_hold[sl] = pos31[self._wd_idx[sl]]
            self.current_target_arm_pos[sl] = torch.from_numpy(
                self._arm_fault_hold[sl]).to(self.current_target_arm_pos)
            self._arm_integral[sl] = 0.0
            self._arm_int_sat_ticks[sl] = 0.0
            stale = [self.arm_joint_names[j]
                     for j in range(i * 7, (i + 1) * 7)
                     if self._wd_consec[j] >= _WATCHDOG_STALE_TICKS]
            _log.critical(
                f"[ARM] ██ WATCHDOG ██ {side} arm motor(s) OFF-BUS: "
                f"{stale} — feedback frozen >= {_WATCHDOG_STALE_TICKS} "
                f"qualified ticks and >= {_WATCHDOG_STALE_MIN_S*1000:.0f} ms "
                f"while the rest of the robot updates. Arm LATCHED into "
                f"damped hold at the measured pose (no chasing, feed-forward "
                f"zeroed). This latch is terminal: inspect the can-bus "
                f"harness / motor, then restart real_env. (07-24 incident "
                f"class: shoulder pair dropped 0.3-0.4 s, self-reset, and "
                f"the recovery reaction slammed the arm.)")
        # >>> 07-25b MODE trigger (closes 07-25 review gap #2): a motor that
        # stays ON the bus but drops out of Running (self-protection Reset
        # after a fault, spontaneous MCU reboot — the 07-18 OVER CURRENT
        # class) keeps its feedback updating, so the freeze rule above is
        # blind to it. The firmware reports mode_status in every status
        # frame; treat "any arm joint not Running" exactly like a freeze:
        # same qualified-tick streak, same wall-clock floor, same terminal
        # latch. Limp joints are condemned so their hold tracks the reported
        # position — a re-enabled motor meets zero position error, no snap.
        ms31 = np.asarray(getattr(self.motor, "mode_status", ()), dtype=int)
        if ms31.size > int(self._wd_idx.max()):
            ec31 = np.asarray(getattr(self.motor, "error_code", ()), dtype=int)
            for i, side in enumerate(("LEFT", "RIGHT")):
                sl = slice(i * 7, (i + 1) * 7)
                arm_ms = ms31[self._wd_idx[sl]]
                bad = arm_ms != 2
                if not np.any(bad):
                    self._wd_mode_consec[i] = 0
                    continue
                if self._wd_mode_consec[i] == 0:
                    self._wd_mode_since[i] = now
                self._wd_mode_consec[i] += 1
                if self._arm_fault[i]:
                    self._wd_condemned[sl] |= bad
                    continue
                if self._wd_mode_consec[i] >= _WATCHDOG_STALE_TICKS and \
                        now - self._wd_mode_since[i] >= _WATCHDOG_STALE_MIN_S:
                    self._arm_fault[i] = True
                    self._arm_fault_hold[sl] = pos31[self._wd_idx[sl]]
                    self.current_target_arm_pos[sl] = torch.from_numpy(
                        self._arm_fault_hold[sl]).to(self.current_target_arm_pos)
                    self._arm_integral[sl] = 0.0
                    self._arm_int_sat_ticks[sl] = 0.0
                    self._wd_condemned[sl] |= bad
                    detail = {self.arm_joint_names[i * 7 + j]:
                              (int(arm_ms[j]),
                               int(ec31[self._wd_idx[sl][j]]) if ec31.size else -1)
                              for j in range(7) if bad[j]}
                    _log.critical(
                        f"[ARM] ██ WATCHDOG/MODE ██ {side} arm motor(s) "
                        f"ON-BUS but NOT RUNNING: {detail} — joint: (mode, "
                        f"error_code); mode 0=Reset 1=Calib 255=Unknown, "
                        f"error bits 1=UV 2=OC 4=OT 8=mag-enc 16=hall-enc "
                        f"32=uncal. The motor self-protected or rebooted "
                        f"while still answering, so its feedback never "
                        f"froze. Arm LATCHED into damped hold; limp joints "
                        f"track their reported position (a re-enabled motor "
                        f"meets zero error — no snap). Terminal: inspect, "
                        f"then restart real_env.")
        # <<< 07-25b MODE trigger
        for i in (0, 1):
            if self._arm_fault[i]:
                sl = slice(i * 7, (i + 1) * 7)
                # Condemned = went stale at least once since the latch. Its
                # hold tracks the REPORTED position permanently — through the
                # blackout (frozen value = latch pose, a no-op) AND after
                # reconnection (the post-jump measured value). Whatever state
                # the motor returns in, its commanded target equals where it
                # actually is: reconnection can never produce a snap. Healthy
                # joints of the latched arm stay pinned at the latch pose —
                # that pin IS the damped hold.
                self._wd_condemned[sl] |= \
                    self._wd_consec[sl] >= _WATCHDOG_STALE_TICKS
                cond = self._wd_condemned[sl]
                if cond.any():
                    self._arm_fault_hold[sl][cond] = pos31[self._wd_idx[sl]][cond]
        self._wd_prev = cur
        self._wd_prev_pos31 = pos31
        self._wd_last_t = now

    def _arm_watchdog_mask_ff(self, ff14: np.ndarray) -> None:
        """Zero a latched arm's 7 entries of a 14-vector feed-forward IN PLACE
        before it is written to motor.mech_torque_ref (left 0-6, right 7-13 —
        both FF branches build their vectors in that order). Pre-masking is
        the race-free guarantee; the post-branch sweep is only defense in
        depth (07-25 review #4: sender threads sample the buffer mid-tick)."""
        if not (self._arm_fault[0] or self._arm_fault[1]):
            return
        for i in (0, 1):
            if self._arm_fault[i]:
                ff14[i * 7:(i + 1) * 7] = 0.0

    def _arm_watchdog_pin(self, target_pos: torch.Tensor) -> None:
        """Overwrite a latched arm's slice of the final joint target with the
        hold pose. Called as the LAST writer before mech_pos_ref, so it
        dominates every upstream contributor (stream baseline, policy residual,
        IK solution, integrator) by construction — one override point instead
        of guarding four writers."""
        for i in (0, 1):
            if self._arm_fault[i]:
                sl = slice(i * 7, (i + 1) * 7)
                target_pos[self._wd_idx_t[sl]] = torch.from_numpy(
                    self._arm_fault_hold[sl]).to(target_pos)

    # FORENSICS PATCH (07-24) — telemetry only.
    @property
    def _fx_bus_names(self) -> list:
        """CAN interface names in the controller's own bus_id order.

        py_motor builds the controller with
        ``can_interfaces = sorted(set(bus for _, bus, _ in motor_setup))`` and
        indexes buses by position in THAT list, so the order is the sorted one
        (note plain string sort puts ``can9`` last, after ``can2x``). Derived
        the same way here rather than hard-coded, so a wiring change cannot
        silently mislabel the queue depths.
        """
        names = getattr(self, "_fx_bus_names_cache", None)
        if names is None:
            from humanoid_config import motor_setup as _ms
            try:
                names = sorted({e[1] for e in _ms})
            except (IndexError, TypeError):
                names = []
            self._fx_bus_names_cache = names
        return names

    def _fx_bus_queue(self) -> list:
        """Frames queued but not yet sent, per bus — non-zero means a backlog.

        Empty list on FakeMotorController (no such binding) and on any driver
        that predates the accessor; the recorder treats a missing series as
        "not available" rather than as zero congestion.
        """
        fn = getattr(self.motor, "get_bus_send_queue_depth", None)
        if fn is None:
            return []
        try:
            return [int(fn(i)) for i in range(len(self._fx_bus_names))]
        except (RuntimeError, ValueError, IndexError):
            return []

    def close(self):
        """Shut down background threads. Call before exit."""
        if self.use_ik:
            self._ik_stop.set()
            self._ik_trigger.set()  # unblock wait() in worker
            self._ik_thread.join(timeout=1.0)
            _log.info("[IK] Background solver thread stopped")

    def get_observation(self) -> torch.Tensor:
        """Build observation"""

        # IMU freshness: counter increments each packet; stale = frozen data
        current_counter = self.imu.counter
        if current_counter == self._imu_prev_counter:
            self._imu_stale_steps += 1
            if self._imu_stale_steps >= _IMU_STALE_STEPS_MAX:
                _log.error(f"[IMU] Stale for {self._imu_stale_steps} steps — entering safe mode")
                # Frozen IMU data causes a balance policy to command destabilizing torques.
                # Enter damping-only mode immediately before unwinding the call stack.
                self.motor.loc_kp[:] = 0
                self.motor.spd_kp[:] = 5
                self.motor.mech_torque_ref[:] = 0
                raise RuntimeError("IMU stale: hardware safe mode engaged")
        else:
            self._imu_stale_steps = 0
        self._imu_prev_counter = current_counter

        # IMU + joint state: fill numpy scratch then one H2D copy (eliminates 7 kernel launches).
        # Gait phase is folded in here too — saves a separate copy_() call (~150 µs).
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        sensor_np = self._sensor_np
        n_joints  = self.n_joints
        sensor_np[0:3]                    = self.imu.transformed_ang_vel
        sensor_np[3:6]                    = self.imu.transformed_gravity_vec
        sensor_np[6:10]                   = self.imu.transformed_quat_wxyz
        sensor_np[10:10+n_joints]         = self.motor.mech_pos
        sensor_np[10+n_joints:10+2*n_joints] = self.motor.mech_vel
        sensor_np[10+2*n_joints:10+3*n_joints] = self.motor.mech_torque
        sensor_np[10+3*n_joints:10+4*n_joints] = self.motor.mech_torque_ref
        if self._has_gait_phase:
            # Advance gait clock (pure Python math, no CUDA); write 4 floats into bundle.
            self._gait_clock._advance(self._command_cpu, self._episode_buf_cpu)
            sensor_np[self._gait_np_offset:self._gait_np_offset + 4] = self._gait_clock._gait_out_np
        self._sensor_cuda.copy_(torch.from_numpy(sensor_np))
        self.imu_timestamp = self.imu.timeStamp
        # base_lin_vel updated externally (e.g. by policy) or stays zero
        if self._profile_enabled: self._prof_tick("obs_h2d_copy", _t0)

        # Extract controlled joint states (or full joint states based on config)
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        if self._observe_full_joints:
            out_joint_pos = self.joint_pos
            out_joint_vel = self.joint_vel
            out_joint_torque = self.joint_torque
            out_joint_effort_ref = self.joint_effort_ref
        else:
            out_joint_pos = self.joint_pos[self.action_to_motor_map]
            out_joint_vel = self.joint_vel[self.action_to_motor_map]
            out_joint_torque = self.joint_torque[self.action_to_motor_map]
            out_joint_effort_ref = self.joint_effort_ref[self.action_to_motor_map]

        # Fetch vicon once per step; last_results reused by _record_step
        wb_pos = wb_quat_wxyz = base_result = None
        if self.vicon is not None:
            self.vicon.fetch_step()
            base_result = self.vicon.last_results.get("vicon")
            wb_pos, wb_quat_wxyz = self.vicon.compute_world_base(base_result)
            if wb_quat_wxyz is not None:
                self.projected_gravity_vicon = self.vicon.projected_gravity(wb_quat_wxyz, self.device)

        self.obs_dict = {
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "projected_gravity": self.projected_gravity,
            "joint_pos": out_joint_pos,
            "joint_vel": out_joint_vel,
            "joint_effort": out_joint_torque,
            "joint_effort_ref": out_joint_effort_ref,
            "actions": self.previous_action,
            "command": self.command,
        }

        if self.current_target_arm_pos is not None:
            self.obs_dict["target_arm_joint_pos"] = self.current_target_arm_pos

        if self.gaze_in_obs:
            self.obs_dict["target_camera_joint_pos"] = self.gaze_reference

        if "waist_joint_pos" in self.obs_structure:
            self.obs_dict["waist_joint_pos"] = self.joint_pos[self._waist_idx:self._waist_idx+1]
            self.obs_dict["waist_joint_vel"] = self.joint_vel[self._waist_idx:self._waist_idx+1]

        if self._has_gait_phase:
            # gait_phase_sensor is a view into _sensor_cuda; already filled in obs_h2d_copy above.
            self.obs_dict["gait_phase"] = self.gait_phase_sensor

        if self._profile_enabled: self._prof_tick("obs_dict_build", _t0)

        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        self.update_kinematics(update_joint_vel=True)
        if self._profile_enabled: self._prof_tick("update_kinematics", _t0)

        if "robot_com_b" in self.obs_structure:
            _t0 = time.perf_counter_ns() if self._profile_enabled else 0
            # Set state for kinematics: orientation from IMU, positions/velocities from encoders.
            # qvel[:6] = 0: freejoint vel unknown (accepted gap, analogous to base_lin_vel = 0).
            com_w = self.mj_data.subtree_com[self.root_body_id]
            root_pos = self.mj_data.xpos[self.root_body_id]
            cvel_lin_w = self.mj_data.cvel[self.root_body_id, 3:6]

            # Inverse-quaternion rotation: R_root⁻¹ × v_world
            q_w = self.base_quat_wxyz[0]
            q_vec = -self.base_quat_wxyz[1:]  # inverse: negate vector part

            v_pos = torch.tensor([com_w[0]-root_pos[0], com_w[1]-root_pos[1], com_w[2]-root_pos[2]], dtype=torch.float32, device=self.device)
            v_vel = torch.tensor(cvel_lin_w, dtype=torch.float32, device=self.device)

            temp = torch.cross(q_vec, v_pos) + q_w * v_pos
            self._com_pos_b.copy_(v_pos + 2.0 * torch.cross(q_vec, temp))

            temp = torch.cross(q_vec, v_vel) + q_w * v_vel
            self._com_vel_b.copy_(v_vel + 2.0 * torch.cross(q_vec, temp))

            self.obs_dict["robot_com_b"] = torch.cat([self._com_pos_b, self._com_vel_b])
            if self._profile_enabled: self._prof_tick("robot_com_b", _t0)


        # Build observation with history buffers
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        if self._hist_combined is not None:
            # Combined double-buffer path: 3 CUDA kernels (1 cat + 2 writes) vs 2N+1 individual.
            torch.cat([self.obs_dict[k].ravel() for k in self._hist_term_order],
                      out=self._hist_cur_frame[0])
            # PATCH (08-12): wrist de-fold — applied to
            # the policy-facing row ONLY. _hist_cur_frame feeds nothing but the
            # history ring, so telemetry, the integrator, the watchdog and every
            # wire path keep seeing raw encoder-frame values.
            if self.obs_frame_fix:
                _f = self._hist_cur_frame[0]
                _f[self._defold_plus] += _WRIST_HALF_PI
                _f[self._defold_minus] -= _WRIST_HALF_PI
                _f[self._defold_neg] *= -1.0
            # 08-12 night: lw2 deviation mirror (see the setup block) — same
            # single policy-facing point, mutually exclusive with the above.
            elif self.lw2_mirror:
                _f = self._hist_cur_frame[0]
                _f[self._lw2_pt_cols] = 2.0 * self._lw2_default - _f[self._lw2_pt_cols]
                _f[self._lw2_vel_col] = -_f[self._lw2_vel_col]
            if not self._hist_init:
                # Backfill all 2H slots with the first frame.
                self._hist_combined.copy_(
                    self._hist_cur_frame.unsqueeze(1).expand(-1, 2 * self._hist_H, -1))
                self._hist_init = True
            else:
                p = self._hist_ptr
                self._hist_combined[:, p]                = self._hist_cur_frame
                self._hist_combined[:, p + self._hist_H] = self._hist_cur_frame
                self._hist_ptr = (p + 1) % self._hist_H
            obs = self._hist_combined[:, self._hist_ptr:self._hist_ptr + self._hist_H, :]
        else:
            obs_parts = []
            for key in self.obs_structure:
                obs_term = self.obs_dict[key]
                if key in self._obs_history_buffers:
                    flatten = self._obs_history_config[key]["flatten"]
                    obs_term = self._obs_history_buffers[key].append(obs_term, flatten=flatten)
                obs_parts.append(obs_term)
            if self._obs_is_sequence:
                if self._obs_buf is None:
                    obs = torch.cat(obs_parts, dim=-1)
                    self._obs_buf = torch.zeros_like(obs)
                    self._obs_buf.copy_(obs)
                else:
                    torch.cat(obs_parts, dim=-1, out=self._obs_buf)
                obs = self._obs_buf
            else:
                torch.cat([p.ravel() for p in obs_parts], out=self._obs_buf[0])
                obs = self._obs_buf
        if self._profile_enabled: self._prof_tick("obs_assembly", _t0)

        # Must copy all live numpy buffers before enqueuing: NNGPublisher encodes
        # asynchronously in a background thread — the control loop will have already
        # overwritten the source arrays by encode time.
        # Compute EE poses in base FLU frame.
        # update_kinematics() already ran this step, so xpos/xmat are current — zero extra FK cost.
        _t0 = time.perf_counter_ns() if self._profile_enabled else 0
        R_world_from_base = self.mj_data.xmat[self.root_body_id].reshape(3, 3)
        base_pos_world    = self.mj_data.xpos[self.root_body_id]
        ee = {}
        _quat_buf = np.empty(4)
        for ee_name, ee_id in self.ee_frame_ids.items():
            side = _EE_SIDE.get(ee_name, ee_name)
            if _EE_FRAME_TYPE_MAP[ee_name] == "site":
                ee_pos_world = self.mj_data.site_xpos[ee_id]
                ee_rot_world = self.mj_data.site_xmat[ee_id].reshape(3, 3)
            else:
                ee_pos_world = self.mj_data.xpos[ee_id]
                ee_rot_world = self.mj_data.xmat[ee_id].reshape(3, 3)
            ee_pos_base = R_world_from_base.T @ (ee_pos_world - base_pos_world)
            ee_rot_base = R_world_from_base.T @ ee_rot_world
            mujoco.mju_mat2Quat(_quat_buf, ee_rot_base.flatten())  # → wxyz
            ee[side] = {
                "pos_actual":  ee_pos_base.tolist(),
                "quat_actual": _quat_buf.tolist(),
                "pos_target":  None,
                "quat_target": None,
                # EE-service grab result (None until a grab completes on this
                # side); rides telemetry so the monitor detection packet and
                # the sweep GUI see it without extra plumbing.
                "grasp_detected": self.ee.grasp_detected.get(side) if self.ee else None,
                # Stable ID of the terminal hand_grab result above.  The
                # auto-operator accepts a verdict only when it matches the ID
                # of that hold's currently outstanding grab burst.
                "grasp_action_id": getattr(
                    self.ee, "grasp_action_id", {}).get(side) if self.ee else None,
                # SAFETY PATCH (07-16): EE-chain liveness.
                # Without this the operator cannot distinguish a DEAD gripper
                # chain (no --ee-service / socket conflict / service crashed /
                # heartbeat lost) from a physical empty pinch — it ran whole
                # missions against a void, logging fake "empty pinch" messages
                # (07-16 grasp-chain audit, top silent-failure finding).
                "ee_alive": bool(self.ee is not None and self.ee.alive),
                # Zero-calibration in flight (2026-08-10): the reach
                # tool's FSM waits for False on both sides before its camera
                # survey starts. None until a heartbeat carries the field.
                "zeroing": getattr(self.ee, "zeroing", {}).get(side) if self.ee else None,
                # SAFETY PATCH (07-18): gripper servo
                # thermal state (max winding temp [C] / |load| per side, from
                # the EE-service heartbeat) — a held cube keeps the servos in
                # permanent partial stall; this feeds the operator's warnings.
                "servo_temp": self.ee.servo_temp.get(side) if self.ee else None,
                "servo_load": self.ee.servo_load.get(side) if self.ee else None,
            }
        if self.use_ik:
            _ik_pos  = self.ee_pos_targets[0].cpu().numpy()   # (2,3) — one H2D sync
            _ik_quat = self.ee_quat_targets[0].cpu().numpy()  # (2,4) — one H2D sync
            for i, side in enumerate(("left", "right")):
                if side in ee:
                    ee[side]["pos_target"]  = _ik_pos[i].tolist()
                    ee[side]["quat_target"] = _ik_quat[i].tolist()

        self.publisher.publish({
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "projected_gravity": self.projected_gravity,
            "joint_pos": self.joint_pos,  # torch tensor — already a copy of motor.mech_pos
            "joint_vel": out_joint_vel,
            "joint_effort": out_joint_torque,
            "joint_effort_ref": out_joint_effort_ref,
            "actions": self.previous_action,
            "command": self.command,
            "imu_quat_wxyz": self.imu.transformed_quat_wxyz.copy(),
            "vicon_pos": base_result[0][0].copy() if base_result else None,
            "wb_pos": wb_pos,
            "wb_quat_wxyz": wb_quat_wxyz,  # safe: new array
            "ee": ee,
            # SAFETY PATCH (07-18): per-motor winding
            # temperature [degC], URDF/motor_setup order (parsed from every
            # feedback frame — motor.hpp:690 — but previously never exposed).
            # Context: left_wrist_2 OVER CURRENT latch after sustained
            # side-home load with zero thermal visibility; Robstride firmware
            # protection trips ~80 C. Feeds the operator's thermal watch.
            "motor_temp": np.asarray(self.motor.temperature, dtype=float).tolist(),
            # FORENSICS PATCH (07-24): arm command-chain
            # and CAN-bus visibility. ADDITIVE ONLY — nothing below is read back
            # by any control path; these keys exist so an incident can be
            # replayed from numbers instead of argued from prose (the recurring
            # right-arm self-collision was diagnosed three separate ways because
            # the decisive values never left this process).
            #
            # mech_pos_ref is THE missing one. The arm reference is built in
            # three layers that nothing downstream can see:
            #     current_target_arm_pos      rate-limited stream target, shaped
            #                                 by the PER-JOINT box clamp above —
            #                                 so the executed joint path is a box
            #                                 path, not the chord the offline
            #                                 staircase audit sampled
            #   + action * policy_action_scale   the policy residual, added every
            #                                 step with no symmetry constraint
            #   + _ARM_KI * _arm_integral     steady-state error correction
            #   -> clamped to joint limits -> mech_pos_ref
            # Only the sum reaches the motors, and only the sum explains where
            # the arm actually went.
            #
            # bus_queue is the direct measure of CAN congestion. motor_setup_dict
            # puts right_shoulder_2..right_wrist_3 — six of the right arm's seven
            # joints — on a single bus (can21), so a send backlog there stalls
            # almost the whole arm and is indistinguishable from a mechanical jam
            # by any other signal we publish. FakeMotorController has no such
            # method, hence the getattr.
            "mech_pos_ref": np.asarray(self.motor.mech_pos_ref, dtype=float).tolist(),
            "mech_torque_ref": np.asarray(self.motor.mech_torque_ref, dtype=float).tolist(),
            "arm_integral": self._arm_integral,
            # Which mode the RECEIVER believes each arm is in. Compared against
            # what the operator published on 9874, this catches protocol
            # mismatches (the class the 07-20 per-arm wire change was about).
            "arm_cart": [bool(self._arm_cart[0]), bool(self._arm_cart[1])],
            "arm_stream_active": bool(self._arm_stream_active),
            "gaze_stream_active": bool(getattr(self, "_gaze_stream_active", False)),
            "bus_names": self._fx_bus_names,
            "bus_queue": self._fx_bus_queue(),
            # 07-25 watchdog state [left, right]: True = that arm is latched in
            # the damped hold after a motor dropout. Lets the operator (and the
            # forensics recorder) see the fault instead of inferring it from a
            # mysteriously frozen arm.
            "arm_fault": [bool(self._arm_fault[0]), bool(self._arm_fault[1])],
            # 08-07 journey audit: the joystick override latch, published so a
            # nav client can SEE that its commands are being discarded. A stick
            # deflection past _NAV_OVERRIDE_THRESHOLD latches this and only
            # [NAV_RESUME] (LB) clears it; until then every nav_cmd on 9873 is
            # dropped in favour of the stick. An autonomous mission had no way
            # to observe that state, so it walked its whole 150 s timeout while
            # the robot stood still and the log printed the speeds it wanted.
            "nav_paused": bool(self._nav_paused),
            # 07-25 firmware-side state, parsed from every status frame:
            # mode_status 0=Reset 1=Calibration 2=Running; error_code fault
            # bits (1=UV 2=OC 4=OT 8=mag-enc 16=hall-enc 32=uncal). A motor
            # that is ON the bus but has dropped out of Running (the class the
            # freeze-detection watchdog cannot see) shows up here as 2->0 with
            # its feedback still updating. FakeMotorController lacks both,
            # hence the getattr.
            "mode_status": np.asarray(getattr(self.motor, "mode_status",
                                              np.zeros(0)), dtype=int).tolist(),
            "error_code": np.asarray(getattr(self.motor, "error_code",
                                             np.zeros(0)), dtype=int).tolist(),
            "stamp": time.time(),
        })
        # SAFETY PATCH (07-18): local thermal console
        # watch for operator-less runs — one warning line per 10 s while any
        # winding is at/above 60 C (firmware trip ~80 C).
        _temps = np.asarray(self.motor.temperature, dtype=float)
        if _temps.size and float(np.nanmax(_temps)) >= 60.0 \
                and time.time() - getattr(self, "_temp_warn_t", 0.0) > 10.0:
            self._temp_warn_t = time.time()
            _hot = {URDF_JOINT_NAMES[i] if i < len(URDF_JOINT_NAMES) else f"m{i}":
                    round(float(v), 1) for i, v in enumerate(_temps) if v >= 60.0}
            _log.warning(f"[TEMP] hot motor windings (firmware trip ~80 C): {_hot}")
        if self._profile_enabled: self._prof_tick("ee_publish", _t0)

        return obs

    def set_command(self, vx: float, vy: float, wz: float):
        """Set velocity command."""
        self.command[0] = vx
        self.command[1] = vy
        self.command[2] = wz
        self._command_cpu[0] = vx
        self._command_cpu[1] = vy
        self._command_cpu[2] = wz

    def process_input(self) -> str | None:
        """Read controller input, update velocity command, handle torque adjustments.
        Returns '[SHUTDOWN]' or '[RESET]' for caller to act on, None otherwise.

        Navigation override logic (reworked 08-11 — user: "I will use the
        controller to tune the standing position, so it must be OK to launch;
        during the journey the controller must not take over except safety
        operations"):
          - With NO fresh nav stream (nobody navigating): large stick deflection
            (> _NAV_OVERRIDE_THRESHOLD) takes manual control, as always — this
            is the positioning/practice mode.
          - When a nav stream STARTS while a STICK-latched pause is held, the
            pause clears itself, loudly: launching a mission is a deliberate
            human act at the terminal, exactly as deliberate as pressing LB.
            The old behaviour (only LB clears, mission FATALs after 10 s) made
            every pre-mission joystick positioning session kill the next
            launch.
          - While a nav stream is ACTIVE: sticks are IGNORED (a bumped stick
            must not steal a mission). The controller keeps exactly the safety
            surface: LB = deliberate PAUSE (and, pressed again, resume),
            A = shutdown, X/B = torque down/up.
          - An LB-latched pause is EXPLICIT and never self-clears — only LB
            resumes it. A mission sees nav_paused in telemetry and refuses
            after its handback window, which is correct: the operator took
            the robot on purpose.
          - No nav_cmd received for _NAV_CMD_TIMEOUT seconds → robot stops
            (failsafe), unchanged.
        """
        commands, axes = self.ctrl.poll()

        # Map joystick axes to velocity
        # FLU frame: +X=forward, +Y=left, +Z=up
        # Joystick: RightY+ = forward, LeftX+ = strafe right, RightX+ = CW rotation
        vx_js = axes.get("RightY", 0.0) * self.VEL_SCALE_LINEAR
        vy_js = -axes.get("LeftX", 0.0) * self.VEL_SCALE_LINEAR
        wz_js = -axes.get("RightX", 0.0) * self.VEL_SCALE_ANGULAR

        # Nav-stream freshness is tracked UNCONDITIONALLY (it used to update
        # only in the not-paused branch, so a paused robot could never see a
        # stream start — the rising edge below needs it every tick).
        nav = self.nav_receiver
        if nav.data_id != self._last_nav_data_id:
            self._last_nav_data_id = nav.data_id
            self._last_nav_recv_time = time.time()
        nav_fresh = (time.time() - self._last_nav_recv_time) < _NAV_CMD_TIMEOUT

        # A nav stream STARTING clears a stick-latched pause (mission launch
        # after joystick positioning — the 08-11 rule). An LB-latched pause is
        # explicit and survives; only LB clears it.
        if nav_fresh and not self._nav_was_fresh and self._nav_paused                 and self._nav_paused_source == "stick":
            self._nav_paused = False
            self._nav_paused_source = None
            _log.warning("[NAV] nav stream started — clearing the joystick "
                         "positioning pause (launching a mission is as "
                         "deliberate as pressing LB)")
        self._nav_was_fresh = nav_fresh

        # Large stick deflection takes over ONLY when nobody is navigating;
        # during an active mission stream the sticks are inert (safety surface
        # is LB/A/X/B). LB toggles pause/resume.
        if max(abs(vx_js), abs(vy_js), abs(wz_js)) > _NAV_OVERRIDE_THRESHOLD                 and not nav_fresh:
            if not self._nav_paused:
                _log.warning("[NAV] Joystick override — manual control active")
                self._nav_paused_source = "stick"
            self._nav_paused = True

        for cmd in commands:
            if cmd in ("[SHUTDOWN]", "[RESET]"):
                return cmd
            elif cmd == "[TORQUE_UP]":
                self.torque_limit = min(0.8, self.torque_limit + 0.1)
                self.motor.set_max_torque_ratio(self.torque_limit)
                self._apply_group_torque_limits()  # PATCH (07-16): keep cam cap
                _log.warning(f">>> [TORQUE_UP] torque={self.torque_limit*100:.0f}%")
            elif cmd == "[TORQUE_DOWN]":
                self.torque_limit = max(0.1, self.torque_limit - 0.1)
                self.motor.set_max_torque_ratio(self.torque_limit)
                self._apply_group_torque_limits()  # PATCH (07-16): keep cam cap
                _log.warning(f">>> [TORQUE_DOWN] torque={self.torque_limit*100:.0f}%")
            elif cmd == "[KEYFRAME]":
                self.recorder.mark_keyframe()
            elif cmd == "[NAV_RESUME]":
                # LB is a TOGGLE since 08-11: resume when paused, DELIBERATE
                # pause when not — the one-button safety takeover the user
                # kept ("Pause/disable are the only controller powers during
                # a journey"). A button pause never self-clears.
                if self._nav_paused:
                    self._nav_paused = False
                    self._nav_paused_source = None
                    _log.warning("[NAV] Resumed — handing back to navigation policy")
                else:
                    self._nav_paused = True
                    self._nav_paused_source = "button"
                    _log.warning("[NAV] PAUSED by LB — nav_cmd discarded until "
                                 "LB again (missions will refuse after their "
                                 "handback window)")
        if self._nav_paused:
            self.set_command(vx_js, vy_js, wz_js)
        else:
            cmd = nav.data.get("nav_cmd") if (nav.data and nav_fresh) else None
            if cmd is not None:
                self.set_command(float(cmd[0]), float(cmd[1]), float(cmd[2]))
            else:
                self.set_command(0.0, 0.0, 0.0)

        return None

    def shutdown(self):
        """Save recording and stop controller, then delegate hardware teardown to base."""
        self.recorder.save_recording(self)
        self.ctrl.stop()
        self.nav_receiver.stop()
        self.arm_receiver.stop()
        if self.ee: self.ee.close()
        self.publisher.close()
        super().shutdown()          # motors wind down + imu.shutdown()
        _log._listener.stop()       # flush remaining log messages before exit


def main():
    from dataclasses import dataclass
    from typing import Optional
    import tyro

    @dataclass
    class Args:
        """Flags for the real-robot policy loop (humanoid_real_env.py). Bring-up line: docs/OPERATIONS.md T3; process overview: the module docstring."""
        # --task journal for the four v2-Grid run directories in the list below
        # (facts kept here verbatim; the choices list carries one-line tags only):
        #
        # ...BankFlatDecoupled — 2026-08-06: upstream c163845, legged_env_dev
        #   master. Same net (35 tensors, obs 120, act 31) and same joint
        #   definitions as v2_best; new training run 2026-08-05_00-34-46
        #   model_0015000. Its env_config is upstream's verbatim EXCEPT the arm
        #   default_pose, which stays ours (see the banner comment in that
        #   file). REJECTED 08-06: cannot hold a stand (std_min binds at cmd 0,
        #   +55% joint power).
        #
        # ...BankFlatDecoupledCosine — 2026-08-11: upstream e5ecf0d, the
        #   cosine-LR retrain of the REJECTED c163845 family, staged from
        #   legged_env_dev deploy/runs (new root-level layout, seed0). Same
        #   interface (obs 120, act 31) but 39 tensors: has_velocity_estimator
        #   flipped TRUE, so this one publishes REAL base_lin_vel. Camera
        #   action_scale rose 0.3/0.2 -> 0.44 (upstream's, verbatim per the
        #   c163845 rule); arm default_pose stays OURS (banner in the
        #   env_config). SEED VERDICT 08-11: seed0 waist-tilts continuously
        #   RIGHT on the hang; seed1 hangs clean and is the DEPLOYED weight
        #   (seed0 kept beside it as policy_deployed.pt.seed0). Hardware gate:
        #   the c163845 isolation protocol (hang, legs straight, cmd 0, joint
        #   power vs v2best) BEFORE any floor stand.
        #
        # ...BankFlatDecoupledMuonLr1e3CosineAll — 2026-08-12 night: upstream
        #   64168a6, Muon-optimizer (lr 1e-3) + CosineAll retrain of the Cosine
        #   family, run 2026-08-07_16-47-56_flash_sac_s1 model_0015000. Config
        #   byte-identical to the Cosine run outside the arm block (verified at
        #   staging); arm default_pose replaced with OURS per the systemic-trap
        #   rule (upstream's original kept as env_config.yaml.*_64168a6).
        #   UNTESTED ON HARDWARE — full hang protocol first (lean check, cmd 0,
        #   then a front-reach hold with the gyro readout: the v2 generation's
        #   clockwise fetch rotation is the open question this retrain may or
        #   may not answer).
        #
        # ...BankFlatDecoupledCosineAR — 2026-08-13 16:04: upstream c0b6b18
        #   (pushed 15:53 the same afternoon), the "AR" retrain of the Cosine
        #   family. TWO SEEDS shipped, both staged: seed1 is policy_deployed.pt,
        #   seed0 sits beside it as policy_deployed.pt.seed0 (same convention as
        #   the Cosine run, where the hang protocol picked seed1 after seed0
        #   waist-tilted right). Interface verified IDENTICAL to the deployed
        #   Cosine at staging: 39 tensors, velocity estimator present, obs 120 /
        #   act 31, same layer dims — drop-in. env_config differs from ours ONLY
        #   in the arm default_pose block (camera action_scale, kp/kd and
        #   everything else byte-identical), so ours was carried over verbatim
        #   per the systemic-trap rule and upstream's original kept as
        #   env_config.yaml.*_c0b6b18. The commit ALSO changed the training env
        #   (event.py +88, humanoid_velocity_env_cfg.py +47, experiments.py
        #   +544) — this is a new recipe, not another seed. UNTESTED ON
        #   HARDWARE: full hang protocol first (lean check, joint power at cmd 0
        #   vs v2best, then a front-reach hold with the gyro readout for the v2
        #   generation's clockwise fetch rotation), and run the protocol PER
        #   SEED.
        task: Literal[
            "humanoid_velocity",
            "HumanoidLegsOnly",
            "HumanoidLegsOnlyFixedArms",
            "HumanoidLegsOnlyRandArmsFullObs",
            "HumanoidLegsOnlyFixedArmsFullObs",
            "HumanoidActorHistory3",
            "HumanoidRandArmsAdditiveCtrl",
            "HumanoidRandArmsAdditiveCtrlTracking",
            "HumanoidVelocityMar03",
            "HumanoidVelocityHighEnergyPenalty",
            "G1LegsOnly",
            "HumanoidVelocityPushAware",
            "HumanoidVelocityStanding",
            "HumanoidVelocityRMACNN",
            "HumanoidVelocityRMACNNShort",
            "HumanoidVelocityRMACNNShortEstimator",
            "HumanoidVelocityRMACNNShortEstimatorPhase",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv5",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv6",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7", #ok
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv8",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv9",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv10",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv11",
            "HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv12",
            "HumanoidRmaVelEstFlashSac",
            "HumanoidRmaVelEstFlashSacTune12", # backward bend
            "HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam", # actuated head cam, L2T student
            "HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam", # (07-19): the CALIBRATED MODEL directory (humanoid_site.DEPLOY_MODEL_TASK); held v2_best weights 07-31..08-24, ships no weights in the release
            "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled", # 2026-08-06, upstream c163845 — REJECTED 08-06 (cannot hold a stand); journal above
            "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine", # 2026-08-11, upstream e5ecf0d cosine-LR retrain; seed1 = THE RELEASED CHECKPOINT (humanoid_site.DEPLOY_TASK default since 2026-08-24); journal above
            "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledMuonLr1e3CosineAll", # 2026-08-12, upstream 64168a6 Muon + CosineAll retrain — UNTESTED ON HARDWARE; journal above
            "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineAR", # 2026-08-13, upstream c0b6b18 "AR" retrain, two seeds — UNTESTED ON HARDWARE; journal above
            "v2_best",
        ] = DEFAULT_TASK
        """Task preset = run directory under <legged_env_v2>/mj_envs/deploy/runs/ (overrides --policy/--config). Default = the deployed checkpoint, humanoid_site.DEPLOY_TASK (HUMANOID_DEPLOY_TASK / site_local.py); docs/OPERATIONS.md T3 passes it explicitly. A deploy task outside this list is still accepted (tyro widens the choice to STR with a warning); the policy/config path check at load time is the real gate."""
        policy: Optional[str] = None
        """Policy path (overrides --task)"""
        config: Optional[str] = None
        """Config path (overrides --task)"""
        torque_limit: float = 0.1
        fake: bool = False
        """Use fake hardware"""
        cmd_vx: float = 0.0
        cmd_vy: float = 0.0
        cmd_wz: float = 0.0
        grav_comp: bool = False
        """Enable gravity compensation"""
        add_right_ee: bool = False
        """Attach EE cube to wrist_3_R for gravity compensation (0.04 m / 0.4 kg)"""
        record: bool = False
        """Record data for simulation replay"""
        record_path: Optional[str] = None
        """Output path for recording (auto-generated if not set)"""
        use_ik: bool = False
        """Enable IK (Cartesian arm control)"""
        telemetry_ip: str = _site.WORKSTATION_IP
        """Workstation IP that subscribes to robot telemetry (port 9870)"""
        high_level_controller_ip: str = _site.WORKSTATION_IP
        """IP of the machine publishing nav commands on port 9873 (humanoid_auto_operator.py, humanoid_curobo_reach.py --journey, or gampad_gui_control.py)"""
        arm_sender_ip: str = _site.WORKSTATION_IP
        """IP of the machine publishing arm/gaze/ee packets on port 9874 (humanoid_auto_operator.py, humanoid_curobo_reach.py; bench: gampad_gui_control.py, humanoid_joint_monkey_hw.py)"""
        ee_service: bool = False
        """Enable EE hand servo service (requires humanoid_end_effector_service.py running separately)"""
        vicon: bool = False
        """Enable Vicon motion capture (host: humanoid_site.VICON_IP; the tracked object name is set where ViconInterface is built in __init__)"""
        profile: bool = False
        """Enable per-step timing profiler (reports every 100 steps)"""
        enable_motor: Literal["true", "false", "leg", "arm", "camera", "arm_camera"] = "true"
        """Motor enable mode: 'true'=all joints, 'leg'=legs+waist only, 'arm'=arms only,
        'camera'=gimbal only, 'arm_camera'=arms+gimbal (bench reach, legs limp), 'false'=none"""
        arm_watchdog: bool = True
        """07-25: latch an arm into a damped hold the moment any of its motors
        goes silent on the CAN bus (frozen feedback >= 3 ticks). Guards against
        the 07-24 incident class where a dropped shoulder pair came back with a
        position jump and the recovery reaction slammed the arm into the torso.
        --no-arm-watchdog restores the old (unprotected) behaviour."""
        pin_waist_standing: bool = False
        """08-12 (user): freeze the waist at its default whenever the nav
        command is zero (0.5 s debounce, 0.5 s blend-in), hand it back to the
        policy the instant any drive command arrives (0.3 s blend-out).
        Symptom-level clamp for the cmd-0 rightward waist grind during
        left-arm front grasps (root cause = the wrist obs frame seam, whose
        direct fix failed hang validation 08-12). The static ancestor
        _PIN_WAIST held July's standing grasps waist-stable."""
        lw2_mirror: bool = False
        """08-12 night: mirror ONLY left_wrist_2's policy-facing deviation
        around its default (obs = 2*default - enc, vel negated, action
        residual flipped at the wire). Motivated by the user's controlled
        finding: seed1 front grasps tilt with the LEFT arm and not the RIGHT —
        left_wrist_2 is the stack's only left-only frame asymmetry (encoder
        sign inverted vs training). Unlike --obs-frame-fix (which failed hang
        validation by moving the static operating point), the static inputs
        are bit-identical to today's; only the wrist's MOTION direction is
        corrected. Mutually exclusive with --obs-frame-fix."""
        obs_frame_fix: bool = False
        """08-12: feed the policy TRAINING-frame wrist observations (the
        encoder frame's wrist_1 -+pi/2 refs removed, left_wrist_2 sign
        unflipped) and flip the left_wrist_2 action residual at the wire
        boundary. Offline A/B replay showed the raw encoder-frame wrist stream
        drives a spurious sustained waist command during LEFT-arm reaches (the
        front-grasp waist yaw); the de-fold removed it 13x in the reach phase
        on Cosine seed1 and quieted v2_best's static waist bias 6x. DEFAULT
        OFF until hang-validated — every checkpoint's hardware history was
        validated around the old (corrupted) observations. Rationale:
        wrist_defold_columns at the module head; offline evidence in
        tests/test_obs_frame_fix.py."""

    args = tyro.cli(Args)

    # Resolve policy and config paths from task or explicit args
    if args.policy is None or args.config is None:
        task_policy, task_config = _get_deploy_paths(args.task)
        args.policy = args.policy or task_policy
        args.config = args.config or task_config

    _log.info("=" * 60)
    _log.info("HUMANOID REAL ROBOT DEPLOYMENT")
    _log.info("=" * 60)

    # PATCH (08-11): pin torch to ONE thread. The
    # robot computer has 24 cores and torch defaulted to 12 intra-op + 24 inter-op
    # threads, while taskset confines this process to 5 cores — 36 threads
    # thrashing 5 cores is exactly the bursty 24-36 ms rate-limiter lates
    # that flooded every run's console. A 1.3 MB policy net gains nothing
    # from intra-op parallelism; single-thread inference is both faster and
    # jitter-free. Must run BEFORE the first torch op (the interop pool
    # cannot be resized once started).
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        _log.warning("[PERF] interop pool already started — thread pin partial")

    # Load policy
    _log.info(f"\nLoading policy: {args.policy}")
    _map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(args.policy, map_location=_map_location)
    policy.eval()

    # Default record_path to include task name if recording but no path specified
    if args.record and args.record_path is None:
        args.record_path = f"recordings/{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    # Create environment
    env = HumanoidRealEnv(args.config, args.torque_limit, use_fake=args.fake,
                          gravity_compensation_enabled=args.grav_comp,
                          add_right_ee=args.add_right_ee, record=args.record,
                          record_path=args.record_path, use_ik=args.use_ik,
                          telemetry_ip=args.telemetry_ip,
                          high_level_controller_ip=args.high_level_controller_ip,
                          arm_sender_ip=args.arm_sender_ip,
                          enable_ee_service=args.ee_service,
                          enable_motor=args.enable_motor,
                          profile=args.profile,
                          vicon=args.vicon,
                          arm_watchdog=args.arm_watchdog,
                          obs_frame_fix=args.obs_frame_fix,
                          pin_waist_standing=args.pin_waist_standing,
                          lw2_mirror=args.lw2_mirror,
                          )

    # Set initial command from CLI args
    if args.cmd_vx != 0 or args.cmd_vy != 0 or args.cmd_wz != 0:
        env.set_command(args.cmd_vx, args.cmd_vy, args.cmd_wz)
        _log.info(f"Command: vx={args.cmd_vx}, vy={args.cmd_vy}, wz={args.cmd_wz}")

    try:
        env.prepare(duration=3.0)
        obs = env.reset()

        _log.info("\nRunning policy... (Ctrl+C to stop)")
        _log.info("=" * 60)

        # PATCH (08-11): pre-warm the JIT. The
        # profile pinned policy_forward's only spike at 24.9 ms on the FIRST
        # forward (torch.jit graph optimization); every later call is
        # ~185 us. Pay it here, while the robot is still holding the
        # prepare stance, instead of on control tick #1.
        with torch.no_grad():
            for _ in range(3):
                if (env._has_velocity_estimator_cfg
                        if env._has_velocity_estimator_cfg is not None
                        else hasattr(policy, "forward_with_estimation")):
                    policy.forward_with_estimation(obs)
                else:
                    policy(obs)
        _log.info("[PERF] policy pre-warmed (3 forwards)")

        from loop_rate_limiters import RateLimiter
        # PATCH (08-11): warn stays TRUE (user:
        # keep the flood). The 10 s [LOOP] summary below aggregates it, and
        # --profile now names the exact stage that eats the budget — main-loop
        # pieces (input_poll, policy_forward) included, not just step().
        rate = RateLimiter(frequency=env.control_freq, warn=True)
        _period_s = 1.0 / env.control_freq
        _late_n, _late_worst_ms, _cyc_sum, _cyc_n = 0, 0.0, 0.0, 0
        _t_prev = time.perf_counter()

        step = 0
        running = True

        while running:
            rate.sleep()
            # PATCH (08-11): loop health accounting
            _t_now = time.perf_counter()
            _cyc = _t_now - _t_prev
            _t_prev = _t_now
            _cyc_sum += _cyc
            _cyc_n += 1
            if _cyc > _period_s * 1.1:
                _late_n += 1
                _late_worst_ms = max(_late_worst_ms, (_cyc - _period_s) * 1e3)
            if _cyc_n >= int(env.control_freq * 10):
                if _late_n:
                    _log.info(f"[LOOP] {_late_n}/{_cyc_n} ticks late, worst "
                              f"+{_late_worst_ms:.1f} ms, avg cycle "
                              f"{_cyc_sum / _cyc_n * 1e3:.1f} ms "
                              f"(budget {_period_s * 1e3:.0f})")
                _late_n, _late_worst_ms, _cyc_sum, _cyc_n = 0, 0.0, 0.0, 0

            _t0_in = time.perf_counter_ns() if env._profile_enabled else 0
            event = env.process_input()
            if env._profile_enabled: env._prof_tick("input_poll", _t0_in)
            if event == "[SHUTDOWN]":
                _log.info(">>> [SHUTDOWN] triggered")
                running = False
            elif event == "[RESET]":
                _log.info(">>> [RESET] Resetting observation state")
                obs = env.reset()

            _t0_fwd = time.perf_counter_ns() if env._profile_enabled else 0
            with torch.no_grad():
                # Hybrid gate: exported has_velocity_estimator flag is authoritative
                # when present; old flag-less configs fall back to hasattr. The
                # FlashSAC wrapper always defines forward_with_estimation but raises
                # without an estimator, so hasattr alone is unreliable for new exports.
                use_est = (env._has_velocity_estimator_cfg
                           if env._has_velocity_estimator_cfg is not None
                           else hasattr(policy, "forward_with_estimation"))
                if use_est:
                    action, est_vel = policy.forward_with_estimation(obs)
                    env.base_lin_vel = est_vel[0]
                else:
                    action = policy(obs)
                    env.base_lin_vel[:] = 0
            if env._profile_enabled: env._prof_tick("policy_forward", _t0_fwd)

            obs = env.step(action)
            step += 1

            if step % 100 == 0:
                cmd_arr = env.command.cpu().numpy()
                ang = env.imu.transformed_ang_vel
                vel = env.base_lin_vel.cpu().numpy()
                # 08-12 (user): gyro added for the base-rotation hunt — `ang`
                # was already fetched here and thrown away. gyro[2] is yaw
                # rate, rad/s; negative = clockwise (the observed grind).
                _log.info(f"[{step:5d}] cmd: {cmd_arr[0]:+.2f} {cmd_arr[1]:+.2f} {cmd_arr[2]:+.2f} | vel: {vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f} | gyro: {ang[0]:+.2f} {ang[1]:+.2f} {ang[2]:+.2f} | torque: {env.torque_limit*100:.0f}%")
                if env.use_ik:
                    ee = env.ee_pos_targets[0].tolist()
                    _log.info(f"[{step:5d}] ee_pos  L=[{ee[0][0]:+.3f} {ee[0][1]:+.3f} {ee[0][2]:+.3f}]  R=[{ee[1][0]:+.3f} {ee[1][1]:+.3f} {ee[1][2]:+.3f}]")

    except KeyboardInterrupt:
        _log.info("\n\nStopping...")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
