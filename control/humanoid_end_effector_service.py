"""End-effector servo service.

Standalone process that subscribes to EE action requests from humanoid_real_env.py
via pynng IPC, executes FT servo commands via C++ nanobind extension, and publishes
status/heartbeat. Background poll thread caches servo state at 200 Hz.

Left and right grippers are controlled independently: every action request
targets one side ("left" or "right") and is tracked/executed independently of
the other side, so e.g. "left close" and "right open" can be in flight at once.

Each side also supports a "zero_gripper" calibration command: the gripper is
driven to a hard, low-torque pinch, the resulting stalled servo position is
read back and recorded as that side's closed position, and the open position
is derived as closed - OPEN_OFFSET_FROM_CLOSE. Calibration runs as an
independent background task per side and does not block the other side or
the heartbeat.

"hand_grab" is also a background task per side: rather than jumping straight
to a fixed position, the gripper closes in small steps from wherever it
currently is, watching load after every step, until load spikes (contact)
or it reaches its fully-closed position (nothing grasped). On contact it
keeps closing a few more small steps at higher torque to apply a real
inward squeeze, bounded by a hard safety cutoff on load.
"""

import asyncio
import glob
import os
import signal
import sys
import threading
from collections import deque
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

from hardware_bindings.ft_servo import FtServo

import tyro

from ipc.publisher import NNGPublisher, NNGSubscriber
from humanoid_utils import make_async_logger
import humanoid_site as _site

# ---------------------------------------------------------------------------
# IPC Configuration
# ---------------------------------------------------------------------------
# URIs for the NNG IPC sockets (shared by both hands; requests carry `side`)
EE_REQUEST_URI = "ipc:///tmp/ee_request.sock"
EE_STATUS_URI  = "ipc:///tmp/ee_status.sock"

# ---------------------------------------------------------------------------
# Hardware Configuration (Default values)
# ---------------------------------------------------------------------------
HAND_SERIAL_PORT_LEFT  = "/dev/ttyACMservoLeft"
HAND_SERIAL_PORT_RIGHT = "/dev/ttyACMservoRight"
LEFT_SERVO_IDS  = [5]
RIGHT_SERVO_IDS = [0]

# USB serial number of each hand's CH340 driver board, as reported in
# /dev/serial/by-id/usb-1a86_USB_Single_Serial_<SERIAL>-if00.
#
# Why this exists: the /dev/ttyACMservo{Left,Right} paths above are udev
# symlinks created by /etc/udev/rules.d/99-servo.rules, which lives outside the
# repo, needs root to edit, and goes stale the moment a driver board is swapped.
# That is exactly how the 07-31 "right gripper failed to open servo port"
# failure happened: the rule still named a board that was no longer plugged in,
# so the symlink was never created and the open failed with ENOENT. Resolving
# by serial here makes the service work with or without the udev rules, and
# puts the board->hand mapping under version control where it can be grepped.
#
# The mapping is deliberately explicit per side, never "whichever other CH340
# board is plugged in": a positional guess would silently route right-hand
# commands to the left hand after a cable swap.
#
# 08-22: the serials themselves are site settings (humanoid_site: env
# HUMANOID_HAND_USB_SERIAL_LEFT / _RIGHT, then site_local.py, then a default
# that is the original rig's boards), because a board serial names one
# physical rig -- another installation sets its own without editing this file
# (docs/SETUP.md section 4: udevadm ... ID_SERIAL_SHORT). An empty serial
# disables the by-id fallback for that side.
HAND_USB_SERIAL = {"left": _site.HAND_USB_SERIAL_LEFT,
                   "right": _site.HAND_USB_SERIAL_RIGHT}

VALID_SIDES = {"left", "right"}

# Position values for different hand states.
# These are the *defaults* used until a side has been zeroed; after zeroing,
# each side's open/close positions are tracked independently on the service
# instance (see EndEffectorService.hand_open / hand_close).
HAND_POS_OPEN  = -3000
HAND_POS_CLOSE = 3200

#SERVO PARAMS
SPEED = 1000
ACC = 100
TORQUE = 500

# Gripper aperture for width-based closing: an UNCALIBRATED linear map
# (90 mm = HAND_POS_OPEN, 0 mm = HAND_POS_CLOSE); width commands are
# approximate until this is measured against the actual finger geometry.
GRIPPER_MAX_APERTURE_MM = 90.0   # aperture at HAND_POS_OPEN
GRIPPER_MIN_APERTURE_MM = 0.0    # aperture at HAND_POS_CLOSE

# Timing configuration
HAND_MOTION_TIME_MS = 500  # Time allowed for servos to reach target

# ---------------------------------------------------------------------------
# Zeroing / calibration configuration
# ---------------------------------------------------------------------------
# The commanded target during zeroing is intentionally well past any real
# closed position. Reduced torque/speed let the servo stall safely against
# the mechanical pinch (or whatever's between the fingers) instead of
# forcing through it. The position it actually settles at is read back and
# becomes the new "closed" reference for that side.
ZERO_CLOSE_TARGET      = 6000  # overshoot target, past any real HAND_POS_CLOSE
ZERO_TORQUE            = 150    # reduced torque during zeroing, for safety
ZERO_SPEED             = 300    # slower speed during zeroing
ZERO_ACC               = 50
# 08-10 (user: "sometimes the gripper does not close to zero"): "closed" is
# now MEASURED, not assumed. The old recipe — command past the pinch, sleep a
# blind 1.5 s, read once — recorded a short `close` whenever the gripper was
# still travelling (or stiction-stalled) at read time, shifting the whole
# calibrated frame (seen live: close=4839 vs a true ~5997). Now: poll until
# the position STOPS MOVING, then back off and pinch a second time — static
# friction breaks loose on the second pass — and take the deeper stall. Fast
# closes finish EARLIER than the old fixed sleep, so stall heat goes down.
ZERO_POLL_S            = 0.15  # position poll cadence while driving in
ZERO_STILL_EPS         = 8     # counts: consecutive reads within this = stopped
ZERO_STILL_READS       = 3     # consecutive still reads = stalled/closed
ZERO_TRAVEL_BUDGET_S   = 6.0   # hard cap per pinch (was a blind 1.5 s sleep)
ZERO_READ_FAILS_MAX    = 6     # consecutive failed reads = serial dead, fail
ZERO_BACKOFF_COUNTS    = 200   # double-tap: back off this far, pinch again
ZERO_BACKOFF_S         = 0.5   # let the backoff move complete
ZERO_STALL_LOAD_MIN    = 60    # decoded |load| bar for a REAL pinch: still +
                               # parked at the commanded target + load under
                               # this = the servo ARRIVED in free space (no
                               # push), i.e. the true close sits DEEPER than
                               # ZERO_CLOSE_TARGET (a swapped board's frame
                               # can shift past it). Same bar the mission's
                               # zero guard uses for "verifiably no grip".
ZERO_RETARGET_STEP     = 800   # free-space arrival: drive this much deeper
ZERO_RETARGET_MAX      = 3     # bounded chase; then warn and take the reading
ZERO_AT_TARGET_SLACK   = 50    # counts: "parked at the commanded target"
ZERO_AGREE_COUNTS      = 30    # pinches agreeing within this = clean zero;
                               # wider disagreement is logged (debris/stiction)
OPEN_OFFSET_FROM_CLOSE = 6000   # open position = measured close - this offset

# ---------------------------------------------------------------------------
# Grasp configuration
# ---------------------------------------------------------------------------

GRASP_STEP_COUNTS       = 158  # user 2026-08-02 (second x1.5): 1.5x larger
                               # increments again (~1.4 mm contact
                               # resolution, still well under cube scale);
                               # load is still sampled every step and the
                               # squeeze torque is unchanged
GRASP_STEP_DELAY_S      = 0.03  # shorter settle time before reading load
GRASP_APPROACH_SPEED    = 4500   # scaled with the step so per-step motion
                                 # time stays flat -> net close time /1.5
                                 # (if the servo saturates below this it
                                 # just runs flat-out; steps still 1.5x)
GRASP_APPROACH_ACC      = 90
GRASP_APPROACH_TORQUE   = 300

GRASP_LOAD_THRESHOLD    = 600   # decoded |load|; HLS magnitude is 0..1023
GRASP_LOAD_MAX          = 950

GRASP_SQUEEZE_STEPS     = 5
GRASP_SQUEEZE_TORQUE    = 350

# -------- New filtering parameters --------

GRASP_LOAD_SAMPLES = 10           # samples after each motion (fewer, but still median-filtered)
GRASP_HISTORY_SIZE = 3           # rolling history of per-step medians
GRASP_CONSECUTIVE_REQUIRED = 2   # median-of-10 + two steps confirms contact
GRASP_OUTLIER_LIMIT = 1000       # documented decoded HLS magnitude ceiling
GRASP_SAMPLE_DELAY = 0.006       # 6 ms between load samples
GRASP_STALL_POSITION_EPS_COUNTS = 8
GRASP_STALL_COMMAND_GAP_COUNTS = 35
GRASP_STALL_CONSECUTIVE_REQUIRED = 3
GRASP_CLOSE_POSITION_TOLERANCE_COUNTS = 20

# GRASP_TIMEOUT_S is a *safety* abort, not a normal exit path -- it must be
# large enough to cover the worst case (closing the full stroke and never
# detecting contact), or it fires before the load threshold ever gets a
# chance to and the grasp silently "fails open" partway through the close.
#
# Per-step cost = GRASP_STEP_DELAY_S + GRASP_LOAD_SAMPLES * GRASP_SAMPLE_DELAY
#               = 0.03 + 10 * 0.006 = ~0.09 s
# Worst-case steps = full stroke / step size
#               = (HAND_POS_CLOSE - HAND_POS_OPEN) / GRASP_STEP_COUNTS
#               = 6200 / 158 ≈ 39 steps
# Worst-case approach time ≈ 39 * 0.09 ≈ 3.5 s
# Add a squeeze-phase margin and headroom -> round up generously.
# (This is computed from the constants above, so it auto-adjusts if you
# retune step size/delays again -- no need to hand-edit it.)
_STEP_COST_S = GRASP_STEP_DELAY_S + GRASP_LOAD_SAMPLES * GRASP_SAMPLE_DELAY
_WORST_CASE_STEPS = abs(HAND_POS_CLOSE - HAND_POS_OPEN) / GRASP_STEP_COUNTS
GRASP_TIMEOUT_S = (_WORST_CASE_STEPS * _STEP_COST_S) * 1.5  # auto-scaled margin

# Service intervals
HEARTBEAT_INTERVAL_SEC = 1.0
POLL_CADENCE_SEC       = 0.005  # 200Hz loop for responsiveness

VALID_COMMANDS = {"hand_open", "hand_close", "hand_grab", "hand_close_to_width", "zero_gripper", "hand_hold"}
# SAFETY PATCH (07-18): sustained-stall protection.
# A held cube keeps the servos in permanent partial stall at the full
# GRASP_SQUEEZE_TORQUE — heat is I^2R, so dropping the torque limit once the
# grip is static cuts winding heat ~3x. "hand_hold" re-commands the STORED
# grasp squeeze target (NOT the measured position — that would zero the
# position error and the grip force with it) at a reduced torque ceiling.
HOLD_TORQUE_DEFAULT   = 200   # static-hold ceiling (squeeze uses 350)
SHUTDOWN_GRIP_TORQUE  = 150   # Ctrl+C with a cube in hand: keep a gentle grip
SERVO_TEMP_POLL_SEC   = 2.0   # per-side temp/load poll cadence (idle bus only)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionRequest:
    """Incoming request from the environment (e.g. humanoid_real_env.py)."""
    action_id: int
    side: str
    command: str
    timestamp: float
    ttl_seconds: float = 1.0
    width_mm: Optional[float] = None
    # SAFETY PATCH (07-18): optional torque ceiling,
    # only meaningful for command="hand_hold" (stall-heat reduction).
    torque: Optional[int] = None

    @classmethod
    def from_raw_dict(cls, data: dict) -> Optional["ActionRequest"]:
        """Parses the nested 'ee_action' packet format."""
        inner = data.get("ee_action", {})
        try:
            width = inner.get("width_mm", None)
            torque = inner.get("torque", None)  # SAFETY PATCH (07-18)
            return cls(
                action_id=int(inner.get("action_id", -1)),
                side=str(inner.get("side", "")).lower(),
                command=str(inner.get("command", "")),
                timestamp=float(inner.get("stamp", 0.0)),
                ttl_seconds=float(inner.get("ttl_s", 1.0)),
                width_mm=float(width) if width is not None else None,
                torque=int(torque) if torque is not None else None,  # SAFETY PATCH
            )
        except (ValueError, TypeError):
            return None

@dataclass
class StatusReport:
    """Outbound status update for the environment."""
    action_id: int
    side: str
    command: str
    state: str  # "accepted", "succeeded", "failed", "replaced", "dropped_stale", "rejected"
    timestamp: float = field(default_factory=time.time)
    error_code: Optional[str] = None
    detail: str = ""
    grasp_detected: Optional[bool] = None

_log = make_async_logger("ee_service")

# UI Helpers
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"


def width_to_servo_position(width_mm: float) -> int:
    """Map a desired gripper aperture (mm) to a servo position command.

    Linearly interpolates between HAND_POS_CLOSE (0 aperture) and
    HAND_POS_OPEN (max aperture). Clamped to the calibrated range.

    NOTE: this still uses the fixed default HAND_POS_OPEN/HAND_POS_CLOSE,
    not a side's zeroed positions. If you want width-based closing to track
    a zeroed gripper, pass that side's calibrated open/close in here instead.
    """
    clamped = max(GRIPPER_MIN_APERTURE_MM, min(GRIPPER_MAX_APERTURE_MM, width_mm))
    span = GRIPPER_MAX_APERTURE_MM - GRIPPER_MIN_APERTURE_MM
    frac_open = 0.0 if span <= 0 else (clamped - GRIPPER_MIN_APERTURE_MM) / span
    return int(round(HAND_POS_CLOSE + frac_open * (HAND_POS_OPEN - HAND_POS_CLOSE)))


def decode_gripper_load(raw: Any) -> Optional[int]:
    """Return a validated HLS load from raw or already-decoded input.

    Production historically shipped a cached-read bug that decoded every word
    with direction bit 15. HLS load actually uses bit 10, so negative loads
    arrived as ``0x400 | magnitude``. Keep this compatibility decoder in
    Python even after the C++ cache is fixed: an old prebuilt ``.so`` then
    becomes safe immediately, while a rebuilt extension already returning a
    negative value is left untouched.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None

    # A rebuilt driver already returns the signed magnitude.
    if -GRASP_OUTLIER_LIMIT <= value <= GRASP_OUTLIER_LIMIT:
        return value

    # The old prebuilt binding exposed bit-10 direction words unchanged.
    if 0x400 <= value <= 0x7FF:
        magnitude = value & 0x3FF
        if magnitude <= GRASP_OUTLIER_LIMIT:
            return -magnitude
    return None

# ---------------------------------------------------------------------------
# Environment-side IPC Client
# ---------------------------------------------------------------------------

class EEServiceClient:
    """IPC client for EE service. Used by humanoid_real_env.py.

    Lifecycle: construct → poll() each step → close() on shutdown.
    `alive` and `grasp_detected` are plain attributes updated by poll().
    `grasp_detected` is keyed by side: {"left": ..., "right": ...}.
    """

    def __init__(self):
        self.alive: bool = False
        self.grasp_detected: dict[str, Optional[bool]] = {"left": None, "right": None}
        self.grasp_action_id: dict[str, Optional[int]] = {"left": None, "right": None}
        # SAFETY PATCH (07-18): gripper servo thermal
        # state from the service heartbeat (max winding temp / |load| per
        # side) — feeds telemetry so the operator can warn before a stalled
        # servo cooks itself. None until the first extended heartbeat.
        self.servo_temp: dict[str, Optional[float]] = {"left": None, "right": None}
        self.servo_load: dict[str, Optional[float]] = {"left": None, "right": None}
        # Zero-calibration in flight per side, mirrored from the heartbeat
        # (2026-08-10). None until a heartbeat carries the field.
        self.zeroing: dict[str, Optional[bool]] = {"left": None, "right": None}
        self._pub = NNGPublisher(EE_REQUEST_URI)
        self._sub = NNGSubscriber(EE_STATUS_URI)
        self._sub.start()
        self._last_data_id = -1
        self._last_heartbeat_time = 0.0
        self._action_id = -1
        _log.info(f"[EE] Client ready | req: {EE_REQUEST_URI} | status: {EE_STATUS_URI}")

    def poll(self) -> None:
        """Check latest status/heartbeat packet. Call once per control step."""
        if self._sub.data_id == self._last_data_id or self._sub.data is None:
            if self.alive and (time.time() - self._last_heartbeat_time) > 3.0:
                self.alive = False
                _log.warning("[EE] Service heartbeat lost — EE commands disabled")
            return

        self._last_data_id = self._sub.data_id
        pkt = self._sub.data

        if "ee_heartbeat" in pkt:
            was_alive = self.alive
            self.alive = bool(pkt["ee_heartbeat"].get("healthy", False))
            if self.alive and not was_alive:
                _log.info("[EE] Service connected and healthy")
            elif not self.alive and was_alive:
                _log.warning("[EE] Service reports unhealthy (serial unavailable?)")
            self._last_heartbeat_time = time.time()
            # SAFETY PATCH (07-18): thermal fields
            hb = pkt["ee_heartbeat"]
            for _s in ("left", "right"):
                if f"servo_temp_{_s}" in hb:
                    self.servo_temp[_s] = hb[f"servo_temp_{_s}"]
                if f"servo_load_{_s}" in hb:
                    self.servo_load[_s] = hb[f"servo_load_{_s}"]
                if f"grasp_detected_{_s}" in hb:
                    self.grasp_detected[_s] = hb[f"grasp_detected_{_s}"]
                if f"grasp_action_id_{_s}" in hb:
                    self.grasp_action_id[_s] = hb[f"grasp_action_id_{_s}"]
                if f"zeroing_{_s}" in hb:
                    self.zeroing[_s] = bool(hb[f"zeroing_{_s}"])

        elif "ee_status" in pkt:
            st = pkt["ee_status"]
            side = st.get("side")
            state = st.get("state", "")
            if state in ("failed", "rejected"):
                _log.warning(f"[EE] {side} action {st.get('action_id')} {state}: "
                             f"{st.get('error_code')} — {st.get('detail')}")
            command = st.get("command")
            if side in self.grasp_detected and command == "hand_grab" \
                    and state in ("succeeded", "failed"):
                # Correlate the verdict with the logical grab. A failed grab is
                # UNKNOWN, not EMPTY; the operator handles both fail-closed.
                self.grasp_detected[side] = st.get("grasp_detected")
                self.grasp_action_id[side] = st.get("action_id")
            elif side in self.grasp_detected and state == "succeeded" \
                    and command in ("hand_open", "zero_gripper"):
                self.grasp_detected[side] = None
                self.grasp_action_id[side] = None

    def send_action(self, side: str, command: str, width_mm: Optional[float] = None,
                    torque: Optional[int] = None,
                    action_id: Optional[int] = None) -> Optional[int]:
        """Forward a discrete hand command for one side to the EE service.

        No-op if service not alive or side is invalid. `width_mm` is only
        meaningful for command="hand_close_to_width". `command="zero_gripper"`
        triggers that side's zeroing/calibration routine. `torque` is only
        meaningful for command="hand_hold" (stall-heat reduction;
        safety patch, 07-18).
        """
        if side not in VALID_SIDES:
            _log.warning(f"[EE] Invalid side {side!r} — ignoring command")
            return
        if not self.alive:
            _log.warning("[EE] Service not alive — ignoring command")
            return
        if action_id is None:
            self._action_id += 1
            selected_id = self._action_id
        else:
            try:
                selected_id = int(action_id)
            except (TypeError, ValueError, OverflowError):
                _log.warning(f"[EE] Invalid action_id {action_id!r} — ignoring command")
                return None
            if selected_id < 0:
                _log.warning(f"[EE] Negative action_id {selected_id} — ignoring command")
                return None
            self._action_id = max(self._action_id, selected_id)
        self._pub.publish({
            "ee_action": {
                "action_id": selected_id,
                "stamp":     time.time(),
                "ttl_s":     1.0,
                "side":      side,
                "command":   command,
                "width_mm":  width_mm,
                "torque":    torque,  # SAFETY PATCH (07-18)
            }
        })
        return selected_id

    def close(self) -> None:
        self._sub.stop()
        self._pub.close()

# ---------------------------------------------------------------------------
# Port resolution
# ---------------------------------------------------------------------------

def resolve_hand_port(side: str, configured: str) -> str:
    """Return a usable serial-port path for one hand.

    Input : side ("left"/"right") and the configured path (CLI arg or the
            module default udev symlink).
    Output: the configured path if it exists, otherwise the /dev/serial/by-id
            entry whose USB serial matches HAND_USB_SERIAL[side]. If neither
            resolves, the configured path is returned unchanged so the caller's
            open fails with the original, recognisable error.

    Assumption: the driver boards keep the serials recorded in
    HAND_USB_SERIAL (from humanoid_site: HUMANOID_HAND_USB_SERIAL_LEFT/_RIGHT
    or site_local.py). Replacing a board means updating that setting (and, if
    the udev rules are still in use, 99-servo.rules) — the fallback cannot
    detect a board it has never been told about, by design (see
    HAND_USB_SERIAL).
    """
    if os.path.exists(configured):
        return configured

    serial = HAND_USB_SERIAL.get(side)
    matches = sorted(glob.glob(f"/dev/serial/by-id/*{serial}*")) if serial else []
    if not matches:
        return configured

    resolved = matches[0]
    _log.warning(
        f"[EE] {configured} missing (stale/absent udev rule); "
        f"{side} hand resolved by USB serial {serial} -> {resolved}"
    )
    return resolved

# ---------------------------------------------------------------------------
# Service Implementation
# ---------------------------------------------------------------------------

class EndEffectorService:
    """Owns both FEETECH grippers for the lifetime of the process.

    Per side ("left"/"right"): a serial port (udev symlink, else resolved by
    USB serial in resolve_hand_port), one FtServo handle from the C++ binding
    (`connect_hardware`: mode 0, torque on, background position poll) and the
    calibrated open/close positions (module defaults until `zero_gripper`).

    IPC: subscribes to ActionRequest dicts on EE_REQUEST_URI and publishes
    {"ee_status": StatusReport} plus a periodic heartbeat on EE_STATUS_URI.
    `main_loop` is one asyncio loop: heartbeat, one-side-per-tick servo
    thermal poll, timed completion of plain moves (`run_servo_move`), then
    dispatch of new requests. hand_grab (`run_grasp_close`, stepped
    load-supervised close) and zero_gripper (`run_zero_calibration`) run as
    per-side background tasks and reject further commands for that side while
    in flight. On stop the grip guard (`_preserve_grippers_on_shutdown`) runs
    before the hardware is closed.
    """
    def __init__(self,
                 left_port: str = HAND_SERIAL_PORT_LEFT,
                 right_port: str = HAND_SERIAL_PORT_RIGHT):
        self.ports = {
            "left":  resolve_hand_port("left", left_port),
            "right": resolve_hand_port("right", right_port),
        }
        self.servo_ids = {"left": LEFT_SERVO_IDS, "right": RIGHT_SERVO_IDS}

        self.status_pub = NNGPublisher(EE_STATUS_URI)
        self.request_sub = NNGSubscriber(EE_REQUEST_URI)

        self.hands: dict[str, Optional[Any]] = {"left": None, "right": None}
        self.healthy: dict[str, bool] = {"left": False, "right": False}

        # Per-side calibrated open/close positions. Start at the module
        # defaults; updated in place whenever a side is zeroed.
        self.hand_open: dict[str, int] = {"left": HAND_POS_OPEN, "right": HAND_POS_OPEN}
        self.hand_close: dict[str, int] = {"left": HAND_POS_CLOSE, "right": HAND_POS_CLOSE}

        # True while a "zero_gripper" calibration is in flight for that side.
        # New commands to that side are rejected until it clears.
        self.zeroing_in_progress: dict[str, bool] = {"left": False, "right": False}

        # True while a "hand_grab" stepped/load-supervised close is in
        # flight for that side. New commands to that side are rejected
        # until it clears.
        self.grasping_in_progress: dict[str, bool] = {"left": False, "right": False}

        # Tracking IDs to prevent duplicates and handle superseding, per side
        self.last_processed_action_id: dict[str, Optional[int]] = {
            "left": None, "right": None
        }
        self.last_nng_data_id = -1

        # Current async task state, tracked independently per side
        self.active_action: dict[str, Optional[StatusReport]] = {"left": None, "right": None}
        self.completion_time_monotonic: dict[str, float] = {"left": 0.0, "right": 0.0}
        # SAFETY PATCH (07-18): stall-heat protection
        # state. grasp_hold_pos = the final squeeze TARGET of the last
        # successful grasp (what hand_hold re-commands at reduced torque);
        # last_grasp_detected mirrors the last grab verdict for the shutdown
        # guard; servo_temp/servo_load = latest per-side maxima for the
        # heartbeat. All cleared by hand_open.
        self.grasp_hold_pos: dict[str, Optional[int]] = {"left": None, "right": None}
        self.last_grasp_detected: dict[str, Optional[bool]] = {"left": None, "right": None}
        self.last_grasp_action_id: dict[str, Optional[int]] = {"left": None, "right": None}
        self.servo_temp: dict[str, Optional[float]] = {"left": None, "right": None}
        self.servo_load: dict[str, Optional[float]] = {"left": None, "right": None}
        self._temp_poll_t = 0.0
        self._temp_poll_side = "left"

    def connect_hardware(self):
        """Initialize FT servos for both hands and start background position poll."""
        for side, ids in self.servo_ids.items():
            port = self.ports[side]
            try:
                hand = FtServo(port)
                hand.set_modes(ids, 0)
                hand.enable_torques(ids, True)
                hand.start_poll(ids)
                self.hands[side] = hand
                self.healthy[side] = True
                _log.info(f"[EE] Connected {side} hand ({port})")
            except Exception as e:
                self.hands[side] = None
                self.healthy[side] = False
                _log.error(f"[EE] Hardware init failed on {side} ({port}): {e}")

    def emit_status(self, report: StatusReport):
        """Log to terminal and publish to NNG."""
        color = _GREEN if report.state == "succeeded" else (_RED if "failed" in report.state or "rejected" in report.state else "")
        msg = f"[EE] side={report.side} action={report.action_id} {report.command!r} -> {report.state}"
        if report.detail:
            msg += f" ({report.detail})"
        if report.grasp_detected is not None:
            msg += f" [Grasp: {report.grasp_detected}]"

        _log.info(f"{color}{msg}{_RESET}")
        self.status_pub.publish({"ee_status": asdict(report)})

    def emit_heartbeat(self):
        """Standard heartbeat to let the environment know the service is alive."""
        self.status_pub.publish({
            "ee_heartbeat": {
                "stamp": time.time(),
                "service": "end_effector_service",
                "healthy": self.healthy["left"] and self.healthy["right"],
                "healthy_left": self.healthy["left"],
                "healthy_right": self.healthy["right"],
                # SAFETY PATCH (07-18): per-side max
                # servo winding temp [C] / |load| (None until first poll)
                "servo_temp_left":  self.servo_temp["left"],
                "servo_temp_right": self.servo_temp["right"],
                "servo_load_left":  self.servo_load["left"],
                "servo_load_right": self.servo_load["right"],
                "grasp_detected_left": self.last_grasp_detected["left"],
                "grasp_detected_right": self.last_grasp_detected["right"],
                "grasp_action_id_left": self.last_grasp_action_id["left"],
                "grasp_action_id_right": self.last_grasp_action_id["right"],
                # Zero-calibration in flight per side (2026-08-10): the
                # mission FSM waits for both to read False before the
                # camera survey starts.
                "zeroing_left": self.zeroing_in_progress["left"],
                "zeroing_right": self.zeroing_in_progress["right"],
            }
        })

    def poll_servo_thermals(self):
        """SAFETY PATCH (07-18): read one side's servo
        temperatures + loads (alternating sides each call). Skipped while
        that side's bus is busy with a supervised grasp/zeroing (their own
        load sampling owns the serial line). Sustained-stall context: a
        held cube keeps the servos in partial stall indefinitely — this is
        the only thermal visibility the grippers have."""
        side = self._temp_poll_side
        self._temp_poll_side = "right" if side == "left" else "left"
        hand = self.hands.get(side)
        if hand is None or not self.healthy[side] \
                or self.grasping_in_progress[side] or self.zeroing_in_progress[side]:
            return
        try:
            temps, loads = [], []
            for sid in self.servo_ids[side]:
                t = hand.get_temperature(sid)
                l = decode_gripper_load(hand.get_load(sid))
                if t is not None:
                    temps.append(float(t))
                if l is not None:
                    loads.append(abs(float(l)))
            if temps:
                self.servo_temp[side] = max(temps)
            if loads:
                self.servo_load[side] = max(loads)
            if temps and max(temps) >= 55.0:
                _log.warning(f"[EE] {side} gripper servo winding at "
                             f"{max(temps):.0f} C (Feetech ceiling ~70 C) — "
                             f"consider resting the grip")
        except Exception as e:
            _log.debug(f"[EE] thermal poll error ({side}): {e}")

    async def run_servo_move(self, side: str, command: str, width_mm: Optional[float] = None,
                             torque: Optional[int] = None) -> bool:
        """Execute the physical movement for one hand in a background thread.

        Handles the "instant" commands only: hand_open, hand_close,
        hand_close_to_width, hand_hold. hand_grab and zero_gripper are
        stepped, supervised processes and are dispatched as their own
        background tasks instead (see run_grasp_close / run_zero_calibration).

        hand_hold (SAFETY PATCH, 07-18): re-commands the
        stored grasp squeeze TARGET at a reduced torque ceiling — same
        position error, lower current limit, ~3x less winding heat during
        long static holds. Requires a stored grasp (rejected upstream
        otherwise). hand_open clears the stored grasp state.
        """
        if not self.healthy[side] or self.hands[side] is None:
            return False

        hand = self.hands[side]
        ids = self.servo_ids[side]

        def _blocking_move():
            try:
                if command == "hand_open":
                    hand.set_positions(ids, [self.hand_open[side]] * len(ids), SPEED, ACC, TORQUE)
                    # SAFETY PATCH (07-18): grip released
                    self.grasp_hold_pos[side] = None
                    self.last_grasp_detected[side] = None
                    self.last_grasp_action_id[side] = None
                elif command == "hand_close":
                    hand.set_positions(ids, [self.hand_close[side]] * len(ids), SPEED, ACC, TORQUE)
                elif command == "hand_close_to_width":
                    pos = width_to_servo_position(width_mm if width_mm is not None else GRIPPER_MIN_APERTURE_MM)
                    hand.set_positions(ids, [pos] * len(ids), SPEED, ACC, TORQUE)
                elif command == "hand_hold":
                    # SAFETY PATCH (07-18)
                    hold_pos = self.grasp_hold_pos[side]
                    if hold_pos is None:
                        return False
                    tq = int(torque) if torque else HOLD_TORQUE_DEFAULT
                    tq = max(100, min(tq, GRASP_SQUEEZE_TORQUE))  # never below a light
                    hand.set_positions(ids, [hold_pos] * len(ids),   # grip, never above
                                       SPEED, ACC, tq)              # the squeeze itself
                    _log.info(f"[EE] {side}: holding grasp at torque {tq} "
                              f"(stall-heat reduction)")
                return True
            except Exception as e:
                _log.error(f"[EE] Servo write error ({side}): {e}")
                return False

        return await asyncio.to_thread(_blocking_move)

    async def run_grasp_close(
        self, side: str
    ) -> tuple[Optional[bool], Optional[int]]:
        """Close under feedback with a fail-closed three-state verdict.

        ``True`` means contact was positively observed. ``False`` is emitted
        only after valid low-load telemetry *and* measured arrival at the
        calibrated close position. Any read/write/timeout uncertainty is
        ``None`` and must never be treated as permission to open the hand.
        """
        hand = self.hands.get(side)
        ids = self.servo_ids[side]
        if hand is None or not self.healthy[side]:
            return None, None

        load_history = deque(maxlen=GRASP_HISTORY_SIZE)
        load_contact_count = 0
        stall_count = 0

        def _read_position() -> Optional[int]:
            try:
                positions = hand.get_positions(ids)
                if not positions or len(positions) != len(ids):
                    return None
                return int(round(sum(int(p) for p in positions) / len(positions)))
            except Exception as e:
                _log.error(f"[EE] Position read error ({side}): {e}")
                return None

        def _read_load() -> Optional[float]:
            per_servo: list[float] = []
            required = max(1, (GRASP_LOAD_SAMPLES + 1) // 2)
            try:
                for sid in ids:
                    samples: list[int] = []
                    for _ in range(GRASP_LOAD_SAMPLES):
                        decoded = decode_gripper_load(hand.get_load(sid))
                        if decoded is not None:
                            samples.append(abs(decoded))
                        time.sleep(GRASP_SAMPLE_DELAY)
                    # Partial/missing telemetry from either servo is unknown,
                    # never evidence that the hand is empty.
                    if len(samples) < required:
                        return None
                    per_servo.append(float(statistics.median(samples)))
            except Exception as e:
                _log.error(f"[EE] Grasp load read error ({side}): {e}")
                return None
            if len(per_servo) != len(ids):
                return None
            return sum(per_servo) / len(per_servo)

        def _feedback() -> tuple[Optional[float], Optional[int]]:
            return _read_load(), _read_position()

        def _move_and_feedback(
            target: int, torque: int, speed: int, acc: int
        ) -> tuple[bool, Optional[float], Optional[int]]:
            try:
                result = hand.set_positions(
                    ids, [target] * len(ids), speed, acc, torque
                )
                if result is False:
                    _log.error(f"[EE] Grasp write rejected ({side})")
                    return False, None, None
            except Exception as e:
                _log.error(f"[EE] Grasp move error ({side}): {e}")
                return False, None, None
            time.sleep(GRASP_STEP_DELAY_S)
            load, actual = _feedback()
            return True, load, actual

        actual = await asyncio.to_thread(_read_position)
        if actual is None:
            return None, None

        close_limit = self.hand_close[side]
        direction = 1 if close_limit >= actual else -1
        step = direction * GRASP_STEP_COUNTS
        commanded = actual
        start_time = time.monotonic()

        while True:
            if time.monotonic() - start_time > GRASP_TIMEOUT_S:
                _log.warning(f"[EE] {side}: grasp telemetry/motion timeout")
                return None, commanded

            remaining = direction * (close_limit - actual)
            if remaining <= GRASP_CLOSE_POSITION_TOLERANCE_COUNTS:
                load, measured = await asyncio.to_thread(_feedback)
                if load is None or measured is None:
                    return None, commanded
                actual = measured
                if load >= GRASP_LOAD_THRESHOLD:
                    break
                if direction * (close_limit - actual) \
                        <= GRASP_CLOSE_POSITION_TOLERANCE_COUNTS:
                    # This is the only path allowed to call the hand empty.
                    return False, actual
                continue

            next_target = commanded + step
            if direction > 0:
                next_target = min(next_target, close_limit)
            else:
                next_target = max(next_target, close_limit)

            previous_actual = actual
            wrote, load, measured = await asyncio.to_thread(
                _move_and_feedback,
                next_target,
                GRASP_APPROACH_TORQUE,
                GRASP_APPROACH_SPEED,
                GRASP_APPROACH_ACC,
            )
            commanded = next_target
            if not wrote or measured is None:
                return None, commanded

            actual = measured
            progress = direction * (actual - previous_actual)
            command_gap = direction * (commanded - actual)

            if load is not None:
                load_history.append(load)
                rolling_load = float(statistics.median(load_history))
                if rolling_load >= GRASP_LOAD_THRESHOLD:
                    load_contact_count += 1
                else:
                    load_contact_count = 0
                if load_contact_count >= GRASP_CONSECUTIVE_REQUIRED:
                    _log.info(
                        f"[EE] {side}: grasp detected by load "
                        f"(median={rolling_load:.1f})"
                    )
                    break

            if command_gap >= GRASP_STALL_COMMAND_GAP_COUNTS \
                    and progress <= GRASP_STALL_POSITION_EPS_COUNTS:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= GRASP_STALL_CONSECUTIVE_REQUIRED:
                _log.info(
                    f"[EE] {side}: grasp detected by measured stall "
                    f"(actual={actual}, target={commanded})"
                )
                break

        # Contact is already established. Missing squeeze feedback stops the
        # squeeze and reports UNKNOWN; it can never turn into an empty verdict.
        for _ in range(GRASP_SQUEEZE_STEPS):
            if time.monotonic() - start_time > GRASP_TIMEOUT_S:
                return None, commanded
            next_target = commanded + step
            if direction > 0:
                next_target = min(next_target, close_limit)
            else:
                next_target = max(next_target, close_limit)
            if next_target == commanded:
                break

            wrote, load, measured = await asyncio.to_thread(
                _move_and_feedback,
                next_target,
                GRASP_SQUEEZE_TORQUE,
                GRASP_APPROACH_SPEED,
                GRASP_APPROACH_ACC,
            )
            commanded = next_target
            if not wrote or measured is None:
                return None, commanded
            actual = measured
            if load is None:
                return None, commanded
            if load >= GRASP_LOAD_MAX:
                break

        return True, commanded

    async def run_zero_calibration(self, side: str) -> Optional[int]:
        """Drive `side`'s gripper to a hard, low-torque pinch and read back
        the position it stalls at.

        Commands a target well past any real closed position, but at reduced
        torque/speed, so the servo stalls safely against the mechanical pinch
        (fingers touching, or against whatever's between them) rather than
        forcing through it. Returns the measured stalled position, or None
        on failure.
        """
        if not self.healthy[side] or self.hands[side] is None:
            return None

        hand = self.hands[side]
        ids = self.servo_ids[side]

        def _write(target: int) -> bool:
            try:
                hand.set_positions(ids, [target] * len(ids),
                                   ZERO_SPEED, ZERO_ACC, ZERO_TORQUE)
                return True
            except Exception as e:
                _log.error(f"[EE] Zero calibration write error ({side}): {e}")
                return False

        def _read() -> Optional[int]:
            try:
                positions = hand.get_positions(ids)
                if not positions:
                    return None
                return int(round(sum(positions) / len(positions)))
            except Exception as e:
                _log.error(f"[EE] Zero calibration read error ({side}): {e}")
                return None

        def _read_load() -> Optional[int]:
            try:
                vals = [decode_gripper_load(hand.get_load(sid)) for sid in ids]
                mags = [abs(int(v)) for v in vals if v is not None]
                return max(mags) if mags else None
            except Exception as e:
                _log.debug(f"[EE] Zero calibration load read error "
                           f"({side}): {e}")
                return None

        async def _pinch() -> Optional[int]:
            # "Closed" is MEASURED (08-10): the position has stopped moving —
            # ZERO_STILL_READS consecutive reads within ZERO_STILL_EPS counts
            # — never a fixed-sleep guess that reads a still-travelling
            # gripper short. Bounded by ZERO_TRAVEL_BUDGET_S.
            target = ZERO_CLOSE_TARGET
            if not await asyncio.to_thread(_write, target):
                return None
            deadline = time.monotonic() + ZERO_TRAVEL_BUDGET_S
            last, still, bad_reads, retargets = None, 0, 0, 0
            while time.monotonic() < deadline:
                await asyncio.sleep(ZERO_POLL_S)
                pos = await asyncio.to_thread(_read)
                if pos is None:
                    bad_reads += 1
                    if bad_reads >= ZERO_READ_FAILS_MAX:
                        return None
                    continue
                bad_reads = 0
                if last is not None and abs(pos - last) <= ZERO_STILL_EPS:
                    still += 1
                    if still >= ZERO_STILL_READS:
                        # STILL IS NECESSARY, NOT SUFFICIENT (08-10, the day
                        # the right board was swapped): a servo that ARRIVES
                        # at its commanded target in free space is also
                        # perfectly still — with the measured true close
                        # sitting 3 counts under the 6000 target, any frame
                        # shift puts the real pinch BEYOND the target and the
                        # jaws "zero" without ever touching. A real pinch
                        # pushes: the load rides the ZERO_TORQUE limit. Still
                        # + parked at the target + no push = chase deeper,
                        # bounded by ZERO_RETARGET_MAX.
                        load = await asyncio.to_thread(_read_load)
                        if load is not None and load < ZERO_STALL_LOAD_MIN:
                            # One retry before judging: a single decode miss
                            # must not condemn a live servo.
                            await asyncio.sleep(ZERO_POLL_S)
                            load = max(load,
                                       (await asyncio.to_thread(_read_load))
                                       or 0)
                        if load is not None and load < ZERO_STALL_LOAD_MIN \
                                and target - pos > ZERO_AT_TARGET_SLACK:
                            # STILL + NOT at the target + NO push = the servo
                            # is not executing at all. Live case (08-10 night,
                            # "It's not zeroing at all!"): an UNPOWERED servo
                            # rail answers position 0 / load 0 on every read —
                            # constant zeros pass the stillness test, both
                            # taps agree at 0, and the calibration invented
                            # close=0/open=-6000 while the jaws never moved.
                            # A frame like that must FAIL loudly, not succeed.
                            _log.error(
                                f"[EE] {side} zero pinch is STILL at {pos} "
                                f"with load {load}, {target - pos} counts "
                                f"short of its target — the servo is not "
                                f"executing (unpowered rail? dead wire?). "
                                f"Refusing to invent a zero frame.")
                            return None
                        if load is not None and load < ZERO_STALL_LOAD_MIN \
                                and target - pos <= ZERO_AT_TARGET_SLACK:
                            if retargets >= ZERO_RETARGET_MAX:
                                _log.warning(
                                    f"[EE] {side} zero pinch still parked at "
                                    f"its target {target} with load {load} "
                                    f"after {retargets} extensions — taking "
                                    f"the reading, but the close target "
                                    f"looks too shallow for this frame")
                                return pos
                            retargets += 1
                            target += ZERO_RETARGET_STEP
                            _log.warning(
                                f"[EE] {side} zero pinch reached its target "
                                f"in FREE SPACE (load {load} < "
                                f"{ZERO_STALL_LOAD_MIN}) — the true close is "
                                f"deeper than the frame assumed; driving to "
                                f"{target} ({retargets}/{ZERO_RETARGET_MAX})")
                            if not await asyncio.to_thread(_write, target):
                                return pos
                            still = 0
                            last = pos
                            continue
                        return pos
                else:
                    still = 0
                last = pos
            _log.warning(f"[EE] {side} zero pinch never went still within "
                         f"{ZERO_TRAVEL_BUDGET_S:.0f}s — using the last "
                         f"reading {last}")
            return last

        first = await _pinch()
        if first is None:
            return None
        # DOUBLE-TAP: back off and pinch again. Static friction can stall
        # the first low-torque close short of the true pinch; the second
        # pass starts from a broken-loose mechanism. The deeper of the two
        # stalls is the truth (the target is inward), and a disagreement
        # beyond ZERO_AGREE_COUNTS earns a THIRD tap (08-10): two stalls that
        # disagree mean at least one of them lied, and "deeper of two liars"
        # is still a guess — one more broken-loose pass costs ~1.5 s and only
        # in the suspicious case.
        if not await asyncio.to_thread(_write, first - ZERO_BACKOFF_COUNTS):
            return first
        await asyncio.sleep(ZERO_BACKOFF_S)
        second = await _pinch()
        if second is None:
            return first
        best = max(first, second)
        if abs(second - first) > ZERO_AGREE_COUNTS:
            _log.warning(f"[EE] {side} zero pinches disagree: {first} vs "
                         f"{second} (> {ZERO_AGREE_COUNTS} counts) — early "
                         f"stall or debris; a third tap decides")
            if await asyncio.to_thread(_write, best - ZERO_BACKOFF_COUNTS):
                await asyncio.sleep(ZERO_BACKOFF_S)
                third = await _pinch()
                if third is not None:
                    best = max(best, third)
        return best

    async def _zero_and_report(self, side: str, action_id: int):
        """Background task: run zero calibration for one side, update that
        side's calibrated open/close positions, and report the outcome.

        Runs independently of the main loop's per-side action tracking so it
        doesn't block heartbeats or the other hand while it settles.
        """
        try:
            self.emit_status(StatusReport(action_id, side, "zero_gripper", "accepted"))
            measured_close = await self.run_zero_calibration(side)

            if measured_close is None:
                self.healthy[side] = False
                self.emit_status(StatusReport(
                    action_id, side, "zero_gripper", "failed",
                    error_code="ZERO_CALIBRATION_FAILED"))
                return

            self.hand_close[side] = measured_close
            self.hand_open[side] = measured_close - OPEN_OFFSET_FROM_CLOSE
            _log.info(f"[EE] {side} zeroed: close={measured_close} open={self.hand_open[side]}")

            # Return to the newly-calibrated open position rather than
            # leaving the gripper pinched shut.
            await self.run_servo_move(side, "hand_open")

            self.emit_status(StatusReport(
                action_id, side, "zero_gripper", "succeeded",
                detail=f"close={measured_close} open={self.hand_open[side]}"))
        finally:
            self.zeroing_in_progress[side] = False

    async def _grasp_and_report(self, side: str, action_id: int):
        """Background task: run the stepped, load-supervised grasp close
        for one side and report the outcome.

        Runs independently of the main loop's per-side action tracking so it
        doesn't block heartbeats or the other hand while it closes -- the
        stepped approach can take a couple of seconds.
        """
        try:
            verdict, final_pos = await self.run_grasp_close(side)
            self.last_grasp_action_id[side] = action_id
            # Preserve every inward target as a possible hold target. A false
            # or unknown classifier verdict is not permission to release a
            # possibly-held object.
            if final_pos is not None:
                self.grasp_hold_pos[side] = int(final_pos)

            self.last_grasp_detected[side] = verdict
            if verdict is None:
                self.emit_status(StatusReport(
                    action_id, side, "hand_grab", "failed",
                    error_code="GRASP_INDETERMINATE",
                    detail=f"last_target={final_pos}",
                    grasp_detected=None))
            else:
                self.emit_status(StatusReport(
                    action_id, side, "hand_grab", "succeeded",
                    grasp_detected=verdict,
                    detail=f"final_position={final_pos}"))
        except Exception as e:
            _log.error(f"[EE] Grasp error ({side}): {e}")
            self.last_grasp_action_id[side] = action_id
            self.last_grasp_detected[side] = None
            self.emit_status(StatusReport(
                action_id, side, "hand_grab", "failed",
                error_code="GRASP_ERROR", detail=str(e),
                grasp_detected=None))
        finally:
            self.grasping_in_progress[side] = False

    def _preserve_grippers_on_shutdown(self) -> None:
        """Keep every hand closed/current when the service exits.

        A known inward target is re-commanded at a gentle torque. If no such
        target exists, no position command is sent at all. In particular,
        shutdown never infers that an unknown/false verdict permits opening.
        """
        for side, hand in self.hands.items():
            if hand is None or not self.healthy[side]:
                continue
            try:
                hold_pos = self.grasp_hold_pos[side]
                if hold_pos is not None:
                    ids = self.servo_ids[side]
                    hand.set_positions(
                        ids, [hold_pos] * len(ids),
                        SPEED, ACC, SHUTDOWN_GRIP_TORQUE,
                    )
                    _log.warning(
                        f"[EE] shutdown: {side} possible payload retained "
                        f"at gentle torque {SHUTDOWN_GRIP_TORQUE}; "
                        "support it and hand_open manually"
                    )
                    time.sleep(0.3)
                else:
                    _log.warning(
                        f"[EE] shutdown: {side} left at its current command; "
                        "software shutdown never auto-opens a gripper"
                    )
            except Exception as e:
                _log.error(f"[EE] shutdown grip guard error ({side}): {e}")

    async def main_loop(self, stop: asyncio.Event):
        """The main asyncio event loop; exits (through the grip-guard
        finally) within one poll cadence of `stop` being set."""
        self.connect_hardware()
        self.request_sub.start()

        last_heartbeat_time = 0.0
        _log.info(f"[EE] Service loop active. URI: {EE_REQUEST_URI}")

        try:
            while not stop.is_set():
                now_mono = asyncio.get_event_loop().time()
                now_wall = time.time()

                # 1. Periodic Heartbeat
                if now_mono - last_heartbeat_time >= HEARTBEAT_INTERVAL_SEC:
                    self.emit_heartbeat()
                    last_heartbeat_time = now_mono

                # 1b. SAFETY PATCH (07-18): servo thermal
                # poll — one side per call, SERVO_TEMP_POLL_SEC cadence.
                if now_mono - self._temp_poll_t >= SERVO_TEMP_POLL_SEC:
                    self._temp_poll_t = now_mono
                    await asyncio.to_thread(self.poll_servo_thermals)

                # 2. Check for action completion, independently per side.
                # (hand_grab and zero_gripper are handled as their own
                # background tasks and don't go through this timer path.)
                for side in ("left", "right"):
                    report = self.active_action[side]
                    if report and now_mono >= self.completion_time_monotonic[side]:
                        report.state = "succeeded"
                        report.timestamp = time.time()
                        self.emit_status(report)
                        self.active_action[side] = None

                # 3. Poll for new IPC requests
                if self.request_sub.data_id != self.last_nng_data_id and self.request_sub.data:
                    self.last_nng_data_id = self.request_sub.data_id
                    req = ActionRequest.from_raw_dict(self.request_sub.data)

                    if not req:
                        _log.warning("[EE] Received malformed IPC packet")
                        continue

                    # -- Filter/Validate Request --
                    if req.side not in VALID_SIDES:
                        self.emit_status(StatusReport(req.action_id, req.side or "?", req.command,
                                                       "rejected", error_code="INVALID_SIDE"))
                        continue

                    # Stable action IDs are retransmitted for reliability.
                    # Ignore an exact replay before BUSY checks so one logical
                    # burst cannot be rejected as a new command mid-motion.
                    if req.action_id == self.last_processed_action_id[req.side]:
                        continue

                    age = now_wall - req.timestamp
                    if age > req.ttl_seconds:
                        self.emit_status(StatusReport(req.action_id, req.side, req.command,
                                                       "dropped_stale", detail=f"age={age:.2f}s"))

                    elif req.command not in VALID_COMMANDS:
                        self.emit_status(StatusReport(req.action_id, req.side, req.command,
                                                       "rejected", error_code="INVALID_COMMAND"))

                    elif req.command == "hand_close_to_width" and req.width_mm is None:
                        self.emit_status(StatusReport(req.action_id, req.side, req.command,
                                                       "rejected", error_code="MISSING_WIDTH"))

                    elif req.command == "hand_hold" \
                            and self.grasp_hold_pos.get(req.side) is None:
                        # SAFETY PATCH (07-18): nothing
                        # gripped — a hold command would squeeze thin air.
                        self.emit_status(StatusReport(req.action_id, req.side, req.command,
                                                       "rejected", error_code="NO_GRASP_HELD"))

                    elif self.zeroing_in_progress[req.side]:
                        self.emit_status(StatusReport(req.action_id, req.side, req.command,
                                                       "rejected", error_code="ZEROING_IN_PROGRESS"))

                    elif self.grasping_in_progress[req.side]:
                        self.emit_status(StatusReport(req.action_id, req.side, req.command,
                                                       "rejected", error_code="GRASP_IN_PROGRESS"))

                    else:
                        side = req.side
                        # -- Valid New Action --
                        if self.active_action[side]:
                            # Pre-empt existing action on this side only
                            self.active_action[side].state = "replaced"
                            self.emit_status(self.active_action[side])

                        self.last_processed_action_id[side] = req.action_id

                        if not self.healthy[side]:
                            self.emit_status(StatusReport(req.action_id, side, req.command,
                                                           "failed", error_code="HARDWARE_DISCONNECTED"))
                        elif req.command == "zero_gripper":
                            # Runs as an independent background task so it
                            # doesn't block the other side or the heartbeat
                            # while it settles.
                            self.zeroing_in_progress[side] = True
                            asyncio.create_task(self._zero_and_report(side, req.action_id))
                        elif req.command == "hand_grab":
                            # Runs as an independent background task: it's a
                            # stepped, load-supervised close, not an instant
                            # move, so it shouldn't block the other side or
                            # the heartbeat while it closes.
                            self.grasping_in_progress[side] = True
                            self.emit_status(StatusReport(req.action_id, side, req.command, "accepted"))
                            asyncio.create_task(self._grasp_and_report(side, req.action_id))
                        else:
                            # Start motion (hand_open, hand_close,
                            # hand_close_to_width, hand_hold)
                            self.emit_status(StatusReport(req.action_id, side, req.command, "accepted"))
                            success = await self.run_servo_move(
                                side, req.command, req.width_mm,
                                req.torque)  # SAFETY PATCH (07-18)

                            if success:
                                self.active_action[side] = StatusReport(req.action_id, side, req.command, "executing")
                                # Set timer for when we should check for completion
                                self.completion_time_monotonic[side] = now_mono + (HAND_MOTION_TIME_MS / 1000.0) + 0.1
                            else:
                                self.healthy[side] = False
                                self.emit_status(StatusReport(req.action_id, side, req.command,
                                                               "failed", error_code="SERIAL_TIMEOUT"))

                # Yield to OS / allow other tasks to run
                await asyncio.sleep(POLL_CADENCE_SEC)

        finally:
            self.request_sub.stop()
            # SAFETY PATCH (07-18): shutdown grip
            # guard. Shutdown is fail-closed: no classifier result, timeout or
            # exception is ever allowed to become an automatic open command.
            self._preserve_grippers_on_shutdown()
            for hand in self.hands.values():
                if hand is not None:
                    hand.close()
            self.status_pub.close()

# ---------------------------------------------------------------------------
# Viser GUI (optional, --gui flag)
# ---------------------------------------------------------------------------

def _run_viser_gui(client: "EEServiceClient", port: int) -> None:
    """Daemon thread: viser web UI for manual per-hand command testing."""
    import viser
    server = viser.ViserServer(host="0.0.0.0", port=port)
    _log.info(f"[EE] Viser GUI at http://0.0.0.0:{port}")

    side_widgets = {}
    for side in ("left", "right"):
        with server.gui.add_folder(f"{side.capitalize()} Hand"):
            btn_open = server.gui.add_button("Open")
            btn_grab = server.gui.add_button("Grab")
            btn_close = server.gui.add_button("Close")
            width_slider = server.gui.add_slider(
                "Close to width (mm)",
                min=GRIPPER_MIN_APERTURE_MM,
                max=GRIPPER_MAX_APERTURE_MM,
                step=1.0,
                initial_value=GRIPPER_MAX_APERTURE_MM / 2,
            )
            btn_width = server.gui.add_button("Close to width")
            btn_zero = server.gui.add_button("Zero Gripper")
        side_widgets[side] = (btn_open, btn_grab, btn_close, width_slider, btn_width, btn_zero)

        def _make_handlers(s):
            def _open(_): client.send_action(s, "hand_open")
            def _grab(_): client.send_action(s, "hand_grab")
            def _close(_): client.send_action(s, "hand_close")
            def _to_width(_):
                _, _, _, slider, _, _ = side_widgets[s]
                client.send_action(s, "hand_close_to_width", width_mm=slider.value)
            def _zero(_): client.send_action(s, "zero_gripper")
            return _open, _grab, _close, _to_width, _zero

        _open, _grab, _close, _to_width, _zero = _make_handlers(side)
        btn_open.on_click(_open)
        btn_grab.on_click(_grab)
        btn_close.on_click(_close)
        btn_width.on_click(_to_width)
        btn_zero.on_click(_zero)

    with server.gui.add_folder("Status"):
        status_md = server.gui.add_markdown("**Service:** initializing…")
        grasp_md = server.gui.add_markdown("**Grasp:** L: — | R: —")

    while True:
        client.poll()
        status_md.content = f"**Service:** {'healthy ✓' if client.alive else 'OFFLINE ✗'}"
        grasp_md.content = (
            f"**Grasp:** L: {client.grasp_detected['left']} | "
            f"R: {client.grasp_detected['right']}"
        )
        time.sleep(0.1)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass
class CLIArgs:
    """End-effector servo service - run alongside humanoid_real_env.py."""
    left_port: str = HAND_SERIAL_PORT_LEFT
    """Serial port for the left-hand ST servo (e.g. /dev/ttyACMservoLeft)."""
    right_port: str = HAND_SERIAL_PORT_RIGHT
    """Serial port for the right-hand ST servo (e.g. /dev/ttyACMservoRight)."""
    gui: bool = False
    """Launch viser control GUI for manual testing (http://localhost:8080)."""
    gui_port: int = 8080
    """Viser web server port."""

async def entrypoint():
    """Process entry: parse CLIArgs, build the service (plus the viser GUI
    thread with --gui), install the two-stage SIGINT/SIGTERM handler (first =
    graceful stop through the grip guard, second = hard exit) and run
    `main_loop` until stopped."""
    args = tyro.cli(CLIArgs)
    service = EndEffectorService(args.left_port, args.right_port)

    client: Optional[EEServiceClient] = None
    if args.gui:
        client = EEServiceClient()
        threading.Thread(
            target=_run_viser_gui, args=(client, args.gui_port), daemon=True
        ).start()

    # Clean shutdown. The old handler here — create_task(sys.exit(0)) —
    # evaluated sys.exit inside the loop callback: main_loop's finally still
    # ran (so the grip guard fired), but the interpreter then blocked forever
    # joining the SDKs' non-daemon threads and every further Ctrl+C was
    # swallowed. Contract now: first SIGINT/SIGTERM requests a graceful stop
    # (grip guard runs), a second one hard-exits no matter where the process
    # is stuck. signal.signal rather than loop.add_signal_handler so the
    # second hit still fires while sync cleanup blocks the loop thread.
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    hits = {"n": 0}

    def _on_signal(signum, frame):
        hits["n"] += 1
        if hits["n"] > 1:
            os.write(2, b"\n[EE] second signal: hard exit "
                        b"(grip guard may not have finished)\n")
            os._exit(130)
        loop.call_soon_threadsafe(stop.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    try:
        await service.main_loop(stop)
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    exit_code = 0
    try:
        asyncio.run(entrypoint())
    except KeyboardInterrupt:
        exit_code = 130  # Ctrl+C before the handlers were installed
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    # By here the grip guard and hardware close have run. The serial/pynng
    # SDKs leave non-daemon threads behind, and letting the interpreter try
    # to join them was the Ctrl+C wall — flush and leave without them.
    _log._listener.stop()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

# python humanoid_end_effector_service.py --left-port /dev/ttyACMservoLeft --right-port /dev/ttyACMservoRight --gui
