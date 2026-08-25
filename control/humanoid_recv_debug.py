"""Receiver Debugger for humanoid_real_env.py telemetry.

Subscribes to the outbound telemetry channel published by humanoid_real_env and
prints live stats: message rate, staleness, per-field min/max/mean, and stamp
latency.  Useful for verifying the robot is publishing correctly before or
during a deployment run.

Channel monitored:
  9870  — main telemetry (joint states, IMU, commands, vicon, EE poses, stamp)

Usage (on workstation, replace <robot_ip> with the robot's IP):
    python humanoid_recv_debug.py --robot-ip <robot-host>

    # local loopback (when running on the robot itself):
    python humanoid_recv_debug.py --robot-ip 127.0.0.1

    # show only a subset of fields:
    python humanoid_recv_debug.py --robot-ip <robot_ip> --fields joint_pos imu_quat_wxyz command

    # open viser window showing live robot pose (http://localhost:8080):
    python humanoid_recv_debug.py --robot-ip <robot_ip> --gui

    # open camera viewer in viser background (default: 2 cameras on ports 5555, 5556):
    python humanoid_recv_debug.py --robot-ip <robot_ip> --camera

    # override camera count or base port:
    python humanoid_recv_debug.py --robot-ip <robot_ip> --camera --n-cameras 3 --camera-port 5555

Exit:
    Ctrl+C
"""


import collections
import dataclasses
import sys
import threading
import time

import numpy as np
import tyro

from ipc.publisher import NNGSubscriber

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from humanoid_model import MJCF_MODEL_PATH

TELEMETRY_PORT      = 9870  # mirrors humanoid_site.TELEMETRY_PORT (kept literal here)
RATE_WINDOW_SEC     = 1.0   # sliding window length for Hz estimate
STAT_WINDOW_SAMPLES = 200   # rolling buffer depth for per-field min/max/mean/std

TELEMETRY_FIELDS = [
    "base_lin_vel", "base_ang_vel", "projected_gravity",
    "joint_pos", "joint_vel", "joint_effort", "joint_effort_ref",
    "actions", "command", "imu_quat_wxyz", "vicon_pos",
    "wb_pos", "wb_quat_wxyz",
]

# ANSI escape codes
_ANSI_GREEN  = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED    = "\033[91m"
_ANSI_CYAN   = "\033[96m"
_ANSI_BOLD   = "\033[1m"
_ANSI_RESET  = "\033[0m"
_ANSI_CLEAR  = "\033[2J\033[H"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FieldStats:
    """Rolling stats for one telemetry field."""
    min:       np.ndarray
    max:       np.ndarray
    mean:      np.ndarray
    std:       np.ndarray
    dim:       int  # number of elements per sample (1 for scalars)
    n_samples: int  # how many samples are in the buffer


class ChannelStats:
    """Receiver-side rolling telemetry stats."""

    def __init__(self):
        self.msg_count = 0
        self.last_data: dict | None = None
        self._recv_timestamps: collections.deque = collections.deque()
        self._field_sample_bufs: dict[str, collections.deque] = {}

    def update(self, data: dict):
        """Add one telemetry packet to rolling stats."""
        now = time.monotonic()
        self.msg_count += 1
        self.last_data = data
        timestamps = self._recv_timestamps
        timestamps.append(now)
        # prune old timestamps here so hz is a cheap O(1) read
        while timestamps and timestamps[0] < now - RATE_WINDOW_SEC:
            timestamps.popleft()
        for key, val in data.items():
            if val is None or isinstance(val, dict):
                continue
            try:
                sample = np.asarray(val, dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            self._field_sample_bufs.setdefault(key, collections.deque(maxlen=STAT_WINDOW_SAMPLES)).append(sample)

    def reset(self):
        self.msg_count = 0
        self._recv_timestamps.clear()
        self._field_sample_bufs.clear()

    @property
    def hz(self) -> float:
        timestamps = self._recv_timestamps
        if len(timestamps) < 2:
            return 0.0
        dt = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / dt

    def staleness_sec(self) -> float | None:
        """Time since last packet in seconds."""
        return None if not self._recv_timestamps else time.monotonic() - self._recv_timestamps[-1]

    def field_stats(self, field: str) -> FieldStats | None:
        """Return rolling stats for one field, or None if no samples."""
        buf = self._field_sample_bufs.get(field)
        if not buf:
            return None
        samples = np.stack(list(buf))  # (N, dim)
        return FieldStats(
            min=samples.min(0), max=samples.max(0),
            mean=samples.mean(0), std=samples.std(0),
            dim=samples.shape[1], n_samples=len(buf),
        )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _level_color(val: float | None, warn: float, crit: float, high_is_good: bool = True) -> str:
    """Return ANSI color: green/yellow/red based on thresholds.
    high_is_good=True  → large values are green (e.g. Hz).
    high_is_good=False → small values are green (e.g. staleness).
    """
    if val is None:
        return _ANSI_RED
    if high_is_good:
        return _ANSI_GREEN if val >= warn else (_ANSI_YELLOW if val >= crit else _ANSI_RED)
    return _ANSI_GREEN if val <= warn else (_ANSI_YELLOW if val <= crit else _ANSI_RED)


def _fmt(arr, float_fmt=".3f") -> str:
    """Format vector/scalar compactly for terminal display."""
    arr = np.asarray(arr).ravel()
    values = arr if arr.size <= 6 else arr[:4]
    body = ", ".join(f"{v:{float_fmt}}" for v in values)
    suffix = f" … ({arr.size})" if arr.size > 6 else ""
    return f"[{body}{suffix}]"


def _render(
    stats: ChannelStats,
    fields_filter: list[str] | None,
) -> str:
    lines = [f"{_ANSI_BOLD}{_ANSI_CYAN}=== humanoid_real_env receiver debugger ==={_ANSI_RESET}"]

    staleness = stats.staleness_sec()
    staleness_color = _level_color(staleness, warn=0.1, crit=0.5, high_is_good=False)
    rate_color      = _level_color(stats.hz,  warn=80,  crit=40,  high_is_good=True)
    staleness_str   = f"{staleness*1000:.1f} ms ago" if staleness is not None else "never"
    lines.append(
        f"\n{_ANSI_BOLD}[Port {TELEMETRY_PORT}]{_ANSI_RESET}  "
        f"msgs={stats.msg_count}  "
        f"rate={rate_color}{stats.hz:.1f} Hz{_ANSI_RESET}  "
        f"last={staleness_color}{staleness_str}{_ANSI_RESET}"
    )

    if stats.last_data is None:
        return "\n".join(lines) + f"\n  {_ANSI_RED}no data received yet{_ANSI_RESET}"

    stamp = stats.last_data.get("stamp")
    if stamp is not None:
        stamp_latency_sec = time.time() - stamp
        if stamp_latency_sec < -0.001:
            stamp_latency_str = f"{_ANSI_RED}{stamp_latency_sec*1000:.1f} ms  [clock skew: robot ahead]{_ANSI_RESET}"
        else:
            stamp_latency_str = f"{stamp_latency_sec*1000:.1f} ms"
    else:
        stamp_latency_str = "n/a"
    lines.append(f"  {'stamp_latency':25s}  {stamp_latency_str}")

    for field in TELEMETRY_FIELDS:
        if fields_filter and field not in fields_filter:
            continue
        val = stats.last_data.get(field)
        if val is None:
            lines.append(f"  {field:25s}  {_ANSI_YELLOW}None{_ANSI_RESET}")
            continue
        field_stat = stats.field_stats(field)
        range_str = ""
        if field_stat:
            if field_stat.dim > 1:
                range_str = (f"  range=[{field_stat.min.min():.3f}, {field_stat.max.max():.3f}]"
                             f"  std_max={field_stat.std.max():.4f}")
            else:
                range_str = (f"  range=[{float(field_stat.min):.3f}, {float(field_stat.max):.3f}]"
                             f"  std={float(field_stat.std):.4f}")
        lines.append(f"  {field:25s}  last={_fmt(val)}{range_str}")

    ee = stats.last_data.get("ee")
    if ee:
        lines.append(f"\n  --- EE poses (base FLU frame) ---")
        for side, data in ee.items():
            pos_a = data.get("pos_actual")
            pos_t = data.get("pos_target")
            quat_a = data.get("quat_actual")
            lines.append(f"  {side:6s}  actual={_fmt(pos_a)}  target={_fmt(pos_t) if pos_t else 'null'}  quat={_fmt(quat_a)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Viser GUI (robot + camera)
# ---------------------------------------------------------------------------

from humanoid_site import VISUAL_SERVOING_ROOT as _VS_ROOT
import humanoid_site as _site

_VS_PATH = str(_VS_ROOT)


def _pad_to_aspect(frame: np.ndarray, vp_w: int, vp_h: int) -> np.ndarray:
    """Letterbox/pillarbox frame to the viewport's aspect ratio with black bars."""
    h, w = frame.shape[:2]
    target_aspect = vp_w / vp_h
    frame_aspect = w / h
    if frame_aspect > target_aspect:
        # frame wider than viewport: add black bars top and bottom
        new_h = int(w / target_aspect)
        pad = (new_h - h) // 2
        return np.pad(frame, ((pad, new_h - h - pad), (0, 0), (0, 0)))
    else:
        # frame taller than viewport: add black bars left and right
        new_w = int(h * target_aspect)
        pad = (new_w - w) // 2
        return np.pad(frame, ((0, 0), (pad, new_w - w - pad), (0, 0)))


def _viser_thread(
    stats: ChannelStats,
    args: "Args",
    quit_flag: threading.Event,
) -> None:
    """Run a viser server showing live robot pose (--gui) and/or camera feed (--camera).

    Robot: HumanoidViserViz renders the MJCF mesh at 30 Hz via update_from_packet().
    Camera: frames are grabbed in an inner daemon thread and pushed as the viser
    background image (BGR→RGB, tiled horizontally for multiple cameras).
    Viewport aspect is read directly from client.camera.image_width/height each frame
    and used to center-crop the image so it fills the background without stretching.
    Quit is driven by the terminal 'q' hotkey; the viser window has no quit hook.
    """
    import viser
    from humanoid_utils import HumanoidViserViz, get_joint_info_from_mjcf

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print(f"[viser] server at http://0.0.0.0:{server.get_port()}")

    joint_names, _, _ = get_joint_info_from_mjcf(MJCF_MODEL_PATH)
    viz = HumanoidViserViz(
        mjcf_path=MJCF_MODEL_PATH,
        joint_names=joint_names,
        show_robot=True,
        server=server,
    )

    # list wrapper so the closure can rebind latest_frame[0] across threads
    # (a plain variable would be read-only inside the nested _cam_capture closure)
    latest_frame: list[np.ndarray | None] = [None]

    if args.camera:
        def _cam_capture():
            if _VS_PATH not in sys.path:
                sys.path.insert(0, _VS_PATH)
            import cv2
            from camera_streaming_utils import MultiStreamingCameraClient

            cam_client = MultiStreamingCameraClient(args.robot_ip, args.n_cameras, args.camera_port)
            cam_client.initialize_all()
            if not cam_client.wait_all(timeout=6.0):
                print(f"[camera] ERROR — not all {args.n_cameras} cameras connected within 6s")
                return
            try:
                while not quit_flag.is_set():
                    cam_client.capture_all_parallel(timeout=0.5)
                    frames = []
                    for cam_buf in cam_client:
                        frame = cam_buf.get_frame_buffer()
                        if frame is None:
                            continue
                        if frame.ndim == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        frames.append(frame)
                    if not frames:
                        continue
                    bgr = np.hstack(frames) if len(frames) > 1 else frames[0]
                    latest_frame[0] = bgr[:, :, ::-1]  # BGR → RGB for viser
            finally:
                cam_client.cleanup()

        threading.Thread(target=_cam_capture, daemon=True).start()

    last_msg_count: int | None = None
    while not quit_flag.is_set():
        packet = stats.last_data
        if packet is not None and stats.msg_count != last_msg_count:
            last_msg_count = stats.msg_count
            viz.update_from_packet(packet)

        frame = latest_frame[0]
        if frame is not None:
            for client in server.get_clients().values():
                vp_w = client.camera.image_width
                vp_h = client.camera.image_height
                if vp_w == 0 or vp_h == 0:
                    continue
                padded = _pad_to_aspect(frame, vp_w, vp_h)
                client.scene.set_background_image(padded, format="jpeg", jpeg_quality=80)

        time.sleep(1 / 30)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    robot_ip:    str              = _site.ROBOT_IP  # IP of the robot running humanoid_real_env
    refresh_hz:  float            = 10.0             # display refresh rate
    fields:      list[str] | None = None             # whitelist of field names to show (default: all)
    print_stats: bool             = False             # print terminal telemetry dashboard
    gui:         bool             = True             # open viser viewer showing live robot pose (http://localhost:8080)
    camera:      bool             = True             # stream camera feed(s) as viser background image
    camera_port: int              = 5555             # base port for camera stream(s)
    n_cameras:   int              = 2                # number of cameras (consecutive ports)


def main():
    args = tyro.cli(Args)

    stats = ChannelStats()
    sub = NNGSubscriber(f"tcp://{args.robot_ip}:{TELEMETRY_PORT}")
    sub.start()

    quit_flag = threading.Event()

    def _monitor():
        # Pull latest packet and update stats.
        last_data_id: int | None = None
        while not quit_flag.is_set():
            if sub.data_id != last_data_id and sub.data is not None:
                last_data_id = sub.data_id
                stats.update(sub.data)
            time.sleep(0.001)

    threading.Thread(target=_monitor, daemon=True).start()

    if args.gui or args.camera:
        threading.Thread(target=_viser_thread, args=(stats, args, quit_flag), daemon=True).start()

    period = 1.0 / max(args.refresh_hz, 0.5)
    try:
        while True:
            if args.print_stats:
                sys.stdout.write(_ANSI_CLEAR + _render(stats, args.fields) + "\n")
                sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        quit_flag.set()
        sub.stop()
        print("\nDebugger stopped.")


if __name__ == "__main__":
    main()
