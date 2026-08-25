#!/usr/bin/env python3
"""
DearPyGui control panel (file name sic) for sending velocity commands over UDP
to the sim-side test receiver (legged_env_v2 `mj_envs/deploy/test_deployment.py`,
not part of this repository), or directly to humanoid_real_env.py on the real
robot via pynng (--robot-ip).

Needs legged_env_v2, located through humanoid_site.LEGGED_ENV_ROOT (override
(no upstream checkout needed since 2026-08-24: `ipc.publisher`
(DataPublisher) is imported from that repository, not from this one. Needs
dearpygui (optional requirement; exits with a one-line hint when absent).

Local test mode (default):
    python gampad_gui_control.py
    → Publishes to udp://127.0.0.1:9869

Real-robot mode:
    python gampad_gui_control.py --robot-ip <ROBOT_IP>
    → Publishes nav velocity to tcp://*:9873  (robot runs: --high_level_controller_ip <THIS_MACHINE_IP>)
    → Publishes arm EE  to tcp://*:9874  (robot runs: --arm_sender_ip <THIS_MACHINE_IP> --use-ik)
"""

import sys
import dataclasses
from pathlib import Path
import time
import math
import tyro
try:
    import dearpygui.dearpygui as dpg
except ImportError as e:
    print(f"dearpygui is optional and not installed: pip install dearpygui ({e})", file=sys.stderr)
    raise SystemExit(2)

# control/ on sys.path for the two transports: NNGPublisher (pynng, the robot's
# command sockets) and DataPublisher (UDP, the sim-side telemetry transport).
# Since 2026-08-24 both come from ipc.publisher; the upstream
# envs.common.publisher.DataPublisher this panel used to import is the same
# class with a FIFO queue instead of latest-value-wins, and nothing here
# depended on the difference.
from humanoid_site import CONTROL_ROOT as _CTRL_ROOT  # noqa: E402
sys.path.insert(0, str(_CTRL_ROOT))

from ipc.publisher import DataPublisher, NNGPublisher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    robot_ip: str | None = None
    """Robot IP for real-robot mode.
    When set, also publishes to pynng ports 9873/9874.
    Leave unset for local UDP test mode only."""

args = tyro.cli(Args)

ROBOT_MODE = args.robot_ip is not None

# ---------------------------------------------------------------------------
# Publishers
# ---------------------------------------------------------------------------
# Local test publisher (unchanged)
PUBLISHER_URL = "udp://127.0.0.1:9869"
publisher = DataPublisher(PUBLISHER_URL, encoding="msgpack")
print(f"Publishing commands to {PUBLISHER_URL}")

# Real-robot pynng publishers (binds; robot connects)
if ROBOT_MODE:
    cmd_publisher = NNGPublisher("tcp://*:9873")   # nav_cmd → robot --high_level_controller_ip:9873
    arm_publisher = NNGPublisher("tcp://*:9874")   # arm EE  → robot arm_sender_ip:9874
    print(f"Real-robot mode: publishing nav to tcp://*:9873, arm to tcp://*:9874")
    print(f"  Robot should connect with: --high_level_controller_ip <THIS_IP> --arm_sender_ip <THIS_IP> --use-ik")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
current_cmd = [0.0, 0.0, 0.0]
# Left Arm [X, Y, Z], Right Arm [X, Y, Z] relative to pelvis (body frame)
current_ee_pos = [[0.15, 0.20, 0.05], [0.15, -0.20, 0.05]]
# Left Arm [R, P, Y], Right Arm [R, P, Y] Euler angles (rad) in body frame
current_ee_rpy = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def _rpy_to_quat_wxyz(rpy):
    """Roll-Pitch-Yaw (intrinsic XYZ) -> wxyz quaternion."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y_, z]


def _publish_all():
    """Publish current state to all active publishers."""
    ee_quat = [_rpy_to_quat_wxyz(current_ee_rpy[0]),
               _rpy_to_quat_wxyz(current_ee_rpy[1])]

    # Local UDP test publisher
    publisher.publish({
        "cmd": current_cmd,
        "ee_pos": current_ee_pos,
        "ee_quat": ee_quat,
    })

    # Real-robot pynng publishers
    if ROBOT_MODE:
        cmd_publisher.publish({"nav_cmd": current_cmd})
        arm_publisher.publish({
            "arm_targets": {
                "ee_pos":  current_ee_pos,
                "ee_quat": ee_quat,
            }
        })


def update_cmd(sender, app_data, user_data):
    """Callback for velocity sliders."""
    idx = user_data
    current_cmd[idx] = float(app_data)


def update_ee_pos(sender, app_data, user_data):
    """Callback for arm EE target sliders."""
    arm_idx, axis_idx = user_data
    current_ee_pos[arm_idx][axis_idx] = float(app_data)


def update_ee_rpy(sender, app_data, user_data):
    """Callback for arm EE orientation (RPY) sliders."""
    arm_idx, axis_idx = user_data
    current_ee_rpy[arm_idx][axis_idx] = float(app_data)


def reset_cmd():
    """Callback for zero button."""
    current_cmd[0] = 0.0
    current_cmd[1] = 0.0
    current_cmd[2] = 0.0
    dpg.set_value("slider_vx", 0.0)
    dpg.set_value("slider_vy", 0.0)
    dpg.set_value("slider_wz", 0.0)

    # Reset arms
    current_ee_pos[0] = [0.15, 0.20, 0.05]
    current_ee_pos[1] = [0.15, -0.20, 0.05]
    dpg.set_value("slider_lx", 0.15)
    dpg.set_value("slider_ly", 0.20)
    dpg.set_value("slider_lz", 0.05)
    dpg.set_value("slider_rx", 0.15)
    dpg.set_value("slider_ry", -0.20)
    dpg.set_value("slider_rz", 0.05)
    current_ee_rpy[0] = [0.0, 0.0, 0.0]
    current_ee_rpy[1] = [0.0, 0.0, 0.0]
    for tag in ("slider_lr", "slider_lp", "slider_lyaw", "slider_rr", "slider_rp", "slider_ryaw"):
        dpg.set_value(tag, 0.0)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
dpg.create_context()

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 5, category=dpg.mvThemeCat_Core)

dpg.bind_theme(global_theme)

with dpg.window(label="Control Panel", width=680, height=610, no_collapse=True, no_close=True):
    dpg.add_text("Robot Velocity Commander")

    # Mode indicator
    if ROBOT_MODE:
        mode_label = f"Mode: Real robot @ {args.robot_ip}  (nav:9873, arm:9874)"
    else:
        mode_label = "Mode: Local test (UDP 9869)"
    with dpg.theme() as mode_theme:
        with dpg.theme_component(dpg.mvText):
            color = [80, 200, 80] if ROBOT_MODE else [200, 200, 80]
            dpg.add_theme_color(dpg.mvThemeCol_Text, color)
    mode_text = dpg.add_text(mode_label)
    dpg.bind_item_theme(mode_text, mode_theme)

    dpg.add_separator()
    dpg.add_spacer(height=5)

    # Velocity Sliders
    dpg.add_slider_float(label="Linear X (Fwd/Rev)", default_value=0.0, min_value=-3.0, max_value=3.0,
                         tag="slider_vx", callback=update_cmd, user_data=0, format="%.2f m/s")
    dpg.add_slider_float(label="Linear Y (Left/Right)", default_value=0.0, min_value=-2.0, max_value=2.0,
                         tag="slider_vy", callback=update_cmd, user_data=1, format="%.2f m/s")
    dpg.add_slider_float(label="Angular Z (Turn)", default_value=0.0, min_value=-2.0, max_value=2.0,
                         tag="slider_wz", callback=update_cmd, user_data=2, format="%.2f rad/s")

    dpg.add_spacer(height=10)
    dpg.add_text("IK Arm Target Control (Body Frame)")
    dpg.add_separator()

    with dpg.group(horizontal=True):
        with dpg.group():
            dpg.add_text("Left Arm")
            dpg.add_slider_float(label="X (Fwd)", default_value=0.15, min_value=0.0, max_value=0.4,
                                 tag="slider_lx", callback=update_ee_pos, user_data=(0, 0), format="%.2f m", width=150)
            dpg.add_slider_float(label="Y (Left)", default_value=0.20, min_value=0.0, max_value=0.4,
                                 tag="slider_ly", callback=update_ee_pos, user_data=(0, 1), format="%.2f m", width=150)
            dpg.add_slider_float(label="Z (Up)", default_value=0.05, min_value=-0.2, max_value=0.4,
                                 tag="slider_lz", callback=update_ee_pos, user_data=(0, 2), format="%.2f m", width=150)
            dpg.add_spacer(height=4)
            dpg.add_text("  Orientation (RPY, body frame)")
            dpg.add_slider_float(label="Roll",  default_value=0.0, min_value=-1.57, max_value=1.57,
                                 tag="slider_lr", callback=update_ee_rpy, user_data=(0, 0), format="%.2f rad", width=150)
            dpg.add_slider_float(label="Pitch", default_value=0.0, min_value=-1.57, max_value=1.57,
                                 tag="slider_lp", callback=update_ee_rpy, user_data=(0, 1), format="%.2f rad", width=150)
            dpg.add_slider_float(label="Yaw",   default_value=0.0, min_value=-3.14, max_value=3.14,
                                 tag="slider_lyaw", callback=update_ee_rpy, user_data=(0, 2), format="%.2f rad", width=150)

        dpg.add_spacer(width=20)

        with dpg.group():
            dpg.add_text("Right Arm")
            dpg.add_slider_float(label="X (Fwd)", default_value=0.15, min_value=0.0, max_value=0.4,
                                 tag="slider_rx", callback=update_ee_pos, user_data=(1, 0), format="%.2f m", width=150)
            dpg.add_slider_float(label="Y (Right)", default_value=-0.20, min_value=-0.4, max_value=0.0,
                                 tag="slider_ry", callback=update_ee_pos, user_data=(1, 1), format="%.2f m", width=150)
            dpg.add_slider_float(label="Z (Up)", default_value=0.05, min_value=-0.2, max_value=0.4,
                                 tag="slider_rz", callback=update_ee_pos, user_data=(1, 2), format="%.2f m", width=150)
            dpg.add_spacer(height=4)
            dpg.add_text("  Orientation (RPY, body frame)")
            dpg.add_slider_float(label="Roll",  default_value=0.0, min_value=-1.57, max_value=1.57,
                                 tag="slider_rr", callback=update_ee_rpy, user_data=(1, 0), format="%.2f rad", width=150)
            dpg.add_slider_float(label="Pitch", default_value=0.0, min_value=-1.57, max_value=1.57,
                                 tag="slider_rp", callback=update_ee_rpy, user_data=(1, 1), format="%.2f rad", width=150)
            dpg.add_slider_float(label="Yaw",   default_value=0.0, min_value=-3.14, max_value=3.14,
                                 tag="slider_ryaw", callback=update_ee_rpy, user_data=(1, 2), format="%.2f rad", width=150)

    dpg.add_spacer(height=10)
    dpg.add_separator()
    dpg.add_spacer(height=10)

    # Red "STOP" button
    with dpg.theme() as stop_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [200, 50, 50])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [230, 80, 80])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [150, 30, 30])

    btn = dpg.add_button(label="STOP (Zero Commands & Reset Arms)", callback=reset_cmd, width=-1, height=40)
    dpg.bind_item_theme(btn, stop_theme)

dpg.create_viewport(title='Robot Control Panel', width=700, height=650, resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()

# ---------------------------------------------------------------------------
# Render loop — publish at 50 Hz
# ---------------------------------------------------------------------------
target_fps = 50.0
sleep_time = 1.0 / target_fps

try:
    while dpg.is_dearpygui_running():
        _publish_all()
        dpg.render_dearpygui_frame()
        time.sleep(sleep_time)
except KeyboardInterrupt:
    pass
finally:
    dpg.destroy_context()
    if ROBOT_MODE:
        cmd_publisher.close()
        arm_publisher.close()
    print("Commander stopped.")
