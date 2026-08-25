#!/usr/bin/env python3
"""Minimal visual manipulation debug GUI.

Subscribes to detected target poses from humanoid_monitor.py over IPC and publishes
arm IK targets and EE action commands to humanoid_real_env.py (port 9874).

Both this process and humanoid_monitor.py must run on the same machine (IPC transport).

Usage:
    python visual_manip_debug_gui.py

Robot prerequisites:
    humanoid_real_env.py --use-ik --arm_sender_ip <THIS_MACHINE_IP> [--ee-service]
    humanoid_end_effector_service.py (if using EE action buttons)

Connection health:
    "Monitor: CONNECTED" = packet received within stale_timeout_s
    "Monitor: NO DATA"   = no packet, or monitor not publishing on IPC socket
"""

from __future__ import annotations

import dataclasses
import sys
import time

try:
    import dearpygui.dearpygui as dpg
except ImportError as e:
    print(f"dearpygui is optional and not installed: pip install dearpygui ({e})", file=sys.stderr)
    raise SystemExit(2)
import tyro

from ipc.publisher import NNGPublisher, NNGSubscriber

DETECTION_IPC_URL = "ipc:///tmp/humanoid_detections.sock"


@dataclasses.dataclass
class MonitorTargetPose:
    key: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    n_inliers: int
    camera_port: int


@dataclasses.dataclass
class Args:
    monitor_url: str = DETECTION_IPC_URL
    arm_port: int = 9874
    publish_hz: float = 20.0
    stale_timeout_s: float = 0.2
    min_inliers: int = 3  # reject poses with fewer tag inliers than this


LEFT_DEFAULT_POS  = [0.15,  0.20, 0.05]
RIGHT_DEFAULT_POS = [0.15, -0.20, 0.05]
NEUTRAL_QUAT      = [1.0, 0.0, 0.0, 0.0]


def _parse_targets(raw: dict | None) -> dict[str, MonitorTargetPose]:
    """Parse the 'targets' field of a MonitorDetectionPacket dict into typed poses."""
    if not raw:
        return {}
    return {
        key: MonitorTargetPose(
            key=str(pose.get("key", key)),
            pos=tuple(float(v) for v in pose["pos"]),
            quat_wxyz=tuple(float(v) for v in pose["quat_wxyz"]),
            n_inliers=int(pose.get("n_inliers", 0)),
            camera_port=int(pose.get("camera_port", -1)),
        )
        for key, pose in raw.get("targets", {}).items()
    }


def main() -> None:
    args = tyro.cli(Args)

    det_sub = NNGSubscriber(args.monitor_url)
    det_sub.start()
    arm_pub = NNGPublisher(f"tcp://*:{args.arm_port}")

    last_publish_t = 0.0
    last_action = "none"
    last_data_id = -1
    last_recv_t = 0.0
    sticky_target = ""
    last_tgt_pos: dict[str, list[float]] = {}

    dpg.create_context()
    with dpg.font_registry():
        _font = dpg.add_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    dpg.bind_font(_font)

    with dpg.window(label="Visual Manip Debug GUI", width=950, height=1500, no_collapse=True, no_close=True):
        dpg.add_text("Monitor -> GUI -> 9874 (arm_targets / ee_action)")
        dpg.add_text(f"Monitor: {args.monitor_url}")
        dpg.add_text(f"Arm pub: tcp://*:{args.arm_port}")
        dpg.add_separator()

        dpg.add_text("Target / Arm")
        dpg.add_listbox(items=[], default_value="", tag="target_combo", width=900, num_items=10)
        dpg.add_radio_button(items=["left", "right"], default_value="right", tag="arm_radio", horizontal=True)

        dpg.add_spacer(height=15)
        dpg.add_text("Base-frame offsets (m)")
        dpg.add_slider_float(label="dx", default_value=0.0, min_value=-0.20, max_value=0.20, tag="off_x", width=510)
        dpg.add_slider_float(label="dy", default_value=0.0, min_value=-0.20, max_value=0.20, tag="off_y", width=510)
        dpg.add_slider_float(label="dz", default_value=0.0, min_value=-0.20, max_value=0.20, tag="off_z", width=510)

        dpg.add_spacer(height=15)
        dpg.add_checkbox(label="Enable streaming arm_targets", default_value=False, tag="stream_enable")

        def _stop_cb():
            dpg.set_value("stream_enable", False)
            arm_pub.publish({"arm_targets": None})  # immediate ramp-to-default on robot

        dpg.add_button(label="STOP", callback=_stop_cb, width=450, height=66)

        dpg.add_spacer(height=15)
        dpg.add_text("EE actions")

        def _send_action(cmd: str):
            nonlocal last_action
            arm_pub.publish({"ee_action": cmd})
            last_action = cmd

        with dpg.group(horizontal=True):
            dpg.add_button(label="Open",  callback=lambda: _send_action("hand_open"),  width=225)
            dpg.add_button(label="Grab",  callback=lambda: _send_action("hand_grab"),  width=225)
            dpg.add_button(label="Close", callback=lambda: _send_action("hand_close"), width=225)

        dpg.add_separator()
        dpg.add_text("Status")
        dpg.add_text("Monitor: NO DATA", tag="st_monitor")
        dpg.add_text("target_visible: False", tag="st_visible")
        dpg.add_text("target_pos: n/a", tag="st_pos")
        dpg.add_text("inliers/cam: n/a", tag="st_inliers")
        dpg.add_text("last_publish: n/a", tag="st_pub")
        dpg.add_text("last_ee_action: none", tag="st_action")
        dpg.add_text("actual_pos:  n/a", tag="st_actual")
        dpg.add_text("err_xyz (m): n/a", tag="st_err_xyz")
        dpg.add_text("err_norm:    n/a", tag="st_err_norm")

    dpg.create_viewport(title="visual_manip_debug_gui", width=950, height=1500, resizable=True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    loop_period = 1.0 / max(args.publish_hz, 1.0)

    try:
        while dpg.is_dearpygui_running():
            now = time.monotonic()

            # Freshness via data_id — avoids cross-process monotonic comparison.
            cur_data_id = det_sub.data_id
            if cur_data_id != last_data_id:
                last_data_id = cur_data_id
                last_recv_t = now
            monitor_connected = last_recv_t > 0 and (now - last_recv_t) <= args.stale_timeout_s
            dpg.set_value("st_monitor", f"Monitor: {'CONNECTED' if monitor_connected else 'NO DATA'}")

            targets = _parse_targets(det_sub.data)
            target_keys = sorted(targets.keys())
            ui_selected = str(dpg.get_value("target_combo") or "")
            if ui_selected and ui_selected != sticky_target:
                sticky_target = ui_selected

            current_items = dpg.get_item_configuration("target_combo").get("items", [])
            if target_keys != current_items:
                dpg.configure_item("target_combo", items=target_keys)

            if sticky_target in target_keys:
                if ui_selected != sticky_target:
                    dpg.set_value("target_combo", sticky_target)
            elif target_keys:
                sticky_target = target_keys[0]
                dpg.set_value("target_combo", sticky_target)
            else:
                sticky_target = ""
                dpg.set_value("target_combo", "")

            selected_target = sticky_target
            selected_arm    = str(dpg.get_value("arm_radio") or "right")
            offset          = [float(dpg.get_value("off_x")), float(dpg.get_value("off_y")), float(dpg.get_value("off_z"))]
            stream_enabled  = bool(dpg.get_value("stream_enable"))

            pose = targets.get(selected_target)
            is_fresh = monitor_connected and pose is not None and pose.n_inliers >= args.min_inliers

            dpg.set_value("st_visible", f"target_visible: {is_fresh}")
            if pose is None:
                dpg.set_value("st_pos",     "target_pos: n/a")
                dpg.set_value("st_inliers", "inliers/cam: n/a")
            else:
                dpg.set_value("st_pos",     f"target_pos: [{pose.pos[0]:+.3f}, {pose.pos[1]:+.3f}, {pose.pos[2]:+.3f}]")
                dpg.set_value("st_inliers", f"inliers/cam: {pose.n_inliers} / {pose.camera_port}")

            if stream_enabled and is_fresh:
                left_pos, right_pos   = LEFT_DEFAULT_POS.copy(), RIGHT_DEFAULT_POS.copy()
                left_quat, right_quat = NEUTRAL_QUAT.copy(), NEUTRAL_QUAT.copy()
                tgt_pos  = [pose.pos[0] + offset[0], pose.pos[1] + offset[1], pose.pos[2] + offset[2]]
                tgt_quat = list(pose.quat_wxyz)
                if selected_arm == "left":
                    left_pos, left_quat = tgt_pos, tgt_quat
                else:
                    right_pos, right_quat = tgt_pos, tgt_quat
                arm_pub.publish({"arm_targets": {"ee_pos": [left_pos, right_pos], "ee_quat": [left_quat, right_quat]}})
                last_tgt_pos[selected_arm] = tgt_pos
                last_publish_t = time.monotonic()

            # Error: target vs actual EE (actual forwarded by monitor from robot telemetry)
            ee_entry = ((det_sub.data or {}).get("ee") or {}).get(selected_arm)
            actual_pos = [float(v) for v in ee_entry["pos_actual"]] if ee_entry else None
            if actual_pos is not None and selected_arm in last_tgt_pos:
                err = [last_tgt_pos[selected_arm][i] - actual_pos[i] for i in range(3)]
                err_norm_mm = (err[0]**2 + err[1]**2 + err[2]**2) ** 0.5 * 1000.0
                dpg.set_value("st_actual",   f"actual_pos:  [{actual_pos[0]:+.3f}, {actual_pos[1]:+.3f}, {actual_pos[2]:+.3f}]")
                dpg.set_value("st_err_xyz",  f"err_xyz (m): [{err[0]:+.3f}, {err[1]:+.3f}, {err[2]:+.3f}]")
                dpg.set_value("st_err_norm", f"err_norm:    {err_norm_mm:.1f} mm")
            else:
                dpg.set_value("st_actual",   "actual_pos:  n/a")
                dpg.set_value("st_err_xyz",  "err_xyz (m): n/a")
                dpg.set_value("st_err_norm", "err_norm:    n/a")

            dpg.set_value("st_pub",    f"last_publish: {(time.monotonic() - last_publish_t)*1000.0:.1f} ms ago" if last_publish_t > 0 else "last_publish: n/a")
            dpg.set_value("st_action", f"last_ee_action: {last_action}")

            dpg.render_dearpygui_frame()
            time.sleep(loop_period)
    except KeyboardInterrupt:
        pass
    finally:
        dpg.destroy_context()
        det_sub.stop()
        arm_pub.close()


if __name__ == "__main__":
    main()
