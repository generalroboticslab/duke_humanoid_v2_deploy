# Humanoid Monitor Architecture (Current Setup)

## At-a-Glance Summary

`humanoid_monitor.py` is the current offboard monitoring process. It:
- subscribes to robot telemetry on **9870**,
- runs AprilTag body detection on **two camera streams** (**5555=head**, **5556=chest**),
- renders telemetry + detections in viser,
- publishes detected rigid-body poses (as `MonitorDetectionPacket`) on an **IPC socket** for downstream consumers.

---

## Scope and Status

### Implemented now (Phase 1 — 2026-04-06)
- Dual-camera detection in `humanoid_monitor.py` using `CameraTagDetector`.
- Telemetry display from `humanoid_real_env.py` over 9870.
- Viser visualization for robot state and tracked bodies.
- Detection output published via IPC (`ipc:///tmp/humanoid_detections.sock`) (`MonitorDetectionPacket`) after each detection frame advance.
- `visual_manip_debug_gui.py` — operator GUI that subscribes to the IPC socket and publishes arm IK + EE commands on 9874.

### Not implemented yet (Phase 2 target)
- Multi-camera fusion algorithm to resolve conflicting detections across cameras.
- Refactor of downstream consumers to use fused output once available.

---

## Ground-Truth Process Topology (Current)

```text
[Robot onboard]
  humanoid_real_env.py
    bind pub :9870  (telemetry out)
    connect sub -> :9871 (operator command in)
    connect sub -> :9873 (nav_cmd in)
    connect sub -> :9874 (arm_targets in)

[Workstation / offboard]
  humanoid_monitor.py
    connect sub -> robot:9870
    connect camera streams -> robot:5555,5556
    run CameraTagDetector(5555=head)
    run CameraTagDetector(5556=chest)
    render viser UI
    bind pub ipc:///tmp/humanoid_detections.sock  (MonitorDetectionPacket — detected target poses via IPC)

  visual_manip_debug_gui.py (optional, visual manipulation debug)
    connect sub -> monitor:ipc_socket (detected target poses via IPC)
    bind pub :9874  (arm_targets / ee_action -> robot)

  humanoid_recv_debug.py (optional)
    connect sub -> robot:9870

  gampad_gui_control.py (optional sender)
    bind pub :9873 (nav_cmd)
    bind pub :9874 (arm_targets)
```

Notes:
- Publisher-binds / subscriber-connects is consistent across this stack.
- `visual_manip_debug_gui.py` and `gampad_gui_control.py` both bind 9874; only one should run at a time.

---

## Current Port Table (Code Truth)

| Port | Binder | Connector(s) | Payload |
|---|---|---|---|
| 5555 | Camera stream server (robot) | `humanoid_monitor` (+ other camera clients) | Camera frames (head cam) |
| 5556 | Camera stream server (robot) | `humanoid_monitor` (+ other camera clients) | Camera frames (chest cam) |
| 9870 | `humanoid_real_env.py` (robot) | `humanoid_monitor`, `humanoid_recv_debug`, other telemetry subscribers | Telemetry packet (`joint_*`, IMU, commands, vicon, `ee_poses`, `stamp`) |
| 9871 | Operator publisher (`gamepad.py` / `ssh_keyboard.py`) | `humanoid_real_env.py` | Operator commands |
| 9873 | High-level publisher (`gampad_gui_control.py`, optional others) | `humanoid_real_env.py` | `nav_cmd` |
| 9874 | Arm-target publisher (`gampad_gui_control.py`, `visual_manip_debug_gui.py`, arm replay) | `humanoid_real_env.py` | `arm_targets` / `ee_action` |
| 9875 | — | — | Not active (EE pose data included in 9870 telemetry as `ee_poses`) |
| IPC socket | `humanoid_monitor.py` | `visual_manip_debug_gui.py`, future subscribers | `MonitorDetectionPacket` (detected rigid-body poses in body frame) |

---

## Current `humanoid_monitor.py` Behavior

### Threads
- `_monitor_telem`: consumes latest telemetry from 9870 and updates in-memory stats.
- `_viser_thread`: updates robot mesh + detection overlays in viser; publishes `MonitorDetectionPacket` via IPC after each detection frame advance.
- `_cam_capture` and `_cam_bg_loop`: optional stitched camera background rendering to viser.

### Detection pipeline
- Default camera ports: `tag_camera_ports = [5555, 5556]`.
- Default transforms: `tag_transforms = {5555: "head", 5556: "chest"}`.
- Each camera has its own frame gate (`last_det_frame_id[port]`), so update cadence is independent.
- Latest-value semantics are used for display state.

### Output behavior
- Visual output: viser scene + optional terminal stats.
- Network output: `MonitorDetectionPacket` via IPC (`ipc:///tmp/humanoid_detections.sock`) (published per detection frame flush).
  - Payload schema: `{stamp, frame_id_by_port: {port: frame_id}, targets: {key: MonitorTargetPose}}`
  - `MonitorTargetPose` fields: `key, pos, quat_wxyz, n_inliers, camera_port, capture_stamp`
  - `capture_stamp` is `time.monotonic()` from the monitor process — not cross-machine comparable.

---

## Current `camera_tag_detector.py` Integration

`visual_servoing/camera_tag_detector.py` provides reusable per-camera detection:
- One detector instance per camera port.
- Produces latest snapshot with per-body poses + per-target-frame poses.
- Supports transform presets: `head`, `chest`.

In current setup, `humanoid_monitor.py` imports this module and instantiates one detector per configured camera port.

---

## Why This Differs from Older Plan Drafts

Older draft sections described a Phase-2 target state (multi-camera fusion) as if it were already wired.
Those remain design goals but are not yet implemented.

What IS now implemented (Phase 1 addition, 2026-04-06):
- Monitor publishes raw per-camera detections (not fused) on the IPC socket.
- `visual_manip_debug_gui.py` consumes the IPC socket for target selection + arm command publishing.

---

## Future Architecture (Phase 2, Not Implemented)

### Planned addition
- Add fusion logic to resolve conflicting detections from multiple cameras into a single authoritative pose per body.
- Continue publishing via IPC with the same `MonitorDetectionPacket` schema (transparent to consumers).

### Entry criteria
- Observed pose inconsistency between cameras for the same body, or
- Need for a single authoritative pose when cameras have overlapping coverage.

---

## Verification Checklist for This Document

This document is considered up-to-date if all checks remain true in code:

1. `humanoid_real_env.py` binds 9870 and subscribes to 9871/9873/9874.
2. `humanoid_monitor.py` subscribes to 9870.
3. `humanoid_monitor.py` defaults to ports 5555+5556 and transforms head/chest.
4. `humanoid_monitor.py` publishes via IPC socket (`ipc:///tmp/humanoid_detections.sock`) and publishes `MonitorDetectionPacket` in `_viser_thread` after each flush.
5. `gampad_gui_control.py` publishes `nav_cmd` on 9873 and `arm_targets` on 9874.
6. `ee_poses` appears in 9870 telemetry payload.
7. `visual_manip_debug_gui.py` subscribes to the IPC socket and binds 9874.

---

## Source Pointers (for quick code re-check)

- `humanoid_real_env.py` — 9870/9871/9873/9874 wiring + telemetry payload.
- `humanoid_monitor.py` — telemetry subscriber, dual-camera detector setup, transforms, IPC publisher.
- `visual_manip_debug_gui.py` — IPC subscriber, 9874 publisher, target selection + EE action GUI.
- `gampad_gui_control.py` — 9873 and 9874 publishers.
- `humanoid_recv_debug.py` — telemetry debug subscriber (9870) and camera-display defaults.
