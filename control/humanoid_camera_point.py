"""Bench script: runs its routine only when invoked (python humanoid_camera_point.py);
importing it is inert; it has no CLI flags. main() constructs and ENABLES the
four gimbal motors (50 % torque ceiling, kp 5 / kd 2), opens every attached
RealSense, drives the gimbal to zero, points camera 2 (right) at a hard-coded
target (OBJ_X/Y/Z) and holds there, streaming JPEG frames over UDP 9871 to
WORKSTATION_IP for camera_viewer.py (frame_streamer.FrameStreamer). Early
gimbal pointing bench. Note the second show_frames/stop_realsense definitions
below (UDP streaming) override the cv2-window versions defined above them.
"""
import humanoid_site as _site

import numpy as np
import time
import cv2
import pyrealsense2 as rs
from tqdm import trange
from ipc.publisher import DataPublisher
from hardware_bindings.motor.py_motor import CanMotorController
from humanoid_config import motor_setup_dict


# ── RealSense setup ────────────────────────────────────────────────────────────
def start_realsense():
    """Start all connected RealSense cameras, return list of (pipeline, name)."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense cameras found!")

    pipelines = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name   = dev.get_info(rs.camera_info.name)
        print(f"Found camera {i}: {name} (serial {serial})")

        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append((pipeline, f"Camera {i}: {name}"))

    return pipelines


def show_frames(pipelines):
    """Grab and display one frame from each camera. Returns False if 'q' pressed."""
    for pipeline, name in pipelines:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            color  = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            cv2.imshow(name, img)
        except RuntimeError:
            pass  # skip this frame if it didn't arrive in time

    return cv2.waitKey(1) & 0xFF != ord('q')


def stop_realsense(pipelines):
    for pipeline, _ in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()


# ── Motor setup (head only) ────────────────────────────────────────────────────
head_setup = [
    motor_setup_dict["cam_yaw_left"],    # index 0
    motor_setup_dict["cam_pitch_left"],  # index 1
    motor_setup_dict["cam_yaw_right"],    # index 2
    motor_setup_dict["cam_pitch_right"],  # index 3
]


# ── Return to zero ─────────────────────────────────────────────────────────────
def go_to_zero(pipelines=None, num_steps: int = 200, dt: float = 5e-3):
    current_pos = np.array(motor.mech_pos, dtype=float)
    zero_pos = np.zeros(4)
    for i in trange(num_steps, desc="Returning to zero"):
        t = i / num_steps
        s = 3 * t**2 - 2 * t**3
        motor.mech_pos_ref[:] = current_pos + s * (zero_pos - current_pos)
        publisher.publish({"motor": {
            "pos_ref":     np.array(motor.mech_pos_ref),
            "pos":         np.array(motor.mech_pos),
            "vel":         np.array(motor.mech_vel),
            "torque":      np.array(motor.mech_torque),
            "temperature": np.array(motor.temperature),
        }})
        if pipelines:
            show_frames(pipelines)
        time.sleep(dt)
    print("At zero position.")


# ── Angle math ─────────────────────────────────────────────────────────────────
def compute_head_angles(
    obj_x: float,
    obj_y: float,
    obj_z: float,
    camera_offset_x: float = 0.0,
    camera_offset_y: float = 0.0,
    camera_offset_z: float = 1.0,
) -> tuple[float, float]:
    dx = obj_x - camera_offset_x
    dy = obj_y - camera_offset_y
    dz = obj_z - camera_offset_z
    yaw   = np.arctan2(dy, dx)
    pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    return yaw, pitch


# ── Replace the existing show_frames / stop_realsense in humanoid_camera_point.py
#    with the versions below, and add the FrameStreamer import + instantiation.
#
# 1) Add near your other imports:
from frame_streamer import FrameStreamer

# 2) Instantiate once, near where you create `publisher` (now inside main()).
#    dest_ip = the IP of YOUR LAPTOP (not the robot!) — same machine you'll
#    run camera_viewer.py on. Adjust the port if you like; must match
#    --port on the viewer.


# 3) Replace show_frames with this version — sends frames over UDP instead
#    of opening a local GUI window. Returns True always (no local 'q' key
#    to check anymore; quit from the laptop-side viewer window instead, or
#    Ctrl+C the robot script as before).
def show_frames(pipelines):
    for pipeline, name in pipelines:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            frame_streamer.send(name, img)
        except RuntimeError:
            pass  # skip this frame if it didn't arrive in time
    return True


# 4) Replace stop_realsense with this version (no more cv2.destroyAllWindows
#    needed since there are no local windows):
def stop_realsense(pipelines):
    for pipeline, _ in pipelines:
        pipeline.stop()
    frame_streamer.close()


# ── Smooth motion to target ────────────────────────────────────────────────────
def point_camera_at(
    obj_x: float,
    obj_y: float,
    obj_z: float,
    pipelines=None,
    camera: int = 1,
    camera_offset_x: float = 0.0,
    camera_offset_y: float = 0.0,
    camera_offset_z: float = 1.0,
    num_steps: int = 200,
    dt: float = 5e-3,
):
    assert camera in (1, 2), "camera must be 1 or 2"

    yaw, pitch = compute_head_angles(
        obj_x, obj_y, obj_z,
        camera_offset_x, camera_offset_y, camera_offset_z,
    )
    print(f"Pointing camera {camera} → yaw={np.degrees(yaw):.1f}°  pitch={np.degrees(pitch):.1f}°")

    yaw_idx   = 0 if camera == 1 else 2
    pitch_idx = 1 if camera == 1 else 3

    GEAR_RATIO = 5.5

    current_pos = np.array(motor.mech_pos, dtype=float)
    target_pos  = np.zeros(4)
    target_pos[yaw_idx]   = -yaw * GEAR_RATIO
    target_pos[pitch_idx] = -pitch * GEAR_RATIO

    print(f"current_pos: {current_pos}")
    print(f"target_pos:  {target_pos}")

    try:
        for i in trange(num_steps, desc=f"Camera {camera} → target"):
            t = i / num_steps
            s = 3 * t**2 - 2 * t**3
            motor.mech_pos_ref[:] = current_pos + s * (target_pos - current_pos)
            publisher.publish({"motor": {
                "pos_ref":     np.array(motor.mech_pos_ref),
                "pos":         np.array(motor.mech_pos),
                "vel":         np.array(motor.mech_vel),
                "torque":      np.array(motor.mech_torque),
                "temperature": np.array(motor.temperature),
            }})
            if pipelines:
                if not show_frames(pipelines):
                    print("User quit.")
                    return
            time.sleep(dt)
    except KeyboardInterrupt:
        print("Interrupted during motion.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 08-22: moved under main() so that importing the module is inert; `motor`,
    # `publisher` and `frame_streamer` stay module globals because go_to_zero(),
    # point_camera_at(), show_frames() and stop_realsense() read them.
    global motor, publisher, frame_streamer
    motor = CanMotorController(head_setup)
    motor.self_check()

    publisher = DataPublisher(f"udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}", encoding="msgpack", broadcast=False)

    motor.set_max_torque_ratio(0.5)
    motor.loc_kp[:] = 5
    motor.spd_kp[:] = 2

    motor.enable()
    motor.start_motion_control_continuously()

    start_pos = np.array(motor.mech_pos)
    print(f"Starting positions: {start_pos}")

    frame_streamer = FrameStreamer(dest_ip=_site.WORKSTATION_IP, dest_port=9871, jpeg_quality=60)

    pipelines = start_realsense()

    try:
        OBJ_X = 5.0   # metres forward
        OBJ_Y = 0.25   # metres left (positive = left)
        OBJ_Z = 2   # metres up

        go_to_zero(pipelines)
        point_camera_at(OBJ_X, OBJ_Y, OBJ_Z, pipelines=pipelines, camera=2, camera_offset_z=1.0)
        # camera = 1 --> LEFT CAMERA
        # camera = 2 --> RIGHT CAMERA

        # Hold and keep showing camera feed until 'q'
        print("Holding position — press 'q' to quit.")
        while show_frames(pipelines):
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        motor.shutdown()
        stop_realsense(pipelines)
        raise e

    go_to_zero(pipelines)
    time.sleep(0.2)
    motor.shutdown()
    stop_realsense(pipelines)


if __name__ == "__main__":
    main()
