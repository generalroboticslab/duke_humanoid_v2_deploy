"""Measure the REAL motor→camera gear ratio for the forward (left) cam yaw & pitch,
using the motor ENCODER (mech_pos, exact) together with the AprilTag bearing (exact).

Principle: hold the cube FIXED in view. Move one cam-motor axis by a known amount
(read from the encoder). The camera optical axis rotates by Δmech/gear, so a FIXED
cube's bearing in the image shifts by -Δmech/gear. Therefore:
        gear = -Δmech_pos / Δbearing        (motor-rad per camera-rad)

Safe: only the forward cam moves, small amounts (yaw 0.4, pitch 0.3 rad MOTOR),
slow rate-limited ramps, returns to zero, shuts down on exit. Rear cam untouched.

Prereqs: camera streaming on 5555, robot motors powered, CAN up, NO real_env / tracker.
Hold grasp_cube_40mm STILL in front of the forward (left) camera the whole time.
"""
import argparse
import math
import sys
import time

import numpy as np

from humanoid_site import CONTROL_ROOT as _CTRL_ROOT, VISUAL_SERVOING_ROOT as _VS_ROOT  # noqa: E402
sys.path.insert(0, str(_VS_ROOT))        # camera_tag_detector + its common
sys.path.append(str(_CTRL_ROOT))         # hardware_bindings, humanoid_config
from camera_tag_detector import CameraTagDetector  # noqa: E402
from hardware_bindings.motor.py_motor import CanMotorController  # noqa: E402
from humanoid_config import motor_setup_dict  # noqa: E402

TARGET = "grasp_cube_40mm"
RAMP_RATE = 0.5          # rad/s motor-side (slow)
DT = 5e-3

_ap = argparse.ArgumentParser(description="Calibrate cam gear+sign (encoder vs AprilTag)")
_ap.add_argument("--side", choices=["left", "right"], default="left")
_args = _ap.parse_args()
# left = forward cam (port 5555, motors yaw=0/pitch=1); right = rear cam (5556, yaw=2/pitch=3)
PORT = 5555 if _args.side == "left" else 5556
YAW_IDX, PITCH_IDX = (0, 1) if _args.side == "left" else (2, 3)


def read_bearing(detector, seconds=1.0):
    """Average (yaw,pitch) camera-frame bearing of the fixed cube over a window."""
    ys, ps, n = [], [], 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        snap = detector.get_latest()
        if snap is not None:
            bp = snap.body_poses.get(TARGET)
            if bp is not None and bp.n_inliers >= 1:
                x, y, z = np.asarray(bp.pos, dtype=float)
                zc = max(z, 1e-3)
                ys.append(math.atan2(x, zc)); ps.append(math.atan2(y, zc)); n += 1
        time.sleep(0.02)
    if n < 3:
        return None
    return np.array([float(np.median(ys)), float(np.median(ps))])


def ramp_axis(motor, idx, target_val):
    """Slowly ramp mech_pos_ref[idx] to target_val (others held at their current ref)."""
    max_step = RAMP_RATE * DT
    while True:
        cur = float(motor.mech_pos_ref[idx])
        if abs(target_val - cur) < 1e-4:
            break
        motor.mech_pos_ref[idx] = cur + np.clip(target_val - cur, -max_step, max_step)
        time.sleep(DT)
    motor.mech_pos_ref[idx] = target_val


head_setup = [motor_setup_dict[k] for k in
              ("cam_yaw_left", "cam_pitch_left", "cam_yaw_right", "cam_pitch_right")]
motor = CanMotorController(head_setup)
motor.set_max_torque_ratio(0.5)
motor.loc_kp[:] = 5
motor.spd_kp[:] = 2
motor.enable()
motor.start_motion_control_continuously()
motor.mech_pos_ref[:] = np.array(motor.mech_pos, dtype=float)
for i in range(4):
    ramp_axis(motor, i, 0.0)

det = CameraTagDetector(camera_host="127.0.0.1", camera_port=PORT,
                        camera_transform=np.eye(4), body_specs=[TARGET])
det.start()
print(f"[{_args.side} cam, port {PORT}] Hold grasp_cube_40mm STILL in front of THIS camera. "
      f"Measuring in 3 s...")
time.sleep(3.0)

try:
    for axis_name, idx, delta, bearing_i in (("YAW", YAW_IDX, 0.4, 0), ("PITCH", PITCH_IDX, 0.3, 1)):
        b0 = read_bearing(det)
        if b0 is None:
            print(f"[{axis_name}] cube not seen — hold it in view and rerun."); continue
        m0 = float(motor.mech_pos[idx])
        ramp_axis(motor, idx, delta)
        time.sleep(0.5)
        b1 = read_bearing(det)
        m1 = float(motor.mech_pos[idx])
        ramp_axis(motor, idx, 0.0)
        if b1 is None:
            print(f"[{axis_name}] cube left view during move — retry with a wider/closer cube."); continue
        d_mech = m1 - m0
        d_bear = b1[bearing_i] - b0[bearing_i]
        gear = (-d_mech / d_bear) if abs(d_bear) > 1e-4 else float("nan")
        print(f"[{axis_name}] Δmech(encoder)={d_mech:+.3f} rad  Δbearing(cam)={d_bear:+.3f} rad "
              f"({math.degrees(d_bear):+.1f}°)  →  gear ≈ {gear:.2f}")
    print("\nDone. gear = motor-rad per camera-rad (humanoid_camera_point uses 5.5).")
finally:
    for i in range(4):
        ramp_axis(motor, i, 0.0)
    time.sleep(0.1)
    motor.shutdown()
    det.stop()
