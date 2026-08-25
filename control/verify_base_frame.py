"""Verify (and auto-solve) the mech_pos → model-joint SIGN convention so the monitor's
GimbalCameraFK yields an ACCURATE, gimbal-invariant base-frame cube position — required
for grasping.

Principle: a cube fixed in the world has a FIXED base-frame position. We move ONE
camera's gimbal through several poses, and at each pose compute
    cube_base = T_base<-camera(joint_pos) · cube_camera_frame(AprilTag)
for each candidate sign of (yaw, pitch). The CORRECT sign makes cube_base constant across
poses (small spread); a wrong sign makes the "fixed" cube swing around. Report the winner.

Then set MODEL_JOINT_SIGN in humanoid_camera_track.py to the winning signs for that camera.

Prereqs: camera streaming on 5555/5556, cam motors powered, CAN up, NO real_env / tracker.
Hold grasp_cube_40mm STILL in front of THIS camera the whole time (it stays fixed; only the
gimbal moves — keep the cube inside the frame across the small ±0.25 rad sweep).
"""
import argparse
import sys
import time

import numpy as np

from humanoid_site import CONTROL_ROOT as _CTRL_ROOT, VISUAL_SERVOING_ROOT as _VS_ROOT  # noqa: E402
sys.path.insert(0, str(_VS_ROOT))
sys.path.append(str(_CTRL_ROOT))
from camera_tag_detector import CameraTagDetector  # noqa: E402
from hardware_bindings.motor.py_motor import CanMotorController  # noqa: E402
from humanoid_config import motor_setup_dict  # noqa: E402
from humanoid_monitor import GimbalCameraFK  # noqa: E402

TARGET = "grasp_cube_40mm"
DT = 5e-3
RAMP_RATE = 0.5

_ap = argparse.ArgumentParser()
_ap.add_argument("--side", choices=["left", "right"], default="left")
_a = _ap.parse_args()
PORT = 5555 if _a.side == "left" else 5556
YAW_MI, PITCH_MI = (0, 1) if _a.side == "left" else (2, 3)       # motor indices
YAW_JI, PITCH_JI = (27, 28) if _a.side == "left" else (29, 30)   # model joint indices
SITE = "cam_left_rgb" if _a.side == "left" else "cam_right_rgb"


def ramp(motor, idx, val):
    step = RAMP_RATE * DT
    while abs(val - float(motor.mech_pos_ref[idx])) > 1e-4:
        cur = float(motor.mech_pos_ref[idx])
        motor.mech_pos_ref[idx] = cur + np.clip(val - cur, -step, step)
        time.sleep(DT)
    motor.mech_pos_ref[idx] = val


def read_cube(det, seconds=0.6):
    xs = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        snap = det.get_latest()
        if snap is not None:
            bp = snap.body_poses.get(TARGET)
            if bp is not None and bp.n_inliers >= 1:
                xs.append(np.asarray(bp.pos, dtype=float))
        time.sleep(0.02)
    return np.median(xs, axis=0) if len(xs) >= 3 else None


head = [motor_setup_dict[k] for k in ("cam_yaw_left", "cam_pitch_left", "cam_yaw_right", "cam_pitch_right")]
motor = CanMotorController(head)
motor.set_max_torque_ratio(0.5)
motor.loc_kp[:] = 5
motor.spd_kp[:] = 2
motor.enable()
motor.start_motion_control_continuously()
motor.mech_pos_ref[:] = np.array(motor.mech_pos, dtype=float)
for i in range(4):
    ramp(motor, i, 0.0)

fk = GimbalCameraFK({PORT: SITE})
det = CameraTagDetector(camera_host="127.0.0.1", camera_port=PORT,
                        camera_transform=np.eye(4), body_specs=[TARGET])
det.start()
print(f"[{_a.side} cam, port {PORT}] Hold {TARGET} STILL in view. Sweeping the gimbal in 3 s...")
time.sleep(3.0)

records = []   # (mech4, cube_camera_frame)
try:
    for y in (-0.25, 0.0, 0.25):
        for p in (0.0, 0.2):
            ramp(motor, YAW_MI, y)
            ramp(motor, PITCH_MI, p)
            time.sleep(0.4)
            cc = read_cube(det)
            if cc is None:
                print(f"  (cube not seen at yaw={y:+.2f} pitch={p:+.2f} — keep it centered) ")
                continue
            records.append((np.array(motor.mech_pos[:4], dtype=float), cc))
            print(f"  pose yaw={y:+.2f} pitch={p:+.2f}: cube_cam={cc.round(3)}")
    for i in range(4):
        ramp(motor, i, 0.0)
finally:
    motor.shutdown()
    det.stop()

if len(records) < 3:
    print("\nNeed >=3 poses with the cube in view — retry keeping it more centered.")
    sys.exit(1)

print(f"\nTesting sign conventions over {len(records)} poses (a FIXED cube ⇒ base pos must be constant):")
best = None
for sy in (+1, -1):
    for sp in (+1, -1):
        bases = []
        for mech, cc in records:
            jp = np.zeros(31)
            jp[YAW_JI] = sy * mech[YAW_MI]
            jp[PITCH_JI] = sp * mech[PITCH_MI]
            T = fk.base_to_camera(PORT, jp)
            bases.append(T[:3, :3] @ cc + T[:3, 3])
        bases = np.array(bases)
        spread = float(np.mean(np.std(bases, axis=0)))
        print(f"  yaw_sign={sy:+d} pitch_sign={sp:+d} → base spread = {spread*1000:6.1f} mm  "
              f"(mean base = {bases.mean(0).round(3)})")
        if best is None or spread < best[0]:
            best = (spread, sy, sp)
print(f"\nBEST for {_a.side} cam: yaw_sign={best[1]:+d}, pitch_sign={best[2]:+d}  "
      f"(spread {best[0]*1000:.1f} mm). If spread is small (<~20 mm) the base frame is accurate;\n"
      f"set MODEL_JOINT_SIGN[{YAW_MI}]={best[1]}, [{PITCH_MI}]={best[2]} in humanoid_camera_track.py.")
