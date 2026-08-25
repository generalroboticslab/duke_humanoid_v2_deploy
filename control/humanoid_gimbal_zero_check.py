"""Is the camera-gimbal encoder zero still calibrated? — read-only, a few seconds (--seconds, default 3).

WHY THIS EXISTS. Every AprilTag detection reaches the base frame through the
gimbal's forward kinematics: the monitor takes the tag pose in the CAMERA frame
and applies `GimbalCameraFK.base_to_camera(port, joint_pos)`. That FK assumes
one calibrated fact — **encoder 0 means the camera looks straight ahead**. If
the gimbal is re-zeroed while pointing somewhere else (e.g. a whole-robot
`set_zero` run done to fix the ARM), the assumption breaks and every detection
lands in the wrong place in the base frame. Worse, the error DEPENDS on where
the gimbal currently points, so the same stationary cube reports different
positions on successive runs — which is exactly the signature this tool was
written for (2026-07-29: a cube reported at r_xy 1.03 m, then behind the robot
at z +0.56 m, after an arm re-zero).

The check is a comparison, not a measurement: it prints where the model
believes each camera is looking, given the live encoders. You then look at the
physical robot. Agreement means the gimbal zero survived; disagreement gives
you the offset directly.

Run with real_env up (any profile that publishes telemetry — the arm need not
be powered). Publishes nothing, commands nothing.

    python humanoid_gimbal_zero_check.py
    python humanoid_gimbal_zero_check.py --robot-ip 127.0.0.1 --seconds 5
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import tyro

from ipc.publisher import NNGSubscriber
from humanoid_monitor import GimbalCameraFK

CAM_SITES = {5555: "cam_left_rgb", 5556: "cam_right_rgb"}
CAM_NAMES = {5555: "LEFT  (forward-mounted)", 5556: "RIGHT (rear-mounted)"}
JOINT_NAMES = ("cam_yaw_left", "cam_pitch_left", "cam_yaw_right", "cam_pitch_right")


@dataclass
class Args:
    robot_ip: str = "127.0.0.1"
    seconds: float = 3.0
    """Sample this long; drift/noise is reported so a wandering gimbal is obvious."""


def optic_axis(fk: GimbalCameraFK, port: int, jpos) -> tuple[float, float, np.ndarray]:
    """(azimuth_deg, elevation_deg, camera_position) of the optical axis in the
    base frame. +azimuth = toward the robot's left, +elevation = upward."""
    T = fk.base_to_camera(port, jpos)
    z = T[:3, 2]                                   # OpenCV: +z is the optical axis
    az = math.degrees(math.atan2(z[1], z[0]))
    el = math.degrees(math.asin(float(np.clip(z[2], -1.0, 1.0))))
    return az, el, T[:3, 3].copy()


def main() -> int:
    args = tyro.cli(Args)
    sub = NNGSubscriber(f"tcp://{args.robot_ip}:9870")
    sub.start()
    print(f"[check] subscribing tcp://{args.robot_ip}:9870 for {args.seconds:.0f}s …")
    deadline = time.monotonic() + 8.0
    while sub.data is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if sub.data is None:
        print("[check] NO TELEMETRY — is humanoid_real_env.py running?")
        sub.stop()
        return 2

    samples = []
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        jp = np.asarray(sub.data.get("joint_pos") or (), dtype=float)
        if jp.size >= 31 and np.all(np.isfinite(jp[27:31])):
            samples.append(jp.copy())
        time.sleep(0.05)
    sub.stop()
    if not samples:
        print("[check] telemetry had no usable gimbal joints")
        return 2

    J = np.stack(samples)
    g = J[:, 27:31]
    jpos = J[-1]
    print(f"[check] {len(samples)} samples\n")
    print("GIMBAL ENCODERS (live)")
    for k, n in enumerate(JOINT_NAMES):
        col = g[:, k]
        print(f"  {n:<17} {col[-1]:+8.4f} rad = {math.degrees(col[-1]):+7.2f} deg"
              f"   (spread {math.degrees(col.max() - col.min()):.2f} deg)")

    fk = GimbalCameraFK(CAM_SITES)
    print("\nWHERE THE MODEL BELIEVES EACH CAMERA LOOKS, given those encoders")
    print("  (azimuth: + = robot's LEFT · elevation: + = UP · 0/0 = straight ahead, level)")
    for port in (5555, 5556):
        az, el, pos = optic_axis(fk, port, jpos)
        print(f"\n  port {port} — {CAM_NAMES[port]}")
        print(f"    optical axis : azimuth {az:+7.2f} deg   elevation {el:+7.2f} deg")
        print(f"    camera at    : {np.round(pos, 4)} m (base frame)")

    print("\nSAME QUESTION AT ENCODER ZERO (the calibrated assumption)")
    zero = np.zeros_like(jpos)
    for port in (5555, 5556):
        az, el, _ = optic_axis(fk, port, zero)
        print(f"  port {port}: azimuth {az:+7.2f} deg   elevation {el:+7.2f} deg")
    print("  ^ the LEFT camera should read ~0/0 and the RIGHT ~180/0 (it faces aft).")

    print("\nNOW LOOK AT THE ROBOT.")
    print("  Do the cameras physically point where the 'live' block above says?")
    print("  * yes  -> the gimbal zero survived; the bad detections are NOT from this")
    print("  * no   -> the gimbal zero was destroyed. The difference between what you")
    print("           see and what is printed IS the offset. Recover it with:")
    print("             1) comment out every non-cam entry in humanoid_config.py's")
    print("                motor_setup_dict  (leave only the 4 cam_* joints)")
    print("             2) hand-point both cameras straight forward and level")
    print("             3) python humanoid_set_zero.py      # SETS zero, does not drive")
    print("             4) RESTORE the full motor_setup_dict  <- skipping this breaks")
    print("                the whole stack (joint count != model)")
    print("           then re-run this check: encoder zero must mean cameras forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
