# chrt -f 80 python <control>/humanoid_lock_and_grip.py
"""
Flow:
  1. Connect to the RIGHT gripper (servo ID 19) and OPEN it right away —
     so it's out of the way while you position the arm/object by hand.
  2. Motors start with no torque applied (self_check only) — move the arm
     by hand into the position you want.
  3. Press Enter when you're happy with the position. The script captures
     that exact position and enables the motors with it as the hold
     reference (no jump — same approach as humanoid_hold_pos.py). Arm
     is now locked.
  4. Press Enter again to CLOSE the gripper on whatever's in it.
  5. The arm continues holding. Ctrl+C at any point ramps the arm's
     stiffness down smoothly before disabling, AND deactivates the
     gripper's torque so it goes limp too (it stays wherever it last was
     — it does not auto re-open on shutdown).

Edit GRIPPER_CLOSE_TARGET / GRIPPER_OPEN_TARGET / GRIPPER_PORT / GRIPPER_ID
below if your wiring changes.
"""

import humanoid_site as _site
import os
import sys
import time
from dataclasses import dataclass

import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from humanoid_config import motor_setup  # reuse your existing arm/leg/head setup
from hardware_bindings.motor.py_motor import CanMotorController
from ipc.publisher import DataPublisher

from ft_servo_python_only import FtServo  # adjust import if your file is named differently

import numpy as np

# ── Right gripper config — from your move_servo.py SERVOS list ─────────────
GRIPPER_PORT = "/dev/ttyACM1"     # matches the entry with id=19 in your script
GRIPPER_ID = 19                   # 19 is the RIGHT gripper
GRIPPER_CLOSE_TARGET = 6038        # closed position
GRIPPER_OPEN_TARGET = -634         # open position
GRIPPER_SPEED = 1000
GRIPPER_ACC = 100
GRIPPER_TORQUE = 500
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    torque_ratio: float = 0.3
    """Max torque ratio while holding the arm. Raise if it sags."""
    loc_kp: float = 20.0
    """Position-loop stiffness while holding."""
    spd_kp: float = 5.0
    """Velocity-loop gain while holding."""
    release_ramp_steps: int = 300
    """Steps to ramp stiffness down to zero on shutdown."""


def connect_gripper():
    """Connects and pings the gripper. Returns the open connection, or
    None if the connection/ping failed (subsequent moves are skipped)."""
    print(f"[{GRIPPER_PORT}] Connecting to right gripper (ID {GRIPPER_ID})...")
    servo = FtServo(GRIPPER_PORT)

    ret = servo.ping(GRIPPER_ID)
    if ret is None:
        print(f"[{GRIPPER_PORT}] Ping failed (ID {GRIPPER_ID}). "
              f"Gripper commands will be skipped — check connection.")
        servo.close()
        return None
    _, error, _ = ret
    print(f"[{GRIPPER_PORT}] Ping OK (ID {GRIPPER_ID}, error=0x{error:02X})")

    start_pos = servo.get_position(GRIPPER_ID)
    print(f"[{GRIPPER_PORT}] Current gripper position: {start_pos}")
    servo.enable_torque(GRIPPER_ID, True)
    return servo


def move_gripper(servo, target_position, label):
    if servo is None:
        print(f"[gripper] Skipping '{label}' move — no connection.")
        return

    print(f"[{GRIPPER_PORT}] {label}: moving to {target_position}...")
    servo.set_position(
        GRIPPER_ID,
        target_position,
        speed=GRIPPER_SPEED,
        acc=GRIPPER_ACC,
        torque=GRIPPER_TORQUE,
    )

    timeout = time.time() + 5
    while time.time() < timeout:
        time.sleep(0.1)
        pos = servo.get_position(GRIPPER_ID)
        if pos is not None and abs(pos - target_position) <= 5:
            break

    final_pos = servo.get_position(GRIPPER_ID)
    print(f"[{GRIPPER_PORT}] {label} complete. Final position: {final_pos}")


def deactivate_gripper(servo):
    if servo is None:
        return
    try:
        servo.enable_torque(GRIPPER_ID, False)
        print(f"[{GRIPPER_PORT}] Gripper torque deactivated.")
    except Exception as e:
        print(f"[{GRIPPER_PORT}] Failed to deactivate gripper torque: {e}")
    finally:
        servo.close()


def main():
    args = tyro.cli(Args)

    publisher = DataPublisher(f"udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}", encoding="msgpack", broadcast=False)
    motor = CanMotorController(motor_setup)

    # ── Step 1: open the gripper right away, before any arm positioning ──
    gripper = connect_gripper()
    move_gripper(gripper, GRIPPER_OPEN_TARGET, "Opening gripper")

    motor.self_check()  # comms check only, no motion, no torque applied

    print("\nMove the arm by hand into the position you want.")
    input("Press Enter once it's in place to lock it...\n")

    # Capture wherever it ended up, right before enabling.
    hold_pos = np.array(motor.mech_pos, dtype=float)
    print(f"Captured position to hold:\n{hold_pos}")

    motor.disable(True)
    time.sleep(0.01)

    motor.set_max_torque_ratio(args.torque_ratio)
    motor.loc_kp[:] = args.loc_kp
    motor.spd_kp[:] = args.spd_kp
    motor.mech_pos_ref[:] = hold_pos  # seed ref BEFORE enabling, avoids a jump

    motor.enable()
    motor.start_motion_control_continuously()
    print("Arm locked.")

    # ── Step 2: close the gripper once you confirm via Enter ──
    # The arm's hold logic runs on its own continuous-control thread
    # internally (started above), so it should keep holding through this
    # blocking input() + gripper move.
    input("\nPress Enter to close the gripper...\n")
    move_gripper(gripper, GRIPPER_CLOSE_TARGET, "Closing gripper")

    dt = 1 / 200
    print("Holding position — press Ctrl+C to release and power down.\n")
    try:
        while True:
            motor.mech_pos_ref[:] = hold_pos
            publisher.publish({"motor": {
                "pos":     motor.mech_pos,
                "vel":     motor.mech_vel,
                "pos_ref": motor.mech_pos_ref,
            }})
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nReleasing — ramping arm stiffness down before shutdown...")
        start_kp = float(args.loc_kp)
        start_kd = float(args.spd_kp)
        for i in range(args.release_ramp_steps):
            frac = 1.0 - (i / args.release_ramp_steps)
            motor.loc_kp[:] = start_kp * frac
            motor.spd_kp[:] = start_kd * frac
            motor.mech_pos_ref[:] = hold_pos
            publisher.publish({"motor": {
                "pos":     motor.mech_pos,
                "vel":     motor.mech_vel,
                "pos_ref": motor.mech_pos_ref,
            }})
            time.sleep(dt)
        motor.loc_kp[:] = 0
        motor.spd_kp[:] = 0
        time.sleep(0.2)

        deactivate_gripper(gripper)

    motor.disable(False)
    print("Disabled. Done.")


if __name__ == "__main__":
    main()
