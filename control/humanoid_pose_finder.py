# chrt -f 80 python <control>/humanoid_pose_finder.py
"""
Interactive pose finder.

Lets you nudge individual joints by small amounts via typed commands (not
held-down keys — avoids runaway motion if a key gets stuck), watch the
arm move, and save the resulting pose under a name once it looks right.
At the end it prints Python dicts ready to paste into
humanoid_grasp_sequence.py as RAISED_ARM_POSE / GRASP_POSE.

Startup is the same "lock wherever it currently is" pattern as
humanoid_hold_pos.py — no jump when motors enable.

Commands (type at the prompt, case-insensitive on joint names):
  list                          show current target position of every
                                 non-zero joint
  <joint_name> <delta>          nudge one joint by `delta` radians
                                 relative to its current target
                                 e.g.  right_shoulder_2 -0.1
  goto <joint_name> <value>     set one joint's target to an absolute
                                 value in radians
                                 e.g.  goto right_elbow -0.8
  step <size>                   change the default jog step size
                                 (used by some shortcuts, informational)
  save <name>                   save the current full pose under `name`
  poses                         list saved pose names
  home                          smoothly move back to all-zero pose
                                 (careful — this can be a big motion)
  joints                        list all valid joint names
  help                          show this command list again
  done / quit / exit            ramp stiffness down, disable, print all
                                 saved poses as Python dicts, and exit

Each nudge/goto is itself smoothly ramped (not an instant jump) and the
arm continues actively holding its last commanded position between
commands, so it won't droop while you're typing.
"""

import humanoid_site as _site
import os
import sys
import time
from dataclasses import dataclass

import tyro
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from humanoid_config import motor_setup, motor_setup_dict
from hardware_bindings.motor.py_motor import CanMotorController
from ipc.publisher import DataPublisher

joint_names = list(motor_setup_dict.keys())
N_JOINTS = len(joint_names)
name_to_idx = {name.lower(): i for i, name in enumerate(joint_names)}


@dataclass
class Args:
    torque_ratio: float = 0.2
    """Max torque ratio while jogging. Keep low for safety while exploring."""
    loc_kp: float = 8.0
    """Position-loop stiffness while jogging — softer than full hold so it's
    gentler/safer to be near while testing, but still self-supporting."""
    spd_kp: float = 2.0
    """Velocity-loop gain while jogging."""
    move_steps: int = 60
    """Steps used to smoothly ramp each nudge/goto command (not instant)."""
    dt: float = 1 / 200
    release_ramp_steps: int = 300


def smooth_move(motor, publisher, target_pos, num_steps, dt):
    start_pos = np.array(motor.mech_pos_ref, dtype=float)
    for i in range(num_steps):
        t = i / num_steps
        s = 3 * t**2 - 2 * t**3
        motor.mech_pos_ref[:] = start_pos + s * (target_pos - start_pos)
        publisher.publish({"motor": {
            "pos": motor.mech_pos, "vel": motor.mech_vel, "pos_ref": motor.mech_pos_ref,
        }})
        time.sleep(dt)


def print_pose_dict(name, pose):
    print(f"\n{name.upper()} = pose_from_overrides({{")
    for i, val in enumerate(pose):
        if abs(val) > 1e-6:
            print(f'    "{joint_names[i]}": {val:.4f},')
    print("})")


def main():
    args = tyro.cli(Args)

    publisher = DataPublisher(f"udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}", encoding="msgpack", broadcast=False)
    motor = CanMotorController(motor_setup)
    motor.self_check()  # comms check only, no motion, no torque applied

    print(__doc__)
    print("\nMove the arm by hand into a comfortable starting pose.")
    input("Press Enter once ready to lock current position and start jogging...\n")

    current_pos = np.array(motor.mech_pos, dtype=float)
    print(f"Captured starting position:\n{current_pos}")

    motor.disable(True)
    time.sleep(0.01)
    motor.set_max_torque_ratio(args.torque_ratio)
    motor.loc_kp[:] = args.loc_kp
    motor.spd_kp[:] = args.spd_kp
    motor.mech_pos_ref[:] = current_pos  # seed ref BEFORE enabling, avoids a jump
    motor.enable()
    motor.start_motion_control_continuously()
    print("Locked. Ready for jog commands. Type 'help' any time.\n")

    saved_poses = {}

    try:
        while True:
            try:
                line = input("jog> ").strip()
            except EOFError:
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("done", "quit", "exit"):
                break

            elif cmd == "help":
                print(__doc__)

            elif cmd == "joints":
                for n in joint_names:
                    print(f"  {n}")

            elif cmd == "list":
                pos = np.array(motor.mech_pos_ref, dtype=float)
                for i, v in enumerate(pos):
                    if abs(v) > 1e-6:
                        print(f"  {joint_names[i]:20s} {v:+.4f} rad")

            elif cmd == "poses":
                if not saved_poses:
                    print("  (none saved yet)")
                for name in saved_poses:
                    print(f"  {name}")

            elif cmd == "save":
                if len(parts) < 2:
                    print("Usage: save <name>")
                    continue
                name = parts[1]
                saved_poses[name] = np.array(motor.mech_pos_ref, dtype=float)
                print(f"Saved current pose as '{name}'.")

            elif cmd == "home":
                confirm = input("This will smoothly move ALL joints to zero. Confirm? [y/N] ")
                if confirm.lower().startswith("y"):
                    smooth_move(motor, publisher, np.zeros(N_JOINTS), args.move_steps * 3, args.dt)
                    print("At home (zero) position.")
                else:
                    print("Cancelled.")

            elif cmd == "goto":
                if len(parts) != 3:
                    print("Usage: goto <joint_name> <value>")
                    continue
                jname, val = parts[1].lower(), parts[2]
                if jname not in name_to_idx:
                    print(f"Unknown joint '{parts[1]}'. Type 'joints' to list valid names.")
                    continue
                try:
                    val = float(val)
                except ValueError:
                    print("Value must be a number (radians).")
                    continue
                target = np.array(motor.mech_pos_ref, dtype=float)
                target[name_to_idx[jname]] = val
                smooth_move(motor, publisher, target, args.move_steps, args.dt)
                print(f"{parts[1]} -> {val:+.4f} rad")

            elif cmd in name_to_idx:
                if len(parts) != 2:
                    print(f"Usage: {parts[0]} <delta>")
                    continue
                try:
                    delta = float(parts[1])
                except ValueError:
                    print("Delta must be a number (radians).")
                    continue
                target = np.array(motor.mech_pos_ref, dtype=float)
                target[name_to_idx[cmd]] += delta
                smooth_move(motor, publisher, target, args.move_steps, args.dt)
                print(f"{parts[0]} -> {target[name_to_idx[cmd]]:+.4f} rad")

            else:
                print(f"Unknown command or joint name: '{cmd}'. Type 'help' or 'joints'.")

    except KeyboardInterrupt:
        print("\n(Ctrl+C) releasing...")

    print("\nRamping stiffness down before shutdown...")
    hold_pos = np.array(motor.mech_pos_ref, dtype=float)
    start_kp, start_kd = float(args.loc_kp), float(args.spd_kp)
    for i in range(args.release_ramp_steps):
        frac = 1.0 - (i / args.release_ramp_steps)
        motor.loc_kp[:] = start_kp * frac
        motor.spd_kp[:] = start_kd * frac
        motor.mech_pos_ref[:] = hold_pos
        publisher.publish({"motor": {
            "pos": motor.mech_pos, "vel": motor.mech_vel, "pos_ref": motor.mech_pos_ref,
        }})
        time.sleep(args.dt)
    motor.loc_kp[:] = 0
    motor.spd_kp[:] = 0
    time.sleep(0.2)
    motor.disable(False)
    print("Disabled.")

    if saved_poses:
        print("\n" + "=" * 60)
        print("Saved poses — paste these into humanoid_grasp_sequence.py:")
        print("=" * 60)
        for name, pose in saved_poses.items():
            print_pose_dict(name, pose)
    else:
        print("\nNo poses were saved (use 'save <name>' before 'done' next time).")


if __name__ == "__main__":
    main()
