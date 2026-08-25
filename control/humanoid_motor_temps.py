"""Read-only motor temperature probe. Run ONLY while humanoid_real_env is
STOPPED (single CAN bus owner). Constructs the controller, runs the passive
param/self-check exchange to solicit feedback frames, then prints one
name/temperature/voltage row per motor (URDF order). Nothing is enabled and
no position command is ever sent.

Usage:  python humanoid_motor_temps.py
"""
import time

import numpy as np

from humanoid_config import motor_setup, motor_setup_dict
from hardware_bindings.motor.py_motor import CanMotorController


def main():
    motor = CanMotorController(motor_setup)
    motor.getAllParams()
    time.sleep(0.2)          # let feedback frames arrive and parse
    names = list(motor_setup_dict.keys())
    temps = np.array(motor.temperature)
    vbus = np.array(motor.v_bus)
    print(f"{'idx':>3} {'joint':<18} {'CAN':>4} {'temp[C]':>8} {'v_bus':>7}")
    print("-" * 46)
    for i, name in enumerate(names):
        can_id = motor_setup_dict[name][0]
        flag = "  <-- HOT" if temps[i] > 60 else ("  (no feedback yet)"
                                                  if temps[i] == 0 else "")
        print(f"{i:>3} {name:<18} {can_id:>4} {temps[i]:>8.1f} "
              f"{vbus[i]:>7.2f}{flag}")
    motor.shutdown()


if __name__ == "__main__":
    main()
