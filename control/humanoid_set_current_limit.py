#!/usr/bin/env python3
"""
Set motor current limits based on motor type.

Usage:
    python set_current_limit.py              # Use default limits
    python set_current_limit.py --scale 0.5  # Scale all limits by 0.5
"""

import argparse
import numpy as np

# 08-22: the old `sys.path.insert(0, <control>/../..)` relic (pre-dating the
# hardware_bindings/ move) is gone — `hardware_bindings` and `humanoid_config`
# live in control/ and resolve from the script directory.
from hardware_bindings.motor.py_motor import (
    CanMotorController, MOTO_PARAM,
    MOTOR_TYPE_NAMES,
    MOTOR_CURRENT_LIMIT_DEFAULTS,
)
from humanoid_config import motor_setup


MAX_CURRENT_LIMIT = 40.0  # Maximum allowed current limit in Amps
# Default current limits per motor type (in Amps)
# Adjust these values based on your motor specifications

# 04 --> scale=0.55



def main():
    parser = argparse.ArgumentParser(description="Set motor current limits")
    parser.add_argument("--scale", type=float, default=0.6,
                        help="Scale factor for default limits")
    args = parser.parse_args()

    cur_limits = np.array([min(MOTOR_CURRENT_LIMIT_DEFAULTS[mt] * args.scale, MAX_CURRENT_LIMIT)
                           for _, _, mt in motor_setup], dtype=np.float32)

    # Print planned limits
    print(f"\n{'Index':>6} {'Type':>6} {'Cur_lim':>10}")
    print("-" * 26)
    for i, (can_id, can_if, mt) in enumerate(motor_setup):
        print(f"{i:>6} {MOTOR_TYPE_NAMES.get(mt, f'?{mt}'):>6} {cur_limits[i]:>10.2f}")

    if input("Press yes to continue: ") != "yes":
        return

    motor = CanMotorController(motor_setup)
    motor.setParam(MOTO_PARAM.cur_limit.value, cur_limits)
    motor.save_data()
    motor.self_check()
    motor.disable(False)


if __name__ == "__main__":
    main()
