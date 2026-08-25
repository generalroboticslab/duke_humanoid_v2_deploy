"""Bench script: runs its routine only when invoked (python humanoid_camera_test.py);
importing it is inert; it has no CLI flags. main() constructs and ENABLES the
four gimbal motors (cam_yaw/cam_pitch, left and right) (10 % torque ceiling,
kp 5 / kd 2) and sweeps them through three 4 s sine cycles of +-1.5 rad from
their current positions (right pair mirrored), publishing telemetry, then
returns to the start and shuts down. Gimbal wiring/sign check.
"""
import humanoid_site as _site
import numpy as np
import time
from tqdm import trange
from ipc.publisher import DataPublisher
from hardware_bindings.motor.py_motor import CanMotorController
from humanoid_config import motor_setup_dict

head_setup = [
    motor_setup_dict["cam_yaw_left"],
    motor_setup_dict["cam_pitch_left"],
    motor_setup_dict["cam_yaw_right"],
    motor_setup_dict["cam_pitch_right"],
]

AMPLITUDE = 1.5
PERIOD_S  = 4.0
n_cycles  = 3
dt        = 5e-3
n         = int(n_cycles * PERIOD_S / dt)
t         = np.linspace(0, n_cycles * 2 * np.pi, n)


def main():
    # 08-22: moved under main() so that importing the module is inert.
    motor = CanMotorController(head_setup)
    motor.self_check()

    publisher = DataPublisher(f"udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}", encoding="msgpack", broadcast=False)

    motor.set_max_torque_ratio(0.1)
    motor.loc_kp[:] = 5
    motor.spd_kp[:] = 2

    motor.enable()
    motor.start_motion_control_continuously()

    start_pos = np.array(motor.mech_pos)
    print(f"Starting positions: {start_pos}")

    # Sine starts at 0 (sin(0)=0) so motors begin at start_pos with no jump
    trajs = start_pos + np.column_stack([
         AMPLITUDE * np.sin(t),
         AMPLITUDE * np.sin(t),
        -AMPLITUDE * np.sin(t),
        -AMPLITUDE * np.sin(t),
    ])

    try:
        for i in trange(n):
            motor.mech_pos_ref[:] = trajs[i]
            publisher.publish({"motor": {
                "pos_ref":     np.array(motor.mech_pos_ref),
                "pos":         np.array(motor.mech_pos),
                "vel":         np.array(motor.mech_vel),
                "torque":      np.array(motor.mech_torque),
                "temperature": np.array(motor.temperature),
            }})
            time.sleep(dt)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        motor.shutdown()
        raise e

    motor.mech_pos_ref[:] = start_pos  # return to start before shutdown
    time.sleep(0.2)
    motor.shutdown()


if __name__ == "__main__":
    main()
