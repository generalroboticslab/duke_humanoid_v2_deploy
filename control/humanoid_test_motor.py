"""Bench script: runs its routine only when invoked (python humanoid_test_motor.py);
importing it is inert; it has no CLI flags. main() constructs and ENABLES all
motors (5 % torque ceiling, kp 10 / kd 5), starts the motion-control loop and
drives the arm joints (index 13 onward) through a 0.1 rad sine at 300 Hz for
9000 ticks, publishing telemetry to WORKSTATION_IP:TELEMETRY_PORT. First
power-on smoke test.
"""
import humanoid_site as _site
import numpy as np
import time
from tqdm import trange

from humanoid_config import motor_setup
from hardware_bindings.motor.py_motor import CanMotorController
from ipc.publisher import DataPublisher


n = 900
trajactory = -0.1* np.sin(np.linspace(0, -2 * np.pi, n))


def main():
    # 08-22: moved under main() so that importing the module is inert.
    motor = CanMotorController(motor_setup)

    target_url = f"udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}"
    publisher = DataPublisher(target_url,encoding="msgpack",broadcast=False)


    motor.set_max_torque_ratio(0.05)
    motor.loc_kp[:] = 10 # position pd, kp value
    motor.spd_kp[:] = 5 # speed pd, kd value




    # motor.set_pos_zero()

    motor.enable()
    motor.start_motion_control_continuously()


    range_obj = trange(n*10)
    try:
        for i in range_obj:


            motor.mech_pos_ref[13:] = trajactory[i%n]

            # print(
                # f"pos_ref: {motor.mech_pos_ref}\n",
                # f"pos: {motor.mech_pos}\n",
                # f"vel: {motor.mech_vel}\n",
                # f"torque: {motor.mech_torque}\n",
                # f"temperature: {motor.temperature}\n",
            # )


            publisher.publish({"motor":{
                "pos_ref": np.array(motor.mech_pos_ref),
                "pos": np.array(motor.mech_pos), # motor.mech_pos,
                "vel": np.array(motor.mech_vel), # motor.mech_vel,
                "torque": np.array(motor.mech_torque), # motor.mech_torque
                "temperature":np.array(motor.temperature),
            }})
            time.sleep(3.333e-3)


    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        motor.shutdown()
        raise e

    motor.shutdown()


if __name__ == "__main__":
    main()
