"""Bench script: runs its routine only when invoked (python humanoid_static_stand.py);
importing it is inert; it has no CLI flags. main() constructs and ENABLES all
27 motors (30 % torque ceiling, kp 100 legs/waist, 40 arms, kd 8), starts the
motion-control loop and ramps through a hand-scripted sequence of arm poses
over zeroed legs/waist at 300 Hz, publishing telemetry to
WORKSTATION_IP:TELEMETRY_PORT. Pre-policy stand/pose bench.
"""
import humanoid_site as _site
import numpy as np
import time
from humanoid_config import motor_setup
from hardware_bindings.motor.py_motor import CanMotorController
from ipc.publisher import DataPublisher

zero_pos = np.zeros(27) # 27 DOF
dt = 3.333e-3 # 300Hz
# dt = 0.01
d2r = np.deg2rad

waist_pos_1 = [0]
l_leg_pos_1 = [0,0,0,0,0,0]
r_leg_pos_1 = [0,0,0,0,0,0]
l_arm_pos_1 = [d2r(45),0,0,d2r(60),0,0,0]
r_arm_pos_1 = [-d2r(45),0,0,-d2r(60),0,0,0]

pos_halfway = np.array(waist_pos_1 + l_leg_pos_1 + r_leg_pos_1 + l_arm_pos_1 + r_arm_pos_1)

waist_pos_2 = [0]
l_leg_pos_2 = [0,0,0,0,0,0]
r_leg_pos_2 = [0,0,0,0,0,0]
l_arm_pos_2 = [d2r(90),0,0,d2r(0),0,0,0]
r_arm_pos_2 = [-d2r(90),0,0,-d2r(0),0,0,0]

pos_up = np.array(waist_pos_2 + l_leg_pos_2 + r_leg_pos_2 + l_arm_pos_2 + r_arm_pos_2)

waist_pos_3 = [0]
l_leg_pos_3 = [0,0,0,0,0,0]
r_leg_pos_3 = [0,0,0,0,0,0]
l_arm_pos_3 = [d2r(90),d2r(30),0,d2r(0),0,0,0]
r_arm_pos_3 = [-d2r(90),d2r(30),0,-d2r(0),0,0,0]


pos_left = np.array(waist_pos_3 + l_leg_pos_3 + r_leg_pos_3 + l_arm_pos_3 + r_arm_pos_3)

waist_pos_4 = [0]
l_leg_pos_4 = [0,0,0,0,0,0]
r_leg_pos_4 = [0,0,0,0,0,0]
l_arm_pos_4 = [d2r(90),-d2r(30),0,d2r(0),0,0,0]
r_arm_pos_4 = [-d2r(90),-d2r(30),0,-d2r(0),0,0,0]

pos_right = np.array(waist_pos_4 + l_leg_pos_4 + r_leg_pos_4 + l_arm_pos_4 + r_arm_pos_4)


def publish_state():
    """
    Publish motor state to the plotter
    """

    publisher.publish({"motor":{
    "pos_ref": np.array(motor.mech_pos_ref),
    "pos": np.array(motor.mech_pos),
    "vel": np.array(motor.mech_vel),
    "torque": np.array(motor.mech_torque),
    "temperature":np.array(motor.temperature),
    }})

def ramp_to_pos_and_hold(current_pos : np.ndarray, target_pos, t_ramp : float = 1, t_hold: float = 1):
    """
    Smooth ramp to target position

    t_ramp: time to ramp, float, default 1 second
    t_hold: hold time at target position, float, default 1 second
    target_pos: target position array, np.ndarray(27), default zero_pos (zero position)
    """
    if target_pos.shape != (27,):
        raise ValueError("target_pos must be of shape (27,)")
    if t_ramp < 0:
        raise ValueError("t_ramp must be positive")
    if t_hold < 0:
        raise ValueError("t_hold must be non-negative")
    if t_ramp != 0:
        #smooth ramp to target pos first
        steps_ramp = int(t_ramp/dt)
        for k in range(steps_ramp):
            alpha = (k+1)/steps_ramp
            motor.mech_pos_ref[:] = current_pos*(1-alpha)+target_pos*alpha
            publish_state()
            time.sleep(dt)

    #hold at target pos
    steps_hold = int(t_hold/dt)
    for k in range(steps_hold):
        motor.mech_pos_ref[:] = target_pos
        publish_state()
        time.sleep(dt)


def main():
    # 08-22: moved under main() so that importing the module is inert; `motor`
    # and `publisher` stay module globals because publish_state() and
    # ramp_to_pos_and_hold() read them.
    global motor, publisher
    motor = CanMotorController(motor_setup)

    target_url = f"udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}"
    publisher = DataPublisher(target_url,encoding="msgpack",broadcast=False)

    motor.set_max_torque_ratio(0.3) # set max torque to 30% of full torque
    motor.loc_kp[:13] = 100 # position pd, kp value for waist and legs
    motor.loc_kp[13:] = 40 # position pd, kp value for arms
    motor.spd_kp[:] = 8 # speed pd, kd value

    motor.enable()
    motor.start_motion_control_continuously()

    print(len(pos_up))
    try:

        ramp_to_pos_and_hold(motor.mech_pos, zero_pos, 2, 0)
        ramp_to_pos_and_hold(zero_pos, pos_halfway, 1, 0)
        ramp_to_pos_and_hold(pos_halfway, pos_up, 1, 0)
        ramp_to_pos_and_hold(pos_up, pos_left, 1, 0)
        ramp_to_pos_and_hold(pos_left, pos_right, 2, 0)
        ramp_to_pos_and_hold(pos_right, pos_up, 1, 0)
        ramp_to_pos_and_hold(pos_up, pos_halfway, 1, 0)
        ramp_to_pos_and_hold(pos_halfway, zero_pos, 2, 0)


    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        motor.shutdown()
        raise e

    motor.shutdown()


if __name__ == "__main__":
    main()
