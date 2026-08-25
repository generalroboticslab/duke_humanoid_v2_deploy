# chrt -f 80 python <control>/humanoid_config.py
"""Motor table: joint name -> (CAN id, bus, motor type) in URDF joint order
(`motor_setup_dict`, `motor_setup`), imported by every motor-driving tool. Run
directly it self-checks every motor; `--zero` additionally ramps all joints to
zero at a 10 % torque ceiling.
"""

import humanoid_site as _site
from dataclasses import dataclass

import tyro

# 08-22: the old `sys.path.insert(0, <control>/..)` relic (pre-dating the
# hardware_bindings/ move) is gone — `hardware_bindings` lives in control/ and
# resolves from the script directory.
from hardware_bindings.motor.py_motor import R00, R02, R03, R04, R05, R06
can22 = "can22"
can9 = "can9"
can12 = "can12"
can24 = "can24"
can23 = "can23"
can21 = "can21"
can19 = "can19"
can25 = "can25"

# match to URDF ordering; dict preserves insertion order → index == URDF joint index
motor_setup_dict = {
    "waist":           (1,  can22, R03),  #0
    "left_hip_1":      (31, can24,  R03),  #1  
    "left_hip_2":      (32, can24,  R03),  #2
    "left_hip_3":      (33, can24,  R03),  #3
    "left_knee":       (34, can24,  R04),  #4
    "left_ankle_1":    (35, can24,  R03),  #5
    "left_ankle_2":    (36, can24,  R06),  #6
    "right_hip_1":     (41, can23, R03),  #7  
    "right_hip_2":     (42, can23, R03),  #8
    "right_hip_3":     (43, can23, R03),  #9  
    "right_knee":      (44, can23, R04),  #10
    "right_ankle_1":   (45, can23, R03),  #11
    "right_ankle_2":   (46, can23, R06),  #12
    "left_shoulder_1": (10, can22, R03),  #13
    "left_shoulder_2": (11, can9, R06),  #14
    "left_shoulder_3": (12, can9, R02),  #15 
    "left_elbow":      (13, can9, R02),  #16
    "left_wrist_1":    (14, can9, R02),  #17
    "left_wrist_2":    (15, can9, R00),  #18 
    "left_wrist_3":    (16, can9, R05),  #19
    "right_shoulder_1":(20, can22, R03),  #20
    "right_shoulder_2":(21, can21, R06),  #21
    "right_shoulder_3":(22, can21, R02),  #22
    "right_elbow":     (23, can21, R02),  #23
    "right_wrist_1":   (24, can21, R02),  #24 
    "right_wrist_2":   (25, can21, R00),  #25
    "right_wrist_3":   (26, can21, R05),  #26
    "cam_yaw_left":   (7, can25, R05),  #29
    "cam_pitch_left":   (8, can25, R05),  #30
    "cam_yaw_right":   (5, can25, R05),  #27
    "cam_pitch_right":   (6, can25, R05),  #28
}

motor_setup = list(motor_setup_dict.values())


@dataclass
class Args:
    zero: bool = False
    """Execute go_to_pos(motor) instead of just self_check."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    import numpy as np
    import time
    # from ipc.publisher import DataPublisher
    from hardware_bindings.motor.py_motor import CanMotorController
    
    from ipc.publisher import DataPublisher
    publisher = DataPublisher(f'udp://{_site.WORKSTATION_IP}:{_site.TELEMETRY_PORT}', encoding="msgpack", broadcast=False)

    
    motor = CanMotorController(motor_setup)
    # motor.set_pos_zero()
    # motor.enable()
    # time.sleep(0.1)
    # motor.disable(True)
    # motor.disable(False)

    def go_to_pos(motor, target_pos=np.zeros_like(motor.mech_pos)):
        motor.disable(True)
        time.sleep(0.01)
        motor.set_max_torque_ratio(0.1)
        motor.loc_kp[:] = 20 #5
        motor.spd_kp[:] = 5 #2
        motor.enable()
        motor.start_motion_control_continuously()
        start_pos = np.array(motor.mech_pos)
        num_steps = 300
        dt = 1/200
        for i in range(num_steps):
            t = i / num_steps
            s = 3 * t**2 - 2 * t**3
            pos_np = start_pos + s * (target_pos - start_pos)
            motor.mech_pos_ref[:] = pos_np
            time.sleep(dt)
            if i % 50 == 49:
                print(f"  {i+1}/{num_steps} ({100*t:.0f}%)")
            # print(motor.mech_pos)
            
            
            publisher.publish({"motor": {
                "pos":motor.mech_pos,
                "vel":motor.mech_vel,
            }})
            
        for i in range(300):
            time.sleep(dt)

        motor.loc_kp[:] = 0
        motor.spd_kp[:] = 5
        time.sleep(1)

        # motor.self_check()

    try:
        # pass
        motor.self_check()
        if args.zero:
            go_to_pos(motor)
        # monitor_motor_pos(motor)

    except Exception as e:
        print(e)
        motor.disable(True)

    motor.disable(False)
