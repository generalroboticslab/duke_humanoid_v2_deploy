"""DESTRUCTIVE mechanical-zeroing one-off: writes the CURRENT position of every
motor in motor_setup as its zero and sets zero_sta=1 (reported range -pi..+pi),
saving both to the drives. Move every joint to its mechanical zero by hand
first; the script asks for a typed "yes" before acting. No CLI flags.
"""
from humanoid_config import motor_setup
from hardware_bindings.motor.py_motor import CanMotorController
from hardware_bindings.motor.py_motor import MOTO_PARAM   
import numpy as np
import time

if __name__ == "__main__":
    
    # keyboard type yes to continue
    if input("Press yes to continue setting zero position: ") != "yes":
        exit()

    # from ipc.publisher import DataPublisher

    motor = CanMotorController(motor_setup)
    ## manually move to zero position first

    motor.set_pos_zero(True)
    motor.save_data()
    
    time.sleep(0.1)

    motor.should_print_send = True
    motor.should_print_recv = True


    # Set zero_sta flag to 1 for all motors (position range: -π to +π instead of 0 to 2π)
    motor.setParam(MOTO_PARAM.zero_sta.value, np.ones(len(motor_setup), dtype=np.uint8))
    motor.save_data()

    # Read zero_sta parameter to confirm it's set
    motor.getParam(MOTO_PARAM.zero_sta.value)    

    motor.should_print_send = False
    motor.should_print_recv = False

    


    # # motor.enable()
    # time.sleep(0.1)
    # try:
    #     print(np.mean(motor.v_bus))
    #     print(motor.v_bus)
    #     print(motor.mech_pos)

    # except Exception as e:
    #     print(e)
    #     motor.disable(True)

    motor.disable(False)
