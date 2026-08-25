"""Bench one-off: runs its routine only when invoked (python disable_servo.py);
importing it is inert; it has no CLI flags. main() opens /dev/ttyACM1
(hard-coded) and torque-disables FEETECH servo ID 19 through the pure-Python
driver, then closes the port. Edit the port/ID literals before use.
"""
from ft_servo_python_only import FtServo


def main():
    servo = FtServo("/dev/ttyACM1")
    servo.enable_torque(19, False)
    servo.close()


if __name__ == "__main__":
    main()
