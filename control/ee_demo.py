"""Minimal demo: remote end-effector hand control via port 9874.

Publishes ee_action commands to humanoid_real_env.py, which forwards them
to humanoid_end_effector_service.py over IPC.

Prerequisites on robot side:
    python humanoid_real_env.py --ee-service --arm_sender_ip <THIS_IP> ...
    python humanoid_end_effector_service.py --port /dev/ttyACM1

Usage:
    python ee_demo.py --robot-ip <robot-host>
    Keys: o=open  g=grab  c=close  q=quit
"""

import sys
import termios
import tty
import time
import dataclasses
import tyro
import humanoid_site as _site

from ipc.publisher import NNGPublisher

EE_PORT = 9874
COMMANDS = {"o": "hand_open", "g": "hand_grab", "c": "hand_close"}


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@dataclasses.dataclass
class Args:
    robot_ip: str = _site.ROBOT_IP


def main():
    args = tyro.cli(Args)
    pub = NNGPublisher(f"tcp://*:{EE_PORT}")
    time.sleep(0.2)  # allow socket to bind before first send

    print(f"EE demo — publishing on tcp://*:{EE_PORT}")
    print(f"Robot should connect with: --arm_sender_ip <THIS_IP> --ee-service")
    print("Keys: o=open  g=grab  c=close  q=quit\n")

    while True:
        ch = _getch()
        if ch == "q":
            break
        cmd = COMMANDS.get(ch)
        if cmd:
            pub.publish({"ee_action": cmd})
            print(f"  → {cmd}")
        else:
            print(f"  (unknown key '{ch}')")

    print("Done.")


if __name__ == "__main__":
    main()
