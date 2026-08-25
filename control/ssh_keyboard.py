"""Bench script: runs its routine only when invoked (python ssh_keyboard.py);
importing it is inert; it has no CLI flags. main() opens a UDP publisher to
WORKSTATION_IP:9871 and starts a blocking sshkeyboard listener (Ctrl+C / ESC
to stop). Legacy: publishes {"command_xy": [vx, vy]} on port 9871, which
nothing in humanoid_real_env subscribes to — the live nav path is nav_cmd on
9873 (gamepad.py, keyboard_gamepad_controller.py).
"""
import humanoid_site as _site
from sshkeyboard import listen_keyboard
from ipc.publisher import DataPublisher
import numpy as np
pi = np.pi


# data_publisher = NNGPublisher("tcp://*:9871")



command_xy = np.zeros(2,dtype=np.float32)
command_xy_current = np.zeros(2,dtype=np.float32)

def press(key):
    global command_xy
    if key == "i":
        # move forward
        command_xy[0] += 0.2
    if key == "k":
        # move backward
        command_xy[0] -= 0.2
    if key == "j":
        # move left
        command_xy[1] += 0.2
    if key == "l":
        # move right
        command_xy[1] -= 0.2
    if key == "0":
        command_xy.fill(0)

    if key == "w":
        # move forward
        command_xy[:] =  0.8,0
    if key == "s":
        # move backward
        command_xy[:] = -0.8,-0.05
    if key == "a":
        # move left
        command_xy[:] = 0,+0.8
    if key == "d":
        # move right
        command_xy[:] = 0,-0.8
    command_xy.round(1)
    # command_xy[np.abs(command_xy)<1e-3] = 0
    with np.printoptions(precision=1, suppress=True, formatter={"float":lambda x: f"{x:+5.1f}"},linewidth=100):
        print("command_xy:",command_xy)
    command_xy_current[:] = command_xy  # update current command

    # add soomthing before publishing



    data_publisher.publish({"command_xy":command_xy})


def main():
    # 08-22: moved under main() so that importing the module is inert;
    # `data_publisher` stays a module global because press() reads it.
    global data_publisher
    print("initializing keyboard listener...")
    data_publisher = DataPublisher(f"udp://{_site.WORKSTATION_IP}:9871", encoding="msgpack", broadcast=False)

    print("initializing keyboard listener publisher...")

    print("\033[92m" + "keyboard listener started" + "\033[0m")
    print("press 'i' to move forward, 'k' to move backward, 'j' to move left, 'l' to move right, '0' to stop")
    listen_keyboard(on_press=press,delay_second_char=0.001,delay_other_chars=0.005,sleep=0.0001)


if __name__ == "__main__":
    main()
