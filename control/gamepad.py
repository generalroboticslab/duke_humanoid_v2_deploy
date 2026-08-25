"""Bench script: runs its routine only when invoked (python gamepad.py); importing
it is inert; it has no CLI flags (`--help` is ignored). main() binds
tcp://*:9873 and starts pygame; with humanoid_real_env connected every
published command drives the base. Ctrl+C to stop.

Bluetooth gamepad -> base velocity command for humanoid_real_env.

    python gamepad.py                 # then run real_env with
                                      #   --high-level-controller-ip <THIS IP>

Left stick = yaw, right stick = lateral, left stick vertical = forward (the
axis map below is unchanged from the original). Publishes ONLY the base
command; the arms are not touched.

PATCH (07-31): WIRED TO THE ROBOT.
As authored upstream (05-26) this could not drive humanoid_real_env at all —
three independent mismatches against the nav path it is meant to feed:

  port   tcp://*:9871  ->  real_env subscribes {high_level_controller_ip}:9873
  key    "cmd"         ->  real_env reads data["nav_cmd"]
  rate   published only when the stick MOVED

The third is the dangerous one and it is invisible in a code read: real_env
gates on `nav_fresh = now - last_nav_recv_time < _NAV_CMD_TIMEOUT` (1.0 s) and
refreshes that stamp only when the publisher's data_id CHANGES. Holding the
stick steady therefore stopped the robot one second in — while the operator
was still commanding it. Now every tick publishes, so a held stick keeps the
stamp fresh; releasing it still zeroes the command, and QUITTING still lets the
1 s timeout stop the base, which is the intended dead-man behaviour.
"""

import time
import numpy as np
import pygame

from ipc.publisher import NNGPublisher


lin_vel_x_max = 0.3
lin_vel_y_max = 0.25
ang_vel_z_max = 0.2

axis_deadzone = 0.08
reconnect_interval_s = 0.5
loop_sleep_s = 0.01
publish_period_s = 0.05      # 20 Hz: well inside real_env's 1.0 s nav timeout


def _attach_first_joystick():
    if pygame.joystick.get_count() <= 0:
        return None
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[gamepad] connected: {js.get_name()}")
    return js


def main():
    # 08-22: moved under main() so that importing the module is inert.
    keyboard_operator_cmd = np.zeros(3)  # vel_x, vel_y, yaw_orientation

    data_publisher = NNGPublisher("tcp://*:9873")   # real_env: --high-level-controller-ip

    pygame.init()
    pygame.joystick.init()

    joystick = None
    running = True
    last_scan_time = 0.0
    last_publish_time = 0.0

    try:
        while running:
            updated = False
            reset = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN and event.button == 4:
                    reset = True
                    updated = True
                elif event.type == getattr(pygame, "JOYDEVICEREMOVED", -1):
                    if joystick is not None:
                        joystick = None
                        keyboard_operator_cmd[:] = 0.0
                        updated = True
                        print("[gamepad] disconnected")
                elif event.type == getattr(pygame, "JOYDEVICEADDED", -1):
                    if joystick is None:
                        joystick = _attach_first_joystick()

            now = time.time()
            if joystick is None and (now - last_scan_time) >= reconnect_interval_s:
                last_scan_time = now
                joystick = _attach_first_joystick()

            if joystick is not None:
                try:
                    raw_x = joystick.get_axis(1)
                    raw_y = joystick.get_axis(3)
                    raw_yaw = joystick.get_axis(0)
                except pygame.error:
                    joystick = None
                    keyboard_operator_cmd[:] = 0.0
                    updated = True
                    print("[gamepad] disconnected")
                else:
                    if abs(raw_x) < axis_deadzone:
                        raw_x = 0.0
                    if abs(raw_y) < axis_deadzone:
                        raw_y = 0.0
                    if abs(raw_yaw) < axis_deadzone:
                        raw_yaw = 0.0

                    new_cmd = np.array([
                        np.round(-raw_x * lin_vel_x_max, decimals=2),
                        np.round(-raw_y * lin_vel_y_max, decimals=2),
                        np.round(-raw_yaw * ang_vel_z_max, decimals=2),
                    ])
                    if not np.array_equal(new_cmd, keyboard_operator_cmd):
                        keyboard_operator_cmd[:] = new_cmd
                        updated = True

            # EVERY tick, not only on change — see the module docstring. `reset` is
            # edge-triggered so it must not be latched into the steady stream.
            if now - last_publish_time >= publish_period_s or updated:
                last_publish_time = now
                data_publisher.publish({
                    "nav_cmd": [float(v) for v in keyboard_operator_cmd],
                    "reset": bool(reset),
                })
                if updated:                       # print on CHANGE only, not at 20 Hz
                    print(f"[gamepad] vx={keyboard_operator_cmd[0]:+.2f} "
                          f"vy={keyboard_operator_cmd[1]:+.2f} "
                          f"wz={keyboard_operator_cmd[2]:+.2f}"
                          + ("  RESET" if reset else ""))

            time.sleep(loop_sleep_s)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
