#!/usr/bin/env python3
"""
Standalone Keyboard & Joystick Controller
Dependencies: pip install sshkeyboard pygame
"""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import time
from queue import Empty, Queue
from threading import Thread

import pygame
from sshkeyboard import listen_keyboard, stop_listening


class KeyboardThread(Thread):
    """Keyboard input using sshkeyboard (SSH-compatible)."""

    def __init__(self, event_queue: Queue):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.running = True
        self.arrows = {"up": False, "down": False, "left": False, "right": False}
        # WASD for movement, QE for turning
        self.wasd = {"w": False, "s": False, "a": False, "d": False, "q": False, "e": False}

    def run(self):
        def on_key(key, pressed):
            if not self.running:
                stop_listening()
                return
            if key in self.arrows:
                self.arrows[key] = pressed
            if key in self.wasd:
                self.wasd[key] = pressed
            self.event_queue.put({"type": "keyboard", "name": key, "pressed": pressed})

        listen_keyboard(on_press=lambda k: on_key(k, True), on_release=lambda k: on_key(k, False))

    def get_axes(self) -> dict[str, float]:
        """WASD/arrows as axes: W/S or up/down = LeftY, A/D or left/right = LeftX, Q/E = RightX."""
        # Forward/backward: W/S or up/down arrows
        y = (float(self.wasd["w"]) - float(self.wasd["s"]) +
             float(self.arrows["up"]) - float(self.arrows["down"]))
        # Strafe left/right: A/D or left/right arrows
        x = (float(self.wasd["d"]) - float(self.wasd["a"]) +
             float(self.arrows["right"]) - float(self.arrows["left"]))
        # Turn: Q/E
        rx = float(self.wasd["e"]) - float(self.wasd["q"])
        return {"LeftX": max(-1, min(1, x)), "LeftY": max(-1, min(1, y)), "RightX": max(-1, min(1, rx))}

    def stop(self):
        self.running = False
        stop_listening()


# bluetoothctl
# power on
# agent on
# default-agent
# scan on
# pair XX:XX:XX:XX:XX:XX
# trust XX:XX:XX:XX:XX:XX
# connect XX:XX:XX:XX:XX:XX

# jstest /dev/input/js0
class JoystickThread(Thread):
    """Joystick input using pygame."""

    # Mapping for Xbox controllers on Linux. 
    BUTTONS = {
        0: "A", 1: "B", 2: "X", 3: "Y", 4: "LB", 5: "RB",
        6: "Back", 7: "Start", 8: "Guide", 9: "LS", 10: "RS"
    }
    AXES = {"LeftX": 0, "LeftY": 1, "RightX": 2, "RightY": 3, "LT": 5, "RT": 4}
    DPAD = {"Up": (1, 1), "Right": (0, 1), "Down": (1, -1), "Left": (0, -1)}

    def __init__(self, event_queue: Queue):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.running = True
        self.connected = False
        self.axes = {k: 0.0 for k in self.AXES}
        self.trigger_state = {"LT": False, "RT": False}

    def run(self):
        pygame.init()

        js = None
        js_instance_id = None

        # Handle already-connected joystick at startup
        if pygame.joystick.get_count() > 0:
            js = pygame.joystick.Joystick(0)
            js.init()
            js_instance_id = js.get_instance_id()
            self.connected = True
            print(f"[Joystick] {js.get_name()}")
        else:
            print("[Joystick] No joystick — waiting for connect...")

        dpad_state = {k: False for k in self.DPAD}
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():  # get() calls pump() internally
                if event.type == pygame.JOYDEVICEADDED:
                    js = pygame.joystick.Joystick(event.device_index)
                    js.init()
                    js_instance_id = js.get_instance_id()
                    self.connected = True
                    dpad_state = {k: False for k in self.DPAD}
                    print(f"[Joystick] Connected: {js.get_name()}")

                elif event.type == pygame.JOYDEVICEREMOVED:
                    if event.instance_id == js_instance_id:
                        js = None
                        js_instance_id = None
                        self.connected = False
                        self.axes = {k: 0.0 for k in self.AXES}
                        self.trigger_state = {"LT": False, "RT": False}
                        print("[Joystick] Disconnected")

                elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP) and js is not None:
                    name = self.BUTTONS.get(event.button, f"Btn{event.button}")
                    if event.type == pygame.JOYBUTTONDOWN:
                        print(f"[Joystick] Button {event.button} -> {name}")
                    self.event_queue.put({"type": "button", "name": name, "pressed": event.type == pygame.JOYBUTTONDOWN})

                elif event.type == pygame.JOYHATMOTION and js is not None:
                    for name, (axis, direction) in self.DPAD.items():
                        pressed = event.value[axis] == direction
                        if pressed != dpad_state[name]:
                            dpad_state[name] = pressed
                            self.event_queue.put({"type": "button", "name": name, "pressed": pressed})

            # Update axes
            if js is not None:
                for name, idx in self.AXES.items():
                    try:
                        val = js.get_axis(idx)
                    except pygame.error:
                        continue

                    if "Y" in name:
                        val = -val  # Invert Y
                    elif name in ("LT", "RT"):
                        val = (val + 1) / 2  # Normalize triggers to [0, 1]
                        # Treat triggers as buttons
                        pressed = val > 0.5
                        if pressed != self.trigger_state[name]:
                            self.trigger_state[name] = pressed
                            if pressed:
                                print(f"[Joystick] Trigger {name} -> Button")
                            self.event_queue.put({"type": "button", "name": name, "pressed": pressed})
                    self.axes[name] = round(val, 3)

            clock.tick(100)

    def stop(self):
        self.running = False


class KeyboardGampadController:
    """Combined keyboard + joystick controller."""

    def __init__(self, kb_triggers: dict = None, js_triggers: dict = None, combo_buttons: set = None):
        self.kb_triggers = kb_triggers or {"esc": "[SHUTDOWN]", "space": "[TEST]"}
        self.js_triggers = js_triggers or {"A": "[SHUTDOWN]", "X": "[FADE_IN]", "B": "[FADE_OUT]"}
        self.combo_buttons = combo_buttons or {"LB", "RB"}
        self.held = set()

        self.queue = Queue(maxsize=200)
        self.keyboard = KeyboardThread(self.queue)
        self.joystick = JoystickThread(self.queue)

    def start(self):
        self.keyboard.start()
        self.joystick.start()
        time.sleep(0.3)

    def stop(self):
        self.keyboard.stop()
        self.joystick.stop()

    def poll(self) -> tuple[list[str], dict[str, float]]:
        """Drain the input queue and return (triggered_commands, axes)."""
        commands = []

        while not self.queue.empty():
            try:
                e = self.queue.get_nowait()
            except Empty:
                break

            cmd = None
            if e["type"] == "keyboard" and not e["pressed"]:
                cmd = self.kb_triggers.get(e["name"])
                if not cmd:
                    print(f"[KB] Key: {e['name']}")
            elif e["type"] == "button":
                if e["name"] in self.combo_buttons:
                    self.held.add(e["name"]) if e["pressed"] else self.held.discard(e["name"])
                
                if e["pressed"]:
                    other_held = sorted(h for h in self.held if h != e["name"])
                    key = "+".join(other_held + [e["name"]]) if other_held else e["name"]
                    cmd = self.js_triggers.get(key)
                    if not cmd:
                        print(f"[JS] Button: {key}")
            if cmd:
                commands.append(cmd)

        # Axes: keyboard takes priority when any key is held
        kb_axes = self.keyboard.get_axes()
        if any(v != 0 for v in kb_axes.values()):
            axes = {**self.joystick.axes, **kb_axes}  # kb overrides js
        else:
            axes = self.joystick.axes  # live dict, no copy needed

        return commands, axes


def main():
    print("Combined Controller | ESC to exit | Arrow keys for movement")
    print("-" * 50)

    ctrl = KeyboardGampadController(
        kb_triggers={"esc": "[SHUTDOWN]", "`": "[REBORN]", "space": "[TEST]"},
        js_triggers={"A": "[SHUTDOWN]", "X": "[FADE_IN]", "B": "[FADE_OUT]", "Y": "[RESET]", "LB+RB+A": "[COMBO]"},
    )
    ctrl.start()

    try:
        while True:
            commands, axes = ctrl.poll()

            if any(abs(v) > 0.1 for v in axes.values()):
                print(f"[AXES] {' '.join(f'{k}={v:+.2f}' for k, v in axes.items() if abs(v) > 0.1)}")

            for cmd in commands:
                print(f">>> {cmd}")
                # if cmd == "[SHUTDOWN]":
                #     ctrl.stop()
                #     return

            time.sleep(0.01)
    except KeyboardInterrupt:
        ctrl.stop()


if __name__ == "__main__":
    main()
