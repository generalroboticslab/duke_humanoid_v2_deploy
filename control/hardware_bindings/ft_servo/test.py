"""
test.py — ft_servo viser GUI.

Live GUI to monitor and control servo positions.

Usage (either form works since 2026-08-22; before that only the module form
did):
    python -m hardware_bindings.ft_servo.test   # from control/
    python test.py                              # any cwd

Bench script, no CLI: acts on invocation — opens both hand servo ports, pings
the IDs, ENABLES TORQUE and starts polling at module level, then serves the
GUI on http://localhost:8080 until Ctrl+C (torque is released on shutdown).
Needs the built ft_servo_ext.abi3.so and viser.
"""

import time
import signal
import sys
if not __package__:
    # Plain-script invocation (`python test.py`): put control/ on sys.path so
    # the absolute package import resolves; `python -m ...` from control/ takes
    # the relative-import branch.
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from hardware_bindings.ft_servo import FtServo
else:
    from . import FtServo

import viser

# ── Config ───────────────────────────────────────────────────────────────────
LEFT_PORT  = "/dev/ttyACMservoLeft"
RIGHT_PORT = "/dev/ttyACMservoRight"
LEFT_IDS  = [1, 2]
RIGHT_IDS = [1, 2]

PORT_IDS = [(LEFT_PORT, LEFT_IDS, "Left Arm"), (RIGHT_PORT, RIGHT_IDS, "Right Arm")]

# ── Startup ─────────────────────────────────────────────────────────────────
servos = []
for port, ids, name in PORT_IDS:
    drv = FtServo(port)
    for sid in ids:
        ret = drv.ping(sid)
        print(f"ping({name},{sid}) = {ret}")
    servos.append((drv, ids, name))

print("\nStarting viser GUI...")

for drv, ids, name in servos:
    for sid in ids:
        drv.enable_torque(sid, True)

for drv, ids, name in servos:
    drv.start_poll(ids, interval_us=5000)
time.sleep(0.2)

# Read initial positions for slider defaults (clamped to valid range)
initial_positions = {}
for drv, ids, name in servos:
    for sid in ids:
        pos = drv.get_positions([sid])[0]
        initial_positions[(name, sid)] = int(drv.get_positions([sid])[0])

print(initial_positions)

# ── Viser Server ────────────────────────────────────────────────────────────
server = viser.ViserServer(port=8080)
tab_group = server.gui.add_tab_group()

# Per-servo handles: { (name, sid): { "pos": handle, "load": handle, "target": handle } }
handles = {}

for drv, ids, name in servos:
    tab = tab_group.add_tab(name)
    with tab:
        for sid in ids:
            server.gui.add_markdown(f"**Servo ID {sid}**")
            handles[(name, sid)] = {
                "pos":    server.gui.add_slider(f"{name}({sid}) pos",    min=0, max=8192, step=1, initial_value=0, disabled=True),
                "load":   server.gui.add_slider(f"{name}({sid}) load",   min=0, max=4096, step=1, initial_value=0, disabled=True),
                "target": server.gui.add_slider(f"{name}({sid}) target",  min=0, max=8192, step=1, initial_value=initial_positions[(name, sid)]),
            }

            # Dragging the target slider automatically sends position
            handles[(name, sid)]["target"].on_update(lambda _, drv=drv, sid=sid, name=name: (
                drv.set_position(sid, handles[(name, sid)]["target"].value, speed=500, acc=100, torque=1000)
            ))

        server.gui.add_markdown("---")

print("GUI ready at http://localhost:8080")

# ── Main loop ────────────────────────────────────────────────────────────────
def shutdown(signum, frame):
    print("\nStopping...")
    for drv, ids, name in servos:
        drv.stop_poll()
        for sid in ids:
            drv.enable_torque(sid, False)
        drv.close()
    server.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

while True:
    for drv, ids, name in servos:
        for sid in ids:
            h = handles[(name, sid)]
            h["pos"].value  = drv.get_positions([sid])[0]
            h["load"].value = drv.get_loads([sid])[0]
    time.sleep(0.1)
