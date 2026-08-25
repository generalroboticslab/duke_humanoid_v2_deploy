# chrt -f 80 python <control>/humanoid_record_path.py
"""
Flow:
  1. Connect to the arm motors and run self_check() only — NO torque is
     ever applied in this script, so the arm stays completely free to move
     by hand the whole time (same idea as the self_check step in
     humanoid_lock_and_grip.py, just never followed by an enable()).
  2. Press Enter when you're ready to start recording.
  3. Move the arm by hand along the path you want. While recording, you
     can press Enter at any point to drop a "waypoint" marker at the
     current sample (handy for tagging key poses along the sweep without
     stopping). This doesn't interrupt sampling — it runs on its own
     listener thread.
  4. Ctrl+C stops recording and saves everything to disk:
       - <name>.npz  -> t (T,), pos (T, N), waypoint_idx (K,)
       - <name>.csv  -> human-readable version of the same data
  5. The motors are never enabled, so there's nothing to ramp down or
     release — the script just exits.

Later, you can load the .npz, optionally pull out just the waypoint rows
(pos[waypoint_idx]) or downsample/smooth the full trajectory, and feed
that into your motion-control loop as mech_pos_ref targets (the same
mech_pos_ref[:] = ... pattern used in humanoid_lock_and_grip.py) to play
the path back.
"""

import csv
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from humanoid_config import motor_setup  # reuse your existing arm/leg/head setup
from hardware_bindings.motor.py_motor import CanMotorController


@dataclass
class Args:
    name: str = ""
    """Base filename for the saved recording (no extension). Defaults to a timestamp."""
    output_dir: str = "./recordings"
    """Directory the .npz / .csv files get written to."""
    hz: float = 200.0
    """Sampling rate while recording."""


def waypoint_listener(mark_queue: "queue.Queue[float]", stop_event: threading.Event):
    """Runs on its own thread: every Enter press during recording drops a
    waypoint marker (timestamped) onto the queue. Doesn't touch sampling."""
    while not stop_event.is_set():
        try:
            input()
        except EOFError:
            return
        if stop_event.is_set():
            return
        mark_queue.put(time.time())


def main():
    args = tyro.cli(Args)

    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or datetime.now().strftime("path_%Y%m%d_%H%M%S")
    npz_path = os.path.join(args.output_dir, f"{name}.npz")
    csv_path = os.path.join(args.output_dir, f"{name}.csv")

    motor = CanMotorController(motor_setup)
    motor.self_check()  # comms check only — no torque, arm stays free to move

    n_motors = len(np.array(motor.mech_pos, dtype=float))
    print(f"Connected. {n_motors} motors detected. No torque applied — "
          f"the arm is free to move by hand.")

    print("\nMove the arm into your starting pose.")
    input("Press Enter to start recording...\n")

    dt = 1.0 / args.hz
    mark_queue: "queue.Queue[float]" = queue.Queue()
    stop_event = threading.Event()
    listener_thread = threading.Thread(
        target=waypoint_listener, args=(mark_queue, stop_event), daemon=True
    )
    listener_thread.start()

    print("Recording... move the arm along the path you want.")
    print("Press Enter any time to drop a waypoint marker. Ctrl+C to stop and save.\n")

    times = []
    positions = []
    waypoint_idx = []

    t0 = time.time()
    next_tick = t0
    try:
        while True:
            now = time.time()
            pos = np.array(motor.mech_pos, dtype=float)
            times.append(now - t0)
            positions.append(pos)

            # Drain any waypoint marks that landed since the last sample —
            # they get attached to the sample index closest to now.
            while not mark_queue.empty():
                mark_queue.get()
                waypoint_idx.append(len(positions) - 1)
                print(f"  [waypoint marked @ t={now - t0:.2f}s, sample {len(positions) - 1}]")

            next_tick += dt
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()

    n_samples = len(positions)
    if n_samples == 0:
        print("No samples recorded — nothing to save.")
        return

    t_arr = np.array(times, dtype=float)
    pos_arr = np.array(positions, dtype=float)  # shape (T, N)
    wp_arr = np.array(sorted(set(waypoint_idx)), dtype=int)

    np.savez(npz_path, t=t_arr, pos=pos_arr, waypoint_idx=wp_arr)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t", "is_waypoint"] + [f"motor_{i}" for i in range(pos_arr.shape[1])]
        writer.writerow(header)
        wp_set = set(wp_arr.tolist())
        for i in range(n_samples):
            row = [t_arr[i], int(i in wp_set)] + pos_arr[i].tolist()
            writer.writerow(row)

    duration = t_arr[-1] - t_arr[0] if n_samples > 1 else 0.0
    print(f"\nSaved {n_samples} samples ({duration:.2f}s, {pos_arr.shape[1]} motors) "
          f"with {len(wp_arr)} waypoint(s) marked.")
    print(f"  {npz_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
