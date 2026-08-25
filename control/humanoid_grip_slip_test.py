"""Bench slip test for the grip-hold torque ceilings (07-18 stall-heat plan).

Run with the EE service up (T5) and a cube between the fingers of ONE side.
Sequence per torque level, high to low: hand_grab (full squeeze) -> settle ->
hand_hold at the level -> 8 s observation window while YOU tug the cube the
way a carry would load it. Note the lowest level with zero slip; set
--hold-torque-carry / --hold-torque-park to ~1.5x that.

Usage:  python humanoid_grip_slip_test.py --side left [--levels 300 250 200 150]
Ctrl+C any time; the script opens the gripper on exit.
"""
import time

import tyro

from humanoid_end_effector_service import EEServiceClient


def main(side: str = "left", levels: tuple[int, ...] = (300, 250, 200, 150),
         observe_s: float = 8.0):
    c = EEServiceClient()
    try:
        for _ in range(60):
            c.poll(); time.sleep(0.1)
            if c.alive:
                break
        if not c.alive:
            print("EE service not alive — start T5 first"); return
        input(f"Place the cube between the {side.upper()} fingers, then Enter…")
        print("[slip] full-torque grab (350)…")
        c.send_action(side, "hand_grab")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            c.poll(); time.sleep(0.1)
            if c.grasp_detected.get(side):
                break
        if not c.grasp_detected.get(side):
            print("[slip] no grasp detected — re-seat the cube and rerun"); return
        for tq in levels:
            print(f"\n[slip] hand_hold torque={tq} — tug the cube for "
                  f"{observe_s:.0f}s (carry-like load). Watch for slip.")
            c.send_action(side, "hand_hold", torque=tq)
            t0 = time.time()
            while time.time() - t0 < observe_s:
                c.poll(); time.sleep(0.1)
            ans = input(f"[slip] torque {tq}: slipped? [y/N] ").strip().lower()
            if ans == "y":
                print(f"\n[slip] RESULT: slip at {tq} — use ceilings ≥ "
                      f"{int(tq * 1.5)} (1.5x margin)")
                break
        else:
            print(f"\n[slip] RESULT: no slip down to {levels[-1]} — the "
                  f"defaults (carry 250 / park 200) have margin")
    finally:
        input("\nSupport the cube, then Enter to OPEN the gripper…")
        c.send_action(side, "hand_open")
        time.sleep(1.0)
        c.close()


if __name__ == "__main__":
    tyro.cli(main)
