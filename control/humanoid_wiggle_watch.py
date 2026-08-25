# python humanoid_wiggle_watch.py
"""Live motor-dropout alarm for the harness wiggle test (07-25).

PURPOSE
  The 07-24 collisions were caused by right_shoulder_2 + right_shoulder_3
  (CAN 21/22, both on can21) going silent on the bus for 0.3-0.4 s — far too
  short to see by eye, loud and unmistakable on telemetry. This tool watches
  the 9870 stream and raises an immediate console alarm (+ terminal bell) the
  moment any watched joint's feedback freezes, so a hand on the harness gets
  instant cause-and-effect: "I pressed THIS connector and THAT pair dropped."

DETECTION
  A joint counts as DROPPED when ALL of:
    * its (pos, vel, eff) telemetry triple is bit-identical for >= --ticks
      consecutive rows (default 5 = ~100 ms; the real 07-24 events lasted
      15-20 rows, huge margin);
    * the vel value it froze AT satisfies |vel| > --vel-eps (default 0.05
      rad/s) — "frozen mid-motion". This is the load-bearing discriminator
      on an UNPOWERED arm: a healthy still motor's velocity decays to ~0
      BEFORE its position stops updating, so only a link break can freeze a
      clearly-nonzero velocity in place;
    * at least two other joints of the same arm are GENUINELY MOVING
      (|d pos| > --move-eps rad) — the trustworthy reference side.
  History: the first hardware run (07-25 v1, pair = pos+eff only, 2 ticks)
  drowned in false alarms — with motors unpowered the effort word is
  constant, the criterion degenerated to position alone, and every joint the
  hand wasn't actively moving "froze" within 40 ms (104 bogus events in 74 s,
  spanning both buses). The vel-in-triple + frozen-mid-motion rules above are
  the fix; both were absent from that run, so its output is unusable.
  Practical consequence: KEEP THE WHOLE ARM STIRRING — especially the joints
  of the branch you are pressing on. A joint must be moving when its link
  breaks for the alarm to catch it.

  A stalled telemetry STREAM (no packets at all) is reported separately and
  explicitly NOT as a motor dropout — that is a real_env-side condition.

USAGE
    python humanoid_wiggle_watch.py                     # right arm, localhost
    python humanoid_wiggle_watch.py --arm left          # control comparison
    python humanoid_wiggle_watch.py --ticks 3           # stricter trigger

  Run against a real_env started in the SAFE bench profile (arms unpowered,
  everything still polled on CAN):
    python humanoid_real_env.py --task HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine \
        --torque_limit 0.01 --enable-motor camera

  Ctrl+C prints a summary of every event with timestamps — read it against
  your notes of which connector was being handled when.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ipc.publisher import NNGSubscriber

ARM_SLICE = {"left": slice(13, 20), "right": slice(20, 27)}
NAMES = ["shoulder_1", "shoulder_2", "shoulder_3", "elbow",
         "wrist_1", "wrist_2", "wrist_3"]
# CAN id + bus per joint (humanoid_config.py) — printed in alarms so the
# operator maps a dropout straight onto the harness diagram.
WIRE = {
    "right": [("ID20", "can22")] + [(f"ID{21+i}", "can21") for i in range(6)],
    "left":  [("ID10", "can22")] + [(f"ID{11+i}", "can9") for i in range(6)],
}
RED, GRN, YEL, OFF = "\033[91m", "\033[92m", "\033[93m", "\033[0m"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", default="127.0.0.1")
    p.add_argument("--arm", default="right", choices=["left", "right"])
    p.add_argument("--ticks", type=int, default=5,
                   help="consecutive frozen rows before the alarm fires "
                        "(~20 ms/row; real 07-24 breaks lasted 15-20 rows)")
    p.add_argument("--move-eps", type=float, default=1e-3,
                   help="rad of per-tick position change for a joint to count "
                        "as a MOVING reference (sensor noise stays below this; "
                        "a hand-rocked arm stays above it)")
    p.add_argument("--vel-eps", type=float, default=0.05,
                   help="rad/s: a frozen joint only alarms if it froze AT a "
                        "velocity above this — 'frozen mid-motion'. A healthy "
                        "still motor decays through ~0 first; a link break "
                        "freezes whatever velocity was live")
    args = p.parse_args()
    sl = ARM_SLICE[args.arm]
    wire = WIRE[args.arm]

    sub = NNGSubscriber(f"tcp://{args.robot_ip}:9870"); sub.start()
    print(f"[watch] {args.arm.upper()} arm dropout alarm — telemetry "
          f"{args.robot_ip}:9870, trigger = {args.ticks} frozen ticks "
          f"AND frozen |vel| > {args.vel_eps} rad/s (frozen mid-motion)")
    print("[watch] joints: " + ", ".join(
        f"{n}({i}/{b})" for n, (i, b) in zip(NAMES, wire)))
    print("[watch] KEEP THE ARM GENTLY MOVING BY HAND — a still, unpowered "
          "arm can freeze legitimately;\n[watch] the alarm needs live joints "
          "to compare against.\n")

    prev = None
    prev_ms = None              # firmware mode_status of this arm's 7 joints
    prev_ec = None              # firmware error_code of same
    consec = np.zeros(7, dtype=int)
    last_move_t = np.zeros(7)   # wall time each joint last moved > move-eps
    last_id = -1
    last_rx = time.time()
    stream_stalled = False
    alarmed: frozenset = frozenset()
    alarm_t0 = 0.0
    alarm_cof: list = []        # co-frozen joints captured at alarm start
    events: list[tuple[float, float, frozenset, tuple]] = []
    ticks = 0
    t0 = time.time()
    last_beat = t0

    try:
        while True:
            now = time.time()
            if sub.data_id == last_id or sub.data is None:
                # Whole-stream stall: real_env stopped publishing. Explicitly
                # NOT a motor dropout — never mix the two in the log.
                if now - last_rx > 0.3 and not stream_stalled and last_id != -1:
                    stream_stalled = True
                    print(f"{YEL}[watch] telemetry stream stalled "
                          f"({now - last_rx:.1f}s) — real_env side, NOT a "
                          f"motor dropout{OFF}")
                time.sleep(0.002)
                continue
            last_id = sub.data_id
            last_rx = now
            if stream_stalled:
                stream_stalled = False
                print(f"{YEL}[watch] telemetry stream resumed{OFF}")
            jp = np.asarray(sub.data.get("joint_pos") or [], float)
            jv = np.asarray(sub.data.get("joint_vel") or [], float)
            je = np.asarray(sub.data.get("joint_effort") or [], float)
            if jp.size < 27 or jv.size < 27 or je.size < 27:
                continue
            cur = np.stack([jp[sl], jv[sl], je[sl]])  # (3,7) pos/vel/eff
            # 07-25b firmware-side watch (needs rebuilt bindings + patched
            # real_env; silently absent otherwise). Catches the class the
            # freeze rule can't: a motor still answering on the bus whose
            # firmware left Running (self-protection / reboot) — feedback
            # keeps updating, so only this line ever reports it.
            ms = np.asarray(sub.data.get("mode_status") or [], int)
            if ms.size >= 27:
                cms = ms[sl]
                if prev_ms is not None and np.any(cms != prev_ms):
                    for j in np.nonzero(cms != prev_ms)[0]:
                        style = GRN if cms[j] == 2 else RED
                        print(f"\a{style}[{now-t0:8.2f}s] ██ MODE ██ "
                              f"{NAMES[j]}({wire[j][0]}/{wire[j][1]}) "
                              f"{int(prev_ms[j])} -> {int(cms[j])}  "
                              f"(0=Reset 1=Calib 2=Running 255=Unknown){OFF}")
                prev_ms = cms
            ec = np.asarray(sub.data.get("error_code") or [], int)
            if ec.size >= 27:
                cec = ec[sl]
                fresh = (cec != 0) & (cec != (prev_ec if prev_ec is not None
                                              else np.zeros(7, int)))
                for j in np.nonzero(fresh)[0]:
                    print(f"\a{RED}[{now-t0:8.2f}s] ██ FAULT ██ "
                          f"{NAMES[j]}({wire[j][0]}/{wire[j][1]}) "
                          f"error_code={int(cec[j])}  (1=UV 2=OC 4=OT "
                          f"8=mag-enc 16=hall-enc 32=uncal){OFF}")
                prev_ec = cec
            ticks += 1
            if prev is not None:
                frozen = np.all(cur == prev, axis=0)  # bitwise per joint
                consec = np.where(frozen, consec + 1, 0)
                # Reference joints must show REAL motion, not just a flipped
                # noise bit — a still arm keeps the gate closed (quiet).
                moving = np.abs(cur[0] - prev[0]) > args.move_eps
                last_move_t[moving] = now
                live = int(np.sum(moving))
                # Frozen-mid-motion: cur[1] IS the frozen velocity while the
                # streak lasts (bit-identical by definition). A joint resting
                # naturally froze at ~0 vel and never alarms.
                hit = frozenset(int(j) for j in range(7)
                                if consec[j] >= args.ticks
                                and abs(cur[1][j]) > args.vel_eps) \
                    if live >= 2 else frozenset()
                if hit and not alarmed:
                    alarm_t0 = now
                    alarm_cof = [j for j in range(7)
                                 if consec[j] >= max(2, args.ticks - 2)
                                 and j not in hit]
                    # Co-frozen joints: same-length freeze streak but frozen
                    # at ~0 vel. Alone they never alarm (a resting joint looks
                    # identical); anchored by a genuine mid-motion freeze they
                    # are almost certainly the same break — the 07-24 incident
                    # itself froze sh2 at -0.001 rad/s and sh3 at 0.134, so
                    # without this line the pair membership would be invisible.
                    # >= ticks-2, not ticks: a co-frozen joint's streak can
                    # start a row or two late (its last live sample happened
                    # to repeat), and this line prints only once per event.
                    cof = [j for j in range(7)
                           if consec[j] >= max(2, args.ticks - 2)
                           and j not in hit]
                    print(f"\a{RED}[{now-t0:8.2f}s] ██ DROPOUT ██  "
                          + ", ".join(f"{NAMES[j]}({wire[j][0]}/{wire[j][1]}"
                                      f", frozen at {cur[1][j]:+.2f} rad/s)"
                                      for j in sorted(hit))
                          + (f"  [co-frozen, near-zero vel: "
                             + ", ".join(NAMES[j] for j in cof) + "]"
                             if cof else "")
                          + f"  — {live} other joint(s) still live{OFF}")
                elif alarmed and not hit:
                    dur = now - alarm_t0
                    events.append((alarm_t0 - t0, dur, alarmed,
                                   tuple(alarm_cof)))
                    print(f"{GRN}[{now-t0:8.2f}s] recovered after {dur:.2f}s  "
                          f"({', '.join(NAMES[j] for j in sorted(alarmed))})"
                          f"{OFF}")
                elif hit and hit != alarmed:
                    print(f"{RED}[{now-t0:8.2f}s]    dropout set changed: "
                          + ", ".join(NAMES[j] for j in sorted(hit)) + OFF)
                alarmed = hit
            prev = cur
            if now - last_beat >= 5.0:
                last_beat = now
                mv = ", ".join(NAMES[j] for j in range(7)
                               if now - last_move_t[j] < 1.0)
                print(f"[{now-t0:8.2f}s] alive · {ticks} ticks · "
                      f"{len(events)} event(s) · moving: {mv or '(none — MOVE THE ARM, alarm is gated off)'}")
    except KeyboardInterrupt:
        pass
    finally:
        sub.stop()
        print(f"\n[watch] ===== SUMMARY: {len(events)} dropout event(s) "
              f"over {time.time()-t0:.0f}s =====")
        for st, dur, js, cof in events:
            line = (f"   t={st:8.2f}s  {dur:5.2f}s  "
                    + ", ".join(f"{NAMES[j]}({wire[j][0]})" for j in sorted(js)))
            if cof:
                line += ("  +co-frozen: "
                         + ", ".join(NAMES[j] for j in sorted(cof)))
            print(line)
        if alarmed:
            print(f"   (still dropped at exit: "
                  + ", ".join(NAMES[j] for j in sorted(alarmed)) + ")")


if __name__ == "__main__":
    main()
