# python humanoid_arm_forensics.py
"""Record everything the RIGHT ARM does, for post-incident replay.

Built 07-24 for the recurring "right arm drives into its own body during the
carry staircase" incident. Every previous session lost the evidence: the
operator log prints state transitions but not one number — no commanded angle,
no measured angle, no torque, no CAN health — so each post-mortem was an
argument about mechanisms instead of a look at the data.

WHAT IT RECORDS (read-only; two Pub0/Sub0 subscriptions, nothing is published)

  9870  humanoid_real_env telemetry, per control tick:
        joint_pos          measured angle          (motor.mech_pos)
        joint_vel          measured velocity
        joint_effort       MEASURED torque
        joint_effort_ref   COMMANDED torque reference
        motor_temp         per-motor winding temperature [degC]
        ee.right           measured EE pose in base frame
        projected_gravity  torso attitude (tilt watch)
  9874  auto-operator arm command, per packet:
        arm_targets.right  either {joint_pos, rate}  (JOINT staircase)
                           or     {ee_pos, ee_quat}  (CARTESIAN reach/hold)
  CAN   error counters for every bus, polled at --can-hz:
        bus-error / error-warning / error-passive / bus-off / restarts,
        plus rx/tx byte+packet counts and drops.

WHY CAN21 GETS ITS OWN TREATMENT
  humanoid_config.py motor_setup_dict puts SIX of the right arm's seven joints
  on can21 (right_shoulder_2 .. right_wrist_3); only right_shoulder_1 lives on
  can22, shared with the waist and left_shoulder_1. So can21 is a single point
  of failure for essentially the whole right arm, and a bus that goes
  error-passive or drops frames shows up as commands not arriving / feedback
  going stale — which looks exactly like a mechanical jam from the outside.
  The summary prints per-bus deltas so a bus event during the incident window
  is impossible to miss.

THE ONE NUMBER THAT IS NOT ON THE WIRE
  real_env's `current_target_arm_pos` — the instantaneous, rate-limited target
  actually fed to the motors — is internal and never published. It matters
  because in JOINT mode the operator sends only the END posture and real_env
  shapes the motion itself (humanoid_real_env.py ~1019):

      step = rate * dt                       # SAME scalar for all 7 joints
      cur += clamp(desired - cur, -step, +step)

  That is a PER-JOINT box clamp: joints with a small delta arrive first, joints
  with a large delta arrive last, so the executed joint-space path is an
  L-shaped box path, NOT the straight chord between the two postures. The
  offline staircase audit sampled the chord. This recorder therefore
  RECONSTRUCTS the box-clamped target offline (--replay) from the published
  endpoint + rate + dt, so the commanded path, the executed path and the chord
  can be compared directly.

USAGE
    python humanoid_arm_forensics.py                       # record until Ctrl+C
    python humanoid_arm_forensics.py --robot-ip 127.0.0.1
    python humanoid_arm_forensics.py --arm left            # watch the other arm
    python humanoid_arm_forensics.py --quiet               # no live console
    python humanoid_arm_forensics.py --replay <file.npz>   # post-mortem summary

Start it BEFORE the operator and leave it running; it is a passive subscriber,
so starting or stopping it never perturbs the robot. On Ctrl+C it writes
    <recordings>/armfx_<MMDD_HHMMSS>.npz
and prints the incident summary.

DESIGN NOTES
  - Samples on TELEMETRY arrival (data_id change), not on a fixed clock: the
    record then has exactly one row per control tick with no interpolation, and
    a stalled telemetry stream is visible as a gap rather than as duplicated
    rows.
  - The arm command is latched, not sampled: 9874 publishes only when the
    operator has something to say, so each telemetry row carries the most
    recent command and the age of that command. A command that stops arriving
    is itself a finding (real_env holds its last target after 0.5 s of silence).
  - Everything is stored raw. No thresholding, no filtering, no downsampling —
    the whole point is that the last incident was diagnosed from prose because
    the numbers were gone.
"""
from __future__ import annotations

import argparse
import humanoid_site as _site
import datetime as _dt
import os
import re
import subprocess
import time

import numpy as np

from ipc.publisher import NNGSubscriber

# URDF index of each arm's 7 joints (humanoid_config.py motor_setup_dict order).
ARM_SLICE = {"left": slice(13, 20), "right": slice(20, 27)}
JOINT_NAMES = {
    "right": ["shoulder_1", "shoulder_2", "shoulder_3", "elbow",
              "wrist_1", "wrist_2", "wrist_3"],
}
JOINT_NAMES["left"] = list(JOINT_NAMES["right"])
# Which CAN bus each arm joint sits on — see the module docstring.
ARM_BUS = {
    "right": ["can22", "can21", "can21", "can21", "can21", "can21", "can21"],
    "left":  ["can22", "can9",  "can9",  "can9",  "can9",  "can9",  "can9"],
}
CAN_BUSES = ["can9", "can21", "can22", "can23", "can24", "can25"]
DT_NOMINAL = 0.02          # real_env control period [s]; only used for replay


# The six SocketCAN controller counters, in the fixed order the kernel prints
# them on the header line directly above their values.
_CAN_COUNTERS = ("re-started", "bus-errors", "arbit-lost",
                 "error-warn", "error-pass", "bus-off")


def _can_stats(bus: str) -> dict:
    """Error counters + traffic for one CAN interface, or {} if unavailable.

    Parsed from `ip -s -d link show <bus>`, which prints

          re-started bus-errors arbit-lost error-warn error-pass bus-off
          0          0          0          0          0          0        <trailing junk>
        RX:  bytes packets errors dropped  missed   mcast
          25773104 3221638      0       0       0       0
        TX:  bytes packets errors dropped carrier collsns
          12891848 1611481      0       0       0       0

    Anchoring on the header lines matters: a naive "first row of six integers"
    scan lands on the COUNTER row and mislabels it as RX (which is how an
    earlier version reported rx_pkts=0 on a bus carrying 3.2 M frames).
    Returns plain ints so the record packs into the npz without object arrays.
    """
    try:
        out = subprocess.run(["ip", "-s", "-d", "link", "show", bus],
                             capture_output=True, text=True, timeout=1.0).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    if not out:
        return {}
    st = {}
    m = re.search(r"can state (\S+)", out)
    st["state"] = m.group(1) if m else "?"

    m = re.search(r"re-started\s+bus-errors.*?\n\s*" + r"(\d+)\s+" * 5 + r"(\d+)",
                  out, re.S)
    if m:
        st.update(dict(zip(_CAN_COUNTERS, (int(v) for v in m.groups()))))

    for tag, pfx in (("RX", "rx"), ("TX", "tx")):
        m = re.search(tag + r":.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", out, re.S)
        if m:
            st[f"{pfx}_pkts"] = int(m.group(2))
            st[f"{pfx}_err"] = int(m.group(3))
            st[f"{pfx}_drop"] = int(m.group(4))
    return st


def _arm_cmd(packet: dict | None, arm: str):
    """Extract (mode, 7-vector or 3-vector, rate) from a 9874 arm packet.

    Returns (mode, values, rate) where mode is "joint" | "cart" | "none".
    Handles BOTH wire schemas: the per-arm dict {left:..., right:...} used since
    the 07-20 protocol change, and the legacy two-slot {ee_pos:[l,r], ...}.
    """
    if not packet:
        return "none", None, 0.0
    at = packet.get("arm_targets")
    if not isinstance(at, dict):
        return "none", None, 0.0
    slot = at.get(arm)
    if isinstance(slot, dict):
        if "joint_pos" in slot:
            return "joint", np.asarray(slot["joint_pos"], float), \
                   float(slot.get("rate", 0.0))
        if "ee_pos" in slot:
            return "cart", np.asarray(slot["ee_pos"], float), 0.0
    if "ee_pos" in at:                      # legacy two-slot schema
        i = 0 if arm == "left" else 1
        v = at["ee_pos"][i]
        if v is not None:
            return "cart", np.asarray(v, float), 0.0
    return "none", None, 0.0


def record(args) -> None:
    arm = args.arm
    sl = ARM_SLICE[arm]
    names = JOINT_NAMES[arm]

    tel = NNGSubscriber(f"tcp://{args.robot_ip}:9870"); tel.start()
    cmd = NNGSubscriber(f"tcp://{args.robot_ip}:9874"); cmd.start()
    print(f"[fx] subscribing telemetry :9870 and arm command :9874 at "
          f"{args.robot_ip} — watching the {arm.upper()} arm")
    print(f"[fx] {arm} arm CAN buses: " +
          ", ".join(f"{n}={b}" for n, b in zip(names, ARM_BUS[arm])))

    can0 = {b: _can_stats(b) for b in CAN_BUSES}
    for b in CAN_BUSES:
        s = can0.get(b) or {}
        if s:
            print(f"[fx] {b}: state={s.get('state','?')} "
                  f"bus-errors={s.get('bus-errors','?')} "
                  f"rx {s.get('rx_pkts','?')}/{s.get('rx_err','?')}err "
                  f"tx {s.get('tx_pkts','?')}/{s.get('tx_err','?')}err")
    print("[fx] recording — Ctrl+C to stop and write the file")

    rows: list[dict] = []
    can_log: list[tuple] = []
    last_tel_id = -1
    last_cmd_id = -1
    last_cmd_t = -1e9
    t0 = time.time()
    last_can = 0.0
    last_print = 0.0
    last_mode = None
    peak_err = 0.0
    bus_names = None

    try:
        while True:
            now = time.time()

            # ---- CAN poll (cheap, low rate; subprocess so keep it off the hot path)
            if now - last_can >= 1.0 / max(args.can_hz, 1e-6):
                last_can = now
                snap = {b: _can_stats(b) for b in CAN_BUSES}
                can_log.append((now - t0, snap))
                for b in CAN_BUSES:
                    a, c = can0.get(b) or {}, snap.get(b) or {}
                    if not a or not c:
                        continue
                    for k in _CAN_COUNTERS + ("rx_err", "tx_err",
                                              "rx_drop", "tx_drop"):
                        if c.get(k, 0) > a.get(k, 0):
                            print(f"[fx] !! CAN {b} {k}: "
                                  f"{a.get(k,0)} -> {c.get(k,0)}"
                                  + ("   <-- RIGHT ARM BUS"
                                     if b in ARM_BUS[arm] else ""))
                    if c.get("state") != a.get("state"):
                        print(f"[fx] !! CAN {b} state {a.get('state')} -> "
                              f"{c.get('state')}")
                can0 = snap

            # ---- arm command (latched: 9874 only speaks when it has something)
            if cmd.data_id != last_cmd_id and cmd.data is not None:
                last_cmd_id = cmd.data_id
                last_cmd_t = now

            # ---- telemetry (one row per control tick)
            if tel.data_id == last_tel_id or tel.data is None:
                time.sleep(0.002)
                continue
            last_tel_id = tel.data_id
            t = tel.data

            jp = np.asarray(t.get("joint_pos") or [], float)
            if jp.size < 27:
                continue
            mode, cval, rate = _arm_cmd(cmd.data, arm)

            def seg(key):
                v = np.asarray(t.get(key) or [], float)
                return v[sl] if v.size >= 27 else np.full(7, np.nan)

            ee = ((t.get("ee") or {}).get(arm) or {})

            def vec(v, n):
                a = np.asarray(v if v is not None else [np.nan] * n, float)
                return a if a.size == n else np.full(n, np.nan)

            # arm_integral covers BOTH arms in arm_motor_indices order
            # (URDF order, left 7 then right 7) — take this arm's half.
            ai = np.asarray(t.get("arm_integral") or [], float)
            ai = (ai[0:7] if arm == "left" else ai[7:14]) if ai.size >= 14 \
                else np.full(7, np.nan)
            row = {
                "t": now - t0,
                "pos": jp[sl].copy(),
                "vel": seg("joint_vel"),
                "eff": seg("joint_effort"),
                "eff_ref": seg("joint_effort_ref"),
                "temp": seg("motor_temp"),
                # FORENSICS PATCH fields — absent on an unpatched real_env, in
                # which case they stay NaN and the summary says so.
                "pos_ref": seg("mech_pos_ref"),
                "tau_ref": seg("mech_torque_ref"),
                "integ": ai,
                "ee_act": vec(ee.get("pos_actual"), 3),
                "ee_tgt": vec(ee.get("pos_target"), 3),
                "ee_quat": vec(ee.get("quat_actual"), 4),
                "grav": vec(t.get("projected_gravity"), 3),
                # Raw policy output. The arm reference gets
                # `action * policy_action_scale` added EVERY step; 07-20 listed
                # this as the one unmeasured term in the whole chain.
                "action": vec(t.get("actions"), 31)[:31],
                "servo_temp": float(ee.get("servo_temp") or np.nan),
                "servo_load": float(ee.get("servo_load") or np.nan),
                "bus_q": vec(t.get("bus_queue"), 6),
                # real_env's OWN view of the wire mode, to compare against what
                # the operator published — disagreement is a protocol fault.
                "rx_cart": float((t.get("arm_cart") or [np.nan, np.nan])
                                 [0 if arm == "left" else 1]),
                "rx_stream": float(t.get("arm_stream_active") or 0),
                "cmd_mode": {"none": 0, "joint": 1, "cart": 2}[mode],
                "cmd_joint": (cval if mode == "joint"
                              else np.full(7, np.nan)),
                "cmd_ee": (cval if mode == "cart" else np.full(3, np.nan)),
                "cmd_rate": rate,
                "cmd_age": now - last_cmd_t,
                # 07-25b firmware-side state (needs rebuilt bindings + patched
                # real_env; NaN on older publishers). mode_status 0=Reset
                # 1=Calib 2=Running 255=Unknown; error_code fault bits.
                "mode_status": seg("mode_status"),
                "error_code": seg("error_code"),
            }
            rows.append(row)
            if bus_names is None:
                bn = t.get("bus_names")
                if bn:
                    bus_names = [str(x) for x in bn]
                    print(f"[fx] real_env bus order: "
                          + ", ".join(f"{i}={n}" for i, n in enumerate(bus_names)))

            # ---- live console -------------------------------------------------
            if mode != last_mode:
                last_mode = mode
                print(f"[fx] {row['t']:7.2f}s  arm wire -> {mode.upper()}"
                      + (f"  rate={rate:.2f} rad/s" if mode == "joint" else ""))
            if mode == "joint" and cval is not None:
                err = float(np.max(np.abs(cval - row["pos"])))
                peak_err = max(peak_err, err)
            if not args.quiet and now - last_print >= 1.0:
                last_print = now
                eff = row["eff"]
                hot = np.nanmax(row["temp"]) if row["temp"].size else np.nan
                worst = int(np.nanargmax(np.abs(eff))) if np.any(
                    np.isfinite(eff)) else 0
                extra = ""
                if mode == "joint" and cval is not None:
                    k = int(np.nanargmax(np.abs(cval - row["pos"])))
                    extra = (f"  trackerr max {np.degrees(np.nanmax(np.abs(cval - row['pos']))):5.1f}deg"
                             f" @{names[k]}")
                print(f"[fx] {row['t']:7.2f}s {mode:5s} "
                      f"|tau|max {np.nanmax(np.abs(eff)):6.2f} @{names[worst]:10s} "
                      f"Tmax {hot:4.1f}C{extra}")
    except KeyboardInterrupt:
        print("\n[fx] stopping")
    finally:
        tel.stop(); cmd.stop()
        if not rows:
            print("[fx] no telemetry received — is real_env up and is "
                  "--robot-ip right?")
            return
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(
            args.out_dir,
            f"armfx_{_dt.datetime.now().strftime('%m%d_%H%M%S')}_{arm}.npz")
        packed = {k: np.stack([r[k] for r in rows])
                  for k in ("pos", "vel", "eff", "eff_ref", "temp",
                            "pos_ref", "tau_ref", "integ", "action", "bus_q",
                            "ee_act", "ee_tgt", "ee_quat", "grav",
                            "cmd_joint", "cmd_ee",
                            "mode_status", "error_code")}
        packed["t"] = np.array([r["t"] for r in rows])
        packed["cmd_mode"] = np.array([r["cmd_mode"] for r in rows])
        packed["cmd_rate"] = np.array([r["cmd_rate"] for r in rows])
        packed["cmd_age"] = np.array([r["cmd_age"] for r in rows])
        for k in ("servo_temp", "servo_load", "rx_cart", "rx_stream"):
            packed[k] = np.array([r[k] for r in rows])
        packed["bus_names"] = np.array(bus_names if bus_names else [])
        packed["arm"] = np.array(arm)
        packed["names"] = np.array(names)
        packed["buses"] = np.array(ARM_BUS[arm])
        packed["can_t"] = np.array([c[0] for c in can_log])
        # CAN snapshots go in as one JSON blob: shapes are irregular and this
        # keeps the npz free of object arrays.
        import json
        packed["can_json"] = np.array(json.dumps(
            [[c[0], c[1]] for c in can_log], default=str))
        np.savez_compressed(path, **packed)
        print(f"[fx] {len(rows)} ticks over {rows[-1]['t']:.1f}s -> {path}")
        _summary(path)


def _summary(path: str) -> None:
    """Post-mortem: the numbers that decide between the competing mechanisms."""
    d = np.load(path, allow_pickle=False)
    names = [str(x) for x in d["names"]]
    buses = [str(x) for x in d["buses"]]
    t, pos, eff = d["t"], d["pos"], d["eff"]
    mode, cj, rate = d["cmd_mode"], d["cmd_joint"], d["cmd_rate"]
    print("\n" + "=" * 68)
    print(f"  ARM FORENSICS — {str(d['arm'])} arm, {len(t)} ticks, {t[-1]:.1f}s")
    print("=" * 68)

    # 1. wire mode timeline — JOINT segments are the unprotected ones (mink's
    #    CollisionAvoidanceLimit only exists inside the Cartesian solve).
    lab = {0: "silent", 1: "JOINT", 2: "cart"}
    segs, cur, start = [], int(mode[0]), t[0]
    for i in range(1, len(mode)):
        if int(mode[i]) != cur:
            segs.append((cur, start, t[i])); cur, start = int(mode[i]), t[i]
    segs.append((cur, start, t[-1]))
    print("\n-- arm wire mode (JOINT segments have NO collision constraint) --")
    for m, a, b in segs:
        if b - a < 0.05:
            continue
        print(f"   {a:7.2f} - {b:7.2f}s  ({b-a:5.2f}s)  {lab[m]}")

    # 2. per-joint peaks
    print("\n-- per joint (whole record) --")
    print("   joint         bus     |tau|max  tau@peak_t   Tmax   range[deg]")
    for j, n in enumerate(names):
        e = eff[:, j]
        k = int(np.nanargmax(np.abs(e))) if np.any(np.isfinite(e)) else 0
        rng = np.degrees(np.nanmax(pos[:, j]) - np.nanmin(pos[:, j]))
        print(f"   {n:12s} {buses[j]:7s} {np.nanmax(np.abs(e)):7.2f}  "
              f"{t[k]:9.2f}s  {np.nanmax(d['temp'][:, j]):5.1f}  {rng:7.1f}")

    # 2b. THE key series once real_env carries the forensics patch: the FINAL
    #     commanded angle (box clamp + policy residual + integrator + limit
    #     clamp) against the measured one. This is the only tracking error that
    #     means anything — the operator's endpoint is not what the motors chase.
    pref = d["pos_ref"] if "pos_ref" in d else None
    if pref is not None and np.any(np.isfinite(pref)):
        terr = np.abs(pref - pos)
        print("\n-- FINAL command (mech_pos_ref) vs measured --")
        for j, n in enumerate(names):
            e = terr[:, j]
            if not np.any(np.isfinite(e)):
                continue
            k = int(np.nanargmax(e))
            print(f"   {n:12s} max {np.degrees(np.nanmax(e)):6.1f} deg "
                  f"@ {t[k]:7.2f}s   (mean {np.degrees(np.nanmean(e)):4.1f})")
        k = int(np.nanargmax(np.nanmax(terr, axis=1)))
        print(f"   WORST TICK t={t[k]:.2f}s: "
              + ", ".join(f"{n}={np.degrees(terr[k, j]):.1f}"
                          for j, n in enumerate(names)))
        # A joint whose error grows while its neighbours shrink is being held
        # back — that is the jam signature, and it is per-joint specific.
    else:
        print("\n-- mech_pos_ref NOT PRESENT: real_env is running without the "
              "07-24 forensics patch, so the true commanded angle is unknown --")

    # 2c. Policy residual on this arm — the term 07-20 flagged as unmeasured.
    if "action" in d and np.any(np.isfinite(d["action"])):
        act = d["action"]
        print(f"\n-- policy action (raw, {act.shape[1]} dims) --")
        print(f"   |action|max over record: {np.nanmax(np.abs(act)):.3f}"
              f"   (scaled by policy_action_scale before it is ADDED to the "
              f"arm reference every step)")

    # 2c2. Firmware-side state (07-25b): mode transitions + fault bits. This
    # is the direct discriminator between "off-bus" (feedback freezes, mode
    # stays stale at its last value) and "on-bus but self-protected" (mode
    # drops 2->0 while feedback keeps updating).
    if "mode_status" in d and np.any(np.isfinite(d["mode_status"])):
        ms, ec = d["mode_status"], d["error_code"]
        print("\n-- firmware mode/error (mode 0=Reset 1=Calib 2=Running) --")
        any_evt = False
        for j, n in enumerate(names):
            col = ms[:, j]
            ch = np.nonzero(np.diff(col) != 0)[0]
            for k in ch[:10]:
                any_evt = True
                print(f"   {n:12s} t={t[k+1]:7.2f}s  mode {int(col[k])} -> "
                      f"{int(col[k+1])}")
            bad = np.nonzero(ec[:, j] != 0)[0]
            if bad.size:
                any_evt = True
                codes = sorted(set(int(v) for v in ec[bad, j]))
                print(f"   {n:12s} error_code nonzero on {bad.size} ticks "
                      f"(t={t[bad[0]]:.2f}s..{t[bad[-1]]:.2f}s) codes={codes} "
                      f"(1=UV 2=OC 4=OT 8=mag 16=hall 32=uncal)")
        if not any_evt:
            print("   all Running, zero faults for the whole record")

    # 2d. CAN send backlog — the direct congestion measure, per bus.
    if "bus_q" in d and np.any(np.isfinite(d["bus_q"])):
        bq = d["bus_q"]
        bnames = [str(x) for x in d["bus_names"]] if "bus_names" in d \
            and d["bus_names"].size else [f"bus{i}" for i in range(bq.shape[1])]
        print("\n-- CAN send-queue depth (non-zero = frames backing up) --")
        for i, bn in enumerate(bnames):
            if i >= bq.shape[1]:
                break
            col = bq[:, i]
            if not np.any(np.isfinite(col)):
                continue
            tag = "  <-- ARM BUS" if bn in buses else ""
            k = int(np.nanargmax(col))
            print(f"   {bn:7s} max {np.nanmax(col):5.0f} @ {t[k]:7.2f}s   "
                  f"mean {np.nanmean(col):5.2f}{tag}")

    # 2e. Wire-mode agreement: what the operator sent vs what real_env believes.
    if "rx_cart" in d and np.any(np.isfinite(d["rx_cart"])):
        op_joint = mode == 1
        rx_joint = (d["rx_cart"] == 0) & (d["rx_stream"] > 0)
        dis = op_joint & ~rx_joint
        if dis.any():
            k = int(np.flatnonzero(dis)[0])
            print(f"\n-- WIRE MODE DISAGREEMENT on {int(dis.sum())} ticks, "
                  f"first t={t[k]:.2f}s: operator sent JOINT but real_env did "
                  f"not register a joint stream for this arm")

    # 3. tracking error inside JOINT segments. A stall shows here first: the
    #    command keeps advancing, the measured angle stops.
    jm = mode == 1
    if jm.any() and np.any(np.isfinite(cj[jm])):
        err = np.abs(cj - pos)
        err[~jm] = np.nan
        print("\n-- JOINT-mode tracking error (command minus measured) --")
        for j, n in enumerate(names):
            e = err[:, j]
            if not np.any(np.isfinite(e)):
                continue
            k = int(np.nanargmax(e))
            print(f"   {n:12s} max {np.degrees(np.nanmax(e)):6.1f} deg "
                  f"@ {t[k]:7.2f}s")
        # Cartesian ticks are masked to NaN, so rows can be all-NaN; reduce with
        # a sentinel instead of nanmax to keep the warning out of the report.
        rowmax = np.where(np.isfinite(err), err, -np.inf).max(axis=1)
        if np.any(np.isfinite(rowmax)):
            k = int(np.argmax(rowmax))
            print(f"   WORST OVERALL at t={t[k]:.2f}s: "
                  + ", ".join(f"{n}={np.degrees(err[k, j]):.1f}"
                              for j, n in enumerate(names)))
        print("   (a joint whose error GROWS while the others shrink is being "
              "held back — jam, or the box clamp waiting on it)")

    # 4. stall detector: high torque + near-zero velocity is a hard contact.
    vel = d["vel"]
    stalled = (np.abs(eff) > np.nanpercentile(np.abs(eff), 90)) & \
              (np.abs(vel) < 0.02)
    if stalled.any():
        print("\n-- STALL SUSPECTS (top-decile torque with |vel| < 0.02) --")
        for j, n in enumerate(names):
            idx = np.flatnonzero(stalled[:, j])
            if idx.size < 5:
                continue
            print(f"   {n:12s} {idx.size:5d} ticks, first {t[idx[0]]:7.2f}s, "
                  f"|tau| up to {np.nanmax(np.abs(eff[idx, j])):.2f}")

    # 5. command silence — real_env holds its last target after 0.5 s of it.
    # Ignore ticks before the FIRST command ever arrived: cmd_age is seeded from
    # a -1e9 sentinel there and would report an epoch-sized gap.
    age = np.where(d["cmd_age"] < 1e6, d["cmd_age"], np.nan)
    if np.any(np.isfinite(age)) and np.nanmax(age) > 0.5:
        k = int(np.nanargmax(age))
        print(f"\n-- COMMAND SILENCE: up to {np.nanmax(age):.2f}s "
              f"@ t={t[k]:.2f}s (real_env holds its last target past 0.5s)")

    # 6. CAN deltas across the whole record.
    import json
    can = json.loads(str(d["can_json"])) if "can_json" in d else []
    if len(can) >= 2:
        first, last = can[0][1], can[-1][1]
        print("\n-- CAN counters (start -> end) --")
        for b in sorted(set(list(first) + list(last))):
            a, c = first.get(b) or {}, last.get(b) or {}
            ch = [f"{k} {a.get(k,0)}->{c.get(k,0)}"
                  for k in _CAN_COUNTERS + ("rx_err", "tx_err",
                                            "rx_drop", "tx_drop")
                  if c.get(k, 0) != a.get(k, 0)]
            tag = "  <-- ARM BUS" if b in buses else ""
            print(f"   {b:7s} state {a.get('state','?')} -> {c.get('state','?')}"
                  + ("   " + "; ".join(ch) if ch else "   (no counter change)")
                  + tag)

    # 7. torso attitude — a tilted base corrupts every body-frame target.
    g = d["grav"]
    if np.any(np.isfinite(g)):
        tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
        print(f"\n-- torso tilt: {np.nanmin(tilt):.1f} - {np.nanmax(tilt):.1f} deg "
              f"(RULE #0 wants this small and steady)")
    print("=" * 68)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", default="127.0.0.1")
    p.add_argument("--arm", default="right", choices=["left", "right"])
    p.add_argument("--out-dir", default=str(_site.RECORDINGS_DIR))
    p.add_argument("--can-hz", type=float, default=2.0,
                   help="CAN counter poll rate [Hz]")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the 1 Hz live line")
    p.add_argument("--replay", default=None,
                   help="print the summary for an existing .npz and exit")
    args = p.parse_args()
    if args.replay:
        _summary(args.replay)
        return
    record(args)


if __name__ == "__main__":
    main()
