"""Stage 1: one commanded velocity step, measured. The first hardware walk.

WHY A SCRIPT AND NOT THE GAMEPAD. Two things have to come out of this run and a
stick gives neither cleanly:

  * HOW FAR THE BASE COASTS after the command drops to zero. JOURNEY_ARRIVE_M
    was raised 0.34 -> 0.50 on 2026-08-07 to pay for the higher NEAR_VX floor,
    and the size of that payment was ESTIMATED, never measured ("~5-8 cm at
    0.1 m/s" in the constant's own comment, written when the floor was 0.1).
    A repeatable step with a known magnitude and a printed stop instant is what
    turns that into a number.
  * WHAT THE POST-STOP YAW ACTUALLY LOOKS LIKE against JOURNEY_STILL_ANG =
    0.10 rad/s. The 08-07 audit changed journey_base_still to DEBIT its sustain
    window rather than null it, on the argument that raw gyro outliers are rare
    and sustained motion is not. That argument is untested. This records the
    trace so it can be checked instead of believed.

A stick cannot hold 0.20 m/s steady, cannot stamp the release instant, and — on
this robot — pushing it past 0.2 of scale latches real_env's `_nav_paused`,
after which the nav socket this script publishes on is IGNORED until LB. Which
is exactly the trap the audit's fix #1 exists to name.

SAFETY, in the order the layers fire:
  1. Ctrl+C: publishes zeros for ZERO_TAIL_S, then exits. Normal stop.
  2. Process dies: real_env's _NAV_CMD_TIMEOUT (1.0 s) zeroes the command.
  3. Gamepad: a large stick deflection seizes manual control instantly (and
     latches _nav_paused — press LB before any later autonomous run). The A
     button is [SHUTDOWN].
  4. The physical E-stop.

This publishes the SAME message on the SAME socket as the journey mission
(`{"nav_cmd": [vx, vy, wz]}` on 9873, base FLU frame), unramped, so what it
measures is what the mission will do. It never touches the arms.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time

from ipc.publisher import NNGPublisher, NNGSubscriber

NAV_URL = "tcp://*:9873"       # mirrors humanoid_site.NAV_COMMAND_PORT (kept literal here)
TELEMETRY_PORT = 9870          # mirrors humanoid_site.TELEMETRY_PORT (kept literal here)
PUB_HZ = 20.0                  # well inside real_env's 1.0 s nav dead-man
PUB_DT = 1.0 / PUB_HZ
SETTLE_S = 3.0                 # zeros first: prove the link before commanding
ZERO_TAIL_S = 6.0              # keep COMMANDING zero after the step, so the
                               # base stops under command rather than under the
                               # dead-man — the coast we want to measure is the
                               # policy's, not the failsafe's
STILL_ANG = 0.10               # JOURNEY_STILL_ANG, the bar being validated
STILL_VEL = 0.05               # JOURNEY_STILL_VEL
UNSTABLE_WZ = 1.0              # rad/s: above this peak the gait is degenerate,
                               # not merely slow. Measured 08-07: clean walks
                               # peaked 0.692 (fwd 0.30) and 0.469 (rev 0.20);
                               # the near-fall at 0.25 peaked 1.670.


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vx", type=float, default=0.2,
                   help="commanded forward velocity, m/s (negative = reverse)")
    p.add_argument("--wz", type=float, default=0.0,
                   help="commanded yaw rate, rad/s. NONZERO = turn-in-place "
                        "mode: vx forced to 0, command [0, 0, wz]. BENCH "
                        "HARNESS ONLY — mission use of in-place rotation is "
                        "FORBIDDEN (upstream ruling 2026-08-08); this measured why: "
                        "0.2 winds the torso, 0.4 near-falls. Kept for "
                        "probing future checkpoints. Positive = CCW.")
    p.add_argument("--seconds", type=float, default=4.0,
                   help="how long to hold it")
    p.add_argument("--robot-ip", default="127.0.0.1")
    p.add_argument("--cube", default=None,
                   help="a STATIONARY tagged body (e.g. grasp_cube_60mm_a) to "
                        "use as a landmark. Its body-frame position moves by "
                        "exactly as much as the base does, which is the only "
                        "odometry this robot has. Needs the monitor running.")
    p.add_argument("--detection-url", default="ipc:///tmp/humanoid_detections.sock")
    a = p.parse_args()

    arc_mode = abs(a.wz) > 1e-9 and abs(a.vx) > 1e-9
    turn_mode = abs(a.wz) > 1e-9 and not arc_mode
    if abs(a.vx) > 0.45:
        print(f"[nav] refusing vx={a.vx}: this is a first-walk harness; "
              f"CRUISE_VX is 0.4 and 0.45 is the probe ceiling")
        return 2
    if arc_mode:
        # The mission's unfaced-arc primitive (sim parity 08-08): vx + wz
        # TOGETHER. The sim measured yaw tracking is BETTER at cruise than in
        # place, so the ceiling is the arc law's own cap, not the turn cap.
        if abs(a.wz) > 0.60:
            print(f"[nav] refusing wz={a.wz} in arc mode: "
                  f"JOURNEY_WZ_UNFACED_MAX is 0.5; 0.60 is the probe ceiling")
            return 2
    elif turn_mode:
        a.vx = 0.0
        if abs(a.wz) > 0.40:
            print(f"[nav] refusing wz={a.wz}: in-place turning is BANNED for "
                  f"missions (upstream ruling 08-08) and 0.40 already near-fell; this "
                  f"mode exists only to probe future checkpoints")
            return 2

    tel = NNGSubscriber(f"tcp://{a.robot_ip}:{TELEMETRY_PORT}")
    tel.start()
    det = None
    if a.cube:
        det = NNGSubscriber(a.detection_url)
        det.start()
    pub = NNGPublisher(NAV_URL)
    time.sleep(0.3)                       # let real_env's dialer attach

    def landmark():
        """The cube's CURRENT body-frame position, or None.

        THE ONLY ODOMETRY THIS ROBOT HAS. base_lin_vel is hard zeros on this
        checkpoint, so nothing in telemetry says how far the base travelled.
        But a cube bolted to a table does not move, and every detection is in
        the BODY frame — so the cube's apparent motion is the base's real
        motion, negated. Same measurement the journey's own arrival test runs
        on, which is the point: if this disagrees with the tape measure, the
        mission's distance sense is what is wrong."""
        if det is None:
            return None
        t = ((det.data or {}).get("targets") or {}).get(a.cube)
        if not t or t.get("n_inliers", 0) < 1:
            return None
        p = t.get("pos")
        return (float(p[0]), float(p[1])) if p and len(p) >= 2 else None

    print(f"[nav] publishing on {NAV_URL} at {PUB_HZ:.0f} Hz")
    if arc_mode:
        r_arc = abs(a.vx / a.wz)
        print(f"[nav] plan: {SETTLE_S:.0f}s zero -> {a.seconds:.1f}s at "
              f"vx={a.vx:+.2f}, wz={a.wz:+.2f} (ARC, radius ~{r_arc:.2f} m) "
              f"-> {ZERO_TAIL_S:.0f}s zero")
        print(f"[nav] expected if tracked 1:1: yaw "
              f"{math.degrees(a.wz * a.seconds):+.0f} deg along a "
              f"{abs(a.vx) * a.seconds:.2f} m curve. CLEAR THE ARC — it "
              f"sweeps to the {'left' if a.wz > 0 else 'right'}.")
    elif turn_mode:
        print(f"[nav] plan: {SETTLE_S:.0f}s zero -> {a.seconds:.1f}s at "
              f"wz={a.wz:+.2f} rad/s (TURN IN PLACE) -> {ZERO_TAIL_S:.0f}s zero")
        print(f"[nav] expected yaw if tracked 1:1: "
              f"{math.degrees(a.wz * a.seconds):+.0f} deg. Watch the feet: "
              f"scrub (the base translating while turning) is what the "
              f"landmark report measures.")
    else:
        print(f"[nav] plan: {SETTLE_S:.0f}s zero -> {a.seconds:.1f}s at "
              f"vx={a.vx:+.2f} -> {ZERO_TAIL_S:.0f}s zero")
        print(f"[nav] Ctrl+C stops. Mark the floor at the START and at the "
              f"STOP instant; the coast is the distance travelled AFTER the "
              f"stop line.")

    trace: list = []                     # (t_rel_to_stop, |wz|, |v_xy|, cmd_on)
    marks: dict = {}                     # phase -> landmark xy
    paused_seen = False
    t0 = time.monotonic()
    t_step = t0 + SETTLE_S
    t_stop = t_step + a.seconds
    t_end = t_stop + ZERO_TAIL_S
    said = -1e9
    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                break
            on = t_step <= now < t_stop
            vx = a.vx if (on and not turn_mode) else 0.0
            wz_cmd = a.wz if (on and (turn_mode or arc_mode)) else 0.0
            pub.publish({"nav_cmd": [float(vx), 0.0, float(wz_cmd)]})

            d = tel.data or {}
            ang = d.get("base_ang_vel") or [0.0, 0.0, 0.0]
            lin = d.get("base_lin_vel") or [0.0, 0.0, 0.0]
            # SIGNED: the turn report integrates this into achieved yaw; the
            # walk report takes abs() where it always did.
            wz = float(ang[2]) if len(ang) >= 3 else 0.0
            vxy = (float(lin[0]) ** 2 + float(lin[1]) ** 2) ** 0.5 \
                if len(lin) >= 2 else 0.0
            trace.append((now - t_stop, wz, vxy, on))

            # THE TRAP THIS RUN EXISTS PARTLY TO EXPOSE. real_env drops every
            # nav_cmd while _nav_paused is latched, and LB is the only reset.
            if d.get("nav_paused") and not paused_seen:
                paused_seen = True
                print("[nav] *** NAV PAUSED — real_env is IGNORING this "
                      "script. Press LB on the gamepad. ***")

            if now >= t_step and now < t_stop and now - said > 1.0:
                said = now
                print(f"[nav] t={now - t_step:+.1f}s  cmd vx={vx:+.2f} "
                      f"wz={wz_cmd:+.2f}  measured |v_xy|={vxy:.3f}  "
                      f"wz={wz:+.3f}")
            if "start" not in marks and now >= t_step:
                lm = landmark()
                if lm:
                    marks["start"] = lm
            # `not on` and not `vx == 0.0`: in turn mode vx is always zero, so
            # the old guard admitted the last still-commanded tick — the STOP
            # banner printed twice and the landmark could stamp ~50 ms early
            # (review 08-08).
            if abs(now - t_stop) < PUB_DT and not on:
                lm = landmark()
                if lm:
                    marks["stop"] = lm
                print(f"\n[nav] ===== STOP COMMANDED (t=0) — mark the floor "
                      f"NOW =====\n")
            if now > t_stop + 1.0:
                lm = landmark()
                if lm:
                    marks["rest"] = lm            # keeps updating to the last
            time.sleep(max(0.0, PUB_DT - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("\n[nav] Ctrl+C — commanding zero")
    finally:
        for _ in range(int(PUB_HZ * 1.5)):
            pub.publish({"nav_cmd": [0.0, 0.0, 0.0]})
            time.sleep(PUB_DT)

    if arc_mode:
        # An arc is graded on BOTH channels: the turn report for yaw tracking,
        # the travel report for the path length the curve actually covered.
        _report_turn(trace, marks, a.wz, a.seconds, paused_seen)
        _report_travel(marks, a.vx, a.seconds)
    elif turn_mode:
        _report_turn(trace, marks, a.wz, a.seconds, paused_seen)
    else:
        _report(trace, a.vx, paused_seen)
        _report_travel(marks, a.vx, a.seconds)
    tel.stop()
    if det is not None:
        det.stop()
    return 0


def _report_travel(marks: dict, vx: float, seconds: float) -> None:
    """Commanded distance vs travelled vs coasted, from the landmark."""
    if "start" not in marks or "stop" not in marks:
        print("\n[travel] no landmark samples — pass --cube <key> with the "
              "monitor running to get the coast measured instead of paced out.")
        return
    def d(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    asked = abs(vx) * seconds
    under_cmd = d(marks["start"], marks["stop"])
    print(f"\n[travel] commanded  {asked:.3f} m ({abs(vx):.2f} m/s x {seconds:.1f}s)")
    print(f"[travel] travelled  {under_cmd:.3f} m while commanded  "
          f"({100.0 * under_cmd / asked:.0f}% of the ask)")
    if "rest" in marks:
        coast = d(marks["stop"], marks["rest"])
        print(f"[travel] COASTED   {coast:.3f} m after the command hit zero")
        from humanoid_auto_operator import JOURNEY_ARRIVE_M as _ARRIVE
        print(f"[travel] -> JOURNEY_ARRIVE_M is {_ARRIVE:.2f}; with this coast "
              f"the cube ends up at ~{_ARRIVE - coast:.2f} m. Envelope is "
              f"0.16-0.60, clean to ~0.51, and the loaded-carry collision band "
              f"starts at 0.26.")
        if _ARRIVE - coast < 0.26:
            print("[travel] !! that is INSIDE the collision band — "
                  "JOURNEY_ARRIVE_M must go up before any journey runs.")


def _report_turn(trace, marks, wz_cmd, seconds, paused_seen) -> None:
    """Turn-in-place verdict: commanded vs achieved yaw, scrub, settle.

    Two independent yaw meters, deliberately: the gyro integral (drifts with
    bias but sees every sample) and the landmark bearing swing (bias-free but
    two samples). Agreement = a clean turn; a big gap = the base TRANSLATED
    while turning (scrub), which the range change then quantifies."""
    if not trace:
        print("[nav] no telemetry — was real_env running?")
        return
    print("\n" + "=" * 68)
    print(f"COMMANDED wz = {wz_cmd:+.2f} rad/s for {seconds:.1f}s  "
          f"(= {math.degrees(wz_cmd * seconds):+.0f} deg if tracked 1:1)")
    if paused_seen:
        print("!! _nav_paused was latched at some point — the numbers below "
              "may describe a robot that never received the command.")
    # Achieved yaw: trapezoid over the SIGNED gyro trace, commanded window only.
    cmd = [(t, w) for t, w, _, on in trace if on]
    if len(cmd) >= 2:
        yaw = sum((cmd[i + 1][0] - cmd[i][0]) * 0.5 * (cmd[i + 1][1] + cmd[i][1])
                  for i in range(len(cmd) - 1))
        peak = max(abs(w) for _, w in cmd)
        print(f"gyro yaw integral   : {math.degrees(yaw):+.0f} deg "
              f"({100.0 * yaw / (wz_cmd * seconds):.0f}% of the ask)")
        print(f"|wz| peak while on  : {peak:.3f} rad/s "
              f"(cmd {abs(wz_cmd):.2f}; a clean turn should hover near the "
              f"command, a shuffle spikes far past it)")
        # The same near-fall verdict the walk report carries (08-07 incident:
        # "moved" is not "moved cleanly"). Measured 08-08: wz=0.4 peaked 0.97
        # with dragging feet and a near fall — this line is what should have
        # shouted.
        if peak > UNSTABLE_WZ:
            print(f"  !! MOVED, BUT NOT CLEANLY. |wz| peaked at {peak:.2f} "
                  f"rad/s ({math.degrees(peak):.0f} deg/s), past the "
                  f"{UNSTABLE_WZ} bar — a degenerate shuffle, not a turn. "
                  f"DO NOT adopt this rate.")
        elif abs(yaw / (wz_cmd * seconds)) < 0.7:
            print(f"  !! UNDER-TRACKED: the base achieved "
                  f"{100.0 * yaw / (wz_cmd * seconds):.0f}% of the commanded "
                  f"yaw — likely torso wind-up or foot drag, not a clean "
                  f"turn. Watch the post-stop coast sign: a REBOUND "
                  f"(opposite sign) means the feet never stepped.")
    if "start" in marks and "stop" in marks:
        sx, sy = marks["start"]
        ex, ey = marks["stop"]
        b0, b1 = math.atan2(sy, sx), math.atan2(ey, ex)
        swing = (b1 - b0 + math.pi) % (2 * math.pi) - math.pi
        r0, r1 = math.hypot(sx, sy), math.hypot(ex, ey)
        # The cube's apparent bearing swings OPPOSITE the base yaw.
        print(f"landmark bearing    : {math.degrees(-swing):+.0f} deg of base "
              f"yaw (cross-check against the gyro integral above)")
        print(f"landmark range      : {r0:.3f} -> {r1:.3f} m  "
              f"(SCRUB {abs(r1 - r0) * 1000:.0f} mm radial — a pure turn "
              f"holds range; the align phase inherits this much approach "
              f"error per turn)")
        if "rest" in marks:
            rx, ry = marks["rest"]
            br = math.atan2(ry, rx)
            tail = (br - b1 + math.pi) % (2 * math.pi) - math.pi
            print(f"post-stop yaw coast : {math.degrees(-tail):+.1f} deg "
                  f"(the turn's stop transient — JOURNEY_ALIGN_SUSTAIN_S "
                  f"exists to absorb exactly this)")
    else:
        print("[turn] no landmark — pass --cube <key> with the monitor "
              "running to measure scrub and cross-check the yaw.")
    after = [(t, abs(w)) for t, w, _, _ in trace if t >= 0.0]
    if after:
        peak = max(w for _, w in after)
        print(f"|wz| after the stop : peak {peak:.3f} rad/s "
              f"(JOURNEY_STILL_ANG = {STILL_ANG})")
        quiet_from = None
        for t, w in after:
            if w >= STILL_ANG:
                quiet_from = None
            elif quiet_from is None:
                quiet_from = t
        print(f"  settles under the bar at t = "
              f"{quiet_from:+.2f}s" if quiet_from is not None else
              "  !! never settled under the bar inside the tail")
    print("=" * 68)


def _report(trace, vx, paused_seen) -> None:
    if not trace:
        print("[nav] no telemetry — was real_env running?")
        return
    # THE DISCRIMINATOR. base_lin_vel is hard zeros on this checkpoint, so the
    # only evidence in telemetry that the robot WALKED is its gyro: a gait
    # swings the torso every step, standing does not. Comparing the commanded
    # window against the settle window that preceded it turns "did it move?"
    # into a ratio instead of a guess. The first version pooled the two and
    # reported one median, which read as "quiet" no matter which it was.
    idle = [abs(w) for _, w, _, cmd in trace if not cmd]
    walk = [abs(w) for _, w, _, cmd in trace if cmd]
    after = [(t, abs(w)) for t, w, _, _ in trace if t >= 0.0]
    lin_seen = any(v > 1e-9 for _, _, v, _ in trace)

    print("\n" + "=" * 68)
    print(f"COMMANDED vx = {vx:+.2f} m/s")
    if paused_seen:
        print("!! _nav_paused was latched at some point — the numbers below "
              "may describe a robot that never received the command.")
    if idle and walk:
        m_idle, m_walk = statistics.median(idle), statistics.median(walk)
        print(f"|wz| standing (cmd 0) : p50 {m_idle:.3f}  max {max(idle):.3f}")
        print(f"|wz| COMMANDED        : p50 {m_walk:.3f}  max {max(walk):.3f}")
        ratio = m_walk / m_idle if m_idle > 1e-6 else float("inf")
        print(f"  gait signature ratio: {ratio:.1f}x")
        if ratio < 3.0:
            print("  !! THE BASE DID NOT WALK. A gait swings the torso every "
                  "step; this is a robot standing still with a command it did "
                  "not act on. Check, in this order: are the FEET LOADED (a "
                  "hoisted robot cannot walk), did real_env log a nonzero "
                  "'cmd:' line, is nav_paused clear?")
        elif max(walk) > UNSTABLE_WZ:
            # MOVING IS NOT WALKING (2026-08-07). The first version reported
            # only the ratio, and at a commanded 0.25 m/s it printed "It
            # walked" twice for a robot the operator described as "almost fall
            # down ... each step is soooooo small" — 96 deg/s of yaw with a
            # ratio of 6.5x. A degenerate, near-falling gait has a HIGHER
            # signature than a clean one, so the ratio alone cannot tell them
            # apart and reads as a pass on the more dangerous run. The clean
            # references measured that day peaked at 0.692 (fwd 0.30) and
            # 0.469 (rev 0.20); the near-fall peaked at 1.670.
            print(f"  !! MOVED, BUT NOT CLEANLY. |wz| peaked at {max(walk):.2f} "
                  f"rad/s ({math.degrees(max(walk)):.0f} deg/s), past the "
                  f"{UNSTABLE_WZ} bar. A clean gait at this robot's usable "
                  f"speeds peaks near 0.5-0.7. Expect tiny steps and a near "
                  f"fall. DO NOT adopt this speed as a floor — the band below "
                  f"a policy's usable range is UNSTABLE, not merely slow.")
        else:
            print("  -> the torso is oscillating like a gait. It walked.")
    print(f"base_lin_vel        : {'REPORTED' if lin_seen else 'HARD ZEROS'} "
          f"({'usable' if lin_seen else 'checkpoint has no velocity estimator '
             '— the stillness gate degrades to gyro + a 2.5 s fixed settle'})")

    if after:
        peak = max(w for _, w in after)
        print(f"|wz| after the stop : peak {peak:.3f} rad/s "
              f"(JOURNEY_STILL_ANG = {STILL_ANG})")
        # When does it get under the bar and STAY under it?
        quiet_from = None
        for i, (t, w) in enumerate(after):
            if w >= STILL_ANG:
                quiet_from = None
            elif quiet_from is None:
                quiet_from = t
        if quiet_from is None:
            print("  !! never settled under the bar inside the tail — "
                  "JOURNEY_STILL_TIMEOUT_S would fire and the leg would SKIP")
        else:
            print(f"  settles under the bar for good at t = {quiet_from:+.2f}s")
        spikes = sum(1 for _, w in after if w >= STILL_ANG)
        print(f"  samples over the bar: {spikes}/{len(after)}")
        print("  -> if those are ISOLATED, the 08-07 debit fix is right; if "
              "they are CONTIGUOUS, the base really is still moving and the "
              "gate is doing its job.")
    print("=" * 68)
    print("MEASURE THE COAST BY HAND: distance from the floor mark made at "
          "'STOP COMMANDED' to where the feet came to rest. That number is "
          "what JOURNEY_ARRIVE_M is paying for.")


if __name__ == "__main__":
    raise SystemExit(main())
