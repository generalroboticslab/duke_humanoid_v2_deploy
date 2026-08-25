"""Post-incident forensic probe: read motor RAM WITHOUT disturbing the evidence.

WHY THIS EXISTS (07-25, right-shoulder dropout campaign)
  After a dropout event the single most decisive fact lives in the motors'
  volatile RAM: parameter 0x7028 (CAN timeout). The host writes 5000 into it
  at every real_env startup. So, probed after an event and before the next
  restart / power cycle:

      0x7028 == 5000       motor RAM intact  -> the silence was a pure LINK
                           BREAK (harness/solder joint); the MCU never died
      0x7028 == other      motor MCU REBOOTED during the event (power dip or
                           firmware crash — a different repair entirely)
      no answer            motor is off-bus RIGHT NOW — the break is still
                           open, go wiggle the harness while this probe loops

  The stock wrapper (py_motor.CanMotorController) CANNOT be used for this:
  its __init__ does disable(clear_fault=True) — wiping the fault registers —
  and then WRITES 0x7028=5000, destroying exactly the evidence we came for.
  This probe talks to the raw nanobind class instead, which only opens
  sockets, and sends nothing but read requests unless --status is given.

USAGE
  1. Incident happens; STOP real_env (Ctrl+C).  Do NOT restart it, do NOT
     power-cycle the robot.
  2.   python humanoid_dropout_probe.py               # passive: 0x7028 + v_bus
       python humanoid_dropout_probe.py --status      # + disable(False) to
                                                      #   elicit mode/error
                                                      #   (sends DISABLE to all
                                                      #   motors — robot must
                                                      #   be hung/supported!)
       python humanoid_dropout_probe.py --loop        # repeat 1 Hz — wiggle
                                                      #   the harness and watch
                                                      #   answers come and go
  3. Read the table; then restart real_env as usual (that rewrites 5000).

  --status does NOT clear faults (disable(False)); error_code bits survive.
  Never run while real_env is up — two processes on one SocketCAN interleave.
"""
import argparse
import time

import numpy as np

from humanoid_config import motor_setup
# The .so must be loaded exactly ONCE per process. py_motor loads it by file
# path under the module name "motor_bindings"; importing it again via the
# package path (hardware_bindings.motor.motor_bindings) loads a second copy
# and nanobind aborts on the duplicate type registration. Reuse py_motor's
# already-loaded module — we still bypass its Wrapper CLASS (which clears
# faults and rewrites 0x7028 in __init__), just not its import machinery.
from hardware_bindings.motor.py_motor import motor_bindings

CanMotorController = motor_bindings.CanMotorController

try:
    from humanoid_base import URDF_JOINT_NAMES as _NAMES
except Exception:                                    # keep the probe runnable
    _NAMES = [f"motor_{i}" for i in range(len(motor_setup))]

RIGHT_ARM_IDS = {20, 21, 22, 23, 24, 25, 26}

GRN, RED, YEL, OFF = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def fast_watch(m: CanMotorController) -> None:
    """Active roll-call at ~3 Hz: the STILL-ARM alternative to wiggle_watch.

    wiggle_watch is passive — it can only flag a joint that froze MID-MOTION,
    so the arm must be kept moving. Here the host actively queries every
    motor (getParam 0x7028) a few times a second and alarms on ANSWER ->
    NO ANSWER transitions. A completely still arm works: press/twist one
    harness element at a time and HOLD each press >= 1 s (detection latency
    is one probe round, ~0.3 s). Motors are never enabled, nothing moves.

    Run this INSTEAD of real_env (stop real_env first — two processes on one
    SocketCAN interleave and confuse each other).
    """
    names = [(_NAMES[i] if i < len(_NAMES) else f"motor_{i}")
             for i in range(len(motor_setup))]
    state: dict[int, bool] = {}
    t0 = time.time()
    last_beat = t0
    print("[probe] FAST roll-call ~3 Hz — arm may stay STILL; hold each "
          "press >= 1 s.\n[probe] Watching all motors; right arm flagged. "
          "Ctrl+C to stop.\n")
    events = []
    while True:
        m.can_timeout[:] = 0
        m.getParam(0x7028)
        time.sleep(0.15)                      # response window
        ct = np.array(m.can_timeout, copy=True)
        now = time.time()
        for i, (cid, bus, _t) in enumerate(motor_setup):
            alive = bool(ct[i] != 0)
            prev = state.get(i)
            state[i] = alive
            if prev is None or alive == prev:
                continue
            tag = " <== RIGHT ARM" if cid in RIGHT_ARM_IDS else ""
            if not alive:
                events.append((now - t0, names[i]))
                print(f"\a{RED}[{now-t0:8.2f}s] ██ NO ANSWER ██ "
                      f"{names[i]} (ID{cid}/{bus}){tag}{OFF}")
            else:
                print(f"{GRN}[{now-t0:8.2f}s] answered again  "
                      f"{names[i]} (ID{cid}/{bus}){tag}{OFF}")
        if now - last_beat >= 5.0:
            last_beat = now
            dead = [names[i] for i, a in state.items() if not a]
            print(f"[{now-t0:8.2f}s] roll-call alive "
                  f"{len(state)-len(dead)}/{len(state)}"
                  + (f" · currently dead: {', '.join(dead)}" if dead else "")
                  + f" · {len(events)} drop event(s)")
        time.sleep(0.18)


def build_raw_controller() -> CanMotorController:
    """Mirror py_motor's constructor args — and NOTHING else it does."""
    can_interfaces = sorted({e[1] for e in motor_setup})
    iface_id = {n: i for i, n in enumerate(can_interfaces)}
    m = CanMotorController(
        can_interfaces,
        np.array([iface_id[e[1]] for e in motor_setup], dtype=np.uint8),
        np.array([e[0] for e in motor_setup], dtype=np.uint32),
        np.array([e[2] for e in motor_setup], dtype=np.uint8),
    )
    m.should_print_send = False
    m.should_print_recv = False
    return m


def probe_once(m: CanMotorController, elicit_status: bool) -> None:
    # 0x7028 readback — the discriminator. Zero the answer buffer first so
    # "still zero afterwards" is unambiguous non-response (the C++ ctor does
    # not initialise can_timeout).
    m.can_timeout[:] = 0
    m.getParam(0x7028)
    time.sleep(0.3)
    ct = np.array(m.can_timeout, copy=True)

    # v_bus etc. — passive reads, second liveness witness.
    m.getAllParams()
    time.sleep(0.5)
    vb = np.array(m.v_bus, copy=True)

    if elicit_status:
        # disable(clear_fault=False): elicits a status frame carrying
        # mode_status + error_code, keeps the fault registers intact.
        m.disable(False)
        time.sleep(0.3)
    ms = np.array(m.mode_status, copy=True)
    ec = np.array(m.error_code, copy=True)

    print(f"\n{'joint':22s} {'id':>3s} {'bus':6s} {'0x7028':>7s}  verdict"
          f"{'':13s} {'v_bus':>6s} {'mode':>4s} {'err':>3s}")
    for i, (cid, bus, _t) in enumerate(motor_setup):
        name = _NAMES[i] if i < len(_NAMES) else f"motor_{i}"
        mark = " <- right arm" if cid in RIGHT_ARM_IDS else ""
        if ct[i] == 5000:
            verdict = f"{GRN}RAM intact: link-only break{OFF}"
        elif ct[i] == 0:
            verdict = f"{RED}NO ANSWER: currently off-bus{OFF}"
        else:
            verdict = f"{YEL}={ct[i]}: MCU rebooted!{OFF}"
        print(f"{name:22s} {cid:3d} {bus:6s} {ct[i]:7d}  {verdict:30s}"
              f" {vb[i]:6.1f} {ms[i]:4d} {ec[i]:3d}{mark}")
    n_alive = int(np.count_nonzero(ct != 0))
    print(f"\n[probe] answered: {n_alive}/{len(motor_setup)}"
          + ("" if elicit_status else
         "   (the mode/err columns are only meaningful with --status — without "
         "status frames they stay at the boot value 0)"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true",
                   help="also send disable(clear_fault=False) to elicit "
                        "mode_status/error_code — sends DISABLE to every "
                        "motor, robot must be hung or supported")
    p.add_argument("--loop", action="store_true",
                   help="probe ~1 Hz forever — wiggle the harness and watch "
                        "a broken branch's answers come and go")
    p.add_argument("--fast", action="store_true",
                   help="active roll-call ~3 Hz with ANSWER/NO-ANSWER "
                        "transition alarms — the still-arm harness test "
                        "(no need to keep the arm moving)")
    args = p.parse_args()

    print("[probe] raw controller (no fault-clear, no 0x7028 write) …")
    m = build_raw_controller()
    try:
        if args.fast:
            fast_watch(m)
        else:
            probe_once(m, args.status)
            while args.loop:
                time.sleep(1.0)
                probe_once(m, args.status)
    except KeyboardInterrupt:
        pass
    finally:
        m.stop()


if __name__ == "__main__":
    main()
