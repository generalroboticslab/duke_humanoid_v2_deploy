"""
scan.py — check that the USB driver board and the FeeTech HLS servos on its bus
are alive.  Cross-platform: Windows (COMx) and Linux (/dev/tty*).

The two failure modes are reported separately, because the fixes differ:
  1. no serial port at all        -> USB cable / VCP driver / board problem
  2. port opens but nobody answers -> servo power, bus wiring, baud or ID problem

Only dependency is pyserial (`pip install pyserial`).  The compiled
`ft_servo_ext` extension is deliberately NOT used, so this runs on a machine
with no build toolchain.

Usage:
    python scan.py                      # auto: every port, common baud rates
    python scan.py COM5                 # one port, common baud rates
    python scan.py COM5 --baud 1000000  # one port, one baud
    python scan.py --list               # list serial ports and exit
    python scan.py --start 0 --end 253  # widen the ID range (default 0-20)

Exit status: 0 if at least one servo answered, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing.  Install it with:  pip install pyserial")

# FtServo lives in the repo root (control/).  The script's own directory is
# searched first so that copying ft_servo_python_only.py next to scan.py is
# enough to run this on a bare test machine.
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent), str(_HERE.parent / "hardware_bindings" / "ft_servo")]
try:
    from ft_servo_python_only import FtServo
except ImportError:
    sys.exit("ft_servo_python_only.py not found — copy it into the same folder as scan.py.")

COMMON_BAUDS = (1_000_000, 500_000, 115_200)
PING_TIMEOUT = 0.02  # s — generous next to the driver's 4 ms, so 115200 still answers


def available_ports():
    """Serial ports, minus Bluetooth virtual ports (opening those can block)."""
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    return [p for p in ports if "bluetooth" not in (p.description or "").lower()]


def usb_ports(ports):
    """
    Ports backed by real USB hardware, i.e. carrying a VID:PID.

    Auto mode probes only these: a USB driver board always enumerates with a
    VID:PID, whereas the machine also advertises legacy/virtual ports (32 of
    /dev/ttyS* on Linux, COM1 on Windows) that would each cost a full ID sweep
    to prove dead.  Symlinks such as /dev/ttyACMservoLeft are dropped here too,
    so the device behind them is not probed twice.
    """
    return [p for p in ports if p.vid is not None]


def probe(port, baud, start_id, end_id):
    """
    Ping every ID in [start_id, end_id] on one port at one baud rate.

    Returns (rows, error).  rows is a list of (id, position, voltage_raw,
    temperature) for each servo that answered; voltage_raw is in 0.1 V units.
    error is a message string when the port could not be opened at all — busy,
    permission denied, no such port — and None otherwise, so the caller can
    tell "board absent" from "bus silent".
    """
    try:
        drv = FtServo(port, baudrate=baud, timeout=PING_TIMEOUT)
    except (serial.SerialException, OSError) as exc:
        return [], str(exc)

    try:
        drv.serial.reset_input_buffer()
        ids = [sid for sid in range(start_id, end_id + 1) if drv.ping(sid) is not None]
        rows = [(sid, drv.get_position(sid), drv.get_voltage(sid), drv.get_temperature(sid))
                for sid in ids]
    finally:
        drv.close()
    return rows, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?",
                    help="serial port, e.g. COM5 or /dev/ttyACM0 (default: try every port found)")
    ap.add_argument("--baud", type=int,
                    help=f"single baud rate (default: try {', '.join(str(b) for b in COMMON_BAUDS)})")
    ap.add_argument("--start", type=int, default=0, help="first servo ID to ping (default 0)")
    ap.add_argument("--end", type=int, default=20, help="last servo ID to ping (default 20)")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    ports = available_ports()
    usb = usb_ports(ports)
    print("USB serial ports detected:")
    for p in usb:
        print(f"  {p.device:<14} {p.description}  [{p.hwid}]")
    if not usb:
        print("  (none)")
    if len(ports) > len(usb):
        print(f"  ({len(ports) - len(usb)} legacy/virtual port(s) hidden — pass one by name to probe it)")

    if args.list:
        return 0

    if not usb and not args.port:
        print("\nNO USB SERIAL PORT -> the driver board is not reachable.  Check:\n"
              "  - USB cable seated, and a DATA cable (charge-only cables carry no data)\n"
              "  - USB-serial driver installed (CH340 / CP210x / FTDI).  Windows Device\n"
              "    Manager shows a yellow '!' under 'Other devices' when it is missing\n"
              "  - board power LED lit")
        return 1

    targets = [args.port] if args.port else [p.device for p in usb]
    bauds = [args.baud] if args.baud else list(COMMON_BAUDS)

    total = 0
    opened_any = False
    for port in targets:
        for baud in bauds:
            rows, error = probe(port, baud, args.start, args.end)
            if error:
                print(f"\n{port} @ {baud}: cannot open — {error}")
                break  # the port itself is the problem; other bauds fail identically
            opened_any = True
            print(f"\n{port} @ {baud}: {len(rows)} servo(s) answered on IDs {args.start}-{args.end}")
            for sid, pos, volt, temp in rows:
                volt_str = f"{volt / 10:.1f}V" if volt is not None else "?"
                print(f"  ID {sid:<4} pos={pos}  volt={volt_str}  temp={temp}C")
            if rows:
                total += len(rows)
                break  # correct baud for this port; no reason to try the rest

    if total:
        print(f"\nOK — {total} servo(s) responding.  Driver board and bus are working.")
        return 0

    if opened_any:
        print("\nPORT OPENS BUT NO SERVO ANSWERS -> the board enumerates, the bus is silent.  Check:\n"
              "  - servo power supply on (USB alone does not power the servos)\n"
              "  - 3-pin bus cable in the right connector, fully seated, not swapped\n"
              "  - servo ID inside the scanned range: retry with --start 0 --end 253\n"
              f"  - baud rate: tried {', '.join(str(b) for b in bauds)}; FeeTech HLS ships at 1 Mbps\n"
              "  - nothing else is holding the port (close the FD/FT SCServo debug tool)")
    else:
        print("\nNO PORT COULD BE OPENED -> see the error above.  Common causes:\n"
              "  - port name typo, or the board was unplugged (re-run with --list)\n"
              "  - another program holds it (FD/FT SCServo tool, a serial terminal)\n"
              "  - permissions: on Linux join the 'dialout' group or use sudo")
    return 1


if __name__ == "__main__":
    sys.exit(main())
