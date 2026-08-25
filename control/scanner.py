"""
Quick diagnostic: is my FeeeTech HLS servo actually connected?

Checks, in order:
  1. Does the serial port exist / open at all?
  2. Bus scan (0-253) for any responding servo IDs — doesn't require
     knowing the ID in advance.
  3. If a known ID is given, ping it directly and read position/voltage/temp.

Usage:
    python check_servo.py /dev/ttyACM0
    python check_servo.py /dev/ttyACM0 --id 1
    python check_servo.py /dev/ttyACM0 --baud 1000000 --id 1
"""

import argparse
import glob
import sys

from ft_servo_python_only import FtServo  # adjust import if your file is named differently


def list_likely_ports():
    candidates = sorted(
        glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/tty.usb*")
    )
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", help="Serial port, e.g. /dev/ttyACM0")
    ap.add_argument("--id", type=int, default=None, help="Known servo ID to ping directly")
    ap.add_argument("--baud", type=int, default=1_000_000)
    args = ap.parse_args()

    port = args.port
    if not port:
        found = list_likely_ports()
        print("No port given. Likely candidates on this machine:")
        for p in found:
            print(f"   {p}")
        if not found:
            print("   (none found — check `ls /dev/tty*` and your USB connection/cable)")
        print("\nRe-run as: python check_servo.py <port>")
        sys.exit(1)

    print(f"Opening {port} @ {args.baud} baud...")
    try:
        servo = FtServo(port, baudrate=args.baud, timeout=0.1)
    except Exception as e:
        print(f"FAILED to open port: {e}")
        print("  - Check the cable/adapter is plugged in")
        print("  - Check the port name (`ls /dev/ttyACM*` or `/dev/ttyUSB*`)")
        print("  - Check you have permission (on Linux: are you in the `dialout` group?)")
        sys.exit(1)

    print("Port opened OK.\n")

    # ── Step 1: bus scan ────────────────────────────────────────────────
    print("Scanning bus for responding servo IDs (0-253)... this takes a few seconds.")
    try:
        found = servo.scan()
    except Exception as e:
        print(f"Scan raised an exception: {e}")
        found = []

    if found:
        print(f"Found {len(found)} servo(s) responding: {found}\n")
    else:
        print("No servos responded to the scan.")
        print("  - Check power to the servo (separate from USB power, usually 6-12V)")
        print("  - Check the data line / half-duplex wiring")
        print("  - Check baudrate matches what's configured on the servo (default often 1Mbps)")
        print("  - Try the other USB port/cable if you have a left/right pair\n")

    # ── Step 2: targeted ping + read, if an ID was given or found ───────
    target_id = args.id if args.id is not None else (found[0] if found else None)

    if target_id is not None:
        print(f"Pinging ID {target_id} directly...")
        ret = servo.ping(target_id)
        if ret is None:
            print(f"  Ping FAILED for ID {target_id} (no/garbled response).")
        else:
            _, error, _ = ret
            print(f"  Ping OK. error byte = 0x{error:02X} ({'no error' if error == 0 else 'servo reported an error flag'})")

            pos = servo.get_position(target_id)
            volt = servo.get_voltage(target_id)
            temp = servo.get_temperature(target_id)
            speed = servo.get_speed(target_id)
            load = servo.get_load(target_id)

            print(f"  Position:    {pos}")
            print(f"  Speed:       {speed}")
            print(f"  Load:        {load}")
            print(f"  Voltage:     {volt/10.0 if volt is not None else '?'} V")
            print(f"  Temperature: {temp} C" if temp is not None else "  Temperature: ?")
    else:
        print("No ID to ping (none found in scan and --id not given). Stopping here.")

    servo.close()


if __name__ == "__main__":
    main()
