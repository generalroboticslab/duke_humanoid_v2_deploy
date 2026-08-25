#!/usr/bin/env python3
"""Setup CAN interfaces. Run this script to get all CAN working."""
import subprocess
import time
import os
import glob
import argparse

CAN_INTERFACES = ["can9", "can21", "can22", "can23", "can24", "can25"]
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_serial_map():
    """Parse udev rules: interface name -> serial number."""
    serial_map = {}
    rules_path = "/etc/udev/rules.d/99-candlelight.rules"
    if os.path.exists(rules_path):
        for line in open(rules_path):
            if 'NAME="' in line and 'ATTRS{serial}=="' in line and not line.startswith("#"):
                try:
                    serial_map[line.split('NAME="')[1].split('"')[0]] = line.split('ATTRS{serial}=="')[1].split('"')[0]
                except IndexError:
                    pass
    return serial_map


def find_usb_path(serial):
    """Find USB device path by serial number."""
    for path in glob.glob("/sys/bus/usb/devices/*/serial"):
        try:
            if open(path).read().strip() == serial:
                d = os.path.dirname(path)
                return f"/dev/bus/usb/{int(open(f'{d}/busnum').read()):03d}/{int(open(f'{d}/devnum').read()):03d}"
        except:
            pass
    return None


def setup(can):
    """Setup CAN interface. Returns True on success."""
    subprocess.run(f"sudo ip link set {can} down", shell=True, capture_output=True)
    
    # Try with restart-ms (supported by most gs_usb adapters)
    r = subprocess.run(
        f"sudo ip link set {can} up type can bitrate 1000000 restart-ms 100",
        shell=True, capture_output=True)
    
    # Fall back without restart-ms (some adapters don't support it)
    if r.returncode != 0:
        r = subprocess.run(
            f"sudo ip link set {can} up type can bitrate 1000000",
            shell=True, capture_output=True)
    
    if r.returncode != 0:
        return False
    
    return subprocess.run(
        f"sudo ifconfig {can} txqueuelen 50",
        shell=True, capture_output=True).returncode == 0

def teardown(can):
    """Bring CAN interface down. Returns True on success."""
    return subprocess.run(f"sudo ip link set {can} down", shell=True, capture_output=True).returncode == 0


parser = argparse.ArgumentParser(description="Setup or teardown CAN interfaces")
parser.add_argument("--down", action="store_true", help="Bring CAN interfaces down instead of up")
parser.add_argument("--interfaces", nargs="+", default=CAN_INTERFACES,
                    help="CAN interface names; default = the robot's six body buses")
args = parser.parse_args()

serial_map = get_serial_map()
failed = []

if args.down:
    # Teardown mode
    for can in args.interfaces:
        print(f"{can}...", end=" ", flush=True)
        if os.path.exists(f"/sys/class/net/{can}") and teardown(can):
            print(f"{GREEN}DOWN{RESET}")
        else:
            print(f"{RED}FAILED{RESET}")
            failed.append(can)
else:
    # Setup mode
    for can in args.interfaces:
        print(f"{can}...", end=" ", flush=True)

        # Try setup
        if os.path.exists(f"/sys/class/net/{can}") and setup(can):
            print(f"{GREEN}OK{RESET}")
            continue

        # Try USB reset
        usb = find_usb_path(serial_map.get(can))
        if usb:
            print("reset...", end=" ", flush=True)
            subprocess.run(["sudo", "usbreset", usb], capture_output=True)
            subprocess.run("sudo udevadm trigger", shell=True, capture_output=True)
            time.sleep(0.5)
            if os.path.exists(f"/sys/class/net/{can}") and setup(can):
                print(f"{GREEN}OK{RESET}")
                continue

        print(f"{RED}FAILED{RESET}")
        failed.append(can)

print()
time.sleep(0.1)
subprocess.run("ip -brief link show type can", shell=True)
if failed:
    print(f"\n{RED}Replug: {', '.join(failed)}{RESET}")
