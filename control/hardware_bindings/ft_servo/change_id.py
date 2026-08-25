"""
change_id.py — change servo ID on a given port.

Usage (either form works since 2026-08-22; before that only the module form
did):
    python -m hardware_bindings.ft_servo.change_id <port> <old_id> <new_id>   # from control/
    python change_id.py <port> <old_id> <new_id>                              # any cwd

Acts only when invoked (importing it is inert); no --help. Either way the package import runs
ft_servo/__init__.py, which loads ft_servo_ext.abi3.so, so the extension must
be built even though this script itself never touches it.

Uses the pure-Python Feetech driver (not the C++ extension) since
EPROM methods (write_id, set_position_offset) are not bound in
ft_servo_ext. The copy imported here is byte-identical (below its header) to
control/ft_servo_python_only.py since 2026-08-22; only ping/write_id are
called here.
"""

import sys
if not __package__:
    # Plain-script invocation (`python change_id.py`): put control/ on sys.path
    # so the absolute package import resolves; `python -m ...` from control/
    # takes the relative-import branch.
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from hardware_bindings.ft_servo.ft_servo_python_only import FtServo
else:
    from .ft_servo_python_only import FtServo

def main(argv=None):
    """Change one servo's ID: <port> <old_id> <new_id>. Pings before and after."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        sys.exit("usage: change_id.py <port> <old_id> <new_id>")
    port, old_id, new_id = args[0], int(args[1]), int(args[2])

    drv = FtServo(port)
    ret = drv.ping(old_id)
    print(f"ping({old_id}) = {ret}")
    if ret is None:
        print("Ping failed")
        drv.close()
        sys.exit(1)

    drv.write_id(old_id, new_id)
    print(f"ID changed: {old_id} -> {new_id}")

    # Verify
    drv2 = FtServo(port)
    ret2 = drv2.ping(new_id)
    print(f"ping({new_id}) = {ret2}")
    drv.close()
    drv2.close()


if __name__ == "__main__":
    main()
