# hardware_bindings

Generic C++ nanobind extensions for motor control, IMU, and FT servo hardware.

## Modules

- `motor_bindings` — CAN motor controller (DukeHumanoidV2 leg/arm motor primitives)
- `imu_nanobind` — IMU serial reader (SYD Dynamics TransducerM TM171, EasyProfile protocol)
- `ft_servo_ext` — Feetech SCServo / HLSCL serial driver

## Build

The sources live here, in this repository (`motor/`, `imu/`, `ft_servo/`,
with the vendor's EasyProfile protocol code under `imu/EasyProfile/` —
SYD Dynamics, BSD-2-Clause, see `imu/EasyProfile/NOTICE.md`; likewise the
FEETECH SDK under `ft_servo/`, MIT, `ft_servo/NOTICE.md`); there is no
submodule to initialise. The extensions are compiled by the CMake project
one level up — `control/CMakeLists.txt` does `add_subdirectory(hardware_bindings)`
— against the vcpkg manifest `control/vcpkg.json` (Eigen3, CSerialPort
(itas109), fmt) plus nanobind `>= 2.2.0` from pip and a Python `>= 3.12`
with development headers. From the repository root:

```bash
cmake -S control -B control/build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DPython_EXECUTABLE="$(which python)"
cmake --build control/build
```

Each module's `.abi3.so` lands **next to its sources** (`motor/motor_bindings.abi3.so`,
`imu/imu_nanobind.abi3.so`, `ft_servo/ft_servo_ext.abi3.so`, each with a copy of
`libnanobind-abi3.so` beside it); that is the one path the Python wrappers
(`motor/py_motor.py`, `ft_servo/__init__.py`) load from. The modules carry
`RUNPATH [$ORIGIN]` and `imu_nanobind` links EasyProfile statically, so they
resolve `libnanobind-abi3.so` from that copy beside them and nothing else from
the build tree: `control/build` may be cleaned and the directory moved after
building (since 2026-08-22). The build is `STABLE_ABI`, so one build serves
every CPython `>= 3.12`. The `.so` files are gitignored: a fresh clone has none
until you build.

The full walk-through — system packages, installing vcpkg, the VS Code
presets, verification and the common failure modes — is
[`../docs/SETUP.md`](../docs/SETUP.md), section 2.

## Install (optional)

```bash
pip install -e .          # from this directory
```

A plain setuptools editable install (`pyproject.toml`) that makes
`hardware_bindings` importable from any directory instead of only from
`control/`. It does not compile anything: the `.abi3.so` files are package
data read from the source directories, so run the CMake build above as well,
before or after. Scripts and tests run from `control/` work without it.

## Import

```python
from hardware_bindings.motor import motor_bindings      # or: from hardware_bindings.motor.py_motor import ...
from hardware_bindings.imu import imu_nanobind
from hardware_bindings.ft_servo import FtServo
```

Verify a build from `control/`:

```bash
python -c "from hardware_bindings.motor import motor_bindings; from hardware_bindings.imu import imu_nanobind; from hardware_bindings.ft_servo import FtServo; print('hardware_bindings OK')"
```

## Servo helper scripts

`ft_servo/scan.py`, `ft_servo/change_id.py` and `ft_servo/test.py` run either
way since 2026-08-22 — as modules from `control/`, or as plain scripts from any
directory (when run as a script each one puts `control/` on `sys.path` and
imports the package absolutely; before that only the module form worked):

```bash
python -m hardware_bindings.ft_servo.scan [port] [--baud B] [--start N --end M] [--list]
python -m hardware_bindings.ft_servo.change_id <port> <old_id> <new_id>
python -m hardware_bindings.ft_servo.test          # viser GUI, no arguments

python hardware_bindings/ft_servo/scan.py                     # every USB serial port, common baud rates
python hardware_bindings/ft_servo/scan.py /dev/ttyACM0 --baud 1000000 --start 0 --end 253
python hardware_bindings/ft_servo/scan.py --list              # list serial ports and exit
```

Either form imports the package, which runs `ft_servo/__init__.py` and loads
`ft_servo_ext.abi3.so` (FileNotFoundError until built) — `change_id.py` needs
the build too even though it drives the bus through the pure-Python driver.
`test.py` acts on invocation (no `--help`, opens the port at module level and
enables torque), so do not import it from other code; `scan.py` (`--help`) and
`change_id.py` (usage line) act only when invoked.

## Twin modules

Two files in this directory have a twin under `control/`; the `control/` copy
is the canonical one in both cases, and since 2026-08-22 each pair is
byte-identical below its header comment (the header names the twin and the
rule: fix the canonical copy first, then mirror it):

- `common/publisher.py` — twin of `control/ipc/publisher.py` (that package was
  `control/common` until 2026-08-22; the rename ended the clash of three
  packages named `common`, see `control/docs/ARCHITECTURE_MAP.md` section 2).
  Identical in behaviour: `strict_map_key=False` on both `msgpack.unpackb`
  calls was ported here, so a subscriber built from this copy no longer drops
  packets with non-string (e.g. int camera-port) dict keys. Importer of this
  copy: `imu/py_imu.py` only (as `hardware_bindings.common.publisher`, with a
  bare `common.publisher` fallback for direct runs that puts *this* directory
  on `sys.path`); `control/humanoid_mission_recorder.py` imports the canonical
  `ipc.publisher`.
- `ft_servo/ft_servo_python_only.py` — twin of `control/ft_servo_python_only.py`.
  Identical except for the header: the sign-magnitude encoding of negative
  positions in `set_position` was ported here. Importer of this copy:
  `ft_servo/change_id.py` (which never calls `set_position`).

Related bench note: the GL40 II one-motor bench (`motor/gl_bench.py`,
`motor/py_gl_motor.py`) brings its single bus up with
`python control/humanoid_setup_can.py --interfaces can26`; the separate
`gl_setup_can.py` copy of that script was removed on 2026-08-22.
