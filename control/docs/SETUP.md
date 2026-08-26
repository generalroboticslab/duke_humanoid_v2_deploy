# Building and setting up the control stack

What has to be in place on the **robot computer** before the bring-up ladder
in [OPERATIONS.md](OPERATIONS.md) can start: the Python environment, the C++
motor/IMU/servo extensions, the CAN buses, the gripper serial ports, the
hand-eye calibration, and the network between the robot computer, the
workstation and the GPU machine.

`<repo>` is the repository root (the directory holding `control/`,
`perception/` and `requirements.txt`). Unless a command says otherwise it is
run from there. Nothing in this document names a particular machine; every
installation-specific value goes through `control/humanoid_site.py` (see
section 6).

Contents

1. [Python environment](#1-python-environment)
2. [Building the C++ extensions](#2-building-the-c-extensions-hardware_bindings)
3. [CAN buses](#3-can-buses)
4. [Servo ports (FEETECH grippers)](#4-servo-ports-feetech-grippers)
5. [Hand-eye calibration](#5-hand-eye-calibration)
6. [Network](#6-network)

---

## 1. Python environment

The stack requires **Python 3.12** (it uses PEP 701 f-strings and will not
parse on 3.11 or earlier; the C++ extensions are built for the 3.12 stable ABI).

```bash
cd <repo>
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install "nanobind>=2.2.0"      # build-time only: needed to compile hardware_bindings
```

Notes:

- `requirements.txt` pins the versions the robot actually runs. `torch` is
  listed by version only; install the `+cpu` / `+cu1xx` / `+rocm` wheel that
  matches your hardware (see the comment block in `requirements.txt`).
- Two dependencies are deliberately absent because they are checkouts, not
  PyPI packages: `legged_env_v2` and cuRobo. Neither is needed on the robot:
  `control/legged_env_bundle/` carries the deploy run directories, the meshes
  and the CPU-side Python the robot loop imports, and `humanoid_site` uses it
  whenever no `legged_env_v2` checkout sits beside this repository. Both are
  needed on the GPU machine that runs the plan server. See the top-level
  `README.md`, "Install". Do not `pip install mj_envs`: that PyPI package is
  an unrelated project whose name collides with the upstream package.
- `nanobind` is not in `requirements.txt` because it is only needed while
  compiling; the built `.abi3.so` files link against the `libnanobind-abi3.so`
  the same build produces (see 2.4 for where it is found at run time) and the
  runtime never imports the Python package. The build needs
  `>= 2.2.0` (older releases fail at the include step; see "Common problems").
- The interpreter you build against must ship its development headers
  (CMake's `FindPython` `Development` component). A venv made from a distro
  Python needs `sudo apt install python3.12-dev`; conda/micromamba
  environments already include them. A conda-style environment works just
  as well as a venv:

  ```bash
  micromamba create -n py312 python=3.12 pip && micromamba activate py312
  pip install -r requirements.txt && pip install "nanobind>=2.2.0"
  ```

Then tell the stack where things are:

```bash
cp control/site_local.example.py control/site_local.py   # and edit
```

---

## 2. Building the C++ extensions (`hardware_bindings`)

`control/hardware_bindings/` holds three nanobind extensions — `motor_bindings`
(CAN motor controller, the 200 Hz servo loop), `imu_nanobind` (TM3xx IMU over
serial / EasyProfile) and `ft_servo_ext` (FEETECH SCServo/HLSCL gripper
driver). They are built by the CMake project in `control/CMakeLists.txt`,
which pulls in `control/hardware_bindings/CMakeLists.txt` with
`add_subdirectory`. The sources are part of this repository (no submodule to
initialise).

### 2.1 System packages

Debian/Ubuntu:

```bash
sudo apt install build-essential cmake ninja-build pkg-config git curl zip unzip tar
```

(`cmake >= 3.15` per the project file; the build was last done with 3.28.
`curl zip unzip tar` are vcpkg's own bootstrap prerequisites.)

### 2.2 vcpkg and the C++ dependencies

The C++ dependencies are declared in `control/vcpkg.json` (vcpkg *manifest
mode*): **eigen3**, **cserialport** (itas109 CSerialPort) and **fmt** — the
whole manifest is `{"name": "v2control", "version": "1.0", "dependencies":
["eigen3", "cserialport", "fmt"]}`. (TBB was listed until 2026-08-22; nothing
ever used it and every fresh checkout was compiling it for nothing.) Install
vcpkg itself once:

```bash
git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
"$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
export VCPKG_ROOT="$HOME/vcpkg"
```

You do not have to run `vcpkg install` by hand: when `CMAKE_TOOLCHAIN_FILE`
points at vcpkg, the configure step below reads `control/vcpkg.json` and
installs the three ports into `control/build/vcpkg_installed/` (gitignored).
The first time this compiles them from source. If you would rather watch that
happen before CMake gets involved,

```bash
cd <repo>/control && "$VCPKG_ROOT/vcpkg" install      # -> control/vcpkg_installed/
```

does the same work (also gitignored) and the later configure step is served
from vcpkg's binary cache instead of compiling again.

### 2.3 Configure and build (command line)

From the repository root, with the project environment active so that
`which python` is the interpreter the stack will run on:

```bash
cmake -S control -B control/build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DPython_EXECUTABLE="$(which python)"
cmake --build control/build
```

- `Python_EXECUTABLE` matters twice: CMake asks that interpreter
  `python -m nanobind --cmake_dir` to locate nanobind, and `FindPython`
  takes the headers from it. It must be the environment of section 1.
- `CMAKE_BUILD_TYPE` defaults to **Release** when unset (set by
  `control/CMakeLists.txt`).
- Targets: the three nanobind modules (`NB_SHARED` + `STABLE_ABI`, so one
  build serves every CPython `>= 3.12`) and `sleep_test`
  (`control/ipc/sleep_test.cpp`, a timer-latency check that ends up at
  `control/build/sleep_test`). The top-level project is
  `duke_humanoid_v2_control` (`project(...)` in `control/CMakeLists.txt`),
  and `control/hardware_bindings/imu/EasyProfile/` is built as a **static**
  library that `imu_nanobind` links in.
- Both `control/CMakeLists.txt` and `control/hardware_bindings/CMakeLists.txt`
  require Python `>= 3.12` (`find_package(Python 3.12 ...)`).

### 2.4 Where the outputs land, and how to verify

`hardware_bindings/CMakeLists.txt` sets each module's `LIBRARY_OUTPUT_DIRECTORY`
to its **own source directory**, and a post-build step copies
`libnanobind-abi3.so` next to it. After a successful build you have:

```
control/hardware_bindings/motor/motor_bindings.abi3.so   (+ libnanobind-abi3.so)
control/hardware_bindings/imu/imu_nanobind.abi3.so       (+ libnanobind-abi3.so)
control/hardware_bindings/ft_servo/ft_servo_ext.abi3.so  (+ libnanobind-abi3.so)
```

That is the single path the Python wrappers load from
(`hardware_bindings/motor/py_motor.py` and `hardware_bindings/ft_servo/__init__.py`
resolve `<their own directory>/<name>.abi3.so`), shared by dev builds and
editable installs alike. `*.so` is gitignored, so a fresh clone has none of
them until you build — until then the test modules that reach a compiled
binding error at import, as described under "Tests" in the top-level `README.md`.

**The modules are relocatable** (since 2026-08-22). Each of the three
`.abi3.so` files carries `RUNPATH [$ORIGIN]` (`readelf -d` shows it, and no
absolute path), so the dynamic loader resolves `libnanobind-abi3.so` from the
copy beside the module; EasyProfile is linked **statically** into
`imu_nanobind`, so there is no `libEasyProfile.so` to find at all. Beyond
nanobind the modules need only the system C/C++ runtime. Consequences:
`control/build` may be deleted or cleaned after building, the checkout — or
just `hardware_bindings/` with its `.so` files — may be moved or copied, and
`LD_LIBRARY_PATH` is never needed. (Before this the `.so` files carried an
absolute `RUNPATH` into `control/build`, and EasyProfile was a shared library
that existed only there, so cleaning or moving the build tree broke the
imports.)

Verify, from `control/` (the directory the scripts and tests run from):

```bash
cd <repo>/control
python -c "from hardware_bindings.motor import motor_bindings; from hardware_bindings.imu import imu_nanobind; from hardware_bindings.ft_servo import FtServo; print('hardware_bindings OK')"
```

### 2.5 The CMake presets (VS Code)

`control/CMakePresets.json` (tracked; preset `vcpkg`: Ninja, build dir
`control/build`, `CMAKE_BUILD_TYPE=Debug`) exists for the VS Code
**cmake-tools** extension (`code --install-extension ms-vscode.cmake-tools`,
then F1 -> "CMake: Build"). `control/CMakeUserPresets.json` is CMake's
per-user file — your own, gitignored; create it with a preset that inherits
`vcpkg` if you want to override values without editing the tracked file.
Two things to know before using them:

- The tracked `CMAKE_TOOLCHAIN_FILE` and `Python_EXECUTABLE` values
  (`$env{HOME}/repo/vcpkg/...`, `$env{HOME}/repo/micromamba/envs/py312/bin/python`)
  encode one developer's directory layout. Point them at your vcpkg clone and
  your interpreter (in the tracked file or in your user presets), or configure
  from the command line as in 2.3, which is the reference procedure.
- The presets build **Debug**; the command line in 2.3 builds Release.

### 2.6 Alternative: `pip install -e`

```bash
cd <repo>
pip install -e control/hardware_bindings
```

`control/hardware_bindings/pyproject.toml` is a plain setuptools package
(`hardware_bindings.motor`, `.imu`, `.ft_servo`, with `*.abi3.so` as package
data). The editable install makes `hardware_bindings` importable from any
directory instead of only from `control/`; it does **not** compile anything,
so the CMake build above is still required — before or after, either order,
because both read the same source-directory `.so` files. Each module finds
`libnanobind-abi3.so` beside itself (`$ORIGIN`, see 2.4), which the build's
copy step puts there; note that the package-data declaration lists only
`*.abi3.so`, so a *non*-editable install (a wheel) would have to carry
`libnanobind-abi3.so` as well — the editable install reads the source
directories and has it. Scripts and tests run from `control/` do not need it.

### Common problems

- *nanobind include / "could not find nanobind"* — `pip show nanobind` in the
  **same** interpreter you passed as `Python_EXECUTABLE`; version must be
  `>= 2.2.0`.
- *"Could not find a package configuration file provided by CSerialPort"
  (or Eigen3 / fmt)* — `CMAKE_TOOLCHAIN_FILE` is missing or points at
  the wrong vcpkg clone; without the toolchain file CMake never sees the
  manifest.
- *Python found but `Development` component missing* — install the
  interpreter's headers (`python3.12-dev`) or build against a conda-style
  environment.
- *`FileNotFoundError: ... .abi3.so — run cmake --build control/build first`*
  at import time — the module was not built, or was built for a different
  `Python_EXECUTABLE` than the one importing it.
- *`ImportError: libnanobind-abi3.so: cannot open shared object file`* — the
  copy that the build places beside the module is missing (2.4): something
  deleted it, or the `.abi3.so` was copied somewhere without it. Re-run
  `cmake --build control/build`, or copy `libnanobind-abi3.so` next to the
  module. (Builds from before 2026-08-22 also raised this — or
  `libEasyProfile.so` — whenever `control/build` was cleaned or moved; a
  rebuild with the current `CMakeLists.txt` removes that dependency.)

---

## 3. CAN buses

The body motors sit on six CAN buses driven by USB-CAN adapters
(`gs_usb` / candlelight class) at 1 Mbit/s, named **`can9`, `can21`, `can22`,
`can23`, `can24`, `can25`** (`can9` carries the left arm). Those names are
what `humanoid_setup_can.py` and the motor layer expect.

### Kernel modules

```bash
sudo modprobe can can_raw gs_usb
```

`gs_usb` is in-tree and normally auto-loads when an adapter is plugged in;
run the line above if `ip link` cannot find the interfaces.

### Stable interface names

Without a rule the adapters enumerate as `can0`, `can1`, ... in plug order,
which the scripts do not use. Name each adapter by its USB serial with a udev
rule, one line per adapter:

```
# /etc/udev/rules.d/99-candlelight.rules
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<adapter-serial>", NAME="can21"
```

(Find the serial with
`udevadm info --attribute-walk --path=/sys/class/net/can0 | grep 'ATTRS{serial}'`,
then `sudo udevadm control --reload-rules && sudo udevadm trigger` and replug.)
`humanoid_setup_can.py` reads this same file: when a bus refuses to come up it
looks the adapter's serial up there, finds the USB device and `usbreset`s it.

### Bring-up — after EVERY robot power cycle (OPERATIONS.md, T0)

```bash
cd <repo>/control
python humanoid_setup_can.py
# verify: all six ERROR-ACTIVE
for c in can9 can21 can22 can23 can24 can25; do
  echo -n "$c: "; ip -details link show $c | grep -o "can state [A-Z-]*" | head -1
done
```

`humanoid_setup_can.py` takes each interface down, brings it up with
`type can bitrate 1000000 restart-ms 100` (falling back to no `restart-ms`
for adapters that reject it) and sets `txqueuelen 50`; an interface that
fails is USB-reset via the rules file above and retried; the run ends with
`ip -brief link show type can` and a `Replug: ...` list of anything still
down. `python humanoid_setup_can.py --down` tears the buses down again, and
`--interfaces <name> ...` replaces the default six with any other list (for
example `--interfaces can26` for the single-motor GL40 bench of
`hardware_bindings/motor/gl_bench.py`, which used to have its own copy of this
script). The script calls `sudo ip` / `sudo ifconfig` / `sudo usbreset`
(`net-tools`, `usbutils`), so run it as a user who may sudo.

Any `[RECV] error frame` in the `humanoid_real_env.py` terminal, or a bus
stuck in ERROR-WARNING -> power-cycle again; if it recurs, inspect the harness
before running.

### Serial devices: `dialout`

The IMU and the gripper driver boards are USB-serial devices. Put the user
in the `dialout` group once, then log out and back in (or `newgrp dialout`
in the current shell):

```bash
sudo usermod -aG dialout $USER
```

---

## 4. Servo ports (FEETECH grippers)

The `/dev/ttyACM*` device names change based on detection order. To get a
stable name, create a udev rule using the device serial number.

### Step 1: Identify which ttyACM is the servo

```bash
# List all ttyACM ports with their udev info
for p in /dev/ttyACM*; do echo "=== $p ==="; udevadm info --name=$p 2>/dev/null | grep -E 'ID_SERIAL|ID_MODEL|ID_VENDOR'; done
```

- **STMicroelectronics** (STM32 Virtual COM Port) = on-board debug UART
- **QinHeng Electronics** (USB Single Serial, CH340) = low-cost USB-serial adapter — likely your servo or similar

If unsure which CH340 is the servo:
1. Unplug all CH340 devices
2. Note which ttyACM disappear
3. Plug servo back in -> new entry = servo
4. Get its serial: `udevadm info --name=/dev/ttyACM{N} | grep ID_SERIAL_SHORT`

### Step 2: Create udev rule

```bash
sudo nano /etc/udev/rules.d/99-servo.rules
```

Add two lines of this shape. The two serials below are **examples** — the
boards of one particular rig as of 2026-07-31 — replace them with your own
boards' serials:

```
SUBSYSTEM=="tty", ATTRS{serial}=="5B3D045039", SYMLINK+="ttyACMservoLeft"
SUBSYSTEM=="tty", ATTRS{serial}=="5B61033903", SYMLINK+="ttyACMservoRight"
```

The same serials go into `humanoid_site` as `HAND_USB_SERIAL_LEFT` and
`HAND_USB_SERIAL_RIGHT` — in `control/site_local.py`, or as
`HUMANOID_HAND_USB_SERIAL_LEFT` / `HUMANOID_HAND_USB_SERIAL_RIGHT` in the
environment (the shipped defaults are the two example boards above, so an
unconfigured rig keeps working only if it happens to be that one). The gripper
service reads them into its `HAND_USB_SERIAL` table. If a udev rule is stale
(board swapped, rule never reloaded), the symlink is simply absent and the
service used to fail with "failed to open servo port" (the 07-31 right-gripper
failure); it now falls back to `/dev/serial/by-id/*<serial>*` using those
serials, so keeping them current matters more than keeping the udev rule
current (an empty value disables the fallback for that side).

### Step 3: Reload udev

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

### Step 4: Verify

```bash
ls -la /dev/ttyACMservoLeft /dev/ttyACMservoRight
```

### Step 5: Use

The gripper service defaults to these two symlinks; they can be overridden
per run:

```bash
cd <repo>/control
python humanoid_end_effector_service.py --left-port /dev/ttyACMservoLeft --right-port /dev/ttyACMservoRight
```

and in code the driver is the compiled extension from section 2:

```python
from hardware_bindings.ft_servo import FtServo
servo = FtServo("/dev/ttyACMservoLeft")
```

---

## 5. Hand-eye calibration

Refines camera extrinsics (`T_base_link_to_camera`) and tagged-cube mounting offsets
(`T_obj_link_to_object`) from live robot data. Replaces hard-coded `CAMERA_TRANSFORMS` in
`perception/camera_tag_detector.py`.

**Prerequisites:** robot telemetry on port 9870, cameras on 5555/5556, `tag_cube_0`/`tag_cube_1` bolted to wrists.

**Where the data lives:** everything below reads and writes `control/calibration/`.
That directory is **not shipped** — it is per-rig data, created by the
recording step (the recorder writes next to `humanoid_monitor.py`, whatever the
current directory), and the solver's auto-discovery and default `--output`
point at the same place. See [`../CALIBRATION.md`](../CALIBRATION.md) for why
the internal captures were left out. All commands in this section run from
`<repo>/control`.

### Normal workflow (all defaults)

```bash
# Step 1 — Record
python humanoid_monitor.py --record-handeye

# Step 2 — Solve (auto-picks most recent recording)
python humanoid_handeye_calibration.py

# Step 3 — Apply
python humanoid_monitor.py --handeye-yaml calibration/camera_handeye.yaml
```

### Step 1 — Record

10-second countdown, then records until **Ctrl+C**. Data saves automatically on exit.
Logs go to `calibration/calib_YYYYMMDD_HHMMSS_port{5555,5556}.npz`.

Live counter while recording:
```
[handeye] port5555: 412fr 280valid  |  port5556: 398fr 310valid
```

**During recording:**
1. Move both arms — at least 30° rotation + 50mm translation at each wrist
2. Include **wrist roll** — otherwise cube rotation about the camera axis is unobservable
3. Moderate arm speed — fast motion is downweighted; slow static poses waste time
4. Both cubes visible from >=1 camera simultaneously is ideal

Observability warnings print if motion is insufficient. Aim for `n_valid >= 100` per port.

Override delay: `--handeye-delay 5`

### Step 2 — Solve

Auto-discovers most recent session from `calibration/`. Output: `calibration/camera_handeye.yaml`.

```bash
python humanoid_handeye_calibration.py
```

Expected output:
```
[handeye] Auto-selected logs (prefix 'calib_20260511_143022'): ['calib_..._port5555.npz', 'calib_..._port5556.npz']
[handeye] Stage 1: linear warm-start...
[handeye] Stage 2: Huber refinement...
[handeye] VALID
[handeye] mean=2.3mm / max=8.1mm | rot mean=0.21° max=0.89°
[handeye] Written to calibration/camera_handeye.yaml
```

Hard failure thresholds (YAML written regardless, `calibration_valid: false`):
- `n_valid < 50` — not enough detected frames
- `max_trans_mm > 20` or `max_rot_deg > 5` — solver diverged

**If warm-start warns `max rotation residual > 15°`** — cube physically twisted relative to wrist.
Measure rotation, then provide rotvec init (rad):
```bash
python humanoid_handeye_calibration.py \
    --obj-rot-inits '{"tag_cube_0": [0.0, 0.0, 0.35]}'
```

Key override args:

| Arg | Default | Meaning |
|-----|---------|---------|
| `--logs` | auto | Explicit npz paths (skip auto-discovery) |
| `--output` | `calibration/camera_handeye.yaml` | Output path |
| `--subsample` | 5 | Every Nth frame (reduce for more data, slower solve) |
| `--dq-thresh` | 0.5 | rad/s arm joint velocity half-weight threshold |
| `--v-link-thresh` | 0.3 | m/s wrist Cartesian speed half-weight threshold |

### Step 3 — Apply

```bash
python humanoid_monitor.py --handeye-yaml calibration/camera_handeye.yaml
```

Invalid YAML (`calibration_valid: false`) silently falls back to hard-coded defaults with a warning.

### Inspect a recording

```bash
python3 -c "
import numpy as np, glob
for p in sorted(glob.glob('calibration/*.npz'))[-2:]:
    f = np.load(p, allow_pickle=True)
    N, M = f['obj_valid'].shape
    valid_per_obj = [f['obj_valid'][:, m].sum() for m in range(M)]
    print(f'{p}: {N} frames, telem={f[\"valid_telem\"].mean():.1%}, obj_valid={valid_per_obj}')
"
```

---

## 6. Network

Three roles, which may or may not be three boxes (OPERATIONS.md, section 0):
the **robot computer** runs everything except the plan server; the **GPU
machine** runs the cuRobo plan/MPC server; the **operator console** is any
laptop with ssh and a browser. "Workstation" below and in the scripts'
`--help` means whichever host runs the monitor, the operator and the reach
tool — in this runbook that is the robot computer itself, so `ROBOT_IP` and
`WORKSTATION_IP` stay at their loopback defaults; set them only if you split
those tools onto another host. Ports:

| Port | Transport | Direction | Carries |
|---|---|---|---|
| `9870` | nng pub/sub | robot -> workstation tools | telemetry (joint state, projected gravity, grasp flags) |
| `9873` | nng | high-level controller -> `humanoid_real_env` | nav command (`vx, vy, wz`) |
| `9874` | nng | operator / GUI -> `humanoid_real_env` -> EE service | arm targets (joint or Cartesian) + `ee_action` + gaze targets |
| `9880` | TCP + msgpack | reach client -> plan server (GPU machine) | cuRobo plan / MPC session |
| `5555` / `5556` | TCP JPEG stream | cameras -> monitor | `5555` = LEFT/front datum, `5556` = RIGHT/rear |
| `8080` | HTTP (viser) | monitor UI (also the EE service `--gui`) | operator console; joint-monkey UI uses `8090` |

Detections travel over a Unix socket (`ipc:///tmp/humanoid_detections.sock`),
and EE status over `ipc:///tmp/ee_status.sock`, so monitor, operator, reach
and EE service must share one host. The monitor UI is bound to `0.0.0.0:8080`;
from a laptop, tunnel it: `ssh -L 18081:localhost:8080 <user>@<robot-host>`
then open `localhost:18081`. Make sure routing allows bidirectional traffic
on the ports above between the robot computer, the workstation and the GPU
machine.

**Who talks to whom is configured in `control/humanoid_site.py`**, never in
the scripts. Each value resolves, first hit wins, from
(1) the environment variable `HUMANOID_<NAME>`, (2) `control/site_local.py`
(copy `site_local.example.py`, gitignored), (3) a portable default — loopback
for every address, so a stack you have not configured talks only to itself.

| `humanoid_site` name | Default | Used for |
|---|---|---|
| `ROBOT_IP` | `127.0.0.1` | host running `humanoid_real_env.py`, as seen from workstation tools (`humanoid_monitor.py --robot-ip`, `ee_demo.py`, `humanoid_recv_debug.py`; the monitor's `--tag-camera-host` defaults to it) |
| `WORKSTATION_IP` | `127.0.0.1` | where the robot PUSHES telemetry and subscribes for commands: `humanoid_real_env.py --telemetry-ip / --high-level-controller-ip / --arm-sender-ip` default to it |
| `PLAN_SERVER` | `tcp://127.0.0.1:9880` | cuRobo plan/MPC server endpoint (`humanoid_curobo_client`, `humanoid_plan_server_probe.py --server`) |
| `VICON_IP` | `127.0.0.1` | Vicon tracker host; read only with `--vicon` |
| `CAMERA_SERIALS` | empty | declared here but has no reader in `control/`. The copy that is read is `perception/vs_site.CAMERA_SERIALS` (set in `perception/vs_site_local.py` or `VS_CAMERA_SERIALS`): by the standalone detector `perception/camera_tag_detection.py` and, when it is non-empty, by the camera streaming server (OPERATIONS.md, T1) as the default of `--devices <left-serial> <right-serial>` — so it is how a rig pins its camera -> port order once. An explicit `--devices` still wins; with neither, USB enumeration order decides |
| `HAND_USB_SERIAL_LEFT` / `HAND_USB_SERIAL_RIGHT` | one example rig's boards | the gripper driver boards' USB serials (section 4); the gripper service's `/dev/serial/by-id` fallback when the udev symlinks are absent |
| `RECORDINGS_DIR` | `~/recordings` | default output of `humanoid_mission_recorder.py --out`, `humanoid_arm_forensics.py --out-dir`, `humanoid_camera_view.py --out-dir` |

The port numbers are also declared there (`TELEMETRY_PORT` 9870,
`NAV_COMMAND_PORT` 9873, `ARM_COMMAND_PORT` 9874, `CAMERA_BASE_PORT` 5555) and
since 2026-08-22 they are read on one side of each channel: `humanoid_real_env`
opens its telemetry, nav and arm sockets from them, the monitor's telemetry
subscriber reads `TELEMETRY_PORT`, and the mission recorder reads
`ARM_COMMAND_PORT` and `CAMERA_BASE_PORT`. The peers — the operator, the reach
tool, the gamepad / GUI senders, the bench and calibration tools — still bind
or dial the literals (`humanoid_site.py` lists them per setting), so a
non-default port in `site_local.py` moves real_env away from them. Treat the
ports in the table above as fixed unless you change every call site together
(ARCHITECTURE_MAP.md, "Port / channel map").
