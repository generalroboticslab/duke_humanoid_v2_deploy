# ft_servo C++ Nanobind Extension

## Why
`st_servo/ft_servo.py` uses pyserial. 200 Hz × 4 servos = 800 blocking `select()`/s, GIL held each call. C++ wrapper + background poll thread → cached reads, no serial on hot path. Expected: 15–30% CPU reduction.

## Status: VERIFIED — both ports + IDs 1/2 working

---

## File layout

> 2026-08-24: the command blocks further down were written for the old `control/ft_servo/` layout; the sources and the built `.abi3.so` now live in `control/hardware_bindings/ft_servo/` (run the scripts from `control/hardware_bindings/`), and `ft_servo_simple_test` is no longer a build target.

```
control/hardware_bindings/ft_servo/   (2026-08-24: the divergent control/ft_servo/ fork was removed; its scan.py moved here)
├── INST.h / SCS.h / SCS.cpp / SCSerial.h / SCSerial.cpp / HLSCL.h / HLSCL.cpp
│   — copied from st_servo/reference/FTServo_Linux/src/
│   — SCSerial.cpp: fixed SDK bug in end(): was fd=-1; close(fd) → close(fd); fd=-1
├── ft_servo_driver.hpp      — C++ driver
├── ft_servo_ext.cpp         — nanobind bindings
├── ft_servo_simple_test.cpp — C++ hardware test binary
├── test.py                  — combined Python hardware test (no args)
├── scan.py                  — scan bus for servo IDs
├── change_id.py             — change servo ID (uses Python ft_servo.py)
├── CMakeLists.txt
└── __init__.py
control/hardware_bindings/CMakeLists.txt — add_subdirectory(ft_servo) after motor
```

---

## API (ft_servo_ext.FtServo)

**Motion** (GIL released):
- `set_position(id, pos, speed=0, acc=50, torque=500)`
- `set_positions(ids, pos, speed, acc, torque)` — sync write
- `set_speed(id, speed, acc=50, torque=500)`
- `set_speeds(ids, speeds, acc=50, torque=500)` — SyncWriteSpe
- `set_mode(id, mode)` / `set_modes(ids, mode)` — 0=pos, 1=wheel, 2=torque; mode→0 stops wheel + 50ms settle
- `enable_torque(id, on)` / `enable_torques(ids, on)`

**Not bound** (use `st_servo/ft_servo.py` directly):
- `write_id`, `set_position_offset` — EPROM methods, still via Python

**Cached reads** (no serial, no GIL release):
- `get_position(id)` / `get_positions(ids)`
- `get_speed(id)` / `get_speeds(ids)`
- `get_load(id)` / `get_loads(ids)` — 0–1000 (0.1% units)

**Direct reads** (GIL released):
- `read_position(id)` / `read_speed(id)` / `read_load(id)`
- `get_voltage(id)` / `get_temperature(id)`
- `ping(id)` → id or -1
- `scan(start_id=0, end_id=253)` → list of found IDs

**Poll control**:
- `start_poll(ids, interval_us=5000)` / `stop_poll()` / `close()`

---

## Poll thread design

- `syncReadBegin(n, 6, 5)` — 6 rx bytes/servo: `[pos_l, pos_h, spd_l, spd_h, load_l, load_h]` at reg 56 (HLSCL_PRESENT_POSITION_L). Contiguous → one TX covers pos+spd+load.
- Per iter: `syncReadPacketTx` → per-ID `syncReadPacketRx` + 3× `syncReadRxPacketToWrod(15)`
- `negBit=15`: HLS sign-magnitude encoding
- Mutex order: `bus_mutex_` → `cache_mutex_` (never reversed)
- GIL: `gil_scoped_release` on all serial methods; cached getters hold mutex only

---

## Build

```bash
cd control/build && ninja ft_servo_ext ft_servo_simple_test
```

## Test sequence

```bash
# 1. Import check (no hardware)
python -c \
  "import sys; sys.path.insert(0,'build/ft_servo'); from ft_servo_ext import FtServo; print(dir(FtServo))"

# 2. Scan bus
python ft_servo/scan.py /dev/ttyACMservoLeft
python ft_servo/scan.py /dev/ttyACMservoRight

# 3. Full hardware test
python ft_servo/test.py

# 4. Change servo ID (if needed — uses Python ft_servo.py)
python ft_servo/change_id.py /dev/ttyACMservoRight 1 2
```

---

## New Servo Setup

**1. Connect servo to right port, scan for ID:**
```bash
python ft_servo/scan.py /dev/ttyACMservoRight
```

**2. If servo ID conflicts (same as another on that bus) — change it:**
```bash
python ft_servo/change_id.py <port> <old_id> <new_id>
```
Note: `change_id.py` uses `st_servo/ft_servo.py` (Python) because EPROM methods are not bound in the C++ extension. C++ driver does not support `write_id`.

**3. Add to `test.py` config:**
```python
LEFT_PORT  = "/dev/ttyACMservoLeft"
RIGHT_PORT = "/dev/ttyACMservoRight"
LEFT_IDS  = [1]        # add new IDs as needed
RIGHT_IDS = [2]        # add new IDs as needed
```

**4. Run full test:**
```bash
python ft_servo/test.py
```

**Current wiring (2026-05-07):**
- Left arm: 1 servo, ID=1, port `/dev/ttyACMservoLeft`
- Right arm: 1 servo, ID=2, port `/dev/ttyACMservoRight`

---

## Assumptions / fallback

- HLS3915 supports `INST_SYNC_READ` (0x82). Ping -1 on open = verify with `python st_servo/ft_servo.py` first.
- Sync-read 6 bytes (pos+spd+load) requires regs 56-61 contiguous — confirmed from register map.
  If load values wrong: fall back to per-ID `ReadLoad()` calls in poll loop instead of extending rx_bytes.
- `HLSCL::End = 0` (little-endian default).
- Single `FtServoDriver` instance per UART bus.

## Scope exclusions

- `st_servo/ft_servo.py` — not modified
- EPROM methods (`write_id`, `set_position_offset`) — still via Python
- Windows — POSIX termios only
