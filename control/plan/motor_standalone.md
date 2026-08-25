# Motor → standalone `motor_bindings` pkg

## Why

`control/motor/` currently:
- 12 files, small, tightly coupled to control's CMake (`py_motor.py:10` hard-imports `DukeHumanoidV2.control.build.motor.my_ext`).
- Imports referenced from 18 control files, 0 sibling repos (`grep` over `argus_hardware`, `visual_servoing`, `legged_env_*`, `mjlab`).
- Motor + control change together every PR. Folder bloat, not coupling.

**Decision:** split `motor_bindings` (generic C++ binding) from `py_motor` (Duke-specific constants). Pkg = binding only. Duke-specific stays in control/. Submodule deferred until 2nd consumer.

**Rejected:**
- **Submodule now** — no 2nd consumer. Submodule = coupling cost without payoff.
- **Flat pkg (A)** — pollutes `motor_bindings` with `R00..R06` (Duke joint IDs). Motor pkg shouldn't know one robot's joint map.
- **In-tree `third_party/motor/` vendored copy** — duplicates code, drifts. No win over standalone pkg + pip.

## Assumptions

- Python 3.12, Linux/CAN, nanobind stable ABI. Motor.hpp uses `<linux/can.h>`, `SCHED_FIFO` → Linux-only build.
- Deps external: Eigen3, fmt, CSerialPort (itas109), EasyProfile, nanobind, TBB. All `find_package`-able.
- One owner. No semver yet → start `0.1.0`, tag after first standalone build.
- Pkg installed via `pip install -e .` (dev) or `pip install` (release). `.so` lands in site-packages.
- EasyProfile + CSerialPort available system-wide on dev machine. If not → `FetchContent` fallback in CMake.

## Scope

**In:**
- New sibling repo `<control>/hardware_bindings/` with: `CMakeLists.txt`, `pyproject.toml`, `src/`, `README.md`, `.gitignore`.
- Move C++ sources: `motor.cpp`, `motor.hpp`, `my_ext.cpp` → `src/`, renamed `my_ext.cpp` → `motor_bindings.cpp`.
- Move `latency_logger.hpp` → `src/` (only motor user, stays inside).
- Move tests: `motor_test.cpp`, `motor_test2.cpp`, `can_test.cpp` → `tests/`.
- Standalone build: produces `motor_bindings*.so` importable as `import motor_bindings`.
- Refactor `py_motor.py`: drop `DukeHumanoidV2.control.build.*` path. `from motor_bindings import MotorParams`. Stay in `control/motor/` (or rename dir) — Duke-specific constants (`R00..R06`, `MAX_TORQUE`, torque constants) belong here.
- Update 18 control importers: `from motor.py_motor import ...` → `from motor.py_motor import ...` (unchanged — wrapper stays in control/, just its internal import drops the hardcoded path).
- Delete `control/build/motor/`. Drop `add_subdirectory(motor)` from `control/CMakeLists.txt`. Delete `control/motor/CMakeLists.txt`.
- project notes update: motor location, build command, import paths.

**Out:**
- Wiring 2nd consumer (argus_hardware, legged_env_*). Blocks submodule justification but not extraction.
- Splitting latency_logger.
- API changes: `MotorParams`, `MAX_TORQUE`, joint IDs unchanged.
- CI for new repo.
- Wheel build / manylinux.
- Cross-platform (Linux-only assumption holds).

## Design invariants

- **No silent-break:** `from motor.py_motor import X` works in all 18 control sites post-migration. Verified by `grep -r "from.*motor\|import.*motor" control/`.
- **No build-dir coupling:** `py_motor.py` import path = installed pkg name. Never `DukeHumanoidV2.control.build.*`.
- **No source duplication:** `control/motor/{cpp,hpp}` deleted post-migration. One canonical copy in `motor_bindings/src/`.
- **Reversible:** until submodule step, `git rm -r motor_bindings && git checkout` restores. No `.gitmodules` entry.
- **Pkg = binding only:** `motor_bindings/` contains zero Duke-specific symbols. No `R00..R06`, no `MAX_TORQUE`. Generic CAN-motor primitives only.

## Dependency chain (strict order)

1. Create `<control>/hardware_bindings/` (no parent repo coupling yet — just a sibling dir).
2. Copy C++ sources → `src/`. Rename `my_ext.cpp` → `motor_bindings.cpp`. Update `#include "my_ext..."` if any (none — checked).
3. Write `CMakeLists.txt`: `nanobind_add_module(motor_bindings ...)`, find Eigen3/CSerialPort/fmt/nanobind/EasyProfile/TBB.
4. Write `pyproject.toml`: scikit-build-core or setuptools + nanobind. `name = "motor_bindings"`.
5. Standalone build: `cd <control>/hardware_bindings && pip install -e .` → produces `motor_bindings*.so` in site-packages.
6. Refactor `control/motor/py_motor.py`:
   - Drop `sys.path.insert(0, ...)` hack.
   - Replace `from DukeHumanoidV2.control.build.motor import my_ext` → `import motor_bindings`.
   - Replace `from DukeHumanoidV2.control.build.motor.my_ext import MotorParams` → `from motor_bindings import MotorParams`.
7. Smoke test standalone: `python -c "import motor_bindings; print(motor_bindings.MotorParams)"`.
8. Migrate control:
   - `control/CMakeLists.txt`: drop `add_subdirectory(motor)`.
   - Delete `control/motor/CMakeLists.txt`, `control/motor/my_ext.cpp`, `control/motor/motor.{cpp,hpp}`, `control/motor/latency_logger.hpp`, `control/motor/can_test.cpp`, `control/motor/motor_test*.cpp`.
   - Keep `control/motor/__init__.py`, `control/motor/py_motor.py` (refactored).
   - `rm -rf control/build/motor`.
9. Rebuild control: CMake configures without `motor/` subdir, `pip install -e motor_bindings` provides the ext.
10. Verify: `python -c "from motor.py_motor import MAX_TORQUE, R00"`, run `humanoid_test_motor.py` smoke.
11. `git init` `<control>/hardware_bindings/`, first commit + tag `0.1.0`.
12. project notes, add entry: motor location, build (`pip install -e <control>/hardware_bindings`), import path.

## File moves (concrete)

```
<control>/motor/motor.cpp              → <control>/hardware_bindings/src/motor.cpp
<control>/motor/motor.hpp              → <control>/hardware_bindings/src/motor.hpp
<control>/motor/my_ext.cpp             → <control>/hardware_bindings/src/motor_bindings.cpp
<control>/motor/latency_logger.hpp     → <control>/hardware_bindings/src/latency_logger.hpp
<control>/motor/motor_test.cpp         → <control>/hardware_bindings/tests/motor_test.cpp
<control>/motor/motor_test2.cpp        → <control>/hardware_bindings/tests/motor_test2.cpp
<control>/motor/can_test.cpp           → <control>/hardware_bindings/tests/can_test.cpp

# New
<control>/hardware_bindings/CMakeLists.txt                       (NEW)
<control>/hardware_bindings/pyproject.toml                       (NEW)
<control>/hardware_bindings/README.md                            (NEW)
<control>/hardware_bindings/.gitignore                           (NEW)

# Refactored, stays in control/
<control>/motor/py_motor.py            (MODIFY: drop sys.path hack, switch import)

# Deleted
<control>/motor/CMakeLists.txt         (DELETE)
<control>/build/motor/                 (DELETE: cmake build artifact)
```

## Decisions made (locked)

| Q | Choice |
|---|---|
| Ext name | `motor_bindings` |
| Pkg name | `motor_bindings` |
| Layout | B (split) — pkg = binding only, Duke-specific in control/ |
| Location | `<control>/hardware_bindings/` (sibling repo, not in-tree) |
| Submodule | Deferred until 2nd consumer |

## Risks

- **High:** EasyProfile or CSerialPort not find_package-able on dev machine → CMake fails at step 3. Mitigation: `FetchContent` fallback in CMake (itas109/CSerialPort, plus EasyProfile source). Verify deps installed before step 3.
- **Med:** missed importer pattern (e.g. `from control.motor...` not `from motor...`). Mitigation: full grep + smoke import test step 10.
- **Med:** ABI mismatch after `.so` relocation. Mitigation: rebuild + run `humanoid_test_motor.py` smoke.
- **Low:** `my_ext.cpp` internal `#include "my_ext..."` (none expected — checked). Mitigation: grep before rename.
- **Low:** `py_motor.py` re-imports motor symbols other than `MotorParams`. Mitigation: read full file before refactor; preserve all symbol re-exports.

## Time

~2 hr: 20min dep audit, 40min CMake+pyproject, 20min refactor+rename, 30min migration+smoke, 10min MEMORY+commit.

## STOP

Approve / modify before execute.
