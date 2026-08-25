"""Control-side IPC: the nng publisher / subscriber (`publisher.py`) and the
timer-latency check `sleep_test.cpp`. This package was `control/common` until
2026-08-22; it was renamed so that no control-side import can collide with
`perception/common` or `hardware_bindings/common` (see docs/ARCHITECTURE_MAP.md,
section 2, "Known seam")."""
