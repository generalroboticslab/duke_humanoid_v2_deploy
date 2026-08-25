# Third-party notice — FEETECH serial servo SDK

The C++ sources in this directory other than `ft_servo_ext.cpp` are the FEETECH
serial-bus servo SDK, redistributed here under its MIT licence (see
`LICENSE.FTServo`, Copyright (c) 2024 FTServo).

Upstream: https://gitee.com/ftservo/FTServo_Linux — pinned at commit `064a6db`.

`ft_servo_ext.cpp` and `ft_servo_driver.hpp` are ours: the nanobind binding
layer that exposes the SDK to Python.

## Local modifications to the upstream sources

`SCSerial.cpp` — one bug fix in the port-close path:

```c
/* upstream */                  /* here */
fd = -1;                        if (fd != -1) { close(fd); fd = -1; }
close(fd);
```

Upstream clears the descriptor before closing it, so `close()` is always called
on `-1` and the real file descriptor is never released. Reopening the servo bus
repeatedly leaks one descriptor per open. Every other upstream file in this
directory is byte-identical to the pinned commit.

The upstream sources carry the vendor's comments in Chinese (about 98 lines
across the seven files `HLSCL.cpp`, `HLSCL.h`, `INST.h`, `SCS.cpp`, `SCS.h`,
`SCSerial.cpp`, `SCSerial.h`); they are left untouched, and the files are
UTF-8 here as received (five of them with a byte-order mark). Do not
re-encode or strip them when editing nearby.
