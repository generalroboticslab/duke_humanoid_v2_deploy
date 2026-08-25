"""Pins for the 08-12 dynamic standing waist pin (--pin-waist-standing).

WaistPin is a pure clock-injected state machine: alpha 0 = policy owns the
waist, 1 = pinned at default. Contract under test: 0.5 s zero-command debounce
before any engage, 0.5 s blend-in, immediate 0.3 s blend-out on any drive
command (no step in either direction), and clean re-engage after a walk.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_real_env import WaistPin  # noqa: E402

DT = 0.02


def advance(pin, seconds, cmd_active, t0):
    """Tick the pin at 50 Hz; returns (alphas, end_time)."""
    alphas = []
    t = t0
    for _ in range(round(seconds / DT)):
        t += DT
        alphas.append(pin.update(cmd_active, t, DT))
    return alphas, t


class TestWaistPin(unittest.TestCase):
    def test_debounce_blocks_early_engage(self):
        pin = WaistPin()
        alphas, _ = advance(pin, 0.48, cmd_active=False, t0=0.0)
        self.assertEqual(alphas[-1], 0.0, "engaged before the 0.5 s debounce")

    def test_engage_blend_reaches_full_pin(self):
        pin = WaistPin()
        _, t = advance(pin, 0.5, cmd_active=False, t0=0.0)   # debounce
        alphas, t = advance(pin, 0.5, cmd_active=False, t0=t)  # blend-in window
        self.assertEqual(alphas[-1], 1.0)
        # monotone: a settling robot never sees the pin retreat
        self.assertEqual(alphas, sorted(alphas))

    def test_release_is_immediate_and_fast(self):
        pin = WaistPin()
        _, t = advance(pin, 1.2, cmd_active=False, t0=0.0)
        self.assertEqual(pin.alpha, 1.0)
        alphas, t = advance(pin, DT, cmd_active=True, t0=t)
        self.assertLess(alphas[0], 1.0, "release did not start on the first tick")
        alphas, _ = advance(pin, 0.3, cmd_active=True, t0=t)
        self.assertEqual(alphas[-1], 0.0, "not fully released after 0.3 s")

    def test_mid_blend_reversal_never_steps(self):
        pin = WaistPin()
        _, t = advance(pin, 0.7, cmd_active=False, t0=0.0)   # part-way engaged
        mid = pin.alpha
        self.assertTrue(0.0 < mid < 1.0)
        a1 = pin.update(True, t + DT, DT)                    # drive arrives
        self.assertLess(a1, mid)
        self.assertLess(mid - a1, 0.1, "release step too large for one tick")

    def test_reengage_needs_fresh_debounce(self):
        pin = WaistPin()
        _, t = advance(pin, 1.2, cmd_active=False, t0=0.0)   # pinned
        _, t = advance(pin, 0.4, cmd_active=True, t0=t)      # walk: released
        self.assertEqual(pin.alpha, 0.0)
        alphas, t = advance(pin, 0.48, cmd_active=False, t0=t)
        self.assertEqual(alphas[-1], 0.0, "re-engaged without a fresh debounce")
        alphas, _ = advance(pin, 0.6, cmd_active=False, t0=t)
        self.assertEqual(alphas[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
