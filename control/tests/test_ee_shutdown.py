"""Pin the EE service's Ctrl+C contract (2026-08-09).

History: the original shutdown hook was
    loop.add_signal_handler(sig, lambda: asyncio.create_task(sys.exit(0)))
sys.exit evaluated inside the loop callback, main_loop's finally still ran
(the grip guard fired), but the interpreter then blocked forever joining the
serial/pynng SDKs' non-daemon threads and every further Ctrl+C was swallowed
— the only way out was pkill -9, mid-mission, with a payload in hand.

Contract now (subprocess-tested, no hardware needed — the service runs with
both sides unhealthy on bogus ports):
  * one SIGINT  -> graceful stop through the grip-guard finally, process EXITS
  * two SIGINTs -> process exits no matter what (second is an os._exit(130))
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest

CONTROL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE = os.path.join(CONTROL_DIR, "humanoid_end_effector_service.py")
BOGUS = ["--left-port", "/dev/nonexistent_ee_L",
         "--right-port", "/dev/nonexistent_ee_R"]
STARTUP_TIMEOUT_S = 15.0
EXIT_TIMEOUT_S = 6.0


def _real_service_running() -> bool:
    """The test binds the real ipc request socket; never fight a live service."""
    probe = subprocess.run(
        ["pgrep", "-f", "humanoid_end_effector_servic[e]"],
        capture_output=True, text=True)
    mine = str(os.getpid())
    pids = [p for p in probe.stdout.split() if p and p != mine]
    return bool(pids)


class EEShutdownTests(unittest.TestCase):

    def _spawn(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, SERVICE, *BOGUS],
            cwd=CONTROL_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        for line in proc.stdout:
            if "Service loop active" in line:
                return proc
            if time.monotonic() > deadline:
                break
        proc.kill()
        proc.wait()
        proc.stdout.close()
        self.fail("service never reached its main loop")

    def _await_exit(self, proc: subprocess.Popen, what: str) -> int:
        try:
            return proc.wait(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self.fail(f"service survived {what} for {EXIT_TIMEOUT_S}s — "
                      "the Ctrl+C wall is back")
        finally:
            proc.stdout.close()

    def setUp(self):
        if _real_service_running():
            self.skipTest("a live EE service holds the request socket")

    def test_one_sigint_exits_through_the_grip_guard(self):
        proc = self._spawn()
        proc.send_signal(signal.SIGINT)
        code = self._await_exit(proc, "one SIGINT")
        self.assertEqual(code, 0)

    def test_mashed_ctrl_c_still_terminates(self):
        """Two rapid SIGINTs: whichever wins the race (graceful stop or the
        second-signal hard exit), the process must DIE."""
        proc = self._spawn()
        proc.send_signal(signal.SIGINT)
        proc.send_signal(signal.SIGINT)
        code = self._await_exit(proc, "two SIGINTs")
        self.assertIn(code, (0, 130))


if __name__ == "__main__":
    unittest.main()
