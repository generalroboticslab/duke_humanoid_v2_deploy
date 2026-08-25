"""Solve-on-lock (user 2026-08-03: 'start computing the waypoints the instant
it locks'). SpeculativeSolve launches the whole solve+gate pipeline in a
background thread the first tick every target cube is selected — overlapping
gripper zeroing / the chest-home raise — and plan_reach adopts the result only
while it still describes the world. These tests pin the lifecycle with an
injected solve_fn and scene_builder (no GPU, no server)."""
import threading
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R
    from humanoid_curobo_client import ARM_JOINTS_ALL, PlanServerError

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

# In-envelope targets: the 2026-08-03 hardware run's own selected positions.
POS_A = np.array([0.263, 0.36, 0.044])
POS_B = np.array([0.293, -0.406, 0.076])


class _Rig:
    def __init__(self):
        self.med = {}

    def median(self, key, now):
        v = self.med.get(key)
        return None if v is None else np.asarray(v, float).copy()


class _Tel:
    def fresh(self):
        return {"joint_pos": [0.0] * 31}


def _spec(rig, tel, solve_fn):
    return R.SpeculativeSolve(
        rig, tel, ("a", "b"), None, solve_fn,
        scene_builder=lambda: ({}, {k: rig.med[k] for k in ("a", "b")}))


def _enc():
    return {n: 0.0 for n in ARM_JOINTS_ALL}


@unittest.skipIf(R is None, _SKIP)
class SpeculativeSolveTests(unittest.TestCase):

    def test_launches_once_when_every_cube_is_selected(self):
        calls = []
        done = threading.Event()

        def solve(scene, targets, enc):
            calls.append((dict(targets), dict(enc)))
            done.set()
            return "BUNDLE", "REPORT"

        rig = _Rig()
        spec = _spec(rig, _Tel(), solve)
        spec.maybe_launch(10.0)
        self.assertIsNone(spec.thread, "launched before any cube was selected")
        rig.med["a"] = POS_A
        spec.maybe_launch(10.1)
        self.assertIsNone(spec.thread, "launched with one cube still missing")
        rig.med["b"] = POS_B
        spec.maybe_launch(10.2)
        self.assertIsNotNone(spec.thread, "did not launch with both selected")
        t = spec.thread
        spec.maybe_launch(10.3)
        self.assertIs(spec.thread, t, "relaunched while a solve was running")
        self.assertTrue(done.wait(2.0), "solve_fn never ran in the thread")
        spec.thread.join(2.0)
        self.assertEqual(len(calls), 1)

    def _launched(self, solve):
        rig = _Rig()
        rig.med["a"], rig.med["b"] = POS_A, POS_B
        spec = _spec(rig, _Tel(), solve)
        spec.maybe_launch(10.0)
        self.assertIsNotNone(spec.thread)
        spec.thread.join(2.0)
        return spec

    def test_fresh_result_is_adopted_exactly_once(self):
        spec = self._launched(lambda s, t, e: ("BUNDLE", "REPORT"))
        got = spec.adopt({"a": POS_A + 0.005, "b": POS_B}, _enc())
        self.assertEqual(got, ("BUNDLE", "REPORT"),
                         "a still-true speculation was not adopted")
        self.assertIsNone(spec.adopt({"a": POS_A, "b": POS_B}, _enc()),
                          "adopt is one-shot; a second take must return None")

    def test_a_moved_target_discards_the_route(self):
        spec = self._launched(lambda s, t, e: ("BUNDLE", "REPORT"))
        moved = {"a": POS_A + np.array([0.10, 0.0, 0.0]), "b": POS_B}
        self.assertIsNone(spec.adopt(moved, _enc()),
                          "a cube 100 mm from the solved scene was adopted")

    def test_one_face_noise_does_not_discard_the_route(self):
        """Hardware 2026-08-03: 1-face PnP wanders 30-70 mm at r_xy ~0.5, and
        the first 20 mm bar threw away a solve that had WON its lottery
        during zeroing. Ordinary detection noise must not cost the ticket."""
        spec = self._launched(lambda s, t, e: ("BUNDLE", "REPORT"))
        noisy = {"a": POS_A + np.array([0.04, 0.0, 0.0]),
                 "b": POS_B + np.array([0.0, -0.04, 0.02])}
        self.assertEqual(spec.adopt(noisy, _enc()), ("BUNDLE", "REPORT"),
                         "1-face-noise-sized drift discarded the route")

    def test_a_moved_arm_discards_the_route(self):
        spec = self._launched(lambda s, t, e: ("BUNDLE", "REPORT"))
        enc = _enc()
        enc[ARM_JOINTS_ALL[0]] = 0.1           # > SPEC_SEED_STALE_RAD
        self.assertIsNone(spec.adopt({"a": POS_A, "b": POS_B}, enc),
                          "a seed 0.1 rad from the live arm was adopted")

    def test_a_transport_failure_is_harmless(self):
        def solve(scene, targets, enc):
            raise PlanServerError("server rebooting")

        spec = self._launched(solve)
        self.assertIsNone(spec.adopt({"a": POS_A, "b": POS_B}, _enc()),
                          "a failed speculation must fall back, not adopt")

    def test_a_gateless_solve_is_not_adopted(self):
        spec = self._launched(lambda s, t, e: (None, "REPORT"))
        self.assertIsNone(spec.adopt({"a": POS_A, "b": POS_B}, _enc()))

    def test_warmup_runs_immediately_and_serializes_with_the_real_solve(self):
        """The ~3.5 s once-per-server CUDA-graph capture is paid by a warm-up
        launched AT CONSTRUCTION (during zeroing), and the real solve must not
        share the lockstep REQ socket with it — the rpc lock forces warm-then-
        solve order even when both threads are live."""
        order = []
        warm_gate = threading.Event()

        def warmup():
            order.append("warm_start")
            warm_gate.wait(2.0)
            order.append("warm_end")

        rig = _Rig()
        rig.med["a"], rig.med["b"] = POS_A, POS_B
        spec = R.SpeculativeSolve(
            rig, _Tel(), ("a", "b"), None,
            lambda s, t, e: (order.append("solve"), ("BUNDLE", "REPORT"))[1],
            scene_builder=lambda: ({}, {k: rig.med[k] for k in ("a", "b")}),
            warmup=warmup)
        self.assertTrue(spec.busy(), "warm-up must start at construction")
        spec.maybe_launch(10.0)          # real solve queued behind the lock
        warm_gate.set()
        spec.join_all()
        self.assertEqual(order, ["warm_start", "warm_end", "solve"],
                         "the real solve overlapped the warm-up on the socket")
        self.assertFalse(spec.busy())
        self.assertEqual(spec.adopt({"a": POS_A, "b": POS_B}, _enc()),
                         ("BUNDLE", "REPORT"))

    def test_a_failed_warmup_is_harmless(self):
        def warmup():
            raise PlanServerError("server rebooting")

        rig = _Rig()
        rig.med["a"], rig.med["b"] = POS_A, POS_B
        spec = R.SpeculativeSolve(
            rig, _Tel(), ("a", "b"), None,
            lambda s, t, e: ("BUNDLE", "REPORT"),
            scene_builder=lambda: ({}, {k: rig.med[k] for k in ("a", "b")}),
            warmup=warmup)
        spec.maybe_launch(10.0)
        spec.join_all()
        self.assertEqual(spec.adopt({"a": POS_A, "b": POS_B}, _enc()),
                         ("BUNDLE", "REPORT"),
                         "a dead warm-up must not poison the real solve")


if __name__ == "__main__":
    unittest.main()
