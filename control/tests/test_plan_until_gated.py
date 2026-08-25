"""Re-solve until OUR gates pass — the acceptance test is our model, not cuRobo's.

WHY THIS EXISTS. The user's requirement, stated plainly: within the arm's
reachable range, cuRobo should be able to produce a self-collision-free route to
the cube. The old code could not deliver that even when such a route existed,
because it asked exactly once and treated a gate rejection as final.

That is the wrong shape for a randomized-seed search. cuRobo restarts from fresh
seeds every call and its clearance varies run to run — hardware 2026-07-30 saw
0.6 mm, then 8.0 mm, then 39.5 mm on comparable targets, at 0.13 s per solve.
One unlucky sample ended the mission.

Nothing here relaxes a gate. A route still has to clear the floor on our mesh
model before a packet is sent; what changed is that failing to FIND such a route
on the first attempt is no longer failing the mission.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import mujoco

    import humanoid_auto_operator as OP
    import humanoid_curobo_client as C
    import humanoid_curobo_reach as R
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _MODEL = None
    _SKIP = f"deploy model unavailable ({type(exc).__name__}: {exc})"

FRONT = None if _MODEL is None else np.asarray(OP.STAGE_JOINTS["front"], float)
SIDE = None if _MODEL is None else np.asarray(OP.STAGE_JOINTS["side"], float)


def _bundle(frames_left):
    """A RouteBundle whose Gate A ground truth is the FK of its own last
    waypoint, so only clearance can decide the verdict."""
    right = np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"), float)
    q = np.stack([np.concatenate([np.asarray(f, float), right])
                  for f in frames_left])
    d = mujoco.MjData(_MODEL)
    d.qpos[:] = _MODEL.qpos0
    for i, name in enumerate(C.ARM_JOINTS_ALL):
        jid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[_MODEL.jnt_qposadr[jid]] = q[-1, i]
    mujoco.mj_kinematics(_MODEL, d)
    site = d.site_xpos[mujoco.mj_name2id(
        _MODEL, mujoco.mjtObj.mjOBJ_SITE, "end_effector_L_site")].copy()
    return C.RouteBundle(
        joint_names=tuple(C.ARM_JOINTS_ALL), q_enc=q, dt=0.5,
        reaches={"end_effector_L_site": {"target": site,
                                         "cands": site.reshape(1, 3), "err": 0.0}},
        controlled_joints=tuple(C.ARM_JOINTS_L), assignment={"L": "cube"},
        goal="reach", solve_s=0.1)


def _good():
    """front -> side: every configuration clears 51 mm on this model."""
    return _bundle([FRONT + (SIDE - FRONT) * t for t in np.linspace(0, 1, 12)])


def _bad():
    """Left arm lerped onto its own mirrored posture: -92.1 mm penetration."""
    crossed = np.asarray(OP.mirror_arm(OP.STAGE_JOINTS["front"], "right"), float)
    return _bundle([FRONT + (crossed - FRONT) * t for t in np.linspace(0, 1, 12)])


class _Client:
    """Returns a canned sequence of verdicts; None means INFEASIBLE."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0
        self.assignments = []

    def plan(self, *a, **kw):
        self.assignments.append(kw.get("assignment"))
        i = min(self.calls, len(self.sequence) - 1)
        self.calls += 1
        item = self.sequence[i]
        return None if item is None else item()


def _run(client, **kw):
    return R.plan_until_gated(client, {}, {}, measured_enc=None, model=_MODEL,
                              max_rate=0.3, clearance_floor_m=0.010, **kw)


@unittest.skipIf(_MODEL is None, _SKIP)
class RetryTests(unittest.TestCase):

    def test_a_gated_route_is_accepted_immediately(self):
        cli = _Client([_good])
        bundle, report = _run(cli)
        self.assertIsNotNone(bundle)
        self.assertTrue(report["ok"])
        self.assertEqual(cli.calls, 1, "a good first solve must not be re-asked")

    def test_a_rejected_route_is_re_solved_not_abandoned(self):
        """THE fix. The old code returned the first bundle and let the caller
        refuse; one unlucky sample ended the mission even though the very next
        seed would have passed."""
        cli = _Client([_bad, _bad, _good])
        bundle, report = _run(cli)
        self.assertIsNotNone(bundle, "gave up while a gated route was available")
        self.assertTrue(report["ok"])
        self.assertEqual(cli.calls, 3)

    def test_infeasible_and_rejection_are_both_retried(self):
        """They are different verdicts — cuRobo failing to solve, and our model
        refusing what it solved — but both mean 'ask again with a fresh seed'."""
        cli = _Client([None, _bad, None, _good])
        bundle, _ = _run(cli)
        self.assertIsNotNone(bundle)
        self.assertEqual(cli.calls, 4)

    def test_it_gives_up_after_the_retry_budget_and_returns_the_last_report(self):
        """The caller has to be able to PRINT why every attempt failed, so the
        last report comes back rather than a bare None."""
        cli = _Client([_bad])
        bundle, report = _run(cli)
        self.assertIsNone(bundle)
        self.assertIsNotNone(report)
        self.assertFalse(report["ok"])
        # the bad fixture trips a CHEAP gate, so since the 2026-08-03
        # short-circuit the expensive C sweep is skipped on it — the report
        # still names the failing gate, which is what the caller prints
        self.assertTrue(any(not g["ok"] for g in report["gates"].values()))
        self.assertIn("skipped", report["gates"]["C_clearance"])
        self.assertEqual(cli.calls, R.PLAN_RETRIES)

    def test_the_gate_floor_is_still_enforced_verbatim(self):
        """Retrying must never become 'eventually accept something worse'. The
        benign route clears 51 mm; demand 60 and every attempt must fail."""
        cli = _Client([_good])
        bundle, report = R.plan_until_gated(
            cli, {}, {}, measured_enc=None, model=_MODEL, max_rate=0.3,
            clearance_floor_m=0.060)
        self.assertIsNone(bundle)
        self.assertFalse(report["gates"]["C_clearance"]["ok"])

    def test_the_route_is_stretched_before_it_is_gated(self):
        """Gate E lives in stretch(), and streaming a route that was gated
        un-stretched would exceed the rate the certificate assumed."""
        cli = _Client([_good])
        bundle, _ = _run(cli)
        self.assertLessEqual(bundle.max_joint_rate(), 0.3 * 1.001)


if __name__ == "__main__":
    unittest.main()
