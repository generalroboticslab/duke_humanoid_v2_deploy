"""The retract has to return to where the mission STARTED.

WHY THIS EXISTS. Hardware 2026-07-30, first successful cuRobo grasp on the real
robot. The cube was picked up cleanly — and then the arm parked somewhere the
run had never been: the server's goal="home" targets the FK of its own
planning_home, which was then 0.975 rad from the chest home the mission started
at. Measured against the live GPU server: the old retract left the left hand
229.4 mm from the chest home; with the posture sent explicitly, 0.0 mm.

2026-07-31 CHANGED THE GEOMETRY BUT NOT THE CONTRACT. The homes are unified
now — planning_home, power-on and the chest home are all the simulator's
HUMANOID_ARM_JOINT_HOME — so on a FRESH mission home_enc merely confirms what
the server would do anyway. The field stays load-bearing for two reasons:

  * mid-mission the return target is the MEASURED mission start, not the
    nominal home — a round-2 retract must not "return" to round 1's failure
    pose (fa3ddfe fixed exactly that), and the measured start is whatever
    legacy home the arms actually sat at (the 07-30 posture is still accepted);
  * the two home constants live in two repos on two machines; if either moves
    alone, home_enc keeps the retract returning to the posture this mission
    really started from while the guard tests catch the drift.

These tests own the CLIENT half — that the posture crosses the units seam
correctly and only on a retract. The server half was verified against the real
GPU, which is the only place it can be.
"""

from __future__ import annotations

import unittest

try:
    import mujoco

    import humanoid_auto_operator as OP
    import humanoid_curobo_client as C
    from humanoid_model import MJCF_MODEL_PATH

    _MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _MODEL = None
    _SKIP = f"deploy model unavailable ({type(exc).__name__}: {exc})"


def _enc(seq):
    e = {}
    for n, v in zip(C.ARM_JOINTS_L, seq):
        e[n] = float(v)
    for n, v in zip(C.ARM_JOINTS_R, OP.mirror_arm(seq, "right")):
        e[n] = float(v)
    return e


class _Client(C.PlanClient):
    """A PlanClient with the socket replaced by a recorder."""

    def __init__(self, model):
        self.model = model
        self._gen = 0
        self.sent = []

    def _rpc(self, req, timeout_s):
        self.sent.append(req)
        return {"gen": req["gen"], "err": None, "route": None}


@unittest.skipIf(_MODEL is None, _SKIP)
class RetractHomeTests(unittest.TestCase):

    def setUp(self):
        self.cli = _Client(_MODEL)
        self.chest = _enc(OP.CHEST_HOME_JOINTS)

    def _plan(self, **kw):
        self.cli.plan({}, {}, measured_enc=_enc(OP.STAGE_JOINTS["front"]),
                      assignment={"L": "cube"}, **kw)
        return self.cli.sent[-1]

    def test_the_posture_is_sent_when_asked_for(self):
        req = self._plan(goal="home", home_enc=self.chest)
        self.assertIsNotNone(req["home_q"],
                             "the retract went out without a destination")
        self.assertEqual(set(req["home_q"]), set(C.ARM_JOINTS_ALL))

    def test_it_crosses_the_units_seam(self):
        """enc = SIGN * cspace + qpos0, and SIGN[left_wrist_2_joint] = -1. A
        raw copy would send that joint mirrored — the units-seam class this
        whole bridge exists to keep honest."""
        req = self._plan(goal="home", home_enc=self.chest)
        self.assertEqual(req["home_q"],
                         C.enc_to_cspace_seed(self.chest, _MODEL))
        j = "left_wrist_2_joint"
        if C.CSPACE_SIGN.get(j, 1.0) < 0:
            self.assertNotAlmostEqual(req["home_q"][j], self.chest[j], places=3,
                                      msg="the flipped joint was copied raw")

    def test_omitting_it_leaves_the_server_default(self):
        """None means planning_home. Anything else here would silently change
        every other caller of this server."""
        req = self._plan(goal="home")
        self.assertIsNone(req["home_q"])

    def test_a_reach_never_carries_one(self):
        """The field is meaningless on a reach; sending a posture there would
        be a claim about the goal that is not true."""
        req = self._plan(goal="reach")
        self.assertIsNone(req["home_q"])

    def test_the_chest_home_now_IS_the_servers_planning_home(self):
        """REVERSED 2026-07-31. Until then this asserted the two postures were
        FAR apart — that distance was the reason home_enc existed. The one-home
        unification made them identical BY DESIGN (both are the sim's
        HUMANOID_ARM_JOINT_HOME), so the assertion flipped: if they ever drift
        apart again, one side moved its home without the other, and every
        fresh-boot retract quietly stops ending where the mission began.

        The hardcoded dict is the server's planning_home in cspace units,
        verbatim from lev2_master's _REACH_READY_ARM_JOINT_POS — deliberately
        NOT imported, because catching the two repos drifting is the point.
        home_enc stays load-bearing regardless (see the module docstring)."""
        server = {"left_shoulder_1_joint": -1.374, "left_shoulder_2_joint": -0.187,
                  "left_shoulder_3_joint": 0.131, "left_elbow_joint": 1.929,
                  "left_wrist_1_joint": 0.3305, "left_wrist_2_joint": 1.4546,
                  "left_wrist_3_joint": -0.009}
        cs = C.enc_to_cspace_seed(self.chest, _MODEL)
        worst = max(abs(cs[j] - v) for j, v in server.items())
        self.assertLess(worst, 1e-3,
                        "the chest home no longer matches the server's "
                        "planning_home — one side's home moved alone; re-run "
                        "the 07-31 one-home migration on the side that lagged")


if __name__ == "__main__":
    unittest.main()
