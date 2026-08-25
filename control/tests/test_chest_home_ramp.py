"""The chest-home return commands JOINTS — the home is not IK-attainable.

WHY THIS FILE CHANGED SHAPE (2026-07-31). It used to guard a Cartesian ramp:
raise_to_chest_home published (CHEST_HOME_EE, CHEST_HOME_QUAT) and the tests
pinned the ramp's step size, because publishing an EE goal raw let the
robot-side IK sweep the whole distance at its own cap — the documented 07-18
hardware incident, "arms shot out to the sides at startup".

That whole design rested on the chest home being REACHABLE by the robot's IK,
which it was by construction: the old CHEST_HOME_JOINTS was solved offline FROM
that 6-DoF target (residual 0.8 mm). The one-home migration inverted the
derivation — the joints are now the simulator's HUMANOID_ARM_JOINT_HOME,
imported verbatim, and CHEST_HOME_EE is merely their FK — and attainability did
not survive the inversion.

Measured with the deployment's own solver (BatchedAnalyticalIK, constructed as
humanoid_real_env.py does), 2000 steps from each recognised start: the worst-arm
EE never reaches CHEST_HOME_TOL_M, and every run lands 1.512 rad (86.6 deg) from
CHEST_HOME_JOINTS with wrist_3 pinned on its limit. The same harness aimed at
the OLD target converges to 1.2 mm. The cause is mink's CollisionAvoidanceLimit,
a hard 50 mm keep-out: at this home shoulder_3~wrist_1 measures 16.5 mm and
wrist_1~wrist_3 10.2 mm, so the QP will not steer INTO the fold. With
collision_min_distance=0 the same target converges to 0.7 mm. The posture is
HOLDABLE — starting there, mink holds it to 4e-4 rad — just not attainable
through a Cartesian command.

So the home is commanded as JOINTS now, and these tests pin that contract. The
codebase already made this exact move once: the journey's side<->tuck glide left
Cartesian on 07-21 for the identical failure ("straight-line EE + IK left the arm
up to 62 deg off the true power-on posture"). The chest-home path was simply
never moved with it.

WHAT REPLACED THE RAMP PROPERTY. In joint mode the rate travels IN the packet
and the receiver lerps toward the target, so there is no jump to ramp away from
— the anti-jump guarantee moved from "small steps between packets" to "the
commanded rate is stated and is the audited one". That is asserted below, along
with the property that matters most now: the commanded vector is EXACTLY
CHEST_HOME_JOINTS, mirrored per side, with no solver in the loop to reinterpret
it.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import humanoid_auto_operator as OP
    import humanoid_curobo_reach as R

    _SKIP = None
except Exception as exc:  # noqa: BLE001
    R = None
    _SKIP = f"reach tool unavailable ({type(exc).__name__}: {exc})"

# A start that is far from the home in every joint — the 07-30 chest posture,
# which is where the robot is physically parked through the migration.
START_L = None if R is None else list(OP.POWERON_JOINTS_LEGACY_0730)


def _jpos(left7):
    jp = [0.0] * 31
    for i, arm in enumerate(("left", "right")):
        jp[13 + 7 * i:20 + 7 * i] = [float(v) for v in OP.mirror_arm(list(left7), arm)]
    return jp


class _Tel:
    """Telemetry whose arms follow the last commanded joint target exactly —
    a perfect follower, which isolates the COMMAND from any tracking lag."""

    def __init__(self, start=None):
        self.left = list(START_L if start is None else start)

    def fresh(self):
        return {"projected_gravity": [0.0, 0.0, -1.0],
                "arm_fault": [False, False],
                "joint_pos": _jpos(self.left)}


class _Pub:
    def __init__(self, tel):
        self.tel = tel
        self.sent = []

    def publish(self, packet):
        slots = (packet or {}).get("arm_targets") or {}
        if not slots:
            return
        self.sent.append(slots)
        if "left" in slots:                     # the arm follows perfectly
            self.tel.left = list(slots["left"]["joint_pos"])


class _Rig:
    def tick(self, tel, packet=None):
        return packet if packet is not None else {}


class _Det:
    def poll(self, rig):
        pass


@unittest.skipIf(R is None, _SKIP)
class ChestHomeJointCommandTests(unittest.TestCase):

    def setUp(self):
        self.tel = _Tel()
        self.pub = _Pub(self.tel)
        why = R.raise_to_chest_home(self.pub, self.tel, _Rig(), _Det())
        self.assertIsNone(why, f"the return refused: {why}")
        self.assertTrue(self.pub.sent, "nothing was ever published")

    def test_every_packet_is_joint_mode_not_cartesian(self):
        """THE contract. A single ee_pos slot here means the command went back
        through mink, which cannot reach this posture — the arm would land
        86.6 deg away and the tool would report success on the EE tolerance."""
        for k, slots in enumerate(self.pub.sent):
            for arm, slot in slots.items():
                self.assertIn("joint_pos", slot, f"packet {k} {arm} is not joint mode")
                self.assertNotIn("ee_pos", slot, f"packet {k} {arm} carries a "
                                                 f"Cartesian target")
                self.assertNotIn("ee_quat", slot)

    def test_the_commanded_vector_is_exactly_the_sim_home(self):
        """Not 'close to' — the whole point of the migration is joint-for-joint
        identity with the simulator, so the command must be the constant
        itself, mirrored per side, with nothing in between to reinterpret it."""
        for arm in ("left", "right"):
            want = [float(v) for v in OP.mirror_arm(OP.CHEST_HOME_JOINTS, arm)]
            for k, slots in enumerate(self.pub.sent):
                self.assertEqual(slots[arm]["joint_pos"], want,
                                 f"packet {k} {arm} is not CHEST_HOME_JOINTS")

    def test_the_rate_is_stated_and_is_the_audited_one(self):
        """In joint mode the receiver lerps at the rate carried in the packet,
        so THIS is the anti-jump guarantee that replaced the old step-size
        check. An unstated rate would let the receiver apply its own."""
        for slots in self.pub.sent:
            for arm, slot in slots.items():
                self.assertIn("rate", slot)
                self.assertAlmostEqual(slot["rate"], OP.STAGE_RATE, places=9)

    def test_it_actually_arrives_and_stops(self):
        """Arrival is judged in JOINT space now (the EE tolerance was what let
        a Cartesian run 86.6 deg off report success), and the loop must end
        when it arrives rather than run out the timeout."""
        last = np.asarray(self.pub.sent[-1]["left"]["joint_pos"], float)
        got = np.asarray(self.tel.left, float)
        self.assertLessEqual(float(np.max(np.abs(got - last))),
                             OP.JOURNEY_POSTURE_TOL_RAD)
        self.assertLess(len(self.pub.sent), CHEST_HOME_TICKS_MAX,
                        "the loop ran to its timeout instead of converging")

    def test_a_fresh_boot_costs_one_tick(self):
        """Since 07-31 power-on IS this posture, so the common case must be a
        no-op — an operator who sees a real move on a fresh boot is looking at
        a robot that did not power on where it claims to."""
        tel = _Tel(start=OP.CHEST_HOME_JOINTS)
        pub = _Pub(tel)
        why = R.raise_to_chest_home(pub, tel, _Rig(), _Det())
        self.assertIsNone(why)
        self.assertEqual(len(pub.sent), 1)


CHEST_HOME_TICKS_MAX = 10_000 if R is None else int(R.CHEST_HOME_WAIT_S / R.PUB_DT)


@unittest.skipIf(R is None, _SKIP)
class ChestHomeRefusalTests(unittest.TestCase):

    def test_silence_until_the_arms_are_measured(self):
        """No encoders means no way to know whether the arms have arrived —
        and commanding blind is how a posture nobody verified gets streamed."""
        class _NoEnc(_Tel):
            def fresh(self):
                d = super().fresh()
                d["joint_pos"] = [0.0] * 4
                return d

        tel = _NoEnc()
        pub = _Pub(tel)
        R.CHEST_HOME_WAIT_S, keep = 0.4, R.CHEST_HOME_WAIT_S
        try:
            why = R.raise_to_chest_home(pub, tel, _Rig(), _Det())
        finally:
            R.CHEST_HOME_WAIT_S = keep
        self.assertIsNotNone(why)
        self.assertEqual(pub.sent, [], "published without measured encoders")

    def test_an_arm_fault_stops_the_return(self):
        class _Fault(_Tel):
            def fresh(self):
                d = super().fresh()
                d["arm_fault"] = [False, True]
                return d

        why = R.raise_to_chest_home(_Pub(_Fault()), _Fault(), _Rig(), _Det())
        self.assertIn("ARM FAULT", why or "")

    def test_the_timeout_message_does_not_blame_the_ik(self):
        """It used to say 'check that real_env is running with --use-ik'. There
        is no IK in this path any more, and sending an operator to look at the
        wrong subsystem costs a bench session."""
        class _Stuck(_Tel):
            def fresh(self):
                return {"projected_gravity": [0.0, 0.0, -1.0],
                        "arm_fault": [False, False],
                        "joint_pos": _jpos(START_L)}   # never follows

        R.CHEST_HOME_WAIT_S, keep = 0.3, R.CHEST_HOME_WAIT_S
        try:
            why = R.raise_to_chest_home(_Pub(_Stuck()), _Stuck(), _Rig(), _Det())
        finally:
            R.CHEST_HOME_WAIT_S = keep
        self.assertIsNotNone(why)
        self.assertNotIn("use-ik", why)
        self.assertIn("watchdog", why)


if __name__ == "__main__":
    unittest.main()
