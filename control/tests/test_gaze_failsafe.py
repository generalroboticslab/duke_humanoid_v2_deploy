"""Pin the gaze silence failsafe: when the operator stops publishing, the
gimbals return to the origin.

Before 2026-08-08 the gaze reference was latched latest-value-wins with no
timeout at all ("a frozen gaze is benign"). On a Ctrl+C that left both eyes
frozen mid-sweep — pitched 47 deg down, yawed wherever the scan happened to
stop — staring at a cube that had already been picked up. The arms had had a
silence failsafe since forever; the cameras did not.

Tests drive the unbound method against a duck-typed self, so no robot, no CAN
and no policy are needed. What is pinned:

  - a live stream keeps the gimbals exactly where the operator put them
  - silence past _GAZE_CMD_TIMEOUT walks the target back to the origin
  - the return is RATE LIMITED, not a step (the gimbal never sees a jump)
  - it actually converges, and then stays put
  - a stream that comes back mid-return takes control again immediately
  - a malformed packet does NOT hold the failsafe off (the timer is stamped
    only where the clamp succeeded)
  - the failsafe target is the same origin zero_gimbals aims at
"""
import os
import sys
import unittest

import torch

_CONTROL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CONTROL)

import humanoid_real_env as E  # noqa: E402

DT = 0.02
N_CAM = 4


class _Env:
    """Duck-typed stand-in carrying only what the gaze path touches."""

    def __init__(self, gaze=(0.0, 0.0, 0.0, 0.0), last_recv=0.0, active=True):
        self.dt = DT
        self.device = "cpu"
        self.gaze_in_obs = True
        self._cam_home = torch.zeros(N_CAM)
        self._gaze_ref_target = torch.tensor(gaze, dtype=torch.float32)
        self.gaze_reference = self._gaze_ref_target.clone()
        self._last_gaze_recv_time = last_recv
        self._gaze_stream_active = active
        # _apply_gaze_command needs these
        self._cam_joint_lower = torch.full((N_CAM,), -4.71)
        self._cam_joint_upper = torch.full((N_CAM,), 4.71)

    step = E.HumanoidRealEnv._gaze_failsafe_step
    apply = E.HumanoidRealEnv._apply_gaze_command

    def run(self, seconds, now=1000.0):
        """Tick the failsafe for `seconds` of silence at DT."""
        for _ in range(int(round(seconds / DT))):
            self.step(now)
        return self._gaze_ref_target


class GazeFailsafeTests(unittest.TestCase):

    def test_a_live_stream_leaves_the_gimbals_exactly_where_they_were(self):
        env = _Env(gaze=(2.4, -0.82, -1.1, -0.82), last_recv=1000.0)
        before = env._gaze_ref_target.clone()
        for _ in range(100):                       # 2 s of ticks, stream fresh
            self.assertFalse(env.step(1000.0))
        self.assertTrue(torch.equal(env._gaze_ref_target, before))

    def test_silence_past_the_timeout_engages_the_return(self):
        env = _Env(gaze=(2.4, -0.82, -1.1, -0.82), last_recv=1000.0)
        self.assertFalse(env.step(1000.0 + E._GAZE_CMD_TIMEOUT - 0.01))
        self.assertTrue(env.step(1000.0 + E._GAZE_CMD_TIMEOUT))

    def test_the_return_is_rate_limited_not_a_step(self):
        # A scan leaves yaw far off zero. One tick must move it by at most
        # _GAZE_RETURN_RATE * dt — a step would whip the gimbal.
        env = _Env(gaze=(3.1, -0.82, -3.1, -0.82), last_recv=0.0)
        before = env._gaze_ref_target.clone()
        env.step(1000.0)
        moved = (before - env._gaze_ref_target).abs().max().item()
        self.assertAlmostEqual(moved, E._GAZE_RETURN_RATE * DT, places=6)

    def test_it_converges_to_the_origin_and_stays(self):
        env = _Env(gaze=(3.1, -0.82, -3.1, -0.82), last_recv=0.0)
        # Worst joint is 3.1 rad out; allow its rate-limited travel plus slack.
        env.run(3.1 / E._GAZE_RETURN_RATE + 0.5)
        self.assertLess(env._gaze_ref_target.abs().max().item(), 1e-6)
        env.run(1.0)                               # and does not drift past it
        self.assertLess(env._gaze_ref_target.abs().max().item(), 1e-6)

    def test_the_return_never_overshoots_the_origin(self):
        # Sign flip would mean the clamp is wrong; check both directions.
        env = _Env(gaze=(0.004, -0.004, 0.004, -0.004), last_recv=0.0)
        signs = env._gaze_ref_target.sign()
        for _ in range(5):
            env.step(1000.0)
            after = env._gaze_ref_target
            self.assertTrue(torch.all((after.sign() == signs) | (after == 0.0)))

    def test_a_returning_stream_takes_control_back_immediately(self):
        env = _Env(gaze=(3.1, 0.0, 0.0, 0.0), last_recv=0.0)
        env.run(1.0)
        parked = env._gaze_ref_target.clone()
        self.assertLess(parked[0].item(), 3.1)     # it did move
        env.apply([2.0, 0.1, -2.0, 0.1])           # operator relaunched
        self.assertFalse(env.step(env._last_gaze_recv_time))
        self.assertAlmostEqual(env._gaze_ref_target[0].item(), 2.0, places=5)

    def test_a_good_packet_stamps_the_timer(self):
        env = _Env(last_recv=0.0)
        env.apply([0.5, 0.1, -0.5, 0.1])
        self.assertGreater(env._last_gaze_recv_time, 0.0)

    def test_a_malformed_packet_does_not_hold_the_failsafe_off(self):
        # 20 Hz of garbage commands nothing; if it refreshed the timer the
        # cameras would stay frozen forever, which is the bug this replaces.
        for bad in ([1.0, 2.0], [float("nan")] * N_CAM, "not a vector"):
            env = _Env(gaze=(3.1, 0.0, 0.0, 0.0), last_recv=0.0)
            env.apply(bad)
            self.assertEqual(env._last_gaze_recv_time, 0.0, msg=repr(bad))
            self.assertTrue(env.step(1000.0), msg=repr(bad))

    def test_the_failsafe_target_is_the_origin_the_operator_parks_at(self):
        # zero_gimbals publishes gaze_targets = 0; cam joints carry no
        # default_pose entry, so default_joint_pos is 0 there too. If a config
        # ever gave the cams a non-zero default these two would silently
        # disagree and the failsafe would park somewhere the operator never
        # aims at.
        import yaml
        cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "legged_env_bundle", "mj_envs", "deploy", "runs",
            "HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine",
            "env_config.yaml")
        with open(cfg) as f:
            jp = yaml.safe_load(f)["default_pose"]["joint_pos"]
        self.assertNotIn(".*", jp)                 # no catch-all default
        self.assertFalse([k for k in jp if k.startswith("cam_")])


if __name__ == "__main__":
    unittest.main()
