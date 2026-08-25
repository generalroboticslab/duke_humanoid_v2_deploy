"""plan-0 and the MPC tracker must choose a grasp from ONE set (audit 2026-08-06).

The tracker's job is to REFINE the approach plan-0 committed to, not to
re-decide it centimetres from the cube. It could not: `mpc_start(grasp_index=-1)`
auto-picked the nearest candidate out of `ik_curobo.cube_grasp_poses_obj`'s
DEFAULTS — flip=True, beta_degs=(90,), z_above=0 — while plan-0 had planned onto
the robot descriptor's own set, HUMANOID_GRASP_BETAS (75/60/45 deg), flip=False,
z_above=GRASP_Z_ABOVE_M. The two are disjoint in orientation, so "nearest" was
15, 30 or 45 degrees away by construction and the tracker commanded that
re-orientation on its FIRST tick, on the terminal approach.

These pins hold the two halves that make the defect real, both on CPU:
the sets genuinely differ, and the arithmetic that turns the log's opaque
`orientation gap 0.0086` into "15 degrees, 44 mm of flange swing" is right.
"""
import math
import sys
import unittest
from pathlib import Path

# The upstream package is `mj_envs` under humanoid_site.LEGGED_ENV_ROOT. An
# unrelated PyPI distribution of the same name (`mj_envs` 0.5.5, MuJoCo
# environments) can be present in site-packages and shadows it whenever nothing
# has put the upstream checkout first on sys.path -- which depends on which test
# module happened to import humanoid_real_env earlier. Resolve it here so this
# module does not depend on collection order; do not install that PyPI package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_site import LEGGED_ENV_ROOT as _LEV  # noqa: E402
if str(_LEV) not in sys.path:
    sys.path.insert(0, str(_LEV))
_loaded = sys.modules.get("mj_envs")
if _loaded is not None and not str(getattr(_loaded, "__file__", "") or "").startswith(str(_LEV)):
    for _k in [k for k in sys.modules if k == "mj_envs" or k.startswith("mj_envs.")]:
        del sys.modules[_k]

try:
    from mj_envs.tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import (
        cube_grasp_poses_obj as _defaults)
    from mj_envs.tasks.visual_manipulation.curobo.planner import CFG_BY_ROBOT

    _CFG = CFG_BY_ROBOT["v2"]
    _SKIP = None
except Exception as exc:  # noqa: BLE001
    _CFG = None
    _SKIP = f"legged_env_v2 unavailable ({type(exc).__name__}: {exc})"


@unittest.skipIf(_CFG is None, _SKIP)
class GraspCandidateSetTests(unittest.TestCase):

    def test_the_descriptor_exposes_the_generator_the_server_needs(self):
        """The fix rebuilds the MPC's candidates from `cfg`, which was already a
        parameter of MpcSession and entirely unused. If either callable moves,
        the server silently falls back to the defaults."""
        self.assertTrue(callable(getattr(_CFG, "cube_grasp_poses_obj", None)))
        self.assertTrue(callable(getattr(_CFG, "grasp_poses_to_base", None)))

    def test_the_two_candidate_sets_really_are_different(self):
        """If these ever agree, the defect is gone and the fix is dead weight —
        this pin is what would say so."""
        gp, _gq = _CFG.cube_grasp_poses_obj(device="cpu")
        dp, _dq = _defaults(device="cpu")
        self.assertNotEqual(tuple(gp.shape), tuple(dp.shape),
                            "plan-0 and ik_curobo now offer the same count")
        z_plan = {round(float(z), 4) for z in gp[:, 2]}
        z_def = {round(float(z), 4) for z in dp[:, 2]}
        # 08-13 (user): GRASP_Z_ABOVE_M = 0.0 — the grasp line sits AT the
        # cube centre. The 08-12 -0.015 dial (high-stack cancellation)
        # collided with the tip-guard hand z-floor on resting cubes (floor
        # above the grasp pose at 1e6 weight -> 30/30 INFEASIBLE, the leg-2
        # refusal); the table z prior + 1-face z snap own the vertical now.
        # CAUTION from 07-22 (OLD wrist): exact 0 was once IK-infeasible
        # (front_back_close 0% reach) — not re-observed on the new wrist.
        # PRIOR: -0.015 (08-12), +0.005 (08-10), the deep-bite era before.
        # The sets stay disjoint in ORIENTATION (flip and beta families),
        # which is what the 08-06 defect was about.
        # 08-13 night (user): +0.005 — grasp line 5 mm ABOVE the centre
        # (briefly +0.008 the same evening). The ledger showed physical
        # closes ~+10 mm over the commanded line all day with CONTACT every
        # time, and a slightly raised line buys tip-guard floor margin and
        # low-rear-cube envelope room. PRIOR: 0.0 (08-13 midday),
        # -0.015 (08-12, collided with the floor), +0.005 (08-10).
        self.assertEqual(z_plan, {0.005},
                         "plan-0 no longer anchors 5 mm above the centre "
                         "(GRASP_Z_ABOVE_M moved off +0.005)")
        # The client mirrors this dial for grasp-point-relative floor
        # arithmetic (the 08-11 "infinite table 2 cm under the grasping
        # point" rule). A silent divergence would move the unseen-table
        # floor without anyone deciding it.
        import humanoid_curobo_reach as _R
        self.assertEqual({round(_R.GRASP_ANCHOR_ABOVE_M, 4)}, z_plan,
                         "client GRASP_ANCHOR_ABOVE_M no longer matches the "
                         "planner's GRASP_Z_ABOVE_M — update the mirror")
        self.assertEqual(z_def, {0.0},
                         "the ik_curobo default is no longer the exact centre")


class OrientationGapArithmeticTests(unittest.TestCase):
    """No legged_env_v2 needed: this is the conversion the server prints."""

    def test_the_orientation_gap_reads_as_an_angle_and_a_swing(self):
        """The server logged `1 - |<q,q'>|`, which nobody reads as an angle:
        0.0086 / 0.0341 / 0.0761 ARE 15 / 30 / 45 degrees, and at the 0.169 m
        tool offset they are 44 / 87 / 129 mm of lateral flange travel."""
        for gap_deg, d_expect, swing_mm in ((15.0, 0.0086, 44),
                                            (30.0, 0.0341, 87),
                                            (45.0, 0.0761, 129)):
            d = 1.0 - abs(math.cos(math.radians(gap_deg) / 2.0))
            self.assertAlmostEqual(d, d_expect, places=4)
            back = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, 1.0 - d))))
            self.assertAlmostEqual(back, gap_deg, places=3)
            swing = 2000.0 * 0.169 * math.sin(math.radians(gap_deg) / 2.0)
            self.assertAlmostEqual(swing, swing_mm, delta=1.0)


if __name__ == "__main__":
    unittest.main()
