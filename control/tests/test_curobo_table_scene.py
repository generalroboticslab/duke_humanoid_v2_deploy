"""The registered-table -> cuRobo-obstacle wiring, and the world declaration.

WHY A SEPARATE FILE FROM test_curobo_bridge.py: that one is scoped to the
client/protocol/gates — the things that decide whether a route may move the
robot. This is the layer above: how the WORLD MODEL is built, which is the other
way a safe-looking route kills something. A route can pass all four gates and
still drive the hand through a table the planner was never told about.

The failure this file exists to prevent is specifically silent. Every number
here — 1.2 m, 0.6 m, 0.03 m, the 15 mm half-thickness offset — is authored once
in visual_servoing/tagged_bodies/table/ and read by two consumers: the monitor
(which draws the slab) and this tool (which hands it to cuRobo as an obstacle).
Nothing downstream can notice if a consumer retypes one of them slightly wrong;
the plan just comes back "collision-free" against a table that is 15 mm or
90 degrees away from the real one.

2026-08-13 (user: "THE TABLE IS A HEIGHT SENSOR, NOT AN OBSTACLE — IN THE
PLANNER TOO"): a registered table no longer enters the scene dict at all. A
modelled slab made a resting cube nearly ungraspable (the lower finger rack
must occupy the very air the slab forbids; 30/30 INFEASIBLE on every leg that
could decode the table tags). What a seen, sane table does now is feed the
TABLE-RESTING Z PRIOR — a target over it gets grasp z = surface + half cube,
its obstacle box moves with it, and rig._z_prior_keys names the targets that
carry it — and the hand z-floor. The same registered geometry, the same
refusals and the same latch therefore still matter, and the scene tests below
pin THAT contract: tables absent from the cuboids, present in the prior.

Skipped, not failed, when the deploy model or visual_servoing is unavailable.
"""

from __future__ import annotations

import contextlib
import io
import math
import unittest

import numpy as np

try:
    import humanoid_curobo_reach as R

    _SKIP = None
    _A = R.TABLE_CONFIGS["lab_table_a"]
except Exception as exc:  # noqa: BLE001
    R = None
    _A = None
    _SKIP = f"reach tool / registered tables unavailable ({type(exc).__name__}: {exc})"

# The PHYSICAL orientation of a table whose tags lie flat facing up. Since
# visual_servoing 8bb9628 (08-11) the table body frame matches grasp_cube:
# +z points UP out of the top surface, origin at the top-surface centre, slab
# at z in [-0.030, 0] — so a flat table IS identity now. (Before the re-frame
# +z pointed INTO the slab and flat meant a 180 deg roll; these tests carried
# that convention until the registration flipped under them.)
Q_FLAT = np.array([1.0, 0.0, 0.0, 0.0])
TOP = np.array([0.55, 0.0, -0.15])          # top surface, 15 cm below base origin


def _mul(a, b):
    """Hamilton product, wxyz — a then b applied in the a frame."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _yaw(deg):
    a = math.radians(deg) / 2.0
    return np.array([math.cos(a), 0.0, 0.0, math.sin(a)])


class _Args:
    """Only the fields build_scene / _check_world read."""

    # cubes defaults to ONE valid name: _check_world also validates the grasp
    # set (1-2 cubes, no repeats) since dual-arm landed, and a world-declaration
    # test must not fail for an unrelated reason.
    def __init__(self, tables=(), table=(), no_table_world=False,
                 cubes=("grasp_cube_40mm",)):
        self.tables = tuple(tables)
        self.table = tuple(table)
        self.no_table_world = no_table_world
        self.cubes = tuple(cubes)


class _Rig:
    """Stub rig: canned table poses, no cubes, no sockets."""

    def __init__(self, poses=None):
        self._poses = poses or {}

    def table_pose(self, key, now):
        return self._poses.get(key)

    def median(self, key, now):
        return np.array([0.4, 0.1, -0.1])

    def latest(self, key, now):
        return None


@unittest.skipIf(R is None, _SKIP)
class SlabTransformTests(unittest.TestCase):

    def test_the_slab_centre_sits_below_the_top_surface(self):
        """The body origin is the TOP surface and body +z points DOWN into the
        slab, so the collision box centre is half a thickness deeper — not at
        the detected origin. Getting this wrong puts the obstacle 15 mm out with
        every other number looking perfect."""
        c = R.table_cuboid(_A, TOP, Q_FLAT)
        self.assertAlmostEqual(c["pose"][2], TOP[2] - _A.thickness_m / 2.0, places=9)
        self.assertLess(c["pose"][2], TOP[2], "slab centre must be BELOW the top")

    def test_the_half_thickness_offset_rotates_with_the_table(self):
        """A tilted table is the case that separates `t + R @ p` from `t + p`.
        A shimmed or sloped table is entirely plausible in a lab, and the naive
        version silently keeps offsetting along base z."""
        tilt = _mul(np.array([math.cos(math.radians(10)),
                              math.sin(math.radians(10)), 0.0, 0.0]), Q_FLAT)
        c = R.table_cuboid(_A, TOP, tilt)
        naive = TOP + np.array([0.0, 0.0, _A.thickness_m / 2.0])
        self.assertGreater(np.linalg.norm(np.asarray(c["pose"][:3]) - naive),
                           1e-4, "the offset was not rotated into the base frame")

    def test_yaw_lives_in_the_quaternion_and_never_in_the_dims(self):
        """dims are along the cuboid's OWN axes, so a yawed table keeps
        [0.6, 1.2, 0.03] and carries the rotation in the quaternion. Guards
        against 'fixing' orientation by swapping length and width, which is
        right at exactly two angles and wrong everywhere else."""
        flat = R.table_cuboid(_A, TOP, Q_FLAT)
        for deg in (30.0, 90.0, 137.0):
            with self.subTest(yaw=deg):
                c = R.table_cuboid(_A, TOP, _mul(_yaw(deg), Q_FLAT))
                self.assertEqual(c["dims"], flat["dims"])
                self.assertAlmostEqual(c["pose"][2], flat["pose"][2], places=9)
                self.assertNotAlmostEqual(
                    abs(float(np.dot(c["pose"][3:], flat["pose"][3:]))), 1.0,
                    places=6, msg="the yaw did not reach the quaternion")

    def test_geometry_comes_from_the_registration_not_from_this_module(self):
        """No slab number may be retyped here: the cuboid must track the
        registration. Verified by mutating the config and watching the output
        follow, which a hardcoded 0.6/1.2/0.03 could not do."""
        slab = _A.slab_cuboid()
        self.assertEqual(R.table_cuboid(_A, TOP, Q_FLAT)["dims"], slab["dims"])
        original = _A.thickness_m
        try:
            _A.thickness_m = 0.05
            c = R.table_cuboid(_A, TOP, Q_FLAT)
            self.assertAlmostEqual(c["pose"][2], TOP[2] - 0.025, places=9)
            self.assertAlmostEqual(c["dims"][2], 0.05, places=9)
        finally:
            _A.thickness_m = original


@unittest.skipIf(R is None, _SKIP)
class WorldDeclarationTests(unittest.TestCase):
    """Exactly one world declaration, and it has to name something real."""

    def test_each_single_declaration_is_accepted(self):
        for args in (_Args(tables=("lab_table_a",)),
                     _Args(table=(0.5, 0.0, -0.2, 0.6, 1.2, 0.03)),
                     _Args(no_table_world=True)):
            with self.subTest(args=vars(args)):
                self.assertIsNone(R._check_world(args))

    def test_zero_or_several_declarations_are_refused(self):
        for args in (_Args(),
                     _Args(tables=("lab_table_a",), no_table_world=True),
                     _Args(tables=("lab_table_a",),
                           table=(0.5, 0.0, -0.2, 0.6, 1.2, 0.03)),
                     _Args(table=(0.5, 0.0, -0.2, 0.6, 1.2, 0.03),
                           no_table_world=True)):
            with self.subTest(args=vars(args)):
                self.assertIn("exactly one", R._check_world(args) or "")

    def test_a_short_manual_table_is_refused(self):
        self.assertIn("6 numbers",
                      R._check_world(_Args(table=(0.5, 0.0, -0.2))) or "")

    def test_an_unregistered_table_name_is_refused_and_lists_the_real_ones(self):
        why = R._check_world(_Args(tables=("kitchen_table",))) or ""
        self.assertIn("kitchen_table", why)
        self.assertIn("lab_table_a", why)      # actionable, not just "unknown"


@unittest.skipIf(R is None, _SKIP)
class SceneAssemblyTests(unittest.TestCase):

    def test_seen_tables_stay_out_of_the_scene_and_feed_the_z_prior(self):
        """Until 2026-08-13 this test pinned both tables INTO the scene under
        their own names. The user then ordered the table out of the planner
        world ("a height sensor, not an obstacle"): a slab under a resting
        cube forbids the very air the lower finger rack must occupy, and every
        leg that could decode the table tags went 30/30 INFEASIBLE. So the pin
        is now the inverse — neither seen table may reach the cuboids — while
        the two jobs the names still do are checked in the same breath: the
        retract looks tables up BY NAME to latch them (both must be written),
        and the slab the cube sits over must feed the z prior (grasp z =
        surface + half cube, obstacle box moved with it, key recorded in
        rig._z_prior_keys). Table b is 0.9 m off the cube's xy — outside its
        footprint — so the prior must come from a, not from whichever slab
        happens to be listed last."""
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT, 0.4),
                    "lab_table_b": (TOP + np.array([0.0, 0.9, 0.0]), Q_FLAT, 0.5)})
        latch: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            scene, targets = R.build_scene(
                _Args(tables=("lab_table_a", "lab_table_b")), rig,
                ("grasp_cube_40mm",), latched=latch)
        self.assertNotIn("lab_table_a", scene, "a seen table re-entered the "
                         "planner world as an obstacle")
        self.assertNotIn("lab_table_b", scene)
        self.assertIn("grasp_cube_40mm", scene, "the cube obstacle vanished")
        self.assertEqual(set(latch), {"lab_table_a", "lab_table_b"},
                         "the latch must be written under each table's name")
        resting_z = TOP[2] + 0.020                 # 40 mm cube, centre
        self.assertEqual(rig._z_prior_keys, {"grasp_cube_40mm"})
        self.assertAlmostEqual(float(targets["grasp_cube_40mm"][2]),
                               resting_z, places=9)
        self.assertAlmostEqual(scene["grasp_cube_40mm"]["pose"][2],
                               resting_z, places=9,
                               msg="the obstacle box did not follow the prior")

    def test_a_declared_but_unseen_table_refuses_instead_of_planning(self):
        """The whole point of requiring an explicit world declaration: a table
        we SAID was there and never measured must not quietly become an empty
        world, which is the one state --no-table-world exists to make someone
        type out on purpose."""
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT, 0.4)})     # b never seen
        with self.assertRaises(R.TableNotSeen) as ctx:
            R.build_scene(_Args(tables=("lab_table_a", "lab_table_b")),
                          rig, ("grasp_cube_40mm",))
        self.assertEqual(ctx.exception.name, "lab_table_b")
        self.assertIn("--bodies", str(ctx.exception))        # names the likely cause


@unittest.skipIf(R is None, _SKIP)
class JourneyOptionalTableTests(unittest.TestCase):
    """08-10 night ("the gripper still been struck by the border of the
    table"): journey legs get --tables so the slab guards its own EDGES, which
    the horizontal z-floor cannot. That needs two departures from the standing
    contract, both pinned here: a leg that cannot see its declared table
    proceeds WITHOUT the slab (a rear-cube stop faces away from it), and any
    remembered pose from before the base last walked is discarded (a
    base-frame slab from a destroyed stop is a WRONG hard obstacle).

    2026-08-13: the slab left the planner world altogether (the table is a
    height sensor — see the module docstring), so "WITHOUT the slab" now
    means without its Z PRIOR, and a stale latch would plant that prior on a
    surface that is not there. The same two departures still decide whether
    a leg gets the prior, and the tests below pin them through
    rig._z_prior_keys and the corrected target z instead of the scene dict."""

    def test_an_unseen_table_is_skipped_not_fatal_when_optional(self):
        """The rear-cube stop faces away from table b: the leg must proceed,
        loudly, and the table it CAN see must keep doing its job. Before
        2026-08-13 that job was "stay a slab in the scene"; now it is the z
        prior, so the pin is: no raise, the SEEN table a corrects the cube's
        grasp z to its surface + half cube, neither table is a cuboid, and
        the skip is printed."""
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT, 0.4)})     # b never seen
        with contextlib.redirect_stdout(io.StringIO()) as out:
            scene, targets = R.build_scene(
                _Args(tables=("lab_table_a", "lab_table_b")), rig,
                ("grasp_cube_40mm",), optional_tables=True)
        self.assertNotIn("lab_table_a", scene)
        self.assertNotIn("lab_table_b", scene)
        self.assertIn("grasp_cube_40mm", scene, "the cube obstacle vanished")
        self.assertEqual(rig._z_prior_keys, {"grasp_cube_40mm"},
                         "the SEEN table must still feed the z prior")
        self.assertAlmostEqual(float(targets["grasp_cube_40mm"][2]),
                               TOP[2] + 0.020, places=9)
        self.assertIn("WITHOUT the slab", out.getvalue(),
                      "the skip must never be quiet")

    def test_a_junk_fit_is_skipped_not_fatal_when_optional(self):
        wobble = R.TABLE_POSE_SPREAD_REFUSE_DEG + 5.0
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT, wobble)})
        with contextlib.redirect_stdout(io.StringIO()):
            scene, targets = R.build_scene(_Args(tables=("lab_table_a",)),
                                           rig, ("grasp_cube_40mm",),
                                           optional_tables=True)
        self.assertNotIn("lab_table_a", scene,
                         "a junk-orientation slab was kept as a hard obstacle")
        # 2026-08-13: tables never enter the scene, so the line above alone
        # no longer proves anything — the junk fit must not feed the z prior
        # either (the same 07-30 fit put the surface 18 mm above the cube).
        self.assertEqual(rig._z_prior_keys, set(),
                         "a junk-orientation slab fed the z prior")
        self.assertAlmostEqual(float(targets["grasp_cube_40mm"][2]), -0.1,
                               places=9, msg="the vision z was not kept")

    def test_a_latch_from_before_the_walk_is_discarded(self):
        """Leg 1 latched the slab, the robot walked, leg 2 cannot see it: the
        latch must NOT resurrect a slab in leg-1 coordinates. Optional mode
        proceeds slab-less; strict mode must refuse rather than certify a
        route against geometry that is not there."""
        rig = _Rig({})
        rig._base_moved_t = 50.0                 # the walk ended at t=50
        stale = {"lab_table_a": (TOP, Q_FLAT, 0.4, 10.0)}   # latched at t=10
        with contextlib.redirect_stdout(io.StringIO()) as out:
            scene, targets = R.build_scene(_Args(tables=("lab_table_a",)),
                                           rig, ("grasp_cube_40mm",),
                                           latched=stale, optional_tables=True)
        self.assertNotIn("lab_table_a", scene,
                         "a slab latched before the walk survived it")
        # 2026-08-13: the slab never enters the scene, so the purge is now
        # observable through the z prior — a leg-1 surface must not set the
        # leg-2 grasp height.
        self.assertEqual(rig._z_prior_keys, set(),
                         "a slab latched before the walk fed the z prior")
        self.assertAlmostEqual(float(targets["grasp_cube_40mm"][2]), -0.1,
                               places=9, msg="the vision z was not kept")
        self.assertNotIn("lab_table_a", stale, "the dead latch must be purged")
        self.assertIn("predates the last base motion", out.getvalue())
        with self.assertRaises(R.TableNotSeen):
            R.build_scene(_Args(tables=("lab_table_a",)), rig,
                          ("grasp_cube_40mm",),
                          latched={"lab_table_a": (TOP, Q_FLAT, 0.4, 10.0)})

    def test_a_latch_from_after_the_walk_still_serves(self):
        """The invalidation must not overreach: a slab latched at THIS stop
        (after the last base motion) is exactly what the latch is for. Since
        2026-08-13 "serves" means the remembered pose feeds the z prior — the
        table is hidden behind the reaching arm, not gone — so the pin is the
        corrected grasp z (surface + half cube), the recorded key, the kept
        latch entry and the printed age, with no slab in the cuboids."""
        rig = _Rig({})
        rig._base_moved_t = 50.0
        fresh = {"lab_table_a": (TOP, Q_FLAT, 0.4, 60.0)}   # latched at t=60
        with contextlib.redirect_stdout(io.StringIO()) as out:
            scene, targets = R.build_scene(_Args(tables=("lab_table_a",)),
                                           rig, ("grasp_cube_40mm",),
                                           latched=fresh, optional_tables=True)
        self.assertNotIn("lab_table_a", scene)
        self.assertIn("lab_table_a", fresh,
                      "a post-walk latch was thrown away with the stale ones")
        self.assertEqual(rig._z_prior_keys, {"grasp_cube_40mm"},
                         "a post-walk latch stopped feeding the z prior")
        self.assertAlmostEqual(float(targets["grasp_cube_40mm"][2]),
                               TOP[2] + 0.020, places=9)
        self.assertAlmostEqual(scene["grasp_cube_40mm"]["pose"][2],
                               TOP[2] + 0.020, places=9)
        self.assertIn("not visible right now", out.getvalue())
        self.assertNotIn("WITHOUT the slab", out.getvalue(),
                         "a served latch must not be reported as a skip")

    def test_walk_era_table_sightings_never_reach_the_pose(self):
        """StaticPoseTrack's window is 2 s — long enough to straddle the
        stillness settle and reach back into the walk, where every sample is
        in a destroyed base frame. The rig passes its base-motion epoch as
        min_t; samples at or before it must not vote."""
        t = R.StaticPoseTrack("lab_table_a")
        for k in range(4):                       # mid-walk sightings
            t.ingest(TOP + [0.5, 0.0, 0.0], Q_FLAT, 100.0 + 0.1 * k)
        self.assertIsNotNone(t.pose(100.4), "test premise: enough samples")
        self.assertIsNone(t.pose(100.4, min_t=100.35),
                          "walk-era samples voted through the fence")


@unittest.skipIf(R is None, _SKIP)
class StaticPoseTrackTests(unittest.TestCase):

    def test_opposite_sign_quaternions_do_not_cancel(self):
        """q and -q are the SAME rotation and an AprilTag fit returns either, so
        a mixed-sign set must not collapse. The medoid compares with |dot|, which
        is blind to the double cover — no hemisphere-alignment step to get wrong."""
        t = R.StaticPoseTrack("t")
        for q in ([1, 0, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0], [-1, 0, 0, 0]):
            t.ingest([0.5, 0.0, -0.1], q, 100.0)
        pos, quat, worst = t.pose(100.0)
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=9)
        self.assertAlmostEqual(abs(float(quat[0])), 1.0, places=6)
        self.assertLess(worst, 1e-6)

    def test_a_minority_of_bad_fits_cannot_move_the_answer(self):
        """THE reason orientation is a medoid rather than a mean. Hardware
        2026-07-30 produced a window spreading 78.5 deg; a mean is dragged by
        whichever samples are worst, while a medoid can only ever return a pose
        the sensor actually reported."""
        t = R.StaticPoseTrack("t")
        good = Q_FLAT
        for _ in range(7):
            t.ingest([0.5, 0.0, -0.1], good, 100.0)
        for deg in (60.0, -75.0, 78.5):                 # a minority of junk fits
            t.ingest([0.5, 0.0, -0.1], _mul(_yaw(deg), good), 100.0)
        _, quat, worst = t.pose(100.0)
        self.assertGreater(abs(float(np.dot(quat, good))), 1 - 1e-9,
                           "the estimate drifted off the majority pose")
        self.assertGreater(worst, 20.0, "the spread must still expose the junk")

    def test_the_reported_spread_is_measured_from_the_estimate(self):
        """A clean window must report a small spread, or the refusal threshold
        would fire on healthy data."""
        t = R.StaticPoseTrack("t")
        for _ in range(6):
            t.ingest([0.5, 0.0, -0.1], Q_FLAT, 100.0)
        self.assertLess(t.pose(100.0)[2], 1e-6)

    def test_a_ninety_degree_outlier_shows_up_in_the_spread(self):
        t = R.StaticPoseTrack("t")
        for _ in range(5):
            t.ingest([0.5, 0.0, -0.1], Q_FLAT, 100.0)
        t.ingest([0.5, 0.0, -0.1], _mul(_yaw(90.0), Q_FLAT), 100.0)
        _, _, worst = t.pose(100.0)
        self.assertGreater(worst, R.TABLE_POSE_SPREAD_WARN_DEG,
                           "a 90 deg mis-fit must exceed the warning threshold")

    def test_nan_samples_are_dropped_and_too_few_is_no_pose(self):
        t = R.StaticPoseTrack("t")
        t.ingest([np.nan, 0.0, 0.0], Q_FLAT, 100.0)
        t.ingest([0.5, 0.0, -0.1], [np.nan, 0, 0, 0], 100.0)
        self.assertEqual(len(t.hist), 0)
        for _ in range(2):
            t.ingest([0.5, 0.0, -0.1], Q_FLAT, 100.0)
        self.assertIsNone(t.pose(100.0), "2 samples must not confirm a pose")
        t.ingest([0.5, 0.0, -0.1], Q_FLAT, 100.0)
        self.assertIsNotNone(t.pose(100.0))

    def test_the_window_expires(self):
        t = R.StaticPoseTrack("t", window_s=1.0)
        for _ in range(4):
            t.ingest([0.5, 0.0, -0.1], Q_FLAT, 100.0)
        self.assertIsNotNone(t.pose(100.5))
        self.assertIsNone(t.pose(102.0), "stale samples must stop confirming")

    def test_position_is_a_median_so_one_wild_sample_cannot_move_the_table(self):
        t = R.StaticPoseTrack("t")
        for _ in range(6):
            t.ingest([0.5, 0.0, -0.1], Q_FLAT, 100.0)
        t.ingest([3.0, 3.0, 3.0], Q_FLAT, 100.0)
        pos, _, _ = t.pose(100.0)
        np.testing.assert_allclose(pos, [0.5, 0.0, -0.1], atol=1e-9)


@unittest.skipIf(R is None, _SKIP)
class HandFloorTests(unittest.TestCase):
    """The floor that was already there.

    cuRobo's build_curobo_planner applies hand_z_floor=+0.060 m at weight 1e6
    unless told otherwise, and HUMANOID_CFG's empty planner_kwargs ACCEPTS that
    default rather than disabling it — confirmed from the server's own boot log
    ("24 gripper spheres >= 0.060 m (base), weight 1e+06"). The pelvis stands
    roughly 0.7 m up, so that line falls around the top of a 0.75 m lab table:
    exactly the height "put the cube on the table" produces. Below it the cost
    fights the reach and says nothing about why.
    """

    def test_a_floor_at_or_above_the_target_is_reported(self):
        why = R.check_hand_floor(target_z=0.05, server_floor=0.06)
        self.assertIsNotNone(why)
        self.assertIn("--hand-z-floor", why)       # actionable
        self.assertIn("INFEASIBLE", why)           # names the symptom

    def test_a_target_well_above_the_floor_is_silent(self):
        self.assertIsNone(R.check_hand_floor(target_z=0.20, server_floor=0.06))

    def test_an_explicitly_disabled_floor_is_silent(self):
        self.assertIsNone(R.check_hand_floor(target_z=-0.40, server_floor=None))

    def test_the_suggested_floor_follows_the_work_surface_when_known(self):
        why = R.check_hand_floor(target_z=-0.10, server_floor=0.06,
                                 surface_z=-0.15)
        self.assertIn(f"{-0.15 + R.HAND_FLOOR_MARGIN_M:.4f}", why)

    def test_without_a_surface_the_suggestion_falls_back_to_the_target(self):
        why = R.check_hand_floor(target_z=-0.10, server_floor=0.06)
        self.assertIn(f"{-0.10 + R.HAND_FLOOR_MARGIN_M:.4f}", why)

    def test_the_floor_is_derived_from_the_registered_tables(self):
        """The number was always derivable — registered geometry plus a tag pose
        put the work surface in the base frame — so nobody should be reading it
        off a printout and restarting a GPU server by hand. Highest surface wins:
        a floor below any work surface stops guarding against diving under it."""
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT, 0.4),
                    "lab_table_b": (TOP + np.array([0.0, 0.9, 0.06]), Q_FLAT, 0.5)})
        got = R.hand_floor_for_tables(rig, ("lab_table_a", "lab_table_b"))
        self.assertAlmostEqual(got, TOP[2] + 0.06 + R.HAND_FLOOR_MARGIN_M, places=9)

    def test_a_surface_above_the_target_is_clamped_not_obeyed(self):
        """The hardware case, 2026-07-30: the table fitted 18 mm ABOVE the cube
        standing on it, so the derived floor sat above the grasp pose and cuRobo
        returned INFEASIBLE five times — after the tool had printed its own
        prediction that it would. A surface measured above an object resting on
        it is a WRONG surface, and a wrong surface must not veto the reach."""
        rig = _Rig({"lab_table_a": (np.array([0.84, -0.28, 0.0656]), Q_FLAT, 1.2)})
        got = R.hand_floor_for_tables(rig, ("lab_table_a",), target_z=0.0483)
        self.assertLess(got, 0.0483, "the floor must end up BELOW the target")
        self.assertAlmostEqual(got, 0.0483 - R.HAND_FLOOR_TARGET_GAP_M, places=9)

    def test_a_healthy_surface_is_still_used_verbatim(self):
        """The clamp must not fire on a normal table, or it would quietly
        disable the protection it is guarding."""
        rig = _Rig({"lab_table_a": (np.array([0.55, 0.0, -0.15]), Q_FLAT, 0.5)})
        got = R.hand_floor_for_tables(rig, ("lab_table_a",), target_z=-0.10)
        self.assertAlmostEqual(got, -0.15 + R.HAND_FLOOR_MARGIN_M, places=9)

    def test_a_small_cube_on_a_table_is_clamped_without_blaming_the_fit(self):
        """The clamp and the WRONG-FIT diagnosis are two different things, and
        collapsing them sends the operator to inspect a slab that is drawn
        perfectly correctly.

        Geometry, not a defect: a 60 mm cube resting on a table has its centre
        30 mm above the surface, while the gripper-sphere margin is 23 mm. The
        floor therefore lands 7 mm below the grasp pose — inside the 10 mm slack
        band, so it still has to be clamped, because a 1e6-weight cost that close
        to the goal fights the reach. But the surface is 30 mm BELOW the target,
        which is exactly where a table under a cube belongs.
        """
        top = -0.150
        target_z = top + 0.030                     # 60 mm cube, centre
        rig = _Rig({"lab_table_a": (np.array([0.45, 0.10, top]), Q_FLAT, 0.5)})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            got = R.hand_floor_for_tables(rig, ("lab_table_a",),
                                          target_z=target_z)
        said = out.getvalue()
        # The clamped floor is the HIGHER of the target cap and the surface
        # tip guard — dial-robust on purpose: at the 08-10 40/30 mm gaps the
        # guard bound (the raw cap landed under the wood, the table-strike
        # pin); at the 08-11 "2 cm under the grasp point" dial (gap 15 mm
        # from centre) the cap itself sits above the guard for an on-table
        # 60 mm cube, which is strictly MORE table clearance.
        expect = max(target_z - R.HAND_FLOOR_TARGET_GAP_M,
                     top + R.HAND_FLOOR_TIP_MARGIN_M)
        self.assertAlmostEqual(got, expect, places=9)
        self.assertLess(got, target_z, "the floor must end up BELOW the target")
        self.assertGreaterEqual(got, top + R.HAND_FLOOR_TIP_MARGIN_M - 1e-9,
                                "the floor dipped below the surface tip guard")
        self.assertIn("NOT a bad fit", said)
        self.assertNotIn("WRONG", said)
        self.assertNotIn("viser", said, "no slab to go inspect — the fit is fine")

    def test_a_noisy_low_target_cannot_drag_the_floor_into_the_wood(self):
        """THE 08-10 night table strike: cube f's 1-face fix read +0.019 on a
        table whose top sits around +0.015, the cap said 'floor = target - 40
        mm' = 35 mm UNDER the surface, and with --no-table-world the floor is
        the ONLY table protection — every fetch drove the fingers into the
        slab. A verifiably real surface must win over a target reading that
        claims to be inside the wood."""
        top = 0.0150
        rig = _Rig({"lab_table_a": (np.array([0.45, 0.10, top]), Q_FLAT, 0.5)})
        with contextlib.redirect_stdout(io.StringIO()):
            got = R.hand_floor_for_tables(rig, ("lab_table_a",),
                                          target_z=0.019)
        self.assertAlmostEqual(got, top + R.HAND_FLOOR_TIP_MARGIN_M, places=9,
                               msg="the floor followed a junk reading under "
                                   "the table top")

    def test_a_wrong_fit_surface_never_applies_the_tip_guard(self):
        """The other branch stays as it was: a surface measured ABOVE the
        object resting on it is a WRONG surface, and bounding the floor by
        junk would re-create the 07-30 INFEASIBLE veto the clamp exists to
        prevent."""
        rig = _Rig({"lab_table_a": (np.array([0.84, -0.28, 0.0656]), Q_FLAT,
                                    1.2)})
        with contextlib.redirect_stdout(io.StringIO()):
            got = R.hand_floor_for_tables(rig, ("lab_table_a",),
                                          target_z=0.0483)
        self.assertAlmostEqual(got, 0.0483 - R.HAND_FLOOR_TARGET_GAP_M,
                               places=9,
                               msg="a junk surface bounded the floor and can "
                                   "veto the reach again")

    def test_a_surface_above_the_target_still_says_the_fit_is_wrong(self):
        """The other side of the same split: when the surface really is measured
        at or above an object resting on it, that IS a defective fit and the
        message has to keep saying so, or the 78-deg-wobble case loses the one
        line that explains it."""
        rig = _Rig({"lab_table_a": (np.array([0.84, -0.28, 0.0656]), Q_FLAT, 1.2)})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.hand_floor_for_tables(rig, ("lab_table_a",), target_z=0.0483)
        said = out.getvalue()
        self.assertIn("WRONG", said)
        self.assertIn("viser", said)

    def test_an_unstable_table_fit_refuses_instead_of_planning(self):
        """78.5 deg of orientation wobble inside one window is what the hardware
        produced. A 1.2 x 0.6 m slab placed at a junk orientation is a WRONG
        obstacle — the route would be certified against geometry that is not
        there, which is worse than no obstacle at all."""
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT, 78.5)})
        with self.assertRaises(R.TablePoseUnstable) as ctx:
            R.build_scene(_Args(tables=("lab_table_a",)), rig, ("grasp_cube_40mm",))
        self.assertAlmostEqual(ctx.exception.spread_deg, 78.5, places=6)
        self.assertIn("occluded", str(ctx.exception))      # names a likely cause

    def test_a_merely_noisy_fit_still_plans(self):
        """Between the warn and refuse thresholds the fit is usable; refusing
        there would block ordinary runs. Until 2026-08-13 "usable" showed as
        the slab entering the scene; now the table is a height sensor, so a
        noisy-but-sane fit must still be ACCEPTED — no raise, and the z prior
        it feeds lands on the target — while staying out of the cuboids."""
        rig = _Rig({"lab_table_a": (TOP, Q_FLAT,
                                    R.TABLE_POSE_SPREAD_REFUSE_DEG - 1.0)})
        with contextlib.redirect_stdout(io.StringIO()):
            scene, targets = R.build_scene(_Args(tables=("lab_table_a",)),
                                           rig, ("grasp_cube_40mm",))
        self.assertNotIn("lab_table_a", scene)
        self.assertEqual(rig._z_prior_keys, {"grasp_cube_40mm"},
                         "a merely noisy fit was refused for the z prior")
        self.assertAlmostEqual(float(targets["grasp_cube_40mm"][2]),
                               TOP[2] + 0.020, places=9)

    def test_no_table_still_sends_a_floor_derived_from_the_target(self):
        """None means 'keep whatever the server has', and the server keeps values
        ACROSS RUNS. Hardware 2026-07-30: a --tables run left +0.0886 behind, the
        next --no-table-world run reached for a cube at +0.023, cuRobo contorted
        around the stale floor, and the left elbow came within 0.6 mm of the
        torso. Gate C caught it — but a gate catching a self-inflicted wound is
        not a design."""
        got = R.hand_floor_for_tables(_Rig({}), (), target_z=0.023)
        self.assertAlmostEqual(got, 0.023 - R.HAND_FLOOR_TARGET_GAP_M, places=9)
        self.assertLess(got, 0.023)

    def test_no_table_and_no_target_is_the_only_no_override_case(self):
        self.assertIsNone(R.hand_floor_for_tables(_Rig({}), ("lab_table_a",)))
        self.assertIsNone(R.hand_floor_for_tables(_Rig({}), ()))

    def test_the_slack_band_is_what_decides_marginal_cases(self):
        floor = 0.06
        self.assertIsNotNone(R.check_hand_floor(
            floor + R.HAND_FLOOR_SLACK_M - 1e-6, floor))
        self.assertIsNone(R.check_hand_floor(
            floor + R.HAND_FLOOR_SLACK_M + 1e-6, floor))


if __name__ == "__main__":
    unittest.main()
