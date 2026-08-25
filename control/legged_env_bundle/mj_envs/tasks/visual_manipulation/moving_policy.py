"""Heuristic base-motion policy: walk the humanoid until a target is arm-reachable.

Two pieces, composed by pickplace_reach_env.py's --walk mission:

  ReachabilityGate      -- answers "could the arm reach this target from where the
                           robot stands right now?" with the learned per-arm
                           reachability scorers (utils/reachability.py).
  HeuristicMovingPolicy -- turns that verdict into base velocity commands:
                           TURN (align to the latched drive direction) -> WALK ->
                           STOP (latched).

Ported from the moving policy running on the real Duke V2 humanoid
(visual_servoing repo, visual_navigator.ProportionalNavPolicy), with three
deliberate changes for this sim deliverable:

  * Drive directions are per-robot, not hardcoded (``drive_dirs``). The original
    "NO turning in place" directive assumed the opposed head-camera pair, which
    lets a target in either half-plane be approached without rotating: V2-fixed
    still runs that way on (+X, -X), and actuated V2 adds (+/-Y) to crab at a
    lateral cube. But g1 has ONE fixed forward camera and a single (+X)
    direction, so it must physically turn to see and to approach -- hence the
    real TURN state and its ``turn_wz_min`` floor. Turning in place is now a
    normal phase, not an excluded one.
  * Arrival is decided by the ReachabilityGate -- "the object entered the arm's
    reach envelope" -- not by a fixed arrival distance. A geometric backstop at
    the validated close-reach distance still exists as a safety net.
  * Lateral velocity is a CAPPED cross-track correction, WALK phase only
    (CRUISE_VY_MAX = 0.6 m/s) -- an explicit,
    user-directed override of the earlier "always zero" premise. The prior
    finding (uncapped free-strafe destabilises the walking policy, both sim
    and real -- visual_navigator.py) still holds and is NOT retested here;
    this is a small bounded correction toward the target's own base-frame y,
    not a free-strafe cruise.

Frozen-policy compatibility notes (why the numbers are what they are):
  * Cruise is a CONSTANT magnitude -- the user-verified stable walking speed --
    in every declared direction (v83's training range is symmetric, lin_vel_x
    in +-1.0). No proportional slow-down on approach: v83 has a low-speed dead
    zone (commands under ~0.1 m/s produce stepping-in-place), so a taper would
    stall the robot short of the reach envelope. (An opt-in WALK-only taper
    exists behind ``taper_dist_m`` / ``WALK_TAPER_DIST_M``; off by default.)
  * The same dead zone is why yaw has a FLOOR, not just a cap: a decaying
    proportional command crawls below what the base will act on, so
    ``TURN_WZ_MIN`` holds the rate up until the heading error is inside
    ``FACE_TOL_RAD``.

The class constants below are authoritative: ``__init__`` builds a ``MoverCfg``
from them, OVERWRITING that dataclass's own defaults. Read the values here, not
in ``control_vec.MoverCfg`` -- several of its defaults are dead for this path
and have already drifted.
"""

import math

import torch


from asset_zoo.grasp_planning import generate_candidates
from utils.reachability import ReachabilityModel
from tasks.visual_manipulation.control_vec import (
    STOP, MoverCfg, MoverState, facings_from_dirs, mover_twist)


class ReachabilityGate:
    """Latched "target is arm-reachable from here" verdict for one target.

    Scoring recipe (mirrors how this repo already treats reachability, see
    grasp_planning.py): sample N_CANDIDATES grasp-plausible EE poses around the
    target (upper-hemisphere approach directions, random roll), score them with
    BOTH per-arm models, and take the max. A single identity-orientation query
    would be wrong twice over: only ~0.8% of feasible cache poses are near
    identity, and an identity wrist is not a grasp pose.

    TWO standoff rings must BOTH pass FOR THE SAME ARM (max over arms of min
    over rings of max over poses):
      0.15 m -- the repo's grasp standoff: wrist there = fingers AT the target;
      0.05 m -- wrist essentially at the target: the reach-demo "EE reaches the
                cube" standard.
    A distance sweep against the arm's anatomy showed the single 0.15 m ring
    admits stances where the fingers can touch the cube but the wrist can NOT
    reach its center (score cliff 0.52->0.56 m vs 0.44->0.48 m base-to-cube;
    both cliffs match the ~0.59 m straight-arm envelope). Requiring both rings
    per arm makes the stop unambiguous under either definition for ~4 cm of
    extra walk, and yields the ARM CHOICE for free: ``best_arm`` is the arm
    whose own both-ring score won, and ``latched_arm`` freezes that choice at
    the moment the gate fires -- the mission reaches with THAT arm, so the stop
    decision and the reach execution can never disagree about which arm.

    The verdict latches: SCORE_THRESHOLD must hold for CONSECUTIVE_HITS steps
    (debounce against flicker at the envelope boundary), and once latched it
    never un-latches -- the robot must not stop-start on the boundary.

    Honest scope: the scorers know kinematics + self-collision only (trained on
    the safe-arm-pose cache). They know nothing about the table as an obstacle;
    the geometric floor and the scene design carry that concern.
    """

    SCORE_THRESHOLD = 0.5    # the trainer's own validation decision boundary
    N_CANDIDATES = 64        # max-over-orientations saturates by ~64 (recon-measured)
    STANDOFFS_M = (0.15, 0.05)  # both rings must pass: fingers-at-cube AND wrist-at-cube
    CONSECUTIVE_HITS = 5     # debounce steps before latching
    GEOMETRIC_FLOOR_M = 0.28 # validated close-reach distance (front_back_close 'on')
    # Score a VIRTUAL target pushed this much farther away (horizontally, along
    # pelvis->target). Stopping only when the farther twin is reachable puts the
    # REAL target this deep INSIDE the envelope instead of on its edge. Without
    # it, stop distances scatter over 0.41-0.45 m while the wrist-at-cube
    # anatomical limit sits at ~0.46 m -- far-end stops left the cube on the
    # boundary and the straight-line reach plateaued a few cm short (observed:
    # min 0.077 m vs the 0.05 m bar). Buys reach slack with half a step of walk.
    MARGIN_M = 0.05

    def __init__(self, device: str = "cpu", robot: str = "humanoid_v21", seed: int = 0):
        try:
            self._models = [ReachabilityModel.from_robot(robot, arm, device)
                            for arm in ("left", "right")]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n\nThe reachability scorers are a build artifact (asset_zoo/cache is "
                f"gitignored). Train them once with:\n"
                f"  python mj_envs/asset_zoo/build_reachability_map.py --robot {robot} --arm both"
            ) from e
        self._device = device
        self._arms = ("left", "right")
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self.last_score = 0.0
        self.last_arm_scores = {"left": 0.0, "right": 0.0}
        self.best_arm = "right"
        self.latched_arm: str | None = None    # frozen at latch time; the mission reaches with it
        self.latched = False
        self.latch_reason: str | None = None   # "score" | "floor"
        self._hits = 0

    def score(self, target_world: torch.Tensor,
              pelvis_pos: torch.Tensor, pelvis_quat: torch.Tensor) -> float:
        """max over arms of (min over standoff rings of max over N_CANDIDATES poses).

        The per-arm min makes the WORST ring the gating one -- the SAME arm must
        find both "fingers at cube" and "wrist at cube" feasible. Scored against
        the MARGIN_M-farther virtual twin of the target (see the constant), so a
        passing verdict means the real target sits that deep inside the envelope.
        Side effects: ``last_arm_scores`` and ``best_arm`` record which arm won.
        """
        virtual = target_world.to("cpu").clone()
        away = virtual[:2] - pelvis_pos.to("cpu")[:2]
        n = float(torch.linalg.norm(away))
        if n > 1e-6:
            virtual[:2] += (away / n) * self.MARGIN_M
        ring_maxes = {arm: [] for arm in self._arms}
        for standoff in self.STANDOFFS_M:
            cands = generate_candidates(
                virtual, torch.tensor([1.0, 0.0, 0.0, 0.0]),
                n=self.N_CANDIDATES, standoff=standoff, arm_assign="both",
                device="cpu", generator=self._gen)
            pos = torch.stack([c.ee_pos_world for c in cands])
            quat = torch.stack([c.ee_quat_world for c in cands])
            for arm, m in zip(self._arms, self._models):
                ring_maxes[arm].append(float(m.score_world_frame(pos, quat, pelvis_pos, pelvis_quat).max()))
        self.last_arm_scores = {arm: min(v) for arm, v in ring_maxes.items()}
        self.best_arm = max(self.last_arm_scores, key=self.last_arm_scores.get)
        return self.last_arm_scores[self.best_arm]

    def step(self, target_world: torch.Tensor, pelvis_pos: torch.Tensor,
             pelvis_quat: torch.Tensor, dist_xy: float) -> bool:
        """Update the gate once per control step; True once latched."""
        if self.latched:
            return True
        self.last_score = self.score(target_world, pelvis_pos, pelvis_quat)
        self._hits = self._hits + 1 if self.last_score > self.SCORE_THRESHOLD else 0
        if self._hits >= self.CONSECUTIVE_HITS:
            self.latched, self.latch_reason = True, "score"
        elif dist_xy <= self.GEOMETRIC_FLOOR_M:
            self.latched, self.latch_reason = True, "floor"
        if self.latched:
            self.latched_arm = self.best_arm
        return self.latched

    def reset(self) -> None:
        self.last_score, self.latched, self.latch_reason, self._hits = 0.0, False, None, 0
        self.last_arm_scores = {"left": 0.0, "right": 0.0}
        self.best_arm, self.latched_arm = "right", None


class HeuristicMovingPolicy:
    """Select nearest configured direction, then TURN/WALK/STOP toward one target.

    compute() maps the target's position in the BASE frame to a body-frame twist
    (vx, vy, wz); vy is a capped proportional cross-track correction toward the
    target's own base-frame y, active only in WALK (see module docstring,
    CRUISE_VY_MAX/K_LATERAL). All channels pass through an acceleration limiter
    (MAX_ACCEL, matching the real robot's RateLimiter) so starts and stops are
    ~0.1 s ramps, never steps.

    A PREFERRED DRIVE DIRECTION is a body-frame unit vector; the robot declares a
    SET of them (``drive_dirs``). Each direction ``d`` has a body angle
    ``phi = atan2(dy, dx)`` and a translational vector. Forward/backward/lateral
    directions are all valid.

    ONE direction is picked per target, on the first compute() after retarget(): the one
    minimizing the turn ``|wrap(bearing - phi)|`` to bring the target onto it
    (ties -> lowest index). Its ``phi`` and drive vector latch for the target. The
    default ``{front, back}`` recovers the old bidirectional FACING CHOICE (a rear
    target latches the back facing, drive backward); a single-facing robot
    ``{front}`` must TURN ~180 deg to face a rear target.

    NOT byte-identical to the earlier no-turn walk (the ``acdf637`` "no-turn
    bidirectional walk per user redesign"): that version drove at CRUISE_VX from
    step 0 while steering. This one first TURNs in place (vx 0) until the heading
    error is under FACE_TOL_RAD, so an off-facing target incurs a short turn-in-
    place before translating. Caller MUST recompute the base-frame target each step
    (the base rotates), else the TURN heading error never nulls and WALK is never
    reached. If the pure no-turn walk is wanted, drive with a single wide facing or
    restore the acdf637 compute() -- design choice, not covered here.

    States:
      TURN -- vx 0, steer toward the latched facing at K_STEER*err, capped at
              TURN_WZ_MAX and FLOORED at TURN_WZ_MIN while the error is still
              outside FACE_TOL_RAD (the proportional term decays below the base's
              yaw dead zone otherwise, and the turn crawls). Hands off to WALK
              after SETTLE_STEPS consecutive steps with the heading error under
              FACE_TOL_RAD AND the yaw ramp CONVERGED TO ITS COMMAND
              (|wz - want_wz| < wz_eps). Note the guard is convergence, NOT
              |wz| ~ 0: inside the tolerance band a proportional law still asks
              for a small nonzero want_wz, so an |wz| ~ 0 test is unsatisfiable
              and TURN deadlocks -- see control_vec.mover_twist.
      WALK -- translate along the selected configured direction; in-stride steering
              (WALK_WZ_MAX cap) holds the target on that direction.
      STOP -- entered via stop() when the ReachabilityGate fires; never resumes.

    retarget() begins the next target (fresh direction pick, re-enters TURN).
    """

    CRUISE_VX = 0.4            # m/s magnitude, every declared X direction -- user-verified stable
    CRUISE_VY_MAX = 0.6        # m/s lateral cross-track cap, WALK only -- capped override of the earlier
    #   always-zero vy (uncapped free-strafe destabilised the checkpoint, user-tested; still true, not
    #   retested -- this caps the magnitude instead of removing the cap entirely). Matches CRUISE_VX.
    K_LATERAL = 0.6            # proportional gain, target base-frame y -> want_vy (before the cap)
    WALK_WZ_MAX = 1.0          # rad/s in-stride steering cap (was 0.2; g1's fixed forward camera needed more aggressive re-aim mid-stride for off-axis cubes, 2026-07-25)
    TURN_WZ_MAX = 1.0          # rad/s turn-in-place cap (inside v83's +-2pi yaw ROM)
    TURN_WZ_MIN = 0.6          # rad/s turn floor out of face_tol: snap the last degrees, no deadzone crawl
    #   0.4 -> 0.6 (2026-08-02, user, ALL robots). This constant OVERWRITES MoverCfg.turn_wz_min in
    #   __init__, so it is the only value any reach robot runs -- the dataclass default is dead here.
    K_STEER = 1.0
    MAX_ACCEL = 3.0            # m/s^2 and rad/s^2 per-channel ramp (real-robot value)
    FACE_TOL_RAD = math.radians(8.0)  # heading-error tolerance to consider a facing achieved.
    #   2 -> 8 deg (2026-08-03). 2 deg was UNWINNABLE against the gait: MoverCfg records that the gait alone
    #   bounces the heading error by +-6 deg, and TURN_WZ_MIN kicks the base back through any window narrower
    #   than that bounce, so SETTLE_STEPS never saw 3 consecutive in-tol steps and TURN never handed off. A
    #   pivoting robot then spun in place with vx == 0 while the target distance stayed frozen -- measured
    #   27.6 s in ONE DISCOVER span on v2_single_fixed, which is what put 2 trials of dyn9 into the tick cap.
    #   8 deg clears the bounce amplitude with margin, so the floor switches off and the yaw ramp can converge.
    #   Chosen over lowering TURN_WZ_MIN (the other end of the same interaction) because the floor was raised
    #   twice on user request to cut settle wall-time, and slackening it gives that back; the cost here is
    #   instead a looser WALK-entry facing, which WALK's own in-stride steering (WALK_WZ_MAX) keeps closing.
    #   Not a facing GUARANTEE for anything downstream: consumers that need one carry their own tolerance
    #   (_REACH_FACE_TOL_RAD 3 deg for g1 commit-time frustum visibility, _APPROACH_FACE_TOL_RAD 15 deg).
    #   v2 is unaffected either way -- it sets walk_turn_cruise and has no TURN state to hang in.
    SETTLE_STEPS = 3          # consecutive in-tolerance steps before TURN -> WALK
    _WZ_EPS = 1e-2            # ramped yaw rate deemed "stopped" for the handoff

    _STATE_STR = ("TURN", "WALK", "STOP")   # MoverState int code -> legacy string

    def __init__(self, step_dt: float, drive_dirs=((1.0, 0.0), (-1.0, 0.0)),
                 cruise_vy_max: float | None = None, taper_dist_m: float | None = None,
                 turn_cruise: bool = False):
        # Thin N=1 shim over the batched, deploy-safe ``control_vec.mover_twist`` (single source of
        # truth for the TURN/WALK/STOP ALGORITHM -- but not for its GAINS). Every field passed below
        # OVERWRITES the matching ``MoverCfg`` default, so the class constants above are what actually
        # runs and several dataclass defaults have already drifted away from them (2026-08-02:
        # turn_wz_max 0.8 vs 1.0, k_steer 0.6 vs 1.0, cruise_vy_max 0.4 vs 0.6). Read the constants here.
        # Per-robot vy override: ``None`` = the class default 0.6, NOT 0.0 -- the humanoid family takes
        # that path (its descriptors leave ``walk_cruise_vy_max`` unset); only g1 overrides, to 0.3.
        effective_vy = self.CRUISE_VY_MAX if cruise_vy_max is None else float(cruise_vy_max)
        self._cfg = MoverCfg(
            step_dt=float(step_dt), cruise_vx=self.CRUISE_VX, cruise_vy_max=effective_vy,
            walk_wz_max=self.WALK_WZ_MAX,
            turn_wz_max=self.TURN_WZ_MAX, turn_wz_min=self.TURN_WZ_MIN, k_steer=self.K_STEER,
            k_lateral=self.K_LATERAL, max_accel=self.MAX_ACCEL,
            face_tol_rad=self.FACE_TOL_RAD, settle_steps=self.SETTLE_STEPS, wz_eps=self._WZ_EPS,
            taper_dist_m=taper_dist_m, turn_cruise=turn_cruise)
        self._face_phi, self._drive_dirs = facings_from_dirs(drive_dirs)
        self._st = MoverState.init(1)
        self._stop = torch.zeros(1, dtype=torch.bool)

    def compute(self, target_x_base: float, target_y_base: float) -> tuple[float, float, float]:
        """Body-frame (vx, vy, wz) for this step; target given in base frame."""
        tgt = torch.tensor([[target_x_base, target_y_base]], dtype=torch.float32)
        tw = mover_twist(tgt, self._st, self._face_phi, self._drive_dirs, self._cfg,
                         stop_mask=self._stop)
        return float(tw[0, 0]), float(tw[0, 1]), float(tw[0, 2])

    def stop(self) -> None:
        """Latch STOP (reachability gate fired). Ramp-down happens in the next compute()."""
        self._stop = torch.ones(1, dtype=torch.bool)

    def retarget(self) -> None:
        """Begin a new target approach: fresh direction pick, keep ramp state."""
        self._st.retarget()
        self._stop = torch.zeros(1, dtype=torch.bool)

    @property
    def state(self) -> str:
        return self._STATE_STR[int(self._st.state.item())]

    @property
    def direction(self) -> tuple[float, float]:
        """Latched unit drive direction; ``(0, 0)`` until target direction selection."""
        if not bool(self._st.picked.item()):
            return 0.0, 0.0
        return tuple(float(v) for v in self._st.drive_dir[0])

    @property
    def direction_label(self) -> str:
        """Cardinal label for evaluation logs; configured directions may also be diagonal."""
        dx, dy = self.direction
        if abs(dx) >= abs(dy):
            return "forward" if dx >= 0.0 else "backward"
        return "left" if dy >= 0.0 else "right"

    @property
    def braking_distance_m(self) -> float:
        """Command-ramp stopping distance for selected target direction, before physical-policy margin."""
        dx, dy = self.direction
        if abs(dx) + abs(dy) < 1e-6:
            return 0.0
        speed = min(
            self.CRUISE_VX / abs(dx) if abs(dx) > 1e-6 else float("inf"),
            self.CRUISE_VY_MAX / abs(dy) if abs(dy) > 1e-6 else float("inf"),
        )
        return speed * speed / (2.0 * self.MAX_ACCEL)

    @property
    def settled(self) -> bool:
        """True once STOP has ramped the actual command (including yaw) to near zero."""
        return (int(self._st.state.item()) == STOP
                and abs(float(self._st.vx.item())) < 1e-3 and abs(float(self._st.vy.item())) < 1e-3
                and abs(float(self._st.wz.item())) < 1e-3)
