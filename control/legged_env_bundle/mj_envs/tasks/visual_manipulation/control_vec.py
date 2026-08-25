"""Batched, deploy-safe base+camera control primitives (ONE mechanism: parallel sim AND real robot).

Pure-tensor control logic on top of ``camera_perception`` (which already holds the batched perception
primitives). Kept free of mujoco / mjlab / env imports so it is unit-testable in isolation (run this file
directly) and runs unchanged on hardware -- the ONLY difference between sim and robot is the measurement
SOURCE feeding these functions (GT ``fov_detect`` / belief in sim; AprilTag+PnP / encoder FK on the
robot), never the control math.

Deployability rule (why base frame, why no world pose):
  - ``gaze_aim`` takes the target and the camera lens position BOTH in the base frame. On hardware the
    lens position comes from gimbal-encoder forward kinematics (base<-camera) and the target from the
    belief / AprilTag -- the absolute world base pose CANCELS, so no world frame is needed. In sim the
    env supplies the identical base-frame numbers (world frame cancels the same way). Never reads a world
    ground-truth object pose -> honest ("do not cheat").
  - ``mover_twist`` consumes only the target BEARING in the base frame and emits a body-frame twist
    (vx, vy, wz) -- the same command surface the deploy locomotion policy already consumes.

Batching: every function is elementwise over a leading ``[N]`` env axis (and ``[K]`` cameras for the
gaze/see paths); there is no python per-env loop and no ``[0]`` scalar collapse. ``mover_twist`` is a
functional batched port of the ``HeuristicMovingPolicy`` state machine -- state is carried in a
``MoverState`` of ``[N]`` tensors and every transition is a ``torch.where`` (no python branch on env).

Yaw-sign note: ``gaze_aim`` uses ``yaw = atan2(-sign*d_y, sign*d_x)``. The ``-`` on d_y is REQUIRED --
the compiled gimbal (both the standalone ``duke_v2/head_camera_dual.xml`` and the humanoid-attached
build) swings the optical axis toward -y for a POSITIVE yaw joint; the older ``atan2(sign*d_y, sign*d_x)``
only agreed with FK at yaw=0 and mis-aimed up to ~57 deg off-axis. Verified against the compiled model.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from tasks.camera_perception import fov_detect


# --------------------------------------------------------------------------------------------------
# Gaze aim (closed form) + see-gate
# --------------------------------------------------------------------------------------------------
def gaze_aim(
    target_base: torch.Tensor,
    cam_pos_base: torch.Tensor,
    aim_sign: torch.Tensor,
    yaw_rom: float,
    pitch_rom: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form gimbal (yaw, pitch) so each camera's optical axis points at its target. Batched.

    Args:
        target_base:  (..., K, 3) target position, BASE frame.
        cam_pos_base: (..., K, 3) camera lens position, BASE frame (encoder FK on hardware).
        aim_sign:     (K,) +1 for a forward-datum (left) camera, -1 for the rear-datum (right) camera.
        yaw_rom/pitch_rom: symmetric joint ROM (rad); output is clamped to +-ROM.
    Returns:
        yaw, pitch: each (..., K), the gimbal joint targets.

    Kinematics (optical axis in base frame, VERIFIED against the compiled gimbal FK):
        forward(yaw, pitch) = ( sign*cos(yaw)*cos(pitch), -sign*sin(yaw)*cos(pitch), -sin(pitch) )
    Inverting for d = normalize(target - cam_pos):
        pitch = asin(-d_z);   yaw = atan2(-sign*d_y, sign*d_x)   (cos(pitch) cancels, no divide).
    """
    d = target_base - cam_pos_base
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pitch = torch.asin(torch.clamp(-d[..., 2], -1.0, 1.0))
    yaw = torch.atan2(-aim_sign * d[..., 1], aim_sign * d[..., 0])
    return yaw.clamp(-yaw_rom, yaw_rom), pitch.clamp(-pitch_rom, pitch_rom)


def gaze_forward(yaw: torch.Tensor, pitch: torch.Tensor, aim_sign: torch.Tensor) -> torch.Tensor:
    """Optical-axis unit vector (base frame) for a gimbal at (yaw, pitch); inverse of ``gaze_aim``.

    The exact ``forward(yaw, pitch)`` relation ``gaze_aim`` inverts -- exposed so callers (and the
    self-test) can reconstruct the aimed direction without touching mujoco. Args broadcast: yaw, pitch
    (..., K); aim_sign (K,). Returns (..., K, 3)."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    return torch.stack([aim_sign * cy * cp, -aim_sign * sy * cp, -sp], dim=-1)


def see_gate(
    cam_pos_base: torch.Tensor,
    cam_mat_base: torch.Tensor,
    target_base: torch.Tensor,
    h_half: float,
    v_half: float,
    near: float,
    far: float,
) -> dict[str, torch.Tensor]:
    """Batched FOV+range visibility gate -- thin wrapper over ``camera_perception.fov_detect``.

    All args BASE frame (``cam_mat_base`` = camera->base rotation, encoder FK on hardware), any leading
    batch. Returns the ``fov_detect`` dict (``in_fov`` gate + ``aim_error`` for a dense reward)."""
    return fov_detect(cam_pos_base, cam_mat_base, target_base, h_half, v_half, near, far)


# --------------------------------------------------------------------------------------------------
# Base mover (batched TURN -> WALK -> STOP)
# --------------------------------------------------------------------------------------------------
# Constants live HERE (deploy-safe module) as the single source of truth; ``moving_policy`` imports them.
# Values verbatim from the original ``HeuristicMovingPolicy`` (real-robot-ported).
TURN, WALK, STOP = 0, 1, 2   # MoverState.state codes
_MAX_PLANAR_SPEED_M_S = 0.8  # hardware safety ceiling; preserve direction when component caps sum above it


@dataclasses.dataclass(frozen=True)
class MoverCfg:
    """Base-mover tuning (real-robot RateLimiter values). ``step_dt`` is the control period (s).

    THESE DEFAULTS ARE NOT WHAT THE ROBOTS RUN. ``HeuristicMovingPolicy.__init__`` passes every gain
    explicitly from its own class constants, so for any reach robot the values here are overwritten and
    several have drifted apart (2026-08-02: ``turn_wz_max`` 0.8 here vs 1.0 there, ``k_steer`` 0.6 vs 1.0,
    ``cruise_vy_max`` 0.4 vs 0.6). They apply only to a caller that constructs ``MoverCfg`` directly --
    which today is the stance-field precompute, not the mission. Change gains in ``moving_policy.py``.
    """
    step_dt: float
    cruise_vx: float = 0.4           # m/s magnitude, every drive direction -- user-verified stable
    cruise_vy_max: float = 0.4       # m/s lateral cross-track correction cap, WALK only; 0.0 disables strafe.
    #   Capped override of a documented dead lever (uncapped free-strafe made the sim checkpoint fall over,
    #   user-tested) -- new premise: small bounded correction, not an uncapped strafe cruise.
    walk_wz_max: float = 1.0         # rad/s in-stride steering cap (was 0.2; g1 forward-cam off-axis re-aim, 2026-07-25)
    turn_wz_max: float = 0.8         # rad/s turn-in-place cap
    turn_wz_min: float = 0.6         # rad/s turn floor while |err| >= face_tol: snap the last degrees instead of
    #   crawling at k_steer*err (decays to ~0.03 rad/s -> hundreds of ticks near the target under the loco
    #   policy's low-yaw sluggishness). Off inside face_tol so want_wz -> 0 and the settle handoff still fires.
    #   Was 0.2; raised to 0.4 (2026-07-25 user request) to halve the no-crawl settle wall-time on g1's far
    #   scenarios where the square-up alone was eating 2-5s per visit, then to 0.6 (2026-08-02 user request,
    #   all robots). Still well below wave-19 g1 anti-sway range; safe vs the 1.0 rad/s TURN cap.
    k_steer: float = 0.6
    k_lateral: float = 0.6           # proportional gain, target base-frame y -> want_vy (before the cruise_vy_max clamp)
    max_accel: float = 3.0           # m/s^2 and rad/s^2 per-channel ramp
    face_tol_rad: float = math.radians(2.0)
    settle_steps: int = 3
    wz_eps: float = 1e-2
    turn_cruise: bool = False        # skip the turn-in-place state: enter WALK immediately and hold the
    #   yaw floor until faced, so the base ARCS into its facing while translating instead of pivoting on
    #   planted feet. Off by default -- only a robot whose descriptor opts in (``walk_turn_cruise``) takes
    #   this path, because a robot with a single forward drive direction would walk AWAY from a rear target
    #   for the whole 180 deg of the turn.
    #   Premise, measured 2026-08-02 on the v2 loco checkpoint: yaw tracking is a function of FORWARD SPEED,
    #   not of the yaw command. Over 400 fixed-command steps ``ang_err`` is 0.299 rad/s at (vx 0, wz 0.6)
    #   but 0.161 at (vx 0.4, wz 0.6); in-place, authority additionally DECAYS with accumulated yaw (ratio
    #   0.9 at 8 deg turned, 0.24 at 87 deg) and recovers only after the robot walks -- the policy turns by
    #   twisting on planted feet and does not re-step. An intermediate creep is the worst of both: 0.2 m/s
    #   is inside the LINEAR dead zone (75% ``velocity_tracking_failure``), and 0.12 m/s measured WORSE than
    #   standing still. Cruise is the only regime where both channels track.
    taper_dist_m: float | None = None  # WALK-only proximity taper: None (default) = off, exact prior
    #   behavior. When set, scales WALK's desired speed down linearly as base-to-target distance falls
    #   below this radius, floored at ``taper_floor`` (never toward the low-speed dead zone). Distinct
    #   from the earlier "no proportional slow-down" premise -- that concern was about starting a fresh
    #   WALK/TURN handoff at low speed (bootstrap needs momentum); here the base is ALREADY in WALK with
    #   an established gait, so a taper only trims an established cruise, not a cold start.
    taper_floor: float = 0.5         # minimum speed fraction at zero distance (0.4 m/s * 0.5 = 0.2 m/s,
    #   safely above the documented ~0.1 m/s dead-zone floor)


@dataclasses.dataclass
class MoverState:
    """Per-env ``[N]`` mover state. All tensors share device/dtype; ``state``/``settle`` are int64."""
    state: torch.Tensor       # (N,) in {TURN, WALK, STOP}
    phi: torch.Tensor         # (N,) latched target-direction angle
    drive_dir: torch.Tensor   # (N,2) normalized body-frame target-drive direction
    vx: torch.Tensor          # (N,) ramped forward velocity
    vy: torch.Tensor          # (N,) ramped lateral velocity (WALK cross-track correction; 0 unless cfg.cruise_vy_max > 0)
    wz: torch.Tensor          # (N,) ramped yaw rate
    settle: torch.Tensor      # (N,) consecutive in-tolerance TURN steps
    picked: torch.Tensor      # (N,) bool: direction chosen for current target

    @classmethod
    def init(cls, n: int, device="cpu", dtype=torch.float32) -> "MoverState":
        z = torch.zeros(n, device=device, dtype=dtype)
        zi = torch.zeros(n, device=device, dtype=torch.int64)
        return cls(state=zi.clone(), phi=z.clone(), drive_dir=torch.zeros(n, 2, device=device, dtype=dtype), vx=z.clone(), vy=z.clone(),
                   wz=z.clone(), settle=zi.clone(), picked=torch.zeros(n, device=device, dtype=torch.bool))

    def retarget(self, env_ids=None) -> None:
        """Begin a new target approach (fresh direction pick, re-enter TURN); keeps ramp state."""
        idx = slice(None) if env_ids is None else env_ids
        self.state[idx] = TURN
        self.picked[idx] = False
        self.settle[idx] = 0


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def mover_twist(
    target_xy_base: torch.Tensor,
    st: MoverState,
    face_phi: torch.Tensor,
    drive_dirs: torch.Tensor,
    cfg: MoverCfg,
    stop_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One batched control step of the TURN->WALK->STOP base mover; mutates ``st`` in place.

    Batched port of ``HeuristicMovingPolicy.compute`` -- byte-identical algorithm, elementwise over N.

    Args:
        target_xy_base: (N, 2) target position in the base frame.
        st:             MoverState of (N,) tensors, updated in place.
        face_phi/drive_dirs: (F,) admissible target bearings + ``(F,2)`` normalized drive directions.
                        One is latched per target, matching the
                        scalar ``_facings`` list order (ties -> lowest index via ``argmin``).
        cfg:            MoverCfg.
        stop_mask:      (N,) bool; True latches STOP for that env (reach/see gate fired) BEFORE this step
                        ramps -- so the step ramps toward zero, exactly like a scalar ``stop()`` between
                        ``compute()`` calls.
    Returns:
        twist: (N,3) body-frame (vx, vy, wz). WALK translates along the selected drive direction at
        the largest speed that respects axis limits, plus bounded perpendicular cross-track correction.
        TURN/STOP zero translation. All channels are acceleration-limited.
    """
    if stop_mask is not None:
        st.state = torch.where(stop_mask, torch.full_like(st.state, STOP), st.state)

    if cfg.turn_cruise:
        # Retire TURN before anything reads the state: entering WALK on tick 1 is what makes the base arc
        # into its facing at cruise instead of pivoting. Done by coercing the STATE rather than by giving
        # TURN a translation, because three callers gate on ``mover.state == "WALK"`` -- including the
        # standoff brake (``reach_policy.py:2340``) -- and a driving-but-still-TURN base would never brake.
        # Measured: with cruise on, the settle criterion (|err| < face_tol AND yaw ramp converged, 3 in a
        # row) rarely fires at all, because the gait alone bounces err by +-6 deg.
        st.state = torch.where(st.state == TURN, torch.full_like(st.state, WALK), st.state)

    bearing = torch.atan2(target_xy_base[:, 1], target_xy_base[:, 0])            # (N,)

    # Pick the nearest permitted direction on the first un-picked step for a target (state != STOP).
    need_pick = (~st.picked) & (st.state != STOP)
    err_f = _wrap(bearing[:, None] - face_phi[None, :]).abs()                    # (N, F)
    idx = err_f.argmin(dim=1)                                                    # (N,) lowest-index tie
    st.phi = torch.where(need_pick, face_phi[idx], st.phi)
    st.drive_dir = torch.where(need_pick[:, None], drive_dirs[idx], st.drive_dir)
    st.picked = st.picked | need_pick

    err = _wrap(bearing - st.phi)
    is_turn = st.state == TURN
    is_stop = st.state == STOP

    wz_cap = torch.where(is_turn, cfg.turn_wz_max, cfg.walk_wz_max)
    want_wz = (cfg.k_steer * err).clamp(-wz_cap, wz_cap)
    # TURN floor: while still out of face tolerance, hold |wz| >= turn_wz_min so the base snaps to facing
    # instead of crawling at the decaying k_steer*err (see MoverCfg.turn_wz_min). Off inside face_tol.
    # Under ``turn_cruise`` there is no TURN state left to gate on, so the floor keys on the heading error
    # alone -- it is the only thing still doing TURN's job.
    turn_floor = err.abs() >= cfg.face_tol_rad
    if not cfg.turn_cruise:
        turn_floor = turn_floor & is_turn
    want_wz = torch.where(turn_floor, torch.sign(err) * want_wz.abs().clamp(min=cfg.turn_wz_min), want_wz)
    want_wz = torch.where(is_stop, torch.zeros_like(st.wz), want_wz)
    # Drive along selected direction. The scalar speed is the largest value whose X/Y components fit
    # their configured caps: cardinal directions therefore command (+/-0.4, 0) or (0, +/-0.6).
    dir_x, dir_y = st.drive_dir[:, 0], st.drive_dir[:, 1]
    inf = torch.full_like(dir_x, float("inf"))
    x_speed = torch.where(dir_x.abs() > 1e-6, cfg.cruise_vx / dir_x.abs(), inf)
    y_speed = torch.where(dir_y.abs() > 1e-6, cfg.cruise_vy_max / dir_y.abs(), inf)
    drive_speed = torch.minimum(x_speed, y_speed)
    primary = st.drive_dir * drive_speed[:, None]
    # Perpendicular correction keeps an off-axis target centered on the selected direction. Clamp final
    # components afterwards so correction cannot exceed either locomotion limit.
    perpendicular = torch.stack([-dir_y, dir_x], dim=-1)
    cross_error = (target_xy_base * perpendicular).sum(dim=-1)
    desired_xy = primary + perpendicular * (cfg.k_lateral * cross_error)[:, None]
    desired_xy[:, 0].clamp_(-cfg.cruise_vx, cfg.cruise_vx)
    desired_xy[:, 1].clamp_(-cfg.cruise_vy_max, cfg.cruise_vy_max)
    if cfg.taper_dist_m is not None:
        dist = target_xy_base.norm(dim=-1)
        taper = (dist / cfg.taper_dist_m).clamp(cfg.taper_floor, 1.0)
        is_walk = st.state == WALK
        desired_xy = torch.where(is_walk[:, None], desired_xy * taper[:, None], desired_xy)
    desired_speed = desired_xy.norm(dim=-1).clamp_min(1e-6)
    desired_xy *= torch.clamp(_MAX_PLANAR_SPEED_M_S / desired_speed, max=1.0)[:, None]
    desired_xy = torch.where((is_turn | is_stop)[:, None], torch.zeros_like(desired_xy), desired_xy)
    want_vx, want_vy = desired_xy[:, 0], desired_xy[:, 1]

    step = cfg.max_accel * cfg.step_dt
    st.vx = st.vx + (want_vx - st.vx).clamp(-step, step)
    st.vy = st.vy + (want_vy - st.vy).clamp(-step, step)
    st.wz = st.wz + (want_wz - st.wz).clamp(-step, step)

    # TURN -> WALK handoff: SETTLE_STEPS consecutive in-tolerance steps (heading in tol AND yaw ramp CONVERGED
    # to its command). The guard is |wz - want_wz|, NOT |wz|: a proportional steering law leaves a residual
    # want_wz = k_steer*err whenever heading is within (but not exactly at) tolerance, so |wz| < wz_eps is
    # UNSATISFIABLE in the band wz_eps/k_steer < |err| < face_tol -> TURN deadlocks (the base cannot null a
    # sub-dead-zone yaw command, so err never reaches 0). Ramp-convergence still blocks a genuine mid-rotation
    # handoff (large |wz| from a big turn, small want_wz) while admitting the small in-tol steering command.
    in_tol = (err.abs() < cfg.face_tol_rad) & ((st.wz - want_wz).abs() < cfg.wz_eps)
    st.settle = torch.where(is_turn, torch.where(in_tol, st.settle + 1,
                                                 torch.zeros_like(st.settle)), st.settle)
    to_walk = is_turn & (st.settle >= cfg.settle_steps)
    st.state = torch.where(to_walk, torch.full_like(st.state, WALK), st.state)

    return torch.stack([st.vx, st.vy, st.wz], dim=-1)


def facings_from_dirs(face_dirs, device="cpu", dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute target bearings and normalized body-frame drive directions from ``[(dx,dy),...]``.

    Directions may be forward, backward, lateral, or diagonal. A zero vector is invalid."""
    phi, directions = [], []
    for dx, dy in face_dirs:
        norm = math.hypot(dx, dy)
        assert norm > 0.0, f"zero drive direction: {(dx, dy)}"
        dx, dy = dx / norm, dy / norm
        phi.append(math.atan2(dy, dx))
        directions.append((dx, dy))
    return (torch.tensor(phi, device=device, dtype=dtype),
            torch.tensor(directions, device=device, dtype=dtype))


# --------------------------------------------------------------------------------------------------
# Walk-to-radius integrator + torso-safe standoff (batched, deploy-safe, mujoco-free)
# --------------------------------------------------------------------------------------------------
def walk_to_target_radius_batched(
    base_xy: torch.Tensor, yaw: torch.Tensor, target_xy: torch.Tensor,
    face_dirs: tuple, cfg: MoverCfg, stop_radius: float,
    max_steps: int = 4000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate N base walk-ins IN PARALLEL through ``mover_twist`` until each target enters
    ``stop_radius`` (one step-loop, no python per-env loop) -- the batched form of the scalar
    walk-until-reachable mover (``moving_policy.HeuristicMovingPolicy`` wrapped in a walk-to-radius
    loop). The SAME mover that runs the parallel env and the real robot; general, NOT task-specific.

    Args:
        base_xy: (N, 2) start base XY; yaw: (N,) start heading; target_xy: (N, 2) targets. All torch.
        face_dirs: robot's admissible body-frame drive directions.
        cfg:       MoverCfg -- ``step_dt`` is the integration period and axis caps set walk speed.
        stop_radius: reach radius the base halts at; ``inf`` = turn-in-place only (the STOP latch needs
                     state != TURN, so the body faces the target without ever translating).
    Returns:
        (base_xy, yaw, settled): final (N,2) / (N,) / (N,) bool -- byte-identical per env to the scalar
        walk-until-radius reference (asserted in the self-test).
    """
    n = base_xy.shape[0]
    dev = base_xy.device
    dt = cfg.step_dt
    face_phi, drive_dirs = facings_from_dirs(face_dirs, device=dev, dtype=base_xy.dtype)
    st = MoverState.init(n, device=dev, dtype=base_xy.dtype)
    base_xy = base_xy.clone()
    yaw = yaw.clone()
    settled = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(max_steps):
        delta = target_xy - base_xy
        dist = delta.norm(dim=-1)
        c, s = torch.cos(yaw), torch.sin(yaw)
        x_base = c * delta[:, 0] + s * delta[:, 1]
        y_base = -s * delta[:, 0] + c * delta[:, 1]
        stop_mask = (dist <= stop_radius) & (st.state != TURN)      # face target before latching STOP
        tw = mover_twist(torch.stack([x_base, y_base], dim=-1), st, face_phi, drive_dirs, cfg,
                         stop_mask=stop_mask)
        active = (~settled).to(base_xy.dtype)
        vx, vy = tw[:, 0] * active, tw[:, 1] * active                # frozen envs stop integrating
        wz = tw[:, 2] * active
        base_xy = base_xy + dt * torch.stack([c * vx - s * vy, s * vx + c * vy], dim=-1)
        yaw = yaw + dt * wz
        settled = settled | ((st.state == STOP) & (st.vx.abs() < 1e-3) & (st.vy.abs() < 1e-3) & (st.wz.abs() < 1e-3))
        if bool(settled.all()):
            break
    return base_xy, yaw, settled


def table_contact_stop_batched(
    base_xy: torch.Tensor, cube_xy: torch.Tensor, edge_xy: torch.Tensor,
    body_half_depth,
) -> torch.Tensor:
    """Batched torso-safe standoff: base XY STOP point where the torso FRONT (``body_half_depth`` ahead
    of base center) meets the table's NEAR FACE, advancing along the base->cube ray. Robot-centric --
    from base-frame bearings to the near-edge anchor (``edge_xy``, caller-selected) and the cube plus the
    body half-depth; NO world pose. The near face is modelled frontal to the base, so its range along the
    ray is ``d_edge / (ray . face_normal)``. Degenerate cases -- edge/cube coincident with the base, or
    the cube not beyond the near face along the ray -- return ``base_xy`` (do not advance). Elementwise
    over N.

    Args:
        base_xy/cube_xy/edge_xy: (N, 2). body_half_depth: scalar or (N,).
    Returns:
        stop_xy: (N, 2).
    """
    to_edge = edge_xy - base_xy
    d_edge = to_edge.norm(dim=-1)
    to_cube = cube_xy - base_xy
    range_cube = to_cube.norm(dim=-1)
    face_normal = to_edge / d_edge.clamp_min(1e-6).unsqueeze(-1)
    ray = to_cube / range_cube.clamp_min(1e-6).unsqueeze(-1)
    denom = (ray * face_normal).sum(dim=-1)
    advance = (d_edge > 1e-6) & (range_cube > 1e-6) & (denom > 1e-6)
    reach = (d_edge / denom.clamp_min(1e-6) - body_half_depth).clamp_min(0.0)
    stop = base_xy + reach.unsqueeze(-1) * ray
    return torch.where(advance.unsqueeze(-1), stop, base_xy)


# --------------------------------------------------------------------------------------------------
# Self-test (run: python mj_envs/tasks/visual_manipulation/control_vec.py). No mujoco needed.
# --------------------------------------------------------------------------------------------------
def _scalar_drive_velocity(direction, target_xy, cfg: MoverCfg) -> tuple[float, float]:
    """Float reference for WALK translation along one configured drive direction."""
    dx, dy = direction
    speed = min(
        cfg.cruise_vx / abs(dx) if abs(dx) > 1e-6 else float("inf"),
        cfg.cruise_vy_max / abs(dy) if abs(dy) > 1e-6 else float("inf"),
    )
    px, py = dx * speed, dy * speed
    perp_x, perp_y = -dy, dx
    cross = target_xy[0] * perp_x + target_xy[1] * perp_y
    vx = max(-cfg.cruise_vx, min(cfg.cruise_vx, px + perp_x * cfg.k_lateral * cross))
    vy = max(-cfg.cruise_vy_max, min(cfg.cruise_vy_max, py + perp_y * cfg.k_lateral * cross))
    norm = math.hypot(vx, vy)
    if norm > _MAX_PLANAR_SPEED_M_S:
        vx, vy = vx * _MAX_PLANAR_SPEED_M_S / norm, vy * _MAX_PLANAR_SPEED_M_S / norm
    return vx, vy


def _scalar_mover_ref(traj_xy, face_dirs, cfg: MoverCfg, stop_at=None):
    """Reference scalar mover (the ORIGINAL HeuristicMovingPolicy.compute algorithm, inlined) so the
    batched parity check has no heavy import. ``traj_xy``: list of (x,y) base-frame targets per step.
    Returns a list of (vx, vy, wz) triples -- vy mirrors ``mover_twist``'s cross-track term, identically
    0 for the default ``cfg.cruise_vy_max == 0.0``."""
    facings = []
    for dx, dy in face_dirs:
        norm = math.hypot(dx, dy)
        facings.append((math.atan2(dy, dx), (dx / norm, dy / norm)))
    state, phi, direction, vx, vy, wz, settle = TURN, 0.0, (0.0, 0.0), 0.0, 0.0, 0.0, 0
    picked = False
    out = []
    for i, (tx, ty) in enumerate(traj_xy):
        if stop_at is not None and i >= stop_at:
            state = STOP
        bearing = math.atan2(ty, tx)
        if state == STOP:
            want_vx, want_wz = 0.0, 0.0
        else:
            if not picked:
                phi, direction = min(facings, key=lambda f: abs(_wrap(torch.tensor(bearing - f[0])).item()))
                picked = True
            err = (bearing - phi + math.pi) % (2 * math.pi) - math.pi
            if state == TURN:
                want_vx = 0.0
                want_vy = 0.0
                want_wz = max(-cfg.turn_wz_max, min(cfg.turn_wz_max, cfg.k_steer * err))
                if abs(err) >= cfg.face_tol_rad:                  # TURN floor (parity with mover_twist)
                    want_wz = math.copysign(max(abs(want_wz), cfg.turn_wz_min), err)
            else:
                want_vx, want_vy = _scalar_drive_velocity(direction, (tx, ty), cfg)
                want_wz = max(-cfg.walk_wz_max, min(cfg.walk_wz_max, cfg.k_steer * err))
        if state == STOP:
            want_vy = 0.0
        step = cfg.max_accel * cfg.step_dt
        vx += max(-step, min(step, want_vx - vx))
        vy += max(-step, min(step, want_vy - vy))
        wz += max(-step, min(step, want_wz - wz))
        if state == TURN:
            err = (bearing - phi + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < cfg.face_tol_rad and abs(wz - want_wz) < cfg.wz_eps:
                settle += 1
                if settle >= cfg.settle_steps:
                    state = WALK
            else:
                settle = 0
        out.append((vx, vy, wz))
    return out


def _scalar_walk_ref(base_xy, yaw, target_xy, face_dirs, cfg: MoverCfg, stop_radius, max_steps=4000):
    """Scalar (single-env, python-float) reference for ``walk_to_target_radius_batched`` -- the same
    inlined mover as ``_scalar_mover_ref`` driving a walk-to-radius integrator. Returns final
    ``(bx, by, yaw, settled)``."""
    facings = []
    for dx, dy in face_dirs:
        norm = math.hypot(dx, dy)
        facings.append((math.atan2(dy, dx), (dx / norm, dy / norm)))
    state, phi, direction, vx, vy, wz, settle = TURN, 0.0, (0.0, 0.0), 0.0, 0.0, 0.0, 0
    picked = False
    bx, by, yw = float(base_xy[0]), float(base_xy[1]), float(yaw)
    tx, ty = float(target_xy[0]), float(target_xy[1])
    settled = False
    for _ in range(max_steps):
        dx_, dy_ = tx - bx, ty - by
        dist = math.hypot(dx_, dy_)
        c, s = math.cos(yw), math.sin(yw)
        x_base = c * dx_ + s * dy_
        y_base = -s * dx_ + c * dy_
        if dist <= stop_radius and state != TURN:      # stop_mask: face target before latching STOP
            state = STOP
        bearing = math.atan2(y_base, x_base)
        if state == STOP:
            want_vx, want_vy, want_wz = 0.0, 0.0, 0.0
        else:
            if not picked:
                phi, direction = min(facings, key=lambda f: abs((bearing - f[0] + math.pi)
                                                                % (2 * math.pi) - math.pi))
                picked = True
            err = (bearing - phi + math.pi) % (2 * math.pi) - math.pi
            if state == TURN:
                want_vx, want_vy = 0.0, 0.0
                want_wz = max(-cfg.turn_wz_max, min(cfg.turn_wz_max, cfg.k_steer * err))
                if abs(err) >= cfg.face_tol_rad:                  # TURN floor (parity with mover_twist)
                    want_wz = math.copysign(max(abs(want_wz), cfg.turn_wz_min), err)
            else:
                want_vx, want_vy = _scalar_drive_velocity(direction, (x_base, y_base), cfg)
                want_wz = max(-cfg.walk_wz_max, min(cfg.walk_wz_max, cfg.k_steer * err))
        step = cfg.max_accel * cfg.step_dt
        vx += max(-step, min(step, want_vx - vx))
        vy += max(-step, min(step, want_vy - vy))
        wz += max(-step, min(step, want_wz - wz))
        if state == TURN:
            err = (bearing - phi + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < cfg.face_tol_rad and abs(wz - want_wz) < cfg.wz_eps:
                settle += 1
                if settle >= cfg.settle_steps:
                    state = WALK
            else:
                settle = 0
        active = 0.0 if settled else 1.0                # frozen once settled (matches batched)
        bx += cfg.step_dt * active * (c * vx - s * vy)
        by += cfg.step_dt * active * (s * vx + c * vy)
        yw += cfg.step_dt * wz * active
        if state == STOP and abs(vx) < 1e-3 and abs(vy) < 1e-3 and abs(wz) < 1e-3:
            settled = True
        if settled:
            break
    return bx, by, yw, settled


def _scalar_table_contact_stop_ref(base_xy, cube_xy, edge_xy, half):
    """Scalar reference for ``table_contact_stop_batched`` -- python-float port of the torso-safe
    standoff geometry. Returns ``(sx, sy)``."""
    bx, by = float(base_xy[0]), float(base_xy[1])
    tex, tey = float(edge_xy[0]) - bx, float(edge_xy[1]) - by
    d_edge = math.hypot(tex, tey)
    tcx, tcy = float(cube_xy[0]) - bx, float(cube_xy[1]) - by
    rng = math.hypot(tcx, tcy)
    if not (d_edge > 1e-6) or not (rng > 1e-6):
        return bx, by
    nx, ny = tex / d_edge, tey / d_edge
    rx, ry = tcx / rng, tcy / rng
    denom = rx * nx + ry * ny
    if not (denom > 1e-6):
        return bx, by
    reach = max(d_edge / denom - float(half), 0.0)
    return bx + reach * rx, by + reach * ry


def _self_test() -> None:
    torch.manual_seed(0)

    # --- gaze_aim: round-trip aim_error 0 for both camera datums, off-axis, over a batch ---
    N, K = 32, 2
    aim_sign = torch.tensor([1.0, -1.0])                        # left, right
    cam_pos = torch.randn(N, K, 3) * 0.05 + torch.tensor([0.02, 0.1, 0.66])
    # targets within ROM: in front-ish, moderate offsets so |yaw|,|pitch| < ROM
    target = cam_pos + torch.stack([
        torch.rand(N, K) * 0.6 + 0.3,                           # +x forward
        (torch.rand(N, K) - 0.5) * 0.6,                         # +-y
        -(torch.rand(N, K) * 0.6 + 0.1),                        # down
    ], dim=-1)
    yaw, pitch = gaze_aim(target, cam_pos, aim_sign, 4.7124, 1.5708)
    fwd = gaze_forward(yaw, pitch, aim_sign)                    # (N, K, 3)
    d = (target - cam_pos)
    d = d / d.norm(dim=-1, keepdim=True)
    cos = (fwd * d).sum(-1).clamp(-1.0, 1.0)
    aim_err = torch.acos(cos)
    assert aim_err.max() < 2e-3, aim_err.max()   # ~0.1 deg; float32 acos near 1 is ill-conditioned

    # right cam yaw sense is opposite the (buggy) un-flipped formula -> guard the sign explicitly:
    # a target to the LEFT (+y) of the left cam must give NEGATIVE yaw joint (optical swings to -y).
    cp = torch.tensor([[0.0, 0.1, 0.66]])
    tp = cp + torch.tensor([[0.4, 0.3, -0.2]])
    yj, _ = gaze_aim(tp, cp, torch.tensor([1.0]), 4.7124, 1.5708)
    assert yj.item() < 0.0, yj.item()

    # --- mover_twist: exact parity vs the scalar reference over random trajectories ---
    cfg = MoverCfg(step_dt=0.02)
    face_dirs = ((1.0, 0.0), (-1.0, 0.0))
    face_phi, drive_dirs = facings_from_dirs(face_dirs)
    Nt, T = 5, 400
    # each env: a fixed target bearing (target recedes so it never enters STOP by itself)
    angles = torch.tensor([0.2, 2.5, -1.3, math.pi - 0.1, -2.9])
    targets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * 3.0   # (Nt, 2)
    st = MoverState.init(Nt)
    batched = []
    for _ in range(T):
        tw = mover_twist(targets, st, face_phi, drive_dirs, cfg)
        batched.append(tw.clone())                              # (Nt, 3) vx, vy, wz
    batched = torch.stack(batched, dim=1)                       # (Nt, T, 3)
    for e in range(Nt):
        ref = _scalar_mover_ref([(float(targets[e, 0]), float(targets[e, 1]))] * T, face_dirs, cfg)
        ref = torch.tensor(ref)                                 # (T, 3), vy == 0 (cfg.cruise_vy_max == 0.0)
        assert torch.allclose(batched[e], ref, atol=1e-6), (e, (batched[e] - ref).abs().max())

    # --- mover_twist: vy cross-track parity vs the scalar reference, cruise_vy_max > 0 opted in ---
    vy_cfg = MoverCfg(step_dt=0.02, cruise_vy_max=0.6, k_lateral=0.6)
    vy_targets = torch.stack([torch.cos(angles) * 3.0, torch.sin(angles) * 3.0 + torch.tensor(
        [0.6, -0.4, 0.3, -0.5, 0.2])], dim=-1)                   # same bearings, offset laterally over time
    st_vy = MoverState.init(Nt)
    batched_vy = []
    for _ in range(T):
        tw = mover_twist(vy_targets, st_vy, face_phi, drive_dirs, vy_cfg)
        batched_vy.append(tw.clone())
    batched_vy = torch.stack(batched_vy, dim=1)
    for e in range(Nt):
        ref = _scalar_mover_ref([(float(vy_targets[e, 0]), float(vy_targets[e, 1]))] * T, face_dirs, vy_cfg)
        ref = torch.tensor(ref)
        assert torch.allclose(batched_vy[e], ref, atol=1e-6), (e, (batched_vy[e] - ref).abs().max())
    # Static off-axis targets stay in TURN because this unit test does not integrate the base yaw. Enter
    # WALK explicitly to exercise lateral correction, which is intentionally disabled during TURN/STOP.
    vy_walk_state = MoverState.init(1)
    vy_walk_state.state.fill_(WALK)
    vy_walk_state.picked.fill_(True)
    vy_walk_state.drive_dir[:, 0].fill_(1.0)
    vy_walk = None
    for _ in range(20):
        vy_walk = mover_twist(torch.tensor([[3.0, 2.0]]), vy_walk_state, face_phi, drive_dirs, vy_cfg)
    assert abs(float(vy_walk[0, 1]) - vy_cfg.cruise_vy_max) < 1e-6, vy_walk

    # Four-direction capability: an exactly lateral target selects +Y, reaches WALK without yaw, and
    # commands the full lateral speed. This is V2's no-body-turn path.
    cardinal_dirs = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    cardinal_phi, cardinal_drive_dirs = facings_from_dirs(cardinal_dirs)
    side_state = MoverState.init(1)
    side_twist = None
    for _ in range(20):
        side_twist = mover_twist(torch.tensor([[0.0, 3.0]]), side_state, cardinal_phi, cardinal_drive_dirs, vy_cfg)
    assert torch.allclose(side_state.drive_dir[0], torch.tensor([0.0, 1.0]))
    assert abs(float(side_twist[0, 0])) < 1e-6 and abs(float(side_twist[0, 1]) - 0.6) < 1e-6, side_twist
    assert abs(float(side_twist[0, 2])) < 1e-6, side_twist

    # --- mover_twist: STOP latch ramps both channels toward zero ---
    st2 = MoverState.init(1)
    stop = torch.tensor([False])
    tgt = torch.tensor([[3.0, 0.5]])
    for i in range(200):
        if i == 120:
            stop = torch.tensor([True])
        mover_twist(tgt, st2, face_phi, drive_dirs, cfg, stop_mask=stop)
    assert st2.state.item() == STOP and abs(st2.vx.item()) < 1e-3 and abs(st2.wz.item()) < 1e-3

    # --- walk_to_target_radius_batched: exact per-env parity vs the scalar walk reference ---
    # float64 so the ~1e3-step integration matches the python-float ref to ~1e-12 (float32 would drift);
    # cruise_vx defaults to 0.4 in MoverCfg (the deploy-verified speed) -- the batched twin's old
    # hardcoded 0.5 would break this assert.
    walk_cfg = MoverCfg(step_dt=0.02)
    fd = ((1.0, 0.0), (-1.0, 0.0))
    starts = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, -0.5], [0.0, 0.0], [-0.3, 0.2]], dtype=torch.float64)
    yaws0 = torch.tensor([0.0, 0.0, 1.2, 0.0, -0.5], dtype=torch.float64)
    tgts = torch.tensor([[2.0, 0.0], [-1.5, 0.3], [2.0, 1.0], [0.0, -1.8], [1.2, 0.9]], dtype=torch.float64)
    # Tol 1e-8 (not 1e-12): the turn_wz_min floor holds wz at a constant clamp through the final convergence,
    # a non-smooth point that shifts the trajectory just enough to compound the per-step float64 rounding to
    # ~1e-8 over the ~1e3-step integration. Still tight enough to catch any real batched/scalar logic drift.
    for r in (0.2, float("inf")):   # finite radius = walk-in; inf = turn-in-place only
        bxy, yw, stld = walk_to_target_radius_batched(starts, yaws0, tgts, fd, walk_cfg, r)
        for e in range(starts.shape[0]):
            rbx, rby, ryw, rst = _scalar_walk_ref(starts[e], yaws0[e], tgts[e], fd, walk_cfg, r)
            assert abs(float(bxy[e, 0]) - rbx) < 1e-8 and abs(float(bxy[e, 1]) - rby) < 1e-8, (e, r)
            assert abs(float(yw[e]) - ryw) < 1e-8, (e, r)
            assert bool(stld[e]) == rst, (e, r)

    # V2 cardinal mode: a +Y target strafes directly with no yaw excursion and reaches the stop radius.
    lateral_cfg = MoverCfg(step_dt=0.02, cruise_vy_max=0.6)
    side_base, side_yaw, side_settled = walk_to_target_radius_batched(
        torch.tensor([[0.0, 0.0]], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([[0.0, 2.0]], dtype=torch.float64), cardinal_dirs, lateral_cfg, 0.2)
    assert bool(side_settled[0]) and abs(float(side_base[0, 0])) < 1e-8, (side_base, side_yaw, side_settled)
    assert 1.75 <= float(side_base[0, 1]) <= 2.0 and abs(float(side_yaw[0])) < 1e-8, (side_base, side_yaw)

    # --- table_contact_stop_batched: per-env parity vs scalar (incl. degenerate branches) ---
    base_s = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    edge_s = torch.tensor([[0.5, 0.0], [0.4, 0.1], [0.0, 0.0], [0.5, 0.5]], dtype=torch.float64)  # rows 3,4 degenerate
    cube_s = torch.tensor([[1.0, 0.0], [0.9, 0.3], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    half = 0.15
    stop_b = table_contact_stop_batched(base_s, cube_s, edge_s, half)
    for e in range(base_s.shape[0]):
        rsx, rsy = _scalar_table_contact_stop_ref(base_s[e], cube_s[e], edge_s[e], half)
        assert abs(float(stop_b[e, 0]) - rsx) < 1e-9 and abs(float(stop_b[e, 1]) - rsy) < 1e-9, e

    # --- shapes/device: large batch on GPU (the real parallel case) without python fallbacks ---
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Nl = 4096
    tb = torch.randn(Nl, K, 3, device=dev)
    cb = torch.randn(Nl, K, 3, device=dev)
    y, p = gaze_aim(tb, cb, aim_sign.to(dev), 4.7124, 1.5708)
    assert y.shape == (Nl, K) and p.shape == (Nl, K)
    assert y.device.type == dev and p.device.type == dev, (y.device, dev)
    face_phi_d, drive_dirs_d = facings_from_dirs(face_dirs, device=dev)
    stL = MoverState.init(Nl, device=dev)
    twL = mover_twist(torch.randn(Nl, 2, device=dev), stL, face_phi_d, drive_dirs_d, cfg)
    assert twL.shape == (Nl, 3) and torch.all(twL[:, 1] == 0.0)
    assert twL.device.type == dev and stL.vx.device.type == dev, (twL.device, dev)

    wbxy, wyw, wst = walk_to_target_radius_batched(
        torch.randn(Nl, 2, device=dev), torch.randn(Nl, device=dev),
        torch.randn(Nl, 2, device=dev), face_dirs, cfg, 0.2, max_steps=50)
    assert wbxy.shape == (Nl, 2) and wyw.shape == (Nl,) and wst.shape == (Nl,)
    assert wbxy.device.type == dev and wst.device.type == dev, (wbxy.device, dev)
    sb = table_contact_stop_batched(torch.randn(Nl, 2, device=dev), torch.randn(Nl, 2, device=dev),
                                    torch.randn(Nl, 2, device=dev), 0.15)
    assert sb.shape == (Nl, 2) and sb.device.type == dev, (sb.device, dev)

    print("control_vec self-test: PASS")


if __name__ == "__main__":
    _self_test()
