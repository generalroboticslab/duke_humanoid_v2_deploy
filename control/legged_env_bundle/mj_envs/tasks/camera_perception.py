"""Camera perception primitives for the active-vision pick-and-place stack.

Pure-tensor geometry + belief state shared across all phases of
plan/ACTIVE_VISION_PICK_PLACE_PLAN.md. Kept free of mjlab/env imports so the
geometry and belief math are unit-testable in isolation (run this file directly).

Conventions (verified against asset/duke_v2/head_cam/head_camera_dual.xml +
humanoid_v21_constants.py FOV frustum):
  - Camera optical site (`left_rgb`/`right_rgb`) frame: +Z = optical axis OUT,
    +X = image right, +Y = image down. This is the SAME convention the FOV
    frustum overlay uses (do NOT mix with MuJoCo `<camera>` which looks down -Z).
  - `site_xmat` is the camera→world rotation R_wc (its columns are the camera
    axes expressed in world). World→camera is R_wc^T.
  - All poses world frame unless a name says base/cam frame.

Design decisions (why):
  - Belief stored in WORLD frame, exposed in base frame on demand. A base-frame
    store would silently drift when the floating base moves (a stale estimate
    would translate/rotate with the base). World store + per-step base projection
    keeps an unseen target's estimate fixed in the world. (Plan invariant.)
  - Direction to a target is exposed as a base-frame UNIT VECTOR (not a yaw
    angle): the ±270° yaw ROM over 360° targets has a wrap discontinuity that a
    raw angle encodes discontinuously. (Plan invariant.)
  - FOV/range test returns the continuous signed angular errors too, so the same
    call feeds both the boolean detection gate AND the dense aim reward.
"""

from __future__ import annotations

import math

import torch


def xmat_optical_axis(site_xmat: torch.Tensor) -> torch.Tensor:
    """Optical axis (camera +Z, world frame) from a site rotation matrix.

    Args:
        site_xmat: (..., 9) row-major or (..., 3, 3) camera→world rotation R_wc.
    Returns:
        (..., 3) unit world-frame optical axis = third column of R_wc.
    """
    if site_xmat.shape[-1] == 9:
        site_xmat = site_xmat.reshape(*site_xmat.shape[:-1], 3, 3)
    return site_xmat[..., :, 2]


def target_in_camera_frame(
    cam_pos: torch.Tensor, site_xmat: torch.Tensor, target_pos: torch.Tensor
) -> torch.Tensor:
    """Express a world target in the camera frame (x right, y down, z out).

    Args:
        cam_pos:    (..., 3) camera site world position.
        site_xmat:  (..., 9) or (..., 3, 3) camera→world rotation R_wc.
        target_pos: (..., 3) target world position.
    Returns:
        (..., 3) target position in camera frame = R_wc^T @ (target - cam_pos).
    """
    if site_xmat.shape[-1] == 9:
        site_xmat = site_xmat.reshape(*site_xmat.shape[:-1], 3, 3)
    rel = (target_pos - cam_pos).unsqueeze(-1)            # (..., 3, 1)
    # R_wc^T @ rel : batched matvec via transpose on last two dims.
    p_cam = torch.matmul(site_xmat.transpose(-1, -2), rel).squeeze(-1)
    return p_cam


def fov_detect(
    cam_pos: torch.Tensor,
    site_xmat: torch.Tensor,
    target_pos: torch.Tensor,
    h_half: float,
    v_half: float,
    near: float,
    far: float,
) -> dict[str, torch.Tensor]:
    """Geometric FOV + range membership (occlusion handled separately).

    A target is geometrically visible when it is in front of the lens (depth>0),
    within the horizontal/vertical half-angles, and within [near, far]. The
    horizontal/vertical angles use the pinhole convention h=atan2(x, z),
    v=atan2(y, z) so they are the natural image-plane angles.

    Args:
        cam_pos/site_xmat/target_pos: see target_in_camera_frame (any leading batch).
        h_half/v_half: horizontal/vertical FOV half-angles (rad).
        near/far: depth-range clamp (m), matching the RGB frustum.
    Returns:
        dict with (all leading-batch shaped):
          in_fov     : bool, geometric visibility gate.
          h_angle    : signed horizontal angle (rad), +right.
          v_angle    : signed vertical angle (rad), +down.
          aim_error  : unsigned angle between optical axis and target dir (rad).
          range      : Euclidean distance lens→target (m).
    """
    p_cam = target_in_camera_frame(cam_pos, site_xmat, target_pos)
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    rng = torch.linalg.norm(p_cam, dim=-1)
    # Guard atan2 with a tiny depth floor so a target exactly on the lens plane
    # (z=0) maps to ±pi/2 rather than NaN; the depth>0 gate rejects it anyway.
    h_angle = torch.atan2(x, z.clamp_min(1e-6))
    v_angle = torch.atan2(y, z.clamp_min(1e-6))
    aim_error = torch.acos((z / rng.clamp_min(1e-6)).clamp(-1.0, 1.0))
    in_fov = (
        (z > 0.0)
        & (h_angle.abs() <= h_half)
        & (v_angle.abs() <= v_half)
        & (rng >= near)
        & (rng <= far)
    )
    return {
        "in_fov": in_fov,
        "h_angle": h_angle,
        "v_angle": v_angle,
        "aim_error": aim_error,
        "range": rng,
    }


def world_to_base_dir(
    root_pos: torch.Tensor, root_xmat: torch.Tensor, target_pos: torch.Tensor
) -> torch.Tensor:
    """Unit direction base→target expressed in the base frame.

    Continuous encoding for 360° targets (no yaw-wrap discontinuity).

    Args:
        root_pos:   (..., 3) base world position.
        root_xmat:  (..., 9) or (..., 3, 3) base→world rotation R_wb.
        target_pos: (..., 3) target world position.
    Returns:
        (..., 3) unit vector base→target in base frame.
    """
    if root_xmat.shape[-1] == 9:
        root_xmat = root_xmat.reshape(*root_xmat.shape[:-1], 3, 3)
    rel = (target_pos - root_pos).unsqueeze(-1)
    d_base = torch.matmul(root_xmat.transpose(-1, -2), rel).squeeze(-1)
    return d_base / torch.linalg.norm(d_base, dim=-1, keepdim=True).clamp_min(1e-6)


class BeliefBuffer:
    """POMDP belief over K targets, batched over N envs. World-store/base-expose.

    Per (env, target): estimated world position, confidence in [0, 1], seen-ever
    flag. On a detection the estimate snaps to the measurement and confidence
    resets to 1; without a detection confidence decays geometrically (the
    estimate is held fixed in WORLD frame so it does not drift under locomotion).

    This is the ONLY interface coupling the camera head to the (later) manip head
    (plan invariant): manip reads the belief, never ground truth.

    Args:
        num_envs, num_targets: buffer dims.
        decay: per-step confidence multiplier when a target is not detected.
        device, dtype: tensor placement.
    """

    def __init__(
        self,
        num_envs: int,
        num_targets: int,
        decay: float = 0.95,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_envs = num_envs
        self.num_targets = num_targets
        self.decay = decay
        self.device = torch.device(device)
        shape = (num_envs, num_targets)
        self.est_pos = torch.zeros(*shape, 3, device=device, dtype=dtype)
        self.confidence = torch.zeros(*shape, device=device, dtype=dtype)
        self.seen_ever = torch.zeros(*shape, device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear belief for the given envs (all by default)."""
        if env_ids is None:
            env_ids = slice(None)
        self.est_pos[env_ids] = 0.0
        self.confidence[env_ids] = 0.0
        self.seen_ever[env_ids] = False

    def update(self, detected: torch.Tensor, measured_pos: torch.Tensor) -> None:
        """Fold one step of detections into the belief.

        Args:
            detected:     (N, K) bool, target detected THIS step.
            measured_pos: (N, K, 3) world position measurement (read only where
                          detected; undetected entries ignored).
        """
        self.confidence = torch.where(
            detected, torch.ones_like(self.confidence), self.confidence * self.decay
        )
        self.est_pos = torch.where(detected.unsqueeze(-1), measured_pos, self.est_pos)
        self.seen_ever = self.seen_ever | detected

    def expose_base(
        self, root_pos: torch.Tensor, root_xmat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project the world-frame belief into the current base frame.

        Args:
            root_pos:  (N, 3) base world position.
            root_xmat: (N, 9) or (N, 3, 3) base→world rotation.
        Returns:
            dict:
              dir_base   : (N, K, 3) unit base→est direction in base frame.
              dist       : (N, K) base→est distance (m).
              confidence : (N, K).
              seen_ever  : (N, K) bool.
        """
        if root_xmat.shape[-1] == 9:
            root_xmat = root_xmat.reshape(self.num_envs, 3, 3)
        rel = self.est_pos - root_pos.unsqueeze(1)                 # (N, K, 3) world
        R_bw = root_xmat.transpose(-1, -2).unsqueeze(1)            # (N, 1, 3, 3)
        rel_base = torch.matmul(R_bw, rel.unsqueeze(-1)).squeeze(-1)  # (N, K, 3)
        dist = torch.linalg.norm(rel_base, dim=-1)
        dir_base = rel_base / dist.clamp_min(1e-6).unsqueeze(-1)
        return {
            "dir_base": dir_base,
            "dist": dist,
            "confidence": self.confidence,
            "seen_ever": self.seen_ever,
        }


def _rotate_vec_by_rotvec(v: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues-rotate batched unit vectors v by a per-env rotation vector.

    Args:
        v:      (N, K, 3) vectors (need not be unit; not renormalized here).
        rotvec: (N, 3) axis*angle (rad); applied to every K vector of its env.
    Returns:
        (N, K, 3) rotated vectors. theta->0 limit is the identity (cos->1, sin->0).
    """
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)        # (N, 1)
    axis = (rotvec / theta.clamp_min(1e-8)).unsqueeze(1)           # (N, 1, 3)
    theta = theta.unsqueeze(1)                                     # (N, 1, 1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    cross = torch.cross(axis.expand_as(v), v, dim=-1)             # (N, K, 3)
    dot = (axis * v).sum(-1, keepdim=True)                         # (N, K, 1)
    return v * cos + cross * sin + axis * dot * (1.0 - cos)


class BeliefBufferBase:
    """Deployable base-frame belief (Route A): same 12D obs as BeliefBuffer, NO world frame.

    ``BeliefBuffer`` stores the estimate in WORLD and re-projects it through the ABSOLUTE base pose
    (root_pos + root_quat) on every obs compute — both inputs are undeployable (hardware has no
    ground-truth world target position and no absolute world base pose). This buffer keeps the belief
    directly in the BASE frame so every update uses only deployable signals:

      - DETECT: snap bearing (unit base->target) + range to a measurement that, on hardware, is the
        AprilTag/PnP target-in-camera pose composed with the gimbal+base encoder kinematics
        (base<-camera). The absolute base pose CANCELS in that composition
        (target_in_base = R_bc @ p_cam + cam_in_base, R_bc and cam_in_base from joint encoders only),
        so the stored quantity needs no world frame. (In sim the env computes the identical number via
        R_bw @ (target_w - root_pos); the world frame cancels, so it is deployable-equivalent.)
      - HELD (undetected): rotate the stored bearing by the base's per-step rotation read from the
        body-frame gyro (apply_egomotion: rotvec = -omega_body * dt) so a world-fixed target's
        BASE-frame bearing tracks the turning base WITHOUT absolute pose. Range is held constant — the
        single approximation vs the world store (standing => base translation negligible; the residual
        drift is bounded by re-detection, sub-second). Confidence decays geometrically as in BeliefBuffer.

    ``expose_base`` IGNORES its root arguments (kept only so it is a drop-in for
    ``camera_belief_obs``, which calls ``env.belief.expose_base(root_pos, root_mat)``) and returns the
    same dict layout as ``BeliefBuffer.expose_base``.

    Distribution note (vs BeliefBuffer, for a FROZEN swap): a never-seen target has bearing 0 here
    (vs BeliefBuffer's unit dir toward the world origin). Both are gated off by seen_ever=0; the
    mismatch touches only the pre-first-detection slice (ttfd ~50 steps, seen_ever saturates ~0.98),
    and the deployable value (no bearing until detected) is the honest one.
    """

    def __init__(
        self,
        num_envs: int,
        num_targets: int,
        decay: float = 0.95,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_envs = num_envs
        self.num_targets = num_targets
        self.decay = decay
        self.device = torch.device(device)
        shape = (num_envs, num_targets)
        self.bearing = torch.zeros(*shape, 3, device=device, dtype=dtype)   # unit base->target
        self.rng = torch.zeros(*shape, device=device, dtype=dtype)          # base->target distance (m)
        self.confidence = torch.zeros(*shape, device=device, dtype=dtype)
        self.seen_ever = torch.zeros(*shape, device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear belief for the given envs (all by default)."""
        if env_ids is None:
            env_ids = slice(None)
        self.bearing[env_ids] = 0.0
        self.rng[env_ids] = 0.0
        self.confidence[env_ids] = 0.0
        self.seen_ever[env_ids] = False

    def apply_egomotion(self, omega_body: torch.Tensor, dt: float) -> None:
        """Rotate every stored bearing by the base's body-frame rotation over one step (gyro bridge).

        A world-fixed direction expressed in the base frame transforms as bearing(t) =
        exp(-[omega*dt]_x) bearing(t-1) when the base rotates by omega*dt (body frame). Applied to ALL
        targets — freshly detected ones are overwritten by the snap in ``update`` immediately after, so
        the egomotion only persists on HELD targets. Args: omega_body (N,3) gyro (rad/s), dt (s)."""
        rotvec = -omega_body * dt
        self.bearing = _rotate_vec_by_rotvec(self.bearing, rotvec)
        self.bearing = self.bearing / torch.linalg.norm(
            self.bearing, dim=-1, keepdim=True
        ).clamp_min(1e-6)

    def update(
        self, detected: torch.Tensor, meas_dir_base: torch.Tensor, meas_dist: torch.Tensor
    ) -> None:
        """Fold one step of detections into the base-frame belief.

        Args:
            detected:      (N, K) bool, target detected THIS step.
            meas_dir_base: (N, K, 3) unit base->target measurement (read only where detected).
            meas_dist:     (N, K) base->target distance measurement (read only where detected).
        """
        self.confidence = torch.where(
            detected, torch.ones_like(self.confidence), self.confidence * self.decay
        )
        self.bearing = torch.where(detected.unsqueeze(-1), meas_dir_base, self.bearing)
        self.rng = torch.where(detected, meas_dist, self.rng)
        self.seen_ever = self.seen_ever | detected

    def expose_base(self, root_pos=None, root_xmat=None) -> dict[str, torch.Tensor]:
        """Return the belief in base frame. Root args ignored (no world pose needed) — present only
        so this is a drop-in for ``BeliefBuffer.expose_base`` in ``camera_belief_obs``."""
        return {
            "dir_base": self.bearing,
            "dist": self.rng,
            "confidence": self.confidence,
            "seen_ever": self.seen_ever,
        }


# --------------------------------------------------------------------------- #
# Self-test (run: python mj_envs/tasks/camera_perception.py). No mujoco needed.
# --------------------------------------------------------------------------- #
def _identity_xmat(*batch: int) -> torch.Tensor:
    return torch.eye(3).reshape(9).expand(*batch, 9).clone()


def _self_test() -> None:
    torch.manual_seed(0)

    # --- xmat_optical_axis: identity → +Z ---
    ax = xmat_optical_axis(_identity_xmat(5))
    assert torch.allclose(ax, torch.tensor([0.0, 0.0, 1.0]).expand(5, 3)), ax

    # --- target_in_camera_frame: identity cam at origin, target on +Z ---
    cam_pos = torch.zeros(3)
    p = target_in_camera_frame(cam_pos, _identity_xmat(), torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(p, torch.tensor([0.0, 0.0, 2.0])), p

    h_half, v_half, near, far = 0.7854, 0.5672, 0.28, 3.0  # 45°, 32.5°

    # --- fov_detect: dead-ahead target visible, zero aim error ---
    d = fov_detect(cam_pos, _identity_xmat(), torch.tensor([0.0, 0.0, 1.5]),
                   h_half, v_half, near, far)
    assert bool(d["in_fov"]) and d["aim_error"].item() < 1e-5, d

    # --- behind camera → not visible ---
    d = fov_detect(cam_pos, _identity_xmat(), torch.tensor([0.0, 0.0, -1.5]),
                   h_half, v_half, near, far)
    assert not bool(d["in_fov"]), d

    # --- just outside horizontal half-angle → not visible; just inside → visible ---
    z = 1.0
    x_out = z * torch.tan(torch.tensor(h_half + 0.05))
    x_in = z * torch.tan(torch.tensor(h_half - 0.05))
    assert not bool(fov_detect(cam_pos, _identity_xmat(),
                               torch.tensor([x_out, 0.0, z]), h_half, v_half, near, far)["in_fov"])
    assert bool(fov_detect(cam_pos, _identity_xmat(),
                           torch.tensor([x_in, 0.0, z]), h_half, v_half, near, far)["in_fov"])

    # --- range gates ---
    assert not bool(fov_detect(cam_pos, _identity_xmat(),
                               torch.tensor([0.0, 0.0, far + 0.5]), h_half, v_half, near, far)["in_fov"])
    assert not bool(fov_detect(cam_pos, _identity_xmat(),
                               torch.tensor([0.0, 0.0, near - 0.1]), h_half, v_half, near, far)["in_fov"])

    # --- sign conventions: +x world (right) → +h_angle; +y world (down) → +v_angle ---
    d = fov_detect(cam_pos, _identity_xmat(), torch.tensor([0.3, 0.2, 1.0]),
                   h_half, v_half, near, far)
    assert d["h_angle"].item() > 0 and d["v_angle"].item() > 0, d

    # --- world_to_base_dir: 90° base yaw rotates a +x_world target to -y_base ---
    c, s = torch.cos(torch.tensor(1.5708)), torch.sin(torch.tensor(1.5708))
    R_wb = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).reshape(9)
    dir_b = world_to_base_dir(torch.zeros(3), R_wb, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(dir_b, torch.tensor([0.0, -1.0, 0.0]), atol=1e-5), dir_b

    # --- BeliefBuffer: detect → confident world store; decay; base-expose fixed under base move ---
    N, K = 4, 2
    bel = BeliefBuffer(N, K, decay=0.5)
    det = torch.zeros(N, K, dtype=torch.bool); det[:, 0] = True
    meas = torch.zeros(N, K, 3); meas[:, 0] = torch.tensor([2.0, 0.0, 0.0])
    bel.update(det, meas)
    assert torch.all(bel.confidence[:, 0] == 1.0) and torch.all(bel.seen_ever[:, 0])
    assert torch.all(bel.confidence[:, 1] == 0.0)
    bel.update(torch.zeros(N, K, dtype=torch.bool), torch.zeros(N, K, 3))
    assert torch.allclose(bel.confidence[:, 0], torch.full((N,), 0.5))   # decayed
    assert torch.allclose(bel.est_pos[:, 0], meas[:, 0])                  # held fixed
    # Base at origin, identity → dir +x, dist 2. Move base to the target → dist 0-ish.
    e = bel.expose_base(torch.zeros(N, 3), _identity_xmat(N))
    assert torch.allclose(e["dir_base"][:, 0], torch.tensor([1.0, 0.0, 0.0]).expand(N, 3), atol=1e-5)
    assert torch.allclose(e["dist"][:, 0], torch.full((N,), 2.0), atol=1e-5)
    e2 = bel.expose_base(torch.tensor([2.0, 0.0, 0.0]).expand(N, 3).clone(), _identity_xmat(N))
    assert torch.all(e2["dist"][:, 0] < 1e-4)        # world store fixed; only base moved

    # --- reset clears ---
    bel.reset(torch.tensor([0, 1]))
    assert torch.all(bel.confidence[:2] == 0.0) and torch.all(bel.confidence[2:, 0] > 0.0)

    # --- BeliefBufferBase: snap, decay, gyro egomotion keeps a held bearing world-fixed ---
    N, K = 3, 2
    belb = BeliefBufferBase(N, K, decay=0.5)
    det = torch.zeros(N, K, dtype=torch.bool); det[:, 0] = True
    dir0 = torch.zeros(N, K, 3); dir0[:, 0, 0] = 1.0                       # target0 dead ahead (+x base)
    dist0 = torch.zeros(N, K); dist0[:, 0] = 2.0
    belb.update(det, dir0, dist0)
    assert torch.all(belb.confidence[:, 0] == 1.0) and torch.all(belb.seen_ever[:, 0])
    assert torch.all(belb.confidence[:, 1] == 0.0)
    assert torch.allclose(belb.bearing[:, 0], dir0[:, 0]) and torch.all(belb.rng[:, 0] == 2.0)
    # No-detect step with ZERO egomotion: bearing held, confidence decays.
    belb.apply_egomotion(torch.zeros(N, 3), dt=0.02)
    belb.update(torch.zeros(N, K, dtype=torch.bool), torch.zeros(N, K, 3), torch.zeros(N, K))
    assert torch.allclose(belb.bearing[:, 0], dir0[:, 0], atol=1e-6)        # held fixed
    assert torch.allclose(belb.confidence[:, 0], torch.full((N,), 0.5))     # decayed
    # Gyro bridge: base yaws +90deg (omega_z*dt = +pi/2). A world-fixed +x_base target must move to
    # -y_base (cross-check vs world_to_base_dir under the same 90deg base yaw).
    omega = torch.zeros(N, 3); omega[:, 2] = (math.pi / 2) / 0.02           # so omega*dt = pi/2
    belb.apply_egomotion(omega, dt=0.02)
    c90, s90 = math.cos(1.5708), math.sin(1.5708)
    R_wb90 = torch.tensor([[c90, -s90, 0.0], [s90, c90, 0.0], [0.0, 0.0, 1.0]]).reshape(9)
    expect = world_to_base_dir(torch.zeros(3), R_wb90, torch.tensor([1.0, 0.0, 0.0]))  # [0,-1,0]
    assert torch.allclose(belb.bearing[:, 0], expect.expand(N, 3), atol=1e-5), belb.bearing[:, 0]
    # reset clears
    belb.reset(torch.tensor([0]))
    assert torch.all(belb.confidence[0] == 0.0) and torch.all(belb.confidence[1:, 0] > 0.0)

    print("camera_perception self-test: PASS")


if __name__ == "__main__":
    _self_test()
