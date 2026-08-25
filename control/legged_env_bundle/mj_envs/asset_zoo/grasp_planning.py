"""
Reachability-aware grasp planning for humanoid upper body.

Implements the two-stage approach from Chen et al. (2018):
  1. Score all N grasp candidates cheaply with the ReachabilityModel.
  2. Run full IK only on the top-K candidates.

The energy function per candidate is:
  E(c) = GQ(c) + λ · R(EE_in_pelvis_frame)

where GQ is the grasp quality (approach alignment with object normal) and R is the
learned reachability score. High E → run IK; low E → skip.

Usage (standalone test):
    python mj_envs/asset_zoo/grasp_planning.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from mj_envs.utils.reachability import ReachabilityModel, world_to_pelvis_frame


# ---------------------------------------------------------------------------
# Quaternion helpers (local copy — avoids circular imports)
# ---------------------------------------------------------------------------

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q. q: (..., 4) wxyz, v: (..., 3)."""
    w, xyz = q[..., 0:1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)

def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1*q2. Both (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def _quat_from_two_vectors(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """Shortest-arc quaternion rotating unit vector v_from to v_to. Both (3,).

    Returns (4,) wxyz unit quaternion.  Handles anti-parallel vectors gracefully.
    """
    cross = torch.cross(v_from, v_to, dim=-1)
    dot   = (v_from * v_to).sum(-1, keepdim=True)
    # w = cos(θ/2) * |v_from||v_to| = sqrt((1 + cos(θ))/2) when inputs are unit
    # Formula: q = [1 + dot, cross] normalised
    q = torch.cat([1.0 + dot, cross], dim=-1)
    norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q = q / norm
    # Degenerate (anti-parallel): return 180° around any perpendicular axis
    is_anti = (dot < -1 + 1e-6)
    if is_anti.any():
        perp = torch.zeros_like(v_from)
        if v_from[0].abs() < 0.9:
            perp[0] = 0.0; perp[1] = -v_from[2]; perp[2] = v_from[1]
        else:
            perp[0] = -v_from[1]; perp[1] = v_from[0]; perp[2] = 0.0
        perp = perp / perp.norm().clamp(min=1e-8)
        q_anti = torch.cat([torch.zeros(1), perp], dim=0)   # [0, perp] = 180° around perp
        q = torch.where(is_anti.squeeze(), q_anti, q)
    return q


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidate:
    """A single candidate end-effector grasp pose in world frame.

    Attributes:
        ee_pos_world:   (3,) EE position in world frame.
        ee_quat_world:  (4,) EE orientation wxyz in world frame.
                        The wrist z-axis (ee_quat applied to [0,0,1]) points toward
                        the object (approach direction).
        approach_dir:   (3,) unit vector from EE toward the object (= −approach approach_axis).
        arm:            Which arm this candidate is assigned to.
        score:          Combined energy score (higher is better). None before scoring.
        reachability:   Raw reachability score from the model. None before scoring.
    """
    ee_pos_world:  torch.Tensor
    ee_quat_world: torch.Tensor
    approach_dir:  torch.Tensor
    arm:           Literal["left", "right"]
    score:         float | None = None
    reachability:  float | None = None


@dataclass
class GraspResult:
    """Output of plan_grasp(): the winning candidate and its IK joint solution.

    Attributes:
        candidate:    The GraspCandidate that passed IK verification.
        arm_joints:   (2*K,) absolute joint positions for both arms: [left…, right…].
        pos_err:      Final EE position error (m) after IK convergence.
        ori_err:      Final EE orientation error (rad) after IK convergence.
    """
    candidate:  GraspCandidate
    arm_joints: torch.Tensor
    pos_err:    float
    ori_err:    float   # approach direction error (rad) — angle between achieved/target wrist z-axis; excludes roll


# ---------------------------------------------------------------------------
# Grasp candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    object_pos: torch.Tensor,      # (3,) world frame
    object_quat: torch.Tensor,     # (4,) wxyz — object orientation (not used for symmetric objects)
    n: int = 512,
    standoff: float = 0.15,        # EE standoff distance from object center (m)
    arm_assign: Literal["auto", "left", "right", "both"] = "auto",
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> list[GraspCandidate]:
    """Generate N grasp candidates by sampling approach directions on the upper hemisphere.

    Each candidate places the EE at `object_pos + approach_dir * standoff`, with the
    wrist z-axis pointing back toward the object (−approach_dir). The roll around the
    approach axis is sampled uniformly.

    Arm assignment:
        "auto"  — left if approach_dir.y > 0, else right (splits hemisphere between arms)
        "left"  — all candidates assigned to left arm
        "right" — all candidates assigned to right arm
        "both"  — full hemisphere, arm auto-assigned, no upper-hemisphere filter

    Args:
        object_pos:   (3,) object position in world frame.
        object_quat:  (4,) object orientation (wxyz), currently unused (symmetric sampling).
        n:            Number of candidates to generate.
        standoff:     Distance from object center to EE (m).
        arm_assign:   Arm assignment strategy.
        device:       Compute device.
        generator:    Optional RNG for reproducibility.

    Returns:
        List of GraspCandidate objects.
    """
    dev = torch.device(device)
    object_pos = object_pos.to(dev)

    # Sample unit vectors on the upper hemisphere (z >= 0) — "approach from above"
    # using the rejection-sampling trick: sample sphere, keep upper half.
    collected_dirs = []
    while len(collected_dirs) < n:
        batch = torch.randn(n * 3, 3, device=dev, generator=generator)
        batch = batch / batch.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        upper = batch[batch[:, 2] >= 0]  # keep upper hemisphere
        collected_dirs.append(upper)

    approach_dirs = torch.cat(collected_dirs, dim=0)[:n]   # (n, 3) unit vectors

    # EE positions: stand off from object along the approach direction
    ee_positions = object_pos.unsqueeze(0) + approach_dirs * standoff   # (n, 3)

    # EE orientations: wrist z-axis points toward the object (−approach_dir)
    # wrist x-axis: random rotation around approach_dir (sample roll)
    z_world = torch.tensor([0., 0., 1.], device=dev)
    rolls = torch.rand(n, device=dev, generator=generator) * 2 * 3.14159265

    candidates = []
    for i in range(n):
        app_dir = approach_dirs[i]              # (3,) unit vector from object to EE
        neg_app = -app_dir                      # wrist should point back toward object

        # Build wrist orientation:
        # 1. Base rotation: wrist_z → neg_app
        q_base = _quat_from_two_vectors(z_world, neg_app)

        # 2. Roll: rotate around neg_app axis by random angle
        half_roll = rolls[i] / 2.0
        q_roll = torch.stack([
            half_roll.cos(),
            neg_app[0] * half_roll.sin(),
            neg_app[1] * half_roll.sin(),
            neg_app[2] * half_roll.sin(),
        ])
        ee_quat = _quat_multiply(q_roll.unsqueeze(0), q_base.unsqueeze(0)).squeeze(0)
        ee_quat = ee_quat / ee_quat.norm().clamp(min=1e-8)

        # Arm assignment
        if arm_assign == "auto":
            arm: Literal["left", "right"] = "left" if app_dir[1].item() > 0 else "right"
        elif arm_assign == "both":
            arm = "left" if app_dir[1].item() > 0 else "right"
        else:
            arm = arm_assign

        candidates.append(GraspCandidate(
            ee_pos_world=ee_positions[i],
            ee_quat_world=ee_quat,
            approach_dir=app_dir,
            arm=arm,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Grasp quality metric
# ---------------------------------------------------------------------------

def grasp_quality(candidates: list[GraspCandidate]) -> torch.Tensor:
    """Compute grasp quality scores for a list of candidates.

    Current metric: approach alignment with world-up vector, as a proxy for
    grasps that come from above (stable, avoids occlusion). This is intentionally
    simple — the reachability score provides the main discriminative signal.

    Quality = 0.5 * (1 + cos(θ)) ∈ [0, 1]
    where θ is the angle between the approach direction and world up [0,0,1].

    Args:
        candidates: List of GraspCandidate objects.

    Returns:
        (N,) float tensor of quality scores in [0, 1]. Higher is better.
    """
    dirs = torch.stack([c.approach_dir for c in candidates], dim=0)   # (N, 3)
    z_world = dirs.new_tensor([0., 0., 1.])
    cos_theta = (dirs * z_world).sum(-1)    # dot product with up axis
    return 0.5 * (1.0 + cos_theta)


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: list[GraspCandidate],
    reach: ReachabilityModel,
    pelvis_pos: torch.Tensor,         # (3,) world frame
    pelvis_quat: torch.Tensor,        # (4,) wxyz world frame
    grasp_quality_weight: float = 1.0,
    reachability_weight: float = 2.0,
) -> torch.Tensor:
    """Score all candidates with the combined energy function and annotate in-place.

    E(c) = grasp_quality_weight * GQ(c) + reachability_weight * R(EE_in_pelvis_frame)

    Args:
        candidates:            List of GraspCandidate objects.
        reach:                 Pre-loaded ReachabilityModel.
        pelvis_pos:            (3,) current pelvis position in world frame.
        pelvis_quat:           (4,) current pelvis orientation wxyz in world frame.
        grasp_quality_weight:  Weight on the grasp quality term (default 1.0).
        reachability_weight:   Weight on the reachability term (default 2.0).

    Returns:
        (N,) combined score tensor (higher → more promising for IK).
        Scores are also written back into candidate.score and candidate.reachability.
    """
    N = len(candidates)
    ee_pos  = torch.stack([c.ee_pos_world  for c in candidates], dim=0)   # (N, 3)
    ee_quat = torch.stack([c.ee_quat_world for c in candidates], dim=0)   # (N, 4)

    # Transform to pelvis frame
    pos_pelvis, quat_pelvis = world_to_pelvis_frame(ee_pos, ee_quat, pelvis_pos, pelvis_quat)

    # Reachability scores
    r_scores = reach.score(pos_pelvis, quat_pelvis).cpu()   # (N,)

    # Grasp quality scores
    gq_scores = grasp_quality(candidates)   # (N,)

    combined = grasp_quality_weight * gq_scores + reachability_weight * r_scores

    # Annotate candidates in-place
    for i, c in enumerate(candidates):
        c.reachability = r_scores[i].item()
        c.score        = combined[i].item()

    return combined


# ---------------------------------------------------------------------------
# Full planning pipeline
# ---------------------------------------------------------------------------

def plan_grasp(
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    current_qpos: torch.Tensor,           # (nq,) current robot joint state
    reach: ReachabilityModel,
    ik,                                   # BatchedAnalyticalIK
    pelvis_pos: torch.Tensor,             # (3,) world frame
    pelvis_quat: torch.Tensor,            # (4,) wxyz world frame
    n_candidates: int = 512,
    top_k: int = 16,
    n_ik_iters: int = 20,
    pos_tol: float = 0.01,               # 1 cm convergence threshold
    ori_tol: float = 0.087,              # 5° convergence threshold (rad)
    standoff: float = 0.15,              # EE standoff from object (m)
    grasp_quality_weight: float = 1.0,
    reachability_weight: float = 2.0,
    verbose: bool = False,
) -> GraspResult | None:
    """Plan a grasp for the given object pose.

    Steps:
        1. Generate N candidates (hemisphere sampling).
        2. Score all with reachability model (fast: single GPU forward pass).
        3. Select top-K by combined score.
        4. Pre-compute FK at current_qpos to get hold positions for the inactive arm.
        5. Reset IK for all K envs to current_qpos (clean warm-start).
        6. Single batched IK solve for all K candidates simultaneously.
        7. Per-candidate FK check; return best converged solution by score.

    Two bugs that caused large errors in the original implementation:
    - Bug 1 (stationary arm zero target): the inactive arm was given world-origin [0,0,0]
      as its hold target, causing IK to drive it to the floor while solving for the active
      arm. Fix: query FK at current_qpos to get the true inactive-arm EE pose.
    - Bug 2 (no IK reset between candidates): q_ik carried the failed state from the
      previous candidate as the warm-start for the next, compounding divergence across K
      sequential solves. Fix: reset all K envs to current_qpos before a single batch solve.

    Args:
        object_pos:   (3,) object position in world frame.
        object_quat:  (4,) object orientation wxyz in world frame.
        current_qpos: (nq,) current robot joint state (for IK warm-start / FK sync).
        reach:        Pre-loaded ReachabilityModel.
        ik:           Pre-constructed BatchedAnalyticalIK; must have num_envs >= top_k.
        pelvis_pos:   (3,) pelvis position in world frame (from state estimator).
        pelvis_quat:  (4,) pelvis orientation wxyz in world frame.
        n_candidates: How many candidates to score before selecting top-K.
        top_k:        How many candidates to verify with full IK.
        n_ik_iters:   DLS iterations per IK call (higher → more accurate but slower).
        pos_tol:      IK position convergence tolerance (m).
        ori_tol:      Approach direction tolerance (rad): angle between achieved/target wrist z-axis.
                      Wrist roll is excluded — the IK intentionally leaves it unconstrained
                      (task weight 0.02), so measuring roll error causes false failures.
        standoff:     EE standoff distance from object center (m).
        grasp_quality_weight: Weight on GQ in the scoring energy.
        reachability_weight:  Weight on R in the scoring energy.
        verbose:      Print per-candidate IK results.

    Returns:
        GraspResult with the best converged solution, or None if no candidate converged.
    """
    import mujoco

    actual_k = min(top_k, n_candidates)
    assert ik.num_envs >= actual_k, (
        f"IK has num_envs={ik.num_envs} but top_k={actual_k}. "
        "Rebuild BatchedAnalyticalIK with num_envs >= top_k."
    )
    device = ik.device

    # Step 1: generate candidates
    candidates = generate_candidates(
        object_pos, object_quat, n=n_candidates, standoff=standoff,
        arm_assign="both", device=str(pelvis_pos.device),
    )

    # Step 2: score all candidates
    scores = score_candidates(
        candidates, reach, pelvis_pos, pelvis_quat,
        grasp_quality_weight=grasp_quality_weight,
        reachability_weight=reachability_weight,
    )

    # Step 3: select top-K (sorted descending by score)
    top_idx       = torch.argsort(scores, descending=True)[:actual_k]
    top_candidates = [candidates[i] for i in top_idx.tolist()]

    if verbose:
        print(f"[plan_grasp] Top-{actual_k} scores: " +
              ", ".join(f"{candidates[i].score:.3f}" for i in top_idx[:5].tolist()))

    # Step 4: query FK at current_qpos to get hold positions for the inactive arm.
    # Previously this was [0,0,0] world (world origin), which caused the IK to waste
    # iterations driving the inactive arm to the floor.
    mj_data_fk = mujoco.MjData(ik.mj_model)
    mj_data_fk.qpos[:] = current_qpos.cpu().numpy()
    mujoco.mj_kinematics(ik.mj_model, mj_data_fk)
    hold_l_pos  = torch.tensor(mj_data_fk.xpos[ik._ee_left_id],  dtype=torch.float32, device=device)
    hold_l_quat = torch.tensor(mj_data_fk.xquat[ik._ee_left_id], dtype=torch.float32, device=device)
    hold_r_pos  = torch.tensor(mj_data_fk.xpos[ik._ee_right_id], dtype=torch.float32, device=device)
    hold_r_quat = torch.tensor(mj_data_fk.xquat[ik._ee_right_id], dtype=torch.float32, device=device)

    # Step 5: build batch targets (K, 2, 3/4) — one row per candidate.
    # targets[:, 0] = left arm, targets[:, 1] = right arm (IK convention).
    # The grasping arm gets the candidate EE target; the other holds its current FK pose.
    pos_targets  = torch.zeros(actual_k, 2, 3, device=device)
    quat_targets = torch.zeros(actual_k, 2, 4, device=device)
    quat_targets[..., 0] = 1.0   # identity default
    for i, cand in enumerate(top_candidates):
        grasp_pos  = cand.ee_pos_world.to(device)
        grasp_quat = cand.ee_quat_world.to(device)
        if cand.arm == "left":
            pos_targets[i, 0],  quat_targets[i, 0] = grasp_pos,  grasp_quat
            pos_targets[i, 1],  quat_targets[i, 1] = hold_r_pos, hold_r_quat
        else:
            pos_targets[i, 0],  quat_targets[i, 0] = hold_l_pos, hold_l_quat
            pos_targets[i, 1],  quat_targets[i, 1] = grasp_pos,  grasp_quat

    # Step 6: reset all K IK envs to current_qpos, then single batched solve.
    # Previously: K sequential solves without reset, so each candidate warm-started
    # from the previous failed configuration.
    physics_qpos_k = current_qpos.to(device).unsqueeze(0).expand(actual_k, -1).contiguous()
    ik.reset(torch.arange(actual_k, device=device), physics_qpos_k)
    arm_joints_k = ik.solve(physics_qpos_k, pos_targets, quat_targets, num_iters=n_ik_iters)
    # arm_joints_k: (K, 2*dof) — absolute positions [left_arm..., right_arm...]

    # Step 7: per-candidate FK check; return first converged solution by score rank.
    arm_addr    = ik._arm_qpos_addr.cpu().numpy()
    mj_data_chk = mujoco.MjData(ik.mj_model)
    qpos_base   = current_qpos.cpu().numpy()

    for rank, (cand, arm_j) in enumerate(zip(top_candidates, arm_joints_k)):
        qpos_check = qpos_base.copy()
        qpos_check[arm_addr] = arm_j.cpu().numpy()
        mj_data_chk.qpos[:] = qpos_check
        mujoco.mj_kinematics(ik.mj_model, mj_data_chk)

        if cand.arm == "left":
            actual_pos  = torch.tensor(mj_data_chk.xpos[ik._ee_left_id],  dtype=torch.float32)
            actual_quat = torch.tensor(mj_data_chk.xquat[ik._ee_left_id], dtype=torch.float32)
        else:
            actual_pos  = torch.tensor(mj_data_chk.xpos[ik._ee_right_id], dtype=torch.float32)
            actual_quat = torch.tensor(mj_data_chk.xquat[ik._ee_right_id], dtype=torch.float32)

        pos_err = (actual_pos - cand.ee_pos_world.cpu()).norm().item()

        # Orientation error: angle between the achieved and target approach directions
        # (wrist z-axis in world frame), NOT full quaternion distance.
        # Full quaternion distance includes wrist roll, which the IK intentionally
        # leaves unconstrained (task weight 0.02). Measuring roll error would
        # produce false failures on otherwise good approach poses. The approach
        # direction (what matters for collision-free grasping) uses z-axis alignment.
        z_local = torch.tensor([0.0, 0.0, 1.0])
        approach_actual = _quat_apply(actual_quat.unsqueeze(0), z_local.unsqueeze(0)).squeeze(0)
        approach_target = _quat_apply(cand.ee_quat_world.cpu().unsqueeze(0), z_local.unsqueeze(0)).squeeze(0)
        cos_a   = (approach_actual * approach_target).sum().clamp(-1.0, 1.0)
        ori_err = torch.acos(cos_a).item()

        if verbose:
            print(f"  [rank {rank+1}/{actual_k}] arm={cand.arm} "
                  f"score={cand.score:.3f} R={cand.reachability:.3f} | "
                  f"pos_err={pos_err*100:.1f}cm approach_err={ori_err*180/3.14159:.1f}°")

        if pos_err <= pos_tol and ori_err <= ori_tol:
            return GraspResult(
                candidate=cand,
                arm_joints=arm_j.cpu(),
                pos_err=pos_err,
                ori_err=ori_err,
            )

    return None   # No candidate converged within tolerance


# ---------------------------------------------------------------------------
# Interactive grasp planning viewer
# ---------------------------------------------------------------------------

def run_grasp_viewer(
    ik,
    reach: ReachabilityModel,
    current_qpos: torch.Tensor,    # (nq,) initial arm pose
    pelvis_pos: torch.Tensor,      # (3,) world frame
    pelvis_quat: torch.Tensor,     # (4,) wxyz
    n_candidates: int = 512,
    top_k: int = 8,
    n_ik_iters: int = 50,
    anim_duration: float = 2.5,    # seconds for the arm sweep
) -> None:
    """Interactive MuJoCo viewer for grasp planning.

    Behaviour:
      1. Plans a grasp to a randomly sampled object position.
      2. Smoothly animates the arm from the current pose to the IK solution
         (smooth-step easing, zero velocity at both ends).
      3. Holds the final pose and waits.
      4. Press Enter → plan a new random target and animate from current → new.
         The arm stays at the previous target; the next animation starts from there.

    Object positions are sampled uniformly within the reachable workspace
    (x: 0.15–0.35 m, y: −0.20–0.30 m, z: −0.05–0.15 m in pelvis frame).
    Retries up to 10 times if plan_grasp fails (infeasible sample).

    User-scene marker:
      - Orange sphere at the object position (goal to grasp).
        The wrist reaching that sphere makes the motion self-explanatory
        without extra clutter.

    Controls:
      Enter  — plan and animate to a new random target.
      Esc    — quit.

    Args:
        ik:            BatchedAnalyticalIK (model, EE IDs, arm qpos addresses).
        reach:         Pre-loaded ReachabilityModel.
        current_qpos:  (nq,) initial robot joint state.
        pelvis_pos:    (3,) pelvis world position.
        pelvis_quat:   (4,) pelvis orientation wxyz.
        n_candidates:  Candidates scored per plan call.
        top_k:         Top-K candidates sent to IK.
        n_ik_iters:    DLS iterations per plan call.
        anim_duration: Seconds to sweep from start to target pose.
    """
    import time
    import numpy as np
    import mujoco
    import mujoco.viewer as viewer_module
    from types import SimpleNamespace

    model    = ik.mj_model
    arm_addr = ik._arm_qpos_addr.cpu().numpy()
    eye3     = np.eye(3, dtype=np.float32).flatten()

    model.opt.gravity[:] = [0.0, 0.0, 0.0]   # kinematic display — no physics sim

    # --- Random object sampler (pelvis-frame offsets within the reachable workspace) ---
    OBJ_X = (0.15, 0.35)
    OBJ_Y = (-0.20, 0.30)
    OBJ_Z = (-0.05, 0.15)

    def _sample_and_plan(start_qpos: torch.Tensor):
        """Randomly sample an object position and run plan_grasp until success."""
        for _ in range(10):
            dx = torch.empty(1).uniform_(*OBJ_X).item()
            dy = torch.empty(1).uniform_(*OBJ_Y).item()
            dz = torch.empty(1).uniform_(*OBJ_Z).item()
            obj_pos  = pelvis_pos + torch.tensor([dx, dy, dz])
            obj_quat = torch.tensor([1., 0., 0., 0.])
            result = plan_grasp(
                obj_pos, obj_quat, start_qpos, reach, ik,
                pelvis_pos, pelvis_quat,
                n_candidates=n_candidates, top_k=top_k, n_ik_iters=n_ik_iters,
                pos_tol=0.02, ori_tol=0.26, verbose=False,
            )
            if result is not None:
                return obj_pos, result
        return None, None

    # --- Initial plan ---
    print("Planning first grasp target...")
    obj_pos, result = _sample_and_plan(current_qpos)
    if result is None:
        print("Could not find a feasible grasp. Exiting.")
        return

    # --- Viewer state ---
    # start_q / goal_q: qpos arrays for the current animation segment.
    # After animation completes the arm stays at goal_q; the next start_q = goal_q.
    state = SimpleNamespace(
        phase      = "animating",   # "animating" | "holding"
        t_anim     = time.time(),
        start_q    = current_qpos.cpu().numpy().copy(),
        goal_q     = current_qpos.cpu().numpy().copy(),
        obj_np     = obj_pos.cpu().numpy().astype(np.float32),
        ee_id      = ik._ee_left_id if result.candidate.arm == "left" else ik._ee_right_id,
        result     = result,
        want_next  = False,         # set by Enter key; consumed in the main loop
    )
    state.goal_q[arm_addr] = result.arm_joints.numpy()

    def key_callback(key: int) -> None:
        # GLFW_KEY_ENTER = 257; also accept Return (13) for terminal compatibility
        if key in (257, 13) and state.phase == "holding":
            state.want_next = True

    data = mujoco.MjData(model)
    data.qpos[:] = state.start_q
    mujoco.mj_forward(model, data)

    with viewer_module.launch_passive(model, data, key_callback=key_callback) as v:
        _print_target(state.result, state.obj_np)

        while v.is_running():
            # ── Animating: sweep arm from start_q → goal_q ──────────────────
            if state.phase == "animating":
                elapsed = time.time() - state.t_anim
                u       = min(elapsed / anim_duration, 1.0)
                alpha   = u * u * (3.0 - 2.0 * u)   # smooth-step: zero velocity at endpoints
                data.qpos[:] = state.start_q
                data.qpos[arm_addr] = (
                    state.start_q[arm_addr] + alpha * (state.goal_q[arm_addr] - state.start_q[arm_addr])
                )
                mujoco.mj_forward(model, data)
                if u >= 1.0:
                    state.phase = "holding"
                    print("  Reached.  Press Enter for next target.")

            # ── Holding: arm stays at goal_q; wait for Enter ─────────────────
            else:
                data.qpos[:] = state.goal_q
                mujoco.mj_forward(model, data)

                if state.want_next:
                    state.want_next = False
                    start_qpos = torch.tensor(state.goal_q, dtype=torch.float32)
                    print("Planning next grasp target...")
                    v.sync()   # push one frame before blocking on IK
                    obj_pos, result = _sample_and_plan(start_qpos)
                    if result is None:
                        print("  Could not find a feasible plan. Press Enter to try again.")
                    else:
                        state.start_q = state.goal_q.copy()
                        state.goal_q  = state.start_q.copy()
                        state.goal_q[arm_addr] = result.arm_joints.numpy()
                        state.obj_np  = obj_pos.cpu().numpy().astype(np.float32)
                        state.ee_id   = (ik._ee_left_id if result.candidate.arm == "left"
                                         else ik._ee_right_id)
                        state.result  = result
                        state.t_anim  = time.time()
                        state.phase   = "animating"
                        _print_target(result, state.obj_np)

            # ── Draw object marker ───────────────────────────────────────────
            mujoco.mjv_initGeom(
                v.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.04, 0.0, 0.0], np.float32),
                state.obj_np, eye3,
                np.array([1.0, 0.6, 0.1, 0.9], np.float32),   # orange
            )
            v.user_scn.ngeom = 1
            v.sync()
            time.sleep(1.0 / 60.0)


def _print_target(result: GraspResult, obj_np) -> None:
    """Print a one-line summary of the current grasp target."""
    print(f"  arm={result.candidate.arm}  object={obj_np.tolist()}  "
          f"pos_err={result.pos_err*100:.1f}cm  approach_err={result.ori_err*180/3.14159:.1f}°")


# ---------------------------------------------------------------------------
# Standalone test (no RL environment needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import mujoco
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_humanoid_v21_robot_cfg
    from mjlab.entity import Entity
    from mj_envs.utils.batched_ik import BatchedAnalyticalIK

    TOP_K = 8   # IK batch size — solve all top-K candidates in parallel

    print("Loading model...")
    cfg   = get_humanoid_v21_robot_cfg()
    model = Entity(cfg).spec.compile()

    left_joints  = [f"left_{j}"  for j in ["shoulder_1_joint", "shoulder_2_joint", "shoulder_3_joint",
                                             "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
    right_joints = [f"right_{j}" for j in ["shoulder_1_joint", "shoulder_2_joint", "shoulder_3_joint",
                                             "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
    ik = BatchedAnalyticalIK(
        mj_model=model, wp_model=None, num_envs=TOP_K, device="cpu",
        ee_left_name="wrist_3_L", ee_right_name="wrist_3_R",
        left_joint_names=left_joints, right_joint_names=right_joints,
    )
    reach = ReachabilityModel.from_robot("humanoid_v21", "both", "cpu")

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    current_qpos = torch.tensor(data.qpos, dtype=torch.float32)

    pelvis_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pelvis_pos  = torch.tensor(data.xpos[pelvis_id],  dtype=torch.float32)
    pelvis_quat = torch.tensor(data.xquat[pelvis_id], dtype=torch.float32)

    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        run_grasp_viewer(ik, reach, current_qpos, pelvis_pos, pelvis_quat)
    else:
        print("No DISPLAY — running headless smoke test...")
        object_pos = pelvis_pos + torch.tensor([0.30, 0.20, 0.10])
        result = plan_grasp(
            object_pos, torch.tensor([1., 0., 0., 0.]), current_qpos, reach, ik,
            pelvis_pos, pelvis_quat, n_candidates=512, top_k=TOP_K, n_ik_iters=50,
            pos_tol=0.02, ori_tol=0.26, verbose=True,
        )
        if result:
            print(f"\nOK  arm={result.candidate.arm} "
                  f"pos_err={result.pos_err*100:.1f}cm "
                  f"approach_err={result.ori_err*180/3.14159:.1f}°")
        else:
            print("\nNo candidate converged.")
