"""
Runtime reachability scoring for end-effector poses.

Provides:
  - ReachabilityModel: wraps a TorchScript MLP that returns p(feasible | EE_pose_in_pelvis_frame)
  - world_to_base_link_frame: transforms EE poses from world frame to base-link frame
  - world_to_pelvis_frame: legacy name for the same rigid-frame transform
  - encode_ee_pose: encodes (pos, quat) → 9-dim input feature [pos, rot_6d]

All tensors use wxyz quaternion convention throughout.

Typical usage:
    reach = ReachabilityModel("asset_zoo/cache/reachability_model_humanoid_v21_both.pt", "cuda:0")
    scores = reach.score(ee_pos_pelvis, ee_quat_pelvis)  # (N,) in [0, 1]
"""

from pathlib import Path
import torch


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Convert wxyz unit quaternions to 6D rotation representation.

    The 6D representation is the concatenation of the first two columns of the
    rotation matrix, which spans SO(3) continuously and is free of singularities.
    See: Zhou et al. (2019), "On the Continuity of Rotation Representations
    in Neural Networks", CVPR 2019.

    Args:
        q: (..., 4) unit quaternions in wxyz order.

    Returns:
        (..., 6) — columns 0 and 1 of the corresponding rotation matrix.
    """
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),  2*(x*z + y*w),
        2*(x*y + z*w),  1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),  2*(y*z + x*w),  1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of wxyz quaternion. (..., 4) -> (..., 4)."""
    return q * q.new_tensor([1., -1., -1., -1.])


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


def encode_ee_pose(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Encode EE pose as 9-dim feature vector.

    Args:
        pos:  (..., 3) EE position in pelvis frame (meters)
        quat: (..., 4) EE orientation wxyz in pelvis frame

    Returns:
        (..., 9) — [pos_xyz (3), rot_6d (6)]
    """
    return torch.cat([pos, quat_to_rot6d(quat)], dim=-1)


# ---------------------------------------------------------------------------
# Frame transform
# ---------------------------------------------------------------------------

def world_to_base_link_frame(
    pos_world: torch.Tensor,    # (N, 3) or (3,)
    quat_world: torch.Tensor,   # (N, 4) or (4,) wxyz
    base_link_pos: torch.Tensor,   # (3,) base-link position in world frame
    base_link_quat: torch.Tensor,  # (4,) base-link orientation wxyz in world frame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform EE poses from world frame to current ``base_link`` frame.

    Used at runtime to express grasp candidates relative to the same rigid
    root frame used when workspace fields and camera geometry were built.

    Args:
        pos_world:   (N, 3) EE positions in world frame
        quat_world:  (N, 4) EE orientations wxyz in world frame
        base_link_pos:  (3,) ``base_link`` origin in world frame.
        base_link_quat: (4,) ``base_link`` orientation wxyz in world frame.

    Returns:
        (pos_base_link, quat_base_link) — each (N, 3) and (N, 4) in ``base_link`` frame.
    """
    inv_base_link = _quat_conjugate(base_link_quat)        # (4,)
    pos_rel = pos_world - base_link_pos                     # (N, 3)
    pos_base_link = _quat_apply(inv_base_link.unsqueeze(0), pos_rel)
    quat_base_link = _quat_multiply(
        inv_base_link.unsqueeze(0).expand_as(quat_world), quat_world
    )
    return pos_base_link, quat_base_link


def world_to_pelvis_frame(
    pos_world: torch.Tensor,
    quat_world: torch.Tensor,
    pelvis_pos: torch.Tensor,
    pelvis_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy compatibility alias for :func:`world_to_base_link_frame`.

    The transform has always been generic; callers whose root is named
    ``pelvis`` retain their existing API while new workspace-field code names
    the frame explicitly.
    """
    return world_to_base_link_frame(pos_world, quat_world, pelvis_pos, pelvis_quat)


# ---------------------------------------------------------------------------
# ReachabilityModel
# ---------------------------------------------------------------------------

class ReachabilityModel:
    """Wraps a TorchScript reachability MLP for fast batch scoring.

    The model returns p(feasible | EE_pose_in_pelvis_frame) for each candidate
    EE pose. Scores close to 1.0 indicate the arm can likely reach that pose
    without self-collision; scores near 0.0 indicate infeasibility.

    Args:
        model_path: Path to the TorchScript .pt model file.
        device:     Device string (e.g., "cuda:0", "cpu").
    """

    def __init__(self, model_path: str | Path, device: str = "cuda:0"):
        self.device = torch.device(device)
        self._model = torch.jit.load(str(model_path), map_location=self.device)
        self._model.eval()

    def score(
        self,
        ee_pos_pelvis: torch.Tensor,    # (N, 3)
        ee_quat_pelvis: torch.Tensor,   # (N, 4) wxyz
    ) -> torch.Tensor:                  # (N,) in [0, 1]
        """Score a batch of EE poses expressed in pelvis frame.

        Args:
            ee_pos_pelvis:  (N, 3) EE positions in pelvis frame.
            ee_quat_pelvis: (N, 4) EE orientations wxyz in pelvis frame.

        Returns:
            (N,) float tensor of feasibility scores in [0, 1].
        """
        feat = encode_ee_pose(
            ee_pos_pelvis.to(self.device),
            ee_quat_pelvis.to(self.device),
        )
        with torch.no_grad():
            return self._model(feat)

    def score_world_frame(
        self,
        ee_pos_world: torch.Tensor,    # (N, 3)
        ee_quat_world: torch.Tensor,   # (N, 4) wxyz
        pelvis_pos: torch.Tensor,      # (3,)
        pelvis_quat: torch.Tensor,     # (4,) wxyz
    ) -> torch.Tensor:                 # (N,) in [0, 1]
        """Score EE poses given in world frame, transforming internally to pelvis frame.

        Convenience wrapper around :meth:`score` + :func:`world_to_pelvis_frame`.

        Args:
            ee_pos_world:  (N, 3) EE positions in world frame.
            ee_quat_world: (N, 4) EE orientations wxyz in world frame.
            pelvis_pos:    (3,) pelvis origin in world frame.
            pelvis_quat:   (4,) pelvis orientation wxyz in world frame.

        Returns:
            (N,) feasibility scores in [0, 1].
        """
        pos_p, quat_p = world_to_pelvis_frame(
            ee_pos_world.to(self.device),
            ee_quat_world.to(self.device),
            pelvis_pos.to(self.device),
            pelvis_quat.to(self.device),
        )
        return self.score(pos_p, quat_p)

    @classmethod
    def from_robot(
        cls,
        robot: str = "humanoid_v21",
        arm: str = "both",
        device: str = "cuda:0",
        cache_dir: str | Path | None = None,
    ) -> "ReachabilityModel":
        """Load a model from the standard cache directory by robot and arm name.

        Args:
            robot:     Robot name matching the cache file prefix.
            arm:       Arm configuration: "left", "right", or "both".
            device:    Compute device.
            cache_dir: Override cache directory. Defaults to asset_zoo/cache/.
        """
        if arm == "both":
            raise ValueError(
                "arm='both' is not supported: it silently concatenates left and right EE positions "
                "as independent positive samples, producing an incorrect joint model. "
                "Train separate models with arm='left' and arm='right' instead."
            )
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "asset_zoo" / "cache"
        model_path = Path(cache_dir) / f"reachability_model_{robot}_{arm}.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reachability model not found: {model_path}\n"
                f"Run: python mj_envs/asset_zoo/build_reachability_map.py --robot {robot} --arm {arm}"
            )
        return cls(model_path, device)
