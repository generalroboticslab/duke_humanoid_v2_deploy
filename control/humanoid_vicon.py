"""Vicon motion-capture interface for HumanoidRealEnv.

Self-contained wrapper around ObjectTrackerNonblocking. Instantiate only when
Vicon is needed; leave HumanoidRealEnv.vicon = None otherwise.

To activate in HumanoidRealEnv.__init__():
    from humanoid_vicon import ViconInterface
    self.vicon = ViconInterface(
        vicon_ip="<vicon-host>",
        objects={
            "vicon":             "grl_hb",   # base link
            "vicon_left_shank":  "grl_hs0",
            "vicon_right_shank": "grl_hs1",
        },
        calibration_path="<legged_env_v2>/mj_envs/deploy/vicon_calibration.yaml",
    )
    self.vicon.verify()
"""
import logging
import os

import numpy as np
import torch
import yaml

from pyvicon.vicon_utils import ObjectTrackerNonblocking
from humanoid_utils import invert_transform_np, make_transform_np, transform_to_pos_quat_np

_log = logging.getLogger(__name__)


class ViconInterface:
    """Per-step Vicon fetch + world-to-base transform.

    objects: recording-prefix → Vicon object name.
    last_results: populated by fetch_step(), read by HumanoidRealEnv._record_step().
    """

    def __init__(self, vicon_ip: str, objects: dict[str, str], calibration_path: str):
        self.objects = objects
        self.last_results: dict = {}
        self._calibrated = False

        self._tracker = ObjectTrackerNonblocking(vicon_ip, list(objects.values()))

        if os.path.exists(calibration_path):
            with open(calibration_path, "r") as f:
                calib = yaml.safe_load(f)
            T_bm = np.array(calib["T_base_marker"])
            T_vf = np.array(calib["T_vicon_floor"])
            self._T_vf_inv = invert_transform_np(T_vf)  # vicon frame → floor
            self._T_bm_inv = invert_transform_np(T_bm)  # vicon marker → base
            self._calibrated = True
        else:
            _log.error(f"[VICON] Calibration file not found: {calibration_path}")

    def verify(self) -> None:
        """Log availability of all tracked objects. Call once at startup."""
        for obj_name in self.objects.values():
            result = self._tracker.get_position(obj_name)
            if result is None:
                _log.warning(f"[VICON] Object '{obj_name}' not available")
            else:
                positions, frame_num, _, _ = result
                _log.info(f"[VICON] {obj_name} frame {frame_num}: pos={positions[0, :3]} quat={positions[0, 3:7]}")

    def fetch_step(self) -> dict:
        """Get latest positions for all objects. Stores result in last_results."""
        self.last_results = {
            prefix: self._tracker.get_position(obj_name)
            for prefix, obj_name in self.objects.items()
        }
        return self.last_results

    def compute_world_base(self, base_result) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Convert raw Vicon base-link result to world-frame (pos, quat_wxyz).

        Returns (None, None) if not calibrated or result is None.
        T_wb = T_vf_inv @ T_vm @ T_bm_inv: vicon marker pose → world-to-base.
        """
        if not self._calibrated or base_result is None:
            return None, None
        positions, _, _, _ = base_result
        T_vm = make_transform_np(positions[0, :3], positions[0, 3:7])
        T_wb = self._T_vf_inv @ T_vm @ self._T_bm_inv
        return transform_to_pos_quat_np(T_wb)

    def projected_gravity(self, wb_quat_wxyz: np.ndarray, device: torch.device) -> torch.Tensor:
        """Rotate world gravity [0,0,-1] into body frame via quaternion inverse."""
        w, x, y, z = wb_quat_wxyz
        return torch.tensor([
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            -(w * w - x * x - y * y + z * z),
        ], dtype=torch.float32, device=device)
