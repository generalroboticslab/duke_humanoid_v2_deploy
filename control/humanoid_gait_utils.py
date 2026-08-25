"""`GaitPhaseClock` — the periodic gait-phase signal in the policy observation,
kept in pure Python scalars with a single torch copy per step.
"""
import math
import random
import torch
import numpy as np


class GaitPhaseClock:
    """Manages the periodic gait phase for humanoid locomotion.

    All state is pure Python scalars — eliminates CUDA kernel launch and Python→C++
    torch dispatch overhead (~50 µs/op × 25 ops = ~1250 µs/step with CPU tensors).

    Two call paths:
      step()    — standalone use: _advance() + one copy_() CUDA op.
      _advance() — folded path: caller writes _gait_out_np into the sensor bundle
                   numpy array before the single sensor copy_(), saving one kernel.

    Behavioral equivalence to the original torch implementation:
    - Phase advance and snap logic are identical. Python float (float64) intermediate
      gives higher precision; output is truncated to float32 at the final copy_().
      Difference from original float32 path: sub-ULP, irrelevant to the policy.
    """

    def __init__(self,
                 period_range: tuple[float, float],
                 linear_velocity_threshold: float,
                 yaw_velocity_threshold: float,
                 step_dt: float,
                 device: torch.device):
        self._period_min = float(period_range[0])
        self._period_max = float(period_range[1])
        self._lin_thresh = float(linear_velocity_threshold)
        self._yaw_thresh = float(yaw_velocity_threshold)
        self._step_dt    = float(step_dt)
        self._device     = device

        self._phase     = 0.0
        self._period    = self._period_min
        self._last_step = -1
        self._snap_win  = 2.0 * step_dt / self._period_min

        self._gait_out_np    = np.zeros(4, dtype=np.float32)
        self._gait_phase_out = torch.zeros(4, device=device)

    def reset(self, deterministic: bool = True):
        """Reset phase to zero and sample a new period."""
        self._phase = 0.0
        if deterministic:
            self._period = (self._period_min + self._period_max) / 2.0
        else:
            self._period = random.uniform(self._period_min, self._period_max)
        self._last_step = -1

    def _advance(self, command: torch.Tensor, episode_step) -> bool:
        """Advance phase state and fill _gait_out_np. No H2D copy.

        Separated from step() so the sensor bundle path (obs_h2d_copy) can fold
        gait into the single sensor copy_() rather than issuing a separate kernel.
        Returns True if phase updated this call, False if same episode_step (idempotent).
        """
        ep = int(episode_step)
        if ep == self._last_step:
            return False
        self._last_step = ep

        vx = float(command[0]); vy = float(command[1]); wz = float(command[2])
        is_moving = (vx * vx + vy * vy > self._lin_thresh * self._lin_thresh) or \
                    (abs(wz) > self._yaw_thresh)

        if is_moving or self._phase > 1e-6:
            new_phase = (self._phase + self._step_dt / self._period) % 1.0
        else:
            new_phase = self._phase

        # Snap to 0 at stride boundary when decelerating to avoid phase drift on stop.
        if (not is_moving) and (new_phase < self._snap_win) and \
                               (self._phase > 1.0 - self._snap_win):
            new_phase = 0.0

        self._phase = new_phase
        phase_R = (new_phase + 0.5) % 1.0
        _TWO_PI = 6.283185307179586
        self._gait_out_np[0] = math.sin(_TWO_PI * new_phase)
        self._gait_out_np[1] = math.cos(_TWO_PI * new_phase)
        self._gait_out_np[2] = math.sin(_TWO_PI * phase_R)
        self._gait_out_np[3] = math.cos(_TWO_PI * phase_R)
        return True

    def step(self, command: torch.Tensor, episode_step) -> torch.Tensor:
        """Advance gait phase clock and return [sin_L, cos_L, sin_R, cos_R].

        Args:
            command: CPU tensor (3,) — [vx, vy, wz].
            episode_step: CPU tensor (1,) long or Python int — current episode step.

        Returns:
            CUDA float32 tensor (4,): [sin(2πφ_L), cos(2πφ_L), sin(2πφ_R), cos(2πφ_R)].
        """
        self._advance(command, episode_step)
        self._gait_phase_out.copy_(torch.from_numpy(self._gait_out_np))
        return self._gait_phase_out
