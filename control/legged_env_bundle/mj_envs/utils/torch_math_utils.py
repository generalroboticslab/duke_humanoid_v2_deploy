"""Quaternion math, SO(3) Jacobians, and smooth collision-distance utilities (PyTorch, wxyz).

All quaternion functions use MuJoCo's wxyz convention: q = [w, x, y, z].
All inputs/outputs are torch.Tensors and support arbitrary batch dimensions (...).

Functions:
    Quaternion:
        _quat_conjugate       — conjugate / inverse of a unit quaternion
        _quat_apply           — rotate a vector by a quaternion
        _quat_multiply        — Hamilton product of two quaternions
        _axis_angle_to_quat   — axis-angle vector → quaternion
        _quat_to_axis_angle   — quaternion → axis-angle vector (shortest path)
        _quat_to_rot_matrix   — quaternion → 3×3 rotation matrix (optional out buffer)
        _rot_matrix_to_quat   — 3×3 rotation matrix → quaternion (Shepperd's method)

    SO(3) Jacobian:
        _jlog                 — left Jacobian inverse of SO(3) (jlog / Jacobian of log map)
                                corrects the orientation Jacobian for manifold curvature

    Collision:
        _colldist_from_sdf    — signed distance → smooth C¹ repulsion cost (pyroki activation)
        _capsule_seg_seg_dist — pairwise capsule centerline distance (N,A,T) via Shene's algorithm,
                                returns (dist, delta, C1) for use in IK collision repulsion
"""

import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Quaternion math helpers (wxyz convention throughout)
# ---------------------------------------------------------------------------

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a unit quaternion, which equals its inverse.

    Negates the vector part (xyz) while keeping the scalar (w) unchanged:
        conj([w, x, y, z]) = [w, -x, -y, -z]
    For unit quaternions (‖q‖ = 1) the conjugate is the rotation inverse.

    Args:
        q: (..., 4) quaternion in wxyz order.

    Returns:
        (..., 4) conjugate quaternion.
    """
    return q * q.new_tensor([1., -1., -1., -1.])

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by unit quaternion q using the Rodrigues sandwich formula.

    Computes q ⊗ [0, v] ⊗ q* efficiently without forming the full 4×4 product:
        t  = 2 * (xyz × v)
        v' = v + w*t + (xyz × t)

    Args:
        q: (..., 4) unit quaternion in wxyz order.
        v: (..., 3) vector to rotate.

    Returns:
        (..., 3) rotated vector.
    """
    w, xyz = q[..., 0:1], q[..., 1:]
    # torch.cross requires equal ndim; broadcast v up to xyz rank when needed
    while v.ndim < xyz.ndim:
        v = v.unsqueeze(0)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)

def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 for unit quaternions in wxyz order.

    Composes two rotations: applying q2 first, then q1.
    Both inputs must use wxyz convention (MuJoCo standard).

    Args:
        q1: (..., 4) left quaternion.
        q2: (..., 4) right quaternion.

    Returns:
        (..., 4) product quaternion.
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def _axis_angle_to_quat(v: torch.Tensor) -> torch.Tensor:
    """Axis-angle to quaternion. v: (..., 3), ||v|| = angle. Returns (..., 4).
    Identity quaternion [1,0,0,0] for near-zero angles (< 1e-8 rad).
    """
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # clamped to avoid division by zero
    half  = theta / 2.0
    wxyz  = torch.cat([torch.cos(half), torch.sin(half) * (v / theta)], dim=-1)
    # Use the original unclamped norm to detect truly zero rotations. After clamping,
    # theta is always >= 1e-8, so we can no longer use it to identify identity rotations.
    raw_theta = v.norm(dim=-1, keepdim=True)
    identity  = wxyz.new_zeros(wxyz.shape)
    identity[..., 0] = 1.0
    return torch.where(raw_theta < 1e-8, identity, wxyz)

def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Quaternion to axis-angle. q: (..., 4). Returns (..., 3).
    Ensures shortest path (w ≥ 0) and is numerically stable near identity.
    """
    q = torch.where(q[..., :1] < 0, -q, q)          # shortest path
    xyz       = q[..., 1:]
    half_sin  = xyz.norm(dim=-1, keepdim=True)
    half_ang  = torch.atan2(half_sin, q[..., :1])
    scale     = torch.where(
        half_sin > 1e-8,
        2.0 * half_ang / half_sin.clamp(min=1e-8),
        2.0 * torch.ones_like(half_sin),
    )
    return xyz * scale

def _quat_to_rot_matrix(q: torch.Tensor,
                        out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert a unit quaternion to its equivalent 3×3 rotation matrix.

    Used in _dls_step to express the orientation Jacobian and error in the
    EE body frame, so task_weights (e.g. relaxed wrist-roll) act on local
    axes rather than world axes.

    Args:
        q:   (..., 4) unit quaternion in wxyz order.
        out: Optional pre-allocated (..., 3, 3) buffer to write into,
             avoiding a heap allocation when called inside the DLS hot loop.

    Returns:
        (..., 3, 3) rotation matrix R such that R @ v_local = v_world.
        If out is provided, returns out (same object, filled in-place).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    res = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device) \
          if out is None else out
    res[..., 0, 0] = 1.0 - 2.0 * (y2 + z2)
    res[..., 0, 1] = 2.0 * (xy - wz)
    res[..., 0, 2] = 2.0 * (xz + wy)

    res[..., 1, 0] = 2.0 * (xy + wz)
    res[..., 1, 1] = 1.0 - 2.0 * (x2 + z2)
    res[..., 1, 2] = 2.0 * (yz - wx)

    res[..., 2, 0] = 2.0 * (xz - wy)
    res[..., 2, 1] = 2.0 * (yz + wx)
    res[..., 2, 2] = 1.0 - 2.0 * (x2 + y2)

    return res

def _rot_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix → unit quaternion (wxyz). (..., 3, 3) → (..., 4).

    Shepperd's branch-free method: all four candidates computed unconditionally,
    selected via float masks. Numerically stable across all orientations.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    safe  = (trace > 0).float()
    sA = (trace + 1.0).clamp(min=1e-10).sqrt() * 2.0
    wA, xA, yA, zA = 0.25*sA, (m21-m12)/sA, (m02-m20)/sA, (m10-m01)/sA
    s00 = (1.0 + m00 - m11 - m22).clamp(min=1e-10).sqrt() * 2.0
    w00, x00, y00, z00 = (m21-m12)/s00, 0.25*s00, (m01+m10)/s00, (m02+m20)/s00
    s11 = (1.0 + m11 - m00 - m22).clamp(min=1e-10).sqrt() * 2.0
    w11, x11, y11, z11 = (m02-m20)/s11, (m01+m10)/s11, 0.25*s11, (m12+m21)/s11
    s22 = (1.0 + m22 - m00 - m11).clamp(min=1e-10).sqrt() * 2.0
    w22, x22, y22, z22 = (m10-m01)/s22, (m02+m20)/s22, (m12+m21)/s22, 0.25*s22
    u00 = ((m00>m11) & (m00>m22) & (trace<=0)).float()
    u11 = ((m11>=m00) & (m11>m22) & (trace<=0)).float()
    u22 = ((m22>=m00) & (m22>=m11) & (trace<=0)).float()
    w = safe*wA + u00*w00 + u11*w11 + u22*w22
    x = safe*xA + u00*x00 + u11*x11 + u22*x22
    y = safe*yA + u00*y00 + u11*y11 + u22*y22
    z = safe*zA + u00*z00 + u11*z11 + u22*z22
    return torch.stack([w, x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# SO(3) log-map Jacobian  (pyroki analytic Jacobian correction)
# ---------------------------------------------------------------------------

def _jlog(aa: torch.Tensor) -> torch.Tensor:
    """Left Jacobian inverse of SO(3) — the Jacobian of the SO(3) log map.

    Corrects an angular-velocity Jacobian J_ori (3×K) for manifold curvature when the
    orientation error is large. Applying this makes the linearisation exact at the current
    error, not just at zero error:

        J_ori_corrected = jlog(err_aa) @ J_ori

    For small angles (θ < ~0.1 rad), jlog → I — no correction needed (and the formula is
    numerically stable via the clamped fallback). For θ > ~20°, the correction meaningfully
    prevents over-rotation steps in the DLS.

    Derivation: J_L^{-1} from Sola "micro Lie theory" (2018), SO(3) section.
    For K = [aa×] = θ·[n̂×] (skew-symmetric matrix of the full axis-angle vector aa):

        jlog(aa) = I  −  (1/2)·K  +  (1 − (θ/2)·cot(θ/2)) / θ²  ·  K²

    The (1/2)·K term is O(||aa||) and vanishes for small errors, so jlog(0) = I correctly.
    [n̂×] is the skew-symmetric matrix of the unit axis n̂; [aa×] = θ·[n̂×].

    Reference: pyroki _pose_residual_analytic_jac.py; Chirikjian "Stochastic Models"
               Vol.2 ch.10; also called the "BCH first-order correction".

    Args:
        aa: (..., 3) axis-angle vector (rotation axis × angle in radians).

    Returns:
        (..., 3, 3) left Jacobian inverse J_log. For zero rotation returns identity.
    """
    theta2 = aa.norm(dim=-1, keepdim=True) ** 2  # (..., 1)
    theta  = theta2.sqrt()                         # (..., 1)

    # Skew-symmetric cross-product matrix K = [aa×] = theta·[n̂×].
    # Using aa directly avoids normalisation — the coefficients below absorb the theta factors.
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1] = -aa[..., 2];  K[..., 0, 2] =  aa[..., 1]
    K[..., 1, 0] =  aa[..., 2];  K[..., 1, 2] = -aa[..., 0]
    K[..., 2, 0] = -aa[..., 1];  K[..., 2, 1] =  aa[..., 0]

    # Coefficient for the K² term: c_nn / θ²  where  c_nn = 1 − (θ/2)·cot(θ/2)
    # Taylor limit at θ→0: c_nn/θ² → 1/12 (avoids 0/0).
    # torch.where evaluates BOTH branches eagerly — guard tan() from θ=0 with a clamp:
    safe_half  = (theta / 2).clamp(min=1e-6)           # safe input for tan(); (..., 1)
    coeff_safe = (1.0 - safe_half / safe_half.tan()) / theta2.clamp(min=1e-8)
    coeff = torch.where(theta > 1e-2, coeff_safe,
                        torch.full_like(theta, 1.0 / 12.0))  # (..., 1)

    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*aa.shape[:-1], 3, 3)

    # Correct formula (Sola "micro Lie theory", right-to-left notation):
    #   jlog(aa) = I  −  (1/2)·K  +  c_nn/θ² · K²
    # where K = [aa×] = θ·[n̂×], so (1/2)·K = (θ/2)·[n̂×] scales as θ and → 0 for small errors.
    # IMPORTANT: do NOT divide K by theta here. The formula uses K directly (not K/theta),
    # so for small aa the (1/2)·K term is O(||aa||) → 0, and jlog → I correctly.
    # (A previous version had 0.5/theta*K which gives [n̂×]/2, θ-independent — numerically wrong.)
    return I - 0.5 * K + coeff.unsqueeze(-1) * (K @ K)


# ---------------------------------------------------------------------------
# Smooth collision distance activation  (pyroki / arxiv:2310.17274 §3.2)
# ---------------------------------------------------------------------------

def _colldist_from_sdf(dist: torch.Tensor, activation_dist: float) -> torch.Tensor:
    """Convert a raw surface-to-surface signed distance into a smooth repulsion magnitude.

    Implements the piecewise smooth activation from pyroki (arxiv:2310.17274, §3.2 /
    Fig. 4).  It replaces the simple linear ramp `(d_act - gap).clamp(min=0)` with a
    C¹-continuous function that has a zero gradient exactly at the activation boundary,
    avoiding the abrupt "kick" the linear ramp produces when a pair first enters range.
    A smooth gradient helps the DLS converge more cleanly and prevents oscillation near
    the activation boundary.

    Piecewise definition (all values ≤ 0; negate to get a repulsion magnitude ≥ 0):

        dist ≥ d_act  →  0                              (out of range, no cost)
        0 ≤ dist < d_act  →  −0.5/d_act · (dist − d_act)²   (quadratic near boundary)
        dist < 0      →  dist − 0.5·d_act               (linear for penetration)

    Continuity at dist = 0:
        quadratic end:  −0.5/d_act · (0 − d_act)² = −0.5·d_act
        linear start:   0 − 0.5·d_act             = −0.5·d_act  ✓

    Gradient continuity at dist = 0:
        d/d(dist) quadratic = (dist − d_act)/d_act → at 0: −1
        d/d(dist) linear    = 1                     → at 0: −1  ✓

    To get repulsion magnitude (≥ 0): `repulsion = -_colldist_from_sdf(dist, d_act)`.

    Args:
        dist:            (...) signed surface-to-surface distance.
                         Positive = separated, 0 = touching, negative = penetrating.
        activation_dist: Repulsion turns on below this distance (meters).

    Returns:
        (...) smooth cost values ≤ 0.  Zero when dist ≥ activation_dist.
    """
    # Cap at activation_dist so envs beyond range contribute exactly zero cost.
    dist_capped = dist.clamp(max=activation_dist)

    # Quadratic branch: smooth ramp from 0 at the activation boundary to −0.5·d_act at contact.
    quadratic = -0.5 / (activation_dist + 1e-6) * (dist_capped - activation_dist) ** 2

    # Linear branch: extends past contact with slope 1 so the repulsion grows unboundedly
    # for penetrating configurations, discouraging deep collisions.
    linear = dist_capped - 0.5 * activation_dist

    # Select branch element-wise: linear for penetration (dist < 0), quadratic otherwise.
    return torch.where(dist_capped < 0, linear, quadratic).clamp(max=0.0)


# ---------------------------------------------------------------------------
# Capsule-capsule distance (Shene's algorithm, broadcast over (N, A, T))
# ---------------------------------------------------------------------------

_NEAR_ZERO    = 1e-8
_PARALLEL_EPS = 1e-5   # sin²θ threshold for parallel detection


def _seg_pt_closest(p: torch.Tensor, q: torch.Tensor, x: torch.Tensor):
    """Closest point on segment [p, q] to point x, and distance.

    Args:
        p, q, x: (..., 3)
    Returns:
        dist:    (...,) Euclidean distance
        closest: (..., 3) closest point on segment
    """
    pq     = q - p
    len_sq = (pq * pq).sum(-1).clamp(min=_NEAR_ZERO)
    t      = ((x - p) * pq).sum(-1) / len_sq
    closest = p + t.clamp(0.0, 1.0).unsqueeze(-1) * pq
    diff   = x - closest
    dist   = (diff * diff).sum(-1).sqrt()
    return dist, closest


def _capsule_seg_seg_dist(
    p1: torch.Tensor, p2: torch.Tensor,
    p3: torch.Tensor, p4: torch.Tensor,
) -> tuple:
    """Pairwise capsule centerline distance between arm and torso capsule sets.

    Implements Shene's closest-point algorithm with one refinement round for the
    non-parallel case, and a min-of-4-projections fallback for parallel segments.
    The parallel fallback correctly handles overlapping side-by-side capsules where
    naive endpoint-to-endpoint distance overestimates the true perpendicular gap.

    Args:
        p1, p2: arm capsule segment endpoints, shape (N, A, 3).
                Expanded to (N, A, 1, 3) internally for broadcasting.
        p3, p4: torso capsule segment endpoints, shape (N, T, 3).
                Expanded to (N, 1, T, 3) internally for broadcasting.

    Returns:
        dist:  (N, A, T) unsigned centerline distance between segment pairs.
        delta: (N, A, T, 3) C1 - C2 (arm closest point minus torso closest point).
               Points away from torso — same sign convention as sphere-sphere delta.
        C1:    (N, A, T, 3) arm closest point in world frame.
               Used as the moment-arm origin for torque: cross(C1 - body_com, force).
    """
    # Broadcast: arm (N, A, 1, 3), torso (N, 1, T, 3) → all ops produce (N, A, T, ...)
    p1 = p1.unsqueeze(2)   # (N, A, 1, 3)
    p2 = p2.unsqueeze(2)
    p3 = p3.unsqueeze(1)   # (N, 1, T, 3)
    p4 = p4.unsqueeze(1)

    d1 = p2 - p1   # (N, A, 1, 3) — arm segment direction
    d2 = p4 - p3   # (N, 1, T, 3) — torso segment direction
    r  = p1 - p3   # (N, A, T, 3)

    a = (d1 * d1).sum(-1)   # (N, A, 1)
    e = (d2 * d2).sum(-1)   # (N, 1, T)
    f = (d2 * r ).sum(-1)   # (N, A, T)
    b = (d1 * d2).sum(-1)   # (N, A, T)
    c = (d1 * r ).sum(-1)   # (N, A, T)

    denom    = a * e - b * b                          # (N, A, T)
    ae       = (a * e).clamp(min=_NEAR_ZERO)
    parallel = denom < _PARALLEL_EPS * ae             # (N, A, T) bool

    # ── Non-parallel: Shene algorithm with one refinement round ──────────────
    s  = ((b * f - c * e) / denom.clamp(min=_NEAR_ZERO)).clamp(0.0, 1.0)
    t  = ((b * s + f)     / e.clamp(min=_NEAR_ZERO)).clamp(0.0, 1.0)
    s  = ((b * t - c)     / a.clamp(min=_NEAR_ZERO)).clamp(0.0, 1.0)   # refine

    C1_np    = p1 + s.unsqueeze(-1) * d1   # (N, A, T, 3)
    C2_np    = p3 + t.unsqueeze(-1) * d2   # (N, A, T, 3)
    delta_np = C1_np - C2_np               # (N, A, T, 3)
    dist_np  = (delta_np * delta_np).sum(-1).sqrt()

    # ── Parallel fallback: min of 4 point-to-segment distances ───────────────
    # Squeeze broadcast dims to pass (N, A, T, 3) to _seg_pt_closest.
    p1s = p1.expand_as(r)   # (N, A, T, 3)
    p2s = p2.expand_as(r)
    p3s = p3.expand_as(r)
    p4s = p4.expand_as(r)

    d_a1_B, cl_a1_B = _seg_pt_closest(p3s, p4s, p1s)   # arm p1 → torso seg
    d_a2_B, cl_a2_B = _seg_pt_closest(p3s, p4s, p2s)   # arm p2 → torso seg
    d_b1_A, cl_b1_A = _seg_pt_closest(p1s, p2s, p3s)   # torso p3 → arm seg
    d_b2_A, cl_b2_A = _seg_pt_closest(p1s, p2s, p4s)   # torso p4 → arm seg

    # delta = arm_point - torso_closest  (points away from torso)
    delta_a1 = p1s   - cl_a1_B
    delta_a2 = p2s   - cl_a2_B
    delta_b1 = cl_b1_A - p3s    # arm closest - torso point
    delta_b2 = cl_b2_A - p4s

    dists_par  = torch.stack([d_a1_B, d_a2_B, d_b1_A, d_b2_A], dim=-1)   # (N, A, T, 4)
    deltas_par = torch.stack([delta_a1, delta_a2, delta_b1, delta_b2], dim=-2)  # (N, A, T, 4, 3)
    C1s_par    = torch.stack([p1s, p2s, cl_b1_A, cl_b2_A], dim=-2)        # (N, A, T, 4, 3)

    idx        = dists_par.argmin(dim=-1, keepdim=True)                    # (N, A, T, 1)
    dist_par   = dists_par.gather(-1, idx).squeeze(-1)                    # (N, A, T)
    delta_par  = deltas_par.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, 3)).squeeze(-2)
    C1_par     = C1s_par.gather(-2, idx.unsqueeze(-1).expand(*idx.shape, 3)).squeeze(-2)

    # ── Select based on parallel flag ────────────────────────────────────────
    mask  = parallel.unsqueeze(-1)   # (N, A, T, 1)
    dist  = torch.where(parallel, dist_par,  dist_np)
    delta = torch.where(mask,     delta_par, delta_np)
    C1    = torch.where(mask,     C1_par,    C1_np)

    return dist, delta, C1
