"""Pure safety predicates for the auto-operator (refactor stage S3).

Extraction contract: each function is the VERBATIM body of the corresponding
`AutoOperator` method, parameterized over the module constants (which stay in
`humanoid_auto_operator.py` — single source, and tests monkeypatch module
globals there). The operator keeps one-line delegate methods, so every call
site keeps its exact textual form. Expression forms are preserved unchanged
(including redundant-looking ones) for bit-identity under the replay harness.

Safety-contract coverage (see docs/auto_operator_safety_contract.md):
- target_safe  → SAFE-ENVELOPE-* (trunk keep-out, r-max tilt ceiling, z band,
  finite-vector rejection; INC-4 tilt corruption / 07-15 hardware-damage class)
- sector_of / sector_left → SAFE-STAGING-* sector classification + hysteresis
  (INC-1 cross-body: a held target must never slide across a basin boundary)
- bearing_deg / wrap → shared angular math for the above and for steering wrap
  (SAFE-NAV backward-leg wrap)
- tilt_deg → SAFE-STARTUP tilt refuse/warn inputs (INC-4)
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def bearing_deg(pos) -> float:
    return abs(math.degrees(math.atan2(float(pos[1]), float(pos[0]))))


def sector_of(pos, front_deg: float, rear_deg: float) -> str:
    """FRONT / SIDE / REAR by base-frame bearing of the target."""
    deg = bearing_deg(pos)
    if deg <= front_deg:
        return "front"
    return "side" if deg <= rear_deg else "rear"


def sector_left(pos, sector: str, front_deg: float, rear_deg: float,
                hyst_deg: float) -> bool:
    """True once `pos` has moved beyond `sector`'s bearing band by more than
    the hysteresis margin — detection jitter at a boundary must not flap a
    held arm between sectors."""
    deg = bearing_deg(pos)
    lo, hi = {"front": (0.0, front_deg),
              "side": (front_deg, rear_deg),
              "rear": (rear_deg, 180.0)}[sector]
    return deg > hi + hyst_deg or deg < lo - hyst_deg


def target_safe(pos, r_min: float, r_max: float,
                z_min: float, z_max: float) -> bool:
    """Coarse safety envelope for ANY point the arms may be sent to: outside
    the trunk keep-out cylinder, inside the r-max reach/tilt ceiling and the
    bench z band. The IK has no self-collision awareness — a cube against the
    torso or at face height would otherwise become a literal IK target
    (07-15 audit, two confirmed hardware-damage findings; r-max added 07-18
    against the torso-tilt corruption signature). Unsafe ⇒ treated exactly
    like out-of-reach. Fails CLOSED on non-finite input."""
    p = np.asarray(pos, dtype=float)
    if not np.all(np.isfinite(p)):
        return False
    return (r_max >= float(np.hypot(p[0], p[1]))
            and float(np.hypot(p[0], p[1])) >= r_min
            and z_min <= float(p[2]) <= z_max)


def tilt_deg(telemetry: dict) -> float | None:
    """Torso tilt off vertical [deg] from telemetry projected_gravity
    (body-frame gravity direction; upright ⇒ [0,0,-1]). None when the
    field is absent/malformed — older real_env or a telemetry gap."""
    g = telemetry.get("projected_gravity")
    if g is None:
        return None
    g = np.asarray(g, dtype=float).reshape(-1)
    if g.shape[0] < 3 or not np.all(np.isfinite(g[:3])):
        return None
    n = float(np.linalg.norm(g[:3]))
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(-g[2] / n, -1.0, 1.0))))
