"""Gaze aiming, panoramic scan, trim and slew (S6). SAFE-GAZE-*; INC-5.

Extraction contract (S6): verbatim operator bodies, self -> op; constants and
Phase resolve LIVE against the owning operator module (importlib multi-load +
test monkeypatch safe). See docs/auto_operator_refactor_plan.md.
"""
from __future__ import annotations

import math
import sys

import numpy as np


def _K(op):
    return sys.modules[type(op).__module__]


def probe_cam_frame(op, port: int) -> tuple[float, float, float]:
    """Measure this camera's optical frame from the model FK: (datum_bearing,
    yaw_sense, pitch_sense).

    datum_bearing: base-frame azimuth of the optical axis at joints=0 (left cam ~0,
    rear-mounted right cam ~π). yaw_sense/pitch_sense: sign of d(bearing)/d(yaw) and
    d(elevation)/d(pitch). Probed, not assumed — the deploy re-export owns these
    conventions and has already flipped yaw once (07-10)."""
    ys, ps = _K(op).GAZE_ORDER[port]

    def optics(yaw: float, pitch: float) -> tuple[float, float]:
        j = np.zeros(31)
        j[27 + ys], j[27 + ps] = yaw, pitch
        fwd = op.fk.base_to_camera(port, j)[:3, :3][:, 2]   # OpenCV: +z = optical axis
        return (math.atan2(fwd[1], fwd[0]),
                math.asin(max(-1.0, min(1.0, float(fwd[2])))))

    eps = 0.01
    b0, e0 = optics(0.0, 0.0)
    b1, _ = optics(eps, 0.0)
    _, e1 = optics(0.0, eps)
    return b0, math.copysign(1.0, op._wrap(b1 - b0)), math.copysign(1.0, e1 - e0)


def aim_angles(op, port: int, target: np.ndarray, jpos) -> tuple[float, float]:
    """Closed-form parallax-corrected aim (policies.py:229-238), + integral trim,
    in the PROBED camera frame (see _probe_cam_frame — no hardcoded signs).

    The raw yaw is unwrapped onto the branch NEAREST the actual joint
    (policies.py:716-735): the rear-mounted right camera tracking a FRONT cube
    sits exactly on the ±π cut, where detection noise would otherwise flip the
    command by 2π and send the gimbal the long way around (yaw ROM is ±4.71 rad,
    so off-principal branches are legal).
    """
    cam_pos = op.fk.base_to_camera(port, jpos)[:3, 3]
    d = target - cam_pos
    d = d / max(np.linalg.norm(d), 1e-6)
    b0, yaw_sense, pitch_sense = op._cam_frame[port]
    # target's absolute bearing/elevation → joint angles in this camera's frame:
    # FK gives bearing ≈ b0 + yaw_sense*yaw, elevation ≈ pitch_sense*pitch.
    yaw = op._wrap(math.atan2(d[1], d[0]) - b0) / yaw_sense
    pitch = math.asin(max(-1.0, min(1.0, float(d[2])))) / pitch_sense
    # Refine against the FULL FK: the closed form assumes an ideal decoupled
    # pan-tilt, but the deploy model's mount has slight axis skew (~4.7° off at
    # combined yaw + up-pitch, measured). Two fixed-point passes kill the coupling
    # and also re-evaluate parallax at the COMMANDED pose.
    ys_slot, ps_slot = _K(op).GAZE_ORDER[port]
    j_ref = (np.array(jpos, dtype=float).copy()
             if (jpos is not None and len(jpos) >= 31) else np.zeros(31))
    for _ in range(2):
        j_ref[27 + ys_slot], j_ref[27 + ps_slot] = yaw, pitch
        T = op.fk.base_to_camera(port, j_ref)
        axis, pos = T[:3, :3][:, 2], T[:3, 3]
        dd = target - pos
        dd = dd / max(np.linalg.norm(dd), 1e-6)
        yaw += op._wrap(math.atan2(dd[1], dd[0]) - math.atan2(axis[1], axis[0])) / yaw_sense
        pitch += (math.asin(max(-1.0, min(1.0, float(dd[2]))))
                  - math.asin(max(-1.0, min(1.0, float(axis[2]))))) / pitch_sense
    if jpos is not None and len(jpos) >= 31:
        yaw_slot, pitch_slot = _K(op).GAZE_ORDER[port]
        act = (float(jpos[27 + yaw_slot]), float(jpos[27 + pitch_slot]))
        yaw = act[0] + op._wrap(yaw - act[0])         # nearest-branch unwrap
        # Integral trim, gated to near-lock (policies.py:737-744): transit error
        # during an acquisition swing must not wind the bias to the clamp.
        tr = op._trim[port]
        if abs(yaw - act[0]) < 0.3 and abs(pitch - act[1]) < 0.3:
            tr[0] = float(np.clip(tr[0] + _K(op).TRIM_GAIN * (yaw - act[0]), -_K(op).TRIM_MAX, _K(op).TRIM_MAX))
            tr[1] = float(np.clip(tr[1] + _K(op).TRIM_GAIN * (pitch - act[1]), -_K(op).TRIM_MAX, _K(op).TRIM_MAX))
        yaw, pitch = yaw + tr[0], pitch + tr[1]
    return yaw, pitch


def scan_angles(op, port: int) -> tuple[float, float]:
    """Single-row panoramic sweep, BOTH eyes (user 2026-08-02: "no serpentine —
    fixed 47° down, no raising or lowering, same rotation range"). Pitch is
    CONSTANT at SCAN_PITCH_DOWN; only yaw sweeps, on the serpentine's exact
    yaw trajectory: from the camera's own NEUTRAL turn LEFT (CCW) half a
    circle, then loop FULL 360° circles with direction alternating per lap so
    each circle unwinds the previous one — the yaw joint bounces within
    exactly ±180° of its zero (inside the ±270° limit). Turnarounds land at
    neutral+180°: rear for the left eye, front for the right. s_rel = bearing
    offset from the camera's own datum (+ = CCW = robot's left); joint =
    s_rel × probed sense — no wrapping, so the sweep passes ±180° continuously
    and stays model-convention-agnostic."""
    _, yaw_sense, pitch_sense = op._cam_frame[port]
    lead_t = math.pi / _K(op).SCAN_YAW_RATE                    # neutral → half-turn
    row_t = 2 * math.pi / _K(op).SCAN_YAW_RATE                 # one full circle
    # Pitch down FIRST, dwell, THEN sweep (user 2026-08-04): rotating while
    # still pitching swept the start sector past an unsettled camera.
    t = op._scan_phase - _K(op).SCAN_SETTLE_S
    row = _K(op).SCAN_PITCH_DOWN                               # 47° down, always
    if t < 0.0:
        s_rel = 0.0                                     # settle: pitch only
    elif t < lead_t:
        s_rel = _K(op).SCAN_YAW_RATE * t                       # 0 → +π (CCW / leftward)
    else:
        u = (t - lead_t) % (2 * row_t)
        k = int(u // row_t)
        w = u - k * row_t
        if k % 2 == 0:                                  # even lap: clockwise
            s_rel = math.pi - _K(op).SCAN_YAW_RATE * w         # +π → −π
        else:                                           # odd lap: counter-clockwise
            s_rel = -math.pi + _K(op).SCAN_YAW_RATE * w        # −π → +π
    # STEP-AND-STARE (user 2026-08-02): quantize the bearing so the gaze slew
    # (1.5 rad/s) sprints between quanta and the camera holds still between
    # them — sharp frames are what decode a tag; a continuous fast pan blurs
    # every frame and sweeps past visible cubes without locking.
    step = _K(op).SCAN_STEP_RAD
    s_rel = max(-math.pi, min(math.pi, round(s_rel / step) * step))
    yaw = s_rel * yaw_sense
    pitch = -row / pitch_sense                          # rows are "rad DOWN"
    return yaw, pitch


def slew_gaze(op, target: np.ndarray) -> np.ndarray:
    """Rate-limit the published gaze reference to _K(op).GAZE_SLEW_RATE (sender-side; the
    robot applies gaze_reference raw since 8045371). Bounds every jump source:
    scan→lock acquisition, branch changes, first-tick offsets."""
    max_step = _K(op).GAZE_SLEW_RATE * _K(op).DT
    return op.gaze + np.clip(np.asarray(target, dtype=float) - op.gaze,
                               -max_step, max_step)


def eye(op, jpos, now: float) -> np.ndarray:
    """Per-camera: _K(op).TRACK its claimed cube while the sighting is fresh, else scan.

    Freshness gate matters: base-frame positions rot as the robot moves, so a
    camera staring at a stale point would never rediscover its cube — falling
    back to scanning is what makes SEARCH re-entry actually search.
    """
    gaze = op.gaze.copy()
    unclaimed_pending = any(c.camera_port is None or not c.fresh(now, _K(op).DET_LOST_RESCAN_S)
                            for c in op.cubes.values() if not c.reached)
    for port in _K(op).CAM_PORTS:
        # RETIRED = PARKED AT ORIGIN, PERMANENTLY (user 2026-08-09: "the
        # camera must stick to the origin when his mission completed. Never
        # join for finding next one"). This SUPERSEDES the 07-26 rejoin rule
        # (a retired eye used to rejoin the serpentine while any pending cube
        # lacked a fresh fix — the fix for the one-eyed ~50 s leg-2 hunt at
        # range): the next leg's search is one-eyed now, BY ORDER, and the
        # journey's walk-then-reacquire happens at ~0.5 m where one eye
        # decodes reliably. perception.ingest enforces the same rule from the
        # claim side (a retired port is never claimable), so no stale binding
        # below can aim this eye either.
        if port in op._cam_retired:
            ys, ps = _K(op).GAZE_ORDER[port]
            gaze[ys], gaze[ps] = 0.0, 0.0
            continue
        # LOCKED = POINTED, WITH A GRACE (user 2026-08-10: "fix on the cube
        # as long as you can; if you lose track, just scan around again" —
        # supersedes the 08-04 permanent stare). The eye keeps aiming at the
        # last known position for GAZE_STARE_GRACE_S past the last sighting:
        # long enough to hold still through a grasp's hand-occlusion window,
        # short enough that a journey walk's rotted body-frame point cannot
        # pin the eye on empty space (seen live 08-10). Past the grace the
        # eye rejoins the serpentine; the binding still survives claim
        # release (last_camera_port) so a brief flicker never moves it.
        mine = next((c for c in op.cubes.values()
                     if not c.reached and c.pos is not None
                     and c.fresh(now, _K(op).GAZE_STARE_GRACE_S)
                     and (c.camera_port == port
                          or (c.camera_port is None
                              and c.last_camera_port == port))), None)
        if mine is not None:
            yaw, pitch = op._aim_angles(port, mine.pos, jpos)
        elif unclaimed_pending and not getattr(op, "scan_hold", False):
            yaw, pitch = op._scan_angles(port)
        elif unclaimed_pending:
            # SCAN HOLD (2026-08-12, hardware): during a journey FETCH the
            # other leg's cube has expired its claim, so the free eye used to
            # serpentine through the whole reach — and a sweeping camera
            # reference DRAGS THE BODY: the gaze curriculum taught every
            # MixedArmsCam checkpoint to recruit base rotation when a camera
            # target runs from its gimbal, so the base ground slowly rightward
            # through every front grasp, on BOTH v2best and Cosine, with the
            # waist pinned (the legs carried it). Killing the operator stopped
            # it instantly — the user's Ctrl+C differential found the driver.
            # While the mission holds a locked target and needs a still base,
            # an eye with nothing to track PARKS instead of sweeping; the
            # search resumes when the mission clears the hold for the next
            # walk. (Standing missions never set scan_hold: both their eyes
            # hold claims through the grasp, so nothing changes there.)
            yaw, pitch = 0.0, 0.0
        else:
            # Nothing left to search for: PARK at neutral instead of sweeping.
            # A sweeping idle camera is not harmless — it periodically crosses
            # the arm/cube region and its grazing single-tag detections OVERWRITE
            # the tracking camera's entries in the monitor's one-entry-per-key
            # targets dict (last camera wins), starving the visual servo.
            yaw, pitch = 0.0, 0.0
        ys, ps = _K(op).GAZE_ORDER[port]
        gaze[ys], gaze[ps] = yaw, pitch
    op._scan_phase += _K(op).DT
    return gaze

