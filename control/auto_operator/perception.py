"""Detection ingest, camera claims, gripper sightings (S6). SAFE-DET-*.

Extraction contract (S6): verbatim operator bodies, self -> op; constants and
Phase resolve LIVE against the owning operator module (importlib multi-load +
test monkeypatch safe). See docs/auto_operator_refactor_plan.md.
"""
from __future__ import annotations

import sys

import numpy as np

from .models import Sighting


def _K(op):
    return sys.modules[type(op).__module__]


def ingest(op, detections: dict, now: float):
    """Update cube tracks from a MonitorDetectionPacket['targets'] dict."""
    for key, cube in op.cubes.items():
        # Claim expiry: cube unseen past _K(op).DET_UNCLAIM_S → release its camera so
        # the next sighting (from either camera) can rebind it.
        if cube.camera_port is not None and not cube.fresh(now, _K(op).DET_UNCLAIM_S):
            print(f"[lock] {key} released from camera {cube.camera_port} "
                  f"(unseen {_K(op).DET_UNCLAIM_S:.0f}s)")
            cube.camera_port = None
        t = detections.get(key)
        if t is None or t.get("n_inliers", 0) < 1:
            continue
        pos = np.asarray(t.get("pos"), dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            continue    # malformed/NaN detection: must never enter a track — a
                        # single NaN here used to poison the gaze reference and
                        # every downstream target (07-15 audit, fail-open class)
        if (now - cube.last_seen) > 1.0:
            cube.first_seen = now          # new sighting streak
        cube.pos = pos
        cube.pos_port = int(t["camera_port"])
        # DEDUPLICATE ON CAPTURE STAMP (audit 2026-08-06). The monitor retains
        # each key's newest pose for DETECTION_RETENTION_S and ships it in
        # EVERY 20 Hz packet (humanoid_monitor.build_retained_targets), so
        # appending on arrival let ONE decode occupy all five slots — and the
        # median of five identical values is that value, i.e. the whole point
        # of this deque quietly evaporated exactly when a hand was occluding
        # its own cube. The stamp is the monitor's monotonic clock, the same
        # clock `now` comes from, and the gripper branch below has always
        # trusted it. Falling back to `now` keeps a publisher that omits the
        # field working, at the old (duplicating) behaviour.
        stamp = float(t.get("capture_stamp", now))
        if not cube.hist or stamp > cube.hist[-1].t:
            # ORIENTATION rides along since 2026-08-13 (see Sighting.quat).
            # Validated to the same standard as `pos`: a malformed or
            # non-unit quaternion is dropped to None rather than entering a
            # track, because downstream it rotates a grasp goalset.
            q = t.get("quat_wxyz")
            if q is not None:
                try:
                    q = np.asarray(q, dtype=float)
                except (TypeError, ValueError):
                    q = None            # a non-numeric payload must DROP, not
                                        # raise: this runs inside the operator
                                        # tick, and one malformed publisher
                                        # field would take the control loop
                                        # down with it
                else:
                    n = float(np.linalg.norm(q)) if q.shape == (4,) else 0.0
                    q = q / n if (q.shape == (4,) and np.all(np.isfinite(q))
                                  and n > 1e-6) else None
            cube.hist.append(Sighting(
                t=stamp, pos=cube.pos, port=cube.pos_port,
                faces=int(t.get("n_inliers", 0) or 0), quat=q))
        # last_seen stays on the INGEST clock on purpose: it drives the camera
        # claim expiry and the gaze/journey freshness gates, whose question is
        # "is this cube still being reported?", not "how old is the decode?".
        # Retention inflates it by at most DETECTION_RETENTION_S.
        cube.last_seen = now
        if cube.camera_port is None and not getattr(cube, "no_claim", False):
            # (no_claim: a CARRIED cube rides in the gripper — re-claiming a
            # camera for it would starve the other cube's search)
            claimed_ports = {c.camera_port for c in op.cubes.values() if c.camera_port is not None}
            port = int(t["camera_port"])
            # FIRST SEER LOCKS (user 2026-08-02): the eye that actually saw
            # the cube locks it immediately; the other eye keeps scanning.
            # The retired cross-claim ("spotted by X, which is taken → assign
            # the free eye") sent an eye to stare at a position it had never
            # decoded, and could livelock: the busy eye keeps grazing the
            # cube on its serpentine — refreshing the claim's freshness —
            # while the assigned eye never decodes it, so NOBODY ever stops
            # on a cube the monitor is visibly drawing. A sighting from a
            # busy eye still updates pos/hist above (GO/REACH use it); the
            # claim just stays open until a FREE eye sees the cube itself —
            # the full-circle serpentine guarantees it comes around.
            # A RETIRED port is out of the pool for good (user 2026-08-09:
            # a camera whose cube was carried home sticks to the origin).
            # This claim used to un-retire the port ("actively needed
            # again"); now the claim stays open until the FREE eye sees the
            # cube itself — the same wait the first-seer rule already
            # imposes on busy-eye sightings. pos/hist still updated above.
            if port not in claimed_ports and port not in op._cam_retired:
                cube.camera_port = port
                cube.last_camera_port = port
                print(f"[lock] {key} → camera {port} (first seer)")
    # Gripper bases are measurement-only (visual servo): latest sighting per arm,
    # never tracked/claimed like cubes, never a gaze target. ≥2 tag faces required:
    # single-16mm-tag PnP poses are garbage-prone and this feeds a control loop.
    for arm, key in _K(op).SERVO_GRIPPER_KEYS.items():
        t = detections.get(key)
        if t is not None and t.get("n_inliers", 0) >= _K(op).SERVO_MIN_INLIERS:
            gpos = np.asarray(t.get("pos"), dtype=float)
            if gpos.shape != (3,) or not np.all(np.isfinite(gpos)):
                continue                   # same NaN/malformed drop as the cubes
            op._grip[arm] = {"pos": gpos,
                               "port": int(t["camera_port"]),
                               "stamp": float(t.get("capture_stamp", now)),
                               "seen": now}

