"""Visual-servo bias estimator with its trust rules (S6). SAFE-SERVO-*. Bounded estimator, NOT an error integrator.

Extraction contract (S6): verbatim operator bodies, self -> op; constants and
Phase resolve LIVE against the owning operator module (importlib multi-load +
test monkeypatch safe). See docs/auto_operator_refactor_plan.md.
"""
from __future__ import annotations

import sys

import numpy as np

from .models import ArmTask


def _K(op):
    return sys.modules[type(op).__module__]


def warn_once(op, key: str, msg: str):
    if key not in op._servo_warned:
        op._servo_warned.add(key)
        print(f"[servo] {msg}")


def grip_base_fk(op, arm: str, jpos) -> np.ndarray | None:
    """FK-predicted base-frame position of the gripper base body (L_base/R_base)
    at the telemetry joint vector — the model's belief of the SAME physical frame
    the camera measures via the base tags. Reuses the GimbalCameraFK model/data
    (single tick thread; every user re-writes qpos before reading)."""
    bid = op._grip_body_id.get(arm)
    if bid is None or jpos is None or len(jpos) < op.fk.n_joints:
        return None
    d = op.fk._data
    d.qpos[7:] = 0.0
    d.qpos[7:7 + op.fk.n_joints] = np.asarray(jpos, dtype=float)[:op.fk.n_joints]
    op.fk._mujoco.mj_kinematics(op.fk._model, d)
    return d.xpos[bid].copy()


def servo_step(op, telemetry: dict, jpos, now: float):
    """One bias-estimator update: b̂ ← EMA of (measured − FK) gripper-base position,
    under the trust rules documented at the SERVO_* constants. Also runs the
    blocked-and-blind decay. Call every REACH/HOLD tick."""
    arm = op._reach_arm
    ee = (telemetry.get("ee") or {}).get(arm) or {}
    actual = ee.get("pos_actual")
    executed = (actual is not None and op._servo_cmd_prev is not None and
                float(np.linalg.norm(np.asarray(actual, dtype=float)
                                     - op._servo_cmd_prev)) <= _K(op).SERVO_SETTLE_M)
    if (not executed and op.phase == _K(op).Phase.HOLD and op._servo_bias is not None
            and (now - op._servo_accept_t) > 5.0):
        # Command stopped being executable (obstacle/limit) and no accepted
        # measurement in 5 s: walk b̂ back toward the validated open-loop pose
        # instead of pressing forever with a possibly-poisoned correction.
        n = float(np.linalg.norm(op._servo_bias))
        if n > 1e-6:
            op._servo_bias *= max(0.0, n - _K(op).SERVO_DECAY_MPS * _K(op).DT) / n
            op._warn_once("decay", "command blocked and no accepted gripper "
                            "measurement — decaying the correction back toward "
                            "the open-loop target")

    grip = op._grip.get(arm)
    if grip is None or (now - grip["seen"]) > _K(op).SERVO_FRESH_S:
        if (now - op._reach_t0) > _K(op).SERVO_GIVEUP_S:
            op._warn_once("unseen", f"{arm} gripper tags not seen — reach stays "
                            f"open-loop (monitor --bodies missing "
                            f"{_K(op).SERVO_GRIPPER_KEYS[arm]}, or ≥2 pads not visible)")
        return
    if grip["port"] != op._reach_port:
        if (now - op._reach_t0) > _K(op).SERVO_GIVEUP_S:
            op._warn_once("port", f"packets carry the {arm} gripper from cam "
                            f"{grip['port']} but the target chain is cam "
                            f"{op._reach_port} — no cross-camera servo. NOTE: the "
                            f"monitor publishes ONE entry per key (last camera wins), "
                            f"so cam {op._reach_port} may well see the gripper and "
                            f"be overwritten — not necessarily an aiming problem")
        return
    op._servo_seen = True
    if grip["stamp"] == op._servo_stamp:           # one attempt per NEW camera frame
        return
    op._servo_stamp = grip["stamp"]
    # quasi-static gates: the frame was captured 50-150 ms ago, so trust it only if
    # nothing (arm, gimbal) moved meaningfully across frames around that window
    ee_static = (op._servo_ee_prev is not None and actual is not None and
                 float(np.linalg.norm(np.asarray(actual, dtype=float)
                                      - op._servo_ee_prev)) < _K(op).SERVO_STATIC_M)
    op._servo_ee_prev = None if actual is None else np.asarray(actual, dtype=float)
    # Only the MEASURING camera's own two joints matter — the other camera may
    # legitimately be sweeping (its motion cannot blur this camera's frames).
    ys, ps = _K(op).GAZE_ORDER[op._reach_port]
    gimbal = None if (jpos is None or len(jpos) < 31) else \
        np.asarray([jpos[27 + ys], jpos[27 + ps]], dtype=float)
    gimbal_static = (gimbal is None or op._servo_gimbal_prev is None or
                     float(np.max(np.abs(gimbal - op._servo_gimbal_prev)))
                     < _K(op).SERVO_STATIC_GIMBAL_RAD)
    op._servo_gimbal_prev = gimbal
    if not (executed and ee_static and gimbal_static):
        return
    p_fk = op._grip_base_fk(arm, jpos)
    if p_fk is None:
        op._warn_once("fk", f"cannot FK the {arm} gripper base (telemetry joint_pos "
                        f"missing/short vs model n_joints={op.fk.n_joints}) — "
                        f"servo inactive, reach stays open-loop")
        return
    b_meas = grip["pos"] - p_fk
    if float(np.linalg.norm(b_meas)) > _K(op).SERVO_MEAS_REJECT_M:
        op._warn_once("reject", f"gripper measurement "
                        f"{np.round(b_meas * 1e3).astype(int).tolist()}mm from FK — "
                        f"mis-ID/garbage, rejected (further ones dropped silently)")
        return
    b = op._servo_bias if op._servo_bias is not None else np.zeros(3)
    delta = _K(op).SERVO_GAIN * (b_meas - b)
    dn = float(np.linalg.norm(delta))
    if dn > _K(op).SERVO_STEP_M:                            # one frame moves the hand ≤1 cm
        delta *= _K(op).SERVO_STEP_M / dn
    b = b + delta
    bn = float(np.linalg.norm(b))
    if bn > _K(op).SERVO_BIAS_MAX_M:
        b *= _K(op).SERVO_BIAS_MAX_M / bn
    op._servo_bias = b
    op._servo_accepted += 1
    op._servo_accept_t = now
    if now - op._servo_log_t > 1.0:
        op._servo_log_t = now
        print(f"[servo] hand bias {np.round(b * 1e3).astype(int).tolist()}mm "
              f"({op._servo_accepted} updates, cam {op._reach_port}, raw "
              f"{np.round(b_meas * 1e3).astype(int).tolist()}mm)")


def servo_target(op) -> np.ndarray:
    """The target to PUBLISH: latched reach target minus the learned hand bias
    (raw latched target until the first accepted update / with the servo off).
    Total deviation from the validated open-loop pose is bounded by ‖b̂‖ ≤ 8 cm."""
    tgt = op._reach_target if (not op.visual_servo or op._servo_bias is None) \
        else op._reach_target - op._servo_bias
    op._servo_cmd_prev = tgt.copy()
    return tgt


def servo_ready(op, now: float) -> bool:
    """May REACH report success? Only after the servo has actually corrected
    (≥_K(op).SERVO_MIN_UPDATES) or provably cannot on this leg — otherwise the leg
    would exit at open-loop accuracy while claiming closed-loop success.
    (The 8 s REACH timeout still exits unconditionally.)"""
    if not op.visual_servo:
        return True
    if op._servo_accepted >= _K(op).SERVO_MIN_UPDATES:
        return True
    return (not op._servo_seen) and (now - op._reach_t0) > _K(op).SERVO_GIVEUP_S


def servo_leg_report(op, now: float):
    """One-line closing diagnostic per leg: starvation must not look like success."""
    if not op.visual_servo:
        return
    if op._servo_accepted > 0:
        print(f"[servo] leg done: {op._servo_accepted} updates, hand bias "
              f"{np.round(op._servo_bias * 1e3).astype(int).tolist()}mm")
    else:
        why = "gripper tags never usable (unseen or wrong camera)" if not op._servo_seen \
            else "no frame passed the executed/static gates (IK tracking error > " \
                 f"{_K(op).SERVO_SETTLE_M * 100:.0f} cm, or arm/gimbal never still)"
        print(f"[servo] leg done OPEN-LOOP: 0 accepted updates — {why}")


# ---------------------------------------------------------------------------
# Independent-arm estimator
# ---------------------------------------------------------------------------

def task_warn_once(task: ArmTask, key: str, msg: str) -> None:
    """Emit one warning for one arm task without touching the other arm."""
    if key not in task.servo.warned:
        task.servo.warned.add(key)
        print(f"[servo:{task.arm}] {msg}")


def task_servo_step(op, task: ArmTask, telemetry: dict, jpos, now: float) -> None:
    """Update one task's bounded hand-bias estimator.

    This is deliberately state-isomorphic to :func:`servo_step`, but every
    mutable estimator field lives under ``task.servo``.  Calling it once for
    left and once for right in the same single-threaded operator tick is safe:
    ``grip_base_fk`` rewrites the shared MuJoCo buffer before every read.
    """
    s = task.servo
    arm = task.arm
    ee = (telemetry.get("ee") or {}).get(arm) or {}
    actual = ee.get("pos_actual")
    if actual is not None:
        actual = np.asarray(actual, dtype=float)
        if actual.shape != (3,) or not np.all(np.isfinite(actual)):
            actual = None
    executed = (actual is not None and s.cmd_prev is not None and
                float(np.linalg.norm(actual - s.cmd_prev))
                <= _K(op).SERVO_SETTLE_M)
    if (not executed and s.bias is not None
            and (now - s.accept_t) > 5.0):
        n = float(np.linalg.norm(s.bias))
        if n > 1e-6:
            s.bias *= max(0.0, n - _K(op).SERVO_DECAY_MPS * _K(op).DT) / n
            task_warn_once(task, "decay", "command blocked and no accepted "
                           "gripper measurement — decaying correction")

    grip = op._grip.get(arm)
    if grip is None or (now - grip["seen"]) > _K(op).SERVO_FRESH_S:
        if task.reach_elapsed > _K(op).SERVO_GIVEUP_S:
            task_warn_once(task, "unseen", f"{arm} gripper tags not seen — "
                           "this reach remains open-loop")
        return
    if grip["port"] != task.port:
        if task.reach_elapsed > _K(op).SERVO_GIVEUP_S:
            task_warn_once(task, "port", f"gripper is measured by cam "
                           f"{grip['port']} but target was latched by cam "
                           f"{task.port}; cross-camera correction is disabled")
        return
    if task.port not in _K(op).GAZE_ORDER:
        task_warn_once(task, "port-schema", "latched target has no valid "
                       "camera port; this reach remains open-loop")
        return
    s.seen = True
    if grip["stamp"] == s.stamp:
        return
    s.stamp = grip["stamp"]

    ee_static = (s.ee_prev is not None and actual is not None and
                 float(np.linalg.norm(actual - s.ee_prev))
                 < _K(op).SERVO_STATIC_M)
    s.ee_prev = None if actual is None else actual.copy()
    ys, ps = _K(op).GAZE_ORDER[task.port]
    gimbal = None if (jpos is None or len(jpos) < 31) else \
        np.asarray([jpos[27 + ys], jpos[27 + ps]], dtype=float)
    if gimbal is not None and not np.all(np.isfinite(gimbal)):
        gimbal = None
    gimbal_static = (gimbal is None or s.gimbal_prev is None or
                     float(np.max(np.abs(gimbal - s.gimbal_prev)))
                     < _K(op).SERVO_STATIC_GIMBAL_RAD)
    s.gimbal_prev = gimbal
    if not (executed and ee_static and gimbal_static):
        return

    p_fk = op._grip_base_fk(arm, jpos)
    if p_fk is not None:
        p_fk = np.asarray(p_fk, dtype=float)
    if p_fk is None or p_fk.shape != (3,) or not np.all(np.isfinite(p_fk)):
        task_warn_once(task, "fk", f"cannot FK the {arm} gripper base; "
                       "this reach remains open-loop")
        return
    b_meas = grip["pos"] - p_fk
    if float(np.linalg.norm(b_meas)) > _K(op).SERVO_MEAS_REJECT_M:
        task_warn_once(task, "reject", "gripper measurement is more than "
                       f"{_K(op).SERVO_MEAS_REJECT_M*100:.0f} cm from FK; rejected")
        return
    b = s.bias if s.bias is not None else np.zeros(3)
    delta = _K(op).SERVO_GAIN * (b_meas - b)
    dn = float(np.linalg.norm(delta))
    if dn > _K(op).SERVO_STEP_M:
        delta *= _K(op).SERVO_STEP_M / dn
    b = b + delta
    bn = float(np.linalg.norm(b))
    if bn > _K(op).SERVO_BIAS_MAX_M:
        b *= _K(op).SERVO_BIAS_MAX_M / bn
    s.bias = b
    s.accepted += 1
    s.accept_t = now
    if now - op._servo_log_t > 1.0:
        op._servo_log_t = now
        print(f"[servo:{arm}] hand bias "
              f"{np.round(b * 1e3).astype(int).tolist()}mm "
              f"({s.accepted} updates, cam {task.port})")


def task_servo_target(op, task: ArmTask) -> np.ndarray:
    """Return and remember the Cartesian target for one arm task."""
    if task.target is None:
        raise ValueError(f"{task.arm} task has no latched target")
    tgt = task.target if (not op.visual_servo or task.servo.bias is None) \
        else task.target - task.servo.bias
    task.servo.cmd_prev = tgt.copy()
    return tgt


def task_servo_ready(op, task: ArmTask) -> bool:
    """Whether this arm's reach may finish without consulting another arm."""
    if not op.visual_servo:
        return True
    if task.servo.accepted >= _K(op).SERVO_MIN_UPDATES:
        return True
    return (not task.servo.seen) and \
        task.reach_elapsed > _K(op).SERVO_GIVEUP_S


def task_servo_report(op, task: ArmTask) -> None:
    """Close one arm's servo diagnostics without resetting the peer."""
    if not op.visual_servo:
        return
    if task.servo.accepted > 0:
        print(f"[servo:{task.arm}] reach done: {task.servo.accepted} updates, "
              f"hand bias {np.round(task.servo.bias * 1e3).astype(int).tolist()}mm")
    else:
        why = "gripper tags never usable" if not task.servo.seen \
            else "no frame passed the executed/static gates"
        print(f"[servo:{task.arm}] reach done OPEN-LOOP: {why}")
