# python humanoid_arm_hold_pose.py
"""Park the arms at a NAMED posture and hold them there, for as long as it runs.

WHY THIS EXISTS (2026-08-10). A gamepad walking test needs the legs driven by
the policy and the arms parked somewhere chosen — but real_env's arm path has
exactly two states: something is streaming arm targets, or the 0.5 s silence
failsafe crawls the arms to `default_pose`. There is no "hold this posture"
mode. So a session with no arm publisher always ends up at the CURRENT home,
whatever posture the operator actually wanted. This is the missing publisher:
one posture, held at 20 Hz, nothing else touched.

    python humanoid_arm_hold_pose.py --pose front      # park + hold, Ctrl+C to release
    python humanoid_arm_hold_pose.py --pose front --dry-run   # audit only, no packets

Design decisions:
- POSTURES ARE THE OPERATOR'S, imported not retyped: STAGE_JOINTS front/side/
  rear, POWERON_JOINTS, the two recognised legacy homes and SIDE_HOME_JOINTS.
  A posture that drifts here while the operator's moves is how a "known home"
  silently stops being one; tests already pin the operator's copies.
- THE GLIDE IS AUDITED BEFORE ANY PACKET, against the path the RECEIVER will
  actually take. real_env does not interpolate along a straight joint segment:
  it clamps every joint toward the target at the same `rate`, so joints with
  small deltas arrive first and the traversed curve is direction-dependent.
  This simulates that curve from the MEASURED start and sweeps it with the
  joint monkey's own clearance auditor (the same preflight cuRobo's gate C
  uses); below --min-clearance-mm it refuses instead of publishing.
- VIA THE SIDE HUB BY DEFAULT, and that is not a style choice. Measured on the
  deploy model, 2026-08-10:
      POWERON -> front           DIRECT   24.3 mm   (right gripper base ~ torso,
                                                     mid-glide, frame 79/198)
      POWERON -> side -> front            39.5 mm   (never worse than the home
                                                     pose's own static floor)
  24.3 mm is under cuRobo's own 25 mm gate floor on the SAME sag-sensitive pair
  that loses ~18 mm at 3 deg of droop. The audited chains are POWERON<->SIDE and
  front<->side<->rear (operator, 07-14/07-17), and routing through the hub stays
  inside them. --no-via-side takes the direct path if the audit passes.
- RETURN IS FREE. front -> POWERON measures 39.5 mm both at 0.3 rad/s and at the
  failsafe's own 0.125 rad/s, so releasing (Ctrl+C -> silence -> failsafe crawl)
  needs no special handling: the arms walk home clean.
- Binds 9874 like every other arm commander, so the operator, humanoid_curobo_
  reach and the joint monkey must all be OFF (single-binder rule).
- Tracking watchdog, same contract as the joint monkey: measured vs commanded
  divergence past WATCH_ERR_RAD for WATCH_ABORT_S stops publishing rather than
  pressing on a jam.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import tyro

import humanoid_auto_operator as OP
from humanoid_joint_monkey_hw import (
    ARM_JOINTS_L,
    ARM_JOINTS_R,
    ARM_MIRROR_SIGN,
    NNGPublisher,
    NNGSubscriber,
    TEL_ARM,
    preflight,
)
from humanoid_model import MJCF_MODEL_PATH

STREAM_HZ = 20
RECEIVER_HZ = 50.0          # real_env's control period — the rate limiter's dt
ARRIVE_TOL_RAD = 0.05       # measured-vs-target for "waypoint reached"
ARRIVE_TIMEOUT_S = 40.0     # per waypoint; a stalled glide stops, never hangs
WATCH_ERR_RAD = 0.50        # == the joint monkey's, for the same harness reason
WATCH_ABORT_S = 2.0

POSES = {
    "front": OP.STAGE_JOINTS["front"],
    "side": OP.STAGE_JOINTS["side"],
    "rear": OP.STAGE_JOINTS["rear"],
    "poweron": OP.POWERON_JOINTS,
    "legacy_0730": OP.POWERON_JOINTS_LEGACY_0730,
    "v83": OP.POWERON_JOINTS_V83,
    "side_home": OP.SIDE_HOME_JOINTS,
}


@dataclass
class Args:
    pose: Literal["front", "side", "rear", "poweron", "legacy_0730", "v83",
                  "side_home"] = "front"
    """Posture to park at (the operator's own constants)."""
    rate: float = 0.3
    """Commanded joint rate [rad/s] for the glide and the hold."""
    via_side: bool = True
    """Route through STAGE side, the audited transit hub. See the module
    docstring: the direct POWERON->front glide measures 24.3 mm."""
    min_clearance_mm: float = 25.0
    """Refuse to publish if the simulated glide gets closer than this."""
    robot_ip: str = "127.0.0.1"
    dry_run: bool = False
    """Audit and print the plan; send nothing."""


def full(left) -> np.ndarray:
    """7 left-arm joints -> the 14-vector, right = the deploy mirror."""
    left = np.asarray(left, dtype=float)
    return np.concatenate([left, ARM_MIRROR_SIGN * left])


def receiver_glide(q0: np.ndarray, q1: np.ndarray, rate: float) -> list:
    """The curve real_env's rate limiter ACTUALLY traverses: every joint
    clamped toward the target at `rate`, so short-delta joints arrive first.
    Not a straight segment, and not symmetric — front->POWERON and
    POWERON->front are different paths through space."""
    q, out, step = q0.copy(), [q0.copy()], rate / RECEIVER_HZ
    while float(np.max(np.abs(q1 - q))) > 1e-9:
        q = q + np.clip(q1 - q, -step, step)
        out.append(q.copy())
    return out


class _Frames:
    """preflight only reads .frames."""

    def __init__(self, frames):
        self.frames = frames


def arm_fault_why(d) -> str | None:
    """The dropout watchdog's TERMINAL latch, named. Refusal string, or None.

    THE FAILURE THIS EXISTS FOR (2026-08-10, first hardware run of this tool):
    left_wrist_3 dropped off the CAN bus ~4 s into the glide, real_env's
    watchdog latched the whole LEFT arm into a damped hold — no chasing,
    feed-forward zeroed — and this tool, which could only compare measured
    against commanded, reported "side not reached in 40s", then the same for
    "front", then "diverged 0.67 rad (jam?)". Three true statements, none of
    them the fault. real_env publishes the latch on 9870 as `arm_fault`
    [left, right]; humanoid_curobo_reach has always refused on it
    (_fatal_telemetry) and this tool had no equivalent."""
    fault = (d or {}).get("arm_fault") or [False, False]
    if not any(fault):
        return None
    who = "+".join(s for s, f in zip(("left", "right"), fault) if f)
    return (f"ARM FAULT latched by the dropout watchdog on the {who} arm — "
            f"that arm is held damped at its measured pose and cannot be "
            f"commanded at all. Inspect the CAN harness / motor, then RESTART "
            f"real_env: the latch is terminal by design.")


def main() -> int:
    args = tyro.cli(Args)
    model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS_L + ARM_JOINTS_R]

    target = full(POSES[args.pose])
    route = [full(OP.STAGE_JOINTS["side"])] if (
        args.via_side and args.pose not in ("side", "side_home")) else []
    route.append(target)

    tel = NNGSubscriber(f"tcp://{args.robot_ip}:9870")
    tel.start()
    print("waiting for telemetry ...")
    t0, jp = time.time(), None
    while time.time() - t0 < 10.0:
        d = tel.data
        if d and d.get("joint_pos") is not None and len(d["joint_pos"]) >= 31:
            jp = np.asarray(d["joint_pos"], dtype=float)
            if np.all(np.isfinite(jp[TEL_ARM])):
                break
        time.sleep(0.1)
    if jp is None or not np.all(np.isfinite(jp[TEL_ARM])):
        print("no finite telemetry on 9870 — is real_env running?")
        return 2
    start = jp[TEL_ARM].copy()
    why = arm_fault_why(tel.data)
    if why:                          # before the audit: a latched arm cannot
        print(f"REFUSED: {why}")     # move, so the route is moot
        return 4

    # AUDIT THE WHOLE ROUTE FROM WHERE THE ARM ACTUALLY IS, not from a home we
    # assume it is at: the glide is what sweeps space, and its shape depends on
    # the start.
    frames, q = [], start.copy()
    for wp in route:
        frames += receiver_glide(q, wp, args.rate)
        q = wp.copy()
    worst, where = preflight(model, data, _Frames(frames), qadr)
    names = (["side"] if len(route) > 1 else []) + [args.pose]
    print(f"route: measured -> {' -> '.join(names)}  "
          f"({len(frames)} frames, {len(frames)/RECEIVER_HZ:.1f}s at {args.rate} rad/s)")
    print(f"preflight: min clearance {worst*1000:.1f} mm at {where}")
    if worst * 1000.0 < args.min_clearance_mm:
        print(f"REFUSED: below the {args.min_clearance_mm:.0f} mm floor. Try "
              f"--via-side (if off), a different --pose, or home the arms first.")
        return 3
    if args.dry_run:
        print("dry-run: nothing published.")
        return 0

    pub = NNGPublisher("tcp://*:9874")
    print(f"holding {args.pose} — Ctrl+C releases (the failsafe then crawls the "
          f"arms back to default_pose at 0.125 rad/s)")
    diverged_since, cmd = None, None
    try:
        for wp, name in zip(route, names):
            cmd = wp
            t_wp, err = time.time(), float("nan")
            while True:
                pub.publish({"arm_targets": {
                    "joint_pos": [float(v) for v in wp],
                    "rate": float(args.rate)}})
                d = tel.data
                why = arm_fault_why(d)
                if why:
                    raise RuntimeError(why)
                if d and d.get("joint_pos") is not None and len(d["joint_pos"]) >= 31:
                    jp = np.asarray(d["joint_pos"], dtype=float)
                    err = float(np.max(np.abs(jp[TEL_ARM] - wp)))
                    if err <= ARRIVE_TOL_RAD:
                        print(f"  reached {name} (worst joint {err:.3f} rad)")
                        break
                if time.time() - t_wp > ARRIVE_TIMEOUT_S:
                    # STOP, do not walk on. The next waypoint's clearance was
                    # audited from THIS one; commanding it from wherever a
                    # stalled arm actually stopped sweeps a path nothing
                    # checked. (2026-08-10: the first hardware run marched
                    # through both timeouts with a dead arm.)
                    raise RuntimeError(
                        f"{name} not reached in {ARRIVE_TIMEOUT_S:.0f}s "
                        f"(worst joint {err:.3f} rad) — refusing to command "
                        f"the next waypoint from an unaudited posture")
                time.sleep(1.0 / STREAM_HZ)
        while True:                      # hold the final posture until Ctrl+C
            pub.publish({"arm_targets": {
                "joint_pos": [float(v) for v in target],
                "rate": float(args.rate)}})
            d = tel.data
            why = arm_fault_why(d)
            if why:
                raise RuntimeError(why)
            if d and d.get("joint_pos") is not None and len(d["joint_pos"]) >= 31:
                jp = np.asarray(d["joint_pos"], dtype=float)
                err = float(np.max(np.abs(jp[TEL_ARM] - target)))
                if not (err <= WATCH_ERR_RAD):      # NaN counts as divergence
                    if diverged_since is None:
                        diverged_since = time.time()
                        print(f"[watch] holding err {err:.2f} rad — watching")
                    elif time.time() - diverged_since > WATCH_ABORT_S:
                        raise RuntimeError(
                            f"arm diverged {err:.2f} rad from the held posture "
                            f"for >{WATCH_ABORT_S}s (jam?) — releasing")
                else:
                    diverged_since = None
            time.sleep(1.0 / STREAM_HZ)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nreleasing: {e if str(e) else 'Ctrl+C'}")
        print("stream silenced — real_env's failsafe crawls the arms to "
              "default_pose at 0.125 rad/s (audited 39.5 mm from front).")
    finally:
        pub.close()
        tel.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
