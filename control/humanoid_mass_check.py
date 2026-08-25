"""Verify the deploy model's ARM MASSES against the real robot (07-28 wrist swap).

WHY THIS EXISTS
  The 07-28 gimbal-lock-free wrist changed the arm's distal inertials, and the
  deploy model was rebuilt as `upstream's new Fusion CAD + our 967 g cable
  compensation` (the compensation lives only in the exported robot.xml -- it is
  in NO committed source, so it had to be reverse-engineered from the previous
  deploy model and transplanted). One number could not be confirmed offline:
  wrist_1's +90.4 g cable allowance, which happens to be within 0.2 g of the
  redesign's own +90.6 g CAD increase. If that transplant is wrong the error
  lands in gravity compensation AND in the mj_inverse feed-forward -- the exact
  path that produced 292 N.m on 07-24. This tool settles it on hardware without
  disassembling anything.

HOW IT DECIDES
  Whatever the model says, the torque a joint must hold at a static posture is a
  property of the REAL arm. So: hold the arm still, read `joint_effort` (the
  motors' measured torque) and `joint_pos`, then evaluate mj_rne (gravity torque,
  Coriolis vanishes at rest) on EACH candidate model AT THE MEASURED POSTURE and
  compare. The model whose prediction tracks the measurement is the correct one;
  a single mis-massed link shows up as a residual that GROWS from the wrist down
  to the shoulder (every joint carries everything distal to it), which localises
  the offending link.

  Only shoulder_2 / shoulder_3 / elbow / wrist_1 carry meaningful gravity load in
  a hanging arm; wrist_2/3 and shoulder_1 are near their own gravity null and are
  reported but not scored.

USAGE
  T1:  python humanoid_setup_can.py && python humanoid_config.py
  T2:  python humanoid_real_env.py --torque_limit 0.4 --grav_comp
  T3:  python humanoid_mass_check.py                 # both arms, 3 s average
       python humanoid_mass_check.py --seconds 10    # longer average
       python humanoid_mass_check.py --compare-old   # also score the pre-swap model

  Keep the robot HUNG WITH THE LEGS STRAIGHT (bent legs tilt the torso and the
  gravity direction is then wrong for every arm joint) and keep hands OFF the
  arms while sampling -- any contact force is read as gravity.
"""
from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np

from humanoid_model import MJCF_MODEL_PATH
from ipc.publisher import NNGSubscriber

ARM_SLICE = {"left": slice(13, 20), "right": slice(20, 27)}
NAMES = ["shoulder_1", "shoulder_2", "shoulder_3", "elbow",
         "wrist_1", "wrist_2", "wrist_3"]
SCORED = [1, 2, 3, 4]            # joints that actually carry gravity when hanging
GRN, YEL, RED, OFF = "\033[92m", "\033[93m", "\033[91m", "\033[0m"


def gravity_torque(model_path: str, jpos31: np.ndarray) -> np.ndarray:
    """mj_rne gravity torque for all 31 actuated joints at this posture."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[7:7 + 31] = jpos31           # skip the 7-dof free joint (pos + quat)
    d.qvel[:] = 0.0
    # mj_forward before mj_rne: rne consumes the forward-kinematics/CoM state and
    # silently returns zeros without it (humanoid_base.gravity_torques() calls
    # update_kinematics() first for exactly this reason).
    mujoco.mj_forward(m, d)
    mujoco.mj_rne(m, d, 0, d.qfrc_bias)
    return d.qfrc_bias[6:6 + 31].copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot-ip", default="127.0.0.1")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="averaging window; longer beats encoder/torque ripple")
    p.add_argument("--compare-old", action="store_true",
                   help="also score robot.xml.oldwrist_backup_0728 beside it")
    p.add_argument("--tol", type=float, default=0.35,
                   help="N.m residual per scored joint that counts as a match")
    p.add_argument("--vel-tol", type=float, default=0.02,
                   help="rad/s mean joint velocity below which the arm counts as "
                        "static (mj_rne assumes qvel=0)")
    p.add_argument("--dump", metavar="PATH",
                   help="write the averaged posture and measured torque to JSON, "
                        "so a residual pattern can be re-analysed offline (e.g. "
                        "searching for a single-joint encoder offset that explains "
                        "it, which mass errors alone cannot)")
    args = p.parse_args()

    sub = NNGSubscriber(f"tcp://{args.robot_ip}:9870")
    sub.start()
    print(f"model under test: {MJCF_MODEL_PATH}")
    print(f"sampling {args.seconds:.0f}s — keep hands OFF the arms ...")

    pos, eff, vel = [], [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        d = sub.data or {}
        jp, je, jv = d.get("joint_pos"), d.get("joint_effort"), d.get("joint_vel")
        if jp is not None and je is not None and len(jp) >= 31 and len(je) >= 31:
            pos.append(np.asarray(jp[:31], dtype=float))
            eff.append(np.asarray(je[:31], dtype=float))
            vel.append(np.asarray(jv[:31], dtype=float) if jv is not None and len(jv) >= 31
                       else np.zeros(31))
        time.sleep(0.02)
    if len(pos) < 10:
        raise SystemExit("not enough telemetry on 9870 — is real_env running?")

    jpos = np.mean(pos, axis=0)
    meas = np.mean(eff, axis=0)
    # Motion test on VELOCITY, not position spread: mj_rne below assumes qvel=0,
    # so what invalidates the comparison is real motion, and a static arm still
    # shows ~0.5-1 deg of encoder noise in position (an earlier position-spread
    # threshold flagged every run as "moving"). Report the worst joint by name so
    # a single restless joint is obvious instead of condemning the whole sample.
    vmean = np.abs(np.mean(np.asarray(vel)[:, 13:27], axis=0))
    k = int(np.argmax(vmean))
    worst_v, worst_n = float(vmean[k]), f"{'left' if k < 7 else 'right'} {NAMES[k % 7]}"
    jitter = float(np.max(np.std(np.asarray(pos)[:, 13:27], axis=0)))
    print(f"{len(pos)} samples | encoder noise {np.degrees(jitter):.2f} deg | "
          f"max mean |vel| {worst_v:.4f} rad/s ({worst_n})")
    if worst_v > args.vel_tol:
        print(f"{YEL}  ARM STILL MOVING (>{args.vel_tol} rad/s) — gravity-only "
              f"comparison is not valid yet; wait and re-run{OFF}")
    else:
        print(f"{GRN}  arm is static — measurement valid{OFF}")

    cands = [("NEW (new CAD + cable)", MJCF_MODEL_PATH)]
    if args.compare_old:
        old = os.path.join(os.path.dirname(MJCF_MODEL_PATH),
                           "robot.xml.oldwrist_backup_0728")
        if os.path.exists(old):
            cands.append(("OLD (pre-swap)", old))

    preds = {tag: gravity_torque(path, jpos) for tag, path in cands}

    for arm, sl in ARM_SLICE.items():
        print(f"\n=== {arm.upper()} ARM ===")
        head = f"  {'joint':<11}{'measured':>10}" + "".join(
            f"{t.split()[0]:>12}" for t in preds)
        print(head + "   (N.m)")
        for k, n in enumerate(NAMES):
            i = sl.start + k
            row = f"  {n:<11}{meas[i]:>10.2f}"
            for tag in preds:
                r = meas[i] - preds[tag][i]
                mark = "" if k not in SCORED else \
                    (GRN if abs(r) <= args.tol else RED)
                row += f"{mark}{preds[tag][i]:>12.2f}{OFF if mark else ''}"
            print(row + ("" if k in SCORED else "   (not scored)"))
        for tag in preds:
            res = np.abs(meas[sl][SCORED] - preds[tag][sl][SCORED])
            ok = np.all(res <= args.tol)
            print(f"  -> {tag:<24} max residual {res.max():5.2f} N.m  "
                  f"{GRN + 'MATCH' + OFF if ok else RED + 'MISMATCH' + OFF}")

    print(f"\nA residual that GROWS from wrist_1 toward shoulder_2 localises a "
          f"mis-massed link: the first joint (counting outward->inward) whose "
          f"residual jumps carries the wrong link just distal to it. A residual "
          f"that instead alternates in SIGN is not a mass error at all — suspect "
          f"an encoder offset putting the model at the wrong posture (--dump).")

    if args.dump:
        import json
        json.dump({"model": MJCF_MODEL_PATH,
                   "joint_pos": jpos.tolist(),
                   "joint_effort": meas.tolist(),
                   "samples": len(pos),
                   "max_mean_vel": worst_v,
                   "arm_slices": {k: [v.start, v.stop] for k, v in ARM_SLICE.items()},
                   "names": NAMES},
                  open(args.dump, "w"), indent=1)
        print(f"dumped -> {args.dump}")


if __name__ == "__main__":
    main()
