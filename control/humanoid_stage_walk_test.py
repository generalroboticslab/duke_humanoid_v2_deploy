"""One-shot hardware validation of the audited joint-space staging chain.

Walks ONE arm through the staged-reach transit postures and back —
    FRONT(power-on) → SIDE(pure coronal abduction 25°) → REAR(mirror-solved) → SIDE → FRONT
— using the JOINT arm-stream schema, so the robot reproduces EXACTLY the
configurations the offline audit certified (limit margins ≥40°, ≥49.9 mm
arm-vs-everything clearance along the true equal-rate path and 300 random
in-envelope poses; see scratchpad joint_stage_audit.py, 2026-07-14).

Deliberately SLOW and deliberately INCOMPLETE:
  * slow: real_env rate-caps the joint stream at _ARM_RETURN_RATE = 0.025 rad/s
    (its failsafe crawl). Biggest hop ≈ 89 s; full out-and-back ≈ 4.5 min.
    This run is a one-time validation, not the demo path — the demo needs the
    robot-side commanded-rate knob (ask pending with upstream).
  * incomplete: NO Cartesian reach at the far end. real_env re-seeds the IK's
    internal arm state from encoders ONLY at startup/reset
    (_ik_snap_orientation_targets); switching joint→Cartesian away from the
    front basin would let the IK solve from a stale front-basin warm start and
    command a jump. Until that re-seed exists for stream-mode switches, this
    test never leaves joint mode.

Safety properties:
  * launch real_env WITHOUT --use-ik → Cartesian packets are structurally
    ignored; only the crawling joint path can move the arm;
  * the OTHER arm is frozen at a start-time encoder snapshot (the joint schema
    is all-14-at-once);
  * Ctrl+C at any point → arm stream goes silent → real_env's failsafe ramps
    the arm back to the power-on pose at the same 0.025 rad/s crawl;
  * per-hop timeout = expected crawl time × 1.4 + 5 s → abort = stop publishing
    (same failsafe ramp home).

Run (operator must NOT be running — this script binds 9874):
    python humanoid_real_env.py --task HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam \
        --torque_limit 0.2 --enable-motor arm --arm-sender-ip 127.0.0.1
    python humanoid_stage_walk_test.py --arm left
"""
from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np

from ipc.publisher import NNGPublisher, NNGSubscriber

RATE_HZ = 20.0
CRAWL_RAD_S = 0.025          # real_env _ARM_RETURN_RATE — the robot-side crawl
ARR_TOL_RAD = 0.03           # per-joint arrival tolerance (1.7 deg). Loose tolerance
                             # CUTS CORNERS at speed: the next hop redirects joints
                             # that are still up to tol away from the waypoint, so the
                             # arm never truly forms the transit posture and deviates
                             # from the audited curve (user observed the SIDE waypoint
                             # being skipped at 0.3 rad/s). 0.03 bounds the deviation
                             # to ~1 cm against the 51 mm audited clearance.
DWELL_S = 0.6                # hold each waypoint this long after arrival — makes the
                             # posture crisp and guarantees the next segment starts
                             # from the audited configuration, at any rate
# Left-arm postures (shoulder_1..wrist_3); the right arm uses the exact negation
# (verified mirror convention of the deploy default pose).
FRONT_L = [-0.1786, -0.6451, -0.4061, 1.4804, -0.4229, -0.5632, -0.5089]
# ^ the V83 front station (== operator POWERON_JOINTS_V83). It was ALSO the
#   power-on pose when this chain was audited; since then power-on has moved
#   twice (chest 07-30, sim home 07-31) and the chain deliberately has NOT —
#   the audit belongs to these waypoints, not to wherever power-on lives now.
# Right arm = left arm negated. Kept as an explicit vector (not a bare -q)
# because upstream briefly broke the negation on wrist_2; see
# humanoid_auto_operator.ARM_MIRROR_SIGN. Duplicated rather than imported:
# this tool is the RECOVERY path the operator's startup-refusal points at
# and must run even when the operator module cannot be imported.
ARM_MIRROR_SIGN = np.array([-1., -1., -1., -1., -1., -1., -1.])
# SIDE (user-corrected definition 07-14): PURE CORONAL ABDUCTION — from the natural
# hang (all-zero arm), raise the straight arm sideways in the frontal plane, zero
# sagittal component. FK-verified: shoulder_2 is exactly the coronal-abduction axis
# (EE x stays 0.000 through the sweep); 25 deg abduction (user-tuned), elbow/wrists neutral.
# Audited (45/60/90 deg all pass): pose clearance 51 mm, all four transitions
# (both directions) 51 mm, limit margin 90 deg. The earlier "DIAG" (v83 hanging
# pose) mixed in a sagittal component and is retired.
SIDE_L = [0.0, -0.4363, 0.0, 0.0, 0.0, 0.0, 0.0]
REAR_L = [0.085, -0.782, 0.401, -1.220, 1.218, -0.686, -0.377]  # mirror-solved, 2.7 mm EE err
CHAIN = [("SIDE", SIDE_L), ("REAR", REAR_L), ("SIDE", SIDE_L), ("FRONT", FRONT_L)]


def main():
    import tyro

    @dataclasses.dataclass
    class Args:
        robot_ip: str = "127.0.0.1"   # telemetry source (real_env on this machine)
        arm: str = "left"             # which arm walks the chain; the other freezes
        tol: float = ARR_TOL_RAD
        dwell: float = DWELL_S        # freeze time at each waypoint [s] — raise to
                                      # visually compare outbound vs return postures
        rate: float = CRAWL_RAD_S     # commanded joint rate [rad/s]; needs the
                                      # real_env per-packet rate knob (ae1ca6c);
                                      # hard-capped robot-side at 0.5

    args = tyro.cli(Args)
    assert args.arm in ("left", "right")
    rate = min(max(args.rate, 0.005), 0.5)
    mir = np.ones(7) if args.arm == "left" else ARM_MIRROR_SIGN
    mv_slice = slice(13, 20) if args.arm == "left" else slice(20, 27)
    fz_slice = slice(20, 27) if args.arm == "left" else slice(13, 20)

    tel = NNGSubscriber(f"tcp://{args.robot_ip}:9870")
    tel.start()
    pub = NNGPublisher("tcp://*:9874")

    def jpos():
        d = tel.data or {}
        jp = d.get("joint_pos")
        return np.asarray(jp, dtype=float) if jp is not None and len(jp) >= 31 else None

    print("waiting for telemetry ...")
    t0 = time.monotonic()
    while jpos() is None:
        if time.monotonic() - t0 > 10.0:
            sys.exit("no telemetry on 9870 — is real_env running?")
        time.sleep(0.1)
    start = jpos()
    frozen = list(start[fz_slice])                # other arm: start-time snapshot
    print(f"telemetry OK. {args.arm} arm walks the chain; other arm frozen at "
          f"{np.round(frozen, 3).tolist()}")
    front_err = float(np.max(np.abs(start[mv_slice] - mir * np.asarray(FRONT_L))))
    if front_err > 0.15:
        sys.exit(f"ABORT: {args.arm} arm is {np.degrees(front_err):.1f} deg away from the "
                 f"V83 front station — the audited chain starts THERE, and since 07-31 "
                 f"that is NOT the power-on pose (power-cycling parks the arms at the sim "
                 f"home instead). Bring the arm to the front station first — the operator's "
                 f"staged glide does it, or command it at low torque watching clearances.")

    try:
        for name, pose_l in CHAIN:
            target = list(mir * np.asarray(pose_l))
            cur = jpos()[mv_slice]
            dist = float(np.max(np.abs(cur - np.asarray(target))))
            budget = dist / rate * 1.4 + 5.0
            print(f"\n→ hop to {name}: max joint travel {np.degrees(dist):.0f} deg, "
                  f"ETA ~{dist / rate:.0f}s at {rate} rad/s (budget {budget:.0f}s)")
            hop_t0 = time.monotonic()
            last_print = 0.0
            while True:
                vals = [0.0] * 14
                if args.arm == "left":
                    vals[0:7], vals[7:14] = target, frozen
                else:
                    vals[0:7], vals[7:14] = frozen, target
                pub.publish({"arm_targets": {"joint_pos": [float(v) for v in vals],
                                             "rate": rate}})
                time.sleep(1.0 / RATE_HZ)
                jp = jpos()
                if jp is None:
                    continue
                err = float(np.max(np.abs(jp[mv_slice] - np.asarray(target))))
                el = time.monotonic() - hop_t0
                if el - last_print > 2.0:
                    print(f"   {el:6.1f}s  max joint err {np.degrees(err):6.2f} deg")
                    last_print = el
                if err < args.tol:
                    print(f"   ✓ {name} reached in {el:.1f}s — settling {args.dwell}s")
                    settle_t0 = time.monotonic()
                    while time.monotonic() - settle_t0 < args.dwell:
                        pub.publish({"arm_targets": {"joint_pos": [float(v) for v in vals],
                                                     "rate": rate}})
                        time.sleep(1.0 / RATE_HZ)
                    break
                if el > budget:
                    print(f"   ✗ TIMEOUT at {name} (err {np.degrees(err):.1f} deg) — "
                          f"stopping stream; failsafe will crawl the arm home")
                    return 1
        print("\nCHAIN VALIDATED: front → diag → rear → diag → front, all hops arrived.")
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C — stream stops; failsafe crawls the arm home at 0.025 rad/s")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
