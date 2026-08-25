"""Record a hand-eye calibration session using the GRIPPER BASE tags as the target.

The wrist tag cubes (tag_cube_0/1) are unavailable; the gripper base
(visual_servoing body ``gripper_L_base``, 4×16 mm pads, rigidly mounted = MJCF body
``L_base``) serves instead. The monitor's ``--handeye-objects`` dict CANNOT be
re-keyed from the CLI (tyro freezes bare-dict keys to the defaults), so this wrapper
rewrites ``humanoid_monitor._HANDEYE_OBJECTS`` in memory before invoking the
monitor's own main() — the upstream file stays untouched on disk. It also forces
``--record-handeye`` and puts the gripper bodies on the tracked-bodies list (the
recorder only warns, then records NaN, if the object is not tracked).

Usage (robot hung, arms limp, gimbal UNPOWERED and taped at the desired lock —
do NOT enable the camera motors: the RL policy residual adds a few-mrad creep that
passes the static checks but biases the solve):

    python humanoid_handeye_gripper_record.py            # left gripper / left camera
    python humanoid_handeye_gripper_record.py --right    # also track the right gripper

Then: humanoid_handeye_calibration.py (obj_inits are auto-passed by
humanoid_handeye_gripper_solve — use that instead of calling the solver directly),
and humanoid_handeye_camera_offset.py for the site correction.
"""
from __future__ import annotations

import dataclasses
import sys

import tyro


@dataclasses.dataclass
class Args:
    robot_ip: str = "127.0.0.1"
    tag_camera_host: str = "127.0.0.1"
    handeye_delay: float = 10.0   # countdown before recording starts
    target: str = "gripper"      # 'gripper' (BOTH gripper bases, 2×16mm coplanar pads
    # — needs flip-frame cleaning downstream) or 'cube' (5-face tag_cube_0 strapped
    # RIGIDLY anywhere on the left hand — bigger non-coplanar tags, no flip ambiguity;
    # the mount offset is solved, so placement is free as long as it cannot move)


def main(args: Args) -> None:
    import humanoid_monitor as hm

    if args.target == "cube":
        mapping = {"tag_cube_0": "end_effector_L"}
    elif args.target == "cube-right":     # the one cube, strapped to the RIGHT wrist
        mapping = {"tag_cube_0": "end_effector_R"}
    elif args.target == "gripper":
        # ALWAYS register both grippers: a hand that never shows up costs nothing
        # (the solve pipeline drops <20-valid objects), while forgetting a flag
        # would silently kill two of the four calibration combinations
        mapping = {"gripper_L_base": "L_base", "gripper_R_base": "R_base"}
    else:
        raise SystemExit(f"--target must be 'gripper' or 'cube', got '{args.target}'")
    # Args' default_factory reads this module global at tyro.cli() time, so mutating
    # it here re-keys the recorder without touching the file on disk.
    hm._HANDEYE_OBJECTS.clear()
    hm._HANDEYE_OBJECTS.update(mapping)

    sys.argv = [
        "humanoid_monitor.py",
        "--record-handeye",
        "--robot-ip", args.robot_ip,
        "--tag-camera-host", args.tag_camera_host,
        "--handeye-delay", str(args.handeye_delay),
        "--bodies", *mapping.keys(),
    ]
    print(f"[record] handeye objects: {mapping}")
    hm.main()


if __name__ == "__main__":
    main(tyro.cli(Args))
