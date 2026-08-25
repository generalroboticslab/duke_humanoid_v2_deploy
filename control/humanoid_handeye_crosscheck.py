"""Cross-validate hand-eye solutions from the four-combination protocol.

The upstream protocol solves each eye via each hand (4 combinations). Redundancy gives
closure checks that a single session cannot:

  · the SAME camera solved through two different hands must yield the SAME
    extrinsic X (else one hand's arm chain / gripper tags are suspect);
  · the SAME gripper mount Y solved through two different cameras must agree
    (else one camera's solve is suspect).

This script takes any number of solver-output YAMLs (typically the per-hand split
solves and the joint solves) and reports every pairwise agreement it can form,
with AGREE/DISAGREE verdicts. It only reads and compares — the authoritative
numbers to bake still come from the joint solves.

Usage:
    python humanoid_handeye_crosscheck.py L_via_lefthand.yaml L_via_righthand.yaml
    python humanoid_handeye_crosscheck.py sessionL.yaml sessionR.yaml   # Y across eyes
"""
from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import tyro
import yaml
from scipy.spatial.transform import Rotation

from humanoid_handeye_calibration import invert_transform

X_TOL_MM, X_TOL_DEG = 5.0, 0.8   # camera-extrinsic agreement gate
Y_TOL_MM, Y_TOL_DEG = 5.0, 0.8   # gripper-mount agreement gate


@dataclasses.dataclass
class Args:
    yamls: tuple[str, ...]
    # two or more solver-output YAMLs to cross-compare


def _delta(Ta, Tb) -> tuple[float, float]:
    d = invert_transform(np.asarray(Ta)) @ np.asarray(Tb)
    return (float(np.linalg.norm(d[:3, 3]) * 1e3),
            float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude())))


def main(args: Args) -> bool:
    if len(args.yamls) < 2:
        raise SystemExit("[crosscheck] need at least two YAMLs")
    sols = {}
    ok = True
    for p in args.yamls:
        s = yaml.safe_load(open(p))
        if not s.get("calibration_valid", False):
            # a failed solve has nothing trustworthy to contribute, and pretending
            # it closed a loop would be worse than checking nothing
            print(f"[crosscheck] {p}: calibration_valid=false "
                  f"({s.get('failure_reason', '?')}) — EXCLUDED, verdict forced FAIL")
            ok = False
            continue
        sols[Path(p).stem] = s

    n_pairs = 0
    for (na, a), (nb, b) in itertools.combinations(sols.items(), 2):
        for port in sorted(set(a["cameras"]) & set(b["cameras"])):
            dp, dr = _delta(a["cameras"][port]["T_base_link_to_camera"],
                            b["cameras"][port]["T_base_link_to_camera"])
            ga = a["cameras"][port].get("gimbal_locked_deg")
            gb = b["cameras"][port].get("gimbal_locked_deg")
            if ga is not None and gb is not None and \
                    float(np.max(np.abs(np.asarray(ga) - np.asarray(gb)))) > 2.0:
                # different locks: this Δ measures the KNOWN gimbal-chain flex, not
                # hand-chain consistency — report it, but outside the verdict
                print(f"[crosscheck] camera {port}:  {na} vs {nb}: "
                      f"Δ={dp:.1f}mm/{dr:.2f}° [CROSS-LOCK {ga}° vs {gb}° — measures "
                      f"gimbal flex, excluded from the verdict]")
                continue
            n_pairs += 1
            verdict = "AGREE" if (dp < X_TOL_MM and dr < X_TOL_DEG) else "DISAGREE"
            ok &= verdict == "AGREE"
            print(f"[crosscheck] camera {port}:  {na} vs {nb}: "
                  f"Δ={dp:.1f}mm/{dr:.2f}° → {verdict}")
        for obj in sorted(set(a.get("objects", {})) & set(b.get("objects", {}))):
            dp, dr = _delta(a["objects"][obj]["T_obj_link_to_object"],
                            b["objects"][obj]["T_obj_link_to_object"])
            n_pairs += 1
            verdict = "AGREE" if (dp < Y_TOL_MM and dr < Y_TOL_DEG) else "DISAGREE"
            ok &= verdict == "AGREE"
            print(f"[crosscheck] mount  {obj}:  {na} vs {nb}: "
                  f"Δ={dp:.1f}mm/{dr:.2f}° → {verdict}")
    if n_pairs == 0:
        print("[crosscheck] NOTHING COMPARED — the YAMLs share no camera port or "
              "object (or all were invalid). No conclusion can be drawn.")
        return False
    print("[crosscheck] " + ("ALL LOOPS CLOSE — solutions are mutually consistent."
                             if ok else "at least one loop FAILS — do not bake; "
                             "the disagreeing pair localizes the suspect chain."))
    return ok


if __name__ == "__main__":
    main(tyro.cli(Args))
