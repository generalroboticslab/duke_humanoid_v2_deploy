"""Solve a gripper-tag hand-eye session: wrap → filter → solve → iteratively clean.

All preprocessing happens on npz COPIES (suffix `_prep`); the upstream solver
(humanoid_handeye_calibration.py) is invoked unchanged. Pipeline:

  1. WRAP: fold all joint angles into [−π, π]. Wrist encoders can report outside
     that range (left_wrist_3 measured +3.73 rad); FK is 2π-periodic so the data is
     fine, but the solver's joint-limit gate would otherwise reject every frame.
  2. FILTER: drop logs with no valid frames (the other camera's port records NaNs)
     and objects with fewer than MIN_OBJ_VALID valid frames (a hand that never
     entered this camera's view would trip the solver's per-object validity gate).
     --objects restricts further — used for the per-hand SPLIT solves of the
     four-combination cross-validation (same eye solved via each hand separately).
  3. SOLVE with zero obj-inits for the gripper bases (they mount essentially AT
     their MJCF bodies; the tyro CLI cannot re-key the solver's obj_inits dict).
  4. CLEAN (×--clean-iters): the 2×16 mm coplanar gripper pads carry a planar-PnP
     flip ambiguity — a contaminated minority of frames with ~20-60° rotation
     error that biases the solve. Compute per-frame residuals against the current
     solution, invalidate frames beyond CLEAN_ROT_DEG / CLEAN_TRANS_MM, re-solve;
     stop early when the kept set is stable. (Cube targets need 0 iterations;
     gripper targets typically converge in 2.)
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from humanoid_handeye_calibration import (
    MJCF_XML_PATH,
    CalibArgs,
    CameraFKSolver,
    _CALIB_DIR,
    _find_latest_logs,
    invert_transform,
    make_transform,
    quat_wxyz_to_xyzw,
    main as calib_main,
)

MIN_OBJ_VALID = 20      # solver's own per-object validity floor
MIN_LOG_VALID = 100     # a log below this is a glimpse (off-target camera), not a session
CLEAN_ROT_DEG = 4.0     # frame residual beyond this vs the current solution → dropped
CLEAN_TRANS_MM = 8.0
KEPT_WARN_FRAC = 0.55   # cleaning kept less than this of the original frames → the
                        # majority may be the FLIP cluster (see PRIOR gate below)
# PRIOR GATE: a genuine mounting correction is millimetres and a couple of degrees;
# a flip-cluster capture (self-consistent flips >50% of frames can hijack the solve
# AND pass every residual gate) lands 20-60° from the model. Solved X further than
# this from the model FK at the session's own locked gimbal pose fails the solve.
PRIOR_MAX_MM = 30.0
PRIOR_MAX_DEG = 8.0
PORT_TO_SITE = {5555: "cam_left_rgb", 5556: "cam_right_rgb"}
OBJ_FIELDS = ("obj_names", "obj_links", "obj_pos", "obj_quat_wxyz",
              "obj_n_inliers", "obj_valid")


def _save_atomic(path: Path, log: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **log)
    os.replace(tmp, path)


@dataclasses.dataclass
class Args:
    logs: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    # recording .npz files; defaults to the most recent session in calibration/
    objects: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    # restrict to these object names (per-hand split solves); empty = keep all
    output: str = str(_CALIB_DIR / "camera_handeye.yaml")
    subsample: int = 2
    clean_iters: int = 3


def _load(path: str) -> dict:
    raw = dict(np.load(path, allow_pickle=True))
    for k in list(raw):
        if raw[k].ndim == 0:
            raw[k] = raw[k].item()
    return raw


def _prepare(path: str, objects: tuple[str, ...]) -> Path | None:
    """Wrap + filter one log; returns the _prep.npz path or None if unusable."""
    log = _load(path)
    q = np.asarray(log["q_pos"], dtype=np.float64)
    log["q_pos"] = np.mod(q + np.pi, 2 * np.pi) - np.pi

    names = [str(n) for n in log["obj_names"]]
    valid = np.asarray(log["obj_valid"], dtype=bool)
    keep_cols = []
    for m, name in enumerate(names):
        n_valid = int(valid[:, m].sum())
        if objects and name not in objects:
            print(f"[solve] {Path(path).name}: {name} excluded by --objects")
        elif n_valid < MIN_OBJ_VALID:
            print(f"[solve] {Path(path).name}: {name} has {n_valid} valid frames "
                  f"(<{MIN_OBJ_VALID}) — dropped")
        else:
            keep_cols.append(m)
    if not keep_cols:
        print(f"[solve] {Path(path).name}: no usable objects — log skipped")
        return None
    total_valid = int(valid[:, keep_cols].sum())
    if total_valid < MIN_LOG_VALID:
        # e.g. the off-target camera glimpsed a gripper for a few dozen near-static
        # frames: enough to sneak a weakly-constrained X into the shared YAML,
        # nowhere near enough to constrain it. A real session has hundreds.
        print(f"[solve] {Path(path).name}: only {total_valid} valid frames total "
              f"(<{MIN_LOG_VALID}) — glimpse, not a session; log skipped")
        return None
    for f in OBJ_FIELDS:
        arr = np.asarray(log[f])
        log[f] = arr[keep_cols] if f in ("obj_names", "obj_links") else arr[:, keep_cols]
    # frozen copy of the pre-cleaning validity: every clean pass re-judges ALL of
    # these frames against the CURRENT solution, so frames wrongly dropped by an
    # early (biased) solution can return once the solution improves
    log["obj_valid_orig"] = np.asarray(log["obj_valid"], dtype=bool).copy()

    out = Path(path).with_name(Path(path).stem + "_prep.npz")
    _save_atomic(out, log)
    return out


def _clean_pass(prep: Path, yaml_path: str) -> int:
    """Re-judge ALL originally-valid frames against the current solution: frames
    beyond the flip thresholds are invalidated, previously-dropped frames that now
    fit are restored. Returns how many frames changed state (0 = converged)."""
    import yaml as _yaml
    sol = _yaml.safe_load(open(yaml_path))
    log = _load(str(prep))
    port = str(int(log["cam_port"]))
    if port not in sol["cameras"]:
        return 0
    X = np.asarray(sol["cameras"][port]["T_base_link_to_camera"])
    invX = invert_transform(X)
    invC = invert_transform(np.asarray(log["cam_transform_approx"], dtype=np.float64))
    q = np.asarray(log["q_pos"], dtype=np.float64)
    names = [str(n) for n in log["obj_names"]]
    links = [str(n) for n in log["obj_links"]]
    fk = CameraFKSolver(obj_links=list(set(links)))
    if q.shape[1] < fk.n_joints:      # legacy pre-gimbal recordings: zero-pad, same
        q = np.pad(q, ((0, 0), (0, fk.n_joints - q.shape[1])))  # rule as the solver
    telem = np.asarray(log["valid_telem"], dtype=bool)
    orig = np.asarray(log["obj_valid_orig"], dtype=bool)
    prev = np.asarray(log["obj_valid"], dtype=bool)
    ov = np.zeros_like(prev)
    for m, (name, link) in enumerate(zip(names, links)):
        if name not in sol.get("objects", {}):
            ov[:, m] = prev[:, m]
            continue
        Y = np.asarray(sol["objects"][name]["T_obj_link_to_object"])
        idx = np.where(orig[:, m] & telem)[0]
        if len(idx) == 0:
            continue
        T_fk = fk.compute_fk_batch(q[idx], link)
        rot = np.zeros(len(idx))
        trans = np.zeros(len(idx))
        for k, i in enumerate(idx):
            R = Rotation.from_quat(quat_wxyz_to_xyzw(
                np.asarray(log["obj_quat_wxyz"])[i, m])).as_matrix()
            meas = invC @ make_transform(R, np.asarray(log["obj_pos"])[i, m])
            d = invert_transform(invX @ T_fk[k] @ Y) @ meas
            rot[k] = np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude())
            trans[k] = np.linalg.norm(d[:3, 3]) * 1e3
        # ADAPTIVE thresholds: an early solution may be biased by the flip population
        # itself, pushing even good frames past the fixed floor — a fixed gate can
        # then drop EVERYTHING. Trimming at 1.5×median always keeps the better half,
        # and tightens to the floor as the re-solves converge on the clean cluster.
        thr_rot = max(CLEAN_ROT_DEG, 1.5 * float(np.median(rot)))
        thr_trans = max(CLEAN_TRANS_MM, 1.5 * float(np.median(trans)))
        ov[idx, m] = (rot <= thr_rot) & (trans <= thr_trans)
    changed = int((ov != prev).sum())
    for m, name in enumerate(names):
        n_orig = int((orig[:, m] & telem).sum())
        n_kept = int(ov[:, m].sum())
        if n_orig > 0:
            frac = n_kept / n_orig
            mark = ("  ⚠ kept <" + f"{KEPT_WARN_FRAC:.0%}" +
                    " — the DROPPED majority may be the real data (flip-cluster "
                    "capture); re-record with more viewing-angle diversity"
                    ) if frac < KEPT_WARN_FRAC else ""
            print(f"[solve]   {name}: kept {n_kept}/{n_orig} ({frac:.0%}){mark}")
    log["obj_valid"] = ov
    _save_atomic(prep, log)
    return changed


def _prior_gate(preps: list[Path], yaml_path: str) -> None:
    """Solved X must sit CLOSE to the model's prediction at the session's own locked
    gimbal pose. A residual-perfect solve can still be a flip-cluster capture 20-60°
    from reality (self-consistent flips >50% hijack the fit and pass every internal
    gate); the model prior is the only independent referee. Annotates the YAML with
    the lock angles + prior deltas, and invalidates it on failure."""
    import mujoco
    import yaml as _yaml
    sol = _yaml.safe_load(open(yaml_path))
    model = mujoco.MjModel.from_xml_path(MJCF_XML_PATH)
    data = mujoco.MjData(model)
    n_j = model.nq - 7
    failures = []
    for p in preps:
        log = _load(str(p))
        port = int(log["cam_port"])
        if str(port) not in sol["cameras"] or port not in PORT_TO_SITE:
            continue
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PORT_TO_SITE[port])
        valid = np.asarray(log["obj_valid"], bool).any(axis=1) & \
            np.asarray(log["valid_telem"], bool)
        q_med = np.median(np.asarray(log["q_pos"], dtype=np.float64)[valid], axis=0)
        q_full = np.zeros(n_j)
        q_full[:min(len(q_med), n_j)] = q_med[:n_j]
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        data.qpos[7:] = q_full
        mujoco.mj_kinematics(model, data)
        T_model = np.eye(4)
        T_model[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        T_model[:3, 3] = data.site_xpos[sid]
        X = np.asarray(sol["cameras"][str(port)]["T_base_link_to_camera"])
        d = invert_transform(T_model) @ X
        dp = float(np.linalg.norm(d[:3, 3]) * 1e3)
        dr = float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()))
        sol["cameras"][str(port)]["gimbal_locked_deg"] = \
            np.degrees(q_full[n_j - 4:]).round(2).tolist()
        sol["cameras"][str(port)]["prior_delta"] = {"pos_mm": round(dp, 1),
                                                    "rot_deg": round(dr, 2)}
        status = "ok" if (dp <= PRIOR_MAX_MM and dr <= PRIOR_MAX_DEG) else "FAIL"
        print(f"[solve] prior gate cam {port}: solved-vs-model Δ={dp:.1f}mm/{dr:.2f}° "
              f"(limit {PRIOR_MAX_MM:.0f}mm/{PRIOR_MAX_DEG:.0f}°) → {status}")
        if status == "FAIL":
            failures.append(f"cam{port} Δ={dp:.0f}mm/{dr:.1f}°")
    if failures:
        sol["calibration_valid"] = False
        sol["failure_reason"] = ("prior gate: solved extrinsic implausibly far from "
                                 "the model (" + ", ".join(failures) + ") — likely "
                                 "flip-cluster capture; re-record")
    with open(yaml_path, "w") as f:
        _yaml.dump(sol, f, default_flow_style=False, sort_keys=False)
    if failures:
        raise SystemExit(f"[solve] PRIOR GATE FAILED: {sol['failure_reason']}")


def main(args: Args) -> None:
    paths = list(args.logs) if args.logs else [str(p) for p in _find_latest_logs()]
    # never ingest our own derivatives: auto-discovery groups *_prep.npz with the raw
    # logs (same _port prefix), which would double-feed the solver and grow
    # _prep_prep chains on every re-run
    paths = [p for p in paths if "_prep" not in Path(p).stem]
    preps = [p for p in (_prepare(pp, args.objects) for pp in paths) if p is not None]
    if not preps:
        raise SystemExit("[solve] no usable logs")

    def solve():
        calib_main(CalibArgs(
            logs=tuple(str(p) for p in preps),
            obj_inits={"gripper_L_base": [0.0, 0.0, 0.0],
                       "gripper_R_base": [0.0, 0.0, 0.0]},
            subsample=args.subsample,
            output=args.output,
        ))

    solve()
    for it in range(args.clean_iters):
        changed = sum(_clean_pass(p, args.output) for p in preps)
        print(f"[solve] clean iteration {it + 1}: {changed} frames changed state")
        if changed == 0:
            break
        solve()
    _prior_gate(preps, args.output)


if __name__ == "__main__":
    main(tyro.cli(Args))
