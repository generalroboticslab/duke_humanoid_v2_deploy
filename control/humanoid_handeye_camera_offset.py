"""Convert a static hand-eye solution into a camera-SITE pose correction (gimbal era).

humanoid_handeye_calibration.py solves T_base_link←camera as a CONSTANT — valid only
if the gimbal never moved during the recording. The cameras now ride the gimbal, so
that constant is one sample of a moving quantity. What IS constant is the mounting
error: the offset between the model's camera site (cam_left_rgb / cam_right_rgb, FK
of the recorded gimbal angles) and the solved true camera pose:

    T_base←cam_true = FK_site(q̄) @ X_corr      →      X_corr = inv(FK_site(q̄)) @ X_solved

X_corr is expressed in the camera site's local frame. IMPORTANT ASSUMPTION: treating
it as valid at other gimbal angles presumes the physical error is rigid in the
camera/pitch-link frame (bracket/lens error). If the error actually sits UPSTREAM of
the gimbal joints (tower mount on the torso, gimbal servo zero offsets), a site-local
bake is exact only at the locked angle and drifts with gimbal excursion (~error ×
excursion). A single locked session CANNOT distinguish the two placements — so before
baking anything into the model, record TWO sessions at two different locked gimbal
angles (e.g. neutral and pitch −25°) and require the two X_corr to agree via
`--compare a.yaml b.yaml` in this script. Agreement ⇒ site-local is right; large
disagreement ⇒ the error is upstream (likely gimbal zeros → re-zero the gimbal first).

This script:
  1. verifies the recording really had a locked gimbal (else the whole solve is void),
  2. pairing guards: the YAML must be newer than the logs and must contain the logs'
     object names (a stale YAML composed with fresh logs would bake silent garbage),
  3. sanity-checks the recorded extrinsic against the model FK at NEUTRAL gimbal —
     the monitor pins FK(all joints = 0) during --record-handeye — catching a wrong
     port↔site mapping or a model changed since recording (hard abort),
  4. computes X_corr per camera and the NEW site pos/quat in its parent body frame —
     directly pasteable into the MJCF <site> line for the re-export.

Inputs:  calibration/camera_handeye.yaml (solver output) + the recording .npz logs
         (auto-discovers the latest session, same rule as the solver).
Output:  printed report + calibration/camera_site_offset.yaml.

Assumptions: identity root (poses are base_link frame), recording q_pos matches the
model's joint count exactly (31 incl. 4 gimbal) — fewer (legacy 27) or more (future
re-export) is rejected: the gimbal slice would silently misalign. Rejected
alternative: solving X_corr inside the optimizer (gimbal-FK-aware forward model) —
unnecessary while recordings lock the gimbal, and the upstream solver stays untouched.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from humanoid_handeye_calibration import (
    MJCF_XML_PATH,
    _CALIB_DIR,
    _find_latest_logs,
    invert_transform,
)

PORT_TO_SITE = {5555: "cam_left_rgb", 5556: "cam_right_rgb"}
N_GIMBAL_JOINTS = 4          # gimbal joints occupy the LAST 4 slots of the joint vector
GIMBAL_STATIC_TOL_RAD = 0.01  # ~0.57°: any valid frame's gimbal further than this from
                              # the median → the static-extrinsic assumption was violated
SITE_STATIC_TOL_M = 0.003     # camera-site FK excursion across the recording (catches
SITE_STATIC_TOL_DEG = 0.5     # waist/torso motion the gimbal-only check cannot see)
APPROX_MATCH_TOL_M = 0.001    # recorded extrinsic vs model FK at NEUTRAL gimbal: the
APPROX_MATCH_TOL_DEG = 0.1    # monitor computed it from the SAME model with all joints
                              # zero, so anything beyond numerics = wrong port↔site
                              # mapping or a changed model → hard abort
COMPARE_TOL_M = 0.003         # two-session X_corr agreement gate: beyond this the
COMPARE_TOL_DEG = 0.3         # error is NOT site-local (upstream mount/zeros) — do
                              # not bake it into the model


@dataclasses.dataclass
class Args:
    handeye_yaml: str = str(_CALIB_DIR / "camera_handeye.yaml")
    logs: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    # recording .npz files; defaults to the latest session in calibration/
    output: str = str(_CALIB_DIR / "camera_site_offset.yaml")
    compare: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    # two camera_site_offset.yaml files from sessions at DIFFERENT locked gimbal
    # angles: verifies X_corr is truly site-local before it may be baked in


def _fk_site_and_parent(model, data, q: np.ndarray, site: str) -> tuple[np.ndarray, np.ndarray]:
    """FK at joint vector q → (T_base←site, T_base←parent_body), identity root."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"site '{site}' not in {MJCF_XML_PATH}")
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    data.qpos[7:] = 0.0
    data.qpos[7:7 + len(q)] = q
    mujoco.mj_kinematics(model, data)
    T_site = np.eye(4)
    T_site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    T_site[:3, 3] = data.site_xpos[sid]
    bid = model.site_bodyid[sid]
    T_parent = np.eye(4)
    T_parent[:3, :3] = data.xmat[bid].reshape(3, 3)
    T_parent[:3, 3] = data.xpos[bid]
    return T_site, T_parent


def _compare_sessions(path_a: str, path_b: str) -> bool:
    """Two-session consistency gate: X_corr from different locked gimbal angles must
    agree, or the physical error is upstream of the gimbal and MUST NOT be baked
    site-local. Returns True on agreement."""
    with open(path_a) as f:
        a = yaml.safe_load(f)
    with open(path_b) as f:
        b = yaml.safe_load(f)
    ok = True
    common = set(a["cameras"]) & set(b["cameras"])
    if not common:
        raise SystemExit("[offset] compare: no common camera port between the two files")
    for port in sorted(common):
        Ta = np.asarray(a["cameras"][port]["site_offset_local_T"])
        Tb = np.asarray(b["cameras"][port]["site_offset_local_T"])
        ga = np.round(a["cameras"][port]["gimbal_locked_deg"], 1).tolist()
        gb = np.round(b["cameras"][port]["gimbal_locked_deg"], 1).tolist()
        if float(np.max(np.abs(np.asarray(ga) - np.asarray(gb)))) < 5.0:
            print(f"[offset] compare port {port}: WARNING — both sessions locked at "
                  f"nearly the same gimbal pose ({ga} vs {gb}); this comparison "
                  f"cannot distinguish upstream from site-local errors")
        d = invert_transform(Ta) @ Tb
        dp = float(np.linalg.norm(d[:3, 3]))
        dr = float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()))
        verdict = "AGREE" if (dp < COMPARE_TOL_M and dr < COMPARE_TOL_DEG) else "DISAGREE"
        if verdict == "DISAGREE":
            ok = False
        print(f"[offset] compare port {port}: Δ={dp * 1e3:.1f}mm / {dr:.3f}° → {verdict} "
              f"(locks {ga}° vs {gb}°)")
    print("[offset] " + ("X_corr is site-local — SAFE to bake into the model." if ok else
                         "X_corr is NOT site-local (upstream mount/servo-zero error). "
                         "Do NOT bake it — re-zero the gimbal / check the tower mount, "
                         "then re-calibrate."))
    return ok


def main(args: Args) -> dict:
    if args.compare:
        if len(args.compare) != 2:
            raise SystemExit("[offset] --compare takes exactly two camera_site_offset.yaml paths")
        return {"agree": _compare_sessions(args.compare[0], args.compare[1])}

    with open(args.handeye_yaml) as f:
        sol = yaml.safe_load(f)
    if not sol.get("calibration_valid", False):
        raise SystemExit(f"[offset] {args.handeye_yaml} has calibration_valid=false "
                         f"({sol.get('failure_reason', '?')}) — fix the recording/solve first")

    model = mujoco.MjModel.from_xml_path(MJCF_XML_PATH)
    data = mujoco.MjData(model)
    n_joints = model.nq - 7

    log_paths = list(args.logs) if args.logs else _find_latest_logs()
    # Pairing guard 1: a YAML older than the logs means the solver was not re-run on
    # this session — composing it with these logs would bake silent garbage.
    import os
    yaml_mtime = os.path.getmtime(args.handeye_yaml)
    for p in log_paths:
        if os.path.getmtime(str(p)) > yaml_mtime:
            raise SystemExit(f"[offset] {args.handeye_yaml} is OLDER than log {p} — "
                             f"re-run humanoid_handeye_calibration.py on this session first")
    out: dict = {"source_yaml": args.handeye_yaml, "mjcf": MJCF_XML_PATH, "cameras": {}}
    for path in log_paths:
        log = np.load(str(path), allow_pickle=True)
        port = int(log["cam_port"])
        if str(port) not in sol["cameras"]:
            print(f"[offset] port {port}: not in the YAML (skipped by solve?) — skipping")
            continue
        site = PORT_TO_SITE.get(port)
        if site is None:
            raise SystemExit(f"[offset] port {port} has no site mapping in PORT_TO_SITE")

        # Pairing guard 2: the YAML must actually describe THIS session's objects.
        # Overlap (not full coverage) is the test: per-hand SPLIT solves legitimately
        # carry a subset of the log's objects; a truly mismatched session (e.g. a
        # tag-cube YAML against gripper logs) shares none.
        rec_names = {str(n) for n in log["obj_names"]}
        if rec_names.isdisjoint(sol.get("objects", {})):
            raise SystemExit(f"[offset] {args.handeye_yaml} objects "
                             f"{sorted(sol.get('objects', {}))} share nothing with the "
                             f"log's {sorted(rec_names)} — YAML is from a different session")

        q_pos = np.asarray(log["q_pos"], dtype=np.float64)   # (N, n_joints)
        if q_pos.shape[1] != n_joints:
            raise SystemExit(f"[offset] {path}: recording has {q_pos.shape[1]} joints, model "
                             f"has {n_joints} — mismatched model/recording pairing; the "
                             f"gimbal slice would misalign, cannot proceed")
        valid = np.asarray(log["obj_valid"], dtype=bool).any(axis=1) \
            & np.asarray(log["valid_telem"], dtype=bool)
        if valid.sum() < 10:
            # e.g. the rear-facing camera never saw the gripper this session — that
            # port simply isn't calibrated by this recording; don't kill the others.
            print(f"[offset] port {port}: only {int(valid.sum())} valid frames — skipping")
            continue

        # 1. Locked-camera check — the static solve is meaningless otherwise. Named
        # gimbal check first (clearest diagnostic), then the combined site-FK pose
        # excursion, which is the quantity that actually matters and a tighter bound
        # than the per-joint check (the camera site depends ONLY on the 4 gimbal
        # joints — cam_base hangs off base_link, so arm/waist motion cannot move it).
        gimbal = q_pos[valid][:, n_joints - N_GIMBAL_JOINTS:n_joints]
        g_med = np.median(gimbal, axis=0)
        g_dev = float(np.max(np.abs(gimbal - g_med)))
        if g_dev > GIMBAL_STATIC_TOL_RAD:
            raise SystemExit(f"[offset] port {port}: gimbal moved during recording "
                             f"(max dev {np.degrees(g_dev):.2f}° > "
                             f"{np.degrees(GIMBAL_STATIC_TOL_RAD):.2f}°) — the static "
                             f"hand-eye assumption is void; re-record with the gimbal locked")

        # 2. Model FK at the recorded (median) joint vector.
        q_med = np.median(q_pos[valid], axis=0)[:n_joints]
        T_model, T_parent = _fk_site_and_parent(model, data, q_med, site)

        v_idx = np.where(valid)[0]
        probe = v_idx[np.linspace(0, len(v_idx) - 1, min(200, len(v_idx))).astype(int)]
        worst_pos, worst_rot = 0.0, 0.0
        R_med_inv = Rotation.from_matrix(T_model[:3, :3]).inv()
        for i in probe:
            T_i, _ = _fk_site_and_parent(model, data, q_pos[i][:n_joints], site)
            worst_pos = max(worst_pos, float(np.linalg.norm(T_i[:3, 3] - T_model[:3, 3])))
            worst_rot = max(worst_rot, float(np.degrees(
                (R_med_inv * Rotation.from_matrix(T_i[:3, :3])).magnitude())))
        if worst_pos > SITE_STATIC_TOL_M or worst_rot > SITE_STATIC_TOL_DEG:
            raise SystemExit(f"[offset] port {port}: camera site MOVED during recording "
                             f"({worst_pos * 1e3:.1f}mm / {worst_rot:.2f}° from median — "
                             f"waist/torso motion?) — re-record with the upper body still")

        # 3. Sanity: during --record-handeye the monitor PINS the detector extrinsic
        # to FK at ALL JOINTS ZERO (humanoid_monitor.py:909 → freeze pin), NOT at the
        # locked angles — so that is the reference to compare against. Same model,
        # same code path ⇒ must match to numerics; anything more = wrong port↔site
        # mapping or the model changed since recording → the solve inputs are
        # untrustworthy, abort.
        C_approx = np.asarray(log["cam_transform_approx"], dtype=np.float64)
        T_zero, _ = _fk_site_and_parent(model, data, np.zeros(n_joints), site)
        d_pos_approx = float(np.linalg.norm(C_approx[:3, 3] - T_zero[:3, 3]))
        d_rot_approx = float(np.degrees((Rotation.from_matrix(C_approx[:3, :3]).inv()
                                         * Rotation.from_matrix(T_zero[:3, :3])).magnitude()))
        if d_pos_approx > APPROX_MATCH_TOL_M or d_rot_approx > APPROX_MATCH_TOL_DEG:
            raise SystemExit(f"[offset] port {port}: recorded extrinsic vs model FK(neutral) "
                             f"differ by {d_pos_approx * 1e3:.1f}mm / {d_rot_approx:.2f}° — "
                             f"wrong port↔site mapping, or the model changed since the "
                             f"recording. Aborting: the solve inputs are untrustworthy.")

        # 4. The offset itself, in the site's local frame (rides the gimbal).
        X_solved = np.asarray(sol["cameras"][str(port)]["T_base_link_to_camera"],
                              dtype=np.float64)
        X_corr = invert_transform(T_model) @ X_solved
        d_pos = X_corr[:3, 3]
        rot = Rotation.from_matrix(X_corr[:3, :3])
        d_rot_deg = float(np.degrees(rot.magnitude()))

        # 5. Pasteable new site pose in its PARENT body frame (for the MJCF re-export).
        T_parent_site_new = invert_transform(T_parent) @ X_solved
        new_pos = T_parent_site_new[:3, 3]
        new_quat_wxyz = Rotation.from_matrix(T_parent_site_new[:3, :3]).as_quat(scalar_first=True)

        print(f"[offset] port {port} ({site}):")
        print(f"  gimbal locked at {np.round(np.degrees(g_med), 2).tolist()}° "
              f"(max dev {np.degrees(g_dev):.3f}°), {int(valid.sum())} valid frames")
        print(f"  site pose offset: Δpos={np.round(d_pos * 1e3, 1).tolist()}mm "
              f"|Δpos|={np.linalg.norm(d_pos) * 1e3:.1f}mm  Δrot={d_rot_deg:.3f}°")
        print(f"  → at 0.5 m range this explains ≈"
              f"{(np.linalg.norm(d_pos) + 0.5 * rot.magnitude()) * 1e3:.0f}mm of world error")
        print(f"  MJCF fix: <site name=\"{site}\" pos=\"{' '.join(f'{v:.5f}' for v in new_pos)}\" "
              f"quat=\"{' '.join(f'{v:.6f}' for v in new_quat_wxyz)}\" .../>")

        out["cameras"][port] = {
            "site": site,
            "gimbal_locked_deg": np.degrees(g_med).tolist(),
            "n_valid_frames": int(valid.sum()),
            "site_offset_local_T": X_corr.tolist(),
            "site_offset_pos_mm": (d_pos * 1e3).tolist(),
            "site_offset_rot_deg": d_rot_deg,
            "new_site_pos_in_parent": new_pos.tolist(),
            "new_site_quat_wxyz_in_parent": new_quat_wxyz.tolist(),
        }

    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f"[offset] Written to {args.output}")
    return out


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
