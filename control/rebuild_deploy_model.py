"""Rebuild the deploy `robot.xml` from the upstream sim source, re-applying our three local layers.

WHY THIS EXISTS
  `humanoid_model.MJCF_MODEL_PATH` points at a file inside legged_env_v2's
  `mj_envs/deploy/runs/<task>/`, which is GITIGNORED. That file is a build
  artifact of the sim-side policy export (`mj_envs/utils/export_util.py`
  serialises `cfg.spec_fn()` there), so it is not versioned anywhere and it does
  not update when upstream updates the CAD. On 2026-07-28 the wrist was redesigned
  upstream and the deploy model was 9 days stale; rebuilding it by hand took a
  day and produced THREE local modifications that exist in no repository:

    1. CABLE COMPENSATION (+967 g). The exported model carries only Fusion CAD
       masses. Someone once added per-link wiring/harness mass directly into the
       exported XML -- it is in NO committed source (verified: the values appear
       nowhere in either repo, and no code adds them), so it had to be reverse-
       engineered from the previous deploy model and transplanted. Feet get
       zero; base_link/waist carry the most; elbow is exactly +50 g. Recovered
       as full rigid-body deltas (mass, first moment, inertia about the body
       origin) rather than a mass scale, because the original also moved COM
       and inertia (elbow: mass x1.083 but inertia x1.14-1.31).
    2. HAND-EYE CALIBRATION. One line: the `cam_left_rgb` site pose from the
       07-12 campaign. The head camera mount is untouched by the wrist redesign,
       so this calibration survives -- it just has to be re-applied.
    3. WRIST_1 REF. Upstream chose "zero = working pose" (option (1) of the
       two it offered),
       which moves the model's wrist_1 zero 90 deg away from the zero the real
       motors are calibrated to. The lab decided the hardware convention is the
       sane one, so we pin it back with MuJoCo `ref` -- the same mechanism the
       pre-redesign model used on wrist_3. INVARIANT: encoder 0 == model qpos 0.
       Signs were fixed by eye in the viewer (a jaw-direction test cannot
       distinguish +90 from -90; both give front-back jaws).
    4. LEFT WRIST_2 AXIS. Upstream also turned left_wrist_2's axis to match the
       right arm's; the physical motor was never rewired. Caught by the hardware
       gravity-torque check, not by eye: the left arm was off by 1.15 N.m with
       wrist_2's term sign-inverted while the right arm matched at 0.18 N.m.
       Flipping the axis takes the left residual to 0.22 N.m AND restores the
       plain `left = -right` mirror on all seven joints -- the convention every
       stored posture in this repo was authored in.

  Layers 3 and 4 are the same failure: upstream changed a convention, the robot
  did not. Both are worth raising upstream; until then they live here.

  Without this script every future upstream model update silently drops all
  three and the robot runs on a model that disagrees with its own hardware.

USAGE
    python rebuild_deploy_model.py                 # rebuild in place (backs up)
    python rebuild_deploy_model.py --dry-run       # build + verify, write nothing
    python rebuild_deploy_model.py --out /tmp/x.xml

  Needs the sim repo checked out at the revision you want (default
  <legged_env_v2>). mjlab is NOT required: the sim's constants
  module only imports four mjlab config dataclasses, which this script stubs.

AFTER RUNNING
  Re-verify the invariant on hardware before powering the arms:
  drive the motors to zero (`humanoid_config.py --zero`) and confirm both
  grippers' jaws open FRONT-BACK, matching this model at qpos=0.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from humanoid_site import LEGGED_ENV_ROOT as SIM_REPO
CABLE_TABLE = pathlib.Path(__file__).with_name("deploy_model_cable_table.json")

# Export options the deploy run was built with. Verified by reproducing the
# 2026-07-19 artifact bit-for-bit on every geometric field.
SPEC_OPTS = dict(head_camera="actuated", end_effector="welded",
                 hand="parallel_gripper")

# 07-12 hand-eye campaign, ported from the v83 deploy. Head-mounted, so the
# 07-28 wrist redesign does not invalidate it.
CAM_SITE_CALIBRATED = ('<site name="cam_left_rgb" pos="-0.03139 -0.01672 0.04084" '
                       'quat="0.486502 0.505843 0.504603 -0.502807"/>')

# See docstring layer 3. Left/right are mirrored; signs confirmed visually.
WRIST1_REF = {"left": -1.5707963267948966, "right": 1.5707963267948966}

# Layer 4 (see docstring). Upstream points left_wrist_2's axis the SAME way as
# the right arm's; the physical motor still turns the other way. Flip it back.
WRIST2_AXIS_FLIP = {"left_wrist_2_joint": ("-1 0 0", "1 0 0")}

_MJLAB_STUB = {
    "mjlab/__init__.py": "",
    "mjlab/actuator.py": '''
from dataclasses import dataclass
from typing import Any
@dataclass
class BuiltinPositionActuatorCfg:
    target_names_expr: Any = ()
    stiffness: Any = 0.0
    damping: Any = 0.0
    effort_limit: Any = 0.0
    armature: Any = 0.0
    frictionloss: Any = 0.0
    delay_min_lag: int = 0
    delay_max_lag: int = 0
    delay_update_period: int = 0
''',
    "mjlab/entity.py": '''
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
@dataclass
class EntityArticulationInfoCfg:
    actuators: tuple = ()
    soft_joint_pos_limit_factor: float = 1.0
@dataclass
class EntityCfg:
    init_state: Any = None
    collisions: tuple = ()
    spec_fn: Optional[Callable] = None
    articulation: Any = None
    @dataclass
    class InitialStateCfg:
        pos: Any = None
        rot: Any = None
        joint_pos: Any = field(default_factory=dict)
        joint_vel: Any = field(default_factory=dict)
class Entity:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.spec = cfg.spec_fn() if cfg is not None and getattr(cfg, "spec_fn", None) else None
''',
    "mjlab/utils/__init__.py": "",
    "mjlab/utils/spec_config.py": '''
from dataclasses import dataclass
from typing import Any
@dataclass
class CollisionCfg:
    geom_names_expr: tuple = ()
    contype: int = 0
    conaffinity: int = 0
    condim: int = 3
    priority: int = 0
    friction: Any = None
    solref: Any = None
    solimp: Any = None
    disable_other_geoms: bool = True
''',
}


def build_cad_xml(out_path: pathlib.Path) -> str:
    """Serialise the sim's robot spec, with mesh paths made relative to `out_path`.

    Mirrors export_util.py's robot.xml step: mjlab is stubbed (the constants
    module only needs its config dataclasses), and `get_spec()` is called
    directly rather than through a trained env, so no GPU/IsaacGym is involved.
    """
    stub = pathlib.Path(tempfile.mkdtemp(prefix="mjlab_stub_"))
    for rel, src in _MJLAB_STUB.items():
        p = stub / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    zoo = SIM_REPO / "mj_envs/asset_zoo/humanoid_v21"
    code = (
        "import sys, os, re, pathlib\n"
        "import humanoid_v21_constants as C\n"
        f"spec = C.get_spec(**{SPEC_OPTS!r})\n"
        "xml = spec.to_xml()\n"
        "meshdir = C.HUMANOID_V21_XML.parent / 'meshes'\n"
        f"outdir = pathlib.Path({str(out_path.parent)!r})\n"
        "xml = re.sub(r'meshdir=\"[^\"]*\"', 'meshdir=\"%s\"' % os.path.relpath(meshdir, outdir), xml)\n"
        "xml = re.sub(r'(<texture[^>]*\\bfile=\")(/[^\"]*)(\")',\n"
        "             lambda m: m.group(1)+os.path.relpath(m.group(2), outdir)+m.group(3), xml)\n"
        "xml = re.sub(r'file=\"(/[^\"]*)\"',\n"
        "             lambda m: 'file=\"%s\"' % os.path.relpath(m.group(1), meshdir), xml)\n"
        "sys.stdout.write(xml)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub), str(SIM_REPO), str(SIM_REPO / "mj_envs"), str(zoo)])
    r = subprocess.run([sys.executable, "-c", code], cwd=SIM_REPO,
                       env=env, capture_output=True, text=True)
    shutil.rmtree(stub, ignore_errors=True)
    if r.returncode != 0:
        raise SystemExit(f"spec build failed:\n{r.stderr[-2000:]}")
    return r.stdout


def _tensor_about_origin(m, i):
    """Full inertia tensor of body i about its BODY FRAME origin."""
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[i])
    R = R.reshape(3, 3)
    c, mass = m.body_ipos[i], m.body_mass[i]
    return (R @ np.diag(m.body_inertia[i]) @ R.T
            + mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c)))


def extract_cable_table(with_cable: str, cad_only: str) -> dict:
    """Recover the cable layer as per-body rigid-body deltas (see docstring)."""
    import mujoco
    a = mujoco.MjModel.from_xml_path(with_cable)
    b = mujoco.MjModel.from_xml_path(cad_only)
    name = lambda m, i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    assert [name(a, i) for i in range(a.nbody)] == [name(b, i) for i in range(b.nbody)], \
        "body sets differ — the two models are not the same robot"
    out = {}
    for i in range(a.nbody):
        dm = float(a.body_mass[i] - b.body_mass[i])
        dS = a.body_mass[i] * a.body_ipos[i] - b.body_mass[i] * b.body_ipos[i]
        dI = _tensor_about_origin(a, i) - _tensor_about_origin(b, i)
        if abs(dm) < 1e-9 and np.abs(dS).max() < 1e-9 and np.abs(dI).max() < 1e-12:
            continue
        out[name(a, i)] = {"mass": dm, "first_moment": dS.tolist(),
                           "inertia_about_origin": dI.tolist()}
    return out


def apply_cable(xml: str, table: dict, cad_path: pathlib.Path) -> str:
    """Add the cable rigid-body deltas onto the CAD inertials.

    Done on the additive invariants (mass, first moment, inertia about the body
    origin), then converted back to MJCF's COM-frame form and written as
    `fullinertia` so no eigen-decomposition round-trip is needed.
    """
    import mujoco
    import xml.etree.ElementTree as ET
    m = mujoco.MjModel.from_xml_path(str(cad_path))
    new = {}
    for i in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if n not in table:
            continue
        c = table[n]
        mass = m.body_mass[i] + c["mass"]
        S = m.body_mass[i] * m.body_ipos[i] + np.asarray(c["first_moment"])
        Io = _tensor_about_origin(m, i) + np.asarray(c["inertia_about_origin"])
        com = S / mass
        Ic = Io - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
        new[n] = (mass, com, Ic)
    root = ET.fromstring(xml)
    patched = 0
    for body in root.iter("body"):
        n = body.get("name")
        if n not in new:
            continue
        mass, com, Ic = new[n]
        el = body.find("inertial")
        assert el is not None, n
        el.set("mass", f"{mass:.9g}")
        el.set("pos", " ".join(f"{v:.9g}" for v in com))
        for k in ("diaginertia", "quat", "fullinertia"):
            el.attrib.pop(k, None)
        el.set("fullinertia", " ".join(
            f"{v:.9g}" for v in (Ic[0, 0], Ic[1, 1], Ic[2, 2],
                                 Ic[0, 1], Ic[0, 2], Ic[1, 2])))
        patched += 1
    assert patched == len(table), f"patched {patched} of {len(table)} cable bodies"
    return ET.tostring(root, encoding="unicode")


def apply_calibration(xml: str) -> str:
    nominal = re.search(r'<site name="cam_left_rgb"[^/]*/>', xml)
    assert nominal, "cam_left_rgb site not found"
    return xml.replace(nominal.group(0), CAM_SITE_CALIBRATED + "\n<!-- hand-eye "
                       "calibrated 07-12; re-applied by rebuild_deploy_model.py -->", 1)


def apply_wrist2_axis_flip(xml: str) -> str:
    """Point left_wrist_2's joint axis the way the physical motor actually turns.

    Measured on hardware 2026-07-28 with both arms held at the mirrored
    default_pose under gravity compensation: with upstream's axis the left arm's
    predicted gravity torque was off by up to 1.15 N.m and the wrist_2 term had
    the WRONG SIGN (predicted -0.16, measured -0.89), while the right arm matched
    at 0.18 N.m. Flipping this one axis drops the left residual to 0.22 N.m and
    fixes the sign. It also restores the plain `left = -right` mirror across all
    seven joints (verified with the sim's own arm_mirror_signs probe), which is
    the convention the deploy stack and every stored posture were authored in.
    """
    for name, (want, other) in WRIST2_AXIS_FLIP.items():
        pat = re.compile(rf'(<joint name="{name}"[^/]*?axis=")([^"]*)(")')
        m = pat.search(xml)
        assert m, f"{name} not found"
        assert m.group(2) == want, \
            f"{name} axis is {m.group(2)!r}, expected upstream's {want!r} — " \
            "upstream may have fixed this; re-measure before assuming"
        xml = pat.sub(rf'\g<1>{other}\g<3>', xml, count=1)
    return xml


def apply_wrist1_ref(xml: str) -> str:
    for arm, ref in WRIST1_REF.items():
        pat = re.compile(rf'(<joint name="{arm}_wrist_1_joint"[^/]*?)(\s*/>)')
        m = pat.search(xml)
        assert m, f"{arm}_wrist_1_joint not found"
        assert 'ref=' not in m.group(1), f"{arm}_wrist_1_joint already has a ref"
        xml = pat.sub(rf'\1 ref="{ref!r}"\2'.replace("!r", ""), xml, count=1)
    return xml


def main() -> None:
    global SIM_REPO
    from humanoid_model import MJCF_MODEL_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=MJCF_MODEL_PATH)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sim-repo", default=str(SIM_REPO))
    args = p.parse_args()

    SIM_REPO = pathlib.Path(args.sim_repo)
    out = pathlib.Path(args.out)
    table = json.loads(CABLE_TABLE.read_text())
    print(f"sim source : {SIM_REPO}")
    print(f"cable table: {len(table)} bodies, "
          f"{sum(v['mass'] for v in table.values()) * 1000:.1f} g")

    # Staging files live IN the output directory, not a subdirectory: the mesh
    # paths this script writes are relative to that directory, so a temp file
    # one level deeper would resolve them wrong.
    cad = out.parent / f".rebuild_cad_{os.getpid()}.xml"
    staged = out.parent / f".rebuild_staged_{os.getpid()}.xml"
    try:
        cad.write_text(build_cad_xml(cad))
        xml = apply_cable(cad.read_text(), table, cad)
        xml = apply_calibration(xml)
        xml = apply_wrist1_ref(xml)
        xml = apply_wrist2_axis_flip(xml)
        staged.write_text(xml)

        import mujoco
        m = mujoco.MjModel.from_xml_path(str(staged))
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        d.qpos[7:7 + 31] = 0.0                   # encoder zero
        mujoco.mj_forward(m, d)
        jaw = {}
        for side in ("L", "R"):
            g = {}
            for k in range(m.ngeom):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k)
                if n and (f"{side}_left_rack" in n or f"{side}_right_rack" in n):
                    g.setdefault("l" if f"{side}_left" in n else "r", []).append(d.geom_xpos[k])
            v = np.mean(g["r"], 0) - np.mean(g["l"], 0)
            jaw[side] = ["front-back", "left-right", "up-down"][int(np.argmax(np.abs(v)))]
        print(f"built      : njnt={m.njnt} nq={m.nq} mass={m.body_mass.sum():.4f} kg")
        print(f"at encoder zero: jaws L={jaw['L']} R={jaw['R']}")
        assert jaw["L"] == jaw["R"] == "front-back", \
            "INVARIANT BROKEN: model qpos 0 no longer matches the hardware zero"

        if args.dry_run:
            print("dry-run: nothing written")
            return
        if out.exists():
            bak = out.with_suffix(out.suffix + ".prev")
            shutil.copy2(out, bak)
            print(f"backed up  : {bak}")
        shutil.copy2(staged, out)
        print(f"wrote      : {out}")
    finally:
        cad.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
