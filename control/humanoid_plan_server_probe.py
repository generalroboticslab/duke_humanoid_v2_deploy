"""Is the cuRobo plan server healthy? Answer without touching the robot.

WHY THIS EXISTS. Hardware 2026-08-05: two rounds parked the left hand 147 and
104 mm short of the cube with the arm reporting it had executed everything it
was asked. Three probes settled it without a single motor command — a
perfect-executor run converged to 0.1 mm (the cost field was innocent), a
governed run at the hardware's measured 0.22 rad/s also converged (the
protocol was innocent), and the server log showed its re-anchor guard firing
every ~12 ticks against the lead the client is DESIGNED to grant. That is
this file: the probe that separates "the server is wrong" from "the robot is
wrong" before anyone stands the robot up.

Run it after ANY server move, restart, environment change, or threshold edit:

    python humanoid_plan_server_probe.py                       # default server
    python humanoid_plan_server_probe.py --server tcp://host:9880

WHAT IT DOES. Drives real mpc_start/mpc_step sessions against the live server
from the chest home, with a SIMULATED executor, and measures whether each
tracked hand actually closes on its goal:

  perfect     the executor is exact (measured := commanded). Isolates the
              solver's cost field: anything but convergence here is the
              server's own doing.
  governed    the executor obeys the tool's real lead clamp AND a per-tick
              speed cap set to the hardware's measured tracking speed. This is
              the case that reproduces a stalled hardware track, because it is
              the one where the server's re-anchor guard meets a legitimately
              lagging arm (see MPC_REANCHOR_RAD server-side and
              MPC_LEAD_SCALE here — those two thresholds are a PAIR).

Cube positions are the 2026-08-05 hardware placements: `front` is the
front-left cube that failed twice, `rear` its peer. A PARKED verdict on
`governed` while `perfect` converges is the exact signature of the re-anchor
sawtooth, and the fix is server-side, not in the arms.
"""
from __future__ import annotations

import argparse
import humanoid_site as _site
import sys

import mujoco
import numpy as np

import humanoid_auto_operator as OP
from humanoid_curobo_client import (ARM_JOINTS_ALL, ARM_JOINTS_L, ARM_JOINTS_R,
                                    PlanClient)
from humanoid_curobo_reach import (MPC_LEAD_SCALE, PUB_DT)
from humanoid_model import MJCF_MODEL_PATH

DEFAULT_SERVER = _site.PLAN_SERVER                 # see humanoid_site / site_local.py

# The 2026-08-05 hardware placements, base frame.
FRONT_CUBE = [0.412, 0.184, 0.076]      # front-left: parked at 147/104 mm
REAR_CUBE = [-0.288, -0.437, 0.053]
CONVERGED_M = 0.035                     # the tool's own grasp-ready bar
PARKED_TAIL_MM = 5.0                    # tail motion below this = equilibrium

_MODEL = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
_DATA = mujoco.MjData(_MODEL)
_SID = {s: mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE,
                             f"end_effector_{s}_site") for s in ("L", "R")}
_JADR = {n: _MODEL.jnt_qposadr[mujoco.mj_name2id(
    _MODEL, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS_ALL}


def chest_home() -> dict[str, float]:
    enc = {n: float(v) for n, v in zip(ARM_JOINTS_L, OP.CHEST_HOME_JOINTS)}
    enc.update({n: float(v) for n, v in
                zip(ARM_JOINTS_R, OP.mirror_arm(OP.CHEST_HOME_JOINTS, "right"))})
    return enc


def fk(enc: dict[str, float]) -> dict[str, np.ndarray]:
    """EE site positions on the DEPLOY model — never the solver's self-report.
    (2026-08-01: cuRobo reported its position_error falling while an
    independent FK showed the reaching hand travelling the wrong way.)"""
    _DATA.qpos[:] = _MODEL.qpos0
    for n in ARM_JOINTS_ALL:
        _DATA.qpos[_JADR[n]] = enc[n]
    mujoco.mj_kinematics(_MODEL, _DATA)
    return {s: _DATA.site_xpos[_SID[s]].copy() for s in ("L", "R")}


def run_case(client, name, goals, *, speed_rad_tick=None, ticks=300,
             max_rate=1.0125):
    """`speed_rad_tick=None` -> perfect executor; a value -> the tool's lead
    clamp followed by that per-tick speed cap."""
    sides = sorted(goals)
    enc = chest_home()
    lead = MPC_LEAD_SCALE * max_rate * PUB_DT
    rep = client.mpc_start({}, exclude=("grasp_cube_60mm_a", "grasp_cube_60mm_b"),
                           reaching_sides=sides, measured_enc=enc,
                           cubes_init={s: np.asarray(goals[s]) for s in sides},
                           grasp_index=-1)
    print(f"[{name}] session up in {rep.get('warm_s', 0.0):.1f}s, "
          f"candidates {rep.get('grasp_index')}")
    dists = []
    for k in range(ticks):
        step = client.mpc_step(enc, {s: np.asarray(goals[s]) for s in sides})
        raw = {n: float(v) for n, v in step["enc"].items()}
        if speed_rad_tick is None:
            enc = raw
        else:
            cand = {n: enc[n] + float(np.clip(raw[n] - enc[n], -lead, lead))
                    for n in ARM_JOINTS_ALL}
            enc = {n: enc[n] + float(np.clip(cand[n] - enc[n], -speed_rad_tick,
                                             speed_rad_tick))
                   for n in ARM_JOINTS_ALL}
        ee = fk(enc)
        dists.append({s: float(np.linalg.norm(ee[s] - np.asarray(goals[s])))
                      for s in sides})
        if k % 100 == 0 or k == ticks - 1:
            print(f"[{name}] tick {k:3d}  " + "  ".join(
                f"{s} {dists[-1][s] * 1000:6.1f} mm" for s in sides))
    client.mpc_stop()

    ok = True
    tail = dists[-min(120, len(dists)):]
    for s in sides:
        final = dists[-1][s]
        moved_mm = abs(tail[0][s] - tail[-1][s]) * 1000
        if final < CONVERGED_M:
            verdict = "CONVERGED"
        elif moved_mm < PARKED_TAIL_MM:
            verdict = "PARKED — equilibrium short of the goal"
            ok = False
        else:
            verdict = "STILL MOVING at the tick budget (inconclusive)"
            ok = False
        print(f"[{name}] {s}: final {final * 1000:.1f} mm, tail motion "
              f"{moved_mm:.1f} mm -> {verdict}")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--speed-rad-s", type=float, default=0.22,
                    help="governed-case executor speed; 0.22 is the measured "
                         "hardware tracking speed under the lead clamp")
    ap.add_argument("--ticks", type=int, default=300)
    args = ap.parse_args()

    print(f"[probe] server {args.server}")
    both = {"L": FRONT_CUBE, "R": REAR_CUBE}
    client = PlanClient(args.server)
    results = {}
    try:
        results["perfect dual"] = run_case(
            client, "perfect dual", both, ticks=args.ticks)
        results["perfect left-only"] = run_case(
            client, "perfect left-only", {"L": FRONT_CUBE}, ticks=args.ticks)
        results["governed dual"] = run_case(
            client, "governed dual", both,
            speed_rad_tick=args.speed_rad_s * PUB_DT, ticks=3 * args.ticks)
    finally:
        client.close()

    print("=" * 62)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if all(results.values()):
        print("\n[probe] server healthy: every tracked hand reached its cube.")
        return 0
    print("\n[probe] FAILED. If `perfect` converges and `governed` parks, the "
          "server's re-anchor guard is confiscating the lead the client "
          "grants — compare server MPC_REANCHOR_RAD against "
          f"MPC_LEAD_SCALE*max_rate*PUB_DT = {MPC_LEAD_SCALE * 1.0125 * PUB_DT:.3f} rad.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
