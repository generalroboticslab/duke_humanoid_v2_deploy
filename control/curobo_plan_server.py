"""cuRobo plan-0 server — runs on the GPU machine, serves the robot over pynng.

WHAT THIS IS. The GPU side of the reach stack: one warm cuRobo session
(plan-0 IK + trajopt, CUDA graph captured once at start-up) plus at most one
live MPC tracking session (`MpcSession`: mpc_start / mpc_step / mpc_stop,
driven by the robot-side client), behind a single pynng Rep0 socket. The
robot computer has no CUDA; the GPU machine does. The upstream cuRobo
integration already isolates every GPU solve in ONE worker process behind a
blocking request/response message tuple (`_reach_mpc_worker`, scene.py)
precisely so that no cuRobo runs per control tick. This file keeps that
contract and swaps the multiprocessing queues for the Rep0 socket, so the
"spawned process" can live on another machine. Nothing else changes: the same
`solve_reach_route`, the same warm-once CUDA graph, the same generation-tagged
replies.

DEPLOYMENT (the GPU machine):

    HUMANOID_LEGGED_ENV_ROOT=<legged_env_v2> python control/curobo_plan_server.py --port 9880

Requirements: cuRobo at commit 8e734f3 installed in that interpreter, and the
training/deploy repository legged_env_v2 (upstream clone name legged_env_dev;
not public at the time of release) — the planner and the scene
(`mj_envs.tasks.visual_manipulation.curobo`) are imported at module level,
BEFORE argparse runs. The server puts `humanoid_site.LEGGED_ENV_ROOT`
(HUMANOID_LEGGED_ENV_ROOT in the environment, then LEGGED_ENV_ROOT in
control/site_local.py, then ~/repo/legged_env_v2) on sys.path itself, at
position 0 — so PYTHONPATH cannot be used for this: whatever LEGGED_ENV_ROOT
resolves to is searched first, and on a release checkout that is the bundled
subset, which carries no planner. Set the environment variable above, or
LEGGED_ENV_ROOT in control/site_local.py. A missing or mis-pointed checkout
fails at import — `--help` included — with
those remedies named rather than a bare ModuleNotFoundError. (The original
recipe also exported MUJOCO_GL=egl for a headless box.) Flags: --port (default 9880),
--hand-z-floor / --no-hand-z-floor / --hand-floor-weight — see `main()`.
Verify from the robot computer with `humanoid_plan_server_probe.py` (three
cases must PASS) before the robot stands; `docs/OPERATIONS.md` section 1 has
the restart-over-ssh caveats. The deploy robot.xml this plans against was
validated against hardware gravity torque on 2026-07-28.

PROTOCOL (msgpack over Rep0, one reply per request, strictly alternating):

  {"kind": "ping"}
      -> {"kind": "pong", "robot": "v2", "warmed": true}
  {"kind": "plan0", "gen": int, "scene": {"cuboid": {name: {"dims": [3],
       "pose": [x,y,z,qw,qx,qy,qz]}}}, "targets": {cube: [x,y,z]},
       "base_pos": [3], "base_quat": [4] (wxyz),
       "assignment": null | {side: cube}, "seed_q": null | {joint: rad},
       "goal": "reach" | "home"}
      -> {"kind": "plan0", "gen": int, "solve_s": float, "err": null | str,
          "route": null | {"joint_names": [...], "route_q": [[...]xH],
                           "interpolation_dt": float,
                           "reaches": {frame: {"target": [3],
                                               "cands": [[3]xG],
                                               "err": float}},
                           "controlled_joints": [...],
                           "base_pos": [3], "base_quat": [4],
                           "assignment": {side: cube}}}

Units seam: everything here is cuRobo cspace / base frame, exactly as
`solve_reach_route` speaks it. The ENCODER-unit fold (+qpos0, and the
left_wrist_2 axis-flip sign) is deliberately NOT done here — it belongs to the
robot-side client, next to the deploy model that defines it, and is verified
there by FK gates before anything moves.

Failure policy: every request gets a reply (Rep0 wedges otherwise); any solver
exception comes back in "err" with the traceback, and the server stays up.
`seed_q` keys are cuRobo joint names; missing keys fail the request loudly
rather than defaulting (a silently-defaulted seed is exactly the wrong-branch
class the route[0] gate on the client exists to catch).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import re
import sys
import time
import traceback

import msgpack
import numpy as np
import pynng

# legged_env_v2 imports — module level, before argparse (see header). The
# planner and the scene live in the training/deploy repository, not here, so
# put its root on sys.path the way humanoid_real_env.py does (humanoid_site
# resolves it: HUMANOID_LEGGED_ENV_ROOT, then control/site_local.py, then
# ~/repo/legged_env_v2). 2026-08-22: without this the import fell through to
# whatever `mj_envs` the interpreter had installed and `--help` died in a bare
# ModuleNotFoundError before argparse ran; a missing checkout now fails with
# the three remedies named, the original error chained underneath.
from humanoid_site import LEGGED_ENV_ROOT as _LEV_ROOT

if str(_LEV_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEV_ROOT))
try:
    from mj_envs.tasks.visual_manipulation.curobo.planner import (  # noqa: E402
        CFG_BY_ROBOT,
        CuroboPlannerSession,
        Kinematics,
        RobotCfg,
    )
    from mj_envs.tasks.visual_manipulation.curobo.scene import (  # noqa: E402
        _nominal_warm_scene_and_targets,
        solve_reach_route,
    )
except ImportError as _e:
    raise ImportError(
        f"cannot import the cuRobo planner/scene "
        f"(mj_envs.tasks.visual_manipulation.curobo) from legged_env_v2 — "
        f"looked under {_LEV_ROOT}: {_e}. The plan server needs the "
        f"training/deploy repository legged_env_v2 checked out and importable, "
        f"with cuRobo (commit 8e734f3) installed in this interpreter (the bundled "
        f"control/legged_env_bundle covers the robot side only, not the planner). Point the "
        f"stack at the checkout one of two ways: set HUMANOID_LEGGED_ENV_ROOT "
        f"in the environment, or set LEGGED_ENV_ROOT in control/site_local.py. "
        f"PYTHONPATH does not work here — LEGGED_ENV_ROOT is inserted at sys.path[0] "
        f"and wins (README 'Install'; docs/OPERATIONS.md section 1).") from _e

ROBOT = "v2"

MPC_OPTIMIZATION_DT = 0.02        # 50 Hz horizon step. The client publishes at
                                  # 20 Hz, so one commanded step is a
                                  # deliberately SMALL fraction of a control
                                  # period — the tracker corrects continuously
                                  # rather than lunging once per tick.
FLOOR_REBUILD_EPS_M = 0.005       # ignore sub-5mm differences: a table pose is a
                                  # fitted measurement and jitters, and rebuilding
                                  # the CUDA graph for detection noise would cost
                                  # ~4 s per run for nothing
MPC_FLOOR_FINGER_DROP_M = 0.045   # the tracker's floor sits this far BELOW the
                                  # plan's sticky floor (2026-08-12 hardware).
                                  # The plan floor is capped 15 mm under the
                                  # grasp target, but the floor cost rides on
                                  # SPHERE CENTRES of the finger racks, and a
                                  # centre grasp needs the lower rack spheres
                                  # 30-45 mm below the grasp line — at 1e6
                                  # weight the tracker equilibrated 2-3 cm HIGH
                                  # on every front grasp ("no progress for 6s",
                                  # deficits all z+30..40, jaws biting the top
                                  # edge of the cube). The plan keeps the full
                                  # floor (its sweeps are where table strikes
                                  # live); the MPC keeps only a deep dive-
                                  # catcher — its last-centimetre shields are
                                  # the table cuboids in its own world, the
                                  # 120 mm leash to a floor-respecting plan
                                  # anchor, and the client close gates.
MPC_COMMIT_IDX = 1                # horizon index to COMMIT. 0 is the current
                                  # state (feeding it back freezes the arm);
                                  # -1 is the horizon END, which converges in
                                  # few ticks but is a sequence of snapshots,
                                  # not a path the robot could execute.
MPC_WARM_ITERS = 50               # upstream's measured knee: p90 6 ms and still
                                  # <1 mm on transit targets; 200 buys 0.1 mm
                                  # for 16 ms, which does not fit the budget.
MPC_REANCHOR_RAD = 0.10           # commanded-vs-measured gap that proves the
                                  # arm is NOT executing the rollout. The
                                  # session carries solver derivatives across
                                  # ticks (see _state_from) — necessary for
                                  # motion, but when the client rate-clamps
                                  # execution (0.3 rad/s vs the solver's own
                                  # limits) the carried velocity is a phantom:
                                  # the rollout runs ahead of the arm and the
                                  # gap GROWS every tick (hardware 2026-08-01:
                                  # 0.48 rad by tick ~110, arm visibly lifted
                                  # off the planned route; reproduced offline
                                  # — a pinned state grows 0 -> 0.118 rad in
                                  # 40 ticks). Past this gap the step
                                  # re-anchors: derivatives dropped,
                                  # reset_robot(measured), one accel-limited
                                  # restart instead of a runaway.

# MPC imports are deferred into MpcSession: plan-0 is the validated path and must
# keep booting even if the MPC surface drifts upstream. A broken import there
# fails the mpc_start request, loudly, instead of the whole server.

# ------------------------------------------------------------------------------
# Executable-limit patch: plan only in the space the robot can actually execute.
#
# The URDF's wrist_1 limits are ±pi about the UPSTREAM zero (the CAD pose); the
# robot's encoders — and the receiver's hard clamp — are ±pi about the ENCODER
# zero, 90 deg away (the deploy model's wrist_1 `ref`). Half of the planner's
# wrist_1 range therefore maps outside the executable window, and the very
# first smoke solve used it (enc -4.71 on a ±pi joint, a clean pi/2 violation
# flagged by the client's Gate D). The fix is at the source: constrain the
# planner to the INTERSECTION of both conventions,
#
#     left_wrist_1  cspace [-pi/2, pi]     (enc = cspace - pi/2  in [-pi, pi])
#     right_wrist_1 cspace [-pi, pi/2]     (enc = cspace + pi/2  in [-pi, pi])
#
# so every route is executable by construction. All other revolute limits are
# shrunk by _LIMIT_SHRINK_RAD because cuRobo willingly plans ONTO its limit
# boundary (observed: shoulder_3 at 0.006 rad from the stop) while the client
# refuses anything inside 0.02 — planning with the margin baked in keeps Gate D
# a pure tripwire instead of a recurring veto. Patch happens BEFORE session
# build, so IK, trajopt and the CUDA graph all see the same limits.
# ------------------------------------------------------------------------------
MPC_SHARE_PLAN0_GRASPS = True     # build the MPC's grasp candidates from the
                                  # ROBOT DESCRIPTOR, so the tracker refines the
                                  # pose plan-0 committed to instead of picking
                                  # a differently-oriented sibling. False
                                  # restores ik_curobo's defaults if a hardware
                                  # round ever needs the old behaviour back.
MPC_GRASP_GAP_WARN_DEG = 5.0      # AUTO-pick orientation gap that means the two
                                  # solvers are not planning the same grasp
TOOL_FLANGE_OFFSET_M = 0.169      # end_effector_*_site out along the flange x
                                  # (robot.xml). The lever that turns an
                                  # orientation gap into a lateral swing.
WRIST1_EXEC_LIMITS = {
    "left_wrist_1_joint": (-math.pi / 2, math.pi),
    "right_wrist_1_joint": (-math.pi, math.pi / 2),
}
_LIMIT_SHRINK_RAD = 0.025


def _patch_urdf_limits(urdf_path: str) -> str:
    """Write a sibling URDF with executable wrist_1 windows and shrunk limits.

    Sibling (same directory) so any relative references keep resolving; line-
    oriented because this URDF keeps each <joint> on one line."""
    src = pathlib.Path(urdf_path)
    out_lines = []
    patched = 0
    for line in src.read_text().splitlines():
        m = re.search(r'<joint name="([^"]+)" type="revolute"', line)
        if m and 'lower="' in line:
            name = m.group(1)
            lo = float(re.search(r'lower="([^"]+)"', line).group(1))
            hi = float(re.search(r'upper="([^"]+)"', line).group(1))
            if name in WRIST1_EXEC_LIMITS:
                lo, hi = WRIST1_EXEC_LIMITS[name]
            lo, hi = lo + _LIMIT_SHRINK_RAD, hi - _LIMIT_SHRINK_RAD
            assert lo < hi, f"limit patch inverted {name}"
            line = re.sub(r'lower="[^"]+"', f'lower="{lo:.8f}"', line)
            line = re.sub(r'upper="[^"]+"', f'upper="{hi:.8f}"', line)
            patched += 1
        out_lines.append(line)
    dst = src.with_suffix(".plan_server.urdf")
    dst.write_text("\n".join(out_lines) + "\n")
    print(f"[server] limit patch: {patched} revolute joints -> {dst.name} "
          f"(wrist_1 windows {WRIST1_EXEC_LIMITS}, shrink {_LIMIT_SHRINK_RAD})",
          flush=True)
    return str(dst)


def upstream_floor_default() -> float | None:
    """The hand z-floor `build_curobo_planner` applies when nobody says otherwise.

    Read by introspection, not copied, because it is the number that decides
    whether a low reach is possible AT ALL and a stale copy here would describe a
    planner we are not running. Asserted at boot so an upstream change is loud.
    """
    import inspect

    from mj_envs.tasks.visual_manipulation.curobo.ik_curobo import (  # noqa: PLC0415
        build_curobo_motion_planner)
    # This is the builder CuroboPlannerSession actually calls for the B=1 pose
    # plan we use (planner.py: self.pose_planner = build_curobo_motion_planner).
    d = inspect.signature(
        build_curobo_motion_planner).parameters["hand_z_floor"].default
    return None if d is None else float(d)


def make_patched_session(cfg, hand_z_floor: float | None = None,
                         hand_floor_weight: float | None = None,
                         disable_hand_floor: bool = False):
    """make_planner_session with the executable-limit URDF swapped in, and the
    scoped hand z-floor cost overridden or switched off.

    THE FLOOR IS ALREADY ON, AND THAT IS THE POINT OF THESE FLAGS. It is easy to
    read HUMANOID_CFG's `planner_kwargs={}` as "no floor cost" — G1_CFG spells
    one out, this robot does not — but `{}` means "take the default", and
    build_curobo_planner defaults to hand_z_floor=0.06 with weight 1e6. Every
    plan this server has ever produced forbade the reaching hand's 24 gripper
    spheres from going below +0.060 m in the BASE frame.

    That default is a table-height assumption, and it is silent when wrong in
    either direction:
      * a work surface whose top sits BELOW +0.060 m (the pelvis is roughly
        0.7 m up, so a 0.75 m lab table lands right around it) makes a 1e6-weight
        cost fight every legitimate reach onto that surface;
      * a cube on the FLOOR is simply unreachable, and the symptom is INFEASIBLE
        or a hovering hand, not a message about a floor.
    So: --hand-z-floor to move it to the surface actually being worked over,
    --no-hand-z-floor to remove it for a genuinely floor-free reach. The weight
    is left alone unless asked for — passing a floor should not silently drop
    1e6 to something weaker.

    WHY LAUNCH-TIME AND NOT PER REQUEST: the cost rides cuRobo's self_collision
    slot and must be attached BEFORE graph capture, so the value is baked into
    the warm CUDA graph. A work surface's height does not change during a
    session, so a flag is the honest shape for it.

    WHAT THE FLOOR BUYS, measured upstream: without one, a greedy path dips the
    reaching site under a table lip and the thin slab then walls it off from
    climbing back (hand-sphere z ~ -0.19 m, 177-244 mm final error). plan_pose's
    graph search is far less exposed than the MPC that produced those numbers,
    but the reactive tracker will be exactly that greedy.
    """
    over = {}
    if disable_hand_floor:
        over["hand_z_floor"] = None
        print("[server] hand z-floor DISABLED (--no-hand-z-floor): the reaching "
              "hand may go arbitrarily low", flush=True)
    elif hand_z_floor is not None:
        over["hand_z_floor"] = float(hand_z_floor)
    if hand_floor_weight is not None:
        over["hand_floor_weight"] = float(hand_floor_weight)
    if over:
        cfg = dataclasses.replace(cfg, planner_kwargs={**cfg.planner_kwargs, **over})
    cfg_dict = cfg.build_robot_cfg_dict(cfg.planning_home_joint_pos)
    kin = cfg_dict["robot_cfg"]["kinematics"]
    kin["urdf_path"] = _patch_urdf_limits(kin["urdf_path"])
    return CuroboPlannerSession(cfg, cfg_dict, Kinematics(RobotCfg.create(cfg_dict).kinematics))


class MpcSession:
    """A warm cuRobo MPCSolver held across requests — the real-time half.

    WHAT THIS IS FOR. plan-0 answers "give me a whole collision-free path", once,
    in ~2 s. It cannot answer "given where the arm is NOW and where the cube is
    NOW, what is the next command" at 20-50 Hz — that is a different solver, and
    cuRobo ships it: MPCSolver optimises a ~1 s receding horizon and returns the
    first commanded step, warm-started from the previous tick, at ~6 ms.

    THE UPSTREAM VERDICT IS THE DESIGN. The upstream 2026-07-06 probe measured
    MPC as a PLANNER to be a dead end — a from-HOME reach with an external 8-candidate select cost
    ~3.5 s against plan_pose's ~28 ms — and concluded it "only pays off as a
    REACTIVE TRACKER in a live 50Hz loop (single moving goal, no candidate loop,
    tracking a plan_pose path)". So this class never plans a reach. It tracks one.

    THREE THINGS FROM THAT PROBE THAT ARE NOT OPTIONAL:

      1. update_tool_pose_criteria MUST be applied AFTER setup(). Applied before,
         setup's cold-start capture keeps the stock factor and the reach only
         makes ~25 mm. This cost upstream a wrong conclusion once already.
      2. max_goalset=1. cuRobo's IN-SOLVER goalset LOOSENS convergence — 8
         identical candidates gave ~23 mm where a single goal gave 0.13 mm, and
         it thrashes between equal-position candidates. The grasp candidate is
         therefore CHOSEN by plan-0 and tracked as one goal.
      3. The goal carries ORIENTATION. With orientation masked, the IK null space
         is unconstrained and the optimiser wanders it, leaving 40-50% of the
         offset as steady-state error. With the full grasp pose pinned at the
         terminal waypoint it reaches 0.0-0.1 mm.

    AND THE ONE THAT MADE UPSTREAM DELETE MPC FROM THE SIM: the grasp target sits ON
    an object that the same solver is being told is a hard obstacle, so the hand
    wedges short of it. Here the cube being grasped is dropped from the MPC
    scene by name (`exclude`), enforced on this side rather than trusted to the
    caller — everything else, tables and other cubes included, stays.
    """

    def __init__(self, cfg, cfg_dict, scene: dict, reaching_sides,
                 seed_q: dict, hand_z_floor: float | None,
                 hand_floor_weight: float, warm_iters: int, grasp_index,
                 cubes_init: dict | None = None):
        import torch
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
        from curobo._src.solver.solver_mpc import MPCSolver
        from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
        from mj_envs.tasks.visual_manipulation.curobo.ik_curobo import (
            attach_hand_floor_cost, cube_grasp_pose_base, mpc_tool_goal)

        self._torch = torch
        self._mpc_tool_goal = mpc_tool_goal
        # ONE GRASP CANDIDATE SET FOR BOTH SOLVERS (audit 2026-08-06, D2).
        # ik_curobo.cube_grasp_pose_base regenerates the candidates internally
        # with cube_grasp_poses_obj's DEFAULTS — flip=True, beta_degs=(90,),
        # z_above=0 — while plan-0 planned onto the descriptor's own set:
        # HUMANOID_GRASP_BETAS (75/60/45 deg), flip=False,
        # z_above=GRASP_Z_ABOVE_M. The two sets are disjoint in orientation, so
        # the AUTO pick could never land nearer than 15, 30 or 45 degrees from
        # the pose plan-0 had committed to, and the tracker commanded that
        # re-orientation on its FIRST tick, on the terminal approach: at the
        # 0.169 m tool offset, 44 / 88 / 129 mm of flange swing. Reproduced
        # against both generators (12 candidates at z=0.010 vs 8 at z=0.000).
        #
        # Rebuilding it here from `cfg` — which was already a parameter and
        # entirely unused — keeps the fix inside this file: legged_env_v2 is
        # shared, and its copy on the GPU machine carries its own patches.
        #
        # THE Z GOAL MOVES WITH IT, and that is the point: the MPC's terminal
        # position becomes cube centre + GRASP_Z_ABOVE_M, the anchor plan-0
        # uses and the one planner.py records as necessary because the exact
        # centre is IK-infeasible for this robot ("front_back_close: 0% reach,
        # every stance"). Expect converged `worst` to settle near 10 mm rather
        # than 0 — the client measures to the cube centre.
        if MPC_SHARE_PLAN0_GRASPS and getattr(cfg, "cube_grasp_poses_obj", None):
            def _grasp(mpc, cube_pos_base, cube_quat_base, grasp_index=0):
                dev = mpc.device_cfg.device
                gp, gq = cfg.cube_grasp_poses_obj(device=dev)
                op = torch.as_tensor(cube_pos_base, device=dev, dtype=torch.float32)
                oq = torch.as_tensor(cube_quat_base, device=dev, dtype=torch.float32)
                cp, cq = cfg.grasp_poses_to_base(op, oq, gp, gq)
                return cp[grasp_index], cq[grasp_index]
            self._cube_grasp_pose_base = _grasp
            print("[mpc] grasp candidates: the PLANNER's set "
                  "(same betas, flip and z_above as plan-0)", flush=True)
        else:
            self._cube_grasp_pose_base = cube_grasp_pose_base
            print("[mpc] grasp candidates: ik_curobo DEFAULTS — plan-0 and the "
                  "tracker may disagree about the grasp orientation", flush=True)
        # DUAL TRACKING (2026-08-01). `reaching_sides` is 1 or 2 of {"L","R"};
        # every listed side tracks ITS OWN live cube goal each step, every
        # unlisted side is pinned at its session-start FK pose (the constant
        # pin — see step()). The single-side path is byte-compatible: one
        # tracked side behaves exactly as the validated 07-30 session did.
        if isinstance(reaching_sides, str):
            reaching_sides = [reaching_sides]
        self.tracked = [s.upper() for s in reaching_sides]
        assert self.tracked and set(self.tracked) <= {"L", "R"} and \
            len(self.tracked) == len(set(self.tracked)), \
            f"bad reaching_sides {reaching_sides!r}"
        # Per-side grasp candidate. -1 = AUTO: pick the candidate whose
        # orientation is nearest the seed posture's own tool orientation, so
        # the tracker refines the pose plan-0 committed to instead of twisting
        # the wrist to an arbitrary sibling candidate centimetres from the
        # cube (the candidates share one position — they differ by yaw/flip).
        if isinstance(grasp_index, dict):
            self.grasp_index = {s.upper(): int(v) for s, v in grasp_index.items()}
        else:
            self.grasp_index = {s: int(grasp_index) for s in self.tracked}
        for s in self.tracked:
            self.grasp_index.setdefault(s, -1)

        solver_cfg = MPCSolverCfg.create(
            robot=cfg_dict,
            scene_model=scene,
            self_collision_check=True,
            optimization_dt=MPC_OPTIMIZATION_DT,
            max_goalset=1,
            store_debug=False,
            warm_start_optimization_num_iters=int(warm_iters),
            cold_start_optimization_num_iters=max(int(warm_iters), 200),
        )
        self.mpc = MPCSolver(solver_cfg)
        # BEFORE setup: the floor rides the self_collision slot and has to be
        # inside the captured CUDA graph. This solver is the greedy one upstream's
        # below-table dive came from, so it wants the floor far more than
        # plan-0 does. EVERY tracked hand gets it — with two hands diving for
        # two cubes, a floor on one of them is half a guarantee.
        if hand_z_floor is not None:
            links = []
            for s in self.tracked:
                links += [f"wrist_3_{s}", f"{s}_left_rack", f"{s}_right_rack"]
            n = attach_hand_floor_cost(self.mpc, links, float(hand_z_floor),
                                       float(hand_floor_weight))
            print(f"[mpc] hand z-floor {hand_z_floor:+.4f} m on {n} spheres "
                  f"across {len(self.tracked)} hand(s) "
                  f"(weight {hand_floor_weight:g})", flush=True)

        # THE GOAL ORDER IS CUROBO'S, NOT OURS. GoalToolPose pairs its stacked
        # poses with mpc.tool_frames POSITIONALLY, so the goal list has to be
        # built in that exact order and each slot filled by NAME. The first
        # version ordered them idle-then-reaching (copying the probe, whose
        # reaching side happens to be R, making [idle, reach] == cuRobo's
        # canonical [L, R]) — and plan-0 assigned us the LEFT arm, so the two
        # goals were SWAPPED: the grasp pose drove the idle hand while the
        # reaching hand was pinned. cuRobo's own position_error fell 629 -> 82 mm
        # the whole time, because the arm it was converging really was reaching
        # the goal; only an independent FK on our deploy model showed the
        # REACHING hand travelling the wrong way (468 -> 718 mm). Never trust one
        # solver's self-report about which hand it moved.
        self.frames = list(self.mpc.tool_frames)
        self.sides = [f.split("_")[2] for f in self.frames]   # end_effector_<S>_site
        assert set(self.sides) == {"L", "R"}, f"unexpected tool frames {self.frames}"
        assert set(self.tracked) <= set(self.sides), \
            f"tracked sides {self.tracked} not among {self.frames}"

        self.joint_names = list(self.mpc.joint_names)
        self._col = {n: i for i, n in enumerate(self.joint_names)}
        self._deriv: dict = {}                 # carried across ticks (see _state_from)
        self._last_cmd = None                  # last q_next, index-aligned with
                                               # joint_names (re-anchor guard)
        self.reanchors = 0
        # The guard watches ONLY the tracked sides' joints. The untracked
        # arm's q_next is BY CONTRACT never executed (the client freezes it at
        # the handoff snapshot; server-side it just orbits its pin), so its
        # commanded-vs-measured gap measures nothing — and letting it trip the
        # guard would cold-start the whole solver over an arm that is behaving
        # exactly as designed (review 2026-08-01).
        prefixes = tuple({"L": "left_", "R": "right_"}[s] for s in self.tracked)
        self._guard_cols = np.asarray(
            [i for n, i in self._col.items() if n.startswith(prefixes)], int)
        state = self._state_from(seed_q)       # at rest: the arm IS stationary here
        t0 = time.monotonic()
        self.mpc.setup(state.clone())          # allocates goal buffer + cold start
        criteria = ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[1.0] * 6,
            non_terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        # AFTER setup — see the class docstring; before it, this silently does
        # nothing and the reach stalls ~25 mm out.
        self.mpc.update_tool_pose_criteria(
            {f"end_effector_{s}_site": criteria for s in self.sides})
        self.mpc.reset_robot(state.clone())
        self.warm_s = time.monotonic() - t0
        kin = self.mpc.compute_kinematics(state)
        # Every UNTRACKED side is pinned at its session-start FK pose — a
        # CONSTANT, never recomputed (see step()'s divergence note). With both
        # sides tracked this dict is simply empty.
        self._pins = {}
        for s in self.sides:
            if s in self.tracked:
                continue
            pose = kin.tool_poses.get_link_pose(f"end_effector_{s}_site")
            self._pins[s] = (pose.position[0].clone(), pose.quaternion[0].clone())
        # AUTO grasp-candidate resolution, per tracked side. The candidates for
        # a cube share one position and differ by yaw/flip orientation; plan-0
        # already committed the wrist to one of them, and that commitment is
        # standing in seed_q. Picking the candidate nearest the seed's own tool
        # orientation means the tracker REFINES the approach instead of
        # re-deciding it centimetres from the cube. Needs the cube's pose,
        # hence cubes_init; -1 with no cube pose is refused loudly.
        for s in self.tracked:
            if self.grasp_index[s] >= 0:
                continue
            if not cubes_init or s not in cubes_init:
                raise ValueError(
                    f"grasp_index AUTO for side {s} needs an initial cube pose "
                    f"in mpc_start['cubes'][{s!r}]")
            cpos, cquat = cubes_init[s]
            tool = kin.tool_poses.get_link_pose(f"end_effector_{s}_site")
            tq = tool.quaternion[0]
            best, best_d = 0, float("inf")
            g = 0
            while True:
                try:
                    _p, cand_q = self._cube_grasp_pose_base(
                        self.mpc, cpos, cquat, g)
                except IndexError:
                    break                       # past the last candidate
                except RuntimeError:
                    # RuntimeError is what a REAL failure raises (malformed
                    # cube pose from the wire, transient CUDA error) — only
                    # torch-version drift could make it mean end-of-list. If
                    # not one candidate was probed, committing to 0 would be
                    # exactly the arbitrary-sibling wrist twist AUTO exists to
                    # prevent: refuse loudly instead.
                    if best_d == float("inf"):
                        raise
                    break
                d = 1.0 - abs(float((cand_q * tq).sum()))
                if d < best_d:
                    best, best_d = g, d
                g += 1
                if g > 64:                      # safety stop, candidate sets are small
                    break
            self.grasp_index[s] = best
            # THE GAP IN DEGREES, BECAUSE NOBODY READS 1-|<q,q>| AS AN ANGLE.
            # d = 1 - |dot| for unit quaternions, so the rotation between them
            # is 2*acos(1-d). This number has been in the log all along in a
            # unit that hid what it meant: plan-0 plans onto a TILTED grasp
            # candidate (HUMANOID_GRASP_BETAS 75/60/45 deg) while this AUTO pick
            # chooses from ik_curobo.cube_grasp_poses_obj's defaults, which are
            # dead level (beta=90) — so the nearest neighbour is 15, 30 or 45
            # degrees away, printed as 0.0086 / 0.0341 / 0.0761 (audit
            # 2026-08-06, reproduced against both generators). At the 0.169 m
            # tool offset that is 44 / 88 / 129 mm of flange swing commanded on
            # tick 1, on the terminal approach, every round.
            gap_deg = math.degrees(2.0 * math.acos(
                max(-1.0, min(1.0, 1.0 - best_d))))
            note = ""
            if gap_deg > MPC_GRASP_GAP_WARN_DEG:
                # WHAT THIS COMPARES: the chosen candidate against the SEED
                # posture's own tool orientation. On the mission the seed is
                # the measured posture at the handoff — i.e. plan-0's route
                # endpoint — so a large gap means the tracker is about to undo
                # the grasp plan-0 committed to. A probe that seeds from chest
                # home instead will read tens of degrees legitimately; judge
                # this number only on a real handoff.
                note = (f"  <-- WARNING: the nearest grasp candidate is "
                        f"{gap_deg:.1f} deg from the SEED posture's tool "
                        f"orientation. On a real handoff the seed is plan-0's "
                        f"route endpoint, so the tracker would re-orient the "
                        f"wrist on its first tick — about "
                        f"{2000.0 * TOOL_FLANGE_OFFSET_M * math.sin(math.radians(gap_deg) / 2.0):.0f}"
                        f" mm of lateral flange travel at the terminal "
                        f"approach.")
            print(f"[mpc] side {s}: AUTO grasp candidate -> {best} "
                  f"(orientation gap {gap_deg:.1f} deg / d={best_d:.4f}, "
                  f"{g} candidates){note}", flush=True)
        self.ticks = 0

    def _state_from(self, q: dict):
        """POSITION from the caller (i.e. from the robot); DERIVATIVES from the
        previous solve.

        This split is not a detail — the first version rebuilt a zero-velocity
        state every tick and the arm crawled at 2.6 mm/s (measured: 629.5 mm ->
        627.5 mm over 40 ticks). The transition model is a B-spline over
        position/velocity/acceleration/jerk; hand it rest every tick and it can
        only ever produce one dt of acceleration-limited motion, then start over.

        Position must still come from the robot, because tracking a MEASURED arm
        is the entire point. Derivatives must NOT: telemetry joint velocity is
        noisy and, more importantly, is not the same quantity as the spline's
        internal state, so feeding it in would fight the optimiser's own
        continuity. Solver derivatives + measured position is the pairing that
        keeps both properties.
        """
        state = self.mpc.default_joint_state.clone().unsqueeze(0)
        missing = [n for n in self.joint_names if n not in q]
        if missing:
            # Loud, never defaulted: a silently-defaulted joint is exactly the
            # wrong-branch class the client's route[0] gate exists to catch.
            raise KeyError(f"state is missing joints {missing[:6]}"
                           f"{' …' if len(missing) > 6 else ''}")
        for name, i in self._col.items():
            state.position[0, i] = float(q[name])
        for field in ("velocity", "acceleration", "jerk"):
            cur = getattr(state, field, None)
            if cur is None:
                continue
            carried = self._deriv.get(field)
            cur[:] = 0.0 if carried is None else carried
        return state

    def step(self, cubes: dict, state_q: dict) -> dict:
        """One receding-horizon solve. Returns the FIRST commanded step.

        `cubes` = {side: (pos[3], quat[4])} for EVERY tracked side, each the
        LIVE pose of that hand's own cube — this per-tick re-aiming is the
        entire point of the session. Untracked sides keep their constant pin:
        The upstream probe recomputes the idle pose from the current state each call,
        which is equivalent in its kinematic loop (the idle arm never moves
        there) but is a feedback loop here — measured 2026-07-30, recomputing
        per tick made the reaching hand touch 85 mm and then DIVERGE to 227,
        where the constant pin converges monotonically. An arm that is supposed
        to hold still must not have a goal the optimiser can chase.
        """
        missing = [s for s in self.tracked if s not in cubes]
        if missing:
            raise KeyError(f"mpc_step is missing cube poses for tracked "
                           f"side(s) {missing}")
        state = self._state_from(state_q)
        # RE-ANCHOR GUARD (hardware 2026-08-01). The carried derivatives assume
        # the arm executes the rollout; a rate-clamped (or gate-held) arm does
        # not, and the rollout then runs ahead of reality with positive
        # feedback — the client sees ever-growing commands, rejects them, the
        # arm stops entirely, and the gap grows even faster (0.48 rad, 21
        # straight rejections, aborted round). When the last commanded step
        # provably did not happen, drop the phantom derivatives and reseed the
        # trajectory buffer from the MEASURED state: one accel-limited restart,
        # paced to what the arm actually executed.
        if self._last_cmd is not None:
            meas = state.position[0].detach().cpu().numpy()
            gap = float(np.abs(meas[self._guard_cols]
                               - self._last_cmd[self._guard_cols]).max())
            if gap > MPC_REANCHOR_RAD:
                self._deriv.clear()
                for field in ("velocity", "acceleration", "jerk"):
                    cur = getattr(state, field, None)
                    if cur is not None:
                        cur[:] = 0.0
                self.mpc.reset_robot(state.clone())
                self.reanchors += 1
                print(f"[mpc] re-anchor #{self.reanchors} at tick "
                      f"{self.ticks}: commanded-vs-measured gap {gap:.3f} rad "
                      f"— the arm is not executing the rollout; restarting "
                      f"it from the measured state", flush=True)
        goals = {}
        for s in self.tracked:
            cpos, cquat = cubes[s]
            goals[s] = self._cube_grasp_pose_base(self.mpc, cpos, cquat,
                                                  self.grasp_index[s])
        positions, quats = [], []
        for s in self.sides:                   # cuRobo's frame order, filled by NAME
            if s in goals:
                positions.append(goals[s][0])
                quats.append(goals[s][1])
            else:
                positions.append(self._pins[s][0])
                quats.append(self._pins[s][1])
        goal = self._mpc_tool_goal(self.sides, positions, quats)
        self.mpc.update_goal_tool_poses(goal, run_ik=False)
        t0 = time.monotonic()
        result = self.mpc.optimize_action_sequence(state)
        solve_s = time.monotonic() - t0
        # Horizon index 1, not 0: index 0 IS the current state, and feeding it
        # back freezes the arm in place (probe's `commit_first` path).
        act = result.action_sequence
        q_next = act.position[0, MPC_COMMIT_IDX].detach().cpu().numpy()
        self._last_cmd = q_next.copy()         # the re-anchor guard's baseline
        for field in ("velocity", "acceleration", "jerk"):
            v = getattr(act, field, None)
            if v is not None:
                self._deriv[field] = v[:, MPC_COMMIT_IDX, :].clone()
        err = result.position_error
        self.ticks += 1
        goal_wire = {s: [float(v) for v in p.detach().cpu().numpy()]
                     for s, (p, _q) in goals.items()}
        return {"q_next": {n: float(q_next[i]) for n, i in self._col.items()},
                "pos_err": None if err is None else float(err.min()),
                "solve_s": solve_s,
                # legacy single-goal field kept for the smoke test; the dual
                # consumer reads `goals`
                "goal_pos": next(iter(goal_wire.values())),
                "goals": goal_wire,
                "ticks": self.ticks}

    def close(self) -> None:
        self.mpc = None
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass


def _route_to_wire(route) -> dict:
    """ReachRoute (frozen dataclass, numpy fields) -> plain msgpack-able dict."""
    return {
        "joint_names": list(route.joint_names),
        "route_q": np.asarray(route.route_q_curobo, dtype=float).tolist(),
        "interpolation_dt": float(route.interpolation_dt),
        "reaches": {
            frame: {
                "target": np.asarray(target, dtype=float).tolist(),
                "cands": np.asarray(cands, dtype=float).tolist(),
                "err": float(err),
            }
            for frame, (target, cands, err) in route.reaches.items()
        },
        "controlled_joints": list(route.controlled_joints),
        "base_pos": np.asarray(route.base_pos, dtype=float).tolist(),
        "base_quat": np.asarray(route.base_quat, dtype=float).tolist(),
        "assignment": dict(route.assignment),
    }


# A CUDA context that has taken an `unspecified launch failure` (or any other
# sticky fault) is poisoned for the lifetime of the PROCESS: every later call
# raises identically. Staying up in that state is the worst outcome — the
# process still answers a pgrep and a GPU-free ping, so the robot side sees a
# healthy server and burns a hardware run per attempt (observed 2026-07-29).
# So: probe the GPU on every ping, and on a fatal fault reply once with the
# traceback and then EXIT, making the death visible to whatever supervises us.
_CUDA_FATAL = ("CUDA error", "AcceleratorError", "cudaErrorLaunchFailure",
               "CUDA_ERROR", "device-side assert")


def _is_cuda_fatal(text: str) -> bool:
    return any(k in text for k in _CUDA_FATAL)


def _gpu_healthy() -> tuple[bool, str]:
    """Touch the context with a trivial kernel; cheap, and it fails loudly on a
    poisoned context — which is exactly what a liveness probe must detect."""
    try:
        import torch
        torch.zeros(1, device="cuda").add_(1).sum().item()
        return True, ""
    except Exception as exc:  # noqa: BLE001 — any failure means unusable
        return False, f"{type(exc).__name__}: {exc}"


def _handle_plan0(session, req: dict) -> dict:
    scene_dict = {"cuboid": {
        name: {"dims": [float(v) for v in box["dims"]],
               "pose": [float(v) for v in box["pose"]]}
        for name, box in req["scene"]["cuboid"].items()
    }}
    targets = {name: np.asarray(pos, dtype=float)
               for name, pos in req["targets"].items()}
    seed_q = req.get("seed_q")
    if seed_q is not None:
        seed_q = {str(k): float(v) for k, v in seed_q.items()}
    assignment = req.get("assignment")
    if assignment is not None:
        assignment = {str(k): str(v) for k, v in assignment.items()}
    # The posture the retract should return to, cspace, by joint name. Absent =>
    # planning_home, which is where the arm used to end up regardless of where
    # the mission began. (Authored on the previous GPU host 07-30 with fcb7d8b's client
    # half; folded back into this authoritative copy 2026-08-01 — it had only
    # ever lived in the deployed tree.)
    home_q = req.get("home_q")
    if home_q is not None:
        home_q = {str(k): float(v) for k, v in home_q.items()}
    t0 = time.monotonic()
    route = solve_reach_route(
        session, scene_dict, targets,
        np.asarray(req["base_pos"], dtype=float),
        np.asarray(req["base_quat"], dtype=float),
        seed_q=seed_q, fixed_assignment=assignment,
        goal=str(req.get("goal", "reach")),
        home_q=home_q,
    )
    solve_s = time.monotonic() - t0
    return {"kind": "plan0", "gen": int(req["gen"]), "solve_s": solve_s,
            "err": None, "route": None if route is None else _route_to_wire(route)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--hand-z-floor", type=float, default=None,
                    help="base-frame z below which the reaching hand's gripper "
                         "spheres are pushed back up (metres). There is ALREADY "
                         "a floor — build_curobo_planner defaults to +0.060 m at "
                         "weight 1e6 — so this MOVES it, typically to the top of "
                         "the surface being worked over plus ~0.023 for the "
                         "sphere radius. Baked into the warm CUDA graph, fixed "
                         "for the session.")
    ap.add_argument("--no-hand-z-floor", action="store_true",
                    help="remove the floor entirely (a cube on the ground is "
                         "unreachable while any positive floor is active)")
    ap.add_argument("--hand-floor-weight", type=float, default=None,
                    help="override the floor cost weight; omit to keep the "
                         "upstream 1e6")
    args = ap.parse_args()

    print(f"[server] building cuRobo session ({ROBOT}) …", flush=True)
    cfg = CFG_BY_ROBOT[ROBOT]
    up = upstream_floor_default()
    if args.no_hand_z_floor:
        effective_floor = None
    else:
        effective_floor = args.hand_z_floor if args.hand_z_floor is not None else up
    print(f"[server] hand z-floor: {effective_floor} "
          f"({'flag' if args.no_hand_z_floor or args.hand_z_floor is not None else 'upstream default'}"
          f"; upstream default {up})", flush=True)
    session = make_patched_session(cfg, args.hand_z_floor, args.hand_floor_weight,
                                   args.no_hand_z_floor)

    # Warm the pose CUDA-graph on the perception-free nominal world before
    # accepting requests — same rationale and same helper as the upstream worker:
    # only the graph SHAPE matters, never the obstacle values, and a failed
    # nominal solve still captures the graph.
    print("[server] warming pose CUDA-graph on the nominal world …", flush=True)
    t0 = time.monotonic()
    try:
        warm_scene, warm_targets = _nominal_warm_scene_and_targets(cfg)
        solve_reach_route(session, warm_scene, warm_targets,
                          cfg.home_base_pos, cfg.home_base_quat, goal="reach")
        print(f"[server] warmed in {time.monotonic() - t0:.1f}s", flush=True)
    except Exception as warm_error:  # noqa: BLE001 — warm failure is non-fatal by design
        print(f"[server] nominal warm failed (non-fatal, graph likely captured): "
              f"{type(warm_error).__name__}: {warm_error}", flush=True)

    mpc: MpcSession | None = None      # at most one live tracker per server
    addr = f"tcp://0.0.0.0:{args.port}"
    with pynng.Rep0(listen=addr) as sock:
        # SEND is bounded (2026-08-01). A client that Ctrl+C's mid-solve is
        # gone by the time its reply is ready; an unbounded send on a dead
        # Rep0 peer can park this loop forever, and a parked loop looks to
        # every NEXT client like "plan server unreachable". The reply is
        # worthless once its requester is gone — bound the send, drop it, and
        # take the next request. (recv stays unbounded: idle is normal.)
        sock.send_timeout = 5000
        print(f"[server] listening {addr}  (robot={ROBOT}, Ctrl+C to stop)", flush=True)
        while True:
            raw = sock.recv()          # blocks; Ctrl+C raises KeyboardInterrupt
            fatal = None
            try:
                req = msgpack.unpackb(raw)
                kind = req.get("kind")
                if kind == "ping":
                    ok, why = _gpu_healthy()
                    # The EFFECTIVE floor, not the flag: the client cannot infer
                    # it, and reporting the flag would say "None" for the very
                    # common case of an active +0.060 m default. A floor above
                    # the surface being worked over is a silent INFEASIBLE
                    # generator, so the tool needs the real number to check.
                    rep = {"kind": "pong", "robot": ROBOT, "warmed": True,
                           "gpu_ok": ok, "gpu_err": why,
                           "hand_z_floor": effective_floor}
                    if not ok:
                        fatal = f"GPU health probe failed: {why}"
                elif kind == "plan0":
                    goal = req.get("goal", "reach")
                    want = req.get("hand_z_floor")
                    if want is not None and (
                            effective_floor is None
                            or abs(float(want) - effective_floor) > FLOOR_REBUILD_EPS_M):
                        # The caller measured the work surface; that beats a
                        # launch-time guess. The value is baked into the warm CUDA
                        # graph, so honouring it means rebuilding the session —
                        # ~4 s, at most once per run, and only when the number
                        # actually changed. Cheaper than a human reading a
                        # printout and restarting the server by hand, which is
                        # what this replaces.
                        print(f"[server] hand z-floor {effective_floor} -> "
                              f"{float(want):+.4f} (measured by the caller) — "
                              f"rebuilding the session", flush=True)
                        if mpc is not None:
                            mpc.close()
                            mpc = None
                        effective_floor = float(want)
                        session = make_patched_session(
                            cfg, effective_floor, args.hand_floor_weight, False)
                    print(f"[server] plan0 gen={req.get('gen')} goal={goal} "
                          f"cuboids={len(req['scene']['cuboid'])} "
                          f"targets={list(req['targets'])}", flush=True)
                    rep = _handle_plan0(session, req)
                    ok = rep["route"] is not None
                    print(f"[server]   -> {'route' if ok else 'INFEASIBLE'} "
                          f"in {rep['solve_s']:.2f}s", flush=True)
                elif kind == "mpc_start":
                    if mpc is not None:
                        mpc.close()
                        # Cleared BEFORE the replacement is built: if
                        # MpcSession.__init__ raises, `mpc` must not keep
                        # pointing at the closed session — a later mpc_step
                        # would pass the no-session guard and die inside
                        # _state_from with a misleading AttributeError instead
                        # of the clean "send mpc_start first" reply.
                        mpc = None
                    # exclude: every cube being grasped leaves the obstacle set
                    # (a target that is also a hard obstacle wedges the hand
                    # short — the finding that made upstream delete MPC from the
                    # sim). Accepts a name or a list of names; legacy single
                    # string unchanged.
                    excl = req.get("exclude")
                    excl = set() if excl is None else \
                        ({excl} if isinstance(excl, str) else set(excl))
                    scene = {"cuboid": {k: v for k, v in
                                        req["scene"]["cuboid"].items()
                                        if k not in excl}}
                    # tracked sides: new `reaching_sides` list, else legacy
                    # single `reaching_side`
                    sides = req.get("reaching_sides") or [req["reaching_side"]]
                    cubes_init = None
                    if req.get("cubes"):
                        cubes_init = {s.upper(): (c["pos"], c["quat"])
                                      for s, c in req["cubes"].items()}
                    print(f"[server] mpc_start gen={req.get('gen')} sides="
                          f"{sides} cuboids={len(scene['cuboid'])}"
                          f" (excluding {sorted(excl)} — a grasp target must "
                          f"not be an obstacle to its own grasp)", flush=True)
                    mpc_floor = (None if effective_floor is None else
                                 float(effective_floor) - MPC_FLOOR_FINGER_DROP_M)
                    mpc = MpcSession(
                        cfg, session.robot_cfg_dict, scene,
                        sides, req["seed_q"],
                        mpc_floor, args.hand_floor_weight or 1e6,
                        int(req.get("warm_iters", MPC_WARM_ITERS)),
                        req.get("grasp_index", 0),
                        cubes_init=cubes_init)
                    rep = {"kind": "mpc_start", "gen": req.get("gen"), "err": None,
                           "joint_names": mpc.joint_names,
                           "action_dt": MPC_OPTIMIZATION_DT,
                           "tracked": list(mpc.tracked),
                           "grasp_index": dict(mpc.grasp_index),
                           "warm_s": mpc.warm_s}
                    print(f"[server]   -> MPC warm in {mpc.warm_s:.1f}s, "
                          f"{len(mpc.joint_names)} joints, tracked "
                          f"{mpc.tracked}, candidates {mpc.grasp_index}",
                          flush=True)
                elif kind == "mpc_step":
                    if mpc is None:
                        rep = {"kind": "mpc_step", "gen": req.get("gen"),
                               "err": "no MPC session — send mpc_start first"}
                    else:
                        # new form: {"cubes": {side: {"pos": [3], "quat": [4]}}};
                        # legacy single cube_pos/cube_quat maps onto the (then
                        # single) tracked side.
                        if req.get("cubes"):
                            cubes = {s.upper(): (c["pos"], c["quat"])
                                     for s, c in req["cubes"].items()}
                        else:
                            cubes = {mpc.tracked[0]:
                                     (req["cube_pos"], req["cube_quat"])}
                        out = mpc.step(cubes, req["state_q"])
                        rep = {"kind": "mpc_step", "gen": req.get("gen"),
                               "err": None, **out}
                elif kind == "mpc_stop":
                    if mpc is not None:
                        print(f"[server] mpc_stop after {mpc.ticks} ticks",
                              flush=True)
                        mpc.close()
                        mpc = None
                    rep = {"kind": "mpc_stop", "gen": req.get("gen"), "err": None}
                else:
                    rep = {"kind": "error", "err": f"unknown kind {kind!r}"}
            except Exception:  # noqa: BLE001 — always reply; Rep0 wedges otherwise
                tb = traceback.format_exc()
                # Echo the REQUEST's gen so the client can match this failure to
                # its own call; -1 only when the request was unparseable.
                gen = req.get("gen", -1) if isinstance(locals().get("req"), dict) else -1
                rep = {"kind": "plan0", "gen": gen, "route": None, "solve_s": 0.0,
                       "err": tb}
                print(f"[server] request failed:\n{tb}", flush=True)
                if _is_cuda_fatal(tb):
                    fatal = "unrecoverable CUDA fault during solve"
            try:
                sock.send(msgpack.packb(rep))
            except pynng.exceptions.Timeout:
                # Requester died waiting (Ctrl+C mid-solve). Nobody is owed
                # this reply; dropping it re-arms the socket for the next
                # request instead of wedging the whole server.
                print("[server] reply dropped: requester gone before the "
                      "solve finished", flush=True)
            if fatal:
                # Reply first (so the client gets the traceback, not a timeout),
                # then die: a poisoned context cannot serve, and exiting is the
                # only signal a process-existence check can see.
                print(f"\n[server] FATAL: {fatal}\n"
                      f"[server] the CUDA context is poisoned for this process; "
                      f"exiting so a restart is unambiguous.", flush=True)
                return


if __name__ == "__main__":
    main()
