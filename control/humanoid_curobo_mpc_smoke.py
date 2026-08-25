"""MPC session smoke: start, step inside budget, converge, and TRACK. No robot.

Runs from the robot's own machine over the real WiFi link, so every timing here
is one the control loop will actually pay. It moves nothing — the loop is closed
KINEMATICALLY (the command is fed straight back as the next state), which
isolates the solver's behaviour from any robot dynamics.

WHY THIS IS A COMMITTED TOOL AND NOT A SCRATCH SCRIPT. It measures the reach
error TWICE: once as cuRobo reports it, and once by folding the command to
encoder units and running FK on OUR deploy model. That redundancy is the only
reason the goal-frame swap was ever found — cuRobo's own position_error fell
629 -> 82 mm while the REACHING hand was travelling the wrong way, because the
solver was faithfully reporting the arm it had been told to move. A single
self-reported number would have looked like success right up to the hardware run.

    python humanoid_curobo_mpc_smoke.py            # warm_iters 50 (the default)
    python humanoid_curobo_mpc_smoke.py 200        # slower, no more accurate

WHAT GOOD LOOKS LIKE (measured 2026-07-30, RTX 5080 over WiFi, floor below the
target): converged < 10 mm around tick 45, then holds 2.5-4 mm while the cube
walks at 3 cm/s; wall p50 21.8 / p90 24.2 ms against a 50 ms budget; the two
error measurements agree to well under a millimetre.

IF IT PLATEAUS TENS OF MILLIMETRES SHORT, suspect the hand z-floor before
anything else: the server's default is +0.060 m at weight 1e6, and a target
below that line is pushed away by a cost strong enough to stall the reach while
still looking like healthy convergence. Restart the server with --hand-z-floor
below the target and re-run.
"""
import sys

import humanoid_site as _site
import time

import msgpack
import numpy as np
import pynng

SERVER = _site.PLAN_SERVER                # see humanoid_site / site_local.py
TICKS = 150          # == upstream's N_TICKS: commit-first advances ~one dt/tick
CUBE_DRIFT_MPS = 0.03   # after convergence, walk the cube sideways at 3 cm/s —
                        # the whole point of the tracker. The frozen-route path
                        # aborts at 5 cm of drift; this one should follow.


def rpc(sock, req, timeout_s=180.0):
    sock.send_timeout = int(timeout_s * 1000)
    sock.recv_timeout = int(timeout_s * 1000)
    t0 = time.perf_counter()
    sock.send(msgpack.packb(req))
    rep = msgpack.unpackb(sock.recv())
    return rep, (time.perf_counter() - t0) * 1e3


with pynng.Req0(dial=SERVER) as sock:
    rep, _ = rpc(sock, {"kind": "ping"}, 20.0)
    print("ping:", rep)

    # A plan-0 first: it gives a realistic seed AND tells us the joint names /
    # cspace convention the MPC session must speak.
    scene = {"table": {"dims": [0.6, 1.2, 0.03],
                       "pose": [0.55, 0.0, -0.165, 1.0, 0.0, 0.0, 0.0]}}
    cube = [0.40, 0.18, -0.10]
    rep, ms = rpc(sock, {"kind": "plan0", "gen": 1,
                         "scene": {"cuboid": scene},
                         "targets": {"cube": cube},
                         "base_pos": [0.0, 0.0, 0.0],
                         "base_quat": [1.0, 0.0, 0.0, 0.0],
                         "assignment": None, "seed_q": None, "goal": "reach"})
    if rep.get("err"):
        print("plan0 FAILED:\n", rep["err"][:2000])
        raise SystemExit(1)
    if rep.get("route") is None:
        print(f"plan0 INFEASIBLE in {ms:.0f} ms — retry or move the target")
        raise SystemExit(1)
    route = rep["route"]
    names = route["joint_names"]
    q0 = np.asarray(route["route_q"][0], dtype=float)
    print(f"plan0: {len(route['route_q'])} waypoints, {len(names)} joints, "
          f"{ms:.0f} ms wall, assignment={route['assignment']}")
    seed = {n: float(q0[i]) for i, n in enumerate(names)}
    side = next(iter(route["assignment"]))
    print(f"reaching side: {side}")

    rep, ms = rpc(sock, {"kind": "mpc_start", "gen": 2,
                         "scene": {"cuboid": scene}, "exclude": "cube",
                         "reaching_side": side, "seed_q": seed,
                         "grasp_index": 0,
                         "warm_iters": int(sys.argv[1]) if len(sys.argv) > 1 else 50})
    if rep.get("err"):
        print("mpc_start FAILED:\n", rep["err"][:3000])
        raise SystemExit(1)
    print(f"mpc_start: warm {rep['warm_s']:.1f}s, action_dt {rep['action_dt']}, "
          f"{len(rep['joint_names'])} joints, {ms:.0f} ms wall")

    # Closed KINEMATIC loop: feed the command straight back as the next state.
    # That isolates the SOLVER's convergence from any robot dynamics, which is
    # what a P1 smoke should measure.
    # --- independent measurement on OUR deploy model -------------------------
    # Stop trusting cuRobo's position_error and measure the thing that matters:
    # fold the command to encoder units and FK the EE site ourselves. Same trick
    # as route Gate A, which is what caught the wrist_1 fold bug.
    import mujoco
    from humanoid_curobo_client import fold_cspace_to_enc, ARM_JOINTS_ALL
    from humanoid_model import MJCF_MODEL_PATH
    M = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    D = mujoco.MjData(M)
    QADR = [M.jnt_qposadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS_ALL]
    SITE = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_SITE,
                             f"end_effector_{side}_site")

    def our_ee(q_cspace: dict):
        row = np.asarray([[q_cspace[n] for n in names]], dtype=float)
        enc = fold_cspace_to_enc(row, names, M)[0]
        col = {n: i for i, n in enumerate(names)}
        D.qpos[:] = M.qpos0
        for k2, j in enumerate(ARM_JOINTS_ALL):
            if j in col:
                D.qpos[QADR[k2]] = enc[col[j]]
        mujoco.mj_kinematics(M, D)
        return D.site_xpos[SITE].copy()

    state = dict(seed)
    cube_quat = [1.0, 0.0, 0.0, 0.0]
    walls, solves, errs, ours = [], [], [], []
    live = list(cube)

    converged_at = None
    for k in range(TICKS + 60):
        # Phase 2: once the hand has arrived, start walking the cube sideways.
        if converged_at is not None and k > converged_at:
            live[1] += CUBE_DRIFT_MPS * 0.02   # one action_dt per tick
        rep, ms = rpc(sock, {"kind": "mpc_step", "gen": 10 + k,
                             "state_q": state, "cube_pos": live,
                             "cube_quat": cube_quat}, 30.0)
        if rep.get("err"):
            print(f"mpc_step {k} FAILED:\n", rep["err"][:3000])
            raise SystemExit(1)
        state = rep["q_next"]
        walls.append(ms)
        solves.append(rep["solve_s"] * 1e3)
        e = rep["pos_err"]
        if e is not None:
            errs.append(e)
            if converged_at is None and e < 0.01:
                converged_at = k
                print(f"  -> CONVERGED at tick {k} ({e*1000:.2f} mm); now walking "
                      f"the cube at {CUBE_DRIFT_MPS*100:.0f} cm/s")
        ee = our_ee(state)
        d_ours = float(np.linalg.norm(ee - np.asarray(rep["goal_pos"])))
        ours.append(d_ours)
        if k in (0, 5, 19, 49, 99, TICKS - 1) or \
                (converged_at is not None and k in (converged_at + 20,
                                                    converged_at + 59)):
            moved = abs(live[1] - cube[1])
            print(f"  tick {k:3d}: wall {ms:5.1f} ms  solve {rep['solve_s']*1e3:5.1f} ms"
                  f"  curobo_err {e*1000:7.2f} mm  OUR_FK_err {d_ours*1000:7.2f} mm"
                  f"  cube moved {moved*1000:5.1f} mm")
        if converged_at is not None and k >= converged_at + 60:
            break

    w, s = np.asarray(walls[1:]), np.asarray(solves[1:])
    print(f"\nWALL  (incl. WiFi RTT) p50 {np.percentile(w,50):.1f} / "
          f"p90 {np.percentile(w,90):.1f} / max {w.max():.1f} ms   [budget 50]")
    print(f"SOLVE (GPU only)        p50 {np.percentile(s,50):.1f} / "
          f"p90 {np.percentile(s,90):.1f} / max {s.max():.1f} ms")
    if errs:
        print(f"curobo position_error {errs[0]*1000:.1f} -> min {min(errs)*1000:.2f} "
              f"-> last {errs[-1]*1000:.2f} mm")
    if ours:
        # Deferred, and read rather than copied: this print exists to tell the
        # operator whether the run would have grasped, so a stale literal here
        # is a lie about the mission. The operator module is heavy and this
        # smoke tool must stay importable without it, hence the local import.
        from humanoid_auto_operator import GRASP_EE_OK_M  # noqa: PLC0415

        print(f"OUR model FK->goal   {ours[0]*1000:.1f} -> min {min(ours)*1000:.2f} "
              f"-> last {ours[-1]*1000:.2f} mm   <- the number that decides a grasp "
              f"(GRASP_EE_OK_M = {GRASP_EE_OK_M*1000:.0f} mm)")
        print(f"cube ended {abs(live[1]-cube[1])*1000:.0f} mm from where it was "
              f"planned — the frozen-route path aborts at 50 mm")

    rep, _ = rpc(sock, {"kind": "mpc_stop", "gen": 99})
    print("mpc_stop:", rep)
