"""Client for the remote cuRobo plan server — protocol, units seam, route gates.

ARCHITECTURE. The GPU machine runs `curobo_plan_server.py` (one warm cuRobo
session behind pynng Rep0). This module is the robot-side counterpart: it
speaks the msgpack protocol, folds the reply into ENCODER units, time-stretches
it to a streamable rate, and — before anyone moves a motor — runs four gates
against OUR deploy model. The design rule inherited from the whole stack:
cuRobo's model says "collision-free"; only our own model is allowed to say
"safe to execute".

THE UNITS SEAM (the silent-90°-bug class, spelled out once):

    enc[j] = SIGN[j] * cspace[j] + qpos0[j]        (fold, server -> robot)
    cspace[j] = SIGN[j] * (enc[j] - qpos0[j])      (seed, robot -> server)

  * qpos0: our deploy model carries `ref` on wrist_1 (left -pi/2, right +pi/2,
    rebuild_deploy_model.py layer 3). Upstream — and therefore the cuRobo cfg,
    which is built from the upstream URDF — moved the wrist_1 zero into the
    geometry instead, so cuRobo cspace zero == CAD pose == our (qpos - qpos0).
    This is the exact fold the upstream executor itself uses (`_route_to_mjlab`:
    route + qpos0; `_seed_from_proprio`: qpos - qpos0).
  * SIGN: upstream turned left_wrist_2's axis to -x; the physical motor was
    never rewired, so our deploy model flips it back to +x (layer 4). Same
    physical rotation, opposite qpos sign => SIGN[left_wrist_2_joint] = -1.
    The upstream seam has no sign term because its mjlab model is the
    unmodified one.
  qpos0 is read from the loaded model, never hardcoded — the model is the
  truth; tests/test_curobo_bridge.py pins the expected values so drift in
  either direction fails a test instead of a grasp.

THE GATES (verify_route), each catching a distinct failure class:
  A  seam/FK    — FK of the folded final waypoint must land the active tool
                  site on a grasp candidate (reach) / the home pose (home).
                  A wrong sign or offset lands centimeters-to-decimeters off;
                  kinematics are identical between the models, so the true
                  tolerance is millimeters and 20 mm is generous.
  B  start      — folded route[0] must equal the measured joints the seed came
                  from. Catches server-side seed scrambling (the "arm flails"
                  class upstream documents at solve_reach_route's seed builder).
  C  clearance  — mj_geomDistance sweep of every waypoint on OUR model, arm-vs
                  -body and arm-vs-arm, structurally-adjacent pairs excluded.
                  Reuses humanoid_joint_monkey_hw.preflight verbatim — the same
                  auditor the hardware joint monkey trusts.
  D  limits     — every waypoint inside our model's joint ranges minus margin.
Gate E (stream continuity) lives in `stretch()`: after time-stretching, the
per-tick delta is <= rate*dt by construction and is asserted anyway.

WORLD MODEL NOTE. Scene cuboids and targets are BASE-frame; we pass
base_quat = identity and rely on the operator-stack tilt gate (<8 deg warn /
15 deg refuse) that every mission already runs — at those tilts the goalset's
45-75 deg approach spread absorbs the gravity misalignment. Revisit if grasping
while badly tilted ever becomes a requirement.
"""

from __future__ import annotations

import dataclasses

import msgpack
import mujoco
import numpy as np
import pynng

from humanoid_joint_monkey_hw import ARM_JOINTS_L, ARM_JOINTS_R, preflight
from humanoid_model import MJCF_MODEL_PATH

ARM_JOINTS_ALL = list(ARM_JOINTS_L) + list(ARM_JOINTS_R)

# Our-encoder sign vs cuRobo cspace. ONLY left_wrist_2 (deploy-model axis
# flip, rebuild_deploy_model.py layer 4). Everything else is +1.
CSPACE_SIGN = {"left_wrist_2_joint": -1.0}

DEFAULT_PORT = 9880
PLAN_TIMEOUT_S = 90.0     # bimanual plan-0 at 30 attempts can be slow; generous
PING_TIMEOUT_S = 5.0
MPC_STEP_TIMEOUT_S = 0.4  # one receding-horizon tick measured p90 24 ms over
                          # WiFi; 0.4 s is ~16x that. The bound is real_env's
                          # 0.5 s arm-silence failsafe: the step RPC blocks the
                          # publisher thread, so any stall longer than the
                          # failsafe window starts the crawl-toward-default.
                          # 0.4 s keeps even a timed-out tick inside the
                          # window; the timeout becomes a PlanServerError and
                          # feeds the caller's consecutive-failure abort.
                          # (Was 2.0 — review 2026-08-01: a stall in the
                          # (0.5, 2.0] band engaged the failsafe invisibly.)

# Gate tolerances (units in the names).
GATE_A_FK_TOL_M = 0.02
GATE_B_START_TOL_RAD = 0.05
GATE_C_FLOOR_M = 0.010    # default clearance floor; staircase audit bound is
                          # 0.0289 but cuRobo routes may legitimately thread
                          # closer — tunable per call, never below zero
GATE_D_LIMIT_MARGIN_RAD = 0.02
GATE_F_WINDUP_RAD = 0.2            # a reach route whose shoulder_1 swings
                                   # AWAY from the cubes (left more negative
                                   # / right more positive than route[0]) by
                                   # more than this is the WIND-UP solution
                                   # family — measured 2026-08-02: ~1/3 of
                                   # solves for front targets wind one
                                   # shoulder 0.97 rad away and reach through
                                   # a different joint path, and the user's
                                   # field logs show those executions
                                   # systematically end in the regrasp path.
                                   # The direct family measures 0.00 away.
                                   # 0.3 -> 0.2 (hardware 2026-08-05, three
                                   # runs same session): a boundary gate on a
                                   # dominant basin turns randomized restarts
                                   # into a sampler that accepts the
                                   # SHALLOWEST windup — rejected draws read
                                   # 0.44-0.66, both accepted routes read
                                   # 0.2888 and 0.29987, hugging the 0.3 tol
                                   # from below by 1-11 mrad. Both aborted
                                   # T12: the windup detour left the hand
                                   # ~150 mm out at the 80% MPC handoff, and
                                   # the remaining sweep at the architecture's
                                   # ~0.22 rad/s ceiling read as a stall. The
                                   # zero-windup run (1.2e-07) converged in
                                   # 3.5 s and grasped. Cost of tightening:
                                   # more restarts on midline-hugging
                                   # placements — a visible plan-time refusal
                                   # instead of a hardware abort mid-reach.
                                   # Front-sector assumption: for rear/odd
                                   # sectors revisit the sign.
GATE_E_HOME_BRANCH_TOL_RAD = 0.35  # a home route must end at the home
                                   # POSTURE, not just the home hand pose.
                                   # goal="home" is an EE-pose goal server-
                                   # side (plan_pose on the FK of home_q), so
                                   # the solver is free to pick any IK branch
                                   # that puts the hands home — and on
                                   # hardware 2026-08-01 night it parked the
                                   # left shoulder in a visibly different
                                   # branch after a successful carry-home.
                                   # Branch flips measure ~1 rad (the
                                   # historical wrist example: 0.98); same-
                                   # branch null-space wander stays well
                                   # under this. Rejected attempts get the
                                   # randomized-restart retry, and the
                                   # measured-trail return backstops a route
                                   # that never matches.


def _qpos0_by_name(model) -> dict[str, float]:
    out = {}
    for name in ARM_JOINTS_ALL:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"deploy model lacks joint {name}"
        out[name] = float(model.qpos0[model.jnt_qposadr[jid]])
    return out


def fold_cspace_to_enc(route_q: np.ndarray, joint_names: list[str], model) -> np.ndarray:
    """(H, n) cuRobo cspace -> encoder qpos units on OUR deploy model."""
    qpos0 = _qpos0_by_name(model)
    sign = np.array([CSPACE_SIGN.get(n, 1.0) for n in joint_names])
    off = np.array([qpos0[n] for n in joint_names])
    return np.asarray(route_q, dtype=float) * sign[None, :] + off[None, :]


def enc_to_cspace_seed(measured_enc: dict[str, float], model) -> dict[str, float]:
    """{joint: encoder rad} -> cuRobo cspace seed dict (the request's seed_q)."""
    qpos0 = _qpos0_by_name(model)
    return {n: CSPACE_SIGN.get(n, 1.0) * (float(measured_enc[n]) - qpos0[n])
            for n in ARM_JOINTS_ALL}


@dataclasses.dataclass(frozen=True)
class RouteBundle:
    """A decoded, ENCODER-unit, time-parameterised route ready to gate and stream."""
    joint_names: tuple[str, ...]         # column order of q_enc
    q_enc: np.ndarray                    # (H, n) encoder qpos units
    dt: float                            # per-waypoint spacing AFTER any stretch
    reaches: dict                        # {frame: {"target": (3,), "cands": (G,3), "err": float}}
    controlled_joints: tuple[str, ...]   # the active arms' joints — the only columns we stream
    assignment: dict                     # {side: cube_name}
    goal: str                            # "reach" | "home"
    solve_s: float

    @property
    def horizon(self) -> int:
        return int(self.q_enc.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.horizon - 1) * self.dt

    @property
    def active_sides(self) -> tuple[str, ...]:
        """("left",), ("right",) or ("left", "right") — the arms this route
        actually commands, read off the controlled_joints prefixes."""
        sides = []
        for side, prefix in (("left", "left_"), ("right", "right_")):
            if any(j.startswith(prefix) for j in self.controlled_joints):
                sides.append(side)
        return tuple(sides)

    def max_joint_rate(self) -> float:
        """Fastest per-tick joint step over the whole route, in encoder rad/s
        (0.0 for a route shorter than two waypoints)."""
        if self.horizon < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(self.q_enc, axis=0))) / self.dt)

    def stretch(self, max_rate: float) -> "RouteBundle":
        """Uniform time stretch so no joint exceeds `max_rate` [rad/s].

        Path SHAPE is untouched — only dt grows — so the collision certificate
        transfers verbatim. Gate E: by construction the fastest joint moves at
        exactly `max_rate` after a stretch, asserted below."""
        rate = self.max_joint_rate()
        if rate <= max_rate or self.horizon < 2:
            return self
        factor = rate / max_rate
        out = dataclasses.replace(self, dt=self.dt * factor)
        assert out.max_joint_rate() <= max_rate * 1.001
        return out

    def q_at(self, t_s: float) -> np.ndarray:
        """Linear interpolation along the stretched timeline, clamped at the ends."""
        if t_s <= 0.0:
            return self.q_enc[0]
        idx = t_s / self.dt
        lo = int(idx)
        if lo >= self.horizon - 1:
            return self.q_enc[-1]
        frac = idx - lo
        return (1.0 - frac) * self.q_enc[lo] + frac * self.q_enc[lo + 1]


def _decode_route(wire: dict, model, goal: str, solve_s: float) -> RouteBundle:
    names = [str(n) for n in wire["joint_names"]]
    q_enc = fold_cspace_to_enc(np.asarray(wire["route_q"], dtype=float), names, model)
    reaches = {
        str(frame): {"target": np.asarray(r["target"], dtype=float),
                     "cands": np.asarray(r["cands"], dtype=float),
                     "err": float(r["err"])}
        for frame, r in wire["reaches"].items()
    }
    return RouteBundle(
        joint_names=tuple(names), q_enc=q_enc,
        dt=float(wire["interpolation_dt"]), reaches=reaches,
        controlled_joints=tuple(str(j) for j in wire["controlled_joints"]),
        assignment={str(k): str(v) for k, v in wire["assignment"].items()},
        goal=goal, solve_s=solve_s,
    )


class PlanServerError(RuntimeError):
    """Server reached but the request failed (solver exception, bad message)."""


class PlanServerDeadError(PlanServerError):
    """The server process is up but its CUDA context is unrecoverable.

    `unspecified launch failure` and friends poison a CUDA context for the
    lifetime of the process: every later call fails identically, so retrying is
    pointless and a liveness probe that does not touch the GPU (a plain pgrep,
    or a ping that only echoes a dict) reports a healthy server. Raised as its
    own type so callers can say "restart the server" instead of surfacing a
    driver traceback."""


_CUDA_FATAL = ("CUDA error", "AcceleratorError", "cudaErrorLaunchFailure",
               "CUDA_ERROR", "device-side assert")


def _classify(err: str) -> PlanServerError:
    if any(k in err for k in _CUDA_FATAL):
        return PlanServerDeadError(
            "the plan server's CUDA context is dead — the process is still up "
            "but every solve will now fail. Restart it on the GPU machine, from "
            "a checkout of this repository (docs/OPERATIONS.md section 1):\n"
            "    ssh <gpu-host> 'pkill -f \"plan_serve[r]\"'\n"
            "    ssh <gpu-host> 'cd <repo> && "
            "HUMANOID_LEGGED_ENV_ROOT=<legged_env_v2> setsid nohup "
            "python control/curobo_plan_server.py --port 9880 "
            "> plan_server.log 2>&1 </dev/null & exit 0'\n"
            "then wait ~40 s for 'listening' in plan_server.log. Keep those\n"
            "two ssh calls SEPARATE: a pkill pattern spelled out beside the\n"
            "relaunch matches its own command line and kills the shell first.\n"
            f"--- server traceback ---\n{err}")
    return PlanServerError(err)


class PlanClient:
    """Blocking Req0 client. One in-flight request at a time; replies carry the
    request's generation tag and anything stale is dropped, mirroring the upstream
    SpawnedReachMpcWorker poll semantics."""

    def __init__(self, addr: str, model=None):
        self.addr = addr
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        self._sock = pynng.Req0(dial=addr, block_on_dial=False)
        self._gen = 0

    def close(self) -> None:
        self._sock.close()

    def _rpc(self, req: dict, timeout_s: float) -> dict:
        self._sock.send_timeout = int(timeout_s * 1000)
        self._sock.recv_timeout = int(timeout_s * 1000)
        # Transport failures surface as PlanServerError HERE, at the one seam
        # every RPC shares. A raw pynng.exceptions.Timeout is not a
        # PlanServerError, so it used to skip every caller's failure handling —
        # mpc_track's consecutive-failure counter, plan_reach's refuse path —
        # and crash the mission mid-approach instead (review 2026-08-01).
        # Req0 recovers cleanly from a timed-out recv: the protocol cancels the
        # outstanding request and the socket accepts the next send.
        try:
            self._sock.send(msgpack.packb(req))
            raw = self._sock.recv()
        except pynng.exceptions.NNGException as e:
            raise PlanServerError(
                f"transport failure on {req.get('kind')!r} "
                f"({type(e).__name__}: {e}) — server unreachable or the reply "
                f"exceeded {timeout_s:.1f}s") from e
        return msgpack.unpackb(raw)

    def ping(self, timeout_s: float = PING_TIMEOUT_S) -> dict:
        """Liveness round trip. Returns the server's pong dict (robot name,
        hand_z_floor, gpu_ok/gpu_err); raises PlanServerError on transport
        failure or a reply that is not a pong."""
        rep = self._rpc({"kind": "ping"}, timeout_s)
        if rep.get("kind") != "pong":
            raise PlanServerError(f"unexpected ping reply: {rep}")
        return rep

    def plan(self, scene_cuboids: dict, targets: dict, *,
             measured_enc: dict[str, float] | None = None,
             assignment: dict | None = None, goal: str = "reach",
             base_pos=(0.0, 0.0, 0.0), base_quat=(1.0, 0.0, 0.0, 0.0),
             hand_z_floor: float | None = None,
             home_enc: dict[str, float] | None = None,
             timeout_s: float = PLAN_TIMEOUT_S) -> RouteBundle | None:
        """One plan-0 round trip. Returns None on a clean INFEASIBLE verdict;
        raises PlanServerError on transport/solver failure.

        `scene_cuboids` = {name: {"dims": [3], "pose": [x,y,z,qw,qx,qy,qz]}},
        base frame. `targets` = {cube_name: [x,y,z]} base frame. `measured_enc`
        (encoder rad by joint name) becomes the cspace seed; REQUIRED for
        goal="home" (the retract must start where the arm is).

        `hand_z_floor` is the MEASURED work surface, and passing it is how the
        floor stops being a launch-time guess: the server rebuilds its session
        when the value changes (~4 s, at most once a run). The registered table
        geometry already knows this number — see hand_floor_for_tables — so a
        human should never have to read it off a printout and restart anything.

        `home_enc` (encoder rad by joint name, goal="home" only) is the posture
        the retract should RETURN TO. Without it the server aims at the FK of
        its own planning_home. When this field was added (hardware 2026-07-30,
        first successful grasp) that posture was one the commander had never
        visited — 0.98 rad from the chest home the mission started at — and
        the arm parked 56 deg from home. Since 07-31 the homes are UNIFIED
        (planning_home == power-on == chest home == the sim's
        HUMANOID_ARM_JOINT_HOME), so on a fresh mission this field merely
        confirms what the server would do anyway. It is NOT dead: mid-mission
        the return target is the MEASURED mission start, not the nominal home
        (a round-2 retract must not "return" to round 1's failure pose), and
        if either side's home constant ever moves alone, this field is what
        keeps the retract honest while the guard tests catch the drift.
        """
        assert goal in ("reach", "home")
        if goal == "home":
            assert measured_enc is not None, "retract requires the measured seed"
        seed = None if measured_enc is None else enc_to_cspace_seed(measured_enc, self.model)
        self._gen += 1
        req = {"kind": "plan0", "gen": self._gen,
               "scene": {"cuboid": scene_cuboids},
               "targets": {k: [float(x) for x in v] for k, v in targets.items()},
               "base_pos": [float(x) for x in base_pos],
               "base_quat": [float(x) for x in base_quat],
               "assignment": assignment, "seed_q": seed, "goal": goal,
               "hand_z_floor": None if hand_z_floor is None else float(hand_z_floor),
               "home_q": None if home_enc is None else
                         enc_to_cspace_seed(home_enc, self.model)}
        rep = self._rpc(req, timeout_s)
        # No stale-reply drain here, deliberately. Req0 correlates request and
        # reply IN THE PROTOCOL — one outstanding request at a time, and recv()
        # returns that request's reply or times out. An earlier draft carried a
        # "drop the stale reply and recv() again" loop ported from the
        # multiprocessing-queue worker, where a shared results queue genuinely
        # could hold an orphaned reply. Over Req0 that second recv() is illegal
        # (the socket is back in send state -> pynng BadState), and because the
        # loop ran BEFORE the error check it fired on the server's own error
        # replies (which carry gen=-1) and replaced a precise server traceback
        # with "BadState: Incorrect state". Cost one hardware run, 2026-07-29.
        if rep.get("err"):
            raise _classify(rep["err"])
        if rep.get("gen", self._gen) != self._gen:
            raise PlanServerError(
                f"reply gen {rep.get('gen')} != request gen {self._gen} — "
                f"protocol desync; restart the client")
        if rep.get("route") is None:
            return None
        return _decode_route(rep["route"], self.model, goal, float(rep.get("solve_s", 0.0)))

    # -- real-time tracking (MPC session) -------------------------------------
    # The units seam is honoured in BOTH directions here, with the same helpers
    # plan-0 uses: encoder state folds to cspace via enc_to_cspace_seed on the
    # way out, and the commanded step folds back via fold_cspace_to_enc on the
    # way in. Doing the fold anywhere else would be a second, divergent seam.

    def mpc_start(self, scene_cuboids: dict, *, exclude, reaching_sides,
                  measured_enc: dict[str, float],
                  cubes_init: dict[str, "np.ndarray"] | None = None,
                  grasp_index=-1, warm_iters: int = 50,
                  timeout_s: float = PLAN_TIMEOUT_S) -> dict:
        """Open (replacing any prior) MPC session tracking `reaching_sides`.

        `exclude` — cube name(s) leaving the obstacle set (the cubes being
        grasped). `cubes_init` = {side: pos[3]} initial cube positions, needed
        by the server's AUTO grasp-candidate pick (grasp_index=-1): the session
        starts from the posture plan-0 ended in, and AUTO keeps the tracked
        goal on the candidate that posture already committed to.
        """
        self._gen += 1
        req = {"kind": "mpc_start", "gen": self._gen,
               "scene": {"cuboid": scene_cuboids},
               "exclude": list(exclude) if not isinstance(exclude, str) else exclude,
               "reaching_sides": [s.upper() for s in reaching_sides],
               "seed_q": enc_to_cspace_seed(measured_enc, self.model),
               "grasp_index": grasp_index,
               "warm_iters": int(warm_iters)}
        if cubes_init:
            req["cubes"] = {s.upper(): {"pos": [float(x) for x in p],
                                        "quat": [1.0, 0.0, 0.0, 0.0]}
                            for s, p in cubes_init.items()}
        rep = self._rpc(req, timeout_s)
        if rep.get("err"):
            raise _classify(rep["err"])
        return rep

    def mpc_step(self, measured_enc: dict[str, float],
                 cube_pos_by_side: dict[str, "np.ndarray"],
                 timeout_s: float = MPC_STEP_TIMEOUT_S) -> dict:
        """One tracking tick: measured encoders + live per-side cube positions
        in, ENCODER-unit commanded step out.

        Cube orientation is sent as identity — the same convention plan-0's
        position-only targets already assume, so the tracked goal and the
        planned goal cannot disagree about the cube's frame.

        Returns {"enc": {joint: rad}, "goals": {side: [3]}, "solve_s": float}.
        """
        self._gen += 1
        req = {"kind": "mpc_step", "gen": self._gen,
               "state_q": enc_to_cspace_seed(measured_enc, self.model),
               "cubes": {s.upper(): {"pos": [float(x) for x in p],
                                     "quat": [1.0, 0.0, 0.0, 0.0]}
                         for s, p in cube_pos_by_side.items()}}
        rep = self._rpc(req, timeout_s)
        if rep.get("err"):
            raise _classify(rep["err"])
        q_next = rep["q_next"]
        names = list(q_next.keys())
        row = np.asarray([[float(q_next[n]) for n in names]])
        enc = fold_cspace_to_enc(row, names, self.model)[0]
        return {"enc": {n: float(enc[i]) for i, n in enumerate(names)},
                "goals": rep.get("goals", {}),
                "solve_s": float(rep.get("solve_s", 0.0))}

    def mpc_stop(self, timeout_s: float = MPC_STEP_TIMEOUT_S) -> None:
        """Best-effort session teardown. Never raises: this runs in `finally`
        blocks and on Ctrl+C paths, where a transport error must not mask the
        original exit reason. The timeout is the SHORT one on purpose — this
        runs unpumped on the publisher thread, and a dead server must not buy
        seconds of arm silence for a teardown the next mpc_start performs
        anyway (the server closes any prior session on start)."""
        self._gen += 1
        try:
            self._rpc({"kind": "mpc_stop", "gen": self._gen}, timeout_s)
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass


# ------------------------------------------------------------------------------
# Gates
# ------------------------------------------------------------------------------

class _FrameShim:
    """Duck-typed stand-in for joint_monkey's Trajectory: preflight only reads .frames."""
    def __init__(self, frames):
        self.frames = frames


def merge_decoupled_bundles(per_side: dict) -> RouteBundle:
    """Combine two SINGLE-ARM solves into one simultaneous 14-joint route.

    WHY (user 2026-08-04): the joint dual-arm solve couples the arms through
    one cost landscape, and some placements (front-left + rear-right) trade
    the arms off into a solution-family lottery the user experienced as a
    deadlock. Solving each arm ALONE removes the coupling; this merge keeps
    the SIMULTANEOUS execution the user requires. Each arm's columns come
    from ITS OWN solve (whatever the solver did with the idle peer arm in a
    single-target problem is discarded), the shorter route is padded at its
    final posture, and the caller MUST re-gate the merged route — the C
    sweep on the merged trajectory is the coupled safety check that replaces
    the joint solver's internal mutual avoidance."""
    bl, br = per_side["L"], per_side["R"]
    if bl.joint_names != br.joint_names:
        raise ValueError("decoupled solves returned different joint orders")
    if abs(bl.dt - br.dt) > 1e-9:
        raise ValueError(f"decoupled solves returned different dt "
                         f"({bl.dt} vs {br.dt}) — cannot time-align")
    horizon = max(bl.horizon, br.horizon)

    def padded(b: RouteBundle) -> np.ndarray:
        if b.horizon == horizon:
            return b.q_enc
        pad = np.repeat(b.q_enc[-1:, :], horizon - b.horizon, axis=0)
        return np.vstack([b.q_enc, pad])

    q = padded(bl).copy()
    qr = padded(br)
    right_cols = [i for i, j in enumerate(bl.joint_names)
                  if j.startswith("right_")]
    q[:, right_cols] = qr[:, right_cols]
    controlled = tuple(j for j in bl.joint_names
                       if j in set(bl.controlled_joints)
                       | set(br.controlled_joints))
    return RouteBundle(
        joint_names=bl.joint_names, q_enc=q, dt=bl.dt,
        reaches={**bl.reaches, **br.reaches},
        controlled_joints=controlled,
        assignment={**bl.assignment, **br.assignment},
        goal="reach", solve_s=bl.solve_s + br.solve_s)


# Per-tick gate defaults. The rate ceiling tracks the receiver's hard cap
# (raised 0.5 -> 0.7 -> 1.05 with the user's arm speed bumps, 2026-08-02) plus
# headroom for one missed tick, because a real-time command is issued against a
# measurement that may be one tick stale: at 20 Hz a fast route legitimately
# shows double its rate in apparent step if a telemetry frame was dropped.
# Above this the command is not a tracking correction, it is a jump.
STEP_MAX_RATE_RAD_S = 1.05
STEP_SEGMENT_POINTS = 3   # collision-check the measured -> commanded SEGMENT, not
                          # just its endpoints: a 0.05 s step can pass through a
                          # thin obstacle with both ends clear.


@dataclasses.dataclass(frozen=True)
class StepVerdict:
    """One per-tick decision. `ok` gates publication of that single command."""
    ok: bool
    why: str                                  # "" when ok; the refusal otherwise
    clearance_m: float = float("nan")
    limit_margin_rad: float = float("nan")
    rate_rad_s: float = float("nan")
    where: tuple[str, str] | None = None      # closest geom pair, for the log


class StepGate:
    """Gate ONE commanded configuration, in the control loop, on OUR model.

    WHY THIS EXISTS SEPARATELY FROM verify_route. verify_route's entire safety
    argument is "the whole route is downloaded and gated before one packet is
    sent" — and that argument CANNOT hold for a real-time path, where the
    command is produced one tick before it executes. This class is the
    replacement contract, not an optimisation of the old one: every commanded
    configuration is checked before publication, and a configuration that fails
    is simply not published (the caller re-sends the last one that passed, and
    escalates to silence — real_env's 0.5 s failsafe — if the condition holds).

    WHY IT IS AFFORDABLE AT 20-50 Hz. preflight() rebuilds its collision-pair
    set on EVERY call: 1640 pairs behind a kinematic-hop filter. That rebuild is
    essentially the whole cost — one frame measures 7.61 ms while mj_kinematics
    itself is 0.006 ms (robot computer, 2026-07-29). Hoisting the set into __init__
    makes one configuration 0.38 ms and the 3-point segment 1.38 ms, which is
    ~3% of a 20 Hz tick. The pair RULE is copied from preflight (> 2 kinematic
    hops apart, "collision" in the geom name) rather than re-invented, so the
    per-tick gate and the route gate cannot disagree about what a collision is.

    WHICH GATES, AND WHY NOT ALL FOUR. Route gate D (limits) and C (clearance)
    apply per configuration and are here. Gate B (route[0] == measured) becomes
    the RATE check: on a real-time path there is no route[0], and the property
    that actually matters is that the command is a small correction from where
    the arm measurably is. Gate A (seam FK) is deliberately NOT per-tick — it
    tests a static property of the fold, so it is verified once per session (by
    a plan-0 route, or by a single FK check at session start); running it every
    tick would only re-prove the same arithmetic.

    THE PEER ARM IS PLACED AT ITS MEASUREMENT, not at a plan. With a live
    optimiser the other arm is wherever it actually is, so the collision-checked
    configuration must be commanded-joints + measured-everything-else. Feeding a
    stale planned peer pose here would gate a robot that does not exist.
    """

    def __init__(self, model=None, *,
                 clearance_floor_m: float = GATE_C_FLOOR_M,
                 limit_margin_rad: float = GATE_D_LIMIT_MARGIN_RAD,
                 max_rate_rad_s: float = STEP_MAX_RATE_RAD_S,
                 segment_points: int = STEP_SEGMENT_POINTS):
        self.model = model if model is not None else \
            mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.clearance_floor_m = float(clearance_floor_m)
        self.limit_margin_rad = float(limit_margin_rad)
        self.max_rate_rad_s = float(max_rate_rad_s)
        self.segment_points = max(2, int(segment_points))
        self._qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS_ALL])
        self._range = np.array([
            self.model.jnt_range[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS_ALL])
        self._pairs = _collision_pairs(self.model)
        assert self._pairs, "collision pair set came up empty — check geom naming"

    # -- the hot path ----------------------------------------------------------
    def check(self, commanded_enc: dict[str, float],
              measured_enc: dict[str, float], dt: float) -> StepVerdict:
        """`commanded_enc` may cover only the driven joints; the rest of the
        robot is placed at `measured_enc`. Both are ENCODER units."""
        q_meas = np.array([float(measured_enc[j]) for j in ARM_JOINTS_ALL])
        q_cmd = q_meas.copy()
        driven = []
        for i, j in enumerate(ARM_JOINTS_ALL):
            if j in commanded_enc:
                q_cmd[i] = float(commanded_enc[j])
                driven.append(i)
        if not driven:
            return StepVerdict(False, "command names no arm joint")
        if not np.all(np.isfinite(q_cmd)):
            return StepVerdict(False, "command contains NaN/inf — dropped")

        # Limits and rate judge the COMMAND, so they are scoped to the driven
        # joints. Applying them to the peer arm — which is here only as measured
        # geometry — would let one un-driven joint veto every command forever,
        # and this hardware makes that concrete: the right arm's measured front
        # posture reads 0.161 rad OUTSIDE the model range, and wrist encoders
        # are known to report past ±pi (they wrap). A stale peer reading must
        # never deadlock the arm that is actually moving. The clearance sweep
        # below is the opposite: it takes the WHOLE robot, peer arm included.
        drv = np.asarray(driven)
        lo, hi = self._range[drv, 0], self._range[drv, 1]
        margins = np.minimum(q_cmd[drv] - lo, hi - q_cmd[drv])
        margin = float(np.min(margins))
        if margin < self.limit_margin_rad:
            k = drv[int(np.argmin(margins))]
            return StepVerdict(False, f"{ARM_JOINTS_ALL[k]} within "
                                      f"{margin:.3f} rad of a limit "
                                      f"(need {self.limit_margin_rad})",
                               limit_margin_rad=margin)

        steps = np.abs(q_cmd[drv] - q_meas[drv])
        rate = float(np.max(steps) / max(dt, 1e-6))
        if rate > self.max_rate_rad_s:
            k = drv[int(np.argmax(steps))]
            return StepVerdict(False, f"{ARM_JOINTS_ALL[k]} step is "
                                      f"{rate:.2f} rad/s (cap "
                                      f"{self.max_rate_rad_s}) — a jump, not a "
                                      f"tracking correction",
                               limit_margin_rad=margin, rate_rad_s=rate)

        worst, where = np.inf, None
        for a in np.linspace(0.0, 1.0, self.segment_points):
            clr, pair = self._clearance(q_meas + (q_cmd - q_meas) * a)
            if clr < worst:
                worst, where = clr, pair
        if worst < self.clearance_floor_m:
            return StepVerdict(False, f"clearance {worst*1000:.1f} mm < floor "
                                      f"{self.clearance_floor_m*1000:.0f} mm "
                                      f"at {where[0]} ~ {where[1]}",
                               clearance_m=worst, limit_margin_rad=margin,
                               rate_rad_s=rate, where=where)
        return StepVerdict(True, "", clearance_m=worst, limit_margin_rad=margin,
                           rate_rad_s=rate, where=where)

    def _clearance(self, q14: np.ndarray) -> tuple[float, tuple[str, str]]:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self._qadr] = q14
        mujoco.mj_kinematics(self.model, self.data)
        worst, where = np.inf, ("", "")
        for g1, g2, n1, n2 in self._pairs:
            d = mujoco.mj_geomDistance(self.model, self.data, g1, g2,
                                       _STEP_DIST_CUTOFF_M, None)
            if d < worst:
                worst, where = d, (n1, n2)
        return float(worst), where


# mj_geomDistance returns the cutoff for anything further than this. Only the
# MINIMUM matters, so the cutoff exists purely to let the narrow-phase bail
# early; it must stay comfortably above the clearance floor.
_STEP_DIST_CUTOFF_M = 0.30


def _collision_pairs(model) -> list[tuple[int, int, str, str]]:
    """preflight's pair rule, hoisted out of the hot path (see StepGate).

    Kept as a module function so the per-tick gate and the route sweep can be
    diffed against each other by eye; if preflight's rule ever changes, this is
    the one place that must follow it.
    """
    def subtree(root: int) -> set[int]:
        out, stack = set(), [root]
        while stack:
            b = stack.pop()
            out.add(b)
            stack += [c for c in range(model.nbody)
                      if model.body_parentid[c] == b and c != b]
        return out

    def geoms_of(bodies: set[int]) -> list[int]:
        return [g for g in range(model.ngeom)
                if model.geom_bodyid[g] in bodies
                and "collision" in (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]

    def chain(b: int) -> list[int]:
        out = [b]
        while b:
            b = model.body_parentid[b]
            out.append(b)
        return out

    def hops(b1: int, b2: int) -> int:
        c1, c2 = chain(b1), chain(b2)
        return min(i + c2.index(x) for i, x in enumerate(c1) if x in c2)

    arm_l = subtree(model.jnt_bodyid[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS_L[0])])
    arm_r = subtree(model.jnt_bodyid[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS_R[0])])
    other = set(range(1, model.nbody)) - arm_l - arm_r
    g_l, g_r, g_o = geoms_of(arm_l), geoms_of(arm_r), geoms_of(other)

    name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
    pairs = []
    for left, right in ((g_l, g_o), (g_r, g_o), (g_l, g_r)):
        for g1 in left:
            for g2 in right:
                if hops(model.geom_bodyid[g1], model.geom_bodyid[g2]) > 2:
                    pairs.append((g1, g2, name(g1), name(g2)))
    return pairs


def verify_route(bundle: RouteBundle, measured_enc: dict[str, float] | None, *,
                 model=None, clearance_floor_m: float = GATE_C_FLOOR_M,
                 home_enc: dict[str, float] | None = None,
                 windup_guard: bool = False) -> dict:
    """Run gates A-D (E for home routes when `home_enc` is given; F for reach
    routes when `windup_guard`). Returns a report dict; report["ok"] gates
    ALL execution.

    Never raises on a gate failure — the caller prints the report and refuses
    to move, which is the whole point: a refused route must leave a readable
    trace, not a stack trace."""
    model = model if model is not None else mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    data = mujoco.MjData(model)
    report = {"ok": True, "gates": {}}

    def fail(gate: str, msg: str) -> None:
        report["ok"] = False
        report["gates"][gate] = {"ok": False, "msg": msg}

    col = {n: i for i, n in enumerate(bundle.joint_names)}

    # --- Gate B: start == measured (catches seed scrambling server-side) ----
    if measured_enc is not None:
        worst, worst_j = 0.0, ""
        for j in bundle.controlled_joints:
            err = abs(float(bundle.q_enc[0, col[j]]) - float(measured_enc[j]))
            if err > worst:
                worst, worst_j = err, j
        if worst > GATE_B_START_TOL_RAD:
            fail("B_start", f"route[0] vs measured: {worst:.3f} rad at {worst_j} "
                            f"(tol {GATE_B_START_TOL_RAD}) — units-seam or seed mismatch; "
                            f"DO NOT EXECUTE, this is the silent-90-deg class")
        else:
            report["gates"]["B_start"] = {"ok": True, "worst_rad": worst, "joint": worst_j}

    # --- Gate F: reject the wind-up solution family (reach routes) ----------
    if windup_guard:
        away = {}
        ls = bundle.q_enc[:, col["left_shoulder_1_joint"]]
        rs = bundle.q_enc[:, col["right_shoulder_1_joint"]]
        away["left_shoulder_1_joint"] = float(ls[0] - ls.min())
        away["right_shoulder_1_joint"] = float(rs.max() - rs[0])
        worst_j = max(away, key=away.get)
        if away[worst_j] > GATE_F_WINDUP_RAD:
            fail("F_windup",
                 f"{worst_j} winds {away[worst_j]:.2f} rad AWAY from the "
                 f"cubes (tol {GATE_F_WINDUP_RAD}) — the wind-up solution "
                 f"family, which the field logs tie to failed grasps; "
                 f"restart for a direct-reach solution")
        else:
            report["gates"]["F_windup"] = {"ok": True,
                                           "worst_rad": away[worst_j],
                                           "joint": worst_j}

    # --- Gate E: a home route ends at the home POSTURE, not just pose -------
    # (see GATE_E_HOME_BRANCH_TOL_RAD: the server's goal="home" is an EE-pose
    # goal, so without this the solver may return the hands home in a
    # different IK branch — hands right, shoulder visibly wrong.)
    if home_enc is not None:
        worst, worst_j = 0.0, ""
        for j in bundle.controlled_joints:
            err = abs(float(bundle.q_enc[-1, col[j]]) - float(home_enc[j]))
            if err > worst:
                worst, worst_j = err, j
        if worst > GATE_E_HOME_BRANCH_TOL_RAD:
            fail("E_home_branch",
                 f"route end vs home posture: {worst:.3f} rad at {worst_j} "
                 f"(tol {GATE_E_HOME_BRANCH_TOL_RAD}) — the solver picked a "
                 f"different IK branch; hands home, posture not")
        else:
            report["gates"]["E_home_branch"] = {"ok": True, "worst_rad": worst,
                                                "joint": worst_j}

    # --- Gate D: joint limits ----------------------------------------------
    worst_margin, worst_j = np.inf, ""
    for j in bundle.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[jid]
        qs = bundle.q_enc[:, col[j]]
        margin = float(min(qs.min() - lo, hi - qs.max()))
        if margin < worst_margin:
            worst_margin, worst_j = margin, j
    if worst_margin < GATE_D_LIMIT_MARGIN_RAD:
        fail("D_limits", f"{worst_j} within {worst_margin:.3f} rad of a limit "
                         f"(margin {GATE_D_LIMIT_MARGIN_RAD})")
    else:
        report["gates"]["D_limits"] = {"ok": True, "worst_margin_rad": worst_margin,
                                       "joint": worst_j}

    # --- Gate A: FK of the final waypoint on OUR model ----------------------
    data.qpos[:] = model.qpos0
    for j in bundle.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        data.qpos[model.jnt_qposadr[jid]] = bundle.q_enc[-1, col[j]]
    mujoco.mj_kinematics(model, data)
    worst_fk, per_frame = 0.0, {}
    for frame, r in bundle.reaches.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame)
        if sid < 0:
            fail("A_fk", f"model has no site {frame}")
            break
        site = data.site_xpos[sid]
        cands = r["cands"].reshape(-1, 3) if r["cands"].size else r["target"].reshape(1, 3)
        d = float(np.min(np.linalg.norm(cands - site[None, :], axis=1)))
        per_frame[frame] = d
        worst_fk = max(worst_fk, d)
    else:
        if worst_fk > GATE_A_FK_TOL_M:
            fail("A_fk", f"final-waypoint FK misses the goal by {worst_fk*1000:.1f} mm "
                         f"(tol {GATE_A_FK_TOL_M*1000:.0f}) per-frame {per_frame} — "
                         f"units-seam broken or models diverged")
        else:
            report["gates"]["A_fk"] = {"ok": True, "worst_mm": worst_fk * 1000,
                                       "per_frame_mm": {k: v * 1000 for k, v in per_frame.items()}}

    # --- Gate C: clearance sweep on OUR model (joint_monkey's auditor) ------
    # SKIPPED when a cheap gate already failed (2026-08-03): this sweep is
    # ~1 s per route (161-181 frames x 1640 pairs) and the retry loop burns
    # one per attempt — a run with 9 wind-up rejections paid ~9 s of sweeps
    # for routes that were already dead. The verdict cannot change (ok is
    # already False) and the cheap gates' messages already name the reason;
    # a rejected route's clearance is a fact about a route nobody will run.
    if not report["ok"]:
        report["gates"]["C_clearance"] = {
            "ok": True, "skipped": "an earlier gate already rejected the "
                                   "route; the 1 s sweep proves nothing"}
        return report
    # preflight expects 14-joint frames; build them in ARM_JOINTS_ALL order,
    # filling any joint absent from the route (none today — routes carry all
    # 14) with its route/home column.
    qadr = []
    for j in ARM_JOINTS_ALL:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qadr.append(model.jnt_qposadr[jid])
    frames = [np.array([bundle.q_enc[h, col[j]] for j in ARM_JOINTS_ALL])
              for h in range(bundle.horizon)]
    worst_clear, where = preflight(model, data, _FrameShim(frames), qadr)
    if worst_clear < clearance_floor_m:
        fail("C_clearance", f"min clearance {worst_clear*1000:.1f} mm < floor "
                            f"{clearance_floor_m*1000:.0f} mm at {where}")
    else:
        report["gates"]["C_clearance"] = {"ok": True, "worst_mm": worst_clear * 1000,
                                          "where": where}
    return report
