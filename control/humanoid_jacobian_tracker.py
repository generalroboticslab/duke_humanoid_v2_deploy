"""Damped-least-squares Jacobian drift-servo: follow a plan-0 route, correct the drift.

PORTED FROM the upstream `mj_envs/tasks/visual_manipulation/jacobian_reach_tracker.py`
(lev2_master), which is the DEFAULT per-tick tracker in the upstream sim
(`DYNAMIC_REACH_TRACKER` defaults to "jacobian") and which upstream wrote to REPLACE
the cuRobo MPC reactive tracker. The algorithm, the constants and the reasoning
below are upstream's; what changed is the two seams to this repo, listed at the bottom.

WHAT IT IS FOR, in upstream's words: "plan-0 already solves a collision-free arm
trajectory (home -> grasp); at runtime the ONLY thing that changes is the
floating base drifting a few cm under arm reaction. Correcting a known-good path
for small drift is one damped-least-squares Jacobian step -- not a full
trajectory re-optimization."

That drift is exactly the phenomenon this robot shows: the torso leans a little
while the arm extends, and because every detection reaches the base frame through
gimbal FK on that same leaning torso, a world-fixed cube MOVES in the base frame.

HOW IT WORKS (the one idea worth understanding):

  rebind()  FK the whole planned route ONCE and record each tool pose RELATIVE TO
            ITS OBJECT. That object-relative path is constant and base-independent
            — it is the shape of the reach, stripped of where the cube happened
            to be at plan time.
  step()    each tick, compose that constant offset with the LIVE object pose to
            get where the tool should be NOW, then take a few damped-LS Newton
            steps from the planned configuration to get there.

So the command is `planned waypoint + a small correction`, never a fresh IK
solve. Two consequences follow directly:

  * a moving cube and a leaning torso are the SAME correction — both just move
    the object pose the path is anchored to;
  * the collision-free SHAPE of the cuRobo route is preserved. A from-scratch
    per-tick IK (mink) would re-derive a configuration with no memory of the
    route and can land in a different branch.

WHERE THE SAFETY COMES FROM. Upstream is explicit that this servo has NO per-tick
collision term: "collision safety comes from the nominal path already being
collision-free plus small, bounded drift". We keep that argument AND add what upstream
does not have — every emitted command still goes through StepGate
(humanoid_curobo_client), a real ~1.5 ms clearance/limit/rate check on this
model. The correction caps below are the "bounded" half of that argument and are
therefore load-bearing, not tuning.

WHY NOT cuRobo MPC (which this repo also implements, and measured). Upstream's three
reasons, each confirmed here on 2026-07-30:
  * SPEED — the upstream MPC was a fixed ~16 ms/tick; measured here 12 ms of solve plus
    WiFi gives wall p90 24 ms and one 232 ms outlier. This servo is pure MuJoCo
    FK plus a small linear solve, sub-millisecond, and runs ON THE ROBOT — so
    the per-tick loop has no network in it at all, and the "link loss strands the
    arm mid-reach" degradation simply does not exist.
  * WEDGE — MPC was asked to reach a target ON the cube while treating the cube
    as a hard obstacle to the same arm, and resolved the contradiction by parking
    short. This servo has no per-tick collision term, so contact with the target
    cube is the intended grasp.
  * The MPC session stays committed as a measured baseline, not as the path.

THE TWO SEAMS THAT DIFFER FROM THE UPSTREAM VERSION:
  1. UNITS. The upstream route is cuRobo cspace and upstream folds `+qpos0` inside the tracker.
     Our RouteBundle is ALREADY in encoder units (humanoid_curobo_client folds it,
     with the left_wrist_2 sign the upstream seam does not need). So this tracker is
     encoder-units throughout and performs no fold — one less place for the
     silent-90-degree class to hide.
  2. NO GRAVITY-COMP FEEDFORWARD. Upstream pre-bends the reference by g(q)/kp to beat
     a PD position servo's droop. real_env already runs --grav_comp, so adding it
     here would compensate twice.
  Also not ported: the outer Cartesian integral (it needs a measured tool pose,
  which belongs with the visual-servo work) and the viewer overlay.
"""

from __future__ import annotations

import numpy as np
import mujoco

from humanoid_curobo_client import ARM_JOINTS_ALL  # noqa: F401  (re-exported for callers)
from humanoid_model import MJCF_MODEL_PATH

# --- upstream's constants, carried over verbatim with its rationale ------------
DLS_LAMBDA = 0.05          # damped-LS regularisation: trades tracking tightness
                           # against conditioning near arm singularities. Raise
                           # if the servo oscillates.
DQ_CLAMP = 0.05            # per-INNER-ITER joint step (rad). Bounds each Newton
                           # step, not the tick total; measured normal-tick net
                           # correction stays <= 0.12 rad, so this costs normal
                           # ticks nothing while cutting the singular worst case.
IK_ITERS = 12              # inner damped-LS iterations per control tick, each a
                           # full FK + one Newton step. The seed is the planned
                           # configuration, so the goal is already close.
IK_TOL_M = 0.002           # early-out only when every active tool is this close
IK_TOL_RAD = 0.03          # ...AND within 1.7 deg. Position converges before
                           # wrist orientation; terminating on position alone
                           # leaves an orientation error plan-0 never has.
NULLSPACE_POSTURE_GAIN = 0.10
# ^ the redundant arm DOFs are pulled toward the collision-checked cuRobo
#   waypoint. Without it the null space wandered into a joint-limit branch under
#   base motion even while the tool target stayed reachable. Acts only through
#   (I - J^+J), so Cartesian tracking still comes first.

# The "bounded drift" half of the safety argument. Two caps, against two
# different failure modes: the first rejects IK branch jumps away from the arm's
# real position, the second stops accumulated base-drift correction from
# spending the route's joint-limit margin.
MAX_CORRECTION_FROM_MEASURED_RAD = 0.35
MAX_CORRECTION_FROM_PLANNED_RAD = 0.35

# OUTPUT SLEW LIMIT — ours, not upstream's, and required by this stack rather than
# optional. Upstream's caps bound the SIZE of a correction; none of them bound its RATE,
# because the upstream sim has no per-tick rate gate. Ours does (StepGate, 0.7 rad/s), and
# a correction that appears in a single tick trips it: measured here, a 2 cm
# object step-change produced 1.58 rad/s on wrist_1 on the very first tick, which
# the gate refused — correctly. A refused tick is not a small problem either: the
# caller republishes the last gated command, so a servo that lunges is a servo
# that stalls. Ramping the correction in over a few ticks makes the emitted
# command satisfy the gate BY CONSTRUCTION.
#
# Below the gate's cap on purpose: the gate is the independent check, not the
# thing being tuned against.
MAX_OUTPUT_RATE_RAD_S = 0.5


def pose_relative(frame, pose):
    """`pose` expressed in `frame`'s coordinates — i.e. frame^-1 . pose."""
    fp, fq = np.asarray(frame[0], float), np.asarray(frame[1], float)
    pp, pq = np.asarray(pose[0], float), np.asarray(pose[1], float)
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, fq)
    d = pp - fp
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, d, inv)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, inv, pq)
    return out, q


def pose_compose(frame, rel):
    """The inverse of `pose_relative`: place `rel` back into the world."""
    fp, fq = np.asarray(frame[0], float), np.asarray(frame[1], float)
    rp, rq = np.asarray(rel[0], float), np.asarray(rel[1], float)
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, rp, fq)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, fq, rq)
    return fp + out, q


def _lerp_pose(a, b, alpha: float):
    """Interpolate base-frame (position, quat_wxyz) without a frame discontinuity."""
    pa, qa = np.asarray(a[0], float), np.asarray(a[1], float)
    pb, qb = np.asarray(b[0], float), np.asarray(b[1], float)
    if float(np.dot(qa, qb)) < 0.0:            # q and -q are the same rotation
        qb = -qb
    dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if dot > 0.9995:
        q = qa + alpha * (qb - qa)
        q /= np.linalg.norm(q)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1.0 - alpha) * angle) * qa + np.sin(alpha * angle) * qb) \
            / np.sin(angle)
    return pa + alpha * (pb - pa), q


class JacobianRouteTracker:
    """Built once; `rebind` re-points it at a new route, `step` emits one tick."""

    def __init__(self, model=None, *, control_dt: float,
                 dls_lambda: float = DLS_LAMBDA,
                 ik_iters: int = IK_ITERS,
                 nullspace_gain: float = NULLSPACE_POSTURE_GAIN,
                 max_from_measured_rad: float = MAX_CORRECTION_FROM_MEASURED_RAD,
                 max_from_planned_rad: float = MAX_CORRECTION_FROM_PLANNED_RAD,
                 max_output_rate_rad_s: float = MAX_OUTPUT_RATE_RAD_S):
        self.model = model if model is not None else \
            mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.dls_lambda = float(dls_lambda)
        self.ik_iters = int(ik_iters)
        self.nullspace_gain = float(nullspace_gain)
        self.max_from_measured = float(max_from_measured_rad)
        self.max_from_planned = float(max_from_planned_rad)
        self.max_output_step = float(max_output_rate_rad_s) * self.control_dt
        # The floating-base free joint is pinned to identity on every FK, so the
        # scratch world IS the base frame and no base world pose is ever read —
        # the property that makes this deploy-faithful.
        self._root_qadr = next(
            int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self._qadr_by_joint = {
            self.model.joint(j).name: int(self.model.jnt_qposadr[j])
            for j in range(self.model.njnt) if self.model.joint(j).name}
        self.bound = False

    # -- binding ---------------------------------------------------------------
    def rebind(self, bundle, objects_plan: dict) -> None:
        """Point at a NEW route.

        `objects_plan` = {object_key: (pos, quat_wxyz)} — each object's BASE-FRAME
        pose AT PLAN TIME, i.e. the detection the route was solved against. The
        route's shape is recorded relative to these; passing the live pose here
        instead would silently zero the very correction this class exists for.
        """
        self.frames = list(bundle.reaches)                 # active tool sites
        self.joint_names = list(bundle.joint_names)
        self.controlled = list(bundle.controlled_joints)
        self.q_route = np.asarray(bundle.q_enc, float)     # (H, n) ENCODER units
        self.horizon = int(self.q_route.shape[0])
        self.dt = float(bundle.dt)
        # A route faster than the servo can emit would not fail — it would LAG,
        # silently and increasingly, with the correction caps then dragging
        # against a waypoint the arm never reached. Refuse at bind time instead;
        # missions already stretch before streaming, so this only catches a
        # caller that forgot.
        route_rate = bundle.max_joint_rate()
        assert route_rate <= self.max_output_step / self.control_dt + 1e-9, (
            f"route moves at {route_rate:.2f} rad/s but the tracker emits at most "
            f"{self.max_output_step / self.control_dt:.2f} — call "
            f"bundle.stretch(max_rate) before rebind(), or the servo will lag the "
            f"route instead of tracking it")
        self._site_id = {f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f)
                         for f in self.frames}
        missing = [f for f, i in self._site_id.items() if i < 0]
        assert not missing, f"tool site(s) {missing} not in the deploy model"
        self._qadr = np.array([self._qadr_by_joint[n] for n in self.joint_names])
        self._dofadr = np.array([
            int(self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in self.joint_names])

        # Which object anchors which tool frame. assignment is {side: object}.
        self.frame_to_object = {}
        for side, obj in bundle.assignment.items():
            frame = f"end_effector_{side.upper()}_site"
            if frame in self._site_id:
                self.frame_to_object[frame] = obj
        missing = [f for f in self.frames if f not in self.frame_to_object]
        assert not missing, (f"no object assigned to active frame(s) {missing}; "
                             f"assignment={bundle.assignment}")
        unseen = [o for o in self.frame_to_object.values() if o not in objects_plan]
        assert not unseen, f"plan-time pose missing for object(s) {unseen}"

        # ONE-TIME FK of the whole path -> a constant, base-independent,
        # object-relative waypoint path per active frame.
        self.path_in_object = {f: [] for f in self.frames}
        for k in range(self.horizon):
            self._fk(self.q_route[k])
            for f in self.frames:
                site = (self.data.site_xpos[self._site_id[f]].copy(), self._site_quat(f))
                self.path_in_object[f].append(
                    pose_relative(objects_plan[self.frame_to_object[f]], site))
        self._elapsed = 0.0
        self._hold_pose = None
        self._last_desired = None
        self._last_q = None                # output slew state; re-seeds per route
        self.bound = True

    # -- forward kinematics ----------------------------------------------------
    def _fk(self, q_arm, extra_qpos: dict | None = None) -> None:
        """Scratch FK with the root pinned to identity (scratch world == base).

        `extra_qpos` should carry the MEASURED leg/waist posture: live articulation
        below the shoulder moves the arm mount, so a base-frame tool orientation
        computed with those joints at qpos0 is wrong even when every arm joint is
        exact.
        """
        self.data.qpos[:] = self.model.qpos0
        if extra_qpos:
            for name, value in extra_qpos.items():
                adr = self._qadr_by_joint.get(name)
                if adr is not None:
                    self.data.qpos[adr] = float(value)
        self.data.qpos[self._root_qadr:self._root_qadr + 3] = 0.0
        self.data.qpos[self._root_qadr + 3:self._root_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self._qadr] = np.asarray(q_arm, float)
        mujoco.mj_forward(self.model, self.data)

    def _site_quat(self, frame: str) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self._site_id[frame]])
        return q

    # -- the hot path ----------------------------------------------------------
    def _track(self, desired: dict, q_nom: np.ndarray,
               measured_enc: dict | None, extra_qpos: dict | None) -> dict:
        """One bounded Cartesian tracking tick. Returns {joint: encoder rad}."""
        if measured_enc is None:
            q = q_nom.copy()
        else:
            q = np.asarray([measured_enc[j] for j in self.joint_names], float)
        q_measured = q.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        pos_err = rot_err = 0.0
        for _ in range(self.ik_iters):
            self._fk(q, extra_qpos)
            rows, errs = [], []
            for f in self.frames:
                e_pos = np.asarray(desired[f][0], float) \
                    - self.data.site_xpos[self._site_id[f]]
                neg, dq_quat, e_rot = np.zeros(4), np.zeros(4), np.zeros(3)
                mujoco.mju_negQuat(neg, self._site_quat(f))
                mujoco.mju_mulQuat(dq_quat, np.asarray(desired[f][1], float), neg)
                mujoco.mju_quat2Vel(e_rot, dq_quat, 1.0)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id[f])
                rows.append(np.vstack([jacp[:, self._dofadr], jacr[:, self._dofadr]]))
                errs.append(np.concatenate([e_pos, e_rot]))
            J = np.vstack(rows)                      # (6 * n_active, n_joints)
            e = np.concatenate(errs)
            pos_err = max(float(np.linalg.norm(x[:3])) for x in errs)
            rot_err = max(float(np.linalg.norm(x[3:])) for x in errs)
            if pos_err < IK_TOL_M and rot_err < IK_TOL_RAD:
                break
            n = J.shape[0]
            J_pinv = J.T @ np.linalg.solve(J @ J.T + (self.dls_lambda ** 2) * np.eye(n),
                                           np.eye(n))
            dq = J_pinv @ e
            dq += self.nullspace_gain * (np.eye(J.shape[1]) - J_pinv @ J) @ (q_nom - q)
            q = q + np.clip(dq, -DQ_CLAMP, DQ_CLAMP)

        # Bounded drift — the load-bearing half of the "no per-tick collision
        # term" safety argument. Against the MEASURED arm first (rejects branch
        # jumps), then against the collision-checked waypoint (stops accumulated
        # correction from spending the route's limit margin).
        if measured_enc is not None:
            np.clip(q, q_measured - self.max_from_measured,
                    q_measured + self.max_from_measured, out=q)
        np.clip(q, q_nom - self.max_from_planned, q_nom + self.max_from_planned, out=q)

        # Slew-limit the OUTPUT so the emitted command satisfies the per-tick rate
        # gate by construction (see MAX_OUTPUT_RATE_RAD_S). Seeded from the
        # MEASURED arm, not from the first solved command: tick 0 is precisely
        # the tick where an object that moved since planning shows up as a jump,
        # so exempting it would exempt the case this exists for.
        ref = self._last_q if self._last_q is not None else q_measured
        np.clip(q, ref - self.max_output_step, ref + self.max_output_step, out=q)
        self._last_q = q.copy()

        self.last_pos_err_m = pos_err
        self.last_rot_err_rad = rot_err
        self.last_correction_rad = float(np.abs(q - q_nom).max())
        col = {n: i for i, n in enumerate(self.joint_names)}
        return {j: float(q[col[j]]) for j in self.controlled}

    def step(self, objects_in_base: dict, measured_enc: dict | None = None,
             extra_qpos: dict | None = None, advance: bool = True) -> dict:
        """Advance one control tick along the route, corrected to the LIVE objects.

        `objects_in_base` = {object_key: (pos, quat_wxyz)} as measured NOW.
        Returns {joint: encoder rad} over the route's controlled joints — feed it
        through StepGate before publishing.
        """
        assert self.bound, "rebind() the tracker on a route first"
        progress = self._elapsed / self.dt
        lo = min(int(progress), self.horizon - 1)
        hi = min(lo + 1, self.horizon - 1)
        alpha = progress - lo
        q_nom = (1.0 - alpha) * self.q_route[lo] + alpha * self.q_route[hi]
        desired = {}
        for f in self.frames:
            rel = _lerp_pose(self.path_in_object[f][lo],
                             self.path_in_object[f][hi], alpha)
            desired[f] = pose_compose(objects_in_base[self.frame_to_object[f]], rel)
        self._last_desired = desired
        command = self._track(desired, q_nom, measured_enc, extra_qpos)
        if advance:
            self._elapsed += self.control_dt
        return command

    @property
    def done(self) -> bool:
        return self._elapsed >= (self.horizon - 1) * self.dt

    def final_step(self, objects_in_base: dict, measured_enc: dict | None = None,
                   extra_qpos: dict | None = None) -> dict:
        """Hold the LAST waypoint while the jaws close, still anchored to the live
        object. The upstream note applies unchanged: freezing this target in the base
        frame at reach completion made a base shift during closure look like a
        grasp miss."""
        assert self.bound, "rebind() the tracker on a route first"
        desired = {f: pose_compose(objects_in_base[self.frame_to_object[f]],
                                   self.path_in_object[f][-1])
                   for f in self.frames}
        self._last_desired = desired
        return self._track(desired, self.q_route[-1], measured_enc, extra_qpos)

    def freeze_hold(self) -> None:
        """Freeze the last tracked target in the BASE frame — used once the jaws
        have closed, so the held pose no longer follows an object the hand is now
        carrying (its detection is occluded and, once lifted, meaningless)."""
        assert self._last_desired is not None, "nothing tracked yet to freeze"
        self._hold_pose = {f: (p[0].copy(), p[1].copy())
                           for f, p in self._last_desired.items()}

    def hold_step(self, measured_enc: dict | None = None,
                  extra_qpos: dict | None = None) -> dict:
        assert self._hold_pose is not None, "freeze_hold() before holding"
        return self._track(self._hold_pose, self.q_route[-1], measured_enc, extra_qpos)
