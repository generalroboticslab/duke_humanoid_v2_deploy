# python humanoid_joint_monkey_hw.py
"""Hardware joint monkey: replay the sim joint-monkey ARM choreography on the
real robot through the 9874 rate-limited joint stream (photo/video sessions).

HANGING ONLY — never run this standing. The legs stay policy-held; this
script drives the 14 arm joints only.

Terminal flow:
    T2  python humanoid_real_env.py --task <task> --use-ik --grav_comp \
            --torque_limit 0.5 --enable-motor arm \
            --high-level-controller-ip 127.0.0.1 --arm-sender-ip 127.0.0.1
    T4  python humanoid_joint_monkey_hw.py             # dry-run: preflight + viser
    T4  python humanoid_joint_monkey_hw.py --live      # stream to the robot

Design decisions:
- The whole trajectory is PRECOMPUTED in model qpos space (== telemetry /
  stream space: hand-eye calibration mm residuals and the 44 cm power-on ->
  side-home glide, measured 43 cm on hardware 07-20, both hold under
  qpos==jpos) and PREFLIGHTED before anything may stream: every frame is
  FK-checked with mj_geomDistance for arm-to-body and arm-to-arm clearance.
  --live refuses to start if the minimum clearance over the entire
  trajectory is below --min-clearance-mm. Streaming exactly what was checked
  is the point: the ramp from the power-on pose into the sweep is part of
  the precomputed trajectory, never left to the receiver's own crawl.
- Sweeps reuse the sim monkey's validated staging choreography (15 deg
  outward shoulder tilt, forward/backward sagittal split, mirrored pairs)
  but hard-cap margin <= 0.7 and speed <= 0.3 rad/s: hard limits risk cable
  wrap and the wrist encoders' known +-pi wrap; real_env's per-packet rate
  cap (0.5) is the second layer below that.
- Arms only. Legs are policy-owned (no direct joint path exists).
- CAMERA FOLLOW (2026-08-10, user): both gimbals track their own arm's
  end-effector for the whole sweep, aimed from the COMMANDED frame (the eye
  leads the motion instead of chasing the arm's following lag) through the
  MEASURED gimbal angles. The aim bodies are the operator's, run verbatim via
  a duck-typed shim; the import is guarded so a broken operator module costs
  the gaze, never the sweep. --no-gaze restores the old arms-only stream.
- Startup gate mirrors the operator's: refuse to stream unless BOTH arms are
  measured within POSTURE_TOL of the power-on pose. Streaming from an
  unknown pose would make the receiver crawl an unchecked path.
- Tracking watchdog (07-20 carry-jam lesson): if measured arm joints diverge
  from the command by more than WATCH_ERR_RAD for WATCH_ABORT_S, the stream
  aborts in place — a jam presses harder the longer it is fed.
- On exit (Ctrl+C, watchdog, or end of sweep) the script holds the current
  frame briefly and goes silent; real_env's 0.5 s arm-silence failsafe then
  crawls the arms to the default pose at 0.025 rad/s — the gentlest
  recovery there is. The final choreography segment already returns to the
  power-on pose, so a clean run ends where the failsafe would.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import tyro

from humanoid_model import MJCF_MODEL_PATH
from humanoid_utils import HumanoidViserViz

from ipc.publisher import NNGPublisher, NNGSubscriber

STREAM_HZ = 20                      # operator's own tick rate; real_env re-limits per 50 Hz step
DT = 1.0 / STREAM_HZ                # THIS loop's tick. Read by auto_operator.gaze
                                    # through its _K(op) contract (see GazeAim):
                                    # slew_gaze's step is GAZE_SLEW_RATE * DT, so
                                    # it must be OUR period, not the operator's.

# --- camera follow (2026-08-10, user: the eyes should stay on the hands) -------
# LAZY AND GUARDED, for two independent reasons.
#
# LAZY: humanoid_curobo_client imports preflight/ARM_JOINTS_* from THIS module
# for its Gate C, so a top-level gaze import would put the operator + monitor
# chain into every cuRobo plan — and humanoid_monitor's import does
# `sys.path.insert(0, "<visual_servoing>")`, whose `common` package
# then shadows control's for every later first-import (the trap
# tests/test_ee_zeroing.py has to purge by hand). Camera follow is this tool's
# feature; nobody else may pay for it. Loaded from main(), only with --gaze.
#
# GUARDED: this file is the recovery/audit path — POWERON_JOINTS is duplicated
# here rather than imported precisely so the sweep still runs when
# humanoid_auto_operator cannot be loaded. A broken operator module must cost
# the GAZE, never the sweep.
#
# The aim math is the operator's, run VERBATIM (same rule humanoid_curobo_reach
# followed for its GazeRig): probed camera frame with no hardcoded signs, the
# closed-form parallax aim refined twice against the full FK, the nearest-branch
# +-pi unwrap that keeps the rear-mounted right camera off the wrap cut, the
# near-lock integral trim, and the sender-side slew. Those bodies resolve their
# constants LIVE via sys.modules[type(op).__module__] — which for GazeAim is
# THIS module — which is why _load_gaze publishes them into these globals.
ao_gaze = None
GimbalCameraFK = None
_GAZE_ERR = "not loaded"
_GAZE_CONSTANTS = ("CAM_PORTS", "CAM_SITES", "GAZE_ORDER", "GAZE_SLEW_RATE",
                   "TRIM_GAIN", "TRIM_MAX")


def _load_gaze():
    """Import the operator's gaze stack on demand. Returns an error string, or
    None once ao_gaze/GimbalCameraFK and the _K constants are live here."""
    global ao_gaze, GimbalCameraFK, _GAZE_ERR
    if ao_gaze is not None:
        return None
    try:
        import humanoid_auto_operator as _OP
        from auto_operator import gaze as _gz
        from humanoid_monitor import GimbalCameraFK as _FK
        globals().update({k: getattr(_OP, k) for k in _GAZE_CONSTANTS})
        ao_gaze, GimbalCameraFK, _GAZE_ERR = _gz, _FK, None
        return None
    except Exception as e:                   # noqa: BLE001 — any import failure
        _GAZE_ERR = f"{type(e).__name__}: {e}"
        return _GAZE_ERR

# Each eye watches its OWN side's hand. The choreography splits the arms (left
# forward, right back) and the mounts already match that split: 5555 is the
# front-datum camera, 5556 the rear-mounted one, so per-side assignment is also
# the shortest bearing for both. Cross-assign here if a shot needs both eyes on
# one hand.
GAZE_EE_SITE = {5555: "end_effector_L_site", 5556: "end_effector_R_site"}
# Gimbal joints in gaze_targets slot order (yaw_l, pitch_l, yaw_r, pitch_r) —
# viser only, so the dry-run SHOWS the follow instead of only publishing it.
CAM_JOINT_NAMES = ("cam_yaw_left", "cam_pitch_left",
                   "cam_yaw_right", "cam_pitch_right")
SPEED_CAP = 0.4                     # rad/s, absolute — flag values above this are
                                    # clipped. 0.3 -> 0.4 (2026-08-10, user: the
                                    # sweep is too slow). 0.4 is not arbitrary:
                                    # the packet rate below is min(speed*1.2,
                                    # 0.5), so 0.4 is the FASTEST sweep that
                                    # still keeps the intended 20% catch-up
                                    # headroom under that 0.5 ceiling (0.48).
                                    # Past it the rate saturates and the arm
                                    # would be asked to track its own commanded
                                    # speed with no margin. Raising it further
                                    # means raising the 0.5 in the publish call
                                    # too — the receiver's own cap is 1.05.
MARGIN_CAP = 0.7                    # fraction of each joint range, absolute cap
POSTURE_TOL = 0.35                  # rad, startup gate tolerance (== operator's)
WATCH_ERR_RAD = 0.50                # tracking watchdog: divergence threshold.
                                    # 0.35 -> 0.50 (2026-08-10): the can21
                                    # right-arm harness drags the arm 0.31-0.39
                                    # rad behind its command under load
                                    # (measured repeatedly in the MPC tracker,
                                    # clean walk-backs every time — drag, not a
                                    # wedge), so a full-ROM sweep aborted on the
                                    # right arm and logged it as a jam. Same
                                    # trade the MPC's own STREAM_FOLLOW_LAG_RAD
                                    # took for the same harness: a TRUE jam now
                                    # gets 0.50 rad of PD error before the
                                    # stream backs out, bounded by the drive
                                    # caps. Replacing that harness earns this
                                    # back down to 0.35.
WATCH_ABORT_S = 2.0                 # ... sustained this long -> abort
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
MIRROR_Y = np.array([1.0, -1.0, 1.0])
# Right arm = left arm negated, per joint. 07-28: briefly (+1) on wrist_2
# while upstream's new wrist had left_wrist_2's axis turned to match the
# right; the physical motor was never rewired, so the deploy model flips it
# back (rebuild_deploy_model.py) and the plain negation holds again. The
# `mirror_sign` probe below DERIVES this from the loaded model and the
# assert loud-fails on disagreement -- that is the guard, not this literal.
ARM_MIRROR_SIGN = np.array([-1., -1., -1., -1., -1., -1., -1.])

# Power-on arm pose, LEFT-arm convention (right = negation) — MUST equal the
# operator's POWERON_JOINTS / the deployed env_config default_pose.
# 07-31: the SIM HOME (cuRobo HUMANOID_ARM_JOINT_HOME through the units seam).
# The previous literal here was the pre-07-30 low tuck — TWO generations stale
# by the time it was caught (the 07-31 consumer survey), during which this
# tool's startup gate refused every legitimately-homed arm. Duplicated rather
# than imported on purpose (this is the recovery/audit path and must run when
# the operator module cannot import), so when the home moves, THIS moves —
# tests/test_model_derived_constants pins the operator copy to the sim source,
# and test_joint_monkey_poweron_matches_operator pins this copy to the
# operator's.
POWERON_JOINTS = np.array([-1.3740, -0.1870, +0.1310, +1.9290, -1.2403,
                           -1.4546, -0.0090])

ARM_JOINTS_L = ["left_shoulder_1_joint", "left_shoulder_2_joint",
                "left_shoulder_3_joint", "left_elbow_joint",
                "left_wrist_1_joint", "left_wrist_2_joint", "left_wrist_3_joint"]
ARM_JOINTS_R = [n.replace("left", "right") for n in ARM_JOINTS_L]
TEL_ARM = slice(13, 27)             # telemetry joint_pos: [13:20] left, [20:27] right

# Choreography constants (== sim monkey)
STAGED_ARM_KEYS = ("shoulder_2", "shoulder_3", "elbow", "wrist")
ARM_FORWARD_Q = np.pi / 2
ARM_OUT_TILT_L = -np.deg2rad(25)  # 15 -> 25 deg (2026-08-10, NEW-WRIST MODEL).
                                  # The tilt exists because raw qpos0 rests the
                                  # grippers INSIDE the hips; 15 deg was sized
                                  # for the OLD gripper and the 07-28 wrist
                                  # redesign ate it. Measured on the deploy
                                  # model, margin 0.6, min clearance over the
                                  # whole trajectory:
                                  #   15 deg -> 15.5 mm  (R_base ~ hip_2_R, in
                                  #                       the ramp at 6.2 s)
                                  #   20 deg -> 34.7 mm  (same pair, late)
                                  #   25 deg -> 39.5 mm  (R_base ~ base_link,
                                  #                       FRAME 0)
                                  #   30..50 -> 39.5 mm  (unchanged)
                                  # 25 is the smallest tilt that reaches the
                                  # plateau, so it is the least change to the
                                  # photographed pose that clears the hips.
                                  # The plateau is NOT this constant's to move:
                                  # at frame 0 the arm is at POWERON_JOINTS and
                                  # the binding pair is the right gripper base
                                  # against the torso — the 39.5 mm static
                                  # floor of the one home, pinned independently
                                  # by tests/test_home_pose_margins. No tilt
                                  # can beat it, and every mission already
                                  # starts and ends there.
                                  # Side effect, deliberate: more tilt moves
                                  # the shoulder_2 rest TOWARD its midpoint, so
                                  # it also relaxes the margin floor below
                                  # (25 deg needs margin >= 0.476, 15 needed
                                  # 0.571) — the two constraints do not fight.


@dataclass
class Args:
    speed: float = 0.4
    """Sweep angular speed (rad/s), hard-capped at SPEED_CAP (0.4). 0.2 -> 0.4
    (2026-08-10, user): the sweep reads too slow on video. Sweeps halve in
    duration; the per-limit `hold` pauses do not, so the run does not halve."""
    margin: float = 0.6
    """Fraction of each range to sweep about its midpoint, hard-capped at 0.7."""
    hold: float = 1.0
    """Hold time (s) at each limit (photo pause)."""
    start_delay: float = 3.0
    """Seconds between the go-confirmation and the first streamed motion."""
    live: bool = False
    """Actually stream to the robot (default: dry-run — preflight + viser only)."""
    robot_ip: str = "127.0.0.1"
    """Robot telemetry host (9870)."""
    min_clearance_mm: float = 25.0
    """Preflight floor: --live refuses if any frame's arm clearance is below this."""
    gaze: bool = True
    """Keep both gimbal cameras aimed at their own arm's end-effector for the
    whole sweep (2026-08-10, user). Rides the same 9874 packets as the arm
    targets — the receiver excludes gaze from the arm-silence timer, so this
    changes nothing about the failsafe. --no-gaze streams arms only."""
    viz: bool = True
    """Mirror the commanded pose to viser (dry-run always visualizes)."""
    port: int = 8090
    """Viser port."""


class Trajectory:
    """Precomputed 14-joint frame list at STREAM_HZ with stage labels."""

    def __init__(self, q0: np.ndarray, speed: float):
        self.frames = [q0.copy()]           # each: (14,) [left7, right7]
        self.labels: dict[int, str] = {}
        self.speed = speed

    @property
    def q(self) -> np.ndarray:
        return self.frames[-1]

    def label(self, text: str) -> None:
        self.labels[len(self.frames) - 1] = text

    def hold(self, sec: float) -> None:
        n = max(0, round(sec * STREAM_HZ))
        self.frames.extend([self.q.copy()] * n)

    def move(self, targets: dict[int, float]) -> None:
        """Simultaneous per-joint linear move; duration set by the slowest joint."""
        start = self.q.copy()
        deltas = {k: t - start[k] for k, t in targets.items()}
        steps = max(1, max(round(abs(d) * STREAM_HZ / self.speed)
                           for d in deltas.values()) if deltas else 1)
        for i in range(1, steps + 1):
            f = start.copy()
            for k, d in deltas.items():
                f[k] = start[k] + d * i / steps
            self.frames.append(f)

    def sweep_pair(self, kl: int, target_l: float, kr: int | None, sign: float,
                   rest: np.ndarray, lim: np.ndarray) -> None:
        """Leader kl to target_l; follower kr tracks rest_r + sign*(l - rest_l),
        clipped to the follower's margin-shrunk range."""
        start = self.q.copy()
        steps = max(1, round(abs(target_l - start[kl]) * STREAM_HZ / self.speed))
        for i in range(1, steps + 1):
            f = self.q.copy()
            f[kl] = start[kl] + (target_l - start[kl]) * i / steps
            if kr is not None:
                f[kr] = np.clip(rest[kr] + sign * (f[kl] - rest[kl]),
                                lim[kr][0], lim[kr][1])
            self.frames.append(f)


class GazeAim:
    """Aim each gimbal camera at its own arm's end-effector, every frame.

    Duck-types exactly the AutoOperator fields auto_operator.gaze touches (fk,
    _cam_frame, _trim, _wrap, gaze), so the operator's aim bodies run unmodified
    — see the import note above for why that matters and what they do.

    The TARGET is computed from the COMMANDED frame, not from telemetry: this
    tool knows exactly where it is sending the hand next, so the eye leads the
    motion instead of chasing it through the arm's following lag. The JPOS
    passed to the aim is the MEASURED one, because that is what the camera FK
    and the trim integrator need (where the gimbal actually is).

    Seeded from the measured gimbal encoders before the first publish — the
    operator's rule, and it is a hardware one: a zeros-init gaze against a
    parked gimbal commands a multi-radian slam on the first packet.
    """

    def __init__(self):
        self.fk = GimbalCameraFK({p: CAM_SITES[p] for p in CAM_PORTS})
        self.gaze = np.zeros(4)
        self._seeded = False
        self._trim = {p: np.zeros(2) for p in CAM_PORTS}
        self._cam_frame = {p: ao_gaze.probe_cam_frame(self, p) for p in CAM_PORTS}

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def seed(self, jpos) -> bool:
        if self._seeded:
            return True
        if jpos is None or len(jpos) < 31 or not np.all(np.isfinite(jpos[27:31])):
            return False
        self.gaze = np.asarray(jpos[27:31], dtype=float).copy()
        self._seeded = True
        return True

    def step(self, targets: dict, jpos) -> list:
        """One tick: aim at each target, slew, return the 4-vector to publish."""
        want = self.gaze.copy()
        for port in CAM_PORTS:
            yaw, pitch = ao_gaze.aim_angles(self, port, targets[port], jpos)
            ys, ps = GAZE_ORDER[port]
            want[ys], want[ps] = yaw, pitch
        self.gaze = ao_gaze.slew_gaze(self, want)
        return [float(v) for v in self.gaze]


def ee_targets(model, data, qadr, site_ids, frame) -> dict:
    """Base-frame EE site positions for a commanded 14-joint frame.

    Root stays at qpos0 (identity), so site_xpos IS the base-frame position the
    gaze aim expects — the same convention preflight uses.
    """
    data.qpos[:] = model.qpos0
    for k, a in enumerate(qadr):
        data.qpos[a] = frame[k]
    mujoco.mj_kinematics(model, data)
    return {p: data.site_xpos[sid].copy() for p, sid in site_ids.items()}


def build_trajectory(model, data, speed: float, margin: float, hold: float):
    """The sim monkey's arm choreography as a precomputed 14-joint frame list,
    prefixed with power-on -> model-home and suffixed with return-to-power-on."""
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ARM_JOINTS_L + ARM_JOINTS_R]
    assert all(j >= 0 for j in jids), "arm joint missing from model"
    qadr = [model.jnt_qposadr[j] for j in jids]
    ranges = model.jnt_range[jids].copy()
    home = np.array([model.qpos0[a] for a in qadr])
    mids = 0.5 * (ranges[:, 0] + ranges[:, 1])
    lim = np.stack([mids + margin * (ranges[:, 0] - mids),
                    mids + margin * (ranges[:, 1] - mids)], axis=1)

    # world joint axes at home -> mirror signs (survives model re-exports)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    def mirror_sign(kl):
        return -1.0 if np.dot(MIRROR_Y * data.xaxis[jids[kl]],
                              data.xaxis[jids[kl + 7]]) > 0 else +1.0

    probed = np.array([mirror_sign(k) for k in range(7)])
    if not np.array_equal(probed, ARM_MIRROR_SIGN):
        raise SystemExit(
            f"MODEL/CODE MIRROR MISMATCH: the loaded model implies arm mirror "
            f"signs {probed.tolist()} but the code carries "
            f"{ARM_MIRROR_SIGN.tolist()}. The deploy model was re-exported with "
            f"a different wrist convention — update ARM_MIRROR_SIGN here AND in "
            f"humanoid_auto_operator.py before moving the arms.")
    q0 = np.concatenate([POWERON_JOINTS, ARM_MIRROR_SIGN * POWERON_JOINTS])
    tr = Trajectory(q0, speed)

    # Baseline = model home WITH the 15 deg outward shoulder tilt already
    # applied. Raw qpos0 rests the grippers INSIDE the hips (the sim monkey
    # says so itself and keeps the tilt for exactly this reason) — on
    # hardware the bare home pose must never be commanded or crossed.
    sh1, sh2 = 0, 1                     # left-arm indices in the 14-vector
    rest = home.copy()
    rest[sh2] = ARM_OUT_TILT_L
    rest[sh2 + 7] = np.clip(home[sh2 + 7] + mirror_sign(sh2) * (ARM_OUT_TILT_L - home[sh2]),
                            lim[sh2 + 7][0], lim[sh2 + 7][1])

    tr.label("ramp: power-on -> tilted home")
    tr.move({k: rest[k] for k in range(14)})
    tr.hold(hold)

    fwd = False
    for kl in range(7):                 # model joint order == sweep order
        name = ARM_JOINTS_L[kl]
        want_fwd = any(k in name for k in STAGED_ARM_KEYS)
        if want_fwd and not fwd:
            tr.label("staging: split arms (left fwd, right back)")
            tr.sweep_pair(sh1, ARM_FORWARD_Q, sh1 + 7, -mirror_sign(sh1), rest, lim)
            rest[sh1], rest[sh1 + 7] = tr.q[sh1], tr.q[sh1 + 7]
            tr.hold(hold)
        sign = -mirror_sign(kl) if kl == sh1 else mirror_sign(kl)
        end = ARM_FORWARD_Q if kl == sh1 else rest[kl]
        tr.label(f"sweep: {name} (+ mirror)")
        for target in (lim[kl][0], lim[kl][1], end):
            tr.sweep_pair(kl, target, kl + 7, sign, rest, lim)
            tr.hold(hold)
        if kl == sh1:                   # segue: the split staging move then no-ops
            rest[sh1], rest[sh1 + 7] = tr.q[sh1], tr.q[sh1 + 7]
            fwd = True

    tr.label("staging: lower arms")
    tr.sweep_pair(sh1, home[sh1], sh1 + 7, -mirror_sign(sh1), rest, lim)
    tr.hold(hold)
    tr.label("return to power-on pose")
    tr.move({k: q0[k] for k in range(14)})
    return tr, qadr


def preflight(model, data, tr: Trajectory, qadr) -> tuple[float, str]:
    """Min arm-to-body / arm-to-arm collision-geom clearance over ALL frames."""
    def subtree(root_body):
        out = set()
        stack = [root_body]
        while stack:
            b = stack.pop()
            out.add(b)
            stack += [c for c in range(model.nbody) if model.body_parentid[c] == b
                      and c != b]
        return out

    def geoms_of(bodies):
        return [g for g in range(model.ngeom)
                if model.geom_bodyid[g] in bodies
                and "collision" in (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]

    armL = subtree(model.jnt_bodyid[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS_L[0])])
    armR = subtree(model.jnt_bodyid[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS_R[0])])
    other = set(range(1, model.nbody)) - armL - armR
    gL, gR, gO = geoms_of(armL), geoms_of(armR), geoms_of(other)
    assert gL and gR and gO, "collision geom sets came up empty — check naming"

    def hops(b1, b2):
        """Kinematic-tree distance between two bodies (via nearest common ancestor)."""
        def chain(b):
            out = [b]
            while b:
                b = model.body_parentid[b]
                out.append(b)
            return out
        c1, c2 = chain(b1), chain(b2)
        return min(i + c2.index(x) for i, x in enumerate(c1) if x in c2)

    # Bodies <=2 hops apart (shoulder mount vs torso) sit at a constant
    # ~10 mm structural proximity for every posture — rigid neighbors, not
    # collision candidates. Verified: shoulder_2_L~base_link is 10.1 mm
    # min AND max over a full trajectory. Elbow is 3 hops, wrists 5.
    pairs = [(a, b) for a, b in
             [(a, b) for a in gL for b in gO] + [(a, b) for a in gR for b in gO]
             + [(a, b) for a in gL for b in gR]
             if hops(model.geom_bodyid[a], model.geom_bodyid[b]) > 2]

    worst, where = np.inf, ""
    for i, f in enumerate(tr.frames):
        data.qpos[:] = model.qpos0
        for k, a in enumerate(qadr):
            data.qpos[a] = f[k]
        mujoco.mj_kinematics(model, data)
        for a, b in pairs:
            dist = mujoco.mj_geomDistance(model, data, a, b, 0.08, None)
            if dist < worst:
                worst = dist
                na = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a)
                nb = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b)
                where = f"frame {i}/{len(tr.frames)} ({i / STREAM_HZ:.1f}s): {na} ~ {nb}"
    return worst, where


def main():
    args = tyro.cli(Args)
    # Floors matter as much as caps: speed<=0 makes every move a single-frame
    # jump (and a negative "rate" exploits the receiver's clamp into a runaway
    # drift); margin<0.5 puts the WRIST_1 rest pose (ref +-pi/2) outside the
    # sweep box and the follower clip then jumps. Adversarial review 07-20.
    # 2026-08-10: this used to say wrist_3, and it was right for the OLD arm.
    # The 07-28 wrist redesign moved the +-pi/2 reference from wrist_3 to
    # wrist_1 (wrist_3 now rests at its own midpoint, range +-pi/2, and can
    # never leave the box). The floor survived only because both joints have
    # +-pi range, so the arithmetic is identical — measured on the deploy
    # model, wrist_1 needs margin >= 0.500 EXACTLY, i.e. this floor has zero
    # slack. Do not lower it without re-deriving it from the model.
    speed = min(max(args.speed, 0.02), SPEED_CAP)
    margin = min(max(args.margin, 0.5), MARGIN_CAP)
    if speed != args.speed or margin != args.margin:
        print(f"[cap] speed -> {speed}, margin -> {margin} (hard caps/floors)")

    model = mujoco.MjModel.from_xml_path(MJCF_MODEL_PATH)
    data = mujoco.MjData(model)
    tr, qadr = build_trajectory(model, data, speed, margin, args.hold)
    step = float(np.abs(np.diff(np.array(tr.frames), axis=0)).max())
    assert step <= speed / STREAM_HZ * 1.1 + 1e-9, \
        f"trajectory discontinuity: {step:.4f} rad/frame — refusing to stream"
    dur = len(tr.frames) / STREAM_HZ
    print(f"model: {MJCF_MODEL_PATH}")
    print(f"trajectory: {len(tr.frames)} frames @ {STREAM_HZ} Hz = {dur / 60:.1f} min, "
          f"speed {speed} rad/s, margin {margin}")

    # Camera follow: built BEFORE the preflight so an unavailable gaze stack is
    # reported while nothing is moving, not three minutes later at the handoff.
    gaze, gaze_data, gaze_sids = None, None, {}
    if args.gaze:
        why = _load_gaze()
        if why is not None:
            print(f"[gaze] camera follow DISABLED — {why}. Arms stream as "
                  f"before; the gimbals hold wherever they are.")
        else:
            gaze_sids = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                         for p, s in GAZE_EE_SITE.items()}
            missing = [s for p, s in GAZE_EE_SITE.items() if gaze_sids[p] < 0]
            if missing:
                print(f"[gaze] camera follow DISABLED — model has no site(s) "
                      f"{missing}")
            else:
                gaze = GazeAim()
                gaze_data = mujoco.MjData(model)
                print(f"[gaze] camera follow ON — "
                      + ", ".join(f"{p}->{GAZE_EE_SITE[p]}" for p in CAM_PORTS)
                      + f" (slew {GAZE_SLEW_RATE} rad/s)")

    print("preflight: scanning clearance over every frame...")
    t0 = time.time()
    worst, where = preflight(model, data, tr, qadr)
    print(f"preflight: min clearance {worst * 1000:.1f} mm at {where} "
          f"({time.time() - t0:.1f}s scan)")
    ok = worst * 1000.0 >= args.min_clearance_mm
    if not ok:
        print(f"preflight: BELOW the {args.min_clearance_mm:.0f} mm floor")

    viz = None
    if args.viz or not args.live:
        hinges = [j for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges]
        viz = HumanoidViserViz(mjcf_path=MJCF_MODEL_PATH, joint_names=names,
                               show_robot=True, port=args.port)
        print(f"viser at http://0.0.0.0:{args.port}")
        while not viz._mj_ready:
            time.sleep(0.1)
        q_full = np.array([model.qpos0[model.jnt_qposadr[j]] for j in hinges])
        arm_k = [names.index(n) for n in ARM_JOINTS_L + ARM_JOINTS_R]
        cam_k = [names.index(n) for n in CAM_JOINT_NAMES if n in names]

        def show(f, g=None):
            q_full[arm_k] = f
            if g is not None and len(cam_k) == len(CAM_JOINT_NAMES):
                q_full[cam_k] = g
            viz.update(imu_quat_wxyz=IDENTITY_WXYZ, wb_pos=np.array([0, 0, 0.6]),
                       wb_quat_wxyz=IDENTITY_WXYZ, joint_pos=q_full)
    else:
        def show(f, g=None):
            pass

    if not args.live:
        print("DRY-RUN (no robot). Playing the trajectory in viser; Ctrl+C to stop.")
        try:
            while True:
                for i, f in enumerate(tr.frames):
                    if i in tr.labels:
                        print(f"  {tr.labels[i]}", flush=True)
                    g = None
                    if gaze is not None:
                        # Offline there is no measured gimbal, so the camera FK
                        # and the trim read the COMMANDED gaze — i.e. an eye
                        # that tracks perfectly. Enough to SEE the follow.
                        jp_v = np.zeros(31)
                        jp_v[TEL_ARM] = f
                        jp_v[27:31] = gaze.gaze
                        gaze.seed(jp_v)
                        g = gaze.step(ee_targets(model, gaze_data, qadr,
                                                 gaze_sids, f), jp_v)
                    show(f, g)
                    time.sleep(1.0 / STREAM_HZ)
        except KeyboardInterrupt:
            print("\nexit")
        return

    # ------------------------------- LIVE -------------------------------
    if not ok:
        raise SystemExit("LIVE REFUSED: preflight clearance below the floor. "
                         "Lower --margin (or raise nothing else).")
    print("\n*** LIVE MODE — HANGING ONLY ***\n"
          "  - robot hanging, legs clear of the ground, E-stop within reach\n"
          "  - T2 running with --enable-motor arm (photographer outside the "
          "sweep envelope)\n"
          "  - gamepad A / Ctrl+C = stop; going silent makes the robot crawl "
          "its arms home on its own failsafe")
    if input("type 'hang' to confirm the robot is HANGING: ").strip() != "hang":
        raise SystemExit("not confirmed — aborting.")

    tel = NNGSubscriber(f"tcp://{args.robot_ip}:9870")
    tel.start()
    print("waiting for telemetry...")
    t0 = time.time()
    jp = None
    while time.time() - t0 < 10.0:
        d = tel.data
        if d and d.get("joint_pos") is not None and len(d["joint_pos"]) >= 31:
            jp = np.asarray(d["joint_pos"], dtype=float)
            if np.all(np.isfinite(jp[TEL_ARM])):
                break
        time.sleep(0.1)
    if jp is None or not np.all(np.isfinite(jp[TEL_ARM])):
        raise SystemExit("no finite telemetry on 9870 — is T2 running/healthy?")
    for arm, sl, mir in (("left", slice(13, 20), np.ones(7)),
                         ("right", slice(20, 27), ARM_MIRROR_SIGN)):
        err = float(np.max(np.abs(jp[sl] - mir * POWERON_JOINTS)))
        if not (err <= POSTURE_TOL):        # NaN counts as refusal
            raise SystemExit(
                f"STARTUP REFUSED: {arm} arm {err:.2f} rad off the power-on pose "
                f"(tol {POSTURE_TOL}). Restart T2 / home the arm first.")
        print(f"  {arm} arm at power-on pose (err {err:.2f} rad)")

    pub = NNGPublisher("tcp://*:9874")
    print(f"streaming in {args.start_delay:.0f}s...", flush=True)
    time.sleep(args.start_delay)

    diverged_since = None
    frame = tr.frames[0]
    jp_now = jp                     # freshest MEASURED 31-vector, for the aim
    gz = None                       # last published gaze, held on the way out
    if gaze is not None and not gaze.seed(jp_now):
        print("[gaze] gimbal encoders unreadable — camera follow OFF for this "
              "run (streaming a zeros-seeded gaze would slam a parked gimbal)")
        gaze = None
    try:
        next_t = time.time()
        for i, frame in enumerate(tr.frames):
            if i in tr.labels:
                print(f"  {tr.labels[i]}", flush=True)
            packet = {"arm_targets": {
                "joint_pos": [float(v) for v in frame],
                "rate": min(speed * 1.2, 0.5)}}    # 20% catch-up headroom
            if gaze is not None:
                # Aim at where the hand is being SENT (this frame), through the
                # gimbal's MEASURED angles. Same packet as the arm targets: the
                # receiver excludes gaze from the arm-silence timer, so nothing
                # about the failsafe changes.
                gz = gaze.step(ee_targets(model, gaze_data, qadr, gaze_sids,
                                          frame), jp_now)
                packet["gaze_targets"] = gz
            pub.publish(packet)
            show(frame, gz)
            d = tel.data
            if d and d.get("joint_pos") is not None and len(d["joint_pos"]) >= 31:
                jp = np.asarray(d["joint_pos"], dtype=float)
                jp_now = jp
                err = float(np.max(np.abs(jp[TEL_ARM] - frame)))
                if not (err <= WATCH_ERR_RAD):   # NaN counts as divergence
                    if diverged_since is None:
                        diverged_since = time.time()
                        print(f"[watch] tracking err {err:.2f} rad — watching")
                    elif time.time() - diverged_since > WATCH_ABORT_S:
                        raise RuntimeError(
                            f"tracking diverged {err:.2f} rad for >{WATCH_ABORT_S}s "
                            "(jam?) — aborting stream")
                else:
                    diverged_since = None
            next_t += 1.0 / STREAM_HZ
            time.sleep(max(0.0, next_t - time.time()))
        print("sweep complete — robot is back at the power-on pose.")
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nstopping: {e if str(e) else 'Ctrl+C'}")
        if isinstance(e, KeyboardInterrupt):
            for _ in range(10):         # hold in place briefly, then go silent
                hold_pkt = {"arm_targets": {
                    "joint_pos": [float(v) for v in frame], "rate": speed}}
                if gz is not None:
                    hold_pkt["gaze_targets"] = gz   # eyes hold too; dropping the
                                                    # key would leave the last
                                                    # reference latched anyway,
                                                    # but saying it is cheaper
                                                    # than relying on that
                pub.publish(hold_pkt)
                time.sleep(1.0 / STREAM_HZ)
        # watchdog abort (jam): go silent IMMEDIATELY — feeding a jammed
        # command presses harder; the receiver failsafe is the recovery.
        print("stream silenced — real_env failsafe crawls the arms to the "
              "default pose at 0.025 rad/s.")
    finally:
        pub.close()
        tel.stop()


if __name__ == "__main__":
    main()
