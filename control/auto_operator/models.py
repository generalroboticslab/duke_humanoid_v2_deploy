"""Typed state models for the auto-operator (refactor stage S2).

S2 contract (see docs/auto_operator_refactor_plan.md): representation ONLY.
Each class replaces a string-keyed runtime dict at its CONSTRUCTION site; every
one of the ~200 existing access sites keeps its exact textual form through the
dict-style access shim below. No default here may differ from the legacy
behavior of a missing key at its `get`/`setdefault` call sites — each field
documents that correspondence. Access sites migrate to attributes in later
stages, key by key, gated by the replay harness.

State *vocabularies* (grasp lifecycle, sectors) stay plain strings in S2 —
converting `h["grasp"] is None` / `== "sent"` string comparisons to enums is a
behavior-adjacent change reserved for a later stage with its own gate. The
constants below document the legal values.
"""
from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import numpy as np


class GraspState:
    """Legal values of HoldState.grasp (legacy: the hold dict's "grasp" key).

    Lifecycle: NONE -> SENT -> LIFTING -> GRASPED -> CARRIED, or -> FAILED
    (plain hold), or -> UNCERTAIN (terminal human-intervention latch). NONE is
    the Python literal None in legacy comparisons.
    """
    NONE = None
    SENT = "sent"
    LIFTING = "lifting"
    GRASPED = "grasped"
    CARRIED = "carried"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class Sector:
    """Legal values of HoldState.sector / _reach_sector."""
    FRONT = "front"
    SIDE = "side"
    REAR = "rear"


class ArmTaskPhase:
    """Lifecycle of one arm's independent pre-hold task.

    The grasp/lift/carry half of the lifecycle continues in :class:`HoldState`
    after a reach is registered.  Keeping these values as strings matches the
    rest of the mechanically-extracted operator state and keeps replay/debug
    dumps readable.
    """

    ASSIGNED = "assigned"      # owns a cube; waiting for per-arm reachability
    WAIT_STOP = "wait_stop"    # reachable; waiting for the shared base to stop
    STAGING = "staging"        # owns/waits for the global joint-mode token
    REACHING = "reaching"      # Cartesian gentle approach


class _DictAccess:
    """Legacy dict-style access shim (S2 only).

    Lets the operator logic keep `x["key"]`, `x.get("key", d)` and
    `x.setdefault("key", d)` verbatim while the state container is typed.
    All legal keys are declared dataclass fields, so `setdefault` never
    inserts — it returns the current value, which matches dict semantics
    because every field's declared default equals the `setdefault` default
    at its call sites (asserted field-by-field in the docstrings below).
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if not hasattr(self, key):          # unreachable for declared fields;
            setattr(self, key, default)     # kept for dict parity
        return getattr(self, key)


@dataclasses.dataclass
class HoldState(_DictAccess):
    """Per-arm hold entry (legacy: the 15+-key dict built in _register_hold).

    Field defaults mirror the legacy dict exactly:
    - keys always present at construction keep their constructor values;
    - keys that legacy code created lazily (`via_home`, `drop_t0`,
      `carry_warn_t`, `grasp_diag_t`, `lift_wps`, `lift_t0`) default to the
      value the legacy `get`/`setdefault` call sites supplied for a MISSING
      key: via_home False (h.get("via_home", False)), drop_t0 None
      (h.get("drop_t0")), carry_warn_t/grasp_diag_t -1e9 (h.setdefault(k,-1e9)),
      lift_wps []/lift_t0 0.0 (always assigned before first read).
    """
    key: str
    target: np.ndarray
    bias: np.ndarray | None
    sector: str
    sector_idx: int
    home: bool = False
    in_hits: int = 0
    out_hits: int = 0
    best_arm: str | None = None
    wrong_t0: float | None = None
    state: str = "tracking"
    track: "TrackMotion | None" = None
    restream: np.ndarray | None = None
    grasp: str | None = GraspState.NONE
    grasp_t0: float = 0.0
    grasp_base: Any = None
    grasp_action_id: int | None = None
    grasp_tries: int = 0
    grasp_move_t: float = 0.0
    via_home: bool = False
    resume_pending: bool = False
    drop_t0: float | None = None
    carry_warn_t: float = -1e9
    grasp_diag_t: float = -1e9
    lift_wps: list = dataclasses.field(default_factory=list)
    lift_t0: float = 0.0


@dataclasses.dataclass
class TrackMotion(_DictAccess):
    """Bidirectional staging-track motion (legacy: the dict from _new_track)."""
    arm: str
    cur: int
    target: int
    seq: list
    frozen: list | None = None
    hop_t0: float = 0.0
    hop_budget: float = 0.0
    dwell_until: float | None = None
    timed_out: bool = False


@dataclasses.dataclass
class ServoState(_DictAccess):
    """Visual-servo estimator state owned by one arm task.

    The legacy coordinator stores this vocabulary in global ``_servo_*``
    attributes.  Independent arms require one copy per task so a new reach on
    one side cannot reset or overwrite the other side's estimator.
    """

    bias: np.ndarray | None = None
    seen: bool = False
    accepted: int = 0
    stamp: float | None = None
    cmd_prev: np.ndarray | None = None
    ee_prev: np.ndarray | None = None
    gimbal_prev: np.ndarray | None = None
    accept_t: float = 0.0
    warned: set = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class ArmTask(_DictAccess):
    """One symmetric arm assignment from target claim through reach.

    Left and right use this exact same state machine.  ``track`` may have to
    wait for the shared 14-joint wire token, but waiting never changes the
    task's target, timer, or ownership.  ``reach_elapsed`` advances only while
    this arm is actually granted a Cartesian command, so another arm's joint
    critical section cannot time it out. ``joint_not_before`` gives held-arm
    Cartesian updates from the latch tick one publish opportunity before a new
    staircase switches the shared wire mode.
    """

    arm: str
    key: str
    phase: str = ArmTaskPhase.ASSIGNED
    assigned_t: float = 0.0
    reach_hits: int = 0
    target: np.ndarray | None = None
    sector: str = Sector.FRONT
    sector_idx: int = 0
    port: int | None = None
    track: TrackMotion | None = None
    joint_not_before: float = 0.0
    stream: np.ndarray | None = None
    reach_t0: float = 0.0
    reach_elapsed: float = 0.0
    servo: ServoState = dataclasses.field(default_factory=ServoState)


@dataclasses.dataclass
class SecondaryReach(_DictAccess):
    """Dual-parallel secondary approach (legacy: the _sec dict)."""
    arm: str
    key: str
    target: np.ndarray
    t0: float
    stream: np.ndarray | None = None


@dataclasses.dataclass
class GripperBurst:
    """One ee_action repeat burst (legacy: the [payload, remaining, sent] list).

    Supports integer indexing and unpacking so the arbitration block keeps its
    exact textual form (`burst[1] -= 1`, `cmd, remaining, _ = burst`).
    """
    payload: dict
    remaining: int
    sent: int = 0

    _FIELDS = ("payload", "remaining", "sent")

    def __getitem__(self, i: int) -> Any:
        return getattr(self, self._FIELDS[i])

    def __setitem__(self, i: int, value: Any) -> None:
        setattr(self, self._FIELDS[i], value)

    def __iter__(self):
        return iter((self.payload, self.remaining, self.sent))


class Sighting(NamedTuple):
    """One accepted cube detection, as stored in CubeTrack.hist.

    WAS a bare (t, pos, port) tuple. Two fields were missing and both cost
    accuracy on the final approach (audit 2026-08-06):

    `faces` — n_inliers, the tag faces behind this PnP fit. Without it the
    reach latch's median could not tell a three-face fit from a one-face
    guess, so a single 1-face sample (this codebase's own figure: "tens of
    mm", leash 120 mm) pulled the median by its full weight — and with two
    samples np.median IS the mean, so it pulled it halfway. Consumers that
    want only the best evidence available can now ask for it.

    `t` is CAPTURE time, not ingest time. The monitor retains each key's
    newest pose for DETECTION_RETENTION_S and republishes it in EVERY 20 Hz
    packet, so stamping on arrival let ONE decode fill all five slots with
    five different timestamps: the median over five identical values is that
    value, and the averaging this deque exists for silently disappeared.
    Deduplicating on capture stamp (see auto_operator/perception.py) is what
    keeps the window five DISTINCT observations.
    """

    t: float                       # capture time [s], monitor's monotonic clock
    pos: np.ndarray                # base-frame position [m]
    port: int                      # camera that measured it
    faces: int                     # n_inliers: tag faces behind the PnP fit
    quat: np.ndarray | None = None
    # 2026-08-13 (user: "could we include the orientation into the cuRobo
    # planner?"). The monitor has always published `quat_wxyz` beside `pos`
    # for every detected body, and this track dropped it on the floor — so
    # the whole pipeline below assumed the cube was axis-aligned: the scene
    # cuboid was written with an identity quaternion, and the grasp goalset
    # (generated in the CUBE's frame, then composed with obj_quat) was
    # rotated by nothing. A cube yawed 30-45 deg — which is exactly how the
    # operator is told to place them, so a second tag face is visible — then
    # got jaw approach directions computed for a square-on cube, closing
    # corner-to-corner: a 60 mm cube presents its 85 mm diagonal to a 90 mm
    # jaw opening, two-point contact with 5 mm to spare.
    # OPTIONAL with a None default on purpose: every other Sighting(...)
    # construction site in the tree keeps working untouched, and a publisher
    # that omits the field degrades to the old behaviour instead of raising.
