"""Site configuration for visual_servoing: what changes between installations.

Same contract as the control stack's `humanoid_site`: each value is resolved
from, in order, a `VS_<NAME>` environment variable, an optional gitignored
`vs_site_local.py`, and a portable default. Camera serial numbers, robot
addresses and sibling-checkout paths are properties of one physical lab, not
of this code, so none of them are literals any more.

Nothing here is a calibration value. Tag sizes, board layouts, camera
intrinsics and the tagged-body geometry stay with the bodies they describe --
those are measured facts about physical objects, not installation settings.
"""
from __future__ import annotations

import os
from pathlib import Path

try:                                    # pragma: no cover - presence varies
    import vs_site_local as _local      # type: ignore
except ImportError:
    _local = None


def _resolve(name: str, default):
    env = os.environ.get(f"VS_{name}")
    if env is not None:
        return env
    if _local is not None and hasattr(_local, name):
        return getattr(_local, name)
    return default


def _path(name: str, default) -> Path:
    return Path(_resolve(name, default)).expanduser()


PACKAGE_ROOT: Path = Path(__file__).resolve().parent
"""This repository's own directory. Discovered, never configured."""

REPO_ROOT: Path = _path("REPO_ROOT", Path.home() / "repo")
LEGGED_ENV_ROOT: Path = _path("LEGGED_ENV_ROOT", REPO_ROOT / "legged_env_v2")
LEGGED_ENV_DEV_ROOT: Path = _path("LEGGED_ENV_DEV_ROOT", REPO_ROOT / "legged_env_dev")
"""Not read by anything in this tree (kept for the lab's other robots' scripts)."""
CONTROL_ROOT: Path = _path("CONTROL_ROOT", REPO_ROOT / "DukeHumanoidV2" / "control")
"""Not read by anything in this tree (kept for the lab's other robots' scripts);
the default names the pre-release layout -- the control stack locates itself via
`humanoid_site.CONTROL_ROOT`."""
MINK_ROOT: Path = _path("MINK_ROOT", REPO_ROOT / "mink" / "src")
"""Not read by anything in this tree (kept for the lab's other robots' scripts)."""
MJLAB_ROOT: Path = _path("MJLAB_ROOT", REPO_ROOT / "mjlab" / "src")
"""Not read by anything in this tree (kept for the lab's other robots' scripts)."""

UR5E_IP: str = str(_resolve("UR5E_IP", "127.0.0.1"))
"""UR5e arm controller address. Only read by the ur5e_* helper scripts, which are
not in this tree (kept for the lab's other robots' scripts)."""


def _serials(default: str) -> tuple[str, ...]:
    raw = _resolve("CAMERA_SERIALS", default)
    if isinstance(raw, (list, tuple)):
        return tuple(str(s) for s in raw)
    return tuple(s for s in str(raw).replace(",", " ").split() if s)


CAMERA_SERIALS: tuple[str, ...] = _serials("")
"""RealSense serials for this rig, in stream order (index 0 -> port 5555 left,
index 1 -> 5556 right).

Empty by default: a serial number names one physical camera and cannot be
guessed. Read by `camera_tag_detection.py` (the standalone detector, which
enumerates attached devices when this is empty) and, when non-empty, by
`camera_streaming_utils.py multi-server` as the `--devices` default -- so this
is how a rig pins its camera->port order once; an explicit `--devices` still
wins, and left empty the order is USB enumeration.
"""
