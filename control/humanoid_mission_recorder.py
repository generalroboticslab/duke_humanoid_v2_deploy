#!/usr/bin/env python3
"""Record both camera streams to MP4, one file per camera per mission run.

Usage (its own terminal, before or after T6 — it waits):
    python humanoid_mission_recorder.py
    python humanoid_mission_recorder.py --out /path/to/recordings

WHAT DEFINES A RUN. The recorder subscribes to the reach tool's 9874
publisher and watches gaze_targets:

  * START  — the first gaze packet. humanoid_curobo_reach publishes from its
             camera-zeroing phase onward, so recording begins exactly where
             the user wants it: "from zeroing".
  * STOP   — after the gimbals have visibly moved (any |gaze| > 0.3 rad),
             their return to ~zero held for 3 s marks the post-grasp park
             (mission DONE parks both cameras at neutral); 3 s of 9874
             silence (T6 exited / Ctrl+C) also stops. Files are finalized
             and the recorder re-arms for the next run.

WHY NETWORK STREAMS, NOT THE DEVICES. camera_record.py opens the RealSense
directly and would fight T1's streaming server for the hardware. This tool
is one more subscriber on ports 5555/5556 — zero contention, and the frames
are exactly what the detection stack saw.

DISK GUARD. The robot computer's root sits near-full; recording refuses to START a
run when free space is under REC_MIN_FREE_GB and stops mid-run if it drops
under half of that, finalizing playable files either way.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# 2026-08-22: the canonical publisher. (Until the control-side package was
# renamed `common` -> `ipc` this file imported the hardware_bindings twin to
# dodge visual_servoing's same-named `common` under test discovery.)
from ipc.publisher import NNGSubscriber  # noqa: E402

from humanoid_site import VISUAL_SERVOING_ROOT as _VS_ROOT  # noqa: E402
import humanoid_site as _site  # noqa: E402
sys.path.insert(0, str(_VS_ROOT))

from camera_tag_detector import StreamingCameraBackend  # noqa: E402

CAM_PORTS = (_site.CAMERA_BASE_PORT, _site.CAMERA_BASE_PORT + 1)   # left, right (5555, 5556)
GAZE_URL = f"tcp://127.0.0.1:{_site.ARM_COMMAND_PORT}"            # the 9874 arm/gaze stream
REC_FPS = 30                # nominal writer rate; streams deliver ~30 Hz
REC_MOVED_RAD = 0.3         # |gaze| beyond this = the sweep visibly started;
                            # only after this can a return-to-zero mean "done"
REC_PARK_TOL_RAD = 0.08     # all |gaze| under this counts as parked at zero
REC_PARK_HOLD_S = 3.0       # parked this long after motion = mission done
REC_SILENT_S = 3.0          # 9874 quiet this long = T6 gone; close the files
REC_MIN_FREE_GB = 1.0       # refuse to start a recording under this


class MissionWindow:
    """Pure trigger logic: feed(now, gaze-or-None) -> 'start' | 'stop' | None.

    Extracted from the I/O loop so the state machine is testable without
    sockets. gaze is the 4-vector from a 9874 packet, or None on a tick with
    no fresh packet."""

    def __init__(self):
        self.recording = False
        self._moved = False
        self._park_since: float | None = None
        self._last_pkt: float | None = None

    def feed(self, now: float, gaze) -> str | None:
        if gaze is not None:
            self._last_pkt = now
            if not self.recording:
                self.recording = True
                self._moved = False
                self._park_since = None
                return "start"
            g = max(abs(float(v)) for v in gaze)
            if g > REC_MOVED_RAD:
                self._moved = True
                self._park_since = None
            elif self._moved and g < REC_PARK_TOL_RAD:
                if self._park_since is None:
                    self._park_since = now
                elif now - self._park_since >= REC_PARK_HOLD_S:
                    self.recording = False
                    return "stop"
            else:
                self._park_since = None
        elif self.recording and self._last_pkt is not None \
                and now - self._last_pkt >= REC_SILENT_S:
            self.recording = False
            return "stop"
        return None


def _free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_site.RECORDINGS_DIR))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    backends = {}
    for port in CAM_PORTS:
        b = StreamingCameraBackend("127.0.0.1", port)
        b.initialize(1280, 720, REC_FPS)
        b.capture_frame_continuously()
        backends[port] = b
    sub = NNGSubscriber(GAZE_URL)
    sub.start()
    print(f"[rec] watching {GAZE_URL} for a mission; cameras "
          f"{list(CAM_PORTS)}; out={args.out}  (Ctrl+C to quit)")

    window = MissionWindow()
    writers: dict[int, cv2.VideoWriter] = {}
    paths: dict[int, str] = {}
    frames: dict[int, int] = {}
    last_stamp: dict[int, float] = {}
    last_id = None
    try:
        while True:
            tick = time.monotonic()
            gaze = None
            if sub.data_id != last_id and sub.data is not None:
                last_id = sub.data_id
                g = sub.data.get("gaze_targets")
                if g is not None and len(g) >= 4:
                    gaze = g
            event = window.feed(tick, gaze)

            if event == "start":
                if _free_gb(args.out) < REC_MIN_FREE_GB:
                    print(f"[rec] REFUSING to record: {_free_gb(args.out):.1f} "
                          f"GB free < {REC_MIN_FREE_GB} GB floor — clear disk")
                    window.recording = False
                else:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    for port in CAM_PORTS:
                        paths[port] = os.path.join(
                            args.out, f"{stamp}_cam{port}.mp4")
                        writers[port] = cv2.VideoWriter(
                            paths[port], fourcc, REC_FPS, (1280, 720))
                        frames[port] = 0
                        last_stamp[port] = 0.0
                    print(f"[rec] ● mission started — recording to "
                          f"{stamp}_cam{{{','.join(map(str, CAM_PORTS))}}}.mp4")

            if writers:
                for port, b in backends.items():
                    st = float(getattr(b, "last_frame_recv_stamp", 0.0) or 0.0)
                    if st <= last_stamp[port]:
                        continue                    # no new frame yet
                    last_stamp[port] = st
                    f = b.get_frame_buffer()
                    if f is None:
                        continue
                    if f.ndim == 2:
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    if f.shape[1::-1] != (1280, 720):
                        f = cv2.resize(f, (1280, 720))
                    writers[port].write(f)
                    frames[port] += 1
                if _free_gb(args.out) < REC_MIN_FREE_GB / 2:
                    print("[rec] disk critically low — finalizing early")
                    event = "stop"
                    window.recording = False

            if event == "stop" and writers:
                for port, w in writers.items():
                    w.release()
                    print(f"[rec] ■ saved {paths[port]}  "
                          f"({frames[port]} frames, "
                          f"{os.path.getsize(paths[port]) / 1e6:.1f} MB)")
                writers.clear()
                print("[rec] re-armed — waiting for the next mission")

            time.sleep(max(0.0, 1.0 / REC_FPS - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        for port, w in writers.items():
            w.release()
            print(f"[rec] ■ saved {paths[port]} ({frames[port]} frames)")
        print("[rec] bye")


if __name__ == "__main__":
    main()
