# python humanoid_camera_view.py
"""Live-view (and optionally record) BOTH head cameras while a mission /
visual servoing runs.

Subscribes to the same pynng streams the monitor uses (camera_streaming_utils
multi-server on 5555/5556) and shows them in OpenCV windows; with --record it
also writes one .mp4 per camera plus an optional side-by-side file. Read-only
on the network: the streams are Pub0/Sub0 best-effort broadcast, so an extra
subscriber changes nothing for the monitor or the operator — start and stop
this at any time, mid-run included.

The camera server (opens the RealSense devices, binds 5555/5556) must ALREADY
be up. That is the command you run first:

    python <visual_servoing>/camera_streaming_utils.py multi-server \\
      --backend realsense --devices <serial> <serial> --base-port 5555 --quality 80

Then, on the machine with a screen:

    python humanoid_camera_view.py                          # view both cams, side-by-side, localhost
    python humanoid_camera_view.py --host <robot-host>     # camera server on the robot
    python humanoid_camera_view.py --record                 # view + write per-camera .mp4s
    python humanoid_camera_view.py --record --record-combined   # + a side-by-side .mp4
    python humanoid_camera_view.py --record --duration 120  # view + record, auto-stop after 2 min
    python humanoid_camera_view.py --separate               # one window per camera
    python humanoid_camera_view.py --snap --snap-tag scene1 # ONE PHOTO per camera
                                                            # (+ side-by-side), then exit
    python humanoid_camera_view.py --offline-format --no-display --duration 60
                                                            # .mp4 + .npz replayable offline

--offline-format writes each camera in the recording format the visual_servoing
tooling already speaks — `<base>.mp4` next to `<base>.npz` (K, fps, resolution,
tag_size, tag_sizes, undistorted), exactly the pair
`camera_offline_utils.OfflineCameraBackend` loads (and `camera_record.py` writes
for a single local camera). Replay a take through the normal tag pipeline with:

    import sys; sys.path.insert(0, str(VISUAL_SERVOING_ROOT))
    from camera_offline_utils import OfflineCameraBackend
    cam = OfflineCameraBackend("<recordings>/0723_181500_cam5555_left")
    cam.initialize(); cam.capture_frame_continuously()

Press q or ESC (with a window focused) to quit; Ctrl+C in the terminal also
exits cleanly (windows are destroyed and every VideoWriter is released in
`finally`, so a killed recording still produces playable video).

WHERE TO RUN: on the machine with a display AND network reach to the camera
server's host:base_port. --host is the camera server's IP as seen from THIS
machine (127.0.0.1 when the server runs locally) and must match the --robot-ip
the operator/monitor use for the camera streams.

Design decisions:
- Frames are COPIED out of each backend's reusable buffer before use, same as
  the recorder: capture_all_parallel overwrites those buffers on the recv
  thread, so a view/write into them would tear.
- RECORDING writes the full-resolution stamped frame; --max-width only shrinks
  the DISPLAY (cv2.imshow), never the saved video.
- Per-camera files are written whenever --record. The side-by-side file
  (--record-combined) is written ONLY on iterations where every camera produced
  a fresh frame — a VideoWriter needs a fixed frame size, and a half-pair on a
  dropped frame would change the width. Display has no such constraint, so the
  window still updates from whatever arrived.
- Timestamped filenames, never a fixed name: a demo session records many takes
  and silently overwriting the good one is unacceptable.
- Writers open LAZILY on the first frame (the server decides resolution) and the
  container FPS is a declared constant (--fps); a burned-in frame counter, FPS
  meter and wall-clock make the true timing recoverable from the picture itself.
- --offline-format saves the per-camera .mp4 RAW — no burned-in overlay, and
  grayscale when the stream is grayscale (--ir), colour when it is colour. The
  overlay would sit on top of the image an AprilTag detector reads, and the
  offline file exists to be re-detected, not watched; --record-combined still
  writes the stamped side-by-side file for watching, so one run yields both.
- The .npz `fps` is the MEASURED arrival rate, not --fps. OfflineCameraBackend
  paces playback from that field, and the stream delivers whatever the server
  and the network allow (typically well under 30) — a declared 30 on a 12 fps
  take would replay 2.5x too fast and break any timing read off the replay.
- No re-encode of the stream: frames arrive as JPEG (server --quality) and are
  re-compressed by mp4v, so an offline take is slightly lossier than what the
  live detector saw. Raise the server's --quality for detection-critical takes.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import tyro

from humanoid_site import VISUAL_SERVOING_ROOT as _VS_ROOT, RECORDINGS_DIR as _REC_DIR  # noqa: E402
sys.path.insert(0, str(_VS_ROOT))
from camera_streaming_utils import MultiStreamingCameraClient  # noqa: E402

CAM_LABEL = {0: "cam5555_left", 1: "cam5556_right"}   # port order == stream order
WINDOW_COMBINED = "head cameras (left | right)"


class _MJPEGState:
    """Latest JPEG frame shared from the capture loop to HTTP stream handlers.
    A Condition lets each /stream handler BLOCK until a new frame exists (no busy
    spin, no stale re-sends), and wakes them at shutdown."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._alive = True

    def update(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq: int, timeout: float = 4.0):
        """Return (jpeg, seq) once seq advances past last_seq, or (None, last_seq)
        on timeout/shutdown so the handler can re-check the connection."""
        with self._cond:
            if self._alive and self._seq == last_seq:
                self._cond.wait(timeout)
            if not self._alive:
                return None, last_seq
            return self._jpeg, self._seq

    def shutdown(self) -> None:
        with self._cond:
            self._alive = False
            self._cond.notify_all()


_INDEX_HTML = (
    b"<!doctype html><meta charset=utf-8><title>head cameras</title>"
    b"<body style='margin:0;background:#111'>"
    b"<img src='/stream' style='width:100vw;height:100vh;object-fit:contain'>"
    b"</body>"
)


def _make_handler(state: _MJPEGState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):    # silence per-request stderr spam
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            last = -1
            try:
                while True:
                    jpeg, last = state.wait_next(last)
                    if jpeg is None:          # timeout (re-check socket) or shutdown
                        if not state._alive:
                            break
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode()
                                     + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                          # browser closed the tab; drop quietly
    return Handler


@dataclass
class Args:
    host: str = "127.0.0.1"
    """Camera-stream host (the robot/workstation running multi-server), as seen
    from this machine. Must match the operator/monitor --robot-ip."""
    base_port: int = 5555
    """First stream port; camera i listens on base_port + i."""
    n_cameras: int = 2
    """How many streams to view/record."""
    separate: bool = False
    """Display: one window per camera instead of a single side-by-side window."""
    max_width: int = 0
    """Shrink the DISPLAYED image so its width <= this (0 = full size). Never
    affects the recorded video."""
    overlay: bool = True
    """Burn label + frame number + measured FPS + wall-clock into the picture
    (shown and, when recording, saved)."""
    display: bool = True
    """Open OpenCV windows. --no-display runs HEADLESS (record-only / --web) — use
    it when running over SSH on the robot: cv2 draws on the server's own monitor
    (DISPLAY :0), which you can't see from an SSH client. To view remotely instead,
    use --web (browser), reconnect with `ssh -Y`, or run on your own computer."""
    web: bool = False
    """Serve the live side-by-side feed as an MJPEG stream over HTTP. Open
    http://<server>:<web_port>/ in a browser on ANY machine that can reach the
    port — the RIGHT way to watch from an SSH client (no X, no local deps). If the
    port isn't directly reachable, tunnel it: `ssh -L 8090:localhost:8090 <server>`
    then browse http://localhost:8090/. Pairs naturally with --no-display."""
    web_port: int = 8090
    """TCP port for the --web MJPEG server."""
    web_host: str = "0.0.0.0"
    """Bind address for --web (0.0.0.0 = all interfaces; 127.0.0.1 = tunnel-only)."""
    web_quality: int = 80
    """JPEG quality (1-100) for the --web stream."""
    snap: bool = False
    """ONE-SHOT PHOTO: save the next frame of EVERY camera as a full-resolution
    PNG in --out-dir (raw, no overlay) plus a side-by-side PNG, then exit. Made
    for the frozen grasp pose (operator --photo-freeze): run it, get the shots,
    it stops by itself. Needs no window (works headless over SSH)."""
    snap_warmup: float = 0.7
    """Seconds of stream discarded before --snap fires. A fresh subscription can
    deliver one queued frame from before the connect, which for a photo session
    would silently show the PREVIOUS pose — this waits it out."""
    snap_tag: str = ""
    """Optional name stitched into the --snap filenames (e.g. --snap-tag scene1),
    so a two-scene session cannot mix its four photos up."""
    record: bool = False
    """Also write one .mp4 per camera to --out-dir."""
    record_combined: bool = False
    """When recording, additionally write a side-by-side .mp4 (both cameras)."""
    offline_format: bool = False
    """Write the per-camera recording in the visual_servoing OFFLINE-REPLAY
    format: raw (un-stamped) .mp4 + a matching .npz sidecar, so
    camera_offline_utils.OfflineCameraBackend can feed the take back through the
    tag-detection pipeline. Implies --record."""
    tag_size: float = 0.03
    """Metadata only: AprilTag edge length [m] recorded in the .npz (30mm tags)."""
    tag_id_range: tuple[int, int] = (0, 63)
    """Metadata only: inclusive tag-id range mapped to --tag-size in the .npz
    `tag_sizes` dict. Per-body sizes come from the tagged_rigid_body specs at
    detection time; this is the uniform convenience map camera_record.py writes."""
    undistorted: bool = True
    """Metadata only: mark the recording as already undistorted — true for the
    RealSense streams (factory-calibrated; K arrives in the stream header)."""
    out_dir: str = str(_REC_DIR)
    """Directory for the .mp4 files (created if missing)."""
    fps: float = 30.0
    """FPS written into the .mp4 container header (see the docstring caveat)."""
    duration: float = 0.0
    """Auto-stop after this many seconds; 0 = run until q/ESC/Ctrl+C."""
    quiet: bool = False
    """Suppress the periodic terminal progress line."""


def _stamp(img: np.ndarray, label: str, n: int, fps: float) -> np.ndarray:
    out = img.copy()
    txt = (f"{label}  #{n:06d}  {fps:4.1f} fps  "
           f"{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _write_sidecar(path: str, K: np.ndarray, fps: float, size: tuple[int, int],
                   tag_size: float, tag_ids: tuple[int, int],
                   undistorted: bool) -> None:
    """Write the .npz metadata that turns a recorded .mp4 into an offline take.

    Inputs: sidecar path, 3x3 intrinsics from the stream header, MEASURED fps,
    (width, height), tag edge length [m], inclusive tag-id range, undistorted flag.

    The key set (K, tag_size, tag_sizes, fps, resolution, undistorted) is not a
    choice: camera_offline_utils.OfflineCameraBackend indexes exactly those and
    raises on any missing one, and camera_record.py writes the same set — the
    offline player must not care which recorder produced the take. tag_sizes is
    a plain dict, saved as an object array, which is what the player's .item()
    call expects.
    """
    lo, hi = tag_ids
    np.savez(path,
             K=np.asarray(K, dtype=np.float64),
             tag_size=float(tag_size),
             tag_sizes={i: float(tag_size) for i in range(lo, hi + 1)},
             fps=float(fps),
             resolution=np.array(size, dtype=np.int32),
             undistorted=bool(undistorted))


def _fit_width(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _hstack_pair(tiles: list[np.ndarray]) -> np.ndarray:
    """Side-by-side, matching every tile's height to the tallest (scale others)."""
    h = max(t.shape[0] for t in tiles)
    matched = []
    for t in tiles:
        if t.shape[0] != h:
            scale = h / t.shape[0]
            t = cv2.resize(t, (int(t.shape[1] * scale), h))
        matched.append(t)
    return matched[0] if len(matched) == 1 else np.hstack(matched)


def _probe_display() -> tuple[bool, str]:
    """Can cv2 actually open a window here? Returns (ok, reason-if-not). Briefly
    creates and destroys a probe window; swallows the failure so the caller can
    fall back to headless recording instead of crashing mid-run."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False, "no DISPLAY/WAYLAND_DISPLAY set (headless shell)"
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        cv2.waitKey(1)
        return True, ""
    except cv2.error as e:
        return False, str(e).splitlines()[-1][:160]


def main():
    args = tyro.cli(Args)

    record = args.record or args.offline_format   # --offline-format implies --record

    show = args.display
    if show:
        ok, why = _probe_display()
        if not ok:
            print(f"[view] cannot open a window: {why}")
            if not record and not args.web and not args.snap:
                print("[view] nothing to do without a window and without --record/--web. "
                      "Fix the display, or add --web (browser) / --record (.mp4).")
                raise SystemExit(1)
            print("[view] continuing HEADLESS (--web / --record only)")
            show = False
        elif os.environ.get("SSH_CONNECTION") and \
                os.environ.get("DISPLAY", "").startswith(":") and not args.web:
            print(f"[view] NOTE: SSH session + local DISPLAY ({os.environ['DISPLAY']}) "
                  "— the window opens on the SERVER's monitor, not your SSH client. "
                  "To watch from here: use --web (browser), reconnect with `ssh -Y`, "
                  "or run this on your own computer. Use --no-display to record only.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if record else None
    session = _dt.datetime.now().strftime("%m%d_%H%M%S")
    if record or args.snap:
        os.makedirs(args.out_dir, exist_ok=True)
    snap_latest: dict[int, np.ndarray] = {}   # newest post-warmup frame per camera
    writers: dict[int, cv2.VideoWriter] = {}   # per-camera
    paths: dict[int, str] = {}
    combined_writer: cv2.VideoWriter | None = None
    combined_path = ""
    rec_counts = {i: 0 for i in range(args.n_cameras)}   # frames WRITTEN per cam
    rec_sizes: dict[int, tuple[int, int]] = {}           # (w, h) actually written
    rec_span = {i: [0.0, 0.0] for i in range(args.n_cameras)}  # [first, last] write time
    combined_count = 0

    client = MultiStreamingCameraClient(args.host, args.n_cameras, args.base_port)
    client.initialize_all()
    backends = client._backends   # the only accessor for the frame buffers
    print(f"[view] waiting for {args.n_cameras} streams at "
          f"{args.host}:{args.base_port}…")
    if not client.wait_all(timeout=10.0):
        print("[view] no frames — is the camera_streaming_utils multi-server up "
              f"at {args.host}:{args.base_port}? (it must already be running)")
        client.cleanup()
        raise SystemExit(1)

    # Intrinsics arrive in the stream header; read them now, while the backends
    # are alive, because the sidecars are written after client.cleanup().
    intrinsics = {i: backends[i].get_intrinsics() for i in range(args.n_cameras)} \
        if args.offline_format else {}
    if args.offline_format:
        print("[rec] offline format: per-camera .mp4 written RAW (no overlay) "
              "+ .npz sidecar (K, measured fps, resolution, tag sizes)")

    web_state: _MJPEGState | None = None
    httpd: ThreadingHTTPServer | None = None
    if args.web:
        web_state = _MJPEGState()
        httpd = ThreadingHTTPServer((args.web_host, args.web_port),
                                    _make_handler(web_state))
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="mjpeg").start()
        where = "localhost" if args.web_host in ("127.0.0.1", "localhost") \
            else args.host
        print(f"[view] MJPEG stream: http://{where}:{args.web_port}/  "
              f"(from an SSH client: `ssh -L {args.web_port}:localhost:{args.web_port} "
              f"<server>` then http://localhost:{args.web_port}/)")

    counts = {i: 0 for i in range(args.n_cameras)}       # frames RECEIVED per cam
    fps = {i: 0.0 for i in range(args.n_cameras)}        # EMA of arrival rate
    last_t = {i: None for i in range(args.n_cameras)}
    t0 = time.time()
    last_report = t0

    try:
        while True:
            # capture_all_parallel returns list[bool] ("a new frame arrived"), NOT
            # the images — those live in each backend's reusable buffer and must be
            # COPIED before the recv thread overwrites them.
            fresh = client.capture_all_parallel(timeout=0.5)
            now = time.time()
            shown = []   # full-res stamped frame per camera this iteration (or None)
            raws = []    # same frame untouched — what --offline-format records
            for i, ok in enumerate(fresh):
                if not ok:
                    shown.append(None)
                    raws.append(None)
                    continue
                buf = backends[i].get_frame_buffer()
                if buf is None or getattr(buf, "size", 0) == 0:
                    shown.append(None)
                    raws.append(None)
                    continue
                frame = np.array(buf, copy=True)
                raws.append(frame)
                img = frame if frame.ndim == 3 else \
                    cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                counts[i] += 1
                if last_t[i] is not None:
                    dt = now - last_t[i]
                    if dt > 0:
                        inst = 1.0 / dt
                        fps[i] = inst if fps[i] == 0.0 else 0.9 * fps[i] + 0.1 * inst
                last_t[i] = now
                label = CAM_LABEL.get(i, f"cam{args.base_port + i}")
                shown.append(_stamp(img, label, counts[i], fps[i])
                             if args.overlay else img)

            # ---- one-shot photo: raw full-res PNG per camera, then quit ----
            if args.snap:
                if now - t0 >= args.snap_warmup:
                    for i, raw in enumerate(raws):
                        if raw is not None:
                            snap_latest[i] = raw
                if len(snap_latest) == args.n_cameras:
                    tag = f"_{args.snap_tag}" if args.snap_tag else ""
                    for i in sorted(snap_latest):
                        label = CAM_LABEL.get(i, f"cam{args.base_port + i}")
                        p = os.path.join(args.out_dir,
                                         f"{session}{tag}_{label}.png")
                        cv2.imwrite(p, snap_latest[i])
                        h, w = snap_latest[i].shape[:2]
                        print(f"[snap] {p}  ({w}x{h})")
                    # side-by-side needs one common channel count (a gray
                    # stream beside a colour one cannot hstack)
                    pair = _hstack_pair(
                        [snap_latest[i] if snap_latest[i].ndim == 3 else
                         cv2.cvtColor(snap_latest[i], cv2.COLOR_GRAY2BGR)
                         for i in sorted(snap_latest)])
                    p = os.path.join(args.out_dir,
                                     f"{session}{tag}_combined.png")
                    cv2.imwrite(p, pair)
                    print(f"[snap] {p}  ({pair.shape[1]}x{pair.shape[0]})")
                    break
                continue      # nothing else to do in snapshot mode

            # ---- record per camera (full resolution, lazy fixed-size writer) ----
            if record:
                for i, img in enumerate(shown):
                    if img is None:
                        continue
                    # offline takes are re-detected, not watched: no overlay, and
                    # keep the source's own colour mode (gray stays gray).
                    out = raws[i] if args.offline_format else img
                    label = CAM_LABEL.get(i, f"cam{args.base_port + i}")
                    if i not in writers:                 # size known only now
                        h, w = out.shape[:2]
                        paths[i] = os.path.join(args.out_dir, f"{session}_{label}.mp4")
                        writers[i] = cv2.VideoWriter(paths[i], fourcc, args.fps, (w, h),
                                                     isColor=out.ndim == 3)
                        rec_sizes[i] = (w, h)
                        rec_span[i][0] = now
                        print(f"[rec] {label}: {w}x{h} "
                              f"{'color' if out.ndim == 3 else 'gray'} -> {paths[i]}")
                    writers[i].write(out)
                    rec_counts[i] += 1
                    rec_span[i][1] = now

            # ---- side-by-side pair: needed for combined display and/or recording ----
            all_fresh = all(s is not None for s in shown) and len(shown) == args.n_cameras
            pair_full = _hstack_pair(shown) if all_fresh else None

            if record and args.record_combined and pair_full is not None:
                if combined_writer is None:              # size known only now
                    combined_path = os.path.join(args.out_dir, f"{session}_combined.mp4")
                    combined_writer = cv2.VideoWriter(
                        combined_path, fourcc, args.fps,
                        (pair_full.shape[1], pair_full.shape[0]))
                    print(f"[rec] combined: {pair_full.shape[1]}x{pair_full.shape[0]} "
                          f"-> {combined_path}")
                combined_writer.write(pair_full)
                combined_count += 1

            # combined side-by-side used by both the window and the web stream:
            # full pair when every camera is fresh, else whatever arrived this tick.
            combo = pair_full if pair_full is not None else \
                (_hstack_pair([s for s in shown if s is not None])
                 if any(s is not None for s in shown) else None)

            # ---- web stream (encode once, push to all connected browsers) ----
            if web_state is not None and combo is not None:
                ok, enc = cv2.imencode(
                    ".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, args.web_quality])
                if ok:
                    web_state.update(enc.tobytes())

            # ---- display ----
            if show:
                if args.separate:
                    for i, img in enumerate(shown):
                        if img is None:
                            continue
                        label = CAM_LABEL.get(i, f"cam{args.base_port + i}")
                        cv2.imshow(label, _fit_width(img, args.max_width))
                elif combo is not None:
                    cv2.imshow(WINDOW_COMBINED, _fit_width(combo, args.max_width))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):                # q or ESC
                    print("\n[view] quit")
                    break

            if not args.quiet and now - last_report > 5.0:
                last_report = now
                got = ", ".join(f"{CAM_LABEL.get(i, i)}={fps[i]:.1f}fps"
                                for i in range(args.n_cameras))
                rec = f"  rec {sum(rec_counts.values())} frames" if record else ""
                print(f"[view] {now - t0:5.1f}s  {got}{rec}")
            if args.duration > 0 and now - t0 >= args.duration:
                print(f"[view] duration {args.duration:.0f}s reached")
                break
    except KeyboardInterrupt:
        print("\n[view] stopping")
    finally:
        cv2.destroyAllWindows()
        if web_state is not None:
            web_state.shutdown()          # wake blocked /stream handlers
        if httpd is not None:
            httpd.shutdown()
        for w in writers.values():
            w.release()
        if combined_writer is not None:
            combined_writer.release()
        client.cleanup()
        if record:
            dur = time.time() - t0
            print(f"[rec] {dur:.1f}s recorded")
            measured = {}
            for i, p in sorted(paths.items()):
                # frames span n-1 intervals between the first and last write —
                # excludes the connect latency ahead of the first frame.
                span = rec_span[i][1] - rec_span[i][0]
                measured[i] = (rec_counts[i] - 1) / span if rec_counts[i] > 1 \
                    and span > 0 else args.fps
                size = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0
                print(f"  {p}  ({rec_counts[i]} frames, {measured[i]:.1f} fps "
                      f"measured, {size:.1f} MB)")
            if combined_path and os.path.exists(combined_path):
                print(f"  {combined_path}  ({combined_count} frames, "
                      f"{os.path.getsize(combined_path) / 1e6:.1f} MB)")
            if args.offline_format:
                for i, p in sorted(paths.items()):
                    K = intrinsics.get(i)
                    if K is None:
                        print(f"[rec] WARNING no intrinsics for camera {i}: "
                              f"{p} has no .npz and cannot be replayed offline")
                        continue
                    meta_path = p[:-4] + ".npz"
                    _write_sidecar(meta_path, K, measured[i], rec_sizes[i],
                                   args.tag_size, args.tag_id_range, args.undistorted)
                    print(f"  {meta_path}  (K, fps={measured[i]:.1f}, "
                          f"{rec_sizes[i][0]}x{rec_sizes[i][1]}, "
                          f"tag_size={args.tag_size}, undistorted={args.undistorted})")
                if paths:
                    print(f"[rec] replay: "
                          f"OfflineCameraBackend('{paths[min(paths)][:-4]}')")
                    if any(m < args.fps * 0.7 for m in measured.values()):
                        print(f"[rec] NOTE the .mp4 header still declares --fps "
                              f"{args.fps:.0f}, so a plain player (VLC) plays these "
                              f"fast; OfflineCameraBackend paces from the sidecar's "
                              f"measured fps and is correct.")
            elif paths and any(rec_counts[i] / max(dur, 1e-6) < args.fps * 0.7
                               for i in paths):
                print(f"[rec] NOTE actual fps is well below the declared --fps "
                      f"{args.fps:.0f}: playback will look fast. Re-encode with "
                      f"the reported 'fps measured' if timing matters.")


if __name__ == "__main__":
    main()
