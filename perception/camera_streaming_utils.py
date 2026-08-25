"""camera_streaming_utils.py

Network-transparent camera streaming over pynng (NNG) PUB/SUB.

Server wraps any CameraBackend and publishes frames on a pynng Pub0 socket.
Client implements CameraBackend and receives frames transparently — drop-in
replacement for any local backend.

Latency design decisions:
- pynng Pub0/Sub0 over TCP: ~35% lower latency than ZMQ at 215KB JPEG frames on LAN
  (benchmark: p99 1.45ms pynng vs 2.31ms ZMQ). Trade-off: +~14% CPU vs ZMQ.
- Manual queue drain on Sub0: after each recv(), drain remaining messages non-blocking
  to keep only the newest frame. Equivalent to ZMQ CONFLATE=1 but implemented in Python.
  recv_buffer_size=1 / send_buffer_size=1: hard queue cap of 1 message — limits drain work.
- JPEG encoding for color (CLI default quality=80): ~40KB vs 900KB raw at 640x480.
  Gray-only streams are always sent raw (lossless), since AprilTag detection is sensitive to
  JPEG artifacts. When streaming color, gray is derived on the client via cvtColor.
- Static header: JSON serialised once at server start (K, dims, mode never change mid-session)

Thread model:
  Server: a dedicated _publish_loop thread owns the Pub0 socket.
  Client: a dedicated _recv_thread owns the Sub0 socket end-to-end (creates, reads, closes).
          pynng sockets are NOT thread-safe; never transferring them across threads avoids crashes.
          Main thread accesses only numpy buffers and threading.Event, which are safe.

Wire format (single message):
  Bytes 0-1    : big-endian uint16 — length N of JSON header
  Bytes 2..N+1 : JSON header  {"w": int, "h": int, "has_color": bool, "has_ir": bool,
                               "jpeg": int, "K": [[...]] }
                   jpeg=0  → raw bytes follow
                   jpeg>0  → JPEG-encoded bytes follow (quality = jpeg value)
  Bytes N+2..  : has_ir=False, has_color=False → raw gray uint8, shape (h, w)
                 has_ir=False, has_color=True  → color frame (raw BGR or JPEG); client
                                                 derives gray via cvtColor
                 has_ir=True → big-endian uint32 colour length, then the colour plane,
                               then the IR plane. Used when a RealSense streams colour
                               and infrared together: gray is then a true IR frame, so
                               it gets its own plane instead of being derived from colour.

Serial numbers below are placeholders: list your own with `rs-enumerate-devices
-s`, or omit --device / --devices entirely to take whatever is attached. Port
order IS stream order -- the first device becomes --base-port, the second
--base-port+1, and downstream code identifies cameras by that port, so keep the
order stable across restarts. For multi-server, `vs_site.CAMERA_SERIALS`
(VS_CAMERA_SERIALS / vs_site_local.py beside this file) is the --devices
default when it is non-empty, so a rig can pin its order once instead of on
every command line; a bare `--devices` (no values) returns to auto-discovery.

# one camera
python camera_streaming_utils.py server --backend realsense --device <serial> --port 5555
python camera_streaming_utils.py client --host <robot-host> --port 5555

# left + right (the two-camera rig the humanoid stack expects: two RealSense
# side by side on the head gimbal, left -> 5555, right -> 5556)
python camera_streaming_utils.py multi-server --backend realsense \
    --devices <left-serial> <right-serial> --base-port 5555 --quality 80
python camera_streaming_utils.py multi-client --host <robot-host> --n-cameras 2 --base-port 5555

# adding a wrist camera
python camera_streaming_utils.py multi-server --backend realsense \
    --devices <left-serial> <right-serial> <wrist-serial> --base-port 5555 --quality 80

"""

import json
import math
import os
import platform
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, Annotated

import cv2
import numpy as np
import pynng
import simplejpeg
import tyro

from camera_utils import CameraBackend, FPS

# Site settings live beside this file (vs_site.py: VS_<NAME> env, then
# vs_site_local.py, then a portable default). Importable whenever camera_utils
# was; the retry covers an import with this directory not yet on sys.path.
try:
    import vs_site as _site
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vs_site as _site

DEFAULT_PORT = 5555

STAMP_SIZE = 16
"""Two little-endian float64 monotonic timestamps between the header and the payload:
capture-return, then send. Splitting them separates server-side processing from network
transit, and lets a viewer report latency from the CAMERA rather than from send() —
which is the number that matters and the one that hides driver-side buffering.

Both ends read time.monotonic(), which on Linux is clock_gettime(CLOCK_MONOTONIC) —
system-wide and non-adjustable — so a subscriber on the SAME HOST can subtract these
directly. Across machines the two monotonic clocks share no epoch and the difference is
meaningless unless you sync them; the derived figures are reported as NaN in that case
rather than as a confident wrong number.
"""

VISER_FLUSH = os.environ.get('VS_VISER_FLUSH', '1') != '0'
"""Whether to flush Viser's outgoing buffer after pushing a frame.

Viser windows outgoing messages at 1/60 s and, after emitting a window, waits on its
flush event — which pushing a frame does not set. Without an explicit flush a frame can
therefore sit up to 16.7 ms before it leaves the process. Set VS_VISER_FLUSH=0 to
restore the windowed behaviour (useful for A/B measurement, or if a future Viser makes
flushing counterproductive).
"""

PREFER_CV2_ENCODE = platform.machine() in ('x86_64', 'AMD64')
"""Which library encodes JPEG faster — measured, not assumed, and it differs by arch.

On this x86_64 box (OpenCV 5.0, libjpeg-turbo behind both) cv2.imencode strictly
dominates simplejpeg for *encoding*: 0.32 vs 0.77 ms at 640x480 and 1.10 vs 1.70 ms at
1280x720 including the tobytes() copy pynng needs, while producing 28% smaller frames at
marginally higher PSNR (58.4 vs 58.2 dB at q80). On aarch64 (Jetson) simplejpeg's NEON
path was the original reason it was chosen; that has not been re-measured here, so ARM
keeps it.

Decoding is the other way round on both arches and is NOT switched: simplejpeg decodes
in 0.41 vs 1.85 ms at 720p — 3x faster than cv2.imdecode — so the client keeps it.
"""


# ============================================================================
# LATENCY PROFILING
# ============================================================================

@dataclass
class LatencyStats:
    """Snapshot of per-frame latency metrics (all in milliseconds)."""
    decode_ms: float = 0.0
    """Wall time spent in JPEG decode / raw copy inside _decode_into_buffers."""
    wire_ms: float = 0.0
    """Gap between consecutive frame arrivals — i.e. the camera's frame PERIOD
    (33 ms at 30 fps), not network transit. See transit_ms for the real thing."""
    endtoend_ms: float = 0.0
    """Elapsed time from recv stamp to consumer's capture_frame() returning."""
    transit_ms: float = float('nan')
    """True server-send -> client-recv, from the on-wire timestamp. NaN when the
    server does not stamp, or when the two clocks share no epoch (different hosts)."""
    encode_ms: float = float('nan')
    """Server-side capture-return -> send: JPEG encode plus message assembly."""


class LatencyTracker:
    """Tracks running statistics for one camera stream."""

    def __init__(self, name: str = ""):
        self.name = name
        self._decode_ms = RunningStats()
        self._wire_ms = RunningStats()
        self._e2e_ms = RunningStats()
        self._transit_ms = RunningStats()
        self._encode_ms = RunningStats()
        self._display_ms = RunningStats()

    def update(self, stats: LatencyStats):
        if stats.decode_ms > 0:
            self._decode_ms.add(stats.decode_ms)
        if stats.wire_ms > 0:
            self._wire_ms.add(stats.wire_ms)
        if stats.endtoend_ms > 0:
            self._e2e_ms.add(stats.endtoend_ms)
        if stats.transit_ms == stats.transit_ms:  # not NaN
            self._transit_ms.add(stats.transit_ms)
        if stats.encode_ms == stats.encode_ms:
            self._encode_ms.add(stats.encode_ms)

    def update_display(self, ms: float):
        """Record camera-capture -> pixels-handed-to-the-renderer for one drawn frame.

        This is the number to optimize. e2e only measures how long a frame sat in the
        client's buffer, which with free-running cameras averages half a frame period
        no matter how fast the viewer renders.
        """
        if ms > 0:
            self._display_ms.add(ms)

    def summary(self) -> str:
        def s(r: RunningStats) -> str:
            return f"avg={r.avg:.2f} p50={r.p50:.2f} p99={r.p99:.2f} max={r.max:.2f}"
        # 'period', not 'wire': this is the gap between consecutive frame arrivals, so
        # it reports the camera's frame rate (33 ms at 30 fps), NOT network transit.
        # Nothing here measures transit — that needs a send-side timestamp on the wire.
        return (
            f"{self.name}: "
            f"display={s(self._display_ms)} ms | "
            f"encode={s(self._encode_ms)} ms | "
            f"transit={s(self._transit_ms)} ms | "
            f"decode={s(self._decode_ms)} ms | "
            f"period={s(self._wire_ms)} ms"
        )


class RunningStats:
    """Sliding-window min/mean/max/p50/p99 over the last `capacity` latency samples.

    Every statistic covers the same window. An earlier version accumulated avg over all
    time while the percentiles came from the ring, so a camera that degraded showed a
    moved p99 and an almost unchanged mean — which reads as "just a blip" when it isn't.
    At 30 fps the default 300 samples is the last ~10 seconds.
    """

    def __init__(self, capacity: int = 300):
        self._samples = np.empty(capacity)
        self._n = 0

    def add(self, v: float):
        self._samples[self._n % len(self._samples)] = v
        self._n += 1

    @property
    def _window(self) -> np.ndarray:
        return self._samples[:min(self._n, len(self._samples))]

    @property
    def avg(self) -> float:
        return float(self._window.mean()) if self._n else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(self._window, 50)) if self._n else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self._window, 99)) if self._n else 0.0

    @property
    def max(self) -> float:
        return float(self._window.max()) if self._n else 0.0


# ============================================================================
# SERVER
# ============================================================================

class CameraStreamingServer:
    """Wraps any CameraBackend and publishes frames over a pynng Pub0 socket.

    Args:
        backend:      Any CameraBackend whose initialize() has already been called.
        port:         TCP port to bind on (default 5555).
        jpeg_quality: JPEG quality for color frames (0=raw lossless, 1-100=JPEG quality).
                      Default 0 (raw). Use 95 for near-lossless over bandwidth-limited links.
        debug:        Print publish FPS to stdout once per second.

    Usage (context manager or manual start/stop):
        with CameraStreamingServer(backend, port=5555, jpeg_quality=95):
            while True: time.sleep(1)   # Ctrl-C to stop
    """

    DEAD_AFTER_S = 3.0
    """Seconds of consecutive capture failures before the publisher gives up its port."""

    def __init__(self, backend: CameraBackend, port: int = DEFAULT_PORT,
                 jpeg_quality: int = 0, debug: bool = False):
        self._backend = backend
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._debug = debug
        self._running = False
        self._thread = None

    def start(self):
        """Start background publish loop (non-blocking).

        capture_frame() is called directly inside _publish_loop — one thread per camera
        instead of two (no separate capture_frame_continuously thread).
        """
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        """True while the publish thread is still running.

        Goes False when the camera stops delivering — the loop exits, closes its
        socket and frees the port, which is what lets a supervisor notice and retry.
        """
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        """Stop capture and signal publish thread to exit."""
        self._running = False
        self._backend.cleanup()
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def _publish_loop(self):
        """Owns the Pub0 socket for its entire lifetime. Runs in background thread.

        Calls capture_frame() directly (blocking) — no separate capture thread.
        One thread per camera instead of two, halving thread count on the Jetson.
        """
        # --- JPEG encoder selection ---
        # Which library is faster is architecture-dependent, so it is chosen by
        # platform rather than assumed. Encoder lambdas are finalized after has_color
        # is known, below.
        if self._jpeg_quality > 0:
            encode_jpeg = True   # finalized after has_color is known
            encoder_name = "cv2" if PREFER_CV2_ENCODE else "turbojpeg"
        else:
            encode_jpeg = None
            encoder_name = "raw"

        # Bind only after the backend is known good. Binding first leaves a
        # listening-but-silent port behind whenever init fails, and a subscriber
        # cannot tell that apart from a healthy idle camera — it just blocks on a
        # frame that never comes. Observed live: a RealSense that could not resolve
        # its stream config still held its port open, and the viewer waited on it.
        if not self._backend.init_complete.wait(timeout=10):
            print(f"CameraStreamingServer: backend for port {self._port} never "
                  f"initialized within 10s — not binding.")
            return
        if self._backend.get_gray_buffer() is None:
            print(f"CameraStreamingServer: backend for port {self._port} initialized "
                  f"but produced no buffers — not binding.")
            return
        # A successful pipeline.start() is not proof of a working camera. A RealSense on
        # a USB 2.1 link starts cleanly, reports its framerate, and then never delivers a
        # single frame — so require an actual frame before claiming the port. Otherwise
        # the port listens forever, silent, and no subscriber can tell it apart from a
        # camera that is merely idle.
        first_frame_deadline = time.perf_counter() + 5.0
        while time.perf_counter() < first_frame_deadline:
            if self._backend.capture_frame():
                break
        else:
            print(f"CameraStreamingServer: camera on port {self._port} started but "
                  f"produced no frame in 5s — not binding.")
            return

        # send_buffer_size=1: drop frames rather than queue them when subscriber is slow.
        # Pub0 send() is best-effort broadcast — if a subscriber's buffer is full, that
        # subscriber drops the message; the send() call itself does not block or raise.
        socket = pynng.Pub0(send_buffer_size=1, listen=f"tcp://*:{self._port}")

        # Pass-through: the backend already holds a compressed frame (a webcam's own
        # MJPEG), so there is nothing to encode. Measured on 3x720p webcams, skipping
        # the decode-then-re-encode round trip drops the publisher from 22.0% to 0.4%
        # of a core and avoids a generation of JPEG loss, at ~50% more bytes on the
        # wire because the camera compresses less aggressively than quality 80.
        # jpeg_quality=0 means the caller explicitly wants raw pixels, which a
        # pass-through stream cannot provide.
        passthrough = (self._jpeg_quality > 0
                       and getattr(self._backend, 'passthrough', False)
                       and self._backend.get_encoded_frame() is not None)

        K = self._backend.get_intrinsics()
        gray = self._backend.get_gray_buffer()
        color = self._backend.get_frame_buffer()
        has_color = True if passthrough else (color is not gray)
        # When the backend runs colour AND infrared together, the gray buffer holds a
        # genuine IR frame rather than a colour-derived luma, so it is worth its own
        # plane on the wire. Without this the IR frame never leaves the server: the
        # has_color path sends colour only and the client recreates gray via cvtColor.
        has_ir = has_color and bool(getattr(self._backend, 'ir_stream_flag', False))
        h, w = gray.shape

        # Finalize JPEG encoder now that has_color is known.
        # simplejpeg always requires ndim=3. Color buffer is (H,W,3); IR/gray buffer is (H,W).
        # For grayscale: np.expand_dims creates a zero-copy (H,W,1) view before encoding.
        encode_gray = None
        if encode_jpeg is True:
            if PREFER_CV2_ENCODE:
                encode_gray = lambda frame, q: cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
                encode_color = encode_gray  # cv2 dispatches on channel count
            else:
                encode_gray = lambda frame, q: simplejpeg.encode_jpeg(
                    np.expand_dims(frame, 2), quality=q, colorspace='GRAY')
                encode_color = lambda frame, q: simplejpeg.encode_jpeg(
                    frame, quality=q, colorspace='BGR')
            encode_jpeg = encode_color if has_color else encode_gray

        print(f"CameraStreamingServer: publishing on tcp://*:{self._port} "
              f"(encoder={'passthrough' if passthrough else encoder_name})")

        # Pre-build static header (K, dims, color mode, encoding are fixed for a session)
        # "stamped" tells the subscriber that a fixed 8-byte send timestamp sits between
        # this header and the payload. It stays out of the JSON so the header can still
        # be serialized once per session rather than per frame.
        header_bytes = json.dumps({
            "w": w, "h": h, "has_color": has_color, "has_ir": has_ir,
            "jpeg": self._jpeg_quality, "K": K.tolist(), "stamped": True
        }).encode()
        header_prefix = struct.pack('>H', len(header_bytes))  # 2-byte big-endian length
        static_prefix = header_prefix + header_bytes  # immutable per session — concat once
        hp = len(static_prefix)

        # Raw path only: pre-allocate a send buffer so np.copyto writes the frame
        # directly in, and a single bytes() call produces the message.
        # Saves 1 allocation vs the naive tobytes() + concat approach.
        # JPEG path: pynng's cffi binds nng_send as void* directly (not ffi.from_buffer),
        # so only bytes/bytearray are accepted — memoryview is rejected at runtime.
        # For JPEG the libjpeg alloc is unavoidable, so static_prefix + frame_bytes
        # (one concat) remains the simplest equivalent approach.
        if encode_jpeg is None and not has_ir:
            _raw_src = color if has_color else gray
            _raw_shape = (h, w, 3) if has_color else (h, w)
            _raw_flen = h * w * 3 if has_color else h * w
            _send_buf = bytearray(hp + STAMP_SIZE + _raw_flen)
            _send_buf[:hp] = static_prefix
            _frame_send_view = np.frombuffer(
                _send_buf, dtype=np.uint8, offset=hp + STAMP_SIZE, count=_raw_flen
            ).reshape(_raw_shape)

        fps = None
        last_print = time.perf_counter()
        warmup_done = False

        last_good = time.perf_counter()
        captured_at = 0.0
        while self._running:
            # Blocking call — runs at camera framerate with no extra thread or Event overhead
            if not self._backend.capture_frame():
                # An unplugged camera fails here forever. Give up so the socket closes
                # and the port is freed: a supervisor can then rebind it when the
                # camera comes back, and meanwhile subscribers see the stream stop
                # rather than a port that listens but never speaks.
                if time.perf_counter() - last_good > self.DEAD_AFTER_S:
                    print(f"CameraStreamingServer: no frames on port {self._port} for "
                          f"{self.DEAD_AFTER_S:.0f}s — releasing the port.")
                    break
                continue
            # Stamped the instant the driver hands the frame over, so everything after
            # this point — encode, transit, render — is attributable.
            captured_at = time.monotonic()
            last_good = time.perf_counter()

            if not warmup_done:
                # Reset timer after first frame so init latency doesn't pollute FPS average
                fps = FPS() if self._debug else None
                last_print = time.perf_counter()
                warmup_done = True

            # Compress FIRST, stamp second. Stamping before the encode would fold the
            # encode into whatever the subscriber computes as transit — which is how
            # RealSense appeared to have 1.55 ms of "network" on loopback while the
            # pass-through webcams had 0.25 ms. The difference was the JPEG.
            payload_prefix = b''
            if passthrough:
                # Already compressed by the camera — straight onto the wire.
                frame_bytes = self._backend.get_encoded_frame()
            elif has_ir:
                # Two planes: colour then IR, with a fixed 4-byte colour length between
                # the static header and the payload. Keeping the length out of the JSON
                # preserves the build-header-once optimization while still allowing the
                # per-frame-variable JPEG sizes.
                if encode_jpeg is not None:
                    color_payload = encode_jpeg(color, self._jpeg_quality)
                    ir_payload = encode_gray(gray, self._jpeg_quality)
                else:
                    color_payload = color.tobytes()
                    ir_payload = gray.tobytes()
                payload_prefix = struct.pack('>I', len(color_payload))
                frame_bytes = color_payload + ir_payload
            elif encode_jpeg is not None:
                # JPEG: concat is unavoidable (pynng requires bytes/bytearray, not memoryview)
                frame_bytes = encode_jpeg(color, self._jpeg_quality)
            else:
                frame_bytes = None   # raw path writes straight into the send buffer

            if frame_bytes is None:
                # Raw: np.copyto into pre-alloc buffer, then one bytes() call.
                # Saves 1 alloc vs the old tobytes() + concat (2 allocs).
                np.copyto(_frame_send_view, _raw_src)
                _send_buf[hp:hp + STAMP_SIZE] = struct.pack(
                    '<dd', captured_at, time.monotonic())
                flen = _raw_flen
                msg = bytes(_send_buf)
            else:
                flen = len(frame_bytes)
                stamp = struct.pack('<dd', captured_at, time.monotonic())
                msg = static_prefix + stamp + payload_prefix + frame_bytes

            try:
                socket.send(msg, block=False)
                if fps is not None:
                    fps.tick()
                    now = time.perf_counter()
                    if now - last_print >= 1.0:
                        print(f"[server] publish {fps.fps:.1f} fps  frame {flen/1024:.0f} KB")
                        last_print = now
            except pynng.TryAgain:
                pass  # no subscriber connected or send buffer full — drop frame silently

        socket.close()


class MultiCameraStreamingServer:
    """Manages N CameraStreamingServer instances — one per camera, consecutive ports.

    Camera i is published on port base_port + i. Each server runs its own
    _publish_loop thread; this class is a pure coordinator with no sockets or threads
    of its own.

    Args:
        backends:     List of CameraBackend instances. initialize() must be called
                      on each before start().
        base_port:    First port. Camera i binds base_port + i.
        jpeg_quality: Passed unchanged to each CameraStreamingServer.
        debug:        Print per-server publish FPS to stdout.

    Design: multi-port chosen over single-socket multiplexing to reuse all existing
    code unchanged and give each camera independent backpressure / failure isolation.
    For 2-6 cameras on a LAN, N TCP connections is negligible.
    """

    def __init__(self, backends: list, base_port: int = DEFAULT_PORT,
                 jpeg_quality: int = 0, debug: bool = False):
        self._servers = [
            CameraStreamingServer(b, port=base_port + i,
                                  jpeg_quality=jpeg_quality, debug=debug)
            for i, b in enumerate(backends)
        ]

    def start(self):
        for s in self._servers:
            s.start()

    def stop(self):
        for s in self._servers:
            s.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def enumerate_realsense(_ctx=[]) -> dict:
    """Currently attached RealSense cameras, as {serial: serial}.

    The context is created once and reused: librealsense keeps its device-watcher
    thread on the context, and building a fresh one every poll is both slow and prone
    to disturbing pipelines that are already streaming.
    """
    import pyrealsense2 as rs
    if not _ctx:
        _ctx.append(rs.context())
    return {d.get_info(rs.camera_info.serial_number):
            d.get_info(rs.camera_info.serial_number)
            for d in _ctx[0].query_devices()}


def enumerate_usb_cameras() -> dict:
    """Currently attached non-RealSense V4L2 capture cameras, as {identity: /dev/videoN}.

    Keyed by USB serial (falling back to the bus path) rather than by /dev/videoN,
    because the kernel reassigns node numbers on every replug — an OBSBOT observed
    moving from video36 to video10 across one unplug. A stable key is what lets a
    camera return to the same TCP port.

    Only index=0 nodes are real capture devices; index=1 is the metadata node that
    shares the same USB device and cannot deliver frames.
    """
    import os
    found = []  # (serial or None, bus path, /dev node)
    for node in sorted(os.listdir('/sys/class/video4linux')):
        base = f'/sys/class/video4linux/{node}'
        try:
            with open(f'{base}/name') as f:
                name = f.read().strip()
            with open(f'{base}/index') as f:
                index = f.read().strip()
        except OSError:
            continue
        if index != '0' or not name or 'RealSense' in name or 'Depth' in name:
            continue

        # Walk up from the V4L2 node to the USB device directory (the one holding
        # idVendor), which is where a serial number lives if the camera has one.
        path = os.path.realpath(f'{base}/device')
        serial, bus = None, node
        for _ in range(6):
            if os.path.exists(f'{path}/idVendor'):
                bus = os.path.basename(path)  # e.g. '3-1.2'
                try:
                    with open(f'{path}/serial') as f:
                        serial = f.read().strip() or None
                except OSError:
                    serial = None
                break
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        found.append((serial, bus, f'/dev/{node}'))

    # Prefer the serial, because it survives being moved to a different socket. But
    # cheap webcams ship hard-coded serials — two of the same model would both claim
    # 'SN0001' and fight over one port — so a serial that is not unique here is
    # discarded in favour of the bus path, which always is.
    counts = {}
    for serial, _, _ in found:
        if serial:
            counts[serial] = counts.get(serial, 0) + 1
    return {(serial if serial and counts[serial] == 1 else bus): dev
            for serial, bus, dev in found}


class HotplugCameraStreamingServer:
    """Keeps a publisher running for every camera currently attached.

    Polls the device list, starts a CameraStreamingServer for each camera that appears
    and tears one down for each that vanishes, so cameras can be plugged and unplugged
    while the process runs.

    Ports are assigned to a camera's *identity* on first sight and held for the life of
    the process, so a camera that is unplugged and plugged back in returns to the port
    its subscribers already know. New cameras take the lowest free slot in
    [base_port, base_port + max_cameras).

    Args:
        enumerate_devices: Callable returning {identity: device_arg} for what is attached.
        make_backend:      Callable device_arg -> CameraBackend (not yet initialized).
        base_port:         First port of the reserved range.
        max_cameras:       Size of the reserved port range.
        width/height/fps:  Passed to each backend's initialize().
        jpeg_quality:      Passed to each CameraStreamingServer.
        debug:             Print per-server publish FPS.
        poll_interval:     Seconds between device scans.
    """

    def __init__(self, enumerate_devices, make_backend, base_port: int = DEFAULT_PORT,
                 max_cameras: int = 16, width: int = 640, height: int = 480,
                 fps: int = 30, jpeg_quality: int = 0, debug: bool = False,
                 poll_interval: float = 1.0):
        self._enumerate = enumerate_devices
        self._make_backend = make_backend
        self._base_port = base_port
        self._max_cameras = max_cameras
        self._dims = (width, height, fps)
        self._jpeg_quality = jpeg_quality
        self._debug = debug
        self._poll_interval = poll_interval

        self._ports = {}    # identity -> port, sticky for the life of the process
        self._servers = {}  # identity -> CameraStreamingServer, only while running
        self._running = False
        self._thread = None

    def _port_for(self, identity):
        """Return this camera's sticky port, assigning the lowest free one on first sight."""
        if identity in self._ports:
            return self._ports[identity]
        taken = set(self._ports.values())
        for port in range(self._base_port, self._base_port + self._max_cameras):
            if port not in taken:
                self._ports[identity] = port
                return port
        print(f"Hotplug: no free port for {identity} "
              f"(range {self._base_port}-{self._base_port + self._max_cameras - 1} full)")
        return None

    def _start_camera(self, identity, device):
        port = self._port_for(identity)
        if port is None:
            return
        try:
            backend = self._make_backend(device)
            backend.initialize(*self._dims)
        except Exception as e:
            print(f"Hotplug: failed to open {identity} ({device}): {e}")
            return
        server = CameraStreamingServer(backend, port=port,
                                       jpeg_quality=self._jpeg_quality, debug=self._debug)
        server.start()
        self._servers[identity] = server
        print(f"Hotplug: + {identity} ({device}) -> port {port}")

    def _stop_camera(self, identity, reason):
        server = self._servers.pop(identity, None)
        if server is None:
            return
        server.stop()
        print(f"Hotplug: - {identity} ({reason}) — port {self._ports[identity]} released "
              f"and reserved for its return")

    def _monitor_loop(self):
        while self._running:
            try:
                present = self._enumerate()
            except Exception as e:
                print(f"Hotplug: device scan failed: {e}")
                present = {}

            for identity in [k for k in self._servers if k not in present]:
                self._stop_camera(identity, "unplugged")
            # A publisher whose thread died (camera stopped delivering while still
            # enumerated) is cleared here so the next pass reopens it.
            for identity in [k for k, s in self._servers.items() if not s.alive]:
                self._stop_camera(identity, "stopped delivering")

            for identity, device in present.items():
                if identity not in self._servers:
                    self._start_camera(identity, device)

            time.sleep(self._poll_interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        for identity in list(self._servers):
            self._stop_camera(identity, "shutdown")
        if self._thread:
            self._thread.join(timeout=3)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ============================================================================
# CLIENT
# ============================================================================

class StreamingCameraBackend(CameraBackend):
    """CameraBackend that receives frames from a remote CameraStreamingServer.

    Drop-in replacement for any local CameraBackend. Actual resolution and
    framerate are determined by the server; values passed to initialize() are ignored.

    Args:
        host: Server IP address or hostname.
        port: Server TCP port (default 5555).

    Assumptions:
        - Server is reachable within 5s of initialize() being called.
        - Camera parameters (K, resolution, color mode) are fixed for a session.
        - Single consumer thread for wait_for_frame() / capture_frame().

    Thread model:
        A dedicated _recv_thread creates, uses, and closes the Sub0 socket entirely
        within itself (pynng thread-safety requirement). All other threads interact
        only through numpy buffers and threading.Event primitives.

        capture_frame() is therefore a blocking wait on the frame_ready event
        rather than a direct socket read. capture_frame_continuously() is a no-op
        because the recv thread runs automatically from initialize().
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, decode: bool = True):
        """
        Args:
            host/port: Publisher to subscribe to.
            decode: Decode each frame to pixels. Set False for a consumer that only
                forwards the compressed frame onward (the direct viewer hands it to a
                browser as-is), which removes the entire decode cost. In that mode
                get_frame_buffer()/get_gray_buffer() stay None and only
                get_encoded_color()/get_encoded_ir() are valid.
        """
        super().__init__(device=f"{host}:{port}")
        self._host = host
        self._port = port
        self._decode = decode
        self._got_frame = False
        self._color_jpeg = None
        self._ir_jpeg = None
        self._K = None
        self._has_color = False
        self._has_ir = False  # True when the server sends a separate IR plane
        self._jpeg_quality = 0  # set from first frame header
        self._header_len = 0    # cached offset to pixel data; set from first frame
        self._header_raw = None  # last adopted header bytes; a change triggers reconfigure
        self._h = self._w = 0
        self._stamped = False    # server sends a per-frame send timestamp
        self._payload_offset = 0
        self._transit_ms = float('nan')  # true send->recv, only when _stamped
        self._encode_ms = float('nan')   # server-side capture->send
        self._send_stamp = 0.0
        self._capture_stamp = 0.0
        self.last_frame_recv_stamp = 0.0  # monotonic timestamp of latest decoded frame
        # Profiling: timestamps recorded by _recv_thread / _decode_into_buffers / capture_frame
        self._decode_start_stamp = 0.0
        self._decode_ms = 0.0
        self._wire_ms = 0.0
        self.last_frame_client_stamp = 0.0  # monotonic timestamp when consumer received the frame
        # Optional Event shared with sibling backends; MultiStreamingCameraClient
        # installs one so a consumer can wake on the first camera to deliver.
        self.group_ready = None

    def initialize(self, width: int, height: int, framerate: int) -> tuple:
        """Connect to server and start background recv thread.

        width/height/framerate are ignored — server determines actual resolution.
        Returns immediately (non-blocking). Wait on init_complete before use.
        """
        self._running = True
        threading.Thread(target=self._recv_thread, daemon=True).start()
        return (width, height)

    def _recv_thread(self):
        """Owns the Sub0 socket end-to-end. Creates it, receives all frames, closes it."""
        # recv_timeout=200ms: poll timeout — keeps the loop responsive to self._running=False.
        # recv_buffer_size=1: NNG's Sub0 queue drops the OLDEST message when full, not
        # the newest — verified by stalling a subscriber for 1s at buffer=10 and getting
        # back the 10 *newest* frames (0.2-300 ms old), never the 300-1000 ms ones. The
        # newest frame therefore survives at any depth, so depth 1 gives identical
        # freshness with nothing for the drain loop below to throw away.
        socket = pynng.Sub0(recv_timeout=200, recv_buffer_size=1)
        socket.subscribe("")   # empty string = receive all messages
        # block=False: non-blocking dial; pynng retries in the background so a slow-starting
        # server doesn't raise Timeout here and kill the thread before init_complete is set.
        socket.dial(f"tcp://{self._host}:{self._port}", block=False)

        # One loop for the whole session — there is no separate handshake phase.
        # A subscriber that gave up after a fixed deadline could never pick up a camera
        # that is plugged in later, and pynng's background dial reconnects on its own
        # when a publisher reappears. So we wait here indefinitely, and (re)configure
        # whenever the header changes: first frame, server restart, camera replug at a
        # different resolution.
        while self._running:
            try:
                data = socket.recv()
            except pynng.Closed:
                break  # socket closed (cleanup() called)
            except (pynng.Timeout, pynng.TryAgain):
                continue

            # Manual queue drain — keep only the newest frame (equivalent to ZMQ CONFLATE=1).
            # recv_buffer_size=1 bounds the number of iterations here to at most 1 extra recv.
            while True:
                try:
                    data = socket.recv(block=False)
                except (pynng.Closed, pynng.TryAgain, pynng.Timeout):
                    break

            # The header is byte-identical for every frame of a session, so comparing
            # raw bytes is a ~200-byte memcmp per frame and reconfigures exactly when
            # the stream really changed. Cheaper than parsing the JSON every frame and
            # more robust than parsing it only once.
            header_len = struct.unpack('>H', data[:2])[0]
            header_raw = bytes(data[:2 + header_len])
            if header_raw != self._header_raw:
                self._configure(header_raw, header_len)

            self._decode_start_stamp = time.monotonic()
            self._wire_ms = (self._decode_start_stamp - self.last_frame_recv_stamp) * 1000.0
            if self._stamped:
                capture_stamp, send_stamp = struct.unpack_from('<dd', data, 2 + header_len)
                # Same-host only: a cross-machine difference is not a duration. A
                # negative or absurd value means the clocks share no epoch.
                def _sane(v):
                    return v if -1.0 < v < 10_000.0 else float('nan')
                self._transit_ms = _sane((self._decode_start_stamp - send_stamp) * 1000.0)
                self._encode_ms = _sane((send_stamp - capture_stamp) * 1000.0)
                self._send_stamp = send_stamp
                self._capture_stamp = capture_stamp
            self._decode_into_buffers(data)
            self._signal_frame()

        socket.close()  # closed in the same thread that created it

    def _configure(self, header_raw: bytes, header_len: int):
        """Adopt a new stream header: record intrinsics and (re)allocate buffers.

        Called on the first frame and again any time the publisher's header changes —
        which is how a replugged camera coming back at a different resolution is
        absorbed without restarting the client.
        """
        header = json.loads(header_raw[2:])
        w, h = header['w'], header['h']
        self._has_color = header['has_color']
        self._has_ir = header.get('has_ir', False)  # absent on older servers
        self._jpeg_quality = header.get('jpeg', 0)
        self._K = np.array(header['K'], dtype=np.float64)
        self._header_len = header_len
        self._header_raw = header_raw
        self._stamped = header.get('stamped', False)  # absent on older servers
        self._payload_offset = 2 + header_len + (STAMP_SIZE if self._stamped else 0)

        # Ring of buffers: recv writes to the next slot, consumer reads the front
        # pointer. After each write the front pointer is reassigned atomically
        # (CPython attribute assignment is GIL-protected), so the consumer never
        # sees a partially-written buffer (numpy releases the GIL during copyto).
        #
        # Three slots, not two. With two, a consumer's pointer is only valid for
        # strictly less than 2 camera periods (66 ms at 30 fps), because the second
        # write after it grabbed the pointer lands back in the slot it holds. The
        # 15-tile viewer's main loop measures p99 = 72 ms, so two slots are already
        # too few — the symptom is intermittently wrong AprilTag poses rather than
        # visible tearing. A third slot costs 0.9 MB per camera at 640x480.
        self._write_idx = 0
        self._h, self._w = h, w
        self._got_frame = True
        if self._decode:
            NBUF = 3
            self._gray_bufs = [np.empty((h, w), dtype=np.uint8) for _ in range(NBUF)]
            if self._has_color:
                self._color_bufs = [np.empty((h, w, 3), dtype=np.uint8)
                                    for _ in range(NBUF)]
            # gray_buf_internal / frame_buf_internal are public references; they always
            # point to the last fully-written buffer (swapped in _decode_into_buffers)
            self.gray_buf_internal = self._gray_bufs[0]
            self.frame_buf_internal = (self._color_bufs[0] if self._has_color
                                       else self.gray_buf_internal)

        self._wire_ms = 0.0  # no comparable prior frame to measure an inter-arrival gap from

        enc = f"jpeg@{self._jpeg_quality}" if self._jpeg_quality > 0 else "raw"
        print(f"StreamingCameraBackend: connected {self._host}:{self._port} "
              f"→ {w}x{h}, color={self._has_color}, ir={self._has_ir}, encoding={enc}")
        self.init_complete.set()

    def _signal_frame(self):
        """Publish 'a new frame is in the front buffer' to this backend's waiters."""
        self.frame_ready.set()
        if self.group_ready is not None:
            self.group_ready.set()

    def _decode_into_buffers(self, data: bytes):
        """Write pixel data into the back buffer, then swap it to the front.

        Ring-buffer pattern: recv always writes into _gray_bufs[_write_idx] while the
        consumer reads gray_buf_internal (a different slot). After the write completes,
        gray_buf_internal is atomically reassigned to the freshly written buffer and
        _write_idx advances. CPython attribute assignment is GIL-protected so the swap
        is atomic from the consumer's perspective.
        """
        idx = self._write_idx
        offset = self._payload_offset  # past the header and the send timestamp
        # Cached at handshake rather than read off _gray_bufs[0], whose shape stops
        # being authoritative once decode_jpeg rebinds individual slots.
        h, w = self._h, self._w

        if not self._decode:
            # Forward-only: keep the compressed planes, touch no pixels. Slicing bytes
            # copies, which is what we want — `data` belongs to the socket.
            if self._has_ir:
                color_len = struct.unpack('>I', data[offset:offset + 4])[0]
                c0 = offset + 4
                self._color_jpeg = data[c0:c0 + color_len]
                self._ir_jpeg = data[c0 + color_len:]
            else:
                self._color_jpeg = data[offset:]
                self._ir_jpeg = None
            self.last_frame_recv_stamp = time.monotonic()
            self._decode_ms = 0.0
            return

        data_mv = memoryview(data)  # zero-copy view; slicing bytes creates a copy
        if self._has_ir:
            # [4B colour length][colour plane][IR plane] — gray is a real IR frame
            # here, not a luma derived from colour, so it is decoded, never computed.
            color_len = struct.unpack('>I', data[offset:offset + 4])[0]
            c0 = offset + 4
            c1 = c0 + color_len
            if self._jpeg_quality > 0:
                self._color_bufs[idx] = simplejpeg.decode_jpeg(data_mv[c0:c1], colorspace='BGR')
                self._gray_bufs[idx] = simplejpeg.decode_jpeg(data_mv[c1:], colorspace='GRAY').squeeze()
            else:
                np.copyto(self._color_bufs[idx],
                          np.frombuffer(data, dtype=np.uint8, offset=c0, count=h * w * 3).reshape(h, w, 3))
                np.copyto(self._gray_bufs[idx],
                          np.frombuffer(data, dtype=np.uint8, offset=c1, count=h * w).reshape(h, w))
            self.frame_buf_internal = self._color_bufs[idx]
        elif self._has_color:
            if self._jpeg_quality > 0:
                # data_mv[offset:] is O(1) — avoids a ~300KB copy vs data[offset:]
                # simplejpeg allocates a new array; assign directly to avoid np.copyto overhead
                self._color_bufs[idx] = simplejpeg.decode_jpeg(data_mv[offset:], colorspace='BGR')
            else:
                np.copyto(self._color_bufs[idx],
                          np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(h, w, 3))
            # Derive gray from color — avoids sending a redundant gray channel over the wire
            cv2.cvtColor(self._color_bufs[idx], cv2.COLOR_BGR2GRAY, dst=self._gray_bufs[idx])
            self.frame_buf_internal = self._color_bufs[idx]
        else:
            if self._jpeg_quality > 0:
                # simplejpeg GRAY decode returns (H,W,1); squeeze to (H,W) for the gray buffer
                self._gray_bufs[idx] = simplejpeg.decode_jpeg(data_mv[offset:], colorspace='GRAY').squeeze()
            else:
                np.copyto(self._gray_bufs[idx],
                          np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(h, w))
            self.frame_buf_internal = self._gray_bufs[idx]

        # Swap front pointer — consumer now sees the fully-written buffer
        self.gray_buf_internal = self._gray_bufs[idx]
        self._write_idx = (idx + 1) % len(self._gray_bufs)  # next slot in the ring
        decode_end = time.monotonic()
        self._decode_ms = (decode_end - self._decode_start_stamp) * 1000.0
        self.last_frame_recv_stamp = decode_end

    @property
    def port(self) -> int:
        """TCP port this backend subscribes to."""
        return self._port

    @property
    def connected(self) -> bool:
        """True once a first frame arrived.

        Stays True after a camera goes away — the last frame is still readable. Use
        `live` to ask whether frames are still arriving.
        """
        return self._got_frame

    def get_encoded_color(self):
        """Newest colour plane, still JPEG-compressed. Only in decode=False mode."""
        return self._color_jpeg

    def get_encoded_ir(self):
        """Newest IR plane, still JPEG-compressed, or None. Only in decode=False mode."""
        return self._ir_jpeg

    @property
    def width(self) -> int:
        """Frame width reported by the publisher (0 before the first frame)."""
        return self._w if self._header_raw else 0

    @property
    def height(self) -> int:
        """Frame height reported by the publisher (0 before the first frame)."""
        return self._h if self._header_raw else 0

    @property
    def last_send_stamp(self) -> float:
        """Server's send timestamp for the frame currently in the front buffer.

        0.0 if the server does not stamp. Same-host comparisons only — see STAMP_SIZE.
        """
        return self._send_stamp

    @property
    def last_capture_stamp(self) -> float:
        """When the driver handed the server the frame now in the front buffer.

        Subtracting this from 'now' at render time gives latency measured from the
        CAMERA, which is the figure that matters; measuring from send() hides both the
        server's encode and any driver-side frame queue.
        """
        return self._capture_stamp

    def live(self, stale_after: float = 1.5) -> bool:
        """True if a frame arrived within stale_after seconds.

        Distinguishes 'unplugged mid-session' (connected, not live — the tile holds a
        real but frozen image) from 'never showed up' (not connected).
        """
        return (self._got_frame
                and (time.monotonic() - self.last_frame_recv_stamp) < stale_after)

    @property
    def has_ir(self) -> bool:
        """True when get_gray_buffer() holds a real IR frame from a separate plane."""
        return self._has_ir

    def get_intrinsics(self) -> np.ndarray:
        self.init_complete.wait(timeout=4)
        return self._K

    def capture_frame(self, undistort: bool = True) -> bool:
        """Block until the next frame arrives from the server (up to 500ms).

        undistort is ignored — server-side concern.
        The background recv thread continuously receives frames; this method
        waits on the frame_ready event rather than reading the socket directly.
        """
        if not self.init_complete.is_set():
            return False
        ok = self.wait_for_frame(timeout=0.5)
        if ok:
            self.last_frame_client_stamp = time.monotonic()
        return ok

    def get_latency_stats(self) -> LatencyStats:
        """Return a snapshot of the most recent frame's latency metrics (ms)."""
        e2e = 0.0
        if self.last_frame_recv_stamp > 0 and self.last_frame_client_stamp > 0:
            e2e = (self.last_frame_client_stamp - self.last_frame_recv_stamp) * 1000.0
        return LatencyStats(
            decode_ms=self._decode_ms,
            wire_ms=self._wire_ms,
            endtoend_ms=e2e,
            transit_ms=self._transit_ms,
            encode_ms=self._encode_ms,
        )

    def capture_frame_continuously(self):
        """No-op: recv thread starts automatically in initialize()."""
        pass

    def get_gray_buffer(self) -> np.ndarray:
        return self.gray_buf_internal

    def get_frame_buffer(self) -> np.ndarray:
        return self.frame_buf_internal

    def cleanup(self):
        """Signal recv thread to exit (it will close the socket on its own)."""
        self._running = False


class MultiStreamingCameraClient:
    """Connects to N CameraStreamingServer instances (same host, consecutive ports).

    Creates one StreamingCameraBackend per camera (ports base_port … base_port+N-1).
    Acts as an indexable list of StreamingCameraBackend instances.

    Args:
        host:       Server IP or hostname.
        n_cameras:  Number of cameras.
        base_port:  First port (matches MultiCameraStreamingServer convention).

    Design invariant: capture_all_parallel() does NOT spawn threads. Each backend
    already owns a recv thread, so the round is a sequential wait against a shared
    deadline (require_all=True) or a wake on the shared _group_ready event
    (require_all=False). Only one consumer may call wait_for_frame() per backend
    per round — the event is consumed on read.
    """

    def __init__(self, host: str, n_cameras: int, base_port: int = DEFAULT_PORT,
                 decode: bool = True):
        self._backends = [
            StreamingCameraBackend(host, base_port + i, decode=decode)
            for i in range(n_cameras)
        ]
        # Set by whichever camera decodes a frame first, so capture_all_parallel()
        # can wake on *any* arrival instead of blocking on the slowest camera.
        self._group_ready = threading.Event()
        for b in self._backends:
            b.group_ready = self._group_ready

    def initialize_all(self, width: int = 0, height: int = 0, framerate: int = 0):
        """Start background recv threads for all cameras (non-blocking).

        width/height/framerate are ignored by StreamingCameraBackend (server
        determines actual resolution); accepted for API consistency.
        """
        for b in self._backends:
            b.initialize(width, height, framerate)

    def wait_all(self, timeout: float = 6.0) -> bool:
        """Wait for all cameras to complete initialization.

        Uses a shared deadline across sequential waits — equivalent to parallel
        waiting because each backend's _recv_thread fires init_complete within ~5s
        independently. Returns True if all backends set init_complete before deadline.
        """
        deadline = time.perf_counter() + timeout
        for b in self._backends:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not b.init_complete.wait(timeout=remaining):
                return False
        return True

    def capture_all_parallel(self, timeout: float = 0.5, require_all: bool = True) -> list:
        """Wait for the next frame on all cameras simultaneously.

        Since each StreamingCameraBackend has its own background recv thread,
        waiting on their threading.Events sequentially with a shared deadline
        provides the exact same parallel throughput without thread overhead.

        Args:
            timeout: Shared deadline in seconds for the whole round.
            require_all: True waits until every camera has produced a frame, so the
                round rate is the *slowest* camera's rate — one 20 fps webcam drags
                eight 30 fps RealSenses down with it. False returns as soon as any
                camera has a new frame; each backend still exposes its own latest
                buffer, so a viewer redraws at the fastest rate and slow cameras
                simply repeat. Use False for display, True for synchronised capture.

        Returns list[bool] of length N — which cameras produced a new frame.
        """
        if not require_all:
            self._group_ready.clear()
            results = [b.wait_for_frame(timeout=0) for b in self._backends]
            if not any(results) and self._group_ready.wait(timeout):
                results = [b.wait_for_frame(timeout=0) for b in self._backends]
            now = time.monotonic()
            for b, ok in zip(self._backends, results):
                if ok:
                    b.last_frame_client_stamp = now
            return results

        deadline = time.perf_counter() + timeout
        results = []
        for b in self._backends:
            remaining = max(0.0, deadline - time.perf_counter())
            ok = b.wait_for_frame(timeout=remaining)
            if ok:
                b.last_frame_client_stamp = time.monotonic()
            results.append(ok)
        return results

    def drop_disconnected(self) -> list:
        """Forget cameras that never completed the handshake; return their ports.

        Call after wait_all(). A viewer should keep showing the cameras that *are*
        alive rather than refusing to start because one is unplugged — with six
        RealSense on shared USB hubs, one dropping out is routine.
        """
        dead = [b for b in self._backends if not b.connected]
        for b in dead:
            b.cleanup()
        self._backends = [b for b in self._backends if b.connected]
        return [b.port for b in dead]

    def get_latency_stats(self, i: int) -> LatencyStats:
        """Return latency snapshot for camera i (0-indexed)."""
        return self._backends[i].get_latency_stats()

    def get_all_latency_stats(self) -> list:
        """Return latency snapshots for all cameras."""
        return [b.get_latency_stats() for b in self._backends]

    def cleanup(self):
        for b in self._backends:
            b.cleanup()

    def __len__(self):
        return len(self._backends)

    def __getitem__(self, i):
        return self._backends[i]

    def __iter__(self):
        return iter(self._backends)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()


# ============================================================================
# DISPLAY LAYOUT
# ============================================================================

def tile_frames(frames: list, labels: list, target_aspect: float = 16 / 9,
                max_height: int = 1080, dirty: list = None,
                _canvas_cache: dict = {}) -> np.ndarray:
    """Compose frames into the row/column split that best fills a target_aspect screen.

    A single row is badly wrong past ~3 cameras: 9 tiles side by side is a 12:1
    strip that shrinks to nothing on a 16:9 display. Instead every (rows, cols)
    with rows*cols >= N is scored by the scale at which it would fit the target
    viewport, and the best is chosen — 9 cameras land on 3x3, 8 on 2x4, 6 on 2x3.
    Ties go to the arrangement wasting fewer empty cells.

    Frames are letterboxed (not stretched) into a uniform cell, so a fleet mixing
    640x480 RealSense with 1280x720 USB keeps every camera's true aspect ratio.

    Args:
        frames: BGR frames, any mix of resolutions.
        labels: One caption per frame, drawn at the top-left of its cell.
        target_aspect: Width/height of the display to fill (16/9 default).
        max_height: Cap on composite height, to bound per-frame JPEG encode cost.
        dirty: Optional per-frame flags; falsy entries keep whatever the reused canvas
            already holds instead of being rescaled. Rendering at 60 Hz from 30 fps
            cameras means most tiles are unchanged on any given pass, and rescaling
            them again produces identical pixels for nothing. None redraws everything.

    Returns:
        Single BGR composite image. The buffer is reused between calls of the same
        layout, so copy it if you need to hold it past the next call.
    """
    n = len(frames)
    # Median aspect, so one odd camera can't dictate the whole layout.
    cell_aspect = sorted(f.shape[1] / f.shape[0] for f in frames)[n // 2]

    rows, cols = grid_shape(n, cell_aspect, target_aspect)

    cell_h = max(1, min(max_height // rows, 720))
    cell_w = max(1, round(cell_h * cell_aspect))

    # Reuse the canvas across calls: at 2400x1080 a fresh np.zeros costs ~0.33 ms
    # and ~2000 page faults every frame, and each tile fully overwrites its own
    # region, so only the (always-black) letterbox bars persist. Keyed by shape, so
    # a camera appearing or disappearing gets a clean canvas.
    H, W = rows * cell_h, cols * cell_w
    canvas = _canvas_cache.get((H, W))
    fresh_canvas = canvas is None
    if fresh_canvas:
        canvas = _canvas_cache[(H, W)] = np.zeros((H, W, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        # A brand-new canvas holds nothing, so every tile must be drawn regardless.
        if dirty is not None and not fresh_canvas and not dirty[i]:
            continue
        s = min(cell_w / f.shape[1], cell_h / f.shape[0])
        fw, fh = max(1, int(f.shape[1] * s)), max(1, int(f.shape[0] * s))
        y0, x0 = (i // cols) * cell_h, (i % cols) * cell_w
        # Centre within the cell; leftover space stays black (letterbox/pillarbox).
        oy, ox = (cell_h - fh) // 2, (cell_w - fw) // 2
        # INTER_AREA costs 8-9x INTER_LINEAR (0.573 vs 0.071 ms for 640x480->480x360;
        # 1.036 vs 0.115 ms for 1280x720->480x270) and only earns it below ~0.5 scale,
        # where INTER_LINEAR starts to alias. dst= writes straight into the canvas,
        # avoiding a temporary per tile.
        interp = cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR
        ty = y0 + oy
        ly = y0 + 22
        # Wipe the label strip BEFORE the image lands on it. A letterboxed tile does
        # not cover its own text row, so the anti-aliased glyphs of a changing fps
        # reading would otherwise pile up on each other; wiping afterwards instead
        # would punch a black bar across the picture.
        canvas[max(0, ly - 18):ly + 6, x0:x0 + cell_w] = 0
        cv2.resize(f, (fw, fh), dst=canvas[ty:ty + fh, x0 + ox:x0 + ox + fw],
                   interpolation=interp)
        cv2.putText(canvas, labels[i], (x0 + 8, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def grid_shape(n: int, cell_aspect: float, target_aspect: float = 16 / 9) -> tuple:
    """Rows/cols that best fill a target_aspect screen with n cells of cell_aspect.

    Shared by the composite tiler and the direct viewer so both lay cameras out the
    same way. See tile_frames for why a single row is wrong past ~3 cameras.
    """
    best = None
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        scale = min(target_aspect / (cols * cell_aspect), 1.0 / rows)
        key = (scale, -(rows * cols - n))
        if best is None or key > best[0]:
            best = (key, rows, cols)
    return best[1], best[2]


class DirectViserViewer:
    """Shows each camera on its own Viser image plane, forwarding JPEGs undecoded.

    The composite viewer decodes every stream, rescales it into one big canvas and
    re-encodes that canvas as JPEG — for a viewer, all of it is avoidable. Here the
    bytes that arrive on the wire are handed to the browser untouched, so the client
    does no image processing at all and each camera refreshes at its own rate instead
    of waiting for a shared render tick.

    The catch this solves: scene.add_image maps image row 0 to the BOTTOM of the plane,
    and the frames are not pre-flipped because that would mean decoding them. A
    negative Y scale flips the plane instead. viser 1.0.30's `scale` property getter
    reports 1.0 for a tuple, but the value does reach the renderer — verified by
    rendering the scene through a headless browser.
    """

    FOV = 0.6

    PUSH_BUDGET_HZ = 150.0
    """Total image updates per second to hand the browser, across all tiles.

    A browser decodes and composites every JPEG it is given. Past its capacity the
    frames do not appear sooner — they queue, and latency climbs while the extra frames
    are never displayed. Measured on this machine (9 cameras, GPU, minus the 21.4 ms
    screencast floor): 270 pushes/s -> 96 ms, 135/s -> 40 ms, 90/s -> 35 ms. Spending
    the budget rather than exceeding it is therefore strictly better for latency, and
    costs nothing in quality or resolution — only frames that would not have been shown.

    Well under the budget (few cameras) nothing is capped and every frame goes straight
    through. Raise it if your client machine is faster; 0 disables capping.
    """

    def __init__(self, server, host: str):
        self.server = server
        self.host = host
        self._handles = {}
        self._labels = {}
        self._label_text = {}   # last text pushed, so unchanged labels cost no message
        self._layout = None
        self._fit_distance = 4.0
        self._last_push = {}      # tile key -> monotonic time of its last image update
        self.push_budget_hz = self.PUSH_BUDGET_HZ
        # Dark grey stand-in for a port with no camera yet, so an empty slot reads as a
        # slot rather than a rendering failure. Encoded once.
        self._placeholder_jpeg = cv2.imencode(
            '.jpg', np.full((240, 320, 3), 40, np.uint8))[1].tobytes()

        # Frame the grid for every client, not just those connected when the layout was
        # built — a browser opened later would otherwise keep its default oblique view.
        @server.on_client_connect
        def _on_connect(client):
            self._frame(client)

    def _frame(self, client):
        """Point a client's camera straight at the grid from +Z.

        180 degrees about X, not Y: viser uses OpenCV camera axes (look=+Z, up=-Y), so
        both aim at the origin, but a Y-rotation also negates right and up, rendering
        the scene upside-down and mirrored.
        """
        client.camera.position = (0.0, 0.0, self._fit_distance)
        client.camera.wxyz = (0.0, 1.0, 0.0, 0.0)
        client.camera.fov = self.FOV

    def _rebuild_layout(self, tiles: list):
        """(Re)create one image plane per tile, arranged in a grid facing the camera."""
        for h in list(self._handles.values()) + list(self._labels.values()):
            h.remove()
        self._handles.clear()
        self._labels.clear()
        self._label_text.clear()

        n = len(tiles)
        aspects = sorted(t['aspect'] for t in tiles)
        rows, cols = grid_shape(n, aspects[n // 2])
        # Each cell is 1 unit tall with a small gap; the grid is centred on the origin.
        cell_h, gap = 1.0, 0.06
        cell_w = aspects[n // 2] * cell_h
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            x = (c - (cols - 1) / 2) * (cell_w + gap)
            # World +Y is UP on screen with this camera — verified by rendering four
            # planes at known coordinates through a headless browser. So row 0 takes
            # the largest y.
            y = ((rows - 1) / 2 - r) * (cell_h + gap)
            w = min(cell_w, cell_h * t['aspect'])
            h = w / t['aspect']
            self._handles[t['key']] = self.server.scene.add_image(
                f"/cam/{t['key']}", np.zeros((2, 2, 3), np.uint8),
                render_width=w, render_height=h,
                position=(x, y, 0.0), scale=(1.0, -1.0, 1.0), format='jpeg')
            # A SIBLING of the image, never a child. Viser composes a child's transform
            # with its parent's, so a label under /cam/<key> would inherit both the
            # plane's position and its (1,-1,1) scale — which displaced captions into
            # neighbouring cells and mirrored their offsets.
            self._labels[t['key']] = self.server.scene.add_label(
                f"/label/{t['key']}", t['label'],
                position=(x - w / 2, y + h / 2 + 0.03, 0.2),
                anchor='bottom-left', depth_test=False)
            self._label_text[t['key']] = t['label']
        self._layout = [t['key'] for t in tiles]
        # Distance at which the whole grid fits a 16:9 viewport, then re-frame everyone.
        total_h = rows * (cell_h + gap)
        total_w = cols * (cell_w + gap)
        # Assume a 3:2 viewport rather than 16:9: a browser window is usually squarer
        # than 16:9 once toolbars are subtracted, and guessing too wide crops the grid
        # horizontally. The extra 15% keeps the captions inside the frame too.
        self._fit_distance = (max(total_h, total_w / 1.5) / 2
                              / math.tan(self.FOV / 2) * 1.15)
        for client in self.server.get_clients().values():
            self._frame(client)

    def update(self, tiles: list):
        """Push new JPEG bytes to the tiles that changed, rebuilding the grid if needed.

        Only tiles flagged 'fresh' are pushed. The loop wakes on *any* camera, so with
        several cameras running it iterates far faster than any one of them produces
        frames; re-assigning every handle each pass would queue the same bytes onto the
        websocket over and over, adding delay ahead of the frames that are actually new.
        """
        keys = [t['key'] for t in tiles]
        if keys != self._layout:
            self._rebuild_layout(tiles)
            for t in tiles:   # a fresh layout has nothing on it yet
                t = dict(t, fresh=True)
                self._handles[t['key']]._data = t['jpeg'] or self._placeholder_jpeg
        # Share the budget evenly across tiles. With few tiles min_gap is smaller than
        # a camera period, so nothing is throttled; it only engages once the fleet
        # would outrun the browser.
        min_gap = (len(tiles) / self.push_budget_hz) if self.push_budget_hz > 0 else 0.0
        now = time.monotonic()

        # Tried server.atomic() around this loop, on the theory that per-tile client
        # reconciliation was the cost: no measurable change (63.4 vs 64.3 ms), so the
        # backlog is inside the client's message/decode queue, not per-update overhead.
        pushed = False
        for t in tiles:
            key = t['key']
            if t.get('fresh') and t['jpeg'] is not None:
                if now - self._last_push.get(key, 0.0) < min_gap:
                    continue   # browser is behind; this frame would only queue
                self._last_push[key] = now
                self._handles[key]._data = t['jpeg']
                pushed = True
            if self._label_text[key] != t['label']:
                self._labels[key].text = t['label']
                self._label_text[key] = t['label']
                pushed = True
        if pushed and VISER_FLUSH:
            # See VISER_FLUSH: without this a frame can wait out a 16.7 ms window,
            # which is larger than everything else in this pipeline combined.
            self.server.flush()


# ============================================================================
# CLI
# ============================================================================

@dataclass
class ServerConfig:
    """Run streaming server on this machine"""
    backend: Literal["realsense", "usb"]
    """Camera backend to use"""
    device: str
    """RealSense serial number or USB device path (e.g. /dev/video0)"""
    width: int = 1280
    """Image width"""
    height: int = 720
    """Image height"""
    fps: int = 30
    """Camera framerate"""
    port: int = DEFAULT_PORT
    """TCP port to bind on"""
    quality: int = 80
    """JPEG quality for color stream (0=raw lossless, 1-100=JPEG). Default 80."""
    ir: bool = False
    """Stream IR/grayscale instead of color (default: color)"""
    debug: bool = False
    """Print publish FPS to stdout"""

@dataclass
class ClientConfig:
    """View remote camera stream"""
    host: str
    """Server IP or hostname"""
    port: int = DEFAULT_PORT
    """Server TCP port"""

@dataclass
class MultiServerConfig:
    """Run multi-camera streaming server on this machine"""
    backend: Literal["realsense", "usb"]
    """Camera backend for all cameras"""
    devices: list[str] = field(default_factory=lambda: list(_site.CAMERA_SERIALS))
    """Device serial numbers or paths — one per camera, in port order (e.g. --devices
    serial1 serial2). Defaults to `vs_site.CAMERA_SERIALS` (VS_CAMERA_SERIALS /
    vs_site_local.py) when that is non-empty. Empty — the shipped default, or a bare
    `--devices` with no values — auto-discovers every attached camera of this backend
    type, in USB enumeration order."""
    hotplug: bool = True
    """Watch for cameras being plugged in and unplugged while running. Each camera keeps
    a sticky port for the life of the process, so one that is unplugged and plugged back
    in returns to the port its subscribers already know. Implies auto-discovery, so it
    is ignored when --devices is non-empty (given explicitly or from the site default)."""
    max_cameras: int = 16
    """Size of the reserved port range when hotplugging: [base_port, base_port+N)."""
    passthrough: bool = True
    """USB backend only: forward the webcam's own MJPEG instead of decoding it and
    re-encoding. Cuts publisher CPU from 22% to 0.4% of a core across three 720p
    cameras and removes a generation of JPEG loss, at ~50% more bytes on the wire.
    Turn off if the link is bandwidth-limited, or if something on the server side needs
    decoded pixels (nothing in multi-server mode does)."""
    base_port: int = DEFAULT_PORT
    """First TCP port; camera i binds base_port + i"""
    width: int = 1280
    """Image width"""
    height: int = 720
    """Image height"""
    fps: int = 30
    """Camera framerate"""
    quality: int = 80
    """JPEG quality (0=raw lossless, 1-100=JPEG)"""
    ir: bool = False
    """Stream IR/grayscale instead of color"""
    with_ir: bool = False
    """Stream IR *alongside* color, as a second plane (RealSense only). The viewer
    shows each such camera as two tiles: color and infrared."""
    exposure: Optional[int] = 8000
    """Manual exposure in µs for the RealSense *infrared* imager. 8000 is chosen so the
    IR tiles are actually viewable; drop to ~100 for AprilTag work, where a short
    integration keeps tag edges sharp at the cost of a near-black image. Must stay under
    one frame period (33000 µs at 30fps) or the sensor slows down. None = auto-exposure."""
    color_exposure: Optional[int] = None
    """Manual exposure in µs for the RealSense *colour* imager, which needs a far
    longer integration than IR to look right. Default None = auto-exposure, pinned to
    the requested framerate so it can never trade fps for brightness."""
    debug: bool = False
    """Print publish FPS to stdout"""

@dataclass
class MultiClientConfig:
    """View N remote camera streams tiled side-by-side"""
    host: str
    """Server IP or hostname"""
    n_cameras: int
    """Number of cameras"""
    base_port: int = DEFAULT_PORT
    """First TCP port (camera i is at base_port + i)"""
    direct: bool = True
    """Give every camera its own image plane in the browser and forward its JPEG
    undecoded, instead of decoding all streams into one composite and re-encoding that.
    The client then does no image processing at all and each camera refreshes at its own
    rate rather than on a shared render tick: measured over 9 cameras, 118% -> 14% client
    CPU and 8.2 ms -> 0.7 ms from server-send to pixels-on-screen.

    Pass --no-direct for the composite path, which is what you want if the client needs
    the decoded pixels, or if the browser machine would rather decode one stream than N."""
    push_budget_hz: float = 150.0
    """--direct only: total image updates per second handed to the browser, shared
    across tiles. Beyond what the browser can decode and composite, extra frames do not
    appear sooner — they queue, and latency climbs while those frames are never shown.
    Measured with 9 cameras: 270 pushes/s -> 96 ms, 135/s -> 40 ms. Raise it for a
    faster client machine; 0 disables capping. Costs no quality or resolution, only
    frames that would not have been displayed."""
    render_hz: float = 60.0
    """Upper bound on redraws per second. Cameras run at ~30 fps with independent
    phase, so a frame waits on average half a render period before it is drawn:
    60 Hz costs one extra composite per camera period and halves that wait, taking
    end-to-end latency from ~19.8 ms to ~10.7 ms. Uncapped, the loop burns a full
    core redrawing pixels that have not changed."""

if __name__ == '__main__':
    from camera_visualizer import create_visualizer

    config = tyro.cli(
        Union[
            Annotated[ServerConfig,      tyro.conf.subcommand("server")],
            Annotated[ClientConfig,      tyro.conf.subcommand("client")],
            Annotated[MultiServerConfig, tyro.conf.subcommand("multi-server")],
            Annotated[MultiClientConfig, tyro.conf.subcommand("multi-client")],
        ]
    )

    if isinstance(config, ServerConfig):
        if config.backend == 'realsense':
            from camera_realsense_utils import RealSenseCameraBackend
            backend = RealSenseCameraBackend(
                config.device,
                ir_stream_flag=config.ir,
                color_stream_flag=not config.ir,
                depth_stream_flag=False,
                compute_gray_in_capture=config.ir,
            )
        else:
            from camera_utils import USBCameraBackend
            backend = USBCameraBackend(device=config.device)

        backend.initialize(config.width, config.height, config.fps)
        print("Press Ctrl-C to stop.")
        with CameraStreamingServer(backend, port=config.port, jpeg_quality=config.quality, debug=config.debug):
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping server.")

    elif isinstance(config, MultiServerConfig):
        def make_backend(device):
            if config.backend == 'realsense':
                from camera_realsense_utils import RealSenseCameraBackend
                # with_ir keeps the colour stream and adds infrared; the backend then
                # fills gray from the IR imager, so no colour->gray conversion is needed.
                return RealSenseCameraBackend(device,
                                              ir_stream_flag=config.ir or config.with_ir,
                                              color_stream_flag=not config.ir,
                                              depth_stream_flag=False,
                                              exposure=config.exposure,
                                              color_exposure=config.color_exposure,
                                              compute_gray_in_capture=config.ir)
            from camera_utils import USBCameraBackend
            return USBCameraBackend(device=device, passthrough=config.passthrough)

        enumerate_devices = (enumerate_realsense if config.backend == 'realsense'
                             else enumerate_usb_cameras)

        if config.devices:
            # Explicit device list: fixed ports, no discovery, no hotplug.
            backends = []
            for device in config.devices:
                b = make_backend(device)
                b.initialize(config.width, config.height, config.fps)
                backends.append(b)
            print("Press Ctrl-C to stop.")
            server = MultiCameraStreamingServer(backends, base_port=config.base_port,
                                                jpeg_quality=config.quality,
                                                debug=config.debug)
        else:
            print(f"Auto-discovering {config.backend} cameras on ports "
                  f"{config.base_port}-{config.base_port + config.max_cameras - 1}"
                  f"{' (hotplug enabled)' if config.hotplug else ''}. Press Ctrl-C to stop.")
            server = HotplugCameraStreamingServer(
                enumerate_devices, make_backend,
                base_port=config.base_port, max_cameras=config.max_cameras,
                width=config.width, height=config.height, fps=config.fps,
                jpeg_quality=config.quality, debug=config.debug,
                # Without hotplug this still discovers cameras at startup; it just
                # never rescans, so a later plug or unplug is not picked up.
                poll_interval=1.0 if config.hotplug else 1e9)

        with server:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping multi-camera server.")

    elif isinstance(config, MultiClientConfig):
        client = MultiStreamingCameraClient(config.host, config.n_cameras,
                                            config.base_port, decode=not config.direct)
        client.initialize_all()
        client.wait_all(timeout=6)
        # Deliberately no drop_disconnected() here. Every configured port keeps a tile
        # for the whole session: a port that is quiet now may be a camera that has not
        # been plugged in yet, and its subscriber keeps dialling, so it lights up on its
        # own when the camera arrives. A fixed tile count also keeps the grid layout
        # from reshuffling every time a camera comes or goes.
        waiting = [b.port for b in client if not b.connected]
        if waiting:
            print(f"Waiting on port(s) {', '.join(map(str, waiting))} — "
                  f"they will appear automatically when their camera is plugged in.")

        if config.direct:
            import viser
            viser_server = viser.ViserServer(port=8080)
            print(f"[Viser] Web viewer: http://localhost:{viser_server.get_port()}")
            viewer = DirectViserViewer(viser_server, config.host)
            viewer.push_budget_hz = config.push_budget_hz
            trackers = [LatencyTracker(name=f"cam{i}") for i in range(len(client))]
            fps_counters = [FPS() for _ in client]
            drawn = [0.0] * len(client)
            last_print = time.perf_counter()
            try:
                while True:
                    results = client.capture_all_parallel(timeout=0.5, require_all=False)
                    tiles = []
                    for i, b in enumerate(client):
                        if results[i]:
                            fps_counters[i].tick()
                            trackers[i].update(client.get_latency_stats(i))
                        # Captions live inside the tile, so keep them short: the port
                        # alone identifies the camera and the host is the same for all.
                        if not b.connected:
                            tiles.append({'key': f'p{b.port}', 'aspect': 4 / 3,
                                          'label': f':{b.port} waiting', 'jpeg': None,
                                          'fresh': False})
                            continue
                        stale = "" if b.live() else " UNPLUGGED"
                        aspect = b.width / b.height
                        tiles.append({
                            'key': f'p{b.port}', 'aspect': aspect,
                            'label': f':{b.port} {fps_counters[i].fps:.0f}fps{stale}',
                            'jpeg': b.get_encoded_color(), 'fresh': results[i]})
                        if b.has_ir and b.get_encoded_ir() is not None:
                            tiles.append({'key': f'p{b.port}ir', 'aspect': aspect,
                                          'label': f':{b.port} IR{stale}',
                                          'jpeg': b.get_encoded_ir(), 'fresh': results[i]})
                    viewer.update(tiles)
                    now = time.monotonic()
                    for i, b in enumerate(client):
                        s = b.last_capture_stamp
                        if b.connected and s > 0 and s != drawn[i]:
                            drawn[i] = s
                            trackers[i].update_display((now - s) * 1000.0)
                    if time.perf_counter() - last_print >= 1.0:
                        for t in trackers:
                            print(t.summary())
                        last_print = time.perf_counter()
            finally:
                client.cleanup()
            raise SystemExit(0)

        # 9 decode threads are already live; letting OpenCV also spin up one worker per
        # core for a 480x360 resize is pure overhead. Measured: 263% -> 202% client CPU
        # at identical latency.
        cv2.setNumThreads(4)

        viz = create_visualizer()
        per_cam_fps = [FPS() for _ in client]
        latency_trackers = [LatencyTracker(name=f"cam{i}") for i in range(len(client))]
        # Shown for ports with no camera yet. Dark grey rather than black so an empty
        # slot is visibly a slot, not a rendering failure.
        placeholder = np.full((480, 640, 3), 40, dtype=np.uint8)
        # Send stamp of the last frame each camera has actually had drawn, so display
        # latency is counted once per frame rather than once per render tick.
        last_drawn_stamp = [0.0] * len(client)
        # Previous liveness per camera, so a tile that changes to or from UNPLUGGED is
        # redrawn even though no new frame arrived to mark it dirty.
        live_flag = [False] * len(client)
        last_stats_print = time.perf_counter()
        last_render = 0.0
        render_period = 1.0 / config.render_hz if config.render_hz > 0 else 0.0
        try:
            while True:
                # Capture all cameras simultaneously and record client-side timestamps.
                # require_all=False: a viewer must not run at the slowest camera's rate.
                results = client.capture_all_parallel(timeout=0.5, require_all=False)
                # Rate-limit the redraw, not the wake: capture_all_parallel() already
                # returned the instant a frame landed, so this only gives back CPU we
                # would have spent compositing pixels that have not changed.
                slack = render_period - (time.perf_counter() - last_render)
                if slack > 0:
                    time.sleep(slack)
                last_render = time.perf_counter()
                now = time.perf_counter()
                for i, ok in enumerate(results):
                    if ok:
                        per_cam_fps[i].tick()
                        stats = client.get_latency_stats(i)
                        latency_trackers[i].update(stats)
                # Print latency summary every second
                if now - last_stats_print >= 1.0:
                    for lt in latency_trackers:
                        print(lt.summary())
                    last_stats_print = now
                # Build display. `dirty` marks the tiles whose camera actually
                # delivered since the last redraw; the rest keep the pixels already
                # on the reused canvas.
                frames_bgr, labels, dirty = [], [], []
                for i, b in enumerate(client):
                    addr = f"{config.host}:{b.port}"
                    fresh = results[i]
                    if not b.connected:
                        # No camera on this port yet. Hold its slot so the grid does not
                        # reshuffle when it arrives, and say plainly that it is empty.
                        frames_bgr.append(placeholder)
                        labels.append(f"{addr}  waiting for camera")
                        dirty.append(False)
                        continue
                    f = b.get_frame_buffer()
                    if f.ndim == 2:
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    # A camera unplugged mid-session keeps its last frame on screen
                    # forever; say so rather than showing a convincing freeze.
                    was_live = live_flag[i]
                    is_live = b.live()
                    stale = "" if is_live else "  UNPLUGGED"
                    live_flag[i] = is_live
                    frames_bgr.append(f)
                    labels.append(f"{addr}  {per_cam_fps[i].fps:.1f} fps{stale}")
                    # Redraw on a new frame, and also on the transition into or out of
                    # UNPLUGGED so that label change actually reaches the screen.
                    dirty.append(bool(fresh) or is_live != was_live)
                    if getattr(b, 'has_ir', False):
                        # Same camera, second tile: the infrared imager.
                        frames_bgr.append(cv2.cvtColor(b.get_gray_buffer(), cv2.COLOR_GRAY2BGR))
                        labels.append(f"{addr}  IR{stale}")
                        dirty.append(dirty[-1])
                display = tile_frames(frames_bgr, labels, dirty=dirty)
                viz.show("Multi Camera", display)
                # Latency of each frame until it FIRST reaches the renderer. Recorded
                # once per distinct frame, keyed on the server's send stamp: sampling
                # every tick instead would average the age of whatever is on screen,
                # which comes out the same at any render rate (a frame redrawn 3x is
                # 2 ms old, then 13, then 24) and so cannot tell render rates apart.
                drawn_at = time.monotonic()
                for i, b in enumerate(client):
                    stamp = b.last_capture_stamp
                    if b.connected and stamp > 0 and stamp != last_drawn_stamp[i]:
                        last_drawn_stamp[i] = stamp
                        latency_trackers[i].update_display((drawn_at - stamp) * 1000.0)
                # wait_key(1) sleeps a flat 1 ms per iteration for nothing; the render
                # cap above is what paces the loop now.
                if viz.wait_key(0) == ord('q'):
                    break
        finally:
            client.cleanup()
            viz.cleanup("Multi Camera")

    else:  # client viewer (ClientConfig)
        backend = StreamingCameraBackend(host=config.host, port=config.port)
        backend.initialize(0, 0, 0)
        backend.init_complete.wait(timeout=6)

        if backend.gray_buf_internal is None:
            print("Failed to connect to server. Exiting.")
            raise SystemExit(1)

        viz = create_visualizer()
        fps = FPS()
        tracker = LatencyTracker(name=f"{config.host}:{config.port}")
        last_stats_print = time.perf_counter()
        try:
            while True:
                if not backend.capture_frame():
                    continue
                fps.tick()
                stats = backend.get_latency_stats()
                tracker.update(stats)
                now = time.perf_counter()
                if now - last_stats_print >= 1.0:
                    print(tracker.summary())
                    last_stats_print = now
                frame = backend.get_frame_buffer()
                if frame.ndim == 2:
                    # Normalize for display — raw IR values can be very low (e.g. 15-27)
                    # and appear black without stretching. Raw values still flow to callers.
                    pmin, pmax = int(frame.min()), int(frame.max())
                    span = pmax - pmin or 1
                    stretched = ((frame.astype(np.float32) - pmin) * (255.0 / span)).astype(np.uint8)
                    display = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
                    viz.add_text(display, f"raw px: {pmin}-{pmax}", (10, 90), 0.6, (200, 200, 0), 1)
                else:
                    display = frame
                viz.add_text(display, f"FPS: {fps.fps:.1f}", (10, 30), 0.8, (0, 255, 0), 2)
                viz.add_text(display, f"{config.host}:{config.port}", (10, 60), 0.6, (200, 200, 200), 1)
                viz.show("Remote Camera", display)
                if viz.wait_key(1) == ord('q'):
                    break
        finally:
            backend.cleanup()
            viz.cleanup("Remote Camera")
