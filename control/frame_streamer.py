"""
frame_streamer.py — send JPEG-compressed camera frames over UDP.

Drop this file next to humanoid_camera_point.py on the robot, then
import FrameStreamer and call .send(name, img) wherever you currently
call cv2.imshow(name, img).

UDP is used (not TCP) so a dropped/late frame is simply skipped rather
than stalling the control loop — important since this runs in the same
process as your real-time motor control code.

Each frame is sent as a single UDP datagram, JPEG-encoded and capped in
size so it fits comfortably under typical MTU-safe limits over wifi.
If a frame is too large after compression, it's silently dropped (and
a low-rate warning printed) rather than fragmented, since IP fragmentation
is unreliable over wifi and we'd rather skip a frame than corrupt one.
"""

import socket
import struct
import time
import cv2
import numpy as np

# Max UDP payload we'll send in one packet. Real-world safe limit well
# under the 65507-byte theoretical max, and under common MTU fragmentation
# thresholds for wifi.
MAX_PACKET_BYTES = 60000


class FrameStreamer:
    def __init__(self, dest_ip: str, dest_port: int = 9871, jpeg_quality: int = 60):
        """
        dest_ip:  IP of the machine you're viewing on (your laptop).
        dest_port: UDP port the viewer listens on.
        jpeg_quality: 1-100, lower = smaller/faster but blockier.
        """
        self.dest = (dest_ip, dest_port)
        self.jpeg_quality = jpeg_quality
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow a decently large send buffer so bursts don't block.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._last_warn_time = 0.0

    def send(self, name: str, img: np.ndarray):
        """Encode `img` as JPEG and send it tagged with `name` (e.g. camera id)."""
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return

        data = buf.tobytes()
        if len(data) > MAX_PACKET_BYTES:
            now = time.time()
            if now - self._last_warn_time > 2.0:
                print(
                    f"[FrameStreamer] frame for '{name}' is {len(data)} bytes, "
                    f"over the {MAX_PACKET_BYTES}-byte limit — dropping. "
                    f"Lower jpeg_quality or resolution."
                )
                self._last_warn_time = now
            return

        # Simple header: 1 byte name length, name bytes, then JPEG bytes.
        name_bytes = name.encode("utf-8")[:255]
        header = struct.pack("!B", len(name_bytes)) + name_bytes
        try:
            self.sock.sendto(header + data, self.dest)
        except OSError:
            # e.g. network temporarily unreachable — just skip this frame.
            pass

    def close(self):
        self.sock.close()
