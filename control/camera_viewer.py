"""
camera_viewer.py — run this on your LAPTOP (not the robot).

Listens for JPEG frames sent by frame_streamer.py and displays them in
local cv2.imshow windows. No SSH/X11 involved — this is a fully local
GUI process, so it's fast and reliable on wifi.

Usage:
    python camera_viewer.py [--port 9871]

Press 'q' in any window to quit.
"""

import argparse
import socket
import time

import cv2
import numpy as np

MAX_PACKET_BYTES = 60000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9871, help="UDP port to listen on")
    parser.add_argument(
        "--bind", default="0.0.0.0", help="Local IP to bind to (default: all interfaces)"
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    print(f"Listening for camera frames on UDP port {args.port}...")
    print("Press 'q' in a camera window to quit.")

    last_frame_time = {}

    try:
        while True:
            try:
                packet, addr = sock.recvfrom(MAX_PACKET_BYTES + 300)
            except socket.timeout:
                # No frame recently; still service the GUI event loop so
                # windows stay responsive and 'q' is detected even when idle.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if len(packet) < 1:
                continue

            name_len = packet[0]
            name = packet[1 : 1 + name_len].decode("utf-8", errors="replace")
            jpeg_bytes = packet[1 + name_len :]

            img_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            now = time.time()
            fps = 1.0 / (now - last_frame_time[name]) if name in last_frame_time else 0.0
            last_frame_time[name] = now

            cv2.putText(
                img,
                f"{fps:4.1f} fps",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(name, img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
