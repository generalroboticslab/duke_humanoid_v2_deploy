"""Record the IMU (hardware_bindings.imu) and a Vicon object's pose at --freq Hz
(default 200) into a pickle (default logs/recording_imu_vicon_<timestamp>.pkl);
optional live viser view (--viser). Needs pyvicon (optional requirement); tyro
CLI, so --help is safe. IMU/Vicon calibration data capture.
"""
import humanoid_site as _site
import sys, os, time, signal, pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hardware_bindings.imu.py_imu import IMU
try:
    from pyvicon.vicon_utils import ObjectTrackerNonblocking
except ImportError as e:
    print(f"pyvicon is optional and not installed: install the pyvicon package (Vicon DataStream wrapper providing pyvicon.vicon_utils) ({e})", file=sys.stderr)
    raise SystemExit(2)
import tyro


@dataclass
class Args:
    ip: str = _site.VICON_IP    # Vicon IP address
    object: str = "grl_hcal"    # Vicon object name
    output: Optional[str] = None # Output pkl path (default: logs/recording_imu_vicon_TIMESTAMP.pkl)
    viser: bool = False          # Enable Viser visualization
    freq: float = 200.0          # Recording frequency (Hz)


@dataclass
class RecordedData:
    time:             List[float]      = field(default_factory=list)
    imu_acc:          List[np.ndarray] = field(default_factory=list)
    imu_ang_vel:      List[np.ndarray] = field(default_factory=list)
    imu_ang_acc:      List[np.ndarray] = field(default_factory=list)
    imu_quat_wxyz:    List[np.ndarray] = field(default_factory=list)
    imu_gravity:      List[np.ndarray] = field(default_factory=list)
    vicon_pos:        List[np.ndarray] = field(default_factory=list)
    vicon_quat_wxyz:  List[np.ndarray] = field(default_factory=list)
    vicon_frame:      List[int]        = field(default_factory=list)
    vicon_latency:    List[float]      = field(default_factory=list)
    vicon_timestamp:  List[float]      = field(default_factory=list)


def setup_viser():
    import viser
    srv = viser.ViserServer(host="0.0.0.0", port=8080)
    print("  Viser:  http://localhost:8080")
    srv.scene.add_frame("/world", axes_length=0.5, axes_radius=0.01)
    frames = {
        'vicon': srv.scene.add_frame("/vicon", axes_length=0.3, axes_radius=0.008),
        'imu':   srv.scene.add_frame("/imu",   axes_length=0.2, axes_radius=0.005),
    }
    srv.scene.add_label("/vicon/label", "Vicon", position=(0., 0., 0.35))
    srv.scene.add_label("/imu/label",   "IMU",   position=(0., 0., 0.25))
    return frames


def update_viser(frames, imu_quat_wxyz, vicon_pos, vicon_frame, vicon_quat_wxyz):
    try:
        frames['imu'].wxyz = imu_quat_wxyz
        if vicon_frame != 0:
            frames['vicon'].position = vicon_pos
            frames['vicon'].wxyz = vicon_quat_wxyz
            frames['imu'].position = vicon_pos
    except Exception as e:
        print(f"Viser error: {e}")


def main():
    args = tyro.cli(Args)

    if args.output is None:
        Path("logs").mkdir(exist_ok=True)
        output_path = f"logs/recording_imu_vicon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    else:
        output_path = args.output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Recorder...\n  Output: {output_path}\n  Freq:   {args.freq} Hz")

    frames = setup_viser() if args.viser else {}

    print("  IMU:    Connecting...")
    imu = IMU()
    for _ in range(10):
        if imu.counter > 0: break
        time.sleep(0.1)
    print("  IMU:    Connected." if imu.counter > 0 else "  WARNING: IMU counter is 0.")

    vicon = None
    try:
        vicon = ObjectTrackerNonblocking(args.ip, [args.object])
        print(f"  Vicon:  Connected to {args.ip}.")
    except Exception as e:
        print(f"  WARNING: Vicon failed: {e}")

    data = RecordedData()
    running = [True]
    signal.signal(signal.SIGINT, lambda s, f: running.__setitem__(0, False))
    print("\nRecording started. Press Ctrl+C to stop.")

    dt = 1.0 / args.freq
    prev_ang_vel, prev_t = None, None
    start = time.time()

    try:
        while running[0]:
            t0 = time.time()
            t  = t0 - start

            # --- IMU ---
            imu_acc       = imu.raw_acc
            imu_ang_vel   = imu.ang_vel
            imu_quat_xyzw = imu.quat_xyzw
            imu_quat_wxyz = imu_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
            imu_gravity   = imu.gravity_vec
            imu_ang_acc   = (imu_ang_vel - prev_ang_vel) / (t - prev_t) \
                            if prev_ang_vel is not None and t - prev_t > 1e-6 else np.zeros(3)
            prev_ang_vel, prev_t = imu_ang_vel, t

            # --- Vicon ---
            vicon_pos, vicon_quat_wxyz = np.zeros(3), np.array([1., 0., 0., 0.])
            vicon_frame, vicon_latency, vicon_timestamp = 0, 0.0, 0.0
            if vicon:
                res = vicon.get_position(args.object)
                if res is not None:
                    positions, vicon_frame, vicon_latency, vicon_timestamp = res
                    vicon_pos       = positions[0, :3].copy()
                    vicon_quat_wxyz = positions[0, 3:7].copy()
            # --- Store ---
            data.time.append(t)
            data.imu_acc.append(imu_acc);         data.imu_ang_vel.append(imu_ang_vel)
            data.imu_ang_acc.append(imu_ang_acc); data.imu_quat_wxyz.append(imu_quat_wxyz)
            data.imu_gravity.append(imu_gravity)
            data.vicon_pos.append(vicon_pos);     data.vicon_quat_wxyz.append(vicon_quat_wxyz)
            data.vicon_frame.append(vicon_frame); data.vicon_latency.append(vicon_latency)
            data.vicon_timestamp.append(vicon_timestamp)

            if args.viser:
                update_viser(frames, imu_quat_wxyz, vicon_pos, vicon_frame, vicon_quat_wxyz)

            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if len(data.time) % int(args.freq) == 0:
                print(f"[{t:.1f}s] IMU: {imu.counter} (Q:{imu_quat_wxyz[:2]}...), "
                      f"Vicon frame: {vicon_frame} | Pos: {vicon_pos[:2]}")
    finally:
        n = len(data.time)
        if n > 0:
            print(f"\nSaving {n} samples to {output_path}...")
            with open(output_path, 'wb') as f:
                # Convert all dataclass list fields to numpy arrays
                pickle.dump({f.name: np.array(getattr(data, f.name)) for f in fields(data)}, f)
            print("Saved.")
        else:
            print("No data recorded.")
        imu.shutdown()


if __name__ == "__main__":
    main()
