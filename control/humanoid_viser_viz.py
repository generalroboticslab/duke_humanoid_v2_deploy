"""Standalone viser visualizer — subscribes to humanoid_real_env telemetry (pynng).

Usage:
    python humanoid_viser_viz.py --robot-ip <robot>   # telemetry on port 9870
    python humanoid_viser_viz.py --no-show-robot      # hide MJCF robot mesh
"""
import time
from dataclasses import dataclass
import tyro

from ipc.publisher import NNGSubscriber
from humanoid_model import MJCF_MODEL_PATH
from humanoid_utils import HumanoidViserViz


@dataclass
class Args:
    robot_ip: str = "127.0.0.1"
    """IP of robot running humanoid_real_env.py"""
    port: int = 9870
    """Port robot publishes viz data on"""
    show_robot: bool = True
    """Show MJCF robot mesh (requires vicon calibration data in the stream)"""


def main():
    args = tyro.cli(Args)

    from humanoid_utils import get_joint_info_from_mjcf
    joint_names, _, _ = get_joint_info_from_mjcf(MJCF_MODEL_PATH)

    viz = HumanoidViserViz(
        mjcf_path=MJCF_MODEL_PATH,
        joint_names=joint_names,
        show_robot=args.show_robot,
    )
    print(f"[viser_viz] Connecting to tcp://{args.robot_ip}:{args.port}, viser at http://0.0.0.0:8080")

    rx = NNGSubscriber(f"tcp://{args.robot_ip}:{args.port}")
    rx.start()

    last_data_id = -1
    while True:
        if rx.data_id != last_data_id:
            d = rx.data
            if d is not None:
                viz.update_from_packet(d)
                last_data_id = rx.data_id
        time.sleep(1 / 30)  # 30 Hz render rate


if __name__ == "__main__":
    main()
