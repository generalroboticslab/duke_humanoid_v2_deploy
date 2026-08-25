"""Per-motor CAN round-trip latency profile: runs the motion-control loop at a
1 % torque ceiling (no position command) for --duration seconds, then prints
avg/std/min/max/p99 per bus and saves latency_report.png (matplotlib, imported
lazily). CAN/adapter diagnosis; argparse, so --help is safe.
"""
import time
import sys
import os
import argparse
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from humanoid_config import motor_setup
from hardware_bindings.motor.py_motor import CanMotorController


@dataclass
class MotorStats:
    id:    int
    iface: str
    avg:   float
    std:   float
    min:   float
    max:   float
    p99:   float
    n:     int
    data:  np.ndarray


def compute_stats(motor_id: int, iface: str, history: list) -> MotorStats | None:
    """Filter out zero-filled buffer slots and compute latency stats."""
    arr = np.array([x for x in history if x > 0.001])
    if len(arr) == 0:
        return None
    return MotorStats(
        id=motor_id, iface=iface,
        avg=arr.mean(), std=arr.std(),
        min=arr.min(),  max=arr.max(),
        p99=np.percentile(arr, 99), n=len(arr),
        data=arr,
    )


def print_results(bus_stats: dict[str, list[MotorStats]]):
    HDR  = f"  {'ID':>3}  {'Avg ms':>7}  {'Std ms':>7}  {'Min ms':>7}  {'Max ms':>7}  {'p99 ms':>7}  {'N':>6}"
    LINE = "  " + "─" * (len(HDR) - 2)

    all_avgs = []
    for iface, stats in sorted(bus_stats.items()):
        print(f"\n{LINE}\n  {iface}  ({len(stats)} motors)\n{HDR}\n{LINE}")
        for s in stats:
            print(f"  {s.id:>3}  {s.avg:>7.3f}  {s.std:>7.3f}  {s.min:>7.3f}  {s.max:>7.3f}  {s.p99:>7.3f}  {s.n:>6}")
            all_avgs.append(s.avg)
        print(f"  {'avg':>3}  {np.mean([s.avg for s in stats]):>7.3f}")

    print(f"\n{'═'*60}")
    print(f"  System Average: {np.mean(all_avgs):.3f} ms  ({len(all_avgs)} motors)")
    print(f"{'═'*60}")


def make_plot(bus_stats: dict[str, list[MotorStats]], duration: float):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    active_ifaces = sorted(bus_stats)
    iface_color = {iface: plt.cm.tab10(i / 10) for i, iface in enumerate(active_ifaces)}

    # Flatten stats into plot-order arrays, grouped by interface
    ids, avgs, datas, colors = [], [], [], []
    group_start = {}
    for iface in active_ifaces:
        group_start[iface] = len(ids) + 1
        for s in bus_stats[iface]:
            ids.append(s.id)
            avgs.append(s.avg)
            datas.append(s.data)
            colors.append(iface_color[iface])

    def annotate_groups(ax):
        for k, iface in enumerate(active_ifaces):
            start = group_start[iface]
            end = group_start[active_ifaces[k + 1]] - 1 if k + 1 < len(active_ifaces) else len(ids)
            ax.text((start + end) / 2, 1.01, iface,
                    ha='center', va='bottom', fontsize=8, color=iface_color[iface],
                    transform=ax.get_xaxis_transform())
            if k > 0:
                ax.axvline(x=start - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    xs = range(1, len(ids) + 1)
    labels = [str(i) for i in ids]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Motor Latency by CAN Interface  ({duration}s)', fontsize=13, fontweight='bold')

    # Boxplot: latency distribution per motor
    bp = ax1.boxplot(datas, tick_labels=labels,
                     patch_artist=True, medianprops=dict(color='black', linewidth=2))
    for k, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[k])
        box.set_alpha(0.5)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_xlabel('Motor ID')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(handles=[Patch(color=iface_color[iface], label=iface) for iface in active_ifaces],
               loc='lower right', fontsize=8)
    annotate_groups(ax1)

    # Bar chart: average latency per motor
    ax2.bar(xs, avgs, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Avg Latency (ms)')
    ax2.set_xlabel('Motor ID')
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    annotate_groups(ax2)

    plt.tight_layout()
    plt.savefig('latency_report.png')
    print("[PROFILE] Saved latency_report.png")


def main():
    parser = argparse.ArgumentParser(description="Profile Motor CAN Bus Latency")
    parser.add_argument("--duration", type=float, default=5.0, help="Profiling duration (s)")
    args = parser.parse_args()

    print(f"{'═'*60}\n  MOTOR LATENCY PROFILING  ({args.duration}s)\n{'═'*60}")

    motor = CanMotorController(motor_setup)
    motor.set_max_torque_ratio(0.01)  # safety
    motor.start_motion_control_continuously()
    print(f"[PROFILE] Collecting for {args.duration}s ...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass

    bus_stats: dict[str, list[MotorStats]] = {}
    for i in range(motor.num_motors):
        iface = motor.can_interfaces[motor.can_interface_ids[i]]
        s = compute_stats(i, iface, motor.get_motor_latency_history(i))
        if s is not None:
            bus_stats.setdefault(iface, []).append(s)

    print_results(bus_stats)
    make_plot(bus_stats, args.duration)


if __name__ == "__main__":
    main()
