"""
Honeycomb grid generator and visualizer.

Used to design/verify AprilTag placement patterns on the 6-tag holder.
Run directly to visualize the grid and print center/vertex coordinates.
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


@dataclass
class Hexagon:
    center: tuple[float, float]
    vertices: np.ndarray  # shape (6, 2), flat-top orientation


def make_hexagon(cx: float, cy: float, r: float) -> Hexagon:
    """Create a flat-top hexagon centered at (cx, cy) with circumradius r."""
    angles = np.radians([30, 90, 150, 210, 270, 330])
    vertices = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
    return Hexagon(center=(cx, cy), vertices=vertices)


def flower_pattern(r: float) -> np.ndarray:
    """
    Return origins of 7 hexagons: 1 center + 6 surrounding neighbors.

    r: center-to-center distance between adjacent hexagons.

    Returns shape (7, 2): center first, then 6 neighbors in CCW order.
    """
    neighbor_angles = np.radians([0, 60, 120, 180, 240, 300])
    neighbors = np.column_stack([r * np.cos(neighbor_angles), r * np.sin(neighbor_angles)])
    return np.vstack([[0.0, 0.0], neighbors])


def honeycomb_grid(rows: int, cols: int, r: float) -> np.ndarray:
    """
    Return origins of a honeycomb grid of flat-top hexagons.

    r: center-to-center distance between adjacent hexagons.
      - Horizontal spacing: r
      - Vertical spacing:   r * sqrt(3) / 2
      - Odd rows are offset right by r / 2.

    Returns shape (rows * cols, 2), row-major order.
    """
    dx = r
    dy = r * np.sqrt(3) / 2
    row_idx, col_idx = np.mgrid[0:rows, 0:cols]
    x = col_idx * dx + (row_idx % 2) * (dx / 2)
    y = row_idx * dy
    return np.column_stack([x.ravel(), y.ravel()])


def visualize(hexagons: list[Hexagon], title: str = "Honeycomb") -> None:
    """Plot the honeycomb grid with centers marked."""
    _, ax = plt.subplots(figsize=(10, 8))

    patches = [Polygon(h.vertices, closed=True) for h in hexagons]
    ax.add_collection(PatchCollection(
        patches, facecolor="#FFD700", edgecolor="#333333", linewidth=1.5, alpha=0.8
    ))

    centers = np.array([h.center for h in hexagons])
    ax.scatter(centers[:, 0], centers[:, 1], color="red", s=20, zorder=5)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    R = 14.625  # center-to-center distance between adjacent hexagons
    circumradius = R / np.sqrt(3)  # vertex distance, used only for drawing

    # --- Flower: 1 center + 6 surrounding hexagons ---
    origins = flower_pattern(R)  # shape (7, 2)
    print("=== Flower pattern ===")
    for i, (x, y) in enumerate(origins):
        label = "center" if i == 0 else f"neighbor {i}"
        print(f"  Hex {i} ({label}) | origin ({x:.3f}, {y:.3f})")
    flower_hexagons = [make_hexagon(x, y, circumradius) for x, y in origins]
    visualize(flower_hexagons, title=f"Flower pattern — 1 center + 6 neighbors (r={R})")

    # --- Grid: full honeycomb ---
    ROWS, COLS = 5, 6
    origins = honeycomb_grid(ROWS, COLS, R)  # shape (ROWS*COLS, 2)
    print(f"\n=== Honeycomb grid {ROWS}×{COLS} ===")
    for i, (x, y) in enumerate(origins):
        print(f"  Hex {i:02d} | origin ({x:.3f}, {y:.3f})")
    grid_hexagons = [make_hexagon(x, y, circumradius) for x, y in origins]
    visualize(grid_hexagons, title=f"Honeycomb — {ROWS}×{COLS} grid, flat-top (r={R})")
