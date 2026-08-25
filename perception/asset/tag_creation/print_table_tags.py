#!/usr/bin/env python3
# =============================================================================
# STALE — DO NOT RUN WITHOUT UPDATING FIRST.
#
# This generator's geometry and coordinate-system assumptions no longer match
# tagged_bodies/table/TableConfig. It still assumes: length along body +y, width
# along +x, +z DOWN into the slab, and a tag strip laid PERPENDICULAR to the
# reference edge. The current TableConfig uses the opposite: length along +x,
# width along +y, +z UP, and the tables carry a row of tags ALONG the length.
# It also still models the old two-roll scheme; TAG_ROLLS now holds a single
# roll ("flat_on_top_x_along_y").
#
# lab_table_a / lab_table_b are now HAND-MAINTAINED (edit those files directly).
# Running this script as-is would emit configs in the WRONG frame/layout, so
# main() hard-stops below. Before reusing it for any table, rewrite the frame,
# the layout parameterisation and the PDF-guide drawing to match TableConfig,
# then remove the guard in main().
# =============================================================================
"""Register a tagged work table: emit its config AND a to-scale sticking guide.

WHY THE GUIDE MATTERS. A tag lying flat can be stuck at any of four 90-degree
rolls; the resulting table pose differs by that roll. On a 1200 x 600 table a
90-degree error swaps length for width in the collision model — the planner
then believes the table is somewhere it is not. No sentence resolves this
reliably ("x axis points into the table" and "the pattern reads upright" are
both natural readings and differ by exactly 90 degrees), so this script draws
the ACTUAL tag bitmaps at their ACTUAL roll on a 1:1 plan of the table. Decide
by looking; the config is written from the same constant that drew the guide,
so the picture and the robot's belief cannot drift apart.

The guide has two pages:
  1. table plan, 1:1 near the tag strip — cut it out, lay it on the table,
     or just hold a sticker next to the drawing and match the orientation;
  2. a dimensioned diagram of the strip with every distance a ruler can check.

Body frame, tag roll options and the slab cuboid all live in
``tagged_bodies/table`` — this script only measures and draws.

Example (the lab's two 1200x600 tables, stickers 71-73 and 74-76):
    python asset/tag_creation/print_table_tags.py \\
        --names lab_table_a lab_table_b --ids 71-73 74-76
"""
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")                        # PDF only; runs headless
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from moms_apriltag import TagGenerator2, TagGenerator3  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from tagged_bodies.table import TAG_ROLLS  # noqa: E402

MM_IN = 25.4


@dataclass
class Args:
    names: List[str]
    """One body name per table, e.g. lab_table_a lab_table_b."""

    ids: List[str]
    """One id group per table, e.g. 71-73 74-76. Order = from the reference
    edge INTO the table."""

    length_mm: float = 1200.0
    """Along the reference edge."""
    width_mm: float = 600.0
    """Away from the reference edge."""
    thickness_mm: float = 30.0

    black_mm: float = 30.0
    """Tag size, black border to black border — the detector's tag_size."""
    sticker_mm: float = 44.0
    """Whole printed square including the white quiet zone."""
    pitch_mm: float = 44.0
    """Centre-to-centre spacing of adjacent tags."""
    first_centre_mm: float = 22.0
    """Reference edge to the FIRST tag's centre."""

    tag_roll: str = "flat_on_top_x_along_y"
    """Key into tagged_bodies.table.TAG_ROLLS. NOTE: stale — the roll scheme and
    frame changed; kept only so imports resolve (main() hard-stops, see banner)."""

    family: str = "tag36h11"
    output: Optional[str] = None
    write_configs: bool = True


def parse_group(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(",", " ").split():
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def check_geometry(a: Args, n_tags: int) -> None:
    """Refuse to emit a config whose numbers cannot describe a real table."""
    quiet = (a.sticker_mm - a.black_mm) / 2.0
    cell = a.black_mm / 8.0                      # tag36h11: 6 data + 2 border cells
    last = a.first_centre_mm + (n_tags - 1) * a.pitch_mm
    problems = []
    if quiet < cell:
        problems.append(f"quiet zone {quiet:.1f}mm < one cell {cell:.2f}mm — undetectable")
    if a.pitch_mm < a.black_mm:
        problems.append(f"pitch {a.pitch_mm}mm < black {a.black_mm}mm — tags overlap")
    if a.first_centre_mm < a.black_mm / 2:
        problems.append(f"first tag's black area hangs over the reference edge")
    if last + a.black_mm / 2 > a.width_mm:
        problems.append(f"last tag ({last:.0f}mm) runs off the {a.width_mm:.0f}mm width")
    if problems:
        raise SystemExit("geometry rejected:\n  " + "\n  ".join(problems))


def tag_centres_x(a: Args, n: int) -> List[float]:
    """Body-frame x of each tag centre (metres). Origin is the top-surface
    centre, so a tag `d` mm from the reference edge sits at d - width/2."""
    return [(a.first_centre_mm + k * a.pitch_mm - a.width_mm / 2.0) / 1000.0
            for k in range(n)]


def draw_plan(pdf, tg, ids, name, a: Args) -> None:
    """Page 1 — 1:1 plan of the tag strip, tags drawn at their true roll."""
    # Draw a 1:1 window around the strip: full length is impractical on paper,
    # so show 300 mm of the edge centred on the strip. The tags themselves are
    # true size, which is the part being verified.
    win_y = 300.0
    win_x = a.first_centre_mm + (len(ids) - 1) * a.pitch_mm + a.sticker_mm
    fig_w, fig_h = win_y / MM_IN + 1.2, win_x / MM_IN + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.6 / fig_w, 0.9 / fig_h, win_y / MM_IN / fig_w, win_x / MM_IN / fig_h])
    ax.set_xlim(-win_y / 2, win_y / 2)          # along the reference edge (body y)
    ax.set_ylim(0, win_x)                        # away from the edge (body x)
    ax.set_aspect("equal")
    ax.axis("off")

    # table surface + reference edge
    ax.add_patch(mpatches.Rectangle((-win_y / 2, 0), win_y, win_x,
                                    facecolor="#f6f5f1", edgecolor="none"))
    ax.plot([-win_y / 2, win_y / 2], [0, 0], color="#c0392b", lw=2.5)
    ax.text(0, -6, "REFERENCE EDGE  (the tagged 1200 mm long edge)",
            ha="center", va="top", fontsize=7, color="#c0392b")

    roll = np.array(TAG_ROLLS[a.tag_roll])
    # Image rotation: how many CCW 90-deg steps to apply to the bitmap so that
    # the drawn pattern matches `roll`. The bitmap as generated has its own +x
    # to the RIGHT (+y of this plot) and its top edge AWAY from the edge.
    k90 = {"pattern_up_into_table": 0, "tag_x_into_table": 1}[a.tag_roll]

    for k, tag_id in enumerate(ids):
        cx = a.first_centre_mm + k * a.pitch_mm          # from edge (plot y)
        img = np.rot90(tg.generate(tag_id), k90)
        ax.imshow(img, cmap="gray", interpolation="nearest",
                  extent=[-a.black_mm / 2, a.black_mm / 2,
                          cx - a.black_mm / 2, cx + a.black_mm / 2], zorder=3)
        ax.add_patch(mpatches.Rectangle((-a.sticker_mm / 2, cx - a.sticker_mm / 2),
                                        a.sticker_mm, a.sticker_mm, fill=False,
                                        edgecolor="#9aa0a6", lw=0.6, ls=(0, (3, 2)), zorder=2))
        ax.text(a.sticker_mm / 2 + 5, cx, f"{tag_id}", va="center", ha="left",
                fontsize=8, color="#333")
        # this tag's own +x, drawn from its centre
        vx = roll[:, 0]        # tag +x in body coords -> (x=away from edge, y=along edge)
        ax.annotate("", xy=(vx[1] * 20, cx + vx[0] * 20), xytext=(0, cx),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#c0392b"), zorder=4)

    ax.annotate("", xy=(0, win_x - 8), xytext=(0, win_x - 38),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#1f6f3f"))
    ax.text(4, win_x - 12, "body +x  (into the table)", fontsize=7, color="#1f6f3f")

    fig.text(0.5, 1 - 0.42 / fig_h, f"{name} — sticking guide, 1:1", ha="center",
             fontsize=12, fontweight="bold")
    fig.text(0.5, 1 - 0.68 / fig_h,
             f"roll = {a.tag_roll}   ·   red arrow = each tag's own +x   ·   "
             f"dashed = {a.sticker_mm:.0f} mm sticker outline",
             ha="center", fontsize=7.5, color="#52514e")
    fig.text(0.5, 0.30 / fig_h,
             "Print at 100%. Check a tag measures "
             f"{a.black_mm:.0f} mm black-to-black before sticking anything.",
             ha="center", fontsize=7, color="#8a8984")
    pdf.savefig(fig)
    plt.close(fig)


def draw_dims(pdf, ids, name, a: Args) -> None:
    """Page 2 — every distance a ruler can check, on one dimensioned diagram."""
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_axes([0.07, 0.10, 0.86, 0.72])
    n = len(ids)
    span = a.first_centre_mm + (n - 1) * a.pitch_mm + a.black_mm
    ax.set_xlim(-40, 260); ax.set_ylim(-span * 0.22, span * 1.18)
    ax.set_aspect("equal"); ax.axis("off")

    ax.plot([-30, 250], [0, 0], color="#c0392b", lw=2.5)
    ax.text(110, -span * 0.09, "REFERENCE EDGE", ha="center", fontsize=8, color="#c0392b")

    for k, tag_id in enumerate(ids):
        cx = a.first_centre_mm + k * a.pitch_mm
        ax.add_patch(mpatches.Rectangle((0, cx - a.sticker_mm / 2), a.sticker_mm,
                                        a.sticker_mm, fill=False, edgecolor="#9aa0a6",
                                        lw=0.7, ls=(0, (3, 2))))
        ax.add_patch(mpatches.Rectangle((7, cx - a.black_mm / 2), a.black_mm,
                                        a.black_mm, facecolor="#1a1a1a"))
        ax.text(a.sticker_mm + 6, cx, f"ID {tag_id}", va="center", fontsize=9)
        ax.annotate("", xy=(150, cx), xytext=(150, 0),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="#52514e"))
        ax.text(154, cx / 2, f"{cx:.0f} mm", fontsize=8, color="#52514e", va="center")

    rows = [f"table            {a.length_mm:.0f} x {a.width_mm:.0f} x {a.thickness_mm:.0f} mm",
            f"family           {a.family}",
            f"tag black        {a.black_mm:.0f} mm   (detector tag_size)",
            f"sticker          {a.sticker_mm:.0f} mm   (quiet zone "
            f"{(a.sticker_mm - a.black_mm) / 2:.0f} mm per side)",
            f"pitch            {a.pitch_mm:.0f} mm centre-to-centre",
            f"strip centred    {a.length_mm / 2:.0f} mm from each short edge",
            f"roll             {a.tag_roll}"]
    ax.text(196, span * 0.98, "\n".join(rows), fontsize=8, family="monospace",
            va="top", ha="left", color="#0b0b0b")

    fig.text(0.5, 0.93, f"{name} — dimensions", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.885, "every number below is measurable with a ruler on the real table",
             ha="center", fontsize=8, color="#52514e")
    pdf.savefig(fig)
    plt.close(fig)


def write_config(ids, name, a: Args) -> str:
    xs = tag_centres_x(a, len(ids))
    lines = [f'        "tag_{i}": TagConfig(id={i}, '
             f'pos=[{x:.6f}, 0.000000, 0.000000], rot=TAG_ROLL),'
             for i, x in zip(ids, xs)]
    content = f'''"""Auto-generated by asset/tag_creation/print_table_tags.py — do not hand-edit.

Table "{name}": {a.length_mm:.0f} x {a.width_mm:.0f} x {a.thickness_mm:.0f} mm, {len(ids)} x {a.family}
stickers ({a.black_mm:.0f} mm black) along one long edge at {a.first_centre_mm:.0f}/{"/".join(
    f"{a.first_centre_mm + k * a.pitch_mm:.0f}" for k in range(1, len(ids)))} mm from
that edge, centred on its midpoint. Tag roll: {a.tag_roll}.

Body frame: origin at the TOP SURFACE centre, +x from the reference edge into
the table, +y along the edge to the right, +z down into the slab.
"""
import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_bodies.table import TAG_ROLLS, TableConfig
from tagged_rigid_body import TagConfig, verify_bodies

TAG_ROLL = TAG_ROLLS["{a.tag_roll}"]

cfg = TableConfig(
    name="{name}",
    length_m={a.length_mm / 1000.0:.4f},
    width_m={a.width_mm / 1000.0:.4f},
    thickness_m={a.thickness_mm / 1000.0:.4f},
    tag_size={a.black_mm / 1000.0:.4f},
    tag_border_size={(a.sticker_mm - a.black_mm) / 2 / 1000.0:.4f},
    tag_family="{a.family}",
    tag_roll="{a.tag_roll}",
    tags={{
{chr(10).join(lines)}
    }},
)

if __name__ == "__main__":
    verify_bodies(cfg)
'''
    path = os.path.join(_ROOT, "tagged_bodies", "table", f"{name}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def main() -> None:
    # Hard guard: this generator is stale (see the banner at the top of the file).
    # Its frame/layout no longer match TableConfig, so it would silently emit
    # wrong-frame configs. Refuse to run until it is updated and this guard removed.
    raise SystemExit(
        "print_table_tags.py is STALE and disabled: its coordinate frame and layout "
        "no longer match tagged_bodies/table/TableConfig (see the banner at the top of "
        "this file). lab_table_a/lab_table_b are now hand-maintained. Update the frame, "
        "layout and PDF-guide drawing, then remove this guard.")
    a = tyro.cli(Args)
    if a.tag_roll not in TAG_ROLLS:
        raise SystemExit(f"--tag-roll must be one of {list(TAG_ROLLS)}")
    groups = [parse_group(s) for s in a.ids]
    if len(groups) != len(a.names):
        raise SystemExit(f"{len(a.names)} names for {len(groups)} id groups")
    flat = [i for g in groups for i in g]
    if len(set(flat)) != len(flat):
        raise SystemExit(f"duplicate ids: {sorted(flat)}")
    for g in groups:
        check_geometry(a, len(g))

    out = a.output or os.path.join(
        _HERE, f"table_tags_{a.family}_{int(a.black_mm)}mm_ids_{min(flat)}-{max(flat)}.pdf")
    try:
        tg = TagGenerator2(a.family)
    except Exception:
        tg = TagGenerator3(a.family)

    print(f"table {a.length_mm:.0f} x {a.width_mm:.0f} x {a.thickness_mm:.0f} mm · "
          f"{a.family} {a.black_mm:.0f}mm black / {a.sticker_mm:.0f}mm sticker · "
          f"pitch {a.pitch_mm:.0f}mm · roll {a.tag_roll}")
    print("-" * 74)
    with PdfPages(out) as pdf:
        for ids, name in zip(groups, a.names):
            draw_plan(pdf, tg, ids, name, a)
            draw_dims(pdf, ids, name, a)
            xs = tag_centres_x(a, len(ids))
            print(f"{name:<15} ids {ids}")
            for i, x in zip(ids, xs):
                d = a.first_centre_mm + (ids.index(i)) * a.pitch_mm
                print(f"{'':<15}  {i}: {d:6.0f} mm from edge  ->  body x = {x * 1000:+7.1f} mm")
            if a.write_configs:
                print(f"{'':<15}  -> {write_config(ids, name, a)}")
    print("-" * 74)
    print(f"guide: {out}   ({2 * len(groups)} pages: plan + dimensions per table)")


if __name__ == "__main__":
    main()
