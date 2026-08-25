#!/usr/bin/env python3
"""Generate print-ready tag BOARDS: vertical strips of tags at an EXACT pitch.

WHY NOT print_apriltags.py. That script fills a sheet of paper: it derives the
row/column count from the page and then MAXIMISES the gap so the block is
spread and centred. That is right for a wall of tags, wrong for a board whose
geometry is the specification ("50 mm tags, 1 cm gaps, 1 cm margins"). Here the
geometry is the input and the page is the output: each board is its own page,
cut to size, so what you measure on the print is what the config declares.

Frame: the emitted TagBoardConfig is x-UP (see tagged_bodies/tag_board), so all
boards share one orientation convention regardless of how many tags they carry.

Printing accuracy is the whole point of a printed fiducial, so:
  * page size == board size, and the PDF is drawn at true scale;
  * --printer-scale pre-enlarges the drawing so that a printer set to "fit to
    printable area" (which silently shrinks by e.g. 0.9388) still lands the
    physical tag at exactly --size-mm — the same compensation print_apriltags.py
    uses, kept identical so both tools behave the same on the same printer;
  * a 10 mm ruler tick and the nominal size are printed in the margin: measure
    it after printing, before sticking anything to hardware.

Examples:
    # the four 3-tag boards, ids 26-37, 50 mm tags, 1 cm gaps/margins
    python asset/tag_creation/print_tag_board.py \\
        --ids 26-28 29-31 32-34 35-37 --names box_1 box_2 box_3 box_4

    # school printer that shrinks to fit
    ... --printer-scale 0.9388
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")          # PDF only; never open a GUI (this runs headless on the robot PC)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from moms_apriltag import TagGenerator2, TagGenerator3  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # visual_servoing/
sys.path.insert(0, _ROOT)

MM_PER_IN = 25.4


@dataclass
class Args:
    ids: List[str]
    """One group per board, each 'a-b' or comma-separated, e.g. 26-28 29-31."""

    names: List[str] = field(default_factory=list)
    """Body name per board (default board_0, board_1, ...). Must match --ids length."""

    size_mm: float = 50.0
    """Tag size, BLACK BORDER TO BLACK BORDER — the same definition the detector
    and every tagged_bodies config uses."""

    gap_mm: float = 10.0
    """White gap between adjacent tags' black borders."""

    margin_mm: float = 10.0
    """White backing margin around the strip (also the page margin)."""

    family: str = "tag36h11"
    output: Optional[str] = None
    """Output PDF (default asset/tag_creation/tag_board_{family}_{size}mm_{ids}.pdf)."""

    printer_scale: float = 1.0
    """Scale the printer will apply (<1.0 shrinks). The PDF is pre-enlarged by
    1/printer_scale so the PHYSICAL tag still measures --size-mm."""

    show_labels: bool = True
    """Print each tag's ID in the margin beside it."""

    write_configs: bool = True
    """Emit tagged_bodies/tag_board/{name}.py for each board."""


def parse_group(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(",", " ").split():
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def make_generator(family: str):
    try:
        return TagGenerator2(family)
    except Exception:
        return TagGenerator3(family)


def board_page(pdf, tg, ids, name, args) -> None:
    """One board == one PDF page, drawn at (pre-scaled) true size."""
    inv = 1.0 / args.printer_scale
    tag_in = args.size_mm * inv / MM_PER_IN
    gap_in = args.gap_mm * inv / MM_PER_IN
    mar_in = args.margin_mm * inv / MM_PER_IN
    page_w = tag_in + 2 * mar_in
    page_h = len(ids) * tag_in + (len(ids) - 1) * gap_in + 2 * mar_in

    fig = plt.figure(figsize=(page_w, page_h))
    for row, tag_id in enumerate(ids):
        x0 = mar_in
        # rows run top -> bottom; matplotlib y is measured from the bottom
        y0 = page_h - mar_in - (row + 1) * tag_in - row * gap_in
        ax = fig.add_axes([x0 / page_w, y0 / page_h, tag_in / page_w, tag_in / page_h])
        ax.imshow(tg.generate(tag_id), cmap="gray", interpolation="nearest")
        ax.axis("off")
        if args.show_labels:
            fig.text((x0 + tag_in / 2) / page_w, (y0 - mar_in * 0.45) / page_h,
                     f"{tag_id}", ha="center", va="center", fontsize=6, color="0.75")

    # x-axis arrow up the left margin — the frame this board's config declares
    axf = fig.add_axes([0, 0, 1, 1]); axf.set_xlim(0, page_w); axf.set_ylim(0, page_h); axf.axis("off")
    ax_x = mar_in * 0.42
    axf.annotate("", xy=(ax_x, page_h - mar_in * 0.3), xytext=(ax_x, mar_in * 0.3),
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="red"))
    axf.text(ax_x + 0.03, page_h - mar_in * 0.3, "x", color="red", fontsize=6,
             ha="left", va="top")

    # print-accuracy check: a true 10 mm tick + the nominal geometry
    tick = 10.0 * inv / MM_PER_IN
    ty = mar_in * 0.35
    axf.plot([page_w - mar_in * 0.3 - tick, page_w - mar_in * 0.3], [ty, ty], color="0.4", lw=0.8)
    for tx in (page_w - mar_in * 0.3 - tick, page_w - mar_in * 0.3):
        axf.plot([tx, tx], [ty - 0.02, ty + 0.02], color="0.4", lw=0.8)
    label = (f"{name} · {args.family} · {args.size_mm:.0f}mm tags · "
             f"{args.gap_mm:.0f}mm gap · tick=10mm")
    if args.printer_scale != 1.0:
        label += f" · printer_scale={args.printer_scale}"
    axf.text(page_w / 2, ty * 0.45, label, ha="center", va="center", fontsize=4.5, color="0.55")

    pdf.savefig(fig)
    plt.close(fig)


def write_config(ids, name, args) -> str:
    """Emit the x-up TagBoardConfig. Positions are along body +x (up), strip
    centred on the origin; -0.3 mm z puts the tag face just proud of the
    backing (same convention as print_apriltags.py)."""
    size_m = args.size_mm / 1000.0
    pitch = size_m + args.gap_mm / 1000.0
    top = (len(ids) - 1) / 2.0 * pitch          # first id sits highest
    lines = []
    for k, tag_id in enumerate(ids):
        x = top - k * pitch
        lines.append(f'        "tag_{tag_id}": TagConfig(id={tag_id}, '
                     f'pos=[{x:.6f}, 0.000000, -0.000300], rot=TAG_ROT_X_UP),')
    body = "\n".join(lines)
    content = f'''"""Auto-generated by asset/tag_creation/print_tag_board.py — do not hand-edit.

Board "{name}": {len(ids)} x {args.family} tags, {args.size_mm:.0f} mm, {args.gap_mm:.0f} mm gaps,
{args.margin_mm:.0f} mm margin. Body frame is x-UP (see tagged_bodies/tag_board);
ids run top -> bottom, so tag_{ids[0]} is the TOP one.
"""
import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_bodies.tag_board import TAG_ROT_X_UP, TagBoardConfig
from tagged_rigid_body import TagConfig, verify_bodies

cfg = TagBoardConfig(
    name="{name}",
    n_tags={len(ids)},
    tag_size={size_m:.4f},
    gap_m={args.gap_mm / 1000.0:.4f},
    margin_m={args.margin_mm / 1000.0:.4f},
    tag_family="{args.family}",
    tags={{
{body}
    }},
)

if __name__ == "__main__":
    verify_bodies(cfg)
'''
    path = os.path.join(_ROOT, "tagged_bodies", "tag_board", f"{name}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def main() -> None:
    args = tyro.cli(Args)
    groups = [parse_group(s) for s in args.ids]
    names = args.names or [f"board_{i}" for i in range(len(groups))]
    if len(names) != len(groups):
        raise SystemExit(f"--names has {len(names)} entries for {len(groups)} boards")

    flat = [i for g in groups for i in g]
    if len(set(flat)) != len(flat):
        raise SystemExit(f"duplicate tag ids across boards: {sorted(flat)}")

    out = args.output or os.path.join(
        _HERE, f"tag_board_{args.family}_{int(args.size_mm)}mm_"
               f"ids_{min(flat)}-{max(flat)}.pdf")
    tg = make_generator(args.family)

    w = args.size_mm + 2 * args.margin_mm
    print(f"family {args.family} · tag {args.size_mm:.0f}mm · gap {args.gap_mm:.0f}mm · "
          f"margin {args.margin_mm:.0f}mm")
    if args.printer_scale != 1.0:
        print(f"printer_scale {args.printer_scale} — PDF pre-enlarged by "
              f"{1/args.printer_scale:.4f}x")
    print("-" * 68)
    with PdfPages(out) as pdf:
        for ids, name in zip(groups, names):
            h = (len(ids) * args.size_mm + (len(ids) - 1) * args.gap_mm
                 + 2 * args.margin_mm)
            board_page(pdf, tg, ids, name, args)
            xs = [(len(ids) - 1) / 2.0 * (args.size_mm + args.gap_mm) / 1000.0
                  - k * (args.size_mm + args.gap_mm) / 1000.0 for k in range(len(ids))]
            print(f"{name:<10} ids {ids}  board {w:.0f} x {h:.0f} mm  "
                  f"x(up) = {[round(v, 4) for v in xs]}")
            if args.write_configs:
                print(f"{'':<10} -> {write_config(ids, name, args)}")
    print("-" * 68)
    print(f"PDF: {out}   ({len(groups)} pages — print at 100%, one board per page)")


if __name__ == "__main__":
    main()
