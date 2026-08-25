#!/usr/bin/env python3
"""
Generate a PDF with multiple AprilTags precisely sized on US Letter or Legal paper.
Includes support for printer scaling (e.g. 93.88% shrinking by "Fit to Printable Area")
via the --printer-scale argument.

Example:
    python asset/tag_creation/print_apriltags.py --ids 401-420 --size-mm 180 --paper letter 
    for school printer use --printer-scale 0.9388
    # Create a single TaggedRigidBodyConfig combining 5 sheets horizontally:
    python asset/tag_creation/print_apriltags.py --ids 401-420 --size-mm 180 --paper letter --wall-name wall_0
"""

import numpy as np
import tyro
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from moms_apriltag import TagGenerator2, TagGenerator3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

@dataclass
class Args:
    """Generate AprilTags on a PDF with precise sizing
    """


    ids: List[str]
    """List of IDs or ranges (e.g. 0-5, 10, 15-20)"""

    size_mm: float = 100.0
    """Size of the tag (black border to black border) in millimeters"""

    family: str = "tag36h11"
    """AprilTag family (tag36h11, tagStandard41h12, etc.)"""

    output: Optional[str] = None
    """Output PDF filename. Defaults to asset/tag_creation/{family}_{size_mm}mm_ids_{range}.pdf"""

    paper: str = "letter"
    """Paper size: letter (8.5x11), legal (8.5x14), a4 (210x297mm), or custom 'WxH' in mm (e.g. '216x279')"""

    margin_mm: float = 5.0
    """Page margin in millimeters (default 0.0mm)"""

    spacing_mm: float = 5.0
    """Minimum spacing between tags in millimeters (default 5.0mm)"""


    label_size: float = 10.0
    """Font size for ID labels"""

    show_labels: bool = True
    """Whether to print the ID under each tag"""

    show_cutting_lines: bool = False
    """Whether to draw a thin border around each tag for easier cutting"""

    body_name: Optional[str] = None
    """Name for the generated TaggedRigidBodyConfig (e.g. 'sheet_0'). If set, creates a .py config file."""

    wall_name: Optional[str] = None
    """If set, creates a single TaggedRigidBodyConfig combining all pages horizontally."""

    printer_scale: float = 1.0
    """Scale factor applied by the printer (e.g., 0.9388 for 'Fit to Printable Area'). Set < 1.0 to enlarge PDF so physical tags match size_mm."""


def parse_ids(id_strings: List[str]) -> List[int]:
    """Parse list of strings containing individual IDs or ranges."""
    ids = []
    for s in id_strings:
        s = s.strip(",") # Handle comma separated IDs in a single string if needed
        if '-' in s:
            try:
                start, end = map(int, s.split('-'))
                ids.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Could not parse range '{s}'. Skipping.")
        else:
            try:
                ids.append(int(s))
            except ValueError:
                print(f"Warning: Could not parse ID '{s}'. Skipping.")
    return sorted(list(set(ids)))


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def get_paper_size(paper: str) -> Tuple[float, float]:
    """Returns (width, height) in inches."""
    paper = paper.lower()
    if paper == "legal":
        return 8.5, 14.0
    elif paper == "letter":
        return 8.5, 11.0
    elif paper == "a4":
        return 210 / 25.4, 297 / 25.4
    elif "x" in paper:
        try:
            w_mm, h_mm = map(float, paper.split("x"))
            return mm_to_inch(w_mm), mm_to_inch(h_mm)
        except ValueError:
            pass
            
    print(f"Unknown paper size '{paper}', defaulting to US Letter.")
    return 8.5, 11.0


def main():
    args = tyro.cli(Args)

    # Paper size in inches
    PAGE_WIDTH, PAGE_HEIGHT = get_paper_size(args.paper)

    ids = parse_ids(args.ids)
    if not ids:
        print("Error: No valid IDs provided.")
        return

    # Determine output path and body name
    if len(ids) > 1:
        id_range = f"{ids[0]}-{ids[-1]}"
    else:
        id_range = f"{ids[0]}"
    
    # Informative file stem always encodes family, size, and IDs
    file_stem = f"{args.family}_{int(args.size_mm)}mm_ids_{id_range}"
    # cfg_name is what goes inside the PaperSheetConfig; defaults to file_stem
    cfg_name = args.body_name if args.body_name else file_stem

    """
    Combined Wall Configuration Rationale:
    Purpose: To generate a single TaggedRigidBodyConfig representing multiple physical sheets 
    aligned horizontally (edge-to-edge), e.g., for wall-mounted tags.
    
    Design Decisions:
    - Centering: The wall is centered at the origin (geometric center of N sheets).
      This ensures the standard PaperSheetConfig box mesh aligns perfectly with tag coordinates.
    - Reuse: Reuses PaperSheetConfig because a row of N papers is functionally 
      equivalent to a single wide sheet of paper.
    - Coordinate Flow: Sheets are aligned left-to-right (increasing page_idx moves sheets +x).

    Assumptions:
    - Papers are aligned perfectly edge-to-edge without physical gaps.
    - Printer scaling is centered relative to the physical page.
    """

    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, f"{file_stem}.pdf")
    else:
        output_path = args.output
        file_stem = os.path.splitext(os.path.basename(output_path))[0]
    
    # Initialize tag generator
    # Try TagGenerator2 first (most common families), then TagGenerator3
    try:
        tg = TagGenerator2(args.family)
    except:
        try:
            tg = TagGenerator3(args.family)
        except Exception as e:
            print(f"Error: Could not initialize tag family {args.family}: {e}")
            return

    # Save target sizes to use for the Python config output
    target_size_mm = args.size_mm
    target_margin_mm = args.margin_mm
    target_spacing_mm = args.spacing_mm
    target_label_height_mm = 5.0

    # Pre-scale the layout sizes. By dividing by a scale < 1.0 (e.g. 0.9388),
    # we draw the tags *larger* in the PDF. When the printer subsequently 
    # shrinks the page by 0.9388, the physical tag on paper will be exactly target_size_mm.
    pdf_size_mm = target_size_mm / args.printer_scale
    pdf_margin_mm = target_margin_mm / args.printer_scale
    pdf_spacing_mm = target_spacing_mm / args.printer_scale
    pdf_label_height_mm = target_label_height_mm / args.printer_scale

    # Dimensions for PDF layout in inches (enlarged if printer_scale < 1.0)
    tag_size_in = mm_to_inch(pdf_size_mm)
    margin_in = mm_to_inch(pdf_margin_mm)
    spacing_in = mm_to_inch(pdf_spacing_mm)
    
    label_height_in = mm_to_inch(pdf_label_height_mm) if args.show_labels else 0.0
    
    usable_w = PAGE_WIDTH - 2 * margin_in
    usable_h = PAGE_HEIGHT - 2 * margin_in

    # gap_in = visual inter-image gap, equal in both directions.
    # The label (height=label_height_in) sits inside the vertical gap just below the tag.
    # Constraint: gap >= label_height so the label doesn't overlap the next tag.
    min_gap_in = max(spacing_in, label_height_in)

    # Calculate how many columns and rows fit
    cols = int((usable_w + min_gap_in) // (tag_size_in + min_gap_in))
    # vertical step = tag + gap (same as horizontal); account for label on last row
    rows = int((usable_h - label_height_in + min_gap_in) // (tag_size_in + min_gap_in))

    if cols == 0 or rows == 0:
        print(f"Error: Tag size ({pdf_size_mm:.1f}mm) and margins are too large for {args.paper} paper.")
        return

    # Max visual inter-image gap fitting both directions
    max_gap_w_mm = 1000.0
    if cols > 1:
        max_gap_w_mm = np.floor((usable_w - cols * tag_size_in) / (cols - 1) * 25.4)

    max_gap_h_mm = 1000.0
    if rows > 1:
        # block_h = rows*tag + (rows-1)*gap + label  →  gap = (usable_h - rows*tag - label) / (rows-1)
        max_gap_h_mm = np.floor((usable_h - rows * tag_size_in - label_height_in) / (rows - 1) * 25.4)

    gap_mm = max(max(pdf_spacing_mm, label_height_in * 25.4), min(max_gap_w_mm, max_gap_h_mm))
    gap_in = gap_mm / 25.4

    # Total block size (step = tag + gap in both directions; label sits inside vertical gap)
    block_w = cols * tag_size_in + (cols - 1) * gap_in
    block_h = rows * tag_size_in + label_height_in + (rows - 1) * gap_in
    
    # Offset to center the block on page
    x_offset = (PAGE_WIDTH - block_w) / 2
    y_offset = (PAGE_HEIGHT - block_h) / 2

    tags_per_page = cols * rows
    total_pages = (len(ids) + tags_per_page - 1) // tags_per_page
    all_tags_config = {}

    print(f"Generating tags for family: {args.family}")
    print(f"Paper: {args.paper} ({PAGE_WIDTH}\" x {PAGE_HEIGHT}\")")
    if args.printer_scale != 1.0:
        print(f"Tag size: {target_size_mm}mm (Drawn in PDF as {pdf_size_mm:.1f}mm, printer_scale={args.printer_scale})")
    else:
        print(f"Tag size: {target_size_mm}mm")
    print(f"Layout: {cols} columns x {rows} rows ({tags_per_page} tags per page)")
    print(f"Uniform Gap in PDF: {gap_mm:.1f}mm")
    print(f"Total tags: {len(ids)}")
    print("-" * 40)
    print("Tag coordinates relative to PAGE CENTER (meters):")

    with PdfPages(output_path) as pdf:
        for p_idx in range(0, len(ids), tags_per_page):
            page_ids = ids[p_idx : p_idx + tags_per_page]
            
            # Create a figure with paper size
            fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
            
            page_tags_config = {}
            page_idx = p_idx // tags_per_page

            # For the combined wall config, offset this page horizontally
            wall_offset_x_m = 0.0
            if args.wall_name:
                wall_offset_x_m = (page_idx - (total_pages - 1) / 2.0) * PAGE_WIDTH * 0.0254

            for idx, tag_id in enumerate(page_ids):
                r = idx // cols
                c = idx % cols
                
                # Top-left of the tag in this grid
                x_start = x_offset + c * (tag_size_in + gap_in)
                y_top = PAGE_HEIGHT - (y_offset + r * (tag_size_in + gap_in))
                y_start = y_top - tag_size_in
                
                # Tag center relative to page center in the PDF.
                # Body frame aligns with tag frame: x=right, y=down, z=into-paper.
                # Mapping from old (x=right, y=up, z=out): new_x=x, new_y=-y, new_z=-z
                cx_pdf_m = (x_start + tag_size_in/2 - PAGE_WIDTH/2) * 0.0254
                cy_pdf_m = (y_start + tag_size_in/2 - PAGE_HEIGHT/2) * 0.0254

                # The printer shrinks the printed content onto the physical paper.
                # Thus, the physical distance from the center is shorter by printer_scale.
                cx_physical_m = cx_pdf_m * args.printer_scale
                cy_physical_m = cy_pdf_m * args.printer_scale

                # 0.3mm offset: tag sits slightly in front of paper (-z in new frame)
                pos = [cx_physical_m, -cy_physical_m, -0.0003]
                print(f"  ID {tag_id:3d}: pos={[round(v,4) for v in pos]}")

                # Store for python config
                tag_cfg_str = f"TagConfig(id={tag_id}, pos=[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}])"
                page_tags_config[f"tag_{tag_id}"] = tag_cfg_str
                
                if args.wall_name:
                    wall_pos = [pos[0] + wall_offset_x_m, pos[1], pos[2]]
                    all_tags_config[f"tag_{tag_id}"] = f"TagConfig(id={tag_id}, pos=[{wall_pos[0]:.6f}, {wall_pos[1]:.6f}, {wall_pos[2]:.6f}])"

                # Tag generation
                tag_img = tg.generate(tag_id)
                
                # Add axes for the tag image
                ax_pos = [x_start / PAGE_WIDTH, y_start / PAGE_HEIGHT, 
                          tag_size_in / PAGE_WIDTH, tag_size_in / PAGE_HEIGHT]
                ax = fig.add_axes(ax_pos)
                ax.imshow(tag_img, cmap='gray', interpolation='nearest')
                ax.axis('off')

                # Add cutting lines if requested
                if args.show_cutting_lines:
                    rect = patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor='gray', facecolor='none', transform=ax.transAxes)
                    ax.add_patch(rect)
                
                # Add label if requested
                if args.show_labels:
                    fig.text(
                        (x_start + tag_size_in / 2) / PAGE_WIDTH,
                        (y_start - label_height_in/2) / PAGE_HEIGHT,
                        f"ID: {tag_id}",
                        ha='center', va='center', fontsize=args.label_size,
                        color='0.8'
                    )
            
            # Page name (computed here so it can be used in footer and config)
            page_suffix = f"_p{p_idx//tags_per_page}" if len(ids) > tags_per_page else ""
            final_file_stem = f"{file_stem}{page_suffix}"
            final_cfg_name = f"{cfg_name}{page_suffix}"

            # Footer: page name centered at bottom
            footer_text = final_file_stem
            if args.printer_scale != 1.0:
                footer_text += f" | printer_scale={args.printer_scale}"
            
            # Use a small fixed offset (5mm) from bottom to ensure visibility
            footer_y = max(5.0 / 25.4, margin_in / 2)
            fig.text(0.5, footer_y / PAGE_HEIGHT, footer_text,
                     ha='center', va='center', fontsize=7, color='0.5')

            # Draw coordinate frame indicator in bottom-left corner
            arrow_len_in = mm_to_inch(8)
            ox = margin_in + arrow_len_in * 0.15
            oy = margin_in + arrow_len_in  # origin; y arrow goes downward
            ax_fr = fig.add_axes([0, 0, 1, 1])
            ax_fr.set_xlim(0, PAGE_WIDTH)
            ax_fr.set_ylim(0, PAGE_HEIGHT)
            ax_fr.axis('off')
            ap = dict(arrowstyle='->', lw=1.2)
            fs = 7
            # x → right (red)
            ax_fr.annotate('', xy=(ox + arrow_len_in, oy), xytext=(ox, oy),
                           arrowprops={**ap, 'color': 'red'})
            ax_fr.text(ox + arrow_len_in + 0.03, oy, 'x', color='red',
                       fontsize=fs, va='center', ha='left')
            # y ↓ down (green)
            ax_fr.annotate('', xy=(ox, oy - arrow_len_in), xytext=(ox, oy),
                           arrowprops={**ap, 'color': 'green'})
            ax_fr.text(ox, oy - arrow_len_in - 0.03, 'y', color='green',
                       fontsize=fs, va='top', ha='center')
            # z ⊗ into paper (blue) at origin
            r = arrow_len_in * 0.13
            ax_fr.add_patch(plt.Circle((ox, oy), r, color='blue', fill=False, lw=1.2))
            d = r * 0.65
            ax_fr.plot([ox - d, ox + d], [oy + d, oy - d], color='blue', lw=1.2)
            ax_fr.plot([ox - d, ox + d], [oy - d, oy + d], color='blue', lw=1.2)
            ax_fr.text(ox + r + 0.02, oy + r + 0.01, 'z (into paper)',
                       color='blue', fontsize=fs - 1, va='bottom', ha='left')

            pdf.savefig(fig)
            plt.close(fig)

            # Generate python config for this page
            if not args.wall_name:
                config_content = f"""\"\"\"Auto-generated config for {final_file_stem}\"\"\"
import os, sys
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TagConfig, verify_bodies
from tagged_bodies.paper_sheet import PaperSheetConfig

cfg = PaperSheetConfig(
    name="{final_cfg_name}",
    width_m={PAGE_WIDTH * 0.0254:.4f},
    height_m={PAGE_HEIGHT * 0.0254:.4f},
    tag_size={target_size_mm / 1000.0:.4f},
    tags={{
"""
                for label, tag_str in page_tags_config.items():
                    config_content += f"        \"{label}\": {tag_str},\n"
                
                config_content += """    }
)

if __name__ == "__main__":
    verify_bodies(cfg)
"""
                # Determine config path relative to script
                root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                config_path = os.path.join(root_dir, "tagged_bodies", "paper_sheet", f"{final_file_stem}.py")
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w") as f:
                    f.write(config_content)
                print(f"Generated TaggedRigidBodyConfig: {config_path}")
    print("-" * 40)

    # Generate combined wall config if requested
    if args.wall_name:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        wall_config_path = os.path.join(root_dir, "tagged_bodies", "paper_sheet", f"{args.wall_name}.py")
        
        wall_width_m = total_pages * PAGE_WIDTH * 0.0254
        
        wall_content = f"""\"\"\"Auto-generated combined wall config for {args.wall_name}\"\"\"
import os, sys
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_rigid_body import TagConfig, verify_bodies
from tagged_bodies.paper_sheet import PaperSheetConfig

cfg = PaperSheetConfig(
    name="{args.wall_name}",
    width_m={wall_width_m:.4f},
    height_m={PAGE_HEIGHT * 0.0254:.4f},
    tag_size={target_size_mm / 1000.0:.4f},
    tags={{
"""
        for label, tag_str in all_tags_config.items():
            wall_content += f"        \"{label}\": {tag_str},\n"
        
        wall_content += """    }
)

if __name__ == "__main__":
    verify_bodies(cfg)
"""
        with open(wall_config_path, "w") as f:
            f.write(wall_content)
        print(f"Generated Combined TaggedRigidBodyConfig: {wall_config_path}")

    print(f"Successfully saved {len(ids)} tags to {output_path}")


if __name__ == "__main__":
    main()
