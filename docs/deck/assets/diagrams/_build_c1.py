"""Build C1_phase1_experimental_design.svg + preview PNG.

Layout:
- Top zone: experimental grid 5×5 (donors × timepoints) with V|S|I|C 4-arm notation per cell
- "BTK+JAK CONFIRMED" callout
- ~500,000 cells total
- 3-card "WHY" row

Section accent: cyan (proprietary moat = our color)
Run: python3 docs/deck/assets/diagrams/_build_c1.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, FONT_MONO, START_X, W, H,
    svg_open, background, header, footer, render_png,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC QurieSeq Phase 1 experimental design")]
    background(parts)
    header(
        parts,
        appendix_id="C1",
        section="QURIESEQ PHASE 1",
        title="The Data Architected For The Model — Phase 1 lands Q3 2026",
        subtitle=(
            "5 donors × 5 timepoints × 4-arm perturbations · "
            "RNA + Protein per cell, donor-level ATAC at t=0 · ≈500,000 cells total"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # TOP CALLOUT: BTK+JAK CONFIRMED (above grid)
    # ====================================================================
    cc_y = 220
    parts.append(
        f'<rect x="{W // 2 - 280}" y="{cc_y}" width="560" height="40" rx="20" '
        f'fill="{OK_GREEN}" fill-opacity="0.16" stroke="{OK_GREEN}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{W // 2 - 264}" y="{cc_y + 26}" fill="{OK_GREEN}" font-family="{FONT}" '
        f'font-size="18" font-weight="700">✓</text>'
    )
    parts.append(
        f'<text x="{W // 2 - 236}" y="{cc_y + 26}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="15" font-weight="700" letter-spacing="1.5">'
        f'BTK + JAK combination CONFIRMED for Phase 1</text>'
    )

    # ====================================================================
    # EXPERIMENTAL GRID  (y ≈ 280..620)
    # 5 donors × 5 timepoints. Donor 1 in detail, donors 2-5 abbreviated.
    # ====================================================================
    G_Y = 280
    G_X = START_X + 160      # leave 160px for left donor labels
    G_W = W - G_X - START_X  # ≈1568
    TP_COL_W = G_W // 5      # 5 timepoint columns, ≈313 each
    ROW_H_FULL = 64           # detail row for Donor 1
    ROW_H_ABBR = 36           # abbreviated rows for Donors 2-5

    # Top column header strip
    parts.append(
        f'<text x="{G_X}" y="{G_Y - 14}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">TIMEPOINTS</text>'
    )
    timepoints = ["0 min", "5 min", "30 min", "60 min", "180 min"]
    tp_colors = [TEXT_MUTED, CYAN_HI, PURPLE, LAVENDER, TEXT_BODY]
    for i, (tp, c) in enumerate(zip(timepoints, tp_colors)):
        cx = G_X + i * TP_COL_W + TP_COL_W // 2
        parts.append(
            f'<text x="{cx}" y="{G_Y + 16}" fill="{c}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" text-anchor="middle">{tp}</text>'
        )

    # Donor 1 — full detail row
    D1_Y = G_Y + 32
    # Left label
    parts.append(
        f'<text x="{START_X}" y="{D1_Y + ROW_H_FULL // 2 - 6}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">DONOR 1</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="{D1_Y + ROW_H_FULL // 2 + 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" font-style="italic">(detail)</text>'
    )
    # 5 timepoint cells with V|S|I|C 4-arm pills
    for i in range(5):
        cx = G_X + i * TP_COL_W
        # Cell border
        parts.append(
            f'<rect x="{cx + 6}" y="{D1_Y}" width="{TP_COL_W - 12}" height="{ROW_H_FULL}" rx="8" '
            f'fill="{SURFACE}" stroke="{CYAN_HI}" stroke-width="1" stroke-opacity="0.4"/>'
        )
        # 4-arm pills: V S I C
        arms = [("V", TEXT_MUTED), ("S", OK_GREEN), ("I", PURPLE), ("C", CYAN_HI)]
        pill_w = (TP_COL_W - 12 - 16) // 4
        for j, (arm, ac) in enumerate(arms):
            px = cx + 6 + 8 + j * pill_w
            parts.append(
                f'<rect x="{px + 2}" y="{D1_Y + 16}" width="{pill_w - 4}" height="{ROW_H_FULL - 32}" rx="5" '
                f'fill="{ac}" fill-opacity="0.18" stroke="{ac}" stroke-width="1" stroke-opacity="0.6"/>'
            )
            parts.append(
                f'<text x="{px + pill_w // 2}" y="{D1_Y + ROW_H_FULL // 2 + 5}" fill="{ac}" '
                f'font-family="{FONT}" font-size="14" font-weight="700" text-anchor="middle">{arm}</text>'
            )

    # ATAC bar under Donor 1 spanning full width
    AT_Y = D1_Y + ROW_H_FULL + 6
    parts.append(
        f'<rect x="{G_X + 6}" y="{AT_Y}" width="{G_W - 12}" height="24" rx="6" '
        f'fill="{LAVENDER}" fill-opacity="0.10" stroke="{LAVENDER}" stroke-width="1" stroke-opacity="0.55"/>'
    )
    parts.append(
        f'<text x="{G_X + 18}" y="{AT_Y + 16}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2">◆ ATAC (chromatin signature) — donor-level static input at t=0 only</text>'
    )

    # Donor 2-5 abbreviated rows
    abbr_y0 = AT_Y + 28 + 12
    for k in range(4):
        ry = abbr_y0 + k * (ROW_H_ABBR + 4)
        parts.append(
            f'<text x="{START_X}" y="{ry + ROW_H_ABBR // 2 + 4}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700">DONOR {k + 2}</text>'
        )
        # Single bar with summary text instead of repeating pills
        parts.append(
            f'<rect x="{G_X + 6}" y="{ry}" width="{G_W - 12}" height="{ROW_H_ABBR}" rx="6" '
            f'fill="{SURFACE}" stroke="{DIVIDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{G_X + G_W // 2}" y="{ry + ROW_H_ABBR // 2 + 5}" fill="{TEXT_MUTED}" '
            f'font-family="{FONT_BODY}" font-size="13" font-weight="400" font-style="italic" '
            f'text-anchor="middle">same 5-timepoint × 4-arm grid · RNA + Protein per cell · ATAC at t=0</text>'
        )

    # Legend + total cells block (right-aligned, under grid)
    LG_Y = abbr_y0 + 4 * (ROW_H_ABBR + 4) + 8
    parts.append(
        f'<text x="{G_X}" y="{LG_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">LEGEND</text>'
    )
    legend = [
        ("V", "Vehicle baseline",  TEXT_MUTED),
        ("S", "Stim alone",        OK_GREEN),
        ("I", "Inhibitor alone",   PURPLE),
        ("C", "Stim + Inh combo",  CYAN_HI),
    ]
    lg_x = G_X + 100
    for letter, desc, color in legend:
        parts.append(
            f'<text x="{lg_x}" y="{LG_Y}" fill="{color}" font-family="{FONT}" '
            f'font-size="13" font-weight="700">{letter}</text>'
        )
        parts.append(
            f'<text x="{lg_x + 18}" y="{LG_Y}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">{desc}</text>'
        )
        lg_x += 250

    # Total cells callout (right)
    TC_X = W - START_X - 460
    parts.append(
        f'<rect x="{TC_X}" y="{LG_Y - 28}" width="460" height="46" rx="10" '
        f'fill="{CYAN}" fill-opacity="0.10" stroke="{CYAN_HI}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{TC_X + 18}" y="{LG_Y - 8}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">TOTAL</text>'
    )
    parts.append(
        f'<text x="{TC_X + 18}" y="{LG_Y + 12}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="20" font-weight="700">5 × 5 × 4 × ~5,000 cells = ~500,000 cells</text>'
    )

    # ====================================================================
    # BOTTOM ZONE: 3-card WHY row  (y ≈ 656..880)
    # ====================================================================
    WZ_Y = 656
    WZ_H = 244
    WZ_W = 560
    WZ_GAP = 24
    wz_x0 = (W - 3 * WZ_W - 2 * WZ_GAP) // 2  # 96

    parts.append(
        f'<text x="{wz_x0}" y="{WZ_Y - 12}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">WHY THIS DESIGN</text>'
    )
    parts.append(
        f'<line x1="{wz_x0 + 220}" y1="{WZ_Y - 18}" x2="{W - START_X}" y2="{WZ_Y - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    why_cards = [
        ("WHY 5 TIMEPOINTS",   CYAN_HI,
         ["Phospho-level signaling at 5 min (Phase 2 ready)",
          "Transcriptional onset at 30 min",
          "Stable phenotype at 180 min",
          "Non-uniform spacing maps Neural ODE (A4)"]),
        ("WHY 4-ARM PER PERT", LAVENDER,
         ["Vehicle = baseline",
          "Stim only = activation alone",
          "Inh only = inhibition alone",
          "Stim + Inh = synergy → direct match to A3"]),
        ("WHY 5 DONORS",       OK_GREEN,
         ["Donor-conditioned static context (ATAC)",
          "5 biological replicates of full 5×4 grid",
          "Phase 2 scales to 20 donors",
          "Cross-donor generalization eval ready"]),
    ]
    for i, (t, color, bullets) in enumerate(why_cards):
        wx = wz_x0 + i * (WZ_W + WZ_GAP)
        parts.append(
            f'<rect x="{wx}" y="{WZ_Y}" width="{WZ_W}" height="{WZ_H}" rx="14" '
            f'fill="{SURFACE}" stroke="{color}" stroke-width="1.5" stroke-opacity="0.55"/>'
        )
        parts.append(
            f'<text x="{wx + 20}" y="{WZ_Y + 32}" fill="{color}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" letter-spacing="2.5">{t}</text>'
        )
        for j, b in enumerate(bullets):
            parts.append(
                f'<text x="{wx + 20}" y="{WZ_Y + 70 + j * 36}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
                f'font-size="14" font-weight="400">'
                f'<tspan fill="{color}" font-weight="700">›</tspan>  {b}</text>'
            )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: QurieSeq Phase 1 design (Thiago confirmation, 2026-05-12) · "
            "Architecture spec v1.1 §3.2 (4-arm decomposition) + §4 (Neural ODE temporal) · "
            "Cell count estimate from Mimitou ASAP-seq protocol scaled to 5 donors × 5 timepoints"
        ),
        slide_handle="C1 / 12",
        handle_color=CYAN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "C1_phase1_experimental_design.svg"
    png_path = here / "C1_phase1_experimental_design_preview.png"
    svg_path.write_text(build_svg())
    print(f"wrote {svg_path}")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
