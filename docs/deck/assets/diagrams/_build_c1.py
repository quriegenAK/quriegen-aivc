"""Build C1_phase1_experimental_design.svg + preview PNG.

v2 layout (Step 3 — phospho-in-Phase-1 correction):
- Visual hero = MODALITY × TIMEPOINT MATRIX (replaces the v1 donor × timepoint
  × 4-arm grid). Phospho row lavender-emphasized as the proprietary modality
  no public dataset has on PBMCs.
- Center strip below matrix: "4 MODALITIES MEASURED · 5 TIMEPOINTS · 5 DONORS
  · ~125K CELLS · BTK+JAK COMBO INCLUDED" + "No public dataset has this
  combination" subtitle.
- Bottom: two side-by-side blocks (PERTURBATION CONDITIONS softened-framing +
  WET-LAB PARAMETERS).
- BTK+JAK CONFIRMED callout retained as visual anchor above matrix.

Section accent: cyan (proprietary moat = our color).

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
    check_no_text_collisions,
)

# Muted blue for ATAC — chromatin layer, slow-varying
ATAC_BLUE = "#5EAACC"


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC QuRIE-seq Phase 1 modality × timepoint matrix")]
    background(parts)
    header(
        parts,
        appendix_id="C1",
        section="QURIESEQ PHASE 1",
        title="QuRIE-seq Phase 1 — 5 donors × 5 timepoints × 4 modalities × BTK+JAK combo",
        subtitle=(
            "First proprietary perturbation-aware multi-omics dataset on primary human PBMCs · "
            "phospho is integral to QuRIE-seq · BTK+JAK combo confirmed · Q3 2026 delivery"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # TOP CALLOUT: BTK+JAK CONFIRMED (above matrix)
    # ====================================================================
    cc_y = 210
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
    # MODALITY × TIMEPOINT MATRIX (visual hero, y ≈ 270..560)
    # ====================================================================
    # 5 timepoint columns × (4 Phase 1 modality rows + 1 Phase 2 VDJ row)
    M_X = START_X + 180          # 180px for modality row labels
    M_Y = 280
    M_W = W - M_X - START_X       # ≈ 1548
    TP_COL_W = M_W // 5           # ≈ 309 each
    ROW_H = 48
    VDJ_ROW_H = 36
    HDR_H = 36                    # column header band

    # Section eyebrow
    parts.append(
        f'<text x="{START_X}" y="{M_Y - 14}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'PHASE 1 · MODALITY × TIMEPOINT MATRIX</text>'
    )
    parts.append(
        f'<line x1="{START_X + 380}" y1="{M_Y - 20}" x2="{W - START_X}" y2="{M_Y - 20}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Column headers (timepoints)
    timepoints = [("t=0", "baseline"), ("t=5", "early signal"),
                  ("t=30", "transcription"), ("t=60", "stable"),
                  ("t=180", "endpoint")]
    for i, (tp, sub) in enumerate(timepoints):
        cx = M_X + i * TP_COL_W + TP_COL_W // 2
        # Header background band
        parts.append(
            f'<rect x="{M_X + i * TP_COL_W + 2}" y="{M_Y}" width="{TP_COL_W - 4}" height="{HDR_H}" rx="6" '
            f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{M_Y + 18}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="16" font-weight="700" text-anchor="middle">{tp}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{M_Y + 32}" fill="{TEXT_DIM}" font-family="{FONT_BODY}" '
            f'font-size="10" font-style="italic" text-anchor="middle">{sub}</text>'
        )

    # Modality rows
    # (label, sublabel, color, presence_mask[5], emphasized=True for phospho)
    modality_rows = [
        ("RNA",     "gene expression",       CYAN,      [1, 1, 1, 1, 1], False),
        ("Protein", "surface markers",       OK_GREEN,  [1, 1, 1, 1, 1], False),
        ("Phospho", "kinase signaling · proprietary",   LAVENDER,  [1, 1, 1, 1, 1], True),
        ("ATAC",    "chromatin · slow-varying", ATAC_BLUE, [1, 0, 0, 0, 1], False),
    ]

    row_y = M_Y + HDR_H + 6
    for (label, sub, color, mask, emph) in modality_rows:
        # Row label (left)
        parts.append(
            f'<text x="{START_X}" y="{row_y + ROW_H // 2 - 2}" fill="{color}" font-family="{FONT}" '
            f'font-size="16" font-weight="700" letter-spacing="1.5">{label}</text>'
        )
        parts.append(
            f'<text x="{START_X}" y="{row_y + ROW_H // 2 + 16}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-style="italic">{sub}</text>'
        )

        # Cells
        for i in range(5):
            cx = M_X + i * TP_COL_W
            present = mask[i]
            # Cell background
            cell_fill = color if present else SURFACE_2
            cell_op = "0.22" if (present and emph) else ("0.14" if present else "0.5")
            stroke_op = "0.85" if (present and emph) else ("0.55" if present else "0.3")
            stroke_w = "1.6" if (present and emph) else "1"
            parts.append(
                f'<rect x="{cx + 4}" y="{row_y + 2}" width="{TP_COL_W - 8}" height="{ROW_H - 4}" rx="6" '
                f'fill="{cell_fill}" fill-opacity="{cell_op}" stroke="{color if present else DIVIDER}" '
                f'stroke-width="{stroke_w}" stroke-opacity="{stroke_op}"/>'
            )
            # Checkmark (or blank if not present)
            if present:
                check_color = color
                check_size = 26 if emph else 22
                parts.append(
                    f'<text x="{cx + TP_COL_W // 2}" y="{row_y + ROW_H // 2 + 9}" fill="{check_color}" '
                    f'font-family="{FONT}" font-size="{check_size}" font-weight="700" text-anchor="middle">✓</text>'
                )

        # Lavender glow ring around the Phospho row (visual emphasis)
        if emph:
            parts.append(
                f'<rect x="{M_X - 4}" y="{row_y - 2}" width="{M_W + 8}" height="{ROW_H + 4}" rx="9" '
                f'fill="none" stroke="{LAVENDER}" stroke-width="1" stroke-opacity="0.35" '
                f'stroke-dasharray="6 4"/>'
            )

        row_y += ROW_H + 6

    # VDJ row (Phase 2 — amber tinted, lower visual weight)
    vdj_y = row_y + 4
    parts.append(
        f'<text x="{START_X}" y="{vdj_y + VDJ_ROW_H // 2 - 2}" fill="{WARN_AMBER}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="1.5">VDJ</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="{vdj_y + VDJ_ROW_H // 2 + 14}" fill="{TEXT_DIM}" font-family="{FONT_BODY}" '
        f'font-size="10" font-style="italic">5th modality</text>'
    )
    # Single Phase 2 strip across all timepoints
    parts.append(
        f'<rect x="{M_X + 4}" y="{vdj_y + 2}" width="{M_W - 8}" height="{VDJ_ROW_H - 4}" rx="6" '
        f'fill="{WARN_AMBER}" fill-opacity="0.08" stroke="{WARN_AMBER}" stroke-width="1" '
        f'stroke-opacity="0.5" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{M_X + M_W // 2}" y="{vdj_y + VDJ_ROW_H // 2 + 5}" fill="{WARN_AMBER}" '
        f'font-family="{FONT_BODY}" font-size="13" font-weight="600" font-style="italic" '
        f'text-anchor="middle">PHASE 2 (2027) — adds clonal repertoire as 5th modality</text>'
    )

    # ====================================================================
    # CENTER STRIP — "No public dataset has this combination"
    # ====================================================================
    strip_y = vdj_y + VDJ_ROW_H + 22
    parts.append(
        f'<text x="{W // 2}" y="{strip_y}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="15" font-weight="700" letter-spacing="2" text-anchor="middle">'
        f'4 MODALITIES MEASURED · 5 TIMEPOINTS · 5 DONORS · ~125K CELLS · BTK+JAK COMBO INCLUDED</text>'
    )
    parts.append(
        f'<text x="{W // 2}" y="{strip_y + 22}" fill="{LAVENDER}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="600" font-style="italic" text-anchor="middle">'
        f'└── No public dataset combines these modalities on primary PBMCs with perturbation-aware design ──┘</text>'
    )

    # ====================================================================
    # BOTTOM ZONE: two side-by-side blocks (perturbation + wet-lab)
    # ====================================================================
    BZ_Y = strip_y + 48
    BZ_H = 240
    BZ_GAP = 32
    BZ_W = (W - 2 * START_X - BZ_GAP) // 2

    # LEFT — PERTURBATION CONDITIONS (softened framing)
    LB_X = START_X
    parts.append(
        f'<rect x="{LB_X}" y="{BZ_Y}" width="{BZ_W}" height="{BZ_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{CYAN_HI}" stroke-width="1.3" stroke-opacity="0.55"/>'
    )
    parts.append(
        f'<text x="{LB_X + 22}" y="{BZ_Y + 28}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="2.5">PERTURBATION CONDITIONS</text>'
    )
    pert_items = [
        ("Vehicle control", "baseline"),
        ("Stimulus condition", "panel under final wet-lab spec review"),
        ("Inhibitor singles", "BTK, JAK CONFIRMED · additional panel TBC"),
        ("Inhibitor combinations", "BTK+JAK CONFIRMED — Stage 3b headline demo"),
        ("Combo sizing", "supports pre-registered compositional eval"),
    ]
    for i, (lhs, rhs) in enumerate(pert_items):
        py = BZ_Y + 58 + i * 32
        parts.append(
            f'<text x="{LB_X + 22}" y="{py}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">'
            f'<tspan fill="{CYAN_HI}" font-weight="700">›</tspan>  '
            f'<tspan font-weight="700">{lhs}</tspan> — {rhs}</text>'
        )

    # RIGHT — WET-LAB PARAMETERS
    RB_X = LB_X + BZ_W + BZ_GAP
    parts.append(
        f'<rect x="{RB_X}" y="{BZ_Y}" width="{BZ_W}" height="{BZ_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{LAVENDER}" stroke-width="1.3" stroke-opacity="0.55"/>'
    )
    parts.append(
        f'<text x="{RB_X + 22}" y="{BZ_Y + 28}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="2.5">WET-LAB PARAMETERS</text>'
    )
    wl_items = [
        ("5 donors", "Sanquin, blood-type-only metadata"),
        ("All major PBMC lineages", "B / T / NK / monocyte / DC"),
        ("~5k cells / donor / timepoint", "~125k cells total"),
        ("QuRIE-seq protocol family", "proprietary multi-omics assay"),
        ("Q3 2026 delivery target", "Phase 2 (2027) scales to 20 donors + VDJ"),
    ]
    for i, (lhs, rhs) in enumerate(wl_items):
        py = BZ_Y + 58 + i * 32
        parts.append(
            f'<text x="{RB_X + 22}" y="{py}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">'
            f'<tspan fill="{LAVENDER}" font-weight="700">›</tspan>  '
            f'<tspan font-weight="700">{lhs}</tspan> — {rhs}</text>'
        )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: QurieSeq Phase 1 design (Thiago confirmation, 2026-05-12) · "
            "phase1_modality_correction_2026_05_17.md (canonical) · "
            "Architecture spec v1.1 §3.2 (4-arm decomposition) + §4 (Neural ODE temporal)"
        ),
        slide_handle="C1 / 14",
        handle_color=CYAN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "C1_phase1_experimental_design.svg"
    png_path = here / "C1_phase1_experimental_design_preview.png"
    svg = build_svg()
    # Collision-guard smoke. Filter footer pagination + source citation
    # (heuristic over-estimates text-anchor=end width vs source italics).
    cols = check_no_text_collisions(svg, min_gap=2)
    blocking = [
        c for c in cols
        if "C1 / 14" not in (c[0], c[1])
        and not c[0].startswith("Source:") and not c[1].startswith("Source:")
    ]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"C1 collision-guard FAIL:\n{msg}")
    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
