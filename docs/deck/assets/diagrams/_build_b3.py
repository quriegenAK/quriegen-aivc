"""Build B3_mechanism_pre_demo.svg + preview PNG.

Layout:
- Top zone: Two parallel columns (Public-Data Substitute cyan / QurieSeq Phase 1 amber)
  with HELD OUT callouts in each
- Middle zone: 3-card substitute justification (Architecture / Data / Mechanism)
- Bottom zone: Clinical grounding footer (NCT02912754, Maddocks 2016, pJAK1)

Section accent: cyan + amber (for Q3 2026 framing)
Run: python3 docs/deck/assets/diagrams/_build_b3.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER, DANGER_RED,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, TEXT_DISABLED, DIVIDER,
    FONT, FONT_BODY, FONT_MATH, START_X, W, H,
    svg_open, background, header, footer, arrow, render_png,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC mechanism pre-demo: public-data substitute validates synergy mechanism")]
    background(parts)
    header(
        parts,
        appendix_id="B3",
        section="VALIDATION EVIDENCE",
        title="Synergy Mechanism — Public-Data Substitute → BTK+JAK",
        subtitle=(
            "Mimitou CRISPR double-KO held out today is the architectural substitute for QurieSeq BTK+JAK held out in Q3 2026 · "
            "same mechanism, same architecture, different perturbation"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # TOP ZONE: Two parallel columns (Public-Data | QurieSeq)
    # x = 96..1824 split into two columns ~832 wide each, gap 64
    # ====================================================================
    COL_Y = 232
    COL_H = 416
    COL_GAP = 64
    COL_W = (W - 2 * START_X - COL_GAP) // 2   # 832

    columns = [
        {
            "x": START_X,
            "accent": CYAN_HI,
            "eyebrow": "PUBLIC-DATA SUBSTITUTE",
            "time_tag": "TODAY",
            "time_color": OK_GREEN,
            "title": "Mimitou CRISPR",
            "subtitle": "ASAP-seq CD4 T-cell sub-study",
            "training_label": "TRAINING ARMS (SEEN)",
            "training": [
                "CD3E single KO",
                "CD4 single KO",
                "ZAP70 single KO",
                "NFKB2 single KO",
                "NTC",
            ],
            "heldout_label": "HELD OUT",
            "heldout": "CD3E + CD4 double KO",
            "heldout_note": "architectural substitute for BTK+JAK",
        },
        {
            "x": START_X + COL_W + COL_GAP,
            "accent": WARN_AMBER,
            "eyebrow": "QURIESEQ PHASE 1",
            "time_tag": "Q3 2026",
            "time_color": WARN_AMBER,
            "title": "BTK + JAK demo",
            "subtitle": "QurieSeq proprietary PBMC data",
            "training_label": "TRAINING ARMS (WILL BE SEEN)",
            "training": [
                "BTK inhibitor alone",
                "JAK inhibitor alone",
                "Other inhibitor singles",
                "All stimuli × vehicle",
                "All 4-arm controls",
            ],
            "heldout_label": "HELD OUT",
            "heldout": "BTK + JAK combination",
            "heldout_note": "the clinical demo target",
        },
    ]

    for col in columns:
        cx = col["x"]
        # Column card
        parts.append(
            f'<rect x="{cx}" y="{COL_Y}" width="{COL_W}" height="{COL_H}" rx="14" '
            f'fill="{SURFACE}" stroke="{col["accent"]}" stroke-width="1.5" stroke-opacity="0.65"/>'
        )
        # Eyebrow
        parts.append(
            f'<text x="{cx + 24}" y="{COL_Y + 36}" fill="{col["accent"]}" font-family="{FONT}" '
            f'font-size="12" font-weight="700" letter-spacing="3">{col["eyebrow"]}</text>'
        )
        # Time tag (top-right of column)
        tw = 110
        parts.append(
            f'<rect x="{cx + COL_W - tw - 24}" y="{COL_Y + 18}" width="{tw}" height="24" rx="12" '
            f'fill="{col["time_color"]}" fill-opacity="0.18" stroke="{col["time_color"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{cx + COL_W - tw / 2 - 24}" y="{COL_Y + 34}" fill="{col["time_color"]}" '
            f'font-family="{FONT}" font-size="11" font-weight="700" letter-spacing="2" '
            f'text-anchor="middle">{col["time_tag"]}</text>'
        )
        # Title + subtitle
        parts.append(
            f'<text x="{cx + 24}" y="{COL_Y + 76}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="26" font-weight="700">{col["title"]}</text>'
        )
        parts.append(
            f'<text x="{cx + 24}" y="{COL_Y + 100}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400" font-style="italic">{col["subtitle"]}</text>'
        )
        # Training arms label
        ta_y = COL_Y + 138
        parts.append(
            f'<text x="{cx + 24}" y="{ta_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2.5">{col["training_label"]}</text>'
        )
        for j, item in enumerate(col["training"]):
            parts.append(
                f'<text x="{cx + 24}" y="{ta_y + 24 + j * 22}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
                f'font-size="14" font-weight="400">'
                f'<tspan fill="{col["accent"]}" font-weight="700">›</tspan>  {item}</text>'
            )
        # Divider
        div_y = ta_y + 24 + len(col["training"]) * 22 + 12
        parts.append(
            f'<line x1="{cx + 24}" y1="{div_y}" x2="{cx + COL_W - 24}" y2="{div_y}" '
            f'stroke="{DIVIDER}" stroke-width="1"/>'
        )
        # HELD OUT callout
        ho_y = div_y + 28
        parts.append(
            f'<text x="{cx + 24}" y="{ho_y}" fill="{CYAN_HI}" font-family="{FONT}" '
            f'font-size="11" font-weight="700" letter-spacing="2.5">◆ {col["heldout_label"]}</text>'
        )
        parts.append(
            f'<text x="{cx + 24}" y="{ho_y + 26}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="20" font-weight="700">{col["heldout"]}</text>'
        )
        parts.append(
            f'<text x="{cx + 24}" y="{ho_y + 46}" fill="{TEXT_DIM}" font-family="{FONT_BODY}" '
            f'font-size="12" font-weight="400" font-style="italic">{col["heldout_note"]}</text>'
        )
        # PREDICT arrow callout at bottom-right of column
        pa_x = cx + COL_W - 200
        pa_y = ho_y + 24
        parts.append(
            f'<text x="{pa_x}" y="{pa_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2.5">↓ PREDICT</text>'
        )
        parts.append(
            f'<text x="{pa_x}" y="{pa_y + 28}" fill="{col["accent"]}" font-family="{FONT}" '
            f'font-size="22" font-weight="700">zero-shot</text>'
        )

    # Cross-column bridge arrow: connect HELD OUT of column 1 to HELD OUT of column 2
    cb_y = COL_Y + COL_H // 2 + 100
    arrow(parts,
          columns[0]["x"] + COL_W - 4, cb_y,
          columns[1]["x"] + 4, cb_y,
          color=CYAN, opacity=0.6, width=2)
    parts.append(
        f'<text x="{W // 2}" y="{cb_y - 12}" fill="{TEXT_BODY}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5" text-anchor="middle">'
        f'SAME MECHANISM</text>'
    )

    # ====================================================================
    # MIDDLE ZONE: 3 substitute-justification cards
    # x = 96..1824, y = 676..836
    # ====================================================================
    MZ_Y = 676
    MZ_H = 156
    SUB_W = 560
    SUB_GAP = 24
    sub_x0 = (W - 3 * SUB_W - 2 * SUB_GAP) // 2  # 96

    parts.append(
        f'<text x="{sub_x0}" y="{MZ_Y - 12}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'WHY THIS IS A VALID SUBSTITUTE</text>'
    )
    parts.append(
        f'<line x1="{sub_x0 + 360}" y1="{MZ_Y - 18}" x2="{W - START_X}" y2="{MZ_Y - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    subs = [
        ("ARCHITECTURE",
         ["Same 4-arm decomposed readout",
          "Same zero-arm L2 constraint (λ = 1.0)",
          "Same synergy head trained only on the residual signal"]),
        ("DATA",
         ["Mimitou double-KO has 74 cells post-split",
          "Bootstrap CI from pre-registered eval",
          "Target ≥0.70 zero-shot synergy accuracy"]),
        ("MECHANISM",
         ["Each single perturbation alters TCR signaling",
          "Double KO yields non-additive phenotype",
          "Exactly what the synergy head must learn"]),
    ]
    for i, (sub_title, bullets) in enumerate(subs):
        sx = sub_x0 + i * (SUB_W + SUB_GAP)
        parts.append(
            f'<rect x="{sx}" y="{MZ_Y}" width="{SUB_W}" height="{MZ_H}" rx="12" '
            f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="1.2" stroke-opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{sx + 20}" y="{MZ_Y + 28}" fill="{CYAN}" font-family="{FONT}" '
            f'font-size="11" font-weight="700" letter-spacing="2.5">{sub_title}</text>'
        )
        for j, b in enumerate(bullets):
            parts.append(
                f'<text x="{sx + 20}" y="{MZ_Y + 56 + j * 28}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
                f'font-size="13" font-weight="400">'
                f'<tspan fill="{CYAN}" font-weight="700">›</tspan>  {b}</text>'
            )

    # ====================================================================
    # BOTTOM ZONE: Clinical grounding footer  (above main footer)
    # y ≈ 850..908
    # ====================================================================
    CG_Y = 868
    parts.append(
        f'<text x="{START_X}" y="{CG_Y}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">WHY THIS COMBINATION MATTERS</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="{CG_Y + 22}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="400">'
        f'<tspan fill="{LAVENDER}" font-weight="700">›</tspan>  '
        f'<tspan font-weight="700">Ibrutinib</tspan> (BTK inhibitor) + '
        f'<tspan font-weight="700">Ruxolitinib</tspan> (JAK1/2 inhibitor) — '
        f'<tspan font-style="italic">CLL Phase Ib/II trial</tspan> '
        f'<tspan fill="{CYAN}" font-weight="700">NCT02912754</tspan>'
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'<tspan font-weight="700">Maddocks 2016</tspan>, Blood '
        f'<tspan fill="{TEXT_DIM}">(PMID 26819050)</tspan>'
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'Thiago wet-lab IP: '
        f'<tspan fill="{CYAN_HI}" font-weight="700">pJAK1</tspan> unexpectedly active in BCR pathway'
        f'</text>'
    )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Stage 3 Part 1 verdict (2026-05-06, 0.68 CD3E×CD4 zero-shot) · "
            "QurieSeq Phase 1 design (Thiago confirmation, 2026-05-12) · "
            "Architecture spec v1.1 §5.1 (≥0.70 pre-registered threshold) · "
            "Maddocks et al. Blood 2016, PMID 26819050, NCT02912754"
        ),
        slide_handle="B3 / 12",
        handle_color=CYAN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "B3_mechanism_pre_demo.svg"
    png_path = here / "B3_mechanism_pre_demo_preview.png"
    svg_path.write_text(build_svg())
    print(f"wrote {svg_path}")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
