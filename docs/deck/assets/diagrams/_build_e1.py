"""Build E1_five_year_trajectory.svg + preview PNG.

Layout:
- Top zone: 4-phase progression cards (2026 / 2027 / 2028 / 2029-2031)
  with decreasing visual confidence: filled → outlined as horizon recedes
- Bottom zone: 3-card compounding loops (DATA / MODEL / CLINICAL INFRA)

Section accent: white/pale (the horizon)
NO IPO / Series A / exit per Ash strategic direction.
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, TEXT_DISABLED, DIVIDER,
    FONT, FONT_BODY, START_X, W, H,
    svg_open, background, header, footer, render_png,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC E1 five-year trajectory 2026 to 2031")]
    background(parts)
    header(
        parts,
        appendix_id="E1",
        section="5-YEAR HORIZON",
        title="From Validated Platform to First-in-Class Candidates · 2026 → 2031",
        subtitle=(
            "Three distinct phases · platform validation (2026-27) → extension + early pipelines (2027-28) → "
            "maturation + clinical translation (2029-31) · each phase compounds on the last"
        ),
        eyebrow_color=TEXT_BODY,
    )

    # ====================================================================
    # TOP ZONE: 4 phase progression cards  (y=232..624)
    # Decreasing visual confidence: 2026 most-filled → 2029-31 outlined only
    # ====================================================================
    TZ_Y = 232
    TZ_H = 392
    CARD_GAP = 24
    CARD_W = (W - 2 * START_X - 3 * CARD_GAP) // 4   # ≈420

    parts.append(
        f'<text x="{START_X}" y="{TZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'4-PHASE PROGRESSION · DECREASING VISIBILITY AS HORIZON EXTENDS</text>'
    )
    parts.append(
        f'<line x1="{START_X + 700}" y1="{TZ_Y - 6}" x2="{W - START_X}" y2="{TZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    phases = [
        # (year, title_kicker, title, accent_color, fill_opacity, stroke_w, stroke_op, text_color, bullets)
        ("2026",
         "PHASE 1",
         "VALIDATION",
         CYAN_HI,
         0.18, 2.0, 0.95, TEXT_BODY,
         ["BTK+JAK zero-shot demo (Q4 2026)",
          "Stage 3 ships against QurieSeq Phase 1",
          "Encoder + adapter + readout validated",
          "BSC compute infrastructure proven"]),
        ("2027",
         "PHASE 2",
         "EXTENSION",
         LAVENDER,
         0.14, 1.5, 0.85, TEXT_BODY,
         ["Phospho integration (Phase 2 data)",
          "VDJ + 20-donor scale-up",
          "Pipeline 1 starts (target ID)",
          "Pipeline 2 starts (parallel)",
          "Stage 4 wraps"]),
        ("2028",
         "PHASE 3",
         "MATURATION",
         PURPLE,
         0.08, 1.2, 0.6, TEXT_BODY,
         ["Causal-readiness layer (Stage 5)",
          "Pipeline 1 target validation",
          "Clinical framework groundwork",
          "Pharma partnership conversations"]),
        ("2029–2031",
         "PHASE 4",
         "TRANSLATION",
         TEXT_BODY,
         0.04, 1.0, 0.35, TEXT_MUTED,
         ["First-in-class candidates emerge",
          "Pharma partnerships scale",
          "Pipeline 1 target validated",
          "Pipeline 2 lead selection",
          "Platform = OS for immune drug discovery"]),
    ]

    for i, (year, kicker, title, accent, fop, sw, sop, tc, bullets) in enumerate(phases):
        cx = START_X + i * (CARD_W + CARD_GAP)
        # Stroke style: solid for first two, dashed for later phases
        dash_attr = ' stroke-dasharray="8 5"' if i >= 2 else ''
        # Card
        parts.append(
            f'<rect x="{cx}" y="{TZ_Y + 24}" width="{CARD_W}" height="{TZ_H - 24}" rx="14" '
            f'fill="{accent}" fill-opacity="{fop}" stroke="{accent}" stroke-width="{sw}" '
            f'stroke-opacity="{sop}"{dash_attr}/>'
        )
        # Year ribbon
        parts.append(
            f'<text x="{cx + 24}" y="{TZ_Y + 62}" fill="{accent}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" letter-spacing="3">{year}</text>'
        )
        parts.append(
            f'<line x1="{cx + 24}" y1="{TZ_Y + 72}" x2="{cx + 80}" y2="{TZ_Y + 72}" '
            f'stroke="{accent}" stroke-width="1.5" stroke-opacity="{sop}"/>'
        )
        # Kicker (PHASE 1, etc.)
        parts.append(
            f'<text x="{cx + 24}" y="{TZ_Y + 102}" fill="{TEXT_MUTED}" font-family="{FONT}" '
            f'font-size="11" font-weight="700" letter-spacing="2.5">{kicker}</text>'
        )
        # Title
        title_color = TEXT_TITLE if i < 3 else TEXT_BODY
        parts.append(
            f'<text x="{cx + 24}" y="{TZ_Y + 140}" fill="{title_color}" font-family="{FONT}" '
            f'font-size="28" font-weight="700">{title}</text>'
        )
        # Bullets
        for j, b in enumerate(bullets):
            parts.append(
                f'<text x="{cx + 24}" y="{TZ_Y + 180 + j * 28}" fill="{tc}" font-family="{FONT_BODY}" '
                f'font-size="13" font-weight="400">'
                f'<tspan fill="{accent}" font-weight="700">›</tspan>  {b}</text>'
            )
        # Confidence label at bottom of card
        conf_labels = [
            ("HIGH CONFIDENCE", OK_GREEN),
            ("HIGH CONFIDENCE", OK_GREEN),
            ("PLANNED · CONTINGENT", WARN_AMBER),
            ("DIRECTIONAL", TEXT_MUTED),
        ]
        cl_text, cl_color = conf_labels[i]
        parts.append(
            f'<text x="{cx + 24}" y="{TZ_Y + TZ_H - 28}" fill="{cl_color}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2">◆ {cl_text}</text>'
        )

    # ====================================================================
    # BOTTOM ZONE: 3-card compounding loops  (y=652..888)
    # ====================================================================
    BZ_Y = 652
    BZ_H = 224
    SUB_W = 560
    SUB_GAP = 24
    sub_x0 = (W - 3 * SUB_W - 2 * SUB_GAP) // 2  # 96
    parts.append(
        f'<text x="{sub_x0}" y="{BZ_Y - 12}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">WHY THE PLATFORM COMPOUNDS</text>'
    )
    parts.append(
        f'<line x1="{sub_x0 + 360}" y1="{BZ_Y - 18}" x2="{W - START_X}" y2="{BZ_Y - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    comp_loops = [
        ("DATA COMPOUNDS", CYAN_HI,
         "Every quarter adds wet-lab data to the training corpus",
         ["Phase 1 · 5 donors × 3 modalities",
          "Phase 2 · 20 donors × 5 modalities",
          "Phase 3 · B-cell lines + disease samples"]),
        ("MODEL COMPOUNDS", LAVENDER,
         "Every stage adds capability without re-architecting",
         ["3 modalities → 5 modalities",
          "Single donor → 20-donor scale",
          "Static → temporal Neural ODE",
          "Correlation → causal-readiness"]),
        ("CLINICAL INFRA COMPOUNDS", OK_GREEN,
         "Every milestone adds clinical-partnership readiness",
         ["Regulatory-grade provenance",
          "Computational diligence package",
          "Audit trails + version control"]),
    ]
    for i, (name, color, tagline, bullets) in enumerate(comp_loops):
        cx = sub_x0 + i * (SUB_W + SUB_GAP)
        parts.append(
            f'<rect x="{cx}" y="{BZ_Y}" width="{SUB_W}" height="{BZ_H}" rx="14" '
            f'fill="{SURFACE}" stroke="{color}" stroke-width="1.5" stroke-opacity="0.6"/>'
        )
        parts.append(
            f'<text x="{cx + 20}" y="{BZ_Y + 30}" fill="{color}" font-family="{FONT}" '
            f'font-size="12" font-weight="700" letter-spacing="2.5">{name}</text>'
        )
        parts.append(
            f'<text x="{cx + 20}" y="{BZ_Y + 60}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400" font-style="italic">{tagline}</text>'
        )
        for j, b in enumerate(bullets):
            parts.append(
                f'<text x="{cx + 20}" y="{BZ_Y + 100 + j * 30}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
                f'font-size="14" font-weight="400">'
                f'<tspan fill="{color}" font-weight="700">›</tspan>  {b}</text>'
            )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 (Stage 3/4/5 sequencing) · QurieSeq Phase 1/2/3 plan · "
            "D1 quarterly roadmap (this deck) · "
            "Pipeline starts contingent on Stage 3 verdict + Phase 2 data quality · "
            "2029-2031 directional, not committed"
        ),
        slide_handle="E1 / 12",
        handle_color=TEXT_BODY,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "E1_five_year_trajectory.svg"
    png_path = here / "E1_five_year_trajectory_preview.png"
    svg_path.write_text(build_svg())
    print(f"wrote {svg_path}")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
