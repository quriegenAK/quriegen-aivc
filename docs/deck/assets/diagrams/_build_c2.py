"""Build C2_btk_jak_demo_plan.svg + preview PNG.

Layout:
- Top zone: eval flow diagram with HELD OUT box bypassing training → zero-shot predict
- Middle zone: pre-registered verdict thresholds table (GREEN / AMBER / RED)
- Bottom zone: clinical context footer (NCT02912754, Maddocks 2016, pJAK1)

Section accent: cyan (continues C1) + GREEN/AMBER/RED for thresholds
Run: python3 docs/deck/assets/diagrams/_build_c2.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER, DANGER_RED,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, TEXT_DISABLED, DIVIDER,
    FONT, FONT_BODY, FONT_MATH, FONT_MONO, START_X, W, H,
    svg_open, background, header, footer, arrow, render_png,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC C2 BTK+JAK demo plan with pre-registered eval thresholds")]
    background(parts)
    header(
        parts,
        appendix_id="C2",
        section="QURIESEQ PHASE 1",
        title="BTK + JAK Headline Demo — Pre-Registered Eval",
        subtitle=(
            "Train on singles, hold out the combo, predict zero-shot · "
            "verdict thresholds locked in spec before any QurieSeq data is collected"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # TOP ZONE: Eval flow diagram  (y=232..480)
    # ====================================================================
    FZ_X, FZ_Y, FZ_W, FZ_H = START_X, 232, W - 2 * START_X, 248

    parts.append(
        f'<text x="{FZ_X}" y="{FZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'EVAL FLOW · STAGE 3B TRAINING → ZERO-SHOT PREDICTION</text>'
    )
    parts.append(
        f'<line x1="{FZ_X + 580}" y1="{FZ_Y - 6}" x2="{FZ_X + FZ_W}" y2="{FZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Three boxes in a row:
    # 1. Single-arm training data (left)
    # 2. Trained model (center)
    # 3. Predict zero-shot (right)
    # Plus a HELD OUT box below #1, bypassing #2 → arrow into #3
    BOX_Y = FZ_Y + 24
    BOX_H = 168
    BOX_GAP = 80

    b1_x = FZ_X
    b1_w = 480
    b2_x = b1_x + b1_w + BOX_GAP    # 656
    b2_w = 440
    b3_x = b2_x + b2_w + BOX_GAP    # 1176
    b3_w = 552

    # Training-data box (left)
    parts.append(
        f'<rect x="{b1_x}" y="{BOX_Y}" width="{b1_w}" height="{BOX_H}" rx="12" '
        f'fill="{SURFACE}" stroke="{CYAN}" stroke-width="1.4" stroke-opacity="0.65"/>'
    )
    parts.append(
        f'<text x="{b1_x + 18}" y="{BOX_Y + 26}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">SINGLE-ARM TRAINING DATA</text>'
    )
    train_items = [
        "BTK alone",
        "JAK alone",
        "IKK16, Idelalisib, Rapamycin",
        "All vehicle + stimuli arms",
        "Other combos (NOT BTK+JAK)",
    ]
    for j, item in enumerate(train_items):
        parts.append(
            f'<text x="{b1_x + 18}" y="{BOX_Y + 54 + j * 22}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">'
            f'<tspan fill="{CYAN}" font-weight="700">›</tspan>  {item}</text>'
        )

    # Trained model box (center)
    parts.append(
        f'<rect x="{b2_x}" y="{BOX_Y}" width="{b2_w}" height="{BOX_H}" rx="12" '
        f'fill="{SURFACE}" stroke="{LAVENDER}" stroke-width="1.4" stroke-opacity="0.65"/>'
    )
    parts.append(
        f'<text x="{b2_x + 18}" y="{BOX_Y + 26}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">TRAINED MODEL</text>'
    )
    model_items = [
        "Frozen encoder",
        "Trained adapter",
        "Trained 4-arm readout",
        "Neural ODE temporal",
    ]
    for j, item in enumerate(model_items):
        parts.append(
            f'<text x="{b2_x + 18}" y="{BOX_Y + 54 + j * 22}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">'
            f'<tspan fill="{LAVENDER}" font-weight="700">›</tspan>  {item}</text>'
        )

    # Predict zero-shot box (right)
    parts.append(
        f'<rect x="{b3_x}" y="{BOX_Y}" width="{b3_w}" height="{BOX_H}" rx="12" '
        f'fill="{CYAN}" fill-opacity="0.12" stroke="{CYAN_HI}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{b3_x + 18}" y="{BOX_Y + 26}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">◆ PREDICT ZERO-SHOT</text>'
    )
    parts.append(
        f'<text x="{b3_x + 18}" y="{BOX_Y + 62}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="22" font-weight="700">BTK + JAK combination</text>'
    )
    parts.append(
        f'<text x="{b3_x + 18}" y="{BOX_Y + 88}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="400">'
        f'<tspan fill="{CYAN_HI}" font-weight="700">›</tspan>  '
        f'response trajectory 0 → 180 min</text>'
    )
    parts.append(
        f'<text x="{b3_x + 18}" y="{BOX_Y + 110}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="400">'
        f'<tspan fill="{CYAN_HI}" font-weight="700">›</tspan>  '
        f'score vs measured combo data</text>'
    )
    parts.append(
        f'<text x="{b3_x + 18}" y="{BOX_Y + 144}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" font-style="italic">'
        f'training never sees BTK+JAK combination</text>'
    )

    # HELD OUT box (below training-data box, narrow)
    ho_y = BOX_Y + BOX_H + 16
    ho_h = 56
    parts.append(
        f'<rect x="{b1_x}" y="{ho_y}" width="{b1_w}" height="{ho_h}" rx="10" '
        f'fill="{DANGER_RED}" fill-opacity="0.08" stroke="{DANGER_RED}" stroke-width="1.5" stroke-opacity="0.8" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{b1_x + 18}" y="{ho_y + 22}" fill="{DANGER_RED}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">◆ HELD OUT DURING TRAINING</text>'
    )
    parts.append(
        f'<text x="{b1_x + 18}" y="{ho_y + 44}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="16" font-weight="700">BTK + JAK combination</text>'
    )

    # Arrows: training-data → model → predict
    arrow(parts, b1_x + b1_w + 4, BOX_Y + BOX_H // 2, b2_x - 4, BOX_Y + BOX_H // 2,
          color=CYAN, opacity=0.7)
    arrow(parts, b2_x + b2_w + 4, BOX_Y + BOX_H // 2, b3_x - 4, BOX_Y + BOX_H // 2,
          color=LAVENDER, opacity=0.7)
    # Big arrow: held-out box → predict box (bypass model — that's the zero-shot meaning)
    arrow(parts, b1_x + b1_w + 4, ho_y + ho_h // 2,
          b3_x - 4, ho_y + ho_h // 2,
          color=DANGER_RED, opacity=0.65, width=2.2)
    parts.append(
        f'<text x="{(b1_x + b1_w + b3_x) // 2}" y="{ho_y + ho_h // 2 - 10}" fill="{DANGER_RED}" '
        f'font-family="{FONT}" font-size="11" font-weight="700" letter-spacing="2.5" text-anchor="middle">'
        f'BYPASS · NEVER SEEN DURING TRAINING</text>'
    )

    # ====================================================================
    # MIDDLE ZONE: Pre-registered verdict thresholds table  (y=540..820)
    # ====================================================================
    TZ_Y = 540
    TZ_H = 274
    parts.append(
        f'<text x="{START_X}" y="{TZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'PRE-REGISTERED STAGE 3B VERDICT THRESHOLDS · SPEC v1.1 §5.1</text>'
    )
    parts.append(
        f'<line x1="{START_X + 580}" y1="{TZ_Y - 6}" x2="{W - START_X}" y2="{TZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Table outer card
    parts.append(
        f'<rect x="{START_X}" y="{TZ_Y + 16}" width="{W - 2 * START_X}" height="{TZ_H - 16}" rx="14" '
        f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="1.2"/>'
    )

    # Column header
    HDR_Y = TZ_Y + 50
    col_x_range = START_X + 56
    col_x_verdict = START_X + 800
    col_x_action = START_X + 1180
    parts.append(
        f'<text x="{col_x_range}" y="{HDR_Y}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">ZERO-SHOT SYNERGY ACCURACY</text>'
    )
    parts.append(
        f'<text x="{col_x_verdict}" y="{HDR_Y}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">VERDICT</text>'
    )
    parts.append(
        f'<text x="{col_x_action}" y="{HDR_Y}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">ACTION</text>'
    )
    parts.append(
        f'<line x1="{START_X + 36}" y1="{HDR_Y + 14}" x2="{W - START_X - 36}" y2="{HDR_Y + 14}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Rows
    rows = [
        ("≥ 0.75",                    "GREEN", OK_GREEN,   "Demo ready · publish + show"),
        ("0.65 — 0.75   CI ⊇ 0.70",   "GREEN", OK_GREEN,   "Demo ready · publish with CI"),
        ("0.65 — 0.75   CI ∌ 0.70",   "AMBER", WARN_AMBER, "Expand sample · re-run"),
        ("0.55 — 0.65   any CI",      "AMBER", WARN_AMBER, "Reduce λ_zero · re-train"),
        ("&lt; 0.55",                 "RED",   DANGER_RED, "Architecture-class pivot · SDE fallback (§7.1)"),
    ]
    ROW_H = 34
    row_y0 = HDR_Y + 28
    for i, (rng, verdict, color, action) in enumerate(rows):
        ry = row_y0 + i * ROW_H
        parts.append(
            f'<text x="{col_x_range}" y="{ry + ROW_H // 2 + 4}" fill="{TEXT_BODY}" font-family="{FONT_MONO}" '
            f'font-size="14" font-weight="600" dominant-baseline="middle">{rng}</text>'
        )
        # Verdict badge
        parts.append(
            f'<text x="{col_x_verdict}" y="{ry + ROW_H // 2 + 4}" fill="{color}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" letter-spacing="1" dominant-baseline="middle">◆ {verdict}</text>'
        )
        # Action
        parts.append(
            f'<text x="{col_x_action}" y="{ry + ROW_H // 2 + 4}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400" dominant-baseline="middle">{action}</text>'
        )
        # Subtle row divider
        if i < len(rows) - 1:
            parts.append(
                f'<line x1="{START_X + 36}" y1="{ry + ROW_H - 2}" x2="{W - START_X - 36}" y2="{ry + ROW_H - 2}" '
                f'stroke="{DIVIDER}" stroke-width="0.5" stroke-opacity="0.6"/>'
            )

    # ====================================================================
    # BOTTOM ZONE: Clinical context line (above footer)
    # ====================================================================
    CG_Y = 868
    parts.append(
        f'<text x="{START_X}" y="{CG_Y}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">WHY THIS COMBINATION MATTERS</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="{CG_Y + 22}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="400">'
        f'<tspan font-weight="700">Ibrutinib</tspan> (BTK inhibitor) + '
        f'<tspan font-weight="700">Ruxolitinib</tspan> (JAK1/2 inhibitor) — '
        f'<tspan font-style="italic">CLL Phase Ib/II trial</tspan> '
        f'<tspan fill="{CYAN}" font-weight="700">NCT02912754</tspan>'
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'<tspan font-weight="700">Maddocks 2016</tspan>, Blood '
        f'<tspan fill="{TEXT_DIM}">(PMID 26819050)</tspan>'
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'Thiago wet-lab IP: '
        f'<tspan fill="{CYAN_HI}" font-weight="700">pJAK1</tspan> unexpectedly active in BCR pathway → biological rationale'
        f'</text>'
    )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 §5.1 (pre-registered Stage 3b thresholds) + §7.1 (SDE fallback) · "
            "QurieSeq Phase 1 design (Thiago confirmation, 2026-05-12) · "
            "Ibrutinib + Ruxolitinib CLL trial NCT02912754 · Maddocks et al. Blood 2016, PMID 26819050"
        ),
        slide_handle="C2 / 12",
        handle_color=CYAN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "C2_btk_jak_demo_plan.svg"
    png_path = here / "C2_btk_jak_demo_plan_preview.png"
    svg_path.write_text(build_svg())
    print(f"wrote {svg_path}")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
