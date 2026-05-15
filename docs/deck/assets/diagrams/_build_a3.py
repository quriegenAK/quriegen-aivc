"""Build A3_decomposed_readout.svg + A3_decomposed_readout_preview.png.

Layout:
- Top-left/center: 4-arm equation, color-coded heads (h_base / Δ_stim / Δ_inh / Δ_synergy)
- Below equation: zero-arm constraint block (theorem-style)
- Right inset: 3-row generalization table (BTK+JAK row highlighted)

Style locked to A1 v2.

Note on math typography:
  Indicator function 𝟙 (U+1D7D9) is not in Inter / Arial / most sans-serif
  fonts and falls back unreliably. Per the spec acceptance criteria
  (which explicitly allows "styled '1' with subscript"), we render
  indicators as a bold styled "1" followed by Iverson brackets:
      "1[s]·Δ_stim"     — semantically the indicator function
  Variables (z, c, s, i, t) are rendered italic, operators upright.

Run: python3 docs/deck/assets/diagrams/_build_a3.py
"""
from __future__ import annotations
import pathlib

# ---- Palette ----
BG          = "#070A14"
SURFACE     = "#0F1428"
SURFACE_2   = "#0B1020"  # deeper surface for theorem block
CYAN        = "#26DDF9"
CYAN_HI     = "#00F2FF"
PURPLE      = "#8B5CF6"
LAVENDER    = "#B47DF0"
TEXT_TITLE  = "#F7FAFF"
TEXT_BODY   = "#EAF6FF"
TEXT_MUTED  = "#A8B4C2"
TEXT_DIM    = "#94A3B8"
OK_GREEN    = "#4ADE80"
WARN_AMBER  = "#FBBF24"
DANGER_RED  = "#FF4D6D"
DIVIDER     = "#1A2235"

# ---- Color code for the 4 heads (reused on C2 per spec) ----
COL_BASE    = TEXT_BODY        # #EAF6FF — pale, baseline
COL_STIM    = OK_GREEN         # #4ADE80 — activation
COL_INH     = PURPLE           # #8B5CF6 — inhibition
COL_SYN     = CYAN_HI          # #00F2FF — synergy (the brand-defining color)

W, H = 1920, 1080
START_X = 96
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
FONT_BODY = "Arial, Inter, 'Helvetica Neue', sans-serif"
FONT_MATH = "Inter, 'Cambria Math', 'STIX Two Math', serif"


def header(parts, appendix_id, section, title, subtitle):
    parts.append(
        f'<text x="{START_X}" y="78" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="4">APPENDIX {appendix_id} · {section}</text>'
    )
    parts.append(
        f'<line x1="{START_X + 380}" y1="72" x2="{START_X + 600}" y2="72" '
        f'stroke="{CYAN}" stroke-opacity="0.4" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{START_X}" y="138" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="40" font-weight="700" letter-spacing="-0.5">{title}</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="186" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="18" font-weight="400">{subtitle}</text>'
    )


def footer(parts, source_text, slide_handle):
    fy_line = H - 132
    fy_text = H - 100
    parts.append(
        f'<line x1="{START_X}" y1="{fy_line}" x2="{W - START_X}" y2="{fy_line}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{START_X}" y="{fy_text}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" font-style="italic">{source_text}</text>'
    )
    parts.append(
        f'<text x="{W - START_X}" y="{fy_text}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2" text-anchor="end">{slide_handle}</text>'
    )


def background(parts):
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    parts.append(
        '<defs>'
        f'<radialGradient id="glow1" cx="0.85" cy="0.15" r="0.55">'
        f'<stop offset="0%" stop-color="{CYAN}" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="{CYAN}" stop-opacity="0"/>'
        f'</radialGradient>'
        f'<radialGradient id="glow2" cx="0.15" cy="0.95" r="0.5">'
        f'<stop offset="0%" stop-color="{PURPLE}" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="{PURPLE}" stop-opacity="0"/>'
        f'</radialGradient>'
        '</defs>'
        f'<rect width="{W}" height="{H}" fill="url(#glow1)"/>'
        f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>'
    )


def indicator_badge(parts, x_anchor: int, y_baseline: int, content: str, color: str = TEXT_DIM) -> int:
    """Render '1[content]' indicator notation. x_anchor is left edge; returns x_right."""
    # Bold styled "1"
    parts.append(
        f'<text x="{x_anchor}" y="{y_baseline}" fill="{color}" font-family="{FONT}" '
        f'font-size="34" font-weight="700" font-style="italic">1</text>'
    )
    parts.append(
        f'<text x="{x_anchor+18}" y="{y_baseline}" fill="{color}" font-family="{FONT_MATH}" '
        f'font-size="34" font-weight="500">[{content}]</text>'
    )
    return x_anchor + 18 + 14 * len(content) + 20  # approx width estimate


def build_svg() -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="AIVC decomposed readout: how synergy generalizes">'
    )
    background(parts)
    header(
        parts,
        appendix_id="A3",
        section="ARCHITECTURE DEPTH",
        title="Decomposed Readout — How Synergy Generalizes",
        subtitle=(
            "Four parallel heads · baseline + stim + inhibitor + synergy · "
            "the synergy head learns only the non-additive correction — that's what enables zero-shot drug combinations"
        ),
    )

    # ====================================================================
    # LEFT ZONE: 4-arm equation  (x=96..1240, y=232..540)
    # ====================================================================
    EQ_X, EQ_Y = 96, 232
    EQ_W, EQ_H = 1144, 380

    # Eyebrow
    parts.append(
        f'<text x="{EQ_X}" y="{EQ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">4-ARM DECOMPOSITION</text>'
    )
    parts.append(
        f'<line x1="{EQ_X + 200}" y1="{EQ_Y-6}" x2="{EQ_X + EQ_W}" y2="{EQ_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    # Card
    parts.append(
        f'<rect x="{EQ_X}" y="{EQ_Y + 12}" width="{EQ_W}" height="{EQ_H - 12}" rx="14" '
        f'fill="{SURFACE}" stroke="{CYAN}" stroke-width="1.5" stroke-opacity="0.5"/>'
    )

    # Render the equation. Use staggered lines, color-coded by head.
    # Top line: "ŷ(c, s, i, t) =  h_base(z, t)" with caption
    line_x = EQ_X + 50
    eq_top = EQ_Y + 68

    # ŷ(c, s, i, t)  — bold pale text
    parts.append(
        f'<text x="{line_x}" y="{eq_top}" fill="{TEXT_TITLE}" font-family="{FONT_MATH}" '
        f'font-size="38" font-weight="600">'
        f'<tspan font-style="italic">ŷ</tspan>'
        f'<tspan fill="{TEXT_MUTED}" font-style="italic">(c, s, i, t)</tspan>'
        f'<tspan fill="{TEXT_DIM}">  =  </tspan>'
        f'<tspan fill="{COL_BASE}" font-weight="700">h<tspan font-size="26" baseline-shift="-30%">base</tspan></tspan>'
        f'<tspan fill="{COL_BASE}" font-style="italic">(z, t)</tspan>'
        f'</text>'
    )
    # v2: top-line "← always active (baseline)" annotation removed — color-coding
    # + the right-side compositional generalization table convey the same info
    # without overlapping the equation terms at slide-fill rendering.

    # 4-arm continuation lines, each starting with "+" indented and color-coded head
    LINE_DY = 70
    indent = line_x + 220   # match ŷ width approx so '+ ...' sits under ' = '

    # v2: arm_line drops the right-side `caption` annotation entirely.
    # The annotations ("← stim present", "← active if inhibitor present",
    # "← combination only (the zero-shot win)") overlapped with Δ_stim
    # and Δ_inh at slide-fill rendering. Color-coding (h_base white,
    # Δ_stim green, Δ_inh purple, Δ_synergy cyan) + the right-side
    # compositional generalization table now do the job the annotations
    # were doing — without competing visually with the equation.
    # v2 bug fix: `indicator_visual_chars` is the count of *rendered* glyphs
    # inside the brackets, not the HTML string length. Previously used
    # len(indicator_content) which counted tspan markup chars (200+ for
    # line 4) and pushed Δ_synergy off the card to x=1912.
    def arm_line(y, plus_color, indicator_content, indicator_visual_chars,
                 indicator_color, head_label, head_subscript, head_args, head_color):
        parts.append(
            f'<text x="{indent - 60}" y="{y}" fill="{plus_color}" font-family="{FONT_MATH}" '
            f'font-size="36" font-weight="700">+</text>'
        )
        # Bold italic 1
        parts.append(
            f'<text x="{indent - 10}" y="{y}" fill="{indicator_color}" font-family="{FONT}" '
            f'font-size="32" font-weight="700" font-style="italic">1</text>'
        )
        # [s] / [i] / [s ∧ i]
        parts.append(
            f'<text x="{indent + 10}" y="{y}" fill="{indicator_color}" font-family="{FONT_MATH}" '
            f'font-size="32" font-weight="500"><tspan fill="{TEXT_DIM}">[</tspan>'
            f'{indicator_content}<tspan fill="{TEXT_DIM}">]</tspan></text>'
        )
        # · operator (positioned based on indicator visual width, not HTML chars)
        ind_w = 38 + indicator_visual_chars * 16
        parts.append(
            f'<text x="{indent + 10 + ind_w}" y="{y}" fill="{TEXT_DIM}" font-family="{FONT_MATH}" '
            f'font-size="32" font-weight="700">·</text>'
        )
        # head label: Δ_subscript (z, args)
        head_x = indent + 10 + ind_w + 28
        parts.append(
            f'<text x="{head_x}" y="{y}" fill="{head_color}" font-family="{FONT_MATH}" '
            f'font-size="34" font-weight="700">Δ<tspan font-size="22" baseline-shift="-30%">{head_subscript}</tspan>'
            f'<tspan font-style="italic" font-weight="500" fill="{head_color}">({head_args})</tspan></text>'
        )

    arm_line(eq_top + LINE_DY, COL_STIM,
             "<tspan font-style='italic'>s</tspan>", 1,
             COL_STIM, "Δ_stim", "stim", "z, s, t", COL_STIM)
    arm_line(eq_top + 2 * LINE_DY, COL_INH,
             "<tspan font-style='italic'>i</tspan>", 1,
             COL_INH, "Δ_inh", "inh", "z, i, t", COL_INH)
    arm_line(eq_top + 3 * LINE_DY, COL_SYN,
             "<tspan font-style='italic'>s</tspan><tspan fill='" + TEXT_DIM + "'> ∧ </tspan><tspan font-style='italic'>i</tspan>",
             5,  # visual: 's' + ' ' + '∧' + ' ' + 'i' = 5 glyphs
             COL_SYN, "Δ_synergy", "synergy", "z, s, i, t", COL_SYN)

    # ====================================================================
    # CONSTRAINT BLOCK (theorem-style)  x=96..1240, y=560..820
    # ====================================================================
    CB_X, CB_Y, CB_W, CB_H = 96, 560, 1144, 256
    parts.append(
        f'<text x="{CB_X}" y="{CB_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">LOAD-BEARING CONSTRAINT</text>'
    )
    parts.append(
        f'<line x1="{CB_X + 240}" y1="{CB_Y-6}" x2="{CB_X + CB_W}" y2="{CB_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    # Theorem box
    parts.append(
        f'<rect x="{CB_X}" y="{CB_Y + 12}" width="{CB_W}" height="{CB_H - 12}" rx="14" '
        f'fill="{SURFACE_2}" stroke="{LAVENDER}" stroke-width="1.5" stroke-opacity="0.5"/>'
    )
    # Theorem header (small label like a math lemma)
    parts.append(
        f'<text x="{CB_X + 24}" y="{CB_Y + 50}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="13" font-weight="700" letter-spacing="2.5">ZERO-ARM PENALTY · L₂ SOFT CONSTRAINT</text>'
    )
    # Three condition lines
    cond_lines = [
        ("For NTC (no stim, no inh):",         "Δ_stim, Δ_inh, Δ_synergy = 0"),
        ("For stim-only cells:",               "Δ_inh, Δ_synergy = 0"),
        ("For inhibitor-only cells:",          "Δ_stim, Δ_synergy = 0"),
    ]
    cy0 = CB_Y + 92
    for i, (lhs, rhs) in enumerate(cond_lines):
        cy = cy0 + i * 32
        parts.append(
            f'<text x="{CB_X + 32}" y="{cy}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="16" font-weight="400">{lhs}</text>'
        )
        parts.append(
            f'<text x="{CB_X + 360}" y="{cy}" fill="{TEXT_BODY}" font-family="{FONT_MATH}" '
            f'font-size="16" font-weight="700">{rhs}</text>'
        )
    # Penalty line, set apart
    py = cy0 + 3 * 32 + 26
    parts.append(
        f'<line x1="{CB_X + 32}" y1="{py - 18}" x2="{CB_X + CB_W - 32}" y2="{py - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{CB_X + 32}" y="{py + 4}" fill="{TEXT_BODY}" font-family="{FONT_MATH}" '
        f'font-size="20" font-weight="700">'
        f'<tspan fill="{TEXT_MUTED}" font-weight="400">Penalty: </tspan>'
        f'L<tspan font-size="14" baseline-shift="-30%">zero-arm</tspan>'
        f'<tspan fill="{TEXT_DIM}"> = </tspan>'
        f'<tspan fill="{CYAN_HI}">λ</tspan>'
        f'<tspan fill="{TEXT_DIM}"> · Σ </tspan>'
        f'‖Δ‖²  '
        f'<tspan fill="{TEXT_MUTED}" font-weight="400">where condition fails</tspan>'
        f'</text>'
    )
    parts.append(
        f'<text x="{CB_X + CB_W - 32}" y="{py + 4}" fill="{CYAN_HI}" font-family="{FONT_MATH}" '
        f'font-size="20" font-weight="700" text-anchor="end">'
        f'λ = 1.0'
        f'</text>'
    )
    parts.append(
        f'<text x="{CB_X + CB_W - 32}" y="{py + 26}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" font-style="italic" text-anchor="end">'
        f'architecture spec v1.1, §3.2.2</text>'
    )

    # ====================================================================
    # RIGHT INSET: Generalization table  x=1284..1824, y=232..820
    # ====================================================================
    RT_X, RT_Y, RT_W, RT_H = 1284, 232, 540, 584
    parts.append(
        f'<text x="{RT_X}" y="{RT_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">COMPOSITIONAL GENERALIZATION</text>'
    )
    parts.append(
        f'<line x1="{RT_X + 256}" y1="{RT_Y-6}" x2="{RT_X + RT_W}" y2="{RT_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    rows = [
        # (eyebrow, training_data, target, status_color, status_label, status_detail)
        ("VALIDATED",  "Mimitou CRISPR · CD3E single + CD4 single + CD3E×CD4 double",
         "Predict held-out CD3E + CD4 double-KO",
         OK_GREEN, "0.68 accuracy", "Stage 3 Part 1 · 2.27× chance"),
        ("ZERO-SHOT TARGET", "QurieSeq Phase 1 · BTK single + JAK single + 4-arm controls",
         "Predict BTK + JAK combination, never trained on either pair",
         CYAN_HI, "Headline demo", "Q4 2026"),
        ("ARCHITECTURE CLAIM", "Mimitou · single perturbations",
         "Compose any pairwise combination at inference",
         LAVENDER, "Compositional", "no retraining"),
    ]
    row_h = 178
    row_y0 = RT_Y + 16
    for i, (eyebrow_t, training, target, sc, slabel, sdetail) in enumerate(rows):
        ry = row_y0 + i * (row_h + 8)
        highlight = (i == 1)
        stroke_w = "2" if highlight else "1.2"
        op = "0.85" if highlight else "0.55"
        fill_op = "0.16" if highlight else "0.0"
        # row card
        parts.append(
            f'<rect x="{RT_X}" y="{ry}" width="{RT_W}" height="{row_h}" rx="12" '
            f'fill="{sc}" fill-opacity="{fill_op}" stroke="{sc}" stroke-width="{stroke_w}" stroke-opacity="{op}"/>'
        )
        # eyebrow
        parts.append(
            f'<text x="{RT_X + 18}" y="{ry + 24}" fill="{sc}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2.5">{eyebrow_t}</text>'
        )
        # training data label
        parts.append(
            f'<text x="{RT_X + 18}" y="{ry + 52}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="400" font-style="italic">TRAIN ON</text>'
        )
        parts.append(
            f'<text x="{RT_X + 18}" y="{ry + 72}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">{training}</text>'
        )
        # target
        parts.append(
            f'<text x="{RT_X + 18}" y="{ry + 100}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="400" font-style="italic">PREDICT</text>'
        )
        parts.append(
            f'<text x="{RT_X + 18}" y="{ry + 120}" fill="{TEXT_TITLE}" font-family="{FONT_BODY}" '
            f'font-size="14" font-weight="700">{target}</text>'
        )
        # status pill bottom-right
        sp_w = 130
        sp_x = RT_X + RT_W - sp_w - 18
        sp_y = ry + row_h - 44
        parts.append(
            f'<rect x="{sp_x}" y="{sp_y}" width="{sp_w}" height="28" rx="14" '
            f'fill="{sc}" fill-opacity="0.18" stroke="{sc}" stroke-width="1" stroke-opacity="0.7"/>'
        )
        parts.append(
            f'<text x="{sp_x + sp_w/2}" y="{sp_y + 19}" fill="{sc}" font-family="{FONT}" '
            f'font-size="12" font-weight="700" letter-spacing="0.5" text-anchor="middle">{slabel}</text>'
        )
        parts.append(
            f'<text x="{RT_X + 18}" y="{ry + row_h - 24}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="400" font-style="italic">{sdetail}</text>'
        )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 §3.2 (4-arm decomposition) + §3.2.2 (zero-arm constraint, λ=1.0) · "
            "Stage 3 Part 1 verdict (2026-05-11) · "
            "Implementation aivc/skills/decomposed_readout.py · "
            "Test test_zero_arm_loss_double_perturbation_no_constraint"
        ),
        slide_handle="A3 / 12",
    )

    parts.append("</svg>")
    return "\n".join(parts)


def build_png(svg_path, png_path):
    import cairosvg
    cairosvg.svg2png(
        bytestring=svg_path.read_bytes(),
        write_to=str(png_path),
        output_width=W, output_height=H,
    )


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "A3_decomposed_readout.svg"
    png_path = here / "A3_decomposed_readout_preview.png"
    svg = build_svg()
    svg_path.write_text(svg)
    print(f"wrote {svg_path} ({len(svg)} bytes)")
    build_png(svg_path, png_path)
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes)")
