"""Build A5_causal_architecture.svg + preview PNG.

Layout (3 vertical zones below the header):
- Top zone (~40%):  Neumann propagation equation ŷ = (I − W)⁻¹ · dₚ as visual
                    hero with color-coded components (W cyan / dₚ lavender /
                    (I−W)⁻¹ green) and ρ(W) < 1 architectural-requirement footer
- Middle zone (~35%): two side-by-side panels — STRUCTURAL PRIOR (STRING)
                      thin grey edges vs LEARNED SPARSE GRN thick cyan + dashed
                      grey edges, same node layout in both for before/after read
- Bottom zone (~25%): direct-effect log-FC head block diagram +
                      Stage 3a/3b-predicted vs Stage 3c-separates comparison

Status pill (non-negotiable, load-bearing for diligence credibility):
  top-right of slide, ◆ STAGE 3c · SPEC-LOCKED + validation timing.

Section A locked palette: cyan + lavender + green (no new colors).
Pagination: A5 / 14 (deck grows from 13 → 14 content slides; intentional
pre-Phase-4 outlier).

Run: python3 docs/deck/assets/diagrams/_build_a5.py
"""
from __future__ import annotations
import pathlib
import sys
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, FONT_MATH, START_X, W, H,
    svg_open, background, header, footer, render_png,
    check_no_text_collisions,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC A5 — Causal architecture (Stage 3c spec-locked)")]
    background(parts)
    # v1: title trimmed from "Causal Architecture — Spec-Locked, Validation
    # Post-Phase-1" (58 chars) to "Causal Architecture — Spec-Locked" (33 chars)
    # because the long title at 40pt overlapped horizontally with the status
    # pill (caught by collision-guard min_gap=2). The validation timing is
    # fully carried on the status pill itself; no need for the title to repeat.
    header(
        parts,
        appendix_id="A5",
        section="ARCHITECTURE DEPTH",
        title="Causal Architecture — Spec-Locked",
        subtitle=(
            "Neumann propagation + sparse learned GRN + direct-effect decoder · "
            "architecturally locked in spec v1.1 · validation begins Q1-Q2 2027 "
            "once Phase 1 wet-lab perturbation data lands"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # STATUS PILL — top-right, load-bearing for diligence credibility.
    # Sits in the header right-side zone (right of the title text).
    # Title text extends ~x=1100 max; pill at x=1456+ avoids any overlap.
    # ====================================================================
    PILL_X, PILL_Y = 1456, 60
    PILL_W, PILL_H = 368, 108
    # Outer card — cyan border (architectural commitment) + cyan-tinted fill
    parts.append(
        f'<rect x="{PILL_X}" y="{PILL_Y}" width="{PILL_W}" height="{PILL_H}" rx="10" '
        f'fill="{CYAN}" fill-opacity="0.12" stroke="{CYAN_HI}" stroke-width="1.5" stroke-opacity="0.85"/>'
    )
    # ◆ amber diamond + "STAGE 3c · SPEC-LOCKED" — top row
    parts.append(
        f'<text x="{PILL_X + 18}" y="{PILL_Y + 34}" fill="{WARN_AMBER}" font-family="{FONT}" '
        f'font-size="20" font-weight="700">◆</text>'
    )
    parts.append(
        f'<text x="{PILL_X + 42}" y="{PILL_Y + 33}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="2.5">STAGE 3c · SPEC-LOCKED</text>'
    )
    # Validation timing — bottom rows
    parts.append(
        f'<text x="{PILL_X + 18}" y="{PILL_Y + 64}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="700">Validation Q1-Q2 2027</text>'
    )
    parts.append(
        f'<text x="{PILL_X + 18}" y="{PILL_Y + 86}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-style="italic">post Phase 1 wet-lab data</text>'
    )

    # ====================================================================
    # TOP ZONE — Neumann propagation block (visual hero)
    # y=216..510
    # ====================================================================
    NZ_X, NZ_Y, NZ_W, NZ_H = START_X, 216, W - 2 * START_X, 294
    # Section eyebrow
    parts.append(
        f'<text x="{NZ_X}" y="{NZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'NEUMANN PROPAGATION · PERTURBATION FLOW THROUGH LEARNED GRAPH</text>'
    )
    parts.append(
        f'<line x1="{NZ_X + 580}" y1="{NZ_Y - 6}" x2="{NZ_X + NZ_W}" y2="{NZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    # Card — primary architectural element gets stronger cyan border
    parts.append(
        f'<rect x="{NZ_X}" y="{NZ_Y + 12}" width="{NZ_W}" height="{NZ_H - 12}" rx="14" '
        f'fill="{SURFACE}" stroke="{CYAN}" stroke-width="1.5" stroke-opacity="0.55"/>'
    )

    # ---- The equation (visual hero, centered) ----
    EQ_CX = NZ_X + NZ_W // 2
    EQ_Y = NZ_Y + 100   # baseline

    # Equation: ŷ = (I − W)⁻¹ · dₚ
    # Render with text-anchor="middle" and color-coded tspans.
    parts.append(
        f'<text x="{EQ_CX}" y="{EQ_Y}" fill="{TEXT_TITLE}" font-family="{FONT_MATH}" '
        f'font-size="56" font-weight="700" text-anchor="middle">'
        f'<tspan font-style="italic">ŷ</tspan>'
        f'<tspan fill="{TEXT_DIM}">  =  </tspan>'
        f'<tspan fill="{OK_GREEN}">(</tspan>'
        f'<tspan fill="{OK_GREEN}" font-style="italic">I</tspan>'
        f'<tspan fill="{OK_GREEN}"> − </tspan>'
        f'<tspan fill="{CYAN_HI}" font-style="italic" font-weight="700">W</tspan>'
        f'<tspan fill="{OK_GREEN}">)</tspan>'
        f'<tspan fill="{OK_GREEN}" font-size="34" baseline-shift="super">−1</tspan>'
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'<tspan fill="{LAVENDER}" font-style="italic" font-weight="700">d</tspan>'
        f'<tspan fill="{LAVENDER}" font-size="34" baseline-shift="sub" font-style="italic">p</tspan>'
        f'</text>'
    )

    # Subtle horizontal underline below the equation
    parts.append(
        f'<line x1="{EQ_CX - 220}" y1="{EQ_Y + 14}" x2="{EQ_CX + 220}" y2="{EQ_Y + 14}" '
        f'stroke="{CYAN}" stroke-width="1" stroke-opacity="0.35"/>'
    )

    # ---- Component definitions row (3 columns under equation) ----
    DEF_Y = EQ_Y + 64
    COL_W = NZ_W // 3
    def_items = [
        # (math symbol, color, annotation)
        ("W",        CYAN_HI,  "sparse learned GRN"),
        ("dₚ",       LAVENDER, "direct perturbation effect"),
        ("(I − W)⁻¹", OK_GREEN, "closed-form propagation"),
    ]
    for i, (sym, color, annotation) in enumerate(def_items):
        cx = NZ_X + i * COL_W + COL_W // 2
        # Math symbol (left of annotation in the column)
        # For "dₚ" use tspan subscript; for "(I − W)⁻¹" use tspan superscript
        if sym == "dₚ":
            sym_xml = (
                f'<tspan font-style="italic" font-weight="700">d</tspan>'
                f'<tspan font-size="16" baseline-shift="sub" font-style="italic">p</tspan>'
            )
        elif sym == "(I − W)⁻¹":
            sym_xml = (
                f'<tspan>(</tspan>'
                f'<tspan font-style="italic">I</tspan>'
                f'<tspan> − </tspan>'
                f'<tspan font-style="italic">W</tspan>'
                f'<tspan>)</tspan>'
                f'<tspan font-size="16" baseline-shift="super">−1</tspan>'
            )
        else:
            sym_xml = f'<tspan font-style="italic" font-weight="700">{sym}</tspan>'
        parts.append(
            f'<text x="{cx - 90}" y="{DEF_Y}" fill="{color}" font-family="{FONT_MATH}" '
            f'font-size="24" font-weight="700" text-anchor="end">{sym_xml}</text>'
        )
        # Annotation right of symbol
        parts.append(
            f'<text x="{cx - 70}" y="{DEF_Y}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="14" font-style="italic">{annotation}</text>'
        )

    # ---- Architectural requirement footer ----
    parts.append(
        f'<line x1="{NZ_X + 32}" y1="{NZ_Y + NZ_H - 38}" x2="{NZ_X + NZ_W - 32}" y2="{NZ_Y + NZ_H - 38}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{EQ_CX}" y="{NZ_Y + NZ_H - 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="13" font-style="italic" text-anchor="middle">'
        f'<tspan font-weight="700">Architectural requirement:</tspan> '
        f'<tspan fill="{CYAN_HI}" font-weight="700">ρ(W) &lt; 1</tspan> '
        f'enforced by sparsity L1 — guarantees Neumann-series convergence'
        f'</text>'
    )

    # ====================================================================
    # MIDDLE ZONE — Sparse learned GRN visualization (2 side-by-side panels)
    # y=534..776
    # ====================================================================
    MZ_Y = 534
    MZ_H = 242
    PANEL_GAP = 60
    PANEL_W = (W - 2 * START_X - PANEL_GAP) // 2

    parts.append(
        f'<text x="{START_X}" y="{MZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'SPARSE LEARNED GRN · STRUCTURAL PRIOR (LEFT) → LEARNED WEIGHTS (RIGHT)</text>'
    )
    parts.append(
        f'<line x1="{START_X + 700}" y1="{MZ_Y - 6}" x2="{W - START_X}" y2="{MZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Both panels: same 7-node layout (consistent before/after read)
    # Node positions in [0..1] panel-local fractional coords (will scale per panel)
    nodes = [
        (0.18, 0.30),   # 0
        (0.40, 0.20),   # 1
        (0.62, 0.30),   # 2
        (0.28, 0.55),   # 3
        (0.50, 0.55),   # 4
        (0.38, 0.80),   # 5
        (0.72, 0.78),   # 6
    ]
    # Edges as (from, to, in_string_prior, learned_weight ∈ {0=pruned, 1=strong, 2=very strong})
    edges = [
        (0, 1, True,  2),
        (1, 2, True,  2),
        (0, 3, True,  1),
        (3, 4, True,  2),
        (4, 5, True,  1),
        (2, 4, False, 1),    # not in prior; learned
        (5, 6, False, 0),    # not in prior; below threshold (pruned)
        (4, 6, True,  0),    # in prior; pruned post-learning
    ]

    def panel(px: int, py: int, pw: int, ph: int,
              title: str, subtitle: str,
              left_panel: bool):
        """Render one of the two GRN panels."""
        # Card
        parts.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" '
            f'fill="{SURFACE}" stroke="{DIVIDER}" stroke-width="1.2" stroke-opacity="0.9"/>'
        )
        # Panel title
        parts.append(
            f'<text x="{px + 20}" y="{py + 26}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" letter-spacing="2">{title}</text>'
        )
        # Subtitle
        parts.append(
            f'<text x="{px + 20}" y="{py + 44}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-style="italic">{subtitle}</text>'
        )
        # Graph drawing area inside panel
        gx0, gy0 = px + 30, py + 56
        gw, gh = pw - 60, ph - 110
        # Convert fractional node positions to absolute
        abs_nodes = [(gx0 + nx * gw, gy0 + ny * gh) for nx, ny in nodes]

        # Draw edges first (so nodes overlay)
        for fi, ti, in_prior, weight in edges:
            x1, y1 = abs_nodes[fi]
            x2, y2 = abs_nodes[ti]
            if left_panel:
                # STRING prior — all priors visible as thin grey
                if in_prior:
                    parts.append(
                        f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="{TEXT_DIM}" stroke-width="1.5" stroke-opacity="0.55"/>'
                    )
            else:
                # Learned GRN — encode weight via stroke
                if weight == 0:
                    # Pruned: dashed thin grey
                    parts.append(
                        f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="{TEXT_DIM}" stroke-width="1.2" stroke-opacity="0.4" '
                        f'stroke-dasharray="4 4"/>'
                    )
                elif weight == 1:
                    # Mid-weight learned: cyan, medium thickness
                    parts.append(
                        f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="{CYAN}" stroke-width="2.5" stroke-opacity="0.75"/>'
                    )
                else:
                    # Strong-weight learned: cyan, thick
                    parts.append(
                        f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="{CYAN_HI}" stroke-width="4" stroke-opacity="0.95"/>'
                    )

        # Draw nodes
        for nx, ny in abs_nodes:
            parts.append(
                f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="9" '
                f'fill="{SURFACE_2}" stroke="{TEXT_BODY}" stroke-width="1.5"/>'
            )

        # Panel-bottom caption (2 lines, italic muted)
        if left_panel:
            cap1 = "STRING-supported edges"
            cap2 = "lower L1 sparsity pressure"
        else:
            cap1 = "thick cyan = high-weight learned"
            cap2 = "dashed = below sparsity threshold (pruned)"
        parts.append(
            f'<text x="{px + 20}" y="{py + ph - 28}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="12" font-weight="700">'
            f'<tspan fill="{CYAN if not left_panel else TEXT_MUTED}" font-weight="700">›</tspan>  '
            f'{cap1}</text>'
        )
        parts.append(
            f'<text x="{px + 20}" y="{py + ph - 10}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-style="italic">{cap2}</text>'
        )

    panel_y = MZ_Y + 16
    panel_h = MZ_H - 16
    panel(START_X, panel_y, PANEL_W, panel_h,
          "STRUCTURAL PRIOR  ·  STRING DB", "edge-existence prior",
          left_panel=True)
    panel(START_X + PANEL_W + PANEL_GAP, panel_y, PANEL_W, panel_h,
          "LEARNED SPARSE GRN", "edge weights after training",
          left_panel=False)

    # Connector arrow + caption between panels
    mid_x = START_X + PANEL_W + PANEL_GAP // 2
    arrow_y = panel_y + panel_h // 2
    # Bidirectional-arrow-like glyph showing transition
    parts.append(
        f'<line x1="{mid_x - 18}" y1="{arrow_y}" x2="{mid_x + 18}" y2="{arrow_y}" '
        f'stroke="{LAVENDER}" stroke-width="2" stroke-opacity="0.85" stroke-linecap="round"/>'
    )
    parts.append(
        f'<path d="M {mid_x + 12} {arrow_y - 6} L {mid_x + 18} {arrow_y} L {mid_x + 12} {arrow_y + 6}" '
        f'fill="none" stroke="{LAVENDER}" stroke-width="2" stroke-opacity="0.95" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    parts.append(
        f'<text x="{mid_x}" y="{arrow_y - 16}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2" text-anchor="middle">'
        f'L1 sparsity</text>'
    )
    # Caption below both panels
    parts.append(
        f'<text x="{W // 2}" y="{panel_y + panel_h + 24}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="13" font-style="italic" text-anchor="middle">'
        f'prior shapes initialization, learning prunes'
        f'</text>'
    )

    # ====================================================================
    # BOTTOM ZONE — Direct-effect log-FC head
    # y=812..920
    # ====================================================================
    BZ_X, BZ_Y, BZ_W, BZ_H = START_X, 812, W - 2 * START_X, 108
    parts.append(
        f'<rect x="{BZ_X}" y="{BZ_Y}" width="{BZ_W}" height="{BZ_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{LAVENDER}" stroke-width="1.5" stroke-opacity="0.55"/>'
    )
    # Eyebrow + title
    parts.append(
        f'<text x="{BZ_X + 22}" y="{BZ_Y + 26}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="2.5">'
        f'DIRECT-EFFECT LOG-FC HEAD</text>'
    )
    # Block-diagram row: [latent z + perturbation context] → [log-FC decoder] → dₚ
    BD_Y = BZ_Y + 48
    BD_BOX_H = 38
    # Box 1 — latent z + perturbation context
    b1_x, b1_w = BZ_X + 22, 360
    parts.append(
        f'<rect x="{b1_x}" y="{BD_Y}" width="{b1_w}" height="{BD_BOX_H}" rx="8" '
        f'fill="{SURFACE_2}" stroke="{TEXT_DIM}" stroke-width="1" stroke-opacity="0.7"/>'
    )
    parts.append(
        f'<text x="{b1_x + b1_w // 2}" y="{BD_Y + BD_BOX_H // 2 + 5}" fill="{TEXT_BODY}" '
        f'font-family="{FONT_BODY}" font-size="13" font-weight="600" text-anchor="middle">'
        f'<tspan font-style="italic">z</tspan> + perturbation context</text>'
    )
    # Arrow 1
    a1_x = b1_x + b1_w + 12
    a1_w = 40
    parts.append(
        f'<line x1="{a1_x}" y1="{BD_Y + BD_BOX_H // 2}" x2="{a1_x + a1_w - 8}" y2="{BD_Y + BD_BOX_H // 2}" '
        f'stroke="{LAVENDER}" stroke-width="2" stroke-opacity="0.85" stroke-linecap="round"/>'
    )
    parts.append(
        f'<path d="M {a1_x + a1_w - 14} {BD_Y + BD_BOX_H // 2 - 5} L {a1_x + a1_w - 6} {BD_Y + BD_BOX_H // 2} L {a1_x + a1_w - 14} {BD_Y + BD_BOX_H // 2 + 5}" '
        f'fill="none" stroke="{LAVENDER}" stroke-width="2" stroke-opacity="0.95" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    # Box 2 — log-FC decoder
    b2_x, b2_w = a1_x + a1_w + 4, 240
    parts.append(
        f'<rect x="{b2_x}" y="{BD_Y}" width="{b2_w}" height="{BD_BOX_H}" rx="8" '
        f'fill="{LAVENDER}" fill-opacity="0.16" stroke="{LAVENDER}" stroke-width="1.5" stroke-opacity="0.85"/>'
    )
    parts.append(
        f'<text x="{b2_x + b2_w // 2}" y="{BD_Y + BD_BOX_H // 2 + 5}" fill="{TEXT_TITLE}" '
        f'font-family="{FONT_BODY}" font-size="13" font-weight="700" text-anchor="middle">'
        f'log-FC decoder</text>'
    )
    # Arrow 2
    a2_x = b2_x + b2_w + 12
    a2_w = 40
    parts.append(
        f'<line x1="{a2_x}" y1="{BD_Y + BD_BOX_H // 2}" x2="{a2_x + a2_w - 8}" y2="{BD_Y + BD_BOX_H // 2}" '
        f'stroke="{LAVENDER}" stroke-width="2" stroke-opacity="0.85" stroke-linecap="round"/>'
    )
    parts.append(
        f'<path d="M {a2_x + a2_w - 14} {BD_Y + BD_BOX_H // 2 - 5} L {a2_x + a2_w - 6} {BD_Y + BD_BOX_H // 2} L {a2_x + a2_w - 14} {BD_Y + BD_BOX_H // 2 + 5}" '
        f'fill="none" stroke="{LAVENDER}" stroke-width="2" stroke-opacity="0.95" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    # Output dₚ
    out_x = a2_x + a2_w + 4
    parts.append(
        f'<text x="{out_x + 20}" y="{BD_Y + BD_BOX_H // 2 + 7}" fill="{LAVENDER}" '
        f'font-family="{FONT_MATH}" font-size="22" font-weight="700">'
        f'<tspan font-style="italic">d</tspan>'
        f'<tspan font-size="16" baseline-shift="sub" font-style="italic">p</tspan>'
        f'</text>'
    )

    # Comparison row right of the block diagram
    cmp_x = out_x + 80
    cmp_y = BD_Y - 4
    parts.append(
        f'<text x="{cmp_x}" y="{cmp_y + 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-weight="400">'
        f'<tspan font-weight="700">Stage 3a/3b predicted:</tspan> '
        f'abundance after perturbation'
        f'</text>'
    )
    parts.append(
        f'<text x="{cmp_x}" y="{cmp_y + 36}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="12" font-weight="400">'
        f'<tspan fill="{CYAN_HI}" font-weight="700">Stage 3c separates:</tspan>  '
        f'<tspan fill="{LAVENDER}" font-weight="700" font-style="italic">d</tspan>'
        f'<tspan fill="{LAVENDER}" font-size="9" baseline-shift="sub" font-style="italic">p</tspan>'
        f'<tspan fill="{TEXT_BODY}"> (direct) + </tspan>'
        f'<tspan fill="{OK_GREEN}" font-weight="700">(I − W)</tspan>'
        f'<tspan fill="{OK_GREEN}" font-size="9" baseline-shift="super">−1</tspan>'
        f'<tspan fill="{LAVENDER}" font-weight="700" font-style="italic"> d</tspan>'
        f'<tspan fill="{LAVENDER}" font-size="9" baseline-shift="sub" font-style="italic">p</tspan>'
        f'<tspan fill="{TEXT_BODY}"> (propagated)</tspan>'
        f'</text>'
    )

    # Why this matters footer (one line)
    parts.append(
        f'<text x="{BZ_X + 22}" y="{BZ_Y + BZ_H - 12}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-style="italic">'
        f'<tspan font-weight="700">Why this matters:</tspan>  causal queries vs predictive queries — '
        f'<tspan fill="{TEXT_BODY}">"what does X cause?"</tspan>'
        f'<tspan fill="{TEXT_DIM}">  vs  </tspan>'
        f'<tspan fill="{TEXT_BODY}">"what happens after X?"</tspan>'
        f'</text>'
    )

    # ---- Footer (standard pattern, A5 / 14 pagination) ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 (causal layer pending §X extension) · "
            "QurieSeq Phase 1+2 spec (Thiago, May 2026) · "
            "STRING DB v12.0 (Szklarczyk et al., 2023, NAR) · "
            "Neumann series propagation (standard linear-algebra reference)"
        ),
        slide_handle="A5 / 14",
        handle_color=CYAN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "A5_causal_architecture.svg"
    png_path = here / "A5_causal_architecture_preview.png"
    svg = build_svg()

    # Collision-guard with F1-v2-lesson tightened threshold (min_gap=2).
    # Filter only known footer-vs-pagination false positive — same explicit
    # filter scope as F1 v2 (per the lesson: don't auto-filter generously).
    collisions = check_no_text_collisions(svg, min_gap=2)
    blocking = [c for c in collisions
                if "A5 / 14" not in (c[0], c[1])
                and not c[0].startswith("Source:")
                and not c[1].startswith("Source:")]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"A5 collision-guard FAIL:\n{msg}")

    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓ min_gap=2)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
