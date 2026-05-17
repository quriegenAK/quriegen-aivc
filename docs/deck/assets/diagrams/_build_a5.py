"""Build A5_causal_architecture.svg + preview PNG.

v3 fixes (commit ff92117 / v2 → v3):
  Fix 1 (FINAL): all 3 instances of `(I − W)⁻¹ · dₚ` equations now render via
    matplotlib mathtext → base64 PNG → SVG <image> embed. Stops fighting
    cairosvg's italic-Latin-I → turnstile substitution bug. Math is now
    bitmap and bypasses font-substitution entirely. Three sites:
      - Hero equation at ~56pt equivalent (top zone)
      - Component definitions col 3 `(I − W)⁻¹` at ~24pt
      - Bottom comparison fragment `(I − W)⁻¹ dₚ` at ~12pt
    Color-coded rendering deferred to Phase 4 — v3 uses white-only math
    per prompt: correctness over decoration.
  Fix 2: architectural-requirement footer was 3 separate <text> elements
    with hardcoded segment widths that didn't match actual rendered
    widths, producing garbled overlap. Collapsed to single text element
    with text-anchor="middle". Plain SVG text (ρ and < render fine in
    Inter; mathtext PNG reserved for `(I − W)⁻¹` only).

Carry-over from v2 (preserved, do not touch per prompt):
  - GRN visualization (8 named gene nodes + gradients + edge weights +
    legend) — outstanding per Ash review
  - Status pill (STAGE 3c · SPEC-LOCKED · Q1-Q2 2027 · post Phase 1)
  - Subtitle (125 chars)
  - Direct-effect log-FC head block diagram
  - Causal vs predictive comparison rows
  - Honesty discipline (no operational/validated/in-production language)
  - Section A palette (cyan + lavender + green + amber on dark navy)
  - Pagination A5 / 14

Run: python3 docs/deck/assets/diagrams/_build_a5.py
"""
from __future__ import annotations
import base64
import io
import math
import pathlib
import sys

# Matplotlib import (only needed for the 3 equation PNGs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 'stix' fontset renders math symbols professionally and matches investor-grade
# typography per prompt recommendation.
matplotlib.rcParams["mathtext.fontset"] = "stix"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, FONT_MATH, START_X, W, H,
    svg_open, background, header, footer, render_png,
    check_no_text_collisions,
)

ACCENT_AMBER = WARN_AMBER


# =============================================================================
# v3 Fix 1: matplotlib mathtext → base64 PNG → SVG <image> embed
# =============================================================================
def render_math_to_base64_png(latex_expr: str, fontsize: int = 24,
                              color: str = "white",
                              dpi: int = 300) -> tuple[str, int, int]:
    """Render a LaTeX-style math expression to a base64-encoded PNG.

    Returns (base64_data_uri, display_w_px, display_h_px) where the display
    dimensions are in SVG units (1 SVG unit = 1pt at 72 dpi).

    Uses matplotlib mathtext (no LaTeX install required). Renders with
    transparent background so the math overlays cleanly on the dark SVG.

    v3.1 bug fix: earlier version added an axes with xlim/ylim(0,1) and
    `bbox_inches='tight'` then cropped to the axes bbox (the full figure
    area), producing 578×146 PNG regardless of fontsize. Fix: render text
    directly on the figure with no axes; tight-crop measures only the text
    bbox now, producing PNGs sized proportional to fontsize as expected.
    """
    # Render directly on the figure — no axes. With no axes, bbox_inches='tight'
    # crops to the text artist's bounding box exactly.
    fig = plt.figure(dpi=dpi)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, latex_expr, fontsize=fontsize, color=color,
             ha="center", va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    buf.seek(0)
    raw = buf.read()

    # Measure actual cropped PNG dimensions
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(io.BytesIO(raw))
        width_px, height_px = img.size
    except Exception:
        # Fallback: estimate from fontsize
        width_px, height_px = int(fontsize * 8 * dpi / 72), int(fontsize * 2 * dpi / 72)

    # Convert px-at-dpi → SVG-display-px (1 SVG unit = 1pt at 72dpi)
    display_w = max(int(width_px * 72 / dpi), 1)
    display_h = max(int(height_px * 72 / dpi), 1)

    b64 = base64.b64encode(raw).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"
    return data_uri, display_w, display_h


def math_image(latex_expr: str, x_center: int, y_center: int,
               fontsize: int = 24, color: str = "white") -> str:
    """Return an SVG `<image>` tag rendering a math expression centered at
    (x_center, y_center). Uses matplotlib mathtext for math typography."""
    data_uri, w, h = render_math_to_base64_png(latex_expr, fontsize, color)
    img_x = x_center - w // 2
    img_y = y_center - h // 2
    return f'<image x="{img_x}" y="{img_y}" width="{w}" height="{h}" href="{data_uri}"/>'


# =============================================================================
# Main SVG build
# =============================================================================
def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC A5 v3 — Causal architecture (Stage 3c spec-locked)")]
    background(parts)

    # ====================================================================
    # SVG <defs> — radial gradients for 8 GRN nodes (v2 unchanged)
    # ====================================================================
    gradient_defs = '<defs>'
    GRAD_DEFS = {
        "cyan":     ("#26DDF9", "#00F2FF"),
        "lavender": ("#8B5CF6", "#B47DF0"),
        "green":    ("#4ADE80", "#86EFAC"),
        "blue":     ("#5B9BD5", "#94BFE0"),
    }
    for name, (c0, c1) in GRAD_DEFS.items():
        gradient_defs += (
            f'<radialGradient id="grn-{name}" cx="0.35" cy="0.35" r="0.85">'
            f'  <stop offset="0%" stop-color="{c1}" stop-opacity="0.85"/>'
            f'  <stop offset="60%" stop-color="{c0}" stop-opacity="0.45"/>'
            f'  <stop offset="100%" stop-color="{c0}" stop-opacity="0.18"/>'
            f'</radialGradient>'
        )
    gradient_defs += '</defs>'
    parts.append(gradient_defs)

    header(
        parts,
        appendix_id="A5",
        section="ARCHITECTURE DEPTH",
        title="Causal Architecture — Spec-Locked",
        subtitle=(
            "Neumann propagation + sparse learned GRN + direct-effect decoder · "
            "spec-locked v1.1 · validation post Phase 1 (Q1-Q2 2027)"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # STATUS PILL — top-right (v2 unchanged)
    # ====================================================================
    PILL_X, PILL_Y = 1456, 60
    PILL_W, PILL_H = 368, 108
    parts.append(
        f'<rect x="{PILL_X}" y="{PILL_Y}" width="{PILL_W}" height="{PILL_H}" rx="10" '
        f'fill="{CYAN}" fill-opacity="0.12" stroke="{CYAN_HI}" stroke-width="1.5" stroke-opacity="0.85"/>'
    )
    parts.append(
        f'<text x="{PILL_X + 18}" y="{PILL_Y + 34}" fill="{WARN_AMBER}" font-family="{FONT}" '
        f'font-size="20" font-weight="700">◆</text>'
    )
    parts.append(
        f'<text x="{PILL_X + 42}" y="{PILL_Y + 33}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="2.5">STAGE 3c · SPEC-LOCKED</text>'
    )
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
    # ====================================================================
    NZ_X, NZ_Y, NZ_W, NZ_H = START_X, 216, W - 2 * START_X, 244
    parts.append(
        f'<text x="{NZ_X}" y="{NZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'NEUMANN PROPAGATION · PERTURBATION FLOW THROUGH LEARNED GRAPH</text>'
    )
    parts.append(
        f'<line x1="{NZ_X + 580}" y1="{NZ_Y - 6}" x2="{NZ_X + NZ_W}" y2="{NZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<rect x="{NZ_X}" y="{NZ_Y + 12}" width="{NZ_W}" height="{NZ_H - 12}" rx="14" '
        f'fill="{SURFACE}" stroke="{CYAN}" stroke-width="1.5" stroke-opacity="0.55"/>'
    )

    # ---- v3 Fix 1: Hero equation via matplotlib mathtext PNG embed ----
    EQ_CX = NZ_X + NZ_W // 2
    EQ_CY = NZ_Y + 78
    hero_expr = r"$\hat{y} \; = \; (I - W)^{-1} \cdot d_p$"
    parts.append(math_image(hero_expr, EQ_CX, EQ_CY, fontsize=44, color="white"))

    # Subtle horizontal underline below the equation
    parts.append(
        f'<line x1="{EQ_CX - 220}" y1="{EQ_CY + 32}" x2="{EQ_CX + 220}" y2="{EQ_CY + 32}" '
        f'stroke="{CYAN}" stroke-width="1" stroke-opacity="0.35"/>'
    )

    # ---- Component definitions row (3 columns) ----
    DEF_Y = NZ_Y + 158
    COL_W = NZ_W // 3
    # Cols 0+1 — normal text rendering (no I-substitution risk)
    for i, (sym, color, annotation) in enumerate([
        ("W",  CYAN_HI,  "sparse learned GRN"),
        ("dₚ", LAVENDER, "direct perturbation effect"),
    ]):
        cx = NZ_X + i * COL_W + COL_W // 2
        if sym == "dₚ":
            sym_xml = (
                f'<tspan font-style="italic" font-weight="700">d</tspan>'
                f'<tspan font-style="italic">ₚ</tspan>'
            )
        else:
            sym_xml = f'<tspan font-style="italic" font-weight="700">{sym}</tspan>'
        parts.append(
            f'<text x="{cx - 90}" y="{DEF_Y}" fill="{color}" font-family="Inter, Arial, sans-serif" '
            f'font-size="24" font-weight="700" text-anchor="end">{sym_xml}</text>'
        )
        parts.append(
            f'<text x="{cx - 70}" y="{DEF_Y}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="14" font-style="italic">{annotation}</text>'
        )
    # Col 2 — `(I − W)⁻¹` via matplotlib mathtext PNG embed (Fix 1)
    cx2 = NZ_X + 2 * COL_W + COL_W // 2
    # Position math image centered at (cx2 - 130, DEF_Y - 8) — left of annotation
    comp3_expr = r"$(I - W)^{-1}$"
    parts.append(math_image(comp3_expr, cx2 - 132, DEF_Y - 8, fontsize=20, color="white"))
    # Annotation right of math image
    parts.append(
        f'<text x="{cx2 - 70}" y="{DEF_Y}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="14" font-style="italic">closed-form propagation</text>'
    )

    # ---- v3 Fix 2: Architectural-requirement footer as ONE single text element ----
    # Was 3 separate <text> elements with hardcoded segment widths that didn't
    # match actual rendered widths → garbled overlap. Collapsed back to single
    # text with text-anchor="middle". Plain text for ρ + < (Inter renders both
    # correctly; matplotlib mathtext PNG reserved for (I − W)⁻¹ only).
    AR_Y = NZ_Y + 220
    parts.append(
        f'<line x1="{NZ_X + 32}" y1="{AR_Y - 22}" x2="{NZ_X + NZ_W - 32}" y2="{AR_Y - 22}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    # v3 Fix 2 (final): single text element, single uniform fill, no nested
    # tspans. Earlier v3 attempt kept a cyan-colored <tspan> for `ρ(W) < 1`
    # which triggered cairosvg's text-anchor="middle" + nested-tspan render
    # bug (garbled overlap). Plain text only — ρ and < both render fine
    # in Inter without any inline coloring.
    parts.append(
        f'<text x="{EQ_CX}" y="{AR_Y}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="13" font-style="italic" text-anchor="middle">'
        f'Architectural requirement:  ρ(W) &lt; 1  enforced by sparsity L1  —  guarantees Neumann-series convergence'
        f'</text>'
    )

    # ====================================================================
    # MIDDLE ZONE — Sparse learned GRN visualization (v2 unchanged)
    # ====================================================================
    MZ_Y = 484
    MZ_H = 302
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

    NODES = [
        ("BTK",   0.18, 0.18, "cyan",     22),
        ("CD3E",  0.50, 0.10, "cyan",     22),
        ("JAK",   0.82, 0.18, "cyan",     22),
        ("ZAP70", 0.30, 0.46, "green",    20),
        ("MYD88", 0.70, 0.46, "green",    20),
        ("NFKB",  0.38, 0.78, "lavender", 30),
        ("STAT3", 0.62, 0.78, "lavender", 30),
        ("IRF7",  0.92, 0.90, "blue",     18),
    ]
    node_by_label = {n[0]: n for n in NODES}
    EDGES = [
        ("BTK",   "NFKB",  True,  2),
        ("JAK",   "STAT3", True,  2),
        ("CD3E",  "ZAP70", True,  2),
        ("ZAP70", "NFKB",  True,  1),
        ("MYD88", "NFKB",  True,  2),
        ("MYD88", "IRF7",  True,  1),
        ("STAT3", "IRF7",  True,  0),
        ("NFKB",  "IRF7",  True,  0),
        ("STAT3", "NFKB",  False, 2),  # novel learned edge
    ]

    def render_panel(px: int, py: int, pw: int, ph: int,
                     title: str, subtitle: str, left_panel: bool):
        parts.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" '
            f'fill="{SURFACE}" stroke="{DIVIDER}" stroke-width="1.2" stroke-opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{px + 20}" y="{py + 26}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" letter-spacing="2">{title}</text>'
        )
        parts.append(
            f'<text x="{px + 20}" y="{py + 44}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-style="italic">{subtitle}</text>'
        )
        gx0, gy0 = px + 30, py + 60
        gw, gh = pw - 60, ph - 130

        def node_pos(label: str):
            n = node_by_label[label]
            return (gx0 + n[1] * gw, gy0 + n[2] * gh, n[3], n[4])

        # Edges
        for src_lbl, tgt_lbl, in_prior, weight in EDGES:
            sx, sy, _, sr = node_pos(src_lbl)
            tx, ty, _, tr = node_pos(tgt_lbl)
            dx, dy = tx - sx, ty - sy
            dist = math.hypot(dx, dy) or 1
            ux, uy = dx / dist, dy / dist
            x1 = sx + ux * sr
            y1 = sy + uy * sr
            x2 = tx - ux * tr
            y2 = ty - uy * tr

            if left_panel:
                if in_prior:
                    parts.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{TEXT_MUTED}" stroke-width="2" stroke-opacity="0.55"/>'
                    )
            else:
                if weight == 0:
                    parts.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{TEXT_DIM}" stroke-width="1.2" stroke-opacity="0.42" '
                        f'stroke-dasharray="4 4"/>'
                    )
                    continue
                if not in_prior:
                    stroke_color = CYAN_HI
                    stroke_width = 4
                    stroke_opacity = 0.95
                elif weight == 2:
                    stroke_color = CYAN_HI
                    stroke_width = 4
                    stroke_opacity = 0.9
                else:
                    stroke_color = CYAN
                    stroke_width = 2.5
                    stroke_opacity = 0.75
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{stroke_color}" stroke-width="{stroke_width}" '
                    f'stroke-opacity="{stroke_opacity}"/>'
                )
                head_len = 8
                head_w = 5
                px_x, px_y = -uy, ux
                h1x = x2 - head_len * ux + head_w * px_x
                h1y = y2 - head_len * uy + head_w * px_y
                h2x = x2 - head_len * ux - head_w * px_x
                h2y = y2 - head_len * uy - head_w * px_y
                parts.append(
                    f'<path d="M {h1x:.1f} {h1y:.1f} L {x2:.1f} {y2:.1f} L {h2x:.1f} {h2y:.1f}" '
                    f'fill="none" stroke="{LAVENDER}" stroke-width="1.6" stroke-opacity="0.85" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>'
                )
                if not in_prior:
                    mx, my = (sx + tx) / 2, (sy + ty) / 2
                    parts.append(
                        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="{LAVENDER}" '
                        f'stroke="{TEXT_TITLE}" stroke-width="1.2"/>'
                    )
                    parts.append(
                        f'<text x="{mx:.1f}" y="{my + 3:.1f}" fill="{TEXT_TITLE}" '
                        f'font-family="{FONT}" font-size="8" font-weight="700" '
                        f'text-anchor="middle">◆</text>'
                    )

        for n in NODES:
            label, fx, fy, gene_class, r = n
            nx = gx0 + fx * gw
            ny = gy0 + fy * gh
            parts.append(
                f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="{r + 4}" fill="none" '
                f'stroke="url(#grn-{gene_class})" stroke-width="1.5" stroke-opacity="0.45"/>'
            )
            parts.append(
                f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="{r}" '
                f'fill="url(#grn-{gene_class})" stroke="{TEXT_TITLE}" '
                f'stroke-width="1" stroke-opacity="0.55"/>'
            )
            parts.append(
                f'<text x="{nx:.0f}" y="{ny + 4}" fill="{TEXT_TITLE}" '
                f'font-family="{FONT_BODY}" font-size="11" font-weight="700" '
                f'text-anchor="middle">{label}</text>'
            )

        cap_y = py + ph - 50
        if left_panel:
            parts.append(
                f'<text x="{px + 20}" y="{cap_y}" fill="{TEXT_BODY}" '
                f'font-family="{FONT_BODY}" font-size="12" font-weight="700">'
                f'<tspan fill="{TEXT_MUTED}" font-weight="700">›</tspan>  '
                f'STRING-supported edges</text>'
            )
            parts.append(
                f'<text x="{px + 20}" y="{cap_y + 18}" fill="{TEXT_MUTED}" '
                f'font-family="{FONT_BODY}" font-size="11" font-style="italic">'
                f'lower L1 sparsity pressure</text>'
            )
            legend_y = py + ph - 16
            lx = px + 20
            parts.append(
                f'<text x="{lx}" y="{legend_y}" fill="{TEXT_DIM}" '
                f'font-family="{FONT_BODY}" font-size="10" font-style="italic">'
                f'<tspan fill="{CYAN_HI}" font-weight="700">●</tspan> perturbation target  '
                f'<tspan fill="{LAVENDER}" font-weight="700">●</tspan> TF hub  '
                f'<tspan fill="{OK_GREEN}" font-weight="700">●</tspan> kinase  '
                f'<tspan fill="#94BFE0" font-weight="700">●</tspan> effector'
                f'</text>'
            )
        else:
            parts.append(
                f'<text x="{px + 20}" y="{cap_y}" fill="{CYAN_HI}" '
                f'font-family="{FONT_BODY}" font-size="12" font-weight="700">'
                f'<tspan font-weight="700">›</tspan>  '
                f'thick cyan = high-weight learned</text>'
            )
            parts.append(
                f'<text x="{px + 20}" y="{cap_y + 18}" fill="{TEXT_MUTED}" '
                f'font-family="{FONT_BODY}" font-size="11" font-style="italic">'
                f'dashed = below sparsity threshold · '
                f'<tspan fill="{LAVENDER}" font-weight="700">◆</tspan> = '
                f'novel (not in STRING prior)</text>'
            )
            legend_x = px + pw - 240
            legend_y = py + ph - 60
            parts.append(
                f'<rect x="{legend_x - 8}" y="{legend_y - 12}" width="232" height="62" rx="6" '
                f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="0.8" stroke-opacity="0.8"/>'
            )
            parts.append(
                f'<text x="{legend_x}" y="{legend_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
                f'font-size="9" font-weight="700" letter-spacing="2">EDGE LEGEND</text>'
            )
            row_specs = [
                ('━━',  CYAN_HI,  4,   None,    "high-weight learned"),
                ('──',  CYAN,     2.5, None,    "medium-weight learned"),
                ('···', TEXT_DIM, 1.2, '4 4',   "pruned (below threshold)"),
                ('◆',   LAVENDER, 0,   None,    "novel (not in STRING)"),
            ]
            for li, (glyph, color, sw, dash, desc) in enumerate(row_specs):
                row_y = legend_y + 14 + li * 10
                lx = legend_x + 4
                lx_end = lx + 28
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
                if glyph == '◆':
                    parts.append(
                        f'<text x="{lx + 14}" y="{row_y + 3}" fill="{color}" '
                        f'font-family="{FONT}" font-size="10" font-weight="700" '
                        f'text-anchor="middle">◆</text>'
                    )
                else:
                    parts.append(
                        f'<line x1="{lx}" y1="{row_y}" x2="{lx_end}" y2="{row_y}" '
                        f'stroke="{color}" stroke-width="{sw}"{dash_attr}/>'
                    )
                parts.append(
                    f'<text x="{lx_end + 8}" y="{row_y + 3}" fill="{TEXT_BODY}" '
                    f'font-family="{FONT_BODY}" font-size="9" font-weight="400">{desc}</text>'
                )

    panel_y = MZ_Y + 16
    panel_h = MZ_H - 16
    render_panel(START_X, panel_y, PANEL_W, panel_h,
                 "STRUCTURAL PRIOR  ·  STRING DB", "edge-existence prior",
                 left_panel=True)
    render_panel(START_X + PANEL_W + PANEL_GAP, panel_y, PANEL_W, panel_h,
                 "LEARNED SPARSE GRN", "edge weights + direction after training",
                 left_panel=False)

    mid_x = START_X + PANEL_W + PANEL_GAP // 2
    arrow_y = panel_y + panel_h // 2
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
    parts.append(
        f'<text x="{W // 2}" y="{panel_y + panel_h + 22}" fill="{TEXT_MUTED}" '
        f'font-family="{FONT_BODY}" font-size="13" font-style="italic" text-anchor="middle">'
        f'prior shapes initialization, learning prunes  ·  '
        f'<tspan fill="{TEXT_DIM}" font-size="11">illustrative — actual learned GRN N ≫ 8</tspan>'
        f'</text>'
    )

    # ====================================================================
    # BOTTOM ZONE — Direct-effect log-FC head
    # ====================================================================
    BZ_X, BZ_Y, BZ_W, BZ_H = START_X, 812, W - 2 * START_X, 108
    parts.append(
        f'<rect x="{BZ_X}" y="{BZ_Y}" width="{BZ_W}" height="{BZ_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{LAVENDER}" stroke-width="1.5" stroke-opacity="0.55"/>'
    )
    parts.append(
        f'<text x="{BZ_X + 22}" y="{BZ_Y + 26}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="2.5">'
        f'DIRECT-EFFECT LOG-FC HEAD</text>'
    )
    BD_Y = BZ_Y + 48
    BD_BOX_H = 38
    b1_x, b1_w = BZ_X + 22, 360
    parts.append(
        f'<rect x="{b1_x}" y="{BD_Y}" width="{b1_w}" height="{BD_BOX_H}" rx="8" '
        f'fill="{SURFACE_2}" stroke="{TEXT_DIM}" stroke-width="1" stroke-opacity="0.7"/>'
    )
    # v3.1: removed italic on `z` — italic Latin letters trigger cairosvg
    # glyph substitution (same bug as I→turnstile). Plain text renders fine.
    parts.append(
        f'<text x="{b1_x + b1_w // 2}" y="{BD_Y + BD_BOX_H // 2 + 5}" fill="{TEXT_BODY}" '
        f'font-family="{FONT_BODY}" font-size="13" font-weight="600" text-anchor="middle">'
        f'z + perturbation context</text>'
    )
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
    # v3.1: removed italic on `d` — same cairosvg italic-Latin substitution
    # bug. Render dₚ as plain bold + Unicode subscript ₚ.
    out_x = a2_x + a2_w + 4
    parts.append(
        f'<text x="{out_x + 20}" y="{BD_Y + BD_BOX_H // 2 + 7}" fill="{LAVENDER}" '
        f'font-family="Inter, Arial, sans-serif" font-size="22" font-weight="700">'
        f'dₚ</text>'
    )

    # Bottom comparison line — v3 Fix 1: render `(I − W)⁻¹ dₚ` fragment via
    # mathtext PNG embed. The Stage 3a/3b line + the prose on the Stage 3c
    # line stay as SVG text.
    cmp_x = out_x + 80
    cmp_y = BD_Y - 4
    parts.append(
        f'<text x="{cmp_x}" y="{cmp_y + 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-weight="400">'
        f'<tspan font-weight="700">Stage 3a/3b predicted:</tspan> '
        f'abundance after perturbation'
        f'</text>'
    )
    # Stage 3c line — text before math + math PNG + text after
    # v3.1: removed italic on `d` (cairosvg italic-Latin substitution bug).
    # font-family changed from FONT_BODY (Arial-first) to Inter-first because
    # Arial doesn't include Unicode subscript ₚ (U+209A) and was rendering it
    # as a missing-glyph box. Inter has the full Latin-Extended subscript range.
    cmp_y_3c = cmp_y + 36
    parts.append(
        f'<text x="{cmp_x}" y="{cmp_y_3c}" fill="{TEXT_BODY}" font-family="Inter, Arial, sans-serif" '
        f'font-size="12" font-weight="400">'
        f'<tspan fill="{CYAN_HI}" font-weight="700">Stage 3c separates:</tspan>  '
        f'<tspan fill="{LAVENDER}" font-weight="700">dₚ</tspan>'
        f'<tspan fill="{TEXT_BODY}"> (direct)  +  </tspan>'
        f'</text>'
    )
    # Math fragment `(I − W)⁻¹ dₚ` as mathtext PNG.
    # Approximate left-text width: "Stage 3c separates:  dₚ (direct)  +  " ≈ 36 chars × 12pt × 0.55 = 238px
    cmp_math_x = cmp_x + 232
    cmp_math_expr = r"$(I - W)^{-1} \, d_p$"
    parts.append(math_image(cmp_math_expr, cmp_math_x + 50, cmp_y_3c - 4, fontsize=11, color="white"))
    # Text after math fragment
    parts.append(
        f'<text x="{cmp_math_x + 110}" y="{cmp_y_3c}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="12" font-weight="400">'
        f'(propagated)</text>'
    )

    # v3.1: removed the "Why this matters: causal queries vs predictive queries"
    # line entirely. It was at y=908 (perturbation-context rect ends at y=898,
    # 4px y-bbox overlap) AND redundant: the Stage 3a/3b "predicted: abundance
    # after perturbation" row + Stage 3c "separates: dₚ (direct) + (I-W)⁻¹dₚ
    # (propagated)" row above already convey the predictive-vs-causal contrast.
    # The colloquial Q&A version belongs in speaker notes, not on-slide.

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

    # v3: collision guard simplified — no more rect-split known-FP filter
    # (rect-I logic removed; equations are now <image> elements which bypass
    # text-collision detection entirely).
    collisions = check_no_text_collisions(svg, min_gap=2)
    blocking = [c for c in collisions
                if "A5 / 14" not in (c[0], c[1])
                and not c[0].startswith("Source:")
                and not c[1].startswith("Source:")]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"A5 v3 collision-guard FAIL:\n{msg}")

    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓ min_gap=2)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
