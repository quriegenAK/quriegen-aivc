"""Build A4_temporal_dynamics.svg + A4_temporal_dynamics_preview.png.

Layout:
- Top: continuous-time latent trajectory plot (Bezier curve) with 5 non-uniform
       sample points at t = 0, 5, 30, 60, 180 min
- Bottom: 3-card architecture row — Neural ODE (PRIMARY), Latent SDE (FALLBACK),
          Discrete Transformer (REJECTED)

Style locked to A1 v2.

Run: python3 docs/deck/assets/diagrams/_build_a4.py
"""
from __future__ import annotations
import pathlib

# ---- Palette ----
BG          = "#070A14"
SURFACE     = "#0F1428"
SURFACE_2   = "#0B1020"
SURFACE_REJ = "#10141F"  # extra-recessed surface for REJECTED card
CYAN        = "#26DDF9"
CYAN_HI     = "#00F2FF"
PURPLE      = "#8B5CF6"
LAVENDER    = "#B47DF0"
TEXT_TITLE  = "#F7FAFF"
TEXT_BODY   = "#EAF6FF"
TEXT_MUTED  = "#A8B4C2"
TEXT_DIM    = "#94A3B8"
TEXT_DISABLED = "#5B6478"
OK_GREEN    = "#4ADE80"
WARN_AMBER  = "#FBBF24"
DANGER_RED  = "#FF4D6D"
DIVIDER     = "#1A2235"

W, H = 1920, 1080
START_X = 96
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
FONT_BODY = "Arial, Inter, 'Helvetica Neue', sans-serif"

# Per-timepoint biology colors
TP_COLORS = {
    0:   TEXT_MUTED,    # neutral grey — baseline
    5:   CYAN_HI,       # early signaling (phospho-ready for Phase 2)
    30:  PURPLE,        # transcriptional onset
    60:  LAVENDER,      # peak response window
    180: TEXT_BODY,     # stable phenotype (pale)
}
TP_LABELS = {
    0:   ("0 min",   "Baseline",            "pre-perturbation state"),
    5:   ("5 min",   "Early signaling",     "phospho-active · RNA latent"),
    30:  ("30 min",  "Transcriptional onset","RNA dynamics begin"),
    60:  ("60 min",  "Peak response",       "maximum activation window"),
    180: ("180 min", "Stable phenotype",    "RNA + Protein equilibrium"),
}
TP_SEQUENCE = [0, 5, 30, 60, 180]


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
        f'<linearGradient id="curvegrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{TEXT_MUTED}"/>'
        f'<stop offset="15%" stop-color="{CYAN_HI}"/>'
        f'<stop offset="45%" stop-color="{PURPLE}"/>'
        f'<stop offset="70%" stop-color="{LAVENDER}"/>'
        f'<stop offset="100%" stop-color="{TEXT_BODY}"/>'
        f'</linearGradient>'
        '</defs>'
        f'<rect width="{W}" height="{H}" fill="url(#glow1)"/>'
        f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>'
    )


def catmull_rom_to_bezier_path(points: list[tuple[float, float]]) -> str:
    """Convert a list of (x, y) points to an SVG path using cubic Bezier
    segments derived from a Catmull-Rom spline. Produces a smooth curve
    that interpolates the points (visual: smooth continuous dynamics,
    not a polyline)."""
    if len(points) < 2:
        return ""
    # Duplicate first and last points for the spline phantom-control trick
    p = [points[0]] + list(points) + [points[-1]]
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        # Catmull-Rom → Bezier control points
        c1x = p1[0] + (p2[0] - p0[0]) / 6
        c1y = p1[1] + (p2[1] - p0[1]) / 6
        c2x = p2[0] - (p3[0] - p1[0]) / 6
        c2y = p2[1] - (p3[1] - p1[1]) / 6
        cmds.append(f"C {c1x:.2f} {c1y:.2f}, {c2x:.2f} {c2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}")
    return " ".join(cmds)


def build_svg() -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="AIVC temporal dynamics via Neural ODE">'
    )
    background(parts)
    header(
        parts,
        appendix_id="A4",
        section="ARCHITECTURE DEPTH",
        title="Temporal Dynamics — Continuous-Time via Neural ODE",
        subtitle=(
            "Cells respond on irregular timescales · 0 → 180 min · "
            "Neural ODE handles non-uniform sampling natively — no discretization, no interpolation artifacts"
        ),
    )

    # ====================================================================
    # TOP ZONE: Trajectory plot  (x=96..1824, y=232..588)
    # ====================================================================
    TZ_X, TZ_Y, TZ_W, TZ_H = 96, 232, 1728, 356
    parts.append(
        f'<text x="{TZ_X}" y="{TZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">LATENT TRAJECTORY · z(t)</text>'
    )
    parts.append(
        f'<line x1="{TZ_X + 200}" y1="{TZ_Y-6}" x2="{TZ_X + TZ_W}" y2="{TZ_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Plot area (inside zone, leaving padding for axes + sample annotations)
    PX0, PY0 = TZ_X + 60, TZ_Y + 36     # top-left of plot area
    PW, PH   = TZ_W - 100, 240          # plot dimensions
    PX1, PY1 = PX0 + PW, PY0 + PH

    # Map t (min) → screen x. Non-uniform per spec — gives 5 min real visual space
    # while still ending at 180.
    # Use a piecewise-linear map: t=0→PX0, t=5→0.18*PW, t=30→0.45*PW, t=60→0.62*PW, t=180→PX1
    T_TO_X = {
        0:   PX0 + 0.00 * PW,
        5:   PX0 + 0.18 * PW,
        30:  PX0 + 0.42 * PW,
        60:  PX0 + 0.62 * PW,
        180: PX0 + 0.97 * PW,
    }
    # Latent magnitude per timepoint (illustrative biology response curve)
    Z_AT = {
        0:   0.10,
        5:   0.38,
        30:  0.72,
        60:  0.94,
        180: 0.82,
    }
    def t_to_xy(t):
        return (T_TO_X[t], PY1 - Z_AT[t] * PH)

    # Axes
    # y-axis line
    parts.append(
        f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" '
        f'stroke="{DIVIDER}" stroke-width="1.5"/>'
    )
    # x-axis line
    parts.append(
        f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" '
        f'stroke="{DIVIDER}" stroke-width="1.5"/>'
    )
    # y-axis label (vertical)
    parts.append(
        f'<text x="{PX0 - 24}" y="{PY0 + PH/2}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="13" font-weight="700" letter-spacing="2" text-anchor="middle" '
        f'transform="rotate(-90, {PX0 - 24}, {PY0 + PH/2})">LATENT  z(t)</text>'
    )
    # x-axis title under axis
    parts.append(
        f'<text x="{PX0 + PW/2}" y="{PY1 + 78}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3" text-anchor="middle">'
        f'TIME · MINUTES POST-PERTURBATION (non-uniform spacing)</text>'
    )

    # Horizontal grid (3 subtle lines)
    for frac in (0.25, 0.5, 0.75):
        gy = PY1 - frac * PH
        parts.append(
            f'<line x1="{PX0}" y1="{gy}" x2="{PX1}" y2="{gy}" '
            f'stroke="{DIVIDER}" stroke-width="0.5" stroke-opacity="0.5" '
            f'stroke-dasharray="2 6"/>'
        )

    # Smooth trajectory curve (Catmull-Rom → cubic Bezier)
    pts = [t_to_xy(t) for t in TP_SEQUENCE]
    path_d = catmull_rom_to_bezier_path(pts)
    # Glow under curve (wider, transparent)
    parts.append(
        f'<path d="{path_d}" fill="none" stroke="{CYAN}" stroke-width="14" '
        f'stroke-opacity="0.16" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    # Actual curve with gradient stroke
    parts.append(
        f'<path d="{path_d}" fill="none" stroke="url(#curvegrad)" stroke-width="3.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    # Sample-point markers + tick + label
    for t in TP_SEQUENCE:
        x, y = t_to_xy(t)
        c = TP_COLORS[t]
        # Vertical tick on x-axis
        parts.append(
            f'<line x1="{x}" y1="{PY1}" x2="{x}" y2="{PY1 + 8}" '
            f'stroke="{c}" stroke-width="2"/>'
        )
        # Dotted vertical guide from x-axis to point
        parts.append(
            f'<line x1="{x}" y1="{PY1}" x2="{x}" y2="{y}" '
            f'stroke="{c}" stroke-width="1" stroke-opacity="0.30" stroke-dasharray="2 4"/>'
        )
        # Glow halo around marker
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="14" fill="{c}" fill-opacity="0.18"/>'
        )
        # Solid marker
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="7" fill="{c}" stroke="{BG}" stroke-width="2"/>'
        )
        # Time label below x-axis
        parts.append(
            f'<text x="{x}" y="{PY1 + 30}" fill="{c}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" text-anchor="middle">{TP_LABELS[t][0]}</text>'
        )
        # Biology micro-label below time
        parts.append(
            f'<text x="{x}" y="{PY1 + 50}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="700" text-anchor="middle">{TP_LABELS[t][1]}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{PY1 + 65}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="10" font-weight="400" font-style="italic" text-anchor="middle">{TP_LABELS[t][2]}</text>'
        )

    # ODE equation overlay (top-left corner of plot)
    EQ_OX, EQ_OY = PX0 + 16, PY0 + 30
    parts.append(
        f'<rect x="{EQ_OX - 14}" y="{EQ_OY - 26}" width="270" height="48" rx="10" '
        f'fill="{SURFACE_2}" fill-opacity="0.85" stroke="{CYAN}" stroke-width="1" stroke-opacity="0.45"/>'
    )
    parts.append(
        f'<text x="{EQ_OX}" y="{EQ_OY}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="22" font-weight="700">'
        f'<tspan fill="{TEXT_DIM}">d</tspan>'
        f'<tspan font-style="italic">z</tspan>'
        f'<tspan fill="{TEXT_DIM}"> / </tspan>'
        f'<tspan fill="{TEXT_DIM}">d</tspan>'
        f'<tspan font-style="italic">t</tspan>'
        f'<tspan fill="{TEXT_DIM}"> = </tspan>'
        f'<tspan fill="{CYAN}" font-weight="700">f<tspan font-size="14" baseline-shift="-30%">θ</tspan></tspan>'
        f'<tspan fill="{TEXT_MUTED}" font-style="italic">(z, p, t)</tspan>'
        f'</text>'
    )

    # ====================================================================
    # BOTTOM ZONE: 3 architecture cards (PRIMARY / FALLBACK / REJECTED)
    # x=96..1824, y=620..900
    # ====================================================================
    CZ_X, CZ_Y, CZ_W, CZ_H = 96, 620, 1728, 296
    parts.append(
        f'<text x="{CZ_X}" y="{CZ_Y - 12}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">ARCHITECTURE CHOICE</text>'
    )
    parts.append(
        f'<line x1="{CZ_X + 220}" y1="{CZ_Y - 18}" x2="{CZ_X + CZ_W}" y2="{CZ_Y - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    CARD_W = 560
    CARD_GAP = 24
    # Card 1: Neural ODE — PRIMARY (filled brand)
    # Card 2: Latent SDE — FALLBACK (outlined muted)
    # Card 3: Discrete Transformer — REJECTED (greyed with red X)
    cards = [
        {
            "title":   "Neural ODE",
            "status":  "PRIMARY",
            "status_color": CYAN_HI,
            "fill_color":   CYAN,
            "fill_opacity": "0.18",
            "stroke_color": CYAN_HI,
            "stroke_width": "2",
            "subtitle": "continuous-time backbone",
            "items": [
                ("✓", "Continuous time",            OK_GREEN, TEXT_BODY),
                ("✓", "Irregular sampling native",   OK_GREEN, TEXT_BODY),
                ("✓", "Deterministic trajectories",  OK_GREEN, TEXT_BODY),
                ("✓", "Reuses ≈130K-param adapter",  OK_GREEN, TEXT_BODY),
                ("✓", "torchdiffeq · spec §4",       OK_GREEN, TEXT_MUTED),
            ],
            "tag_color": CYAN_HI,
            "tag_label": "PRIMARY",
        },
        {
            "title":   "Latent SDE",
            "status":  "FALLBACK",
            "status_color": WARN_AMBER,
            "fill_color":   WARN_AMBER,
            "fill_opacity": "0.05",
            "stroke_color": WARN_AMBER,
            "stroke_width": "1.5",
            "subtitle": "documented downgrade path",
            "items": [
                ("◐", "Stochastic dynamics support", WARN_AMBER, TEXT_BODY),
                ("◐", "Same f_θ drift, zero-init noise", WARN_AMBER, TEXT_BODY),
                ("◐", "Trigger: NaN >3 / 100 batches",   WARN_AMBER, TEXT_MUTED),
                ("◐", "Trigger: plateau >5 epochs",      WARN_AMBER, TEXT_MUTED),
                ("◐", "Trigger: spectral radius >5.0",   WARN_AMBER, TEXT_MUTED),
            ],
            "tag_color": WARN_AMBER,
            "tag_label": "FALLBACK",
        },
        {
            "title":   "Discrete Transformer",
            "status":  "REJECTED",
            "status_color": DANGER_RED,
            "fill_color":   "#000000",
            "fill_opacity": "0.20",
            "stroke_color": DANGER_RED,
            "stroke_width": "1.2",
            "subtitle": "architectural invariant violated",
            "items": [
                ("✗", "Fixed timesteps assumption",     DANGER_RED, TEXT_DISABLED),
                ("✗", "Interpolation artifacts",        DANGER_RED, TEXT_DISABLED),
                ("✗", "Loses 5-min ⟶ 30-min spacing info", DANGER_RED, TEXT_DISABLED),
                ("✗", "Architectural invariant rejection", DANGER_RED, TEXT_DISABLED),
                ("✗", "Spec v1.1 §2 · invariant fixed",  DANGER_RED, TEXT_DISABLED),
            ],
            "tag_color": DANGER_RED,
            "tag_label": "REJECTED",
        },
    ]

    for i, card in enumerate(cards):
        cx0 = CZ_X + i * (CARD_W + CARD_GAP)
        # Greyed card surface (REJECTED) gets the recessed background
        bg_color = SURFACE_REJ if card["status"] == "REJECTED" else SURFACE
        parts.append(
            f'<rect x="{cx0}" y="{CZ_Y}" width="{CARD_W}" height="{CZ_H}" rx="14" '
            f'fill="{bg_color}" stroke="{card["stroke_color"]}" '
            f'stroke-width="{card["stroke_width"]}" stroke-opacity="0.6"/>'
        )
        # Brand-tinted overlay
        parts.append(
            f'<rect x="{cx0}" y="{CZ_Y}" width="{CARD_W}" height="{CZ_H}" rx="14" '
            f'fill="{card["fill_color"]}" fill-opacity="{card["fill_opacity"]}" '
            f'stroke="none"/>'
        )

        # Status tag (top-right corner)
        tag_w = 110
        parts.append(
            f'<rect x="{cx0 + CARD_W - tag_w - 16}" y="{CZ_Y + 16}" width="{tag_w}" height="24" rx="12" '
            f'fill="{card["tag_color"]}" fill-opacity="0.22" stroke="{card["tag_color"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{cx0 + CARD_W - tag_w/2 - 16}" y="{CZ_Y + 32}" fill="{card["tag_color"]}" '
            f'font-family="{FONT}" font-size="11" font-weight="700" letter-spacing="2" '
            f'text-anchor="middle">{card["tag_label"]}</text>'
        )

        # Title
        title_color = TEXT_TITLE if card["status"] != "REJECTED" else TEXT_DISABLED
        parts.append(
            f'<text x="{cx0 + 24}" y="{CZ_Y + 46}" fill="{card["status_color"]}" font-family="{FONT}" '
            f'font-size="11" font-weight="700" letter-spacing="2.5">0{i+1}  ·  {card["status"]}</text>'
        )
        parts.append(
            f'<text x="{cx0 + 24}" y="{CZ_Y + 86}" fill="{title_color}" font-family="{FONT}" '
            f'font-size="28" font-weight="700">{card["title"]}</text>'
        )
        sub_color = TEXT_MUTED if card["status"] != "REJECTED" else TEXT_DISABLED
        parts.append(
            f'<text x="{cx0 + 24}" y="{CZ_Y + 112}" fill="{sub_color}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400" font-style="italic">{card["subtitle"]}</text>'
        )

        # Item bullets
        item_y0 = CZ_Y + 144
        for j, (icon, text, icon_color, text_color) in enumerate(card["items"]):
            iy = item_y0 + j * 28
            parts.append(
                f'<text x="{cx0 + 24}" y="{iy}" fill="{icon_color}" font-family="{FONT}" '
                f'font-size="16" font-weight="700">{icon}</text>'
            )
            parts.append(
                f'<text x="{cx0 + 50}" y="{iy}" fill="{text_color}" font-family="{FONT_BODY}" '
                f'font-size="14" font-weight="400">{text}</text>'
            )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 §4 (Neural ODE backbone) + §7.1 (Latent SDE fallback, trigger conditions) + §2 (Transformer-rejection invariant) · "
            "QurieSeq Phase 1 design (Thiago confirmation, 2026-05-12) · "
            "torchdiffeq + torchsde libraries"
        ),
        slide_handle="A4 / 12",
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
    svg_path = here / "A4_temporal_dynamics.svg"
    png_path = here / "A4_temporal_dynamics_preview.png"
    svg = build_svg()
    svg_path.write_text(svg)
    print(f"wrote {svg_path} ({len(svg)} bytes)")
    build_png(svg_path, png_path)
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes)")
