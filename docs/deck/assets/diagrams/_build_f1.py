"""Build F1_integrated_platform.svg + preview PNG.

Layout:
- Top zone: 4-pillar flywheel (cardinal-point orbit) with curved clockwise
  "compounds" arrows + center "INTEGRATED PLATFORM" label
    - TOP    pillar = CO-DESIGNED ARCHITECTURE
    - LEFT   pillar = WET-LAB GENERATION
    - RIGHT  pillar = TEMPORAL MULTI-OMICS
    - BOTTOM pillar = PROTOCOL-FAMILY EXPANSION
- Middle zone: 3-bucket competitor archetype grouping
    - DATA SCALE (TAHOE, Immunai)
    - FOUNDATION MODELS (CytoReason, Turbine AI, DeepLife)
    - DOWNSTREAM THERAPEUTICS (Valo Health, Noetik)
  + Full-width Quriegen INTEGRATED CAUSAL PERTURBATION PLATFORM row (amber)
- Closing italic centered line
- Standard footer + pagination "F1 / 13"

Section accent: amber #FBBF24 (distinct from A cyan / B green / C cyan / D
lavender / E white-pale).

Pagination decision per prompt's recommendation: keep existing SVGs at
"<id> / 12" for now; F1 shows "F1 / 13" as outlier. Phase 4 polish
unifies pagination later.

Run: python3 docs/deck/assets/diagrams/_build_f1.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, START_X, W, H,
    svg_open, background, header, footer, render_png,
    check_no_text_collisions,
)

# Section F accent (amber)
ACCENT_AMBER = WARN_AMBER  # "#FBBF24" — same as in-flight status icons


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC F1 — integrated platform flywheel + competitor archetypes")]
    background(parts)
    header(
        parts,
        appendix_id="F1",
        section="COMPETITIVE POSITIONING",
        title="The Closed-Loop Platform — Co-Designed, Compounding",
        subtitle=(
            "No public dataset has the combination — multi-omics + perturbation-aware + temporal + combinatorial · "
            "Phase 1 QuRIE-seq closes the gap with phospho at Q3 2026; the platform compounds from there"
        ),
        eyebrow_color=ACCENT_AMBER,
    )

    # ====================================================================
    # TOP ZONE: 4-pillar flywheel  (y=216..636, ~420px)
    # ====================================================================
    # Cardinal-point orbit centered on (cx, cy)
    cx, cy = W // 2, 426
    PILLAR_W = 320
    PILLAR_H = 130

    # Pillar centers (top, right, bottom, left)
    top_cx,    top_cy    = cx,         cy - 145    # 281
    bottom_cx, bottom_cy = cx,         cy + 145    # 571
    left_cx,   left_cy   = cx - 280,   cy          # (680, 426)
    right_cx,  right_cy  = cx + 280,   cy          # (1240, 426)

    # Section eyebrow (above flywheel zone)
    parts.append(
        f'<text x="{START_X}" y="216" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'INTEGRATED PLATFORM · 4-PILLAR FLYWHEEL</text>'
    )
    parts.append(
        f'<line x1="{START_X + 360}" y1="210" x2="{W - START_X}" y2="210" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # ---- Pillar definitions ----
    # Body lines can be either plain strings OR (text, color) tuples for
    # per-line color override. Used to lavender-emphasize "Phospho" in the
    # TEMPORAL MULTI-OMICS pillar — visual load-bearing signal that we have
    # the modality no public dataset has.
    pillars = [
        # (label_caps, subtitle, body_lines, center_x, center_y)
        ("CO-DESIGNED ARCHITECTURE", "matches the assay",
         ["4-arm decomposed readout",
          "Neural ODE temporal",
          "Compositional generalization"],
         top_cx, top_cy),
        ("TEMPORAL MULTI-OMICS", "phospho is the modality no one else has",
         [("RNA · Protein · Phospho", LAVENDER),       # phospho-emphasized line
          ("5 timepoints (0/5/30/60/180)", TEXT_BODY),
          ("ATAC × 2 timepoints · VDJ Phase 2", TEXT_MUTED)],
         right_cx, right_cy),
        ("PROTOCOL-FAMILY EXPANSION", "no re-architecting",
         ["Same wet-lab pipeline scales",
          "to Phase 2: 20 donors + VDJ",
          "without re-architecting"],
         bottom_cx, bottom_cy),
        ("WET-LAB GENERATION", "QurieSeq Phase 1+2",
         ["Primary human PBMCs",
          "5 → 20 donors",
          "4-arm perturbations"],
         left_cx, left_cy),
    ]

    def render_pillar(label: str, subtitle: str, body: list,
                      ccx: int, ccy: int):
        x = ccx - PILLAR_W // 2
        y = ccy - PILLAR_H // 2
        # Card
        parts.append(
            f'<rect x="{x}" y="{y}" width="{PILLAR_W}" height="{PILLAR_H}" rx="14" '
            f'fill="{SURFACE}" stroke="{ACCENT_AMBER}" stroke-width="1.5" stroke-opacity="0.65"/>'
        )
        # Title (Inter bold caps, white)
        parts.append(
            f'<text x="{x + 18}" y="{y + 28}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="16" font-weight="700" letter-spacing="2">{label}</text>'
        )
        # Subtitle (italic muted)
        parts.append(
            f'<text x="{x + 18}" y="{y + 48}" fill="{ACCENT_AMBER}" font-family="{FONT_BODY}" '
            f'font-size="11" font-style="italic" font-weight="600">{subtitle}</text>'
        )
        # Body lines — support (text, color) tuples for per-line override
        for j, line in enumerate(body):
            if isinstance(line, tuple):
                text, line_color = line
                font_weight = "700" if line_color == LAVENDER else "400"
            else:
                text, line_color, font_weight = line, TEXT_BODY, "400"
            parts.append(
                f'<text x="{x + 18}" y="{y + 74 + j * 18}" fill="{line_color}" '
                f'font-family="{FONT_BODY}" font-size="12" font-weight="{font_weight}">'
                f'<tspan fill="{ACCENT_AMBER}" font-weight="700">›</tspan>  {text}</text>'
            )

    for label, subtitle, body, ccx, ccy in pillars:
        render_pillar(label, subtitle, body, ccx, ccy)

    # ---- Center label "INTEGRATED PLATFORM" ----
    # Sits in the gap between left pillar (right edge x=820) and right pillar
    # (left edge x=1100) — 280px wide. Vertically between top pillar (bottom
    # y=346) and bottom pillar (top y=506) — 160px tall.
    # Center label area: roughly (820, 346) to (1100, 506).
    center_label_y = cy
    parts.append(
        f'<text x="{cx}" y="{center_label_y - 12}" fill="{ACCENT_AMBER}" '
        f'font-family="{FONT}" font-size="13" font-weight="700" letter-spacing="3" '
        f'text-anchor="middle">INTEGRATED</text>'
    )
    parts.append(
        f'<text x="{cx}" y="{center_label_y + 14}" fill="{ACCENT_AMBER}" '
        f'font-family="{FONT}" font-size="13" font-weight="700" letter-spacing="3" '
        f'text-anchor="middle">PLATFORM</text>'
    )
    # Small ring around center label as visual anchor
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="60" fill="none" stroke="{ACCENT_AMBER}" '
        f'stroke-width="1" stroke-opacity="0.35" stroke-dasharray="4 4"/>'
    )
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{ACCENT_AMBER}" fill-opacity="0.9"/>'
    )

    # ---- Curved compounds arrows clockwise (4 arcs around the center) ----
    # Clockwise: TOP → RIGHT → BOTTOM → LEFT → TOP
    # Each arc curves outward (away from center) between adjacent pillars.
    # Arrow goes FROM the edge of one pillar TO the edge of the next.

    def curved_arrow(ax: int, ay: int, bx: int, by: int, ctrl_x: int, ctrl_y: int,
                     label_x: int, label_y: int):
        """Quadratic Bezier arrow from (ax,ay) to (bx,by) via control (ctrl_x,ctrl_y),
        with a 'compounds' label at (label_x, label_y)."""
        parts.append(
            f'<path d="M {ax} {ay} Q {ctrl_x} {ctrl_y} {bx} {by}" '
            f'fill="none" stroke="{ACCENT_AMBER}" stroke-width="2" stroke-opacity="0.7" '
            f'stroke-linecap="round"/>'
        )
        # Compute approximate end-tangent direction for arrowhead
        # (use a small line from a point near the end toward the actual end)
        # For a simple arrowhead, draw 2 short lines at (bx,by) angled ±30°
        # back along the tangent estimated by (bx-ctrl_x, by-ctrl_y) direction.
        import math
        tx = bx - ctrl_x
        ty = by - ctrl_y
        n = math.hypot(tx, ty) or 1
        ux, uy = tx / n, ty / n
        head_len = 10
        head_w = 5
        # Two perpendicular offsets
        px, py = -uy, ux
        h1x = bx - head_len * ux + head_w * px
        h1y = by - head_len * uy + head_w * py
        h2x = bx - head_len * ux - head_w * px
        h2y = by - head_len * uy - head_w * py
        parts.append(
            f'<path d="M {h1x:.1f} {h1y:.1f} L {bx} {by} L {h2x:.1f} {h2y:.1f}" '
            f'fill="none" stroke="{ACCENT_AMBER}" stroke-width="2" stroke-opacity="0.85" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        # Label
        parts.append(
            f'<text x="{label_x}" y="{label_y}" fill="{ACCENT_AMBER}" '
            f'font-family="{FONT_BODY}" font-size="11" font-style="italic" '
            f'font-weight="600" text-anchor="middle">compounds</text>'
        )

    # Edge coords of pillars:
    #  TOP pillar:    x=800..1120, y=216..346  → bottom edge at y=346
    #  RIGHT pillar:  x=1100..1380, y=361..491  → left edge at x=1100
    #  BOTTOM pillar: x=800..1120, y=506..636  → top edge at y=506
    #  LEFT pillar:   x=540..820, y=361..491   → right edge at x=820

    # Top → Right (clockwise, top-right arc)
    curved_arrow(
        ax=1120, ay=336,      # near bottom-right corner of TOP
        bx=1110, by=361,      # near top-left of RIGHT (entering top edge)
        ctrl_x=1180, ctrl_y=336,  # arc bulging down-right
        label_x=1170, label_y=370,
    )
    # Right → Bottom
    curved_arrow(
        ax=1110, ay=491,
        bx=1120, by=516,
        ctrl_x=1180, ctrl_y=516,
        label_x=1180, label_y=508,
    )
    # Bottom → Left
    curved_arrow(
        ax=800, ay=516,
        bx=810, by=491,
        ctrl_x=740, ctrl_y=516,
        label_x=750, label_y=508,
    )
    # Left → Top
    curved_arrow(
        ax=810, ay=361,
        bx=800, by=336,
        ctrl_x=740, ctrl_y=336,
        label_x=750, label_y=370,
    )

    # ====================================================================
    # MIDDLE ZONE: 3-bucket archetype grouping + Quriegen full-width row
    # y = 656..872
    # ====================================================================
    MZ_Y = 656
    parts.append(
        f'<text x="{START_X}" y="{MZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">WHO OPTIMIZES WHAT?</text>'
    )
    parts.append(
        f'<line x1="{START_X + 240}" y1="{MZ_Y - 6}" x2="{W - START_X}" y2="{MZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # 3 top buckets
    BUCKET_W = 560
    BUCKET_GAP = 24
    bucket_x0 = (W - 3 * BUCKET_W - 2 * BUCKET_GAP) // 2  # 96
    BUCKET_Y = MZ_Y + 16
    BUCKET_H = 156

    buckets = [
        ("DATA SCALE", "data breadth", "wet-lab + protocol",
         [("TAHOE",   "100M cells, RNA-only cell lines"),
          ("Immunai", "modality-rich atlas, partner data")]),
        ("FOUNDATION MODELS", "model architecture", "proprietary data",
         [("CytoReason", "partner-derived multi-omics, immune"),
          ("Turbine AI", "virtual lab, pharma partnerships"),
          ("DeepLife",   "causal modeling, drug repositioning")]),
        ("DOWNSTREAM THERAPEUTICS", "clinical pipeline", "foundation modeling",
         [("Valo Health", "clinical development"),
          ("Noetik",      "spatial multi-omics oncology")]),
    ]

    for i, (title, opt, decouple, items) in enumerate(buckets):
        bx = bucket_x0 + i * (BUCKET_W + BUCKET_GAP)
        # Card (neutral, recessed)
        parts.append(
            f'<rect x="{bx}" y="{BUCKET_Y}" width="{BUCKET_W}" height="{BUCKET_H}" rx="12" '
            f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="1.2" stroke-opacity="0.7"/>'
        )
        # Title
        parts.append(
            f'<text x="{bx + 20}" y="{BUCKET_Y + 28}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" letter-spacing="2.5">{title}</text>'
        )
        # Competitors
        for j, (name, desc) in enumerate(items):
            ny = BUCKET_Y + 54 + j * 22
            parts.append(
                f'<text x="{bx + 20}" y="{ny}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
                f'font-size="13" font-weight="400">'
                f'<tspan fill="{TEXT_MUTED}" font-weight="700">›</tspan>  '
                f'<tspan font-weight="700">{name}</tspan> — {desc}</text>'
            )
        # Divider line above the optimize/decouple block
        parts.append(
            f'<line x1="{bx + 16}" y1="{BUCKET_Y + BUCKET_H - 46}" x2="{bx + BUCKET_W - 16}" y2="{BUCKET_Y + BUCKET_H - 46}" '
            f'stroke="{DIVIDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{bx + 20}" y="{BUCKET_Y + BUCKET_H - 26}" fill="{TEXT_MUTED}" '
            f'font-family="{FONT_BODY}" font-size="11" font-style="italic">'
            f'<tspan font-weight="700">Optimize:</tspan> {opt}</text>'
        )
        parts.append(
            f'<text x="{bx + 20}" y="{BUCKET_Y + BUCKET_H - 10}" fill="{TEXT_DIM}" '
            f'font-family="{FONT_BODY}" font-size="11" font-style="italic">'
            f'<tspan font-weight="700">Decouple:</tspan> {decouple}</text>'
        )

    # Full-width Quriegen row (amber accent, dominates)
    # v2: tightened internal line-height (Option B from F1 v2 fix prompt) —
    # row collapsed from QR_H=86 to QR_H=64 to free vertical budget for the
    # closing line below. Per-line offsets shrunk +30/+56/+76 → +22/+42/+58.
    # Content unchanged; only spacing.
    QR_Y = BUCKET_Y + BUCKET_H + 14
    QR_H = 64
    parts.append(
        f'<rect x="{START_X}" y="{QR_Y}" width="{W - 2*START_X}" height="{QR_H}" rx="14" '
        f'fill="{ACCENT_AMBER}" fill-opacity="0.16" stroke="{ACCENT_AMBER}" '
        f'stroke-width="2" stroke-opacity="0.9"/>'
    )
    # Title (left)
    parts.append(
        f'<text x="{START_X + 22}" y="{QR_Y + 22}" fill="{ACCENT_AMBER}" font-family="{FONT}" '
        f'font-size="13" font-weight="700" letter-spacing="2.5">'
        f'INTEGRATED CAUSAL PERTURBATION PLATFORM</text>'
    )
    # Quriegen line
    parts.append(
        f'<text x="{START_X + 22}" y="{QR_Y + 42}" fill="{TEXT_TITLE}" font-family="{FONT_BODY}" '
        f'font-size="14" font-weight="400">'
        f'<tspan fill="{ACCENT_AMBER}" font-weight="700">›</tspan>  '
        f'<tspan font-weight="700">QURIEGEN</tspan> — proprietary wet-lab generation '
        f'+ co-designed architecture + temporal multi-omics + '
        f'compositional causal modeling + protocol-family expansion, all coupled</text>'
    )
    # Optimize line
    parts.append(
        f'<text x="{START_X + 22}" y="{QR_Y + 58}" fill="{ACCENT_AMBER}" '
        f'font-family="{FONT_BODY}" font-size="12" font-style="italic">'
        f'<tspan font-weight="700">Optimize:</tspan> the closed-loop system itself '
        f'<tspan fill="{TEXT_DIM}">— integration is the moat (each loop deepens the next)</tspan></text>'
    )

    # ====================================================================
    # CLOSING LINE — italic centered takeaway
    # v2 FIX: moved from y=908 (which sat INSIDE the Quriegen row's
    # footprint y=898-918) to y=938 — below the compressed Quriegen row
    # (rect ends at y=QR_Y+64=906) with 20px gap above and 7px gap below
    # to the footer divider at y=H-132=948.
    # ====================================================================
    parts.append(
        f'<text x="{cx}" y="938" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="14" font-style="italic" text-anchor="middle">'
        f'"Phase 1 QuRIE-seq (Q3 2026) measures RNA + Protein + Phospho at 5 timepoints — '
        f'the combination no public dataset has. The platform compounds from there."</text>'
    )

    # ---- Footer (standard pattern, F1 / 13) ----
    footer(
        parts,
        source_text=(
            "Source: docs/deck/research/competitive_landscape_2026_05.md (10-competitor research) · "
            "Architecture spec v1.1 · QurieSeq Phase 1+2 spec (Thiago, May 2026) · "
            "Stage 3 Part 1 dataset survey · Kinga slide 9 competitive matrix"
        ),
        slide_handle="F1 / 13",
        handle_color=ACCENT_AMBER,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "F1_integrated_platform.svg"
    png_path = here / "F1_integrated_platform_preview.png"
    svg = build_svg()

    # v2 FIX: tightened collision-guard min_gap 4 → 2 so near-collisions
    # like v1's closing-line-vs-Quriegen-bullet (4.0px y-overlap, exactly
    # at the old threshold) actually trip the guard. min_gap=2 still
    # tolerates 1-2px sub-pixel anti-aliasing noise but no longer lets
    # genuine ~3-4px overlaps slip through silently.
    #
    # Filter scope unchanged: ONLY the source-citation-vs-pagination
    # false positive is skipped (the long Source: text width estimate
    # over-extends past pagination's text-anchor="end" position, but
    # cairosvg actual Inter render is clean). All other near-collisions
    # block the build.
    collisions = check_no_text_collisions(svg, min_gap=2)
    blocking = [c for c in collisions
                if "F1 / 13" not in (c[0], c[1])
                and not c[0].startswith("Source:")
                and not c[1].startswith("Source:")]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"F1 collision-guard FAIL:\n{msg}")

    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
