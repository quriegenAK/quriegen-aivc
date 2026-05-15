"""Build B2_adapter_verdict.svg + preview PNG.

Layout:
- Top zone: pre-registered threshold table (3 rows) with WE ARE HERE pointer on 0.57 row
- Bottom zone: per-class bar chart with 0.25 chance baseline

Section accent: cyan (the result is the moat)
Run: python3 docs/deck/assets/diagrams/_build_b2.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER, DANGER_RED,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, TEXT_DISABLED, DIVIDER,
    FONT, FONT_BODY, FONT_MATH, FONT_MONO, START_X, W, H,
    svg_open, background, header, footer, render_png,
    check_no_text_collisions,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC encoder probe: 0.57 synergy accuracy, ADAPTER_RECOMMENDED")]
    background(parts)
    header(
        parts,
        appendix_id="B2",
        section="VALIDATION EVIDENCE",
        title="Encoder Probe Verdict — ADAPTER_RECOMMENDED",
        subtitle=(
            "0.57 synergy 4-class accuracy on held-out Mimitou perturbations · "
            "pre-registered threshold lands cleanly in the adapter-strategy band · 2.27× chance baseline"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # TOP ZONE: Pre-registered threshold table  (y=232..560)
    # ====================================================================
    TZ_X, TZ_Y, TZ_W = START_X, 232, W - 2 * START_X
    parts.append(
        f'<text x="{TZ_X}" y="{TZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'PRE-REGISTERED VERDICT THRESHOLDS · ARCHITECTURE SPEC v1.1 §5</text>'
    )
    parts.append(
        f'<line x1="{TZ_X + 580}" y1="{TZ_Y-6}" x2="{TZ_X + TZ_W}" y2="{TZ_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    TBL_Y = TZ_Y + 24
    TBL_H = 320
    # Outer table card
    parts.append(
        f'<rect x="{TZ_X}" y="{TBL_Y}" width="{TZ_W}" height="{TBL_H}" rx="14" '
        f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="1.2"/>'
    )
    # Header row
    HDR_Y = TBL_Y + 40
    col_x_range = TZ_X + 64
    col_x_verdict = TZ_X + 720
    col_x_action = TZ_X + 1280
    parts.append(
        f'<text x="{col_x_range}" y="{HDR_Y}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">SYNERGY 4-CLASS ACCURACY</text>'
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
        f'<line x1="{TZ_X + 36}" y1="{HDR_Y + 16}" x2="{TZ_X + TZ_W - 36}" y2="{HDR_Y + 16}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # 3 verdict rows
    rows = [
        ("≥ 0.80",         "FROZEN_ENCODER_OK",  "Use encoder as-is",                       OK_GREEN, False),
        ("0.50 — 0.80",    "ADAPTER_RECOMMENDED","Train lightweight adapter",               CYAN_HI,  True),  # We are here
        ("&lt; 0.50",      "FINE_TUNE_REQUIRED", "Re-train encoder",                        DANGER_RED, False),
    ]
    ROW_H = 76
    row_y0 = HDR_Y + 36
    for i, (rng, verdict, action, color, highlight) in enumerate(rows):
        ry = row_y0 + i * ROW_H
        if highlight:
            # v2: 0.57 hero relocated to its own dedicated pill between the
            # range column and the verdict column (Option A from prompt).
            # Resolves the v1 collision where 0.57 at (132, 464) and
            # "0.50 — 0.80" at (160, 452) shared the same visual band.
            #
            # New layout within highlighted row:
            #   row top:     ◆ WE ARE HERE (eyebrow, top-left)
            #   row middle:  [0.50 — 0.80]   [◆ 0.57 ◆ pill]   [ADAPTER_RECOMMENDED]   [Train ...]
            parts.append(
                f'<rect x="{TZ_X + 24}" y="{ry - 2}" width="{TZ_W - 48}" height="{ROW_H}" rx="10" '
                f'fill="{CYAN}" fill-opacity="0.16" stroke="{CYAN_HI}" stroke-width="2"/>'
            )
            # WE ARE HERE eyebrow (top, small caps)
            parts.append(
                f'<text x="{TZ_X + 36}" y="{ry + 22}" fill="{CYAN_HI}" font-family="{FONT}" '
                f'font-size="10" font-weight="700" letter-spacing="2.5">◆ WE ARE HERE</text>'
            )
            # Range "0.50 — 0.80" in column 1 (unchanged position)
            parts.append(
                f'<text x="{col_x_range}" y="{ry + 54}" fill="{TEXT_BODY}" font-family="{FONT_MONO}" '
                f'font-size="22" font-weight="700">{rng}</text>'
            )
            # 0.57 hero pill — dedicated column between range and verdict.
            # Range column ends around x≈400 (col_x_range=160 + ~240px range text).
            # Verdict column starts at col_x_verdict=816.
            # Place pill at x≈460, width 200, centered text.
            pill_x, pill_w = TZ_X + 380, 200
            pill_y = ry + 14
            pill_h = ROW_H - 26
            parts.append(
                f'<rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="{pill_h}" rx="14" '
                f'fill="{CYAN_HI}" fill-opacity="0.24" stroke="{CYAN_HI}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{pill_x + pill_w / 2}" y="{pill_y + pill_h / 2 + 2}" '
                f'fill="{CYAN_HI}" font-family="{FONT}" '
                f'font-size="28" font-weight="700" '
                f'text-anchor="middle" dominant-baseline="middle">◆ 0.57</text>'
            )
            # Verdict (column 3) and Action (column 4) — unchanged
            parts.append(
                f'<text x="{col_x_verdict}" y="{ry + 54}" fill="{CYAN_HI}" font-family="{FONT}" '
                f'font-size="22" font-weight="700">{verdict}</text>'
            )
            parts.append(
                f'<text x="{col_x_action}" y="{ry + 54}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
                f'font-size="16" font-weight="700">{action}</text>'
            )
        else:
            # Non-highlighted: just the row text with subtle row divider
            parts.append(
                f'<line x1="{TZ_X + 36}" y1="{ry + ROW_H - 4}" x2="{TZ_X + TZ_W - 36}" y2="{ry + ROW_H - 4}" '
                f'stroke="{DIVIDER}" stroke-width="1" stroke-opacity="0.6"/>'
            )
            parts.append(
                f'<text x="{col_x_range}" y="{ry + 44}" fill="{TEXT_BODY}" font-family="{FONT_MONO}" '
                f'font-size="20" font-weight="600">{rng}</text>'
            )
            parts.append(
                f'<text x="{col_x_verdict}" y="{ry + 44}" fill="{color}" font-family="{FONT}" '
                f'font-size="20" font-weight="700">{verdict}</text>'
            )
            parts.append(
                f'<text x="{col_x_action}" y="{ry + 44}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
                f'font-size="15" font-weight="400">{action}</text>'
            )

    # ====================================================================
    # BOTTOM ZONE: per-class bar chart  (y=600..900)
    # ====================================================================
    CZ_Y = 588
    parts.append(
        f'<text x="{TZ_X}" y="{CZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'PER-CLASS ACCURACY · MIMITOU CRISPR HELD-OUT TEST · CHANCE = 0.25</text>'
    )
    parts.append(
        f'<line x1="{TZ_X + 720}" y1="{CZ_Y - 6}" x2="{TZ_X + TZ_W}" y2="{CZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Chart geometry
    CH_X = TZ_X + 240  # left edge of bars
    CH_Y = CZ_Y + 28   # top of chart
    CH_W = 1200        # max bar width (corresponds to 1.0 accuracy)
    BAR_H = 40
    BAR_GAP = 16

    # Bars data
    bars = [
        ("CD3E",            "TCR pathway",           0.91, OK_GREEN,   "strong baseline"),
        ("CD3E + CD4",      "double KO",             0.68, CYAN_HI,    "synergy demo target"),  # highlighted
        ("NTC",             "no perturbation",       0.39, TEXT_DIM,   ""),
        ("CD4",             "single KO",             0.39, TEXT_DIM,   ""),
    ]

    for i, (label, sublabel, val, color, badge) in enumerate(bars):
        by = CH_Y + i * (BAR_H + BAR_GAP)
        # Left axis label
        parts.append(
            f'<text x="{CH_X - 16}" y="{by + BAR_H // 2 - 4}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="15" font-weight="700" text-anchor="end" dominant-baseline="middle">{label}</text>'
        )
        parts.append(
            f'<text x="{CH_X - 16}" y="{by + BAR_H // 2 + 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="400" font-style="italic" text-anchor="end" dominant-baseline="middle">{sublabel}</text>'
        )
        # Bar
        bw = int(val * CH_W)
        is_synergy = (i == 1)
        parts.append(
            f'<rect x="{CH_X}" y="{by}" width="{bw}" height="{BAR_H}" rx="6" '
            f'fill="{color}" fill-opacity="{0.22 if is_synergy else 0.16}" stroke="{color}" '
            f'stroke-width="{2 if is_synergy else 1.2}" stroke-opacity="{0.95 if is_synergy else 0.6}"/>'
        )
        # Value label at end of bar
        parts.append(
            f'<text x="{CH_X + bw + 16}" y="{by + BAR_H // 2 + 2}" fill="{color}" font-family="{FONT}" '
            f'font-size="20" font-weight="700" dominant-baseline="middle">{val:.2f}</text>'
        )
        # Badge for highlighted row
        if badge:
            parts.append(
                f'<text x="{CH_X + bw + 80}" y="{by + BAR_H // 2 + 2}" fill="{color}" font-family="{FONT}" '
                f'font-size="12" font-weight="700" letter-spacing="1.5" dominant-baseline="middle">◆ {badge.upper()}</text>'
            )

    # X-axis tick marks (0.0, 0.25, 0.5, 0.75, 1.0)
    ax_y = CH_Y + 4 * (BAR_H + BAR_GAP) + 4
    parts.append(
        f'<line x1="{CH_X}" y1="{ax_y}" x2="{CH_X + CH_W}" y2="{ax_y}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    for v, lbl in [(0.0, "0.0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")]:
        tx = CH_X + int(v * CH_W)
        parts.append(
            f'<line x1="{tx}" y1="{ax_y}" x2="{tx}" y2="{ax_y + 6}" '
            f'stroke="{DIVIDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tx}" y="{ax_y + 24}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="400" text-anchor="middle">{lbl}</text>'
        )

    # Chance baseline (dashed vertical line at 0.25)
    chance_x = CH_X + int(0.25 * CH_W)
    parts.append(
        f'<line x1="{chance_x}" y1="{CH_Y - 8}" x2="{chance_x}" y2="{ax_y}" '
        f'stroke="{WARN_AMBER}" stroke-width="1.5" stroke-dasharray="6 4" stroke-opacity="0.85"/>'
    )
    parts.append(
        f'<text x="{chance_x}" y="{CH_Y - 14}" fill="{WARN_AMBER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="1.5" text-anchor="middle">CHANCE 0.25</text>'
    )

    # Caption / baselines line
    cap_y = ax_y + 50
    parts.append(
        f'<text x="{TZ_X}" y="{cap_y}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="14" font-weight="400">'
        f'<tspan fill="{CYAN_HI}" font-weight="700">0.57 overall 4-class</tspan> = '
        f'<tspan fill="{CYAN_HI}" font-weight="700">2.27×</tspan> chance '
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'Random projection baseline = <tspan fill="{TEXT_BODY}" font-weight="700">0.29</tspan> (sanity) '
        f'<tspan fill="{TEXT_DIM}">  ·  </tspan>'
        f'Raw TF-IDF baseline = <tspan fill="{TEXT_BODY}" font-weight="700">0.50</tspan> '
        f'<tspan fill="{TEXT_DIM}" font-style="italic">(encoder near input ceiling — right regime for adapter strategy)</tspan>'
        f'</text>'
    )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Stage 3 Part 1 verdict (2026-05-06) · architecture spec v1.1 §5 (pre-registered thresholds) · "
            "Mimitou ASAP-seq CRISPR sub-study · 74-cell post-split bootstrap CI logic · "
            "tests in tests/test_decomposed_readout.py"
        ),
        slide_handle="B2 / 12",
        handle_color=CYAN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "B2_adapter_verdict.svg"
    png_path = here / "B2_adapter_verdict_preview.png"
    svg = build_svg()
    # v2 collision-guard smoke test (per _deck_common.py helpers added 2026-05-15).
    # Filter the known-benign footer-vs-pagination false-positive (long source
    # text bounding-box estimate over-extends past the pagination's text-anchor
    # ="end" position; cairosvg actual render does not collide).
    collisions = check_no_text_collisions(svg, min_gap=4)
    blocking = [c for c in collisions if "D2 / 12" not in (c[0], c[1])
                                       and "B2 / 12" not in (c[0], c[1])
                                       and "D1 / 12" not in (c[0], c[1])
                                       and not c[0].startswith("Source:")
                                       and not c[1].startswith("Source:")]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"B2 collision-guard FAIL:\n{msg}")
    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
