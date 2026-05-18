"""Build D1_quarterly_roadmap.svg + preview PNG.

Layout: full-slide Gantt with 4 swimlanes × 10 quarters (Q3 2026 → Q4 2028),
7 diamond milestone markers at top, Q4 2026 BTK+JAK demo emphasized as anchor.

Swimlanes:
- WET LAB (Phase 1 / Phase 2 / Phase 3)
- MODEL (Stage 3a / 3b / 3c / 4 / 5)
- DRUG PIPELINES (Pipeline 1 / Pipeline 2 + target validation)
- PUBLICATIONS (Stage 3 verdict + BTK+JAK demo deck / Stage 4+5 peer-reviewed)

Section accent: lavender #B47DF0
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER, DANGER_RED,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, FONT_MONO, START_X, W, H,
    svg_open, background, header, footer, render_png,
    check_no_text_collisions,
)

# 10 quarters Q3'26 → Q4'28
QUARTERS = [
    "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27",
    "Q4'27", "Q1'28", "Q2'28", "Q3'28", "Q4'28",
]
N_Q = len(QUARTERS)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC D1 quarterly roadmap Q3 2026 to Q4 2028")]
    background(parts)
    header(
        parts,
        appendix_id="D1",
        section="ROADMAP",
        title="11 Quarters · 5 Stages · 2 Drug Pipelines · One Platform Plan",
        subtitle=(
            "Stage 3 ships against QurieSeq Phase 1 · Stage 4 + 5 build the platform out as Phase 2 lands · "
            "every milestone tied to a specific quarter and dependency"
        ),
        eyebrow_color=LAVENDER,
    )

    # ====================================================================
    # GANTT GEOMETRY
    # ====================================================================
    # v2: push gantt down by 36px to give the staggered milestone label rows
    # (now BELOW the diamonds) breathing room without crowding the header
    # subtitle. Total gantt height (4 lanes × 120 + 3 × 16 = 528) still
    # comfortably clears the footer at y=948.
    GT_X = START_X + 200      # leave 200px for lane labels
    GT_Y = 284                # top of gantt area (was 248 in v1)
    GT_W = W - GT_X - START_X # ≈ 1528
    Q_W = GT_W // N_Q         # ≈ 152

    LANE_H = 120
    LANE_GAP = 16
    LANES = [
        ("WET LAB",         "QurieSeq data delivery",          CYAN_HI),
        ("MODEL",           "Architecture stages",             LAVENDER),
        ("DRUG PIPELINES",  "Target identification → valid.",  OK_GREEN),
        ("PUBLICATIONS",    "Demos + peer-reviewed papers",    TEXT_BODY),
    ]
    N_LANES = len(LANES)
    GANTT_H = N_LANES * LANE_H + (N_LANES - 1) * LANE_GAP

    # Milestone diamonds. v2: diamond row near top of gantt zone with the
    # labels staggered BELOW it (clearer of header subtitle than v1's
    # above-diamond placement).
    MS_Y = 220   # diamond y-baseline (well below subtitle ending ~y=189)
    parts.append(
        f'<text x="{START_X}" y="{MS_Y - 10}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">MILESTONES</text>'
    )

    # v2: milestone labels staggered between two y-positions to clear the
    # v1 collision at Q3'26 / Q4'26 / Q1'27 (labels overlapped horizontally).
    # "BTK+JAK ZERO-SHOT DEMO" shortened to "BTK+JAK demo" — visual emphasis
    # (glow ring + larger marker + accent color) already conveys "anchor",
    # so the long string was redundant and contributed to the overflow.
    milestones = [
        # (quarter_index, label, color, anchor, label_row)
        # label_row: 0 = upper (y_high), 1 = lower (y_low) — alternating zigzag
        (0,  "QurieSeq P1 lands",      CYAN_HI,    False, 0),
        (1,  "BTK+JAK demo",           CYAN_HI,    True,  1),  # shortened from ZERO-SHOT DEMO
        (2,  "Phase 2 VDJ on",         LAVENDER,   False, 0),
        (3,  "Pipeline 1 starts",      OK_GREEN,   False, 1),
        (5,  "Stage 4 wraps",          LAVENDER,   False, 0),
        (7,  "Pipeline 2 · P1 valid.", OK_GREEN,   False, 1),
        (9,  "Stage 5 wraps",          LAVENDER,   False, 0),
    ]
    # Two label rows BELOW the diamond — upper row sits closer to diamond,
    # lower row further away. Diamonds anchored at MS_Y; labels offset down.
    Y_HIGH = MS_Y + 22   # upper label row (closer to diamond)
    Y_LOW  = MS_Y + 44   # lower label row (further below)
    for q_idx, label, color, anchor, label_row in milestones:
        mx = GT_X + q_idx * Q_W + Q_W // 2
        # Diamond marker
        size = 9 if anchor else 6
        parts.append(
            f'<path d="M {mx} {MS_Y - size} L {mx + size} {MS_Y} L {mx} {MS_Y + size} L {mx - size} {MS_Y} Z" '
            f'fill="{color}" stroke="{color}" stroke-width="{2 if anchor else 1.2}" '
            f'fill-opacity="{0.7 if anchor else 0.4}"/>'
        )
        if anchor:
            parts.append(
                f'<circle cx="{mx}" cy="{MS_Y}" r="16" fill="none" stroke="{color}" '
                f'stroke-width="1" stroke-opacity="0.4"/>'
            )
        # Connector line from diamond down to its staggered label
        label_y = Y_HIGH if label_row == 0 else Y_LOW
        if label_row == 1:
            # Lower row: thin connector from diamond bottom edge down to label baseline
            parts.append(
                f'<line x1="{mx}" y1="{MS_Y + size}" x2="{mx}" y2="{label_y - 12}" '
                f'stroke="{color}" stroke-width="1" stroke-opacity="0.45"/>'
            )
        # Label text — staggered y, text-anchor="start" with mx+12 offset
        parts.append(
            f'<text x="{mx + 12}" y="{label_y}" fill="{color}" font-family="{FONT}" '
            f'font-size="{12 if anchor else 11}" font-weight="700" '
            f'text-anchor="start">{label}</text>'
        )

    # Quarter column headers (faint vertical dividers below)
    HDR_Y = GT_Y - 4
    for i, q in enumerate(QUARTERS):
        cx = GT_X + i * Q_W + Q_W // 2
        # Vertical divider through gantt
        x_div = GT_X + i * Q_W
        parts.append(
            f'<line x1="{x_div}" y1="{GT_Y}" x2="{x_div}" y2="{GT_Y + GANTT_H}" '
            f'stroke="{DIVIDER}" stroke-width="0.5" stroke-opacity="0.7"/>'
        )
        # Quarter label
        parts.append(
            f'<text x="{cx}" y="{HDR_Y}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" text-anchor="middle">{q}</text>'
        )
    # Final right edge divider
    parts.append(
        f'<line x1="{GT_X + N_Q * Q_W}" y1="{GT_Y}" x2="{GT_X + N_Q * Q_W}" y2="{GT_Y + GANTT_H}" '
        f'stroke="{DIVIDER}" stroke-width="0.5" stroke-opacity="0.7"/>'
    )

    # Lane labels (left) + bar definitions
    # Bar spec: (lane_index, start_q, end_q, label, sublabel, color, fill_opacity, stroke_w, emphasize)
    bars = [
        # WET LAB lane
        (0, 0, 1,  "Phase 1 delivery",                "5 donors · 4 modalities + BTK+JAK",  CYAN_HI,  0.20, 1.5, False),
        (0, 2, 5,  "Phase 2",                         "VDJ + 20-donor scale",               CYAN,     0.16, 1.3, False),
        (0, 6, 9,  "Phase 3",                         "B-cell lines + disease samples",     CYAN,     0.10, 1.0, False),
        # MODEL lane
        (1, 0, 1,  "Stage 3a",                        "adapter + decomposed readout",       LAVENDER, 0.18, 1.3, False),
        (1, 1, 2,  "Stage 3b",                        "Neural ODE · BTK+JAK demo",          CYAN_HI,  0.32, 2.4, True),  # ANCHOR
        (1, 2, 4,  "Stage 3c",                        "causal layer + Neumann GRN",         LAVENDER, 0.16, 1.3, False),
        (1, 4, 7,  "Stage 4",                         "VDJ + 20-donor scale",               LAVENDER, 0.16, 1.3, False),
        (1, 7, 9,  "Stage 5",                         "causal + clinical-ready",            LAVENDER, 0.12, 1.2, False),
        # DRUG PIPELINES lane
        (2, 2, 7,  "Pipeline 1",                      "target ID → validation",             OK_GREEN, 0.18, 1.3, False),
        (2, 6, 9,  "Pipeline 2",                      "+ Pipeline 1 target validation",     OK_GREEN, 0.14, 1.2, False),
        # PUBLICATIONS lane
        (3, 0, 2,  "Stage 3 verdict + BTK+JAK demo",  "investor deck · public closure",     TEXT_BODY, 0.16, 1.3, False),
        (3, 6, 9,  "Stage 4 + 5 peer-reviewed",       "publication · clinical handoff",     TEXT_BODY, 0.12, 1.1, False),
    ]

    # Render lane backgrounds + labels
    for li, (lbl, sublabel, color) in enumerate(LANES):
        ly = GT_Y + li * (LANE_H + LANE_GAP)
        # Lane row background (very subtle)
        parts.append(
            f'<rect x="{GT_X}" y="{ly}" width="{N_Q * Q_W}" height="{LANE_H}" '
            f'fill="{SURFACE_2}" fill-opacity="0.5"/>'
        )
        # Lane label (left)
        parts.append(
            f'<text x="{START_X}" y="{ly + LANE_H // 2 - 8}" fill="{color}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" letter-spacing="2">{lbl}</text>'
        )
        parts.append(
            f'<text x="{START_X}" y="{ly + LANE_H // 2 + 12}" fill="{TEXT_MUTED}" '
            f'font-family="{FONT_BODY}" font-size="11" font-weight="400" font-style="italic">{sublabel}</text>'
        )

    # Render bars (skip the lane-bg already drawn)
    # Stagger bars within a lane based on sub-row to avoid overlap
    bar_sub_rows = {0: 0, 1: 0, 2: 0, 3: 0}  # current sub-row per lane
    # For wet lab (lane 0), 3 bars: stagger if any overlap; here they're sequential so use single row
    # For model (lane 1), 5 bars sequential so use single row
    # For pipelines (lane 2), 2 bars overlap (P1 q2-q7, P2 q6-q9) — use 2 sub-rows
    # For publications (lane 3), 2 bars non-overlapping
    # Sub-row heights
    sub_h = (LANE_H - 20) // 2  # for 2-row lane
    full_h = LANE_H - 24

    # Assign sub-row per bar
    lane_layout = {
        0: "single",  # wet lab: tier 1 sequential
        1: "single",
        2: "double",  # pipelines: 2 bars overlap
        3: "single",
    }
    seen_in_lane = {0: 0, 1: 0, 2: 0, 3: 0}

    for li, sq, eq, lbl, sublabel, color, fop, sw, emph in bars:
        ly = GT_Y + li * (LANE_H + LANE_GAP)
        if lane_layout[li] == "double":
            sr = seen_in_lane[li]
            by = ly + 10 + sr * (sub_h + 6)
            bh = sub_h
            seen_in_lane[li] += 1
        else:
            by = ly + 12
            bh = full_h
        bx = GT_X + sq * Q_W + 4
        bw = (eq - sq + 1) * Q_W - 8

        # Bar
        parts.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" '
            f'fill="{color}" fill-opacity="{fop}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-opacity="{0.95 if emph else 0.65}"/>'
        )
        if emph:
            # Extra glow ring around BTK+JAK demo bar
            parts.append(
                f'<rect x="{bx - 3}" y="{by - 3}" width="{bw + 6}" height="{bh + 6}" rx="11" '
                f'fill="none" stroke="{color}" stroke-width="1" stroke-opacity="0.35"/>'
            )
            # ANCHOR badge
            parts.append(
                f'<text x="{bx + 10}" y="{by + 18}" fill="{color}" font-family="{FONT}" '
                f'font-size="9" font-weight="700" letter-spacing="2">◆ ANCHOR DEMO</text>'
            )
        # Title text (inside bar)
        title_y = by + (bh // 2 + 4) if not emph else by + (bh // 2 + 14)
        parts.append(
            f'<text x="{bx + 10}" y="{title_y}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="13" font-weight="700">{lbl}</text>'
        )
        # Sub-label (italic small) — only if bar wide enough
        if bw > 240:
            parts.append(
                f'<text x="{bx + 10}" y="{title_y + 18}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
                f'font-size="11" font-weight="400" font-style="italic">{sublabel}</text>'
            )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 §5 (Stage 3 sequencing) + §6 (Stage 4/5 scope) · "
            "QurieSeq Phase 1/2 plan (Thiago confirmation, 2026-05-12) · "
            "Pipeline starts contingent on Stage 3 verdict + Phase 1 perturbation-response data quality"
        ),
        slide_handle="D1 / 12",
        handle_color=LAVENDER,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "D1_quarterly_roadmap.svg"
    png_path = here / "D1_quarterly_roadmap_preview.png"
    svg = build_svg()
    # v2 collision-guard: filter footer-vs-pagination false positive only;
    # any milestone-label collision should now be cleared by the staggered
    # y_high/y_low layout introduced in v2.
    collisions = check_no_text_collisions(svg, min_gap=4)
    blocking = [c for c in collisions if "D1 / 12" not in (c[0], c[1])
                                       and not c[0].startswith("Source:")
                                       and not c[1].startswith("Source:")]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"D1 collision-guard FAIL:\n{msg}")
    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
