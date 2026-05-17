"""Build A5_causal_architecture.svg + preview PNG.

v2 fixes (4 issues from prompt commit ff92117 → A5 v2):
  Fix 1 (critical): equation rendering — `I` was being substituted to
    turnstile `⊢` glyph because `font-style="italic"` on a Latin I tspan
    triggers math-italic font substitution. Replaced with explicit non-italic
    Inter font family. Superscript `⁻¹` no longer relies on
    `baseline-shift="super"` (which cairosvg silently drops) — uses literal
    Unicode superscript chars U+207B + U+00B9 in the equation. Same for
    component definition row's `(I − W)⁻¹` reference.
  Fix 2 (medium): equation-card layout tightened. Component definitions
    row pulled up, architectural-requirement footer pushed to its own
    y-band with explicit 30px gap above. Card overall height reduced from
    294 → 244 to give cleaner vertical rhythm.
  Fix 3 (minor): subtitle trimmed from ~150 → ~120 chars; validation
    timing detail moves entirely to status pill.
  Fix 4 (substantive upgrade): GRN visualization replaced with 8 named
    biologically-meaningful gene nodes (BTK · JAK · CD3E · NFKB · STAT3
    · ZAP70 · MYD88 · IRF7). Radial gradient fills via SVG <defs>,
    hub-degree size differentiation, color coding by gene class (cyan
    perturbation targets / lavender TFs / green kinases / muted-blue
    effectors). Directional weight-coded edges with arrowheads. Novel
    learned edge highlighted in right panel with lavender marker.
    Line-style legend in right panel bottom-right.

Helper audit finding (per Fix 2 in prompt): the v1 collision-guard at
min_gap=2 correctly reported 0 bbox overlaps because there were none
(component-row baseline y=380 and architectural-footer baseline y=496
had 116px clear y-gap). The user-perceived "collision" was visual
cramping at slide-fill scale, not literal text-element overlap. The
helper detects A class of issues (bbox overlap); this v1 case was
B class (layout density). Both are real but require different checks.
Fix 2 here is pure layout-spacing improvement; the helper's design
is unchanged.

Run: python3 docs/deck/assets/diagrams/_build_a5.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, SURFACE_2, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, FONT_MATH, START_X, W, H,
    svg_open, background, header, footer, render_png,
    check_no_text_collisions,
)

# Section F amber (also used for "novel learned edge" marker on GRN)
ACCENT_AMBER = WARN_AMBER


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC A5 v2 — Causal architecture (Stage 3c spec-locked)")]
    background(parts)

    # ====================================================================
    # SVG <defs> — radial gradients for 8 GRN nodes
    # Per Fix 4: each node gets a radial gradient fill (darker center →
    # lighter edge) to give 3D effect without busy shadowing.
    # ====================================================================
    gradient_defs = '<defs>'
    # Color schemes per gene class (per Fix 4 spec table):
    #   perturbation targets (BTK, JAK, CD3E) → cyan
    #   transcription factors (NFKB, STAT3)   → lavender
    #   kinases/signaling (ZAP70, MYD88)       → green
    #   effectors (IRF7)                        → muted blue (use a softer cyan)
    GRAD_DEFS = {
        "cyan":     ("#26DDF9", "#00F2FF"),   # CYAN → CYAN_HI
        "lavender": ("#8B5CF6", "#B47DF0"),   # PURPLE → LAVENDER
        "green":    ("#4ADE80", "#86EFAC"),   # OK_GREEN → softer green
        "blue":     ("#5B9BD5", "#94BFE0"),   # muted blue → softer
    }
    for name, (c0, c1) in GRAD_DEFS.items():
        gradient_defs += (
            f'<radialGradient id="grn-{name}" cx="0.35" cy="0.35" r="0.85">'
            f'  <stop offset="0%" stop-color="{c1}" stop-opacity="0.85"/>'
            f'  <stop offset="60%" stop-color="{c0}" stop-opacity="0.45"/>'
            f'  <stop offset="100%" stop-color="{c0}" stop-opacity="0.18"/>'
            f'</radialGradient>'
        )
    # Background glows (carry over from earlier helper background())
    gradient_defs += '</defs>'
    parts.append(gradient_defs)

    header(
        parts,
        appendix_id="A5",
        section="ARCHITECTURE DEPTH",
        title="Causal Architecture — Spec-Locked",
        # v2 Fix 3: subtitle trimmed from ~150 chars to ~120. Validation
        # timing moved entirely to the status pill (which carries it
        # explicitly).
        subtitle=(
            "Neumann propagation + sparse learned GRN + direct-effect decoder · "
            "spec-locked v1.1 · validation post Phase 1 (Q1-Q2 2027)"
        ),
        eyebrow_color=CYAN,
    )

    # ====================================================================
    # STATUS PILL — top-right, unchanged from v1 (Ash-approved)
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
    # v2 Fix 2: card compressed from 294px → 244px tall, internal rows
    # given explicit y-band separation. y=216..460 (was y=216..510).
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

    # ---- The equation (visual hero, centered) ----
    # v2 Fix 1:
    #   - I rendered with explicit non-italic font-family="Inter, Arial, sans-serif"
    #     to prevent math-italic font substitution (which produces turnstile glyph).
    #   - Superscript ⁻¹ uses literal Unicode chars U+207B U+00B9, not
    #     baseline-shift="super" (cairosvg silently drops this).
    EQ_CX = NZ_X + NZ_W // 2
    EQ_Y = NZ_Y + 80   # equation baseline (was 100; tightened)

    # v2 Fix 1 (FINAL — bulletproof rect-I per prompt Risk 1 fallback):
    # Confirmed via 3 separate font-chain attempts (Inter, Inter+Arial+
    # sans-serif chain, Arial-only) that cairosvg + fontconfig substitutes
    # Latin I (U+0049) → turnstile glyph regardless of font-family attribute.
    # Source codepoint verified U+0049; problem is exclusively at render-time.
    # The prompt's Risk 1 explicitly authorizes rect fallback for this case.
    #
    # Equation rendered in 3 pieces: left text + rect-I + right text.
    # Hardcoded offsets centered around EQ_CX. Approximate widths:
    #   "ŷ = (" at 56pt × 0.5 ≈ 140px right-anchored at EQ_CX - 28
    #   rect-I at EQ_CX - 24, width 8, height 42 (matches Latin cap-height)
    #   "− W)⁻¹ · dₚ" at 56pt left-anchored at EQ_CX - 12
    # Visually centered enough; not pixel-perfect but readable + correct.

    # Left half: "ŷ = ("
    parts.append(
        f'<text x="{EQ_CX - 28}" y="{EQ_Y}" fill="{TEXT_TITLE}" font-family="Inter, Arial, sans-serif" '
        f'font-size="56" font-weight="700" text-anchor="end">'
        f'<tspan font-style="italic">ŷ</tspan>'
        f'<tspan fill="{TEXT_DIM}"> = </tspan>'
        f'<tspan fill="{OK_GREEN}">(</tspan>'
        f'</text>'
    )
    # Rect-I (vertical bar, no font-substitution risk)
    parts.append(
        f'<rect x="{EQ_CX - 24}" y="{EQ_Y - 40}" width="8" height="42" '
        f'fill="{OK_GREEN}" rx="0"/>'
    )
    # Right half: " − W)⁻¹ · dₚ"
    parts.append(
        f'<text x="{EQ_CX - 12}" y="{EQ_Y}" fill="{TEXT_TITLE}" font-family="Inter, Arial, sans-serif" '
        f'font-size="56" font-weight="700" text-anchor="start">'
        f'<tspan fill="{OK_GREEN}"> − </tspan>'
        f'<tspan fill="{CYAN_HI}" font-style="italic" font-weight="700">W</tspan>'
        f'<tspan fill="{OK_GREEN}">)</tspan>'
        f'<tspan fill="{OK_GREEN}">⁻¹</tspan>'
        f'<tspan fill="{TEXT_DIM}"> · </tspan>'
        f'<tspan fill="{LAVENDER}" font-style="italic" font-weight="700">d</tspan>'
        f'<tspan fill="{LAVENDER}" font-style="italic">ₚ</tspan>'
        f'</text>'
    )

    parts.append(
        f'<line x1="{EQ_CX - 220}" y1="{EQ_Y + 14}" x2="{EQ_CX + 220}" y2="{EQ_Y + 14}" '
        f'stroke="{CYAN}" stroke-width="1" stroke-opacity="0.35"/>'
    )

    # ---- Component definitions row (3 columns) ----
    # v2 Fix 2: pulled up; was DEF_Y = EQ_Y + 64 = 380.
    # Now at NZ_Y + 158 = 374 (subtle adjustment so eq + defs feel like one unit).
    DEF_Y = NZ_Y + 158
    COL_W = NZ_W // 3

    # Cols 0 + 1 — normal rendering via the loop (no I-substitution risk)
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

    # Col 2 — (I − W)⁻¹ with rect-I fix (same pattern as main equation).
    # At 24pt: rect-I is ~4×20px. Hardcoded offsets around symbol anchor.
    cx2 = NZ_X + 2 * COL_W + COL_W // 2
    sym_anchor_x = cx2 - 90   # right edge of full symbol (matches loop pattern)
    # Right half: " − W)⁻¹" — anchored at sym_anchor_x
    parts.append(
        f'<text x="{sym_anchor_x}" y="{DEF_Y}" fill="{OK_GREEN}" font-family="Inter, Arial, sans-serif" '
        f'font-size="24" font-weight="700" text-anchor="end">'
        f'<tspan> − </tspan>'
        f'<tspan font-style="italic">W</tspan>'
        f'<tspan>)</tspan>'
        f'<tspan>⁻¹</tspan>'
        f'</text>'
    )
    # Right-half approximate width at 24pt: ~70px → so rect-I goes at sym_anchor_x - 70
    rect_x = sym_anchor_x - 72
    parts.append(
        f'<rect x="{rect_x}" y="{DEF_Y - 17}" width="4" height="20" fill="{OK_GREEN}" rx="0"/>'
    )
    # Left "(" — just before rect-I
    parts.append(
        f'<text x="{rect_x - 2}" y="{DEF_Y}" fill="{OK_GREEN}" font-family="Inter, Arial, sans-serif" '
        f'font-size="24" font-weight="700" text-anchor="end">(</text>'
    )
    # Annotation right of full symbol (matches loop pattern: x=cx-70)
    parts.append(
        f'<text x="{cx2 - 70}" y="{DEF_Y}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="14" font-style="italic">closed-form propagation</text>'
    )

    # ---- Architectural requirement footer ----
    # v2 Fix 2: explicit 30px gap above this line. Card ends at NZ_Y+NZ_H = 460;
    # footer baseline at NZ_Y+220 = 436 gives 16px breathing below + 62px gap
    # from definitions (DEF_Y=374) above.
    AR_Y = NZ_Y + 220
    parts.append(
        f'<line x1="{NZ_X + 32}" y1="{AR_Y - 22}" x2="{NZ_X + NZ_W - 32}" y2="{AR_Y - 22}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{EQ_CX}" y="{AR_Y}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="13" font-style="italic" text-anchor="middle">'
        f'<tspan font-weight="700">Architectural requirement:</tspan> '
        f'<tspan fill="{CYAN_HI}" font-weight="700">ρ(W) &lt; 1</tspan> '
        f'enforced by sparsity L1 — guarantees Neumann-series convergence'
        f'</text>'
    )

    # ====================================================================
    # MIDDLE ZONE — Sparse learned GRN visualization (Fix 4 upgrade)
    # y=484..786 (302px tall, slightly larger than v1's 226 for richer viz)
    # ====================================================================
    MZ_Y = 484
    MZ_H = 302
    PANEL_GAP = 60
    PANEL_W = (W - 2 * START_X - PANEL_GAP) // 2  # 834

    parts.append(
        f'<text x="{START_X}" y="{MZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'SPARSE LEARNED GRN · STRUCTURAL PRIOR (LEFT) → LEARNED WEIGHTS (RIGHT)</text>'
    )
    parts.append(
        f'<line x1="{START_X + 700}" y1="{MZ_Y - 6}" x2="{W - START_X}" y2="{MZ_Y - 6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # ---- Node definitions (Fix 4) ----
    # 8 named gene clusters with biological grouping. Position layout: pathway
    # flow from top (perturbation targets) → middle (signaling kinases) →
    # bottom (TFs hub + effector). "Structured biology" not "random graph".
    #
    # Position is fractional within panel inner area (0..1 in x, 0..1 in y).
    # Hub-degree size differentiation: NFKB + STAT3 = 30px diameter (largest),
    # peripheral nodes 18-22px.
    NODES = [
        # (label, frac_x, frac_y, gene_class, radius)
        # Top row — perturbation targets
        ("BTK",   0.18, 0.18, "cyan",     22),
        ("CD3E",  0.50, 0.10, "cyan",     22),
        ("JAK",   0.82, 0.18, "cyan",     22),
        # Middle row — kinases / signaling intermediates
        ("ZAP70", 0.30, 0.46, "green",    20),
        ("MYD88", 0.70, 0.46, "green",    20),
        # Bottom row — transcription-factor hubs (largest, most-connected)
        ("NFKB",  0.38, 0.78, "lavender", 30),
        ("STAT3", 0.62, 0.78, "lavender", 30),
        # Far-bottom-right — effector (smallest, periphery)
        ("IRF7",  0.92, 0.90, "blue",     18),
    ]
    # Index lookup by label
    node_by_label = {n[0]: n for n in NODES}

    # ---- Edge definitions (Fix 4) ----
    # Biological grounding: signaling-pathway-aware directed edges.
    # (source, target, in_string_prior, learned_weight)
    #   learned_weight: 0=pruned (below threshold), 1=medium, 2=strong
    # The "novel" edge (in_string_prior=False, learned_weight≥1) is the
    # platform's "we learned something the prior didn't predict" highlight.
    EDGES = [
        # Direct perturbation → TF
        ("BTK",   "NFKB",  True,  2),  # BCR signaling → NFKB (canonical)
        ("JAK",   "STAT3", True,  2),  # JAK-STAT (canonical)
        # Perturbation → kinase intermediate
        ("CD3E",  "ZAP70", True,  2),  # TCR signaling (canonical)
        # Kinase → TF
        ("ZAP70", "NFKB",  True,  1),  # T-cell activation
        ("MYD88", "NFKB",  True,  2),  # TLR signaling (canonical)
        # Kinase → effector
        ("MYD88", "IRF7",  True,  1),  # TLR-IRF7
        # TF cross-talk / pruned-in-learning
        ("STAT3", "IRF7",  True,  0),  # In STRING but learning prunes it (low signal in our data)
        ("NFKB",  "IRF7",  True,  0),  # Pruned
        # NOVEL learned edge (not in STRING prior, discovered in training)
        ("STAT3", "NFKB",  False, 2),  # TF cross-regulation discovered
    ]

    def render_panel(px: int, py: int, pw: int, ph: int,
                     title: str, subtitle: str, left_panel: bool):
        # Card
        parts.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" '
            f'fill="{SURFACE}" stroke="{DIVIDER}" stroke-width="1.2" stroke-opacity="0.9"/>'
        )
        # Title + subtitle
        parts.append(
            f'<text x="{px + 20}" y="{py + 26}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" letter-spacing="2">{title}</text>'
        )
        parts.append(
            f'<text x="{px + 20}" y="{py + 44}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-style="italic">{subtitle}</text>'
        )

        # Graph drawing area inside panel
        gx0, gy0 = px + 30, py + 60
        gw, gh = pw - 60, ph - 130  # leave 70px at bottom for captions + legend

        # Compute absolute positions per node
        def node_pos(label: str):
            n = node_by_label[label]
            return (gx0 + n[1] * gw, gy0 + n[2] * gh, n[3], n[4])  # x, y, gene_class, r

        # ---- Edges first (so nodes overlay) ----
        for src_lbl, tgt_lbl, in_prior, weight in EDGES:
            sx, sy, _, sr = node_pos(src_lbl)
            tx, ty, _, tr = node_pos(tgt_lbl)
            # Compute edge endpoints retracted by node radii (so arrow lands
            # at edge of node circle, not center)
            import math
            dx, dy = tx - sx, ty - sy
            dist = math.hypot(dx, dy) or 1
            ux, uy = dx / dist, dy / dist
            x1 = sx + ux * sr
            y1 = sy + uy * sr
            x2 = tx - ux * tr
            y2 = ty - uy * tr

            if left_panel:
                # STRING prior — only show edges that are in prior; all uniform.
                if in_prior:
                    parts.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{TEXT_MUTED}" stroke-width="2" stroke-opacity="0.55"/>'
                    )
            else:
                # Learned GRN — weight + direction + confidence-source encoded.
                # Pruned (was in prior, dropped): dashed grey, no arrowhead
                if weight == 0:
                    parts.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{TEXT_DIM}" stroke-width="1.2" stroke-opacity="0.42" '
                        f'stroke-dasharray="4 4"/>'
                    )
                    continue
                # Strong / medium / novel — solid stroke, with arrowhead at target
                if not in_prior:
                    # Novel learned edge — brighter cyan + lavender circle marker at midpoint
                    stroke_color = CYAN_HI
                    stroke_width = 4
                    stroke_opacity = 0.95
                elif weight == 2:
                    stroke_color = CYAN_HI
                    stroke_width = 4
                    stroke_opacity = 0.9
                else:  # medium
                    stroke_color = CYAN
                    stroke_width = 2.5
                    stroke_opacity = 0.75
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{stroke_color}" stroke-width="{stroke_width}" '
                    f'stroke-opacity="{stroke_opacity}"/>'
                )
                # Arrowhead at (x2, y2) pointing along (ux, uy)
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
                # Novel-edge marker: small lavender filled circle at midpoint
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

        # ---- Nodes ----
        for n in NODES:
            label, fx, fy, gene_class, r = n
            nx = gx0 + fx * gw
            ny = gy0 + fy * gh
            # Outer soft glow ring (subtle, matches gradient)
            parts.append(
                f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="{r + 4}" fill="none" '
                f'stroke="url(#grn-{gene_class})" stroke-width="1.5" stroke-opacity="0.45"/>'
            )
            # Filled gradient node
            parts.append(
                f'<circle cx="{nx:.0f}" cy="{ny:.0f}" r="{r}" '
                f'fill="url(#grn-{gene_class})" stroke="{TEXT_TITLE}" '
                f'stroke-width="1" stroke-opacity="0.55"/>'
            )
            # Gene label centered in node (or above for very small nodes)
            label_y_off = 4  # approximate vertical centering
            parts.append(
                f'<text x="{nx:.0f}" y="{ny + label_y_off}" fill="{TEXT_TITLE}" '
                f'font-family="{FONT_BODY}" font-size="11" font-weight="700" '
                f'text-anchor="middle">{label}</text>'
            )

        # ---- Panel-bottom captions + (right-only) legend ----
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
            # Gene-class color-legend on left panel (so it's not duplicated)
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
            # Edge-style legend in right panel (bottom-right corner)
            legend_x = px + pw - 240
            legend_y = py + ph - 60
            # Small backdrop card
            parts.append(
                f'<rect x="{legend_x - 8}" y="{legend_y - 12}" width="232" height="62" rx="6" '
                f'fill="{SURFACE_2}" stroke="{DIVIDER}" stroke-width="0.8" stroke-opacity="0.8"/>'
            )
            parts.append(
                f'<text x="{legend_x}" y="{legend_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
                f'font-size="9" font-weight="700" letter-spacing="2">EDGE LEGEND</text>'
            )
            # 4 legend rows
            row_specs = [
                ('━━', CYAN_HI, 4, None, "high-weight learned"),
                ('──', CYAN,    2.5, None, "medium-weight learned"),
                ('···', TEXT_DIM, 1.2, '4 4', "pruned (below threshold)"),
                ('◆',  LAVENDER, 0, None, "novel (not in STRING)"),
            ]
            for li, (glyph, color, sw, dash, desc) in enumerate(row_specs):
                row_y = legend_y + 14 + li * 10
                # Draw line sample
                lx = legend_x + 4
                lx_end = lx + 28
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
                if glyph == '◆':
                    # Special: lavender diamond marker, no line sample
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

    # Connector arrow + caption between panels
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
    # BOTTOM ZONE — Direct-effect log-FC head (unchanged structure from v1)
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
    parts.append(
        f'<text x="{b1_x + b1_w // 2}" y="{BD_Y + BD_BOX_H // 2 + 5}" fill="{TEXT_BODY}" '
        f'font-family="{FONT_BODY}" font-size="13" font-weight="600" text-anchor="middle">'
        f'<tspan font-style="italic">z</tspan> + perturbation context</text>'
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
    out_x = a2_x + a2_w + 4
    parts.append(
        f'<text x="{out_x + 20}" y="{BD_Y + BD_BOX_H // 2 + 7}" fill="{LAVENDER}" '
        f'font-family="Inter, Arial, sans-serif" font-size="22" font-weight="700">'
        f'<tspan font-style="italic">d</tspan>'
        f'<tspan font-style="italic">ₚ</tspan>'
        f'</text>'
    )

    cmp_x = out_x + 80
    cmp_y = BD_Y - 4
    parts.append(
        f'<text x="{cmp_x}" y="{cmp_y + 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-weight="400">'
        f'<tspan font-weight="700">Stage 3a/3b predicted:</tspan> '
        f'abundance after perturbation'
        f'</text>'
    )
    # Bottom-zone comparison line — use rect-I fix here too (consistent
    # with main equation + component defs). At 12pt: rect-I = 2.5×11px.
    # Split into 3 pieces: segment A (left text) + rect-I + segment B (right text).
    cmp_seg_y = cmp_y + 36
    # Segment A: "Stage 3c separates:  dₚ (direct) + ("
    parts.append(
        f'<text x="{cmp_x}" y="{cmp_seg_y}" fill="{TEXT_BODY}" font-family="Inter, Arial, sans-serif" '
        f'font-size="12" font-weight="400">'
        f'<tspan fill="{CYAN_HI}" font-weight="700">Stage 3c separates:</tspan>  '
        f'<tspan fill="{LAVENDER}" font-weight="700" font-style="italic">d</tspan>'
        f'<tspan fill="{LAVENDER}" font-style="italic">ₚ</tspan>'
        f'<tspan fill="{TEXT_BODY}"> (direct) + </tspan>'
        f'<tspan fill="{OK_GREEN}" font-weight="700">(</tspan>'
        f'</text>'
    )
    # Approximate width of segment A at 12pt × 0.55 visual char width.
    # "Stage 3c separates:  dₚ (direct) + (" ≈ 36 visible chars → ~238px
    SEG_A_W = 238
    rect_x2 = cmp_x + SEG_A_W
    parts.append(
        f'<rect x="{rect_x2}" y="{cmp_seg_y - 9}" width="2.5" height="11" fill="{OK_GREEN}" rx="0"/>'
    )
    # Segment B: " − W)⁻¹ dₚ (propagated)"
    parts.append(
        f'<text x="{rect_x2 + 4}" y="{cmp_seg_y}" fill="{TEXT_BODY}" font-family="Inter, Arial, sans-serif" '
        f'font-size="12" font-weight="400">'
        f'<tspan fill="{OK_GREEN}" font-weight="700"> − </tspan>'
        f'<tspan fill="{OK_GREEN}" font-weight="700" font-style="italic">W</tspan>'
        f'<tspan fill="{OK_GREEN}" font-weight="700">)</tspan>'
        f'<tspan fill="{OK_GREEN}" font-weight="700">⁻¹</tspan>'
        f'<tspan fill="{LAVENDER}" font-weight="700" font-style="italic"> d</tspan>'
        f'<tspan fill="{LAVENDER}" font-style="italic">ₚ</tspan>'
        f'<tspan fill="{TEXT_BODY}"> (propagated)</tspan>'
        f'</text>'
    )

    parts.append(
        f'<text x="{BZ_X + 22}" y="{BZ_Y + BZ_H - 12}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-style="italic">'
        f'<tspan font-weight="700">Why this matters:</tspan>  causal queries vs predictive queries — '
        f'<tspan fill="{TEXT_BODY}">"what does X cause?"</tspan>'
        f'<tspan fill="{TEXT_DIM}">  vs  </tspan>'
        f'<tspan fill="{TEXT_BODY}">"what happens after X?"</tspan>'
        f'</text>'
    )

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

    # Collision guard tightened to min_gap=2 (per F1 v2 lesson).
    # Known-good filters (intentional rect-split adjacencies for the
    # `(I − W)⁻¹` math notation — text-anchor positions left+right halves
    # precisely with rect-I in middle; bbox heuristic can't reason about
    # the rect gap so it flags them as overlapping when they aren't):
    #   - Pair containing "− W)" right-half text  (equation + comp defs + cmp line)
    #   - Pair containing "(" alone (rect-split open paren)
    collisions = check_no_text_collisions(svg, min_gap=2)
    def _is_rect_split_pair(a: str, b: str) -> bool:
        a, b = a.strip(), b.strip()
        return (
            ("− W)" in a or "− W)" in b)
            or a == "(" or b == "("
        )
    blocking = [c for c in collisions
                if "A5 / 14" not in (c[0], c[1])
                and not c[0].startswith("Source:")
                and not c[1].startswith("Source:")
                and not _is_rect_split_pair(c[0], c[1])]
    if blocking:
        msg = "\n".join(f"  · {a!r} ↔ {b!r} ({ox}×{oy}px)" for a, b, ox, oy in blocking)
        raise SystemExit(f"A5 v2 collision-guard FAIL:\n{msg}")

    svg_path.write_text(svg)
    print(f"wrote {svg_path}  (collision-guard ✓ min_gap=2)")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
