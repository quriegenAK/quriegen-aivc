"""Build A1_system_architecture.svg from the locked palette + typography.

Run: python3 docs/deck/assets/diagrams/_build_a1.py
Output: docs/deck/assets/diagrams/A1_system_architecture.svg
Preview (optional): pip install cairosvg && python3 -c \\
  "import cairosvg, pathlib; cairosvg.svg2png(\\
   bytestring=pathlib.Path('docs/deck/assets/diagrams/A1_system_architecture.svg').read_bytes(),\\
   write_to='docs/deck/assets/diagrams/A1_system_architecture_preview.png',\\
   output_width=1920, output_height=1080)"

Visual conventions (mirror slide 37 of Kinga's deck):
- Dark background #070A14
- 5-block horizontal flow with rounded-rect containers
- Step numbers in step-cycle colors (cyan / lavender / cyan / purple / off-white)
- Section labels in letter-spaced bold caps
- Status row below blocks, invariant-keyword row below status
- Inline SVG padlock on ENCODER block (frozen-substrate indicator)

Palette + typography source of truth:
- docs/deck/assets/color_palette.md
- docs/deck/assets/typography.md
- docs/deck/assets/icon_inventory.md
"""
from __future__ import annotations
import pathlib

# ---- Palette (locked in docs/deck/assets/color_palette.md) ----
BG          = "#070A14"
SURFACE     = "#0F1428"  # slightly elevated block fill
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
PENDING     = "#A8B4C2"
DIVIDER     = "#1A2235"

# ---- Layout (1920 x 1080 viewBox, 16:9) ----
W, H = 1920, 1080
N = 5
BLOCK_W = 320
BLOCK_H = 380
GAP = 32
TOTAL_BLOCK_W = N * BLOCK_W + (N - 1) * GAP
BLOCK_Y = 232
START_X = (W - TOTAL_BLOCK_W) // 2  # 96

BLOCKS = [
    dict(step="01", color=CYAN_HI,  title="INPUT",    sub="Multi-omics, per cell",
         # Fix 3 (v2): "210-D panel" → "30–210 surface markers" — honest range.
         # 210-D was unverified for QurieSeq (Mimitou uses 37 markers); C1 speaker
         # notes flagged this as pending Kinga confirmation.
         body=["RNA · 36,601 genes", "ATAC · 323,500 peaks", "Protein · 30–210 surface markers"],
         foot="Mimitou · DOGMA-seq · QurieSeq"),
    dict(step="02", color=LAVENDER, title="ENCODER",  sub="Frozen + adapter",   lock=True,
         # Fix 2 (v2): "Trimodal" → "Multi-omics" — keeps 5-modality platform
         # framing alive on A1 (per A2 v4 decision, commit 1b61964).
         body=["Multi-omics → 256-D latent", "≈130K-param adapter", "LayerNorm + GELU"],
         foot="Pretrained on DOGMA-seq"),
    dict(step="03", color=CYAN,     title="TEMPORAL", sub="Neural ODE",
         body=["z(t₀) → z(t)", "Continuous-time state", "0 → 180 min"],
         foot="QurieSeq Phase 1 design"),
    dict(step="04", color=PURPLE,   title="READOUT",  sub="4-arm decomposed",
         body=["h_base + Δ_stim", "+ Δ_inh + Δ_synergy", "Zero-arm L2 constraint"],
         foot="Compositional generalization"),
    dict(step="05", color=TEXT_BODY,title="OUTPUT",   sub="Pathway-aware",
         body=["RNA dynamics", "Protein dynamics", "58 pathway scores"],
         foot="Phospho plugs in Phase 2"),
]

STATUS = [
    dict(icon="●",   color=OK_GREEN,   label="Real data",                  sub="Mimitou + DOGMA"),
    dict(icon="●●", color=OK_GREEN, label="Pretrain ✓  Adapter ✓", sub="Stages 1+2 · S3 Part 1"),
    dict(icon="○",   color=PENDING,    label="Stage 3b",                   sub="Q3 2026 · QurieSeq"),
    # Fix 4 (v2): "In-flight training" → "Infra ready · training May" — more
    # honest: Stage 3a code is shipped (87 tests green) but the BSC training
    # run hasn't kicked off yet. Amber color preserved (between green and grey).
    dict(icon="◐",   color=WARN_AMBER, label="Stage 3a",                   sub="Infra ready · training May"),
    dict(icon="○",   color=PENDING,    label="Stage 3c",                   sub="Q1 2027 · phospho"),
]

INVARIANTS = [
    "MODALITY-AGNOSTIC",
    "CROSS-CORPUS TRANSFER",
    "IRREGULAR TIMEPOINTS",
    "COMPOSITIONAL GENERALIZATION",
    "BIOLOGICAL INTERPRETABILITY",
]

FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
FONT_BODY = "Arial, Inter, 'Helvetica Neue', sans-serif"


def lock_icon(cx: int, cy: int, color: str) -> str:
    return (
        f'<g transform="translate({cx-8},{cy-9})">'
        f'  <rect x="0" y="6" width="16" height="12" rx="2" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.6"/>'
        f'  <path d="M3 6 V3.5 A4.5 4.5 0 0 1 13 3.5 V6" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>'
        f'  <circle cx="8" cy="12" r="1.2" fill="{color}"/>'
        f'</g>'
    )


def build_svg() -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="AIVC foundation model system architecture">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    parts.append(
        '<defs>'
        f'  <radialGradient id="glow1" cx="0.85" cy="0.15" r="0.55">'
        f'    <stop offset="0%" stop-color="{CYAN}" stop-opacity="0.10"/>'
        f'    <stop offset="100%" stop-color="{CYAN}" stop-opacity="0"/>'
        f'  </radialGradient>'
        f'  <radialGradient id="glow2" cx="0.15" cy="0.95" r="0.5">'
        f'    <stop offset="0%" stop-color="{PURPLE}" stop-opacity="0.10"/>'
        f'    <stop offset="100%" stop-color="{PURPLE}" stop-opacity="0"/>'
        f'  </radialGradient>'
        '</defs>'
        f'<rect width="{W}" height="{H}" fill="url(#glow1)"/>'
        f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>'
    )

    # Header
    parts.append(
        f'<text x="{START_X}" y="78" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="4">APPENDIX A1 · ARCHITECTURE DEPTH</text>'
    )
    parts.append(
        f'<line x1="{START_X + 380}" y1="72" x2="{START_X + 600}" y2="72" '
        f'stroke="{CYAN}" stroke-opacity="0.4" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{START_X}" y="138" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="40" font-weight="700" letter-spacing="-0.5">'
        f'AIVC Foundation Model — System Architecture</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="186" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="18" font-weight="400">'
        f'One unified model · three input modalities · continuous time · pathway-grounded outputs '
        f'<tspan fill="{TEXT_DIM}">— same trained model predicts every PBMC cell type at any timepoint, no retraining</tspan>'
        f'</text>'
    )

    # Blocks
    for i, b in enumerate(BLOCKS):
        x = START_X + i * (BLOCK_W + GAP)
        y = BLOCK_Y
        color = b["color"]
        parts.append(
            f'<rect x="{x}" y="{y}" width="{BLOCK_W}" height="{BLOCK_H}" rx="14" '
            f'fill="{SURFACE}" stroke="{color}" stroke-width="1.5" stroke-opacity="0.65"/>'
        )
        parts.append(
            f'<text x="{x+22}" y="{y+38}" fill="{color}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" letter-spacing="3">{b["step"]}</text>'
        )
        parts.append(
            f'<line x1="{x+22}" y1="{y+50}" x2="{x+62}" y2="{y+50}" '
            f'stroke="{color}" stroke-width="1.5" stroke-opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{x+22}" y="{y+92}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="30" font-weight="700">{b["title"]}</text>'
        )
        if b.get("lock"):
            # Fix 1 (v2): lock icon moved to upper-right corner of the card,
            # mirroring the step number on the upper-left. FROZEN text label
            # removed — subtitle ("FROZEN + ADAPTER") + visual lock = two signals.
            parts.append(lock_icon(x + BLOCK_W - 30, y + 34, color))
        parts.append(
            f'<text x="{x+22}" y="{y+122}" fill="{color}" font-family="{FONT}" '
            f'font-size="13" font-weight="600" letter-spacing="1.5">{b["sub"].upper()}</text>'
        )
        body_y0 = y + 168
        for j, line in enumerate(b["body"]):
            parts.append(
                f'<text x="{x+22}" y="{body_y0 + j*30}" fill="{TEXT_BODY}" '
                f'font-family="{FONT_BODY}" font-size="15" font-weight="400">'
                f'<tspan fill="{color}" font-weight="700">›</tspan>  {line}</text>'
            )
        parts.append(
            f'<line x1="{x+22}" y1="{y+BLOCK_H-46}" x2="{x+BLOCK_W-22}" y2="{y+BLOCK_H-46}" '
            f'stroke="{DIVIDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x+22}" y="{y+BLOCK_H-22}" fill="{TEXT_MUTED}" '
            f'font-family="{FONT_BODY}" font-size="11" font-style="italic">{b["foot"]}</text>'
        )
        if i < N - 1:
            ax1 = x + BLOCK_W + 4
            ax2 = x + BLOCK_W + GAP - 4
            ay = y + BLOCK_H // 2
            parts.append(
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2 - 6}" y2="{ay}" '
                f'stroke="{CYAN}" stroke-width="2" stroke-opacity="0.55" stroke-linecap="round"/>'
            )
            parts.append(
                f'<path d="M {ax2-8} {ay-5} L {ax2} {ay} L {ax2-8} {ay+5}" '
                f'fill="none" stroke="{CYAN}" stroke-width="2" stroke-opacity="0.85" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )

    # Status row
    # Fix 5 (v2): tightened vertical layout (Option A from prompt).
    # Was +36 → +28: pulls VALIDATION STATUS eyebrow ~8px closer to cards.
    status_y = BLOCK_Y + BLOCK_H + 28
    parts.append(
        f'<text x="{START_X}" y="{status_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">VALIDATION STATUS</text>'
    )
    parts.append(
        f'<line x1="{START_X + 160}" y1="{status_y-6}" x2="{W - START_X}" y2="{status_y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    for i, s in enumerate(STATUS):
        x = START_X + i * (BLOCK_W + GAP)
        cx = x + BLOCK_W // 2
        # Fix 5 (v2): tighten internal status spacing (was +36/+30/+50)
        sy = status_y + 32
        parts.append(
            f'<text x="{cx}" y="{sy}" fill="{s["color"]}" font-family="{FONT}" '
            f'font-size="24" font-weight="700" text-anchor="middle">{s["icon"]}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{sy+28}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" text-anchor="middle">{s["label"]}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{sy+48}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="12" font-weight="400" text-anchor="middle">{s["sub"]}</text>'
        )

    # Invariant row
    # Fix 5 (v2): tighten gap above invariants (was status_y + 140).
    inv_y = status_y + 120
    parts.append(
        f'<text x="{START_X}" y="{inv_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">INVARIANT GUARANTEES</text>'
    )
    parts.append(
        f'<line x1="{START_X + 200}" y1="{inv_y-6}" x2="{W - START_X}" y2="{inv_y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    inv_color_cycle = [CYAN_HI, LAVENDER, CYAN, PURPLE, TEXT_BODY]
    for i, kw in enumerate(INVARIANTS):
        x = START_X + i * (BLOCK_W + GAP)
        cx = x + BLOCK_W // 2
        ky = inv_y + 42
        pill_w, pill_h = 264, 36
        parts.append(
            f'<rect x="{cx - pill_w//2}" y="{ky-22}" width="{pill_w}" height="{pill_h}" '
            f'rx="18" fill="{inv_color_cycle[i]}" fill-opacity="0.10" '
            f'stroke="{inv_color_cycle[i]}" stroke-width="1" stroke-opacity="0.45"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{ky+2}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="12" font-weight="700" letter-spacing="2.5" text-anchor="middle">{kw}</text>'
        )

    # Footer
    # Fix 5 (v2): pull footer up from y=H-60/H-28 (1020/1052) to ~y=948/980.
    # Closes ~72px of empty band below the invariant pills while preserving
    # the ~100px bottom margin called for in the prompt.
    footer_line_y = H - 132   # was H - 60
    footer_text_y = H - 100   # was H - 28  (bottom margin 100px from baseline)
    parts.append(
        f'<line x1="{START_X}" y1="{footer_line_y}" x2="{W - START_X}" y2="{footer_line_y}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{START_X}" y="{footer_text_y}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" font-style="italic">'
        f'Source: AIVC architecture spec v1.1 · Phase 6.5g.2 closure (2026-05-04) · '
        f'Stage 3 Part 1 verdict (2026-05-11) · 73% Calderon cross-corpus · 0.57 synergy 4-class (2.27× chance)'
        f'</text>'
    )
    parts.append(
        f'<text x="{W - START_X}" y="{footer_text_y}" fill="{CYAN}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2" text-anchor="end">A1 / 12</text>'
    )
    parts.append('</svg>')
    return "\n".join(parts)


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent / "A1_system_architecture.svg"
    svg = build_svg()
    out.write_text(svg)
    print(f"wrote {out}  ({len(svg)} bytes)")
