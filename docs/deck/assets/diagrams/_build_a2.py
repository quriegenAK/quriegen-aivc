"""Build A2_encoder_evidence.svg + A2_encoder_evidence_preview.png.

Three-zone layout:
- Top-left: multi-omics encoder schematic (5 modalities → contrastive fusion → 256-D latent)
- Top-right: hero 73% cross-corpus accuracy callout
- Bottom: DOGMA-seq credibility footer (Mimitou 2021, Nat Biotech)

Style locked to A1 v2: same palette, typography, header/footer conventions.

Run:  python3 docs/deck/assets/diagrams/_build_a2.py
"""
from __future__ import annotations
import pathlib

# ---- Palette (from docs/deck/assets/color_palette.md) ----
BG          = "#070A14"
SURFACE     = "#0F1428"
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

W, H = 1920, 1080
START_X = 96
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
FONT_BODY = "Arial, Inter, 'Helvetica Neue', sans-serif"


def lock_icon(cx: int, cy: int, color: str, scale: float = 1.0) -> str:
    s = scale
    return (
        f'<g transform="translate({cx-8*s},{cy-9*s}) scale({s})">'
        f'<rect x="0" y="6" width="16" height="12" rx="2" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.6"/>'
        f'<path d="M3 6 V3.5 A4.5 4.5 0 0 1 13 3.5 V6" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>'
        f'<circle cx="8" cy="12" r="1.2" fill="{color}"/>'
        f'</g>'
    )


def header(parts: list[str], appendix_id: str, section: str, title: str, subtitle: str):
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


def footer(parts: list[str], source_text: str, slide_handle: str):
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


def background(parts: list[str]):
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
        f'<radialGradient id="hero73" cx="0.5" cy="0.5" r="0.6">'
        f'<stop offset="0%" stop-color="{CYAN}" stop-opacity="0.18"/>'
        f'<stop offset="100%" stop-color="{CYAN}" stop-opacity="0"/>'
        f'</radialGradient>'
        '</defs>'
        f'<rect width="{W}" height="{H}" fill="url(#glow1)"/>'
        f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>'
    )


# ---- Modality pill (used in A2 left zone) ----
def modality_pill(parts: list[str], x: int, y: int, w: int, h: int, label: str,
                  detail: str, *, validated: bool):
    if validated:
        fill = CYAN
        fill_op = "0.18"
        stroke = CYAN_HI
        stroke_w = "1.5"
        dash = ""
        icon = "✓"
        icon_color = OK_GREEN
        text_color = TEXT_BODY
    else:
        fill = WARN_AMBER
        fill_op = "0.08"
        stroke = WARN_AMBER
        stroke_w = "1.5"
        dash = ' stroke-dasharray="6 4"'
        icon = "○"
        icon_color = WARN_AMBER
        text_color = TEXT_MUTED
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{fill}" fill-opacity="{fill_op}" stroke="{stroke}" '
        f'stroke-width="{stroke_w}"{dash}/>'
    )
    parts.append(
        f'<text x="{x+18}" y="{y+h/2 + 2}" fill="{icon_color}" font-family="{FONT}" '
        f'font-size="20" font-weight="700" dominant-baseline="middle">{icon}</text>'
    )
    parts.append(
        f'<text x="{x+44}" y="{y+h/2 - 4}" fill="{text_color}" font-family="{FONT}" '
        f'font-size="18" font-weight="700" dominant-baseline="middle">{label}</text>'
    )
    parts.append(
        f'<text x="{x+44}" y="{y+h/2 + 14}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" dominant-baseline="middle">{detail}</text>'
    )


def build_svg() -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="AIVC multi-omics encoder: the frozen substrate">'
    )
    background(parts)
    header(
        parts,
        appendix_id="A2",
        section="ARCHITECTURE DEPTH",
        title="Multi-Omics Encoder — The Frozen Substrate",
        subtitle=(
            "Modality-extensible foundation, pretrained on RNA + ATAC + Protein today · "
            "Phospho and VDJ slot in as QurieSeq Phase 2 lands · cross-corpus validated before any downstream training"
        ),
    )

    # ====================================================================
    # Top-left zone: ENCODER SCHEMATIC  (x=96..1010, y=232..712)
    # ====================================================================
    LZ_X, LZ_Y, LZ_W, LZ_H = 96, 232, 914, 480
    # Zone eyebrow + outline
    parts.append(
        f'<text x="{LZ_X}" y="{LZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">MULTI-OMICS ENCODER</text>'
    )
    parts.append(
        f'<line x1="{LZ_X + 200}" y1="{LZ_Y-6}" x2="{LZ_X + LZ_W}" y2="{LZ_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Modality column (left side of zone)
    MOD_X, MOD_Y = LZ_X, LZ_Y + 24
    MOD_W, MOD_H = 240, 56
    GAP = 12
    # "TODAY" group label
    parts.append(
        f'<text x="{MOD_X}" y="{MOD_Y + 4}" fill="{OK_GREEN}" font-family="{FONT}" '
        f'font-size="10" font-weight="700" letter-spacing="2.5">TODAY · 3 OF 5 VALIDATED</text>'
    )
    # 3 validated modalities
    today_pills = [
        ("RNA",     "36,601 genes · gene expression"),
        ("ATAC",    "323,500 peaks · chromatin"),
        ("Protein", "30–210 surface markers"),
    ]
    for i, (lbl, det) in enumerate(today_pills):
        modality_pill(parts, MOD_X, MOD_Y + 16 + i * (MOD_H + GAP), MOD_W, MOD_H,
                      lbl, det, validated=True)

    PHASE2_Y = MOD_Y + 16 + 3 * (MOD_H + GAP) + 18
    parts.append(
        f'<text x="{MOD_X}" y="{PHASE2_Y + 4}" fill="{WARN_AMBER}" font-family="{FONT}" '
        f'font-size="10" font-weight="700" letter-spacing="2.5">PHASE 2 · QURIESEQ INTEGRATION</text>'
    )
    phase2_pills = [
        ("Phospho", "early signaling · 5 min"),
        ("VDJ",     "clonal repertoire"),
    ]
    for i, (lbl, det) in enumerate(phase2_pills):
        modality_pill(parts, MOD_X, PHASE2_Y + 16 + i * (MOD_H + GAP), MOD_W, MOD_H,
                      lbl, det, validated=False)

    # Arrows from modality column → encoder block
    # Encoder fusion block
    ENC_X, ENC_Y = MOD_X + MOD_W + 60, LZ_Y + 140
    ENC_W, ENC_H = 280, 140
    parts.append(
        f'<rect x="{ENC_X}" y="{ENC_Y}" width="{ENC_W}" height="{ENC_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{CYAN_HI}" stroke-width="1.6" stroke-opacity="0.7"/>'
    )
    # Encoder eyebrow + title + lock
    parts.append(
        f'<text x="{ENC_X+20}" y="{ENC_Y+28}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">ENCODER</text>'
    )
    parts.append(lock_icon(ENC_X + ENC_W - 24, ENC_Y + 24, CYAN_HI, scale=1.1))
    parts.append(
        f'<text x="{ENC_X+20}" y="{ENC_Y+66}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="22" font-weight="700">Contrastive</text>'
    )
    parts.append(
        f'<text x="{ENC_X+20}" y="{ENC_Y+94}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="22" font-weight="700">multi-omics fusion</text>'
    )
    parts.append(
        f'<text x="{ENC_X+20}" y="{ENC_Y+122}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-style="italic">≈130K-param adapter on top · frozen substrate</text>'
    )

    # Arrows from modality pills (right edge) → encoder block (left edge)
    def arrow(x1, y1, x2, y2, color=CYAN, opacity=0.55):
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2-6}" y2="{y2}" stroke="{color}" '
            f'stroke-width="2" stroke-opacity="{opacity}" stroke-linecap="round"/>'
        )
        parts.append(
            f'<path d="M {x2-8} {y2-5} L {x2} {y2} L {x2-8} {y2+5}" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-opacity="{opacity+0.2}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # Centroids of TODAY pills → top of encoder block
    for i in range(3):
        py = MOD_Y + 16 + i * (MOD_H + GAP) + MOD_H // 2
        arrow(MOD_X + MOD_W + 2, py, ENC_X, ENC_Y + 70)
    # PHASE 2 pills → encoder (dashed, amber)
    for i in range(2):
        py = PHASE2_Y + 16 + i * (MOD_H + GAP) + MOD_H // 2
        parts.append(
            f'<line x1="{MOD_X + MOD_W + 2}" y1="{py}" x2="{ENC_X - 6}" y2="{ENC_Y + 120}" '
            f'stroke="{WARN_AMBER}" stroke-width="2" stroke-opacity="0.55" '
            f'stroke-dasharray="6 4" stroke-linecap="round"/>'
        )

    # Encoder block → latent output box
    LAT_X, LAT_Y = ENC_X + ENC_W + 50, ENC_Y + 8
    LAT_W, LAT_H = 264, 124
    parts.append(
        f'<rect x="{LAT_X}" y="{LAT_Y}" width="{LAT_W}" height="{LAT_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{PURPLE}" stroke-width="1.5" stroke-opacity="0.65"/>'
    )
    parts.append(
        f'<text x="{LAT_X+18}" y="{LAT_Y+26}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">LATENT SPACE</text>'
    )
    # Investor-friendly notation: avoid math-only glyphs (ℝ, ∈, superscripts)
    # that depend on a math font; PowerPoint Inter doesn't ship those either.
    parts.append(
        f'<text x="{LAT_X+LAT_W/2}" y="{LAT_Y+76}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="46" font-weight="700" text-anchor="middle" letter-spacing="-1">256-D</text>'
    )
    parts.append(
        f'<text x="{LAT_X+LAT_W/2}" y="{LAT_Y+106}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-style="italic" text-anchor="middle">frozen after pretrain · feeds every downstream task</text>'
    )
    arrow(ENC_X + ENC_W + 2, ENC_Y + 70, LAT_X, LAT_Y + LAT_H / 2, color=CYAN_HI, opacity=0.7)

    # Below schematic, a thin caption line
    parts.append(
        f'<text x="{LZ_X}" y="{LZ_Y + LZ_H - 18}" fill="{TEXT_DIM}" font-family="{FONT_BODY}" '
        f'font-size="13" font-style="italic">Encoder is shared by every downstream task — adapter, decomposed readout, Neural ODE temporal backbone.</text>'
    )

    # ====================================================================
    # Top-right zone: HERO 73% CALLOUT  (x=1056..1824, y=232..712)
    # ====================================================================
    RZ_X, RZ_Y, RZ_W, RZ_H = 1056, 232, 768, 480
    parts.append(
        f'<text x="{RZ_X}" y="{RZ_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">CROSS-CORPUS VALIDATION</text>'
    )
    parts.append(
        f'<line x1="{RZ_X + 240}" y1="{RZ_Y-6}" x2="{RZ_X + RZ_W}" y2="{RZ_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    # Card
    parts.append(
        f'<rect x="{RZ_X}" y="{RZ_Y + 16}" width="{RZ_W}" height="{RZ_H - 16}" rx="18" '
        f'fill="{SURFACE}" stroke="{CYAN_HI}" stroke-width="1.6" stroke-opacity="0.7"/>'
    )
    # Watermark glow behind the 73
    parts.append(
        f'<rect x="{RZ_X+40}" y="{RZ_Y+60}" width="{RZ_W-80}" height="{RZ_H-160}" '
        f'fill="url(#hero73)"/>'
    )
    # Big 73%
    hero_cx = RZ_X + RZ_W / 2
    parts.append(
        f'<text x="{hero_cx}" y="{RZ_Y + 240}" fill="{CYAN_HI}" font-family="{FONT}" '
        f'font-size="180" font-weight="700" text-anchor="middle" '
        f'letter-spacing="-4">73<tspan fill="{CYAN}" font-size="120">%</tspan></text>'
    )
    # Caption directly below
    parts.append(
        f'<text x="{hero_cx}" y="{RZ_Y + 296}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="20" font-weight="700" text-anchor="middle">'
        f'cross-corpus cell-type accuracy</text>'
    )
    parts.append(
        f'<text x="{hero_cx}" y="{RZ_Y + 324}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="16" font-weight="400" text-anchor="middle">'
        f'on Calderon 2019 · pre-registered pseudo-bulk centroid-NN</text>'
    )
    # 3 bullets in a row
    bullets = [
        ("●", "Independent dataset",   "different donors + protocols"),
        ("●", "Zero retraining",       "encoder frozen end-to-end"),
        ("●", "5 PBMC lineages",       "T (CD4/CD8), NK, B, Mono, DC"),
    ]
    by = RZ_Y + 380
    bx = RZ_X + 56
    bw = (RZ_W - 80) // 3
    for i, (dot, hdr, det) in enumerate(bullets):
        cx = bx + i * bw + bw // 2
        parts.append(
            f'<text x="{cx}" y="{by}" fill="{CYAN_HI}" font-family="{FONT}" '
            f'font-size="11" font-weight="700" text-anchor="middle">{dot}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{by+22}" fill="{TEXT_BODY}" font-family="{FONT}" '
            f'font-size="13" font-weight="700" text-anchor="middle">{hdr}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{by+40}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="11" font-weight="400" text-anchor="middle">{det}</text>'
        )

    # ====================================================================
    # Bottom zone: DOGMA-seq CREDIBILITY FOOTER  (x=96..1824, y=744..892)
    # ====================================================================
    DZ_X, DZ_Y, DZ_W, DZ_H = 96, 744, 1728, 152
    parts.append(
        f'<rect x="{DZ_X}" y="{DZ_Y}" width="{DZ_W}" height="{DZ_H}" rx="14" '
        f'fill="{SURFACE}" stroke="{DIVIDER}" stroke-width="1" stroke-opacity="0.9"/>'
    )
    # Left section: title
    parts.append(
        f'<text x="{DZ_X+24}" y="{DZ_Y+30}" fill="{LAVENDER}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2.5">PRETRAINING + PERTURBATION DATA</text>'
    )
    parts.append(
        f'<text x="{DZ_X+24}" y="{DZ_Y+62}" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="22" font-weight="700">DOGMA-seq</text>'
    )
    parts.append(
        f'<text x="{DZ_X+24}" y="{DZ_Y+88}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
        f'font-size="14" font-weight="400">Mimitou 2021 · <tspan font-style="italic">Nature Biotechnology</tspan></text>'
    )
    parts.append(
        f'<text x="{DZ_X+24}" y="{DZ_Y+114}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="12" font-weight="400" font-style="italic">Source of encoder pretraining + perturbation training data (ASAP-seq CRISPR sub-study).</text>'
    )

    # Right section: 4 specs as small fact pills, horizontally laid out
    # Note: "Trimodal" is explicitly banned from our voice (A1 v2 / A2 v4
    # decision) — use "same-cell" framing instead.
    facts = [
        ("Same-cell assay",    "RNA + ATAC + Protein", "all three from one cell"),
        ("Primary biology",    "Human PBMCs",          "not cell lines"),
        ("Scale",              "6 healthy donors",     "≈30K cells"),
        ("Provenance",         "Peer-reviewed",        "Nat Biotech 2021"),
    ]
    fact_x0 = DZ_X + 460
    fact_w = (DZ_W - 460 - 24) // 4
    for i, (eyebrow_t, line1, line2) in enumerate(facts):
        fx = fact_x0 + i * fact_w
        # vertical separator
        if i > 0:
            parts.append(
                f'<line x1="{fx-8}" y1="{DZ_Y+22}" x2="{fx-8}" y2="{DZ_Y+DZ_H-22}" '
                f'stroke="{DIVIDER}" stroke-width="1"/>'
            )
        parts.append(
            f'<text x="{fx+12}" y="{DZ_Y+34}" fill="{CYAN}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2.5">{eyebrow_t.upper()}</text>'
        )
        parts.append(
            f'<text x="{fx+12}" y="{DZ_Y+66}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="18" font-weight="700">{line1}</text>'
        )
        parts.append(
            f'<text x="{fx+12}" y="{DZ_Y+92}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="12" font-weight="400">{line2}</text>'
        )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Phase 6.5g.2 closure (2026-05-04, 73% Calderon pre-registered) · "
            "Mimitou et al., Nat Biotech 2021 (DOGMA-seq) · "
            "Architecture spec v1.1 §3.1 · "
            "Eval methodology docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md"
        ),
        slide_handle="A2 / 12",
    )

    parts.append("</svg>")
    return "\n".join(parts)


def build_png(svg_path: pathlib.Path, png_path: pathlib.Path):
    import cairosvg
    cairosvg.svg2png(
        bytestring=svg_path.read_bytes(),
        write_to=str(png_path),
        output_width=W, output_height=H,
    )


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "A2_encoder_evidence.svg"
    png_path = here / "A2_encoder_evidence_preview.png"
    svg = build_svg()
    svg_path.write_text(svg)
    print(f"wrote {svg_path} ({len(svg)} bytes)")
    build_png(svg_path, png_path)
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes)")
