"""Build B1_three_datasets_methodology.svg + preview PNG.

Layout:
- Top zone: 3-column dataset cards (DOGMA cyan / Calderon green / Mimitou ASAP-seq lavender)
- Center callout: "No data overlap between roles"
- Bottom zone: 4-step pre-registration workflow

Section accent: green #4ADE80
Run: python3 docs/deck/assets/diagrams/_build_b1.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _deck_common import (  # type: ignore
    BG, SURFACE, CYAN, CYAN_HI, PURPLE, LAVENDER, OK_GREEN, WARN_AMBER,
    TEXT_TITLE, TEXT_BODY, TEXT_MUTED, TEXT_DIM, DIVIDER,
    FONT, FONT_BODY, START_X, W, H,
    svg_open, background, header, footer, arrow, render_png,
)


def build_svg() -> str:
    parts: list[str] = [svg_open("AIVC methodology rigor: three datasets, pre-registered evals")]
    background(parts)
    header(
        parts,
        appendix_id="B1",
        section="VALIDATION EVIDENCE",
        title="Three Datasets · Pre-Registered Evals · No Cherry-Picking",
        subtitle=(
            "Pretraining, validation, and perturbation data come from three independently produced public datasets · "
            "evaluation methodology registered before results were generated"
        ),
        eyebrow_color=OK_GREEN,
    )

    # ====================================================================
    # TOP ZONE: 3 dataset cards  (y=232..640)
    # ====================================================================
    CARD_W = 560
    CARD_H = 408
    CARD_GAP = 24
    CARD_Y = 232
    total_w = 3 * CARD_W + 2 * CARD_GAP   # 1728
    card_x0 = (W - total_w) // 2          # 96

    parts.append(
        f'<text x="{card_x0}" y="{CARD_Y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'THREE INDEPENDENT DATASETS · ROLE-SEPARATED</text>'
    )
    parts.append(
        f'<line x1="{card_x0 + 360}" y1="{CARD_Y-6}" x2="{card_x0 + total_w}" y2="{CARD_Y-6}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # Card data
    cards = [
        {
            "accent": CYAN_HI, "step": "01",
            "title": "DOGMA-seq",
            "credit": "Mimitou 2021 · Nat Biotech",
            "modalities": "RNA + ATAC + Protein",
            "facts": ["6 healthy donors", "≈30K cells", "Same-cell measurement"],
            "role_label": "ROLE",
            "role": "Encoder pretraining",
            "result_label": "RESULT",
            "result": "Encoder produces",
            "result_2": "256-D latent · frozen",
            "result_color": CYAN_HI,
        },
        {
            "accent": OK_GREEN, "step": "02",
            "title": "Calderon 2019",
            "credit": "Immune cell atlas",
            "modalities": "Bulk + scATAC · stim-driven PBMCs",
            "facts": ["Different donors", "Different protocol", "Independent study"],
            "role_label": "ROLE",
            "role": "Cross-corpus validation",
            "result_label": "RESULT",
            "result": "73% cell-type",
            "result_2": "accuracy · zero retraining",
            "result_color": OK_GREEN,
        },
        {
            "accent": LAVENDER, "step": "03",
            "title": "Mimitou ASAP-seq",
            "credit": "CRISPR sub-study",
            "modalities": "ATAC + Protein + HTO",
            "facts": ["CRISPR-perturbed", "CD4 T cells", "Mimitou lab pipeline"],
            "role_label": "ROLE",
            "role": "Perturbation adapter probe",
            "result_label": "RESULT",
            "result": "0.57 synergy 4-class",
            "result_2": "→ ADAPTER_RECOMMENDED",
            "result_color": LAVENDER,
        },
    ]

    for i, c in enumerate(cards):
        cx = card_x0 + i * (CARD_W + CARD_GAP)
        # Card surface
        parts.append(
            f'<rect x="{cx}" y="{CARD_Y + 16}" width="{CARD_W}" height="{CARD_H}" rx="14" '
            f'fill="{SURFACE}" stroke="{c["accent"]}" stroke-width="1.5" stroke-opacity="0.6"/>'
        )
        # Step number top-left
        parts.append(
            f'<text x="{cx + 24}" y="{CARD_Y + 56}" fill="{c["accent"]}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" letter-spacing="3">{c["step"]}</text>'
        )
        parts.append(
            f'<line x1="{cx + 24}" y1="{CARD_Y + 68}" x2="{cx + 64}" y2="{CARD_Y + 68}" '
            f'stroke="{c["accent"]}" stroke-width="1.5"/>'
        )
        # Title
        parts.append(
            f'<text x="{cx + 24}" y="{CARD_Y + 108}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="28" font-weight="700">{c["title"]}</text>'
        )
        # Credit
        parts.append(
            f'<text x="{cx + 24}" y="{CARD_Y + 134}" fill="{c["accent"]}" font-family="{FONT}" '
            f'font-size="12" font-weight="600" letter-spacing="2">{c["credit"].upper()}</text>'
        )
        # Modalities
        parts.append(
            f'<text x="{cx + 24}" y="{CARD_Y + 168}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="14" font-weight="700">{c["modalities"]}</text>'
        )
        # Facts (3 bullet lines)
        for j, f in enumerate(c["facts"]):
            parts.append(
                f'<text x="{cx + 24}" y="{CARD_Y + 196 + j*22}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
                f'font-size="13" font-weight="400">'
                f'<tspan fill="{c["accent"]}" font-weight="700">›</tspan>  {f}</text>'
            )
        # ROLE label box
        rl_y = CARD_Y + 282
        parts.append(
            f'<rect x="{cx + 24}" y="{rl_y}" width="{CARD_W - 48}" height="56" rx="8" '
            f'fill="{c["accent"]}" fill-opacity="0.10" stroke="{c["accent"]}" stroke-width="1" stroke-opacity="0.5"/>'
        )
        parts.append(
            f'<text x="{cx + 36}" y="{rl_y + 22}" fill="{c["accent"]}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2.5">ROLE</text>'
        )
        parts.append(
            f'<text x="{cx + 36}" y="{rl_y + 44}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="16" font-weight="700">{c["role"]}</text>'
        )
        # Result
        re_y = CARD_Y + 356
        parts.append(
            f'<text x="{cx + 24}" y="{re_y}" fill="{TEXT_MUTED}" font-family="{FONT}" '
            f'font-size="10" font-weight="700" letter-spacing="2.5">RESULT</text>'
        )
        parts.append(
            f'<text x="{cx + 24}" y="{re_y + 22}" fill="{c["result_color"]}" font-family="{FONT}" '
            f'font-size="18" font-weight="700">{c["result"]}</text>'
        )
        parts.append(
            f'<text x="{cx + 24}" y="{re_y + 42}" fill="{TEXT_BODY}" font-family="{FONT_BODY}" '
            f'font-size="13" font-weight="400">{c["result_2"]}</text>'
        )

    # ====================================================================
    # CENTER CALLOUT: "No data overlap between roles"
    # ====================================================================
    CO_Y = CARD_Y + 16 + CARD_H + 36   # 696
    parts.append(
        f'<line x1="{START_X + 240}" y1="{CO_Y}" x2="{W - START_X - 240}" y2="{CO_Y}" '
        f'stroke="{OK_GREEN}" stroke-width="1.5" stroke-opacity="0.6"/>'
    )
    parts.append(
        f'<rect x="{W//2 - 220}" y="{CO_Y - 18}" width="440" height="36" rx="18" '
        f'fill="{BG}" stroke="{OK_GREEN}" stroke-width="1.5" stroke-opacity="0.75"/>'
    )
    parts.append(
        f'<text x="{W//2}" y="{CO_Y + 6}" fill="{OK_GREEN}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="3" text-anchor="middle">'
        f'NO DATA OVERLAP BETWEEN ROLES</text>'
    )

    # ====================================================================
    # BOTTOM ZONE: 4-step pre-registration workflow  (y≈740..900)
    # ====================================================================
    WF_Y = 760
    parts.append(
        f'<text x="{START_X}" y="{WF_Y - 12}" fill="{TEXT_MUTED}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">'
        f'PRE-REGISTRATION WORKFLOW — METHODOLOGY LOCKED BEFORE RESULTS</text>'
    )
    parts.append(
        f'<line x1="{START_X + 580}" y1="{WF_Y - 18}" x2="{W - START_X}" y2="{WF_Y - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # 4 stepped boxes with arrows
    STEP_W = 380
    STEP_GAP = 56
    total_wf = 4 * STEP_W + 3 * STEP_GAP   # 1688
    wf_x0 = (W - total_wf) // 2            # 116
    STEP_H = 110
    steps = [
        ("1", "Spec written",     "Architecture spec v1.1 (May 2026)"),
        ("2", "Eval defined",     "Pseudo-bulk centroid-NN methodology"),
        ("3", "Results generated","Pretrained encoder on Calderon + Mimitou CRISPR"),
        ("4", "Verdict applied",  "ADAPTER_RECOMMENDED · per pre-registered threshold"),
    ]
    for i, (num, t, desc) in enumerate(steps):
        sx = wf_x0 + i * (STEP_W + STEP_GAP)
        # Step box (small card)
        parts.append(
            f'<rect x="{sx}" y="{WF_Y}" width="{STEP_W}" height="{STEP_H}" rx="12" '
            f'fill="{SURFACE}" stroke="{OK_GREEN}" stroke-width="1.2" stroke-opacity="0.55"/>'
        )
        # Step number badge
        parts.append(
            f'<circle cx="{sx + 28}" cy="{WF_Y + 30}" r="14" '
            f'fill="{OK_GREEN}" fill-opacity="0.20" stroke="{OK_GREEN}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{sx + 28}" y="{WF_Y + 35}" fill="{OK_GREEN}" font-family="{FONT}" '
            f'font-size="14" font-weight="700" text-anchor="middle">{num}</text>'
        )
        # Title
        parts.append(
            f'<text x="{sx + 56}" y="{WF_Y + 36}" fill="{TEXT_TITLE}" font-family="{FONT}" '
            f'font-size="18" font-weight="700">{t}</text>'
        )
        # Description
        parts.append(
            f'<text x="{sx + 24}" y="{WF_Y + 68}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
            f'font-size="12" font-weight="400">{desc}</text>'
        )
        # Arrow to next
        if i < 3:
            arrow(parts, sx + STEP_W + 4, WF_Y + STEP_H // 2,
                  sx + STEP_W + STEP_GAP - 4, WF_Y + STEP_H // 2,
                  color=OK_GREEN, opacity=0.7)

    # Lower line callout
    parts.append(
        f'<text x="{W//2}" y="{WF_Y + STEP_H + 32}" fill="{TEXT_DIM}" font-family="{FONT_BODY}" '
        f'font-size="13" font-weight="400" font-style="italic" text-anchor="middle">'
        f'← methodology pre-registered BEFORE any results were observed · '
        f'no post-hoc threshold adjustment, no goalpost moving →</text>'
    )

    # ---- Footer ----
    footer(
        parts,
        source_text=(
            "Source: Architecture spec v1.1 · Phase 6.5g.2 closure (2026-05-04, 73% Calderon pre-registered) · "
            "Stage 3 Part 1 verdict (2026-05-11, 0.57 ADAPTER_RECOMMENDED) · "
            "Eval methodology docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md"
        ),
        slide_handle="B1 / 12",
        handle_color=OK_GREEN,
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    svg_path = here / "B1_three_datasets_methodology.svg"
    png_path = here / "B1_three_datasets_methodology_preview.png"
    svg_path.write_text(build_svg())
    print(f"wrote {svg_path}")
    render_png(svg_path, png_path)
    print(f"wrote {png_path}")
