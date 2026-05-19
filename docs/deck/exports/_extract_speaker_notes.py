"""Generate aivc_appendix_v5_speaker_notes.md — standalone companion doc.

Mechanical extraction from 14 content specs at docs/deck/content/*.md.
The .pptx remains canonical; this markdown is reader convenience.

Run: python3 docs/deck/exports/_extract_speaker_notes.py
"""
from __future__ import annotations
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTENT_DIR = REPO / "docs" / "deck" / "content"
OUT_PATH = REPO / "docs" / "deck" / "exports" / "aivc_appendix_v5_speaker_notes.md"

SPECS = [
    ("A1", "A1_system_architecture.md", "AIVC Foundation Model: System Architecture"),
    ("A2", "A2_encoder_substrate.md", "Multi-Omics Encoder: The Frozen Substrate"),
    ("A3", "A3_decomposed_readout.md", "Decomposed Readout: How Synergy Generalizes"),
    ("A4", "A4_temporal_neural_ode.md", "Temporal Dynamics via Neural ODE"),
    ("A5", "A5_causal_architecture.md", "Causal Architecture: Where Inference Becomes Causal"),
    ("B1", "B1_methodology_rigor.md", "Methodology: Three Datasets, Pre-Registered Evals"),
    ("B2", "B2_encoder_probe_verdict.md", "Encoder Probe: The Adapter Verdict"),
    ("B3", "B3_synergy_pre_demo.md", "Synergy Pre-Demo: Zero-Shot On Public Data"),
    ("C1", "C1_phase1_design.md", "QurieSeq Phase 1: The Data That Makes The Model"),
    ("C2", "C2_btk_jak_demo.md", "BTK + JAK Headline Demo: Pre-Registered Eval"),
    ("D1", "D1_quarterly_roadmap.md", "Quarterly Roadmap: Q3 2026 → Q4 2028"),
    ("D2", "D2_seed_allocation.md", "Seed Allocation: Where The $10M Goes"),
    ("E1", "E1_five_year_trajectory.md", "5-Year Trajectory: Pipeline + Clinical Maturation"),
    ("F1", "F1_competitive_positioning.md", "Integrated Causal Perturbation Platform"),
]


def extract_speaker_notes(spec_path: Path) -> str | None:
    """Extract everything between '## Speaker notes' and the next '## ' section.

    Matches "## Speaker notes" optionally followed by parenthetical text (e.g.
    "(NOT on slide — ...)") through to the next "## " heading at column 0 (or
    end of file). Stops at "## " but NOT at "### " (subsections within the
    speaker notes — Three-state framing, Technical glossary, etc.).
    """
    content = spec_path.read_text()
    pattern = r"## Speaker notes[^\n]*\n(.*?)(?=\n## (?!#)|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_headline(spec_path: Path) -> str:
    """Extract the slide's headline (first non-blank line under ## Headline).

    Strips markdown bold markers ** and trims whitespace.
    """
    content = spec_path.read_text()
    match = re.search(r"## Headline\s*\n+\*?\*?([^\n*]+)", content)
    if match:
        return match.group(1).strip()
    return ""


def build() -> str:
    out: list[str] = []
    out.append("# AIVC GeneLink Technical Appendix — Speaker Notes (v5)\n\n")
    out.append("**Companion to**: `aivc_appendix_v5.pptx` (21 slides, commit `19cc560`)\n\n")
    out.append("**Purpose**: Reader-convenience standalone version of speaker notes embedded in the pptx\n\n")
    out.append("**Authored**: 2026-05-17\n\n")
    out.append("**Source-of-truth**: Content specs at `docs/deck/content/*.md`\n\n")
    out.append(
        "This document concatenates the speaker notes from all 14 content slides "
        "for reviewers who prefer reading notes outside PowerPoint. Same content "
        "as embedded in the .pptx — single source of truth (the content specs). "
        "Sections retain their three-state framing + technical glossary + "
        "equations (where applicable) + diligence Q&A structure.\n\n"
    )
    out.append("---\n\n")

    # Table of contents
    out.append("## Table Of Contents\n\n")
    for sid, _, title in SPECS:
        anchor = f"slide-{sid.lower()}"
        out.append(f"- [Slide {sid} — {title}](#{anchor})\n")
    out.append("- [Appendix: Cross-Slide Glossary Reference](#appendix-cross-slide-glossary-reference)\n\n")
    out.append("---\n\n")

    # Per-slide sections
    for sid, filename, title in SPECS:
        spec_path = CONTENT_DIR / filename
        if not spec_path.exists():
            out.append(f"## Slide {sid}\n\n")
            out.append(f"### {title}\n\n")
            out.append(f"_(Content spec missing at {spec_path} — skipped)_\n\n---\n\n")
            continue
        notes = extract_speaker_notes(spec_path)
        headline = extract_headline(spec_path)
        out.append(f"## Slide {sid}\n\n")
        out.append(f"### {title}\n\n")
        if headline:
            out.append(f"**Headline**: {headline}\n\n")
        if notes:
            out.append(notes)
            out.append("\n\n")
        else:
            out.append(
                f"_(Speaker notes extraction failed for {sid} — verify content spec structure)_\n\n"
            )
        out.append("---\n\n")

    # Appendix — quick-reference glossary
    out.append("## Appendix: Cross-Slide Glossary Reference\n\n")
    out.append(
        "Key terms appearing across multiple slides — defined once here for quick "
        "reference. Per-slide sections above contain slide-specific definitions; "
        "this appendix is navigation convenience.\n\n"
    )
    out.append(
        "For the full master glossary with all ~100 terms and equation reading "
        "guides, see `docs/deck/research/glossary_2026_05_17.md`.\n\n"
    )

    key_terms = [
        ("Phase 1 / Phase 2",
         "QuRIE-seq wet-lab data generation phases. Phase 1 = Q3 2026 (5 donors × 5 timepoints × 4 modalities including phospho; ATAC at t=0 and t=180). Phase 2 = 2027 (20 donors + VDJ as 5th modality)."),
        ("Stage 3a / 3b / 3c / 4 / 5",
         "Model training stages. Stage 3a = current public-data engine (adapter on Mimitou). Stage 3b = BTK+JAK demo Q4 2026 (Phase 1 data). Stage 3c = causal architecture validation Q1-Q2 2027 (Phase 1 phospho signal). Stage 4 = VDJ + 20-donor scale 2027. Stage 5 = causal-ready + clinical handoff 2028."),
        ("QuRIE-seq",
         "Quriegen's proprietary single-cell multi-omics assay measuring RNA + Protein + Phospho-proteins from the same cell in a single workflow. Phospho is integral to the protocol — every QuRIE-seq run generates phospho. The defining capability."),
        ("DOGMA-seq",
         "Mimitou 2021 (Nature Biotechnology) single-cell method measuring RNA + ATAC + surface Protein on the same cell. Our encoder pretraining dataset; also source of perturbation training data (ASAP-seq CRISPR sub-study)."),
        ("Neural ODE",
         "Continuous-time dynamics model. Latent state evolves per learned differential equation `dz/dt = f_θ(z, perturbation, t)`. Handles irregular timepoint spacing (0/5/30/60/180 min) natively — discrete-time models would require resampling."),
        ("4-arm decomposed readout",
         "Decoder architecture: `ŷ = h_base + 𝟙[s]·Δ_stim + 𝟙[i]·Δ_inh + 𝟙[s∧i]·Δ_synergy`. Synergy arm captures non-additive combination biology. Zero-arm constraint (L2 λ=1.0) enables zero-shot compositional generalization."),
        ("Neumann propagation `(I−W)⁻¹·dₚ`",
         "Closed-form perturbation flow through learned sparse GRN. Stage 3c causal architecture mechanism. Requires spectral radius ρ(W) < 1; enforced by L1 sparsity during training."),
        ("Adapter strategy",
         "Lightweight neural net (~130K params) trained on top of frozen pretrained encoder. Approved by Stage 3 Part 1 ADAPTER_RECOMMENDED verdict. Enforced mechanically by AIVC_GRAD_GUARD environment flag."),
        ("Pseudo-bulk centroid-NN",
         "Cross-corpus evaluation method. Aggregate cells by cell-type label within each dataset to produce centroids; nearest-neighbor match across datasets gives accuracy. Pre-registered methodology."),
        ("73% Calderon",
         "Cross-corpus generalization result. Encoder trained on Mimitou DOGMA-seq, evaluated on independent Calderon 2019 PBMC dataset, 73% pseudo-bulk centroid-NN accuracy on 5-class lineage classification (B/T/NK/monocyte/DC). Chance = 20%; 3.65× chance."),
        ("0.57 ADAPTER_RECOMMENDED",
         "Stage 3 Part 1 verdict. Frozen encoder probe on Mimitou CRISPR perturbations scored 0.57 4-class accuracy (chance = 0.25, 2.27× chance). In pre-registered 0.50-0.80 band → adapter strategy approved (vs <0.50 = fine-tune required, ≥0.80 = encoder generalizes natively)."),
        ("Compositional generalization",
         "Model's ability to predict combinations from singletons. Train on BTK alone + JAK alone, predict BTK+JAK combo response zero-shot. The 4-arm decomposition + zero-arm constraint structurally supports this."),
        ("Phospho-proteomics",
         "Measurement of phosphorylated proteins. Reveals kinase activation state — immediate signaling response, minutes vs hours for RNA. Integral to QuRIE-seq from Phase 1 (Q3 2026). No public single-cell dataset has phospho on PBMCs under perturbation — structural moat."),
        ("BTK + JAK combo",
         "Headline demo target (Stage 3b). BTK = Bruton tyrosine kinase, BCR pathway, Ibrutinib target (approved CLL drug). JAK = Janus kinase, cytokine signaling, Ruxolitinib target (approved myelofibrosis drug). Combination has CLL clinical evidence (NCT02912754, PMID 26819050)."),
        ("STRING database (v12.0)",
         "Protein-Protein Interaction database (Szklarczyk et al., 2023, Nucleic Acids Research). Provides edge-existence priors for sparse learned GRN in Stage 3c. High-confidence edges (≥700 STRING score) face lower L1 sparsity pressure."),
        ("AIVC_GRAD_GUARD",
         "Environment variable flag (`AIVC_GRAD_GUARD=1`) blocking gradient flow into encoder during downstream training. Enforces frozen-encoder discipline mechanically. Set in all production runs post Stage 3 Part 1 verdict."),
        ("Calderon 2019",
         "Published PBMC dataset under stimulation. Independent from Mimitou — different lab, different donors, different protocol. Used as cross-corpus hold-out test for encoder generalization."),
        ("Pre-registered evaluation",
         "Eval methodology, metric, and thresholds committed in writing before running the eval (architecture spec v1.1). Prevents post-hoc cherry-picking. Both the 73% Calderon and 0.57 Mimitou CRISPR results were pre-registered."),
        ("Sci [PENDING IDENTIFICATION]",
         "Reference Kinga mentioned in her speaker notes ask. Systematic scan of slide text + content specs found no Sci-prefix library on slides. Possibilities: SciPlex, sci-RNA-seq, or unrelated to current scope. Awaiting Kinga clarification at v5 review."),
    ]
    for term, definition in key_terms:
        out.append(f"**{term}** — {definition}\n\n")

    return "".join(out)


if __name__ == "__main__":
    text = build()
    OUT_PATH.write_text(text)
    print(f"Wrote {OUT_PATH}  ({len(text):,} bytes, {OUT_PATH.stat().st_size:,} bytes on disk)")
