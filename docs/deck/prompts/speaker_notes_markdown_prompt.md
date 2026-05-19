# Standalone Speaker Notes Companion Doc — Cowork Task

**Owner**: Cowork (execution)
**Estimated time**: ~15-20 min
**Input**: 14 content specs at `docs/deck/content/*.md` (commit `19cc560` on origin/main)
**Output**: `docs/deck/exports/aivc_appendix_v5_speaker_notes.md`
**Strategy**: Mechanical concatenation of all 14 slides' `## Speaker notes` sections into a single navigable markdown doc

---

## Context

Step 5 embedded comprehensive speaker notes into the pptx v5. The notes are accessible via PowerPoint's View → Notes / Presenter View. However, reviewers (Kinga + Jan) may prefer to read notes on mobile, in a browser, or without opening PowerPoint at all.

This step generates a **standalone markdown companion doc** containing all 14 slides' speaker notes content concatenated with clear navigation, so reviewers can read the technical glossary + diligence Q&As without launching PowerPoint.

The .pptx remains the canonical deliverable; this markdown doc is a reader-convenience companion.

---

## What To Build

Single deliverable: `docs/deck/exports/aivc_appendix_v5_speaker_notes.md`

### Document Structure

```markdown
# AIVC GeneLink Technical Appendix — Speaker Notes (v5)

**Companion to**: `aivc_appendix_v5.pptx` (21 slides, commit `19cc560`)
**Purpose**: Reader-convenience standalone version of speaker notes embedded in the pptx
**Authored**: 2026-05-17
**Source-of-truth**: Content specs at `docs/deck/content/*.md`

This document concatenates the speaker notes from all 14 content slides for reviewers who prefer reading notes outside PowerPoint. Same content as embedded in the .pptx — single source of truth.

---

## Table Of Contents

- [Slide A1 — AIVC Foundation Model: System Architecture](#slide-a1)
- [Slide A2 — Multi-Omics Encoder: The Frozen Substrate](#slide-a2)
- [Slide A3 — Decomposed Readout: How Synergy Generalizes](#slide-a3)
- [Slide A4 — Temporal Dynamics via Neural ODE](#slide-a4)
- [Slide A5 — Causal Architecture: Where Inference Becomes Causal](#slide-a5)
- [Slide B1 — Methodology: Three Datasets, Pre-Registered Evals](#slide-b1)
- [Slide B2 — Encoder Probe: The Adapter Verdict](#slide-b2)
- [Slide B3 — Synergy Pre-Demo: Zero-Shot On Public Data](#slide-b3)
- [Slide C1 — QurieSeq Phase 1: The Data That Makes The Model](#slide-c1)
- [Slide C2 — BTK + JAK Headline Demo: Pre-Registered Eval](#slide-c2)
- [Slide D1 — Quarterly Roadmap: Q3 2026 → Q4 2028](#slide-d1)
- [Slide D2 — Seed Allocation: Where The $10M Goes](#slide-d2)
- [Slide E1 — 5-Year Trajectory: Pipeline + Clinical Maturation](#slide-e1)
- [Slide F1 — Integrated Causal Perturbation Platform](#slide-f1)
- [Appendix: Cross-Slide Glossary Reference](#appendix-glossary)

---

## Slide A1
### AIVC Foundation Model: System Architecture

**Headline**: One unified foundation model. Three input modalities. Four perturbation states. Continuous time. Pathway-grounded outputs.

[Extract and embed full Speaker notes section from A1 content spec verbatim]

---

## Slide A2
### Multi-Omics Encoder: The Frozen Substrate

**Headline**: Multi-omics encoder — trained on public, ready for proprietary

[Extract and embed full Speaker notes section from A2 content spec verbatim]

---

[Continue for all 14 slides in order]

---

## Appendix: Cross-Slide Glossary Reference

Key terms used across multiple slides — defined once here for quick reference. Each per-slide section above contains slide-specific definitions; this appendix is for navigation convenience.

[Brief 1-line definitions for ~15-20 most-referenced terms across the deck — pulled from master glossary, kept terse for cross-reference utility]
```

---

## How To Extract Speaker Notes From Each Content Spec

Each content spec at `docs/deck/content/*.md` has a `## Speaker notes` section. Extract everything between `## Speaker notes` and the next `## ` heading.

Recommended extraction approach:

```python
import re
from pathlib import Path

CONTENT_DIR = Path("docs/deck/content")
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

def extract_speaker_notes(spec_path):
    """Extract everything between ## Speaker notes and the next ## section."""
    content = spec_path.read_text()
    # Match "## Speaker notes" (possibly with trailing text like "(NOT on slide — ...)") 
    # through to next "## " section heading
    pattern = r"## Speaker notes[^\n]*\n(.*?)(?=\n## (?!#)|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_headline(spec_path):
    """Extract the slide's headline text for the section subtitle."""
    content = spec_path.read_text()
    # Match "## Headline\n[non-empty content]" — first non-blank line under it
    match = re.search(r"## Headline\s*\n+\*?\*?([^\n*]+)", content)
    if match:
        return match.group(1).strip()
    return ""

# Build the doc
output = ["# AIVC GeneLink Technical Appendix — Speaker Notes (v5)\n"]
output.append("**Companion to**: `aivc_appendix_v5.pptx` (21 slides, commit `19cc560`)\n")
output.append("**Purpose**: Reader-convenience standalone version of speaker notes embedded in the pptx\n")
output.append("**Authored**: 2026-05-17\n")
output.append("**Source-of-truth**: Content specs at `docs/deck/content/*.md`\n\n")
output.append("This document concatenates the speaker notes from all 14 content slides for reviewers ")
output.append("who prefer reading notes outside PowerPoint. Same content as embedded in the .pptx — ")
output.append("single source of truth.\n\n---\n\n")

# Table of contents
output.append("## Table Of Contents\n\n")
for slide_id, _, title in SPECS:
    output.append(f"- [Slide {slide_id} — {title}](#slide-{slide_id.lower()})\n")
output.append("- [Appendix: Cross-Slide Glossary Reference](#appendix-glossary)\n\n---\n\n")

# Each slide
for slide_id, filename, title in SPECS:
    spec_path = CONTENT_DIR / filename
    notes = extract_speaker_notes(spec_path)
    headline = extract_headline(spec_path)
    
    output.append(f"## Slide {slide_id}\n")
    output.append(f"### {title}\n\n")
    if headline:
        output.append(f"**Headline**: {headline}\n\n")
    if notes:
        output.append(notes)
        output.append("\n\n")
    else:
        output.append(f"_(Speaker notes extraction failed for {slide_id} — verify content spec structure)_\n\n")
    output.append("---\n\n")

# Optional appendix — quick-reference glossary
output.append("## Appendix: Cross-Slide Glossary Reference\n\n")
output.append("Key terms appearing across multiple slides — defined once here for quick reference. ")
output.append("Per-slide sections above contain slide-specific definitions; this appendix is navigation convenience.\n\n")
output.append("For full glossary with all ~100 terms and equation reading guides, see ")
output.append("`docs/deck/research/glossary_2026_05_17.md`.\n\n")

# Curated 15-20 most-referenced terms, terse definitions (pull from master glossary)
key_terms = [
    ("Phase 1 / Phase 2", "QuRIE-seq wet-lab data generation phases. Phase 1 = Q3 2026 (5 donors, 4 modalities including phospho). Phase 2 = 2027 (20 donors + VDJ)."),
    ("Stage 3a / 3b / 3c / 4 / 5", "Model training stages. 3a = current public-data engine. 3b = BTK+JAK demo Q4 2026. 3c = causal architecture validation Q1-Q2 2027. 4 = scale 2027. 5 = causal-ready 2028."),
    ("QuRIE-seq", "Quriegen's proprietary single-cell multi-omics assay measuring RNA + Protein + Phospho-proteins from the same cell. Phospho integral to the protocol."),
    ("DOGMA-seq", "Mimitou 2021 single-cell method measuring RNA + ATAC + surface Protein. Our encoder pretraining dataset."),
    ("Neural ODE", "Continuous-time dynamics model. Latent state evolves per learned differential equation. Handles irregular timepoint spacing (0/5/30/60/180 min) natively."),
    ("4-arm decomposed readout", "Decoder architecture: ŷ = h_base + 𝟙[s]·Δ_stim + 𝟙[i]·Δ_inh + 𝟙[s∧i]·Δ_synergy. Synergy arm captures non-additive combination biology."),
    ("Neumann propagation (I−W)⁻¹ dₚ", "Closed-form perturbation flow through learned sparse GRN. Stage 3c causal architecture mechanism."),
    ("Adapter strategy", "Lightweight neural net on top of frozen pretrained encoder. Approved by Stage 3 Part 1 ADAPTER_RECOMMENDED verdict."),
    ("Pseudo-bulk centroid-NN", "Cross-corpus evaluation method. Aggregate cells by type into centroids; nearest-neighbor match across datasets gives accuracy."),
    ("73% Calderon", "Cross-corpus generalization result. Encoder trained on Mimitou DOGMA-seq, evaluated on independent Calderon 2019 PBMC dataset, 73% on 5-class lineage classification (chance = 20%)."),
    ("0.57 ADAPTER_RECOMMENDED", "Stage 3 Part 1 verdict result. Frozen encoder probe on Mimitou CRISPR perturbations scored 0.57 4-class accuracy (chance = 0.25, 2.27× chance). Triggers adapter strategy approval."),
    ("Compositional generalization", "Model's ability to predict combinations from singletons. Train on BTK alone + JAK alone, predict BTK+JAK combo zero-shot."),
    ("Phospho-proteomics", "Measurement of phosphorylated proteins. Reveals kinase activation state — immediate signaling response. Integral to QuRIE-seq from Phase 1."),
    ("BTK + JAK combo", "Headline demo target. BTK (Bruton tyrosine kinase, BCR pathway, Ibrutinib target). JAK (Janus kinase, cytokine signaling, Ruxolitinib target). Combination has CLL clinical evidence."),
    ("STRING database", "Protein-Protein Interaction database. Provides edge-existence priors for sparse learned GRN in Stage 3c."),
    ("AIVC_GRAD_GUARD", "Environment variable enforcing frozen-encoder discipline. Blocks gradient flow into encoder during downstream training."),
    ("Sci [PENDING IDENTIFICATION]", "Reference Kinga mentioned in her speaker notes ask. Systematic scan found no Sci-prefix library on slides. Possibilities: SciPlex, sci-RNA-seq, or misread. Awaiting clarification."),
]

for term, definition in key_terms:
    output.append(f"**{term}** — {definition}\n\n")

# Write
out_path = Path("docs/deck/exports/aivc_appendix_v5_speaker_notes.md")
out_path.write_text("".join(output))
print(f"Wrote {out_path}: {out_path.stat().st_size} bytes")
```

---

## Acceptance Criteria

- ✓ Single output file at `docs/deck/exports/aivc_appendix_v5_speaker_notes.md`
- ✓ All 14 slides represented in order (A1, A2, A3, A4, A5, B1, B2, B3, C1, C2, D1, D2, E1, F1)
- ✓ Each slide section has headline + extracted speaker notes
- ✓ Table of contents at top with anchor links
- ✓ Cross-slide glossary appendix at end (~15-20 key terms, terse definitions)
- ✓ Total document size ~100-150 KB (mirrors the ~124k chars of speaker notes content)
- ✓ Markdown renders cleanly in GitHub viewer + standard markdown viewers
- ✓ No content duplication — extraction is verbatim from content specs, not paraphrased

### Quality checks (run after write)

```bash
# Confirm all 14 slide sections present
grep -c "^## Slide " docs/deck/exports/aivc_appendix_v5_speaker_notes.md
# Expected: 14

# Confirm "Three-state framing" present from all slides
grep -c "Three-state framing" docs/deck/exports/aivc_appendix_v5_speaker_notes.md
# Expected: 14+

# Confirm "Technical glossary" present from all slides  
grep -c "Technical glossary" docs/deck/exports/aivc_appendix_v5_speaker_notes.md
# Expected: 14+

# Confirm all 84 "If asked:" Q&As present
grep -c "If asked" docs/deck/exports/aivc_appendix_v5_speaker_notes.md
# Expected: 84

# Confirm doc size in expected range
wc -c docs/deck/exports/aivc_appendix_v5_speaker_notes.md
# Expected: 100,000 - 160,000 bytes
```

---

## Deliverable

Single commit:

```bash
git add docs/deck/exports/aivc_appendix_v5_speaker_notes.md
git commit -m "docs(deck): standalone speaker notes companion doc for v5"
git push origin main
```

Single-line commit message.

---

## What's Out Of Scope

- Modifying any content spec (specs are source-of-truth)
- Regenerating the pptx (v5 stays as the canonical pptx)
- Editing master glossary
- Phase 4A visual polish
- Any new content authoring — pure mechanical extraction

---

## Risks To Flag

1. **Anchor link generation in markdown TOC**: GitHub markdown auto-generates anchors from headings using lowercase + hyphens. The TOC `[Slide A1 — Title](#slide-a1)` should work for GitHub renderer. Other markdown renderers may differ slightly; test in at least GitHub web view before declaring complete.

2. **Speaker notes extraction regex** depends on each content spec having `## Speaker notes` followed by `## ` next-section heading. All 14 specs verified to have this pattern post-Step-5 (notesSlide14.xml in pptx confirms extraction worked). Should be clean.

3. **No content alteration during extraction** — speaker notes content stays verbatim from spec. If a spec has a typo, the markdown has the same typo. Don't fix-as-you-extract; that creates spec-vs-markdown drift.

4. **File size at ~120-150KB** is large but reasonable for the content density. GitHub renders files up to ~1MB cleanly.

5. **Single source of truth**: the .pptx remains canonical for the deck; this markdown is reader convenience. If the .pptx is rebuilt for v6 with edits, this markdown doc must be regenerated to stay in sync. Worth a note at the top of the doc OR a future Phase 4 task to auto-regenerate when pptx rebuilds.

---

## After This Lands

Cover note (revised version Ash has on disk) gets sent with:
1. `aivc_appendix_v5.pptx`
2. `aivc_appendix_v5_speaker_notes.md` (this deliverable)

Both shipped to Kinga + Jan together. They have full speaker notes access in two formats — PowerPoint Notes view OR standalone markdown.
