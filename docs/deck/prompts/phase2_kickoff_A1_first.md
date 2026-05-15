# Phase 2 Kickoff — A1 SVG Generation + Visual Style Lock

**Owner**: Cowork (execution)
**Estimated time**: 30-45 minutes for A1 + style extraction
**Dependencies**: Source deck at `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx` (gitignored, local-only)

---

## Context

Stage 3 of deck production. We've completed:
- **Phase 1**: 12 slide content specs in `docs/deck/content/` (commits f5531ce through e27cbbb on origin/main)
- **Now Phase 2**: SVG diagram generation, one per slide

We're starting with **A1 (system architecture)** as the visual anchor. Every subsequent SVG must match A1's visual conventions. Get A1 right; the rest follow.

---

## Task 1 — Extract Visual Style From Kinga's Deck

**Input**: `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx` (108MB, gitignored, local file)

**Required outputs**:

### 1a. Color palette
Extract dominant colors from Kinga's primary slides (especially slides 8, 9, 37 — the technical depth slides we map to). Save as:

`docs/deck/assets/color_palette.md`

Format:
```markdown
# QurieGen Deck Color Palette
**Source**: Extracted from QurieGen_SEED_ROUND_05_2026_new.pptx, slides 8/9/37

| Role | Hex | RGB | Where used in source |
|---|---|---|---|
| Primary brand | #...... | ... | Title slides, hero callouts |
| Primary accent | #...... | ... | Highlighted text, key numbers |
| Secondary accent | #...... | ... | Secondary highlights |
| Dark background | #...... | ... | Slide 8 background |
| Light background | #...... | ... | Body slide backgrounds |
| Body text | #...... | ... | Default text |
| Muted text | #...... | ... | Captions, footnotes |
| Success / green | #...... | ... | Checkmark icons, GREEN verdicts |
| Warning / amber | #...... | ... | AMBER verdicts |
| Danger / red | #...... | ... | RED verdicts, rejected options |
```

Extraction method: read XML inside the .pptx (it's a ZIP archive of XML files). Theme colors live in `ppt/theme/theme1.xml`. Slide-specific color overrides live in `ppt/slides/slideN.xml`. Use python-pptx if convenient.

### 1b. Typography
Extract font choices (title font, body font, monospace if used). Save as:

`docs/deck/assets/typography.md`

Format:
```markdown
# QurieGen Deck Typography
**Source**: Extracted from QurieGen_SEED_ROUND_05_2026_new.pptx

| Role | Font family | Size hint |
|---|---|---|
| Slide title | ... | 32-44pt |
| Section header | ... | 24-28pt |
| Body text | ... | 14-18pt |
| Callout | ... | 18-22pt |
| Caption / footnote | ... | 10-12pt |
| Code / monospace (if any) | ... | 11-13pt |
```

If Kinga uses non-default fonts that may not be available in matplotlib, fall back to closest open-source equivalents (e.g., Helvetica → Arial, Avenir → Inter, Calibri → Source Sans Pro). Document fallbacks in the file.

### 1c. Slide dimensions + aspect ratio
Confirm the source deck's slide size (likely 16:9, 1920×1080 or 13.333"×7.5") so SVG viewBox matches.

Save dimensions to top of `color_palette.md` or in a separate `dimensions.md`.

**Commit these style files first** as a separate commit before generating A1.

---

## Task 2 — Generate A1 SVG (Visual Anchor)

**Spec**: `docs/deck/content/A1_system_architecture.md`

**Output**: `docs/deck/assets/diagrams/A1_system_architecture.svg`

### A1 visual structure (from spec)

A horizontal flow diagram with 5 stacked blocks left to right:

```
INPUT → ENCODER → TEMPORAL → READOUT → OUTPUT
```

Plus two annotation rows below:
- Row 2 — validation status per block (✅ / 🟡 / ⏸ icons)
- Row 3 — single-keyword invariant guarantees per block

Read the full A1 spec for visual details. Key requirements:

1. **5 horizontal blocks** with arrows between (→), each labeled with title + content
2. **Frozen encoder** indicated with lock icon 🔒
3. **Multi-omics encoder input** shows RNA + ATAC + Protein (Phase 2 modalities phospho + VDJ NOT shown on A1 — that's A2's job)
4. **Row 2 status icons**: ✅ for completed Stage 1+2, 🟡 for Stage 3a in-flight, ⏸ for Stage 3b/3c pending
5. **Row 3 keywords**: modality-agnostic / cross-corpus transfer / irregular timepoint handling / compositional generalization / biological interpretability

### Visual style requirements

- Use the color palette extracted in Task 1a
- Use the typography extracted in Task 1b (or matplotlib-compatible fallback)
- Match Kinga's deck aesthetic — investors should feel A1 is from the same document family
- 1920×1080 viewBox (or whatever Task 1c determines)
- SVG output (not PNG) for scalability

### Implementation guidance

Recommended tool: **matplotlib** with custom `Rectangle` patches and text annotations. Save as SVG via `plt.savefig('A1_system_architecture.svg', format='svg', bbox_inches='tight')`.

If matplotlib feels limiting for this layout, consider:
- `svgwrite` for direct SVG generation
- `drawsvg` for higher-level SVG primitives
- Inkscape + Python via `inkex` (overkill for this)

Keep it lean — no decorative gradients, no shadows beyond what Kinga's source uses, no chart-junk.

### Iteration protocol

After generating A1, save the SVG and commit. Don't generate A2-E1 yet. Ash will review A1, request adjustments if needed, then we batch the remaining 11.

---

## Task 3 — Deliverable Summary

After completing Tasks 1 and 2, commit in this order:

```bash
# Commit 1: visual style assets
git add docs/deck/assets/color_palette.md \
        docs/deck/assets/typography.md
git commit -m "docs(deck): extract visual style assets from Kinga's source deck

Colors and typography pulled from slides 8/9/37 (the technical-depth
slides we map to in the appendix). All subsequent diagram generation
references these as the visual style source of truth."
git push origin main

# Commit 2: A1 SVG
git add docs/deck/assets/diagrams/A1_system_architecture.svg
git commit -m "docs(deck): A1 system architecture SVG

Visual anchor for the technical appendix. 5-block horizontal flow
(INPUT → ENCODER → TEMPORAL → READOUT → OUTPUT) with validation
status row + invariant-guarantee row.

Matches Kinga's deck color palette + typography per
docs/deck/assets/color_palette.md and typography.md.

Awaiting Ash review before batch generation of A2-E1."
git push origin main
```

---

## Acceptance Criteria — What Ash Will Check

When you ship A1, Ash will verify:

1. **Visual coherence with Kinga's deck** — open Kinga's source PPTX side-by-side. A1 should feel native, not foreign.
2. **5-block flow is clear** — INPUT → ENCODER → TEMPORAL → READOUT → OUTPUT scans in 5 seconds
3. **Validation status row is honest** — Stage 1+2 ✅, Stage 3a 🟡, Stage 3b/3c ⏸ correctly marked
4. **No chart-junk** — clean, technical, minimalist
5. **Lock icon on encoder** — visually conveys "frozen substrate" without needing to read text
6. **Readable at slide size** — text legible when SVG is sized to fill a 16:9 slide

If any criterion fails, expect iteration before the batch step.

---

## What Comes After A1 Lands

Once Ash approves A1, you'll get a single batch prompt for A2-E1 (11 diagrams). Each has its own spec in `docs/deck/content/<slide>.md` with a "Followup prompt for Cowork" section at the bottom.

The batch will reference A1 as the visual style baseline. Estimated batch time: 2-3 hours of generation + iteration.

After all 12 SVGs land, Phase 3 (.pptx assembly via Cowork's pptx skill) and Phase 4 (visual polish via Claude Design on hero diagrams) follow.

---

## Risks To Flag

1. **Source .pptx is gitignored** — only exists locally at `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx`. Confirm it's there before running Task 1. If not, ask Ash.
2. **python-pptx may not extract embedded theme colors cleanly** — fallback: unzip the .pptx (it's a ZIP), read `ppt/theme/theme1.xml` directly.
3. **Fonts referenced in Kinga's deck may not be installable on Cowork's matplotlib environment** — document the fallbacks used.
4. **The "multi-omics with 3 validated + 2 Phase 2" framing from A2** doesn't apply to A1 — A1 shows the CURRENT architecture flow (RNA + ATAC + Protein in encoder). A2 introduces the extensibility framing. Don't conflate them.

---

## Open Questions To Resolve Before A2-E1 Batch

These don't block A1 generation but should be answered during A1 iteration:

1. **How does Kinga's deck handle "phase 2 / coming soon" elements?** — affects A2's 🟡 outlined modality treatment
2. **What's the source-deck color for the BTK+JAK story / Aduro narrative?** — should propagate to C2's headline-demo accent
3. **Are there reusable icons in Kinga's deck** (checkmark, lock, arrow styles)? — extract any that apply

Document findings in `docs/deck/assets/icon_inventory.md` during A1 work for future batch reference.

---

## Tool Selection Confirmation

This task is for **Cowork** (Python execution agent on Mac filesystem), not Claude Design.

Reason: matplotlib SVG generation is structural, not artistic. Design polish on hero diagrams happens in Phase 4 after structural SVGs are stable.

---

## Status After Phase 2 Completes

12 SVG diagrams in `docs/deck/assets/diagrams/`:
- A1_system_architecture.svg
- A2_encoder_evidence.svg
- A3_decomposed_readout.svg
- A4_temporal_dynamics.svg
- B1_three_datasets_methodology.svg
- B2_adapter_verdict.svg
- B3_mechanism_pre_demo.svg
- C1_phase1_experimental_design.svg
- C2_btk_jak_demo_plan.svg
- D1_quarterly_roadmap.svg
- D2_seed_allocation.svg
- E1_five_year_trajectory.svg

Plus style references in `docs/deck/assets/{color_palette,typography}.md`.

Phase 3 (.pptx assembly) unlocks.
