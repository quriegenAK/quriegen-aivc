# Slide 9 Competitive Extraction (Kinga's deck)

**Source**: `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx`, slide 9
**Extracted**: 2026-05-15
**Method**: Programmatic parse of `ppt/slides/slide9.xml` — each `<p:sp>` shape's `<a:off>` position + `<a:t>` text content, sorted into a 2-D grid by y/x coordinates.

---

## Slide title + framing tagline

- **Title**: "COMPETITIVE LANDSCAPE"
- **Tagline (top of slide)**: *"We build tech to generate data & lab-in-the loop capabilities to develop the best causal model."*

## Three positioning pillars (callouts above the matrix)

Left → right:
1. **Proprietary data generation platform** (left, ~x=967K EMU)
2. **Causal and translational AIVC models** (center, ~x=4.3M EMU)
3. **Lab-in-the loop approach** (right, ~x=7.7M EMU)

These three phrases anchor the slide's "why us" narrative. Each maps to a subset of the matrix columns below.

---

## Competitors listed (7 total)

In the order Kinga presents them, top to bottom in the matrix:

1. **TAHOE**
2. **Deep Life**
3. **Turbine**
4. **Cytoreason**
5. **Valo**
6. **Noetik**
7. **Immunai**

Plus **QurieGen** as the reference row (always ✓ across all columns).

## Comparison dimensions (6 columns)

In Kinga's left → right column order:

1. **OWN SINGLE-CELL MULTI-OMICS** — does the company generate / own its own single-cell multi-omics data?
2. **DEEP INTRACELLULAR PROTEOMICS** — phospho / intracellular protein readouts (vs surface protein only)
3. **PROPRIETARY DATA GEN** — does the company run its own wet-lab data generation (separate from owning multi-omics specifically)?
4. **LAB-IN-THE-LOOP** — iterative model → wet-lab → model feedback loop
5. **VIRTUAL CELL MODEL** — has a foundation/virtual-cell model claim
6. **THERAPEUTICS PIPELINE(S)** — has internal drug/therapeutic candidates

---

## Kinga's matrix as-is

| Company | OWN SC-MULTI-OMICS | DEEP INTRACELLULAR PROTEOMICS | PROPRIETARY DATA GEN | LAB-IN-THE-LOOP | VIRTUAL CELL MODEL | THERAPEUTICS PIPELINE(S) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **QurieGen** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TAHOE | — | — | ✓ | ✓ | ✓ | ✓ |
| Deep Life | — | — | — | — | ✓ | — |
| Turbine | — | — | — | ✓ | ✓ | — |
| Cytoreason | — | — | ✓ | — | — | — |
| Valo | — | — | ✓ | — | — | ✓ |
| Noetik | — | — | ✓ | — | ✓ | ✓ |
| Immunai | — | — | — | ✓ | ✓ | — |

**Per-column QurieGen-only territory** (where Kinga claims zero competitor coverage):
- OWN SC-MULTI-OMICS — 0 / 7 competitors marked ✓
- DEEP INTRACELLULAR PROTEOMICS — 0 / 7 competitors marked ✓

**Per-column shared territory**:
- PROPRIETARY DATA GEN — TAHOE, Cytoreason, Valo, Noetik (4 / 7)
- LAB-IN-THE-LOOP — TAHOE, Turbine, Immunai (3 / 7)
- VIRTUAL CELL MODEL — TAHOE, Deep Life, Turbine, Noetik, Immunai (5 / 7)
- THERAPEUTICS PIPELINE(S) — TAHOE, Valo, Noetik (3 / 7)

---

## Visual treatment notes

- Single-page checkbox matrix. ✓ and `—` (em-dash) glyphs only — binary mark, no partial / qualitative scoring.
- QurieGen's row visually distinct from competitors (full ✓ row, same as the 3 positioning pillars highlighted above).
- 3 grouping callouts above the matrix (Proprietary Data Gen / Causal AIVC Models / Lab-in-the-loop) hint at how Kinga thinks the 6 columns cluster into 3 conceptual pillars — but the matrix below stays 6-column granular.

---

## Notes on Kinga's framing

1. **Strongest QurieGen-only claims** are "OWN single-cell multi-omics" and "DEEP intracellular proteomics" — both 0/7 competitor coverage on the slide. These are the binary differentiators Kinga is leading with.

2. **TAHOE is the strongest mapped competitor** in Kinga's framing — 4 / 6 ✓ (data gen, lab-in-loop, virtual cell, pipelines). Only missing the two QurieGen-only columns. Any "why us vs TAHOE" answer needs to land cleanly during diligence.

3. **Cytoreason and Valo are the narrowest mappings** — each only checks 1 column besides therapeutics. May be under-characterized; worth verifying during Step 2 research whether Kinga's checkbox treatment is accurate or whether they have additional capabilities not surfaced on this slide.

4. **No qualitative grading** — Kinga's checkbox is binary. Step 2 research should produce per-competitor depth that lets us replace binary checkmarks with substantive claims (e.g., "TAHOE: 100M-cell perturbation atlas published 2024" vs "✓ PROPRIETARY DATA GEN").

5. **Gaps to consider during Step 2 research** — possible adjacent competitors NOT on this list to investigate:
   - **Recursion Pharmaceuticals** (cell painting + foundation model, public company) — clear adjacency
   - **Insitro** (machine learning + functional genomics, ~$650M raised) — clear adjacency
   - **Owkin** (multi-modal foundation models in oncology + drug discovery) — adjacency depending on PBMC focus
   - **Genentech / NewLimit / Inceptive** — possibly out of scope (pure aging or RNA therapeutics)
   - **Cellarity** (cell-state ML) — adjacency
   - **Tempus AI** (clinical genomics, public) — different segment, may exclude
   - **Genomic Expression / SingulaBio / Vevo Therapeutics** — smaller players to verify

   Step 2 will research the 7 named + investigate the top 3-5 adjacencies to surface "Competitors NOT on Kinga's slide 9" section per the prompt.

6. **No funding stage info on slide** — Kinga doesn't surface valuation or stage for the comparison. Step 2 should add this dimension; it's an important investor-deck data point (e.g., "Recursion is public, $1B+ valuation; Noetik is seed-stage" changes how comparison should be framed).

---

## Next: Step 2 research

The 7 named competitors plus up to 3-5 adjacent competitors discovered during research. Target depth: 5-10 unique sources per competitor, per-competitor entry covering URL / funding / modality / data strategy / model architecture / validation evidence / honest gap analysis vs QurieGen. Estimated time: ~2 hours.
