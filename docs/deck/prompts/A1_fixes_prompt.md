# A1 Visual Fixes — Iteration v2

**Owner**: Cowork (execution)
**Estimated time**: 20-30 min
**Input file**: `docs/deck/assets/diagrams/A1_system_architecture.svg`
**Build script**: `docs/deck/assets/diagrams/_build_a1.py`
**Reference (current)**: commit `42cd59f`

---

## Context

A1 v1 shipped clean — color palette extraction + typography are production-grade and locked. Visual style approved for the deck. Ash reviewed the rendered output and identified 5 specific fixes before A1 becomes the visual anchor for batching A2-E1.

**Strategic principle**: every issue caught here is an issue NOT propagated to 11 other diagrams. This iteration prevents 11x more iteration later.

---

## The 5 Fixes Required

### Fix 1 — Resolve lock-icon collision with ENCODER title

**Problem**: The lock icon and the word "ENCODER" overlap at the same vertical band. The lock icon sits over the letter "R" of "ENCODER", and the explicit "FROZEN" text label below it adds redundancy. Three visual signals (lock icon + "FROZEN" text + "FROZEN + ADAPTER" subtitle) all conveying the same thing.

**Fix**:
- Move the lock icon to the **upper-right corner** of the ENCODER card, mirrored opposite the "02" step number. Roughly at `x=720, y=265` (right edge of card, same vertical band as the step number).
- **Remove** the explicit "FROZEN" text label below the lock icon (line 43 in current SVG).
- Keep the subtitle "FROZEN + ADAPTER" — that's sufficient with the lock icon visible.
- Result: one lock icon (top-right) + subtitle (below title). Two signals, not three.

---

### Fix 2 — Strip "Trimodal" from encoder body bullet

**Problem**: Current ENCODER body bullet reads `› Trimodal → 256-D latent`. We made an explicit decision in A2 v4 (commit `1b61964`) to strip "trimodal" from our voice — it caps platform ambition at 3 modalities when we're a 5-modality platform with 3 currently validated.

**Fix**:
- Change `Trimodal → 256-D latent` to `Multi-omics → 256-D latent`.
- Alternative wording acceptable: `RNA + ATAC + Protein → 256-D latent` if it fits the card width.
- Recommendation: `Multi-omics → 256-D latent` is cleaner and keeps the 5-modality framing alive even on A1.

---

### Fix 3 — Fix the protein panel number

**Problem**: Current INPUT body bullet reads `› Protein · 210-D panel`. The 210-D number is from TotalSeq-A reference panels but **has NOT been confirmed for QurieSeq Phase 1**. Mimitou (our actual training data) uses a 37-antibody panel. C1 speaker notes explicitly flag "protein panel size pending — could be ~37 like Mimitou or larger" as an open question for Kinga.

**Fix** — pick the most defensible option:

Option A (recommended): `› Protein · 30–210 surface markers`
- Honest about the range, doesn't commit to one number
- Conveys ambition without overstating

Option B: `› Protein · CITE-seq panel`
- Removes the number entirely
- Cleaner, but loses signal of scale

Option C: `› Protein · 37 markers (Mimitou)`
- Specific and verifiable, but undersells the Phase 2 expansion

**Decision**: Use Option A unless Cowork has a strong reason otherwise.

---

### Fix 4 — Clarify Stage 3a status in the validation row

**Problem**: Stage 3a (amber ◐ "In-flight training") is currently positioned under the READOUT block. This is technically right (Stage 3a trains adapter + decomposed readout), but the caption "In-flight training" is ambiguous. As of today (May 15, 2026):
- Stage 3a code infrastructure: ✅ complete (Day 1 + Day 2 PRs landed, 87 tests green)
- Stage 3a real-data training: ⏸ pending Day 4-5 BSC training run

The amber dot suggests training is *running*; technically the infrastructure is ready but training hasn't started yet.

**Fix**:
- Change caption under the amber Stage 3a dot from `In-flight training` → `Infra ready · training May`
- Keep the amber color (◐ half-circle) — it's the right "between green and grey" status

This is a minor word change but reads more honestly. An investor reading "in-flight training" expects to see partial results; "infra ready · training May" sets the right expectation.

---

### Fix 5 — Tighten vertical layout

**Problem**: Cards extend to y=612, validation row starts at y=684, invariant pills at y=808, source citation floats alone at y=1052. There's significant empty space between elements and an awkward gap below the invariant pills.

**Fix** (pick one):

Option A — Tighten everything up
- Reduce gap between cards (y=612) and validation row (y=684) → start validation row at y=660
- Reduce gap between invariant pills (y=808) and source citation (y=1052) → push source citation up to y=940-960
- Bottom margin of slide stays ~100px

Option B — Add a horizontal divider/tagline strip in the empty space
- Below the invariant pills, add a thin horizontal divider line at y=900
- Below that at y=930, add a small italicized tagline (≤12pt) like: `Same trained model. Five PBMC lineages. Any timepoint. Any combination.`
- Source citation stays at y=1052

**Decision**: Use **Option A** (tighten) unless Cowork sees a better use of the space. Option B is acceptable if executed sparingly — no decoration for decoration's sake.

---

## What Should NOT Change

These elements are working well and should be preserved:

- Color palette and step-number rotation (01 cyan → 02 lavender → 03 cyan → 04 purple → 05 white)
- Typography choices (Inter for titles, Arial for body)
- Card structure (5 blocks with title + subtitle + 3 body bullets + footer italic note)
- "APPENDIX A1 · ARCHITECTURE DEPTH" header
- "A1 / 12" pagination indicator
- Source citation line at the bottom (just reposition per Fix 5)
- Background dark navy + corner radial glows
- The pill-shaped invariant guarantees row

---

## Deliverable Sequence

```bash
# Single commit with all 5 fixes applied
git add docs/deck/assets/diagrams/A1_system_architecture.svg \
        docs/deck/assets/diagrams/_build_a1.py
git commit -m "docs(deck): A1 v2 — 5 visual fixes per Ash review

Fix 1: Move FROZEN lock icon to upper-right of ENCODER card,
       resolve collision with title. Remove redundant FROZEN text label.
Fix 2: Strip 'Trimodal' from encoder body. Use 'Multi-omics → 256-D'
       to preserve 5-modality platform framing (per A2 v4 decision).
Fix 3: Protein panel '210-D' → '30–210 surface markers' (range, not
       unverified commitment). Speaker note in C1 flagged 210-D as
       pending Kinga confirmation.
Fix 4: Validation row caption 'In-flight training' → 'Infra ready ·
       training May'. More honest — Stage 3a code is shipped (87 tests
       green) but real-data BSC training run is pending.
Fix 5: Tighten vertical layout — reduce gaps between cards/validation/
       invariants/source citation. Bottom margin ~100px."
git push origin main
```

---

## Acceptance Criteria For v2

When Cowork ships A1 v2, Ash will verify:

1. ✅ Lock icon does not overlap "ENCODER" title text
2. ✅ Word "Trimodal" does not appear anywhere on the slide
3. ✅ Protein panel reads as a range (or is removed/replaced cleanly)
4. ✅ Stage 3a caption reads honestly about current status
5. ✅ Vertical spacing feels balanced — no awkward gaps

If all 5 land, A1 is locked and the visual style anchor is final. We then ship a single batch prompt for A2-E1 (11 diagrams) using A1 as the style reference.

---

## What's NOT in scope for this iteration

These are issues Ash flagged as discussable but not required to fix:

- The arrows between cards (subtle but readable — leaving as-is unless Cowork has a strong opinion)
- The corner radial glows (acceptable, don't simplify unless requested in a future pass)
- The slide-title hierarchy (working as-is)

If Cowork wants to address these as quality polish, it's optional. Don't over-iterate.

---

## Output File

Same path, new content:
- `docs/deck/assets/diagrams/A1_system_architecture.svg`
- `docs/deck/assets/diagrams/_build_a1.py` (updated build script with the changes)

Optional: PNG preview regenerated at `_preview.png` for fast visual verification.

---

## What Comes After A1 v2

If A1 v2 passes review, the batch prompt for A2-E1 follows immediately. Each of the 11 remaining specs has its own "Followup prompt for Cowork" section embedded — those become the basis for the batch.

Estimated batch generation time: 2-3 hours (parallel where possible, iterative where needed).

Then Phase 3 (.pptx assembly via pptx skill) and Phase 4 (Claude Design visual polish on hero diagrams) follow.
