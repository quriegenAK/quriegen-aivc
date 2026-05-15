# A3 Visual Fix — Remove Equation Annotations

**Owner**: Cowork (execution)
**Estimated time**: 10-15 min
**Input file**: `docs/deck/assets/diagrams/A3_decomposed_readout.svg`
**Build script**: `docs/deck/assets/diagrams/_build_a3.py`
**Reference (current)**: commit `199f29e`

---

## Context

A3 ships clean on every dimension except one: the right-side equation annotations (e.g., "← always active (baseline)", "← stim present", "← active if inhibitor present", "← combination only") **overlap with the equation terms themselves** at slide-fill rendering. The annotations sit inside the equation column instead of to its right.

After Ash visual review, the decision is to **remove the annotations entirely** rather than reposition them. Reasoning:

1. **The right-side compositional generalization table already conveys what each head does** — annotations were duplicative.
2. **Color-coding does the same job** — h_base (white), Δ_stim (green), Δ_inh (purple), Δ_synergy (cyan) makes the head identity visually obvious.
3. **Removing the annotations elevates the equation to pure mathematical statement** — reads as architecture proof, not explanation.
4. **A3 then becomes**: equation as proof (left) → use-case table as context (right). Two-column reading, no clutter.

---

## The Fix

### Remove these annotations from A3

All four right-side annotation strings:

```
← always active (baseline)
← stim present
← active if inhibitor present
← combination only (the zero-shot win)
```

These appear as `<text>` elements positioned to the right of each equation line in the current SVG. Find them in `_build_a3.py` (likely as a list of `(y_coord, text)` pairs feeding a draw loop) and remove the draw block entirely.

### What stays (everything else)

Confirmed working on A3 v1, do NOT change:
- The 4-arm equation with color-coded heads
- Indicator function notation `1[s]`, `1[i]`, `1[s ∧ i]`
- Color rotation: h_base white, Δ_stim green, Δ_inh purple, Δ_synergy cyan
- Load-bearing constraint box (theorem-style)
- Right-side "Compositional Generalization" 3-row table
- BTK+JAK row visual emphasis (cyan fill, 2px stroke)
- All badges ("0.68 accuracy", "Headline demo", "Compositional")
- Header eyebrow, title, sub-headline, source footer, pagination
- All other layout positions and styling

### Optional layout tightening

After removing the annotations, the left column will have ~400px of horizontal whitespace where they used to be. Two options:

**Option A** — Leave it as breathing room (recommended):
- Equation now centers in its left-column space with elegant negative space to its right
- Reads cleaner

**Option B** — Center the equation horizontally within its column:
- Shift equation 50-80px right so it visually balances within its space
- Slight risk: equation feels disconnected from constraint box below

**Decision**: Use Option A unless Cowork sees a strong reason otherwise. The breathing room is a feature.

---

## Acceptance Criteria For v2

When Cowork ships A3 v2:

1. ✅ No `← always active`, `← stim present`, `← active if inhibitor present`, or `← combination only` text anywhere on A3
2. ✅ Equation still renders with color-coded heads in left column
3. ✅ Load-bearing constraint box unchanged
4. ✅ Right-side 3-row table unchanged
5. ✅ Source footer unchanged
6. ✅ A3 still matches A1/A2/A4 style (dark bg, same header pattern, same pagination)

---

## Deliverable

Single commit, same path:

```bash
# Single commit with annotation removal
git add docs/deck/assets/diagrams/A3_decomposed_readout.svg \
        docs/deck/assets/diagrams/A3_decomposed_readout_preview.png \
        docs/deck/assets/diagrams/_build_a3.py
git commit -m "docs(deck): A3 v2 — remove equation annotations

Annotations overlapped with Δ_stim and Δ_inh equation terms at
slide-fill rendering. Annotations were duplicative anyway — the
right-side compositional generalization table conveys the same
information without competing with the equation visually.

A3 now reads as: equation as architecture proof (left) →
use-case table as context (right). Two columns, no clutter."
git push origin main
```

---

## What's Out Of Scope For This Iteration

- Any other A3 visual changes (color, typography, layout positioning of other elements)
- A2 or A4 modifications (both approved as-is)
- Batch 2 (B1-E1) — starts after A3 v2 lands

---

## After A3 v2 Approves

Single message from Ash: "approved, ship Batch 2 prompt." Then Batch 2 (B1, B2, B3, C1, C2, D1, D2, E1 — 8 diagrams) follows immediately.

---

## Risks To Flag

1. **The build script's draw-annotation block** might be tightly coupled with other text rendering (shared font/size definitions). If removing it cleanly is non-trivial, comment out the draw call rather than deleting code — keeps the option open if we want to re-add annotations later in a different position.

2. **The PNG regeneration** must be part of this commit per our paired-artifact rule. Same pattern as A1 v2 and Batch 1.
