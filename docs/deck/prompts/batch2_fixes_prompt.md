# Batch 2 Visual Fixes — B2 / D1 / D2 Text Collisions

**Owner**: Cowork (execution)
**Estimated time**: 30-45 min
**Input commit**: `66aa0de`
**Input files**:
- `docs/deck/assets/diagrams/B2_adapter_verdict.svg` + `_build_b2.py`
- `docs/deck/assets/diagrams/D1_quarterly_roadmap.svg` + `_build_d1.py`
- `docs/deck/assets/diagrams/D2_seed_allocation.svg` + `_build_d2.py`
- `docs/deck/assets/diagrams/_deck_common.py`

---

## Context

Batch 2 shipped 8 diagrams. Ash visual review approved 5 (B1, B3, C1, C2, E1) and flagged 3 with the **same class of bug**: two text elements positioned at conflicting coordinates, rendering on top of each other.

The pattern matters: this is the same class of issue as A3 v1's off-card Δ_synergy (visible-width miscalculation). The shared underlying problem is **position math without collision detection**.

This iteration fixes the 3 concrete instances AND adds a guard to `_deck_common.py` to prevent recurrence.

---

## Fix 1 — B2 ADAPTER_RECOMMENDED Row Text Collision

**Problem**: In the highlighted middle threshold row, three text elements compete for the same horizontal band:
- "WE ARE HERE" badge (top, cyan small caps)
- "0.50 — 0.80" threshold range label (white, large)
- "0.57" hero number (cyan, large)

At slide-fill scale, "0.57" renders **directly on top of "0.50"** — the cyan 0.57 visually overlaps the white 0.50, producing garbled `0,57|0.50` text at the left edge of the highlighted row.

**Root cause** (inspect `_build_b2.py`): the 0.57 hero label is positioned with the same or nearly-same x-coordinate as the threshold range label, but at a slightly different y. They occupy the same visual band and collide.

**Fix — pick the cleanest of these layout options**:

**Option A** (recommended): Move 0.57 to its own dedicated column, between the threshold range column and the verdict column:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ◆ WE ARE HERE                                                       │
│   0.50 — 0.80      [◆ 0.57 ◆]    ADAPTER_RECOMMENDED   Train light…  │
└──────────────────────────────────────────────────────────────────────┘
```

The 0.57 gets its own visual "pill" or callout box (cyan filled), positioned between the range and the verdict. No collision, hero stays prominent.

**Option B**: Stack vertically in the same cell — threshold range on top, "← we are here 0.57" below:

```
┌──────────────────────────────────────────────────────────────────────┐
│   0.50 — 0.80                  ADAPTER_RECOMMENDED   Train light…    │
│   ◆ WE ARE HERE: 0.57                                                │
└──────────────────────────────────────────────────────────────────────┘
```

**Option C**: Replace the range "0.50 — 0.80" with "0.57 (in 0.50–0.80 band)" — single text element, no collision:

```
┌──────────────────────────────────────────────────────────────────────┐
│   ◆ 0.57 (in 0.50–0.80 band) ◆  ADAPTER_RECOMMENDED   Train light…   │
└──────────────────────────────────────────────────────────────────────┘
```

**Recommendation**: **Option A** — keeps the 0.57 prominent as the slide's hero number while letting the threshold range stay visible. Pill/badge styling around 0.57 reinforces "this is our result."

If Cowork sees a layout reason Option A doesn't fit (e.g., row height becomes too cramped), Option C is the second choice — it eliminates the collision deterministically by merging into one element.

### Acceptance criteria
- ✓ "0.57" and "0.50" are not in the same x+y rendering band
- ✓ "WE ARE HERE" badge still visible
- ✓ "ADAPTER_RECOMMENDED" verdict still in the verdict column
- ✓ Row still highlighted with cyan accent (unchanged)
- ✓ Threshold range information still legible

---

## Fix 2 — D1 Milestone Label Collision at Q3'26 / Q4'26

**Problem**: At the top of the Gantt timeline, milestone labels for adjacent quarters overlap each other. Specifically:
- "QurieSeq P1 lands" (Q3'26)
- "BTK+JAK ZERO-SHOT DEMO" (Q4'26)
- "Phase 2 phospho on" (Q1'27)

These three labels sit at horizontally adjacent positions and their text strings collide. "BTK+JAK ZERO-SHOT DEMO" is the longest label; it visually overlaps "Phase 2 phospho on" on the right side.

**Root cause** (inspect `_build_d1.py`): all 7 milestone labels render at the same y-coordinate above the quarter dividers. With variable label widths and quarter columns at ~140px each, longer labels overflow into the next column's space.

**Fix — stagger label heights (alternating zigzag)**:

```
y_high (~145)    [QurieSeq P1 lands]                  [Pipeline 1 starts]              [Pipeline 2 / P1 valid.]
y_low  (~165)                       [BTK+JAK DEMO]                       [Stage 4 wraps]                       [Stage 5 wraps]
                       ◆ Q3'26          ◆ Q4'26          ◆ Q1'27   ◆ Q2'27    ◆ Q3'27   ◆ Q4'27    ◆ Q1'28   ◆ Q2'28
```

Alternate labels between two y-positions — odd-indexed milestones at `y_high`, even-indexed at `y_low` (or vice versa). Vertical offset ~20px, enough to clear most label heights.

This is the standard mitigation Cowork pre-flagged as Risk #1 in the Batch 2 shipping notes.

**Additional consideration**: Shorten "BTK+JAK ZERO-SHOT DEMO" to "BTK+JAK demo" or "Headline demo" to reduce label length. The longer string is partially redundant — the visual emphasis (glow ring + accent color + size) already conveys "this is the anchor". A short label keeps that.

**Recommendation**: Apply BOTH — stagger label heights AND shorten the longest label. Belt-and-suspenders against re-occurrence as new milestones get added.

### Acceptance criteria
- ✓ No adjacent milestone labels overlap visually
- ✓ Q4'26 BTK+JAK demo still has visual hierarchy (glow ring + larger marker + accent color)
- ✓ All 7 milestones still labeled
- ✓ Labels still readable at slide-fill scale (~12pt or larger)
- ✓ Stagger pattern visually deliberate, not random

---

## Fix 3 — D2 Summary Line Text Collision

**Problem**: Between the 3-card strategic re-grouping and the source footer, there's a region that should contain:
1. A math summary: `$5.5M + $2.5M + $2.0M = $10M`
2. A disclosure caption: `estimates pending CEO confirmation — see speaker notes for budget assumptions`

These two text elements are rendered at the **same y-coordinate**, producing garbled output:
```
· estim$5.5Mer$2.5MCE$2.0Mfirm$10M · see speaker notes for budget assumptions
```

Both elements have valid intent; they just need separate positions.

**Root cause** (inspect `_build_d2.py`): two `<text>` calls with the same `y` parameter but different `x` parameters that don't fully separate. Likely a copy-paste y-coord error.

**Fix — separate to two lines**:

```
                $5.5M + $2.5M + $2.0M = $10M

   Allocation estimates pending Kinga (CEO) final confirmation — see speaker notes
```

Math summary at one y-coordinate (e.g., y=688), disclosure caption at a y-coordinate 28-32px below it (e.g., y=716). Both small text, lower visual priority than the 3 cards above. Disclosure caption in muted grey.

**Alternative**: Merge into a single line if space permits:
```
Total $5.5M + $2.5M + $2.0M = $10M · estimates pending Kinga (CEO) confirmation
```

**Recommendation**: Two separate lines. The math summary is verification (investors will mentally check), the disclosure caption is honesty signal. Both serve different purposes; visual separation reflects that.

### Acceptance criteria
- ✓ Math summary `$5.5M + $2.5M + $2.0M = $10M` renders as one clean unbroken line
- ✓ Disclosure caption `estimates pending CEO confirmation` renders as one clean unbroken line
- ✓ Two text elements are on separate y-coordinates with visible vertical gap
- ✓ Source footer at the very bottom of slide is unchanged

---

## Fix 4 — Tech Debt: Position-Collision Guard in `_deck_common.py`

**Problem**: Three of three remaining Batch 2 bugs are the same class — two text elements positioned at conflicting coordinates without anyone noticing. Pure visual review missed B2's collision (Ash caught it on closer inspection); the textual acceptance checks can't catch it because both text strings are technically "present in the SVG."

**Fix**: Add a position-collision check to `_deck_common.py` that can be called as a sanity step before saving an SVG.

**Proposed API**:

```python
# In _deck_common.py

def check_no_text_collisions(svg_xml: str, *, min_gap: int = 8, group_threshold: int = 60) -> list:
    """Scan SVG <text> elements and detect probable rendering collisions.
    
    A collision is suspected when two text elements have:
    - x-coordinates within `group_threshold` pixels of each other (same column)
    - y-coordinates within `min_gap` pixels of each other (same band)
    
    Returns a list of (text1, text2, x_diff, y_diff) tuples describing each
    suspected collision, empty if clean.
    
    Notes:
    - Heuristic — does not account for font-size or text-anchor
    - Use as smoke test, not authoritative; visual review still required
    - Run in builder's verification step before file write
    """
    ...

def check_text_within_bounds(svg_xml: str, *, parent_bounds: list) -> list:
    """Verify every <text> element's estimated bounding box fits within 
    one of the parent_bounds rects supplied. Catches off-card text 
    (the A3 v2 Δ_synergy bug class).
    
    Returns a list of (text_content, x, y, parent_violated_or_None) for each
    text element outside any supplied bound, empty if clean.
    """
    ...
```

**Implementation guidance**:
- Parse SVG XML with `xml.etree.ElementTree` (standard library, already used elsewhere)
- Extract every `<text>` element with its x, y attributes
- For each pair, compute Euclidean distance; flag if within thresholds
- Treat `text-anchor="end"` and `text-anchor="middle"` by adjusting effective x (estimate width as `len(visible_chars) * font_size_pt * 0.6` — same heuristic as visible-char-width used in A3 v2)

**Don't refactor existing builders to use the new helpers in this commit.** Just add the functions to `_deck_common.py` and use them in the new builds for B2/D1/D2 v2 to verify the fixes don't reintroduce collisions. Future builders adopt going forward.

### Acceptance criteria
- ✓ `_deck_common.py` exports `check_no_text_collisions` and `check_text_within_bounds`
- ✓ Both functions have docstrings explaining the heuristic and its limits
- ✓ B2/D1/D2 v2 builders call at least one of them as a verification step before file write
- ✓ All three v2 builds run clean (no collisions reported)
- ✓ Existing builders (A1-A4, B1, B3, C1, C2, E1) are NOT modified in this commit — tech debt only adds, doesn't refactor

---

## Deliverable Sequence

Two acceptable patterns:

**Pattern A — Single fix commit covering all 4 changes**:
```
docs(deck): Batch 2 fixes — B2/D1/D2 collisions + collision-guard helpers

- B2: 0.57 hero number relocated to own pill (Option A) — no longer
  overlaps "0.50 — 0.80" threshold range label
- D1: Milestone labels staggered alternating y-positions to clear 
  the Q3'26/Q4'26/Q1'27 collision; "BTK+JAK ZERO-SHOT DEMO" shortened
  to "BTK+JAK demo" to reduce label width
- D2: Math summary line and disclosure caption split to separate 
  y-coordinates, garbled overlap resolved
- _deck_common.py: added check_no_text_collisions() and 
  check_text_within_bounds() helpers; used in B2/D1/D2 v2 builders 
  as pre-write verification step
```

**Pattern B — Four separate commits**:
```
fix(deck): B2 — relocate 0.57 hero, resolve threshold-range collision
fix(deck): D1 — stagger milestone labels, shorten anchor label
fix(deck): D2 — split summary line from disclosure caption
chore(deck): add collision-guard helpers to _deck_common.py
```

Cowork's call. Pattern A is simpler ship workflow; Pattern B has cleaner git blame for future archaeology.

---

## What Ash Will Check On Review

For each of the 3 fixed diagrams, zoom into the previously-problematic region (not just full-slide view):

**B2**: zoom into the ADAPTER_RECOMMENDED row, verify "0.57" and "0.50 — 0.80" are separately readable
**D1**: zoom into Q3'26 / Q4'26 / Q1'27 milestone label area, verify no overlap
**D2**: zoom into the area between cards and source footer, verify clean two-line layout

Plus a banned-term sweep and xmllint on all three SVGs (Cowork's script should already do this).

Plus a smoke test of `_deck_common.py`:
- `check_no_text_collisions` returns empty list for all 3 v2 builds
- `check_text_within_bounds` returns empty list for all 3 v2 builds

---

## Out Of Scope For This Iteration

- B1, B3, C1, C2, E1 — approved, do not modify
- A1-A4 — locked, do not modify
- Refactoring existing builders to use the new collision-guard helpers (just add helpers; future builders adopt)
- Anything in Phase 3 (.pptx assembly) — that happens after these fixes land

---

## What Comes After

If all 3 fixes land clean (and collision-guard helpers smoke-test green), **Phase 2 is fully complete**: 12 diagrams shipped, visual style coherent, no known issues.

Then **Phase 3 — .pptx assembly** unlocks. Cowork uses the pptx skill to assemble `aivc_appendix_v1.pptx` from:
- 12 content specs in `docs/deck/content/`
- 12 SVGs in `docs/deck/assets/diagrams/`
- Speaker notes from each content spec embedded as slide notes

Estimated Phase 3 time: 30-60 min. Output: `docs/deck/exports/aivc_appendix_v1.pptx`.

After Phase 3, optional Phase 4 (Claude Design visual polish on hero diagrams — A1, A2, A3, B2, C1, C2 are highest priority).

**Total path to investor-ready deck v1 from where we are**: 1-2 hours of focused work after this iteration lands.

---

## Risks To Flag

1. **Option A pill for B2's 0.57** — fitting it in the row width without crowding "WE ARE HERE" badge above and threshold range to the right may require adjusting row height or label sizing. If layout gets tight, fall back to Option C (merge into single text element).

2. **D1 milestone stagger** — alternating y-positions can look mechanical if not done carefully. The pattern should feel deliberate. Consider grouping by importance: anchor milestones (Q4'26 demo) stay at primary y; supporting milestones stagger.

3. **_deck_common.py collision guard heuristics** — these are heuristics, not proofs. Edge cases (text-anchor="end" with long strings, text inside `<g transform="...">` groups, multi-line tspan) may produce false negatives. Document the limits clearly. Visual review still required as the final check.

4. **Banned-term sweep needs to stay green** — the fix shouldn't accidentally reintroduce "Trimodal", "210-D panel", "IPO", or "Series A" terms. Ship script should re-run the full sweep, not just the changed slides.
