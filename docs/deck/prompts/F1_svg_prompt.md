# F1 SVG Generation — Cowork Task

**Owner**: Cowork (execution)
**Estimated time**: 1.5-2 hours
**Input**: `docs/deck/content/F1_competitive_positioning.md` (committed at `740e054`)
**Strategy**: New Section F slide visually distinct from A-E, matching Phase 2 style discipline

---

## Context

Content spec for F1 (competitive positioning) committed to repo. F1 is the new Section F slide — adds 1 content slide + 1 section divider to the existing 18-slide pptx, bringing the final deck to **20 slides** (1 cover + 6 dividers + 13 content).

After this SVG ships, `_build_appendix_pptx.py` re-runs to assemble the 20-slide v2 deck.

This is the **final SVG generation task** before Phase 4 polish. Section F locks the appendix's visual completeness.

---

## Hard Requirements

### Style coherence with Sections A-E

- Same dark background `#070A14` + corner radial glows
- Same Inter title typography (32-44pt), Arial body (14-18pt)
- Same `APPENDIX F1 · COMPETITIVE POSITIONING` cyan eyebrow header at y=55
- Same title + sub-headline pattern (y=93 title, y=136 sub)
- Same `F1 / 13` cyan pagination indicator at top-right (note: pagination is now `/ 13` since we have 13 content slides, OR keep `/ 12` if Cowork prefers to keep parity with v1 — Cowork's call, document the choice)
- Same source citation footer at y≈980
- 1920×1080 viewBox
- Paired SVG + PNG per locked convention

### Section F accent color

**Amber `#FBBF24`** — distinct from A (cyan), B (green), C (cyan), D (lavender), E (white/pale).

This is the same amber we used for Phase 2 in-flight status icons; reusing it for Section F gives the appendix a 6-color visual rotation across sections. Document the choice in Cowork's notes.

### Paired SVG + PNG (locked pattern)

Same convention as every Phase 2 + Batch 2 SVG. Ship both artifacts in the same commit:
- `docs/deck/assets/diagrams/F1_integrated_platform.svg`
- `docs/deck/assets/diagrams/F1_integrated_platform_preview.png`
- `docs/deck/assets/diagrams/_build_f1.py`

Use `_deck_common.py` helpers (`check_no_text_collisions`, `check_text_within_bounds`, `collision_guard`) as pre-write verification — same pattern Cowork added for B2/D1/D2.

### Banned terms sweep (carry forward)

- No "Trimodal" anywhere
- No "210-D panel" anywhere
- No "Series A" / "IPO" anywhere
- No "category-of-one" or similar marketing phrases on-slide (preserved in speaker notes only per content spec)
- Visible character width math, not HTML tspan markup length (A3 v2 lesson)

---

## Layout Specification

Three vertical zones, top to bottom:

### Top zone — Flywheel diagram (visual hero, ~50% of slide vertical space)

The 4-pillar circular flywheel. This is the slide's conceptual centerpiece.

**Structure**:

```
                                      INTEGRATED PLATFORM
                                      (center label, amber section accent,
                                       letter-spaced small caps, 14pt)
                                      
                                      
                  ┌──────────────────────────┐
                  │  CO-DESIGNED             │
                  │  ARCHITECTURE            │
                  │                          │
                  │  4-arm decomposed readout │
                  │  Neural ODE temporal      │
                  │  Compositional gen.       │
                  └──────────────┬───────────┘
                                 │
                              ⤵ compounds
                                 ↓
   ┌──────────────────────┐                          ┌──────────────────────┐
   │  WET-LAB             │                          │  TEMPORAL            │
   │  GENERATION          │  ←─── compounds ─────→   │  MULTI-OMICS         │
   │                      │                          │                      │
   │  QurieSeq Phase 1+2  │                          │  RNA + ATAC +        │
   │  primary PBMCs       │                          │  Protein  (+phospho  │
   │  5 → 20 donors        │                          │  + VDJ Phase 2)      │
   │  4-arm perturbations │                          │  0/5/30/60/180 min   │
   └──────────────────────┘                          └──────────────────────┘
                  ↑                                          │
              ⤵ compounds                              ⤴ compounds
                  │                                          ↓
                  ┌──────────────────────────────────────────┐
                  │  PROTOCOL-FAMILY                          │
                  │  EXPANSION                                │
                  │                                           │
                  │  Same wet-lab pipeline extends to         │
                  │  Phase 2 phospho + VDJ without            │
                  │  re-architecting                          │
                  └──────────────────────────────────────────┘

         Caption (centered below flywheel, italic muted):
         "Each loop deepens the next. Every QurieSeq phase trains
          the architecture; every architecture extension informs
          the next wet lab. Integration is the moat."
```

**Visual treatment**:

- Each of the 4 pillars in its own rounded card (rx=14, dark fill `#0F1428`, 1.5px amber stroke at 0.65 opacity)
- 4-corner orbit layout (top, right, bottom, left positions) OR clean rectangular 2×2 grid with curved connecting arrows between — Cowork's call on which renders cleaner
- Curved "compounds" arrows between adjacent pillars (clockwise flow):
  - Architecture → Multi-omics
  - Multi-omics → Protocol-family
  - Protocol-family → Wet-lab
  - Wet-lab → Architecture
- Center label "INTEGRATED PLATFORM" in amber, letter-spaced, small caps
- Each pillar title in white (Inter 22pt bold), body in pale `#EAF6FF` (Arial 13pt)
- Caption below in italic muted grey (Arial 13pt)

**Critical**: the flywheel must visually feel like a **loop**, not 4 disconnected boxes. The curved arrows showing the compounding loop are what makes the visual work. If 4 corners + curves are too complex, fall back to **vertical stack with explicit "→ compounds →" labels** between pillars. The loop concept matters more than the literal circular shape.

### Middle zone — Competitor archetype grouping (~30% of slide vertical space)

Three buckets side-by-side, then one full-width row below.

**Structure**:

```
WHO OPTIMIZES WHAT?
(small caps eyebrow, muted, letter-spaced)

┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│  DATA SCALE             │ │  FOUNDATION MODELS     │ │  DOWNSTREAM            │
│  (bucket title, white,  │ │  (bucket title)        │ │  THERAPEUTICS          │
│   Inter 16pt bold)      │ │                        │ │  (bucket title)        │
│                         │ │                        │ │                        │
│  › TAHOE — 100M cells,  │ │  › CytoReason —        │ │  › Valo Health —       │
│    RNA-only cell lines  │ │    partner-derived     │ │    clinical            │
│  › Immunai — modality-  │ │    multi-omics, immune │ │    development         │
│    rich atlas, partner  │ │  › Turbine AI —        │ │  › Noetik — spatial    │
│    data                 │ │    virtual lab, pharma │ │    multi-omics         │
│                         │ │  › DeepLife — causal   │ │    oncology            │
│                         │ │    modeling, drug      │ │                        │
│                         │ │    repositioning       │ │                        │
│                         │ │                        │ │                        │
│  ─────────────────────  │ │  ─────────────────────  │ │  ─────────────────────  │
│  Optimize: data breadth │ │  Optimize: model arch.  │ │  Optimize: clinical    │
│  Decouple: wet-lab +    │ │  Decouple: proprietary  │ │            pipeline    │
│  protocol               │ │  data                   │ │  Decouple: foundation  │
│                         │ │                        │ │            modeling    │
└────────────────────────┘ └────────────────────────┘ └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  INTEGRATED CAUSAL PERTURBATION PLATFORM                                            │
│  (full-width row, amber accent, strong stroke)                                      │
│                                                                                     │
│  › QURIEGEN — proprietary wet-lab generation + co-designed architecture +           │
│               temporal multi-omics + compositional causal modeling +                │
│               protocol-family expansion, all coupled                                │
│                                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│  Optimize: the closed-loop system itself                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Visual treatment**:

- Top three buckets: equal-width, neutral cards (`#0F1428` fill, thin grey stroke `#1A2235` at 0.45 opacity)
- Bottom Quriegen row: **full-width, amber accent** (filled at low opacity OR strong amber stroke at full opacity — Cowork's call on which reads better visually)
- "Optimize: X" and "Decouple: Y" lines in each top bucket use small italic text (Arial 12pt)
- Quriegen row uses "Optimize: the closed-loop system itself" — no "Decouple" line (we don't decouple any layer)
- Competitor names in white bold, brief descriptors in pale `#EAF6FF`
- No checkmarks, no winner/loser language
- Visual hierarchy: top buckets recede; bottom Quriegen row dominates

### Bottom zone — Closing line (~10% of slide vertical space, above source footer)

```
"No public dataset has the combination drug combination prediction requires.
The wet lab, the architecture, and the protocol family are co-designed."
```

- Italic, centered (text-anchor="middle")
- Muted grey `#A8B4C2`, Arial 14pt
- Functions as the takeaway investors carry out

### Source citation footer (standard pattern, y≈980)

```
Source: docs/deck/research/competitive_landscape_2026_05.md · Architecture spec v1.1 · 
QurieSeq Phase 1+2 spec (Thiago, May 2026) · Stage 3 Part 1 dataset survey
```

Small text (Arial 11pt), muted, left-aligned. Standard pattern. Right-aligned `F1 / 13` pagination in cyan at the same y-band.

---

## Acceptance Criteria

Before staging, verify each of these as textual checks:

### Required content (all 4 pillars + 3 buckets + competitors named)

- ✓ "WET-LAB GENERATION" present
- ✓ "CO-DESIGNED ARCHITECTURE" present
- ✓ "TEMPORAL MULTI-OMICS" present
- ✓ "PROTOCOL-FAMILY EXPANSION" present
- ✓ "INTEGRATED PLATFORM" present (center label)
- ✓ "DATA SCALE", "FOUNDATION MODELS", "DOWNSTREAM THERAPEUTICS" all present (bucket titles)
- ✓ "INTEGRATED CAUSAL PERTURBATION PLATFORM" present (Quriegen row)
- ✓ All 7 competitor names present: TAHOE, Immunai, CytoReason, Turbine, DeepLife, Valo, Noetik
- ✓ "QURIEGEN" or "Quriegen" present (in the bottom row)
- ✓ "100M cells" present (TAHOE scale)
- ✓ "QurieSeq Phase 1+2" or similar present
- ✓ "0/5/30/60/180 min" or "0/5/30/60/180" present (timepoints in multi-omics pillar)
- ✓ "5 → 20 donors" or similar present (Phase 2 scaling)
- ✓ "compounds" appears multiple times (between pillars + in caption)
- ✓ Closing line "No public dataset has the combination" present
- ✓ "co-designed" appears at least 2 times

### Banned terms (carry forward)

- ✗ "Trimodal" absent
- ✗ "210-D panel" absent  
- ✗ "Series A" absent
- ✗ "IPO" absent
- ✗ "category-of-one" absent (preserved in speaker notes only per content spec)

### Structural checks

- ✓ Pagination shows F1 (Cowork's call: `F1 / 13` if updating to 13 content slides, or `F1 / 12` if keeping parity — document choice)
- ✓ Section eyebrow reads "APPENDIX F1 · COMPETITIVE POSITIONING"
- ✓ xmllint validates as well-formed XML
- ✓ `check_no_text_collisions` returns 0 blocking collisions
- ✓ `check_text_within_bounds` returns 0 violations

### Visual coherence (smoke test)

- ✓ Section accent is amber `#FBBF24` (distinct from A-E)
- ✓ Card style matches A-E (rx=14, dark fill, 1.5px stroke at 0.65 opacity)
- ✓ Typography matches locked stack (Inter titles, Arial body)
- ✓ Background `#070A14` matches A-E
- ✓ Pagination + eyebrow + source footer positions match A-E

---

## Build Script Pattern

Same template as B1-E1 builders. Recommended structure:

```python
#!/usr/bin/env python3
"""Build F1 SVG + PNG preview — competitive positioning."""

import cairosvg
from _deck_common import (
    # ... shared constants (colors, fonts, dimensions)
    check_no_text_collisions,
    check_text_within_bounds,
    collision_guard,
)

# Section F accent
ACCENT_AMBER = "#FBBF24"

def build_flywheel(x_center, y_center, radius):
    """Render the 4-pillar flywheel with curved compounds arrows."""
    # ...
    pass

def build_archetype_buckets(x, y, width):
    """3 top buckets + 1 full-width Quriegen row."""
    # ...
    pass

def build_closing_line(x, y, width):
    """Italic centered takeaway line."""
    # ...
    pass

def build_svg():
    """Generate F1 SVG."""
    parts = []
    parts.append(build_header("F1", "COMPETITIVE POSITIONING",
                              "The closed-loop platform — proprietary data, co-designed architecture, compounding over time.",
                              "No public dataset has the combination drug combination prediction requires. The wet lab, the architecture, and the protocol family are co-designed."))
    parts.append(build_flywheel(...))
    parts.append(build_archetype_buckets(...))
    parts.append(build_closing_line(...))
    parts.append(build_source_footer(...))
    parts.append(build_pagination("F1", total=13))  # or 12, document choice
    return wrap_svg(parts)

if __name__ == "__main__":
    svg = build_svg()
    # Run collision guard before write
    collisions = check_no_text_collisions(svg)
    blocking = [c for c in collisions if not is_known_false_positive(c)]
    if blocking:
        print(f"BLOCKING COLLISIONS: {blocking}")
        raise SystemExit(1)
    
    with open("F1_integrated_platform.svg", "w") as f:
        f.write(svg)
    
    cairosvg.svg2png(url="F1_integrated_platform.svg",
                     write_to="F1_integrated_platform_preview.png",
                     output_width=1920, output_height=1080)
    print("Built F1 SVG + PNG preview")
```

---

## Deliverable Sequence

Single commit covering all 3 files:

```bash
git add docs/deck/assets/diagrams/F1_integrated_platform.svg \
        docs/deck/assets/diagrams/F1_integrated_platform_preview.png \
        docs/deck/assets/diagrams/_build_f1.py
git commit -m "docs(deck): F1 SVG - integrated platform flywheel

4-pillar flywheel (wet-lab generation / co-designed architecture /
temporal multi-omics / protocol-family expansion) showing closed-loop
compounding system. Center label INTEGRATED PLATFORM in amber.

3-bucket competitor archetype grouping below flywheel (DATA SCALE /
FOUNDATION MODELS / DOWNSTREAM THERAPEUTICS) with 7 competitors
distributed by optimization layer. Quriegen full-width row alone in
INTEGRATED CAUSAL PERTURBATION PLATFORM category.

Section F amber accent (#FBBF24) distinct from A-E. Style coherence
with Phase 2 SVGs preserved. Collision-guard helpers used pre-write."
git push origin main
```

After this lands, separate prompt for re-running `_build_appendix_pptx.py` to assemble the 20-slide v2 deck with F1 + new Section F divider added.

---

## What Ash Will Check On Review

Same protocol as previous SVG reviews — visual verification at slide-fill scale + zoomed inspection of any dense regions:

1. **Flywheel visual hierarchy**: Does it read as a loop (compounding cycle) or as 4 disconnected boxes? The loop concept must be visually unmistakable.
2. **Compounds arrows**: Are the curved/labeled arrows visible enough to convey flow?
3. **Archetype grouping clarity**: Do the 3 top buckets visually recede so the Quriegen row dominates? Or does everything compete?
4. **Closing line readability**: Italic centered takeaway should land as conclusion, not afterthought.
5. **Competitor names**: All 7 named visibly (not buried in tiny text)
6. **Section accent consistency**: Amber used throughout, no leakage of cyan/green/lavender from other sections
7. **Pagination + eyebrow**: F1 indicator correct, COMPETITIVE POSITIONING section title visible

If any specific element needs iteration, single fix prompt — same workflow as A3 v2 and Batch 2 fixes.

---

## What's Out Of Scope For This Task

- Modifying any existing SVG (A1-E1 locked)
- Modifying content specs (F1 spec locked at commit 740e054)
- Updating the pptx (separate prompt after F1 SVG lands)
- Phase 4 visual polish

---

## Risks To Flag

1. **Flywheel layout is hard to render cleanly**. 4 pillars + curved arrows + center label + caption is dense. If a clean circular flywheel doesn't fit the visual budget, fall back to **vertical stack with explicit arrows between** ("→ compounds →") — the loop concept matters more than literal circular geometry. Pre-flag your layout choice in Cowork's notes if you deviate.

2. **Bucket density**: 3 top buckets with 2-3 competitors each + 1 full-width Quriegen row at the bottom = a lot of text in the middle zone. If the buckets feel cramped, consider shortening competitor descriptors (e.g., "TAHOE — 100M cells, RNA-only" instead of full company descriptions).

3. **Amber + dark navy contrast**: amber `#FBBF24` against `#070A14` is high-contrast and reads bright. Use 0.65 stroke opacity (matching A-E pattern) to keep the strong accent visually balanced with the rest of the deck.

4. **Pagination decision**: With F1 added, total content slides becomes 13. Either update all pagination to `/ 13` (consistency-correct, requires regenerating ALL SVGs) OR keep `/ 12` and accept F1 shows `F1 / 13` as outlier (low effort, slight inconsistency). **My recommendation: keep current SVGs at `/ 12` for now, F1 shows `F1 / 13`, and accept the inconsistency until Phase 4 polish unifies pagination.** Cowork's call to confirm.

5. **Competitor descriptors must be neutral**: Don't use language that competitors would object to. "Partner-derived multi-omics" is factual for CytoReason; "uses other people's data" would be loaded. Stay clinical.

6. **Collision-guard heuristic may flag false positives**: Same caveat as B2/D1/D2 — filter known footer-vs-pagination false positives explicitly. Document the filter.

---

## After This Lands

If F1 SVG ships clean:
1. **Re-run `_build_appendix_pptx.py`** to assemble 20-slide v2 deck (1 cover + 6 dividers + 13 content)
2. Add Section F divider slide (Section F · Competitive Positioning, amber accent, slides F1)
3. Insert F1 content slide
4. Output: `aivc_appendix_v2.pptx`

This is the **final visual deliverable** before Phase 4 polish. After v2 .pptx ships, the deck is feature-complete.

---

## Tool Selection Confirmation

**Cowork** (Python svgwrite/matplotlib + cairosvg PNG render) — same as all Phase 2 + Batch 2 work.

Not Claude Design (that's Phase 4 polish only). Not Claude Code (Cowork handles end-to-end with ship script pattern).
