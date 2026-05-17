# A5 SVG Generation — Cowork Task

**Owner**: Cowork (execution)
**Estimated time**: 1.5-2.5 hours
**Input**: `docs/deck/content/A5_causal_architecture.md` (committed at `ff92117`)
**Strategy**: New Section A slide extending Architecture Depth from 4 to 5 slides

---

## Context

A5 content spec committed. New Section A slide between A4 and B section divider, bringing Section A to 5 slides total.

A5 is the **most architecturally loaded slide in the appendix**. It surfaces Stage 3c causal architecture (Neumann propagation + sparse learned GRN with STRING prior + direct-effect log-FC head) as spec-locked architectural commitment with explicit validation timing (Q1-Q2 2027 post Phase 1 wet-lab data).

**Status pill is non-negotiable** and load-bearing. Without it, A5 reads as operational claim. With it, A5 reads as honest forward-looking architectural commitment. This is the **B3/C2/D1 honesty discipline** applied to causal architecture.

After A5 SVG ships, separate prompt for pptx v3 reassembly (21 slides, pagination → /14).

---

## Hard Requirements

### Style coherence with Section A locked palette

- Same dark background `#070A14` + corner radial glows
- Same Inter title typography (32-44pt), Arial body (14-18pt)
- Section A primary accents: **cyan `#26DDF9` + lavender `#8B5CF6` + green `#4ADE80`** — matching A1-A4 palette exactly. No new color introductions.
- Same `APPENDIX A5 · ARCHITECTURE DEPTH` cyan eyebrow header at y=55
- Same title + sub-headline pattern (y=93 title, y=136 sub)
- Same pagination treatment: `A5 / 14` at top-right in cyan (note: pagination is `/ 14` since A5 brings content slide total to 14, F1 remains at `/ 13` until Phase 4 unification)
- Same source citation footer at y≈980
- 1920×1080 viewBox
- Paired SVG + PNG per locked convention

### Paired SVG + PNG (hard requirement)

Ship both artifacts in the same commit:
- `docs/deck/assets/diagrams/A5_causal_architecture.svg`
- `docs/deck/assets/diagrams/A5_causal_architecture_preview.png`
- `docs/deck/assets/diagrams/_build_a5.py`

Use `_deck_common.py` helpers (`check_no_text_collisions`, `check_text_within_bounds`, `collision_guard`) with `min_gap=2` (F1 v2 lesson — `min_gap=4` is too permissive). Run as pre-write verification.

### Banned terms sweep (carry forward)

- No "Trimodal" anywhere
- No "210-D panel" anywhere
- No "Series A" / "IPO" anywhere
- No "category-of-one" or similar marketing phrases
- Visible character width math, not HTML tspan markup length (A3 v2 lesson)
- **No claims that Stage 3c is "operational" or "in production"** — A5 explicitly frames as spec-locked + validation-scheduled

---

## Critical Visual Element — The Status Pill

**The status pill is the single most important element on this slide.** It carries 100% of the diligence credibility load.

### Status pill specification

**Position**: Top-right of slide, prominent, **not** overlapping with the `A5 / 14` pagination indicator. Recommend placing the pill at top-right with pagination above or beside it (Cowork's call on exact layout — both must be visible at slide-fill scale).

**Content**:
```
◆ STAGE 3c · SPEC-LOCKED
Validation Q1-Q2 2027
post Phase 1 wet-lab data
```

**Visual treatment**:
- Diamond `◆` bullet in amber `#FBBF24` (signals forward-looking status)
- "STAGE 3c · SPEC-LOCKED" in cyan `#26DDF9`, bold, letter-spaced (signals architectural commitment)
- Body lines in pale `#EAF6FF` Arial 13pt (validation timing)
- Border: 1.5px cyan stroke at 0.85 opacity (strong but not loud)
- Fill: cyan at 0.12 opacity (subtle background)
- Rounded corners (rx=10)
- ~280px wide, ~80px tall

**Why this matters**: investors flipping the appendix see the pill before reading the math. The pill sets expectation as "this is architectural commitment, not operational claim." If a diligence reviewer asks "where are validation results?" — the pill already answered.

### Acceptance criterion for the pill

- Pill is visible at slide-fill scale (≥18pt effective text size)
- "STAGE 3c · SPEC-LOCKED" text is the most visually prominent text in the top-right zone
- Validation timing is readable (not crammed)
- Pill does NOT obscure or overlap with pagination indicator

---

## Layout Specification

Three vertical zones below the header band, top to bottom:

### Top zone — Neumann propagation equation (visual hero, ~40% of vertical space)

The mathematical centerpiece. Same prominence treatment as A3's decomposed readout equation.

```
┌──────────────────────────────────────────────────────────────────┐
│  NEUMANN PROPAGATION                                              │
│  perturbation flow through learned graph structure                │
│                                                                   │
│         ŷ = (I − W)⁻¹ · dₚ                                        │
│         ─────────────────                                          │
│                                                                   │
│         W ∈ ℝᴺˣᴺ       sparse learned GRN          (cyan)         │
│         dₚ ∈ ℝᴺ        direct perturbation effect  (lavender)     │
│         (I − W)⁻¹      closed-form propagation     (green)        │
│                                                                   │
│  Architectural requirement: ρ(W) < 1 enforced by sparsity L1     │
└──────────────────────────────────────────────────────────────────┘
```

**Equation typography**:
- `ŷ = (I − W)⁻¹ · dₚ` rendered in large math typography (~48pt equivalent)
- Use same Greek/math letter rendering pattern as A3 (Δ, indicator function) — the SVG `<tspan>` with proper math glyph fallback
- `W` rendered in cyan
- `dₚ` rendered in lavender (with subscript `p` smaller)
- `(I − W)⁻¹` rendered with the "⁻¹" superscript in green
- The minus sign in `I − W` should be a proper Unicode minus (U+2212), not hyphen
- Equation centered in the card

**Component definitions** (smaller text below equation):
- `W ∈ ℝᴺˣᴺ` with annotation "sparse learned GRN" — `W` in cyan, set notation in white, annotation in muted
- `dₚ ∈ ℝᴺ` with annotation "direct perturbation effect" — `dₚ` in lavender
- `(I − W)⁻¹` with annotation "closed-form propagation" — operator in green

**Architectural requirement** (footer of card, muted, small):
- "Architectural requirement: ρ(W) < 1 enforced by sparsity L1"
- Italic Arial 12pt in muted grey `#A8B4C2`

Card style: dark fill `#0F1428`, 1.5px stroke (cyan at 0.55 opacity for primary architectural element), rx=14.

### Middle zone — Sparse learned GRN visualization (~35% of vertical space)

Two side-by-side panels showing the structural prior → learned GRN transition.

**Left panel — STRUCTURAL PRIOR (STRING)**:
```
┌──────────────────────────────────┐
│  STRUCTURAL PRIOR (STRING DB)     │
│  edge-existence prior              │
│                                    │
│       ●────●────●                  │
│       │         │                  │
│       ●────●                       │
│             │                      │
│              ●────●                │
│                                    │
│  STRING-supported edges            │
│  lower L1 sparsity pressure        │
└──────────────────────────────────┘
```

**Right panel — LEARNED SPARSE GRN**:
```
┌──────────────────────────────────┐
│  LEARNED SPARSE GRN                │
│  edge weights after training       │
│                                    │
│       ●━━━━●━━━━●                  │
│       ┃                            │
│       ●━━━━●                       │
│             ┃                      │
│              ●··  ····●            │
│                                    │
│  thick = high-weight learned       │
│  dashed = below sparsity threshold │
└──────────────────────────────────┘
```

**Visual treatment for both panels**:
- 6-8 nodes per panel (illustrative, representative of immune-relevant gene clusters — labels optional, kept generic since GRN is learned)
- Same node positions in both panels (so the visual reads as "before → after")
- Left panel: edges as **thin grey strokes** at 0.5 opacity (representing STRING priors)
- Right panel: edges as **thick cyan strokes** for high-weight learned + **dashed grey strokes** for sub-threshold pruned
- Optional small connecting graphic between panels: arrow with caption "L1 sparsity →" in italic muted
- Bottom caption (centered, below both panels): "prior shapes initialization, learning prunes"

**Card style for both panels**: dark fill `#0F1428`, 1.5px stroke (grey at 0.45 opacity — supporting elements, not primary), rx=14.

**Width**: each panel ~880px wide, with ~60px gap between them.

### Bottom zone — Direct-effect log-FC head (~25% of vertical space)

```
┌──────────────────────────────────────────────────────────────────┐
│  DIRECT-EFFECT LOG-FC HEAD                                        │
│                                                                   │
│  [latent z + perturbation context]  →  [log-FC decoder]  →  dₚ    │
│                                                                   │
│  Stage 3a/3b predicted:  abundance after perturbation              │
│  Stage 3c separates:     dₚ (direct) + (I−W)⁻¹ dₚ (propagated)    │
│                                                                   │
│  Why this matters: causal queries vs predictive queries           │
│  "what does X cause?"        vs    "what happens after X?"        │
└──────────────────────────────────────────────────────────────────┘
```

**Visual treatment**:
- Block diagram (3 connected rounded boxes): `latent z + perturbation context` → `log-FC decoder` → `dₚ`
- Arrows between boxes in lavender (perturbation accent)
- Two-row comparison below the diagram:
  - "Stage 3a/3b predicted: abundance after perturbation" — muted
  - "Stage 3c separates: dₚ (direct) + (I−W)⁻¹ dₚ (propagated)" — cyan accent on the math, white on the prose
- "Why this matters" footer line: italic, two-clause format with "vs" emphasis

Card style: dark fill `#0F1428`, 1.5px stroke (lavender at 0.55 opacity — perturbation/causal element), rx=14.

### Source citation footer (standard pattern, y≈980)

```
Source: Architecture spec v1.1 (causal layer pending §X extension) · 
QurieSeq Phase 1+2 spec (Thiago, May 2026) · 
STRING DB v12.0 (Szklarczyk et al., 2023, NAR) · 
Neumann series propagation (standard linear-algebra reference)
```

Small text (Arial 11pt), muted, left-aligned. Standard pattern.

**Note**: The "(causal layer pending §X extension)" annotation is intentional — A5 anchors a forthcoming spec extension. The slide drives the spec, which is the right direction for forward-looking architectural commitment.

---

## Acceptance Criteria

Before staging, verify each of these as textual + structural checks:

### Required content

- ✓ "NEUMANN PROPAGATION" present (zone 1 title)
- ✓ "STRUCTURAL PRIOR" or "STRING" present (zone 2 left panel)
- ✓ "LEARNED SPARSE GRN" present (zone 2 right panel)
- ✓ "DIRECT-EFFECT LOG-FC HEAD" present (zone 3 title)
- ✓ "STAGE 3c" present (status pill)
- ✓ "SPEC-LOCKED" present (status pill)
- ✓ "Validation Q1-Q2 2027" present (status pill)
- ✓ "post Phase 1 wet-lab data" present (status pill)
- ✓ Equation `(I − W)⁻¹` present (verify Unicode minus U+2212, not hyphen)
- ✓ `dₚ` present (or `d_p` if Unicode subscript fallback)
- ✓ `ρ(W) < 1` present (architectural requirement)
- ✓ "L1 sparsity" present (middle zone caption)
- ✓ "prior shapes initialization, learning prunes" present
- ✓ "causal queries vs predictive queries" present (zone 3)
- ✓ "what does X cause?" and "what happens after X?" both present
- ✓ "Stage 3a/3b predicted" and "Stage 3c separates" both present

### Banned terms

- ✗ "Trimodal" absent
- ✗ "210-D panel" absent
- ✗ "Series A" absent
- ✗ "IPO" absent
- ✗ "category-of-one" absent
- ✗ "operational" not used in context that implies Stage 3c is operational today
- ✗ "in production" absent
- ✗ "validated" not used in context that implies Stage 3c has validation results today

### Structural checks

- ✓ Pagination shows `A5 / 14`
- ✓ Section eyebrow reads "APPENDIX A5 · ARCHITECTURE DEPTH"
- ✓ xmllint validates as well-formed XML
- ✓ `check_no_text_collisions(min_gap=2)` returns 0 blocking collisions
- ✓ `check_text_within_bounds` returns 0 violations
- ✓ Helper smoke at `min_gap=0`: 0 blocking collisions (proves the layout is genuinely clean, not just at the threshold)

### Visual coherence (smoke test)

- ✓ Section accent palette matches A1-A4 (cyan + lavender + green, no new colors)
- ✓ Status pill visible at slide-fill scale
- ✓ Status pill positioned top-right without overlapping pagination
- ✓ Neumann equation is the largest visual element on the slide (~48pt equivalent)
- ✓ Card styles match A1-A4 (rx=14, dark fill, 1.5px stroke at 0.55 opacity for primary, 0.45 for supporting)
- ✓ Typography matches locked stack (Inter titles, Arial body)
- ✓ Background `#070A14` matches A1-A4

---

## Build Script Pattern

Same template as A1-F1 builders. Recommended structure:

```python
#!/usr/bin/env python3
"""Build A5 SVG + PNG preview — causal architecture Stage 3c."""

import cairosvg
from _deck_common import (
    # ... shared constants
    check_no_text_collisions,
    check_text_within_bounds,
    collision_guard,
)

def build_status_pill(x, y, width, height):
    """The non-negotiable forward-looking status pill."""
    pass

def build_neumann_block(x, y, width, height):
    """Top zone — the equation visual hero."""
    pass

def build_grn_panels(x, y, width, height):
    """Middle zone — STRING prior + learned GRN side-by-side."""
    pass

def build_log_fc_head(x, y, width, height):
    """Bottom zone — direct-effect decoder block."""
    pass

def build_svg():
    """Generate A5 SVG."""
    parts = []
    parts.append(build_header("A5", "ARCHITECTURE DEPTH",
                              "Causal architecture — spec-locked, validation post-Phase-1",
                              "Neumann propagation + sparse learned GRN + direct-effect decoder. Architecturally locked in spec v1.1. Validation begins Q1-Q2 2027 once Phase 1 wet-lab perturbation data lands."))
    parts.append(build_status_pill(...))  # top-right, prominent
    parts.append(build_neumann_block(...))
    parts.append(build_grn_panels(...))
    parts.append(build_log_fc_head(...))
    parts.append(build_source_footer(...))
    parts.append(build_pagination("A5", total=14))
    return wrap_svg(parts)

if __name__ == "__main__":
    svg = build_svg()
    # Run collision guard with tightened threshold
    collisions = check_no_text_collisions(svg, min_gap=2)
    blocking = [c for c in collisions if not is_known_false_positive(c)]
    if blocking:
        print(f"BLOCKING COLLISIONS: {blocking}")
        raise SystemExit(1)
    
    with open("A5_causal_architecture.svg", "w") as f:
        f.write(svg)
    
    cairosvg.svg2png(url="A5_causal_architecture.svg",
                     write_to="A5_causal_architecture_preview.png",
                     output_width=1920, output_height=1080)
    print("Built A5 SVG + PNG preview")
```

---

## Deliverable Sequence

Single commit covering all 3 files:

```bash
git add docs/deck/assets/diagrams/A5_causal_architecture.svg \
        docs/deck/assets/diagrams/A5_causal_architecture_preview.png \
        docs/deck/assets/diagrams/_build_a5.py
git commit -m "docs(deck): A5 SVG - causal architecture Stage 3c"
git push origin main
```

Single-line commit message per zsh history-expansion lesson from earlier.

After this lands, separate prompt for pptx v3 reassembly to insert A5 between A4 and B section divider.

---

## What Ash Will Check On Review

Same protocol as previous SVG reviews — visual verification at slide-fill scale + zoomed inspection of dense regions:

1. **Status pill prominence and placement**: visible immediately at slide-fill scale, doesn't overlap pagination, reads as "spec-locked + forward-looking" not "operational claim"
2. **Neumann equation typography**: math glyphs render correctly (especially `(I − W)⁻¹` with Unicode minus + superscript, `dₚ` with subscript)
3. **Color coding consistency**: W cyan / dₚ lavender / propagation operator green — matches Section A palette
4. **GRN visualization clarity**: STRING prior panel vs learned sparse GRN panel — visual contrast obvious (thick weighted edges vs thin grey edges)
5. **Bottom zone direct-effect head**: block diagram readable, two-row comparison shows the Stage 3a/3b vs Stage 3c distinction clearly
6. **No collisions, no off-card text**: helper smoke clean at `min_gap=0`
7. **Honesty signals throughout**: no language anywhere claims Stage 3c is operational/validated today

If any specific element needs iteration, single fix prompt — same workflow as A3 v2, B2/D1/D2 fixes, F1 v2.

---

## What's Out Of Scope For This Task

- Modifying any existing SVG (A1-F1 locked)
- Modifying A5 content spec (committed at `ff92117`)
- Updating the pptx (separate prompt after A5 SVG lands)
- Writing the architecture spec v1.2 causal-layer extension (Phase 4 scope or separate task)
- Phase 4 visual polish

---

## Risks To Flag

1. **Math typography is the biggest visual risk**. `(I − W)⁻¹ · dₚ` requires Unicode minus (U+2212), proper superscript `⁻¹`, subscript `dₚ`. If any of these fall back to ASCII (hyphen, `-1`, `dp`), the equation visually degrades. Test the rendered PNG at full slide-fill scale before commit.

2. **Status pill positioning**. Top-right is dense (pagination + pill). If they compete visually, options: (a) stack pagination above pill, (b) place pill below pagination at narrower width, (c) use full-width pill at very top of content area below header. Cowork's judgment, but pill must dominate pagination.

3. **GRN graph rendering**. 6-8 nodes with edges is non-trivial in raw SVG. Two viable approaches: (a) hand-author node positions + edges in SVG XML, or (b) use matplotlib + networkx with cairosvg export. Either works. Avoid: random network-graph-library defaults that produce ugly layouts.

4. **Color-coding overload**. W cyan + dₚ lavender + propagation operator green is 3 colors on a single equation. Test legibility at slide-fill scale. If too busy, fall back to 2 colors (W cyan, dₚ lavender, propagation operator white).

5. **"Spec-locked" vs "validated" language**. The slide must never use "validated" in a way that implies Stage 3c has validation results today. "Spec-locked" is the correct framing. Speaker notes can elaborate on validation timing. On-slide language stays tight.

6. **Pagination outlier**. A5 shows `/ 14`; F1 still shows `/ 13`; A1-E1 still show `/ 12`. This is intentional pre-Phase-4. Phase 4 unifies. Don't preemptively change other slides' pagination during A5 work.

7. **Collision-guard at min_gap=2 may flag legitimate close-spacing**. If a real false positive appears, document it explicitly with reasoning. Don't auto-filter without inspection (F1 v1 lesson).

---

## After This Lands

If A5 SVG ships clean:
1. **Re-run `_build_appendix_pptx.py` for v3** to assemble 21-slide deck (1 cover + 6 dividers + 14 content)
2. A5 content slide inserts between A4 (slide 6) and Section B divider (slide 7)
3. Bumps pptx slide count: 20 → 21
4. F1's slide number unchanged (still last), A5 inserts mid-deck

This is **the final SVG generation task** before Phase 4 polish begins.

---

## Tool Selection Confirmation

**Cowork** (Python svgwrite/matplotlib + cairosvg PNG render) — same pattern as all Phase 2 + F1 + Batch 2 work.

Not Claude Design (that's Phase 4 polish only). Not Claude Code.
