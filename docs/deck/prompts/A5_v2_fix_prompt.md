# A5 v2 — Equation Rendering + Collision + Subtitle + GRN Visualization Upgrade

**Owner**: Cowork (execution)
**Estimated time**: 45-90 min (4 fixes, 1 is substantive visual work)
**Input commit**: `85283ab`
**Input files**:
- `docs/deck/assets/diagrams/A5_causal_architecture.svg`
- `docs/deck/assets/diagrams/_build_a5.py`
- `docs/deck/assets/diagrams/_deck_common.py`

---

## Context

A5 v1 shipped at commit `85283ab`. Ash visual review approved the structure (status pill, three-zone layout, honesty discipline) but flagged **four issues** requiring v2 iteration:

1. **Critical**: Neumann equation `I` renders as `⊢` (turnstile glyph); superscript `⁻¹` renders as `-1` flat
2. **Medium**: Text collisions in the equation card footer (`ρ(W) < 1 ... sparsity L1` overlaps with component definitions)
3. **Minor**: Subtitle overflows past slide right edge (~150 chars)
4. **Upgrade**: GRN visualization is functionally correct but visually generic — needs informative nodes (named clusters), weight-coded edges, directional rendering, confidence encoding

A5 is the architectural-depth hero slide. The math being visually correct and the GRN visualization being investor-grade are non-negotiable. v2 addresses all four.

---

## Fix 1 — Critical: Neumann Equation Rendering

### The Bug

Current rendering of `ŷ = (I − W)⁻¹ · dₚ`:

```
Renders as:    ŷ = (⊢ W )-1 · dₚ
Should be:     ŷ = (I − W)⁻¹ · dₚ
```

Two failures:
- **`I` glyph corruption**: Latin capital I is rendering as turnstile `⊢` (U+22A2)
- **Superscript broken**: `⁻¹` renders as flat `-1` instead of raised, smaller

### Likely Causes

**For the `I → ⊢` substitution**:
- The `I` may be encoded as U+2110 (SCRIPT CAPITAL I) or U+1D43C (mathematical italic I) instead of U+0049 (Latin capital I)
- The `<tspan>` for the I has font-family fallback chain that lands on a math font where the script-I → turnstile glyph
- The tspan has `font-style="italic"` triggering math-italic substitution

**For the superscript failure**:
- `baseline-shift="super"` SVG attribute is being silently dropped by cairosvg
- Or the superscript content `⁻¹` is two separate characters that aren't being shifted as a unit
- Or font-size reduction wasn't applied alongside baseline-shift

### Fix

**For the `I`**: explicit Latin capital I, no italic, no math-font substitution allowed:

```python
# In _build_a5.py, the equation tspan for I:
# BAD (probably what's there now):
'<tspan font-style="italic" baseline-shift="0">I</tspan>'

# GOOD: explicit Latin script + no italic + no math-substitution risk
'<tspan font-family="Inter, Arial, sans-serif" font-style="normal">I</tspan>'
```

Verify the character is genuinely U+0049 by inspecting the source SVG file:
```bash
python3 -c "
with open('docs/deck/assets/diagrams/A5_causal_architecture.svg') as f:
    content = f.read()
# Find the I in the equation context
import re
for m in re.finditer(r'>I[^<]', content):
    char = content[m.start()+1]
    print(f'I codepoint: U+{ord(char):04X}')
"
```

If codepoint isn't `U+0049`, fix the source string.

**For the superscript `⁻¹`**: don't rely on `baseline-shift`. Use Unicode superscript characters directly:

```python
# BAD (cairosvg may drop baseline-shift):
'<tspan baseline-shift="super" font-size="32">-1</tspan>'

# GOOD: use Unicode superscript minus + superscript one
# U+207B = SUPERSCRIPT MINUS
# U+00B9 = SUPERSCRIPT ONE
# Render at full equation size; they're already superscript-sized glyphs
'<tspan>⁻¹</tspan>'  # literal Unicode superscript characters
```

These Unicode characters render as proper superscripts in Inter and Arial without needing CSS/SVG baseline-shift. No risk of silent failure.

### Acceptance for Fix 1

- ✓ Source SVG file contains literal Latin capital I (U+0049) in the equation, not math-italic or script variants
- ✓ Source SVG file contains `⁻¹` as Unicode characters U+207B + U+00B9, not `<tspan baseline-shift="super">-1</tspan>`
- ✓ Rendered PNG at slide-fill scale shows `(I − W)⁻¹` reading correctly:
  - `I` as a vertical Latin capital letter, NOT a turnstile
  - `⁻¹` raised and smaller, NOT flat baseline
- ✓ Unicode minus U+2212 still in use (not hyphen) — verify retained from v1

---

## Fix 2 — Equation Card Text Collisions

### The Bug

In the equation card (top zone), the architectural-requirement footer `Architectural requirement: ρ(W) < 1 enforced by sparsity L1 — guarantees Neumann-series convergence` collides with the component-definition row above it.

Visual evidence: at slide-fill scale, `(I−W)⁻¹ closed-form propagation` annotation overlaps with the footer's `sparsity L1 — guarantees...` text. The two text elements occupy the same y-band.

### Fix

Move the architectural-requirement footer to its own y-band, below all component definitions, with explicit 20px vertical gap:

```
Equation card y-band layout (proposed):
y=210-260:  Equation `ŷ = (I − W)⁻¹ · dₚ` (visual hero)
y=290-330:  Component definitions row (W cyan / dₚ lavender / (I-W)⁻¹ green)
y=380-410:  Architectural requirement footer (italic muted, own y-band, 50px gap above)
```

Card bottom: y=425 minimum (some buffer below footer).

### Helper Audit (CRITICAL — pattern repeated)

**This is the 5th text-collision bug we've hit** (B2/D1/D2/F1/A5). The helper has caught some but not all. Cowork's F1 v2 work tightened `min_gap` from 4 to 2; A5 v1 shipped at `min_gap=2` with collisions still present.

**Don't just fix A5's specific collision. Investigate why min_gap=2 missed it.** Three possibilities:

1. **Y-overlap was below the threshold**: report the actual y-overlap for the failing pair
2. **Filter scope swept it**: report whether the failing pair matched any filter pattern
3. **Heuristic estimate failed**: glyph width × char count under-estimated actual rendered width

Document findings in the commit message. If the fix is "tighten min_gap further" → confirm at `min_gap=0` strict. If the fix is "remove filter pattern X" → name the pattern. If the fix is "improve width heuristic" → describe the change.

**Sweep recommendation for `_deck_common.py`**: change the default `min_gap` from 4 to 2 across the module. Existing builders (B2/D1/D2/F1 v2/A5 v2) already pass `min_gap=2` explicitly so no regression risk. Future builders inherit the safer default.

### Acceptance for Fix 2

- ✓ Component definitions and architectural-requirement footer at separate y-bands with ≥20px vertical gap
- ✓ Visual zoom into the equation card shows no text overlap anywhere
- ✓ Helper audit report in commit message: which case (1/2/3 above) explains why v1 missed this collision
- ✓ `_deck_common.py` default `min_gap` changed from 4 to 2 (sweep recommendation)
- ✓ Helper smoke clean at `min_gap=0` (strictest setting)

---

## Fix 3 — Subtitle Trim

### The Bug

Current subtitle (~150 chars) overflows past the slide's right edge:
```
Neumann propagation + sparse learned GRN + direct-effect decoder · architecturally locked in spec v1.1 · validation begins Q1-Q2 2027 once Phase 1 wet-lab perturbation data lands
```

### Fix

Trim to ~100 chars, single-line:

```
Neumann propagation + sparse learned GRN + direct-effect decoder · spec-locked v1.1 · validation post Phase 1 (Q1-Q2 2027)
```

The validation timing detail moves entirely to the status pill (which already carries it). Subtitle gets terser, slide gains breathing room.

### Acceptance for Fix 3

- ✓ Subtitle character count ≤ 130 (with comfortable buffer)
- ✓ Subtitle renders as single line at slide-fill scale, doesn't overflow right edge
- ✓ All key concepts preserved (Neumann + GRN + log-FC + spec-lock + post-Phase-1 validation)

---

## Fix 4 — GRN Visualization Upgrade (Substantive Visual Work)

### The Strategic Problem

A5's GRN visualization is functionally correct (7 generic nodes, before/after with thin grey priors vs thick cyan learned) but **visually generic**. This is the architectural-depth hero slide; the GRN diagram should be the most visually compelling element after the equation, not the most pedestrian.

Ash's direction: "the graphs need to be nicer and more informative."

### Upgrade Spec — Four Improvements

**Improvement A — Named biologically-meaningful node clusters**

Replace 7 generic circles with **named gene clusters tied to the platform's actual biological scope**. This gives the GRN narrative coherence with C2 (BTK+JAK demo) and B3 (CD3E+CD4 substitute):

Proposed 8-node GRN (representative, immune-perturbation-relevant):

| Node | Cluster | Role |
|---|---|---|
| **BTK** | BCR pathway | Direct perturbation target — Phase 1 Q3 2026 |
| **JAK** | JAK-STAT signaling | Direct perturbation target — Phase 1 Q3 2026 |
| **CD3E** | TCR complex | Stage 3 Part 1 validated perturbation (Mimitou CRISPR) |
| **NFKB** | Transcription factor | Downstream effector, hub node |
| **STAT3** | Transcription factor | JAK downstream, hub node |
| **ZAP70** | TCR signaling | T-cell activation kinase |
| **MYD88** | Innate immunity | Cross-pathway hub |
| **IRF7** | Effector | Interferon response gene |

Each node labeled with abbreviated gene name (3-5 chars max). Font: Arial bold 11pt, white on dark background.

**Improvement B — Node visual treatment**

Not generic circles. Each node:
- **Radial gradient fill** (darker center, lighter edge) — gives 3D effect without busy shadowing
- **Size differentiation by hub-degree**: hub nodes (NFKB, STAT3) larger (~28px diameter); peripheral nodes (IRF7, ZAP70) smaller (~20px diameter)
- **Color coding by gene class** (subtle, doesn't compete with edge weights):
  - Direct perturbation targets (BTK, JAK, CD3E): cyan fill (`#26DDF9` at 0.25 opacity center → 0.1 edge)
  - Transcription factors (NFKB, STAT3): lavender fill (`#8B5CF6` at 0.25 → 0.1)
  - Kinases / signaling intermediates (ZAP70, MYD88): green fill (`#4ADE80` at 0.25 → 0.1)
  - Effectors (IRF7): muted blue fill
- **Soft outer glow** (1.5px stroke at 0.45 opacity matching the fill color) — separates nodes from background without harsh outlines

**Improvement C — Directional + weight-coded edges**

GRN is a directed graph. Edges should show direction.

Edge encoding (3 channels simultaneously):
- **Direction**: small triangular arrowhead at target end (lavender, ~6px)
- **Weight**: stroke width — 4px (high), 2.5px (medium), 1.5px (low)
- **Confidence/source**:
  - STRING-supported edges (left panel + retained in right panel): solid stroke, base color cyan
  - Novel learned edges (right panel only, not in STRING): solid stroke, brighter cyan with subtle glow
  - Pruned edges (right panel): dashed grey at 0.4 opacity

**Improvement D — Information density without clutter**

Left panel (STRUCTURAL PRIOR):
- All STRING-supported edges visible
- Edge thickness uniform (all 2px) — prior doesn't carry weight info, just existence
- Color: muted grey `#A8B4C2` at 0.55 opacity
- Arrowheads optional (priors can be undirected to keep visual simple)

Right panel (LEARNED SPARSE GRN):
- Same nodes, **subset of edges** with weight + direction
- Strong learned edges: cyan `#26DDF9` at 4px stroke with arrowheads
- Medium learned edges: cyan at 2.5px
- Pruned edges (below sparsity threshold, originally in STRING prior): dashed grey 1.2px at 0.4 opacity
- **Novel discovered edge** (not in STRING prior but learned): bright cyan + small lavender circle annotation marker — visually highlights "we learned something the prior didn't predict"

Caption below both panels (existing, unchanged):
> "prior shapes initialization, learning prunes"

Plus add a small legend (bottom-right of right panel):
```
━━ high-weight learned
─── medium-weight learned  
··· pruned (below threshold)
◆ novel (not in STRING prior)
```

### Reference For Visual Aesthetic

Ash mentioned the quriegen-demo repo has similar visualizations. **Don't read that repo** — Ash explicitly said it's unnecessary. The spec above contains enough detail to execute.

Conceptually the aesthetic to aim for:
- Network science publication quality (think *Cell Systems* figure or BioRender illustration)
- Not "PowerPoint default org chart"
- Information-dense but not cluttered
- Each visual encoding (color, size, weight, direction, line style) carries one specific meaning

### Implementation Note

Two viable rendering paths:

**Path A — Hand-author node positions + edges in SVG**
- More control, deterministic positioning
- More work to encode
- Right approach for an 8-node graph

**Path B — networkx + matplotlib export to SVG**
- Faster for larger graphs
- Less control over per-node aesthetic
- May produce generic-looking output

**Recommendation: Path A.** Eight nodes is small enough that hand-authoring positions gives much better visual control. Layout nodes manually on a deliberate grid (e.g., 3 hub nodes in center column, peripheral nodes around them) so the GRN reads as "structured biology" not "random graph."

### Acceptance for Fix 4

- ✓ 8 named gene nodes present (BTK, JAK, CD3E, NFKB, STAT3, ZAP70, MYD88, IRF7) — exact names per spec table above
- ✓ Each node has gene label visible at slide-fill scale (≥10pt effective size)
- ✓ Hub nodes (NFKB, STAT3) visually larger than peripheral nodes
- ✓ Color coding by gene class implemented (cyan = perturbation targets, lavender = TFs, green = signaling, muted blue = effectors)
- ✓ Nodes have radial gradient fills, not flat colors
- ✓ Edges show direction (arrowheads on target end)
- ✓ Edge weights distinguishable (high/medium/low stroke widths)
- ✓ Right panel shows: STRING-retained edges + at least 1 novel learned edge (highlighted) + at least 2 pruned edges (dashed)
- ✓ Legend present in right panel showing line-style encoding
- ✓ Caption "prior shapes initialization, learning prunes" preserved
- ✓ Both panels read as "before → after" with clear visual contrast

---

## Combined Acceptance Criteria For A5 v2

Beyond the 4 fix-specific acceptance items above:

### No regression on v1 approved elements

- ✓ Status pill preserved (STAGE 3c · SPEC-LOCKED · Validation Q1-Q2 2027 · post Phase 1)
- ✓ Status pill positioning unchanged (top-right, doesn't overlap pagination)
- ✓ Direct-effect log-FC head block (bottom zone) preserved
- ✓ Causal vs predictive comparison rows preserved
- ✓ Honesty discipline maintained (no "operational" / "validated" / "in production")
- ✓ Section A palette (cyan + lavender + green) — no new colors
- ✓ Header eyebrow + pagination + source footer pattern preserved

### Helper validation

- ✓ Helper smoke at `min_gap=2`: 0 blocking
- ✓ Helper smoke at `min_gap=1`: 0 blocking
- ✓ Helper smoke at `min_gap=0`: 0 blocking
- ✓ Helper audit findings documented (why v1 missed the collision)
- ✓ `_deck_common.py` default min_gap updated from 4 to 2

### Math typography (Fix 1)

- ✓ `I` renders as Latin capital, not turnstile
- ✓ `⁻¹` renders as raised superscript, not flat
- ✓ Unicode minus U+2212 preserved in `I − W`
- ✓ Greek `ρ` preserved in architectural requirement

---

## Deliverable

Single commit covering all 4 files:

```bash
git add docs/deck/assets/diagrams/A5_causal_architecture.svg \
        docs/deck/assets/diagrams/A5_causal_architecture_preview.png \
        docs/deck/assets/diagrams/_build_a5.py \
        docs/deck/assets/diagrams/_deck_common.py
git commit -m "fix(deck): A5 v2 - equation rendering, collisions, GRN upgrade"
git push origin main
```

Single-line commit message per zsh history-expansion lesson.

---

## What Ash Will Check On Review

Same protocol as previous SVG reviews — visual verification at slide-fill scale + zoomed inspection on:

1. **Neumann equation**: zoom in, verify `I` is Latin capital not turnstile, `⁻¹` is raised superscript
2. **Equation card**: zoom in, verify component definitions and architectural-requirement footer are at separate y-bands with clear gap
3. **Subtitle**: full-slide check, verify single line within slide width
4. **GRN visualization**: zoom in, verify named nodes (BTK, JAK, CD3E, NFKB, STAT3, ZAP70, MYD88, IRF7), hub-vs-peripheral sizing, color coding by gene class, directional edges, weight differentiation, novel-edge highlighting in right panel, legend present

If any element still needs iteration → single fix prompt. But four iterations on the same slide is the upper bound — at some point we ship v3 and move on.

---

## What's Out Of Scope

- Modifying A5 content spec (committed at `ff92117`)
- Modifying any other SVG (A1-F1 locked)
- pptx assembly (separate prompt after A5 v2 lands)
- Phase 4 polish work
- Reading the quriegen-demo repo (Ash explicitly excluded)

---

## Risks To Flag

1. **The `I → ⊢` rendering bug may have a root cause I haven't fully diagnosed.** If after fixing as proposed (explicit U+0049, explicit Inter font, no italic) the issue persists, that's evidence cairosvg or Inter has a deeper substitution rule. Fallback: render `I` as a simple vertical bar SVG `<line>` element with the same dimensions as a glyph, labeled with a `<title>` for accessibility. Ugly but bulletproof.

2. **GRN visualization is the time-intensive fix**. Estimate 30-45 min alone for Fix 4 (vs ~15 min each for Fixes 1-3). Budget accordingly.

3. **Named gene clusters create cross-slide dependency**. BTK and JAK appear on the GRN; if C2 ever changes its demo target away from BTK+JAK (unlikely but possible), A5's GRN becomes inconsistent. Flag this in the commit message as a known coupling.

4. **The helper false-negative on A5 v1 is concerning**. min_gap=2 caught some collisions (the title-vs-pill collision Cowork caught pre-write) but missed others (the equation-card-footer collision). Helper audit per Fix 2 acceptance is mandatory — don't ship A5 v2 without diagnosing why min_gap=2 missed the v1 case.

5. **Novel-edge highlighting in right panel** is a content claim. If Stage 3c isn't yet implemented (which we said in v1 honesty discipline), we don't actually know what novel edges the GRN will learn. The visualization is illustrative, not derived from data. A small caption note ("illustrative — actual learned GRN structure depends on Phase 1 data") may be appropriate. Cowork's judgment on whether this caption fits the layout budget; if not, speaker note covers it.

6. **8 named nodes is the upper bound for the node count**. More than 8 + edge clutter + labels overwhelms the panels. Stay at 8; don't expand to "show more genes."

---

## After This Lands

If A5 v2 ships clean:
1. Visual verification on Mac
2. **pptx v3 reassembly** — same prompt pattern as pptx v2, adds A5 + Section A grows from 4 to 5 slides, total deck 21 slides
3. Phase 4 polish prompt drafted with audit-driven content additions + A5 implications baked in

We're at iteration 2 on A5. Iteration 3 (v3) is a soft cap — if A5 v2 still has issues, we step back and reassess scope rather than iterate further.
