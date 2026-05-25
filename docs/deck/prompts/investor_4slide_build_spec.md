# Investor 4-Slide Deck — Build Spec

**Owner**: Cowork (execution)
**Estimated time**: 3-5 hours (this is a fresh deck, not an iteration on existing)
**Production path**: **Path A — python-pptx native shapes** (fully editable in PowerPoint)
**Input source**: Kinga CEO brief (2026-05-XX) + v5 deck content where aligned (commit `19cc560`)
**Output**: `docs/deck/exports/aivc_investor_4slide_v1.pptx`

---

## Audience & Style Constraints

**Audience**: Biotech investors, VCs, non-technical executives. Reviewers will spend ~20 seconds per slide on first pass.

**Style**:
- Modern, premium, AI + biotech aesthetic
- One clear message per slide
- Large readable fonts (minimum 18pt body, 28-36pt headlines)
- High contrast (dark navy backgrounds with accent colors, OR clean white with dark text)
- Generous whitespace
- Minimal text — bullets ≤ 8 words where possible
- No equations, no dense plots, no scientific jargon
- 2-3 accent colors maximum

**Editability requirement (critical)**:
- ALL slides must use **native PowerPoint shapes, text boxes, connectors, and editable charts**
- NO flattened images for diagrams or text
- NO SVG-as-PNG embedded as picture
- Reviewers must be able to edit any text, recolor any shape, move any element directly in PowerPoint on Mac
- Use python-pptx primitives: `add_shape()`, `add_textbox()`, `add_connector()`, `MSO_SHAPE.*`, `Run.font.size`, etc.

---

## Color Palette (Premium AI + Biotech)

Use exactly these RGB values across all 4 slides for consistency:

| Token | RGB | Hex | Use |
|---|---|---|---|
| BG_DARK | (10, 14, 26) | #0A0E1A | Slide background (dark navy) |
| FG_PRIMARY | (255, 255, 255) | #FFFFFF | Primary text |
| FG_SECONDARY | (160, 175, 200) | #A0AFC8 | Secondary text, captions |
| ACCENT_CYAN | (38, 221, 249) | #26DDF9 | Primary accent — "Now" / public substrate |
| ACCENT_LAVENDER | (139, 92, 246) | #8B5CF6 | Phase 1 accent — proprietary phospho |
| ACCENT_AMBER | (245, 158, 11) | #F59E0B | Phase 2 accent — VDJ + scale |
| ACCENT_GREEN | (74, 222, 128) | #4ADE80 | KPI / value / success |
| BORDER_SUBTLE | (45, 58, 87) | #2D3A57 | Card borders, dividers |

**Color discipline**: Each phase has ONE assigned accent (Now=cyan, Phase 1=lavender, Phase 2=amber). KPIs use green. No other colors.

---

## Typography

Use ONE font family throughout: **Calibri** (universal PowerPoint default — guarantees no font substitution issues on reviewer's Mac).

Font scale:
- Slide title: 36pt bold
- Slide subtitle: 18pt regular
- Section header (inside slide): 22pt semibold
- Body bullet: 14pt regular
- Caption / footnote: 11pt italic
- KPI / hero number: 60pt bold

If reviewer wants different font later, they change it once in Master Slide — preserves editability.

---

## Slide Dimensions

Standard 16:9 widescreen: **13.333" × 7.5"** (12192000 × 6858000 EMU).

All 4 slides same dimensions.

---

## SLIDE 1 — AIVC General Architecture

**Title**: "AIVC Platform Evolution"
**Subtitle**: "From public benchmarking to scalable causal biological intelligence"

**Layout**: 3-panel horizontal flow (Now → Phase 1 → Phase 2) with footer strip for Phase 3.

### Panel 1 — NOW (left, cyan accent)

**Header**:
- "NOW" (small caps, 14pt, cyan)
- "Foundation & Benchmarking" (22pt semibold, white)

**Content (vertical text stack, no bullets, just lines)**:
- "Public multimodal datasets"
- "3 reference papers"
  - sub-line: "DOGMA-seq · Calderon 2019 · Mimitou CRISPR" (12pt, secondary color)
- "Multi-omics encoder pretraining"
- "73% cross-corpus accuracy" (cyan accent, slightly larger)

**Visual element below content**:
- 3 small modality icons in a horizontal row: RNA helix · ATAC stripes · Protein blob
- Use simple geometric shapes (circles, rectangles) with cyan fill, not raster icons
- Label each icon below in 11pt secondary color

**Bottom of panel**:
- Single sentence in italic, 13pt:
- *"Foundational biological representation system, built on validated public data."*

### Panel 2 — PHASE 1 (middle, lavender accent)

**Header**:
- "PHASE 1" (small caps, 14pt, lavender)
- "Controlled Perturbation Learning" (22pt semibold, white)
- Small line below: "Q3 2026" (12pt, lavender)

**Content (vertical text stack)**:
- "QuRIE-seq · proprietary multi-omics"
- "3 modalities directly measured:"
  - "RNA · Proteins · Phosphoproteins" (lavender accent)
- "PBMCs · 5 donors · 5 timepoints"
- "5 stimuli · 10 inhibitors"
- "BTK + JAK combo" (lavender, slightly emphasized)

**Visual element below content**:
- Simple perturbation diagram: 3 small modality icons (RNA, Protein, Phospho) feeding INTO a central encoder shape, with arrows
- Use native PowerPoint shapes (rectangles with rounded corners for modalities, hexagon for encoder)
- Lavender stroke, lavender fill at ~20% transparency

**Bottom of panel**:
- *"Learning causal biological responses under controlled perturbations."*

### Panel 3 — PHASE 2 (right, amber accent)

**Header**:
- "PHASE 2" (small caps, 14pt, amber)
- "Scalable Causal Discovery" (22pt semibold, white)
- Small line below: "2027" (12pt, amber)

**Content (vertical text stack)**:
- "QuRIE-seq + CRISPR + VDJ"
- "5 modalities:"
  - "RNA · Epigenetics · Proteins · Phosphoproteins · VDJ" (amber accent)
- "20-25 donors"
- "Soft perturbations: 30 stimuli + inhibitors"
- "Hard perturbations: CRISPR screening library"

**Visual element below content**:
- Slightly larger version of Phase 1 diagram showing 5 modalities feeding encoder + CRISPR arrow + VDJ arrow
- Amber stroke

**Bottom of panel**:
- *"Scaling toward large multimodal causal biological intelligence."*

### Footer (full width, bottom of slide)

- Horizontal divider line (BORDER_SUBTLE, 1pt)
- Below: "PHASE 3 — Continuation of data generation at scale + therapeutic pipeline" (14pt, FG_SECONDARY, italic)

### Connector elements

- Between Panel 1 → Panel 2: thin arrow connector (FG_SECONDARY, →)
- Between Panel 2 → Panel 3: thin arrow connector (FG_SECONDARY, →)
- Between Panel 3 → Footer: subtle continuation indicator

---

## SLIDE 2 — Causal Model + Validation

**Title**: "Causal Biological Intelligence"
**Subtitle**: "First learn structure. Then learn how signals flow."

**Layout**: Two large halves (Left = Topology Learning, Right = Directional Causal Learning) + bottom validation strip.

### LEFT HALF — Topology Learning

**Header**:
- "TOPOLOGY LEARNING" (small caps, 14pt, cyan)
- "Discover biological structure" (22pt semibold, white)

**Visual** (this is the slide's hero element):

- Draw a small network: 7-9 circles (nodes) connected by THIN UNDIRECTED LINES
- Place nodes in roughly circular cluster arrangement
- Use python-pptx oval shapes for nodes (each ~25-35pt diameter)
- Use line connectors for edges (no arrowheads in this version)
- Nodes: cyan fill, white border
- Edges: BORDER_SUBTLE color, 1.5pt

**Below visual**, 2 single-line points (14pt, white):
- "Identify how biological components organize"
- "Build the latent map of cell biology"

### RIGHT HALF — Directional Causal Learning

**Header**:
- "DIRECTIONAL CAUSAL LEARNING" (small caps, 14pt, lavender)
- "Model perturbation flow" (22pt semibold, white)

**Visual** (mirrors left side but with directional arrows):

- Same node arrangement as left
- Now edges are ARROWS (directional connectors)
- Arrow thickness VARIES — 3 thick arrows (3pt, lavender), 4 medium arrows (2pt, lavender at 70% opacity), rest thin (1pt, BORDER_SUBTLE)
- One node should have a small "lightning bolt" or filled circle indicator showing it's the perturbation source
- Lavender accent on the perturbed node and outgoing arrows

**Below visual**, 3 single-line points (14pt, white):
- "Infer directional signaling"
- "Estimate influence strength (edge bandwidth)"
- "Trace perturbation effects through network"

### BOTTOM STRIP — Validation

Horizontal row of 3 small validation cards (rounded rectangles, BORDER_SUBTLE border, no fill).

Each card has:
- Small icon at top (geometric shape, green color)
- Card title (16pt semibold, white)
- One-line description (12pt, secondary)

**Card 1**: ✓ icon · "Perturbation Validation" · "Holds out perturbations, tests predictions"

**Card 2**: ✓ icon · "Pathway Recovery" · "Recovers known biological pathways from data"

**Card 3**: ✓ icon · "Cross-State Consistency" · "Consistent predictions across cell states"

### Footer note (very subtle, 11pt italic, secondary):

*"AIVC learns both biological structure and how signals propagate through the system."*

---

## SLIDE 3 — Multimodal Encoder + Client Value

**Title**: "Multimodal Encoder Architecture"
**Subtitle**: "Unifying complex multi-omics biology into actionable intelligence"

**Layout**: Two columns. Left = technical architecture flow. Right = 4 value cards.

### LEFT COLUMN — Technical Architecture

**Header**: "ARCHITECTURE" (small caps, 14pt, cyan)

**Visual flow** (vertical, top to bottom):

**Top row** — 5 input modality boxes in a horizontal strip:
- Box per modality, rounded rectangle, BORDER_SUBTLE border, small icon + label
- Modalities: "RNA" (cyan) · "Epigenetics" (cyan) · "Proteins" (cyan) · "Phosphoproteins" (lavender) · "VDJ" (amber)
- Color-code: cyan = available today, lavender = Phase 1, amber = Phase 2
- Tiny legend below: "✓ Today  ◆ Phase 1  ▲ Phase 2" (11pt)

**Arrows down** from all 5 modalities converging on:

**Middle** — Central encoder block:
- Large rounded rectangle, lavender fill (20% opacity), lavender border (2pt)
- Centered text: "Unified Multimodal Encoder" (18pt bold, white)
- Sub-line: "256-dimensional latent representation" (11pt, secondary, italic)

**Arrows down** from encoder to:

**Bottom row** — 3 output boxes in horizontal strip:
- Box per output, smaller than input boxes
- "Biological State" · "Perturbation Response" · "Causal Inference"
- All cyan tint

### RIGHT COLUMN — Client / Commercial Value

**Header**: "VALUE TO PARTNERS" (small caps, 14pt, green)

**Layout**: 2×2 grid of value cards.

Each card: rounded rectangle, BORDER_SUBTLE border, no fill, ~2.5" × 1.5".

Each card contains:
- Small icon at top (geometric, green color, ~24pt)
- Card title (18pt semibold, white)
- One-line description (12pt, secondary)

**Card 1** (top-left): 💊 icon · "Drug Response Prediction" · "Predict combination efficacy"

**Card 2** (top-right): 🔬 icon · "Biomarker Discovery" · "Identify patient stratification markers"

**Card 3** (bottom-left): 🎯 icon · "Target Prioritization" · "Rank therapeutic targets by causal evidence"

**Card 4** (bottom-right): 👥 icon · "Patient Stratification" · "Match patients to optimal interventions"

(Note: Icons should be drawn as simple PowerPoint shapes — circles, triangles, etc. — NOT emoji, to keep editability. Emoji shown above for illustration of intent only.)

### BOTTOM FOOTER (full width):

- Thin horizontal divider
- "Designed to scale across datasets, perturbations, and therapeutic programs." (13pt italic, secondary)

---

## SLIDE 4 — Roadmap, Budget & Key Inflection Points

**Title**: "Roadmap & Key Inflection Points"
**Subtitle**: "Execution plan and platform value compounding"

**Layout**: Top section = horizontal roadmap timeline. Bottom section = inflection point cards.

### TOP SECTION — Horizontal Roadmap Timeline

**Visual structure**:

Horizontal axis with 5 milestone markers, each labeled below the axis line.

The axis itself: a horizontal line (BORDER_SUBTLE, 2pt) spanning the slide width, with 5 evenly-spaced filled circles (12pt diameter) along it.

Each milestone has:
- Filled circle on the axis (color-coded)
- Label ABOVE: short title + timing
- Description BELOW: 1-2 lines + expected outcome

**Milestone 1** (cyan): "Public Dataset Benchmarking" · "2025-Q2 2026" · Outcome: "Validated encoder, 73% cross-corpus"

**Milestone 2** (lavender): "Phase 1 Perturbation Learning" · "Q3 2026" · Outcome: "Causal signal learning + BTK+JAK demo"

**Milestone 3** (amber): "Phase 2: CRISPR + Multimodal Expansion" · "2027" · Outcome: "5-modality scaling, 20-25 donors"

**Milestone 4** (amber): "Scaled Data Generation" · "2027-2028" · Outcome: "Cross-state reasoning"

**Milestone 5** (green): "Therapeutic Discovery Applications" · "2028+" · Outcome: "Discovery enablement"

### BOTTOM SECTION — Key Inflection Points

**Header**: "KEY INFLECTION POINTS" (small caps, 14pt, green)

**Layout**: 5 small cards in a horizontal row.

Each card: rounded rectangle, BORDER_SUBTLE border, no fill, small icon + 1-line title.

**Card 1**: "Proprietary Multimodal Data" (Phase 1 lands)

**Card 2**: "Perturbation-Scale Expansion" (Phase 2 lands)

**Card 3**: "Causal Validation" (Stage 3c)

**Card 4**: "Strategic Partnerships" (Pipeline 1 starts)

**Card 5**: "Therapeutic Discovery Enablement" (Stage 5)

### Bottom footer (very subtle):

- *"Platform value compounds through proprietary data, causal learning, and scalable multimodal biological intelligence."* (11pt italic, secondary)

---

## Implementation Notes (python-pptx specifics)

### Required imports
```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
```

### Helper patterns Cowork should use

**Slide background color** (dark navy):
```python
def set_slide_bg(slide, rgb):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = rgb
```

**Text box with styled run**:
```python
def add_text(slide, left, top, width, height, text, size=14, bold=False, color=None, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Calibri"
    if color:
        run.font.color.rgb = color
    return box
```

**Rounded rectangle (card)**:
```python
def add_card(slide, left, top, width, height, fill_color=None, border_color=None, border_width=1):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.adjustments[0] = 0.05  # corner radius
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_width)
    return shape
```

**Connector arrow**:
```python
def add_arrow(slide, x1, y1, x2, y2, color, width=1.5):
    connector = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = color
    connector.line.width = Pt(width)
    # Add arrowhead
    line_elem = connector.line._get_or_add_ln()
    tailEnd = line_elem.find(qn('a:tailEnd'))
    if tailEnd is None:
        from pptx.oxml.ns import qn
        from lxml import etree
        tailEnd = etree.SubElement(line_elem, qn('a:tailEnd'))
    tailEnd.set('type', 'triangle')
    return connector
```

**Oval (network node)**:
```python
def add_node(slide, cx, cy, diameter, fill_color, border_color):
    radius = diameter / 2
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - radius), Inches(cy - radius), Inches(diameter), Inches(diameter))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    return shape
```

### Color RGB constants

```python
BG_DARK         = RGBColor(0x0A, 0x0E, 0x1A)
FG_PRIMARY      = RGBColor(0xFF, 0xFF, 0xFF)
FG_SECONDARY    = RGBColor(0xA0, 0xAF, 0xC8)
ACCENT_CYAN     = RGBColor(0x26, 0xDD, 0xF9)
ACCENT_LAVENDER = RGBColor(0x8B, 0x5C, 0xF6)
ACCENT_AMBER    = RGBColor(0xF5, 0x9E, 0x0B)
ACCENT_GREEN    = RGBColor(0x4A, 0xDE, 0x80)
BORDER_SUBTLE   = RGBColor(0x2D, 0x3A, 0x57)
```

---

## Build Script Structure

Single script: `docs/deck/investor_4slide/_build_investor_deck.py`

Structure:
```python
def build_slide1_evolution(prs): ...
def build_slide2_causal(prs): ...
def build_slide3_encoder(prs): ...
def build_slide4_roadmap(prs): ...

def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    build_slide1_evolution(prs)
    build_slide2_causal(prs)
    build_slide3_encoder(prs)
    build_slide4_roadmap(prs)
    
    out = "docs/deck/exports/aivc_investor_4slide_v1.pptx"
    prs.save(out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
```

Build each slide function independently for maintainability.

---

## Acceptance Criteria

- ✓ Exactly 4 slides
- ✓ 16:9 widescreen (13.333" × 7.5")
- ✓ All slides use BG_DARK background
- ✓ All text in Calibri font
- ✓ NO embedded PNG/JPG images for diagrams or text (verify with: `unzip -l output.pptx | grep -E "ppt/media/" | grep -v "thumbnail"` should be EMPTY)
- ✓ All visual elements are native PowerPoint shapes (verify by opening in PowerPoint and confirming shapes are selectable/editable)
- ✓ Color palette used consistently (cyan = Now, lavender = Phase 1, amber = Phase 2, green = KPI/value)
- ✓ Each slide has clear title + subtitle in consistent position
- ✓ Network visual on Slide 2 uses oval shapes + line connectors (not flattened image)
- ✓ Encoder architecture on Slide 3 uses rounded rectangles + connectors
- ✓ Roadmap on Slide 4 uses horizontal line + circle markers + text boxes
- ✓ File size <500 KB (no embedded media means it should be small)
- ✓ Opens cleanly in PowerPoint Mac without "missing fonts" or "broken visual" warnings

### Manual review (Ash on Mac)

After Cowork ships, open in PowerPoint Mac and verify:
1. Click any text element — can edit text directly
2. Click any shape — can resize/recolor/move
3. Click any connector arrow — can adjust endpoints
4. Try changing a color via shape format — works as expected
5. No flattened "picture" elements (selecting shows shape properties, not picture properties)

---

## What Content NOT To Include

These come from v5 but are too technical for this investor deck:
- Equations (Neumann propagation, decomposed readout)
- Stage 3a/3b/3c/4/5 model training nomenclature (use plain English: "encoder validation", "perturbation demo", "causal architecture")
- Layer L1-L5 dataset strategy
- Specific paper PMIDs / GSE accessions
- Encoder backbone details (latent space dimensionality, AIVC_GRAD_GUARD)
- Cross-corpus 73% Calderon result detail — use simplified "73% cross-corpus accuracy" claim only
- Detailed BTK+JAK clinical literature
- Budget breakdown (D2-level detail) — Slide 4 mentions budget conceptually; CEO/Kinga will fill specific numbers if they want them

---

## What Content TO Pull From v5 (Aligned)

These v5 framings align with Kinga's brief and should be carried forward:

- Phase 1: 5 donors, 5 timepoints (her: "5 stimuli" — slight semantic shift), QuRIE-seq integral phospho
- Phase 2: VDJ + 20-25 donors + scaled perturbations
- Phase 3: continuation of data generation at scale + therapeutic pipeline
- 5-modality vision (RNA + ATAC + Protein + Phospho + VDJ)
- Cross-corpus encoder validation result (73%)
- Causal architecture concept (topology then directional)
- Drug combination prediction commercial framing

---

## Risks To Flag

1. **"3 modalities" in Phase 1 (Kinga's text) requires interpretation** — she didn't list the 3 explicitly but our v5 reconciliation has RNA + Proteins + Phosphoproteins. Use that. Worth verbal confirmation on v1 review.

2. **"5 stimuli" vs "5 timepoints"** — Kinga's text says "5 stimuli" but our v5 says "5 timepoints (0/5/30/60/180 min)." These are different things. **Recommendation**: use both — "5 timepoints" for temporal sampling and "5 stimuli" for distinct stimulation conditions. Worth quick verbal confirmation that 5 stimuli is intended OR if she meant 5 timepoints.

3. **"10 inhibitors" is new specificity** — Thiago previously hadn't confirmed this number. CEO-level commitment now per Kinga. Use it.

4. **"30 stimuli + inhibitors" Phase 2** — interpret as 30 total combined soft perturbations, not 30+ each.

5. **Slide 3 modalities (5 vision-forward)** — confirmed by Ash. Show all 5 with the legend distinguishing today-vs-future.

6. **Editability check is non-negotiable** — if any visual element ships as a flattened PNG, the deck fails CEO requirement. Verify with the `unzip -l` check before declaring complete.

7. **Visual polish ceiling** — python-pptx native shapes will look clean but not "designer-grade" like a custom Figma deck. Lean on color discipline + whitespace + typography to make it feel premium. If CEO wants Figma-grade polish later, this becomes a manual design pass in PowerPoint.

8. **No B-cell-line CRISPR / Mimitou / Schmidt detail in this deck** — keep all dataset specifics out of investor-facing. "3 reference papers" is generic enough.

---

## After This Lands

1. Ash spot-checks on Mac in PowerPoint — confirms editability + visual polish
2. If approved → send to Kinga + CEO for content review
3. Iterate based on their feedback (v2, v3 as needed)
4. v5 technical appendix stays parked — separate audience, separate use case
5. If Kinga + CEO want the 21-slide technical appendix updated to align with new direction, that's a separate effort

---

## Out Of Scope

- Speaker notes (investor decks don't typically use them; CEO/founder presents the deck verbally)
- Master slide template extensibility
- Image/icon library beyond simple geometric shapes
- Animation / transitions
- Auto-generated PDF export
