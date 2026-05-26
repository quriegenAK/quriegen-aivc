# Investor Deck v3 — Full 4-Slide Build Spec

**Owner**: Cowork (execution)
**Estimated time**: 3-5 hours (all 4 slides)
**Production path**: Path A — python-pptx native shapes (continues from v1/v2)
**Output**: `docs/deck/exports/aivc_investor_4slide_v3.pptx`
**Input source**: Kinga CEO brief + Ash design constraints (Apple-keynote direction confirmed)

---

## Design Direction (Non-Negotiable, Applies To All 4 Slides)

**Apple-keynote minimal aesthetic** — radical simplicity. Three structural levers do ALL the work:

1. **Typography weight + scale** — massive, confident. Where v2 used 18-40pt, v3 uses 24-72pt for hero elements
2. **Color saturation + contrast** — color = semantics. Active phase pops, others recede. No subtle 60%-opacity borders
3. **Whitespace as structure** — negative space IS the design. Aggressive empty area carries premium feel

**Explicit cuts from v2** that DO NOT carry forward:
- ❌ Card border @ 60% opacity (replace with no-border OR full-saturation single accent)
- ❌ Card-bottom italic taglines + divider line
- ❌ Three equal-weight panels (replace with visual hierarchy — current phase emphasized)
- ❌ All micro-flourishes (dot patterns, dashed connectors as decoration)

**What carries forward from v2**:
- ✓ Color palette (cyan/lavender/amber per phase, green for output/value)
- ✓ Calibri throughout
- ✓ Dark navy background
- ✓ 16:9 (13.333" × 7.5")
- ✓ Path A: native python-pptx shapes only, no embedded images
- ✓ Phase 1/Phase 2/Phase 3 content from Kinga's brief

---

## Color Palette (Unchanged from v1/v2)

```python
BG_DARK         = RGBColor(0x0A, 0x0E, 0x1A)
FG_PRIMARY      = RGBColor(0xFF, 0xFF, 0xFF)
FG_SECONDARY    = RGBColor(0xA0, 0xAF, 0xC8)
FG_MUTED        = RGBColor(0x60, 0x70, 0x88)   # NEW — for very subtle text
ACCENT_CYAN     = RGBColor(0x26, 0xDD, 0xF9)
ACCENT_LAVENDER = RGBColor(0x8B, 0x5C, 0xF6)
ACCENT_AMBER    = RGBColor(0xF5, 0x9E, 0x0B)
ACCENT_GREEN    = RGBColor(0x4A, 0xDE, 0x80)
BORDER_SUBTLE   = RGBColor(0x2D, 0x3A, 0x57)
```

---

## Typography Hierarchy (Locked Across All 4 Slides)

Apple-keynote scale. Bigger than v2 across the board.

| Element | Font | Size | Weight | Color |
|---|---|---|---|---|
| Hero number/word | Calibri | 96pt | Bold | Phase color |
| Slide title | Calibri | 44pt | Bold | FG_PRIMARY |
| Slide subtitle | Calibri | 18pt | Regular Italic | FG_SECONDARY |
| Panel header (now uppercase) | Calibri | 22pt | Bold | Phase color |
| Identity statement | Calibri | 18pt | Regular | FG_PRIMARY |
| Supporting line | Calibri | 15pt | Regular | FG_PRIMARY |
| Caption/metadata | Calibri | 12pt | Regular | FG_SECONDARY |
| Phase progression label | Calibri | 14pt | Bold | Phase color |

**Letter-spacing**: +1pt on all uppercase labels (continues v2's `spc="100"` pattern).

---

## SLIDE 1 — AIVC Platform Evolution

### Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  AIVC Platform Evolution                                           │
│  From public benchmarking to scalable causal biological            │
│  intelligence                                                      │
│                                                                    │
│                                                                    │
│      ●────────────────●━━━━━━━━━━━━━━●─────────·──→               │
│     NOW            PHASE 1          PHASE 2    Phase 3             │
│                                                                    │
│                                                                    │
│  ────────────────  ━━━━━━━━━━━━━━━━  ────────────────              │
│  NOW               PHASE 1           PHASE 2                       │
│                                                                    │
│  Public            Proprietary       Scaled                        │
│  benchmarking      perturbation      causal                        │
│                    learning          discovery                     │
│                                                                    │
│  3 reference       QuRIE-seq         + CRISPR + VDJ                │
│  papers            multi-omics                                     │
│                                                                    │
│  Pretrained        3 modalities      5 modalities                  │
│  encoder           5 donors          20–25 donors                  │
│                    5 stimuli         Soft + hard                   │
│  73% cross-        10 inhibitors     perturbations                 │
│  corpus                                                            │
│  validation        BTK + JAK         CRISPR screening              │
│                    demo              library                       │
│                                                                    │
│                                                                    │
│  Phase 3 — Continuation at scale + therapeutic pipeline →          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Critical Design Difference From v2

**No card containers.** Each phase is just typography + color on whitespace. Phase is delineated by:
- Top accent line (thick, full-saturation phase color, ~1.5" wide)
- Phase name in uppercase above content
- Color discipline (all text colored to phase OR white)
- Whitespace gutters (0.5" between phases)

Result: each phase reads as a column of pure typography, not as a card with borders. **The slide breathes.**

### Specifications

**Title block** (top zone, 0.6"-1.6"):
- Title: 44pt Calibri Bold, white, left aligned at x=0.6"
- Subtitle: 18pt Calibri Regular Italic, FG_SECONDARY, immediately below title

**Progression line** (1.9"-2.4"):
- Same as v2 (horizontal line, 4 dots, dashed segment to Phase 3) BUT:
- Line thickness: 2.5pt (was 1.5pt in v2)
- Dot diameter: 0.22" (was 0.16" in v2)
- Active phase color emphasized: dot for "current focus" (PHASE 1) gets a subtle outer ring (0.32" diameter, 1pt phase color stroke, no fill) to read as "we are building this now"
- Active phase label gets bold treatment + larger (16pt vs 14pt)

**Phase columns** (3.0"-6.5"):
- 3 columns, equal width, NO borders/cards
- Each column starts with a top accent line (thick rule, 0.18" thick, ~1.5" wide, phase color, full saturation)
- Below accent line: phase name uppercase (22pt Bold, phase color)
- Below phase name (0.4" gap): identity statement (18pt Regular, white)
- Below identity (0.5" gap): supporting lines stacked (15pt Regular, white) with 0.25" line-spacing
- **No tagline at bottom.** Whitespace is the closure.

Column x positions:
- Column 1 (NOW): x=0.8", width=3.6"
- Column 2 (PHASE 1): x=4.9", width=3.6"
- Column 3 (PHASE 2): x=9.0", width=3.6"
- Gutter between columns: 0.5"

**Phase 3 footer** (6.8"-7.1"):
- Single line: "Phase 3 — Continuation at scale + therapeutic pipeline →"
- 13pt Calibri Regular Italic, FG_MUTED
- "Phase 3" prefix in 13pt Bold, same color
- Centered horizontally
- Position: y=6.9"

### Content Per Column

**Column 1 — NOW** (cyan):

```
[cyan accent line]

NOW

Public
benchmarking

3 reference papers
Pretrained encoder
73% cross-corpus
validation
```

**Column 2 — PHASE 1** (lavender, current-focus emphasis):

```
[lavender accent line]

PHASE 1

Proprietary
perturbation
learning

QuRIE-seq multi-omics
3 modalities
5 donors
5 stimuli
10 inhibitors

BTK + JAK demo
```

**Column 3 — PHASE 2** (amber):

```
[amber accent line]

PHASE 2

Scaled
causal
discovery

+ CRISPR + VDJ
5 modalities
20–25 donors
Soft + hard perturbations

CRISPR screening
library
```

### Visual Hierarchy Mechanism

- Top zone (title) dominates by font scale (44pt)
- Progression line is **the second strongest element** (thickness + dot size pulls eye)
- Phase headers in uppercase bold colored type carry the third tier
- Content reads as supporting evidence
- Phase 3 footer is intentionally weakest (muted color, smaller, italic)

This is the "Apple keynote" hierarchy: title → visual narrative element → section labels → content → footnote. Five clear tiers.

---

## SLIDE 2 — Causal Biological Intelligence

### Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Causal Biological Intelligence                                    │
│  First learn structure. Then learn how signals flow.               │
│                                                                    │
│                                                                    │
│  ┌──────────────────────────┐    ┌──────────────────────────┐      │
│  │                          │    │                          │      │
│  │   [network: 9 nodes,     │    │  [network: 9 nodes,      │      │
│  │   undirected lines,      │ →  │   directional arrows,    │      │
│  │   cyan]                  │    │   varied thickness,      │      │
│  │                          │    │   lavender, one          │      │
│  │                          │    │   "source" node]         │      │
│  │                          │    │                          │      │
│  └──────────────────────────┘    └──────────────────────────┘      │
│                                                                    │
│  TOPOLOGY                         DIRECTIONAL                      │
│  Discover structure               Model perturbation flow          │
│                                                                    │
│                                                                    │
│        ✓                  ✓                   ✓                    │
│   Perturbation       Pathway            Cross-state               │
│   validation         recovery           consistency               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Critical Design Choices

- Two-column split (not left half / right half but two **discrete network visualizations** with a connecting arrow between them — narrative is "first this, then this")
- Networks are the heroes (~3.2" diameter each, dominate slide)
- Connecting arrow between networks (8pt thick, FG_SECONDARY, MSO_SHAPE.RIGHT_ARROW or similar) — communicates "evolution from one to the other"
- Validation row at bottom is 3 simple text columns with checkmarks (no card containers)

### Specifications

**Title block** (0.6"-1.6"): Same pattern as Slide 1.

**Network zone** (1.9"-4.8"):

Two networks, mirrored:
- Left network at x=1.3", y=1.9", area ~3.0" × 2.9"
- Right network at x=8.9", y=1.9", area ~3.0" × 2.9"
- Arrow connector between them: starts x=4.6", ends x=8.6", y=3.3" (vertically centered)
- Arrow style: thick triangle right (8pt body, large arrow head), color FG_SECONDARY

**LEFT NETWORK** (Topology — cyan):
- 9 oval nodes arranged in roughly hexagonal/cluster layout
- Node diameter: 0.32"
- Node fill: ACCENT_CYAN, no border
- ~12-15 connecting lines (undirected, straight, no arrowheads)
- Line color: BORDER_SUBTLE, 1.5pt
- Node positions roughly: center + 8 surrounding (avoid perfect circle — slightly organic placement)

**RIGHT NETWORK** (Directional — lavender):
- Same 9 nodes in same positions as left network
- Node fill: ACCENT_LAVENDER, no border
- One node (top-center, the "source") gets a 0.4" outer glow effect (single concentric circle at 50% opacity lavender, 1pt stroke)
- Connecting lines REPLACED with directional arrows:
  - 3 arrows: thick (3pt, lavender, 100% opacity)
  - 4 arrows: medium (2pt, lavender, 70% opacity)
  - 5-6 arrows: thin (1pt, BORDER_SUBTLE)
- All arrows have arrowheads (small triangle, lavender)

### Network labels (4.9"-5.4"):

Two labels, centered under their respective networks:

- Left: "TOPOLOGY" (22pt Bold Cyan letter-spaced) + 18pt Regular White "Discover structure"
- Right: "DIRECTIONAL" (22pt Bold Lavender letter-spaced) + 18pt Regular White "Model perturbation flow"

### Validation row (5.8"-6.8"):

3 text columns, equal spacing across slide width. NO card borders. Each column:
- Centered checkmark icon (drawn as PowerPoint shape, ~0.3" tall, ACCENT_GREEN)
- Below checkmark (0.15" gap): 15pt Calibri Bold White, centered
- Below text (0.1" gap): 12pt Calibri Regular FG_SECONDARY, centered

Content:
- Column 1: ✓ · "Perturbation validation" · "Held-out perturbations match predictions"
- Column 2: ✓ · "Pathway recovery" · "Recovers known biological pathways"
- Column 3: ✓ · "Cross-state consistency" · "Stable across cell states"

### What's CUT From v1 Slide 2

- ❌ Card containers for validation items
- ❌ Bullet lists below network ("Identify how biological components organize" etc.)
- ❌ Subtitle "AIVC learns both biological structure and how signals propagate through the system" footer
- ❌ Spec text inside the visual area

---

## SLIDE 3 — Multimodal Encoder + Value

### Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Multimodal Encoder                                                │
│  Unifying multi-omics biology into actionable intelligence         │
│                                                                    │
│                                                                    │
│   RNA      ATAC    Protein   Phospho    VDJ                        │
│   ●        ●       ●         ◆          ▲                          │
│                                                                    │
│         ↓       ↓        ↓         ↓        ↓                      │
│                                                                    │
│         ┌─────────────────────────────────────┐                    │
│         │                                     │                    │
│         │      UNIFIED ENCODER                │                    │
│         │     256-D latent representation     │                    │
│         │                                     │                    │
│         └─────────────────────────────────────┘                    │
│                                                                    │
│                       ↓                                            │
│                                                                    │
│   Drug       Biomarker      Target           Patient               │
│   response   discovery      prioritization   stratification        │
│   prediction                                                       │
│                                                                    │
│  ●  Today    ◆  Phase 1    ▲  Phase 2                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Critical Design Choices

- **No left/right split.** Top-down architecture flow uses the full slide width.
- Modality symbols are large + bold, color-coded by phase (cyan/lavender/amber for available/Phase 1/Phase 2)
- Encoder is a single horizontal bar (full-width-ish), NOT a centered box
- Value outputs at bottom are 4 simple text columns (no cards, no icons beyond small geometric markers)
- Legend at bottom-left explains the symbol/color code

### Specifications

**Title block** (0.6"-1.6"): Same pattern.

**Modality row** (2.0"-2.9"):

5 modalities in a single horizontal row, evenly spaced.
For each:
- Modality name (18pt Calibri Bold, color-coded by phase, centered)
- Below name (0.15" gap): symbol (geometric shape, ~0.35" diameter, color-coded)
  - Circle (●): cyan = "available today"
  - Diamond (◆): lavender = "Phase 1"
  - Triangle (▲): amber = "Phase 2"

Modality positions (horizontal centers):
- RNA: x=2.0"
- ATAC: x=4.4"
- Protein: x=6.7"
- Phospho: x=9.0"
- VDJ: x=11.3"

Modality colors (text + symbol):
- RNA: ACCENT_CYAN, circle symbol
- ATAC: ACCENT_CYAN, circle symbol
- Protein: ACCENT_CYAN, circle symbol
- Phospho: ACCENT_LAVENDER, diamond symbol
- VDJ: ACCENT_AMBER, triangle symbol

**Convergence arrows** (3.0"-3.5"):

5 thin arrows pointing down from each modality symbol toward the encoder. All FG_SECONDARY, 1pt, small arrowheads.

**Encoder bar** (3.7"-4.6"):

Single horizontal bar:
- Position: x=1.5", y=3.7", width=10.3", height=0.9"
- Shape: rounded rectangle (corner radius 0.06")
- Fill: ACCENT_LAVENDER at 15% opacity
- Border: ACCENT_LAVENDER, 1.5pt, full saturation
- Inside, two text lines centered:
  - Line 1: "UNIFIED ENCODER" (22pt Bold White)
  - Line 2: "256-D latent representation" (14pt Regular Italic FG_SECONDARY)

**Single down arrow** (4.7"-5.2"):

One thick arrow from encoder center pointing down (full saturation lavender, 4pt, large head). Indicates "this all flows into value below."

**Value row** (5.4"-6.3"):

4 value outputs in a single horizontal row, evenly spaced. NO cards. Pure typography.
For each:
- Value name (16pt Calibri Bold White, centered, 2-line wrap allowed)
- Below name (no gap): single supporting line (12pt FG_SECONDARY, centered, italic)

Value positions:
- Drug response prediction: x=1.5", with subtitle "Predict combinations"
- Biomarker discovery: x=4.5", with subtitle "Stratify patients"
- Target prioritization: x=7.5", with subtitle "Rank by causal evidence"
- Patient stratification: x=10.5", with subtitle "Match to interventions"

**Legend strip** (6.7"-7.0"):

Single line at bottom-left, 11pt Calibri Regular:
- "● Today    ◆ Phase 1    ▲ Phase 2"
- Symbols inline with phase colors, text in FG_SECONDARY

### What's CUT From v1 Slide 3

- ❌ 2x2 value card grid with icon placeholders
- ❌ "ARCHITECTURE" + "VALUE TO PARTNERS" section headers
- ❌ "Designed to scale across datasets..." footer
- ❌ 3 output boxes ("Biological State / Perturbation Response / Causal Inference")
- ❌ Diamond emoji placeholder icons

---

## SLIDE 4 — Roadmap & Inflection Points

### Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Roadmap                                                           │
│  Execution plan and value compounding                              │
│                                                                    │
│                                                                    │
│    ●━━━━━━━━●━━━━━━━━●━━━━━━━━●━━━━━━━━●                          │
│  2025       Q3'26     2027      2027-28   2028+                    │
│                                                                    │
│  Public     Phase 1   Phase 2    Scaled    Therapeutic            │
│  benchmark  data      CRISPR     data      discovery              │
│             lands     + VDJ      gen                              │
│                                                                    │
│                                                                    │
│                                                                    │
│  Inflection points                                                 │
│                                                                    │
│  ●                ◆                ◆                ▲              │
│  Proprietary      Perturbation     Causal           Therapeutic    │
│  multimodal       scale            validation       discovery      │
│  data             expansion                         enablement     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Critical Design Choices

- Top half: horizontal milestone timeline (5 dots on a line, with year + outcome below each)
- Bottom half: inflection points as 4 text columns (no cards)
- Whitespace dominates middle of slide (between roadmap and inflection points) — intentional breathing room

### Specifications

**Title block** (0.6"-1.6"): Same pattern. Title is just "Roadmap" — single word for impact.

**Roadmap timeline** (2.2"-3.5"):

Horizontal line spanning slide width:
- Line: y=2.5", from x=1.5" to x=11.8", BORDER_SUBTLE color, 2.5pt
- 5 dots evenly spaced on the line:
  - Dot 1: x=1.5" (NOW/2025) — cyan
  - Dot 2: x=4.0" (Q3'26) — lavender
  - Dot 3: x=6.6" (2027) — amber
  - Dot 4: x=9.2" (2027-28) — amber
  - Dot 5: x=11.8" (2028+) — green
- Dot diameter: 0.22"

**Below each dot (3.0"-3.5"):**

Year label (14pt Calibri Bold, dot color, centered):
- 2025
- Q3'26
- 2027
- 2027-28
- 2028+

Below year label (0.2" gap), outcome label (15pt Calibri Regular White, centered, 2-line wrap):
- Public benchmark
- Phase 1 data lands
- Phase 2 CRISPR + VDJ
- Scaled data generation
- Therapeutic discovery

**Whitespace zone** (3.6"-4.8"): Empty. Intentional breathing room. **Do not fill.**

**Inflection points section** (5.0"-6.8"):

Section header at x=0.8", y=5.0":
- "Inflection points" (22pt Calibri Bold, ACCENT_GREEN, letter-spaced)

Below header (0.5" gap, y=5.6"), 4 text columns evenly spaced. NO cards.
For each:
- Symbol at top (geometric shape, ~0.32" diameter, color-coded)
- Below symbol (0.2" gap): inflection name (16pt Bold White, centered, 2-line wrap)
- No subtitle line

Inflection positions:
- x=2.0": ● (cyan circle) · "Proprietary multimodal data"
- x=5.0": ◆ (lavender diamond) · "Perturbation scale expansion"
- x=8.0": ◆ (lavender diamond) · "Causal validation"
- x=11.0": ▲ (amber triangle) · "Therapeutic discovery enablement"

**Footer** (6.9"-7.1"):

Optional single italic line at bottom center, 12pt FG_MUTED:
*"Value compounds through proprietary data, causal learning, multimodal intelligence."*

Per design rule "kill if it adds nothing": **omit this footer in v3**. Whitespace closes the slide.

### What's CUT From v1 Slide 4

- ❌ 5 inflection cards with borders (replaced with text-only columns)
- ❌ Detailed outcome descriptions under each milestone
- ❌ "KEY INFLECTION POINTS" all-caps label (now "Inflection points" sentence case)
- ❌ Italic footer line

---

## Cross-Slide Visual Language (Locked)

All 4 slides share:

1. **Same title position + size** (top-left, 44pt Calibri Bold White, subtitle 18pt italic FG_SECONDARY directly below)
2. **Same color palette + semantics** (cyan=Now/available, lavender=Phase 1, amber=Phase 2, green=value/output)
3. **Same symbol vocabulary** (● circle = today, ◆ diamond = Phase 1, ▲ triangle = Phase 2)
4. **No card containers anywhere** — typography + color + whitespace only
5. **Same dark navy background** (BG_DARK)
6. **No footer page numbers** (intentional — keep clean)
7. **No icons beyond geometric symbols** (no emoji, no SVG, no SmartArt icons)

---

## Implementation Notes (python-pptx specifics)

Continue using existing helper functions from v1/v2 build scripts. Add new helpers:

```python
def add_thick_accent_line(slide, x, y, length, color, thickness_pt=0.18):
    """Slide 1 top accent line above each phase column."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y),
        Inches(length), Inches(thickness_pt)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()  # no border
    return shape

def add_modality_symbol(slide, x, y, symbol_type, color, size_in=0.35):
    """Slide 3 modality row + Slide 4 inflection symbols."""
    if symbol_type == "circle":
        mso_shape = MSO_SHAPE.OVAL
    elif symbol_type == "diamond":
        mso_shape = MSO_SHAPE.DIAMOND
    elif symbol_type == "triangle":
        mso_shape = MSO_SHAPE.ISOCELES_TRIANGLE
    
    shape = slide.shapes.add_shape(
        mso_shape,
        Inches(x - size_in/2), Inches(y - size_in/2),
        Inches(size_in), Inches(size_in)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_network_node(slide, cx, cy, diameter, fill_color, glow=False):
    """Slide 2 network nodes. Optionally with glow ring."""
    radius = diameter / 2
    if glow:
        # Outer concentric ring at 50% opacity
        glow_diameter = diameter * 1.4
        glow_radius = glow_diameter / 2
        glow_shape = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(cx - glow_radius), Inches(cy - glow_radius),
            Inches(glow_diameter), Inches(glow_diameter)
        )
        glow_shape.fill.background()
        glow_shape.line.color.rgb = fill_color
        glow_shape.line.width = Pt(1)
        # NOTE: python-pptx doesn't directly support opacity on lines; 
        # accept full-saturation glow ring as visual indicator
    
    node = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(cx - radius), Inches(cy - radius),
        Inches(diameter), Inches(diameter)
    )
    node.fill.solid()
    node.fill.fore_color.rgb = fill_color
    node.line.fill.background()
    return node

def add_network_edge(slide, x1, y1, x2, y2, color, thickness_pt=1.5, 
                     directed=False, opacity_pct=100):
    """Slide 2 network edges. Undirected (left) or directed (right)."""
    connector = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = color
    connector.line.width = Pt(thickness_pt)
    if directed:
        # Add arrowhead via XML
        line_elem = connector.line._get_or_add_ln()
        from lxml import etree
        from pptx.oxml.ns import qn
        tailEnd = line_elem.find(qn('a:tailEnd'))
        if tailEnd is None:
            tailEnd = etree.SubElement(line_elem, qn('a:tailEnd'))
        tailEnd.set('type', 'triangle')
        tailEnd.set('w', 'sm')
        tailEnd.set('len', 'sm')
    # Opacity via solidFill alpha (XML hack)
    if opacity_pct < 100:
        # apply alpha to line fill — exact implementation depends on Cowork's existing pattern
        pass
    return connector
```

### Network layout for Slide 2

Suggested 9-node positions (relative to network center cx, cy with 1.3" radius):

```python
# Roughly hex-cluster, slightly organic offsets
NETWORK_NODES = [
    (0.0, -1.2),    # top
    (1.05, -0.6),   # top-right
    (1.15, 0.55),   # bottom-right
    (0.4, 1.15),    # bottom-right-low
    (-0.4, 1.2),    # bottom-left-low
    (-1.1, 0.6),    # bottom-left
    (-1.15, -0.55), # top-left
    (-0.05, 0.1),   # center
    (0.55, -0.35),  # mid-right (offset)
]
```

Edge list (12-15 connections, no duplicates):
```python
NETWORK_EDGES_LEFT = [
    (0,1), (0,6), (0,7),       # top connects to neighbors + center
    (1,2), (1,7), (1,8),
    (2,3), (2,7),
    (3,4), (3,7),
    (4,5), (4,7),
    (5,6), (5,7),
    (6,7), (7,8),               # center has many connections
]
```

For right (directional) network, same edge structure but:
- Source node = node 0 (top)
- 3 thick arrows: 0→1, 0→7, 0→6 (immediate neighbors of source)
- 4 medium arrows: 1→2, 6→5, 7→8, 7→3 (second hop)
- Remaining: thin

---

## Acceptance Criteria

### Mechanical (Cowork verifies before declaring done)

- ✓ 4 slides in output pptx
- ✓ 16:9 widescreen
- ✓ BG_DARK background on all slides
- ✓ NO embedded images (`unzip -l v3.pptx | grep ppt/media/` empty)
- ✓ Calibri font throughout
- ✓ File size < 200 KB
- ✓ Opens cleanly in PowerPoint Mac

### Layout (visual verification via LibreOffice render)

- ✓ All 4 slides have consistent title-block treatment (top-left, 44pt bold)
- ✓ No card containers anywhere — only typography + color + accent lines + dots/networks
- ✓ Color discipline consistent (cyan=Now, lavender=Phase 1, amber=Phase 2, green=value)
- ✓ Symbol vocabulary consistent across slides (●◆▲)
- ✓ Whitespace generous on every slide — no slide feels cramped

### 10-Second Comprehension Test (Per Slide)

For each slide, imagine showing for 10 seconds:
- **Slide 1**: "AIVC is a platform that progressed from public benchmarking → proprietary data (Phase 1) → scaled discovery (Phase 2)"
- **Slide 2**: "AIVC first learns biological structure, then learns how perturbations flow through it"
- **Slide 3**: "AIVC fuses 5 modalities into a unified encoder that powers 4 commercial applications"
- **Slide 4**: "AIVC's roadmap has 5 milestones from 2025 to 2028+, with 4 inflection points along the way"

If a slide doesn't pass — iterate Slide N only before extending.

---

## Risks To Flag In Cowork Prep Output

1. **No-card aesthetic risks looking "unfinished"** — without borders/containers, the slides may read as "draft" not "designed." Mitigation: aggressive whitespace + strong typography + accent lines must do the work of structure. If it reads as draft, the accent lines + typography weight need to bump up further (not add containers).

2. **Network glow on Slide 2** — python-pptx doesn't support opacity on shape lines directly. Workaround: use a full-saturation outer ring instead of true glow. If visual reads as "two concentric circles" instead of "glowing focal node," replace with a brighter fill on the source node OR add a subtle radial gradient (via XML manipulation).

3. **Modality symbols on Slide 3 + 4 must look identical** — same symbol vocabulary across slides. Use the `add_modality_symbol()` helper consistently. Verify visually that Slide 3's ◆ matches Slide 4's ◆ exactly (size, fill, position relative to label).

4. **Long single words on Slide 4** ("Proprietary multimodal data") may force 2-line wrapping at 16pt. Adjust positioning so labels can wrap without overlapping neighbors. If still tight, drop to 15pt.

5. **Whitespace zone on Slide 4 (3.6"-4.8")** — this is intentional empty space. Reviewers may flag it as "missing content." Per design principle, hold the line. If CEO objects in review, we add a single line of text or KPI band in this zone in v4.

6. **Slide 1 accent line above each phase** — thin 0.18" rectangle. Must render crisply. If it looks pixelated or interrupted, increase to 0.22" or add a subtle gradient.

7. **Phase column gutters on Slide 1 (0.5")** — must be visually generous. If columns feel cramped, increase gutter to 0.7" and reduce column width to 3.4".

8. **No commit until Ash approves** — same pattern as v1/v2. Cowork ships pptx + build script, leaves working tree clean. Ash spot-checks on Mac PowerPoint, then commits.

---

## What's Out Of Scope For v3

- Per-card icon differentiation (geometric symbols only — no pills, targets, DNA helices)
- Animation / transitions
- Speaker notes (investor decks don't use them; CEO presents verbally)
- Image-based icons (still no flattened images)
- Master slide template
- Auto-export to PDF
- v1 polish pass (Ash explicitly decided: v1 stays as-is for CEO comparison)

---

## After v3 Lands — Review Sequence

1. **Cowork prep output** → Ash reviews structure + content
2. **Ash spot-checks on Mac PowerPoint** → confirms editability + 10-second test per slide
3. **If approved** → commit + push to origin/main, send to CEO with cover note
4. **If issues** → identify which slide(s) failed, iterate slide-by-slide rather than whole deck

For CEO comparison:
- v1: `aivc_investor_4slide_v1.pptx` (already on origin/main at commit `8b7a626`)
- v3: `aivc_investor_4slide_v3.pptx` (this deliverable)

CEO opens both, picks direction or asks for v4 with hybrid.

---

## Honest Risks I'm Tracking

1. **Apple-keynote minimal may feel too sparse for a biotech VC audience.** VC audiences for technical biotech expect some visual density to read as "substantive." If v3 feels like a startup pitch deck rather than a technical platform deck, we'll need to add back selective density in v4 (one element per slide, with explicit purpose).

2. **No hero numbers means the "73% cross-corpus" + "5 donors" + "20-25 donors" credibility anchors are buried in supporting text.** If CEO/VC says "where's the proof?" we add hero numbers in v4 — but per Ash's locked decision, v3 keeps them in supporting text.

3. **The "we are here" indicator** (active phase outer ring on PHASE 1 dot in Slide 1) is the only "you are here" cue across all 4 slides. If CEO wants stronger temporal grounding ("we're 80% through PHASE 1, not just starting"), that's a v4 add.

4. **No animation/transitions** means the deck is fully static. For live VC presentation, this is fine. For asynchronous send, OK. If CEO wants build-up animations on the progression line or value cards, that's a separate effort.

5. **Phase 3 dimmed treatment** — consistent across Slides 1 + 4. If Phase 3 is more strategically important than v3 implies (e.g., post-Series-A clinical pipeline), promote it to a brighter color OR give it more textual space.

6. **No legal/IP markings** — no copyright footer, no "confidential" watermark, no draft stamp. Per design principle of cleanliness. CEO can add these manually in PowerPoint if needed for distribution.
