# Slide 1 v2 — Pilot Build Spec

**Owner**: Cowork (execution)
**Estimated time**: 1-1.5 hours
**Production path**: Path A — python-pptx native shapes (continues from v1)
**Input source**: Kinga CEO brief + v1 (`8b7a626`) + Ash feedback on v1 (need stronger hierarchy, less text density, premium feel)
**Output**: `docs/deck/exports/aivc_investor_4slide_v2.pptx` (Slide 1 only — slides 2-4 stay as v1 placeholders or omitted for this pilot iteration)

---

## Design Principles (Non-Negotiable)

These constrain every implementation decision. Cowork pushes back if a design choice violates these:

1. **10-second comprehension test** — A VC glancing at the slide must extract "what AIVC is, where it's today, where it's going" within 10 seconds. Anything that doesn't serve this is cut.

2. **Two-layer reading model**:
   - **Layer 1 (10-second scan)**: panel headers + identity tagline + progression line
   - **Layer 2 (30-second lean-in)**: supporting content (3-4 lines max per panel)
   - Typography hierarchy must make Layer 1 dominate visually

3. **No hero numbers per panel** (Ash decision — Option A pure structure)

4. **No competing architecture pattern** (no INPUT/CORE/OUTPUT framing on this slide — save that for slides where it actually fits content)

5. **Aggressive whitespace**. Premium ≠ ornate. If in doubt, delete.

6. **Color discipline**: cyan = Now, lavender = Phase 1, amber = Phase 2. No other colors except WHITE for primary text, SECONDARY for support, DIMMED for Phase 3.

---

## Slide Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  TITLE (large, bold, white)                                         │
│  Subtitle (smaller, italic, secondary)                              │
│                                                                     │
│  ───●─────────────────●─────────────────●─────────·─→               │
│   NOW             PHASE 1            PHASE 2     Phase 3            │
│  (cyan)          (lavender)          (amber)    (dimmed)            │
│                                                                     │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐    │
│  │                 │                 │                         │    │
│  │  FOUNDATION     │  CONTROLLED     │  SCALABLE CAUSAL        │    │
│  │  & BENCHMARKING │  PERTURBATION   │  DISCOVERY              │    │
│  │                 │  LEARNING       │                         │    │
│  │                 │                 │                         │    │
│  │  Public         │  QuRIE-seq      │  + CRISPR + VDJ         │    │
│  │  multimodal     │  proprietary    │                         │    │
│  │  datasets       │                 │  5 modalities           │    │
│  │                 │  3 modalities   │  20-25 donors           │    │
│  │  Pretrained     │  5 donors       │  Soft + hard            │    │
│  │  encoder        │  5 stimuli      │  perturbations          │    │
│  │                 │  10 inhibitors  │                         │    │
│  │  73% cross-     │                 │  CRISPR screening       │    │
│  │  corpus         │  BTK + JAK      │  library                │    │
│  │  validation     │  demo           │                         │    │
│  │                 │                 │                         │    │
│  │  ─────────      │  ─────────      │  ─────────              │    │
│  │  Validated      │  Causal         │  Cross-state            │    │
│  │  substrate      │  learning       │  reasoning              │    │
│  │                 │                 │                         │    │
│  └─────────────────┴─────────────────┴─────────────────────────┘    │
│                                                                     │
│  PHASE 3 ─── Continuation at scale + therapeutic pipeline           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layout Specifications (Precise)

Slide dimensions: 13.333" × 7.5" (16:9 widescreen, same as v1).

Margins: 0.6" all sides → usable area 12.13" × 6.3".

### Vertical zones (top to bottom)

| Zone | Top | Height | Purpose |
|---|---|---|---|
| Title block | 0.6" | 0.9" | Title + subtitle |
| Progression line | 1.7" | 0.6" | Horizontal line + 4 phase dots + dot labels |
| Panel cards | 2.5" | 4.0" | 3 panel cards side-by-side |
| Phase 3 footer | 6.7" | 0.4" | Single italic line + arrow |

### Horizontal zones (left to right)

3 equal panels, 0.3" gutter between:
- Panel 1 (NOW): left=0.6", width=3.84"
- Gutter: 0.3"
- Panel 2 (PHASE 1): left=4.74", width=3.84"
- Gutter: 0.3"
- Panel 3 (PHASE 2): left=8.88", width=3.84"

---

## TITLE BLOCK (0.6"-1.5")

**Title**: "AIVC Platform Evolution"
- Position: left=0.6", top=0.6", width=12.13", height=0.55"
- Font: Calibri Bold, 40pt
- Color: FG_PRIMARY (white)
- Alignment: left

**Subtitle**: "From public benchmarking to scalable causal biological intelligence"
- Position: left=0.6", top=1.1", width=12.13", height=0.35"
- Font: Calibri Regular Italic, 16pt
- Color: FG_SECONDARY (#A0AFC8)
- Alignment: left

---

## PROGRESSION LINE (1.7"-2.3")

This is the load-bearing element for the temporal narrative. Build it carefully.

### Horizontal line

- Single horizontal line spanning panel centers
- Position: y=1.95" (vertical center of the progression zone)
- Start x: panel 1 horizontal center (= 0.6" + 3.84"/2 = 2.52")
- End x: extends 0.4" past Phase 3 dot position (continuation indicator)
- Line: BORDER_SUBTLE color (#2D3A57), 1.5pt thickness

### Phase dots — 4 dots total (3 active + 1 Phase 3 dimmed)

Dot positions (horizontal centers):
- Dot 1 (NOW): x = 2.52" (panel 1 center)
- Dot 2 (PHASE 1): x = 6.66" (panel 2 center)
- Dot 3 (PHASE 2): x = 10.80" (panel 3 center)
- Dot 4 (Phase 3): x = 12.20" (0.4" past Phase 2, before slide edge)

Dot specifications:
- Diameter: 0.16" (large enough to read, small enough to not dominate)
- Y-center: 1.95" (aligned with line)
- Dot 1: cyan fill (ACCENT_CYAN), no border
- Dot 2: lavender fill (ACCENT_LAVENDER), no border
- Dot 3: amber fill (ACCENT_AMBER), no border
- Dot 4: BORDER_SUBTLE fill, dashed white border 1pt (signals "future / continuation")

### Line styling between dots

- Between dots 1-2-3: solid line, BORDER_SUBTLE color
- Between dot 3 and dot 4: **dashed** line, same color (signals discontinuity in time/funding)
- After dot 4: thin arrow indicator (→), BORDER_SUBTLE color, 12pt

### Dot labels (BELOW dots)

Each label positioned 0.15" below its dot, horizontally centered on dot.

- "NOW": 12pt Calibri Bold, ACCENT_CYAN, letter-spaced 1pt
- "PHASE 1": 12pt Calibri Bold, ACCENT_LAVENDER, letter-spaced 1pt
- "PHASE 2": 12pt Calibri Bold, ACCENT_AMBER, letter-spaced 1pt
- "Phase 3": 11pt Calibri Regular Italic, FG_SECONDARY (NOT bold, NOT uppercase — visual indicator that it's tentative/future)

---

## PANEL CARDS (2.5"-6.5")

3 equal panel cards. Same structure for all three (color varies).

### Card container

Each card:
- Rounded rectangle, corner radius 0.08"
- Fill: BG_DARK (matches slide background — card is delineated by border only, not fill)
- Border: panel color (cyan / lavender / amber), 1.5pt
- Border opacity: 60% (subtle, not loud)
- No drop shadow

### Card internal padding

- Top padding: 0.3"
- Left/right padding: 0.3"
- Bottom padding: 0.3"

### Card content structure (top to bottom)

**Block 1 — Header** (top of card):
- Section name in uppercase, 2 lines max
- Font: Calibri Bold, 18pt
- Color: panel color (cyan/lavender/amber)
- Line spacing: tight (1.0)
- Letter-spacing: slight (+0.5pt) for premium feel

**Block 2 — Identity line** (single line below header, after 0.2" gap):
- The single most-compact identity statement
- Font: Calibri Regular, 14pt
- Color: FG_PRIMARY (white)

**Block 3 — Supporting details** (3-5 lines, after 0.25" gap):
- Stacked key-value or compact statements
- Font: Calibri Regular, 13pt
- Color: FG_PRIMARY (white) for key terms, FG_SECONDARY for supporting context
- Line spacing: 1.4 (generous, breathing room)

**Block 4 — Divider** (subtle horizontal line, after 0.3" gap):
- Width: 1.5" centered horizontally within card
- Color: panel color at 30% opacity
- Thickness: 1pt

**Block 5 — Tagline** (single line, italic, after 0.2" gap from divider):
- 3-word tagline summarizing the panel's essence
- Font: Calibri Regular Italic, 12pt
- Color: FG_SECONDARY
- Centered within card

---

### Panel 1 — NOW (cyan accent)

**Header (Block 1)**:
```
FOUNDATION
& BENCHMARKING
```
(2 lines, line break between FOUNDATION and & BENCHMARKING)

**Identity line (Block 2)**:
"Public multimodal datasets"

**Supporting details (Block 3)** — 3 lines:
```
3 reference papers
Pretrained encoder
73% cross-corpus validation
```

**Tagline (Block 5)**:
"Validated foundation"

---

### Panel 2 — PHASE 1 (lavender accent)

**Header (Block 1)**:
```
CONTROLLED
PERTURBATION
LEARNING
```
(3 lines — slightly more text in header is OK; this is the central narrative panel)

**Identity line (Block 2)**:
"QuRIE-seq · proprietary multi-omics"

**Supporting details (Block 3)** — 4 lines:
```
3 modalities (RNA · Protein · Phospho)
5 donors · 5 timepoints
5 stimuli · 10 inhibitors
BTK + JAK headline demo
```

**Tagline (Block 5)**:
"Causal learning, in motion"

---

### Panel 3 — PHASE 2 (amber accent)

**Header (Block 1)**:
```
SCALABLE
CAUSAL DISCOVERY
```
(2 lines)

**Identity line (Block 2)**:
"+ CRISPR + VDJ"

**Supporting details (Block 3)** — 4 lines:
```
5 modalities (+ ATAC · VDJ)
20–25 donors
Soft + hard perturbations
CRISPR screening library
```

**Tagline (Block 5)**:
"Cross-state reasoning"

---

## PHASE 3 FOOTER (6.7"-7.1")

Single line, centered horizontally on slide:

"PHASE 3 ─── Continuation at scale + therapeutic pipeline →"

- Font: Calibri Regular Italic, 13pt
- Color: FG_SECONDARY
- The "PHASE 3" prefix in bold (Calibri Bold, same size/color)
- The arrow `→` is a Unicode character or geometric shape
- Position: y=6.85" (vertical center of footer zone), horizontally centered

(Note: this is intentionally low-prominence — Phase 3 lives in the progression line dot above; this footer is the verbal expansion only.)

---

## Color Palette (Continued From v1)

```python
BG_DARK         = RGBColor(0x0A, 0x0E, 0x1A)
FG_PRIMARY      = RGBColor(0xFF, 0xFF, 0xFF)
FG_SECONDARY    = RGBColor(0xA0, 0xAF, 0xC8)
ACCENT_CYAN     = RGBColor(0x26, 0xDD, 0xF9)
ACCENT_LAVENDER = RGBColor(0x8B, 0x5C, 0xF6)
ACCENT_AMBER    = RGBColor(0xF5, 0x9E, 0x0B)
BORDER_SUBTLE   = RGBColor(0x2D, 0x3A, 0x57)
```

**One addition to palette for v2**:
```python
DIMMED          = RGBColor(0x60, 0x70, 0x88)  # Phase 3 dot border + dimmed elements
```

---

## Typography Hierarchy (Locked)

| Element | Font | Size | Weight | Color |
|---|---|---|---|---|
| Slide title | Calibri | 40pt | Bold | FG_PRIMARY |
| Slide subtitle | Calibri | 16pt | Regular Italic | FG_SECONDARY |
| Phase dot label (active) | Calibri | 12pt | Bold | Panel color |
| Phase 3 dot label | Calibri | 11pt | Regular Italic | FG_SECONDARY |
| Panel header | Calibri | 18pt | Bold | Panel color |
| Panel identity line | Calibri | 14pt | Regular | FG_PRIMARY |
| Panel supporting | Calibri | 13pt | Regular | FG_PRIMARY |
| Panel tagline | Calibri | 12pt | Regular Italic | FG_SECONDARY |
| Phase 3 footer | Calibri | 13pt | Italic (PHASE 3 bold) | FG_SECONDARY |

---

## What's EXPLICITLY CUT From v1

These v1 elements DO NOT carry forward. Cowork must not re-add:

- ❌ Modality icon row (RNA/ATAC/Protein circles in Panel 1)
- ❌ Perturbation arrow diagrams (Panel 2 + Panel 3 visuals)
- ❌ Encoder shape (hexagon/rectangle in panels)
- ❌ "BTK + JAK combo" pill emphasis
- ❌ Detailed italic descriptions at panel bottoms
- ❌ Phase 3 horizontal divider line above footer
- ❌ Subtitle dates ("Q3 2026", "2027") inside panel headers — these are implied by phase position, not stated
- ❌ Sub-line under encoder card

---

## Build Script Structure

Continue from v1's existing build script. Add new function `build_slide1_v2(prs)`. Keep v1's `build_slide1_evolution(prs)` available for comparison if needed but use only v2 for this output.

Recommended structure:
```python
def build_slide1_v2(prs):
    """v2 pilot — Option A + minimal progression line, 10-second comprehension priority."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    set_slide_bg(slide, BG_DARK)
    
    # Title block
    add_title(slide, ...)
    add_subtitle(slide, ...)
    
    # Progression line + 4 dots + labels
    add_progression_line(slide, ...)
    add_phase_dot(slide, x=2.52, color=ACCENT_CYAN, label="NOW", ...)
    add_phase_dot(slide, x=6.66, color=ACCENT_LAVENDER, label="PHASE 1", ...)
    add_phase_dot(slide, x=10.80, color=ACCENT_AMBER, label="PHASE 2", ...)
    add_phase_dot_dimmed(slide, x=12.20, label="Phase 3", ...)
    
    # 3 panel cards
    build_panel(slide, x=0.6, color=ACCENT_CYAN, ...)  # NOW
    build_panel(slide, x=4.74, color=ACCENT_LAVENDER, ...)  # PHASE 1
    build_panel(slide, x=8.88, color=ACCENT_AMBER, ...)  # PHASE 2
    
    # Footer
    add_phase3_footer(slide, ...)
    
    return slide

def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    build_slide1_v2(prs)
    prs.save("docs/deck/exports/aivc_investor_4slide_v2.pptx")
```

For this pilot, **output is single-slide pptx** with only Slide 1 v2. After approval, slides 2/3/4 get rebuilt with same design system.

---

## Acceptance Criteria

### Mechanical (Cowork verifies before declaring done)

- ✓ 1 slide only in output pptx
- ✓ 16:9 widescreen (13.333" × 7.5")
- ✓ BG_DARK background
- ✓ NO embedded images (`unzip -l output.pptx | grep ppt/media/` returns empty)
- ✓ All visual elements are native PowerPoint shapes
- ✓ Calibri font throughout
- ✓ File size < 100 KB
- ✓ Opens cleanly in PowerPoint Mac

### Layout (visual verification via LibreOffice render)

- ✓ No text overlap anywhere
- ✓ Progression line + dots clearly visible
- ✓ 3 panel cards aligned horizontally with consistent height
- ✓ Color discipline: cyan/lavender/amber strictly per phase
- ✓ Phase 3 dot is visibly dimmer than active phase dots
- ✓ Dashed line between Phase 2 and Phase 3 dots renders correctly
- ✓ Panel headers wrap to 2-3 lines as specified (not 1 line, not 4+)
- ✓ Bottom of panels has consistent baseline (taglines align)
- ✓ Whitespace is generous — slide does NOT feel cramped

### 10-second comprehension test (subjective, Cowork judges)

Imagine showing this to someone for 10 seconds, then asking: "What does AIVC do, and what's the progression?"

If the answer requires reading panel detail bullets, the design failed Layer 1. Iterate.

If headers + identity lines + progression line carry the answer, the design succeeded.

---

## Risks To Flag In Cowork Prep Output

1. **Tagline italic quality at 12pt** — may look small or weak. If so, bump to 13pt italic OR cut the divider+tagline entirely and let supporting details breathe with extra padding.

2. **Phase 3 dimmed dot** — needs to look like "future" not "broken." If the dashed border doesn't read well at slide-fill scale, alternative: dot with 50% opacity fill + no border.

3. **Panel header line breaks** — "CONTROLLED / PERTURBATION / LEARNING" on 3 lines may make Panel 2 visually taller than Panels 1 + 3. Compensate by adjusting Block 3 supporting detail count, OR force Panels 1 + 3 to match header height with extra padding.

4. **Letter-spacing** in python-pptx is non-trivial (no direct API). Use character-by-character spacing via XML manipulation if needed, OR drop letter-spacing if it complicates the build — typography hierarchy doesn't depend on it.

5. **Dashed line in python-pptx** — `line.dash_style = MSO_LINE_DASH_STYLE.DASH` (or similar). Verify this renders correctly in PowerPoint Mac (some dash styles look different in PowerPoint vs LibreOffice preview).

6. **Phase 3 footer arrow** — `→` Unicode renders in Calibri on Mac, but verify. If not, use a geometric shape (small triangle pointing right) instead.

---

## What's Out Of Scope For This Pilot

- Slides 2, 3, 4 — pilot is Slide 1 only. After approval, design system extends.
- Animation / transitions
- Speaker notes
- Master slide template
- Image-based icons (still no flattened images allowed)

---

## After Slide 1 v2 Lands

1. **Ash spot-checks on Mac**: opens pptx in PowerPoint Mac, verifies 10-second comprehension test passes, checks editability of all elements (click any element → Shape Format tab, not Picture)
2. **If approved** → I draft v3 spec covering slides 2/3/4 using the validated v2 design system
3. **If not approved** → identify which design decision failed (typography? color? layout? cuts too aggressive?) and iterate Slide 1 only before extending

---

## Honest Risks I'm Tracking

1. **Risk of being "too minimal"** — 10-second comprehension priority means we cut deep. CEO may want more "richness." If so, we add back selectively in v3 with explicit purpose for each addition.

2. **Risk of three plain boxes** — even with the progression line, panel cards may still read as static. Mitigation: typography hierarchy + tagline italics + color discipline. If still flat, next iteration explores subtle visual weight (Panel 2 slightly larger / brighter as "we are here" indicator).

3. **No hero numbers means no "WOW moment"** — VCs love specific numbers. Per Ash decision we don't have them on this slide. If CEO/VC feedback says "where's the 73%, where's the donor count" we may need a v3 that adds *one* hero metric per panel.

4. **Phase 3 minimization** — Phase 3 lives in dot + footer line only. If CEO says "Phase 3 is too buried," next iteration adds a fourth small card to the right OR strengthens the footer.

5. **No "demo" or "validation" callouts in Phase 1 panel** — BTK + JAK demo is one of 4 bullet lines, not a hero. May be too understated for investor framing if BTK+JAK is the next-12-months proof point. Flag in v2 review.
