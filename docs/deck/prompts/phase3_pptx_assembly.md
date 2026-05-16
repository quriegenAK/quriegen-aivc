# Phase 3 — .pptx Assembly via Cowork

**Owner**: Cowork (execution)
**Estimated time**: 45-90 min
**Strategy**: Assemble single .pptx from 12 content specs + 12 SVGs (+ 12 PNG fallback)
**Goal**: investor-ready technical appendix that reads as a systems architecture document, not an oversized backup deck

---

## Context

Phase 1 (content specs) + Phase 2 (12 SVG diagrams + PNGs) are complete:

- **Content**: 12 markdown specs in `docs/deck/content/` — each contains headline, sub-headline, body bullets, source data table, speaker notes, investor framing
- **Visuals**: 12 SVGs + 12 PNG previews in `docs/deck/assets/diagrams/`
- **Style assets**: `docs/deck/assets/{color_palette,typography,icon_inventory}.md`

The .pptx assembly is the deliverable that goes into Kinga's primary deck as a technical appendix.

After this lands, Phase 4 (Claude Design polish on hero slides) is optional.

---

## Deck Structure (LOCKED — Ash decision)

**Total slides: 17** (1 cover + 4 section dividers + 12 content slides)

```
Slide 01  →  APPENDIX COVER
Slide 02  →  SECTION DIVIDER: A · ARCHITECTURE DEPTH
Slide 03  →  A1 — System Architecture
Slide 04  →  A2 — Multi-Omics Encoder
Slide 05  →  A3 — Decomposed Readout
Slide 06  →  A4 — Temporal Neural ODE
Slide 07  →  SECTION DIVIDER: B · VALIDATION EVIDENCE
Slide 08  →  B1 — Methodology Rigor
Slide 09  →  B2 — Encoder Probe Verdict
Slide 10  →  B3 — Synergy Pre-Demo
Slide 11  →  SECTION DIVIDER: C · QURIESEQ PHASE 1
Slide 12  →  C1 — Phase 1 Experimental Design
Slide 13  →  C2 — BTK+JAK Headline Demo
Slide 14  →  SECTION DIVIDER: D · ROADMAP + BUDGET
Slide 15  →  D1 — Quarterly Roadmap
Slide 16  →  D2 — Seed Allocation
Slide 17  →  SECTION DIVIDER: E · STRATEGIC HORIZON
Slide 18  →  E1 — 5-Year Trajectory
```

(Counting error in my header — final count is 18 slides. Update if you want a different cover handling.)

---

## Asset Strategy (LOCKED — Ash decision)

**SVG primary, PNG fallback** — embed SVGs as the master visual asset; PNGs serve as a safety net for renderers that can't handle SVG.

### How To Do This In python-pptx

`python-pptx` does NOT natively support SVG embedding (as of current versions). Two viable paths:

**Path A (recommended)**: Embed PNGs as the visual lead. SVGs ship alongside in the repo. PowerPoint slides display PNGs reliably across all platforms.

Why this is correct:
- "SVG master, PNG fallback" means SVG is the source of truth (regeneratable from build scripts), PNG is the rendered artifact embedded in the deck
- Kinga's existing deck uses raster images (per the 108MB file size)
- PowerPoint rendering of embedded SVG is inconsistent across versions and Macs vs Windows — risky for an investor deck
- PNGs render identically everywhere
- 250KB × 18 slides ≈ 4.5MB total — entirely acceptable

**Path B**: Embed SVG as an OOXML extension (PowerPoint 2016+). More complex, version-dependent.

**Decision: Path A.** SVG master files live in `docs/deck/assets/diagrams/*.svg` (always regeneratable from `_build_*.py`); PNGs embedded in the .pptx are the visual lead.

If at Phase 4 polish stage Design wants to swap to SVG embedding, that's a Phase 4 decision.

---

## Slide-By-Slide Build Spec

### Slide 01 — APPENDIX COVER

**Layout**: Full-slide cover, dark navy background (`#070A14`).

**Content**:

```
        APPENDIX

        AIVC GeneLink — Technical Appendix
        ────────────────────────────────

        Architecture Depth · Validation Evidence · QurieSeq Phase 1 · 
        Roadmap & Budget · Strategic Horizon


        Quriegen · May 2026
```

- Eyebrow: "APPENDIX" in cyan `#26DDF9`, 18pt, letter-spaced
- Title: "AIVC GeneLink — Technical Appendix" in white `#F7FAFF`, 48pt Inter Bold
- Subtitle: section names (A/B/C/D/E names) in muted `#A8B4C2`, 18pt Arial
- Footer: "Quriegen · May 2026" in cyan, 14pt

No diagrams on cover. Pure typography statement.

**Speaker note**: None (cover slide).

---

### Slide 02 — SECTION DIVIDER: A · ARCHITECTURE DEPTH

**Layout**: Section divider, dark navy background. Center-left aligned typography.

**Content**:

```
        SECTION A

        Architecture Depth
        ──────────────────

        How the model works · what's frozen, what trains, 
        what generalizes


        Slides A1–A4
```

- Eyebrow: "SECTION A" in cyan, 16pt
- Title: "Architecture Depth" in white, 56pt Inter Bold
- Sub: "How the model works · what's frozen, what trains, what generalizes" in muted, 20pt Arial
- Footer: "Slides A1–A4" in muted, 14pt

**Speaker note**: None (divider slide).

---

### Slides 03-06 — A1, A2, A3, A4 (Content Slides)

For each content slide, layout pattern:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Headline (from content spec)                                            │  ← top, 32pt
│  Sub-headline (from content spec)                                        │  ← below, 18pt muted
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │  [Diagram PNG embedded — 1920×1080 scaled to fit slide width]   │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Source citation (from content spec)                                     │  ← bottom small
└─────────────────────────────────────────────────────────────────────────┘
```

**Critical**: the diagram PNGs ALREADY include the slide title, sub-headline, body content, and source citation rendered into the SVG itself. Each PNG is a self-contained 1920×1080 slide visual.

**So the .pptx slide layout is simpler than it first appears**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  [Full diagram PNG fills the entire slide area]                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The PNG IS the slide. Just embed it as the full-slide image, with the slide background set to dark navy `#070A14` to match the PNG edges.

This is the cleanest approach because:
- The diagrams already include all text (headline, body, source citation)
- No font fallback issues (PNG is rasterized)
- No layout drift between SVG design and PowerPoint rendering
- One image per slide, perfect 1:1 mapping

**For each content slide**:
1. Set slide background to `#070A14`
2. Insert PNG at full slide dimensions (0,0 to slide width × height)
3. Add speaker notes (see below)

### Speaker Notes Content

For each content slide, populate the speaker notes from the corresponding content spec's "Speaker notes" section.

For example, A2's content spec has:
```
## Speaker notes

**If asked: "Why DOGMA-seq specifically?"**
> DOGMA-seq is the first published protocol...

**If asked: "How does it perform per cell type?"**
> Overall 73% across the major PBMC lineages...
```

This becomes A2's slide notes in the .pptx — formatted as the markdown-extracted Q&A pairs.

**Important**: Keep the Q&A format ("If asked: X" / "> Answer"). Don't reformat into bullet points. The presenter (Ash or Kinga) reads them as conversational prompts.

---

### Slides 07, 11, 14, 17 — Section Dividers (B, C, D, E)

Same pattern as Slide 02, customized per section:

**Slide 07 — Section B · Validation Evidence**
```
SECTION B

Validation Evidence
───────────────────

How we test — three datasets, pre-registered evals, 
no cherry-picking

Slides B1–B3
```

**Slide 11 — Section C · QurieSeq Phase 1**
```
SECTION C

QurieSeq Phase 1
────────────────

The proprietary data engine · 5 donors × 5 timepoints × 
4-arm × ~500K cells · BTK+JAK confirmed

Slides C1–C2
```

**Slide 14 — Section D · Roadmap + Budget**
```
SECTION D

Roadmap + Budget
────────────────

11 quarters · 5 stages · 2 drug pipelines · 
$10M seed allocation

Slides D1–D2
```

**Slide 17 — Section E · Strategic Horizon**
```
SECTION E

Strategic Horizon
─────────────────

From validated platform to first-in-class candidates · 
2026 → 2031

Slide E1
```

Each divider uses the same typography pattern as Slide 02 — eyebrow + title + sub + footer. No diagrams.

---

### Slides 08-10, 12-13, 15-16, 18 — Remaining Content Slides

Same pattern as A1-A4: PNG fills full slide, dark navy background, speaker notes from content spec.

Per-slide mapping:

| .pptx Slide | Content spec file | PNG file |
|---|---|---|
| 03 | A1_system_architecture.md | A1_system_architecture.png (note: file is `A1_system_architecture_preview.png` per repo convention — check the path) |
| 04 | A2_encoder_substrate.md | A2_encoder_evidence_preview.png |
| 05 | A3_decomposed_readout.md | A3_decomposed_readout_preview.png |
| 06 | A4_temporal_neural_ode.md | A4_temporal_dynamics_preview.png |
| 08 | B1_methodology_rigor.md | B1_three_datasets_methodology_preview.png |
| 09 | B2_encoder_probe_verdict.md | B2_adapter_verdict_preview.png |
| 10 | B3_synergy_pre_demo.md | B3_mechanism_pre_demo_preview.png |
| 12 | C1_phase1_design.md | C1_phase1_experimental_design_preview.png |
| 13 | C2_btk_jak_demo.md | C2_btk_jak_demo_plan_preview.png |
| 15 | D1_quarterly_roadmap.md | D1_quarterly_roadmap_preview.png |
| 16 | D2_seed_allocation.md | D2_seed_allocation_preview.png |
| 18 | E1_five_year_trajectory.md | E1_five_year_trajectory_preview.png |

---

## Implementation Guidance

### Recommended approach

Single build script: `docs/deck/exports/_build_appendix_pptx.py`

```python
#!/usr/bin/env python3
"""Build aivc_appendix_v1.pptx from content specs + PNGs."""

from pptx import Presentation
from pptx.util import Inches, Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTENT_DIR = REPO / "docs/deck/content"
DIAGRAMS_DIR = REPO / "docs/deck/assets/diagrams"
OUTPUT = REPO / "docs/deck/exports/aivc_appendix_v1.pptx"

# 16:9 widescreen — match SVG viewBox (1920×1080)
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

BG_DARK = RGBColor(0x07, 0x0A, 0x14)

# Speaker note extraction: parse markdown content spec, grab "## Speaker notes" section
def extract_speaker_notes(content_spec_path):
    text = content_spec_path.read_text()
    match = re.search(
        r"## Speaker notes.*?\n(.*?)(?=\n---|\n## |\Z)",
        text,
        re.DOTALL
    )
    return match.group(1).strip() if match else ""

# Build cover slide
def add_cover_slide(prs):
    layout = prs.slide_layouts[6]  # blank layout
    slide = prs.slides.add_slide(layout)
    # background
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK
    # text placement (eyebrow, title, sub, footer)
    # ... [add text boxes with proper positioning]
    return slide

# Build section divider
def add_section_divider(prs, section_letter, section_title, sub, slides_range):
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    # bg + text boxes
    return slide

# Build content slide
def add_content_slide(prs, png_path, speaker_notes):
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    # background
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK
    # full-slide PNG
    slide.shapes.add_picture(
        str(png_path),
        left=0, top=0,
        width=SLIDE_WIDTH, height=SLIDE_HEIGHT
    )
    # speaker notes
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = speaker_notes
    return slide

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    
    # Slide 01 — cover
    add_cover_slide(prs)
    
    # Slide 02 — A divider
    add_section_divider(prs, "A", "Architecture Depth",
                        "How the model works · what's frozen, what trains, what generalizes",
                        "A1–A4")
    
    # Slides 03-06 — A content
    for slide_id, png_name, spec_name in [
        ("A1", "A1_system_architecture_preview.png", "A1_system_architecture.md"),
        ("A2", "A2_encoder_evidence_preview.png", "A2_encoder_substrate.md"),
        ("A3", "A3_decomposed_readout_preview.png", "A3_decomposed_readout.md"),
        ("A4", "A4_temporal_dynamics_preview.png", "A4_temporal_neural_ode.md"),
    ]:
        notes = extract_speaker_notes(CONTENT_DIR / spec_name)
        add_content_slide(prs, DIAGRAMS_DIR / png_name, notes)
    
    # ... repeat for B/C/D/E sections + dividers
    
    prs.save(OUTPUT)
    print(f"Built {OUTPUT}")

if __name__ == "__main__":
    main()
```

The above is a skeleton — Cowork's call on exact implementation details (font handling, text positioning on cover/dividers, etc.).

### Critical PNG path note

The PNG filenames in the repo use the `_preview` suffix from the build scripts. Verify before referencing:

```bash
ls docs/deck/assets/diagrams/*_preview.png
```

Expected output: 12 files (one per content slide).

### Speaker notes formatting

Markdown formatting in content specs (bold, italic, blockquotes) doesn't translate to .pptx slide notes natively. Two options:

**Simple (recommended)**: Strip markdown syntax, convert to plain text. The Q&A structure is preserved as readable text:
```
If asked: "Why DOGMA-seq specifically?"

DOGMA-seq is the first published protocol for measuring RNA, chromatin accessibility, and surface protein from the same single cell...
```

**Rich**: Use python-pptx's run-level formatting to apply bold to questions, regular to answers. More work, marginal benefit since speaker notes are typically read in presenter mode where formatting is minimal.

Go with simple.

---

## Acceptance Criteria

When Cowork ships, verify:

1. ✅ **18 slides total** in `aivc_appendix_v1.pptx` (1 cover + 4 dividers + 12 content + 1 final divider — recount)
2. ✅ **Slide aspect ratio**: 16:9 widescreen, 13.333" × 7.5" (matches Kinga's deck format)
3. ✅ **Background**: dark navy `#070A14` on every slide
4. ✅ **Cover slide**: clean typography, no diagrams
5. ✅ **Section dividers**: 5 of them (A/B/C/D/E), each with eyebrow + title + sub + footer
6. ✅ **Content slides**: 12 of them, each with full-slide PNG, no overlap text from .pptx layer
7. ✅ **Speaker notes**: each of 12 content slides has notes populated from the content spec's "## Speaker notes" section
8. ✅ **No speaker notes** on cover or section divider slides
9. ✅ **File opens cleanly** in PowerPoint for Mac (Kinga's primary tool) and PowerPoint Online
10. ✅ **File size reasonable**: < 10MB total (12 × ~300KB PNGs + .pptx overhead)

### Manual verification step

After build, Cowork should:
1. Open the .pptx in `Numbers` or whatever .pptx-compatible viewer is available in the sandbox
2. Render the first 3 slides (cover, A divider, A1) as PNG screenshots
3. Visually confirm: backgrounds dark, typography clean, A1 diagram fills the slide
4. If sandbox can't render .pptx, ship as-is and Ash verifies on Mac

---

## Deliverable Sequence

Single commit with all artifacts:

```bash
git add docs/deck/exports/aivc_appendix_v1.pptx \
        docs/deck/exports/_build_appendix_pptx.py
git commit -m "docs(deck): Phase 3 — appendix .pptx assembly

18 slides: 1 cover + 5 section dividers + 12 content slides.
Each content slide is full-slide PNG (rendered from SVG master);
speaker notes embedded from content spec Q&A sections.

Output: docs/deck/exports/aivc_appendix_v1.pptx
Build: docs/deck/exports/_build_appendix_pptx.py (regeneratable)

Format: 16:9, dark navy background, matches Kinga's source deck
aesthetic. SVG masters in docs/deck/assets/diagrams/ remain the
source of truth for any future diagram regeneration."
git push origin main
```

The .pptx file commits to repo since it's part of the deliverable. Size budget < 10MB.

---

## What Comes After Phase 3

**Phase 4 — Claude Design polish** (optional)

Hand-tune visual hierarchy on the highest-priority slides:
- Cover slide typography (impression-setting)
- A1 (most-referenced architecture diagram)
- A3 (most-novel architecture claim)
- B2 (verdict slide with hero number)
- C1 (proprietary moat reveal)
- C2 (headline demo plan)
- D1 (roadmap legibility)
- E1 (closing horizon)

Other slides (A2, A4, B1, B3, D2) hold quality with Cowork output.

Phase 4 happens AFTER Kinga reviews Phase 3 output. Two reasons:
1. Kinga may have suggested edits that change Phase 4 scope
2. Phase 3 output may be "good enough as-is" — no need to spend Phase 4 budget

---

## Risks To Flag

1. **Font rendering in PowerPoint**: PNGs sidestep font issues entirely. But the cover slide and section dividers use .pptx-native text — must use a font available on Kinga's Mac. Inter is recommended; fallback to Arial. Document the choice.

2. **Slide notes formatting limits**: python-pptx notes are plain text. Markdown blockquotes (`>`) render as literal `>` characters in PowerPoint. Use the simple text approach; the Q&A structure remains readable.

3. **PNG file paths**: filenames are `_preview.png` suffixed per the diagram build convention. Verify all 12 expected files exist before building. If any are missing, the build script should fail loudly, not silently skip.

4. **PowerPoint version compatibility**: pptx generated by python-pptx targets PowerPoint 2007+ OOXML format. Kinga's modern PowerPoint will open it fine. Test on PowerPoint Online if uncertain — should work identically.

5. **Re-running the build**: Build script must be idempotent — running it twice produces the same .pptx. Don't append to an existing file; rebuild from scratch each time.

6. **Section divider count**: My initial count said "4 section dividers"; with 5 sections (A/B/C/D/E) we have 5 dividers. Final slide count: 1 cover + 5 dividers + 12 content = **18 slides**. Verify the math is right.

---

## What's Out Of Scope For Phase 3

- Modifying any SVG content (Phase 2 locked)
- Modifying any content spec (Phase 1 locked)
- Visual polish on hero slides (Phase 4)
- Animation or transition effects (not investor-deck standard)
- Notes pages output (PowerPoint handles separately if needed for printing)
- Embedded fonts in .pptx (Kinga has Inter/Arial; we don't need to embed)

---

## Confirmation Before Starting

If Cowork hits any ambiguity, surface it before coding:
- PNG file paths uncertain → list available files and confirm
- Speaker notes formatting choice unclear → try "simple text" approach first
- Cover/divider typography sizing → use the px/pt suggestions; iterate if visually off
- Slide count math → verify against the 18-slide table above

Don't guess silently — surface and check.
