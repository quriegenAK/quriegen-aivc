# QurieGen Deck Color Palette

**Source**: Extracted from `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx`, slides 8 / 9 / 37 (the technical-depth body slides the appendix maps to).
**Extracted**: 2026-05-15, via direct XML parse of `ppt/slides/slide{8,9,37}.xml` (the .pptx is a ZIP archive).
**Slide size**: 13.333" × 7.5" (EMU 12192000 × 6858000) → 16:9 widescreen → **SVG viewBox 1920 × 1080**.

---

## Canonical palette (use these in every appendix SVG)

| Role | Hex | RGB | Where used in source |
|---|---|---|---|
| Dark background | `#070A14` | 7, 10, 20 | Uniform body-slide background (slides 2–17, 37; confirmed via `<p:bg>` parse) |
| Deeper background | `#0A0A1A` | 10, 10, 26 | Slide 8 cell backgrounds (nested dark surface) |
| Primary brand cyan | `#26DDF9` | 38, 221, 249 | Primary brand accent — titles, highlights, frames, divider rules |
| Electric cyan | `#00F2FF` | 0, 242, 255 | Slide 37 step "01" marker, hero accents, neon edges |
| Primary purple | `#8B5CF6` | 139, 92, 246 | Headlines, secondary blocks, SOLUTION callout (slide 7) |
| Lavender | `#B47DF0` | 180, 125, 240 | Slide 37 step "02" marker, secondary glow |
| Vivid purple | `#B469E8` | 180, 105, 232 | Slide 8 large headline color |
| Pale body text | `#EAF6FF` | 234, 246, 255 | Slide 37 section labels (INPUT DATA / CAUSAL CORE / OUTPUT LAYER) — primary body text on dark |
| Off-white | `#F7FAFF` | 247, 250, 255 | Slide titles, primary text on dark backgrounds |
| White | `#FFFFFF` | 255, 255, 255 | Hero text, high-contrast labels |
| Muted text | `#A8B4C2` | 168, 180, 194 | Slide 37 secondary captions / supporting copy |
| Slate muted | `#94A3B8` | 148, 163, 184 | Tertiary captions, footnotes |
| Light slate | `#C8CFE8` | 200, 207, 232 | Slide 8 body-text tone |
| Success green | `#4ADE80` | 74, 222, 128 | Slide 8 success border / verified check — **✅ status icon color** |
| Warning amber | `#FBBF24` | 251, 191, 36 | **Not native to deck; derived (Tailwind amber-400)** to fit deck's modern neon family — **🟡 in-flight status** |
| Pending grey | `#A8B4C2` | 168, 180, 194 | **⏸ pending status** — re-use muted text tone (same as muted text) |
| Danger red | `#FF4D6D` | 255, 77, 109 | PROBLEM ICEBERG slide (slide 4), competitive landscape negatives — RED verdicts |

---

## Notes on derived colors

The source deck does **not** contain an explicit amber/warning color in the technical slides — Kinga's narrative uses a binary green/red scheme. For the appendix we need a third status (🟡 in-flight) to honestly mark Stage 3a as live training. `#FBBF24` (Tailwind amber-400) was chosen because:

1. It sits in the same modern-neon family as the deck's other Tailwind-adjacent colors (`#4ADE80` = green-400, `#8B5CF6` = violet-500).
2. It reads cleanly against `#070A14` background at small icon sizes.
3. It does not collide with any existing brand color.

Document this as a deliberate extension; surface in deck QA if Kinga wants a different amber.

---

## Step-number color rotation (mirrors slide 37 convention)

Slide 37 numbers its three blocks `01` (cyan) → `02` (lavender) → `03` (white). For the A1 5-block flow we extend the rotation:

| Step | Color | Block |
|---|---|---|
| 01 | `#00F2FF` | INPUT |
| 02 | `#B47DF0` | ENCODER |
| 03 | `#26DDF9` | TEMPORAL |
| 04 | `#8B5CF6` | READOUT |
| 05 | `#EAF6FF` | OUTPUT |

---

## CSS variable export (for SVG/HTML reuse)

```css
:root {
  --bg-dark:       #070A14;
  --bg-deeper:     #0A0A1A;
  --brand-cyan:    #26DDF9;
  --brand-cyan-hi: #00F2FF;
  --brand-purple:  #8B5CF6;
  --brand-lavender:#B47DF0;
  --text-primary:  #EAF6FF;
  --text-title:    #F7FAFF;
  --text-muted:    #A8B4C2;
  --status-ok:     #4ADE80;
  --status-warn:   #FBBF24;
  --status-pending:#A8B4C2;
  --status-fail:   #FF4D6D;
}
```
