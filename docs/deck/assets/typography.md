# QurieGen Deck Typography

**Source**: Extracted from `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx` by parsing `<a:rPr>` runs across `ppt/slides/slide{8,9,37}.xml` and `ppt/theme/theme1.xml`.
**Extracted**: 2026-05-15.

---

## Observed fonts (frequency across all 40 slides)

| Typeface | Run count | Role in deck |
|---|---|---|
| `Arial` | 22,591 | Default body font everywhere — universal fallback |
| `Quattrocento Sans` | 1,368 | Occasional uppercase accent on early slides |
| `-apple-system-` / `-apple-system` | 2,610 | Slide 37 section labels + step numbers (CSS web font alias, **not a real installable typeface**) |
| `Calibri` | 643 | Slide 8 body — Office default fallback |
| `Inter` | 116 | Slide 37 explicit declarations (real typeface) |
| `Century Gothic`, `Corbel`, `Oswald`, `Dosis` | <500 each | Minor decorative accents on cover / non-technical slides |

The theme (`ppt/theme/theme1.xml`) declares Office defaults `Calibri Light` (major) / `Calibri` (minor), but **every body slide overrides** with either Arial or `-apple-system-`. `-apple-system-` is a CSS web alias (SF Pro on Apple platforms, falls back to system sans elsewhere) — not a real font we can ship in matplotlib.

---

## Canonical typography stack for appendix SVGs

| Role | Primary | Fallback chain | Size hint |
|---|---|---|---|
| Slide title | `Inter`, weight 700 | `Inter, -apple-system, "Helvetica Neue", Arial, sans-serif` | 32–36 pt |
| Block title (e.g. "ENCODER") | `Inter`, weight 700 | same | 22–26 pt |
| Section label / step number | `Inter`, weight 700, uppercase, letter-spacing 0.2em | same | 10–11 pt |
| Body text (block content) | `Arial`, weight 400 | `Inter, Arial, "Helvetica Neue", sans-serif` | 12–14 pt |
| Caption / footnote | `Arial`, weight 400 | same | 9–10 pt |
| Status row (icons + keyword) | `Inter`, weight 600 | same | 11–12 pt |
| Monospace / code (if needed) | `Menlo` | `"SF Mono", Menlo, Consolas, monospace` | 11 pt |

**SVG `font-family` attribute** (use this exact stack for portability across PowerPoint / browsers / matplotlib):

```
font-family="Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
```

For body text (when we want to match slide 37's body tone more directly):

```
font-family="Arial, Inter, 'Helvetica Neue', sans-serif"
```

---

## Slide 37 specifics (our A1 visual anchor)

| Element | Font | Size | Color | Bold | Letter spacing |
|---|---|---|---|---|---|
| Headline "AI virtual cell model" | Inter | 32 pt | `#26DDF9` / `#F7FAFF` | Yes | `spc=-180` (tight) |
| Tagline ("PRIMARY IMMUNE CELLS · 4-MODALITY FUSION · CAUSAL BY DESIGN") | `-apple-system-` | 10 pt | `#26DDF9` | Yes | `spc=180` (wide) |
| Step number ("01", "02", "03") | `-apple-system-` | 10 pt | block-specific (cyan / lavender / white) | Yes | `spc=200` |
| Block label ("INPUT DATA", "CAUSAL CORE", "OUTPUT LAYER") | `-apple-system-` | 10 pt | `#EAF6FF` | Yes | `spc=300` (wider) |
| Body copy ("Primary immune cells · 5 modalities + perturbation") | `-apple-system-` / `Arial` | 12 pt | `#A8B4C2` | Mixed | `spc=100` |

Approach for A1: use **Inter** as the primary face (it's the only real typeface explicitly declared on slide 37), fall back to Arial. The `-apple-system-` runs are CSS aliases — they resolve to whatever sans the renderer picks; we get the same visual outcome by declaring Inter explicitly.

---

## matplotlib environment notes

- Inter ships with most Mac dev environments via Google Fonts cache (`~/Library/Fonts/`); on BSC compute / Linux containers it usually doesn't, so matplotlib will fall back to **DejaVu Sans** (its bundled default).
- DejaVu Sans is a close-enough open-source sans for SVG generation — character widths differ from Inter by ~3–5% but the SVG `font-family` attribute is preserved on output, so the rendered .pptx will pick up Inter on Kinga's Mac.
- **No need to embed fonts in SVG**: PowerPoint resolves `font-family` against the host system. Inter is installable on Kinga's Mac if not already there.

If matplotlib raises "font not found" warnings during A1 generation, that's expected and harmless — the SVG file itself carries the `font-family` declaration verbatim.

---

## Decisions locked

1. **All appendix SVGs use Inter (with Arial fallback)**. No exotic fonts.
2. **No Calibri** — it's the Office default that Kinga's slides explicitly override.
3. **No Quattrocento Sans / Oswald / Dosis** — those are decorative cover-slide accents, not appropriate for the technical appendix.
4. **Letter-spacing matches slide 37**: 0.2em on step numbers, 0.3em on section labels.
