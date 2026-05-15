# QurieGen Deck Icon Inventory

**Source**: Scanned `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx` for reusable icon primitives across slides 1–40.
**Extracted**: 2026-05-15.

---

## What Kinga's deck actually uses

The deck contains **143 media items** in `ppt/media/`, but most are JPEG/PNG photos (team headshots, market visuals) — not vector icons. The technical slides (8, 9, 37) build their iconography from **PowerPoint preset geometry shapes**, not imported icon sets.

Slide 37 (our A1 anchor) preset-geometry inventory:

| Shape | Count | Role |
|---|---|---|
| `rect` | 62 | Block borders, dividers, frame elements |
| `roundRect` | 34 | Primary block container (this is the dominant visual primitive) |
| `ellipse` | 6 | Step-number circles, accent dots |
| `line` | 5 | Flow arrows / dividers |

Slide 8 inventory:

| Shape | Count | Role |
|---|---|---|
| `rect` | 55 | Same as slide 37 |
| `roundRect` | 22 | Same |
| `ellipse` | 11 | Used as success markers (with `#4ADE80` border) |
| `line` | 4 | Dividers |

---

## Reusable primitives for appendix SVGs

### Block container (canonical)
- **Shape**: rounded rectangle, corner radius ≈ 12 px on 1920×1080 viewBox
- **Fill**: `#0F1428` (synthetic, slightly lighter than `#070A14` background; gives subtle elevation) or no fill (transparent)
- **Stroke**: 1.5 px solid in step-number color (cyan / lavender / etc.)
- **Inner padding**: 24 px on 1920×1080

### Step-number marker
- Small rounded rect, 36×24 px, fill = step color, white text
- OR small circle/ellipse, 28 px diameter, fill = step color
- Slide 37 actually uses **text only** for "01"/"02"/"03" with step-color fill — no circle/badge. Simplest is to mirror this.

### Flow arrow
- Slide 37 uses subtle thin `line` primitives, not filled arrowheads.
- Recommended for appendix: 4 px stroke, color `#26DDF9` at 60% alpha, no arrowhead (or minimal triangle), rounded line caps.

### Status icons — **rendered as inline glyphs, not images**

Unicode glyphs that render universally without needing an icon font:

| Status | Glyph | Unicode | Hex color | Use case |
|---|---|---|---|---|
| Completed / validated | `✓` (U+2713) **or** `●` colored green | n/a | `#4ADE80` | Stage 1+2 done, validation landed |
| In-flight | `◐` (U+25D0) **or** `●` colored amber | n/a | `#FBBF24` | Stage 3a training, BTK+JAK demo running |
| Pending | `○` (U+25CB) **or** `●` colored grey | n/a | `#A8B4C2` | Stage 3b/3c, Phase 2 modalities |
| Locked / frozen | `🔒` (U+1F512) **or** custom-drawn padlock SVG | n/a | `#26DDF9` | Frozen encoder indicator |
| Adapter / learn | `△` (U+25B3) **or** `↻` (U+21BB) | n/a | `#B47DF0` | Trainable adapter sub-block |

**Note on emoji rendering**: A1 spec uses `✅` / `🟡` / `⏸` emoji directly. These render fine in modern PowerPoint and browsers but their color is fixed by the platform's emoji font. For maximum visual coherence with the deck's neon palette, **prefer colored Unicode glyphs (✓ / ◐ / ○) over emoji**, drawn in the SVG with our brand hex colors.

### Lock icon (frozen encoder)
The `🔒` emoji renders with platform-specific styling (Apple = blue padlock, Windows = grey). For appendix coherence, draw a minimal padlock as inline SVG:

```svg
<g stroke="#26DDF9" stroke-width="2" fill="none">
  <rect x="0" y="6" width="14" height="10" rx="2" fill="#26DDF9" fill-opacity="0.15"/>
  <path d="M3 6 V3 a4 4 0 0 1 8 0 V6" />
</g>
```

Render at 14×16 px next to the ENCODER block title.

---

## Decisions locked

1. **No external icon sets** (Font Awesome, Lucide, etc.). Use inline SVG primitives only — keeps appendix self-contained and matches Kinga's preset-geometry approach.
2. **Status indicators**: colored Unicode glyphs (`✓` / `◐` / `○`), not emoji.
3. **Lock icon**: inline SVG padlock (above), drawn in `#26DDF9`.
4. **No drop shadows / gradients** on any icon — keep flat and crisp; matches Kinga's slide 37 aesthetic.

---

## Open questions for Ash (to resolve during A1 review)

1. **Emoji vs Unicode glyph for status row**: A1 spec lines 56–62 use `✅` / `🟡` / `⏸`. Recommend overriding with `✓` / `◐` / `⏸` rendered in brand hex (above). Confirm during A1 review.
2. **Drop the `🔒` emoji in favor of inline SVG padlock**: cleaner visual hierarchy. Recommend yes.
3. **No reusable amber/warning icon found in source** — derived from Tailwind family (see `color_palette.md`).
