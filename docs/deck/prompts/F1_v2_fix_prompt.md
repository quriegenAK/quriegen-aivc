# F1 v2 — Closing Line Collision Fix

**Owner**: Cowork (execution)
**Estimated time**: 15-25 min
**Input commit**: `1cbfe80`
**Input files**:
- `docs/deck/assets/diagrams/F1_integrated_platform.svg`
- `docs/deck/assets/diagrams/_build_f1.py`

---

## Context

F1 v1 shipped at commit `1cbfe80`. Ash visual review approved everything except one issue: the **closing italic line is rendering inside the Quriegen row's footprint** instead of below it.

This is the **same class of bug** Cowork's collision-guard helpers were built to catch (B2/D1/D2). The helper either didn't trip on this specific case or its known-false-positive filter accidentally swept it. Both possibilities worth flagging.

---

## The Bug

### Visual evidence (Ash's zoom into Quriegen row, F1 v1)

Three text elements are stacked at y=898 / y=908 / y=918 — 10px vertical gap between elements at font sizes that need ~14px clearance:

| y-coord | Text content |
|---|---|
| 898 | "all coupled" (end of Quriegen row bullet) |
| 908 | "No public dataset has the combination..." (closing italic line) |
| 918 | "closed-loop system itself — integration is the moat" (Quriegen Optimize line) |

The closing italic line at y=908 sits **between** the Quriegen row's two visible lines. Visually it reads as a third Quriegen-row element rather than the takeaway it's meant to be.

### Root cause (verify in `_build_f1.py`)

The closing line was positioned at y≈908 expecting the Quriegen row to end above it. But the Quriegen row's bottom edge actually extends to ~y=930 (containing the bullet text + the Optimize/Decouple line). The closing line's y-coordinate is inside the row's bounding box.

Two likely causes:
1. Y-coordinate calculation used Quriegen row's top + some offset, but underestimated the row's actual height
2. Layout was designed visually with breathing room, then closing line crept up during refinement

Either way, fix is to recompute the closing line's y-coordinate based on the actual bottom edge of the Quriegen row, with adequate vertical gap.

---

## The Fix

### Move the closing line below the Quriegen row's bottom edge

**Target y-coordinate for closing line**: y ≈ 970-980 (above source footer at y≈1000, below Quriegen row's actual bottom edge)

**Required vertical gap above closing line**: minimum 20px between the Quriegen row's bottom edge and the closing line's top edge (ascent zone).

### Verify the Quriegen row's actual extent

Before repositioning the closing line, measure the actual bottom edge of the Quriegen row's bounding rect + lowest text element. The bottom edge is whichever is lower: the rect's `y + height` OR the lowest text element's `y + font_size × 0.15` (descent).

If the Quriegen row currently extends to y≈930, the closing line should be at y≈955+ (with 25px buffer). If it extends further, closing line moves lower.

### Adjust Quriegen row dimensions if needed

If moving the closing line to y=970 leaves no room above the source footer (source at y≈1000 means closing line bottom edge must be ≤y≈985), consider:
- **Option A**: Compress Quriegen row height by removing the "(each loop deepens the next)" parenthetical (slight content loss, lower visual weight)
- **Option B**: Tighten line-height inside Quriegen row by 2-4px to gain vertical space
- **Option C**: Reduce closing line font size from current to slightly smaller (preserves position, reduces visual prominence — not recommended since closing line is the takeaway)

**Recommendation**: Option B (tighten line-height) preserves content + position. Try B first; fall back to A if B insufficient.

### Run collision-guard pre-write — and AUDIT its filter

Before file write, run `check_no_text_collisions(svg_xml)`. **This time, do NOT auto-filter "footer-like" elements** — log every collision and inspect manually. The v1 false-positive filter likely swept this real collision because the closing italic line resembles source-citation text patterns.

If the helper still doesn't detect the v2 layout as collision-free, that's a helper limitation worth documenting (not a blocker for ship).

---

## Acceptance Criteria

For F1 v2 to ship:

### Layout fix
- ✅ Closing italic line "No public dataset has the combination..." at y ≥ 950
- ✅ Closing italic line bottom edge (y + font_size × 0.15) ≤ y=985 (above source footer)
- ✅ Quriegen row's bottom edge at y ≤ closing line top - 20px
- ✅ Zero visible overlap between Quriegen row text elements and closing line text elements

### No regression
- ✅ All v1 acceptance checks still pass (30+ checks from `_phase2_f1_ship.sh`)
- ✅ Flywheel layout unchanged
- ✅ 3 archetype buckets unchanged
- ✅ Quriegen row content unchanged (same text, possibly tighter line-height)
- ✅ Source citation footer unchanged
- ✅ Pagination `F1 / 13` unchanged
- ✅ Amber accent unchanged

### Helper validation
- ✅ Run `check_no_text_collisions` WITHOUT the auto-filter
- ✅ Document any remaining collisions detected (and confirm they are genuine false positives, not real bugs)
- ✅ Update the filter logic if needed to avoid false-negatives on takeaway-line patterns

---

## Deliverable

Single commit:

```bash
git add docs/deck/assets/diagrams/F1_integrated_platform.svg \
        docs/deck/assets/diagrams/F1_integrated_platform_preview.png \
        docs/deck/assets/diagrams/_build_f1.py
git commit -m "fix(deck): F1 v2 - closing line collision with Quriegen row

Closing italic line was rendering at y=908 inside the Quriegen
row's footprint (row extends to ~y=930). Moved closing line to
y>=950 with explicit gap above. Tightened Quriegen row line-height
to preserve vertical budget against source footer at y=1000.

Collision-guard helper auto-filter was too permissive - swept this
real collision as if it were a source-vs-pagination false positive.
Filter scope tightened: only Source:-prefixed and pagination
elements skip collision check. Other 'footer-like' patterns now
trip the guard correctly."
git push origin main
```

---

## Self-Review Checklist (Cowork)

Before ship, do a zoomed visual check on Quriegen row + closing line region in the v2 PNG:
1. Open `F1_integrated_platform_preview.png` at full resolution
2. Crop the region from y≈840 to y≈1010
3. Verify visually: Quriegen row reads as a single unit, closing italic line sits below it as a separate takeaway, source footer sits below closing line

If anything visually competes for the same vertical band, iterate before ship.

---

## What's Out Of Scope

- Modifying any other SVG (A1-E1 locked)
- Changing F1 content spec (committed at `740e054`)
- Refactoring all collision-guard filter logic (only F1-relevant scope)
- pptx v2 reassembly (separate prompt after F1 v2 lands)

---

## Risks To Flag

1. **The helper's filter logic now has a known false-negative** — the closing-italic-line pattern wasn't caught. Fixing F1's specific instance is one thing; updating the filter to prevent the same blind spot in future SVGs is harder. Don't try to make the filter bulletproof — just tighten its scope per the commit message and document remaining limitations.

2. **Source footer at y≈1000 is the hard floor** — closing line must end above it with visible gap. If the math is tight, prefer compressing Quriegen row over pushing closing line above its current vertical zone.

3. **No regression on v1 acceptance** — the 30+ existing acceptance checks all need to keep passing. The fix is layout-only, no content change.

4. **PNG re-render** — paired SVG+PNG requirement still holds. The build script must re-render the PNG, not commit a stale one.

---

## After This Lands

If F1 v2 ships clean → next prompt is `_build_appendix_pptx.py` re-run for 20-slide v2 deck:
- 1 cover (unchanged)
- 6 dividers (5 existing + 1 new Section F · Competitive Positioning, amber accent)
- 13 content slides (12 existing + F1)
- Final output: `docs/deck/exports/aivc_appendix_v2.pptx`

Then **Phase 2 + 3 fully complete** → Phase 4 (Claude Design polish + expanded speaker notes) begins.
