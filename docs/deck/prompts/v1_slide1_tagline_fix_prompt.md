# v1 Slide 1 — Surgical Fix (Tagline Removal)

**Owner**: Cowork (execution)
**Estimated time**: 5-10 minutes
**Trigger**: Pre-CEO-ship visual review caught text overlap on v1 Slide 1 — italic taglines in Panel 2 and Panel 3 overlap the encoder hexagon shapes immediately below them.
**Output**: `docs/deck/exports/aivc_investor_4slide_v1_1.pptx`
**Scope**: Surgical — Slide 1 only. Slides 2/3/4 unchanged from v1.

---

## The Bug

In v1 Slide 1 (commit `8b7a626` and persisted in `0a5d7a2`), each phase panel has an italic tagline near the bottom:

- Panel 1: *"Foundational biological representation system, built on validated public data."*
- Panel 2: *"Learning causal biological responses under controlled perturbations."*
- Panel 3: *"Scaling toward large multimodal causal biological intelligence."*

In Panel 2 and Panel 3, the tagline text overlaps the encoder hexagon shape positioned below the modality icon row. Visible at slide-fill scale. Unprofessional for CEO/VC review.

## The Fix

**Delete all 3 italic taglines from Slide 1.** Do nothing else.

Specifically:
- Remove the italic-tagline text box (and any associated paragraph runs) from Panel 1, Panel 2, Panel 3 in `build_slide1_evolution(prs)` (or whatever the v1 build script names the Slide 1 function)
- Do NOT reposition the encoder hexagon, modality icons, content bullets, or Phase 3 footer
- Do NOT touch Slides 2, 3, 4

This is the minimum-change fix. Cowork should resist any urge to "improve" v1 beyond removing the broken taglines.

## Why Delete Rather Than Reposition

1. Taglines were the weakest content on each panel anyway (marketing language, not investor signal)
2. Repositioning risks introducing new overlaps or layout shifts
3. CEO is comparing v1 (density direction) vs v3 (minimal direction) — v1 should stay v1 in spirit, just without the broken text
4. 5-minute change vs 15-20 minute restructure

## Implementation

Open the existing v1 build script (`docs/deck/investor_4slide/_build_investor_deck.py` based on prior commits) and:

1. Identify the `build_slide1_*` function
2. Find the 3 sections that create italic tagline text boxes for each panel
3. Comment out or delete those sections
4. Update output filename:
   ```python
   OUTPUT = "docs/deck/exports/aivc_investor_4slide_v1_1.pptx"
   ```
5. Update `.gitignore` exception:
   ```python
   # in .gitignore: add line
   !aivc_investor_4slide_v1_1.pptx
   ```
6. Run the script
7. Verify Slide 1 no longer has overlapping italic text

Keep `_build_investor_deck.py` (the v1 script) intact — duplicate to `_build_investor_deck_v1_1.py` if helpful for traceability, OR add a function flag. Cowork's call on which is cleaner.

## Acceptance Criteria

- ✓ 4 slides in output pptx
- ✓ Slide 1: no italic tagline text in any panel
- ✓ Slide 1: encoder hexagons + modality icons + content bullets unchanged in position
- ✓ Slides 2, 3, 4: byte-identical to v1
- ✓ File size approximately same as v1 (40 KB ± 2 KB)
- ✓ `unzip -l v1_1.pptx | grep ppt/media/` empty (still no embedded images)
- ✓ Opens cleanly in PowerPoint Mac
- ✓ Visual verification (LibreOffice render): no text overlapping any shape on Slide 1

## Deliverable

```bash
git add docs/deck/investor_4slide/_build_investor_deck_v1_1.py \
        docs/deck/exports/aivc_investor_4slide_v1_1.pptx \
        docs/deck/exports/.gitignore
git commit -m "docs(deck): investor v1.1 - remove italic taglines from slide 1 (encoder overlap fix)"
git push origin main
```

(Single-line commit. If `_build_investor_deck_v1_1.py` is created as a duplicate; if instead a flag is added to the existing script, adjust the staged paths accordingly.)

## Out Of Scope

- Touching Slides 2/3/4 (they stay as-is from v1)
- Any "polish" beyond tagline removal
- Restructuring panels
- Color/font/spacing changes
- New content additions
- Speaker notes
- v3 changes (v3 stays untouched)

## Risks To Flag

1. **Phase 3 footer position may need slight adjustment** if its vertical position was calculated relative to the tagline positions. If so, keep Phase 3 footer at its current y-coordinate and accept any extra whitespace above it — extra whitespace is preferable to layout drift.

2. **If the build script generates taglines via a loop**, the deletion should remove all 3 taglines uniformly. Don't introduce inconsistency (e.g., delete from Panel 2 + 3 but leave Panel 1).

3. **Panel 1's tagline doesn't overlap anything** (Panel 1 has the modality icons but no encoder hexagon below them — there's no overlap target). Still delete it for consistency. Asymmetric removal (kill 2/3, keep 1/3) would look intentional in a way that's worse than removing all 3.
