# pptx v3 Assembly — Add A5 Between A4 And Section B Divider

**Owner**: Cowork (execution)
**Estimated time**: 15-30 min
**Strategy**: Small mechanical insertion into existing `_build_appendix_pptx.py`
**Goal**: assemble 21-slide v3 deck with A5 inserted mid-deck

---

## Context

`aivc_appendix_v2.pptx` shipped at commit `7604343` — 20 slides (1 cover + 6 dividers + 13 content). Build script: `docs/deck/exports/_build_appendix_pptx.py`.

A5 SVG + PNG locked at commit `8aa32fd`. Adding A5 between A4 (current slide 6) and Section B divider (current slide 7) brings total to **21 slides** (1 cover + 6 dividers + 14 content).

A5 inserts mid-deck. F1 stays at the end. All other slides keep their content unchanged — only their slide numbers shift down by 1 after A5's position.

---

## What Changes In The Build Script

Three additions to `_build_appendix_pptx.py`:

### Addition 1 — Insert A5 in CONTENT_SLIDES list

Find the existing tuple list. Insert A5 at position 5 (after A4, before B1):

```python
CONTENT_SLIDES = [
    ("A1", "A1_system_architecture_preview.png", "A1_system_architecture.md"),
    ("A2", "A2_encoder_evidence_preview.png", "A2_encoder_substrate.md"),
    ("A3", "A3_decomposed_readout_preview.png", "A3_decomposed_readout.md"),
    ("A4", "A4_temporal_dynamics_preview.png", "A4_temporal_neural_ode.md"),
    # NEW: A5 inserted here
    ("A5", "A5_causal_architecture_preview.png", "A5_causal_architecture.md"),
    ("B1", "B1_three_datasets_methodology_preview.png", "B1_methodology_rigor.md"),
    # ... rest unchanged through F1
]
```

The build script's slide-generation loop iterates this list, so inserting A5 here makes everything work without any logic change.

### Addition 2 — Output filename

Change output from `aivc_appendix_v2.pptx` to `aivc_appendix_v3.pptx`:

```python
OUTPUT = REPO / "docs/deck/exports/aivc_appendix_v3.pptx"
```

v2 stays in repo as historical artifact (same pattern as v1 preserved when v2 shipped).

### Addition 3 — `.gitignore` exception

Add `!aivc_appendix_v3.pptx` to `docs/deck/exports/.gitignore` alongside v1 and v2 exceptions.

---

## Final Deck Structure (21 Slides)

```
Slide 01  →  APPENDIX COVER
Slide 02  →  SECTION DIVIDER: A · Architecture Depth (cyan)
Slide 03  →  A1 — System Architecture
Slide 04  →  A2 — Multi-Omics Encoder
Slide 05  →  A3 — Decomposed Readout
Slide 06  →  A4 — Temporal Neural ODE
Slide 07  →  A5 — Causal Architecture (NEW)
Slide 08  →  SECTION DIVIDER: B · Validation Evidence (green)
Slide 09  →  B1 — Methodology Rigor
Slide 10  →  B2 — Encoder Probe Verdict
Slide 11  →  B3 — Synergy Pre-Demo
Slide 12  →  SECTION DIVIDER: C · QurieSeq Phase 1 (cyan)
Slide 13  →  C1 — Phase 1 Experimental Design
Slide 14  →  C2 — BTK+JAK Headline Demo
Slide 15  →  SECTION DIVIDER: D · Roadmap + Budget (lavender)
Slide 16  →  D1 — Quarterly Roadmap
Slide 17  →  D2 — Seed Allocation
Slide 18  →  SECTION DIVIDER: E · Strategic Horizon (white-pale)
Slide 19  →  E1 — 5-Year Trajectory
Slide 20  →  SECTION DIVIDER: F · Competitive Positioning (amber)
Slide 21  →  F1 — Integrated Platform
```

A5 is the new slide. Everything below it shifts down by 1 from v2.

---

## Speaker Notes For A5

A5's content spec at `docs/deck/content/A5_causal_architecture.md` has a full `## Speaker notes` section with 7 diligence Q&As (Stage 3c spec status, why Neumann, STRING confidence handling, A3 relationship, fallback paths, BTK+JAK connection, when operational, competitive differentiation).

The existing `extract_speaker_notes()` regex in the build script should grab this section verbatim — same pattern that worked for F1's 7 Q&As. **Verify extraction**: open the generated v3 .pptx, navigate to slide 7, verify all 7 A5 Q&As appear in the notes panel.

---

## Acceptance Criteria

When Cowork ships, verify:

1. ✅ **21 slides total** in `aivc_appendix_v3.pptx`
2. ✅ **16:9 widescreen** 13.333" × 7.500" preserved
3. ✅ **Slide 7 = A5 content** with full-slide A5 PNG embedded
4. ✅ **A5 speaker notes populated** with all 7 Q&As from content spec
5. ✅ **Section A grows**: Section A divider (slide 2) still says "Slides A1-A4" — **needs update to "Slides A1-A5"**
6. ✅ **All existing 20 slides** (v2) preserve their content; only slide numbers shift for everything below A5
7. ✅ **File size**: ~2.8MB → ~3.1MB (adds 1 PNG at ~280KB + small text overhead)
8. ✅ **Existing v1 + v2 files preserved** as historical artifacts
9. ✅ **v3 file at**: `docs/deck/exports/aivc_appendix_v3.pptx`
10. ✅ **`.gitignore` exception** updated to include `!aivc_appendix_v3.pptx`

### Regression checks specifically

- Slides with PNG embeds: 13 → 14 (added A5)
- Slides with speaker notes: 12 → 13 (added A5)
- Section dividers: 6 (unchanged — F divider already exists from v2)
- Total slides: 20 → 21

---

## Section A Divider Update

The Section A divider (slide 2) currently says "Slides A1-A4" per v2. With A5 added, this needs to read "Slides A1-A5".

Check the `SECTIONS` definition in `_build_appendix_pptx.py` for the Section A entry's `slides_range` field. Update from `"A1-A4"` (or whatever the current value is) to `"A1-A5"`.

This is the **only** change to existing slide content — all other slides stay identical to v2.

---

## Ship Script Compatibility (Mac, already proven for v2)

Ship script should use the Mac-compatible patterns from v2:

- `stat -f %z 2>/dev/null || stat -c %s` for file size check
- `python-pptx` install check with `pip3 install --break-system-packages` fallback
- Single-line commit message to avoid zsh history-expansion issues

Same patterns that worked clean on v2 ship — should run without friction.

---

## Deliverable

Single commit:

```bash
git add docs/deck/exports/aivc_appendix_v3.pptx \
        docs/deck/exports/_build_appendix_pptx.py \
        docs/deck/exports/.gitignore
git commit -m "docs(deck): pptx v3 assembly - 21 slides with A5 added"
git push origin main
```

---

## What Ash Will Check On Review

1. Open `aivc_appendix_v3.pptx` on Mac in PowerPoint
2. Verify 21 slides in slide panel
3. Verify slide 2 (Section A divider) reads "Slides A1-A5"
4. Verify slide 7 = A5 content slide (causal architecture, status pill, equation, GRN visualization)
5. Open Presenter View on slide 7 — verify all 7 A5 Q&As in notes panel
6. Spot-check 2-3 existing slides (e.g., slide 3 A1, slide 11 B3, slide 21 F1) for regression — should look identical to v2

---

## What's Out Of Scope

- Phase 4 visual polish (Claude Design)
- Speaker notes expansion (Phase 4 adds technical glossary across all slides)
- Pagination unification (A1-E1 still `/12`, F1 still `/13`, A5 at `/14` — Phase 4 sweeps)
- A1 blank speaker notes (Phase 4 adds these)
- Modifying any SVG (Phase 2 + A5 v3 locked)

---

## Risks To Flag

1. **A5 PNG file size**: A5's PNG is ~280KB (vs typical ~225KB for other slides) due to the GRN visualization complexity + embedded math PNGs. Still well under budget (deck total ~3.1MB << 10MB).

2. **Section A divider update** is the only content change to existing slides. If the divider's `slides_range` field is hard-coded rather than auto-computed, it must be manually updated. If auto-computed from CONTENT_SLIDES, no explicit change needed (worth verifying).

3. **Speaker notes regex on A5's content spec** — A5's notes are structurally similar to F1's (multiple Q&As, varied formatting). The regex worked clean on F1 → expect clean extraction for A5. Verify in the generated .pptx anyway.

4. **A5 content spec references "v1.1 §X causal layer pending"** in the source footer — this is intentional honesty. Don't try to "fix" this reference; A5 anchors a forthcoming spec extension.

5. **v3 .pptx commits to repo** at ~3.1MB. v1 (2.5MB) + v2 (2.8MB) + v3 (3.1MB) = ~8.4MB total historical artifacts in repo. Acceptable; deck deliverables are reasonable repo assets.

---

## After This Lands

**Phase 2 + 3 fully complete** with A5 included. Technical appendix is feature-complete at 21 slides.

Path forward:

1. **Ash visual review** of `aivc_appendix_v3.pptx` on Mac in PowerPoint
2. **Optional**: share with Kinga + Thiago for feedback
3. **Phase 4 polish** (Claude Design):
   - Visual polish on 9 hero slides (cover + A1 + A3 + A5 + B2 + C1 + C2 + D1 + E1 + F1) — note A5 now in hero list
   - Expanded speaker notes (technical glossary) on all 14 content slides
   - A1 speaker notes added (currently blank)
   - Pagination unification across all 14 SVGs
   - Color coding restoration on A5 equation (3 separate mathtext PNGs for W cyan / dₚ lavender / (I−W)⁻¹ green)
   - Architecture spec v1.2 §X causal-layer extension (optional Phase 4 scope)
   - Other tech debt items rolled in

Phase 4 is the final deliverable polish pass before investor circulation.
