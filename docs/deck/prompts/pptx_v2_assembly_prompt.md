# pptx v2 Assembly — Add Section F + F1 To Appendix Deck

**Owner**: Cowork (execution)
**Estimated time**: 20-40 min
**Strategy**: Small mechanical extension to existing `_build_appendix_pptx.py`
**Goal**: assemble 20-slide v2 deck with F1 + new Section F divider added

---

## Context

Phase 3 v1 shipped `aivc_appendix_v1.pptx` at commit `c856132` — 18 slides (1 cover + 5 dividers + 12 content). Build script: `docs/deck/exports/_build_appendix_pptx.py`.

F1 SVG + PNG now locked at commit `7e6c31c`. Adding F1 + new Section F divider to the deck brings total to **20 slides** (1 cover + 6 dividers + 13 content).

This is a small extension to the existing build script — same code path, two new entries.

---

## What Changes In The Build Script

Three additions to `_build_appendix_pptx.py`:

### Addition 1 — New Section F divider entry

Add Section F to the section dividers list (or whatever structure the script uses to enumerate dividers):

```python
SECTION_F = {
    "letter": "F",
    "title": "Competitive Positioning",
    "sub": "Why us · the closed-loop integrated platform · proprietary data, co-designed architecture, compounding over time",
    "slides_range": "F1",
    "accent_color": "#FBBF24",  # amber, matches F1 SVG section accent
}
```

This divider sits after Section E divider, before F1 content slide.

### Addition 2 — F1 content slide entry

Add F1 to the content slides list:

```python
("F1", "F1_integrated_platform_preview.png", "F1_competitive_positioning.md"),
```

Following the same tuple pattern as the existing 12 content slides.

### Addition 3 — Output filename

Change output from `aivc_appendix_v1.pptx` to `aivc_appendix_v2.pptx`:

```python
OUTPUT = REPO / "docs/deck/exports/aivc_appendix_v2.pptx"
```

Keep v1 in the repo as a historical artifact; v2 is the new deliverable.

---

## Final Deck Structure (20 Slides)

```
Slide 01  →  APPENDIX COVER
Slide 02  →  SECTION DIVIDER: A · Architecture Depth (cyan)
Slide 03  →  A1 — System Architecture
Slide 04  →  A2 — Multi-Omics Encoder
Slide 05  →  A3 — Decomposed Readout
Slide 06  →  A4 — Temporal Neural ODE
Slide 07  →  SECTION DIVIDER: B · Validation Evidence (green)
Slide 08  →  B1 — Methodology Rigor
Slide 09  →  B2 — Encoder Probe Verdict
Slide 10  →  B3 — Synergy Pre-Demo
Slide 11  →  SECTION DIVIDER: C · QurieSeq Phase 1 (cyan)
Slide 12  →  C1 — Phase 1 Experimental Design
Slide 13  →  C2 — BTK+JAK Headline Demo
Slide 14  →  SECTION DIVIDER: D · Roadmap + Budget (lavender)
Slide 15  →  D1 — Quarterly Roadmap
Slide 16  →  D2 — Seed Allocation
Slide 17  →  SECTION DIVIDER: E · Strategic Horizon (white-pale)
Slide 18  →  E1 — 5-Year Trajectory
Slide 19  →  SECTION DIVIDER: F · Competitive Positioning (amber)  ← NEW
Slide 20  →  F1 — Integrated Platform                              ← NEW
```

---

## Section F Divider Visual Spec

Match the pattern of the existing 5 section dividers, customized for Section F:

```
       SECTION F
       
       Competitive Positioning
       ───────────────────────  ← amber rule
       
       Why us · the closed-loop integrated platform · 
       proprietary data, co-designed architecture, 
       compounding over time
       
       
       Slide F1
```

- **Eyebrow**: "SECTION F" in amber `#FBBF24`, 16pt, letter-spaced
- **Title**: "Competitive Positioning" in white `#F7FAFF`, 56pt Inter Bold
- **Divider rule**: amber underline below title (same length as existing dividers)
- **Sub**: muted `#A8B4C2`, 20pt Arial — "Why us · the closed-loop integrated platform · proprietary data, co-designed architecture, compounding over time"
- **Footer**: muted, 14pt — "Slide F1"

Same dark navy `#070A14` background as other dividers.

---

## Speaker Notes For F1

F1's content spec at `docs/deck/content/F1_competitive_positioning.md` has a full `## Speaker notes` section with 7 diligence Q&As pre-loaded (Cellarity/Recursion/Insitro adjacent comparison, TAHOE scale, Immunai VDJ, pharma deal gap, peer-reviewed paper timing, AI biotech category positioning, "couldn't a competitor build all five layers").

The existing `extract_speaker_notes()` regex in the build script should grab this section verbatim. Verify it extracts cleanly — F1's speaker notes are longer and more structurally varied than the existing 11 (multiple Q&A blocks with longer answers).

If the regex misses any Q&A block, fix it before commit.

---

## Acceptance Criteria

When Cowork ships, verify:

1. ✅ **20 slides total** in `aivc_appendix_v2.pptx`
2. ✅ **16:9 widescreen** 13.333" × 7.500" preserved
3. ✅ **Slide 19 is Section F divider** with amber accent
4. ✅ **Slide 20 is F1 content** with full-slide F1 PNG embedded
5. ✅ **F1 speaker notes populated** with all 7 Q&As from the content spec
6. ✅ **All existing 18 slides unchanged** (regression check on structure + speaker notes)
7. ✅ **File size**: 2.5MB → ~2.8MB (adds 1 PNG at ~250KB + small text overhead)
8. ✅ **Existing v1 file preserved** at `docs/deck/exports/aivc_appendix_v1.pptx` (historical artifact)
9. ✅ **v2 file at**: `docs/deck/exports/aivc_appendix_v2.pptx`
10. ✅ **`.gitignore` exception updated** to include `!aivc_appendix_v2.pptx`

### Regression checks specifically

The existing 18 slides should be byte-for-byte unchanged (modulo timestamps). Verify by counting:
- Slides with PNG embeds: 12 → 13 (added F1)
- Slides with speaker notes: 11 → 12 (added F1)
- Section dividers: 5 → 6 (added F divider)
- Total slides: 18 → 20

---

## Deliverable

Single commit:

```bash
git add docs/deck/exports/aivc_appendix_v2.pptx \
        docs/deck/exports/_build_appendix_pptx.py \
        docs/deck/exports/.gitignore
git commit -m "docs(deck): pptx v2 assembly - 20 slides with F1 added"
git push origin main
```

Single-line commit message to avoid zsh history-expansion issues we hit on Phase 3 v1.

---

## Ship Script Compatibility Notes (Mac)

Cowork's Phase 3 v1 ship script had two known issues we should pre-empt:

### Issue 1 — `stat -c` Linux vs `stat -f` Mac

If the ship script includes a file-size check, use BSD-compatible syntax:

```bash
# BAD (Linux GNU stat — fails on Mac)
size_bytes=$(stat -c %s "$pptx")

# GOOD (BSD/Mac stat)
size_bytes=$(stat -f %z "$pptx" 2>/dev/null || stat -c %s "$pptx")
```

The fallback chain (`stat -f %z 2>/dev/null || stat -c %s`) works on both Mac and Linux.

### Issue 2 — `python-pptx` may not be installed on Mac

If ship script verifies the .pptx structure programmatically via python-pptx, the script must check and install:

```bash
python3 -c "import pptx" 2>/dev/null || {
    echo "Installing python-pptx..."
    pip3 install --break-system-packages python-pptx
}
```

Both issues are tech debt from Phase 3 v1 worth fixing in this iteration.

---

## What Ash Will Check On Review

1. Open `aivc_appendix_v2.pptx` on Mac in PowerPoint
2. Verify 20 slides in the slide panel
3. Verify slide 19 = new Section F divider (amber accent, "Competitive Positioning" title)
4. Verify slide 20 = F1 content slide (flywheel + competitor buckets visible)
5. Open Presenter View on slide 20 — verify F1 speaker notes are populated with the 7 Q&As
6. Spot-check 2-3 existing slides (e.g., slide 03 A1, slide 10 B3, slide 18 E1) to confirm regression — they should look identical to v1

---

## What's Out Of Scope

- Phase 4 visual polish (Claude Design)
- Speaker notes expansion (Phase 4 — adds technical glossary across all slides)
- Pagination unification (currently 12 SVGs at `/ 12`, F1 at `/ 13` — Phase 4 sweeps)
- Modifying A1's blank speaker notes (Phase 4 adds them)
- Refactoring any SVG (Phase 2 + F1 v2 locked)

---

## Risks To Flag

1. **Speaker notes regex on F1's content spec** — F1's notes section has more structural variety than the existing 11 (multiple Q&A blocks with longer answers, italic bold markdown markers). If `extract_speaker_notes()` is regex-based, it may miss content. Test before commit — open the generated .pptx, navigate to slide 20, verify all 7 Q&As appear in the notes panel.

2. **Section F divider styling fidelity** — the existing 5 dividers were styled with specific font/spacing/color choices. Section F must match that pattern exactly except for accent color (amber). Don't reinvent — extend.

3. **v1 preservation** — `aivc_appendix_v1.pptx` should stay in the repo as the historical artifact. Don't delete or overwrite it. v2 is a separate file.

4. **Build script idempotency** — running the build script twice should produce the same v2 output. If the script writes to v1 OR v2 conditionally, make sure the v2 path is deterministic.

5. **PNG fetch from origin/main** — F1 PNG is at commit `7e6c31c`. Build script should reference the local file path on disk (not fetch from GitHub). Verify the PNG exists at `docs/deck/assets/diagrams/F1_integrated_platform_preview.png` before running build.

---

## After This Lands

**Phase 2 + 3 fully complete** — the technical appendix is feature-complete at 20 slides.

Path forward:

1. **Ash visual review** of `aivc_appendix_v2.pptx` on Mac in PowerPoint
2. **Optional**: share with Kinga + Thiago for feedback
3. **Phase 4 polish** (Claude Design):
   - Visual polish on 9 hero slides (cover, A1, A3, B2, C1, C2, D1, E1, F1)
   - Expanded speaker notes (technical glossary) on all 13 content slides
   - A1 speaker notes added (currently blank)
   - Pagination unification across all 13 SVGs
   - `_deck_common.py` `min_gap=2` default sweep across builders
   - Tech debt items rolled in

Phase 4 is the final deliverable polish pass before investor circulation.
