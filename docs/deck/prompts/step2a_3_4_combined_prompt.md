# Step 2a + Step 3 + Step 4 — C1 Cleanup + SVG Rebuilds + pptx v4

**Owner**: Cowork (execution)
**Estimated time**: 3-4 hours total (3 substeps batched)
**Input commits**: 
- `644b1f8` (canonical correction doc)
- `7952b6b` (Step 2 content spec updates)
**Strategy**: Single iteration covering C1 spec cleanup, SVG rebuilds for A2/C1/D1/F1, and pptx v4 mechanical reassembly

---

## Why Batched

These three substeps form a single feedback loop. Splitting them adds ship-script + commit overhead per substep without strategic benefit. Single iteration: cleanup spec → rebuild affected SVGs → reassemble pptx → ship.

If any substep surfaces an issue, the iteration stops, we fix, and resume. But planned path is one continuous flow.

---

## Substep 1 — C1 Spec Cleanup (~15 min)

### Issues to fix in `docs/deck/content/C1_phase1_design.md`

**Fix 1A — Source data / claims table has stale rows**

The source-data table at the bottom of C1 contains rows that contradict the new top-of-doc framing:
- "Phospho deferred to Phase 2" — STALE (phospho is in Phase 1 now)
- "ATAC at donor level" — STALE if it implies single-timepoint vs the new t=0/t=180 framing

**Action**: Review every row in the C1 source-data table. Update rows that conflict with the Phase 1 modality matrix (Phospho at all 5 timepoints; ATAC at t=0 and t=180). Each updated row should trace to either:
- `docs/deck/research/phase1_modality_correction_2026_05_17.md` (canonical correction)
- Thiago confirmation (May 12 conversation — 5 donors, 5 timepoints, BTK+JAK combo)
- Kinga clarification (May 17 — phospho integral to QuRIE-seq, ATAC at endpoints)

**Fix 1B — Perturbation conditions block has uncertain placeholders**

Current C1 spec includes:
```
PERTURBATION CONDITIONS (17 total)
· Vehicle control (baseline)
· Stimulus (PMA/Iono or equivalent — TBD with Thiago)
· 6 inhibitor singles: BTK, JAK, [+4 others TBD]
· 6 inhibitor + stim combos
· 3 inhibitor + inhibitor combos including BTK+JAK
```

The specific counts (6 singles, 3 combos, 17 total) and the unnamed 4 inhibitors are not Thiago-confirmed. Drafting numbers on the slide that may shift creates rework risk.

**Action**: Soften to conservative framing that preserves the architectural argument without committing to specific counts:

```
PERTURBATION CONDITIONS
· Vehicle control (baseline)
· Stimulus condition (panel under final wet-lab spec review)
· Inhibitor singles — BTK, JAK confirmed for Phase 1 (additional panel TBC)
· Inhibitor combinations — BTK + JAK combo CONFIRMED for Phase 1 (Stage 3b headline)
· Combo conditions sized to support pre-registered compositional generalization eval
```

This preserves the load-bearing claim (BTK+JAK combo is in Phase 1; supports Stage 3b demo) while not committing to numbers we haven't verified.

**Fix 1C — Add panel-size speaker note Q&A**

To handle the predictable diligence question, add to C1 speaker notes:

```markdown
**If asked: "What's the full perturbation panel size?"**
> Final panel is under wet-lab spec review with Thiago. The architecturally load-bearing condition — BTK+JAK combo — is locked for Phase 1 (Thiago confirmation May 12). Total panel size expected to land at ~15-20 conditions including vehicle, stimulus, inhibitor singles, and combinations. The exact perturbation count is a wet-lab parameter, not an architectural commitment — the architecture is sized to support compositional generalization regardless of final panel size as long as BTK+JAK combo is included.
```

### Acceptance for Substep 1

- ✓ C1 source-data table has no rows contradicting the Phase 1 modality matrix
- ✓ Perturbation conditions block uses conservative framing (no unverified specific counts)
- ✓ Speaker notes include "panel size" Q&A
- ✓ All other C1 content from `7952b6b` preserved (headline, sub-headline, body bullets, visual spec for modality matrix, other speaker notes)

---

## Substep 2 — SVG Rebuilds (~2-3 hours total)

Four SVGs need rebuild. **A5 SVG NOT touched** (speaker notes only update per Step 2 spec).

### SVG Rebuild 2A — C1 Phase 1 Experimental Design

**File**: `docs/deck/assets/diagrams/C1_phase1_experimental_design.svg`

**The visual hero changes substantively**. Current C1 SVG likely has a generic Phase 1 design visual; replace with the modality × timepoint matrix as the central element.

**New visual structure**:

```
TOP ZONE — Section eyebrow + headline + sub-headline (standard pattern)

CENTER ZONE — Modality × Timepoint Matrix (visual hero, ~45% vertical space)

             ┌──────┬──────┬──────┬──────┬───────┐
             │ t=0  │ t=5  │ t=30 │ t=60 │ t=180 │
┌────────────┼──────┼──────┼──────┼──────┼───────┤
│ RNA        │  ✓   │  ✓   │  ✓   │  ✓   │   ✓   │   <- cyan checkmarks
├────────────┼──────┼──────┼──────┼──────┼───────┤
│ Protein    │  ✓   │  ✓   │  ✓   │  ✓   │   ✓   │   <- green checkmarks
├────────────┼──────┼──────┼──────┼──────┼───────┤
│ Phospho    │  ✓   │  ✓   │  ✓   │  ✓   │   ✓   │   <- lavender checkmarks (PROPRIETARY)
├────────────┼──────┼──────┼──────┼──────┼───────┤
│ ATAC       │  ✓   │      │      │      │   ✓   │   <- muted blue checkmarks
└────────────┴──────┴──────┴──────┴──────┴───────┘

VDJ row (amber tinted, Phase 2 label)

Below matrix:
"4 MODALITIES MEASURED · 5 TIMEPOINTS · 5 DONORS · ~125K CELLS · BTK+JAK COMBO INCLUDED"
"└── No public dataset has this combination ──┘"

BOTTOM ZONE — Two side-by-side blocks (~30% vertical space)

LEFT block: PERTURBATION CONDITIONS (use the softened framing from Substep 1)
RIGHT block: WET-LAB PARAMETERS (5 donors, Sanquin, all major lineages, etc.)

FOOTER — Source citation + pagination "C1 / 14"
```

**Critical visual treatment**:
- **Phospho row should visually pop** — lavender accent stronger than other rows since it's the proprietary modality
- **ATAC row** should visually show the 2-timepoint coverage clearly (empty cells, not "✗" — keeps it positive framing)
- **VDJ row** muted/amber tinted with "Phase 2" label — present in the matrix but visually distinct from Phase 1 rows
- "No public dataset has this combination" subtitle directly under the matrix — this is the strategic anchor

**Section accent**: Section C cyan (`#26DDF9`) — matches existing C2.

### SVG Rebuild 2B — A2 Multi-Omics Encoder

**File**: `docs/deck/assets/diagrams/A2_encoder_evidence.svg`

**Smaller change than C1**. Keep the existing 73% Calderon cross-corpus visual as hero. **Add** a modality progression strip showing the three-state framing.

**Layout addition**:

```
[Existing A2 visual: 73% Calderon cross-corpus result as hero]

NEW STRIP (below or beside the cross-corpus visual):

ENCODER MODALITY PROGRESSION
─────────────────────────────────────────────────────────────

TODAY                  PHASE 1 (Q3 2026)         PHASE 2 (2027)
────────               ─────────────────         ──────────────
RNA                    + Phospho                 + VDJ  
ATAC                   [proprietary,             [proprietary,
Protein                 5 timepoints]             5th modality]

[public DOGMA-seq      [QuRIE-seq Phase 1]       [QuRIE-seq Phase 2]
 73% Calderon eval]
```

**Visual treatment**:
- Three columns equal width
- Today column: standard cyan (existing palette)
- Phase 1 column: emphasize **+ Phospho** in lavender (proprietary modality)
- Phase 2 column: emphasize **+ VDJ** in amber
- Each column has a small footer label showing data source

**Don't disturb** the existing 73% Calderon hero result. It's shipped evidence.

### SVG Rebuild 2C — D1 Quarterly Roadmap

**File**: `docs/deck/assets/diagrams/D1_quarterly_roadmap.svg`

**Single milestone label change**. The Q1'27 milestone diamond currently reads "Phase 2 phospho on".

**Action**: Change to "Phase 2 VDJ on" (or "Phase 2 VDJ + 20-donor scale" if space allows).

**Optionally** (depending on existing milestone text): The Q3'26 Phase 1 delivery milestone may benefit from a more specific label. If currently it just says "Phase 1 delivery" — leave as-is. If it tries to enumerate modalities and shows "RNA + Protein" only — update to "Phase 1: 4 modalities + BTK+JAK combo".

**Nothing else on D1 changes**. The Gantt structure, swim lanes, all other milestones, source footer — all preserved exactly as in the current shipped D1 SVG.

### SVG Rebuild 2D — F1 Competitive Positioning

**File**: `docs/deck/assets/diagrams/F1_integrated_platform.svg`

**Three changes**:

**Change 1 — TEMPORAL MULTI-OMICS flywheel pillar text**:

```
OLD:
TEMPORAL MULTI-OMICS
RNA + ATAC + Protein
(+phospho + VDJ Phase 2)
0/5/30/60/180 min

NEW:
TEMPORAL MULTI-OMICS
RNA · Protein · Phospho        <- "Phospho" in lavender, bold-emphasized
5 timepoints (0/5/30/60/180)
ATAC × 2 timepoints
VDJ — Phase 2
```

The phospho emphasis is **load-bearing visual signal** that we have the modality no public dataset has.

**Change 2 — PROTOCOL-FAMILY EXPANSION flywheel pillar text**:

```
OLD:
PROTOCOL-FAMILY EXPANSION
Same wet-lab pipeline extends
to Phase 2 phospho + VDJ
without re-architecting

NEW:
PROTOCOL-FAMILY EXPANSION
Same wet-lab pipeline scales
to Phase 2: 20 donors + VDJ
without re-architecting
```

Phase 2's actual scope is now VDJ + scale, not phospho+VDJ.

**Change 3 — Closing italic line**:

```
OLD:
"No public dataset has the combination drug combination prediction requires.
The wet lab, the architecture, and the protocol family are co-designed."

NEW:
"Phase 1 QuRIE-seq (Q3 2026) measures RNA + Protein + Phospho at 5 timepoints.
The combination no public dataset has — the platform compounds from there."
```

This **shifts from aspirational to imminent** — Q3 2026 is the next investor diligence touchpoint.

**Everything else on F1 unchanged**: 4-pillar flywheel structure, 3-bucket archetype grouping (TAHOE/Immunai/CytoReason/etc.), Quriegen amber row, section eyebrow, pagination.

### General requirements for all SVG rebuilds

- Use `min_gap=2` collision guard (established standard)
- Run helper smoke at `min_gap=0` strict to verify no collisions
- Preserve paired SVG + PNG output convention (`*_preview.png` rendered via cairosvg)
- Preserve build script convention (`_build_*.py` per slide)
- Preserve section accent palette (no new colors)
- Banned terms still: "Trimodal", "210-D panel", "Series A", "IPO", "category-of-one", "in production", "operational" (in Phase-1-claim context)

---

## Substep 3 — pptx v4 Reassembly (~30 min)

After 4 SVGs rebuilt:

**Update `_build_appendix_pptx.py`**:
- Change `OUTPUT` path to `aivc_appendix_v4.pptx`
- Update `.gitignore` with `!aivc_appendix_v4.pptx` exception
- v1 + v2 + v3 stay preserved as historical artifacts

**Run the build script**. Verify 21-slide output with updated PNGs for A2, C1, D1, F1.

**No new slides added in v4**. No Section divider changes. Just refreshed content slides + the new output filename.

### Acceptance for Substep 3

- ✓ 21 slides total (same as v3)
- ✓ Slide 7 = A5 (unchanged from v3)
- ✓ Slides for A2, C1, D1, F1 show updated visuals
- ✓ A5 PNG unchanged (speaker notes will update via Step 5, not here)
- ✓ Section A divider footer still reads "Slides A1 – A5"
- ✓ Speaker notes update for A5's 2 Q&As (timing language) included
- ✓ v1 + v2 + v3 preserved at original paths
- ✓ v4 at `docs/deck/exports/aivc_appendix_v4.pptx`

---

## Deliverable

Single batch commit covering everything:

```bash
git add docs/deck/content/C1_phase1_design.md \
        docs/deck/assets/diagrams/A2_encoder_evidence.svg \
        docs/deck/assets/diagrams/A2_encoder_evidence_preview.png \
        docs/deck/assets/diagrams/_build_a2.py \
        docs/deck/assets/diagrams/C1_phase1_experimental_design.svg \
        docs/deck/assets/diagrams/C1_phase1_experimental_design_preview.png \
        docs/deck/assets/diagrams/_build_c1.py \
        docs/deck/assets/diagrams/D1_quarterly_roadmap.svg \
        docs/deck/assets/diagrams/D1_quarterly_roadmap_preview.png \
        docs/deck/assets/diagrams/_build_d1.py \
        docs/deck/assets/diagrams/F1_integrated_platform.svg \
        docs/deck/assets/diagrams/F1_integrated_platform_preview.png \
        docs/deck/assets/diagrams/_build_f1.py \
        docs/deck/exports/aivc_appendix_v4.pptx \
        docs/deck/exports/_build_appendix_pptx.py \
        docs/deck/exports/.gitignore
git commit -m "docs(deck): pptx v4 - phospho phase 1 correction across A2 C1 D1 F1"
git push origin main
```

Single-line commit per zsh history-expansion lesson.

---

## Mac Compatibility (Carry Forward)

Use the patterns from v2 + v3 ship scripts:
- `stat -f %z 2>/dev/null || stat -c %s` for size checks
- `python-pptx` install check with `--break-system-packages` fallback
- Single-line commit messages
- Heredoc with `<< 'EOF'` if multi-line strings needed (prevents shell expansion)

---

## What Ash Will Check On Review

After ship, Ash will spot-check on Mac in PowerPoint:

1. **Slide 4 (A2)**: 73% Calderon hero still present + new modality progression strip showing three-state framing
2. **Slide 13 (C1)**: Modality × timepoint matrix is visual hero, perturbation conditions block uses softened framing, "No public dataset has this combination" subtitle visible
3. **Slide 16 (D1)**: Q1'27 milestone reads "Phase 2 VDJ on" (not "Phase 2 phospho on")
4. **Slide 21 (F1)**: Flywheel TEMPORAL MULTI-OMICS pillar shows phospho lavender-emphasized + 5 timepoints + ATAC×2 + VDJ Phase 2; closing line shifted to Q3 2026 imminent framing
5. **Regression on locked slides** (A1, A3, A4, A5, B1, B2, B3, C2, D2, E1): unchanged from v3
6. **File size**: should land near v3's 3.0MB ± 0.2MB

If any zone visibly off, single fix prompt before moving to Step 5.

---

## What's Out Of Scope For This Combined Step

- Speaker notes glossary expansion (Step 5 — separate iteration after v4 ships)
- A5 SVG changes (speaker notes only, updated in Step 5)
- A1 speaker notes filling (Step 5)
- Pagination unification (Phase 4A visual polish, post-feedback)
- Font sizing audit (Phase 4A, post-feedback)
- Color coding restoration on A5 equation (Phase 4A)
- Any new content beyond the corrections doc

---

## Risks To Flag

1. **C1 visual is the largest change**. The modality × timepoint matrix is new visual territory. If matrix rendering looks crowded or unbalanced, prioritize **clarity over information density** — drop the VDJ Phase 2 row if it makes the Phase 1 matrix harder to read, mention VDJ in the subtitle instead.

2. **A2 modality progression strip placement**. Below the 73% Calderon hero OR beside it — depends on existing A2 visual layout. Don't disturb the hero result. If layout doesn't accommodate a side-by-side strip, place below.

3. **F1 closing line length** (~150 chars new version). Test wrap at slide-fill scale. If two-line wrap looks cramped, trim: "Phase 1 QuRIE-seq (Q3 2026) measures RNA + Protein + Phospho — the combination no public dataset has."

4. **D1 milestone label change is mechanical** but verify the diamond's text fits the new label width. "Phase 2 VDJ on" is shorter than "Phase 2 phospho on" so should be fine.

5. **Collision-guard tightening from earlier work applies**. `min_gap=2` is the new standard; helper smoke at `min_gap=0` should be clean.

6. **Don't rebuild A5 SVG**. A5 only changes via speaker notes in Step 5. A5 SVG should be byte-identical to v3 in v4.

---

## After This Lands

**Next iteration = Step 5: Speaker notes glossary expansion across all 14 slides** (substantial drafting work, ~3-4 hours my drafting + Cowork mechanical pptx v5 rebuild).

Then ship pptx v5 to Kinga + Jan for content review.

Then Phase 4A visual polish (font sizing audit, pagination unification, color coding restoration on A5) after their feedback.
