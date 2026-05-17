# Audit Task — Kinga Slides 8 + 37 Content vs Appendix

**Owner**: Cowork (execution, audit + light strategic analysis)
**Estimated time**: 45-60 min
**Strategy**: Extract → Cross-reference → Architectural placement
**Goal**: clear the prior-knowledge audit gap before Phase 4 polish begins

---

## Context

The technical appendix is feature-complete at 20 slides (v2 .pptx shipped at commit `7604343`). Before drafting Phase 4 polish, Ash flagged a question:

Kinga's slide 8 has a center-piece showing **prior knowledge tools and info**. Did our research/architecture work capture those? Are they reflected in any current appendix slide? And — strategically important — for prior-knowledge components we don't use yet but will use in future phases, **where do they sit architecturally and when do they activate?**

This audit closes that gap before Phase 4 begins. Otherwise Phase 4 polish risks being spent on slides that overlap with Kinga's deck OR missing prior-knowledge content that should anchor a specific slide.

---

## The 3-Layer Task

### Layer 1 — Extraction

**Input**: `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx`, slides 8 and 37
**Output**: `docs/deck/research/kinga_slides_8_37_extraction.md`
**Time**: ~10-15 min

Programmatically open Kinga's pptx (same pattern as the F1 competitive extraction earlier). Extract:

**For each of slide 8 and slide 37**:
- Slide title
- All text content (bullets, body text, annotations)
- Center-piece content (often a diagram with labels)
- Any visual element labels (databases, tools, methods, references named)
- Source citations or version markers if any
- Annotations Ash may have added (per Ash's note: "I added some stuff there")

Use `python-pptx` for structured text + `unzip` on the .pptx → `ppt/slides/slide8.xml` / `slide37.xml` for any text that python-pptx misses (especially text inside grouped shapes, text frames in custom layouts, or embedded chart annotations).

Document format:

```markdown
# Kinga Slides 8 + 37 — Content Extraction

**Source**: docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx
**Extracted**: 2026-05-16

---

## Slide 8

**Title**: [as-is]
**Position in deck**: [if discernible from context]

### Body content
- [bullet 1]
- [bullet 2]
- ...

### Center piece / diagram labels
- [item 1]
- [item 2]
- ...

### Annotations / Ash-added content
- [if any detected]

### Source citations
- [if any]

---

## Slide 37

[Same structure]

---

## Notes on extraction
- [Any items that were ambiguous, e.g., text in grouped shapes hard to extract cleanly]
- [Any visual content that doesn't translate to text but is conceptually load-bearing — e.g., flow arrows, color-coded categories]
```

**Don't analyze yet. Just extract what's there.**

---

### Layer 2 — Cross-reference To Appendix

**Input**: Layer 1 extraction + 13 appendix content specs in `docs/deck/content/`
**Output**: Append a `## Cross-Reference Analysis` section to the same file
**Time**: ~15-20 min

For each content item extracted from slides 8 and 37, classify it against the 13 appendix slides:

```markdown
## Cross-Reference Analysis

### Items from slide 8

| Item from Kinga slide 8 | Appendix coverage | Classification |
|---|---|---|
| [Item 1, e.g. "scGPT pretrained encoder"] | A2 (mentions encoder pretraining); not explicitly named | **Missing-but-belongs** → recommend adding to A2 speaker notes |
| [Item 2, e.g. "Reactome pathway database"] | Not covered in any appendix slide | **Missing — placement TBD** |
| [Item 3, e.g. "Mimitou 2021 DOGMA-seq dataset"] | A2 + B1 explicit | **Covered** |
| [Item 4, e.g. "Quriegen platform overview narrative"] | E1 covers strategic horizon, not architectural overview | **Kinga-only — appropriate as primary deck content** |
| ... | ... | ... |

### Items from slide 37

[Same table structure]
```

**Three classifications**:

- **Covered**: appendix already has this item; no action needed
- **Missing-but-belongs**: appendix should reflect this; identify proposed location (existing slide expansion OR new slide)
- **Kinga-only**: appropriate where it sits in Kinga's deck, no appendix mirror needed (e.g., investor-narrative framing that doesn't need technical depth)

**Discipline**: don't over-classify "missing-but-belongs." If a Kinga-slide item is investor narrative rather than technical detail, it's correctly Kinga-only. The appendix is technical depth; Kinga's deck is strategic narrative. They serve different purposes.

---

### Layer 3 — Architectural Placement For Prior Knowledge

**Input**: Layer 2 classifications + architecture spec v1.1
**Output**: Append a `## Architectural Placement For Prior Knowledge` section to the same file
**Time**: ~15-20 min

For every item from slides 8 and 37 classified as **"Missing-but-belongs"** OR every item that's a **prior-knowledge component** (database, pretrained model, pathway annotation, reference atlas, foundation method) — **whether we use it today or plan to use it in future phases** — answer four questions:

```markdown
## Architectural Placement For Prior Knowledge

For each prior-knowledge component (used today OR planned for future phases):

### [Component name, e.g. "Reactome pathway database"]

**What it is**: [1-line description]

**Where it sits in the architecture stack** (choose one):
- [ ] Training data input (feeds encoder pretraining or perturbation training)
- [ ] Pretrained encoder weights / embeddings (used as initialization or frozen substrate)
- [ ] Pathway annotation layer (post-prediction enrichment / interpretation)
- [ ] Evaluation reference (gold-standard for benchmarking)
- [ ] Biological prior for architecture decisions (informs model design but not runtime input)
- [ ] Other: [specify]

**When it activates** (which Stage or Phase):
- [ ] Stage 3a (current)
- [ ] Stage 3b (BTK+JAK demo, Q4 2026)
- [ ] Stage 3c (multi-perturbation expansion)
- [ ] Stage 4 (VDJ + 20-donor scale, 2027)
- [ ] Stage 5 (causal-readiness, 2028)
- [ ] Phase 1 wet-lab integration (Q3 2026)
- [ ] Phase 2 wet-lab extension (2027)
- [ ] Phase 3 wet-lab (2027+)
- [ ] N/A — already in use today

**Which appendix slide should anchor it** (where it gets named/described):
- [Suggested existing slide ID, e.g. "A2 — extend body bullet 2"]
- [OR "new slide needed — proposed: AX prior-knowledge stack"]

**Justification**: [1-2 sentences why this placement makes architectural sense]

---

[Repeat for each prior-knowledge component]
```

**This is the layer Ash specifically asked for**: "we have to figure out the perfect spot and time for it architecturally where ever it fits."

Don't speculate beyond what the architecture spec and content specs support. If a component's architectural placement is unclear from the existing materials, flag it as **"placement TBD — needs architectural decision from Ash + Claude"** rather than guessing.

---

## Hard Requirements

### No invention

Every item documented must trace back to either:
- Kinga's slide 8 or 37 (Layer 1 extraction)
- Architecture spec v1.1 (Layer 3 placement)
- An existing appendix content spec (Layer 2 cross-reference)

If something appears on Kinga's slide that has no architectural basis in our spec, **document the discrepancy** rather than inventing a fit.

### No premature classification

Don't classify items as "Kinga-only" just to minimize follow-up work. If an item genuinely should be in the appendix but isn't, flag it as "Missing-but-belongs." Phase 4 will address those gaps.

### Distinguish current-use vs future-use

Some prior-knowledge components are in use today (DOGMA-seq dataset is in A2 + B1). Others are planned for future phases (e.g., specific pathway annotations that activate in Stage 4 or 5). Mark each clearly. Phase 4 polish on a future-use component is different from polish on a current-use component.

### Date-stamp the document

Every file Cowork ships includes `**Extracted/analyzed**: 2026-05-16` so future iteration can refresh if Kinga's deck changes.

---

## Deliverable

Single commit covering the audit doc:

```bash
git add docs/deck/research/kinga_slides_8_37_extraction.md
git commit -m "docs(deck): audit Kinga slides 8+37 prior-knowledge content"
git push origin main
```

Single-line commit message to avoid zsh history-expansion issues.

---

## What Ash + Claude Do After Audit Ships

Convene to review the doc together. For each item classified as **"Missing-but-belongs"** or **"placement TBD"**:

1. **Confirm or override** Cowork's proposed placement
2. **Decide phase of action**:
   - Add to existing slide during Phase 4 polish (small change to content spec + speaker note expansion)
   - Add as new appendix slide (Phase 5 if scope warrants)
   - Defer to a future iteration (specify trigger condition)
   - Leave Kinga-only (no appendix change needed)

3. **Document decisions** in a follow-up commit:
   `docs/deck/research/prior_knowledge_placement_decisions_2026_05_16.md`

This becomes the **input to Phase 4 polish prompt** — instead of generic "polish hero slides," Phase 4 has specific content additions per slide based on the audit.

---

## What's Out Of Scope

- Modifying Kinga's source .pptx (read-only)
- Adding any appendix content during this task (audit only)
- Phase 4 polish execution (separate prompt after audit)
- F1 modifications (locked at commit `7e6c31c`)
- Re-running pptx assembly (no content changes yet)

---

## Risks To Flag

1. **Kinga's slide 8 center piece may be a diagram with non-text labels** — visual elements that `python-pptx` doesn't extract cleanly. If text extraction misses content, document the gap explicitly: "Slide 8 contains a diagram with labeled boxes — text labels extracted as best-effort, but visual hierarchy may not transfer to markdown. Manual review of the rendered slide recommended."

2. **Ash-added content may be in text-frame overlays rather than the base slide** — some PowerPoint annotations live in floating text boxes that may be extracted as separate elements. Group them under "Annotations / Ash-added content" rather than mixing with base-slide bullets.

3. **Slide 37 is far into Kinga's deck** — may be a summary slide, an appendix-style slide in her primary deck, or a "what's next" placeholder. Extract first, classify second; don't assume its purpose.

4. **Some prior-knowledge components may not have clear architectural placement** in the current spec. That's a useful finding — flag as "needs architectural decision" rather than forcing a fit.

5. **The audit doc may be long** (potentially 50-100 KB). That's expected for a 3-layer analysis. Don't try to compress at the cost of clarity.

6. **Some items may straddle classifications** — e.g., "scGPT" might be both a competitive reference (relevant to F1 speaker notes) AND a prior-knowledge component (potentially relevant to A2). Flag dual-relevance items explicitly rather than forcing single classification.

---

## After This Lands

Ash + Claude convene to:
1. Review audit doc
2. Make per-item placement decisions
3. Draft Phase 4 polish prompt with concrete content additions baked in (rather than generic "polish")

Estimated total path from audit ship to Phase 4 start: 30-45 min strategic conversation + 15-20 min drafting Phase 4 prompt.

Phase 4 becomes more concrete and less speculative because the audit eliminated the "do we cover prior knowledge?" question.
