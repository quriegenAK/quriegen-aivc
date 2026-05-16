# Competitive Landscape Research — Foundation For Appendix Section F

**Owner**: Cowork (execution, research-heavy)
**Estimated time**: 2.5-3 hours total
**Strategy**: Extract → Research → Surface (3 steps, no slide synthesis yet)
**Goal**: produce the research foundation for a new appendix slide F1 — "Competitive Positioning"

---

## Context

Phase 1+2+3 of the technical appendix shipped 12 content slides + 18-slide pptx. New ask: add a competitive positioning slide.

**Slide placement** (locked): New Section F appended after Section E. Final appendix structure becomes:

```
Cover · A (4) · B (3) · C (2) · D (2) · E (1) · F (1)
```

13 content slides total + 1 cover + 6 dividers = 20 slides in final pptx.

**This task is research only.** Cowork produces the competitive landscape document. Ash + Claude then write the F1 content spec strategically based on the research. A separate later iteration generates the F1 SVG and updates the pptx.

---

## The 3-Step Task

### Step 1 — Extract Competitors From Kinga's Slide 9

**Input**: `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx`, slide 9
**Output**: `docs/deck/research/slide9_competitive_extraction.md`
**Time**: ~15 min

Open Kinga's slide 9 (the competitive checkbox matrix). Extract programmatically:

1. **Competitor names listed** — exact company names as Kinga has them
2. **Comparison dimensions** — the columns/criteria of her checkbox matrix (e.g., "Multi-omics ✓", "Causal-ready ✓", "Virtual cell ✓", etc.)
3. **Kinga's existing positioning** — what she claims about each competitor (check vs gap) and what she claims about Quriegen
4. **Visual treatment** — note color coding, icons, any visual hierarchy

Use `python-pptx` to programmatically open the .pptx and extract slide 9's text content + table structure. If table structure is complex, also use unzip on the .pptx and parse `ppt/slides/slide9.xml` directly.

Document format:

```markdown
# Slide 9 Competitive Extraction (Kinga's deck)

**Source**: docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx, slide 9
**Extracted**: 2026-05-15

## Competitors listed

1. [Company name 1]
2. [Company name 2]
...

## Comparison dimensions (columns)

1. [Dimension 1]
2. [Dimension 2]
...

## Kinga's matrix as-is

| Company | Dim 1 | Dim 2 | ... |
|---|---|---|---|
| Quriegen | ✓ | ✓ | ✓ |
| Competitor 1 | ✓ | ✗ | partial |
...

## Notes on Kinga's framing
- [Observations about what she emphasizes]
- [Any obvious gaps in the competitive set]
```

**Don't research yet. Just extract what's on her slide today.**

---

### Step 2 — Research Each Competitor

**Input**: extracted competitor list from Step 1
**Output**: `docs/deck/research/competitive_landscape_2026_05.md`
**Time**: ~2 hours

For each competitor extracted in Step 1, research and document:

#### Required fields per competitor

```markdown
## [Company Name]

**URL**: [primary website]
**Founded**: [year]
**Funding stage**: [seed / A / B / public — most recent round]
**Recent valuation or raise**: [public info only, "N/A" if unknown]
**Primary public claim**: [one sentence — what they pitch]

**Modality coverage**:
- RNA: [yes/no]
- ATAC: [yes/no]
- Protein: [yes/no]
- Phospho: [yes/no]
- VDJ: [yes/no]
- Other: [list]

**Data strategy**:
- Public data only? [yes/no]
- Proprietary wet lab? [yes/no — details on scale if known]
- Partnerships for data? [yes/no — who]

**Model architecture**:
- Foundation model? [yes/no/unclear]
- Architecture type: [transformer / graph / mechanistic / hybrid / unknown]
- Modalities trained on: [list]
- Public model weights? [yes/no]

**Validation evidence**:
- Peer-reviewed publications? [count, top venues]
- Cross-corpus generalization shown? [yes/no/unclear]
- Perturbation prediction validation? [yes/no/unclear]
- Specific benchmarks they've published numbers on: [list]

**Differentiation they would claim against Quriegen**:
- [3-5 specific claims they would make]

**Gap vs Quriegen (honest)**:
- What they have that we don't: [list]
- What we have that they don't: [list]
- What we're roughly equivalent on: [list]

**Sources** (every claim above must trace to one):
- [URL or paper citation 1]
- [URL or paper citation 2]
- ...
```

#### Research tooling

Use web search aggressively. Suggested query patterns:
- `"[company name]" foundation model single-cell`
- `"[company name]" multi-omics platform`
- `"[company name]" perturbation prediction`
- `"[company name]" wet lab data generation`
- `site:nature.com "[company name]"` for publications
- `site:bioRxiv.org "[company name]"` for preprints
- LinkedIn for team size, recent hires (signal of strategic direction)

For each competitor, target **5-10 unique sources** before writing the entry. Don't write from memory or general knowledge.

#### What "competitor" means here

Include in the research:
1. **Direct competitors** named on Kinga's slide 9
2. **Adjacent competitors** Kinga didn't list but you encounter during research that fit our space (PBMC + perturbation + foundation model). Add a "Competitors not on Kinga's slide 9" section for these.
3. **Don't include** companies that are obviously different (e.g., pure drug discovery without a foundation model, or pure target ID without wet lab, or pure CRO services)

If you find 3 adjacent competitors Kinga missed, that's signal worth surfacing — but don't expand the list beyond reasonable scope. Aim for **8-15 competitors total** in the final research doc.

---

### Step 3 — Surface Findings, Don't Synthesize The Slide

**Input**: completed competitive landscape doc
**Output**: short summary section appended to the research doc
**Time**: ~10-15 min

After completing Step 2, append a synthesis section to `competitive_landscape_2026_05.md`:

```markdown
## Synthesis — Patterns Worth Noting

### Where the competitive set converges
- [3-5 capabilities most competitors have/claim]

### Where the competitive set diverges
- [3-5 capabilities only some have, with named competitors per cluster]

### Capabilities NO competitor currently has (our defensible territory)
- [3-5 items with sources confirming the gap]

### Capabilities multiple competitors claim that we lack (honest gaps)
- [3-5 items with sources showing competitor claims]

### Quriegen's strongest defensible angles (ranked, with reasoning)
1. [Strongest angle] — Why: [reasoning]
2. [Second] — Why: [reasoning]
3. [Third] — Why: [reasoning]

### Open questions for Ash + Claude (strategic decisions, not research)
- [E.g., "Three competitors claim 'causal-ready' but interpretations vary widely — should our slide define what 'causal' means?"]
- [E.g., "Wet-lab scale comparisons are hard — competitor X publishes cell counts, others don't. Should we lead with our 500K-cell Phase 1 number?"]
```

**Do NOT write the slide content spec.** Don't suggest headlines. Don't draft body bullets. Don't propose a visual layout. That's Ash + Claude's strategic work after seeing the research.

The synthesis is **observations and patterns**, not slide design.

---

## Hard Requirements

### Honesty over marketing

If a competitor has a capability Quriegen lacks, document it honestly. The slide we eventually write may choose not to emphasize that point, but the **research document must show the full landscape including our gaps**. The point of this work is to identify our **real defensible differentiation**, not invent one.

A research doc that only shows our strengths is useless. A research doc that shows both strengths and gaps is the foundation for a credible "why us" slide.

### No fabricated claims

Every factual claim about a competitor must trace to a cited source (URL or paper). If you can't find a source for something, the claim doesn't appear — document the unknown as "unclear" instead.

Example of good vs bad:

❌ "Recursion uses transformer-based architecture for perturbation prediction." (no source = guess)
✅ "Recursion's published Phenom-1 model uses [arxiv 2024.xxxx](https://arxiv.org/...) with vision-transformer backbone for cell painting." (specific publication cited)
✅ "Architecture type: unclear from public information; published materials describe high-level approach but not weights or model card." (honest unknown documented)

### No competitor cherry-picking

Include even competitors who are clearly stronger than us in some dimension. The research doc must be defensible if an investor reads it. If we omit a strong competitor, that's a credibility risk during diligence.

### Date-stamp everything

Every entry includes "Last researched: 2026-05-15" so future iterations can refresh stale data. Competitive landscape moves fast — research has a shelf life of ~3 months.

---

## Deliverable Sequence

Three commits in sequence:

```bash
# Commit 1: Extraction
git add docs/deck/research/slide9_competitive_extraction.md
git commit -m "docs(deck): extract competitor list from Kinga's slide 9

Programmatic extraction of competitor names + comparison dimensions
from QurieGen_SEED_ROUND_05_2026_new.pptx slide 9. Source for
upcoming competitive landscape research."
git push origin main

# Commit 2: Research (the big one)
git add docs/deck/research/competitive_landscape_2026_05.md
git commit -m "docs(deck): competitive landscape research May 2026

Per-competitor entries covering: URL, funding stage, modality
coverage, data strategy, model architecture, validation evidence,
differentiation claims, honest gap analysis vs Quriegen.

All claims sourced. Includes competitors named on Kinga's slide 9
plus adjacent competitors discovered during research.

Foundation for upcoming F1 appendix slide on competitive positioning."
git push origin main

# Commit 3: Synthesis (appended to same file as commit 2, OR new file)
git add docs/deck/research/competitive_landscape_2026_05.md
git commit -m "docs(deck): competitive synthesis - patterns + defensible angles

Patterns observed across competitor set:
- where landscape converges (table-stakes capabilities)
- where it diverges (differentiation axes)
- white space (capabilities nobody has)
- honest gaps (capabilities competitors have, we don't)
- Quriegen's defensible angles ranked

Open strategic questions surfaced for Ash + Claude. Synthesis is
observations, not slide design. F1 content spec written separately."
git push origin main
```

Pattern A (3 separate commits) gives clean git archaeology — the extraction, the research, and the synthesis are distinct work products. Pattern B (single batch commit) is fine if Cowork prefers, but the 3-step structure must be visible in the file structure.

---

## What's Out Of Scope For This Task

- **Writing the F1 content spec** — that's Ash + Claude strategic work after research lands
- **Generating the F1 SVG** — separate later Cowork iteration
- **Updating the pptx** — separate later iteration
- **Modifying any existing slide** — A1-E1 stay locked
- **Phase 4 polish on existing slides** — parked
- **Workstream 2 (Kinga slides 8/37 content integration)** — parked

---

## Risks To Flag

1. **Research depth varies by competitor** — established public companies (Recursion, Tempus) have rich source material; stealth-mode startups may have nothing public beyond a landing page. Document the asymmetry honestly. For low-info competitors, the entry may be short and explicitly flag "limited public information."

2. **Competitor claims vs competitor reality** — companies overclaim. Differentiate "they claim X" (in their marketing) from "they demonstrably have X" (with peer-reviewed evidence). The research doc should track both.

3. **Web search reliability** — some competitor websites are SEO-driven and don't surface the actual technical depth. Cross-reference with their LinkedIn (team), their publications (real work), their press releases (claims). Three independent sources per claim where possible.

4. **Timing on web search** — competitive landscape as of 2026-05-15. Don't include "could potentially announce X in 2027" speculation; only document what they have/claim today.

5. **Honest gap section is the most important** — if Cowork writes a research doc where Quriegen wins on every dimension, the research is broken. Real competitive analysis finds at least 1-2 dimensions where we're roughly equivalent or behind. Surface them.

6. **Time budget management**: if Step 2 is running over 2.5 hours, ship what's been done (8-10 competitors deeply researched) rather than 15 competitors superficially. Depth > breadth.

---

## After This Lands

Ash + Claude review the research doc and produce:

1. **F1 content spec**: `docs/deck/content/F1_competitive_positioning.md` (same pattern as the existing 12 content specs — headline, sub-headline, body, visual spec, speaker notes, source data, etc.)

2. **F1 SVG**: separate Cowork iteration generates the SVG following the locked visual style (probably amber/orange accent — distinct from existing A/B/C/D/E section accents)

3. **Updated pptx**: re-run `_build_appendix_pptx.py` with the new slide + new Section F divider → 20-slide final deck

4. **Phase 4 polish**: AFTER F1 lands, hero-slide polish pass via Claude Design

---

## Confirmation Before Starting

If Cowork hits ambiguity, surface it before continuing:
- Competitor name spelling/identity unclear → confirm with Ash
- Source quality questionable → document the doubt, don't paper over
- Research running over time budget → ship what's done with explicit "more competitors deferred" note

Don't fabricate. Don't speculate. When in doubt, write "unclear from public sources" — that's a useful data point itself.
