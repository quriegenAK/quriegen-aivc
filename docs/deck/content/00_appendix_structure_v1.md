# AIVC Technical Appendix Deck — Master Structure

**Status**: Active spec
**Owner**: Ash Khan
**Created**: 2026-05-12
**Purpose**: Investor-ready technical appendix that extends Kinga's primary deck (slides 8, 9, 37) without editing it.

---

## Working Principle

- Kinga's main deck = commercial narrative (problem, solution, business, ask)
- This appendix = technical depth on demand (architecture, validation, roadmap, budget)
- Both delivered together. Appendix called on when investors do technical due diligence.

---

## 12-Slide Structure

### Section A — Architecture Depth (4 slides)
Anchored to Kinga slides 8 / 9 / 37.

| # | Title | Maps to Kinga | Visual Lead | Status |
|---|---|---|---|---|
| A1 | AIVC Foundation Model — System Architecture | Slide 37 | Full system diagram | Pending |
| A2 | Trimodal Encoder — The Frozen Substrate | Slide 8 | Encoder → latent space diagram | Pending |
| A3 | Decomposed Readout — How Synergy Generalizes | Slide 9 | 4-arm equation diagram | Pending |
| A4 | Temporal Dynamics via Neural ODE | Slide 8 | Latent trajectory plot | Pending |

### Section B — Validation Evidence (3 slides)
The proof points. Real numbers, not aspirational claims.

| # | Title | Maps to Kinga | Visual Lead | Status |
|---|---|---|---|---|
| B1 | Cross-Corpus Generalization — 73% Calderon | New | Bar chart + confusion matrix | Pending |
| B2 | Encoder Probe — ADAPTER_RECOMMENDED 0.57 | New | Per-class accuracy bars | Pending |
| B3 | BTK+JAK Mechanism Pre-Demo on Public Data | New | CD3E+CD4 synergy validation | Pending |

### Section C — QurieSeq Phase 1 (2 slides)
The moat in motion. Specific. Confirmed by Thiago.

| # | Title | Maps to Kinga | Visual Lead | Status |
|---|---|---|---|---|
| C1 | QurieSeq Phase 1 Experimental Design | Slide 9 (extend) | Time-course + 4-arm grid | Pending |
| C2 | BTK+JAK Headline Demo — Pre-Registered Eval | New | Pre-registered flow diagram | Pending |

### Section D — Roadmap + Budget (2 slides)
What happens next. Where money goes.

| # | Title | Maps to Kinga | Visual Lead | Status |
|---|---|---|---|---|
| D1 | Quarterly Roadmap Q3 2026 → Q4 2028 | Extends slide 14 | Gantt-style timeline | Pending |
| D2 | Seed Allocation — Where The $10M Goes | Slide 17 (extend) | Stacked bar / sankey | Pending |

### Section E — Strategic Horizon (1 slide)
5-year trajectory. Pipeline + clinical, no IPO mention.

| # | Title | Maps to Kinga | Visual Lead | Status |
|---|---|---|---|---|
| E1 | 5-Year Trajectory — Pipeline + Clinical Maturation | Slide 14 (extend) | Phase progression diagram | Pending |

---

## Slide Dependency Order (Content Creation Sequence)

1. **A1** (system architecture) — anchor slide everything else refers to
2. **A2** (encoder) — required before A3
3. **A3** (decomposed readout) — required before B2/B3/C2
4. **A4** (temporal) — required before C1/C2
5. **B1** (Calderon validation) — already in production
6. **B2** (encoder probe) — already in production
7. **B3** (BTK+JAK pre-demo) — pulls from A3/B2
8. **C1** (Phase 1 design) — pulls from Thiago specs
9. **C2** (BTK+JAK demo plan) — pulls from C1 + A3
10. **D1** (roadmap) — sequential locked at this point
11. **E1** (5-year horizon) — extends D1
12. **D2** (budget) — placeholder for Kinga's numbers

---

## Validation Sources (No Hallucinations)

Every technical claim in the deck must trace to one of:

- `docs/specs/stage3_part2_architecture_proposal_2026_05_06.md` — architecture spec v1.1
- `docs/memory/project_aivc_stage3_part1_verdict_2026_05_11.md` — Stage 3 Part 1 verdict
- `docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md` — Phase 6.5g.2 closure
- `docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md` — eval methodology
- Cowork-confirmed results from Stage 3a Day 1 + Day 2 PRs (commits 87d6a9a, aca6b09)
- Confirmed Phase 1 specs from Thiago (May 12 conversation)

---

## Tool Division Of Labor (Per Slide)

| Stage | Tool | Output |
|---|---|---|
| Content spec (markdown) | Claude Chat (Ash + Claude) | `docs/deck/content/<NN>_<slide_name>.md` |
| Diagram code (SVG/PNG generation) | Cowork or Claude Code | `docs/deck/assets/diagrams/<name>.svg` |
| .pptx slide generation | Cowork | `docs/deck/exports/aivc_appendix_v<N>.pptx` |
| Visual polish (hero diagrams only) | Claude Design | `docs/deck/assets/diagrams/<name>_design.svg` |
| Review + iteration | Claude Chat | Inline review |

---

## Visual Coherence Standards

To be locked once first diagram lands. Initial defaults:

- **Color palette**: Match Kinga's deck (extract from source/ once available)
- **Typography**: Match Kinga's deck
- **Iconography**: Reusable from `docs/deck/assets/icons/`
- **Diagram style**: Clean, technical, no decorative elements. Black/white with single accent color.

---

## What This Appendix Is NOT

- NOT a re-pitch of the commercial story (Kinga's deck does that)
- NOT a research paper (no methods detail beyond what's investor-relevant)
- NOT aspirational (every claim grounded in shipped code, validated results, or confirmed wet-lab plans)
- NOT exhaustive (12 slides, not 30)

---

## Next Action

Build content spec for **Slide A1 (AIVC Foundation Model — System Architecture)**.
Save to `docs/deck/content/A1_system_architecture.md`.
Then iterate forward through the dependency order.
