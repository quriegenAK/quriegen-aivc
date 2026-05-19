# Slide D2 — Seed Allocation: Where The $10M Goes

- **Maps to Kinga's deck**: Extends slide 17 (use of funds) with technical execution breakdown
- **Section**: D — Roadmap + Budget (closing slide)
- **Visual lead**: Allocation breakdown chart (stacked bar OR sankey) with milestone anchors
- **Status**: Draft v1 — ESTIMATED allocations pending Kinga's actual numbers

---

## Headline

**$10M seed → 10 quarters of platform execution.**

(Alternative: *"Every dollar mapped to a milestone."*)

(Alternative: *"Where the seed goes — and what it ships."*)

---

## Sub-headline (one line under headline)

Allocation aligned to the roadmap on slide D1. Wet lab and AI/ML team are the dominant spend categories — together representing the data engine and the modeling engine that make QurieSeq the moat.

---

## Body content (3 bullets max)

- **~40% wet lab (Phase 1 delivery + Phase 2 prep)**: ~$4M. Funds QuRIE-seq Phase 1 delivery (Q3 2026, includes integrated phospho-proteomics panel as part of QuRIE-seq protocol) and Phase 2 readiness (Q1-Q2 2027, VDJ panel + donor scale-up to 20). Equipment, reagents, donor procurement (Sanquin), CITE-seq + phospho antibody panels, BTK + JAK + additional inhibitor procurement, ATAC integration pipeline. Phase 1 phospho is not a separate prep cost — it's part of the QuRIE-seq line.

- **~25% AI/ML team + compute**: ~$2.5M. Engineering scale-up (3-4 ML engineers including current team), compute infrastructure (BSC cluster access + cloud burst capacity for training peaks), GPU allocation for Stage 3a/3b/3c training, evaluation infrastructure, MLOps tooling.

- **~35% wet lab team + BD + G&A + IP**: ~$3.5M. Wet lab scientists and technicians (biologists running QurieSeq experiments), business development for pharma partnerships, IP filings on the AIVC architecture + QurieSeq protocol, regulatory consulting, office + operational overhead.

---

## Visual spec (allocation breakdown)

A two-part visual:

**Top — allocation breakdown** (stacked horizontal bar OR donut):

```
$10M Seed Round Allocation
──────────────────────────

Wet Lab (Phase 1 + 2 prep)           ████████████████████░░░░░░░░░░░  40%   $4.0M
AI/ML Team + Compute                  █████████████░░░░░░░░░░░░░░░░░░  25%   $2.5M
Wet Lab Team                          ███████░░░░░░░░░░░░░░░░░░░░░░░░  15%   $1.5M
Business Development                  █████░░░░░░░░░░░░░░░░░░░░░░░░░░  10%   $1.0M
G&A + IP + Legal                      █████░░░░░░░░░░░░░░░░░░░░░░░░░░  10%   $1.0M
                                      ─────────────────────────────────────────
                                                                       100%   $10M
```

**Bottom — spend → milestone mapping** (3-card row):

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│ DATA ENGINE            │  │ MODEL ENGINE           │  │ COMMERCIAL BACKBONE    │
│ $5.5M (55%)            │  │ $2.5M (25%)            │  │ $2.0M (20%)            │
│                        │  │                        │  │                        │
│ • Phase 1 delivery     │  │ • Stage 3a/3b/3c       │  │ • Pharma BD pipeline   │
│   Q3 2026              │  │   training (3 stages)  │  │   2027 onwards         │
│ • Phase 2 VDJ panel    │  │ • BTK+JAK demo         │  │ • IP filings on        │
│   + 20-donor scale     │  │   eval Q4 2026         │  │   architecture +       │
│   Q1-Q2 2027           │  │ • Stage 4 + 5          │  │   QurieSeq protocol    │
│ • Phase 3 wet lab      │  │   platform extensions  │  │ • Regulatory readiness │
│   Q3 2027+             │  │   2027-2028            │  │   Stage 5 (2028)       │
│ • CITE-seq + inhibitor │  │ • Compute infra        │  │ • Office + G&A         │
│   procurement          │  │   (BSC + cloud burst)  │  │                        │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
```

---

## Notes for design

- **The stacked bar / donut is the slide.** Use clear distinct colors per category — wet lab (warm), AI/ML (cool), team/BD/G&A (neutral).
- **Make the 25% AI/ML callout visually emphasized** — investors care that the model gets serious investment. ~$2.5M is non-trivial for a seed-stage AI biotech.
- **The 3-card grouping** (Data Engine / Model Engine / Commercial Backbone) reframes the allocation strategically. Investors think in terms of "where does the moat come from?" — data + model are the moat-generating spend; commercial backbone is supporting infrastructure.
- **Flag ESTIMATED status visually** — small footnote: "Final allocation subject to Phase 1 readiness assessment. See speaker notes for budget assumptions."
- **No charts within the cards** — keep them text-only to avoid visual noise competing with the top breakdown.

---

## Why this slide matters

D2 closes the budget question that every seed investor asks: **"How do you spend $10M, and what does it ship?"**

Three things it earns:

1. **Spending discipline**: 55% to data/model engine = competitive moat investment. Investors at the seed stage want to see capital flowing into things that compound (data and IP), not burn (rent and software).
2. **Allocation honesty**: We don't claim 80% to AI to look tech-forward. Wet lab is the dominant spend because QurieSeq is the moat — that's the right balance for a deep-tech biotech.
3. **Roadmap-budget coupling**: Every dollar maps to a specific milestone on D1. No floating "10% for opportunities" mystery bucket.

---

## Source data / claims

| Claim | Source |
|---|---|
| $10M seed round target | Kinga's deck slide 17 |
| Phase 1 QurieSeq cost basis (~$3-4M) | Industry standard for 5-donor multi-omics × 4-arm × 5-timepoint study at this scale |
| Phase 1 phospho panel cost (~$1M, included in Phase 1 wet-lab line) | ~17 antibody panels × validation + procurement + protocol dev. Phase 2 adds VDJ-specific reagents (separate $0.5-1M estimate, pending wet-lab spec). |
| AI/ML team cost (~$2-2.5M) | 3-4 ML engineers × 24 months at competitive salaries + benefits |
| Compute cost (~$0.5M) | BSC allocation + cloud burst for ~10 GPU-months/year over 24 months |
| BD cost ratio (~10%) | Industry standard for early-stage biotech BD pipeline |
| G&A ratio (~10%) | Industry standard for biotech operating company |

**IMPORTANT**: All numbers are MODEL-GROUNDED ESTIMATES pending Kinga's actual confirmed allocations. Final slide replaces with confirmed numbers before investor circulation.

---

## Speaker notes

### Three-state framing
- **Today**: $10M seed allocation budgeted for Phase 1 + Phase 2 wet lab + AI/ML team + compute + BD + IP + G&A.
- **Phase 1 (Q3 2026)**: ~$4M wet lab spend lands. Phase 1 includes proprietary phospho panel as part of QuRIE-seq integral protocol — NOT a separate Phase 2 line. ATAC integration pipeline + BTK/JAK inhibitor procurement + Sanquin donors.
- **Phase 2 (Q1 2027 onwards)**: Phase 2 onboarding (VDJ panel + 20-donor scale) within the wet-lab line. Pipeline 1 starts Q2'27 from drug pipeline budget.

### Technical glossary
**$10M seed allocation** — Total funding round size. Allocations are estimates pending Kinga's final confirmation.

**Wet lab (~40% / $4M)** — Phase 1 delivery + Phase 2 prep. Includes integrated phospho panel as part of QuRIE-seq protocol (Phase 1 cost, NOT Phase 2 separate cost). Equipment, reagents, donor procurement, antibody panels, inhibitor procurement, ATAC integration pipeline.

**AI/ML team + compute (~25% / $2.5M)** — 3-4 ML engineers + BSC cluster compute + cloud burst capacity + MLOps tooling.

**Wet-lab team scientists (~15% / $1.5M)** — Wet-lab biologists, technicians, lab management.

**Business development (~10% / $1M)** — Pharma partnerships, customer development.

**G&A + IP + legal (~10% / $1M)** — General + administrative + IP filings + legal.

**BSC cluster** — Barcelona Supercomputing Center compute allocation. Primary training infrastructure.

**MLOps** — Machine Learning Operations. Tooling for managing model training, deployment, monitoring.

**~17 antibody panel** — Phase 1 phospho antibody panel size. Cost ~$1M (panels + validation + procurement + protocol development). Included in Phase 1 wet-lab line because phospho is integral to QuRIE-seq Phase 1, NOT a Phase 2 readiness expense as the spec previously implied.

**IP filings** — Intellectual property protections on QuRIE-seq protocol family, architecture (decomposed readout, Neural ODE temporal), and platform integration.

### Diligence Q&A

**If asked: "Why so much to wet lab vs AI?"**

> Because the data is the moat. The AI model architecture is open-source-replicable in principle; the QurieSeq dataset is what no competitor has. ~40% to wet lab funds Phase 1 delivery and Phase 2 readiness — the experimental data that makes the model commercially defensible. The 25% to AI/ML is sufficient for a 3-4 engineer team plus compute; that's appropriate scale for a foundation-model platform, not an under-investment.

**If asked: "Is 25% enough for AI/ML?"**

> Yes, for the stage we're at. We're not training a 100B-parameter foundation model from scratch — we have an existing encoder validated at 73% cross-corpus, a Stage 3a adapter training in May 2026, and a clear architectural extension plan through Stage 5. The infrastructure investment is for execution (training jobs, eval pipelines, MLOps) not for greenfield research. At the next funding round, AI/ML spend grows materially as the team scales and compute requirements increase for Stage 4 + Stage 5.

**If asked: "What about the BTK + JAK inhibitor costs specifically?"**

> Inhibitor procurement is bundled in the Phase 1 wet-lab budget. Acalabrutinib (BTK), Ruxolitinib or similar JAK inhibitor, plus the other Phase 1 inhibitors (idelalisib, IKK16, rapamycin) — total cost is small relative to the donor procurement and CITE-seq panel costs. The line item is not visible on the high-level breakdown because it's a sub-component of the wet lab category.

**If asked: "Where does Series A money go?"**

> Out of scope for this deck — we don't anchor seed allocation to Series A planning. Series A is a separate fundraise that follows the Stage 3b BTK+JAK demo results and Phase 2 onboarding. The seed funds get us through Q4 2026 demos and Phase 2 readiness; Series A funds Phase 2 execution + drug pipeline establishment + Stage 5 platform extensions.

**If asked: "What happens if the seed runs short?"**

> Two contingency paths. First, the Phase 1 wet lab is the largest variable cost — if procurement runs over budget, we deliver fewer donors initially (e.g., 4 instead of 5) and scale up donor count with the next funding round. Second, BSC compute is currently subsidized academic access — if cloud burst costs run higher than expected, we shift more training to BSC and accept slightly slower iteration. Neither contingency materially affects the BTK+JAK demo timing or quality, but both extend timeline by ~1 quarter if triggered.

**If asked: "Why is G&A only 10%?"**

> Because we're a small operating team — Kinga (CEO), Thiago (CSO + wet lab lead), Ash (CTO + AI/ML), plus the engineering and wet lab staff. Office is shared / minimal, legal is mostly IP filings (one-time costs), accounting is outsourced. 10% G&A is appropriate for a 6-10 person biotech operating at seed scale.

---

## Investor framing (one-paragraph elevator)

> The $10M seed allocates 55% to the data + model engines that compound competitive advantage. Wet lab (~40%, $4M) funds QuRIE-seq Phase 1 delivery in Q3 2026 (4 modalities including integrated phospho-proteomics) and Phase 2 onboarding (VDJ + 20-donor scale) in Q1-Q2 2027 — the proprietary data that makes the platform defensible. AI/ML team + compute (~25%, $2.5M) funds the 3-4 engineer team executing Stage 3a through Stage 5, including BSC compute allocation and cloud burst capacity. The remaining 35% covers wet lab team scientists (~15%), business development for pharma partnerships (~10%), and G&A + IP + legal (~10%). Every dollar maps to a roadmap milestone on slide D1. These allocations are model-grounded estimates pending final confirmation from Kinga.

---

## What's NOT on this slide (intentionally)

- Salary specifics for any team member
- Specific equipment line items (sequencers, etc.) — too granular for investor view
- Cloud cost vs BSC breakdown — speaker notes if asked
- Inhibitor-by-inhibitor cost breakdown — speaker notes if asked
- Phase 3 wet lab funding source (likely Series A, not seed)
- IP filing costs by territory
- Specific BD targets / pharma partners — competitive risk

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — stacked horizontal bar OR donut chart + 3-card row.

**File output**: `docs/deck/assets/diagrams/D2_seed_allocation.svg`

**Followup prompt for Cowork** (when ready):
"Generate `D2_seed_allocation.svg` per spec in `docs/deck/content/D2_seed_allocation.md`. Top: stacked horizontal bar (or donut) showing $10M seed broken into 5 categories:
- Wet Lab Phase 1 + 2 prep: $4M (40%) — warm color
- AI/ML Team + Compute: $2.5M (25%) — accent/emphasized color
- Wet Lab Team: $1.5M (15%) — warm secondary
- Business Development: $1.0M (10%) — neutral
- G&A + IP + Legal: $1.0M (10%) — neutral

Bottom: 3-card row labeled DATA ENGINE ($5.5M total — wet lab + wet lab team) / MODEL ENGINE ($2.5M — AI/ML + compute) / COMMERCIAL BACKBONE ($2M — BD + G&A) with milestone bullets per card.

Footer note: 'Allocation estimates pending CEO confirmation' (small, subtle).

Output 1920×1080 viewBox. Use Kinga's deck color palette."

---

## Risk callouts (NOT to include on slide; for tracking only)

- All numbers ESTIMATED. Kinga has the actual breakdown — slide must be updated with her real figures before any investor circulation.
- Phase 1 wet lab cost basis assumed at industry-standard rates; Quriegen actual contracts (BSC, donor providers like Sanquin, antibody vendors) may yield different numbers.
- AI/ML team scale of 3-4 ML engineers includes current team; final headcount + comp depends on hiring market and timing.
- Phase 1 phospho panel cost (~$1M, included in QuRIE-seq Phase 1 wet-lab line) shifts the Phase 1 budget weight earlier than the spec previously implied. Phase 1 spend now includes phospho antibodies + validation; Phase 2 adds VDJ-specific reagents. Final budget disambiguation pending Kinga's confirmation.
- The "$5.5M data engine" framing combines two budget categories — strategically useful for narrative but requires the line items to remain visible.

---

## What's NEXT after D2 is committed

Move to **E1 (5-Year Trajectory — Pipeline + Clinical Maturation)** — closes the appendix. Extends the roadmap beyond Q4 2028 into the 2029-2031 horizon without IPO/exit mentions per Ash's strategic direction.
