# Slide D1 — Quarterly Roadmap: Q3 2026 → Q4 2028

- **Maps to Kinga's deck**: Extends slide 14 (roadmap / growth trajectory) with technical execution detail
- **Section**: D — Roadmap + Budget
- **Visual lead**: Quarterly Gantt-style timeline with milestone anchors
- **Status**: Draft — pending Ash review

---

## Headline

**11 quarters. 5 stages. Two drug pipelines. One coherent platform plan.**

(Alternative: *"From validated mechanism to drug pipeline, quarter by quarter."*)

(Alternative: *"What we ship, when, and why each milestone matters."*)

---

## Sub-headline (one line under headline)

Stage 3 ships against QurieSeq Phase 1. Stage 4 and 5 build the platform out as QurieSeq Phase 2 lands and drug pipelines establish — with every milestone tied to a specific quarter and dependency.

---

## Body content (3 bullets max)

- **Stage 3 (Q3 2026 – Q1 2027) — validation completes**: Stage 3a wraps in Q3 2026 (adapter trained on Mimitou, dress-rehearsal synergy demo). Stage 3b runs immediately as QurieSeq Phase 1 lands (Q3-Q4 2026) — the BTK+JAK headline demo. Stage 3c integrates phospho readouts in Q1 2027 as Phase 2 phospho panels arrive.

- **Stage 4 (Q2 2027 – Q4 2027) — platform extends**: VDJ encoder integration (T/B-cell repertoire), donor scale to 20 (Phase 2 wet-lab), cross-disease transfer evaluation. The platform graduates from "validated on Phase 1" to "production-ready across the immune system."

- **Stage 5 (Q1 2028 – Q4 2028) — pipeline + clinical readiness**: Causal-readiness layer for drug-target reasoning. First two internal drug pipelines establish (Q1-Q2 2027 → Q4 2028 timeline). Clinical translation framework: regulatory-grade provenance, computational diligence package, partnership-ready data architecture.

---

## Visual spec (the quarterly roadmap)

A horizontal Gantt-style timeline with quarters across the top and 4 swimlanes:

```
TIMELINE                Q3'26   Q4'26   Q1'27   Q2'27   Q3'27   Q4'27   Q1'28   Q2'28   Q3'28   Q4'28
                       ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
                       │     │     │     │     │     │     │     │     │     │     │

SWIMLANE 1             │█████████│
WET LAB                │ Phase 1   │█████████████████│
                       │ delivery  │ Phase 2 (phospho│█████████████████████████│
                       │           │  + VDJ + 20     │  Phase 3 (B-cell line +  │
                       │           │  donors)        │  disease samples)        │
                       │           │                 │                           │
                       │                                                         │
SWIMLANE 2             │█████│                                                  │
MODEL ARCHITECTURE     │ 3a  │█████████████│                                    │
                       │     │  Stage 3b   │█████│                              │
                       │     │  BTK+JAK    │ 3c  │██████████████│              │
                       │     │  HEADLINE   │phospho│  Stage 4  │██████████████│
                       │     │  DEMO ◀──── │   integ│  VDJ +    │ Stage 5      │
                       │     │             │        │  20 donor │  causal +    │
                       │                              │   scale │  clinical    │
                       │                              │         │  framework   │
                       │                                                         │
SWIMLANE 3             │                              │█████████│
DRUG PIPELINES         │                              │ Pipeline│██████████████│
                       │                              │ 1 starts│ Pipeline 2   │
                       │                              │         │ starts +     │
                       │                              │         │ Pipeline 1   │
                       │                              │         │ target valid │
                       │                                                         │
SWIMLANE 4             │█████████████████│            │█████████████████│
PUBLICATIONS &         │ Stage 3 verdict │            │ Stage 4 + 5     │
INVESTOR DEMOS         │ + BTK+JAK demo  │            │ benchmark +     │
                       │ deck-grade      │            │ peer-reviewed   │
                       │                 │            │ publication     │
                       └─────────────────────────────────────────────────────────┘
                        Q3'26   Q4'26   Q1'27   Q2'27   Q3'27   Q4'27   Q1'28   Q2'28   Q3'28   Q4'28
```

**Key milestone anchors (large markers on the timeline):**

```
◆  Q3 2026 — QurieSeq Phase 1 delivery: 4 modalities + BTK+JAK combo
◆  Q4 2026 — BTK+JAK ZERO-SHOT DEMO  (the headline)
◆  Q1 2027 — Phase 2 VDJ on, Stage 3c integration
◆  Q2 2027 — Drug pipeline 1 starts
◆  Q4 2027 — Stage 4 wraps (VDJ + 20 donors)
◆  Q2 2028 — Drug pipeline 2 starts, pipeline 1 target validation
◆  Q4 2028 — Stage 5 wraps (causal + clinical-ready)
```

**Rationale**: Phase 2's actual modality addition is VDJ, not phospho. Phospho is integral to QuRIE-seq and lands with Phase 1 in Q3 2026 (see C1 + A2 for the corrected three-state framing). The Q1'27 milestone is therefore VDJ on.

---

## Notes for design

- **The Gantt timeline is the slide.** Make swimlanes visually distinct (Wet Lab = warm tone, Model = cool tone, Pipelines = accent, Publications = neutral).
- **Q4 2026 BTK+JAK DEMO must be the visual anchor**. Make it stand out — bold, accent color, the only milestone with explicit "HEADLINE" callout. Everything else is supporting.
- **Dependencies should be visible** with subtle dotted arrows: Phase 1 → 3b; Phase 2 phospho → 3c; Phase 2 VDJ → Stage 4; Stage 5 causal layer connects to drug pipelines.
- **Use ◆ diamond markers** for milestone anchors at the top of the timeline — easy scan path.
- **Color the BTK+JAK demo bar in the model swimlane**. Make Q4 2026 unmistakable.

---

## Why this slide matters

D1 is where the **execution credibility lives**.

Without D1:
- C2's BTK+JAK Q3 2026 commitment looks isolated
- Stage 4 and Stage 5 are mentioned in other slides but never visualized
- Drug pipelines are aspirational without timing anchors

With D1:
- Every claim earlier in the deck (architecture, validation, demo plan) maps to a specific quarter
- Dependencies are explicit: Phase 1 unlocks 3b unlocks demo; Phase 2 unlocks 3c unlocks Stage 4 unlocks Stage 5
- Drug pipelines emerge with visible timing, anchored to Phase 2 readiness

The slide also **shows discipline**: 11 quarters of execution with concrete milestones rather than vague "we'll grow."

---

## Source data / claims

| Claim | Source |
|---|---|
| QurieSeq Phase 1 Q3 2026 delivery | Thiago confirmation, May 12 |
| QurieSeq Phase 2 Q1-Q2 2027 (phospho + VDJ + 20 donors) | QurieSeq roadmap (Phase 2 spec) |
| Stage 3a wraps Q3 2026 | Architecture spec v1.1, §6 |
| Stage 3b BTK+JAK demo Q4 2026 | Architecture spec v1.1, §5.1 |
| Stage 3c phospho integration Q1 2027 | Architecture spec v1.1, §6 |
| Stage 4 VDJ + 20 donors + cross-disease (Q2-Q4 2027) | Internal roadmap |
| Stage 5 causal-readiness + clinical framework (2028) | Internal roadmap — confirmed Ash strategic answers May 12 |
| Drug pipeline 1 starts Q1-Q2 2027 | Internal roadmap — confirmed Ash strategic answers May 12 |
| Drug pipeline 2 starts Q2 2028 | Internal roadmap projection |
| Phase 3 wet lab (B-cell line + disease samples) Q3 2027+ | Thiago wet-lab plan + L3 dataset strategy |

---

## Speaker notes

**If asked: "What's the dependency chain?"**

> Three chains. First, QurieSeq Phase 1 (Q3 2026) unlocks Stage 3b (Q4 2026, BTK+JAK demo). Without Phase 1 data, the demo slides to whenever data arrives. Second, QurieSeq Phase 2 (Q1-Q2 2027) unlocks Stage 3c phospho integration and Stage 4 VDJ. The Phase 2 data is what extends the platform from 3 to 5 modalities. Third, Stage 4 + the early drug pipeline work unlocks Stage 5's causal-readiness layer (Q1 2028+). Each major stage has one upstream dependency clearly identified.

**If asked: "What happens if Phase 1 slips?"**

> Stage 3b BTK+JAK demo slides with it — Q4 2026 → Q1 2027 if Phase 1 ships in Q4. Stage 3a (Mimitou-based) is independent and ships on schedule regardless, so the dress-rehearsal demo on public data is the contingency. Mid-roadmap (Stage 4 / Stage 5) is loose enough that 1-quarter slippage compresses without major restructuring. Two-quarter slippage starts compressing Stage 5; three-quarter slippage requires re-planning.

**If asked: "Why are drug pipelines so early — Q1-Q2 2027?"**

> Drug pipeline establishment isn't drug development from scratch — it's applying the validated model architecture to specific drug discovery questions. Phase 1 data + the BTK+JAK demo proves the model's combination-prediction capability. From that, the first pipeline is "use the model to identify combination targets in immune-driven diseases" (Q1-Q2 2027). Pipeline 2 is broader target discovery (Q2 2028). Neither pipeline is "Phase I clinical by 2028" — they're target identification and validation pipelines, with the longer clinical timeline running into Stage 6+.

**If asked: "Stage 5 'clinical translation framework' — what does that mean?"**

> Two things. First, regulatory-grade provenance: full audit trail of model training data, validation runs, version control on every model output. This is what clinical decision support requires. Second, computational diligence package: standardized documentation that any pharma partner or regulatory body can review to assess model outputs. We're not running clinical trials in 2028 — we're getting the platform clinical-trial-ready so partnerships and downstream pipelines can move on a credible timeline.

**If asked: "What's the 'causal-readiness' layer?"**

> The current decomposed-readout architecture (slide A3) is *correlationally* compositional — the synergy head learns the residual signal between combinations, which lets it generalize. Causal-readiness layers explicit causal structure on top: counterfactual queries ("what would CD4 T cells do if BTK were active but IL-2 receptor were not?"), intervention prediction, and target prioritization based on causal effect sizes. The biology stays the same; the inference machinery extends to support drug-target reasoning at the level pharma partners need.

**If asked: "Why don't we see Series A or financing milestones?"**

> Roadmap is technical execution. Funding events happen on a separate financial track; we don't anchor product roadmaps to fundraising calendars. The technical deliverables are what determine fundability, not the other way around.

**If asked: "What exactly do Phase 1 and Phase 2 mean here?"**

> Phase 1 and Phase 2 refer specifically to QuRIE-seq proprietary wet-lab data generation phases — not clinical trial phases, not company funding rounds, not model training stages. Phase 1 (Q3 2026) generates 5-donor, 5-timepoint dataset with 4 modalities (RNA + Protein + Phospho at all 5 timepoints; ATAC at t=0 and t=180). Phase 2 (Q1 2027 onwards) scales to 20 donors and adds VDJ as 5th modality. Model training stages (Stage 3a current, Stage 3b BTK+JAK demo Q4 2026, Stage 3c causal architecture Q1-Q2 2027, Stage 4 scale 2027, Stage 5 causal-ready 2028) are a separate framework — shown in the model swim lane of this Gantt. The unified quarterly view is what links wet-lab phases to model stages.

**If asked: "How does this 11-quarter view map to Kinga's 24-month trajectory on slide 8?"**

> Slide 8 compresses the same plan into a 4-phase visual for investor narrative. D1 is the canonical per-quarter detail with explicit milestone dependencies. Same plan, different visual decomposition for different audiences.

---

## Investor framing (one-paragraph elevator)

> The roadmap is 11 quarters from Q3 2026 through Q4 2028, organized across 4 swimlanes: wet lab, model architecture, drug pipelines, and publications/demos. The visual anchor is Q4 2026 — the BTK+JAK zero-shot demo, the platform's first investor-grade publication. Phase 1 QurieSeq data delivers in Q3 2026 (Thiago confirmed); Phase 2 adds phospho and VDJ in Q1-Q2 2027. Drug pipelines establish from Q1-Q2 2027 onward as the model graduates from validated-on-Phase-1 to production-ready. Stage 5 in 2028 layers causal-readiness on top — explicit support for drug-target reasoning and clinical translation framework. Dependencies are explicit: every model stage has a wet-lab dependency cleanly identified. Slippage in any one swimlane has clear contingency paths.

---

## What's NOT on this slide (intentionally)

- Specific financial milestones (Series A, B, etc.) — different slide territory
- Headcount targets per quarter — D2 budget slide handles
- Partnership / BD pipeline timing — Kinga's deck handles commercially
- Clinical trial timelines beyond Stage 5 framework — out of 2028 scope per Ash's strategic answers (no IPO or exit-prep mentions)
- Specific drug-pipeline target choices — competitive risk to publish in deck
- Public-data layer integration (L1/L2/L3) — handled in C1 + speaker notes

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — horizontal Gantt-style chart with 4 swimlanes.

**File output**: `docs/deck/assets/diagrams/D1_quarterly_roadmap.svg`

**Followup prompt for Cowork** (when ready):
"Generate `D1_quarterly_roadmap.svg` per spec in `docs/deck/content/D1_quarterly_roadmap.md`. Horizontal Gantt chart, 10 quarters wide (Q3'26 → Q4'28), 4 swimlanes stacked vertically:

Lane 1 (WET LAB) — Phase 1 delivery Q3 2026, Phase 2 Q1-Q2 2027 → Q2 2027, Phase 3 Q3 2027+
Lane 2 (MODEL ARCHITECTURE) — Stage 3a Q3 2026, Stage 3b Q3-Q4 2026 (highlight BTK+JAK demo Q4), Stage 3c Q1-Q2 2027, Stage 4 Q2-Q4 2027, Stage 5 Q1-Q4 2028
Lane 3 (DRUG PIPELINES) — Pipeline 1 starts Q1-Q2 2027 → Q4 2028, Pipeline 2 starts Q2 2028 → ongoing
Lane 4 (PUBLICATIONS & INVESTOR DEMOS) — Stage 3 verdict + BTK+JAK demo deck-grade Q4 2026, Stage 4+5 peer-reviewed publication Q3-Q4 2027

Dotted dependency arrows: Phase 1 → 3b; Phase 2 phospho → 3c; Phase 2 VDJ → Stage 4; Stage 4 + pipelines → Stage 5 causal layer.

7 ◆ diamond milestone markers at the top labeled with quarter + brief title.

Q4 2026 BTK+JAK DEMO milestone gets emphasis — bold accent color, marker stands out from the other 6.

Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- Phase 1 timing slippage cascades 1-quarter through the model swimlane.
- Drug pipeline timeline assumes Phase 1 + Phase 2 deliver on schedule. If Phase 2 slips substantially (>1 quarter), Pipeline 1's Q1-Q2 2027 start becomes optimistic.
- Stage 5's "clinical translation framework" is scoped intentionally vaguely — we're not committing to clinical trials in 2028, but to clinical-readiness infrastructure. If pressed, defer to speaker note.
- The roadmap as visualized assumes consistent BSC compute access through 2028. NVIDIA hardware backstop is in early discussion but not yet locked.

---

## What's NEXT after D1 is committed

Move to **D2 (Seed Allocation — Where The $10M Goes)** — closes Section D. Translates the roadmap into spending priorities. Uses my earlier estimated allocation pending Kinga's actual numbers.
