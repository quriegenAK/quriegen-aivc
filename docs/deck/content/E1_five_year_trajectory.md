# Slide E1 — 5-Year Trajectory: Pipeline + Clinical Maturation

- **Maps to Kinga's deck**: Extends slide 14 (growth trajectory) into the 2029-2031 horizon
- **Section**: E — Strategic Horizon (closing slide of technical appendix)
- **Visual lead**: Phase progression diagram — platform maturity stages from 2026 to 2031
- **Status**: Draft — pending Ash review

---

## Headline

**From validated platform to first-in-class candidates — 2026 to 2031.**

(Alternative: *"What 5 years of continuous data + model + drug pipeline integration looks like."*)

(Alternative: *"The platform compounds. The pipelines emerge. The clinical work matures."*)

---

## Sub-headline (one line under headline)

The 5-year arc has three distinct phases: platform validation (2026-27), platform extension + early pipelines (2027-28), pipeline maturation + clinical translation (2029-31). Each phase compounds on the last — data accumulates, model capability extends, pipelines progress toward clinical handoff.

---

## Body content (3 bullets max)

- **2026-2027 — Platform validation (Stage 3a/b/c)**: BTK+JAK headline demo (Q4 2026) on Phase 1 data (4 modalities including phospho, integral to QuRIE-seq Phase 1). Stage 3c causal architecture validation (Q1-Q2 2027) leverages Phase 1 phospho signal. VDJ + 20-donor cross-disease transfer (Q2-Q4 2027) via Phase 2. The model graduates from validated-on-Phase-1 to multi-modality, multi-disease production-ready. **Foundation locked.**

- **2028 — Platform extension + early pipelines (Stage 4/5)**: Causal-readiness layer for drug-target reasoning. Two internal drug pipelines actively running (Pipeline 1 from Q1-Q2 2027, Pipeline 2 from Q2 2028). Clinical translation framework — regulatory-grade provenance, computational diligence packages — built into the platform infrastructure. **Pipelines and clinical infrastructure mature in parallel.**

- **2029-2031 — Pipeline maturation + clinical translation**: First-in-class candidates emerge from internal pipelines. Pharma partnerships scale through the clinical translation framework. Pipeline 1 reaches target-validated stage; Pipeline 2 reaches lead-selection. Platform supports both internal R&D and external partners with regulatory-grade outputs. **The platform becomes the operating system for immune-system drug discovery.**

---

## Visual spec (the maturity progression diagram)

A horizontal timeline / phase progression with three major phases:

```
2026                  2027                    2028                    2029-2031
─────                 ─────                   ─────                   ─────────

█████████████████│   │█████████████████│      │█████████████████│      │██████████████████│
                                                                                          
  PHASE 1               PHASE 2                   PHASE 3                    PHASE 4
  ───────               ───────                   ───────                    ───────
  VALIDATION            EXTENSION                 MATURATION                 TRANSLATION
                                                                              
  Model proves           Platform extends         Pipelines mature           First-in-class
  on Phase 1 +           to 5 modalities          + causal-readiness         candidates
  Phase 2 data           + 20 donors +            layer                      emerge
                         cross-disease            
  Outputs:               transfer                 Outputs:                   Outputs:
  • BTK+JAK demo                                  • Pipeline 1               • Target-validated
  • Phospho integ        Outputs:                   target-validated          assets
  • VDJ + 20 donor       • Stage 4 publication    • Pipeline 2               • Pharma partnerships
                         • Stage 5 framework        active                     scale
                         • Pipeline 1 start       • Regulatory-grade         • Platform = OS
                         • Pipeline 2 start         provenance built          for immune-system
                                                                              drug discovery
```

**Below the phase progression — three "compounding" indicators**:

```
┌────────────────────────────────┐  ┌────────────────────────────────┐  ┌────────────────────────────────┐
│ DATA COMPOUNDS                 │  │ MODEL COMPOUNDS                │  │ CLINICAL INFRA COMPOUNDS       │
│                                │  │                                │  │                                │
│ Phase 1 (5 donors, 3 modal)    │  │ 3 modalities → 5 modalities    │  │ Regulatory-grade provenance    │
│ Phase 2 (20 donors, 5 modal)   │  │ Single donor → 20 donor scale  │  │ Computational diligence pkg    │
│ Phase 3 (B-cell + disease      │  │ Static → temporal Neural ODE   │  │ Audit trails + version control │
│         samples)               │  │ Correlation → causal-readiness │  │                                │
│                                │  │                                │  │                                │
│ Every quarter adds wet-lab     │  │ Every stage adds capability    │  │ Every milestone adds clinical  │
│ data to the training corpus    │  │ without re-architecting        │  │ partnership readiness          │
└────────────────────────────────┘  └────────────────────────────────┘  └────────────────────────────────┘
```

---

## Notes for design

- **Phase progression is the slide.** Use a consistent visual rhythm — 4 phases of equal-width bars, each labeled with its purpose and outputs.
- **2026 phase is darkest / most filled** (we know exactly what we'll ship); **2027 phase is fully colored** (high confidence); **2028 phase is colored but lighter** (planned but contingent on earlier success); **2029-2031 phase is outlined / lighter still** (directional, not literal).
- **The three "compounds" cards at the bottom** are critical — they translate the abstract "platform maturity" into three concrete compounding loops investors understand: data, model, infrastructure.
- **No specific numbers in this slide** beyond the phase years. E1 is structural, not numerical.
- **Color**: Use Kinga's deck palette consistently — earlier phases in primary brand color, later phases in lighter shades.
- **No IPO / exit / Series A mentions** anywhere. Per Ash's strategic direction.

---

## Why this slide matters

E1 closes the entire technical appendix with a **forward-looking but grounded view**. Three things it earns:

1. **Multi-year thinking**: Most seed-stage decks stop at 18-24 months. E1 shows 5 years of coherent execution without overpromising — investors who think long-term recognize the discipline.

2. **Compounding loops visible**: The three "compounds" cards (data, model, infrastructure) explain *why* the platform gets stronger over time, not just that it does. This is what justifies the moat thesis.

3. **No exit-driven thinking visible**: Per your direction, no IPO / no acquisition / no Series A timing. The slide stays focused on platform maturation and clinical translation infrastructure. Sophisticated investors find this refreshing — operational focus, not financial-engineering focus.

---

## Source data / claims

| Claim | Source |
|---|---|
| BTK+JAK demo Q4 2026 | Architecture spec v1.1, §5.1 + Thiago Phase 1 confirmation |
| Phospho integration Q1 2027 | QurieSeq Phase 2 spec (~17 antibody panels) |
| VDJ + 20-donor scale Q2-Q4 2027 | Internal roadmap |
| Causal-readiness layer 2028 | Internal roadmap — confirmed Ash strategic direction May 12 |
| Clinical translation framework 2028 | Internal roadmap — confirmed Ash strategic direction May 12 |
| Pipeline 1 timeline (Q1-Q2 2027 start → 2029 target validation) | D1 quarterly roadmap |
| Pipeline 2 timeline (Q2 2028 start → 2030 lead selection) | D1 quarterly roadmap |
| Phase 2 modality expansion (5 modalities) | QurieSeq Phase 2 spec |
| Phase 3 wet lab (B-cell line + disease samples) | Thiago L3 wet-lab plan |
| Data compounding (Phase 1 → Phase 2 → Phase 3) | QurieSeq roadmap |
| Model architecture extensibility | Architecture spec v1.1, §3.1 (frozen encoder, modular extensions) |

---

## Speaker notes

### Three-state framing
- **Today through Phase 1 (2026-2027)**: Stage 3a-3c validation completes on Phase 1 data. BTK+JAK demo Q4 2026. Stage 3c causal architecture validation Q1-Q2 2027 (Phase 1 phospho data). VDJ + 20-donor cross-disease transfer in Stage 4 Phase 2 (Q2-Q4 2027).
- **Phase 2 + Stage 4-5 (2027-2028)**: Platform matures to production-ready. Multi-modality, multi-disease. Drug pipelines start (Pipeline 1 Q2'27, Pipeline 2 Q2'28).
- **Beyond seed (2029-2031)**: Clinical translation framework. Causal-readiness for regulatory contexts. The 5-year horizon for the platform.

### Technical glossary
**5-year horizon** — Strategic outlook through 2031. Includes seed-funded execution (2026-2028) + post-Series-A clinical maturation (2029-2031).

**Stage 3c causal architecture validation** — Validates Neumann propagation + sparse GRN + direct-effect log-FC head on Phase 1 phospho-rich data. Q1-Q2 2027. Gated on Phase 1, not Phase 2.

**VDJ + cross-disease transfer (Stage 4)** — Phase 2 extension. VDJ as 5th modality (T/B-cell adaptive immune receptors). Cross-disease transfer = does the platform generalize across disease contexts (CLL, autoimmune, oncology). Q2-Q4 2027.

**Drug pipelines (Pipeline 1, Pipeline 2)** — Internal target identification → validation workflows. Pipeline 1 Q2 2027 (post Stage 3 verdict). Pipeline 2 Q2 2028.

**Clinical translation framework (Stage 5)** — 2028+. Explicit support for drug-target reasoning, regulatory-grade explanation, clinical decision support. Causal architecture is the foundation; Stage 5 adds clinical-readiness features.

**Multi-modality production-ready** — Platform has all 5 modalities (RNA + Protein + Phospho + ATAC + VDJ) operational across multiple diseases. Phase 2 + Stage 4 milestone.

**Pipeline 1 target ID → validation** — Discovery workflow. Identify candidate drug targets from platform predictions; validate via wet-lab. Pipeline 1 starts Q2 2027.

**Causal-readiness** — Platform produces causal inference (mechanism, "what does X cause?") not only predictive (response, "what happens after X?"). Stage 3c is the architectural commitment; Stage 5 extends with clinical-grade features.

### Diligence Q&A

**If asked: "Why 5 years? Most pitches stop at 18 months."**

> Two reasons. First, biotech is a long game — investors who understand the field expect 5-year thinking, especially for platform plays where the moat compounds over time. Second, the platform's value comes from multi-year data accumulation: Phase 1 → Phase 2 → Phase 3 wet lab work each takes 1-2 quarters, and the cumulative effect is what makes the platform defensible. A 5-year arc is the minimum to see the compounding loop in action.

**If asked: "What does 'first-in-class candidates' actually mean?"**

> Internal drug pipelines produce drug candidates targeting immune-system mechanisms that no other team is working on — first-in-class by construction because the platform identifies targets that traditional approaches miss. Pipeline 1 (starting Q1-Q2 2027) is focused on combination targets in immune-driven hematological malignancies; Pipeline 2 (starting Q2 2028) extends to broader immune-target discovery. By 2029-2031, these pipelines reach target-validated and lead-selection stages — the candidates exist; the platform is what generated them. Clinical trials are downstream of this work, outside the 5-year window.

**If asked: "What's 'platform = OS for immune-system drug discovery'?"**

> By 2029-2031, the platform's combination of validated multi-omics encoder + temporal dynamics + perturbation prediction + causal-readiness + clinical translation infrastructure becomes a complete operating system: any team — internal R&D or external pharma partner — can query the platform with a target hypothesis or combination question and get production-quality predictions back, with full audit trail and regulatory-grade provenance. That's what "OS" means here — not just a model, but the infrastructure for using the model at scale.

**If asked: "What's the relationship between Stage 5 in 2028 and pipeline maturation in 2029+?"**

> Stage 5's causal-readiness layer in 2028 is what makes pipeline maturation accelerate from 2029 onward. Without explicit causal modeling, target prioritization is correlational — useful but slow. With causal-readiness, the platform can answer counterfactual questions ("what if BTK were active but JAK1 inactive in this cell type?") which is what production drug discovery requires. Stage 5 is the substrate; pipeline maturation is what runs on it.

**If asked: "How is this different from the AI biotech hype companies?"**

> Three differences. First, the moat is data (QurieSeq), not the architecture — we don't pretend our model is novel; we have a coherent architecture spec on GitHub. Second, the roadmap has explicit dependency chains — every milestone has a wet-lab or model prerequisite cleanly identified. Third, the clinical translation infrastructure is treated as a 2028 build, not a 2030+ future hope — it's a deliverable on the roadmap with associated budget. We don't position the platform as solving biology; we position it as solving the data-and-model infrastructure that makes biological problems addressable at scale.

**If asked: "What about pharma partnerships? When do they materialize?"**

> Partnerships start in the BD pipeline track from 2027 onward (visible on D1 roadmap). By 2029-2031, partnerships scale through the clinical translation framework — meaning the platform exports regulatory-grade outputs to partner workflows, not just informal collaboration. We're not committing to specific partnership counts because the BD pipeline is competitive risk to publish, but the infrastructure is built to support significant partnership scale.

---

## Investor framing (one-paragraph elevator)

> The 5-year trajectory has four distinct phases. 2026 validates the platform with BTK+JAK demo + Phase 1 onboarding. 2027 extends to 5 modalities + 20-donor scale + cross-disease transfer + first drug pipeline starts. 2028 adds the causal-readiness layer + clinical translation framework + second drug pipeline. 2029-2031 matures pipelines toward target-validated candidates and scales pharma partnerships through the clinical translation infrastructure. Each phase compounds: data accumulates (Phase 1 → Phase 2 → Phase 3 wet lab), model capability extends (3 modalities → 5, single donor → 20, correlational → causal), clinical infrastructure builds (provenance, diligence packages, audit trails). By 2031, the platform is the operating system for immune-system drug discovery — usable by internal R&D and external pharma partners with regulatory-grade outputs.

---

## What's NOT on this slide (intentionally)

- IPO, M&A, exit, or any liquidity event references — per Ash's strategic direction
- Series A / B / C fundraise timing — per Ash's strategic direction
- Specific pharma partnership names or terms — competitive risk
- Clinical trial timelines (Phase I/II/III) — beyond 5-year scope, would over-promise
- Revenue projections / financial growth charts — Kinga's deck handles
- Specific drug-pipeline target choices — competitive risk

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — horizontal phase progression + 3-card compounding row.

**File output**: `docs/deck/assets/diagrams/E1_five_year_trajectory.svg`

**Followup prompt for Cowork** (when ready):
"Generate `E1_five_year_trajectory.svg` per spec in `docs/deck/content/E1_five_year_trajectory.md`. Top: 4-phase horizontal progression labeled 2026 / 2027 / 2028 / 2029-2031 with phase titles PHASE 1 VALIDATION / PHASE 2 EXTENSION / PHASE 3 MATURATION / PHASE 4 TRANSLATION. Each phase has 4-5 output bullets visible. Visual treatment: 2026 darkest+most-filled (highest confidence), 2027 fully colored, 2028 colored but lighter, 2029-2031 outlined-only (directional). Bottom: 3-card row labeled DATA COMPOUNDS / MODEL COMPOUNDS / CLINICAL INFRA COMPOUNDS — each card describes a specific compounding loop. Output 1920×1080 viewBox. NO mention of IPO/exit/Series A anywhere."

---

## Risk callouts (NOT to include on slide; for tracking only)

- The 2029-2031 phase is directional, not literal. Any investor question requesting specifics in this window gets a "depends on Phase 2 results + pipeline progression" answer.
- "First-in-class candidates" framing requires successful Pipeline 1 target validation in Stage 5 — which depends on Phase 2 data quality + causal-readiness layer working as designed. If those slip, the 2029-2031 narrative slips with them.
- The "platform = OS for immune-system drug discovery" framing is ambitious — needs to be backed up if challenged. The basis: validated foundation model + multi-modality + causal-readiness + clinical infrastructure = OS-grade capability. Defensible if pressed.
- No specific numbers means no pre-registered thresholds for the 2029+ window. This is appropriate — pre-registration applies to current milestones, not future possibilities. But be honest about this if asked.

---

## What's NEXT after E1 is committed

**All 12 appendix content specs complete.**

Next phase: SVG diagram generation. Each spec includes a Cowork prompt for the diagram. We send them to Cowork for batch generation, then iterate on visuals. After SVGs are in place, Cowork generates the .pptx from specs. Final visual polish via Claude Design.

Estimated next-day work:
- Batch SVG generation: 1-2 hours of Cowork iteration
- .pptx generation: 30-60 min from specs + SVGs
- Visual polish: variable

Total path to investor-ready draft: ~3 days from where we are now.
