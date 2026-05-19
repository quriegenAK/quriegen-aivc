# Slide F1 — Integrated Causal Perturbation Platform

- **Maps to Kinga's deck**: Extends slide 9 (competitive checkbox matrix) with platform-architecture framing
- **Section**: F — Competitive Positioning (new section)
- **Visual lead**: Flywheel diagram + competitor archetype grouping
- **Status**: Draft — pending Ash review

---

## Headline

**The closed-loop platform — proprietary data, co-designed architecture, compounding over time.**

(Alternative: *"Most platforms optimize one layer. We're building the integrated stack."*)

(Alternative: *"Wet lab + architecture + protocol family, co-designed as one system."*)

---

## Sub-headline (one line under headline)

No public dataset has the combination — multi-omics + perturbation-aware + temporal + combinatorial. Phase 1 QuRIE-seq closes the gap with phospho at Q3 2026. The platform compounds from there.

---

## Body content (3 bullets max)

- **Public data was the bootstrap; QuRIE-seq is the platform.** No public dataset combines RNA + Protein + Phospho + perturbation-aware temporal sampling on primary PBMCs. The gap is structural — phospho-proteomics on PBMCs under combinatorial perturbation does not exist publicly. QuRIE-seq Phase 1 closes this gap at Q3 2026 with phospho integral to the assay; Phase 2 adds VDJ. The data engine exists because the data doesn't.

- **Five pillars co-designed as one system**: proprietary wet-lab generation (QuRIE-seq family), architecture co-designed with the assay (decomposed readout matches 4-arm experimental design; Neural ODE matches irregular timepoint sampling 0/5/30/60/180; phospho head matches QuRIE-seq's intracellular protein channel), temporal perturbation-aware data creation, compositional training strategy, unified protocol family for Phase 2 VDJ extension. Each pillar reinforces the others; the integration is the moat.

- **Competitors optimize one layer each**: TAHOE optimizes single-modality data scale (100M cells, RNA-only cell lines, no phospho); Immunai optimizes modality-rich atlases (RNA + Protein + VDJ via partners, no phospho); Cellarity, CytoReason, Turbine, DeepLife optimize foundation models on partner-derived data without proprietary wet-lab pipelines; Valo and Noetik optimize downstream therapeutics. We optimize the closed-loop system itself, including phospho — the dimension no competitor measures.

---

## Visual spec — the flywheel + the archetype grouping

### Top zone — the flywheel (visual hero)

Center of slide: a circular flywheel diagram showing the closed-loop platform with **4 stages**, each driving the next:

```
                           ┌─────────────────────┐
                           │  CO-DESIGNED        │
                           │  ARCHITECTURE       │
                           │                     │
                           │  4-arm decomposed   │
                           │  readout · Neural   │
                           │  ODE temporal ·     │
                           │  compositional      │
                           │  generalization     │
                           └──────────┬──────────┘
                                      │
                                      ↓
   ┌──────────────────────┐                       ┌──────────────────────┐
   │  WET-LAB             │                       │  TEMPORAL            │
   │  GENERATION          │  ←─── compounds ───→  │  MULTI-OMICS         │
   │                      │                       │                      │
   │  QurieSeq Phase 1+2  │                       │  RNA · Protein ·     │
   │  primary human PBMCs │                       │  **Phospho** (lavender) │
   │  5 → 20 donors       │                       │  5 timepoints        │
   │  4-arm perturbations │                       │  (0/5/30/60/180)     │
   │                      │                       │  ATAC × 2 timepoints │
   │                      │                       │  VDJ — Phase 2       │
   └──────────┬───────────┘                       └──────────┬───────────┘
              │                                              │
              └──────────────────┬───────────────────────────┘
                                 ↓
                       ┌─────────────────────┐
                       │  PROTOCOL-FAMILY    │
                       │  EXPANSION          │
                       │                     │
                       │  Same wet-lab       │
                       │  pipeline extends   │
                       │  to Phase 2 VDJ +   │
                       │  20-donor scale     │
                       │  without re-arch    │
                       └─────────────────────┘

         Each loop deepens the next. Every QurieSeq phase trains the
         architecture; every architecture extension informs the next wet
         lab. Integration is the moat.
```

Style notes:
- Circular flywheel (or 4-corner orbit) with curved arrows indicating clockwise flow
- Each pillar in its own card with section accent (amber `#FBBF24` for F section)
- Center label: "INTEGRATED PLATFORM" with small caps, cyan, letter-spaced
- Curved "compounds" arrows between adjacent pillars
- Bottom caption explains the compounding loop in one sentence

### Middle zone — competitor archetypes (3-bucket grouping)

Below the flywheel, group the 7 named competitors by **what layer they optimize**, not by checkbox count:

```
WHO OPTIMIZES WHAT?
───────────────────

┌─────────────────────────────────┐  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  DATA SCALE                     │  │  FOUNDATION MODELS              │  │  DOWNSTREAM THERAPEUTICS        │
│                                 │  │                                 │  │                                 │
│  • TAHOE — 100M cells,          │  │  • CytoReason — partner-derived │  │  • Valo Health — clinical       │
│    RNA-only cell lines          │  │    multi-omics, immune focus    │  │    development                  │
│  • Immunai — modality-rich      │  │  • Turbine AI — virtual lab,    │  │  • Noetik — spatial multi-omics │
│    atlas, partner data          │  │    pharma partnerships          │  │    oncology                     │
│                                 │  │  • DeepLife — causal modeling,  │  │                                 │
│                                 │  │    drug repositioning           │  │                                 │
│                                 │  │                                 │  │                                 │
│  Optimize: data breadth         │  │  Optimize: model architecture   │  │  Optimize: clinical pipeline    │
│  Decouple: wet-lab + protocol   │  │  Decouple: proprietary data     │  │  Decouple: foundation modeling  │
└─────────────────────────────────┘  └─────────────────────────────────┘  └─────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  INTEGRATED CAUSAL PERTURBATION PLATFORM                                                                 │
│                                                                                                          │
│  • Quriegen — proprietary wet-lab + co-designed architecture + temporal multi-omics + compositional      │
│    causal modeling + protocol-family expansion, all coupled                                              │
│                                                                                                          │
│  Optimize: the closed-loop system itself                                                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Style notes:
- Top three buckets equal-width, neutral accent
- Quriegen row full-width, **amber accent** (`#FBBF24` section color, filled or strong stroke)
- Visual hierarchy: top buckets recede; bottom row dominates
- No checkmarks, no winner/loser language
- Each top-bucket card shows "Optimize: X" and "Decouple: Y" so the contrast with the integrated row is structural, not pejorative

### Bottom zone — closing line

```
"Phase 1 QuRIE-seq (Q3 2026) measures RNA + Protein + Phospho at 5
timepoints — the combination no public dataset has. The platform
compounds from there."
```

Italic, muted, centered. Functions as the takeaway investors carry out of the slide — shifts the framing from aspirational to imminent (Q3 2026 is the next investor diligence touchpoint).

---

## Notes for design

- **Flywheel is the visual center**. The four-pillar loop with curved compounding arrows is the slide's strongest conceptual element. Make it dominant.
- **Three-bucket competitor grouping below is supporting evidence**, not the headline. Smaller visual weight than the flywheel.
- **Amber section accent** (`#FBBF24`) distinguishes F from A/B/C/D/E. Use throughout.
- **No checkmark matrix**. Kinga's slide 9 does the checkbox comparison; F1's job is to reframe what the comparison should measure.
- **"Integration is the moat" tagline** in the flywheel caption — central insight, low visual ornamentation.
- **Avoid "category-of-one" or similar marketing phrases on-slide**. Let the visual demonstrate the categorical separation. Speaker notes can use stronger language.

---

## Why this slide matters

F1 closes the appendix's competitive argument with the right strategic frame. Three things it earns:

1. **Reframes the comparison axis**: Kinga's slide 9 (checkbox matrix) makes the comparison granular. F1 makes it categorical — we play a fundamentally different game than the competitive set. Sophisticated investors recognize that as a stronger positioning argument than "more checkboxes."

2. **Justifies QurieSeq as necessity, not luxury**: "No public dataset has the combination" makes the wet-lab strategy unavoidable rather than optional. This is the most defensible answer to "why are you spending $4M on wet lab?"

3. **Reveals the compounding flywheel**: investors understand compounding systems faster than capability matrices. Each pillar reinforcing the others is what justifies the multi-year platform investment thesis.

---

## Source data / claims

| Claim | Source |
|---|---|
| TAHOE 100M cells, RNA-only cell lines, 3B-param Tahoe-x1 FM | `docs/deck/research/competitive_landscape_2026_05.md` (Tahoe entry) |
| Immunai AMICA atlas with RNA + Protein + VDJ today | Competitive research doc (Immunai entry) |
| CytoReason multi-omics from partner data, Pfizer $110M + Sanofi extensions | Competitive research doc (CytoReason entry) |
| Turbine AI virtual lab, AZ + Bayer + Ono partnerships | Competitive research doc (Turbine entry) |
| DeepLife TwinCell causal cell model, drug repositioning | Competitive research doc (DeepLife entry) |
| Valo Health clinical pipeline, Novo Nordisk $4.6B deal | Competitive research doc (Valo entry) |
| Noetik spatial multi-omics oncology, GSK $50M deal | Competitive research doc (Noetik entry) |
| No public dataset matches the combination required | Stage 3 Part 1 Report 1 (datasets) + L3 B-cell CRISPR exhaustive search |
| Architecture co-designed with assay (decomposed readout = 4-arm; Neural ODE = irregular timepoints) | Architecture spec v1.1 §3.2, §4 + QurieSeq Phase 1 design (Thiago, May 12) |
| 5 pillars: proprietary wet lab, co-designed architecture, temporal multi-omics, compositional training, protocol-family expansion | Architecture spec v1.1 + QurieSeq Phase 1/2 specs |
| Phase 1 (Q3 2026): 4 modalities including phospho. Phase 2 (Q1 2027+): adds VDJ + 20-donor scale within the same protocol family without re-architecting | QurieSeq Phase 1+2 spec (Thiago + Kinga, May 12) + 2026-05-17 phospho correction |

---

## Speaker notes

### Three-state framing
- **Today (public-data substrate)**: 3-modality encoder validated on DOGMA-seq + Calderon at 73%. 0.57 Mimitou CRISPR ADAPTER_RECOMMENDED. Public-data evidence shipped.
- **Phase 1 (Q3 2026 — flywheel activates)**: 4-modality QuRIE-seq Phase 1 (RNA + Protein + Phospho all 5 timepoints + ATAC 2 timepoints) + BTK+JAK combo. The "no public dataset has the combination" claim becomes literally true at Phase 1.
- **Phase 2 (2027 — flywheel compounds)**: VDJ 5th modality + 20-donor scale. Same protocol family. Disease-context samples (Phase 3 onwards).

### Technical glossary
**Integrated platform** — Five pillars co-designed as one system: (1) proprietary wet-lab generation (QuRIE-seq family), (2) co-designed architecture (decomposed readout matches 4-arm; Neural ODE matches irregular timepoints), (3) temporal multi-omics including phospho from Phase 1, (4) compositional perturbation modeling, (5) unified protocol family for Phase 2 VDJ extension.

**Flywheel** — Self-reinforcing cycle. Every QuRIE-seq phase trains the next architecture extension; every architecture extension informs the next wet-lab design. The integration compounds.

**Closed-loop platform** — Tight coupling between wet lab and model. Data generation, architecture, training, prediction, and next-round design all reference each other. Distinguished from open-loop platforms where data and model are decoupled.

**Co-designed (wet lab + architecture)** — Wet-lab experimental design (4-arm, 5 timepoints, BTK+JAK combo, phospho integral) matches model architecture (decomposed readout, Neural ODE temporal, synergy head, phospho input). Neither retrofitted to the other.

**TAHOE** — Tahoe Therapeutics. 100M-cell foundation model. RNA-only, cell-line-derived. Optimizes for foundation-model substrate scale. Our 500K Phase 1 PBMC data optimizes a different axis (modality depth + perturbation-aware design).

**Immunai (AMICA atlas)** — Modality-rich immune atlas. RNA + Protein + VDJ via partner data. No phospho. Different from our platform (we have phospho Phase 1; they have VDJ today; Phase 2 closes our VDJ gap).

**CytoReason, Turbine AI, DeepLife** — Foundation model competitors using partner-derived data. No proprietary wet-lab pipeline. Different from our closed-loop integration.

**Cellarity** — Foundation model + cell-state correction architecture. Partner data. Causal modeling but different architectural choices than ours.

**Valo Health, Noetik** — Downstream therapeutics companies. Use foundation models for drug development. Different layer of the stack — potentially substrate users of platforms like ours.

**Drug combination prediction (causal)** — Predicting how combinations of drugs affect cells. Requires multi-omics + perturbation-aware data + temporal coverage + combinatorial conditions. Our 5-pillar architecture is designed for this; competitors optimize one layer each.

**Three archetype buckets** — Data scale (TAHOE, Immunai), foundation models (CytoReason, Turbine AI, DeepLife), downstream therapeutics (Valo, Noetik). Each is doing important work in their layer. Our platform spans all three layers via the closed-loop integration.

**Phospho-as-no-public-data structural moat** — Phospho-proteomics on PBMCs under perturbation does not exist in any public single-cell dataset. We close this gap with Phase 1 QuRIE-seq (Q3 2026). Phospho is the modality our competitors structurally cannot match without building a proprietary wet-lab pipeline.

**Same protocol family (Phase 1 + Phase 2)** — Phase 2 extends Phase 1's QuRIE-seq protocol with VDJ + scale, NOT by switching protocols. Same encoder backbone, same readout architecture, new input head only. Compounding rather than rebuilding.

### Diligence Q&A

**If asked: "How is this different from Cellarity / Recursion / Insitro?"**

> Three companies investors might mention that aren't on the slide. Cellarity is closest on modality — they published a 3-modality (RNA + ATAC + surface) atlas in *Science* in October 2025, and they're in Phase 1 clinical trials with CLY-124. But Cellarity is a therapeutics company; their model is internal infrastructure for their own pipeline, not a platform that compounds across drug combinations. Recursion optimizes for phenotypic screening at scale via cell painting — different modality stack, image-based not sequencing-based. Insitro runs a phenotype-to-genotype pipeline; great science, different architecture goal. We compete with none of them directly on the integrated multi-omics-perturbation axis — they're adjacent companies in the broader AI biotech space.

**If asked: "TAHOE has 100M cells. You have 500K. Why isn't that decisive?"**

> Different optimization. TAHOE optimizes for open foundation-model substrate from cell-line perturbations — 100M cells, RNA-only, no phospho, no perturbation-aware multi-omics. We optimize for proprietary integrated platform on primary human PBMCs with 4 modalities including phospho in Phase 1 (Q3 2026). The 500K Phase 1 cell count is intentionally small for modality depth + perturbation-aware design; Phase 2 scales to 20 donors. TAHOE's scale gives them open-source RNA foundation models; our depth gives us closed-loop causal drug combination prediction with the modality (phospho) no public dataset has. Different layers of the stack — they don't compete with our platform, they're potentially substrate beneath it.

**If asked: "Immunai already has VDJ. You don't until 2027. Doesn't that make them ahead?"**

> Immunai has VDJ in their AMICA atlas today — that's a real capability we don't have until Phase 2 (2027). But Immunai doesn't have phospho. We have phospho in Phase 1 (Q3 2026). The modality stacks are different: Immunai = RNA + Protein + VDJ (immune-receptor diversity); us = RNA + Protein + Phospho (kinase signaling state) + ATAC. Different use cases. Immunai's VDJ supports clinical biomarker discovery on patient samples; our phospho supports drug combination mechanism prediction. Phase 2 closes the VDJ gap on our side; their phospho gap is structural — they don't generate it because they don't have a proprietary wet-lab pipeline for phospho measurement. The modality presence question is necessary but not sufficient — what matters is how the modality integrates with the rest of the platform.

**If asked: "Why does phospho matter? What does it give you that RNA + protein don't?"**

> Phospho measures kinase activation state — the immediate signaling response to a perturbation, before transcriptional changes propagate. For drug combination prediction in pathway-driven diseases like CLL, phospho is the readout that distinguishes additive effects ("drug A and drug B affect the same kinase") from synergistic effects ("drug A blocks JAK and drug B blocks BTK so the combination hits both arms of the BCR pathway"). RNA shows downstream consequences hours later; phospho shows immediate mechanism in minutes. The 5-minute timepoint in QuRIE-seq Phase 1 captures phospho responses that no other dataset measures at single-cell resolution on primary PBMCs.

**If asked: "Most of these competitors have $50M+ pharma deals. You have zero. Why?"**

> True today and disclosed in our research doc. Two reasons. First, we're seed-stage; pharma deals tend to follow architectural validation milestones, which our roadmap places at Stage 3b (BTK+JAK demo Q4 2026) and Stage 4 (multi-disease transfer 2027). Second, the pharma BD pipeline is a Stage 4-5 deliverable on our roadmap (D1) with $1M allocated in seed (D2). We're not pre-revenue by accident; we're pre-revenue by design. Pharma partnerships materialize when the platform has shipped the demos that justify them. That sequence is on the roadmap.

**If asked: "Where are your peer-reviewed publications?"**

> Stage 3 verdict + BTK+JAK demo go to peer-reviewed publication in Q4 2026 — it's explicit on the D1 roadmap. Today's evidence is on GitHub: architecture spec v1.1, Phase 6.5g.2 closure report with dual-conclusion methodology, Stage 3 Part 1 verdict, pre-registered eval thresholds, working implementation with 87 unit tests. We chose to ship reproducible architecture before ship a paper. The paper follows the data, not the other way around — that's discipline, not absence of evidence.

**If asked: "Aren't you just one of many AI biotechs?"**

> AI biotech is a category. Inside it, companies cluster by what they optimize: data scale (TAHOE), modality breadth (Immunai), foundation models (CytoReason, Turbine, DeepLife), clinical pipelines (Valo, Noetik, Cellarity). We sit alone in the category that optimizes the closed-loop integrated platform — proprietary wet-lab data generation + architecture co-designed with the assay + temporal multi-omics + compositional perturbation modeling + protocol-family expansion. The category isn't crowded; the category is empty except for us.

**If asked: "Why is integration the moat? Couldn't a competitor build all five layers?"**

> They could. The question is timing and coordination cost. Building a wet-lab pipeline producing single-cell multi-omics perturbation data, architecting a model to consume it, and running them as one feedback loop requires three coordinated bets simultaneously. Most teams chose one bet — that's why the competitive map fragments by optimization axis. Catching us up means starting all three simultaneously and waiting 2-3 years for the integration to compound. Meanwhile we're shipping QurieSeq Phase 1 in Q3 2026, Phase 2 in 2027, and our architecture is already validated cross-corpus at 73% with the adapter strategy locked. The integration is the moat because every quarter widens it.

---

## Investor framing (one-paragraph elevator)

> No public single-cell dataset combines what causal drug combination prediction requires — multi-omics, perturbation-aware, temporal, combinatorial, protocol-aligned for modality expansion. That structural gap is why QuRIE-seq exists: a proprietary wet-lab platform generating data co-designed with our model architecture. Five pillars run as one integrated system — wet-lab data generation, architecture co-designed with the assay (the decomposed readout matches the 4-arm experimental design; Neural ODE matches irregular timepoint sampling), temporal multi-omics including phospho-proteomics integral to QuRIE-seq from Phase 1 (Q3 2026), compositional perturbation modeling, and unified protocol family for Phase 2 VDJ extension. Competitors optimize one layer each: TAHOE optimizes single-modality data scale (100M cells, cell lines); Immunai optimizes modality-rich atlases via partner data; CytoReason, Turbine, DeepLife optimize foundation models on partner-derived data; Valo and Noetik optimize downstream therapeutics. We optimize the closed-loop system itself. Every QuRIE-seq phase trains the next architecture extension; every architecture extension informs the next wet-lab design. The integration compounds. Modality coverage, cell counts, and partner deals are downstream properties of that integrated platform — not the headline argument.

---

## What's NOT on this slide (intentionally)

- Recursion, Insitro, Cellarity by name — adjacent companies, addressed in speaker notes only
- Specific competitor funding amounts or valuations — distracts from architectural positioning
- The peer-reviewed publication gap explicitly — handled in speaker notes if asked
- Cell count comparison (100M vs 500K) — addressed in speaker notes if asked, frame is wrong axis for F1
- "Category-of-one" or similar marketing language — let the visual demonstrate categorical separation
- A checkmark matrix — Kinga's slide 9 covers that; F1 reframes the axis

---

## Diagram generation strategy

**Tool**: Cowork (Python matplotlib + svgwrite hybrid for the flywheel curves).

**File output**: `docs/deck/assets/diagrams/F1_integrated_platform.svg`

**Followup prompt for Cowork** (when ready):
"Generate `F1_integrated_platform.svg` per spec in `docs/deck/content/F1_competitive_positioning.md`. 
Top zone (visual hero): 4-pillar flywheel diagram with curved arrows showing the compounding loop — WET-LAB GENERATION / CO-DESIGNED ARCHITECTURE / TEMPORAL MULTI-OMICS / PROTOCOL-FAMILY EXPANSION. Center label: 'INTEGRATED PLATFORM' in section accent amber. Curved arrows between pillars show clockwise flow with 'compounds' labels.
Middle zone: 3 competitor archetype buckets (DATA SCALE / FOUNDATION MODELS / DOWNSTREAM THERAPEUTICS) above one full-width INTEGRATED row showing Quriegen alone. Each top bucket lists 2-3 competitors with their named optimization layer. Quriegen row uses amber accent (full strength); top buckets neutral.
Bottom zone: italic centered closing line 'No public dataset has the combination drug combination prediction requires. The wet lab, the architecture, and the protocol family are co-designed.'
Section accent throughout: amber `#FBBF24`. Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- Cellarity's *Science* paper + Phase 1 clinical is the most uncomfortable adjacent competitor — speaker notes addressed but a savvy investor may pivot to "how do you differ from Cellarity specifically?" Have the answer rehearsed.
- The "compounding" claim is forward-looking — Phase 1 Q3 2026 must ship for the flywheel narrative to hold. If Phase 1 slips materially, F1's integration argument weakens.
- "Architecture co-designed with the assay" is the line that earns technical credibility — must be defensible if pressed. The decomposed readout → 4-arm match and Neural ODE → irregular timepoints match are concrete. Practice the explanation.
- Competitor characterizations on the slide should be accurate, not strawman. Each entry in the buckets must trace to the research doc. If any competitor pushback ("we do more than scale!") is anticipated, the speaker notes should preemptively concede the nuance.
- The integrated platform argument depends on **execution continuity**. If Phase 1 ships but Phase 2 doesn't, the flywheel breaks at the modality-expansion pillar. The roadmap (D1) backstops this — but F1's claim only holds with the roadmap.

---

## What's NEXT after F1 is committed

Move to **F1 SVG generation** via Cowork — flywheel diagram + 3-bucket archetype grouping + closing line. Estimated 1-1.5 hours. Then re-run `_build_appendix_pptx.py` to assemble the 20-slide v2 deck (1 cover + 6 dividers + 13 content slides).

After F1 lands: **Phase 4 polish** (Claude Design) on hero slides (cover + A1 + A3 + B2 + C1 + C2 + D1 + E1 + F1) plus expanded speaker notes for technical glossary across all content slides.
