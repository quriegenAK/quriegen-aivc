# Phase 1 Modality Correction + Three-State Framing — 2026-05-17

**Status**: Locked, canonical
**Trigger**: Kinga clarification 2026-05-17 that phospho is integral to QuRIE-seq Phase 1 + ATAC measured at t=0 and t=180 only
**Inputs**:
- Kinga messages 2026-05-17 (phospho, ATAC scope)
- Thiago confirmations May 12 (5 donors, timepoints 0/5/30/60/180, BTK+JAK combo in Phase 1, VDJ deferred)
- Stage 3 dataset L1-L5 strategy conversation (CIPHER-seq passed, Rubin L3-v0 approved, L5 phospho closed by QuRIE-seq)
**Drives**: A2, C1, D1, F1 content spec updates; A5 speaker notes update; pptx v4 rebuild; speaker notes glossary expansion

---

## The Core Correction

**Old (incorrect) framing in deck**:
> Phase 1 QuRIE-seq = RNA + Protein/CITE-seq + ATAC (via integration); phospho deferred to Phase 2

**Correct framing (per Kinga 2026-05-17)**:
> Phase 1 QuRIE-seq = RNA + Protein + **Phospho** (all integral to QuRIE-seq assay) + **ATAC measured directly at t=0 and t=180 only**
> Phase 2 QuRIE-seq = adds **VDJ** as 5th modality + scale to 20 donors

### Why phospho was treated as Phase 2

The Stage 3 dataset strategy conversation (Kinga/Thiago/Ash) used "L5 phospho deferred to QuRIE-seq" to mean "no public phospho data exists, we'll generate it ourselves." This was a **public-data discussion** that got conflated with phase timing. The deck inherited the wrong framing.

Reality: phospho is integral to the QuRIE-seq assay — every QuRIE-seq run generates phospho. Phase 1 generates phospho. Phase 2 generates phospho. The first availability of phospho is Phase 1 (Q3 2026), not Phase 2.

### Why ATAC has reduced temporal coverage

Per Kinga 2026-05-17: *"we might include ATAC- but only to point 0h, and last time point because this layer of info is more stable. so there is no need to cover all 4-5 time points"*

Biological reasoning: chromatin accessibility changes on slower timescales than transcription. Sampling t=0 and t=180 captures the chromatin endpoint shift without unnecessary measurement at 5/30/60 min where ATAC signal would be unchanged. **This is thoughtful experimental design, not cost-cutting.**

---

## The Phase 1 Modality × Timepoint Matrix

|              | t=0   | t=5   | t=30  | t=60  | t=180 |
|--------------|-------|-------|-------|-------|-------|
| **RNA**      | ✓     | ✓     | ✓     | ✓     | ✓     |
| **Protein**  | ✓     | ✓     | ✓     | ✓     | ✓     |
| **Phospho**  | ✓     | ✓     | ✓     | ✓     | ✓     |
| **ATAC**     | ✓     |       |       |       | ✓     |
| **VDJ**      | (Phase 2 — 2027) |  |  |  |  |

**Phase 1 wet-lab generation parameters (Q3 2026)**:
- 5 donors × ~5k cells/donor/timepoint × 5 timepoints
- BTK + JAK perturbation conditions, including **BTK+JAK combo confirmed for Phase 1** (Thiago 2026-05-12)
- ~125k cells × ~17 conditions (vehicle + stim + 6 inhibitor singles + 6 inhibitor combos + extras)
- ATAC sampled at t=0 and t=180 only (2/5 timepoints)
- Donor metadata: blood type only via Sanquin; chromatin signature functions as donor signature

**Phase 2 wet-lab generation parameters (2027)**:
- 20 donors (4× Phase 1 scale)
- VDJ added as 5th modality
- Extended perturbation conditions
- Disease-state samples (B-cell lines, CLL, autoimmune contexts) follow Phase 2

---

## Phase 1 / Phase 2 Canonical Definitions

**Phase 1** = first proprietary QuRIE-seq wet-lab data generation batch
- Timing: Q3 2026
- Scope: 5 donors, 5 timepoints, 4 modalities directly measured
- Purpose: plug into validated public-data engine; ship BTK+JAK demo (Stage 3b); validate Stage 3c causal architecture
- Output: ~125k cells with proprietary multi-omics including phospho

**Phase 2** = expanded QuRIE-seq wet-lab data generation batch
- Timing: 2027 (Q1 onwards)
- Scope: 20 donors (scaled 4×), adds VDJ as 5th modality
- Purpose: scale validation, enable VDJ-aware analysis, support Stage 4 multi-disease transfer + Stage 5 causal-readiness
- Output: enriched dataset for pipeline 1 + 2 target identification

**Critical clarification for speaker notes**:
- Phase 1 / Phase 2 are **wet-lab data generation phases** (not clinical trial phases, not company funding rounds, not model training stages)
- Model training stages are **Stage 3a (current) → 3b (BTK+JAK demo) → 3c (causal architecture) → 4 (scale) → 5 (causal-ready)** — see D1 for the unified quarterly view that links wet-lab phases + model stages

---

## The Three-State Framing (Applies Throughout Deck)

Every modality and capability has three states. **Don't conflate them.**

| State | What it means | Tense |
|---|---|---|
| **Today (public-data substrate)** | Validated on public data: DOGMA-seq pretrain (RNA+ATAC+Protein), Calderon 2019 cross-corpus eval, Mimitou ASAP-seq CRISPR probe, Parse Cytokine Atlas (training), Rubin Perturb-ATAC (L3-v0) | Present / past-perfect |
| **Phase 1 (QuRIE-seq Q3 2026)** | First proprietary wet-lab data: RNA+Protein+Phospho × 5 timepoints + ATAC × 2 timepoints, 5 donors, BTK+JAK combo included | Future-near |
| **Phase 2 (QuRIE-seq 2027)** | Expanded proprietary data: + VDJ, 20 donors, more conditions, disease samples | Future-medium |

**Public-data substrate is real, shipped evidence — not roadmap.** The deck's credibility rests on:
- B1 (three-dataset methodology) — public-data validation methodology shipped
- B2 (0.57 hero, ADAPTER_RECOMMENDED) — public-data validation result shipped
- A2's "73% cross-corpus" — public-data result shipped

**QuRIE-seq Phase 1 + 2 is the proprietary upgrade** that adds:
- Modalities not available publicly (phospho first time, VDJ later)
- Temporal coverage (5 timepoints proprietary vs static public)
- Combinatorial perturbations on the same protocol family (BTK+JAK combo not in public data)
- Donor-level genetic variation (5 → 20 donors with proprietary chromatin signatures)

**Speaker notes for every relevant slide should explicitly walk these three states.**

---

## Slides Affected By Corrections

### Slides That NEED Content Spec + SVG Updates (4 slides)

#### A2 — Multi-Omics Encoder
**Current claim**: "3 modalities today; +phospho/+VDJ Phase 2"
**Corrected**: "3 modalities trained on DOGMA-seq + validated cross-corpus on Calderon (73% — public data, shipped). Phase 1 wet-lab adds proprietary phospho (4th modality, no public data exists). Phase 2 adds VDJ (5th modality)."
**SVG changes**: any visual element that lists/depicts modality timing. Encoder card may show 3 modalities today + arrow to Phase 1 (+phospho) + arrow to Phase 2 (+VDJ).

#### C1 — Phase 1 Experimental Design
**Current claim**: Likely shows "RNA + Protein at all 5 timepoints"
**Corrected**: Full modality × timepoint matrix (4 modalities, ATAC reduced to 2 timepoints).
**SVG changes**: Replace whatever current modality visual is with the matrix above. **This is now the slide's strongest single visual** — investor-grade experimental design with biological reasoning for ATAC temporal subsampling.

#### D1 — Quarterly Roadmap
**Current claim**: "Phase 2 phospho on Q1'27" milestone diamond on Gantt
**Corrected**: Milestone label changes to "Phase 2 VDJ on Q1'27" (or whichever exact quarter VDJ comes online per current planning).
**SVG changes**: Single milestone label change on Gantt. Phase 1 delivery milestone (Q3'26) text may also need brief modality update if visible.

#### F1 — Competitive Positioning
**Current claim**: Flywheel TEMPORAL MULTI-OMICS pillar shows "RNA + ATAC + Protein (+phospho + VDJ Phase 2)" + speaker notes reference phospho as Phase 2
**Corrected**:
- Flywheel pillar text: "RNA + Protein + Phospho (5 timepoints) · ATAC (t=0, t=180) · VDJ Phase 2"
- Closing line: *"No public dataset has the combination drug combination prediction requires"* — now literally true at Phase 1 (Q3 2026), not aspirational
- Speaker notes: rewrite TAHOE/Immunai/pharma-deal Q&As to reflect 4-modality Phase 1
**SVG changes**: TEMPORAL MULTI-OMICS pillar text update; potential slight adjustment to closing line emphasis.

### Slides That Need Speaker-Notes-Only Update (1 slide)

#### A5 — Causal Architecture
**SVG**: NO change. Status pill, equation, GRN visualization all remain correct.
**Speaker notes only**: Update timing language — "phospho available in Phase 1 (Q3 2026), so causal validation can begin Q4 2026" instead of "post Phase 2."

### Slides That DO NOT Need Changes (9 slides)

These slides are **locked** — no phospho/ATAC content to correct:

- **A1** system architecture (abstract flow, no specific modality timing)
- **A3** decomposed readout (equation-focused)
- **A4** Neural ODE (temporal architecture, abstract)
- **B1** three datasets methodology (public-data, shipped, stays correct)
- **B2** adapter verdict 0.57 (public-data result, stays correct)
- **B3** synergy pre-demo (Mimitou substitute, stays correct)
- **C2** BTK+JAK demo plan (pre-registered methodology, stays correct; BTK+JAK combo confirmed for Phase 1 per Thiago)
- **D2** seed allocation (budget, no modality timing)
- **E1** 5-year trajectory (strategic horizon, stays correct)

**9 of 14 slides stay locked.** The correction scope is bounded.

---

## Strategic Upgrades The Corrections Unlock

This isn't just a bug-fix. The corrections produce **stronger architectural and competitive claims**:

### 1. F1's "no public dataset" claim becomes literally true at Phase 1
Currently framed as forward-looking (Phase 2). Corrected: phospho is the dimension nobody else has, and it's in Phase 1 (Q3 2026). The flywheel **already operates** at Q3 2026, not 2027. Closes the "when does this become real?" question with a concrete near-term answer.

### 2. A5's Stage 3c validation timeline accelerates
Phospho signal becomes available Q3 2026 (Phase 1) instead of Q1 2027 (old Phase 2). Stage 3c causal architecture validation can begin Q4 2026 — **one quarter sooner than the deck currently claims**. Status pill timing on A5 ("Validation Q1-Q2 2027") may want to shift earlier — but check D1 first to maintain canonical timeline.

### 3. C1 becomes architecturally sharper
The modality × timepoint matrix is investor-grade experimental design. Shows discipline: chromatin sampled at biologically-relevant timepoints (slow-varying), proteins/RNA/phospho sampled at fast timepoints (signal-varying). **Mathematical efficiency × biological reasoning = credibility.**

### 4. A2 cross-corpus claim strengthens
Encoder trained on DOGMA-seq (3 modalities, public). Phase 1 adds the 4th modality (phospho) that no public dataset contains. The encoder's job during Phase 1 integration is harder than I'd been claiming — it must learn phospho representations without prior cross-corpus precedent. **This is a more defensible technical position.**

### 5. F1 competitive comparisons sharpen
TAHOE Q&A: "TAHOE 100M cells RNA-only vs our 500K Phase 1 with **RNA + Protein + Phospho measured directly** + ATAC integration — different axes of optimization, and we have phospho that no public foundation model has."

Immunai Q&A: "Immunai's AMICA has VDJ today but no phospho. Our Phase 1 has phospho but no VDJ. Phase 2 closes the VDJ gap; their phospho gap is structural — they don't generate it."

### 6. D1 roadmap honest about Phase 2 scope
Phase 2 is **just VDJ** now, not "phospho + VDJ." Smaller scope addition. More focused. The phasing of the platform looks more realistic to investors who know wet-lab biology.

---

## Tech Debt Captured For Phase 4 (Plus Ash's New Observation)

Items to bundle with Phase 4 polish:

1. **A1 speaker notes** — currently blank in content spec. Phase 4 fills these.
2. **Pagination unification** — A1-E1 at `/12`, F1 at `/13`, A5 at `/14`. Phase 4 unifies all to `/14`.
3. **A5 color coding restoration on equation** — currently white-only mathtext PNG. Phase 4 can do 3 separate mathtext renders (W cyan / dₚ lavender / (I−W)⁻¹ green).
4. **Architecture spec v1.2 §X causal layer extension** — anchored by A5 SVG. Spec doc follows the slide.
5. **`_deck_common.py` defaults sweep** — already done at v2 (min_gap=2), but worth final audit.
6. **Cairosvg italic-Latin substitution** — documented limitation. Future SVGs needing italic Latin must render via matplotlib mathtext OR drop italic.
7. **ASH'S NEW OBSERVATION 2026-05-17**: *"some text is too small for eyes"* — Phase 4 visual polish must audit font sizing across all slides. Specifically check:
   - Body text minimum 14pt at slide-fill scale (currently some 11-12pt Arial body text)
   - Source citation footers (currently 11pt — might need 12-13pt)
   - GRN node labels on A5 (currently ~11pt — possibly too small at presentation distance)
   - Speaker notes glossary additions must not creep below 12pt
   - Any "fine print" italic muted text — audit minimum size

---

## What Phase 4 Speaker Notes Glossary Must Cover

Based on Kinga + Jan's ask + the technical depth of the deck, every relevant slide gets speaker notes that define:

### AI/ML terminology
- Latent space 256-D (the encoder's output dimensionality)
- Neural ODE (Neural Ordinary Differential Equation — continuous-time dynamics model)
- Latent SDE (Stochastic DE — for noise-aware dynamics)
- Compositional generalization (predicting combinations from singletons)
- Perturbation embeddings (vector representation of perturbation context)
- Encoder pretraining (foundation model paradigm)
- Adapter strategy (lightweight per-task layer on frozen encoder)
- Frozen encoder (encoder weights locked during downstream training)
- Decomposed readout (4-arm `h_base + Δ_stim + Δ_inh + Δ_synergy` decoder structure)
- Cross-corpus generalization (generalize across independently-generated datasets)
- Pre-registered evaluation (eval thresholds committed before seeing results)
- Indicator function / Iverson bracket `𝟙[·]` (1 if true, 0 if false)
- L1 / L2 regularization (sparsity-inducing vs magnitude-penalizing)
- AIVC_GRAD_GUARD (frozen-encoder gradient blocking mechanism in code)
- Foundation model (large pretrained model used as substrate)
- Causal attention mask `ATAC → Phospho → RNA → Protein` (architectural dependency ordering)
- Direct-effect log-FC head (decoder outputting direct perturbation effects)
- Sparse learned GRN (gene regulatory network learned with L1 sparsity)
- Neumann propagation `(I − W)⁻¹ dₚ` (closed-form perturbation flow through learned graph)

### Biology / Multi-omics
- scRNA-seq / scATAC-seq (single-cell RNA / chromatin accessibility)
- CITE-seq (RNA + surface protein)
- ASAP-seq (RNA + ATAC + surface protein, T-cell focused)
- **DOGMA-seq** (RNA + ATAC + surface protein, the Mimitou 2021 method we trained on)
- **QuRIE-seq** (Quriegen's proprietary assay: RNA + Protein + **Phospho-proteins** + ATAC[t=0,t=180]; Phase 2 adds VDJ)
- VDJ (Variable, Diversity, Joining gene rearrangement — adaptive immune receptor diversity)
- Phospho-signaling / phospho-proteomics (intracellular phosphorylated protein states — kinase pathway activation readout)
- PBMC (Peripheral Blood Mononuclear Cells)
- BTK (Bruton tyrosine kinase — BCR pathway, CLL target, Ibrutinib mechanism)
- JAK (Janus kinase — cytokine signaling, Ruxolitinib mechanism)
- CD3E, CD4, ZAP70 (TCR signaling complex components)
- NFKB, STAT3 (transcription factor hubs)
- MYD88 (innate immunity signaling)
- IRF7 (interferon response effector)
- TCR / BCR signaling (T-cell / B-cell receptor pathways)
- Synergy (drug combination response measurement — observed combo - predicted independent)
- Perturbation (CRISPR knockout, drug stimulation, knock-in — anything changing cell state)
- Pseudo-bulk centroid-NN (aggregation-then-nearest-neighbor evaluation method)
- Multi-omics integration (combining measurements across data types)

### Databases / Prior Knowledge (from audit decisions Q3)
- **ENCODE** (Encyclopedia of DNA Elements — peak annotation, used in our ATAC harmonization)
- **IMGT** (International ImMunoGeneTics — VDJ reference, Phase 2 relevant)
- **GO** (Gene Ontology — functional gene annotation, partial use Stage 3 Part 1 Report 3)
- **STRING** (Search Tool for Recurring Instances of Neighbouring Genes — PPI database, A5 GRN structural prior)
- **Reactome** (Pathway database — cell-state transition annotation, partial use)

### Statistical / Validation
- Sci/SciPlex (NEEDS CLARIFICATION FROM KINGA — likely SciPlex or sci-RNA-seq family; placeholder pending answer)
- Bootstrap CI (resampling-based confidence interval)
- Pre-registration (commit eval thresholds before seeing results — anti-p-hacking)
- Held-out test set (data not used in training, reserved for evaluation)
- Chance baseline (random-guess performance lower bound; 0.25 for 4-class)
- Random projection baseline (random linear projection — sanity check that encoder beats random features)
- TF-IDF baseline (term-frequency × inverse-document-frequency — bag-of-words baseline for "is the encoder learning anything beyond keyword frequencies")

### Phase nomenclature (CRITICAL — defined explicitly)
- **Phase 1 / Phase 2**: QuRIE-seq wet-lab data generation phases. Phase 1 = Q3 2026, 5 donors, 4 modalities. Phase 2 = 2027, 20 donors, +VDJ.
- **Stages 3a / 3b / 3c / 4 / 5**: model training/architecture phases. Stage 3a = current (Mimitou + DOGMA + Calderon work). Stage 3b = BTK+JAK demo (Q4 2026). Stage 3c = causal architecture (Q1-Q2 2027). Stage 4 = scale (2027). Stage 5 = causal-ready (2028).
- **Layer L1 / L2 / L3 / L4 / L5**: public-data dataset strategy layers used during Stage 3a public-data engine build. L1 = Mimitou in-domain CRISPR. L2 = Parse soft perturbations. L3 = Rubin B-cell ATAC (v0). L4 = deferred (no good public temporal). L5 = phospho (closed by QuRIE-seq Phase 1).

---

## Critical Risks To Track Through Phase 4

1. **Don't erase the public-data substrate.** B1/B2/B3 are real shipped evidence. Speaker notes for every slide must distinguish "validated today on public data" from "Phase 1 will demonstrate" from "Phase 2 will deliver."

2. **Phase 1 vs Phase 2 nomenclature drift.** Define once in C1 + D1 speaker notes (canonical), reference consistently elsewhere. **Don't redefine.**

3. **ATAC reduced temporal coverage is biologically motivated** — speaker notes must explain *why* (chromatin slow-varying), not just *what* (2 timepoints). Otherwise reads as cost-cutting.

4. **A5 Stage 3c timing**: phospho available Q3 2026 unlocks earlier validation possibility. Status pill currently says "Validation Q1-Q2 2027." Decide: keep "Q1-Q2 2027" as conservative validation timeline OR shift earlier to "Q4 2026 - Q1 2027" reflecting Phase 1 data availability. Recommendation: keep Q1-Q2 2027 as conservative (validation requires sufficient Phase 1 data — Phase 1 lands Q3 2026 + Q4 2026 data processing + Q1 2027 first eval).

5. **D1 milestone update**: "Phase 2 phospho on Q1'27" → must change. New label: "Phase 2 VDJ on Q1'27" or similar. Verify exact Phase 2 timing with Thiago if uncertain.

6. **Font sizing audit** (Ash's new observation 2026-05-17): some text "too small for eyes." Phase 4 must audit minimum sizes at slide-fill scale. Don't allow body text under 14pt. GRN labels on A5 need bump from ~11pt to ~13pt minimum.

7. **"Sci" reference pending** — Kinga to clarify (likely SciPlex or sci-RNA-seq). Glossary entry deferred until clarification arrives.

---

## Path Forward — Sequence (Confirmed with Ash)

```
STEP 1 (this doc) ──→ Canonical correction source written.
STEP 2 (Claude)   ──→ Update A2, C1, D1, F1 content specs + A5 speaker notes
STEP 3 (Cowork)   ──→ SVG rebuilds for A2, C1, D1, F1 (A5 unchanged)
STEP 4 (Cowork)   ──→ pptx v4 mechanical rebuild
STEP 5 (Claude + Cowork) ──→ Speaker notes glossary expansion across all 14 slides
                            + Phase 1/Phase 2 + Layer L1-L5 + Stage 3a-5 definitions in C1/D1 speaker notes
                            + three-state framing pattern applied throughout
STEP 6 (Cowork)   ──→ pptx v5 with full speaker notes
STEP 7 (Ash)      ──→ Send pptx v5 to Kinga + Jan for content review
STEP 8 (post-feedback) ──→ Phase 4A visual polish (font sizing + hero slide aesthetic improvements)
```

Estimated total: ~8-12 hours spread across 3-4 sessions.

---

## Confirmation Checklist (Going Into Step 2)

Before drafting A2/C1/D1/F1 content spec updates, confirm:

- [x] Phospho is in Phase 1 (per Kinga 2026-05-17, locked above)
- [x] ATAC measured at t=0 and t=180 only in Phase 1 (per Kinga 2026-05-17, locked above)
- [x] VDJ in Phase 2, not Phase 1 (per Thiago May 12, locked)
- [x] BTK+JAK combo in Phase 1 (per Thiago May 12, locked)
- [x] Public-data substrate stays visible across deck (three-state framing approved by Ash 2026-05-17)
- [x] Phase 1 / Phase 2 = QuRIE-seq wet-lab generation phases (canonical definition above)
- [x] 9 slides locked (A1, A3, A4, B1, B2, B3, C2, D2, E1), 4 slides need SVG updates (A2, C1, D1, F1), 1 slide needs speaker notes only (A5)
- [ ] "Sci" reference clarified (pending Kinga answer; not blocking Step 2)
- [x] Font sizing audit added to Phase 4 scope (Ash observation 2026-05-17)
