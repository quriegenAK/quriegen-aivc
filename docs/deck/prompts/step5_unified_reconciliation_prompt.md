# Step 5 — Unified Reconciliation Pass

**Owner**: Cowork (execution)
**Estimated time**: 3-4 hours (across 14 content specs + pptx v5 rebuild)
**Input commits**: `7daa106` (master glossary on origin/main)
**Strategy**: Combined sweep — stale phospho-Phase-2 cleanup + comprehensive speaker notes glossary embedding + canonical timeline enforcement

---

## Context

Pre-ship review of v4 surfaced a critical finding: the Step 2a content spec updates patched headlines and body bullets where called out, but did **NOT** sweep the full depth of each spec for residual stale phospho-Phase-2 references. A grep audit found 15+ stale references in 9 of 14 specs.

This step:
1. Applies stale-reference fixes across all affected specs (semantic consistency)
2. Embeds comprehensive technical glossary per slide (Kinga + Jan ask)
3. Enforces canonical timeline/phase narrative across the deck
4. Treats Phase 1/2 modality ownership, stage gating, timeline, budget, and architecture as **globally coupled**

After this lands: pptx v5 with reconciled content + complete speaker notes glossary. Ready for Kinga + Jan review.

---

## Canonical Source-of-Truth Reference

The master glossary at `docs/deck/research/glossary_2026_05_17.md` (commit `7daa106`) is canonical. Every term definition must trace back to it.

Two phase-related definitions are load-bearing across this pass:

**Phase 1 (Q3 2026)**: 5 donors × 5 timepoints × 4 modalities measured directly (RNA + Protein + Phospho at all 5 timepoints; ATAC at t=0 and t=180 only) + BTK+JAK combo confirmed.

**Phase 2 (Q1 2027 onwards)**: 20 donors + adds VDJ as 5th modality + extended perturbation conditions + disease-context samples follow.

**Stage 3c (Q1-Q2 2027)**: Causal architecture validation. Gated on **Phase 1 data** (phospho available Q3 2026), NOT Phase 2. The earlier "Phase 2 phospho → Stage 3c" dependency arrow is WRONG.

---

## Part 1 — Stale Reference Cleanup (Surgical Edits)

Apply these find-and-replace edits to specific files in `docs/deck/content/`.

### A1_system_architecture.md

**Line 101 area** — Find the bullet:
> Pathway-aware output dependent on phospho integration in Phase 2 (Q1 2027)

**Replace with**:
> Pathway-aware output uses phospho signal from Phase 1 (Q3 2026 — phospho is integral to QuRIE-seq); pathway dictionary expands across Phases 1-2

### A2_encoder_substrate.md

**Line 159 area** — Find the Q&A answer (likely under a "VDJ deferred" or similar question):
> Phospho doesn't exist in public PBMC data — it's QurieSeq's proprietary modality, our moat. VDJ is being deferred to QurieSeq Phase 2 per Thiago's wet-lab plan. The encoder architecture is modality-extensible by design — Phase 1 phospho and Phase 2 VDJ slot in without retraining the base encoder.

**Replace with**:
> Phospho doesn't exist in public PBMC data — it's QuRIE-seq's proprietary modality, our moat. Phospho is **integral to QuRIE-seq** and available from Phase 1 (Q3 2026). VDJ is the modality being deferred to QurieSeq Phase 2 per Thiago's wet-lab plan. The encoder architecture is modality-extensible by design — Phase 1 phospho and Phase 2 VDJ slot in without retraining the base encoder.

### A4_temporal_neural_ode.md

**Line 66 area** — Find the timepoint description bullet:
> **5 min**: early signaling (phospho-active, RNA latent) — captured by Phase 2 phospho

**Replace with**:
> **5 min**: early signaling (phospho-active, RNA latent) — captured directly by Phase 1 phospho (Q3 2026, integral to QuRIE-seq)

**Line 131 area** — Find the Q&A answer:
> Five minutes is where early signaling lives — phosphorylation cascades, second messengers, kinase activation. The phospho readouts from QurieSeq Phase 2 will populate that window with real biology. Even before phospho lands, the 5-minute sample gives the encoder a "what's already perturbed" reference point that's distinguishable from the 0-minute baseline.

**Replace with**:
> Five minutes is where early signaling lives — phosphorylation cascades, second messengers, kinase activation. The phospho readouts from QuRIE-seq Phase 1 (Q3 2026) populate that window with real biology — phospho is integral to QuRIE-seq, available from the first proprietary wet-lab batch. The 5-minute sample at the phospho level gives the encoder direct mechanistic signal in the window before transcriptional changes propagate.

### C1_phase1_design.md

**Line 188 area** — Find the elevator paragraph containing:
> 5-minute timepoint primes the dataset for Phase 2 phospho integration

**Replace that sentence with**:
> 5-minute timepoint captures Phase 1 phospho directly (phospho is integral to QuRIE-seq — every QuRIE-seq run generates phospho, including Phase 1's Q3 2026 batch).

**Line 194 area** — Find the bullet under "What's NOT on this slide":
> - The Phase 2 phospho panel details (~17 antibodies) — lives in E1 horizon slide

**Replace with**:
> - The Phase 1 phospho antibody panel specifics (~17 markers, integral to QuRIE-seq protocol) — finalization is wet-lab-spec detail, not architecture commitment

**Source data / claims table** (around line 140 area) — Find rows referencing phospho deferral or VDJ. The row:
> | VDJ deferred to Phase 2 (5th modality) | Thiago + Kinga confirmation ("VDJ later") |

is correct and stays. But also check the table for any "Phospho deferred to Phase 2" row — if present, **remove it entirely** (already corrected in Step 2a but verify).

### C2_btk_jak_demo.md

**Line 193 area** — Find the bullet under "What's NOT on this slide":
> - Phase 2 phospho extensions of the BTK+JAK demo — E1 horizon slide

**Replace with**:
> - Phase 1 phospho-channel readouts of the BTK+JAK demo (5-minute kinase activation signature) — narrative beat saved for the live walkthrough rather than the C2 slide itself

### D1_quarterly_roadmap.md (LARGEST CLEANUP — 6 STALE REFS)

**Line 28 area** — Find the body bullet:
> **Stage 3 (Q3 2026 – Q1 2027) — validation completes**: Stage 3a wraps in Q3 2026 (adapter trained on Mimitou, dress-rehearsal synergy demo). Stage 3b runs immediately as QurieSeq Phase 1 lands (Q3-Q4 2026) — the BTK+JAK headline demo. Stage 3c integrates phospho readouts in Q1 2027 as Phase 2 phospho panels arrive.

**Replace with**:
> **Stage 3 (Q3 2026 – Q1 2027) — validation completes**: Stage 3a wraps in Q3 2026 (adapter trained on Mimitou, dress-rehearsal synergy demo). Stage 3b runs immediately as QuRIE-seq Phase 1 lands (Q3-Q4 2026) — the BTK+JAK headline demo. Stage 3c (causal architecture validation: Neumann propagation + sparse learned GRN + direct-effect log-FC head) begins Q1 2027 leveraging Phase 1 phospho + perturbation data already on disk from Q3 2026. Phase 1 data gates Stage 3c, not Phase 2.

**Line 47 area** — ASCII Gantt rendering. Find:
> │ delivery  │ Phase 2 (phospho│█████████████████████████│

**Replace with**:
> │ delivery  │ Phase 2 (VDJ +  │█████████████████████████│

**Line 98 area** — Find the visual design directive bullet:
> **Dependencies should be visible** with subtle dotted arrows: Phase 1 → 3b; Phase 2 phospho → 3c; Phase 2 VDJ → Stage 4; Stage 5 causal layer connects to drug pipelines.

**Replace with**:
> **Dependencies should be visible** with subtle dotted arrows: Phase 1 → 3b; Phase 1 (phospho + perturbation data) → 3c; Phase 2 VDJ → Stage 4; Stage 5 causal layer connects to drug pipelines.

**Line 127 area** — Source data / claims table row:
> | QurieSeq Phase 2 Q1-Q2 2027 (phospho + VDJ + 20 donors) | QurieSeq roadmap (Phase 2 spec) |

**Replace with**:
> | QuRIE-seq Phase 2 Q1-Q2 2027 (VDJ + 20-donor scale) | QurieSeq roadmap (Phase 2 spec, Thiago + Kinga 2026-05-12) |

**Line 130 area** — Source data table row:
> | Stage 3c phospho integration Q1 2027 | Architecture spec v1.1, §6 |

**Replace with**:
> | Stage 3c causal architecture validation Q1-Q2 2027 (post Phase 1) | Architecture spec v1.1 §6 (v1.2 causal-layer extension pending) |

**Line 143 area** — "Three chains" Q&A answer. Find:
> Three chains. First, QurieSeq Phase 1 (Q3 2026) unlocks Stage 3b (Q4 2026, BTK+JAK demo). Without Phase 1 data, the demo slides to whenever data arrives. Second, QurieSeq Phase 2 (Q1-Q2 2027) unlocks Stage 3c phospho integration and Stage 4 VDJ. The Phase 2 data is what extends the platform from 3 to 5 modalities. Third, Stage 4 + the early drug pipeline work unlocks Stage 5's causal-readiness layer (Q1 2028+). Each major stage has one upstream dependency clearly identified.

**Replace with**:
> Three chains. First, QuRIE-seq Phase 1 (Q3 2026, 4 modalities including phospho) unlocks Stage 3b (Q4 2026, BTK+JAK demo) AND Stage 3c (Q1-Q2 2027, causal architecture validation on Phase 1 phospho signal). Without Phase 1 data, both 3b and 3c slip. Second, QuRIE-seq Phase 2 (Q1 2027 onwards, adds VDJ + 20-donor scale) unlocks Stage 4 (VDJ encoder integration + cross-donor generalization). Third, Stage 4 + early drug pipeline work unlocks Stage 5's causal-readiness layer (2028). Phase 1 is the load-bearing dependency for both Stage 3b and 3c — meaning Phase 1 delivery slippage cascades to both demo and causal milestones.

**Line 177 area** — Roadmap design Q&A. Find:
> The roadmap is 11 quarters from Q3 2026 through Q4 2028, organized across 4 swimlanes: wet lab, model architecture, drug pipelines, and publications/demos. The visual anchor is Q4 2026 — the BTK+JAK zero-shot demo, the platform's first investor-grade publication. Phase 1 QurieSeq data delivers in Q3 2026 (Thiago confirmed); Phase 2 adds phospho and VDJ in Q1-Q2 2027. Drug pipelines establish from Q1-Q2 2027 onward as the model graduates from validated-on-Phase-1 to production-ready. Stage 5 in 2028 layers causal-readiness on top — explicit support for drug-target reasoning and clinical translation framework. Dependencies are explicit: every model stage has a wet-lab dependency cleanly identified. Slippage in any one swimlane has clear contingency paths.

**Replace with**:
> The roadmap is 11 quarters from Q3 2026 through Q4 2028, organized across 4 swimlanes: wet lab, model architecture, drug pipelines, and publications/demos. The visual anchor is Q4 2026 — the BTK+JAK zero-shot demo, the platform's first investor-grade publication. Phase 1 QuRIE-seq data delivers in Q3 2026 (Thiago confirmed) with 4 modalities including phospho (integral to QuRIE-seq); Phase 2 adds VDJ + 20-donor scale in Q1-Q2 2027. Drug pipelines establish from Q1-Q2 2027 onward as the model graduates from validated-on-Phase-1 to production-ready. Stage 5 in 2028 layers causal-readiness on top — explicit support for drug-target reasoning and clinical translation framework. Dependencies are explicit: every model stage has a wet-lab dependency cleanly identified. Slippage in any one swimlane has clear contingency paths.

**Line 206 area** — Diagram design note duplicate. Find:
> Dotted dependency arrows: Phase 1 → 3b; Phase 2 phospho → 3c; Phase 2 VDJ → Stage 4; Stage 4 + pipelines → Stage 5 causal layer.

**Replace with**:
> Dotted dependency arrows: Phase 1 → 3b; Phase 1 phospho + perturbation data → 3c; Phase 2 VDJ → Stage 4; Stage 4 + pipelines → Stage 5 causal layer.

### D2_seed_allocation.md (BUDGET IMPLICATIONS)

**This spec has 4 stale phospho-Phase-2 references. Budget reframing required.**

The $1M "Phase 2 phospho prep" line item is now mostly a Phase 1 cost (phospho is integral to QuRIE-seq Phase 1, runs Q3 2026 not Q1 2027). This is a real budget timing shift, not just relabeling.

**Line 28 area** — Wet-lab allocation bullet. Find:
> **~40% wet lab (Phase 1 + Phase 2 prep)**: ~$4M. Funds QurieSeq Phase 1 delivery (Q3 2026) and Phase 2 readiness (Q1-Q2 2027). Equipment, reagents, donor procurement, CITE-seq antibody panels, BTK + JAK + other inhibitor procurement, ATAC integration pipeline, phospho panel preparation for Phase 2 onboarding.

**Replace with**:
> **~40% wet lab (Phase 1 delivery + Phase 2 prep)**: ~$4M. Funds QuRIE-seq Phase 1 delivery (Q3 2026, includes integrated phospho-proteomics panel as part of QuRIE-seq protocol) and Phase 2 readiness (Q1-Q2 2027, VDJ panel + donor scale-up to 20). Equipment, reagents, donor procurement (Sanquin), CITE-seq + phospho antibody panels, BTK + JAK + additional inhibitor procurement, ATAC integration pipeline. Phase 1 phospho is not a separate prep cost — it's part of the QuRIE-seq line.

**Line 64 area** — Allocation breakdown diagram. Find the line:
> │ • Phase 2 phospho      │  │ • BTK+JAK demo         │  │ • IP filings on        │

**Replace with**:
> │ • Phase 2 VDJ panel    │  │ • BTK+JAK demo         │  │ • IP filings on        │

**Line 104 area** — Source data / claims table row. Find:
> | Phase 2 phospho panel cost (~$1M prep) | ~17 antibody panels × validation + procurement + protocol dev |

**Replace with**:
> | Phase 1 phospho panel cost (~$1M, included in Phase 1 wet-lab line) | ~17 antibody panels × validation + procurement + protocol dev. Phase 2 adds VDJ-specific reagents (separate $0.5-1M estimate, pending wet-lab spec). |

**Line 144 area** — Elevator paragraph. Find:
> The $10M seed allocates 55% to the data + model engines that compound competitive advantage. Wet lab (~40%, $4M) funds QurieSeq Phase 1 delivery in Q3 2026 and Phase 2 onboarding (phospho + VDJ + 20 donors) in Q1-Q2 2027 — the proprietary data that makes the platform defensible. AI/ML team + compute (~25%, $2.5M) funds the 3-4 engineer team executing Stage 3a through Stage 5, including BSC compute allocation and cloud burst capacity. The remaining 35% covers wet lab team scientists (~15%), business development for pharma partnerships (~10%), and G&A + IP + legal (~10%). Every dollar maps to a roadmap milestone on slide D1. These allocations are model-grounded estimates pending final confirmation from Kinga.

**Replace with**:
> The $10M seed allocates 55% to the data + model engines that compound competitive advantage. Wet lab (~40%, $4M) funds QuRIE-seq Phase 1 delivery in Q3 2026 (4 modalities including integrated phospho-proteomics) and Phase 2 onboarding (VDJ + 20-donor scale) in Q1-Q2 2027 — the proprietary data that makes the platform defensible. AI/ML team + compute (~25%, $2.5M) funds the 3-4 engineer team executing Stage 3a through Stage 5, including BSC compute allocation and cloud burst capacity. The remaining 35% covers wet lab team scientists (~15%), business development for pharma partnerships (~10%), and G&A + IP + legal (~10%). Every dollar maps to a roadmap milestone on slide D1. These allocations are model-grounded estimates pending final confirmation from Kinga.

**Line 187 area** — Risk callout. Find:
> Phase 2 phospho prep ($1M) is a Phase 2 readiness cost partly captured in the wet lab Phase 1+2 prep bucket — overlap is acceptable for high-level allocation but final budget should disambiguate.

**Replace with**:
> Phase 1 phospho panel cost (~$1M, included in QuRIE-seq Phase 1 wet-lab line) shifts the Phase 1 budget weight earlier than the spec previously implied. Phase 1 spend now includes phospho antibodies + validation; Phase 2 adds VDJ-specific reagents. Final budget disambiguation pending Kinga's confirmation.

### E1_five_year_trajectory.md

**Line 28 area** — Find the bullet:
> **2026-2027 — Platform validation (Stage 3a/b/c)**: BTK+JAK headline demo (Q4 2026), phospho integration (Q1 2027), VDJ + 20-donor cross-disease transfer (Q2-Q4 2027). The model graduates from validated-on-Phase-1 to multi-modality, multi-disease production-ready. **Foundation locked.**

**Replace with**:
> **2026-2027 — Platform validation (Stage 3a/b/c)**: BTK+JAK headline demo (Q4 2026) on Phase 1 data (4 modalities including phospho, integral to QuRIE-seq Phase 1). Stage 3c causal architecture validation (Q1-Q2 2027) leverages Phase 1 phospho signal. VDJ + 20-donor cross-disease transfer (Q2-Q4 2027) via Phase 2. The model graduates from validated-on-Phase-1 to multi-modality, multi-disease production-ready. **Foundation locked.**

### F1_competitive_positioning.md

**Line 180 area** — Source data / claims table row:
> | Phase 2 phospho + VDJ extend the same protocol family without re-architecting | QurieSeq Phase 2 spec (Thiago + Kinga, May 12) |

**Replace with**:
> | Phase 1 (Q3 2026): 4 modalities including phospho. Phase 2 (Q1 2027+): adds VDJ + 20-donor scale within the same protocol family without re-architecting | QurieSeq Phase 1+2 spec (Thiago + Kinga, May 12) + 2026-05-17 phospho correction |

**Line 222 area** — Elevator paragraph. Find:
> No public single-cell dataset combines what causal drug combination prediction requires — multi-omics, perturbation-aware, temporal, combinatorial, protocol-aligned for modality expansion. That structural gap is why QurieSeq exists: a proprietary wet-lab platform generating data co-designed with our model architecture. Five pillars run as one integrated system — wet-lab data generation, architecture co-designed with the assay (the decomposed readout matches the 4-arm experimental design; Neural ODE matches irregular timepoint sampling), temporal multi-omics, compositional perturbation modeling, and unified protocol family for Phase 2 phospho + VDJ extension. Competitors optimize one layer each: TAHOE optimizes single-modality data scale (100M cells, cell lines); Immunai optimizes modality-rich atlases via partner data; CytoReason, Turbine, DeepLife optimize foundation models on partner-derived data; Valo and Noetik optimize downstream therapeutics. We optimize the closed-loop system itself. Every QurieSeq phase trains the next architecture extension; every architecture extension informs the next wet-lab design. The integration compounds. Modality coverage, cell counts, and partner deals are downstream properties of that integrated platform — not the headline argument.

**Replace with**:
> No public single-cell dataset combines what causal drug combination prediction requires — multi-omics, perturbation-aware, temporal, combinatorial, protocol-aligned for modality expansion. That structural gap is why QuRIE-seq exists: a proprietary wet-lab platform generating data co-designed with our model architecture. Five pillars run as one integrated system — wet-lab data generation, architecture co-designed with the assay (the decomposed readout matches the 4-arm experimental design; Neural ODE matches irregular timepoint sampling), temporal multi-omics including phospho-proteomics integral to QuRIE-seq from Phase 1 (Q3 2026), compositional perturbation modeling, and unified protocol family for Phase 2 VDJ extension. Competitors optimize one layer each: TAHOE optimizes single-modality data scale (100M cells, cell lines); Immunai optimizes modality-rich atlases via partner data; CytoReason, Turbine, DeepLife optimize foundation models on partner-derived data; Valo and Noetik optimize downstream therapeutics. We optimize the closed-loop system itself. Every QuRIE-seq phase trains the next architecture extension; every architecture extension informs the next wet-lab design. The integration compounds. Modality coverage, cell counts, and partner deals are downstream properties of that integrated platform — not the headline argument.

---

## Part 2 — Speaker Notes Glossary Embedding (All 14 Slides)

For each content spec, **REPLACE the existing `## Speaker notes` section** with the structure below. Existing Q&As are PRESERVED — they get integrated into the new structure under the "Diligence Q&A" subsection.

### Per-slide structure (canonical)

```markdown
## Speaker notes

### Three-state framing (where applicable)

[Today / Phase 1 / Phase 2 framing block tied to this slide's content]

### Technical glossary (terms appearing on this slide)

[6-12 terms with self-contained 2-3 sentence definitions pulled from master glossary]

### Equations & notation (where applicable)

[Reading-order explanation of any math on the slide]

### Diligence Q&A

[All existing Q&As preserved + new ones where corrections work surfaced gaps]
```

Below, per-slide speaker notes content to embed.

---

### A1 — System Architecture

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (public substrate validated)**: Encoder pretrained on DOGMA-seq (Mimitou 2021, 3 modalities: RNA + ATAC + Protein), cross-corpus validated at 73% on Calderon 2019. Adapter probed on Mimitou CRISPR perturbations at 0.57 (ADAPTER_RECOMMENDED verdict). This 5-block architecture is built and shipped.
- **Phase 1 (Q3 2026)**: QuRIE-seq Phase 1 delivers proprietary data — RNA + Protein + Phospho at all 5 timepoints (0/5/30/60/180), ATAC at t=0 and t=180. Architecture stays unchanged; new modality (phospho) plugs in as 4th encoder input head.
- **Phase 2 (2027)**: VDJ adds as 5th modality. Encoder grows under the same protocol family — no re-architecting required.

### Technical glossary
**Foundation model** — Large pretrained model providing general representations for many downstream tasks via fine-tuning or adapter layers. Our encoder is a foundation model for PBMC multi-omics.

**Frozen encoder** — Encoder weights locked during downstream training. Only adapter and decoder heads update. Prevents encoder representations from drifting during task-specific training. Enforced by AIVC_GRAD_GUARD.

**Adapter** — Small lightweight neural network layer trained on top of a frozen pretrained encoder. The encoder provides general representations; the adapter learns task-specific behavior. Approved by Stage 3 Part 1 verdict.

**Trimodal encoder** — Encoder operating on 3 modalities (RNA + ATAC + Protein). "Trimodal" refers strictly to the Mimitou 2021 DOGMA-seq pretraining protocol. Today's deployed encoder is trimodal; Phase 1 expands to 4 modalities (+phospho) without retraining the backbone.

**Latent space 256-D** — The 256-dimensional vector representation produced by the encoder. Each cell maps to a point in this space. Smaller than typical transformer hidden states because multi-omics input is more constrained than natural language.

**Neural ODE (Neural Ordinary Differential Equation)** — Continuous-time dynamics model. Instead of discrete time steps, the latent state evolves according to a learned differential equation. Handles irregular timepoint spacing (our 0/5/30/60/180 min sampling has unequal gaps) natively.

**4-arm decomposed readout** — Decoder architecture that decomposes a perturbed cell's predicted state as `h_base + Δ_stim + Δ_inh + Δ_synergy`. The synergy arm captures the non-additive part of combinations — enables zero-shot prediction of unseen combinations.

**DOGMA-seq** — Triple-modality single-cell method measuring RNA + ATAC + surface protein on the same cell. From Mimitou 2021 (Nature Biotechnology). Source of our encoder pretraining data.

**Calderon 2019** — Published PBMC dataset under stimulation, used as our cross-corpus hold-out test for the encoder. 73% pseudo-bulk centroid-NN accuracy demonstrates cross-corpus generalization.

**Pathway-aware output** — Decoder produces outputs aligned to known biological pathways (GO + Reactome). Phase 1 phospho data + ATAC priors expand the pathway dictionary.

### Diligence Q&A

[PRESERVE all existing Q&As from the current A1 speaker notes section. If A1 currently has no Q&As — populate with these three]:

**If asked: "Why one model for all PBMC cell types?"**
> The encoder is trained on cell-type-mixed multi-omics data and represents each cell in a shared 256-D latent space. Cell type emerges as structure within that space (validated by the 73% Calderon cross-corpus result on 5-class lineage classification). One unified model means downstream tasks (perturbation prediction, temporal evolution, causal inference) operate on a single coherent substrate rather than per-cell-type specialized models. Cheaper to maintain, more cross-type generalization.

**If asked: "Where does the model see phospho if the encoder was pretrained on RNA + ATAC + Protein only?"**
> Phospho enters the architecture in Phase 1 (Q3 2026) as a new input head added to the encoder. The pretrained RNA + ATAC + Protein backbone stays frozen (AIVC_GRAD_GUARD enforced) while a phospho-specific encoder layer fits to the QuRIE-seq Phase 1 data. This is the same adapter-strategy pattern that the Stage 3 Part 1 verdict approved. Phase 2 extends similarly with VDJ.

**If asked: "What's pre-registered about this?"**
> The Stage 3 evaluation methodology was committed in writing before running the evals — cross-corpus pseudo-bulk centroid-NN as the eval method, threshold bands locked, no post-hoc threshold adjustment. The 73% Calderon result and 0.57 Mimitou CRISPR result both came from pre-registered evals. Stage 3b (BTK+JAK demo) and Stage 3c (causal architecture) also have pre-registered thresholds in architecture spec v1.1.
```

---

### A2 — Multi-Omics Encoder

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (shipped evidence)**: Encoder trained on DOGMA-seq, 73% Calderon cross-corpus accuracy (pre-registered eval, pseudo-bulk centroid-NN). Three modalities: RNA + ATAC + Protein. Public data, public benchmarks. The 73% on the right side of the slide is the shipped result.
- **Phase 1 (Q3 2026 — QuRIE-seq lands)**: Adds phospho as 4th modality (integral to QuRIE-seq). The encoder grows a phospho input head; backbone stays frozen. RNA + Protein + Phospho at 5 timepoints; ATAC at t=0 and t=180 (chromatin slow-varying).
- **Phase 2 (2027 — scale + VDJ)**: Adds VDJ as 5th modality + scale to 20 donors. Same protocol family, same encoder backbone, new input head only.

### Technical glossary
**Multi-omics encoder** — Neural network that takes multi-modality single-cell measurements (RNA + ATAC + Protein, growing to +Phospho +VDJ) and produces a unified vector representation in a shared 256-D latent space.

**Contrastive multi-omics fusion** — The encoder learns to align representations of the same cell across modalities (positive pairs) while differing across cells (negative pairs). Multi-modal contrastive learning produces the 256-D latent.

**Latent space 256-D** — The 256-dimensional vector representation produced by the encoder. Dimensionality chosen to balance expressiveness (capacity to represent cell state) with computational cost (downstream inference speed, GPU memory).

**Pseudo-bulk centroid-NN** — Cross-corpus evaluation method. Aggregate single cells by cell-type label within each dataset to produce one centroid vector per cell type (pseudo-bulk). Then for each held-out test centroid, find the nearest centroid in the training pool by cosine distance (centroid-NN). Accuracy = fraction where the match is the same label.

**73% Calderon (cross-corpus)** — Pseudo-bulk centroid-NN accuracy on 5-class PBMC lineage classification (B / T / NK / monocyte / DC) when training on Mimitou DOGMA-seq and evaluating on independently-generated Calderon 2019. Chance baseline is 20%. The 73% = 3.65× chance.

**Cross-corpus generalization** — Property of a model trained on one dataset generalizing to a different, independently-generated dataset without retraining. Tests for spurious dataset-specific features that would inflate within-dataset accuracy.

**Pre-registered evaluation** — Eval methodology, metric, and thresholds committed in writing before running the eval. Prevents post-hoc cherry-picking. Our 73% result was pre-registered before we ran it.

**Frozen encoder + adapter** — After encoder pretraining, encoder weights are locked. Downstream tasks (perturbation prediction, etc.) train a small adapter on top instead of fine-tuning the encoder. Cheaper, preserves general representations.

**AIVC_GRAD_GUARD** — Environment variable flag (`AIVC_GRAD_GUARD=1`) that blocks gradient flow into the encoder during downstream training. Enforces frozen-encoder discipline mechanically. Set in all production training runs after Stage 3 Part 1 verdict.

**DOGMA-seq** — Single-cell method measuring RNA + ATAC + surface protein on the same cell. From Mimitou 2021 (Nature Biotechnology). Our encoder pretraining dataset.

**ASAP-seq** — Single-cell method measuring ATAC + surface proteins simultaneously. Variant of CITE-seq. Mimitou's CRISPR sub-study used ASAP-seq with hashtag-oligo-encoded CRISPR perturbations.

### Diligence Q&A

[PRESERVE all existing Q&As. Existing important ones include phospho-Phase-1 framing (post-correction), encoder modality extension mechanism, pseudo-bulk centroid-NN definition, AIVC_GRAD_GUARD. Cleanup the line-159 stale answer per Part 1 instructions above. All other existing Q&As stay verbatim.]
```

---

### A3 — Decomposed Readout

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (architecture shipped)**: 4-arm decomposed readout is implemented and trained on Mimitou CRISPR data. CD3E + CD4 single perturbations train the synergy head; CD3E×CD4 double-knockout is the held-out test for compositional generalization.
- **Phase 1 (Q3 2026)**: Same 4-arm readout trains on QuRIE-seq Phase 1 perturbation panel. BTK alone + JAK alone train the inhibitor arms; BTK+JAK combo is the held-out test for the Stage 3b demo. Synergy head's zero-arm constraint becomes the load-bearing claim.
- **Phase 2 (2027)**: 4-arm readout extends to additional inhibitor combinations and donor-level cross-validation across the 20-donor scale.

### Technical glossary
**Decomposed readout (4-arm)** — Decoder architecture: predicted response = `h_base + 𝟙[s]·Δ_stim + 𝟙[i]·Δ_inh + 𝟙[s∧i]·Δ_synergy`. Four learned heads, parallel branches, summed at output. Each head models one component independently.

**Synergy** — When the combined effect of two perturbations exceeds the sum of their individual effects: `Δ_combo > Δ_drug1 + Δ_drug2`. The Δ_synergy arm captures this directly. Mathematically, synergy = the non-additive part of combination biology.

**Zero-arm constraint** — A penalty (L2, λ=1.0) forcing the synergy head to output zero when stimulus or inhibitor is absent. This isn't regularization for stability — it's the architectural choice that forces the synergy head to learn ONLY the non-additive correction, which enables compositional generalization.

**Compositional generalization** — Model's ability to predict combinations from singletons. Train on BTK alone + JAK alone, predict BTK+JAK combo response. The 4-arm decomposition + zero-arm constraint structurally supports this.

**Indicator function 𝟙[s] (Iverson bracket)** — Mathematical notation. `𝟙[s] = 1 if condition s is true, 0 if false`. In our equation: `𝟙[s]` is 1 when stimulus is present, 0 otherwise. Switches arms on/off based on experimental condition.

**Δ (Delta)** — "Change" or "difference". Used in decomposed readout for individual perturbation effects: Δ_stim = stimulus contribution, Δ_inh = inhibitor contribution, Δ_synergy = synergy correction beyond additive.

**L2 regularization** — Penalty on the sum of squared weight values. Used here at λ=1.0 to enforce the zero-arm constraint on the synergy head when single arms are absent.

**Perturbation embedding** — Vector representation of a perturbation context (which drug, at what concentration, for what duration). Combined with cell latent state to predict response. Stored in the readout's Δ arms.

### Equations & notation

**Reading the decomposed readout equation**:
```
ŷ(z, s, i, t) = h_base(z, t)
              + 𝟙[s]·Δ_stim(z, s, t)
              + 𝟙[i]·Δ_inh(z, i, t)
              + 𝟙[s∧i]·Δ_synergy(z, s, i, t)
```

- `ŷ` (y-hat) = predicted response
- `z` = cell latent state (from encoder)
- `s` = stimulus identifier (vector)
- `i` = inhibitor identifier (vector)
- `t` = timepoint
- `h_base(z, t)` = vehicle-control baseline at time t
- `𝟙[s]` = 1 if stimulus s is present, 0 otherwise (Iverson bracket / indicator)
- `Δ_stim(z, s, t)` = additional response contributed by stimulus alone
- `Δ_inh(z, i, t)` = additional response contributed by inhibitor alone
- `Δ_synergy(z, s, i, t)` = non-additive synergy correction when both present
- `s ∧ i` = "both s and i present" (logical AND)

The architectural commitment: synergy head outputs zero when either single arm is absent (`𝟙[s∧i] = 0`), forced by L2 penalty during training. This means the synergy head can only learn the non-additive part — making zero-shot combination prediction possible.

### Diligence Q&A

[PRESERVE all existing Q&As verbatim from current A3 speaker notes section.]
```

---

### A4 — Temporal Neural ODE

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today**: Neural ODE temporal backbone implemented and trained on Mimitou CRISPR + DOGMA-seq data. Continuous-time state evolution validated architecturally — irregular timepoints handled natively.
- **Phase 1 (Q3 2026)**: 5-timepoint Phase 1 design (0/5/30/60/180 min) gives Neural ODE 5 anchor points per donor per perturbation. 5-minute timepoint captures phospho early-signaling (phospho is integral to QuRIE-seq Phase 1) — directly populates the early-signaling window with real biology.
- **Phase 2 (2027)**: 20-donor scale gives cross-donor temporal validation. Same Neural ODE architecture, more donors, no re-architecting.

### Technical glossary
**Neural ODE (Neural Ordinary Differential Equation)** — Continuous-time dynamics model. Instead of discrete time steps (like RNN/transformer), the latent state evolves according to a learned differential equation `dz/dt = f_θ(z, perturbation, t)`. Time is a first-class input, not a discrete index.

**Latent SDE (Latent Stochastic Differential Equation)** — Probabilistic temporal model where latent state evolves with both deterministic drift and stochastic diffusion. Architecture spec v1.1 §7.1 documents this as fallback if Neural ODE proves insufficient for biological noise levels.

**Continuous-time** — Time is a real-valued input variable, not a discrete step. Allows querying the model at any timepoint (e.g., predict state at t=12.5 min) without retraining.

**Irregular timepoint spacing** — Sampling times that don't divide evenly. Our 0/5/30/60/180 min has gaps of 5, 25, 30, 120 minutes — radically unequal. Neural ODE handles this natively; discrete-time models would require resampling or interpolation.

**5-minute timepoint** — Captures early signaling biology (phospho-active, RNA still latent). In QuRIE-seq Phase 1, this gives the encoder a "what's already happening" signal with real phospho-proteomics measurement (phospho is integral to QuRIE-seq).

**30-minute timepoint** — Captures transcriptional onset. RNA changes are detectable; phospho signal is decaying or saturating.

**180-minute timepoint** — Captures stable response phenotype. Both transcription and chromatin remodeling visible. ATAC measurement only at t=0 and t=180 in Phase 1 because chromatin changes slowly relative to the other modalities.

**RSSM (Recurrent State Space Model)** — Considered as alternative temporal architecture, rejected because discrete time steps make irregular sampling clunky.

**Transformer-over-timesteps** — Considered as alternative, rejected because attention over 5 timepoints is overkill and the architecture doesn't naturally handle the temporal causality direction.

### Diligence Q&A

[PRESERVE all existing Q&As. Apply line-66 and line-131 stale references fixes per Part 1 above.]
```

---

### A5 — Causal Architecture

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (spec-locked, validation pending)**: Stage 3c causal architecture is committed in architecture spec v1.1 (v1.2 §X causal-layer extension pending). Neumann propagation, sparse learned GRN with STRING prior, direct-effect log-FC head — all have concrete mathematical definitions. Not yet implemented in code.
- **Phase 1 (Q3 2026)**: Phospho signal becomes available (integral to QuRIE-seq). Stage 3c architecture validation can begin Q4 2026 once Phase 1 data lands. STRING database (PPI prior) integrated for GRN edge initialization.
- **Phase 2 (2027)**: Stage 3c validation reaches publishable result quality on Phase 1+2 combined data. Stage 5 (2028) extends causal architecture with clinical-readiness features.

### Technical glossary
**Causal architecture** — Layer of the platform that produces causal inference rather than only correlative prediction. Distinguishes "what does X cause?" (Stage 3c) from "what happens after X?" (Stage 3a/3b).

**Neumann propagation** — Mathematical technique for computing perturbation flow through a graph. The closed-form `(I − W)⁻¹ dₚ` solves the linear system "what happens at every node given a direct effect dₚ?". Requires spectral radius `ρ(W) < 1` for the series to converge.

**Sparse learned GRN (Gene Regulatory Network)** — Matrix W where W_ij represents the directed regulatory influence of gene i on gene j. "Sparse" because most entries are zero (enforced by L1 regularization + STRING structural prior). "Learned" because non-zero entries are inferred from perturbation-response data.

**STRING database (v12.0)** — Protein-Protein Interaction database (Szklarczyk et al., 2023, Nucleic Acids Research). Provides edge-existence priors for our sparse GRN. STRING-supported edges face lower L1 sparsity pressure; novel edges remain learnable but face higher evidence thresholds.

**Structural prior** — Information about graph topology incorporated before learning. STRING provides our structural prior on GRN edge existence. Different from learned weights — structure shapes initialization, weights update during training.

**Direct-effect log-FC head** — Decoder outputting direct (immediate) perturbation effect in log fold-change units. Stage 3a/3b predicted abundance changes; Stage 3c separates the direct effect `dₚ` from the Neumann-propagated downstream effect.

**Log-FC (log fold-change)** — Standard unit for expression changes. `log_2(post / pre)` where post and pre are expression levels after and before perturbation. Positive = upregulation, negative = downregulation, zero = no change.

**Spectral radius ρ(W)** — Largest absolute eigenvalue of matrix W. Architectural requirement: `ρ(W) < 1` for Neumann series to converge. Enforced by L1 sparsity during GRN learning.

**L1 regularization** — Penalty on sum of absolute weight values. Encourages sparsity — many weights pushed to zero. Used in sparse GRN to enforce that most gene-gene edges are zero, retaining only the strongest learned relationships.

**Stage 3c** — Model training stage focused on causal architecture validation. Q1-Q2 2027. Gated on Phase 1 data (NOT Phase 2 — phospho is in Phase 1).

**Spec-locked** — Architectural commitment is written down in spec v1.1 with concrete mathematical definitions. Implementation and validation still pending. The status pill on the slide signals this honestly.

**8-node GRN visualization** — Illustrative network using BTK, JAK, CD3E (perturbation targets), NFKB, STAT3 (transcription factor hubs), ZAP70, MYD88 (kinases), IRF7 (effector). Actual learned GRN will have N >> 8 nodes; this is for visualization clarity.

**Counterfactual** — A prediction of what would happen under a different perturbation than what was observed. "If we had perturbed gene Y instead of gene X, what would the response look like?" Stage 3c architecture supports counterfactual queries; Stage 3a/3b supports only direct perturbation predictions.

### Equations & notation

**Reading the Neumann propagation equation**:
```
ŷ = (I − W)⁻¹ · dₚ
```

- `ŷ` = predicted full propagated response across all nodes in the network
- `I` = identity matrix (1s on diagonal, 0s elsewhere — represents "self" with no propagation)
- `W` = learned sparse GRN matrix; `W_ij` = directed influence of gene i on gene j
- `(I − W)` = matrix difference; `(I − W)⁻¹` = its inverse
- `dₚ` = direct perturbation effect vector (output of the log-FC head); subscript `p` means "perturbation"
- `·` = matrix-vector multiplication

**Interpretation in plain English**: A perturbation hits gene `p` directly (direct effect `dₚ`). Through the network, the effect propagates to neighbors, then their neighbors, etc. The Neumann series `(I + W + W² + W³ + ...)` adds up all these propagation paths. When `ρ(W) < 1`, this series converges to the closed-form `(I − W)⁻¹`. This is mathematically equivalent to running an infinite simulation but in one matrix inverse.

**Architectural requirement: `ρ(W) < 1`** — Spectral radius bounded below 1. Ensures the Neumann series converges (otherwise the propagation diverges to infinity). Enforced by L1 sparsity during training, which keeps W weights small.

### Diligence Q&A

[PRESERVE all 8 existing Q&As. The 2 timing-language Q&As were already updated in Step 2 to reflect phospho-in-Phase-1.]
```

---

### B1 — Methodology Rigor

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (public-data substrate)**: B1 IS the public-data evidence slide. Three datasets — DOGMA-seq (Mimitou 2021), Calderon 2019, Mimitou ASAP-seq CRISPR — each with a specific role. Methodology pre-registered. This is shipped evidence; not roadmap.
- **Phase 1 (Q3 2026)**: QuRIE-seq Phase 1 data supersedes/complements the public-data work. Encoder retrained or fine-tuned on Phase 1 data (subject to Stage 3a/3b decisions). The pre-registered eval methodology framework extends to Phase 1 — same discipline.
- **Phase 2 (2027)**: B1's methodology pattern continues — pre-registration before evals, hold-out test sets, no cherry-picking.

### Technical glossary
**Pre-registered evaluation** — Eval methodology, metric, and thresholds committed in writing before running the eval. Documented in architecture spec v1.1. Prevents result-driven cherry-picking. Both our 73% Calderon and 0.57 Mimitou CRISPR results were pre-registered.

**Three-dataset role separation** — DOGMA-seq for pretraining, Calderon 2019 for validation, Mimitou ASAP-seq CRISPR sub-study for perturbation probe. Different studies, different donors, different protocols — prevents within-dataset overfitting from inflating cross-validation metrics.

**DOGMA-seq (Mimitou 2021)** — Triple-modality single-cell method (RNA + ATAC + Protein on same cell) from Mimitou et al., Nature Biotechnology 2021. Encoder pretraining source.

**Calderon 2019** — Published PBMC dataset under stimulation. Independent from Mimitou — different lab, different donors, different protocol. Used as cross-corpus hold-out test.

**Mimitou ASAP-seq CRISPR sub-study** — Sub-study of the Mimitou 2021 paper with ATAC + Protein + HTO-encoded CRISPR perturbations on T cells. Used for our Stage 3 Part 1 encoder probe.

**HTO (HashTag Oligonucleotide)** — Short DNA barcode used to multiplex samples in single-cell experiments. In Mimitou ASAP-seq CRISPR, HTOs encode which CRISPR guide perturbed each cell.

**Hold-out test set** — Data reserved from training and validation, used only for final evaluation. Prevents test-set leakage and inflated metrics.

**Pseudo-bulk centroid-NN** — Aggregation-then-nearest-neighbor evaluation. Aggregate single cells by cell-type label to produce centroids; nearest-neighbor match across datasets gives accuracy.

**Bootstrap confidence interval (Bootstrap CI)** — Statistical method for estimating uncertainty by resampling data many times and recomputing the metric. Used for our 73% and 0.57 result uncertainty bands.

**Chance baseline** — Lower-bound accuracy from random guessing. 5-class chance = 20%; 4-class chance = 25%. Results must exceed chance to demonstrate signal.

**Random projection baseline** — Sanity check. Replace encoder with random linear projection. Measures whether encoder learns anything beyond random features.

**TF-IDF baseline (Term Frequency × Inverse Document Frequency)** — Bag-of-words text-style baseline. Treats each gene as a token; measures whether encoder learns more than gene-frequency patterns.

### Diligence Q&A

[PRESERVE all existing Q&As verbatim.]
```

---

### B2 — Encoder Probe Verdict

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today**: 0.57 result is shipped, peer-reviewable, locked. ADAPTER_RECOMMENDED verdict drives architecture decisions. The 0.57 hero number is from real Mimitou CRISPR probe completed before this slide was drafted.
- **Phase 1 (Q3 2026)**: Stage 3b demo runs on Phase 1 data using the adapter architecture B2's verdict approved. The same pre-registration pattern applies — BTK+JAK combo held out, synergy head predicts zero-shot.
- **Phase 2 (2027)**: Adapter strategy validated by Phase 1 results may be revised (frozen vs full fine-tune) based on Phase 1 outcomes. Decision logic pre-registered.

### Technical glossary
**ADAPTER_RECOMMENDED verdict** — Pre-registered Stage 3 Part 1 outcome. If encoder probe accuracy on perturbed cells is in 0.50-0.80 range, the adapter strategy is approved. Above 0.80 = encoder generalizes natively without adapter. Below 0.50 = encoder needs full fine-tune. Our result: 0.57 → adapter approved.

**0.57 synergy 4-class accuracy** — Mimitou CRISPR probe result. 4 classes: CD3E knockout, CD4 knockout, CD3E+CD4 double knockout, non-targeting control (NTC). Chance baseline = 0.25 for 4-class. Our 0.57 = 2.27× chance.

**Frozen encoder probe** — Test of encoder generalization without modifying encoder weights. Run encoder on held-out perturbation data, score classification accuracy. Pure generalization test — no retraining.

**Per-class accuracy** — Accuracy broken down by class. Our per-class: CD3E = 0.91 (high), CD3E+CD4 double = 0.68, NTC = 0.39, CD4 = 0.39. Reveals which classes are easier/harder for the encoder.

**Random projection baseline (0.29)** — Replace encoder with random linear projection of input features. Score should be at or near chance (0.25 for 4-class). Our 0.29 indicates the encoder is doing more than a random feature extraction.

**TF-IDF baseline (0.50)** — Bag-of-words baseline on raw input features. Encoder approaches but doesn't exceed this — indicates the encoder is capturing input-level patterns without significant added signal beyond bag-of-words. This is the architectural read: encoder representations are roughly equivalent to gene-frequency vectors for this task, suggesting adapter strategy (rather than full fine-tune) is appropriate.

**Adapter strategy (~130K parameters)** — Lightweight neural network layer trained on top of frozen encoder. ~130K parameters vs encoder's millions. Trains in minutes vs hours. Approved by B2 verdict.

**Pre-registered thresholds** — Verdict thresholds locked in architecture spec v1.1 before the eval was run. Spec says: ≥0.80 = FROZEN_ENCODER_OK, 0.50-0.80 = ADAPTER_RECOMMENDED, <0.50 = FINE_TUNE_REQUIRED. No post-hoc adjustment.

**CD3E knockout / CD4 knockout** — CRISPR perturbations on T-cell receptor complex components. CD3E and CD4 are markers and signaling components in T cells.

**NTC (Non-Targeting Control)** — Control perturbation that doesn't actually disrupt any gene. Used as baseline.

### Diligence Q&A

[PRESERVE all existing Q&As verbatim.]
```

---

### B3 — Synergy Pre-Demo

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (public-data substitute)**: Mimitou CD3E + CD4 single knockouts in training; CD3E×CD4 double-KO held out for zero-shot test. Architecturally equivalent to BTK+JAK demo but on public data — dress rehearsal.
- **Phase 1 (Q3 2026)**: Same exact architecture trains on Phase 1 BTK alone + JAK alone, predicts BTK+JAK combo zero-shot. Stage 3b demo target ≥0.70 accuracy. Pre-registered.
- **Phase 2 (2027)**: Synergy framework extends to additional drug combinations beyond BTK+JAK. Multi-drug compositional generalization.

### Technical glossary
**Mimitou CRISPR substitute pattern** — Using CD3E×CD4 double-knockout as a public-data analog to BTK+JAK. Both are perturbation pairs where each single arm is in training and the combination is held out. Mathematically equivalent test of compositional generalization.

**Zero-shot synergy prediction** — Predicting the response to a perturbation combination never seen during training, using only observations of individual perturbation arms. Enabled by the 4-arm decomposed readout + zero-arm constraint.

**Dress-rehearsal protocol** — Test the architecture on public data before applying it to proprietary Phase 1 data. Validates the mechanism is sound; only the data type changes.

**CD3E + CD4 double-knockout** — Mimitou CRISPR experimental condition. Both CD3E and CD4 disrupted simultaneously. Used as public-data substitute for combination perturbation.

**ZAP70 / NFKB2 (other Mimitou single KOs)** — Additional Mimitou single-KO arms used in adapter training. ZAP70 = T-cell receptor signaling kinase. NFKB2 = transcription factor.

**Stage 3a pre-registered threshold (≥0.70)** — Pre-registered success bar for dress-rehearsal zero-shot synergy accuracy on Mimitou data. Set in architecture spec v1.1 before running the eval.

**BTK + JAK demo (Phase 1)** — The investor-grade headline result. Train on Phase 1 BTK alone + JAK alone (plus other singles), hold out BTK+JAK combo, predict combo response zero-shot. Pre-registered. Grounded in Ibrutinib+Ruxolitinib CLL clinical literature.

**Ibrutinib** — FDA-approved BTK inhibitor. Standard CLL therapy. The "B" in our BTK demo target.

**Ruxolitinib** — FDA-approved JAK inhibitor. Approved for myelofibrosis. The "J" in our JAK demo target.

**NCT02912754** — Clinical trial identifier for Ibrutinib + Ruxolitinib combination in CLL. Our BTK+JAK demo connects to this trial's biology.

**PMID 26819050** — PubMed ID for published paper on BTK+JAK combination clinical evidence in CLL.

**pJAK1/BCR pathway finding** — Thiago's prior wet-lab observation: JAK1 phosphorylation responds to BCR pathway activity. Underlying biological mechanism for BTK+JAK synergy.

### Diligence Q&A

[PRESERVE all existing Q&As verbatim.]
```

---

### C1 — Phase 1 Experimental Design

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (public-data substrate validated)**: Encoder trained on DOGMA-seq, 73% Calderon, 0.57 Mimitou CRISPR. Public-data work is shipped. C1 is about what Phase 1 ADDS.
- **Phase 1 (Q3 2026 — THIS SLIDE)**: 5 donors × 5 timepoints × 4 modalities (RNA + Protein + Phospho all 5 timepoints; ATAC at t=0 and t=180) + BTK+JAK combo confirmed. Phospho is integral to QuRIE-seq — first time we have phospho.
- **Phase 2 (2027)**: 20 donors + VDJ as 5th modality. Disease-context samples. Same protocol family.

### Technical glossary
**QuRIE-seq** — Quriegen's proprietary single-cell multi-omics assay. Measures RNA + Protein + Phospho-proteins from the same cell. Phospho is integral to the protocol. Phase 1 (Q3 2026) is the first wet-lab batch.

**Multi-omics integration** — Combining measurements across data types (RNA, ATAC, Protein, Phospho) on the same cell. Our platform's defining capability — most platforms measure 1-2 modalities; we measure 4 directly + ATAC integration from public data.

**Phospho-proteomics (phospho)** — Measurement of phosphorylated proteins. Reveals kinase activation state — immediate signaling response. Faster than RNA changes (minutes vs hours). Integral to QuRIE-seq; first publicly available to our team in Phase 1.

**Surface protein (CITE-seq)** — Surface markers measured via antibody-derived tags. Distinguished from intracellular phospho-proteins. CITE-seq panel ~30-210 markers.

**Chromatin accessibility (ATAC)** — Which DNA regions are open for transcription factor binding. Slow-varying — chromatin remodels on hour timescales. Justifies sampling only at t=0 (baseline) and t=180 (endpoint) in Phase 1.

**ATAC slow-varying rationale** — Biological justification for sampling ATAC at only 2 timepoints. Chromatin accessibility doesn't change meaningfully at 5/30/60 min intervals. Endpoint coverage captures the perturbation-induced chromatin shift; baseline anchors donor-level variation.

**5 timepoints (0/5/30/60/180 min)** — Phase 1 sampling design. 0 = baseline, 5 = early signaling (phospho-active), 30 = transcriptional onset, 60 = stable phenotype emerging, 180 = stable response. Matches Neural ODE temporal backbone.

**4-arm experimental structure** — Vehicle / stimulus / inhibitor / inhibitor+stim. Each cell measured under one of these conditions. Matches the 4-arm decomposed readout architecture exactly — wet lab and model are co-designed.

**BTK + JAK combination (Stage 3b headline)** — Confirmed for Phase 1 by Thiago (May 12, 2026). Validates compositional generalization. Connects to Ibrutinib+Ruxolitinib CLL clinical evidence.

**Sanquin** — Dutch national blood bank. Phase 1 blood source. Returns blood-type-only metadata per privacy. Our model uses ATAC at t=0 as chromatin-grounded donor signature.

**~5k cells/donor/timepoint** — Cell yield per QuRIE-seq sample. 5 donors × 5 timepoints × 5k cells × ~17 conditions ≈ ~125k cells in the Phase 1 dataset.

**Perturbation panel** — ~15-20 conditions total (vehicle, stimulus, inhibitor singles, inhibitor combinations including BTK+JAK). Exact panel size under final wet-lab spec review with Thiago.

**Stimulus** — Treatment that activates immune cells (e.g., PMA/Iono or specific cytokines). Triggers the perturbation response we measure.

**Inhibitor** — Small molecule blocking a specific protein function. BTK inhibitor, JAK inhibitor in our Phase 1 panel.

**Donor chromatin signature** — ATAC profile at t=0 functions as biological donor identifier when demographic metadata is unavailable. Replaces age/sex/ethnicity features.

### Diligence Q&A

[PRESERVE all existing Q&As (post-Step-2 update). Particularly the new ones: QuRIE-seq definition, ATAC reduced timepoints biological rationale, public-data foundation preservation, phospho mechanism, 5-donor variance discipline, Sanquin + ATAC donor signature, BTK+JAK clinical context, panel size framing.]
```

---

### C2 — BTK+JAK Demo Plan

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (architecture ready, demo pre-registered)**: Decomposed readout + adapter trained on Mimitou. Stage 3b methodology pre-registered. Awaits Phase 1 data.
- **Phase 1 (Q4 2026 — DEMO RUNS)**: Train adapter + readout on Phase 1 single-arm data (BTK alone, JAK alone, other singles, 4-arm controls). Hold out BTK+JAK combo. Predict zero-shot. Verdict mechanical.
- **Phase 2 (2027)**: Drug combination prediction extends to additional therapeutic targets. Synergy framework validated; new combinations slot in without re-architecting.

### Technical glossary
**Zero-shot synergy prediction** — Predicting combination response without seeing the combination during training. Trained on singles, predicting combos. Our headline capability.

**Held-out combination arm** — During training, BTK+JAK combo data is removed entirely. At test time, the model predicts the held-out combo response. Pure compositional generalization test.

**BTK inhibitor + JAK inhibitor (Ibrutinib + Ruxolitinib)** — FDA-approved drug pair. Combination has CLL clinical evidence (NCT02912754, PMID 26819050). Our Phase 1 demo target.

**Pre-registered eval (Stage 3b)** — Eval methodology + threshold (≥0.70 synergy accuracy) locked in architecture spec v1.1 §5.1 before Phase 1 data arrives. Mechanical verdict on demo success.

**Stage 3b demo** — Q4 2026 milestone. The first investor-grade demonstration of the platform's compositional capability. Headline result for fundraising narrative.

**pJAK1/BCR pathway** — Thiago's wet-lab observation: JAK1 phosphorylation responds to BCR pathway activity. Mechanistic basis for expecting BTK + JAK synergy.

**CLL Phase Ib/II trial** — Clinical trial design for Ibrutinib + Ruxolitinib combination. Our predictive framework references this clinical biology.

**Latent SDE fallback** — Architecture spec v1.1 §7.1 documents Latent Stochastic Differential Equation as fallback if Neural ODE proves insufficient for Phase 1 phospho dynamics. Spec'd before Phase 1 data arrives.

**Compositional eval** — Holding out a combination, training on singles, scoring the combination prediction. Mathematical proof of compositional generalization.

**4-arm controls** — Vehicle / stim / single-inhibitor / combination. Each cell labeled with its arm. The arm structure matches the decomposed readout's arm structure.

### Diligence Q&A

[PRESERVE all existing Q&As verbatim. Apply line-193 stale reference fix per Part 1.]
```

---

### D1 — Quarterly Roadmap

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today (current Stage 3a work)**: Public-data engine build in flight. Adapter trained on Mimitou. Encoder cross-corpus validated. Architecture spec v1.1 shipped. This is everything to the left of Q3 2026 on the Gantt.
- **Phase 1 (Q3 2026 — KEY GATE)**: QuRIE-seq Phase 1 wet-lab delivery is the gate that unlocks Stage 3b (BTK+JAK demo, Q4 2026) AND Stage 3c (causal architecture validation, Q1-Q2 2027). Phase 1 data carries both downstream stages.
- **Phase 2 (Q1 2027 onwards)**: VDJ + 20-donor scale arrives. Unlocks Stage 4 (VDJ encoder integration + cross-donor generalization). Stage 5 (causal-readiness, 2028) follows.

### Technical glossary
**Quarter notation (Q3'26 = Q3 2026)** — Calendar quarters. Q3 = July-September. Q4 = October-December. Used as roadmap milestones.

**11-quarter window** — Q3 2026 through Q4 2028 inclusive. The deck's strategic horizon.

**Wet-lab swimlane** — Phase 1 (Q3'26) → Phase 2 (Q1'27+) → Phase 3 (B-cell lines + disease, 2027+). Wet-lab data generation chain.

**Model swimlane** — Stage 3a (current, public data) → Stage 3b (BTK+JAK demo Q4'26) → Stage 3c (causal architecture Q1-Q2'27) → Stage 4 (VDJ + 20-donor scale, 2027) → Stage 5 (causal-ready, 2028).

**Drug pipelines swimlane** — Pipeline 1 (target ID → validation) starts Q2'27 post Stage 3b verdict. Pipeline 2 follows after Pipeline 1 target validation (Q2'28).

**Publications/demos swimlane** — Stage 3 verdict + BTK+JAK demo investor publication (2027). Stage 4 + 5 peer-reviewed publications (2028+).

**Stage gating dependencies** — Each model stage gated by wet-lab data: Stage 3b needs Phase 1; Stage 3c needs Phase 1 (phospho); Stage 4 needs Phase 2 (VDJ); Stage 5 needs Stage 4 + drug pipelines. Slippage cascades.

**BTK+JAK headline demo (Q4 2026)** — Stage 3b milestone. Zero-shot prediction of BTK+JAK combination response from training on singles. The platform's first investor-grade publication target.

**Phase 1 / Phase 2 / Phase 3 (wet-lab phases)** — See master glossary. Phase 1 = first QuRIE-seq batch, 5 donors, 4 modalities, Q3 2026. Phase 2 = 20-donor scale + VDJ, 2027. Phase 3 = B-cell lines + disease samples, 2027+.

**Stage 3a / 3b / 3c / 4 / 5** — Model training stages. Distinct from wet-lab Phases. Stage 3a = current public-data work. Stage 3b = BTK+JAK demo (Q4 2026). Stage 3c = causal architecture validation (Q1-Q2 2027, gated on Phase 1 phospho data). Stage 4 = VDJ + 20-donor scale (gated on Phase 2). Stage 5 = causal-readiness + clinical handoff (2028).

**Target ID → validation (drug pipeline)** — Discovery workflow. Identify candidate drug targets from the platform's predictions; validate via wet-lab experiments. Pipeline 1 starts Q2 2027.

**B-cell line CRISPR (Phase 3 wet-lab)** — Internal CRISPR perturbation experiments on B-cell lines (different from Mimitou's primary T cells). Closes the L3 disease-context gap in our public-data strategy.

### Diligence Q&A

[PRESERVE all existing Q&As (post-Step-2 update including Phase 1/2 canonical definition + Kinga slide 8 bridge). Apply line-28, line-47, line-98, line-127, line-130, line-143, line-177, line-206 stale reference fixes per Part 1.]
```

---

### D2 — Seed Allocation

Replace the existing `## Speaker notes` section with:

```markdown
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

[PRESERVE all existing Q&As verbatim. Apply line-28, line-64, line-104, line-144, line-187 stale reference fixes per Part 1 — particularly the Phase 1 phospho budget reframing.]
```

---

### E1 — Five-Year Trajectory

Replace the existing `## Speaker notes` section with:

```markdown
## Speaker notes

### Three-state framing
- **Today through Phase 1 (2026-2027)**: Stage 3a-3c validation completes on Phase 1 data. BTK+JAK demo Q4 2026. Stage 3c causal architecture validation Q1-Q2 2027 (Phase 1 phospho data). VDJ + cross-disease transfer in Stage 4 Phase 2 (Q2-Q4 2027).
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

[PRESERVE all existing Q&As verbatim. Apply line-28 stale reference fix per Part 1.]
```

---

### F1 — Competitive Positioning

Replace the existing `## Speaker notes` section with:

```markdown
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

[PRESERVE all existing Q&As (post-Step-2 update including TAHOE, Immunai, phospho mechanism Q&As). Apply line-180 source-data row and line-222 elevator paragraph fixes per Part 1.]
```

---

## Part 3 — Pptx v5 Rebuild

After all 14 content spec edits are applied:

**Update `_build_appendix_pptx.py`**:
- Change `OUTPUT` path to `aivc_appendix_v5.pptx`
- Update `.gitignore` with `!aivc_appendix_v5.pptx` exception
- v1 + v2 + v3 + v4 stay preserved as historical artifacts

**Run the build script**. Verify 21-slide output with speaker notes populated for all 13 content slides (A1 may still be partial — verify).

### Acceptance for v5

- ✓ 21 slides total (same as v4)
- ✓ All visual content (PNGs) unchanged from v4
- ✓ Speaker notes EXPANDED across all 14 content slides
- ✓ Each slide's notes include: three-state framing + technical glossary + equations (where applicable) + diligence Q&A
- ✓ A1 speaker notes populated (was sparse pre-v5)
- ✓ No stale "Phase 2 phospho" / "Phase 2 phospho on" / "phospho integration Q1 2027" / "phospho deferred to Phase 2" references anywhere in any content spec
- ✓ v1-v4 preserved as historical artifacts
- ✓ v5 at `docs/deck/exports/aivc_appendix_v5.pptx`

### Verification commands

After v5 builds, run these checks:

```bash
# 1. No stale phospho-Phase-2 references anywhere
grep -rn "Phase 2 phospho\|phospho integration\|phospho readout\|phospho deferred to Phase 2\|phospho panel preparation for Phase 2\|Phase 2 (phospho" docs/deck/content/

# Expected output: empty (no matches)

# 2. Phase 1 phospho framing present
grep -rn "Phospho is integral\|phospho.*Phase 1\|Phase 1.*phospho\|integral to QuRIE-seq" docs/deck/content/ | wc -l

# Expected output: significant count (10+) confirming corrected framing landed
```

---

## Deliverable

Single batch commit covering all 14 content spec edits + pptx v5:

```bash
git add docs/deck/content/A1_system_architecture.md \
        docs/deck/content/A2_encoder_substrate.md \
        docs/deck/content/A3_decomposed_readout.md \
        docs/deck/content/A4_temporal_neural_ode.md \
        docs/deck/content/A5_causal_architecture.md \
        docs/deck/content/B1_methodology_rigor.md \
        docs/deck/content/B2_encoder_probe_verdict.md \
        docs/deck/content/B3_synergy_pre_demo.md \
        docs/deck/content/C1_phase1_design.md \
        docs/deck/content/C2_btk_jak_demo.md \
        docs/deck/content/D1_quarterly_roadmap.md \
        docs/deck/content/D2_seed_allocation.md \
        docs/deck/content/E1_five_year_trajectory.md \
        docs/deck/content/F1_competitive_positioning.md \
        docs/deck/exports/aivc_appendix_v5.pptx \
        docs/deck/exports/_build_appendix_pptx.py \
        docs/deck/exports/.gitignore
git commit -m "docs(deck): step 5 unified reconciliation - speaker notes glossary + phospho cleanup across all specs"
git push origin main
```

---

## Flagged Contradictions / Implications For Ash + Claude

These need decisions OR Kinga/Thiago verification:

### 1. D2 Budget Timing Implication
The $1M Phase 1 phospho panel cost is now Phase 1 spend (Q3 2026), not Phase 2 readiness spend (Q1 2027). **Phase 1 budget is heavier earlier than the original spec implied.** D2 reframes the bucket but the dollar amount stays at ~$1M. Worth confirming with Kinga that Phase 1 budget can absorb the phospho panel cost in Q3 2026 vs the originally-implied Q1 2027 timing.

### 2. Stage 3c Timing Acceleration Opportunity
A5 status pill says "Validation Q1-Q2 2027 · post Phase 1 wet-lab data". With phospho in Phase 1 (Q3 2026 delivery), Stage 3c validation could begin Q4 2026 — one quarter earlier than the pill claims. The conservative framing (Q1-Q2 2027) is defensible (account for Phase 1 data processing + first eval) but the acceleration opportunity is real. **Decision needed: keep conservative or shift earlier?**

### 3. Phase 1 Perturbation Panel Size
Multiple specs reference Phase 1 condition counts (15-20 total, 6 inhibitor singles, 3 combos) but these are estimates pending Thiago confirmation. C1's softened framing handles this OK; D2's budget assumes ~17 antibody panel cost based on this estimate. **Worth Thiago confirmation before v5 ships, OR accept "TBC" framing.**

### 4. Phase 2 VDJ + 20-Donor Scale Q1'27 Timing
D1 milestone "Phase 2 VDJ on Q1'27" matches Thiago's earlier confirmation but may slip if Phase 1 delivery slips. **No change needed for v5 — but Phase 4A polish should add a "subject to Phase 1 delivery" caveat.**

### 5. "Sci" Reference Still Pending
Master glossary placeholder remains: `Sci [PENDING IDENTIFICATION]`. Systematic scan of slide text and content specs found no Sci-prefix library. **Pending Kinga clarification at v5 review.** Not blocking ship.

### 6. A5 Color Coding Deferred
A5 equation rendering is white-only mathtext PNG (correct math, no color coding). Phase 4A polish can restore W cyan / dₚ lavender / (I−W)⁻¹ green via 3 separate mathtext renders. **Tech debt; not blocking v5.**

### 7. Pagination Outlier
Slides show `/12`, `/13`, `/14` pagination. Phase 4A polish unifies to `/14`. **Tech debt; not blocking v5.**

### 8. Visual Issues Found On v4
Per Ash's v4 visual review (post-ship): A1 capsule tabs text overflow, A2 left-right zone balance, D1 Stage 3 compactness, F1 Quriegen row vertical padding. **All Phase 4A polish scope; not blocking v5 ship.**

---

## What's Out Of Scope For This Step

- Visual polish on hero slides (Phase 4A — after Kinga + Jan feedback)
- Font sizing audit (Phase 4A)
- A5 color coding restoration (Phase 4A)
- Pagination unification (Phase 4A)
- Sci reference identification (Kinga to confirm at v5 review)
- D2 final budget confirmation (Kinga to confirm at v5 review)

---

## After This Lands

**v5 ships to Kinga + Jan.** They have:
- 21-slide investor-ready appendix
- Comprehensive technical glossary in speaker notes per slide
- Three-state framing throughout (today / Phase 1 / Phase 2)
- Reconciled timeline (phospho in Phase 1, VDJ in Phase 2)
- Reconciled budget (Phase 1 includes phospho panel)
- Architecture spec dependencies clearly stated

They review and provide feedback. Phase 4A visual polish follows their feedback round.
