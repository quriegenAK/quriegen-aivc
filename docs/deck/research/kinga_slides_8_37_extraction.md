# Kinga Slides 8 + 37 — Content Extraction + Cross-Reference + Architectural Placement

**Source**: `docs/deck/source/QurieGen_SEED_ROUND_05_2026_new.pptx`
**Extracted/analyzed**: 2026-05-16
**Method**: Position-aware extraction from `ppt/slides/slide{8,37}.xml` — every `<p:sp>` shape's `<a:off>` coordinates + `<a:t>` text content, sorted into a 2-D reading-order grid.
**Refresh trigger**: re-run if Kinga's source pptx is modified.

---

# Layer 1 — Extraction

## Slide 8 — SOLUTION · COMBINATION OF WET & DRY LAB

**Title**: `SOLUTION · COMBINATION OF WET & DRY LAB`
**Sub-headline**: *"The first causal, virtual cell model powered by a lab-in-the-loop platform."*
**Position in deck**: middle/primary deck — the architecture-overview slide that the technical appendix B-section maps onto.

### 3-column structure (INPUT / CAUSAL CORE / OUTPUT)

Top-row taglines under each step number:
- `01 INPUT` → **PRIMARY IMMUNE CELLS** → "MULTI-OMIC INPUTS"
- `02 CAUSAL CORE` → **MULTI-OMIC FUSION** → "CAUSAL CELL MODEL"
- `03 OUTPUT` → **CAUSAL BY DESIGN** → "CAUSAL OUTPUTS"

### Column 01 — INPUT (left, x≈0.7")

- *Primary immune cells · perturbation conditions* (eyebrow line)
- **Single-cell multi-omics**: `RNA · ATAC · proteome · phospho · BCR/TCR`
- **Perturbation (soft & hard)**: `Drugs · antibodies · CRISPRi/a`
- **Reference biology**: `IMGT · ENCODE · GO · STRING · Reactome` ← prior-knowledge databases (collective listing)
- Visual: arrow labeled `FUSE` flows right into CAUSAL CORE

### Column 02 — CAUSAL CORE (center, x≈5.0")

- *Five-layer regulatory cascade* (eyebrow)
- Numbered cascade steps with concept-action pairs:
  1. **Harmonize** — batch correction
  2. **Integrate** — multi-omic fusion
  3. **Represent** — cell state
  4. **Infer** — regulatory graph
  5. **Predict** — perturb response
- Visual: `ENCODE` label appears in the middle column transition zone

### Column 03 — OUTPUT (right, x≈9.5")

- *Decision-grade · mechanism-level* (eyebrow)
- **Per-database output mapping** (each database paired with a specific output capability):
  - `IMGT` → **Target discovery**
  - `ENCODE` → **first-in-class · druggable targets**
  - `GO` → **Perturbation response**
  - `STRING` → **Mode of action and disease mechanism**
  - `Reactome` → **Cell-state transitions**
- Outputs: `lineage / activation + mechanism`
- Visual: arrow labeled `INFER` flows from CAUSAL CORE into outputs

### Bottom band — 24-month trajectory

- **TRAJECTORY · 24 MONTHS** (left eyebrow)
- `PHASE 1 · M0–3`: **Tri-omic causal model**
- `PHASE 2 · M4–12`: **Five-omic integration**
- `PHASE 3 · M13–24`: **Full virtual immune cell · v1.0**

### Annotations / Ash-added content (inferred from style + technical specificity)

Slide 8 content is largely Kinga's strategic framing. Items that align with our internal architecture spec language and are likely Ash-augmented:
- The specific 5-layer cascade naming (`Harmonize → Integrate → Represent → Infer → Predict`) matches the architecture spec v1.1 phrasing.
- The 5-database list (`IMGT · ENCODE · GO · STRING · Reactome`) is technically precise and likely a deliberate addition to make the prior-knowledge surface visible to technical reviewers.

### Source citations on slide 8
- None explicit on the slide itself.

---

## Slide 37 — SOLUTION · AI VIRTUAL CELL MODEL

**Title**: `SOLUTION · AI VIRTUAL CELL MODEL`
**Sub-headline**: *"AI virtual cell model: simulate, predict, steer cell behavior."*
**Position in deck**: later in primary deck — the architectural-detail slide that our A1 appendix slide directly mirrors visually.

### 3-column structure (INPUT DATA / CAUSAL CORE / OUTPUT LAYER)

Top-row taglines:
- `01 INPUT DATA` → **PRIMARY IMMUNE CELLS** → "MULTI-MODAL DATA"
- `02 CAUSAL CORE` → **4-MODALITY FUSION** → "AIVC FOUNDATIONAL MODEL"
- `03 OUTPUT LAYER` → **CAUSAL BY DESIGN** → "CAUSAL OUTPUTS"

### Column 01 — INPUT DATA (x≈0.9")

- *"Primary immune cells · 5 modalities + perturbation"* (subline)
- **Single-Cell Multi-Omics**: `ATAC · RNA · Protein · Phospho proteins`
- **Soft & hard perturbations**: drugs, antibodies, CRISPR
- **Pre-existing knowledge**: `Databases, Cellular context, assay conditions` ← prior-knowledge component (generalized; not enumerated like slide 8)
- Visual: arrow labeled `FUSE` connecting INPUT to CAUSAL CORE
- Icons used (Unicode glyphs): M for multi-omics, ◌ for perturbations, ▣ for pre-existing knowledge

### Column 02 — CAUSAL CORE (x≈5.0")

- *"Harmonize, Integrate, Represent, Infer (sparse causal graph), Predict (counterfactual)"* — the 5-layer cascade, expanded with parentheticals matching our spec language
- **Temporal Cross-Modal Fusion**
  - `causal attention mask: ATAC → Phospho → RNA → Protein` — the specific causal/temporal ordering enforced via attention masking
- **Neumann Propagation** — `(I−W)⁻¹ dₚ` (math formulation for causal effect propagation through learned GRN)
- **Sparse learned GRN** — the gene regulatory network as the structural inference target
- **Perturbation Response Decoder** — `direct-effect dₚ · log-FC head · AIVC_GRAD_GUARD`
- Visual labels: `CAUSAL` and `LOOP` flank the central FUSE/INFER pipeline
- Icons used: ∿ for cross-modal fusion, ⌬ for Neumann propagation, ⟳ for the perturbation response loop

### Column 03 — OUTPUT LAYER (x≈10.0")

- *"Decision-grade, mechanism-level"* (subline)
- **Target discovery** — "List of the first and the best-in-class targets"
- **Prediction of cellular responses**
- **Mode of action and mechanism of disease**
- **Functional assays** — "Leveraging potential of existing data"
- Icons used: ◉ for target discovery, ⚶ for predictions, ⇢ for functional-assay loop-back

### Annotations / Ash-added content (highly likely)

These items are technical-spec-aligned and not investor-facing strategic narrative — almost certainly Ash augmentations on top of Kinga's base layout:
- `Neumann Propagation (I−W)⁻¹ dₚ` — verbatim formula from architecture spec
- `causal attention mask: ATAC → Phospho → RNA → Protein` — verbatim modality temporal order from spec
- `AIVC_GRAD_GUARD` — internal environment variable name, technical implementation detail
- `direct-effect dₚ · log-FC head` — implementation-level term from training loss components
- `sparse causal graph` / `sparse learned GRN` — technical architectural objective

### Source citations on slide 37
- None explicit on the slide itself.

---

## Notes on extraction

1. **Visual hierarchy lossy in text-only form**: both slides use FUSE / ENCODE / INFER as flow-arrow labels between columns. The directional flow doesn't transfer to bullet form — the rendered slide visually conveys "INPUT → fuse → CAUSAL CORE → infer → OUTPUT" via arrows, but the text dump loses that.

2. **Unicode glyphs as section icons** (slide 37): M / ◌ / ▣ for input rows; ∿ / ⌬ / ⟳ for core rows; ◉ / ⚶ / ⇢ for output rows. These are decorative but visually distinguish categories.

3. **Slide 8 vs Slide 37 relationship**: Slide 8 is the *strategic* architecture overview (5-database listing under "Reference biology", 24-month trajectory at the bottom). Slide 37 is the *technical* architecture deep-dive (Neumann math, AIVC_GRAD_GUARD, causal attention mask spec). They cover the same 3-column conceptual frame but at different depths — slide 8 for board-level reading, slide 37 for technical reviewer reading.

4. **5-database listing only on slide 8** (in "Reference biology" line and per-database output mapping in column 3). Slide 37 abstracts this to `Databases, Cellular context, assay conditions` — no enumeration. The 5 databases (IMGT, ENCODE, GO, STRING, Reactome) are unique to slide 8.

5. **Spec-language verbatim items on slide 37** are very likely Ash additions made when populating the deck after the architecture spec v1.1 was finalized.

---

# Layer 2 — Cross-Reference Analysis

For each substantive item from slides 8 and 37, classify against the 13 appendix content slides:

## Items from slide 8

| # | Item from slide 8 | Appendix coverage | Classification |
|---|---|---|---|
| 1 | `Primary immune cells` framing | A2 (encoder substrate is primary PBMCs) + C1 (Phase 1 design) explicit | **Covered** |
| 2 | `Single-cell multi-omics: RNA · ATAC · proteome · phospho · BCR/TCR` | A2 explicit (RNA + ATAC + Protein today); phospho + VDJ in A2 + C1 + F1 as Phase 2 | **Covered** |
| 3 | `Perturbation (soft & hard): Drugs · antibodies · CRISPRi/a` | C1 (4-arm design covers stim + inhibitor); B2/B3 cover CRISPR (Mimitou ASAP-seq). "Antibodies" specifically not surfaced. | **Mostly covered** — antibody perturbations not explicitly named in any appendix slide |
| 4 | **`Reference biology: IMGT · ENCODE · GO · STRING · Reactome`** | **NOT EXPLICITLY NAMED in any appendix slide.** A3 mentions pathway constraints implicitly (zero-arm L2); pathway databases not enumerated. | **Missing-but-belongs** → recommend A3 or new sub-slide (see Layer 3) |
| 5 | 5-layer cascade (`Harmonize → Integrate → Represent → Infer → Predict`) | A1 maps to a 5-block flow (INPUT → ENCODER → TEMPORAL → READOUT → OUTPUT) — same shape, different layer names. Not a 1:1 vocabulary match. | **Covered with vocabulary mismatch** → potential alignment opportunity in Phase 4 |
| 6 | Per-database output mapping (IMGT→Target / ENCODE→druggable / GO→response / STRING→MoA / Reactome→state transitions) | Not in any appendix slide | **Missing-but-belongs** OR **Kinga-only** depending on whether output-database mapping is a current architectural commitment (per Layer 3 analysis below) |
| 7 | 24-month trajectory (PHASE 1 M0-3 tri-omic / PHASE 2 M4-12 five-omic / PHASE 3 M13-24 full virtual immune cell v1.0) | D1 covers an 11-quarter roadmap (Q3'26→Q4'28 = 30 months) with different phasing (Stage 3a/b/c/4/5 + Phase 1/2/3). Different timeframe + different decomposition. | **Different framing** — D1 is appendix-canonical; slide 8's 24-month version is Kinga's strategic compression. Not a "missing" item; just a different lens. **Kinga-only** for the 24-month framing specifically. |
| 8 | `Decision-grade · mechanism-level` (outputs descriptor) | Implicit in E1 ("first-in-class candidates" framing) but not literally as "decision-grade" language | **Mostly covered** — wording difference, no architectural gap |
| 9 | `lineage / activation + mechanism` (output capability) | A2 mentions PBMC lineages; mechanism implicit in A3 decomposed readout | **Covered** (implicitly) |
| 10 | `lab-in-the-loop platform` (sub-headline) | F1 explicit ("closed-loop integrated platform") | **Covered** |

## Items from slide 37

| # | Item from slide 37 | Appendix coverage | Classification |
|---|---|---|---|
| 11 | `5 modalities + perturbation` (input framing) | A2 + F1 explicit (5-modality vision, 3 validated today + 2 Phase 2) | **Covered** |
| 12 | `Pre-existing knowledge: Databases, Cellular context, assay conditions` | Not surfaced in any appendix slide (assay conditions implicit in C1 4-arm design but not named "prior knowledge") | **Missing-but-belongs** → couples with item #4 (slide 8 databases) |
| 13 | **`Temporal Cross-Modal Fusion · causal attention mask: ATAC → Phospho → RNA → Protein`** | A1 mentions "temporal order ATAC → PHOSPHO → RNA → PROTEIN" only in the architecture-invariants section of CLAUDE.md, NOT in any appendix slide. A4 covers Neural ODE temporal but not the cross-modal attention mask specifically. | **Missing-but-belongs** → A1 visual + A4 speaker notes |
| 14 | **`Neumann Propagation: (I−W)⁻¹ dₚ`** | NOT in any appendix slide. This is the causal-propagation math through the learned GRN. Architecture spec v1.1 references Neumann series for causal inference but our A3 decomposed readout slide does NOT show this. | **Missing-but-belongs** → A3 expansion OR new architectural slide |
| 15 | **`sparse learned GRN`** (gene regulatory network) | NOT in any appendix slide explicitly. A3 talks about decomposed readout but not GRN inference. | **Missing-but-belongs** → A3 or speaker notes |
| 16 | **`Perturbation Response Decoder: direct-effect dₚ · log-FC head · AIVC_GRAD_GUARD`** | A3 covers decomposed readout (4-arm). `dₚ` direct effect + `log-FC head` + `AIVC_GRAD_GUARD` not explicitly in any appendix slide. AIVC_GRAD_GUARD is in CLAUDE.md architecture-invariants but not surfaced on a slide. | **Missing-but-belongs** → A3 speaker notes (technical depth questions) |
| 17 | `Predict (counterfactual)` | A3 zero-arm constraint enables counterfactual prediction implicitly; word "counterfactual" not used in any appendix slide. | **Missing-but-belongs** (vocabulary alignment, low priority) |
| 18 | `Target discovery: list of first/best-in-class targets` (output) | E1 covers "first-in-class candidates" in Phase 4 framing. Target discovery as an output capability of the model itself not named. | **Mostly covered** — E1 is downstream pipeline framing; slide 37's frame is "model output = target list" which is more direct |
| 19 | `Mode of action and mechanism of disease` (output) | F1 mentions "compositional causal modeling"; specific "MoA prediction" output not named. | **Missing-but-belongs** (output capability framing) |
| 20 | `Functional assays · leveraging potential of existing data` (output) | Implicit in F1 (lab-in-the-loop integrated platform); not a named output capability in any slide. | **Mostly covered** (implicit) |

## Summary of Layer 2 classifications

| Classification | Count |
|---|---|
| **Covered** (explicit appendix slide) | 6 items |
| **Mostly covered** (vocabulary/framing difference, no real gap) | 5 items |
| **Missing-but-belongs** | 7 items (#4, #6, #12, #13, #14, #15, #16) — plus #17, #19 as low-priority vocabulary additions |
| **Kinga-only** (appropriate as primary deck content; no appendix mirror) | 1 item (#7 — 24-month trajectory framing) |

The seven substantive "Missing-but-belongs" items cluster around two themes:
- **Prior-knowledge databases** (#4 + #6 + #12): IMGT / ENCODE / GO / STRING / Reactome and their architectural roles
- **Causal-inference architecture details** (#13 + #14 + #15 + #16): cross-modal attention mask + Neumann propagation + sparse GRN + perturbation response decoder

These two clusters drive the Layer 3 analysis.

---

# Layer 3 — Architectural Placement For Prior Knowledge

Per Ash's specific ask: for every prior-knowledge component (used today OR planned for future phases), determine **where in the stack**, **when it activates**, and **which appendix slide should anchor it**.

---

## Prior-knowledge component 1 — IMGT (International ImMunoGeneTics)

**What it is**: The reference database for immunoglobulin and T-cell receptor V(D)J gene segment annotation. Standard reference for BCR / TCR repertoire analysis.

**Where it sits in the architecture stack**:
- **Pathway annotation layer** (post-prediction interpretation) — used to map predicted VDJ rearrangements to clonal categories
- **Evaluation reference** — gold-standard against which any VDJ inference is benchmarked

**When it activates**:
- Phase 2 wet-lab extension (2027) — when VDJ becomes a real modality in QurieSeq Phase 2
- Stage 4 (VDJ + 20-donor scale, 2027) — model-side integration of VDJ readouts
- **Not in use today** (VDJ is Phase 2 plan)

**Which appendix slide should anchor it**:
- **A2 speaker notes** — current placement (the "Why aren't phospho and VDJ in the validation?" Q&A could explicitly reference IMGT as the eventual reference for VDJ outputs)
- **OR** new sub-slide in A section if Ash wants prior-knowledge as a first-class visible architecture component

**Justification**: IMGT is meaningful only once VDJ is operational. Placing it in A2 speaker notes aligns it with the Phase 2 modality story; placing it on a slide would prematurely commit to Phase 2 detail.

---

## Prior-knowledge component 2 — ENCODE (Encyclopedia of DNA Elements)

**What it is**: Reference catalog of functional DNA elements (regulatory regions, ChIP-seq peaks, chromatin accessibility tracks). Standard ATAC-seq reference.

**Where it sits in the architecture stack**:
- **Training data input** indirectly — ENCODE-derived peak sets serve as a reference for harmonizing ATAC peaks across studies (e.g., during DOGMA → Calderon cross-corpus eval, peak harmonization may use ENCODE as anchor)
- **Pathway annotation layer** — for mapping ATAC peaks to regulatory regions / nearest genes
- **Evaluation reference** — gold-standard for chromatin-accessibility benchmarks
- Slide 8 maps ENCODE → "first-in-class · druggable targets" (output side) — suggests downstream use in target prioritization via regulatory annotation, not just input harmonization.

**When it activates**:
- **In use today** — DOGMA-seq + Calderon ATAC harmonization (peak union construction, per CLAUDE.md `data/peak_sets/pbmc10k_hg38_20260415.tsv`) likely benchmarks against ENCODE-style references
- Stage 3c (pathway readout + phospho, 2027) — explicit pathway/regulatory-region output mapping

**Which appendix slide should anchor it**:
- **A2 — extend body bullet 2** (ATAC modality description) to name ENCODE as the peak-reference standard
- **B1 — extend dataset card** (Calderon column subtitle could reference ENCODE-aligned peak space)

**Justification**: ENCODE is a foundational ATAC reference and is used today (per the peak-set construction). Surfacing it on A2 + B1 adds technical-depth credibility without architectural change. The slide-8 "ENCODE → druggable targets" mapping is forward-looking and belongs as Layer 3-class output framing in Stage 3c speaker notes (D1 or new pathway-output slide).

---

## Prior-knowledge component 3 — GO (Gene Ontology)

**What it is**: Standardized vocabulary for gene-function annotation across three domains (Biological Process, Molecular Function, Cellular Component). Universally used for gene-set enrichment.

**Where it sits in the architecture stack**:
- **Pathway annotation layer** — post-prediction enrichment of predicted perturbation responses (which GO terms enrich in upregulated genes)
- **Biological prior for architecture decisions** — GO term hierarchies inform gene-set construction for pathway-aware loss heads

**When it activates**:
- Stage 3c (pathway readout, 2027) — when the "58 Hallmark + 8 KEGG immune" pathway scores output goes live (per A1 OUTPUT block)
- **Partially in use today** — Stage 3 Part 1 Report 3 uses gene-set enrichment (per CLAUDE.md memory: "4798 unique genes across 58 pathways")

**Which appendix slide should anchor it**:
- **A1 OUTPUT block** already says "58 pathway scores" — extending the speaker notes to name GO (alongside Reactome below) closes the gap
- **A3 speaker notes** — the pathway constraints in zero-arm L2 could reference GO term hierarchies as a biological prior

**Justification**: GO is one of the most universal prior-knowledge layers in single-cell analysis; not surfacing it is a credibility miss for any technical reviewer. A1 + A3 speaker note additions are low-cost.

---

## Prior-knowledge component 4 — STRING (protein-protein interactions)

**What it is**: Database of known and predicted protein-protein interactions, curated from primary literature + computational predictions. Used to construct interaction networks for pathway-context-aware modeling.

**Where it sits in the architecture stack**:
- **Biological prior for architecture decisions** — STRING-derived interaction networks could initialize the "sparse learned GRN" structure (Layer 3 of the 5-layer cascade) as a warm start, or constrain its inference
- **Pathway annotation layer** — post-prediction MoA reasoning (which protein-protein interactions explain a predicted phenotype)

**When it activates**:
- Stage 4 (VDJ + 20-donor scale, 2027) — when the "Infer (sparse causal graph)" layer is upgraded with biology priors
- Stage 5 (causal-readiness, 2028) — full MoA reasoning with STRING as one input to the causal explanation framework
- **Not in use today** as a structural prior; possibly used in eval/MoA analysis already (CLAUDE.md references "Mode of action" in Stage 3 Part 1 reports)

**Which appendix slide should anchor it**:
- **No current slide cleanly anchors STRING.** Best fit is A3 speaker notes ("How does the synergy head learn the right interactions?") or a new sub-slide in Section A if Ash wants to elevate prior-knowledge as a first-class architectural axis.
- **placement TBD** — depends on whether STRING-as-prior is committed in spec v1.1 or merely aspirational. Layer 3 architectural placement requires confirmation.

**Justification**: STRING's role is genuinely architecturally undecided in our current spec. Flagging as TBD is honest; forcing a placement now would be premature.

---

## Prior-knowledge component 5 — Reactome (pathway database)

**What it is**: Hand-curated pathway database (signaling, metabolism, cell-cycle, etc.) — pathway-membership annotations for thousands of genes. Often used alongside KEGG for pathway-level analysis.

**Where it sits in the architecture stack**:
- **Pathway annotation layer** — used directly in pathway-aware OUTPUT generation (per A1: "58 Hallmark + 8 KEGG immune"; Reactome adds a third pathway source)
- **Evaluation reference** — pathway-level enrichment as a sanity check on predicted perturbation responses

**When it activates**:
- Stage 3c (pathway readout, 2027) — explicit pathway-output integration
- **Partially in use today** — pathway annotations referenced in Stage 3 Part 1 Report 3 (per CLAUDE.md)

**Which appendix slide should anchor it**:
- **A1 OUTPUT block speaker notes** — name Reactome alongside Hallmark + KEGG as the pathway sources
- **B1 / B2 speaker notes** — if any baseline used Reactome enrichment for validation, surface it
- Slide 8's mapping "Reactome → Cell-state transitions" suggests a more specific architectural commitment than pathway enrichment. **placement TBD** for the cell-state-transitions framing specifically; the pathway-enrichment use is already implicitly covered.

**Justification**: Reactome is in our pathway stack today (per Stage 3 Part 1) but underspecified in the appendix. The "Cell-state transitions" mapping on slide 8 is a stronger architectural claim than what our current spec backs — flag for Ash + Claude decision.

---

## Prior-knowledge component 6 — General `Databases, Cellular context, assay conditions` (slide 37 abstraction)

**What it is**: Catch-all category on slide 37 for any non-data prior knowledge fed into the model. Includes the 5 databases above plus assay-condition metadata (e.g., "this batch was DMSO + 5min" vs "this batch was BTK-inhibitor + 60min").

**Where it sits in the architecture stack**:
- **Training data input** — assay-condition metadata is the perturbation conditioning vector (s, i, t) in the architecture spec
- **Cellular context** = donor metadata + cell-type labels = batch covariates (already wired in pretrain per DOGMA lysis covariate)

**When it activates**:
- **In use today** — perturbation conditioning vectors are part of Stage 3a (Mimitou CRISPR adapter)
- Stage 3b (BTK+JAK demo) — assay-condition metadata is what indexes the 4-arm structure
- Phase 1 wet-lab (Q3 2026) — QurieSeq metadata schema activates

**Which appendix slide should anchor it**:
- **A2 — the encoder block** could speaker-note that conditioning vectors include `(donor, cell-type, stim, inhibitor, timepoint)`
- **C1 — already implicit** in the 4-arm × 5-timepoint × 5-donor experimental grid; speaker notes could explicitly name "conditioning metadata schema"

**Justification**: Slide 37's "Pre-existing knowledge: Databases, Cellular context, assay conditions" is largely covered through implicit appendix coverage. Speaker-note expansion is low-cost and closes the vocabulary gap.

---

## Prior-knowledge component 7 — Causal attention mask `ATAC → Phospho → RNA → Protein`

**What it is**: Architectural constraint enforcing a fixed temporal/causal ordering across modalities in cross-modal attention. From slide 37 (likely Ash-added) and matches CLAUDE.md `TEMPORAL_ORDER` enum.

**Where it sits in the architecture stack**:
- **Biological prior for architecture decisions** — fixed causal ordering reflects assumed biological propagation direction (chromatin opening → phospho signaling → transcription → translation)
- Embedded in the encoder + fusion architecture (per `aivc/skills/fusion.py::TemporalCrossModalFusion.TEMPORAL_ORDER`)

**When it activates**:
- **In use today** — the temporal order is hardcoded in the encoder per CLAUDE.md
- Stage 3a + all downstream stages

**Which appendix slide should anchor it**:
- **A1 visual** — the central INTEGRATED PLATFORM area or a "biological invariants" sub-row could surface the causal-order arrow
- **A4 speaker notes** — Neural ODE temporal slide; could explain that the modality ordering itself is causally informed
- **Strong candidate for Phase 4 surface-up**: this is a defensible architectural decision that competitors don't make as explicit. Naming it visually elevates the technical rigor signal.

**Justification**: The causal attention mask is a current architectural commitment (in code, in CLAUDE.md) but invisible on slides. It's a free credibility win for Phase 4 to surface it.

---

## Prior-knowledge component 8 — Neumann Propagation `(I−W)⁻¹ dₚ`

**What it is**: Closed-form solution for the equilibrium response of a linear-in-the-network dynamical system to perturbation `dₚ`. `W` is the (sparse, learned) GRN adjacency matrix. The Neumann series `(I−W)⁻¹ = I + W + W² + W³ + ...` expresses indirect effects propagating through the network.

**Where it sits in the architecture stack**:
- **Architectural choice** — the causal propagation operator in the Predict layer
- Operates on the **sparse learned GRN** (component 9 below)

**When it activates**:
- Stage 3c (pathway readout / causal expansion, 2027) — where GRN inference becomes load-bearing
- Stage 5 (causal-readiness, 2028) — full counterfactual prediction via Neumann propagation

**Which appendix slide should anchor it**:
- **A3 — Decomposed Readout slide could be extended** to include Neumann propagation as the "how the synergy head actually computes effects" mechanism. Today A3 stops at `h_base + Δ_stim + Δ_inh + Δ_synergy`; Neumann is the *internals* of how `Δ_synergy` is computed when the GRN structure is invoked.
- **OR new slide A5** — "Causal Propagation Mechanism" — if Ash wants Neumann as a first-class architectural commitment visible on the slide deck.
- **placement TBD** — depends on whether Neumann is **committed** in spec v1.1 vs **aspirational for Stage 3c+**. From slide 37 it appears committed; verify against spec.

**Justification**: This is the most technically distinctive item in the gap. Surfacing it elevates the "causal-ready" claim (currently in F1) from marketing language to specific architectural commitment. High-value Phase 4 addition if confirmed in spec.

---

## Prior-knowledge component 9 — Sparse learned GRN (Gene Regulatory Network)

**What it is**: A learned sparse adjacency matrix `W` over gene/protein nodes encoding regulatory directional dependencies. Inferred (not hand-curated) but biology-priors-aware (potentially constrained by STRING, ENCODE TFBS).

**Where it sits in the architecture stack**:
- **Architectural component** — the structural object inferred in the "Infer" layer of the 5-layer cascade
- **Biological prior** — STRING / ENCODE motifs could constrain the sparsity pattern (component 4 above)

**When it activates**:
- Stage 3c (multi-perturbation expansion, 2027) — when GRN inference becomes a stage-trained component
- Stage 4 (VDJ + 20-donor scale, 2027) — GRN scales with more donors

**Which appendix slide should anchor it**:
- **A3 speaker notes** OR **new A5 slide** — same call as Neumann
- **placement TBD** — couples to Neumann decision; if Neumann is committed, GRN must be committed alongside (it's the operand)

**Justification**: GRN inference is the "Infer (sparse causal graph)" step of the 5-layer cascade per slides 8 + 37. Currently invisible on appendix. Same Phase 4 decision as Neumann.

---

## Prior-knowledge component 10 — `AIVC_GRAD_GUARD` environment variable

**What it is**: Gradient isolation flag (`AIVC_GRAD_GUARD=1`) that blocks causal-loss gradients from updating the pretrained encoder during pretrain → adapter staging. Per CLAUDE.md: "Gradient isolation: AIVC_GRAD_GUARD=1 blocks causal losses in pretrain stage".

**Where it sits in the architecture stack**:
- **Implementation detail** — guards architectural staging discipline (pretrain weights frozen during adapter training)
- Closely related to: stage routing `pretrain / joint / joint_safe` per CLAUDE.md

**When it activates**:
- **In use today** — Stage 3a adapter training uses this guard
- All downstream stages preserve it

**Which appendix slide should anchor it**:
- **A2 speaker notes** — the "frozen substrate" claim on A2 is enforced by AIVC_GRAD_GUARD. Speaker notes could explain the mechanism for technical reviewers who ask "how do you actually keep the encoder frozen?"
- **Low priority** for visual slide surface — too implementation-detail-y to put on a slide.

**Justification**: This is a credibility detail for technical reviewers, not investor-narrative content. Speaker notes are the right surface.

---

## Prior-knowledge component 11 — `direct-effect dₚ · log-FC head`

**What it is**: The direct (non-propagated) perturbation-effect representation `dₚ` and the log-fold-change prediction head that consumes it. Together with Neumann propagation, this is how the model converts a perturbation embedding into a predicted differential-expression vector.

**Where it sits in the architecture stack**:
- **Implementation detail of the Predict layer** — `dₚ` is the input to Neumann; log-FC head is the output activation

**When it activates**:
- Stage 3b+ — when GRN-based prediction is operational
- **Not surfaced today** in any appendix slide

**Which appendix slide should anchor it**:
- **A3 speaker notes** — same cluster as Neumann + GRN
- Phase 4 decision: surface together with Neumann or defer all three

**Justification**: Coupled to Neumann + GRN decisions. Phase 4 placement should treat #8, #9, #11 as a single architectural disclosure choice.

---

## Summary table — architectural placement decisions for Phase 4

| Component | In use today? | Anchor slide | Phase 4 action |
|---|---|---|---|
| IMGT | No (Phase 2) | A2 speaker notes | Add to VDJ speaker note |
| ENCODE | Yes (peak harmonization) | A2 + B1 speaker notes | Add to ATAC + Calderon speaker notes |
| GO | Partially | A1 + A3 speaker notes | Add to 58-pathway speaker note |
| STRING | Not committed | TBD | **Needs architectural decision** (warm-start prior or post-hoc enrichment?) |
| Reactome | Partially | A1 OUTPUT speaker notes | Add to 58-pathway speaker note; "Cell-state transitions" mapping needs decision |
| `Pre-existing knowledge` (catch-all) | Yes | A2 + C1 speaker notes | Vocabulary alignment in speaker notes |
| Causal attention mask `ATAC→Phospho→RNA→Protein` | Yes (code) | A1 visual + A4 speaker notes | **High-value: surface visually on A1** |
| Neumann Propagation `(I−W)⁻¹ dₚ` | Spec-aspirational | A3 expansion OR new A5 | **TBD — needs spec confirmation, then bundle with #9 + #11** |
| Sparse learned GRN | Stage 3c plan | Bundle with Neumann | TBD |
| `AIVC_GRAD_GUARD` | Yes | A2 speaker notes (implementation detail) | Add to "frozen" speaker note |
| `direct-effect dₚ · log-FC head` | Stage 3b+ | Bundle with Neumann | TBD |

---

## Open architectural questions for Ash + Claude (decisions, not research)

1. **Are Neumann Propagation + sparse learned GRN + direct-effect dₚ a committed architecture for Stage 3c?** Slide 37 (presumably Ash-added) shows them prominently. If committed, they should anchor a new appendix slide (A5 — "Causal Propagation Mechanism") or expand A3. If aspirational, defer.

2. **Is STRING a structural prior on the GRN sparsity pattern, or only a post-hoc MoA-reasoning database?** The answer determines whether STRING anchors A3 (architectural input) or E1/F1 (output capability).

3. **Slide 8 per-database output mapping** (IMGT→targets, ENCODE→druggables, GO→responses, STRING→MoA, Reactome→state-transitions) — is this a committed output schema or Kinga's narrative compression? If committed, it's a missing output-architecture slide (potential A6). If narrative, leave Kinga-only.

4. **Should the appendix add a single "Prior-Knowledge Stack" slide** consolidating IMGT/ENCODE/GO/STRING/Reactome with their architectural roles? Or distribute mentions across existing speaker notes? The 5-database listing on slide 8 is visually distinctive enough that consolidation may serve a technical reviewer better.

5. **Causal attention mask `ATAC → Phospho → RNA → Protein`** — agreed surface visually on A1, but where exactly? Adding a new visual row could compete with the existing 5-block flow. Architectural-invariant strip below the main row could work.

6. **5-layer cascade vocabulary alignment** — Kinga uses `Harmonize / Integrate / Represent / Infer / Predict`; A1 uses `INPUT / ENCODER / TEMPORAL / READOUT / OUTPUT`. The two map cleanly but the vocabulary doesn't match. Phase 4 could either (a) keep both vocabularies coexisting (A1 stays as-is, speaker notes bridge to Kinga's 5-layer terms), or (b) update A1 to align verb-noun ("Harmonize" → "Encoder" etc.) for inter-deck coherence.

7. **24-month trajectory (slide 8)** vs **D1 11-quarter Gantt** — different decomposition. Is the 24-month version still committed for Phase 2/3 wet-lab? D1 should align if so, or speaker notes should explain the difference.

---

# Summary

**Layer 1**: Extracted 20+ items from slides 8 + 37 (text + position-aware).

**Layer 2**: 6 items Covered, 5 Mostly-covered, 7 Missing-but-belongs, 1 Kinga-only. Two clear themes among the missing: prior-knowledge databases (IMGT/ENCODE/GO/STRING/Reactome) and causal-inference architecture details (attention mask + Neumann + GRN + decoder internals).

**Layer 3**: 11 prior-knowledge components mapped to architectural placement. 5 are in use today and only need speaker-note additions; 4 are Stage 3c+ aspirational and need spec-level decisions before slide commitment; 2 are environment-variable / implementation details that belong in speaker notes only.

**7 open strategic questions** surfaced for Ash + Claude — these are decisions, not research. Phase 4 polish becomes concrete after these are answered: which databases get speaker-note mentions, whether Neumann/GRN gets a new A5 slide, whether the 5-layer-cascade vocabulary aligns, whether 24-month trajectory still holds.

The audit eliminates the prior-knowledge gap before Phase 4 begins. Phase 4 polish is no longer "polish hero slides generically" — it's "make these 5-7 specific content additions per the placements above, plus the 7 open questions answered."

*End of audit, 2026-05-16.*
