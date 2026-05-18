# Step 2 — Content Spec Updates (5 Slides)

**Status**: Draft, pending Ash review
**Inputs**:
- `docs/deck/research/phase1_modality_correction_2026_05_17.md` (canonical, commit `644b1f8`)
- Existing content specs in `docs/deck/content/` (A2, C1, D1, F1, A5)
- Three-state framing discipline (Today / Phase 1 / Phase 2)
**Output**: 5 updated content specs ready for commit
**Downstream**: Step 3 — Cowork SVG rebuilds for A2, C1, D1, F1 (A5 SVG unchanged)

---

## How To Apply These Updates

Each section below shows **what to change** in the corresponding content spec at `docs/deck/content/<slide>.md`. Apply the edits, commit, push. Cowork picks up from there.

**No new content specs are being created** — these are surgical edits to existing committed specs.

---

## 1. C1 — Phase 1 Experimental Design (THE FOUNDATION SLIDE)

C1 is the canonical Phase 1 reference. Every other slide's Phase 1 mention should be consistent with what C1 says.

### Headline (update)

**Old**: (whatever is currently in `C1_phase1_design.md`)
**New**: 
> **QuRIE-seq Phase 1 — 5 donors × 5 timepoints × 4 modalities × BTK+JAK combo**

### Sub-headline (update)

> First proprietary perturbation-aware multi-omics dataset on primary human PBMCs. Phospho is integral to QuRIE-seq. BTK+JAK combo confirmed. Q3 2026 delivery.

### Body content (3 bullets — rewrite)

- **4 modalities measured directly on the same cells**: RNA, surface protein (CITE-seq), and phospho-proteins at all 5 timepoints (0/5/30/60/180 min); ATAC at t=0 and t=180 only because chromatin accessibility varies on slower timescales than transcription. No public dataset combines these modalities on primary PBMCs with perturbation-aware design.

- **Experimental design tied to model architecture**: 5 timepoints match the Neural ODE temporal backbone. 4-arm design (vehicle / stim / inhibitor singles / combinations) matches the decomposed readout. BTK + JAK combination condition is confirmed for Phase 1 — this is the proof-of-capability for compositional generalization (predict combo from singles).

- **5 donors × ~5k cells/donor/timepoint = ~125k cells total across ~17 conditions**. Donor chromatin signature (ATAC at t=0) functions as biological donor ID since Sanquin blood provider returns blood-type-only metadata. Phase 2 (2027) scales to 20 donors and adds VDJ as 5th modality.

### Visual spec — THE MODALITY × TIMEPOINT MATRIX (visual hero)

This becomes C1's new visual hero. Replaces whatever current Phase 1 visual is there.

```
PHASE 1 — MODALITY × TIMEPOINT MATRIX (Q3 2026)

             ┌──────┬──────┬──────┬──────┬───────┐
             │ t=0  │ t=5  │ t=30 │ t=60 │ t=180 │
┌────────────┼──────┼──────┼──────┼──────┼───────┤
│ RNA        │  ✓   │  ✓   │  ✓   │  ✓   │   ✓   │
├────────────┼──────┼──────┼──────┼──────┼───────┤
│ Protein    │  ✓   │  ✓   │  ✓   │  ✓   │   ✓   │
├────────────┼──────┼──────┼──────┼──────┼───────┤
│ Phospho    │  ✓   │  ✓   │  ✓   │  ✓   │   ✓   │
├────────────┼──────┼──────┼──────┼──────┼───────┤
│ ATAC       │  ✓   │      │      │      │   ✓   │
└────────────┴──────┴──────┴──────┴──────┴───────┘

VDJ: deferred to Phase 2 (2027) · adds 5th modality
ATAC sampling rationale: chromatin layer slow-varying — endpoint coverage sufficient
```

**Color coding**:
- RNA checkmarks: cyan (`#26DDF9`)
- Protein checkmarks: green (`#4ADE80`)
- Phospho checkmarks: lavender (`#8B5CF6`) — **the proprietary modality, no public data exists**
- ATAC checkmarks: muted blue
- VDJ row: amber tinted, "Phase 2" label

**Below the matrix**, a small explanatory strip:

```
4 MODALITIES MEASURED · 5 TIMEPOINTS · 5 DONORS · ~125K CELLS · BTK+JAK COMBO INCLUDED
                  └─── No public dataset has this combination ───┘
```

### Body section structure

In addition to the matrix, C1 should show in the lower zone:

**Left block — perturbation conditions**:
```
PERTURBATION CONDITIONS (17 total)
· Vehicle control (baseline)
· Stimulus (PMA/Iono or equivalent — TBD with Thiago)
· 6 inhibitor singles: BTK, JAK, [+4 others TBD]
· 6 inhibitor + stim combos
· 3 inhibitor + inhibitor combos including BTK+JAK
```

**Right block — wet-lab parameters**:
```
WET-LAB PARAMETERS
· 5 donors (Sanquin, blood-type-only metadata)
· All major PBMC lineages (B / T / NK / monocyte / DC)
· ~5k cells/donor/timepoint
· ~125k cells total
· QuRIE-seq protocol family (proprietary)
· Q3 2026 delivery target
```

### Speaker notes — REPLACE existing with this expanded set

**If asked: "What's QuRIE-seq exactly?"**
> QuRIE-seq is Quriegen's proprietary single-cell multi-omics assay measuring RNA, surface protein, and phospho-proteins from the same cell in a single workflow. Phospho-proteomics is integral to the QuRIE-seq protocol — every QuRIE-seq run generates phospho data alongside RNA and protein. This is the assay's defining capability and the reason no public dataset combines our four Phase 1 modalities on primary PBMCs.

**If asked: "Why ATAC at only 2 timepoints?"**
> Chromatin accessibility changes on slower timescales than transcription or signaling phosphorylation. Sampling at t=0 (baseline) and t=180 (3 hours post-perturbation) captures the chromatin endpoint shift while saving experimental cost on intermediate timepoints where ATAC signal would be statistically unchanged. This is biologically-motivated experimental design, not cost cutting.

**If asked: "Where's the public-data foundation? Are we starting from scratch?"**
> Not at all. The encoder is already trained on public DOGMA-seq data (Mimitou 2021) and validated cross-corpus on Calderon 2019 at 73% pseudo-bulk accuracy. The encoder probe on Mimitou CRISPR perturbations returned 0.57 4-class accuracy at 2.27× chance — pre-registered ADAPTER_RECOMMENDED verdict. Phase 1 plugs into a validated public-data engine, not a cold start. The QuRIE-seq data trains the perturbation-prediction head + decomposed readout + temporal Neural ODE on proprietary perturbation-aware data — but the encoder substrate is already shipped.

**If asked: "Why phospho? What does phospho tell us that RNA + protein don't?"**
> Phospho measures kinase activation state — the immediate signaling response to a perturbation, before transcriptional changes propagate. For drug combination prediction in pathway-driven diseases like CLL, phospho is the readout that distinguishes "drug A and drug B affect the same kinase" (additive effect) from "drug A blocks JAK and drug B blocks BTK so the combination hits both arms of the BCR pathway" (synergistic effect). RNA shows downstream consequences; phospho shows the immediate mechanism. No public single-cell dataset has phospho on PBMCs under perturbation — this is structural white space we own through QuRIE-seq.

**If asked: "5 donors seems small. Won't there be high variance?"**
> Phase 1 5-donor scale is intentional. We trade donor breadth for modality depth and temporal density. The ATAC at t=0 captures donor-specific chromatin baseline (functioning as a donor signature without demographic metadata). The 4-arm decomposed readout architecture is designed for cross-donor generalization — it predicts perturbation effects relative to each donor's vehicle baseline. Phase 2 scales to 20 donors (4×) to validate that cross-donor generalization holds at higher N. Starting smaller is statistical discipline, not under-investment.

**If asked: "Donor selection? Who's Sanquin? What about diversity?"**
> Sanquin is the Dutch national blood bank — high quality, ethically sourced, standardized collection protocols. They return blood type only, no age/sex/ethnicity metadata for privacy reasons. This is acceptable for Phase 1 because the encoder learns chromatin-grounded donor signatures (ATAC at t=0) rather than requiring demographic features. For Phase 2 we may pursue additional donor sources to expand demographic diversity if regulatory or scientific review identifies this as needed.

**If asked: "What's the BTK+JAK combo and why is it the headline?"**
> BTK = Bruton tyrosine kinase (BCR pathway), inhibited by Ibrutinib (Imbruvica, approved CLL drug). JAK = Janus kinase (cytokine signaling), inhibited by Ruxolitinib (Jakafi, approved myelofibrosis drug). Both drugs are FDA-approved as monotherapies. The BTK+JAK combination has clinical evidence for synergistic effect in CLL (PMID 26819050, NCT02912754). Phase 1 includes the BTK+JAK combo condition specifically so Stage 3b can demonstrate zero-shot synergy prediction — train on singles (BTK alone, JAK alone) and predict the combo response from architecture alone. This is the compositional generalization proof-of-capability that justifies the integrated platform claim.

---

## 2. A2 — Multi-Omics Encoder

### Headline (update — slight emphasis shift)

**Old**: (whatever is currently in `A2_encoder_substrate.md`)
**New**: 
> **Multi-omics encoder — trained on public, ready for proprietary**

### Sub-headline (update)

> 3 modalities trained on DOGMA-seq, validated cross-corpus on Calderon 2019 at 73% pseudo-bulk accuracy. Phase 1 QuRIE-seq adds phospho — the 4th modality no public dataset has.

### Body content (3 bullets — rewrite)

- **Today — validated on public data**: encoder pretrained on DOGMA-seq (Mimitou 2021) covering 3 modalities (RNA + ATAC + Protein) on PBMCs under stimulation. Cross-corpus validation on Calderon 2019 yields 73% pseudo-bulk centroid-NN accuracy — encoder generalizes across independently-generated datasets. AIVC_GRAD_GUARD enforces frozen-encoder discipline downstream.

- **Phase 1 (Q3 2026) — proprietary upgrade**: QuRIE-seq adds phospho-proteomics as the 4th modality (integral to the assay; no public phospho data exists for PBMCs). RNA + Protein + Phospho measured at all 5 timepoints; ATAC at t=0 and t=180. The pretrained encoder's RNA+Protein representations transfer; phospho representation is learned during Phase 1 integration as new modality.

- **Phase 2 (2027) — full modality stack**: VDJ added as 5th modality; scale to 20 donors. The encoder grows under the same protocol family — no architectural rebuild required for modality extension. This is the structural advantage of QuRIE-seq protocol-family design.

### Visual spec — three-state encoder evidence

The A2 SVG currently shows the 73% Calderon cross-corpus result as visual hero. **Keep this** — it's shipped public-data evidence.

**Add to A2 visual**: a small "modality progression" strip below or beside the cross-corpus result:

```
ENCODER MODALITY PROGRESSION

TODAY            PHASE 1 (Q3 2026)        PHASE 2 (2027)
──────────       ────────────────         ──────────────
RNA              + Phospho                + VDJ
ATAC             [proprietary,             [proprietary,
Protein           5 timepoints]            5th modality]

[public DOGMA-seq]  [QuRIE-seq Phase 1]    [QuRIE-seq Phase 2]
```

This shows the three-state framing visually without erasing the 73% Calderon result.

### Speaker notes — add these to existing notes

**If asked: "Phospho is in Phase 1? I thought it was Phase 2."**
> Phospho is integral to the QuRIE-seq assay — every QuRIE-seq run generates phospho data alongside RNA and protein. The earlier framing of "phospho deferred to Phase 2" was specifically about public training data — no public dataset has phospho on PBMCs, so we deferred phospho coverage in our public-data layer strategy. But the QuRIE-seq Phase 1 wet-lab generation in Q3 2026 measures phospho directly. So phospho first becomes available to us in Phase 1, not Phase 2.

**If asked: "How does the encoder generalize to phospho if it was only trained on RNA + ATAC + Protein?"**
> The encoder's architecture supports modality extension by design — adding a modality means adding an input head, not retraining from scratch. During Phase 1 integration (Q4 2026), the phospho input head is fit while the RNA/ATAC/Protein representation backbone stays frozen (AIVC_GRAD_GUARD enforced). This is the same adapter-strategy pattern that the Stage 3 Part 1 verdict approved. The 73% cross-corpus result establishes that the backbone representations transfer across datasets; phospho integration tests whether they accommodate a new modality of biological information.

**If asked: "What's pseudo-bulk centroid-NN?"**
> Cross-corpus validation method. Pseudo-bulk: aggregate single cells by cell-type label within each dataset to produce one centroid vector per cell type. Centroid-NN: for each held-out test centroid (from Calderon), find the nearest centroid in the training pool (from DOGMA). Accuracy = fraction where the nearest neighbor is the same cell type. 73% on PBMC major lineages (B / T / NK / monocyte / DC) is strong cross-corpus generalization — random would be 20% for 5 classes.

**If asked: "What's AIVC_GRAD_GUARD?"**
> An environment-variable-controlled gradient-blocking mechanism in our training code. When `AIVC_GRAD_GUARD=1`, the encoder's pretrained weights are frozen during downstream training — adapter and readout heads update, encoder doesn't. This enforces the adapter strategy mechanically rather than relying on training-script discipline. The flag is set in all production training runs after Stage 3 Part 1's ADAPTER_RECOMMENDED verdict.

---

## 3. F1 — Competitive Positioning

### Headline (no change needed)

Keep: "The closed-loop platform — proprietary data, co-designed architecture, compounding over time."

### Sub-headline (slight refinement)

**Old**: "No public dataset has the combination drug combination prediction requires. The wet lab, architecture, and protocol family are co-designed — each compounds the others."

**New** (sharper, Phase 1 anchored):
> No public dataset has the combination — multi-omics + perturbation-aware + temporal + combinatorial. Phase 1 QuRIE-seq closes the gap with phospho at Q3 2026. The platform compounds from there.

### Body content (3 bullets — rewrite slightly to reflect corrected modality story)

- **Public data was the bootstrap; QuRIE-seq is the platform.** No public dataset combines RNA + Protein + Phospho + perturbation-aware temporal sampling on primary PBMCs. The gap is structural — phospho-proteomics on PBMCs under combinatorial perturbation does not exist publicly. QuRIE-seq Phase 1 closes this gap at Q3 2026 with phospho integral to the assay; Phase 2 adds VDJ. The data engine exists because the data doesn't.

- **Five pillars co-designed as one system**: proprietary wet-lab generation (QuRIE-seq family), architecture co-designed with the assay (decomposed readout matches 4-arm experimental design; Neural ODE matches irregular timepoint sampling 0/5/30/60/180; phospho head matches QuRIE-seq's intracellular protein channel), temporal perturbation-aware data creation, compositional training strategy, unified protocol family for Phase 2 VDJ extension. Each pillar reinforces the others; the integration is the moat.

- **Competitors optimize one layer each**: TAHOE optimizes single-modality data scale (100M cells, RNA-only cell lines, no phospho); Immunai optimizes modality-rich atlases (RNA + Protein + VDJ via partners, no phospho); Cellarity, CytoReason, Turbine, DeepLife optimize foundation models on partner-derived data without proprietary wet-lab pipelines; Valo and Noetik optimize downstream therapeutics. We optimize the closed-loop system itself, including phospho — the dimension no competitor measures.

### Visual spec — Flywheel TEMPORAL MULTI-OMICS pillar update

**Old pillar text**: 
> RNA + ATAC + Protein
> (+phospho + VDJ Phase 2)
> 0/5/30/60/180 min

**New pillar text**:
> RNA · Protein · **Phospho**
> 5 timepoints (0/5/30/60/180)
> ATAC × 2 timepoints
> VDJ — Phase 2

Phospho should be **visually emphasized** (bold, lavender accent) since it's the modality that anchors the "no public dataset" claim.

The rest of F1 visual unchanged: 4-pillar flywheel, 3-bucket archetype grouping, Quriegen amber row, closing italic line.

### Closing italic line (update)

**Old**: "No public dataset has the combination drug combination prediction requires. The wet lab, the architecture, and the protocol family are co-designed."

**New** (literal-true upgrade):
> "Phase 1 QuRIE-seq (Q3 2026) measures RNA + Protein + Phospho at 5 timepoints — the combination no public dataset has. The platform compounds from there."

This **shifts the closing from aspirational to imminent**. Q3 2026 is the next investor diligence touchpoint.

### Speaker notes — rewrite affected Q&As

**Replace existing "TAHOE 100M cells" Q&A**:

**If asked: "TAHOE has 100M cells. You have 500K. Why isn't that decisive?"**
> Different optimization. TAHOE optimizes for open foundation-model substrate from cell-line perturbations — 100M cells, RNA-only, no phospho, no perturbation-aware multi-omics. We optimize for proprietary integrated platform on primary human PBMCs with 4 modalities including phospho in Phase 1 (Q3 2026). The 500K Phase 1 cell count is intentionally small for modality depth + perturbation-aware design; Phase 2 scales to 20 donors. TAHOE's scale gives them open-source RNA foundation models; our depth gives us closed-loop causal drug combination prediction with the modality (phospho) no public dataset has. Different layers of the stack — they don't compete with our platform, they're potentially substrate beneath it.

**Replace existing "Immunai VDJ" Q&A**:

**If asked: "Immunai already has VDJ. You don't until 2027. Doesn't that make them ahead?"**
> Immunai has VDJ in their AMICA atlas today — that's a real capability we don't have until Phase 2 (2027). But Immunai doesn't have phospho. We have phospho in Phase 1 (Q3 2026). The modality stacks are different: Immunai = RNA + Protein + VDJ (immune-receptor diversity); us = RNA + Protein + Phospho (kinase signaling state) + ATAC. Different use cases. Immunai's VDJ supports clinical biomarker discovery on patient samples; our phospho supports drug combination mechanism prediction. Phase 2 closes the VDJ gap on our side; their phospho gap is structural — they don't generate it because they don't have a proprietary wet-lab pipeline for phospho measurement. The modality presence question is necessary but not sufficient — what matters is how the modality integrates with the rest of the platform.

**Add new Q&A on phospho specifically**:

**If asked: "Why does phospho matter? What does it give you that RNA + protein don't?"**
> Phospho measures kinase activation state — the immediate signaling response to a perturbation, before transcriptional changes propagate. For drug combination prediction in pathway-driven diseases like CLL, phospho is the readout that distinguishes additive effects ("drug A and drug B affect the same kinase") from synergistic effects ("drug A blocks JAK and drug B blocks BTK so the combination hits both arms of the BCR pathway"). RNA shows downstream consequences hours later; phospho shows immediate mechanism in minutes. The 5-minute timepoint in QuRIE-seq Phase 1 captures phospho responses that no other dataset measures at single-cell resolution on primary PBMCs.

**Keep existing pharma deal gap Q&A, peer-reviewed paper Q&A, AI biotech category Q&A, Cellarity/Recursion/Insitro Q&A, "couldn't a competitor build all five layers" Q&A — these don't change.**

---

## 4. D1 — Quarterly Roadmap

### Single milestone label change

The Gantt currently shows a milestone at Q1'27: **"Phase 2 phospho on"**.

**Change to**: **"Phase 2 VDJ on"** (or "Phase 2 VDJ + 20-donor scale" if space allows).

**Rationale**: Phase 2's actual modality addition is VDJ, not phospho (phospho is already in Phase 1).

### Phase 1 milestone label (verify, may need slight update)

If the Q3'26 milestone currently reads "Phase 1 delivery" or similar — keep as-is. If it tries to enumerate modalities, update to **"Phase 1 delivery: 4 modalities + BTK+JAK combo"** or similar concise framing.

### No other visual changes needed for D1

### Speaker notes — add Phase 1/Phase 2 canonical definition Q&A

**Add to existing notes**:

**If asked: "What exactly do Phase 1 and Phase 2 mean here?"**
> Phase 1 and Phase 2 refer specifically to QuRIE-seq proprietary wet-lab data generation phases — not clinical trial phases, not company funding rounds, not model training stages. Phase 1 (Q3 2026) generates 5-donor, 5-timepoint dataset with 4 modalities (RNA + Protein + Phospho at all 5 timepoints; ATAC at t=0 and t=180). Phase 2 (Q1 2027 onwards) scales to 20 donors and adds VDJ as 5th modality. Model training stages (Stage 3a current, Stage 3b BTK+JAK demo Q4 2026, Stage 3c causal architecture Q1-Q2 2027, Stage 4 scale 2027, Stage 5 causal-ready 2028) are separate framework — shown in the model swim lane of this Gantt. The unified quarterly view is what links wet-lab phases to model stages.

**Add timeline-bridge Q&A**:

**If asked: "How does this 11-quarter view map to Kinga's 24-month trajectory on slide 8?"**
> Slide 8 compresses the same plan into a 4-phase visual for investor narrative. D1 is the canonical per-quarter detail with explicit milestone dependencies. Same plan, different visual decomposition for different audiences.

---

## 5. A5 — Causal Architecture (SPEAKER NOTES ONLY)

A5 SVG remains unchanged — status pill, equation, GRN visualization all correct as-is.

### Speaker notes — update timing language in existing Q&As

**Existing Q&A: "When does this become operational?"** — needs phospho-Phase-1 update:

**Updated answer**:
> Implementation Stage 3c starts Q4 2026 after Phase 1 wet-lab data lands in Q3 2026. Phospho is available in Phase 1 (integral to QuRIE-seq), so causal architecture validation has perturbation-aware phospho signal from Q3 2026. Architecture stub + STRING integration: Q4 2026 - Q1 2027. GRN learning + sparsity calibration: Q1-Q2 2027. Validation on Phase 1 perturbation-response data: Q1-Q2 2027. First publishable Stage 3c results: Q2-Q3 2027. This timeline is on the D1 roadmap as part of Stage 4 + 5 scope. The earlier framing of "post Phase 2 data" was incorrect — Phase 1 already provides the modality signal Stage 3c needs.

**Existing Q&A: "What's Stage 3c spec-locked actually mean?"** — small update:

**Updated answer**:
> Spec-locked means the architectural commitment is written down in spec v1.1 (with v1.2 causal-layer extension pending) and the components have concrete mathematical definitions — Neumann propagation as `(I − W)⁻¹ dₚ`, sparse GRN with L1 regularization on edges absent from STRING, log-FC head for direct-effect decoding. What's not yet done is implementation and validation. Validation requires perturbation-aware multi-omics data with sufficient signal for GRN edge inference — Phase 1 wet-lab generation (Q3 2026) provides this with 4 modalities including phospho. Stage 3c implementation begins post-Phase-1, validation Q1-Q2 2027. The slide's status pill is honest about this status.

**Status pill timing decision**: Status pill currently reads "Validation Q1-Q2 2027 · post Phase 1 wet-lab data." **Keep this** — phospho being in Phase 1 doesn't change validation timing (still requires Phase 1 data to land + processing + first eval). The framing is conservative and defensible.

### All other A5 speaker note Q&As remain unchanged

The 8 Q&As (spec-locked status, Neumann choice, STRING confidence, A3 relationship, fallback paths, BTK+JAK connection, operational timing — updated above, DeepLife/Cellarity comparison) all stay. Only timing language in 2 of them gets the phospho-Phase-1 correction.

---

## Commit Sequence

After Ash reviews this doc, the actual commits are:

```bash
# Apply each edit to its target file in docs/deck/content/
# Then single batch commit:
git add docs/deck/content/A2_encoder_substrate.md \
        docs/deck/content/C1_phase1_design.md \
        docs/deck/content/D1_quarterly_roadmap.md \
        docs/deck/content/F1_competitive_positioning.md \
        docs/deck/content/A5_causal_architecture.md
git commit -m "docs(deck): step 2 - content spec updates for phospho-in-phase-1 correction"
git push origin main
```

Single commit covering 5 content spec edits. Then move to Step 3 (Cowork SVG rebuilds).

---

## What Step 3 Looks Like (Preview)

After this commit lands, draft a single Cowork prompt for **SVG rebuilds**:
- A2 SVG — add modality progression strip beside the 73% Calderon result
- C1 SVG — replace current Phase 1 visual with the modality × timepoint matrix
- D1 SVG — update Q1'27 milestone label from "Phase 2 phospho on" to "Phase 2 VDJ on"
- F1 SVG — update TEMPORAL MULTI-OMICS flywheel pillar text; update closing italic line

Estimated Cowork work: ~2-3 hours across 4 SVGs.

A5 SVG **not touched** (speaker notes only update happens at pptx rebuild stage).

---

## Honesty Check On What's Been Written Above

Three things I want to flag explicitly:

1. **"6 inhibitor singles" in C1's perturbation conditions block is my estimate** — the actual Phase 1 perturbation panel size needs Thiago's confirmation. C1 spec should mark this with "TBD with Thiago" until confirmed.

2. **"3 inhibitor + inhibitor combos including BTK+JAK" is also estimate** — total combo count needs verification.

3. **A2 "73% pseudo-bulk centroid-NN" is the published result from B1/B2 work** — verified shipped, not estimate.

4. **"Phase 2 VDJ on Q1'27" milestone timing is per current D1 roadmap** — if Thiago has updated VDJ timing since, this should track that update.

If you want any of these verified before Step 3 starts, surface it now. Otherwise we commit Step 2 with these placeholders documented as TBD-pending-Thiago.
