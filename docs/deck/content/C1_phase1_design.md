# Slide C1 — QurieSeq Phase 1: The Data That Makes the Model

- **Maps to Kinga's deck**: Extends slide 9 (lab-in-the-loop) + slide 5 (QurieSeq technology + IP moat)
- **Section**: C — QurieSeq Phase 1
- **Visual lead**: Experimental design grid — donors × timepoints × arms × modalities
- **Status**: Draft — pending Ash review

---

## Headline

**QuRIE-seq Phase 1 — 5 donors × 5 timepoints × 4 modalities × BTK+JAK combo**

(Alternative: *"The data architected for the model. Phase 1 lands Q3 2026."*)

(Alternative: *"Experimental design built around what the model needs to learn."*)

---

## Sub-headline (one line under headline)

First proprietary perturbation-aware multi-omics dataset on primary human PBMCs. Phospho is integral to QuRIE-seq. BTK+JAK combo confirmed. Q3 2026 delivery.

---

## Body content (3 bullets max)

- **4 modalities measured directly on the same cells**: RNA, surface protein (CITE-seq), and phospho-proteins at all 5 timepoints (0/5/30/60/180 min); ATAC at t=0 and t=180 only because chromatin accessibility varies on slower timescales than transcription. No public dataset combines these modalities on primary PBMCs with perturbation-aware design.

- **Experimental design tied to model architecture**: 5 timepoints match the Neural ODE temporal backbone. 4-arm design (vehicle / stim / inhibitor singles / combinations) matches the decomposed readout. BTK + JAK combination condition is confirmed for Phase 1 — this is the proof-of-capability for compositional generalization (predict combo from singles).

- **5 donors × ~5k cells/donor/timepoint = ~125k cells total across ~17 conditions**. Donor chromatin signature (ATAC at t=0) functions as biological donor ID since Sanquin blood provider returns blood-type-only metadata. Phase 2 (2027) scales to 20 donors and adds VDJ as 5th modality.

---

## Visual spec — THE MODALITY × TIMEPOINT MATRIX (visual hero)

C1's new visual hero replaces the previous experimental-design grid. The
matrix below makes the per-modality / per-timepoint coverage immediately
legible — investors should read "4 modalities measured, phospho at all 5
timepoints" in 3 seconds.

**Top panel — modality × timepoint matrix:**

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

**Lower zone — two side-by-side blocks:**

**Left block — perturbation conditions:**

```
PERTURBATION CONDITIONS (17 total)
· Vehicle control (baseline)
· Stimulus (PMA/Iono or equivalent — TBD with Thiago)
· 6 inhibitor singles: BTK, JAK, [+4 others TBD]
· 6 inhibitor + stim combos
· 3 inhibitor + inhibitor combos including BTK+JAK
```

**Right block — wet-lab parameters:**

```
WET-LAB PARAMETERS
· 5 donors (Sanquin, blood-type-only metadata)
· All major PBMC lineages (B / T / NK / monocyte / DC)
· ~5k cells/donor/timepoint
· ~125k cells total
· QuRIE-seq protocol family (proprietary)
· Q3 2026 delivery target
```

---

## Notes for design

- **The experimental design grid is the slide.** Make it readable — investors should see "5 × 5 × 4 × ~5000 cells = ~500,000 cells" without squinting.
- **Use a color-coded modality treatment**: RNA = blue, Protein = green, ATAC = brand accent (highlighting donor-level static input). Consistent with A2 color scheme.
- **The "BTK + JAK CONFIRMED" mention** should feel weighted — small but unmistakable, with a green check icon. This is the moment investors see the headline demo target is locked in the experimental design, not aspirational.
- **Don't show stimuli/inhibitor list as a wall of names**. Reference them in body bullet 3 and speaker notes; on the slide, just the count ("4 stimuli, 4 inhibitors + BTK+JAK combo").
- **The 5×5×4×5K math** is the credibility number — make it readable.

---

## Why this slide matters

This is the **first slide where investors see we own the data flow**. Three things it earns:

1. **Experimental specificity**: Most pitches at this stage hand-wave the proprietary data. We've got the exact design — 5 donors, 5 timepoints, 4 arms, ~500K cells.
2. **Architecture-data coupling**: The slide makes it visually obvious that the experimental design isn't arbitrary — it's chosen to match what the architecture needs (decomposed readout, Neural ODE temporal, donor-conditioned static context).
3. **Headline-demo lock-in**: "BTK + JAK combo CONFIRMED for Phase 1" — this is the moment investors realize the demo isn't aspirational, it's already in the wet-lab plan.

---

## Source data / claims

| Claim | Source |
|---|---|
| 5 donors, ~5,000 cells per donor per timepoint | Thiago confirmation, May 12 |
| 5 timepoints: 0/5/30/60/180 min | Thiago confirmation, May 12 |
| 4-arm design: vehicle / stim / inh / combo | Thiago confirmation + architecture spec v1.1, §5 |
| RNA + Protein measured per timepoint | Thiago confirmation |
| ATAC at donor level (chromatin signature, available to model) | Kinga clarification ("we will be using it"), May 12 |
| Phospho deferred to Phase 2 | Thiago confirmation (Phase 2 scope) |
| VDJ deferred to Phase 2 | Thiago confirmation + Kinga "VDJ later" confirmation |
| Phase 1 stimuli (LPS, IFNγ/TNFα, SEB/TSST-1 + costim) | Thiago Phase 1 spec |
| Phase 1 inhibitors (acalabrutinib, idelalisib, IKK16, rapamycin) | Thiago Phase 1 spec |
| BTK + JAK combo confirmed for Phase 1 | Thiago confirmation, May 12 ("Both inhibitors scheduled to be bought and used in Phase 1") |
| Phase 2 scale to 20 donors | QurieSeq roadmap (Phase 2 specs) |
| All major PBMC lineages | Thiago confirmation ("all major lineages") |

---

## Speaker notes

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

## Investor framing (one-paragraph elevator)

> QurieSeq Phase 1 is the first dataset designed top-down for temporal, perturbation-aware foundation modeling. 5 donors × 5 timepoints × 4-arm experimental structure × ~5,000 cells = ~500,000 cells of multi-omics PBMC data, with RNA and Protein measured per cell per timepoint, donor-level ATAC chromatin context, and confirmed inclusion of the BTK + JAK inhibitor combination — our headline zero-shot synergy demo target. The experimental design is not generic; it's matched to the architecture's needs. 5-minute timepoint primes the dataset for Phase 2 phospho integration. 4-arm structure matches the decomposed readout. 5 donors give us 5 biological replicates with cross-donor generalization tests deferred to Phase 2's 20-donor expansion. This is the moat in motion.

---

## What's NOT on this slide (intentionally)

- The Phase 2 phospho panel details (~17 antibodies) — lives in E1 horizon slide
- VDJ Phase 2 spec — lives in E1
- L1/L2/L3 public-data layer integration — lives in slide D1 roadmap or speaker notes
- Detailed clinical rationale for inhibitor choice — speaker notes only
- The specific Calderon-vs-DOGMA-vs-QurieSeq donor identity strategy — speaker notes

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — gridded experimental design + 3-card design rationale.

**File output**: `docs/deck/assets/diagrams/C1_phase1_experimental_design.svg`

**Followup prompt for Cowork** (when ready):
"Generate `C1_phase1_experimental_design.svg` per spec in `docs/deck/content/C1_phase1_design.md`. Top panel: experimental design grid — 5 donor rows × 5 timepoint columns × 4 arms per cell (vehicle/stim/inh/combo), with RNA+Protein labels in each cell and ATAC chromatin signature shown as a donor-level static input at t=0. Total cell count ~500,000 visible. 'BTK + JAK combo CONFIRMED for Phase 1' as a green-check callout. Bottom panel: 3-card row labeled WHY 5 TIMEPOINTS / WHY 4-ARM / WHY 5 DONORS with biological + architectural rationale per card. Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- ATAC integration approach not yet clarified between Thiago ("not measured ourselves, integrate via RNA anchoring") and Kinga ("we are incorporating it, will be using it"). C1 currently treats ATAC as available to the model regardless of measurement source. If Kinga clarifies that ATAC is measured directly in Phase 1, the slide tightens; if it's integrated only, the slide still stands. Pending Kinga's reply.
- 5-donor sample size is small — bootstrap CI on cross-donor generalization will be wide. Phase 2 expansion addresses this.
- The "BTK + JAK CONFIRMED" claim depends on the wet-lab plan holding through Q3 2026 — if procurement slips or donor recruitment delays, the demo target slips with it.
- Protein antibody panel size (~37 like Mimitou or larger) not yet specified — defer C1 detail to next iteration.

---

## What's NEXT after C1 is committed

Move to **C2 (BTK+JAK Headline Demo Plan — Pre-Registered Eval)**. Closes Section C. Translates the C1 experimental design into the specific eval flow that will run in Q3 2026 — what we train on, what we hold out, what verdict the synergy accuracy maps to.
