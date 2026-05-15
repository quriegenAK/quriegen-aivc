# Slide C1 — QurieSeq Phase 1: The Data That Makes the Model

- **Maps to Kinga's deck**: Extends slide 9 (lab-in-the-loop) + slide 5 (QurieSeq technology + IP moat)
- **Section**: C — QurieSeq Phase 1
- **Visual lead**: Experimental design grid — donors × timepoints × arms × modalities
- **Status**: Draft — pending Ash review

---

## Headline

**The data architected for the model. Phase 1 lands Q3 2026.**

(Alternative: *"Experimental design built around what the model needs to learn."*)

(Alternative: *"5 donors × 5 timepoints × 4 arms × 4 modalities — by design, not accident."*)

---

## Sub-headline (one line under headline)

QurieSeq Phase 1 is the first dataset designed top-down for temporal, perturbation-aware foundation modeling — irregular timepoints for early signaling, 4-arm design for compositional generalization, full PBMC lineage coverage for biological breadth.

---

## Body content (3 bullets max)

- **5 donors, 5 timepoints, 4-arm design, all PBMC lineages**: ~5,000 cells per donor per timepoint. Timepoints 0/5/30/60/180 min — capturing early signaling, transcriptional onset, peak response, and stable phenotypes. 4-arm structure per perturbation: vehicle, stim alone, inhibitor alone, stim+inhibitor combination. All major PBMC lineages: T (CD4/CD8), B, NK, Monocyte, DC.

- **Modality stack aligned with the architecture**: RNA + Protein measured per cell per timepoint via CITE-seq. ATAC integrated as static donor context (chromatin signature at t=0 — the same modality the encoder was pretrained on). Phospho and VDJ deferred to Phase 2 (Q1-Q2 2027) — the encoder is built to extend without re-architecting.

- **Stimuli and inhibitors chosen for clinical relevance**: Phase 1 stimuli include LPS, IFNγ/TNFα, SEB/TSST-1 + anti-CD28, SEB/TSST-1 + anti-IgM. Inhibitors include acalabrutinib (BTK), idelalisib (PI3K), IKK16 (NF-κB), rapamycin (mTOR). **BTK + JAK inhibitor combo CONFIRMED for Phase 1** — the headline demo target is in the experimental design.

---

## Visual spec (the experimental design grid)

Two-panel layout:

**Top panel — the experimental design grid:**

```
                                        T I M E P O I N T S
                              0min     5min     30min    60min    180min
                            ┌────────┬────────┬────────┬────────┬────────┐
Donor 1   │ Vehicle           │  RNA+P │  RNA+P │  RNA+P │  RNA+P │  RNA+P │
          │ Stim alone        │  RNA+P │  RNA+P │  RNA+P │  RNA+P │  RNA+P │
          │ Inhibitor alone   │  RNA+P │  RNA+P │  RNA+P │  RNA+P │  RNA+P │
          │ Stim + Inhibitor  │  RNA+P │  RNA+P │  RNA+P │  RNA+P │  RNA+P │
                            ├────────┴────────┴────────┴────────┴────────┤
                            │ ATAC (chromatin signature)  ◄── t=0 only    │
                            └─────────────────────────────────────────────┘
                            
Donor 2   │ ... same 4-arm × 5-timepoint × RNA+P grid                ...
Donor 3   │ ... same                                                  ...
Donor 4   │ ... same                                                  ...
Donor 5   │ ... same                                                  ...

Total: 5 donors × 5 timepoints × 4 arms × ~5,000 cells = ~500,000 cells
Modalities per cell: RNA + Protein. Donor-level ATAC at t=0.
```

**Bottom panel — design choices, explained (3 cards):**

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│ WHY 5 TIMEPOINTS       │  │ WHY 4-ARM PER PERT     │  │ WHY 5 DONORS           │
│                        │  │                        │  │                        │
│ Captures phospho-level │  │ Vehicle = baseline     │  │ Enough for donor-      │
│ signaling at 5 min     │  │ Stim = activation only │  │ specific static        │
│ (Phase 2 phospho-      │  │ Inh = inhibition only  │  │ context (chromatin     │
│ ready)                 │  │ Stim+Inh = synergy     │  │ signature per donor)   │
│                        │  │                        │  │                        │
│ Transcriptional onset  │  │ Direct match to the    │  │ Phase 2 scales to 20   │
│ at 30 min              │  │ decomposed readout     │  │ donors for cross-      │
│                        │  │ architecture (slide A3)│  │ donor generalization   │
│ Stable phenotype       │  │                        │  │ validation             │
│ at 180 min             │  │ Held-out arm = zero-   │  │                        │
│                        │  │ shot synergy demo      │  │                        │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
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

**If asked: "Why these specific timepoints? Why not evenly spaced?"**

> Three reasons. First, biology doesn't move on evenly spaced clocks — phospho-signaling happens within minutes, transcriptional response over tens of minutes, stable phenotypes over hours. The 0/5/30/60/180 design samples each of these biological timescales. Second, our Neural ODE temporal backbone (slide A4) is built to handle irregular timepoints natively — we don't need uniform spacing. Third, the 5-minute point specifically primes the dataset for Phase 2 phospho integration; even before phospho lands, the model learns "what's already changing very quickly" as a distinct latent state.

**If asked: "Why 5 donors? Isn't that a small N?"**

> 5 donors is right-sized for Phase 1, which is about validating the architecture on real time-course PBMC data. The donor-conditioned static context branch in the architecture (encoder receives donor-level chromatin signature at t=0) uses each donor as a distinct biological context, so 5 donors gives us 5 independent biological replicates of the entire 5×4 timepoint-arm structure — meaningful statistical power for the synergy demo and the donor-generalization eval. Phase 2 scales to 20 donors for cross-donor generalization validation at scale.

**If asked: "Why isn't ATAC measured per-timepoint?"**

> ATAC measures chromatin accessibility — relatively stable at the 3-hour scale of Phase 1. The cell's chromatin state at t=0 is the biologically relevant signature for the entire experimental window; sampling ATAC at every timepoint would multiply cost without proportional information gain. Donor-level ATAC at t=0 functions as the static context branch in the architecture — providing the encoder's "chromatin substrate" for that donor's biology. Phase 2 will explore whether per-timepoint ATAC adds signal beyond the static donor signature.

**If asked: "Why these specific inhibitors and stimuli?"**

> The stimuli (LPS, IFNγ/TNFα, SEB/TSST-1 + costim) span the major PBMC activation axes — TLR/innate, cytokine signaling, and T/B-cell receptor + costimulation. The inhibitors (acalabrutinib for BTK, idelalisib for PI3K, IKK16 for NF-κB, rapamycin for mTOR) target the major signaling hubs downstream of those activation pathways. The BTK + JAK combination specifically connects to the clinical evidence from the Ibrutinib + Ruxolitinib CLL trial. Phase 1 is built to validate the model on combinations that are both biologically informative and clinically grounded.

**If asked: "What about cell-type representation balance?"**

> All major PBMC lineages are present in each sample. T cells (CD4 + CD8) make up the majority; B, NK, Monocytes, and DCs are minorities but well-represented at ~5,000 cells per donor per timepoint. The 4-arm × 5-timepoint structure means each minor cell type gets enough cells (~200-500) per perturbation arm to participate in the synergy demo. Cell-type imbalance is handled via arm-balanced batch sampling during adapter training (implemented in Stage 3a Day 2 PR).

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
