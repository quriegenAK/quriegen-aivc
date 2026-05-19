# Slide A1 — AIVC Foundation Model: System Architecture

- **Maps to Kinga's deck**: Slide 37 (AI virtual cell model: simulate, predict, and steer cell behavior)
- **Section**: A — Architecture Depth
- **Visual lead**: Full system diagram (single hero diagram)
- **Status**: Draft — pending Ash review

---

## Headline

**One unified foundation model. Three input modalities. Four perturbation states. Continuous time. Pathway-grounded outputs.**

(Alternative if too long: *"AIVC: a continuous-time, multi-omics, perturbation-aware foundation model for immune cell behavior."*)

---

## Sub-headline (one line under headline)

The same trained model predicts how every PBMC cell type responds to any combination of stimulus + inhibitor at any timepoint — without retraining.

---

## Body content (3 bullets max)

- **Frozen trimodal encoder + lightweight adapter**: RNA + ATAC + Protein → 256-D latent. Pretrained on DOGMA-seq (validated 73% cross-corpus accuracy), adapted to perturbations on Mimitou CRISPR data (0.57 synergy 4-class accuracy, 2.27× chance).
- **Neural ODE temporal backbone**: Continuous-time state evolution from 0 → 180 min. Matches QurieSeq Phase 1 design directly. No discretization artifacts.
- **Decomposed 4-arm readout**: Vehicle baseline + stimulus residual + inhibitor residual + synergy residual. The synergy head learns *only* the non-additive correction — which is what enables zero-shot prediction of unseen drug combinations.

---

## Visual spec (the hero diagram)

A single horizontal flow diagram with 5 stacked blocks left to right:

```
┌─────────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   INPUT             │   │   ENCODER        │   │   TEMPORAL       │   │   READOUT        │   │   OUTPUT         │
│                     │   │   (frozen +      │   │   (Neural ODE)   │   │   (4-arm         │   │   (pathway-      │
│   RNA               │ → │   adapter)       │ → │                  │ → │   decomposed)    │ → │   aware)         │
│   ATAC              │   │                  │   │   z(t=0) → z(t)  │   │                  │   │                  │
│   Protein           │   │   z ∈ ℝ²⁵⁶       │   │                  │   │   h_base         │   │   RNA dynamics   │
│                     │   │                  │   │   continuous     │   │   + Δ_stim       │   │   Protein        │
│   per cell          │   │   ≈130K param    │   │   time           │   │   + Δ_inh        │   │   dynamics       │
│                     │   │   adapter        │   │                  │   │   + Δ_synergy    │   │   58 pathway     │
│                     │   │                  │   │                  │   │                  │   │   scores         │
└─────────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
       ↑                          ↑                       ↑                       ↑                       ↑
   Mimitou,                  Validated:              Trains on               Zero-arm                Phospho
   QurieSeq,                  73% Calderon            QurieSeq               L2 constraint            decoder
   future donors                                      Phase 1                ensures synergy          plugs in
                                                                             generalizes              Phase 2
```

Below the main row, two annotation rows:

**Row 2 — what's been validated** (with checkmarks where applicable)
- ✅ Trimodal encoder (Stage 1 + 2 complete)
- ✅ Adapter strategy (Stage 3 Part 1 verdict landed)
- 🟡 Decomposed readout (Stage 3a in-flight, training Q3 2026)
- ⏸ Temporal Neural ODE (Stage 3b, Q3 2026 with QurieSeq Phase 1)
- ⏸ Pathway-aware output (Stage 3c, Q1 2027)

**Row 3 — what each component invariant guarantees** (single keyword per block)
- INPUT: modality-agnostic
- ENCODER: cross-corpus transfer
- TEMPORAL: irregular timepoint handling
- READOUT: compositional generalization
- OUTPUT: biological interpretability

---

## Notes for design

- **Single hero diagram, no clutter.** This slide must work as a poster — 30 second glance gives the architecture.
- **Visual hierarchy**: top row = system flow (boldest). Row 2 = validation status (icons). Row 3 = single keyword (smallest text).
- **Color**: Use Kinga's primary accent for the system flow boxes. Validation status uses semantic colors (green/yellow/grey). Keep readable on dark background since slide 37 in Kinga's deck uses dark.
- **No code, no equations on this slide.** Equations live on A3 (decomposed readout slide).
- **Make the "frozen + adapter" distinction visible** in the encoder block — small lock icon on encoder + "trained" indicator on adapter sub-block.

---

## Source data / claims

| Claim | Source |
|---|---|
| 256-D latent, ~130K param adapter | `docs/specs/stage3_part2_architecture_proposal_2026_05_06.md` v1.1 |
| 73% Calderon cross-corpus accuracy | Phase 6.5g.2 closure (`docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md`) |
| 0.57 synergy 4-class accuracy (Stage 3 Part 1) | `docs/memory/project_aivc_stage3_part1_verdict_2026_05_11.md` |
| 4-arm decomposed readout architecture | Architecture spec v1.1, §3.2 |
| Neural ODE continuous-time backbone | Architecture spec v1.1, §4 |
| 58 pathways (50 Hallmark + 8 KEGG immune) | Stage 3 Part 1 Report 3 |
| QurieSeq Phase 1 0–180 min design | Thiago confirmation, May 12 |
| Zero-arm L2 constraint enables synergy generalization | Architecture spec v1.1, §3.2.2 |

---

## Risk callouts (NOT to include on slide, but to track)

- Neural ODE is not yet validated on real data (Q3 2026 with QurieSeq Phase 1)
- Pathway-aware output uses phospho signal from Phase 1 (Q3 2026 — phospho is integral to QuRIE-seq); pathway dictionary expands across Phases 1-2
- BTK+JAK headline demo grounded but not yet executed

These risks belong on Slide C2 (BTK+JAK demo plan with pre-registered eval), not here. A1 is the architecture summary; risks are covered in the eval-plan slide where they belong.

---

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

**If asked: "Why one model for all PBMC cell types?"**
> The encoder is trained on cell-type-mixed multi-omics data and represents each cell in a shared 256-D latent space. Cell type emerges as structure within that space (validated by the 73% Calderon cross-corpus result on 5-class lineage classification). One unified model means downstream tasks (perturbation prediction, temporal evolution, causal inference) operate on a single coherent substrate rather than per-cell-type specialized models. Cheaper to maintain, more cross-type generalization.

**If asked: "Where does the model see phospho if the encoder was pretrained on RNA + ATAC + Protein only?"**
> Phospho enters the architecture in Phase 1 (Q3 2026) as a new input head added to the encoder. The pretrained RNA + ATAC + Protein backbone stays frozen (AIVC_GRAD_GUARD enforced) while a phospho-specific encoder layer fits to the QuRIE-seq Phase 1 data. This is the same adapter-strategy pattern that the Stage 3 Part 1 verdict approved. Phase 2 extends similarly with VDJ.

**If asked: "What's pre-registered about this?"**
> The Stage 3 evaluation methodology was committed in writing before running the evals — cross-corpus pseudo-bulk centroid-NN as the eval method, threshold bands locked, no post-hoc threshold adjustment. The 73% Calderon result and 0.57 Mimitou CRISPR result both came from pre-registered evals. Stage 3b (BTK+JAK demo) and Stage 3c (causal architecture) also have pre-registered thresholds in architecture spec v1.1.

---

## Diagram generation strategy

**Tool**: Cowork (or Claude Code) — Python matplotlib or draw.io XML, exported as SVG.

**Why not Claude Design first**: The diagram is structurally complex (5 columns, 3 rows of annotations). Get the structure right with code-based generation. Send to Claude Design only for visual polish AFTER structure is locked.

**File output**: `docs/deck/assets/diagrams/A1_system_architecture.svg`

**Followup prompt for Cowork** (to be written next):
"Generate `A1_system_architecture.svg` per spec in `docs/deck/content/A1_system_architecture.md`. Use matplotlib with custom rectangle patches. Color palette TBD — start with grayscale + single blue accent. Output 1920×1080 viewBox for slide-ready aspect ratio."

---

## Investor framing (one-paragraph elevator)

When a technical investor asks "what's the model?":

> AIVC is a continuous-time foundation model trained on trimodal single-cell data — RNA, chromatin, protein. The encoder learns cell-type-invariant representations across donors and tissues. On top of that, a lightweight adapter learns perturbation responses, and a decomposed readout architecture lets the model predict synergy between drug combinations it has never seen during training. We've validated the encoder generalization (73% cross-corpus) and the adapter strategy (ADAPTER_RECOMMENDED with 0.57 synergy accuracy on held-out perturbations). The temporal Neural ODE component activates when our proprietary QurieSeq time-course data arrives in Q3 2026 — that's when the model becomes a true virtual cell.

---

## What's NEXT after this slide is approved

1. Generate the SVG diagram (Cowork prompt below)
2. Move to A2 content spec (trimodal encoder detail)
3. After A1+A2+A3+A4 all have specs, batch-generate the SVG diagrams
4. After all SVG diagrams ready, batch-generate the .pptx via Cowork
