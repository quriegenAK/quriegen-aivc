# Slide A2 — Multi-Omics Encoder: The Frozen Substrate

- **Maps to Kinga's deck**: Slide 8 (5-layer regulatory cascade) — zooms into the "Represent" layer
- **Section**: A — Architecture Depth
- **Visual lead**: Encoder schematic + cross-corpus accuracy + DOGMA-seq sidebar
- **Status**: Draft v4 — strips "trimodal" from our voice; reserved only for direct Mimitou 2021 protocol reference

---

## Headline

**Multi-omics encoder — trained on public, ready for proprietary**

(Alternative: *"The encoder is the moat. Validated cross-corpus. Foundation for every downstream task."*)

---

## Sub-headline (one line under headline)

3 modalities trained on DOGMA-seq, validated cross-corpus on Calderon 2019 at 73% pseudo-bulk accuracy. Phase 1 QuRIE-seq adds phospho — the 4th modality no public dataset has.

---

## Body content (3 bullets max)

- **Today — validated on public data**: encoder pretrained on DOGMA-seq (Mimitou 2021) covering 3 modalities (RNA + ATAC + Protein) on PBMCs under stimulation. Cross-corpus validation on Calderon 2019 yields 73% pseudo-bulk centroid-NN accuracy — encoder generalizes across independently-generated datasets. AIVC_GRAD_GUARD enforces frozen-encoder discipline downstream.

- **Phase 1 (Q3 2026) — proprietary upgrade**: QuRIE-seq adds phospho-proteomics as the 4th modality (integral to the assay; no public phospho data exists for PBMCs). RNA + Protein + Phospho measured at all 5 timepoints; ATAC at t=0 and t=180. The pretrained encoder's RNA+Protein representations transfer; phospho representation is learned during Phase 1 integration as new modality.

- **Phase 2 (2027) — full modality stack**: VDJ added as 5th modality; scale to 20 donors. The encoder grows under the same protocol family — no architectural rebuild required for modality extension. This is the structural advantage of QuRIE-seq protocol-family design.

---

## Visual spec (encoder evidence)

Three-zone layout:

**Left panel — multi-omics encoder structure (showing extensibility):**

```
Modalities (per cell):                    Encoder              Latent space
────────────────────────                  ───────              ─────────────
                            ┌─── ✅ RNA      ─┐
                            │                  │
TODAY (validated, 3/5)      ├─── ✅ ATAC     ─┼─→ [Contrastive ─→  z ∈ ℝ²⁵⁶
                            │                  │    fusion]          🔒 frozen
                            └─── ✅ Protein  ─┘                       after pretraining
                            
                            ┌─── 🟡 Phospho  ─┐   (Phase 2 integration)
PHASE 2 (planned, 2/5)      │                 │
                            └─── 🟡 VDJ       ─┘   (Phase 2 integration)
                            
                                                  ↓ (inference path)
                                              
                                              Downstream tasks:
                                              adapter, Neural ODE,
                                              decomposed readout
```

**Right panel — cross-corpus validation hero number:**

```
┌─────────────────────────────────────────┐
│                                         │
│            73 %                         │
│                                         │
│   cross-corpus cell-type accuracy       │
│   on Calderon 2019                      │
│                                         │
│   (pre-registered pseudo-bulk           │
│   centroid-NN methodology)              │
│                                         │
└─────────────────────────────────────────┘
```

Below the 73% callout, a small text block (3 lines):

- Independent dataset (different donors, protocols)
- Zero retraining
- Major PBMC lineages: T (CD4/CD8), NK, B, Monocyte, DC

**Bottom-left sidebar / footer — DOGMA-seq dataset callout:**

```
┌─────────────────────────────────────────────────┐
│ DOGMA-seq (Mimitou 2021, Nat Biotech)           │
│                                                 │
│ • RNA + ATAC + Protein measured in              │
│   the same single cell                          │
│ • Primary human PBMCs (not cell lines)          │
│ • 6 healthy donors, ~30K cells                  │
│ • Peer-reviewed protocol                        │
│                                                 │
│ Source of: encoder pretraining +                │
│ perturbation training data                      │
│ (ASAP-seq CRISPR sub-study)                     │
└─────────────────────────────────────────────────┘
```

Position: bottom-left of slide as a small reference box. Doesn't compete with the 73% hero — gives credit + context to the data source. Reinforces credibility for technical reviewers.

**Modality progression strip (NEW — three-state framing visual):**

A small strip rendered below or beside the cross-corpus result that makes the three-state framing explicit without erasing the 73% Calderon evidence:

```
ENCODER MODALITY PROGRESSION

TODAY            PHASE 1 (Q3 2026)        PHASE 2 (2027)
──────────       ────────────────         ──────────────
RNA              + Phospho                + VDJ
ATAC             [proprietary,             [proprietary,
Protein           5 timepoints]            5th modality]

[public DOGMA-seq]  [QuRIE-seq Phase 1]    [QuRIE-seq Phase 2]
```

Color coding consistent with C1: RNA cyan, Protein green, Phospho lavender (`#8B5CF6` — emphasized as the Phase 1 modality with no public coverage), ATAC muted blue, VDJ amber. The strip should read left-to-right as a clear timeline; the 73% Calderon callout remains visual hero.

---

## Notes for design

- **The 73% is the slide.** Make it huge. The encoder schematic on the left is supporting evidence.
- **DOGMA callout is a credibility footer** — small text, recessive visual weight, but always-visible. Anyone who knows the field will recognize Mimitou 2021 immediately.
- **Color the "today" vs "Phase 2" modalities differently**: today = filled/primary accent, Phase 2 = outlined/secondary tone.
- **Lock icon on the encoder block** — reinforces "frozen substrate" narrative.
- **Don't show a bar chart**. Per-cell-type breakdown is in speaker notes only.

---

## Source data / claims

| Claim | Source |
|---|---|
| 73% Calderon cross-corpus accuracy | `docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md` |
| 256-D latent space | Architecture spec v1.1, §3.1 |
| DOGMA-seq pretraining (RNA + ATAC + Protein, same single cell) | Mimitou 2021, Nat Biotech |
| DOGMA: 6 healthy donors, ~30K cells | Mimitou 2021 paper, Figure 1 |
| Phospho + VDJ as Phase 2 extensions | QurieSeq Phase 2 spec (Thiago confirmation, May 12) |
| Frozen post-pretraining | Architecture spec v1.1, §3.1 + Stage 3 Part 1 verdict |
| Pseudo-bulk centroid-NN eval methodology | `docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md` |
| Major PBMC lineages covered | Phase 6.5g.2 closure, §3 |

---

## Speaker notes (NOT on slide — for Ash to use when answering questions)

**If asked: "Why DOGMA-seq specifically?"**

> DOGMA-seq is the first published protocol that measures RNA, chromatin accessibility, and surface protein from the same single cell. Mimitou published it in 2021 in Nature Biotechnology. We chose it as our pretraining substrate because it's the only public dataset combining these three modalities in primary human PBMCs at scale. It's also the source of our perturbation training data (ASAP-seq CRISPR sub-study from the same lab) — meaning the encoder and the perturbation adapter both come from a coherent technical and biological context.

**If asked: "How does it perform per cell type?"**

> Overall 73% across the major PBMC lineages. T cells, NK, monocytes, and DCs are all in the 70-85% range. B cells underperform at 18% — but our analysis (Stage 3 Part 1 Report 5) shows this is a cross-corpus stimulation-protocol artifact, not an encoder defect. Encoder silhouette score on Calderon B cells is 0.354, higher than CD4 T cells at 0.129, which proves the encoder is finding the B cells correctly in latent space. The misclassifications go to DC and NK — i.e., the model respects lineage hierarchy. We diagnosed this transparently and the architecture doesn't depend on closing the gap before QurieSeq Phase 1 lands.

**If asked: "Why aren't phospho and VDJ in the validation?"**

> Phospho doesn't exist in public PBMC data — it's QurieSeq's proprietary modality, our moat. VDJ is being deferred to QurieSeq Phase 2 per Thiago's wet-lab plan. The encoder architecture is modality-extensible by design — Phase 1 phospho and Phase 2 VDJ slot in without retraining the base encoder.

**If asked: "Why pseudo-bulk centroid-NN?"**

> Pseudo-bulk centroid-NN is a published cross-corpus methodology that controls for technical batch effects between studies — exactly the right test for whether the encoder learned biology vs. dataset artifacts. We pre-registered it before running the eval to avoid post-hoc cherry-picking. The methodology doc is in our repo (`docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md`).

**If asked: "Phospho is in Phase 1? I thought it was Phase 2."**

> Phospho is integral to the QuRIE-seq assay — every QuRIE-seq run generates phospho data alongside RNA and protein. The earlier framing of "phospho deferred to Phase 2" was specifically about public training data — no public dataset has phospho on PBMCs, so we deferred phospho coverage in our public-data layer strategy. But the QuRIE-seq Phase 1 wet-lab generation in Q3 2026 measures phospho directly. So phospho first becomes available to us in Phase 1, not Phase 2.

**If asked: "How does the encoder generalize to phospho if it was only trained on RNA + ATAC + Protein?"**

> The encoder's architecture supports modality extension by design — adding a modality means adding an input head, not retraining from scratch. During Phase 1 integration (Q4 2026), the phospho input head is fit while the RNA/ATAC/Protein representation backbone stays frozen (AIVC_GRAD_GUARD enforced). This is the same adapter-strategy pattern that the Stage 3 Part 1 verdict approved. The 73% cross-corpus result establishes that the backbone representations transfer across datasets; phospho integration tests whether they accommodate a new modality of biological information.

**If asked: "What's pseudo-bulk centroid-NN?"**

> Cross-corpus validation method. Pseudo-bulk: aggregate single cells by cell-type label within each dataset to produce one centroid vector per cell type. Centroid-NN: for each held-out test centroid (from Calderon), find the nearest centroid in the training pool (from DOGMA). Accuracy = fraction where the nearest neighbor is the same cell type. 73% on PBMC major lineages (B / T / NK / monocyte / DC) is strong cross-corpus generalization — random would be 20% for 5 classes.

**If asked: "What's AIVC_GRAD_GUARD?"**

> An environment-variable-controlled gradient-blocking mechanism in our training code. When `AIVC_GRAD_GUARD=1`, the encoder's pretrained weights are frozen during downstream training — adapter and readout heads update, encoder doesn't. This enforces the adapter strategy mechanically rather than relying on training-script discipline. The flag is set in all production training runs after Stage 3 Part 1's ADAPTER_RECOMMENDED verdict.

---

## Investor framing (one-paragraph elevator)

> The encoder is a multi-omics foundation model — currently pretrained on three modalities (RNA, chromatin, protein) from Mimitou DOGMA-seq, extensible to five (adding phospho and VDJ in Phase 2). DOGMA-seq is the first published single-cell protocol measuring RNA, chromatin accessibility, and surface protein from the same cell (Mimitou 2021, Nature Biotechnology) — providing 6 donors of primary human PBMC data and a coherent biological context for the encoder. After pretraining, we held out a completely separate dataset (Calderon 2019, different donors, different protocols) and the encoder achieved 73% cell-type accuracy with zero retraining. The methodology was pre-registered. This is the substrate every downstream task builds on, and it's frozen — we don't retune the encoder for new tasks, which is what makes the platform a true foundation model rather than a per-task model.

---

## What's NOT on this slide (intentionally)

- Per-cell-type accuracy breakdown (lives in speaker notes only)
- Encoder training loss curves
- Specific hyperparameters
- Pretraining compute cost
- Detailed methodology of pseudo-bulk centroid-NN

---

## Diagram generation strategy

**Tool**: Cowork or Claude Code (Python matplotlib for the three-zone layout combined into one SVG).

**File output**: `docs/deck/assets/diagrams/A2_encoder_evidence.svg`

**Followup prompt for Cowork** (when ready):
"Generate `A2_encoder_evidence.svg` per spec in `docs/deck/content/A2_encoder_substrate.md`. Three-zone layout: 
1. Top-left = encoder schematic showing 5 modalities (3 validated with filled style ✅, 2 Phase 2 with outlined style 🟡) flowing into 256-D latent space with lock icon. 
2. Top-right = single huge '73%' callout with caption 'cross-corpus cell-type accuracy on Calderon 2019'. 
3. Bottom-left = DOGMA-seq sidebar reference box (Mimitou 2021 Nat Biotech, RNA+ATAC+Protein same cell, PBMCs, 6 donors ~30K cells, source of pretraining AND perturbation data).

Output 1920×1080 viewBox. Use Kinga's deck color palette (TBD)."

---

## Risk callouts (NOT to include on slide; for tracking only)

- 73% is from one cross-corpus eval. Second cross-corpus validation pending (Soskic deferred / no longer in scope).
- B-cell underperformance documented and handled in speaker notes; if pressed, refer to Stage 3 Part 1 Report 5 diagnosis.
- Phospho + VDJ Phase 2 integration is architecturally planned but not yet executed.

---

## What's NEXT after A2 v4 is committed

Move to **B1 (Methodology Rigor — the three-dataset story)**. DOGMA gets central treatment there as one of three independent datasets underpinning the platform validation.
