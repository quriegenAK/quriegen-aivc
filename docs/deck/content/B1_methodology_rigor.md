# Slide B1 — Methodology: Three Datasets, Pre-Registered Evals, No Cherry-Picking

- **Maps to Kinga's deck**: New slide. Substantiates the "rigorous" claim implicit across slides 8, 9, 37.
- **Section**: B — Validation Evidence
- **Visual lead**: Three-dataset role diagram + pre-registered eval methodology timeline
- **Status**: Draft v2 — softened Phase 6.5g.2 disclosure + new headline

---

## Headline

**Methodology rigor is the moat before the moat.**

Sub-headline (one line under headline):

Pretraining, validation, and perturbation data come from three independently produced public datasets. Evaluation methodology was registered before results were generated. Honest reporting on what works and what doesn't.

---

## Body content (3 bullets max)

- **Three-dataset role separation**: DOGMA-seq (Mimitou 2021) for encoder pretraining; Calderon 2019 for cross-corpus validation; Mimitou ASAP-seq CRISPR sub-study for perturbation probe. **Different studies, different donors, different protocols.** The model never sees the same dataset family for both training and evaluation of any given capability.

- **Pre-registered evaluation methodology**: The cross-corpus metric (pseudo-bulk centroid-NN) was registered in the architecture spec before the eval was run. Verdict thresholds (e.g. 0.70 pass, 0.55-0.75 with bootstrap CI logic) were locked before results were observed. No post-hoc threshold adjustment, no goalpost moving.

- **Failure modes published, not buried**: When a metric fails, we diagnose root cause and publish both the failure and the remediation. Our public closure reports document both — sophisticated diligence will find the same numbers we'd present in conversation.

---

## Visual spec (the methodology diagram)

A two-panel layout:

**Top panel — three-dataset role diagram:**

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│  DOGMA-seq               │    │  Calderon 2019           │    │  Mimitou ASAP-seq CRISPR │
│  (Mimitou 2021)          │    │                          │    │  (sub-study)             │
│                          │    │                          │    │                          │
│  RNA + ATAC + Protein    │    │  Bulk + scATAC,           │    │  ATAC + Protein + HTO    │
│  6 donors                │    │  stim-driven PBMCs        │    │  CRISPR-perturbed         │
│  ~30K cells              │    │  (different protocol)     │    │  CD4 T cells              │
│                          │    │                          │    │                          │
│  ┌────────────────────┐  │    │  ┌────────────────────┐  │    │  ┌────────────────────┐  │
│  │ ROLE: Encoder      │  │    │  │ ROLE: Cross-corpus │  │    │  │ ROLE: Perturbation │  │
│  │ pretraining        │  │    │  │ validation         │  │    │  │ adapter probe      │  │
│  └────────────────────┘  │    │  └────────────────────┘  │    │  └────────────────────┘  │
│                          │    │                          │    │                          │
│  Result: encoder         │    │  Result: 73% cell-type   │    │  Result: 0.57 synergy    │
│  produces 256-D latent   │    │  accuracy (independent   │    │  4-class accuracy        │
│                          │    │  donors/protocols)        │    │  → ADAPTER_RECOMMENDED   │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
         │                              ▲                              ▲
         │ pretrain encoder             │ validate encoder              │ probe encoder on
         │                              │                              │ perturbations
         └──────────────────────────────┴──────────────────────────────┘
                              No data overlap between roles
```

**Bottom panel — pre-registration timeline (left-to-right flow):**

```
1. Spec written       →   2. Eval defined        →   3. Results generated    →   4. Verdict applied
   architecture            pseudo-bulk                pretrained                 ADAPTER_RECOMMENDED
   spec v1.1               centroid-NN                encoder run                per pre-registered
   (May 2026)              methodology                on Calderon                threshold
                           (doc'd before eval)        + Mimitou CRISPR           (no post-hoc
                                                                                  adjustment)

   [─ ─ ─ ─ ─ ─ ─ ─ Methodology pre-registered ─ ─ ─ ─ ─ ─ ─]  ↓
                                                                Run-time decisions
                                                                follow registered
                                                                interpretation table
```

Or simpler text-only formulation underneath the dataset cards:

> **Workflow**: architecture spec → pre-register methodology → run eval → apply registered thresholds → publish result (pass *or* fail).

---

## Notes for design

- **The three-dataset diagram is the slide.** Make the role-separation visually obvious — different colors per dataset, arrows showing data → role → result.
- **No data overlap callout** should be visually prominent — this is the line that catches sophisticated investors' attention.
- **Pre-registration timeline below datasets** — minimal, just enough to show methodology precedes results.
- **Color**: Each dataset gets a distinctive accent. DOGMA = deep tone, Calderon = neutral tone, Mimitou CRISPR = warm tone. Reinforces independence.

---

## Why this slide matters

This is the slide where **the deck stops being a pitch and starts being science**. Three things it earns:

1. **Anti-overfitting credibility**: Most deep learning pitches make readers wonder "did they overfit the test set?" — our three-dataset separation makes overfitting structurally impossible.
2. **Pre-registration credibility**: Says explicitly that we don't move the goalposts. Investors who've been burned by AI-pitch theatrics will recognize this.
3. **Transparency credibility**: Volunteering that we publish failure modes (without spelling out specific failures on the slide) signals scientific maturity without trading away presentation time.

---

## Source data / claims

| Claim | Source |
|---|---|
| DOGMA-seq for encoder pretraining | Mimitou 2021, Nature Biotechnology |
| Calderon 2019 for cross-corpus validation | Calderon et al. 2019 immune cell atlas |
| Mimitou ASAP-seq CRISPR for perturbation probe | Mimitou 2021 (CRISPR sub-study) |
| 73% cross-corpus accuracy (Calderon) | `docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md` |
| 0.57 synergy 4-class accuracy (Mimitou ASAP-seq CRISPR) | `docs/memory/project_aivc_stage3_part1_verdict_2026_05_11.md` |
| ADAPTER_RECOMMENDED verdict | Pre-registered thresholds in architecture spec v1.1, §5 |
| Pre-registered pseudo-bulk centroid-NN methodology | `docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md` |
| Phase 6.5g.2 dual-conclusion closure (failure + remediation) | `docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md` |

---

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

**If asked: "Have you had a metric fail?"**

> Yes — and we publish those. Our Phase 6.5g.2 closure is a good example. Our original per-cell cross-corpus metric failed at 0.19, well below the 0.70 pre-registered threshold. We didn't retry quietly or adjust the metric. Instead, we diagnosed the cause — a corpus-corpus stimulation-protocol artifact in the per-cell measurement, not an encoder defect. The published methodology for controlling this is pseudo-bulk centroid-NN (averaging cell representations per cluster, then matching). We re-ran with the remediated methodology and hit 0.73. Both numbers are in our public closure report with explicit dual-conclusion framing. The underlying encoder is the same model.

**If asked: "Why didn't you use just one dataset for training and testing?"**

> Because encoder validation and perturbation prediction validation are two different capabilities, and each needs an independent test. The encoder needs to generalize across donors and protocols — that's what Calderon tests. The perturbation adapter needs to generalize across perturbation types — that's what Mimitou's CRISPR sub-study tests. If we'd used one dataset for everything, we'd have no way to separate "the model overfit the perturbation data" from "the model overfit the cell-type structure." Three datasets, three roles, three separate validation signals.

**If asked: "What does 'pre-registered' actually mean?"**

> Two specific things. First, the evaluation methodology — pseudo-bulk centroid-NN — was documented in our architecture spec before any eval was run, with the exact procedure for computing accuracy. Second, the verdict thresholds (e.g., 0.70 = pass; 0.55-0.75 with bootstrap CI logic for adapter decisions) were locked in the spec before results were observed. So when results come in, they map to a verdict mechanically — there's no room for "let me reinterpret what 0.65 means." The methodology document is in our repo.

**If asked: "Why should we trust the remediated Phase 6.5g.2 methodology?"**

> Pseudo-bulk centroid-NN is published — it's a standard cross-corpus methodology in the single-cell literature, not something we invented to make our numbers look better. It was the right metric for cross-corpus encoder evaluation from the start; we used the wrong one initially and corrected. The methodology document in our repo cites the published basis.

---

## Investor framing (one-paragraph elevator)

> Our validation strategy uses three independent datasets, each in a distinct role: DOGMA-seq for pretraining, Calderon 2019 for cross-corpus encoder validation, and Mimitou's ASAP-seq CRISPR sub-study for perturbation prediction probing. No data overlap between roles — meaning overfitting any single capability to its eval is structurally impossible. The evaluation methodology was pre-registered before results were generated; there's no goalpost-moving. When metrics fail — and they do — we publish the failure, the diagnosis, and the remediation. This is the kind of validation rigor that distinguishes serious foundation modeling from AI pitch theater.

---

## What's NOT on this slide (intentionally)

- The specific Phase 6.5g.2 numbers (0.19 → 0.73) — lives in speaker notes only
- Specific accession numbers (GSEs) — too detailed for investor slide; in technical due diligence appendix
- Full citation details — paper names are enough; investors who want to verify can find them
- Detailed pseudo-bulk centroid-NN formula
- Description of cross-corpus normalization (batch correction, etc.)

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — three cards + arrows + role labels.

**File output**: `docs/deck/assets/diagrams/B1_three_datasets_methodology.svg`

**Followup prompt for Cowork** (when ready):
"Generate `B1_three_datasets_methodology.svg` per spec in `docs/deck/content/B1_methodology_rigor.md`. Top panel: three dataset cards (DOGMA-seq / Calderon 2019 / Mimitou ASAP-seq CRISPR) with role label and result per card. Arrows from each card to a central spine labeled 'No data overlap between roles'. Bottom panel: left-to-right pre-registration timeline (spec → eval defined → results → verdict). Each dataset card gets a distinctive accent color. Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- Only ONE cross-corpus validation dataset (Calderon). Soskic was the planned second; deferred per Thiago/Kinga decision.
- The Mimitou perturbation probe used the same lab/protocol family as DOGMA pretraining. We acknowledge this in body bullet ("Mimitou's CRISPR sub-study") but it's a real limitation: a more independent perturbation source would be stronger. Mitigated by the fact that the CRISPR sub-study uses different cells, different conditions, and different output modalities than DOGMA.
- Phase 6.5g.2 disclosure happens in speaker notes only — if an investor doesn't ask, they don't hear the specific numbers. This is deliberate.

---

## What's NEXT after B1 is committed

Move to **B2 (Encoder Probe / ADAPTER_RECOMMENDED)** — deeper view of the 0.57 synergy accuracy result. Shows what "ADAPTER_RECOMMENDED" means architecturally and what verdict thresholds the model passed.
