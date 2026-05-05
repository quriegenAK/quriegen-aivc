# Cross-corpus single-cell → bulk transfer eval — canonical methodology

**Status**: Canonical specification, locked 2026-05-04.
**Supersedes**: per-cell linear probe on Calderon (used pre-Phase-6.5g.2).
**Lessons learned from**: `docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md`
(see Corrigendum section).

## TL;DR

For evaluating a single-cell-trained encoder's transferability to a
cross-corpus bulk reference (DOGMA → Calderon being the canonical case),
use **pseudo-bulk DOGMA centroid-NN classification on lineage-resolution
labels**, with explicit handling of noise-centroid classes. Do NOT use
per-cell linear probe.

## When to use this methodology

This spec applies whenever:
- The encoder was trained on **single-cell** data (sparse, noisy per-cell profiles)
- The eval target is **bulk** measurements (dense aggregate signal)
- The eval question is "does the encoder produce cell-type-discriminative
  embeddings that generalize across corpora"
- Both source (training) and target (eval) datasets have cell-type labels

The Calderon (n=175 bulk samples × 25 fine cell types) ↔ DOGMA (n~32k
single cells × 8 Azimuth lineages) case is the canonical instance.

## Methodology

### 1. Pseudo-bulk DOGMA centroids

Aggregate the source single-cell counts per cell-type label by summing
peak counts across cells in each class:

```python
for cls in cell_type_classes:
    mask = obs["cell_type"] == cls
    pseudo_bulk[cls] = atac[mask].sum(axis=0)  # one row per class
```

Sum (not mean) because:
- Bulk samples are physical sums of cells; sum semantics matches
- TF-IDF normalization downstream takes care of scale
- Mean would lose count-sum information that drives technical
  normalization in the encoder

### 2. Lineage-resolution labels

Use **broad lineage labels**, not fine subtype labels. For DOGMA-Calderon:
- Use 6 lineages: B, CD4_T, CD8_T, DC, Monocyte, NK
- Map any subtype labels (Calderon's 25 fine classes; DOGMA's Azimuth
  predicted.celltype.l1) → these 6 lineages
- Drop any sample whose label cannot be cleanly mapped (e.g.,
  Gamma_delta_T, Erythrocyte) — do NOT force-assign

Why lineage resolution:
- Single-cell ATAC produces noisy per-cell profiles; subtype
  discrimination requires sample-level statistical power that bulk
  doesn't provide at n=175
- Lineage-level chromatin signatures are robust across corpora; subtype
  signatures may be donor- or protocol-specific
- The pre-registration target should match the resolution at which
  cross-corpus transfer is biologically realistic

### 3. Noise-centroid exclusion

**Critical**: exclude any centroid that represents a noise / catch-all
class from the comparison space.

For DOGMA-Azimuth labels: exclude `other` and `other_T` (Azimuth-
uncertain calls). When pseudo-bulked, these classes produce "generic"
chromatin profiles that absorb cross-corpus signal — they're closer
to most bulk samples than any specific lineage centroid.

Empirical evidence (from Phase 6.5g.2 Test 1 → Test 1.5):
- WITH `other`/`other_T` centroids:    accuracy 0.3308
- WITHOUT (excluded):                    accuracy 0.7308

The 40-point swing is the noise-distractor effect alone. Do not
underestimate this.

General rule: any centroid built from <50 cells, or labeled as a
catch-all, or with uncertainty-quality < 0.6 should be excluded
from the centroid set.

### 4. Encoding

Encode both centroids and bulk samples through the same encoder, in
its evaluation forward path:

```python
z_dogma = encoder(pseudobulk_centroids)   # (n_centroids, latent_dim)
z_bulk  = encoder(bulk_samples)           # (n_bulk, latent_dim)
```

For trimodal encoders (DOGMA SupCon-trained), use the bare ATAC encoder
forward path — NOT the joint-fusion path. Empirical evidence from
Pivot A: zero-padded RNA + Protein dilute the ATAC signal at fusion
time (joint kfold 0.0629 < bare ATAC 0.1943 < centroid-NN 0.7308).

### 5. Top-1 NN classification + scoring

For each bulk sample, find the centroid with maximum cosine similarity:

```python
sim = z_bulk @ z_dogma.T / (||z_bulk|| * ||z_dogma||)  # (n_bulk, n_centroids)
pred_lineage = centroid_labels[argmax(sim, axis=1)]
accuracy = (pred_lineage == bulk_lineage_labels).mean()
```

Report:
- Overall accuracy
- Per-lineage accuracy (n per lineage matters — flag any with n < 5)
- Confusion matrix
- Chance baseline = 1 / n_centroids

For pre-registration, define both:
- **Stratified k-fold** (typically k=5; primary metric for in-distribution
  rigor)
- **Leave-one-donor-out** (donor-generalization metric; secondary)

### 6. Pre-registration template

Any future phase that pre-registers a cross-corpus eval should specify
**all four** dimensions explicitly:

```yaml
eval_pre_registration:
  metric: top1_nn_accuracy_kfold
  threshold: 0.70
  methodology:
    type: centroid_nn       # NOT per_cell_linear_probe
    similarity: cosine
  label_resolution:
    level: lineage_broad    # NOT subtype_fine
    n_classes: 6            # B, CD4_T, CD8_T, DC, Monocyte, NK
    mapping_table_ref: docs/eval_methodology/calderon_25_to_6_lineage_map.md
  noise_handling:
    exclude_centroid_classes: [other, other_T]
    exclude_centroid_when:
      - n_cells_lt: 50
      - confidence_lt: 0.6
  cv_strategies:
    - stratified_kfold(folds=5)
    - leave_one_donor_out
```

## Known limitation — B-lineage depression under cross-corpus pseudo-bulk

**Symptom**: Calderon B-lineage accuracy comes in at 4/22 = 0.18 even
when the encoder achieves overall 0.7308 on the other 5 lineages.

**Diagnosis (Stage 3 Part 1 Report 5, 2026-05-05)**: B-lineage accuracy
is depressed under cross-corpus pseudo-bulk eval due to **subtype-coverage
mismatch**; cell-level discrimination is intact (silhouette 0.354 for B
vs 0.129 for CD4_T — B clusters MORE tightly than CD4_T at the
single-cell level).

**Mechanism**: Calderon's "B" label = bulk-sorted naive/memory B from
healthy donors. DOGMA's "B" centroid = CD3/CD28-stimulated PBMC B cells,
likely plasmablast-leaning with elevated antigen-presentation markers
(HLA-DR, CD86). The two B definitions encode to different latent regions,
shifting B → DC neighborhood (13 of 18 misclassified cells go to DC,
5 to NK, ZERO to T-cell centroids — lineage hierarchy is respected).

**Implication for THIS spec**: B-lineage accuracy under cross-corpus
pseudo-bulk eval is a known artifact and SHOULD NOT be re-litigated as
an encoder weakness without explicit single-cell-level diagnostic data.
When reporting per-lineage accuracy, footnote B's value with this caveat.

**Remediation**: Do NOT retrain the encoder. For B-cell-disease contexts
(CLL, BCR signaling perturbations), the perturbation predictor's output
head should include B-vs-DC-aware fine-grained disambiguation — this is
decoder-level work, not encoder-level. See memory entry
`project_aivc_bcell_diagnosis_2026_05_05.md` for full diagnostic
breakdown (D1 + D2 + D3 numerics).

## Implementation reference

The canonical implementation lives in:
- `scripts/eval_pseudobulk_compatibility.py` (CLI script)
- `scripts/submit_pseudobulk_eval.slurm` (SLURM submit; takes
  optional 2nd arg for `--exclude_centroid_classes`)
- `aivc/eval/calderon_probe.py::project_calderon_to_dogma_space` and
  `encode_samples` (shared building blocks)

Usage:
```bash
sbatch scripts/submit_pseudobulk_eval.slurm \
    /path/to/encoder_ckpt.pt \
    "other,other_T"          # exclude noise centroids
```

## Why per-cell linear probe was wrong

The originally pre-registered eval was a per-cell logistic regression
linear probe on Calderon's 175 samples × 25 fine classes, with
stratified k-fold CV. This was wrong for three reasons:

1. **Sparsity mismatch**: Calderon is bulk; treating each sample as a
   "cell" with a fine-grained label and doing per-sample classification
   is statistically underpowered at n=175. Bulk samples are aggregates;
   they should be compared to aggregates (centroids), not classified
   individually.

2. **Resolution mismatch**: 25-class fine subtype is harder than
   6-lineage broad classification, and the encoder was trained at
   roughly lineage resolution (Azimuth predicted.celltype.l1). Asking
   it to discriminate naive vs memory CD4 across corpora is a
   different research question.

3. **No noise handling**: the linear probe trains on whatever labels
   are in the obs column, including noise classes. There's no
   mechanism to exclude `other`/`other_T` from the classification
   space. Only centroid-based methods naturally support this.

The empirical gap between methods on the same encoder is enormous:
- Per-cell linear probe (25 classes, no noise handling):  **0.1943**
- Centroid-NN (6 lineages, noise centroids excluded):     **0.7308**

This 50-point gap is entirely a methodology difference, not a model
difference. The encoder is the same.

## What this spec is NOT

- Not a replacement for in-distribution (single-cell ↔ single-cell)
  evaluation. For DOGMA-internal cell-type accuracy, use Azimuth label
  transfer or kBET.
- Not a per-cell evaluation. If the question is "for each cell in a
  query dataset, predict its label," use a different methodology
  (e.g., scANVI label transfer, k-NN on z_joint).
- Not pre-registration discipline waiver. The original Calderon FAIL
  verdict at 0.1943 STANDS as the pre-registered result. This document
  defines what FUTURE phases should pre-register; it does not
  retroactively flip past verdicts.

## See also

- `docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md` (closure
  + corrigendum with empirical evidence chain)
- Memory: `project_aivc_phase65g2_test1_5_remediation.md` (dual
  conclusion + methodology lessons)
- Memory: `feedback_premature_verdict_on_contaminated_run.md` (related
  pre-registration discipline lesson)
