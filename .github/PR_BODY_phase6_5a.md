## Phase 6.5a — Norman 2019 Data Infrastructure + Gene Alignment

### What

Norman et al. 2019 (Perturb-seq, K562, CRISPR) download + gene-vocabulary
alignment + per-perturbation top-50 DE gene computation + synthetic-fallback
warning added to the linear probe.

**New files**

| File | Purpose |
|------|---------|
| `scripts/download_norman2019.sh` | Idempotent download with SHA-256 verification. Tries scperturb zenodo (record 7041849) first, then pertpy API, then GEO GSE133344 instructions. |
| `scripts/build_norman2019_aligned.py` | Gene-vocabulary alignment to the 36,601-gene PBMC10k encoder space + top-50 DE computation. Writes `data/norman2019_aligned.h5ad` + `.meta.json` + `_top50_de.json`. |

**Modified files**

| File | Change |
|------|--------|
| `scripts/linear_probe_pretrain.py` | (1) LOUD `UserWarning` on synthetic fallback. (2) Uses precomputed `uns["top50_de_per_perturbation"]` indices when present; falls back to variance-based selection with a warning. |
| `.gitignore` | Added explicit entries for Norman 2019 aligned artifacts. |

### Why

Phase 6.5 linear probe FAILED — all three arms silently evaluated on
**synthetic fallback data** instead of real Norman 2019 perturbation data.
The R² values produced were biologically meaningless.

Root cause: `_load_dataset` had no warning when the `.h5ad` was absent, so
the synthetic path executed silently. Additionally, the "top-50 DE genes"
metric was computed from training-set variance on synthetic data — a further
layer of meaninglessness.

This PR fixes the infrastructure so that Phase 6.5b (the actual re-run) can
proceed on real biology.

### Alignment method

1. Load Norman 2019 var_names and PBMC10k var_names.
2. Upper-case both sets (resolves most case-mismatch aliases).
3. Compute intersection. Minimum threshold: 2,000 genes
   (halt with informative error if below — indicates a gene-name format
   mismatch such as Ensembl IDs vs symbols).
4. Build (n_cells × 36,601) matrix: intersection genes → Norman expression;
   non-intersection genes → 0.0 (structural zeros, not imputed).
5. Gene order matches PBMC10k var_names exactly.

### Top-50 DE computation

- Reference: mean expression of `obs["perturbation"] ∈ {"control","ctrl",...}`.
- LFC per gene: log₂(mean_pert + ε) − log₂(mean_ctrl + ε), ε = 1e-6.
- Top-50 ranked by |LFC|, stored per perturbation in
  `uns["top50_de_per_perturbation"]` and `data/norman2019_top50_de.json`.

### Reproducibility

> **Note**: the table below will be populated after the first successful
> download and alignment run. SHA-256 hashes must be verified before
> Phase 6.5b proceeds.

| Field | Value |
|-------|-------|
| Input SHA-256 (Norman h5ad) | `<fill after download>` |
| Output SHA-256 (aligned h5ad) | `<fill after build>` |
| n_cells | `<fill>` |
| n_genes_aligned | 36,601 |
| n_genes_intersection | `<fill>` |
| Source URL | https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad |
| Gene alignment method | HGNC symbol upper-case match |

### Not in scope

Phase 6.5b (re-running the linear probe against the aligned Norman 2019 data)
is a separate PR. This PR only delivers the data infrastructure.

### Validation commands

```bash
# 1. Aligned data exists and has correct shape
python -c "
import anndata as ad
a = ad.read_h5ad('data/norman2019_aligned.h5ad')
assert a.shape[1] == 36601, f'Expected 36601 genes, got {a.shape[1]}'
assert 'perturbation' in a.obs.columns or 'condition' in a.obs.columns
print(f'OK: {a.shape[0]} cells x {a.shape[1]} genes')
print(f'Perturbations: {a.obs.perturbation.nunique() if \"perturbation\" in a.obs else a.obs.condition.nunique()}')
"

# 2. Dry-run probe does NOT trigger synthetic fallback
python scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed 17 --condition pretrained 2>&1 | grep -c "SYNTHETIC FALLBACK"
# Must print 0

# 3. Tests pass
pytest tests/ -q --ignore=tests/test_ckpt_loader.py
```
