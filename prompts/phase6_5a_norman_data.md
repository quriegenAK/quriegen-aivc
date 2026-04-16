# Phase 6.5a — Norman 2019 Data Infrastructure

## CONTEXT

Phase 6.5 linear probe FAILED because all three arms ran against
synthetic fallback data, not real Norman 2019 perturbation data.
The evaluation is biologically meaningless.

The real-data pretrained checkpoint exists:
  checkpoints/pretrain/pretrain_encoders.pt
  SHA: 416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e
  n_genes=36601 (PBMC10k gene vocabulary)

Norman 2019 (Norman et al., Science 2019) is a Perturb-seq dataset
of K562 cells with CRISPR perturbations. It has ~5,000-8,000
measured genes in a DIFFERENT gene vocabulary than PBMC10k.

The encoder expects 36,601-dimensional input. Norman 2019 must be
aligned to this vocabulary before the linear probe can run.

Branch: phase-6.5a off latest main.

## TASK

### Step 1 — Download Norman 2019

Create `scripts/download_norman2019.sh` with SHA-256 verification
(same pattern as download_pbmc10k_multiome.sh).

Source options (try in order):
1. scperturb: https://zenodo.org/records/7041849 — look for
   Norman 2019 .h5ad in the scperturb collection
2. pertpy datasets API: `pertpy.data.norman_2019()` — if pertpy
   is installable
3. Original GEO deposit: GSE133344

Whichever source works, the download script must:
- Download to data/raw/norman2019/
- Verify SHA-256 of downloaded file
- Be idempotent (skip if file exists and hash matches)

If the file is an .h5ad, read it with anndata and report:
- n_cells, n_genes
- obs columns (especially perturbation/condition labels)
- var columns (especially gene names/symbols)
- Whether .X is raw counts or normalized

STOP after download and report the above before proceeding
to Step 2. The gene alignment strategy depends on what's in
the file.

### Step 2 — Gene vocabulary alignment

Create `scripts/build_norman2019_aligned.py` that:

a) Reads the downloaded Norman 2019 .h5ad
b) Reads the PBMC10k gene vocabulary from
   data/pbmc10k_multiome.h5ad (var_names or var.index)
c) Computes the gene name intersection. Report:
   - Norman genes: N
   - PBMC10k genes: 36,601
   - Intersection: M
   - If M < 2,000: STOP and report — insufficient overlap
     for meaningful evaluation.
d) Creates an aligned AnnData:
   - Shape: (norman_n_cells, 36,601)
   - For intersection genes: copy Norman expression values
   - For non-intersection genes: fill with 0.0
   - Gene order matches PBMC10k var_names exactly
   - Preserve obs["perturbation"] (or obs["condition"] or
     equivalent perturbation label column)
   - Add obs["n_genes_detected_norman"] = number of nonzero
     genes in the original Norman data per cell (QC metric)
e) Saves to `data/norman2019_aligned.h5ad` (gitignored)
f) Writes a sibling `.meta.json` with:
   - input_sha256
   - output_sha256
   - n_cells, n_genes_aligned (36601), n_genes_intersection
   - gene_intersection list (or path to a .txt)
   - source URL
   - timestamp_utc

Gene name matching: use HGNC symbols. Norman 2019 likely uses
gene symbols directly. PBMC10k var_names should also be symbols
(from Cell Ranger). If there are case mismatches or alias issues,
normalize to uppercase first, then try a fuzzy match on common
aliases. Document any unmatched genes.

### Step 3 — Compute top-50 DE genes

Add to `scripts/build_norman2019_aligned.py` (or a separate
script `scripts/compute_norman_de.py`):

a) For each unique perturbation in obs["perturbation"]:
   - Compute mean expression in perturbed cells vs control
   - Rank genes by absolute log-fold-change
   - Select top-50 DE genes (by absolute LFC)
b) Save a mapping: perturbation → list of top-50 DE gene indices
   (indices into the 36,601-dim aligned vocabulary)
c) Store in the .h5ad as uns["top50_de_per_perturbation"]
   or as a separate JSON at data/norman2019_top50_de.json

This replaces the current approach where top-50 DE is selected
by training-set variance (which is meaningless for perturbation
evaluation).

### Step 4 — Update linear_probe_pretrain.py

Modify `scripts/linear_probe_pretrain.py` _load_dataset() to:

a) When --dataset_path points to a real .h5ad:
   - Load with anndata
   - Extract X (expression matrix) and perturbation labels
   - If uns["top50_de_per_perturbation"] exists, use it
   - Set Y = X (self-reconstruction targets) OR set up proper
     perturbation-response targets if the data supports it

b) Add a LOUD warning when synthetic fallback triggers:
   ```
   warnings.warn(
       f"SYNTHETIC FALLBACK: {name} data not found at {path}. "
       f"Evaluation results are NOT biologically meaningful.",
       UserWarning, stacklevel=2
   )
   ```
   This prevents silent fallback from ever happening again.

c) When using real data, override the top-50 DE gene selection:
   - Instead of selecting by training-set variance, use the
     precomputed top-50 DE indices from the .h5ad metadata
   - Fall back to variance-based selection only if the metadata
     is absent (with a warning)

### Step 5 — Verify data pipeline end-to-end

Run (no W&B, dry run):

```bash
python scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed 17 \
    --condition pretrained
```

Must:
- NOT trigger synthetic fallback (no warning printed)
- Load 36,601-dim data matching encoder n_genes
- Print R² metrics (sign doesn't matter yet — just no crash)
- Log provenance as "norman2019" not "norman2019:synthetic"

### Step 6 — Tests

- pytest tests/ -q --ignore=tests/test_ckpt_loader.py → all pass
- The synthetic fallback warning must NOT fire during normal
  test execution (tests that intentionally use synthetic should
  suppress or expect the warning)

## CONSTRAINTS

- No edits to: aivc/training/*, aivc/skills/*, aivc/data/multiome_loader.py,
  train_week3.py, scripts/pretrain_multiome.py, scripts/harmonize_peaks.py
- Allowed edits:
    scripts/linear_probe_pretrain.py (Step 4 only)
    scripts/exp1_linear_probe_ablation.py (only if --dataset_path
      arg needs minor fixes to pass through correctly)
- New files:
    scripts/download_norman2019.sh
    scripts/build_norman2019_aligned.py
    data/norman2019_aligned.h5ad (gitignored)
    data/norman2019_top50_de.json (gitignored)
    .github/PR_BODY_phase6_5a.md

## VALIDATION

Before PR opens:

```bash
# Aligned data exists and has correct shape
python -c "
import anndata as ad
a = ad.read_h5ad('data/norman2019_aligned.h5ad')
assert a.shape[1] == 36601, f'Expected 36601 genes, got {a.shape[1]}'
assert 'perturbation' in a.obs.columns or 'condition' in a.obs.columns
print(f'OK: {a.shape[0]} cells x {a.shape[1]} genes')
print(f'Perturbations: {a.obs.perturbation.nunique() if \"perturbation\" in a.obs else a.obs.condition.nunique()}')
"

# Dry-run probe does not trigger synthetic fallback
python scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed 17 --condition pretrained 2>&1 | grep -c "SYNTHETIC FALLBACK"
# Must print 0

# Tests pass
pytest tests/ -q --ignore=tests/test_ckpt_loader.py
```

## FAILURE HANDLING

- If Norman 2019 download fails from all three sources, STOP.
  Report which sources were tried and what failed.
- If gene intersection < 2,000, STOP. Report the overlap count
  and the gene name format mismatch details.
- If the aligned .h5ad has n_genes != 36,601, STOP. The alignment
  script has a bug.
- If the dry-run probe crashes on dimension mismatch, STOP.
  The alignment or the loader has a bug.

## PR PREPARATION

Write .github/PR_BODY_phase6_5a.md with:
- What: Norman 2019 download + gene alignment + DE computation +
  synthetic fallback warning
- Why: Phase 6.5 FAIL was caused by synthetic data, not biology
- Reproducibility: input SHA, output SHA, n_cells, n_genes_intersection,
  source URL, gene alignment method
- Not in scope: Phase 6.5 re-run (that's PR B)

Open PR:
  Title: "Phase 6.5a: Norman 2019 data infrastructure + gene alignment"
  Body: .github/PR_BODY_phase6_5a.md

Do not merge — human review required.
