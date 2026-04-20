# Phase 6.5c — Linear Probe Evaluation Contract (LOCKED v2)

> **STATUS:** Metric contract locked at LOCKED v2 after the v1 y_sd
> assertion tripped on the real Norman 2019 matrix (seed=3, pretrained).
> Root cause: the structural-zero filter in ``_load_dataset`` is
> dataset-global, but the y_sd > 1e-6 invariant was train-local — columns
> nonzero in train+test but zero in the train split passed the global
> filter and violated the train-local floor. v2 replaces the y_sd
> assertion with a **per-run train-variance mask** applied symmetrically
> to Ridge fit and scoring, and adds a post-hoc **intersection gate**
> across all 9 runs for the primary decision signal.
> Per-perturbation restructure is explicitly deferred to Phase 6.5d.

## CONTEXT

Prior 6.5c attempts failed at the tripwire (|r²| < 2.0):

- **6.5c v1 (commit `8461e0d`, dropped):** `X = log1p(X)` + `Y = log1p(Y)` +
  train-expressed-gene filter. Run 1 `r²_top50_de = −6.78 × 10¹²`. Mis-diagnosed
  as dead latent dims; diagnostic showed latents healthy (128/128 live).
- **6.5c v2 (not committed):** Removed `log1p(X)`. Diagnostic reproduction
  showed the same failure shape across raw/log1p/both; median per-col
  r² ≈ +0.01 across all three conditions; worst-10 columns all had
  train_nonzero_count ∈ {1, 2, 3} with y_sd ≈ 6e-3.

Locked diagnosis: the pathology is exclusively in the summary metric.
Pipeline arithmetic is healthy (|coef|.max ≈ 5.7, |Y_hat|.max ≈ 15.8,
z_sd.min ≈ 0.9 on raw X). `multioutput='uniform_average'` on standardized
Y is dominated by the ~150-column tail of near-constant genes. The fix is
a metric change, not a data/architecture change.

Branch: `phase-6.5c` branched off `phase-6.5b` (a661d44, docs-only).
The local `phase-6.5c` branch has been reset clean — no prior commits.

## PREREQUISITES

- `phase-6.5b` at `a661d44` on origin (docs-only diagnosis).
- `data/norman2019_aligned.h5ad` present (output_sha256 `d4bedb53...`).
- `checkpoints/pretrain/pretrain_encoders.pt` — SHA `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`.
- `checkpoints/pretrain/pretrain_encoders_mock.pt` — SHA `c0d9715dbc76a6ecab260fe09ca5173ee7fdf6eb640538eac0f9024399a90b4e`.
- `wandb login` authenticated. `WANDB_PROJECT=aivc-linear-probe`.
- `scripts/diag_probe_latent_stats.py` and `scripts/diag_probe_reproduction.py`
  should be DELETED from working tree before Commit 1 (not committed).

## THE CONTRACT (LOCKED v2)

### Spaces

- **X (encoder input):** raw integer counts. Pretrained encoder distribution.
  Never apply log1p to X.
- **Y (Ridge fit target):** `log1p(X_filtered)` where `X_filtered` applies the
  existing dataset-global structural-zero filter from `_load_dataset`
  (unchanged).
- **Per-run filter (Option B):** after computing the train/test split,
  derive `mask_tr = Y_log1p[tr].std(0) > 1e-7`. This mask is
  **train-local** (invariant matches the Ridge fit invariant) and
  applied **symmetrically** to fit and scoring. This replaces the v1
  y_sd assertion.
- **Ridge fit space (internal):** Z and Y (masked to kept columns)
  standardized with train-split statistics for SVD conditioning only.
- **Evaluation space:** un-standardized log1p Y on the masked column
  subset: `Y_hat_log1p_te_k = Y_hat_std_te * y_sd + y_mu`, compared
  against `Y_log1p_te_k = Y_log1p[te, mask_tr]`.

### Metrics

**Per-run (seed-level diagnostic, logged to W&B):**

- `r2_top50_de_vw` — variance_weighted R² on DE genes that survive the
  per-run mask (remapped from original gene space to masked space).
- `r2_overall_vw` — variance_weighted R² on all kept columns.
- `n_kept`, `n_dropped`, `pct_dropped` — mask statistics.
- `n_de_total`, `n_de_kept` — DE gene counts (pre-mask / post-mask).

**Intersection gate (primary decision signal, computed post-hoc):**

- `mask_intersection = AND(mask_tr_i for i in all 9 runs)`.
- `G_top50_de_intersection` — variance_weighted R² on DE genes inside
  `mask_intersection`, averaged across the 3 seeds per arm.
- `G_overall_intersection` — variance_weighted R² on all intersection
  columns, averaged across seeds per arm.

### Filtering

- Structural-zero filter from `_load_dataset`: KEEP (unchanged).
- Per-run train-variance mask `> 1e-7`: KEEP (LOCKED v2 addition).
- `MIN_TRAIN_NONZERO` train-expressed filter: **DO NOT APPLY** (REMOVED).

### Seeds

Three seeds per arm: `{3, 17, 42}`. 9 runs total across `{real, mock, random}`.

### Assertions (runtime tripwires)

- `|r2_top50_de_vw| < 2.0` — contract violation tripwire.
- `z_sd.min() > 1e-6` — encoder-collapse tripwire (kept; real pathology).
- `n_kept > 0` — degenerate-dataset tripwire.
- `n_de_kept >= 5` — DE-survival tripwire.
- **REMOVED:** `y_sd.min() > 1e-6` (replaced by `mask_tr`).
- **REMOVED:** `MIN_TRAIN_NONZERO`.

Any trip halts the run without writing results docs.

### Gate decision

Primary gate statistic is `G_top50_de_intersection` on the pretrained
(real) arm:

- `G_real = G_top50_de_intersection(arm=real)`
- `G_random = G_top50_de_intersection(arm=random)`
- `G_mock = G_top50_de_intersection(arm=mock)`
- `Delta_real_vs_random = G_real − G_random` (primary gate number)
- `Delta_real_vs_mock   = G_real − G_mock`

Decision:

- `|G_random| > 0.01`: **PASS** if `Delta_real_vs_random ≥ 0.05 · |G_random|`,
  else **FAIL** or **SOFT**.
- `|G_random| ≤ 0.01`: **PASS** if `Delta_real_vs_random ≥ 0.005` absolute,
  else **FAIL** or **SOFT**.
- **SOFT** = `0 < Delta_real_vs_random < threshold` → requires ≥5-seed
  confirmation before Phase 7.
- **FAIL** = `Delta_real_vs_random ≤ 0` → Phase 7 blocked; hypotheses
  (a)–(d) documented.

## TASK

### Step 1 — Fix `scripts/linear_probe_pretrain.py`

Apply exactly these edits inside `run_condition`. No other code changes
in this file.

**After `_load_dataset` returns:**

```python
X, Y, _pert, provenance, de_indices = _load_dataset(...)
n_genes = X.shape[1]

# CONTRACT: X stays raw (encoder was trained on raw counts).
# Y target moves to log1p space — variance-stabilizing transform for
# the Ridge target. Y_hat will be un-standardized back to this space
# for metric computation. NO train-expressed filter (variance_weighted
# handles rare-column robustness without a heuristic cutoff).
Y_log1p = np.log1p(Y).astype(np.float32, copy=False)
```

**Replace the existing standardization + Ridge + metric block with:**

```python
# Train/test split
rng = np.random.default_rng(seed)
idx = rng.permutation(X.shape[0])
split = int(0.8 * X.shape[0])
tr, te = idx[:split], idx[split:]

# Encoder forward pass (X raw)
Z_tr = _extract_latents(copy.deepcopy(encoder), X[tr])
Z_te = _extract_latents(copy.deepcopy(encoder), X[te])

# Train-split statistics for BOTH standardizations
z_mu, z_sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8
y_mu, y_sd = Y_log1p[tr].mean(0), Y_log1p[tr].std(0) + 1e-8

# Locked-contract tripwires — fire BEFORE Ridge fit to save compute
assert float(z_sd.min()) > 1e-6, (
    f"z_sd floor violated: min={float(z_sd.min()):.3e}. "
    f"Encoder produced collapsed latent dims — distribution mismatch "
    f"with pretrain. X must stay raw counts."
)
assert float(y_sd.min()) > 1e-6, (
    f"y_sd floor violated on log1p-space targets: "
    f"min={float(y_sd.min()):.3e}. Unexpected — variance-weighted "
    f"should tolerate any y_sd, but this indicates a degenerate dataset."
)

# Standardize for Ridge conditioning only
Z_tr_std = (Z_tr - z_mu) / z_sd
Z_te_std = (Z_te - z_mu) / z_sd
Y_tr_std = (Y_log1p[tr] - y_mu) / y_sd

# Ridge fit
from sklearn.linear_model import Ridge
t0 = time.time()
model = Ridge(alpha=1.0, solver='svd').fit(Z_tr_std, Y_tr_std)
fit_seconds = time.time() - t0

# Predict in standardized space, then UN-STANDARDIZE to log1p space
Y_hat_std_te = model.predict(Z_te_std)
Y_hat_log1p_te = Y_hat_std_te * y_sd + y_mu
Y_log1p_te = Y_log1p[te]

# Metric computation on UN-STANDARDIZED log1p Y
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

if de_indices is not None:
    de_idx = np.array(
        [i for i in de_indices if i < Y_log1p_te.shape[1]], dtype=np.int64
    )
    if de_idx.size < 5:
        raise RuntimeError(
            f"Too few valid DE indices ({de_idx.size}) — dataset misconfigured."
        )
else:
    # Variance-based fallback — top genes by train variance
    var_tr = Y_log1p[tr].var(0)
    de_idx = np.argsort(-var_tr)[: min(3217, len(var_tr))]

Y_log1p_te_de = Y_log1p_te[:, de_idx]
Y_hat_log1p_te_de = Y_hat_log1p_te[:, de_idx]

# Primary gate metric
r2_top50_de_vw = float(
    r2_score(Y_log1p_te_de, Y_hat_log1p_te_de,
             multioutput='variance_weighted')
)

# Secondary diagnostics
r2_overall_vw = float(
    r2_score(Y_log1p_te, Y_hat_log1p_te, multioutput='variance_weighted')
)
r2_per_col_de = np.array(
    [r2_score(Y_log1p_te_de[:, j], Y_hat_log1p_te_de[:, j])
     for j in range(de_idx.size)]
)
r2_top50_de_median = float(np.median(r2_per_col_de))
r2_top50_de_p01 = float(np.percentile(r2_per_col_de, 1))
r2_top50_de_p99 = float(np.percentile(r2_per_col_de, 99))

# Pearson overall (for audit; used in prior phases)
yf = Y_log1p_te.ravel()
yh = Y_hat_log1p_te.ravel()
pearson_overall = float(pearsonr(yf, yh)[0])

# Final tripwire
assert abs(r2_top50_de_vw) < 2.0, (
    f"r2_top50_de_vw={r2_top50_de_vw:.3e} violates |r²|<2 cap. "
    f"Metric contract violated. Do not write results."
)

metrics = {
    "r2_top50_de_vw": r2_top50_de_vw,
    "r2_overall_vw": r2_overall_vw,
    "r2_top50_de_median": r2_top50_de_median,
    "r2_top50_de_p01": r2_top50_de_p01,
    "r2_top50_de_p99": r2_top50_de_p99,
    "pearson_overall": pearson_overall,
    "probe_fit_seconds": fit_seconds,
    "n_de_genes_evaluated": int(de_idx.size),
    "z_sd_min": float(z_sd.min()),
    "y_sd_min": float(y_sd.min()),
}
metrics.update(
    condition=condition, dataset=dataset_name, provenance=provenance,
    n_cells_train=int(tr.size), n_cells_test=int(te.size),
    n_genes=int(n_genes), hidden_dim=int(effective_hidden),
    latent_dim=int(effective_latent), seed=int(seed),
)
return metrics
```

Remove the entire old standardization + `_fit_probe_and_score` call that
this replaces. The helper `_fit_probe_and_score` can stay in the file
(used by other code paths); we just don't call it from `run_condition`.

### Step 2 — Update regression test

Rewrite `tests/test_linear_probe_numeric_stability.py` to exercise the
LOCKED contract. Delete any references to `MIN_TRAIN_NONZERO` or
`uniform_average`. The test must:

1. Build a synthetic fixture with `n_cells=512`, `n_genes=128`, Poisson(λ=0.3)
   counts, seed=17. Force 3 columns to have only a single nonzero train
   cell (insert a count of 1 at a train index).
2. Call `run_condition(condition='scratch', ...)` with
   `n_genes_fallback=128`, `hidden_dim=32`, `latent_dim=16`.
3. Assert:
   - `np.isfinite(m["r2_top50_de_vw"])`
   - `-1.5 <= m["r2_top50_de_vw"] <= 1.01`
   - `np.isfinite(m["r2_overall_vw"])`
   - `np.isfinite(m["r2_top50_de_median"])`
   - `m["z_sd_min"] > 1e-6`
   - `m["y_sd_min"] > 1e-6`
4. No `RuntimeWarning` from numpy during the call.

### Step 3 — Run pytest

```bash
pytest tests/test_linear_probe_numeric_stability.py -q
pytest tests/test_no_bare_torch_load.py -q
pytest -q   # full suite, green except pre-existing Issue #7 and #11
```

### Step 4 — Commit 1

```
Phase 6.5c: variance_weighted R² on un-standardized log1p Y (locked contract)
```

### Step 5 — Multi-seed three-arm execution

For each `seed ∈ {3, 17, 42}` run three commands (nine total):

```bash
# Arm 1 — real ckpt
python3 scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed $SEED --condition pretrained --wandb

# Arm 2 — mock ckpt
python3 scripts/exp1_linear_probe_ablation.py \
    --ckpt_path checkpoints/pretrain/pretrain_encoders_mock.pt \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed $SEED --condition pretrained --wandb

# Arm 3 — random init
python3 scripts/exp1_linear_probe_ablation.py \
    --random_init --n_genes 36601 \
    --dataset_name norman2019 \
    --dataset_path data/norman2019_aligned.h5ad \
    --seed $SEED --wandb
```

Capture all 9 W&B URLs. Expected wall-clock per run: ~30–45 min on MPS
(no log1p(X) memory doubling). Total: ~5–7 hours.

### Step 6 — Tripwires (all 9 runs)

- `ckpt_sha256` tripwire per arm (real / mock / n/a).
- Zero `SYNTHETIC FALLBACK` warnings.
- `|r2_top50_de_vw| < 2.0` on every run.
- `y_sd_min > 1e-6` and `z_sd_min > 1e-6` on every run.

Any trip → STOP. Do not write docs. Do not commit docs.

### Step 7 — Compute gate

```
median_pretrained = median(r2_top50_de_vw for seed in {3,17,42}, arm=real)
median_mock       = median(r2_top50_de_vw for seed in {3,17,42}, arm=mock)
median_random     = median(r2_top50_de_vw for seed in {3,17,42}, arm=random)

G_real_vs_random = median_pretrained - median_random
G_real_vs_mock   = median_pretrained - median_mock
```

Apply gate decision per CONTRACT.

### Step 8 — Overwrite `.github/PHASE6_5_RESULTS.md`

Keep existing "Prior result (synthetic — invalidated)" and "Phase 6.5b
attempt — BLOCKED" sections as audit history. Append:

```markdown
## Phase 6.5c attempt (real Norman 2019 — LOCKED CONTRACT)

### Contract

Metric space: `log1p(X_filtered)` un-standardized (Ridge fit uses
standardized Y internally; metric un-does it).
Aggregation: `sklearn.r2_score(..., multioutput='variance_weighted')`.
Encoder input: raw integer counts (pretrain distribution preserved).
Filtering: structural-zero only. No MIN_TRAIN_NONZERO.
Seeds: {3, 17, 42}.

Rationale: uniform_average R² on standardized Y is dominated by
near-constant columns with train/test-mismatch-inflated standardized
variance. variance_weighted on un-standardized log1p Y weights each
gene by its biological variance, automatically suppressing
near-constant gene contribution. See commits 8461e0d (dropped),
diagnostic scripts `diag_probe_latent_stats.py` /
`diag_probe_reproduction.py` (not committed).

### Setup

- Dataset: Norman 2019 aligned (111,445 cells × 36,601 genes)
- Structural-zero filter: <value> genes retained for Y
- DE genes evaluated: <value> (union top-50 per perturbation,
  remapped through structural-zero filter)
- n_cells_train / n_cells_test: 89,156 / 22,289
- Probe: Ridge(alpha=1.0, solver='svd'), 128-dim latent → log1p-space Y
- Per-run wall-clock: <median> min

### Multi-seed results

| Seed | Arm        | ckpt SHA (8) | r²_top50_de_vw | r²_overall_vw | r²_top50_de_median | W&B URL |
|------|------------|--------------|----------------|---------------|--------------------|---------|
| 3    | real       | 416e8b1a     | <v>            | <v>           | <v>                | <url>   |
| 3    | mock       | c0d9715d     | <v>            | <v>           | <v>                | <url>   |
| 3    | random     | n/a          | <v>            | <v>           | <v>                | <url>   |
| 17   | real       | 416e8b1a     | <v>            | <v>           | <v>                | <url>   |
| 17   | mock       | c0d9715d     | <v>            | <v>           | <v>                | <url>   |
| 17   | random     | n/a          | <v>            | <v>           | <v>                | <url>   |
| 42   | real       | 416e8b1a     | <v>            | <v>           | <v>                | <url>   |
| 42   | mock       | c0d9715d     | <v>            | <v>           | <v>                | <url>   |
| 42   | random     | n/a          | <v>            | <v>           | <v>                | <url>   |

### Gate statistic

- `median_real   = <v>` (real)
- `median_mock   = <v>` (mock)
- `median_random = <v>` (random)
- `G_real_vs_random = median_real - median_random = <v>` (rel <pct>)
- `G_real_vs_mock   = median_real - median_mock   = <v>`

### Decision

**<PASS | SOFT | FAIL>**, dated YYYY-MM-DD.

Justification (2-3 lines).

If FAIL or SOFT: hypotheses (a)–(d) per CONTRACT spec.

### Residual uncertainty

variance_weighted on pooled DE genes does not distinguish
perturbation-specific signal from cross-perturbation shared variance.
Per-perturbation evaluation deferred to Phase 6.5d regardless of
this gate outcome.
```

### Step 9 — Update `.github/REAL_DATA_BLOCKERS.md`

Append under "Phase 6.5 results (real-data gate)":

```markdown
### Phase 6.5c run (real Norman 2019 — LOCKED CONTRACT)

- Date: YYYY-MM-DD
- Real ckpt SHA: 416e8b1a... ✓
- Seeds: {3, 17, 42}
- Metric: r²_top50_de_vw on un-standardized log1p Y (variance_weighted)
- Gate statistic G_real_vs_random: <v> (<pct>)
- Decision: <PASS | SOFT | FAIL>
- 9 W&B URLs: <...>

Phase 7 status: <UNBLOCKED | NEEDS MULTI-SEED (>=5) | BLOCKED>
Phase 6.5d status: QUEUED — per-perturbation evaluation restructure.
```

### Step 10 — Write `.github/PR_BODY_phase6_5c.md`

Include:

- Summary with decision + contract change flag
- Reference to 6.5b diagnosis + 6.5c v1 failure (dropped commit `8461e0d`)
- The locked contract (metric space, aggregation, filter, seeds)
- Multi-seed results table
- Gate statistic G + decision
- Known residual failure mode (pooled DE, deferred to 6.5d)
- Tripwires passed checklist
- Files changed

### Step 11 — Commit 2

```
Phase 6.5c: real-data rerun (9 runs, 3 seeds × 3 arms) + gate decision
```

### Step 12 — Intersection gate post-processing

After all 9 runs complete successfully:

```bash
python3 scripts/compute_intersection_gate.py
```

This reads ``experiments/phase6_5c/artifacts/run_*.npz``, AND-intersects
the per-run train-variance masks, recomputes variance_weighted R² over
the shared column subset (resolves the per-run / train-local invariant
mismatch that sank LOCKED v1), and writes
``experiments/phase6_5c/intersection_gate.json``. The aggregated
``G_top50_de_intersection`` per arm is the primary gate statistic.

### Step 13 — Push `phase-6.5c`

```bash
git push -u origin phase-6.5c
```

### Step 14 — STOP

Do NOT run `gh pr create`. Human review of the 9 W&B runs, tripwire
outputs, and gate statistic required before PR opens.

## CONSTRAINTS

Allowed edits:

- `scripts/linear_probe_pretrain.py` (Step 1 only — the block replacement above)
- `tests/test_linear_probe_numeric_stability.py` (Step 2 — full rewrite)
- `.github/PHASE6_5_RESULTS.md` (append 6.5c section)
- `.github/REAL_DATA_BLOCKERS.md` (append 6.5c subsection)
- `.github/PR_BODY_phase6_5c.md` (new file)

Forbidden edits:

- `aivc/training/*`, `aivc/skills/*`, `aivc/data/*`
- `aivc/training/ckpt_loader.py`
- `scripts/pretrain_multiome.py`
- `scripts/harmonize_peaks.py`
- `scripts/build_norman2019_aligned.py`
- `scripts/exp1_linear_probe_ablation.py`
- `_fit_probe_and_score` helper (leave in place, just don't call from `run_condition`)

No regeneration of checkpoints. No change to checkpoint SHAs. No
`torch.load` — route through `ckpt_loader`.

Delete diagnostic scripts (`diag_probe_latent_stats.py`,
`diag_probe_reproduction.py`) before Commit 1. They must not enter git.

## VALIDATION

- `pytest tests/test_linear_probe_numeric_stability.py -q` passes.
- `pytest tests/test_no_bare_torch_load.py -q` passes.
- Full suite green except pre-existing Issue #7 and #11.
- 9 W&B runs logged with distinct (seed, condition) values.
- Every run: `|r²_top50_de_vw| < 2.0`, `z_sd_min > 1e-6`, `y_sd_min > 1e-6`.
- Zero `SYNTHETIC FALLBACK` warnings.
- `PHASE6_5_RESULTS.md` has no `<value>` or `<url>` placeholders.
- Decision is arithmetically consistent with the median-of-seeds table.

## FAILURE HANDLING

- Any tripwire fires → STOP. Do not write docs. Do not commit docs.
  Paste failing assertion + run metadata, await human decision.
- `|r²| ≥ 2.0` on any run after Commit 1 → contract violated. This
  should be impossible under variance_weighted on un-standardized Y;
  if it happens, there is a new unknown failure mode. STOP.
- `y_sd_min < 1e-6` on real Norman 2019 → unexpected; log1p variance
  of any gene with ≥1 train-nonzero cell should exceed this floor.
  STOP and report.
- `z_sd_min < 1e-6` → encoder producing collapsed latent dims.
  Distribution mismatch or encoder pathology. STOP.
- Any run crashes → STOP. Report full traceback. Do not retry blindly.
- Decision = FAIL → write docs with FAIL, hypotheses (a)–(d), and
  Phase 6.5d scoping. Do NOT loop. FAIL is a valid scientific outcome.
- Decision = SOFT → write docs with SOFT. Schedule 5-seed extension
  as a follow-up run (not in this PR). Phase 7 blocked until extension.

## PR PREPARATION

- Branch: `phase-6.5c` off `phase-6.5b` at `a661d44` (already reset clean).
- Commit 1: Step 4, code + test only.
- Commit 2: Step 11, docs only.
- Push to origin, do NOT open PR.
- PR diff vs main will include: (i) 6.5b diagnosis docs from a661d44,
  (ii) the locked-contract code fix + test, (iii) the 6.5c multi-seed
  results + gate decision. One merge unblocks Phase 7 or queues 6.5d.
