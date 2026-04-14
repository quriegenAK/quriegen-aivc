# AIVC — Eval Hardening Spec (v1)

**File:** `Claude Outputs/aivc_genelink/aivc_spec_eval_hardening_v1.md`
**Scope:** `eval/{metrics.py, benchmarks/norman_eval.py, benchmarks/kang_eval.py, benchmarks/replogle_eval.py, eval_runner.py}`
**Mode:** Scientist
**Author:** Ash (spec)
**Date:** 2026-04-13
**Status:** Design-ready — no code generated yet. Companion to `aivc_spec_tracking_foundation_v1.md`.

---

## 0. Hard invariants (spec header — non-negotiable)

1. **Kang regression guard is absolute.** Any checkpoint producing `pearson_r < 0.873` on Kang 2018 IFN-β held-out stim is *rejected*. It is not registered. It never reaches the Model Registry. `post_run_hook` receives `PostRunDecision(should_register=False, reason="kang_regression_guard_failed")`.
2. **The primary Norman metric is ctrl-subtracted Pearson r.** Raw `pearson_r(pred, truth)` is retained for comparability with older reports but does not drive pass/fail. Rationale in §1.
3. **Delta collapse is a first-class failure.** If `delta_nonzero_pct == 0` on any benchmark, the checkpoint is flagged as collapsed regardless of raw Pearson. The failure path in `post_run_hook` wins over the success path (precedence already fixed in tracking spec §4.3).
4. **Replogle evaluation is restricted to the 267-gene safe set.** Any attempt to evaluate on the full essential-gene screen or Frangieh 2021 melanoma dataset raises `EvalDataBlocked`. This is a data-governance invariant per `context.md`.
5. **Bench order is fixed: Kang → Norman → Replogle.** Kang is the cheapest regression guard; running Norman before Kang wastes GPU if the checkpoint has already drifted below the baseline.

---

## 1. The ctrl-subtraction fix — biological and methodological justification

### 1.1 The bug (current state, April 2026)

Zero-shot evaluation on Norman 2019 reports `pearson_r(predicted_stim, actual_stim) ≈ 0.99` while `delta = predicted_stim − predicted_ctrl ≈ 0` for every perturbation. This is not a success — it is a degenerate solution. The model has learned the identity `f(ctrl, pert) ≈ ctrl` for all perturbations. Raw Pearson is inflated because ctrl-level expression dominates the signal: single-cell log1p-normalised expression is heavily zero-inflated and highly correlated across conditions before any perturbation response is extracted. The fraction of between-condition variance attributable to the perturbation is small (typically 1–5% of total variance on Norman 2019), so correlating on raw expression effectively measures ctrl-vs-ctrl correlation.

### 1.2 The fix

Evaluate Pearson on the *delta* space, not the raw expression space:

```
pred_delta  = predicted_stim − ctrl
truth_delta = actual_stim    − ctrl
r_ctrl_sub  = pearson(pred_delta, truth_delta)
```

This is the standard Norman 2019 evaluation protocol (Supplementary Methods, Equation S3), adopted by scGPT, GEARS, Mean/Linear baselines, and the CausalBench benchmark. It isolates the *perturbation-specific response* from the ctrl-level baseline, which is what the model actually needs to learn. A model that merely copies ctrl gets `pred_delta == 0`, which yields either `r == nan` (zero variance) or `r ≈ 0` — exactly what we want to see when the model is collapsed.

### 1.3 When raw Pearson is retained

Kang 2018 IFN-β benchmark reports `pearson_r` on raw stim historically; the 0.873 baseline was measured that way. To preserve the regression guard semantics, `kang_eval` computes **both** raw and ctrl-subtracted Pearson. Only the raw value is compared against 0.873. The ctrl-subtracted value is reported for cross-benchmark parity and will drive the v2 regression guard once the delta-collapse fix is validated.

---

## 2. `eval/metrics.py` — shared metric functions

Pure functions. No model I/O. Works on NumPy arrays of shape `(n_cells, n_genes)`. All functions log to `logging.getLogger("aivc.eval.metrics")`, never print.

### 2.1 Module-level constants

```python
# eval/metrics.py
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger("aivc.eval.metrics")

# Biological noise floor for log1p-normalised scRNA-seq.
# log1p values typically range 0 – 10; technical noise (sampling + PCR + sequencer)
# sits at ~1e-4 on a per-gene-per-cell basis. 1e-6 is two orders of magnitude below
# the noise floor, so any |Δ| > 1e-6 is a real (non-numerical) signal.
# Do NOT raise this without re-checking against the dataset's empirical noise floor.
DEFAULT_DELTA_EPSILON: float = 1e-6
```

### 2.2 `pearson_r_ctrl_subtracted` — primary Norman/Replogle metric

```python
def pearson_r_ctrl_subtracted(
    pred:  np.ndarray,   # (n_cells, n_genes) predicted stim expression
    ctrl:  np.ndarray,   # (n_cells, n_genes) ctrl   expression (paired)
    truth: np.ndarray,   # (n_cells, n_genes) ground-truth stim expression
) -> float:
    """
    Mean per-cell Pearson r between (pred − ctrl) and (truth − ctrl).

    Primary metric for perturbation-response evaluation (Norman 2019 protocol,
    Supp. Methods eq. S3). Ctrl-subtracted to isolate the perturbation-specific
    response and prevent baseline-expression correlation from inflating the
    score — the failure mode documented in §1 of this spec.

    Edge cases
    ----------
    - If (truth − ctrl).std() == 0 across all entries → return float('nan') and
      log a warning: 'degenerate ground truth delta — all genes at ctrl level'.
      This happens when the 'stim' label is the same as ctrl (data corruption).
    - If (pred − ctrl).std() == 0 for a given cell, that cell contributes 0.0
      to the mean (collapsed prediction) — not skipped, to keep the denominator
      honest.
    - NaN per-cell r values are coerced to 0.0 before averaging.

    Shape contract
    --------------
    pred, ctrl, truth must all be shape (n_cells, n_genes) with identical
    ordering. The caller is responsible for pairing ctrl→stim at the cell or
    pseudobulk level (see benchmark specs below).
    """
```

### 2.3 `delta_nonzero_pct` — collapse detector

```python
def delta_nonzero_pct(
    pred:    np.ndarray,                         # (n_cells, n_genes)
    ctrl:    np.ndarray,                         # (n_cells, n_genes)
    epsilon: float = DEFAULT_DELTA_EPSILON,      # 1e-6
) -> float:
    """
    Percent of (gene, cell) entries where |pred − ctrl| > epsilon.

    Range: [0.0, 100.0].
    0.0 ⇒ the model has fully collapsed to 'predict ctrl for every perturbation'.
          This is the specific failure mode that motivated this spec.

    Epsilon rationale (biological)
    ------------------------------
    1e-6 is two orders of magnitude below the empirical noise floor of
    log1p-normalised scRNA-seq (~1e-4, dominated by PCR + capture + sequencer).
    Any Δ larger than 1e-6 is a real predicted response, not numerical jitter.
    log1p values span roughly 0–10 in the datasets in scope; 1e-6 is effectively
    machine-precision zero in that dynamic range.

    Do not tune this per benchmark. The same epsilon is used across Kang,
    Norman, Replogle to keep the collapse definition dataset-independent.
    """
```

### 2.4 `ctrl_memorisation_score` — cosine(pred, ctrl)

```python
def ctrl_memorisation_score(
    pred: np.ndarray,   # (n_cells, n_genes)
    ctrl: np.ndarray,   # (n_cells, n_genes)
) -> float:
    """
    Mean cosine similarity between pred and ctrl across cells. Range [0, 1]
    assuming non-negative log1p expression (which is the case post-clamp in
    perturbation_model.py: `predicted = (ctrl + pred_delta).clamp(min=0.0)`).

    Interpretation
    --------------
    1.0  → pred == ctrl (model copied ctrl exactly — full collapse).
    >0.95 → near-collapse; predicted response is in the noise.
    <0.5 → model is predicting expression meaningfully away from ctrl.

    This is a complementary signal to delta_nonzero_pct: a model could have
    delta_nonzero_pct > 0 but cosine ≈ 1 (tiny deltas, collapsed) or
    delta_nonzero_pct == 0 but cosine < 1 (impossible — flag as bug).
    The sanity invariant:
        if delta_nonzero_pct == 0.0: assert ctrl_memorisation_score > 0.999
    must hold; a violation indicates a numerical error in the caller.
    """
```

### 2.5 `top_k_gene_overlap` — supplementary metric

```python
def top_k_gene_overlap(
    pred_delta:  np.ndarray,   # (n_cells, n_genes)  pred  − ctrl
    truth_delta: np.ndarray,   # (n_cells, n_genes)  truth − ctrl
    k: int = 20,
) -> float:
    """
    Jaccard overlap |P ∩ T| / |P ∪ T| between the top-k up-regulated genes of
    pred_delta and truth_delta (mean over cells, then rank-order). Range [0,1].

    Supplementary metric. Used in Obsidian bench notes and the
    NormanEvalReport.top20_gene_overlap field. NOT part of the pass/fail gate
    in v1 — biological relevance of top-k is noisy at n_cells low.

    k=20 is the Norman 2019 default for ISG-panel overlap.
    """
```

### 2.6 Utility — per-cell Pearson helper

Used internally by `pearson_r_ctrl_subtracted`. Matches the tolerance logic already in `train_v11.py::compute_pearson_r` for drift-free parity.

```python
def _per_cell_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Per-row Pearson r between a and b. Returns (n_cells,).
    Rows with std < 1e-10 on either side are zero-filled (matches
    train_v11.compute_pearson_r).
    """
```

---

## 3. `eval/benchmarks/norman_eval.py`

### 3.1 Data-loading assumptions (verified, not guessed)

Verified against `data/norman2019.h5ad` on disk (April 2026):

| `adata.obs` column | Role                                | Values                              |
|--------------------|-------------------------------------|-------------------------------------|
| `perturbation`     | Perturbation identifier (string)    | 237 unique, e.g. `"KLF1"`, `"SET_KLF1"` |
| `nperts`           | Number of perturbed genes per cell  | `{0, 1, 2}` — **0 == control**      |
| `cell_line`        | Cell line                           | `"K562"` (single value)             |
| `gemgroup`         | Biological replicate / 10x lane     | 8 unique, used as donor analogue    |

**Ctrl mask contract:** `ctrl_mask = (adata.obs["nperts"] == 0).values`.
**Stim mask contract (per perturbation `p`):** `stim_mask = (adata.obs["perturbation"] == p).values & (adata.obs["nperts"] > 0).values`.

> Do not key off `perturbation == "ctrl"` — Norman 2019 does not use that string; `nperts == 0` is the canonical control definition. This was verified by inspecting the h5ad directly.

`adata.var` keys used:
- `adata.var_names` — primary gene index (33,694 genes in raw, subset to the 3,010-gene model vocabulary via `gene_to_idx` from `train_v11.py`).

### 3.2 Gene-set alignment

The model was trained on `n_genes = 3010` from `data/gene_names.txt`. Norman 2019 has ~33k genes. The eval must:
1. Load `gene_names.txt` → `model_gene_order: list[str]` (length 3010).
2. Intersect with `adata.var_names`. Assert intersection is exactly 3010 (if not, fail loudly with missing genes listed).
3. Reindex `adata.X` to match `model_gene_order`. This is the canonical gene axis for all downstream metrics.

### 3.3 Pairing strategy — pseudo-bulk per (perturbation, gemgroup)

Norman 2019 is unpaired (each cell is either ctrl or perturbed, not both). For eval we use pseudo-bulk averaging within `(perturbation, gemgroup)` groups, mirroring the training protocol in `build_ot_pairs.py`:

- For each `(perturbation p, gemgroup g)` with `≥ MIN_CELLS = 30` cells:
  - `ctrl_bulk = mean(X[ctrl_mask & gemgroup == g], axis=0)`
  - `stim_bulk = mean(X[stim_mask(p) & gemgroup == g], axis=0)`
- Pseudo-bulks with `< 30` cells on either side are dropped (logged).
- Output: `ctrl_array, truth_array ∈ R^(n_pseudobulks, 3010)`.

### 3.4 Inference pass

```python
from perturbation_model import PerturbationPredictor
# checkpoint restored via torch.load(checkpoint_path, map_location=device)
model.eval()
pert_id_stim = torch.tensor([1], device=device)           # stim token
ct_ids = torch.zeros(n_pseudobulks, dtype=torch.long, device=device)  # K562 → 0

with torch.no_grad():
    pred_delta = model.forward_batch(
        ctrl_array_tensor, edge_index, pert_id_stim, ct_ids
    )                                                      # (N, 3010)
    pred_array = (ctrl_array_tensor + pred_delta).clamp(min=0.0).cpu().numpy()
```

> Note on the current model: `PerturbationPredictor` takes a scalar `pert_id ∈ {0, 1}`, not a gene-specific perturbation ID. It cannot condition on *which* Norman gene was perturbed — it only knows "stim vs ctrl". This is a **known model-capacity limit** of v1.1, not an eval bug. The ctrl-subtracted metric will reveal the ceiling this imposes. Flagged as Risk #2 in §8.

### 3.5 `NormanEvalReport` — Pydantic schema

```python
from pydantic import BaseModel, ConfigDict, Field

class NormanEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # ── identity ──────────────────────────────────────────────
    run_id: str

    # ── primary metrics ───────────────────────────────────────
    pearson_r_ctrl_sub: float = Field(..., ge=-1.0, le=1.0,
        description="Pearson r on (pred − ctrl) vs (truth − ctrl). Primary metric.")
    pearson_r_raw: float = Field(..., ge=-1.0, le=1.0,
        description="Pearson r on raw pred vs truth. Retained for comparison only.")

    # ── collapse detectors ────────────────────────────────────
    delta_nonzero_pct: float = Field(..., ge=0.0, le=100.0)
    ctrl_memorisation_score: float = Field(..., ge=0.0, le=1.0)
    top20_gene_overlap: float = Field(..., ge=0.0, le=1.0)

    # ── provenance ────────────────────────────────────────────
    n_perturbations_evaluated: int = Field(..., ge=1)
    n_genes: int = Field(..., description="Must equal 3010.")
    n_pseudobulks: int = Field(..., ge=1)

    # ── gate ──────────────────────────────────────────────────
    passed: bool
    failure_reason: str | None = None

    # ── model-side validators ────────────────────────────────
    @model_validator(mode="after")
    def _n_genes_is_3010(self) -> "NormanEvalReport":
        if self.n_genes != 3010:
            raise ValueError(f"n_genes must be 3010, got {self.n_genes}")
        return self

    @model_validator(mode="after")
    def _collapse_invariant(self) -> "NormanEvalReport":
        # Sanity: if delta collapsed, cosine must be ~1.
        if self.delta_nonzero_pct == 0.0 and self.ctrl_memorisation_score < 0.999:
            raise ValueError(
                "Collapse invariant violated: delta_nonzero_pct==0 but "
                f"ctrl_memorisation_score={self.ctrl_memorisation_score} < 0.999"
            )
        return self
```

### 3.6 Pass/fail gate (with biological justification)

```
passed = (delta_nonzero_pct > 1.0) AND (pearson_r_ctrl_sub > 0.0)
```

**`delta_nonzero_pct > 1.0%`** — in a 3,010-gene model, 1% corresponds to ~30 genes showing a predicted response. That is the minimum detectable single-gene CRISPRi effect size at the screen-throughput used in Norman 2019 and Replogle 2022 (per Replogle 2022 Figure S3: median affected-gene count per guide is 20–40 in log1p-normalised expression). Below 30 genes, a predicted perturbation effect cannot be distinguished from batch noise.

**`pearson_r_ctrl_sub > 0.0`** — deliberately permissive for v1. Any positive correlation with the ground-truth delta is progress beyond ctrl-memorisation. v2 raises this to `> 0.10` once the delta collapse is confirmed fixed on at least two sweep configurations.

### 3.7 Public entrypoint

```python
def run_norman_eval(
    checkpoint_path: str,
    adata_path: str = "data/norman2019.h5ad",
    *,
    run_id: str,
    device: str | None = None,
    min_cells_per_pseudobulk: int = 30,
) -> NormanEvalReport: ...
```

### 3.8 Failure-reason strings (deterministic taxonomy)

| Condition                                            | `failure_reason`                                    |
|------------------------------------------------------|-----------------------------------------------------|
| `delta_nonzero_pct == 0.0`                           | `"delta_collapse: model memorised ctrl"`            |
| `0 < delta_nonzero_pct <= 1.0`                       | `"delta_sub_threshold: <1% genes responding"`       |
| `pearson_r_ctrl_sub <= 0.0`                          | `"ctrl_sub_pearson_non_positive"`                   |
| `pearson_r_ctrl_sub == nan`                          | `"degenerate_truth_delta"`                          |

---

## 4. `eval/benchmarks/kang_eval.py`

### 4.1 Data-loading assumptions (verified)

Verified against `data/kang2018_pbmc_fixed.h5ad` via `build_ot_pairs.py`:

| `adata.obs` column | Role                  | Values                              |
|--------------------|-----------------------|-------------------------------------|
| `label`            | Condition             | `{"ctrl", "stim"}`                  |
| `replicate`        | Donor                 | 8 donors                            |
| `cell_type`        | Cell type             | 8 PBMC types                        |

`adata.var["name"]` (fall back to `adata.var_names`) is the gene name axis. Normalisation state is detected via `"log1p" in adata.uns`.

### 4.2 Pairing

Kang is already paired in the repo as `data/X_ctrl_paired.npy` and `data/X_stim_paired.npy` (pseudo-bulk per `(donor, cell_type)`), with `data/pairing_manifest.csv` holding the ordering. The eval:
- Prefers the pre-computed paired arrays when present (keeps parity with training).
- Falls back to re-pairing from `kang2018_pbmc_fixed.h5ad` via the same OT logic (`build_ot_pairs.py`) if the `.npy`s are missing.
- Uses the donor split defined in `test_split_info.pt` (last `n_donors - n_train - n_val` donors) — identical to the test split used in `train_v11.py` so the Pearson number is directly comparable to the 0.873 baseline.

### 4.3 `KangEvalReport` — Pydantic schema

```python
class KangEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str

    pearson_r: float = Field(..., ge=-1.0, le=1.0,
        description="Raw Pearson r(pred, truth). Compared against 0.873 baseline.")
    pearson_r_ctrl_sub: float = Field(..., ge=-1.0, le=1.0,
        description="Ctrl-subtracted Pearson for cross-benchmark parity.")
    delta_nonzero_pct: float = Field(..., ge=0.0, le=100.0)

    n_genes: int
    n_pseudobulks: int = Field(..., ge=1)

    passed: bool
    regression_guard_passed: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _n_genes_is_3010(self) -> "KangEvalReport":
        if self.n_genes != 3010:
            raise ValueError(f"n_genes must be 3010, got {self.n_genes}")
        return self
```

### 4.4 Regression guard — exact logic (non-negotiable)

```python
BASELINE_R = 0.873  # v1.0 held-out pearson on Kang IFN-β, anchor for the guard

def run_kang_eval(...) -> KangEvalReport:
    ...
    r_raw     = pearson_r_on_test_set(pred, truth)
    r_ctrl_sub = pearson_r_ctrl_subtracted(pred, ctrl, truth)
    dnz       = delta_nonzero_pct(pred, ctrl)

    if r_raw < BASELINE_R:
        return KangEvalReport(
            run_id=run_id,
            pearson_r=r_raw,
            pearson_r_ctrl_sub=r_ctrl_sub,
            delta_nonzero_pct=dnz,
            n_genes=N_GENES,
            n_pseudobulks=len(ctrl),
            passed=False,
            regression_guard_passed=False,
            failure_reason=f"Kang regression guard: r={r_raw:.4f} < {BASELINE_R} baseline",
        )
    ...
```

### 4.5 Downstream contract

Kang failure → `post_run_hook` must raise `CheckpointRejected` (see §6.2). A rejected checkpoint is **never** registered, regardless of its Norman/Replogle scores (which in fact won't be computed — see §5 ordering).

---

## 5. `eval/benchmarks/replogle_eval.py`

### 5.1 Data-governance invariant (from `context.md`)

> Only the 267-gene safe set from Replogle 2022 is cleared for training and evaluation. The full 2,058-gene essential-gene screen and Frangieh 2021 melanoma dataset are blocked pending data-sharing agreement review.

Enforcement: `run_replogle_eval` must assert:
1. Input `adata_path` resolves (by name match or `adata.uns["aivc_dataset_tag"]`) to the safe-set h5ad. If not, raise `EvalDataBlocked`.
2. After gene-set alignment, `n_genes_evaluated == 267` exactly. If greater, raise — the caller passed a larger gene panel than allowed.

### 5.2 Data-loading assumptions

Replogle 2022 safe set (published as `replogle2022_safe_267.h5ad` by the data team):

| `adata.obs` column | Role                               | Values                                 |
|--------------------|------------------------------------|----------------------------------------|
| `gene_target`      | Perturbed gene name                | 267 unique + `"non-targeting"` for ctrl |
| `sgRNA`            | Guide identifier                   | ~2–4 guides per target                  |
| `batch`            | Sequencing batch                   | 4 batches                               |

Ctrl mask: `adata.obs["gene_target"].str.lower() == "non-targeting"`.

> This schema is taken from the Replogle 2022 Cell paper (Table S1) and the published h5ad on FigShare. **Verify on the actual safe-set file before implementation** — the data team's export script may rename columns. If it does, this spec's column names are overridden by the shipped file.

### 5.3 `ReplogleEvalReport` — Pydantic schema

```python
class ReplogleEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str

    pearson_r_ctrl_sub: float = Field(..., ge=-1.0, le=1.0)
    delta_nonzero_pct: float  = Field(..., ge=0.0, le=100.0)

    n_genes_evaluated: int
    n_perturbations_evaluated: int

    passed: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _safe_set_267(self) -> "ReplogleEvalReport":
        if self.n_genes_evaluated != 267:
            raise ValueError(
                f"Replogle eval must use 267-gene safe set only; "
                f"got n_genes_evaluated={self.n_genes_evaluated}. "
                "Full essential-gene screen is blocked by data governance."
            )
        return self
```

### 5.4 Pass/fail gate

Same as Norman:
```
passed = (delta_nonzero_pct > 1.0) AND (pearson_r_ctrl_sub > 0.0)
```
Rationale: Replogle is also a perturbation-response benchmark with the same collapse failure mode. Uniform thresholds across perturbation benchmarks keep the gate comparable.

### 5.5 Public entrypoint

```python
def run_replogle_eval(
    checkpoint_path: str,
    adata_path: str,                 # must resolve to the 267-safe-set file
    *,
    run_id: str,
    device: str | None = None,
) -> ReplogleEvalReport: ...
```

---

## 6. `eval/eval_runner.py` — consolidated runner

### 6.1 `EvalSuite` schema

```python
class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str
    checkpoint_path: str

    kang:     KangEvalReport
    norman:   NormanEvalReport | None = None      # None if Kang gated it
    replogle: ReplogleEvalReport | None = None    # None if Kang/Norman gated it

    overall_passed: bool
    halt_reason: str | None = None                 # set iff early-halted
```

### 6.2 `run_eval_suite` — sequencing and short-circuit logic

```python
class CheckpointRejected(Exception):
    """Raised when the Kang regression guard fails. Never caught inside the runner —
    it propagates to post_run_hook so the orchestrator can log and route."""


def run_eval_suite(
    checkpoint_path: str,
    *,
    run_id: str,
    kang_adata:     str = "data/kang2018_pbmc_fixed.h5ad",
    norman_adata:   str = "data/norman2019.h5ad",
    replogle_adata: str = "data/replogle2022_safe_267.h5ad",
    device: str | None = None,
) -> EvalSuite:
    """
    Sequence — fixed, non-reorderable:
      1. Kang   → regression guard. If fails, halt. Norman/Replogle skipped.
      2. Norman → delta collapse check.
      3. Replogle → 267-safe-set validation.

    Rationale for the order:
      Kang is the cheapest (pre-paired arrays on disk). If the checkpoint has
      degraded below r=0.873, nothing else matters — Norman and Replogle cost
      GPU time and their reports would be thrown away anyway.
      Norman is second because delta collapse was the failure mode this whole
      spec exists to catch; fixing it on Norman is the primary signal.
      Replogle is last because the 267-safe-set has the fewest perturbations
      and is the most expensive to load (large h5ad).
    """
```

Pseudo-flow:

```
kang = run_kang_eval(...)
if not kang.regression_guard_passed:
    return EvalSuite(run_id, checkpoint_path, kang=kang,
                     overall_passed=False,
                     halt_reason=kang.failure_reason)

norman = run_norman_eval(...)
# Norman does NOT halt the suite even if it fails — Replogle still runs
# because both failures are informative for the failure-note downstream.

replogle = run_replogle_eval(...)

overall = kang.passed and norman.passed and replogle.passed
return EvalSuite(run_id, checkpoint_path, kang, norman, replogle,
                 overall_passed=overall,
                 halt_reason=None)
```

### 6.3 CLI entrypoint

```
python -m eval.eval_runner \
  --checkpoint models/v1.1/sweep_beta0.20_K3_l10.001.pt \
  --run-id 20260413_wsweep_17 \
  [--kang-adata ...] [--norman-adata ...] [--replogle-adata ...]
```

Exit code: 0 if `overall_passed`, 1 otherwise. A rejected Kang regression is *also* exit 1 — checkpoint degradation is a halt-the-pipeline event.

---

## 7. Integration with `RunMetadata` and `post_run_hook`

### 7.1 Field mapping — `EvalSuite → RunMetadata`

The evaluator must populate `RunMetadata` **before** `ExperimentLogger.finish()` is called, so the `post_run_hook` sees populated metrics.

| `RunMetadata` field       | Source in `EvalSuite`                                          |
|---------------------------|----------------------------------------------------------------|
| `run_id`                  | `suite.run_id`                                                 |
| `pearson_r`               | `suite.kang.pearson_r` (raw — matches 0.873 baseline semantics) |
| `delta_nonzero_pct`       | `suite.norman.delta_nonzero_pct` (primary collapse signal)     |
| `ctrl_memorisation_score` | `suite.norman.ctrl_memorisation_score`                         |
| `frozen_modules`          | From training config — not overwritten by eval                 |
| `checkpoint_path`         | `suite.checkpoint_path`                                        |
| `dataset`                 | `"kang2018_pbmc_ifnb"` (primary) — eval tag appended as `aivc_eval_datasets` tag: `"kang+norman+replogle-267"` |
| `modalities`              | `[Modality.RNA]`                                               |
| `mlflow_run_id`           | Set by `ExperimentLogger`, not the eval                        |
| `wandb_url`               | Set by `ExperimentLogger`, not the eval                        |

### 7.2 `post_run_hook` decision table (combining tracking + eval)

| `kang.regression_guard_passed` | `norman.passed` | `replogle.passed` | `should_register` | `should_trigger_training_agent` | `should_trigger_research_agent` | `reason`                                       |
|-------------------------------|-----------------|-------------------|-------------------|---------------------------------|---------------------------------|-----------------------------------------------|
| False                         | *               | *                 | False             | True                            | False                           | `kang_regression_guard_failed`                |
| True                          | False (delta=0) | *                 | False             | True                            | False                           | `norman_delta_collapse`                       |
| True                          | False (r≤0)     | *                 | False             | True                            | False                           | `norman_ctrl_sub_pearson_non_positive`        |
| True                          | True            | False             | False             | True                            | False                           | `replogle_failed`                             |
| True                          | True            | True              | True              | False                           | True                            | `all_passed — checkpoint registered`          |

The hook writes `failure_notes/{run_id}.md` whenever `should_register == False` (uses the template from tracking spec §4.5, with an extra "Eval Suite" section appended that dumps `suite.model_dump_json()`).

### 7.3 Registration tags (on success)

When `should_register == True`, `mlflow.register_model` is called with tags that mirror the eval suite, so the Model Registry is queryable by biological provenance:

```python
tags = {
    "pearson_r_kang":            f"{suite.kang.pearson_r:.4f}",
    "pearson_r_ctrl_sub_kang":   f"{suite.kang.pearson_r_ctrl_sub:.4f}",
    "pearson_r_ctrl_sub_norman": f"{suite.norman.pearson_r_ctrl_sub:.4f}",
    "delta_nonzero_pct_norman":  f"{suite.norman.delta_nonzero_pct:.2f}",
    "ctrl_memorisation_norman":  f"{suite.norman.ctrl_memorisation_score:.4f}",
    "pearson_r_ctrl_sub_replogle": f"{suite.replogle.pearson_r_ctrl_sub:.4f}",
    "eval_spec_version":         "v1",
    "anndata_hash_kang":         "<sha256[:16]>",
    "anndata_hash_norman":       "<sha256[:16]>",
    "anndata_hash_replogle":     "<sha256[:16]>",
}
```

---

## 8. Risks, gaps, unknowns

| # | Item                                                                               | Severity | Mitigation                                                                                                 |
|---|------------------------------------------------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------|
| 1 | `perturbation_model.PerturbationPredictor` takes scalar `pert_id ∈ {0,1}`, cannot condition on gene identity. On Norman/Replogle it will predict the *same* delta for every perturbation. | **HIGH** | This is a known v1.1 limit. The ctrl-sub metric will expose the ceiling. v2 must add gene-conditioned perturbation embedding before these benchmarks are meaningful at per-perturbation granularity. Flag in the spec summary so reviewers don't expect strong Norman numbers. |
| 2 | `data/replogle2022_safe_267.h5ad` — file existence and exact obs column names not verified on disk; spec is based on Replogle 2022 Cell paper conventions. | MED      | Implementation PR must first run a small read-and-assert script and update §5.2 if columns differ.        |
| 3 | Gene-vocabulary mismatch: Norman has 33k genes, model has 3,010. Intersection could drop below 3,010 if naming conventions differ (e.g. Ensembl vs HGNC).     | MED      | Assert hard: intersection == 3010 before proceeding. Fail-loud with the list of missing symbols. Use `adata.var["ensemble_id"]` as fallback if needed. |
| 4 | Norman cell line is K562; model's cell-type embedding was trained on PBMC types. Passing `ct_id=0` is a lie.                                                    | MED      | Document that Norman eval uses `ct_id=0` for all cells — this is a placeholder. v2 should expose a cross-cell-type eval flag. Record the limitation in `NormanEvalReport.failure_reason` when it's the dominant failure mode. |
| 5 | `pearson_r_ctrl_sub` can return NaN on degenerate truth. Downstream `RunMetadata` validator requires float — NaN will serialise to "NaN" string or fail.       | LOW      | Pydantic v2 default rejects NaN floats. Add `allow_inf_nan=False` on the field and coerce NaN → 0.0 at the boundary with a logged warning.                                           |
| 6 | `test_split_info.pt` may not exist in every workspace; Kang fallback re-runs OT pairing which is non-deterministic unless seed is frozen.                       | LOW      | Seed the OT re-pair with `SEED=42` (same as `train_v11.py`). Cache the resulting split to disk on first eval.                                                                        |
| 7 | `EvalDataBlocked` exception class has no defined location.                         | LOW      | Define in `eval/exceptions.py` alongside `CheckpointRejected`. Both must be importable from `eval/__init__.py`. |
| 8 | Top-k overlap at `k=20` is noisy at `n_pseudobulks < 50`.                          | LOW      | Keep it informational only. Do not use in pass/fail gate. The spec already does this.                      |

---

## 9. Acceptance tests (for implementation PR)

1. `pearson_r_ctrl_subtracted(ctrl, ctrl, ctrl)` → `nan` (degenerate truth), warning logged.
2. `delta_nonzero_pct(ctrl, ctrl)` → `0.0`.
3. `ctrl_memorisation_score(ctrl, ctrl)` → `1.0` (± 1e-12).
4. `NormanEvalReport(delta_nonzero_pct=0.0, ctrl_memorisation_score=0.5, ...)` raises `ValidationError` (collapse invariant).
5. `run_kang_eval` on a noised-ctrl checkpoint (r < 0.873) returns `regression_guard_passed=False` with the exact failure-reason string.
6. `run_eval_suite` skips Norman + Replogle when Kang fails, sets `halt_reason` correctly, returns exit code 1 via CLI.
7. `run_replogle_eval` raises `EvalDataBlocked` when handed a 2,058-gene essential-screen file.
8. `ReplogleEvalReport(n_genes_evaluated=500, ...)` raises `ValidationError` (267-safe-set invariant).
9. `post_run_hook` on `EvalSuite(kang.regression_guard_passed=False)` returns `PostRunDecision(should_register=False, should_trigger_training_agent=True, reason="kang_regression_guard_failed")`.
10. Mirror of tracking-spec test #4: `post_run_hook` with `delta_nonzero_pct==0` wins over `pearson_r==0.9` (failure precedence).

---

## Key Risks

- **Scalar `pert_id` is the ceiling.** The current model cannot per-perturbation condition. Even a perfectly hardened eval will show weak Norman/Replogle ctrl-sub Pearson. The eval's value is as a *collapse detector* and *regression guard* today, not a leaderboard. Communicate this before anyone expects SOTA numbers from the first hardened run.
- **`n_genes_evaluated == 267` invariant is a silent safety net.** If the data team ships a file tagged as the safe set but actually containing more genes, the Pydantic validator catches it — but only at eval time, not at ingestion. A sha256 of the file + an allow-list in `context.md` would be stronger; out of scope for this spec.
- **`pearson_r_ctrl_sub` on a fully collapsed model is `nan`, not `0`.** NaN propagates through MLflow/W&B as a string and is easy to miss in dashboards. The hook must treat `nan` as `pearson_r_ctrl_sub = 0.0` for routing while logging the original NaN in the run-level tag.
- **Kang test split must be bit-for-bit identical to training.** A different donor split will produce a different number, and 0.873 is a sharp threshold. The spec mandates `test_split_info.pt` as the source of truth; deviation invalidates the regression guard.
- **Gene-vocabulary drift.** 3,010 is encoded in the model weights and the API response schemas. Any eval-time reindexing that silently pads missing genes with zeros will look like delta collapse on those genes. Hard-assert that the intersection is exactly 3,010 before metrics run.

## Recommended Next Step

Implement `eval/metrics.py` first (pure, no I/O, unit-testable), with the 10 acceptance tests above driving the design. Ship behind `pytest` green. Then implement `kang_eval.py` second — it reuses the pre-paired arrays already on disk, so it's the lowest-risk integration point and proves the plumbing to `RunMetadata`. Norman and Replogle come after Kang is green, because their data-governance and gene-set alignment layers add meaningful risk on top of the core metric logic.

## What I need from you

1. Confirm the Replogle safe-set file name and the exact `obs` column for the perturbed gene (`gene_target` vs `guide_identity` vs something else). Shipping path from the data team?
2. Confirm whether we accept the v1.1 Norman limitation (scalar `pert_id` → same predicted delta for every perturbation) as a *known ceiling*, or whether this spec should block until a gene-conditioned perturbation embedding lands.
3. Confirm that `pearson_r_ctrl_sub = nan` should be coerced to `0.0` at the `RunMetadata` boundary (vs. failing the run outright). My recommendation is coerce + warn; it's a failure signal, not a crash condition.
4. Sign off on the `delta_nonzero_pct > 1.0%` threshold. If product/research thinks 30 responding genes is too lax, we can raise to 2–5%; the spec is written so the number is changeable in one place.
