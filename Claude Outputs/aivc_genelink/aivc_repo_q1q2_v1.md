# AIVC — Repo Q1/Q2 Findings (v1)

**File:** `Claude Outputs/aivc_genelink/aivc_repo_q1q2_v1.md`
**Date:** 2026-04-13
**Scope:** Replogle file presence + `pert_id` architecture verification.

---

## Q1 — Replogle file

**Result: not found.** No file in `data/` contains "replogle" (case-insensitive).

Full listing of `data/`:

```
9606.protein.info.v12.0.txt
9606.protein.info.v12.0.txt.gz
9606.protein.links.v12.0.txt
9606.protein.links.v12.0.txt.gz
BL--network.csv
Expression.csv
Label.csv
TF.csv
Target.csv
X_ctrl_ot.npy
X_ctrl_paired.npy
X_stim_ot.npy
X_stim_paired.npy
cell_type_ot.npy
donor_ot.npy
edge_list.csv
edge_list_fixed.csv
gene_names.txt
geo_cache/
kang2018_pbmc.h5ad
kang2018_pbmc_decontaminated.h5ad
kang2018_pbmc_fixed.h5ad
norman2019.h5ad
ot_pairing_log.txt
ot_pairing_manifest.csv
pairing_certificates/
pairing_manifest.csv
validation_certificates/
```

**Implication for the eval spec:** `eval/benchmarks/replogle_eval.py` has no on-disk dataset yet. The Replogle branch of `eval_runner.py` must be gated behind a file-existence check or a `--skip-replogle` flag. Risk #2 in the eval hardening spec should be elevated from MED → HIGH: the 267-safe-set file is *not staged*, not merely unverified.

Candidate location to watch: `geo_cache/` (may contain a pending download). Not inspected here per instruction to list only, not read contents.

---

## Q2 — `pert_id` architecture

### Exact `PerturbationPredictor.forward()` signature + first 10 lines of body

```python
def forward(self, x_ctrl: torch.Tensor, edge_index: torch.Tensor,
            pert_id: torch.Tensor,
            cell_type_ids: torch.Tensor = None) -> torch.Tensor:
    """Predict stimulated expression from control expression.

    Args:
        x_ctrl: (n_genes,) or (n_genes, 1) — mean ctrl expression per gene.
        edge_index: (2, n_edges) — graph connectivity.
        pert_id: scalar or (1,) — perturbation index (1 for stim).
        cell_type_ids: optional scalar or (1,) — cell type index.
                      If None: cell type embedding is zero (backward compatible).
                      If provided: adds cell type embedding to features.
```

(First 10 executable lines of the body, after the docstring:)

```python
    # 1. Expand scalar features
    x = self.feature_expander(x_ctrl)  # (n_genes, feature_dim)

    # 2. Add perturbation signal
    x = self.pert_embedding(x, pert_id)  # (n_genes, feature_dim)

    # 3. Add cell type signal (if available and embedding exists)
    if cell_type_ids is not None and hasattr(self, "cell_type_embedding"):
        x = self.cell_type_embedding(x, cell_type_ids)

    # 4. GAT message passing
    z = self.genelink(x, edge_index)  # (n_genes, gnn_out_dim)
```

### Is `pert_id` a scalar {0,1} or a gene-vocabulary index?

**Currently a scalar index into a tiny vocabulary, not a gene index.** The embedding table is sized by `num_perturbations`, instantiated with `num_perturbations=2` throughout training and the API:

- `train_v11.py:262` — `PerturbationPredictor(..., num_perturbations=2, ...)`
- `api/server.py:100` — `PerturbationPredictor(..., num_perturbations=2, ...)`
- `api/server.py:176` — `pert_id = torch.tensor([1])  # stim`

So today `pert_id ∈ {0, 1}` (0 = ctrl, 1 = stim). The `PerturbationEmbedding.forward` broadcasts a single learned vector across all genes:

```python
def forward(self, x: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
    if pert_id.dim() == 0:
        pert_id = pert_id.unsqueeze(0)
    pert_vec = self.embedding(pert_id[0:1])   # (1, embedding_dim)
    return x + pert_vec                        # broadcast over all n_genes
```

Note `pert_id[0:1]` — even if a `(batch,)` tensor is passed, only index 0 is used. There is no path in the current code where different rows see different perturbation identities.

**The architecture is *extensible* — the docstring of `PerturbationEmbedding.__init__` explicitly says `"2 for ctrl/stim in Week 2; extensible to 90+ for Parse 10M"` — but it is not currently extended. Norman 2019 (237 perturbations) and Replogle 267-safe-set cannot be meaningfully evaluated without re-instantiating with `num_perturbations=237` (Norman) or `num_perturbations=267` (Replogle) *and* retraining.**

### Exact embedding layer definition

Two `nn.Embedding` layers exist, both defined in `perturbation_model.py`:

```python
# Line 140 — PerturbationEmbedding.__init__
self.embedding = nn.Embedding(num_perturbations, embedding_dim)

# Line 434 — CellTypeEmbedding.__init__
self.embedding = nn.Embedding(num_cell_types, embedding_dim)
```

`PerturbationEmbedding` is attached to `PerturbationPredictor` as `self.pert_embedding` (line 251):

```python
self.pert_embedding = PerturbationEmbedding(num_perturbations, feature_dim)
```

No gene-level embedding layer exists. The model treats genes as graph nodes with scalar-expanded features, not as tokens in an embedding table.

---

## Compact answers (as requested)

**Q1:** `not found` | `—` | `—`
  - Full `data/` listing in §Q1 above. No replogle file on disk. Eval spec Replogle branch is currently un-stage-able.

**Q2:**
  - `forward signature`: `def forward(self, x_ctrl: torch.Tensor, edge_index: torch.Tensor, pert_id: torch.Tensor, cell_type_ids: torch.Tensor = None) -> torch.Tensor`
  - `pert_id type`: scalar integer index into a size-2 vocabulary (`{0=ctrl, 1=stim}`). *Not* a gene index. Architecture is extensible to larger vocabularies but not currently instantiated that way.
  - `embedding def`: `self.embedding = nn.Embedding(num_perturbations, embedding_dim)` inside `PerturbationEmbedding` (perturbation_model.py:140), attached as `self.pert_embedding` on `PerturbationPredictor` (line 251) with `num_perturbations=2` at call sites.

---

## Key Risks (implications for the eval hardening spec)

- **Replogle data is not present on disk.** Promote Risk #2 from MED to HIGH. `run_eval_suite` must fail-closed (or skip-with-warning) when `data/replogle2022_safe_267.h5ad` is absent. Do not let the suite report `replogle=None, overall_passed=False` ambiguously — distinguish "not evaluated" from "failed".
- **Scalar `pert_id` makes Norman/Replogle per-perturbation eval meaningless today.** Every perturbation collapses to the same predicted delta because `PerturbationEmbedding.forward` broadcasts a single vector, and only `pert_id[0:1]` is ever read. The eval hardening spec's collapse detector (`delta_nonzero_pct`) is the right-sized tool for this regime, but top-k / per-perturbation Pearson metrics will be uninformative until the embedding vocabulary is expanded *and* retrained.
- **Fixing this requires both code and data.** `num_perturbations` must be lifted to ≥237 (Norman) or ≥267 (Replogle), `pert_id` plumbing in `forward_batch` must be made per-row (currently `pert_vec = ...[pert_id[0:1]]`), and a new training run must learn the gene-specific perturbation vectors. This is a v1.2 scope item, not a v1.1 eval-hardening item.

## Recommended Next Step

Block the Replogle branch of the eval hardening spec's v1 implementation until the data-team ships the safe-set h5ad. Ship Kang + Norman evals first — both have the data on disk and both can already exercise the collapse detectors. Open a v1.2 ticket to expand `num_perturbations` and add per-row `pert_id` indexing before revisiting Norman/Replogle as per-perturbation benchmarks.

## What I need from you

1. ETA on the Replogle 267-safe-set h5ad from the data team (and the governance allow-list entry in `context.md`).
2. Decision: ship v1 eval with `replogle=None` handling as "not-evaluated" (my recommendation), or block v1 until the file lands.
3. Sign-off on deferring gene-conditioned `pert_id` to v1.2.
