"""
analyse_pairing.py — Cell pairing analysis for perturbation response prediction.

Analyses the Kang 2018 PBMC dataset to build matched (ctrl, stim) pseudo-bulk
expression pairs grouped by (donor, cell_type). These pairs become the training
data for the perturbation prediction model.

Outputs:
    data/pairing_manifest.csv  — manifest of all donor x cell_type groups
    data/X_ctrl_paired.npy     — ctrl pseudo-bulk expression (n_pairs, n_genes)
    data/X_stim_paired.npy     — stim pseudo-bulk expression (n_pairs, n_genes)
"""
import random
import numpy as np
import pandas as pd
import anndata as ad

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# 1. Load data
# ----------------------------
print("Loading data/kang2018_pbmc_fixed.h5ad...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

# ----------------------------
# 2. Examine obs columns
# ----------------------------
print("\n  All obs columns and unique values:")
for col in adata.obs.columns:
    n_unique = adata.obs[col].nunique()
    if n_unique <= 20:
        vals = adata.obs[col].unique().tolist()
        print(f"    {col}: {n_unique} unique -> {vals}")
    else:
        print(f"    {col}: {n_unique} unique values")

# Identify the donor column
donor_col = None
for candidate in ["replicate", "donor_id", "donor", "patient", "sample"]:
    if candidate in adata.obs.columns:
        donor_col = candidate
        break
if donor_col is None:
    raise ValueError("No donor column found in adata.obs. Checked: replicate, donor_id, donor, patient, sample")
print(f"\n  Using donor column: '{donor_col}'")

# Identify the cell type column
celltype_col = None
for candidate in ["cell_type", "celltype", "cell.type", "leiden", "louvain"]:
    if candidate in adata.obs.columns:
        celltype_col = candidate
        break
if celltype_col is None:
    raise ValueError("No cell type column found in adata.obs.")
print(f"  Using cell type column: '{celltype_col}'")

donors = sorted(adata.obs[donor_col].unique().tolist())
cell_types = sorted(adata.obs[celltype_col].unique().tolist())
print(f"  Donors ({len(donors)}): {donors}")
print(f"  Cell types ({len(cell_types)}): {cell_types}")

# ----------------------------
# 3. Compute per-donor, per-cell-type counts
# ----------------------------
print("\n" + "=" * 80)
print("PAIRING ANALYSIS: cells per (donor, cell_type, condition)")
print("=" * 80)

# Get gene names for later
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
n_genes = len(gene_names)

# Get expression matrix (dense)
X_dense = adata.X
if hasattr(X_dense, "toarray"):
    X_dense = X_dense.toarray()

labels = adata.obs["label"].values
donor_ids = adata.obs[donor_col].values
cell_type_ids = adata.obs[celltype_col].values

# ----------------------------
# 4. Build pairing matrix
# ----------------------------
manifest_rows = []
ctrl_paired_list = []
stim_paired_list = []
warnings = []

for donor in donors:
    for ct in cell_types:
        mask_ctrl = (donor_ids == donor) & (cell_type_ids == ct) & (labels == "ctrl")
        mask_stim = (donor_ids == donor) & (cell_type_ids == ct) & (labels == "stim")
        n_ctrl = mask_ctrl.sum()
        n_stim = mask_stim.sum()

        paired = (n_ctrl > 0) and (n_stim > 0)

        if n_ctrl > 0 and n_ctrl < 10:
            warnings.append(f"  WARNING: {donor} / {ct} has only {n_ctrl} ctrl cells (< 10)")
        if n_stim > 0 and n_stim < 10:
            warnings.append(f"  WARNING: {donor} / {ct} has only {n_stim} stim cells (< 10)")

        manifest_rows.append({
            "donor_id": donor,
            "cell_type": ct,
            "n_ctrl_cells": int(n_ctrl),
            "n_stim_cells": int(n_stim),
            "paired": paired,
        })

        if paired:
            ctrl_mean = X_dense[mask_ctrl, :].mean(axis=0)
            stim_mean = X_dense[mask_stim, :].mean(axis=0)
            ctrl_paired_list.append(ctrl_mean)
            stim_paired_list.append(stim_mean)

manifest = pd.DataFrame(manifest_rows)

# Print the manifest
print(f"\n{'Donor':<18} {'Cell Type':<22} {'Ctrl':<8} {'Stim':<8} {'Paired'}")
print("-" * 80)
for _, row in manifest.iterrows():
    paired_str = "YES" if row["paired"] else "---"
    print(f"  {row['donor_id']:<16} {row['cell_type']:<22} {row['n_ctrl_cells']:<8} {row['n_stim_cells']:<8} {paired_str}")

# Print warnings
if warnings:
    print()
    for w in warnings:
        print(w)

# ----------------------------
# 5. Summary statistics
# ----------------------------
n_total_pairs = len(ctrl_paired_list)
n_possible = len(donors) * len(cell_types)
n_unpaired = manifest[~manifest["paired"]].shape[0]

paired_manifest = manifest[manifest["paired"]]
min_ctrl = paired_manifest["n_ctrl_cells"].min() if n_total_pairs > 0 else 0
min_stim = paired_manifest["n_stim_cells"].min() if n_total_pairs > 0 else 0

print()
print("=" * 80)
print("PAIRING SUMMARY")
print("=" * 80)
print(f"  Total (donor, cell_type) combinations: {n_possible}")
print(f"  Paired combinations (both ctrl & stim present): {n_total_pairs}")
print(f"  Unpaired (excluded): {n_unpaired}")
print(f"  Minimum ctrl cells in a paired group: {min_ctrl}")
print(f"  Minimum stim cells in a paired group: {min_stim}")

# Check for zero-variance genes in ctrl
X_ctrl_paired = np.array(ctrl_paired_list)
X_stim_paired = np.array(stim_paired_list)

ctrl_var = X_ctrl_paired.var(axis=0)
n_zero_var = (ctrl_var < 1e-10).sum()
print(f"  Genes with zero variance in ctrl pseudo-bulk: {n_zero_var}")
if n_zero_var > 0:
    zero_var_genes = [gene_names[i] for i in range(n_genes) if ctrl_var[i] < 1e-10]
    print(f"    (flagged, not removed): {zero_var_genes[:10]}{'...' if n_zero_var > 10 else ''}")

# ----------------------------
# 6. Save pairing manifest
# ----------------------------
manifest.to_csv("data/pairing_manifest.csv", index=False)
print(f"\n  Saved: data/pairing_manifest.csv ({len(manifest)} rows)")

# ----------------------------
# 7. Save paired expression matrices
# ----------------------------
np.save("data/X_ctrl_paired.npy", X_ctrl_paired)
np.save("data/X_stim_paired.npy", X_stim_paired)
print(f"  Saved: data/X_ctrl_paired.npy  shape={X_ctrl_paired.shape}")
print(f"  Saved: data/X_stim_paired.npy  shape={X_stim_paired.shape}")

# ----------------------------
# 8. Final count
# ----------------------------
print(f"\n  Total training pairs: {n_total_pairs}")
print(f"  Expected: ~{len(donors)} donors x {len(cell_types)} cell types = {n_possible}")
if n_total_pairs > 0:
    print("\nOVERALL: PASS")
else:
    print("\nOVERALL: FAIL — no paired data found")
