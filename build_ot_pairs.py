"""
build_ot_pairs.py — Optimal Transport cell pairing for perturbation prediction.

Replaces pseudo-bulk averaging with per-cell OT matching.
For each (donor, cell_type) group, matches each ctrl cell to its nearest
stim cell using Wasserstein optimal transport in PCA-reduced space.

Result: 1,000+ individual cell pairs instead of 60 pseudo-bulk averages.

Outputs:
    data/X_ctrl_ot.npy           — (N_pairs, 3010) ctrl expression
    data/X_stim_ot.npy           — (N_pairs, 3010) stim expression
    data/cell_type_ot.npy        — (N_pairs,) cell type labels
    data/donor_ot.npy            — (N_pairs,) donor labels
    data/ot_pairing_manifest.csv — pairing log
    data/ot_pairing_log.txt      — detailed log
"""
import random
import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# OT library import with fallback
# ----------------------------
USE_OT = False
try:
    import ot
    USE_OT = True
    print("OT library: POT loaded successfully")
except ImportError:
    print("WARNING: POT library not available. Using greedy nearest-neighbour fallback.")
    print("  Install with: pip install POT")

# ----------------------------
# 1. Load data
# ----------------------------
print("\nLoading data/kang2018_pbmc_fixed.h5ad...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Check normalisation state
already_normalised = "log1p" in adata.uns
if already_normalised:
    print("  Data is ALREADY normalised (log1p in adata.uns). Will skip renormalisation.")
else:
    print("  Data is NOT normalised. Will normalise after pairing.")

# Gene names
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()
n_genes = len(gene_names)
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# Expression matrix (dense)
X_all = adata.X
if hasattr(X_all, "toarray"):
    X_all = X_all.toarray()

labels = adata.obs["label"].values
donor_col = "replicate"
celltype_col = "cell_type"
donor_ids = adata.obs[donor_col].values
cell_type_ids = adata.obs[celltype_col].values

donors = sorted(np.unique(donor_ids).tolist())
cell_types = sorted(np.unique(cell_type_ids).tolist())
print(f"  Donors ({len(donors)}): {donors}")
print(f"  Cell types ({len(cell_types)}): {cell_types}")

# JAK-STAT genes for monitoring
jakstat_genes = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]
ifit1_idx = gene_to_idx.get("IFIT1")

# ----------------------------
# 2. OT pairing per group
# ----------------------------
print(f"\n{'='*80}")
print("OPTIMAL TRANSPORT CELL PAIRING")
print(f"{'='*80}")

X_ctrl_list = []
X_stim_list = []
ct_label_list = []
donor_label_list = []
log_rows = []
log_text_lines = []

MAX_CELLS = 500
MIN_CELLS = 5
PCA_COMPONENTS = 50

for donor in donors:
    for ct in cell_types:
        mask_ctrl = (donor_ids == donor) & (cell_type_ids == ct) & (labels == "ctrl")
        mask_stim = (donor_ids == donor) & (cell_type_ids == ct) & (labels == "stim")

        ctrl_idx = np.where(mask_ctrl)[0]
        stim_idx = np.where(mask_stim)[0]
        n_ctrl = len(ctrl_idx)
        n_stim = len(stim_idx)

        # Skip if insufficient cells
        if n_ctrl < MIN_CELLS or n_stim < MIN_CELLS:
            msg = f"Skipped {donor} {ct}: insufficient cells (ctrl={n_ctrl}, stim={n_stim})"
            log_text_lines.append(msg)
            print(f"  {msg}")
            log_rows.append({
                "donor_id": donor, "cell_type": ct,
                "n_ctrl": n_ctrl, "n_stim": n_stim,
                "n_pairs": 0, "method": "skipped",
                "reason": "insufficient cells",
            })
            continue

        # Subsample if too large
        ctrl_subsampled = False
        stim_subsampled = False
        if n_ctrl > MAX_CELLS:
            np.random.seed(SEED)
            ctrl_idx = np.random.choice(ctrl_idx, MAX_CELLS, replace=False)
            ctrl_subsampled = True
        if n_stim > MAX_CELLS:
            np.random.seed(SEED + 1)
            stim_idx = np.random.choice(stim_idx, MAX_CELLS, replace=False)
            stim_subsampled = True

        if ctrl_subsampled or stim_subsampled:
            msg = f"Subsampled {donor} {ct}: ctrl {n_ctrl}->{len(ctrl_idx)}, stim {n_stim}->{len(stim_idx)}"
            log_text_lines.append(msg)

        # Get expression data
        X_ctrl_group = X_all[ctrl_idx]  # (n_ctrl, n_genes)
        X_stim_group = X_all[stim_idx]  # (n_stim, n_genes)

        # PCA for OT distance computation
        n_c = X_ctrl_group.shape[0]
        n_s = X_stim_group.shape[0]
        combined = np.vstack([X_ctrl_group, X_stim_group])
        n_pcs = min(PCA_COMPONENTS, combined.shape[0] - 1, combined.shape[1])
        pca = PCA(n_components=n_pcs, random_state=SEED)
        combined_pca = pca.fit_transform(combined)
        ctrl_pca = combined_pca[:n_c]
        stim_pca = combined_pca[n_c:]

        # Compute cost matrix
        cost_matrix = cdist(ctrl_pca, stim_pca, metric="euclidean")
        cost_max = cost_matrix.max()
        if cost_max > 0:
            cost_norm = cost_matrix / cost_max
        else:
            cost_norm = cost_matrix

        # Solve OT or use greedy fallback
        method_used = "greedy_nn"
        if USE_OT:
            try:
                a = np.ones(n_c) / n_c
                b = np.ones(n_s) / n_s
                transport_plan = ot.emd(a, b, cost_norm)
                # Extract hard pairs: for each ctrl cell, find best stim match
                stim_assignments = transport_plan.argmax(axis=1)
                method_used = "ot_emd"
            except Exception as e:
                log_text_lines.append(
                    f"OT failed for {donor} {ct}: {type(e).__name__}: {e}. "
                    f"Using greedy NN fallback."
                )
                # Greedy nearest-neighbour fallback
                stim_assignments = cost_matrix.argmin(axis=1)
                method_used = "greedy_nn"
        else:
            stim_assignments = cost_matrix.argmin(axis=1)
            method_used = "greedy_nn"

        # Store pairs in original expression space
        for i in range(n_c):
            j = stim_assignments[i]
            X_ctrl_list.append(X_ctrl_group[i])
            X_stim_list.append(X_stim_group[j])
            ct_label_list.append(ct)
            donor_label_list.append(donor)

        n_pairs = n_c
        log_rows.append({
            "donor_id": donor, "cell_type": ct,
            "n_ctrl": n_ctrl, "n_stim": n_stim,
            "n_pairs": n_pairs, "method": method_used,
            "reason": "paired",
        })
        print(f"  {donor:<16} {ct:<22} ctrl={len(ctrl_idx):<4} stim={len(stim_idx):<4} pairs={n_pairs:<4} ({method_used})")

# ----------------------------
# 3. Stack and save
# ----------------------------
if len(X_ctrl_list) == 0:
    print("\nFATAL: No OT pairs generated. Cannot continue.")
    sys.exit(1)

X_ctrl_ot = np.array(X_ctrl_list, dtype=np.float32)
X_stim_ot = np.array(X_stim_list, dtype=np.float32)
cell_type_ot = np.array(ct_label_list)
donor_ot = np.array(donor_label_list)

print(f"\n{'='*80}")
print("SAVING OT PAIRED DATA")
print(f"{'='*80}")

np.save("data/X_ctrl_ot.npy", X_ctrl_ot)
np.save("data/X_stim_ot.npy", X_stim_ot)
np.save("data/cell_type_ot.npy", cell_type_ot)
np.save("data/donor_ot.npy", donor_ot)

print(f"  data/X_ctrl_ot.npy      shape={X_ctrl_ot.shape}")
print(f"  data/X_stim_ot.npy      shape={X_stim_ot.shape}")
print(f"  data/cell_type_ot.npy   shape={cell_type_ot.shape}")
print(f"  data/donor_ot.npy       shape={donor_ot.shape}")

# Save manifest
ot_manifest = pd.DataFrame(log_rows)
ot_manifest.to_csv("data/ot_pairing_manifest.csv", index=False)
print(f"  data/ot_pairing_manifest.csv  rows={len(ot_manifest)}")

# Save log
with open("data/ot_pairing_log.txt", "w") as f:
    f.write("Optimal Transport Cell Pairing Log\n")
    f.write(f"Method: {'OT (EMD)' if USE_OT else 'Greedy NN (fallback)'}\n")
    f.write(f"Data normalised: {already_normalised}\n")
    f.write(f"PCA components: {PCA_COMPONENTS}\n")
    f.write(f"Max cells per group: {MAX_CELLS}\n")
    f.write(f"Min cells threshold: {MIN_CELLS}\n\n")
    for line in log_text_lines:
        f.write(line + "\n")
print("  data/ot_pairing_log.txt")

# ----------------------------
# 4. Pairing report
# ----------------------------
print(f"\n{'='*80}")
print("PAIRING REPORT")
print(f"{'='*80}")
total_pairs = X_ctrl_ot.shape[0]
print(f"  Total OT pairs: {total_pairs}")
print(f"  Week 2 pseudo-bulk pairs: 60")
print(f"  Increase: {total_pairs / 60:.1f}x")

print(f"\n  Per (donor, cell_type) breakdown:")
print(f"  {'Donor':<16} {'Cell Type':<22} {'Pairs':<8} {'Method'}")
print(f"  {'-'*60}")
for _, row in ot_manifest.iterrows():
    if row["n_pairs"] > 0:
        print(f"  {row['donor_id']:<16} {row['cell_type']:<22} {row['n_pairs']:<8} {row['method']}")

# IFIT1 distribution in stim pairs
if ifit1_idx is not None:
    ctrl_ifit1 = X_ctrl_ot[:, ifit1_idx]
    stim_ifit1 = X_stim_ot[:, ifit1_idx]
    # Fold change: stim / max(ctrl, eps)
    eps = 1e-6
    fc_ifit1 = (stim_ifit1 + eps) / (ctrl_ifit1 + eps)

    print(f"\n  IFIT1 fold change distribution in stim pairs:")
    print(f"    Min:     {fc_ifit1.min():.2f}x")
    print(f"    Median:  {np.median(fc_ifit1):.2f}x")
    print(f"    90th %:  {np.percentile(fc_ifit1, 90):.2f}x")
    print(f"    Max:     {fc_ifit1.max():.2f}x")
    print(f"    Mean:    {fc_ifit1.mean():.2f}x")

    # How many cells show > 5x fold change
    n_high = (fc_ifit1 > 5.0).sum()
    print(f"    Cells with > 5x FC: {n_high}/{total_pairs} ({100*n_high/total_pairs:.1f}%)")

# ----------------------------
# 5. Pairing quality validation
# ----------------------------
print(f"\n{'='*80}")
print("PAIRING QUALITY VALIDATION")
print(f"{'='*80}")

for ct in cell_types:
    ct_mask = cell_type_ot == ct
    n_ct = ct_mask.sum()
    if n_ct < 2:
        continue

    ctrl_ct = X_ctrl_ot[ct_mask]
    stim_ct = X_stim_ot[ct_mask]

    # Paired cosine similarity
    from numpy.linalg import norm
    paired_cos = []
    for i in range(min(n_ct, 200)):
        c = ctrl_ct[i]
        s = stim_ct[i]
        cn = norm(c)
        sn = norm(s)
        if cn > 0 and sn > 0:
            paired_cos.append(np.dot(c, s) / (cn * sn))
    paired_mean = np.mean(paired_cos) if paired_cos else 0

    # Random baseline
    np.random.seed(SEED)
    rand_cos = []
    rand_idx = np.random.permutation(min(n_ct, 200))
    for i in range(len(rand_idx)):
        c = ctrl_ct[i]
        s = stim_ct[rand_idx[i]]
        cn = norm(c)
        sn = norm(s)
        if cn > 0 and sn > 0:
            rand_cos.append(np.dot(c, s) / (cn * sn))
    rand_mean = np.mean(rand_cos) if rand_cos else 0

    improvement = paired_mean - rand_mean
    status = "OK" if improvement > 0.01 else "WARN"
    print(f"  {ct:<22} paired_cos={paired_mean:.4f}  random_cos={rand_mean:.4f}  delta={improvement:+.4f}  [{status}]")

    if improvement <= 0.01:
        log_text_lines.append(f"WARNING: {ct} OT pairing quality marginal (delta={improvement:+.4f})")

# Check total pair count
if total_pairs < 200:
    print(f"\n  WARNING: Only {total_pairs} OT pairs generated (< 200 threshold).")
    print("  Pseudo-bulk fallback may be needed for stable training.")
elif total_pairs < 500:
    print(f"\n  CAUTION: {total_pairs} OT pairs is moderate. Training may benefit from more data.")
else:
    print(f"\n  {total_pairs} OT pairs generated. Sufficient for mini-batch training.")

print(f"\nOVERALL: {'PASS' if total_pairs >= 200 else 'WARNING — low pair count'}")
print(f"  OT pairing method: {'EMD (optimal transport)' if USE_OT else 'Greedy nearest-neighbour'}")
