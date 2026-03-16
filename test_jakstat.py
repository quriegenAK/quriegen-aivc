"""
test_jakstat.py — Check JAK-STAT pathway gene expression in ctrl vs stim.
Uses the fixed data with force-included pathway genes.
"""
import anndata as ad
import scanpy as sc
import numpy as np

# Load fixed dataset (both conditions, with forced pathway genes)
print("Loading fixed Kang 2018 PBMC data...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")

# Data is already normalized and log-transformed from fix_gene_selection.py
# Verify by checking for counts layer
if "counts" not in adata.layers:
    print("  Re-normalizing (counts layer not found)...")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

# Build gene name to index mapping
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()

gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# JAK-STAT pathway genes to check (expanded list including IFITM genes)
jakstat_genes = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]

# Get expression matrix (dense)
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()

# Split by condition
ctrl_mask = (adata.obs["label"] == "ctrl").values
stim_mask = (adata.obs["label"] == "stim").values

ctrl_expr = X[ctrl_mask, :]
stim_expr = X[stim_mask, :]

# Check each gene
print()
print("=" * 75)
print(f"  {'Gene':<10} {'In set':<10} {'Ctrl mean':<12} {'Stim mean':<12} {'Fold change':<12} {'Responder'}")
print("=" * 75)

found_count = 0
responder_count = 0
found_genes = []
responder_genes = []

for gene in jakstat_genes:
    idx = gene_to_idx.get(gene)
    in_set = idx is not None
    if in_set:
        found_count += 1
        found_genes.append(gene)
        ctrl_mean = ctrl_expr[:, idx].mean()
        stim_mean = stim_expr[:, idx].mean()
        # Avoid division by zero
        if ctrl_mean > 0.001:
            fold_change = stim_mean / ctrl_mean
        else:
            fold_change = float("inf") if stim_mean > 0 else 1.0

        is_responder = fold_change > 2.0
        if is_responder:
            responder_count += 1
            responder_genes.append(gene)

        fc_str = f"{fold_change:.2f}x" if fold_change < 1000 else "INF"
        resp_str = "YES **" if is_responder else "no"
        print(f"  {gene:<10} {'YES':<10} {ctrl_mean:<12.4f} {stim_mean:<12.4f} {fc_str:<12} {resp_str}")
    else:
        print(f"  {gene:<10} {'NO':<10} {'—':<12} {'—':<12} {'—':<12} {'—'}")

# Summary
print()
print("=" * 75)
print("JAK-STAT BIOLOGY SUMMARY (FIXED GENE SET)")
print("=" * 75)
print(f"  JAK-STAT genes in gene set: {found_count}/{len(jakstat_genes)}")
print(f"    Found: {', '.join(found_genes) if found_genes else 'none'}")
print(f"  Confirmed responders (fold change > 2x): {responder_count}")
print(f"    Responders: {', '.join(responder_genes) if responder_genes else 'none'}")

# Pass/fail
print()
if found_count >= 13 and responder_count >= 5:
    print(f"OVERALL: PASS — {found_count}/{len(jakstat_genes)} genes present, {responder_count} confirmed responders")
elif found_count >= 10:
    print(f"OVERALL: PASS — {found_count}/{len(jakstat_genes)} genes present, {responder_count} responders")
elif found_count >= 5:
    print(f"OVERALL: PARTIAL — {found_count}/{len(jakstat_genes)} genes present")
else:
    print(f"OVERALL: FAIL — Only {found_count}/{len(jakstat_genes)} genes present")
