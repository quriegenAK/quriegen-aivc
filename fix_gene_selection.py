"""
fix_gene_selection.py — Fix gene selection by force-including JAK-STAT pathway genes
on top of variance-based HVG selection (n_top_genes=3000).
Saves result as data/kang2018_pbmc_fixed.h5ad.
"""
import anndata as ad
import scanpy as sc
import numpy as np

# 1. Load full dataset
print("Loading Kang 2018 PBMC data...")
adata = ad.read_h5ad("data/kang2018_pbmc.h5ad")
print(f"  Full dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")

# 2. Normalize and log transform
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# 3. Run HVG selection with n_top_genes=3000 (NOT subset=True yet)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
n_hvg_before = adata.var["highly_variable"].sum()
print(f"  HVGs selected by variance: {n_hvg_before}")

# 4. Define must-include pathway genes
must_include = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]

# 5. Get gene names
if "name" in adata.var.columns:
    gene_names = adata.var["name"].tolist()
else:
    gene_names = adata.var_names.tolist()

gene_name_set = set(gene_names)

# 6. Force set highly_variable = True for must_include genes
forced_count = 0
missing_genes = []
for gene in must_include:
    if gene in gene_name_set:
        # Find the index in var
        if "name" in adata.var.columns:
            mask = adata.var["name"] == gene
        else:
            mask = adata.var_names == gene
        if mask.any():
            was_hvg = adata.var.loc[mask, "highly_variable"].values[0]
            adata.var.loc[mask, "highly_variable"] = True
            if not was_hvg:
                forced_count += 1
                print(f"  FORCED: {gene} (was not in top 3000 HVGs)")
            else:
                print(f"  ALREADY HVG: {gene}")
    else:
        missing_genes.append(gene)
        print(f"  MISSING FROM DATASET: {gene}")

# 7. Apply the combined mask
n_total_selected = adata.var["highly_variable"].sum()
print(f"\n  Total genes after combining: {n_total_selected}")
print(f"  ({n_hvg_before} HVGs + {forced_count} forced pathway genes)")

adata = adata[:, adata.var["highly_variable"]].copy()

# 8. Verify must_include genes and compute fold change table
if "name" in adata.var.columns:
    final_gene_names = adata.var["name"].tolist()
else:
    final_gene_names = adata.var_names.tolist()
final_gene_set = set(final_gene_names)
gene_to_idx = {g: i for i, g in enumerate(final_gene_names)}

# Get expression matrix
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()

ctrl_mask = (adata.obs["label"] == "ctrl").values
stim_mask = (adata.obs["label"] == "stim").values
ctrl_expr = X[ctrl_mask, :]
stim_expr = X[stim_mask, :]

print(f"\n  Must-include genes present: ", end="")
present_count = sum(1 for g in must_include if g in final_gene_set)
print(f"{present_count}/{len(must_include)}")

print()
print("=" * 75)
print(f"  {'Gene':<10} {'Present':<10} {'Ctrl mean':<12} {'Stim mean':<12} {'Fold change'}")
print("=" * 75)
for gene in must_include:
    idx = gene_to_idx.get(gene)
    if idx is not None:
        ctrl_mean = ctrl_expr[:, idx].mean()
        stim_mean = stim_expr[:, idx].mean()
        if ctrl_mean > 0.001:
            fc = stim_mean / ctrl_mean
        else:
            fc = float("inf") if stim_mean > 0 else 1.0
        fc_str = f"{fc:.2f}x" if fc < 1000 else "INF"
        print(f"  {gene:<10} {'YES':<10} {ctrl_mean:<12.4f} {stim_mean:<12.4f} {fc_str}")
    else:
        print(f"  {gene:<10} {'NO':<10} {'—':<12} {'—':<12} {'—'}")

# 9. Save
outpath = "data/kang2018_pbmc_fixed.h5ad"
adata.write_h5ad(outpath)
print(f"\nSaved fixed AnnData to {outpath}")
print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

# 10. Pass/fail
print()
if missing_genes:
    print(f"OVERALL: FAIL — Missing genes from dataset: {', '.join(missing_genes)}")
elif present_count == len(must_include):
    print(f"OVERALL: PASS — All {present_count}/{len(must_include)} pathway genes present")
else:
    missing = [g for g in must_include if g not in final_gene_set]
    print(f"OVERALL: FAIL — Missing from final set: {', '.join(missing)}")
