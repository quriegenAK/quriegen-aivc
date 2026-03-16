"""
Build edge_list.csv: STRING PPI edges filtered to top 2000 HVGs from Kang 2018 PBMC data.
"""

import scanpy as sc
import pandas as pd
import anndata as ad

# 1. Load Kang 2018 data
print("Loading Kang 2018 PBMC data...")
adata = ad.read_h5ad("data/kang2018_pbmc.h5ad")
print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

# 2. Normalize and log transform
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# 3. Select top 2000 HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
hvg_names = set(adata.var_names[adata.var.highly_variable].tolist())
print(f"  Selected {len(hvg_names)} HVGs")

# Also check if var has a 'name' column with gene symbols
if "name" in adata.var.columns:
    # Map var_names to gene symbols
    hvg_symbols = set(adata.var.loc[adata.var.highly_variable, "name"].tolist())
    print(f"  HVG symbols (from 'name' column): {len(hvg_symbols)}")
else:
    hvg_symbols = hvg_names
    print("  Using var_names as gene symbols (no 'name' column)")

# Use both var_names and symbol names for matching
all_hvg_ids = hvg_names | hvg_symbols

# 4. Load STRING protein info (maps protein IDs to gene names)
print("Loading STRING protein info...")
info = pd.read_csv(
    "data/9606.protein.info.v12.0.txt",
    sep="\t",
    usecols=["#string_protein_id", "preferred_name"],
)
info.columns = ["protein_id", "gene_name"]
protein_to_gene = dict(zip(info["protein_id"], info["gene_name"]))
print(f"  Loaded {len(protein_to_gene)} protein-to-gene mappings")

# 5. Load STRING protein links
print("Loading STRING protein links (this may take a moment)...")
links = pd.read_csv(
    "data/9606.protein.links.v12.0.txt",
    sep=" ",
    usecols=["protein1", "protein2", "combined_score"],
)
print(f"  Loaded {len(links)} total protein links")

# 6. Map protein IDs to gene names
links["gene_a"] = links["protein1"].map(protein_to_gene)
links["gene_b"] = links["protein2"].map(protein_to_gene)

# Drop unmapped
links = links.dropna(subset=["gene_a", "gene_b"])
print(f"  After mapping to gene names: {len(links)} links")

# 7. Filter: combined_score >= 700 AND both genes in HVG list
mask = (
    (links["combined_score"] >= 700)
    & (links["gene_a"].isin(all_hvg_ids))
    & (links["gene_b"].isin(all_hvg_ids))
)
filtered = links.loc[mask, ["gene_a", "gene_b", "combined_score"]].reset_index(drop=True)
print(f"  After filtering (score >= 700, both in HVGs): {len(filtered)} edges")

# 8. Save
filtered.to_csv("data/edge_list.csv", index=False)
print(f"\nSaved edge_list.csv with {len(filtered)} edges")
print("\nFirst 5 rows:")
print(filtered.head())
