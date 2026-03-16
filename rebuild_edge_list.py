"""
rebuild_edge_list.py — Rebuild edge list using the fixed (expanded) HVG set.
Saves result as data/edge_list_fixed.csv.
"""
import anndata as ad
import pandas as pd

# 1. Load fixed data
print("Loading fixed AnnData...")
adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

# 2. Get gene names
if "name" in adata.var.columns:
    gene_names = set(adata.var["name"].tolist())
else:
    gene_names = set(adata.var_names.tolist())
print(f"  Gene count: {len(gene_names)}")

# 3. Load STRING protein info
print("Loading STRING protein info...")
info = pd.read_csv(
    "data/9606.protein.info.v12.0.txt",
    sep="\t",
    usecols=["#string_protein_id", "preferred_name"],
)
info.columns = ["protein_id", "gene_name"]
protein_to_gene = dict(zip(info["protein_id"], info["gene_name"]))
print(f"  Protein-to-gene mappings: {len(protein_to_gene)}")

# 4. Load STRING protein links
print("Loading STRING protein links...")
links = pd.read_csv(
    "data/9606.protein.links.v12.0.txt",
    sep=" ",
    usecols=["protein1", "protein2", "combined_score"],
)
print(f"  Total links: {len(links)}")

# 5. Map protein IDs to gene names
links["gene_a"] = links["protein1"].map(protein_to_gene)
links["gene_b"] = links["protein2"].map(protein_to_gene)
links = links.dropna(subset=["gene_a", "gene_b"])

# 6. Filter: combined_score >= 700 AND both genes in the fixed HVG set
mask = (
    (links["combined_score"] >= 700)
    & (links["gene_a"].isin(gene_names))
    & (links["gene_b"].isin(gene_names))
)
filtered = links.loc[mask, ["gene_a", "gene_b", "combined_score"]].reset_index(drop=True)

# 7. Save
filtered.to_csv("data/edge_list_fixed.csv", index=False)
print(f"\nSaved edge_list_fixed.csv with {len(filtered)} edges")

# Count JAK-STAT edges specifically
jakstat_genes = {
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IRF1", "MX1", "MX2", "ISG15",
    "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
}

jakstat_mask = (
    filtered["gene_a"].isin(jakstat_genes) | filtered["gene_b"].isin(jakstat_genes)
)
jakstat_edges = filtered[jakstat_mask]
n_jakstat_edges = len(jakstat_edges)

# Which JAK-STAT genes have at least one edge?
jakstat_with_edges = set()
jakstat_with_edges.update(jakstat_edges["gene_a"][jakstat_edges["gene_a"].isin(jakstat_genes)].tolist())
jakstat_with_edges.update(jakstat_edges["gene_b"][jakstat_edges["gene_b"].isin(jakstat_genes)].tolist())

print(f"  Edges involving JAK-STAT genes: {n_jakstat_edges}")
print(f"  JAK-STAT genes with edges: {len(jakstat_with_edges)}/{len(jakstat_genes)}")
print(f"    Connected: {', '.join(sorted(jakstat_with_edges))}")

jakstat_no_edges = jakstat_genes - jakstat_with_edges
if jakstat_no_edges:
    print(f"    No edges: {', '.join(sorted(jakstat_no_edges))}")

# 8. Pass/fail
print()
if len(filtered) > 5000 and len(jakstat_with_edges) >= 1:
    print(f"OVERALL: PASS — {len(filtered)} edges, {n_jakstat_edges} JAK-STAT edges, {len(jakstat_with_edges)} pathway genes connected")
elif len(filtered) <= 5000:
    print(f"OVERALL: FAIL — Only {len(filtered)} edges (expected > 5000)")
else:
    print(f"OVERALL: FAIL — No JAK-STAT genes have edges in the graph")
