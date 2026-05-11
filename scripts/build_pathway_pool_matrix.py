"""Stage 3a — build the sparse pathway pool matrix from Report 3 outputs.

Per docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §3.5.

Reads data/pathway_annotations/gene_to_pathway_map.csv (produced by
scripts/prepare_pathway_annotations.py) and emits a sparse
(n_pathways, n_genes) row-normalized matrix as a .npz file. Downstream
PathwayAware*Decoders register this as a non-trainable buffer.

Row normalization: each row sums to 1.0, so the pathway score is the
mean expression over pathway member genes.

Inputs:
    --gene_to_pathway_csv  CSV with columns: gene_symbol, pathway_name, source_db
    --gene_universe        Optional: TSV listing all genes the decoder will see
                          (one gene per line). If absent, uses the union of
                          genes appearing in the gene_to_pathway_csv.

Outputs:
    --output_npz          sparse .npz with the pool matrix
    --output_meta_json    JSON with pathway names, gene order, dims

Usage:
    python3 scripts/build_pathway_pool_matrix.py \
        --gene_to_pathway_csv data/pathway_annotations/gene_to_pathway_map.csv \
        --output_npz          data/pathway_annotations/pathway_pool_matrix.npz \
        --output_meta_json    data/pathway_annotations/pathway_pool_meta.json

Sanity printed on stdout:
  - Total pathways (expected: 58 = 50 hallmark + 8 KEGG)
  - Total unique genes (Report 3 reported 4,798)
  - Matrix shape + nnz
  - Min/max/median pathway size
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_pool_matrix(
    gene_to_pathway_csv: Path,
    gene_universe: list[str] | None = None,
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Build the pool matrix.

    Returns:
        pool:          (n_pathways, n_genes) sparse CSR, row-normalized
        pathway_names: list of pathway names (length = n_pathways)
        gene_names:    list of gene symbols (length = n_genes), matching
                       column order of `pool`
    """
    print(f"Reading gene→pathway map: {gene_to_pathway_csv}")
    df = pd.read_csv(gene_to_pathway_csv)
    required_cols = {"gene_symbol", "pathway_name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns {required_cols}; got {list(df.columns)}"
        )
    print(f"  rows: {len(df)}")

    # Group genes by pathway
    pathway_to_genes: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        pathway_to_genes[row["pathway_name"]].add(row["gene_symbol"])

    pathway_names = sorted(pathway_to_genes.keys())
    if gene_universe is None:
        gene_names = sorted({g for genes in pathway_to_genes.values() for g in genes})
        print(f"  using gene universe from CSV: {len(gene_names)} unique genes")
    else:
        gene_names = list(gene_universe)
        print(f"  using provided gene universe: {len(gene_names)} genes")
    gene_to_col = {g: i for i, g in enumerate(gene_names)}

    rows, cols, vals = [], [], []
    pathway_sizes = []
    for r, pw in enumerate(pathway_names):
        members = [g for g in pathway_to_genes[pw] if g in gene_to_col]
        n_members = len(members)
        pathway_sizes.append(n_members)
        if n_members == 0:
            print(f"  WARN: pathway {pw!r} has 0 members in gene universe; skipping")
            continue
        weight = 1.0 / n_members  # row-normalized
        for g in members:
            rows.append(r)
            cols.append(gene_to_col[g])
            vals.append(weight)

    pool = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(len(pathway_names), len(gene_names)),
    )

    print(f"  pathway sizes: min={min(pathway_sizes)} median={int(np.median(pathway_sizes))} "
          f"max={max(pathway_sizes)} mean={np.mean(pathway_sizes):.1f}")
    return pool, pathway_names, gene_names


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene_to_pathway_csv", required=True, type=Path)
    p.add_argument("--gene_universe", type=Path, default=None,
                   help="Optional TSV listing all genes the decoder will see "
                        "(one per line). Default: union of CSV genes.")
    p.add_argument("--output_npz", required=True, type=Path)
    p.add_argument("--output_meta_json", required=True, type=Path)
    args = p.parse_args()

    gene_universe = None
    if args.gene_universe is not None:
        if not args.gene_universe.exists():
            raise FileNotFoundError(f"--gene_universe not found: {args.gene_universe}")
        gene_universe = pd.read_csv(args.gene_universe, sep="\t",
                                    header=None)[0].astype(str).tolist()

    pool, pathway_names, gene_names = build_pool_matrix(
        args.gene_to_pathway_csv, gene_universe=gene_universe,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.output_npz, pool)
    print(f"\nWrote {args.output_npz}")
    print(f"  shape: {pool.shape}")
    print(f"  nnz:   {pool.nnz}")
    print(f"  density: {100.0 * pool.nnz / (pool.shape[0] * pool.shape[1]):.4f}%")

    # Sanity-check row normalization
    row_sums = np.asarray(pool.sum(axis=1)).ravel()
    nonzero_rows = (row_sums > 0).sum()
    if nonzero_rows > 0:
        nz_sums = row_sums[row_sums > 0]
        print(f"  row sums (nonzero rows): min={nz_sums.min():.4f} "
              f"max={nz_sums.max():.4f} mean={nz_sums.mean():.4f} "
              f"(expected ≈ 1.0)")

    meta = {
        "n_pathways": pool.shape[0],
        "n_genes": pool.shape[1],
        "nnz": int(pool.nnz),
        "pathway_names": pathway_names,
        "gene_names": gene_names,
        "source_csv": str(args.gene_to_pathway_csv),
        "build_note": (
            "Sparse pool matrix for Stage 3 pathway-aware decoders. "
            "Each row is a pathway; columns are gene symbols. Values are "
            "1/n_members so the pathway score = mean expression over members."
        ),
    }
    args.output_meta_json.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.output_meta_json}")


if __name__ == "__main__":
    main()
