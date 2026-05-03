"""Re-encode an h5ad with sparse CSR for obsm matrices.

Phase 6.5g.2 path-X labeled h5ads ballooned to 41 GB joint because
`obsm['atac_peaks']` was being stored DENSE float32. The matrix is
~95% zero — proper sparse CSR storage gives ~7× compression even
without gzip, ~30× with gzip.

BSC compatibility: defaults to `compression=None` per the project
memory (system HDF5 + offline-installed h5py wheel mismatch on
compressed sparse-group attributes). `--compression gzip` available
as opt-in; verify on a BSC compute node before relying on it for
training-time loads.

Usage:
    python scripts/reencode_h5ad_sparse.py \\
        --in_h5ad data/phase6_5g_2/dogma_h5ads/dogma_joint_labeled.h5ad \\
        --out_h5ad data/phase6_5g_2/dogma_h5ads/dogma_joint_labeled_sparse.h5ad

Or in-place:
    python scripts/reencode_h5ad_sparse.py \\
        --in_h5ad <path>.h5ad \\
        --out_h5ad <path>.h5ad \\
        --inplace
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


# obsm keys to convert to sparse CSR if they are dense ndarrays
SPARSIFY_OBSM_KEYS = ("atac_peaks",)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--in_h5ad", required=True, type=Path)
    p.add_argument("--out_h5ad", required=True, type=Path)
    p.add_argument("--compression", default="none",
                   choices=["none", "gzip"],
                   help="HDF5 compression. Default 'none' for BSC compat.")
    p.add_argument("--inplace", action="store_true",
                   help="Allow in_h5ad == out_h5ad (writes to a temp then renames)")
    p.add_argument("--obsm_keys", default=",".join(SPARSIFY_OBSM_KEYS),
                   help=f"Comma-separated obsm keys to sparsify (default: {','.join(SPARSIFY_OBSM_KEYS)})")
    args = p.parse_args()

    in_path = args.in_h5ad.resolve()
    out_path = args.out_h5ad.resolve()
    if in_path == out_path and not args.inplace:
        raise ValueError(
            "in_h5ad == out_h5ad. Pass --inplace to confirm overwrite intent."
        )

    print(f"Loading {in_path}")
    print(f"  size on disk: {in_path.stat().st_size / 1e9:.2f} GB")
    adata = ad.read_h5ad(in_path)
    print(f"  shape: {adata.shape}")

    sparsify_keys = [k.strip() for k in args.obsm_keys.split(",") if k.strip()]
    print(f"\n=== obsm sparsification ===")
    for key in sparsify_keys:
        if key not in adata.obsm:
            print(f"  {key}: NOT PRESENT — skipping")
            continue
        mat = adata.obsm[key]
        if sp.issparse(mat):
            print(f"  {key}: already sparse ({type(mat).__name__}, "
                  f"nnz={mat.nnz}, density={mat.nnz / (mat.shape[0] * mat.shape[1]):.4f})"
                  f" — leaving alone")
            continue
        # Dense path
        n_total = mat.size
        n_zero = int((np.asarray(mat) == 0).sum())
        density = 1 - n_zero / n_total
        print(f"  {key}: dense ndarray shape={mat.shape}, density={density:.4f}, "
              f"converting to CSR...")
        adata.obsm[key] = sp.csr_matrix(mat)
        print(f"    nnz={adata.obsm[key].nnz}, "
              f"sparse memory ~ {(adata.obsm[key].data.nbytes + adata.obsm[key].indices.nbytes + adata.obsm[key].indptr.nbytes) / 1e9:.2f} GB")

    # Also sparsify X if it's dense and largely zero
    if not sp.issparse(adata.X):
        x_density = 1 - int((np.asarray(adata.X) == 0).sum()) / adata.X.size
        if x_density < 0.5:
            print(f"\n  X: dense, density={x_density:.4f} — converting to CSR")
            adata.X = sp.csr_matrix(adata.X)

    # --- Write ---
    compression = None if args.compression == "none" else args.compression
    if in_path == out_path:
        tmp_path = out_path.with_suffix(".tmp.h5ad")
        print(f"\nWriting (in-place) to temp: {tmp_path} (compression={compression})")
        adata.write_h5ad(tmp_path, compression=compression)
        del adata
        tmp_path.replace(out_path)
        print(f"Renamed temp to {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting {out_path} (compression={compression})")
        adata.write_h5ad(out_path, compression=compression)

    print(f"\n=== Size comparison ===")
    in_size = in_path.stat().st_size / 1e9
    out_size = out_path.stat().st_size / 1e9
    if in_path != out_path:
        print(f"  input:  {in_size:.2f} GB")
        print(f"  output: {out_size:.2f} GB")
        print(f"  reduction: {100 * (1 - out_size / in_size):.1f}%")
    else:
        print(f"  final:  {out_size:.2f} GB (was {in_size:.2f} GB before re-encode)")


if __name__ == "__main__":
    main()
