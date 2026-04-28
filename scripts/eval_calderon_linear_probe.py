"""CLI: run Calderon linear-probe eval against on-disk artifacts.

Inputs:
- Calderon hg38 h5ad (with cell_type, donor in .obs)
- Projection matrix .npz (n_calderon × n_dogma)

Output:
- metrics.json with per-fold + aggregate accuracy/f1.

Mock encoder used for scaffold; replace with real checkpoint in PR #41.

Usage:
    python scripts/eval_calderon_linear_probe.py \\
        --calderon data/calderon2019/calderon_atac_hg38.h5ad \\
        --projection data/calderon2019/calderon_to_dogma_lll_M.npz \\
        --out data/calderon2019/probe_metrics_mock.json \\
        --latent-dim 64 \\
        --cv leave-one-donor-out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import scipy.sparse as sp

from aivc.eval.calderon_probe import (
    MockEncoder,
    encode_samples,
    project_calderon_to_dogma_space,
    run_linear_probe,
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--calderon", required=True, type=Path)
    p.add_argument("--projection", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--cv", choices=["leave-one-donor-out", "stratified-kfold"],
                   default="leave-one-donor-out")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Used only when --cv=stratified-kfold")
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--group-col", default="donor")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading Calderon: {args.calderon}")
    calderon = ad.read_h5ad(args.calderon)
    print(f"  shape: {calderon.shape}")

    print(f"Loading projection: {args.projection}")
    M = sp.load_npz(args.projection)
    print(f"  shape: {M.shape}, nnz: {M.nnz}")

    # Project
    X_dogma_space = project_calderon_to_dogma_space(calderon.X, M)
    print(f"Projected: {X_dogma_space.shape}, nnz: {X_dogma_space.nnz}")

    # Encode (mock)
    encoder = MockEncoder(n_peaks=X_dogma_space.shape[1], latent_dim=args.latent_dim, seed=args.seed)
    embeddings = encode_samples(X_dogma_space, encoder)
    print(f"Embeddings: {embeddings.shape}")

    # Probe
    labels = calderon.obs[args.label_col].astype(str).values
    if args.cv == "leave-one-donor-out":
        groups = calderon.obs[args.group_col].astype(str).values
        metrics = run_linear_probe(embeddings, labels, groups=groups, random_state=args.seed)
    else:
        metrics = run_linear_probe(embeddings, labels, cv_folds=args.cv_folds, random_state=args.seed)

    metrics["encoder"] = "MockEncoder"
    metrics["latent_dim"] = args.latent_dim
    metrics["cv_strategy"] = args.cv
    metrics["n_samples"] = int(embeddings.shape[0])

    print(f"\nMean accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
    print(f"Mean F1 macro: {metrics['mean_f1_macro']:.4f}")
    print(f"N classes: {metrics['n_classes']}")
    print(f"Chance accuracy (1/n_classes): {1.0/metrics['n_classes']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
