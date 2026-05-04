"""Pivot A: Calderon linear-probe eval through the JOINT FUSION pathway.

Forwards Calderon ATAC through the full encoder stack (with zero-padded
RNA + Protein) and runs linear probe on z_supcon — the SAME joint
embedding SupCon was trained on at PR #54c.

Hypothesis: SupCon shaped the joint embedding (`z_supcon`), not the
per-modality latents. The existing `eval_calderon_linear_probe.py`
forwards through `PeakLevelATACEncoder` alone, missing the cross-modal
alignment. If joint-fusion eval lifts kfold accuracy meaningfully above
the bare-ATAC eval (locked at 0.1943 from job 39961124 / 39964435),
the encoder is fine and the bare-ATAC eval was the wrong target.

Usage:
    python scripts/eval_calderon_joint_fusion.py \\
        --calderon data/calderon2019/calderon_atac_hg38.h5ad \\
        --projection data/calderon2019/calderon_to_dogma_union_M.npz \\
        --encoder_ckpt /path/to/pretrain_encoders.pt \\
        --out results/calderon_joint_fusion_kfold.json \\
        --cv stratified-kfold

Outcome decision tree (from docs/reports/phase_6_5g_3_pivot_A_scope.md):
    >= 0.70 -> threshold passes via joint pathway (re-evaluate framing)
    0.50-0.70 -> joint preserves transfer signal; below threshold but
                 informative for Pivot B/C choice
    0.30-0.50 -> modest lift; both encoder + fusion need work (Pivot C)
    < 0.30   -> joint also fails to transfer; Pivot B (PeakVI)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import scipy.sparse as sp

from aivc.eval.calderon_probe import (
    encode_samples_via_joint_fusion,
    load_full_encoders_from_ckpt,
    project_calderon_to_dogma_space,
    run_linear_probe,
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--calderon", required=True, type=Path)
    p.add_argument("--projection", type=Path,
                   default=Path("data/calderon2019/calderon_to_dogma_union_M.npz"),
                   help="Sparse projection M (n_calderon × n_dogma_union)")
    p.add_argument("--encoder_ckpt", required=True, type=Path,
                   help="Pretrain checkpoint with all 3 encoders + 3 projections")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cv", choices=["leave-one-donor-out", "stratified-kfold"],
                   default="stratified-kfold")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--group-col", default="donor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda",
                   help="cpu or cuda. CPU works (175 cells); GPU is faster.")
    args = p.parse_args()

    print(f"Loading Calderon: {args.calderon}")
    calderon = ad.read_h5ad(args.calderon)
    print(f"  shape: {calderon.shape}")

    print(f"Loading projection: {args.projection}")
    M = sp.load_npz(args.projection)
    print(f"  shape: {M.shape}, nnz: {M.nnz}")

    X_dogma_space = project_calderon_to_dogma_space(calderon.X, M)
    print(f"Projected: {X_dogma_space.shape}, nnz: {X_dogma_space.nnz}")

    print(f"\nLoading FULL encoder stack from: {args.encoder_ckpt}")
    encoders = load_full_encoders_from_ckpt(
        args.encoder_ckpt,
        expected_n_peaks=X_dogma_space.shape[1],
        map_location=args.device,
    )
    print(f"  n_genes={encoders['n_genes']} n_peaks={encoders['n_peaks']} "
          f"n_proteins={encoders['n_proteins']} proj_dim={encoders['proj_dim']}")

    print(f"\nForwarding Calderon ATAC through joint-fusion pipeline "
          f"(zero-padded RNA + Protein)...")
    embeddings = encode_samples_via_joint_fusion(
        X_dogma_space, encoders, device=args.device,
    )
    print(f"z_supcon embeddings: {embeddings.shape}")

    labels = calderon.obs[args.label_col].astype(str).values
    if args.cv == "leave-one-donor-out":
        groups = calderon.obs[args.group_col].astype(str).values
        metrics = run_linear_probe(embeddings, labels, groups=groups,
                                   random_state=args.seed)
    else:
        metrics = run_linear_probe(embeddings, labels, cv_folds=args.cv_folds,
                                   random_state=args.seed)

    metrics["encoder"] = "joint_fusion_z_supcon"
    metrics["encoder_provenance"] = {
        "ckpt_path": str(args.encoder_ckpt),
        "n_genes": encoders["n_genes"],
        "n_peaks": encoders["n_peaks"],
        "n_proteins": encoders["n_proteins"],
        "proj_dim": encoders["proj_dim"],
        "ckpt_arm": encoders["config"].get("arm"),
        "n_lysis_categories": encoders["config"].get("n_lysis_categories", 0),
        "fusion_formula": "L2(mean(L2(z_rna)+L2(z_atac)+L2(z_protein)))",
        "calderon_inference": "ATAC-only; RNA + Protein zero-padded",
    }
    metrics["latent_dim"] = encoders["proj_dim"]
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
