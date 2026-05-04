"""Test 1: pseudo-bulk DOGMA centroid vs Calderon bulk compatibility check.

Validates whether the encoder captures cell-type biology at the BULK level
(by pseudo-bulking DOGMA per Azimuth lineage) — separating the model
question from the evaluation question.

If Calderon samples cluster meaningfully with their mapped DOGMA lineage
centroids in encoder space, the encoder is fine and the bare-ATAC eval
(0.1943 on 25-class Calderon) was confounded by sample sparsity / class
granularity / domain shift between single-cell and bulk regimes.

Pipeline:
  1. Load dogma_lll_union_labeled.h5ad (paper-grade Azimuth labels)
  2. Aggregate ATAC counts per cell_type (sum) → 8 pseudo-bulk profiles
  3. Load Calderon h5ad + union M projection
  4. Project Calderon → DOGMA peak space (existing pipeline)
  5. Map Calderon's 25 cell_types → 6 mappable lineages; drop unmappable
  6. Encode BOTH (pseudo-bulk DOGMA + projected Calderon) through bare
     PeakLevelATACEncoder
  7. Compute cosine similarity matrix (n_calderon, 8)
  8. Top-1 NN: argmax per row → predicted lineage
  9. Accuracy = matches mapped lineage / total mappable

Decision matrix:
  ≥ 0.50  → encoder captures biology at bulk; eval was the problem
  0.20-0.50 → mixed; rare-lineage performance varies
  < 0.20  → encoder genuinely doesn't transfer; proceed to Test 2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# Calderon 25 → 6 lineage map. Token-based (split on non-word) for clean
# word-boundary matching — avoids false hits like 'unKNown' → NK.
# Order: CD4_T, CD8_T, B, NK, Monocyte, DC. Unmappable → None (dropped).
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def map_to_lineage(label: str) -> str | None:
    """Return mapped lineage name, or None if unmappable (will be dropped)."""
    tokens = set(t for t in _TOKEN_SPLIT_RE.split(label.lower()) if t)

    # CD4 T (check FIRST so cd4-containing labels don't fall through to others)
    if any(t.startswith("cd4") for t in tokens):
        return "CD4_T"
    if any(t in {"tfh", "th1", "th2", "th17", "treg", "follicular",
                  "regulatory"} for t in tokens):
        return "CD4_T"

    # CD8 T
    if any(t.startswith("cd8") for t in tokens):
        return "CD8_T"
    if "cytotoxic" in tokens:
        return "CD8_T"

    # B cell
    if any(t in {"b", "plasmablast", "plasmacells"} for t in tokens):
        return "B"
    if "bulk" in tokens and "b" in tokens:
        return "B"  # "Bulk_B" → tokens contains both

    # NK
    if "nk" in tokens:
        return "NK"

    # Monocyte
    if any("mono" in t for t in tokens):
        return "Monocyte"

    # DC
    if any(t in {"dc", "pdc", "mdc", "cdc"} for t in tokens):
        return "DC"
    if any("dendritic" in t for t in tokens):
        return "DC"

    return None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dogma_labeled", required=True, type=Path,
                   help="dogma_lll_union_labeled.h5ad with obs['cell_type']")
    p.add_argument("--calderon", required=True, type=Path,
                   help="calderon_atac_hg38.h5ad")
    p.add_argument("--projection", required=True, type=Path,
                   help="calderon_to_dogma_union_M.npz")
    p.add_argument("--encoder_ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="JSON output: similarity matrix + accuracy + provenance")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--label_col", default="cell_type")
    args = p.parse_args()

    import torch
    from aivc.eval.calderon_probe import (
        encode_samples,
        load_atac_encoder_from_ckpt,
        project_calderon_to_dogma_space,
    )

    # --- 1. Load DOGMA labeled h5ad ---
    print(f"Loading DOGMA: {args.dogma_labeled}")
    dogma = ad.read_h5ad(args.dogma_labeled)
    print(f"  shape: {dogma.shape}")
    if "atac_peaks" not in dogma.obsm:
        raise ValueError(f"obsm['atac_peaks'] missing from DOGMA h5ad")
    atac = dogma.obsm["atac_peaks"]
    if not sp.issparse(atac):
        atac = sp.csr_matrix(atac)
    labels = dogma.obs[args.label_col].astype(str).values
    classes = sorted(set(labels))
    print(f"  cell_type classes: {classes}")
    print(f"  per-class counts: {pd.Series(labels).value_counts().to_dict()}")

    # --- 2. Aggregate per cell_type → pseudo-bulk profiles ---
    print(f"\nAggregating pseudo-bulk per cell_type (sum)...")
    centroids_X = []
    centroid_labels = []
    for cls in classes:
        mask = labels == cls
        n = int(mask.sum())
        # Sum counts across cells of this class → 1 row of (n_peaks,)
        bulk_row = np.asarray(atac[mask].sum(axis=0)).ravel()
        centroids_X.append(bulk_row)
        centroid_labels.append(cls)
        print(f"  {cls:<12s}: aggregated from {n:>5d} cells, "
              f"sum_counts={bulk_row.sum():.0f}, "
              f"nnz={int((bulk_row > 0).sum())}")
    centroids_X = sp.csr_matrix(np.vstack(centroids_X))
    print(f"  pseudo-bulk shape: {centroids_X.shape}")

    # --- 3. Load Calderon + projection ---
    print(f"\nLoading Calderon: {args.calderon}")
    calderon = ad.read_h5ad(args.calderon)
    print(f"  shape: {calderon.shape}")
    print(f"\nLoading projection: {args.projection}")
    M = sp.load_npz(args.projection)
    print(f"  shape: {M.shape}, nnz: {M.nnz}")

    # --- 4. Project Calderon → DOGMA peak space ---
    cal_dogma = project_calderon_to_dogma_space(calderon.X, M)
    print(f"  Calderon projected: {cal_dogma.shape}, nnz: {cal_dogma.nnz}")

    # --- 5. Map Calderon labels → 6 lineages, drop unmappable ---
    cal_labels_25 = calderon.obs[args.label_col].astype(str).values
    cal_lineages = np.array([map_to_lineage(lbl) for lbl in cal_labels_25],
                            dtype=object)
    keep_mask = cal_lineages != None  # noqa: E711
    n_dropped = int((~keep_mask).sum())
    cal_dropped_types = sorted(set(cal_labels_25[~keep_mask]))
    print(f"\nLineage mapping (25 → 6):")
    print(f"  kept: {int(keep_mask.sum())}/{len(cal_labels_25)} samples")
    print(f"  dropped: {n_dropped} samples (types: {cal_dropped_types})")
    cal_dogma_kept = cal_dogma[keep_mask]
    cal_lineages_kept = cal_lineages[keep_mask].astype(str)
    cal_labels_kept = cal_labels_25[keep_mask]
    print(f"  kept lineage distribution: "
          f"{pd.Series(cal_lineages_kept).value_counts().to_dict()}")

    # --- 6. Load encoder ---
    print(f"\nLoading encoder: {args.encoder_ckpt}")
    encoder, ckpt_config = load_atac_encoder_from_ckpt(
        args.encoder_ckpt,
        expected_n_peaks=cal_dogma.shape[1],
        map_location=args.device,
    )
    print(f"  encoder: {type(encoder).__name__}, attn_dim={encoder.attn_dim}")

    # --- 7. Encode both pseudo-bulk DOGMA and projected Calderon ---
    print(f"\nEncoding pseudo-bulk DOGMA (n=8) + Calderon (n={cal_dogma_kept.shape[0]})...")
    z_dogma = encode_samples(centroids_X, encoder, batch_size=8, device=args.device)
    z_calderon = encode_samples(cal_dogma_kept, encoder, batch_size=64,
                                device=args.device)
    print(f"  z_dogma: {z_dogma.shape}, z_calderon: {z_calderon.shape}")

    # --- 8. Cosine similarity (Calderon × 8 DOGMA centroids) ---
    z_dogma_norm = z_dogma / (np.linalg.norm(z_dogma, axis=1, keepdims=True) + 1e-12)
    z_cal_norm = z_calderon / (np.linalg.norm(z_calderon, axis=1, keepdims=True) + 1e-12)
    sim = z_cal_norm @ z_dogma_norm.T  # (n_calderon, 8)
    print(f"\nCosine similarity matrix: {sim.shape}")

    # --- 9. Top-1 NN accuracy ---
    pred_idx = sim.argmax(axis=1)
    pred_lineage = np.array([centroid_labels[i] for i in pred_idx])
    correct = pred_lineage == cal_lineages_kept
    accuracy = float(correct.mean())
    print(f"\n=== Top-1 NN accuracy ===")
    print(f"  overall: {accuracy:.4f} ({correct.sum()}/{len(correct)})")
    print(f"  chance baseline (1/{len(centroid_labels)}): {1.0/len(centroid_labels):.4f}")

    # Per-lineage breakdown
    print(f"\n  Per-lineage accuracy:")
    per_lineage = {}
    for ln in sorted(set(cal_lineages_kept)):
        mask = cal_lineages_kept == ln
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(correct[mask].mean())
        per_lineage[ln] = {"n": n, "accuracy": acc}
        print(f"    {ln:<12s}: {correct[mask].sum():>3d}/{n:<3d} = {acc:.4f}")

    # Confusion matrix (true × pred), 6 × 8
    print(f"\n  Confusion (true_lineage → pred_lineage counts):")
    header_label = "true \\ pred"
    print(f"    {header_label:<12s}", end="")
    for cl in centroid_labels:
        print(f" {cl:>10s}", end="")
    print()
    for ln in sorted(set(cal_lineages_kept)):
        mask = cal_lineages_kept == ln
        print(f"    {ln:<12s}", end="")
        for cl in centroid_labels:
            count = int(((pred_lineage == cl) & mask).sum())
            print(f" {count:>10d}", end="")
        print()

    # --- 10. Decision matrix outcome ---
    print(f"\n=== Decision matrix outcome ===")
    if accuracy >= 0.50:
        verdict = ("ENCODER CAPTURES BIOLOGY AT BULK LEVEL — eval was the problem. "
                   "Action: fix eval (centroid-NN or similarity-based classification), "
                   "no encoder change needed.")
    elif accuracy >= 0.20:
        verdict = ("MIXED — encoder works for some lineages but not others. "
                   "Action: run Test 2 (gene-activity diagnostic) to disambiguate.")
    else:
        verdict = ("ENCODER GENUINELY DOESN'T TRANSFER. "
                   "Action: proceed to Test 2 to check whether peak space itself "
                   "is the bottleneck before committing to Pivot B (PeakVI).")
    print(f"  verdict: {verdict}")

    # --- Write JSON ---
    out_payload = {
        "test": "pseudobulk_compatibility",
        "ckpt_path": str(args.encoder_ckpt),
        "dogma_centroids": {
            "n_centroids": int(centroids_X.shape[0]),
            "labels": centroid_labels,
            "per_class_n_cells": {
                cls: int((labels == cls).sum()) for cls in classes
            },
        },
        "calderon": {
            "n_total": int(len(cal_labels_25)),
            "n_kept": int(keep_mask.sum()),
            "n_dropped": n_dropped,
            "dropped_types": cal_dropped_types,
            "kept_lineage_distribution": {
                k: int(v) for k, v in pd.Series(cal_lineages_kept).value_counts().items()
            },
        },
        "top1_nn_accuracy": accuracy,
        "chance_baseline": 1.0 / len(centroid_labels),
        "per_lineage_accuracy": per_lineage,
        "cosine_similarity_matrix": sim.tolist(),
        "calderon_kept_25class_labels": cal_labels_kept.tolist(),
        "calderon_kept_lineage_labels": cal_lineages_kept.tolist(),
        "predicted_centroid_labels": pred_lineage.tolist(),
        "centroid_label_order": centroid_labels,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
