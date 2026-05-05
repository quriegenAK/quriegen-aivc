"""Stage 3 Part 1 Report 2 (stage B) — Mimitou CRISPR perturbation probe.

Tests whether the validated DOGMA SupCon encoder (0.7308 cross-corpus
centroid-NN on Calderon) can ALSO discriminate within-cell-type
PERTURBATION states. This is a strictly harder eval — all cells are
CD4_T, the differences are limited to CRISPR-knockout chromatin/protein
effects post-stim, NOT cell-type identity.

Decision (CEO direction, 2026-05-04):
  ≥ 0.80   → frozen encoder OK; use as-is in Stage 3a state encoder
  0.50-0.80 → encoder marginal; train light adapter (Linear projection
              + LayerNorm) on top of frozen latent before Stage 3a
  < 0.50   → encoder insufficient for perturbation discrimination;
              fine-tune last encoder block on Mimitou before Stage 3a

The reported top-line metric is the 4-class accuracy on the synergy
demo trio + control: {NTC, CD3E, CD4, CD3E_CD4_double}. This is what
the Stage 3a demo actually requires — a clean 4-arm decomposition for
the synergy head. The 5-class score (+ ZAP70) and 6-class score
(+ NFKB2) are reported for completeness but do not gate the verdict.

Three baselines establish the bounds:
  - chance (1 / n_arms)
  - random projection (Linear(n_peaks → latent_dim) with N(0,1) weights,
    L2-normalized) — controls for "any non-trivial projection works"
  - raw ATAC TF-IDF cosine similarity on un-encoded counts — establishes
    the ceiling achievable from the input modality alone

Eval modes:
  - cell_level: encode each test cell individually, classify against
    train centroids. Single-cell-resolution test, most demanding.
  - pseudobulk: encode K mini-pseudo-bulks per arm (binned by random
    cell groups). Closer to canonical Test 1.5 protocol.

Per-arm split: each perturbation's cells split 50/50 into train (centroid
build) and test (classify). If --donor_holdout AND >= 2 donors exist,
overrides random split with leave-one-donor-out for arm-level train/test.

Usage:
  python scripts/probe_mimitou_perturbation_clustering.py \\
      --mimitou_h5ad data/phase6_5g_2/dogma_h5ads/mimitou_crispr_union.h5ad \\
      --encoder_ckpt /gpfs/scratch/.../pretrain_encoders.pt \\
      --out results/stage3_prep/mimitou_perturbation_probe.json \\
      [--donor_holdout]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# Synergy demo trio + control — gates the verdict
SYNERGY_ARMS = ["NTC", "CD3E", "CD4", "CD3E_CD4_double"]
ALL_ARMS = ["NTC", "CD3E", "CD4", "ZAP70", "NFKB2", "CD3E_CD4_double"]


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def split_train_test(
    perturbation: np.ndarray,
    donor_id: np.ndarray,
    arms: list[str],
    donor_holdout: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each arm in `arms`, return boolean masks (train, test).
    Cells outside `arms` are excluded from both.
    """
    rng = np.random.RandomState(seed)
    n = len(perturbation)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    if donor_holdout:
        donors = sorted(set(donor_id))
        if len(donors) < 2:
            print("  donor_holdout requested but <2 donors present; "
                  "falling back to random 50/50 split.")
            donor_holdout = False

    for arm in arms:
        idx = np.where(perturbation == arm)[0]
        n_arm = len(idx)
        if n_arm < 4:
            print(f"  WARN arm {arm!r}: only {n_arm} cells; skipping")
            continue
        if donor_holdout:
            # Hold out the donor with the smallest contribution as test
            d_counts = pd.Series(donor_id[idx]).value_counts()
            test_donor = d_counts.idxmin()
            test_mask[idx[donor_id[idx] == test_donor]] = True
            train_mask[idx[donor_id[idx] != test_donor]] = True
        else:
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            half = n_arm // 2
            train_mask[shuffled[:half]] = True
            test_mask[shuffled[half:]] = True
    return train_mask, test_mask


def build_centroids_in_latent(
    z_train: np.ndarray, train_arms: np.ndarray, arms: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Mean-of-cells centroid per arm in latent space, L2-normalize."""
    rows = []
    kept_arms = []
    for arm in arms:
        mask = train_arms == arm
        if not mask.any():
            continue
        c = z_train[mask].mean(axis=0)
        rows.append(c)
        kept_arms.append(arm)
    centroids = l2_normalize(np.vstack(rows))
    return centroids, kept_arms


def classify_cell_level(
    z_test: np.ndarray, true_arms: np.ndarray,
    centroids: np.ndarray, centroid_arms: list[str],
) -> tuple[float, dict, np.ndarray]:
    """Cell-level cosine NN against arm centroids."""
    z_test_norm = l2_normalize(z_test)
    sim = z_test_norm @ centroids.T  # (n_test, n_arms)
    pred_idx = sim.argmax(axis=1)
    pred_arms = np.array([centroid_arms[i] for i in pred_idx])
    correct = pred_arms == true_arms
    overall = float(correct.mean())
    per_arm = {}
    for arm in centroid_arms:
        mask = true_arms == arm
        if mask.sum() == 0:
            continue
        per_arm[arm] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "n_correct": int(correct[mask].sum()),
        }
    return overall, per_arm, pred_arms


def classify_pseudobulk(
    X_test: sp.spmatrix,
    true_arms: np.ndarray,
    centroids_in_latent: np.ndarray,
    centroid_arms: list[str],
    encoder,
    device: str,
    lysis_idx,
    bins_per_arm: int = 5,
    seed: int = 0,
) -> tuple[float, dict]:
    """Bin test cells into K random groups per arm, encode each group's
    pseudo-bulk (sum of counts), classify each via cosine NN.
    """
    from aivc.eval.calderon_probe import encode_samples
    rng = np.random.RandomState(seed)

    bulk_rows = []
    bulk_arms = []
    for arm in centroid_arms:
        mask = true_arms == arm
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        rng.shuffle(idx)
        groups = np.array_split(idx, min(bins_per_arm, len(idx)))
        for g in groups:
            if len(g) == 0:
                continue
            row = np.asarray(X_test[g].sum(axis=0)).ravel()
            bulk_rows.append(row)
            bulk_arms.append(arm)
    bulk_arms = np.array(bulk_arms)
    bulk_X = sp.csr_matrix(np.vstack(bulk_rows))
    # Encode pseudo-bulks; use lysis_idx=0 for all (LLL)
    z_bulk = encode_samples(
        bulk_X, encoder, batch_size=16, device=device,
        lysis_idx=np.zeros(bulk_X.shape[0], dtype=np.int64) if lysis_idx is not None else None,
    )
    z_bulk_norm = l2_normalize(z_bulk)
    sim = z_bulk_norm @ centroids_in_latent.T
    pred_idx = sim.argmax(axis=1)
    pred_arms = np.array([centroid_arms[i] for i in pred_idx])
    correct = pred_arms == bulk_arms
    overall = float(correct.mean())
    per_arm = {}
    for arm in centroid_arms:
        mask = bulk_arms == arm
        if mask.sum() == 0:
            continue
        per_arm[arm] = {
            "n_pseudobulks": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
        }
    return overall, per_arm


# --- Baselines ---

def random_projection_baseline(
    X_train: sp.spmatrix, X_test: sp.spmatrix,
    train_arms: np.ndarray, test_arms: np.ndarray,
    arms: list[str], latent_dim: int = 128, seed: int = 0,
) -> dict:
    """Random Linear(n_peaks → latent_dim) projection w/ L2 norm, then
    centroid-NN. Controls for 'any non-trivial projection works'.
    """
    rng = np.random.RandomState(seed)
    n_peaks = X_train.shape[1]
    M = rng.randn(n_peaks, latent_dim).astype(np.float32) / np.sqrt(n_peaks)
    # X is csr; X @ dense → np.matrix in some scipy versions. Force ndarray.
    z_train = np.asarray(X_train @ M)
    z_test = np.asarray(X_test @ M)
    centroids, kept = build_centroids_in_latent(z_train, train_arms, arms)
    overall, per_arm, _ = classify_cell_level(z_test, test_arms, centroids, kept)
    return {"overall": overall, "per_arm": per_arm, "kept_arms": kept}


def tfidf_raw_baseline(
    X_train: sp.spmatrix, X_test: sp.spmatrix,
    train_arms: np.ndarray, test_arms: np.ndarray,
    arms: list[str],
) -> dict:
    """Cosine-NN on TF-IDF-weighted raw ATAC counts (no encoder).
    Establishes the input-modality ceiling.
    """
    # Standard ATAC TF-IDF: rows ÷ row_sums, cols ÷ log(n_cells / col_freq)
    Xtrain = X_train.astype(np.float32)
    Xtest = X_test.astype(np.float32)
    n_cells = Xtrain.shape[0]
    col_freq = np.asarray((Xtrain > 0).sum(axis=0)).ravel().astype(np.float32) + 1.0
    idf = np.log(n_cells / col_freq).astype(np.float32)
    row_sums = np.asarray(Xtrain.sum(axis=1)).ravel().astype(np.float32) + 1e-6
    tf_train = sp.diags(1.0 / row_sums) @ Xtrain
    tf_test_rs = np.asarray(Xtest.sum(axis=1)).ravel().astype(np.float32) + 1e-6
    tf_test = sp.diags(1.0 / tf_test_rs) @ Xtest
    z_train = (tf_train @ sp.diags(idf)).toarray()
    z_test = (tf_test @ sp.diags(idf)).toarray()

    centroids, kept = build_centroids_in_latent(z_train, train_arms, arms)
    overall, per_arm, _ = classify_cell_level(z_test, test_arms, centroids, kept)
    return {"overall": overall, "per_arm": per_arm, "kept_arms": kept}


# --- Verdict logic ---

def verdict_from_synergy_acc(synergy_acc: float, n_arms_used: int) -> dict:
    """Apply CEO decision rule on the 4-class (or n_arms_used-class)
    synergy demo accuracy.
    """
    chance = 1.0 / n_arms_used
    if synergy_acc >= 0.80:
        decision = "FROZEN_ENCODER_OK"
        action = (
            "Use frozen encoder as-is for Stage 3a state encoder. "
            "Above the 0.80 bar — proceed without adapter."
        )
    elif synergy_acc >= 0.50:
        decision = "ADAPTER_RECOMMENDED"
        action = (
            "Train a light adapter (Linear(latent_dim → latent_dim) + "
            "LayerNorm + GELU) on top of frozen latent. ~1 day. Adapter "
            "trained on Mimitou perturbation labels, then frozen for Stage 3a."
        )
    else:
        decision = "FINE_TUNE_REQUIRED"
        action = (
            "Encoder insufficient for within-cell-type perturbation "
            "discrimination. Fine-tune last encoder block (or full ATAC "
            "encoder) on Mimitou before Stage 3a. ~2-3 days. Re-validate "
            "on Calderon to confirm cell-type discrimination not lost."
        )
    return {
        "synergy_4class_accuracy": synergy_acc,
        "chance_baseline": chance,
        "decision": decision,
        "action": action,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mimitou_h5ad", required=True, type=Path,
                   help="Output of prepare_mimitou_crispr.py")
    p.add_argument("--encoder_ckpt", required=True, type=Path,
                   help="DOGMA SupCon encoder checkpoint")
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSON")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--donor_holdout", action="store_true",
                   help="Leave-one-donor-out instead of random 50/50 (if >=2 donors)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins_per_arm", type=int, default=5,
                   help="Number of pseudo-bulks per arm in pseudobulk eval")
    args = p.parse_args()

    from aivc.eval.calderon_probe import (
        load_atac_encoder_from_ckpt, encode_samples,
    )

    # --- Load Mimitou h5ad ---
    print(f"Loading Mimitou harmonized h5ad: {args.mimitou_h5ad}")
    adata = ad.read_h5ad(args.mimitou_h5ad)
    if "atac_peaks" not in adata.obsm:
        raise ValueError("obsm['atac_peaks'] missing")
    atac = adata.obsm["atac_peaks"]
    if not sp.issparse(atac):
        atac = sp.csr_matrix(atac)
    perturbation = adata.obs["perturbation"].astype(str).values
    donor_id = adata.obs.get("donor_id", pd.Series(["unknown"] * len(adata))).astype(str).values

    print(f"  shape: {adata.shape}, atac_peaks: {atac.shape}")
    print(f"  perturbation distribution: {pd.Series(perturbation).value_counts().to_dict()}")
    print(f"  donor distribution: {pd.Series(donor_id).value_counts().to_dict()}")

    # --- Sanity: 4-arm completeness ---
    n_per_arm = pd.Series(perturbation).value_counts()
    missing_synergy = [a for a in SYNERGY_ARMS if int(n_per_arm.get(a, 0)) < 30]
    if missing_synergy:
        print(f"\n  WARN: synergy arms with < 30 cells: {missing_synergy}")
        print(f"  Centroid-NN reliability degrades below 30 cells/arm; "
              f"interpret synergy verdict with caution.")

    # --- Determine which arms to evaluate ---
    eval_arms = [a for a in ALL_ARMS if int(n_per_arm.get(a, 0)) >= 4]
    print(f"\n  arms with >=4 cells (eligible for split): {eval_arms}")

    # --- Train/test split ---
    train_mask, test_mask = split_train_test(
        perturbation, donor_id, eval_arms,
        donor_holdout=args.donor_holdout, seed=args.seed,
    )
    print(f"\nSplit: {train_mask.sum()} train + {test_mask.sum()} test cells "
          f"(donor_holdout={args.donor_holdout})")

    X_train = atac[train_mask]
    X_test = atac[test_mask]
    train_arms = perturbation[train_mask]
    test_arms = perturbation[test_mask]

    # --- Load encoder ---
    print(f"\nLoading encoder: {args.encoder_ckpt}")
    encoder, ckpt_config = load_atac_encoder_from_ckpt(
        args.encoder_ckpt,
        expected_n_peaks=atac.shape[1],
        map_location=args.device,
    )
    has_lysis_cov = getattr(encoder, "n_lysis_categories", 0) > 0
    print(f"  encoder: {type(encoder).__name__}, lysis_covariate={has_lysis_cov}")
    lysis_idx = np.zeros(atac.shape[0], dtype=np.int64) if has_lysis_cov else None

    # --- Encode train + test ---
    print(f"\nEncoding train ({X_train.shape[0]}) + test ({X_test.shape[0]}) cells...")
    z_train = encode_samples(
        X_train, encoder, batch_size=64, device=args.device,
        lysis_idx=lysis_idx[train_mask] if lysis_idx is not None else None,
    )
    z_test = encode_samples(
        X_test, encoder, batch_size=64, device=args.device,
        lysis_idx=lysis_idx[test_mask] if lysis_idx is not None else None,
    )
    print(f"  z_train: {z_train.shape}, z_test: {z_test.shape}")

    # --- Build centroids in latent ---
    centroids_lat, centroid_arms = build_centroids_in_latent(z_train, train_arms, eval_arms)
    print(f"  centroids: {centroids_lat.shape} for arms {centroid_arms}")

    # --- Eval Mode A: cell-level centroid-NN (DOGMA encoder) ---
    print(f"\n=== Mode A: cell-level centroid-NN (DOGMA encoder) ===")
    cell_acc_full, per_arm_full, pred_full = classify_cell_level(
        z_test, test_arms, centroids_lat, centroid_arms,
    )
    print(f"  overall ({len(centroid_arms)}-class): {cell_acc_full:.4f}")
    print(f"  chance baseline: {1.0 / len(centroid_arms):.4f}")
    for arm, r in per_arm_full.items():
        print(f"    {arm:<20s}: {r['n_correct']:>4d}/{r['n']:<4d} = {r['accuracy']:.4f}")

    # Restrict to synergy trio + NTC for verdict
    syn_in_eval = [a for a in SYNERGY_ARMS if a in centroid_arms]
    if len(syn_in_eval) < 4:
        print(f"  WARN: only {len(syn_in_eval)} of 4 synergy arms present "
              f"({syn_in_eval}). Verdict will use this reduced set.")
    syn_test_mask = np.isin(test_arms, syn_in_eval)
    z_test_syn = z_test[syn_test_mask]
    arms_test_syn = test_arms[syn_test_mask]
    syn_centroid_idx = [centroid_arms.index(a) for a in syn_in_eval]
    centroids_syn = centroids_lat[syn_centroid_idx]
    cell_acc_syn, per_arm_syn, _ = classify_cell_level(
        z_test_syn, arms_test_syn, centroids_syn, syn_in_eval,
    )
    print(f"\n  synergy trio + control ({len(syn_in_eval)}-class): {cell_acc_syn:.4f}")
    print(f"    chance: {1.0/len(syn_in_eval):.4f}")
    for arm, r in per_arm_syn.items():
        print(f"    {arm:<20s}: {r['n_correct']:>4d}/{r['n']:<4d} = {r['accuracy']:.4f}")

    # --- Eval Mode B: pseudobulk centroid-NN (DOGMA encoder) ---
    print(f"\n=== Mode B: pseudobulk centroid-NN (DOGMA encoder) ===")
    pb_acc_full, pb_per_arm_full = classify_pseudobulk(
        X_test, test_arms, centroids_lat, centroid_arms,
        encoder, device=args.device, lysis_idx=lysis_idx,
        bins_per_arm=args.bins_per_arm, seed=args.seed,
    )
    print(f"  overall ({len(centroid_arms)}-class pseudobulk): {pb_acc_full:.4f}")
    for arm, r in pb_per_arm_full.items():
        print(f"    {arm:<20s}: pb_n={r['n_pseudobulks']:<3d}  acc={r['accuracy']:.4f}")

    # --- Baseline 1: Random projection ---
    print(f"\n=== Baseline 1: Random projection ===")
    rp = random_projection_baseline(
        X_train, X_test, train_arms, test_arms, eval_arms, seed=args.seed,
    )
    print(f"  overall: {rp['overall']:.4f}")

    # --- Baseline 2: Raw TF-IDF cosine ---
    print(f"\n=== Baseline 2: Raw ATAC TF-IDF cosine ===")
    tfidf = tfidf_raw_baseline(
        X_train, X_test, train_arms, test_arms, eval_arms,
    )
    print(f"  overall: {tfidf['overall']:.4f}")

    # --- Verdict ---
    print(f"\n=== Verdict ===")
    verdict = verdict_from_synergy_acc(cell_acc_syn, n_arms_used=len(syn_in_eval))
    print(f"  Synergy {len(syn_in_eval)}-class accuracy: {cell_acc_syn:.4f}")
    print(f"  Chance baseline: {verdict['chance_baseline']:.4f}")
    print(f"  Random projection (full {len(eval_arms)}-class): {rp['overall']:.4f}")
    print(f"  Raw TF-IDF (full {len(eval_arms)}-class): {tfidf['overall']:.4f}")
    print(f"  Decision: {verdict['decision']}")
    print(f"  Action: {verdict['action']}")

    # Sanity check: encoder should beat random projection
    if cell_acc_syn <= rp["overall"]:
        warning = (
            "ENCODER UNDERPERFORMS RANDOM PROJECTION on synergy arms — "
            "this is a strong signal that the encoder has not learned "
            "perturbation-discriminative features. Recommend full "
            "fine-tune or alternate encoder approach."
        )
        verdict["sanity_warning"] = warning
        print(f"\n  ⚠ {warning}")

    # --- Write JSON ---
    out_payload = {
        "test": "mimitou_perturbation_centroid_nn",
        "mimitou_h5ad": str(args.mimitou_h5ad),
        "encoder_ckpt": str(args.encoder_ckpt),
        "donor_holdout": args.donor_holdout,
        "n_train_cells": int(train_mask.sum()),
        "n_test_cells": int(test_mask.sum()),
        "eval_arms": eval_arms,
        "synergy_arms_present": syn_in_eval,
        "mode_a_cell_level": {
            "full_arm_set": {
                "n_classes": len(centroid_arms),
                "overall_accuracy": cell_acc_full,
                "per_arm": per_arm_full,
            },
            "synergy_arm_set": {
                "n_classes": len(syn_in_eval),
                "overall_accuracy": cell_acc_syn,
                "per_arm": per_arm_syn,
            },
        },
        "mode_b_pseudobulk": {
            "n_classes": len(centroid_arms),
            "bins_per_arm": args.bins_per_arm,
            "overall_accuracy": pb_acc_full,
            "per_arm": pb_per_arm_full,
        },
        "baselines": {
            "chance_synergy": 1.0 / len(syn_in_eval),
            "random_projection_full": rp,
            "raw_tfidf_full": tfidf,
        },
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
