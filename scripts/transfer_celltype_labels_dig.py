"""Transfer Azimuth cell-type labels from LLL to DIG via protein-space classifier.

Path X follow-up. Published Azimuth labels exist for LLL only
(asap_reproducibility/pbmc_stim_multiome/output/LLL_module_scores.csv).
DIG cells come from the same TotalSeq-A panel, same donors, same time
window — so a classifier trained on LLL protein space with Azimuth
labels as ground truth should transfer well.

Method:
  1. Load dogma_lll_union_labeled.h5ad → protein matrix + obs['cell_type']
  2. Per-marker z-score (StandardScaler fit on LLL only — no leakage)
  3. Stratified 5-fold CV with kNN(k=15) and LogisticRegression(balanced)
  4. Pick classifier with higher macro F1; report per-class P/R/F1
  5. CV gate: macro F1 >= --min_macro_f1 (default 0.85) — else exit non-zero
  6. Train final classifier on full LLL
  7. Apply LLL-fit z-score to DIG protein; predict
  8. Stamp obs['cell_type'], obs['cell_type_confidence'], obs['cell_type_source']
  9. Write dogma_dig_union_labeled.h5ad

Sub-0.85 macro F1 outcome is a hard signal that protein-space transfer is
insufficient for DIG. In that case, fall back to Option A (Azimuth in R) or
Option C (skip DIG, SupCon on LLL only).

Usage:
    python scripts/transfer_celltype_labels_dig.py \\
        --lll_labeled data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad \\
        --dig_in data/phase6_5g_2/dogma_h5ads/dogma_dig_union.h5ad \\
        --dig_out data/phase6_5g_2/dogma_h5ads/dogma_dig_union_labeled.h5ad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CLASSIFIER_CANDIDATES = ("knn", "logreg")


def cv_evaluate(X: np.ndarray, y: np.ndarray, kind: str, n_splits: int = 5,
                random_state: int = 0) -> tuple[float, dict]:
    """Stratified k-fold CV. Returns (macro_f1, per_class_metrics_dict)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_scores = []
    all_y_true, all_y_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf = build_classifier(kind, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        f1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)
        f1_scores.append(f1)
        all_y_true.append(y[test_idx])
        all_y_pred.append(pred)

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    macro_f1 = float(np.mean(f1_scores))
    per_class = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True
    )
    return macro_f1, per_class


def build_classifier(kind: str, random_state: int = 0):
    """Build (StandardScaler + classifier) pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    elif kind == "logreg":
        # NOTE: multi_class kwarg removed in sklearn 1.7; lbfgs auto-uses
        # multinomial when n_classes > 2.
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier {kind!r}")
    return make_pipeline(StandardScaler(), clf)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--lll_labeled", required=True, type=Path,
                   help="dogma_lll_union_labeled.h5ad with obs['cell_type']")
    p.add_argument("--dig_in", required=True, type=Path,
                   help="dogma_dig_union.h5ad (unlabeled)")
    p.add_argument("--dig_out", required=True, type=Path,
                   help="Output dogma_dig_union_labeled.h5ad")
    p.add_argument("--min_macro_f1", type=float, default=0.85,
                   help="CV gate; below this, exit non-zero (default 0.85)")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_classes", default="other,other_T",
                   help="Comma-separated cell_type values to drop from training "
                        "(default: 'other,other_T' — Azimuth-uncertain catch-alls; "
                        "their F1 collapses macro and they're not real lineages)")
    p.add_argument("--low_conf_threshold", type=float, default=0.0,
                   help="If > 0, DIG cells with max-class confidence < threshold "
                        "are stamped with cell_type='Unknown' (default 0.0 = disabled; "
                        "all DIG cells get a 6-lineage prediction).")
    args = p.parse_args()

    # --- Load LLL ---
    print(f"Loading LLL labeled h5ad: {args.lll_labeled}")
    lll = ad.read_h5ad(args.lll_labeled)
    print(f"  shape: {lll.shape}")
    if "protein" not in lll.obsm:
        raise ValueError("obsm['protein'] missing from LLL h5ad")
    if "cell_type" not in lll.obs.columns:
        raise ValueError("obs['cell_type'] missing — run apply_published_celltype_labels.py first")

    X_lll_full = np.asarray(lll.obsm["protein"], dtype=np.float64)
    y_lll_full = np.asarray(lll.obs["cell_type"].astype(str))
    print(f"  protein shape: {X_lll_full.shape}")
    print(f"  full label distribution:")
    for lbl, cnt in pd.Series(y_lll_full).value_counts().items():
        print(f"    {lbl:<12s}: {cnt:>6d}")

    # --- Filter training set ---
    exclude = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    if exclude:
        print(f"\n  Excluding classes from training: {sorted(exclude)}")
        keep_mask = ~pd.Series(y_lll_full).isin(exclude).values
        X_lll = X_lll_full[keep_mask]
        y_lll = y_lll_full[keep_mask]
        print(f"  Training-set cells after exclusion: {len(y_lll)} "
              f"(dropped {(~keep_mask).sum()})")
        print(f"  Filtered label distribution:")
        for lbl, cnt in pd.Series(y_lll).value_counts().items():
            print(f"    {lbl:<12s}: {cnt:>6d}")
    else:
        X_lll, y_lll = X_lll_full, y_lll_full

    # --- CV both classifiers ---
    cv_results = {}
    for kind in CLASSIFIER_CANDIDATES:
        print(f"\n=== CV: {kind} (k={args.cv_folds} stratified folds) ===")
        macro_f1, per_class = cv_evaluate(
            X_lll, y_lll, kind=kind, n_splits=args.cv_folds, random_state=args.seed
        )
        cv_results[kind] = {"macro_f1": macro_f1, "per_class": per_class}
        print(f"  macro F1: {macro_f1:.4f}")
        # Per-class summary
        print(f"  per-class precision/recall/F1 (support):")
        for cls in sorted(set(y_lll)):
            row = per_class.get(cls, {})
            if not row:
                continue
            print(
                f"    {cls:<12s}  P={row.get('precision', 0):.3f}  "
                f"R={row.get('recall', 0):.3f}  F1={row.get('f1-score', 0):.3f}  "
                f"(n={int(row.get('support', 0))})"
            )

    # --- Pick winner ---
    winner = max(cv_results, key=lambda k: cv_results[k]["macro_f1"])
    winner_f1 = cv_results[winner]["macro_f1"]
    print(f"\n=== Winner: {winner} (macro F1 = {winner_f1:.4f}) ===")

    # --- CV gate ---
    if winner_f1 < args.min_macro_f1:
        print(
            f"\n!!! CV GATE FAILED: macro F1 {winner_f1:.4f} < threshold {args.min_macro_f1} !!!\n"
            f"DIG labels NOT written. Consider:\n"
            f"  - Option A: run Azimuth in R on DIG (matches paper exactly)\n"
            f"  - Option C: skip DIG labeling, SupCon on LLL only\n",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"CV gate PASSED ({winner_f1:.4f} >= {args.min_macro_f1})")

    # --- Train final on full LLL ---
    print(f"\nTraining final {winner} on full LLL ({len(y_lll)} cells)...")
    final_clf = build_classifier(winner, random_state=args.seed)
    final_clf.fit(X_lll, y_lll)

    # --- Predict on DIG ---
    print(f"\nLoading DIG h5ad: {args.dig_in}")
    dig = ad.read_h5ad(args.dig_in)
    print(f"  shape: {dig.shape}")
    if "protein" not in dig.obsm:
        raise ValueError("obsm['protein'] missing from DIG h5ad")
    X_dig = np.asarray(dig.obsm["protein"], dtype=np.float64)
    if X_dig.shape[1] != X_lll.shape[1]:
        raise ValueError(
            f"Protein dim mismatch: LLL={X_lll.shape[1]} vs DIG={X_dig.shape[1]}"
        )

    print(f"Predicting on DIG ({X_dig.shape[0]} cells)...")
    dig_pred = final_clf.predict(X_dig)
    dig_proba = final_clf.predict_proba(X_dig)
    classes = final_clf.classes_
    dig_conf = dig_proba.max(axis=1)

    # --- Optional low-confidence routing ---
    n_low = int((dig_conf < args.low_conf_threshold).sum()) if args.low_conf_threshold > 0 else 0
    if args.low_conf_threshold > 0 and n_low > 0:
        print(f"\n  Routing {n_low} cells with conf < {args.low_conf_threshold} → 'Unknown'")
        dig_pred_final = np.where(
            dig_conf < args.low_conf_threshold, "Unknown", dig_pred
        )
    else:
        dig_pred_final = dig_pred

    # --- Stamp obs ---
    # Categories are the trained lineage classes + optional 'Unknown'.
    final_cats = sorted(set(y_lll))
    if args.low_conf_threshold > 0 and n_low > 0:
        final_cats = final_cats + ["Unknown"]
    dig.obs["cell_type"] = pd.Categorical(dig_pred_final, categories=final_cats)
    dig.obs["cell_type_confidence"] = dig_conf.astype(np.float32)
    dig.obs["cell_type_source"] = f"transfer_from_lll_azimuth_{winner}_2026-05-03"

    # --- Distribution + confidence summary ---
    print(f"\n=== DIG label distribution (final) ===")
    n_total = len(dig_pred_final)
    for lbl, cnt in pd.Series(dig_pred_final).value_counts().items():
        pct = 100 * cnt / n_total
        mask = dig_pred_final == lbl
        mean_conf = float(np.mean(dig_conf[mask])) if mask.any() else float("nan")
        print(f"  {lbl:<12s}: {cnt:>6d}  ({pct:>5.1f}%)  mean_conf={mean_conf:.3f}")

    print(f"\n=== Confidence distribution (over all DIG cells) ===")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  P{int(q*100):02d}: {np.quantile(dig_conf, q):.3f}")
    n_lt_06 = int((dig_conf < 0.6).sum())
    print(f"  cells with confidence < 0.6: {n_lt_06} ({100*n_lt_06/n_total:.1f}%)")

    # --- Provenance manifest ---
    manifest = {
        "transfer_date_utc": "2026-05-03",
        "winner_classifier": winner,
        "cv_macro_f1": winner_f1,
        "cv_min_threshold": args.min_macro_f1,
        "cv_folds": args.cv_folds,
        "excluded_classes": sorted(exclude),
        "lll_train_n": int(len(y_lll)),
        "lll_full_n": int(len(y_lll_full)),
        "dig_predict_n": int(X_dig.shape[0]),
        "classes": list(map(str, sorted(set(y_lll)))),
        "low_conf_threshold": args.low_conf_threshold,
        "low_confidence_count_lt_0.6": n_lt_06,
        "all_classifiers_cv_f1": {
            k: v["macro_f1"] for k, v in cv_results.items()
        },
    }
    dig.uns["cell_type_transfer_manifest"] = json.dumps(manifest, indent=2)

    # --- Write ---
    args.dig_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.dig_out}")
    dig.write_h5ad(args.dig_out, compression=None)
    print("Done.")


if __name__ == "__main__":
    main()
