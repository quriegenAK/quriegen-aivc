"""
scripts/compute_intersection_gate.py — Phase 6.5c LOCKED v2 post-hoc gate.

Runs AFTER the 9-run matrix (3 seeds × 3 arms) has completed. Reads the
per-run artifacts from ``experiments/phase6_5c/artifacts/run_*.npz``,
intersects the per-run train-variance masks across all 9 runs, and
recomputes ``variance_weighted`` R² on the intersection column set — the
primary gate signal for Phase 6.5c (per-run r² is a seed-level
diagnostic; the gate is over the shared column subset).

Outputs:
  - Markdown table on stdout (per-run table + per-arm aggregation).
  - ``.github/phase6_5c_gate.json`` with all numerics (committed).

DOES NOT touch ``train_week3.py``, ``linear_probe_pretrain.py``, or any
Phase 1–6 training code — only reads artifacts and the source
``.h5ad`` through ``_load_dataset``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.linear_probe_pretrain import _load_dataset


ART_DIR = _REPO_ROOT / "experiments/phase6_5c/artifacts"
DEFAULT_DATASET_PATH = _REPO_ROOT / "data/norman2019_aligned.h5ad"
# Single source of truth for the gate JSON: committed under .github/.
DEFAULT_OUT_JSON = _REPO_ROOT / ".github/phase6_5c_gate.json"
ARMS = ("real", "mock", "random")
SEEDS = (3, 17, 42)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts_dir", type=Path, default=ART_DIR)
    p.add_argument("--dataset_name", type=str, default="norman2019")
    p.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    return p.parse_args(argv)


def _load_artifact(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    return {
        "mask_tr": arr["mask_tr"].astype(bool),
        "te_idx": arr["te_idx"].astype(np.int64),
        "Y_hat_log1p_te_full": arr["Y_hat_log1p_te_full"],
        "de_idx_orig": arr["de_idx_orig"].astype(np.int64),
        "seed": int(arr["seed"]),
        "arm": str(arr["arm"]),
        "path": path,
    }


def _recompute_y_log1p(dataset_name: str, dataset_path: Path) -> np.ndarray:
    """Reproduce the exact Y_log1p space that run_condition sees.

    ``_load_dataset`` applies the structural-zero filter identically,
    so Y_log1p from here aligns column-wise with ``mask_tr`` from the
    per-run artifacts (which were computed in the same post-filter space).
    """
    _X, Y, _pert, _prov, _de = _load_dataset(
        name=dataset_name,
        path=dataset_path,
        n_genes_fallback=0,
        seed=0,
    )
    return np.log1p(Y).astype(np.float32, copy=False)


def _r2_vw(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(Y_true, Y_pred, multioutput="variance_weighted"))


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)

    # -------- Load available artifacts (accept N ≥ 6 per 6.5c finalization) --------
    runs: List[dict] = []
    runs_skipped: List[dict] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = args.artifacts_dir / f"run_{arm}_seed{seed}.npz"
            if not path.exists():
                runs_skipped.append({"arm": arm, "seed": int(seed)})
                continue
            runs.append(_load_artifact(path))
    if len(runs) < 6:
        raise RuntimeError(
            f"Too few artifacts ({len(runs)}) — need at least 6 "
            f"(≥2 arms × 3 seeds) to compute a meaningful gate. "
            f"Skipped: {runs_skipped}"
        )
    if runs_skipped:
        print(
            f"[warn] {len(runs_skipped)} run(s) skipped (missing artifact): "
            f"{runs_skipped}. Intersection gate computed across the "
            f"{len(runs)} available runs."
        )

    # -------- Shape + de_idx_orig sanity across runs --------
    n_full = runs[0]["mask_tr"].shape[0]
    de_ref = runs[0]["de_idx_orig"]
    for r in runs:
        assert r["mask_tr"].shape == (n_full,), (
            f"Shape mismatch in run {r['path'].name}: "
            f"{r['mask_tr'].shape} vs ({n_full},)"
        )
        assert np.array_equal(r["de_idx_orig"], de_ref), (
            f"de_idx_orig drift in run {r['path'].name} — "
            f"refuse to aggregate across inconsistent DE sets."
        )

    # -------- Intersection of per-run masks --------
    mask_inter = np.ones(n_full, dtype=bool)
    for r in runs:
        mask_inter &= r["mask_tr"]
    n_kept_inter = int(mask_inter.sum())
    assert n_kept_inter > 0, (
        "Intersection mask has zero columns — no shared support across runs."
    )

    # DE indices inside the intersection
    de_in_inter = de_ref[mask_inter[de_ref]]
    n_de_total = int(de_ref.size)
    n_de_kept_inter = int(de_in_inter.size)
    assert n_de_kept_inter >= 5, (
        f"Only {n_de_kept_inter}/{n_de_total} DE genes survived the "
        f"intersection — gate cannot be evaluated."
    )

    # -------- Recompute Y_log1p once --------
    Y_log1p = _recompute_y_log1p(args.dataset_name, args.dataset_path)
    assert Y_log1p.shape[1] == n_full, (
        f"Y_log1p has {Y_log1p.shape[1]} cols but artifacts expect {n_full}. "
        f"Dataset / structural-zero filter drifted from run-time state."
    )

    # -------- Per-run intersection-space r² --------
    per_run: List[dict] = []
    for r in runs:
        te_idx = r["te_idx"]
        Y_hat_full = r["Y_hat_log1p_te_full"]
        Y_true = Y_log1p[te_idx][:, mask_inter]
        Y_pred = Y_hat_full[:, mask_inter]
        assert not np.isnan(Y_pred).any(), (
            f"NaN in Y_pred over intersection mask for run {r['path'].name} "
            f"— invariant violated."
        )
        r2_overall_is = _r2_vw(Y_true, Y_pred)

        Y_true_de = Y_log1p[te_idx][:, de_in_inter]
        Y_pred_de = Y_hat_full[:, de_in_inter]
        assert not np.isnan(Y_pred_de).any()
        r2_top50_de_is = _r2_vw(Y_true_de, Y_pred_de)

        per_run.append({
            "arm": r["arm"],
            "seed": r["seed"],
            "artifact": r["path"].name,
            "r2_top50_de_is": r2_top50_de_is,
            "r2_overall_is":  r2_overall_is,
        })

    # -------- Aggregate per arm (mean across seeds) --------
    by_arm: Dict[str, List[dict]] = defaultdict(list)
    for row in per_run:
        by_arm[row["arm"]].append(row)

    arm_agg: Dict[str, Dict[str, float]] = {}
    for arm, rows in by_arm.items():
        de_vals = np.array([r["r2_top50_de_is"] for r in rows], dtype=float)
        ov_vals = np.array([r["r2_overall_is"]  for r in rows], dtype=float)
        arm_agg[arm] = {
            "G_top50_de_intersection": float(de_vals.mean()),
            "G_overall_intersection":  float(ov_vals.mean()),
            "n_seeds": int(de_vals.size),
            "seeds": [int(r["seed"]) for r in rows],
        }

    # -------- Print markdown report --------
    print("# Phase 6.5c intersection-gate report\n")
    print(f"- Artifacts dir: `{args.artifacts_dir}`")
    print(f"- Dataset: `{args.dataset_path}`")
    print(f"- Total genes (post structural-zero filter): **{n_full}**")
    print(f"- `n_kept_intersection`: **{n_kept_inter}** "
          f"({n_kept_inter / n_full:.1%})")
    print(f"- `n_de_total`: {n_de_total}")
    print(f"- `n_de_kept_intersection`: **{n_de_kept_inter}** "
          f"({n_de_kept_inter / n_de_total:.1%})\n")

    print("## Per-run r² on intersection mask\n")
    print("| arm    | seed | r²_top50_de_is | r²_overall_is | artifact |")
    print("|--------|------|----------------|---------------|----------|")
    for row in per_run:
        print(f"| {row['arm']:<6} | {row['seed']:>4} "
              f"| {row['r2_top50_de_is']:>+.6f}     "
              f"| {row['r2_overall_is']:>+.6f}    "
              f"| {row['artifact']} |")
    print()

    print("## Per-arm aggregation (mean across 3 seeds)\n")
    print("| arm    | G_top50_de_intersection | G_overall_intersection "
          "| n_kept_intersection | n_de_kept_intersection |")
    print("|--------|-------------------------|------------------------"
          "|---------------------|------------------------|")
    for arm in ARMS:
        if arm not in arm_agg:
            continue
        a = arm_agg[arm]
        print(f"| {arm:<6} | {a['G_top50_de_intersection']:>+.6f}"
              f"               | {a['G_overall_intersection']:>+.6f}"
              f"              | {n_kept_inter:>19} "
              f"| {n_de_kept_inter:>22} |")
    print()

    # -------- Decision (gate on primary arm pair: real vs random) --------
    per_arm_gate: Dict[str, Dict[str, float]] = {}
    for arm, a in arm_agg.items():
        per_arm_gate[arm] = {
            "G_top50_de_intersection": a["G_top50_de_intersection"],
            "G_overall_intersection":  a["G_overall_intersection"],
            "n_seeds":                 a["n_seeds"],
            "seeds":                   a["seeds"],
        }

    decision = "UNEVALUATED"
    delta_de = float("nan")
    delta_overall = float("nan")
    if "real" in arm_agg and "random" in arm_agg:
        g_real_de  = arm_agg["real"]["G_top50_de_intersection"]
        g_rand_de  = arm_agg["random"]["G_top50_de_intersection"]
        g_real_ov  = arm_agg["real"]["G_overall_intersection"]
        g_rand_ov  = arm_agg["random"]["G_overall_intersection"]
        delta_de       = float(g_real_de - g_rand_de)
        delta_overall  = float(g_real_ov - g_rand_ov)

        abs_rand = abs(g_rand_de)
        if abs_rand > 0.01:
            pass_threshold = 0.05 * abs_rand
        else:
            pass_threshold = 0.005
        if delta_de <= 0:
            decision = "FAIL"
        elif delta_de >= pass_threshold:
            decision = "PASS"
        else:
            decision = "SOFT"
        print(
            f"[gate] Delta_real_vs_random_de={delta_de:+.6f} "
            f"(threshold={pass_threshold:+.6f}) → **{decision}**"
        )
    else:
        print(
            "[gate] real and/or random arm missing — decision UNEVALUATED."
        )

    # -------- Write JSON --------
    payload = {
        "artifacts_dir": str(args.artifacts_dir),
        "dataset_path": str(args.dataset_path),
        "n_runs_used": len(runs),
        "runs_skipped": runs_skipped,
        "n_full_genes": n_full,
        "intersection_mask_size": n_kept_inter,
        "n_kept_intersection": n_kept_inter,
        "pct_kept_intersection": n_kept_inter / n_full,
        "n_de_total": n_de_total,
        "n_de_kept_intersection": n_de_kept_inter,
        "per_run": per_run,
        "per_arm": arm_agg,
        "per_arm_gate": per_arm_gate,
        "delta_real_vs_random_de": delta_de,
        "delta_real_vs_random_overall": delta_overall,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[ok] wrote {args.out_json}")

    return payload


if __name__ == "__main__":
    main()
