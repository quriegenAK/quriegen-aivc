"""
aggregate_sweep_results.py — Collate per-config JSON outputs from train_hpc.py
into a single sweep summary + best-model selection.

Runs OUTSIDE the SLURM array (CPU-only, fast). Use after all 36 array tasks
have reported a result_*.json file.

Usage:
    python scripts/aggregate_sweep_results.py \
        --output-dir outputs/v1.1 \
        --checkpoint-dir checkpoints/v1.1 \
        --best-out models/v1.1/model_v11_best.pt \
        --baseline-r 0.863

Selection logic (same as train_v11.py):
    1. Filter out configs whose test_r < baseline_r (no regression allowed)
    2. Among survivors, pick the one with highest jakstat_within_3x
    3. Tiebreak: highest test_r
    4. If no config passes the regression guard, pick highest test_r overall
       and mark the sweep as REGRESSED.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def load_results(output_dir: Path) -> list[dict]:
    files = sorted(output_dir.glob("result_*.json"))
    if not files:
        raise SystemExit(f"No result_*.json files found in {output_dir}")
    results = []
    for f in files:
        try:
            results.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}", file=sys.stderr)
    return results


def select_best(results: list[dict], baseline_r: float) -> tuple[dict, bool]:
    survivors = [r for r in results if r.get("test_r", 0.0) >= baseline_r]
    if survivors:
        best = max(survivors, key=lambda r: (
            r.get("jakstat_within_3x", 0), r.get("test_r", 0.0)))
        return best, False
    best = max(results, key=lambda r: r.get("test_r", 0.0))
    return best, True


def print_table(results: list[dict]) -> None:
    print(f"  {'Beta':<6} {'K':<4} {'L1':<8} {'Test r':<8} {'JS 3x':<6} "
          f"{'JS 10x':<7} {'IFIT1 FC':<10} {'CD14 r':<8}")
    print(f"  {'-' * 62}")
    for r in sorted(
        results,
        key=lambda r: (r.get("lfc_beta", 0), r.get("neumann_K", 0),
                       r.get("lambda_l1", 0)),
    ):
        print(
            f"  {r.get('lfc_beta', 0):<6.2f} "
            f"{r.get('neumann_K', 0):<4} "
            f"{r.get('lambda_l1', 0):<8.4f} "
            f"{r.get('test_r', 0):<8.4f} "
            f"{r.get('jakstat_within_3x', 0):<6} "
            f"{r.get('jakstat_within_10x', 0):<7} "
            f"{r.get('ifit1_pred_fc', 0):<10.1f} "
            f"{r.get('cd14_mono_r', 0):<8.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--best-out", type=Path,
                   default=Path("models/v1.1/model_v11_best.pt"))
    p.add_argument("--summary-out", type=Path,
                   default=None,
                   help="Default: <output-dir>/sweep_summary.json")
    p.add_argument("--baseline-r", type=float, default=0.863)
    args = p.parse_args()

    results = load_results(args.output_dir)
    print(f"\n{'=' * 70}")
    print(f"AIVC v1.1 sweep — aggregated {len(results)} / 36 configs")
    print(f"{'=' * 70}")
    print_table(results)

    best, regressed = select_best(results, args.baseline_r)
    tag = f"beta{best['lfc_beta']}_K{best['neumann_K']}_l1{best['lambda_l1']}"
    print(f"\n  BEST CONFIG: {tag}")
    print(f"    Test r:  {best.get('test_r', 0):.4f}")
    print(f"    JAK-STAT 3x:  {best.get('jakstat_within_3x', 0)}/15")
    print(f"    IFIT1:  {best.get('ifit1_pred_fc', 0):.1f}x "
          f"(actual {best.get('ifit1_actual_fc', 0):.1f}x)")
    if regressed:
        print(f"\n  WARNING: No config beat baseline_r={args.baseline_r}. "
              "Sweep has regressed vs v1.0.")

    # Copy best checkpoint to canonical location
    src_ckpt = args.checkpoint_dir / tag / "best.pt"
    if src_ckpt.exists():
        args.best_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ckpt, args.best_out)
        print(f"\n  Copied best checkpoint: {src_ckpt} -> {args.best_out}")
    else:
        print(f"\n  [WARN] Best checkpoint not found at {src_ckpt}")

    # Write sweep summary
    summary_out = args.summary_out or (args.output_dir / "sweep_summary.json")
    summary_out.write_text(json.dumps({
        "n_configs": len(results),
        "baseline_r": args.baseline_r,
        "regressed_vs_baseline": regressed,
        "best": {
            "lfc_beta": best["lfc_beta"],
            "neumann_K": best["neumann_K"],
            "lambda_l1": best["lambda_l1"],
            "test_r": best.get("test_r"),
            "jakstat_within_3x": best.get("jakstat_within_3x"),
            "jakstat_within_10x": best.get("jakstat_within_10x"),
            "ifit1_pred_fc": best.get("ifit1_pred_fc"),
            "cd14_mono_r": best.get("cd14_mono_r"),
            "best_val_r": best.get("best_val_r"),
            "best_epoch": best.get("best_epoch"),
            "save_path": str(args.best_out),
        },
        "all_results": results,
    }, indent=2, default=float))
    print(f"  Summary: {summary_out}")
    print(f"\n{'=' * 70}")

    sys.exit(2 if regressed else 0)


if __name__ == "__main__":
    main()
