"""
Query MLflow for the best AIVC v1.1 sweep configuration.

Usage:
  python scripts/query_best_config.py
  python scripts/query_best_config.py --metric jakstat/recovery_3x
  python scripts/query_best_config.py --min-r 0.870
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Query MLflow for best AIVC v1.1 sweep configuration."
    )
    parser.add_argument(
        "--metric", default="test/pearson_r",
        help="Metric to optimise (default: test/pearson_r)"
    )
    parser.add_argument(
        "--min-r", type=float, default=0.873,
        help="Minimum Pearson r to consider (default: 0.873)"
    )
    parser.add_argument(
        "--table", action="store_true", default=True,
        help="Print full sweep table"
    )
    args = parser.parse_args()

    from aivc.memory.mlflow_backend import MLflowBackend

    backend = MLflowBackend()
    if not backend._available:
        print(
            "MLflow not available. Install with: pip install mlflow\n"
            "Or run training first: python train_v11.py"
        )
        sys.exit(1)

    print(f"\nQuerying best config by: {args.metric} (min r={args.min_r})\n")

    best = backend.get_best_run(metric=args.metric, min_r=args.min_r)
    if best is None:
        print(f"No runs found with test_r >= {args.min_r}.")
        sys.exit(0)

    p = best.get("params", {})
    m = best.get("metrics", {})

    print("BEST CONFIGURATION:")
    print(f"  lfc_beta:   {p.get('lfc_beta', 'N/A')}")
    print(f"  neumann_K:  {p.get('neumann_K', 'N/A')}")
    print(f"  lambda_l1:  {p.get('lambda_l1', 'N/A')}")
    print(f"  test_r:     {float(m.get('test/pearson_r', 0)):.4f}")
    print(f"  jakstat 3x: {int(float(m.get('jakstat/recovery_3x', 0)))}/15")
    print(f"  IFIT1 FC:   {float(m.get('jakstat/ifit1_pred_fc', 0)):.1f}x "
          f"(actual: {float(m.get('jakstat/ifit1_actual_fc', 107)):.1f}x)")
    print(f"  CD14 r:     {float(m.get('celltype/CD14pos_Monocytes_r', 0)):.4f}")
    print(f"  W density:  {float(m.get('w_matrix/final_density', 1)):.3f}")
    print(f"  Run ID:     {best.get('run_id', 'N/A')}")

    if args.table:
        print()
        backend.print_sweep_table()


if __name__ == "__main__":
    main()
