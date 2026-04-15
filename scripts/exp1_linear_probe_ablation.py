"""
scripts/exp1_linear_probe_ablation.py — Phase 6 ablation driver.

Runs ``scripts/linear_probe_pretrain.py`` twice (pretrained vs scratch)
under a fixed seed on a target dataset, emits a summary table
(R² top-50 DE, overall Pearson, probe convergence seconds) to both
stdout and W&B, and records the pretrained checkpoint SHA-256 in the
W&B run config.

DOES NOT touch ``train_week3.py``, ``fusion.py``, ``losses.py``,
``loss_registry.py``, ``NeumannPropagation``, or
``PerturbationPredictor``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.linear_probe_pretrain import run_condition


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", type=Path, required=True,
                   help="Phase 5 pretrained checkpoint.")
    p.add_argument("--dataset_name", type=str, default="norman2019")
    p.add_argument("--dataset_path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--expected_schema_version", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--summary_json", type=Path, default=None,
                   help="Optional path to dump the summary dict as JSON.")
    return p.parse_args(argv)


def _setup_wandb(enabled: bool, config: dict):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    return wandb.init(project="aivc-linear-probe", config=config, reinit=True)


def _print_summary(pretrained: dict, scratch: dict) -> None:
    headers = ("condition", "r2_top50_de", "r2_overall",
               "pearson_overall", "probe_fit_seconds")
    print("\n=== Linear-probe ablation summary ===")
    print("{:<12} {:>12} {:>12} {:>18} {:>18}".format(*headers))
    for row in (pretrained, scratch):
        print("{:<12} {:>12.4f} {:>12.4f} {:>18.4f} {:>18.4f}".format(
            row["condition"],
            row["r2_top50_de"],
            row["r2_overall"],
            row["pearson_overall"],
            row["probe_fit_seconds"],
        ))

    delta_r2 = pretrained["r2_top50_de"] - scratch["r2_top50_de"]
    rel = (delta_r2 / abs(scratch["r2_top50_de"])
           if scratch["r2_top50_de"] != 0 else float("inf"))
    print(f"\nΔR²(top-50 DE) = pretrained − scratch = {delta_r2:+.4f} "
          f"(relative: {rel:+.2%})")
    print("Interpretation gate: proceed to Phase 6.5 only if "
          "relative ΔR² >= +5% on Norman 2019.\n")


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)

    ckpt_hash = _hash_file(args.ckpt_path)
    config = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "expected_schema_version": args.expected_schema_version,
        "ckpt_path": str(args.ckpt_path),
        "ckpt_sha256": ckpt_hash,
    }
    run = _setup_wandb(args.wandb, config)

    pretrained = run_condition(
        condition="pretrained",
        ckpt_path=args.ckpt_path,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        expected_schema_version=args.expected_schema_version,
    )
    # Scratch condition shares the synthetic feature space with
    # pretrained by routing through the same ckpt_path for n_genes
    # discovery (the encoder weights themselves are NOT used).
    scratch = run_condition(
        condition="scratch",
        ckpt_path=args.ckpt_path,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        expected_schema_version=args.expected_schema_version,
    )

    summary = {
        "pretrained": pretrained,
        "scratch": scratch,
        "ckpt_sha256": ckpt_hash,
        "dataset": args.dataset_name,
        "seed": args.seed,
    }

    _print_summary(pretrained, scratch)

    if run is not None:
        run.log({
            "pretrained/r2_top50_de": pretrained["r2_top50_de"],
            "pretrained/r2_overall": pretrained["r2_overall"],
            "pretrained/pearson_overall": pretrained["pearson_overall"],
            "scratch/r2_top50_de": scratch["r2_top50_de"],
            "scratch/r2_overall": scratch["r2_overall"],
            "scratch/pearson_overall": scratch["pearson_overall"],
            "delta_r2_top50_de": pretrained["r2_top50_de"] - scratch["r2_top50_de"],
            "ckpt_sha256": ckpt_hash,
        })
        run.finish()

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    return summary


if __name__ == "__main__":
    main()
