"""
scripts/exp1_linear_probe_ablation.py — Phase 6 ablation driver.

Runs ``scripts/linear_probe_pretrain.py``'s ``run_condition`` under a
fixed seed on a target dataset, emits summary metrics (R² top-50 DE,
overall Pearson, probe convergence seconds) to both stdout and W&B,
and records the pretrained checkpoint SHA-256 in the W&B run config.

Phase 6.5-infra refactor
------------------------
This script now supports three invocation shapes:

1. ``--ckpt_path <path>`` (default): run BOTH ``pretrained`` and
   ``scratch`` arms against the same checkpoint inside one W&B run.
   Pre-refactor behaviour — preserved for backward compatibility.

2. ``--ckpt_path <path> --condition {pretrained|scratch}``: run ONLY
   that arm. Produces one clean metrics row per W&B run, which is
   what the Phase 6.5 three-run contract (real / mock / random)
   consumes.

3. ``--random_init``: skip checkpoint loading entirely, build a
   fresh ``SimpleRNAEncoder``, and run ONLY the ``scratch`` arm.
   This is the random-init floor for the Phase 6.5 table.
   ``ckpt_sha256`` is logged as ``"n/a"``. Mutually exclusive with
   ``--ckpt_path``.

DOES NOT touch ``train_week3.py``, ``fusion.py``, ``losses.py``,
``loss_registry.py``, ``NeumannPropagation``, or
``PerturbationPredictor``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
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
    # --ckpt_path is no longer required=True: --random_init is the
    # alternative path. Mutual exclusion + "at least one" validation
    # is enforced manually after parse (argparse's
    # mutually_exclusive_group cannot express "exactly one of
    # {flag-without-value, flag-with-value}").
    p.add_argument("--ckpt_path", type=Path, default=None,
                   help="Phase 5 pretrained checkpoint. Required unless "
                        "--random_init is set.")
    p.add_argument("--random_init", action="store_true", default=False,
                   help="Skip checkpoint loading and build a fresh "
                        "SimpleRNAEncoder. Forces --condition=scratch. "
                        "Mutually exclusive with --ckpt_path.")
    # default=None so we can distinguish an explicit user choice from
    # the implicit default when --random_init overrides it.
    p.add_argument("--condition", choices=("pretrained", "scratch", "both"),
                   default=None,
                   help="Which arm to run. Defaults to 'both' (bundled "
                        "pretrained+scratch in one W&B run); pass "
                        "'pretrained' or 'scratch' for single-condition "
                        "runs needed by the Phase 6.5 three-run table.")
    p.add_argument("--n_genes", type=int, default=None,
                   help="Explicit gene count used only when --random_init "
                        "is set and no --dataset_path is provided. When "
                        "--dataset_path resolves to a real .h5ad, the "
                        "dataset's own gene count is used instead.")
    p.add_argument("--dataset_name", type=str, default="norman2019")
    p.add_argument("--dataset_path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--expected_schema_version", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--summary_json", type=Path, default=None,
                   help="Optional path to dump the summary dict as JSON.")
    args = p.parse_args(argv)

    # Mutual exclusion + "at least one" validation.
    if args.random_init and args.ckpt_path is not None:
        p.error("--random_init and --ckpt_path are mutually exclusive.")
    if not args.random_init and args.ckpt_path is None:
        p.error("--ckpt_path is required unless --random_init is set.")

    # --random_init forces --condition=scratch (the pretrained arm is
    # undefined without a checkpoint). Warn only if the user actively
    # requested a different condition — not when the default None
    # carried through.
    if args.random_init:
        if args.condition is not None and args.condition != "scratch":
            warnings.warn(
                f"--random_init forces --condition=scratch; "
                f"ignoring --condition={args.condition}.",
                stacklevel=2,
            )
        args.condition = "scratch"
    else:
        if args.condition is None:
            args.condition = "both"

    return args


def _setup_wandb(enabled: bool, config: dict):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    return wandb.init(project="aivc-linear-probe", config=config, reinit=True)


def _print_pair_summary(pretrained: dict, scratch: dict) -> None:
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


def _print_single_summary(row: dict) -> None:
    headers = ("condition", "r2_top50_de", "r2_overall",
               "pearson_overall", "probe_fit_seconds")
    print("\n=== Linear-probe single-condition summary ===")
    print("{:<12} {:>12} {:>12} {:>18} {:>18}".format(*headers))
    print("{:<12} {:>12.4f} {:>12.4f} {:>18.4f} {:>18.4f}".format(
        row["condition"],
        row["r2_top50_de"],
        row["r2_overall"],
        row["pearson_overall"],
        row["probe_fit_seconds"],
    ))
    print("Single-condition run; no Δ computed. Pair with the matching "
          "sibling run off-script (Phase 6.5 three-row table).\n")


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)

    if args.random_init:
        ckpt_hash = "n/a"
    else:
        ckpt_hash = _hash_file(args.ckpt_path)

    config = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "expected_schema_version": args.expected_schema_version,
        "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
        "ckpt_sha256": ckpt_hash,
        "condition": args.condition,
        "random_init": bool(args.random_init),
        "n_genes_override": args.n_genes,
    }
    run = _setup_wandb(args.wandb, config)

    run_pretrained = args.condition in ("pretrained", "both")
    run_scratch = args.condition in ("scratch", "both")

    pretrained: dict | None = None
    scratch: dict | None = None

    if run_pretrained:
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

    if run_scratch:
        # n_genes discovery for the scratch arm:
        #  - When --ckpt_path is set, run_condition peeks the ckpt
        #    config so pretrained and scratch share a feature space.
        #  - When --random_init is set (no ckpt), run_condition
        #    prefers the dataset's own gene count if --dataset_path
        #    resolves to a real .h5ad; otherwise falls back to
        #    args.n_genes (user-supplied override) and, failing
        #    that, the run_condition default of 512 for the
        #    synthetic dataset.
        scratch = run_condition(
            condition="scratch",
            ckpt_path=args.ckpt_path,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            expected_schema_version=args.expected_schema_version,
            n_genes_fallback=args.n_genes,
        )

    summary: dict = {
        "pretrained": pretrained,
        "scratch": scratch,
        "ckpt_sha256": ckpt_hash,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "condition": args.condition,
        "random_init": bool(args.random_init),
    }

    if pretrained is not None and scratch is not None:
        _print_pair_summary(pretrained, scratch)
    elif pretrained is not None:
        _print_single_summary(pretrained)
    elif scratch is not None:
        _print_single_summary(scratch)

    if run is not None:
        log_dict: dict = {"ckpt_sha256": ckpt_hash}
        if pretrained is not None:
            log_dict.update({
                "pretrained/r2_top50_de": pretrained["r2_top50_de"],
                "pretrained/r2_overall": pretrained["r2_overall"],
                "pretrained/pearson_overall": pretrained["pearson_overall"],
            })
        if scratch is not None:
            log_dict.update({
                "scratch/r2_top50_de": scratch["r2_top50_de"],
                "scratch/r2_overall": scratch["r2_overall"],
                "scratch/pearson_overall": scratch["pearson_overall"],
            })
        if pretrained is not None and scratch is not None:
            log_dict["delta_r2_top50_de"] = (
                pretrained["r2_top50_de"] - scratch["r2_top50_de"]
            )
        run.log(log_dict)
        run.finish()

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    return summary


if __name__ == "__main__":
    main()
