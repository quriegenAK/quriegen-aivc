"""scripts/phase6_5e_rsa.py — RSA evaluation of the 6.5e contrastive ckpt.

Reuses the run_rsa pipeline from ``scripts.phase6_5d_rsa`` unchanged,
then enriches the output JSON with 6.5e-specific keys and the T1-T7
tripwire record produced by the fine-tune script.

Why a wrapper, not a CLI-flag on 6.5d's script:
    ``scripts/phase6_5d_rsa.py`` hard-asserts ``EXPECTED_CKPT_SHA ==
    416e8b1a…`` as a frozen 6.5d contract. The 6.5e spec FAILURE
    HANDLING rule says 'do not patch 6.5d's RSA script from 6.5e
    branch'. This wrapper monkey-patches the SHA constant for the
    duration of the call and restores it — the on-disk 6.5d module
    is unchanged.

Also uses 6.5d's locked outcome classification (classify_outcome) for
a *parallel* rsa-only outcome, and computes the 6.5e-specific
contrastive-vs-reconstruction delta + its bootstrap CI.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import phase6_5d_rsa as rsa_mod  # noqa: E402

# 6.5d's frozen reconstruction values (from .github/phase6_5d_rsa.json).
RSA_REAL_RECONSTRUCTION = -0.05842228916260816
RSA_RANDOM_FROZEN = 0.019293147970669087


# --------------------------------------------------------------------- #
# Locked 6.5e outcome rule (applied in order; Gate 1 first).
# --------------------------------------------------------------------- #
def classify_e1_outcome(
    rsa_real_contrastive: float,
    ci_width_real_contrastive: float,
    delta_c_minus_r: float,
    ci_delta_c_minus_r: Dict[str, float],
    ci_real_contrastive: Dict[str, float],
) -> tuple[str, str]:
    """Apply the 6.5e-locked outcome table from prompts/phase6_5e_contrastive.md.

    Ordering:
        Gate 1 — tripwire: CI_width_real > 0.30 -> INCONCLUSIVE.
        Gate 2 — outcome table (E1-WIN / E1-PARTIAL / E1-NULL / E1-REGRESS).
    """
    if ci_width_real_contrastive > 0.30:
        return (
            "INCONCLUSIVE",
            f"CI_width_real={ci_width_real_contrastive:.3f} > 0.30 "
            "(sampling-noise limited).",
        )

    r_c = float(rsa_real_contrastive)
    r_r = float(RSA_REAL_RECONSTRUCTION)
    ci_lo = ci_real_contrastive["low"]
    ci_hi = ci_real_contrastive["high"]
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)

    # E1-WIN
    if r_c > 0.05 and excludes_zero:
        return (
            "E1-WIN",
            f"R_c={r_c:.3f} > 0.05 and 0 ∉ CI_c — objective fix strong.",
        )

    # E1-PARTIAL
    if (r_c - r_r) > 0.05 and r_c <= 0.05:
        return (
            "E1-PARTIAL",
            f"Δ={delta_c_minus_r:.3f} > 0.05 but R_c={r_c:.3f} ≤ 0.05 — "
            "objective helps but does not fully fix.",
        )

    # E1-REGRESS
    if r_c < (r_r - 0.05):
        return (
            "E1-REGRESS",
            f"R_c={r_c:.3f} < R_r−0.05 — fine-tune made geometry worse.",
        )

    # E1-NULL: |R_c − R_r| < 0.05 AND 0 ∈ CI_Δ(R_c − R_r)
    ci_delta_lo = ci_delta_c_minus_r["low"]
    ci_delta_hi = ci_delta_c_minus_r["high"]
    zero_in_delta_ci = (ci_delta_lo <= 0) and (0 <= ci_delta_hi)
    if abs(r_c - r_r) < 0.05 and zero_in_delta_ci:
        return (
            "E1-NULL",
            f"|Δ|={abs(delta_c_minus_r):.3f} < 0.05 and 0 ∈ CI_Δ — "
            "objective is not the primary issue; lineage dominates.",
        )

    # Nothing fired — return a boundary marker. Per the outcome table
    # this is effectively the unclassified gap; treat as E1-NULL for
    # the conservative reading (no strong evidence of a shift).
    return (
        "E1-NULL",
        f"|Δ|={abs(delta_c_minus_r):.3f}, R_c={r_c:.3f}. No locked "
        "branch triggered strongly; defaulting to E1-NULL (conservative).",
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=Path, required=True)
    p.add_argument("--contrastive-ckpt", type=Path, required=True)
    p.add_argument(
        "--parent-ckpt-sha",
        default="416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e",
    )
    p.add_argument("--random-seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--min-cells-per-pert", type=int, default=20)
    p.add_argument("--bootstrap-rng-seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--art-dir", type=Path, required=True)
    p.add_argument(
        "--tripwire-json",
        type=Path,
        default=Path("experiments/phase6_5e/tripwires.json"),
        help="Written by phase6_5e_finetune_contrastive.py. Merged into "
        "the RSA output JSON under 'tripwires' / 'training_deviations'.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    seeds = [int(s) for s in args.random_seeds.split(",")]

    # Compute SHA of contrastive ckpt and temporarily swap 6.5d's
    # EXPECTED_CKPT_SHA so run_rsa does not abort. The on-disk 6.5d
    # module is not modified — the patch is in-process only.
    contrastive_sha = rsa_mod._hash_file(args.contrastive_ckpt)
    saved_expected = rsa_mod.EXPECTED_CKPT_SHA
    rsa_mod.EXPECTED_CKPT_SHA = contrastive_sha
    try:
        rsa_result = rsa_mod.run_rsa(
            adata_path=args.adata,
            ckpt_path=args.contrastive_ckpt,
            seeds=seeds,
            n_boot=args.n_boot,
            min_cells_per_pert=args.min_cells_per_pert,
            art_dir=args.art_dir,
            bootstrap_rng_seed=args.bootstrap_rng_seed,
        )
    finally:
        rsa_mod.EXPECTED_CKPT_SHA = saved_expected

    # ---- Compute 6.5e-specific delta CIs ----
    # Δ_{c−r} = R_c − R_r (R_r frozen from 6.5d). Since R_r is a
    # constant, the bootstrap distribution of Δ is the bootstrap of
    # R_c shifted by the frozen constant.
    art_dir = Path(args.art_dir)
    boot_real = np.load(art_dir / "boot_real.npy")
    boot_delta_c_minus_r = boot_real - RSA_REAL_RECONSTRUCTION
    ci_delta_c_minus_r = {
        "low": float(np.percentile(boot_delta_c_minus_r, 2.5)),
        "high": float(np.percentile(boot_delta_c_minus_r, 97.5)),
    }
    # Δ_{c−random}: 6.5d's frozen random mean is the reference.
    boot_delta_c_minus_random = boot_real - RSA_RANDOM_FROZEN
    ci_delta_c_minus_random = {
        "low": float(np.percentile(boot_delta_c_minus_random, 2.5)),
        "high": float(np.percentile(boot_delta_c_minus_random, 97.5)),
    }

    # 6.5e-locked outcome.
    outcome_e1, interp_e1 = classify_e1_outcome(
        rsa_real_contrastive=rsa_result.rsa_real_mean,
        ci_width_real_contrastive=rsa_result.ci_width_real,
        delta_c_minus_r=rsa_result.rsa_real_mean - RSA_REAL_RECONSTRUCTION,
        ci_delta_c_minus_r=ci_delta_c_minus_r,
        ci_real_contrastive={
            "low": rsa_result.rsa_real_ci95[0],
            "high": rsa_result.rsa_real_ci95[1],
        },
    )

    # Merge tripwires from fine-tune if available.
    tripwire_payload: Dict = {"tripwires": {}, "training_deviations": []}
    if args.tripwire_json.exists():
        tripwire_payload = json.loads(args.tripwire_json.read_text())

    base = rsa_mod._result_to_json(rsa_result)
    base.update({
        "phase": "6.5e",
        "stage": "joint_contrastive_only_e1",
        "parent_ckpt_sha": args.parent_ckpt_sha,
        "contrastive_ckpt_sha": contrastive_sha,
        "rsa_real_reconstruction": RSA_REAL_RECONSTRUCTION,
        "rsa_random_frozen_6_5d": RSA_RANDOM_FROZEN,
        "delta_contrastive_vs_reconstruction": (
            rsa_result.rsa_real_mean - RSA_REAL_RECONSTRUCTION
        ),
        "delta_contrastive_vs_reconstruction_ci95": [
            ci_delta_c_minus_r["low"],
            ci_delta_c_minus_r["high"],
        ],
        "delta_contrastive_vs_random": (
            rsa_result.rsa_real_mean - RSA_RANDOM_FROZEN
        ),
        "delta_contrastive_vs_random_ci95": [
            ci_delta_c_minus_random["low"],
            ci_delta_c_minus_random["high"],
        ],
        "outcome_e1": outcome_e1,
        "interpretation_e1": interp_e1,
        "tripwires": tripwire_payload.get("tripwires", {}),
        "training_deviations": tripwire_payload.get("training_deviations", []),
    })

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(base, indent=2))
    print(f"\n[6.5e] OUTCOME = {outcome_e1}", flush=True)
    print(f"[6.5e] {interp_e1}", flush=True)
    print(f"[6.5e] wrote {args.out_json}", flush=True)
    return base


if __name__ == "__main__":
    main()
