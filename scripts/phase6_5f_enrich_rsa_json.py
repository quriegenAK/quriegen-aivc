"""scripts/phase6_5f_enrich_rsa_json.py — RSA evaluation wrapper for
the Phase 6.5f frozen-projection ckpt.

Reuses ``scripts.phase6_5d_rsa.run_rsa`` unchanged (monkey-patches
``EXPECTED_CKPT_SHA`` in-process only; the on-disk 6.5d module is not
modified — same pattern as ``scripts/phase6_5e_rsa.py``).

Enriches the 6.5d output JSON with 6.5f-specific fields:

  - rsa_real_frozen_proj (+ CI, copy of rsa_real_mean)
  - rsa_real_contrastive_only (frozen -0.0621 from 6.5e)
  - rsa_real_reconstruction_dominant (frozen -0.0584 from 6.5d)
  - rsa_random_mean (frozen +0.0193 from 6.5d)
  - delta_frozen_proj_vs_contrastive_only (primary gate, with pair-bootstrap CI)
  - delta_frozen_proj_vs_reconstruction_dominant (with CI)
  - delta_frozen_proj_vs_random (with CI)
  - projections_frozen: true
  - loss_weights (= 6.5e's mask)
  - parent_ckpt_sha, phase6_5e_baseline_ckpt_sha, frozen_proj_ckpt_sha
  - parent_stage, child_stage
  - tripwires block T1-T8 (merged from experiments/phase6_5f/tripwires.json)
  - outcome_f (F-WIN / F-PARTIAL / F-NULL / F-REGRESS / INCONCLUSIVE)
  - interpretation_f (one-sentence)

Outcome classification is the locked Gate 1 (CI_width > 0.30 →
INCONCLUSIVE) followed by Gate 2 (the 6.5f outcome table; compared
against 6.5e baseline R_c_e1 = -0.0621).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import phase6_5d_rsa as rsa_mod  # noqa: E402


# Frozen baselines (pre-registered, do NOT recompute in 6.5f).
RSA_REAL_CONTRASTIVE_ONLY_E1 = -0.062079189678334507  # 6.5e mean
RSA_REAL_CONTRASTIVE_ONLY_E1_CI95 = (-0.07385323118867206, -0.05031984043606941)
RSA_REAL_RECONSTRUCTION_DOMINANT = -0.05842228916260816  # 6.5d mean
RSA_RANDOM_FROZEN = 0.019293147970669087  # 6.5d random mean
PARENT_CKPT_SHA = (
    "416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e"
)
PHASE6_5E_BASELINE_SHA = (
    "6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee"
)


def classify_f_outcome(
    rsa_real_frozen_proj: float,
    ci_width_real: float,
    ci_real: Dict[str, float],
    delta_b_minus_c: float,
    ci_delta_b_minus_c: Dict[str, float],
) -> tuple[str, str]:
    """Apply the 6.5f locked outcome table (Gate 1 then Gate 2)."""
    # Gate 1 — tripwire: CI_width > 0.30 → INCONCLUSIVE.
    if ci_width_real > 0.30:
        return (
            "INCONCLUSIVE",
            f"CI_width_real={ci_width_real:.3f} > 0.30 "
            "(sampling-noise limited).",
        )

    r_b = float(rsa_real_frozen_proj)
    r_c = float(RSA_REAL_CONTRASTIVE_ONLY_E1)
    ci_lo = ci_real["low"]
    ci_hi = ci_real["high"]
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)

    # F-WIN: R_b > +0.05 AND 0 ∉ CI_b.
    if r_b > 0.05 and excludes_zero:
        return (
            "F-WIN",
            f"R_b={r_b:.3f} > 0.05 and 0 ∉ CI_b — projection absorption "
            "was load-bearing; freezing projections lets the encoder "
            "shift into the perturbation-relevant geometry.",
        )

    # F-PARTIAL: Δ_{b-c_e1} > +0.05 AND R_b ≤ +0.05.
    if delta_b_minus_c > 0.05 and r_b <= 0.05:
        return (
            "F-PARTIAL",
            f"Δ_{{b-c_e1}}={delta_b_minus_c:.3f} > 0.05 but R_b={r_b:.3f} "
            "≤ 0.05 — freezing helps but doesn't fully recover; both "
            "projection-absorption and lineage contribute.",
        )

    # F-REGRESS: R_b < R_c_e1 − 0.05.
    if r_b < (r_c - 0.05):
        return (
            "F-REGRESS",
            f"R_b={r_b:.3f} < R_c_e1−0.05={r_c - 0.05:.3f} — freezing "
            "projections actively harmed alignment; contradicts the "
            "projection-absorption hypothesis.",
        )

    # F-NULL: |Δ_{b-c_e1}| < 0.05 AND 0 ∈ CI_{b-c_e1}.
    ci_delta_lo = ci_delta_b_minus_c["low"]
    ci_delta_hi = ci_delta_b_minus_c["high"]
    zero_in_delta_ci = (ci_delta_lo <= 0) and (0 <= ci_delta_hi)
    if abs(delta_b_minus_c) < 0.05 and zero_in_delta_ci:
        return (
            "F-NULL",
            f"|Δ_{{b-c_e1}}|={abs(delta_b_minus_c):.3f} < 0.05 and "
            "0 ∈ CI_Δ — projection absorption was NOT the cause; "
            "encoder direction is stable under contrastive pressure "
            "on PBMC regardless of gradient routing; lineage dominates.",
        )

    # Boundary gap — default to F-NULL (conservative), same policy as
    # 6.5e's classifier.
    return (
        "F-NULL",
        f"|Δ_{{b-c_e1}}|={abs(delta_b_minus_c):.3f}, R_b={r_b:.3f}. "
        "No locked branch triggered strongly; defaulting to F-NULL "
        "(conservative).",
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=Path, required=True)
    p.add_argument("--frozen-proj-ckpt", type=Path, required=True)
    p.add_argument("--random-seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--min-cells-per-pert", type=int, default=20)
    p.add_argument("--bootstrap-rng-seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--art-dir", type=Path, required=True)
    p.add_argument(
        "--tripwire-json",
        type=Path,
        default=Path("experiments/phase6_5f/tripwires.json"),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    seeds = [int(s) for s in args.random_seeds.split(",")]

    frozen_sha = rsa_mod._hash_file(args.frozen_proj_ckpt)

    # Guardrails against accidentally evaluating the wrong ckpt.
    if frozen_sha == PARENT_CKPT_SHA:
        raise RuntimeError(
            f"[6.5f:enrich] frozen-proj ckpt SHA equals parent "
            f"{PARENT_CKPT_SHA}. STOP."
        )
    if frozen_sha == PHASE6_5E_BASELINE_SHA:
        raise RuntimeError(
            f"[6.5f:enrich] frozen-proj ckpt SHA equals 6.5e baseline "
            f"{PHASE6_5E_BASELINE_SHA}. STOP."
        )

    saved_expected = rsa_mod.EXPECTED_CKPT_SHA
    rsa_mod.EXPECTED_CKPT_SHA = frozen_sha
    try:
        rsa_result = rsa_mod.run_rsa(
            adata_path=args.adata,
            ckpt_path=args.frozen_proj_ckpt,
            seeds=seeds,
            n_boot=args.n_boot,
            min_cells_per_pert=args.min_cells_per_pert,
            art_dir=args.art_dir,
            bootstrap_rng_seed=args.bootstrap_rng_seed,
        )
    finally:
        rsa_mod.EXPECTED_CKPT_SHA = saved_expected

    # ---- Pair-bootstrap deltas (R_c_e1, R_r, R_random are frozen constants;
    # bootstrap distribution of Δ is bootstrap of R_b shifted by the constant).
    art_dir = Path(args.art_dir)
    boot_real = np.load(art_dir / "boot_real.npy")

    boot_delta_b_minus_c = boot_real - RSA_REAL_CONTRASTIVE_ONLY_E1
    ci_delta_b_minus_c = {
        "low": float(np.percentile(boot_delta_b_minus_c, 2.5)),
        "high": float(np.percentile(boot_delta_b_minus_c, 97.5)),
    }
    boot_delta_b_minus_r = boot_real - RSA_REAL_RECONSTRUCTION_DOMINANT
    ci_delta_b_minus_r = {
        "low": float(np.percentile(boot_delta_b_minus_r, 2.5)),
        "high": float(np.percentile(boot_delta_b_minus_r, 97.5)),
    }
    boot_delta_b_minus_random = boot_real - RSA_RANDOM_FROZEN
    ci_delta_b_minus_random = {
        "low": float(np.percentile(boot_delta_b_minus_random, 2.5)),
        "high": float(np.percentile(boot_delta_b_minus_random, 97.5)),
    }

    delta_b_minus_c = rsa_result.rsa_real_mean - RSA_REAL_CONTRASTIVE_ONLY_E1
    delta_b_minus_r = rsa_result.rsa_real_mean - RSA_REAL_RECONSTRUCTION_DOMINANT
    delta_b_minus_random = rsa_result.rsa_real_mean - RSA_RANDOM_FROZEN

    outcome_f, interp_f = classify_f_outcome(
        rsa_real_frozen_proj=rsa_result.rsa_real_mean,
        ci_width_real=rsa_result.ci_width_real,
        ci_real={
            "low": rsa_result.rsa_real_ci95[0],
            "high": rsa_result.rsa_real_ci95[1],
        },
        delta_b_minus_c=delta_b_minus_c,
        ci_delta_b_minus_c=ci_delta_b_minus_c,
    )

    # Merge tripwires from the fine-tune script.
    tripwire_payload: Dict = {"tripwires": {}, "training_deviations": []}
    if args.tripwire_json.exists():
        tripwire_payload = json.loads(args.tripwire_json.read_text())

    base = rsa_mod._result_to_json(rsa_result)
    base.update({
        "phase": "6.5f",
        "parent_stage": "joint",
        "child_stage": "joint_contrastive_only_e1",
        "stage": "joint_contrastive_only_e1",
        "projections_frozen": True,
        "loss_weights": {
            "masked_rna_recon": 0.0,
            "masked_atac_recon": 0.0,
            "cross_modal_infonce": 1.0,
            "peak_to_gene_aux": 0.0,
        },
        "parent_ckpt_sha": PARENT_CKPT_SHA,
        "phase6_5e_baseline_ckpt_sha": PHASE6_5E_BASELINE_SHA,
        "frozen_proj_ckpt_sha": frozen_sha,

        "rsa_real_frozen_proj": rsa_result.rsa_real_mean,
        "rsa_real_frozen_proj_ci95": list(rsa_result.rsa_real_ci95),
        "rsa_real_contrastive_only": RSA_REAL_CONTRASTIVE_ONLY_E1,
        "rsa_real_contrastive_only_ci95": list(RSA_REAL_CONTRASTIVE_ONLY_E1_CI95),
        "rsa_real_reconstruction_dominant": RSA_REAL_RECONSTRUCTION_DOMINANT,
        "rsa_random_mean": RSA_RANDOM_FROZEN,

        "delta_frozen_proj_vs_contrastive_only": delta_b_minus_c,
        "delta_frozen_proj_vs_contrastive_only_ci95": [
            ci_delta_b_minus_c["low"],
            ci_delta_b_minus_c["high"],
        ],
        "delta_frozen_proj_vs_reconstruction_dominant": delta_b_minus_r,
        "delta_frozen_proj_vs_reconstruction_dominant_ci95": [
            ci_delta_b_minus_r["low"],
            ci_delta_b_minus_r["high"],
        ],
        "delta_frozen_proj_vs_random": delta_b_minus_random,
        "delta_frozen_proj_vs_random_ci95": [
            ci_delta_b_minus_random["low"],
            ci_delta_b_minus_random["high"],
        ],

        "outcome_f": outcome_f,
        "interpretation_f": interp_f,
        "tripwires": tripwire_payload.get("tripwires", {}),
        "training_deviations": tripwire_payload.get("training_deviations", []),
    })

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(base, indent=2))
    print(f"\n[6.5f] OUTCOME = {outcome_f}", flush=True)
    print(f"[6.5f] {interp_f}", flush=True)
    print(f"[6.5f] wrote {args.out_json}", flush=True)
    return base


if __name__ == "__main__":
    main()
