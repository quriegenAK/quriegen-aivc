"""
scripts/linear_probe_pretrain.py — Phase 6 linear-probe ablation.

Measures whether a Phase 5 pretrained ``SimpleRNAEncoder`` produces
representations that predict Norman 2019 / Kang 2018 perturbation
targets *better than a from-scratch encoder of identical capacity*.

Design decisions
----------------
* No causal head is wired in. ``PerturbationPredictor`` operates on
  per-gene scalars via a GAT and has no slot for a cell×gene MLP
  encoder; a structural refactor is deferred to Phase 6.5.
* This script never mutates ``train_week3.py``, ``fusion.py``,
  ``losses.py``, ``loss_registry.py``, ``NeumannPropagation``, or
  ``PerturbationPredictor``. It only loads the pretrained encoder
  (strict, via ``aivc.training.ckpt_loader``) and fits a sklearn
  Ridge probe on its latents.
* W&B project: ``aivc-linear-probe`` — intentionally separate from
  ``aivc-pretrain`` and the main ``aivc`` project.

DOES NOT touch ``train_week3.py`` or any Phase 1–5 training code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.skills.rna_encoder import SimpleRNAEncoder
from aivc.training.ckpt_loader import (
    load_pretrained_simple_rna_encoder,
    peek_pretrain_ckpt_config,
)


LINEAR_PROBE_CKPT_SCHEMA_VERSION = 1


def _scatter_back(Y_masked: np.ndarray, mask: np.ndarray, n_full: int) -> np.ndarray:
    """Place masked-space predictions back into full gene space, NaN elsewhere."""
    out = np.full((Y_masked.shape[0], n_full), np.nan, dtype=np.float32)
    out[:, mask] = Y_masked
    return out


def _arm_label_from_ckpt(condition: str, ckpt_path: "Path | None") -> str:
    """Map (condition, ckpt_path) → {real, mock, random} arm label used by
    the phase-6.5c 9-run matrix and intersection-gate post-processing."""
    if condition == "scratch":
        return "random"
    if condition == "pretrained" and ckpt_path is not None:
        return "mock" if "mock" in Path(ckpt_path).stem.lower() else "real"
    return condition


# --------------------------------------------------------------------- #
# Dataset loading.
# --------------------------------------------------------------------- #
def _load_anndata(path: Path):
    """Load an .h5ad via anndata. Kept thin so tests can monkeypatch."""
    import anndata as ad  # lazy import

    return ad.read_h5ad(path)


def _synth_dataset(
    n_cells: int,
    n_genes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback synthetic dataset used when .h5ad is absent.

    Returns (X, Y, perturbation_labels). Y is a random linear function
    of X so that *both* probes can fit a non-degenerate signal; this is
    a plumbing smoke-test, not a biological result.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    W = rng.normal(scale=0.05, size=(n_genes, n_genes)).astype(np.float32)
    Y = X @ W + 0.1 * rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    pert = rng.integers(0, 4, size=n_cells)
    return X, Y.astype(np.float32), pert


def _load_dataset(
    name: str,
    path: "Path | None",
    n_genes_fallback: int,
    seed: int,
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[List[int]]]":
    """Return (X, Y, pert_labels, provenance_tag, de_indices_or_None).

    ``Y`` is the per-cell perturbation-response target; here we use the
    cell's own expression vector as the target (self-recon probe) when
    a public perturbation target is unavailable, which is valid for
    representation-quality measurement: a better encoder reconstructs
    expression better from its own latents.

    ``de_indices_or_None`` is a list of gene indices (into the aligned
    36,601-dim vocabulary) representing the union of top-50 DE genes
    across all perturbations, as precomputed by build_norman2019_aligned.py.
    When present it is used instead of the variance-based selection in
    ``_fit_probe_and_score``.  ``None`` triggers variance-based fallback.
    """
    if path is not None and path.exists():
        adata = _load_anndata(path)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        # For aligned datasets (e.g. Norman→PBMC10k), ~40% of genes may
        # be structural zeros from non-overlapping vocabularies.  The
        # encoder needs the full n_genes-dim input, but the Ridge probe
        # targets should exclude structural zeros: including them inflates
        # R² (predicting zeros is trivial) and causes numerical instability
        # in Ridge at large cell counts (ill-conditioned Gram matrix).
        nonzero_mask = (X != 0).any(axis=0)
        n_nonzero = int(nonzero_mask.sum())
        n_total = X.shape[1]
        if n_nonzero < n_total:
            _log_msg = (
                f"Filtering Ridge targets to {n_nonzero:,}/{n_total:,} "
                f"genes with nonzero expression ({n_total - n_nonzero:,} "
                f"structural zeros removed). Encoder input unchanged."
            )
            warnings.warn(_log_msg, UserWarning, stacklevel=2)
            Y = X[:, nonzero_mask].copy()
        else:
            Y = X.copy()
        if "perturbation" in adata.obs.columns:
            pert = adata.obs["perturbation"].astype("category").cat.codes.to_numpy()
        elif "condition" in adata.obs.columns:
            pert = adata.obs["condition"].astype("category").cat.codes.to_numpy()
        else:
            pert = np.zeros(X.shape[0], dtype=np.int64)
        # Extract precomputed top-50 DE indices if present in uns.
        # These are indices into the original 36,601-dim aligned vocabulary.
        # If we filtered to nonzero genes, remap them to the filtered space.
        de_indices: Optional[List[int]] = None
        if "top50_de_per_perturbation" in adata.uns:
            de_map: dict = adata.uns["top50_de_per_perturbation"]
            # Build the union of all per-perturbation top-50 indices as a
            # sorted, deduplicated list so the probe evaluates on genes that
            # are DE in at least one perturbation.
            union_idx: set = set()
            for idx_list in de_map.values():
                union_idx.update(idx_list)
            orig_de_indices = sorted(union_idx)
            # Remap to filtered gene space if structural zeros were removed
            if n_nonzero < n_total:
                # Build old→new index map for nonzero genes
                old_to_new = {}
                new_idx = 0
                for old_idx in range(n_total):
                    if nonzero_mask[old_idx]:
                        old_to_new[old_idx] = new_idx
                        new_idx += 1
                de_indices = [old_to_new[i] for i in orig_de_indices
                              if i in old_to_new]
                n_dropped = len(orig_de_indices) - len(de_indices)
                if n_dropped > 0:
                    warnings.warn(
                        f"{n_dropped} DE gene indices fell in structural-zero "
                        f"columns and were dropped from evaluation.",
                        UserWarning, stacklevel=2,
                    )
            else:
                de_indices = orig_de_indices
        else:
            warnings.warn(
                f"uns['top50_de_per_perturbation'] not found in {path}. "
                "Falling back to variance-based DE gene selection. "
                "Run build_norman2019_aligned.py to add precomputed DE indices.",
                UserWarning,
                stacklevel=2,
            )
        return X, Y, pert, f"{name}:{path}", de_indices
    # Synthetic fallback — LOUD warning so silent fallback never hides bad data.
    warnings.warn(
        f"SYNTHETIC FALLBACK: {name!r} data not found at {path}. "
        "Evaluation results are NOT biologically meaningful.",
        UserWarning,
        stacklevel=2,
    )
    X, Y, pert = _synth_dataset(n_cells=512, n_genes=n_genes_fallback, seed=seed)
    return X, Y, pert, f"{name}:synthetic", None


# --------------------------------------------------------------------- #
# Encoder construction.
# --------------------------------------------------------------------- #
def _build_scratch_encoder(
    n_genes: int, hidden_dim: int, latent_dim: int, seed: int
) -> SimpleRNAEncoder:
    g = torch.Generator().manual_seed(seed)
    enc = SimpleRNAEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, latent_dim=latent_dim
    )
    # Re-init parameters deterministically from `seed` so scratch runs
    # are reproducible and independent of global RNG.
    for p in enc.parameters():
        if p.ndim >= 2:
            torch.nn.init.xavier_uniform_(p, generator=g)
        else:
            torch.nn.init.zeros_(p)
    return enc


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- #
# Linear probe.
# --------------------------------------------------------------------- #
def _extract_latents(
    encoder: SimpleRNAEncoder, X: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    encoder = encoder.to(device).eval()
    with torch.no_grad():
        t = torch.from_numpy(X).to(device)
        z, _ = encoder(t)
    return z.cpu().numpy()


def _fit_probe_and_score(
    Z_train: np.ndarray,
    Y_train: np.ndarray,
    Z_test: np.ndarray,
    Y_test: np.ndarray,
    top_k_de: int = 50,
    de_indices: "Optional[List[int]]" = None,
) -> Dict[str, float]:
    """Fit Ridge on latents → expression; report R² on top-K DE genes
    and overall Pearson.

    When ``de_indices`` is provided (precomputed from build_norman2019_aligned.py),
    those gene indices are used directly for the top-50 DE R² metric.
    Otherwise, top-K genes are selected by train-set variance (fallback).
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    t0 = time.time()
    # solver='svd' avoids Cholesky decomposition failures on
    # ill-conditioned Gram matrices (common at >100K cells).
    model = Ridge(alpha=1.0, solver="svd")
    model.fit(Z_train, Y_train)
    fit_seconds = time.time() - t0

    Y_hat = model.predict(Z_test)

    # Select top-K DE genes: use precomputed indices when available,
    # fall back to train-set variance otherwise.
    if de_indices is not None:
        # Clip to valid range in case the aligned vocab differs.
        top_idx = np.array(
            [i for i in de_indices if i < Y_train.shape[1]], dtype=np.int64
        )
        if len(top_idx) == 0:
            warnings.warn(
                "Precomputed de_indices are all out of range for current "
                f"n_genes={Y_train.shape[1]}; falling back to variance selection.",
                UserWarning,
                stacklevel=2,
            )
            de_indices = None  # trigger fallback below
    if de_indices is None:
        # Cheap proxy: top-K by train-set variance.
        var = Y_train.var(axis=0)
        k = min(top_k_de, var.shape[0])
        top_idx = np.argsort(-var)[:k]

    r2_top = float(r2_score(Y_test[:, top_idx], Y_hat[:, top_idx]))
    r2_all = float(r2_score(Y_test, Y_hat))

    y_flat = Y_test.reshape(-1)
    yh_flat = Y_hat.reshape(-1)
    if y_flat.std() == 0 or yh_flat.std() == 0:
        pearson = 0.0
    else:
        pearson = float(pearsonr(y_flat, yh_flat).statistic)

    return {
        "r2_top50_de": r2_top,
        "r2_overall": r2_all,
        "pearson_overall": pearson,
        "probe_fit_seconds": fit_seconds,
    }


# --------------------------------------------------------------------- #
# Condition runner.
# --------------------------------------------------------------------- #
def run_condition(
    condition: str,
    ckpt_path: Path | None,
    dataset_name: str,
    dataset_path: Path | None,
    seed: int,
    hidden_dim: int,
    latent_dim: int,
    expected_schema_version: int = 1,
    n_genes_fallback: int | None = None,
) -> Dict[str, float]:
    """Run one (condition, dataset) cell of the ablation."""
    # For synthetic-fallback runs, align n_genes to the pretrained ckpt
    # when available so pretrained and scratch conditions share a
    # feature space.
    if n_genes_fallback is None:
        if ckpt_path is not None and Path(ckpt_path).exists():
            # Route config discovery through ckpt_loader so the CI
            # check in tests/test_no_bare_torch_load.py stays green.
            _cfg = peek_pretrain_ckpt_config(
                ckpt_path,
                expected_schema_version=expected_schema_version,
            )
            n_genes_fallback = int(_cfg["n_genes"])
        else:
            n_genes_fallback = 512
    X, Y, _pert, provenance, de_indices = _load_dataset(
        name=dataset_name,
        path=dataset_path,
        n_genes_fallback=n_genes_fallback,
        seed=seed,
    )
    n_genes = X.shape[1]

    # CONTRACT: X stays raw (encoder was trained on raw counts).
    # Y target moves to log1p space — variance-stabilizing transform for
    # the Ridge target. Y_hat will be un-standardized back to this space
    # for metric computation. NO train-expressed filter (variance_weighted
    # handles rare-column robustness without a heuristic cutoff).
    Y_log1p = np.log1p(Y).astype(np.float32, copy=False)

    if condition == "pretrained":
        if ckpt_path is None:
            raise ValueError("pretrained condition requires --ckpt_path")
        encoder = load_pretrained_simple_rna_encoder(
            ckpt_path, expected_schema_version=expected_schema_version
        )
        if encoder.n_genes != n_genes:
            raise RuntimeError(
                f"Pretrained encoder n_genes={encoder.n_genes} does not "
                f"match dataset n_genes={n_genes}. Linear probe cannot "
                f"proceed without a compatible feature space."
            )
        effective_hidden = encoder.hidden_dim
        effective_latent = encoder.latent_dim
    elif condition == "scratch":
        encoder = _build_scratch_encoder(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seed=seed,
        )
        effective_hidden = hidden_dim
        effective_latent = latent_dim
    else:
        raise ValueError(f"unknown condition {condition!r}")

    # Train/test split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    split = int(0.8 * X.shape[0])
    tr, te = idx[:split], idx[split:]

    # Encoder forward pass (X raw)
    Z_tr = _extract_latents(copy.deepcopy(encoder), X[tr])
    Z_te = _extract_latents(copy.deepcopy(encoder), X[te])

    # Train-split statistics for Z (BOTH standardizations use train stats)
    z_mu, z_sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8

    # ------- Option B mask: train-split variance filter -------
    # Deterministic floor — replaces the v1 y_sd tripwire. Structural zero
    # filtering is dataset-global (in _load_dataset); the mask below is
    # TRAIN-LOCAL so the Ridge fit and its scoring invariants agree.
    y_sd_tr_raw = Y_log1p[tr].std(0)
    mask_tr = y_sd_tr_raw > 1e-7                     # bool [n_genes]
    n_kept = int(mask_tr.sum())
    n_dropped = int((~mask_tr).sum())
    pct_dropped = n_dropped / len(mask_tr)
    print(f"[mask] n_kept={n_kept} n_dropped={n_dropped} "
          f"pct_dropped={pct_dropped:.4f}")
    assert n_kept > 0, "All columns filtered — dataset is degenerate."

    # Subset Y to kept columns for fit + score (symmetric)
    Y_log1p_k = Y_log1p[:, mask_tr]
    y_mu = Y_log1p_k[tr].mean(0)
    y_sd = Y_log1p_k[tr].std(0) + 1e-8               # safe: all > 1e-7

    # z_sd tripwire stays — real pathology if it trips
    assert float(z_sd.min()) > 1e-6, (
        f"z_sd floor violated: min={float(z_sd.min()):.3e}. "
        f"Encoder latent collapse — distribution mismatch with pretrain."
    )

    # Ridge fit on masked Y
    from sklearn.linear_model import Ridge
    t0 = time.time()
    Z_tr_std = (Z_tr - z_mu) / z_sd
    Z_te_std = (Z_te - z_mu) / z_sd
    Y_tr_std = (Y_log1p_k[tr] - y_mu) / y_sd
    model = Ridge(alpha=1.0, solver='svd').fit(Z_tr_std, Y_tr_std)
    fit_seconds = time.time() - t0
    Y_hat_std_te = model.predict(Z_te_std)
    Y_hat_log1p_te_k = Y_hat_std_te * y_sd + y_mu
    Y_log1p_te_k = Y_log1p_k[te]

    # Map DE indices (original gene space) into masked gene space
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    if de_indices is not None:
        de_idx_orig = np.array(
            [i for i in de_indices if i < Y_log1p.shape[1]], dtype=np.int64
        )
        if de_idx_orig.size < 5:
            raise RuntimeError(
                f"Too few valid DE indices ({de_idx_orig.size}) — dataset misconfigured."
            )
    else:
        # Variance-based fallback — top genes by train variance (pre-mask)
        var_tr = Y_log1p[tr].var(0)
        de_idx_orig = np.argsort(-var_tr)[: min(3217, len(var_tr))].astype(np.int64)

    de_idx_orig = np.asarray(de_idx_orig, dtype=np.int64)
    de_kept = mask_tr[de_idx_orig]
    n_de_total = int(len(de_idx_orig))
    n_de_kept = int(de_kept.sum())
    orig_to_masked = -np.ones(len(mask_tr), dtype=np.int64)
    orig_to_masked[mask_tr] = np.arange(n_kept)
    de_idx_in_k = orig_to_masked[de_idx_orig[de_kept]]
    assert (de_idx_in_k >= 0).all()
    if n_de_kept < 5:
        raise RuntimeError(
            f"Only {n_de_kept}/{n_de_total} DE genes survived mask — "
            f"dataset / filter misconfigured."
        )

    # Per-run metrics on per-run mask
    r2_top50_de_vw = float(r2_score(
        Y_log1p_te_k[:, de_idx_in_k],
        Y_hat_log1p_te_k[:, de_idx_in_k],
        multioutput='variance_weighted',
    ))
    r2_overall_vw = float(r2_score(
        Y_log1p_te_k, Y_hat_log1p_te_k,
        multioutput='variance_weighted',
    ))

    # Final tripwire — |r²| < 2.0
    assert abs(r2_top50_de_vw) < 2.0, (
        f"Tripwire: |r²_top50_de_vw|={abs(r2_top50_de_vw):.3e} >= 2.0. "
        f"Halt per LOCKED contract."
    )

    # Pearson on un-standardized log1p Y (audit; consumed by exp1 driver)
    yf = Y_log1p_te_k.ravel()
    yh = Y_hat_log1p_te_k.ravel()
    pearson_overall = float(pearsonr(yf, yh)[0])

    # Persist per-run artifacts for intersection post-processing
    arm_label = _arm_label_from_ckpt(condition, ckpt_path)
    art_dir = Path(__file__).resolve().parents[1] / "experiments/phase6_5c/artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{arm_label}_seed{seed}"
    np.savez_compressed(
        art_dir / f"run_{run_tag}.npz",
        mask_tr=mask_tr,
        te_idx=np.asarray(te, dtype=np.int64),
        Y_hat_log1p_te_full=_scatter_back(
            Y_hat_log1p_te_k.astype(np.float32), mask_tr, len(mask_tr)
        ),
        de_idx_orig=de_idx_orig,
        seed=np.int32(seed),
        arm=np.array(arm_label),
    )

    metrics = {
        "r2_top50_de_vw": float(r2_top50_de_vw),
        "r2_overall_vw":  float(r2_overall_vw),
        "n_kept":         n_kept,
        "n_dropped":      n_dropped,
        "pct_dropped":    float(pct_dropped),
        "n_de_total":     n_de_total,
        "n_de_kept":      n_de_kept,
        # Aliases for downstream compatibility (exp1_linear_probe_ablation.py)
        "r2_top50_de":    float(r2_top50_de_vw),
        "r2_overall":     float(r2_overall_vw),
        # Backward-compat fields required by exp1_linear_probe_ablation.py
        # print/log plumbing (forbidden to edit). Minimal surface kept stable.
        "pearson_overall": pearson_overall,
        "probe_fit_seconds": fit_seconds,
        "condition":       condition,
        "arm":             arm_label,
        "dataset":         dataset_name,
        "provenance":      provenance,
        "n_cells_train":   int(tr.size),
        "n_cells_test":    int(te.size),
        "n_genes":         int(n_genes),
        "hidden_dim":      int(effective_hidden),
        "latent_dim":      int(effective_latent),
        "seed":            int(seed),
        "z_sd_min":        float(z_sd.min()),
    }
    return metrics


# --------------------------------------------------------------------- #
# W&B plumbing.
# --------------------------------------------------------------------- #
def _setup_wandb(project: str, enabled: bool, config: dict):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    return wandb.init(project=project, config=config, reinit=True)


# --------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", type=Path, default=None,
                   help="Pretrained checkpoint path (required for pretrained condition).")
    p.add_argument("--condition", choices=("pretrained", "scratch"),
                   required=True)
    p.add_argument("--dataset_name", type=str, required=True,
                   help="norman2019 | kang2018 | <custom>")
    p.add_argument("--dataset_path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--wandb", action="store_true",
                   help="Log results to W&B project aivc-linear-probe.")
    p.add_argument("--expected_schema_version", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Dict[str, float]:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_hash = None
    if args.condition == "pretrained" and args.ckpt_path is not None:
        ckpt_hash = _hash_file(args.ckpt_path)

    config = {
        "condition": args.condition,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "expected_schema_version": args.expected_schema_version,
        "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
        "ckpt_sha256": ckpt_hash,
    }
    run = _setup_wandb("aivc-linear-probe", args.wandb, config)

    metrics = run_condition(
        condition=args.condition,
        ckpt_path=args.ckpt_path,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        expected_schema_version=args.expected_schema_version,
    )
    metrics["ckpt_sha256"] = ckpt_hash

    print("[linear-probe] metrics:")
    for k, v in metrics.items():
        print(f"  {k} = {v}")

    if run is not None:
        run.log(metrics)
        run.finish()

    return metrics


if __name__ == "__main__":
    main()
