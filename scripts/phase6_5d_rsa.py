"""scripts/phase6_5d_rsa.py — Phase 6.5d RSA objective-mismatch test.

Pre-registered representation similarity analysis (RSA) between the
pretrained ``SimpleRNAEncoder``'s latent space and Norman 2019
perturbation response space, with a 5-init random baseline and
1000-resample pair-bootstrap 95% CIs.

NO TRAINING. NO PROBE. Pure geometric analysis. Outcome is classified
by the locked interpretation rule in ``scripts/lib/rsa.classify_outcome``.

Contract: see ``prompts/phase6_5d_rsa.md`` (LOCKED).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.skills.rna_encoder import SimpleRNAEncoder  # noqa: E402
from aivc.training.ckpt_loader import (  # noqa: E402
    load_pretrained_simple_rna_encoder,
)
from scripts.lib.rsa import (  # noqa: E402
    classify_outcome,
    cosine_dist_matrix,
    pair_bootstrap_rsa,
    spearman_rsa,
    upper_tri,
)


PERTURBATION_COL = "perturbation"
NTC_LABEL = "control"
EXPECTED_CKPT_SHA = (
    "416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e"
)
EXPECTED_DATA_SHA = (
    "d4bedb53082735735993ced3cf30864b349d62f21078523b8b93137f55b333c9"
)
EXPECTED_NTC_CELLS = 11855
SPEC_NOTED_N_PERTURBATIONS = 105  # see prompts/phase6_5d_rsa.md prereqs


@dataclasses.dataclass
class RSAResult:
    rsa_real_mean: float
    rsa_real_ci95: Tuple[float, float]
    rsa_random_mean: float
    rsa_random_ci95: Tuple[float, float]
    rsa_random_per_seed: Dict[int, float]
    delta: float
    delta_ci95: Tuple[float, float]
    outcome: str
    interpretation_notes: str
    n_pert_included: int
    n_pert_dropped: int
    dropped_perturbations: List[str]
    perturbation_col: str
    ntc_label: str
    n_boot: int
    random_seeds: List[int]
    ckpt_real_sha: str
    data_sha: str
    n_cells_total: int
    n_genes_total: int
    n_genes_filtered: int
    ci_width_real: float
    ci_width_random: float
    elapsed_seconds: float
    schema_count_warning: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_dense_chunk(X, start: int, stop: int) -> np.ndarray:
    chunk = X[start:stop]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk, dtype=np.float32)


def _structural_zero_mask(X) -> np.ndarray:
    """Return boolean mask over genes that are nonzero in at least one cell."""
    if sp.issparse(X):
        nnz_per_col = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        nnz_per_col = (X != 0).sum(axis=0)
    return nnz_per_col > 0


def _compute_latents_batched(
    encoder: SimpleRNAEncoder,
    X,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run encoder forward pass on (n_cells, n_genes) raw X in chunks.

    Returns latents as float32 numpy array of shape (n_cells, latent_dim).
    """
    encoder = encoder.to(device).eval()
    n = X.shape[0]
    latents = np.empty((n, encoder.latent_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            stop = min(i + batch_size, n)
            chunk = _to_dense_chunk(X, i, stop)
            t = torch.from_numpy(chunk).to(device)
            z, _ = encoder(t)
            latents[i:stop] = z.detach().cpu().numpy()
    return latents


def _compute_response_centroids(
    X,
    nonzero_mask: np.ndarray,
    perturbation_labels: np.ndarray,
    non_ntc_perts: Sequence[str],
    ntc_label: str,
) -> np.ndarray:
    """Per-perturbation response centroid in log1p-space, NTC-relative.

    For each perturbation p:
        r_p = mean_cells(log1p(X_filtered)) - mean_NTC(log1p(X_filtered))
    """
    ntc_mask = perturbation_labels == ntc_label
    if ntc_mask.sum() == 0:
        raise ValueError(f"No cells with NTC label {ntc_label!r} found.")

    # NTC centroid first.
    ntc_idx = np.flatnonzero(ntc_mask)
    n_filt = int(nonzero_mask.sum())
    r_ntc = np.zeros(n_filt, dtype=np.float64)
    chunk = 4096
    for i in range(0, ntc_idx.size, chunk):
        rows = ntc_idx[i : i + chunk]
        sub = X[rows]
        if sp.issparse(sub):
            sub = sub.toarray()
        sub = np.asarray(sub, dtype=np.float32)[:, nonzero_mask]
        r_ntc += np.log1p(sub).sum(axis=0)
    r_ntc /= float(ntc_idx.size)

    R = np.empty((len(non_ntc_perts), n_filt), dtype=np.float32)
    for k, p in enumerate(non_ntc_perts):
        rows = np.flatnonzero(perturbation_labels == p)
        if rows.size == 0:
            raise ValueError(f"Perturbation {p!r} has 0 cells (filter logic bug).")
        acc = np.zeros(n_filt, dtype=np.float64)
        for i in range(0, rows.size, chunk):
            sub_rows = rows[i : i + chunk]
            sub = X[sub_rows]
            if sp.issparse(sub):
                sub = sub.toarray()
            sub = np.asarray(sub, dtype=np.float32)[:, nonzero_mask]
            acc += np.log1p(sub).sum(axis=0)
        acc /= float(rows.size)
        R[k] = (acc - r_ntc).astype(np.float32)
    return R


def _aggregate_latent_centroids(
    latents_all: np.ndarray,
    perturbation_labels: np.ndarray,
    non_ntc_perts: Sequence[str],
) -> np.ndarray:
    Z = np.empty((len(non_ntc_perts), latents_all.shape[1]), dtype=np.float32)
    for k, p in enumerate(non_ntc_perts):
        rows = np.flatnonzero(perturbation_labels == p)
        Z[k] = latents_all[rows].mean(axis=0)
    return Z


def run_rsa(
    adata_path: Path,
    ckpt_path: Path,
    seeds: Sequence[int],
    n_boot: int,
    min_cells_per_pert: int,
    art_dir: Path,
    bootstrap_rng_seed: int = 42,
) -> RSAResult:
    import anndata as ad

    t_start = time.time()
    art_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-flight: SHAs + schema ----
    data_sha = _hash_file(adata_path)
    if data_sha != EXPECTED_DATA_SHA:
        raise RuntimeError(
            f"Norman2019 SHA mismatch: expected {EXPECTED_DATA_SHA}, "
            f"got {data_sha}."
        )
    ckpt_sha = _hash_file(ckpt_path)
    if ckpt_sha != EXPECTED_CKPT_SHA:
        raise RuntimeError(
            f"Pretrain ckpt SHA mismatch: expected {EXPECTED_CKPT_SHA}, "
            f"got {ckpt_sha}."
        )

    print(f"[6.5d] loading {adata_path} ...", flush=True)
    adata = ad.read_h5ad(adata_path)
    n_cells, n_genes = adata.shape
    print(f"[6.5d] adata shape: {adata.shape}", flush=True)

    if PERTURBATION_COL not in adata.obs.columns:
        raise RuntimeError(
            f"Required column {PERTURBATION_COL!r} missing from adata.obs "
            "(data provenance regression)."
        )
    perturbation_labels = adata.obs[PERTURBATION_COL].astype(str).to_numpy()
    if NTC_LABEL not in set(perturbation_labels):
        raise RuntimeError(
            f"NTC label {NTC_LABEL!r} not present in obs[{PERTURBATION_COL!r}]."
        )
    n_ntc_cells = int((perturbation_labels == NTC_LABEL).sum())
    if abs(n_ntc_cells - EXPECTED_NTC_CELLS) > max(1, EXPECTED_NTC_CELLS // 100):
        raise RuntimeError(
            f"NTC cell count {n_ntc_cells} drifted from expected "
            f"{EXPECTED_NTC_CELLS} (>1%)."
        )

    # Counts.
    unique_perts, counts = np.unique(perturbation_labels, return_counts=True)
    n_pert_total = int(unique_perts.size)
    n_pert_non_ntc = n_pert_total - 1
    schema_count_warning = ""
    if abs(n_pert_total - SPEC_NOTED_N_PERTURBATIONS) > max(1, SPEC_NOTED_N_PERTURBATIONS // 100):
        schema_count_warning = (
            f"OBSERVED n_perturbations_total={n_pert_total} (non-NTC="
            f"{n_pert_non_ntc}) differs from spec-noted "
            f"{SPEC_NOTED_N_PERTURBATIONS}. Column + NTC label confirmed; "
            "RSA contract unaffected (more pairs improves estimate)."
        )
        warnings.warn(schema_count_warning, UserWarning, stacklevel=2)

    # MIN_CELLS_PER_PERT filter.
    keep_mask = counts >= min_cells_per_pert
    kept_perts = unique_perts[keep_mask].tolist()
    dropped_perts = unique_perts[~keep_mask].tolist()
    if NTC_LABEL not in kept_perts:
        raise RuntimeError(
            f"NTC {NTC_LABEL!r} dropped by MIN_CELLS_PER_PERT filter "
            "(insufficient NTC cells)."
        )
    non_ntc_perts = sorted(p for p in kept_perts if p != NTC_LABEL)
    if len(non_ntc_perts) < 30:
        raise RuntimeError(
            f"Only {len(non_ntc_perts)} non-NTC perturbations passed "
            f"MIN_CELLS_PER_PERT={min_cells_per_pert} — below tripwire 30."
        )
    print(
        f"[6.5d] kept {len(non_ntc_perts)} non-NTC perturbations; "
        f"dropped {len(dropped_perts)}.",
        flush=True,
    )

    # ---- Structural-zero filter (response space only) ----
    nonzero_mask = _structural_zero_mask(adata.X)
    n_genes_filtered = int(nonzero_mask.sum())
    print(
        f"[6.5d] structural-zero filter: {n_genes_filtered:,}/{n_genes:,} "
        f"genes retained.",
        flush=True,
    )

    # ---- Response centroids ----
    print("[6.5d] computing response centroids (log1p space, NTC-relative) ...",
          flush=True)
    R = _compute_response_centroids(
        adata.X, nonzero_mask, perturbation_labels, non_ntc_perts, NTC_LABEL
    )
    if not np.isfinite(R).all():
        raise RuntimeError("Response centroids contain NaN/Inf.")
    np.save(art_dir / "response_centroids.npy", R)

    # ---- Latent centroids: pretrained ----
    device = _select_device()
    print(f"[6.5d] device: {device}", flush=True)
    print("[6.5d] loading pretrained encoder ...", flush=True)
    real_encoder = load_pretrained_simple_rna_encoder(
        ckpt_path, expected_schema_version=1
    )
    if real_encoder.n_genes != n_genes:
        raise RuntimeError(
            f"Encoder n_genes={real_encoder.n_genes} ≠ adata n_genes={n_genes}."
        )
    print("[6.5d] forward pass: pretrained encoder ...", flush=True)
    Z_real_all = _compute_latents_batched(real_encoder, adata.X, device)
    if not np.isfinite(Z_real_all).all():
        raise RuntimeError("Pretrained latents contain NaN/Inf.")
    Z_real = _aggregate_latent_centroids(
        Z_real_all, perturbation_labels, non_ntc_perts
    )
    np.save(art_dir / "latents_real.npy", Z_real)
    del Z_real_all

    # ---- Latent centroids: 5 random inits ----
    Z_random: Dict[int, np.ndarray] = {}
    for seed in seeds:
        print(f"[6.5d] forward pass: random init seed={seed} ...", flush=True)
        torch.manual_seed(seed)
        rand_encoder = SimpleRNAEncoder(
            n_genes=real_encoder.n_genes,
            hidden_dim=real_encoder.hidden_dim,
            latent_dim=real_encoder.latent_dim,
        )
        Z_rand_all = _compute_latents_batched(rand_encoder, adata.X, device)
        if not np.isfinite(Z_rand_all).all():
            raise RuntimeError(f"Random latents (seed={seed}) contain NaN/Inf.")
        Z_random[seed] = _aggregate_latent_centroids(
            Z_rand_all, perturbation_labels, non_ntc_perts
        )
        np.save(art_dir / f"latents_random_seed{seed}.npy", Z_random[seed])
        del Z_rand_all, rand_encoder

    # ---- Distance matrices + flatten ----
    print("[6.5d] computing cosine distance matrices ...", flush=True)
    D_r = cosine_dist_matrix(R)
    D_z_real = cosine_dist_matrix(Z_real)
    D_z_random = {seed: cosine_dist_matrix(Z) for seed, Z in Z_random.items()}

    flat_r = upper_tri(D_r)
    flat_z_real = upper_tri(D_z_real)
    flat_z_random = {seed: upper_tri(D) for seed, D in D_z_random.items()}
    n_pairs = flat_r.shape[0]
    print(f"[6.5d] n_pairs={n_pairs:,}", flush=True)

    # ---- Point estimates ----
    rsa_real_point = spearman_rsa(flat_z_real, flat_r)
    rsa_random_per_seed = {
        seed: spearman_rsa(flat_z_random[seed], flat_r) for seed in seeds
    }
    if any(np.isnan(v) for v in [rsa_real_point, *rsa_random_per_seed.values()]):
        raise RuntimeError(
            "NaN in RSA point estimate — degenerate distance matrix."
        )
    rsa_random_point = float(np.mean(list(rsa_random_per_seed.values())))

    # ---- Pair-bootstrap CIs ----
    print(f"[6.5d] bootstrap (n_boot={n_boot}) ...", flush=True)
    rng = np.random.default_rng(bootstrap_rng_seed)

    boot_real = pair_bootstrap_rsa(flat_z_real, flat_r, n_boot=n_boot, rng=rng)
    ci_real = (
        float(np.percentile(boot_real, 2.5)),
        float(np.percentile(boot_real, 97.5)),
    )
    np.save(art_dir / "boot_real.npy", boot_real)

    boot_random_aggregated = []
    for seed in seeds:
        boot_seed = pair_bootstrap_rsa(
            flat_z_random[seed], flat_r, n_boot=n_boot, rng=rng
        )
        np.save(art_dir / f"boot_random_seed{seed}.npy", boot_seed)
        boot_random_aggregated.append(boot_seed)
    boot_random_aggregated = np.concatenate(boot_random_aggregated)
    ci_random = (
        float(np.percentile(boot_random_aggregated, 2.5)),
        float(np.percentile(boot_random_aggregated, 97.5)),
    )
    np.save(art_dir / "boot_random_aggregated.npy", boot_random_aggregated)

    # ---- Delta + delta CI (paired pair-bootstrap) ----
    print("[6.5d] bootstrap delta (paired resample) ...", flush=True)
    boot_delta = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        r_real, _ = spearmanr(flat_z_real[idx], flat_r[idx])
        if np.isnan(r_real):
            r_real = 0.0
        rand_vals = []
        for seed in seeds:
            r_rnd, _ = spearmanr(flat_z_random[seed][idx], flat_r[idx])
            rand_vals.append(0.0 if np.isnan(r_rnd) else r_rnd)
        boot_delta[i] = r_real - float(np.mean(rand_vals))
    ci_delta = (
        float(np.percentile(boot_delta, 2.5)),
        float(np.percentile(boot_delta, 97.5)),
    )
    np.save(art_dir / "boot_delta.npy", boot_delta)

    delta_point = rsa_real_point - rsa_random_point
    ci_width_real = ci_real[1] - ci_real[0]
    ci_width_random = ci_random[1] - ci_random[0]

    outcome, notes = classify_outcome(
        rsa_real=rsa_real_point,
        delta=delta_point,
        ci_delta=ci_delta,
        ci_width_real=ci_width_real,
    )

    elapsed = time.time() - t_start

    return RSAResult(
        rsa_real_mean=rsa_real_point,
        rsa_real_ci95=ci_real,
        rsa_random_mean=rsa_random_point,
        rsa_random_ci95=ci_random,
        rsa_random_per_seed={int(k): float(v) for k, v in rsa_random_per_seed.items()},
        delta=float(delta_point),
        delta_ci95=ci_delta,
        outcome=outcome,
        interpretation_notes=notes,
        n_pert_included=len(non_ntc_perts),
        n_pert_dropped=len(dropped_perts),
        dropped_perturbations=[str(p) for p in dropped_perts],
        perturbation_col=PERTURBATION_COL,
        ntc_label=NTC_LABEL,
        n_boot=n_boot,
        random_seeds=list(seeds),
        ckpt_real_sha=ckpt_sha,
        data_sha=data_sha,
        n_cells_total=int(n_cells),
        n_genes_total=int(n_genes),
        n_genes_filtered=int(n_genes_filtered),
        ci_width_real=float(ci_width_real),
        ci_width_random=float(ci_width_random),
        elapsed_seconds=float(elapsed),
        schema_count_warning=schema_count_warning,
    )


def _result_to_json(result: RSAResult) -> dict:
    d = dataclasses.asdict(result)
    # Convert tuples to lists for clean JSON.
    d["rsa_real_ci95"] = list(result.rsa_real_ci95)
    d["rsa_random_ci95"] = list(result.rsa_random_ci95)
    d["delta_ci95"] = list(result.delta_ci95)
    return d


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adata", type=Path, required=True)
    p.add_argument("--real-ckpt", type=Path, required=True)
    p.add_argument("--random-seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--min-cells-per-pert", type=int, default=20)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--art-dir", type=Path, required=True)
    p.add_argument("--bootstrap-rng-seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> RSAResult:
    args = _parse_args(argv)
    seeds = [int(s) for s in args.random_seeds.split(",")]
    if args.n_boot != 1000:
        warnings.warn(
            f"n_boot={args.n_boot} ≠ 1000 (locked contract default).",
            UserWarning,
            stacklevel=2,
        )
    result = run_rsa(
        adata_path=args.adata,
        ckpt_path=args.real_ckpt,
        seeds=seeds,
        n_boot=args.n_boot,
        min_cells_per_pert=args.min_cells_per_pert,
        art_dir=args.art_dir,
        bootstrap_rng_seed=args.bootstrap_rng_seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_result_to_json(result), indent=2))
    print(f"\n[6.5d] OUTCOME = {result.outcome}", flush=True)
    print(f"[6.5d] {result.interpretation_notes}", flush=True)
    print(f"[6.5d] wrote {args.out_json}", flush=True)
    return result


if __name__ == "__main__":
    main()
