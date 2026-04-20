"""scripts/lib/rsa.py — Phase 6.5d RSA helper module.

Pure, side-effect-free numerics for representation similarity analysis.
Used by ``scripts/phase6_5d_rsa.py`` and exercised by
``tests/test_phase6_5d_rsa.py``.

Contract is locked by ``prompts/phase6_5d_rsa.md`` — do NOT modify
behaviour here without updating the contract first.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import spearmanr


def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix for rows of ``X`` (n, d) → (n, n).

    Distance is ``1 − cos(x_i, x_j)``. Self-distance is 0. Zero-norm
    rows are treated as having unit norm to avoid NaN; all distances
    involving a zero row collapse to 1.0 (orthogonal). Matrix is
    symmetric and clipped to ``[0, 2]``.
    """
    if X.ndim != 2:
        raise ValueError(f"cosine_dist_matrix expects 2D array, got shape {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid divide-by-zero for degenerate rows.
    safe = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / safe
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    D = 1.0 - sim
    # Force exact zero diagonal (numerical noise can leave 1e-7).
    np.fill_diagonal(D, 0.0)
    # Force exact symmetry.
    D = 0.5 * (D + D.T)
    return D


def upper_tri(M: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle (k=1) of a square matrix."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"upper_tri expects square 2D array, got shape {M.shape}")
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def _flatten_if_matrix(A: np.ndarray) -> np.ndarray:
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        return upper_tri(A)
    if A.ndim == 1:
        return A
    raise ValueError(
        f"Expected 1D vector or square 2D matrix, got shape {A.shape}"
    )


def spearman_rsa(D_a: np.ndarray, D_b: np.ndarray) -> float:
    """Spearman correlation between flattened upper-triangles of two
    distance matrices. Accepts either 2D distance matrices or already-
    flattened 1D pair vectors of equal length.
    """
    a = _flatten_if_matrix(D_a)
    b = _flatten_if_matrix(D_b)
    if a.shape != b.shape:
        raise ValueError(
            f"Pair-vector shape mismatch: {a.shape} vs {b.shape}"
        )
    r, _ = spearmanr(a, b)
    if np.isnan(r):
        return float("nan")
    return float(r)


def pair_bootstrap_rsa(
    D_a: np.ndarray,
    D_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pair-bootstrap Spearman RSA.

    Resamples ``n_pairs`` indices with replacement from the flattened
    upper-triangle pair vectors and computes Spearman per resample.
    Returns array of length ``n_boot``.
    """
    a = _flatten_if_matrix(D_a)
    b = _flatten_if_matrix(D_b)
    if a.shape != b.shape:
        raise ValueError(
            f"Pair-vector shape mismatch: {a.shape} vs {b.shape}"
        )
    n = a.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(a[idx], b[idx])
        out[i] = r if not np.isnan(r) else 0.0
    return out


def classify_outcome(
    rsa_real: float,
    delta: float,
    ci_delta: Tuple[float, float],
    ci_width_real: float,
    *,
    delta_threshold: float = 0.05,
    rsa_real_threshold: float = 0.15,
    ci_width_max: float = 0.30,
) -> Tuple[str, str]:
    """Apply the locked Phase 6.5d interpretation rule.

    Returns ``(outcome, interpretation_notes)`` where outcome is one of
    {"A", "B", "C", "C_WEAK", "D", "INCONCLUSIVE"}.

    Locked thresholds (do not retune post-hoc):
    * ``|Δ| < 0.05`` → magnitude below pre-registered floor.
    * ``RSA_real > 0.15`` → outcome A vs B split.
    * ``CI_width_real > 0.30`` → INCONCLUSIVE override (sampling noise).
    """
    lower, upper = float(ci_delta[0]), float(ci_delta[1])
    abs_delta = abs(delta)
    ci_includes_zero = (lower <= 0.0 <= upper)

    if ci_width_real > ci_width_max:
        return (
            "INCONCLUSIVE",
            f"CI_width_real={ci_width_real:.4f} > {ci_width_max}; "
            "sampling-noise limited.",
        )

    if abs_delta < delta_threshold and ci_includes_zero:
        return (
            "INCONCLUSIVE",
            f"|Δ|={abs_delta:.4f} < {delta_threshold} and CI_Δ=({lower:.4f}, "
            f"{upper:.4f}) includes 0.",
        )
    if abs_delta >= delta_threshold and ci_includes_zero:
        return (
            "INCONCLUSIVE",
            f"|Δ|={abs_delta:.4f} ≥ {delta_threshold} but CI_Δ=("
            f"{lower:.4f}, {upper:.4f}) overlaps zero (sampling noise too high).",
        )
    if abs_delta < delta_threshold and not ci_includes_zero:
        return (
            "C_WEAK",
            f"|Δ|={abs_delta:.4f} < {delta_threshold} but statistically "
            f"significant (CI_Δ=({lower:.4f}, {upper:.4f}) excludes 0). "
            "Objective mismatch confirmed but with magnitude caveat.",
        )

    if delta > delta_threshold:
        if rsa_real > rsa_real_threshold:
            return (
                "A",
                f"Δ={delta:.4f} > {delta_threshold} and RSA_real={rsa_real:.4f} "
                f"> {rsa_real_threshold}. Pretrain captures perturbation "
                "structure; 6.5c FAIL is probe-artifact.",
            )
        return (
            "B",
            f"Δ={delta:.4f} > {delta_threshold} but RSA_real={rsa_real:.4f} "
            f"≤ {rsa_real_threshold}. Pretrain helps but weakly.",
        )
    if delta < -delta_threshold:
        return (
            "D",
            f"Δ={delta:.4f} < -{delta_threshold}. Active distractor bias; "
            "PBMC-pretrain erodes K562-relevant axes.",
        )
    # Boundary case: |Δ| == delta_threshold exactly with CI excluding 0.
    return (
        "C",
        f"-{delta_threshold} ≤ Δ={delta:.4f} ≤ +{delta_threshold} (boundary). "
        "Objective mismatch; pretrain neither helps nor hurts.",
    )
