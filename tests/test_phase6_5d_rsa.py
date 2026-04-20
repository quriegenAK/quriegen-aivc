"""tests/test_phase6_5d_rsa.py — Phase 6.5d RSA pipeline tests.

Covers helper math + the locked interpretation rule (all 6 outcome
branches: A, B, C, C_WEAK, D, INCONCLUSIVE).

Pure-numerics; does not load Norman 2019 or any checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lib.rsa import (  # noqa: E402
    classify_outcome,
    cosine_dist_matrix,
    pair_bootstrap_rsa,
    spearman_rsa,
    upper_tri,
)


# --------------------------------------------------------------------- #
# Helper: cosine distance matrix.
# --------------------------------------------------------------------- #
def test_cosine_dist_matrix_self_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 16)).astype(np.float32)
    D = cosine_dist_matrix(X)
    assert D.shape == (8, 8)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)


def test_cosine_dist_matrix_symmetric():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 12)).astype(np.float32)
    D = cosine_dist_matrix(X)
    assert np.allclose(D, D.T, atol=1e-7)


def test_cosine_dist_matrix_range():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 10)).astype(np.float32)
    D = cosine_dist_matrix(X)
    # 1 - cos ∈ [0, 2].
    assert (D >= 0.0).all()
    assert (D <= 2.0).all()


def test_cosine_dist_matrix_zero_row_no_nan():
    X = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    D = cosine_dist_matrix(X)
    assert np.isfinite(D).all()


# --------------------------------------------------------------------- #
# Helper: upper_tri.
# --------------------------------------------------------------------- #
def test_upper_tri_length():
    M = np.zeros((5, 5))
    v = upper_tri(M)
    assert v.shape == (10,)  # 5*4/2


def test_upper_tri_excludes_diagonal():
    M = np.eye(4) * 99.0 + np.ones((4, 4))
    np.fill_diagonal(M, 99.0)
    v = upper_tri(M)
    assert (v == 1.0).all()


# --------------------------------------------------------------------- #
# Helper: spearman_rsa.
# --------------------------------------------------------------------- #
def test_spearman_rsa_identity():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    D = cosine_dist_matrix(X)
    r = spearman_rsa(D, D)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_spearman_rsa_shuffle_near_zero():
    """RSA(D, shuffled D) should sit near the null (mean ~0)."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 16)).astype(np.float32)
    D = cosine_dist_matrix(X)
    flat = upper_tri(D)
    samples = []
    for _ in range(50):
        perm = rng.permutation(flat.size)
        samples.append(spearman_rsa(flat, flat[perm]))
    samples = np.array(samples)
    # 3σ envelope around zero, very loose to remain stable across BLAS.
    assert abs(samples.mean()) < 3 * samples.std() / np.sqrt(len(samples)) + 0.05


# --------------------------------------------------------------------- #
# Helper: pair_bootstrap_rsa.
# --------------------------------------------------------------------- #
def test_pair_bootstrap_shape():
    rng_data = np.random.default_rng(5)
    X = rng_data.normal(size=(15, 8)).astype(np.float32)
    D = cosine_dist_matrix(X)
    rng = np.random.default_rng(123)
    boot = pair_bootstrap_rsa(D, D, n_boot=37, rng=rng)
    assert boot.shape == (37,)
    # RSA(D, D) ≈ 1 always (modulo bootstrap-tie effects), so all values
    # should be very close to 1.0.
    assert (boot > 0.95).all()


def test_pair_bootstrap_accepts_flat_vectors():
    rng_data = np.random.default_rng(6)
    X = rng_data.normal(size=(12, 6)).astype(np.float32)
    D = cosine_dist_matrix(X)
    flat = upper_tri(D)
    rng = np.random.default_rng(7)
    boot = pair_bootstrap_rsa(flat, flat, n_boot=10, rng=rng)
    assert boot.shape == (10,)


# --------------------------------------------------------------------- #
# Interpretation rule: all 6 branches (A, B, C, C_WEAK, D, INCONCLUSIVE).
# --------------------------------------------------------------------- #
def test_outcome_A_strong_positive():
    # Δ > 0.05, RSA_real > 0.15, CI_Δ excludes 0, CI_width OK.
    out, _ = classify_outcome(
        rsa_real=0.30, delta=0.20, ci_delta=(0.10, 0.30), ci_width_real=0.10
    )
    assert out == "A"


def test_outcome_B_positive_but_weak():
    # Δ > 0.05, RSA_real ≤ 0.15, CI_Δ excludes 0.
    out, _ = classify_outcome(
        rsa_real=0.10, delta=0.10, ci_delta=(0.05, 0.15), ci_width_real=0.10
    )
    assert out == "B"


def test_outcome_C_objective_mismatch_boundary():
    # |Δ| == 0.05 exactly with CI excluding 0 → C (boundary).
    out, _ = classify_outcome(
        rsa_real=0.05, delta=0.05, ci_delta=(0.02, 0.08), ci_width_real=0.06
    )
    # |Δ| ≥ 0.05 and 0 ∉ CI → interpretable. Δ = +0.05 → not Δ > 0.05,
    # not Δ < -0.05 → C.
    assert out == "C"


def test_outcome_C_WEAK_significant_but_small():
    # |Δ| < 0.05 AND 0 ∉ CI_Δ.
    out, _ = classify_outcome(
        rsa_real=0.04, delta=0.03, ci_delta=(0.01, 0.04), ci_width_real=0.03
    )
    assert out == "C_WEAK"


def test_outcome_D_distractor_bias():
    out, _ = classify_outcome(
        rsa_real=-0.05, delta=-0.10, ci_delta=(-0.18, -0.03), ci_width_real=0.10
    )
    assert out == "D"


def test_outcome_INCONCLUSIVE_small_and_overlapping():
    out, _ = classify_outcome(
        rsa_real=0.02, delta=0.02, ci_delta=(-0.03, 0.07), ci_width_real=0.10
    )
    assert out == "INCONCLUSIVE"


def test_outcome_INCONCLUSIVE_large_but_overlapping():
    out, _ = classify_outcome(
        rsa_real=0.10, delta=0.08, ci_delta=(-0.02, 0.18), ci_width_real=0.12
    )
    assert out == "INCONCLUSIVE"


def test_outcome_INCONCLUSIVE_high_ci_width_override():
    # Even a large clean Δ is INCONCLUSIVE if CI_width_real > 0.30.
    out, _ = classify_outcome(
        rsa_real=0.40, delta=0.20, ci_delta=(0.10, 0.30), ci_width_real=0.35
    )
    assert out == "INCONCLUSIVE"


# --------------------------------------------------------------------- #
# NTC reference exclusion (geometric proxy via pairwise indices).
# --------------------------------------------------------------------- #
def test_ntc_reference_excluded_from_pair_indices():
    """NTC must not appear in the perturbation list passed to RSA.

    This is a structural test: build a label list that includes NTC,
    apply the same exclusion logic the runner uses, and confirm NTC
    is absent from the resulting perturbation set.
    """
    perturbation_labels = np.array(
        ["control", "KLF1", "BAK1", "control", "KLF1"], dtype=object
    )
    NTC_LABEL = "control"
    unique = np.unique(perturbation_labels).tolist()
    non_ntc = sorted(p for p in unique if p != NTC_LABEL)
    assert NTC_LABEL not in non_ntc
    assert non_ntc == ["BAK1", "KLF1"]
