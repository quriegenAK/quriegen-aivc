"""Tests for scripts/harmonize_calderon_peaks.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from harmonize_calderon_peaks import (  # noqa: E402
    apply_projection,
    compute_projection_matrix,
)


def _var(rows):
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    df.index = [f"{c}:{s}-{e}" for c, s, e in rows]
    df.index.name = "peak_id"
    return df


def test_identical_peak_sets_full_coverage():
    rows = [("chr1", 100, 200), ("chr1", 500, 600), ("chr2", 50, 150)]
    M, stats = compute_projection_matrix(_var(rows), _var(rows))
    assert M.shape == (3, 3)
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    assert stats["frac_calderon_with_any_overlap"] == 1.0
    assert stats["n_orphan_calderon"] == 0


def test_no_overlap():
    dogma = _var([("chr1", 100, 200)])
    calderon = _var([("chr1", 500, 600), ("chr2", 50, 150)])
    M, stats = compute_projection_matrix(dogma, calderon)
    assert M.nnz == 0
    assert stats["n_orphan_calderon"] == 2
    assert stats["frac_calderon_with_any_overlap"] == 0.0


def test_partial_overlap_fractional_weight():
    # Calderon (100-200), DOGMA (150-250). Overlap=50, calderon_len=100. Weight=0.5.
    dogma = _var([("chr1", 150, 250)])
    calderon = _var([("chr1", 100, 200)])
    M, stats = compute_projection_matrix(dogma, calderon)
    assert M.shape == (1, 1)
    np.testing.assert_allclose(M.toarray()[0, 0], 0.5, atol=1e-6)
    assert stats["mean_row_sum"] < 1.0


def test_multi_mapping_calderon_to_dogma():
    # C: chr1:100-300 (len 200) overlaps D1:100-200 + D2:200-300; weights 0.5+0.5
    dogma = _var([("chr1", 100, 200), ("chr1", 200, 300)])
    calderon = _var([("chr1", 100, 300)])
    M, _ = compute_projection_matrix(dogma, calderon)
    arr = M.toarray()
    np.testing.assert_allclose(arr[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(arr[0, 1], 0.5, atol=1e-6)
    np.testing.assert_allclose(arr.sum(axis=1)[0], 1.0, atol=1e-6)


def test_row_sums_bounded():
    dogma = _var([("chr1", 100, 150), ("chr1", 150, 200), ("chr1", 200, 250),
                  ("chr1", 250, 300), ("chr1", 300, 400)])
    calderon = _var([("chr1", 100, 400)])
    M, _ = compute_projection_matrix(dogma, calderon)
    row_sum = M.sum(axis=1)[0, 0]
    assert row_sum <= 1.0 + 1e-6
    np.testing.assert_allclose(row_sum, 1.0, atol=1e-6)


def test_apply_projection_mass_conservation():
    dogma = _var([("chr1", 100, 200), ("chr1", 200, 300)])
    calderon = _var([("chr1", 100, 300)])
    M, _ = compute_projection_matrix(dogma, calderon)

    X = sp.csr_matrix(np.array([[10.0], [20.0], [5.0]]))
    X_proj = apply_projection(X, M)
    assert X_proj.shape == (3, 2)
    np.testing.assert_allclose(
        np.asarray(X_proj.sum(axis=1)).ravel(),
        np.asarray(X.sum(axis=1)).ravel(),
        atol=1e-6,
    )


@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_dogma_calderon():
    """Smoke test against real DOGMA + Calderon h5ad files."""
    dogma_path = Path(os.environ.get(
        "DOGMA_H5AD", "data/phase6_5g_2/dogma_lll.h5ad"))
    calderon_path = Path(os.environ.get(
        "CALDERON_H5AD", "data/calderon2019/calderon_atac.h5ad"))
    if not dogma_path.exists() or not calderon_path.exists():
        pytest.skip("DOGMA or Calderon h5ad missing")

    dogma = ad.read_h5ad(dogma_path, backed="r")
    calderon = ad.read_h5ad(calderon_path, backed="r")

    M, stats = compute_projection_matrix(dogma.var.copy(), calderon.var.copy())
    assert M.shape == (len(calderon.var), len(dogma.var))
    assert stats["frac_calderon_with_any_overlap"] > 0.05, (
        f"Suspiciously low overlap: {stats['frac_calderon_with_any_overlap']:.3%}"
    )
    assert stats["max_row_sum"] <= 1.0 + 1e-5
    print(f"Real-data smoke stats: {stats}")
