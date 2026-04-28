"""Tests for scripts/harmonize_calderon_peaks.py.

Synthetic tests cover both AnnData layouts:
- .var-style: peaks in adata.var (Calderon convention)
- DOGMA-style: peaks in adata.uns['atac_feature_names'] (DOGMA convention)

Real-data smoke (load-bearing, gated by @pytest.mark.real_data) runs
the full pipeline against actual on-disk DOGMA LLL + Calderon h5ads.
This is the regression test for PR #38's layout-dispatch bug fix.
"""
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
    extract_peaks_to_var_format,
)


def _var(rows):
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    df.index = [f"{c}:{s}-{e}" for c, s, e in rows]
    df.index.name = "peak_id"
    return df


def _make_var_layout_adata(rows):
    """Build an AnnData with peaks in .var (Calderon-style)."""
    var = _var(rows)
    n_samples = 3
    X = sp.csr_matrix(np.zeros((n_samples, len(rows)), dtype=np.float32))
    return ad.AnnData(X=X, var=var)


def _make_dogma_layout_adata(peak_strs, n_cells=10):
    """Build an AnnData with peaks in .uns['atac_feature_names'] (DOGMA-style)."""
    n_peaks = len(peak_strs)
    n_genes = 5  # arbitrary; .X holds RNA in DOGMA
    rng = np.random.default_rng(0)
    return ad.AnnData(
        X=sp.csr_matrix(rng.poisson(0.5, (n_cells, n_genes)).astype(np.float32)),
        obsm={"atac_peaks": sp.csr_matrix(np.zeros((n_cells, n_peaks), dtype=np.float32))},
        uns={"atac_feature_names": np.asarray(peak_strs, dtype=object)},
    )


# --- compute_projection_matrix tests (existing, unchanged) -------------------

def test_identical_peak_sets_full_coverage():
    rows = [("chr1", 100, 200), ("chr1", 500, 600), ("chr2", 50, 150)]
    M, stats = compute_projection_matrix(_var(rows), _var(rows))
    assert M.shape == (3, 3)
    np.testing.assert_allclose(np.asarray(M.sum(axis=1)).ravel(), 1.0, atol=1e-6)
    assert stats["frac_calderon_with_any_overlap"] == 1.0


def test_no_overlap():
    dogma = _var([("chr1", 100, 200)])
    calderon = _var([("chr1", 500, 600), ("chr2", 50, 150)])
    M, stats = compute_projection_matrix(dogma, calderon)
    assert M.nnz == 0
    assert stats["n_orphan_calderon"] == 2


def test_partial_overlap_fractional_weight():
    dogma = _var([("chr1", 150, 250)])
    calderon = _var([("chr1", 100, 200)])
    M, stats = compute_projection_matrix(dogma, calderon)
    np.testing.assert_allclose(M.toarray()[0, 0], 0.5, atol=1e-6)


def test_multi_mapping_calderon_to_dogma():
    dogma = _var([("chr1", 100, 200), ("chr1", 200, 300)])
    calderon = _var([("chr1", 100, 300)])
    M, _ = compute_projection_matrix(dogma, calderon)
    arr = M.toarray()
    np.testing.assert_allclose(arr[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(arr[0, 1], 0.5, atol=1e-6)


def test_row_sums_bounded():
    dogma = _var([("chr1", 100, 150), ("chr1", 150, 200), ("chr1", 200, 250),
                  ("chr1", 250, 300), ("chr1", 300, 400)])
    calderon = _var([("chr1", 100, 400)])
    M, _ = compute_projection_matrix(dogma, calderon)
    np.testing.assert_allclose(M.sum(axis=1)[0, 0], 1.0, atol=1e-6)


def test_apply_projection_mass_conservation():
    dogma = _var([("chr1", 100, 200), ("chr1", 200, 300)])
    calderon = _var([("chr1", 100, 300)])
    M, _ = compute_projection_matrix(dogma, calderon)
    X = sp.csr_matrix(np.array([[10.0], [20.0], [5.0]]))
    X_proj = apply_projection(X, M)
    np.testing.assert_allclose(
        np.asarray(X_proj.sum(axis=1)).ravel(),
        np.asarray(X.sum(axis=1)).ravel(),
        atol=1e-6,
    )


def test_dogma_self_overlapping_peaks_row_sum_above_one():
    """PR #38: real DOGMA peak sets contain peaks that overlap each other
    (~78% of DOGMA LLL peaks have a sibling overlap). When two DOGMA peaks
    both fully cover the same Calderon peak, the sum of fractional weights
    legitimately exceeds 1.0. The compute path must NOT raise; a stat
    captures the redundancy.
    """
    # D1=[100,200] and D2=[120,180] both cover Calderon C=[100,200] (length 100).
    # weight(C, D1) = 100/100 = 1.0; weight(C, D2) = 60/100 = 0.6. Sum = 1.6.
    dogma = _var([("chr1", 100, 200), ("chr1", 120, 180)])
    calderon = _var([("chr1", 100, 200)])
    M, stats = compute_projection_matrix(dogma, calderon)
    assert M.shape == (1, 2)
    np.testing.assert_allclose(M.sum(axis=1)[0, 0], 1.6, atol=1e-5)
    assert stats["max_row_sum"] > 1.0
    assert stats["n_calderon_with_dogma_redundancy"] == 1
    assert stats["frac_calderon_with_dogma_redundancy"] == 1.0


def test_compute_projection_raises_only_on_extreme_overcount(tmp_path):
    """The defensive raise still fires on truly absurd overcount (>2.0),
    which only happens with duplicate Calderon coordinates or a join bug.
    """
    # 5 DOGMA peaks each fully covering Calderon -> sum = 5.0 > 2.0 cap
    dogma = _var([("chr1", 100, 200)] * 5)
    # Note: _var() builds duplicate index strings; pyranges still works.
    calderon = _var([("chr1", 100, 200)])
    # Need to give DOGMA distinct rows; rebuild with offset that all still cover
    dogma = pd.DataFrame({
        "chrom": ["chr1"] * 5,
        "start": [100, 100, 100, 100, 100],
        "end":   [200, 200, 200, 200, 200],
    })
    dogma.index = [f"d{i}" for i in range(5)]
    with pytest.raises(RuntimeError, match="exceed 2.0"):
        compute_projection_matrix(dogma, calderon)


# --- PR #38: extract_peaks_to_var_format tests -------------------------------

def test_extract_peaks_var_layout():
    """Calderon-style: peaks in .var with chrom/start/end columns."""
    rows = [("chr1", 100, 200), ("chr2", 500, 600)]
    a = _make_var_layout_adata(rows)
    df = extract_peaks_to_var_format(a, "test")
    assert list(df.columns) == ["chrom", "start", "end"]
    assert len(df) == 2
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["start"] == 100


def test_extract_peaks_dogma_layout_colon_dash():
    """DOGMA-style with colon-dash separator (chr1:1000-2000)."""
    a = _make_dogma_layout_adata([
        "chr1:1000-2000",
        "chrX:100-500",
        "chr20:51489470-51489886",
    ])
    df = extract_peaks_to_var_format(a, "test")
    assert len(df) == 3
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["start"] == 1000
    assert df.iloc[2]["chrom"] == "chr20"
    assert df.iloc[2]["start"] == 51489470
    assert df.iloc[2]["end"] == 51489886


def test_extract_peaks_dogma_layout_underscore():
    """DOGMA-style with underscore separator (chr1_1000_2000)."""
    a = _make_dogma_layout_adata([
        "chr1_10045_10517",
        "chr2_20000_20500",
    ])
    df = extract_peaks_to_var_format(a, "test")
    assert len(df) == 2
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["start"] == 10045
    assert df.iloc[0]["end"] == 10517


def test_extract_peaks_unrecognized_layout_raises():
    """Empty .var + no .uns['atac_feature_names'] should raise with diagnostic."""
    a = ad.AnnData(X=np.zeros((3, 5)))
    with pytest.raises(ValueError, match="unrecognized peak layout"):
        extract_peaks_to_var_format(a, "test")


def test_extract_peaks_uns_takes_precedence():
    """If BOTH .uns['atac_feature_names'] AND .var have peaks, .uns wins.

    Rationale: DOGMA layout always populates .uns; .var-style scripts
    populate .var but never set .uns['atac_feature_names']. The .uns
    layout is the more specific signal.
    """
    rows = [("chr99", 1, 2)]
    a = _make_var_layout_adata(rows)
    a.uns["atac_feature_names"] = np.asarray(["chr1:100-200"], dtype=object)
    df = extract_peaks_to_var_format(a, "test")
    # Should have used .uns, not .var
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["start"] == 100


# --- Real-data smoke (load-bearing for PR #38 + ongoing) ---------------------

@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_dogma_calderon():
    """Smoke against actual on-disk DOGMA LLL + Calderon h5ads.

    PR #38 fix: harmonize script previously crashed on DOGMA's
    .uns['atac_feature_names'] layout. After this PR, the full pipeline
    runs end-to-end and produces stats consistent with Path A diagnostic
    (frac_overlap ~9.81% pre-liftOver; >5% threshold).

    Default DOGMA path matches the actual assembly script output:
    data/phase6_5g_2/dogma_h5ads/dogma_lll.h5ad
    (NOT data/phase6_5g_2/dogma_lll.h5ad -- that was the original
    expected path; assembly puts h5ads in dogma_h5ads/ subdir).
    """
    dogma_path = Path(os.environ.get(
        "DOGMA_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_lll.h5ad"))
    calderon_path = Path(os.environ.get(
        "CALDERON_H5AD",
        "data/calderon2019/calderon_atac.h5ad"))
    if not dogma_path.exists() or not calderon_path.exists():
        pytest.skip(f"DOGMA or Calderon h5ad missing")

    dogma = ad.read_h5ad(dogma_path, backed="r")
    calderon = ad.read_h5ad(calderon_path, backed="r")

    # PR #38: layout-aware extraction
    dogma_var = extract_peaks_to_var_format(dogma, name="dogma")
    calderon_var = extract_peaks_to_var_format(calderon, name="calderon")

    M, stats = compute_projection_matrix(dogma_var, calderon_var)

    # Shape sanity
    assert M.shape == (len(calderon_var), len(dogma_var))

    # Coordinate-system canary (Path A diagnostic measured 9.81%)
    assert stats["frac_calderon_with_any_overlap"] > 0.05, (
        f"frac_overlap below 5% canary: {stats['frac_calderon_with_any_overlap']:.4%}. "
        "Likely cause: hg19/hg38 mismatch worsened or DOGMA peak set changed."
    )
    # PR #38: real DOGMA peak sets contain self-overlapping peaks; row sums
    # legitimately exceed 1.0 in those regions. Hard cap is at 2.0 (above which
    # we'd suspect duplicate Calderon coordinates / join bug).
    assert stats["max_row_sum"] <= 2.0 + 1e-3
    print(f"Real-data smoke stats: {stats}")
