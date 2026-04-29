"""Tests for scripts/build_dogma_peak_union.py.

8 synthetic tests + 1 real_data smoke against actual LLL/DIG h5ads.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_dogma_peak_union import (  # noqa: E402
    build_union_peaks,
    parse_peaks,
    project_to_union,
)


def test_parse_peaks_colon_dash_format():
    peaks = parse_peaks(["chr1:100-200", "chr20:51489470-51489886"])
    assert peaks == [("chr1", 100, 200), ("chr20", 51489470, 51489886)]


def test_parse_peaks_underscore_format():
    peaks = parse_peaks(["chr1_100_200", "chr2_500_600"])
    assert peaks[0] == ("chr1", 100, 200)
    assert peaks[1] == ("chr2", 500, 600)


def test_parse_peaks_unplaced_contig():
    """PEAK_RE must accept GL/KI-style contig names (no chr prefix required)."""
    peaks = parse_peaks(["GL000194.1:100320-100413"])
    assert peaks[0] == ("GL000194.1", 100320, 100413)


def test_build_union_disjoint():
    lll = [("chr1", 100, 200), ("chr2", 300, 400)]
    dig = [("chr3", 500, 600), ("chr4", 700, 800)]
    union, [lll_map, dig_map] = build_union_peaks([lll, dig])
    assert len(union) == 4
    assert {p[0] for p in union} == {"chr1", "chr2", "chr3", "chr4"}
    assert union[lll_map[0]] == lll[0]
    assert union[dig_map[0]] == dig[0]


def test_build_union_with_overlap_dedupes():
    lll = [("chr1", 100, 200), ("chr2", 300, 400)]
    dig = [("chr1", 100, 200), ("chr3", 500, 600)]
    union, [lll_map, dig_map] = build_union_peaks([lll, dig])
    assert len(union) == 3
    # Both arms point to same union index for the shared peak
    assert lll_map[0] == dig_map[0]
    assert union[lll_map[0]] == ("chr1", 100, 200)


def test_build_union_sorted():
    lll = [("chr2", 500, 600), ("chr1", 100, 200)]
    dig = []
    union, _ = build_union_peaks([lll, dig])
    assert union == [("chr1", 100, 200), ("chr2", 500, 600)]


def test_project_to_union_zero_fill():
    X_arm = np.array([[1.0, 2.0], [3.0, 4.0]])
    idx_map = np.array([0, 2])
    X_union = project_to_union(X_arm, idx_map, n_union=4)
    assert X_union.shape == (2, 4)
    np.testing.assert_array_equal(X_union[0], [1.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(X_union[1], [3.0, 0.0, 4.0, 0.0])


def test_project_to_union_mass_conserved():
    """Per-cell total counts preserved under projection."""
    X_arm = np.array([[1.0, 2.0, 3.0]])
    idx_map = np.array([0, 5, 9])
    X_union = project_to_union(X_arm, idx_map, n_union=12)
    assert X_union.sum() == X_arm.sum()


def test_project_to_union_sparse_input_supported():
    X_arm = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    idx_map = np.array([1, 3, 5])
    X_union = project_to_union(X_arm, idx_map, n_union=7)
    np.testing.assert_array_equal(X_union[0], [0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(X_union[1], [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])


@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_dogma_union():
    """Build union from actual LLL+DIG h5ads."""
    lll_path = Path(os.environ.get(
        "DOGMA_LLL_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_lll.h5ad"))
    dig_path = Path(os.environ.get(
        "DOGMA_DIG_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_dig.h5ad"))
    if not lll_path.exists() or not dig_path.exists():
        pytest.skip("DOGMA h5ads missing")

    lll = ad.read_h5ad(lll_path, backed="r")
    dig = ad.read_h5ad(dig_path, backed="r")

    lll_peaks = parse_peaks(list(lll.uns["atac_feature_names"]))
    dig_peaks = parse_peaks(list(dig.uns["atac_feature_names"]))

    union, [lll_map, dig_map] = build_union_peaks([lll_peaks, dig_peaks])

    print(f"\nLLL: {len(lll_peaks)}, DIG: {len(dig_peaks)}, "
          f"Union: {len(union)}, Shared: {len(set(lll_peaks) & set(dig_peaks))}")

    # Path A diagnostic measured: LLL=198947, DIG=124619, intersection=66
    # Union expected = 198947 + 124619 - 66 = 323500
    assert 320000 < len(union) < 326000
    assert len(lll_map) == len(lll_peaks)
    assert len(dig_map) == len(dig_peaks)

    # Verify mappings are consistent — first 10 entries
    for i in range(min(10, len(lll_peaks))):
        assert union[lll_map[i]] == lll_peaks[i]
    for i in range(min(10, len(dig_peaks))):
        assert union[dig_map[i]] == dig_peaks[i]
