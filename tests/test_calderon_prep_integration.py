"""Integration tests for scripts/prepare_calderon_2019.py.

Synthetic fixture matches REAL GEO GSE118189 format:
- Header has N sample IDs only (no leading peak-column label)
- Data rows have peak_id (col 0) + N counts (cols 1..N)
- Peak coordinates use underscore separators (chr1_start_end)

PR #37 added test_real_format_no_leading_header_label as explicit
regression coverage for the bugs surfaced during Path A diagnostic.

7 synthetic tests run by default. 1 real-data smoke test gated behind
@pytest.mark.real_data -- opt-in via:
    pytest -m real_data tests/test_calderon_prep_integration.py
"""
from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_calderon_2019 import (  # noqa: E402
    build_anndata,
    parse_calderon_counts,
    parse_sample_ids_to_metadata,
)


# --- Synthetic fixture (REAL Calderon format) --------------------------------
# Header: 3 sample IDs only, no leading peak-column label.
# Peaks: underscore separator (chr1_1000_2000), as in real GSE118189.

SYNTH_HEADER = "1001-CD4_Teff-S\t1001-CD4_Teff-U\t1002-Bulk_B-S"
SYNTH_ROWS = [
    "chr1_1000_2000\t5\t0\t12",
    "chr1_5000_6000\t0\t3\t0",
    "chrX_100_500\t7\t8\t0",
]


@pytest.fixture
def synth_counts(tmp_path: Path) -> Path:
    p = tmp_path / "synth_counts.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(SYNTH_HEADER + "\n")
        for row in SYNTH_ROWS:
            fh.write(row + "\n")
    return p


@pytest.fixture
def synth_counts_plain(tmp_path: Path) -> Path:
    p = tmp_path / "synth_counts.txt"
    p.write_text(SYNTH_HEADER + "\n" + "\n".join(SYNTH_ROWS) + "\n")
    return p


# --- Tests --------------------------------------------------------------------

def test_parse_sample_ids_basic():
    df = parse_sample_ids_to_metadata(
        ["1001-CD4_Teff-S", "1002-Bulk_B-U", "garbage_id"]
    )
    assert df.loc["1001-CD4_Teff-S", "donor"] == "1001"
    assert df.loc["1001-CD4_Teff-S", "cell_type"] == "CD4_Teff"
    assert df.loc["1001-CD4_Teff-S", "stim_state"] == "stim"
    assert df.loc["1002-Bulk_B-U", "stim_state"] == "unstim"
    assert df.loc["garbage_id", "donor"] == "unknown"


def test_parse_counts_shape_gz(synth_counts: Path):
    X, var_df, sids = parse_calderon_counts(synth_counts)
    assert X.shape == (3, 3), f"expected (3 samples, 3 peaks), got {X.shape}"
    assert len(sids) == 3
    assert len(var_df) == 3


def test_parse_counts_shape_plain(synth_counts_plain: Path):
    X, var_df, sids = parse_calderon_counts(synth_counts_plain)
    assert X.shape == (3, 3)


def test_parse_counts_values_correct(synth_counts: Path):
    X, var_df, sids = parse_calderon_counts(synth_counts)
    dense = X.toarray()
    # Sample 0 (1001-CD4_Teff-S): peak0=5, peak1=0, peak2=7
    # Sample 1 (1001-CD4_Teff-U): peak0=0, peak1=3, peak2=8
    # Sample 2 (1002-Bulk_B-S):   peak0=12, peak1=0, peak2=0
    assert dense[0, 0] == 5
    assert dense[0, 2] == 7
    assert dense[1, 1] == 3
    assert dense[2, 0] == 12
    assert dense[1, 2] == 8


def test_var_df_peak_parsing_underscore_format(synth_counts: Path):
    """Real Calderon format: peak IDs use underscore separators."""
    X, var_df, sids = parse_calderon_counts(synth_counts)
    assert var_df.loc["chr1_1000_2000", "chrom"] == "chr1"
    assert var_df.loc["chr1_1000_2000", "start"] == 1000
    assert var_df.loc["chr1_1000_2000", "end"] == 2000
    assert var_df.loc["chrX_100_500", "chrom"] == "chrX"


def test_var_df_peak_parsing_legacy_colon_dash_format(tmp_path: Path):
    """Forward-compat: PEAK_RE still accepts the legacy colon-dash format
    used by some older GEO supplementary files."""
    p = tmp_path / "legacy_format.txt"
    p.write_text(
        "S1\tS2\n"
        "chr1:100-200\t1\t2\n"
        "chr2:300-400\t3\t4\n"
    )
    X, var_df, sids = parse_calderon_counts(p)
    assert X.shape == (2, 2)
    assert var_df.loc["chr1:100-200", "chrom"] == "chr1"
    assert var_df.loc["chr1:100-200", "start"] == 100


def test_real_format_no_leading_header_label(tmp_path: Path):
    """REGRESSION TEST: real GSE118189 header has N sample IDs only --
    NOT a leading peak-column label followed by N samples. The original
    PR #32 implementation did sample_ids = header[1:] which silently
    stripped the first real sample. Surfaced during Path A diagnostic
    on the actual GSE118189_ATAC_counts.txt.gz download.

    This test exercises the exact regression: 4 sample IDs in the
    header, 4+1=5 columns per data row. If the parser misreads the
    header as having a leading label, n_samples=3 and the column-count
    assertion fires on the first data row.
    """
    p = tmp_path / "regression.txt"
    p.write_text(
        "1001-Bulk_B-U\t1002-CD4_Teff-S\t1003-Naive_Teffs-U\t1011-Naive_Teffs-S\n"
        "chr1_10045_10517\t1\t2\t3\t4\n"
        "chr2_20000_20500\t5\t0\t6\t7\n"
    )
    X, var_df, sample_ids = parse_calderon_counts(p)
    assert X.shape == (4, 2), f"expected (4 samples, 2 peaks), got {X.shape}"
    assert sample_ids == [
        "1001-Bulk_B-U",
        "1002-CD4_Teff-S",
        "1003-Naive_Teffs-U",
        "1011-Naive_Teffs-S",
    ], "first sample must NOT be stripped"
    assert var_df.iloc[0]["chrom"] == "chr1"
    assert var_df.iloc[0]["start"] == 10045

    # Verify counts unflipped: sample 0 (1001-Bulk_B-U) at peak 0 = 1
    dense = X.toarray()
    assert dense[0, 0] == 1
    assert dense[3, 0] == 4  # 1011-Naive_Teffs-S at peak 0


def test_build_anndata_uns_contract(synth_counts: Path):
    adata = build_anndata(synth_counts)
    assert adata.uns["source"] == "GSE118189"
    assert adata.uns["variant"] == "raw_peaks"
    assert adata.uns["n_samples"] == 3
    assert adata.uns["n_peaks"] == 3
    for col in ("donor", "cell_type", "stim_state"):
        assert col in adata.obs.columns


# --- Real-data smoke (opt-in) -------------------------------------------------

@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_calderon():
    """Smoke against actual GSE118189_ATAC_counts.txt.gz on disk."""
    default_path = Path("data/calderon2019/GSE118189_ATAC_counts.txt.gz")
    counts_path = Path(os.environ.get("CALDERON_COUNTS_PATH", str(default_path)))
    if not counts_path.exists():
        pytest.skip(f"Calderon counts not found at {counts_path}")

    adata = build_anndata(counts_path)
    assert adata.shape[0] > 100, f"too few samples: {adata.shape[0]}"
    assert adata.shape[1] > 100_000, f"too few peaks: {adata.shape[1]}"
    assert adata.X.nnz > 0
    assert (adata.obs["stim_state"].isin(["stim", "unstim", "unknown"])).all()
