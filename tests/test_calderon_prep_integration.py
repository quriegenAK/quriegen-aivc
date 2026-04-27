"""Integration tests for scripts/prepare_calderon_2019.py.

6 synthetic tests run by default. 1 real-data smoke test gated behind
@pytest.mark.real_data — opt-in via:
    pytest -m real_data tests/test_calderon_prep_integration.py
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_calderon_2019 import (  # noqa: E402
    build_anndata,
    parse_calderon_counts,
    parse_sample_ids_to_metadata,
)


# --- Synthetic fixture --------------------------------------------------------

SYNTH_HEADER = "peak\t1001-CD4_Teff-S\t1001-CD4_Teff-U\t1002-Bulk_B-S"
SYNTH_ROWS = [
    "chr1:1000-2000\t5\t0\t12",
    "chr1:5000-6000\t0\t3\t0",
    "chrX:100-500\t7\t8\t0",
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


# --- Synthetic tests ----------------------------------------------------------

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
    # Sample order = column order in header. Peak order = row order in file.
    # Sample 0 (1001-CD4_Teff-S): peak0=5, peak1=0, peak2=7
    # Sample 1 (1001-CD4_Teff-U): peak0=0, peak1=3, peak2=8
    # Sample 2 (1002-Bulk_B-S):   peak0=12, peak1=0, peak2=0
    assert dense[0, 0] == 5
    assert dense[0, 2] == 7
    assert dense[1, 1] == 3
    assert dense[2, 0] == 12
    assert dense[1, 2] == 8


def test_var_df_peak_parsing(synth_counts: Path):
    X, var_df, sids = parse_calderon_counts(synth_counts)
    assert var_df.loc["chr1:1000-2000", "chrom"] == "chr1"
    assert var_df.loc["chr1:1000-2000", "start"] == 1000
    assert var_df.loc["chr1:1000-2000", "end"] == 2000
    assert var_df.loc["chrX:100-500", "chrom"] == "chrX"


def test_build_anndata_uns_contract(synth_counts: Path):
    adata = build_anndata(synth_counts)
    assert adata.uns["source"] == "GSE118189"
    assert adata.uns["variant"] == "raw_peaks"
    assert adata.uns["n_samples"] == 3
    assert adata.uns["n_peaks"] == 3
    for col in ("donor", "cell_type", "stim_state"):
        assert col in adata.obs.columns


# --- Real-data smoke test (opt-in) -------------------------------------------

@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_calderon():
    """Smoke test against the real GSE118189 download.

    Requires: data/calderon2019/GSE118189_ATAC_counts.txt.gz on disk.
    Set CALDERON_COUNTS_PATH env var to override.
    """
    default_path = Path("data/calderon2019/GSE118189_ATAC_counts.txt.gz")
    counts_path = Path(os.environ.get("CALDERON_COUNTS_PATH", str(default_path)))
    if not counts_path.exists():
        pytest.skip(f"Calderon counts not found at {counts_path}")

    adata = build_anndata(counts_path)
    assert adata.shape[0] > 100, f"too few samples: {adata.shape[0]}"
    assert adata.shape[1] > 100_000, f"too few peaks: {adata.shape[1]}"
    assert adata.X.nnz > 0
    assert (adata.obs["stim_state"].isin(["stim", "unstim", "unknown"])).all()
