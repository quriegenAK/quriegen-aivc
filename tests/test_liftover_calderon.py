"""Tests for scripts/liftover_calderon_to_hg38.py.

Synthetic tests use a mock LiftOver object (no UCSC chain download).
1 real_data smoke runs the full pipeline against actual Calderon h5ad
+ harmonization against DOGMA, asserts post-liftOver frac_overlap > 0.50.
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

from liftover_calderon_to_hg38 import (  # noqa: E402
    lift_anndata,
    lift_var_coordinates,
)


class MockLiftOver:
    """Mock pyliftover.LiftOver that returns deterministic mappings."""

    def __init__(self, mapping=None, drop_chroms=None, chrom_changes=None):
        self.mapping = mapping or {}
        self.drop_chroms = set(drop_chroms or [])
        self.chrom_changes = set(chrom_changes or [])

    def convert_coordinate(self, chrom, pos):
        if chrom in self.drop_chroms:
            return []
        if chrom in self.chrom_changes:
            new_chrom = "chr_alt" if pos % 2 == 0 else chrom
            return [(new_chrom, pos + 100, "+", 0)]
        if (chrom, pos) in self.mapping:
            new = self.mapping[(chrom, pos)]
            return [(new[0], new[1], "+", 0)]
        return [(chrom, pos + 1000, "+", 0)]


def _var(rows):
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    df.index = [f"{c}:{s}-{e}" for c, s, e in rows]
    return df


def test_lift_simple_shift():
    """Default shift: positions move by +1000."""
    var = _var([("chr1", 100, 200), ("chr1", 500, 600)])
    new_var, keep, stats = lift_var_coordinates(var, MockLiftOver())
    assert stats["n_input"] == 2
    assert stats["n_lifted"] == 2
    assert stats["frac_lifted"] == 1.0
    assert list(new_var["start"]) == [1100, 1500]
    assert list(new_var["end"]) == [1200, 1600]


def test_lift_drops_unmappable():
    """Peaks on chroms that LiftOver returns [] for are dropped."""
    var = _var([("chr1", 100, 200), ("chrUn_fake", 300, 400), ("chr2", 500, 600)])
    lo = MockLiftOver(drop_chroms={"chrUn_fake"})
    new_var, keep, stats = lift_var_coordinates(var, lo)
    assert stats["n_input"] == 3
    assert stats["n_lifted"] == 2
    assert stats["n_dropped_no_lift"] == 1
    assert "chrUn_fake" not in new_var["chrom"].values


def test_lift_drops_chrom_change():
    """Peaks where start and end lift to different chroms are dropped.

    MockLiftOver.chrom_changes returns chr_alt for even positions, original
    chrom for odd. A peak with start=101 (odd) end=200 (even) lifts to
    different chromosomes → must be dropped.
    """
    var = _var([("chr1", 100, 200), ("chr_swap", 101, 200), ("chr3", 500, 600)])
    lo = MockLiftOver(chrom_changes={"chr_swap"})
    new_var, keep, stats = lift_var_coordinates(var, lo)
    assert stats["n_dropped_chrom_change"] == 1
    assert "chr_swap" not in new_var["chrom"].values


def test_lift_anndata_subsets_X():
    """lift_anndata correctly subsets .X along var axis."""
    var = _var([("chr1", 100, 200), ("chrUn_fake", 300, 400), ("chr2", 500, 600)])
    n_samples = 4
    X = sp.csr_matrix(np.arange(n_samples * 3).reshape(n_samples, 3).astype(np.float32))
    a = ad.AnnData(X=X, var=var)

    lo = MockLiftOver(drop_chroms={"chrUn_fake"})
    new_a, stats = lift_anndata(a, lo)
    assert new_a.shape == (4, 2)
    assert stats["n_lifted"] == 2
    # Sample 0: original [0, 1, 2]; after dropping middle peak -> [0, 2]
    np.testing.assert_array_equal(new_a.X.toarray()[0], [0, 2])
    assert new_a.uns["build"] == "hg38"
    assert "liftover_hg19_to_hg38_stats" in new_a.uns


def test_lift_handles_empty_input():
    """Empty AnnData produces empty AnnData with stats."""
    var = pd.DataFrame(columns=["chrom", "start", "end"])
    new_var, keep, stats = lift_var_coordinates(var, MockLiftOver())
    assert stats["n_input"] == 0
    assert stats["n_lifted"] == 0
    assert stats["frac_lifted"] == 0.0


# --- Real-data smoke: full pipeline post-liftOver -----------------------------

@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_smoke_post_liftover_harmonization():
    """End-to-end: lift Calderon hg19 -> hg38 -> harmonize against DOGMA.

    Asserts post-liftOver frac_overlap > 0.50 (Path A diagnostic measured
    9.81% pre-liftOver; expected 60-90% post-liftOver). This is the
    load-bearing signal that liftOver is actually doing its job.
    """
    calderon_path = Path(os.environ.get(
        "CALDERON_H5AD",
        "data/calderon2019/calderon_atac.h5ad"))
    dogma_path = Path(os.environ.get(
        "DOGMA_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_lll.h5ad"))
    if not calderon_path.exists() or not dogma_path.exists():
        pytest.skip("Calderon or DOGMA h5ad missing")

    calderon = ad.read_h5ad(calderon_path)
    print(f"Calderon hg19 shape: {calderon.shape}")
    new_calderon, lift_stats = lift_anndata(calderon)
    print(f"Lift stats: {lift_stats}")
    assert lift_stats["frac_lifted"] > 0.90, (
        f"Suspiciously low frac_lifted: {lift_stats['frac_lifted']:.3%}"
    )

    sys.path.insert(0, str(SCRIPTS))
    from harmonize_calderon_peaks import (
        compute_projection_matrix,
        extract_peaks_to_var_format,
    )
    dogma = ad.read_h5ad(dogma_path, backed="r")
    dogma_var = extract_peaks_to_var_format(dogma, name="dogma")
    calderon_var = extract_peaks_to_var_format(new_calderon, name="calderon_hg38")

    M, harm_stats = compute_projection_matrix(dogma_var, calderon_var)
    print(f"Post-liftOver harmonization stats: {harm_stats}")

    # Load-bearing assertion: DOGMA-side coverage. The eval scaffold
    # trains a probe on DOGMA encoder features and uses Calderon as
    # held-out test data projected into DOGMA peak space. What matters
    # is what fraction of DOGMA peaks receive any Calderon signal --
    # NOT what fraction of Calderon peaks find a DOGMA home (Calderon
    # has ~4x more peaks across 25+ cell types and 7 donors; many are
    # in regions DOGMA's narrower scRNA+ATAC profile never called).
    #
    # Pre-liftOver Path A diagnostic measured Calderon->DOGMA = 9.81%.
    # The asymmetric DOGMA->Calderon direction is the load-bearing one.
    dogma_has_calderon = np.asarray((M > 0).sum(axis=0)).ravel() > 0
    frac_dogma_covered = float(dogma_has_calderon.mean())
    print(f"frac_dogma_with_any_calderon: {frac_dogma_covered:.4%}")
    print(f"frac_calderon_with_any_dogma: {harm_stats['frac_calderon_with_any_overlap']:.4%} (diagnostic only)")

    assert frac_dogma_covered > 0.50, (
        f"Post-liftOver DOGMA coverage {frac_dogma_covered:.4%} below 50%. "
        "Expected majority DOGMA-peak coverage by Calderon (Calderon has "
        "~4x more peaks than DOGMA across cell types). If this fires, "
        "investigate peak-calling parameter divergence, not liftOver."
    )
