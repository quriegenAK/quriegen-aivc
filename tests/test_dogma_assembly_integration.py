"""Integration test for scripts/assemble_dogma_h5ad.py (raw-peaks path).

Exercises the full assembly pipeline end-to-end against a synthetic
fixture that mimics the bug classes surfaced during PR #30 debug arc:

  bug #1: PYTHONPATH for `from aivc.data.modality_mask import ...`
  bug #2: per-lane peak hstack with different peak counts per lane
  bug #3: ADT (kite) bare barcodes - no -1 suffix
  bug #4: end-to-end QC chain producing > 0 cells

Plus DD4 obs-column stamping, lysis_protocol/condition, anchor
intersection (LLL), post-subset cell-count sanity bounds.
"""
from __future__ import annotations

import gzip
import sys
import pathlib
import pytest
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import coo_matrix

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))


LANE_DEF = [
    ("LLL_CTRL", "1", 120, 800),
    ("LLL_STIM", "2", 120, 950),
    ("DIG_CTRL", "3", 100, 780),
    ("DIG_STIM", "4", 100, 820),
]
N_GENES = 800   # FIX bug A: must be > 500 to satisfy n_feat_rna > 500 QC
N_ABS = 20


def _build_h5(path, n_cells, n_peaks, seed):
    rng = np.random.RandomState(seed)
    barcodes = [f"BC_{seed}_{i:05d}-1" for i in range(n_cells)]
    gene_names = [f"MT-{i}" for i in range(5)] + [f"GENE_{i}" for i in range(N_GENES - 5)]
    peak_names = [f"chr1:{i*1000}-{i*1000+500}" for i in range(n_peaks)]
    n_feat = N_GENES + n_peaks
    feature_names = gene_names + peak_names
    feature_types = ["Gene Expression"] * N_GENES + ["Peaks"] * n_peaks

    n_pass = int(n_cells * 0.8)
    cols, rows, vals = [], [], []
    for c in range(n_cells):
        # FIX bug A: pass-cells get 700 distinct genes (> 500 threshold)
        n_genes_per_cell = 700 if c < n_pass else 50
        gidx = rng.choice(range(5, N_GENES), n_genes_per_cell, replace=False)
        cols.extend([c] * len(gidx))
        rows.extend(gidx.tolist())
        vals.extend(rng.poisson(8, len(gidx)).tolist())
        n_peaks_per_cell = 150 if c < n_pass else 30
        pidx = rng.choice(range(n_peaks), n_peaks_per_cell, replace=False) + N_GENES
        cols.extend([c] * len(pidx))
        rows.extend(pidx.tolist())
        vals.extend(rng.poisson(2, len(pidx)).tolist())

    mat = coo_matrix(
        (np.array(vals, dtype=np.int32),
         (np.array(rows), np.array(cols))),
        shape=(n_feat, n_cells),
    ).tocsc()

    with h5py.File(path, "w") as f:
        mg = f.create_group("matrix")
        mg.create_dataset("data", data=mat.data)
        mg.create_dataset("indices", data=mat.indices)
        mg.create_dataset("indptr", data=mat.indptr)
        mg.create_dataset("shape", data=np.array([n_feat, n_cells], dtype=np.int64))
        mg.create_dataset("barcodes", data=[b.encode() for b in barcodes])
        fg = mg.create_group("features")
        fg.create_dataset("name", data=[n.encode() for n in feature_names])
        fg.create_dataset("feature_type", data=[t.encode() for t in feature_types])
    return barcodes


def _build_metrics(path, barcodes, seed):
    rng = np.random.RandomState(seed + 1000)
    n = len(barcodes)
    n_pass = int(n * 0.8)
    is_cell = np.array([1] * n_pass + [0] * (n - n_pass))
    atac_total = rng.randint(2000, 8000, n)
    pct = np.where(is_cell == 1, rng.uniform(0.55, 0.85, n), rng.uniform(0.10, 0.30, n))
    atac_peaks = (atac_total * pct).astype(int)
    df = pd.DataFrame({
        "barcode": barcodes,
        "is_cell": is_cell,
        "atac_peak_region_fragments": atac_peaks,
        "atac_fragments": atac_total,
    })
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)


def _build_adt(adt_dir, lane, barcodes, seed):
    rng = np.random.RandomState(seed + 2000)
    n = len(barcodes)
    n_pass = int(n * 0.8)
    bare = [b[:-2] if b.endswith("-1") else b for b in barcodes]
    ab_names = ["Ctrl-IgG"] + ["CD4-RPA-T4"] + ["CD8-OKT8"] + [f"AB_{i}" for i in range(N_ABS - 3)]

    rows, cols, vals = [], [], []
    for c in range(n):
        if c < n_pass:
            for a in range(N_ABS):
                v = rng.poisson(20 if a > 0 else 2)
                if v > 0:
                    rows.append(a); cols.append(c); vals.append(v)
        else:
            for a in range(N_ABS):
                v = rng.poisson(2)
                if v > 0:
                    rows.append(a); cols.append(c); vals.append(v)

    # FIX bug B: 3-number dims header [rows, cols, nnz] (matches script parser)
    mtx_path = adt_dir / f"{lane}__featurecounts.mtx.gz"
    with gzip.open(mtx_path, "wt") as f:
        f.write(f"{N_ABS} {n} {len(rows)}\n")
        for i, j, v in zip(rows, cols, vals):
            f.write(f"{i+1} {j+1} {v}\n")

    (adt_dir / f"{lane}__featurecounts.barcodes.txt").write_text("\n".join(bare))
    (adt_dir / f"{lane}__featurecounts.genes.txt").write_text("\n".join(ab_names))


def _build_synthetic_dogma_fixture(tmp_path):
    raw_path = tmp_path / "raw"
    (raw_path / "h5").mkdir(parents=True)
    (raw_path / "metrics").mkdir()
    (raw_path / "adt").mkdir()

    all_lll_anchor = []
    for lane, idx, n_cells, n_peaks in LANE_DEF:
        seed = int(idx) * 7
        bcs = _build_h5(raw_path / "h5" / f"{lane}__feature_bc_matrix.h5",
                        n_cells, n_peaks, seed)
        _build_metrics(raw_path / "metrics" / f"{lane}__per_barcode_metrics.csv.gz",
                       bcs, seed)
        _build_adt(raw_path / "adt", lane, bcs, seed)

        if lane.startswith("LLL"):
            n_pass = int(n_cells * 0.8)
            for b in bcs[:n_pass]:
                all_lll_anchor.append(b.replace("-1", f"-{idx}"))

    anchor_path = tmp_path / "pairs_of_wnn.tsv"
    pd.DataFrame({"barcode": all_lll_anchor}).to_csv(anchor_path, sep="\t", index=False)
    return raw_path, anchor_path


@pytest.fixture
def dogma_fixture(tmp_path):
    return _build_synthetic_dogma_fixture(tmp_path)


def test_assemble_lll_arm_produces_nonzero_cells(dogma_fixture):
    """Bug class #4: end-to-end QC chain must produce > 0 cells."""
    import assemble_dogma_h5ad as assembly
    raw_path, anchor_path = dogma_fixture
    df = pd.read_csv(anchor_path, sep="\t")
    anchor = set(df["barcode"])

    adata = assembly.assemble_arm(
        arm="LLL", raw_path=raw_path,
        anchor_barcodes=anchor, use_chromvar=False,
    )
    assert adata.n_obs > 0
    assert adata.n_obs <= len(anchor)


def test_per_lane_peak_union_handles_different_sizes(dogma_fixture):
    """Bug class #2: per-lane peaks of different sizes (800 vs 950)."""
    import assemble_dogma_h5ad as assembly
    raw_path, anchor_path = dogma_fixture
    df = pd.read_csv(anchor_path, sep="\t")
    adata = assembly.assemble_arm("LLL", raw_path, set(df["barcode"]), use_chromvar=False)
    assert adata.obsm["atac_peaks"].shape[1] >= 950


def test_adt_bare_barcode_join_succeeds(dogma_fixture):
    """Bug class #3: kite bare barcodes need suffix concat, not regex sub."""
    import assemble_dogma_h5ad as assembly
    raw_path, _ = dogma_fixture
    adata = assembly.assemble_arm("DIG", raw_path, None, use_chromvar=False)
    assert adata.n_obs > 0
    assert "protein" in adata.obsm
    assert adata.obsm["protein"].shape == (adata.n_obs, N_ABS)


def test_dd4_obs_columns_stamped(dogma_fixture):
    import assemble_dogma_h5ad as assembly
    raw_path, _ = dogma_fixture
    adata = assembly.assemble_arm("LLL", raw_path, None, use_chromvar=False)
    for col in ("has_rna", "has_atac", "has_protein", "has_phospho"):
        assert col in adata.obs.columns
    assert adata.obs["has_rna"].all()
    assert adata.obs["has_atac"].all()
    assert adata.obs["has_protein"].all()
    assert not adata.obs["has_phospho"].any()


def test_lysis_protocol_and_condition_obs(dogma_fixture):
    import assemble_dogma_h5ad as assembly
    raw_path, _ = dogma_fixture
    lll = assembly.assemble_arm("LLL", raw_path, None, use_chromvar=False)
    dig = assembly.assemble_arm("DIG", raw_path, None, use_chromvar=False)
    assert (lll.obs["lysis_protocol"] == "LLL").all()
    assert (dig.obs["lysis_protocol"] == "DIG").all()
    for arm in (lll, dig):
        assert set(arm.obs["condition"].unique()) <= {"CTRL", "STIM"}


def test_anchor_intersection_filters_lll_to_anchor_set(dogma_fixture):
    import assemble_dogma_h5ad as assembly
    raw_path, anchor_path = dogma_fixture
    anchor = set(pd.read_csv(anchor_path, sep="\t")["barcode"])
    adata = assembly.assemble_arm("LLL", raw_path, anchor, use_chromvar=False)
    assert set(adata.obs_names).issubset(anchor)


def test_python_path_aware_invocation():
    """Bug class #1: import succeeds when scripts/ is on sys.path."""
    import assemble_dogma_h5ad
    assert hasattr(assemble_dogma_h5ad, "assemble_arm")
    assert hasattr(assemble_dogma_h5ad, "validate_with_pairing_cert")


def test_real_data_smoke_test_lll():
    """Optional: skip if cached real data not present."""
    import os
    raw = pathlib.Path(os.path.expanduser("~/aivc_dogma_ncells_py"))
    if not raw.exists():
        pytest.skip("real DOGMA raw data not present")

    import assemble_dogma_h5ad as assembly
    anchor_file = (
        pathlib.Path(__file__).resolve().parents[1]
        / "data/phase6_5g_2/external_evidence/dogma_ncells_measured_2026-04-23/pairs_of_wnn.tsv"
    )
    if not anchor_file.exists():
        pytest.skip("anchor file not on this checkout")

    anchor = set(pd.read_csv(anchor_file, sep="\t")["barcode"])
    adata = assembly.assemble_arm("LLL", raw, anchor, use_chromvar=False)
    assert 13_500 <= adata.n_obs <= 14_000
