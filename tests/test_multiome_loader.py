"""Phase 4 unit tests for MultiomeLoader."""
import os
import sys
import tempfile

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aivc.data.dataset_kind import DatasetKind
from aivc.data.multiome_loader import MultiomeLoader


def _make_obsm_h5ad(tmpdir, n_cells=20, n_genes=50, n_peaks=200):
    import anndata as ad
    rng = np.random.RandomState(0)
    X = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
    atac = rng.poisson(0.2, size=(n_cells, n_peaks)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obsm["ATAC"] = atac
    path = os.path.join(tmpdir, "multiome.h5ad")
    adata.write_h5ad(path)
    return path, X, atac


def _fake_peak_set(tmpdir):
    path = os.path.join(tmpdir, "peak_set.bed")
    with open(path, "w") as fh:
        fh.write("chr1\t0\t1000\tp1\n")
    return path


def test_peak_set_required():
    with pytest.raises(ValueError, match="scripts/harmonize_peaks.py"):
        MultiomeLoader(h5ad_path="unused.h5ad", peak_set_path=None)


def test_obsm_schema_yields_correct_keys_and_kind():
    with tempfile.TemporaryDirectory() as td:
        h5ad, _, _ = _make_obsm_h5ad(td)
        peak_set = _fake_peak_set(td)
        loader = MultiomeLoader(
            h5ad_path=h5ad, schema="obsm_atac",
            peak_set_path=peak_set, atac_key="ATAC",
        )
        batch = loader[0]
        assert set(batch.keys()) == {"rna", "atac_peaks", "dataset_kind"}
        assert batch["dataset_kind"] == DatasetKind.OBSERVATIONAL.value
        assert batch["rna"].shape == (50,)
        assert batch["atac_peaks"].shape == (200,)


def test_both_schemas_equivalent_outputs(monkeypatch):
    """Both schemas, fed equivalent data, must produce identical output
    dict shapes.

    The real `mudata` package is intentionally NOT added as a test
    dependency in Phase 4 (keeping dep surface minimal and avoiding
    version churn with anndata). We inject a lightweight stub module
    exposing the attribute surface MultiomeLoader consumes:
    `mudata.read(path)` returning an object with a `.mod` dict of
    AnnData-like entries.
    """
    import sys
    import types
    import anndata as ad

    with tempfile.TemporaryDirectory() as td:
        peak_set = _fake_peak_set(td)
        n_cells, n_genes, n_peaks = 10, 25, 80
        rng = np.random.RandomState(0)
        rna_X = rng.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
        atac_X = rng.poisson(0.2, size=(n_cells, n_peaks)).astype(np.float32)

        # obsm_atac version
        obsm_path = os.path.join(td, "obsm.h5ad")
        a = ad.AnnData(X=rna_X.copy())
        a.obsm["ATAC"] = atac_X.copy()
        a.write_h5ad(obsm_path)

        # mudata stub — minimal surface: read(path) -> obj with .mod dict
        class _Stub:
            def __init__(self, mod):
                self.mod = mod

        stub_payload = _Stub({
            "rna": ad.AnnData(X=rna_X.copy()),
            "atac": ad.AnnData(X=atac_X.copy()),
        })
        fake_md = types.ModuleType("mudata")
        fake_md.read = lambda path: stub_payload  # noqa: E731
        fake_md.read_h5mu = lambda path: stub_payload  # noqa: E731
        monkeypatch.setitem(sys.modules, "mudata", fake_md)

        # Path is irrelevant for the stub; any non-.h5mu triggers read().
        mu_path = os.path.join(td, "stub.h5ad")

        L1 = MultiomeLoader(h5ad_path=obsm_path, schema="obsm_atac",
                            peak_set_path=peak_set, atac_key="ATAC")
        L2 = MultiomeLoader(h5ad_path=mu_path, schema="mudata",
                            peak_set_path=peak_set,
                            rna_key="rna", atac_key="atac")

        b1, b2 = L1[0], L2[0]
        assert set(b1.keys()) == set(b2.keys())
        assert b1["rna"].shape == b2["rna"].shape
        assert b1["atac_peaks"].shape == b2["atac_peaks"].shape
        assert b1["dataset_kind"] == b2["dataset_kind"] == DatasetKind.OBSERVATIONAL.value
