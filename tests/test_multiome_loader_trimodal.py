"""Tests for MultiomeLoader tri-modal (DOGMA) extension."""
import os
import tempfile
import shutil
import numpy as np
import pytest

from aivc.data.modality_mask import (
    ModalityKey,
    RNA_KEY,
    ATAC_KEY,
    PROTEIN_KEY,
    MASK_KEY,
    LYSIS_KEY,
    PROTEIN_PANEL_KEY,
)


@pytest.fixture
def synthetic_dogma_h5ad():
    """Synthetic DOGMA h5ad: RNA in .X, ATAC in obsm['atac_peaks'], Protein in obsm['protein']."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    n_cells, n_genes, n_peaks, n_ab = 100, 500, 1000, 210
    X_rna = csr_matrix(np.random.poisson(1.0, (n_cells, n_genes)).astype(np.float32))
    adata = ad.AnnData(X=X_rna)
    adata.obsm["atac_peaks"] = np.random.poisson(1.0, (n_cells, n_peaks)).astype(np.float32)
    adata.obsm["protein"] = np.random.poisson(10.0, (n_cells, n_ab)).astype(np.float32)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    tmp = tempfile.mkdtemp()
    h5ad_path = os.path.join(tmp, "dogma_test.h5ad")
    peak_set_path = os.path.join(tmp, "peaks.tsv")
    adata.write_h5ad(h5ad_path)
    with open(peak_set_path, "w") as f:
        f.write("chr\tstart\tend\n")
        for i in range(n_peaks):
            f.write(f"chr1\t{i*1000}\t{(i+1)*1000}\n")

    yield h5ad_path, peak_set_path, tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_trimodal_loader_basic(synthetic_dogma_h5ad):
    """Loader loads RNA + ATAC + Protein from a tri-modal h5ad."""
    from aivc.data.multiome_loader import MultiomeLoader
    h5ad_path, peak_set_path, _ = synthetic_dogma_h5ad

    loader = MultiomeLoader(
        h5ad_path=h5ad_path,
        schema="obsm_atac",
        peak_set_path=peak_set_path,
        atac_key="atac_peaks",
        protein_obsm_key="protein",
        lysis_protocol="LLL",
        protein_panel_id="totalseq_a_210",
    )

    assert len(loader) == 100
    item = loader[0]
    assert RNA_KEY in item
    assert ATAC_KEY in item
    assert PROTEIN_KEY in item
    assert MASK_KEY in item
    assert LYSIS_KEY in item
    assert PROTEIN_PANEL_KEY in item
    assert item[LYSIS_KEY] == "LLL"
    assert item[PROTEIN_PANEL_KEY] == "totalseq_a_210"
    assert item[RNA_KEY].shape == (500,)
    assert item[ATAC_KEY].shape == (1000,)
    assert item[PROTEIN_KEY].shape == (210,)


def test_modality_mask_phospho_absent_dogma(synthetic_dogma_h5ad):
    """DOGMA loader emits (4,) modality_mask with Phospho (idx 1) = 0."""
    from aivc.data.multiome_loader import MultiomeLoader
    h5ad_path, peak_set_path, _ = synthetic_dogma_h5ad

    loader = MultiomeLoader(
        h5ad_path=h5ad_path,
        schema="obsm_atac",
        peak_set_path=peak_set_path,
        atac_key="atac_peaks",
        protein_obsm_key="protein",
        lysis_protocol="DIG",
    )
    mask = loader[0][MASK_KEY]
    assert mask.shape == (4,)
    assert mask[int(ModalityKey.ATAC)] == 1.0
    assert mask[int(ModalityKey.PHOSPHO)] == 0.0  # Phospho always absent for DOGMA
    assert mask[int(ModalityKey.RNA)] == 1.0
    assert mask[int(ModalityKey.PROTEIN)] == 1.0


def test_bimodal_loader_no_protein_backward_compat(synthetic_dogma_h5ad):
    """Loader without protein_obsm_key = 2-modal; Protein and Phospho absent in mask."""
    from aivc.data.multiome_loader import MultiomeLoader
    h5ad_path, peak_set_path, _ = synthetic_dogma_h5ad

    loader = MultiomeLoader(
        h5ad_path=h5ad_path,
        schema="obsm_atac",
        peak_set_path=peak_set_path,
        atac_key="atac_peaks",
    )
    item = loader[0]
    assert RNA_KEY in item
    assert ATAC_KEY in item
    assert PROTEIN_KEY not in item  # not emitted for bimodal
    mask = item[MASK_KEY]
    assert mask[int(ModalityKey.ATAC)] == 1.0
    assert mask[int(ModalityKey.PHOSPHO)] == 0.0
    assert mask[int(ModalityKey.RNA)] == 1.0
    assert mask[int(ModalityKey.PROTEIN)] == 0.0


def test_mutually_exclusive_protein_params():
    """Setting both protein_obsm_key and protein_path raises."""
    from aivc.data.multiome_loader import MultiomeLoader
    with pytest.raises(ValueError, match="Only one of protein_obsm_key / protein_path"):
        MultiomeLoader(
            h5ad_path="/nonexistent.h5ad",
            peak_set_path="/nonexistent.tsv",
            protein_obsm_key="protein",
            protein_path="/other.h5ad",
            lazy=True,
        )


def test_factory_make_dogma_lll_lazy():
    """make_dogma_lll produces a loader with LLL + 210-ab panel config."""
    from aivc.data.multiome_loader import MultiomeLoader
    loader = MultiomeLoader.make_dogma_lll(
        base_path="/nonexistent",
        peak_set_path="/nonexistent/peaks.tsv",
        lazy=True,  # skip load; just verify constructor args
    )
    assert loader.lysis_protocol == "LLL"
    assert loader.protein_panel_id == "totalseq_a_210"
    assert loader.protein_obsm_key == "protein"
    assert loader.atac_key == "atac_peaks"
    assert loader.h5ad_path.endswith("dogma_lll.h5ad")


def test_factory_make_dogma_dig_lazy():
    """make_dogma_dig produces a loader with DIG + 210-ab panel config."""
    from aivc.data.multiome_loader import MultiomeLoader
    loader = MultiomeLoader.make_dogma_dig(
        base_path="/nonexistent",
        peak_set_path="/nonexistent/peaks.tsv",
        lazy=True,
    )
    assert loader.lysis_protocol == "DIG"
    assert loader.protein_panel_id == "totalseq_a_210"
    assert loader.h5ad_path.endswith("dogma_dig.h5ad")


def test_factory_eager_load_fails_on_missing_file():
    """Non-lazy factory raises FileNotFoundError for missing h5ad."""
    from aivc.data.multiome_loader import MultiomeLoader
    with pytest.raises((FileNotFoundError, OSError)):
        MultiomeLoader.make_dogma_lll(
            base_path="/nonexistent",
            peak_set_path="/nonexistent/peaks.tsv",
            lazy=False,
        )
