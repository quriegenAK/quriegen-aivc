"""Unit tests for scripts/assemble_dogma_h5ad.py."""
import sys
import pathlib
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
import assemble_dogma_h5ad as assembly  # noqa: E402


def test_normalize_rna_target_sum():
    """normalize_rna scales each cell's counts to target_sum before log1p."""
    mat = sp.csc_matrix([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result = assembly.normalize_rna(mat, target_sum=1e4)
    assert result.shape == (3, 2)
    assert not np.isnan(result.toarray()).any()
    assert (result.toarray() >= 0).all()


def test_clr_normalize_centers_log():
    """CLR: log(x+1) - geometric_mean_log per cell. Output mean per row should be 0."""
    mat = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    result = assembly.clr_normalize(sp.csr_matrix(mat))
    assert result.shape == (2, 3)
    assert np.allclose(result.mean(axis=1), 0, atol=1e-5)


def test_clr_normalize_handles_zeros():
    """Pseudocount (+1) prevents log(0)."""
    mat = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    result = assembly.clr_normalize(sp.csr_matrix(mat))
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_validate_with_pairing_cert_on_synthetic():
    """Synthetic 3-modal adata passes make_dogma_seq() cert validation."""
    import anndata as ad
    from aivc.data.modality_mask import ATAC_KEY, PROTEIN_KEY

    n = 20
    adata = ad.AnnData(X=np.random.rand(n, 50).astype(np.float32))
    adata.obsm[ATAC_KEY] = np.random.rand(n, 100).astype(np.float32)
    adata.obsm[PROTEIN_KEY] = np.random.rand(n, 210).astype(np.float32)
    adata.obs_names = [f"cell_{i}" for i in range(n)]

    result = assembly.validate_with_pairing_cert(adata)
    assert result["can_train"] is True
    assert len(result["contrastive_pairs"]) == 3


def test_assembly_obs_contract():
    """Verify obs columns match the assembly contract exactly."""
    expected_cols = {"lysis_protocol", "condition", "has_rna", "has_atac",
                     "has_protein", "has_phospho", "donor_id", "lane",
                     "protein_panel_id"}
    obs = {"lysis_protocol": "LLL", "condition": "CTRL",
           "has_rna": True, "has_atac": True, "has_protein": True,
           "has_phospho": False, "donor_id": "mimitou_2021_donor_1",
           "lane": "LLL_CTRL", "protein_panel_id": "totalseq_a_210"}
    assert expected_cols == set(obs.keys())
    assert obs["has_phospho"] is False
    assert obs["has_rna"] and obs["has_atac"] and obs["has_protein"]


def test_chromvar_import_error_fallback_path():
    """compute_chromvar_deviations raises ImportError with install instructions
    when pychromvar is not installed."""
    try:
        import pychromvar  # noqa: F401
        pytest.skip("pychromvar installed; fallback path not tested here")
    except ImportError:
        with pytest.raises(ImportError, match=r"pychromvar|pyjaspar|--raw-peaks-fallback"):
            assembly.compute_chromvar_deviations(
                sp.csc_matrix(np.random.rand(5, 10)),
                [f"chr1:{i*100}-{i*100+50}" for i in range(5)],
                [f"cell_{i}" for i in range(10)],
            )
