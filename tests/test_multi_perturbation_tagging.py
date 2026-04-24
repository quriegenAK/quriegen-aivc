"""Tests for DD4 modality-presence tagging in MultiPerturbationLoader."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import anndata as ad


SEED = 42


def _mock_adata(n_cells=20, n_genes=30, donor_prefix="kang", condition="ctrl"):
    rng = np.random.RandomState(SEED)
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        "donor_id": [f"{donor_prefix}_donor{i % 3}" for i in range(n_cells)],
        "cell_type": "CD4 T cells",
        "condition": condition,
        "label": condition,
    }, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_load_kang_stamps_rna_only():
    from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
    adata = _mock_adata()
    path = os.path.join(tempfile.mkdtemp(), "kang.h5ad")
    adata.write_h5ad(path)

    loader = MultiPerturbationLoader(kang_path=path, kang_test_donors=[])
    kang = loader.load_kang()

    for col in ["has_rna", "has_atac", "has_protein", "has_phospho"]:
        assert col in kang.obs.columns, f"{col} missing after load_kang"
    assert kang.obs["has_rna"].all()
    assert not kang.obs["has_atac"].any()
    assert not kang.obs["has_protein"].any()
    assert not kang.obs["has_phospho"].any()


def test_stamp_helper_direct():
    from aivc.data.multi_perturbation_loader import _stamp_modality_tags
    adata = _mock_adata()
    _stamp_modality_tags(adata, has_rna=True, has_atac=True, has_protein=True)
    assert adata.obs["has_rna"].all()
    assert adata.obs["has_atac"].all()
    assert adata.obs["has_protein"].all()
    assert not adata.obs["has_phospho"].any()


def test_build_combined_corpus_requires_has_columns():
    """build_combined_corpus auto-defaults missing has_* columns to False."""
    from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
    from aivc.data.dataset_kind import DatasetKind

    adata = _mock_adata()
    path = os.path.join(tempfile.mkdtemp(), "kang.h5ad")
    adata.write_h5ad(path)

    loader = MultiPerturbationLoader(kang_path=path, kang_test_donors=[])
    loader.load_kang()
    combined = loader.build_combined_corpus()

    assert combined is not None
    for col in ["has_rna", "has_atac", "has_protein", "has_phospho"]:
        assert col in combined.obs.columns, f"{col} absent from combined corpus"
    assert combined.obs["has_rna"].all()
    assert not combined.obs["has_atac"].any()


def test_build_combined_corpus_defaults_bool_not_int():
    """The default-setter must assign False (bool) for has_* columns."""
    from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
    from aivc.data.dataset_kind import DatasetKind

    adata = _mock_adata()
    for col in ["has_rna", "has_atac", "has_protein", "has_phospho"]:
        if col in adata.obs.columns:
            del adata.obs[col]
    adata.obs["perturbation_id"] = 0
    adata.obs["dataset_id"] = 0
    adata.obs["donor_id"] = "test"
    adata.obs["in_test_set"] = False
    adata.obs["USE_FOR_W_ONLY"] = False
    adata.obs["dataset_kind"] = DatasetKind.INTERVENTIONAL.value

    loader = MultiPerturbationLoader()
    loader._datasets["kang"] = adata
    combined = loader.build_combined_corpus()

    assert combined.obs["has_rna"].dtype == bool
    assert not combined.obs["has_rna"].any()


def test_mask_from_obs_integrates_with_tagged_dataset():
    """End-to-end: tagged Kang dataset -> mask_from_obs yields RNA-only mask."""
    from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
    from aivc.data.modality_mask import mask_from_obs, ModalityKey

    adata = _mock_adata()
    path = os.path.join(tempfile.mkdtemp(), "kang.h5ad")
    adata.write_h5ad(path)

    loader = MultiPerturbationLoader(kang_path=path, kang_test_donors=[])
    kang = loader.load_kang()

    mask = mask_from_obs(kang.obs.iloc[0])
    assert mask.shape == (4,)
    assert mask[int(ModalityKey.RNA)] == 1.0
    assert mask[int(ModalityKey.ATAC)] == 0.0
    assert mask[int(ModalityKey.PROTEIN)] == 0.0
    assert mask[int(ModalityKey.PHOSPHO)] == 0.0
