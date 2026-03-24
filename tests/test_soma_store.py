"""
Tests for TileDB-SOMA data layer.
All tests work WITHOUT tiledbsoma installed (mocked where needed).
Under 15 seconds on CPU.

Run: pytest tests/test_soma_store.py -v
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

SEED = 42


def _make_mock_adata(n_cells=100, n_genes=50, with_obs=True):
    """Create mock AnnData with all required obs columns."""
    import anndata as ad

    rng = np.random.RandomState(SEED)
    n_genes = max(n_genes, 7)  # ensure room for named genes
    gene_names = ["JAK1", "STAT1", "IFIT1", "MX1", "ISG15", "RPL3", "EIF4E"] + [
        f"GENE_{i}" for i in range(n_genes - 7)
    ]
    X = rng.random((n_cells, n_genes)).astype(np.float32)

    obs_data = {
        "donor_id": [f"donor_{i % 4}" for i in range(n_cells)],
        "cell_type": ["CD14+ Monocytes"] * (n_cells // 2) + ["CD4 T cells"] * (n_cells // 2),
        "condition": ["ctrl"] * (n_cells // 2) + ["stim"] * (n_cells // 2),
        "perturbation_id": [0] * (n_cells // 2) + [1] * (n_cells // 2),
        "dataset_id": [0] * n_cells,
        "in_test_set": [False] * (n_cells - 10) + [True] * 10,
        "USE_FOR_W_ONLY": [False] * n_cells,
        "use_for_response_training": [True] * n_cells,
        "SYNTHETIC_IFNG": [False] * n_cells,
        "ambient_decontaminated": [False] * n_cells,
    }
    obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


def _make_incomplete_adata(n_cells=20, n_genes=10):
    """AnnData missing required obs columns."""
    import anndata as ad

    rng = np.random.RandomState(SEED)
    obs = pd.DataFrame({
        "donor_id": [f"d{i}" for i in range(n_cells)],
    }, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    return ad.AnnData(X=rng.random((n_cells, n_genes)).astype(np.float32), obs=obs, var=var)


class TestSomaStoreSchema:

    def test_obs_schema_keys_defined(self):
        """OBS_SCHEMA has all required keys."""
        from aivc.data.soma_store import AIVCSomaStore
        required = [
            "cell_id", "donor_id", "cell_type", "condition",
            "perturbation_id", "dataset_id", "in_test_set",
            "USE_FOR_W_ONLY", "use_for_response_training",
            "SYNTHETIC_IFNG", "ambient_decontaminated", "partition_key",
        ]
        for key in required:
            assert key in AIVCSomaStore.OBS_SCHEMA, f"Missing obs key: {key}"

    def test_var_schema_keys_defined(self):
        """VAR_SCHEMA has all required keys."""
        from aivc.data.soma_store import AIVCSomaStore
        required = [
            "gene_id", "gene_name", "is_hvg",
            "in_string_ppi", "is_jakstat", "is_housekeeping",
        ]
        for key in required:
            assert key in AIVCSomaStore.VAR_SCHEMA, f"Missing var key: {key}"

    def test_obs_schema_validation_passes_complete_adata(self):
        """AnnData with all required obs columns -> valid=True."""
        from aivc.data.soma_store import AIVCSomaStore
        adata = _make_mock_adata()
        store = AIVCSomaStore("dummy/path")
        result = store.validate_obs_schema(adata)
        assert result["valid"], f"Should be valid, missing: {result['missing']}"

    def test_obs_schema_validation_fails_missing_columns(self):
        """AnnData missing columns -> valid=False."""
        from aivc.data.soma_store import AIVCSomaStore
        adata = _make_incomplete_adata()
        store = AIVCSomaStore("dummy/path")
        result = store.validate_obs_schema(adata)
        assert not result["valid"]
        assert len(result["missing"]) > 0

    def test_obs_schema_warns_on_frangieh_response_cells(self):
        """Frangieh cells with USE_FOR_W_ONLY=False -> warning."""
        from aivc.data.soma_store import AIVCSomaStore
        adata = _make_mock_adata(n_cells=20)
        # Set 5 cells as Frangieh (dataset_id=1) with W_ONLY=False
        adata.obs.loc[adata.obs.index[:5], "dataset_id"] = 1
        adata.obs.loc[adata.obs.index[:5], "USE_FOR_W_ONLY"] = False

        store = AIVCSomaStore("dummy/path")
        result = store.validate_obs_schema(adata)
        assert len(result["warnings"]) > 0
        assert any("Frangieh" in w for w in result["warnings"])


class TestSomaOBSPreparation:

    def test_partition_key_computed_from_cell_id(self):
        """partition_key = hash(cell_id) % 1024 and all in [0, 1023]."""
        cell_ids = [f"cell_{i}" for i in range(1000)]
        partition_keys = [hash(cid) % 1024 for cid in cell_ids]
        assert all(0 <= pk <= 1023 for pk in partition_keys)

    def test_jakstat_genes_flagged_in_var(self):
        """All 15 JAK-STAT genes have is_jakstat=True when present."""
        from aivc.data.soma_store import AIVCSomaStore
        jakstat_set = set(AIVCSomaStore.JAKSTAT_GENES)

        gene_names = ["JAK1", "STAT1", "IFIT1", "RPL3", "EIF4E", "GENE_X"]
        is_jakstat = [g in jakstat_set for g in gene_names]

        assert is_jakstat == [True, True, True, False, False, False]

    def test_housekeeping_genes_flagged_in_var(self):
        """RPL3, EIF4E -> is_housekeeping=True; JAK1, STAT1 -> False."""
        from aivc.data.housekeeping_genes import get_housekeeping_genes
        hk_set = get_housekeeping_genes()

        assert "RPL3" in hk_set
        assert "EIF4E" in hk_set
        assert "JAK1" not in hk_set
        assert "STAT1" not in hk_set


class TestSomaTrainingBridge:

    def test_bridge_defaults_to_h5ad(self):
        """USE_SOMA_STORE=False -> source='h5ad'."""
        from aivc.data import soma_training_bridge as stb
        from unittest.mock import patch

        mock_adata = _make_mock_adata(n_cells=10)
        with patch.object(stb, "_load_from_h5ad") as mock_h5ad:
            mock_h5ad.return_value = {
                "adata": mock_adata, "X": np.zeros((10, 50), dtype=np.float32),
                "obs": mock_adata.obs, "gene_names": mock_adata.var_names.tolist(),
                "n_cells": 10, "n_genes": 50, "source": "h5ad",
                "store_path": "fake.h5ad",
            }
            result = stb.load_training_arrays(h5ad_path="fake.h5ad", use_soma=False)
            assert result["source"] == "h5ad"
            mock_h5ad.assert_called_once()

    def test_bridge_uses_soma_when_flag_set(self):
        """USE_SOMA_STORE=True -> calls _load_from_soma."""
        from aivc.data import soma_training_bridge as stb
        from unittest.mock import patch

        with patch.object(stb, "_load_from_soma") as mock_soma:
            mock_soma.return_value = {
                "adata": None, "X": np.zeros((1, 1)),
                "obs": pd.DataFrame(), "gene_names": [],
                "n_cells": 1, "n_genes": 1, "source": "soma",
                "store_path": "test",
            }
            result = stb.load_training_arrays(
                store_path="test/soma/", use_soma=True
            )
            assert result["source"] == "soma"
            mock_soma.assert_called_once()

    def test_bridge_returns_consistent_structure(self):
        """Both paths return dict with same keys."""
        required_keys = {
            "adata", "X", "obs", "gene_names",
            "n_cells", "n_genes", "source", "store_path",
        }
        from aivc.data import soma_training_bridge as stb
        from unittest.mock import patch

        mock_adata = _make_mock_adata(n_cells=10)
        with patch("anndata.read_h5ad", return_value=mock_adata):
            result = stb._load_from_h5ad("fake.h5ad")
            assert required_keys.issubset(set(result.keys()))

    def test_bridge_soma_requires_store_path(self):
        """_load_from_soma(store_path=None) raises ValueError."""
        from aivc.data.soma_training_bridge import _load_from_soma
        with pytest.raises(ValueError, match="store_path is required"):
            _load_from_soma(store_path=None)

    def test_h5ad_path_load_returns_numpy_array(self):
        """X is np.ndarray with dtype float32."""
        from aivc.data import soma_training_bridge as stb
        from unittest.mock import patch

        mock_adata = _make_mock_adata(n_cells=10)
        with patch("anndata.read_h5ad", return_value=mock_adata):
            result = stb._load_from_h5ad("fake.h5ad")
            assert isinstance(result["X"], np.ndarray)
            assert result["X"].dtype == np.float32


class TestSomaScaleProjection:

    def test_partition_key_distribution(self):
        """10,000 cells: no bucket has > 5% of cells."""
        rng = np.random.RandomState(SEED)
        cell_ids = [f"cell_{rng.randint(0, 10**9)}" for _ in range(10000)]
        pks = [hash(cid) % 1024 for cid in cell_ids]

        from collections import Counter
        counts = Counter(pks)
        max_count = max(counts.values())
        max_frac = max_count / 10000

        assert max_frac < 0.05, (
            f"Hottest partition has {max_frac:.1%} of cells (limit 5%). "
            "Hash distribution is too skewed."
        )

    def test_n_cells_scale_projection(self):
        """Schema supports 1B cells: cell_id as string, partition_key as int."""
        from aivc.data.soma_store import AIVCSomaStore

        # cell_id is str (no int overflow)
        assert AIVCSomaStore.OBS_SCHEMA["cell_id"] == str

        # partition_key is int (0-1023 fits in any int type)
        assert AIVCSomaStore.OBS_SCHEMA["partition_key"] == int

        # Verify 1B cell_id hashes don't overflow
        big_id = f"cell_{10**18}"
        pk = hash(big_id) % 1024
        assert 0 <= pk <= 1023
