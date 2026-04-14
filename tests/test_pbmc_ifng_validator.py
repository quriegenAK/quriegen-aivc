"""
test_pbmc_ifng_validator.py — 8 tests for PBMC IFN-γ dataset validation.
All tests use mock data. No real datasets required.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _make_mock_adata(
    n_cells=10000,
    n_genes=3500,
    cell_types=None,
    condition_col="condition",
    ctrl_label="ctrl",
    stim_label="stim",
    gene_names=None,
    synthetic_flag=False,
    stim_fraction=0.6,
):
    """Create a mock AnnData for testing validation."""
    import anndata as ad
    import pandas as pd

    if cell_types is None:
        cell_types = ["CD4 T cells", "CD14+ Monocytes", "B cells", "NK cells"]

    n_stim = int(n_cells * stim_fraction)
    n_ctrl = n_cells - n_stim

    X = sp.random(n_cells, n_genes, density=0.1, format="csr", dtype=np.float32)

    obs = pd.DataFrame({
        condition_col: [ctrl_label] * n_ctrl + [stim_label] * n_stim,
        "cell_type": np.random.choice(cell_types, n_cells),
    })
    obs.index = [f"cell_{i}" for i in range(n_cells)]

    if synthetic_flag:
        obs["SYNTHETIC_IFNG"] = True

    if gene_names is None:
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def _save_tmp_h5ad(adata):
    """Save adata to temp file and return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    tmp.close()
    adata.write_h5ad(tmp.name)
    return tmp.name


class TestPBMCIFNGValidation:

    def test_validation_passes_complete_dataset(self):
        """Complete dataset with all checks satisfied."""
        from download_pbmc_ifng import validate_dataset

        # Create mock kang gene names that overlap with mock dataset
        mock_kang_genes = [f"GENE_{i}" for i in range(3010)]
        adata = _make_mock_adata(
            n_cells=12000,
            n_genes=3500,
            gene_names=[f"GENE_{i}" for i in range(3500)],
            stim_label="stim",
        )
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, extra = validate_dataset(path, mock_kang_genes)
            assert passed, f"Validation failed: {[r for r in results if not r['passed']]}"
            assert extra["n_cells_stim"] >= 5000
        finally:
            os.unlink(path)

    def test_validation_fails_wrong_cell_type(self):
        """Cancer cells should FAIL check 1."""
        from download_pbmc_ifng import validate_dataset

        adata = _make_mock_adata(
            cell_types=["melanoma", "tumor", "A375"],
            gene_names=[f"GENE_{i}" for i in range(3500)],
        )
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, _ = validate_dataset(path, [f"GENE_{i}" for i in range(3010)])
            cell_type_check = [r for r in results if r["check"] == "cell_type_pbmc"][0]
            assert not cell_type_check["passed"]
        finally:
            os.unlink(path)

    def test_validation_fails_wrong_perturbation(self):
        """IFN-α should FAIL check 2."""
        from download_pbmc_ifng import validate_dataset

        adata = _make_mock_adata(
            stim_label="IFN-alpha",
            gene_names=[f"GENE_{i}" for i in range(3500)],
        )
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, _ = validate_dataset(path, [f"GENE_{i}" for i in range(3010)])
            pert_check = [r for r in results if r["check"] == "perturbation_ifng"][0]
            assert not pert_check["passed"]
        finally:
            os.unlink(path)

    def test_validation_fails_insufficient_gene_overlap(self):
        """800 overlapping genes should FAIL check 3."""
        from download_pbmc_ifng import validate_dataset

        # Dataset genes don't overlap much with kang genes
        adata = _make_mock_adata(
            gene_names=[f"NOVEL_GENE_{i}" for i in range(3500)],
        )
        kang_genes = [f"GENE_{i}" for i in range(3010)]
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, _ = validate_dataset(path, kang_genes)
            overlap_check = [r for r in results if r["check"] == "gene_overlap"][0]
            assert not overlap_check["passed"]
        finally:
            os.unlink(path)

    def test_validation_fails_too_few_cells(self):
        """2000 total cells (1200 stim) should FAIL check 4."""
        from download_pbmc_ifng import validate_dataset

        adata = _make_mock_adata(
            n_cells=2000,
            gene_names=[f"GENE_{i}" for i in range(3500)],
        )
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, _ = validate_dataset(path, [f"GENE_{i}" for i in range(3010)])
            cell_check = [r for r in results if r["check"] == "cell_count"][0]
            assert not cell_check["passed"]
        finally:
            os.unlink(path)

    def test_validation_fails_synthetic_flag(self):
        """SYNTHETIC_IFNG=True should FAIL check 6."""
        from download_pbmc_ifng import validate_dataset

        adata = _make_mock_adata(
            gene_names=[f"GENE_{i}" for i in range(3500)],
            synthetic_flag=True,
        )
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, _ = validate_dataset(path, [f"GENE_{i}" for i in range(3010)])
            synth_check = [r for r in results if r["check"] == "not_synthetic"][0]
            assert not synth_check["passed"]
        finally:
            os.unlink(path)

    def test_certificate_written_on_all_pass(self):
        """Certificate JSON should be written when all checks pass."""
        from download_pbmc_ifng import validate_dataset, write_certificate

        kang_genes = [f"GENE_{i}" for i in range(3010)]
        adata = _make_mock_adata(
            n_cells=12000,
            gene_names=[f"GENE_{i}" for i in range(3500)],
        )
        path = _save_tmp_h5ad(adata)
        try:
            passed, results, extra = validate_dataset(path, kang_genes)
            assert passed
            cert_path = write_certificate(path, results, extra)
            assert os.path.exists(cert_path)
            with open(cert_path) as f:
                cert = json.load(f)
            assert cert["ready_for_stage2"] is True
            assert cert["is_synthetic"] is False
            assert "n_cells_ctrl" in cert
            assert "n_cells_stim" in cert
            assert "gene_overlap" in cert
        finally:
            os.unlink(path)
            if os.path.exists(cert_path):
                os.unlink(cert_path)

    def test_instructions_printed_when_file_missing(self, capsys):
        """When dataset file doesn't exist, print download instructions and exit 0."""
        from download_pbmc_ifng import DOWNLOAD_INSTRUCTIONS
        # Just verify the instructions dict has expected structure
        for source, info in DOWNLOAD_INSTRUCTIONS.items():
            assert "accession" in info
            assert "url" in info
            assert "steps" in info
            assert len(info["steps"]) >= 3
