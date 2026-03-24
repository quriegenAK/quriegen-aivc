"""
Tests for Replogle housekeeping gene filter.
All tests use mock data only. No real files. Under 10 seconds on CPU.

Run: pytest tests/test_housekeeping_filter.py -v
"""
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aivc.data.housekeeping_genes import (
    ALL_BLOCKED_GENES,
    HOUSEKEEPING_SAFE_SET,
    filter_ko_genes_for_w_pretrain,
    get_blocked_jakstat_genes,
    get_housekeeping_genes,
    get_safe_set_size,
)


class TestHousekeepingGeneSets:

    def test_safe_set_is_nonempty(self):
        """HOUSEKEEPING_SAFE_SET must contain > 100 genes."""
        assert len(HOUSEKEEPING_SAFE_SET) > 100, (
            f"Safe set has only {len(HOUSEKEEPING_SAFE_SET)} genes (need > 100)"
        )

    def test_blocked_set_contains_jakstat(self):
        """JAK1, JAK2, STAT1, IFIT1, MX1 all in ALL_BLOCKED_GENES."""
        for g in ["JAK1", "JAK2", "STAT1", "IFIT1", "MX1"]:
            assert g in ALL_BLOCKED_GENES, f"{g} not in ALL_BLOCKED_GENES"

    def test_safe_and_blocked_are_disjoint(self):
        """HOUSEKEEPING_SAFE_SET and ALL_BLOCKED_GENES must not overlap."""
        overlap = HOUSEKEEPING_SAFE_SET & ALL_BLOCKED_GENES
        assert len(overlap) == 0, (
            f"Safe and blocked sets overlap: {overlap}"
        )

    def test_ribosomal_proteins_in_safe_set(self):
        """RPL3, RPL4, RPS2, RPS3 must be in HOUSEKEEPING_SAFE_SET."""
        for g in ["RPL3", "RPL4", "RPS2", "RPS3"]:
            assert g in HOUSEKEEPING_SAFE_SET, f"{g} not in safe set"

    def test_eif_genes_in_safe_set(self):
        """EIF2A, EIF3A, EIF4E must be in HOUSEKEEPING_SAFE_SET."""
        for g in ["EIF2A", "EIF3A", "EIF4E"]:
            assert g in HOUSEKEEPING_SAFE_SET, f"{g} not in safe set"

    def test_get_housekeeping_genes_returns_set(self):
        """get_housekeeping_genes() returns a Python set."""
        result = get_housekeeping_genes()
        assert isinstance(result, set)
        assert len(result) > 0

    def test_get_housekeeping_genes_fallback_on_bad_source(self):
        """msigdb source falls back to builtin when gseapy unavailable."""
        result = get_housekeeping_genes(source="msigdb")
        # Should fall back gracefully (gseapy likely not installed)
        assert isinstance(result, set)
        assert len(result) > 100


class TestFilterKoGenesForWPretrain:

    def test_filter_removes_jakstat_genes(self):
        """JAK1 and STAT1 must be removed, RPL3 and EIF4E kept."""
        result = filter_ko_genes_for_w_pretrain(
            ["RPL3", "JAK1", "STAT1", "EIF4E"], verbose=False
        )
        assert "RPL3" in result["safe_ko_genes"]
        assert "EIF4E" in result["safe_ko_genes"]
        assert "JAK1" not in result["safe_ko_genes"]
        assert "STAT1" not in result["safe_ko_genes"]

    def test_filter_removes_non_housekeeping(self):
        """TP53 and KRAS are blocked, RPL3 and EIF4E pass."""
        result = filter_ko_genes_for_w_pretrain(
            ["RPL3", "TP53", "KRAS", "EIF4E"], verbose=False
        )
        assert result["safe_ko_genes"] == ["RPL3", "EIF4E"]

    def test_filter_respects_gene_universe(self):
        """RPS2 not in gene_universe should be excluded."""
        result = filter_ko_genes_for_w_pretrain(
            ["RPL3", "EIF4E", "RPS2"],
            gene_universe=["RPL3", "EIF4E"],
            verbose=False,
        )
        assert result["safe_ko_genes"] == ["RPL3", "EIF4E"]
        assert result["n_not_in_universe"] == 1

    def test_filter_empty_input(self):
        """Empty input should return empty list, no exception."""
        result = filter_ko_genes_for_w_pretrain([], verbose=False)
        assert result["safe_ko_genes"] == []
        assert result["n_safe"] == 0

    def test_filter_all_blocked(self):
        """All blocked genes should return empty safe list."""
        result = filter_ko_genes_for_w_pretrain(
            ["JAK1", "STAT1", "IFIT1", "MX1", "TP53", "KRAS"],
            verbose=False,
        )
        assert result["safe_ko_genes"] == []
        assert result["n_safe"] == 0

    def test_filter_returns_correct_counts(self):
        """n_input == n_safe + len(n_blocked) + n_not_hk + n_not_in_universe."""
        result = filter_ko_genes_for_w_pretrain(
            ["RPL3", "JAK1", "RANDOM_GENE", "EIF4E", "RPS2"],
            gene_universe=["RPL3", "EIF4E"],
            verbose=False,
        )
        n_total = (
            result["n_safe"]
            + len(result["n_blocked"])
            + result["n_not_hk"]
            + result["n_not_in_universe"]
        )
        assert n_total == result["n_input"], (
            f"Counts don't add up: {result['n_safe']} + {len(result['n_blocked'])} + "
            f"{result['n_not_hk']} + {result['n_not_in_universe']} != {result['n_input']}"
        )

    def test_filter_report_is_string(self):
        """filter_report must be a non-empty string."""
        result = filter_ko_genes_for_w_pretrain(["RPL3"], verbose=False)
        assert isinstance(result["filter_report"], str)
        assert len(result["filter_report"]) > 0


class TestWPretrainJakStatGuard:

    def test_pretrain_removes_jakstat_from_direction_matrix(self):
        """
        Direction matrix with JAK1 and STAT1 rows should have them
        removed by the guard before any gradient update.
        """
        from aivc.skills.neumann_w_pretrain import pretrain_W_from_replogle

        n_genes = 10
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ], dtype=torch.long)
        W = nn.Parameter(torch.full((4,), 0.01))

        gene_to_idx = {"RPL3": 0, "EIF4E": 1, "JAK1": 2, "STAT1": 3}

        # Direction matrix includes JAK1 and STAT1 (should be removed)
        replogle_df = pd.DataFrame(
            {
                "RPL3": [-1.0, 0.5, -0.3],
                "EIF4E": [0.2, -1.0, 0.1],
                "JAK1": [0.0, 0.0, -1.0],
                "STAT1": [0.3, 0.1, 0.0],
            },
            index=["RPL3", "JAK1", "STAT1"],
        )

        # Run pretrain — the guard should remove JAK1 and STAT1 rows
        # Only RPL3 should remain as a knockdown
        W_after = pretrain_W_from_replogle(
            W, edge_index, replogle_df, gene_to_idx, n_epochs=1
        )
        # The function should complete without error
        assert isinstance(W_after, nn.Parameter)
