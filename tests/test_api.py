"""
Tests for AIVC inference API.
Uses FastAPI TestClient — no running server needed.
All tests on mock model (demo mode). Under 20 seconds.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

MOCK_CTRL = [0.1] * 3010


@pytest.fixture(scope="module")
def client():
    """Create TestClient with lifespan activated."""
    from api.server import app
    with TestClient(app) as c:
        yield c


class TestHealthAndInfo:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_returns_model_loaded_true(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True

    def test_model_info_returns_correct_structure(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        for key in ["model_version", "n_genes", "n_edges", "benchmark_r",
                     "what_is_live", "what_is_not_live", "endpoints"]:
            assert key in body, f"Missing key: {key}"

    def test_model_info_n_genes_is_3010(self, client):
        r = client.get("/model/info")
        assert r.json()["n_genes"] == 3010

    def test_model_info_lists_unimplemented_correctly(self, client):
        r = client.get("/model/info")
        not_live = str(r.json()["what_is_not_live"])
        assert "Protein encoder" in not_live
        assert "Active learning loop" in not_live


class TestPredictEndpoint:

    def test_predict_returns_200_with_valid_input(self, client):
        r = client.post("/predict", json={"ctrl_expression": MOCK_CTRL})
        assert r.status_code == 200

    def test_predict_response_has_correct_keys(self, client):
        r = client.post("/predict", json={"ctrl_expression": MOCK_CTRL})
        body = r.json()
        for key in ["predicted_expression", "top_genes", "model_version",
                     "is_trained", "latency_ms", "pearson_r_benchmark"]:
            assert key in body, f"Missing: {key}"

    def test_predict_predicted_expression_length(self, client):
        r = client.post("/predict", json={"ctrl_expression": MOCK_CTRL})
        assert len(r.json()["predicted_expression"]) == 3010

    def test_predict_top_genes_sorted_by_abs_lfc(self, client):
        r = client.post("/predict", json={
            "ctrl_expression": MOCK_CTRL, "return_top_k": 10,
        })
        top = r.json()["top_genes"]
        for i in range(len(top) - 1):
            assert abs(top[i]["log2_fc"]) >= abs(top[i + 1]["log2_fc"])

    def test_predict_rejects_wrong_length_expression(self, client):
        r = client.post("/predict", json={"ctrl_expression": [0.1] * 100})
        assert r.status_code == 422

    def test_predict_rejects_negative_expression(self, client):
        bad = MOCK_CTRL.copy()
        bad[0] = -1.0
        r = client.post("/predict", json={"ctrl_expression": bad})
        assert r.status_code == 422

    def test_predict_jakstat_predictions_present_when_requested(self, client):
        r = client.post("/predict", json={
            "ctrl_expression": MOCK_CTRL, "include_jakstat": True,
        })
        body = r.json()
        assert "jakstat_predictions" in body
        assert body["jakstat_predictions"] is not None

    def test_predict_default_perturbation_is_stim(self, client):
        r = client.post("/predict", json={"ctrl_expression": MOCK_CTRL})
        assert r.json()["perturbation_id"] == 1

    def test_predict_latency_ms_is_positive_float(self, client):
        r = client.post("/predict", json={"ctrl_expression": MOCK_CTRL})
        assert r.json()["latency_ms"] > 0.0


class TestInterveneEndpoint:

    def test_intervene_gene_ko_returns_200(self, client):
        r = client.post("/intervene", json={
            "intervention_type": "gene_ko",
            "target_genes": ["GENE_0"],
            "ctrl_expression": MOCK_CTRL,
        })
        assert r.status_code == 200

    def test_intervene_response_has_correct_keys(self, client):
        r = client.post("/intervene", json={
            "intervention_type": "gene_ko",
            "target_genes": ["GENE_0"],
            "ctrl_expression": MOCK_CTRL,
        })
        body = r.json()
        for key in ["is_valid", "intervention_type", "intervened_genes",
                     "top_affected_genes", "ifit1_delta", "jakstat_deltas",
                     "w_density", "latency_ms", "warnings"]:
            assert key in body, f"Missing: {key}"

    def test_intervene_invalid_type_returns_422(self, client):
        r = client.post("/intervene", json={
            "intervention_type": "invalid_type",
            "target_genes": ["GENE_0"],
            "ctrl_expression": MOCK_CTRL,
        })
        assert r.status_code == 422

    def test_intervene_pathway_block_multiple_genes(self, client):
        r = client.post("/intervene", json={
            "intervention_type": "pathway_block",
            "target_genes": ["GENE_0", "GENE_1", "GENE_2"],
            "ctrl_expression": MOCK_CTRL,
        })
        assert r.status_code == 200
        assert len(r.json()["intervened_genes"]) == 3

    def test_intervene_gene_oe_uses_fold_change(self, client):
        r = client.post("/intervene", json={
            "intervention_type": "gene_oe",
            "target_genes": ["GENE_0"],
            "ctrl_expression": MOCK_CTRL,
            "fold_change": 5.0,
        })
        assert r.status_code == 200
        assert r.json()["intervention_type"] == "do_gene_oe"


class TestEdgeEndpoints:

    def test_top_edges_returns_list(self, client):
        r = client.get("/model/top_edges?n=10")
        assert r.status_code == 200
        assert len(r.json()["edges"]) == 10

    def test_jakstat_endpoint_returns_structure(self, client):
        r = client.get("/model/jakstat")
        assert r.status_code == 200
        body = r.json()
        assert "edges" in body
        assert "w_density" in body
