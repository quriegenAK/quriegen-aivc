"""Phase 5 -> Phase 6 contract: pretrained checkpoint schema freeze.

Runs scripts/pretrain_multiome.py for 1 step on a tiny mock batch and
asserts the saved checkpoint contains every key listed in
aivc/training/PRETRAIN_CKPT_SCHEMA.md, including schema_version.
"""
import os
import subprocess
import sys
import tempfile

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "rna_encoder",
    "atac_encoder",
    "pretrain_head",
    "rna_encoder_class",
    "atac_encoder_class",
    "pretrain_head_class",
    "config",
}

EXPECTED_RNA_KEYS = {
    "net.0.weight", "net.0.bias",
    "net.2.weight", "net.2.bias",
    "decoder.weight", "decoder.bias",
}

EXPECTED_HEAD_KEYS = {
    "rna_proj.0.weight", "rna_proj.0.bias",
    "rna_proj.2.weight", "rna_proj.2.bias",
    "atac_proj.0.weight", "atac_proj.0.bias",
    "atac_proj.2.weight", "atac_proj.2.bias",
    "peak_to_gene.weight", "peak_to_gene.bias",
}


@pytest.fixture(scope="module")
def _saved_ckpt(tmp_path_factory):
    ckpt_dir = tmp_path_factory.mktemp("pretrain_ckpt")
    cmd = [
        sys.executable, "scripts/pretrain_multiome.py",
        "--steps", "1",
        "--no_wandb",
        "--n_cells", "64",
        "--n_genes", "32",
        "--n_peaks", "64",
        "--batch_size", "16",
        "--rna_latent", "16",
        "--atac_latent", "8",
        "--proj_dim", "8",
        "--checkpoint_dir", str(ckpt_dir),
    ]
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True, capture_output=True)
    ckpt_path = ckpt_dir / "pretrain_encoders.pt"
    assert ckpt_path.exists(), "pretrain script did not produce a checkpoint"
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def test_top_level_keys_match_schema(_saved_ckpt):
    assert set(_saved_ckpt.keys()) >= EXPECTED_TOP_LEVEL_KEYS, (
        f"Missing top-level keys: "
        f"{EXPECTED_TOP_LEVEL_KEYS - set(_saved_ckpt.keys())}"
    )


def test_schema_version_present_and_int(_saved_ckpt):
    assert "schema_version" in _saved_ckpt
    assert isinstance(_saved_ckpt["schema_version"], int)
    assert _saved_ckpt["schema_version"] == 1


def test_rna_encoder_state_dict_keys(_saved_ckpt):
    keys = set(_saved_ckpt["rna_encoder"].keys())
    missing = EXPECTED_RNA_KEYS - keys
    assert not missing, f"RNA encoder state_dict missing {missing!r}; got {keys!r}"


def test_pretrain_head_state_dict_keys(_saved_ckpt):
    keys = set(_saved_ckpt["pretrain_head"].keys())
    missing = EXPECTED_HEAD_KEYS - keys
    assert not missing, f"Head state_dict missing {missing!r}; got {keys!r}"


def test_class_strings_match_promoted_paths(_saved_ckpt):
    assert _saved_ckpt["rna_encoder_class"] == "aivc.skills.rna_encoder.SimpleRNAEncoder"
    assert _saved_ckpt["atac_encoder_class"] == "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder"
    assert _saved_ckpt["pretrain_head_class"] == "aivc.training.pretrain_heads.MultiomePretrainHead"
