"""
aivc/training/ckpt_loader.py — strict loader for Phase 5 pretrained
checkpoints.

The loader is the Phase 6/6.5 integration contract for consuming
``checkpoints/pretrain/pretrain_encoders.pt``. It validates
``schema_version`` exactly, asserts every top-level key listed in
``aivc/training/PRETRAIN_CKPT_SCHEMA.md`` is present, and performs a
strict ``load_state_dict`` so missing/unexpected keys fail loud.

Any future consumer (linear probe, fine-tune entrypoint, downstream
ablations) MUST go through this module rather than calling
``torch.load`` directly. A silent skip at load time would invalidate
the pretraining ablation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch

from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
from aivc.skills.rna_encoder import SimpleRNAEncoder
from aivc.training.pretrain_heads import MultiomePretrainHead


EXPECTED_TOP_LEVEL_KEYS = (
    "schema_version",
    "rna_encoder",
    "atac_encoder",
    "pretrain_head",
    "rna_encoder_class",
    "atac_encoder_class",
    "pretrain_head_class",
    "config",
)

EXPECTED_RNA_ENCODER_STATE_KEYS = (
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "decoder.weight",
    "decoder.bias",
)


class CheckpointSchemaError(RuntimeError):
    """Raised on any schema / key / version mismatch."""


def _validate_top_level(ckpt: dict, expected_schema_version: int) -> None:
    missing = [k for k in EXPECTED_TOP_LEVEL_KEYS if k not in ckpt]
    if missing:
        raise CheckpointSchemaError(
            f"Pretrained checkpoint missing required top-level keys: "
            f"{missing}. Expected all of {list(EXPECTED_TOP_LEVEL_KEYS)}."
        )

    got_version = ckpt["schema_version"]
    if got_version != expected_schema_version:
        raise CheckpointSchemaError(
            f"Pretrained checkpoint schema_version mismatch: "
            f"expected {expected_schema_version!r}, got {got_version!r}. "
            f"See aivc/training/PRETRAIN_CKPT_SCHEMA.md."
        )


def _validate_rna_state_dict(rna_state: dict) -> None:
    got = set(rna_state.keys())
    expected = set(EXPECTED_RNA_ENCODER_STATE_KEYS)

    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing or unexpected:
        raise CheckpointSchemaError(
            f"rna_encoder state_dict mismatch. "
            f"missing_keys={missing}, unexpected_keys={unexpected}. "
            f"Expected exactly {sorted(expected)}."
        )


def peek_pretrain_ckpt_config(
    ckpt_path: Union[str, Path],
    expected_schema_version: int = 1,
) -> dict:
    """Return the ``config`` dict from a pretrain checkpoint after
    schema validation.

    Exists so downstream callers (linear probe, fine-tune entrypoints)
    do not need a bare ``torch.load`` just to read ``n_genes`` or
    ``hidden_dim``. The CI check in
    ``tests/test_no_bare_torch_load.py`` treats any bare
    ``torch.load(`` in ``scripts/`` or ``aivc/training/`` outside this
    module as a regression.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise CheckpointSchemaError(
            f"Pretrained checkpoint at {ckpt_path} is not a dict "
            f"(got {type(ckpt).__name__})."
        )
    _validate_top_level(ckpt, expected_schema_version)
    rna_state = ckpt["rna_encoder"]
    config = dict(ckpt.get("config") or {})
    # Backfill n_genes/hidden_dim/latent_dim from tensor shapes if the
    # saved config dict is missing them.
    config.setdefault("n_genes", int(rna_state["net.0.weight"].shape[1]))
    config.setdefault("hidden_dim", int(rna_state["net.0.weight"].shape[0]))
    config.setdefault("latent_dim", int(rna_state["net.2.weight"].shape[0]))
    return config


def load_pretrained_simple_rna_encoder(
    ckpt_path: Union[str, Path],
    expected_schema_version: int = 1,
) -> SimpleRNAEncoder:
    """Load a ``SimpleRNAEncoder`` from a Phase 5 pretrain checkpoint.

    Strictly validates:
      * ``schema_version`` matches ``expected_schema_version`` exactly.
      * All top-level keys in ``PRETRAIN_CKPT_SCHEMA.md`` are present.
      * The ``rna_encoder`` state_dict has exactly the expected keys
        (no missing, no unexpected).
      * ``load_state_dict(strict=True)`` succeeds.

    Raises
    ------
    CheckpointSchemaError
        On any version/key mismatch, naming the offending keys.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise CheckpointSchemaError(
            f"Pretrained checkpoint at {ckpt_path} is not a dict "
            f"(got {type(ckpt).__name__}). Expected a torch.save dict "
            f"per PRETRAIN_CKPT_SCHEMA.md."
        )

    _validate_top_level(ckpt, expected_schema_version)

    rna_state = ckpt["rna_encoder"]
    if not isinstance(rna_state, dict):
        raise CheckpointSchemaError(
            f"ckpt['rna_encoder'] is not a state_dict mapping "
            f"(got {type(rna_state).__name__})."
        )
    _validate_rna_state_dict(rna_state)

    # Recover constructor args from the saved config dict.
    config = ckpt.get("config", {}) or {}
    n_genes = int(
        config.get("n_genes")
        or rna_state["net.0.weight"].shape[1]
    )
    hidden_dim = int(
        config.get("hidden_dim")
        or rna_state["net.0.weight"].shape[0]
    )
    latent_dim = int(
        config.get("latent_dim")
        or rna_state["net.2.weight"].shape[0]
    )

    encoder = SimpleRNAEncoder(
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    # strict=True: any missing or unexpected key is a hard failure.
    missing_keys, unexpected_keys = encoder.load_state_dict(
        rna_state, strict=True
    )
    # Belt-and-braces: strict=True already raises, but PyTorch returns
    # named tuples on success so we assert empties for safety.
    if missing_keys or unexpected_keys:  # pragma: no cover - defensive
        raise CheckpointSchemaError(
            f"load_state_dict reported missing={list(missing_keys)}, "
            f"unexpected={list(unexpected_keys)} despite strict=True."
        )

    return encoder


def load_full_pretrain_checkpoint(
    ckpt_path: Union[str, Path],
    expected_schema_version: int = 1,
) -> Tuple[SimpleRNAEncoder, PeakLevelATACEncoder, MultiomePretrainHead, dict]:
    """Load the complete Phase 5 pretrain checkpoint for fine-tuning.

    Returns ``(rna_encoder, atac_encoder, pretrain_head, config)`` with
    every state_dict loaded via ``strict=True``. Phase 6.5e's
    contrastive fine-tune entrypoint is the first consumer. Existing
    callers (linear probe, RSA) only need the RNA encoder and stay on
    ``load_pretrained_simple_rna_encoder``.

    Intentionally additive: extending the loader here rather than
    re-introducing a bare ``torch.load`` in the fine-tune script keeps
    the ``tests/test_no_bare_torch_load.py`` CI invariant intact.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise CheckpointSchemaError(
            f"Pretrained checkpoint at {ckpt_path} is not a dict "
            f"(got {type(ckpt).__name__})."
        )
    _validate_top_level(ckpt, expected_schema_version)

    rna_state = ckpt["rna_encoder"]
    if not isinstance(rna_state, dict):
        raise CheckpointSchemaError(
            f"ckpt['rna_encoder'] is not a state_dict mapping "
            f"(got {type(rna_state).__name__})."
        )
    _validate_rna_state_dict(rna_state)

    config = dict(ckpt.get("config") or {})
    n_genes = int(config.get("n_genes") or rna_state["net.0.weight"].shape[1])
    hidden_dim = int(config.get("hidden_dim") or rna_state["net.0.weight"].shape[0])
    latent_dim = int(config.get("latent_dim") or rna_state["net.2.weight"].shape[0])

    rna_encoder = SimpleRNAEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, latent_dim=latent_dim
    )
    rna_encoder.load_state_dict(rna_state, strict=True)

    atac_state = ckpt["atac_encoder"]
    # Recover ATAC encoder constructor args from state_dict shapes so
    # we do not depend on the saved config dict having every field.
    n_peaks = int(atac_state["lsi.weight"].shape[1])
    svd_dim = int(atac_state["lsi.weight"].shape[0])
    # mlp.1 is the first Linear after the opening GroupNorm (index 0
    # is GroupNorm). mlp.5 is the final Linear (after GroupNorm-GELU-Dropout).
    hidden_dim_atac = int(atac_state["mlp.1.weight"].shape[0])
    attn_dim = int(atac_state["mlp.5.weight"].shape[0])
    atac_encoder = PeakLevelATACEncoder(
        n_peaks=n_peaks,
        svd_dim=svd_dim,
        hidden_dim=hidden_dim_atac,
        attn_dim=attn_dim,
    )
    atac_encoder.load_state_dict(atac_state, strict=True)

    head_state = ckpt["pretrain_head"]
    proj_dim = int(head_state["rna_proj.2.weight"].shape[0])
    head_hidden = int(head_state["rna_proj.0.weight"].shape[0])
    n_genes_head = int(head_state["peak_to_gene.weight"].shape[0])
    pretrain_head = MultiomePretrainHead(
        rna_dim=latent_dim,
        atac_dim=attn_dim,
        proj_dim=proj_dim,
        n_genes=n_genes_head,
        hidden_dim=head_hidden,
    )
    pretrain_head.load_state_dict(head_state, strict=True)

    config.setdefault("n_genes", n_genes)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("latent_dim", latent_dim)
    config.setdefault("n_peaks", n_peaks)
    config.setdefault("atac_attn_dim", attn_dim)
    config.setdefault("proj_dim", proj_dim)
    return rna_encoder, atac_encoder, pretrain_head, config
