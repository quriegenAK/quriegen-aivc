# Pretrained checkpoint schema (Phase 5 → Phase 6 contract)

Frozen as of Phase 5 (`scripts/pretrain_multiome.py`). Phase 6's
`--pretrained_ckpt` loader **MUST** validate every key below and **fail
loudly** on mismatch — silent skipping invalidates the pretraining
ablation.

## Checkpoint dict keys

A `torch.save()` dict at `checkpoints/pretrain/pretrain_encoders.pt`
contains:

| Key | Type | Source |
| --- | ---- | ------ |
| `schema_version` | `int` | `scripts/pretrain_multiome.PRETRAIN_CKPT_SCHEMA_VERSION` (current: `1`) |
| `rna_encoder` | `state_dict` | `aivc.skills.rna_encoder.SimpleRNAEncoder` |
| `atac_encoder` | `state_dict` | `aivc.skills.atac_peak_encoder.PeakLevelATACEncoder` |
| `pretrain_head` | `state_dict` | `aivc.training.pretrain_heads.MultiomePretrainHead` |
| `rna_encoder_class` | `str` | `"aivc.skills.rna_encoder.SimpleRNAEncoder"` |
| `atac_encoder_class` | `str` | `"aivc.skills.atac_peak_encoder.PeakLevelATACEncoder"` |
| `pretrain_head_class` | `str` | `"aivc.training.pretrain_heads.MultiomePretrainHead"` |
| `config` | `dict` | argparse namespace vars from the pretrain run |

## RNA encoder state_dict keys (schema_version=1)

`SimpleRNAEncoder` is:
```
net.0: Linear(n_genes -> hidden_dim)
net.1: GELU
net.2: Linear(hidden_dim -> latent_dim)
decoder: Linear(latent_dim -> n_genes)
```

Expected parameter keys:
- `net.0.weight`, `net.0.bias`
- `net.2.weight`, `net.2.bias`
- `decoder.weight`, `decoder.bias`

## ATAC encoder state_dict keys (schema_version=1)

Delegated to `PeakLevelATACEncoder` (Phase 4). Parameter keys include
`lsi.weight` and the `mlp.*` subtree; exact list is determined by that
class and must be read live via `state_dict().keys()` rather than
hard-coded here.

## Pretrain head state_dict keys (schema_version=1)

`MultiomePretrainHead` is:
```
rna_proj:     Sequential(Linear -> GELU -> Linear)
atac_proj:    Sequential(Linear -> GELU -> Linear)
peak_to_gene: Linear(atac_dim -> n_genes)
```

Expected parameter keys:
- `rna_proj.0.weight`, `rna_proj.0.bias`
- `rna_proj.2.weight`, `rna_proj.2.bias`
- `atac_proj.0.weight`, `atac_proj.0.bias`
- `atac_proj.2.weight`, `atac_proj.2.bias`
- `peak_to_gene.weight`, `peak_to_gene.bias`

## Phase 6 consumer

Phase 6 fine-tuning must:

1. Load the checkpoint with `map_location="cpu"` first, read
   `schema_version`, and compare against its own constant.
   Mismatch → `RuntimeError` (NOT warning, NOT silent skip).
2. Instantiate `SimpleRNAEncoder`, `PeakLevelATACEncoder`, and
   `MultiomePretrainHead` with shapes compatible with the saved
   `config` dict.
3. Call `load_state_dict(..., strict=True)`. Missing/unexpected keys
   must raise.
4. (Architectural note.) `PerturbationPredictor` does NOT currently
   have an `rna_encoder` submodule — it operates on per-gene scalars
   via a GAT. Phase 6's fine-tuning head must therefore introduce a
   cell-level branch that composes `SimpleRNAEncoder` rather than
   grafting it into `PerturbationPredictor`. See Phase 5 PR body,
   "Phase 6 interface resolution".

## Versioning policy

Bump `schema_version` on any breaking change to checkpoint layout
(new/removed top-level keys, renamed submodules, shape-incompatible
parameter changes). Older Phase 6 binaries loading a newer schema
must raise; newer Phase 6 binaries loading an older schema may
adapt if they can, else raise.
