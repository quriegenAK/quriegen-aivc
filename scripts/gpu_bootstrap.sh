#!/usr/bin/env bash
# scripts/gpu_bootstrap.sh — GPU instance bootstrap for AIVC pretrain run.
# Run once on a fresh GPU instance before launching training.
#
# IMPORTANT: This script contains placeholders (marked TODO) that must be
# filled in with cluster-specific values before use. Do NOT run as-is on
# a fresh instance.
#
# Usage:
#   bash scripts/gpu_bootstrap.sh
#
# After bootstrap, launch training:
#   python scripts/pretrain_multiome.py --config configs/dogma_pretrain.yaml \
#       --arm joint --checkpoint_dir checkpoints/pretrain
#
# Resume on preemption (load any periodic ckpt — they are all resume-able):
#   python scripts/pretrain_multiome.py --config configs/dogma_pretrain.yaml \
#       --arm joint --checkpoint_dir checkpoints/pretrain \
#       --resume checkpoints/pretrain/pretrain_encoders_epoch_NNNN.pt

set -euo pipefail

echo "[bootstrap] AIVC GPU instance setup starting..."
echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- 1. Cluster modules (TODO: fill in actual module names) ---
# Examples for common HPC clusters:
#   module load python/3.11
#   module load cuda/12.1
#   module load cudnn/8.9
# TODO: REPLACE WITH CLUSTER-SPECIFIC MODULE LOADS

# --- 2. W&B credentials (TODO: fill in) ---
export WANDB_ENTITY="${WANDB_ENTITY:-quriegen}"  # TODO: confirm entity name
# export WANDB_API_KEY="..."  # TODO: set securely (e.g., via secret manager)
echo "[bootstrap] WANDB_ENTITY=${WANDB_ENTITY:-unset}"
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "[bootstrap] WANDB_API_KEY=set"
else
  echo "[bootstrap] WANDB_API_KEY=NOT SET — set before training or run wandb login"
fi

# --- 3. Verify Python ---
echo "[bootstrap] Python: $(which python)"
echo "[bootstrap] Python version: $(python --version 2>&1)"
PY_MAJOR_MINOR=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PY_MAJOR_MINOR" != "3.11" ] && [ "$PY_MAJOR_MINOR" != "3.12" ]; then
  echo "[bootstrap] WARNING: project tested on Python 3.11; you have $PY_MAJOR_MINOR"
fi

# --- 4. Install dependencies ---
echo "[bootstrap] Installing project dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# --- 5. Verify GPU visibility ---
python - <<'PY'
import sys
import torch
print(f"[bootstrap] torch version: {torch.__version__}")
print(f"[bootstrap] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[bootstrap] CUDA version: {torch.version.cuda}")
    print(f"[bootstrap] GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[bootstrap] GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"            memory: {props.total_memory / 1024**3:.1f} GB, "
              f"compute capability: {props.major}.{props.minor}")
else:
    print("[bootstrap] WARNING: CUDA not available — full DOGMA run unworkable on CPU")
    sys.exit(1)
PY

# --- 6. Sanity check pretrain script ---
python scripts/pretrain_multiome.py --help > /dev/null && echo "[bootstrap] pretrain script: OK"

# --- 7. Verify required artifacts (TODO: adjust paths if rsynced elsewhere) ---
REQUIRED_ARTIFACTS=(
    "configs/dogma_pretrain.yaml"
    "data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad"
    "data/phase6_5g_2/dogma_h5ads/dogma_dig_union.h5ad"
)
for f in "${REQUIRED_ARTIFACTS[@]}"; do
    if [ -f "$f" ]; then
        echo "[bootstrap]   OK    $f ($(du -h "$f" | cut -f1))"
    else
        echo "[bootstrap]   MISSING  $f — TODO rsync from local OR pull from blob store"
    fi
done

echo "[bootstrap] Setup complete."
echo "[bootstrap] Launch with: python scripts/pretrain_multiome.py --config configs/dogma_pretrain.yaml --arm joint --checkpoint_dir checkpoints/pretrain"
