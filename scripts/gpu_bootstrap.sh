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

# --- 1. BSC MareNostrum5 ACC partition modules (validated on H100 node 2026-04-30) ---
# Load order is load-bearing. CUDA stack is 11.8 (via nvidia-hpc-sdk), NOT 12.4.
# pytorch/2.4.0 module supplies torch via the system, so pip install can use
# --no-deps for project itself and skip torch reinstall (offline wheel workflow
# on BSC: transfer non-torch wheels from Mac, install with --no-index).
module load gcc/11.4.0
module load mkl/2024.0
module load impi/2021.11
module load hdf5/1.14.1-2-gcc
module load openblas/0.3.27-gcc
module load nccl/2.19.4
module load nvidia-hpc-sdk/23.11-cuda11.8
module load cudnn/9.0.0-cuda11
module load tensorrt/10.0.0-cuda11
module load python/3.11.5-gcc
module load pytorch/2.4.0
echo "[bootstrap] Modules loaded: gcc/11.4.0 + mkl/2024.0 + impi/2021.11 + hdf5/1.14.1-2-gcc + openblas/0.3.27-gcc + nccl/2.19.4 + nvidia-hpc-sdk/23.11-cuda11.8 + cudnn/9.0.0-cuda11 + tensorrt/10.0.0-cuda11 + python/3.11.5-gcc + pytorch/2.4.0"

# --- 1b. PYTHONPATH for user-level offline-installed wheels ---
# BSC login nodes lack pypi access; deps installed via wheel transfer from Mac
# land at ~/.local/lib/python3.11/site-packages. Prepend so Python finds them.
export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"
echo "[bootstrap] PYTHONPATH includes ~/.local/lib/python3.11/site-packages"

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

# --- 7. Verify required artifacts (BSC scratch layout) ---
SCRATCH_BASE="${SCRATCH_BASE:-/gpfs/scratch/ehpc748/quri020505/aivc_genelink}"
REQUIRED_ARTIFACTS=(
    "$SCRATCH_BASE/configs/dogma_pretrain.yaml"
    "$SCRATCH_BASE/data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad"
    "$SCRATCH_BASE/data/phase6_5g_2/dogma_h5ads/dogma_dig_union.h5ad"
    "$SCRATCH_BASE/data/calderon2019/calderon_atac_hg38.h5ad"
    "$SCRATCH_BASE/data/calderon2019/calderon_to_dogma_lll_M.npz"
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
