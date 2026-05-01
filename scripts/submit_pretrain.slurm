#!/bin/bash
#SBATCH --job-name=aivc_pretrain
#SBATCH --account=ehpc748
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/ehpc748/quri020505/logs/pretrain_%j.out
#SBATCH --error=/gpfs/scratch/ehpc748/quri020505/logs/pretrain_%j.err

# AIVC GeneLink — DOGMA tri-modal pretrain, joint LLL+DIG, 50 epochs.
#
# Single-GPU per spec section 9 (compute budget: 1 GPU, 12-24 GPU-h).
# DDP support is a future PR if wall time becomes constraining.
#
# Submit from project root after gpu_bootstrap.sh has installed deps
# on a login node:
#     export WANDB_API_KEY=<your_key>  # or `wandb login` once on cluster
#     sbatch scripts/submit_pretrain.slurm

set -euo pipefail

SCRATCH=/gpfs/scratch/ehpc748/quri020505
PROJECT=$SCRATCH/aivc_genelink
LOGS=$SCRATCH/logs
CKPT_DIR=$SCRATCH/checkpoints/pretrain
SMOKE_CKPT_DIR=$SCRATCH/checkpoints/smoke

mkdir -p $LOGS $CKPT_DIR $SMOKE_CKPT_DIR

# --- Modules (mirror gpu_bootstrap.sh; BSC MareNostrum5 ACC partition) ---
# Validated on H100 node as01r1b10 with job 39833102 (2026-04-30).
# CUDA stack is 11.8 via nvidia-hpc-sdk; pytorch/2.4.0 supplies torch from system.
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

# --- PYTHONPATH for user-level offline-installed wheels ---
# Project deps installed via offline wheel workflow land at ~/.local/lib/python3.11/site-packages.
export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"

# --- W&B credentials ---
export WANDB_ENTITY=quriegen
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in env before sbatch (or run wandb login)}"

cd $PROJECT

# --- Sanity: GPU visible, deps installed ---
echo "[job $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
nvidia-smi
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible'
print(f'GPU: {torch.cuda.get_device_name(0)}, '
      f'mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

# Verify deps installed (do NOT pip install inside GPU allocation)
python -c "import aivc, anndata, scipy, pyranges, sklearn, pyliftover, wandb; print('deps OK')" \
    || { echo "ERROR: dependencies missing. Run gpu_bootstrap.sh on login node first."; exit 1; }

# --- Smoke test (100 steps; ~5-10 min on a CUDA GPU) ---
echo "[job $SLURM_JOB_ID] Smoke: --arm joint --max-steps 100"
python scripts/pretrain_multiome.py \
    --config configs/dogma_pretrain.yaml \
    --arm joint \
    --max-steps 100 \
    --checkpoint_dir $SMOKE_CKPT_DIR \
    > $LOGS/smoke_${SLURM_JOB_ID}.log 2>&1
echo "[job $SLURM_JOB_ID] Smoke complete. Log tail:"
tail -20 $LOGS/smoke_${SLURM_JOB_ID}.log

# --- Auto-detect latest periodic ckpt for resume ---
LATEST_CKPT=""
if compgen -G "$CKPT_DIR/pretrain_encoders_epoch_*.pt" > /dev/null; then
    LATEST_CKPT=$(ls -t $CKPT_DIR/pretrain_encoders_epoch_*.pt | head -1)
    echo "[job $SLURM_JOB_ID] Resuming from periodic ckpt: $LATEST_CKPT"
elif [ -f "$CKPT_DIR/pretrain_encoders.pt" ]; then
    LATEST_CKPT=$CKPT_DIR/pretrain_encoders.pt
    echo "[job $SLURM_JOB_ID] Resuming from end-of-training ckpt: $LATEST_CKPT"
else
    echo "[job $SLURM_JOB_ID] No prior ckpt; starting fresh."
fi

# --- Full training run ---
RESUME_FLAG=""
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume $LATEST_CKPT"

echo "[job $SLURM_JOB_ID] Full run: --arm joint $RESUME_FLAG"
python scripts/pretrain_multiome.py \
    --config configs/dogma_pretrain.yaml \
    --arm joint \
    --checkpoint_dir $CKPT_DIR \
    $RESUME_FLAG \
    > $LOGS/pretrain_${SLURM_JOB_ID}.log 2>&1
EXIT=$?

echo "[job $SLURM_JOB_ID] Training exit code: $EXIT"
echo "[job $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished"
exit $EXIT
