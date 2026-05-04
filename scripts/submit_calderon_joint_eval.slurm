#!/bin/bash
#SBATCH --job-name=aivc_caldjoint
#SBATCH --account=ehpc748
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/ehpc748/quri020505/logs/caldjoint_%j.out
#SBATCH --error=/gpfs/scratch/ehpc748/quri020505/logs/caldjoint_%j.err

# Pivot A: Calderon eval through JOINT FUSION pathway.
#
# Forwards Calderon ATAC through full encoder stack with zero-padded
# RNA + Protein, runs linear probe on z_supcon. Compare against:
#   bare-ATAC kfold (job 39961124 / 39964435):     0.1943 ± 0.055
#   bare-ATAC LOO:                                  0.1780 ± 0.092
#   random projection mock encoder:                 0.4914
#   pre-registered threshold:                       0.70
#
# Usage:
#   sbatch scripts/submit_calderon_joint_eval.slurm <ckpt_path>
#
# Example:
#   sbatch scripts/submit_calderon_joint_eval.slurm \
#       /gpfs/scratch/ehpc748/quri020505/checkpoints/pretrain/pretrain_encoders.pt

set -euo pipefail

CKPT_PATH="${1:?ABORT: pass ckpt path as first arg}"
[[ -f "$CKPT_PATH" ]] || { echo "ABORT: ckpt not found: $CKPT_PATH"; exit 1; }

SCRATCH=/gpfs/scratch/ehpc748/quri020505
PROJECT=$SCRATCH/aivc_genelink
LOGS=$SCRATCH/logs
RESULTS_DIR=$SCRATCH/results/calderon_eval

CKPT_BASENAME=$(basename "$CKPT_PATH" .pt)
mkdir -p "$RESULTS_DIR" "$LOGS"

# --- Modules (mirror pretrain SLURM) ---
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

export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"

cd $PROJECT

echo "[caldjoint $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
echo "[caldjoint $SLURM_JOB_ID] ckpt: $CKPT_PATH"
echo

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK')"

CALDERON_H5AD=$PROJECT/data/calderon2019/calderon_atac_hg38.h5ad
PROJECTION_M=$PROJECT/data/calderon2019/calderon_to_dogma_union_M.npz
for f in "$CALDERON_H5AD" "$PROJECTION_M"; do
    [[ -f "$f" ]] || { echo "ABORT: missing $f"; exit 2; }
done

# --- Run 1: stratified k-fold (pre-registered metric) ---
KFOLD_OUT=$RESULTS_DIR/${CKPT_BASENAME}_joint_fusion_kfold.json
echo "[caldjoint $SLURM_JOB_ID] Run 1: stratified-kfold via joint fusion"
python scripts/eval_calderon_joint_fusion.py \
    --calderon "$CALDERON_H5AD" \
    --projection "$PROJECTION_M" \
    --encoder_ckpt "$CKPT_PATH" \
    --out "$KFOLD_OUT" \
    --cv stratified-kfold \
    --cv-folds 5 \
    --device cuda
echo

# --- Run 2: leave-one-donor-out ---
LOO_OUT=$RESULTS_DIR/${CKPT_BASENAME}_joint_fusion_loo_donor.json
echo "[caldjoint $SLURM_JOB_ID] Run 2: leave-one-donor-out via joint fusion"
python scripts/eval_calderon_joint_fusion.py \
    --calderon "$CALDERON_H5AD" \
    --projection "$PROJECTION_M" \
    --encoder_ckpt "$CKPT_PATH" \
    --out "$LOO_OUT" \
    --cv leave-one-donor-out \
    --device cuda
echo

# --- Headline summary ---
echo "==========================================================="
echo "[caldjoint $SLURM_JOB_ID] PIVOT A HEADLINE RESULTS"
echo "==========================================================="
python -c "
import json
kfold = json.load(open('$KFOLD_OUT'))
loo = json.load(open('$LOO_OUT'))
print(f'  ckpt: $CKPT_BASENAME')
print(f'  encoder: {kfold[\"encoder\"]}')
print()
print(f'  --- joint-fusion stratified-kfold ---')
print(f'    accuracy: {kfold[\"mean_accuracy\"]:.4f} ± {kfold[\"std_accuracy\"]:.4f}')
print(f'    f1_macro: {kfold[\"mean_f1_macro\"]:.4f}')
print()
print(f'  --- joint-fusion leave-one-donor-out ---')
print(f'    accuracy: {loo[\"mean_accuracy\"]:.4f} ± {loo[\"std_accuracy\"]:.4f}')
print(f'    f1_macro: {loo[\"mean_f1_macro\"]:.4f}')
print()
print(f'  --- comparison ---')
print(f'    chance (1/25):                          0.0400')
print(f'    bare-ATAC kfold (39961124):             0.1943   <- joint vs this')
print(f'    iteration-3 unsupervised:               0.1371')
print(f'    mock random projection (union M):       0.4914')
print(f'    PRE-REGISTERED THRESHOLD:              >=0.7000')
print()
joint_kfold = kfold['mean_accuracy']
print(f'  PIVOT A DIAGNOSIS (joint kfold = {joint_kfold:.4f}):')
if joint_kfold >= 0.70:
    print(f'    threshold passes via joint pathway — re-evaluate framing')
elif joint_kfold >= 0.50:
    print(f'    substantial lift over bare-ATAC; below threshold but informative')
elif joint_kfold >= 0.30:
    print(f'    modest lift; both encoder+fusion need work; favor Pivot C')
else:
    print(f'    no lift; encoder genuinely doesn\\'t transfer; favor Pivot B (PeakVI)')
"
echo "==========================================================="

echo "[caldjoint $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished"
echo "[caldjoint $SLURM_JOB_ID] results:"
echo "    $KFOLD_OUT"
echo "    $LOO_OUT"
