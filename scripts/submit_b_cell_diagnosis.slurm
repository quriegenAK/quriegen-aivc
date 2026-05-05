#!/bin/bash
#SBATCH --job-name=aivc_bcell_diag
#SBATCH --account=ehpc748
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scratch/ehpc748/quri020505/logs/bcell_diag_%j.out
#SBATCH --error=/gpfs/scratch/ehpc748/quri020505/logs/bcell_diag_%j.err

# Stage 3 Part 1 Report 5 — B-cell weakness diagnosis.
# Operates on the existing dogma_lll_union_labeled.h5ad on BSC scratch.
# Uses the validated DOGMA encoder at the production ckpt path.

set -euo pipefail

CKPT_PATH="${1:?ABORT: pass ckpt path as first arg}"
[[ -f "$CKPT_PATH" ]] || { echo "ABORT: ckpt not found: $CKPT_PATH"; exit 1; }

SCRATCH=/gpfs/scratch/ehpc748/quri020505
PROJECT=$SCRATCH/aivc_genelink
LOGS=$SCRATCH/logs
RESULTS_DIR=$SCRATCH/results/stage3_prep

mkdir -p "$RESULTS_DIR" "$LOGS"

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

DOGMA_LABELED=$PROJECT/data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad
TEST1_5_JSON=$SCRATCH/results/calderon_eval/pretrain_encoders_pseudobulk_test1_excl_other_other_T.json
CALDERON=$PROJECT/data/calderon2019/calderon_atac_hg38.h5ad
OUT_JSON=$RESULTS_DIR/b_cell_diagnosis.json

for f in "$DOGMA_LABELED" "$TEST1_5_JSON" "$CALDERON"; do
    [[ -f "$f" ]] || { echo "ABORT: missing $f"; exit 2; }
done

echo "[bcell_diag $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK')"

python scripts/probe_b_cell_diagnosis.py \
    --dogma_labeled "$DOGMA_LABELED" \
    --encoder_ckpt "$CKPT_PATH" \
    --test1_5_results_json "$TEST1_5_JSON" \
    --calderon "$CALDERON" \
    --out "$OUT_JSON" \
    --device cuda

echo
echo "==========================================================="
echo "[bcell_diag $SLURM_JOB_ID] B-cell diagnosis verdict"
echo "==========================================================="
python -c "
import json
r = json.load(open('$OUT_JSON'))
v = r['verdict']
print(f'  Dominant hypothesis: {v[\"dominant_hypothesis\"]}')
print(f'  H1 score: {v[\"h1_score\"]} / 2')
print(f'  H2 score: {v[\"h2_score\"]} / 2')
print(f'  H3 score: {v[\"h3_score\"]} / 2')
print()
for note in v['notes']:
    print(f'  - {note}')
print()
print(f'  Recommendation: {v[\"recommendation\"]}')
"
echo "==========================================================="
echo "[bcell_diag $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished"
echo "[bcell_diag $SLURM_JOB_ID] result: $OUT_JSON"
