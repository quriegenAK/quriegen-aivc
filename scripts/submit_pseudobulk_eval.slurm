#!/bin/bash
#SBATCH --job-name=aivc_pseudobulk
#SBATCH --account=ehpc748
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scratch/ehpc748/quri020505/logs/pseudobulk_%j.out
#SBATCH --error=/gpfs/scratch/ehpc748/quri020505/logs/pseudobulk_%j.err

# Test 1: pseudo-bulk DOGMA centroid vs Calderon bulk compatibility check.
#
# Usage:
#   sbatch scripts/submit_pseudobulk_eval.slurm <ckpt_path>
#
# Validates whether the encoder captures biology at BULK level by
# pseudo-bulking DOGMA per Azimuth lineage and computing top-1 NN
# accuracy of Calderon samples against those centroids.
#
# Decision matrix:
#   >= 0.50 → encoder is fine; eval was the problem
#   0.20-0.50 → mixed; run Test 2
#   < 0.20  → encoder doesn't transfer; proceed to Test 2

set -euo pipefail

CKPT_PATH="${1:?ABORT: pass ckpt path as first arg}"
[[ -f "$CKPT_PATH" ]] || { echo "ABORT: ckpt not found: $CKPT_PATH"; exit 1; }
EXCLUDE_CLASSES="${2:-}"  # optional: comma-sep cell_type names to drop from centroids

SCRATCH=/gpfs/scratch/ehpc748/quri020505
PROJECT=$SCRATCH/aivc_genelink
LOGS=$SCRATCH/logs
RESULTS_DIR=$SCRATCH/results/calderon_eval

CKPT_BASENAME=$(basename "$CKPT_PATH" .pt)
# When --exclude is set, suffix the result file so Test 1 and Test 1.5 don't clobber
RESULT_SUFFIX="test1"
if [[ -n "$EXCLUDE_CLASSES" ]]; then
    SAFE=$(echo "$EXCLUDE_CLASSES" | tr ',' '_')
    RESULT_SUFFIX="test1_excl_${SAFE}"
fi
mkdir -p "$RESULTS_DIR" "$LOGS"

# --- Modules ---
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

echo "[pseudobulk $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
echo "[pseudobulk $SLURM_JOB_ID] ckpt: $CKPT_PATH"
echo

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK')"

# --- Required inputs on disk ---
DOGMA_LABELED=$PROJECT/data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad
CALDERON=$PROJECT/data/calderon2019/calderon_atac_hg38.h5ad
PROJECTION=$PROJECT/data/calderon2019/calderon_to_dogma_union_M.npz
for f in "$DOGMA_LABELED" "$CALDERON" "$PROJECTION"; do
    [[ -f "$f" ]] || { echo "ABORT: missing $f"; exit 2; }
done

# --- Run Test 1 / Test 1.5 ---
OUT_JSON=$RESULTS_DIR/${CKPT_BASENAME}_pseudobulk_${RESULT_SUFFIX}.json
echo "[pseudobulk $SLURM_JOB_ID] Running Test 1 (exclude='${EXCLUDE_CLASSES}')..."
python scripts/eval_pseudobulk_compatibility.py \
    --dogma_labeled "$DOGMA_LABELED" \
    --calderon "$CALDERON" \
    --projection "$PROJECTION" \
    --encoder_ckpt "$CKPT_PATH" \
    --out "$OUT_JSON" \
    --exclude_centroid_classes "$EXCLUDE_CLASSES" \
    --device cuda
echo

# --- Headline summary block ---
echo "==========================================================="
echo "[pseudobulk $SLURM_JOB_ID] TEST 1 HEADLINE"
echo "==========================================================="
python -c "
import json
r = json.load(open('$OUT_JSON'))
print(f'  ckpt: $CKPT_BASENAME')
print(f'  encoder: PeakLevelATACEncoder (bare ATAC, not joint fusion)')
print()
print(f'  --- Pseudo-bulk DOGMA centroids ---')
for cls, n in r['dogma_centroids']['per_class_n_cells'].items():
    print(f'    {cls:<12s}: aggregated from {n} cells')
print()
print(f'  --- Calderon mapping (25 → 6 lineages) ---')
print(f'    kept:    {r[\"calderon\"][\"n_kept\"]} / {r[\"calderon\"][\"n_total\"]}')
print(f'    dropped: {r[\"calderon\"][\"n_dropped\"]}  '
      f'(types: {r[\"calderon\"][\"dropped_types\"]})')
print(f'    kept distribution: {r[\"calderon\"][\"kept_lineage_distribution\"]}')
print()
print(f'  --- Per-lineage accuracy (Top-1 NN against pseudo-bulk centroids) ---')
for ln, d in r['per_lineage_accuracy'].items():
    print(f'    {ln:<12s}: {d[\"accuracy\"]:.4f} (n={d[\"n\"]})')
print()
print(f'  TOP-1 NN OVERALL ACCURACY: {r[\"top1_nn_accuracy\"]:.4f}')
print(f'  Chance baseline (1/8):     {r[\"chance_baseline\"]:.4f}')
print(f'  Pre-registered threshold:  >=0.7000  (still applies, but Test 1')
print(f'                              is diagnostic — not the verdict point)')
print()
acc = r['top1_nn_accuracy']
print(f'  TEST 1 DIAGNOSIS (accuracy = {acc:.4f}):')
if acc >= 0.50:
    print(f'    ENCODER CAPTURES BIOLOGY AT BULK LEVEL')
    print(f'    Eval pipeline was the problem; no encoder change needed.')
    print(f'    Action: fix eval to use centroid-NN or similarity-based')
    print(f'            classification rather than per-cell linear probe.')
elif acc >= 0.20:
    print(f'    MIXED — encoder partial transfer.')
    print(f'    Action: run Test 2 (gene-activity diagnostic) to disambiguate.')
else:
    print(f'    ENCODER GENUINELY DOESNT TRANSFER.')
    print(f'    Action: run Test 2 to check whether peak space itself is')
    print(f'            the bottleneck before committing to Pivot B (PeakVI).')
"
echo "==========================================================="

echo "[pseudobulk $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished"
echo "[pseudobulk $SLURM_JOB_ID] result JSON:"
echo "    $OUT_JSON"
