#!/bin/bash
#SBATCH --job-name=aivc_mimitou_probe
#SBATCH --account=ehpc748
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch/ehpc748/quri020505/logs/mimitou_probe_%j.out
#SBATCH --error=/gpfs/scratch/ehpc748/quri020505/logs/mimitou_probe_%j.err

# Stage 3 Part 1 Report 2 — Mimitou CD4 CRISPR perturbation probe.
#
# Pipeline:
#   Stage A: prepare_mimitou_crispr.py
#     Mimitou raw (Cell Ranger ATAC h5 + kite protein + guide demux CSV)
#     → harmonized labeled h5ad in DOGMA union peak space (323,500 peaks)
#   Stage B: probe_mimitou_perturbation_clustering.py
#     Frozen DOGMA encoder centroid-NN per perturbation arm + baselines.
#     Verdict: frozen-OK | adapter | fine-tune.
#
# Required inputs (must be pre-staged on BSC scratch):
#   $1 = path to encoder ckpt (e.g. /gpfs/scratch/.../pretrain_encoders.pt)
#   $2 = directory containing Mimitou CRISPR raw files
#        (filtered_peak_bc_matrix.h5, protein_counts.mtx,
#         protein_barcodes.tsv, protein_features.tsv, guide_demux.csv)
#
# Optional:
#   $3 = "donor_holdout" to enable leave-one-donor-out split
#
# Usage:
#   sbatch scripts/submit_mimitou_perturbation_probe.slurm \
#       /gpfs/scratch/ehpc748/quri020505/checkpoints/pretrain/pretrain_encoders.pt \
#       /gpfs/scratch/ehpc748/quri020505/data/mimitou_crispr_raw

set -euo pipefail

CKPT_PATH="${1:?ABORT: pass ckpt path as first arg}"
MIMITOU_DIR="${2:?ABORT: pass mimitou raw dir as second arg}"
DONOR_FLAG="${3:-}"

[[ -f "$CKPT_PATH" ]] || { echo "ABORT: ckpt not found: $CKPT_PATH"; exit 1; }
[[ -d "$MIMITOU_DIR" ]] || { echo "ABORT: mimitou raw dir not found: $MIMITOU_DIR"; exit 1; }

SCRATCH=/gpfs/scratch/ehpc748/quri020505
PROJECT=$SCRATCH/aivc_genelink
LOGS=$SCRATCH/logs
RESULTS_DIR=$SCRATCH/results/stage3_prep
H5AD_DIR=$PROJECT/data/phase6_5g_2/dogma_h5ads

mkdir -p "$RESULTS_DIR" "$LOGS"

# Verify required files exist in the raw dir.
# Layout matches scripts/download_mimitou_crispr.sh output (asap_reproducibility
# repo): Cell Ranger ATAC h5 at top + kite-format ADT (protein) + kite-format
# HTO (encodes per-cell sgRNA target via TotalSeq hashtag multiplexing).
REQUIRED_FILES=(
    filtered_peak_bc_matrix.h5
    adt/featurecounts.mtx
    adt/featurecounts.barcodes.txt
    adt/featurecounts.genes.txt
    hto/featurecounts.mtx
    hto/featurecounts.barcodes.txt
    hto/featurecounts.genes.txt
)
for f in "${REQUIRED_FILES[@]}"; do
    [[ -f "$MIMITOU_DIR/$f" ]] || {
        echo "ABORT: required input missing: $MIMITOU_DIR/$f"
        echo "Expected files in MIMITOU_DIR:"
        printf '  %s\n' "${REQUIRED_FILES[@]}"
        exit 2
    }
done

UNION_MANIFEST=$H5AD_DIR/UNION_MANIFEST.json
[[ -f "$UNION_MANIFEST" ]] || {
    # Fall back to TSV if JSON not present
    UNION_MANIFEST=$H5AD_DIR/dogma_union_peaks.tsv
    [[ -f "$UNION_MANIFEST" ]] || {
        echo "ABORT: no union manifest at $H5AD_DIR/UNION_MANIFEST.json"
        echo "       or $H5AD_DIR/dogma_union_peaks.tsv"
        exit 3
    }
}

OUT_H5AD=$H5AD_DIR/mimitou_crispr_union.h5ad
OUT_JSON=$RESULTS_DIR/mimitou_perturbation_probe.json

# --- Module stack (CLAUDE.md production sequence) ---
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

cd "$PROJECT"

echo "[mimitou_probe $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK')"

# --- Stage A: prepare ---
echo
echo "==========================================================="
echo "[mimitou_probe] Stage A: prepare_mimitou_crispr.py"
echo "==========================================================="

if [[ -f "$OUT_H5AD" ]]; then
    echo "[mimitou_probe] $OUT_H5AD already exists; skipping prep."
    echo "[mimitou_probe]   delete it to re-prep."
else
    python scripts/prepare_mimitou_crispr.py \
        --mimitou_crispr_dir "$MIMITOU_DIR" \
        --union_manifest "$UNION_MANIFEST" \
        --out "$OUT_H5AD"
fi
ls -lh "$OUT_H5AD"

# --- Stage B: probe ---
echo
echo "==========================================================="
echo "[mimitou_probe] Stage B: probe_mimitou_perturbation_clustering.py"
echo "==========================================================="

PROBE_FLAGS=()
if [[ "$DONOR_FLAG" == "donor_holdout" ]]; then
    PROBE_FLAGS+=(--donor_holdout)
fi

python scripts/probe_mimitou_perturbation_clustering.py \
    --mimitou_h5ad "$OUT_H5AD" \
    --encoder_ckpt "$CKPT_PATH" \
    --out "$OUT_JSON" \
    --device cuda \
    "${PROBE_FLAGS[@]}"

# --- Verdict tail ---
echo
echo "==========================================================="
echo "[mimitou_probe $SLURM_JOB_ID] verdict"
echo "==========================================================="
python -c "
import json
r = json.load(open('$OUT_JSON'))
v = r['verdict']
print(f'  Synergy 4-class accuracy:  {v[\"synergy_4class_accuracy\"]:.4f}')
print(f'  Chance baseline:           {v[\"chance_baseline\"]:.4f}')
rp = r['baselines']['random_projection_full']['overall']
tf = r['baselines']['raw_tfidf_full']['overall']
print(f'  Random projection (full):  {rp:.4f}')
print(f'  Raw TF-IDF (full):         {tf:.4f}')
print(f'  Decision:                  {v[\"decision\"]}')
print(f'  Action: {v[\"action\"]}')
if 'sanity_warning' in v:
    print(f'  ⚠ {v[\"sanity_warning\"]}')
"
echo "==========================================================="
echo "[mimitou_probe $SLURM_JOB_ID] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished"
echo "[mimitou_probe $SLURM_JOB_ID] result: $OUT_JSON"
