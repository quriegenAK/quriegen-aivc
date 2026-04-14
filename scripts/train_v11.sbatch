#!/bin/bash
#SBATCH --job-name=aivc-v11-sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-35%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
# ============================================================================
# train_v11.sbatch — MareNostrum5 SLURM array launcher for the AIVC v1.1 sweep
#
# Target:    MareNostrum5 Accelerated (4x H100 64GB per node)
# Project:   EHPC-AIF-2026PG01-323 (EuroHPC AI Factory Playground)
# Budget:    5000 GPU-hours / 1250 node-hours (2026-04-08 -> 2026-07-08)
# Sweep:     36 configs (4 lfc_betas x 3 Ks x 3 lambda_l1s)
# Array:     0..35, max 8 concurrent (throttle so other jobs can run)
#
# FOOTPRINT PER ARRAY TASK:
#   * 1 GPU, 20 CPUs, 80 GB RAM, 8h walltime, signal USR1 120s before kill
#   * Expected runtime per config: ~45-60 min on H100 (est, unverified)
#   * Full sweep wall-clock: ~5-7h with 8 concurrent
#
# SLURM FEATURES USED:
#   --array=0-35%N      : fan out configs, throttle concurrency
#   --signal=B:USR1@120 : Bash trap; script snapshots + exits 124 120s early
#   Array task id       : $SLURM_ARRAY_TASK_ID maps to sweep index 0..35
#
# REQUEUE ON WALLTIME:
#   Exit code 124 from train_hpc.py (SIGTERM/USR1 path) -> requeue same task
#   via `scontrol requeuehold` from the trap below. Loop count capped at 3.
#
# CONFIGURE BEFORE FIRST RUN:
#   1. PARTITION    — confirm with `sinfo -s` (likely acc_bscls or similar)
#   2. QOS/ACCOUNT  — confirm with `sacctmgr show user -s $USER`
#   3. PROJECT_DIR  — set to /gpfs/projects/<groupid>/aivc after first login
#   4. SCRATCH_DIR  — set to /gpfs/scratch/<groupid>/<user>/aivc
#   5. CONTAINER    — absolute path to aivc-train.sif built via containers/aivc-train.def
# ============================================================================

set -euo pipefail

# ─── Partition/QOS/account (UNCOMMENT + FILL after `sinfo -s` confirms names)
# #SBATCH --partition=acc_bscls
# #SBATCH --qos=acc_bscls
# #SBATCH --account=<your_account>

# ─── Paths (override via env vars if needed) ────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/gpfs/projects/ehpc323/aivc}"          # code + containers
SCRATCH_DIR="${SCRATCH_DIR:-/gpfs/scratch/ehpc323/aivc}"            # data + checkpoints
CONTAINER="${CONTAINER:-${PROJECT_DIR}/containers/aivc-train.sif}"
CODE_DIR="${PROJECT_DIR}/quriegen-aivc"
DATA_DIR="${SCRATCH_DIR}/data"
CKPT_DIR="${SCRATCH_DIR}/checkpoints/v1.1"
OUT_DIR="${SCRATCH_DIR}/outputs/v1.1"
MLRUNS_DIR="${SCRATCH_DIR}/mlruns"
LOG_DIR="${CODE_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${OUT_DIR}" "${MLRUNS_DIR}" "${LOG_DIR}"

SWEEP_IDX="${SLURM_ARRAY_TASK_ID}"
TAG="sweep_${SWEEP_IDX}"
echo "============================================================"
echo "  AIVC v1.1 sweep task ${SWEEP_IDX}"
echo "  Host:        $(hostname)"
echo "  Node:        ${SLURM_NODELIST:-unknown}"
echo "  GPU(s):      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Walltime:    ${SLURM_TIME_FORMAT:-unknown}"
echo "  Container:   ${CONTAINER}"
echo "  Code:        ${CODE_DIR}"
echo "  Data:        ${DATA_DIR}"
echo "  Checkpoints: ${CKPT_DIR}"
echo "============================================================"

# ─── Module environment (MareNostrum5 specifics) ────────────────────────────
# Apptainer is available as a module on MN5. CUDA itself is baked into the
# Apptainer image so no `module load cuda` is required.
module purge || true
module load apptainer || module load singularity || true

# ─── Preflight checks ───────────────────────────────────────────────────────
if [ ! -f "${CONTAINER}" ]; then
    echo "[FATAL] Container not found: ${CONTAINER}"
    echo "        Build it with:"
    echo "        apptainer build ${CONTAINER} ${CODE_DIR}/containers/aivc-train.def"
    exit 2
fi
if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
    echo "[FATAL] Data dir empty or missing: ${DATA_DIR}"
    echo "        Stage it from login node with scripts/stage_data.sh"
    exit 3
fi

# ─── SIGTERM / USR1 trap — re-queue this array task ─────────────────────────
cleanup_and_requeue() {
    echo "[AIVC] Caught USR1 — python should snapshot and exit 124."
    # The trap fires in bash; the python process will see SIGTERM/USR1 via
    # its own signal handler (install_signal_handlers) and exit 124 cleanly.
    wait  # wait for python to finish its snapshot
    RC=$?
    if [ "$RC" = "124" ]; then
        echo "[AIVC] Task snapshotted cleanly. Requeuing array task ${SWEEP_IDX}."
        scontrol requeue "${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" || true
    fi
    exit $RC
}
trap 'cleanup_and_requeue' USR1

# ─── Bind mounts for Apptainer ──────────────────────────────────────────────
BIND_ARGS=(
    --bind "${CODE_DIR}:/workspace"
    --bind "${DATA_DIR}:/data"
    --bind "${CKPT_DIR}:/checkpoints"
    --bind "${OUT_DIR}:/outputs"
    --bind "${MLRUNS_DIR}:/mlruns"
)

# ─── Run (background + wait so the trap works) ──────────────────────────────
# `--resume` is always set — first run is a no-op, subsequent re-queues continue.
apptainer exec --nv \
    "${BIND_ARGS[@]}" \
    "${CONTAINER}" \
    python /workspace/scripts/train_hpc.py \
        --data-dir /data \
        --output-dir /outputs \
        --checkpoint-dir /checkpoints \
        --mlflow-tracking-uri "file:/mlruns" \
        --experiment-name "aivc_v11_mn5" \
        --sweep-index "${SWEEP_IDX}" \
        --epochs 200 \
        --batch-size 8 \
        --checkpoint-every 5 \
        --resume &

PY_PID=$!
wait "${PY_PID}"
RC=$?
echo "[AIVC] python exit code: ${RC}"
exit "${RC}"
