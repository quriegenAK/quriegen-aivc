# BSC MareNostrum5 — AIVC Pretrain Launch Reference

Validated 2026-04-30 on H100 node `as01r1b10`, account `ehpc748`, user `quri020505`.

Organized by location. Each step shows exact commands. Run in order on first launch; later runs only need the steps marked **(repeat)**.

---

## Cluster topology

| Node class | Hostname | Internet | Python | Role |
|---|---|---|---|---|
| Mac | local | yes | any | source of truth; build wheels here |
| Transfer node | `transfer1.bsc.es` | restricted | n/a | rsync/scp endpoint only — no pip, no python, no training |
| Login node (standard) | `alogin1.bsc.es` | none | 3.9 | salloc, sbatch, squeue, tail logs |
| Login node (internet) | `glogin4.bsc.es` / `alogin4.bsc.es` | yes (BSC VPN) | 3.9 | optional outbound for one-off downloads |
| Compute node (interactive) | via `salloc` | none | 3.11 (after modules) | pip install from wheels, smoke tests |
| Compute node (batch) | via `sbatch` | none | 3.11 (after modules) | production training |

---

## 1. Mac (local machine)

Working directory throughout: `/Users/ashkhan/Projects/aivc_genelink`.

### 1.1 Initial rsync of code (first launch only)

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv311' \
    /Users/ashkhan/Projects/aivc_genelink/ \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/
```

### 1.2 scp of large h5ad files (one at a time; rsync is unreliable on >1GB)

```bash
scp /Users/ashkhan/Projects/aivc_genelink/data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/data/phase6_5g_2/dogma_h5ads/

scp /Users/ashkhan/Projects/aivc_genelink/data/phase6_5g_2/dogma_h5ads/dogma_dig_union.h5ad \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/data/phase6_5g_2/dogma_h5ads/

scp /Users/ashkhan/Projects/aivc_genelink/data/calderon2019/calderon_atac_hg38.h5ad \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/data/calderon2019/

scp /Users/ashkhan/Projects/aivc_genelink/data/calderon2019/calderon_to_dogma_lll_M.npz \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/data/calderon2019/
```

### 1.3 pip download of wheels for offline install (first launch only)

```bash
mkdir -p /tmp/aivc_wheels
pip download \
    --dest /tmp/aivc_wheels \
    --python-version 3.11 \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --no-deps \
    -r /Users/ashkhan/Projects/aivc_genelink/requirements.txt
```

### 1.4 rsync of wheels to transfer node (first launch only)

```bash
rsync -avz /tmp/aivc_wheels/ \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_wheels/
```

### 1.5 Delta rsync of scripts after code changes (repeat)

```bash
rsync -avz --exclude='__pycache__' --exclude='.git' \
    /Users/ashkhan/Projects/aivc_genelink/scripts/ \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/scripts/

rsync -avz --exclude='__pycache__' \
    /Users/ashkhan/Projects/aivc_genelink/configs/ \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/configs/
```

---

## 2. Transfer node (`transfer1.bsc.es`)

**Role: file movement only. No pip, no python, no training.**

### 2.1 SSH

```bash
ssh quri020505@transfer1.bsc.es
```

### 2.2 Change password (first login only)

```bash
passwd
```

### 2.3 Verify scratch contents (optional sanity check)

```bash
du -sh /gpfs/scratch/ehpc748/quri020505/aivc_genelink/
ls /gpfs/scratch/ehpc748/quri020505/aivc_genelink/data/phase6_5g_2/dogma_h5ads/
```

---

## 3. Login node (`alogin1.bsc.es`)

**Role: SLURM control plane. Modules NOT available here. System Python is 3.9 (cannot install project deps here).**

### 3.1 SSH

```bash
ssh quri020505@alogin1.bsc.es
```

### 3.2 Move to project root

```bash
cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink
```

### 3.3 salloc for interactive compute session (first launch only — for pip install)

```bash
salloc --account=ehpc748 --qos=acc_ehpc --partition=acc \
       --gres=gpu:1 --cpus-per-task=20 --time=01:00:00
```

### 3.4 Set W&B credentials (repeat)

```bash
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=quriegen
```

### 3.5 Submit production training job (repeat)

```bash
sbatch scripts/submit_pretrain.slurm
```

### 3.6 Monitor queue (repeat)

```bash
squeue -u quri020505
```

### 3.7 Tail job logs (repeat)

```bash
JOBID=<from_sbatch_output>
tail -f /gpfs/scratch/ehpc748/quri020505/logs/pretrain_${JOBID}.out
tail -f /gpfs/scratch/ehpc748/quri020505/logs/smoke_${JOBID}.log
tail -f /gpfs/scratch/ehpc748/quri020505/logs/pretrain_${JOBID}.log
```

### 3.8 Cancel a job

```bash
scancel <JOBID>
```

---

## 4. Compute node (interactive, via `salloc`)

**Role: pip install from wheels, smoke tests, manual debug. Python 3.11 available only after modules are loaded.**

### 4.1 Move to project

```bash
cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink
```

### 4.2 Module load sequence (full stack — ORDER MATTERS)

```bash
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
```

### 4.3 PYTHONPATH for user-level wheels

```bash
export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"
```

### 4.4 Verify Python + GPU

```bash
python --version
nvidia-smi
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 4.5 pip install from local wheels (first launch only)

```bash
pip install --user --no-deps --no-index \
    --find-links=/gpfs/scratch/ehpc748/quri020505/aivc_wheels/ \
    -r /gpfs/scratch/ehpc748/quri020505/aivc_genelink/requirements.txt

pip install --user --no-deps -e /gpfs/scratch/ehpc748/quri020505/aivc_genelink
```

### 4.6 Test import (verifies wheels installed correctly)

```bash
python -c "import aivc, anndata, scipy, pyranges, sklearn, pyliftover, wandb, h5py; print('deps OK')"
```

### 4.7 Smoke test (5 steps; ~2-5 min)

```bash
python scripts/pretrain_multiome.py \
    --config configs/dogma_pretrain.yaml \
    --arm joint \
    --max-steps 5 \
    --no_wandb \
    --checkpoint_dir /gpfs/scratch/ehpc748/quri020505/checkpoints/manual_smoke
```

### 4.8 Verify smoke checkpoint

```bash
ls -lh /gpfs/scratch/ehpc748/quri020505/checkpoints/manual_smoke/
python -c "
import torch
ckpt = torch.load('/gpfs/scratch/ehpc748/quri020505/checkpoints/manual_smoke/pretrain_encoders.pt', map_location='cpu', weights_only=False)
print(f'global_step: {ckpt[\"resume_state\"][\"global_step\"]}')
print(f'config.arm: {ckpt[\"config\"][\"arm\"]}')
print(f'n_lysis_categories: {ckpt[\"config\"][\"n_lysis_categories\"]}')
"
```

### 4.9 Exit interactive session

```bash
exit
```

---

## 5. Key facts (pinned)

- **No internet on standard login nodes (`alogin1`)** — pip install will hang or fail. Use `glogin4` / `alogin4` (BSC VPN required) for one-off outbound, or compute nodes via `salloc`.
- **All pip installs must run on compute node with Python 3.11**, not login node (Python 3.9). Mismatch installs to `~/.local/lib/python3.9/site-packages` and the SLURM job won't see them.
- **h5ad files must be written with `compression=None`** for cluster HDF5 compatibility. Wheel-installed h5py + cluster system HDF5 conflict on attribute decoding for compressed sparse groups. Re-run on Mac with `adata.write_h5ad(path, compression=None)` before scp.
- **BSC ACC partition requires 20 CPUs per GPU** — `#SBATCH --cpus-per-task=20` for `--gres=gpu:1`. Lower values are rejected.
- **User-installed packages live at `~/.local/lib/python3.11/site-packages`**. Always export `PYTHONPATH` to include this path before training (the SLURM script and `gpu_bootstrap.sh` do this).
- **`WANDB_API_KEY` must be exported in the shell before `sbatch`** — SLURM inherits the env. Without it, the `:?` parameter expansion in `submit_pretrain.slurm` fails immediately.
- **Module load order is load-bearing.** `gcc → mkl → impi → hdf5 → openblas → nccl → nvidia-hpc-sdk → cudnn → tensorrt → python → pytorch`. Loading out of order produces silent ABI mismatches that surface as random segfaults during training.
- **CUDA stack is 11.8 (via `nvidia-hpc-sdk/23.11-cuda11.8`)**, not 12.x. `pytorch/2.4.0` module is built against CUDA 11.8.
- **`pytorch/2.4.0` from module replaces `torch==2.2.2` from `requirements.txt`.** Use `pip install --no-deps` to skip torch reinstall on the compute node.
- **scratch quota at `/gpfs/scratch/ehpc748/quri020505/`** — verify with `du -sh` before large transfers.

---

## 6. End-to-end first launch (compressed)

```bash
# Mac (~30 min for full sync)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv311' \
    /Users/ashkhan/Projects/aivc_genelink/ quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/
scp <each h5ad> quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/data/...
pip download --dest /tmp/aivc_wheels --python-version 3.11 --platform manylinux2014_x86_64 \
    --only-binary=:all: --no-deps -r requirements.txt
rsync -avz /tmp/aivc_wheels/ quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_wheels/

# Login node
ssh quri020505@alogin1.bsc.es
cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink
salloc --account=ehpc748 --qos=acc_ehpc --partition=acc --gres=gpu:1 --cpus-per-task=20 --time=01:00:00

# Compute node (after salloc)
module load gcc/11.4.0 mkl/2024.0 impi/2021.11 hdf5/1.14.1-2-gcc openblas/0.3.27-gcc \
            nccl/2.19.4 nvidia-hpc-sdk/23.11-cuda11.8 cudnn/9.0.0-cuda11 \
            tensorrt/10.0.0-cuda11 python/3.11.5-gcc pytorch/2.4.0
export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"
pip install --user --no-deps --no-index --find-links=/gpfs/scratch/ehpc748/quri020505/aivc_wheels/ \
    -r /gpfs/scratch/ehpc748/quri020505/aivc_genelink/requirements.txt
pip install --user --no-deps -e /gpfs/scratch/ehpc748/quri020505/aivc_genelink
python -c "import aivc, anndata, scipy, pyranges, sklearn, pyliftover, wandb; print('deps OK')"
exit  # leave compute node, back to login

# Login node
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=quriegen
sbatch scripts/submit_pretrain.slurm
squeue -u quri020505
```

---

## 7. Subsequent launches (after first install)

```bash
# Mac
rsync -avz --exclude='__pycache__' /Users/ashkhan/Projects/aivc_genelink/scripts/ \
    quri020505@transfer1.bsc.es:/gpfs/scratch/ehpc748/quri020505/aivc_genelink/scripts/

# Login node
ssh quri020505@alogin1.bsc.es
cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink
export WANDB_API_KEY=<your_key>
sbatch scripts/submit_pretrain.slurm
squeue -u quri020505
tail -f /gpfs/scratch/ehpc748/quri020505/logs/pretrain_<JOBID>.out
```
