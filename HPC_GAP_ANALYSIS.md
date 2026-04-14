# AIVC HPC GPU Readiness — Structured Gap Analysis

**Date:** 2026-04-10
**Author:** Review for Ash Khan
**Scope:** End-to-end path from 10x Multiome ingest → GeneLink GPU training → AIVC inference, evaluated against "cold HPC node, shared POSIX FS, Singularity only, SLURM scheduler, no AWS egress" standard.

---

## 0. Investigation Access Report (Read First)

The review standard is: flag anything I could not access, do not assume contents. Four of six repositories and the Confluence infrastructure docs were unreachable from this environment. These are investigation gaps in their own right.

| Target | Status | Notes |
|---|---|---|
| `quriegenAK/quriegen-aivc` | ACCESSED (public) | Full clone; identical to the local `aivc_genelink` workspace (same git remote). Treated as one codebase. |
| Local `aivc_genelink` workspace | ACCESSED | Deep read. This is the GeneLink training + AIVC preprocessing + FastAPI code in one repo. |
| `quriegen/nf-cellranger-multi` | NOT ACCESSIBLE (HTTP 404) | Private or missing under `quriegen` org. No public mirror found. |
| `quriegen/nf-star-align` | NOT ACCESSIBLE (HTTP 404) | Only related public repo is `quriegen/nf-star-rnaseq-tutorial-wout`, which is a 20‑line stub with hardcoded `/home/ec2-user/...` paths and no SLURM profile. |
| `quriegen/bioinf-containers` | NOT ACCESSIBLE (HTTP 404) | No Dockerfiles, no Singularity defs, no image manifest visible. **Cannot verify pinning, Singularity builds, or registry accessibility.** |
| `quriegen/aivc-process-10x-atac-rna` | NOT ACCESSIBLE (HTTP 404) | Internal analog exists: `aivc/preprocessing/atac_rna_pipeline.py` in the local repo. Treated as the authoritative preprocessing code unless told otherwise. |
| Confluence `quriegen.atlassian.net` AWS infra docs | NOT ACCESSIBLE | `quriegen.atlassian.net` is not on the network allowlist (HTTP 403 at proxy). Content of S3 bucket design, Nextflow execution config, and pipeline run instructions was not verified. |

**This is a silent risk in itself.** Any claims below about nf-cellranger-multi, nf-star-align, bioinf-containers, or aivc-process-10x-atac-rna are inferences from the code I *can* see (preprocessing pipeline module, ambient-RNA QC, setup shell scripts). Any gap analysis of those repos must be re-run with direct read access before the HPC run.

---

## 1. Executive Summary

HPC readiness verdict: **NOT READY. Do not submit a training job yet.**

- **Hard blockers:** 8. At least two of them (no Nextflow HPC profile, no SLURM launcher for `train_v11.py`) make it impossible to run *anything* on the cluster in its current state. The sweep cannot survive wall-time limits because there is no checkpoint resume logic. The ATAC pipeline crashes at Step 6 on real data because `tf_motif_scanner.py` is a stub (`NotImplementedError`).
- **Soft gaps:** 9. Mostly reproducibility, version pinning, and config hygiene.
- **Silent risk items:** 6. The dangerous ones: (a) the preprocessing pipeline emits `.h5mu` (MuData) while `train_v11.py` loads `.h5ad` with a pseudo-bulk PBMC layout — the multi-omics data contract has never been exercised end-to-end; (b) the inference API loads a checkpoint path that no training run currently produces at the canonical location; (c) no reference genome / JASPAR / barcode whitelist version is pinned anywhere visible.
- **Estimated effort before first usable HPC training run:** 4–6 engineering days, assuming the private `nf-*` and `bioinf-containers` repos already exist in reasonable shape. If those are empty or stub-quality, add another 3–5 days to productionise them for HPC.

---

## 2. Hard Blockers — must fix before any HPC job can run

### BLOCK-1 — No Nextflow HPC profile exists in reviewable code
- **Location:** Entire repo: zero `*.nf`, `nextflow.config`, `conf/*.config`, or `profiles` files found (`find /Users/ashkhan/Projects/aivc_genelink -name "*.nf" -o -name "nextflow.config"` returns nothing). The private `nf-cellranger-multi` / `nf-star-align` repos could not be read. The one public analog (`nf-star-rnaseq-tutorial-wout/main.nf`) hardcodes `params.fastq_dir = "/home/ec2-user/fastq"` and `params.index = "/home/ec2-user/genome_index"`, has no SLURM profile, and does not even define the `ALIGN_READS` process.
- **Finding:** There is no evidence of a Nextflow SLURM executor profile, process-level resource directives, or a queue mapping.
- **Impact:** `nextflow run ...` on an HPC login node will either default to `local` executor (stampedes the login node) or fail out immediately with no matching profile.
- **Fix:** Add an `hpc` profile to `nextflow.config` in every pipeline repo with: `executor = 'slurm'`, `queue = '<cluster-queue>'`, per-process `cpus`/`memory`/`time` directives, `singularity.enabled = true`, `singularity.cacheDir = '/shared/sif_cache'`, and `workDir = '/shared/scratch/<user>/nf-work'`. Pin executor version. Remove all `ec2-user` paths.
- **Effort:** 6–10 hours per pipeline once the private repos are accessible.

### BLOCK-2 — No SLURM job script for GeneLink training exists
- **Location:** `train_v11.py`, `scripts/run_stage2_unlock.sh`. Repo-wide grep for `sbatch`, `#SBATCH`, `srun` returns zero hits.
- **Finding:** `train_v11.py` is designed to be run as `python train_v11.py` from the project root. It expects cwd = repo root (see all the `"data/..."` relative paths on lines 90, 100–103, 133–135, 144, 462). There is no launcher that binds it to SLURM, sets `SINGULARITY_BIND`, activates the correct CUDA module, or configures the MLflow tracking URI.
- **Impact:** No way to queue the training. Even if you `srun` it manually, data loading will fail on any mount that is not the repo root.
- **Fix:** Write `scripts/train_v11.sbatch` (or a Nextflow process in a dedicated `nf-aivc-train` pipeline) that: `#SBATCH --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=24:00:00`, binds the shared data mount to `/workspace/data`, sets `MLFLOW_TRACKING_URI`, changes directory to the repo, and runs `singularity exec --nv $SIF python train_v11.py`. Pin everything.
- **Effort:** 4 hours.

### BLOCK-3 — No checkpoint resume logic in `train_v11.py`
- **Location:** `train_v11.py` lines 334–537 (training loop) and 542 (`model.load_state_dict(torch.load(save_path, ...))` — used only to reload best at the END of a config for test-set eval).
- **Finding:** The only place a checkpoint is saved is `torch.save(model.state_dict(), save_path)` inside the epoch loop when `val_r > best_val_r` (lines 459–462). There is **no optimizer state, no scheduler state, no epoch counter, no sweep-index saved**. There is no `if os.path.exists(save_path): load ...` branch to resume a killed run. The full sweep is 36 configs × up to 200 epochs. Any wall-time interrupt restarts from scratch.
- **Impact:** On an HPC queue with any wall-time cap (typical: 24h, 48h, 72h), a long sweep will be killed and lose all progress. Wasted GPU hours compound with each retry.
- **Fix:** Add `--resume <dir>` CLI. Persist `{model_state, optimizer_state, scheduler_state, epoch, sweep_index, rng_state, best_val_r}` to `checkpoints/<config_tag>/epoch_XXX.pt` every N epochs AND on SIGTERM (install a `signal.signal(signal.SIGTERM, handler)` to snapshot before SLURM kills the job). On startup, scan the sweep directory and skip completed configs.
- **Effort:** 6–8 hours.

### BLOCK-4 — `train_v11.py` is not parameterisable (no CLI, no config file)
- **Location:** `train_v11.py` top of file and `SWEEP` dict at line 644.
- **Finding:** All paths (`data/kang2018_pbmc_fixed.h5ad`, `data/X_ctrl_ot.npy`, `data/edge_list_fixed.csv`, `models/v1.1/`, `results/`), the sweep grid, seed, epochs, and batch size are literal constants in the file body. There is no `argparse`, no Hydra, no env-var fallback for data paths. Compare with `run_stage2_unlock.sh` which explicitly notes the gap: *"train_v11.py does not yet accept --stage and --ifng-data flags"* (run_stage2_unlock.sh around the Step 2 block).
- **Impact:** On HPC, where the data lives on a shared mount (e.g. `/scratch/aivc/data/`), the script will `FileNotFoundError` unless you `cd` to a repo copy that has the data symlinked to `./data/`. You cannot run multiple concurrent sweeps on the same node or inject different data directories for Stage 2. The sweep is not parallelisable across array jobs.
- **Fix:** Add `argparse` (or Hydra) with: `--data-dir`, `--output-dir`, `--checkpoint-dir`, `--adata`, `--ot-ctrl`, `--ot-stim`, `--edges`, `--epochs`, `--batch-size`, `--lfc-beta`, `--neumann-k`, `--lambda-l1`, `--resume`. Drop `SWEEP` dict; instead let the outer SLURM array job fan out one config per task (`#SBATCH --array=0-35`, each task trains one `(beta, K, l1)` combination).
- **Effort:** 4–6 hours.

### BLOCK-5 — ATAC preprocessing pipeline crashes at Step 6 on real data
- **Location:** `aivc/preprocessing/atac_rna_pipeline.py` lines 367–407 (Step 6) → `aivc/preprocessing/tf_motif_scanner.py` (stub).
- **Finding:** README acknowledges this explicitly: *"KNOWN STUB: Step 6 (compute_chromvar_scores in tf_motif_scanner.py) raises NotImplementedError. The ATAC pipeline will crash at Step 6 when run on real 10x Multiome data."* The open-bugs table in `README.md` confirms: *"tf_motif_scanner.py NotImplementedError — OPEN — implement chromVAR-equivalent (JASPAR 2024)"*. The exported `.h5mu` contract requires `mdata['atac'].obsm['chromvar_scores']` (line 419), which cannot be produced.
- **Impact:** The moment `nf-cellranger-multi` or any other pipeline feeds real filtered matrices into `aivc/preprocessing/atac_rna_pipeline.py`, Step 6 raises, the pipeline dies, no `.h5mu` is produced, and therefore no multi-modal training data reaches the GPU job.
- **Fix:** Implement `compute_chromvar_scores(atac, jaspar_db=...)` with a chromVAR-equivalent (pycisTopic + JASPAR2024 PWMs, or MotifMatch + background correction). Pin the JASPAR version in a container and document the expected genome build. Add a real-data regression test on a 500-cell public 10x Multiome sample.
- **Effort:** 2–3 days (this is a real implementation, not a config change).

### BLOCK-6 — GeneLink training assumes AnnData/npy inputs, not the MuData that preprocessing outputs
- **Location:** `train_v11.py` lines 90–141 vs. `aivc/preprocessing/atac_rna_pipeline.py` lines 412–457 (Step 7 exports `.h5mu`).
- **Finding:** `train_v11.py` reads `data/kang2018_pbmc_fixed.h5ad` (AnnData, bulk PBMC) and pre-computed OT numpy arrays `data/X_ctrl_ot.npy`, `data/X_stim_ot.npy`, `data/cell_type_ot.npy`, `data/donor_ot.npy`, `data/edge_list_fixed.csv`. The ATAC pipeline produces `.h5mu` with modalities `rna`/`atac` and obs fields `cell_type`, `donor_id`, `condition`, `atac_weight`, plus `mdata['atac'].obsm['chromvar_scores']` and `mdata.uns['peak_gene_links']`. There is **no loader that bridges MuData → the numpy/AnnData tensors the training loop consumes**, and no script that produces OT pairs from multiome data. The only multi-omics entry points are `aivc/skills/multimodal_smoke_test.py` and tests with mock mu-data.
- **Impact:** Even when the ATAC pipeline is fixed (BLOCK-5), the output is not consumable by `train_v11.py`. There is no end-to-end path from 10x Multiome → GeneLink GPU training. The HPC run will train *only* on the existing Kang2018 bulk matrix, which is already trained and saved — so you would be re-running v1.0/v1.1 on the same inputs, not new multi-omics data.
- **Fix:** Either (a) add a `--mudata` code path in `train_v11.py` that extracts the normalised RNA layer, per-cell donor/cell_type/condition obs, and constructs OT pairs on the fly, or (b) write a `scripts/mudata_to_training_tensors.py` that converts `.h5mu` to the `X_ctrl_ot.npy` / `X_stim_ot.npy` / `edge_list.csv` contract. Option (b) is the lower-risk path. Either way, add a schema-validation step (expected obs columns, dtypes, gene list intersection with STRING PPI).
- **Effort:** 1.5–2 days, because OT pair generation is non-trivial on large multiome datasets.

### BLOCK-7 — Container strategy assumes Docker Hub pulls and no Singularity cache
- **Location:** `setup_gpu_instance.sh` lines 33–99 (direct pip install flow, no container use); `bioinf-containers` repo (unreachable).
- **Finding:** The only bootstrapped GPU environment is `setup_gpu_instance.sh`, which assumes an Ubuntu 22.04 Deep Learning AMI, a Python venv named `aivc_env`, `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118`, and zero container use. On HPC: (a) pip install from login nodes is typically blocked, (b) the CUDA 11.8 pinned here will conflict with the cluster's `module load cuda` versions if they are ≥12.x, (c) there is no Singularity `.sif` referenced anywhere, (d) I cannot verify whether `bioinf-containers` even has Singularity `%post` sections because I cannot read that repo.
- **Impact:** Containers must be the unit of execution on HPC. Without a pre-built `.sif` pulled onto shared storage, the Nextflow executor will either (i) fall back to Docker (not available), (ii) try to pull from Docker Hub (rate limits + proxy blocks), or (iii) bootstrap from scratch inside the compute node (slow, unreliable, often blocked).
- **Fix:** In `bioinf-containers`, produce Singularity definition files for each tool (`cellranger-9.x.def`, `star-2.7.x.def`, `aivc-train-cu121.def`, `aivc-preprocess.def`). Build on a trusted host, push `.sif` files to `/shared/sif_cache/` on the HPC cluster. Pin every image with a digest, not a tag. In `nextflow.config` set `singularity.cacheDir`, `singularity.autoMounts = true`, `singularity.runOptions = '--nv'`. Forbid `:latest` tags by grep-fail in CI.
- **Effort:** 1–2 days if `bioinf-containers` already has Dockerfiles to adapt; 3+ days if building from scratch.

### BLOCK-8 — No Confluence AWS infra documentation accessible
- **Location:** `https://quriegen.atlassian.net/wiki/spaces/PM/pages/223674380/...` — blocked at the network allowlist.
- **Finding:** I could not read the AWS S3 bucket design, Nextflow execution configuration, or pipeline run instructions that are defined as "ground truth for the current system state."
- **Impact:** I cannot cross-check what the current production AWS flow actually does versus what the code suggests. Any S3 bucket structure, IAM boundaries, VPC endpoint setup, or transfer mechanism may differ from what the shell scripts imply. The cloud-to-HPC migration plan MUST be anchored in the current S3 layout.
- **Fix:** Either (a) export the Confluence pages to PDF/HTML and put them in `docs/aws_infra/`, or (b) add `*.atlassian.net` to the Cowork network allowlist (Settings → Capabilities), then re-run this investigation against the live pages. Do not begin HPC migration until someone has walked through the current AWS layout side-by-side with the repo state.
- **Effort:** 30 minutes for the access fix; 2–3 hours to reconcile the docs with code.

---

## 3. Soft Gaps — should fix, not immediate blockers

### GAP-1 — `requirements.txt` is not a lockfile
- **Location:** `requirements.txt` (all 20 lines).
- **Finding:** `scipy>=1.11.0`, `pandas>=2.0.0`, `scanpy==1.11.5` (strict), `anndata>=0.10.0`, `fastapi>=0.110.0`, `POT>=0.9.0`. This is a mix of hard pins and open ranges.
- **Impact:** Two independent HPC builds may resolve different scipy/pandas versions, which can shift numerical results in tests like `test_neumann_sparsity` or change DataFrame indexing behaviour.
- **Fix:** Generate `requirements.lock` (`pip-compile` or `uv pip compile`). Rebuild the container from the lockfile. Store the lockfile alongside the `.sif` digest in the model registry.
- **Effort:** 1 hour.

### GAP-2 — `torch-geometric==2.7.0` + `torch==2.2.2 cu118` is a narrow wheel combination
- **Location:** `requirements.txt` lines 1–3, `setup_gpu_instance.sh` line 59.
- **Finding:** torch-geometric wheels for cu118 + torch 2.2.2 exist but are not the default on HPC CUDA modules (many clusters are CUDA 12.x). If the HPC node has CUDA 12, `torch-geometric` will not find its compiled extensions (`torch_scatter`, `torch_sparse`) unless they are pre-installed with a matching index URL.
- **Impact:** First `import torch_geometric` on an H100 CUDA-12 node may throw `OSError: undefined symbol` or fall back to the slow Python path (and silently lose 5–10× throughput).
- **Fix:** Either upgrade to `torch==2.4+ cu121` (matches most modern clusters) or explicitly build the torch-geometric extensions inside the container for the cluster's CUDA version. Pin `torch_scatter`, `torch_sparse`, `torch_cluster` wheel URLs.
- **Effort:** 2–4 hours, including a smoke test.

### GAP-3 — No MLflow tracking URI / artefact store for HPC
- **Location:** `train_v11.py` line 642 (`mlflow_backend = MLflowBackend()`), `aivc/memory/mlflow_backend.py` (not read, but instantiated with default args).
- **Finding:** There is no env var, CLI flag, or config file that sets `MLFLOW_TRACKING_URI`. On HPC compute nodes, the default is the local filesystem (`./mlruns/`), which means every sweep run writes to a different node-local directory and the sweep is not aggregable.
- **Impact:** Sweep results scatter across compute nodes; you cannot compare configs after the run.
- **Fix:** Define `MLFLOW_TRACKING_URI=file:///shared/aivc/mlruns` in the SLURM script OR stand up a remote MLflow tracking server with S3/Minio artefact store. Add a CLI flag to `train_v11.py` that sets it.
- **Effort:** 2 hours for file-based, 1 day for remote server.

### GAP-4 — `torch.compile(mode="reduce-overhead")` will hit graph breaks on non-H100
- **Location:** `train_v11.py` lines 336–353, and the revert check at lines 517–533.
- **Finding:** Code already gracefully reverts to eager mode if >10 graph breaks are detected in epoch 1. Good. But `reduce-overhead` mode uses CUDA graphs which require stable input shapes; the mini-bulk path has variable group counts (line 212), so graph capture will fail every epoch and the code will revert to eager.
- **Impact:** You will lose the headline `torch.compile` speedup that is implicit in the "H100 readiness" framing.
- **Fix:** Switch to `mode="default"` or `"max-autotune"` with `dynamic=True`; pad mini-bulk groups to a fixed count, or precompute them upfront.
- **Effort:** 2–3 hours to test both modes.

### GAP-5 — Seed reproducibility is broken by `torch.backends.cudnn.deterministic = True` + mixed precision + torch.compile
- **Location:** `train_v11.py` lines 54–61.
- **Finding:** Seed is set, cudnn determinism is requested, but (a) BF16 autocast is not deterministic across H100 vs A100, (b) `torch.compile` introduces non-determinism via its autotuner, (c) the edge iteration at line 147 uses DataFrame `.iterrows()` which is stable but the *result* of edge ordering depends on pandas version (GAP-1).
- **Impact:** Two runs on different GPU generations will produce different checkpoints, which undermines the "no regression from v1.0 r=0.873" selection rule.
- **Fix:** Document "deterministic within a single GPU generation" and baseline v1.1 results on the exact H100 you will use. Or accept non-determinism and use ≥3 seeds per config (blows up sweep to 108 runs → use array jobs).
- **Effort:** Documentation only, 1 hour. Re-seed experiments are part of the training run.

### GAP-6 — `api/server.py` looks for checkpoints under paths the training code does not produce at those locations
- **Location:** `api/server.py` lines 91–97 vs. `train_v11.py` line 332.
- **Finding:** API tries these paths in order: `$AIVC_CHECKPOINT`, `models/v1.1/model_v11_best.pt`, `models/v1.0/aivc_v1.0_best.pt`, `model_week3_best.pt`, `model_week3.pt`. But `train_v11.py` saves to `models/v1.1/sweep_beta0.10_K3_l10.001.pt` (derived from tag) — there is no step that promotes the best sweep config to `models/v1.1/model_v11_best.pt`.
- **Impact:** After the HPC sweep finishes, the inference API will fall back to `model_week3_best.pt` (an older v1.0 checkpoint) or "demo-untrained" mode. The headline v1.1 improvements will not be served.
- **Fix:** Add a final "promote" step in `train_v11.py` (or a post-sweep script): copy the winning config's checkpoint to `models/v1.1/model_v11_best.pt` AND write a `models/v1.1/manifest.json` with the winning hyperparameters. This is also the artefact that should be uploaded to the model registry.
- **Effort:** 1 hour.

### GAP-7 — FastAPI inference is CPU-only by default
- **Location:** `api/server.py` line 115 (`torch.load(path, map_location="cpu")`), line 132 (`model.eval()` with no `.to("cuda")`).
- **Finding:** The model is loaded to CPU and stays there. Inference requests to `/predict` and `/intervene` will run on CPU even if the API container has a GPU.
- **Impact:** Real-time perturbation prediction will be seconds per request instead of tens of milliseconds. Not an HPC blocker but important for the "AIVC platform" story.
- **Fix:** Add `DEVICE = os.getenv("AIVC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")`, `map_location=DEVICE`, then `model.to(DEVICE)`. For the `SCMEngine` wrapper at lines 165–184, move inputs to device inside `forward`.
- **Effort:** 30 minutes + GPU smoke test.

### GAP-8 — No CI/CD or container smoke test for the training stack
- **Location:** `.github/` absent; `tests/test_h100_readiness.py` runs CPU-only (line 8).
- **Finding:** `test_h100_readiness.py` is explicitly documented as "All tests run on CPU only — no GPU required." It validates device-placement patterns but cannot catch real CUDA/cuDNN/cuBLAS issues on a real H100.
- **Impact:** The first real CUDA failure will be on the HPC node with a 4+ hour queue wait.
- **Fix:** Stand up a small CI job on a CUDA-capable runner (or an HPC dev partition) that does: `nvidia-smi → python -c "import torch; print(torch.cuda.is_available())" → python train_v11.py --smoke-test --epochs 2` against a tiny synthetic dataset. This is also a requirement for GAP-6 promotion logic.
- **Effort:** 4–8 hours.

### GAP-9 — Sweep wall-clock budget is unknown
- **Location:** `train_v11.py` line 644 (36 configs), line 282 default `n_epochs=200`.
- **Finding:** There is no documentation of GPU-hours per config, per full sweep, on A100 vs H100. There is no `--max-hours` safety switch. The sweep will run until done.
- **Impact:** You may blow past the SLURM walltime budget and get killed (GAP relates to BLOCK-3).
- **Fix:** Run one small-epoch benchmark on a single GPU to project (hours_per_config × 36) + checkpointing overhead. Document it in README and set SLURM `--time=` accordingly.
- **Effort:** 3 hours benchmark + 1 hour docs.

---

## 4. Silent Risk Items — won't crash, but will produce wrong results

### RISK-1 — No reference genome version is pinned anywhere
- **Location:** Repo-wide grep for `GRCh`, `ensembl`, `refdata`, `gencode` returns zero hits. `aivc/preprocessing/atac_rna_pipeline.py` accepts a `cellranger_output_dir` but never asserts the genome build used.
- **Finding:** nf-cellranger-multi must have been run against *some* CellRanger reference (e.g. `refdata-gex-GRCh38-2024-A`) and nf-star-align against *some* STAR index, but nowhere in the accessible code is the build, Ensembl version, or GTF release recorded. The STRING PPI edge list (`data/edge_list_fixed.csv`, `9606.protein.links.v12.0.txt`) pins STRING v12.0 / taxon 9606, which is GRCh38-compatible — but no assertion links them.
- **Impact:** A CellRanger reference built on GRCh37 (or a non-matching GTF release) will produce a gene name set that does not intersect cleanly with the 3,010-gene GeneLink vocabulary. Silent failure mode: the gene intersection drops, model still trains, test r drops ~0.1, you assume it's a training bug.
- **Fix:** Add a top-of-pipeline assertion: `mdata.uns['genome_build'] == 'GRCh38.p14'`, `mdata.uns['gtf_release'] == 'ensembl_110'`. Write those keys in the CellRanger wrapper of nf-cellranger-multi. Fail fast if the gene intersection with `data/gene_names.txt` drops below a threshold (e.g. 95%).
- **Effort:** 2 hours (pipeline side) + 1 hour (training side assertion).

### RISK-2 — Gene list has been frozen at 3,010 but no versioning on it
- **Location:** `api/server.py` line 32 (`N_GENES = 3010`), `data/gene_names.txt`, `train_v11.py` line 95 (`n_genes = len(gene_names)` from the h5ad var).
- **Finding:** The inference API hardcodes 3,010 but the training script reads it from the `adata.var` at runtime. There is no assertion that the training data has exactly 3,010 genes in the exact order of `data/gene_names.txt`.
- **Impact:** If a new Kang2018 preprocessing changes gene count or order, the trained model's `perturbation_model.PerturbationPredictor(n_genes=n_genes)` will mismatch the API's `N_GENES = 3010`, and either (i) load_state_dict will crash, or (ii) state dict will load with shape mismatch on the last layer and silently produce garbage.
- **Fix:** Make `N_GENES` derived from the checkpoint manifest (GAP-6). Assert at training start: `assert gene_names == open('data/gene_names.txt').read().splitlines()`. Write `gene_names.txt` into the checkpoint folder alongside the `.pt`.
- **Effort:** 2 hours.

### RISK-3 — `edge_index.cpu()` / `edge_attr.cpu()` passed to NeumannPropagation, then `.to(device)`'d
- **Location:** `train_v11.py` lines 271–277.
- **Finding:** The edge index and attributes are already on-device (lines 156–157), but are explicitly moved back to CPU when constructing `NeumannPropagation`, then the whole module is moved back to device on line 277. This is not wrong, but it does mean every sweep config triggers an unnecessary CPU↔GPU round-trip for the STRING PPI graph.
- **Impact:** Tiny throughput hit (PPI is small), but more importantly: if `NeumannPropagation.__init__` builds a dense `W` matrix or a sparse adjacency on CPU first (the code is not reviewed here), then moves it to GPU, the peak-memory spike is charged to the host memory node, which may exceed the SLURM `--mem=` allocation on some clusters.
- **Fix:** Read `NeumannPropagation.__init__` and confirm the adjacency construction memory profile. Pre-build once outside the sweep loop and pass the pre-built module instead.
- **Effort:** 1 hour investigation + 1 hour refactor.

### RISK-4 — ATAC pipeline `window_kb` default is correct but silently configurable
- **Location:** `aivc/preprocessing/atac_rna_pipeline.py` lines 321–362.
- **Finding:** `step05_link_peaks_to_genes` default is `window_kb=300` (correct per Wout's note that 100 is a testing shortcut). But it's a function argument, and nothing forbids a caller from passing `window_kb=100`. There is only a `logger.warning(...)` at line 328. No hard fail.
- **Impact:** A future caller "just testing with 100kb to be quick" produces a `.h5mu` that silently misses ~35% of enhancer-gene links. Downstream training uses it without noticing. This is exactly the kind of silent degradation that costs weeks of debugging.
- **Fix:** Promote the warning to a `ValueError` unless `os.getenv("AIVC_ALLOW_SMALL_WINDOW") == "1"`. Write the effective `window_kb` into `mdata.uns['pipeline_params']` and assert it downstream.
- **Effort:** 30 minutes.

### RISK-5 — No pairing certificate enforcement between multiome and GeneLink training
- **Location:** `README.md` "QuRIE-seq pairing status" section; `data/pairing_certificates/quriegen_pending.json`; `aivc/data/pairing_certificate.py` (not read directly).
- **Finding:** Pairing cert is `"unknown"` pending wet-lab confirmation. Contrastive loss is gated on it. But there is no gate that prevents a fresh 10x Multiome run from being used for multi-modal training if the pairing cert is still `"unknown"`. `train_v11.py` does not check the certificate at all.
- **Impact:** A multi-modal training run on unconfirmed paired data will silently use an invalid biological assumption (same-cell co-measurement) and produce results that look valid but are not.
- **Fix:** In `train_v11.py` startup, if the input is multi-modal, require that `data/pairing_certificates/*.json` for the dataset exists with `pairing_type` ∈ {`same_cell`, `same_nucleus`}. Fail fast otherwise.
- **Effort:** 1–2 hours.

### RISK-6 — `torch.load(path, weights_only=False)` implicit in all checkpoint loads
- **Location:** `train_v11.py` line 542, `api/server.py` line 115, `aivc/skills/*.py` likely.
- **Finding:** PyTorch ≥2.4 will start defaulting `weights_only=True` and throw on arbitrary-pickle checkpoints. Current code uses the default, which means if HPC has a newer torch, load will throw; if HPC has 2.2.2, load silently uses pickle and is vulnerable to an untrusted checkpoint path (less important on HPC, but relevant if checkpoints come from AWS S3).
- **Impact:** Silent version-drift risk; minor supply-chain risk on any checkpoint pulled from an external bucket.
- **Fix:** Make all `torch.load(...)` calls explicit: `torch.load(path, map_location=DEVICE, weights_only=True)`. Verify sweep checkpoints contain only state dicts (not arbitrary objects).
- **Effort:** 1 hour.

---

## 5. Confirmed Ready (explicitly verified)

- `train_v11.py` uses BF16 autocast only on CUDA, falls back to FP32 on CPU (lines 82–83) — correct behaviour on H100/A100.
- `train_v11.py` correctly gates TF32 on H100/A100 only (lines 78–80).
- `train_v11.py` uses `optimizer.zero_grad(set_to_none=True)` (line 422) — memory-bandwidth win on H100, enforced by `tests/test_h100_readiness.py::test_set_to_none_in_zero_grad`.
- `train_v11.py` does not use `GradScaler` (enforced by `test_no_gradscaler_in_train_file`) — correct for BF16.
- `train_v11.py` has a NaN/Inf loss guard that aborts immediately with a clear message (lines 412–417). This is exactly what you want for not wasting GPU hours.
- `train_v11.py` has a GPU-allocated-memory smoke test at step 1 (`allocated > 0.1`) that catches the "model silently lives on CPU" bug (lines 424–432).
- `train_v11.py` computes Pearson r with a broadcast-safe per-sample routine that won't divide-by-zero (lines 218–231).
- Seed is set deterministically across `random`, `numpy`, and `torch` (lines 54–61).
- `tests/test_h100_readiness.py` has 19 tests covering device placement, mixed precision, train/eval mode, OT pair shuffle, and NaN guard — all CPU-runnable.
- `api/server.py` lifespan correctly loads the model once on startup and does not re-instantiate per request (lines 150–215).
- The ATAC pipeline has a mock-MuData unit test path (`aivc/preprocessing/tests/test_atac_rna_pipeline.py`), so Steps 1–5 are exercised end-to-end in tests.
- `run_workflow.py` has a `--check-gpu` subcommand, which is the right preflight on a fresh node.
- `edge_list_fixed.csv` carries `combined_score` which is used as edge weight if present (line 153) — so STRING confidence is preserved into the graph.
- Donor-based train/val/test split in `train_v11.py` (lines 171–184) prevents donor leakage, which is the most common benchmark inflation in PBMC GRN papers.

---

## 6. Reference Data Staging Plan

I cannot enumerate the S3 bucket contents because I could not access Confluence or list the bucket. The accessible-code-derived list is below; expand with whatever the nf-* repos actually reference.

| Artefact | Current / implied location | Required HPC path | Est. size | Source |
|---|---|---|---|---|
| CellRanger reference (`refdata-gex-GRCh38-*`) | Presumed S3 (unverified) | `/shared/refs/cellranger/refdata-gex-GRCh38-2024-A/` | ~15 GB | Needed by nf-cellranger-multi |
| STAR index (matching Ensembl release) | Presumed S3 | `/shared/refs/star/GRCh38_ensembl110/` | ~30 GB | Needed by nf-star-align |
| 10x barcode whitelists (GEX + ATAC) | Bundled with CellRanger typically | `/shared/refs/10x_whitelists/` | ~200 MB | nf-cellranger-multi |
| JASPAR2024 PWMs (`atac_rna_pipeline.step06`) | Not yet available — stub raises | `/shared/refs/jaspar/2024/` | ~50 MB | Needed after BLOCK-5 fix |
| Hoa et al. PBMC reference (`step03_annotate_cell_types`) | `reference_path` arg, not in repo | `/shared/refs/pbmc_hoa/pbmc_multimodal.h5seurat` | ~2 GB | atac_rna_pipeline step 3 |
| `data/kang2018_pbmc_fixed.h5ad` | `s3://quriegen-aivc-data/data/` (from setup_s3_bucket.sh line 70) | `/shared/aivc/data/kang2018_pbmc_fixed.h5ad` | 2.8 GB on-disk (local clone) | Training input |
| `data/X_{ctrl,stim}_ot.npy`, `cell_type_ot.npy`, `donor_ot.npy` | S3 bucket | `/shared/aivc/data/` | ~500 MB each | Training input |
| `data/edge_list_fixed.csv`, `9606.protein.links.v12.0.txt.gz` | Repo-bundled | `/shared/aivc/data/` | ~50 MB | STRING PPI |
| `data/norman2019.h5ad`, `data/kang2018_pbmc*.h5ad` zero-shot eval | Repo data/ | `/shared/aivc/data/eval/` | ~5 GB | Evaluation |
| Existing checkpoints (`model_week3_best.pt`, `model_perturbation*.pt`) | Repo root + S3 | `/shared/aivc/models/v1.0/` | ~10 MB | Baseline for regression checks |
| PBMC IFN-γ real dataset (Stage 2) | NOT YET DOWNLOADED | `/shared/aivc/data/pbmc_ifng_real.h5ad` | ~3 GB (est.) | Blocks Stage 2 |
| Future 10x Multiome FASTQs | Unknown — depends on wet lab | `/shared/aivc/data/fastq/<run_id>/` | TBs | Input to nf-cellranger-multi |

**Staging strategy:** One-time transfer via `aws s3 sync` run from a jump host that has both AWS credentials AND NFS mount. Never from a compute node. Verify with MD5 checksums written alongside (`aws s3api head-object` → ETag if single-part, else sidecar manifest). Make the shared directory immutable (`chmod -w`) after transfer.

**Unknowns that must be filled before staging:**
- Exact CellRanger reference version used historically (needed for RISK-1 consistency).
- Exact STAR index version (release number, sjdbOverhang, star version that built it).
- Whether the S3 bucket has KMS-encrypted objects that require extra IAM permissions.

---

## 7. HPC Configuration Checklist

Files that must be created before the first job submission. None of these exist in the current codebase; all are green-field work.

1. **`nextflow.config` HPC profile** (add to each nf-* repo)
   ```groovy
   profiles {
     hpc {
       executor { name = 'slurm'; queueSize = 50; submitRateLimit = '10/1min' }
       process {
         queue = params.slurm_queue ?: 'gpu'
         cpus = 8
         memory = '64 GB'
         time = '24h'
         errorStrategy = 'retry'; maxRetries = 2
         withLabel: 'gpu'       { clusterOptions = '--gres=gpu:1' }
         withLabel: 'cellranger'{ cpus = 16; memory = '128 GB'; time = '48h' }
         withLabel: 'star'      { cpus = 16; memory = '64 GB';  time = '12h' }
       }
       singularity { enabled = true; autoMounts = true; cacheDir = '/shared/sif_cache'; runOptions = '--nv' }
       workDir = '/shared/scratch/${USER}/nf-work'
     }
   }
   ```

2. **`scripts/train_v11.sbatch`** — SLURM launcher with `--array=0-35` for sweep, resume logic, env-var plumbing (MLFLOW_TRACKING_URI, DATA_DIR, OUTPUT_DIR), singularity exec wrapper.

3. **`scripts/stage_data.sh`** — one-time staging from S3 to `/shared/aivc/data/` with checksum verification, run from jump host.

4. **`containers/aivc-train.def`** — Singularity definition: base `nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04`, Python 3.11, install from `requirements.lock`, set `CUDA_HOME`, bake in PyTorch 2.4+ (cu121) for the cluster's CUDA version.

5. **`containers/aivc-preprocess.def`** — Singularity for ATAC/RNA preprocessing: scanpy, muon, JASPAR2024 PWMs pre-staged, chromVAR equivalent. Cannot be finalised until BLOCK-5 is resolved.

6. **`requirements.lock`** — `pip-compile` output from `requirements.txt`, committed.

7. **`checkpoints/README.md`** — describes the manifest JSON format written alongside each final checkpoint (for GAP-6 promotion logic).

8. **`docs/hpc_runbook.md`** — step-by-step: `ssh login → module load cuda/12.1 → singularity pull ... → sbatch scripts/train_v11.sbatch`. Include all module names.

9. **`docs/data_contracts.md`** — authoritative definition of the `.h5mu` → `train_v11.py` boundary: which obs columns, which layers, which uns keys, expected dtypes. This is the missing governance artefact called out by BLOCK-6.

10. **`nextflow.config` for nf-aivc-train** (new repo, or a process added to an existing one) — if you want to orchestrate the GeneLink training via Nextflow (recommended: gives you provenance, retries, and config versioning for free).

---

## 8. Ordered Remediation Plan

Follow in order. Do not skip.

**Phase 1 — Infrastructure setup (unblocks everything)**
1. Add `*.atlassian.net` to the Cowork network allowlist and re-read the AWS infra docs (BLOCK-8). Reconcile findings with this report.
2. Get direct read access to `nf-cellranger-multi`, `nf-star-align`, `bioinf-containers`, `aivc-process-10x-atac-rna`. Re-run this gap analysis against the actual code.
3. Stand up the shared HPC data directory structure (`/shared/aivc/{data,models,sif_cache,mlruns,nf-work}`). Chown / chmod as a group.
4. Audit `bioinf-containers`: every Dockerfile must have (a) pinned base image digest, (b) pinned tool versions, (c) a `.def` sibling for Singularity builds. Build `.sif` files on a trusted host. `scp` to `/shared/sif_cache/` (BLOCK-7).
5. One-time data staging from S3 (or wherever the canonical copies live) to `/shared/aivc/data/`. MD5-verify. Freeze (`chmod -w`).

**Phase 2 — Pipeline fixes (make Nextflow work on SLURM)**
6. Add `hpc` profile to every `nf-*` repo's `nextflow.config` with Singularity + SLURM directives (BLOCK-1).
7. Remove all EC2/AWS hardcoded paths from pipeline configs.
8. Implement `compute_chromvar_scores()` in `aivc/preprocessing/tf_motif_scanner.py` with JASPAR2024 PWMs + test on a real 10x Multiome sample (BLOCK-5).
9. Tighten `window_kb` default to a hard fail with env-var override (RISK-4).
10. Smoke-test one end-to-end run from FASTQ → CellRanger → ATAC pipeline → `.h5mu` on a small public multiome dataset.

**Phase 3 — Data contract fixes (make preprocessing feed training)**
11. Write `scripts/mudata_to_training_tensors.py` that converts a `.h5mu` to the numpy/AnnData layout `train_v11.py` consumes. Add schema validation (gene list intersection, required obs columns, dtypes). Write a contract doc (`docs/data_contracts.md`). (BLOCK-6)
12. Add genome-build + gene-list assertions at the top of `train_v11.py`. Fail fast on mismatch. (RISK-1, RISK-2)
13. Add pairing-certificate enforcement gate before enabling multi-modal training. (RISK-5)

**Phase 4 — Training configuration (make the GPU run survivable)**
14. Refactor `train_v11.py`: add `argparse`, remove hardcoded paths, remove the in-file `SWEEP` dict. Make each invocation train one config. (BLOCK-4)
15. Add full checkpoint state (model + optimizer + scheduler + epoch + rng + best_val_r) and `--resume` logic. Install SIGTERM handler. (BLOCK-3)
16. Generate `requirements.lock` and rebuild the training container on CUDA 12.1 with PyTorch 2.4+ (GAP-1, GAP-2).
17. Write `scripts/train_v11.sbatch` with `--array=0-35`, Singularity exec, MLflow env, resume-on-restart. (BLOCK-2)
18. Add checkpoint-promotion logic at end of sweep so `models/v1.1/model_v11_best.pt` is populated and the API can find it. (GAP-6)
19. Set `AIVC_DEVICE` env in the API container and move the model to GPU. (GAP-7)
20. Explicit `weights_only=True` on all `torch.load` calls. (RISK-6)

**Phase 5 — Validation run**
21. Run a **1-config 5-epoch smoke-test** inside the HPC container on a single GPU using the real dataset. Assert: CUDA used, memory > 0.1 GB, no NaN, val_r trending up, checkpoint file written, resume-from-checkpoint produces identical val_r on epoch 6.
22. Run the **full 36-config sweep** as a SLURM array job. Each task writes its own checkpoint. MLflow aggregates.
23. Run the **promotion step**: pick winning config under the baseline-r ≥ 0.863 gate, copy to `models/v1.1/model_v11_best.pt`, write manifest.
24. Re-run `api/server.py` against the promoted checkpoint. Hit `/predict` and `/model/info` and confirm version = `v1.1`.
25. Run `evaluate_zero_shot.py` on Norman 2019 and Kang 2018 test set. Confirm no regression below 0.863.

**Do not run Phase 5 until all of Phase 1–4 is green.**

---

## 9. Open Questions Requiring Human Decision

These are things I could not resolve from the code alone. You need to answer each before HPC cutover.

1. **Confluence access** — can `quriegen.atlassian.net` be added to the Cowork egress allowlist (Settings → Capabilities → Network), or do you want me to work from a local export of those pages instead? Without the AWS infra docs, any S3 structure I assume is a guess.
2. **Private repo access** — are `quriegen/nf-cellranger-multi`, `nf-star-align`, `bioinf-containers`, `aivc-process-10x-atac-rna` actually at those paths (they 404 for me), or are they under a different org? The only publicly visible `quriegen` repo is `nf-star-rnaseq-tutorial-wout` and the only `quriegenAK` repos are `quriegen-aivc` and `varentis-next`. If the private repos exist elsewhere, I need the correct org/name.
3. **HPC cluster specifics** — which SLURM queue/partition gets GPU allocations? What CUDA module version (`module load cuda/X.Y`)? Which GPU generation (H100 vs A100) and how many per node? Wall-time cap? Filesystem type (NFS vs Lustre vs GPFS) — affects `workDir` choice. None of these are derivable from the repo.
4. **CellRanger license and version** — what version of CellRanger will be run, and is the 10x Genomics EULA already accepted on the cluster? The container build strategy depends on this.
5. **Reference genome build of historical runs** — which GRCh38 patch, which Ensembl release, which GTF? I need to bake this into the top-of-pipeline assertion (RISK-1).
6. **MLflow deployment** — file-based (`file:///shared/aivc/mlruns`) or remote tracking server? If remote, where, and does the HPC compute node have outbound network access to it?
7. **Sweep time budget** — how much GPU-time are you willing to spend on the full 36-config sweep? This determines whether we cut the grid (e.g. fix `lambda_l1=0.01` per the README note at line 39) down to 12 configs.
8. **Model registry** — is there an existing artefact store (S3 bucket, MLflow artefact server, W&B, plain NFS) where the winning checkpoint should land? The API's `$AIVC_CHECKPOINT` env var lets me point it anywhere, but I need a canonical location.
9. **Pairing certificate timeline** — when does Thiago confirm the QuRIE-seq pairing type? Until this is done, multi-modal training is blocked by policy (RISK-5), so the HPC run is effectively *only* the RNA-only GeneLink sweep on Kang 2018 + Stage 2 PBMC IFN-γ. Worth confirming that is what you actually want to run first.
10. **Stage 2 real IFN-γ dataset** — has the PBMC IFN-γ dataset been downloaded and validated yet? `scripts/run_stage2_unlock.sh` explicitly notes the training wiring (`--stage 2 --ifng-data`) does not exist in `train_v11.py` yet. Stage 2 is currently a shell stub.

---

## Key Risks

- **Pipeline-to-training data contract is unverified.** The ATAC/RNA pipeline emits `.h5mu`; the training script reads `.h5ad` + `.npy` pseudo-bulk tensors. No bridge exists (BLOCK-6). The "end-to-end multi-omics AI platform" story is not end-to-end yet.
- **Sweep cannot survive HPC walltime limits.** No resume logic, no array-job fanout (BLOCK-3, BLOCK-4). First wall-time kill wastes all accumulated sweep progress.
- **Private repo contents are unverified.** Four of the six repos and the Confluence infra docs were inaccessible. Anything I say about them is inference from the visible code, not a ground-truth read.
- **Reference genome / gene list / JASPAR versions are not pinned.** A silent mismatch between historical S3 data and any new pipeline run will degrade r without raising an exception (RISK-1, RISK-2).

## Recommended Next Step

Before writing any HPC config, do two things in parallel:
1. Grant me read access to the four inaccessible repos and whitelist `*.atlassian.net`, then I re-run this analysis against actual code and produce the final, non-inferential version of this gap list.
2. In the meantime, start BLOCK-3 (checkpoint resume) and BLOCK-4 (argparse refactor) on `train_v11.py` — these are pure application-layer changes, independent of infra, and unblock everything downstream.

## What I Need From You

- Confluence page export OR network allowlist update.
- Direct repo access (read) to `nf-cellranger-multi`, `nf-star-align`, `bioinf-containers`, `aivc-process-10x-atac-rna`.
- Answers to the 10 open questions in Section 9.
- HPC cluster runbook (queue names, CUDA module, GPU partition, wall-time cap, FS type).
