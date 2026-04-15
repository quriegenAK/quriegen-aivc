# Installing MACS2 for `scripts/harmonize_peaks.py`

`harmonize_peaks.py` requires MACS2 (Model-based Analysis of ChIP-Seq,
v2.x) on `PATH`. There is deliberately **no Python-only fallback**: the
peak set is part of the reproducibility record of Phase 6.7b's real-data
pretraining run, and swapping the peak caller invalidates cross-run
hash comparisons.

## Recommended: pip in an isolated venv

MACS2 is published to PyPI and works with CPython 3.9–3.11.

```bash
python3.11 -m venv .venv-macs2
source .venv-macs2/bin/activate
pip install --upgrade pip
pip install "macs2>=2.2.9"
macs2 --version      # expected: macs2 2.2.9.1 (or newer patch)
```

Point `harmonize_peaks.py` at this venv's Python, or activate the venv
before running the script. MACS2 only needs to be on `PATH` — it is
invoked via `subprocess`, not imported.

## Alternative: conda / bioconda

```bash
conda create -n macs2 -c bioconda -c conda-forge macs2
conda activate macs2
macs2 --version
```

## macOS notes

MACS2 ships Cython extensions. On Apple Silicon, ensure the Xcode
command-line tools are installed (`xcode-select --install`). If pip
fails to build, prefer the conda/bioconda route, which ships prebuilt
binaries.

## Linux notes

On Ubuntu 22.04+:

```bash
sudo apt-get install -y python3.11-dev build-essential zlib1g-dev
pip install "macs2>=2.2.9"
```

## Verifying

After install, from the repo root:

```bash
which macs2
macs2 --version
python scripts/harmonize_peaks.py --help
```

All three must succeed before running the full pipeline. The script
refuses to start if `macs2` is not on `PATH`.
