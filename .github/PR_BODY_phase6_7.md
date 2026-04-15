# Phase 6.7: harmonize_peaks — real-data peak set artifact (code only)

> ## ⚠ CODE ONLY — no execution artifacts in this PR
> This PR ships the code needed to produce the peak-set artifact on a
> MACS2-equipped host. It intentionally contains **19 `<PENDING:
> execution on MACS2-equipped host>` markers** (3 in the download
> script, 6 in `REAL_DATA_BLOCKERS.md` — 5 in "Artifacts produced" +
> 1 pattern-reference inside the Phase 6.7-exec Step 6 instruction,
> 10 in this PR body). Every marker is intentional — they are **not**
> oversights. Execution and marker-fill happens on a separate host
> per the Phase 6.7-exec checklist in `REAL_DATA_BLOCKERS.md`, across
> three follow-up commits. Do not treat a PENDING marker as a gap to
> fix in this PR.

## What
- Promoted `scripts/harmonize_peaks.py` from stub to executable
  MACS2-on-pooled-fragments pipeline (`--fragments`, `--output`,
  `--genome {hg38,mm10}`, `--min_cells`, `--force`). Idempotent:
  refuses to overwrite without `--force`; writes a sibling
  `.meta.json` with input/output SHA-256 and run metadata.
- Added `scripts/download_pbmc10k_multiome.sh` for the 10x PBMC
  Multiome 10k demo, with per-file SHA-256 verification. Expected
  hash constants are `<PENDING>` until first verified run.
- Added `scripts/INSTALL_MACS2.md` documenting the pip/conda install
  paths. No Python-only peak caller fallback is provided by design.
- Established `data/peak_sets/` with a committed `README.md`
  documenting schema, naming convention, and regeneration steps.
- Updated `.gitignore` to exclude `data/raw/` and
  `data/peak_sets/*.tsv`/`*.meta.json` while keeping the README.

## Why
Unblocks Phase 6.7b (real-data pretrain rerun) and Phase 6.5
(linear probe against real checkpoint). All Phase 6 numbers to
date are mock-data only; this PR is the code foundation for the
first real-data signal. Execution is intentionally deferred — see
"Execution outstanding" below.

## Scope of this PR: CODE ONLY

This PR does NOT include a produced peak-set artifact. The host
where this PR was authored does not have MACS2 installed, and the
task explicitly forbids (a) Python-only fallback peak callers,
(b) running `harmonize_peaks.py` against mock data, and
(c) fabricating hashes / peak counts. A follow-up commit on a
MACS2-equipped host fills the `<PENDING>` markers with the real
stdout values from that run.

## Execution outstanding (follow-up commit)

On a host with MACS2 installed and ≥20 GB free in `data/raw/`:

- `data/peak_sets/pbmc10k_hg38_20260415.tsv` path: `<PENDING: execution on MACS2-equipped host>`
- Peak count: `<PENDING: execution on MACS2-equipped host>`
- Output SHA-256: `<PENDING: execution on MACS2-equipped host>`
- Input fragments SHA-256: `<PENDING: execution on MACS2-equipped host>`
- MACS2 version: `<PENDING: execution on MACS2-equipped host>`
- Exact command line: `<PENDING: execution on MACS2-equipped host>`

Checklist the exec host must satisfy before the follow-up commit:
1. `macs2 --version` succeeds.
2. `bash scripts/download_pbmc10k_multiome.sh` exits 0 after
   populating the `EXPECTED_*_SHA256` constants.
3. `python scripts/harmonize_peaks.py ... --genome hg38` exits 0
   with ≥50,000 peaks in the output TSV.
4. Re-running with `--force` on the same fragments file produces a
   byte-identical output TSV (SHA-256 match across runs).
5. `input_sha256=`, `output_sha256=`, `n_peaks=` lines from stdout
   are copied verbatim into `.github/REAL_DATA_BLOCKERS.md`
   "Artifacts produced" and into the bullets above.

## Invariants preserved
- No model, training, encoder, loss, or test file modified.
- `train_week3.py` untouched.
- `aivc/data/multiome_loader.py` untouched; `peak_set_path` contract
  is unchanged.
- All existing tests still pass (static-only changes for Phase 6.7).

## Reproducibility
- Input fragments SHA-256: `<PENDING: execution on MACS2-equipped host>`
- Output peak_set SHA-256: `<PENDING: execution on MACS2-equipped host>`
- MACS2 version: `<PENDING: execution on MACS2-equipped host>`
- Command line: `<PENDING: execution on MACS2-equipped host>`

Note: none of these markers is to be replaced locally on a host
that re-derives the values independently. They must be copied from
the stdout of the authoritative exec-host run that also updates
`REAL_DATA_BLOCKERS.md`.

## Phase 6.7b tripwires
- `scripts/pretrain_multiome.py` must consume the peak_set via
  `MultiomeLoader`'s `peak_set_path` — no other load path.
- Resulting checkpoint SHA-256 MUST differ from the mock-data hash
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
  recorded in `REAL_DATA_BLOCKERS.md`. Asserting this difference is
  a Phase 6.7b validation gate.

## Not in scope
- Pretraining rerun (Phase 6.7b).
- Linear probe rerun (Phase 6.5).
- Any model code changes.
- Producing or committing the peak-set artifact itself (separate
  follow-up commit on an exec host; see "Execution outstanding").

## Known issues
- **Test pollution in `tests/test_ckpt_loader.py`** —
  quriegenAK/quriegen-aivc#7. The file fails when run as part of a
  full-suite pytest invocation but passes in isolation. Reproduces
  on `main` at 0918c36 without this PR's changes, so it is
  pre-existing and explicitly out of scope for Phase 6.7 (which does
  not touch any model/training/test file). Filed as a separate issue
  to be tracked, not absorbed into this PR's summary.
