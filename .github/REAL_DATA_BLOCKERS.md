# Real-data rerun blockers

Tracked deliverables that MUST be closed before Phase 6.5 can start.
This file replaces `PHASE6_BLOCKERS.md` — the blocker is no longer
Phase-6-specific, it gates every downstream ablation that claims a
biological (rather than methodology) signal.

Phase 5 shipped without it, Phase 6 shipped without it. If Phase 6.5
starts before these items close, the next ablation is also
methodology-only — the same footnote repeats a third time.

## Mock-data checkpoint of record

The current checkpoint at `checkpoints/pretrain/pretrain_encoders.pt`
was produced on the synthetic Multiome fallback in
`scripts/pretrain_multiome.py`.

- **SHA-256:**
  `10665544bd888dbeddd1c3fdd4b880515e7bb05c16f9a235b453c88b50c5a24b`
- **Config:** `n_genes=500`, `hidden_dim=256`, `latent_dim=128`,
  `schema_version=1`.
- **Use:** methodology validation only. The linear-probe numbers in
  `.github/PR_BODY_phase6.md` derive from this checkpoint.

The real-data rerun MUST produce a checkpoint with a SHA-256 different
from the hash above. Same hash = same checkpoint = mock data was
reused → reject.

## Deliverable: harmonize_peaks + real-data pretrain

- **Owner:** Ash Khan
- **Target date:** 2026-04-29 (two weeks from Phase 6 PR merge)
- **Definition of done** (all three):
  1. Peak-set artifact exists at `data/peak_sets/pbmc10k_v1.tsv`
     (or equivalent documented path), produced by
     `scripts/harmonize_peaks.py` on 10x PBMC Multiome fragments.
  2. One successful `scripts/pretrain_multiome.py` run consuming that
     peak-set artifact (not the mock fallback), with loss monotone on
     moving average and `schema_version=1` validated on save.
  3. The resulting checkpoint is stored at
     `checkpoints/pretrain/pretrain_encoders.pt` with a SHA-256
     **different** from the mock-data hash recorded above, and the
     new hash is appended to the appendix at the bottom of this file.

### Command sketch

```
python scripts/harmonize_peaks.py \
    --fragments pbmc_multiome_fragments/*.tsv.gz \
    --out data/peak_sets/pbmc10k_v1.tsv

python scripts/pretrain_multiome.py \
    --multiome_h5ad data/pbmc_multiome.h5ad \
    --peak_set data/peak_sets/pbmc10k_v1.tsv \
    --steps 10000
```

## Real-data linear-probe rerun (activates the Phase 6 gate)

Once the real-data checkpoint exists:

- Run:
  ```
  python scripts/exp1_linear_probe_ablation.py \
      --ckpt_path checkpoints/pretrain/pretrain_encoders.pt \
      --dataset_name norman2019 --seed 17 --wandb
  ```
- Confirm the logged `ckpt_sha256` differs from the mock hash above.
- This rerun — not the numbers in `PR_BODY_phase6.md` — is what
  activates the Phase 6 interpretation gate (≥ +5% relative R² on
  top-50 DE → proceed to Phase 6.5).

## Escalation

If `harmonize_peaks` turns out to require a multi-day investment, this
stops being a PR-closeout item and becomes a named work item on its
own branch. Do not open Phase 6.5 work until the definition of done
above is fully satisfied.

## Appendix: real-data checkpoint hashes (append on rerun)

<!-- On each real-data rerun, append a line: YYYY-MM-DD <sha256> <notes> -->
