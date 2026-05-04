# Phase 6.5g.3 — Pivot A scope (~1 hour diagnostic)

## Hypothesis

The Calderon eval (`scripts/eval_calderon_linear_probe.py`) forwards
projected Calderon ATAC counts through `PeakLevelATACEncoder` **in
isolation** — never touching the cross-modal fusion module. But the
scientific value of DOGMA pretrain is the *joint* encoder, where SupCon
operated on the L2-normalized mean of three per-modality projections.

If we forward Calderon ATAC through the full pipeline (with masked
RNA + Protein, modality_mask = [1, 0, 0, 0]), the cross-modal alignment
trained during pretrain may help even when only ATAC is observed at
inference. SupCon's effect was on the joint embedding, not the bare
ATAC latent — so it makes sense that the bare ATAC eval misses the
benefit.

This is a cheap test of "the encoder is fine, the eval target is wrong"
before committing to expensive Pivot B (PeakVI swap, ~2 days) or Pivot C
(gene-activity scores, ~5 days).

## Test design

1. **Construct an ATAC-only batch** for Calderon at inference time:
   - `rna = torch.zeros(175, n_genes=36601)`
   - `atac = projected Calderon counts (175, 323500)`
   - `protein = torch.zeros(175, n_proteins=210)`
   - `modality_mask = [[1, 0, 0, 0]] * 175` (ATAC=0 present, others absent)
   - `lysis_idx = torch.zeros(175, dtype=long)` (assume LLL — Calderon has no DIG)

2. **Forward through the same encoders + projections that produced
   z_supcon during training**:
   - `rna_latent = SimpleRNAEncoder(rna_zeros, lysis_idx=zeros)`
   - `atac_latent = PeakLevelATACEncoder(atac, lysis_idx=zeros)`
   - `protein_latent = ProteinEncoder(protein_zeros, rna_emb=rna_latent, lysis_idx=zeros)`
   - `z_rna = rna_proj(rna_latent)`, `z_atac = atac_proj(atac_latent)`,
     `z_protein = protein_proj(protein_latent)`
   - `z_supcon = L2_normalize(mean(L2(z_rna) + L2(z_atac) + L2(z_protein)))`

3. **Linear probe on `z_supcon`** instead of just `atac_latent`. Same
   stratified-kfold and LOO-donor CV as the existing eval.

4. **Compare**:
   - Bare-ATAC kfold (existing, locked at 0.1943)
   - Joint-fusion-zero-RNA-Protein kfold (new, this test)

## Implementation

New file: `scripts/eval_calderon_joint_fusion.py` (~80-100 lines).
Wraps the existing pipeline; adds a `--use_joint_embedding` flag mode
that forwards the full-modality encoder stack (zero-padded RNA/Protein).

Key details:

- Reuse `aivc/eval/calderon_probe.py::project_calderon_to_dogma_space`
  unchanged
- Reuse `aivc/eval/calderon_probe.py::run_linear_probe` unchanged
- New helper `aivc/eval/calderon_probe.py::encode_samples_via_joint_fusion`
  that loads all three encoders + projections from the ckpt and forwards
  the masked batch
- New SLURM submit script `scripts/submit_calderon_joint_eval.slurm`
  parallel to the existing one — takes a ckpt path, runs the joint
  forward, prints headline

## Outcome decision tree

| Joint kfold accuracy | Interpretation | Next move |
|---|---|---|
| **≥ 0.70** | Pre-registered threshold passes via joint pathway. SupCon worked, eval target was wrong. | Phase 6.5g.2 reopens to PASS. Re-evaluate the pre-registration framing. |
| **0.50 – 0.70** | Substantial lift over bare-ATAC (0.1943); joint pathway preserves transfer signal. Below threshold but architecturally informative. | Don't reopen the verdict (pre-registration discipline). But strongly suggests Pivot B/C should focus on better fusion, not necessarily replacing the ATAC encoder. |
| **0.30 – 0.50** | Modest lift; joint helps marginally. | Still pivot. Suggests both ATAC encoder AND fusion need work. Prioritize Pivot C (gene activities) since it changes both. |
| **< 0.30 (no lift)** | Joint pathway also doesn't transfer. | The DOGMA-trained encoder genuinely doesn't generalize to Calderon. Pivot B (PeakVI) is the obvious next step. |

## Cost

- **Implementation**: ~1 hour (small wrapper around existing eval; no new training)
- **Compute**: 1 SLURM job, ~1 min on H100
- **Risk**: low — no production code changes, just a parallel diagnostic script
- **Reversibility**: complete — if Pivot A is informative, we proceed; if not, we delete the wrapper and move to Pivot B

## Blockers / open questions

1. **lysis_idx for Calderon**: Calderon is bulk ATAC, no lysis protocol
   to assign. Default to LLL (lysis_idx=0) — matches what our existing
   eval has been silently doing in `aivc/eval/calderon_probe.py::encode_samples`
   (PR #43 default).

2. **modality_mask shape consistency**: the trained encoders were
   trained with `modality_mask = [[1, 0, 1, 1]]` (no PHOSPHO). For
   Calderon eval, we set `[[1, 0, 0, 0]]`. The encoders' forward
   doesn't read modality_mask directly (only the loss does), so this
   should be safe — but worth a unit test before the SLURM submission.

3. **Which ckpt to eval**: `pretrain_encoders.pt` (final, end-of-training).
   epoch_0025 was the peak in the bare-ATAC eval; if joint-fusion eval
   shows different peak epoch, worth running across all 11 ckpts to map
   the joint learning curve.

## Greenlight

If you authorize Pivot A, I will:

1. Write `scripts/eval_calderon_joint_fusion.py` + `scripts/submit_calderon_joint_eval.slurm`
2. Add a unit test exercising the joint forward path on a synthetic batch
3. rsync to BSC
4. Print the manual sbatch command (no auto-submit; same discipline as before)
5. After eval completes, paste the headline + interpret per the decision tree above
