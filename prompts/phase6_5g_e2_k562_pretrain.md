# Phase 6.5g / E2 — K562 lineage pretraining: domain-invariance test

## STATUS

**Type:** Pre-registered domain-invariance experiment (not a pretraining iteration).
**Purpose:** Test whether lineage-matched pretraining (K562 → K562 probe) produces
RSA lift vs. PBMC-pretraining (baseline = 6.5e R_c_e1).
**Parent state:** `main` at tag `phase-6.5-closed` (commit `8e3a527`).
**Date drafted:** 2026-04-20.
**Decisions locked (no drift permitted without explicit re-registration):**

| Axis | Locked choice |
|---|---|
| D1 — Architecture | **B** — Multiome encoder (RNA + ATAC), identical to 6.5 parent |
| D2 — Gene-space harmonization | **A** — Intersection across all E2 source datasets (invariant vector space) |
| D3 — Pretraining data composition | **C** — Norman 2019 controls + multiome K562 + non-perturbed K562 atlases |
| D4 — Batch correction | **scVI-style latent batch correction** (primary); tripwire compares corrected vs. uncorrected RSA |

**Hypothesis under test.** Phase 6.5e and 6.5f jointly falsified the "projection-absorption" and "objective-reweighting" explanations for the PBMC-pretrain RSA null against Norman 2019. The dominant remaining hypothesis is **lineage mismatch** — PBMC-derived pretraining geometry is non-transferable to K562 probe. E2 tests this directly by pretraining on K562-lineage data and measuring RSA against the same Norman 2019 probe batch used in 6.5e/6.5f.

---

## CONTEXT

E2 is not "another pretraining run." Three orthogonal interventions on optimization (recipe), architecture (projection freezing), and objective weighting have all returned NULL at |Δ| < 0.01 on the same Norman 2019 probe. The hypothesis space has been narrowed to two remaining candidates:

1. **Lineage hypothesis:** pretraining tissue must match probe tissue for RSA signal. E2 tests this.
2. **Model-class limitation:** the current encoder family (SimpleRNAEncoder + ATACPeakEncoder + fusion head) is fundamentally insufficient to produce lineage-aligned geometry. **This is what E2-NULL would imply.**

Pre-registered interpretation of outcomes is below (§ Pre-registered outcomes). The FAILURE HANDLING section locks the interpretation of E2-NULL so post-hoc recipe-tuning on K562 is not an admissible next step.

Baseline for comparison: **R_c_e1 = −0.0621** (from `.github/phase6_5e_rsa.json` on main, the 6.5e contrastive-only real-data RSA). CI95 width at probe batch: ≈ 0.023.

---

## GAPS REQUIRING USER INPUT BEFORE EXECUTION

These four items are **not yet locked** and must be resolved before the execution invocation is run. Listing them at the top of the spec so they cannot be silently defaulted.

1. **Multiome K562 dataset source.** Candidates: 10x Genomics Multiome K562 demo (~5k cells, publicly available from 10x website) or Mimitou 2021 ASAP-seq K562 (smaller, protein-tagged). If neither is accessible with clean licensing, fall back to RNA-only K562 and use modality-masking during training (see gap 2).

2. **Modality-missing handling.** D3 includes Norman 2019 (RNA-only) but D1 is multiome. Three sub-options:
   (a) Mask ATAC input + exclude ATAC loss for RNA-only cells (standard multi-modal VAE practice; preferred).
   (b) Drop Norman from pretrain entirely; shrinks D3 to atlas-only.
   (c) Two-stage: multiome pretrain on K562 multiome, then RNA-only fine-tune with Norman controls.
   **My default:** (a). Flag for user approval.

3. **Expected cell counts per source.** The spec assumes:
   - Norman 2019 controls: ~10k cells (10% of ~100k, depending on control proportion in GEO GSE133344)
   - Multiome K562 (10x demo): ~5k cells
   - K562 atlas controls (Gasperini 2019 GSE120861 + Adamson 2016 GSE90546 parental K562): ~20–50k cells combined
   **Total target:** 35–65k cells. If realized N < 20k, flag under-training risk before execution.

4. **Compute budget + training duration.** Parent 6.5 pretrain was 1 epoch. E2 must declare pretrain-epochs up front (pre-registered — no mid-run extension). Options:
   (a) 1 epoch (matches 6.5 parent; cheapest; may under-train if D3 realized N is large)
   (b) 3 epochs with early-stop on held-out recon loss
   (c) Fixed-step budget (e.g. 50k gradient steps) regardless of epoch
   **My default:** (b). Flag for user approval.

---

## Locked gap resolutions

Resolved 2026-04-20 by user sign-off on defaults. These are now **immutable** for E2 execution. Any change requires a new registered decision turn with justification recorded here before execution resumes.

| Gap | Resolution |
|---|---|
| **1 — Multiome K562 source** | 10x Multiome K562 demo (~5k cells, publicly available from 10x Genomics) as primary source. Mimitou 2021 ASAP-seq considered and deferred; RNA-only fallback deferred. |
| **2 — Modality handling** | Option (a): for every cell with `source ∈ {"norman_2019_ctrl", "k562_atlas"}`, ATAC input tensor is exactly zero AND per-cell ATAC loss mask is False. Assertion enforced at the batch-sampler level, not downstream (tripwire T4). |
| **3 — Data composition target** | 35–65k cells combined across Norman 2019 controls + 10x Multiome K562 + Gasperini 2019 / Adamson 2016 parental K562 controls. If realized N at concat time is < 20k, flag under-training risk in PR body **before** Step 6 (pretrain) and pause for user sign-off. |
| **4 — Training duration** | 3 epochs with early-stop on held-out validation reconstruction loss. Validation split: 10% of concat AnnData, stratified by `source` column, seed=3, held out once at Step 3 and reused verbatim for both corrected and uncorrected variants. Early-stop patience: 1 epoch of no improvement (rel. tol. 1e-4). Max wall-clock: 12 GPU-hours per variant; abort + report if exceeded. |

Branch: `phase-6.5g-e2` (matches phase convention from 6.5c/d/e/f).

---

## PREREQUISITES

- `main` branch at tag `phase-6.5-closed` (`git rev-parse 'phase-6.5-closed^{commit}'` == `8e3a52744b07a45a14b2307a6714c184e84313bd`)
- 6.5e probe batch cache on main: `experiments/phase6_5e/probe_batch.npz` (seed=3, Norman 2019 test split indices, SHA anchored in `.github/REAL_DATA_BLOCKERS.md`)
- 6.5e RSA baseline on main: `.github/phase6_5e_rsa.json` (R_c_e1 = −0.0621)
- All 6.5 tripwire patterns (T1 parent SHA, T2 data provenance SHA, T3 vocab discipline, T7 no prior-pipeline edits, T8 weight-motion) carry forward verbatim unless explicitly overridden here.

---

## THE CONTRACT

### Locked architecture (D1 = B)

Use the existing `aivc.skills.rna_encoder.SimpleRNAEncoder`, `aivc.skills.atac_peak_encoder.ATACPeakEncoder`, and `aivc.skills.fusion.FusionHead` module paths **unchanged**. No architectural edits permitted in this phase. If an edit proves necessary, STOP and re-scope as a separate phase.

Stage name: new `pretrain_e2_k562` stage registered in `aivc/training/loss_registry.py`. This stage is a strict copy of the 6.5 parent `pretrain` stage (recon 1.0 + 1.0, contrastive 0.5, aux 0.1) — **the 6.5 loss recipe is locked**, since 6.5f ruled out the recipe as the bottleneck.

### Locked vocabulary (D2 = A)

Compute gene intersection across all D3 sources offline. Commit the resulting ordered gene list as `experiments/phase6_5g/gene_intersection.tsv` with SHA anchored in `.github/REAL_DATA_BLOCKERS.md`. All pretrain AnnData objects are projected to this intersection before training. Expected intersection size: 14–22k genes (likely bounded below by Norman 2019's annotation).

**Invariant:** the intersection gene set must be a subset of the 6.5 pretrain vocab *and* the 6.5e probe vocab. If it isn't, the RSA comparison against R_c_e1 is invalid. This is T3 below.

### Locked data composition (D3 = C)

Three source classes, combined via an AnnData concat with a `source` obs column:

- `source == "norman_2019_ctrl"`: Norman 2019 controls (RNA only; ATAC input zero-padded + masked per Gap 2 option (a))
- `source == "multiome_k562"`: 10x Multiome K562 demo (RNA + ATAC)
- `source == "k562_atlas"`: Gasperini 2019 + Adamson 2016 parental K562 controls (RNA only; ATAC masked)

Obs columns required: `source`, `batch` (dataset-id-derived), `n_counts`, `n_genes`, `pct_mito`. Standard scanpy QC applied per-source before concat. SHA-anchor each source h5ad file in `REAL_DATA_BLOCKERS.md`.

### Locked batch correction (D4 = scVI)

Primary: scVI with `batch_key = "batch"` trained on the concat AnnData, latent dim 128 (matches 6.5 parent). scVI provides the batch-corrected latent *as a regularization signal during pretrain*, not as a replacement for the encoder output. Exact mechanism: scVI latent is concatenated as an additional input to the fusion head at train time only, and the pretrain loss includes a KL regularization term pulling the encoder latent toward the scVI latent with weight 0.01 (low — diagnostic, not dominant).

**Tripwire T_BC (new, pre-registered):** train a parallel uncorrected variant (scVI disabled, batch column ignored). Compare RSA_corrected vs. RSA_uncorrected. Interpretation table in § Pre-registered outcomes. If corrected RSA exceeds uncorrected by > 0.05, the primary result is **INCONCLUSIVE (correction-invented signal)** regardless of other outcomes.

### Deterministic probe reuse

Reuse `experiments/phase6_5e/probe_batch.npz` verbatim. No re-sampling. Seed = 3, indices sorted int64, SHA anchored in REAL_DATA_BLOCKERS.md (already on main).

### Checkpoint contract

- Primary output: `checkpoints/pretrain/pretrain_encoders_e2_k562.pt` (schema_version=1)
- Uncorrected variant: `checkpoints/pretrain/pretrain_encoders_e2_k562_uncorrected.pt`
- Load both via `aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder()` — no bare `torch.load()`.
- SHA anchor both in `.github/REAL_DATA_BLOCKERS.md` under a new `### Phase 6.5g — E2 K562 pretraining` block.

---

## PRE-REGISTERED OUTCOMES

Outcome classification is locked before the run. No post-hoc threshold changes.

Baseline: **R_c_e1 = −0.0621** (6.5e).
Random baseline: **R_random** ≈ 0 (shuffled pairs control, computed on same probe batch).

| Outcome | Primary condition | CI / corroboration | Interpretation |
|---|---|---|---|
| **E2-WIN** | R_e2 − R_c_e1 > +0.05 | CI95 lower bound on Δ > +0.02 AND R_e2 − R_random > +0.05 | Lineage hypothesis confirmed. K562 pretrain produces transferable geometry. Proceed to E3 scoping. |
| **E2-PARTIAL** | +0.02 < (R_e2 − R_c_e1) ≤ +0.05 | CI95 lower bound on Δ > 0.0 | Weak lineage signal. Not publication-grade. Triggers E2-extension phase (longer training or larger atlas). |
| **E2-NULL** | \|R_e2 − R_c_e1\| ≤ 0.02 | — | **Lineage hypothesis falsified under current architecture class.** See FAILURE HANDLING for locked interpretation. |
| **E2-REGRESS** | R_e2 − R_c_e1 < −0.02 | — | K562 pretraining actively harmed probe geometry. Strong signal of architectural mismatch; triggers architecture review phase. |

**Batch-correction confound flag (overrides primary outcome):**

| Condition | Primary outcome becomes |
|---|---|
| R_corrected − R_uncorrected > +0.05 | **INCONCLUSIVE (correction-invented signal)** |
| R_corrected − R_uncorrected ≤ +0.05 AND uncorrected also exceeds baseline thresholds | Primary outcome valid |

Statistics: pair-bootstrap 95% CI on ΔRSA with n_boot = 1000, seed = 3.

---

## TRIPWIRES (pre-registered, LOCKED)

| ID | Name | Assertion |
|---|---|---|
| T1 | Parent SHA match | `git rev-parse HEAD~` is reachable from `phase-6.5-closed`; base branch is `main` at or ahead of `8e3a527` |
| T2 | Data provenance | All 3+ source h5ad files SHA-anchored in `REAL_DATA_BLOCKERS.md` BEFORE training; computed SHAs match recorded values at training-start |
| T3 | Vocab subset invariant | Intersection gene set is a strict subset of (6.5 parent vocab ∩ 6.5e probe vocab). Abort if not. |
| T4 | Modality masking correctness | For every cell with `source ∈ {"norman_2019_ctrl", "k562_atlas"}`, ATAC input tensor is exactly zero AND the per-cell ATAC loss mask is False. Assert at batch sampler level, not downstream. |
| T5 | Batch correction tripwire (T_BC) | Uncorrected variant trained with identical seed, hyperparams, data, recipe. Δ_RSA logged. See outcome table. |
| T6 | Probe batch reuse | SHA of `experiments/phase6_5e/probe_batch.npz` matches the value on main. No re-sampling. |
| T7 | No prior-pipeline edits | `git diff phase-6.5-closed -- aivc/data/ aivc/skills/ aivc/training/` touches ONLY the new `pretrain_e2_k562` stage definition and scVI hookup. No edits to 6.5c/6.5d/6.5e/6.5f code paths. |
| T8 | Encoder weight motion | `max(|W_post − W_pre|)` > 1e-4 for at least one encoder parameter. Exact-zero aborts with "training did not move encoder weights." |
| T9 | NaN/Inf guard | Pretrain loss has no NaN/Inf at any step; abort + save last-known-good snapshot if encountered. |
| T10 | Ckpt SHA anchoring | Both primary and uncorrected ckpt SHAs written to `.github/REAL_DATA_BLOCKERS.md` before PR open. |
| T11 | Drift-ratio logging (diagnostic, non-gating) | Log `\|Δ_rna_enc\| / \|Δ_rna_proj\|` per epoch as in 6.5e/6.5f. Informational for post-run analysis. |

All T1–T10 must report PASS (or ✓) in the PR body tripwire table. T11 is logged but not gating.

---

## TASK (execution outline)

Step 0. Resolve the four GAPS. Commit decisions to the spec as a `## Locked gap resolutions` section before Step 1. No execution until Step 0 is signed off.

Step 1. Acquire and SHA-anchor all 3 source datasets. Write to `data/` (gitignored). Record SHAs in `REAL_DATA_BLOCKERS.md`.

Step 2. Compute gene intersection offline. Commit `experiments/phase6_5g/gene_intersection.tsv` + SHA. Assert T3 invariant vs. 6.5 parent vocab and 6.5e probe vocab.

Step 3. QC + concat sources into a single AnnData. Write intermediate to `data/phase6_5g_concat.h5ad` (gitignored). Record cell counts per source.

Step 4. Train scVI on the concat AnnData (standard scvi-tools config, batch_key="batch", n_latent=128). Persist scVI model for reproducibility.

Step 5. Register `pretrain_e2_k562` stage in `aivc/training/loss_registry.py` — additive edit, no other stage edits. Add tests to `tests/test_phase6_5g_e2.py`.

Step 6. Pretrain (corrected variant). Duration per Gap 4 resolution. Save primary ckpt + metadata.

Step 7. Pretrain (uncorrected variant) with scVI-regularization weight set to 0, batch column ignored. Same seed, same data, same everything else. Save uncorrected ckpt.

Step 8. Linear probe both ckpts against Norman 2019 test split using the `experiments/phase6_5e/probe_batch.npz` indices verbatim. Compute RSA for both + pair-bootstrap CI on Δ.

Step 9. Emit `.github/phase6_5g_rsa.json` with: R_e2_corrected, R_e2_uncorrected, R_c_e1 (quoted from 6.5e), R_random, Δ and CI values, outcome_e2, batch_correction_flag, ckpt SHAs, gap resolutions.

Step 10. Generate `.github/PR_BODY_phase6_5g.md` with: status block, locked decisions, gap resolutions, outcome, full T1–T10 tripwire table, reproducibility anchors section with all four SHAs (h5ad sources, scVI model, primary ckpt, uncorrected ckpt), E3 implication section (gated on outcome).

Step 11. STOP before PR open. User reviews; separate invocation opens PR #20 as Draft.

---

## FAILURE HANDLING

**E2-NULL interpretation is locked.** If outcome is E2-NULL:

1. DO NOT propose recipe-tuning on K562 (LR sweep, longer training, different optimizer). This is a category error per the user's directive: *"your representation space is not lineage-aligned under current architecture class — that is a model class limitation, not a hyperparameter problem."*
2. Report explicit conclusion: **"Under the current SimpleRNAEncoder + ATACPeakEncoder + fusion architecture, lineage-matched pretraining does not produce transferable RSA signal against the Norman 2019 probe."**
3. Next phase (E3) MUST be an architecture-class intervention, not a pretraining-recipe intervention. Candidates: graph-based encoders (PyG GNN over gene regulatory prior), larger transformer-class encoders (ESM-style), contrastive-only pretraining with supervised lineage labels. Re-scope required.

**Other failure modes:**

- T1 fail (parent SHA mismatch): STOP. Main has drifted from `phase-6.5-closed`; investigate before re-running.
- T2 fail (data SHA drift): STOP. Re-download source + re-anchor.
- T3 fail (vocab subset invariant): STOP. Intersection is not a subset of parent/probe vocab. Either (a) the datasets use incompatible annotations, or (b) 6.5 vocab was not what the spec assumed. Report and re-scope vocab strategy.
- T4 fail (modality masking): STOP. Training on un-masked RNA-only cells corrupts ATAC encoder gradients. Abort run.
- T5 (T_BC) flag: do not abort; proceed to full run but outcome is flagged INCONCLUSIVE per outcome table.
- T6 fail: STOP. Probe batch has been mutated vs. 6.5e. Invalid comparison.
- T7 fail: STOP. Scope creep detected.
- T8 fail: STOP. Training did not move encoder weights; likely optimizer misconfig.
- T9 fail: abort training, save last-known-good snapshot for diagnosis.
- T10 fail: do not open PR.

---

## PR PREPARATION

- Branch: `phase-6.5g-e2` off `main` at `phase-6.5-closed`
- Draft PR title: `Phase 6.5g / E2 — K562 lineage pretraining (<OUTCOME_LABEL>)`
- PR body: `.github/PR_BODY_phase6_5g.md` (auto-generated in Step 10 from the template in this spec)
- Open as **Draft only**. Do NOT run `gh pr ready`. Do NOT merge.
- Pre-merge gate template: same 7-check structure as PR #18/#19 with E2-specific substitutions. Will be issued as a separate paste-ready invocation post-PR-open.

Squash-merge convention preserved. No co-author tags.

---

## APPENDIX — decision audit trail

- D1 = B justified in turn 2026-04-20: "objective/weight/projection tweaks don't move the geometry meaningfully … changing modality now would confound the only remaining variable: biology."
- D2 = A justified: "Intersection = invariant evaluation space. Union = dataset leakage. HVG-only = redefines the whole experiment class."
- D3 = C justified: "controls-only → underfits K562 manifold (false negative risk); perturbed inclusion → leaks label structure; atlas K562 controls → gives domain coverage without task leakage."
- D4 = scVI justified: batch normalization strategy required when combining multi-lab sources; user's direct quote: "Otherwise reviewers will correctly challenge interpretability."
- NULL-interpretation guardrail: direct quote baked into FAILURE HANDLING.
