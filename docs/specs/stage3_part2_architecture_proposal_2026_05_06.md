# Stage 3 Part 2 — Architecture Proposal

**Status**: Locked for implementation. Production-ready, no exploratory branches.
**Authored**: 2026-05-06
**Authorizes**: Stage 3a (Mimitou adapter + held-out CD3E+CD4 zero-shot synergy demo) AND Stage 3b (QurieSeq Phase 1 temporal dynamics).
**Supersedes**: None — this is the canonical Stage 3 architecture spec.
**Prerequisites**: Stage 3 Part 1 closed (all 5 reports). Report 2 verdict: ADAPTER_RECOMMENDED at 0.5676 synergy 4-class accuracy.

---

## 0. Decision summary

| Decision               | Choice                                  | Why                                                                                              |
|------------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------|
| Encoder modification   | **Frozen + Adapter**                    | Preserves validated 0.7308 cell-type discrimination; 0.5676 perturbation accuracy is adapter-class, not fine-tune-class. |
| Temporal backbone      | **Neural ODE (torchdiffeq)**            | Continuous-time matches QurieSeq's 0–180 min sampling; stable training; no discrete timestep hyperparameter. |
| Perturbation handling  | **4-head decomposed readout**           | Vehicle baseline + Δ_stim + Δ_inh + Δ_synergy. Zero-arm inductive bias enables zero-shot synergy. |
| Donor handling         | **5-donor static context embedding**    | z_static(donor) = adapter(encoder_atac(x_ATAC_t0)). Conditions all dynamics + heads.             |
| Output decomposition   | **Per-modality + pathway-aware heads**  | RNA + Protein + Phospho heads, each pathway-regularized via Hallmark + KEGG (58 sets, 4798 genes). |
| Stage 3a primary eval  | **Zero-shot CD3E+CD4 synergy on Mimitou** | Held-out double-KO arm; predict Δ_xy via single-arm-trained synergy head; expect ≥0.70 cosine sim to true centroid. |
| Stage 3b primary eval  | **Zero-shot BTK+JAK synergy on QurieSeq** | Held-out donor + held-out drug pair. Threshold pre-registered when QurieSeq labels land.        |

---

## 1. System contract

### 1.1 High-level data flow

```
                  STATIC DONOR CONTEXT                  DYNAMIC STATE TRAJECTORY
                  ═════════════════════                 ════════════════════════
              x_ATAC(donor, t=0)                    x_RNA(t), x_Protein(t)
                       │                                       │
                       ▼                                       ▼
               [FROZEN ENCODER]                      [FROZEN RNA + PROT ENC]
              z_atac_raw (donor)                    z_rna(t), z_prot(t)
                       │                                       │
                       ▼                                       ▼
                  [ADAPTER]                           [LATE FUSION → z_dyn(t)]
                  Linear-LN-GELU                       (frozen, DOGMA fusion)
                       │                                       │
                       ▼                                       │
              z_static(donor) ∈ ℝ^d                            │
                       │                                       │
                       └─────────────────┬─────────────────────┘
                                         │
                                         ▼
                       [TEMPORAL TRANSITION  f_θ via Neural ODE]
                       dz_dyn / dt = f_θ(z_dyn, z_static, t, donor_emb, pert)
                                         │
                                         ▼
                                  z_dyn(t = 0 → 180 min)
                                         │
                       ┌─────────────────┼─────────────────┬───────────────┐
                       ▼                 ▼                 ▼               ▼
                [BASELINE HEAD]   [STIM HEAD]      [INH HEAD]      [SYNERGY HEAD]
                  h_b(·)         Δ_s(·, stim)    Δ_i(·, inh)    Δ_xy(·, stim, inh)
                                         │
                                  ŷ(t) = h_b + I_s·Δ_s + I_i·Δ_i + I_s∧I_i·Δ_xy
                                         │
                       ┌─────────────────┼─────────────────┬───────────────┐
                       ▼                 ▼                 ▼               ▼
                [RNA DECODER]   [PROTEIN DEC]    [PHOSPHO DEC]    [PATHWAY SUMMARY]
                  per-gene       per-antibody    pathway-aligned    Hallmark + KEGG
                  logits         expression      (Phase 2 only)     scores
```

### 1.2 Mathematical contract

**State at time `t`**: `z_dyn(t) ∈ ℝ^d` where `d = latent_dim` (current encoder: 256).

**Donor context**: `z_static(donor) ∈ ℝ^d`. Computed once per donor at `t=0`, conditions all downstream.

**Continuous-time dynamics** (Neural ODE):
```
dz_dyn/dt = f_θ(z_dyn(t), z_static, t, donor_emb, pert_emb)
z_dyn(0)  = encoder_rna_prot(x_RNA(0), x_Protein(0))
```

**Decomposed readout at time `t`**:
```
ŷ(t) = h_b(z_dyn(t), z_static, t)                        # vehicle baseline drift
     + I[stim] · Δ_s(z_dyn(t), z_static, t, stim_emb)    # stim-induced delta
     + I[inh]  · Δ_i(z_dyn(t), z_static, t, inh_emb)     # inh-induced delta
     + I[stim ∧ inh] · Δ_xy(z_dyn(t), z_static, t, stim_emb, inh_emb)  # synergy
```

where `I[·]` is the perturbation-presence indicator (0 or 1).

**Zero-arm constraint** (inductive bias for zero-shot synergy):
- NTC + vehicle cells: `Δ_s = Δ_i = Δ_xy = 0`
- Stim-only cells: `Δ_i = Δ_xy = 0`
- Inh-only cells: `Δ_s = Δ_xy = 0`
- Stim + inh cells: no constraint (training signal for `Δ_xy`)

This is enforced as an L2 penalty during training; `Δ_xy` then learns *only* the non-additive synergy correction, which is what makes held-out double-KO prediction tractable.

### 1.3 Invariants — DO NOT relax

These are locked decisions, not options:

1. **Encoder is frozen.** SHA `416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e`. Adapter is the only learnable layer on top.
2. **Adapter is frozen after Stage 3a training.** Stage 3b temporal dynamics train on top of frozen adapter output, not the other way around.
3. **No transformers on the dynamics path.** Continuous-time biological state transition requires Neural ODE / latent SDE / RSSM. Transformers' attention mechanism does not match the biological constraint of local-in-time state evolution.
4. **4-head decomposed readout, not single conditional head.** The zero-arm constraint is mathematically equivalent to a single conditional head IF the conditional head learns zero outputs for missing conditions. Explicit decomposition makes the constraint inductive bias rather than learned behavior — critical for zero-shot synergy.
5. **Donor as static context, not trainable embedding mixed into dynamics.** `z_static(donor)` is computed once from t=0 ATAC and held constant through the trajectory. Donor effects propagate through the dynamics function via this fixed context, not as a learnable time-varying parameter.
6. **Output is per-modality + pathway-aware.** RNA, Protein, Phospho decoders are separate. Each is regularized by pathway-level summary scores aligned to the canonical Hallmark + 8 KEGG immune set from Report 3.

---

## 2. Phased plan

### Phase 3a — Mimitou adapter training + zero-shot synergy validation (NOW, public data)

**Duration**: ~5-7 days.
**Goal**: Train + freeze the adapter; validate the 4-head decomposed readout architecture via held-out CD3E+CD4 zero-shot synergy prediction.
**Inputs**: Mimitou CD4 CRISPR h5ad (existing on BSC), 6 perturbation arms.
**Outputs**: Frozen adapter checkpoint, zero-shot synergy validation report.

**Why this phase**: Mimitou is the only public PBMC dataset with a clean 4-arm factorial design (NTC + sgCD3E + sgCD4 + sgCD3E_CD4_double). It cannot exercise the temporal dynamics (single endpoint at 16h), but it CAN validate the adapter + decomposed-readout architecture before QurieSeq lands.

**Pre-registered eval**:
- Train heads on {NTC, sgCD3E_only, sgCD4_only, unrelated CRISPRs (ZAP70, NFKB2)}.
- Hold out the CD3E_CD4_double arm entirely.
- At inference, predict the double-KO embedding via the synergy head: `ŷ_double = h_b + Δ_s(CD3E) + Δ_i(CD4) + Δ_xy(CD3E, CD4)`.
- Compare predicted embedding to actual held-out double-KO centroid.
- **Threshold**: cosine similarity ≥ 0.70 to true double-KO centroid.
- **Null baseline**: "no-synergy" prediction = `h_b + Δ_s + Δ_i` (sum of single deltas, Δ_xy=0). Expected to land ~0.50.
- **Sanity check**: random-projection synergy head should fail (sim < chance).

### Phase 3b — QurieSeq Phase 1 temporal dynamics (July 2026, proprietary data)

**Duration**: ~3-4 weeks after QurieSeq Phase 1 data lands.
**Goal**: Train the Neural ODE temporal backbone + per-time-point decomposed readouts on QurieSeq's 5-donor × 5-timepoint × N_stim × M_inh factorial.
**Inputs**: QurieSeq Phase 1 RNA + Protein + ATAC (t=0 only) labeled by donor + stim + inh + timepoint.
**Outputs**: Trained Neural ODE + readout heads; donor-generalization + drug-synergy eval reports.

**Pre-registered eval gates** (TBD when QurieSeq labels are finalized; placeholders):
- **Donor generalization**: hold out 1 of 5 donors; predict trajectories. Threshold ≥ 0.65 cosine sim averaged over timepoints + perturbations.
- **BTK+JAK zero-shot synergy** (the primary clinical demo): train on {vehicle, BTK_only, JAK_only, other drug pairs}, hold out BTK+JAK. Predict trajectory; compare to held-out true.
- **Trajectory consistency**: at any held-out timepoint t, `ŷ(t)` should be on the path between `ŷ(t-Δ)` and `ŷ(t+Δ)` — measured by interpolation error.

### Phase 3c — QurieSeq Phase 2 phospho integration (Q4 2026, proprietary data)

**Duration**: ~2-3 weeks after Phase 2 data lands.
**Goal**: Add phospho readout decoder; align readouts to pathway-level summary scores (pJAK1 → JAK_STAT_signaling, pERK → MAPK_signaling, etc.).
**Inputs**: Phospho readouts in QurieSeq Phase 2.
**Outputs**: Phospho decoder; pathway-level summary score outputs; demo-ready inference pipeline.

---

## 3. Model components

### 3.1 Adapter (Stage 3a deliverable)

```python
class PerturbationAdapter(nn.Module):
    """Frozen-encoder → perturbation-discriminative projection.

    Parameters: 2 * d * d + 2 * d  ≈ 130K at d=256.
    Trained: Stage 3a only. Frozen for 3b/3c.
    """
    def __init__(self, d: int = 256):
        super().__init__()
        self.proj_1 = nn.Linear(d, d)
        self.ln    = nn.LayerNorm(d)
        self.act   = nn.GELU()
        self.proj_2 = nn.Linear(d, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj_2(self.act(self.ln(self.proj_1(z))))
```

**Training**: SupCon (Khosla 2020) over Mimitou's 6 perturbation arms.
- Temperature: τ = 0.07 (matches DOGMA pretrain).
- λ_supcon: 0.5.
- Optimizer: AdamW(lr=1e-3, weight_decay=0.01).
- Epochs: ~30 with early-stop on held-out synergy eval.
- Batch: 256 cells, balanced over perturbation arms.

**Frozen-encoder-as-residual decision**: the adapter is NOT a residual layer (no `z + adapter(z)` skip). Empirical: in Report 2, raw encoder achieved 0.5676 on synergy. Residual would re-introduce raw encoder's bias toward cell-type-only structure. Pure projection lets the adapter learn a perturbation-discriminative subspace freely. If post-training the adapter ends up close to identity, that's a learned property, not an imposed constraint.

### 3.2 Donor static context

```python
def compute_donor_static_context(
    atac_t0: sp.csr_matrix,         # (n_cells_for_this_donor, n_union_peaks)
    encoder: nn.Module,             # frozen DOGMA encoder
    adapter: nn.Module,             # frozen post-Stage-3a adapter
    lysis_idx: torch.Tensor = None, # 0 for LLL (QurieSeq default)
) -> torch.Tensor:
    """Compute z_static(donor) by pseudo-bulking donor t=0 ATAC and
    encoding through frozen encoder → adapter.
    """
    pseudo_bulk = atac_t0.sum(axis=0)  # (n_union_peaks,)
    z = encoder(pseudo_bulk.unsqueeze(0), lysis_idx=lysis_idx)
    z = adapter(z)
    return z.squeeze(0)  # (d,)
```

Computed once per donor at training start; cached. No gradients flow through this path.

**Why pseudo-bulk for donor context**: a donor's *epigenetic* identity is captured by the population-level chromatin landscape, not per-cell variation. Pseudo-bulking averages out single-cell noise and produces a stable donor-level fingerprint.

### 3.3 Temporal transition (Neural ODE)

```python
class TemporalODEFunc(nn.Module):
    """f_θ(z_dyn, z_static, t, donor_emb, pert_emb).

    The continuous-time vector field for the cell state trajectory.
    Donor and perturbation embeddings condition the dynamics; time
    enters via a sinusoidal positional encoding (matches ODE solver
    requirement that f is callable at arbitrary t).
    """
    def __init__(self, d: int = 256, n_donors: int = 5, pert_dim: int = 32):
        super().__init__()
        self.donor_emb = nn.Embedding(n_donors, d)
        self.t_enc_dim = 16  # sinusoidal time encoding
        # MLP: concat(z_dyn, z_static, t_enc, donor_emb, pert_emb) → d
        in_dim = d + d + self.t_enc_dim + d + pert_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2*d),
            nn.GELU(),
            nn.Linear(2*d, 2*d),
            nn.GELU(),
            nn.Linear(2*d, d),
        )

    def forward(self, t: torch.Tensor, z_dyn: torch.Tensor,
                z_static, donor_idx, pert_emb) -> torch.Tensor:
        t_enc = sinusoidal_time_encoding(t, self.t_enc_dim)
        donor_e = self.donor_emb(donor_idx)
        inp = torch.cat([z_dyn, z_static, t_enc, donor_e, pert_emb], dim=-1)
        return self.mlp(inp)  # dz_dyn/dt
```

**ODE solver**: `torchdiffeq.odeint` with method `'rk4'` and `step_size=5 min` (QurieSeq's coarsest sampling). For inference at irregular times, use adaptive `'dopri5'`.

**Why Neural ODE over latent SDE**:
- Continuous time matches QurieSeq's 0–180 min sampling exactly.
- No noise injection at training time — biological cell-to-cell heterogeneity is captured in the encoder's input distribution, not in the dynamics function.
- If Phase 3c shows the deterministic ODE underfits high-variance perturbations (e.g., apoptosis cascades), upgrade to latent SDE then.

**Why Neural ODE over RSSM**:
- RSSM (Hafner et al.) is discrete-time. Biological signaling is continuous; discrete-time models force a hyperparameter for timestep granularity that biology doesn't have.

### 3.4 Decomposed readout heads (4 parallel MLPs)

```python
class DecomposedReadout(nn.Module):
    """Four parallel MLPs: h_b, Δ_s, Δ_i, Δ_xy.

    All four operate on (z_dyn, z_static, t_enc) with optional
    perturbation embeddings.
    """
    def __init__(self, d: int = 256, pert_dim: int = 32, output_dim: int = 256):
        super().__init__()
        t_enc_dim = 16
        base_in   = d + d + t_enc_dim                              # baseline
        stim_in   = d + d + t_enc_dim + pert_dim                   # + stim
        inh_in    = d + d + t_enc_dim + pert_dim                   # + inh
        syn_in    = d + d + t_enc_dim + pert_dim + pert_dim        # + both

        def mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 2*d), nn.GELU(),
                nn.Linear(2*d, output_dim),
            )

        self.h_b   = mlp(base_in)
        self.delta_s  = mlp(stim_in)
        self.delta_i  = mlp(inh_in)
        self.delta_xy = mlp(syn_in)

    def forward(self, z_dyn, z_static, t_enc,
                stim_emb=None, inh_emb=None):
        """Return ŷ(t) = h_b + I_s·Δ_s + I_i·Δ_i + I_s∧I_i·Δ_xy.

        stim_emb/inh_emb may be None (indicates that perturbation is absent).
        """
        base = self.h_b(torch.cat([z_dyn, z_static, t_enc], dim=-1))
        out = base
        if stim_emb is not None:
            out = out + self.delta_s(torch.cat([z_dyn, z_static, t_enc, stim_emb], dim=-1))
        if inh_emb is not None:
            out = out + self.delta_i(torch.cat([z_dyn, z_static, t_enc, inh_emb], dim=-1))
        if stim_emb is not None and inh_emb is not None:
            out = out + self.delta_xy(torch.cat(
                [z_dyn, z_static, t_enc, stim_emb, inh_emb], dim=-1))
        return out
```

Output of `forward` is a `(B, d)` latent — which then feeds the modality decoders below.

### 3.5 Pathway-aware modality decoders

```python
class PathwayAwareRNADecoder(nn.Module):
    """RNA expression decoder with pathway-level summary regularization.

    Outputs per-gene logits + pathway-level summary scores. The pathway
    summary is computed via a linear pool over genes weighted by the
    static gene-to-pathway map (data/pathway_annotations/).
    """
    def __init__(self, d: int = 256, n_genes: int = 36601, gene_to_pathway_W = None):
        super().__init__()
        self.gene_logits = nn.Linear(d, n_genes)
        # Register pathway pool as a buffer (not trainable; static map)
        # gene_to_pathway_W: (n_pathways, n_genes) sparse, normalized
        self.register_buffer("g2p_W", gene_to_pathway_W)

    def forward(self, z: torch.Tensor):
        gene_logits = self.gene_logits(z)
        pathway_scores = gene_logits @ self.g2p_W.T  # (B, n_pathways)
        return gene_logits, pathway_scores

# Analogous: PathwayAwareProteinDecoder, PathwayAwarePhosphoDecoder.
```

**Pathway pool matrix**: built once from `data/pathway_annotations/gene_to_pathway_map.csv`. Shape `(58 pathways, n_genes_in_pathway_union)`. Weights: row-normalized to 1.0 over pathway members so each pathway score is a mean-over-members.

**Why pathway-aware**: enables downstream QurieSeq Phase 2 phospho alignment (pJAK1 → JAK_STAT_signaling, pERK → MAPK_signaling) at the readout level, not as a post-hoc analysis. Pathway summary scores become a natural anchor for cross-modality (RNA/Protein/Phospho) consistency losses.

---

## 4. Training procedure

### 4.1 Loss composition

```
L = L_recon                    (modality reconstruction; mean over RNA + Prot + Phospho)
  + λ_zero · L_zero_arm        (zero-arm constraint on Δ_s, Δ_i, Δ_xy)
  + λ_pathway · L_pathway      (cross-modality pathway consistency)
  + λ_smooth · L_smooth        (trajectory smoothness regularizer — Phase 3b only)
  + λ_synergy · L_synergy_held_out  (eval-only; not back-propagated)
```

#### L_recon
```
L_recon = Σ_modality [ MSE(ŷ_modality(t), y_modality_true(t)) ]
```
Each modality contributes with weight 1.0; can rebalance if a modality (e.g., phospho with sparse readouts) dominates.

#### L_zero_arm — the inductive bias enabling zero-shot synergy
```
For NTC + vehicle cells:     ||Δ_s||² + ||Δ_i||² + ||Δ_xy||²
For stim-only cells:         ||Δ_i||² + ||Δ_xy||²
For inh-only cells:          ||Δ_s||² + ||Δ_xy||²
For stim + inh cells:        no constraint
```
λ_zero = 1.0 (full weight). Without this, the decomposed heads collapse to a single conditional head and zero-shot synergy fails.

#### L_pathway — cross-modality consistency
```
L_pathway = Σ_(m, m') MSE(pathway_score_m(t), pathway_score_{m'}(t))
```
Forces RNA, Protein, and Phospho pathway-summary scores to agree at each timepoint. Critical for Phase 3c phospho integration.

#### L_smooth (Phase 3b temporal only)
```
L_smooth = Σ_t ||ŷ(t) - ŷ(t-Δt)||² / Δt²
```
Penalizes high-frequency oscillations in the trajectory. Helps the ODE solver converge.

### 4.2 Training schedule

**Phase 3a (adapter only)**:
- Stage 1: warm up adapter with SupCon over Mimitou perturbation arms (~10 epochs)
- Stage 2: jointly train adapter + decomposed-readout heads on Mimitou (~20 epochs)
- Stage 3: freeze adapter. Validate on held-out CD3E+CD4 double-KO arm.

**Phase 3b (temporal dynamics)**:
- Stage 1: freeze adapter + encoder. Initialize ODE func + decomposed heads.
- Stage 2: train on QurieSeq Phase 1 with full loss (recon + zero-arm + pathway + smooth). ~50 epochs.
- Stage 3: freeze everything. Validate on held-out donor + held-out drug pair.

**Phase 3c (phospho)**:
- Stage 1: freeze adapter + encoder + ODE + RNA/Protein heads. Train phospho decoder only. ~20 epochs.
- Stage 2: optionally unfreeze pathway-pool weights with low LR if cross-modality pathway scores diverge. ~10 epochs.

---

## 5. Eval gates (pre-registered)

### 5.1 Stage 3a gate

| Metric                                              | Threshold | Failure → |
|-----------------------------------------------------|-----------|----------|
| CD3E+CD4 zero-shot synergy cosine sim (vs true)     | ≥ 0.70    | Architecture-class pivot to single conditional head + zero-shot regularization (deferred). |
| "No-synergy" null baseline (Δ_xy=0)                 | ≤ 0.55    | If null already ≥ 0.55, synergy head is uninformative — re-examine zero-arm constraint. |
| Random-projection synergy head sanity               | < chance  | If random projection beats the real synergy head, the architecture is broken. |
| Per-arm accuracy on Mimitou full-cycle reconstruction (NTC, CD3E, CD4) | ≥ 0.70    | Adapter has degraded encoder's discrimination; investigate. |

#### Bootstrap CI interpretation (pre-committed)

The held-out double-KO arm has only 97 cells, splitting to ~49 train / ~48 test under the 50/50 split. Bootstrap CI on the synergy cosine sim metric is expected to be approximately ±0.10 at n=48. To avoid post-hoc threshold litigation, the verdict mapping below is **pre-committed**:

| Reported metric           | CI behavior                       | Verdict          |
|---------------------------|-----------------------------------|------------------|
| ≥ 0.75                    | regardless                        | **GREEN — pass**     |
| 0.65 – 0.75               | bootstrap 95% CI includes 0.70    | **GREEN — pass**     |
| 0.65 – 0.75               | bootstrap 95% CI excludes 0.70    | **AMBER — re-run** with full Mimitou-corpus expanded test set (~32K cells across all 6 arms; lowers per-arm CI but loses true zero-shot purity) |
| 0.55 – 0.65               | regardless                        | **AMBER — re-run** with λ_zero reduced from 1.0 → 0.5 (Risk #3 banked mitigation) |
| < 0.55                    | regardless                        | **RED — architecture-class pivot** to single conditional head + zero-shot regularization. Do NOT continue Stage 3a-as-designed. |

Computational eval procedure: 1000-iteration bootstrap, sampling test cells with replacement, recompute centroid + cosine sim, report mean + 2.5/97.5 percentile bounds. Random seed pinned for reproducibility (seed=0 by spec convention).

### 5.2 Stage 3b gate (placeholders pending QurieSeq finalization)

| Metric                                              | Threshold | Failure → |
|-----------------------------------------------------|-----------|----------|
| BTK+JAK zero-shot synergy (held-out donor + drug)   | ≥ 0.70    | Demo-class failure. Investigate per-donor vs per-drug-class contribution. |
| Donor generalization (held-out donor, trained drugs) | ≥ 0.65   | Donor-static context is insufficient; consider learned donor embedding update. |
| Trajectory smoothness (interpolation error)         | RMS < 0.15 | ODE undertrained or solver step too coarse. |
| Cross-modality pathway consistency (RNA ↔ Protein)  | r ≥ 0.70  | λ_pathway needs increase. |

### 5.3 Stage 3c gate (Phase 2 phospho)

| Metric                                              | Threshold | Failure → |
|-----------------------------------------------------|-----------|----------|
| pJAK1 ↔ JAK_STAT_signaling pathway score correlation | r ≥ 0.65 | Pathway pool weights need tuning or phospho-specific decoder needs deeper head. |
| pERK ↔ MAPK_signaling correlation                    | r ≥ 0.65 | (same) |
| Cross-modality pathway consistency (RNA ↔ Phospho)   | r ≥ 0.65 | (same) |

---

## 6. Implementation steps (file-by-file)

All paths relative to `aivc_genelink/`. Estimated LOC (lines of code) for each new file.

### Stage 3a (immediate — ~5-7 days)

| File                                              | LOC  | Purpose                                                       |
|---------------------------------------------------|------|--------------------------------------------------------------|
| `aivc/skills/adapter.py`                          | ~80  | `PerturbationAdapter` module + load/save helpers.            |
| `aivc/skills/decomposed_readout.py`               | ~150 | `DecomposedReadout` module + zero-arm constraint helpers.    |
| `aivc/skills/pathway_decoder.py`                  | ~120 | `PathwayAwareRNADecoder`, `...ProteinDecoder` + pathway pool load. |
| `aivc/training/perturbation_loss.py`              | ~180 | Composite loss: L_recon + L_zero_arm + L_pathway. SupCon for adapter. |
| `aivc/data/mimitou_loader.py`                     | ~100 | Mimitou-aware DataLoader; perturbation embedding lookup; arm-balanced batch sampler. |
| `scripts/build_pathway_pool_matrix.py`            | ~80  | One-shot: load gene_to_pathway_map.csv, build sparse (n_pathways, n_genes) pool. |
| `scripts/train_stage3a_adapter.py`                | ~250 | Entrypoint: SupCon warmup → joint train → freeze adapter → eval. |
| `scripts/eval_synergy_zeroshot.py`                | ~200 | Held-out CD3E+CD4 synergy eval; cosine sim + null baseline. |
| `scripts/submit_stage3a_train.slurm`              | ~50  | BSC SLURM wrapper (H100, 24h budget for adapter training).   |
| `tests/test_decomposed_readout.py`                | ~150 | Zero-arm constraint enforcement; gradient flow per-head; output shape. |
| `tests/test_perturbation_adapter.py`              | ~100 | Forward shape; identity-initialization stability; backward.  |
| `tests/test_stage3a_integration.py`               | ~120 | Real-data smoke (gated on `AIVC_RUN_REAL_DATA_SMOKE=1`).     |

**Stage 3a deliverables**: trained + frozen adapter checkpoint, zero-shot synergy validation report at `docs/reports/stage3a_validation.md`.

### Stage 3b (when QurieSeq Phase 1 lands — ~3-4 weeks)

| File                                              | LOC  | Purpose                                                       |
|---------------------------------------------------|------|--------------------------------------------------------------|
| `aivc/skills/temporal_ode.py`                     | ~220 | `TemporalODEFunc` + ODE solver wrapper.                      |
| `aivc/skills/donor_static_context.py`             | ~80  | Pseudo-bulk encoder for z_static(donor).                     |
| `aivc/data/quriseq_loader.py`                     | ~250 | QurieSeq labeled DataLoader; per-cell (donor, stim, inh, time). |
| `aivc/training/temporal_loss.py`                  | ~150 | Trajectory smoothness + extended recon over time.            |
| `scripts/train_stage3b_temporal.py`               | ~300 | Entrypoint: train ODE + heads with frozen adapter+encoder.   |
| `scripts/eval_btk_jak_synergy.py`                 | ~250 | Held-out donor + drug-pair eval.                             |
| `scripts/submit_stage3b_train.slurm`              | ~50  | BSC SLURM wrapper (H100, 48h budget).                        |
| `tests/test_temporal_ode.py`                      | ~150 | ODE solver determinism, gradient stability.                  |
| `tests/test_stage3b_integration.py`               | ~150 | Synthetic-time-course smoke test.                            |

### Stage 3c (Phase 2 — ~2-3 weeks after Phase 2 data)

| File                                              | LOC  | Purpose                                                       |
|---------------------------------------------------|------|--------------------------------------------------------------|
| `aivc/skills/phospho_decoder.py`                  | ~100 | Phospho readout decoder + pathway-readout alignment.         |
| `scripts/train_stage3c_phospho.py`                | ~200 | Phospho head training on frozen ODE.                         |
| Updates to `aivc/training/perturbation_loss.py`   | ~30  | Add cross-modality pathway consistency (RNA ↔ Protein ↔ Phospho). |

---

## 7. Key risks (Stage 3a critical)

1. **Mimitou adapter may not generalize to QurieSeq's distribution.** Mimitou is genetic CRISPR-KO at 16h endpoint; QurieSeq will be pharmacological inhibitors at 0–180 min. The adapter learns a perturbation-discriminative projection in CRISPR-perturbation-space; this space may not align with drug-perturbation-space. Mitigation: re-evaluate adapter on QurieSeq Phase 1 first batch (~1 donor + ~5 timepoints + ~5 conditions); decide whether to (a) keep adapter as-is, (b) train a 2nd adapter on QurieSeq for Phase 3b/3c, or (c) unfreeze and joint-train. Bank as known risk.
2. **CD3E+CD4 double-KO arm has only 97 cells (49 train / 48 test post-split).** This is the lower bound for reliable held-out synergy eval. If the zero-shot cosine sim comes in marginally (e.g., 0.65), interpretation is bounded by small-n uncertainty. Mitigation: report bootstrap CI; if CI overlaps the 0.70 threshold, treat verdict as inconclusive.
3. **The zero-arm constraint may be too rigid.** Real biology has some "leakage" — a stim-only cell does have *some* baseline inhibitor pathway activity. Hard-zero on Δ_i for stim-only cells may over-penalize. Mitigation: use a soft L2 penalty (λ_zero = 1.0 initial; tunable) rather than hard zero. If Stage 3a converges with degraded recon, reduce λ_zero to 0.3.
4. **The 4-bug pattern in `prepare_mimitou_crispr.py` (2 days, 6 distinct bugs) suggests insufficient test coverage** for the data loader stack. Stage 3a needs a real-data integration test gated on `AIVC_RUN_REAL_DATA_SMOKE=1` before merging. Already banked in feedback memory.
5. **Pathway pool matrix size** at (58, 36601) sparse with ~30K non-zeros is fine, but a future expansion to Reactome (~2000 pathways, ~10K genes) blows up to ~10M non-zeros — would need a different matrix structure. Defer until Phase 3c.

### 7.1 SDE fallback contingency (Stage 3b)

Neural ODE is the primary backbone choice. If it fails on Stage 3b QurieSeq training, the fallback is **latent SDE via `torchsde.sdeint_adjoint`**. Trigger conditions and switch procedure are pre-registered here so the decision isn't litigated mid-run.

**Trigger any one of**:
- **NaN loss** in training: gradient explosion or stiff ODE behavior. Single NaN is OK (auto-skip); >3 NaN batches in 100 → trigger.
- **Validation loss plateau >5 epochs** with training loss still decreasing: ODE is overfitting trajectory means while losing variance structure (deterministic bottleneck).
- **Eigenvalue drift** in the Jacobian `∂f/∂z` exceeding spectral radius 5.0 averaged over validation cells: indicates the ODE is becoming numerically unstable.
- **Trajectory variance collapse**: per-timepoint output variance across donors/perturbations drops below 0.1× the input variance — the deterministic ODE is averaging out biologically meaningful cell-to-cell heterogeneity.

**Switch procedure (no architecture change beyond `f_θ` wrapper)**:

```python
# Before (Neural ODE):
from torchdiffeq import odeint_adjoint
z_traj = odeint_adjoint(f_theta, z_dyn_0, t_grid, method='rk4', adjoint_method='dopri5')

# After (latent SDE):
import torchsde
class LatentSDE(torchsde.SDEIto):
    def __init__(self, f_theta, g_diffusion):
        self.f = f_theta             # reuse the trained drift function
        self.g = g_diffusion         # NEW: small diffusion network, init close to zero
    def f_step(self, t, y): return self.f(t, y)
    def g_step(self, t, y): return self.g(t, y) * 0.1  # bounded diffusion scale
z_traj = torchsde.sdeint_adjoint(
    LatentSDE(f_theta, g_diff), z_dyn_0, t_grid,
    method='euler', dt=1e-2
)
```

The drift function `f_θ` is reused as-is — only a small diffusion network `g_diffusion` is added (~50K params at d=256). Initialize `g` close to zero so the SDE starts near the ODE solution, then anneal upward. This preserves trained ODE state and avoids a from-scratch retrain.

**Dep**: `torchsde` is a separate pip install. Pre-stage BSC wheels for `manylinux2014_x86_64 + py3.11` before Stage 3b training kickoff (`pip download torchsde --platform manylinux2014_x86_64 --python-version 3.11`).

**Decision authority**: trigger conditions monitored automatically in training loop; fallback switch requires explicit human authorization (single bash flag flip `--use_sde`). Not auto-triggered.

If SDE fallback also fails, the next escalation is RSSM (discrete-time state-space model with stochastic latents), but that's a deeper architecture-class pivot and would require its own spec amendment.

---

## 8. Out of scope (explicit non-goals for Stage 3)

- **No RNA velocity / lineage tracing.** Stage 3 predicts within-cell-type perturbation response, not differentiation.
- **No causal inference (do-calculus, counterfactual).** The decomposed readout produces predictions, not causal estimates. Future work.
- **No subtype-resolution outputs** (B vs naive_B vs memory_B). Encoder is lineage-resolution. See `cross_corpus_pseudobulk_centroid_nn.md` for B-lineage caveat.
- **No multi-cell-type prediction in a single forward.** Stage 3a is CD4_T only (Mimitou); Stage 3b/3c will extend to PBMC lineages as QurieSeq covers them.
- **No drug repurposing screen interface.** That's a Stage 4 product layer; Stage 3 is the prediction engine.

---

## 9. Recommended next step

Author `aivc/skills/adapter.py` + `aivc/training/perturbation_loss.py::supcon_loss` + `scripts/train_stage3a_adapter.py` as the first PR. Estimated PR size: ~600 LOC + ~250 LOC tests. Expected duration: 2-3 days authoring + 1 day BSC training + 1 day eval.

Concurrently: author `scripts/build_pathway_pool_matrix.py` to convert `data/pathway_annotations/` outputs into the sparse pool matrix the decoders will consume.

Sequencing:
1. Day 1: adapter module + tests + pathway pool builder.
2. Day 2: perturbation loss + Mimitou dataloader + training script.
3. Day 3: zero-shot synergy eval script.
4. Day 4: BSC training (~24h H100).
5. Day 5: eval + validation report.
6. Day 6 (buffer): bug fixes + write-up.

---

## See also

- `project_aivc_stage3_part1_closure_2026_05_06.md` (Part 1 closure context + Report 2 verdict)
- `docs/specs/stage3_part1_prep_reports_2026_05_04.md` (Part 1 prep reports)
- `docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md` (canonical eval methodology)
- `data/pathway_annotations/` (pathway pool inputs)
- `/gpfs/scratch/ehpc748/quri020505/results/stage3_prep/mimitou_perturbation_probe.json` (Report 2 raw numerics)
- `reference_public_perturbation_data_landscape.md` (data constraint context)
