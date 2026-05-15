# Slide A4 — Temporal Dynamics via Neural ODE

- **Maps to Kinga's deck**: Slide 8 (5-layer regulatory cascade) — substantiates "Infer" layer's temporal modeling
- **Section**: A — Architecture Depth
- **Visual lead**: Continuous-time latent trajectory diagram aligned with QurieSeq sampling
- **Status**: Draft — pending Ash review

---

## Headline

**Cells respond on irregular timescales. The model handles that natively.**

(Alternative: *"Continuous-time dynamics from 0 to 180 minutes — without discretization."*)

(Alternative: *"Built for the way biology actually moves."*)

---

## Sub-headline (one line under headline)

A Neural ODE backbone evolves the cell state continuously through time. Sampling can be at any timepoint — 5 minutes for early signaling, 30 minutes for transcriptional response, 180 minutes for stable phenotypes. One model, irregular timepoints, no architectural compromises.

---

## Body content (3 bullets max)

- **Continuous-time state evolution**: The latent representation `z(t)` evolves according to a learned ordinary differential equation `dz/dt = f_θ(z, perturbation, t)`. Time is a first-class input, not a discrete index.

- **Matches QurieSeq Phase 1 design directly**: 5 timepoints across 0 → 180 min (0, 5, 30, 60, 180). The 5-minute point captures early signaling (phospho-ready), 30 minutes captures transcriptional onset, 180 minutes captures stable response. Neural ODE handles non-uniform spacing natively — no resampling, no interpolation hacks.

- **Architectural choice rationale**: Considered alternatives (latent SDE, RSSM, Transformer-over-timesteps) and rejected them. Neural ODE is the right primary choice for this regime; latent SDE remains a documented fallback if deterministic dynamics prove insufficient on QurieSeq data.

---

## Visual spec (the temporal dynamics diagram)

A two-part visual:

**Top — the trajectory plot:**

A continuous curve in latent space (showing `z(t)` over time) with 5 sample points marked:

```
   z(t)                    ←  continuous trajectory
    │
    │      ●5min
    │     ╱        ●30min
    │    ╱        ╱
    │   ╱        ╱       ●60min  ●180min
    │  ●0min    ╱       ╱       ╱
    │          ╱       ╱       ╱
    │
    └──────────────────────────────────→  time t
       0    5    30    60         180 min
                              
       ↑                              ↑
   Stimulus +                    Stable
   inhibitor                   phenotype
   applied
```

Annotation under the trajectory:

- **0 min**: baseline state (pre-perturbation)
- **5 min**: early signaling (phospho-active, RNA latent) — captured by Phase 2 phospho
- **30 min**: transcriptional onset (RNA dynamics begin)
- **60 min**: peak response window
- **180 min**: stable phenotype (RNA + Protein equilibrium)

**Bottom — the architectural choice rationale (3 cards):**

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Neural ODE          │  │ Latent SDE          │  │ Discrete Transformer│
│ (PRIMARY)           │  │ (FALLBACK)          │  │ (REJECTED)          │
│                     │  │                     │  │                     │
│ ✓ Continuous time   │  │ ✓ Stochastic        │  │ ✗ Fixed timesteps   │
│ ✓ Irregular sampling│  │   dynamics support  │  │ ✗ Interpolation     │
│ ✓ Deterministic     │  │ ✓ Triggers          │  │   artifacts         │
│   trajectories      │  │   documented        │  │ ✗ Architectural     │
│ ✓ ~130K params      │  │   (NaN >3/100,      │  │   invariant         │
│   (no extra cost    │  │   plateau >5 epochs,│  │   violation         │
│   over baseline)    │  │   variance collapse)│  │                     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## Notes for design

- **The trajectory plot is the slide.** Curve must look smooth and continuous, with 5 discrete sample points clearly highlighted on it. Visual metaphor: biology moves continuously; we sample it at chosen moments.
- **Color the 5 timepoints**: 0 min in neutral grey, 5 min in early-signaling color, 30/60/180 min progressively in transcription/protein-response colors. Reinforces the biological interpretation.
- **Architectural choice cards: PRIMARY = filled with brand accent; FALLBACK = outlined; REJECTED = greyed out with red X icons.** Visual hierarchy makes the choice obvious.
- **Don't show the ODE solver internals** (Euler, RK4, dopri5, etc.) — investor-irrelevant detail.

---

## Why this slide matters

Neural ODE is the architecture's answer to **"how does the platform handle time?"** Three concrete things it earns:

1. **Continuous-time claim**: Slide 37 in Kinga's deck calls the model "temporal." A4 is where that word gets technical substance.
2. **QurieSeq design alignment**: The 0/5/30/60/180 timepoint design isn't arbitrary — it's chosen because Neural ODE handles non-uniform spacing well. Slide makes the design-architecture coupling visible.
3. **Risk transparency**: Showing SDE as documented fallback (not just "we'll figure it out") signals engineering maturity to technical investors.

---

## Source data / claims

| Claim | Source |
|---|---|
| Neural ODE backbone (`dz/dt = f_θ`) | Architecture spec v1.1, §4 |
| QurieSeq Phase 1 timepoints 0/5/30/60/180 min | Thiago confirmation (Slack, May 12) |
| Latent SDE as documented fallback | Architecture spec v1.1, §7.1 (added in v1.1 amendment) |
| Trigger conditions for SDE fallback | Spec §7.1: NaN >3/100, validation plateau >5 epochs, Jacobian spectral radius >5.0, variance collapse |
| Architectural invariant: transformers rejected | Architecture spec v1.1, §2 |
| Implementation tool: `torchdiffeq` | Spec §4 |
| ~130K params (no extra cost over baseline) | Spec §3.2 (adapter sizing) |

---

## Speaker notes

**If asked: "Why not RNN or Transformer for time?"**

> Transformers and RNNs are discrete — they assume fixed timesteps. If we sampled at 0 and 5 minutes, then a Transformer effectively concatenates the two as adjacent tokens. That loses the information that 5 minutes is *fast* relative to the next gap (5→30 = 25 min) and *very fast* relative to 60→180 (120 min). Neural ODE represents the actual continuous trajectory, so non-uniform spacing is handled by integration, not by architectural workarounds.

**If asked: "Why is 5 minutes interesting if RNA changes slowly?"**

> Five minutes is where early signaling lives — phosphorylation cascades, second messengers, kinase activation. The phospho readouts from QurieSeq Phase 2 will populate that window with real biology. Even before phospho lands, the 5-minute sample gives the encoder a "what's already perturbed" reference point that's distinguishable from the 0-minute baseline.

**If asked: "What if Neural ODE training diverges?"**

> We have a documented fallback to latent SDE — same `f_θ` drift function reused, zero-initialized diffusion term, switching procedure pre-registered in the architecture spec. Trigger conditions include NaN loss frequency, validation plateau, and Jacobian spectral analysis. We don't need to discover that ODE failed mid-Stage-3b and panic — the fallback is planned and authorized.

**If asked: "Have you trained a Neural ODE on real biological data yet?"**

> Not yet — Stage 3a (current work) is the adapter on Mimitou single-endpoint CRISPR data. Neural ODE comes online in Stage 3b (Q3 2026) when QurieSeq Phase 1 time-course data lands. Until then, we're planning in-silico temporal sanity checks using synthetic dynamics to confirm ODE convergence and trajectory recovery before real data arrives. This de-risks the July go-live.

**If asked: "How does perturbation enter the ODE?"**

> The drift function `f_θ(z, p, t)` takes the perturbation embedding `p` (stimulus and/or inhibitor) as input. Different perturbations produce different trajectory curvatures in latent space. The decomposed readout (slide A3) handles how perturbation effects compose; the Neural ODE handles how they evolve over time.

---

## Investor framing (one-paragraph elevator)

> Cellular responses unfold continuously over time — phospho-signaling at minutes, transcription at tens of minutes, stable phenotypes at hours. Most deep learning approaches discretize time into fixed steps, losing information about non-uniform sampling. Our Neural ODE backbone evolves the cell state continuously, treating time as a first-class input rather than a discrete index. This is what lets the model handle QurieSeq's irregular timepoint design (0, 5, 30, 60, 180 minutes) natively — the early 5-minute sample captures signaling biology that a Transformer would either miss or distort. Neural ODE is our primary architectural choice with a documented latent SDE fallback if dynamics prove insufficient on real data.

---

## What's NOT on this slide (intentionally)

- ODE solver method choice (`dopri5` vs `rk4`)
- Adjoint training vs naive backprop
- Numerical stability hyperparameters
- Full SDE drift+diffusion equations (fallback is mentioned; details live in spec §7.1)
- Comparison to existing temporal foundation models (Genie, Geneformer, etc.)

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) for the trajectory plot; Cowork or Claude Code for the 3-card rationale block.

**File outputs**:
- `docs/deck/assets/diagrams/A4_trajectory.svg` (top — continuous latent trajectory with sample points)
- `docs/deck/assets/diagrams/A4_architecture_choice.svg` (bottom — 3 cards: primary/fallback/rejected)

Or combined: `A4_temporal_dynamics.svg` with stacked layout.

**Followup prompt for Cowork**:
"Generate `A4_temporal_dynamics.svg` per spec in `docs/deck/content/A4_temporal_neural_ode.md`. Top: 2D continuous trajectory plot showing latent state `z(t)` over time t∈[0, 180] min, with 5 marked sample points at t = 0/5/30/60/180. Bottom: 3 architecture choice cards — Neural ODE (PRIMARY, filled accent), Latent SDE (FALLBACK, outlined), Discrete Transformer (REJECTED, greyed with red X). Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- Neural ODE has never been trained on QurieSeq data because QurieSeq data doesn't exist yet. First real-data validation is Q3 2026.
- The "0.65–0.75 GREEN if CI includes 0.70" threshold from spec §5.1 applies to Stage 3a synergy eval, NOT Stage 3b temporal eval. Stage 3b thresholds are not yet pre-registered — that work happens before QurieSeq lands.
- Latent SDE fallback uses `torchsde.sdeint_adjoint`. Library is mature but less battle-tested than `torchdiffeq` for our exact use case.

---

## What's NEXT after A4 is committed

Move to **B1 (Cross-Corpus Validation Evidence)** — Section B starts. B1 deepens the 73% Calderon claim with the methodology rigor. Less novel than A3 architecturally, but it's the slide that proves the encoder is real. After B1, B2 (encoder probe) and B3 (BTK+JAK pre-demo) complete the validation section.
