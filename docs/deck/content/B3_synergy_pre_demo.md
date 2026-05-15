# Slide B3 — Mechanism Validation: Zero-Shot Synergy on Public Data

- **Maps to Kinga's deck**: Bridges to Aduro story (slides 3-4) and BTK+JAK clinical evidence — answers "why we believe the BTK+JAK demo will work"
- **Section**: B — Validation Evidence (closing slide)
- **Visual lead**: Substitute-experiment diagram + held-out arm protocol
- **Status**: Draft — pending Ash review

---

## Headline

**The synergy mechanism validates on public data — before BTK+JAK runs on our own.**

(Alternative: *"Zero-shot synergy on Mimitou is the dress rehearsal for BTK+JAK on QurieSeq."*)

(Alternative: *"We test the mechanism today. We apply it to BTK+JAK in Q3 2026."*)

---

## Sub-headline (one line under headline)

No public dataset has BTK+JAK perturbations — but the architectural mechanism is the same. We use CD3E+CD4 double-knockout as the public-data substitute, hold it out during training, and predict its response zero-shot. Same architecture, same zero-arm constraint, same compositional generalization.

---

## Body content (3 bullets max)

- **The substitution rationale**: BTK+JAK combinations don't exist in any public PBMC dataset (confirmed across Mimitou, Parse, CIPHER-seq, Soskic). CD3E+CD4 in Mimitou is the mathematically equivalent test — a perturbation pair where each single arm is in training and the combination is held out. **Same architectural mechanism, available data**.

- **The dress-rehearsal protocol**: Train the adapter on Mimitou single-KO arms (CD3E alone, CD4 alone, ZAP70, NFKB2) — never see the CD3E+CD4 double KO during training. At test time, predict the double-KO response with the synergy head active. Pre-registered Stage 3a target: ≥0.70 zero-shot synergy accuracy.

- **From mechanism to QurieSeq BTK+JAK demo**: When QurieSeq Phase 1 lands (Q3 2026, confirmed by Thiago) with BTK inhibitor, JAK inhibitor, and BTK+JAK combo conditions, the *exact same architecture* trains on single-arm data and predicts combination zero-shot. Mimitou validates the mechanism on cell-state perturbations. QurieSeq applies it to clinically relevant drug combinations.

---

## Visual spec (the substitute-experiment diagram)

A three-part visual stacked vertically:

**Top — The mathematical substitution:**

```
Public-data substitute (Mimitou CRISPR)              QurieSeq Phase 1 (Q3 2026)
────────────────────────────────────────              ──────────────────────────

Training arms (seen):                                 Training arms (will be seen):
  • CD3E single KO            (seen by adapter)        • BTK inhibitor alone        (seen)
  • CD4 single KO             (seen by adapter)        • JAK inhibitor alone        (seen)
  • ZAP70 single KO           (seen by adapter)        • Other inhibitor singles    (seen)
  • NFKB2 single KO           (seen by adapter)        • All 4-arm controls         (seen)
  • NTC (no perturbation)     (seen by adapter)

Held-out arm:                                         Held-out arm:
  ◆ CD3E + CD4 double KO   ◀── PREDICT ──▶            ◆ BTK + JAK combo  ◀── PREDICT ──▶
    (architecturally equivalent                          (the clinical demo target,
     to the BTK+JAK test)                                Ibrutinib + Ruxolitinib basis)
```

**Middle — Why CD3E+CD4 is the right substitute (3-card row):**

```
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│ ARCHITECTURE            │  │ DATA                    │  │ MECHANISM               │
│                         │  │                         │  │                         │
│ Same 4-arm decomposed   │  │ Mimitou double-KO has   │  │ Each single perturbation│
│ readout                 │  │ 74 cells post-split —   │  │ alters TCR signaling;   │
│                         │  │ sufficient for           │  │ the double KO produces  │
│ Same zero-arm L2        │  │ pre-registered eval     │  │ a non-additive          │
│ constraint              │  │ (Stage 3a target ≥0.70  │  │ phenotype the synergy   │
│                         │  │ with bootstrap CI)      │  │ head must learn         │
│ Same synergy head       │  │                         │  │                         │
│ trained only on residual│  │                         │  │                         │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

**Bottom — The connection to clinical BTK+JAK evidence:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Clinical grounding for the eventual BTK+JAK demo                         │
│                                                                          │
│ • Ibrutinib (BTK inhibitor) + Ruxolitinib (JAK1/2 inhibitor)             │
│ • CLL trial: NCT02912754 (Phase Ib/II)                                   │
│ • Published rationale: PMID 26819050 (Maddocks 2016)                     │
│ • Thiago wet-lab finding: pJAK1 unexpectedly active in BCR pathway       │
│   → biological rationale for the combination beyond pathway              │
│   independence                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Notes for design

- **The substitution diagram (top) is the slide's center of gravity.** Two parallel columns, structurally identical — the only difference is "Mimitou now vs QurieSeq Q3 2026". This makes the mechanism transfer visually obvious.
- **Use the same accent color** for the held-out arm in both columns (e.g., the synergy color from A3). Visual continuity of architecture from public data to proprietary data.
- **The middle 3-card row** should be subdued visually — supporting evidence, not new claims.
- **Clinical grounding box at bottom** should look like a footer reference — small text, lots of citation, signals "this is documented external evidence."
- **No bar charts on this slide.** B2 had the numerical bar chart; B3 is structural/conceptual.

---

## Why this slide matters

This is the slide that **converts the 0.57 number into clinical relevance**.

Without B3:
- The 0.57 from B2 is a generic architectural result
- BTK+JAK on C2 sounds disconnected from anything we've actually run
- Investors wonder how we get from "encoder works on CRISPR" to "we predict drug combinations"

With B3:
- 0.57 maps to a specific architectural mechanism that's the same mechanism BTK+JAK will use
- The bridge from public-data validation to QurieSeq Phase 1 is structural, not aspirational
- Clinical context (Ibrutinib+Ruxolitinib CLL trial, pJAK1/BCR finding) makes the demo target medically grounded

---

## Source data / claims

| Claim | Source |
|---|---|
| No public dataset has BTK+JAK perturbations | Stage 3 Part 1 Report 4 (BTK+JAK feasibility study) |
| CD3E+CD4 as architectural substitute | Stage 3 Part 1 Report 4 + Stage 3 Part 1 verdict |
| 74-cell CD3E+CD4 double KO post-split | `docs/memory/project_aivc_stage3_part1_verdict_2026_05_11.md` |
| Pre-registered Stage 3a target ≥0.70 zero-shot synergy | Architecture spec v1.1, §5.1 |
| Same 4-arm decomposed readout, same zero-arm L2 | Architecture spec v1.1, §3.2 (B3 → A3 cross-reference) |
| BTK+JAK combo confirmed in QurieSeq Phase 1 | Thiago confirmation, May 12 |
| Ibrutinib + Ruxolitinib CLL trial NCT02912754 | ClinicalTrials.gov |
| Phase Ib/II rationale PMID 26819050 | Maddocks 2016, Blood |
| pJAK1 unexpectedly in BCR pathway | Thiago wet-lab finding (internal, cited as Quriegen IP) |

---

## Speaker notes

**If asked: "Why didn't you find a public dataset with BTK+JAK?"**

> We did exhaustive search across PubMed, GEO, scperturb.org, PerturBase, PerturbSeq.db, and 8 years of literature. BTK+JAK pharmacological perturbations don't exist in any public PBMC dataset. The closest are CRISPR knockouts of upstream pathway components (e.g., BTK knockout in B-cell lines) but not the inhibitor combination. The structural reason is that pharmacological combination screening in primary PBMCs is hard and expensive — it's exactly what QurieSeq is built to produce. The absence in public data is itself evidence that our proprietary data is the moat.

**If asked: "How is CD3E+CD4 the right substitute? They're not drug combinations."**

> The substitute is at the architectural level, not the biological level. The architecture treats every perturbation pair identically — stim plus inh plus combo. Whether the perturbations are genetic (CRISPR) or pharmacological (drugs) doesn't change the math. CD3E+CD4 in Mimitou gives us a real held-out double-perturbation arm to test the synergy head's compositional generalization. If the architecture passes that test, it has earned the right to apply the same mechanism to BTK+JAK on QurieSeq.

**If asked: "What's the pre-registered ≥0.70 threshold based on?"**

> Two principles. First, 0.70 is the architectural threshold above which synergy prediction is "useful enough" for downstream applications (target discovery, combination prioritization). Second, it's far enough above chance (0.25) and the encoder baseline (0.57 from B2) to demonstrate the adapter is materially adding capability — not just propagating the encoder's existing signal. The interpretation rules account for the small sample size: bootstrap CI must include 0.70 for a GREEN call, otherwise it's AMBER with a documented re-run protocol.

**If asked: "What's the connection to the Ibrutinib+Ruxolitinib clinical trial?"**

> Ibrutinib is a BTK inhibitor. Ruxolitinib is a JAK1/2 inhibitor. Their combination has been tested in CLL (Phase Ib/II, NCT02912754) with published rationale (Maddocks 2016, Blood). What makes this combination particularly relevant for us is Thiago's wet-lab finding that pJAK1 — phosphorylated JAK1 — appears in the BCR signaling pathway, which is biologically surprising and provides a non-redundant mechanistic basis for combining BTK and JAK inhibition. Our QurieSeq Phase 1 design includes both inhibitors and their combination, so we can predict the combo response zero-shot and validate against measured data.

**If asked: "When will we know if the dress rehearsal works?"**

> Stage 3a training on Mimitou perturbations runs in May 2026 on BSC GPUs (Day 4-5 of the implementation plan). The zero-shot CD3E+CD4 synergy eval runs immediately after. Results will be available within ~1 week. If we hit the pre-registered ≥0.70 GREEN threshold, the architecture is validated and we proceed to BTK+JAK on QurieSeq with high confidence. If we land in the AMBER zone (0.55-0.65 with CI logic), we have documented remediation paths — adjust λ_zero on the zero-arm constraint, or move to a hard projection. The pre-registered interpretation logic prevents post-hoc panic.

---

## Investor framing (one-paragraph elevator)

> We can't run BTK+JAK on public data because no public dataset has that combination — its absence is itself evidence that QurieSeq is the right moat. But the architectural mechanism we need for BTK+JAK is the same mechanism we can validate on public data. CD3E+CD4 in Mimitou's CRISPR screen gives us a real held-out double-perturbation arm: train on single knockouts, hold out the double, predict it zero-shot. Same 4-arm decomposed readout, same zero-arm L2 constraint, same synergy head. Pre-registered Stage 3a target is ≥0.70 zero-shot synergy accuracy. This is the dress rehearsal. The actual BTK+JAK demo runs in Q3 2026 on our own QurieSeq data, with clinical grounding in the Ibrutinib+Ruxolitinib CLL trial and a non-obvious biological rationale from Thiago's pJAK1 finding.

---

## What's NOT on this slide (intentionally)

- Stage 3a actual training results (will land before deck v2; currently shown as "target")
- QurieSeq Phase 1 experimental design details (lives in C1)
- BTK+JAK demo execution plan (lives in C2)
- Full clinical rationale of the BTK+JAK combination (lives in clinical-context note within speaker notes)
- The 0.57 number itself — already on B2, not repeated

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — 3-zone vertical layout.

**File output**: `docs/deck/assets/diagrams/B3_mechanism_pre_demo.svg`

**Followup prompt for Cowork** (when ready):
"Generate `B3_mechanism_pre_demo.svg` per spec in `docs/deck/content/B3_synergy_pre_demo.md`. Top zone: two parallel columns labeled 'Public-data substitute (Mimitou)' and 'QurieSeq Phase 1 (Q3 2026)' — each column showing training arms (single perturbations, listed) above and held-out arm (double perturbation, highlighted in synergy accent color) below, with 'PREDICT' arrow indicator. Middle zone: 3-card row labeled ARCHITECTURE / DATA / MECHANISM with substitute justification. Bottom zone: clinical context footer (Ibrutinib+Ruxolitinib NCT02912754, PMID 26819050, pJAK1 wet-lab finding). Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- The 0.68 CD3E+CD4 number from B2 is a *probe* result (encoder + simple readout). The *adapter-trained* result is what we'll have in ~1 week. If the adapter-trained number doesn't materially exceed the 0.68 probe number, the architecture's adapter strategy is in question — banked Stage 3a Day 5-6 wrap.
- BTK+JAK clinical extrapolation depends on QurieSeq Phase 1 wet-lab execution (donor recruitment, sequencing quality, inhibitor batch reliability). If Phase 1 produces low-quality data, B3 to C2 storytelling is structurally weakened.
- The Maddocks 2016 PMID and NCT02912754 should be hand-verified before final deck commit — Cowork can verify or one of us reads the abstract.

---

## What's NEXT after B3 is committed

Section B complete. Move to **Section C — QurieSeq Phase 1**.
- **C1**: Phase 1 Experimental Design (5 donors, 5 timepoints, 4-arm structure, modality grid)
- **C2**: BTK+JAK Headline Demo Plan (pre-registered eval, what we measure, when)
