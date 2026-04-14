# Claude Review Prompt for AIVC Architecture Documents

## How to Use

1. Open a **new Claude chat session** (claude.ai or Claude Code)
2. Paste the prompt below as your first message
3. Then attach or paste the architecture document content
4. Iterate on the feedback

---

## Prompt (copy everything below this line)

```
You are acting as three concurrent reviewers of a multi-omics AI platform architecture document. You must give independent, brutally honest assessments from each perspective. Do NOT be polite or agreeable — find real problems.

REVIEWER 1: Principal ML Systems Architect (ex-Google Brain / DeepMind)
- Focus: Is this actually buildable? Are the technology choices justified?
- Flag: Buzzword architecture (components named but no integration story), missing failure modes, unrealistic scale claims, technology mismatches
- Ask: "Could I hand this to a team of 5 engineers and have them build it in 6 months?"

REVIEWER 2: Computational Biologist (Broad Institute / Sanger level)
- Focus: Are the biological assumptions correct? Does the data model capture real experimental complexity?
- Flag: Oversimplified biology, missing metadata (batch effects, library prep, sequencing depth), incorrect causal claims, missing QC steps
- Ask: "Would I trust this system's predictions enough to design a wet-lab experiment around them?"

REVIEWER 3: Adversarial Red Team
- Focus: What's missing? What will break first? Where are the silent assumptions?
- Flag: Single points of failure, untested claims, circular logic, things that work at 10K cells but break at 10M
- Ask: "What's the first thing that will go wrong when real data hits this system?"

FORMAT YOUR RESPONSE AS:

## Reviewer 1: Systems Architecture
### What's Good (max 3 items)
### Critical Issues (ranked by severity)
### Missing Pieces
### Specific Recommendations (actionable, not vague)

## Reviewer 2: Biological Validity
### What's Good (max 3 items)
### Critical Issues (ranked by severity)
### Missing Pieces
### Specific Recommendations

## Reviewer 3: Red Team
### Attack Surface (what breaks first)
### Silent Assumptions (things assumed but never stated)
### Scale Traps (what fails at 100x current size)
### Recommended Stress Tests

## Consensus: Top 5 Action Items
(All three reviewers must agree these are the highest priority)

RULES:
- No hedging language ("might", "could consider", "it would be nice")
- Every issue must have a concrete fix, not just a complaint
- If you agree with a design choice, say WHY in one sentence, then move on
- Spend 80% of your output on problems, 20% on praise
- If the document claims something works but shows no evidence, call it out explicitly
- Compare against real systems: Geneformer, scGPT, CellOracle, CPA, GEARS, Insilico PandaOmics

Here is the architecture document to review:

[PASTE THE CONTENT OF AIVC_Architecture_Review.html OR AIVC_Architecture_Diagram.html HERE]
```

---

## Tips for Best Results

### Round 1: Architecture Review
Use the prompt above with `AIVC_Architecture_Review.html` content.

### Round 2: Code Review
For code-level feedback, use this follow-up in the same session:

```
Now review the actual implementation against the architecture you just critiqued. Here are the core source files. For each file, tell me:
1. Does it match the architecture document's claims?
2. What's the gap between documented design and actual code?
3. What would you change in the first refactoring sprint?

[PASTE KEY FILES: perturbation_model.py, losses.py, train_v11.py, fusion.py, scm.py]
```

### Round 3: ERD Stress Test
```
Here is the ERD. Attack it with these scenarios:
1. A QuRIE-seq experiment with 500K cells, 4 modalities, 30% missing ATAC
2. Merging data from 3 labs with different gene panels
3. A time-series perturbation (0h, 6h, 24h, 72h) with replicate donors
4. Retrospectively adding a 5th modality (spatial transcriptomics)

For each scenario, show me exactly which queries would be needed and where the schema breaks.
```

### Round 4: Implementation Roadmap
```
Given all the issues identified, create a 12-week implementation roadmap. Constraints:
- Team: 2 ML engineers, 1 bioinformatician, 1 platform engineer
- Must maintain Kang 2018 Pearson r >= 0.873 at all times
- First real QuRIE-seq 4-modality data expected at week 8
- Prioritize: what unblocks the most downstream work first?

Format: Week-by-week with specific deliverables, owners, and dependencies.
```
