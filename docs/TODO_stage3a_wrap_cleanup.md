# Stage 3a Wrap — Repository Cleanup Task

**Owner**: Cowork
**Estimated effort**: 1 day focused
**Trigger**: After Stage 3a Day 5-6 completion (training script + eval + first real-data smoke green on BSC)
**Why deferred**: Mid-Stage-3a is critical path; cleanup at phase boundary is safer

## Scope

1. **Repository hygiene audit**
   - Root-level cruft: model_*.pt files, AIVC_Architecture_*.html, REVIEW_PROMPT.md, REFERENCE_REPOS_EVALUATION.md
   - Lock files: ~$QurieGen_AIVC_GeneLink_Deck.pptx
   - Duplicate or stale directories: aivc/, aivc_platform/, aivc_env/, agents/
   - Old experiment artifacts: WEEK4_RESULTS.md, HPC_GAP_ANALYSIS.md
   - Stale ship scripts from completed phases (~/_stage3_part1_*.sh)

2. **Formalize conventions**
   - Memory directory: docs/memory/ (established this week)
   - Strategy docs: docs/strategy/
   - Spec docs: docs/specs/ (existing)
   - Closure reports: docs/reports/ (existg)
   - Eval methodology: docs/eval_methodology/ (existing)
   - Feedback bank: docs/feedback/ (currently scattered)
   - TODO/backlog: docs/TODO_*.md (this file)
   
3. **Repository structure spec**
   - Write docs/REPO_STRUCTURE.md documenting top-level layout
   - Document where each file type belongs
   - Add to CLAUDE.md as guidance for future sessions

4. **gitignore hardening**
   - Audit current .gitignore
   - Add patterns for ~$*.pptx lock files, *.DS_Store, __pycache__
   - Confirm no checkpoints/data leaking through

5. **Pre-flight protections**
   - Add a `scripts/check_repo_hygiene.py` that fails CI if root-level model_*.pt files exist
   - Add doc-lint to PR checks (specs in docs/specs/, not in root)

## Out of Scope

- Code refactoring (separate task)
- aivc/ package restructuring (separate task)
- Test reorganization (separate task)

## Deliverables

1. PR: cleaned root directory, files moved or archived
2. docs/REPO_STRUCTURE.md spec
3. Updated .gitignore
4. Updated CLAUDE.md with repo conventions
5. scripts/check_repo_hygiene.py with CI integration

## Why This Matters

Repo entropy raises onboarding cost for new team members (Series A hire) and increases bug surface area (e.g., scripts pointing to stale paths). Cleanup at phase boundaries is cheaper than continuous cleanup mid-execution.

## Status

DEFERRED — created 2026-05-11, trigger on Stage 3a Day 5-6 completion.
