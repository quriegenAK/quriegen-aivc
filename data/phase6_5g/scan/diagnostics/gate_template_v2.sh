#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Pre-merge 7-check gate template v2
# Origin: phase-6.5g.1 (2026-04-23) — fixes two defects from v1:
#   (a) check-7 used `grep -A 20 | head -2 | tail -1` which parsed the
#       WRONG line (the next YAML key, not the tripwires_fired value).
#       v2 uses inline-YAML regex: `grep -oE '^tripwires_fired: \[.*\]'`
#   (b) check-7c (ledger-clean) originally flagged any row with n_hits=-1
#       as a failure — but MANUAL-by-design probes legitimately use
#       n_hits=-1 with `url: MANUAL:...` prefix. v2 excludes them via
#       `$3 !~ /^MANUAL:/` before asserting failure.
# Usage: source this file in phase PRs; override EXPECTED_SPEC_SHA and
# PR_NUMBER per phase.
# ---------------------------------------------------------------------
set -u

PR_NUMBER="${PR_NUMBER:-20}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-phase-6.5g.1-scoping}"
EXPECTED_SPEC_SHA="${EXPECTED_SPEC_SHA:-48f6f6fbcaf2a1615b027b7cbcbf5d5326f3996f0da8e3172a1fb92558d02659}"
SPEC_PATH="${SPEC_PATH:-prompts/phase6_5g_1_k562_multiome_scoping.md}"
OUTCOME_YAML="${OUTCOME_YAML:-data/phase6_5g/scan/outcome_empty.yaml}"
LEDGER_TSV="${LEDGER_TSV:-data/phase6_5g/scan/scoping_ledger.tsv}"
THIAGO_MD="${THIAGO_MD:-data/phase6_5g/external_evidence/thiago_reply_2026-04-22.md}"

echo "=== 7-CHECK GATE v2 ==="

BR=$(git rev-parse --abbrev-ref HEAD)
[ "$BR" = "$EXPECTED_BRANCH" ] && echo "  [1/7] branch: OK" || echo "  [1/7] branch: FAIL ($BR)"

if command -v gh >/dev/null 2>&1; then
  CI=$(gh pr checks "$PR_NUMBER" --json state -q '.[].state' 2>/dev/null | sort -u | tr '\n' ',' || echo NONE)
  echo "  [2/7] CI: ${CI:-NONE}"
else
  echo "  [2/7] CI: gh missing"
fi

echo "  [3/7] tests: N/A (docs/evidence phase)"

SPEC_SHA=$(shasum -a 256 "$SPEC_PATH" | awk '{print $1}')
[ "$SPEC_SHA" = "$EXPECTED_SPEC_SHA" ] \
  && echo "  [4/7] spec SHA: OK" \
  || echo "  [4/7] spec SHA: FAIL ($SPEC_SHA)"

for f in "$OUTCOME_YAML" "$LEDGER_TSV" "$THIAGO_MD"; do
  [ -f "$f" ] && echo "  [5/7] anchor $f: OK" || echo "  [5/7] anchor $f: FAIL"
done

if command -v gh >/dev/null 2>&1; then
  BL=$(gh pr view "$PR_NUMBER" --json body -q '.body' 2>/dev/null | wc -c || echo 0)
  [ "${BL:-0}" -gt 500 ] && echo "  [6/7] PR body: OK (${BL})" || echo "  [6/7] PR body: FAIL (${BL:-0})"
fi

FIRED=$(grep -oE '^tripwires_fired: \[.*\]' "$OUTCOME_YAML" | sed 's/^tripwires_fired: //')
[ "$FIRED" = "[]" ] && echo "  [7a/7] tripwires clean: OK" \
  || echo "  [7a/7] tripwires clean: REVIEW ($FIRED)"

INCONSISTENT=$(python3 -c "
import yaml
d = yaml.safe_load(open('$OUTCOME_YAML'))
pc = d.get('probe_coverage', {})
bad = []
for label in ('rerun_2026_04_23_v2',):
    block = pc.get(label, {})
    for probe, entry in block.items():
        if entry.get('url') == 'FETCH_FAILED' or entry.get('n_hits', -1) < 0:
            bad.append(probe)
print(','.join(bad) if bad else 'OK')
")
[ "$INCONSISTENT" = "OK" ] && echo "  [7b/7] YAML self-consistency: OK" \
  || echo "  [7b/7] YAML self-consistency: FAIL ($INCONSISTENT)"

LEDGER_BAD=$(awk -F'\t' \
  'NR>1 && $3 !~ /^MANUAL:/ && ($3=="FETCH_FAILED" || $4=="-1") {print $1}' \
  "$LEDGER_TSV" | tr '\n' ',')
[ -z "$LEDGER_BAD" ] && echo "  [7c/7] ledger clean (automated rows): OK" \
  || echo "  [7c/7] ledger clean (automated rows): FAIL ($LEDGER_BAD)"

echo
echo "Gate v2 complete. Merge authorized iff 1-7a-7b-7c all PASS."
