# Pre-merge gate check-7 parser bug (post-mortem, 2026-04-23)

## Symptom
PROMPT D's check-7 shipped:
    grep -A 20 "tripwires_fired:" outcome_empty.yaml | head -2 | tail -1 | tr -d ' '

`head -2 | tail -1` grabs the line AFTER `tripwires_fired:`, which is the
next YAML key (`tripwires_not_fired:`), not the value of tripwires_fired.
Result: gate reports REVIEW even when fired list is [] (inline YAML).

## Correct form (for future gate templates)
    FIRED=$(grep -oE '^tripwires_fired: \[.*\]' outcome_empty.yaml | sed 's/^tripwires_fired: //')
    [ "$FIRED" = "[]" ] && echo OK || echo "REVIEW ($FIRED)"

Or, robustly:
    FIRED=$(python3 -c "import yaml; print(yaml.safe_load(open('outcome_empty.yaml'))['tripwires_fired'])")

## Impact
Outcome unaffected — tripwires_fired is genuinely []. Script-only defect.
Update feedback_premerge_gate_template.md with corrected check-7 form.
