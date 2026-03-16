#!/bin/bash
# run_all_tests.sh — Full regression suite for AIVC GeneLink
#
# Tests 1-4: Week 1 (link prediction)
# Tests 5-7: Week 2 (perturbation prediction)

source aivc_env/bin/activate

PASS=0
FAIL=0
TOTAL=7

echo "================================================================"
echo "  AIVC GeneLink — Full Regression Suite"
echo "================================================================"
echo ""

# --- Test 1: Sanity ---
echo "=== TEST 1: SANITY ==="
python3 test_sanity.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Test 2: JAK-STAT Biology ---
echo "=== TEST 2: JAK-STAT BIOLOGY ==="
python3 test_jakstat.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Test 3: Model Inference ---
echo "=== TEST 3: MODEL INFERENCE ==="
python3 test_model.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Test 4: AUC Validation ---
echo "=== TEST 4: AUC VALIDATION ==="
python3 test_auc.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Test 5: JAK-STAT Fixed Validation ---
echo "=== TEST 5: JAK-STAT FIXED VALIDATION ==="
python3 test_jakstat_fixed.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Test 6: Full Benchmark Evaluation ---
echo "=== TEST 6: FULL BENCHMARK EVALUATION ==="
python3 evaluate_model.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Test 7: Regression Guard ---
echo "=== TEST 7: REGRESSION GUARD ==="
python3 test_regression.py
if [ $? -eq 0 ]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Summary ---
echo "================================================================"
echo "  RESULTS: $PASS/$TOTAL PASSED, $FAIL/$TOTAL FAILED"
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    echo "  STATUS: SOME TESTS FAILED"
    exit 1
else
    echo "  STATUS: ALL TESTS PASSED"
    exit 0
fi
