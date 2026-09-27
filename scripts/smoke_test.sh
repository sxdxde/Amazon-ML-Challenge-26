#!/usr/bin/env bash
# Smoke test: fresh venv -> unit tests -> tiny CPU pipeline -> output checks -> full-size writer check.
# Run from anywhere; it cds to the directory that contains dataset/ and utils/.
#   bash scripts/smoke_test.sh            # fresh venv every time
#   SMOKE_REUSE_VENV=1 bash scripts/smoke_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PKG="code/business_entity_resolution"
VENV="$ROOT/.venv_smoke"
PY_BIN="${PYTHON:-python3}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"

step() { echo; echo "================ $* ================"; }

step "1. fresh virtualenv + pinned requirements (CPU torch)"
if [[ "${SMOKE_REUSE_VENV:-0}" != "1" || ! -x "$VENV/bin/python" ]]; then
  rm -rf "$VENV"
  "$PY_BIN" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  if [[ "$(uname)" == "Linux" ]]; then
    "$VENV/bin/pip" install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
  fi
  "$VENV/bin/pip" install -q -r "$PKG/requirements.txt"
fi
PY="$VENV/bin/python"
"$PY" -c "import sys, torch, lightgbm, rapidfuzz, sparse_dot_topn, transformers; print('python', sys.version.split()[0], '| torch', torch.__version__)"

step "2. unit tests (metric, expected-F0.5 decision rule, normalisation)"
"$PY" "$PKG/tests/test_metric.py"
"$PY" "$PKG/tests/test_decide.py"
"$PY" "$PKG/tests/test_normalize.py"

step "3. pipeline --smoke on CPU"
START=$(date +%s)
"$PY" "$PKG/src/run_all.py" --smoke --force --device cpu
ELAPSED=$(( $(date +%s) - START ))
echo "smoke pipeline wall time: ${ELAPSED}s"

step "4. smoke outputs exist and are well-formed"
test -s output_smoke/matching_results.tsv
test -s output_smoke/candidate_pairs.tsv
"$PY" utils/validate_submission.py --matching output_smoke/matching_results.tsv \
  --candidate output_smoke/candidate_pairs.tsv --test-dir artifacts/smoke/prepare/test_dir --check-ids | tee /tmp/ber_smoke_val.txt
grep -q "^PASS" /tmp/ber_smoke_val.txt
"$PY" - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "code/business_entity_resolution/src")
from io_utils import check_submission, read_tsv
d = Path("artifacts/smoke/prepare/test_dir")
req = set(read_tsv(d / "test_source1.tsv")["entity_id"])
valid = set(read_tsv(d / "test_source2.tsv")["entity_id"]) | set(read_tsv(d / "test_source3.tsv")["entity_id"])
errs = check_submission(Path("output_smoke/matching_results.tsv"), Path("output_smoke/candidate_pairs.tsv"), req, valid)
assert not errs, errs
countries = set(read_tsv(d / "test_source1.tsv")["country"])
print("internal strict check OK (matches ⊆ candidates, one row per S1); test countries:", sorted(countries))
EOF

step "5. full-size dummy submission through the writer path (all rows empty)"
DUMMY="output_smoke/dummy_full"
mkdir -p "$DUMMY"
"$PY" -c "
import sys; from pathlib import Path
sys.path.insert(0, 'code/business_entity_resolution/src')
from io_utils import write_empty_submission
m, c = write_empty_submission(Path('dataset/test/test_source1.tsv'), Path('$DUMMY'))
print('wrote', m, c)"
"$PY" utils/validate_submission.py --matching "$DUMMY/matching_results.tsv" \
  --candidate "$DUMMY/candidate_pairs.tsv" --test-dir dataset/test | tee /tmp/ber_dummy_val.txt
grep -q "^PASS" /tmp/ber_dummy_val.txt

step "6. smoke CV score vs all-empty baseline"
"$PY" - <<'EOF'
import json
c = json.load(open("artifacts/smoke/decide/choice.json"))
best = c["results"][c["variant"]]["all"]
baseline = best["singleton_rate"]  # predicting empty everywhere scores 1 on singletons, 0 elsewhere
print(f"smoke OOF macro F0.5 = {best['f05']:.4f} (variant {c['variant']}); all-empty baseline = {baseline:.4f}")
assert best["f05"] > baseline, "smoke score not above the all-empty baseline"
EOF
echo
echo "SMOKE TEST PASSED (pipeline ${ELAPSED}s)"
