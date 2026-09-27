#!/usr/bin/env bash
# Build <team_name>_submission.zip = output/ + code/business_entity_resolution/ + Documentation_template.md
#   bash scripts/package_submission.sh <team_name>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TEAM="${1:?usage: package_submission.sh <team_name>}"
PY="${PY:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"

for f in output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 1; }
done
"$PY" utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test | tee /tmp/ber_pkg_val.txt
grep -q "^PASS" /tmp/ber_pkg_val.txt || { echo "validator did not PASS — not packaging"; exit 1; }

STAGE="$(mktemp -d)"
mkdir -p "$STAGE/output" "$STAGE/code"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  code/business_entity_resolution "$STAGE/code/"
cp Documentation_template.md "$STAGE/"
ZIP="$ROOT/${TEAM}_submission.zip"
rm -f "$ZIP"
(cd "$STAGE" && zip -qr "$ZIP" output code Documentation_template.md)
rm -rf "$STAGE"
echo "built $ZIP"
unzip -l "$ZIP" | tail -n +1 | head -40
