#!/bin/bash
set -euo pipefail
PY=/path/to/amazon_ml_2026/.venv/bin/python
S3B=/path/to/amazon_ml_2026/runs/stage3b
PS=/path/to/amazon_ml_2026/runs/pseudo
echo "$(date +%T) LOCO US->India"
cd $S3B && $PY loco_pseudo_usin.py > loco_usin.out 2>&1
D=$($PY -c "import json;print(json.load(open(\"$S3B/loco_pseudo.json\"))[-1][\"delta\"])")
echo "$(date +%T) US->India pseudo delta = $D"
if $PY -c "import sys; sys.exit(0 if float(\"$D\") > 0 else 1)"; then
  echo "$(date +%T) positive in both directions -> France gbm2 pseudo-label retrain"
  cd $PS && $PY gbm2_pseudo.py 0.97 0.03 > gbm2_pseudo.out 2>&1
  echo "$(date +%T) gbm2 pseudo done -> decide + predict"
  cd /path/to/amazon_ml_2026/code/business_entity_resolution && $PY src/run_all.py --profile full --from-stage decide --force --device auto --root $PS --data-dir /path/to/amazon_ml_2026/dataset --log-file $PS/decide_predict.log
  echo "$(date +%T) CHAIN DONE"
else
  echo "$(date +%T) pseudo-labeling did NOT help US->India -> stopping (no France retrain)"
fi
