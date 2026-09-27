#!/bin/bash
set -euo pipefail
ROM=/path/to/amazon_ml_2026/runs/rom
PROD=/path/to/amazon_ml_2026
PY=$PROD/.venv/bin/python
CODE=$PROD/code/business_entity_resolution
echo "$(date +%T) waiting for embed_norm to finish"
while pgrep -f "scripts/embed_norm.py" >/dev/null; do sleep 60; done
[ -f $ROM/artifacts/full/embed/_DONE.json ] || { echo "$(date +%T) embed_norm did NOT verify -> stopping"; exit 1; }
echo "$(date +%T) embeddings verified -> dense-only reblock (train,test)"
cd $ROM && $PY scripts/fast_reblock.py --src-root $PROD --emb-root $ROM --out-root $ROM --splits train,test
[ -f $ROM/artifacts/full/block/_DONE.json ] || { echo "$(date +%T) reblock did not complete -> stopping"; exit 1; }
echo "$(date +%T) reblock done -> run_all from features (big xenc, wide band)"
cd $CODE && $PY src/run_all.py --profile full --from-stage features --force --device auto \
  --root $ROM --data-dir $PROD/dataset \
  --set xenc_band_lo=0.01 xenc_band_hi=0.998 xenc_max_per_s1=8 xenc_batch=24 xenc_grad_accum=3 xenc_infer_batch=384 \
  --log-file $ROM/logs/pipeline_rom.log
echo "$(date +%T) CHAIN FINISHED"
