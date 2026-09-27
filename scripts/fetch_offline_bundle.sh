#!/usr/bin/env bash
# Run on a machine WITH normal internet (your Mac). Downloads everything the remote needs so that it
# can install and run fully OFFLINE (e.g. behind a firewall that blocks download.pytorch.org):
#   offline_wheels/  Linux x86_64 wheels for Python 3.10, glibc >= 2.17 (CentOS 7 compatible):
#                    torch 2.6.0+cu118 (+ its CUDA 11.8 runtime wheels), pinned requirements, xgboost
#   hf_cache/        model weights (bge-m3, bge-reranker-v2-m3, smoke-test models)
# Then: bash scripts/upload_to_remote.sh   (uploads both folders; remote setup detects them)
#   PYVER=3.10 TORCH_VER=2.6.0 TORCH_CUDA=cu118 bash scripts/fetch_offline_bundle.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PKG="code/business_entity_resolution"
PYVER="${PYVER:-3.10}"
TORCH_VER="${TORCH_VER:-2.6.0}"
TORCH_CUDA="${TORCH_CUDA:-cu118}"
PY="${PY:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
OUT="offline_wheels"
mkdir -p "$OUT"
PLAT=(--platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2010_x86_64
      --platform manylinux1_x86_64 --platform linux_x86_64 --python-version "$PYVER"
      --implementation cp --only-binary=:all:)

# `pip download --platform` evaluates environment markers for THIS machine (macOS), silently dropping
# Linux-only deps (NVIDIA CUDA wheels, triton) and picking the wrong scipy line. So resolve the full
# dependency set for the TARGET (Linux x86_64, glibc 2.17, Python $PYVER) with uv, then download
# exactly those pins with --no-deps.
UV="$(command -v uv || echo "$PY -m uv")"
REQ_IN="$(mktemp)"; LOCK="offline_wheels/requirements.lock.txt"
echo "torch==${TORCH_VER}+${TORCH_CUDA}" > "$REQ_IN"
grep -v -E '^(torch|lightgbm)==' "$PKG/requirements.txt" | grep -v '^#' | grep -v '^$' >> "$REQ_IN"
echo "xgboost==2.1.4" >> "$REQ_IN"
echo "== resolving for x86_64-manylinux_2_17 / Python ${PYVER}"
$UV pip compile "$REQ_IN" -o "$LOCK" --python-version "$PYVER" --python-platform x86_64-manylinux_2_17 \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --no-header --no-annotate
rm -f "$REQ_IN"
echo "resolved $(grep -c '==' "$LOCK") packages:"; grep '==' "$LOCK" | tr '\n' ' '; echo
echo "== downloading wheels"
"$PY" -m pip download -r "$LOCK" --no-deps "${PLAT[@]}" -d "$OUT" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" --extra-index-url https://pypi.org/simple
echo "wheels: $(ls "$OUT" | wc -l) files, $(du -sh "$OUT" | cut -f1)"

echo "== model weights -> hf_cache/"
HF_HOME="$ROOT/hf_cache" HF_HUB_DISABLE_XET=1 "$PY" scripts/download_models.py
du -sh hf_cache
echo
echo "OFFLINE BUNDLE READY. Next:  bash scripts/upload_to_remote.sh"
