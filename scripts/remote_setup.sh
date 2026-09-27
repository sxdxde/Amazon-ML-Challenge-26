#!/usr/bin/env bash
# Remote (1x A100 40GB) setup: venv, CUDA torch, pinned requirements, GPU checks, model pre-download.
#   bash scripts/remote_setup.sh
# Env overrides: PYTHON=python3.11  TORCH_CUDA=cu118|cu126|cu128 (auto-detected from the driver)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PKG="code/business_entity_resolution"
PY_BIN="${PYTHON:-python3}"
# CUDA wheel: cu126 needs NVIDIA driver >= 525; older drivers (e.g. 470 / CUDA 11.4) use cu118.
if [[ -z "${TORCH_CUDA:-}" ]]; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
  if [[ -n "$DRV" && "$DRV" -lt 525 ]]; then TORCH_CUDA=cu118; else TORCH_CUDA=cu126; fi
  echo "NVIDIA driver major: ${DRV:-unknown} -> torch wheel ${TORCH_CUDA}"
fi
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"

"$PY_BIN" - <<'EOF'
import sys
assert sys.version_info >= (3, 10), f"Python >= 3.10 required, found {sys.version}"
print("python", sys.version.split()[0])
EOF

echo "== venv"
rm -rf .venv
"$PY_BIN" -m venv .venv

if [[ -d offline_wheels ]]; then
  # ---- fully offline install from the bundle made by scripts/fetch_offline_bundle.sh
  echo "== OFFLINE install from offline_wheels/ ($(ls offline_wheels | wc -l) files)"
  # exact pins resolved for Linux x86_64 / glibc 2.17 / Python 3.10 by fetch_offline_bundle.sh
  .venv/bin/pip install --no-index --find-links offline_wheels -r offline_wheels/requirements.lock.txt
else
  .venv/bin/pip install -q --upgrade pip
  # glibc decides which binary wheels can install (CentOS/RHEL 7 = glibc 2.17)
  GLIBC="$(.venv/bin/python -c 'import platform; print(platform.libc_ver()[1] or "0")')"
  NEW_GLIBC="$(.venv/bin/python -c "print(int(tuple(map(int, '$GLIBC'.split('.')[:2])) >= (2, 28)))")"
  if [[ "$NEW_GLIBC" == "1" ]]; then TORCH_VER=2.7.1; else TORCH_VER=2.6.0; fi
  echo "== glibc $GLIBC -> torch $TORCH_VER ($TORCH_CUDA)"
  .venv/bin/pip install "torch==${TORCH_VER}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
  echo "== pinned requirements (without torch / lightgbm)"
  grep -v -E '^(torch|lightgbm)==' "$PKG/requirements.txt" > /tmp/ber_reqs_$$.txt
  .venv/bin/pip install -r /tmp/ber_reqs_$$.txt
  rm -f /tmp/ber_reqs_$$.txt
  echo "== gradient boosting: LightGBM (wheel needs glibc >= 2.28, else source build), fallback XGBoost"
  if .venv/bin/pip install "lightgbm==4.6.0" && .venv/bin/python -c "import lightgbm"; then
    echo "LightGBM OK"
  else
    echo "LightGBM unavailable on this system -> installing XGBoost 2.1.4 (Apache-2.0)"
    .venv/bin/pip uninstall -y lightgbm >/dev/null 2>&1 || true
    .venv/bin/pip install "xgboost==2.1.4"
  fi
fi
# Note: dense kNN is exact torch matmul on the GPU, so faiss is not required.

echo "== GPU"
nvidia-smi || echo "WARNING: nvidia-smi not available"
.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}, {p.total_memory/1e9:.1f} GB, bf16: {torch.cuda.is_bf16_supported()}")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); print("matmul ok:", float((x @ x).float().abs().mean()))
else:
    print("WARNING: CUDA not available — the full run would fall back to CPU (very slow)")
EOF

echo "== model weights -> $HF_HOME"
.venv/bin/python scripts/download_models.py

touch .venv/.setup_complete
echo "== resources"
df -h "$ROOT" | tail -1
(free -g 2>/dev/null || vm_stat) | head -3
nproc 2>/dev/null || sysctl -n hw.ncpu
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
echo
echo "Setup complete. Recommended: >=64 GB RAM, >=120 GB free disk (embeddings ~50 GB fp16)."
echo "Next: bash scripts/run_full.sh"
