#!/usr/bin/env bash
# Run on the REMOTE machine, inside the uploaded folder:  bash scripts/start_remote.sh
# One-shot: environment setup (first time only) -> GPU checks -> full pipeline in the background.
# Re-running resumes: setup is skipped if .venv exists and finished stages are skipped.
#
# Everything (pip packages, model weights) is downloaded HERE. On networks with HTTPS interception
# (campus proxy, "self-signed certificate in certificate chain") pip and the model downloader are
# pointed at the system CA bundle, which normally trusts the proxy certificate.
# Last resort (skips certificate checks for pip + model downloads):
#   ALLOW_INSECURE_DOWNLOADS=1 bash scripts/start_remote.sh
#
# Options:  PYTHON=/path/to/python3.10+   GPU=0   MIN_FREE_GB=8   EXTRA="--set use_xenc=False"
#           CA_BUNDLE=/path/ca.pem   SKIP_SMOKE=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUTF8=1
export HF_HOME="$ROOT/hf_cache"
GPU="${GPU:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

echo "== project folder: $ROOT"
[[ -d dataset/train && -d dataset/test && -d code/business_entity_resolution ]] || {
  echo "dataset/ or code/ missing — did the upload finish?"; exit 1; }

# ---------------------------------------------------------------- 0. Python + HTTPS reachability
if [[ -f .venv/.setup_complete ]]; then PYTHON="$ROOT/.venv/bin/python"; fi
# pick an existing Python >= 3.10 (no conda env creation needed)
if [[ -z "${PYTHON:-}" ]]; then
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PYTHON="$(command -v "$c")"; break
    fi
  done
fi
[[ -n "${PYTHON:-}" ]] || { echo "No Python >= 3.10 found. Activate a conda env that has one"
                            echo "(check with: python --version) or pass PYTHON=/path/to/python"; exit 1; }
echo "== using interpreter $PYTHON ($("$PYTHON" --version 2>&1))"

if [[ -d offline_wheels && -d hf_cache/hub ]]; then
  echo "== offline bundle found (offline_wheels/, hf_cache/): no internet needed"
  export HF_HUB_OFFLINE=1
elif [[ "${ALLOW_INSECURE_DOWNLOADS:-0}" == "1" ]]; then
  echo "== WARNING: certificate verification disabled for downloads (ALLOW_INSECURE_DOWNLOADS=1)"
  export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org download-r2.pytorch.org"
else
  echo "== checking HTTPS from Python (pypi.org, huggingface.co)"
  PIP_CFG_CERT="$("$PYTHON" -m pip config get global.cert 2>/dev/null || true)"
  if OK_BUNDLE="$("$PYTHON" scripts/net_check.py "${CA_BUNDLE:-}" "$PIP_CFG_CERT" "${REQUESTS_CA_BUNDLE:-}" \
        /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt \
        /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem /etc/ssl/cert.pem | tail -1)"; then
    if [[ "$OK_BUNDLE" != "DEFAULT" && -n "$OK_BUNDLE" ]]; then
      echo "== HTTPS works with CA bundle $OK_BUNDLE"
      export PIP_CERT="$OK_BUNDLE" REQUESTS_CA_BUNDLE="$OK_BUNDLE" SSL_CERT_FILE="$OK_BUNDLE" CURL_CA_BUNDLE="$OK_BUNDLE"
    else
      echo "== HTTPS works with default certificates"
    fi
  else
    echo "Python cannot reach https://pypi.org + https://huggingface.co with any CA bundle (errors above)."
    echo "Options: (a) ask your admin for the proxy CA file and run  CA_BUNDLE=/path/ca.pem bash scripts/start_remote.sh"
    echo "         (b) skip certificate checks for downloads:        ALLOW_INSECURE_DOWNLOADS=1 bash scripts/start_remote.sh"
    exit 1
  fi
fi

# ---------------------------------------------------------------- 1. environment (first run only)
if [[ ! -f .venv/.setup_complete ]]; then   # (re)build if missing or a previous setup failed

  PYTHON="$PYTHON" bash scripts/remote_setup.sh
else
  echo "== .venv already set up, skipping (delete .venv to reinstall)"
fi

# ---------------------------------------------------------------- 2. models (downloads anything missing)
.venv/bin/python scripts/download_models.py || {
  echo "Model download failed. If it is a certificate error, re-run with CA_BUNDLE=... or"
  echo "ALLOW_INSECURE_DOWNLOADS=1 (see top of this script)."; exit 1; }

# ---------------------------------------------------------------- 3. GPU sanity + free memory
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
FREE_MB="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "GPU $GPU free memory: ${FREE_MB} MiB"
if (( FREE_MB < MIN_FREE_GB * 1024 )); then
  echo "Only $((FREE_MB/1024)) GB free on GPU $GPU (need ${MIN_FREE_GB} GB). The GPU is shared —"
  echo "wait for other jobs or override: MIN_FREE_GB=12 bash scripts/start_remote.sh"
  exit 1
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -c "
import torch; assert torch.cuda.is_available(), 'CUDA not available in .venv torch'
print('torch', torch.__version__, 'cuda', torch.version.cuda, '->', torch.cuda.get_device_name(0))"
echo "RAM:"; free -g | head -2
echo "CPUs: $(nproc)   disk free here: $(df -h . | tail -1 | awk '{print $4}')"

# ---------------------------------------------------------------- 4. quick smoke run on THIS machine first
if [[ "${SKIP_SMOKE:-0}" != "1" && ! -f artifacts/.smoke_ok ]]; then
  echo "== smoke run (~5 min) to validate this machine's environment before the long run"
  if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
     .venv/bin/python code/business_entity_resolution/src/run_all.py --smoke --force > logs_smoke_remote.txt 2>&1; then
    mkdir -p artifacts && touch artifacts/.smoke_ok
    grep -E "gbm backend|PASS|decision variants" logs_smoke_remote.txt | tail -3 || true
    echo "== smoke run OK"
  else
    echo "SMOKE RUN FAILED — last lines of logs_smoke_remote.txt:"; tail -30 logs_smoke_remote.txt
    echo "Paste this output to get it fixed. (SKIP_SMOKE=1 bypasses this check.)"; exit 1
  fi
fi

# ---------------------------------------------------------------- 5. launch
GPU="$GPU" bash scripts/run_full.sh ${EXTRA:-}
echo
echo "Running in the background — safe to log out. Monitor with:"
echo "  tail -f $ROOT/logs/full_*.log"
echo "Outputs will be in $ROOT/output/ when the log says 'ALL DONE'."
