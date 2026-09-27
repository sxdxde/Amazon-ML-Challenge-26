#!/usr/bin/env bash
# Apple-silicon environment for entity_resolution.ipynb (M1 Pro, native arm64).
#   bash setup_mac.sh                 create/refresh venv, install pins, register kernel "er-mac"
#   PYTHON=/path/to/python3.12 bash setup_mac.sh   use a specific interpreter
# This script does not run the notebook, download models or train anything.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
           /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "ERROR: no python3.11/3.12 found. Set PYTHON=/path/to/python3.12"; exit 1; }
VENV=".venv_mlx"

# 1. Native arm64 only. An x86_64 interpreter runs under Rosetta: no MLX, no Metal, several x slower.
ARCH="$("$PY" -c 'import platform; print(platform.machine())')"
if [ "$ARCH" != "arm64" ]; then
  echo "ERROR: $PY reports platform.machine()=$ARCH (Rosetta / Intel build)."
  echo "Install a native arm64 Python 3.11 or 3.12 (python.org universal2 installer or Homebrew"
  echo "under /opt/homebrew) and rerun with PYTHON=/path/to/that/python."
  exit 1
fi
"$PY" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)], sys.version' \
  || { echo "ERROR: need Python 3.11 or 3.12, got $("$PY" -V)"; exit 1; }
echo "python: $PY  ($("$PY" -V), $ARCH)"

# 2. venv + pinned packages + Jupyter kernel.
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -c 'import platform; assert platform.machine()=="arm64"' \
  || { echo "ERROR: $VENV is not arm64. Delete it (rm -rf $VENV) and rerun."; exit 1; }
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VPY" -r requirements.txt
else
  "$VPY" -m pip install -U pip
  "$VPY" -m pip install -r requirements.txt
fi
"$VPY" -m ipykernel install --user --name er-mac --display-name "er-mac (MLX, arm64)"

# 3. LightGBM needs OpenMP (libomp) on macOS.
if ! "$VPY" -c 'import lightgbm' 2>"$VENV/lgbm_import_err.txt"; then
  cat "$VENV/lgbm_import_err.txt"
  if grep -qiE 'omp|libomp' "$VENV/lgbm_import_err.txt"; then
    echo ""
    echo ">>> LightGBM cannot find OpenMP. Fix with:   brew install libomp"
    echo ">>> then rerun:  bash setup_mac.sh"
  fi
  exit 1
fi

# 4. Import check only (parity/benchmarks live in notebook cells 1a/1b).
"$VPY" - <<'EOF'
import platform, mlx.core as mx, torch, lightgbm, sklearn, rapidfuzz, pandas, numpy
print("arch", platform.machine(), "| mlx", mx.__version__, "| default device", mx.default_device())
print("torch", torch.__version__, "| mps available", torch.backends.mps.is_available())
print("lightgbm", lightgbm.__version__, "| sklearn", sklearn.__version__, "| rapidfuzz", rapidfuzz.__version__,
      "| pandas", pandas.__version__, "| numpy", numpy.__version__)
EOF
echo "OK: select the 'er-mac' kernel in Jupyter (jupyter lab  or  $VENV/bin/jupyter lab)."
