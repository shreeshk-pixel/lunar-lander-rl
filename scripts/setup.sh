#!/usr/bin/env bash
# Create .venv at the repo root and install deps (requirements.txt if present).
# CPU-only torch saves ~2 GB: TORCH_CPU=1 scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
[[ -x .venv/bin/pip ]] && .venv/bin/pip --version >/dev/null 2>&1 || { rm -rf .venv; "$PY" -m venv .venv; }
.venv/bin/pip install --upgrade pip
.venv/bin/pip install swig   # Box2D build needs swig
if [[ "${TORCH_CPU:-0}" == "1" ]]; then
  .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
fi
if [[ -f requirements.txt ]]; then
  .venv/bin/pip install -r requirements.txt
else
  .venv/bin/pip install "gymnasium[box2d]" stable-baselines3 moviepy imageio imageio-ffmpeg matplotlib pandas
fi
.venv/bin/python -c "import gymnasium as g, stable_baselines3, torch; g.make('LunarLander-v3'); print('OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
