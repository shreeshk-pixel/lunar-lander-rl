#!/usr/bin/env bash
# Baseline PPO (no reward wrapper). Env overrides: SEEDS, TIMESTEPS, DEVICE, N_ENVS, EPISODES, RUN_PREFIX
set -euo pipefail
cd "$(dirname "$0")/.."
APPEND_TIMES=1 exec scripts/_run_variant.sh plain
