#!/usr/bin/env bash
# Run every variant in order; skips scripts that don't exist yet.
# Env overrides (passed through): SEEDS, TIMESTEPS, DEVICE, N_ENVS, EPISODES, RUN_PREFIX, PENALTY
set -uo pipefail
cd "$(dirname "$0")/.."
for s in run_baseline.sh run_time_penalty.sh run_hack.sh run_fix.sh; do
  if [[ -x "scripts/$s" ]]; then
    echo "##### scripts/$s"; "scripts/$s" || echo "WARNING: scripts/$s failed (exit $?)" >&2
  elif [[ -f "scripts/$s" ]]; then
    echo "##### scripts/$s (not executable, running with bash)"; bash "scripts/$s" || echo "WARNING: scripts/$s failed" >&2
  else
    echo "WARNING: scripts/$s not found, skipping" >&2
  fi
done
