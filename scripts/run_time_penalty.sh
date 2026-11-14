#!/usr/bin/env bash
# PPO with TimePenaltyWrapper. Extra override: PENALTY (default 0.05 per step).
set -euo pipefail
cd "$(dirname "$0")/.."
APPEND_TIMES=1 exec scripts/_run_variant.sh time_penalty "{\"penalty\": ${PENALTY:-0.05}}"
