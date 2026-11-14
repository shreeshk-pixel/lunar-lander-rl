#!/usr/bin/env bash
# Internal helper: train + evaluate + record one wrapper variant for each seed.
# Usage: _run_variant.sh <wrapper> [wrapper-kwargs-json]
# Env overrides: SEEDS ("0 1"), TIMESTEPS (1000000), DEVICE (cpu), N_ENVS (8),
#                EPISODES (30), RUN_PREFIX (run name = ${RUN_PREFIX}<wrapper>_s<seed>),
#                APPEND_TIMES (0; 1 = append a row to results/${RUN_PREFIX}training_times.csv)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
WRAPPER="$1"; KWARGS="${2:-{\}}"
SEEDS="${SEEDS:-0 1}"; TIMESTEPS="${TIMESTEPS:-1000000}"; DEVICE="${DEVICE:-cpu}"
N_ENVS="${N_ENVS:-8}"; EPISODES="${EPISODES:-30}"; RUN_PREFIX="${RUN_PREFIX:-}"
for SEED in $SEEDS; do
  RUN="${RUN_PREFIX}${WRAPPER}_s${SEED}"
  echo "=== [$RUN] train wrapper=$WRAPPER kwargs=$KWARGS seed=$SEED timesteps=$TIMESTEPS device=$DEVICE"
  python src/train.py --wrapper "$WRAPPER" --wrapper-kwargs "$KWARGS" --seed "$SEED" \
      --timesteps "$TIMESTEPS" --n-envs "$N_ENVS" --device "$DEVICE" --run-name "$RUN"
  if [ "${APPEND_TIMES:-0}" = "1" ]; then
    python - "$RUN" "$RUN_PREFIX" <<'PY'
import csv, json, os, sys
run, pre = sys.argv[1], sys.argv[2]
j = json.load(open(f"results/{run}_train.json")); path = f"results/{pre}training_times.csv"
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["run", "wrapper", "seed", "timesteps", "n_envs", "device",
                    "wall_clock_seconds", "wall_clock_minutes", "finished_at"])
    w.writerow([j["run"], j["wrapper"], j["seed"], j["actual_timesteps"], j["n_envs"],
                j.get("device", ""), j["wall_clock_seconds"], j["wall_clock_minutes"], j["finished_at"]])
PY
  fi
  echo "=== [$RUN] evaluate ($EPISODES episodes, plain env)"
  python src/evaluate.py --model "models/$RUN.zip" --episodes "$EPISODES" --device "$DEVICE"
  echo "=== [$RUN] record clip"
  python src/record.py --model "models/$RUN.zip" --episodes 2 --device "$DEVICE"
done
