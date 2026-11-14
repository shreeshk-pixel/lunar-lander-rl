#!/usr/bin/env bash
# Reward FIX: PPO with PotentialShapingFix: r + gamma*Phi(s') - Phi(s) - fee,
# Phi = 1[upright & slow], Phi(terminal) = 0, fee 0.05/frame -> should land again.
# Per seed: train + evaluate (plain env) + record, via scripts/_run_variant.sh.
# Then records the landing clip assets/${RUN_PREFIX}fix_landing.{gif,mp4} from the
# seed with the highest plain-env eval mean_return (ties -> first seed listed).
# Env overrides: SEEDS ("0 1"), TIMESTEPS (1000000), DEVICE (cpu), N_ENVS (8),
#   EPISODES (30), RUN_PREFIX (""), WRAPPER_KWARGS (JSON, default '{}' = wrapper
#   defaults: scale=1.0, gamma=0.999 (= train.py PPO gamma), fee=0.05,
#   max_angle=0.2, max_speed=0.3).
set -euo pipefail
cd "$(dirname "$0")/.."
W=fix
WRAPPER_KWARGS="${WRAPPER_KWARGS:-{\}}"
export SEEDS="${SEEDS:-0 1}" TIMESTEPS="${TIMESTEPS:-1000000}" DEVICE="${DEVICE:-cpu}"
export EPISODES="${EPISODES:-30}" RUN_PREFIX="${RUN_PREFIX:-}"
scripts/_run_variant.sh fix "$WRAPPER_KWARGS"
# Append wall-clock training time per run to results/${RUN_PREFIX}training_times.csv.
.venv/bin/python - "$W" <<'PY'
import csv, json, os, sys
w = sys.argv[1]; pre = os.environ["RUN_PREFIX"]; path = f"results/{pre}training_times.csv"  # prefixed runs (smoke tests) get their own file
for s in os.environ["SEEDS"].split():
    j = json.load(open(f"results/{pre}{w}_s{s}_train.json"))
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        cw = csv.writer(f)
        if new:
            cw.writerow(["run", "wrapper", "seed", "timesteps", "n_envs", "device",
                         "wall_clock_seconds", "wall_clock_minutes", "finished_at"])
        cw.writerow([j["run"], j["wrapper"], j["seed"], j["actual_timesteps"], j["n_envs"],
                     j.get("device", ""), j["wall_clock_seconds"], j["wall_clock_minutes"], j["finished_at"]])
PY
BEST=$(.venv/bin/python - fix mean_return <<'PY'
import json, os, sys
w, key = sys.argv[1], sys.argv[2]
pre, seeds = os.environ["RUN_PREFIX"], os.environ["SEEDS"].split()
best, val = seeds[0], None
for s in seeds:
    p = f"results/{pre}{w}_s{s}_eval.json"
    if os.path.exists(p):
        v = json.load(open(p))[key]
        if val is None or v > val: best, val = s, v
print(best)
PY
)
echo "=== [${RUN_PREFIX}fix] landing clip from seed $BEST"
.venv/bin/python src/record.py --model "models/${RUN_PREFIX}fix_s${BEST}.zip" --episodes 2 \
    --device "$DEVICE" --name "${RUN_PREFIX}fix_landing"
