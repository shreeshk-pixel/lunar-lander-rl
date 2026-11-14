# src/ — shared Lunar Lander RL code

All commands run from the repo root with the venv: `P=.venv/bin/python`
(Python 3.13, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu). CPU only.

| file | purpose |
|---|---|
| `wrappers.py` | Reward wrappers + `WRAPPERS` registry (`plain`, `time_penalty`, `hack`, `fix`, ...). Add a class and a `WRAPPERS["name"] = Cls` line to register a new one. |
| `train.py` | PPO `MlpPolicy` training with an optional reward wrapper. |
| `evaluate.py` | Deterministic eval on the **plain** env (never wrapped). |
| `record.py` | GIF (<5 MB) + MP4 of a model or a random agent on the plain env. |
| `common.py` | Paths, plain env factory, `ShapedReturnLogger`, monitor-CSV loading, curve plotting. |

## Train
```bash
$P src/train.py --wrapper plain --seed 0                       # baseline, 1M steps
$P src/train.py --wrapper time_penalty --wrapper-kwargs '{"penalty": 0.05}' --seed 0
$P src/train.py --wrapper hack --seed 0 --run-name hack_s0     # teammate wrappers
```
Args: `--wrapper/--reward-wrapper` (WRAPPERS key; `none` = alias of `plain`; default `plain`),
`--wrapper-kwargs` JSON (passed as `Cls(env, **kwargs)`), `--seed` (0), `--timesteps` (default **1_000_000**),
`--n-envs` (default **8** = min(8, nproc); DummyVecEnv), `--run-name` (default `<wrapper>_s<seed>`),
`--torch-threads` (default 1, so several runs can share the 8-core box).
PPO hyperparams (RL-Zoo-like): n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, lr=3e-4.
With 8 envs one rollout = 8192 steps, so `--timesteps 1000000` actually runs 1,007,616 steps.

**Env stack (inner → outer):** `gym.make('LunarLander-v3') → Monitor → <reward wrapper> → ShapedReturnLogger`.
Monitor is *inside* the reward wrapper, so `logs/<run>/<i>.monitor.csv` and SB3's `ep_rew_mean`
always record the **plain env return** — learning curves are comparable across all variants.
The shaped (training) return is logged separately to `logs/<run>/shaped_<i>.csv`
(and `info["shaped_episode"]`).

Outputs: `models/<run>.zip`, `logs/<run>/` (monitor CSVs, shaped CSVs, SB3 `progress.csv`, `curve.csv`),
`assets/<run>_curve.png`, `results/<run>_train.json` (config + `wall_clock_seconds`).
Re-plot anytime: `$P -c "import sys; sys.path.insert(0,'src'); from common import plot_curve; plot_curve('logs/RUN','assets/RUN_curve.png','logs/RUN/curve.csv')"`.

## Evaluate (plain env, deterministic)
```bash
$P src/evaluate.py --model models/plain_s0.zip --episodes 20   # -> results/plain_s0_eval.json
```
Reports mean/std/min/max return, `mean_length`, and:
- `landing_rate` — **episode terminated with final-step reward +100** (Gymnasium gives +100 only when the
  lander comes to rest, i.e. a safe landing; crash = -100, timeout = truncated). This is our landing definition.
- `on_pad_rate` — landed with |x| ≤ 0.2 (between the flags).
- `solved_rate` — return ≥ 200 (Gymnasium's "solved" threshold).
- `crash_rate`, `truncated_rate` (hit 1000-step limit, e.g. hovering).
Eval episodes use seeds 1000..1000+N-1 (`--seed` to change), identical across models.

## Record
```bash
$P src/record.py --random --name random_crash --episodes 3
$P src/record.py --model models/plain_s0.zip --episodes 2      # -> assets/plain_s0.{gif,mp4}
```
GIF: scale 0.5, every 2nd frame (auto-shrinks until < 5 MB). MP4: full-res, 50 fps.

## Devices / GPU
`train.py`, `evaluate.py`, `record.py` take `--device auto|cpu|cuda` (default `cpu`).
A GPU does **not** speed up small-MLP PPO (2×64 net; the bottleneck is CPU env stepping — SB3 itself warns
about this), so `cpu` is recommended even on a CUDA laptop. On CUDA machines the default pip `torch` wheel works
as-is; to save ~2 GB install the CPU-only wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
(or `TORCH_CPU=1 scripts/setup.sh`). Exact versions used on the box: `results/pinned_versions.txt`.

## scripts/ (one-shot pipelines)
```bash
scripts/setup.sh                 # create .venv, install deps (requirements.txt if present)
scripts/run_baseline.sh          # plain:        train -> eval (20 ep, plain env) -> record, per seed
scripts/run_time_penalty.sh      # time_penalty: same (PENALTY=0.05 default)
scripts/run_all.sh               # baseline, time_penalty, run_hack.sh, run_fix.sh (skips missing ones)
```
Env overrides: `SEEDS` ("0 1"), `TIMESTEPS` (1000000), `DEVICE` (cpu), `N_ENVS` (8), `EPISODES` (20),
`RUN_PREFIX` (prepended to run names, e.g. `RUN_PREFIX=smoke_ TIMESTEPS=5000 SEEDS=0 scripts/run_baseline.sh`).
Run names are `${RUN_PREFIX}<wrapper>_s<seed>`.
