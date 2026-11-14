"""Train PPO (MlpPolicy) on LunarLander-v3 with an optional reward wrapper.

Env stack per parallel env (inner -> outer):
    gym.make('LunarLander-v3') -> Monitor -> <reward wrapper> -> ShapedReturnLogger
Monitor sits INSIDE the reward wrapper, so logs/<run>/*.monitor.csv and SB3's
ep_rew_mean always show the PLAIN env return; curves are comparable across
variants. The shaped return is logged to logs/<run>/shaped_<i>.csv.

Example:
    .venv/bin/python src/train.py --wrapper time_penalty \
        --wrapper-kwargs '{"penalty": 0.05}' --seed 0 --timesteps 1000000

Outputs (run name defaults to f"{wrapper}_s{seed}"):
    models/<run>.zip                     trained model
    logs/<run>/                          monitor CSVs, shaped CSVs, SB3 progress.csv
    assets/<run>_curve.png, logs/<run>/curve.csv   learning curve + raw data
    results/<run>_train.json             config + wall-clock training time
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from common import ASSETS, ENV_ID, LOGS, MODELS, RESULTS, ShapedReturnLogger, plot_curve, write_json  # noqa: E402
from wrappers import WRAPPERS  # noqa: E402

ALIASES = {"none": "plain"}

# PPO hyperparameters (close to RL-Zoo's tuned LunarLander config).
PPO_KWARGS = dict(
    n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999,
    gae_lambda=0.98, ent_coef=0.01, learning_rate=3e-4,
)


def build_env_fn(wrapper_name, wrapper_kwargs, log_dir, rank, seed):
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env, os.path.join(log_dir, str(rank)))  # plain reward logged
        cls = WRAPPERS.get(wrapper_name)
        if cls is not None:
            env = cls(env, **wrapper_kwargs)
        env = ShapedReturnLogger(env, os.path.join(log_dir, f"shaped_{rank}.csv"))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrapper", "--reward-wrapper", dest="wrapper", default="plain",
                   help=f"one of {sorted(WRAPPERS)} (alias 'none' -> 'plain')")
    p.add_argument("--wrapper-kwargs", default="{}", help="JSON dict passed to the wrapper constructor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--run-name", default=None, help="output basename (default: <wrapper>_s<seed>)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch CPU threads (1 lets several runs share the box)")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                   help="torch device for PPO (default cpu; GPU gives little/no speedup for a small MLP)")
    p.add_argument("--verbose", type=int, default=1)
    a = p.parse_args(argv)

    wrapper = ALIASES.get(a.wrapper, a.wrapper)
    if wrapper not in WRAPPERS:
        p.error(f"unknown wrapper {a.wrapper!r}; choices: {sorted(WRAPPERS)}")
    wkw = json.loads(a.wrapper_kwargs)
    run = a.run_name or f"{wrapper}_s{a.seed}"
    torch.set_num_threads(a.torch_threads)

    log_dir = LOGS / run
    log_dir.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([build_env_fn(wrapper, wkw, str(log_dir), i, a.seed) for i in range(a.n_envs)])
    model = PPO("MlpPolicy", venv, seed=a.seed, verbose=a.verbose, device=a.device, **PPO_KWARGS)
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

    print(f"[train] run={run} wrapper={wrapper} kwargs={wkw} seed={a.seed} "
          f"timesteps={a.timesteps} n_envs={a.n_envs}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=a.timesteps)
    wall = time.time() - t0
    model_path = MODELS / f"{run}.zip"
    model.save(str(model_path))
    venv.close()

    curve_png = ASSETS / f"{run}_curve.png"
    plot_curve(log_dir, curve_png, log_dir / "curve.csv", title=f"PPO {run} (plain-env return)")
    out = dict(run=run, env_id=ENV_ID, wrapper=wrapper, wrapper_kwargs=wkw, seed=a.seed,
               timesteps=a.timesteps, actual_timesteps=int(model.num_timesteps), n_envs=a.n_envs,
               ppo_kwargs=PPO_KWARGS, device=str(model.device), wall_clock_seconds=round(wall, 1),
               wall_clock_minutes=round(wall / 60, 2), model_path=str(model_path),
               log_dir=str(log_dir), curve_png=str(curve_png),
               finished_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    write_json(RESULTS / f"{run}_train.json", out)
    print(f"[train] done in {wall/60:.1f} min -> {model_path}", flush=True)


if __name__ == "__main__":
    main()
