"""Deterministic evaluation of a saved PPO model on the PLAIN LunarLander-v3.

No reward wrapper is ever applied here, so shaping can't flatter the score.

Landing definition ("landed"): the episode TERMINATED (not truncated at 1000
steps) with a final-step reward of +100, which LunarLander only gives when the
lander comes to rest (body asleep) — i.e. a safe landing, not a crash (-100)
and not flying off-screen. We also report "solved_rate" = fraction of episodes
with return >= 200 (Gymnasium's "solved" threshold) and "on_pad_rate" = landed
with |x| <= 0.2 (between the flags).

Example:
    .venv/bin/python src/evaluate.py --model models/plain_s0.zip --episodes 20
Writes results/<model-name>_eval.json (or --out).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from common import ENV_ID, RESULTS, make_plain_env, write_json  # noqa: E402


def evaluate(model_path, episodes=20, seed=1000, deterministic=True, device="cpu"):
    model = PPO.load(model_path, device=device)
    env = make_plain_env()
    eps = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        ret, n, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(action)
            ret += float(r); n += 1
            done = term or trunc
        landed = bool(term and r == 100)
        crashed = bool(term and r == -100)
        eps.append(dict(seed=seed + i, ret=round(ret, 3), length=n, terminated=bool(term),
                        truncated=bool(trunc and not term), landed=landed, crashed=crashed,
                        on_pad=bool(landed and abs(obs[0]) <= 0.2)))
    env.close()
    R = np.array([e["ret"] for e in eps]); L = np.array([e["length"] for e in eps])
    return dict(
        model=str(model_path), env_id=ENV_ID, wrapper="none (plain env)", episodes=episodes,
        deterministic=deterministic, eval_seeds=[seed, seed + episodes - 1],
        mean_return=round(float(R.mean()), 2), std_return=round(float(R.std()), 2),
        min_return=round(float(R.min()), 2), max_return=round(float(R.max()), 2),
        landing_rate=round(float(np.mean([e["landed"] for e in eps])), 3),
        on_pad_rate=round(float(np.mean([e["on_pad"] for e in eps])), 3),
        solved_rate=round(float(np.mean(R >= 200)), 3),
        crash_rate=round(float(np.mean([e["crashed"] for e in eps])), 3),
        truncated_rate=round(float(np.mean([e["truncated"] for e in eps])), 3),
        mean_length=round(float(L.mean()), 1),
        landing_definition="terminated with final reward +100 (lander at rest; not crash, not timeout)",
        per_episode=eps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000, help="first eval-episode seed")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    a = p.parse_args(argv)
    res = evaluate(a.model, a.episodes, a.seed, not a.stochastic, a.device)
    name = os.path.splitext(os.path.basename(a.model))[0]
    out = a.out or str(RESULTS / f"{name}_eval.json")
    write_json(out, res)
    print({k: v for k, v in res.items() if k != "per_episode"})
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
