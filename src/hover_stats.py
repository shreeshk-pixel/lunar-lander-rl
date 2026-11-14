"""Hover diagnostics for a saved model on the PLAIN LunarLander-v3.

Per episode (deterministic, seeds 1000..1000+N-1, same as evaluate.py):
plain return, length, landed/crashed/timeout, and the fraction of frames that
are upright & slow (wrappers.upright_and_slow, |angle|<0.2, speed<0.3) with
NEITHER leg touching the ground ("hover frames"), i.e. frames on which the
'hack' bonus is paid while airborne.

Usage: .venv/bin/python src/hover_stats.py --model models/hack_s0.zip [--episodes 30]
Writes results/<model-name>_hover.json (or --out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from common import RESULTS, make_plain_env  # noqa: E402
from wrappers import upright_and_slow  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    model = PPO.load(a.model, device=a.device)
    env = make_plain_env()
    eps = []
    for i in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + i)
        ret, n, hover, us_any, done = 0.0, 0, 0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            ret += float(r); n += 1
            us = upright_and_slow(obs)
            us_any += us
            hover += us and obs[6] < 0.5 and obs[7] < 0.5
            done = term or trunc
        eps.append(dict(seed=a.seed + i, ret=round(ret, 2), length=n,
                        landed=bool(term and r == 100), crashed=bool(term and r == -100),
                        timeout=bool(trunc and not term),
                        hover_frac=round(hover / n, 4), upright_slow_frac=round(us_any / n, 4)))
    env.close()
    m = lambda k: round(float(np.mean([e[k] for e in eps])), 3)  # noqa: E731
    res = dict(model=a.model, episodes=a.episodes, mean_return=m("ret"),
               std_return=round(float(np.std([e["ret"] for e in eps])), 2),
               mean_length=m("length"), landing_rate=m("landed"), crash_rate=m("crashed"),
               timeout_rate=m("timeout"), hover_frame_frac=m("hover_frac"),
               upright_slow_frame_frac=m("upright_slow_frac"),
               hover_frames_total=int(sum(e["hover_frac"] * e["length"] for e in eps) + 0.5),
               frames_total=int(sum(e["length"] for e in eps)), per_episode=eps)
    out = a.out or str(RESULTS / f"{os.path.splitext(os.path.basename(a.model))[0]}_hover.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print({k: v for k, v in res.items() if k != "per_episode"})
    print(f"[hover] wrote {out}")


if __name__ == "__main__":
    main()
