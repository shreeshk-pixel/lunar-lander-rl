"""Shared helpers: paths, env construction, CSV loading, plotting."""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

ENV_ID = "LunarLander-v3"
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
for _d in (MODELS, LOGS, RESULTS, ASSETS):
    _d.mkdir(parents=True, exist_ok=True)


def make_plain_env(render_mode=None) -> gym.Env:
    """The unmodified LunarLander-v3 (used for all evaluation / recording)."""
    return gym.make(ENV_ID, render_mode=render_mode)


class ShapedReturnLogger(gym.Wrapper):
    """Outermost wrapper: logs the *shaped* (training) episode return to a CSV.

    Does not touch info['episode'] (that comes from the inner Monitor and holds
    the PLAIN env return). Adds info['shaped_episode'] = {'r','l'} at episode end.
    """

    def __init__(self, env: gym.Env, path: str):
        super().__init__(env)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["r_shaped", "l"])
        self._f.flush()
        self._ret, self._len = 0.0, 0

    def reset(self, **kwargs):
        self._ret, self._len = 0.0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._ret += float(r)
        self._len += 1
        if term or trunc:
            info["shaped_episode"] = {"r": self._ret, "l": self._len}
            self._w.writerow([round(self._ret, 6), self._len])
            self._f.flush()
        return obs, r, term, trunc, info

    def close(self):
        try:
            self._f.close()
        finally:
            super().close()


def load_monitor_curve(log_dir: str | Path):
    """Merge all *.monitor.csv in log_dir (one per parallel env) into one
    chronological series. Returns (timesteps_cumulative, returns, lengths)."""
    rows = []
    for fn in glob.glob(os.path.join(str(log_dir), "*.monitor.csv")):
        with open(fn) as f:
            f.readline()  # JSON header line written by SB3 Monitor
            for row in csv.DictReader(f):
                rows.append((float(row["t"]), float(row["r"]), int(row["l"])))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return np.array([]), np.array([]), np.array([])
    t, r, l = map(np.array, zip(*rows))
    return np.cumsum(l), r, l


def plot_curve(log_dir, out_png, out_csv=None, title=None, window=100):
    """Learning curve (plain-env episode return vs env steps) + rolling mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, rets, lens = load_monitor_curve(log_dir)
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "episode_return_plain", "episode_length"])
            w.writerows(zip(steps.tolist(), rets.tolist(), lens.tolist()))
    if len(rets) == 0:
        return
    roll = np.convolve(rets, np.ones(window) / window, mode="valid") if len(rets) >= window else rets
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rets, alpha=0.25, lw=0.6, label="episode return (plain reward)")
    ax.plot(steps[len(steps) - len(roll):], roll, lw=2, label=f"rolling mean ({window})")
    ax.axhline(200, color="green", ls="--", lw=1, label="solved = 200")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return (plain env reward)")
    ax.set_title(title or str(log_dir))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
