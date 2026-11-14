"""Record GIF + MP4 of a saved model (or a random agent) on the PLAIN env.

Uses render_mode='rgb_array'. GIF is downscaled / frame-skipped to stay < 5 MB
(auto-shrinks further if needed); the MP4 is full-res at 50 fps.

Examples:
    .venv/bin/python src/record.py --random --name random_crash --episodes 3
    .venv/bin/python src/record.py --model models/plain_s0.zip --episodes 2
Outputs assets/<name>.gif and assets/<name>.mp4 (name defaults to model basename).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import ASSETS, make_plain_env  # noqa: E402

MAX_GIF_BYTES = 5 * 1024 * 1024


def rollout_frames(model_path=None, episodes=2, seed=0, max_steps=1000, deterministic=True, device="cpu"):
    model = None
    if model_path:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
    env = make_plain_env(render_mode="rgb_array")
    env.action_space.seed(seed)
    frames, summary = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        frames.append(env.render())
        ret = 0.0
        for t in range(max_steps):
            action = model.predict(obs, deterministic=deterministic)[0] if model else env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            ret += float(r)
            frames.append(env.render())
            if term or trunc:
                break
        frames.extend([frames[-1]] * 25)  # hold last frame 0.5 s
        summary.append(dict(episode=ep, ret=round(ret, 1), length=t + 1, final_reward=r))
    env.close()
    return frames, summary


def write_gif(frames, path, scale=0.5, skip=2, fps=25):
    while True:
        small = [np.asarray(Image.fromarray(f).resize(
            (int(f.shape[1] * scale), int(f.shape[0] * scale)), Image.BILINEAR)) for f in frames[::skip]]
        imageio.mimsave(path, small, duration=1000 * skip / 50 / 1, loop=0)
        size = os.path.getsize(path)
        if size < MAX_GIF_BYTES or scale < 0.2:
            return size, scale, skip
        if skip < 4:
            skip += 1
        else:
            scale *= 0.8


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--random", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--gif-scale", type=float, default=0.5)
    p.add_argument("--gif-skip", type=int, default=2)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    a = p.parse_args(argv)

    name = a.name or ("random_agent" if a.random else os.path.splitext(os.path.basename(a.model))[0])
    frames, summary = rollout_frames(None if a.random else a.model, a.episodes, a.seed, a.max_steps, device=a.device)
    print(f"[record] {len(frames)} frames; episodes: {summary}")
    gif = ASSETS / f"{name}.gif"
    size, scale, skip = write_gif(frames, str(gif), a.gif_scale, a.gif_skip)
    print(f"[record] {gif} ({size/1e6:.2f} MB, scale={scale:.2f}, skip={skip})")
    if not a.no_mp4:
        mp4 = ASSETS / f"{name}.mp4"
        imageio.mimsave(str(mp4), frames, fps=50, codec="libx264", quality=7, macro_block_size=8)
        print(f"[record] {mp4} ({os.path.getsize(mp4)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
