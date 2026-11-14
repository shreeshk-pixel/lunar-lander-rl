"""Smoke test for the 'hack' and 'fix' reward wrappers.

Runs episodes with a random policy and with Gymnasium's built-in heuristic
lander (which actually lands, so the upright-slow predicate and termination
paths get exercised). Checks:
  * hack: shaped_bonus == bonus * upright_slow_frames, shaped = plain + bonus
  * fix:  discounted sum of F == gamma^T*Phi_T - Phi_0 (Phi_T = 0 if terminated),
          undiscounted sum of F + fees == shaped_bonus
Usage: python src/test_wrappers.py [--episodes N]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gymnasium.envs.box2d.lunar_lander import heuristic  # noqa: E402

from wrappers import DEFAULT_GAMMA, make_env, upright_and_slow  # noqa: E402


def run(kind, policy, seed):
    env = make_env(kind)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    unwrapped = env.unwrapped
    shaped, Fs, t, done = 0.0, [], 0, False
    us_count = 0
    while not done:
        a = env.action_space.sample() if policy == "random" else heuristic(unwrapped, obs)
        obs, r, term, trunc, info = env.step(a)
        shaped += r
        us_count += upright_and_slow(obs)
        if "shaping_F" in info:
            Fs.append(info["shaping_F"])
        t += 1
        done = term or trunc
    env.close()
    return dict(shaped=shaped, info=info, Fs=Fs, T=t, term=term, us=us_count,
                phi_T=0.0 if term else (1.0 if upright_and_slow(obs) else 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    ok = True
    for kind in ("hack", "fix"):
        for policy in ("random", "heuristic"):
            for ep in range(args.episodes):
                res = run(kind, policy, seed=1000 + ep)
                inf = res["info"]
                plain, bonus, us = inf["plain_return"], inf["shaped_bonus"], inf["upright_slow_frames"]
                checks = [abs(res["shaped"] - (plain + bonus)) < 1e-6, us == res["us"],
                          inf["episode_frames"] == res["T"]]
                extra = ""
                if kind == "hack":
                    checks.append(abs(bonus - 1.0 * us) < 1e-9)
                else:
                    g = DEFAULT_GAMMA
                    Fs = np.array(res["Fs"])
                    disc = float(np.sum(g ** np.arange(len(Fs)) * Fs))
                    # Phi_0 recovered from telescoping identity
                    phi0 = g ** res["T"] * res["phi_T"] - disc
                    checks.append(abs(phi0 - round(phi0)) < 1e-9 and round(phi0) in (0, 1))
                    checks.append(abs(bonus - (Fs.sum() - 0.05 * res["T"])) < 1e-9)
                    extra = (f" sumF={Fs.sum():+.3f} discF={disc:+.3f} phi0={phi0:+.3f}"
                             f" fees={-0.05*res['T']:.2f}")
                ok &= all(checks)
                print(f"{kind:4s} {policy:9s} ep{ep} T={res['T']:4d} term={int(res['term'])} "
                      f"plain={plain:+8.2f} bonus={bonus:+8.2f} shaped={res['shaped']:+8.2f} "
                      f"us_frames={us:4d}{extra} {'OK' if all(checks) else 'FAIL'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
