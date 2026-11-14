"""Reward wrappers for LunarLander-v3.

Shared registry: ``WRAPPERS`` maps a CLI name -> wrapper class (or None for the
plain, unmodified env). train.py looks names up here and constructs the wrapper
as ``cls(env, **wrapper_kwargs)``. Teammates: add your class below and register
it in ``WRAPPERS``; please don't remove other entries.

Agreed keys: 'plain' (no wrapper), 'time_penalty', 'hack', 'fix'.
"""
from __future__ import annotations

import gymnasium as gym


class TimePenaltyWrapper(gym.Wrapper):
    """Subtract a constant ``penalty`` from the reward on every step.

    The built-in LunarLander reward has no time cost, so this nudges the agent
    to land rather than dawdle. Default 0.05/step => at most -50 over a
    1000-step episode (small compared with the +100 landing bonus).

    The unshaped env reward is kept in ``info['original_reward']``.
    """

    def __init__(self, env: gym.Env, penalty: float = 0.05):
        super().__init__(env)
        self.penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        return obs, reward - self.penalty, terminated, truncated, info


# ---------------------------------------------------------------------------
# Registry. 'plain' = no wrapper. Teammates register 'hack' / 'fix' here.
# ---------------------------------------------------------------------------
WRAPPERS: dict = {
    "plain": None,
    "time_penalty": TimePenaltyWrapper,
}


# ===========================================================================
# Reward-hacking experiment wrappers ('hack' and 'fix'), added by the
# reward-hacking experimenter. Appended; entries above are untouched.
# Inside train.py's stack (gym.make -> Monitor -> wrapper -> ShapedReturnLogger)
# Monitor sees the PLAIN reward; the info['plain_return'] / ['shaped_bonus']
# keys below are extras for scripts that put Monitor outside the wrapper.
# ===========================================================================
import math

import numpy as np

# Observation indices (LunarLander-v3): 0 x, 1 y, 2 vx, 3 vy, 4 angle,
# 5 angular velocity, 6 left leg contact, 7 right leg contact.
#
# "Upright and slow" thresholds, in observation units:
#   |angle| < 0.2 rad (~11.5 deg): a lander tilted this little still lands
#     safely, and random flailing rarely stays inside it, so the predicate
#     marks controlled flight, not luck.
#   speed = sqrt(vx^2 + vy^2) < 0.3: obs velocities are scaled; the spawn
#     velocity and free fall reach roughly 0.5-1.5, a gentle hover or final
#     approach sits well under 0.3. A hovering lander passes easily.
DEFAULT_MAX_ANGLE = 0.2
DEFAULT_MAX_SPEED = 0.3
DEFAULT_HACK_BONUS = 1.0      # +1 per upright-and-slow frame (the guide's rule)
DEFAULT_PHI_SCALE = 1.0       # potential height; matches the hack's +1
DEFAULT_GAMMA = 0.999         # must equal PPO gamma: src/train.py uses gamma=0.999
DEFAULT_FRAME_FEE = 0.05      # per-frame fee in the fix; max -50 per 1000 frames


def upright_and_slow(obs, max_angle: float = DEFAULT_MAX_ANGLE,
                     max_speed: float = DEFAULT_MAX_SPEED) -> bool:
    """True when |obs[4]| < max_angle and sqrt(obs[2]^2 + obs[3]^2) < max_speed."""
    return bool(abs(float(obs[4])) < max_angle
                and math.hypot(float(obs[2]), float(obs[3])) < max_speed)


class _EpisodeStatsWrapper(gym.Wrapper):
    """Shared bookkeeping for the shaping wrappers.

    Every step sets ``info['original_reward']`` (the plain env reward, same key
    as TimePenaltyWrapper) and ``info['shaped_bonus_step']``. When an episode
    ends (terminated or truncated) it also sets:
      info['plain_return']        sum of plain env rewards this episode
      info['shaped_bonus']        sum of everything this wrapper added
                                  (shaped return = plain_return + shaped_bonus)
      info['upright_slow_frames'] number of post-step observations that were
                                  upright and slow
      info['episode_frames']      episode length
    Use these with ``Monitor(env, info_keywords=('plain_return', 'shaped_bonus',
    'upright_slow_frames'))``; Monitor's own 'r' column holds whatever reward
    reaches it (shaped, if Monitor wraps this wrapper).
    """

    STAT_KEYS = ("plain_return", "shaped_bonus", "upright_slow_frames",
                 "episode_frames")

    def __init__(self, env, max_angle=DEFAULT_MAX_ANGLE, max_speed=DEFAULT_MAX_SPEED):
        super().__init__(env)
        self.max_angle = float(max_angle)
        self.max_speed = float(max_speed)
        self._reset_stats()

    def _reset_stats(self):
        self._plain = 0.0
        self._bonus = 0.0
        self._us_frames = 0
        self._frames = 0

    def _is_us(self, obs) -> bool:
        return upright_and_slow(obs, self.max_angle, self.max_speed)

    def _record(self, reward, bonus, us, terminated, truncated, info):
        self._plain += float(reward)
        self._bonus += float(bonus)
        self._us_frames += int(us)
        self._frames += 1
        info["original_reward"] = float(reward)
        info["shaped_bonus_step"] = float(bonus)
        if terminated or truncated:
            info["plain_return"] = self._plain
            info["shaped_bonus"] = self._bonus
            info["upright_slow_frames"] = self._us_frames
            info["episode_frames"] = self._frames
        return info


class UprightSlowBonus(_EpisodeStatsWrapper):
    """THE HACK: add ``bonus`` (+1) to the env reward on every frame whose
    resulting observation is upright and slow (see module thresholds).

    This is the guide's rule. It pays whenever the predicate holds, including
    while hovering in mid-air. Landing terminates the episode, which stops the
    payments, while TimeLimit truncates at 1000 frames. So a lander that
    hovers upright can collect up to ~1000 bonus, minus at most ~300 main-engine
    fuel, far more than the +100 landing reward. Hovering out-earns landing,
    and PPO can learn exactly that.
    """

    def __init__(self, env, bonus: float = DEFAULT_HACK_BONUS,
                 max_angle: float = DEFAULT_MAX_ANGLE,
                 max_speed: float = DEFAULT_MAX_SPEED):
        super().__init__(env, max_angle, max_speed)
        self.bonus = float(bonus)

    def reset(self, **kwargs):
        self._reset_stats()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        us = self._is_us(obs)
        b = self.bonus if us else 0.0
        info = self._record(reward, b, us, terminated, truncated, info)
        return obs, reward + b, terminated, truncated, info


class PotentialShapingFix(_EpisodeStatsWrapper):
    """THE FIX: potential-based shaping (Ng, Harada & Russell 1999) plus a
    per-frame fee.

        r' = r + gamma * Phi(s') - Phi(s) - fee
        Phi(s) = scale * 1[upright and slow](s),  Phi(terminal) = 0

    * Pays for getting better (entering the upright-slow region) instead of
      for sitting in it: staying there earns gamma*scale - scale = -0.01*scale
      per frame (-0.001 at gamma=0.999), so hovering is no longer a reward source.
    * Phi(terminal) = 0 on ``terminated`` (land or crash) keeps the shaping
      policy-invariant: the discounted sum telescopes to -Phi(s0), which no
      policy can change. On TimeLimit ``truncated`` Phi(s') is kept as usual,
      since PPO bootstraps through truncation.
    * The indicator potential matches the hack's predicate exactly so the
      two runs are directly comparable. (A smooth Phi, e.g. -k*(|angle|+speed),
      would give denser gradients; it is not used here.)
    * ``fee`` (0.05/frame, at most -50 over 1000 frames) makes waiting cost
      something, as the guide suggests. Set fee=0 for pure potential shaping.
    * ``gamma`` must equal the PPO discount for the invariance to hold. Default
      0.999 matches PPO_KWARGS in src/train.py (SB3's own default is 0.99).

    ``info['shaped_bonus']`` = sum of (F - fee) over the episode.
    """

    def __init__(self, env, scale: float = DEFAULT_PHI_SCALE,
                 gamma: float = DEFAULT_GAMMA, fee: float = DEFAULT_FRAME_FEE,
                 max_angle: float = DEFAULT_MAX_ANGLE,
                 max_speed: float = DEFAULT_MAX_SPEED):
        super().__init__(env, max_angle, max_speed)
        self.scale = float(scale)
        self.gamma = float(gamma)
        self.fee = float(fee)
        self._phi = 0.0

    def phi(self, obs) -> float:
        return self.scale if self._is_us(obs) else 0.0

    def reset(self, **kwargs):
        self._reset_stats()
        obs, info = self.env.reset(**kwargs)
        self._phi = self.phi(obs)
        info["phi0"] = self._phi
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        us = self._is_us(obs)
        phi_next = 0.0 if terminated else (self.scale if us else 0.0)
        f = self.gamma * phi_next - self._phi
        b = f - self.fee
        self._phi = phi_next
        info["shaping_F"] = f
        info = self._record(reward, b, us, terminated, truncated, info)
        return obs, reward + b, terminated, truncated, info


WRAPPERS["hack"] = UprightSlowBonus
WRAPPERS["fix"] = PotentialShapingFix


def make_env(kind: str = "plain", seed: int | None = None, render_mode=None,
             **wrapper_kwargs):
    """Build LunarLander-v3 wrapped with ``WRAPPERS[kind]`` (any registered key:
    'plain', 'time_penalty', 'hack', 'fix'). If ``seed`` is given, the env is
    reset once with it and the action space is seeded. Wrapper kwargs go to
    the wrapper constructor."""
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    cls = WRAPPERS[kind]
    if cls is not None:
        env = cls(env, **wrapper_kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
