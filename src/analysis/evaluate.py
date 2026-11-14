#!/usr/bin/env python3
"""Evaluate one or more Stable-Baselines3 PPO models on plain LunarLander-v3.

Always uses the unmodified Gymnasium environment (no reward wrappers), so
shaping bonuses cannot inflate scores. Actions are deterministic by default.

Multiple model paths or globs can share a label; every training seed under
that label is evaluated individually and then aggregated together.

Landing rates (both are reported)
    return ≥ 200
        Fraction of episodes whose plain-env episodic return is greater than
        or equal to 200 (Gymnasium's official "solved" threshold). This is
        a return-based criterion, not a check of the terminal bonus.
    at-rest bonus (+100)
        Fraction of episodes that terminated (not truncated) with a final-step
        reward of exactly +100. LunarLander only gives that bonus when the
        lander comes to rest — the Trainer's landing definition.

Outcome categories (mutually exclusive)
    An episode is counted in exactly one of: landed-at-rest, crashed, or
    timed out (or "other" in the rare case none of those apply).

    crashed
        Terminated with a final-step reward of -100 (crash or out of
        bounds), including a crash that happens on the 1000th step.
    timed out
        Truncated at the 1000-step time limit *without* an at-rest +100
        or crash -100 on that step. This is the signature of a hovering
        policy. A lander that comes to rest exactly at step 1000 (both
        `terminated` and `truncated` can be True) is counted as landed,
        not as a timeout. The Trainer/Hacker `src/evaluate.py` can
        double-count that case in `landing_rate` and `truncated_rate`.

Usage:
    .venv/bin/python src/analysis/evaluate.py \
        --model plain=models/plain_s*.zip \
        --model time_penalty=models/time_penalty_s*.zip \
        --n-episodes 20 \
        --n-eval-seeds 3 \
        --eval-seed-start 2000 \
        --output-dir results \
        --output-stem analysis_eval_trainer

Quote globs so the shell does not expand them. Repeat --model for each label.
Writes <output-dir>/<output-stem>.csv and .md (defaults: results/eval_results.*).
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ENV_ID = "LunarLander-v3"
MAX_EPISODE_STEPS = 1000
LANDING_RETURN_THRESHOLD = 200  # official Gymnasium "solved" threshold
AT_REST_BONUS = 100
CRASH_PENALTY = -100


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--model must be LABEL=PATH_OR_GLOB, got {spec!r}"
        )
    label, pattern = spec.split("=", 1)
    label, pattern = label.strip(), pattern.strip()
    if not label or not pattern:
        raise argparse.ArgumentTypeError(
            f"--model must be LABEL=PATH_OR_GLOB, got {spec!r}"
        )
    return label, pattern


def expand_model_paths(pattern: str) -> list[Path]:
    matches = sorted(Path(p) for p in glob.glob(pattern))
    if not matches and Path(pattern).is_file():
        matches = [Path(pattern)]
    # Keep only SB3 zip archives (ignore stray files a glob might catch).
    zips = [p for p in matches if p.is_file() and p.suffix == ".zip"]
    if not zips:
        raise FileNotFoundError(
            f"no PPO .zip models matched {pattern!r}"
        )
    return zips


def eval_seeds_for(n_eval_seeds: int, seed_start: int) -> list[int]:
    return list(range(seed_start, seed_start + n_eval_seeds))


def episode_env_seed(eval_seed: int, episode_index: int, n_episodes: int) -> int:
    """Stable, non-overlapping env seed for one (eval seed, episode) pair."""
    return eval_seed * max(n_episodes, 10_000) + episode_index


def run_episode(model, env, seed: int) -> dict:
    obs, _info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total = 0.0
    length = 0
    terminated = False
    truncated = False
    final_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        final_reward = float(reward)
        total += final_reward
        length += 1
        done = bool(terminated or truncated)
    terminated = bool(terminated)
    truncated = bool(truncated)
    outcome = classify_outcome(terminated, truncated, final_reward, length)
    return {
        "return": total,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
        "final_reward": final_reward,
        "outcome": outcome,
        "landed": total >= LANDING_RETURN_THRESHOLD,
        "landed_at_rest": outcome == "landed_at_rest",
        "crashed": outcome == "crashed",
        "timed_out": outcome == "timeout",
        "other": outcome == "other",
    }


def classify_outcome(terminated: bool, truncated: bool, final_reward: float, length: int) -> str:
    """Assign exactly one outcome. Rest at the 1000-step limit counts as landed."""
    # The +100 at-rest bonus is decisive, even when TimeLimit also sets
    # truncated=True on the same step.
    if final_reward == AT_REST_BONUS:
        return "landed_at_rest"
    if terminated and final_reward == CRASH_PENALTY:
        return "crashed"
    if truncated:
        return "timeout"
    return "other"


def evaluate_model(model_path: Path, n_episodes: int, seeds: list[int], device: str) -> list[dict]:
    import gymnasium as gym
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device=device)
    env = gym.make(ENV_ID)  # plain env, no wrappers
    rows: list[dict] = []
    try:
        for eval_seed in seeds:
            for ep_i in range(n_episodes):
                result = run_episode(model, env, episode_env_seed(eval_seed, ep_i, n_episodes))
                result.update(
                    model=str(model_path),
                    eval_seed=eval_seed,
                    episode=ep_i,
                )
                rows.append(result)
    finally:
        env.close()
    return rows


def aggregate(label: str, model_paths: list[Path], episodes: list[dict], scope: str) -> dict:
    returns = [e["return"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    n = len(episodes)
    mean_score = statistics.fmean(returns) if n else float("nan")
    # Sample std (ddof=1) when we have at least two episodes; otherwise 0.
    std_score = statistics.stdev(returns) if n >= 2 else 0.0
    n_landed = sum(1 for e in episodes if e["landed"])
    n_landed_at_rest = sum(1 for e in episodes if e["landed_at_rest"])
    n_crashed = sum(1 for e in episodes if e["crashed"])
    n_timeout = sum(1 for e in episodes if e["timed_out"])
    n_other = sum(1 for e in episodes if e["other"])
    n_terminated = sum(1 for e in episodes if e["terminated"])
    n_truncated = sum(1 for e in episodes if e["truncated"])
    return {
        "label": label,
        "scope": scope,
        "n_models": len(model_paths),
        "models": ";".join(str(p) for p in model_paths),
        "n_episodes": n,
        "mean_score": mean_score,
        "std_score": std_score,
        "landing_rate": n_landed / n if n else float("nan"),
        "landing_rate_at_rest": n_landed_at_rest / n if n else float("nan"),
        "crash_rate": n_crashed / n if n else float("nan"),
        "timeout_rate": n_timeout / n if n else float("nan"),
        "other_rate": n_other / n if n else float("nan"),
        "terminated_rate": n_terminated / n if n else float("nan"),
        "truncated_rate": n_truncated / n if n else float("nan"),
        "mean_episode_length": statistics.fmean(lengths) if n else float("nan"),
        "n_landed": n_landed,
        "n_landed_at_rest": n_landed_at_rest,
        "n_crashed": n_crashed,
        "n_timeout": n_timeout,
        "n_other": n_other,
        "n_terminated": n_terminated,
        "n_truncated": n_truncated,
    }


CSV_COLUMNS = [
    "label",
    "scope",
    "n_models",
    "n_episodes",
    "mean_score",
    "std_score",
    "landing_rate",
    "landing_rate_at_rest",
    "crash_rate",
    "timeout_rate",
    "other_rate",
    "terminated_rate",
    "truncated_rate",
    "mean_episode_length",
    "n_landed",
    "n_landed_at_rest",
    "n_crashed",
    "n_timeout",
    "n_other",
    "n_terminated",
    "n_truncated",
    "models",
]


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("mean_score", "std_score", "mean_episode_length"):
                out[key] = f"{out[key]:.4f}"
            for key in (
                "landing_rate",
                "landing_rate_at_rest",
                "crash_rate",
                "timeout_rate",
                "other_rate",
                "terminated_rate",
                "truncated_rate",
            ):
                out[key] = f"{out[key]:.4f}"
            writer.writerow({k: out[k] for k in CSV_COLUMNS})


def write_markdown(path: Path, rows: list[dict], n_episodes: int, seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lunar Lander evaluation results",
        "",
        f"Evaluated on plain Gymnasium `{ENV_ID}` (no reward wrappers) with "
        "deterministic PPO actions.",
        "",
        f"Each model was run for **{n_episodes} episodes** on each of "
        f"**{len(seeds)} eval seeds** (`{seeds}`). Training seeds that share "
        "a label are reported individually and then aggregated.",
        "",
        "**Landing rate (return ≥ 200)** is the fraction of episodes whose "
        f"episodic return is ≥ {LANDING_RETURN_THRESHOLD} (Gymnasium's official "
        "solved threshold).",
        "",
        "**Landing rate (at-rest +100)** is the fraction of episodes that "
        f"terminated with a final-step reward of +{AT_REST_BONUS} (the lander "
        "came to rest). This is the Trainer's landing definition.",
        "",
        "**Crash rate**, **timeout rate** and **landing rate (at-rest +100)** "
        "are mutually exclusive and should sum to 100% (plus **other** if "
        "an episode fits none of those). A lander that comes to rest on "
        f"step {MAX_EPISODE_STEPS} (Gymnasium may set both `terminated` and "
        "`truncated`) is counted as landed, not as a timeout. Raw "
        "`truncated=True` is still recorded in the CSV as `truncated_rate` "
        "and can be higher than the exclusive timeout rate.",
        "",
        "**Crash rate** is the fraction of episodes that ended with a "
        f"final-step reward of {CRASH_PENALTY} (crash or out of bounds), "
        "including a crash on the time-limit step.",
        "",
        f"**Timeout rate** is the fraction of episodes cut off at the "
        f"{MAX_EPISODE_STEPS}-step time limit without an at-rest or crash "
        "outcome. A high timeout rate is the signature of a hovering policy.",
        "",
        "| Label | Scope | Models | Episodes | Mean score | Std | "
        "Land (≥200) | Land (+100) | Crash | Timeout | Mean length |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {scope} | {n_models} | {n_episodes} | {mean_score:.2f} | "
            "{std_score:.2f} | {land200} | {land100} | {crash} | {timeout} | "
            "{mean_len:.1f} |".format(
                label=row["label"],
                scope=row["scope"],
                n_models=row["n_models"],
                n_episodes=row["n_episodes"],
                mean_score=row["mean_score"],
                std_score=row["std_score"],
                land200=format_rate(row["landing_rate"]),
                land100=format_rate(row["landing_rate_at_rest"]),
                crash=format_rate(row["crash_rate"]),
                timeout=format_rate(row["timeout_rate"]),
                mean_len=row["mean_episode_length"],
            )
        )
    lines.extend(
        [
            "",
            "Std is the sample standard deviation of all episode returns "
            "pooled under the row (across training seeds where aggregated, "
            "eval seeds, and episodes).",
            "",
            "Generated by `src/analysis/evaluate.py`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_summary_line(row: dict) -> str:
    return (
        f"  {row['label']} [{row['scope']}]: mean={row['mean_score']:.2f} "
        f"std={row['std_score']:.2f} "
        f"land>=200={row['landing_rate']*100:.1f}% "
        f"land+100={row['landing_rate_at_rest']*100:.1f}% "
        f"crash={row['crash_rate']*100:.1f}% "
        f"timeout={row['timeout_rate']*100:.1f}% "
        f"other={row['other_rate']*100:.1f}% "
        f"len={row['mean_episode_length']:.1f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        metavar="LABEL=PATH_OR_GLOB",
        help="Repeatable. Each matching zip is evaluated on its own and then "
        "aggregated under the label. Example: --model plain=models/plain_s*.zip",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Episodes to run per model per eval seed (default: 20).",
    )
    parser.add_argument(
        "--n-eval-seeds",
        type=int,
        default=3,
        help="Number of evaluation seeds (default: 3).",
    )
    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=0,
        help="First evaluation seed; seeds are start, start+1, ... (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the CSV and Markdown reports (default: results).",
    )
    parser.add_argument(
        "--output-stem",
        default="eval_results",
        help="Filename stem for the reports (default: eval_results, so "
        "results/eval_results.csv and results/eval_results.md).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to PPO.load (default: cpu).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_episodes < 1 or args.n_eval_seeds < 1:
        print("error: --n-episodes and --n-eval-seeds must be >= 1", file=sys.stderr)
        return 2

    grouped: dict[str, list[Path]] = defaultdict(list)
    seen: set[tuple[str, Path]] = set()
    for spec in args.models:
        label, pattern = parse_model_spec(spec)
        for path in expand_model_paths(pattern):
            key = (label, path.resolve())
            if key not in seen:
                grouped[label].append(path)
                seen.add(key)

    seeds = eval_seeds_for(args.n_eval_seeds, args.eval_seed_start)
    summaries = []
    for label, model_paths in grouped.items():
        print(
            f"[eval] label={label} models={len(model_paths)} "
            f"episodes={args.n_episodes} x seeds={seeds}",
            flush=True,
        )
        episodes: list[dict] = []
        for path in model_paths:
            print(f"[eval]   loading {path}", flush=True)
            model_eps = evaluate_model(path, args.n_episodes, seeds, args.device)
            episodes.extend(model_eps)
            summaries.append(aggregate(path.stem, [path], model_eps, scope="model"))
        if len(model_paths) > 1 or model_paths[0].stem != label:
            summaries.append(aggregate(label, model_paths, episodes, scope="aggregate"))

    out_dir = args.output_dir
    csv_path = out_dir / f"{args.output_stem}.csv"
    md_path = out_dir / f"{args.output_stem}.md"
    write_csv(csv_path, summaries)
    write_markdown(md_path, summaries, args.n_episodes, seeds)
    print(f"[eval] wrote {csv_path}")
    print(f"[eval] wrote {md_path}")
    for row in summaries:
        print(format_summary_line(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
