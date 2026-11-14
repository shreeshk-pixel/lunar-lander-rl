#!/usr/bin/env python3
"""Plot combined learning curves from SB3 Monitor CSVs (or TensorBoard logs).

Reads one glob pattern per label, groups files that live in the same parent
directory as a single training seed (this matches SB3's layout of one
*.monitor.csv per DummyVecEnv worker), then draws a rolling mean of episode
reward versus environment timesteps with a shaded band across seeds.

Important: a Monitor placed *outside* a reward-shaping wrapper logs the
*shaped* reward, so those curves are not comparable to plain-env returns.
A Monitor placed *inside* the wrapper (as in this project's trainer) logs
the plain-env return and is comparable. Use --annotation to mark a label
whose Monitor may be recording shaped reward, e.g.:

    --annotation hover="Monitor is outside the shaping wrapper; rewards are shaped"

Usage:
    .venv/bin/python src/analysis/plot_curves.py \\
        --label plain=logs/plain_s*/*.monitor.csv \\
        --label time_penalty=logs/time_penalty_s*/*.monitor.csv \\
        --annotation hover="shaped reward (Monitor outside wrapper)" \\
        --output assets/learning_curves.png

Quote globs so the shell does not expand them. If a label's glob matches no
Monitor CSVs, TensorBoard event files under the same pattern are used
instead (requires the optional `tensorboard` package).
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_spec(spec: str, flag: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"{flag} must be LABEL=VALUE, got {spec!r}")
    label, value = spec.split("=", 1)
    label, value = label.strip(), value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError(f"{flag} must be LABEL=VALUE, got {spec!r}")
    return label, value


def expand_paths(pattern: str) -> list[Path]:
    matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    literal = Path(pattern)
    if not matches and (literal.is_file() or literal.is_dir()):
        matches = [literal]
    # If the pattern pointed at a directory, include its contents.
    expanded: list[Path] = []
    for path in matches:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.monitor.csv")))
            expanded.extend(sorted(path.glob("events.out.tfevents.*")))
            expanded.extend(sorted(path.glob("**/events.out.tfevents.*")))
        else:
            expanded.append(path)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def is_monitor_csv(path: Path) -> bool:
    name = path.name
    return path.is_file() and (name.endswith(".monitor.csv") or name.endswith(".csv"))


def is_tfevents(path: Path) -> bool:
    return path.is_file() and "tfevents" in path.name


def load_monitor_csv(path: Path) -> pd.DataFrame:
    """Load one SB3 Monitor CSV (optional `# {json}` header, then r,l,t)."""
    with path.open() as f:
        first = f.readline()
    skip = 1 if first.startswith("#") else 0
    df = pd.read_csv(path, skiprows=skip)
    missing = {"r", "l"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is not a Monitor CSV (missing columns {missing})")
    return pd.DataFrame(
        {
            "r": df["r"].astype(float),
            "l": df["l"].astype(int),
            "t": df["t"].astype(float) if "t" in df.columns else np.arange(len(df), dtype=float),
        }
    )


def merge_monitor_group(files: list[Path]) -> pd.DataFrame:
    """Merge parallel-env Monitor CSVs in one run directory into a single series.

    Rows are ordered by wall-clock `t` (SB3 Monitor's elapsed time), then
    timesteps are the cumulative sum of episode lengths — the same convention
    Stable-Baselines3 uses for env steps.
    """
    frames = [load_monitor_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    return df


def _iter_tf_records(path: Path):
    import struct

    with path.open("rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            crc_len = f.read(4)
            if len(crc_len) < 4:
                return
            data = f.read(length)
            if len(data) < length:
                return
            crc_data = f.read(4)
            if len(crc_data) < 4:
                return
            yield data


def _try_event_accumulator(files: list[Path]) -> pd.DataFrame | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None

    # EventAccumulator wants a directory; group by parent and merge.
    rows = []
    parents = sorted({p.parent.resolve() for p in files})
    for parent in parents:
        acc = EventAccumulator(str(parent))
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        reward_tag = None
        for candidate in (
            "rollout/ep_rew_mean",
            "episode/reward",
            "rollout/ep_rew_mean_raw",
        ):
            if candidate in tags:
                reward_tag = candidate
                break
        if reward_tag is None:
            for tag in tags:
                if "ep_rew" in tag or tag.endswith("/reward"):
                    reward_tag = tag
                    break
        if reward_tag is None:
            continue
        for event in acc.Scalars(reward_tag):
            rows.append({"timesteps": int(event.step), "r": float(event.value)})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("timesteps").drop_duplicates("timesteps")
    return df.reset_index(drop=True)


def load_tfevents_group(files: list[Path]) -> pd.DataFrame:
    df = _try_event_accumulator(files)
    if df is not None:
        return df
    # Lightweight fallback: scan event records for a scalar tag we recognise.
    # This avoids a hard dependency on the `tensorboard` package.
    import struct

    tags_of_interest = (b"rollout/ep_rew_mean", b"episode/reward")
    rows = []
    for path in files:
        for data in _iter_tf_records(path):
            if not any(tag in data for tag in tags_of_interest):
                continue
            # Event.step is protobuf field 2 (int64 varint) after field 1 (double wall_time).
            # We also need the simple_value float near the tag. This is best-effort.
            step = _proto_event_step(data)
            value = _proto_simple_value_near_tag(data, tags_of_interest)
            if step is None or value is None:
                continue
            rows.append({"timesteps": int(step), "r": float(value)})
    if not rows:
        raise RuntimeError(
            "matched TensorBoard event files but could not read rollout/ep_rew_mean "
            "(install the optional `tensorboard` package for a more reliable reader)"
        )
    return (
        pd.DataFrame(rows)
        .sort_values("timesteps")
        .drop_duplicates("timesteps")
        .reset_index(drop=True)
    )


def _proto_event_step(data: bytes) -> int | None:
    """Best-effort parse of Event.step (field 2, varint) from a raw Event proto."""
    i = 0
    n = len(data)
    step = None
    while i < n:
        key, i2 = _read_varint(data, i)
        if key is None:
            break
        i = i2
        field, wire = key >> 3, key & 7
        if wire == 0:  # varint
            val, i = _read_varint(data, i)
            if val is None:
                return step
            if field == 2:
                step = val
        elif wire == 1:  # 64-bit
            i += 8
        elif wire == 2:  # length-delimited
            length, i = _read_varint(data, i)
            if length is None:
                return step
            i += length
        elif wire == 5:  # 32-bit
            i += 4
        else:
            break
    return step


def _proto_simple_value_near_tag(data: bytes, tags: tuple[bytes, ...]) -> float | None:
    """Find a protobuf float (wire type 5) after a known tag string."""
    import struct

    for tag in tags:
        idx = data.find(tag)
        if idx < 0:
            continue
        # Search a small window after the tag for a 32-bit float field.
        window = data[idx + len(tag) : idx + len(tag) + 32]
        j = 0
        while j + 5 <= len(window):
            key = window[j]
            wire = key & 7
            if wire == 5 and j + 5 <= len(window):
                value = struct.unpack("<f", window[j + 1 : j + 5])[0]
                if np.isfinite(value):
                    return float(value)
                j += 5
            elif wire == 0:
                # skip varint
                j += 1
                while j < len(window) and window[j] & 0x80:
                    j += 1
                j += 1
            else:
                j += 1
    return None


def _read_varint(data: bytes, i: int) -> tuple[int | None, int]:
    shift = 0
    result = 0
    n = len(data)
    while i < n:
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
        if shift > 70:
            break
    return None, i


def collect_seed_curves(paths: list[Path]) -> list[pd.DataFrame]:
    """Group matched files by parent directory and load one curve per seed."""
    monitors = [p for p in paths if is_monitor_csv(p)]
    events = [p for p in paths if is_tfevents(p)]
    curves: list[pd.DataFrame] = []
    if monitors:
        grouped: dict[Path, list[Path]] = defaultdict(list)
        for path in monitors:
            grouped[path.parent.resolve()].append(path)
        for _parent, files in sorted(grouped.items()):
            curves.append(merge_monitor_group(files))
        return curves
    if events:
        grouped = defaultdict(list)
        for path in events:
            grouped[path.parent.resolve()].append(path)
        for _parent, files in sorted(grouped.items()):
            curves.append(load_tfevents_group(files))
        return curves
    return []


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    win = max(1, min(window, len(values)))
    series = pd.Series(values, dtype=float)
    # min_periods keeps the start of short runs visible instead of all-NaN.
    return series.rolling(window=win, min_periods=max(1, win // 10 or 1)).mean().to_numpy()


def interpolate_to_grid(curve: pd.DataFrame, grid: np.ndarray, window: int) -> np.ndarray:
    if "timesteps" not in curve.columns:
        raise ValueError("curve is missing a 'timesteps' column")
    steps = curve["timesteps"].to_numpy(dtype=float)
    smoothed = rolling_mean(curve["r"].to_numpy(dtype=float), window)
    # np.interp needs a strictly increasing x.
    order = np.argsort(steps, kind="mergesort")
    steps, smoothed = steps[order], smoothed[order]
    # Drop duplicate timesteps (keep the last value).
    _, uniq = np.unique(steps[::-1], return_index=True)
    uniq = len(steps) - 1 - uniq
    uniq = np.sort(uniq)
    steps, smoothed = steps[uniq], smoothed[uniq]
    if len(steps) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    y = np.interp(grid, steps, smoothed, left=np.nan, right=np.nan)
    # Do not extrapolate past this seed's last timestep.
    y[grid > steps[-1]] = np.nan
    y[grid < steps[0]] = np.nan
    return y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        required=True,
        metavar="LABEL=GLOB",
        help="Repeatable. Glob of Monitor CSVs (or TensorBoard event files / log dirs) "
        "for one method. Files in the same directory are treated as one seed.",
    )
    parser.add_argument(
        "--annotation",
        dest="annotations",
        action="append",
        default=None,
        metavar="LABEL=TEXT",
        help="Optional note drawn in the legend for a label (e.g. shaped-reward warning).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/learning_curves.png"),
        help="Destination PNG (default: assets/learning_curves.png).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling-mean window in episodes (default: 100).",
    )
    parser.add_argument(
        "--title",
        default="Lunar Lander learning curves",
        help="Plot title.",
    )
    parser.add_argument(
        "--solved-threshold",
        type=float,
        default=200.0,
        help="Horizontal reference line (default: 200, Gymnasium solved threshold). "
        "Pass a negative number to disable.",
    )
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--grid-points", type=int, default=500)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    annotations: dict[str, str] = {}
    for spec in args.annotations or []:
        label, text = parse_spec(spec, "--annotation")
        annotations[label] = text

    series: dict[str, list[pd.DataFrame]] = {}
    for spec in args.labels:
        label, pattern = parse_spec(spec, "--label")
        paths = expand_paths(pattern)
        curves = collect_seed_curves(paths)
        if not curves:
            print(
                f"error: no Monitor CSVs or TensorBoard event files matched "
                f"{pattern!r} for label {label!r}",
                file=sys.stderr,
            )
            return 2
        series[label] = curves
        print(f"[plot] label={label} seeds={len(curves)} files={len(paths)}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    # Distinct, colourblind-friendlier cycle.
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color") or [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    for i, (label, curves) in enumerate(series.items()):
        max_step = max(float(c["timesteps"].max()) for c in curves if len(c))
        min_step = min(float(c["timesteps"].min()) for c in curves if len(c))
        if not np.isfinite(max_step) or max_step <= 0:
            print(f"warning: label {label!r} has no timesteps to plot", file=sys.stderr)
            continue
        grid = np.linspace(min_step, max_step, num=max(args.grid_points, 50))
        stacked = np.vstack(
            [interpolate_to_grid(c, grid, args.window) for c in curves]
        )
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stacked, axis=0)
            counts = np.sum(~np.isnan(stacked), axis=0)
            std = np.zeros_like(mean)
            # Sample std is only defined where at least two seeds have data.
            mask = counts >= 2
            if np.any(mask):
                std[mask] = np.nanstd(stacked[:, mask], axis=0, ddof=1)
        color = colors[i % len(colors)]
        legend_label = label
        if label in annotations:
            legend_label = f"{label} — {annotations[label]}"
        n_seeds = stacked.shape[0]
        legend_label = f"{legend_label} (n={n_seeds})"
        ax.plot(grid, mean, color=color, lw=2, label=legend_label)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.22, linewidth=0)

    if args.solved_threshold is not None and args.solved_threshold >= 0:
        ax.axhline(
            args.solved_threshold,
            color="0.35",
            ls="--",
            lw=1,
            label=f"solved threshold ({args.solved_threshold:g})",
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Episode reward (rolling mean, window={args.window})")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
