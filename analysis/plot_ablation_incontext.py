"""Overlay per-iteration eval curves for in-context ablation runs.

Consumes one or more eval log files written by run_incontext_exploration.py
(via incontext_eval_log.IncontextEvalLog) and renders a multi-panel overlay
plot. The plotting code is intentionally kept close to
UWLab-patrick-private/scripts/plot_dagger_ablation.py so that the visual
style matches; the only substantive differences are that data is loaded from
local JSON log files instead of wandb, and the x-axis is DAgger iteration
rather than wandb _step.

Typical usage:
    python plot_ablation_incontext.py
        --log Baseline=logs/run-a/eval_log.json
        --log Aux-only=logs/run-b/eval_log.json
        --out plots/incontext_ablation.png

If --log is passed without a label= prefix, the log's exp_name (or the
basename of the parent directory) is used as the label.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Live under analysis/ but import incontext_eval_log from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from incontext_eval_log import IncontextEvalLog  # noqa: E402

# (key, title, use_log, ylim, show_raw)
METRICS: list[tuple[str, str, bool, tuple[float, float] | None, bool]] = [
    ("Metrics/success_rate",                                "Eval success rate",               False, (0.0, 1.05), True),
    ("Metrics/task_command/end_of_episode_success_rate",    "EOE success rate (task command)", False, (0.0, 1.05), True),
    ("Metrics/task_command/end_of_episode_pos_align_error", "EOE position align error (m)",    False, (0.0, 0.40), False),
    ("Metrics/task_command/end_of_episode_rot_align_error", "EOE rotation align error (rad)",  False, (0.0, 3.20), False),
    ("Metrics/mean_episode_length",                         "Mean episode length (steps)",     False, None,        True),
    ("Episode_Reward/success",                              "Episode reward: success",         False, None,        True),
]


def _load_log_as_dataframe(log_path: str) -> tuple[str, pd.DataFrame]:
    """Return (exp_name, iteration-indexed DataFrame) for a single eval log."""
    log = IncontextEvalLog.load(log_path)
    rows = [it.flat() for it in log.iterations]
    if not rows:
        return log.exp_name or "", pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)
    return log.exp_name or "", df


def _resolve_default_label(log_path: str, exp_name: str) -> str:
    if exp_name:
        return exp_name
    parent = os.path.basename(os.path.dirname(os.path.abspath(log_path)))
    return parent or os.path.basename(log_path)


def _parse_log_spec(spec: str) -> tuple[str | None, str]:
    """Parse a --log value of the form label=path or just path."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip() or None, path.strip()
    return None, spec.strip()


def plot_group(
    data: dict[str, pd.DataFrame],
    group_labels: list[str],
    title: str,
    out_path: str,
    metrics: list[tuple[str, str, bool, tuple[float, float] | None, bool]] = METRICS,
) -> None:
    """Render group_labels as overlays into one multi-panel figure."""
    group_data = [(lbl, data[lbl]) for lbl in group_labels if lbl in data and not data[lbl].empty]
    if not group_data:
        print(f"no data for group {title}; skipping", file=sys.stderr)
        return

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False)
    axes = axes.flatten()

    # Wong palette (colorblind-safe, Nature Methods 2011); skip yellow #F0E442.
    CB_COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#000000",
    ]
    LINESTYLES = ["-", "-", "-", "--", ":", ":", "-."]
    MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

    SMOOTH = 200
    for ax_idx, (key, panel_title, use_log, ylim, show_raw) in enumerate(metrics):
        ax = axes[ax_idx]
        any_plotted = False
        for i, (label, hist) in enumerate(group_data):
            if key not in hist.columns or hist[key].dropna().empty:
                continue
            y = hist[key].dropna()
            x = hist["iteration"].loc[y.index]
            if len(y) >= SMOOTH:
                y_smooth = y.rolling(SMOOTH, min_periods=1).mean()
            else:
                y_smooth = y
            color = CB_COLORS[i % len(CB_COLORS)]
            ls = LINESTYLES[i % len(LINESTYLES)]
            marker = MARKERS[i % len(MARKERS)]
            if show_raw:
                ax.plot(x, y, color=color, alpha=0.12, linewidth=0.8, linestyle=ls)
            n = max(len(x) // 10, 1)
            ax.plot(
                x, y_smooth, label=label, color=color, linewidth=2.0,
                linestyle=ls, marker=marker, markevery=n, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6,
            )
            any_plotted = True
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("iteration")
        if use_log and any_plotted:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=7, loc="best", frameon=True)

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def _load_metrics_spec(path: str) -> list[tuple[str, str, bool, tuple[float, float] | None, bool]]:
    """Load a JSON list of [key, title, use_log, ylim|null, show_raw] entries."""
    with open(path, "r", encoding="utf-8") as f:
        raw: list[list[Any]] = json.load(f)
    parsed: list[tuple[str, str, bool, tuple[float, float] | None, bool]] = []
    for entry in raw:
        key, title, use_log, ylim, show_raw = entry
        ylim_t: tuple[float, float] | None = (
            (float(ylim[0]), float(ylim[1])) if ylim is not None else None
        )
        parsed.append((str(key), str(title), bool(use_log), ylim_t, bool(show_raw)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay per-iteration eval curves from incontext eval logs."
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        required=True,
        help=(
            "Repeat for each run to include. Format 'label=path/to/eval_log.json' or just"
            " 'path/to/eval_log.json' (label falls back to the log's exp_name / parent dir)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="plots/incontext_ablation.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="In-context exploration - ablation overlay",
        help="Figure suptitle.",
    )
    parser.add_argument(
        "--metrics_config",
        type=str,
        default=None,
        help=(
            "Optional JSON file overriding the default METRICS list."
            " Schema: list of [key, title, use_log, [ymin,ymax]|null, show_raw]."
        ),
    )
    parser.add_argument(
        "--list_metrics",
        action="store_true",
        help="Print the union of metric keys across all input logs and exit.",
    )
    args = parser.parse_args()

    data: dict[str, pd.DataFrame] = {}
    ordered_labels: list[str] = []
    for spec in args.log:
        label, path = _parse_log_spec(spec)
        if not os.path.exists(path):
            print(f"skipping missing log: {path}", file=sys.stderr)
            continue
        exp_name, df = _load_log_as_dataframe(path)
        resolved_label = label or _resolve_default_label(path, exp_name)
        if resolved_label in data:
            print(f"duplicate label {resolved_label!r}; disambiguate with label=...", file=sys.stderr)
            resolved_label = f"{resolved_label} ({len(data)})"
        data[resolved_label] = df
        ordered_labels.append(resolved_label)
        print(f"loaded {resolved_label}: {len(df)} iterations from {path}", file=sys.stderr)

    if not data:
        print("no log files loaded; exiting", file=sys.stderr)
        return

    if args.list_metrics:
        keys: set[str] = set()
        for df in data.values():
            keys.update(df.columns)
        keys.discard("iteration")
        for k in sorted(keys):
            print(k)
        return

    metrics = _load_metrics_spec(args.metrics_config) if args.metrics_config else METRICS

    plot_group(
        data,
        ordered_labels,
        title=args.title,
        out_path=args.out,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
