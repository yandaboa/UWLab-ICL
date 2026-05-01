"""Aggregate + overlay per-iteration eval curves for the discrete-AR
architecture sweep produced by ``scripts_v2/sweep_disc_ar_arch.sh``.

Mirrors ``plot_ablation_incontext.py`` (which mirrors UWLab-patrick-private's
``plot_dagger_ablation.py``); the only differences here are:
  * auto-discovery: a single ``--sweep_dir`` walks every
    ``<sweep_dir>/<tag>/<run_name>/<timestamp>/eval_log.json`` and labels each
    line by the ``l{D}_h{H}_d{W}`` tag derived from the directory.
  * plots both the headline ``Metrics/task_command/end_of_episode_success_rate``
    *and* the any-time ``Metrics/success_rate`` so it's obvious which one is
    being read.
  * optional ``--baseline path=label`` overlays a single non-DAgger BC
    run's eval_log.json (or, with ``--baseline_eval_stats``, a single
    eval_stats.json file evaluated as iter-0).

Typical usage:
    python plot_sweep_disc_ar_arch.py \
        --sweep_dir logs/sweep_disc_ar \
        --since 2026-04-29_10-58-41 \
        --baseline_eval_stats logs/bc_baseline_disc_ar/.../iteration_0/eval_stats_latest.json \
        --baseline_label "BC (80k demos)" \
        --out plots/sweep_disc_ar_arch.png

To see what metric keys are available in your logs:
    python plot_sweep_disc_ar_arch.py --sweep_dir logs/sweep_disc_ar --list_metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Live under analysis/ but import incontext_eval_log from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from incontext_eval_log import IncontextEvalLog  # noqa: E402

# (key, panel_title, use_log, ylim, show_raw)
DEFAULT_METRICS: list[tuple[str, str, bool, tuple[float, float] | None, bool]] = [
    ("Metrics/task_command/end_of_episode_success_rate", "End-of-episode success rate (headline)", False, (0.0, 1.05), True),
    ("Metrics/task_command/end_of_episode_pos_align_error", "EOE position align error (m)",          False, (0.0, 0.40), False),
    ("Metrics/task_command/end_of_episode_rot_align_error", "EOE rotation align error (rad)",        False, (0.0, 3.20), False),
    ("Episode_Reward/abnormal_robot",                     "Episode reward: abnormal_robot",          False, None,        True),
    ("Episode_Reward/success_reward",                     "Episode reward: success_reward",          False, None,        True),
]

ARCH_TAG_RE = re.compile(r"l(?P<L>\d+)_h(?P<H>\d+)_d(?P<D>\d+)")


def _arch_sort_key(tag: str) -> tuple[int, int, int]:
    """Sort 'l{L}_h{H}_d{D}' tags by (L, H, D) so legend order is predictable."""
    m = ARCH_TAG_RE.match(tag)
    if not m:
        return (10**9, 10**9, 10**9)
    return (int(m["L"]), int(m["H"]), int(m["D"]))


def _load_eval_log(path: str) -> pd.DataFrame:
    """Iteration-indexed DataFrame from an IncontextEvalLog JSON file."""
    log = IncontextEvalLog.load(path)
    rows = [it.flat() for it in log.iterations]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


def _df_from_eval_stats_file(path: str, iteration: int = 0) -> pd.DataFrame:
    """Treat a single eval_stats.json as a one-row DataFrame at the given iter."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    flat: dict[str, Any] = {"iteration": iteration}
    flat["Metrics/success_rate"] = d.get("success_rate", float("nan"))
    metrics = d.get("metrics", {}) or {}
    for k, v in metrics.items():
        flat[k] = v
    return pd.DataFrame([flat])


def _discover_sweep_runs(sweep_dir: str, since: str | None) -> list[tuple[str, str]]:
    """Return [(tag, eval_log_path)] for every per-arch run under sweep_dir.

    Selects the LATEST timestamped run per arch tag (so a tag run twice
    only contributes its most recent log). When ``since`` is provided, runs
    older than that timestamp directory name are excluded — useful for
    isolating the most recent sweep launch.
    """
    pattern = os.path.join(sweep_dir, "*", "*", "*", "eval_log.json")
    found: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(glob.glob(pattern)):
        # path layout: sweep_dir/<tag>/sweep_disc_ar_<tag>/<timestamp>/eval_log.json
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) < 5:
            continue
        ts = parts[-2]
        tag = parts[-4]
        if since is not None and ts < since:
            continue
        found.setdefault(tag, []).append((ts, path))
    out: list[tuple[str, str]] = []
    for tag, runs in found.items():
        ts, path = sorted(runs)[-1]
        out.append((tag, path))
    return sorted(out, key=lambda p: _arch_sort_key(p[0]))


def plot_overlays(
    data: dict[str, pd.DataFrame],
    label_order: list[str],
    title: str,
    out_path: str,
    metrics: list[tuple[str, str, bool, tuple[float, float] | None, bool]] = DEFAULT_METRICS,
    horizontal_labels: set[str] | None = None,
) -> None:
    """``horizontal_labels`` lists labels that should render as a dotted horizontal
    reference line spanning the x-axis (e.g. a non-DAgger baseline) instead of a
    normal per-iteration curve. Other single-iter overlays (e.g. an in-flight
    sweep run that only has iter-0 done) still get plotted as a single point.
    """
    horizontal_labels = horizontal_labels or set()
    group_data = [(lbl, data[lbl]) for lbl in label_order if lbl in data and not data[lbl].empty]
    if not group_data:
        print(f"no data for {title}; nothing to plot", file=sys.stderr)
        return

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False)
    axes = axes.flatten()

    # Wong palette (colorblind-safe), skip yellow.
    CB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                 "#D55E00", "#000000", "#999999", "#882255", "#117733"]
    LINESTYLES = ["-"] * 10
    MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]

    for ax_idx, (key, panel_title, use_log, ylim, show_raw) in enumerate(metrics):
        ax = axes[ax_idx]
        any_plotted = False
        for i, (label, hist) in enumerate(group_data):
            if key not in hist.columns or hist[key].dropna().empty:
                continue
            y = hist[key].dropna()
            x = hist["iteration"].loc[y.index]
            color = CB_COLORS[i % len(CB_COLORS)]
            ls = LINESTYLES[i % len(LINESTYLES)]
            marker = MARKERS[i % len(MARKERS)]
            # Explicit baselines render as a dotted horizontal reference line
            # spanning the x-axis. Sweep runs with only one iteration so far
            # still get plotted as a single point so it's visible they're
            # in flight, not a finished baseline.
            if label in horizontal_labels:
                ax.axhline(
                    y=float(y.iloc[0]), color=color, linewidth=2.0, linestyle=":",
                    label=label, alpha=0.85,
                )
            else:
                ax.plot(
                    x, y, label=label, color=color, linewidth=2.0, linestyle=ls,
                    marker=marker, markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                )
            any_plotted = True
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("DAgger iteration")
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


def _load_metrics_spec(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    parsed = []
    for entry in raw:
        key, ttl, use_log, ylim, show_raw = entry
        ylim_t = (float(ylim[0]), float(ylim[1])) if ylim is not None else None
        parsed.append((str(key), str(ttl), bool(use_log), ylim_t, bool(show_raw)))
    return parsed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot sweep_disc_ar_arch sweep eval curves (per-iteration overlay).",
    )
    p.add_argument("--sweep_dir", type=str, default="logs/sweep_disc_ar",
                   help="Root sweep dir (default: logs/sweep_disc_ar).")
    p.add_argument("--since", type=str, default=None,
                   help="Only include run timestamps >= this (e.g. '2026-04-29_10-58-41'). "
                        "Useful when previous sweep runs left stale eval_log.json files behind.")
    p.add_argument("--baseline_eval_stats", type=str, default=None,
                   help="Optional path to a single eval_stats.json (e.g. BC baseline) — "
                        "rendered as a single-iter horizontal reference at iter=0.")
    p.add_argument("--baseline_label", type=str, default="baseline",
                   help="Legend label for --baseline_eval_stats.")
    p.add_argument("--out", type=str, default="plots/sweep_disc_ar_arch.png",
                   help="Output PNG path.")
    p.add_argument("--title", type=str,
                   default="Discrete-AR head: arch sweep (hidden_depth × n_head, hidden_dim=512)",
                   help="Figure suptitle.")
    p.add_argument("--metrics_config", type=str, default=None,
                   help="Optional JSON list overriding DEFAULT_METRICS.")
    p.add_argument("--list_metrics", action="store_true",
                   help="Print the union of metric keys across all input logs and exit.")
    args = p.parse_args()

    runs = _discover_sweep_runs(args.sweep_dir, since=args.since)
    if not runs:
        print(f"no eval_log.json found under {args.sweep_dir}"
              + (f" with timestamp >= {args.since}" if args.since else ""),
              file=sys.stderr)
        return

    data: dict[str, pd.DataFrame] = {}
    label_order: list[str] = []
    for tag, path in runs:
        df = _load_eval_log(path)
        if df.empty:
            print(f"[skip empty] {tag}: {path}", file=sys.stderr)
            continue
        data[tag] = df
        label_order.append(tag)
        print(f"[loaded] {tag}: {len(df)} iterations from {path}", file=sys.stderr)

    horizontal_labels: set[str] = set()
    if args.baseline_eval_stats is not None:
        if not os.path.exists(args.baseline_eval_stats):
            print(f"baseline file missing: {args.baseline_eval_stats}", file=sys.stderr)
        else:
            df = _df_from_eval_stats_file(args.baseline_eval_stats, iteration=0)
            data[args.baseline_label] = df
            # Put the baseline first so it's foreground in the legend.
            label_order = [args.baseline_label] + label_order
            horizontal_labels.add(args.baseline_label)
            print(f"[loaded baseline] {args.baseline_label}: 1 iteration from {args.baseline_eval_stats}",
                  file=sys.stderr)

    if args.list_metrics:
        keys: set[str] = set()
        for df in data.values():
            keys.update(df.columns)
        keys.discard("iteration")
        for k in sorted(keys):
            print(k)
        return

    metrics = _load_metrics_spec(args.metrics_config) if args.metrics_config else DEFAULT_METRICS
    plot_overlays(
        data, label_order, title=args.title, out_path=args.out,
        metrics=metrics, horizontal_labels=horizontal_labels,
    )


if __name__ == "__main__":
    main()
