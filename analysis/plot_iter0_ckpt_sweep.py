"""Plot eval performance vs training step for a fixed iteration's checkpoints.

Reads every ``<sweep_outdir>/step_<STEP>/eval_stats.json`` under the dir given
on the command line and renders end-of-episode success rate + a couple of
context metrics as a function of training step.

Usage:
    python plot_iter0_ckpt_sweep.py \\
        --sweep_outdir logs/sweep_disc_ar/l8_h4_d512/iter0_ckpt_sweep_20260429_152826 \\
        --out plots/iter0_ckpt_sweep_l8_h4_d512.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt

STEP_RE = re.compile(r"step_(\d+)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_outdir", required=True,
                   help="Dir containing step_*/eval_stats.json subdirs.")
    p.add_argument("--out", default="plots/iter0_ckpt_sweep.png",
                   help="Output PNG path.")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rows: list[tuple[int, float, float, float, float]] = []
    for f in sorted(glob.glob(os.path.join(args.sweep_outdir, "step_*", "eval_stats.json"))):
        m = STEP_RE.search(os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        step = int(m.group(1))
        with open(f) as fh:
            d = json.load(fh)
        metrics = d.get("metrics", {}) or {}
        rows.append((
            step,
            float(metrics.get("Metrics/task_command/end_of_episode_success_rate", float("nan"))),
            float(d.get("success_rate", float("nan"))),
            float(metrics.get("Metrics/task_command/end_of_episode_pos_align_error", float("nan"))),
            float(metrics.get("Metrics/task_command/end_of_episode_rot_align_error", float("nan"))),
        ))
    if not rows:
        print(f"no eval_stats.json under {args.sweep_outdir}", file=sys.stderr)
        return
    rows.sort()

    steps = [r[0] for r in rows]
    eos   = [r[1] for r in rows]
    anyt  = [r[2] for r in rows]
    pos   = [r[3] for r in rows]
    rot   = [r[4] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.plot(steps, eos, "o-", color="#0072B2", linewidth=2, markersize=8)
    ax.set_title("End-of-episode success rate (headline)")
    ax.set_xlabel("training step"); ax.set_ylabel("EOE success"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, anyt, "o-", color="#009E73", linewidth=2, markersize=8)
    ax.set_title("Any-time success rate")
    ax.set_xlabel("training step"); ax.set_ylabel("any-time success"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, pos, "o-", color="#E69F00", linewidth=2, markersize=8)
    ax.set_title("End-of-episode position align error (m)")
    ax.set_xlabel("training step"); ax.set_ylabel("pos err (m)"); ax.set_ylim(0, max(0.4, max(pos) * 1.1)); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, rot, "o-", color="#CC79A7", linewidth=2, markersize=8)
    ax.set_title("End-of-episode rotation align error (rad)")
    ax.set_xlabel("training step"); ax.set_ylabel("rot err (rad)"); ax.set_ylim(0, max(3.2, max(rot) * 1.1)); ax.grid(alpha=0.3)

    fig.suptitle(args.title or f"Iter-0 checkpoint sweep: {os.path.basename(os.path.dirname(args.sweep_outdir))}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    print()
    print("step       eos     any_t    pos_err   rot_err")
    for s, e, a, pp, rr in rows:
        print(f"{s:>8}  {e:.4f}   {a:.4f}   {pp:.4f}    {rr:.4f}")


if __name__ == "__main__":
    main()
