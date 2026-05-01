"""All-in-one plotter for the full_eval_2048ep_<TS> dir.

Produces three plots from the same input dir:
  1. ``eos_vs_anytime.png``  — scatter of EOE vs any-time success across
     ARCH-prefixed dirs, with Pearson r.
  2. ``arch_perf_vs_layers_heads.png`` — two side-by-side panels of EOE
     success against ``hidden_depth`` and ``n_head`` for ARCH-prefixed runs.
  3. ``iter_progression.png`` — EOE success vs DAgger iteration index for
     ITER-prefixed runs (l8_h4_d512 only).
  4. ``iter3_ckpt_sweep.png`` — EOE success vs training step within iter-3
     for ITER3CKPT-prefixed runs.

Usage:
    python plot_full_eval.py --eval_dir logs/full_eval_2048ep_<TS>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

ARCH_RE = re.compile(r"l(\d+)_h(\d+)_d(\d+)")


def load(eval_dir: str) -> dict:
    rows = {"ARCH": [], "ITER": [], "ITER3CKPT": []}
    for f in sorted(glob.glob(os.path.join(eval_dir, "*", "eval_stats.json"))):
        tag = os.path.basename(os.path.dirname(f))
        if "__" not in tag:
            continue
        prefix, suffix = tag.split("__", 1)
        if prefix not in rows:
            continue
        with open(f) as fh:
            d = json.load(fh)
        m = d.get("metrics", {}) or {}
        rows[prefix].append(dict(
            tag=suffix,
            any_t=float(d.get("success_rate", float("nan")) or 0.0),
            eos=float(m.get("Metrics/task_command/end_of_episode_success_rate", float("nan")) or 0.0),
            pos=float(m.get("Metrics/task_command/end_of_episode_pos_align_error", float("nan")) or 0.0),
            rot=float(m.get("Metrics/task_command/end_of_episode_rot_align_error", float("nan")) or 0.0),
        ))
    return rows


def plot_eos_vs_anytime(arch_rows: list, out: str) -> None:
    if not arch_rows:
        print("[skip] no ARCH rows for eos_vs_anytime", file=sys.stderr)
        return
    xs = np.array([r["any_t"] for r in arch_rows])
    ys = np.array([r["eos"] for r in arch_rows])
    labels = [r["tag"] for r in arch_rows]
    pearson = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xs, ys, s=80, color="#0072B2", edgecolor="white", linewidth=0.8, zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    # y=x reference
    lo = min(xs.min(), ys.min(), 0.0)
    hi = max(xs.max(), ys.max(), 1.0)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, linewidth=1, label="y = x (no gap)")
    ax.set_xlabel("Any-time success rate")
    ax.set_ylabel("End-of-episode success rate")
    ax.set_title(f"EOE vs any-time success across architectures (Pearson r = {pearson:.3f}, N={len(xs)})")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (pearson={pearson:.4f})")


def plot_arch_perf(arch_rows: list, out: str) -> None:
    if not arch_rows:
        print("[skip] no ARCH rows for arch_perf", file=sys.stderr)
        return
    parsed = []
    for r in arch_rows:
        m = ARCH_RE.match(r["tag"])
        if not m:
            continue
        parsed.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                       r["tag"], r["eos"], r["any_t"]))
    if not parsed:
        print("[skip] no parseable l*_h*_d* tags", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # vs hidden_depth: group by depth, show points + means
    ax = axes[0]
    by_depth: dict[int, list[float]] = {}
    for L, H, D, t, eos, _ in parsed:
        by_depth.setdefault(L, []).append(eos)
    depths = sorted(by_depth)
    for L, H, D, t, eos, _ in parsed:
        ax.scatter(L, eos, s=80, color="#0072B2", alpha=0.7, edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(f"h={H}", (L, eos), xytext=(5, 3), textcoords="offset points", fontsize=8)
    means = [np.mean(by_depth[L]) for L in depths]
    ax.plot(depths, means, "-", color="#E69F00", linewidth=2, marker="D", markersize=8, label="mean over n_head")
    ax.set_xlabel("hidden_depth (# transformer layers)")
    ax.set_ylabel("End-of-episode success rate")
    ax.set_title("EOE success vs # layers (d=512)")
    ax.set_xticks(depths); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()

    # vs n_head
    ax = axes[1]
    by_head: dict[int, list[float]] = {}
    for L, H, D, t, eos, _ in parsed:
        by_head.setdefault(H, []).append(eos)
    heads = sorted(by_head)
    for L, H, D, t, eos, _ in parsed:
        ax.scatter(H, eos, s=80, color="#009E73", alpha=0.7, edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(f"l={L}", (H, eos), xytext=(5, 3), textcoords="offset points", fontsize=8)
    means = [np.mean(by_head[H]) for H in heads]
    ax.plot(heads, means, "-", color="#CC79A7", linewidth=2, marker="D", markersize=8, label="mean over hidden_depth")
    ax.set_xlabel("n_head (# attention heads)")
    ax.set_ylabel("End-of-episode success rate")
    ax.set_title("EOE success vs # heads (d=512)")
    ax.set_xticks(heads); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_iter(iter_rows: list, out: str, title: str = "EOE success vs DAgger iteration (l8_h4_d512)") -> None:
    if not iter_rows:
        print("[skip] no ITER rows for iter_progression", file=sys.stderr)
        return
    iter_rows = sorted(iter_rows, key=lambda r: r["tag"])
    xs = [int(r["tag"].replace("iter", "")) for r in iter_rows]
    ys = [r["eos"] for r in iter_rows]
    ay = [r["any_t"] for r in iter_rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, ys, "o-", color="#0072B2", linewidth=2, markersize=10, label="EOE success")
    ax.plot(xs, ay, "s--", color="#009E73", linewidth=1.5, markersize=8, alpha=0.7, label="any-time success")
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("Success rate")
    ax.set_xticks(xs)
    ax.set_title(title); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_iter3_ckpt(rows: list, out: str) -> None:
    if not rows:
        print("[skip] no ITER3CKPT rows", file=sys.stderr)
        return
    rows = sorted(rows, key=lambda r: int(r["tag"].split("_")[1]))
    xs = [int(r["tag"].split("_")[1]) for r in rows]
    ys = [r["eos"] for r in rows]
    ay = [r["any_t"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, ys, "o-", color="#0072B2", linewidth=2, markersize=10, label="EOE success")
    ax.plot(xs, ay, "s--", color="#009E73", linewidth=1.5, markersize=8, alpha=0.7, label="any-time success")
    ax.set_xlabel("Training step within iter-3"); ax.set_ylabel("Success rate")
    ax.set_title("EOE success vs train step within iter-3 (l8_h4_d512, 10s eval)")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--out_dir", default="plots")
    args = p.parse_args()

    rows = load(args.eval_dir)
    print(f"loaded: ARCH={len(rows['ARCH'])}, ITER={len(rows['ITER'])}, ITER3CKPT={len(rows['ITER3CKPT'])}")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_eos_vs_anytime(rows["ARCH"], os.path.join(args.out_dir, "eos_vs_anytime.png"))
    plot_arch_perf(rows["ARCH"], os.path.join(args.out_dir, "arch_perf_vs_layers_heads.png"))
    plot_iter(rows["ITER"], os.path.join(args.out_dir, "iter_progression.png"))
    plot_iter3_ckpt(rows["ITER3CKPT"], os.path.join(args.out_dir, "iter3_ckpt_sweep.png"))

    # Print summary tables
    print()
    print("=== ARCH summary (training-max episode length) ===")
    for r in sorted(rows["ARCH"], key=lambda r: r["tag"]):
        print(f"  {r['tag']:<14}  any={r['any_t']:.4f}  eos={r['eos']:.4f}  pos={r['pos']:.4f}  rot={r['rot']:.4f}")

    print()
    print("=== ITER summary (l8_h4_d512 latest.ckpt of each iter) ===")
    for r in sorted(rows["ITER"], key=lambda r: r["tag"]):
        print(f"  {r['tag']:<10}  any={r['any_t']:.4f}  eos={r['eos']:.4f}")

    print()
    print("=== ITER3CKPT summary (l8_h4_d512 iter-3 ckpt sweep, 10s eval) ===")
    for r in sorted(rows["ITER3CKPT"], key=lambda r: int(r["tag"].split("_")[1])):
        print(f"  {r['tag']:<14}  any={r['any_t']:.4f}  eos={r['eos']:.4f}")


if __name__ == "__main__":
    main()
