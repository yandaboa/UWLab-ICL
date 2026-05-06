"""Two summary plots for the full-DAgger run + headline comparison.

1. ``full_dagger_per_iter.png`` — EOE / any-time success rate per DAgger iter
   for ``full_dagger_l8h16d512_4iters_20k``.
2. ``full_dagger_vs_bc_best.png`` — bar chart comparing the full-DAgger best
   iter against the best BC results across the priv_baseline batch:
     - mark-disc BC d=512 (no priv obs, 74.6 %)
     - priv-disc BC d=512 (with priv obs, 75.7 %)
     - priv-disc BC d=1024 (with priv obs, 74.0 %)
   plus the priv-disc DAgger best (80.1 %, the headline best DAgger).
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
PLOT_DIR = _REPO / "plots" / "priv_baseline"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_iters(eval_log_path: pathlib.Path):
    if not eval_log_path.exists():
        return []
    with open(eval_log_path) as f:
        d = json.load(f)
    out = []
    for e in d.get("iterations", []):
        eoe = e.get("metrics", {}).get(
            "Metrics/task_command/end_of_episode_success_rate", 0.0
        )
        sr = e.get("success_rate", 0.0)
        out.append((int(e.get("iteration", 0)), eoe * 100, sr * 100))
    return sorted(out, key=lambda x: x[0])


def find_eval_log(run_subdir: str) -> pathlib.Path | None:
    base = _REPO / "logs" / "priv_baseline" / run_subdir
    hits = list(base.rglob("eval_log.json"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Plot 1: full_dagger per-iter EOE + any-time
# ---------------------------------------------------------------------------
def plot_full_dagger_per_iter(out_path):
    eval_log = (
        _REPO / "logs/full_dagger/l8h16d512_4iters_20k"
        "/full_dagger_l8h16d512_4iters_20k/2026-05-06_11-22-18/eval_log.json"
    )
    iters = load_iters(eval_log)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    if iters:
        xs = [it for it, _, _ in iters]
        eoes = [e for _, e, _ in iters]
        anys = [a for _, _, a in iters]

        ax.plot(xs, eoes, color="tab:blue", marker="o", markersize=11,
                linewidth=2.5, label="EOE success rate")
        ax.plot(xs, anys, color="tab:orange", marker="s", markersize=10,
                linewidth=2, linestyle="--", label="any-time success rate")
        for x, y in zip(xs, eoes):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=10, color="tab:blue")
        for x, y in zip(xs, anys):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=9, color="tab:orange")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([
        "iter-0\n(BC, 20k)",
        "iter-1\n(full DAgger)",
        "iter-2\n(full DAgger)",
        "iter-3\n(full DAgger)",
    ])
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Success rate (%)  •  n=512 episodes")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.3, 3.3)
    ax.set_title(
        "Full DAgger — EOE & any-time success per iteration\n"
        "8L / 16H / 512D disc-AR, 4×20k demos, augmented (perturbation) env"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    if not iters or len(iters) < 4:
        ax.text(0.02, 0.97,
                f"({len(iters)}/4 iters complete; iter-3 still in flight)" if iters else "(no iters yet)",
                transform=ax.transAxes, fontsize=9, color="gray", va="top")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}  (iters: {[it for it,_,_ in iters]})")


# ---------------------------------------------------------------------------
# Plot 2: full_dagger best vs best BC across the batch
# ---------------------------------------------------------------------------
def plot_full_dagger_vs_bc_best(out_path):
    fd_log = (
        _REPO / "logs/full_dagger/l8h16d512_4iters_20k"
        "/full_dagger_l8h16d512_4iters_20k/2026-05-06_11-22-18/eval_log.json"
    )
    fd_iters = load_iters(fd_log)
    fd_best_eoe = max((e for _, e, _ in fd_iters), default=0.0)
    fd_best_iter = max(fd_iters, key=lambda x: x[1])[0] if fd_iters else None
    fd_iter0_eoe = next((e for it, e, _ in fd_iters if it == 0), 0.0)

    bars = [
        ("mark-disc BC\nd=512 (r4)\nno priv",
         next((e for _, e, _ in load_iters(find_eval_log("r4_mark_disc_bc80k_d512")) if True), 0.0),
         "tab:blue", "BC"),
        ("priv-disc BC\nd=512\npriv obs",
         next((e for _, e, _ in load_iters(find_eval_log("b_priv_disc_bc80k_d512")) if True), 0.0),
         "tab:green", "BC"),
        ("priv-disc BC\nd=1024\npriv obs",
         next((e for _, e, _ in load_iters(find_eval_log("b_priv_disc_bc80k_d1024")) if True), 0.0),
         "tab:green", "BC"),
        ("priv-disc DAgger\nd=1024 (best)\npriv obs (intervention)",
         max((e for _, e, _ in load_iters(find_eval_log("b_priv_disc_dagger_d1024"))), default=0.0),
         "darkgreen", "intervention DAgger"),
        (f"full-DAgger\n8L/16H/d=512\nno priv  (iter-{fd_best_iter})" if fd_best_iter is not None
         else "full-DAgger\n8L/16H/d=512\nno priv",
         fd_best_eoe, "tab:red", "full DAgger"),
    ]

    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    # full-dagger iter-0 (BC at start of full-DAgger pipeline) marker
    if fd_iter0_eoe > 0:
        ax.axhline(fd_iter0_eoe, color="tab:red", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(2.0, fd_iter0_eoe - 2.5,
                f"full-DAgger iter-0 BC = {fd_iter0_eoe:.1f}%",
                ha="center", va="top", fontsize=9, color="tab:red")

    # mark-disc baseline marker
    baseline = vals[0]
    ax.axhline(baseline, color="tab:blue", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0, baseline + 0.3, f"r4 baseline = {baseline:.1f}%",
            ha="left", va="bottom", fontsize=8, color="tab:blue")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best EOE success rate (%)  •  n=512 eval episodes")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Full DAgger best vs best BC / intervention-DAgger across the priv_baseline batch\n"
        "augmented (perturbation) env, peg+peghole"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}  (full_dagger best: {fd_best_eoe:.1f}% at iter {fd_best_iter})")


def main():
    print(f"[plot] writing to {PLOT_DIR}")
    plot_full_dagger_per_iter(PLOT_DIR / "full_dagger_per_iter.png")
    plot_full_dagger_vs_bc_best(PLOT_DIR / "full_dagger_vs_bc_best.png")
    print("[plot] done.")


if __name__ == "__main__":
    main()
