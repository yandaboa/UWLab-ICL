"""Visualize the valid priv_baseline experiments + takeaways.

Produces 4 figures under ``plots/priv_baseline/``:

1. ``summary_bc_bar.png`` — single-iter BC EOE comparison across methods.
2. ``summary_dagger_best.png`` — best-of-iter DAgger EOE comparison.
3. ``summary_dagger_curves.png`` — per-iter DAgger EOE trajectories.
4. ``summary_width_sweep.png`` — EOE vs hidden_dim for r4 BC and r5 DAgger best.

Eval points come from ``logs/priv_baseline/<run>/.../eval_log.json`` with
n=512 episodes per checkpoint. ±5 pp sampling-noise band shown where it makes
the comparison legible.

The Gaussian-MLP runs (r1/r2/A pre-fix) are excluded because they hit the
``MLPImagePolicy.compute_loss`` training bug — see ``valid_results_log.md``.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
PLOT_DIR = _REPO / "plots" / "priv_baseline"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_iters(run_subdir: str):
    """Return list of (iter, eoe, anytime) for a run."""
    base = _REPO / "logs" / "priv_baseline" / run_subdir
    eval_logs = list(base.rglob("eval_log.json"))
    if not eval_logs:
        return []
    with open(eval_logs[0]) as f:
        d = json.load(f)
    out = []
    for e in d.get("iterations", []):
        eoe = e.get("metrics", {}).get(
            "Metrics/task_command/end_of_episode_success_rate", 0.0
        )
        sr = e.get("success_rate", 0.0)
        out.append((int(e.get("iteration", 0)), eoe * 100, sr * 100))
    return sorted(out, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Color scheme: one hue per method family.
# ---------------------------------------------------------------------------
C_MARK = "#1f77b4"     # mark-disc (no priv obs)
C_PRIV = "#2ca02c"     # privileged disc-AR
C_AUX = "#d62728"      # aux loss
NOISE_BAND = 5.0       # ±5 pp sampling noise at n=512


# ---------------------------------------------------------------------------
# Figure 1: BC bar chart
# ---------------------------------------------------------------------------
def fig_bc_bar(out_path):
    runs = [
        ("r4 mark-disc\nd=256",    "r4_mark_disc_bc80k_d256",    C_MARK),
        ("r4 mark-disc\nd=512 ★",  "r4_mark_disc_bc80k_d512",    C_MARK),
        ("r4 mark-disc\nd=1024",   "r4_mark_disc_bc80k_d1024",   C_MARK),
        ("b priv-disc\nd=512",     "b_priv_disc_bc80k_d512",     C_PRIV),
        ("b priv-disc\nd=1024",    "b_priv_disc_bc80k_d1024",    C_PRIV),
        ("c aux w=0.1\nd=512",     "c_aux_w0p1_bc80k_d512",      C_AUX),
        ("c aux w=0.5\nd=512",     "c_aux_w0p5_bc80k_d512",      C_AUX),
        ("c aux w=1.0\nd=512",     "c_aux_w1p0_bc80k_d512",      C_AUX),
    ]
    eoes, labels, colors = [], [], []
    for label, run, color in runs:
        its = load_iters(run)
        if not its:
            continue
        labels.append(label)
        eoes.append(its[0][1])
        colors.append(color)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, eoes, color=colors, edgecolor="black", linewidth=0.5)
    for x, y in zip(xs, eoes):
        ax.text(x, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=9)

    # Baseline marker
    baseline = next((e for lbl, e in zip(labels, eoes) if "★" in lbl), None)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(len(labels) - 0.5, baseline + 0.3, f"r4 baseline = {baseline:.1f}%",
                ha="right", va="bottom", fontsize=8, color="gray")
    # Sampling noise band
    if baseline is not None:
        ax.axhspan(baseline - NOISE_BAND, baseline + NOISE_BAND,
                   color="black", alpha=0.05)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("EOE success rate (%)  •  n=512 episodes")
    ax.set_ylim(0, 100)
    ax.set_title(
        "BC (single-iter) EOE comparison: mark-disc vs privileged disc-AR vs aux-loss\n"
        "80k expert demos, augmented (perturbation) env"
    )
    ax.grid(axis="y", alpha=0.3)
    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_MARK, label="mark-disc (no priv obs)"),
        plt.Rectangle((0, 0), 1, 1, color=C_PRIV, label="priv disc-AR (priv obs)"),
        plt.Rectangle((0, 0), 1, 1, color=C_AUX, label="aux loss (priv via MSE only)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: DAgger best-of-iter bar chart
# ---------------------------------------------------------------------------
def fig_dagger_best(out_path):
    runs = [
        ("r5 mark-disc\nd=256",   "r5_mark_disc_dagger_d256_v2",  C_MARK),
        ("r5 mark-disc\nd=512",   "r5_mark_disc_dagger_d512_v2",  C_MARK),
        ("r5 mark-disc\nd=1024",  "r5_mark_disc_dagger_d1024",    C_MARK),
        ("b priv-disc\nd=512",    "b_priv_disc_dagger_d512",      C_PRIV),
        ("b priv-disc\nd=1024 ★", "b_priv_disc_dagger_d1024",     C_PRIV),
        ("c aux w=0.1\nd=512",    "c_aux_w0p1_dagger_d512",       C_AUX),
        ("c aux w=0.5\nd=512",    "c_aux_w0p5_dagger_d512",       C_AUX),
        ("c aux w=1.0\nd=512",    "c_aux_w1p0_dagger_d512",       C_AUX),
    ]
    best_eoes, best_iters, labels, colors = [], [], [], []
    for label, run, color in runs:
        its = load_iters(run)
        if not its:
            continue
        i_best = max(its, key=lambda x: x[1])
        labels.append(label)
        best_eoes.append(i_best[1])
        best_iters.append(i_best[0])
        colors.append(color)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(labels))
    ax.bar(xs, best_eoes, color=colors, edgecolor="black", linewidth=0.5)
    for x, y, it in zip(xs, best_eoes, best_iters):
        ax.text(x, y + 0.8, f"{y:.1f}\n(it{it})",
                ha="center", va="bottom", fontsize=9, linespacing=0.9)

    # Champion marker
    champion = max(best_eoes)
    ax.axhline(champion, color="black", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(len(labels) - 0.5, champion + 0.3,
            f"best = {champion:.1f}%", ha="right", va="bottom", fontsize=8, color="gray")
    ax.axhspan(champion - NOISE_BAND, champion + NOISE_BAND, color="black", alpha=0.05)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best-of-iter EOE (%)  •  n=512 episodes")
    ax.set_ylim(0, 100)
    ax.set_title(
        "DAgger (4×20k) best-of-iter EOE comparison\n"
        "augmented (perturbation) env, peg+peghole"
    )
    ax.grid(axis="y", alpha=0.3)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_MARK, label="mark-disc (no priv obs)"),
        plt.Rectangle((0, 0), 1, 1, color=C_PRIV, label="priv disc-AR (priv obs)"),
        plt.Rectangle((0, 0), 1, 1, color=C_AUX, label="aux loss"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: per-iter DAgger trajectories
# ---------------------------------------------------------------------------
def fig_dagger_curves(out_path):
    series = [
        ("mark-disc d=512",   "r5_mark_disc_dagger_d512_v2", C_MARK, "-",  "o"),
        ("mark-disc d=1024",  "r5_mark_disc_dagger_d1024",   C_MARK, "--", "s"),
        ("priv-disc d=512",   "b_priv_disc_dagger_d512",     C_PRIV, "-",  "o"),
        ("priv-disc d=1024",  "b_priv_disc_dagger_d1024",    C_PRIV, "--", "s"),
        ("aux w=0.1",         "c_aux_w0p1_dagger_d512",      C_AUX,  "-",  "o"),
        ("aux w=0.5",         "c_aux_w0p5_dagger_d512",      C_AUX,  "--", "^"),
        ("aux w=1.0",         "c_aux_w1p0_dagger_d512",      C_AUX,  ":",  "v"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for label, run, color, ls, marker in series:
        its = load_iters(run)
        if not its:
            continue
        xs = [it for it, _, _ in its]
        ys = [e for _, e, _ in its]
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                label=label, linewidth=2, markersize=8)

    # r4 BC d=512 baseline as horizontal line
    r4 = load_iters("r4_mark_disc_bc80k_d512")
    if r4:
        baseline = r4[0][1]
        ax.axhline(baseline, color="black", linestyle="-", alpha=0.5, linewidth=1)
        ax.text(3.05, baseline, f"r4 BC d=512 = {baseline:.1f}%",
                ha="left", va="center", fontsize=9, color="black")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("EOE success rate (%)  •  n=512 episodes")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.2, 3.7)
    ax.set_title("DAgger per-iter EOE trajectory across methods")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: width sweep
# ---------------------------------------------------------------------------
def fig_width_sweep(out_path):
    widths = [256, 512, 1024]
    fig, ax = plt.subplots(figsize=(8, 5.2))

    # r4 BC (single iter)
    bc = []
    for w in widths:
        run = f"r4_mark_disc_bc80k_d{w}"
        its = load_iters(run)
        if its:
            bc.append((w, its[0][1]))
    if bc:
        xs, ys = zip(*bc)
        ax.plot(xs, ys, color=C_MARK, marker="o", markersize=10, linewidth=2,
                label="mark-disc BC (iter-0)")
        for x, y in bc:
            ax.text(x, y + 1.5, f"{y:.1f}", ha="center", fontsize=9, color=C_MARK)

    # r5 DAgger best-of-iter
    dag = []
    for w in widths:
        run = f"r5_mark_disc_dagger_d{w}_v2" if w != 1024 else f"r5_mark_disc_dagger_d{w}"
        its = load_iters(run)
        if its:
            best = max(its, key=lambda x: x[1])
            dag.append((w, best[1], best[0]))
    if dag:
        xs = [w for w, _, _ in dag]
        ys = [y for _, y, _ in dag]
        ax.plot(xs, ys, color=C_MARK, marker="s", markersize=10, linewidth=2,
                linestyle="--", label="mark-disc DAgger (best-of-iter)")
        for x, y, it in dag:
            ax.text(x, y - 4, f"{y:.1f} (it{it})", ha="center", fontsize=9, color=C_MARK)

    # priv-disc BC + DAgger best (only d=512 and d=1024)
    priv_widths = [512, 1024]
    pbc = []
    for w in priv_widths:
        run = f"b_priv_disc_bc80k_d{w}"
        its = load_iters(run)
        if its:
            pbc.append((w, its[0][1]))
    if pbc:
        xs, ys = zip(*pbc)
        ax.plot(xs, ys, color=C_PRIV, marker="o", markersize=10, linewidth=2,
                label="priv-disc BC (iter-0)")

    pdag = []
    for w in priv_widths:
        run = f"b_priv_disc_dagger_d{w}"
        its = load_iters(run)
        if its:
            best = max(its, key=lambda x: x[1])
            pdag.append((w, best[1], best[0]))
    if pdag:
        xs = [w for w, _, _ in pdag]
        ys = [y for _, y, _ in pdag]
        ax.plot(xs, ys, color=C_PRIV, marker="s", markersize=10, linewidth=2,
                linestyle="--", label="priv-disc DAgger (best-of-iter)")
        for x, y, it in pdag:
            ax.text(x, y + 1.5, f"{y:.1f} (it{it})", ha="center", fontsize=9, color=C_PRIV)

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlabel("hidden_dim  (transformer trunk, 6 layers, 8 heads)")
    ax.set_ylabel("EOE success rate (%)  •  n=512 episodes")
    ax.set_ylim(0, 100)
    ax.set_title("Performance vs model size  •  augmented env, BC and DAgger")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    print(f"[plot] writing to {PLOT_DIR}")
    fig_bc_bar(PLOT_DIR / "summary_bc_bar.png")
    fig_dagger_best(PLOT_DIR / "summary_dagger_best.png")
    fig_dagger_curves(PLOT_DIR / "summary_dagger_curves.png")
    fig_width_sweep(PLOT_DIR / "summary_width_sweep.png")
    print("[plot] done.")


if __name__ == "__main__":
    main()
