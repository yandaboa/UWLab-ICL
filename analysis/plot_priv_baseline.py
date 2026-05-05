"""Plot the privileged-baseline sweep results.

Outputs three figures under ``plots/priv_baseline/``:

1. ``perf_vs_model_size.png`` — EOE success rate vs hidden_dim, one line per row
   (r1, r2, r4 = single iter; r5 = LAST DAgger iter).
2. ``perf_vs_dagger_iter.png`` — EOE success rate vs DAgger iteration for r5
   at each width. Companion line for r4 (BC, single point) for reference.
3. ``action_dist_at_state.png`` — empirical action distribution within a single
   tight (state, perturbation) bucket, per action dim, with best-fit Gaussian
   overlay. Shows whether a Gaussian head can express the BC target.

Reads eval logs from ``logs/priv_baseline/*/priv_baseline_*/*/eval_log.json``
and one zarr from ``logs/priv_baseline/r1_priv_mlp_bc20k_d256/.../dataset-iteration-0/data.zarr``
for the action distribution panel.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import zarr

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

PLOT_DIR = _REPO_ROOT / "plots" / "priv_baseline"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_GLOB = str(_REPO_ROOT / "logs" / "priv_baseline" / "*" / "priv_baseline_*" / "*" / "eval_log.json")

# Map run-dir name → (row, width). Some have a `_v2` suffix from re-runs after
# the success-filter fix; we prefer those.
_RUN_RE = re.compile(r"(r[1-5])_[a-z0-9_]+_d(\d+)(?:_v2)?$")


def parse_eval_logs() -> dict:
    """Return mapping (row, width) -> {'iters': [(iter, eoe_sr, any_sr), ...], 'src': path}."""
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(glob.glob(EVAL_GLOB)):
        run_dir = pathlib.Path(f).parts[-4]
        m = _RUN_RE.match(run_dir)
        if not m:
            continue
        row = m.group(1)
        width = int(m.group(2))
        is_v2 = run_dir.endswith("_v2")
        with open(f) as fh:
            log = json.load(fh)
        iters = []
        for e in log.get("iterations", []):
            mt = e.get("metrics", {})
            iters.append(
                (
                    e.get("iteration"),
                    mt.get("Metrics/task_command/end_of_episode_success_rate", 0.0),
                    e.get("success_rate", 0.0),
                )
            )
        # Prefer v2 over non-v2 for r5 (the v2 reruns after the success-filter fix).
        key = (row, width)
        if key not in out or (is_v2 and not out[key]["v2"]):
            out[key] = {"iters": iters, "src": f, "v2": is_v2}
    return out


def plot_perf_vs_model_size(data: dict, out_path: pathlib.Path) -> None:
    widths = [256, 512, 1024]
    fig, ax = plt.subplots(figsize=(7, 5))
    series = {
        "r1: priv-MLP BC 20k (Gaussian)": ("r1", "single", "tab:red", "o", "--"),
        "r2: priv-MLP BC 80k (Gaussian)": ("r2", "single", "tab:orange", "o", "-"),
        "r4: mark-disc BC 80k (disc-AR)": ("r4", "single", "tab:blue", "s", "-"),
        "r5: mark-disc DAgger 4×20k (disc-AR, last iter)": ("r5", "last", "tab:green", "^", "-"),
    }
    for label, (row, mode, color, marker, ls) in series.items():
        xs, ys = [], []
        for w in widths:
            iters = data.get((row, w), {}).get("iters", [])
            if not iters:
                continue
            if mode == "single":
                eoe = next((eoe for it, eoe, _ in iters if it == 0), None)
            else:
                last_iter = max(it for it, _, _ in iters)
                eoe = next(eoe for it, eoe, _ in iters if it == last_iter)
            if eoe is not None:
                xs.append(w)
                ys.append(eoe)
        if xs:
            ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, label=label, linewidth=2, markersize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlabel("hidden_dim (6 layers)")
    ax.set_ylabel("end-of-episode success rate (eval, n=512)")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Performance vs model size")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_perf_vs_dagger_iter(data: dict, out_path: pathlib.Path) -> None:
    widths = [256, 512, 1024]
    colors = {256: "tab:purple", 512: "tab:green", 1024: "tab:cyan"}
    fig, ax = plt.subplots(figsize=(7, 5))

    for w in widths:
        iters = data.get(("r5", w), {}).get("iters", [])
        if not iters:
            continue
        iters_sorted = sorted(iters, key=lambda x: x[0])
        xs = [it for it, _, _ in iters_sorted]
        ys = [eoe for _, eoe, _ in iters_sorted]
        ax.plot(xs, ys, marker="o", color=colors[w], label=f"r5 mark-disc DAgger d={w}", linewidth=2, markersize=9)

        # BC reference (r4 single point at iter 0)
        r4_iters = data.get(("r4", w), {}).get("iters", [])
        if r4_iters:
            r4_eoe = next((eoe for it, eoe, _ in r4_iters if it == 0), None)
            if r4_eoe is not None:
                ax.scatter([0], [r4_eoe], color=colors[w], marker="*", s=200, edgecolor="black",
                           linewidth=1.0, label=f"r4 mark-disc BC 80k d={w}", zorder=5)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("DAgger iteration  (iter-0 = pure BC on iter-0 expert demos)")
    ax.set_ylabel("end-of-episode success rate (eval, n=512)")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Performance vs DAgger iteration (mark-disc, full DAgger)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_action_dist_at_state(zarr_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Find the densest (state×perturbation) bucket in the priv-MLP dataset and plot
    per-dim empirical action histogram with best-fit Gaussian overlay."""
    root = zarr.open(str(zarr_path), mode="r")
    actions = np.asarray(root["data/actions"])
    ee_pose = np.asarray(root["data/obs/end_effector_pose"])
    insertive = np.asarray(root["data/obs/insertive_asset_pose"])
    receptive = np.asarray(root["data/obs/receptive_asset_pose"])
    aoff = np.asarray(root["data/obs/action_offset"])
    asc = np.asarray(root["data/obs/action_scale"])

    state_feat = np.concatenate([ee_pose[:, :3], insertive[:, :3], receptive[:, :3]], axis=1)
    pert_feat = np.concatenate([aoff, asc], axis=1)

    def quantize(x, n):
        out = np.zeros_like(x, dtype=np.int64)
        for j in range(x.shape[1]):
            edges = np.unique(np.quantile(x[:, j], np.linspace(0, 1, n + 1)))
            out[:, j] = np.clip(np.searchsorted(edges, x[:, j]) - 1, 0, max(len(edges) - 2, 0))
        return out

    state_bins = quantize(state_feat, n=10)
    pert_bins = quantize(pert_feat, n=4)
    composite = np.concatenate([state_bins, pert_bins], axis=1)

    # Group by composite key, find biggest bucket
    buckets: defaultdict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(composite):
        buckets[tuple(k)].append(i)
    sizes = sorted(((len(v), k) for k, v in buckets.items()), reverse=True)
    best_size, best_key = sizes[0]
    idx = buckets[best_key]
    cluster_actions = actions[idx]
    print(f"[plot] densest (state, perturbation) bucket has {best_size} transitions")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    dim_names = ["arm dx", "arm dy", "arm dz", "arm rx", "arm ry", "arm rz", "gripper"]
    for d in range(7):
        ax = axes.flat[d]
        vals = cluster_actions[:, d]
        ax.hist(vals, bins=40, density=True, color="tab:blue", alpha=0.55, label="empirical")
        mu, sigma = float(vals.mean()), float(vals.std(ddof=1))
        if sigma > 1e-6:
            xs = np.linspace(vals.min(), vals.max(), 200)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
            ax.plot(xs, pdf, color="tab:red", linewidth=2.5, label=f"Gaussian fit\nμ={mu:.2f}, σ={sigma:.2f}")
        ax.set_title(dim_names[d])
        ax.set_xlabel("action value (env scale)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    axes.flat[7].axis("off")
    axes.flat[7].text(
        0.05, 0.5,
        f"Bucket: densest (state×perturbation)\n"
        f"  state bins: 10/dim, perturbation bins: 4/dim\n"
        f"  cluster size: {best_size} transitions\n\n"
        f"If the empirical histogram is single-peaked and the\n"
        f"Gaussian fit hugs it tightly, a Gaussian-head MLP can\n"
        f"learn this. Multi-modal or heavy-tailed shapes mean\n"
        f"the head averages over modes and fails at eval.",
        fontsize=10, va="center", ha="left", wrap=True,
    )

    fig.suptitle(
        "Empirical action distribution at a single (state × perturbation) bucket\n"
        f"(r1_d256 priv-MLP dataset, ~7% of dataset spread retained after conditioning)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    print(f"[plot] writing to {PLOT_DIR}")
    data = parse_eval_logs()
    found = sorted((row, w) for (row, w) in data.keys())
    print(f"[plot] runs found: {found}")
    print()
    print("[plot] perf vs model size")
    plot_perf_vs_model_size(data, PLOT_DIR / "perf_vs_model_size.png")
    print("[plot] perf vs DAgger iter")
    plot_perf_vs_dagger_iter(data, PLOT_DIR / "perf_vs_dagger_iter.png")
    print("[plot] action distribution at state")
    zarr_p = (
        _REPO_ROOT / "logs" / "priv_baseline" / "r1_priv_mlp_bc20k_d256"
        / "priv_baseline_r1_priv_mlp_bc20k_d256" / "2026-05-02_07-29-07"
        / "dataset-iteration-0" / "data.zarr"
    )
    if zarr_p.exists():
        plot_action_dist_at_state(zarr_p, PLOT_DIR / "action_dist_at_state.png")
    else:
        print(f"  (zarr not found: {zarr_p})")


if __name__ == "__main__":
    main()
