"""Compare DIAYN discriminator argmax accuracy across checkpoints, splitting by:
- steps inside episodes that never reached a success ("no-success eps")
- pre-success steps inside episodes that did succeed
- post-success steps inside episodes that did succeed

Reads <run_dir>/discriminator_eval/model<N>_raw.pt files produced by
analysis/eval_discriminator.py.
"""

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_DIR = (
    _REPO_ROOT
    / "logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-05_12-28-43/discriminator_eval"
)
DEFAULT_OUT = _REPO_ROOT / "analysis/discriminator_accuracy_compare.png"

CKPT_RE = re.compile(r"model(\d+)_raw\.pt$")


def compute_buckets(raw):
    """Return dict of bucket -> (n_correct, n_total)."""
    true_skill = raw["true_skill"]  # [T, E]
    pred_skill = raw["pred_skill"]  # [T, E]
    success = raw["success"].bool()  # [T, E]
    active = raw["active"].bool()    # [T, E]
    first_success_step = raw["first_success_step"]  # [E], -1 if no success

    T, E = true_skill.shape
    correct = (pred_skill == true_skill)

    # ever-succeeded-by-step within first episode
    ever = torch.zeros_like(success)
    cum = torch.zeros(E, dtype=torch.bool)
    for t in range(T):
        cum = cum | (success[t] & active[t])
        ever[t] = cum

    succ_eps = (first_success_step >= 0).unsqueeze(0).expand(T, E)  # broadcast [T,E]

    no_success_mask = active & ~succ_eps
    pre_success_in_succ_mask = active & succ_eps & ~ever
    post_success_in_succ_mask = active & succ_eps & ever

    def stat(mask):
        n = int(mask.sum().item())
        c = int((correct & mask).sum().item())
        return c, n

    return {
        "no_success_eps": stat(no_success_mask),
        "pre_success_in_succ_eps": stat(pre_success_in_succ_mask),
        "post_success_in_succ_eps": stat(post_success_in_succ_mask),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=pathlib.Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    raw_files = sorted(args.raw_dir.glob("model*_raw.pt"))
    ckpts = []
    bucket_data = []
    chance = None
    for f in raw_files:
        m = CKPT_RE.search(f.name)
        if not m:
            continue
        ckpt = int(m.group(1))
        raw = torch.load(f, weights_only=False, map_location="cpu")
        buckets = compute_buckets(raw)
        ckpts.append(ckpt)
        bucket_data.append(buckets)

        # pull chance from sibling summary if available
        summary_path = f.with_name(f.name.replace("_raw.pt", "_summary.json"))
        if summary_path.exists() and chance is None:
            chance = json.loads(summary_path.read_text()).get("chance_accuracy", 0.1)

    if not ckpts:
        raise SystemExit(f"no model*_raw.pt files in {args.raw_dir}")
    if chance is None:
        chance = 0.1

    # plot grouped bars
    bucket_names = ["no_success_eps", "pre_success_in_succ_eps", "post_success_in_succ_eps"]
    bucket_labels = ["No-success eps\n(steps in failed eps)", "Pre-success\n(in successful eps)", "Post-success\n(in successful eps)"]
    bucket_colors = ["#444444", "#d9534f", "#5cb85c"]

    n_ckpt = len(ckpts)
    n_bucket = len(bucket_names)
    bar_w = 0.8 / n_bucket
    x = np.arange(n_ckpt)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (bn, bl, col) in enumerate(zip(bucket_names, bucket_labels, bucket_colors)):
        accs = []
        ns = []
        for buckets in bucket_data:
            c, n = buckets[bn]
            accs.append(c / n if n > 0 else float("nan"))
            ns.append(n)
        offsets = x + (i - (n_bucket - 1) / 2) * bar_w
        bars = ax.bar(offsets, accs, width=bar_w, color=col, edgecolor="black", linewidth=0.5, label=bl)
        for bar, a, n in zip(bars, accs, ns):
            if np.isnan(a):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.04, "n/a\n(0 steps)",
                        ha="center", va="bottom", fontsize=8, color="gray")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, a + 0.012,
                        f"{a:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8)

    ax.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"chance = {chance:.0%}")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"model_{c}.pt" for c in ckpts])
    ax.set_ylabel("discriminator argmax accuracy")
    ax.set_title("DIAYN discriminator accuracy vs training progress\n(v1 run; 256 envs / 1 episode each, random skills)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out.relative_to(_REPO_ROOT)}")
    print()
    print("Per-checkpoint bucket accuracies:")
    for ckpt, buckets in zip(ckpts, bucket_data):
        print(f"  model_{ckpt}.pt:")
        for bn in bucket_names:
            c, n = buckets[bn]
            acc_str = f"{c / n:.4f}" if n > 0 else "n/a"
            print(f"    {bn:30s} acc={acc_str}  (n={n})")


if __name__ == "__main__":
    main()
