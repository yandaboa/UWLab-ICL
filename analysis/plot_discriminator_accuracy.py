"""Bar chart of pre-success / post-success / overall discriminator accuracy
from a discriminator_eval_summary.json.

Default points at the model_5400 v1 eval; pass --summary to override.
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_SUMMARY = (
    _REPO_ROOT
    / "logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-05_12-28-43/discriminator_eval/discriminator_eval_summary.json"
)
DEFAULT_OUT = _REPO_ROOT / "analysis/discriminator_accuracy_v1_model5400.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=pathlib.Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    s = json.loads(args.summary.read_text())

    labels = ["Pre-success", "Post-success", "Overall"]
    values = [s["pre_success_accuracy"], s["post_success_accuracy"], s["overall_accuracy"]]
    counts = [s["pre_success_steps"], s["post_success_steps"], s["overall_steps"]]
    colors = ["#d9534f", "#5cb85c", "#777777"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val, n in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.1%}\n(n={n:,})",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(s["chance_accuracy"], color="black", linestyle="--", linewidth=1, label=f"chance = {s['chance_accuracy']:.0%}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("discriminator argmax accuracy")

    ckpt_name = pathlib.Path(s["checkpoint"]).stem
    fsm = s.get("first_success_step_mean")
    fsmed = s.get("first_success_step_median")
    succ_eps = s["num_episodes_with_success"]
    total_eps = s["num_episodes_total"]
    title = (
        f"DIAYN discriminator accuracy — {ckpt_name}\n"
        f"{succ_eps}/{total_eps} eps succeeded; first-success step mean={fsm:.0f}, median={fsmed:.0f}"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
