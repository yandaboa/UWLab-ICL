"""Compare Depth-DAgger vs RGB-DAgger student train/eval success rates.

Pulls all runs from the two W&B projects, plots each run's full history for
``Metrics/success_student_train`` and ``Metrics/success_student_eval``, and
saves two PNGs side-by-side coloured by modality.

Usage:
    python compare_depth_vs_rgb_dagger.py --out_dir .
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "learning-to-improve"
PROJECTS: Dict[str, str] = {
    "RGB": (
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-"
        "Pretrained-Weighted-PCTeacher-FullSysidDR-v0"
    ),
    "Depth": (
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-"
        "Pretrained-Weighted-PCTeacher-Lean-FullSysidDR-v0"
    ),
}
COLORS = {"RGB": "tab:blue", "Depth": "tab:orange"}
METRICS = {
    "student_train": "Metrics/success_student_train",
    "student_eval": "Metrics/success_student_eval",
}


def fetch_histories(entity: str, project: str, keys: List[str]) -> List[pd.DataFrame]:
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    dfs: List[pd.DataFrame] = []
    for r in runs:
        df = r.history(keys=keys, pandas=True, samples=10_000)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=keys, how="all")
        df["_run_name"] = r.name
        df["_run_id"] = r.id
        dfs.append(df)
    return dfs


def plot_metric(
    histories: Dict[str, List[pd.DataFrame]],
    metric_key: str,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, dfs in histories.items():
        color = COLORS[tag]
        for i, df in enumerate(dfs):
            if metric_key not in df.columns:
                continue
            sub = df.dropna(subset=[metric_key])
            if sub.empty:
                continue
            label = tag if i == 0 else None
            ax.plot(
                sub["_step"],
                sub[metric_key],
                color=color,
                alpha=0.75,
                linewidth=1.4,
                label=label,
            )
    ax.set_xlabel("step")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    keys = list(METRICS.values())
    histories: Dict[str, List[pd.DataFrame]] = {}
    for tag, proj in PROJECTS.items():
        print(f"Fetching runs for {tag} ({proj}) ...")
        dfs = fetch_histories(ENTITY, proj, keys)
        n_pts = sum(len(d) for d in dfs)
        print(f"  {len(dfs)} run(s) with data, {n_pts} total history rows")
        histories[tag] = dfs

    os.makedirs(args.out_dir, exist_ok=True)
    plot_metric(
        histories,
        METRICS["student_train"],
        "Student train success: Depth vs RGB DAgger (FullSysidDR)",
        os.path.join(args.out_dir, "depth_vs_rgb_student_train_success.png"),
    )
    plot_metric(
        histories,
        METRICS["student_eval"],
        "Student eval success: Depth vs RGB DAgger (FullSysidDR)",
        os.path.join(args.out_dir, "depth_vs_rgb_student_eval_success.png"),
    )


if __name__ == "__main__":
    main()
