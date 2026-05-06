"""Plot per-(reset, checkpoint) end-effector trajectories overlaying all skills.

Reads ee_traj-*.pt files written by play.py --save_ee_traj. One figure per
(reset_type, ckpt); each figure has a top-down xy panel plus a height-over-time
panel, with all 3 skills overlaid.
"""

import argparse
import pathlib
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_TRAJ_DIR = (
    _REPO_ROOT
    / "logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-05_12-28-43/rollouts/play"
)
DEFAULT_OUT_DIR = _REPO_ROOT / "plots/diversity_skill_trajectories"
DEFAULT_OUT_DIR_3D = _REPO_ROOT / "plots/diversity_skill_trajectories_3d"

FILENAME_RE = re.compile(r"ee_traj-model(\d+)-skill(\d+)-([A-Za-z]+)-\d+-\d+\.pt$")
SKILL_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", type=pathlib.Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--out_dir", type=pathlib.Path, default=None)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR_3D if args.mode == "3d" else DEFAULT_OUT_DIR
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # group by (reset, ckpt) -> {skill: ee_pos_w[T,3]}
    groups = defaultdict(dict)
    for path in sorted(args.traj_dir.glob("ee_traj-model*-skill*-*.pt")):
        m = FILENAME_RE.match(path.name)
        if not m:
            print(f"  skip (no match): {path.name}")
            continue
        ckpt = int(m.group(1))
        skill = int(m.group(2))
        reset = m.group(3)
        d = torch.load(path, weights_only=False, map_location="cpu")
        ee = d["ee_pos_w"]
        if ee.ndim == 3:
            ee = ee[:, 0, :]  # single env
        groups[(reset, ckpt)][skill] = ee.numpy()

    print(f"Loaded {sum(len(v) for v in groups.values())} trajectories across {len(groups)} (reset, ckpt) groups")

    for (reset, ckpt), skill_dict in sorted(groups.items()):
        if args.mode == "3d":
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            for skill, traj in sorted(skill_dict.items()):
                color = SKILL_COLORS[skill]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.5, label=f"skill {skill}")
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, marker="o", s=40, edgecolor="k", zorder=5)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, marker="X", s=60, edgecolor="k", zorder=5)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.set_title(f"reset={reset}  ckpt=model_{ckpt}.pt\no=start  X=end", fontsize=10)
            ax.legend(loc="best", fontsize=8)
        else:
            fig, (ax_xy, ax_zt) = plt.subplots(1, 2, figsize=(11, 4.5))
            for skill, traj in sorted(skill_dict.items()):
                color = SKILL_COLORS[skill]
                ax_xy.plot(traj[:, 0], traj[:, 1], color=color, lw=1.5, label=f"skill {skill}")
                ax_xy.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40, edgecolor="k", zorder=5)
                ax_xy.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="X", s=60, edgecolor="k", zorder=5)
                t = range(len(traj))
                ax_zt.plot(t, traj[:, 2], color=color, lw=1.5, label=f"skill {skill}")
            ax_xy.set_xlabel("x (m)")
            ax_xy.set_ylabel("y (m)")
            ax_xy.set_title("Top-down (xy)  -  o=start  X=end")
            ax_xy.set_aspect("equal", adjustable="datalim")
            ax_xy.grid(alpha=0.3)
            ax_xy.legend(loc="best", fontsize=8)
            ax_zt.set_xlabel("step")
            ax_zt.set_ylabel("z (m)")
            ax_zt.set_title("Height vs time")
            ax_zt.grid(alpha=0.3)
            ax_zt.legend(loc="best", fontsize=8)
            fig.suptitle(f"reset={reset}  ckpt=model_{ckpt}.pt", fontsize=11)
        fig.tight_layout()

        out_path = args.out_dir / f"{reset}_model{ckpt}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  wrote {out_path.relative_to(_REPO_ROOT)}  (skills={sorted(skill_dict)})")


if __name__ == "__main__":
    main()
