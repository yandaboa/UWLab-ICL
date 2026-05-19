"""Plot each (skill 0, skill 1) episode pair as its own 3D figure.

Reads the ``episodes.pt`` written by ``analysis/diversity_skill_ee_traj.py``.
For each plot, picks one skill-0 episode and one skill-1 episode whose
starting positions (in receptive-object frame) are closest — a greedy
nearest-neighbour assignment — so each panel compares the two skills from
roughly the same reset state, isolating skill-conditioned path differences
from reset variance.

One PNG per pair, written under ``<input_dir>/plot_3d_per_episode/``.
Pure plotting; no Isaac Sim, no GPU.

Usage:

    python analysis/plot_diversity_skill_ee_traj_per_episode.py \\
        --episodes_path logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/\\
2026-05-18_11-02-44/diversity_skill_ee_traj/episodes.pt
"""
import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_EPISODES_PATH = (
    _REPO_ROOT
    / "logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-18_11-02-44/"
    / "diversity_skill_ee_traj/episodes.pt"
)

SKILL_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def _greedy_pairs(starts_a: np.ndarray, starts_b: np.ndarray) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour pairing of indices from ``starts_a`` and ``starts_b``.

    Returns a list of ``(i, j)`` pairs where ``i`` indexes ``starts_a`` and
    ``j`` indexes ``starts_b``. Each index appears in at most one pair.
    Length is ``min(len(starts_a), len(starts_b))``.
    """
    a_remaining = list(range(len(starts_a)))
    b_remaining = list(range(len(starts_b)))
    pairs: list[tuple[int, int]] = []
    while a_remaining and b_remaining:
        # pairwise distances over current remaining sets
        a_pts = starts_a[a_remaining]
        b_pts = starts_b[b_remaining]
        d = np.linalg.norm(a_pts[:, None, :] - b_pts[None, :, :], axis=2)
        i_local, j_local = np.unravel_index(int(d.argmin()), d.shape)
        pairs.append((a_remaining[i_local], b_remaining[j_local]))
        a_remaining.pop(i_local)
        b_remaining.pop(j_local)
    return pairs


def _plot_pair(ep_a: dict, ep_b: dict, title: str, out_path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    for ep in (ep_a, ep_b):
        traj = ep["ee_pos_local"].numpy()
        color = SKILL_COLORS.get(ep["skill"], "tab:gray")
        ls = "-" if ep["succeeded"] else "--"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.85, linewidth=1.6, linestyle=ls)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=40, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=60, marker="x")

    # receptive object at origin (by construction — we plotted in receptive frame)
    ax.scatter([0], [0], [0], color="black", s=80, marker="*", label="receptive object")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0],
            color=SKILL_COLORS.get(ep_a["skill"], "tab:gray"), lw=2,
            label=f"skill {ep_a['skill']} (env {ep_a['env_id']}, T={ep_a['cutoff']}{', success' if ep_a['succeeded'] else ', no success'})",
        ),
        Line2D(
            [0], [0],
            color=SKILL_COLORS.get(ep_b["skill"], "tab:gray"), lw=2,
            label=f"skill {ep_b['skill']} (env {ep_b['env_id']}, T={ep_b['cutoff']}{', success' if ep_b['succeeded'] else ', no success'})",
        ),
        Line2D([0], [0], color="black", marker="o", linestyle="", markeredgecolor="black", label="start"),
        Line2D([0], [0], color="black", marker="x", linestyle="", label="cutoff (pre-success)"),
        Line2D([0], [0], color="black", marker="*", linestyle="", label="receptive object"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_xlabel("x (m, receptive-frame)")
    ax.set_ylabel("y (m, receptive-frame)")
    ax.set_zlabel("z (m, receptive-frame)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_path", type=pathlib.Path, default=DEFAULT_EPISODES_PATH)
    ap.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=None,
        help="Defaults to <episodes_path.parent>/plot_3d_per_episode/.",
    )
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = args.episodes_path.parent / "plot_3d_per_episode"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(args.episodes_path, weights_only=False)
    eps: list[dict] = d["episodes"]
    print(f"[INFO] Loaded {len(eps)} episodes from {args.episodes_path}")

    by_skill: dict[int, list[dict]] = {}
    for ep in eps:
        by_skill.setdefault(ep["skill"], []).append(ep)
    skills_present = sorted(by_skill.keys())
    print(f"[INFO] Skills present: {skills_present}; counts: " + ", ".join(f"{s}:{len(by_skill[s])}" for s in skills_present))

    if len(skills_present) < 2:
        raise ValueError(f"Need at least 2 skills for per-episode pairing, got {skills_present}.")

    # Pair skill 0 vs skill 1
    a_eps = by_skill[skills_present[0]]
    b_eps = by_skill[skills_present[1]]
    a_starts = np.stack([e["ee_pos_local"][0].numpy() for e in a_eps])
    b_starts = np.stack([e["ee_pos_local"][0].numpy() for e in b_eps])
    pairs = _greedy_pairs(a_starts, b_starts)
    print(f"[INFO] Pairing {len(a_eps)} skill-{skills_present[0]} eps with {len(b_eps)} skill-{skills_present[1]} eps → {len(pairs)} pairs")

    for idx, (i_a, i_b) in enumerate(pairs):
        ep_a = a_eps[i_a]
        ep_b = b_eps[i_b]
        start_a = ep_a["ee_pos_local"][0].numpy()
        start_b = ep_b["ee_pos_local"][0].numpy()
        start_gap = float(np.linalg.norm(start_a - start_b))
        title = (
            f"Pair {idx:02d}: skill {ep_a['skill']} env{ep_a['env_id']} vs skill {ep_b['skill']} env{ep_b['env_id']}\n"
            f"start-pos gap = {start_gap*100:.1f} cm (rec frame)"
        )
        out_path = args.out_dir / f"pair_{idx:02d}.png"
        _plot_pair(ep_a, ep_b, title, out_path)
        print(f"[INFO] wrote {out_path}  (start gap {start_gap*100:.1f} cm)")

    # Unmatched leftovers (in case skill counts differ): plot individually
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    leftover_a = [i for i in range(len(a_eps)) if i not in matched_a]
    leftover_b = [i for i in range(len(b_eps)) if i not in matched_b]
    leftovers = [(a_eps[i], skills_present[0]) for i in leftover_a] + [(b_eps[j], skills_present[1]) for j in leftover_b]
    for idx, (ep, _) in enumerate(leftovers, start=len(pairs)):
        title = f"Unpaired #{idx:02d}: skill {ep['skill']} env{ep['env_id']} (no skill-{1 - ep['skill']} counterpart)"
        out_path = args.out_dir / f"pair_{idx:02d}_unpaired.png"
        # Reuse the plotter by passing the same ep twice — but cleaner: small standalone plot here.
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        traj = ep["ee_pos_local"].numpy()
        color = SKILL_COLORS.get(ep["skill"], "tab:gray")
        ls = "-" if ep["succeeded"] else "--"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.85, linewidth=1.6, linestyle=ls)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=40, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=60, marker="x")
        ax.scatter([0], [0], [0], color="black", s=80, marker="*")
        ax.set_xlabel("x (m, receptive-frame)")
        ax.set_ylabel("y (m, receptive-frame)")
        ax.set_zlabel("z (m, receptive-frame)")
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[INFO] wrote {out_path}")

    print(f"[INFO] Done. {len(pairs) + len(leftovers)} plots in {args.out_dir}")


if __name__ == "__main__":
    main()
