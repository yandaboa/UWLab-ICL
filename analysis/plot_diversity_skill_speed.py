"""Visualize per-skill end-effector speed differences.

Reads the ``raw.pt`` written by ``analysis/diversity_skill_ee_traj.py``,
finite-differences the world-frame EE position into a per-step speed
(m/step), restricts each env to its first episode up to (but excluding) the
first-success step, and produces a two-panel figure:

    top:    speed-over-time, one line per episode, colored by skill
    bottom: histogram of *mean* pre-success speed per episode, per skill

This makes the "different speed" finding from the EE-trajectory analysis
explicit: even though the spatial paths overlap in 3D, the discriminator
can — and does — separate skills along the kinetic axis (it has both
``joint_vel`` and ``end_effector_vel_lin_ang_b`` in its obs group).

Pure plotting; no Isaac Sim, no GPU.

Usage:

    python analysis/plot_diversity_skill_speed.py \\
        --raw_path logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/\\
2026-05-18_11-02-44/diversity_skill_ee_traj/raw.pt
"""
import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_RAW_PATH = (
    _REPO_ROOT
    / "logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-18_11-02-44/"
    / "diversity_skill_ee_traj/raw.pt"
)

# Env step dt for this task: decimation=12, sim.dt=1/120 → 0.1s/step.
DEFAULT_DT_S = 0.1

SKILL_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", type=pathlib.Path, default=DEFAULT_RAW_PATH)
    ap.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=None,
        help="Defaults to <raw_path.parent>/plot_speed.png.",
    )
    ap.add_argument("--dt_s", type=float, default=DEFAULT_DT_S, help="Env step dt in seconds (default 0.1).")
    args = ap.parse_args()

    if args.out_path is None:
        args.out_path = args.raw_path.parent / "plot_speed.png"

    raw = torch.load(args.raw_path, weights_only=False)
    ee_pos_w = raw["ee_pos_w"]  # (T, N, 3)
    first_success_step = raw["first_success_step"]  # (N,) -1 if never succeeded
    first_episode_end_step = raw["first_episode_end_step"]  # (N,)
    initial_skills = raw["initial_skills"]  # (N,)
    num_skills = int(raw["num_skills"])
    horizon = int(raw["horizon"])
    print(f"[INFO] raw: horizon={horizon}, num_envs={ee_pos_w.shape[1]}, num_skills={num_skills}")

    # per-step EE speed via finite difference (m / step), then m/s.
    delta = ee_pos_w[1:] - ee_pos_w[:-1]  # (T-1, N, 3)
    speed_step = delta.norm(dim=-1)  # (T-1, N) in m/step
    speed_ms = speed_step / args.dt_s  # m/s

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))

    # ---- top: speed over time per episode ---------------------------------
    per_skill_mean_speed: dict[int, list[float]] = {s: [] for s in range(num_skills)}
    per_skill_dur_s: dict[int, list[float]] = {s: [] for s in range(num_skills)}
    num_envs = ee_pos_w.shape[1]
    plotted_per_skill = {s: 0 for s in range(num_skills)}

    for env_id in range(num_envs):
        skill = int(initial_skills[env_id].item())
        if first_success_step[env_id] >= 0:
            cutoff = int(first_success_step[env_id].item())  # exclusive of success step
        else:
            cutoff = int(first_episode_end_step[env_id].item()) + 1
        # Need at least 2 steps for finite difference (so cutoff >= 2)
        if cutoff < 2:
            continue
        # speed indexed 0..cutoff-2 corresponds to ee_pos transitions 0->1 ... (cutoff-2)->(cutoff-1)
        spd = speed_ms[: cutoff - 1, env_id].numpy()
        t_s = np.arange(spd.shape[0]) * args.dt_s
        color = SKILL_COLORS.get(skill, "tab:gray")
        ax_top.plot(t_s, spd, color=color, alpha=0.6, linewidth=1.0)
        per_skill_mean_speed[skill].append(float(spd.mean()))
        per_skill_dur_s[skill].append(float(spd.shape[0] * args.dt_s))
        plotted_per_skill[skill] += 1

    ax_top.set_xlabel("time since reset (s)")
    ax_top.set_ylabel("EE speed (m/s)")
    ax_top.set_title(
        "Pre-success EE speed over time, by skill — same task, same env reset distribution.\n"
        "If the two colors separate vertically the policies move at systematically different speeds."
    )
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=SKILL_COLORS.get(s, "tab:gray"), lw=2, label=f"skill {s} ({plotted_per_skill[s]} eps)")
        for s in range(num_skills) if plotted_per_skill[s] > 0
    ]
    ax_top.legend(handles=legend, loc="upper right")
    ax_top.grid(True, alpha=0.3)

    # ---- bottom: histogram of mean speed per episode, by skill -----------
    bins = np.linspace(
        min(min(v) for v in per_skill_mean_speed.values() if v),
        max(max(v) for v in per_skill_mean_speed.values() if v),
        15,
    )
    for s in range(num_skills):
        if not per_skill_mean_speed[s]:
            continue
        ax_bot.hist(
            per_skill_mean_speed[s],
            bins=bins,
            alpha=0.55,
            color=SKILL_COLORS.get(s, "tab:gray"),
            label=f"skill {s}: mean={np.mean(per_skill_mean_speed[s]):.3f} m/s "
            f"(median={np.median(per_skill_mean_speed[s]):.3f}, n={len(per_skill_mean_speed[s])})",
            edgecolor="black",
            linewidth=0.5,
        )
    ax_bot.set_xlabel("episode-mean EE speed (m/s)")
    ax_bot.set_ylabel("# episodes")
    ax_bot.set_title("Distribution of mean pre-success EE speed per episode")
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"DIAYN skill speed comparison — {raw.get('task', '<unknown task>').split('-')[-2]} task, "
        f"ckpt {pathlib.Path(raw.get('checkpoint', '<unknown>')).name}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] wrote {args.out_path}")

    # also print the summary table
    print()
    print("=== Per-skill mean pre-success EE speed ===")
    for s in range(num_skills):
        v = per_skill_mean_speed[s]
        d = per_skill_dur_s[s]
        if not v:
            continue
        print(
            f"  skill {s} ({len(v):2d} eps): "
            f"speed mean={np.mean(v):.3f} ± {np.std(v):.3f} m/s | "
            f"duration mean={np.mean(d):.2f} ± {np.std(d):.2f} s"
        )


if __name__ == "__main__":
    main()
