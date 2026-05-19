"""UMAP of joint configurations during pre-success steps, colored by skill.

Directly tests the "arm posture is the discriminator's cheat-channel"
hypothesis: if the two skills converge to the same gripper-space behaviour
but explore different regions of joint-configuration space, a 2D UMAP of
the raw ``joint_pos`` vectors should show separation by skill color.

Two panels:
  left:  per-step UMAP — every pre-success timestep across all envs is one
         point. Skill color, time-in-episode encoded as marker alpha (early
         steps fainter, late steps darker), so trajectory direction is
         visible in UMAP space.
  right: per-episode UMAP — each episode contributes one point (mean joint
         config over the pre-success trajectory). Less crowded; easier to
         see clustering.

Pure plotting; no Isaac Sim, no GPU.

Usage:

    python analysis/plot_diversity_skill_joint_umap.py \\
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

SKILL_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

# UR5e arm joints — we strip Robotiq gripper joints (mimic, ~constant during
# arm motion) so the UMAP reflects arm posture only. Falls back to all joints
# if the names list doesn't match.
ARM_JOINT_PREFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", type=pathlib.Path, default=DEFAULT_RAW_PATH)
    ap.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=None,
        help="Defaults to <raw_path.parent>/plot_joint_umap.png.",
    )
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_gripper", action="store_true", help="Include Robotiq gripper joints (default: arm only).")
    args = ap.parse_args()

    if args.out_path is None:
        args.out_path = args.raw_path.parent / "plot_joint_umap.png"

    raw = torch.load(args.raw_path, weights_only=False)
    if "joint_pos" not in raw:
        raise KeyError(
            "raw.pt has no 'joint_pos' field — re-run diversity_skill_ee_traj.py to "
            "regenerate raw.pt with the joint-pos buffer."
        )
    joint_pos = raw["joint_pos"]  # (T, N, J)
    joint_names: list[str] = list(raw.get("joint_names", []))
    first_success_step = raw["first_success_step"]
    first_episode_end_step = raw["first_episode_end_step"]
    initial_skills = raw["initial_skills"]
    num_skills = int(raw["num_skills"])
    horizon, num_envs, num_joints_full = joint_pos.shape
    print(f"[INFO] raw: horizon={horizon}, num_envs={num_envs}, joints={num_joints_full}, num_skills={num_skills}")

    # Optionally restrict to arm joints
    if joint_names and not args.include_gripper:
        arm_idx = [i for i, n in enumerate(joint_names) if any(n.startswith(p) for p in ARM_JOINT_PREFIXES)]
        if arm_idx:
            print(f"[INFO] Restricting to {len(arm_idx)} arm joints: {[joint_names[i] for i in arm_idx]}")
            joint_pos = joint_pos[:, :, arm_idx]
        else:
            print(f"[WARN] No arm joints matched in {joint_names}; using all {num_joints_full} joints.")
    else:
        print(f"[INFO] Using all {num_joints_full} joints (gripper included={args.include_gripper or not joint_names}).")

    num_joints = joint_pos.shape[-1]

    # ============================================================
    # Build (timestep, env_id, skill, step_in_episode) point set
    # ============================================================
    per_step_X: list[np.ndarray] = []
    per_step_skill: list[int] = []
    per_step_step_in_ep: list[int] = []
    per_step_env: list[int] = []
    per_episode_X: list[np.ndarray] = []
    per_episode_skill: list[int] = []
    per_episode_succ: list[bool] = []
    per_episode_len: list[int] = []

    for env_id in range(num_envs):
        skill = int(initial_skills[env_id].item())
        succeeded = bool(first_success_step[env_id] >= 0)
        if succeeded:
            cutoff = int(first_success_step[env_id].item())
        else:
            cutoff = int(first_episode_end_step[env_id].item()) + 1
        if cutoff < 2:
            continue
        jp = joint_pos[:cutoff, env_id, :].numpy()  # (cutoff, J)
        for t in range(cutoff):
            per_step_X.append(jp[t])
            per_step_skill.append(skill)
            per_step_step_in_ep.append(t)
            per_step_env.append(env_id)
        per_episode_X.append(jp.mean(axis=0))
        per_episode_skill.append(skill)
        per_episode_succ.append(succeeded)
        per_episode_len.append(cutoff)

    X_step = np.array(per_step_X)
    skills_step = np.array(per_step_skill)
    step_in_ep = np.array(per_step_step_in_ep)
    X_ep = np.array(per_episode_X)
    skills_ep = np.array(per_episode_skill)
    succ_ep = np.array(per_episode_succ)
    lens_ep = np.array(per_episode_len)
    print(f"[INFO] per-step: {len(X_step)} points across {len(X_ep)} episodes")
    print(
        "[INFO] per-episode counts: "
        + ", ".join(f"skill {s}: {int((skills_ep == s).sum())}" for s in range(num_skills))
    )

    # ============================================================
    # UMAP fits
    # ============================================================
    import umap  # imported here so the script's --help works without UMAP installed

    # Per-step UMAP — fit on the per-step matrix.
    reducer_step = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        metric="euclidean",
    )
    emb_step = reducer_step.fit_transform(X_step)

    # Per-episode UMAP — small N, use small n_neighbors.
    reducer_ep = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(5, len(X_ep) - 1)),
        min_dist=args.min_dist,
        random_state=args.seed,
        metric="euclidean",
    )
    emb_ep = reducer_ep.fit_transform(X_ep)

    # ============================================================
    # Plot
    # ============================================================
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left: per-step UMAP
    for s in range(num_skills):
        m = skills_step == s
        if not m.any():
            continue
        # alpha ramps from 0.15 (start of episode) to 0.85 (cutoff step)
        # use per-episode max length so alpha is comparable across episodes
        # of different lengths; if an episode is short it spans the full range.
        step_t = step_in_ep[m].astype(np.float32)
        # normalize per-env so each episode's max is 1.0
        env_t = np.array(per_step_env)[m]
        norm = np.zeros_like(step_t)
        for e in np.unique(env_t):
            mm = env_t == e
            top = step_t[mm].max()
            norm[mm] = step_t[mm] / max(top, 1.0)
        alpha = 0.15 + 0.7 * norm
        ax_l.scatter(
            emb_step[m, 0],
            emb_step[m, 1],
            c=SKILL_COLORS.get(s, "tab:gray"),
            s=8,
            alpha=alpha,
            label=f"skill {s} ({int(m.sum())} steps)",
            linewidths=0,
        )
    ax_l.set_xlabel("UMAP-1")
    ax_l.set_ylabel("UMAP-2")
    ax_l.set_title(
        "UMAP of per-step arm joint config (pre-success)\n"
        "marker alpha grows along episode (faint = early, dark = late)"
    )
    ax_l.legend(loc="best", fontsize=9)
    ax_l.grid(True, alpha=0.3)

    # Right: per-episode UMAP (each point = mean joint config of one episode)
    for s in range(num_skills):
        m = skills_ep == s
        if not m.any():
            continue
        ax_r.scatter(
            emb_ep[m, 0],
            emb_ep[m, 1],
            c=SKILL_COLORS.get(s, "tab:gray"),
            s=120,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
            marker="o",
            label=f"skill {s} (n={int(m.sum())})",
        )
        # annotate episode length
        for i in np.where(m)[0]:
            ax_r.annotate(
                f"{lens_ep[i]}",
                (emb_ep[i, 0], emb_ep[i, 1]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
                color="black",
            )
    ax_r.set_xlabel("UMAP-1")
    ax_r.set_ylabel("UMAP-2")
    ax_r.set_title(
        "UMAP of per-episode mean joint config\n"
        "label = pre-success episode length (steps)"
    )
    ax_r.legend(loc="best", fontsize=9)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        f"DIAYN skill joint-config UMAP — task State-Diversity, "
        f"ckpt {pathlib.Path(raw.get('checkpoint', '')).name}  "
        f"(joints used: {num_joints})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] wrote {args.out_path}")


if __name__ == "__main__":
    main()
