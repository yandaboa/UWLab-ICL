"""Linear-probe each component of the State-Diversity discriminator's input.

For each of the 7 obs terms that feed the trained discriminator
(``joint_pos``, ``joint_vel``, ``end_effector_pose``,
``end_effector_vel_lin_ang_b``, ``insertive_asset_pose``,
``receptive_asset_pose``, ``insertive_asset_in_receptive_asset_frame``),
fit a logistic regression that predicts the skill label from the per-step
feature values, evaluated with 5-fold ``GroupKFold`` keyed on ``env_id``
(so train/test splits never share an env, preventing temporal leakage).
Report mean ± std test accuracy.

This pinpoints which feature carries the discriminator's classification
signal — i.e. where the symmetry-collapse "cheat channel" actually lives.

Pure post-hoc analysis. Reads only the ``raw.pt`` written by
``analysis/diversity_skill_ee_traj.py``; no Isaac Sim, no GPU.

Usage:

    python analysis/probe_diversity_skill_disc_obs.py \\
        --raw_path logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/\\
2026-05-18_11-02-44/diversity_skill_ee_traj/raw.pt
"""
import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def _build_pre_success_mask(active, first_success_step):
    """Per (t, env_id) bool: True iff still in first episode AND step is before
    that env's first-success step (or the env never succeeded)."""
    T, N = active.shape
    t_idx = torch.arange(T, device=active.device).unsqueeze(1).expand(T, N)
    fss = first_success_step.unsqueeze(0).expand(T, N)
    pre = (fss < 0) | (t_idx < fss)
    return active.bool() & pre.bool()


def _slice_disc_obs(disc_obs: torch.Tensor, term_dim: dict[str, int]) -> dict[str, torch.Tensor]:
    """Split (T, N, D) concat tensor by term in insertion order."""
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name, dim in term_dim.items():
        out[name] = disc_obs[..., offset : offset + dim].clone()
        offset += dim
    if offset != disc_obs.shape[-1]:
        raise ValueError(
            f"slice mismatch: per-term dims sum to {offset}, disc_obs has {disc_obs.shape[-1]}"
        )
    return out


def _probe_term(X_t: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> dict:
    """GroupKFold logistic regression. Returns mean/std test accuracy + per-fold."""
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    fold_acc = []
    for train_idx, test_idx in gkf.split(X_t, y, groups):
        Xtr, Xte = X_t[train_idx], X_t[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        if len(np.unique(ytr)) < 2:
            continue
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr)
        fold_acc.append(float(pipe.score(Xte, yte)))
    if not fold_acc:
        return {"mean": float("nan"), "std": float("nan"), "folds": []}
    return {"mean": float(np.mean(fold_acc)), "std": float(np.std(fold_acc)), "folds": fold_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", type=pathlib.Path, default=DEFAULT_RAW_PATH)
    ap.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=None,
        help="Defaults to <raw_path.parent>/plot_probe.png.",
    )
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()
    if args.out_path is None:
        args.out_path = args.raw_path.parent / "plot_probe.png"

    raw = torch.load(args.raw_path, weights_only=False)
    for k in ("disc_obs", "disc_term_dim", "initial_skills", "first_success_step", "active"):
        if k not in raw:
            raise KeyError(f"raw.pt missing required key '{k}'; re-run diversity_skill_ee_traj.py.")

    disc_obs = raw["disc_obs"]  # (T, N, 54)
    disc_term_dim: dict[str, int] = dict(raw["disc_term_dim"])
    active = raw["active"]
    first_success_step = raw["first_success_step"]
    initial_skills = raw["initial_skills"]
    T, N, D = disc_obs.shape
    print(f"[INFO] disc_obs: T={T}, N={N}, D={D}")
    print(f"[INFO] disc_term_dim (in concat order): {disc_term_dim}")

    # Sanity: per-term dims sum to D
    if sum(disc_term_dim.values()) != D:
        raise ValueError(
            f"Saved per-term dims sum to {sum(disc_term_dim.values())} but disc_obs has D={D}."
        )

    # Cross-check the joint_pos slice against the separately-saved joint_pos buffer.
    # If they match, the saved dict order matches the concat order.
    term_tensors = _slice_disc_obs(disc_obs, disc_term_dim)
    if "joint_pos" in raw and "joint_pos" in term_tensors:
        gap = (term_tensors["joint_pos"] - raw["joint_pos"]).abs().max().item()
        print(f"[INFO] joint_pos slice vs robot.data.joint_pos buffer: max abs diff = {gap:.2e}")
        if gap > 1e-3:
            print(
                "[WARN] joint_pos slice doesn't match robot.data.joint_pos buffer — "
                "the saved disc_term_dim order may not match the concat order. "
                "Probe results below may be on misaligned slices."
            )

    # Build the per-step keep mask
    keep = _build_pre_success_mask(active, first_success_step)  # (T, N) bool
    if not keep.any():
        raise RuntimeError("No pre-success steps to probe.")

    keep_idx = keep.nonzero(as_tuple=False)  # (M, 2): rows of (t, env_id)
    t_keep = keep_idx[:, 0]
    env_keep = keep_idx[:, 1]
    y_all = initial_skills[env_keep].numpy().astype(int)
    groups_all = env_keep.numpy().astype(int)

    n_per_skill = np.bincount(y_all, minlength=int(initial_skills.max().item()) + 1)
    print(f"[INFO] kept {len(y_all)} pre-success steps; per-skill counts: {n_per_skill.tolist()}")
    print(f"[INFO] groups (envs) covered: {len(np.unique(groups_all))}")

    # Probe each term + a global "all features" probe for comparison
    chance = float(n_per_skill.max() / n_per_skill.sum())
    print(f"[INFO] chance accuracy (majority class): {chance:.3f}")

    results: dict[str, dict] = {}
    for name, X in term_tensors.items():
        Xf = X[t_keep, env_keep, :].numpy().astype(np.float32)
        r = _probe_term(Xf, y_all, groups_all, n_splits=args.n_splits)
        results[name] = r
        print(f"  {name:46s} acc = {r['mean']:.3f} ± {r['std']:.3f}  (folds: {[f'{f:.2f}' for f in r['folds']]})")

    # Global probe
    X_all = disc_obs[t_keep, env_keep, :].numpy().astype(np.float32)
    r_all = _probe_term(X_all, y_all, groups_all, n_splits=args.n_splits)
    print(f"  {'<all features>':46s} acc = {r_all['mean']:.3f} ± {r_all['std']:.3f}")

    # ============================================================
    # Plot
    # ============================================================
    names = list(results.keys()) + ["<all features>"]
    means = [results[n]["mean"] for n in results] + [r_all["mean"]]
    stds = [results[n]["std"] for n in results] + [r_all["std"]]
    order = np.argsort(means)[::-1]
    names_o = [names[i] for i in order]
    means_o = [means[i] for i in order]
    stds_o = [stds[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    y_pos = np.arange(len(names_o))
    colors = ["tab:red" if n == "<all features>" else "tab:blue" for n in names_o]
    ax.barh(y_pos, means_o, xerr=stds_o, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(chance, color="black", linestyle="--", linewidth=1, label=f"chance ({chance:.2f})")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_o)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold GroupKFold logistic-regression accuracy")
    ax.set_xlim(0.4, 1.02)
    ax.set_title(
        "Linear probe per discriminator-obs term — predicts skill from feature.\n"
        "Higher = that feature carries the skill signal; cluster near chance = no signal in that feature alone."
    )
    # annotate values
    for y, m, s in zip(y_pos, means_o, stds_o):
        ax.text(m + 0.005, y, f"{m:.3f}", va="center", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] wrote {args.out_path}")


if __name__ == "__main__":
    main()
