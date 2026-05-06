"""Plot the trained priv-MLP Gaussian vs trained mark-disc per-bin probability
distribution at one representative state, alongside the empirical action targets
in the matching (state × perturbation) bucket.

Improvements over v1:

1. Pick a bucket with HIGH empirical action std (so multimodality is actually
   present in the data) rather than the densest one, which tends to be a
   near-stationary state where both models trivially agree.
2. Query the policies at a single REAL transition's observation (the bucket's
   first member) rather than at the centroid of all 43 — avoids off-manifold
   averaged states.
3. Extract the disc-AR's actual per-bin softmax probabilities (one bar per bin
   over the 100 arm bins, sigmoid for the gripper) conditional on the AR-greedy
   choice of prior dims. This shows the categorical's shape directly, without
   the sampling noise / low-entropy ambiguity from v1.

For each of the 7 action dims:
  - Empirical histogram: actions in the bucket from the dataset.
  - Gaussian curve: priv-MLP's predicted Normal(μ, σ) at the query state.
  - Bar plot: mark-disc's per-bin probability mass at the query state, scaled
    so its bars sum to 1 (i.e. it's a true categorical PMF, drawn at the bin
    centers in env-scale units).
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402

PLOT_OUT = _REPO_ROOT / "plots" / "priv_baseline" / "gaussian_vs_discrete_at_state_v8_vs_r4.png"
PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

# ``PRIV_MLP_RUN`` is the latest valid Gaussian-MLP policy (v8, bug-fixed
# compute_loss + corrected env on PrivilegedTrainCfg). ``MARK_DISC_RUN`` is the
# non-privileged disc-AR baseline (r4) — same shape_meta keys (BasePolicyCfg)
# so both policies' normalizers will filter out anything from the dataset they
# don't recognize.
PRIV_MLP_RUN = (
    _REPO_ROOT / "logs/priv_baseline/a_priv_mlp_noperturb_d2048_50k_v8"
    "/priv_baseline_a_priv_mlp_noperturb_d2048_50k_v8"
)
MARK_DISC_RUN = (
    _REPO_ROOT / "logs/priv_baseline/r4_mark_disc_bc80k_d512"
    "/priv_baseline_r4_mark_disc_bc80k_d512"
)
# Use r1's zarr for bucketing: it has ``action_offset`` and ``action_scale``
# obs keys (from the PrivilegedKnown augmented env), needed to bucket by
# (state, perturbation) and find a high-empirical-std state where multimodality
# is actually present in the data. The state's BasePolicyCfg keys also feed
# both policies.
PRIV_MLP_DATASET = (
    _REPO_ROOT / "logs/priv_baseline/r1_priv_mlp_bc20k_d256"
    "/priv_baseline_r1_priv_mlp_bc20k_d256/2026-05-02_07-29-07"
    "/dataset-iteration-0/data.zarr"
)
MIN_BUCKET_SIZE = 15  # require at least this many transitions for a reliable empirical hist


def latest_ckpt(run_root: pathlib.Path) -> pathlib.Path:
    iter0 = next(run_root.glob("*/iteration_0/checkpoints"))
    ckpts = sorted(iter0.glob("step_*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no step_*.ckpt under {iter0}")
    return ckpts[-1]


def load_policy(ckpt_path: pathlib.Path, device: torch.device):
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    assert isinstance(workspace, BaseWorkspace)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    return policy.eval().to(device), cfg


def find_bucket_with_high_std(zarr_path: pathlib.Path, min_size: int) -> tuple[dict, list[int]]:
    """Among (state×perturbation) buckets with size >= ``min_size``, return the
    one whose actions have the highest mean per-dim std (so multimodality / wide
    spread is most visible)."""
    root = zarr.open(str(zarr_path), mode="r")
    out = {
        k: np.asarray(root["data/" + k])
        for k in [
            "actions",
            "obs/end_effector_pose",
            "obs/end_effector_vel_lin_ang_b",
            "obs/joint_pos",
            "obs/joint_vel",
            "obs/insertive_asset_pose",
            "obs/receptive_asset_pose",
            "obs/insertive_asset_in_receptive_asset_frame",
            "obs/prev_actions",
            "obs/action_offset",
            "obs/action_scale",
        ]
    }

    state_feat = np.concatenate(
        [out["obs/end_effector_pose"][:, :3], out["obs/insertive_asset_pose"][:, :3], out["obs/receptive_asset_pose"][:, :3]],
        axis=1,
    )
    pert_feat = np.concatenate([out["obs/action_offset"], out["obs/action_scale"]], axis=1)

    def quantize(x, n):
        q = np.zeros_like(x, dtype=np.int64)
        for j in range(x.shape[1]):
            edges = np.unique(np.quantile(x[:, j], np.linspace(0, 1, n + 1)))
            q[:, j] = np.clip(np.searchsorted(edges, x[:, j]) - 1, 0, max(len(edges) - 2, 0))
        return q

    state_bins = quantize(state_feat, n=10)
    pert_bins = quantize(pert_feat, n=4)
    composite = np.concatenate([state_bins, pert_bins], axis=1)

    buckets: defaultdict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(composite):
        buckets[tuple(k)].append(i)

    best = None
    for key, idx in buckets.items():
        if len(idx) < min_size:
            continue
        a = out["actions"][idx]
        # Average per-dim std over the 6 ARM dims (dims 0..5; dim 6 = gripper is bimodal by design)
        score = a[:, :6].std(axis=0).mean()
        if best is None or score > best[0]:
            best = (score, len(idx), key, idx)

    if best is None:
        raise RuntimeError("no bucket met min_size; lower MIN_BUCKET_SIZE")
    score, sz, key, idx = best
    print(f"[plot] selected bucket: size={sz}, mean per-dim arm std={score:.3f}")
    return out, idx


def build_obs_dict(arrays: dict, query_idx: int, keys: list[str], horizon: int, device: torch.device):
    """Use the obs at ``query_idx`` (one real transition) tiled across the time dim."""
    obs_dict = {}
    for k in keys:
        full_key = "obs/" + k
        arr = arrays[full_key]
        picked = arr[query_idx:query_idx + 1]  # [1, D]
        tiled = np.broadcast_to(picked, (1, horizon, *picked.shape[1:])).copy()
        obs_dict[k] = torch.from_numpy(tiled).to(device).float()
    return obs_dict


def disc_ar_per_dim_probs(disc_policy, disc_obs, T, device, num_samples: int = 1024):
    """Estimate the true per-dim marginal P(a_k | state) by Monte-Carlo over the
    autoregressive joint:

        P(a_k | state) = E_{a_{<k} ~ P(a_{<k} | state)} [softmax(bin_proj(h_k))]

    We run ``num_samples`` independent stochastic AR rollouts; at each AR step we
    record the conditional softmax over bins given the sampled prior dims, then
    average across rollouts. For the gripper dim we record P(gripper=close) per
    rollout (a Bernoulli) and average.

    Returns:
      probs: list of length D; arm dims → np.array shape (num_bins,) (marginal),
             gripper → np.array shape (2,) [P(open), P(close)].
      bin_centers: list of length D in env-scale units.
      greedy_action: greedy (argmax) action at this state, for reference.
    """
    head = disc_policy.output_head
    D = disc_policy.action_dim
    num_bins = head.num_bins
    arm_centers = head.arm_bin_centers.detach().cpu().numpy()

    # Monte-Carlo estimate: accumulate per-dim conditional softmax across many
    # stochastic rollouts. Each rollout calls step_inference once per dim;
    # within a rollout the conditioning chain (a_0, ..., a_{k-1}) is sampled,
    # so averaging the recorded softmaxes is an unbiased estimate of the marginal
    # P(a_k | state).
    arm_acc = np.zeros((D, num_bins))
    gripper_acc = 0.0
    counts = np.zeros(D, dtype=np.int64)

    original_step = head.step_inference

    rollout_idx = {"k": 0}

    def patched_step(hidden, dim, sample):
        # Record the softmax/sigmoid given the AR-sampled prior dims of THIS rollout.
        if dim == head.gripper_dim:
            logit = head.gripper_proj(hidden).squeeze(-1)
            p_close = torch.sigmoid(logit).cpu().numpy()
            nonlocal_acc(p_close, dim)
        else:
            logits = head.bin_proj(hidden)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            nonlocal_acc(probs, dim)
        return original_step(hidden, dim, sample)

    def nonlocal_acc(probs, dim):
        if dim == head.gripper_dim:
            nonlocal gripper_acc
            gripper_acc += float(probs[0])
        else:
            arm_acc[dim] += probs
        counts[dim] += 1

    head.step_inference = patched_step
    disc_policy.sample_action = True  # stochastic AR — samples a_{<k} from the model's joint

    attn = torch.zeros(1, T, dtype=torch.bool, device=device)
    attn[:, 0] = True

    def fresh_input():
        # predict_action pops "attention_mask" from the dict, so build a new dict per call.
        d = dict(disc_obs)
        d["attention_mask"] = attn
        return d

    try:
        with torch.no_grad():
            for _ in range(num_samples):
                _ = disc_policy.predict_action(fresh_input())
    finally:
        head.step_inference = original_step

    # Greedy reference (one greedy AR pass without recording).
    disc_policy.sample_action = False
    with torch.no_grad():
        out = disc_policy.predict_action(fresh_input())
    greedy_action = out["action"]
    if greedy_action.ndim == 3:
        greedy_action = greedy_action[:, 0, :]
    greedy_action = greedy_action.cpu().numpy()[0]
    disc_policy.sample_action = True  # restore for downstream callers

    # Normalize.
    probs_per_dim = []
    centers_per_dim = []
    for d in range(D):
        if d == head.gripper_dim:
            p_close = gripper_acc / max(counts[d], 1)
            probs_per_dim.append(np.array([1.0 - p_close, p_close]))
            centers_per_dim.append(np.array([-1.0, 1.0]))
        else:
            mean_probs = arm_acc[d] / max(counts[d], 1)
            mean_probs /= mean_probs.sum()  # numerical sanity
            probs_per_dim.append(mean_probs)
            centers_per_dim.append(arm_centers)
    return probs_per_dim, centers_per_dim, greedy_action


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[plot] using device {device}")

    print("[plot] loading priv-MLP")
    mlp_ckpt = latest_ckpt(PRIV_MLP_RUN); print(f"  ckpt: {mlp_ckpt}")
    mlp_policy, mlp_cfg = load_policy(mlp_ckpt, device)
    print("[plot] loading mark-disc")
    disc_ckpt = latest_ckpt(MARK_DISC_RUN); print(f"  ckpt: {disc_ckpt}")
    disc_policy, disc_cfg = load_policy(disc_ckpt, device)

    print("[plot] selecting bucket with high empirical action std")
    arrays, idx = find_bucket_with_high_std(PRIV_MLP_DATASET, MIN_BUCKET_SIZE)
    query_idx = idx[0]
    print(f"[plot] query state = transition #{query_idx} (one real obs from the bucket)")

    mlp_keys = list(mlp_cfg.task.shape_meta.obs.keys())
    disc_keys = list(disc_cfg.task.shape_meta.obs.keys())
    mlp_obs = build_obs_dict(arrays, query_idx, mlp_keys, mlp_cfg.horizon, device)
    disc_obs = build_obs_dict(arrays, query_idx, disc_keys, disc_cfg.horizon, device)

    # ---- priv-MLP forward → Normal(mean, std) per dim, in env scale ----
    with torch.no_grad():
        nobs = mlp_policy.normalizer.normalize({k: v for k, v in mlp_obs.items() if k in mlp_policy.normalizer.params_dict})
        from diffusion_policy.common.pytorch_util import dict_apply
        To = mlp_policy.n_obs_steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        feat = mlp_policy.obs_encoder(this_nobs).reshape(1, To, -1).reshape(1, -1)
        dist = mlp_policy.forward(feat)
        n_mean = dist.mean.cpu().numpy()[0]
        n_std = dist.scale.cpu().numpy()[0]
        action_norm = mlp_policy.normalizer.params_dict["action"]
        scale = action_norm["scale"].cpu().numpy()
        offset = action_norm["offset"].cpu().numpy()
        gauss_mean = (n_mean - offset) / scale
        gauss_std = np.abs(1.0 / scale) * n_std
    print(f"[plot] priv-MLP gaussian mean: {np.round(gauss_mean, 3).tolist()}")
    print(f"[plot] priv-MLP gaussian std : {np.round(gauss_std, 3).tolist()}")

    # ---- mark-disc TRUE per-dim marginal P(a_k | state) via MC over the joint ----
    probs_per_dim, centers_per_dim, greedy = disc_ar_per_dim_probs(
        disc_policy, disc_obs, disc_cfg.horizon, device, num_samples=2048,
    )
    print(f"[plot] disc greedy action: {np.round(greedy, 3).tolist()}")
    for d in range(7):
        p = probs_per_dim[d]
        if len(p) == 2:
            print(f"  dim {d} (gripper): P(open={-1})={p[0]:.3f}, P(close={1})={p[1]:.3f}")
        else:
            top5 = np.argsort(-p)[:5]
            print(f"  dim {d}: top-5 bin probs = "
                  + ", ".join(f"{centers_per_dim[d][i]:+.1f}→{p[i]:.3f}" for i in top5))

    # ---- empirical actions ----
    empirical = arrays["actions"][idx]

    # ---- plot ----
    dim_names = ["arm dx", "arm dy", "arm dz", "arm rx", "arm ry", "arm rz", "gripper"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for d in range(7):
        ax = axes.flat[d]
        emp = empirical[:, d]
        centers = centers_per_dim[d]
        probs = probs_per_dim[d]

        if d == 6:  # gripper: discrete -1 / +1
            x_lo, x_hi = -1.5, 1.5
            ax.bar(centers, probs, width=0.4, color="tab:green", alpha=0.7, label="mark-disc P(bin)")
            ax.hist(emp, bins=np.linspace(-1.5, 1.5, 30), density=True, color="tab:blue",
                    alpha=0.45, label=f"empirical (n={len(emp)})")
        else:
            x_lo = float(min(emp.min(), centers.min(), gauss_mean[d] - 4 * gauss_std[d]))
            x_hi = float(max(emp.max(), centers.max(), gauss_mean[d] + 4 * gauss_std[d]))
            # Restrict bin display to the relevant range so we see structure
            in_range = (centers >= x_lo) & (centers <= x_hi)
            bin_w = float(centers[1] - centers[0]) * 0.9 if len(centers) > 1 else 0.5
            # Show per-bin probability on the right y-axis (so it's visible alongside density).
            ax2 = ax.twinx()
            ax2.bar(centers[in_range], probs[in_range], width=bin_w, color="tab:green",
                    alpha=0.55, label="mark-disc P(bin)")
            ax2.set_ylabel("mark-disc P(bin)", color="tab:green")
            ax2.set_ylim(0, max(probs.max() * 1.2, 0.05))

            ax.hist(emp, bins=40, density=True, color="tab:blue", alpha=0.5, label=f"empirical (n={len(emp)})")
            xs = np.linspace(x_lo, x_hi, 400)
            if gauss_std[d] > 1e-6:
                pdf = (1 / (gauss_std[d] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - gauss_mean[d]) / gauss_std[d]) ** 2)
                ax.plot(xs, pdf, color="tab:red", linewidth=2.2,
                        label=f"priv-MLP Normal\nμ={gauss_mean[d]:.2f}, σ={gauss_std[d]:.2g}")
            ax.axvline(gauss_mean[d], color="tab:red", linestyle=":", alpha=0.6)
            ax.set_xlim(x_lo, x_hi)

        ax.set_title(dim_names[d])
        ax.set_xlabel("action value (env scale)")
        ax.set_ylabel("empirical density / Gaussian PDF", color="tab:red")
        ax.grid(alpha=0.25)
        # Combined legend
        if d != 6:
            handles_a, labels_a = ax.get_legend_handles_labels()
            handles_b, labels_b = ax2.get_legend_handles_labels()
            ax.legend(handles_a + handles_b, labels_a + labels_b, fontsize=7, loc="upper right")
        else:
            ax.legend(fontsize=7, loc="upper right")

    axes.flat[7].axis("off")
    axes.flat[7].text(
        0.0, 0.5,
        f"Bucket selected for HIGH action std\n"
        f"  bucket size: {len(empirical)} transitions\n"
        f"  mean per-dim arm std (data): {empirical[:, :6].std(axis=0).mean():.2f}\n\n"
        f"Query state: one real transition's obs from the bucket.\n"
        f"  priv-MLP r1_d1024 (sees state + offset, scale)\n"
        f"  mark-disc r4_d512 (sees state only)\n\n"
        f"Green bars = mark-disc TRUE marginal P(a_k | state)\n"
        f"  (softmax over 100 logits per arm dim, sigmoid for\n"
        f"   gripper). MC-averaged over 2048 stochastic AR\n"
        f"   rollouts so the prior dims a_<k are sampled from\n"
        f"   the model's joint. Right-axis scale.\n"
        f"Red curve = priv-MLP Normal(μ, σ). Left-axis scale.\n"
        f"Blue hist = empirical action targets in the bucket.",
        fontsize=9, va="center", ha="left",
    )
    fig.suptitle(
        "Trained priv-MLP Gaussian vs trained mark-disc per-bin probability vs empirical actions\n"
        "at one real (state × perturbation) bucket from the priv-MLP dataset",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=140)
    plt.close(fig)
    print(f"[plot] saved → {PLOT_OUT}")


if __name__ == "__main__":
    main()
