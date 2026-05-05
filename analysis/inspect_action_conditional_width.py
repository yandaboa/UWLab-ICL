"""Measure how much conditioning on (state, perturbation) narrows the action distribution.

We bucket transitions by EE-pose voxels and (optionally) by perturbation-parameter bins,
then compute the within-bucket action std. The comparison answers:

- Unconditional std: raw spread of action targets.
- State-only conditional std: spread among transitions in the same EE-pose voxel.
- State + perturbation conditional std: spread among transitions in the same voxel
  AND with similar (scale, offset).

If state+perturbation std is much smaller than state-only std, the privileged
perturbation params should make the BC target nearly deterministic — and a Gaussian
head ought to learn it. If they're comparable, the BC target is multimodal even
given (state, perturbation), and a Gaussian head averaging over modes will fail.

Usage:
    python analysis/inspect_action_conditional_width.py <path/to/data.zarr>
"""
from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import zarr  # noqa: E402


def load_arrays(zarr_path: pathlib.Path):
    root = zarr.open(str(zarr_path), mode="r")
    actions = np.asarray(root["data/actions"])
    ee_pose = np.asarray(root["data/obs/end_effector_pose"])
    insertive = np.asarray(root["data/obs/insertive_asset_pose"])
    receptive = np.asarray(root["data/obs/receptive_asset_pose"])
    has_offset = "data/obs/action_offset" in root
    if has_offset:
        action_offset = np.asarray(root["data/obs/action_offset"])
        action_scale = np.asarray(root["data/obs/action_scale"])
    else:
        action_offset = action_scale = None
    return actions, ee_pose, insertive, receptive, action_offset, action_scale


def bucketed_std(values: np.ndarray, bucket_keys: np.ndarray, min_bucket_size: int = 5):
    """For each bucket, compute the per-dim std of values inside; return the population-
    weighted RMS over buckets and the count of buckets that meet the size threshold."""
    buckets: defaultdict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(bucket_keys):
        buckets[tuple(k)].append(i)

    per_dim_sumsq = np.zeros(values.shape[1])
    total_n = 0
    n_used = 0
    for k, idx in buckets.items():
        if len(idx) < min_bucket_size:
            continue
        sub = values[idx]
        s = sub.std(axis=0, ddof=1)  # per-dim std within bucket
        per_dim_sumsq += (s ** 2) * len(idx)
        total_n += len(idx)
        n_used += 1

    if total_n == 0:
        return None, 0, 0
    pooled = np.sqrt(per_dim_sumsq / total_n)
    return pooled, n_used, total_n


def quantize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform quantile binning, returns int bin indices in [0, n_bins) per column."""
    out = np.zeros_like(x, dtype=np.int64)
    for j in range(x.shape[1]):
        edges = np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            out[:, j] = 0
        else:
            out[:, j] = np.clip(np.searchsorted(edges, x[:, j]) - 1, 0, len(edges) - 2)
    return out


def main(zarr_path: str):
    p = pathlib.Path(zarr_path)
    print(f"[inspect] loading {p}")
    actions, ee_pose, insertive, receptive, action_offset, action_scale = load_arrays(p)
    N, A = actions.shape
    print(f"[inspect] N={N} transitions, action_dim={A}")
    print(f"[inspect] privileged perturbation obs present: {action_offset is not None}")

    # 1. Unconditional std
    uncond = actions.std(axis=0, ddof=1)
    print()
    print(f"unconditional std        per-dim: {np.round(uncond, 3).tolist()}")
    print(f"  RMS over dims: {float(np.sqrt((uncond ** 2).mean())):.3f}")

    # State-bucket key from EE pose (3D position + axis-angle, axis-angle ignored for binning)
    state_feat = np.concatenate(
        [ee_pose[:, :3], insertive[:, :3], receptive[:, :3]], axis=1
    )  # 9D position-only state
    state_bins = quantize(state_feat, n_bins=8)  # 8^9 buckets max but realistically far fewer

    # 2. State-only conditional std
    pooled_state, n_buckets_state, n_state = bucketed_std(actions, state_bins)
    print()
    if pooled_state is None:
        print("(no state-only buckets met min size)")
    else:
        print(
            f"state-only cond std      per-dim: {np.round(pooled_state, 3).tolist()}"
        )
        print(
            f"  RMS over dims: {float(np.sqrt((pooled_state ** 2).mean())):.3f}"
            f"   ({n_buckets_state} buckets, {n_state}/{N} transitions used)"
        )

    # 3. State + perturbation conditional std
    if action_offset is None:
        print()
        print("(privileged perturbation obs absent — skipping state+pert bucketing)")
        return

    pert_feat = np.concatenate([action_offset, action_scale], axis=1)  # 12D
    pert_bins = quantize(pert_feat, n_bins=4)
    composite = np.concatenate([state_bins, pert_bins], axis=1)
    pooled_full, n_buckets_full, n_full = bucketed_std(actions, composite)
    print()
    if pooled_full is None:
        print("(no state+pert buckets met min size — try lowering min_bucket_size)")
    else:
        print(
            f"state+pert cond std      per-dim: {np.round(pooled_full, 3).tolist()}"
        )
        print(
            f"  RMS over dims: {float(np.sqrt((pooled_full ** 2).mean())):.3f}"
            f"   ({n_buckets_full} buckets, {n_full}/{N} transitions used)"
        )

    if pooled_state is not None and pooled_full is not None:
        rms_uncond = float(np.sqrt((uncond ** 2).mean()))
        rms_state = float(np.sqrt((pooled_state ** 2).mean()))
        rms_full = float(np.sqrt((pooled_full ** 2).mean()))
        print()
        print(
            f"narrowing summary:  uncond={rms_uncond:.3f}  →  state-only={rms_state:.3f}"
            f"  ({(1 - rms_state / rms_uncond) * 100:.1f}% reduction)"
            f"  →  state+pert={rms_full:.3f}"
            f"  ({(1 - rms_full / rms_state) * 100:.1f}% extra reduction from privileged info)"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
