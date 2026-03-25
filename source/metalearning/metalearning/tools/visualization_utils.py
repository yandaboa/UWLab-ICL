"""Shared utilities for episode visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

_POSES_IN_END_EFFECTOR_FRAME = {
    "insertive_asset_pose",
    "receptive_asset_pose",
    "debug/insertive_asset_pose",
    "debug/receptive_asset_pose",
}


def _skew_matrix(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            torch.stack((zeros, -v[..., 2], v[..., 1]), dim=-1),
            torch.stack((v[..., 2], zeros, -v[..., 0]), dim=-1),
            torch.stack((-v[..., 1], v[..., 0], zeros), dim=-1),
        ),
        dim=-2,
    )


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle with last dim 3, got {tuple(axis_angle.shape)}")
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    theta_safe = torch.clamp(theta, min=1.0e-8)
    axis = axis_angle / theta_safe
    k = _skew_matrix(axis)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(*axis_angle.shape[:-1], 3, 3)
    sin_term = torch.sin(theta)[..., None] * k
    cos_term = (1.0 - torch.cos(theta))[..., None] * (k @ k)
    return eye + sin_term + cos_term


def _matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (...,3,3), got {tuple(rotation.shape)}")
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    axis = vee / (2.0 * sin_theta[..., None].clamp(min=1.0e-8))
    axis_angle = axis * theta[..., None]
    small = theta < 1.0e-6
    if torch.any(small):
        # First-order approximation around zero rotation.
        axis_angle_small = 0.5 * vee
        axis_angle = torch.where(small[..., None], axis_angle_small, axis_angle)
    return axis_angle


def _compose_axis_angle_pose(pose_ab: torch.Tensor, pose_bc: torch.Tensor) -> torch.Tensor:
    if pose_ab.ndim != 2 or pose_bc.ndim != 2:
        raise ValueError("Expected 2D pose tensors [T, D].")
    if pose_ab.shape[0] != pose_bc.shape[0]:
        raise ValueError(f"Pose length mismatch: {pose_ab.shape[0]} vs {pose_bc.shape[0]}")
    if pose_ab.shape[1] < 6 or pose_bc.shape[1] < 6:
        raise ValueError(f"Expected pose dims >= 6, got {pose_ab.shape[1]} and {pose_bc.shape[1]}")
    t_ab = pose_ab[:, :3]
    aa_ab = pose_ab[:, 3:6]
    t_bc = pose_bc[:, :3]
    aa_bc = pose_bc[:, 3:6]

    r_ab = _axis_angle_to_matrix(aa_ab)
    r_bc = _axis_angle_to_matrix(aa_bc)
    r_ac = r_ab @ r_bc
    t_ac = t_ab + torch.matmul(r_ab, t_bc.unsqueeze(-1)).squeeze(-1)
    aa_ac = _matrix_to_axis_angle(r_ac)

    out = pose_bc.clone()
    out[:, :3] = t_ac
    out[:, 3:6] = aa_ac
    return out


def _invert_axis_angle_pose(pose_ab: torch.Tensor) -> torch.Tensor:
    if pose_ab.ndim != 2:
        raise ValueError("Expected 2D pose tensor [T, D].")
    if pose_ab.shape[1] < 6:
        raise ValueError(f"Expected pose dim >= 6, got {pose_ab.shape[1]}")
    t_ab = pose_ab[:, :3]
    aa_ab = pose_ab[:, 3:6]
    r_ab = _axis_angle_to_matrix(aa_ab)
    r_ba = torch.transpose(r_ab, -1, -2)
    t_ba = -torch.matmul(r_ba, t_ab.unsqueeze(-1)).squeeze(-1)
    aa_ba = _matrix_to_axis_angle(r_ba)
    out = pose_ab.clone()
    out[:, :3] = t_ba
    out[:, 3:6] = aa_ba
    return out


def _resolve_pose_in_robot_base(
    obs: torch.Tensor | Mapping[str, Any],
    pose: torch.Tensor,
    resolved_key: str,
) -> Tuple[torch.Tensor, str]:
    if not isinstance(obs, Mapping):
        return pose, resolved_key
    if resolved_key not in _POSES_IN_END_EFFECTOR_FRAME:
        return pose, resolved_key

    # Receptive pose is observed w.r.t. robotiq_base_link (without gripper metadata offset),
    # while end_effector_pose uses metadata offsets. To avoid mixing these roots, reconstruct
    # receptive pose in the same frame as insertive pose:
    #   T_ee_receptive = T_ee_insertive * inv(T_receptive_insertive)
    # then map to base with end_effector_pose.
    if resolved_key in {"receptive_asset_pose", "debug/receptive_asset_pose"}:
        try:
            insertive_in_ee, insertive_ee_key = get_named_pose_obs(obs, "insertive_asset_pose")
            insertive_in_receptive, ins_rec_key = get_named_pose_obs(obs, "insertive_asset_in_receptive_asset_frame")
            receptive_in_insertive = _invert_axis_angle_pose(insertive_in_receptive)
            receptive_in_ee = _compose_axis_angle_pose(insertive_in_ee, receptive_in_insertive)
            ee_pose, ee_key = get_named_pose_obs(obs, "end_effector_pose")
            transformed = _compose_axis_angle_pose(ee_pose, receptive_in_ee)
            return (
                transformed,
                f"{resolved_key}_in_robot_base(via_{ee_key}+{insertive_ee_key}+{ins_rec_key})",
            )
        except (KeyError, ValueError, TypeError):
            # Fallback when triangulation terms are unavailable.
            return pose, f"{resolved_key}_untransformed(frame_mismatch)"

    ee_pose, ee_key = get_named_pose_obs(obs, "end_effector_pose")
    transformed = _compose_axis_angle_pose(ee_pose, pose)
    return transformed, f"{resolved_key}_in_robot_base(via_{ee_key})"


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """Load episodes saved by EpisodeStorage."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "pairs" in data:
        raise ValueError("Found paired rollouts; use _load_pairs instead.")
    if isinstance(data, dict) and "episodes" in data:
        episodes = data["episodes"]
    elif isinstance(data, dict) and "episode_groups" in data:
        episode_groups = data["episode_groups"]
        episodes = [episode for group in episode_groups for episode in group]
    elif isinstance(data, list):
        episodes = data
    elif isinstance(data, dict) and "obs" in data:
        episodes = [data]
    else:
        raise ValueError(f"Unsupported file format in {path}.")
    if not episodes:
        raise ValueError(f"No episodes found in {path}.")
    return episodes


def select_episode(episodes: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Select a single episode by index."""
    if index < 0 or index >= len(episodes):
        raise IndexError(f"Episode index {index} out of range (0..{len(episodes) - 1}).")
    return episodes[index]


def load_pairs(path: Path) -> list[dict[str, Any]]:
    """Load paired rollouts saved by RolloutPairStorage."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
    elif isinstance(data, list):
        pairs = data
    else:
        raise ValueError(f"Unsupported paired file format in {path}.")
    if not pairs:
        raise ValueError(f"No paired rollouts found in {path}.")
    return pairs


def select_pair(pairs: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """Select a single rollout pair by index."""
    if index < 0 or index >= len(pairs):
        raise IndexError(f"Pair index {index} out of range (0..{len(pairs) - 1}).")
    return pairs[index]


def get_pose_obs(
    obs: torch.Tensor | Mapping[str, Any], obs_key: Optional[str]
) -> Tuple[torch.Tensor, str]:
    """Extract pose observations from an episode."""
    if isinstance(obs, Mapping):
        if obs_key is not None:
            if obs_key not in obs:
                raise KeyError(f"obs key '{obs_key}' not found.")
            value = obs[obs_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs key '{obs_key}' is not a tensor.")
            return value, obs_key
        if "debug/end_effector_pose" in obs:
            value = obs["debug/end_effector_pose"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'debug/end_effector_pose' is not a tensor.")
            return value, "debug/end_effector_pose"
        if "end_effector_pose" in obs:
            value = obs["end_effector_pose"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'end_effector_pose' is not a tensor.")
            return value, "end_effector_pose"
        if "debug" in obs:
            value = obs["debug"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("obs key 'debug' is not a tensor.")
            return value, "debug"
        debug_keys = [key for key in obs.keys() if key.startswith("debug/")]
        if debug_keys:
            key = debug_keys[0]
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs key '{key}' is not a tensor.")
            return value, key
        raise ValueError("No pose obs tensor found.")
    if not isinstance(obs, torch.Tensor):
        raise TypeError("obs is not a tensor.")
    return obs, "obs"


def get_named_pose_obs(
    obs: torch.Tensor | Mapping[str, Any], pose_key: str
) -> Tuple[torch.Tensor, str]:
    """Resolve a named pose key, trying both base and debug/ variants."""
    if isinstance(obs, Mapping):
        if pose_key.startswith("debug/"):
            base_key = pose_key[len("debug/") :]
            candidates = [pose_key, base_key]
        else:
            candidates = [pose_key, f"debug/{pose_key}"]
        for candidate in candidates:
            if candidate not in obs:
                continue
            value = obs[candidate]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs key '{candidate}' is not a tensor.")
            return value, candidate
        raise KeyError(f"None of the pose keys were found: {candidates}")
    if not isinstance(obs, torch.Tensor):
        raise TypeError("obs is not a tensor.")
    if pose_key in {"obs", "end_effector_pose", "debug/end_effector_pose"}:
        return obs, "obs"
    raise KeyError(f"Cannot resolve pose key '{pose_key}' when obs is a raw tensor.")


def get_pose_obs_multi(
    obs: torch.Tensor | Mapping[str, Any], obs_keys: Sequence[str]
) -> list[Tuple[torch.Tensor, str]]:
    """Resolve one or more pose keys to tensors and their concrete resolved keys."""
    if len(obs_keys) == 0:
        return [get_pose_obs(obs, None)]
    resolved: list[Tuple[torch.Tensor, str]] = []
    for key in obs_keys:
        pose, resolved_key = get_named_pose_obs(obs, key)
        pose_in_base, resolved_key_in_base = _resolve_pose_in_robot_base(obs, pose, resolved_key)
        resolved.append((pose_in_base, resolved_key_in_base))
    return resolved


def trim_to_length(tensor: torch.Tensor, length: Optional[int]) -> torch.Tensor:
    """Trim a tensor to an episode length if available."""
    if length is None:
        return tensor
    if tensor.shape[0] <= length:
        return tensor
    return tensor[:length]


def _blend_color(color: Any, target: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    tgt = np.array(target)
    blended = rgb * (1.0 - amount) + tgt * amount
    return tuple(blended.tolist())


def plot_series(data: torch.Tensor, title: str, y_label: str, out_path: Optional[Path]) -> None:
    """Plot time series data for each dimension."""
    data_np = data.detach().cpu().numpy()
    if data_np.ndim == 1:
        data_np = data_np[:, None]
    num_dims = data_np.shape[1]
    fig, axes = plt.subplots(num_dims, 1, sharex=True, figsize=(8, 2.4 * num_dims))
    if num_dims == 1:
        axes = [axes]
    for dim in range(num_dims):
        axes[dim].plot(data_np[:, dim])
        axes[dim].set_ylabel(f"{y_label}[{dim}]")
    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_traj3d(data: torch.Tensor, title: str, out_path: Optional[Path]) -> None:
    """Plot a 3D trajectory from (T, 3+) data."""
    data_np = data.detach().cpu().numpy()
    if data_np.ndim != 2 or data_np.shape[1] < 3:
        raise ValueError(f"Expected (T, >=3) data, got {data_np.shape}.")
    data_np = data_np[:, :3]
    mask = np.isfinite(data_np).all(axis=1)
    data_np = data_np[mask]
    if data_np.shape[0] < 2:
        raise ValueError("Not enough finite points to plot.")
    x, y, z = data_np.T
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    line = ax.plot(x, y, z, linewidth=2.0)[0]
    line_color = line.get_color()
    start_color = _blend_color(line_color, (1.0, 1.0, 1.0), 0.35)
    end_color = _blend_color(line_color, (0.0, 0.0, 0.0), 0.25)
    ax.scatter(x[0], y[0], z[0], s=60, marker="o", color=start_color)
    ax.scatter(x[-1], y[-1], z[-1], s=60, marker="^", color=end_color)
    mins = data_np.min(axis=0)
    maxs = data_np.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect((maxs - mins))
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_traj3d_pair(
    demo: torch.Tensor,
    rollout: torch.Tensor,
    title: str,
    out_path: Optional[Path],
    colors: tuple[str, str] = ("tab:blue", "tab:orange"),
) -> None:
    """Plot paired 3D trajectories from (T, 3+) data."""
    demo_np = demo.detach().cpu().numpy()
    rollout_np = rollout.detach().cpu().numpy()
    if demo_np.ndim != 2 or demo_np.shape[1] < 3:
        raise ValueError(f"Expected demo (T, >=3) data, got {demo_np.shape}.")
    if rollout_np.ndim != 2 or rollout_np.shape[1] < 3:
        raise ValueError(f"Expected rollout (T, >=3) data, got {rollout_np.shape}.")
    demo_np = demo_np[:, :3]
    rollout_np = rollout_np[:, :3]
    demo_np = demo_np[np.isfinite(demo_np).all(axis=1)]
    rollout_np = rollout_np[np.isfinite(rollout_np).all(axis=1)]
    if demo_np.shape[0] < 2 or rollout_np.shape[0] < 2:
        raise ValueError("Not enough finite points to plot.")
    demo_x, demo_y, demo_z = demo_np.T
    roll_x, roll_y, roll_z = rollout_np.T
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(demo_x, demo_y, demo_z, linewidth=2.0, color=colors[0], label="demo")
    ax.plot(roll_x, roll_y, roll_z, linewidth=2.0, color=colors[1], label="rollout")
    demo_start = _blend_color(colors[0], (1.0, 1.0, 1.0), 0.35)
    demo_end = _blend_color(colors[0], (0.0, 0.0, 0.0), 0.25)
    roll_start = _blend_color(colors[1], (1.0, 1.0, 1.0), 0.35)
    roll_end = _blend_color(colors[1], (0.0, 0.0, 0.0), 0.25)
    ax.scatter(demo_x[0], demo_y[0], demo_z[0], s=50, marker="o", color=demo_start)
    ax.scatter(demo_x[-1], demo_y[-1], demo_z[-1], s=50, marker="^", color=demo_end)
    ax.scatter(roll_x[0], roll_y[0], roll_z[0], s=50, marker="o", color=roll_start)
    ax.scatter(roll_x[-1], roll_y[-1], roll_z[-1], s=50, marker="^", color=roll_end)
    mins = np.minimum(demo_np.min(axis=0), rollout_np.min(axis=0))
    maxs = np.maximum(demo_np.max(axis=0), rollout_np.max(axis=0))
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect((maxs - mins))
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def create_traj3d_pair_figure(
    demo: torch.Tensor,
    rollout: torch.Tensor,
    title: str,
    colors: tuple[str, str] = ("tab:blue", "tab:orange"),
):
    """Create a matplotlib figure for paired 3D trajectories."""
    demo_np = demo.detach().cpu().numpy()
    rollout_np = rollout.detach().cpu().numpy()
    if demo_np.ndim != 2 or demo_np.shape[1] < 3:
        raise ValueError(f"Expected demo (T, >=3) data, got {demo_np.shape}.")
    if rollout_np.ndim != 2 or rollout_np.shape[1] < 3:
        raise ValueError(f"Expected rollout (T, >=3) data, got {rollout_np.shape}.")
    demo_np = demo_np[:, :3]
    rollout_np = rollout_np[:, :3]
    demo_np = demo_np[np.isfinite(demo_np).all(axis=1)]
    rollout_np = rollout_np[np.isfinite(rollout_np).all(axis=1)]
    if demo_np.shape[0] < 2 or rollout_np.shape[0] < 2:
        raise ValueError("Not enough finite points to plot.")
    demo_x, demo_y, demo_z = demo_np.T
    roll_x, roll_y, roll_z = rollout_np.T
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(demo_x, demo_y, demo_z, linewidth=2.0, color=colors[0], label="demo")
    ax.plot(roll_x, roll_y, roll_z, linewidth=2.0, color=colors[1], label="rollout")
    demo_start = _blend_color(colors[0], (1.0, 1.0, 1.0), 0.35)
    demo_end = _blend_color(colors[0], (0.0, 0.0, 0.0), 0.25)
    roll_start = _blend_color(colors[1], (1.0, 1.0, 1.0), 0.35)
    roll_end = _blend_color(colors[1], (0.0, 0.0, 0.0), 0.25)
    ax.scatter(demo_x[0], demo_y[0], demo_z[0], s=50, marker="o", color=demo_start)
    ax.scatter(demo_x[-1], demo_y[-1], demo_z[-1], s=50, marker="^", color=demo_end)
    ax.scatter(roll_x[0], roll_y[0], roll_z[0], s=50, marker="o", color=roll_start)
    ax.scatter(roll_x[-1], roll_y[-1], roll_z[-1], s=50, marker="^", color=roll_end)
    mins = np.minimum(demo_np.min(axis=0), rollout_np.min(axis=0))
    maxs = np.maximum(demo_np.max(axis=0), rollout_np.max(axis=0))
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect((maxs - mins))
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
