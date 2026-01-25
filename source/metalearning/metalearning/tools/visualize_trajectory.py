"""Visualize end effector and action trajectories from episode files.

Example usage:
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --episode-idx 3
    python -m metalearning.tools.visualize_trajectory /path/to/episodes_000000.pt --out-dir /tmp/plots
    python -m metalearning.tools.visualize_trajectory /path/to/rollout_pairs_000000.pt --episode-idx 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch

from .visualization_utils import (
    get_pose_obs,
    _load_episodes,
    _load_pairs,
    _plot_series,
    _plot_traj3d,
    _plot_traj3d_pair,
    _select_episode,
    _select_pair,
    _trim_to_length,
)


def visualize_demo_rollout_3d(
    demo_episode: Mapping[str, Any],
    rollout_episode: Mapping[str, Any],
    obs_key: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """Visualize demo and rollout trajectories in a shared 3D plot."""
    demo_length = int(demo_episode["length"]) if "length" in demo_episode else None
    rollout_length = int(rollout_episode["length"]) if "length" in rollout_episode else None
    demo_obs, demo_key = _get_pose_obs(demo_episode["obs"], obs_key)
    rollout_obs, rollout_key = _get_pose_obs(rollout_episode["obs"], obs_key)
    demo_obs = _trim_to_length(demo_obs, demo_length)
    rollout_obs = _trim_to_length(rollout_obs, rollout_length)
    if demo_obs.shape[-1] < 3 or rollout_obs.shape[-1] < 3:
        raise ValueError("Pose obs last dim must be at least 3.")
    demo_eef = demo_obs[..., :3]
    rollout_eef = rollout_obs[..., :3]
    _plot_traj3d_pair(demo_eef, rollout_eef, "Demo vs Rollout", out_path)
    return demo_key, rollout_key


def main() -> None:
    """Run the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize episode trajectories from .pt files.")
    parser.add_argument("path", type=Path, help="Path to a .pt episode file.")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--obs-key", type=str, default=None, help="Override key for debug obs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: <episode_dir>/visualizations).",
    )
    parser.add_argument("--plot-actions", action="store_true", help="Plot actions.")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.path.parent / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.path, map_location="cpu")
    if isinstance(data, dict) and "pairs" in data:
        pairs = _load_pairs(args.path)
        pair = _select_pair(pairs, args.episode_idx)
        demo_episode = pair["demo"]
        rollout_episode = pair["rollout"]
        eef_out = out_dir / f"pair_{args.episode_idx:04d}_eef.png"
        demo_key, rollout_key = visualize_demo_rollout_3d(
            demo_episode, rollout_episode, obs_key=args.obs_key, out_path=eef_out
        )
        if args.plot_actions and isinstance(rollout_episode.get("actions"), torch.Tensor):
            actions = rollout_episode["actions"]
            rollout_length = int(rollout_episode["length"]) if "length" in rollout_episode else None
            actions = _trim_to_length(actions, rollout_length)
            action_out = out_dir / f"pair_{args.episode_idx:04d}_actions.png"
            _plot_series(actions, f"Rollout Actions ({rollout_key})", "action", action_out)
    else:
        episodes = _load_episodes(args.path)
        episode = _select_episode(episodes, args.episode_idx)
        length = int(episode["length"]) if "length" in episode else None
        obs = episode["obs"]
        debug_obs, debug_key = _get_pose_obs(obs, args.obs_key)
        debug_obs = _trim_to_length(debug_obs, length)
        if debug_obs.shape[-1] < 3:
            raise ValueError(f"debug_obs has last dim {debug_obs.shape[-1]}, expected at least 3.")
        eef_obs = debug_obs[..., :3]
        eef_out = out_dir / f"episode_{args.episode_idx:04d}_eef.png"
        _plot_traj3d(eef_obs, f"End Effector ({debug_key}[:3])", eef_out)
        if args.plot_actions:
            actions = episode.get("actions")
            if isinstance(actions, torch.Tensor):
                actions = _trim_to_length(actions, length)
                action_out = out_dir / f"episode_{args.episode_idx:04d}_actions.png"
                _plot_series(actions, "Actions", "action", action_out)


if __name__ == "__main__":
    main()
