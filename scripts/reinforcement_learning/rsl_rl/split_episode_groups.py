#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split grouped demo episodes into per-task PT files for supervised fine-tuning."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def _nested_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return torch.allclose(lhs, rhs, rtol=1e-3, atol=1e-3)
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        if lhs.keys() != rhs.keys():
            print(f"Dict keys mismatch: {lhs.keys()} != {rhs.keys()}")
            return False
        return all(_nested_equal(lhs[key], rhs[key]) for key in lhs)
    if isinstance(lhs, list) and isinstance(rhs, list):
        if len(lhs) != len(rhs):
            print(f"List length mismatch: {len(lhs)} != {len(rhs)}")
            return False
        return all(_nested_equal(l_item, r_item) for l_item, r_item in zip(lhs, rhs))
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        if len(lhs) != len(rhs):
            print(f"Tuple length mismatch: {len(lhs)} != {len(rhs)}")
            return False
        return all(_nested_equal(l_item, r_item) for l_item, r_item in zip(lhs, rhs))
    return lhs == rhs


def _assert_all_equal(items: list[Any], label: str, group_idx: int) -> None:
    if not items:
        return
    reference = items[0]
    for idx, item in enumerate(items[1:], start=1):
        assert _nested_equal(reference, item), (
            f"Expected all {label} in group {group_idx} to be identical, "
            f"but item 0 and item {idx} differ."
        )


def _plot_suffix(backend: str) -> str:
    return ".html" if backend == "plotly" else ".png"


def _ensure_metalearning_source_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "source" / "metalearning"
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.append(source_root_str)


def _plot_first_group_subset(
    group: list[dict[str, Any]],
    out_path: Path,
    subset_size: int,
    obs_keys: list[str],
    backend: str,
) -> None:
    _ensure_metalearning_source_on_path()
    from metalearning.tools.visualization_utils import get_pose_obs_multi, trim_to_length
    from metalearning.tools.visualize_trajectory import plot_traj3d_multi, plot_traj3d_multi_plotly

    if subset_size <= 0:
        raise ValueError(f"plot_subset_size must be > 0, got {subset_size}")
    if len(group) == 0:
        raise ValueError("Cannot plot subset from empty group.")

    num_to_plot = min(subset_size, len(group))
    eef_trajs: list[torch.Tensor] = []
    labels: list[str] = []
    color_group_ids: list[int] = []
    for traj_idx, episode in enumerate(group[:num_to_plot]):
        if "obs" not in episode:
            raise KeyError(f"Episode {traj_idx} in first group missing 'obs'.")
        length = int(episode["length"]) if "length" in episode else None
        resolved_poses = get_pose_obs_multi(episode["obs"], obs_keys)
        for key_idx, (pose_obs, _resolved_key) in enumerate(resolved_poses):
            pose_obs = trim_to_length(pose_obs, length)
            if pose_obs.ndim != 2 or pose_obs.shape[-1] < 3:
                raise ValueError(
                    f"Episode {traj_idx} pose obs must have shape [T, >=3], got {tuple(pose_obs.shape)}"
                )
            eef_trajs.append(pose_obs[..., :3])
            short_key = obs_keys[key_idx].replace("debug/", "")
            labels.append(f"traj_{traj_idx:02d}:{short_key}")
            color_group_ids.append(traj_idx)

    title = "First Group Pose Trajectories"
    if backend == "plotly":
        plot_traj3d_multi_plotly(eef_trajs, labels, title, out_path, color_group_ids=color_group_ids)
    else:
        plot_traj3d_multi(eef_trajs, labels, title, out_path, color_group_ids=color_group_ids)


def _parse_obs_keys(obs_key_arg: str) -> list[str]:
    keys = [value.strip() for value in obs_key_arg.split(",") if value.strip()]
    if not keys:
        raise ValueError("plot_obs_key must contain at least one key.")
    return keys


def _flatten_obs_dict(obs: dict[str, torch.Tensor], length: int, include_debug_obs: bool) -> tuple[torch.Tensor, list[str]]:
    selected_keys = [key for key in sorted(obs.keys()) if include_debug_obs or not key.startswith("debug/")]
    if not selected_keys:
        raise ValueError("No observation keys selected for flattening.")

    chunks: list[torch.Tensor] = []
    for key in selected_keys:
        value = obs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor obs for key '{key}', got {type(value)}")
        if value.ndim != 2:
            raise ValueError(f"Expected obs['{key}'] shape [T, D], got {tuple(value.shape)}")
        if value.shape[0] < length:
            raise ValueError(
                f"Observation key '{key}' has fewer timesteps than episode length: {value.shape[0]} < {length}"
            )
        chunks.append(value[:length].float())
    return torch.cat(chunks, dim=-1), selected_keys


def _extract_episode_states(episode: dict[str, Any], include_debug_obs: bool) -> tuple[torch.Tensor, list[str] | None]:
    length = int(episode.get("length", 0))
    if length <= 0:
        raise ValueError(f"Episode has non-positive length: {length}")

    obs = episode.get("obs")
    if isinstance(obs, torch.Tensor):
        if obs.ndim != 2:
            raise ValueError(f"Expected episode['obs'] with shape [T, D], got {tuple(obs.shape)}")
        if obs.shape[0] < length:
            raise ValueError(f"episode['obs'] timesteps {obs.shape[0]} < length {length}")
        return obs[:length].float(), None
    if isinstance(obs, dict):
        return _flatten_obs_dict(obs, length=length, include_debug_obs=include_debug_obs)

    raise ValueError("Episode does not contain tensor/dict 'obs' needed to build supervised states.")


def _build_group_dataset(
    group: list[dict[str, Any]],
    include_debug_obs: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str] | None, list[int], torch.Tensor]:
    if not group:
        raise ValueError("Cannot build dataset from empty group.")

    state_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    expected_obs_keys: list[str] | None = None
    episode_lengths: list[int] = []
    sample_episode_indices: list[torch.Tensor] = []

    for episode_idx, episode in enumerate(group):
        actions = episode.get("actions")
        length = int(episode.get("length", 0))
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"Episode {episode_idx} missing tensor 'actions'.")
        if actions.ndim != 2:
            raise ValueError(f"Episode {episode_idx} expected actions shape [T, A], got {tuple(actions.shape)}")
        if actions.shape[0] < length:
            raise ValueError(
                f"Episode {episode_idx} actions timesteps {actions.shape[0]} < declared length {length}"
            )
        if "states" not in episode:
            raise KeyError(f"Episode {episode_idx} is missing reset 'states' metadata.")
        if "physics" not in episode:
            raise KeyError(f"Episode {episode_idx} is missing reset 'physics' metadata.")

        states, obs_keys = _extract_episode_states(episode, include_debug_obs=include_debug_obs)
        if states.shape[0] != length:
            raise ValueError(f"Episode {episode_idx} state timesteps {states.shape[0]} != length {length}")

        if obs_keys is not None:
            if expected_obs_keys is None:
                expected_obs_keys = obs_keys
            elif obs_keys != expected_obs_keys:
                raise ValueError(
                    "Observation keys mismatch across episodes in group. "
                    f"Expected {expected_obs_keys}, got {obs_keys} at episode {episode_idx}."
                )

        state_chunks.append(states)
        action_chunks.append(actions[:length].float())
        episode_lengths.append(length)
        sample_episode_indices.append(torch.full((length,), episode_idx, dtype=torch.long))

    states_tensor = torch.cat(state_chunks, dim=0)
    actions_tensor = torch.cat(action_chunks, dim=0)
    if states_tensor.shape[0] != actions_tensor.shape[0]:
        raise ValueError(
            f"States/actions sample mismatch after flattening group: {states_tensor.shape[0]} vs {actions_tensor.shape[0]}"
        )
    sample_episode_index = torch.cat(sample_episode_indices, dim=0)
    return states_tensor, actions_tensor, expected_obs_keys, episode_lengths, sample_episode_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Split grouped episode file into one PT file per episode group.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to grouped episode file (.pt) containing key 'episode_groups'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <input_dir>/individual_tasks.",
    )
    parser.add_argument(
        "--include_debug_obs",
        action="store_true",
        help="Include obs keys starting with 'debug/' when flattening dict observations.",
    )
    parser.add_argument(
        "--expect_group_size",
        type=int,
        default=None,
        help="Optional strict check for a fixed group size (e.g. 64).",
    )
    parser.add_argument(
        "--postfilter_group_length",
        type=int,
        default=None,
        help=(
            "Optional target group length for saved outputs. "
            "Groups shorter than this are skipped, groups longer are randomly subsampled to this size."
        ),
    )
    parser.add_argument(
        "--postfilter_seed",
        type=int,
        default=0,
        help="Random seed used when subsampling groups larger than --postfilter_group_length.",
    )
    parser.add_argument(
        "--plot_subset_size",
        type=int,
        default=4,
        help="How many trajectories from the first selected group to plot",
    )
    parser.add_argument(
        "--plot_obs_key",
        type=str,
        default="end_effector_pose,receptive_asset_pose,insertive_asset_pose",
        help="Comma-separated observation keys to use for trajectory plotting.",
    )
    parser.add_argument(
        "--plot_backend",
        type=str,
        choices=("plotly", "matplotlib"),
        default="matplotlib",
        help="Plotting backend for first-group subset visualization.",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Optional output path for the first-group subset plot.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    payload = torch.load(input_path, map_location="cpu")
    if not (isinstance(payload, dict) and "episode_groups" in payload):
        raise ValueError(f"{input_path} must contain a dict with key 'episode_groups'.")

    episode_groups = payload["episode_groups"]
    if not isinstance(episode_groups, list):
        raise TypeError("Expected 'episode_groups' to be a list.")

    source_dir = os.path.dirname(input_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir is not None else os.path.join(source_dir, "individual_tasks", str(args.postfilter_group_length) + "samples")
    os.makedirs(output_dir, exist_ok=True)

    group_lengths = [len(group) for group in episode_groups]
    if group_lengths:
        length_counts = Counter(group_lengths)
        print(f"[INFO] Found {len(episode_groups)} groups in {input_path}")
        print(f"[INFO] Group length distribution: {dict(sorted(length_counts.items()))}")
    else:
        print(f"[INFO] No groups found in {input_path}")

    configured_group_size = payload.get("num_similar_trajectories")
    if configured_group_size is not None:
        print(f"[INFO] num_similar_trajectories in payload: {configured_group_size}")
    if args.expect_group_size is not None and any(length != args.expect_group_size for length in group_lengths):
        raise ValueError(
            f"Not all groups have size {args.expect_group_size}. "
            f"Observed sizes: {sorted(set(group_lengths))}"
        )
    if args.postfilter_group_length is not None and args.postfilter_group_length <= 0:
        raise ValueError(
            f"postfilter_group_length must be > 0, got {args.postfilter_group_length}"
        )
    rng = random.Random(args.postfilter_seed)
    plot_obs_keys = _parse_obs_keys(args.plot_obs_key)

    written = 0
    skipped_short = 0
    cropped_long = 0
    first_group_plotted = False
    for group_idx, group in enumerate(episode_groups):
        if not isinstance(group, list):
            raise TypeError(f"Group {group_idx} is not a list.")
        if len(group) == 0:
            continue

        selected_group = group
        selected_indices = list(range(len(group)))
        if args.postfilter_group_length is not None:
            target_len = int(args.postfilter_group_length)
            if len(group) < target_len:
                skipped_short += 1
                continue
            if len(group) > target_len:
                sampled_indices = sorted(rng.sample(range(len(group)), target_len))
                selected_indices = sampled_indices
                selected_group = [group[i] for i in sampled_indices]
                cropped_long += 1

        states, actions, obs_keys, episode_lengths, sample_episode_index = _build_group_dataset(
            selected_group, include_debug_obs=args.include_debug_obs
        )
        reset_states = [episode["states"] for episode in selected_group]
        reset_physics = [episode["physics"] for episode in selected_group]
        if not first_group_plotted:
            print(states.shape)
        # print(f"reset_states: {reset_states[0]}")
        # _assert_all_equal(reset_states, label="reset states", group_idx=group_idx)
        # _assert_all_equal(reset_physics, label="reset physics", group_idx=group_idx)

        out_payload: dict[str, Any] = {
            "states": states,
            "actions": actions,
            "reset_states": reset_states,
            "reset_physics": reset_physics,
            "episode_lengths": episode_lengths,
            "sample_episode_index": sample_episode_index,
            "episode_group": selected_group,
            "group_index": group_idx,
            "num_group_episodes": len(selected_group),
            "num_group_samples": int(states.shape[0]),
            "source_path": input_path,
            "num_similar_trajectories": configured_group_size,
            "flattened_obs_keys": obs_keys,
            "include_debug_obs": bool(args.include_debug_obs),
            "postfilter_group_length": args.postfilter_group_length,
            "original_group_length": len(group),
            "selected_group_indices": selected_indices,
        }
        out_path = os.path.join(output_dir, f"task_group_{group_idx:06d}.pt")
        torch.save(out_payload, out_path)
        written += 1

        if not first_group_plotted:
            if args.plot_output is None:
                plot_path = Path(output_dir) / f"first_group_subset{_plot_suffix(args.plot_backend)}"
            else:
                plot_path = Path(args.plot_output).expanduser().resolve()
                plot_path.parent.mkdir(parents=True, exist_ok=True)
            _plot_first_group_subset(
                group=selected_group,
                out_path=plot_path,
                subset_size=args.plot_subset_size,
                obs_keys=plot_obs_keys,
                backend=args.plot_backend,
            )
            print(f"[INFO] Saved first-group subset trajectory plot: {plot_path}")
            first_group_plotted = True

    print(f"[INFO] Wrote {written} group files to: {output_dir}")
    if args.postfilter_group_length is not None:
        print(
            "[INFO] Postfilter summary: "
            f"target_len={args.postfilter_group_length}, "
            f"skipped_short={skipped_short}, cropped_long={cropped_long}"
        )


if __name__ == "__main__":
    main()
