#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate normalized forward-dynamics residuals on held-out episodes."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOURCE_DIR = os.path.join(ROOT_DIR, "source")
PKG_DIRS = [
    SOURCE_DIR,
    os.path.join(SOURCE_DIR, "uwlab_rl"),
    os.path.join(SOURCE_DIR, "uwlab_tasks"),
]
for path in (ROOT_DIR, *PKG_DIRS):
    if path not in sys.path:
        sys.path.append(path)

from uwlab_rl.rsl_rl.forward_dynamics_utils import (  # noqa: E402
    ForwardDynamicsResidualMLP,
    expand_episode_paths,
    flatten_obs,
)
from uwlab_rl.rsl_rl.state_solver import NormalizationStats  # noqa: E402


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _parse_obs_keys(value: str) -> list[str] | None:
    stripped = value.strip()
    if stripped == "" or stripped.lower() == "none":
        return None
    return [key.strip() for key in stripped.split(",") if key.strip()]


def _load_model_and_norms(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ForwardDynamicsResidualMLP, NormalizationStats]:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    meta = payload.get("meta", {})
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    trainer_cfg = payload.get("trainer_cfg", {})
    model_cfg: dict[str, Any] = trainer_cfg.get("model", {}) if isinstance(trainer_cfg, dict) else {}
    model = ForwardDynamicsResidualMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        activation=str(model_cfg.get("activation", "relu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    norm_payload = payload.get("normalization", {})
    if not isinstance(norm_payload, dict) or not norm_payload:
        norm_payload = model.get_input_normalization_payload()
    norms = NormalizationStats(
        state_mean=torch.as_tensor(norm_payload["state_mean"], dtype=torch.float32, device=device),
        state_std=torch.as_tensor(norm_payload["state_std"], dtype=torch.float32, device=device),
        action_mean=torch.as_tensor(norm_payload["action_mean"], dtype=torch.float32, device=device),
        action_std=torch.as_tensor(norm_payload["action_std"], dtype=torch.float32, device=device),
    )
    return model, norms


def _residual_metrics(
    model: ForwardDynamicsResidualMLP,
    norms: NormalizationStats,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        delta_raw = model(states, actions)
        residual_n = (norms.encode_state(states) + delta_raw / norms.state_std) - norms.encode_state(next_states)
        residual_l2 = torch.linalg.norm(residual_n, dim=-1)
        residual_mse = (residual_n * residual_n).mean(dim=-1)
    return residual_l2, residual_mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalized residuals for a forward dynamics model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to forward dynamics checkpoint (.pt).")
    parser.add_argument(
        "--episode_paths",
        type=str,
        nargs="+",
        required=True,
        help="Episode .pt paths and/or glob patterns.",
    )
    parser.add_argument(
        "--obs_keys",
        type=str,
        default="joint_pos,end_effector_pose,insertive_asset_pose,receptive_asset_pose,insertive_asset_in_receptive_asset_frame",
        help="Comma-separated observation keys for dict obs (default: model/data defaults when omitted).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation (cuda or cpu).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, norms = _load_model_and_norms(args.checkpoint, device)
    episode_paths = expand_episode_paths(args.episode_paths)
    obs_keys = _parse_obs_keys(args.obs_keys)

    all_l2: list[torch.Tensor] = []
    all_mse: list[torch.Tensor] = []
    total_transitions = 0

    for episode_path in episode_paths:
        payload = torch.load(episode_path, map_location="cpu")
        episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
        file_l2: list[torch.Tensor] = []
        file_mse: list[torch.Tensor] = []
        file_transitions = 0
        for episode in episodes:
            actions = episode["actions"]
            length = int(episode.get("length", actions.shape[0]))
            if length < 2:
                continue
            obs_seq = flatten_obs(episode["obs"], obs_keys)[:length]
            act_seq = actions[:length].reshape(length, -1)
            states = obs_seq[:-1].to(device=device, dtype=torch.float32)
            actions_t = act_seq[:-1].to(device=device, dtype=torch.float32)
            next_states = obs_seq[1:].to(device=device, dtype=torch.float32)
            l2, mse = _residual_metrics(model, norms, states, actions_t, next_states)
            file_l2.append(l2.detach().cpu())
            file_mse.append(mse.detach().cpu())
            file_transitions += int(l2.numel())

        if file_transitions == 0:
            print(f"[residuals] {episode_path}: no valid transitions")
            continue

        file_l2_all = torch.cat(file_l2, dim=0)
        file_mse_all = torch.cat(file_mse, dim=0)
        all_l2.append(file_l2_all)
        all_mse.append(file_mse_all)
        total_transitions += file_transitions
        print(
            f"[residuals] {os.path.basename(episode_path)} "
            f"N={file_transitions} "
            f"l2_mean={file_l2_all.mean().item():.6f} "
            f"l2_median={file_l2_all.median().item():.6f} "
            f"l2_p90={torch.quantile(file_l2_all, 0.9).item():.6f} "
            f"mse_mean={file_mse_all.mean().item():.6f}"
        )

    if total_transitions == 0:
        raise RuntimeError("No valid transitions found across provided episode paths.")

    l2_all = torch.cat(all_l2, dim=0)
    mse_all = torch.cat(all_mse, dim=0)
    print(
        "[residuals][overall] "
        f"N={total_transitions} "
        f"l2_mean={l2_all.mean().item():.6f} "
        f"l2_median={l2_all.median().item():.6f} "
        f"l2_p90={torch.quantile(l2_all, 0.9).item():.6f} "
        f"mse_mean={mse_all.mean().item():.6f} "
        f"mse_median={mse_all.median().item():.6f}"
    )


if __name__ == "__main__":
    main()
