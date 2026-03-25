#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Small-batch CCIL solver checklist."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Callable, cast

import torch
from omegaconf import OmegaConf

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

from uwlab_rl.rsl_rl.ccil_generation_config import CcilGenerationCfg
from uwlab_rl.rsl_rl.disturb_actions import sample_disturbed_action
from uwlab_rl.rsl_rl.forward_dynamics_utils import (
    ForwardDynamicsResidualMLP,
    expand_episode_paths,
    flatten_obs,
)
from uwlab_rl.rsl_rl.state_solver import NormalizationStats, solve_predecessor_states


def load_cfg_dict(cfg_path: str | None) -> dict | None:
    if cfg_path is None:
        return None
    if cfg_path.endswith(".pt"):
        payload = torch.load(cfg_path, map_location="cpu")
        if isinstance(payload, dict) and "cfg" in payload:
            payload = payload["cfg"]
        if not isinstance(payload, dict):
            raise ValueError("Config checkpoint must contain a dict or a dict under 'cfg'.")
        return payload
    if cfg_path.endswith((".yaml", ".yml")):
        container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if not isinstance(container, dict):
            raise ValueError("YAML config must contain a dictionary at the top level.")
        return container
    raise ValueError("Unsupported config file type. Use .pt, .yaml, or .yml.")


def apply_cfg_overrides(cfg: CcilGenerationCfg, cfg_dict: dict | None, overrides: list[str]) -> None:
    merged = OmegaConf.create(asdict(cfg))
    if cfg_dict:
        merged = OmegaConf.merge(merged, OmegaConf.create(cfg_dict))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    merged_dict = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(merged_dict, dict):
        raise ValueError("Merged config must resolve to a dictionary.")
    from_dict_fn = cast(Callable[[dict], None], getattr(cfg, "from_dict"))
    from_dict_fn(merged_dict)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _load_model_and_norms(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ForwardDynamicsResidualMLP, NormalizationStats]:
    payload = torch.load(checkpoint_path, map_location=device)
    meta = payload["meta"]
    trainer_cfg = payload.get("trainer_cfg", {})
    model_cfg = trainer_cfg.get("model", {}) if isinstance(trainer_cfg, dict) else {}
    model = ForwardDynamicsResidualMLP(
        state_dim=int(meta["state_dim"]),
        action_dim=int(meta["action_dim"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="CCIL solver sanity test on a small batch.")
    parser.add_argument("--config", type=str, default=None, help="Optional .pt/.yaml/.yml config file.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of transitions to test.")
    args, overrides = parser.parse_known_args()

    cfg = CcilGenerationCfg()
    cfg_dict = load_cfg_dict(args.config)
    apply_cfg_overrides(cfg, cfg_dict, overrides)
    assert cfg.model.forward_dynamics_checkpoint != "", "model.forward_dynamics_checkpoint is required."
    assert len(cfg.data.episode_paths) > 0, "data.episode_paths must be provided."
    torch.manual_seed(int(cfg.runtime.seed))
    torch.cuda.manual_seed_all(int(cfg.runtime.seed))
    device = _resolve_device(cfg.runtime.device)
    model, norms = _load_model_and_norms(cfg.model.forward_dynamics_checkpoint, device)

    all_s: list[torch.Tensor] = []
    all_a: list[torch.Tensor] = []
    all_next: list[torch.Tensor] = []
    for episode_path in expand_episode_paths(cfg.data.episode_paths):
        payload = torch.load(episode_path, map_location="cpu")
        episodes = payload.get("episodes", [])
        for episode in episodes:
            actions = episode["actions"]
            length = int(episode.get("length", actions.shape[0]))
            if length < 2:
                continue
            obs_seq = flatten_obs(episode["obs"], cfg.data.obs_keys)[:length]
            act_seq = actions[:length].reshape(length, -1)
            all_s.append(obs_seq[:-1])
            all_a.append(act_seq[:-1])
            all_next.append(obs_seq[1:])
            total = sum(t.shape[0] for t in all_s)
            if total >= int(args.batch_size):
                break
        if sum(t.shape[0] for t in all_s) >= int(args.batch_size):
            break
    if not all_s:
        raise RuntimeError("No transitions found for solver test.")

    s_star = torch.cat(all_s, dim=0)[: int(args.batch_size)].to(device=device, dtype=torch.float32)
    a_star = torch.cat(all_a, dim=0)[: int(args.batch_size)].to(device=device, dtype=torch.float32)
    s_next_star = torch.cat(all_next, dim=0)[: int(args.batch_size)].to(device=device, dtype=torch.float32)
    a_g = sample_disturbed_action(a_star, sigma_action=float(cfg.generation.sigma_action))
    clamp_spec = (
        float(cfg.generation.action_norm_clip_min),
        float(cfg.generation.action_norm_clip_max),
    )
    s_g, keep_mask, stats = solve_predecessor_states(
        s_star=s_star,
        a_g=a_g,
        s_next_star=s_next_star,
        norms=norms,
        dyn_model=model,
        K=int(cfg.generation.K),
        lr_s=float(cfg.generation.lr_s),
        eps_opt=float(cfg.generation.eps_opt),
        r_max=float(cfg.generation.r_max),
        max_delta_s=float(cfg.generation.max_delta_s),
        grad_clip_norm=float(cfg.generation.grad_clip_norm),
        clamp_action_norm=clamp_spec,
    )
    residual_norm = cast(torch.Tensor, stats["residual_norm"])
    residual_mse = cast(torch.Tensor, stats["residual_mse"])
    predecessor_dist = cast(torch.Tensor, stats["predecessor_distance"])
    with torch.no_grad():
        a_g_n = norms.encode_action(a_g).clamp(min=clamp_spec[0], max=clamp_spec[1])
        a_g_clamped = norms.decode_action(a_g_n)
        pred_next_n = norms.encode_state(s_g) + model(s_g, a_g_clamped) / norms.state_std
        pred_next_error = torch.linalg.norm(pred_next_n - norms.encode_state(s_next_star), dim=-1)

    keep_float = keep_mask.float()
    kept_residual = residual_norm[keep_mask]
    kept_residual_mse = residual_mse[keep_mask]
    kept_dist = predecessor_dist[keep_mask]
    kept_pred_err = pred_next_error[keep_mask]
    print(f"batch_size: {s_star.shape[0]}")
    print(f"acceptance_rate: {keep_float.mean().item():.6f}")
    print(f"median_residual_norm: {residual_norm.median().item():.6f}")
    print(f"median_residual_mse: {residual_mse.median().item():.6f}")
    print(f"median_distance_to_expert_predecessor: {predecessor_dist.median().item():.6f}")
    print(f"median_pred_next_error: {pred_next_error.median().item():.6f}")
    if kept_residual.numel() > 0:
        print(f"median_residual_norm_kept: {kept_residual.median().item():.6f}")
        print(f"median_residual_mse_kept: {kept_residual_mse.median().item():.6f}")
        print(f"median_distance_kept: {kept_dist.median().item():.6f}")
        print(f"median_pred_next_error_kept: {kept_pred_err.median().item():.6f}")
    else:
        print("No kept samples. Try lowering sigma_action or lr_s, increasing K, or loosening max_delta_s.")


if __name__ == "__main__":
    main()
