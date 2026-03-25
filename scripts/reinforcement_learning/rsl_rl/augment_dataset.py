#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate CCIL synthetic transitions from expert episodes."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, cast

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

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


def apply_cfg_overrides(
    cfg: CcilGenerationCfg,
    cfg_dict: dict | None,
    overrides: list[str],
) -> None:
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
    if not isinstance(payload, dict):
        raise ValueError("Forward dynamics checkpoint payload must be a dict.")
    meta = payload.get("meta", {})
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    trainer_cfg = payload.get("trainer_cfg", {})
    model_cfg = trainer_cfg.get("model", {}) if isinstance(trainer_cfg, dict) else {}
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


def _append_to_step_bucket(
    bucket: dict[int, dict[str, list[torch.Tensor]]],
    step_idx: int,
    state_t: torch.Tensor,
    action_t: torch.Tensor,
    state_next: torch.Tensor,
) -> None:
    if step_idx not in bucket:
        bucket[step_idx] = {"state": [], "action": [], "next_state": []}
    bucket[step_idx]["state"].append(state_t.detach().cpu())
    bucket[step_idx]["action"].append(action_t.detach().cpu())
    bucket[step_idx]["next_state"].append(state_next.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment episodes with CCIL synthetic transitions.")
    parser.add_argument("--config", type=str, default=None, help="Optional .pt/.yaml/.yml config file.")
    args, overrides = parser.parse_known_args()

    cfg = CcilGenerationCfg()
    cfg_dict = load_cfg_dict(args.config)
    apply_cfg_overrides(cfg, cfg_dict, overrides)
    assert cfg.model.forward_dynamics_checkpoint != "", "model.forward_dynamics_checkpoint is required."
    assert cfg.generation.num_augs_per_step > 0, "generation.num_augs_per_step must be > 0."
    assert len(cfg.data.episode_paths) > 0, "data.episode_paths must be provided."

    torch.manual_seed(int(cfg.runtime.seed))
    torch.cuda.manual_seed_all(int(cfg.runtime.seed))
    device = _resolve_device(cfg.runtime.device)
    model, norms = _load_model_and_norms(cfg.model.forward_dynamics_checkpoint, device)
    episode_paths = expand_episode_paths(cfg.data.episode_paths)
    configured_out_dir = Path(cfg.data.output_dir)

    total_attempted = 0
    total_kept = 0
    progress = tqdm(episode_paths, desc="CCIL augment files", unit="file")
    step_progress = tqdm(desc="CCIL processed transitions", unit="step")
    for episode_file in progress:
        payload = torch.load(episode_file, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Episode payload must be dict: {episode_file}")
        episodes = payload.get("episodes", [])
        if not isinstance(episodes, list):
            raise ValueError(f"Expected list payload['episodes'] in: {episode_file}")
        file_attempted = 0
        file_kept = 0

        for episode in episodes:
            obs = episode["obs"]
            actions = episode["actions"]
            length = int(episode.get("length", actions.shape[0]))
            obs_seq = flatten_obs(obs, cfg.data.obs_keys)[:length]
            actions_seq = actions[:length].reshape(length, -1)

            max_steps = int(cfg.runtime.max_steps_per_episode) if cfg.runtime.max_steps_per_episode is not None else (length - 1)
            num_transitions = max(0, min(length - 1, max_steps))
            if num_transitions <= 0:
                episode["ccil_synthetic_by_timestep"] = {}
                episode["ccil_synthetic_flat"] = {
                    "state": torch.zeros((0, obs_seq.shape[-1]), dtype=torch.float32),
                    "action": torch.zeros((0, actions_seq.shape[-1]), dtype=torch.float32),
                    "next_state": torch.zeros((0, obs_seq.shape[-1]), dtype=torch.float32),
                }
                continue

            s_star_base = obs_seq[:num_transitions].to(device=device, dtype=torch.float32)
            a_star_base = actions_seq[:num_transitions].to(device=device, dtype=torch.float32)
            s_next_base = obs_seq[1 : num_transitions + 1].to(device=device, dtype=torch.float32)
            step_base = torch.arange(1, num_transitions + 1, device=device, dtype=torch.long)

            n_aug = int(cfg.generation.num_augs_per_step)
            s_star_rep = s_star_base.repeat_interleave(n_aug, dim=0)
            a_star_rep = a_star_base.repeat_interleave(n_aug, dim=0)
            s_next_rep = s_next_base.repeat_interleave(n_aug, dim=0)
            step_rep = step_base.repeat_interleave(n_aug, dim=0)

            step_bucket: dict[int, dict[str, list[torch.Tensor]]] = {}
            residual_mse_kept: list[torch.Tensor] = []
            dist_kept: list[torch.Tensor] = []
            batch_size = int(cfg.generation.solve_batch_size)
            clamp_spec = (
                float(cfg.generation.action_norm_clip_min),
                float(cfg.generation.action_norm_clip_max),
            )
            for start in range(0, s_star_rep.shape[0], batch_size):
                end = min(start + batch_size, s_star_rep.shape[0])
                s_star = s_star_rep[start:end]
                a_star = a_star_rep[start:end]
                s_next = s_next_rep[start:end]
                step_idx = step_rep[start:end]
                a_g = sample_disturbed_action(a_star, sigma_action=float(cfg.generation.sigma_action))
                s_g, keep_mask, solve_stats = solve_predecessor_states(
                    s_star=s_star,
                    a_g=a_g,
                    s_next_star=s_next,
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

                kept_ids = torch.nonzero(keep_mask, as_tuple=True)[0]
                if kept_ids.numel() > 0:
                    kept_step = step_idx[kept_ids].detach().cpu().tolist()
                    kept_s = s_g[kept_ids]
                    kept_a = a_g[kept_ids]
                    kept_next = s_next[kept_ids]
                    for i, step in enumerate(kept_step):
                        _append_to_step_bucket(
                            step_bucket,
                            int(step),
                            kept_s[i],
                            kept_a[i],
                            kept_next[i],
                        )
                    residual_mse = cast(torch.Tensor, solve_stats["residual_mse"])
                    predecessor_distance = cast(torch.Tensor, solve_stats["predecessor_distance"])
                    residual_mse_kept.append(residual_mse[kept_ids].detach().cpu())
                    dist_kept.append(predecessor_distance[kept_ids].detach().cpu())

                file_attempted += int(s_star.shape[0])
                file_kept += int(keep_mask.sum().item())
                step_progress.update(int(s_star.shape[0]))
                step_progress.set_postfix(
                    file=os.path.basename(episode_file),
                    file_kept=f"{file_kept}/{file_attempted}",
                )

            by_timestep: dict[int, dict[str, torch.Tensor]] = {}
            flat_state: list[torch.Tensor] = []
            flat_action: list[torch.Tensor] = []
            flat_next: list[torch.Tensor] = []
            for step_idx, values in step_bucket.items():
                state_tensor = torch.stack(values["state"], dim=0)
                action_tensor = torch.stack(values["action"], dim=0)
                next_tensor = torch.stack(values["next_state"], dim=0)
                by_timestep[step_idx] = {
                    "state": state_tensor,
                    "action": action_tensor,
                    "next_state": next_tensor,
                }
                flat_state.append(state_tensor)
                flat_action.append(action_tensor)
                flat_next.append(next_tensor)

            if flat_state:
                state_flat = torch.cat(flat_state, dim=0)
                action_flat = torch.cat(flat_action, dim=0)
                next_flat = torch.cat(flat_next, dim=0)
            else:
                state_flat = torch.zeros((0, obs_seq.shape[-1]), dtype=torch.float32)
                action_flat = torch.zeros((0, actions_seq.shape[-1]), dtype=torch.float32)
                next_flat = torch.zeros((0, obs_seq.shape[-1]), dtype=torch.float32)

            episode["ccil_synthetic_by_timestep"] = by_timestep
            episode["ccil_synthetic_flat"] = {
                "state": state_flat,
                "action": action_flat,
                "next_state": next_flat,
            }
            accepted_residual_mse = torch.cat(residual_mse_kept, dim=0) if residual_mse_kept else torch.zeros((0,), dtype=torch.float32)
            accepted_dist = torch.cat(dist_kept, dim=0) if dist_kept else torch.zeros((0,), dtype=torch.float32)
            episode["ccil_stats"] = {
                "num_attempted": file_attempted,
                "num_kept": file_kept,
                "acceptance_rate": 0.0 if file_attempted == 0 else float(file_kept / file_attempted),
                "median_residual_mse_kept": float(accepted_residual_mse.median().item()) if accepted_residual_mse.numel() > 0 else float("nan"),
                "median_predecessor_dist_kept": float(accepted_dist.median().item()) if accepted_dist.numel() > 0 else float("nan"),
            }

        payload["ccil_generation_cfg"] = asdict(cfg)
        payload["ccil_generation_stats"] = {
            "attempted": file_attempted,
            "kept": file_kept,
            "acceptance_rate": 0.0 if file_attempted == 0 else float(file_kept / file_attempted),
        }

        in_path = Path(episode_file)
        if configured_out_dir.is_absolute():
            out_dir = configured_out_dir
        else:
            # Relative output directories are created next to each source episode file.
            out_dir = in_path.parent / configured_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{in_path.stem}{cfg.data.output_suffix}.pt"
        out_path = out_dir / out_name
        if out_path.exists() and not bool(cfg.data.overwrite):
            raise FileExistsError(f"Output exists and overwrite=False: {out_path}")
        torch.save(payload, out_path)
        print(
            f"[CCIL] {in_path.name} -> {out_path.name} "
            f"(kept={file_kept}/{file_attempted}, rate={0.0 if file_attempted == 0 else file_kept / file_attempted:.4f})"
        )
        total_attempted += file_attempted
        total_kept += file_kept
        running_rate = 0.0 if total_attempted == 0 else total_kept / total_attempted
        progress.set_postfix(
            kept=f"{total_kept}/{total_attempted}",
            rate=f"{running_rate:.4f}",
        )
    progress.close()
    step_progress.close()

    total_rate = 0.0 if total_attempted == 0 else total_kept / total_attempted
    print(f"[CCIL] Total kept={total_kept}/{total_attempted}, acceptance_rate={total_rate:.4f}")


if __name__ == "__main__":
    main()
