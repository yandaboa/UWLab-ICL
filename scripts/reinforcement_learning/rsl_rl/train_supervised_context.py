#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone supervised training for context-conditioned transformers."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import MISSING, asdict
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
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

from uwlab_rl.rsl_rl.context_sequence_policy import ContextSequencePolicy
from uwlab_rl.rsl_rl.lr_utils import build_lr_scheduler
from uwlab_rl.rsl_rl.supervised_context_cfg import SupervisedContextTrainerCfg
# from uwlab_tasks.manager_based.manipulation.from_demo.config.ur5e_robotiq_2f85.agents.supervised_context_cfg import SupervisedContextRunnerCfg
from uwlab_rl.rsl_rl.supervised_context_utils import (
    ContextStepDataset,
    collate_context_steps,
    resolve_action_bin_values,
    load_action_discretization_spec,
    reduce_loss_if_needed,
    resolve_distributed,
    resolve_action_bins,
    seed_everything,
)

from isaaclab.utils.io import dump_yaml


def _should_log(is_multi_gpu: bool, rank: int) -> bool:
    return not is_multi_gpu or rank == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised context training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config .pt/.yaml file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--episode_paths", nargs="+", default=None, help="Episode .pt files or globs.")
    parser.add_argument(
        "--include_current_trajectory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append current rollout to context before prediction.",
    )
    parser.add_argument(
        "--context_token_layout",
        choices=["merged", "state_action", "state_only"],
        default=None,
        help="Token layout for context inputs.",
    )
    parser.add_argument(
        "--include_actions_in_context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include action terms in merged context tokens.",
    )
    parser.add_argument(
        "--include_rewards_in_context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include reward terms in merged context tokens.",
    )
    parser.add_argument(
        "--share_current_and_context_obs_projection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse one projection for current_obs and context_obs (requires matching dims).",
    )
    parser.add_argument(
        "--encoding_projection_hidden_dim",
        type=int,
        default=None,
        help="Optional hidden size for projection MLP (in->hidden->embedding).",
    )
    parser.add_argument("--num_steps", type=int, default=None, help="Number of optimizer steps.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--max_context_length", type=int, default=None, help="Optional max context length.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loader workers.")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of layers.")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of heads.")
    args, _ = parser.parse_known_args()

    cfg = SupervisedContextTrainerCfg()
    if args.config is not None:
        cfg_dict = torch.load(args.config) if args.config.endswith(".pt") else None
        if cfg_dict is not None:
            for k, v in cfg_dict.items():
                setattr(cfg, k, v)
    if args.episode_paths is not None:
        cfg.data.episode_paths = args.episode_paths
    if args.include_current_trajectory is not None:
        cfg.input.include_current_trajectory = args.include_current_trajectory
    if args.context_token_layout is not None:
        cfg.model.context_token_layout = args.context_token_layout
    if args.include_actions_in_context is not None:
        cfg.model.include_actions_in_context = args.include_actions_in_context
    if args.include_rewards_in_context is not None:
        cfg.model.include_rewards_in_context = args.include_rewards_in_context
    if args.share_current_and_context_obs_projection is not None:
        cfg.model.share_current_and_context_obs_projection = args.share_current_and_context_obs_projection
    if args.encoding_projection_hidden_dim is not None:
        cfg.model.encoding_projection_hidden_dim = args.encoding_projection_hidden_dim
    if args.num_steps is not None:
        cfg.optim.num_steps = args.num_steps
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.max_context_length is not None:
        cfg.data.max_context_length = args.max_context_length
    if args.hidden_dim is not None:
        cfg.model.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        cfg.model.num_layers = args.num_layers
    if args.num_heads is not None:
        cfg.model.num_heads = args.num_heads
    if args.run_name is not None:
        cfg.logging.run_name = args.run_name
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if cfg.distributed.distributed and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    is_multi_gpu, world_size, rank = resolve_distributed(cfg.distributed.distributed)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(args.seed + rank)

    if cfg.data.episode_paths is MISSING:
        raise ValueError("episode_paths must be provided via config or --episode_paths.")
    episode_paths = cfg.data.episode_paths
    dataset = ContextStepDataset(episode_paths, cfg.data.obs_keys)
    sampler = DistributedSampler(dataset) if is_multi_gpu else None
    effective_drop_last = len(dataset) >= int(cfg.data.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle if sampler is None else False,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        drop_last=effective_drop_last,
        collate_fn=collate_context_steps,
    )

    if len(dataset) == 0:
        raise RuntimeError("No episodes found for supervised training.")
    sample = dataset[0]
    obs_dim = int(sample["obs"].shape[-1])
    action_dim = int(sample["actions"].shape[-1])
    reward_dim = int(sample["rewards"].shape[-1])
    cfg.model.num_actions = action_dim
    action_bins = None
    action_bin_values = None
    if cfg.model.action_distribution == "categorical":
        spec = load_action_discretization_spec(
            cfg.data.episode_paths,
            cfg.model.action_discretization_spec_path,
        )
        assert spec is not None, "Categorical actions require action discretization spec."
        action_bins = resolve_action_bins(spec, action_dim)
        action_bin_values = resolve_action_bin_values(spec, action_dim)
    model = ContextSequencePolicy(
        cfg,
        obs_dim,
        action_dim,
        reward_dim,
        action_bins=action_bins,
        action_bin_values=action_bin_values,
    ).to(device)
    if is_multi_gpu:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer_class = getattr(torch.optim, cfg.optim.optimizer_class)
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
    )
    lr_scheduler = build_lr_scheduler(cfg.optim, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.use_amp and torch.cuda.is_available())

    log_root = os.path.join("logs", "rsl_rl", cfg.logging.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.logging.run_name:
        run_name = f"{cfg.logging.run_name}"
    log_dir = os.path.join(log_root, run_name)
    if _should_log(is_multi_gpu, rank):
        os.makedirs(log_dir, exist_ok=True)
        params_dir = os.path.join(log_dir, "params")
        os.makedirs(params_dir, exist_ok=True)
        dump_yaml(os.path.join(params_dir, "trainer.yaml"), asdict(cfg))
    wandb_run = None
    if _should_log(is_multi_gpu, rank) and cfg.logging.use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            wandb = None
        if wandb is not None:
            project_name = cfg.logging.log_project_name or cfg.logging.experiment_name
            wandb_run = wandb.init(
                project=project_name,
                name=cfg.logging.run_name or run_name,
                config=asdict(cfg),
            )

    total_steps = 0
    total_steps_target = int(cfg.optim.num_steps)
    progress_bar = None
    if _should_log(is_multi_gpu, rank):
        progress_bar = tqdm(total=total_steps_target, desc="Supervised updates", unit="step")
    for epoch in range(int(math.ceil(cfg.optim.num_steps / max(len(loader), 1)))):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            demo_obs = batch["demo_obs"].to(device)
            demo_actions = batch["demo_actions"].to(device)
            demo_rewards = batch["demo_rewards"].to(device)
            demo_lengths = batch["demo_lengths"].to(device)
            current_obs = batch["current_obs"].to(device)
            target_action = batch["target_action"].to(device)
            model_module = model.module if isinstance(model, DistributedDataParallel) else model
            with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                if cfg.input.include_current_trajectory:
                    raise ValueError("include_current_trajectory=True is not supported yet.")
                loss = model_module.compute_supervised_loss(
                    demo_obs=demo_obs,
                    demo_actions=demo_actions,
                    demo_rewards=demo_rewards,
                    demo_lengths=demo_lengths,
                    current_obs=current_obs,
                    target_action=target_action,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if total_steps % cfg.logging.log_interval == 0:
                reduced_loss = reduce_loss_if_needed(loss, is_multi_gpu)
                if _should_log(is_multi_gpu, rank):
                    grad_norm_value = float(grad_norm)
                    lr_value = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
                    # print(
                    #     f"[step {total_steps}] loss={reduced_loss.item():.6f} "
                    #     f"grad_norm={grad_norm_value:.6f} lr={lr_value}"
                    # )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": reduced_loss.item(),
                                "train/grad_norm": grad_norm_value,
                                "train/lr": lr_value,
                                "train/step": total_steps,
                            }
                        )
            if _should_log(is_multi_gpu, rank) and total_steps % cfg.logging.save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{total_steps:06d}.pt")
                model_module = model.module if isinstance(model, DistributedDataParallel) else model
                payload = model_module.get_state_dict_payload()
                payload["meta"] = {
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "reward_dim": reward_dim,
                    "action_bins": action_bins,
                    "action_bin_values": None
                    if action_bin_values is None
                    else [values.tolist() for values in action_bin_values],
                }
                torch.save(payload, ckpt_path)
            total_steps += 1
            if progress_bar is not None:
                progress_bar.update(1)
            if total_steps >= cfg.optim.num_steps:
                break
        if total_steps >= cfg.optim.num_steps:
            break
    if progress_bar is not None:
        progress_bar.close()


if __name__ == "__main__":
    main()
