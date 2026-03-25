#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a supervised MLP policy checkpoint in IsaacLab."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a supervised MLP policy in IsaacLab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_steps", type=int, required=True, help="Number of environment steps to run.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log_interval", type=int, default=100, help="How often to print eval stats.")
parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable wandb logging for eval stats.")
parser.add_argument("--wandb_project", type=str, default="supervised_policy_eval", help="Weights & Biases project.")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional Weights & Biases run name.")
parser.add_argument("--task_episodes", type=str, default=None, help="Optional task episodes to evaluate performance on. Environment starts at these configurations")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from tqdm import tqdm

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

try:
    from uwlab_rl.rsl_rl.supervised_mlp_policy import load_supervised_policy_checkpoint
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[3]
    _source_root = _repo_root / "source" / "uwlab_rl"
    if _source_root.is_dir():
        sys.path.append(str(_source_root))
    from uwlab_rl.rsl_rl.supervised_mlp_policy import load_supervised_policy_checkpoint


def _flatten_observation_value(value: Any, device: torch.device) -> torch.Tensor:
    """Flatten observation structure into shape [num_envs, features]."""
    if isinstance(value, torch.Tensor):
        tensor_value = value.to(device=device, dtype=torch.float32)
        return tensor_value.reshape(tensor_value.shape[0], -1)

    if isinstance(value, Mapping):
        chunks = [_flatten_observation_value(value[key], device=device) for key in value.keys()]
        if not chunks:
            raise RuntimeError("Empty observation mapping cannot be flattened.")
        return torch.cat(chunks, dim=-1)

    tensor_value = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor_value.ndim == 1:
        tensor_value = tensor_value.unsqueeze(0)
    return tensor_value.reshape(tensor_value.shape[0], -1)


def _extract_policy_state(obs: Any, device: torch.device) -> torch.Tensor:
    """Extract policy observation and flatten it for the supervised MLP."""
    if isinstance(obs, Mapping):
        if "policy" in obs:
            policy_obs = obs["policy"]
        else:
            first_key = next(iter(obs))
            policy_obs = obs[first_key]
        return _flatten_observation_value(policy_obs, device=device)
    return _flatten_observation_value(obs, device=device)


def _unwrap_initial_obs(obs_payload: Any) -> Any:
    if isinstance(obs_payload, tuple):
        return obs_payload[0]
    return obs_payload


def _init_wandb() -> Any | None:
    if not args_cli.use_wandb:
        return None
    try:
        import wandb as wandb_module  # type: ignore
    except ImportError:
        print("[WARN] --use_wandb requested but wandb is not installed.")
        return None
    wandb_api: Any = wandb_module
    return wandb_api.init(project=args_cli.wandb_project, name=args_cli.wandb_run_name, config=vars(args_cli))


def _extract_scalar_log_metrics_from_extras(extras: Any) -> dict[str, float]:
    if not isinstance(extras, Mapping):
        return {}
    log_section = extras.get("log")
    if not isinstance(log_section, Mapping):
        return {}
    payload: dict[str, float] = {}
    for key, value in log_section.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                payload[str(key)] = float(value.detach().item())
            continue
        if isinstance(value, (int, float)):
            payload[str(key)] = float(value)
    return payload


def _to_wandb_config_value(value: Any) -> Any:
    """Convert checkpoint payload values to wandb-config-friendly types."""
    if isinstance(value, Mapping):
        return {str(k): _to_wandb_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wandb_config_value(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Load supervised policy and roll out stochastic actions."""
    if args_cli.num_steps <= 0:
        raise ValueError(f"--num_steps must be > 0, got {args_cli.num_steps}")
    if args_cli.log_interval <= 0:
        raise ValueError(f"--log_interval must be > 0, got {args_cli.log_interval}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.task_episodes is not None:
        task_episodes = torch.load(args_cli.task_episodes)
        reset_states = task_episodes["reset_states"]
        reset_state_idx_raw = reset_states[0]["multi_reset_state_index"]
        # Normalize to a 1-D list so reset event code can safely call len(...).
        reset_state_idx = [int(v) for v in torch.as_tensor(reset_state_idx_raw).reshape(-1).tolist()]
        env_cfg.events.reset_from_reset_states.params["state_indices_override"] = reset_state_idx

    resume_path = retrieve_file_path(args_cli.checkpoint)
    model_name = os.path.basename(os.path.dirname(resume_path))
    task_episodes_path = os.path.abspath(args_cli.task_episodes) if args_cli.task_episodes is not None else "None"
    print(f"[INFO] Evaluating model={model_name}")
    print(f"[INFO] Evaluation checkpoint path={os.path.abspath(resume_path)}")
    print(f"[INFO] Evaluation task_episodes path={task_episodes_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval_supervised"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = torch.device(agent_cfg.device)
    print(f"[INFO] Loading supervised checkpoint from: {resume_path}")
    policy, checkpoint = load_supervised_policy_checkpoint(resume_path, device=device)
    policy.eval()
    print(
        f"[INFO] Loaded model with state_dim={checkpoint['state_dim']} "
        f"action_dim={checkpoint['action_dim']} loss_type={checkpoint.get('loss_type', 'gaussian_nll')}"
    )
    wandb_run = _init_wandb()
    if wandb_run is not None:
        checkpoint_config = checkpoint.get("config", {})
        if isinstance(checkpoint_config, Mapping):
            wandb_model_config = _to_wandb_config_value(dict(checkpoint_config))
        else:
            wandb_model_config = {
                "state_dim": int(checkpoint["state_dim"]),
                "action_dim": int(checkpoint["action_dim"]),
                "hidden_dims": list(checkpoint.get("hidden_dims", [])),
                "loss_type": str(checkpoint.get("loss_type", "gaussian_nll")),
                "log_std_min": float(checkpoint.get("log_std_min", -5.0)),
                "log_std_max": float(checkpoint.get("log_std_max", 2.0)),
            }
        wandb_run.config.update(
            {
                "eval_model_name": model_name,
                "eval_checkpoint_path": os.path.abspath(resume_path),
                "eval_task_episodes_path": task_episodes_path,
                "eval_model_config": wandb_model_config,
            },
            allow_val_change=True,
        )

    dt = float(env.unwrapped.step_dt)
    obs = _unwrap_initial_obs(env.get_observations())
    steps_executed = 0
    num_envs = int(env.num_envs)
    running_returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
    running_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    recent_episode_returns: deque[float] = deque(maxlen=200)
    recent_episode_lengths: deque[int] = deque(maxlen=200)
    total_reward_sum = 0.0
    latest_extras_log_metrics: dict[str, float] = {}

    total_successes = 0
    total_episodes = 0

    progress_bar = tqdm(total=args_cli.num_steps, desc="Evaluating supervised policy", unit="step")
    try:
        while steps_executed < args_cli.num_steps and simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                state = _extract_policy_state(obs, device=device)
                actions = policy.act(state, stochastic=True)
                obs, rewards, dones, extras = env.step(actions)
                latest_extras_log_metrics = _extract_scalar_log_metrics_from_extras(extras)
                rewards = rewards.reshape(-1).to(device=device, dtype=torch.float32)
                dones = dones.reshape(-1).to(device=device)
                running_returns += rewards
                running_lengths += 1
                total_reward_sum += float(rewards.sum().item())
                done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    total_episodes += done_ids.numel()
                    total_successes += int((rewards[done_ids] > 0.1).sum().item())
                    done_returns = running_returns[done_ids]
                    done_lengths = running_lengths[done_ids]
                    recent_episode_returns.extend(done_returns.detach().cpu().tolist())
                    recent_episode_lengths.extend(done_lengths.detach().cpu().tolist())
                    running_returns[done_ids] = 0.0
                    running_lengths[done_ids] = 0
            steps_executed += 1
            progress_bar.update(1)

            if steps_executed % args_cli.log_interval == 0 or steps_executed == args_cli.num_steps:
                mean_step_reward = total_reward_sum / max(1, steps_executed * num_envs)
                stats_message = (
                    f"[STATS] step={steps_executed}/{args_cli.num_steps} "
                    f"mean_step_reward={mean_step_reward:.6f}"
                )
                log_payload: dict[str, float | int] = {
                    "eval/mean_step_reward": mean_step_reward,
                    "eval/step": steps_executed,
                }
                if recent_episode_returns:
                    mean_episode_return = float(sum(recent_episode_returns) / len(recent_episode_returns))
                    mean_episode_length = float(sum(recent_episode_lengths) / len(recent_episode_lengths))
                    stats_message += (
                        f" mean_recent_episode_return={mean_episode_return:.6f}"
                        f" mean_recent_episode_length={mean_episode_length:.2f}"
                        f" num_recent_episodes={len(recent_episode_returns)}"
                    )
                    log_payload["eval/mean_recent_episode_return"] = mean_episode_return
                    log_payload["eval/mean_recent_episode_length"] = mean_episode_length
                    log_payload["eval/recent_episode_count"] = len(recent_episode_returns)
                if latest_extras_log_metrics:
                    log_payload.update(latest_extras_log_metrics)
                    if "Metrics/task_command/end_of_episode_success_rate" in latest_extras_log_metrics:
                        stats_message += (
                            " task_success_rate="
                            f"{latest_extras_log_metrics['Metrics/task_command/end_of_episode_success_rate']:.4f}"
                        )
                    if "Metrics/task_command/average_pos_align_error" in latest_extras_log_metrics:
                        stats_message += (
                            " task_pos_align_error="
                            f"{latest_extras_log_metrics['Metrics/task_command/average_pos_align_error']:.6f}"
                        )
                print(stats_message)
                if wandb_run is not None:
                    wandb_run.log(log_payload, step=steps_executed)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        success_rate = (float(total_successes) / float(total_episodes)) if total_episodes > 0 else 0.0
        print(f"[INFO] Completed supervised evaluation for {steps_executed} steps.")
        print(f"[INFO] Total successes: {total_successes}")
        print(f"[INFO] Total episodes: {total_episodes}")
        print(f"[INFO] Success rate: {success_rate * 100:.2f}%")

        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "success_rate": success_rate,
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                }
            )
            wandb_run.summary["eval_model_name"] = model_name
            wandb_run.summary["eval_checkpoint_path"] = os.path.abspath(resume_path)
            wandb_run.summary["eval_task_episodes_path"] = task_episodes_path
    finally:
        progress_bar.close()
        if wandb_run is not None:
            wandb_run.finish()


    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
