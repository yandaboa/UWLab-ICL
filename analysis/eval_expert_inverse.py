"""Evaluate the RL teacher (expert) under inverse-action-mapping in the
augmented (perturbation) student-eval env.

Same physics + perturbation events as the env we eval distilled students on
(``OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0``),
same inverse-mapping used during data collection (``arm_term.inverse_process_actions``
with the expert's training-time scale, so the env's per-env action_scale is
compensated). Reports EOE and any-time success rates.

Use this to set the upper-bound number for "what does the dataset signal that a
perfect distilled student could achieve" — anything above this is impossible
(eval env = student env), anything below is the distillation gap.

Usage:
    python analysis/eval_expert_inverse.py --num_episodes 512 --num_envs 64
"""
from __future__ import annotations

import argparse
import os
import sys

# numpy/numba pre-import to avoid Isaac Sim's pip_prebundle clobbering them
import numpy as np  # noqa: F401, E402
import numba  # noqa: F401, E402
import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-DataCollection-v0",
    help="Default = augmented data-collection env (same as expert demo collection).",
)
parser.add_argument(
    "--expert_checkpoint",
    default="logs/rsl_rl/teacher/exported/policy.pt",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=512)
parser.add_argument("--episode_length_s", type=float, default=11.0)
parser.add_argument("--insertive_object", default="peg")
parser.add_argument("--receptive_object", default="peghole")
parser.add_argument(
    "--expert_action_scale",
    nargs=6,
    type=float,
    default=[0.01, 0.01, 0.002, 0.02, 0.02, 0.2],
    help="Scale the expert was trained with (used as original_scale for inverse_process_actions).",
)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# isaaclab + uwlab imports must come AFTER AppLauncher
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402, F401
from tqdm import tqdm  # noqa: E402


def _get_progress_context(env):
    rm = env.unwrapped.reward_manager
    try:
        return rm.get_term_cfg("progress_context").func
    except Exception:
        return None


def main():
    device = torch.device("cuda")

    # Build env (no recorder)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = args_cli.episode_length_s
    # NOTE: env_cfg.scene.insertive_object/receptive_object are RigidObjectCfg
    # defaults already pointing at peg/peghole. To select different assets
    # you'd need to pass full Hydra overrides; we just default to peg+peghole.
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.concatenate_terms = False
    # The teacher TorchScript expects a flat concatenated tensor (the obs format
    # it was trained against). BasePolicyCfg defaults concatenate_terms=False,
    # so explicitly set the expert_obs group to True.
    if hasattr(env_cfg.observations, "expert_obs") and env_cfg.observations.expert_obs is not None:
        env_cfg.observations.expert_obs.concatenate_terms = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[eval-expert] task = {args_cli.task}")
    print(f"[eval-expert] num_envs = {args_cli.num_envs}, target episodes = {args_cli.num_episodes}")

    # Load teacher TorchScript
    expert = torch.jit.load(args_cli.expert_checkpoint, map_location=device).to(device).eval()

    # Inverse-mapping prerequisites
    arm_term = env.unwrapped.action_manager._terms.get("arm")
    if arm_term is None or not hasattr(arm_term, "inverse_process_actions"):
        raise RuntimeError("env's 'arm' action term is missing or has no inverse_process_actions")
    expert_original_scale = torch.tensor(
        args_cli.expert_action_scale, device=device, dtype=torch.float32
    )
    print(f"[eval-expert] expert_original_scale = {args_cli.expert_action_scale}")

    def expert_obs(env):
        ob = env.unwrapped.obs_buf
        if isinstance(ob, dict) and "expert_obs" in ob and ob["expert_obs"] is not None:
            return ob["expert_obs"]
        if isinstance(ob, dict) and "policy" in ob:
            return ob["policy"]
        return ob

    obs_dict, _ = env.reset()
    progress_ctx = _get_progress_context(env)
    if progress_ctx is None or not hasattr(progress_ctx, "success"):
        raise RuntimeError("env has no progress_context.success — can't measure success.")

    # Per-env trackers
    episode_ever_successful = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    episodes = 0
    eoe_successes = 0
    anytime_successes = 0

    pbar = tqdm(total=args_cli.num_episodes, desc="Eval expert (EOE: 0%, any: 0%)")

    while simulation_app.is_running() and episodes < args_cli.num_episodes:
        with torch.inference_mode():
            ob = expert_obs(env)
            mean, _ = expert.compute_distribution(ob)
            # inverse_process_actions modifies in-place
            env_actions = arm_term.inverse_process_actions(
                mean.clone(), original_scale=expert_original_scale
            )
            step_result = env.step(env_actions)
            if len(step_result) == 4:
                obs_dict, _, dones, _ = step_result
            else:
                obs_dict, _, terminated, truncated, _ = step_result
                dones = terminated | truncated

            # Track any-time success: was success ever True during this episode?
            episode_ever_successful |= progress_ctx.success

            # On done: count, and capture EOE = success state at the terminal step.
            if dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                # progress_ctx.success at this moment (after the step that triggered done)
                # is the terminal-step success → EOE.
                eoe_at_term = progress_ctx.success[reset_ids]
                anytime_at_term = episode_ever_successful[reset_ids]
                num_new = len(reset_ids)
                episodes += num_new
                eoe_successes += int(eoe_at_term.sum().item())
                anytime_successes += int(anytime_at_term.sum().item())
                episode_ever_successful[reset_ids] = False
                pbar.update(num_new)
                eoe_pct = 100.0 * eoe_successes / max(episodes, 1)
                any_pct = 100.0 * anytime_successes / max(episodes, 1)
                pbar.set_description(f"Eval expert (EOE: {eoe_pct:.1f}%, any: {any_pct:.1f}%)")

    pbar.close()
    eoe_rate = eoe_successes / max(episodes, 1)
    any_rate = anytime_successes / max(episodes, 1)

    print()
    print("=" * 60)
    print(f"[eval-expert] FINAL ({episodes} episodes)")
    print(f"  EOE success rate:        {100*eoe_rate:.2f}%  ({eoe_successes}/{episodes})")
    print(f"  any-time success rate:   {100*any_rate:.2f}%  ({anytime_successes}/{episodes})")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
