# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to simulate.")
parser.add_argument(
    "--skill",
    type=int,
    default=None,
    help=(
        "Force every env to this skill index (Diversity tasks only). Sets "
        "env.observations.policy.skill.params.force_skill and tags the video filename."
    ),
)
parser.add_argument(
    "--action_rescale",
    action="store_true",
    default=False,
    help=(
        "Apply the hard-coded inverse-action rescale to the finetune-eval action scale "
        "[0.01, 0.01, 0.002, 0.02, 0.02, 0.2]. Off by default; pass this only for policies "
        "trained at those scales (Stage-2 / finetune-eval)."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--num_bins",
    type=int,
    default=0,
    help=(
        "If > 0, discretize the 6 continuous arm action dims into this many uniform bins "
        "over [--discretize_clip_val, +--discretize_clip_val]. The binary gripper dim (index 6) "
        "is always thresholded at 0. 0 disables discretization (continuous actions)."
    ),
)
parser.add_argument(
    "--discretize_clip_val",
    type=float,
    default=2.0,
    help="Symmetric clip range used when binning continuous arm actions. Default: 2.0.",
)
parser.add_argument(
    "--save_rollout",
    action="store_true",
    default=False,
    help="If set, save rollout policy observations and actions to log_dir/rollouts/play/<timestamp>.pt.",
)
parser.add_argument(
    "--save_ee_traj",
    action="store_true",
    default=False,
    help="If set, save per-step world-frame end-effector position (wrist_3_link) to log_dir/rollouts/play/ee_traj-<tag>.pt.",
)
parser.add_argument(
    "--save_rollout_steps",
    type=int,
    default=0,
    help="Number of inference steps to log for rollout. If <= 0, logs until simulation exits.",
)
parser.add_argument(
    "--plot_privileged_debug",
    action="store_true",
    default=False,
    help=(
        "If set, sweep privileged policy observations at each step and save overlaid action-distribution plots "
        "for the single active environment."
    ),
)
parser.add_argument(
    "--privileged_debug_sweep_key",
    type=str,
    default="friction",
    help=(
        "Privileged observation to sweep. "
        "'friction' / 'restitution' sweep material-property terms (joint insertive/receptive via "
        "--privileged_debug_joint_insertive_receptive). "
        "'action_offset' / 'task_frame_force_bias' sweep each of the 6 dims of the corresponding "
        "augmentation observation term (robot_action_offset / robot_task_frame_force_bias); "
        "default per-dim sweep ranges match [-hi, +hi] from augmentation_handler."
    ),
)
parser.add_argument(
    "--privileged_debug_sweep_min",
    type=float,
    default=None,
    help=(
        "Optional minimum counterfactual value for the privileged debug sweep. "
        "For 'action_offset' / 'task_frame_force_bias', overrides the per-dim defaults uniformly."
    ),
)
parser.add_argument(
    "--privileged_debug_sweep_max",
    type=float,
    default=None,
    help=(
        "Optional maximum counterfactual value for the privileged debug sweep. "
        "For 'action_offset' / 'task_frame_force_bias', overrides the per-dim defaults uniformly."
    ),
)
parser.add_argument(
    "--privileged_debug_num_points",
    type=int,
    default=9,
    help="Number of counterfactual points to evaluate in each privileged debug sweep.",
)
parser.add_argument(
    "--privileged_debug_joint_insertive_receptive",
    action="store_true",
    default=False,
    help=(
        "Also add a coupled sweep that varies insertive and receptive object privileged material properties "
        "together with the same counterfactual value."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from tqdm import tqdm

from rsl_rl.runners import DistillationRunner, DiversityRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab.envs.mdp.observations import last_action

from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx
from uwlab_rl.utils.action_discretize import discretize_actions

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from privileged_policy_debug import PrivilegedPolicyDebugger
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

def get_perturbed_env_solving_actions(env, actions: torch.Tensor) -> torch.Tensor:
    """Get the perturbed environment-solving actions.
    We assume that the actions are coming from a post-trained policy on sysID gains
    
    Args:
        env: The environment to get the perturbed actions for.
        actions: The actions to perturb.
        
    Returns:
    """
    action_scale = torch.tensor([0.01, 0.01, 0.002, 0.02, 0.02, 0.2], device=actions.device, dtype=actions.dtype)
    return env.action_manager._terms.get("arm").inverse_process_actions(actions, original_scale=action_scale)

def set_action_override(env, actions: torch.Tensor) -> torch.Tensor:
    """Set the action override.
    """
    env.action_override = actions

# PLACEHOLDER: Extension template (do not remove this comment)


def _to_cpu_detached(value):
    """Convert tensors/nested structures to CPU tensors for serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_detached(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {k: _to_cpu_detached(v) for k, v in value.items()}
    return value


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Optional: lock every env to a single skill (Diversity tasks only).
    if args_cli.skill is not None:
        skill_term = getattr(getattr(env_cfg.observations, "policy", None), "skill", None)
        if skill_term is None:
            raise ValueError(
                "--skill was passed but env_cfg.observations.policy has no 'skill' term. "
                "Use a Diversity-* task."
            )
        skill_term.params["force_skill"] = int(args_cli.skill)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        skill_tag = f"-skill{args_cli.skill}" if args_cli.skill is not None else ""
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"play{skill_tag}-{video_timestamp}",
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.plot_privileged_debug:
        assert env.num_envs == 1, "--plot_privileged_debug requires --num_envs 1."

    if args_cli.num_bins > 0:
        print(f"[INFO]: Action discretization ENABLED — {args_cli.num_bins} bins, clip_val={args_cli.discretize_clip_val}")
    else:
        print("[INFO]: Action discretization DISABLED (continuous)")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DiversityRunner":
        runner = DiversityRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # play only needs the actor for inference, so we skip loading the critic and its
    # obs normalizer. this allows replaying checkpoints whose critic obs space differs
    # from the current env (e.g. legacy/privileged critics with extra terms).
    loaded_dict = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
    actor_only_state_dict = {
        k: v
        for k, v in loaded_dict["model_state_dict"].items()
        if not (k.startswith("critic.") or k.startswith("critic_obs_normalizer."))
    }
    skipped_keys = sorted(set(loaded_dict["model_state_dict"]) - set(actor_only_state_dict))
    if skipped_keys:
        print(f"[INFO]: Skipping critic params from checkpoint: {len(skipped_keys)} keys")
    runner.alg.policy.load_state_dict(actor_only_state_dict, strict=False)
    if "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]

    # If this is a Diversity run, also load the discriminator weights from the same
    # checkpoint and re-publish on the env so the diversity reward term works.
    if isinstance(runner, DiversityRunner) and "discriminator_state_dict" in loaded_dict:
        runner.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        runner.alg.attach_env(env)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    privileged_debugger = None
    if args_cli.plot_privileged_debug:
        if (args_cli.privileged_debug_sweep_min is None) != (args_cli.privileged_debug_sweep_max is None):
            raise ValueError(
                "Set both --privileged_debug_sweep_min and --privileged_debug_sweep_max, or leave both unset."
            )
        sweep_range = None
        if args_cli.privileged_debug_sweep_min is not None and args_cli.privileged_debug_sweep_max is not None:
            sweep_range = (args_cli.privileged_debug_sweep_min, args_cli.privileged_debug_sweep_max)
        privileged_debugger = PrivilegedPolicyDebugger(
            policy_nn,
            Path(log_dir) / "privileged_debug",
            sweep_key=args_cli.privileged_debug_sweep_key,
            sweep_range=sweep_range,
            num_sweep_points=args_cli.privileged_debug_num_points,
            include_joint_insertive_receptive=args_cli.privileged_debug_joint_insertive_receptive,
        )

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    rollout_data = None
    rollout_save_path = None
    if args_cli.save_rollout:
        rollout_timestamp = time.strftime("%Y%m%d_%H%M%S")
        rollout_save_path = Path(log_dir) / "rollouts" / "play" / f"{rollout_timestamp}.pt"
        rollout_data = {
            "policy_obs": [],
            "actions": [],
            "task": args_cli.task,
            "checkpoint": str(resume_path),
        }

    ee_traj_buf = None
    ee_body_idx = None
    ee_traj_save_path = None
    if args_cli.save_ee_traj:
        robot = env.unwrapped.scene["robot"]
        ee_body_idx = robot.body_names.index("wrist_3_link")
        ee_traj_buf = []
        ee_tag = time.strftime("%Y%m%d-%H%M%S")
        if args_cli.skill is not None:
            ee_tag = f"skill{args_cli.skill}-{ee_tag}"
        ee_traj_save_path = Path(log_dir) / "rollouts" / "play" / f"ee_traj-{ee_tag}.pt"

    # infer a finite horizon for progress reporting when available
    progress_total = None
    if args_cli.video:
        progress_total = args_cli.video_length
    if args_cli.num_steps is not None:
        progress_total = args_cli.num_steps if progress_total is None else min(progress_total, args_cli.num_steps)
    if rollout_data is not None and args_cli.save_rollout_steps > 0:
        progress_total = (
            args_cli.save_rollout_steps if progress_total is None else min(progress_total, args_cli.save_rollout_steps)
        )

    # simulate environment
    num_episodes = 0
    num_successes = 0
    progress_bar = tqdm(total=progress_total, desc="Play rollout", unit="step")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if rollout_data is not None:
                rollout_data["policy_obs"].append(_to_cpu_detached(obs))
            if privileged_debugger is not None:
                privileged_debugger.plot_step(obs, timestep)
            # agent stepping
            actions = policy(obs)
            if rollout_data is not None:
                rollout_data["actions"].append(_to_cpu_detached(actions))
            # rescale actions, this solves for action scale + offset changes.
            # remove this line if you dont want to compensate for action scale + offset for the policy automatically
            set_action_override(env.unwrapped, actions.clone())
            if args_cli.action_rescale:
                actions = get_perturbed_env_solving_actions(env.unwrapped, actions)
            if args_cli.num_bins > 0:
                actions = discretize_actions(actions, args_cli.num_bins, args_cli.discretize_clip_val)

            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            if ee_traj_buf is not None:
                ee_traj_buf.append(
                    env.unwrapped.scene["robot"].data.body_pos_w[:, ee_body_idx, :].detach().cpu().clone()
                )
            if dones.any():
                num_episodes += dones.sum().item()
                num_successes += torch.logical_and(rewards > 0.1, dones).sum().item()
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        
        timestep += 1
        progress_bar.update(1)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        if args_cli.num_steps is not None and timestep >= args_cli.num_steps:
            break
        
        if rollout_data is not None and args_cli.save_rollout_steps > 0 and timestep >= args_cli.save_rollout_steps:
            print(f"[INFO] Collected requested rollout steps: {args_cli.save_rollout_steps}")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    progress_bar.close()

    if privileged_debugger is not None:
        debug_video_num_steps = args_cli.num_steps if args_cli.num_steps is not None else timestep
        privileged_debugger.save_time_series_videos(rollout_step_dt=dt, target_num_steps=debug_video_num_steps)

    if rollout_data is not None:
        rollout_save_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_data["num_steps"] = len(rollout_data["actions"])
        rollout_data["dt"] = dt
        torch.save(rollout_data, rollout_save_path)
        print(f"[INFO] Saved rollout to: {rollout_save_path}")

    if ee_traj_buf is not None and ee_traj_save_path is not None:
        ee_traj_save_path.parent.mkdir(parents=True, exist_ok=True)
        ee_traj_tensor = torch.stack(ee_traj_buf, dim=0)
        torch.save(
            {
                "ee_pos_w": ee_traj_tensor,
                "task": args_cli.task,
                "checkpoint": str(resume_path),
                "skill": args_cli.skill,
                "dt": dt,
            },
            ee_traj_save_path,
        )
        print(f"[INFO] Saved EE trajectory to: {ee_traj_save_path}  shape={tuple(ee_traj_tensor.shape)}")

    print(f"Number of episodes: {num_episodes}")
    print(f"Number of successes: {num_successes}")
    if num_episodes:
        print(f"Success rate: {num_successes / num_episodes:.2%}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
