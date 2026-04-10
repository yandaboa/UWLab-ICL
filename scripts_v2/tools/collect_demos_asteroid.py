# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with expert and optional exploration policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import gymnasium as gym
import os
import torch
from tqdm import tqdm
import dill
import hydra
from types import MethodType
from typing import Sequence

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations from trained RL policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr", help="Output dataset path.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use the mean of the policy distribution instead of sampling.",
)
parser.add_argument(
    "--exploration_checkpoint",
    type=str,
    default=None,
    help="Path to exploration policy checkpoint.",
)
parser.add_argument(
    "--min_exploration_horizon",
    type=float,
    default=0.02,
    help="Minimum exploration horizon ratio for exploration policy.",
)
parser.add_argument(
    "--max_exploration_horizon",
    type=float,
    default=0.3,
    help="Maximum exploration horizon ratio for exploration policy.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=5.0,
    help="Episode length in seconds.",
)
parser.add_argument("--render", action="store_true", default=False, help="Render environment while collecting demos.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from uwlab.utils.datasets import ZarrDatasetFileHandler

import uwlab_tasks  # noqa: F401
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerTransformedActionCfg,
)
from uwlab_tasks.utils.hydra import hydra_task_compose
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def process_agent_cfg(env_cfg, agent_cfg):
    if hasattr(agent_cfg.algorithm, "behavior_cloning_cfg"):
        if agent_cfg.algorithm.behavior_cloning_cfg is None:
            del agent_cfg.algorithm.behavior_cloning_cfg
        else:
            bc_cfg = agent_cfg.algorithm.behavior_cloning_cfg
            if bc_cfg.experts_observation_group_cfg is not None:
                import importlib

                # resolve path to the module location
                mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                mod = importlib.import_module(mod_name)
                cfg_cls = mod
                for attr in attr_name.split("."):
                    cfg_cls = getattr(cfg_cls, attr)
                cfg = cfg_cls()
                setattr(env_cfg.observations, "expert_obs", cfg)

    if hasattr(agent_cfg.algorithm, "offline_algorithm_cfg"):
        if agent_cfg.algorithm.offline_algorithm_cfg is None:
            del agent_cfg.algorithm.offline_algorithm_cfg
        else:
            if agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg is None:
                del agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
            else:
                bc_cfg = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
                if bc_cfg.experts_observation_group_cfg is not None:
                    import importlib

                    # resolve path to the module location
                    mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                    mod = importlib.import_module(mod_name)
                    cfg_cls = mod
                    for attr in attr_name.split("."):
                        cfg_cls = getattr(cfg_cls, attr)
                    cfg = cfg_cls()
                    setattr(env_cfg.observations, "expert_obs", cfg)
    return agent_cfg


def record_pre_reset(self, env_ids: Sequence[int] | None, force_export_or_skip=None) -> None:
    """Patch recorder manager to gate saved demos by success and exploration usage."""
    if len(self.active_terms) == 0:
        return

    if env_ids is None:
        env_ids = list(range(self._env.num_envs))
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()

    for term in self._terms.values():
        key, value = term.record_pre_reset(env_ids)
        self.add_to_episodes(key, value, env_ids)

    success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
    if hasattr(self._env, "termination_manager") and "success" in self._env.termination_manager.active_terms:
        success_results |= self._env.termination_manager.get_term("success")[env_ids]

    if hasattr(self, "exploration_lengths"):
        episode_lengths = self._env.episode_length_buf[env_ids]
        exploration_lengths = self.exploration_lengths[env_ids]
        exploration_ratios = exploration_lengths / torch.clamp(episode_lengths, min=1)
        success_results = success_results & (exploration_ratios < 0.95)

    self.set_success_to_episodes(env_ids, success_results)

    if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
        self.export_episodes(env_ids)


def load_exploration_policy(checkpoint_path: str, device: torch.device, num_envs: int) -> DiffusionPolicyWrapper:
    """Load exploration diffusion policy from checkpoint."""
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy = policy.eval().to(device)
    return DiffusionPolicyWrapper(policy, device, n_obs_steps=policy.n_obs_steps, num_envs=num_envs)


def sample_exploration_horizons(
    num_envs: int, min_horizon: int, max_horizon: int, device: torch.device
) -> torch.Tensor:
    """Sample exploration horizon in simulation steps per environment."""
    if max_horizon <= 0:
        return torch.zeros((num_envs,), device=device, dtype=torch.int32)
    min_h = min(max(min_horizon, 0), max_horizon)
    max_h = max(max_horizon, min_h)
    return torch.randint(min_h, max_h + 1, (num_envs,), device=device)


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect demonstrations using expert policy and optional diffusion exploration policy."""
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith(".zarr")
    if use_zarr_format:
        dataset_handler = ZarrDatasetFileHandler
    else:
        dataset_handler = HDF5DatasetFileHandler

    # Setup recorder for actions/observations and expert-vs-exploration mask.
    env_cfg.recorders = ActionStateRecorderManagerTransformedActionCfg()

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.observations.policy.concatenate_terms = False

    # add expert obs into env_cfg
    agent_cfg = process_agent_cfg(env_cfg, agent_cfg)

    episode_length = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.sim.render_interval))
    max_exploration_horizon = int(args_cli.max_exploration_horizon * episode_length)
    min_exploration_horizon = int(args_cli.min_exploration_horizon * episode_length)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load expert
    bc = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
    assert len(bc.experts_path) == 1, "Only one expert is supported for now."
    expert_obs_fn = bc.experts_observation_func
    loader = bc.experts_loader
    if not callable(loader):
        loader = eval(loader)
    device = torch.device(env_cfg.sim.device if isinstance(env_cfg.sim.device, str) else "cuda:0")
    expert_policy = loader(bc.experts_path[0]).to(device)
    expert_policy.eval()

    # optional exploration policy
    num_envs = env.num_envs
    exploration_policy = None
    if args_cli.exploration_checkpoint:
        exploration_policy = load_exploration_policy(args_cli.exploration_checkpoint, device, num_envs)
        reset_ids = torch.arange(num_envs, device=device)
        exploration_policy.reset(reset_ids)
        print(f"[Exploration] Loaded checkpoint: {args_cli.exploration_checkpoint}")
    else:
        print("[Exploration] No checkpoint provided; collecting expert-only demonstrations.")

    print(f"[Expert] {'Deterministic (mean)' if args_cli.deterministic else 'Stochastic (sampled)'} actions")

    recorder_manager = env.unwrapped.recorder_manager
    expert_mask_recorder = recorder_manager._terms.get("record_pre_step_expert_mask")
    if expert_mask_recorder is None:
        raise RuntimeError("record_pre_step_expert_mask recorder term is not configured.")

    exploration_horizons = sample_exploration_horizons(
        num_envs, min_exploration_horizon, max_exploration_horizon, device
    )
    exploration_lengths = torch.zeros((num_envs,), device=device, dtype=torch.int32)
    recorder_manager.exploration_lengths = exploration_lengths
    if exploration_policy is not None:
        recorder_manager.record_pre_reset = MethodType(record_pre_reset, recorder_manager)

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        # Initialize tqdm progress bar if num_demos > 0
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")

        while True:
            # choose expert or exploration policy per environment based on horizon.
            episode_steps = env.unwrapped.episode_length_buf
            use_exploration = (episode_steps < exploration_horizons) & (exploration_policy is not None)
            use_expert = ~use_exploration
            exploration_lengths += use_exploration.int()
            recorder_manager.exploration_lengths = exploration_lengths

            expert_policy_obs = expert_obs_fn(env)
            mean, std = expert_policy.compute_distribution(expert_policy_obs)
            actions = torch.zeros((num_envs, env.action_space.shape[-1]), device=device)
            if use_expert.any():
                expert_actions = mean if args_cli.deterministic else torch.normal(mean, std)
                actions[use_expert] = expert_actions[use_expert]
            if use_exploration.any() and exploration_policy is not None:
                exploration_actions = exploration_policy.predict_action(obs_dict).to(device)
                actions[use_exploration] = exploration_actions[use_exploration]

            # Mask actions to zero for environments in their first step after reset since first image may not be valid
            first_step_mask = env.unwrapped.episode_length_buf == 0
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            expert_mask = use_expert.unsqueeze(-1)
            expert_mask_recorder.set_mask(expert_mask)

            # Inject expert distribution into obs_buf so recorder saves them alongside observations
            env.unwrapped.obs_buf["data_collection"]["expert_action_mean"] = mean.clone()
            env.unwrapped.obs_buf["data_collection"]["expert_action_std"] = std.clone()

            # env stepping
            env.step(actions)

            if env.unwrapped.reset_buf.any():
                reset_ids = env.unwrapped.reset_buf.nonzero(as_tuple=False).reshape(-1)
                exploration_horizons[reset_ids] = sample_exploration_horizons(
                    len(reset_ids), min_exploration_horizon, max_exploration_horizon, device
                )
                exploration_lengths[reset_ids] = 0
                if exploration_policy is not None:
                    exploration_policy.reset(reset_ids)

            # print out the current demo count if it has changed
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(increment)

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            if args_cli.render:
                env.render()

        pbar.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
