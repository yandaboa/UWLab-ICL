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

# NOTE: Pre-import numpy and numba BEFORE isaaclab / AppLauncher. Isaac Sim's Kit runtime
# mutates sys.path at startup, which causes any later `import numba` to resolve to
# /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/numba (0.59.x, incompatible with
# numpy 2.x) instead of the conda env's pinned numba 0.64. By importing them here while
# PYTHONPATH ordering still holds, sys.modules caches the correct versions and every
# subsequent `import numba` (e.g. via diffusion_policy.common.sampler when Hydra loads
# TrainMLPImageWorkspace for the learner checkpoint) returns the cached conda module.
import numpy  # noqa: F401
import numba  # noqa: F401

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
parser.add_argument(
    "--disable_exploration_ratio_filter",
    action="store_true",
    default=False,
    help=(
        "If set, disable the check that rejects demos where the learner/exploration policy drove"
        " more than 95%% of the episode. ON by default; pass this flag only for tasks where you"
        " want to keep learner-dominated demos (e.g. pure imitation from the exploration policy)."
    ),
)
parser.add_argument(
    "--disable_task_success_filter",
    action="store_true",
    default=False,
    help=(
        "If set, admit every completed episode (regardless of the task-defined `success`"
        " termination) as long as it passes the exploration-ratio filter. REQUIRES"
        " --disable_exploration_ratio_filter to NOT be set, otherwise there is no filter left"
        " and we'd admit literally every episode; an assert enforces this."
    ),
)
parser.add_argument(
    "--transformer_mini_batch_size",
    type=int,
    default=64,
    help=(
        "Mini-batch size used by DiffusionPolicyWrapper when serializing transformer inference"
        " across envs. Bounds peak activation memory; too-small values (e.g. 8) dominate wall"
        " time for large num_envs."
    ),
)
parser.add_argument(
    "--no_kv_cache",
    action="store_true",
    default=False,
    help=(
        "Disable the incremental KV-cached inference path inside DiffusionPolicyWrapper"
        " (falls back to re-encoding the full trajectory each step). Useful for A/B profiling."
    ),
)
parser.add_argument(
    "--kv_cache_max_seq_len",
    type=int,
    default=None,
    help=(
        "Upper bound on per-env KV cache length. Defaults to the transformer's n_positions"
        " (typically 1024). Lower it to reduce preallocated cache memory."
    ),
)
parser.add_argument(
    "--use_inverse_actions",
    action="store_true",
    default=False,
    help=(
        "If set, compute the analytically optimal action for the augmented environment: "
        "expert_action is mapped through the arm OSC term's inverse_process_actions so that "
        "feeding the result to the perturbed env produces the same physical effect as the expert's "
        "intended action. This allows collecting high-quality demos even when the expert was "
        "trained on a non-augmented MDP."
    ),
)
parser.add_argument(
    "--num_bins",
    type=int,
    default=0,
    help=(
        "If > 0, discretize the 6 continuous arm action dims into this many uniform bins over "
        "[--discretize_clip_val, +--discretize_clip_val] before saving and executing actions. "
        "Gripper dim (index 6) is always sign-thresholded to {-1, +1}. "
        "0 disables discretization. A discretize_spec.json is written alongside the dataset."
    ),
)
parser.add_argument(
    "--discretize_clip_val",
    type=float,
    default=2.0,
    help="Symmetric clip range for binning continuous arm action dims. Default: 2.0.",
)
parser.add_argument(
    "--expert_action_scale",
    type=float,
    nargs=6,
    default=[0.01, 0.01, 0.002, 0.02, 0.02, 0.2],
    metavar=("sx", "sy", "sz", "rx", "ry", "rz"),
    help=(
        "Six-element action scale the expert was trained with (XYZ + axis-angle). "
        "Used as the 'original_scale' for inverse_process_actions when --use_inverse_actions is set. "
        "Default: [0.01, 0.01, 0.002, 0.02, 0.02, 0.2]."
    ),
)
parser.add_argument(
    "--full_dagger",
    action="store_true",
    default=False,
    help=(
        "Full DAgger mode: the exploration (student) policy drives EVERY env for the full episode,"
        " but the recorded ``actions`` field is overridden with the inverse-mapped (and possibly"
        " discretized) expert action — the expert provides supervision at every step regardless"
        " of who physically acted. Requires --exploration_checkpoint and is incompatible with"
        " --disable_exploration_ratio_filter (the ratio is degenerate in this mode and the filter"
        " is force-disabled internally)."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

if args_cli.disable_task_success_filter and args_cli.disable_exploration_ratio_filter:
    raise SystemExit(
        "--disable_task_success_filter requires the exploration-ratio filter to stay ON."
        " Remove --disable_exploration_ratio_filter, or drop --disable_task_success_filter."
    )

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
from uwlab_rl.utils.action_discretize import discretize_actions, make_discretize_spec, save_discretize_spec
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
    """Patch recorder manager to gate saved demos by success and exploration usage.

    Admission uses two independent gates whose enable-state is read off the recorder:
    (1) ``apply_exploration_ratio_filter`` — reject demos where the exploration policy
    drove >=95% of the successful episode. (2) ``disable_task_success_filter`` — if set,
    admit episodes that end in any reason as long as the ratio gate passes. If both
    gates are disabled the configuration is nonsensical (no quality filter remains) and
    the CLI enforces the invariant up front.
    """
    if len(self.active_terms) == 0:
        return

    if env_ids is None:
        env_ids = list(range(self._env.num_envs))
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()

    for term in self._terms.values():
        key, value = term.record_pre_reset(env_ids)
        self.add_to_episodes(key, value, env_ids)

    device = self._env.device
    n = len(env_ids)

    task_success = torch.zeros(n, dtype=bool, device=device)
    if hasattr(self._env, "termination_manager") and "success" in self._env.termination_manager.active_terms:
        task_success |= self._env.termination_manager.get_term("success")[env_ids]

    ratio_pass = torch.ones(n, dtype=bool, device=device)
    if hasattr(self, "exploration_lengths"):
        episode_lengths = self._env.episode_length_buf[env_ids]
        exploration_lengths = self.exploration_lengths[env_ids]
        # Per-env fraction of steps driven by the exploration policy; reject runs
        # where the learner dominated (>=95%) of the successful episode.
        exploration_ratios = exploration_lengths / torch.clamp(episode_lengths, min=1)
        ratio_pass = exploration_ratios < 0.95

    task_gate = (
        torch.ones(n, dtype=bool, device=device)
        if getattr(self, "disable_task_success_filter", False)
        else task_success
    )
    ratio_gate = (
        ratio_pass
        if getattr(self, "apply_exploration_ratio_filter", True)
        else torch.ones(n, dtype=bool, device=device)
    )
    success_results = task_gate & ratio_gate

    self.set_success_to_episodes(env_ids, success_results)

    if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
        self.export_episodes(env_ids)


def load_exploration_policy(
    checkpoint_path: str,
    device: torch.device,
    num_envs: int,
    mini_batch_size: int = 64,
    use_kv_cache: bool = True,
    kv_cache_max_seq_len: int | None = None,
) -> DiffusionPolicyWrapper:
    """Load exploration diffusion policy from checkpoint (stochastic inference for data collection)."""
    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy = policy.eval().to(device)
    return DiffusionPolicyWrapper(
        policy,
        device,
        n_obs_steps=policy.n_obs_steps,
        num_envs=num_envs,
        mini_batch_size=mini_batch_size,
        use_kv_cache=use_kv_cache,
        kv_cache_max_seq_len=kv_cache_max_seq_len,
        profile_name="exploration",
        sample_action=True,
    )


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

    episode_length = int(env_cfg.episode_length_s / env_cfg.decimation * env_cfg.sim.dt)
    max_exploration_horizon = int(args_cli.max_exploration_horizon * episode_length)
    min_exploration_horizon = int(args_cli.min_exploration_horizon * episode_length)
    print(f"Episode length: {episode_length}, Max exploration horizon: {max_exploration_horizon}, Min exploration horizon: {min_exploration_horizon}")


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

    # Inverse-action + discretization setup
    use_inverse = args_cli.use_inverse_actions
    num_bins = args_cli.num_bins
    disc_clip = args_cli.discretize_clip_val
    arm_term = env.unwrapped.action_manager._terms.get("arm")

    if use_inverse and arm_term is None:
        raise RuntimeError("Could not find 'arm' action term in environment — cannot compute inverse actions.")

    # ``original_scale`` for inverse_process_actions — must be supplied via CLI.
    expert_original_scale = torch.tensor(
        args_cli.expert_action_scale, device=device, dtype=torch.float32
    )

    if use_inverse:
        print(f"[InverseAction] ENABLED — expert original scale: {expert_original_scale.tolist()}")
    if num_bins > 0:
        disc_spec = make_discretize_spec(num_bins, disc_clip, action_dim=env.action_space.shape[-1])
        spec_path = save_discretize_spec(disc_spec, output_dir)
        print(f"[Discretize] ENABLED — {num_bins} bins, clip_val={disc_clip}")
        print(f"[Discretize] Spec saved → {spec_path}")
    else:
        disc_spec = None
        print("[Discretize] DISABLED (continuous actions saved)")

    # Per-env buffer tracking the expert's intended action (in expert-scale space).
    # Patched into obs_buf["expert_obs"]["prev_actions"] each step so the expert
    # sees its own previous action rather than the discretized/perturbed env action.
    num_envs_val = env.num_envs
    action_dim_val = env.action_space.shape[-1]
    prev_expert_actions = torch.zeros((num_envs_val, action_dim_val), device=device)

    # optional exploration policy
    num_envs = env.num_envs
    exploration_policy = None
    if args_cli.exploration_checkpoint:
        exploration_policy = load_exploration_policy(
            args_cli.exploration_checkpoint,
            device,
            num_envs,
            mini_batch_size=args_cli.transformer_mini_batch_size,
            use_kv_cache=not args_cli.no_kv_cache,
            kv_cache_max_seq_len=args_cli.kv_cache_max_seq_len,
        )
        reset_ids = torch.arange(num_envs, device=device)
        exploration_policy.reset(reset_ids)
        print(f"[Exploration] Loaded checkpoint: {args_cli.exploration_checkpoint}")
        print(
            f"[Exploration] KV cache {'ENABLED' if not args_cli.no_kv_cache else 'DISABLED'} "
            f"(max_seq_len={args_cli.kv_cache_max_seq_len}, "
            f"mini_batch={args_cli.transformer_mini_batch_size})"
        )
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

    # Full-DAgger setup: student drives every env, but the recorded action is the inverse-mapped
    # expert supervision. Implemented by stashing per-step expert supervision on the env and
    # monkey-patching the actions recorder to read from there instead of the action manager.
    full_dagger = args_cli.full_dagger
    if full_dagger:
        if exploration_policy is None:
            raise SystemExit("--full_dagger requires --exploration_checkpoint to drive every env.")
        if not use_inverse:
            raise SystemExit(
                "--full_dagger requires --use_inverse_actions so the expert supervision is correct"
                " under the perturbation."
            )
        # Both filters off in full DAgger:
        #   - exploration-ratio: degenerate (student drives 100% by construction).
        #   - task-success: the env's consecutive-success termination is too strict to fire under
        #     a moderately-trained student even when EOE/any-time success is high. The expert
        #     label is independent of student outcome, so admitting all completed episodes keeps
        #     the supervision correct; abnormal-robot terminations still cull crashed episodes.
        args_cli.disable_exploration_ratio_filter = True
        args_cli.disable_task_success_filter = True
        actions_recorder_term = recorder_manager._terms.get("record_pre_step_actions")
        if actions_recorder_term is None:
            raise RuntimeError("record_pre_step_actions recorder term is not configured.")
        env.unwrapped.expert_supervision_action = torch.zeros(
            (num_envs, action_dim_val), device=device
        )

        def _record_expert_supervision(self):
            return "actions", self._env.expert_supervision_action

        actions_recorder_term.record_pre_step = MethodType(
            _record_expert_supervision, actions_recorder_term
        )
        print("[FullDAgger] ENABLED — student drives every env; recorded action = inverse-mapped expert.")

    # Install gated record_pre_reset + filter config regardless of exploration, so that
    # --disable_task_success_filter (expert-only admit-all) works even when no exploration
    # policy is loaded.
    recorder_manager.record_pre_reset = MethodType(record_pre_reset, recorder_manager)
    recorder_manager.apply_exploration_ratio_filter = not args_cli.disable_exploration_ratio_filter
    recorder_manager.disable_task_success_filter = args_cli.disable_task_success_filter
    if args_cli.disable_exploration_ratio_filter:
        print("[Filter] exploration-ratio < 0.95 filter is DISABLED by CLI flag.")
    else:
        print("[Filter] exploration-ratio < 0.95 filter is ENABLED (default).")
    if args_cli.disable_task_success_filter:
        print(
            "[Filter] task-success filter is DISABLED — admitting ALL episodes that pass the"
            " exploration-ratio filter."
        )
    else:
        print("[Filter] task-success filter is ENABLED (default) — only successful episodes saved.")

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    num_episodes = 0
    num_successes = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        # Initialize tqdm progress bar if num_demos > 0
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations (Success: 0.00%)", unit="demo")

        while True:
            # ── (1) Patch expert obs: replace prev_actions with what expert *thinks* it executed ──
            # Isaac Lab's obs_buf["expert_obs"]["prev_actions"] reflects the last *env* action
            # (which may be discretized/perturbed). We override it with prev_expert_actions so
            # the expert policy stays in-distribution across its own action-feedback loop.
            if use_inverse and isinstance(env.unwrapped.obs_buf.get("expert_obs"), dict):
                env.unwrapped.obs_buf["expert_obs"]["prev_actions"] = prev_expert_actions.clone()

            # ── (2) Choose expert or exploration policy per environment ──
            episode_steps = env.unwrapped.episode_length_buf
            if full_dagger:
                # Student drives every env every step; the inverse-mapped expert action is recorded
                # as supervision (see step 6 below).
                use_exploration = torch.ones((num_envs,), dtype=torch.bool, device=device)
            else:
                use_exploration = (episode_steps < exploration_horizons) & (exploration_policy is not None)
            use_expert = ~use_exploration
            exploration_lengths += use_exploration.int()
            recorder_manager.exploration_lengths = exploration_lengths

            # ── (3) Query expert: get intended actions in expert's unaugmented action space ──
            expert_policy_obs = expert_obs_fn(env)
            mean, std = expert_policy.compute_distribution(expert_policy_obs)
            # mean shape: (num_envs, action_dim)
            expert_mean = mean
            # expert_mean = mean if args_cli.deterministic else torch.normal(mean, std)

            # ── (4) Compute env actions: apply inverse OSC mapping so the augmented env
            #        produces the same physical effect as the expert's intended action ──
            if use_inverse:
                # inverse_process_actions modifies in-place → clone first
                env_actions = arm_term.inverse_process_actions(
                    expert_mean.clone(), original_scale=expert_original_scale
                )
            else:
                env_actions = expert_mean.clone()

            # ── (5) Discretize if requested ──
            if num_bins > 0:
                env_actions = discretize_actions(env_actions, num_bins, disc_clip)

            # ── (6) Build final action tensor (expert for expert envs, exploration for others) ──
            actions = torch.zeros((num_envs, action_dim_val), device=device)
            if use_expert.any():
                actions[use_expert] = env_actions[use_expert]
            if use_exploration.any() and exploration_policy is not None:
                exploration_env_ids = use_exploration.nonzero(as_tuple=False).reshape(-1)
                obs_buf = env.unwrapped.obs_buf
                policy_obs = obs_buf.get("policy", obs_buf) if isinstance(obs_buf, dict) else obs_buf
                exploration_obs = {k: v[use_exploration] for k, v in policy_obs.items()}
                exploration_actions = exploration_policy.predict_action(exploration_obs, exploration_env_ids)
                actions[use_exploration] = exploration_actions.to(device)

            # In full-DAgger the recorded ``actions`` field is the inverse-mapped expert action,
            # not the student action that physically drives the env. Stash it on the env so the
            # monkey-patched ``record_pre_step_actions`` term reads it instead of action_manager.
            if full_dagger:
                env.unwrapped.expert_supervision_action = env_actions.clone()

            # Mask actions to zero on the first step after reset (image may not yet be valid)
            first_step_mask = env.unwrapped.episode_length_buf == 0
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # keep gripper closed

            expert_mask = use_expert.unsqueeze(-1)
            expert_mask_recorder.set_mask(expert_mask)

            # Inject expert distribution into obs_buf so recorder saves them alongside observations
            env.unwrapped.obs_buf["data_collection"]["expert_action_mean"] = mean.clone()
            env.unwrapped.obs_buf["data_collection"]["expert_action_std"] = std.clone()

            # ── (7) Step the environment with the (possibly discretized/inverse) action ──
            _obs, rewards, dones, _infos = env.step(actions)
            if dones.any():
                num_episodes += dones.sum().item()
                num_successes += torch.logical_and(rewards > 0.1, dones).sum().item()

            # ── (8) Update prev_expert_actions for ALL envs.
            #        Expert envs: track expert_mean (what expert intended).
            #        Exploration envs: also store expert_mean so the expert stays in-distribution
            #        when it resumes on those envs next iteration.
            prev_expert_actions = expert_mean.clone()

            if env.unwrapped.reset_buf.any():
                reset_ids = env.unwrapped.reset_buf.nonzero(as_tuple=False).reshape(-1)
                exploration_horizons[reset_ids] = sample_exploration_horizons(
                    len(reset_ids), min_exploration_horizon, max_exploration_horizon, device
                )
                exploration_lengths[reset_ids] = 0
                # Reset the expert-action history for reset envs so the policy sees a clean start
                prev_expert_actions[reset_ids] = 0.0
                if exploration_policy is not None:
                    exploration_policy.reset(reset_ids)

            # print out the current demo count if it has changed
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(increment)
                rate = (num_successes / num_episodes * 100) if num_episodes > 0 else 0.0
                pbar.set_description(f"Recording Demonstrations (Success: {rate:.2f}%)")

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            if args_cli.render:
                env.render()

        pbar.close()

    print(f"Number of episodes: {num_episodes}")
    print(f"Number of successes: {num_successes}")
    if num_episodes:
        print(f"Success rate: {num_successes / num_episodes:.2%}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
