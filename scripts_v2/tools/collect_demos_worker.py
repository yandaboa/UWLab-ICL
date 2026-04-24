# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-lived data-collection worker that keeps Isaac Sim alive across DAgger iterations.

The worker boots Isaac Sim once, loads the expert policy and (optionally) the task
environment's exploration policy, and then waits for "collect" jobs over a
``multiprocessing.connection`` socket. Each job specifies where to write the zarr
dataset, how many demos to record, the exploration-horizon schedule, the episode
length, and (optionally) a new exploration checkpoint to swap in. When the job
finishes, the worker reports back and waits for the next one.

This is meant to be driven by ``run_incontext_exploration.py`` -- see that file for
the orchestration loop.
"""

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
import math
import os
import time
import traceback
from multiprocessing.connection import Client

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Persistent demo-collection worker for Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max_episode_length_s",
    type=float,
    default=30.0,
    help=(
        "Maximum episode length in seconds used to construct the env. Per-job episode lengths must be"
        " <= this value; episodes are truncated manually to the requested length."
    ),
)
parser.add_argument("--socket_path", type=str, required=True, help="Unix socket path to connect back to the orchestrator.")
parser.add_argument("--auth_key", type=str, default="dagger", help="Authentication key for the control socket.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use the mean of the expert policy distribution instead of sampling.",
)
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
        " termination) as long as it passes the exploration-ratio filter. Useful when the"
        " exploration policy itself produces good trajectories that the task success condition"
        " might miss. REQUIRES --disable_exploration_ratio_filter to NOT be set, otherwise there"
        " is no filter left and we'd admit literally every episode; an assert enforces this."
    ),
)
parser.add_argument("--seed", type=int, default=0, help="Base random seed for the env.")
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
        " (falls back to re-encoding the full trajectory each step). Useful for A/B"
        " profiling; normally you want KV caching on."
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dill  # noqa: E402
import hydra  # noqa: E402
import torch  # noqa: E402
from types import MethodType  # noqa: E402
from typing import Sequence  # noqa: E402
from tqdm import tqdm  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

from uwlab.utils.datasets import ZarrDatasetFileHandler  # noqa: E402

import uwlab_tasks  # noqa: F401, E402
from collect_demos_logging import CollectionProgressLogger, log_swap_recorder  # noqa: E402

# Key written by the task's curriculum/success-monitor events into ``env.extras["log"]``.
# Same signal the curriculum thresholds off of (see
# ``uwlab_tasks.manager_based.manipulation.omnireset.mdp.events`` — ``curriculum_success_log_key``).
SUCCESS_RATE_LOG_KEY = "Metrics/task_command/end_of_episode_success_rate"
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (  # noqa: E402
    ActionStateRecorderManagerTransformedActionCfg,
)
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from diffusion_policy.policy.base_image_policy import BaseImagePolicy  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helpers copied from collect_demos_asteroid.py
# ---------------------------------------------------------------------------


def process_agent_cfg(env_cfg, agent_cfg):
    if hasattr(agent_cfg.algorithm, "behavior_cloning_cfg"):
        if agent_cfg.algorithm.behavior_cloning_cfg is None:
            del agent_cfg.algorithm.behavior_cloning_cfg
        else:
            bc_cfg = agent_cfg.algorithm.behavior_cloning_cfg
            if bc_cfg.experts_observation_group_cfg is not None:
                import importlib

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

    # Raw signals are computed regardless of which filters are enabled so the
    # 2x2 task_success × ratio_pass cause table reflects what *each* gate would
    # have done. Admission still honors the enabled-filter flags.
    device = self._env.device
    n = len(env_ids)

    task_success = torch.zeros(n, dtype=bool, device=device)
    if hasattr(self._env, "termination_manager") and "success" in self._env.termination_manager.active_terms:
        task_success |= self._env.termination_manager.get_term("success")[env_ids]

    episode_lengths = self._env.episode_length_buf[env_ids]
    ratio_pass = torch.ones(n, dtype=bool, device=device)
    if hasattr(self, "exploration_lengths"):
        exploration_lengths = self.exploration_lengths[env_ids]
        exploration_ratios = exploration_lengths / torch.clamp(episode_lengths, min=1)
        ratio_pass = exploration_ratios < 0.95

    task_gate = (
        torch.ones(n, dtype=bool, device=device)
        if getattr(self, "disable_task_success_filter", False)
        else task_success
    )
    ratio_gate = (
        ratio_pass
        if getattr(self, "apply_exploration_ratio_filter", False)
        else torch.ones(n, dtype=bool, device=device)
    )
    success_results = task_gate & ratio_gate

    stats = getattr(self, "_filter_stats", None)
    if stats is not None:
        stats["ts_pass_ratio_pass"] += int((task_success & ratio_pass).sum().item())
        stats["ts_pass_ratio_fail"] += int((task_success & ~ratio_pass).sum().item())
        stats["ts_fail_ratio_pass"] += int((~task_success & ratio_pass).sum().item())
        stats["ts_fail_ratio_fail"] += int((~task_success & ~ratio_pass).sum().item())
        stats["total_resets"] += n
        stats["admitted"] += int(success_results.sum().item())

        # Per-termination-term firing counts and mean episode length at reset.
        # Terms are not mutually exclusive (e.g. both ``success`` and ``time_out``
        # can fire on the same step), so sums don't have to equal ``total_resets``.
        if hasattr(self._env, "termination_manager"):
            tm = self._env.termination_manager
            term_counts = stats.setdefault("term_counts", {})
            term_step_sums = stats.setdefault("term_step_sums", {})
            for term_name in tm.active_terms:
                term_mask = tm.get_term(term_name)[env_ids]
                if term_mask.any():
                    term_counts[term_name] = term_counts.get(term_name, 0) + int(term_mask.sum().item())
                    term_step_sums[term_name] = term_step_sums.get(term_name, 0) + int(
                        episode_lengths[term_mask].sum().item()
                    )

        if success_results.any():
            stats["admitted_step_sum"] = stats.get("admitted_step_sum", 0) + int(
                episode_lengths[success_results].sum().item()
            )

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
    )


def sample_exploration_horizons(
    num_envs: int, min_horizon: int, max_horizon: int, device: torch.device
) -> torch.Tensor:
    if max_horizon <= 0:
        return torch.zeros((num_envs,), device=device, dtype=torch.int32)
    min_h = min(max(min_horizon, 0), max_horizon)
    max_h = max(max_horizon, min_h)
    return torch.randint(min_h, max_h + 1, (num_envs,), device=device)


# ---------------------------------------------------------------------------
# Worker implementation
# ---------------------------------------------------------------------------


class CollectionSession:
    """Holds long-lived state for the demo-collection worker."""

    def __init__(
        self,
        env,
        env_cfg,
        agent_cfg,
        device,
        max_episode_length: int,
        deterministic: bool,
        apply_exploration_ratio_filter: bool = True,
        disable_task_success_filter: bool = False,
        transformer_mini_batch_size: int = 64,
        use_kv_cache: bool = True,
        kv_cache_max_seq_len: int | None = None,
    ):
        assert not (disable_task_success_filter and not apply_exploration_ratio_filter), (
            "disable_task_success_filter=True requires apply_exploration_ratio_filter=True."
            " Otherwise every completed episode would be admitted unfiltered — refusing to"
            " produce a dataset with no quality gate."
        )

        self.env = env
        self.env_cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.device = device
        self.max_episode_length = max_episode_length
        self.deterministic = deterministic
        self.apply_exploration_ratio_filter = apply_exploration_ratio_filter
        self.disable_task_success_filter = disable_task_success_filter
        self.transformer_mini_batch_size = transformer_mini_batch_size
        self.use_kv_cache = use_kv_cache
        self.kv_cache_max_seq_len = kv_cache_max_seq_len

        bc = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
        assert len(bc.experts_path) == 1, "Only one expert is supported for now."
        self.expert_obs_fn = bc.experts_observation_func
        loader = bc.experts_loader
        if not callable(loader):
            loader = eval(loader)
        print(f"[worker] loading expert policy from {bc.experts_path[0]}...", flush=True)
        _t0 = time.time()
        expert_policy = loader(bc.experts_path[0]).to(device)
        expert_policy.eval()
        self.expert_policy = expert_policy
        print(f"[worker] expert loaded in {time.time() - _t0:.1f}s", flush=True)

        # Exploration policy cache: path -> DiffusionPolicyWrapper
        self._exploration_cache: dict[str, DiffusionPolicyWrapper] = {}
        self._current_exploration_path: str | None = None
        self._current_exploration_policy: DiffusionPolicyWrapper | None = None

        recorder_manager = env.unwrapped.recorder_manager
        expert_mask_recorder = recorder_manager._terms.get("record_pre_step_expert_mask")
        if expert_mask_recorder is None:
            raise RuntimeError("record_pre_step_expert_mask recorder term is not configured.")
        self.recorder_manager = recorder_manager
        self.expert_mask_recorder = expert_mask_recorder

        # Install the gated record_pre_reset (safe to install once; it no-ops if
        # exploration_lengths is absent on the recorder).
        recorder_manager.record_pre_reset = MethodType(record_pre_reset, recorder_manager)
        recorder_manager.apply_exploration_ratio_filter = apply_exploration_ratio_filter
        recorder_manager.disable_task_success_filter = disable_task_success_filter

        # Job counter so logs clearly distinguish iteration-1 / iteration-2 / ...
        # collections (useful for diagnosing per-iteration slowdowns).
        self.job_counter = 0

    def _get_exploration_policy(self, checkpoint_path: str | None) -> DiffusionPolicyWrapper | None:
        if checkpoint_path is None:
            self._current_exploration_path = None
            self._current_exploration_policy = None
            return None
        if checkpoint_path == self._current_exploration_path and self._current_exploration_policy is not None:
            return self._current_exploration_policy
        if checkpoint_path in self._exploration_cache:
            policy = self._exploration_cache[checkpoint_path]
        else:
            print(f"[worker] loading exploration checkpoint: {checkpoint_path}", flush=True)
            policy = load_exploration_policy(
                checkpoint_path,
                self.device,
                self.env.num_envs,
                mini_batch_size=self.transformer_mini_batch_size,
                use_kv_cache=self.use_kv_cache,
                kv_cache_max_seq_len=self.kv_cache_max_seq_len,
            )
            self._exploration_cache[checkpoint_path] = policy
        reset_ids = torch.arange(self.env.num_envs, device=self.device)
        policy.reset(reset_ids)
        self._current_exploration_path = checkpoint_path
        self._current_exploration_policy = policy
        return policy

    def _swap_recorder_output(self, dataset_file: str):
        """Close the current dataset file and open a new one at the given path.

        Also clears per-env success counters and episode buffers so counts are
        scoped to the new job.
        """
        _t0 = time.time()
        rm = self.recorder_manager
        # Close previous dataset file handler if any.
        if getattr(rm, "_dataset_file_handler", None) is not None:
            try:
                rm._dataset_file_handler.close()
            except Exception:
                pass

        # Create a fresh handler for the new file.
        output_dir = os.path.dirname(dataset_file)
        output_file_name = os.path.basename(dataset_file)
        os.makedirs(output_dir, exist_ok=True)

        # Update cfg so later code (e.g. close()) uses the new path.
        rm.cfg.dataset_export_dir_path = output_dir
        rm.cfg.dataset_filename = output_file_name

        handler = rm.cfg.dataset_file_handler_class_type()
        env_name = getattr(self.env.unwrapped.cfg, "env_name", None)
        handler.create(os.path.join(output_dir, output_file_name), env_name=env_name)
        rm._dataset_file_handler = handler

        # Reset counters/buffers.
        rm._exported_successful_episode_count = {}
        rm._exported_failed_episode_count = {}
        rm._filter_stats = {
            "ts_pass_ratio_pass": 0,
            "ts_pass_ratio_fail": 0,
            "ts_fail_ratio_pass": 0,
            "ts_fail_ratio_fail": 0,
            "total_resets": 0,
            "admitted": 0,
            "admitted_step_sum": 0,
            "term_counts": {},
            "term_step_sums": {},
        }
        from isaaclab.managers.recorder_manager import EpisodeData  # local import to avoid top-level dep
        for env_id in range(self.env.num_envs):
            rm._episodes[env_id] = EpisodeData()
        log_swap_recorder(dataset_file, time.time() - _t0)

    def collect(
        self,
        dataset_file: str,
        num_demos: int,
        min_exploration_horizon: float,
        max_exploration_horizon: float,
        episode_length_s: float,
        exploration_checkpoint: str | None,
        seed: int,
    ) -> dict:
        """Run a single data-collection job and return result metadata."""
        env = self.env
        num_envs = env.num_envs
        device = self.device

        self.job_counter += 1
        job_num = self.job_counter

        # Reconfigure recorder output for this job.
        self._swap_recorder_output(dataset_file)

        # Per-job episode length (steps). Source: Isaac Lab's
        # ``ManagerBasedEnv.step_dt = sim.dt * decimation`` — use the live env
        # as the source of truth. (render_interval only gates the renderer
        # inside the decimation loop and must NOT be used here.)
        step_dt = env.unwrapped.step_dt
        episode_length_steps = math.ceil(episode_length_s / step_dt)
        if episode_length_steps > self.max_episode_length:
            raise RuntimeError(
                f"Requested episode_length_s={episode_length_s} (→{episode_length_steps} steps) exceeds worker max"
                f" of {self.max_episode_length} steps. Restart the worker with a larger --max_episode_length_s."
            )

        max_exploration_horizon_steps = int(max_exploration_horizon * episode_length_steps)
        min_exploration_horizon_steps = int(min_exploration_horizon * episode_length_steps)

        logger = CollectionProgressLogger(
            job_num=job_num,
            num_envs=num_envs,
            num_demos=num_demos,
            episode_length_s=episode_length_s,
            episode_length_steps=episode_length_steps,
            step_dt=step_dt,
            dataset_file=dataset_file,
            min_exploration_horizon=min_exploration_horizon,
            max_exploration_horizon=max_exploration_horizon,
            min_exploration_horizon_steps=min_exploration_horizon_steps,
            max_exploration_horizon_steps=max_exploration_horizon_steps,
        )
        logger.log_start()

        _t0 = time.time()
        exploration_policy = self._get_exploration_policy(exploration_checkpoint)
        logger.log_event("exploration policy ready", time.time() - _t0, extra=f"(ckpt={exploration_checkpoint})")

        # Reset exploration bookkeeping.
        exploration_horizons = sample_exploration_horizons(
            num_envs, min_exploration_horizon_steps, max_exploration_horizon_steps, device
        )
        exploration_lengths = torch.zeros((num_envs,), device=device, dtype=torch.int32)
        self.recorder_manager.exploration_lengths = exploration_lengths

        current_recorded_demo_count = 0
        deterministic = self.deterministic

        # NOTE: env.reset() must run inside inference_mode. After the first rollout,
        # Isaac Lab's PhysX-backed buffers (e.g. ``self._data.root_link_pose_w``) become
        # inference tensors and cannot be written to outside inference_mode on subsequent
        # jobs. The rollout's internal ``_reset_idx`` already runs inside inference_mode
        # for the same reason; we extend the context to cover the per-job reset here.
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            # Reset all envs to make sure the recorder starts cleanly for this job.
            _t_reset0 = time.time()
            env.reset()
            if exploration_policy is not None:
                exploration_policy.reset(torch.arange(num_envs, device=device))
            logger.log_event("initial env.reset()", time.time() - _t_reset0)

            pbar = tqdm(
                total=num_demos,
                desc=f"job#{job_num} → {os.path.basename(dataset_file)}",
                unit="demo",
                dynamic_ncols=True,
            )
            logger.on_loop_start()

            while True:
                with logger.timed("expert"):
                    episode_steps = env.unwrapped.episode_length_buf
                    use_exploration = (episode_steps < exploration_horizons) & (exploration_policy is not None)
                    use_expert = ~use_exploration
                    exploration_lengths += use_exploration.int()
                    self.recorder_manager.exploration_lengths = exploration_lengths

                    expert_policy_obs = self.expert_obs_fn(env)
                    mean, std = self.expert_policy.compute_distribution(expert_policy_obs)
                    actions = torch.zeros((num_envs, env.action_space.shape[-1]), device=device)
                    if use_expert.any():
                        expert_actions = mean if deterministic else torch.normal(mean, std)
                        actions[use_expert] = expert_actions[use_expert]

                with logger.timed("explore"):
                    if use_exploration.any() and exploration_policy is not None:
                        # Match OctiLab collect_demos.py convention: only feed obs for envs actually
                        # running exploration (so transformer per-env trajectories grow only on those
                        # steps) and pass their absolute env ids alongside.
                        exploration_env_ids = use_exploration.nonzero(as_tuple=False).reshape(-1)
                        obs_dict = env.unwrapped.obs_buf
                        policy_obs = obs_dict.get("policy", obs_dict) if isinstance(obs_dict, dict) else obs_dict
                        exploration_obs = {k: v[use_exploration] for k, v in policy_obs.items()}
                        exploration_actions = exploration_policy.predict_action(exploration_obs, exploration_env_ids)
                        exploration_actions = exploration_actions.to(device)
                        actions[use_exploration] = exploration_actions

                # Zero actions on the first step after a reset (first image may not be valid).
                first_step_mask = env.unwrapped.episode_length_buf == 0
                if torch.any(first_step_mask):
                    actions[first_step_mask, :-1] = 0.0
                    actions[first_step_mask, -1] = -1.0  # close gripper

                expert_mask = use_expert.unsqueeze(-1)
                self.expert_mask_recorder.set_mask(expert_mask)

                # Inject expert distribution into obs_buf so recorder saves them alongside observations.
                env.unwrapped.obs_buf["data_collection"]["expert_action_mean"] = mean.clone()
                env.unwrapped.obs_buf["data_collection"]["expert_action_std"] = std.clone()

                with logger.timed("step"):
                    env.step(actions)

                with logger.timed("reset"):
                    natural_reset = env.unwrapped.reset_buf.clone().bool()
                    too_long = env.unwrapped.episode_length_buf >= episode_length_steps
                    manual_truncate = too_long & ~natural_reset
                    if manual_truncate.any():
                        assert False, "Manual truncation should not be happening during this debug."
                        truncate_ids = manual_truncate.nonzero(as_tuple=False).reshape(-1)
                        # Log per-job timeout truncations explicitly. These do NOT necessarily
                        # correspond to Isaac Lab's built-in `time_out` termination term because
                        # the env itself was constructed with a longer max episode length
                        # (`--max_episode_length_s`) to allow per-job episode_length_s.
                        stats = getattr(env.unwrapped.recorder_manager, "_filter_stats", None)
                        if isinstance(stats, dict):
                            term_counts = stats.setdefault("term_counts", {})
                            term_step_sums = stats.setdefault("term_step_sums", {})
                            ep_lens = env.unwrapped.episode_length_buf[truncate_ids]
                            term_counts["timeout"] = term_counts.get("timeout", 0) + int(truncate_ids.numel())
                            term_step_sums["timeout"] = term_step_sums.get("timeout", 0) + int(ep_lens.sum().item())
                        # Mirror the sequence that env.step() uses internally for resets so the
                        # recorder writes the episode out properly.
                        env.unwrapped.recorder_manager.record_pre_reset(truncate_ids)
                        env.unwrapped._reset_idx(truncate_ids)
                        env.unwrapped.recorder_manager.record_post_reset(truncate_ids)

                    all_reset = natural_reset | manual_truncate
                    if all_reset.any():
                        reset_ids = all_reset.nonzero(as_tuple=False).reshape(-1)
                        exploration_horizons[reset_ids] = sample_exploration_horizons(
                            len(reset_ids), min_exploration_horizon_steps, max_exploration_horizon_steps, device
                        )
                        exploration_lengths[reset_ids] = 0
                        if exploration_policy is not None:
                            exploration_policy.reset(reset_ids)

                new_count = self.recorder_manager.exported_successful_episode_count
                if new_count > current_recorded_demo_count:
                    increment = new_count - current_recorded_demo_count
                    current_recorded_demo_count = new_count
                    pbar.update(increment)

                # extras["log"] is only rewritten inside _reset_idx, so only trust it on
                # iters where something actually reset; otherwise pass None (stale).
                success_rate: float | None = None
                if bool(all_reset.any()):
                    extras = getattr(env.unwrapped, "extras", None)
                    log_dict = extras.get("log") if isinstance(extras, dict) else None
                    sr = log_dict.get(SUCCESS_RATE_LOG_KEY) if isinstance(log_dict, dict) else None
                    if hasattr(sr, "item"):
                        sr = sr.item()
                    if sr is not None:
                        success_rate = float(sr)

                n_expert = int(use_expert.sum().item())
                n_explore = int(use_exploration.sum().item())
                raw_fs = getattr(self.recorder_manager, "_filter_stats", {}) or {}
                filter_stats = {
                    k: (dict(v) if isinstance(v, dict) else v) for k, v in raw_fs.items()
                }

                logger.on_iter_end(
                    current_recorded_demo_count,
                    pbar=pbar,
                    success_rate=success_rate,
                    expert_count=n_expert,
                    explore_count=n_explore,
                    filter_stats=filter_stats,
                )

                if num_demos > 0 and new_count >= num_demos:
                    break

                if env.unwrapped.sim.is_stopped():
                    break

            pbar.close()

        # Flush the dataset file to make sure it's readable by the training process. Redundant for Zarr but needed for torch handler
        assert self.recorder_manager._dataset_file_handler is not None, "Dataset file handler is not set."
        self.recorder_manager._dataset_file_handler.flush()

        per_env_exports = {
            env_id: int(self.recorder_manager._exported_successful_episode_count.get(env_id, 0))
            for env_id in range(num_envs)
        }
        metrics = logger.log_end(current_recorded_demo_count, per_env_exports=per_env_exports)
        return {
            "demos_recorded": int(current_recorded_demo_count),
            "dataset_file": dataset_file,
            **metrics,
        }


def _connect_to_orchestrator(socket_path: str, auth_key: str, timeout_s: float = 60.0):
    """Retry connecting to the orchestrator for up to ``timeout_s`` seconds."""
    start = time.time()
    last_exc: Exception | None = None
    while time.time() - start < timeout_s:
        try:
            return Client(socket_path, family="AF_UNIX", authkey=auth_key.encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(0.5)
    raise RuntimeError(f"Failed to connect to orchestrator at {socket_path}: {last_exc}")


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Build the env once, then service collection jobs from the orchestrator."""

    # Recorder manager. We give it a temporary path; we'll swap paths per-job via CollectionSession.
    env_cfg.recorders = ActionStateRecorderManagerTransformedActionCfg()
    env_cfg.recorders.dataset_export_dir_path = "/tmp"
    env_cfg.recorders.dataset_filename = f"_worker_placeholder_{os.getpid()}.zarr"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = ZarrDatasetFileHandler

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.max_episode_length_s
    env_cfg.observations.policy.concatenate_terms = False

    agent_cfg = process_agent_cfg(env_cfg, agent_cfg)

    print("[worker] building gym env...", flush=True)
    _t0 = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[worker] gym.make done in {time.time() - _t0:.1f}s", flush=True)
    env = RslRlVecEnvWrapper(env)
    print("[worker] RslRlVecEnvWrapper done", flush=True)

    # Source of truth for timing is the live env, matching Isaac Lab's
    # ``ManagerBasedEnv.step_dt = sim.dt * decimation`` and
    # ``max_episode_length = ceil(episode_length_s / step_dt)``. The previous
    # formula used ``sim.render_interval`` which has nothing to do with env
    # step rate (it only controls how often the renderer ticks inside the
    # decimation loop), and silently produced ~12x inflated step counts when
    # ``decimation != render_interval`` (the typical case).
    step_dt = env.unwrapped.step_dt
    max_episode_length = int(env.unwrapped.max_episode_length)
    print(
        f"[worker] env timing: sim.dt={env_cfg.sim.dt:.5f} decimation={env_cfg.decimation} "
        f"render_interval={env_cfg.sim.render_interval} → step_dt={step_dt:.5f}s, "
        f"max_episode_length={max_episode_length} steps "
        f"(max_episode_length_s={env_cfg.episode_length_s})",
        flush=True,
    )

    device = torch.device(env_cfg.sim.device if isinstance(env_cfg.sim.device, str) else "cuda:0")
    print(f"[worker] creating CollectionSession on device={device}...", flush=True)
    _t0 = time.time()
    session = CollectionSession(
        env=env,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        device=device,
        max_episode_length=max_episode_length,
        deterministic=args_cli.deterministic,
        apply_exploration_ratio_filter=not args_cli.disable_exploration_ratio_filter,
        disable_task_success_filter=args_cli.disable_task_success_filter,
        transformer_mini_batch_size=args_cli.transformer_mini_batch_size,
        use_kv_cache=not args_cli.no_kv_cache,
        kv_cache_max_seq_len=args_cli.kv_cache_max_seq_len,
    )
    print(
        f"[worker] KV cache {'ENABLED' if not args_cli.no_kv_cache else 'DISABLED'} "
        f"(max_seq_len={args_cli.kv_cache_max_seq_len})",
        flush=True,
    )
    print(f"[worker] CollectionSession ready in {time.time() - _t0:.1f}s", flush=True)
    if args_cli.disable_exploration_ratio_filter:
        print("[worker] exploration-ratio < 0.95 filter is DISABLED by CLI flag.", flush=True)
    else:
        print("[worker] exploration-ratio < 0.95 filter is ENABLED (default).", flush=True)
    if args_cli.disable_task_success_filter:
        print(
            "[worker] task-success filter is DISABLED — admitting ALL episodes that pass the"
            " exploration-ratio filter.",
            flush=True,
        )
    else:
        print("[worker] task-success filter is ENABLED (default) — only successful episodes saved.", flush=True)

    # Delete the placeholder file we created on env init (it's a temp zarr).
    try:
        import shutil
        placeholder = os.path.join("/tmp", env_cfg.recorders.dataset_filename)
        if os.path.isdir(placeholder):
            shutil.rmtree(placeholder, ignore_errors=True)
        elif os.path.isfile(placeholder):
            os.remove(placeholder)
    except Exception:
        pass

    # Connect back to the orchestrator and announce readiness.
    print(f"[worker] connecting back to orchestrator at {args_cli.socket_path}...", flush=True)
    conn = _connect_to_orchestrator(args_cli.socket_path, args_cli.auth_key)
    conn.send({"status": "ready", "num_envs": env.num_envs, "max_episode_length": max_episode_length})
    print(f"[worker] connected to orchestrator at {args_cli.socket_path}; ready for jobs.", flush=True)

    # Main message loop.
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            print("[worker] orchestrator closed connection; shutting down.", flush=True)
            break

        cmd = msg.get("cmd")
        job_id = msg.get("job_id")
        if cmd == "shutdown":
            conn.send({"status": "bye", "job_id": job_id})
            break
        if cmd == "ping":
            conn.send({"status": "pong", "job_id": job_id})
            continue
        if cmd != "collect":
            conn.send({"status": "error", "job_id": job_id, "message": f"unknown cmd: {cmd}"})
            continue

        try:
            result = session.collect(
                dataset_file=msg["dataset_file"],
                num_demos=int(msg["num_demos"]),
                min_exploration_horizon=float(msg.get("min_exploration_horizon", 0.0)),
                max_exploration_horizon=float(msg.get("max_exploration_horizon", 0.0)),
                episode_length_s=float(msg["episode_length_s"]),
                exploration_checkpoint=msg.get("exploration_checkpoint"),
                seed=int(msg.get("seed", 0)),
            )
            conn.send({"status": "done", "job_id": job_id, "result": result})
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[worker] job {job_id} failed: {e}\n{tb}", flush=True)
            conn.send({"status": "error", "job_id": job_id, "message": str(e), "traceback": tb})

    env.close()
    conn.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore
    finally:
        simulation_app.close()
