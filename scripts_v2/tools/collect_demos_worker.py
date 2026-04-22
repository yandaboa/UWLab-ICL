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
    "--enable_exploration_ratio_filter",
    action="store_true",
    default=False,
    help=(
        "If set, enable the check that rejects demos where the learner/exploration policy drove more"
        " than 95%% of the episode. Off by default; some tasks want this gate, most don't."
    ),
)
parser.add_argument("--seed", type=int, default=0, help="Base random seed for the env.")

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

    success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
    if hasattr(self._env, "termination_manager") and "success" in self._env.termination_manager.active_terms:
        success_results |= self._env.termination_manager.get_term("success")[env_ids]

    if hasattr(self, "exploration_lengths") and getattr(self, "apply_exploration_ratio_filter", False):
        episode_lengths = self._env.episode_length_buf[env_ids]
        exploration_lengths = self.exploration_lengths[env_ids]
        exploration_ratios = exploration_lengths / torch.clamp(episode_lengths, min=1)
        success_results = success_results & (exploration_ratios < 0.95)

    self.set_success_to_episodes(env_ids, success_results)

    if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
        self.export_episodes(env_ids)


def load_exploration_policy(checkpoint_path: str, device: torch.device, num_envs: int) -> DiffusionPolicyWrapper:
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
        apply_exploration_ratio_filter: bool = False,
    ):
        self.env = env
        self.env_cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.device = device
        self.max_episode_length = max_episode_length
        self.deterministic = deterministic
        self.apply_exploration_ratio_filter = apply_exploration_ratio_filter

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
            policy = load_exploration_policy(checkpoint_path, self.device, self.env.num_envs)
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
        from isaaclab.managers.recorder_manager import EpisodeData  # local import to avoid top-level dep
        for env_id in range(self.env.num_envs):
            rm._episodes[env_id] = EpisodeData()

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

        # Reconfigure recorder output for this job.
        self._swap_recorder_output(dataset_file)

        # Per-job episode length (steps).
        step_dt = self.env_cfg.sim.dt * self.env_cfg.sim.render_interval
        episode_length_steps = int(episode_length_s / step_dt)
        if episode_length_steps > self.max_episode_length:
            raise RuntimeError(
                f"Requested episode_length_s={episode_length_s} (→{episode_length_steps} steps) exceeds worker max"
                f" of {self.max_episode_length} steps. Restart the worker with a larger --max_episode_length_s."
            )

        max_exploration_horizon_steps = int(max_exploration_horizon * episode_length_steps)
        min_exploration_horizon_steps = int(min_exploration_horizon * episode_length_steps)

        # Resolve exploration policy (cache-aware).
        exploration_policy = self._get_exploration_policy(exploration_checkpoint)

        # Reset exploration bookkeeping.
        exploration_horizons = sample_exploration_horizons(
            num_envs, min_exploration_horizon_steps, max_exploration_horizon_steps, device
        )
        exploration_lengths = torch.zeros((num_envs,), device=device, dtype=torch.int32)
        self.recorder_manager.exploration_lengths = exploration_lengths

        current_recorded_demo_count = 0
        start_time = time.time()
        deterministic = self.deterministic

        # NOTE: env.reset() must run inside inference_mode. After the first rollout,
        # Isaac Lab's PhysX-backed buffers (e.g. ``self._data.root_link_pose_w``) become
        # inference tensors and cannot be written to outside inference_mode on subsequent
        # jobs. The rollout's internal ``_reset_idx`` already runs inside inference_mode
        # for the same reason; we extend the context to cover the per-job reset here.
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            # Reset all envs to make sure the recorder starts cleanly for this job.
            env.reset()
            if exploration_policy is not None:
                exploration_policy.reset(torch.arange(num_envs, device=device))

            pbar = tqdm(total=num_demos, desc=f"Recording demos → {os.path.basename(dataset_file)}", unit="demo")

            while True:
                # Choose expert vs exploration per-env based on per-env horizon.
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

                env.step(actions)

                # Compose natural resets (from env.step) with manual per-job truncation.
                natural_reset = env.unwrapped.reset_buf.clone().bool()
                too_long = env.unwrapped.episode_length_buf >= episode_length_steps
                manual_truncate = too_long & ~natural_reset
                if manual_truncate.any():
                    truncate_ids = manual_truncate.nonzero(as_tuple=False).reshape(-1)
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

                if num_demos > 0 and new_count >= num_demos:
                    break

                if env.unwrapped.sim.is_stopped():
                    break

            pbar.close()

        # Flush the dataset file to make sure it's readable by the training process. Redundant for Zarr but needed for torch handler
        assert self.recorder_manager._dataset_file_handler is not None, "Dataset file handler is not set."
        self.recorder_manager._dataset_file_handler.flush()

        elapsed = time.time() - start_time
        return {
            "demos_recorded": int(current_recorded_demo_count),
            "elapsed_s": elapsed,
            "dataset_file": dataset_file,
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

    step_dt = env_cfg.sim.dt * env_cfg.sim.render_interval
    max_episode_length = int(env_cfg.episode_length_s / step_dt)

    print("[worker] building gym env...", flush=True)
    _t0 = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[worker] gym.make done in {time.time() - _t0:.1f}s", flush=True)
    env = RslRlVecEnvWrapper(env)
    print("[worker] RslRlVecEnvWrapper done", flush=True)

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
        apply_exploration_ratio_filter=args_cli.enable_exploration_ratio_filter,
    )
    print(f"[worker] CollectionSession ready in {time.time() - _t0:.1f}s", flush=True)
    if args_cli.enable_exploration_ratio_filter:
        print("[worker] exploration-ratio < 0.95 filter is ENABLED by CLI flag.", flush=True)

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
