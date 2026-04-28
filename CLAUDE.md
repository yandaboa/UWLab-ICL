# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UW Lab is a robotics research framework built on top of [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and NVIDIA Isaac Sim. The primary active research direction is **OmniReset** — RL-based manipulation without reward engineering — and **ASTEROID** (in-context adaptation via DAgger with an exploration–expert policy handoff).

## Environment Setup

All experiments must be run inside the Isaac Sim Docker container with the LTI conda environment activated:

```bash
bash ../isaac-start.sh       # launch the Isaac Sim Docker container
source ../activate_conda.sh  # activate the lti conda env inside the container
```

All `python` commands below assume this environment is active.

## Key Commands

### RL Training (RSL-RL)

Single GPU (debug):
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-v0 \
    --num_envs 16 --logger wandb --headless \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Multi-GPU (4 GPUs):
```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-v0 \
    --num_envs 16384 --logger wandb --headless --distributed \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

### Evaluation

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Play-v0 \
    --num_envs 1 --checkpoint <checkpoint.pt> \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

### ASTEROID (In-Context Adaptation)

```bash
python run_incontext_exploration_parallel.py \
  --expert_policy_checkpoint logs/rsl_rl/.../exported/policy.pt \
  --config_name in_context_adaptation.yaml \
  --exp_name incontext_adaptation \
  --wandb_project incontext_adaptation \
  --output_dir logs/in_context_adaptation \
  --num_demos 20000 --num_data_envs 8096 --num_eval_envs 32 \
  --max_iterations 4 --data_gpu 2 --train_gpu 3 \
  --insertive_object peg --receptive_object peghole --no_video
```

### Evaluating a Distilled (Diffusion) Policy

```bash
python scripts_v2/tools/eval_distilled_policy.py \
    --task <task_id> --checkpoint <diffusion_checkpoint> \
    --num_envs 1 --num_trajectories 100
```

### Linting / Formatting

```bash
pre-commit run --all-files   # runs isort, codespell, pyright
```

Run a single pytest (most tests require Isaac Sim; mark with `isaacsim_ci`):
```bash
pytest -m "not isaacsim_ci" <test_file>
```

## Architecture

### Source Packages (`source/`)

| Package | Purpose |
|---|---|
| `uwlab` | Core extensions: controllers, actuators, sensors, scene utilities |
| `uwlab_assets` | Robot/object USDs and configs (UR5e + Robotiq 2F-85, assembly objects) |
| `uwlab_tasks` | Gymnasium environments registered with Isaac Lab's `ManagerBasedRLEnv` |
| `uwlab_rl` | RL wrappers: `DiffusionPolicyWrapper`, KV-cache manager, RSL-RL / SKRL configs |
| `metalearning` | Data utilities and tools for meta/in-context learning |

### Submodules

- `rsl_rl/` — fork of RSL-RL extended with `BCPPO` (behavior cloning + PPO), `DistillationRunner`, and custom rollout storage.
- `diffusion_policy/` — fork of the Chi et al. Diffusion Policy repo; used as the student/distillation policy architecture (transformer low-dim and image variants).

### Environment System (`uwlab_tasks`)

All manipulation environments live under:
```
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/
  config/ur5e_robotiq_2f85/   ← env configs & gym.register() calls (__init__.py)
    agents/rsl_rl_cfg.py       ← runner / algorithm / policy arch configs
    rl_state_cfg.py            ← state-based training env
    privileged_training_cfg.py ← asymmetric AC (privileged critic)
    asteroid_env_cfg.py        ← data-collection env pinned to terminal curriculum
  mdp/
    actions/                   ← task-space OSC actions
    observations.py / rewards.py / events.py / terminations.py
```

Environments follow Isaac Lab's Manager-Based pattern: `ObservationsCfg`, `EventCfg`, `RewardsCfg`, `TerminationsCfg` are dataclasses composed into an env config. Task-specific MDP terms live in `omnireset/mdp/`; generic terms from `isaaclab.envs.mdp` are also used.

Key env naming convention:
- `...-State-v0` — state-based RL training
- `...-State-Privileged-v0` — asymmetric AC (privileged critic sees full state)
- `...-State-Privileged-Augmented-v0` — adds force/time augmentations for sim2real
- `...-Play-v0` — eval variants (usually `num_envs=1`, video-compatible)
- `...-DataCollection-v0` — data collection for distillation / DAgger

The `env.scene.insertive_object` / `env.scene.receptive_object` Hydra overrides select the assembly pair (e.g. `peg`/`peghole`, `fbleg`/`fbtabletop`, `fbdrawerbottom`/`fbdrawerbox`).

### ASTEROID Pipeline

`run_incontext_exploration_parallel.py` orchestrates the iterative DAgger loop:

1. **`CollectionWorker`** boots Isaac Sim once on the data GPU and keeps it alive across iterations via a `multiprocessing.connection` socket (`collect_demos_worker.py`).
2. Each iteration the worker runs the hybrid expert/exploration rollout: for the first `exploration_horizon` steps the exploration (student) policy acts; then the expert takes over. Only episodes that pass the task-success and exploration-ratio filters are admitted.
3. **Training** runs as a `subprocess.Popen` on the training GPU in parallel with the next iteration's collection (true pipeline parallelism).
4. Evaluation (`eval_distilled_policy.py`) runs after each training job.

Important `collect_demos_worker.py` notes:
- `step_dt` is pulled from `env.unwrapped.step_dt` (not `sim.dt * render_interval`) — see `HANDOFF.md` for the history.
- The ratio filter (exploration ratio < 0.95) is ON by default; disable with `--disable_exploration_ratio_filter`.
- `--disable_task_success_filter` allows admitting all exploration-ratio-passing episodes.

### Policy Wrappers (`uwlab_rl/wrappers/diffusion.py`)

`DiffusionPolicyWrapper` wraps a Hydra-loaded diffusion/transformer checkpoint for Isaac Lab rollouts. Key features:
- `ObservationHistoryManager` subclasses manage sliding observation windows per env.
- `TransformerKVCacheManager` provides per-environment KV cache for transformer policies, enabling ~12× inference speedup (`--use_kv_cache` flag).
- Cache is invalidated on env reset via `reset_envs()`.

### RSL-RL Extensions (`rsl_rl/`)

- `BCPPO` (`algorithms/bc_ppo.py`) — PPO with an optional behavior-cloning loss term for DAgger.
- `DistillationRunner` (`runners/distillation_runner.py`) — teacher-student training runner built on `OnPolicyRunner`.
- Agent configs in `agents/rsl_rl_cfg.py` define `Base_PPORunnerCfg`, `Base_DAggerRunnerCfg`, `Asymmetric_DAggerRunnerCfg`, etc., composing `RslRlFancyActorCriticCfg` (supports FiLM conditioning, privileged-obs encoder, gSDE noise).

## Isaac Sim / Isaac Lab Specifics

- Scripts must boot Isaac Sim before any `isaaclab` imports. The pattern is always:
  ```python
  from isaaclab.app import AppLauncher
  AppLauncher.add_app_launcher_args(parser)
  args_cli, hydra_args = parser.parse_known_args()
  app_launcher = AppLauncher(args_cli)
  simulation_app = app_launcher.app
  # then import isaaclab modules
  ```
- `numpy` and `numba` must be imported **before** `AppLauncher` to prevent Isaac Sim's Kit runtime from overriding them with incompatible versions.
- Hydra overrides are passed as positional `key=value` args after `--`.
- Logs are written to `logs/rsl_rl/<experiment_name>/<timestamp>/`.
- Checkpoints: `model_<iter>.pt`; exported TorchScript: `exported/policy.pt`.

## Data / Assets

- Zarr datasets for distillation/DAgger are written to `logs/` or paths specified by `--output_dir`.
- Reset state datasets are in `reset_state_datasets/` and `reset_state_datasets_large/`.
- Robot and object USD assets are fetched from `UWLAB_CLOUD_ASSETS_DIR` (configured in `uwlab_assets`).
- Pre-trained checkpoints available on HuggingFace: `UW-Lab/uwlab-assets`.
