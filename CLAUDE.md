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

## Repository conventions

- **Visualization / debug / analysis scripts live under `analysis/`** — never
  in the repo root. This includes anything that loads checkpoints purely to
  read them, anything that walks log directories to summarize runs, anything
  that produces plots / tables / videos for human consumption, and one-off
  inspection helpers (e.g. `analysis/inspect_zarr.py`). When adding a new
  script of this kind, place it directly under `analysis/`. If the script
  needs to import a repo-root module (`incontext_eval_log`, etc.), prepend
  the repo root to `sys.path` near the top of the script:
  ```python
  import pathlib, sys
  _REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
  if str(_REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(_REPO_ROOT))
  ```
  Plot output paths (`plots/`) are gitignored.
- **`logs/`, `plots/`, `*.pt` checkpoints, datasets, and rollouts are gitignored** — never commit them.

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

For anything involving the **discrete-AR action head**, **arch sweeps over
hidden_depth × n_head × hidden_dim**, **BC vs DAgger comparisons**,
**checkpoint-selection / sample-size / episode-length-mismatch debugging**,
the **multi-GPU sweep launcher** (`scripts_v2/sweep_disc_ar_arch.sh`), or the
suite of plotting / analysis scripts under `analysis/` (e.g.
`analysis/plot_sweep_disc_ar_arch.py`, `analysis/plot_iter0_ckpt_sweep.py`,
`analysis/plot_full_eval.py`), consult
[`in_context_adaptation.md`](in_context_adaptation.md) — it has the
canonical commands, sample-size and episode-length rules, and recovery
recipes for the failure modes that came up while building this pipeline
(numpy `_core` missing, headless render mode, stale editable install,
under-trained-checkpoint eval, missing-success-termination, etc.).

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
- `sample_action: bool` is set on the wrapped policy at construction (`True` for data collection, `False` for eval — greedy/argmax). Plumbed into `InterleavedTransformerImagePolicy._decode_step` / `_ar_inference_loop`. The wrapper also auto-detects discrete-AR heads via `outputs_raw_action`, suppresses `LinearNormalizer.unnormalize` for them, and exposes `save_discretize_spec(path)` so eval can persist the bin spec next to the checkpoint.

### Action Heads (`diffusion_policy/diffusion_policy/model/head/output_head.py`)

`OutputHead` is the abstract action-output base; subclasses produce action predictions and compute training losses given a backbone hidden state.
- `GaussianOutputHead` — diagonal-Gaussian over the `LinearNormalizer`-normalized action space (the original continuous head).
- `DiscreteAROutputHead` — autoregressive discrete head over a fixed bin vocabulary. **Does not own a transformer.** The host policy drives decoding by running its own trunk D times per env-step (D = `action_dim`); each AR step appends one new token (`bin_embed(prev_bin) + dim_embed[dim]`) to a transient sub-sequence, while the persistent KV cache only contains obs/context tokens. Arm dims share a single `Linear(hidden_dim, num_bins)` projection (`bin_proj`); the gripper (must be the last dim) uses a separate `Linear(hidden_dim, 1)` projection (`gripper_proj`) trained with `BCEWithLogitsLoss` — sigmoid-threshold at inference, `0 → -1` (open), `1 → +1` (close). `outputs_raw_action = True` so the host policy skips `unnormalize` for both training-target and predicted actions. Bin centers + spec are written via `get_spec()` and persisted as `discretize_spec.json`.

`InterleavedTransformerImagePolicy` (`diffusion_policy/policy/interleaved_transformer_image_policy.py`) selects the head via `head_type: "gaussian" | "discrete_ar"`. For `discrete_ar` it overrides `_decode_step`, `predict_action`, and adds `_ar_inference_loop` (KV-cached AR rollout) plus `_ar_train_forward` (teacher-forced AR pass) — both reuse the main trunk. Don't add a custom `output_head` config when `head_type=discrete_ar`; specify `num_bins` / `clip_val` / `gripper_dim` instead.

### Action Discretization Pipeline (`source/uwlab_rl/uwlab_rl/utils/action_discretize.py`)

Shared helpers used by both data collection (`scripts_v2/tools/collect_demos_asteroid.py`) and eval/training:
- `discretize_actions(actions, num_bins, clip_val)` — snaps 6 continuous arm dims to bin centers on `[-clip_val, +clip_val]` and sign-thresholds the gripper to `±1`.
- `make_discretize_spec(...)` / `save_discretize_spec(...)` / `load_discretize_spec(...)` — JSON spec carrying `num_bins`, `clip_val`, `arm_bin_centers`, `gripper_bins`, `arm_dims`, `gripper_dim`. Saved alongside every dataset and every trained checkpoint so downstream code can reconstruct the categorical head without hardcoded constants.

Inside `collect_demos_asteroid.py` the discretization slots in **after** the inverse-action mapping: `expert_action → arm_term.inverse_process_actions(..., original_scale=expert_action_scale) → discretize_actions → env.step`. The `expert_action_scale` defaults to `[0.01, 0.01, 0.002, 0.02, 0.02, 0.2]` (XYZ + axis-angle scales the expert was trained with) — passed via `--expert_action_scale` since the env's `_scale_default` is not an accurate stand-in. The expert's `prev_actions` observation is patched to its own *intended* (un-discretized, un-perturbed) action each step so it stays in distribution across the action-feedback loop.

### Success Criterion & Eval Metrics

The primary success boolean is recomputed each step inside `progress_context` (in `omnireset/mdp/rewards.py`) **and** `task_command` (in `omnireset/mdp/commands.py`) — both apply identical math (same offsets from USD metadata, same `subtract_frame_transforms`, same `xyz_distance < success_position_threshold & euler_xy_distance < success_orientation_threshold`). The reward term explicitly reads thresholds off `task_command.success_position_threshold` / `success_orientation_threshold`, so the two cannot drift.

Two derived metrics, *temporally aggregated differently*:
- **Any-time success rate** (`success_rate` in `eval_stats.json`) — `OR` over all timesteps in the episode. Computed in `eval_distilled_policy.py` by accumulating `progress_ctx.success` into a per-env `episode_ever_successful` buffer, summed at reset.
- **End-of-episode success rate** (`metrics["Metrics/task_command/end_of_episode_success_rate"]`) — value at the *terminal* step, written by the command term's `_update_metrics` when `episode_length_buf == 0`. **This is the headline metric**; report this as the comparison number across runs.

EOE is strictly more demanding (the policy must enter *and stay in* the success region until the timer expires); any-time is easier. The gap can be 25-30 pp on overtrained / open-loop-overshoot policies. Don't conflate them.

### Evaluation Correctness Rules

A few non-obvious rules learned the hard way:

1. **Sample size**: with N=128 episodes, EOE has roughly **±10 pp** sampling noise — enough to fabricate U-shaped per-iteration "non-monotonicity" out of nothing. **Use N=512 minimum, N=2048 for headline numbers.**
2. **Episode length must match training horizon**: the orchestrator collects iter-0 at 6.0s, iter-N>0 at `episode_length_s[N-1]` (8.0 / 9.0 / 10.0 / 11.0). Evaluating an iter-0 checkpoint at 11s when it was only trained on 6s episodes inflates the any-time-vs-EOE gap by 25-40 pp because the policy enters and exits the success region during the unfamiliar tail. Always pass `env.episode_length_s=<iter_max>` to the eval (Hydra override).
3. **Checkpoint selection**: `_expected_train_checkpoint(step=50000)` priority is `step_{step:07d}.ckpt` → highest `step_*.ckpt` → `latest.ckpt`. The earlier default of `step=5000` was severely under-trained and produced ~0 % success across the board. The orchestrator **does not** auto-select by val-loss; if you want best-by-val-loss you'd need to wire it explicitly (we tried and abandoned the attempt — see commit history / `in_context_adaptation.md`).
4. **Eval doesn't require a `success` termination term**. `eval_distilled_policy.py` reads success from `progress_context.success` and accumulates the any-time / end-of-episode buffers itself. This means you can (and should) remove the `success` termination from eval task configs to avoid biasing summary statistics toward "fast" successes. If a task's `progress_context` term is missing, the eval falls back to the legacy termination flag.

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
