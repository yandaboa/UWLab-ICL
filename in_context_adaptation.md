# In-context Adaptation: Workflow & Commands

Commands used to drive the discrete-AR / DAgger sweep + analysis. Run inside
the Isaac Sim container with the lti conda env active:

```bash
bash ../isaac-start.sh        # spawn or join the container
source ../activate_conda.sh   # activate `lti` conda env
```

The Docker container is started with `--shm-size=512g` (`isaac-start.sh`) and
auto-injects `WANDB_API_KEY` from the host's `~/.netrc` so `wandb.init`
works out of the box.

---

## Single training+eval run

Reference invocation. Defaults to `--checkpoint_num=50000` for eval (falls
through to highest available `step_*.ckpt` then `latest.ckpt`):

```bash
python run_incontext_exploration.py \
  --expert_policy_checkpoint logs/rsl_rl/teacher/exported/policy.pt \
  --config_name in_context_adaptation_interleave.yaml \
  --exp_name <name> \
  --wandb_project incontext_adaptation \
  --output_dir logs/<name> \
  --num_demos 20000 \
  --num_data_envs 1024 \
  --num_eval_envs 64 --num_eval_episodes 128 \
  --max_iterations 4 \
  --insertive_object peg --receptive_object peghole \
  --no_video \
  --use_inverse_actions --num_bins 100 --discretize_clip_val 50.0 \
  --config_overrides policy.hidden_depth=8 policy.n_head=16 policy.hidden_dim=512
```

Key flags:
- `--use_inverse_actions` — collect demos via `inverse_process_actions` so a
  non-augmented expert produces correct demos in the augmented MDP.
- `--num_bins 100 --discretize_clip_val 50.0` — discretize 6 arm dims into 100
  bins on `[-50, +50]` before recording / executing.
- `--config_overrides` — Hydra overrides forwarded to `train.py`. Used for
  arch sweeps (e.g. `policy.hidden_depth=L policy.n_head=H policy.hidden_dim=512`).
- `--checkpoint_num` (default 50000) — eval checkpoint preference. Override
  to a specific step number if needed.

## BC baseline (1 iteration, 80k expert demos)

```bash
python run_incontext_exploration.py \
  --expert_policy_checkpoint logs/rsl_rl/teacher/exported/policy.pt \
  --config_name in_context_adaptation_interleave.yaml \
  --exp_name bc_baseline_disc_ar \
  --wandb_project incontext_adaptation_bc_baseline \
  --output_dir logs/bc_baseline_disc_ar \
  --num_demos 80000 \
  --num_data_envs 1024 \
  --num_eval_envs 64 --num_eval_episodes 128 \
  --max_iterations 1 \
  --insertive_object peg --receptive_object peghole \
  --no_video \
  --use_inverse_actions --num_bins 100 --discretize_clip_val 50.0 \
  --config_overrides \
    training.max_gradient_steps=200000 \
    policy.hidden_depth=4 policy.n_head=8 policy.hidden_dim=512
```

`--max_iterations 1` skips the post-iter collection step → pure expert demos
only. `training.max_gradient_steps=200000` matches the 4× per-iter DAgger
optimisation budget for a fair comparison.

---

## Architecture sweep on multiple GPUs

Sweep `hidden_depth × n_head` for `hidden_dim=512`. Worker pool runs one
training+eval pipeline per GPU; new combos launch as GPUs free up.

```bash
bash scripts_v2/sweep_disc_ar_arch.sh
```

Key environment overrides:
```bash
GPU_IDS="0 1 2 3 4 5 6"            # default; reserves GPU 7
HIDDEN_DEPTHS="4 8 12"
N_HEADS="4 8 16"
HIDDEN_DIMS="512"
```

Per-job command (rendered by the script for each combo):
```bash
CUDA_VISIBLE_DEVICES=<g> python run_incontext_exploration.py \
  --config_name in_context_adaptation_interleave.yaml \
  --exp_name sweep_disc_ar_l<L>_h<H>_d<D> \
  --output_dir logs/sweep_disc_ar/l<L>_h<H>_d<D> \
  --num_demos 20000 --num_data_envs 1024 \
  --num_eval_envs 64 --num_eval_episodes 128 --max_iterations 4 \
  --insertive_object peg --receptive_object peghole --no_video \
  --use_inverse_actions --num_bins 100 --discretize_clip_val 50.0 \
  --config_overrides \
    policy.hidden_depth=<L> policy.n_head=<H> policy.hidden_dim=<D>
```

Logs go to `logs/sweep_disc_ar/sweep_logs/<TS>/<arch_tag>.log` plus a
top-level `dispatcher_<TS>.out`. Tail in real time:
```bash
docker exec isaac-sim bash -c 'tail -f $(cat /tmp/last_sweep_log)'
docker exec isaac-sim bash -c 'tail -f logs/sweep_disc_ar/sweep_logs/<TS>/l8_h16_d512.log'
```

---

## Re-evaluating a checkpoint (no DAgger pipeline)

Standalone eval against a single checkpoint. Use this when the orchestrator's
default `--checkpoint_num` selection picked the wrong snapshot, or to re-eval
with different settings (more episodes, different `episode_length_s`, etc.).

```bash
python scripts_v2/tools/eval_distilled_policy.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0 \
  --checkpoint <path>/iteration_N/checkpoints/latest.ckpt \
  --num_envs 256 --num_trajectories 2048 --transformer_mini_batch_size 256 \
  --headless \
  --stats_output_path <path>/iteration_N/eval_stats_2048ep.json \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  env.episode_length_s=10.0
```

**Important sample-size note**: 128 episodes is too few — the per-episode
EOE-success indicator has roughly ±10 pp standard error on N=128, enough to
fabricate U-shaped per-iteration "noise." 2048 episodes (or at least 512) is
the right operating point; the per-iteration sweep curves smooth out
substantially at higher N.

**Episode length must match training horizon**:
- iter-0 was trained on 6.0s episodes (`initial_episode_length_s`)
- iter-1 on 6.0/8.0s mixture (`episode_length_s[0]=8.0`)
- iter-2 on 6.0/8.0/9.0s (`episode_length_s[1]=9.0`)
- iter-3 on 6.0/8.0/9.0/10.0s (`episode_length_s[2]=10.0`)

Eval at the iteration's max-train length to avoid a synthetic any-time-vs-EOE
gap. (e.g. evaluating an iter-0 checkpoint at 11s when it was only trained on
6s episodes inflates the gap by 25-40 pp.)

## Recording a viewport video

```bash
python scripts_v2/tools/eval_distilled_policy.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0 \
  --checkpoint <ckpt> \
  --num_envs 1 --num_trajectories 50 --transformer_mini_batch_size 1 \
  --headless --enable_cameras \
  --record_viewport_video --video_length 500 \
  --video_output_dir <out_dir> \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  env.episode_length_s=11.0
```

`--headless --enable_cameras` together force PARTIAL_RENDERING; without
`--headless` AppLauncher tries to open a display, fails, and falls back to
`NO_GUI_OR_RENDERING` which makes `env.render()` raise. The
`--record_viewport_video` path uses `gym.wrappers.RecordVideo` and works
even on tasks that have no rgb camera observations.

---

## Plotting

### Sweep arch comparison (vs BC overlay as horizontal reference line)

```bash
python analysis/plot_sweep_disc_ar_arch.py \
  --sweep_dir logs/sweep_disc_ar \
  --since 2026-04-29_10-58-41 \
  --baseline_eval_stats logs/bc_baseline_disc_ar/.../iteration_0/eval_stats_latest.json \
  --baseline_label "BC (80k demos, 200k steps)" \
  --out plots/sweep_disc_ar_arch.png
```

### iter-0 within-training checkpoint sweep

```bash
python analysis/plot_iter0_ckpt_sweep.py \
  --sweep_outdir logs/sweep_disc_ar/l8_h4_d512/iter0_ckpt_sweep_<TS> \
  --out plots/iter0_ckpt_sweep_l8_h4_d512.png
```

### Full multi-arch eval (bar charts, scatter, iter progression)

```bash
python analysis/plot_full_eval.py \
  --eval_dir logs/full_eval_2048ep_<TS> \
  --out_dir plots/full_eval_<TS>
```

Produces:
- `eos_vs_anytime.png` — scatter of EOE vs any-time success across archs
  with Pearson r.
- `arch_perf_vs_layers_heads.png` — perf vs depth + perf vs heads.
- `iter_progression.png` — l8_h4_d512 across DAgger iterations.
- `iter3_ckpt_sweep.png` — within-iter-3 training-step curve.

---

## High-throughput parallel eval scheduler

For evaluating many checkpoints across multiple GPUs, the
`/tmp/launch_full_eval.sh` pattern (see session history) parallelises a JOBS
list across a `GPU_POOL`. JOB format: `"<TAG__suffix> <ckpt_path> <ep_len_s>"`.
Each spec writes to `<OUT_BASE>/<TAG__suffix>/eval_stats.json`. Critical
gotcha when copying that pattern: **declare `local rtag` (or any other
loop-internal vars) inside the `reap` function**, otherwise it clobbers the
outer `tag` and every job ends up writing to a single output directory.

The default eval invocation under that scheduler:
```bash
CUDA_VISIBLE_DEVICES=<g> python scripts_v2/tools/eval_distilled_policy.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0 \
  --checkpoint <ckpt> \
  --num_envs 256 --num_trajectories 2048 --transformer_mini_batch_size 256 \
  --headless \
  --stats_output_path <out>/eval_stats.json \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  env.episode_length_s=<ep>
```

---

## Useful operational notes

### Container shell

```bash
docker exec -it isaac-sim bash
source /mnt/storage/lti/activate_conda.sh lti
cd /mnt/storage/lti/UWLab-ICL
```

### Killing a runaway sweep / orphaned children

```bash
docker exec isaac-sim pkill -f sweep_disc_ar_arch
docker exec isaac-sim pkill -f run_incontext_exploration
docker exec isaac-sim pkill -f collect_demos_asteroid
docker exec isaac-sim pkill -f diffusion_policy/train
docker exec isaac-sim pkill -f eval_distilled_policy
```

### Common-cause failure recipes

- **`module 'numpy' has no attribute '_core'`** — Kit's bundled numpy is being
  resolved instead of conda's. The fix is in `diffusion_policy/train.py`'s
  prelude (strip `/isaac-sim/...` from `sys.path` + evict cached numpy /
  numba from `sys.modules` + pre-import). If it ever recurs, repair conda
  numpy with:
  ```bash
  sudo rm -rf /home/ubuntu/miniforge3/envs/lti/lib/python3.11/site-packages/~umpy-*
  sudo /home/ubuntu/miniforge3/envs/lti/bin/python -m pip install --force-reinstall --no-deps numpy==2.4.4
  ```

- **`Cannot render 'rgb_array' when ... NO_GUI_OR_RENDERING`** — pass
  `--headless --enable_cameras` together when calling `eval_distilled_policy.py`
  with `--record_viewport_video`.

- **`No API key configured. Use 'wandb login'`** — container started without
  the `WANDB_API_KEY` env var. Restart with the updated `isaac-start.sh`
  (extracts the key from `~/.netrc` on the host and forwards via
  `-e WANDB_API_KEY` plus a bind mount of `/root/.netrc`).

- **`module 'diffusion_policy' has no attribute 'workspace.train_mlp_image_workspace'`
  via the editable install** — the conda env had `pip install -e` pointing
  at a stale `/home/ubuntu/lti/UWLab-ICL/diffusion_policy/` rather than
  `/mnt/storage/lti/UWLab-ICL/diffusion_policy/`. Fix once with:
  ```bash
  sudo /home/ubuntu/miniforge3/envs/lti/bin/python -m pip uninstall -y diffusion-policy
  sudo /home/ubuntu/miniforge3/envs/lti/bin/python -m pip install --no-deps \
    -e /mnt/storage/lti/UWLab-ICL/diffusion_policy
  ```

- **Eval metrics return 0% success** — orchestrator picked an under-trained
  checkpoint. Default selection priority is
  `step_{checkpoint_num:07d}.ckpt` → highest `step_*.ckpt` → `latest.ckpt`.
  Increase `--checkpoint_num` (50000 is the current default), or re-eval
  `latest.ckpt` directly with `eval_distilled_policy.py`.

- **Eval crashes on `Success term not found in termination manager`** — fixed
  in `eval_distilled_policy.py`: success is now derived from the
  `progress_context` reward term's `position_aligned & orientation_aligned`
  buffer (same source the `success_reward` reads from), accumulated across
  the episode. Works even when the task has no `success` termination term.

### Dataset / checkpoint layout

```
logs/sweep_disc_ar/<arch_tag>/sweep_disc_ar_<arch_tag>/<TS>/
├── dataset-iteration-{0..3}/        ← zarr datasets per DAgger iter
│   ├── data.zarr/
│   └── discretize_spec.json
├── iteration_{0..3}/                ← train output dirs
│   ├── checkpoints/
│   │   ├── latest.ckpt              ← final state
│   │   ├── step_0001000.ckpt … step_0NNNNNN.ckpt
│   │   └── discretize_spec.json     ← bin spec, written by eval
│   └── eval_stats.json              ← per-iter eval result (orchestrator)
└── eval_log.json                    ← cumulative eval log across iterations
```

### Eval stats schema (`eval_stats.json`)

```json
{
  "checkpoint": "<ckpt path>",
  "task":       "<task id>",
  "iteration":  N,
  "episodes":   2048,
  "successful_episodes": 1399,
  "success_rate": 0.6831,                                                 ← any-time
  "metrics": {
    "Metrics/task_command/end_of_episode_success_rate": 0.5847,           ← EOE (headline)
    "Metrics/task_command/end_of_episode_pos_align_error": 0.0386,
    "Metrics/task_command/end_of_episode_rot_align_error": 0.4636,
    "Episode_Reward/abnormal_robot": -0.55,
    "Episode_Reward/success_reward": 0.21,
    ...
  }
}
```

`metrics["Metrics/task_command/end_of_episode_success_rate"]` is the headline
number — counts an episode as successful only if the alignment criteria are
satisfied at the final step (not just at *some* step during the episode).
