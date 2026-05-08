# DIAYN-style Diversity Training — session handoff (2026-05-05/06)

Adds a per-episode skill index and a DIAYN-style discriminator bonus on top of the
existing OmniReset PPO pipeline. Goal: produce a base policy that solves the peg-in-hole
task in `K = 10` qualitatively distinct ways.

This file documents what's been built, the gotchas hit, and where to pick up.

---

## What was added

### `rsl_rl` submodule

| Path | What |
|---|---|
| `rsl_rl/rsl_rl/modules/skill_discriminator.py` | `SkillDiscriminator` MLP classifier (q(z\|s)) with optional `EmpiricalNormalization`. |
| `rsl_rl/rsl_rl/algorithms/diversity_ppo.py` | `DiversityPPO(PPO)`. Owns the discriminator + AdamW optimiser, lazy-builds it in `init_storage`, runs a CE update every `update_frequency` PPO steps, and exposes `attach_env(env)` so the env-side reward term can call the discriminator. Multi-GPU: `broadcast_parameters` and `_reduce_discriminator_gradients` keep weights and grads in sync across ranks. |
| `rsl_rl/rsl_rl/runners/diversity_runner.py` | `DiversityRunner(OnPolicyRunner)`. Calls `attach_env` after construction; persists discriminator state in `save`/`load`. |

### Task-side MDP terms (`uwlab_tasks/.../omnireset/mdp/diversity.py`)

- `SkillObs(ManagerTermBase)` — per-env `LongTensor[num_envs]` skill buffer, resampled on
  reset; emits one-hot to policy. Publishes `env.diversity_skill_idx` and
  `env.diversity_num_skills`. Supports `force_skill: int` (default `-1` = uniform random)
  to lock all envs to a single skill at eval time.
- `DiversityReward(ManagerTermBase)` — calls `env.diversity_discriminator(s_{t+1})`,
  computes `log q(z|s) - log p(z)`, scaled by `env.diversity_reward_scale`. Falls back to
  zeros when the discriminator hasn't been attached yet.
  - Maintains a per-env `task_done` latch (True once `progress_context.success` fires
    within an episode). Once latched, the diversity bonus is zeroed for that env until
    reset. Latch is published as `env.diversity_task_done`.
- `diversity_task_done_obs(env)` — returns the latch as a `(num_envs, 1)` float tensor for
  the `diversity_meta` obs group (used by the algorithm to mask post-success transitions).

### Env / agent / task registration

- `omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py`
  - `NUM_SKILLS = 10` (must stay in sync with the agent cfg).
  - `DiversityObservationsCfg` extends `ObservationsCfg` with:
    - `policy.skill` one-hot term added to `PolicyCfg`.
    - `discriminator_obs` group: same task state as the policy minus the skill (no
      leakage), `concatenate_terms=True`, `history_length=1`.
    - `diversity_meta` group: just the `task_done` latch, `concatenate_terms=True`,
      `history_length=1`. Carried in rollout storage so the algorithm can read it.
  - `DiversityRewardsCfg` extends `RewardsCfg` with the diversity reward term (runs
    *after* `progress_context`, so the success bit is current).
  - `Ur5eRobotiq2f85RelCartesianOSCDiversityTrainCfg` — Stage-1 training env + diversity.
  - `Ur5eRobotiq2f85RelCartesianOSCDiversityPlayCfg` — same skills/rewards layered onto
    the Stage-1 `-Play-v0` event set (single `ObjectAnywhereEEAnywhere` reset path, no
    sysid DR / OSC-gain randomization).
- `agents/rsl_rl_cfg.py`: `Diversity_PPORunnerCfg(class_name="DiversityRunner")`.
  Discriminator hyperparameters live in `RslRlDiversityPpoAlgorithmCfg.discriminator_cfg`
  (a `DiscriminatorTrainingCfg` — see `source/uwlab_rl/uwlab_rl/rsl_rl/rl_cfg.py`).
- `omnireset/config/ur5e_robotiq_2f85/__init__.py` registers
  `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-v0` and
  `…-State-Diversity-Play-v0`.

### Training / play scripts

- `scripts/reinforcement_learning/rsl_rl/train.py` dispatches `DiversityRunner` when
  `agent_cfg.class_name == "DiversityRunner"`.
- `scripts/reinforcement_learning/rsl_rl/play.py` adds:
  - `--skill <int>` — sets `env.observations.policy.skill.params.force_skill` and tags
    the recorded video filename (`play-skill<N>-<timestamp>-step-0.mp4`).
  - `--action_rescale` (default **False**) — only apply the hard-coded inverse-action
    rescale to `[0.01, 0.01, 0.002, 0.02, 0.02, 0.2]` (the finetune-eval scale). Off by
    default; pass it only when replaying a Stage-2 / finetune-eval policy. Stage-1
    policies (incl. Diversity) train at `[0.02, 0.02, 0.02, 0.02, 0.02, 0.2]` and need
    no rescale.
  - Loads discriminator weights when the checkpoint contains `discriminator_state_dict`.

---

## Launch commands

### Train (4 × GPU 4–7)

```bash
docker exec -e CUDA_VISIBLE_DEVICES=4,5,6,7 isaac-sim bash -lc \
  'source /mnt/storage/lti/activate_conda.sh lti && \
   python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 --master_port 29501 \
     scripts/reinforcement_learning/rsl_rl/train.py \
     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-v0 \
     --num_envs 16384 --logger wandb --headless --distributed \
     env.scene.insertive_object=peg env.scene.receptive_object=peghole'
```

`--master_port 29501` avoids the default `29500` which collides with Patrick's GPU 0–3
training.

### Per-skill 500-step videos (single GPU)

```bash
CKPT=/mnt/storage/lti/UWLab-ICL/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/<TS>/model_<N>.pt
docker exec -e CUDA_VISIBLE_DEVICES=4 isaac-sim bash -lc "source /mnt/storage/lti/activate_conda.sh lti && \
  for skill in 0 1 2 3 4 5 6 7 8 9; do
    python scripts/reinforcement_learning/rsl_rl/play.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-Play-v0 \
      --num_envs 1 --headless --enable_cameras \
      --video --video_length 500 --skill \$skill \
      --checkpoint $CKPT \
      env.scene.insertive_object=peg env.scene.receptive_object=peghole
  done"
```

Videos land in `logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/<TS>/videos/play/`,
named `play-skill<N>-<timestamp>-step-0.mp4`.

`act_inference` returns the deterministic Gaussian mean (no sampling) since
`state_dependent_std=False`. Different skills should therefore produce different
deterministic trajectories.

---

## Gotchas hit during this session

### 1. `sanitize_rsl_rl_cfg` strips kwargs that aren't explicit in the algorithm signature

`scripts/.../cli_args.py:sanitize_rsl_rl_cfg` introspects
`inspect.signature(alg_class.__init__).parameters.keys()` and deletes alg-cfg keys not in
that set. Forwarding PPO args via `**kwargs` makes them invisible and they get stripped.
**Fix:** declare all parent-PPO kwargs explicitly in `DiversityPPO.__init__` and forward
to `super().__init__` (mirror BCPPO).

### 2. Default `master_port` 29500 collides with concurrent DDP runs

Patrick's run on GPUs 0–3 was already holding port 29500. Always pass
`--master_port 29501` (or another free port) to second-on-host DDP launches.

### 3. play.py's hard-coded inverse-action rescale silently mis-scales actions

`get_perturbed_env_solving_actions` was unconditionally calling
`inverse_process_actions(..., original_scale=[0.01, 0.01, 0.002, 0.02, 0.02, 0.2])` —
the **finetune-eval** scale. Stage-1 / Diversity policies train at
`[0.02, 0.02, 0.02, 0.02, 0.02, 0.2]`, so xyz commands were silently shrunk by 0.5× and
z by 0.1× (especially bad for the insertion direction).
**Fix:** rescale is now off by default; pass `--action_rescale` only when replaying a
finetune-eval policy.

### 4. NCCL deadlock when discriminator update has rank-divergent batch sizes

The first attempt at masking post-success / post-done transitions used
`s_next, z_next = s_next[keep], z_next[keep]` followed by minibatching over the filtered
batch. Each rank ended up with a different `batch_size` -> different number of
minibatches -> different number of all-reduce calls inside `_reduce_discriminator_gradients`.
NCCL all-reduce stalled at the 600 s watchdog timeout and the run aborted at iter ~25.
**Fix (current code in `diversity_ppo.py`):** keep all transitions, build a per-sample
weight `weight = (~prev_dones) & (~prev_latch).float()` and use a weighted CE loss
`(per_sample_loss * w).sum() / w.sum().clamp_min(1)`. All ranks now run the same number
of epochs × minibatches × all-reduces in lockstep. The all-zero-weight branch returns
metric stubs; **never an early `return None`** in the multi-rank update path.

> **Important rule for any future change to `_update_discriminator_from_storage`:**
> every rank must take the same control-flow path through the all-reduces. No
> `return None` after `_update_step % freq != 0`-style guards is OK (those increment in
> lockstep), but anything depending on rank-local rollout state (mask density, sample
> count, etc.) must not branch around an all-reduce.

### 5. `EmpiricalNormalization` running stats are NOT all-reduced

Matches the existing pattern for actor/critic obs normalizers in this codebase. Small
cross-rank drift is possible but historically not load-bearing. The discriminator weights
themselves stay in sync via gradient all-reduce + weight broadcast.

### 6. Reward-term ordering matters for the latch

`DiversityReward` reads `progress_context.success` to update the per-env task-done
latch. `RewardsCfg` lists `progress_context` before the diversity term, and Python
configclass dataclass-field order is preserved, so progress_context fires first and
DIAYN sees the current-step success bit. **Don't reorder the cfg.**

---

## Run log

- `2026-05-05_11-47-15` — first 2-iter smoke (8 envs, 1 GPU): pipeline boots, discriminator
  fires, diversity reward registers. Validated end-to-end.
- `2026-05-05_12-28-43` — first full 4-GPU run (16384 envs, port 29501). Reached iter
  ~5400 / 40000 (~9 hours). Wandb / TB logs there. Killed by user. EOE success rate
  ~0.42 with `Episode_Reward/diversity ≈ 0.20`. **Subjective issue raised by the user:
  the 10 skills produced visually similar trajectories.** Hypothesis: most of the
  reward signal came from late-episode (post-success) frames where the policy is free
  to do whatever — discriminator learns to classify those, not the task-relevant
  prefix.
  - Ten 500-step videos under `videos/play/` exist for `model_5400.pt` (still recorded
    with the old action_rescale-on default; first batch was deleted, second batch was
    re-recorded with rescale-off but the loop got mid-killed during code edits — only
    skills 0/0/1 survived in `play-skill*-2026050{5,6}*.mp4`. Re-record after v3 lands).
- `2026-05-06_07:28` — v2 launch: latch + masking + `reward_scale=0.2`. **Aborted at
  iter ~25** by NCCL all-reduce timeout (600 s) because of the rank-divergence bug
  documented above. The fix is now committed to disk in `diversity_ppo.py`.

### Pending: launch v3

Code is patched. Pending re-launch (one of):

```bash
docker exec -e CUDA_VISIBLE_DEVICES=4,5,6,7 isaac-sim bash -lc \
  'source /mnt/storage/lti/activate_conda.sh lti && \
   python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 --master_port 29501 \
     scripts/reinforcement_learning/rsl_rl/train.py \
     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-v0 \
     --num_envs 16384 --logger wandb --headless --distributed \
     env.scene.insertive_object=peg env.scene.receptive_object=peghole' \
  > /mnt/storage/lti/UWLab-ICL/logs/diversity_runs/diversity_4gpu_v3.log 2>&1 &
```

(Logs in `logs/diversity_runs/`, checkpoints in `logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/<new TS>/`.)

---

## Where to look next

- If skills are still visually similar after v3 trains for a few hours, the lever to
  pull is the discriminator's input. The `DiscriminatorObsCfg` group currently mirrors
  the full policy state minus skill (joint pos + ee/asset poses). Narrowing it to just
  the **end-effector trajectory** in the receptive frame would push behavioral diversity
  toward visibly different ee paths instead of joint-config diversity that the camera
  may not capture.
- Other knobs in `Diversity_PPORunnerCfg.algorithm.discriminator_cfg`:
  `reward_scale` (currently 0.2 — bigger = more behavioral diversity, but watch out for
  it dwarfing the task reward), `update_frequency` (currently 4 = every 4 PPO steps),
  `hidden_dims` (currently `[256, 256, 256]`).
- The current actor/critic in `Base_PPORunnerCfg.policy` was sized down by the user mid-
  session to `actor_hidden_dims=[512, 256, 128, 64]`. Critic is still
  `[1024, 512, 256, 128]`. Keep an eye on whether the smaller actor can absorb the
  10-way skill conditioning + base task.

## Files of interest (for grep / future Claude)

- `rsl_rl/rsl_rl/algorithms/diversity_ppo.py`
- `rsl_rl/rsl_rl/runners/diversity_runner.py`
- `rsl_rl/rsl_rl/modules/skill_discriminator.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/diversity.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/agents/rsl_rl_cfg.py`
- `source/uwlab_rl/uwlab_rl/rsl_rl/rl_cfg.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`
