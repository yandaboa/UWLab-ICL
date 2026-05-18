# Diversity (DIAYN) experiments

Central log for DIAYN-style diversity experiments on the OmniReset peg-insertion
task. **This is the single source of truth for diversity work.** Read this
before launching, killing, or interpreting a diversity run. Append new sections
and status entries here — do not spawn per-experiment markdown files.

The doc has four parts:
1. **Setup** — how DIAYN is wired into this repo (what's built, where it lives).
2. **Operating** — launch templates, how to kill, knobs, gotchas.
3. **Active runs** — what's training right now.
4. **Experiment history** — newest-first, with verdicts.

---

## 1. Setup — DIAYN in this repo

### Pieces

| Layer | Path | What |
|---|---|---|
| Algorithm | `rsl_rl/rsl_rl/algorithms/diversity_ppo.py` | `DiversityPPO(PPO)`. Owns the discriminator + AdamW opt, lazy-builds in `init_storage`, runs CE update every `update_frequency` PPO steps. Multi-GPU: weight broadcast + gradient all-reduce. Exposes `attach_env(env)` so the env-side reward term can call the discriminator. |
| Module | `rsl_rl/rsl_rl/modules/skill_discriminator.py` | `SkillDiscriminator` MLP for `q(z\|s)`, with optional `EmpiricalNormalization`. |
| Runner | `rsl_rl/rsl_rl/runners/diversity_runner.py` | `DiversityRunner(OnPolicyRunner)`. Calls `attach_env` after construction; persists discriminator state in `save`/`load`. |
| MDP terms | `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/diversity.py` | `SkillObs` (per-env skill buffer, one-hot to policy, supports `force_skill: int` for eval). `DiversityReward` (computes `log q(z\|s) - log p(z)`, scaled by `env.diversity_reward_scale`, with a per-env post-success latch). `diversity_task_done_obs` (latch as obs for the `diversity_meta` group). |
| Env cfg | `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py` | `Ur5eRobotiq2f85RelCartesianOSCDiversityTrainCfg` + `…PlayCfg`. Adds `DiversityObservationsCfg` (one-hot skill in policy obs, separate `discriminator_obs` group with no skill leakage, `diversity_meta` carrying the latch) and `DiversityRewardsCfg`. `NUM_SKILLS` at `rl_state_cfg.py:736` must match agent-side `number_of_skills`. |
| Task IDs | `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/__init__.py:91-108` | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-v0` (train) and `…-State-Diversity-Play-v0` (eval). Both wired to `Diversity_PPORunnerCfg` in `agents/rsl_rl_cfg.py:103`. |
| Train entry | `scripts/reinforcement_learning/rsl_rl/train.py` | Dispatches `DiversityRunner` when `agent_cfg.class_name == "DiversityRunner"`. |
| Play entry | `scripts/reinforcement_learning/rsl_rl/play.py` | `--skill <int>` sets `force_skill` + tags video. Loads discriminator weights when the ckpt contains `discriminator_state_dict`. |

### The post-success latch (why it exists)

`DiversityReward` maintains a per-env `task_done` latch (True once
`progress_context.success` fires within an episode). Once latched, the
diversity bonus is zeroed for that env until reset, and the discriminator's
CE update masks those transitions out. Without this, the policy gets free
diversity reward by doing arbitrary things *after* solving the task, and the
discriminator over-fits to post-success states.

**Reward-term ordering constraint:** `RewardsCfg` lists `progress_context`
before the diversity term, so `progress_context` fires first and DIAYN sees
the current-step success bit. **Don't reorder the cfg.**

### Knobs

| Override | What | Default |
|---|---|---|
| `env.rewards.diversity.weight=<W>` | Scales the diversity reward term (`rl_state_cfg.py:832`) | `1.0` |
| `env.observations.policy.skill.params.num_skills=<K>` | Size of the skill alphabet (must match agent-side `number_of_skills`) | `10` (`NUM_SKILLS` in `rl_state_cfg.py:736`) |
| `agent.algorithm.number_of_skills=<K>` | Algorithm-side skill count | `10` |
| `agent.num_steps_per_env=<N>` | Rollout length per env per iter | see `Diversity_PPORunnerCfg` in `agents/rsl_rl_cfg.py:103` |
| `env.observations.policy.skill.params.force_skill=<idx>` | Lock all envs to skill `idx` at eval time | `-1` (= uniform random) |
| `agent.algorithm.discriminator_cfg.reward_scale` | In-algorithm scaling applied alongside the env weight | (see runner cfg) |
| `agent.algorithm.discriminator_cfg.update_frequency` | CE update every N PPO steps | 4 |
| `agent.algorithm.discriminator_cfg.hidden_dims` | Discriminator MLP shape | `[256, 256, 256]` |

`DiscriminatorObsCfg` (the input to `q(z\|s)`) currently mirrors the full
policy state minus skill (joint pos + ee/asset poses). **If skills end up
visually similar**, narrow this to just the end-effector trajectory in the
receptive-object frame — pushes behavioural diversity toward visibly different
ee paths instead of joint-config diversity the camera may not capture.

---

## 2. Operating

### Launch template

```bash
nohup docker exec -e CUDA_VISIBLE_DEVICES=<GPUS> isaac-sim bash -lc \
  'source /mnt/storage/lti/activate_conda.sh lti && \
   python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 --master_port <PORT> \
     scripts/reinforcement_learning/rsl_rl/train.py \
     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-v0 \
     --num_envs 32768 --logger wandb --headless --distributed \
     env.scene.insertive_object=peg env.scene.receptive_object=peghole \
     env.observations.policy.skill.params.num_skills=<K> \
     agent.algorithm.number_of_skills=<K> \
     agent.num_steps_per_env=16 \
     env.rewards.diversity.weight=<W>' \
  > logs/diversity_runs/<run_name>.log 2>&1 &
```

- `<PORT>`: unique master port per concurrent run (29501, 29502, …). Default
  29500 collides with Patrick's training (see gotcha #2).
- Boot to iter 0 takes ~60-90s. Mimic-joint `[Error]` lines during prim load
  are non-fatal noise.
- All run logs live under `logs/diversity_runs/`. (The May 2026 sweep
  temporarily used `logs/diversity_weight_sweep/`; treat that as a sub-folder.)

### Per-skill 500-step video rollouts (single GPU)

```bash
CKPT=/mnt/storage/lti/UWLab-ICL/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/<TS>/model_<N>.pt
docker exec -e CUDA_VISIBLE_DEVICES=4 isaac-sim bash -lc "source /mnt/storage/lti/activate_conda.sh lti && \
  for skill in $(seq 0 $((K-1))); do
    python scripts/reinforcement_learning/rsl_rl/play.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-Play-v0 \
      --num_envs 1 --headless --enable_cameras \
      --video --video_length 500 --skill \$skill \
      --checkpoint \$CKPT \
      env.scene.insertive_object=peg env.scene.receptive_object=peghole
  done"
```

`act_inference` returns the deterministic Gaussian mean (no sampling) since
`state_dependent_std=False`, so different skills should produce different
deterministic trajectories. Videos land in
`logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/<TS>/videos/play/`.

**Don't pass `--action_rescale`** for Diversity (Stage-1) policies — see
gotcha #3.

### Killing a run

Host `kill` doesn't work because the workers are in a docker namespace; host
PIDs don't translate inside the container. Find in-container PIDs and kill
from inside:

```bash
docker exec isaac-sim ps -eo pid,cmd | grep "train.py.*Diversity" | grep -v grep
docker exec isaac-sim kill <pid> <pid> ...
```

GPU memory takes ~15-30s to release after the workers die.

### Gotchas (don't re-hit these)

1. **`sanitize_rsl_rl_cfg` strips kwargs not explicit in the algorithm signature.** Forwarding parent-PPO args via `**kwargs` makes them invisible. Declare all parent-PPO kwargs explicitly in `DiversityPPO.__init__` (mirror BCPPO).
2. **Default `master_port` 29500 collides** with Patrick's GPU 0-3 training. Always pass `--master_port 29501+`.
3. **`play.py`'s `--action_rescale` is a footgun for Stage-1 policies.** `get_perturbed_env_solving_actions` used to unconditionally inverse-rescale to the **finetune-eval scale** `[0.01, 0.01, 0.002, 0.02, 0.02, 0.2]`. Stage-1 / Diversity policies train at `[0.02, 0.02, 0.02, 0.02, 0.02, 0.2]`, so xyz commands were silently shrunk 0.5× and z by 0.1× (catastrophic for insertion). Now off by default; only pass `--action_rescale` for finetune-eval policies.
4. **NCCL deadlock if discriminator update has rank-divergent batch sizes.** The post-success masking has to use a per-sample weight (`weight = (~prev_dones) & (~prev_latch)` then weighted CE), not filter-and-minibatch — otherwise different ranks run different numbers of all-reduces and stall at the 600s watchdog. The fix is in `diversity_ppo.py`.
   > **Rule for any future change to `_update_discriminator_from_storage`**: every rank must take the same control-flow path through the all-reduces. No `return None` based on rank-local rollout state.
5. **`EmpiricalNormalization` running stats are NOT all-reduced** (same as actor/critic norms). Small cross-rank drift is possible but not load-bearing; discriminator weights stay in sync via gradient all-reduce + weight broadcast.
6. **Reward-term ordering matters.** `progress_context` must run before `DiversityReward` so the success bit is current when the latch updates. `RewardsCfg` field order in `rl_state_cfg.py` enforces this. Don't reorder.

---

## 3. Active runs

### Sweep-A — `weight=5.0`, `num_skills=2` (running)

- **Launched**: 2026-05-18, GPUs 0-3, port 29501.
- **Log**: `logs/diversity_weight_sweep/weight5p0_gpu0-3.log`.
- **Wandb experiment_name**: `ur5e_robotiq_2f85_omnireset_diversity` (`agents/rsl_rl_cfg.py:111`); run timestamped under that.
- **Current in-container launcher PID**: re-check with `docker exec isaac-sim ps -eo pid,cmd | grep "weight=5.0"`.

Status entries (timestamp / iter / `Ep_Rew/diversity` / EOE / decision):

- **2026-05-18 launch** — iter 0, diversity -0.004, EOE 0.0009. Normal cold-start.
- **2026-05-18 ~iter 2515** — diversity 0.26, EOE 0.67, ~18s/iter. Sustained diversity signal (vs v1's collapse to 0.004) but ~30 pp below v1's task success. Healthy; let it keep running.

---

## 4. Experiment history (newest first)

### Sweep-B — `weight=10.0`, `num_skills=2` (killed 2026-05-18)

- Launched alongside Sweep-A on GPUs 4-7, port 29502.
- **Killed at ~iter 2569**: EOE stuck at 0.11 while v1 (`weight=1.0`) had cleared 50% by iter 800. Diversity reward 1.12 was huge but the policy was clearly farming the bonus and ignoring the task.
- **Verdict**: `weight=10.0` is too aggressive for `num_skills=2` on this task. The diversity bonus needs to be small enough that the task reward can still drive learning past the success threshold.
- Log preserved at `logs/diversity_weight_sweep/weight10p0_gpu4-7.log`.

### 2-skill v1 — `weight=1.0`, `num_skills=2` (superseded 2026-05-18)

- **Started 2026-05-16**, killed at iter ~1680 on 2026-05-18 to free GPUs 0-3 for the weight sweep.
- **Setup**: 4 GPUs (0-3), `num_envs=32768`, `num_steps_per_env=16`. The user also edited `Diversity_PPORunnerCfg.num_steps_per_env=16` directly in `agents/rsl_rl_cfg.py`; the Hydra override was belt-and-braces.
- **Headline curve**:
  - iter 0 → 200: diversity reward 0.002 → 0.10 (discriminator learning fast with only 2 classes).
  - iter 200 → 1000: diversity plateau ~0.07-0.10; EOE crept 0.14 → 0.68.
  - iter 1000 → 1500: EOE jumped 0.68 → 0.98 in ~500 iters; **diversity reward collapsed** 0.04 → 0.005 as the latch zeroed more and more of each episode.
  - iter 1500 → 1680: EOE held at ~0.98-0.99, diversity ~0.004. Task essentially solved.
- **Verdict / why superseded**: even though task success is near-perfect, the discriminator has almost no non-masked data to learn from once success arrives in the first few env steps. Whole point of DIAYN is the diversity bonus, so we need to push the weight harder.
- **Iter time**: ~17-19s steady (versus the older 10-skill v3 run's ~10s/iter at half the env count, double the rollout length; same total volume, so the gap is sync overhead).

### 10-skill v3 — pending re-launch as of 2026-05-06 handoff (not started under this name)

After the v2 NCCL fix, a v3 re-launch (`num_envs=16384`, GPUs 4-7, port 29501)
was queued in the handoff but the 2-skill experiments took priority. If we
come back to 10 skills, that command is the right starting point.

### 10-skill v2 — `weight=?` (default), `num_skills=10` (aborted 2026-05-06)

- Launch: `2026-05-06_07:28` with the latch + masking and `reward_scale=0.2`.
- **Aborted at iter ~25** by NCCL all-reduce timeout (600 s) — root cause was rank-divergent batch sizes in the discriminator update (gotcha #4). Fix is committed in `diversity_ppo.py`.

### 10-skill v1 — defaults, `num_skills=10` (killed 2026-05-05)

- Launch: `2026-05-05_12-28-43`, first full 4-GPU run, `num_envs=16384`, port 29501.
- Reached iter ~5400 / 40000 (~9 hours). Killed by user.
- **Final metrics**: EOE success ~0.42, `Episode_Reward/diversity ≈ 0.20`.
- **Subjective issue raised**: the 10 skills produced visually similar trajectories. Hypothesis at the time: most of the reward signal came from late-episode (post-success) frames where the policy is free to do whatever — discriminator learns to classify those, not the task-relevant prefix. That hypothesis motivated the latch+masking fix that landed in v2/v3 code.
- **Videos** under `videos/play/` for `model_5400.pt` — only skills 0/0/1 survived (`play-skill*-2026050{5,6}*.mp4`); the rest were lost in the rescale-on-vs-off transition. Re-record after any future 10-skill run lands.

### Smoke test (2026-05-05)

`2026-05-05_11-47-15` — first 2-iter smoke (8 envs, 1 GPU): pipeline boots,
discriminator fires, diversity reward registers. End-to-end validated.

---

## Open questions / followups

- **Are the skills *behaviorally* distinct?** No run so far has verified this — high `Ep_Rew/diversity` is necessary, not sufficient. The discriminator could be exploiting irrelevant state. Eval path: roll out each skill with `play.py --skill <idx>` and compare trajectories, or build a `analysis/eval_discriminator.py` that scores rollouts against the trained `q(z|s)`.
- **Sweep-A landing point.** If EOE keeps climbing past 0.67 with `weight=5.0`, that's the sweet spot. If it plateaus low while diversity stays high, we may need an even smaller weight (try 2.0 or 3.0) — or to narrow `DiscriminatorObsCfg` to the ee trajectory only, so the discriminator can't cheat off joint config.
- **Policy capacity.** Actor was sized down to `[512, 256, 128, 64]` mid-session; critic is still `[1024, 512, 256, 128]`. If we go back to 10 skills, watch whether the smaller actor can absorb 10-way skill conditioning + base task.

---

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
