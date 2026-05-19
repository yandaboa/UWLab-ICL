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

## Strategic direction — moving DIAYN to the gravity-based ScenePC task

**Decided 2026-05-18.** All future diversity work moves off the State-Diversity
task and onto the gravity-based point-cloud setup. Two motivations:

1. **Point clouds vs state for symmetric objects.** State-based obs (pose +
   joint positions) breaks the symmetry of objects like the peg: rotation
   around the axis of symmetry produces different state values, so the
   discriminator can chase "symmetry-equivalent" diversity that looks visually
   identical. Point clouds — being inherently a set, not an oriented frame —
   represent symmetric objects symmetrically, so the discriminator can only
   reward *visually distinguishable* behaviour.
2. **Reset distribution.** The current State-Diversity task draws from three
   resets (object-anywhere, partial-assembly, **grasped**). The grasped reset
   spawns the policy in the middle of a successful trajectory, which is
   exactly the regime where the discriminator has nothing to grab onto — the
   handful of remaining steps are all near-success. Patrick's gravity setup
   uses only `[ZeroGAnywhere, ZeroGPartialAssembly]` (50/50), so every episode
   starts from a state where there's still episode-length runway for the
   policy to express a chosen skill.

### Reference: `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Train-v0`

Lives in **`UWLab-patrick-private`** on branch **`pat/dagger-symmetry`** (added
as remote `patrick` in this repo; fetch with `git fetch patrick pat/dagger-symmetry`).

- Cfg class: `ZeroGScenePCSysidSim2RealTrainCfg` in `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/gravity_cfg.py:872`. Registered in that repo's `omnireset/.../__init__.py:168`.
- Agent: `ScenePCPPORunnerCfg` (`agents/rsl_rl_cfg.py:348`) — plain PPO with shared MLP encoder. Obs groups: `policy = [proprio, pointcloud]`, `critic = [proprio, pointcloud, time_left]`. Pointcloud encoder: `[256, 128] → 32d`.
- Obs cfg: `ScenePCObsCfg` (`gravity_cfg.py:120`) — 512-pt point cloud (`PointcloudCfg`), proprio (`GroupCfg`), `time_left`, `success_classifier`.
- Resets: `ZeroGGPSEventCfg.reset_from_states` (`gravity_cfg.py:355`) — `MultiResetManager` over `["ZeroGAnywhere", "ZeroGPartialAssembly"]` with the GPS curriculum.
- Curriculum: `gravity_curriculum` (`gravity_cfg.py:333`) — the gravity-trick (ramps gravity from 0 toward -9.81 as success rate improves). Typical Hydra overrides: `env.curriculum.gravity_curriculum.params.reduction=monitor_mean env.curriculum.gravity_curriculum.params.floor=0.1`.
- Termination: terminate on success (`consecutive_success_state` for eval).
- Action: `Ur5eRobotiq2f85RelativeOSCEvalAction`, robot `EXPLICIT_UR5E_ROBOTIQ_2F85`.

### Port plan — open design questions

Before any code moves, decide:

1. **Where does the new diversity code live?**
   - **(A) Inside `UWLab-patrick-private`**, as a `gravity_diversity_cfg.py` that extends `ZeroGScenePCSysidSim2RealTrainCfg` with a skill obs + diversity reward. We'd port `diversity.py` MDP terms, `DiversityPPO`/`DiversityRunner` algorithm + module files into Patrick's repo. **Smaller diff** because the gravity ecosystem stays where it is; ~5-10 files copied in.
   - **(B) Inside this repo (`UWLab-ICL`)**, by porting `gravity_cfg.py` and its missing dependencies in. Missing here: `GravityCurriculum`, `ScenePointCloud`, `HeuristicsCfg`, `object_out_of_bound`, and the actor/critic encoder machinery. Bigger diff (~20+ files), invasive merge.
   - **(C) Merge `patrick/pat/dagger-symmetry` into `adaptive`** outright. Cleanest from a "single repo" perspective but pulls in everything (multitask, rgb_dagger, depth_dagger, …) and likely conflicts with current `adaptive` changes.

   Default recommendation: **(A)**. Patrick's repo is where the gravity-based eco-system is canonical; the diversity layer is small enough to live there.
2. **Discriminator input.** Symmetry was the motivation, so `discriminator_obs` should include the point cloud. Cheapest path: feed the `pointcloud` group through a dedicated copy of the same `pointcloud` encoder used by the actor/critic (separate weights, no leakage). Open: should the discriminator also see proprio? My instinct is **point cloud only** — we want diversity in *object-relative behaviour*, not in arm pose.
3. **Post-success latch.** Gravity task terminates on success, so episodes end before the policy has many post-success steps. The latch may be largely redundant, but the masking code is generic and cheap to keep. Recommendation: **keep it** (don't re-invite the rank-divergence NCCL bug, gotcha #4).
4. **Skill alphabet size.** With the 2-skill State-Diversity setup converging in ~1500 iters (v1), we have headroom. Start at `num_skills=2` to validate the port, then sweep up (4, 8) once we confirm the discriminator gets signal.
5. **Weight & gravity-curriculum interaction.** The gravity-curriculum ramps difficulty as success rate climbs. With a strong diversity weight, success rate plateaus low (Sweep-B was stuck at 0.11), which would *prevent gravity from ramping*. The curriculum's `floor` knob (`env.curriculum.gravity_curriculum.params.floor=0.1`) sets a minimum gravity so the policy still sees the deployment regime — important to set this when running diversity.

### Files to scout / decide on before porting

- `UWLab-patrick-private/source/uwlab_tasks/.../omnireset/mdp/` — does it have `RewardManager.success` plumbing equivalent to what `DiversityReward`'s latch reads? Patrick's term names look similar (`progress_context`, `consecutive_success_state`) but I haven't traced `progress_context.success` end-to-end on that branch yet.
- `UWLab-patrick-private/rsl_rl/` (if a separate submodule on that branch) — does it already have any algorithm extensions we'd collide with?
- The pointcloud encoder used by `RslRlActorCriticWithEncoderCfg` — confirm we can instantiate a second copy for the discriminator with the same hidden_dims/output_dim.

## 3. Active runs

### Gravity-PCObjectsOnly-w5.0 — current run (running, started 2026-05-19 ~10:15)

Discriminator obs is a **robot-stripped scene PC** (256 insertive + 256
receptive, 0 robot). Closes the arm-position cheat-channel that the
State-Diversity probe identified — the only signal the discriminator can
extract is *how the objects move*, not how the arm is configured. This is
the right test of whether DIAYN on this task can learn behaviour-conditioned
diversity rather than arm-configuration diversity.

- **Task**: `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Diversity-PCObjectsOnly-Train-v0`
- **GPUs**: 4-7, port 29503.
- **Log**: `UWLab-patrick-private/logs/diversity_runs/gravity_pc_objects_only_w5p0.log`.
- **Knobs**: same as the previous run (`--seed 23`, `--num_envs 32768`,
  `agent.num_steps_per_env=16`, K=2, `env.rewards.diversity.weight=5.0`,
  gravity curriculum `reduction=monitor_mean`, `floor=0.1`).
- **Iter 0 status**: ~20s/iter cold-start, all 4 GPUs at 80-91% util.
  Confirmed two ScenePointCloud instances build correctly:
  policy=(285 robot, 87 ins, 140 rec), discriminator=(0, 256, 256).
- **Babysit loop**: NOT restarted. User can /loop again if needed.

Branch state in `UWLab-patrick-private` (still on `yanda/diversity-gravity`,
not pushed): 4 commits — the original port, the patlab compat fixes, the
`storage.observations` fix, and the **discriminator-obs-group refactor +
PCObjectsOnly variant**.

### Gravity-PC-w5.0 — superseded 2026-05-19 ~10:15

Original run on `…-Diversity-PC-Train-v0`. Killed after ~7h at iter ~558
to free GPUs for the PCObjectsOnly variant. Disc was still at exact log 2
(loss=0.6931, acc=0.50) for the entire 7h — confirming the State-Diversity
finding that with arm-config in the disc obs, the disc has a "rich" enough
input that it never bootstraps off behaviour. Log preserved at
`UWLab-patrick-private/logs/diversity_runs/gravity_pc_only_w5p0_v2.log`.

Three commits in branch `yanda/diversity-gravity` (the port itself, the
patlab-PPO compat fixes, and the `storage.observations` fix) preceded the
refactor described above.

- **Task**: `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Diversity-PC-Train-v0`
- **GPUs**: 4-7, port 29503.
- **Log**: `UWLab-patrick-private/logs/diversity_runs/gravity_pc_only_w5p0_v2.log` (v2 = after the iter-3 `policy_observations` crash fix; the original `_v0` log preserved for postmortem).
- **Discriminator obs**: pointcloud only (the symmetry-preserving choice).
- **Discriminator arch**: separate-weights copy of the policy's PC encoder
  (`[256, 128] → 32d`) + 3×256-MLP classification head.
- **Knobs**: `--seed 23`, `--num_envs 32768`, `agent.num_steps_per_env=16`,
  `K=2`, `env.rewards.diversity.weight=5.0` (matches the State-Diversity
  Sweep-A working point), `discriminator_cfg.reward_scale=0.2` (cfg default),
  gravity curriculum at the standard pretraining knobs
  (`reduction=monitor_mean`, `floor=0.1`).
- **Iter 6 status (2026-05-19 ~01:50)**: ~13-15s/iter, all 4 GPUs (4-7) at
  high util (~43-47 GB each). Disc first update at iter 3 (loss=0.69,
  acc=0.50 — exactly chance for K=2 with random init). Diversity reward
  small negative (~-0.005, disc still learning). EOE ~0.06 (cold-start).

#### Tick log

(timestamp / iter / iter time / disc loss·acc / Episode_Reward/diversity / EOE / note)

- **2026-05-19 ~01:50 (tick 0, launch)** — iter 6/40000, ~13-15s/iter, disc loss=0.69 / acc=0.50 (first update at iter 3, exactly chance for K=2 random init), diversity bonus ~-0.005, EOE 0.06. GPUs 4-7 at 88-89% util, ~43-47 GB. Healthy.
- **2026-05-19 ~02:13 (tick 1)** — iter 85/40000, ~16s/iter, disc loss=0.6931 / acc=0.5045 (**still at chance after 21 disc updates** — flagging), diversity bonus volatile and large-negative (-2 to -8 typical, one outlier of -927 at iter 84), EOE 0.17 (task starting to learn). GPUs 4-7 at 54-90%, ~43-47 GB. Healthy mechanically, but the disc isn't differentiating skills — classic DIAYN chicken-and-egg if the policy hasn't diverged enough for q(z|s) to find signal. Worth watching for another 1-2 ticks before declaring a research problem.
  - **Sweep-A status change**: now at iter 3540, EOE **0.99** (was 0.77 at iter 2535) — converged at near-perfect task success. Diversity bonus stable at 0.12. Mirrors v1's EOE trajectory but with sustained diversity signal (v1 collapsed to 0.004; A holds at ~0.12).
- **2026-05-19 ~02:43 (tick 2)** — iter 197/40000, ~16s/iter, disc loss=**0.6931** / acc=**0.5028** (=log 2 to 4 decimals, **stuck at chance after ~49 updates**), diversity bonus -0.8 to -11 (still large-negative & volatile), EOE 0.16-0.17 — **flat for 112 iters / 30 min, task also stuck**. GPUs healthy (86-91%, ~43-47 GB). The disc is failing to extract any signal from the 1536d PC obs; in State-Diversity v1 (also K=2) the disc had bonus 0.10 by iter 200, so this is much worse. Hypothesis: with K=2 + random-init shared policy, both skills produce ~identical PC obs distributions and the MLP-on-flat-PC encoder can't pick up subtle behavioural differences. Decision: keep watching through tick 4 (~iter 400); if disc loss still = log 2, surface to user — likely need bigger disc encoder / lower diversity weight to break the chicken-and-egg, or PC+proprio variant.
- **2026-05-19 ~03:13 (tick 3)** — iter 310/40000, ~16s/iter, disc loss=**0.6932** / acc=**0.5032** (same as tick 2, ~77 updates total), diversity bonus -0.83 to -20.5 (another big outlier), EOE **0.16-0.17 — still flat, 225 iters with zero progress on either metric**. GPUs healthy (86-90%). 3 ticks of identical-to-4-decimals disc-at-chance is very strong evidence the disc has collapsed; one more tick of confirmation then I'll surface + recommend a kill.
- **2026-05-19 ~03:43 (tick 4, FINAL — research stall, surfacing)** — iter 421/40000, disc loss=**0.6931** / acc=**0.5021** (~105 updates, identical to ticks 1-3), diversity -0.65 to -4.5, EOE 0.16-0.17 still flat (336 iters / 90 min of zero metric progress). Run is mechanically fine but research-dead. **Stopping the babysit loop (CronDelete) and surfacing via PushNotification.** The training proc is still alive on GPUs 4-7 — user needs to decide whether to kill it. See "Postmortem & next-step options" below.

#### Update 2026-05-19 ~06:40 — partial walkback

User clarified that **the gravity task naturally takes 8-12 hours of training
before EOE starts climbing** — so the EOE-flat-at-0.16 observation through
ticks 1-4 is **normal**, not a stall. The remaining concerning signal is
just the disc-loss-stuck-at-log(2). That alone is not enough to kill the
run; the disc may still come online once the policy starts diverging
(8-12h+). Gravity training proc 1545334 left running on GPUs 4-7. Babysit
loop remains stopped — restart with adjusted criteria (longer
intervals, don't alarm on EOE flatline) if desired.

#### Postmortem & next-step options — Gravity-PC-w5.0 research stall

**Symptom**: 4 ticks (105 disc updates / iter 421) of disc loss = log 2
(0.6931–0.6932) and accuracy ≈ 0.50 — i.e. the discriminator has
collapsed to constant ~uniform logits and produces no skill signal. The
policy can't differentiate skills because the disc gives no useful
gradient, so EOE has plateaued at 0.16-0.17 for 336 iters.

**Probable cause**: chicken-and-egg with a *very* hard-to-bootstrap
representation. With K=2 + a fresh shared policy, both skills initially
produce ~identical PC obs distributions. The MLP-on-flat-1536d encoder
collapses to a constant feature output rather than learning subtle
behavioural differentiators. Once the disc is at uniform output, the
diversity reward is symmetric (mean 0, but with huge negative outliers
for the wrong-skill tail), which destabilises policy learning further
*without* providing differentiation pressure. Net result: stable bad
equilibrium.

**Aux symptom**: diversity bonus has frequent large-negative outliers
(-7, -20, -927 at one point) — these come from rare, very-confident-wrong
disc predictions on individual transitions. They hit the policy gradient
hard and likely contribute to the flat EOE.

**Comparison reference**: State-Diversity v1 (UWLab-ICL, K=2, weight=1.0)
had disc reward 0.10 by iter 200 and EOE rising. State obs gives the
disc easy fingerprints (joint pos / ee pose differ slightly across
skills from the very first step) that PC obs doesn't.

**Next-step options** (cheapest-first, to ask user to pick):
1. **Lower diversity weight to 1.0** — kills the outlier-driven policy
   destabilization. Let the task learn first, then crank weight back up
   later. `env.rewards.diversity.weight=1.0` Hydra override.
2. **Try PCProprio variant** — `…-Diversity-PCProprio-Train-v0`. Proprio
   gives the disc easy fingerprints to bootstrap on; once it's learned
   to separate skills using proprio, gradient flows into the PC encoder
   too.
3. **Bigger disc encoder** — `discriminator_cfg.encoder_groups.pointcloud.
   hidden_dims=[512, 256, 128]` + `output_dim=64`. Currently it's
   `[256, 128] → 32` (mirrors policy encoder).
4. **Replace the MLP encoder with a real PointNet** — invasive; only
   worth it if 1-3 all fail.

Recommendation: try **option 2 first** (PCProprio) — it directly tests
the "PC alone can't break symmetry from cold start" hypothesis without
touching code. Run on the same GPUs after killing the current run.

#### Postmortem — `gravity_pc_only_w5p0.log` (the v0 crash)

First launch crashed at iter 3 with
`AttributeError: 'RolloutStorage' object has no attribute 'policy_observations'`.
Root cause: the original DiversityPPO port read skill labels from
`storage.policy_observations`, which exists in UWLab-ICL's rsl_rl fork but
not in patlab's. The skill obs lives as a top-level group, so reading from
`storage.observations` works. Fixed and relaunched as v2. **Lesson for the
next port: always check whether a UWLab-ICL-only storage attribute exists
in the target fork before porting.**

### Sweep-A — `weight=5.0`, `num_skills=2` (killed 2026-05-19, converged)

- **Launched**: 2026-05-18, GPUs 0-3, port 29501. Killed 2026-05-19 ~06:40 at iter ~3820 (~19h training), EOE 0.99 / div 0.11 — converged.
- **Log**: `logs/diversity_weight_sweep/weight5p0_gpu0-3.log`.
- **Checkpoint dir**: `logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-18_11-02-44/`; final ckpt `model_4000.pt`.
- **Skill rollout videos** (2026-05-19, same `--seed 42` for comparable resets, 500 steps each):
  - `videos/play/play-skill0-20260519-064900-step-0.mp4`
  - `videos/play/play-skill1-20260519-065211-step-0.mp4`
  - **`play.py` quirk**: must override `env.observations.policy.skill.params.num_skills=2 agent.algorithm.number_of_skills=2` (matching the training overrides) — otherwise the Play cfg defaults to `num_skills=10` and the actor input shape mismatches the trained model by `(10-2)×history=40` dims.
  - **Host quirk**: `play.py` crashes at renderer init on GPU 0 (`libgpu.foundation.plugin` fatal at 99ms, before any user code). Use any non-zero GPU for play rollouts.

Status entries (timestamp / iter / `Ep_Rew/diversity` / EOE / decision):

- **2026-05-18 launch** — iter 0, diversity -0.004, EOE 0.0009. Normal cold-start.
- **2026-05-18 ~iter 2515** — diversity 0.26, EOE 0.67, ~18s/iter. Sustained diversity signal (vs v1's collapse to 0.004) but ~30 pp below v1's task success. Healthy; let it keep running.
- **2026-05-19 ~iter 3820** — diversity 0.11, EOE 0.99. Converged. Killed to free GPUs and record skill videos.

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

## Empirical evidence the task admits multiple solutions

2026-05-19: User confirmed by visual inspection that **two seeds of the
same `…-ZeroG-ScenePC-SysID-Train-v0` training (seeds 22 and 23, in
`UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/`
`2026-05-14_09-29-21/` and `2026-05-16_06-55-03/`) end up with policies
that pick up the object differently**. Same task, same hyperparameters,
same reset distribution — different basin.

**Why this matters for DIAYN**:

* This is direct evidence the task has multiple natural solutions, not a
  unique attractor. So a discriminator-driven loss *should* be able to
  separate them.
* The chicken-and-egg / cold-start problem for DIAYN here isn't "the task
  has no diverse solutions to find" — it's purely a representation /
  bootstrapping issue. The signal exists; the discriminator just has to
  pick it up.
* Eval videos for direct comparison:
  - `UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-16_06-55-03/videos/play/rl-video-step-0.mp4` (seed 23, model_10500)
  - `UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-14_09-29-21/videos/play/rl-video-step-0.mp4` (seed 22, model_8800)
  - Same eval task, same env-seed at play time (42) so resets line up; any visible behavior difference is policy-induced.

This contrasts with the State-Diversity analysis (2026-05-19) where the
discriminator hit ~87% accuracy without producing visually distinct
skills — the per-step state distributions differed, but the geometric
trajectories overlapped. The seed-22 vs seed-23 comparison says we
*can* get visually distinct policies on this task; the DIAYN setup
just needs to *find* them rather than collapsing to one.

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
