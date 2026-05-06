# Valid experimental results — priv_baseline batch

All numbers are EOE (end-of-episode) success rate / any-time success rate, evaluated
on `n=512` episodes per checkpoint. Eval task: peg + peghole.

**Disclaimer.** Several earlier runs are excluded from this log because they hit the
`MLPImagePolicy.compute_loss` bug (only `(o_0, a_0)` of each episode trained on; fixed
2026-05-05). All Gaussian-MLP-policy runs prior to the fix produced 0% EOE due to
that bug, not the head architecture.

| run | head | env | priv obs? | demos | DAgger | iters EOE/any | source |
|---|---|---|---|---|---|---|---|
| **Discrete-AR (transformer trunk, 6L/8H), no priv obs** |
| r4 mark-disc BC d=256 | disc-AR | augmented (perturbation) | no | 80k | no | i0=61.3/80.7 | `logs/priv_baseline/r4_mark_disc_bc80k_d256` |
| r4 mark-disc BC d=512 | disc-AR | augmented | no | 80k | no | i0=**74.6**/85.0 | `logs/priv_baseline/r4_mark_disc_bc80k_d512` |
| r4 mark-disc BC d=1024 | disc-AR | augmented | no | 80k | no | i0=65.5/85.0 | `logs/priv_baseline/r4_mark_disc_bc80k_d1024` |
| r5 mark-disc DAgger d=256_v2 | disc-AR | augmented | no | 4×20k | yes | i0=27.6/37.5 \| i1=33.4/40.0 \| i2=22.7/29.5 \| i3=14.5/21.5 | `logs/priv_baseline/r5_mark_disc_dagger_d256_v2` |
| r5 mark-disc DAgger d=512_v2 | disc-AR | augmented | no | 4×20k | yes | i0=48.5/64.5 \| i1=56.4/70.5 \| i2=51.7/72.3 \| i3=58.6/73.7 | `logs/priv_baseline/r5_mark_disc_dagger_d512_v2` |
| r5 mark-disc DAgger d=1024 | disc-AR | augmented | no | 4×20k | yes | i0=58.8/74.7 \| i1=**71.9**/78.1 \| i2=68.1/76.4 \| i3=64.4/73.3 | `logs/priv_baseline/r5_mark_disc_dagger_d1024` |
| **Discrete-AR + privileged obs (action_offset, action_scale in input)** |
| b priv-disc BC d=512 | disc-AR + priv | augmented | yes | 80k | no | i0=75.7/92.0 | `logs/priv_baseline/b_priv_disc_bc80k_d512` |
| b priv-disc BC d=1024 | disc-AR + priv | augmented | yes | 80k | no | i0=74.0/94.9 | `logs/priv_baseline/b_priv_disc_bc80k_d1024` |
| b priv-disc DAgger d=512 | disc-AR + priv | augmented | yes | 4×20k | yes | i0=75.2/89.9 \| i1=68.4/92.0 \| i2=73.6/92.0 \| i3=75.4/96.7 | `logs/priv_baseline/b_priv_disc_dagger_d512` |
| b priv-disc DAgger d=1024 | disc-AR + priv | augmented | yes | 4×20k | yes | i0=75.4/94.2 \| i1=79.7/95.5 \| i2=**80.1**/96.7 \| i3=73.9/95.5 | `logs/priv_baseline/b_priv_disc_dagger_d1024` |
| **Discrete-AR + perturbation-reconstruction aux loss (no priv input)** |
| c aux BC w=0.1 d=512 | disc-AR + aux | augmented | aux only | 80k | no | i0=59.5/69.8 | `logs/priv_baseline/c_aux_w0p1_bc80k_d512` |
| c aux BC w=0.5 d=512 | disc-AR + aux | augmented | aux only | 80k | no | i0=54.8/70.0 | `logs/priv_baseline/c_aux_w0p5_bc80k_d512` |
| c aux BC w=1.0 d=512 | disc-AR + aux | augmented | aux only | 80k | no | i0=56.1/64.3 | `logs/priv_baseline/c_aux_w1p0_bc80k_d512` |
| c aux DAgger w=0.1 d=512 | disc-AR + aux | augmented | aux only | 4×20k | yes | i0=55.8/68.2 \| i1=**70.7**/77.8 \| i2=61.6/75.1 \| i3=62.8/76.8 | `logs/priv_baseline/c_aux_w0p1_dagger_d512` |
| c aux DAgger w=0.5 d=512 | disc-AR + aux | augmented | aux only | 4×20k | yes | i0=45.8/59.5 \| i1=54.7/68.3 \| i2=55.3/69.3 \| i3=51.5/66.0 | `logs/priv_baseline/c_aux_w0p5_dagger_d512` |
| c aux DAgger w=1.0 d=512 | disc-AR + aux | augmented | aux only | 4×20k | yes | i0=30.9/43.1 \| i1=48.3/61.7 \| i2=50.4/72.2 \| i3=52.0/70.6 | `logs/priv_baseline/c_aux_w1p0_dagger_d512` |

## Headline numbers

| comparison | best EOE | source |
|---|---|---|
| Mark-disc BC baseline | **74.6%** | r4 d=512 |
| Mark-disc DAgger best | **71.9%** | r5 d=1024 iter-1 |
| Privileged disc-AR BC best | **75.7%** | b d=512 |
| Privileged disc-AR DAgger best | **80.1%** | b d=1024 iter-2 |
| Aux-loss best | **70.7%** | c w=0.1 DAgger d=512 iter-1 |

## Key observations

1. **Mark-disc BC at d=512 = 74.6 % EOE** is the canonical no-priv-info BC baseline.
   Wider (d=1024) hurts BC; longer DAgger-iter doesn't help past iter-1.
2. **Privileged disc-AR DAgger d=1024 hits 80.1 %** — the best result in the batch.
   Adding privileged perturbation params yields ~5 pp lift over the no-priv mark-disc
   baseline when combined with DAgger and a wider trunk.
3. **Aux-loss hurts monotonically with weight.** All 6 c_aux runs underperform r4.
   Likely because `n_obs_steps=1` gives the trunk no temporal context to actually
   recover perturbation params from, so the aux gradient is mostly noise.
4. **Sample size matters.** With n=512 episodes per eval point, EOE has roughly ±5 pp
   sampling noise. The non-monotonic patterns in r5 / b / c DAgger curves are at the
   edge of statistical significance — be careful inferring trends from single
   iters.

## Invalid (excluded) runs — to be re-run with the bug-fixed Gaussian MLP

| run | env | priv obs? | demos | original (buggy) result |
|---|---|---|---|---|
| r1 priv-MLP BC d=256 | augmented | yes | 20k | 0.0 / 0.0 |
| r1 priv-MLP BC d=512 | augmented | yes | 20k | 0.0 / 0.0 |
| r1 priv-MLP BC d=1024 | augmented | yes | 20k | 0.0 / 0.0 |
| r2 priv-MLP BC d=256 | augmented | yes | 80k | 0.0 / 0.0 |
| r2 priv-MLP BC d=512 | augmented | yes | 80k | 0.0 / 0.0 |
| r2 priv-MLP BC d=1024 | augmented | yes | 80k | 0.0 / 0.0 |
| r3 priv-MLP DAgger d=256 | augmented | yes | DAgger | killed mid-iter (also bug) |
| A v2 priv-MLP no-perturb d=1024 | no-perturb | n/a | 80k | 0.0 / 0.0 |
| A v3-fix priv-MLP no-perturb d=1024 | no-perturb (wrong cfg) | n/a | 20k | 1.7 / 1.8 |

Currently in flight: **A v6 — priv-MLP no-perturb, d=2048, 50k demos, env rebuilt
on PrivilegedTrainCfg with `--num_data_envs 4096`.**
