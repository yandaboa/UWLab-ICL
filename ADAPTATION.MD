## Training a policy with privileged observations.
Debug:
```
python scripts/reinforcement_learning/rsl_rl/train.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-v0     --num_envs 16     --logger wandb     --headless     env.scene.insertive_object=peg     env.scene.receptive_object=peghole
```

Full multi-GPU run:
```
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-v0  \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
```

## Evaling:
```
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Play-v0 \
    --num_envs 1 \
    --checkpoint <finetuned_checkpoint.pt> \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
```
or
```
python scripts/reinforcement_learning/rsl_rl/play.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-ContactDynamics-Play-v0     --num_envs 1024     --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-04-08_18-31-13/model_3200.pt    env.scene.insertive_object=fbdrawerbottom     env.scene.receptive_object=fbdrawerbox --headless --num_steps 500
```

## Evaling a policy trained on state + time based action + force augmentations:
```
python scripts/reinforcement_learning/rsl_rl/play.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Play-v0     --num_envs 1     --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-04-21_01-36-13/model_3700.pt     env.scene.insertive_object=peg     env.scene.receptive_object=peghole --video --video_length 100 --headless
```

## Running ASTEROID:
```
python run_incontext_exploration_parallel.py \
  --expert_policy_checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-04-21_01-36-13/exported/policy.pt \
  --config_name in_context_adaptation.yaml \
  --exp_name incontext_adaptation \
  --wandb_project incontext_adaptation \
  --output_dir logs/in_context_adaptation \
  --num_demos 20000 \
  --num_data_envs 8096 \
  --num_eval_envs 32 \
  --num_eval_episodes 32 \
  --max_iterations 4 \
  --data_gpu 2 \
  --train_gpu 3 \
  --insertive_object peg \
  --receptive_object peghole \
  --no_video
```