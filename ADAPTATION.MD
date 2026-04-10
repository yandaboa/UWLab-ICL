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

## Running ASTEROID:
```
python run_incontext_exploration.py \
    --data_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0 \
    --eval_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0 \
    --expert_policy_checkpoint  \
    --config_name baseline_pg_spoc.yaml \
    --output_dir logs/incontext_exploration \
    --exp_name incontext_exploration \
    --wandb_project incontext_exploration \
    --num_demos 10 \
    --num_data_envs 2 \
    --num_eval_envs 2 \
    --num_eval_episodes 10 \
    --max_iterations 3 \
    --insertive_object peg \
    --receptive_object peghole
```