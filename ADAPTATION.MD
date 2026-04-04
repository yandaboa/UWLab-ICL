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

Evaling:
```
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Play-v0 \
    --num_envs 1 \
    --checkpoint <finetuned_checkpoint.pt> \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
```