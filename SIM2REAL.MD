## Normal Omnireset training, debug settings:
```
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
    --num_envs 16 \
    --logger wandb \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
```

## Running depth training:

### Single process:
```
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-Train-v0 \
    --num_envs 1024 \
    --logger wandb \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
```

```
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-Train-v0 \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
```