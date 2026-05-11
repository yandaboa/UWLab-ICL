## Training RL Policy:
```
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=fbleg \
    env.scene.receptive_object=fbtabletop
```

## Eval a finetuned (sys-id gains) checkpoint

```
wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts_finetuned/peg_state_rl_expert_finetuned_seed42.pt

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
    --num_envs 1 \
    --checkpoint peg_state_rl_expert_finetuned_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole --headless
```

Data collection

```
python scripts_v2/tools/collect_demos.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0 \
    --dataset_file /gscratch/scrubbed/yanda/omnireset/peg/rgb_seed10/rgb0.zarr \
    --num_envs 64 \
    --num_demos 40000 \
    --enable_cameras \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["logs/policies/seed10/exported/policy.pt"]'
```