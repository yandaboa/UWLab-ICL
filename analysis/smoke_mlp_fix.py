"""Smoke test the fixed MLPImagePolicy.compute_loss on an existing zarr.

Verifies:
1. Forward returns finite loss
2. Backward produces non-zero gradients on trunk + mean_head
3. The loss actually depends on all T timesteps (not just t=0): zeroing actions at
   t>=1 should change the loss substantially.
"""
import os, sys, torch
sys.path.insert(0, "/mnt/storage/lti/UWLab-ICL")

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

cfg_dir = os.path.abspath("/mnt/storage/lti/UWLab-ICL/diffusion_policy/diffusion_policy/config")
zarr_path = "logs/priv_baseline/a_priv_mlp_noperturb_d1024_v2/priv_baseline_a_priv_mlp_noperturb_d1024_v2/2026-05-03_23-33-12/dataset-iteration-0/data.zarr"

os.chdir("/mnt/storage/lti/UWLab-ICL")
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(
        config_name="in_context_privileged_mlp_noperturb.yaml",
        overrides=[
            "policy.hidden_dim=1024",
            "policy.hidden_depth=6",
            f"task.dataset.dataset_config=[{{dataset_path: {zarr_path}, sampling_ratio: 1.0}}]",
        ],
    )

ds = hydra.utils.instantiate(cfg.task.dataset)
normalizer = ds.get_normalizer()
policy = hydra.utils.instantiate(cfg.policy)
policy.set_normalizer(normalizer)
policy.cuda()

from diffusion_policy.common.sampler import get_collate_fn
collate = get_collate_fn()
batch_items = [ds[i] for i in range(4)]
batch = collate(batch_items)


def to_cuda(v):
    if torch.is_tensor(v):
        return v.cuda()
    if isinstance(v, dict):
        return {kk: to_cuda(vv) for kk, vv in v.items()}
    return v

batch = {k: to_cuda(v) for k, v in batch.items()}

print("== shapes ==")
print("  obs[end_effector_pose]:", batch["obs"]["end_effector_pose"].shape)
print("  action:", batch["action"].shape)
print("  expert_mask:", batch["expert_mask"].shape)
print("  attention_mask:", batch["attention_mask"].shape)

policy.zero_grad()
loss = policy.compute_loss(batch)
print(f"\ncompute_loss -> {loss.item():.4f}")
loss.backward()

mean_head_grad = policy.mean_head.weight.grad.abs().mean().item()
trunk_grad = policy.trunk[0].weight.grad.abs().mean().item()
print(f"mean_head.weight grad mean = {mean_head_grad:.6f}")
print(f"trunk[0].weight    grad mean = {trunk_grad:.6f}")


def clone_dict(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = clone_dict(v)
        else:
            out[k] = v
    return out


batch2 = clone_dict(batch)
batch2["action"][:, 1:] = 0.0
policy.zero_grad()
loss2 = policy.compute_loss(batch2)
print(f"\ncompute_loss (a_t=0 for t>=1) -> {loss2.item():.4f}")
print(f"  abs diff vs baseline = {abs(loss.item() - loss2.item()):.4f}")

# Sharper test: replace future actions with values far outside the prior. With a
# wide initial Gaussian, replacing with extreme values should produce a much
# bigger log-prob hit, scaling with how many timesteps contribute.
batch3 = clone_dict(batch)
batch3["action"][:, 1:] = 100.0  # very OOD target
policy.zero_grad()
loss3 = policy.compute_loss(batch3)
print(f"\ncompute_loss (a_t=100 for t>=1) -> {loss3.item():.4f}")
print(f"  abs diff vs baseline = {abs(loss.item() - loss3.item()):.4f}")
print(f"  (should be MUCH larger than 0; if ~0 the fix isn't using t>=1)")

# Sanity: how many valid timesteps per batch
mask = batch["expert_mask"][..., 0] * batch["attention_mask"]
n_valid = mask.sum().item()
print(f"\nvalid timesteps in mask: {n_valid:.0f} / {mask.numel()} ({100*n_valid/mask.numel():.0f}%)")
print(f"if fix is correct, the t=100 loss should be roughly baseline + (n_valid-B)/n_valid * huge_per_step_penalty")
