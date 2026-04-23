# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone profiler for ``DiffusionPolicyWrapper`` transformer inference.

Builds a ``TransformerImagePolicy`` matching the shape_meta / policy settings from
``diffusion_policy/diffusion_policy/config/in_context_adaptation.yaml`` (defaults to the
``sim2real_arm_dynamics_pomdp`` task's low-dim obs spec), wraps it in
``DiffusionPolicyWrapper`` with and then without KV caching, and measures per-step
inference wall time across a growing trajectory of randomized observations.

No Isaac Sim, no datasets, no training. All observations are random tensors sized to match
the obs spec, so the measured cost is purely transformer inference + KV cache bookkeeping.

Run::

    python scripts_v2/tools/profile_kv_cache.py --num_envs 1024 --num_steps 100 \
        --transformer_mini_batch_size 128 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Make the project modules importable when this file is run from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (
    os.path.join(_REPO_ROOT, "diffusion_policy"),
    os.path.join(_REPO_ROOT, "source", "uwlab_rl"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder  # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer  # noqa: E402
from diffusion_policy.policy.transformer_image_policy import TransformerImagePolicy  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper  # noqa: E402
import torch.nn as nn  # noqa: E402


# Shape spec mirrors diffusion_policy/config/task/sim2real_arm_dynamics_pomdp.yaml
DEFAULT_SHAPE_META = {
    "obs": {
        "end_effector_pose": {"shape": [6], "type": "low_dim"},
        "joint_pos": {"shape": [12], "type": "low_dim"},
        "prev_actions": {"shape": [7], "type": "low_dim"},
        "insertive_asset_pose": {"shape": [6], "type": "low_dim"},
        "receptive_asset_pose": {"shape": [6], "type": "low_dim"},
        "insertive_asset_in_receptive_asset_frame": {"shape": [6], "type": "low_dim"},
    },
    "action": {"shape": [7]},
}


def build_policy(
    shape_meta: dict,
    hidden_dim: int,
    hidden_depth: int,
    n_head: int,
    n_obs_steps: int,
    n_action_steps: int,
    horizon: int,
    device: torch.device,
) -> TransformerImagePolicy:
    """Build a randomly initialized TransformerImagePolicy on ``device``."""
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=torch.nn.Identity(),
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=False,
        share_rgb_model=False,
        imagenet_norm=False,
    )
    policy = TransformerImagePolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        n_head=n_head,
        dropout=0.0,
        horizon=horizon,
    ).to(device)
    # Fit the normalizer on a fake batch so normalize()/unnormalize() are well-defined.
    with torch.no_grad():
        fake_stats = {}
        for k, spec in shape_meta["obs"].items():
            fake_stats[k] = torch.randn(128, *spec["shape"], device=device)
        fake_stats["action"] = torch.randn(128, *shape_meta["action"]["shape"], device=device)
        normalizer = LinearNormalizer()
        normalizer.fit(fake_stats)
        policy.set_normalizer(normalizer)
    policy.eval()
    return policy


def make_fake_obs(
    shape_meta: dict, num_envs: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate one new step of fake obs shaped ``(num_envs, *shape)`` per key."""
    obs = {}
    for key, spec in shape_meta["obs"].items():
        obs[key] = torch.randn(num_envs, *spec["shape"], device=device)
    return obs


def run_rollout(
    wrapper: DiffusionPolicyWrapper,
    num_steps: int,
    num_envs: int,
    shape_meta: dict,
    device: torch.device,
    label: str,
    warmup_steps: int = 3,
) -> dict:
    """Step ``num_steps`` times through ``wrapper`` with random obs; return timing summary.

    Replicates the collection loop's call pattern: an initial full reset, then one call to
    ``predict_action`` per step with the obs for all envs. No resets mid-rollout — that
    matches the "worst case" where the KV cache keeps growing for the full episode, which
    is exactly where the non-cached path's O(T) cost blows up.
    """
    wrapper.reset(torch.arange(num_envs, device=device))

    for _ in range(warmup_steps):
        obs = make_fake_obs(shape_meta, num_envs, device)
        _ = wrapper.predict_action({"policy": obs}, env_indices=list(range(num_envs)))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    per_step_s = []
    t_total = time.time()
    for step in range(num_steps):
        obs = make_fake_obs(shape_meta, num_envs, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        actions = wrapper.predict_action({"policy": obs}, env_indices=list(range(num_envs)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_step_s.append(time.time() - t0)
        if step % max(1, num_steps // 10) == 0:
            print(
                f"[{label}] step={step:04d} per_step={per_step_s[-1] * 1000:.1f}ms "
                f"actions.shape={tuple(actions.shape)}",
                flush=True,
            )
    total_s = time.time() - t_total

    return {
        "label": label,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "total_s": total_s,
        "per_step_s": per_step_s,
        "mean_step_s": sum(per_step_s) / max(1, len(per_step_s)),
        "first_step_s": per_step_s[0] if per_step_s else 0.0,
        "last_step_s": per_step_s[-1] if per_step_s else 0.0,
    }


class TwoTokenPerStepTransformerImagePolicy(TransformerImagePolicy):
    """Example subclass that emits ``num_new_tokens=2`` per env step.

    Demonstrates that the KV cache machinery is token-structure-agnostic: we only override
    :meth:`_embed_new_step` (to produce 2 tokens instead of 1) and :meth:`_decode_step`
    (to read the action from the second of those 2 tokens). Nothing in the cache manager
    or :class:`DiffusionPolicyWrapper` changes.

    Concretely, the two new tokens per step are:

    * token 0: the usual obs embedding from the base class's obs encoder + ``input_proj``
    * token 1: the same obs embedding passed through a second learned projection
      (``input_proj_b``) — stands in for a hypothetical "action" / "separator" token

    The reference non-cached path runs the transformer on the full interleaved sequence
    ``[token0_t=0, token1_t=0, token0_t=1, token1_t=1, ...]`` in one shot and we compare
    action outputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_proj_b = nn.Linear(self.input_proj[0].in_features, self.input_proj[0].out_features)

    def _embed_obs_features(self, inputs: dict) -> torch.Tensor:
        """Return the pre-input_proj obs features (B, obs_feature_dim) for one step."""
        inputs = dict(inputs)
        # No prev_action/prev_reward support in this toy subclass.
        assert not self.include_action_in_context and not self.include_reward_in_context
        obs_with_time = {k: v.unsqueeze(1) for k, v in inputs.items()}
        nobs = self.normalizer.normalize(obs_with_time)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        return self.obs_encoder(this_nobs).reshape(B, 1, -1)  # (B, 1, D)

    def _embed_new_step(self, inputs: dict):
        obs_feats = self._embed_obs_features(inputs)  # (B, 1, D)
        t0 = self.input_proj(obs_feats)               # (B, 1, H)
        t1 = self.input_proj_b(obs_feats)             # (B, 1, H)
        inputs_embeds = torch.cat([t0, t1], dim=1)    # (B, 2, H)
        return inputs_embeds, 2

    def _decode_step(self, last_hidden_state: torch.Tensor):
        # Read the action from the SECOND of the 2 new tokens (so we actually exercise
        # num_new_tokens > 1 end-to-end rather than just padding unused tokens).
        h_last = last_hidden_state[:, 1:2, :]
        action_pred = self.output_head.predict(h_last, sample=True)
        action = self.normalizer['action'].unnormalize(action_pred)
        return {'action': action.squeeze(1), 'action_pred': action_pred.squeeze(1)}


def _build_full_sequence_tokens(
    policy: TwoTokenPerStepTransformerImagePolicy, obs_per_step: list[dict]
) -> torch.Tensor:
    """Concatenate per-step (t0, t1) tokens into one long (B, 2*T, H) sequence."""
    per_step_pairs = []
    for step_obs in obs_per_step:
        of = policy._embed_obs_features(step_obs)   # (B, 1, D)
        t0 = policy.input_proj(of)                   # (B, 1, H)
        t1 = policy.input_proj_b(of)                 # (B, 1, H)
        per_step_pairs.append(torch.cat([t0, t1], dim=1))  # (B, 2, H)
    return torch.cat(per_step_pairs, dim=1)  # (B, 2*T, H)


def check_correctness(
    shape_meta: dict,
    hidden_dim: int,
    hidden_depth: int,
    n_head: int,
    horizon: int,
    device: torch.device,
    num_envs: int = 8,
    num_steps: int = 12,
    atol: float = 1e-4,
) -> None:
    """Verify KV-cached inference matches a full-sequence reference, for k_new=1 and k_new=2.

    Runs three checks:

    1. ``k_new=1`` homogeneous: all envs start from step 0 and step forward in lockstep.
       Compares the KV-cached transformer hidden states to a single full-sequence forward.
    2. ``k_new=1`` heterogeneous: envs have different per-env past lengths (simulates
       asynchronous resets). Verifies padding + attention-mask logic.
    3. ``k_new=2`` homogeneous: subclass emits 2 tokens per step. Compares final-step
       actions between the KV-cached wrapper and a full-interleaved-sequence forward.
    """
    torch.manual_seed(0)

    # ---- Check 1 & 2: k_new=1 (default policy) ----
    base = build_policy(
        shape_meta=shape_meta,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        n_head=n_head,
        n_obs_steps=1,
        n_action_steps=1,
        horizon=horizon,
        device=device,
    )
    base.eval()

    # Build a deterministic per-step obs sequence.
    all_obs = [make_fake_obs(shape_meta, num_envs, device) for _ in range(num_steps)]

    # -- k_new=1 homogeneous: compare last_hidden_state of KV forward vs. full-seq forward --
    with torch.no_grad():
        # Full-sequence reference: drive the per-step embedder the same way the cached
        # path does, but concatenate all T tokens and run a single full forward. Using
        # `_embed_new_step` keeps us in sync with any future policy-side refactor
        # (tokens_per_step, token interleaving, type embeddings, ...).
        per_step_embeds = []
        for t in range(num_steps):
            embeds, k_new = base._embed_new_step(all_obs[t])
            assert k_new == 1, f"default policy should return num_new_tokens=1, got {k_new}"
            per_step_embeds.append(embeds)
        ref_inputs_embeds = torch.cat(per_step_embeds, dim=1)  # (B, T, H)
        attn = torch.ones(num_envs, num_steps, device=device, dtype=torch.long)
        ref_h = base.transformer(
            inputs_embeds=ref_inputs_embeds, attention_mask=attn,
            position_ids=torch.arange(num_steps, device=device).unsqueeze(0).expand(num_envs, -1),
        ).last_hidden_state  # (B, T, H)

        # Streaming: feed one step at a time through kv_cached_step.
        past_kvs = None
        past_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        streamed_h = []
        for t in range(num_steps):
            max_past = int(past_lengths.max().item())
            result = base.kv_cached_step(
                {k: v for k, v in all_obs[t].items()},
                past_key_values=past_kvs, past_lengths=past_lengths, max_past=max_past,
            )
            past_kvs = result["past_key_values"]
            past_lengths = past_lengths + result["num_new_tokens"]
        # The last hidden state is hidden in _decode_step; re-run forward to capture h.
        # For this numerical check we compare final-step hidden via a manual forward:
        # easier path: compare each step's output head input by monkey-stubbing _decode_step.
        # Simpler: re-run but capture via monkey-patch.
        streamed_h_tensors = []

        def capture_decode(self_, h):
            streamed_h_tensors.append(h.clone())
            return {"action": torch.zeros(num_envs, base.action_dim, device=device),
                    "action_pred": torch.zeros(num_envs, base.action_dim, device=device)}
        orig = type(base)._decode_step
        type(base)._decode_step = capture_decode
        try:
            past_kvs = None
            past_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
            for t in range(num_steps):
                max_past = int(past_lengths.max().item())
                result = base.kv_cached_step(
                    all_obs[t], past_key_values=past_kvs,
                    past_lengths=past_lengths, max_past=max_past,
                )
                past_kvs = result["past_key_values"]
                past_lengths = past_lengths + result["num_new_tokens"]
        finally:
            type(base)._decode_step = orig

        streamed_h = torch.cat(streamed_h_tensors, dim=1)  # (B, T*1, H)

    err = (streamed_h - ref_h).abs().max().item()
    print(f"[correctness] k_new=1 homogeneous max|Δ last_hidden_state| = {err:.3e}", flush=True)
    assert err < atol, f"k_new=1 homogeneous mismatch: {err}"

    # -- k_new=1 heterogeneous: async resets drive different per-env past_lengths --
    # Each env is assigned a random ``start_offset`` in [0, num_steps//2) and is only
    # queried on steps ``t >= start_offset[i]``. At any step after some envs have started
    # and others haven't, the TransformerKVCacheManager sees a batch with genuinely
    # heterogeneous per-env past_lengths (``past_lengths[i] = t - start_offset[i]``).
    # This exercises:
    #   * the attention-mask construction in ``kv_cached_step`` that blanks padded past
    #     slots per-env,
    #   * the per-env ``position_ids`` built from ``past_lengths`` rather than a shared
    #     counter,
    #   * the env-first scatter in ``TransformerKVCacheManager.append``, which has to
    #     write into different ``slots`` per env every call.
    # Reference: for each env, replay its own contiguous sub-sequence starting at its
    # offset through a plain full-sequence forward and compare per-step hidden states.
    with torch.no_grad():
        torch.manual_seed(1)
        start_offsets = torch.randint(0, num_steps // 2, (num_envs,), device=device)

        # Full-sequence reference: ref_h_per_env[i][tau] = hidden state for env i at its
        # (tau)-th post-reset step, i.e. absolute step t = start_offsets[i] + tau.
        ref_h_per_env: list[torch.Tensor] = []
        for i in range(num_envs):
            offset = int(start_offsets[i].item())
            seq_len = num_steps - offset
            per_step_embeds = []
            for t in range(offset, num_steps):
                step_obs = {k: all_obs[t][k][i:i + 1] for k in shape_meta["obs"].keys()}
                emb_t, k_new = base._embed_new_step(step_obs)
                assert k_new == 1
                per_step_embeds.append(emb_t)
            emb_i = torch.cat(per_step_embeds, dim=1)  # (1, seq_len, H)
            h_i = base.transformer(
                inputs_embeds=emb_i,
                attention_mask=torch.ones(1, seq_len, device=device, dtype=torch.long),
                position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
            ).last_hidden_state  # (1, seq_len, H)
            ref_h_per_env.append(h_i[0])  # (seq_len, H)

        # Wrapper streaming run. fp32 storage so we can assert tight numerical
        # equivalence (bf16 storage is separately validated in Check 4). Mini-batch sized
        # to the full active count so the monkey-patched ``_decode_step`` captures all
        # active envs in one shot per step.
        wrapper = DiffusionPolicyWrapper(
            base, device=device, n_obs_steps=1, num_envs=num_envs,
            mini_batch_size=num_envs, use_kv_cache=True,
            kv_cache_max_seq_len=horizon,
            kv_cache_storage_dtype=torch.float32,
            profile_name="check-hetero",
        )
        wrapper.reset(torch.arange(num_envs, device=device))

        captured: dict = {}

        def capture_decode(self_, h):
            # ``h`` is shape (B_active, num_new_tokens=1, H). Capture and return dummy
            # actions so the wrapper's postprocessing stays happy.
            captured["h"] = h.clone()
            B_active = h.shape[0]
            return {
                "action": torch.zeros(B_active, base.action_dim, device=device),
                "action_pred": torch.zeros(B_active, base.action_dim, device=device),
            }

        orig = type(base)._decode_step
        type(base)._decode_step = capture_decode
        max_err = 0.0
        try:
            for t in range(num_steps):
                active_idx = (start_offsets <= t).nonzero(as_tuple=True)[0].tolist()
                if not active_idx:
                    continue
                # Pre-slice obs to just the active envs so env_indices matches the batch
                # rows, as the wrapper's contract requires for the transformer path.
                obs_active = {k: all_obs[t][k][active_idx] for k in shape_meta["obs"].keys()}
                _ = wrapper.predict_action({"policy": obs_active}, env_indices=active_idx)
                h_active = captured["h"]  # (len(active_idx), 1, H)
                for j, i in enumerate(active_idx):
                    offset_i = int(start_offsets[i].item())
                    ref_h_it = ref_h_per_env[i][t - offset_i]  # (H,)
                    err = (h_active[j, 0] - ref_h_it).abs().max().item()
                    if err > max_err:
                        max_err = err
        finally:
            type(base)._decode_step = orig

    print(
        f"[correctness] k_new=1 heterogeneous (async-reset, per-env past_lengths) "
        f"max|Δ last_hidden_state| = {max_err:.3e}",
        flush=True,
    )
    assert max_err < atol, f"k_new=1 heterogeneous mismatch: {max_err}"

    # ---- Check 3: k_new=2 via TwoTokenPerStepTransformerImagePolicy ----
    torch.manual_seed(2)
    two_tok = TwoTokenPerStepTransformerImagePolicy(
        shape_meta=shape_meta,
        obs_encoder=MultiImageObsEncoder(
            shape_meta=shape_meta, rgb_model=torch.nn.Identity(), resize_shape=None,
            crop_shape=None, random_crop=False, use_group_norm=False,
            share_rgb_model=False, imagenet_norm=False,
        ),
        n_action_steps=1, n_obs_steps=1, hidden_dim=hidden_dim,
        hidden_depth=hidden_depth, n_head=n_head, dropout=0.0, horizon=horizon,
    ).to(device).eval()
    with torch.no_grad():
        fake_stats = {k: torch.randn(128, *spec["shape"], device=device)
                      for k, spec in shape_meta["obs"].items()}
        fake_stats["action"] = torch.randn(128, *shape_meta["action"]["shape"], device=device)
        norm = LinearNormalizer()
        norm.fit(fake_stats)
        two_tok.set_normalizer(norm)

    with torch.no_grad():
        # Full-interleaved-sequence reference: (B, 2*T, H)
        ref_seq = _build_full_sequence_tokens(two_tok, all_obs)
        seq_len = ref_seq.shape[1]
        ref_h = two_tok.transformer(
            inputs_embeds=ref_seq,
            attention_mask=torch.ones(num_envs, seq_len, device=device, dtype=torch.long),
            position_ids=torch.arange(seq_len, device=device).unsqueeze(0).expand(num_envs, -1),
        ).last_hidden_state  # (B, 2*T, H)

        # Streaming: KV-cached per step, capture the 2-token hidden state per step.
        streamed = []
        def capture_two(self_, h):
            streamed.append(h.clone())  # (B, 2, H)
            return {"action": torch.zeros(num_envs, two_tok.action_dim, device=device),
                    "action_pred": torch.zeros(num_envs, two_tok.action_dim, device=device)}
        orig_two = TwoTokenPerStepTransformerImagePolicy._decode_step
        TwoTokenPerStepTransformerImagePolicy._decode_step = capture_two
        try:
            past_kvs = None
            past_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
            for t in range(num_steps):
                max_past = int(past_lengths.max().item())
                result = two_tok.kv_cached_step(
                    all_obs[t], past_key_values=past_kvs,
                    past_lengths=past_lengths, max_past=max_past,
                )
                past_kvs = result["past_key_values"]
                past_lengths = past_lengths + result["num_new_tokens"]
            assert past_lengths[0].item() == 2 * num_steps, (
                f"Expected {2 * num_steps} cached tokens, got {past_lengths[0].item()}"
            )
        finally:
            TwoTokenPerStepTransformerImagePolicy._decode_step = orig_two

    streamed_h = torch.cat(streamed, dim=1)  # (B, 2*T, H)
    err2 = (streamed_h - ref_h).abs().max().item()
    print(f"[correctness] k_new=2 homogeneous max|Δ last_hidden_state| = {err2:.3e}", flush=True)
    assert err2 < atol, f"k_new=2 mismatch: {err2}"

    # -- k_new=2 heterogeneous: async resets + multi-token scatter at varying slot offsets --
    # Same async-reset setup as Check 2 but with the 2-token policy. At step t, active env i
    # writes 2 new KV rows into slots ``[2*(t - offset_i), 2*(t - offset_i)+1]`` — every
    # env has a different pair of slots, and the env-first scatter loop must place them
    # correctly without cross-env bleed.
    with torch.no_grad():
        torch.manual_seed(3)
        start_offsets_k2 = torch.randint(0, num_steps // 2, (num_envs,), device=device)

        # Reference: per-env (2*seq_len_i, H) hidden states, where the tau-th step
        # contributes two rows at positions 2*tau and 2*tau+1.
        ref_h_per_env_k2: list[torch.Tensor] = []
        for i in range(num_envs):
            offset = int(start_offsets_k2[i].item())
            seq_len_steps = num_steps - offset
            per_step_pairs = []
            for t in range(offset, num_steps):
                of = two_tok._embed_obs_features({k: all_obs[t][k][i:i + 1] for k in shape_meta["obs"].keys()})
                t0 = two_tok.input_proj(of)
                t1 = two_tok.input_proj_b(of)
                per_step_pairs.append(torch.cat([t0, t1], dim=1))  # (1, 2, H)
            emb_i = torch.cat(per_step_pairs, dim=1)  # (1, 2*seq_len_steps, H)
            seq_len_tokens = emb_i.shape[1]
            h_i = two_tok.transformer(
                inputs_embeds=emb_i,
                attention_mask=torch.ones(1, seq_len_tokens, device=device, dtype=torch.long),
                position_ids=torch.arange(seq_len_tokens, device=device).unsqueeze(0),
            ).last_hidden_state  # (1, 2*seq_len_steps, H)
            ref_h_per_env_k2.append(h_i[0])  # (2*seq_len_steps, H)

        wrapper_k2 = DiffusionPolicyWrapper(
            two_tok, device=device, n_obs_steps=1, num_envs=num_envs,
            mini_batch_size=num_envs, use_kv_cache=True,
            kv_cache_max_seq_len=horizon,
            kv_cache_storage_dtype=torch.float32,
            profile_name="check-hetero-k2",
        )
        wrapper_k2.reset(torch.arange(num_envs, device=device))

        captured_k2: dict = {}

        def capture_decode_k2(self_, h):
            # ``h`` is (B_active, 2, H) for k_new=2.
            captured_k2["h"] = h.clone()
            B_active = h.shape[0]
            return {
                "action": torch.zeros(B_active, two_tok.action_dim, device=device),
                "action_pred": torch.zeros(B_active, two_tok.action_dim, device=device),
            }

        orig_two_k2 = TwoTokenPerStepTransformerImagePolicy._decode_step
        TwoTokenPerStepTransformerImagePolicy._decode_step = capture_decode_k2
        max_err_k2 = 0.0
        try:
            for t in range(num_steps):
                active_idx = (start_offsets_k2 <= t).nonzero(as_tuple=True)[0].tolist()
                if not active_idx:
                    continue
                obs_active = {k: all_obs[t][k][active_idx] for k in shape_meta["obs"].keys()}
                _ = wrapper_k2.predict_action({"policy": obs_active}, env_indices=active_idx)
                h_active = captured_k2["h"]  # (B_active, 2, H)
                for j, i in enumerate(active_idx):
                    tau = t - int(start_offsets_k2[i].item())
                    # Reference has both tokens at positions 2*tau and 2*tau+1.
                    ref_pair = ref_h_per_env_k2[i][2 * tau : 2 * tau + 2]  # (2, H)
                    err = (h_active[j] - ref_pair).abs().max().item()
                    if err > max_err_k2:
                        max_err_k2 = err
        finally:
            TwoTokenPerStepTransformerImagePolicy._decode_step = orig_two_k2

    print(
        f"[correctness] k_new=2 heterogeneous (async-reset, per-env past_lengths) "
        f"max|Δ last_hidden_state| = {max_err_k2:.3e}",
        flush=True,
    )
    assert max_err_k2 < atol, f"k_new=2 heterogeneous mismatch: {max_err_k2}"

    # ---- Check 4: env-first layout + bf16 storage round-trip through the wrapper ----
    # Exercises the full DiffusionPolicyWrapper path (TransformerKVCacheManager gather /
    # append) to catch bugs in the new layout or scatter semantics. We run this for BOTH
    # the default 1-token-per-step policy and the k_new=2 TwoTokenPerStepTransformerImagePolicy so the
    # env-first ``append`` scatter loop is exercised with multiple new tokens per step
    # (past_lengths growing by 2 each call means different slot offsets into the cache).
    #
    # For each policy we compare:
    #   (a) fp32-storage wrapper — numerical reference; identity with respect to the
    #       kv_cached_step path validated by Checks 1–3.
    #   (b) bf16-storage wrapper vs. (a) — must match within bf16's ~7-bit mantissa.
    def _run_via_wrapper(policy, storage_dtype: torch.dtype, tag: str) -> torch.Tensor:
        wrapper = DiffusionPolicyWrapper(
            policy, device=device, n_obs_steps=1, num_envs=num_envs,
            mini_batch_size=max(1, num_envs // 2), use_kv_cache=True,
            kv_cache_max_seq_len=horizon, kv_cache_storage_dtype=storage_dtype,
            profile_name=f"check-{tag}-{storage_dtype}",
        )
        wrapper.reset(torch.arange(num_envs, device=device))
        all_actions = []
        for t in range(num_steps):
            # Deterministic: reseed so output_head sampling is reproducible across runs.
            torch.manual_seed(1000 + t)
            act = wrapper.predict_action({"policy": all_obs[t]}, env_indices=list(range(num_envs)))
            all_actions.append(act.clone())
        return torch.stack(all_actions, dim=0)  # (T, B, Da)

    for policy_obj, tag, expected_k_new in [(base, "k1", 1), (two_tok, "k2", 2)]:
        fp32_actions = _run_via_wrapper(policy_obj, torch.float32, tag)
        bf16_actions = _run_via_wrapper(policy_obj, torch.bfloat16, tag)
        # Sanity: the cache should have expected_k_new tokens per env step in fp32 run.
        # (We can only query the wrapper that still exists; rebuild a fresh one briefly
        # just to read per-env lengths after running.)
        wrapper_sanity = DiffusionPolicyWrapper(
            policy_obj, device=device, n_obs_steps=1, num_envs=num_envs,
            mini_batch_size=max(1, num_envs // 2), use_kv_cache=True,
            kv_cache_max_seq_len=horizon, kv_cache_storage_dtype=torch.float32,
            profile_name=f"sanity-{tag}",
        )
        wrapper_sanity.reset(torch.arange(num_envs, device=device))
        for t in range(num_steps):
            torch.manual_seed(1000 + t)
            _ = wrapper_sanity.predict_action({"policy": all_obs[t]}, env_indices=list(range(num_envs)))
        assert wrapper_sanity.kv_cache is not None
        final_lengths = wrapper_sanity.kv_cache.lengths
        assert (final_lengths == expected_k_new * num_steps).all(), (
            f"[{tag}] expected all envs to have {expected_k_new * num_steps} cached tokens after "
            f"{num_steps} steps (k_new={expected_k_new}), got lengths={final_lengths.tolist()}"
        )

        wrapper_diff = (fp32_actions - bf16_actions).abs().max().item()
        print(
            f"[correctness] wrapper ({tag}, num_new_tokens={expected_k_new}) bf16 vs fp32 storage "
            f"max|Δ action| = {wrapper_diff:.3e} "
            f"(expected O(1e-2) due to bf16 mantissa; per-env length={expected_k_new * num_steps})",
            flush=True,
        )
        assert wrapper_diff < 5e-2, (
            f"[{tag}] bf16 vs fp32 KV cache storage disagree by {wrapper_diff:.3e} — larger "
            f"than the ~1e-2 expected from bf16 mantissa, suggests a real bug in the new "
            f"env-first layout or the k_new={expected_k_new} append-scatter loop."
        )

    print("[correctness] all checks passed.", flush=True)


def summarize(result: dict) -> str:
    return (
        f"[{result['label']}] total={result['total_s']:.2f}s "
        f"mean_step={result['mean_step_s'] * 1000:.1f}ms "
        f"first_step={result['first_step_s'] * 1000:.1f}ms "
        f"last_step={result['last_step_s'] * 1000:.1f}ms "
        f"env_throughput={result['num_envs'] * result['num_steps'] / result['total_s']:.0f} envs/s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=100, help="Trajectory length to simulate.")
    parser.add_argument("--transformer_mini_batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_depth", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--kv_cache_max_seq_len", type=int, default=None)
    parser.add_argument(
        "--cache_storage_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Preallocated KV cache dtype. bf16 halves memory at negligible precision loss; "
        "fp32 is useful for tight numerical equivalence checks.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="kv,nokv",
        help="Comma-separated subset of {kv, nokv}. Run only one to save time.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check", action="store_true",
        help="Run correctness checks (incl. a k_new=2 subclass) and exit without profiling.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape_meta = DEFAULT_SHAPE_META

    print(
        f"[profile_kv_cache] device={device} num_envs={args.num_envs} num_steps={args.num_steps} "
        f"mini_batch={args.transformer_mini_batch_size} hidden={args.hidden_dim}x{args.hidden_depth} "
        f"heads={args.n_head} horizon={args.horizon}",
        flush=True,
    )

    if args.check:
        check_correctness(
            shape_meta=shape_meta,
            hidden_dim=args.hidden_dim,
            hidden_depth=args.hidden_depth,
            n_head=args.n_head,
            horizon=args.horizon,
            device=device,
        )
        return

    print("[profile_kv_cache] building policy...", flush=True)
    t0 = time.time()
    policy = build_policy(
        shape_meta=shape_meta,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        n_head=args.n_head,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        horizon=args.horizon,
        device=device,
    )
    print(f"[profile_kv_cache] policy built in {time.time() - t0:.1f}s", flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: list[dict] = []

    for mode in modes:
        use_kv = mode == "kv"
        print(f"\n========== mode={mode} (use_kv_cache={use_kv}) ==========", flush=True)
        cache_storage_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.cache_storage_dtype]
        wrapper = DiffusionPolicyWrapper(
            policy,
            device=device,
            n_obs_steps=args.n_obs_steps,
            num_envs=args.num_envs,
            mini_batch_size=args.transformer_mini_batch_size,
            use_kv_cache=use_kv,
            kv_cache_max_seq_len=args.kv_cache_max_seq_len,
            kv_cache_storage_dtype=cache_storage_dtype,
            profile_name=f"profile-{mode}",
        )
        label = "kv_cached" if use_kv else "no_kv_cached"
        result = run_rollout(
            wrapper=wrapper,
            num_steps=args.num_steps,
            num_envs=args.num_envs,
            shape_meta=shape_meta,
            device=device,
            label=label,
        )
        results.append(result)
        print(summarize(result), flush=True)
        del wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) == 2:
        kv = next(r for r in results if r["label"] == "kv_cached")
        nokv = next(r for r in results if r["label"] == "no_kv_cached")
        speedup_total = nokv["total_s"] / max(1e-9, kv["total_s"])
        speedup_last = nokv["last_step_s"] / max(1e-9, kv["last_step_s"])
        print(
            f"\n[profile_kv_cache] total-time speedup (nokv / kv) = {speedup_total:.2f}x; "
            f"last-step speedup = {speedup_last:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
