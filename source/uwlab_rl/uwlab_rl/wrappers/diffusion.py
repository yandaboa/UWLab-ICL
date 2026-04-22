# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import ABC, abstractmethod
from typing import Any


class ObservationHistoryManager(ABC):
    """Abstract base class for managing observation history."""

    def __init__(self, num_envs: int, n_obs_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.n_obs_steps = n_obs_steps
        self.device = device
        self.history = None
        self.needs_init = set()

    @abstractmethod
    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        """Initialize the history with the first observation."""
        pass

    @abstractmethod
    def update(self, processed_obs: dict[str, torch.Tensor]):
        """Update history with new observations."""
        pass

    @abstractmethod
    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        """Get observation batch for specific environments."""
        pass

    @abstractmethod
    def reset_envs(self, env_indices: list[int]):
        """Reset history for specific environments."""
        pass


class LowDimObservationHistory(ObservationHistoryManager):
    """Manages observation history for low-dimensional policies as a fixed rolling buffer."""

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        obs_shape = processed_obs["obs"].shape
        history_shape = (self.num_envs, self.n_obs_steps, obs_shape[-1])
        self.history = torch.zeros(history_shape, device=self.device, dtype=processed_obs["obs"].dtype)

    def update(self, processed_obs: dict[str, torch.Tensor]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init:
            for env_idx in list(self.needs_init):
                first_obs = processed_obs["obs"][env_idx : env_idx + 1]
                for step in range(self.n_obs_steps):
                    self.history[env_idx, step] = first_obs[0]
                self.needs_init.remove(env_idx)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = processed_obs["obs"]

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        if self.history is None:
            return {"obs": torch.zeros((len(env_indices), self.n_obs_steps, 0), device=self.device)}
        env_obs = self.history[env_indices]
        return {"obs": env_obs}

    def reset_envs(self, env_indices: list[int]):
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationHistory(ObservationHistoryManager):
    """Manages observation history for image-based policies as a fixed rolling buffer per key."""

    def __init__(self, num_envs: int, obs_keys: list[str], n_obs_steps: int, device: torch.device):
        super().__init__(num_envs, n_obs_steps, device)
        self.obs_keys = obs_keys

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        self.history = {}
        for key in self.obs_keys:
            obs_shape = processed_obs[key].shape
            history_shape = (self.num_envs, self.n_obs_steps) + obs_shape[1:]
            self.history[key] = torch.zeros(history_shape, device=self.device, dtype=processed_obs[key].dtype)

    def update(self, processed_obs: dict[str, torch.Tensor]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    for key in self.obs_keys:
                        first_obs = processed_obs[key][env_idx : env_idx + 1]
                        for step in range(self.n_obs_steps):
                            self.history[key][env_idx, step] = first_obs[0]
                    self.needs_init.remove(env_idx)
        if self.obs_keys is not None:
            for key in self.obs_keys:
                self.history[key][:, :-1] = self.history[key][:, 1:].clone()
                self.history[key][:, -1] = processed_obs[key]

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        if self.history is None or self.obs_keys is None:
            return {}
        obs_batch = {}
        for key in self.obs_keys:
            env_obs = self.history[key][env_indices]
            obs_batch[key] = env_obs
        return obs_batch

    def reset_envs(self, env_indices: list[int]):
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationSequence(ImageObservationHistory):
    """Stores full trajectory per-environment as lists and returns padded batches with attention masks.

    Used for transformer policies that perform in-context exploration: each env's trajectory grows
    from reset, and the policy receives a padded (B, max_len, ...) batch with an attention_mask of
    shape (B, max_len) where 1 = real observation, 0 = padding.
    """

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        self.history = {}
        for key in self.obs_keys:
            self.history[key] = [[] for _ in range(self.num_envs)]

    def update(self, processed_obs: dict[str, torch.Tensor], env_indices: list[int]):
        """Append the current observation to each listed env's trajectory, clearing any env flagged for init."""
        if self.history is None:
            self.initialize(processed_obs)

        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    for key in self.obs_keys:
                        self.history[key][env_idx].clear()
                    self.needs_init.remove(env_idx)

        if self.obs_keys is not None:
            for key in self.obs_keys:
                tensor = processed_obs[key]
                # `tensor` is pre-filtered to the exploration subset (shape: (len(env_indices), ...));
                # `idx` indexes into that subset while `env_idx` is the absolute env index into history.
                for idx, env_idx in enumerate(env_indices):
                    obs = tensor[idx]
                    self.history[key][env_idx].append(obs.detach().clone().to(self.device))

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        if self.history is None or self.obs_keys is None:
            return {}

        lengths = [len(self.history[self.obs_keys[0]][env]) for env in env_indices]
        max_len = max(lengths) if lengths else 0

        obs_batch: dict[str, torch.Tensor] = {}

        lengths_tensor = torch.tensor(lengths, device=self.device)
        # (max_len,) < (B, 1) → (B, max_len) bool: True for positions < seq_len, i.e. real obs.
        attention_mask = (torch.arange(max_len, device=self.device).unsqueeze(0) < lengths_tensor.unsqueeze(1)).long()

        for key in self.obs_keys:
            obs_batch[key] = torch.nn.utils.rnn.pad_sequence(
                [torch.stack(self.history[key][env], dim=0) for env in env_indices],
                batch_first=True,
                padding_value=0.0,
            )

        obs_batch["attention_mask"] = attention_mask
        return obs_batch


class DiffusionPolicyWrapper:
    """Wraps diffusion policy to handle Isaac Lab environment observations and action execution."""

    def __init__(self, policy, device: torch.device, n_obs_steps: int = 2, num_envs: int = 1):
        """Initialize the policy wrapper.

        Args:
            policy: The diffusion policy to wrap.
            device: Device to run the policy on.
            n_obs_steps: Number of observation steps to maintain in history (unused by the
                transformer path, which keeps a full per-env trajectory).
            num_envs: Number of environments to handle.
        """
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.num_envs = num_envs

        self.is_image_policy = self._is_image_policy()
        self.is_transformer = self._is_transformer()
        if hasattr(policy.obs_encoder, "keys"):
            obs_keys = policy.obs_encoder.keys
        else:
            obs_keys = policy.obs_encoder.rgb_keys + policy.obs_encoder.low_dim_keys
        if self.is_transformer:
            self.obs_history_manager = ImageObservationSequence(num_envs, obs_keys, n_obs_steps, device)
        elif self.is_image_policy:
            self.obs_history_manager = ImageObservationHistory(num_envs, obs_keys, n_obs_steps, device)
        else:
            self.obs_history_manager = LowDimObservationHistory(num_envs, n_obs_steps, device)

        self.action_queue = [[] for _ in range(num_envs)]

        self.policy.reset()

    def _is_transformer(self) -> bool:
        """Detect if the policy is a transformer-based model that requires attention_mask."""
        policy_class_name = self.policy.__class__.__name__.lower()
        transformer_indicators = ["transformer", "gpt", "bert", "dpt", "aawr"]
        return any(indicator in policy_class_name for indicator in transformer_indicators)

    def _is_image_policy(self) -> bool:
        """Detect if this is an image policy based on class name."""
        policy_class_name = self.policy.__class__.__name__.lower()
        image_policy_indicators = ["image", "hybrid", "video"]
        return any(indicator in policy_class_name for indicator in image_policy_indicators)

    def reset(self, reset_ids: torch.Tensor):
        """Reset the policy wrapper and clear observation history and action queue."""
        reset_indices = reset_ids.tolist() if hasattr(reset_ids, "tolist") else reset_ids
        for i in reset_indices:
            self.action_queue[i].clear()

        if isinstance(reset_indices, torch.Tensor):
            reset_indices = reset_indices.tolist()
        self.obs_history_manager.reset_envs(reset_indices)
        self.policy.reset()

    def predict_action(self, obs_dict: dict[str, Any], env_indices: list[int] | None = None) -> torch.Tensor:
        """Predict action given Isaac Lab environment observations.

        For transformer policies, ``obs_dict`` should be pre-filtered to only the envs in
        ``env_indices`` (so their per-env trajectories only grow on steps where this policy
        is actually executed). For other policies, ``obs_dict`` contains all envs and
        ``env_indices`` is ignored.

        Args:
            obs_dict: Raw observations from Isaac Lab environment.
            env_indices: Absolute env indices corresponding to rows of ``obs_dict`` (transformer path).

        Returns:
            Action tensor with shape (len(env_indices), action_dim) for the transformer path, or
            (num_envs, action_dim) otherwise.
        """
        processed_obs = self._process_obs(obs_dict)

        if self.is_transformer:
            if env_indices is None:
                env_indices = list(range(self.num_envs))
            self.obs_history_manager.update(processed_obs, env_indices)
            need_new_actions = [i for i in range(self.num_envs) if len(self.action_queue[i]) == 0 and i in env_indices]
        else:
            self.obs_history_manager.update(processed_obs)
            need_new_actions = [i for i in range(self.num_envs) if len(self.action_queue[i]) == 0]

        if need_new_actions:
            new_actions = self._get_action_chunks(need_new_actions)
            for idx, env_idx in enumerate(need_new_actions):
                self.action_queue[env_idx].extend(new_actions[idx])

        if self.is_transformer:
            actions = torch.zeros(
                len(env_indices), self.action_queue[env_indices[0]][0].shape[-1], device=self.device, dtype=torch.float32
            )
            for idx, env_idx in enumerate(env_indices):
                actions[idx] = self.action_queue[env_idx].pop(0)
            return actions

        actions = torch.zeros(self.num_envs, self.action_queue[0][0].shape[-1], device=self.device, dtype=torch.float32)
        for i in range(self.num_envs):
            actions[i] = self.action_queue[i].pop(0)

        return actions

    def _process_obs(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert Isaac Lab observations to format expected by diffusion policy."""
        if isinstance(obs_dict, dict):
            obs = obs_dict.get("policy", obs_dict)
        else:
            obs = obs_dict

        if self.is_image_policy:
            return self._process_image_obs(obs)
        else:
            return self._process_lowdim_obs(obs)

    def _process_image_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process observations for image-based policies with batched operations."""
        processed_obs = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device)
            else:
                tensor = torch.tensor(value, device=self.device)
            processed_obs[key] = tensor
        return processed_obs

    def _process_lowdim_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process observations for low-dimensional policies with batched operations."""
        obs_components = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, torch.Tensor):
                obs_components.append(value.to(self.device))
            else:
                obs_components.append(torch.tensor(value, device=self.device))

        if obs_components:
            obs_tensor = torch.cat(obs_components, dim=-1)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            processed_obs = {"obs": obs_tensor}
        else:
            processed_obs = {"obs": torch.zeros((self.num_envs, 0), device=self.device)}

        return processed_obs

    def _get_action_chunks(self, env_indices: list[int]) -> list[torch.Tensor]:
        """Get action chunks for specific environments, mini-batched to bound transformer memory."""
        mini_batch_size = 8
        mini_batch_actions = []
        for start in range(0, len(env_indices), mini_batch_size):
            end = start + mini_batch_size
            mini_batch_envs = env_indices[start:end]
            obs_batch = self.obs_history_manager.get_batch(mini_batch_envs)
            with torch.no_grad():
                result = self.policy.predict_action(obs_batch)
            if isinstance(result, dict):
                action_chunk = result["action"]
            else:
                action_chunk = result
            mini_batch_actions.extend(action_chunk)
        action_chunk = torch.stack(mini_batch_actions, dim=0)

        action_chunks = []
        if action_chunk.ndim == 3:
            for i in range(action_chunk.shape[0]):
                env_action_chunk = action_chunk[i]
                action_chunks.append(env_action_chunk)
        else:
            # Single action per env: (B, Da) → unsqueeze to (B, 1, Da) as a per-env chunk of length 1.
            action_chunks = action_chunk.unsqueeze(1)

        return action_chunks
