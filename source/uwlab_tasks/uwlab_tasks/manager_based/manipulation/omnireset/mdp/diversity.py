# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""DIAYN-style diversity MDP terms.

Provides:
- :class:`SkillObs` — a per-env skill index sampled at reset, exposed as a one-hot observation
  consumed by the policy. Also publishes the integer skill buffer + alphabet size on the env so
  the diversity reward term can read them without going through the manager.
- :func:`diversity_reward` — queries ``env.diversity_discriminator`` (set by the algorithm) for
  log q(z | s_{t+1}) and returns it as a per-env reward. ``s_{t+1}`` is recomputed from the
  configured discriminator obs group (the env's reward step happens before the next observation
  manager pass updates ``obs_buf``, so we recompute the group fresh from current scene state).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SkillObs(ManagerTermBase):
    """Per-env discrete skill index, refreshed on every env reset.

    Returns a one-hot tensor of shape ``(num_envs, num_skills)`` for the current skill. Also
    writes:
      - ``env.diversity_skill_idx``: ``LongTensor[num_envs]`` of integer skill labels.
      - ``env.diversity_num_skills``: ``int`` size of the alphabet.

    so the reward term and discriminator update path can read them without re-traversing the
    observation manager.
    """

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self.num_skills = int(cfg.params.get("num_skills", 10))
        # ``force_skill`` overrides random sampling: when >= 0 every env is locked to this
        # skill index on init *and* on every reset. -1 (default) means resample uniformly.
        self.force_skill = int(cfg.params.get("force_skill", -1))
        if self.force_skill >= 0:
            if not (0 <= self.force_skill < self.num_skills):
                raise ValueError(
                    f"force_skill={self.force_skill} out of range for num_skills={self.num_skills}"
                )
            self.skill_idx = torch.full(
                (env.num_envs,), self.force_skill, dtype=torch.long, device=env.device
            )
        else:
            self.skill_idx = torch.randint(
                low=0, high=self.num_skills, size=(env.num_envs,), dtype=torch.long, device=env.device
            )
        # Publish on the env so the reward term and the algorithm can find them.
        env.diversity_skill_idx = self.skill_idx
        env.diversity_num_skills = self.num_skills

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self.force_skill >= 0:
            self.skill_idx[env_ids] = self.force_skill
        else:
            self.skill_idx[env_ids] = torch.randint(
                low=0,
                high=self.num_skills,
                size=(env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        num_skills: int = 10,
        force_skill: int = -1,
    ) -> torch.Tensor:
        return F.one_hot(self.skill_idx, num_classes=self.num_skills).float()


class DiversityReward(ManagerTermBase):
    """DIAYN-style diversity reward: ``log q(z | s_{t+1}) - log p(z)``.

    Maintains a per-env ``task_done`` latch that turns True the first step the task's success
    criterion fires (read off the ``progress_context`` reward term). Once latched, the
    diversity bonus is zeroed for that env until reset — the policy gets no extra reward for
    weird behavior after the task is solved. The latch is also exposed via
    ``env.diversity_task_done`` and via the ``diversity_meta`` obs group so the algorithm can
    mask out post-success transitions from the discriminator update.

    Pulls the discriminator from the env (set by :class:`DiversityRunner`). Until the
    discriminator is attached (e.g. on the very first step before training begins), returns
    zeros. The state ``s_{t+1}`` is read by recomputing the configured obs group from current
    scene state — RewardManager fires before the next ObservationManager pass updates
    ``obs_buf``, but the underlying scene tensors have already been advanced by physics.
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self.obs_group: str = cfg.params.get("obs_group", "discriminator_obs")
        self.success_term_name: str = cfg.params.get("success_term_name", "progress_context")

        # Per-env latch: True once the task has ever been solved within the current episode.
        # Published on the env so other terms (the diversity_meta obs term) can read it.
        if not hasattr(env, "diversity_task_done"):
            env.diversity_task_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.task_done = env.diversity_task_done

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self.task_done[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        obs_group: str = "discriminator_obs",
        success_term_name: str = "progress_context",
    ) -> torch.Tensor:
        # Latch the per-step success bit from progress_context. This reward term is registered
        # after progress_context in the reward manager order, so progress_context.success is
        # already up-to-date for the current step.
        try:
            success_term = env.reward_manager.get_term_cfg(self.success_term_name).func
            cur_success = success_term.success.bool()
        except Exception:
            cur_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.task_done |= cur_success

        discriminator = getattr(env, "diversity_discriminator", None)
        skill_idx: torch.Tensor | None = getattr(env, "diversity_skill_idx", None)
        if discriminator is None or skill_idx is None:
            return torch.zeros(env.num_envs, device=env.device)

        with torch.no_grad():
            disc_obs = env.observation_manager.compute_group(self.obs_group, update_history=False)
            if isinstance(disc_obs, dict):
                disc_obs = torch.cat([disc_obs[k] for k in disc_obs.keys()], dim=-1)

            logits = discriminator(disc_obs)
            log_q = F.log_softmax(logits, dim=-1)
            log_q_z = log_q.gather(1, skill_idx.view(-1, 1)).squeeze(-1)

            use_log_prior = bool(getattr(env, "diversity_use_log_prior", True))
            scale = float(getattr(env, "diversity_reward_scale", 1.0))
            num_skills = int(getattr(env, "diversity_num_skills", logits.shape[-1]))

            if use_log_prior:
                log_p_z = -math.log(max(num_skills, 1))
                bonus = log_q_z - log_p_z
            else:
                bonus = log_q_z

            # Zero out the bonus for envs that have already completed the task this episode.
            bonus = bonus * (~self.task_done).float()

        return scale * bonus


def diversity_task_done_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env post-success latch as a (num_envs, 1) float tensor.

    Used by the algorithm (via the ``diversity_meta`` obs group in storage) to mask post-
    success transitions out of the discriminator's CE update. Returns zeros when the latch
    hasn't been set up yet (first step before :class:`DiversityReward` initializes it).
    """
    flag = getattr(env, "diversity_task_done", None)
    if flag is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return flag.float().unsqueeze(-1)
