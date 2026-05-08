# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


@configclass
class BehaviorCloningCfg:
    experts_path: list[str] = MISSING  # type: ignore
    """The path to the expert data."""

    experts_loader: callable = "torch.jit.load"
    """The function to construct the expert. Default is None, for which is loaded in the same way student is loaded."""

    experts_env_mapping_func: callable = None
    """The function to map the expert to env_ids. Default is None, for which is mapped to all env_ids"""

    experts_observation_group_cfg: str | None = None
    """The observation group of the expert which may be different from student"""

    experts_observation_func: callable = None
    """The function that returns expert observation data, default is None, same as student observation."""

    experts_action_group_cfg: str | None = None
    """The action group of the expert which may be different from student"""

    learn_std: bool = False
    """Whether to learn the standard deviation of the expert policy."""

    cloning_loss_coeff: float = MISSING  # type: ignore
    """The coefficient for the cloning loss."""

    loss_decay: float = 1.0
    """The decay for the cloning loss coefficient. default to 1, no decay."""


@configclass
class OffPolicyAlgorithmCfg:
    """Configuration for the off-policy algorithm."""

    update_frequencies: float = 1
    """The frequency to update relative to online update."""

    batch_size: int | None = None
    """The batch size for the offline algorithm update, default to None, same of online size."""

    num_learning_epochs: int | None = None
    """The number of learning epochs for the offline algorithm update."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the offline behavior cloning(dagger)."""

@configclass
class DiscriminatorTrainingCfg:
    """Configuration for the DIAYN-style discriminator q(z | s).

    Trained with cross-entropy to predict the skill index from the next state. Hyperparameters
    cover both the network parameterization and the optimisation schedule.
    """

    # -- Optimiser
    learning_rate: float = 1.0e-3
    """Adam learning rate for the discriminator."""

    weight_decay: float = 0.0
    """Weight decay for the discriminator optimiser."""

    num_mini_batches: int = 4
    """Mini-batches per update (rollout is split into this many minibatches per epoch)."""

    num_learning_epochs: int = 4
    """Epochs over the rollout per discriminator update."""

    update_frequency: int = 1
    """How often (in PPO updates) to run the discriminator update. 1 = every PPO update."""

    max_grad_norm: float = 1.0
    """Gradient clip for the discriminator."""

    # -- Network parameterization
    hidden_dims: list[int] = [256, 256]
    """MLP hidden sizes for q(z | s)."""

    activation: str = "elu"
    """Activation function for the discriminator MLP."""

    obs_normalization: bool = True
    """Whether to apply EmpiricalNormalization to the discriminator input."""

    obs_group: str = "discriminator_obs"
    """Observation group fed into the discriminator (s_{t+1})."""

    # -- Reward shaping
    reward_scale: float = 1.0
    """Multiplier on the diversity bonus before it is added to the env reward."""

    use_log_prior: bool = True
    """If True, the bonus is log q(z|s) - log p(z) (DIAYN). If False, just log q(z|s)."""

    label_smoothing: float = 0.0
    """Optional CE label smoothing for stability when q is overconfident."""

@configclass
class RslRlFancyActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the fancy actor-critic networks."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation."""

    noise_std_type: Literal["scalar", "log", "gsde"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    film_obs_key: str | None = None
    """The observation group that's encoded into a latent space for feature wise linear modulation."""

    film_application_mode: Literal["actor", "critic", "both"] = "actor"
    """Where FiLM is applied when `film_obs_key` is provided."""

    film_hiddens: list[int] = [128]

    privileged_obs_encoder_dims: list[int] = [128]

    use_privileged_obs_encoder: bool = False

@configclass
class RslRlFancyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the online behavior cloning."""

    offline_algorithm_cfg: OffPolicyAlgorithmCfg | None = None
    """The configuration for the offline algorithms."""

    weight_decay: float = 0.0
    """The weight decay for the optimizer."""

@configclass
class RslRlDiversityPpoAlgorithmCfg(RslRlFancyPpoAlgorithmCfg):
    """PPO + DIAYN-style skill discriminator."""

    class_name: str = "DiversityPPO"
    """Algorithm class resolved by the runner. Maps to ``rsl_rl.algorithms.DiversityPPO``."""

    discriminator_cfg: DiscriminatorTrainingCfg = DiscriminatorTrainingCfg()
    """Discriminator network + training configuration."""

    number_of_skills: int = 10
    """Size of the skill alphabet ``z``. Skills are sampled uniformly per-episode."""

    skill_obs_key: str = "skill"
    """Name of the per-term skill observation in the policy obs group."""