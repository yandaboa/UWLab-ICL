from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class SupervisedContextDataCfg:
    """Dataset configuration for supervised context training."""

    episode_paths: list[str] = [
        "episodes/20260218_023144/episodes_000000.pt",
        "episodes/20260218_023144/episodes_000001.pt",
        "episodes/20260218_023144/episodes_000002.pt",
        "episodes/20260218_023144/episodes_000003.pt",
        "episodes/20260218_023144/episodes_000004.pt",
        "episodes/20260218_023144/episodes_000005.pt",
        "episodes/20260218_023144/episodes_000006.pt",
        "episodes/20260218_023144/episodes_000007.pt",
        "episodes/20260218_023144/episodes_000008.pt",
        "episodes/20260218_023144/episodes_000009.pt",
    ]
    """List of episode .pt files or glob patterns."""

    obs_keys: list[str] | None = ["joint_pos", "end_effector_pose"]
    """Optional ordered obs keys for dict observations."""

    max_context_length: int | None = None
    """Optional cap on context length per episode."""

    batch_size: int = 128
    """Batch size for training."""

    num_workers: int = 4
    """Data loader workers."""

    shuffle: bool = True
    """Shuffle episodes in the dataset."""


@configclass
class SupervisedContextModelCfg:
    """Model configuration for supervised context training."""

    num_actions: int = 7  # type: ignore
    """Action dimension (always inferred from data or checkpoint state, overrides this value. This is just a placeholder)."""

    action_distribution: Literal["normal", "categorical"] = "categorical"
    """Action distribution type."""

    action_discretization_spec_path: str = ""
    """Optional path to action discretization spec (defaults to episode folder)."""

    context_token_layout: str = "state_only"
    """Token layout: merged, state_action, state_only."""

    include_actions_in_context: bool = True
    """Include action terms in merged context tokens."""

    include_rewards_in_context: bool = True
    """Include reward terms in merged context tokens."""

    share_current_and_context_obs_projection: bool = True
    """Reuse one projection for current_obs and context_obs; requires matching feature sizes."""

    encoding_projection_hidden_dim: int | None = None
    """Optional hidden size for obs encoders (in->hidden->embedding instead of single linear)."""

    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 4
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0


@configclass
class SupervisedContextOptimizationCfg:
    """Optimization configuration for supervised context training."""

    num_steps: int = 100000
    """Total optimizer steps."""

    learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    optimizer_class: str = "AdamW"
    lr_warmup_steps: int = 200
    lr_schedule: str | None = "cosine_annealing_with_warmup"

    use_amp: bool = True
    """Use automatic mixed precision when CUDA is available."""


@configclass
class SupervisedContextInputCfg:
    """Input configuration for supervised context training."""

    include_current_trajectory: bool = False
    """Whether to append current rollout to context before prediction (disabled)."""


@configclass
class SupervisedContextDistributedCfg:
    """Distributed configuration for supervised context training."""

    distributed: bool = False
    """Enable multi-GPU distributed training."""


@configclass
class SupervisedContextLoggingCfg:
    """Logging configuration for supervised context training."""

    experiment_name: str = "supervised_context"
    run_name: str = "default"
    log_interval: int = 1
    save_interval: int = 5000
    log_project_name: str | None = None
    use_wandb: bool = True


@configclass
class SupervisedContextTrainerCfg:
    """Top-level configuration for supervised context training."""

    data: SupervisedContextDataCfg = SupervisedContextDataCfg()
    model: SupervisedContextModelCfg = SupervisedContextModelCfg()
    optim: SupervisedContextOptimizationCfg = SupervisedContextOptimizationCfg()
    input: SupervisedContextInputCfg = SupervisedContextInputCfg()
    distributed: SupervisedContextDistributedCfg = SupervisedContextDistributedCfg()
    logging: SupervisedContextLoggingCfg = SupervisedContextLoggingCfg()
