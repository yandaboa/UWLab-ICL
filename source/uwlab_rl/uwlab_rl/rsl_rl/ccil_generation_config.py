from __future__ import annotations

from dataclasses import dataclass, field


def _update_dataclass_from_dict(target: object, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _update_dataclass_from_dict(current, value)
        else:
            setattr(target, key, value)


@dataclass
class CcilDataCfg:
    """Inputs/outputs for CCIL synthetic label generation."""

    episode_paths: list[str] = field(default_factory=list)
    obs_keys: list[str] | None = field(
        default_factory=lambda: [
            "joint_pos",
            "end_effector_pose",
            "insertive_asset_pose",
            "receptive_asset_pose",
            "insertive_asset_in_receptive_asset_frame",
        ]
    )
    output_dir: str = "episodes_ccil_augmented"
    output_suffix: str = "_ccil"
    overwrite: bool = False


@dataclass
class CcilModelCfg:
    """Forward dynamics checkpoint input for CCIL."""

    forward_dynamics_checkpoint: str = ""


@dataclass
class CcilGenerationHyperparamsCfg:
    """CCIL augmentation and solve hyperparameters."""

    num_augs_per_step: int = 2
    sigma_action: float = 0.01
    K: int = 200
    lr_s: float = 4.0
    eps_opt: float = 1.0e-3
    r_max: float = 2.0e-2
    max_delta_s: float = 0.5
    grad_clip_norm: float = 1.0
    action_norm_clip_min: float = -3.0
    action_norm_clip_max: float = 3.0
    solve_batch_size: int = 4096


@dataclass
class CcilRuntimeCfg:
    """Runtime options for generation scripts."""

    seed: int = 0
    device: str = "cuda"
    max_steps_per_episode: int | None = None


@dataclass
class CcilGenerationCfg:
    """Top-level CCIL generation configuration."""

    data: CcilDataCfg = field(default_factory=CcilDataCfg)
    model: CcilModelCfg = field(default_factory=CcilModelCfg)
    generation: CcilGenerationHyperparamsCfg = field(default_factory=CcilGenerationHyperparamsCfg)
    runtime: CcilRuntimeCfg = field(default_factory=CcilRuntimeCfg)

    def from_dict(self, values: dict) -> None:
        _update_dataclass_from_dict(self, values)
