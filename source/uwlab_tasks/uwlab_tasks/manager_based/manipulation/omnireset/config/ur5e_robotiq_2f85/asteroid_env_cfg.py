# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASTEROID data-collection and student-eval environments.

Distills :class:`Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedTrainCfg` into a
student that observes only :class:`BasePolicyCfg`.
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from ... import mdp as task_mdp
from .data_collection_rgb_cfg import (
    DataCollectionRGBEventCfg,
    RGBEventCfg,
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
)
from .privileged_training_cfg import (
    BasePolicyCfg,
    PrivilegedPolicyCfg,
    StateTriggeredAugmentationEvalEventsCfg,
    StateTriggeredAugmentationTrainEventsCfg,
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedTrainCfg,
)
from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCTrainCfg


@configclass
class PrivilegedAugmentedPolicyCfg(PrivilegedPolicyCfg):
    """Mirrors ``observations.policy`` of the augmented privileged training cfg."""

    augmentation_active_mask = ObsTerm(
        func=task_mdp.get_augmentation_active_mask,
        params={"event_name": "augmentation_handler"},
    )

    augmentation_external_wrench = ObsTerm(
        func=task_mdp.get_augmentation_external_wrench,
        params={"event_name": "augmentation_handler"},
)


@configclass
class AsteroidDistillationObservationsCfg:
    """Observation groups for the augmented-privileged state-only distillation env.

    Declared as class-level attributes (instead of being assigned in post_init) so
    the ObservationManager sees a single, consistent group structure at env
    construction. Student policy and recorded stream share BasePolicyCfg; expert
    sees PrivilegedPolicyCfg (augmentation-specific terms are attached in
    post_init, matching the pattern used in privileged_training_cfg.py).
    """

    policy: BasePolicyCfg = BasePolicyCfg()
    data_collection: BasePolicyCfg | None = BasePolicyCfg()
    expert_obs: PrivilegedPolicyCfg | None = PrivilegedPolicyCfg()
    critic = None


@configclass
class AsteroidStudentEvalObservationsCfg(AsteroidDistillationObservationsCfg):
    """Student-eval variant: only the student ``policy`` group is populated."""

    data_collection: BasePolicyCfg | None = None
    expert_obs: PrivilegedPolicyCfg | None = None


@configclass
class AsteroidDataCollectionEventCfg(StateTriggeredAugmentationEvalEventsCfg):
    """Eval-mode augmentation + non-trivial reset distribution.

    Keeps ``randomize_env_cfg_unified`` from the augmented training stack, but forces
    ``coupled_progress_range`` and ``action_scale_progress_range`` to ``(1.0, 1.0)`` so
    randomization is centered on the terminal (sysid-like) regime while preserving the
    same action config as privileged training.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.randomize_env_cfg_unified is not None:
            self.randomize_env_cfg_unified.params["coupled_progress_range"] = (1.0, 1.0)
            self.randomize_env_cfg_unified.params["action_scale_progress_range"] = (1.0, 1.0)

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.4, 0.3, 0.3],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class AsteroidStudentEvalEventCfg(StateTriggeredAugmentationEvalEventsCfg):
    """Student eval events with terminal-regime unified randomization + augmentation."""

    def __post_init__(self):
        super().__post_init__()
        if self.randomize_env_cfg_unified is not None:
            self.randomize_env_cfg_unified.params["coupled_progress_range"] = (1.0, 1.0)
            self.randomize_env_cfg_unified.params["action_scale_progress_range"] = (1.0, 1.0)

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectRestingEEGrasped",
                # "ObjectAnywhereEEGrasped",
                # "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg
):
    """State-only data-collection env for distilling the augmented privileged policy.

    Inherits from the non-privileged train cfg on purpose: the privileged /
    privileged-augmented parents mutate ``observations.policy`` /
    ``observations.critic`` in their post_init, which would clobber the
    class-level :class:`AsteroidDistillationObservationsCfg` groups. Events are
    supplied directly via :class:`AsteroidDataCollectionEventCfg` (augmentation
    handler + non-trivial resets), scene robot is pinned to
    ``EXPLICIT_UR5E_ROBOTIQ_2F85`` to match the explicit-actuator behavior of
    the privileged training stack.
    """

    events: AsteroidDataCollectionEventCfg = AsteroidDataCollectionEventCfg()
    observations: AsteroidDistillationObservationsCfg = AsteroidDistillationObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 32.0

        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        event_name = "augmentation_handler"
        if self.observations.expert_obs is not None:
            self.observations.expert_obs.augmentation_active_mask = ObsTerm(
                func=task_mdp.get_augmentation_active_mask, params={"event_name": event_name}
            )
            self.observations.expert_obs.augmentation_external_wrench = ObsTerm(
                func=task_mdp.get_augmentation_external_wrench, params={"event_name": event_name}
            )

        self.terminations.success = DoneTerm(
            func=task_mdp.consecutive_success_state_with_min_length,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )
        self.terminations.early_success = DoneTerm(
            func=task_mdp.early_success_termination,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )
        self.terminations.abnormal_robot = DoneTerm(
            func=task_mdp.abnormal_robot_state,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedStudentEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg
):
    """Student eval env: same physics/augmentation as data collection, single reset distribution."""

    events: AsteroidStudentEvalEventCfg = AsteroidStudentEvalEventCfg()
    observations: AsteroidStudentEvalObservationsCfg = AsteroidStudentEvalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=task_mdp.consecutive_success_state,
            params={"num_consecutive_successes": 10},
        )
        if hasattr(self.terminations, "early_success"):
            self.terminations.early_success = None


# =============================================================================
# RGB variants: student observes images; expert still observes state.
# =============================================================================


@configclass
class AsteroidRGBDataCollectionEventCfg(DataCollectionRGBEventCfg):
    """RGB sim2real-targeted events + eval-mode augmentation_handler.

    Inherits from ``DataCollectionRGBEventCfg`` which already provides:
      - Visual randomization (8-mesh appearance randomization, HDRI sky light).
      - Camera pose + focal-length randomization (front / side / wrist).
    Replaces the inherited fixed arm/gain randomization with
    ``randomize_env_cfg_unified`` pinned at terminal progress
    (``coupled_progress_range = action_scale_progress_range = (1.0, 1.0)``), matching
    the state-based distillation fix. Also adds the augmentation_handler in eval mode
    and rebiases resets toward non-trivial starting states.
    """

    randomize_arm_sysid: EventTerm | None = None
    randomize_osc_gains: EventTerm | None = None

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.4, 0.3, 0.3],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    def __post_init__(self):
        unified_term = StateTriggeredAugmentationTrainEventsCfg().randomize_env_cfg_unified
        self.randomize_env_cfg_unified = unified_term
        if self.randomize_env_cfg_unified is not None:
            self.randomize_env_cfg_unified.params["coupled_progress_range"] = (1.0, 1.0)
            self.randomize_env_cfg_unified.params["action_scale_progress_range"] = (1.0, 1.0)

        self.augmentation_handler = StateTriggeredAugmentationTrainEventsCfg().augmentation_handler
        if self.augmentation_handler is not None:
            self.augmentation_handler.params["eval_mode"] = True


@configclass
class AsteroidRGBStudentEvalEventCfg(AsteroidRGBDataCollectionEventCfg):
    """RGB student-eval events + eval-mode augmentation_handler."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectRestingEEGrasped",
                # "ObjectAnywhereEEGrasped",
                # "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedRGBDistillationCfg(
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg
):
    """RGB data-collection env for distilling the augmented privileged policy.

    Student observes RGB images (``policy`` = ``RGBPolicyCfg``); recorded stream is
    ``data_collection`` = ``RGBDataCollectionCfg``; expert sees
    ``PrivilegedAugmentedPolicyCfg`` (state + augmentation info).
    """

    events: AsteroidRGBDataCollectionEventCfg = AsteroidRGBDataCollectionEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.observations.expert_obs = PrivilegedAugmentedPolicyCfg()

        if hasattr(self.observations, "critic"):
            self.observations.critic = None


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedRGBStudentEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedRGBDistillationCfg
):
    """RGB student eval env: single reset distribution, clean success termination."""

    events: AsteroidRGBStudentEvalEventCfg = AsteroidRGBStudentEvalEventCfg()

    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.observations, "data_collection"):
            self.observations.data_collection = None
        if hasattr(self.observations, "expert_obs"):
            self.observations.expert_obs = None

        self.terminations.success = DoneTerm(
            func=task_mdp.consecutive_success_state,
            params={"num_consecutive_successes": 10},
        )
        if hasattr(self.terminations, "early_success"):
            self.terminations.early_success = None
