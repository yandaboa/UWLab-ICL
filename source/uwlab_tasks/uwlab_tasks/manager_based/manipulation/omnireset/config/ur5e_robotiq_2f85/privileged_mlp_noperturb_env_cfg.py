# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-action-perturbation distillation env, built directly on top of
``Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg``.

The earlier version inherited from the augmented-privileged distillation cfg and
nulled out the perturbation events. That used the wrong reset mix and the wrong
action-scale defaults, so the expert was effectively out of distribution and
collection success rate hovered around 49%. This rewrite reuses the exact
events / actions / scene / sim that the RL teacher was trained against and
swaps in the distillation-friendly observation groups (BasePolicyCfg for
policy / data_collection / expert_obs, no privileged critic). It also adds
success terminations so successful expert episodes end early.

Used for the Gaussian-head sanity check (experiment A): if the priv-MLP can
solve this env (no per-step action perturbation, in-distribution gain
randomization), the head + training pipeline are sound.
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .asteroid_env_cfg import (
    AsteroidDistillationObservationsCfg,
    AsteroidStudentEvalObservationsCfg,
)
from .privileged_training_cfg import (
    RandomizeGainsTrainEventsCfg,
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg,
)


@configclass
class NoPerturbDataCollectionEventsCfg(RandomizeGainsTrainEventsCfg):
    """Data-collection event cfg: same gain randomization as the training env, but the
    reset distribution is restricted to the 3 grasped-EE paths (matching
    ``AsteroidDataCollectionEventCfg`` minus the augmentation handler).

    The 4-path TrainEventCfg distribution that PrivilegedTrainCfg inherits includes
    ``ObjectAnywhereEEAnywhere`` (peg on table, EE elsewhere, neither grasped). That
    reset path empirically deadlocked Isaac Sim mid-collection — multiple v* runs
    hung for hours after a few thousand demos. Dropping it (mirroring what the
    augmented data-collection env already does) avoids the hang."""

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
class NoPerturbStudentEvalEventsCfg(RandomizeGainsTrainEventsCfg):
    """Eval-time event cfg: same gain randomization as the training env, but the
    reset distribution is restricted to a single canonical reset type
    (``ObjectAnywhereEEAnywhere``) so EOE numbers are reported on a single
    well-defined starting distribution rather than the 4-path training mix."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedNoPerturbDistillationCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg
):
    """Data-collection env: same physics/events/actions as the privileged training env
    (``RandomizeGainsTrainEventsCfg``: 4-path resets + scale/gain domain randomization,
    no action perturbation). Observations are reorganized for distillation: student
    sees BasePolicyCfg, data_collection and expert_obs both record BasePolicyCfg, no
    privileged critic. Success terminations cause successful episodes to end early."""

    events: NoPerturbDataCollectionEventsCfg = NoPerturbDataCollectionEventsCfg()
    observations: AsteroidDistillationObservationsCfg = AsteroidDistillationObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # PrivilegedTrainCfg's post_init reassigns observations.policy/critic. Override
        # afterwards with the distillation observation layout (policy/data_collection/
        # expert_obs all = BasePolicyCfg, no critic).
        self.observations = AsteroidDistillationObservationsCfg()

        # End successful episodes early — without these, a successful expert pose
        # still runs the full horizon, which inflates wall-clock per collected demo
        # and lets the robot drift into bad states post-success.
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
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedNoPerturbStudentEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedNoPerturbDistillationCfg
):
    """Student-eval variant: same gain randomization as training, but the reset
    distribution is restricted to ``ObjectAnywhereEEAnywhere`` only (instead of the
    4-path training mix) so EOE is measured on a single canonical starting
    distribution. Observations are populated only for the student's ``policy``
    group; success termination is disabled so the student must hold success until
    horizon (makes EOE strictly more demanding than any-time)."""

    events: NoPerturbStudentEvalEventsCfg = NoPerturbStudentEvalEventsCfg()
    observations: AsteroidStudentEvalObservationsCfg = AsteroidStudentEvalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations = AsteroidStudentEvalObservationsCfg()
        # EOE metric requires the policy to *hold* success until the timer expires.
        # Disabling early-terminations makes any-time-success and EOE comparable.
        self.terminations.success = None
        if hasattr(self.terminations, "early_success"):
            self.terminations.early_success = None
