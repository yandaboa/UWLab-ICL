# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-perturbation distillation env: same ASTEROID Distillation/StudentEval scaffolding
as the augmented variant, but with ``randomize_env_cfg_unified`` and
``augmentation_handler`` disabled so the action perturbation events do not fire.

Used for the Gaussian-head sanity check: if the priv-MLP can solve this Markovian
peg env (no per-episode action perturbation), the head + training pipeline are sound,
and the priv-MLP failure on the augmented env is fundamentally about action
multimodality given (state, perturbation), not a bug.
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .asteroid_env_cfg import (
    AsteroidDataCollectionEventCfg,
    AsteroidStudentEvalEventCfg,
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg,
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedStudentEvalCfg,
)


@configclass
class NoPerturbDataCollectionEventCfg(AsteroidDataCollectionEventCfg):
    """Strip action perturbation: disable augmentation_handler and
    randomize_env_cfg_unified so neither offset nor scale ever changes."""

    def __post_init__(self):
        super_post = getattr(super(), "__post_init__", None)
        if callable(super_post):
            super_post()
        self.augmentation_handler = None
        self.randomize_env_cfg_unified = None


@configclass
class NoPerturbStudentEvalEventCfg(AsteroidStudentEvalEventCfg):
    """Same strip on the eval side."""

    def __post_init__(self):
        super_post = getattr(super(), "__post_init__", None)
        if callable(super_post):
            super_post()
        self.augmentation_handler = None
        self.randomize_env_cfg_unified = None


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedNoPerturbDistillationCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg
):
    """Distillation cfg with no action perturbation. Observations stay
    ``AsteroidDistillationObservationsCfg`` (BasePolicyCfg only — no offset/scale,
    since there is no perturbation to expose)."""

    events: NoPerturbDataCollectionEventCfg = NoPerturbDataCollectionEventCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedNoPerturbStudentEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedStudentEvalCfg
):
    """Student-eval cfg with no action perturbation."""

    events: NoPerturbStudentEvalEventCfg = NoPerturbStudentEvalEventCfg()
