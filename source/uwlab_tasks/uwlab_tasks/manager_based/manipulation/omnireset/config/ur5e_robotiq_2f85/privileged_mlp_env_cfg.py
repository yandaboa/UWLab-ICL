# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged-MLP env: same physics as the augmented-privileged distillation env, but
the student observes the action perturbation `(scale, offset)` directly via two new
obs terms appended to ``observations.policy``.

Used to bound the BC distillation gap independent of identification: a Markovian MLP
with the perturbation handed to it should achieve near-oracle performance, and the
gap to the context-conditioned student isolates how much the context policy fails at
identifying the perturbation.
"""
from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .asteroid_env_cfg import (
    AsteroidDistillationObservationsCfg,
    AsteroidStudentEvalObservationsCfg,
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg,
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedStudentEvalCfg,
)
from .privileged_training_cfg import BasePolicyCfg


@configclass
class PrivilegedKnownPerturbationPolicyCfg(BasePolicyCfg):
    """``BasePolicyCfg`` augmented with the per-env action perturbation parameters.

    Adds the 6-dim raw additive offset and the 6-dim per-axis scale (post-randomization)
    so the student has a sufficient statistic for the action perturbation. Both terms
    are floats and will be normalized by ``LinearNormalizer`` from the dataset.
    """

    action_offset = ObsTerm(
        func=task_mdp.get_action_offset,
        params={"action_name": "arm"},
    )
    action_scale = ObsTerm(
        func=task_mdp.get_action_scale,
        params={"action_name": "arm"},
    )


@configclass
class PrivilegedKnownPerturbationDistillationObservationsCfg(AsteroidDistillationObservationsCfg):
    """Distillation env: student / data_collection / expert all see the perturbation."""

    policy: PrivilegedKnownPerturbationPolicyCfg = PrivilegedKnownPerturbationPolicyCfg()
    data_collection: PrivilegedKnownPerturbationPolicyCfg | None = PrivilegedKnownPerturbationPolicyCfg()
    expert_obs: PrivilegedKnownPerturbationPolicyCfg | None = PrivilegedKnownPerturbationPolicyCfg()


@configclass
class PrivilegedKnownPerturbationStudentEvalObservationsCfg(AsteroidStudentEvalObservationsCfg):
    """Student-eval env: only the ``policy`` group is populated, mirroring the
    ``AsteroidStudentEvalObservationsCfg`` parent's behavior."""

    policy: PrivilegedKnownPerturbationPolicyCfg = PrivilegedKnownPerturbationPolicyCfg()
    data_collection: PrivilegedKnownPerturbationPolicyCfg | None = None
    expert_obs: PrivilegedKnownPerturbationPolicyCfg | None = None


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedKnownPerturbationDistillationCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg
):
    """Distillation cfg: same as the augmented-privileged distillation env but the
    student's observation group includes the action perturbation params."""

    observations: PrivilegedKnownPerturbationDistillationObservationsCfg = (
        PrivilegedKnownPerturbationDistillationObservationsCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedKnownPerturbationStudentEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedStudentEvalCfg
):
    """Student-eval cfg: same as the augmented-privileged student-eval env but the
    policy obs group includes the action perturbation params."""

    observations: PrivilegedKnownPerturbationStudentEvalObservationsCfg = (
        PrivilegedKnownPerturbationStudentEvalObservationsCfg()
    )
