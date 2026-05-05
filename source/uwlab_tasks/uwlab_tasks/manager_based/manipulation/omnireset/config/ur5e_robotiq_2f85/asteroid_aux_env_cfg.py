# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASTEROID env variant with action perturbation parameters logged into the
``data_collection`` observation group only.

The ``policy`` (student input) group remains :class:`BasePolicyCfg` — same as the
standard ASTEROID env, so the student does not directly see ``(action_offset,
action_scale)``. But the ``data_collection`` group, which is what gets persisted
to zarr, includes both perturbation params so they are available as **aux loss
targets** during training.

Used by experiment C: train a markovian-context disc-AR student with an
auxiliary head that reconstructs the per-episode ``(action_offset, action_scale)``
from the trunk's last hidden state. Tests whether forcing the trunk to form a
latent perturbation belief improves performance over the standard ASTEROID disc-AR.
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
class AuxDataCollectionPolicyCfg(BasePolicyCfg):
    """``BasePolicyCfg`` augmented with per-env action perturbation params.

    Logged into the zarr via the ``data_collection`` obs group, so they're
    available as targets for a perturbation-reconstruction aux loss. The
    ``policy`` group remains BasePolicyCfg, so the student never sees these
    as direct inputs — only the aux head consumes them.
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
class AsteroidAuxDistillationObservationsCfg(AsteroidDistillationObservationsCfg):
    """Distillation env: policy=BasePolicyCfg, data_collection includes aux info,
    expert_obs=BasePolicyCfg (the expert was trained on BasePolicyCfg so it cannot
    consume the augmented variant)."""

    data_collection: AuxDataCollectionPolicyCfg | None = AuxDataCollectionPolicyCfg()


@configclass
class AsteroidAuxStudentEvalObservationsCfg(AsteroidStudentEvalObservationsCfg):
    """Student-eval env: only the policy group is populated, mirroring the
    parent's ``data_collection=None, expert_obs=None`` behavior."""

    pass


@configclass
class Ur5eRobotiq2f85RelCartesianOSCAuxDistillationCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedDistillationCfg
):
    """Distillation cfg whose ``data_collection`` group also logs the action
    perturbation parameters as aux targets."""

    observations: AsteroidAuxDistillationObservationsCfg = (
        AsteroidAuxDistillationObservationsCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCAuxStudentEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedStudentEvalCfg
):
    """Student-eval cfg matching :class:`Ur5eRobotiq2f85RelCartesianOSCAuxDistillationCfg`."""

    observations: AsteroidAuxStudentEvalObservationsCfg = (
        AsteroidAuxStudentEvalObservationsCfg()
    )
