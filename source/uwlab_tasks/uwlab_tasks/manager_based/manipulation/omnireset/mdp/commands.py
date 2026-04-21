# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm

from ..assembly_keypoints import Offset
from . import utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TaskCommandCfg, TaskDependentCommandCfg


class TaskDependentCommand(CommandTerm):
    cfg: TaskDependentCommandCfg

    def __init__(self, cfg: TaskDependentCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.reset_terms_when_resample = cfg.reset_terms_when_resample
        self.interval_reset_terms = []
        self.reset_terms = []
        self.ALL_INDICES = torch.arange(self.num_envs, device=self.device)
        for name, term_cfg in self.reset_terms_when_resample.items():
            if not (term_cfg.mode == "reset" or term_cfg.mode == "interval"):
                raise ValueError(f"Term '{name}' in 'reset_terms_when_resample' must have mode 'reset' or 'interval'")
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            if term_cfg.mode == "reset":
                self.reset_terms.append(term_cfg)
            elif term_cfg.mode == "interval":
                if term_cfg.interval_range_s != (0, 0):
                    raise ValueError(
                        "task dependent events term with interval mode current only supports range of (0, 0)"
                    )
                self.interval_reset_terms.append(term_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        for term in self.reset_terms:
            func = term.func
            func(self._env, env_ids, **term.params)
        for term in self.interval_reset_terms:
            func = term.func
            func.reset(env_ids)

    def _update_command(self):
        for term in self.interval_reset_terms:
            func = term.func
            func(self._env, self.ALL_INDICES, **term.params)

    def get_event(self, event_term_name: str):
        """Get the event term by name."""
        return self.reset_terms_when_resample.get(event_term_name).func


class TaskCommand(TaskDependentCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TaskCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TaskCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]
        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )
        self.receptive_asset_offset = Offset(
            pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
            quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
        )
        self.success_position_threshold: float = receptive_meta.get("success_thresholds").get("position")
        self.success_orientation_threshold: float = receptive_meta.get("success_thresholds").get("orientation")

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        # Lazily-resolved augmentation handler metrics. CommandManager is built before
        # EventManager, so we can't bind to the handler here; resolve on first
        # ``_update_metrics`` instead. CommandManager averages metric tensors across
        # envs at reset, so a bool mask cast to float becomes the fraction of envs
        # currently being augmented.
        self._augmentation_event_name: str | None = getattr(cfg, "augmentation_event_name", None)
        self._augmentation_handler = None
        self._augmentation_categories: tuple[str, ...] = ()
        self._augmentation_resolved: bool = False

        # Lazily-resolved OSC action term; used to log pre-noise task-frame force.
        # ActionManager is also built after CommandManager, hence the deferred resolve.
        self._arm_action_name: str | None = getattr(cfg, "arm_action_name", None)
        self._arm_action_term = None
        self._arm_action_resolved: bool = False

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xy_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current data
        insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
            self.insertive_asset
        )
        receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self.receptive_asset_offset.apply(
            self.receptive_asset
        )
        insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                receptive_asset_alignment_pos_w,
                receptive_asset_alignment_quat_w,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
        )
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

        self._maybe_resolve_augmentation_handler()
        if self._augmentation_handler is not None:
            active_mask = self._augmentation_handler._active_mask
            any_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for cat in self._augmentation_categories:
                cat_mask = active_mask[cat]
                self.metrics[f"augmentation_active_frac/{cat}"][:] = cat_mask.float()
                any_active = any_active | cat_mask
            self.metrics["augmentation_active_frac/any"][:] = any_active.float()

        self._maybe_resolve_arm_action_term()
        if self._arm_action_term is not None:
            # Pre-bias task-frame wrench snapshot from the last physics substep.
            # CommandManager averages across envs on reset, giving mean magnitudes.
            tf = self._arm_action_term._last_task_force_pre_bias
            self.metrics["task_force_pre_bias/force_norm"][:] = torch.norm(tf[:, :3], dim=-1)
            self.metrics["task_force_pre_bias/torque_norm"][:] = torch.norm(tf[:, 3:], dim=-1)
            self.metrics["task_force_pre_bias/abs_mean"][:] = tf.abs().mean(dim=-1)

    def _maybe_resolve_augmentation_handler(self) -> None:
        """Resolve the augmentation event term on first call and lazily register
        per-category activation-fraction metrics. No-ops if the term is missing or
        isn't a ``conditional_arm_augmentation`` instance."""
        if self._augmentation_resolved:
            return
        event_manager = getattr(self._env, "event_manager", None)
        if event_manager is None or self._augmentation_event_name is None:
            return
        try:
            term_cfg = event_manager.get_term_cfg(self._augmentation_event_name)
        except ValueError:
            self._augmentation_resolved = True
            return
        handler = getattr(term_cfg, "func", None)
        categories = getattr(handler, "_CATEGORIES", None)
        if handler is not None and categories is not None and hasattr(handler, "_active_mask"):
            self._augmentation_handler = handler
            self._augmentation_categories = tuple(categories)
            for cat in self._augmentation_categories:
                self.metrics[f"augmentation_active_frac/{cat}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["augmentation_active_frac/any"] = torch.zeros(self.num_envs, device=self.device)
        self._augmentation_resolved = True

    def _maybe_resolve_arm_action_term(self) -> None:
        """Resolve the OSC action term on first call and register pre-bias task-force
        magnitude metrics. No-ops if the term is missing or lacks the expected buffer."""
        if self._arm_action_resolved:
            return
        action_manager = getattr(self._env, "action_manager", None)
        if action_manager is None or self._arm_action_name is None:
            return
        try:
            term = action_manager.get_term(self._arm_action_name)
        except (KeyError, ValueError):
            self._arm_action_resolved = True
            return
        if term is not None and hasattr(term, "_last_task_force_pre_bias"):
            self._arm_action_term = term
            for key in (
                "task_force_pre_bias/force_norm",
                "task_force_pre_bias/torque_norm",
                "task_force_pre_bias/abs_mean",
            ):
                self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
        self._arm_action_resolved = True

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
