from isaaclab.sim import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import (
    ObservationsCfg,
    RlStateSceneCfg,
    TrainEventCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)
from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg

from ... import mdp as task_mdp
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR


@configclass
class BasePolicyCfg(ObsGroup):
    """Base policy observations for the UR5e + Robotiq 2F-85 robot."""

    prev_actions = ObsTerm(func=task_mdp.last_action)

    joint_pos = ObsTerm(func=task_mdp.joint_pos)

    joint_vel = ObsTerm(func=task_mdp.joint_vel)

    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
    )

    end_effector_vel_lin_ang_b = ObsTerm(
        func=task_mdp.asset_link_velocity_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
        },
    )

    insertive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
        },
    )

    receptive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
        self.history_length = 1

@configclass
class PrivilegedPolicyCfg(ObsGroup):
    """Privileged policy observations for the UR5e + Robotiq 2F-85 robot."""

    prev_actions = ObsTerm(func=task_mdp.last_action)

    joint_pos = ObsTerm(func=task_mdp.joint_pos)

    joint_vel = ObsTerm(func=task_mdp.joint_vel)

    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
    )

    end_effector_vel_lin_ang_b = ObsTerm(
        func=task_mdp.asset_link_velocity_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
        },
    )

    insertive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
        },
    )

    receptive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
        },
    )

    insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
        },
    )

    """Contact Dynamics Observation Terms"""

    # insertive_object_material_properties = ObsTerm(
    #     func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    # )

    # receptive_object_material_properties = ObsTerm(
    #     func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    # )

    # table_material_properties = ObsTerm(
    #     func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
    # )

    # robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

    # insertive_object_mass = ObsTerm(
    #     func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    # )

    # receptive_object_mass = ObsTerm(
    #     func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    # )

    # table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

    """Arm Dynamics Observation Terms"""

    robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_joint_stiffness = ObsTerm(
        func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_joint_dynamic_friction = ObsTerm(
        func=task_mdp.get_joint_dynamic_friction, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_joint_viscous_friction = ObsTerm(
        func=task_mdp.get_joint_viscous_friction, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_osc_gains = ObsTerm(func=task_mdp.get_osc_gains, params={"action_name": "arm"})

    robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_action_scale = ObsTerm(func=task_mdp.get_action_scale, params={"action_name": "arm"})

    robot_delay = ObsTerm(func=task_mdp.get_action_delay, params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "arm"})

    """Augmentation-specific observation terms.

    Existing obs (joint friction/armature/damping, OSC gains, action scale) already
    reflect the *current* augmented value because they read live sim/action-term
    state. These terms expose augmentation-only signals that aren't otherwise
    visible: the raw action offset, the task-frame force bias, the external wrench
    applied to a wrench asset, and the per-category active mask. Safe to include
    even when ``conditional_arm_augmentation`` isn't in the event config:
    ``action_offset`` / ``task_frame_force_bias`` just stay at zero, and the
    mask/wrench terms would raise at construction if the event is absent, so they
    are only enabled under ``*AugmentedTrainCfg`` via the subclass below.
    """
    robot_action_offset = ObsTerm(func=task_mdp.get_action_offset, params={"action_name": "arm"})

    robot_task_frame_force_bias = ObsTerm(
        func=task_mdp.get_task_frame_force_bias, params={"action_name": "arm"}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
        self.history_length = 1


@configclass
class PrivilegedCriticCfg(ObservationsCfg.CriticCfg):
    """Privileged critic observations for the UR5e + Robotiq 2F-85 robot."""

    robot_joint_dynamic_friction = ObsTerm(
        func=task_mdp.get_joint_dynamic_friction, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_joint_viscous_friction = ObsTerm(
        func=task_mdp.get_joint_viscous_friction, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    robot_osc_gains = ObsTerm(func=task_mdp.get_osc_gains, params={"action_name": "arm"})

    robot_action_scale = ObsTerm(func=task_mdp.get_action_scale, params={"action_name": "arm"})

    robot_delay = ObsTerm(func=task_mdp.get_action_delay, params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "arm"})

    """Augmentation-specific observation terms (see PrivilegedPolicyCfg for notes)."""
    robot_action_offset = ObsTerm(func=task_mdp.get_action_offset, params={"action_name": "arm"})

    robot_task_frame_force_bias = ObsTerm(
        func=task_mdp.get_task_frame_force_bias, params={"action_name": "arm"}
    )

    def __post_init__(self):
        super().__post_init__()

@configclass
class PrivilegedInfoObservationCfg(ObsGroup):
    """Privileged information observations for the UR5e + Robotiq 2F-85 robot."""

    """Contact dynamics observations"""
    insertive_object_material_properties = ObsTerm(
        func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    )

    receptive_object_material_properties = ObsTerm(
        func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    )

    table_material_properties = ObsTerm(
        func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
    )

    robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

    insertive_object_mass = ObsTerm(
        func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    )

    receptive_object_mass = ObsTerm(
        func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    )

    table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

    """Arm dynamics observations"""
    # robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

    # robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

    # robot_joint_stiffness = ObsTerm(
    #     func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # robot_joint_dynamic_friction = ObsTerm(
    #     func=task_mdp.get_joint_dynamic_friction, params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # robot_joint_viscous_friction = ObsTerm(
    #     func=task_mdp.get_joint_viscous_friction, params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # robot_osc_gains = ObsTerm(func=task_mdp.get_osc_gains, params={"action_name": "arm"})

    # robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

    # robot_action_scale = ObsTerm(func=task_mdp.get_action_scale, params={"action_name": "arm"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1

@configclass
class RandomizeGainsTrainEventsCfg(TrainEventCfg):
    # We randomize over a large domain of values for the arm dynamics + OSC controller, including sys-ided values
    # a bit of cheating lol

    # randomize_arm_sysid = EventTerm(
    #     func=task_mdp.randomize_arm_from_sysid_fixed,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "joint_names": [
    #             "shoulder_pan_joint",
    #             "shoulder_lift_joint",
    #             "elbow_joint",
    #             "wrist_1_joint",
    #             "wrist_2_joint",
    #             "wrist_3_joint",
    #         ],
    #         "actuator_name": "arm",
    #         "scale_range": (0.1, 1.2),
    #         "delay_range": (0, 0),
    #     },
    # )

    randomize_env_cfg_unified = EventTerm(
        func=task_mdp.randomize_env_cfg_unified,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "actuator_name": "arm",
            "action_name": "arm",
            "arm_scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
            "kp_scale_range": (0.8, 1.2),
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "initial_scales": (0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
            "target_scales": (0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
            "coupled_progress_range": (0.0, 1.0),
            "action_scale_progress_range": (0.0, 1.0),
        },
    )

    # this is based on pre-train gains. If we want coverage over sim2real gains (refer to UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL)
    # we need to aggressively randomize
    # randomize_osc_gains = EventTerm(
    #     func=task_mdp.randomize_action_term_fields,
    #     mode="reset",
    #     params={
    #         "action_name": "arm",
    #         # "action_scale_scale_range": (0.8, 1.2), We probably dont need to randomize over scale?
    #         "kp_scale_range": (0.8, 5.0)
    #     },
    # )


@configclass
class StateTriggeredAugmentationTrainEventsCfg(RandomizeGainsTrainEventsCfg):
    """State/step-triggered augmentation layered on top of gain randomization.

    - ``activation_start_step_range`` is the (inclusive) step range from which a random
      activation start step is sampled per env at reset, clamped to
      ``[0, env.max_episode_length]``. When ``sample_augmentation_independently=True``,
      a *different* start step is sampled per env per augmentation category
      (``dynamics``, ``gains``, ``action``, ``force``).
    - Every magnitude parameter is a ``(lo, hi)`` range sampled uniformly per env at
      the moment its category transitions to active. Each of ``lo`` / ``hi`` may be a
      scalar (broadcast) or a per-dim sequence.
    - ``curriculum_*`` params shrink ``activation_start_step_range`` toward
      ``curriculum_min_activation_start_step_range`` once mean success exceeds the
      threshold.
    """

    augmentation_handler = EventTerm(
        func=task_mdp.conditional_arm_augmentation,  # type: ignore[arg-type]
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "action_name": "arm",
            # Fires when insertive-to-assembled + gripper-to-insertive distances are
            # both within per-env sampled thresholds. See ProximityAugmentationActivationCondition.
            # For a contact-based alternative, see task_mdp.ContactAugmentationActivationCondition.
            "activation_expr": task_mdp.ProximityAugmentationActivationCondition(
                max_assembly_distance_range=(1.0, 1.25),
                max_gripper_to_insertive_distance_range=(0.22, 0.28),
            ),
            "activation_start_step_range": (15, 90),
            "sample_augmentation_independently": True,
            "curriculum_decay_percentage": 0.04,
            "curriculum_success_threshold": 0.5,
            "curriculum_min_activation_start_step_range": (2, 15),
            "curriculum_update_every_n_steps": 500,
            # Dynamics: per-joint multiplicative scales on the captured baseline.
            "armature_scale_range": (0.8, 1.2),
            "static_friction_scale_range": (0.8, 1.2),
            "dynamic_friction_scale_range": (0.8, 1.2),
            "viscous_friction_scale_range": (0.8, 1.2),
            # Gains: per-axis Kp/damping-ratio scales on the captured baseline.
            "kp_scale_range": (
                (0.7, 0.7, 0.7, 0.8, 0.8, 0.8),
                (1.3, 1.3, 1.3, 1.2, 1.2, 1.2),
            ),
            "damping_ratio_scale_range": (0.9, 1.1),
            # Action: per-axis action scale + raw action offset (6-DOF delta pose).
            "action_scale_range": (0.9, 1.1),
            "action_offset_range": (
                (3.0, 3.0, 3.0, 2.0, 2.0, 2.0),
                (7.0, 7.0, 7.0, 3.0, 3.0, 3.0),
                # (1.0, 1.0, 1.0, 0.5, 0.5, 0.5),
                # (4.0, 4.0, 4.0, 2.0, 2.0, 2.0),
            ),
            # Force: per-axis task-frame force bias (added to OSC task-force output).
            "task_frame_force_bias_range": (
                (3.0, 3.0, 3.0, 2.0, 2.0, 2.0),
                (10.0, 10.0, 10.0, 5.0, 5.0, 5.0),
                # (1.0, 1.0, 1.0, 0.5, 0.5, 0.5),
                # (5.0, 5.0, 5.0, 2.0, 2.0, 2.0),
            ),
        },
    )


@configclass
class StateTriggeredAugmentationEvalEventsCfg(StateTriggeredAugmentationTrainEventsCfg):
    """Play events for state/step-triggered augmentation layered on top of gain randomization."""

    def __post_init__(self):
        if self.augmentation_handler is not None:
            self.augmentation_handler.params["eval_mode"] = True

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            # "reset_types": ["ObjectAnywhereEEAnywhere"],
            "reset_types": ["ObjectRestingEEGrasped"], # making reset states easier
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


# @configclass
# class ContactTriggeredAugmentationExampleSceneCfg(RlStateSceneCfg):
#     """Example scene wiring for contact-triggered augmentation activation.

#     This class is intentionally not referenced by any env cfg. It exists as a
#     copy-paste template showing how to wire the contact sensors required by
#     ``task_mdp.ContactAugmentationActivationCondition``.
#     """

#     robot = RlStateSceneCfg.robot.replace(
#         spawn=RlStateSceneCfg.robot.spawn.replace(activate_contact_sensors=True)
#     )
#     insertive_object = RlStateSceneCfg.insertive_object.replace(  # type: ignore[attr-defined]
#         spawn=RlStateSceneCfg.insertive_object.spawn.replace(activate_contact_sensors=True)  # type: ignore[attr-defined]
#     )
#     table = RlStateSceneCfg.table.replace(  # type: ignore[attr-defined]
#         spawn=RlStateSceneCfg.table.spawn.replace(activate_contact_sensors=True)  # type: ignore[attr-defined]
#     )

#     # Single-body sensors on the insertive object with one filter each.
#     insertive_vs_gripper_contact = ContactSensorCfg(
#         prim_path="{ENV_REGEX_NS}/InsertiveObject",
#         filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*finger.*"],
#         update_period=0.0,
#         history_length=1,
#     )
#     insertive_vs_table_contact = ContactSensorCfg(
#         prim_path="{ENV_REGEX_NS}/InsertiveObject",
#         filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
#         update_period=0.0,
#         history_length=1,
#     )


# @configclass
# class ContactTriggeredAugmentationExampleEventsCfg(StateTriggeredAugmentationTrainEventsCfg):
#     """Example event cfg that uses ``ContactAugmentationActivationCondition``.

#     This class is intentionally not referenced by any env cfg.
#     """

#     def __post_init__(self):
#         if self.augmentation_handler is None:
#             return
#         self.augmentation_handler.params["activation_expr"] = task_mdp.ContactAugmentationActivationCondition(
#             groups=[
#                 task_mdp.ContactGroup(
#                     sensor_name="insertive_vs_gripper_contact",
#                     filter_idx=0,
#                     touching=True,
#                     force_threshold=1.0,
#                 ),
#                 task_mdp.ContactGroup(
#                     sensor_name="insertive_vs_table_contact",
#                     filter_idx=0,
#                     touching=False,
#                     force_threshold=1.0,
#                 ),
#             ]
#         )

@configclass
class RandomizeContactDynamicsTrainEventsCfg(TrainEventCfg):
    """Randomize contact dynamics for the UR5e + Robotiq 2F-85 robot and the objects."""

    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="reset",
        params={
            "static_friction_range": (0.1, 3.0),
            "dynamic_friction_range": (0.1, 3.0),
            # "static_friction_range": (9.0, 10.0),
            # "dynamic_friction_range": (9.0, 10.0),
            # "static_friction_range": (0.0, 0.1),
            # "dynamic_friction_range": (0.0, 0.1),
            # "restitution_range": (0.0, 0.1),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 8192,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="reset",
        params={
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.1, 1.0),
            # "static_friction_range": (0.0, 0.1),
            # "dynamic_friction_range": (0.0, 0.1),
            # "restitution_range": (0.0, 0.1),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 8192,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="reset",
        params={
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 8192,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # we assume insertive object is somewhere between 20g and 200g
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.5, 3.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

@configclass
class RandomizeContactDynamicsEvalEventsCfg(RandomizeContactDynamicsTrainEventsCfg):
    """Reset only to end effector anywhere dataset"""

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
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Privileged observation training configuration for the UR5e + Robotiq 2F-85 robot."""

    events: RandomizeGainsTrainEventsCfg = RandomizeGainsTrainEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = PrivilegedPolicyCfg()
        self.observations.critic = PrivilegedCriticCfg()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.observations.privileged_info = PrivilegedInfoObservationCfg()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedTrainCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg):
    """Privileged training configuration with state/time-triggered augmentation."""

    events: StateTriggeredAugmentationTrainEventsCfg = StateTriggeredAugmentationTrainEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # These obs terms read directly from the ``augmentation_handler`` event's
        # per-category active mask / sampled wrench buffers, so they are wired in
        # only for configs that actually include that event term.
        event_name = "augmentation_handler"
        self.observations.policy.augmentation_active_mask = ObsTerm(
            func=task_mdp.get_augmentation_active_mask, params={"event_name": event_name}
        )
        self.observations.policy.augmentation_external_wrench = ObsTerm(
            func=task_mdp.get_augmentation_external_wrench, params={"event_name": event_name}
        )
        self.observations.critic.augmentation_active_mask = ObsTerm(
            func=task_mdp.get_augmentation_active_mask, params={"event_name": event_name}
        )
        self.observations.critic.augmentation_external_wrench = ObsTerm(
            func=task_mdp.get_augmentation_external_wrench, params={"event_name": event_name}
        )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedPlayCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedAugmentedTrainCfg):
    """Privileged play configuration with state/time-triggered augmentation."""

    events: StateTriggeredAugmentationEvalEventsCfg = StateTriggeredAugmentationEvalEventsCfg()
@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainWithContactDynamicsCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg):
    """Privileged observation training configuration for the UR5e + Robotiq 2F-85 robot with contact dynamics."""

    events: RandomizeContactDynamicsTrainEventsCfg = RandomizeContactDynamicsTrainEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        # self.observations.privileged_info = PrivilegedInfoObservationCfg()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedEvalWithContactDynamicsCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg):
    """Privileged observation evaluation configuration for the UR5e + Robotiq 2F-85 robot with contact dynamics."""

    events: RandomizeContactDynamicsEvalEventsCfg = RandomizeContactDynamicsEvalEventsCfg()

    def __post_init__(self):
        super().__post_init__()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Privileged observation evaluation configuration for the UR5e + Robotiq 2F-85 robot."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = PrivilegedPolicyCfg()
        self.observations.critic = PrivilegedCriticCfg()
        # self.observations.privileged_info = PrivilegedInfoObservationCfg()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCArmDynamicsPOMDPTrainCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg):
    """No privileged information, massive arm dynamics randomization. Requires use of history"""

    def __post_init__(self):
        super().__post_init__()
        
        self.observations.policy = BasePolicyCfg()
        self.terminations.success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 10})
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"],
                "probs": [0.25, 0.25, 0.25, 0.25],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCArmDynamicsPOMDPEvalCfg(Ur5eRobotiq2f85RelCartesianOSCArmDynamicsPOMDPTrainCfg):

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_from_reset_states = EventTerm(
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
class Ur5eRobotiq2f85RelCartesianOSCContactDynamicsPOMDPTrainCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedEvalWithContactDynamicsCfg):
    """No privileged information, massive contact dynamics randomization. Requires use of history"""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = BasePolicyCfg()
        self.observations.data_collection = BasePolicyCfg()
        self.terminations.success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 10})
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"],
                "probs": [0.25, 0.25, 0.25, 0.25],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )

@configclass
class Ur5eRobotiq2f85RelCartesianOSCContactDynamicsPOMDPEvalCfg(Ur5eRobotiq2f85RelCartesianOSCContactDynamicsPOMDPTrainCfg):
    """No privileged information, massive contact dynamics randomization. Requires use of history"""

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )