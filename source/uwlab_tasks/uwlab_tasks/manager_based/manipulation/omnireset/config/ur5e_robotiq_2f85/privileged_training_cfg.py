from isaaclab.sim import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCTrainCfg, TrainEventCfg, ObservationsCfg, Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg
from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

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

    """Arm Dynamics Observation Terms"""

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

    # robot_delay = ObsTerm(func=task_mdp.get_action_delay, params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "arm"})

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
class RandomizeContactDynamicsTrainEventsCfg(TrainEventCfg):
    """Randomize contact dynamics for the UR5e + Robotiq 2F-85 robot and the objects."""

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="reset",
        params={
            "static_friction_range": (0.1, 5.0),
            "dynamic_friction_range": (0.1, 5.0),
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
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainWithContactDynamicsCfg(Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg):
    """Privileged observation training configuration for the UR5e + Robotiq 2F-85 robot with contact dynamics."""

    events: RandomizeContactDynamicsTrainEventsCfg = RandomizeContactDynamicsTrainEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.privileged_info = PrivilegedInfoObservationCfg()

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