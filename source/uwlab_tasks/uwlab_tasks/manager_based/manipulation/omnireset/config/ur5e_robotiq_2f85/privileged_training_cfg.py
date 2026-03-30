from isaaclab.sim import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCTrainCfg, TrainEventCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from ... import mdp as task_mdp



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

    robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

    insertive_object_mass = ObsTerm(
        func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    )

    receptive_object_mass = ObsTerm(
        func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    )

    table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

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

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
        self.history_length = 1

@configclass
class PrivilegedInfoObservationCfg(ObsGroup):
    """Privileged information observations for the UR5e + Robotiq 2F-85 robot."""

    robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

    insertive_object_mass = ObsTerm(
        func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
    )

    receptive_object_mass = ObsTerm(
        func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
    )

    table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

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

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1

@configclass
class RandomizeGainsTrainEventsCfg(TrainEventCfg):
    # We randomize over a large domain of values for the arm dynamics + OSC controller, including sys-ided values
    # a bit of cheating lol

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
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
            "scale_range": (0.1, 0.3),
            "delay_range": (0, 0),
        },
    )

    # this is based on pre-train gains. If we want coverage over sim2real gains (refer to UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL)
    # we need to aggressively randomize
    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_action_term_fields,
        mode="reset",
        params={
            "action_name": "arm",
            "action_scale_scale_range": (0.8, 1.2),
            # "kp_scale_range": (0.8, 1.2), We probably dont need to randomize over scale?
        },
    )

@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Privileged observation training configuration for the UR5e + Robotiq 2F-85 robot."""

    events: RandomizeGainsTrainEventsCfg = RandomizeGainsTrainEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = PrivilegedPolicyCfg()
        # self.observations.privileged_info = PrivilegedInfoObservationCfg()