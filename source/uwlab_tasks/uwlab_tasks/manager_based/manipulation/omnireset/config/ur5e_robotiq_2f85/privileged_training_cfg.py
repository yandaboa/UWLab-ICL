from isaaclab.sim import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCTrainCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

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

    robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1

@configclass
class Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Privileged observation training configuration for the UR5e + Robotiq 2F-85 robot."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = PrivilegedPolicyCfg()