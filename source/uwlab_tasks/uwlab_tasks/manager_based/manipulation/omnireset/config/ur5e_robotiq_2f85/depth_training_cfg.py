import torch

from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RlStateCfg, RlStateSceneCfg, TrainEventCfg, ObservationsCfg

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
    Ur5eRobotiq2f85RelativeOSCEvalAction,
)

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import mdp as task_mdp

front_camera_offset = MultiMeshRayCasterCfg.OffsetCfg(
    pos=(1.0770121, -0.1679045, 0.4486344),
    rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
)
front_camera_offset = front_camera_offset.replace(rot=convert_camera_frame_orientation_convention(torch.tensor(front_camera_offset.rot), "opengl", "world"))

side_camera_offset = MultiMeshRayCasterCfg.OffsetCfg(
    pos=(0.8323904, 0.5877843, 0.2805111),
    rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
)
side_camera_offset = side_camera_offset.replace(rot=convert_camera_frame_orientation_convention(torch.tensor(side_camera_offset.rot), "opengl", "world"))

wrist_camera_offset = MultiMeshRayCasterCfg.OffsetCfg(
    pos=(0.0182505, -0.00408447, -0.0689107),
    rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
)
wrist_camera_offset = wrist_camera_offset.replace(rot=convert_camera_frame_orientation_convention(torch.tensor(wrist_camera_offset.rot), "opengl", "world"))

@configclass
class DepthObjectSceneCfg(RlStateSceneCfg):

    # replace all 3 cameras with depth readings

    front_camera: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_front_camera",
        update_period=0,
        offset=front_camera_offset,
        mesh_prim_paths=[
            "{ENV_REGEX_NS}/InsertiveObject",
            "{ENV_REGEX_NS}/ReceptiveObject",
            "{ENV_REGEX_NS}/Table",
            "{ENV_REGEX_NS}/Robot"
        ],
        ray_alignment="world",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=13.20,
            width=256,
            height=256,
        ),
        debug_vis=False
    )

    side_camera: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_side_camera",
        update_period=0,
        offset=side_camera_offset,
        mesh_prim_paths=[
            "{ENV_REGEX_NS}/InsertiveObject",
            "{ENV_REGEX_NS}/ReceptiveObject",
            "{ENV_REGEX_NS}/Table",
            "{ENV_REGEX_NS}/Robot"
        ],
        ray_alignment="world",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=20.10,
            width=256,
            height=256,
        ),
        debug_vis=False
    )

    wrist_camera: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/depth_wrist_camera",
        update_period=0,
        offset=wrist_camera_offset,
        mesh_prim_paths=[
            "{ENV_REGEX_NS}/InsertiveObject",
            "{ENV_REGEX_NS}/ReceptiveObject",
            "{ENV_REGEX_NS}/Table",
            "{ENV_REGEX_NS}/Robot"
        ],
        ray_alignment="world",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.55,
            width=128,
            height=128,
        ),
        debug_vis=True
    )

@configclass
class DepthObjectObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

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

    @configclass
    class DepthObservationsCfg(ObsGroup):
        
        front_camera_depth = ObsTerm(
            func=task_mdp.ray_hit_distances,
            params={"sensor_cfg": SceneEntityCfg("front_camera")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 10.0),
        )

        side_camera_depth = ObsTerm(
            func=task_mdp.ray_hit_distances,
            params={"sensor_cfg": SceneEntityCfg("side_camera")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 10.0),
        )

        wrist_camera_depth = ObsTerm(
            func=task_mdp.ray_hit_distances,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(0.0, 10.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 4
    
    policy: PolicyCfg = PolicyCfg()
    depth: DepthObservationsCfg = DepthObservationsCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class Ur5eRobotiq2f85DepthTrainingCfg(Ur5eRobotiq2f85RlStateCfg):
    scene: DepthObjectSceneCfg = DepthObjectSceneCfg(num_envs=32, env_spacing=1.5)
    observations: DepthObjectObservationsCfg = DepthObjectObservationsCfg()

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()