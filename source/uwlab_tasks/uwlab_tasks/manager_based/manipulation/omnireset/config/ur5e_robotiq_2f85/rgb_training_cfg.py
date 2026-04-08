from isaaclab.sim import configclass

from pathlib import Path

from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCTrainCfg, Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg
from .data_collection_rgb_cfg import RGBObservationsCfg, DataCollectionRGBObjectSceneCfg, BaseRGBEventCfg, Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg
from .rl_state_cfg import ObservationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp

# smaller output size to make training more tractable
@configclass
class RGBTrainingObservationsCfg(RGBObservationsCfg.RGBPolicyCfg):
    front_rgb = ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg("front_camera"),
            "data_type": "rgb",
            "process_image": True,
            "output_size": (128, 128),
        },
    )

    side_rgb = ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg("side_camera"),
            "data_type": "rgb",
            "process_image": True,
            "output_size": (128, 128),
        },
    )

    wrist_rgb = ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "data_type": "rgb",
            "process_image": True,
            "output_size": (128, 128),
        },
    )

@configclass
class TrainingObservationsCfg:
    policy: RGBTrainingObservationsCfg = RGBTrainingObservationsCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()
    expert: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

@configclass
class RGBTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    corrupted_camera = DoneTerm(
        func=task_mdp.corrupted_camera_detected,
        params={"camera_names": ["front_camera", "side_camera", "wrist_camera"], "std_threshold": 10.0},
    )

@configclass
class RGBTrainingEventsCfg(BaseRGBEventCfg):
    """RGB training events: inherits fixed sysid + OSC gains from BaseRGBEventCfg, limits dynamics + table/curtain randomization for more tractable training."""

    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.7, 0.7),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.4, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.4, 0.4),
            "dynamic_friction_range": (0.3, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.5, 0.5),
            "dynamic_friction_range": (0.4, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # we assume insertive object is somewhere between 20g and 200g
            "mass_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_distribution_params": (1.25, 1.25),
            "damping_distribution_params": (1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_table_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "event_name": "randomize_table_event",
            "mesh_names": ["visuals/vention_mat"],
            "texture_prob": 0.0,
            "texture_config_path": str(Path(__file__).parent / "resources" / "training_texture_paths.yaml"),
            "diffuse_tint_range": ((0.95, 0.95, 0.95), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.95, 1.0), "g": (0.95, 1.0), "b": (0.95, 1.0)},
            "texture_scale_range": (2.5, 2.5),
            "roughness_range": (0.5, 0.5),
            "metallic_range": (0.2, 0.3),
            "specular_range": (0.9, 1.0),
        },
    )

    randomize_curtain_left_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_left"),
            "event_name": "randomize_curtain_left_event",
            "mesh_names": [],
            "texture_prob": 0.0,
            "texture_config_path": str(Path(__file__).parent / "resources" / "training_texture_paths.yaml"),
            "diffuse_tint_range": ((0.95, 0.95, 0.95), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.95, 1.0), "g": (0.95, 1.0), "b": (0.95, 1.0)},
            "texture_scale_range": (2.5, 2.5),
            "roughness_range": (0.5, 0.5),
            "metallic_range": (0.2, 0.3),
            "specular_range": (0.9, 1.0),
        },
    )

    randomize_curtain_back_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_back"),
            "event_name": "randomize_curtain_back_event",
            "mesh_names": [],
            "texture_prob": 0.0,
            "texture_config_path": str(Path(__file__).parent / "resources" / "training_texture_paths.yaml"),
            "diffuse_tint_range": ((0.95, 0.95, 0.95), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.95, 1.0), "g": (0.95, 1.0), "b": (0.95, 1.0)},
            "texture_scale_range": (2.5, 2.5),
            "roughness_range": (0.5, 0.5),
            "metallic_range": (0.2, 0.3),
            "specular_range": (0.9, 1.0),
        },
    )

    randomize_curtain_right_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_right"),
            "event_name": "randomize_curtain_right_event",
            "mesh_names": [],
            "texture_prob": 0.0,
            "texture_config_path": str(Path(__file__).parent / "resources" / "training_texture_paths.yaml"),
            "diffuse_tint_range": ((0.95, 0.95, 0.95), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.95, 1.0), "g": (0.95, 1.0), "b": (0.95, 1.0)},
            "texture_scale_range": (2.5, 2.5),
            "roughness_range": (0.5, 0.5),
            "metallic_range": (0.2, 0.3),
            "specular_range": (0.9, 1.0),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class RGBUr5eRobotiq2f85RelCartesianOSCTrainCfg(Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg):
    observations : TrainingObservationsCfg = TrainingObservationsCfg()
    terminations: RGBTerminationsCfg = RGBTerminationsCfg()
    events: RGBTrainingEventsCfg = RGBTrainingEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0

        self.observations.expert.concatenate_terms = True