from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_train.tasks.manager_based.beyond_mimic.mdp as mdp

VELOCITY_RANGE = {
    "x": (-0.8, 0.8),
    "y": (-0.6, 0.6),
    "z": (-0.4, 0.4),
    "roll": (-0.5, 0.5),
    "pitch": (-0.9, 0.9),
    "yaw": (-0.8, 0.8),
}


@configclass
class SoccerSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    robot: ArticulationCfg = MISSING

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=50.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=5.0,
                linear_damping=0.01,
                angular_damping=4.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.1,
                restitution=0.73,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.11)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        ball_pos_b = ObsTerm(
            func=mdp.noisy_ball_pos_b,
            params={
                "ball_asset_name": "ball",
                "robot_asset_name": "robot",
                "anchor_body_name": "Trunk",
                "base_sigma": 0.01,
                "dist_scale": 0.02,
                "speed_scale": 0.05,
            },
        )
        goal_pos_b = ObsTerm(
            func=mdp.noisy_goal_pos_b,
            params={
                "robot_asset_name": "robot",
                "anchor_body_name": "Trunk",
                "base_sigma": 0.01,
                "dist_scale": 0.015,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        ball_pos_b = ObsTerm(
            func=mdp.ball_pos_b,
            params={"ball_asset_name": "ball", "robot_asset_name": "robot", "anchor_body_name": "Trunk"},
        )
        goal_pos_b = ObsTerm(func=mdp.goal_pos_b, params={"robot_asset_name": "robot", "anchor_body_name": "Trunk"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.72, 1.03),
            "dynamic_friction_range": (0.04, 0.20),
            "restitution_range": (0.66, 0.80),
            "num_buckets": 32,
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-0.5, 0.5)}},
    )

    reset_soccer = EventTerm(
        func=mdp.reset_ball_and_goal,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "Trunk",
            "ball_height": 0.11,
            "nominal_ball_offset_xy": (1.0, 0.0),
            "spawn_angle_range": (-0.35, 0.35),
            "spawn_radius_range": (-0.15, 0.25),
            "goal_center_local": (5.0, 0.0, 0.0),
            "goal_rect_size_xy": (1.0, 0.5),
            "rolling_ball_probability": 0.2,
            "rolling_speed_range": (0.1, 0.3),
        },
    )


@configclass
class RewardsCfg:
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.35},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.45},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.6,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.6,
        params={"command_name": "motion", "std": 3.14},
    )
    motion_foot_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["left_foot_link", "right_foot_link"]},
    )

    ball_distance = RewTerm(
        func=mdp.ball_distance_reward,
        weight=2.0,
        params={
            "ball_asset_name": "ball",
            "robot_asset_name": "robot",
            "anchor_body_name": "Trunk",
            "std": 0.7,
            "freeze_after_valid_contact": True,
            "left_foot_name": "left_foot_link",
            "right_foot_name": "right_foot_link",
            "contact_distance": 0.12,
            "min_ball_speed_for_valid_contact": 0.5,
        },
    )
    face_ball = RewTerm(
        func=mdp.face_ball_reward,
        weight=0.8,
        params={"ball_asset_name": "ball", "robot_asset_name": "robot", "anchor_body_name": "Trunk"},
    )
    correct_foot_strike = RewTerm(
        func=mdp.correct_foot_strike_reward,
        weight=3.0,
        params={
            "ball_asset_name": "ball",
            "robot_asset_name": "robot",
            "left_foot_name": "left_foot_link",
            "right_foot_name": "right_foot_link",
            "ball_contact_distance": 0.12,
            "min_ball_speed_for_valid_contact": 0.5,
            "outcome_window_steps": 20,
        },
    )
    shot_outcome = RewTerm(
        func=mdp.shot_outcome_reward,
        weight=2.0,
        params={
            "ball_asset_name": "ball",
            "min_ball_speed_for_outcome": 0.5,
            "align_weight": 1.0,
            "planar_speed_weight": 0.4,
            "vertical_speed_penalty": 0.2,
        },
    )

    feet_separation = RewTerm(
        func=mdp.feet_separation_reward,
        weight=0.4,
        params={
            "robot_asset_name": "robot",
            "left_foot_name": "left_foot_link",
            "right_foot_name": "right_foot_link",
            "target_min_distance": 0.18,
            "max_clip": 0.1,
        },
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.5)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.30},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.9},
    )
    ball_far = DoneTerm(
        func=mdp.ball_too_far,
        params={"ball_asset_name": "ball", "robot_asset_name": "robot", "anchor_body_name": "Trunk", "threshold": 4.0},
    )
    missed_strike = DoneTerm(
        func=mdp.missed_strike_timeout,
        params={"fraction_of_episode": 0.6},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class TrackingSoccerEnvCfg(ManagerBasedRLEnvCfg):
    scene: SoccerSceneCfg = SoccerSceneCfg(num_envs=4096, env_spacing=3.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.viewer.origin_type = "world"
        self.viewer.eye = (4.0, -6.0, 2.2)
        self.viewer.lookat = (0.0, 0.0, 0.9)
