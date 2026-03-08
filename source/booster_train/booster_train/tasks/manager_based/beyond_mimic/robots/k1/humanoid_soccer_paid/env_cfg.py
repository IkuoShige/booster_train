import os

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import booster_train.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE

from .tracking_env_cfg import TrackingSoccerEnvCfg

MOTION_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",
    "..",
    "..",
    "..",
    "..",
    "k1_instep.npz",
)


@configclass
class FlatEnvCfg(TrackingSoccerEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE

        self.commands.motion.motion_file = MOTION_FILE
        self.commands.motion.tail_len = 5
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = [
            "Trunk",
            "Head_2",
            "Left_Hip_Roll",
            "Left_Shank",
            "left_foot_link",
            "Right_Hip_Roll",
            "Right_Shank",
            "right_foot_link",
            "Left_Arm_2",
            "Left_Arm_3",
            "left_hand_link",
            "Right_Arm_2",
            "Right_Arm_3",
            "right_hand_link",
        ]


@configclass
class Stage2EnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Easier Stage-II curriculum: static ball and narrower spawn arc first.
        self.events.reset_soccer.params["rolling_ball_probability"] = 0.0
        self.events.reset_soccer.params["rolling_speed_range"] = (0.1, 0.3)
        self.events.reset_soccer.params["spawn_angle_range"] = (-0.2, 0.2)
        self.events.reset_soccer.params["spawn_radius_range"] = (-0.05, 0.15)
        # Stage-II curriculum: disable pushes while the policy learns ball approach/strike.
        self.events.push_robot = None
        # Relax anchor height termination for adaptation from pure tracking to ball-conditioned behavior.
        self.terminations.anchor_pos.params["threshold"] = 0.45
        # Early-stage stabilization: loosen orientation/missed-strike terminations.
        self.terminations.anchor_ori.params["threshold"] = 1.2
        self.terminations.missed_strike.params["fraction_of_episode"] = 0.95
        # Keep Stage-II regularization close to paper scale to avoid penalty domination.
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.undesired_contacts.weight = -0.1
        # Relax valid-contact gate so strike events become observable earlier.
        self.rewards.ball_distance.params["contact_distance"] = 0.18
        self.rewards.ball_distance.params["min_ball_speed_for_valid_contact"] = 0.3
        self.rewards.correct_foot_strike.params["ball_contact_distance"] = 0.18
        self.rewards.correct_foot_strike.params["min_ball_speed_for_valid_contact"] = 0.3


@configclass
class Stage3EnvCfg(Stage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Increase state-dependent observation randomization for sim-to-real robustness.
        self.observations.policy.ball_pos_b.params["base_sigma"] = 0.015
        self.observations.policy.ball_pos_b.params["dist_scale"] = 0.03
        self.observations.policy.ball_pos_b.params["speed_scale"] = 0.08
        self.observations.policy.goal_pos_b.params["base_sigma"] = 0.015
        self.observations.policy.goal_pos_b.params["dist_scale"] = 0.02

        self.events.reset_soccer.params["rolling_ball_probability"] = 0.35
        self.events.reset_soccer.params["spawn_angle_range"] = (-0.35, 0.35)
        self.events.reset_soccer.params["spawn_radius_range"] = (-0.15, 0.25)
        # Use paper Table-VIII contact profiles (hard/grass) per reset.
        self.events.ball_material = None
        self.events.set_ball_profiles = EventTerm(
            func=mdp.set_ball_contact_profiles_from_paper,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "hard_probability": 0.5,
                "hard_profile": (0.77, 0.07, 0.75),
                "grass_profile": (0.98, 0.15, 0.71),
            },
        )
        # Damping values from the paper are (0.01 linear, 4.28/4.95 angular).
        # IsaacLab's material randomizer does not expose per-env damping updates, so we keep
        # linear damping fixed at 0.01 and set angular damping near the midpoint.
        self.scene.ball.spawn.rigid_props.linear_damping = 0.01
        self.scene.ball.spawn.rigid_props.angular_damping = 4.62
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-0.5, 0.5)}},
        )
        # Tighten termination again for fine-tuning.
        self.terminations.anchor_ori.params["threshold"] = 0.9
        self.terminations.missed_strike.params["fraction_of_episode"] = 0.6
        self.terminations.anchor_pos.params["threshold"] = 0.35
        # Restore stricter strike validity for final policy quality.
        self.rewards.ball_distance.params["contact_distance"] = 0.12
        self.rewards.ball_distance.params["min_ball_speed_for_valid_contact"] = 0.5
        self.rewards.correct_foot_strike.params["ball_contact_distance"] = 0.12
        self.rewards.correct_foot_strike.params["min_ball_speed_for_valid_contact"] = 0.5


@configclass
class PlayEnvCfg(Stage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.play = True
        self.events.push_robot = None
        self.scene.num_envs = 64


@configclass
class EvalStaticEnvCfg(Stage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = None
        self.events.reset_soccer.params["rolling_ball_probability"] = 0.0
        self.scene.num_envs = 256


@configclass
class EvalRollingEnvCfg(Stage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = None
        self.events.reset_soccer.params["rolling_ball_probability"] = 1.0
        self.events.reset_soccer.params["rolling_speed_range"] = (0.1, 0.3)
        self.scene.num_envs = 256
