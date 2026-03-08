from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _get_assets(
    env: ManagerBasedEnv,
    robot_asset_name: str,
    ball_asset_name: str,
) -> tuple[Articulation, RigidObject]:
    robot = env.scene[robot_asset_name]
    ball = env.scene[ball_asset_name]
    return robot, ball


def _anchor_index(robot: Articulation, anchor_body_name: str) -> int:
    return robot.body_names.index(anchor_body_name)


def _ensure_soccer_buffers(env: ManagerBasedEnv):
    device = env.device
    if not hasattr(env, "_soccer_goal_pos_w"):
        env._soccer_goal_pos_w = torch.zeros(env.num_envs, 3, device=device)
    if not hasattr(env, "_soccer_ball_spawn_pos_w"):
        env._soccer_ball_spawn_pos_w = torch.zeros(env.num_envs, 3, device=device)
    if not hasattr(env, "_soccer_target_left_foot"):
        env._soccer_target_left_foot = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_soccer_strike_registered"):
        env._soccer_strike_registered = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_soccer_outcome_window"):
        env._soccer_outcome_window = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    if not hasattr(env, "_soccer_surface_is_hard"):
        env._soccer_surface_is_hard = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_soccer_ball_prox_frozen"):
        env._soccer_ball_prox_frozen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_soccer_ball_prox_frozen_value"):
        env._soccer_ball_prox_frozen_value = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    if not hasattr(env, "_soccer_first_contact_consumed"):
        env._soccer_first_contact_consumed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)


def set_ball_contact_profiles_from_paper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    hard_probability: float = 0.5,
    hard_profile: tuple[float, float, float] = (0.77, 0.07, 0.75),
    grass_profile: tuple[float, float, float] = (0.98, 0.15, 0.71),
):
    """Assign paper Table-VIII ball contact profiles per-environment.

    The profile is applied to PhysX material parameters (static/dynamic friction, restitution).
    """
    _ensure_soccer_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    env_ids_cpu = env_ids.cpu()
    hard_mask = torch.rand(len(env_ids), device=env.device) < hard_probability
    env._soccer_surface_is_hard[env_ids] = hard_mask

    materials = ball.root_physx_view.get_material_properties()
    total_num_shapes = ball.root_physx_view.max_shapes

    hard_tensor = torch.tensor(hard_profile, device="cpu").view(1, 1, 3)
    grass_tensor = torch.tensor(grass_profile, device="cpu").view(1, 1, 3)
    profile_tensor = torch.where(
        hard_mask.cpu().view(-1, 1, 1),
        hard_tensor.expand(len(env_ids), total_num_shapes, 3),
        grass_tensor.expand(len(env_ids), total_num_shapes, 3),
    )
    materials[env_ids_cpu] = profile_tensor
    ball.root_physx_view.set_material_properties(materials, env_ids_cpu)


def _anchor_pose(
    robot: Articulation,
    anchor_body_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_idx = _anchor_index(robot, anchor_body_name)
    anchor_pos_w = robot.data.body_pos_w[:, anchor_idx]
    anchor_quat_w = robot.data.body_quat_w[:, anchor_idx]
    anchor_lin_vel_w = robot.data.body_lin_vel_w[:, anchor_idx]
    return anchor_pos_w, anchor_quat_w, anchor_lin_vel_w


def _foot_ball_contact_state(
    robot: Articulation,
    ball: RigidObject,
    left_foot_name: str,
    right_foot_name: str,
    contact_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_idx = robot.body_names.index(left_foot_name)
    right_idx = robot.body_names.index(right_foot_name)

    left_dist = torch.norm(ball.data.root_pos_w - robot.data.body_pos_w[:, left_idx], dim=-1)
    right_dist = torch.norm(ball.data.root_pos_w - robot.data.body_pos_w[:, right_idx], dim=-1)
    left_contact = left_dist < contact_distance
    right_contact = right_dist < contact_distance
    any_contact = left_contact | right_contact

    # Resolve first-contact foot identity robustly if both feet are close.
    choose_left = torch.where(
        left_contact & (~right_contact),
        torch.ones_like(any_contact),
        torch.where((~left_contact) & right_contact, torch.zeros_like(any_contact), left_dist <= right_dist),
    )
    return left_contact, right_contact, any_contact, choose_left


def ball_pos_b(
    env: ManagerBasedEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
) -> torch.Tensor:
    robot, ball = _get_assets(env, robot_asset_name, ball_asset_name)
    anchor_pos_w, anchor_quat_w, _ = _anchor_pose(robot, anchor_body_name)
    rel_pos_b = math_utils.quat_apply_inverse(anchor_quat_w, ball.data.root_pos_w - anchor_pos_w)
    return rel_pos_b


def ball_vel_b(
    env: ManagerBasedEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
) -> torch.Tensor:
    robot, ball = _get_assets(env, robot_asset_name, ball_asset_name)
    _, anchor_quat_w, anchor_lin_vel_w = _anchor_pose(robot, anchor_body_name)
    rel_vel_b = math_utils.quat_apply_inverse(anchor_quat_w, ball.data.root_lin_vel_w - anchor_lin_vel_w)
    return rel_vel_b


def goal_pos_b(
    env: ManagerBasedEnv,
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
) -> torch.Tensor:
    _ensure_soccer_buffers(env)
    robot = env.scene[robot_asset_name]
    anchor_pos_w, anchor_quat_w, _ = _anchor_pose(robot, anchor_body_name)
    rel_pos_b = math_utils.quat_apply_inverse(anchor_quat_w, env._soccer_goal_pos_w - anchor_pos_w)
    return rel_pos_b


def noisy_ball_pos_b(
    env: ManagerBasedEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
    base_sigma: float = 0.01,
    dist_scale: float = 0.02,
    speed_scale: float = 0.05,
) -> torch.Tensor:
    pos_b = ball_pos_b(
        env,
        ball_asset_name=ball_asset_name,
        robot_asset_name=robot_asset_name,
        anchor_body_name=anchor_body_name,
    )
    vel_b = ball_vel_b(
        env,
        ball_asset_name=ball_asset_name,
        robot_asset_name=robot_asset_name,
        anchor_body_name=anchor_body_name,
    )
    sigma = base_sigma + dist_scale * torch.norm(pos_b, dim=-1) + speed_scale * torch.norm(vel_b, dim=-1)
    return pos_b + sigma.unsqueeze(-1) * torch.randn_like(pos_b)


def noisy_goal_pos_b(
    env: ManagerBasedEnv,
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
    base_sigma: float = 0.01,
    dist_scale: float = 0.02,
) -> torch.Tensor:
    goal_b = goal_pos_b(env, robot_asset_name=robot_asset_name, anchor_body_name=anchor_body_name)
    sigma = base_sigma + dist_scale * torch.norm(goal_b, dim=-1)
    return goal_b + sigma.unsqueeze(-1) * torch.randn_like(goal_b)


def reset_ball_and_goal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    anchor_body_name: str = "Trunk",
    ball_height: float = 0.11,
    nominal_ball_offset_xy: tuple[float, float] = (1.0, 0.0),
    spawn_angle_range: tuple[float, float] = (-0.35, 0.35),
    spawn_radius_range: tuple[float, float] = (-0.15, 0.25),
    goal_center_local: tuple[float, float, float] = (5.0, 0.0, 0.0),
    goal_rect_size_xy: tuple[float, float] = (1.0, 0.5),
    rolling_ball_probability: float = 0.2,
    rolling_speed_range: tuple[float, float] = (0.1, 0.3),
):
    _ensure_soccer_buffers(env)

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    anchor_pos_w, anchor_quat_w, _ = _anchor_pose(robot, anchor_body_name)
    yaw_quat = math_utils.yaw_quat(anchor_quat_w[env_ids])

    nominal_radius = math.sqrt(nominal_ball_offset_xy[0] ** 2 + nominal_ball_offset_xy[1] ** 2)
    nominal_angle = math.atan2(nominal_ball_offset_xy[1], nominal_ball_offset_xy[0])

    angle_noise = math_utils.sample_uniform(
        spawn_angle_range[0], spawn_angle_range[1], (len(env_ids),), device=env.device
    )
    radius_noise = math_utils.sample_uniform(
        spawn_radius_range[0], spawn_radius_range[1], (len(env_ids),), device=env.device
    )

    angles = nominal_angle + angle_noise
    radii = torch.clamp(nominal_radius + radius_noise, min=0.25)

    local_ball = torch.zeros(len(env_ids), 3, device=env.device)
    local_ball[:, 0] = radii * torch.cos(angles)
    local_ball[:, 1] = radii * torch.sin(angles)
    # z is applied in world frame to keep the ball on the ground plane.
    local_ball[:, 2] = 0.0

    world_ball = anchor_pos_w[env_ids] + math_utils.quat_apply(yaw_quat, local_ball)
    world_ball[:, 2] = env.scene.env_origins[env_ids, 2] + ball_height

    goal_center = torch.tensor(goal_center_local, device=env.device).repeat(len(env_ids), 1)
    goal_center[:, 0] += math_utils.sample_uniform(
        -0.5 * goal_rect_size_xy[0], 0.5 * goal_rect_size_xy[0], (len(env_ids),), device=env.device
    )
    goal_center[:, 1] += math_utils.sample_uniform(
        -0.5 * goal_rect_size_xy[1], 0.5 * goal_rect_size_xy[1], (len(env_ids),), device=env.device
    )
    world_goal = anchor_pos_w[env_ids] + math_utils.quat_apply(yaw_quat, goal_center)
    world_goal[:, 2] = env.scene.env_origins[env_ids, 2] + goal_center_local[2]

    root_state = ball.data.root_state_w.clone()
    root_state[env_ids, :3] = world_ball
    root_state[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[env_ids, 7:] = 0.0

    rolling_mask = torch.rand(len(env_ids), device=env.device) < rolling_ball_probability
    if torch.any(rolling_mask):
        rolling_ids = env_ids[rolling_mask]
        speeds = math_utils.sample_uniform(
            rolling_speed_range[0], rolling_speed_range[1], (rolling_ids.shape[0],), device=env.device
        )
        rolling_to_robot = anchor_pos_w[rolling_ids, :2] - world_ball[rolling_mask, :2]
        rolling_dir = rolling_to_robot / rolling_to_robot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        root_state[rolling_ids, 7:9] = rolling_dir * speeds.unsqueeze(-1)

    ball.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)

    env._soccer_goal_pos_w[env_ids] = world_goal
    env._soccer_ball_spawn_pos_w[env_ids] = world_ball
    env._soccer_target_left_foot[env_ids] = local_ball[:, 1] >= 0.0
    env._soccer_strike_registered[env_ids] = False
    env._soccer_outcome_window[env_ids] = 0
    env._soccer_ball_prox_frozen[env_ids] = False
    env._soccer_ball_prox_frozen_value[env_ids] = 0.0
    env._soccer_first_contact_consumed[env_ids] = False


def ball_distance_reward(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
    std: float = 0.6,
    freeze_after_valid_contact: bool = True,
    left_foot_name: str = "left_foot_link",
    right_foot_name: str = "right_foot_link",
    contact_distance: float = 0.12,
    min_ball_speed_for_valid_contact: float = 0.5,
) -> torch.Tensor:
    _ensure_soccer_buffers(env)
    rel_pos = ball_pos_b(
        env,
        ball_asset_name=ball_asset_name,
        robot_asset_name=robot_asset_name,
        anchor_body_name=anchor_body_name,
    )
    dist_sq = torch.sum(torch.square(rel_pos[:, :2]), dim=-1)
    reward_now = torch.exp(-dist_sq / (std ** 2))

    if not freeze_after_valid_contact:
        return reward_now

    robot, ball = _get_assets(env, robot_asset_name, ball_asset_name)
    left_contact, right_contact, _, _ = _foot_ball_contact_state(
        robot,
        ball,
        left_foot_name=left_foot_name,
        right_foot_name=right_foot_name,
        contact_distance=contact_distance,
    )
    labeled_contact = torch.where(env._soccer_target_left_foot, left_contact, right_contact)
    ball_speed_xy = ball.data.root_lin_vel_w[:, :2].norm(dim=-1)
    valid_contact = labeled_contact & (ball_speed_xy >= min_ball_speed_for_valid_contact)

    newly_frozen = (~env._soccer_ball_prox_frozen) & valid_contact
    env._soccer_ball_prox_frozen_value = torch.where(
        newly_frozen,
        reward_now,
        env._soccer_ball_prox_frozen_value,
    )
    env._soccer_ball_prox_frozen |= valid_contact
    return torch.where(env._soccer_ball_prox_frozen, env._soccer_ball_prox_frozen_value, reward_now)


def face_ball_reward(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
) -> torch.Tensor:
    rel_pos = ball_pos_b(
        env,
        ball_asset_name=ball_asset_name,
        robot_asset_name=robot_asset_name,
        anchor_body_name=anchor_body_name,
    )
    direction = rel_pos[:, :2] / rel_pos[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)

    robot, _ = _get_assets(env, robot_asset_name, ball_asset_name)
    _, anchor_quat_w, _ = _anchor_pose(robot, anchor_body_name)
    forward_w = math_utils.quat_apply(
        anchor_quat_w,
        torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
    )
    forward_xy = forward_w[:, :2] / forward_w[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return torch.sum(forward_xy * direction, dim=-1).clamp(min=0.0)


def side_kick_prior_reward(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
    left_foot_name: str = "left_foot_link",
    right_foot_name: str = "right_foot_link",
    left_expected_sign: float = 1.0,
    right_expected_sign: float = -1.0,
    pre_contact_only: bool = True,
    prestrike_distance_threshold: float | None = None,
    foot_ball_contact_distance: float | None = None,
    disable_if_ball_prox_frozen: bool = False,
    min_planar_speed: float = 1.0e-3,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Leg-conditioned lateral swing prior for side-kick shaping.

    Provisional paper-aligned definition:
    - evaluate only the kick foot selected by the leg label,
    - use the normalized foot planar velocity in pelvis/anchor frame,
    - reward positive lateral cosine in the expected left/right direction.
    """
    _ensure_soccer_buffers(env)
    robot, ball = _get_assets(env, robot_asset_name, ball_asset_name)

    left_idx = robot.body_names.index(left_foot_name)
    right_idx = robot.body_names.index(right_foot_name)

    _, anchor_quat_w, anchor_lin_vel_w = _anchor_pose(robot, anchor_body_name)
    left_vel_b = math_utils.quat_apply_inverse(anchor_quat_w, robot.data.body_lin_vel_w[:, left_idx] - anchor_lin_vel_w)
    right_vel_b = math_utils.quat_apply_inverse(
        anchor_quat_w, robot.data.body_lin_vel_w[:, right_idx] - anchor_lin_vel_w
    )

    target_left = env._soccer_target_left_foot
    foot_vel_xy = torch.where(target_left.unsqueeze(-1), left_vel_b[:, :2], right_vel_b[:, :2])
    speed_xy = torch.norm(foot_vel_xy, dim=-1)
    unit_xy = foot_vel_xy / speed_xy.unsqueeze(-1).clamp(min=eps)

    expected_sign = torch.where(
        target_left,
        torch.full((env.num_envs,), left_expected_sign, device=env.device),
        torch.full((env.num_envs,), right_expected_sign, device=env.device),
    )
    expected_lateral = torch.stack([torch.zeros_like(expected_sign), expected_sign], dim=-1)
    rew = torch.sum(unit_xy * expected_lateral, dim=-1).clamp(min=0.0)
    rew = torch.where(speed_xy > min_planar_speed, rew, torch.zeros_like(rew))

    if pre_contact_only:
        rew = torch.where(env._soccer_strike_registered, torch.zeros_like(rew), rew)

    if prestrike_distance_threshold is not None:
        rel_pos = ball_pos_b(
            env,
            ball_asset_name=ball_asset_name,
            robot_asset_name=robot_asset_name,
            anchor_body_name=anchor_body_name,
        )
        d_xy = torch.norm(rel_pos[:, :2], dim=-1)
        rew = torch.where(d_xy < prestrike_distance_threshold, rew, torch.zeros_like(rew))

    if foot_ball_contact_distance is not None:
        left_pos_w = robot.data.body_pos_w[:, left_idx]
        right_pos_w = robot.data.body_pos_w[:, right_idx]
        strike_foot_pos_w = torch.where(target_left.unsqueeze(-1), left_pos_w, right_pos_w)
        dist = torch.norm(ball.data.root_pos_w - strike_foot_pos_w, dim=-1)
        rew = torch.where(dist > foot_ball_contact_distance, rew, torch.zeros_like(rew))

    if disable_if_ball_prox_frozen:
        rew = torch.where(env._soccer_ball_prox_frozen, torch.zeros_like(rew), rew)

    return rew


def correct_foot_strike_reward(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    left_foot_name: str = "left_foot_link",
    right_foot_name: str = "right_foot_link",
    ball_contact_distance: float = 0.12,
    min_ball_speed_for_valid_contact: float = 0.5,
    outcome_window_steps: int = 20,
) -> torch.Tensor:
    """First-contact correctness reward + valid-strike registration.

    - Reward: one-time first ball-contact event (correct foot => 1, else 0).
    - Registration: valid strike is tracked separately (correct-foot contact and speed gate),
      and opens the post-contact outcome window.
    """
    _ensure_soccer_buffers(env)

    robot, ball = _get_assets(env, robot_asset_name, ball_asset_name)
    left_contact, right_contact, any_contact, choose_left = _foot_ball_contact_state(
        robot,
        ball,
        left_foot_name=left_foot_name,
        right_foot_name=right_foot_name,
        contact_distance=ball_contact_distance,
    )
    correct_foot = torch.where(env._soccer_target_left_foot, choose_left, ~choose_left)

    first_contact_event = any_contact & (~env._soccer_first_contact_consumed)
    rew = torch.where(
        first_contact_event & correct_foot,
        torch.ones(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )
    env._soccer_first_contact_consumed |= first_contact_event

    labeled_contact = torch.where(env._soccer_target_left_foot, left_contact, right_contact)
    ball_speed_xy = ball.data.root_lin_vel_w[:, :2].norm(dim=-1)
    valid_contact = labeled_contact & (ball_speed_xy >= min_ball_speed_for_valid_contact)
    new_valid_strike = valid_contact & (~env._soccer_strike_registered)
    env._soccer_strike_registered |= valid_contact
    env._soccer_outcome_window = torch.where(
        new_valid_strike,
        torch.full_like(env._soccer_outcome_window, outcome_window_steps),
        env._soccer_outcome_window,
    )

    return rew


def shot_outcome_reward(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    min_ball_speed_for_outcome: float = 0.5,
    align_weight: float = 1.0,
    planar_speed_weight: float = 0.5,
    vertical_speed_penalty: float = 0.25,
) -> torch.Tensor:
    _ensure_soccer_buffers(env)

    ball: RigidObject = env.scene[ball_asset_name]

    active = env._soccer_outcome_window > 0
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    ball_speed_xy = ball_vel_xy.norm(dim=-1)

    desired_xy = env._soccer_goal_pos_w[:, :2] - env._soccer_ball_spawn_pos_w[:, :2]
    desired_xy = desired_xy / desired_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    vel_dir = ball_vel_xy / ball_speed_xy.unsqueeze(-1).clamp(min=1e-6)
    align = torch.sum(desired_xy * vel_dir, dim=-1).clamp(min=0.0)
    vertical_speed = torch.abs(ball.data.root_lin_vel_w[:, 2])

    speed_gate = ball_speed_xy > min_ball_speed_for_outcome
    rew = torch.zeros(env.num_envs, device=env.device)
    rew = torch.where(
        active & speed_gate,
        align_weight * align + planar_speed_weight * ball_speed_xy - vertical_speed_penalty * vertical_speed,
        rew,
    )

    env._soccer_outcome_window = torch.clamp(env._soccer_outcome_window - 1, min=0)
    return rew


def feet_separation_reward(
    env: ManagerBasedRLEnv,
    robot_asset_name: str = "robot",
    left_foot_name: str = "left_foot_link",
    right_foot_name: str = "right_foot_link",
    target_min_distance: float = 0.18,
    max_clip: float = 0.15,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_asset_name]
    left_idx = robot.body_names.index(left_foot_name)
    right_idx = robot.body_names.index(right_foot_name)

    left_xy = robot.data.body_pos_w[:, left_idx, :2]
    right_xy = robot.data.body_pos_w[:, right_idx, :2]
    dist = torch.norm(left_xy - right_xy, dim=-1)
    return torch.clamp(dist - target_min_distance, min=0.0, max=max_clip)


def ball_too_far(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    anchor_body_name: str = "Trunk",
    threshold: float = 4.0,
) -> torch.Tensor:
    rel_pos = ball_pos_b(
        env,
        ball_asset_name=ball_asset_name,
        robot_asset_name=robot_asset_name,
        anchor_body_name=anchor_body_name,
    )
    return torch.norm(rel_pos[:, :2], dim=-1) > threshold


def missed_strike_timeout(
    env: ManagerBasedRLEnv,
    fraction_of_episode: float = 0.6,
) -> torch.Tensor:
    _ensure_soccer_buffers(env)
    max_step = int(env.max_episode_length * fraction_of_episode)
    return (env.episode_length_buf > max_step) & (~env._soccer_strike_registered)
