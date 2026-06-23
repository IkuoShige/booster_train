"""Foot contact and joint-level tracking reward terms for motion tracking."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_joint_pos_error_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Exponential reward for matching the motion's joint positions on selected joints.

    Uses joint-level tracking instead of body-position IK. Useful for joints where the
    target robot cannot reproduce the reference body position (e.g. K1 arms with limited DOF).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_indices = asset_cfg.joint_ids
    if joint_indices is None or joint_indices == slice(None):
        joint_indices = list(range(asset.num_joints))

    motion_joint_pos = command.motion.joint_pos[command.time_steps][:, joint_indices]
    robot_joint_pos = asset.data.joint_pos[:, joint_indices]
    error = torch.norm(motion_joint_pos - robot_joint_pos, dim=-1)
    return torch.exp(-(error / std) ** 2)


def motion_foot_contact_mismatch(
    env: "ManagerBasedRLEnv",
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    motion_foot_z_threshold: float = 0.02,
    force_clip: float = 50.0,
) -> torch.Tensor:
    """Continuous penalty: when motion says foot is in air, penalize by contact force magnitude.

    Returns: (num_envs,) sum of clipped contact forces (normalized) over feet that are
    wrongly grounded according to the reference motion.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    sensor = env.scene.sensors[sensor_cfg.name]

    motion_foot_indices = torch.tensor(
        [command.cfg.body_names.index(name) for name in foot_body_names],
        device=env.device,
        dtype=torch.long,
    )
    contact_body_indices, _ = sensor.find_bodies(foot_body_names)
    contact_body_indices = torch.tensor(contact_body_indices, device=env.device, dtype=torch.long)

    motion_foot_pos = command.motion.body_pos_w[command.time_steps]
    motion_foot_z = motion_foot_pos[:, motion_foot_indices, 2]
    motion_foot_in_air = (motion_foot_z > motion_foot_z_threshold).float()

    contact_forces = sensor.data.net_forces_w[:, contact_body_indices, :]
    foot_force = torch.norm(contact_forces, dim=-1).clamp(max=force_clip) / force_clip

    return (foot_force * motion_foot_in_air).sum(dim=-1)
