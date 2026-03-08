# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


gym.register(
    id="Booster-K1-HumanoidSoccer-PAiD-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Stage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Stage2PPORunnerCfg",
    },
)


gym.register(
    id="Booster-K1-HumanoidSoccer-PAiD-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Stage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Stage3PPORunnerCfg",
    },
)


gym.register(
    id="Booster-K1-HumanoidSoccer-PAiD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Stage3PPORunnerCfg",
    },
)


gym.register(
    id="Booster-K1-HumanoidSoccer-PAiD-Eval-Static-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:EvalStaticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Stage3PPORunnerCfg",
    },
)


gym.register(
    id="Booster-K1-HumanoidSoccer-PAiD-Eval-Rolling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:EvalRollingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:Stage3PPORunnerCfg",
    },
)
