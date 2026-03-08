from isaaclab.utils import configclass

from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class Stage2PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 60000
    experiment_name = "k1_humanoid_soccer_paid_stage2"
    save_interval = 500

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.12
        self.algorithm.entropy_coef = 0.002


@configclass
class Stage3PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 30000
    experiment_name = "k1_humanoid_soccer_paid_stage3"
    save_interval = 500

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.12
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 5.0e-4
