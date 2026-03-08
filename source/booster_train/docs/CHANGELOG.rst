Changelog
---------

Unreleased
~~~~~~~~~~

Added
^^^^^

* Added PAiD humanoid-soccer tasks for Booster K1 (Stage2/Stage3/Play/Eval).
* Added soccer-specific MDP utilities (ball/goal observations, soccer rewards, reset logic, and terminations).
* Added direct resume checkpoint argument to ``scripts/rsl_rl/train.py``.
* Added system-identification helper script ``scripts/soccer_sysid_fit.py``.
* Added reproduction/research notes under ``source/booster_train/docs``.
* Added ``side_kick_prior_reward`` to soccer MDP utilities (kept unbound from Stage2 by default).
* Added reproduction-fidelity audit document ``HUMANOID_SOCCER_REPRO_AUDIT.md``.
* Updated Stage-II soccer reward internals with provisional paper-aligned semantics:
  first-contact correctness reward and ball-prox freeze-after-valid-contact behavior.
* Added Stage-II ``motion_foot_pos`` reward (weight 1.0) and tuned early-stage stabilization:
  reduced action/contact penalties, relaxed early termination, and lower PPO init action-noise.
* Added a staged-easier Stage-II curriculum for strike emergence:
  static/narrow ball spawn, later missed-strike timeout, relaxed valid-contact gates,
  and reduced entropy/noise in PPO runner config.

0.1.0 (2025-11-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
