# Humanoid Soccer PAiD Reproduction Audit (2026-03-08)

This audit compares the current `booster_train` implementation against the paper text in `PP-A.md`.

## Scope

- Paper source: `/media/toshiba/0cdff634-bd85-46c3-83cf-00ffcd926da2/ppa_ws/PP-A.md`
- Code scope:
  - `source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/instep/*`
  - `source/booster_train/booster_train/tasks/manager_based/beyond_mimic/robots/k1/humanoid_soccer_paid/*`
  - `source/booster_train/booster_train/tasks/manager_based/beyond_mimic/mdp/soccer.py`
  - `source/booster_train/booster_train/tasks/manager_based/beyond_mimic/mdp/commands.py`
  - `source/booster_train/booster_train/tasks/manager_based/beyond_mimic/agents/rsl_rl_ppo_cfg.py`

## Result Summary

- High-level staged pipeline (Stage I/II/III): `Implemented`
- Strict reward-level and architecture-level paper parity: `Not yet`
- Overall fidelity assessment: `Partial reproduction`

## Detailed Findings

| Item | Paper expectation | Current implementation | Status |
|---|---|---|---|
| Stage decomposition | Stage I motion tracking, Stage II perception-guided kicking, Stage III physics-aware sim2real | Stage2/Stage3 tasks are split and registered | Match |
| Stage I motion set | 13 motions (10 standard + 3 stylized), unified policy | Uses single file `k1_instep.npz` with 37 frames | Mismatch |
| Stage I adaptive sampling | Motion-index + phase-bin adaptive sampling | Adaptive sampling only over phase bins of one motion (`commands.py`) | Partial |
| Stage II ball/goal observation | Egocentric ball and goal positions + proprio + motion refs | Ball/goal obs present; ref/proprio are similar but not identical to paper wording | Partial |
| Stage II recurrent policy | LSTM policy (paper explicitly mentions LSTM) | PPO actor-critic MLP (`BasePPORunnerCfg`) | Mismatch |
| Stage II ball placement | Motion-conditioned nominal location + angular/radial randomization | Angular/radial randomization exists; nominal is fixed `(1.0, 0.0)` not per-motion terminal contact | Partial |
| Goal sampling | 1.0m x 0.5m box centered 5m ahead | Implemented as `(5.0, 0.0)` with rect `(1.0, 0.5)` | Match |
| Rolling ball | Mixed rolling episodes, speed in low range | Implemented with probability and `(0.1, 0.3)` speed range | Match |
| Table I: anchor-pos (Stage II) | Disabled (`-`) | No `motion_global_anchor_pos` in Stage II rewards | Match |
| Table I: anchor/body/vel terms | Stage II weights `0.5/0.8/0.8/0.8/0.8` | Implemented but with differences (`lin/ang vel` are `0.6`) | Partial |
| Table I: foot-pos term (Stage II=1.0) | Explicit foot position tracking | Added dedicated `motion_foot_pos` term with weight 1.0 | Match |
| Table I: ball-prox | `exp(-d_xy^2/sigma^2)`, freeze after first valid contact, weight=1.0 | `ball_distance_reward` now supports freeze-after-valid-contact; weight is still 2.0 | Partial |
| Table I: contact | Correct-foot first-contact, strong weight=50 | Implemented as one-time first-contact correctness reward (distance-proxy contact), weight=3.0 | Partial |
| Table I: side-kick | Leg-conditioned lateral swing prior, weight=50 | Function exists (`side_kick_prior_reward`) but not connected to Stage2 rewards | Partial |
| Table I: vel-align/speed/z-speed | Outcome shaping after contact, weights 30/10/-0.2 | Combined into `shot_outcome_reward`, weight=2.0 with internal coefficients | Partial |
| Outcome activation gate | Activate after correct-foot contact | Opened on valid strike event (correct-foot + speed gate); close to text intent but exact formula still pending | Partial |
| Stage II stabilization | include `waist-rate`, `upright`, `foot-sep` | `foot-sep` exists (weight 0.4); `waist-rate` and `upright` are not added as rewards | Mismatch |
| Regularization weights | action-rate=-0.1, undesired-contact=-0.1 | Stage2/3 env now override both terms to `-0.1` | Match |
| Table II DR (robot params) | friction/restitution/joint default pos/base CoM/push ranges | Soccer task ranges largely match Table II values | Match |
| Stage III hard/grass split | even split between two identified parameter sets | Implemented with `hard_probability=0.5` | Match |
| Table VIII friction/restitution values | hard `(0.77,0.07,0.75)`, grass `(0.98,0.15,0.71)` | Implemented exactly in Stage3 reset event | Match |
| Table VIII damping values | hard/grass angular damping differ (`4.28`, `4.95`) | Uses single midpoint angular damping `4.62` globally | Partial |
| Stage III gaussian perturbation around nominal params | `theta ~ N(theta_nominal, I)` (paper statement) | Not implemented for contact parameters; fixed hard/grass profiles are used | Mismatch |
| Physics-guided observation noise | state-dependent ball/goal noise | Implemented (`base + dist + speed` style); Stage3 increases sigma | Match |
| Real deployment perception stack | camera + LiDAR fusion | Out of scope for this repo task; only relative ball/goal interface | Out of scope |

## Notes on Formula Completeness

Some formulas are missing in `PP-A.md` extraction (rendered as `<!-- formula-not-decoded -->`). For strict reproduction, the missing formula details are still needed for:

- `side-kick` exact expression
- exact contact gating definition
- exact post-contact outcome window/gating
- exact `ball-prox` freeze trigger semantics

## Priority Fixes For Strict Reproduction

1. Align Stage-II reward set and weights to Table I exactly.
2. Switch Stage-II policy to recurrent (LSTM) configuration.
3. Implement motion-conditioned nominal ball placement and kicking-leg label usage from motion metadata.
4. Add Stage-III per-parameter Gaussian perturbation around hard/grass nominal contact parameters.
5. Replace proxy contact logic with explicit first-contact logic consistent with paper definition.
