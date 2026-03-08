# Humanoid Soccer (PAiD) Notes for Booster K1

This note documents the paper-aligned design decisions used in `booster_train`.

## Paper Summary (from `PP-A.md`)

- Framework: **PAiD (Perception-Action integrated Decision-making)**.
- Stage I: motion-skill acquisition via motion tracking.
- Stage II: perception-guided positional generalization with lightweight task rewards.
- Stage III: physics-aware sim-to-real transfer via ball parameter identification + observation randomization.

## Key Equivalents in This Repository

- Stage I baseline: existing K1 motion-tracking checkpoint (`k1_instep/model_4000.pt`) is treated as the starting point.
- Stage II task: `Booster-K1-HumanoidSoccer-PAiD-Stage2-v0`.
- Stage III task: `Booster-K1-HumanoidSoccer-PAiD-Stage3-v0`.

## Observation Design

- Policy observations:
  - motion command/reference terms.
  - proprioception (base angular velocity, joint positions/velocities, previous action).
  - soccer terms: ball and goal position in robot frame.
- Critic observations include clean (non-noisy) soccer terms.

## Reward Design (Stage II/III)

Implemented reward groups:

- Motion priors:
  - anchor/body orientation/position/velocity tracking terms.
- Soccer shaping:
  - ball proximity reward.
  - facing-ball reward.
  - correct-foot strike gating reward.
  - post-strike outcome reward (direction alignment + planar speed - vertical speed penalty).
- Stabilization:
  - action-rate regularization.
  - joint-limit and undesired-contact penalties.
  - feet separation reward.

## Ball/Goal Randomization

- Goal target is sampled in a 1.0m x 0.5m rectangle centered 5m ahead in local frame.
- Ball is sampled around a nominal local offset with angle/radius perturbations.
- Rolling-ball episodes are mixed during training.

## Stage III Sim-to-Real Hooks

- Ball contact properties are randomized around paper-inspired ranges.
- State-dependent observation noise is injected into ball/goal observations.
- System identification helper script:
  - `scripts/soccer_sysid_fit.py`
  - Inputs: drop trajectory CSV and rolling trajectory CSV.
  - Output: YAML profile (friction/restitution/damping + sigma).

Paper Table-VIII values are directly used in Stage III reset events:

- hard: `(static, dynamic, restitution) = (0.77, 0.07, 0.75)`
- grass: `(static, dynamic, restitution) = (0.98, 0.15, 0.71)`

## Known Gaps vs Full Paper Setup

- Current implementation is validated for a **single motion** pipeline while preserving extension points for multi-motion expansion.
- The original paper uses 13 motions and richer real-world perception stack (camera + LiDAR). In this repository, deployment interface remains relative ball/goal position inputs.

## Formula Clarification Needed For Strict Stage-II Reproduction

- The exact formula of the `side-kick` prior term in Table I:
  - target direction definition in robot frame.
  - whether the term uses foot velocity, position offset, or both.
  - active window/gating condition (always-on, pre-contact only, or contact-proximal only).
  - exact normalization (`exp`, cosine, hinge, or linear form) and scale parameters.
- The exact freeze condition for `ball-prox`:
  - whether freezing occurs after first *valid* contact or any first contact.

## Provisional Spec (2026-03-08)

Based on user-provided provisional interpretation of the paper:

- `side-kick` should be treated as a directional prior on kick-foot planar swing:
  - compute kick-foot planar velocity in pelvis frame,
  - normalize to direction and reward positive lateral cosine,
  - left label maps to `+y`, right label maps to `-y`.
- `side-kick` should be used as pre-contact prior (not a goal-direction term).
- `ball-prox` freeze should be tied to first valid strike-like contact, not any incidental touch.

Current code reflects this as an unbound utility function (`mdp.soccer.side_kick_prior_reward`) and keeps Stage2 reward wiring unchanged until strict formula details are finalized.

Additional provisional implementation now applied:

- `ball-prox` (`ball_distance_reward`) supports freeze-after-valid-contact behavior.
- `contact` (`correct_foot_strike_reward`) is separated as first-contact one-time correctness reward.
- valid strike registration for outcome-window activation is handled independently with speed gating.
