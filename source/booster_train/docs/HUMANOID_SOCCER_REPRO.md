# Humanoid Soccer PAiD Reproduction (Booster K1)

## Environment Versions

- Isaac Sim: 5.1.0
- IsaacLab: 2.3.0
- Task package: `booster_train`

## Install

```bash
python -m pip install -e source/booster_train
```

If using `Isaac_uv_template`, set IsaacLab source paths before running scripts:

```bash
source /media/toshiba/0cdff634-bd85-46c3-83cf-00ffcd926da2/ppa_ws/Isaac_uv_template/.venv/bin/activate
ISAACLAB_SITE=/media/toshiba/0cdff634-bd85-46c3-83cf-00ffcd926da2/ppa_ws/Isaac_uv_template/.venv/lib/python3.11/site-packages/isaaclab/source
export PYTHONPATH=$ISAACLAB_SITE/isaaclab:$ISAACLAB_SITE/isaaclab_tasks:$ISAACLAB_SITE/isaaclab_rl:$PYTHONPATH
export OMNI_KIT_ACCEPT_EULA=YES
```

## List Added Tasks

```bash
python scripts/list_envs.py | rg "HumanoidSoccer"
```

## Stage II Training (resume from Stage I checkpoint)

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-HumanoidSoccer-PAiD-Stage2-v0 \
  --headless \
  --device cuda:0 \
  --resume_checkpoint /media/toshiba/0cdff634-bd85-46c3-83cf-00ffcd926da2/ppa_ws/booster_train/logs/rsl_rl/k1_instep/2026-03-07_20-28-37/2026-03-07_20-28-37/model_4000.pt
```

W&B で学習中動画を確認する場合:

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-HumanoidSoccer-PAiD-Stage2-v0 \
  --headless \
  --device cuda:0 \
  --logger wandb \
  --log_project_name booster_k1_soccer \
  --video \
  --video_interval 500 \
  --video_length 300 \
  --resume_checkpoint /media/toshiba/0cdff634-bd85-46c3-83cf-00ffcd926da2/ppa_ws/booster_train/logs/rsl_rl/k1_instep/2026-03-07_20-28-37/2026-03-07_20-28-37/model_4000.pt
```

## Stage III Fine-tuning

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-HumanoidSoccer-PAiD-Stage3-v0 \
  --headless \
  --device cuda:0 \
  --resume_checkpoint <PATH_TO_STAGE2_CHECKPOINT>
```

## Evaluation

Static ball:

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-HumanoidSoccer-PAiD-Eval-Static-v0 \
  --checkpoint <PATH_TO_STAGE3_CHECKPOINT>
```

Rolling ball:

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-HumanoidSoccer-PAiD-Eval-Rolling-v0 \
  --checkpoint <PATH_TO_STAGE3_CHECKPOINT>
```

## Export for Deployment

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-HumanoidSoccer-PAiD-Play-v0 \
  --checkpoint <PATH_TO_STAGE3_CHECKPOINT>
```

Exported artifacts are generated under:

- `logs/rsl_rl/<EXPERIMENT>/<RUN>/exported/`

## System Identification (optional)

Fit one contact profile from measured trajectories:

```bash
python scripts/soccer_sysid_fit.py \
  --drop_csv <DROP_DATA.csv> \
  --rolling_csv <ROLL_DATA.csv> \
  --surface hard \
  --output_yaml source/booster_train/docs/ball_contact_profiles.hard.yaml
```

Repeat with `--surface grass` for grass profile.
