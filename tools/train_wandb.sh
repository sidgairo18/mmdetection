#!/usr/bin/env bash
set -euo pipefail

CFG="$1"               # path to your config.py
GPUS="${2:-1}"         # num gpus for dist_train.sh
WORK_DIR="${3:-work_dirs/test_wandb}"
PROJECT="codetr_minicoco"

mkdir -p "$WORK_DIR"

# 0) Clear stale env (important when re-sourcing shells)
unset WANDB_RESUME WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_TAGS WANDB_PROJECT WANDB_ENTITY

# 1) Seed W&B env from cfg (writes $WORK_DIR/_wandb_env.sh and wandb_run_id.txt)
ENV_FILE=$(python -m mmdet.engine.hooks.wandb_helpers \
  --cfg "$CFG" --work-dir "$WORK_DIR" --project "$PROJECT" --entity "$ENTITY") || {
  echo "wandb_helpers failed"; exit 1; }

echo "ENV_FILE=$ENV_FILE"
if [[ -z "${ENV_FILE:-}" || ! -f "$ENV_FILE" ]]; then
  echo "No env file produced: $ENV_FILE"; exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

echo "W&B env seeded:"
echo "  PROJECT=$WANDB_PROJECT"
echo "  ENTITY=$WANDB_ENTITY"
echo "  NAME=$WANDB_NAME"
echo "  RUN_GROUP=$WANDB_RUN_GROUP"
echo "  TAGS=$WANDB_TAGS"
echo "  RUN_ID=$WANDB_RUN_ID RESUME=$WANDB_RESUME"

# 2) Launch training
CUDA_VISIBLE_DEVICES=0,1 PORT=29509 \
bash ./tools/dist_train.sh "$CFG" "$GPUS" \
  --work-dir "$WORK_DIR" \
  --auto-scale-lr \
  --resume
