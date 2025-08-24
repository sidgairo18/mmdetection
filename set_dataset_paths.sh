#!/usr/bin/env bash
# Unified cluster env + dataset links
# Usage: source set_dataset_paths.sh

# --- repo root detection (from cluster.env) ---
if [ -z "${MMDET_PATH:-}" ]; then
  if [ -f tools/train.py ] && [ -d mmdet ]; then
    export MMDET_PATH="$(pwd)"
  else
    export MMDET_PATH="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  fi
fi

mkdir -p "$MMDET_PATH/data"

# --- cluster detection (override with: export CLUSTER_ID=...) ---
if [ -z "${CLUSTER_ID:-}" ]; then
  if [ -d /raven/datasets ]; then
    CLUSTER_ID="raven"
  elif [ -d /ista/datasets ]; then
    CLUSTER_ID="ista"
  else
    hn="$(hostname -f 2>/dev/null || hostname)"
    case "$hn" in
      raven*|*raven*)                 CLUSTER_ID="raven" ;;
      slurm-submit*|*slurm-submit*)   CLUSTER_ID="local" ;;
      gpu[0-9]*)                      CLUSTER_ID="ista"  ;;  # fallback for ISTA naming
      *)                              CLUSTER_ID="local" ;;
    esac
  fi
fi
export CLUSTER_ID

# --- per-cluster dataset roots (from cluster.env) ---
case "$CLUSTER_ID" in
  raven)
    export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/ptmp/sgairola/work/data/DETECTRON2_DATASETS}"
    export COCO_ROOT="$DETECTRON2_DATASETS/coco"
    export LVIS_ROOT="$DETECTRON2_DATASETS/lvis"
    export CURR_MAMBA_ENV="openmmlab_mmdet_xx"
    ;;
  ista)
    export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/nfs/scistore19/locatgrp/sgairola/data/DETECTRON2_DATASETS}"
    export COCO_ROOT="$DETECTRON2_DATASETS/coco"
    export LVIS_ROOT="$DETECTRON2_DATASETS/lvis"
    export CURR_MAMBA_ENV="mmdet308_c121"
    ;;
  *)
    # LOCAL default (keeps your previous path, overrideable)
    export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS}"
    export COCO_ROOT="$DETECTRON2_DATASETS/coco"
    export LVIS_ROOT="$DETECTRON2_DATASETS/lvis"
    export CURR_MAMBA_ENV="openmmlab_mmdet_xx"
    ;;
esac

# --- link_if_missing (from scripts/links.sh) ---
link_if_missing() {
  local target="$1" link="$2"
  mkdir -p "$(dirname "$link")"

  if [ -L "$link" ]; then
    local current; current="$(readlink "$link")" || current=""
    if [ "$current" = "$target" ]; then
      echo "ok: $link -> $current"
    else
      echo "update: $link (was $current) -> $target"
      ln -sfn "$target" "$link"
    fi
  elif [ -e "$link" ]; then
    echo "error: $link exists and is not a symlink" >&2
    return 1
  else
    ln -s "$target" "$link"
    echo "created: $link -> $target"
  fi
}

# --- wire datasets (from set_dataset_paths.sh) ---
link_if_missing "$COCO_ROOT" "$MMDET_PATH/data/coco"
link_if_missing "$LVIS_ROOT" "$MMDET_PATH/data/lvis"

# --- summary ---
echo "Cluster detected: $CLUSTER_ID"
echo "MMDET_PATH: $MMDET_PATH"
echo "COCO_ROOT:  $COCO_ROOT"
echo "LVIS_ROOT:  $LVIS_ROOT"
echo "CURR_MAMBA_ENV:  $CURR_MAMBA_ENV"
echo "DETECTRON2_DATASETS: $DETECTRON2_DATASETS"
