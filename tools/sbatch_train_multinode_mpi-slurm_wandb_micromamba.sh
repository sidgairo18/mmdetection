#!/bin/bash

NGPUS=$1
CONFIG=$2
WORK_DIR=$3
PY_ARGS=${@:4}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
TIME_LIMIT=${TIME_LIMIT:-"23:59:00"}
JN=${JN:-"train"}
PART=${PART:-"gpu17,gpu16,gpu22,gpu24"}
PROJECT=${PROJECT:-"codetr_minicoco"}
ENTITY=${ENTITY:-"sidgairo18-saarland-informatics-campus"}
CHAIN_JOBS=${CHAIN_JOBS:-"20"}

# Find a free port
while true; do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null || break
done

echo "My Port ${PORT}"
echo "NUM_GPUS=${NGPUS}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
NNODES=$((NGPUS / GPUS_PER_NODE))
MEMORY=$((GPUS_PER_NODE * 110))
echo "NNODES=${NNODES}"
echo "python args: ${PY_ARGS}"

mkdir -p ${WORK_DIR}/"slurm_logs"
echo "WORK_DIR ${WORK_DIR}"
ls ${WORK_DIR}
#SLURM_LOGS="$(pwd -P)"${WORK_DIR}/slurm_logs
SLURM_LOGS=${WORK_DIR}/slurm_logs
echo ${SLURM_LOGS}

sbatch << EOT
#!/bin/bash

#SBATCH --job-name=${JN}
#SBATCH --partition=${PART}
#SBATCH --nodes=${NNODES}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=5
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}GB
#SBATCH -o ${SLURM_LOGS}/slurm-exp_${JN}_%A_%a.out
#SBATCH -a 1-${CHAIN_JOBS}%1

echo "My Port ${PORT}"
echo "Start time: \$(date)"
echo "using GPU \${CUDA_VISIBLE_DEVICES}"
echo "Slurm job ID \${SLURM_ARRAY_TASK_ID}"
echo "python args: ${PY_ARGS}"

# Export helper useful env variables
source set_dataset_paths.sh

source ~/.bashrc
eval "\$(conda shell.bash hook)"
micromamba activate ${CURR_MAMBA_ENV}

export PYTHONPATH=\$MMDET_PATH:\$PYTHONPATH
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export PATH=\$PATH:/sbin

echo "PYTHON PATH: \$PYTHONPATH"
echo "Conda env: \$CONDA_DEFAULT_ENV"
which python
python -c "import mmengine" || { echo "mmengine not found in env! Exiting."; exit 1; }
python -c "import mmcv" || { echo "mmcv not found in env! Exiting."; exit 1; }

# SETUP WANDB
# 0) Clear stale env (important when re-sourcing shells)                                      
unset WANDB_RESUME WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_TAGS WANDB_PROJECT WANDB_ENTITY
                                                                                              
# 1) Seed W&B env from cfg (writes $WORK_DIR/_wandb_env.sh and wandb_run_id.txt)
ENV_FILE=\$(python -m mmdet.engine.hooks.wandb_helpers --cfg "$CONFIG" --work-dir "$WORK_DIR" --project "$PROJECT" --entity "$ENTITY") || { echo "wandb_helpers failed"; exit 1; }
                                                                                              
echo "ENV_FILE=\$ENV_FILE"
if [[ -z "\${ENV_FILE:-}" || ! -f "\$ENV_FILE" ]]; then
  echo "No env file produced: \$ENV_FILE"; exit 1
fi
# shellcheck disable=SC1090
source "\$ENV_FILE"

echo "W&B env seeded:"
echo "  PROJECT=\$WANDB_PROJECT"
echo "  ENTITY=\$WANDB_ENTITY"
echo "  NAME=\$WANDB_NAME"
echo "  RUN_GROUP=\$WANDB_RUN_GROUP"
echo "  TAGS=\$WANDB_TAGS"
echo "  RUN_ID=\$WANDB_RUN_ID RESUME=\$WANDB_RESUME"

srun --cpu-bind=none ${SRUN_ARGS} python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}

# Cancel pending jobs once one task finishes successfully
if [ \$? -eq 0 ]; then
    echo "Experiments over"
    scancel -t PENDING \${SLURM_ARRAY_JOB_ID}
    exit
fi

EOT
