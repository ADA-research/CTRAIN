#!/bin/bash
#SBATCH --job-name=MTL_ALPHA_SWEEP
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --partition=KathleenG
#SBATCH --qos=gpu
#SBATCH --output=papers/rethinking_evaluation_paradigms/logs/submitit/mtl_alpha_sweep/%A_%a.out
#SBATCH --error=papers/rethinking_evaluation_paradigms/logs/submitit/mtl_alpha_sweep/%A_%a.err

set -euo pipefail

# module load GCCcore/.13.2.0
# module load Python/3.11.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${REPO_ROOT}/papers/rethinking_evaluation_paradigms/logs/submitit/mtl_alpha_sweep"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/papers/rethinking_evaluation_paradigms:${REPO_ROOT}:${PYTHONPATH:-}"

EPS_VALUES=("2/255" "8/255")
N_ALPHA_VALUES=20
EPS_INDEX=$((SLURM_ARRAY_TASK_ID / N_ALPHA_VALUES))
ALPHA_INDEX=$((SLURM_ARRAY_TASK_ID % N_ALPHA_VALUES))
EPS="${EPS_VALUES[${EPS_INDEX}]}"

echo "Running MTL-IBP alpha sweep for eps=${EPS}, alpha_index=${ALPHA_INDEX}"
srun --unbuffered "${PYTHON}" -u \
  papers/rethinking_evaluation_paradigms/mo_hpo/run_mtl_ibp_alpha_sweep.py \
  "${EPS}" \
  --alpha-index "${ALPHA_INDEX}"
