#!/usr/bin/bash
# ICML main-paper protocol: three seeds, 100 trials each, test-set objectives.
#SBATCH --job-name=CC_DIST_MOHPO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=KathleenG
#SBATCH --gres=gpu:h100:1
#SBATCH --time=14-00:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2%3
#SBATCH --output=/storage/work/kaulen/ctrain_dev/CTRAIN/slurm_logs/cc-dist-hpo-%A_%a.log
#SBATCH --chdir=/storage/work/kaulen/ctrain_dev/CTRAIN

set -euo pipefail
STUDY_ROOT="${1:?Pass the fresh study root containing frozen source and teacher weights}"
module load Python/3.11
source /storage/work/kaulen/ctrain_dev/venv/bin/activate
export PYTHONPATH="${STUDY_ROOT}/source"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ctrain-mpl-${SLURM_JOB_ID}"
cd "${STUDY_ROOT}/source"
if [[ "${2:-}" == "--check" ]]; then
  python -u tests/test_cc_dist_hpo.py --cuda --teacher "${STUDY_ROOT}/teacher/model.pt"
  exit
fi
python -u papers/rethinking_evaluation_paradigms/mo_hpo/run_hpo.py \
  --dataset cifar10 --network cnn7 --method cc_dist --eps 0.00784313725490196 \
  --seed "$SLURM_ARRAY_TASK_ID" --epochs 160 --batch-size 512 \
  --budget-trials 100 --sampler botorch --min-cert-acc 0.40 --min-nat-acc 0.60 \
  --teacher "${STUDY_ROOT}/teacher/model.pt" --teacher-sha256 "$(cat "${STUDY_ROOT}/teacher/sha256.txt")" \
  --data-root /storage/work/kaulen/ctrain_dev/CTRAIN/data \
  --output-root "${STUDY_ROOT}/optuna_studies" --device cuda
