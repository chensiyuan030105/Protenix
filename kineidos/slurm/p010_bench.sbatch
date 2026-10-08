#!/usr/bin/env bash
# P010 D9: measure one training step, so section 8's compute table stops being
# an extrapolation.
#
# The first attempt (job 2149158, 2026-10-07) hit its one-hour limit having
# printed six lines and written no json.  Three things are different here, and
# only the third is about the script:
#
#   * **two hours, not one**, and kineidos.bench_step now has its own
#     --max-seconds so it stops and writes rather than being killed;
#   * **a quiet node with an 80 GB card.**  2149158 landed on deep-chungus-3:
#     an A100-PCIE-40GB at CPULoad 38, shared with our own P006 GROMACS array.
#     Load correlates cleanly with step time (4-6.5 -> 6-7.7 s/it, 33 -> 14.1),
#     and 48 diffusion samples of a 470-atom window plus a 20-step mini-rollout
#     is not a 40 GB job to run beside someone else's;
#   * **the三道保险**, which 2149158 did not have at all.
#
#   sbatch --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G \
#          --nodelist=deep-chungus-9,deep-chungus-10,deep-chungus-11 \
#          --export=ALL,WP_MODE=random \
#          repos/research/kineidos-v3-diag/kineidos/slurm/p010_bench.sbatch
#SBATCH --job-name=p010-bench
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=deep-h-1,deep-h-2,deep-h-3,deep-chungus-8,deep-gpu-10,deep-gpu-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v3-diag}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
WP_MODE="${WP_MODE:-random}"
OUT="${OUT:-${WORKSPACE}/artifacts/reports/P010/step_budget.json}"

mkdir -p "${WORKSPACE}/runs/slurm" "$(dirname "${OUT}")"
cd "${WORKSPACE}"

# AGENTS.md's four, all written in the command.  The diagnostic tree, not v3:
# the four P009 arms are reading v3 and it must not move under them.
export PYTHONPATH="${TREE}:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-cuequivariance}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "TREE=${TREE} WP_MODE=${WP_MODE} OUT=${OUT}"
NGPU_WANT=1
# shellcheck source=./p010_preflight.sh
source "${TREE}/kineidos/slurm/p010_preflight.sh"

exec "${ENV}/bin/python" -u -m kineidos.bench_step \
  --mode "${WP_MODE}" \
  --out "${OUT}" \
  --max-seconds "${MAX_SECONDS:-6000}" \
  "$@"
