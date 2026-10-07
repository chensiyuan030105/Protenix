#!/usr/bin/env bash
# P009 acceptance, the parts that can be settled before the arms start.
#
#   kineidos.check_dt_range  -- section 6.1 condition 2 (what dt the dataloader
#                               really produces) and condition 4's floor of 50
#                               windows per bin, which is a property of
#                               EVAL_WINDOWS and the seed, not of the training.
#   kineidos.check_fusion    -- section 6.1 condition 3: the fusion is still
#                               invisible at initialisation, so `zero` and
#                               `none` are still bit-identical at step 0.  The
#                               point of rerunning P004's script is to show
#                               that changing DT_MAX_NS did not touch it.
#
# CPU only: neither check needs a GPU, and check_dt_range's cost is reading 52
# trajectories (~6 GB).  That is more than a login node should carry, which is
# why this is a batch job (AGENTS.md).
#
#   sbatch kineidos/slurm/p009_checks.sbatch
#SBATCH --job-name=p009-checks
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"

cd "${WORKSPACE}"
export PYTHONPATH="repos/research/kineidos-v3:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
# No GPU here, so the cuequivariance kernels are not available; check_fusion
# builds AtomAttentionEncoders on CPU and does not reach them, but the triangle
# modules are constructed during config parsing elsewhere, so say torch.
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-torch}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"

status=0
echo "=============== kineidos.check_dt_range ==============="
"${ENV}/bin/python" -m kineidos.check_dt_range || status=1
echo
echo "=============== kineidos.check_fusion ================="
"${ENV}/bin/python" -m kineidos.check_fusion || status=1

echo
echo "p009-checks exit status: ${status}"
exit "${status}"
