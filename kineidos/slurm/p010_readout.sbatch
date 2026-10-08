#!/usr/bin/env bash
# P010 section 6's readings, and the two acceptances that are file comparisons.
#
# A slurm job rather than a login-node command, by the rule AGENTS.md and P006
# D19 state without qualification: **every CPU or GPU heavy task goes through
# sbatch; the login node submits, inspects lightly, and runs git.**  No GPU is
# requested -- nothing here touches one -- but ten arms at 11k rows is 110k
# json objects to parse, join on three keys and reduce, and the login node is
# shared.
#
#   # the D7 reading, off the P009 checkpoints
#   sbatch --export=ALL,STEP=2999 \
#          repos/research/kineidos-v3-diag/kineidos/slurm/p010_readout.sbatch
#
#   # once the diagnostic arms exist, their own step
#   sbatch --export=ALL,STEP=2000,ARGS='--random random-1gpu --zero zero-1gpu --floor random-1gpu-seed2 --none "" --oracle oracle --decoy oracle-decoy --random1 random-1gpu --zero1 zero-1gpu' ...
#
#   # section 4's acceptance: none == zero at step 0, and a rerun is identical
#   sbatch --export=ALL,MODE=identical,A=...,B=... ...
#SBATCH --job-name=p010-readout
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v3-diag}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"

MODE="${MODE:-read}"
STEP="${STEP:-2999}"
RUNS="${RUNS:-${WORKSPACE}/runs/p010/sigma_grid}"
OUT="${OUT:-${WORKSPACE}/artifacts/reports/P010/readout.md}"

mkdir -p "${WORKSPACE}/runs/slurm" "$(dirname "${OUT}")"
cd "${WORKSPACE}"

# No GPU, so no cuequivariance and no libcuda: the triangle kernels would be
# built if a model were built, and none is.  The rest is AGENTS.md's.
export PYTHONPATH="${TREE}:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-torch}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-torch}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname) MODE=${MODE}"

if [ "${MODE}" = "identical" ]; then
  exec "${ENV}/bin/python" -u -m kineidos.read_sigma_grid \
    --identical "${A:?set A to the first jsonl}" "${B:?set B to the second}"
fi

# The config-and-arithmetic acceptance, which is cheap but is still a job: the
# rule has no size qualifier, and "it is only a parse" is how things end up on
# the login node.  TREES lets one job check all three branches, since each adds
# its own layer to the bottom of check_p010.
if [ "${MODE}" = "check" ]; then
  RC=0
  for T in ${TREES:-${TREE}}; do
    echo ""
    echo "=== $(basename "${T}") ==="
    PYTHONPATH="${T}:repos/research/wp-v2" \
      "${ENV}/bin/python" -u -m kineidos.check_p010 || RC=1
  done
  exit "${RC}"
fi

# shellcheck disable=SC2086
exec "${ENV}/bin/python" -u -m kineidos.read_sigma_grid \
  --runs "${RUNS}" \
  --step "${STEP}" \
  --out "${OUT}" \
  ${ARGS:-} "$@"
