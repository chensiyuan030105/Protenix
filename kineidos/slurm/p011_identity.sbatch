#!/usr/bin/env bash
# P011 section 6: the dt pathway's step-0 identity, plus P010's fusion identity
# re-run on this tree.
#
# Two scripts, because they answer two different "is it invisible at step 0"
# questions and D8 says a check cannot stand in for a different check:
#   kineidos.check_dt_conditioning  the dt pathway (section 6), with the
#                                   perturbation control that catches a pathway
#                                   wired to nothing;
#   kineidos.check_fusion           the fusion point, inherited from P004.3 and
#                                   re-run here because `kineidos-v4` is a new
#                                   tree -- `zero-v4` has to equal `none` on it,
#                                   not on the branch it was last measured on.
#
# CPU, no preflight: both build small modules and never touch a device.
#
#   sbatch --cpus-per-task=4 --mem=16G \
#          repos/research/kineidos-v4/kineidos/slurm/p011_identity.sbatch
#SBATCH --job-name=p011-identity
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=deep-h-1,deep-h-2,deep-h-3,deep-chungus-7,deep-chungus-8,deep-gpu-10,deep-gpu-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v4}"
WPTREE="${WPTREE:-repos/research/wp-v3}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"

mkdir -p "${WORKSPACE}/runs/slurm"
cd "${WORKSPACE}"

export PYTHONPATH="${TREE}:${WPTREE}"
export PYTHONUNBUFFERED=1
# See p011_checkup.sbatch: NFS attribute caching makes Python import bytecode
# older than the source it was compiled from.
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# The criterion here *is* bit identity, which is what determinism.py's
# docstring says this switch is for.
export KINEIDOS_DETERMINISTIC="${KINEIDOS_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname) DATE=$(date -Is)"
echo "${TREE} HEAD=$(git -C "${TREE}" rev-parse --short HEAD)"
echo "${WPTREE} HEAD=$(git -C "${WPTREE}" rev-parse --short HEAD)"

RC=0
for mod in kineidos.check_dt_conditioning kineidos.check_fusion; do
  echo ""
  echo "=================================================================="
  echo "=== ${mod}"
  echo "=================================================================="
  # `|| rc=$?` per script, so one failure does not stop the other from being
  # read.  Under `set -e` a bare call would exit here and the second check
  # would never run -- and then "which of the two failed" needs another job.
  rc=0
  "${ENV}/bin/python" -u -m "${mod}" || rc=$?
  echo "--- ${mod} exit=${rc}"
  [ "${rc}" -eq 0 ] || RC=1
done

echo ""
echo "=== section 6 + P004.3 identity: $([ "${RC}" -eq 0 ] && echo PASS || echo FAIL) ==="
exit "${RC}"
