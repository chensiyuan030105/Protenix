#!/usr/bin/env bash
# P011 section 5.3: the checkup, which has to pass before any arm starts (D8).
#
# Two of its four gates exist because P009 got `h` wrong four times in a row,
# each time by reasoning about it rather than measuring it.  See
# kineidos/probe_wp_content.py for what each one catches.
#
# **CPU, and no preflight**, for the same reason as p011_symmetry.sbatch: the
# bridge never leaves the CPU here, so a gres request would hold an A100 idle
# while the cluster has single-digit free cards, and p010_preflight.sh exists to
# stop a GPU job from silently running on the CPU -- not a failure mode a CPU
# job has.
#
#   sbatch --cpus-per-task=8 --mem=48G \
#          repos/research/kineidos-v4/kineidos/slurm/p011_checkup.sbatch
#SBATCH --job-name=p011-checkup
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=deep-h-1,deep-h-2,deep-h-3,deep-chungus-7,deep-chungus-8,deep-gpu-10,deep-gpu-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v4}"
WPTREE="${WPTREE:-repos/research/wp-v3}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
OUTDIR="${OUTDIR:-${WORKSPACE}/artifacts/reports/P011}"
WINDOWS="${WINDOWS:-256}"
WP_SEED="${WP_SEED:-42}"

mkdir -p "${WORKSPACE}/runs/slurm" "${OUTDIR}"
cd "${WORKSPACE}"

export PYTHONPATH="${TREE}:${WPTREE}"
export PYTHONUNBUFFERED=1
# **PYTHONDONTWRITEBYTECODE, and it is not hygiene.**  /mnt/xfs is NFS, whose
# attribute cache can serve a stale mtime for up to a minute.  Python decides a
# `.pyc` is current by comparing the source mtime recorded in its header against
# the source's, so a compute node that sees the cached mtime happily imports
# bytecode older than the file.  Jobs 2150585 and 2150589 both did exactly that:
# 2150589 started three seconds after the source was written and still raised
# `TypeError: __init__() got an unexpected keyword argument` for an argument
# that was in the signature on disk, from a .pyc ten minutes old.  Same family
# as AGENTS.md's editable-install and bf16-matmul entries -- the code that ran
# was not the code in the tree, and nothing said so.
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
# Eight threads here, one in p011_symmetry.sbatch, and the difference is not an
# oversight.  That probe gates on a residual of ~1e-4 and the multithreaded
# `index_add` spread was the same size, so it had to be single-threaded to
# measure anything.  This one gates on per-atom *fractions* (>10%, >50%) and on
# a cosine above 0.999 -- four orders of magnitude above that spread -- and has
# 256 windows to get through.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export KINEIDOS_DETERMINISTIC="${KINEIDOS_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname) DATE=$(date -Is)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "${TREE} HEAD=$(git -C "${TREE}" rev-parse --short HEAD)"
echo "${WPTREE} HEAD=$(git -C "${WPTREE}" rev-parse --short HEAD)"
echo "WINDOWS=${WINDOWS} WP_SEED=${WP_SEED}"

exec "${ENV}/bin/python" -u -m kineidos.probe_wp_content \
  --windows "${WINDOWS}" \
  --seed "${WP_SEED}" \
  --report "${OUTDIR}/checkup.json" \
  "$@"
