#!/usr/bin/env bash
# P007: the CPU half of the benchmark -- the model-free baselines and the
# scoring of a rollout that has already been generated.
#
# Kept apart from the GPU half because none of it needs a GPU and all of it is
# arm-independent: the persistence, static and cross-replica baselines are
# properties of the MD data, so they are computed once and read by all four
# arms' reports.  Twelve GPUs are busy with P009's four arms; nothing here has
# to wait for them.
#
#   sbatch --export=ALL,BENCH_ARGS="--out runs/p007/oneshot/dryrun --stage baselines --n-windows 32" \
#          kineidos/bench/slurm/p007_cpu.sbatch
#
# BENCH_MODULE defaults to kineidos.bench.oneshot; set it to
# kineidos.bench.rollout for the rollout scoring pass.
#SBATCH --job-name=p007-cpu
#SBATCH --partition=background
#SBATCH --qos=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=deep-chungus-8,deep-h-1,deep-h-2,deep-h-3
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
# The GAGU dataloader goes through Protenix's featurizer, so this needs the v2
# environment even though no model is built.  mdtraj is present in both.
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
BENCH_MODULE="${BENCH_MODULE:-kineidos.bench.oneshot}"
BENCH_ARGS="${BENCH_ARGS:?set BENCH_ARGS to the arguments for BENCH_MODULE}"

cd "${WORKSPACE}"
export PYTHONPATH="repos/research/kineidos-v3:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
# No GPU here, so the cuequivariance kernels cannot load (they dlopen
# libcuda.so.1); the torch paths are numerically the same thing.
export TRIANGLE_ATTENTION=torch
export TRIANGLE_MULTIPLICATIVE=torch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname) $(date -Is)"
echo "MODULE=${BENCH_MODULE} ARGS=${BENCH_ARGS}"
exec "${ENV}/bin/python" -u -m "${BENCH_MODULE}" ${BENCH_ARGS}
