#!/usr/bin/env bash
# P004: Protenix diffusion head + WorldParticle injection, AF3 trunk frozen.
#
# One GPU on purpose.  runner/train.py builds DDP with static_graph=True, which
# requires the same parameters to participate every iteration, and
# main_train_loop drops embeddings at two places by chance; turning the
# confidence head off removes one of the two and the combination has never been
# run on more than one device (plan 2.14).  Multi-GPU comes after a two-card
# smoke test, not before.
#
# The arm is the one argument that must be given:
#   sbatch --export=ALL,WP_MODE=zero kineidos/slurm/p004_train.sbatch
#SBATCH --job-name=p004-kineidos
#SBATCH --partition=background
#SBATCH --qos=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
CHECKPOINT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt"

WP_MODE="${WP_MODE:?set WP_MODE to none, zero, random or pretrained}"
RUN_NAME="${RUN_NAME:-p004_${WP_MODE}}"
MAX_STEPS="${MAX_STEPS:-2000}"
SEED="${SEED:-42}"
# Disk arithmetic, because /mnt/xfs sits at 97% (AGENTS.md).  save_checkpoint
# writes the whole 368.60M state_dict plus Adam's two moments for the 203.37M
# trainable parameters -- about 3.1 GB a file -- and the frozen trunk in every
# one of them is a byte-for-byte copy of the pretrained checkpoint.  Upstream's
# saver is left alone because resumability is worth more than the space here,
# but the interval is what bounds the cost: at 500 and MAX_STEPS=2000 that is
# five files, ~16 GB per arm, ~47 GB for none/zero/random.  Raise the interval
# before raising MAX_STEPS.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"

mkdir -p "${WORKSPACE}/runs/slurm" "${WORKSPACE}/runs/p004"
cd "${WORKSPACE}"

# Every one of these is load-bearing; see AGENTS.md's "Python 环境与 PYTHONPATH".
export PYTHONPATH="repos/research/kineidos-v2:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV}/lib"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
# Defaults to the plain-PyTorch LayerNorm, which needs no nvcc.  The fused path
# is faster where nvcc exists and is weight-identical (same state_dict keys), so
# it is safe to switch -- but it must be a decision, written in the command.
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
# The cuequivariance kernels need a GPU; they are the default here and only
# have to drop to "torch" on the login node.
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-cuequivariance}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "WP_MODE=${WP_MODE} RUN_NAME=${RUN_NAME} MAX_STEPS=${MAX_STEPS} SEED=${SEED}"
"${ENV}/bin/python" -u -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())'

# env.lock is written by the trainer, into the run directory it creates --
# init_basics appends a timestamp to run_name, so a lock written from here would
# land next to the run instead of inside it.  See KineidosTrainer.write_env_lock.

exec "${ENV}/bin/python" -u -m kineidos.train.main \
  --wp.mode "${WP_MODE}" \
  --load_checkpoint_path "${CHECKPOINT}" \
  --project p004 \
  --run_name "${RUN_NAME}" \
  --base_dir "${WORKSPACE}/runs/p004" \
  --max_steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --log_interval 20 \
  --eval_interval -1 \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --use_wandb false \
  --skip_confidence true \
  --skip_distogram true \
  --loss.weight.alpha_confidence 0.0 \
  --loss.weight.alpha_distogram 0.0 \
  "$@"
