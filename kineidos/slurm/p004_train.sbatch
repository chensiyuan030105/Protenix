#!/usr/bin/env bash
# P004/P009: Protenix diffusion head + WorldParticle injection, trunk frozen.
#
# Written for P004 and kept at this path because P009 section 4 cites it by
# name.  Under P009 it drives the v3 protocol: dt in [0.1, 1] ns (the constant
# lives in kineidos/data/windows.py), 256 held-out windows, 10000 steps, and
# runs/p009 for the output.  Everything about the fusion, the freezing and the
# four arms is P004's and unchanged.
#
# DDP via torchrun.  runner/train.py builds DDP with static_graph=True, which
# requires the same parameters to participate every iteration.  That turned out
# to be satisfied: main_train_loop's two probabilistic drops set
# use_conditioning=False, which *zeroes the tensors* rather than skipping
# modules (diffusion.py:143), so the graph is identical every iteration -- and
# both drop rates are 0.0 by default anyway.  find_unused_parameters=False is
# likewise fine: frozen parameters are excluded from DDP's reducer rather than
# being "unused", and the WorldParticle bridge is a submodule whose output is in
# the graph every iteration.
#
# All arms must share one world size.  Different GPU counts mean different
# numbers of windows per step, which is a different optimisation trajectory --
# the ablation would stop being an ablation.  They need not start at the same
# time, only to match.
#
#   sbatch --gres=gpu:a100:3 --cpus-per-task=12 --mem=160G \
#          --nodelist=deep-chungus-7 \
#          --export=ALL,WP_MODE=zero,NGPU=3 \
#          kineidos/slurm/p004_train.sbatch
#
# The fourth P009 arm is the noise floor: the same WP_MODE=zero with a different
# training seed, which is a different 10,000 windows as well as different
# initialisation.  Give it its own RUN_NAME or it resumes into the first one:
#   --export=ALL,WP_MODE=zero,NGPU=3,SEED=43,RUN_NAME=p009_zero_seed2
#SBATCH --job-name=p009-kineidos
# background on the A100 nodes, with --requeue and a resume path.
#
# Not h100: those GPUs are broken.  deep-h-1 is drained with reason
# "UserRequestedGPURecovery-20260919", and on deep-h-2 nvidia-smi itself fails
# with "Unable to determine the device handle for GPU0 ... Unknown Error" while
# slurm still happily allocates gres:gpu:1 and torch reports cuda=False.  Three
# jobs ran there CPU-only before this was noticed.  Sixteen idle GPUs on a
# cluster whose A100s are fully booked is a symptom, not an opportunity.
#
# Not high-priority either: it is tier 10 and would not be preempted, but it
# hits QOSGrpJobsLimit in practice.  background has capacity on the working
# A100 nodes.  The cost is PriorityTier=5 under PreemptMode=REQUEUE with
# PreemptExemptTime=0 -- killed without grace whenever a tier-10 job wants the
# node -- which is why KineidosTrainer.resume and --requeue exist.  Verified on
# deep-chungus-3: A100-PCIE-40GB, torch sees it, and open3d 0.20.0's compiled
# continuous_conv runs on CUDA forward and backward with nonzero gradients.
# high-priority（2026-10-07 改）。原先用 background 是因为 high-priority 当时被
# QOSGrpJobsLimit 挡住（全组 GrpJobs=8）；那个限制后来松开了。改的理由是实测代价：
# background 是 PriorityTier=5、PreemptMode=REQUEUE、PreemptExemptTime=0，`none`
# 档在 14 小时内被抢占两次，每次损失到上一个 checkpoint 的距离加一次 10–35 分钟的
# 启动，而且 requeue 复用同一个作业 ID ⟹ **日志被覆盖**，它前三个干净 eval 点就是
# 这么丢的。high-priority 是 tier 10，不被抢占；每用户 12 张 GPU 的额度装得下三档
# 共 9 张。
#SBATCH --partition=high-priority
#SBATCH --qos=high-priority
#SBATCH --account=grad-students
#SBATCH --requeue
# 追加而非覆盖。requeue 复用作业 ID，默认的覆盖模式会抹掉重排前的全部日志——
# 包括抢占本身的证据，于是连"为什么重启了"都查不到。
#SBATCH --open-mode=append
# deep-chungus-7's GPU 7 is broken; see AGENTS.md's 已知坏卡 and
# kineidos/slurm/p010_bench.sbatch for the measurements.  A requeued arm
# must not land on it: this template's startup check asks
# torch.cuda.is_available(), which may answer True on a card whose kernels
# then fail, and the arm would simply be "the slow one".
#SBATCH --exclude=deep-h-1,deep-h-2,deep-h-3,deep-chungus-7,deep-chungus-8,deep-gpu-10,deep-gpu-11
# One task; torchrun forks one process per GPU inside it.  --gres and --mem are
# passed on the sbatch command line so NGPU can vary without editing this file.
#SBATCH --nodes=1
#SBATCH --ntasks=1
# MaxTime is UNLIMITED; DefaultTime is 08:00:00.  Resume makes a long request
# safe, since an interruption of any kind now costs at most one checkpoint
# interval rather than the whole run.
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
CHECKPOINT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt"

WP_MODE="${WP_MODE:?set WP_MODE to none, zero, random or pretrained}"
NGPU="${NGPU:-1}"
RUN_NAME="${RUN_NAME:-p009_${WP_MODE}}"
# Explicit and identical across arms.  P004's three arms stopped at different
# step counts, which left exactly two checkpoints that could be compared across
# arms; at ~6.5 s a step, 10000 steps is about 18 hours (P009 section 4, item 4).
MAX_STEPS="${MAX_STEPS:-10000}"
SEED="${SEED:-42}"
# Disk arithmetic, because /mnt/xfs sits at 97% (AGENTS.md).  save_checkpoint
# writes the whole 368.60M state_dict plus Adam's two moments for the 203.37M
# trainable parameters -- about 3.1 GB a file -- and the frozen trunk in every
# one of them is a byte-for-byte copy of the pretrained checkpoint.  Upstream's
# saver is left alone because resumability is worth more than the space here,
# but the interval is what bounds the cost: at 1000 and MAX_STEPS=10000 that is
# ten files, ~31 GB per arm, ~124 GB for the four P009 arms, plus one latest.pt
# per arm in RESUME_DIR.  Checked against df before submitting: 3.9 TB free of
# 112 TB on 2026-10-07, so this fits -- but /mnt/xfs sits at 97% and the margin
# is the cluster's, not ours.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
# Held-out scoring runs inside training.  P004 used 64 windows -- enough for a
# mean, not enough to split.  P009 reads the result in three dt bins (stride
# 1-2, 3-5, 6-10) and asks for at least 50 windows in each, and the log-uniform
# draw puts only 2.2% of windows at stride 10, so 256 is the smallest round
# number that fills the thin bin: about 102 / 88 / 66 by section 2.4's weights.
# Four times P004's eval cost, still far below the 500 training steps it covers.
# The config default stays 64; this is passed through --kineidos.eval_windows.
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_WINDOWS="${EVAL_WINDOWS:-256}"

# Train / held-out split.  The 64 GAGU trajectories are a balanced
# 2 salt x 4 sequence x 2 start x 4 replicate design, so the split can be made
# along one axis cleanly.  Replicates r1-r3 train, r4 is held out: same
# chemistry, same conditions, different MD noise.  That asks "did it learn the
# dynamics" rather than "did it memorise this trajectory" -- the weaker of the
# two questions, and the one to settle first.  Holding out a whole sequence
# instead asks about chemistry generalisation and is a separate set of runs.
GAGU_ROOT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/processed/gagu_internal_loop_v0_1"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-$({ ls -d ${GAGU_ROOT}/*_r[123] 2>/dev/null || true; } | xargs -r -n1 basename | paste -sd,)}"
HELD_OUT="${HELD_OUT:-$({ ls -d ${GAGU_ROOT}/*_r4 2>/dev/null || true; } | xargs -r -n1 basename | paste -sd,)}"
N_TRAIN=$(echo "${TRAIN_SAMPLES}" | tr ',' '\n' | grep -c .)
N_HELD=$(echo "${HELD_OUT}" | tr ',' '\n' | grep -c .)

# Per-arm, and stable across requeues -- init_basics appends a timestamp to the
# run directory, so a requeued job cannot find its predecessor's checkpoints by
# any name it can derive itself.
RESUME_DIR="${RESUME_DIR:-${WORKSPACE}/runs/p009/${RUN_NAME}_resume}"

mkdir -p "${WORKSPACE}/runs/slurm" "${WORKSPACE}/runs/p009" "${RESUME_DIR}"
cd "${WORKSPACE}"

# Every one of these is load-bearing; see AGENTS.md's "Python 环境与 PYTHONPATH".
# v3, not v2: P009 changes the protocol, so it gets the version number and v2
# stays equal to "the code P004 ran" (P009 section 3).  env.lock records which
# tree the import actually resolved to, so a stale value here is caught rather
# than inherited.
export PYTHONPATH="repos/research/kineidos-v3:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
# Prepend, do not replace.  AGENTS.md documents the replacing form, which works
# on the login node only because libcuda is not needed there; on a GPU node it
# would hide the driver libraries that live in the system paths.
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
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
# Suggested by the allocator itself when it ran out.  Our own footprint is small
# -- a two-rank probe was using 3.9 GiB when it died -- but the GPUs here are not
# exclusive: slurm allocated us a device on deep-chungus-3 that already had
# another user's 35.18 GiB process on it.  gres accounting says a device is free;
# it does not say the memory is.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "WP_MODE=${WP_MODE} RUN_NAME=${RUN_NAME} MAX_STEPS=${MAX_STEPS} SEED=${SEED}"
echo "TRAIN_SAMPLES: ${N_TRAIN} trajectories (r1-r3); HELD_OUT: ${N_HELD} (r4)"
echo "RESUME_DIR=${RESUME_DIR} (exists: $([ -e "${RESUME_DIR}/latest.pt" ] && echo yes || echo no))"
# Fatal, not informational.  Three jobs ran to completion on CPU on deep-h-2
# because slurm allocated gres:gpu:1 on a node whose driver could not talk to
# the devices -- torch reported cuda=False and nothing stopped the run.  A
# silently CPU-only training job is ~100x too slow and produces a run directory
# that looks exactly like a real one.
"${ENV}/bin/python" -u - "${NGPU}" <<'PYEOF'
import sys, torch
want = int(sys.argv[1])
have = torch.cuda.device_count()
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} devices={have}")
if not torch.cuda.is_available() or have < want:
    raise SystemExit(
        f"refusing to start: asked for {want} GPU(s), torch sees {have}. "
        f"slurm can allocate a GPU on a node whose driver is broken "
        f"(deep-h-2 did exactly that); check nvidia-smi on this node."
    )
for i in range(have):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i} {p.name} {p.total_memory / 2**30:.0f} GiB")
PYEOF

# env.lock is written by the trainer, into the run directory it creates --
# init_basics appends a timestamp to run_name, so a lock written from here would
# land next to the run instead of inside it.  See KineidosTrainer.write_env_lock.

# One master port per job, or two jobs landing on the same node collide.
MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
if [ "${NGPU}" -gt 1 ]; then
  LAUNCH=("${ENV}/bin/torchrun" "--nproc_per_node=${NGPU}" "--master_port=${MASTER_PORT}")
  echo "DDP: ${NGPU} ranks, master_port=${MASTER_PORT}"
else
  LAUNCH=("${ENV}/bin/python" "-u")
fi

exec "${LAUNCH[@]}" -m kineidos.train.main \
  --wp.mode "${WP_MODE}" \
  --load_checkpoint_path "${CHECKPOINT}" \
  --project p009 \
  --run_name "${RUN_NAME}" \
  --base_dir "${WORKSPACE}/runs/p009" \
  --max_steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --log_interval 50 \
  --eval_interval "${EVAL_INTERVAL}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --use_wandb false \
  --skip_confidence true \
  --skip_distogram true \
  --loss.weight.alpha_confidence 0.0 \
  --loss.weight.alpha_distogram 0.0 \
  --kineidos.train_samples "${TRAIN_SAMPLES}" \
  --kineidos.num_workers 4 \
  --kineidos.resume_dir "${RESUME_DIR}" \
  --kineidos.held_out_samples "${HELD_OUT}" \
  --kineidos.eval_windows "${EVAL_WINDOWS}" \
  "$@"
