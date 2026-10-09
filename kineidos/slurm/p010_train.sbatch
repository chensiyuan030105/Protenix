#!/usr/bin/env bash
# P010 section 5: one diagnostic arm, one GPU, 2000 steps.
#
# The same template for all nine arms.  Three of them live on
# research/kineidos-v3-diag-oracle and research/kineidos-v3-diag-gamma, which
# inherit this file from p010-base, so TREE selects the worktree and the arm's
# own switches go through "$@".  That is deliberate: if each branch had its own
# submission script, "the arms differ only in the thing under test" would be a
# claim about four files instead of one.
#
#   # arm 0 -- D5's precondition, 200 steps, first of everything
#   sbatch --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G --nodes=1 \
#          --nodelist=deep-chungus-9 \
#          --export=ALL,ARM=base-nomini,WP_MODE=random,MAX_STEPS=200 \
#          repos/research/kineidos-v3-diag/kineidos/slurm/p010_train.sbatch
#
#   # the diagnostic baseline and its zero point
#   --export=ALL,ARM=random-1gpu,WP_MODE=random
#   --export=ALL,ARM=zero-1gpu,WP_MODE=zero
#   --export=ALL,ARM=random-1gpu-seed2,WP_MODE=random,SEED=43
#
#   # oracle arms (TREE=repos/research/kineidos-v3-diag-oracle)
#   --export=ALL,ARM=oracle,WP_MODE=oracle,TREE=...-diag-oracle
#
# **One GPU, not three, and that is not an economy.**  The P009 arms run three
# ranks, so each optimizer step sees three windows; here it sees one.  gamma's
# per-step dynamics are therefore not the same curve, and the mini-rollout is
# off in every arm here (D5) while it is on in all four P009 arms.  These arms
# are comparable to each other and to nothing else -- D3 says so, and the run
# name says p010 so that no later reader has to remember it.
#SBATCH --job-name=p010-arm
# background, not high-priority: our 12-GPU QoS allowance there is fully taken
# by the four P009 arms.  The cost of background is PriorityTier=5 with
# PreemptMode=REQUEUE and PreemptExemptTime=0 -- killed without grace whenever
# a tier-10 job wants the node -- so these are short (2000 steps) and
# checkpoint every 500, and KineidosTrainer.resume picks them up.  --requeue
# and --open-mode=append are both load-bearing: a requeue reuses the job id, so
# the default overwrite mode erases the evidence of the preemption along with
# the log (P009 lost `none`'s first three eval points that way).
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --requeue
#SBATCH --open-mode=append
# Excluded, and each name is a measured failure rather than a precaution:
#   deep-h-1/-2/-3  broken GPUs (nvidia-smi itself cannot get a device handle,
#                   while slurm happily allocates gres:gpu:1 -- three P009
#                   jobs ran to completion on CPU before this was noticed)
#   deep-chungus-7  **GPU 7 is broken** (AGENTS.md's 已知坏卡, 2026-10-07 on):
#                   nvidia-smi reports a healthy, empty A100 80GB and both the
#                   visibility and the memory check pass, but the CUDA runtime
#                   fails to initialise in-process.  P006 measured 12 ns/day
#                   against a sibling's 173 on the same node, exit code 0, log
#                   clean; and in one 54-task array, 35 tasks were rejected by
#                   the startup check and every one of them was on that card.
#                   A card that fails or crawls stays *idle on slurm's books*,
#                   so the scheduler keeps feeding it -- a bad card pumps an
#                   array dry.  --gres cannot name a device, so the node goes.
#                   Our preflight's third safeguard allocates and multiplies,
#                   which catches it; excluding the node saves the queue slot.
#   deep-chungus-8, deep-gpu-10/-11  DOWN+NOT_RESPONDING
#   deep-chungus-6  IDLE with no GPU at all
#   deep-chungus-3/-4/-5  our own P006 GROMACS array has their CPUs at load
#                   38-74, and load tracks step time cleanly (4-6.5 -> 6-7.7
#                   s/it, 33 -> 14.1)
# 排除列表，**每一条带日期和判据**（2026-10-09 实测复核）。
#   deep-h-1/-2/-3   三台 h100 全部 IDLE+DRAIN。h-2 的 GPU 连 nvidia-smi 都拿不到句柄；
#                    h-3 的卡是好的（7 张 H100 80GB 可见、/dev/nvidia* 齐全、驱动
#                    610.57.04），但驱动 API 直接 `cuInit(0)` 返回 999，所以也用不了。
#                    **注意：AGENTS.md 旧版用「ldconfig 里没有 libcuda.so.1」当判据，那是错的**
#                    —— h-3 上它就在 /usr/lib/x86_64-linux-gnu/，只是不在 ldconfig 缓存里。
#                    复查判据：在上面跑 `cuInit` 返回 0 就可以把它们删掉。
#   deep-chungus-6   `Gres=(null)`，**真的一张卡都没有**（上面那些作业是纯 CPU 的）。
#   deep-chungus-7   GPU 7 坏：nvidia-smi 把它报成健康、显存全空，而 CUDA runtime 在
#                    进程内起不来。--gres 不能指定设备号，只能排整个节点。
#   deep-chungus-8   IDLE+DRAIN。它 10-09 凌晨从 DOWN 恢复过，我们在上面跑过两条臂，
#                    随后又被 drain。**所以这一条是会变的，提交前值得瞄一眼。**
#   deep-gpu-10/-11  DOWN+NOT_RESPONDING（2080 Ti / V100，本来也不适配 cu126 的构建）。
#
# **删掉的（原因已过期，2026-10-09）**：deep-chungus-3/-4/-5。原注释写「我们自己的 P006
# GROMACS array 把它们的 CPU 压到 load 38-74」—— 那是 10-08 的瞬时状态。10-09 复核：
# chungus-4 九张卡**只占两张、空七张**，load 20。**一条按当时负载写死的排除，
# 在集群上会一直挡着一个早就空出来的节点。**负载是用 `--nodelist` 挑的理由，不是 `--exclude` 的。
#SBATCH --exclude=deep-h-1,deep-h-2,deep-h-3,deep-chungus-6,deep-chungus-7,deep-chungus-8,deep-gpu-10,deep-gpu-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v3-diag}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
CHECKPOINT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt"

ARM="${ARM:?set ARM to the arm name, e.g. random-1gpu}"
WP_MODE="${WP_MODE:?set WP_MODE to none, zero, random, pretrained or oracle}"
RUN_NAME="${RUN_NAME:-p010_${ARM}}"
MAX_STEPS="${MAX_STEPS:-2000}"
SEED="${SEED:-42}"
# Every 500 rather than every 1000: the arms are 2000 steps on a preemptible
# partition, so 1000 would mean losing up to half an arm to one preemption.
# Four files of ~3.1 GB per arm, ~12 GB an arm; /mnt/xfs is at 97%, which is
# why this is not smaller still.
# 250, not 500.  SIGMA_GRID_INTERVAL is also 500, so with both at 500 the
# first readout lands at step 499 and the first checkpoint at 500 -- the whole
# stretch from launch to the first useful number had no resume point.  On
# `background`, which preempts with no grace period, that stretch is both the
# most valuable and the only unresumable one: two oracle-v2 arms were preempted
# at steps 149 and 249 and restarted from zero.  Two extra 3.1 GB checkpoints
# per 1000-step arm, against 2.5 TB free.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
# Eight rounds over 2000 steps.  oracle should show within 500 (section 6,
# item 2) and gamma's half-life is 1156-2115 steps, so the curve needs points
# at that scale rather than at P009's 500.
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_WINDOWS="${EVAL_WINDOWS:-256}"
# D5: on in every arm here, passed explicitly rather than changed in the code.
# The config default stays false so that merging the tools back into v3 cannot
# alter P009's protocol (D11).
SKIP_MINI_ROLLOUT="${SKIP_MINI_ROLLOUT:-true}"
# D1: the per-sigma file beside the per-window one.
SIGMA_GRID="${SIGMA_GRID:-true}"
# How often, and over how many windows.  Both default to "every round, all of
# them", which is what the config defaults to; they are here because the grid
# costs 3.28 s a window (measured, artifacts/reports/P010/step_budget.json) and
# at 256 windows every 250 steps that is 1.9 GPU-hours against the arm's own
# 2.5.  The held-out rounds stay at EVAL_INTERVAL whatever this says -- the
# oracle gate reads the first 500 steps and arm B the 500 after its release,
# and those are the rounds that cadence is for.
#
# 500 is D13, decided 2026-10-08: four grid rounds an arm instead of eight,
# 40 GPU-hours instead of 47 over the ten arms.  The config default stays 0
# (= every round), so D11's "the defaults are the protocol" is untouched and
# this is P010's submission choice, written where the submission is.
#
# What is NOT reduced is EVAL_INTERVAL: the per-window held-out rows stay at
# every 250 steps, because the oracle gate reads the first 500 steps and arm B
# reads the 500 after its release (section 6 items 2 and 3).  The grid tracks
# gamma and the band structure, which move on a scale of a thousand steps, so
# four points over 2000 steps is not undersampling them.
SIGMA_GRID_INTERVAL="${SIGMA_GRID_INTERVAL:-500}"
SIGMA_GRID_WINDOWS="${SIGMA_GRID_WINDOWS:-0}"

GAGU_ROOT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/processed/gagu_internal_loop_v0_1"
# The same split as p004_train.sbatch: r1-r3 train, r4 held out.  It has to be
# the same split, or these arms' held-out numbers are not on the same windows
# as the per-sigma scores of the P009 checkpoints.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-$({ ls -d ${GAGU_ROOT}/*_r[123] 2>/dev/null || true; } | xargs -r -n1 basename | paste -sd,)}"
HELD_OUT="${HELD_OUT:-$({ ls -d ${GAGU_ROOT}/*_r4 2>/dev/null || true; } | xargs -r -n1 basename | paste -sd,)}"
# Refuse an empty list rather than passing one on the command line.  The
# defaults above are `ls | xargs | paste` pipelines, so a renamed or missing
# GAGU_ROOT makes them the empty string -- the same failure that cost job
# 2149587 (an empty value does not merely vanish: parse_sys_args builds
# `f"{k} {v} "` and parse_configs splits it, so the key is left bare and every
# later --key/value pair shifts by one), but reached from a missing dataset
# rather than from a hardcoded "".  check_p010's static scan cannot see this
# one, because there is no "" written anywhere.
#
# `:?` because it fires on empty *and* unset, at the point the value is used,
# without anyone having to remember to scan for it (the idiom is the P006
# session's).  Printing the count below is not enough: it reports and
# continues.
: "${TRAIN_SAMPLES:?no training trajectories matched under ${GAGU_ROOT} -- check the path, do not pass an empty list}"
: "${HELD_OUT:?no held-out trajectories matched under ${GAGU_ROOT} -- without them evaluate() scores nothing and the arm would train for its whole length and produce no rows}"
RESUME_DIR="${RESUME_DIR:-${WORKSPACE}/runs/p010/${RUN_NAME}_resume}"

mkdir -p "${WORKSPACE}/runs/slurm" "${WORKSPACE}/runs/p010" "${RESUME_DIR}"
cd "${WORKSPACE}"

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
echo "ARM=${ARM} TREE=${TREE} WP_MODE=${WP_MODE} SEED=${SEED} MAX_STEPS=${MAX_STEPS}"
echo "skip_mini_rollout=${SKIP_MINI_ROLLOUT} sigma_grid=${SIGMA_GRID}"
echo "sigma_grid_interval=${SIGMA_GRID_INTERVAL} (0 = every eval round) sigma_grid_windows=${SIGMA_GRID_WINDOWS} (0 = all)"
echo "TRAIN: $(echo "${TRAIN_SAMPLES}" | tr ',' '\n' | grep -c .) trajectories; HELD_OUT: $(echo "${HELD_OUT}" | tr ',' '\n' | grep -c .)"
echo "RESUME_DIR=${RESUME_DIR} (exists: $([ -e "${RESUME_DIR}/latest.pt" ] && echo yes || echo no))"
NGPU_WANT=1
# shellcheck source=./p010_preflight.sh
source "${TREE}/kineidos/slurm/p010_preflight.sh"

# env.lock is written by the trainer into the run directory it creates, and it
# records which tree the import resolved to and which tags point at its HEAD --
# the check that an arm ran on p010-base / p010-oracle / p010-gamma and not on
# whatever the worktree had drifted to (P010 section 3).

exec "${ENV}/bin/python" -u -m kineidos.train.main \
  --wp.mode "${WP_MODE}" \
  --load_checkpoint_path "${CHECKPOINT}" \
  --project p010 \
  --run_name "${RUN_NAME}" \
  --base_dir "${WORKSPACE}/runs/p010" \
  --max_steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --log_interval 25 \
  --eval_interval "${EVAL_INTERVAL}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --skip_mini_rollout "${SKIP_MINI_ROLLOUT}" \
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
  --kineidos.sigma_grid "${SIGMA_GRID}" \
  --kineidos.sigma_grid_arm "${ARM}" \
  --kineidos.sigma_grid_interval "${SIGMA_GRID_INTERVAL}" \
  --kineidos.sigma_grid_windows "${SIGMA_GRID_WINDOWS}" \
  "$@"
