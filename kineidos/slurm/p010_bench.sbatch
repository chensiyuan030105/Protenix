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
