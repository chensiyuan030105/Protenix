#!/usr/bin/env bash
# P010 D1 / D7: the per-sigma held-out reading, off a checkpoint, on a GPU.
#
# One job for several arms, run one after another in the same allocation.  Four
# arms at ~20 minutes each fit in one queue slot, and right now a queue slot is
# the scarce thing: every A100 on the cluster is allocated (2026-10-08 ~10:00),
# so four separate one-GPU jobs would wait four times rather than once.
#
# Each arm is a separate python process, because `none` builds a different
# architecture from the other three and the config that decides that is read in
# a constructor.
#
#   # D7: the four P009 arms at their step-3000 checkpoint (the file is 2999.pt)
#   sbatch --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G --nodes=1 \
#          --exclude=<busy nodes> \
#          --export=ALL,STEP=2999,ARMS='p009_random:random p009_zero:zero p009_zero_seed2:zero p009_none:none' \
#          repos/research/kineidos-v3-diag/kineidos/slurm/p010_sigma_grid.sbatch
#
#   # Section 4's acceptance: none == zero at step 0, and a repeat run of one
#   # arm is bit-identical to the first.
#   --export=ALL,STEP=0,ARMS='step0_none:none step0_zero:zero',CKPT=pretrained
#   --export=ALL,STEP=2999,ARMS='p009_zero:zero',SUFFIX=_repeat
#SBATCH --job-name=p010-sigma
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
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v3-diag}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
PRETRAINED="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt"

# "<arm name>:<wp.mode>" per entry.  The arm name is both the row label and the
# run directory prefix the checkpoint is found under, so it is not cosmetic.
ARMS="${ARMS:?set ARMS to a space separated list of <arm>:<wp.mode>}"
STEP="${STEP:?set STEP to the checkpoint step, e.g. 2999, or 0 with CKPT=pretrained}"
RUNS="${RUNS:-${WORKSPACE}/runs/p009}"
OUT="${OUT:-${WORKSPACE}/runs/p010/sigma_grid}"
SUFFIX="${SUFFIX:-}"
EVAL_WINDOWS="${EVAL_WINDOWS:-256}"
SIGMA_NOISE="${SIGMA_NOISE:-4}"

GAGU_ROOT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/processed/gagu_internal_loop_v0_1"
# r4 is held out, r1-r3 train -- the same split p004_train.sbatch makes, and it
# has to be the same split or the windows are not the ones the arms were scored
# on.  train_samples is named too, because init_data refuses an empty list;
# nothing here trains, so one trajectory is enough to satisfy it.
HELD_OUT="${HELD_OUT:-$({ ls -d ${GAGU_ROOT}/*_r4 2>/dev/null || true; } | xargs -r -n1 basename | paste -sd,)}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-gagu_100mM_K_agaguu_startI_r1}"
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

mkdir -p "${WORKSPACE}/runs/slurm" "${OUT}" "${WORKSPACE}/runs/p010/scratch"
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
# Both halves of torch's determinism story, and both have to be here rather
# than in python: cuBLAS reads its workspace configuration when the first
# handle is created, which is before any line of ours runs.
export KINEIDOS_DETERMINISTIC="${KINEIDOS_DETERMINISTIC:-0}"
if [ "${KINEIDOS_DETERMINISTIC}" = "1" ]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname)"
echo "TREE=${TREE} STEP=${STEP} OUT=${OUT} ARMS=${ARMS}"
echo "HELD_OUT: $(echo "${HELD_OUT}" | tr ',' '\n' | grep -c .) trajectories (r4)"
NGPU_WANT=1
# shellcheck source=./p010_preflight.sh
source "${TREE}/kineidos/slurm/p010_preflight.sh"

newest_run () {
  # The newest run directory for one arm prefix.  Anchored on the timestamp
  # init_basics appends, for read_heldout.py's reason: a plain prefix match
  # makes p009_zero swallow p009_zero_seed2, which would score the noise-floor
  # arm under the baseline's name.
  # `|| true` is load-bearing.  `ls` exits 2 when the glob matches nothing,
  # `2>/dev/null` hides the message but not the status, and `pipefail` hands
  # that 2 to the whole pipeline -- so `RUN_DIR="$(newest_run ...)"` killed the
  # script under `set -e` with status 2 and no output at all, before the
  # friendly "no run directory for ..." two lines below could ever print.
  # That is what job 2150100 died of, and the cause was only a wrong arm
  # prefix.  Third time this shape has cost a job (the others: `xargs -n1
  # basename` on empty input, and a non-matching `ls` glob in the submit
  # script).  **Under `set -euo pipefail`, "fail quietly then print something
  # helpful" never reaches the helpful part.**
  { ls -d "${RUNS}"/"$1"_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] \
      2>/dev/null || true; } | sort | tail -1
}

FAILED=""
for entry in ${ARMS}; do
  ARM="${entry%%:*}"
  MODE="${entry##*:}"
  if [ "${CKPT:-}" = "pretrained" ]; then
    TARGET="pretrained"
  else
    RUN_DIR="$(newest_run "${ARM}")"
    [ -n "${RUN_DIR}" ] || { echo "no run directory for ${ARM} under ${RUNS}" >&2; exit 1; }
    TARGET="${RUN_DIR}/checkpoints/${STEP}.pt"
    [ -f "${TARGET}" ] || { echo "no checkpoint ${TARGET}" >&2; exit 1; }
  fi
  # The resume path, which `background` makes mandatory and this job did not
  # have: on 2026-10-08 a single tier-10 job on deep-chungus-10 preempted four
  # of ours at once (PreemptExemptTime=0, no grace), and a requeue restarted
  # the whole sweep from the first arm.  An arm's rows are worth ~14 minutes
  # and the file is written once at the end, so finished arms should be skipped.
  #
  # The marker, not the jsonl's existence: a preemption mid-arm leaves a
  # partial file behind, and "the file is there" would skip it and leave a
  # truncated arm in the readout looking like a complete one.  The marker is
  # written only after python exits 0.
  # The marker carries the commit that wrote it, not just the fact that
  # something did.  Same reason as the resume gate: an arm scored before a
  # change to score_sigma_grid and skipped after it would put rows from two
  # code versions in one table, with every gate green -- the file is complete,
  # the row count is right, the windows pair.  The P006 session's input
  # fingerprint gate, applied here.
  TREE_SHA=$(git -C "${TREE}" rev-parse HEAD 2>/dev/null || echo unknown)
  TREE_DIRTY=$([ -n "$(git -C "${TREE}" status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)
  FP="${TREE_SHA} ${TREE_DIRTY}"
  DONE="${OUT}/.done_${ARM}${SUFFIX}_step${STEP}"
  if [ -f "${DONE}" ] && [ "${FORCE:-0}" != "1" ]; then
    WAS=$(cat "${DONE}" 2>/dev/null || echo "")
    if [ "${WAS}" = "${FP}" ] && [ "${TREE_DIRTY}" = "clean" ]; then
      echo ""
      echo "=== ${ARM}: already done at ${TREE_SHA:0:12}; FORCE=1 to rescore ==="
      continue
    fi
    echo ""
    echo "=== ${ARM}: rescoring -- the marker says '${WAS}' and this tree is"
    echo "    '${FP}'. A dirty tree never counts as done, because the commit"
    echo "    does not identify the code. ==="
  fi
  echo ""
  echo "=== ${ARM} (wp.mode=${MODE}) <- ${TARGET} ==="
  # A scratch base_dir per arm: the trainer creates a run directory and writes
  # an env.lock into it, which is exactly what AGENTS.md asks of a run, and it
  # records wp.mode, both commits and the two hand-placed libraries' sha256.
  if "${ENV}/bin/python" -u -m kineidos.score_sigma_grid \
      --model_name protenix_base_default_v1.0.0 \
      --wp.mode "${MODE}" \
      --load_checkpoint_path "${PRETRAINED}" \
      --project p010 \
      --run_name "sigma_grid_${ARM}${SUFFIX}" \
      --base_dir "${WORKSPACE}/runs/p010/scratch" \
      --max_steps 1 \
      --eval_interval -1 \
      --checkpoint_interval -1 \
      --use_wandb false \
      --skip_confidence true \
      --skip_distogram true \
      --loss.weight.alpha_confidence 0.0 \
      --loss.weight.alpha_distogram 0.0 \
      --kineidos.train_samples "${TRAIN_SAMPLES}" \
      --kineidos.num_workers 0 \
      --kineidos.held_out_samples "${HELD_OUT}" \
      --kineidos.eval_windows "${EVAL_WINDOWS}" \
      --kineidos.score_checkpoint "${TARGET}" \
      --kineidos.sigma_grid_arm "${ARM}${SUFFIX}" \
      --kineidos.sigma_grid_noise "${SIGMA_NOISE}" \
      --kineidos.sigma_grid_out "${OUT}" \
      "$@"; then
    printf '%s\n' "${FP}" > "${DONE}"
    echo "=== ${ARM}: ok (${TREE_SHA:0:12} ${TREE_DIRTY}) ==="
  else
    # One arm failing must not throw away the arms that already scored: the
    # files are written per arm and the comparison can be made with what came
    # back.  The exit code still reports it.
    echo "=== ${ARM}: FAILED ===" >&2
    FAILED="${FAILED} ${ARM}"
  fi
done

if [ -n "${FAILED}" ]; then
  echo "failed arms:${FAILED}" >&2
  exit 1
fi
echo ""
echo "all arms scored -> ${OUT}"
ls -la "${OUT}"
