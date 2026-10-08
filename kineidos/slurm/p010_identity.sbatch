#!/usr/bin/env bash
# P010 section 6, item 5b: the two test branches must be p010-base with their
# switches off.
#
# Nine arms share one set of baselines (D10), and that is only allowed if the
# oracle and gamma branches reduce to p010-base exactly when their switches are
# off.  "Exactly" means bit-identical, not "close": each branch adds code to a
# path every arm runs through, and a change that shifts the RNG stream by one
# draw -- a projection built from the global generator, an augmentation call
# skipped -- would make the arms unpaired while leaving every number plausible.
# That failure has no other detector.
#
# All three trees in one job, one after another in the same allocation, then
# compared.  One allocation rather than three because the comparison is the
# point and three jobs could land on three different nodes; GPU arithmetic is
# deterministic on one device for one code path, and across devices it is a
# claim nobody should have to make.
#
#   sbatch --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G --nodes=1 \
#          --exclude=<busy> \
#          repos/research/kineidos-v3-diag/kineidos/slurm/p010_identity.sbatch
#
# Two steps, not one, and the reason is in kineidos/train/check_trainer.py:
# af3_lr_scheduler warms up, so the *first* optimizer step runs at lr = 0 and
# moves nothing.  A one-step run would compare three models that had not
# changed, which passes without testing the gradient path at all.
#SBATCH --job-name=p010-identity
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
CHECKPOINT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt"

BASE_TREE="${BASE_TREE:-repos/research/kineidos-v3-diag}"
TREES="${TREES:-repos/research/kineidos-v3-diag repos/research/kineidos-v3-diag-oracle repos/research/kineidos-v3-diag-gamma}"
WP_MODE="${WP_MODE:-random}"
MAX_STEPS="${MAX_STEPS:-2}"
EVAL_WINDOWS="${EVAL_WINDOWS:-64}"
SIGMA_NOISE="${SIGMA_NOISE:-4}"
SEED="${SEED:-42}"
OUT="${OUT:-${WORKSPACE}/runs/p010/identity}"
REPORT="${REPORT:-${WORKSPACE}/artifacts/reports/P010/identity_check.md}"

GAGU_ROOT="/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/processed/gagu_internal_loop_v0_1"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-$(ls -d ${GAGU_ROOT}/*_r[123] 2>/dev/null | xargs -n1 basename | paste -sd,)}"
HELD_OUT="${HELD_OUT:-$(ls -d ${GAGU_ROOT}/*_r4 2>/dev/null | xargs -n1 basename | paste -sd,)}"

mkdir -p "${WORKSPACE}/runs/slurm" "${OUT}" "$(dirname "${REPORT}")"
cd "${WORKSPACE}"

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
echo "TREES=${TREES}"
echo "BASE_TREE=${BASE_TREE} WP_MODE=${WP_MODE} MAX_STEPS=${MAX_STEPS}"
NGPU_WANT=1
# shellcheck source=./p010_preflight.sh
source "${BASE_TREE}/kineidos/slurm/p010_preflight.sh"

for TREE in ${TREES}; do
  LABEL=$(basename "${TREE}")
  echo ""
  echo "=== ${LABEL} ==="
  # A per-tree base_dir, and a resume_dir that must NOT be shared: resuming
  # one tree's checkpoint into another is exactly the confound this job is
  # looking for, and leaving resume_dir empty is how that is ruled out.
  export PYTHONPATH="${TREE}:repos/research/wp-v2"
  "${ENV}/bin/python" -u -m kineidos.train.main \
    --wp.mode "${WP_MODE}" \
    --load_checkpoint_path "${CHECKPOINT}" \
    --project p010 \
    --run_name "identity_${LABEL}" \
    --base_dir "${OUT}" \
    --max_steps "${MAX_STEPS}" \
    --seed "${SEED}" \
    --log_interval 1 \
    --eval_interval "${MAX_STEPS}" \
    --checkpoint_interval -1 \
    --skip_mini_rollout true \
    --use_wandb false \
    --skip_confidence true \
    --skip_distogram true \
    --loss.weight.alpha_confidence 0.0 \
    --loss.weight.alpha_distogram 0.0 \
    --kineidos.train_samples "${TRAIN_SAMPLES}" \
    --kineidos.num_workers 0 \
    --kineidos.resume_dir "" \
    --kineidos.held_out_samples "${HELD_OUT}" \
    --kineidos.eval_windows "${EVAL_WINDOWS}" \
    --kineidos.sigma_grid true \
    --kineidos.sigma_grid_arm "identity_${LABEL}" \
    --kineidos.sigma_grid_noise "${SIGMA_NOISE}" \
    "$@"
  echo "=== ${LABEL}: done ==="
done

# ---------------------------------------------------------------- compare
export PYTHONPATH="${BASE_TREE}:repos/research/wp-v2"
BASE_LABEL=$(basename "${BASE_TREE}")

newest () { ls -d "${OUT}"/identity_"$1"_[0-9]*_[0-9]* 2>/dev/null | sort | tail -1; }

BASE_DIR=$(newest "${BASE_LABEL}")
[ -n "${BASE_DIR}" ] || { echo "no run directory for ${BASE_LABEL}" >&2; exit 1; }

{
  echo "# P010 §6 第 5 条 (b)：恒等检查"
  echo
  echo "作业 \`${SLURM_JOB_ID:-none}\`，节点 \`$(hostname)\`，$(date -u '+%Y-%m-%d %H:%MZ')。"
  echo
  echo "问的是：两个测试分支在各自的开关关闭时，是不是 \`p010-base\` 本身。"
  echo "判据是**逐位**，不是「接近」——每个分支都在所有臂都要走的路径上加了代码，"
  echo "而一个把 RNG 流挪了一个抽样的改动（用全局 generator 造投影、跳过一次增广调用）"
  echo "会让九个臂不再配对，同时每个数字看起来都正常。这个失败没有别的探测器。"
  echo
  echo "两步而不是一步：af3_lr_scheduler 有 warmup，**第一个 optimizer step 的 lr = 0**，"
  echo "一步跑完三个模型都没动过，检查会通过而根本没测到梯度路径"
  echo "（见 \`kineidos/train/check_trainer.py\` 第 5 节）。"
  echo
  echo "| 树 | 运行目录 | 逐窗口 jsonl | 逐 σ jsonl | 训练 loss 行 |"
  echo "|---|---|---|---|---|"
} > "${REPORT}"

FAILED=""
for TREE in ${TREES}; do
  LABEL=$(basename "${TREE}")
  DIR=$(newest "${LABEL}")
  if [ -z "${DIR}" ]; then
    echo "| \`${LABEL}\` | — | 缺 | 缺 | 缺 |" >> "${REPORT}"
    FAILED="${FAILED} ${LABEL}(missing)"
    continue
  fi
  if [ "${LABEL}" = "${BASE_LABEL}" ]; then
    echo "| \`${LABEL}\` | \`$(basename "${DIR}")\` | 基准 | 基准 | 基准 |" >> "${REPORT}"
    continue
  fi
  HO="ok"; SG="ok"; LO="ok"
  for f in "${BASE_DIR}"/heldout/*.jsonl; do
    b=$(basename "$f")
    cmp -s "$f" "${DIR}/heldout/${b}" || HO="**差异**"
  done
  # The per-sigma files carry the arm name, so they differ by name and have to
  # be matched by step rather than by basename.
  for f in "${BASE_DIR}"/sigma_grid/*.jsonl; do
    tailpart="${f##*_step}"
    g=$(ls "${DIR}"/sigma_grid/*_step"${tailpart}" 2>/dev/null | head -1 || true)
    if [ -z "$g" ]; then
      SG="**缺**"
    else
      # Byte comparison would trip on the `arm` field alone, which is the tree
      # name by construction.  --identical reports which metric moved and by
      # how much, which is the distinction that matters here.
      "${ENV}/bin/python" -u -m kineidos.read_sigma_grid --identical "$f" "$g" \
        | tee -a "${REPORT}.log" | grep -q "PASS" || SG="**差异**"
    fi
  done
  # The training-side half.  The eval rows only exercise the forward pass; the
  # periodic metrics line is where the backward and the optimizer show up.
  a=$(grep -h "train metrics:" "${WORKSPACE}/runs/slurm/p010-identity-${SLURM_JOB_ID:-none}.err" 2>/dev/null | head -4 | md5sum | cut -c1-8)
  echo "| \`${LABEL}\` | \`$(basename "${DIR}")\` | ${HO} | ${SG} | ${a} |" >> "${REPORT}"
  case "${HO}${SG}" in *差异*|*缺*) FAILED="${FAILED} ${LABEL}";; esac
done

{
  echo
  if [ -n "${FAILED}" ]; then
    echo "**不通过：${FAILED}**。按 §6 第 5 条 (b)，该分支自己补跑"
    echo "\`random-1gpu\` / \`zero-1gpu\`，并在报告里标明；不许跨分支硬比。"
  else
    echo "**全部逐位一致。** 九个臂共用 \`p010-base\` 的基线臂（D10）。"
  fi
  echo
  echo "逐 σ 比较的完整输出在 \`$(basename "${REPORT}").log\`。"
} >> "${REPORT}"

cat "${REPORT}"
[ -z "${FAILED}" ]
