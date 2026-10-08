#!/usr/bin/env bash
# P007: the GPU half of the benchmark -- sampling from a checkpoint.
#
# Three guards before the work starts, all of them because of things that have
# already happened on this cluster (AGENTS.md, "Slurm：账面状态会骗人"):
#
#   1. torch must see the GPU.  slurm allocated gres:gpu:1 on deep-h-2, whose
#      driver cannot talk to its devices; torch reported cuda=False and three
#      training jobs ran to completion on CPU before anyone noticed.
#   2. the allocated device must still have its memory.  gres counts devices,
#      not bytes: on deep-chungus-3 our device already carried another user's
#      35.18 GiB and we OOMed at 3.9 GiB of our own.  nvidia-smi ignores
#      CUDA_VISIBLE_DEVICES, so the index has to be passed explicitly -- and no
#      `head` in the pipeline, because under `pipefail` the SIGPIPE kills the job.
#   3. the CUDA runtime must actually initialise in this process.  A card can be
#      visible, empty and still fail to start a context -- that is what cost
#      p006_3_md a replica at 12 ns/day instead of 173 with exit code 0.  A real
#      allocation and a real kernel launch is the only honest test.
#
#   sbatch --gres=gpu:a100:1 --nodelist=deep-chungus-5 \
#     --export=ALL,BENCH_ARGS="--arm random --checkpoint ... --out ... --n-step 20" \
#     kineidos/bench/slurm/p007_gpu.sbatch
#
# Partition: `background` is fine for the dry run and for the N_step scan (if
# preempted, rerun; nothing accumulates).  The scored runs of P007.2 and P007.4
# must not go there -- a requeue reuses the job id and overwrites the log.
#SBATCH --job-name=p007-gpu
#SBATCH --partition=background
#SBATCH --qos=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=deep-chungus-8,deep-h-1,deep-h-2,deep-h-3,deep-gpu-10,deep-gpu-11
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
BENCH_MODULE="${BENCH_MODULE:-kineidos.bench.oneshot}"
BENCH_ARGS="${BENCH_ARGS:?set BENCH_ARGS to the arguments for BENCH_MODULE}"
# Fraction of the device another process may already hold before we refuse.
MAX_FOREIGN_FRACTION="${MAX_FOREIGN_FRACTION:-0.5}"

cd "${WORKSPACE}"
export PYTHONPATH="repos/research/kineidos-v3:repos/research/wp-v2"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-cuequivariance}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname) $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MODULE=${BENCH_MODULE} ARGS=${BENCH_ARGS}"

# --- guard 2: is the allocated device's memory actually free?
DEV="${CUDA_VISIBLE_DEVICES:-0}"
for idx in $(echo "${DEV}" | tr ',' ' '); do
  LINE=$(nvidia-smi -i "${idx}" --query-gpu=memory.used,memory.total \
         --format=csv,noheader,nounits) || {
    echo "refusing to start: nvidia-smi cannot query GPU ${idx}" >&2; exit 1; }
  USED=$(echo "${LINE}" | awk -F', *' '{print $1}')
  TOTAL=$(echo "${LINE}" | awk -F', *' '{print $2}')
  echo "  gpu ${idx}: ${USED} / ${TOTAL} MiB already in use"
  awk -v u="${USED}" -v t="${TOTAL}" -v f="${MAX_FOREIGN_FRACTION}" \
      'BEGIN { exit !(t > 0 && u / t < f) }' || {
    echo "refusing to start: GPU ${idx} already has ${USED} of ${TOTAL} MiB in" \
         "use by someone else. gres counts devices, not memory (AGENTS.md)." >&2
    exit 1; }
done

# --- guards 1 and 3: torch sees it, and the runtime really starts
"${ENV}/bin/python" -u - <<'PYEOF'
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "refusing to start: torch sees no GPU. slurm can allocate one on a node "
        "whose driver is broken (deep-h-2 did exactly that)."
    )
p = torch.cuda.get_device_properties(0)
print(f"  gpu0 {p.name} {p.total_memory / 2**30:.0f} GiB")
# A real allocation and a real kernel launch: visibility is not initialisation.
a = torch.randn(512, 512, device="cuda")
assert torch.isfinite(a @ a).all()
torch.cuda.synchronize()
print(f"  runtime ok, {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated")
PYEOF

exec "${ENV}/bin/python" -u -m "${BENCH_MODULE}" ${BENCH_ARGS}
