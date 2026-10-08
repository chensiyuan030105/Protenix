#!/usr/bin/env bash
# The three safeguards AGENTS.md asks every GPU job to carry, in one place.
#
# Sourced rather than copied, because the list has grown twice and the copies
# did not: AGENTS.md's "新作业都应照抄" is how p004_train.sbatch ended up with
# only the first of them.  Expects $ENV (the conda env prefix) to be set and
# $NGPU_WANT to say how many devices the job needs; exits non-zero on failure,
# so the caller must be running under `set -e` or check the return.
#
# Why each one exists -- all three are observed failures, not hypotheticals:
#
#  1. **The device must be visible.**  slurm will allocate gres:gpu:1 on a node
#     whose driver cannot talk to its GPUs (deep-h-2 does exactly that, and
#     nvidia-smi itself reports "Unable to determine the device handle").  torch
#     then reports cuda=False and nothing stops the job, which runs ~100x too
#     slow on CPU and leaves a run directory indistinguishable from a real one.
#     Three P009 jobs were lost that way.
#  2. **gres counts devices, not memory.**  On deep-chungus-3 the device slurm
#     gave us already held another user's 35.18 GiB, and our own 3.9 GiB
#     allocation OOMed.  nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the query
#     must pass -i explicitly or it reports somebody else's card; and no `head`
#     in the pipe, because SIGPIPE under `pipefail` kills the whole job.
#  3. **A visible device is not a working CUDA runtime.**  A GROMACS job on
#     2026-10-07 passed both checks above and still fell back to CPU because
#     the runtime failed to initialise inside that one process -- 12 ns/day
#     against 173 for its sibling, exit code 0.  The torch analogue of
#     GROMACS's `CUDA runtime: N/A` line is to allocate and multiply: context
#     creation and a kernel launch are what actually fail, and neither
#     is exercised by torch.cuda.is_available().

NGPU_WANT="${NGPU_WANT:-1}"

DEV="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES=${DEV}"
SMI=$(nvidia-smi -i "${DEV}" --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader,nounits 2>&1) || {
  echo "拒绝启动：nvidia-smi 失败，这个节点的 GPU 不可用：${SMI}" >&2; exit 1; }
echo "分到的卡：${SMI}"
N_SEEN=$(printf '%s\n' "${SMI}" | grep -c . || true)
[ "${N_SEEN}" -ge "${NGPU_WANT}" ] || {
  echo "拒绝启动：要 ${NGPU_WANT} 张卡，nvidia-smi 只看到 ${N_SEEN} 张" >&2; exit 1; }

# 保险 2，每一张分到的卡都查，不只第一张。
while IFS= read -r line; do
  [ -n "${line}" ] || continue
  IDX=$(printf '%s\n' "${line}" | awk -F', *' '{print $1}')
  USED=$(printf '%s\n' "${line}" | awk -F', *' '{print $3}')
  TOTAL=$(printf '%s\n' "${line}" | awk -F', *' '{print $4}')
  echo "GPU${IDX} 显存 ${USED}/${TOTAL} MiB 已被占用"
  if [ "${USED}" -gt $(( TOTAL / 2 )) ]; then
    echo "拒绝启动：GPU${IDX} 已被占掉一半以上显存（${USED}/${TOTAL} MiB）；" \
         "gres 账面只管设备数，不管显存。换节点重投。" >&2
    exit 1
  fi
done <<< "${SMI}"

# 保险 3：runtime 必须真的起来，而不只是 is_available() 说真。
"${ENV}/bin/python" -u - "${NGPU_WANT}" <<'PYEOF'
import sys
import torch

want = int(sys.argv[1])
have = torch.cuda.device_count()
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"devices={have}", flush=True)
if not torch.cuda.is_available() or have < want:
    raise SystemExit(
        f"refusing to start: asked for {want} GPU(s), torch sees {have}. "
        f"slurm can allocate a GPU on a node whose driver is broken "
        f"(deep-h-2 did exactly that); check nvidia-smi on this node."
    )
for i in range(want):
    p = torch.cuda.get_device_properties(i)
    # Allocate and launch, not merely query.  This is the line that would have
    # caught the 2026-10-07 GROMACS failure, where the device was visible and
    # the runtime still could not start inside the process.
    try:
        a = torch.randn(256, 256, device=f"cuda:{i}")
        got = float((a @ a).sum())
    except Exception as exc:                        # noqa: BLE001
        raise SystemExit(
            f"refusing to start: gpu{i} is visible but a 256x256 matmul on it "
            f"failed ({type(exc).__name__}: {exc}). That is a per-process CUDA "
            f"runtime failure, not a node-level one -- resubmit elsewhere "
            f"rather than deleting this check."
        ) from exc
    if got != got:                                  # NaN
        raise SystemExit(f"refusing to start: gpu{i} returned NaN for a matmul")
    print(f"  gpu{i} {p.name} {p.total_memory / 2**30:.0f} GiB  "
          f"matmul ok", flush=True)
    del a
    torch.cuda.empty_cache()
PYEOF
