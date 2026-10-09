#!/usr/bin/env bash
# P011 section 4: the symmetry probe, with its positive controls.
#
# D8 is the reason this is a gate rather than a report: "关掉等于基线" is silent
# about "打开能用", so every acceptance check in this plan carries a case that
# **must fail on the old code**.  Here that case is `wp-v2`: the same probe, the
# same window, the same seed, and a residual of order 1.  A probe that only ever
# prints small numbers cannot tell "invariant" from "accidentally not measuring
# anything" -- which is how an earlier version of this very probe reported 1e-13
# from an underdetermined least-squares fit and concluded "equivariant".
#
# Six runs.  The first submission (job 2150572) used four and got two of its
# own criteria wrong; both mistakes are recorded here because they are the
# reason the list looks like this.
#
#   **The positive control must have canonicalisation OFF.**  2150572 ran
#   `wp-v2 --align` as a control and it came out *invariant*, 3.4e-07 -- not
#   because wp-v2 is invariant but because `canonicalize_window` rotates the
#   window back to a fixed pose before WorldParticle sees it, so the probe was
#   handing both passes the same input.  That is P004 2.9.3's remedy working
#   exactly as designed, and it is also precisely why P010 concluded the
#   content still carried the pose: the symmetry was in the preparation, not in
#   `h`.  A control that cannot fail is not a control.
#
#   **The probe has to feed the velocity slot zeros.**  2150572 left the real
#   velocities in and wp-v3 came out at 7.5e-04 at stage 1 -- the stage that is
#   *only* the radial convolution and dense0, both invariant by construction.
#   The cause is not either of them: `molecular_feats` is
#   `[ones, inp_vel, other_feats]`, and the three velocity components are a
#   lab-frame vector that rotates with the window, so dense0_molecular cannot
#   be invariant whatever the convolution does.  The size matches, too -- the
#   velocities carry 2.6e-04 of that concat's weight against `ones` = 1.0
#   (P009 10.7), and rotating them changes their own contribution by order 1.
#   D2 sends zeros down that slot, so the velocity runs below are kept as a
#   *measurement of why D2 is necessary*, which is a stronger reason than the
#   two the plan gives (thin content, and nobody in the literature does it).
#
#   v3-gate       wp-v3, zero velocity, canonicalisation OFF, geometry radius.
#                 **This is the registered gate.**  Nothing is helping it: no
#                 canonicalisation, no velocity, a generic rotation.
#   v3-exact      the same with `--exact-rotation`.  A signed permutation of
#                 the axes is a proper rotation whose matrix is 0s and 1s, so
#                 `pos @ R.T` costs no float32 rounding.  This is what
#                 separates "the architecture is asymmetric" from "float32
#                 stored the rotated coordinates to 1e-7 and the RBF slope
#                 amplified it": the first survives, the second disappears.
#   v3-bridge     wp-v3 at particle_radius 0.0778, the radius the bridge
#                 passes.  An invariance that held at only one neighbourhood
#                 size would be a coincidence.  **Canonicalisation off**, and
#                 that was the same mistake a second time: run 2150578 had
#                 `--align` here and the run reported 0.000e+00 at every stage,
#                 because canonicalisation maps the plain and the rotated
#                 window to the same pose and the two passes then have
#                 bit-identical input.  A perfect score from a probe that is
#                 comparing a tensor with itself.  Any run with `--align` is
#                 mute about rotation invariance; it tests only the separate
#                 claim that canonicalising does not change `h`, which is what
#                 `canon resid` reports in every run anyway.
#   v3-vel        wp-v3 with the real velocities.  Expected to FAIL the
#                 tolerance; that is the D2 measurement described above.
#   v2-gate       wp-v2, zero velocity, canonicalisation OFF.  **The positive
#                 control: must be orders of magnitude worse.**
#   v2-canon      wp-v2 with canonicalisation ON, to put the 3.4e-07 above on
#                 the record as what the old remedy did and did not buy.
#
# **No GPU, and no preflight.**  kineidos.probe_h_symmetry never moves a tensor
# off the CPU, so a gres request would hold an A100 idle while the cluster is at
# 7 free cards across 16 nodes; and p010_preflight.sh exists to stop a *GPU* job
# from silently running on the CPU, which is not a failure mode a CPU job has.
# This is the one place in P011 where that file is deliberately not sourced.
#
#   sbatch --cpus-per-task=8 --mem=32G \
#          repos/research/kineidos-v4/kineidos/slurm/p011_symmetry.sbatch
#SBATCH --job-name=p011-symmetry
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --requeue
#SBATCH --open-mode=append
# deep-chungus-7's GPU 7 is broken (AGENTS.md 已知坏卡) and -8/-gpu-10/-gpu-11
# and -h-1..3 are DOWN or DRAIN.  A CPU-only job cannot be hurt by a broken
# GPU, but a DOWN node will not run it at all, and excluding chungus-7 costs
# nothing -- "知道有坏卡" and "每次提交都排除它" are two different things, and
# AGENTS.md records what the second one costs when it is left to memory.
#SBATCH --exclude=deep-h-1,deep-h-2,deep-h-3,deep-chungus-7,deep-chungus-8,deep-gpu-10,deep-gpu-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
TREE="${TREE:-repos/research/kineidos-v4}"
ENV="/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm"
OUTDIR="${OUTDIR:-${WORKSPACE}/artifacts/reports/P011}"
FRAMES="${FRAMES:-4}"
# Non-zero on purpose: with window_from=0 the window's anchor *is* the
# reference conformer, the unrotated alignment degenerates to the identity, and
# the canonicalisation half of the test stops exercising the case training
# sees.  The probe's own docstring says so.
WINDOW_FROM="${WINDOW_FROM:-2000}"
BRIDGE_RADIUS="${BRIDGE_RADIUS:-0.0778}"

mkdir -p "${WORKSPACE}/runs/slurm" "${OUTDIR}"
cd "${WORKSPACE}"

# AGENTS.md's four, written in the command.  LD_LIBRARY_PATH is *prepended*,
# not replaced: the env's lib/ holds the two hand-placed libEGL/libGLdispatch
# that open3d_torch_ops.so needs, and replacing the path would hide the system
# libraries behind them.
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
# **One thread, and deterministic kernels.**  Not a precaution: with 8 threads
# the probe's own run-to-run spread was larger than the rotation residual it
# gates (jobs 2150572/2150574 reported the canonicalisation residual -- which
# does not involve a rotation -- as 1.97e-05 and 3.54e-04 for two runs
# differing only in the rotation matrix).  `RadialMessagePassing` ends in an
# `index_add`, a parallel scatter with no fixed summation order, and the
# features at random initialisation are nearly constant across atoms, so every
# RMSNorm divides the constant out and amplifies whatever per-atom difference
# is left.  The probe measures and reports the floor either way (section 3b);
# this makes the floor small instead of merely known.  It costs nothing -- the
# whole job is minutes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KINEIDOS_DETERMINISTIC="${KINEIDOS_DETERMINISTIC:-1}"
# `:-`, not `:`.  `${X::4096:8}` is substring expansion and expands to the
# empty string -- valid bash, silently wrong, and the determinism helper
# would then have printed "CUBLAS_WORKSPACE_CONFIG is unset" into a log
# nobody was reading yet.  Caught before the job ran; AGENTS.md 16.1 keeps
# a list of this family.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "JOB_ID=${SLURM_JOB_ID:-none} HOST=$(hostname) DATE=$(date -Is)"
echo "TREE=${TREE} OUTDIR=${OUTDIR} FRAMES=${FRAMES} WINDOW_FROM=${WINDOW_FROM}"
echo "wp-v3 HEAD=$(git -C repos/research/wp-v3 rev-parse --short HEAD)"
echo "wp-v2 HEAD=$(git -C repos/research/wp-v2 rev-parse --short HEAD)"
echo "${TREE} HEAD=$(git -C "${TREE}" rev-parse --short HEAD)"

# `run <label> <wp tree> [extra probe args...]`.
# The probe exits non-zero when its own shape/order checks fail, which is a
# real failure; but a *rotation verdict* of NEITHER is the expected answer for
# wp-v2, so the exit status is recorded rather than allowed to kill the script.
# `|| rc=$?` and not `set +e`, so one failing run cannot make the rest silently
# skip their checks.
run () {
  local label="$1" wptree="$2"; shift 2
  local rc=0
  echo ""
  echo "=================================================================="
  echo "=== ${label}  (PYTHONPATH=${TREE}:${wptree})"
  echo "=================================================================="
  PYTHONPATH="${TREE}:${wptree}" "${ENV}/bin/python" -u \
    -m kineidos.probe_h_symmetry \
    --frames "${FRAMES}" \
    --window-from "${WINDOW_FROM}" \
    --report "${OUTDIR}/symmetry_${label}.json" \
    "$@" || rc=$?
  echo "--- ${label} exit=${rc}"
  RESULTS="${RESULTS}${label}=${rc} "
}

RESULTS=""
run v3-gate   repos/research/wp-v3 --zero-velocity
run v3-exact  repos/research/wp-v3 --zero-velocity --exact-rotation
run v3-bridge repos/research/wp-v3 --zero-velocity \
                                   --particle-radius "${BRIDGE_RADIUS}"
run v3-vel    repos/research/wp-v3
run v2-gate   repos/research/wp-v2 --zero-velocity
run v2-canon  repos/research/wp-v2 --zero-velocity --align

echo ""
echo "=== exit codes: ${RESULTS}"

# The gate, read off the four json files rather than off the log, so that it is
# the same numbers the report will quote.  Registered before the run (D8), and
# it names which parts it depends on -- P010 section 12's lesson is that a
# criterion outlives the components it was written against and then means
# something else.
#
# Depends on: wp-v3 @ the commit printed above (models/invariant.py and the
# three call sites), kineidos/probe_h_symmetry.py @ the tree printed above,
# float32 throughout, no autocast, eval mode.  Change any of those and re-read
# this gate before quoting it.
"${ENV}/bin/python" -u - "${OUTDIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
TOL = 1e-4

def load(label):
    p = outdir / f"symmetry_{label}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

rows = {k: load(k) for k in ("v3-gate", "v3-exact", "v3-bridge", "v3-vel",
                             "v2-gate", "v2-canon")}

# The registered tolerance.  1e-4 for a generic rotation, as plan section 8
# item 1 says; 1e-5 for the exact rotation, where no input rounding enters and
# the only remaining source of difference is the order in which neighbour
# contributions are summed.
TOL = 1e-4
TOL_EXACT = 1e-5
# "Orders of magnitude worse" for the control.  Not 1.0: P004's 0.999 is the
# residual of the best *linear map*, a different quantity, and the plain
# invariance residual measured for wp-v2 here is 6.7e-02.  1e-2 is 100x the
# tolerance and 6x below the measured value, so the control has room to move
# without either passing by accident or failing for being merely smaller than
# someone guessed.
TOL_CONTROL = 1e-2

print()
print("=== P011 section 4 gate ===")
print(f"{'run':<11} {'rot resid':>11} {'canon resid':>12} {'transl':>11} "
      f"{'neigh':>7}  verdict")
for k, r in rows.items():
    if r is None:
        print(f"{k:<11} {'MISSING':>11}")
        continue
    rot = r["stages"]["3_super_particle_decoder"]["invariance_residual"]
    print(f"{k:<11} {rot:>11.3e} {r['h_canonicalization_residual']:>12.3e} "
          f"{r['h_translation_invariance_residual']:>11.3e} "
          f"{r['mean_neighbours']:>7.1f}  {r['h_rotation_verdict']}")

print()
print("stage by stage (rotation residual), to show where any asymmetry enters:")
stages = ("1_local_feature_extractor", "2_slot_attention",
          "3_super_particle_decoder", "4_output_network")
print(f"{'run':<11} " + " ".join(f"{s.split('_', 1)[1][:13]:>14}" for s in stages))
for k, r in rows.items():
    if r is None:
        continue
    cells = []
    for st in stages:
        v = r["stages"].get(st, {}).get("invariance_residual")
        cells.append(f"{v:>14.3e}" if v is not None else f"{'-':>14}")
    print(f"{k:<11} " + " ".join(cells))

def rot_of(k):
    r = rows[k]
    return None if r is None else \
        r["stages"]["3_super_particle_decoder"]["invariance_residual"]

print()
print("is the residual at `h` an asymmetry, or ToMe's partition flipping?")
print(f"{'run':<11} {'stage1 (conv)':>14} {'h':>11} "
      f"{'super-particle eq':>18}  partition")
for k, r in rows.items():
    if r is None:
        continue
    s1v = r["stages"]["1_local_feature_extractor"]["invariance_residual"]
    print(f"{k:<11} {s1v:>14.3e} {rot_of(k):>11.3e} "
          f"{r['super_particle_equivariance_residual']:>18.3e}  "
          f"{'same' if r['tome_partition_same'] else 'CHANGED'}")

print()
print("noise floor (same input twice) and the ratio that matters:")
print(f"{'run':<11} {'rot resid':>11} {'floor':>11} {'ratio':>8}")
for k, r in rows.items():
    if r is None:
        continue
    print(f"{k:<11} {rot_of(k):>11.3e} {r['h_self_residual']:>11.3e} "
          f"{r['h_rotation_over_noise_floor']:>8.2f}x")

checks = {}
for k in ("v3-gate", "v3-bridge", "v3-exact"):
    r = rows[k]
    # Two ways to pass, and either is enough, because they are two ways of
    # saying the same thing.  Absolute: the residual is under the registered
    # 1e-4.  Relative: the residual is within 3x the probe's own run-to-run
    # floor, i.e. rotating the window changed `h` by no more than running the
    # identical input twice does -- which is the strongest statement a
    # measurement with that floor can support, and is why the floor is
    # measured rather than assumed.
    absolute = r is not None and rot_of(k) < TOL
    within_floor = r is not None and r["h_rotation_over_noise_floor"] < 3.0
    checks[f"{k}: rotation residual < {TOL:.0e} OR within 3x its noise floor"] = (
        absolute or within_floor)
    checks[f"{k}: canonicalisation agrees OR within 3x floor"] = (
        r is not None and (r["h_canonicalization_agrees"]
                           or r["h_canonicalization_residual"]
                           < 3.0 * r["h_self_residual"]))
    checks[f"{k}: translation invariant OR within 3x floor"] = (
        r is not None and (r["h_translation_invariant"]
                           or r["h_translation_invariance_residual"]
                           < 3.0 * r["h_self_residual"]))
# The control has to beat the floor by a wide margin or it is not a control.
r2 = rows["v2-gate"]
checks[f"v2-gate: POSITIVE CONTROL rotation residual > {TOL_CONTROL:.0e}"] = (
    rot_of("v2-gate") is not None and rot_of("v2-gate") > TOL_CONTROL)
checks["v2-gate: POSITIVE CONTROL is >100x its own noise floor"] = (
    r2 is not None and rot_of("v2-gate") > 100.0 * r2["h_self_residual"])
# And the stage the fix actually targets, stated on its own.  conv0_molecular
# is where D1(a) acts, and it is the one place the claim is clean: no RMSNorm
# has amplified anything yet.
s1 = "1_local_feature_extractor"
v3s1 = rows["v3-gate"]["stages"][s1]["invariance_residual"] \
    if rows["v3-gate"] else None
v2s1 = rows["v2-gate"]["stages"][s1]["invariance_residual"] \
    if rows["v2-gate"] else None
checks["conv+dense0 (where D1(a) acts): v3 < 1e-5"] = (
    v3s1 is not None and v3s1 < 1e-5)
checks["conv+dense0: v2 POSITIVE CONTROL > 1e-2"] = (
    v2s1 is not None and v2s1 > 1e-2)

print()
ok = True
for name, passed in checks.items():
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok &= bool(passed)

# Reported, not gated.  v3-vel is expected to fail the tolerance -- that is the
# D2 measurement, and asserting it either way would be asserting a number
# nobody has a prediction for.
v3v, v3g = rot_of("v3-vel"), rot_of("v3-gate")
if v3v is not None and v3g is not None:
    print()
    print(f"  [INFO] velocities in the slot cost "
          f"{v3v / max(v3g, 1e-30):.0f}x the rotation residual "
          f"({v3v:.3e} vs {v3g:.3e}) -- D2's third argument, measured")
v2c = rot_of("v2-canon")
if v2c is not None:
    print(f"  [INFO] wp-v2 + canonicalisation reaches {v2c:.3e}: the old "
          f"remedy made the *input* pose-free, not `h`")

print()
print(f"=== section 4 gate: {'PASS' if ok else 'FAIL'} ===")
raise SystemExit(0 if ok else 1)
PYEOF
