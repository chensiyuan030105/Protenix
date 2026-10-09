#!/usr/bin/env bash
# P010 section 5's arm table, executable.
#
# Ten submissions with one source of truth for which tree, which mode and
# which switches each arm gets.  Ten hand-typed sbatch lines is nine chances
# to give an arm the wrong tree, and an arm on the wrong tree is not a crash --
# it is a plausible number for an experiment that did not run.  env.lock
# records the tree and its tags, so it would be caught eventually; this is so
# it does not have to be.
#
# **It refuses to submit until the three tags exist.**  p010-base is what
# section 6 item 5a's acceptances gate (the profile, the per-sigma tool's two
# bit-identity checks, read_heldout's refusal, arm 0), and p010-oracle and
# p010-gamma are what item 5b's identity check gates.  Submitting before the
# tags is how nine arms come to be compared against a baseline that was never
# accepted.
#
#   bash repos/research/kineidos-v3-diag/kineidos/slurm/p010_submit_arms.sh
#   DRY=1 bash .../p010_submit_arms.sh      # print the submissions, send none
#   ARMS_ONLY="oracle oracle-decoy" bash ...  # resubmit a subset

set -euo pipefail

WORKSPACE="/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace"
cd "${WORKSPACE}"

DIAG="repos/research/kineidos-v3-diag"
ORACLE="repos/research/kineidos-v3-diag-oracle"
GAMMA="repos/research/kineidos-v3-diag-gamma"

# Nodes to stay off, rather than one node to aim at.  --nodelist with more than
# one name is read as -N <count> ("invalid number of nodes (-N 3-1)"), and
# aiming at a single node is a bet on it staying free.  Excluded: the h100s
# (broken GPUs, AGENTS.md), chungus-8 and the deep-gpus (down), chungus-6 (no
# GPU at all), and 3/4/5/7 while our own P006 GROMACS array has their CPUs at
# load 38-74 -- load correlates cleanly with step time (4-6.5 -> 6-7.7 s/it,
# 33 -> 14.1).
#
# Spreading also limits a single preemption: on 2026-10-08 one tier-10 job
# landing on deep-chungus-10 took out four of ours in the same second, because
# they had all been placed there together.
EXCLUDE="${EXCLUDE:-deep-h-1,deep-h-2,deep-h-3,deep-chungus-3,deep-chungus-4,deep-chungus-5,deep-chungus-6,deep-chungus-7,deep-chungus-8,deep-gpu-10,deep-gpu-11}"

# arm | tree | wp.mode | extra --export fields | extra trailing args
# Section 5's table, in the same order.
# Section 6 item 2's gate needs exactly four of these -- oracle, oracle-decoy,
# random-1gpu, zero-1gpu -- and the gate is what decides where P011 goes.  So
# those four are submitted first: with fewer than eight slots free, slurm
# starts jobs in submission order, and a partial start should be a readable
# gate rather than four arms of a table that cannot be read yet.
read -r -d '' TABLE <<'EOF' || true
oracle|ORACLE|oracle||--wp.oracle_source target
oracle-decoy|ORACLE|oracle||--wp.oracle_source decoy
random-1gpu|DIAG|random||
zero-1gpu|DIAG|zero||
random-1gpu-seed2|DIAG|random|SEED=43|
oracle-noaug|ORACLE|oracle||--wp.oracle_source target --wp.target_augmentation false
# P010 section 8: the linear oracle hands the network a two-dimensional shadow,
# so its negative gate reads on the encoding and not on the pathway.  These two
# repeat the gate with an encoding that loses nothing (trilateration
# reconstruction 4e-15 nm) and carries no frame (invariance 5.7e-14).  Decoy is
# not optional -- D2 item 6.
oracle-v2|ORACLE|oracle||--wp.oracle_source target --wp.oracle_encoding fourier_anchor
oracle-v2-decoy|ORACLE|oracle||--wp.oracle_source decoy --wp.oracle_encoding fourier_anchor
# readout.md 13.3: the decoy is not a null for a distance encoding (same
# molecule, 82% of the effect).  This one keeps the signal and destroys only
# the atom-to-geometry correspondence.
oracle-v2-shuffle|ORACLE|oracle||--wp.oracle_source target --wp.oracle_encoding fourier_anchor --wp.oracle_shuffle true
gamma-freeze-A|GAMMA|random||--wp.freeze_layernorm_until_step 2000
gamma-freeze-B|GAMMA|random||--wp.freeze_layernorm_until_step 1000
EOF

want_tag () {
  # The tag must point at that worktree's HEAD, not merely exist: a tag left
  # behind on an earlier commit would pass an existence check while the arm
  # ran on code nobody accepted.
  local tree="$1" tag="$2"
  git -C "${tree}" tag --points-at HEAD | grep -qx "${tag}"
}

# DEPEND short-circuits the tag check, and only that: when the arms are
# submitted as the tail of a dependency chain, the acceptance jobs ARE the
# gate -- slurm's afterok will not start an arm if the identity check exits
# non-zero, and that check now stamps the three tags itself.  Checking for
# tags at submit time would be checking for something that cannot exist yet.
if [ -n "${DEPEND:-}" ]; then
  echo "DEPEND=${DEPEND}: the tag check is the dependency's job."
  echo "afterok will hold the arms until the identity check passes, and that"
  echo "check stamps p010-base / p010-oracle / p010-gamma on its way out."
  echo ""
fi

MISSING=""
if [ -z "${DEPEND:-}" ]; then
  want_tag "${DIAG}"   p010-base   || MISSING="${MISSING} ${DIAG}:p010-base"
  want_tag "${ORACLE}" p010-oracle || MISSING="${MISSING} ${ORACLE}:p010-oracle"
  want_tag "${GAMMA}"  p010-gamma  || MISSING="${MISSING} ${GAMMA}:p010-gamma"
fi
if [ -n "${MISSING}" ]; then
  echo "refusing to submit: these tags do not point at their worktree's HEAD:" >&2
  for m in ${MISSING}; do echo "    ${m}" >&2; done
  echo "" >&2
  echo "p010-base is gated on section 6 item 5a (the profile, the per-sigma" >&2
  echo "tool's two bit-identity checks, read_heldout's refusal, arm 0), and" >&2
  echo "the other two on item 5b's identity check. Submitting before the tags" >&2
  echo "is how nine arms come to share a baseline that was never accepted." >&2
  exit 1
fi

# Same precondition as p010_identity.sbatch, checked before anything is
# submitted: an arm on a tree that is not DIAG plus one layer is an arm whose
# baseline is not the one the other arms share.
for tree in "${ORACLE}" "${GAMMA}"; do
  if ! git -C "${tree}" merge-base --is-ancestor \
       "$(git -C "${DIAG}" rev-parse HEAD)" HEAD 2>/dev/null; then
    echo "refusing to submit: ${tree} is not a descendant of ${DIAG}'s HEAD." >&2
    echo "  Behind by $(git -C "${tree}" rev-list --count \
         "HEAD..$(git -C "${DIAG}" rev-parse HEAD)" 2>/dev/null || echo '?')" \
         "commits. Rebase it first -- nine arms cannot share a baseline that" >&2
    echo "  two of the trees do not contain." >&2
    exit 1
  fi
done

for tree in "${DIAG}" "${ORACLE}" "${GAMMA}"; do
  if [ -n "$(git -C "${tree}" status --porcelain)" ]; then
    echo "refusing to submit: ${tree} is dirty. An arm's env.lock would say" >&2
    echo "DIRTY and the tag would describe code that is not what ran." >&2
    git -C "${tree}" status --short >&2
    exit 1
  fi
done

echo "tags check out:"
for tree in "${DIAG}" "${ORACLE}" "${GAMMA}"; do
  echo "  $(basename "${tree}")  $(git -C "${tree}" rev-parse --short HEAD)" \
       "@$(git -C "${tree}" tag --points-at HEAD | paste -sd,)"
done
echo ""

SUBMITTED=""
while IFS='|' read -r ARM TREEVAR MODE EXTRA_EXPORT EXTRA_ARGS; do
  [ -n "${ARM}" ] || continue
  if [ -n "${ARMS_ONLY:-}" ] && ! printf '%s\n' ${ARMS_ONLY} | grep -qx "${ARM}"; then
    continue
  fi
  case "${TREEVAR}" in
    DIAG) TREE="${DIAG}";; ORACLE) TREE="${ORACLE}";; GAMMA) TREE="${GAMMA}";;
    *) echo "unknown tree ${TREEVAR}" >&2; exit 1;;
  esac
  EXPORTS="ALL,ARM=${ARM},WP_MODE=${MODE},TREE=${TREE}"
  [ -n "${EXTRA_EXPORT}" ] && EXPORTS="${EXPORTS},${EXTRA_EXPORT}"
  CMD=(sbatch --parsable
       --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G --nodes=1
       --exclude="${EXCLUDE}"
       --export="${EXPORTS}")
  # high-priority by default now, which D3 could not use: its 12-GPU
  # allowance was taken by P009's four arms.  Eight arms at one GPU fit under
  # that cap and under the QoS's GrpJobs=8, and tier 10 is not preempted --
  # which matters because one tier-10 job of somebody else's took out four of
  # our background jobs in the same second today, and a preempted 2000-step
  # arm has to resume, which is the path the provenance gate had to be built
  # for.  While P009 still holds the 12 GPUs these simply queue on
  # QOSGrpGpuLimit and drain in as its arms finish, which is what "queue
  # behind P009" means.
  [ -n "${PARTITION:-high-priority}" ] && CMD+=(--partition="${PARTITION:-high-priority}" --qos="${PARTITION:-high-priority}")
  [ -n "${DEPEND:-}" ] && CMD+=(--dependency=afterok:"${DEPEND}")
  CMD+=("${DIAG}/kineidos/slurm/p010_train.sbatch")
  # shellcheck disable=SC2086
  [ -n "${EXTRA_ARGS}" ] && CMD+=(${EXTRA_ARGS})
  if [ "${DRY:-0}" = "1" ]; then
    printf '%-20s %s\n' "${ARM}" "${CMD[*]}"
    continue
  fi
  JID=$("${CMD[@]}")
  printf '%-20s %s  tree=%s mode=%s %s\n' "${ARM}" "${JID}" \
         "$(basename "${TREE}")" "${MODE}" "${EXTRA_ARGS}"
  SUBMITTED="${SUBMITTED} ${JID}"
done <<< "${TABLE}"

[ "${DRY:-0}" = "1" ] && exit 0
echo ""
echo "submitted:${SUBMITTED}"
echo "the two baseline short arms (base-nomini, base-mini) are section 6"
echo "item 4's and were submitted before p010-base; they are not in this table."
