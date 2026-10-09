#!/usr/bin/env bash
# P010 §14.1 as a figure for the board's one-page summary.
#
#   sbatch repos/research/kineidos-v3-diag/kineidos/slurm/p010_plot.sbatch
#
# CPU only. It is small, but reading and reducing published numbers is not an
# exempt category (AGENTS.md), and a one-liner left on the login node is how the
# next, larger version of it stays there.
#SBATCH --job-name=p010-plot
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail
WS=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace
ENV=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm
cd "${WS}"
export PYTHONPATH=repos/research/kineidos-v3-diag:repos/research/wp-v2
export LAYERNORM_TYPE=torch ATTN_IMPL=sdpa MPLBACKEND=Agg
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
# NFS + an edited module: a .pyc from another host can be served as current.
export PYTHONDONTWRITEBYTECODE=1
echo "host $(hostname)  $(date -Is)"
git -C repos/research/kineidos-v3-diag rev-parse --short HEAD
"${ENV}/bin/python" -m kineidos.plot_sigma_arms --step 499
echo "done $(date -Is)"
