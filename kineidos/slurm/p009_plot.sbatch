#!/usr/bin/env bash
# P009's loss figure.
#
# Reading and aggregating is not an exempt category (AGENTS.md, and the user has
# said so three times): this job joins four arms x 15-18 rounds x 256 windows x
# 3 rank shards of jsonl, then scans ~8 GB of slurm logs for the train curve.
# Small today, ten times this on the next grid -- and a one-liner on the login
# node is how it stays there.  So it is a module in the repo plus this file.
#
# CPU only: no model is constructed, nothing touches a GPU.
#
#   sbatch repos/research/kineidos-v3/kineidos/slurm/p009_plot.sbatch
#SBATCH --job-name=p009-plot
#SBATCH --partition=background
#SBATCH --account=grad-students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.out
#SBATCH --error=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace/runs/slurm/%x-%j.err

set -euo pipefail

WS=/mnt/xfs/home/mhg/Projects/ForSiyuan/Kineidos-Workspace
ENV=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm
cd "${WS}"

# Same four as every other P009 run, so the command keeps its meaning.
export PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2
export LAYERNORM_TYPE=torch
export ATTN_IMPL=sdpa
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
# The tree lives on NFS and this module gets edited between runs; a stale
# kineidos/__pycache__/*.pyc on the compute node silently ran the previous
# version once (2026-10-08).  Analysis jobs are short, so don't cache.
export PYTHONDONTWRITEBYTECODE=1

echo "host $(hostname)  started $(date -Is)"
git -C repos/research/kineidos-v3 rev-parse --short HEAD

"${ENV}/bin/python" -m kineidos.plot_heldout \
  --out artifacts/reports/P009/p009_loss_curves.png \
  --json artifacts/reports/P009/p009_loss_curves.json \
  --verify artifacts/reports/P009/readout.md

echo "done $(date -Is)"

# the same three arguments as a 16:9 strip, for the one-page report frame
"${ENV}/bin/python" -m kineidos.plot_heldout --layout report \
  --out artifacts/reports/P009/p009_loss_report.png \
  --json artifacts/reports/P009/p009_loss_curves.json \
  --verify artifacts/reports/P009/readout.md

echo "report done $(date -Is)"
