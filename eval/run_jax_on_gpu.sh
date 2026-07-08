#!/usr/bin/env bash
# Run the 2p JAX-engine eval on a CUDA Linux host and (optionally) sync the replays back into
# local_replays/2p/. Native-Windows jax is CPU-only, so all jax runs go to a Linux GPU box.
# eval.py (numpy + official engine) is the deploy ground truth and also does the 4p all-perm eval;
# it runs locally. This jax twin covers 2p ONLY (the two public 4p opponents are torch rule-bots and
# only run on eval.py's official/numpy path). Output format is identical, so the same viewer plays both.
#
# You can also just run eval_jax.py directly on any CUDA Linux host in the kaggle-orbitwar conda env;
# this wrapper only adds env activation + a sync hint.
#
# Prereq: the solution folder (or at least 2p/training + its checkpoints) under $PROJ + the
# kaggle-orbitwar conda env.
set -e
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"   # solution root; override with PROJ=...
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"  # override for your conda
source "$CONDA_SH"
conda activate kaggle-orbitwar
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2   # small; eval only
cd "$PROJ/eval"

# 2p: the two final checkpoints against each other (u55000 main vs u53000 partner), balanced 10/10, 20 games.
# Defaults load 2p/training/checkpoints/ckpt_u55000.msgpack + ckpt_u53000.msgpack; override with --a/--b.
python eval_jax.py --games 20

echo "jax 2p replays written to local_replays/2p/ (merged into INDEX.json alongside the official matchups)."
echo "If you ran this on a remote GPU host, sync back with:"
echo "  scp -r <USER>@<GPU_HOST>:$PROJ/local_replays/2p  <local-solution-root>/local_replays/"
