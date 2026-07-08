# Kaggle Orbitwar Solution

My final competition solution for the Kaggle [**Orbit Wars**](https://www.kaggle.com/competitions/orbit-wars) simulation competition.

**TL;DR.** My final submissions use a transformer with one token per planet, trained separately for the 2p and 4p tracks by self-play PPO with league training on a single GPU, 2p from a random start and 4p from an imitation-learning initialization. The final model is chosen by local evaluation, the average win-rate over all pairs in a pool of candidates. An inference-time strategy adds a small boost on top, and one uses it while the other does not.

---

### Environment setup

Follow the steps below to reproduce the training results and generate sample games from the checkpoints.

Fork this repo, then:
```bash
python -m venv .venv && . .venv/bin/activate     # or a conda env, Python 3.11
pip install -r requirements.txt                   # numpy, kaggle-environments==1.30.1, jax, flax, optax, torch
```

### Train

Two separate agents, self-play PPO on the JAX engine. Each is shown twice: the **deployed config** (the actual run that produced the model, on a 180 GB B200) and a **16 GB config** that fits an RTX A4000 — same model, just fewer parallel envs and smaller minibatches (10-update smoke-tested on an RTX 6000: peak 9.5 GB for 2p, 14 GB for 4p). Boards are procedural by default; a 2048-board sample ships under `training/shared/board_pool*/`.

**2p** — self-play PPO league:
```bash
cd 2p/training
# deployed (B200, ~180 GB)
python train_league.py --n_envs 1024 --T 128 --num_minibatches 64 --E 128 --n_layers 6 --n_heads 4 \
    --lr 1e-4 --lr_schedule cosine --warmup_updates 1000 --lr_min 3e-5 --cosine_updates 100000 \
    --ent 0.01 --margin_lam 0 --margin_D 300 --greedy_opp 0 --league 1 --max_slots 30 --admit_thresh 0.7 \
    --board_pool_path shared/board_pool/f256/boards.npz --updates 550000 --save_every 200 --save_dir checkpoints
# 16 GB (RTX A4000): same model, smaller batch, procedural boards  ->  peak ~9.5 GB
python train_league.py --n_envs 64 --num_minibatches 16 --T 128 --E 128 --n_layers 6 --n_heads 4 \
    --lr 1e-4 --ent 0.01 --league 1 --updates 550000 --save_every 500 --save_dir checkpoints_16g
# quick CPU smoke (~2 updates)
python train_league.py --n_envs 4 --updates 2 --save_every 1 --save_dir ./_smoke
```
**4p** — an **imitation-learning warm-start**, then a self-play league (the IL pipeline is in `4p/training/README.md`):
```bash
cd 4p/training
# deployed (exp25, B200)
python train_league_4p.py --n_envs 512 --T 128 --num_minibatches 64 --E 128 --n_layers 6 --n_heads 4 \
    --lr 3e-4 --lr_schedule warmflat --warmup_updates 1000 --ent 0.05 --league_p 0.2 --max_slots 30 \
    --admit_thresh 0.7 --init_from checkpoints_il/ckpt_uBEST.msgpack --updates 550000 --save_every 500 --save_dir checkpoints_sp
# 16 GB (RTX A4000): heavier model -> needs rematerialization + n_envs 16  ->  peak ~14 GB
GARRISON_REMAT=1 python train_league_4p.py --n_envs 16 --num_minibatches 16 --T 128 --E 128 --n_layers 6 --n_heads 4 \
    --lr 3e-4 --ent 0.05 --init_from checkpoints_il/ckpt_uBEST.msgpack --updates 550000 --save_every 500 --save_dir checkpoints_16g
```

To keep the repo lean, the shipped `checkpoints/` holds only the **four deployed submission checkpoints** (2p `u55000` + `u53000`, 4p `u44000` + `u39000`, byte-verified against `inference/weights/`); the full training runs and the 4p IL warm-start are not bundled — retrain to regenerate them. `train.py` is a legacy from-scratch variant — use `train_league.py`.

### Run eval

The same eval that produced the shipped replays: play our submissions and record, in one pass, viewer-ready.

```bash
cd eval
python eval.py                     # 2p (merge vs greedy, balanced 10/10) + 4p (1-vs-3 vs Orbit-Lite), 20 games each
python eval.py --track 2p          # only 2p
python eval.py --games 2 --perms 2 # quick smoke (2 games; first 2 of the 4p seatings)
python eval_jax.py --games 20      # 2p RL-vs-RL twin on the JAX engine (CUDA host; see run_jax_on_gpu.sh)
```
Each game is written straight to `local_replays/<Np>/<pair_id>/seed_NN/replay.json` (official kaggle schema + embedded `_aux{meta,winprob}` — RL win-confidence harvested live during play) plus a rebuilt `local_replays/<Np>/INDEX.json`, with no post-processing. Orbit-Lite is a torch rule-bot, so the 4p eval is official-engine only while `eval_jax.py` covers 2p. Agents: `2p_greedy 2p_merge 4p_greedy 4p_merge public_agent_1`.

### Watch a replay

The viewer is the project's real viewer (`viewer/`, verbatim). It reads over HTTP, so serve from the root:
```bash
python -m http.server 8000            # run from the solution root
```
Then open `http://localhost:8000/viewer/visualize_local.html` — it reads `local_replays/{2p,4p}/` (the eval output from step 2). Pick **Players: 2p / 4p**; the info bar shows per-turn ships/production, RL win-confidence (value head), passive projection, and a side panel. A sample game ships; regenerate or extend `local_replays/` with `eval/eval.py` (step 2).

### Use a model directly (inference)

Load a model and get moves for one observation:
```python
import sys; sys.path.insert(0, "2p/inference")   # 4p: "4p/inference"
import agent
act = agent.make_agent("merge")      # or "greedy"
moves = act(obs)                     # obs = a kaggle orbit_wars observation
```

### Results (final Kaggle leaderboard)

Two submissions: greedy `53993524` and merge `53993338`. Final Elo 1640.0 (greedy) and 1620.0 (merge), rank 13, swinging between 6th and 28th over the evaluation phase.
