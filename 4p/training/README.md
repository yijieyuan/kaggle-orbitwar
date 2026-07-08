# experiment-025-simplefrac (4p) / v1

The **4-player** analog of the 2p `experiment-024-ilselfplay`. ONE self-contained experiment chaining
combined-IL → pick best → 4p FFA self-play **league** → eval, on the **v37 4p econ-CNN** architecture
(econ-CNN OrbitNet19 + CoordFracGauss, 2-tree `{net, frac}`, pointer self=hold + continuous
clipped-Gaussian fraction, **6-d ALL-IN frac edge_tid**).

```
  (a) combined IL/BC ─> (b) pick best IL ckpt ─> (c) 4p self-play league ─> (d) eval (FFA vsOL)
   train_il_v5.py        eval + pick best IL      train_league_4p.py         eval/eval.py --track 4p
   (2p + 4p .npz)        pick best .msgpack       --init_from <best IL>      pick best .msgpack
```

## Model / features — v37 4p econ-CNN (user: "参考 exp20v37")

Forked from `2p/experiments/experiment-020-imitate/v37/` (which is *already* the 4p econ-CNN port of the
2p v36): `jax_env/`, `model.py`, `train.py` (econ-correct helper module `tb`), `train_il_v5.py`,
`rl_infer.py`, `decode_parity.py`, `export_weights.py`, `agent.py`.

- **Seat-canonical / seat-invariant.** Any seat is rotated to a fixed canonical frame by a C4 rotation
  (`env._rot_xy/_rot_vec`); homes are FIXED `0=BR 1=BL 2=TR 3=TL`, ring order `0→1→3→2`. Opponent roles
  are **relative**: `q1`=clockwise neighbour, `q2`=antipodal, `q3`=ccw neighbour (`_ring_role`). So
  positions, velocities, and the per-role owner channels mean the **same relative player in every seat's
  view** — the environment, engine, and every feature are seat-invariant by construction.
- **Feature dims:** `static (P,30)` (5-hot owner + 3-hot type + 9 geom + 13 dyn), `ts (P,50,10)`
  (proj_ships + 5 role + exists + x,y + ramp), `glob (34)`, `econ_curves (50,8)` (per-role [ship,prod] ×
  {me,q1,q2,q3}), `edge (P,P,6)`. `net.apply(prm, static, ts, glob, reach, mask, edge, econ)` = **7
  args** (econ-CNN). frac `edge_tid = edge[ar, tid]` is **6-d ALL-IN only**.
- **Size:** `E=128, 6-layer trunk, 4 heads` (the exp25 deploy defaults in `train_league_4p.py`).
- **Elimination** is handled implicitly: an eliminated player owns no planet/fleet, so its owner channels
  collapse to 0; the FFA reward and `is_done` (alive≤1) handle it. A 2p game = a 4p board with two
  players already eliminated (the antipodal seats `{0,3}` occupied) — see IL data below.

### Engine fix vs raw v37 (important)
v37's `engine.py` walked the deploy first-hit horizon to turn 449; the jax env's `predict_first_hits`
uses `H_MAX_FIRSTHIT=100`. We adopt **exp023-4p's fixed, self-contained engine** (`engine.py` walks
`min(LAST_TURN, step+100)`, imports `utils/physics`) so the build/deploy `f_target/f_arrival` match the
training-time env. Deploy is **NATIVE 4p**: `me` = real seat `{0,1,2,3}`, **NO 2p→4p owner remap**
(`rl_infer.decode`/`value_of` had a v37 2p-deploy remap that would corrupt real seat 1/BL — removed).

## (a) Combined IL data (user 2026-06-18)

IL pretrains on a **COMBINED** dataset; `train_il_v5.py --data` is a **comma-separated** list of dirs.
Per episode the trainer detects 2p vs 4p from `len(meta["seats"])` and picks the seat loop + remap
per-sample (a batch can freely mix), so all positions are trained:

| source | builder | floor (user rule) | seats trained | `me` |
|---|---|---|---|---|
| **native 4p** | `build_dataset_4p.py` (`--min_all_score 1400`) | **lowest of 4 players ≥ 1400** | all 4 | `me = seat` |
| **2p** | `build_dataset_2p.py` (`--min_score 1400`) | **lowest of 2 players ≥ 1400** (= existing rule) | the 2 real seats | `{0,3}` (owner 1→seat 3, antipodal) |

The 2p data is the **existing** `2p/experiments/experiment-020-imitate/v1/data/v1` (32,219 episodes,
already min-of-2 ≥ 1400) — referenced by absolute path, not rebuilt. The 4p data is built fresh with the
lowest-of-4 ≥ 1400 rule.

```bash
# native-4p dataset (CPU; reads <official-replays>/<date>/4p). ~20 days available (2026-05-28..06-17).
python build_dataset_4p.py --days 2026-05-28,2026-05-29,...,2026-06-17 --min_all_score 1400 --out data/4p

# IL (GPU host). --data = the new 4p dir + the existing 2p dir.
python -u train_il_v5.py \
    --data "data/4p,<solution-root>/2p/experiments/experiment-020-imitate/v1/data/v1" \
    --E 128 --n_layers 6 --n_heads 4 \
    --weight_mode wscore --launch_w 6.0 --w_frac 0.5 --w_value 0.5 \
    --batch 256 --lr 3e-4 --updates 119000 --save_every 8000 --save_dir checkpoints_il
# -> checkpoints_il/ckpt_uXXXXX.msgpack (2-tree {net,frac}); w_value 0.5 co-trains the critic for the
#    RL warm-start. The "DATA episodes=N (4p=.. 2p=..)" line confirms the mix.
```

## (b)/(d) Eval — 4p FFA 1-vs-3 vs Orbit-Lite (all seats)

Export a checkpoint's deploy weights into `4p/inference/weights/`, then run the shipped 4p eval (each of
our models 1-vs-3 against Orbit-Lite, rotating through all 4 seats) from the solution root:

```bash
python export_weights.py checkpoints_sp/ckpt_uXXXXX.msgpack ../inference/weights/weights_4p_u44000.npz
python eval/eval.py --track 4p          # 4p model 1-vs-3 vs Orbit-Lite, each of the 4 seats
```
Winrate = strict 1st-place-by-ships share (seat-balanced). Per-checkpoint selection during training used
a cluster FFA-battery tool that is not bundled; `eval/eval.py` evaluates the shipped final model.
[official kaggle env, local OK — numpy deploy agents only.]
Confirm the numpy deploy port before trusting magnitudes (GPU host, Rule 7):
`GARRISON_REMAT=0 python decode_parity.py <ckpt.msgpack> <weights.npz>` (native-4p board pool, seats 0..3).

## (c) 4p self-play LEAGUE (warm-started from best IL)

`train_league_4p.py` — anchor seat 0 is always the current policy; non-anchor seats 1,2,3 are **each
independently** `1-league_p` current / `league_p` league (default 0.2); **3 independent** PFSP-sampled
pool members per update (one per non-anchor seat); **every current-controlled seat is recorded**
(league seats masked from the loss); per-seat FFA reward; **NO RESIGN**.

**Two-winrate league pool** (`LeaguePool4p`):
- `ema_first` — each member's strict **1st-place rate** → PFSP `P ∝ ema_first^pfsp_p + floor` (linear,
  prefer strong; obsolete members decay out).
- `ema_rank` — "current(anchor) **out-ranks** this member" rate → **admission** gate. A new save-grid
  ckpt is admitted when the **reference's** `ema_rank ≥ admit_thresh` (0.70) after `admit_min_games`
  (== the 2p "mastered the reference 70%", recast for FFA via pairwise rank), else force-admit every
  `max_admit_interval`. `u < min_admit_u` never admitted; IL base seeded as the first reference at u=0.

```bash
python -u train_league_4p.py \
    --n_envs 128 --T 128 --num_minibatches 16 --epochs 1 \
    --E 128 --n_layers 6 --n_heads 4 \
    --init_from checkpoints_il/ckpt_uBEST.msgpack \
    --league_p 0.2 --pfsp_p 1.0 --pfsp_floor 0.02 --ema_alpha 0.02 \
    --max_slots 30 --admit_thresh 0.70 --max_admit_interval 10000 --admit_min_games 256 --min_admit_u 1000 \
    --lr 3e-4 --lr_schedule warmflat --warmup_updates 1000 \
    --gamma 0.999 --lam 0.95 --clip 0.2 --ent 0.05 --margin_lam 0 --seed 0 \
    --updates 550000 --save_every 500 --save_dir checkpoints_sp
```

## Files

`jax_env/` (4p env, FFA reward, seat-canonical) · `model.py` (econ-CNN OrbitNet19 + CoordFracGauss) ·
`train.py` (`tb` econ-correct helpers) · `train_il_v5.py` (combined 2p+4p IL) ·
`train_league_4p.py` (4p league) · `build_dataset_4p.py` / `build_dataset_2p.py` · `obs_state.py` /
`engine.py` / `utils/physics/` (state reconstruction, fixed H=100) · `rl_infer.py` (numpy deploy port) ·
`agent.py` (`ORBIT4P_WEIGHTS`) · `decode_parity.py` · `export_weights.py` · `agent_meta.json`.

Board pool: procedural by default (`gen_init_states(reset(4))`); pass `--board_pool_path <boards.npz>` to
load a saved pool (the exact deploy pool is not bundled).

## Status

DEPLOYED — this is the shipped 4p model (experiment-025-simplefrac): IL warm-start (`checkpoints_il/`) →
4p FFA self-play league (`checkpoints_sp/`), byte-verified vs the deployed `4p/inference/weights/`.
