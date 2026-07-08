#!/usr/bin/env python
"""Our 2p eval on the JAX training engine — the CUDA twin of eval.py, in ONE pass.

Plays our two 2p checkpoints against each other on the JAX engine (native-Windows jax is CPU-only, so
run this on a CUDA-jax host, e.g. the cluster) with BALANCED seating (10 games each seat, 20 total) and
writes each game DIRECTLY in the final viewer format — same schema + embedded _aux{meta,winprob} as the
official/numpy eval.py, into the SAME store so one viewer plays both:

    local_replays/2p/<pair_id>/seed_NN/replay.json     official-style schema + _aux (engine="jax")
    local_replays/2p/INDEX.json                          rebuilt from disk (merges the python + jax matchups)

Only 2p: the JAX engine plays our own RL checkpoints (the two public opponents are torch rule-bots and
only run on the official/numpy path in eval.py). "merge" is a numpy+overage deploy concept with no jax
form, so the jax twin is the two adjacent checkpoints (u55000 vs u53000) greedy-decoded. Win-confidence
(value head (V+1)/2) is harvested turn-by-turn during play and embedded as _aux.winprob.

Usage (GPU box, kaggle-orbitwar env):
    python eval_jax.py                                 # default ckpts u55000 vs u53000, balanced, 20 games
    python eval_jax.py --a <ckptA.msgpack> --b <ckptB.msgpack> --games 20
"""
import argparse
import json
import time
import uuid
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOL = HERE.parent


def load_engine(track="2p"):
    tdir = SOL / track / "training"
    sys.path.insert(0, str(tdir)); sys.path.insert(0, str(tdir / "jax_env"))
    import jax, jax.numpy as jnp
    import flax.serialization as fser
    from state import JaxState                                   # noqa
    from step import step as env_step                            # noqa
    from env import gen_init_states, basic_features, _forecast, is_done, ship_totals  # noqa
    from targeting import reach_solve_static, lead_for_ships     # noqa
    from model import OrbitNet19, SimpleFracMLP                  # noqa
    import train as tb                                           # edge_features, first_hit_gate
    net = OrbitNet19(E=128, n_layers=6, n_heads=4)
    frac = SimpleFracMLP(E=128, n_heads=4)
    def load(p): return jax.tree_util.tree_map(jnp.asarray, fser.msgpack_restore(open(p, "rb").read()))
    return dict(jax=jax, jnp=jnp, JaxState=JaxState, env_step=env_step, gen_init_states=gen_init_states,
                basic_features=basic_features, _forecast=_forecast, is_done=is_done, ship_totals=ship_totals,
                reach_solve_static=reach_solve_static, lead_for_ships=lead_for_ships, net=net, frac=frac,
                tb=tb, load=load)


def _build_step_fn(E):
    """Build the jit-compiled 2-seat transition: (st, prm0, prm1) -> (new_st, v0, v1). Both seats decode
    greedily (their value head = 5th net.apply output), actions merge by owner (neutral -> self/no-launch),
    gate + env_step advance. jitted -> compiled ONCE per process, then each step is a fast device call
    (the eager version was ~min/game because every op paid Python<->XLA dispatch latency)."""
    jax = E["jax"]; jnp = E["jnp"]; net = E["net"]; frac = E["frac"]
    _forecast = E["_forecast"]; reach_solve_static = E["reach_solve_static"]
    lead_for_ships = E["lead_for_ships"]; basic_features = E["basic_features"]
    env_step = E["env_step"]; tb = E["tb"]

    def _seat(prm, st, me, fc, R, Rg, edge):
        ar = jnp.arange(st.p_owner.shape[0])
        static, ts, glob, m, econ = basic_features(st, me, fc=fc)
        is_mine = (st.p_owner == me) & st.p_mask
        reach_me = Rg & is_mine[:, None]
        tgt, emb, gemb, _b, v = net.apply(prm["net"], static, ts, glob, reach_me, m, edge, econ)
        acting = is_mine & (st.p_ships > 0)
        tid = jnp.argmax(tgt, -1)
        is_real = acting & (tid != ar) & R[ar, tid]
        mu, sigma = frac.apply(prm["frac"], emb, emb[tid], gemb)
        f = jnp.clip(mu, 0.0, 1.0)
        ships_i = jnp.clip(jnp.round(f * st.p_ships.astype(jnp.float32)).astype(jnp.int32), 0, st.p_ships)
        return tid, ships_i, is_real, v

    def transition(st, prm0, prm1):
        ar = jnp.arange(st.p_owner.shape[0])
        fc = _forecast(st)
        R, ANG, TURNS, Rg = reach_solve_static(st)
        _, _, _, edge = tb.edge_features(st, fc=fc, lead=(R, ANG, TURNS))
        t0, s0, r0, v0 = _seat(prm0, st, 0, fc, R, Rg, edge)
        t1, s1, r1, v1 = _seat(prm1, st, 1, fc, R, Rg, edge)
        owner = st.p_owner
        e0 = (owner == 0); e1 = (owner == 1)                     # neutral (owner<0) -> self / no launch
        tid = jnp.where(e0, t0, jnp.where(e1, t1, ar))
        ships_sent = jnp.where(e0, s0, jnp.where(e1, s1, jnp.zeros_like(ar)))
        is_real = jnp.where(e0, r0, jnp.where(e1, r1, jnp.zeros(ar.shape, bool)))
        Rx, ANGx, TURNSx = lead_for_ships(st, ships_sent)
        angle = ANGx[ar, tid]; turns = TURNSx[ar, tid]
        cand = is_real & (ships_sent > 0) & Rx[ar, tid]
        launch = cand & tb.first_hit_gate(st, tid, angle, ships_sent)
        ships_final = jnp.where(launch, ships_sent, 0)
        new_st = env_step(st, launch, angle, ships_final, tid, st.step + turns)
        return new_st, v0.reshape(-1)[0], v1.reshape(-1)[0]

    return jax.jit(transition)


def play(E, params_by_seat, seed, n_players=2, max_steps=500, step_fn=None):
    jax = E["jax"]
    if step_fn is None:
        step_fn = _build_step_fn(E)
    prm0, prm1 = params_by_seat
    pool = E["gen_init_states"](1, seed)
    st = jax.tree_util.tree_map(lambda x: x[0], pool)            # single board
    av = float(jax.device_get(st).angular_velocity) if hasattr(st, "angular_velocity") else 0.0
    steps = []
    wp = [[None] * max_steps for _ in range(n_players)]          # seat -> per-step winprob (densified below)
    init_planets = None
    for t in range(max_steps):
        sth = jax.device_get(st)                                 # ONE host transfer/step (not per-scalar)
        pm = sth.p_mask
        planets = [[int(sth.p_id[i]), int(sth.p_owner[i]), float(sth.p_x[i]), float(sth.p_y[i]),
                    float(sth.p_radius[i]), float(sth.p_ships[i]), float(sth.p_prod[i])]
                   for i in range(pm.shape[0]) if bool(pm[i])]
        fm = sth.f_mask
        fleets = [[int(i), int(sth.f_owner[i]), float(sth.f_x[i]), float(sth.f_y[i]),
                   float(sth.f_angle[i]), int(sth.f_from[i]), float(sth.f_ships[i])]
                  for i in range(fm.shape[0]) if bool(fm[i])]
        if init_planets is None:
            init_planets = planets
        obs = {"step": t, "planets": planets, "fleets": fleets,
               "initial_planets": init_planets, "angular_velocity": av}
        steps.append([{"action": [], "observation": (obs if p == 0 else {k: v for k, v in obs.items() if k != "step"}),
                       "reward": 0, "status": "ACTIVE"} for p in range(n_players)])
        if bool(E["is_done"](st)):
            break
        st, v0, v1 = step_fn(st, prm0, prm1)                     # jitted 2-seat transition (+ both values)
        wp[0][t] = round((float(v0) + 1.0) / 2.0, 4)
        wp[1][t] = round((float(v1) + 1.0) / 2.0, 4)

    n_steps = len(steps)
    tot = E["ship_totals"](st)
    ships = [int(tot[p]) for p in range(n_players)]
    winner = int(max(range(n_players), key=lambda i: ships[i]))
    rewards = [1 if winner == p else -1 for p in range(n_players)]
    for p in range(n_players):
        steps[-1][p]["status"] = "DONE"; steps[-1][p]["reward"] = rewards[p]
    winprob = {str(p): wp[p][:n_steps] for p in range(n_players)}   # index t aligns with steps[t]
    return steps, rewards, winner, ships, n_steps, winprob


# ---- participant descriptors (derived from ckpt filenames; jax-tagged so they don't collide with
#      eval.py's official merge/greedy matchup in the shared local_replays/2p store) ----
def _stem(ckpt):
    return Path(ckpt).stem.replace("ckpt_", "")


def _desc(ckpt):
    return {"kind": "exp", "id": f"{_stem(ckpt)}-jax", "version": None}


def _pair_id(ckpts):
    return "_vs_".join(_stem(c) for c in ckpts) + "_jax"     # e.g. u55000_vs_u53000_jax


def _build_index(out_np_dir):
    """Rescan local_replays/2p/<pair_id>/seed_NN/replay.json -> INDEX.json (merge, not overwrite): a jax
    run must not wipe the official/numpy matchups. Reads each replay's embedded _aux.meta."""
    matches = []
    for pair_dir in sorted(p for p in out_np_dir.iterdir() if p.is_dir()):
        seeds, p0, p1, latest = [], None, None, 0.0
        for jf in sorted(pair_dir.glob("seed_*/replay.json")):
            meta = (json.loads(jf.read_text(encoding="utf-8")).get("_aux", {}) or {}).get("meta", {})
            if p0 is None:
                p0 = meta.get("match_p0"); p1 = meta.get("match_p1")
            mt = jf.stat().st_mtime
            latest = max(latest, mt)
            seeds.append({"seed": jf.parent.name, "winner": meta.get("winner"),
                          "scores": meta.get("scores", []), "n_turns": meta.get("n_turns"),
                          "seats": meta.get("seats", []), "mtime": mt})
        if seeds:
            matches.append({"pair_id": pair_dir.name, "p0": p0, "p1": p1,
                            "seeds": seeds, "_latest_mtime": latest})
    out_np_dir.mkdir(parents=True, exist_ok=True)
    (out_np_dir / "INDEX.json").write_text(json.dumps({"matches": matches}, indent=1), encoding="utf-8")
    return sum(len(m["seeds"]) for m in matches), len(matches)


# ---- multiprocessing worker: loads the jax engine + ckpt params ONCE per process, then reuses ----
_JW = {"E": None, "params": {}, "step_fn": None}


def _jax_play_task(task):
    """One jax game (module-level so multiprocessing workers can pickle it). task = (a, b, seed,
    out_np_str, pair_id). The engine (unpicklable jax modules) + params are loaded inside the worker
    and cached in _JW, so each worker pays the jax import/warmup once and reuses it across its games."""
    a, b, seed, out_np_str, pair_id = task
    if _JW["E"] is None:
        _JW["E"] = load_engine("2p")
        _JW["step_fn"] = _build_step_fn(_JW["E"])               # compile the jitted transition ONCE per process
    E = _JW["E"]
    for c in (a, b):
        if c not in _JW["params"]:
            _JW["params"][c] = E["load"](c)
    seat_ckpts = [a, b] if seed % 2 == 0 else [b, a]             # balanced 10/10
    params_by_seat = [_JW["params"][c] for c in seat_ckpts]
    steps, rewards, winner, ships, n_steps, winprob = play(E, params_by_seat, seed, 2, step_fn=_JW["step_fn"])
    display = [f"{_stem(c)}-jax" for c in seat_ckpts]
    replay = {"name": "orbit_wars", "id": str(uuid.uuid4()), "info": {"seed": int(seed)},
              "rewards": rewards, "statuses": ["DONE"] * 2, "steps": steps,
              "configuration": {"episodeSteps": 500}, "engine": "jax",
              "_aux": {
                  "meta": {"winner": winner, "scores": ships, "n_turns": n_steps, "n_agents": 2,
                           "seed": int(seed), "seats": display, "team_names": display,
                           "agent_a": display[0], "agent_b": display[1],
                           "pair_id": pair_id, "match_p0": _desc(a), "match_p1": _desc(b)},
                  "winprob": winprob}}
    out_dir = Path(out_np_str) / pair_id / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    return (_stem(seat_ckpts[winner]), "/".join(_stem(c) for c in seat_ckpts), seed, winner, ships, n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None, help="ckpt A (default checkpoints/ckpt_u55000.msgpack)")
    ap.add_argument("--b", default=None, help="ckpt B (default checkpoints/ckpt_u53000.msgpack)")
    ap.add_argument("--games", type=int, default=20, help="games total (balanced 10/10 seating)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (each loads jax once; CPU jax is slow)")
    ap.add_argument("--out-root", type=str, default=None, help="output root (default local_replays/)")
    args = ap.parse_args()

    ckdir = SOL / "2p" / "training" / "checkpoints"
    a = args.a or str(ckdir / "ckpt_u55000.msgpack")
    b = args.b or str(ckdir / "ckpt_u53000.msgpack")
    out_root = Path(args.out_root).resolve() if args.out_root else (SOL / "local_replays")
    out_np = out_root / "2p"
    pair_id = _pair_id([a, b])                                    # canonical order a,b (regardless of per-game seating)
    print(f"[jax 2p] {_stem(a)} vs {_stem(b)}, balanced 10/10, {args.games} games (jobs={args.jobs}) -> {out_np.relative_to(SOL)}", flush=True)

    tasks = [(a, b, seed, str(out_np), pair_id) for seed in range(args.games)]
    counts = {}

    def _tally(res):
        w, order, seed, winner, ships, n_steps = res
        counts[w] = counts.get(w, 0) + 1
        print(f"    seed {seed:02d} [{order}]: winner=seat{winner} ({w}) ships={ships} steps={n_steps}", flush=True)

    if args.jobs and args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for res in pool.imap_unordered(_jax_play_task, tasks):
                _tally(res)
    else:
        for t in tasks:
            _tally(_jax_play_task(t))
    print("    -> win counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    ng, nm = _build_index(out_np)
    print(f"DONE jax 2p: {nm} matchups on disk, {ng} games -> {(out_np / 'INDEX.json').relative_to(SOL)}")
    print("JAX_EVAL_DONE")


if __name__ == "__main__":
    main()
