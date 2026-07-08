#!/usr/bin/env python
"""Our eval on the OFFICIAL kaggle engine — play our submissions and record, in ONE pass.

This IS the eval we run: it plays our final agents against each other / the public opponents on
the official kaggle engine and writes each game DIRECTLY in the final viewer format — no separate
post-processing step. Per game it writes:

    local_replays/<Np>/<pair_id>/seed_NN/replay.json    official env.toJSON() schema
                                                         + embedded _aux{meta, winprob}
and, per player-count, one index the viewer reads verbatim:

    local_replays/<Np>/INDEX.json                        {matches:[{pair_id,p0,p1,seeds:[...]}]}

The RL win-confidence line (value head V(s) -> (V+1)/2 per turn) is harvested turn-by-turn DURING
play and embedded as _aux.winprob[seat] — so viewer/visualize_local.html reads the output straight,
no winprob/index build step afterwards.

Matchups (defaults):
  2p : our two submissions  2p_merge  vs  2p_greedy, BALANCED seating (10 games each seat), 20 games.
  4p : each of {4p_merge, 4p_greedy} plays 1-vs-3 against Orbit-Lite (public_agent_1), rotating
       through all 4 seats -> 8 seatings x 20 games.
       (4p has no jax twin: Orbit-Lite is a torch rule-bot and only runs on this official/numpy
        path. eval_jax.py records the 2p RL-vs-RL twin on the jax engine.)

Agents (any at any seat; n_players = number of seats):
  2p_greedy 2p_merge   -> final 2p model (exp-026), Kaggle subs 53993524 / 53993338
  4p_greedy 4p_merge   -> final 4p model (exp-025), Kaggle subs 53993524 / 53993338
  public_agent_1       -> Orbit-Lite   (the public opponent)

Usage:
  python eval.py                              # 2p balanced + 4p 1-vs-3, 20 games each
  python eval.py --track 2p                   # only 2p
  python eval.py --games 2 --perms 2          # quick smoke (2 games; first 2 of the 4p seatings)
  python eval.py --combo 4p_merge,public_agent_1,public_agent_1,public_agent_1 --games 20   # one seating
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["EVAL_FORCE_MERGE"] = "1"  # local eval has no real Kaggle time limit -> merge agents skip the overage-bank
                                      # fallback and always run the full rollout (also fixes 3-merge shared _self_tally)
HERE = Path(__file__).resolve().parent
SOL = HERE.parent

# ---- participant registry: name -> INDEX descriptor {kind,id,version} + short slug (pair_id/paths) ----
PARTS = {
    "2p_merge":  {"kind": "exp",   "id": "merge",  "version": None, "slug": "merge"},
    "2p_greedy": {"kind": "exp",   "id": "greedy", "version": None, "slug": "greedy"},
    "4p_merge":  {"kind": "agent", "id": "merge",  "version": None, "slug": "merge"},
    "4p_greedy": {"kind": "agent", "id": "greedy", "version": None, "slug": "greedy"},
    "public_agent_1": {"kind": "agent", "id": "pub1", "version": None, "slug": "pub1"},  # Orbit-Lite
}
OUR_AGENTS = {"2p_merge", "2p_greedy", "4p_merge", "4p_greedy"}   # seats with an RL value head (winprob)


def _desc(name):
    p = PARTS[name]
    return {"kind": p["kind"], "id": p["id"], "version": p["version"]}


def _pair_id(names):
    return "_vs_".join(PARTS[n]["slug"] for n in names)


# ---- agent registry: name -> zero-arg factory returning agent(obs, config=None) ----
_CACHE = {}


def _load_file_module(pyfile, modname):
    pyfile = str(pyfile)
    d = os.path.dirname(pyfile)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(modname, pyfile)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


def _make(name):
    if name in _CACHE:
        return _CACHE[name]
    if name in ("2p_greedy", "2p_merge"):
        m = _load_file_module(SOL / "2p" / "inference" / "agent.py", "sol_agent_2p")
        fn = m.make_agent("greedy" if name.endswith("greedy") else "merge")
    elif name in ("4p_greedy", "4p_merge"):
        m = _load_file_module(SOL / "4p" / "inference" / "agent.py", "sol_agent_4p")
        fn = m.make_agent("greedy" if name.endswith("greedy") else "merge")
    elif name == "public_agent_1":
        m = _load_file_module(HERE / "opponents" / "public_agent_1_orbitlite" / "agent.py", "pub1_agent")
        fn = m.agent
    else:
        raise ValueError(f"unknown agent {name!r}")
    _CACHE[name] = fn
    return fn


def _make_public_fresh(seat):
    """Public opponents keep a STATEFUL rolling movement cache (module-level _RUNTIME), so a 1-vs-3
    game -- which seats the same bot 3x -- needs an INDEPENDENT instance per seat. Load agent.py under
    a per-seat module name -> a fresh _RUNTIME (the orbit_lite lib it imports is stateless, shared)."""
    m = _load_file_module(HERE / "opponents" / "public_agent_1_orbitlite" / "agent.py", f"pub1_agent_seat{seat}")
    return m.agent


def _call(fn, obs, player):
    try:
        return fn(obs, None) or []
    except TypeError:
        return fn(obs) or []


def _get(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


class _Winprob:
    """Turn-by-turn RL win-confidence (V(s)+1)/2 for one seat, harvested live during play.

    Uses the deploy module's own primitives (NO edits to the parity-verified agents): a private
    AgentState fed every turn + rl_infer.value_of on the greedy checkpoint's value head
    (2p: u55000; 4p: u44000). Both our submissions on a track share that value head, so one
    computer type serves either seat. Public opponents have no value head -> no _Winprob is made.
    Samples are (step_index, wp) pairs so the final series aligns EXACTLY to replay.steps.
    """

    def __init__(self, track):
        if track == "2p":
            import rl_agent_greedy as G            # imported when the 2p_greedy seat was made
        else:
            import agent_greedy as G               # 4p flat inference; imported when the 4p_greedy seat was made
        self._obs_to_arr = G._obs_to_arr
        self._value_of = G.R.value_of
        self._W = G._W
        self.state = G.AgentState()
        self.samples = []                          # list[(step_index, wp_or_None)]

    def step(self, obs, me, step_index):
        self.state.update(obs)
        arr, _ = self._obs_to_arr(obs, self.state)
        if arr["p_x"].shape[0] == 0:
            self.samples.append((step_index, None)); return
        v = self._value_of(arr, self._W, me)
        self.samples.append((step_index, round(float((v + 1.0) / 2.0), 4)))


def play_and_record(seat_agents, seed, track, out_np_dir, pair_id, max_steps=500):
    """Play one game (n = len(seat_agents)) on the official kaggle engine; write the final-format
    replay (official schema + embedded _aux) to out_np_dir/pair_id/seed_NN/replay.json. Returns the
    INDEX seed row for this game."""
    from kaggle_environments import make
    n = len(seat_agents)
    fns = [_make_public_fresh(i) if a == "public_agent_1" else _make(a)  # repeated bot -> fresh per-seat instance
           for i, a in enumerate(seat_agents)]
    wps = [_Winprob(track) if a in OUR_AGENTS else None for a in seat_agents]

    random.seed(seed)  # orbit_wars board determinism comes from global random state at make() time
    env = make("orbit_wars", debug=False, configuration={"episodeSteps": max_steps})
    env.reset(n)
    env.step([[]] * n)
    while not env.done:
        si = len(env.steps) - 1                    # index of the step we're acting on == replay.steps[si]
        step0 = env.steps[-1][0].observation.get("step")   # kaggle sets `step` ONLY in seat-0's obs
        acts = []
        for p in range(n):
            po = env.steps[-1][p].observation
            if p != 0 and po.get("step") is None and step0 is not None:
                po = dict(po); po["step"] = step0  # inject: else seat>0's turn feature (step/500) sticks at 0 -> plays broken -> fake seat bias
            if wps[p] is not None:
                try:
                    wps[p].step(po, p, si)
                except Exception:
                    wps[p].samples.append((si, None))
            acts.append(_call(fns[p], po, p))
        env.step([a or [] for a in acts])

    replay = env.toJSON()
    n_steps = len(replay["steps"])

    # winner = seat with most ships (planets + fleets) at the end
    last = env.steps[-1][0].observation
    ships = [0] * n
    for pl in last.get("planets", []) or []:
        o = int(pl[1])
        if 0 <= o < n:
            ships[o] += pl[5]
    for fl in last.get("fleets", []) or []:
        o = int(fl[1])
        if 0 <= o < n:
            ships[o] += fl[6]
    winner = int(max(range(n), key=lambda i: ships[i]))

    # densify winprob to full replay length so index t == replay.steps[t]
    winprob = {}
    for p, w in enumerate(wps):
        if w is None:
            continue
        series = [None] * n_steps
        for (si, v) in w.samples:
            if 0 <= si < n_steps:
                series[si] = v
        winprob[str(p)] = series

    display = [PARTS[a]["slug"] for a in seat_agents]
    replay["_aux"] = {
        "meta": {
            "winner": winner, "scores": ships, "n_turns": n_steps, "n_agents": n,
            "seed": int(seed), "seats": display, "team_names": display,
            "agent_a": display[0], "agent_b": (display[1] if n > 1 else None),
            # INDEX rescan helpers (viewer ignores extra keys):
            "pair_id": pair_id, "match_p0": _desc(seat_agents[0]), "match_p1": _desc(seat_agents[1]),
        },
        "winprob": winprob,
    }

    out_dir = out_np_dir / pair_id / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    return {"seed": f"seed_{seed:02d}", "winner": winner, "scores": ships,
            "n_turns": n_steps, "seats": list(seat_agents), "mtime": time.time()}


def _seatings_2p(games):
    """Balanced 10/10: even game -> merge at p0, odd -> greedy at p0. 20 games => 10 each seat."""
    for seed in range(games):
        yield seed, (["2p_merge", "2p_greedy"] if seed % 2 == 0 else ["2p_greedy", "2p_merge"])


# 4p eval: each of our models plays 1-vs-3 against Orbit-Lite, rotating through all 4 seats (8 seatings).
FOUR_P_MODELS = ["4p_merge", "4p_greedy"]
FOUR_P_OPP = "public_agent_1"                       # Orbit-Lite fills the other 3 seats


def _seatings_4p():
    """Each of our two models placed in each of the 4 seats vs 3 Orbit-Lite -> 8 seatings; measures
    seat-balanced 1-vs-3 FFA performance against the single public opponent."""
    for model in FOUR_P_MODELS:
        for pos in range(4):
            seats = [FOUR_P_OPP] * 4
            seats[pos] = model
            yield seats


def _build_index(out_np_dir, n):
    """Rescan out_np_dir/<pair_id>/seed_NN/replay.json -> INDEX.json (merge, not overwrite): a run
    that records one matchup must not wipe others from the listing. Per-game winner/scores/seats are
    read back from each replay's embedded _aux.meta."""
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
    (out_np_dir).mkdir(parents=True, exist_ok=True)
    (out_np_dir / "INDEX.json").write_text(json.dumps({"matches": matches}, indent=1), encoding="utf-8")
    return sum(len(m["seeds"]) for m in matches), len(matches)


def _play_task(task):
    """One game — module-level so multiprocessing workers can pickle it. task = (seats, seed, track,
    out_np_str, pair_id). Each worker only ever loads ONE track's agents (2p and 4p are dispatched in
    separate pools), so the flat 2p/4p rl_infer/engine names never collide inside a worker."""
    seats, seed, track, out_np_str, pair_id = task
    row = play_and_record(list(seats), seed, track, Path(out_np_str), pair_id, max_steps=500)
    return (pair_id, list(seats), row)


def _dispatch(tasks, jobs):
    """Play all game tasks (jobs>1 -> multiprocessing pool of `jobs` workers) and print per-game
    results as they finish. Returns {agent_name: win_count}."""
    counts = {}
    def _tally(res):
        pid, seats, row = res
        w = seats[row["winner"]]
        counts[w] = counts.get(w, 0) + 1
        print(f"    {row['seed']} [{pid}]: winner=seat{row['winner']} ({w})  ships={row['scores']}  steps={row['n_turns']}", flush=True)
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            for res in pool.imap_unordered(_play_task, list(tasks)):
                _tally(res)
    else:
        for t in tasks:
            _tally(_play_task(t))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["2p", "4p", "both"], default="both")
    ap.add_argument("--games", type=int, default=20, help="games per matchup/permutation (default 20)")
    ap.add_argument("--perms", type=int, default=None, help="4p: cap number of seatings (default all 8)")
    ap.add_argument("--combo", type=str, default=None, help="single explicit seating, comma-separated (overrides track defaults)")
    ap.add_argument("--out-root", type=str, default=None, help="output root (default local_replays/)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes per track (default 1)")
    args = ap.parse_args()

    # 4p/inference is flat (no rl4p/ subpackage), so the 2p and 4p tracks share top-level module names
    # (rl_infer, engine) and must NOT be imported in the same process. Run each track in its own subprocess.
    if args.track == "both" and not args.combo:
        import subprocess
        rc = 0
        for tk in ("2p", "4p"):
            cmd = [sys.executable, os.path.abspath(__file__), "--track", tk,
                   "--games", str(args.games), "--jobs", str(args.jobs)]
            if args.perms is not None:
                cmd += ["--perms", str(args.perms)]
            if args.out_root:
                cmd += ["--out-root", args.out_root]
            print(f"=== eval.py --track {tk} (isolated subprocess) ===", flush=True)
            rc |= subprocess.run(cmd).returncode
        sys.exit(rc)

    out_root = Path(args.out_root).resolve() if args.out_root else (SOL / "local_replays")

    touched = set()  # np subdirs to (re)index

    if args.combo:
        seats = args.combo.split(",")
        n = len(seats)
        track = "2p" if n == 2 else "4p"
        out_np = out_root / f"{n}p"
        pid = _pair_id(["2p_merge", "2p_greedy"]) if track == "2p" else _pair_id(seats)
        print(f"[{n}p] combo {args.combo}: {args.games} games (jobs={args.jobs})")
        _dispatch([(seats, s, track, str(out_np), pid) for s in range(args.games)], args.jobs)
        touched.add((out_np, n))
    elif args.track == "2p":
        out_np = out_root / "2p"
        pid = _pair_id(["2p_merge", "2p_greedy"])
        print(f"[2p] 2p_merge vs 2p_greedy, balanced 10/10, {args.games} games (jobs={args.jobs}) -> {out_np.relative_to(SOL)}")
        tasks = [(seats, seed, "2p", str(out_np), pid) for seed, seats in _seatings_2p(args.games)]
        counts = _dispatch(tasks, args.jobs)
        print("    -> agent win counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        touched.add((out_np, 2))
    elif args.track == "4p":
        out_np = out_root / "4p"
        seatings = list(_seatings_4p())
        if args.perms is not None:
            seatings = seatings[:args.perms]
        print(f"[4p] {len(seatings)} seatings (each model 1-vs-3 Orbit-Lite, all 4 seats) x {args.games} games ({len(seatings) * args.games} total, jobs={args.jobs}) -> {out_np.relative_to(SOL)}")
        tasks = [(list(s), seed, "4p", str(out_np), _pair_id(list(s)))
                 for s in seatings for seed in range(args.games)]
        counts = _dispatch(tasks, args.jobs)
        print("    -> agent win counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        touched.add((out_np, 4))

    print()
    for out_np, n in sorted(touched, key=lambda x: x[1]):
        ng, nm = _build_index(out_np, n)
        print(f"DONE {n}p: {nm} matchups, {ng} games -> {(out_np / 'INDEX.json').relative_to(SOL)}")


if __name__ == "__main__":
    main()
