"""exp25-4p IL - COMBINED 2p+4p imitation on the v41 4p arch (pointer self=hold, LOCAL emb pointer query
+ econ-CNN trunk; frac = SimpleFracMLP([emb[s] || emb_tid || gemb]) = 3E SOURCE+TARGET+GLOBAL, NO edge,
NO coordination; E=128, 6-layer, 8 heads). Same 4p env/features/deploy as exp24-4p; ONLY model arch (v37->v41).

exp24-4p (user 2026-06-18): the 4p analog of the 2p exp24 IL+self-play. IL pretrains the v37 4p model
(static30 / ts(50,10) / glob34 / econ(50,8) / edge6, C4 seat-canonical) on a COMBINED dataset:
  * NATIVE 4p episodes (build_dataset_4p.py, min-of-4 score >= 1400): ALL 4 seats trained, me = seat.
  * 2p episodes (build_dataset_2p.py, min-of-2 score >= 1400): viewed as a 4p board with TWO players
    already eliminated -> owner {0,1} remapped to 4p seats {0,3} (the antipodal/diagonal corners; each
    2p player sees the other in the q2 antipodal channel). me in {0,3}. This is the "2p == 4p with 2
    eliminated" case the user described; both real seats are trained.
`--data` is a COMMA-SEPARATED list of dataset dirs (each manifest.csv + per-episode npz). Per episode we
detect 2p vs 4p from len(meta["seats"]) and pick the seat loop + remap accordingly (per-sample, so a
batch can freely mix 2p and 4p rows). The architecture/loss are byte-identical to the 2p exp24 v36 IL.

User 2026-06-14: try IL on exp22 v1's CONTINUOUS HEAD. exp22 = exp21 edge body (pointer self=hold +
edge-bias trunk & pointer-logit term) + a 2nd tree `frac` = CoordFracGauss(ctx, emb[tid], edge[tid],
intend) -> (mu, sigma) for f = clip(N(mu,sigma),0,1) (f=0 HOLD, f=1 ALL-IN; the extremes are CDF atoms).
So IL learns the expert's ACTUAL ship fraction (incl the ~15% partial commits), not just all-in (v4).

LOSS (2-tree {net, frac}):
  pointer CE (self=hold, launch_w-weighted; same as v4)  +  w_frac * clipped-Gaussian fraction NLL.
  frac NLL = -cg_logp(lab_f, mu, sigma) on LAUNCHED planets (teacher-forced on lab_tid). lab_f = ships/gar
  kept EXACT (all-in -> f=1 -> atom1 log_ndtr((mu-1)/sigma); partials -> Gaussian interior). cg_logp is
  exp22's verbatim (tb.cg_logp).
Data: ../v1/data/v1 (arch-agnostic raw-state npz). wscore weighting, FROM SCRATCH, 3 epochs.
Ckpt = 2-tree {net, frac} msgpack -> exp22 JAX eval (train.greedy_action/eval_h2h) works as-is.
NOTE: exp22's numpy rl_infer is STALE -> a forward_sim vsOL curve needs the edge+gauss numpy port (later).
"""
import argparse, csv, json, math, os, queue, sys, threading, time

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "jax_env"))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization as fser

import train as tb                                    # exp22: edge_features + cg_logp (clipped-Gaussian)
from state import JaxState
from constants import N_PLAYERS                        # 4
from env import basic_features, EPISODE_STEPS, _forecast
from targeting import reach_solve_static
from model import OrbitNet19, SimpleFracMLP

P_MAX, F_MAX, L = 48, 512, 64
F_EPS = 1e-3


class ILData:
    """Episodes pre-padded to P_MAX at load; batch() is thread-safe (caller passes its own rng).
    Arch-agnostic: stores raw state arrays + executed-launch labels (incl lab_f = ships/gar)."""

    def __init__(self, data_dirs, val_days, weight_mode, min_train_score=0.0):
        self.min_train_score = min_train_score
        if isinstance(data_dirs, str):
            data_dirs = [d.strip() for d in data_dirs.split(",") if d.strip()]
        self.eps = []
        idx_ep, idx_t, idx_seat, idx_w, is_val = [], [], [], [], []
        val_set = set(d.strip() for d in val_days.split(",")) if val_days else set()
        n2p = n4p = 0
        for data_dir in data_dirs:
          man = list(csv.DictReader(open(os.path.join(data_dir, "manifest.csv"))))
          for r in man:
            z = np.load(os.path.join(data_dir, f"{r['episode_id']}.npz"))
            P = z["p_id"].shape[0]
            T = z["p_mask"].shape[0]
            e = {"P": P, "T": T, "av": np.float32(z["av"]),
                 "path_tab_x": z["path_tab_x"], "path_tab_y": z["path_tab_y"],
                 "path_len": z["path_len"].astype(np.int32),
                 "t_step": z["t_step"], "f_off": z["f_off"],
                 "f_owner": z["f_owner"].astype(np.int32), "f_ships": z["f_ships"],
                 "f_target": z["f_target"].astype(np.int32), "f_arrival": z["f_arrival"],
                 "meta": json.loads(bytes(z["meta"]).decode())}
            def padT(a, fill=0, dt=None):
                o = np.full((T, P_MAX), fill, dt or a.dtype); o[:, :P] = a; return o
            def padP(a, fill=0, dt=None):
                o = np.full(P_MAX, fill, dt or a.dtype); o[:P] = a; return o
            e["p_owner"] = padT(z["p_owner"].astype(np.int32), -1)
            e["p_ships"] = padT(z["p_ships"])
            e["p_x"] = padT(z["p_x"]); e["p_y"] = padT(z["p_y"])
            e["p_mask"] = padT(z["p_mask"], False)
            e["p_cidx"] = padT(z["p_cidx"].astype(np.int32))
            e["p_pathref"] = padT(z["p_pathref"].astype(np.int32), -1)
            e["p_radius"] = padP(z["p_radius"]); e["p_prod"] = padP(z["p_prod"].astype(np.int32))
            e["p_is_comet"] = padP(z["p_is_comet"], False)
            e["p_is_orbiting"] = padP(z["p_is_orbiting"], False)
            e["p_orbital_r"] = padP(z["p_orbital_r"])
            orb = e["p_is_orbiting"][None, :] & e["p_mask"]
            oa = np.zeros((T, P_MAX), np.float32)
            oa[orb] = np.arctan2(e["p_y"][orb] - 50.0, e["p_x"][orb] - 50.0)
            e["p_orbital_a"] = oa
            lab = z["labels"]
            by_ts = {}
            for row in lab:
                by_ts.setdefault((int(row[0]), int(row[1])), []).append(row)
            e["lab_by_ts"] = {k: np.array(v, np.int32) for k, v in by_ts.items()}
            e["is_4p"] = (len(e["meta"]["seats"]) == N_PLAYERS)   # native 4p vs 2p (2-seat) episode
            ei = len(self.eps)
            self.eps.append(e)
            if e["is_4p"]:
                n4p += 1
            else:
                n2p += 1
            val = r["date"] in val_set
            seats = range(N_PLAYERS) if e["is_4p"] else (0, 1)   # 4p: all seats; 2p: the 2 real seats
            for seat in seats:
                rew = e["meta"]["seats"][seat]["reward"]
                sc = e["meta"]["seats"][seat]["score"] or 1200.0
                w = {"winner": 1.0 if (rew or 0) > 0 else 0.25,
                     "score": max(0.1, (sc - 1200.0) / 200.0),
                     "wscore": (1.0 if (rew or 0) > 0 else 0.3) * max(0.2, (sc - 1200.0) / 200.0),
                     "winner_only": (max(0.2, (sc - 1200.0) / 200.0) if (rew or 0) > 0 else 0.0),
                     }.get(weight_mode, 1.0)
                if sc < self.min_train_score:
                    w = 0.0
                for t in range(T):
                    idx_ep.append(ei); idx_t.append(t); idx_seat.append(seat)
                    idx_w.append(w); is_val.append(val)
        idx = np.array([idx_ep, idx_t, idx_seat], np.int32).T
        w = np.array(idx_w, np.float32); v = np.array(is_val, bool)
        self.train_idx, tw = idx[~v], w[~v]
        self.train_p = tw / tw.sum()
        self.val_idx = idx[v]
        print(f"DATA episodes={len(self.eps)} (4p={n4p} 2p={n2p}) train_samples={len(self.train_idx):,} "
              f"val_samples={len(self.val_idx):,} weight_mode={weight_mode}", flush=True)

    def batch(self, B, rng, val=False):
        idx = self.val_idx if val else self.train_idx
        if val:
            pick = idx[rng.choice(len(idx), size=B, replace=False)]
        else:
            pick = idx[rng.choice(len(idx), size=B, replace=True, p=self.train_p)]
        out = dict(
            p_owner=np.empty((B, P_MAX), np.int32), p_ships=np.empty((B, P_MAX), np.int32),
            p_x=np.empty((B, P_MAX), np.float32), p_y=np.empty((B, P_MAX), np.float32),
            p_mask=np.empty((B, P_MAX), bool), p_radius=np.empty((B, P_MAX), np.float32),
            p_prod=np.empty((B, P_MAX), np.int32), p_is_comet=np.empty((B, P_MAX), bool),
            p_is_orbiting=np.empty((B, P_MAX), bool), p_orbital_r=np.empty((B, P_MAX), np.float32),
            p_orbital_a=np.empty((B, P_MAX), np.float32),
            p_comet_path_x=np.zeros((B, P_MAX, L), np.float32),
            p_comet_path_y=np.zeros((B, P_MAX, L), np.float32),
            p_comet_idx=np.empty((B, P_MAX), np.int32), p_comet_len=np.zeros((B, P_MAX), np.int32),
            f_owner=np.zeros((B, F_MAX), np.int32), f_ships=np.zeros((B, F_MAX), np.int32),
            f_target=np.full((B, F_MAX), -1, np.int32), f_arrival=np.full((B, F_MAX), -1, np.int32),
            f_mask=np.zeros((B, F_MAX), bool),
            step=np.empty(B, np.int32), av=np.empty(B, np.float32),
            me=np.empty(B, np.int32), outcome=np.empty(B, np.float32),
            lab_tid=np.zeros((B, P_MAX), np.int32), lab_mask=np.zeros((B, P_MAX), bool),
            lab_launch=np.zeros((B, P_MAX), bool), lab_f=np.ones((B, P_MAX), np.float32))
        for b, (ei, t, seat) in enumerate(pick):
            e = self.eps[ei]; t = int(t); seat = int(seat)
            for f in ("p_owner", "p_ships", "p_x", "p_y", "p_mask", "p_orbital_a"):
                out[f][b] = e[f][t]
            out["p_comet_idx"][b] = e["p_cidx"][t]
            for f in ("p_radius", "p_prod", "p_is_comet", "p_is_orbiting", "p_orbital_r"):
                out[f][b] = e[f]
            ref = e["p_pathref"][t]
            live = np.where(ref >= 0)[0]
            if live.size:
                out["p_comet_path_x"][b, live] = e["path_tab_x"][ref[live]]
                out["p_comet_path_y"][b, live] = e["path_tab_y"][ref[live]]
                out["p_comet_len"][b, live] = e["path_len"][ref[live]]
            a, c = int(e["f_off"][t]), int(e["f_off"][t + 1])
            nf = min(c - a, F_MAX)
            out["f_owner"][b, :nf] = e["f_owner"][a:a + nf]
            out["f_ships"][b, :nf] = e["f_ships"][a:a + nf]
            out["f_target"][b, :nf] = e["f_target"][a:a + nf]
            out["f_arrival"][b, :nf] = e["f_arrival"][a:a + nf]
            out["f_mask"][b, :nf] = True
            out["step"][b] = e["t_step"][t]; out["av"][b] = e["av"]
            # me = seat for native 4p; for 2p data we view it as a 4p board with seats 1,2 ELIMINATED:
            # owner 1 -> 4p SEAT 3 (the antipodal/diagonal corner of seat 0 under the C4 layout). So
            # 2p `me` is in {0,3}. Labels/owned use the ORIGINAL data seat (slot-indexed; the owner
            # VALUE remap below does not move any slot, so per-slot labels stay valid).
            out["me"][b] = seat if e["is_4p"] else (3 if seat == 1 else 0)
            out["outcome"][b] = float(e["meta"]["seats"][seat]["reward"] or 0.0)
            owned = (e["p_owner"][t] == seat) & e["p_mask"][t] & (e["p_ships"][t] > 0)
            out["lab_tid"][b] = np.arange(P_MAX, dtype=np.int32)        # default = self (HOLD)
            out["lab_mask"][b] = owned
            rows = e["lab_by_ts"].get((t, seat))
            if rows is not None:
                src, tid, ships, gar = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
                out["lab_tid"][b, src] = tid
                out["lab_launch"][b, src] = True
                # lab_f = ships/garrison kept EXACT: all-in -> 1.0 (CDF atom1); partial -> interior.
                out["lab_f"][b, src] = np.clip(ships.astype(np.float32) / np.maximum(gar.astype(np.float32), 1.0),
                                               F_EPS, 1.0)
            if not e["is_4p"]:
                # 2p->4p owner remap (VALUE-only, slot-preserving), THIS sample only: owner 1 -> seat 3
                # (seats 1,2 stay empty). `me` set to {0,3} above so is_mine/role channels in
                # env.basic_features (which key off p_owner == me) stay consistent. Native-4p rows are
                # left untouched -> a batch can freely mix 2p-remapped and native-4p samples.
                po = out["p_owner"][b]; po[po == 1] = 3
                fo = out["f_owner"][b]; fm = out["f_mask"][b]
                fo[(fo == 1) & fm] = 3
        return out


class Prefetcher:
    def __init__(self, data, B, seed, nthreads=3, depth=4):
        self.q = queue.Queue(maxsize=depth)
        self.stop = False
        def work(k):
            rng = np.random.default_rng(seed + 1000 + k)
            while not self.stop:
                self.q.put(data.batch(B, rng))
        self.threads = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(nthreads)]
        for t in self.threads:
            t.start()
    def get(self):
        return self.q.get()


def to_state(d):
    B = d["p_owner"].shape[0]
    zf = lambda: jnp.zeros((B, F_MAX), jnp.float32)
    zi = lambda: jnp.zeros((B, F_MAX), jnp.int32)
    return JaxState(
        p_id=jnp.zeros((B, P_MAX), jnp.int32), p_owner=jnp.asarray(d["p_owner"]),
        p_x=jnp.asarray(d["p_x"]), p_y=jnp.asarray(d["p_y"]), p_radius=jnp.asarray(d["p_radius"]),
        p_ships=jnp.asarray(d["p_ships"]), p_prod=jnp.asarray(d["p_prod"]),
        p_mask=jnp.asarray(d["p_mask"]), p_is_comet=jnp.asarray(d["p_is_comet"]),
        p_is_orbiting=jnp.asarray(d["p_is_orbiting"]), p_orbital_r=jnp.asarray(d["p_orbital_r"]),
        p_orbital_a=jnp.asarray(d["p_orbital_a"]), p_comet_path_x=jnp.asarray(d["p_comet_path_x"]),
        p_comet_path_y=jnp.asarray(d["p_comet_path_y"]), p_comet_idx=jnp.asarray(d["p_comet_idx"]),
        p_comet_len=jnp.asarray(d["p_comet_len"]),
        f_owner=jnp.asarray(d["f_owner"]), f_x=zf(), f_y=zf(), f_angle=zf(),
        f_ships=jnp.asarray(d["f_ships"]), f_from=zi(), f_speed=zf(),
        f_mask=jnp.asarray(d["f_mask"]), f_target=jnp.asarray(d["f_target"]),
        f_arrival=jnp.asarray(d["f_arrival"]),
        sched_px=jnp.zeros((B, 5, 4, L), jnp.float32), sched_py=jnp.zeros((B, 5, 4, L), jnp.float32),
        sched_len=jnp.zeros((B, 5, 4), jnp.int32), sched_ships=jnp.zeros((B, 5), jnp.int32),
        comet_base_id=jnp.zeros(B, jnp.int32),
        step=jnp.asarray(d["step"]), av=jnp.asarray(d["av"]),
        next_fleet_id=jnp.zeros(B, jnp.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "4p"),
                    help="COMMA-SEPARATED dataset dirs (each a manifest.csv + per-episode npz). Mix "
                         "native-4p dirs and 2p dirs freely; 2p episodes are auto-remapped to seats "
                         "{0,3}. e.g. 'data/4p,<solution-root>/2p/experiments/experiment-020-imitate/v1/data/v1'")
    ap.add_argument("--val_days", default="")
    ap.add_argument("--weight_mode", default="wscore", choices=["uniform", "winner", "score", "wscore", "winner_only"])
    ap.add_argument("--launch_w", type=float, default=6.0)   # up-weight launch-source rows (~14% base rate)
    ap.add_argument("--w_frac", type=float, default=0.5)     # weight on the clipped-Gaussian fraction NLL
    ap.add_argument("--w_value", type=float, default=0.0)    # value-head MSE (off by default)
    ap.add_argument("--min_train_score", type=float, default=0.0)
    ap.add_argument("--updates", type=int, default=119000)   # ~3 epochs of data/v1 (~10.15M states / 256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_final", type=float, default=1e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--E", type=int, default=128)        # exp25 v41-arch: E=128
    ap.add_argument("--n_layers", type=int, default=6)   # 6-layer trunk (user-confirmed 2026-06-18)
    ap.add_argument("--n_heads", type=int, default=4)    # 4 heads -> d=32 (user-confirmed: 8->4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loader_threads", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=8000)
    ap.add_argument("--save_dir", default=os.path.join(_HERE, "checkpoints"))
    ap.add_argument("--init_from", default="")
    args = ap.parse_args()
    print("DEVICES:", jax.devices(), flush=True)

    data = ILData(args.data, args.val_days, args.weight_mode, min_train_score=args.min_train_score)

    net = OrbitNet19(E=args.E, n_layers=args.n_layers, n_heads=args.n_heads)
    frac = SimpleFracMLP(E=args.E, n_heads=args.n_heads)
    rng = jax.random.PRNGKey(args.seed)
    d0 = data.batch(2, np.random.default_rng(args.seed))
    st0 = jax.tree_util.tree_map(lambda x: x[0], to_state(d0))
    fc0 = _forecast(st0)
    R0, A0, T0, Rg0 = reach_solve_static(st0)
    static0, ts0, glob0, m0, econ0 = basic_features(st0, 0, fc=fc0)
    reach0 = Rg0 & ((st0.p_owner == 0) & st0.p_mask)[:, None]
    _, _, _, edge0 = tb.edge_features(st0, fc=fc0, lead=(R0, A0, T0))
    print("FEATURE_DIMS static=%s ts=%s glob=%s econ_curves=%s edge=%s (4p IL: target 30/(50,10)/34/(50,8)/(P,P,6))"
          % (tuple(static0.shape), tuple(ts0.shape), tuple(glob0.shape),
             tuple(econ0.shape), tuple(edge0.shape)), flush=True)
    rng, ki, ki2 = jax.random.split(rng, 3)
    net_p = net.init(ki, static0, ts0, glob0, reach0, m0, edge0, econ0)
    P0 = st0.p_owner.shape[0]; ar0 = jnp.arange(P0)
    tgt0, emb0, gemb0, _b0, _v0 = net.apply(net_p, static0, ts0, glob0, reach0, m0, edge0, econ0)
    tid0 = jnp.argmax(tgt0, -1)
    # v41: frac = SimpleFracMLP([emb[s] || emb_tid || gemb]) = 3E (SOURCE+TARGET+GLOBAL); NO edge.
    frac_p = frac.init(ki2, emb0, emb0[tid0], gemb0)
    params = {"net": net_p, "frac": frac_p}
    if args.init_from:
        with open(args.init_from, "rb") as fh:
            params = jax.tree_util.tree_map(jnp.asarray, fser.msgpack_restore(fh.read()))
        print("WARM-START from " + args.init_from, flush=True)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print("MODEL_PARAMS %d  (exp20 v9 IL on exp22 V3 11-d multi-op-point edge+gauss: pointer self=hold + clipped-Gaussian frac; "
          "loss = pointer CE + %s*fracNLL, launch_w=%s, weight_mode=%s)"
          % (n_params, args.w_frac, args.launch_w, args.weight_mode), flush=True)

    total = max(1, args.updates)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, max(1, int(args.warmup_frac * total)),
                                               total, args.lr_final)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    opt_state = opt.init(params)
    LW = args.launch_w

    def one_loss(params, st, me, lab_tid, lab_mask, lab_launch, lab_f, outcome):
        fc = _forecast(st)
        R, _A, _T, Rg = reach_solve_static(st)
        static, ts, glob, m, econ = basic_features(st, me, fc=fc)
        is_mine = (st.p_owner == me) & st.p_mask
        reach = Rg & is_mine[:, None]
        _, _, _, edge = tb.edge_features(st, fc=fc, lead=(R, _A, _T))
        tgt, emb, gemb, _b, v = net.apply(params["net"], static, ts, glob, reach, m, edge, econ)
        P = tgt.shape[0]; ar = jnp.arange(P)
        lsm = jax.nn.log_softmax(tgt, -1)
        pick = lsm[ar, lab_tid]                                   # self=hold always legal; target legal iff reach
        legal = pick > -1e8
        use = lab_mask & legal
        w = jnp.where(lab_launch, LW, 1.0)
        ce_s = -jnp.sum(jnp.where(use, w * pick, 0.0))
        n_w = jnp.sum(jnp.where(use, w, 0.0))
        pred = jnp.argmax(tgt, -1)
        usl = use & lab_launch; n_l = usl.sum()
        ush = use & (~lab_launch); n_h = ush.sum()
        lacc_s = jnp.sum(jnp.where(usl, pred == lab_tid, 0.0))    # launched: exact target match
        hacc_s = jnp.sum(jnp.where(ush, pred == ar, 0.0))        # held: chose self (hold)
        lrec_s = jnp.sum(jnp.where(usl, pred != ar, 0.0))        # launched: chose to launch at all
        n_ill = jnp.sum(lab_mask & lab_launch & ~legal)
        # ---- CONTINUOUS frac head (teacher-forced on lab_tid): clipped-Gaussian NLL on launches ----
        # v41: frac = SimpleFracMLP([emb[s] || emb_tid || gemb]) = 3E (SOURCE+TARGET+GLOBAL); NO edge.
        emb_tid = emb[lab_tid]
        mu, sigma = frac.apply(params["frac"], emb, emb_tid, gemb)
        flogp = tb.cg_logp(lab_f, mu, sigma)
        frac_nll_s = -jnp.sum(jnp.where(usl, flogp, 0.0))
        fmae_s = jnp.sum(jnp.where(usl, jnp.abs(jnp.clip(mu, 0.0, 1.0) - lab_f), 0.0))
        vmse = (v - outcome) ** 2
        return ce_s, n_w, lacc_s, n_l, hacc_s, n_h, lrec_s, frac_nll_s, fmae_s, vmse, n_ill

    def loss_fn(params, st, me, lt, lm, ll, lf, oc):
        outs = jax.vmap(lambda s, a, b, c, dd, e, g: one_loss(params, s, a, b, c, dd, e, g),
                        in_axes=(0,) * 7)(st, me, lt, lm, ll, lf, oc)
        ce_s, n_w, lacc_s, n_l, hacc_s, n_h, lrec_s, frac_nll_s, fmae_s, vmse, n_ill = outs
        NW = jnp.clip(n_w.sum(), 1.0); NL = jnp.clip(n_l.sum(), 1.0); NH = jnp.clip(n_h.sum(), 1.0)
        ce = ce_s.sum() / NW
        frac_nll = frac_nll_s.sum() / NL
        vm = vmse.mean()
        L_ = ce + args.w_frac * frac_nll + args.w_value * vm
        aux = (ce, lacc_s.sum() / NL, hacc_s.sum() / NH, lrec_s.sum() / NL,
               frac_nll, fmae_s.sum() / NL, vm, n_ill.sum())
        return L_, aux

    @jax.jit
    def step_fn(params, opt_state, st, me, lt, lm, ll, lf, oc):
        (L_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, st, me, lt, lm, ll, lf, oc)
        upd, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, L_, aux

    os.makedirs(args.save_dir, exist_ok=True)
    mf = open(os.path.join(args.save_dir, "metrics_il.csv"), "w", newline="")
    mw = csv.writer(mf)
    mw.writerow(["update", "loss", "ce", "launch_acc", "hold_acc", "launch_recall",
                 "frac_nll", "frac_mae", "vmse", "illegal", "sps", "dt20_s", "elapsed_s"])

    def save_ckpt(u):
        with open(os.path.join(args.save_dir, "ckpt_u%05d.msgpack" % u), "wb") as fh:
            fh.write(fser.msgpack_serialize(jax.device_get(params)))
        json.dump({"update": int(u), "n_params": int(n_params), "il": True, "trainer": "train_il_v5",
                   "arch": "exp24-4p v37 econ-CNN pointer self=hold + 6-d ALL-IN clipped-Gaussian frac (2-tree); "
                           "combined 2p(remap->{0,3})+native-4p IL", "n_players": int(N_PLAYERS), "data": args.data,
                   "weight_mode": args.weight_mode, "launch_w": args.launch_w, "w_frac": args.w_frac,
                   "args": vars(args)}, open(os.path.join(args.save_dir, "meta.json"), "w"), default=str)

    pf = Prefetcher(data, args.batch, args.seed, nthreads=args.loader_threads)
    epoch_b = len(data.train_idx) / args.batch
    print("IL TRAIN v5 (exp22 edge+gauss): B=%d updates=%d (~%.2f epochs; %.0f batches/epoch)"
          % (args.batch, args.updates, args.updates / epoch_b, epoch_b), flush=True)
    t0 = time.time(); t_prev = t0
    for u in range(args.updates):
        d = pf.get()
        st = to_state(d)
        params, opt_state, L_, aux = step_fn(
            params, opt_state, st, jnp.asarray(d["me"]), jnp.asarray(d["lab_tid"]),
            jnp.asarray(d["lab_mask"]), jnp.asarray(d["lab_launch"]), jnp.asarray(d["lab_f"]),
            jnp.asarray(d["outcome"]))
        if u % 20 == 0 or u == args.updates - 1:
            ce, la_, ha_, lr_, fnll, fmae, vm, nil = [float(x) for x in jax.device_get(aux)]
            now = time.time(); elapsed = now - t0; dt20 = now - t_prev; t_prev = now  # dt20 = wall per ~20 upd
            sps = args.batch * (u + 1) / elapsed
            lr_now = float(sched(u))                                                  # current scheduled lr
            pct = 100.0 * (u + 1) / args.updates
            eta_s = (args.updates - 1 - u) * (dt20 / 20.0)                            # remaining at current rate
            el = "%dm%02ds" % (int(elapsed) // 60, int(elapsed) % 60)
            eta = "%dh%02dm" % (int(eta_s) // 3600, (int(eta_s) % 3600) // 60)
            samp = (u + 1) * args.batch                                               # samples trained
            print("u%6d/%d %4.1f%% | %.1fM/%.1fM smp | lr %.2e | L %+6.3f | ce %5.3f | lacc %.3f"
                  " | hacc %.3f | lrec %.3f | fNLL %+5.3f | fMAE %.3f | vmse %5.3f | ill %2.0f"
                  " | %5.0f sps | dt20 %5.1fs | el %s | ETA %s"
                  % (u, args.updates, pct, samp / 1e6, args.updates * args.batch / 1e6, lr_now,
                     float(L_), ce, la_, ha_, lr_, fnll, fmae, vm, nil, sps, dt20, el, eta), flush=True)
            mw.writerow([u, "%.6f" % lr_now, "%.4f" % float(L_), "%.4f" % ce, "%.4f" % la_, "%.4f" % ha_,
                         "%.4f" % lr_, "%.4f" % fnll, "%.4f" % fmae, "%.4f" % vm, "%.0f" % nil,
                         "%.0f" % sps, "%.1f" % dt20, "%d" % int(elapsed)]); mf.flush()
        if args.save_every and (u % args.save_every == 0 or u == args.updates - 1) and u > 0:
            save_ckpt(u)
    pf.stop = True
    mf.close()
    print("PIPELINE_OK", flush=True)


if __name__ == "__main__":
    main()
