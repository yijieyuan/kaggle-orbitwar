"""build_dataset.py — official replays -> exp20 IL shards (one npz per episode + dataset manifest).

Design (user 2026-06-12):
  PRECOMPUTED (expensive / shrinks data): obs->arr state reconstruction (reuses the deploy stack's
  AgentState + _obs_to_arr verbatim — same f_target/f_arrival the RL features see), executed-action
  matching + tid labels (first-hit of the actually-spawned fleet), decided-tail turn filtering,
  per-seat metadata (LB scores, win/loss).
  LEFT FOR TRAINING TIME (cheap on GPU): feature tensors (jax_env basic_features on the stored
  state arrays), hold labels (owned planets minus launch labels), sampling weights (from manifest).

Per-episode npz layout (global slot frame, slots keyed by official planet id, comet id reuse OK):
  static : p_id (P,), p_radius, p_prod, p_is_comet, p_is_orbiting, p_orbital_r   (P,)
  paths  : path_tab_x/y (n_path, L=64), path_len (n_path,)  [deduped comet paths]
  turns  : t_step (T,), p_mask/p_owner/p_ships (T,P), p_x/p_y (T,P) f32,
           p_cidx (T,P) i16, p_pathref (T,P) i16 (-1 = not a live comet), share0 (T,) f32
  fleets : f_off (T+1,) i64 + f_owner/f_ships/f_target/f_arrival (concat over turns)
  labels : lab_t, lab_seat, lab_src, lab_tid, lab_ships, lab_gar  (n_lab,)  [EXECUTED launches only,
           tid = real first-hit slot of the spawned fleet; multi-launch-per-planet keeps largest]
  meta   : json string (episode_id, date, per-seat score/reward/submission, cut info, counts)

Usage: python build_dataset.py --days 2026-06-02,2026-06-03,... [--min_score 1400] [--limit 20]
       [--out data/v1] [--workers 28]
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, _HERE)

OR_DIR = os.path.join(_ROOT, "official_replay")
L = 64                                   # MAX comet path slots (matches agent.py _L)
F_ID, F_OWNER, F_X, F_Y, F_ANG, F_FROM, F_SHIPS = range(7)   # fleet tuple (debug.md)


def load_agent_meta():
    """episode_agents.csv -> {episode_id: {seat_index: (reward, updated_score, submission_id)}}"""
    out = {}
    with open(os.path.join(OR_DIR, "episode_agents.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                ep = int(r["EpisodeId"]); idx = int(r["Index"])
                out.setdefault(ep, {})[idx] = (float(r["Reward"] or 0), float(r["UpdatedScore"] or 0),
                                               int(r["SubmissionId"]))
            except (ValueError, KeyError):
                continue
    return out


def process_episode(args_tuple):
    fp, seats_meta, date, decided_share, decided_turns, out_dir = args_tuple
    from obs_state import _obs_to_arr        # deploy-verbatim reconstruction (no weight load)
    from engine import AgentState
    try:
        ep = json.load(open(fp))
        steps = ep["steps"]
        if len(steps) < 30:
            return ("skip_short", os.path.basename(fp))
        epid = int(os.path.basename(fp).split(".")[0])   # filename IS the episode id (json 'id' is a uuid)

        st = AgentState()
        arrs, pid_maps, fleet_ids = [], [], []
        hit_by_fid = {}                      # engine pops vanished fleets' hits — snapshot them per turn
        for t in range(len(steps)):
            obs = steps[t][0]["observation"]
            st.update(obs)
            arr, p_id = _obs_to_arr(obs, st)
            arrs.append(arr); pid_maps.append({int(p): i for i, p in enumerate(p_id)})
            fleet_ids.append([f[F_ID] for f in (obs.get("fleets") or [])])
            hit_by_fid.update(st.fleet_hit)  # overwrite keeps comet-spawn recomputes current

        # ---- global slot frame (official ids; comet-id reuse maps to the same slot by design) ----
        all_ids = sorted({pid for m in pid_maps for pid in m})
        gslot = {pid: i for i, pid in enumerate(all_ids)}
        P = len(all_ids)
        T = len(steps)

        # ---- labels: action at step t acts FROM state t-1, fleet appears in frame t (debug.md) ----
        labels = []
        n_multi = n_invalid = n_nonplanet = 0
        for t in range(1, T):
            prev_arr, prev_map = arrs[t - 1], pid_maps[t - 1]
            obs_t = steps[t][0]["observation"]
            new_fids = set(fleet_ids[t]) - set(fleet_ids[t - 1])
            fl_t = {f[F_ID]: f for f in (obs_t.get("fleets") or [])}
            cur_map = pid_maps[t]
            for seat in (0, 1):
                acts = steps[t][seat].get("action") or []
                per_src = {}
                for a in acts:
                    try:
                        pid, _ang, ships = int(a[0]), float(a[1]), int(a[2])
                    except (TypeError, ValueError, IndexError):
                        continue
                    sl = prev_map.get(pid)
                    if sl is None:
                        n_invalid += 1; continue
                    g = int(prev_arr["p_ships"][sl])
                    if not (0 < ships <= g) or int(prev_arr["p_owner"][sl]) != seat:
                        n_invalid += 1; continue          # env drops over-requests entirely
                    # match the actually-spawned fleet (executed launches only)
                    fid = next((i for i in new_fids
                                if fl_t[i][F_OWNER] == seat and fl_t[i][F_FROM] == pid
                                and fl_t[i][F_SHIPS] == ships), None)
                    if fid is None:
                        n_invalid += 1; continue
                    hit = hit_by_fid.get(fid)             # snapshot taken at frame t (see state pass)
                    if not hit or hit["kind"] != "planet" or hit["planet"] is None:
                        n_nonplanet += 1; continue        # sun/oob shots: junk, not a learnable label
                    tid_sl = cur_map.get(hit["planet"])
                    if tid_sl is None:
                        n_nonplanet += 1; continue
                    key = (seat, pid)
                    if key in per_src:                    # multi-launch same planet: keep largest
                        n_multi += 1
                        if ships <= per_src[key][2]:
                            continue
                    per_src[key] = (t - 1, seat, ships, g, tid_sl)
                for (seat_, pid_), (tm1, se, ships, g, tid_sl) in per_src.items():
                    tid_pid = next(p for p, i in cur_map.items() if i == tid_sl)
                    labels.append((tm1, se, gslot[pid_], gslot[tid_pid], ships, g))

        # ---- per-turn stacked arrays in the global frame + share curve ----
        p_mask = np.zeros((T, P), bool); p_owner = np.full((T, P), -1, np.int8)
        p_ships = np.zeros((T, P), np.int32)
        p_x = np.zeros((T, P), np.float32); p_y = np.zeros((T, P), np.float32)
        p_cidx = np.zeros((T, P), np.int16); p_pathref = np.full((T, P), -1, np.int16)
        share0 = np.zeros(T, np.float32)
        path_tab = {}                                      # hash -> (path_id, x(L,), y(L,), len)
        f_off = [0]; f_owner = []; f_ships = []; f_target = []; f_arrival = []
        static_done = {}
        p_radius = np.zeros(P, np.float32); p_prod = np.zeros(P, np.int16)
        p_is_comet = np.zeros(P, bool); p_is_orb = np.zeros(P, bool); p_orb_r = np.zeros(P, np.float32)

        for t, (arr, pmap) in enumerate(zip(arrs, pid_maps)):
            inv = {i: pid for pid, i in pmap.items()}
            n = arr["p_x"].shape[0]
            gs = np.array([gslot[inv[i]] for i in range(n)], np.int64)
            p_mask[t, gs] = arr["p_mask"]
            p_owner[t, gs] = arr["p_owner"].astype(np.int8)
            p_ships[t, gs] = arr["p_ships"]
            p_x[t, gs] = arr["p_x"]; p_y[t, gs] = arr["p_y"]
            p_cidx[t, gs] = arr["p_comet_idx"].astype(np.int16)
            for i in range(n):
                g = int(gs[i])
                if g not in static_done:
                    p_radius[g] = arr["p_radius"][i]; p_prod[g] = arr["p_prod"][i]
                    p_is_comet[g] = arr["p_is_comet"][i]; p_is_orb[g] = arr["p_is_orbiting"][i]
                    p_orb_r[g] = arr["p_orbital_r"][i]
                    static_done[g] = True
                if arr["p_is_comet"][i] and arr["p_comet_len"][i] > 0:
                    px = arr["p_comet_path_x"][i]; py = arr["p_comet_path_y"][i]
                    h = hash((px.tobytes(), py.tobytes()))
                    if h not in path_tab:
                        path_tab[h] = (len(path_tab), px.copy(), py.copy(), int(arr["p_comet_len"][i]))
                    p_pathref[t, g] = path_tab[h][0]
            # fleets (target slots remapped into the global frame)
            ft = arr["f_target"]
            ft_g = np.array([gslot[inv[s]] if s >= 0 else -1 for s in ft], np.int16) if ft.size else np.zeros(0, np.int16)
            f_owner.append(arr["f_owner"].astype(np.int8)); f_ships.append(arr["f_ships"])
            f_target.append(ft_g); f_arrival.append(arr["f_arrival"])
            f_off.append(f_off[-1] + arr["f_owner"].shape[0])
            ts0 = p_ships[t][(p_owner[t] == 0) & p_mask[t]].sum() + arr["f_ships"][arr["f_owner"] == 0].sum()
            ts1 = p_ships[t][(p_owner[t] == 1) & p_mask[t]].sum() + arr["f_ships"][arr["f_owner"] == 1].sum()
            share0[t] = float(ts0) / max(float(ts0 + ts1), 1.0)

        # ---- decided-tail cut: share beyond threshold for K consecutive turns -> drop the rest ----
        cut = T
        streak = 0
        for t in range(T):
            if max(share0[t], 1 - share0[t]) >= decided_share:
                streak += 1
                if streak >= decided_turns:
                    cut = t - decided_turns + 1
                    break
            else:
                streak = 0
        cut = max(cut, 30)                                 # never cut into the opening
        labels = [lb for lb in labels if lb[0] < cut]
        if not labels:
            return ("skip_nolabels", os.path.basename(fp))

        lab = np.array(labels, np.int32)                   # (n,6): t, seat, src, tid, ships, gar
        n_path = max(len(path_tab), 1)
        ptx = np.zeros((n_path, L), np.float32); pty = np.zeros((n_path, L), np.float32)
        plen = np.zeros(n_path, np.int16)
        for _h, (pi, px, py, ln) in path_tab.items():
            ptx[pi], pty[pi], plen[pi] = px, py, ln
        fo = np.concatenate(f_owner) if f_owner else np.zeros(0, np.int8)
        fs = np.concatenate(f_ships) if f_ships else np.zeros(0, np.int32)
        ftg = np.concatenate(f_target) if f_target else np.zeros(0, np.int16)
        fa = np.concatenate(f_arrival) if f_arrival else np.zeros(0, np.int32)
        foff = np.array(f_off, np.int64)

        meta = {"episode_id": epid, "date": date, "P": P, "T": T, "cut": int(cut),
                "seats": [{"reward": seats_meta.get(s, (None, None, None))[0],
                           "score": seats_meta.get(s, (None, None, None))[1],
                           "submission": seats_meta.get(s, (None, None, None))[2]} for s in (0, 1)],
                "n_labels": int(lab.shape[0]), "n_multi": n_multi,
                "n_invalid": n_invalid, "n_nonplanet": n_nonplanet}
        out_fp = os.path.join(out_dir, f"{epid}.npz")
        np.savez_compressed(
            out_fp, av=np.float32(arrs[0]["av"]),
            p_id=np.array(all_ids, np.int32), p_radius=p_radius, p_prod=p_prod,
            p_is_comet=p_is_comet, p_is_orbiting=p_is_orb, p_orbital_r=p_orb_r,
            path_tab_x=ptx, path_tab_y=pty, path_len=plen,
            t_step=np.arange(cut, dtype=np.int32),
            p_mask=p_mask[:cut], p_owner=p_owner[:cut], p_ships=p_ships[:cut],
            p_x=p_x[:cut], p_y=p_y[:cut], p_cidx=p_cidx[:cut], p_pathref=p_pathref[:cut],
            share0=share0[:cut],
            f_off=foff[:cut + 1], f_owner=fo[:foff[cut]], f_ships=fs[:foff[cut]],
            f_target=ftg[:foff[cut]], f_arrival=fa[:foff[cut]],
            labels=lab, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        return ("ok", {"episode_id": epid, "date": date, "path": out_fp,
                       "score0": meta["seats"][0]["score"], "score1": meta["seats"][1]["score"],
                       "reward0": meta["seats"][0]["reward"], "reward1": meta["seats"][1]["reward"],
                       "T_kept": int(cut), "T_total": T, "n_labels": int(lab.shape[0]),
                       "n_multi": n_multi, "n_invalid": n_invalid, "n_nonplanet": n_nonplanet})
    except Exception as e:
        import traceback
        return ("error", f"{os.path.basename(fp)}: {type(e).__name__}: {e} | {traceback.format_exc(limit=2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True, help="comma-separated dates (official_replay/<date>)")
    ap.add_argument("--min_score", type=float, default=1400.0)
    # decided-tail cut DROPPED by default (user 2026-06-13): it removed ~33% of turns to fight the
    # no-op imbalance, but the gate class-balance (gate_pos_w=6) handles no-op at the loss now, so the
    # cut was redundant + threw away data and close-out skill. vkhydras (10th) keeps full games. Set
    # --decided_share <1 to re-enable.
    ap.add_argument("--decided_share", type=float, default=1.01)   # >1 = never cut (keep full games)
    ap.add_argument("--decided_turns", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(_HERE, "data", "v1"))
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    ap.add_argument("--limit", type=int, default=0, help="pilot: max episodes per day (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seats = load_agent_meta()
    jobs = []
    for day in args.days.split(","):
        day = day.strip()
        man = os.path.join(OR_DIR, day, "manifest.csv")
        rows = [r for r in csv.DictReader(open(man)) if float(r["min_score"]) >= args.min_score]
        if args.limit:
            rows = rows[: args.limit]
        for r in rows:
            fp = os.path.join(OR_DIR, day, "2p", f"{r['episode_id']}.npz".replace(".npz", ".json"))
            if os.path.exists(fp) and not os.path.exists(os.path.join(args.out, f"{r['episode_id']}.npz")):
                jobs.append((fp, seats.get(int(r["episode_id"]), {}), day,
                             args.decided_share, args.decided_turns, args.out))
    print(f"episodes to process: {len(jobs)} (min_score>={args.min_score})", flush=True)

    man_fp = os.path.join(args.out, "manifest.csv")
    new_file = not os.path.exists(man_fp)
    mf = open(man_fp, "a", newline="")
    mw = csv.writer(mf)
    if new_file:
        mw.writerow(["episode_id", "date", "path", "score0", "score1", "reward0", "reward1",
                     "T_kept", "T_total", "n_labels", "n_multi", "n_invalid", "n_nonplanet"])
    stats = {"ok": 0, "error": 0, "skip_short": 0, "skip_nolabels": 0}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (kind, payload) in enumerate(ex.map(process_episode, jobs, chunksize=4)):
            stats[kind] = stats.get(kind, 0) + 1
            if kind == "ok":
                mw.writerow([payload[k] for k in ["episode_id", "date", "path", "score0", "score1",
                                                  "reward0", "reward1", "T_kept", "T_total",
                                                  "n_labels", "n_multi", "n_invalid", "n_nonplanet"]])
                mf.flush()
            elif kind == "error" and stats["error"] <= 5:
                print("ERR:", payload, flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}  {stats}  {time.time()-t0:.0f}s", flush=True)
    mf.close()
    print(f"DONE {stats} in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
