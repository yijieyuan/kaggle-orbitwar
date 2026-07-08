"""Minimal extraction of shared.sim.runner.state_to_obs for the flat 4p-MERGE inference bundle.

ONLY state_to_obs is needed by the merge rollout (forward_sim GameState -> kaggle-style obs).
It uses nothing but the state object's attributes + stdlib, so the extraction is self-contained
(no shared.sim.agents / labels / official_meta imports — those are the heavy parts of the full
runner.py we deliberately DON'T bundle). Verbatim copy of the function body (2026-06-23)."""


def state_to_obs(state, player: int, initial_planets: list) -> dict:
    """Serialize a forward_sim GameState into a kaggle-style per-player observation
    (inverse of forward_sim.from_kaggle_obs). Tuple layouts match the env exactly:
    planet [id,owner,x,y,radius,ships,production]; fleet [id,owner,x,y,angle,from_id,ships]."""
    planets = [[int(p.id), int(p.owner), float(p.x), float(p.y), float(p.radius),
                int(p.ships), int(p.production)] for p in state.planets]
    fleets = [[int(f.id), int(f.owner), float(f.x), float(f.y), float(f.angle),
               int(f.from_planet_id), int(f.ships)] for f in state.fleets]
    comet_ids, comet_paths, comet_pidx = [], [], 0
    for p in state.planets:
        if getattr(p, "is_comet", False):
            comet_ids.append(int(p.id))
            comet_paths.append(getattr(p, "comet_path", []) or [])
            comet_pidx = int(getattr(p, "comet_path_index", 0) or 0)
    # official 'comets' = ONE group carrying all comet planets together: {path_index, paths, planet_ids}
    # (paths aligned 1:1 with planet_ids, matching what the deploy agent zips on real kaggle obs).
    comets = [{"path_index": comet_pidx, "paths": comet_paths, "planet_ids": comet_ids}] if comet_ids else []
    return {"step": int(state.step), "planets": planets, "fleets": fleets,
            "comets": comets, "comet_planet_ids": comet_ids,
            "initial_planets": initial_planets, "angular_velocity": float(state.angular_velocity),
            "next_fleet_id": int(state.next_fleet_id), "player": int(player),
            "remainingOverageTime": 60}
