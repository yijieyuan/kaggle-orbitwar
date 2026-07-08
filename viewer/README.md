# Replay viewer

A self-contained canvas player (no build step) rendering the official replay schema
(`steps[t][seat].observation.{planets,fleets}`):

| viewer | reads | picker |
|--------|-------|--------|
| `visualize_local.html` | `local_replays/{2p,4p}/` | 2p\|4p selector, then exp/opp/seed |

**Serve from the solution root** (the viewer `fetch()`es its replay JSON, so a `file://` origin will
not work):

```bash
cd <solution-root>
python -m http.server 8000
```

Then open `http://localhost:8000/viewer/visualize_local.html`. The renderer (`viewer_core.js`) draws
2p and 4p games identically. Regenerate or add more games with `eval/eval.py` (see `eval/`).
