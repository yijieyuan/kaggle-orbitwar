# Luca vs Isaiah @ Tufa Labs — the one head-to-head win

The single evaluation-phase 2-player game in which our submission **Luca** (the merge,
sub `53993338`) beat **Isaiah @ Tufa Labs** head-to-head. Luca went 1–5 in 2p against
Isaiah's two submissions overall; this is that one win.

- **Episode** `84096506` · **seed** `432888502`
- **P0** Isaiah @ Tufa Labs (blue) · **P1** Luca (red, winner)
- Luca wins by full-board conquest at turn 143 (all 36 planets, 2006 ships).

## Files

| file | what |
|------|------|
| `episode_84096506_raw.json` | the ORIGINAL replay, as downloaded from Kaggle (`GET /api/v1/competitions/episodes/84096506/replay`) |
| `winprob_84096506.json` | per-turn RL win-confidence for seat 1 (Luca), `(V(s)+1)/2` from the 2p greedy value head |
| `make_replay_gif.py` | computes the win-prob from the replay (via `2p/inference`, numpy) and renders the GIF |
| `luca_vs_tufa_ep84096506.gif` | the rendered animation (1040×880, 4× playback) |

## Regenerate

```
conda run -n kaggle-orbitwar python make_replay_gif.py
```

Rendered in the local replay viewer style (`viewer/viewer_core.js`): dark board with the
sun, planets (ship count / `+prod` / `pID`) and heading-rotated fleet triangles, a right
info panel, and two half-height charts below (total ships | win probability). Playback is
4× (83 ms/turn, one frame per turn). The win-probability curve uses the deploy value head
verbatim and was verified to reproduce a local replay's stored `_aux.winprob` to
`max|Δ| = 0.00000`, so the offline replay→winprob is exact.
