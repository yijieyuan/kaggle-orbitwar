# figures/src — figure generators

The final figures live one level up in `figures/` (that is what `write-up.md`
references). This folder holds everything used to *produce* them.

- The matplotlib generators (`make_*.py`) each write their output(s) into
  `figures/` and load the bundled fonts from `figures/fonts/`. They resolve
  those paths relative to the repo, so run them from anywhere, e.g.
  `python figures/src/make_final_eval.py`.
- `prompt-model_architecture.txt` is the text prompt fed to an image model to
  produce `model_architecture.jpg` (not a script).

| generator | output(s) in `figures/` |
|---|---|
| `make_feature_tables.py` | `token_planet.png`, `token_global.png`, `edge_features.png` |
| `make_model_profile.py` | `model_profile.png` |
| `make_il_vs_public.py` | `il_vs_public.png` |
| `make_cross_eval.py` | `cross_2p.png`, `cross_4p.png` |
| `make_final_eval.py` | `final_eval.png` |
| `make_eval_phase.py` | `eval_h2h.png`, `eval_field.png` |
| `prompt-model_architecture.txt` | `model_architecture.jpg` (image-gen prompt) |

Two figures are produced elsewhere:
- `rank_trends.png` comes from the rank tracker (`tools/rank_tracker`, private repo).
- `model_architecture.jpg` is image-generated from the prompt above.

Superseded/unused figures were moved to `_archive/figures/`.
