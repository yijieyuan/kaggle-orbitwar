"""Render the two imitation-learning loss equations as PNGs, for the KAGGLE copy
of the write-up only. Kaggle's markdown renderer (Showdown, no KaTeX) does not
render LaTeX, so `write-up.kaggle.md` embeds these images instead. The GitHub
`write-up.md` keeps the native $$...$$ LaTeX and does not use these.

    python figures/src/make_equations.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # figures/ (script lives in figures/src/)

EQS = {
    "eq_ptr":  r"$L_{\mathrm{ptr}} = -\sum_i w_i\,\log p_\theta(t_i), \qquad w_i = 6 \ \mathrm{for\ a\ launch,\ else}\ 1$",
    "eq_frac": r"$L_{\mathrm{frac}} = -\sum_{i\,\in\,\mathrm{launches}} \log \mathcal{N}_{[0,1]}(f_i \mid \mu_i, \sigma_i)$",
}


def draw():
    for name, eq in EQS.items():
        fig = plt.figure(figsize=(8, 0.9), dpi=220)
        fig.text(0.5, 0.5, eq, ha="center", va="center", fontsize=20, color="#2b2b2b")
        out = os.path.join(HERE, name + ".png")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.14, facecolor="white")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    draw()
