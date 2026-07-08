"""Reproducible feature-table figures (planet token / global token).

Deterministic Nature/Science-style tables, rendered with bundled Inter (a clean
Helvetica-adjacent grotesque) so the output is byte-identical every run — no
image-model sampling. Fonts live in ./fonts/ ; run from anywhere:

    python figures/make_feature_tables.py

Style: no title/caption; horizontal rules only, ALL the same thickness; no vertical
lines/shading; Inter Light body; group label = SemiBold name + Light description;
auto-measured columns (4P CHANGE sits right after MEANING, no dead gap); the only
color is the dark-red change arrows; footnote gray except its arrow.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # figures/ (script lives in figures/src/)
FDIR = os.path.join(HERE, "fonts")
FP_BODY = fm.FontProperties(fname=os.path.join(FDIR, "Inter-Light.ttf"))
FP_BOLD = fm.FontProperties(fname=os.path.join(FDIR, "Inter-SemiBold.ttf"))
FP_ITAL = fm.FontProperties(fname=os.path.join(FDIR, "Inter-LightItalic.ttf"))

CHAR = "#2b2b2b"   # charcoal body
DRED = "#8f1d1d"   # dark red (change arrows only)
GRAY = "#7d7d7d"   # footnote gray
RULE = "#333333"   # rules
RULE_LW = 1.0      # thin middle rules (header, between-group)
RULE_LW_EDGE = 2.0 # thick top & bottom rules (booktabs)

BODY = 12.5; HEAD = 12.5; GNAME = 14.0; GDESC = 12.5; FOOT = 10.5

# layout constants (inches) — large text, tight leading (dense, minimal whitespace)
MARGIN = 0.24; GUT = 0.30; ROWH = 0.235
TOP_MARGIN = 0.18; TOPRULE_PAD = 0.09; HEADH = 0.36
GROUP_GAP = 0.16; RULE_PAD = 0.06; LABEL_H = 0.30; LABEL_GAP = 0.02
PRE_BOTTOM = 0.12; FOOT_GAP = 0.05; FOOT_H = 0.30; BOT_MARGIN = 0.18


def _fp(base, size):
    p = base.copy(); p.set_size(size); return p


def tw(s, size=BODY, fp=FP_BODY):
    if not s:
        return 0.0
    return TextPath((0, 0), s, size=size, prop=fp).get_extents().width / 72.0


def col_widths(specs):
    """Max width per column across ALL tables, so they share one layout (matched
    total width + columns that line up when stacked)."""
    rows = [r for spec in specs for g in spec["groups"] for r in g["rows"]]
    return {"f": max(tw(s) for s in [r[0] for r in rows] + ["FEATURE"]),
            "d": max(tw(s) for s in [r[1] for r in rows] + ["DIMS"]),
            "t": max(tw(s) for s in [r[2] for r in rows] + ["TYPE / SCALE"]),
            "m": max(tw(s) for s in [r[3] for r in rows] + ["MEANING"]),
            "p": max(tw(s) for s in [r[4] for r in rows] + ["4P CHANGE"])}


def draw(spec, out, cw):
    groups = spec["groups"]
    rows_all = [r for g in groups for r in g["rows"]]

    # shared column widths (same layout for every table)
    w_f, w_d, w_t, w_m, w_p = cw["f"], cw["d"], cw["t"], cw["m"], cw["p"]

    x_f = MARGIN
    x_d = x_f + w_f + GUT
    x_dc = x_d + w_d / 2.0
    x_t = x_d + w_d + GUT
    x_tc = x_t + w_t / 2.0
    x_m = x_t + w_t + GUT
    x_p = x_m + w_m + GUT
    x_pc = x_p + w_p / 2.0
    total_w = x_p + w_p + MARGIN

    ng = len(groups); nrow = len(rows_all)
    H = (TOP_MARGIN + TOPRULE_PAD + HEADH
         + sum((GROUP_GAP if gi > 0 else 0) + RULE_PAD + LABEL_H + LABEL_GAP
               + len(g["rows"]) * ROWH for gi, g in enumerate(groups))
         + PRE_BOTTOM + FOOT_GAP + FOOT_H + BOT_MARGIN)

    fig = plt.figure(figsize=(total_w, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total_w); ax.set_ylim(0, H)

    def rule(y, lw=RULE_LW):
        ax.plot([MARGIN, total_w - MARGIN], [y, y], color=RULE, lw=lw,
                solid_capstyle="butt", zorder=3)

    def cell(x, y, s, color=CHAR, ha="left", size=BODY):
        ax.text(x, y, s, ha=ha, va="center", color=color,
                fontproperties=_fp(FP_BODY, size), zorder=4)

    def label(x, y, name, desc):
        b = TextArea(name, textprops=dict(fontproperties=_fp(FP_BOLD, GNAME), color=CHAR))
        d = TextArea(desc, textprops=dict(fontproperties=_fp(FP_BODY, GDESC), color=CHAR))
        box = HPacker(children=[b, d], align="baseline", pad=0, sep=5)
        ax.add_artist(AnnotationBbox(box, (x, y), xycoords="data", frameon=False,
                                     box_alignment=(0, 0.5), zorder=4))

    y = H - TOP_MARGIN
    rule(y, RULE_LW_EDGE)                               # top rule (thick)
    y -= TOPRULE_PAD
    for x, s, ha in [(x_f, "FEATURE", "left"), (x_dc, "DIMS", "center"),
                     (x_tc, "TYPE / SCALE", "center"), (x_m, "MEANING", "left"),
                     (x_pc, "4P CHANGE", "center")]:
        cell(x, y - HEADH / 2, s, ha=ha, size=HEAD)
    y -= HEADH
    rule(y)                                             # header rule (= group-1 rule)

    for gi, g in enumerate(groups):
        if gi > 0:
            y -= GROUP_GAP
            rule(y)                                     # between-group rule (uniform)
        y -= RULE_PAD
        label(x_f, y - LABEL_H / 2, g["name"], g["desc"])
        y -= LABEL_H + LABEL_GAP
        for feat, dims, typ, mean, p4 in g["rows"]:
            yc = y - ROWH / 2
            cell(x_f, yc, feat)
            cell(x_dc, yc, dims, ha="center")
            cell(x_tc, yc, typ, ha="center")
            cell(x_m, yc, mean)
            cell(x_pc, yc, p4, ha="center", color=CHAR if p4 == "none" else DRED)
            y -= ROWH

    y -= PRE_BOTTOM
    rule(y, RULE_LW_EDGE)                               # bottom rule (thick)
    y -= FOOT_GAP
    fi = _fp(FP_ITAL, FOOT)
    parts = [("one-hot features are not further normalized.   4p change: 2-player dims ", GRAY),
             ("→", DRED),
             (" 4-player dims, none = unchanged.", GRAY)]
    fbox = HPacker(children=[TextArea(t, textprops=dict(fontproperties=fi, color=c))
                             for t, c in parts], align="baseline", pad=0, sep=0)
    ax.add_artist(AnnotationBbox(fbox, (x_f, y - FOOT_H / 2), xycoords="data",
                                 frameon=False, box_alignment=(0, 0.5), zorder=4))

    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


planet = {"groups": [
    {"name": "Static (15)", "desc": "  — current state", "rows": [
        ("owner", "3", "one-hot", "who holds it: me / opp / neutral", "3 → 5"),
        ("planet type", "3", "one-hot", "static / orbiting / comet", "none"),
        ("position", "2", "/100", "location in p0's frame", "none"),
        ("orbital velocity", "2", "none", "per-turn motion (0 if static)", "none"),
        ("radius", "1", "/5", "size", "none"),
        ("garrison", "1", "/500", "ships on it now", "none"),
        ("production", "1", "/5", "ships added per turn", "none"),
        ("remaining life", "1", "/100", "turns to game end / comet lifespan", "none"),
        ("inverse-log ships", "1", "0-1", "1 / (log(ships+1)+1)", "none"),
    ]},
    {"name": "Rollout summary (8)", "desc": "  — from a passive roll-forward", "rows": [
        ("first-flip turn", "1", "/50", "turn it first flips; 50 (horizon) = never", "none"),
        ("first-flip owner", "2", "one-hot", "who captures it if it flips", "2 → 4"),
        ("secured", "1", "0/1", "never flips over the 50-turn horizon", "none"),
        ("final owner", "2", "one-hot", "who holds it at the horizon end (turn 50)", "2 → 5"),
        ("fraction held", "2", "0-1", "share of horizon held by me / by current owner", "none"),
    ]},
    {"name": "Forecast features (50 × 7)", "desc": "  — per-turn series", "rows": [
        ("projected garrison", "1", "/500", "ship count that turn", "none"),
        ("owner (me / opp)", "2", "one-hot", "who holds it that turn", "2 → 5"),
        ("exists", "1", "0/1", "planet present (comets appear / expire)", "none"),
        ("future position", "2", "/100", "where it is that turn", "none"),
        ("time ramp", "1", "0-1", "t / 50", "none"),
    ]},
]}

glob = {"groups": [
    {"name": "Global (14)", "desc": "  — board summary", "rows": [
        ("turn", "1", "/500", "current turn", "none"),
        ("turns-left", "1", "/500", "turns to game end", "none"),
        ("angular velocity", "1", "×10", "orbit speed of the inner planets", "none"),
        ("comet-spawn countdown", "1", "/100", "turns to the next comet batch", "none"),
        ("my / opp ships", "2", "/2000", "total ships each side (planets + fleets)", "2 → 4"),
        ("my / opp production", "2", "/40", "total production each side", "2 → 4"),
        ("my / opp future production", "2", "/10000", "production × turns-left", "2 → 4"),
        ("planet count", "1", "/20", "planets in play", "none"),
        ("live comets", "1", "/5", "comets currently present", "none"),
        ("in-flight fleets", "2", "/256", "fleets each side has in flight", "2 → 4"),
    ]},
    {"name": "Economy summary (14)", "desc": "  — per-curve stats (ship-lead & production-lead)", "rows": [
        ("lead fraction", "2", "0-1", "share of the horizon I'm ahead on the lead", "none"),
        ("first-flip turn", "2", "/50", "when the lead first changes sign", "none"),
        ("secured", "2", "0/1", "never flips over the 50-turn horizon", "none"),
        ("ahead now", "2", "0/1", "ahead on the lead this turn (lead > 0)", "none"),
        ("behind now", "2", "0/1", "behind on the lead this turn (lead < 0)", "2 → 0"),
        ("ahead at end", "2", "0/1", "ahead at the horizon end (turn 50)", "none"),
        ("behind at end", "2", "0/1", "behind at the horizon end (turn 50)", "none"),
    ]},
    {"name": "Economy features (50 × 2)", "desc": "  — per-turn series", "rows": [
        ("ship lead", "1", "/2000", "my ships minus opp's, each future turn", "1 → 4"),
        ("production lead", "1", "/40", "my production minus opp's, each future turn", "1 → 4"),
    ]},
]}

edge = {"groups": [
    {"name": "Edge (P × P × 6)", "desc": "  — per ordered (source → target) pair, attention bias", "rows": [
        ("distance", "1", "/diag", "distance between the two planets", "none"),
        ("reachability", "1", "0/1", "can any fleet get from source to target", "none"),
        ("travel time", "1", "/50", "turns for the fleet to arrive", "none"),
        ("defender garrison at arrival", "1", "/500", "enemy ships waiting when it lands", "none"),
        ("attack margin", "1", "/500", "my ships minus the defense (ships - def - 1)", "none"),
        ("capture", "1", "0/1", "whether the launch takes the planet (margin > 0)", "none"),
    ]},
]}

if __name__ == "__main__":
    cw = col_widths([planet, glob, edge])   # shared -> matched width + aligned columns
    draw(planet, os.path.join(HERE, "token_planet.png"), cw)
    draw(glob, os.path.join(HERE, "token_global.png"), cw)
    draw(edge, os.path.join(HERE, "edge_features.png"), cw)
