"""Single chain diagram of the octree discretization of translation R^3 (the NeST SE(3) translation factor).

A horizontal chain, matching the style of the Hopf--HEALPix chain (gen_hopf_chain_fig.py) but with
isometric octree cubes (same iso/_oct_faces machinery as the SE(3) decode tree):

  t in R^3  --anchor+scale-->  bounded box [t_hat +- A]  --octree split x8-->  8 equal-volume octants
            --select-->  one octant (t = t_hat + A g)  --refine x8-->  self-similar tree, N(l)=8^l cells.

The octree is equivolumetric by construction: every split halves each side, so each of the 8 children has
exactly 1/8 the parent volume -- the translation analogue of the equal-area HEALPix x uniform-roll partition.

Run:  cd /home/kavinder/SPACE-HOP && PYTHONPATH=src python -I scripts/figs/gen_octree_chain_fig.py
Out:  logs/figs/octree_chain.pdf/.png
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
from se3_head import TRANS_OCT_OFF                     # (8,3) in {-0.5,+0.5}: the 8 octant offsets

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
INK = "#272727"; GOLD = "#FFD700"; GOLD_LINE = "#C99200"; BLUE = "#0F4D92"; GREY = "#6f7782"
BLUE2 = np.array([0.215, 0.459, 0.729]); GOLD_RGB = np.array([1.0, 0.843, 0.0])
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/octree_chain"

# ---------------- isometric cube machinery (shared look with gen_se3_tree_fig.py) ----------------
def iso(p): return np.array([(p[0] - p[1]) * 0.866, p[2] * 0.95 + (p[0] + p[1]) * 0.5])
SHADE = {"top": 1.0, "right": 0.80, "left": 0.63}

def _faces(c, hs):
    def P(dx, dy, dz): return iso(c + np.array([dx, dy, dz]) * hs)
    return {"top":   [P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)],
            "right": [P(1, -1, -1), P(1, 1, -1), P(1, 1, 1), P(1, -1, 1)],
            "left":  [P(-1, 1, -1), P(1, 1, -1), P(1, 1, 1), P(-1, 1, 1)]}

def cube(ax, c, hs, base, alpha=1.0, ec="#3a3a3a", lw=0.7, z=3):
    for fn, fv in _faces(c, hs).items():
        ax.add_patch(Polygon(fv, closed=True, fc=tuple(np.clip(base * SHADE[fn], 0, 1)),
                             ec=ec, lw=lw, alpha=alpha, zorder=z))

def wire(ax, c, hs, ec="#9aa2ad", lw=1.0, z=2):
    v = {(sx, sy, sz): iso(c + np.array([sx, sy, sz]) * hs) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
    for a in v:
        for b in v:
            if sum(abs(np.subtract(a, b))) == 2:         # share an edge
                ax.plot([v[a][0], v[b][0]], [v[a][1], v[b][1]], color=ec, lw=lw, zorder=z)

def octants(ax, parent_hs, gap, highlight=None, base_all=None, dim=True):
    """8 children of a cube, exploded by `gap`, depth-sorted; `highlight` drawn gold and last."""
    chs = parent_hs / 2.0
    cen = {k: TRANS_OCT_OFF[k].astype(float) * parent_hs + np.sign(TRANS_OCT_OFF[k]) * gap for k in range(8)}
    rest = sorted([k for k in range(8) if k != highlight], key=lambda k: -iso(cen[k] + [0, 0, 1e-4])[1])
    for k in rest:
        base = GOLD_RGB if base_all == "gold" else BLUE2
        a = 0.95 if (base_all or not dim) else 0.9
        cube(ax, cen[k], chs, base, alpha=a, ec="#5f7fae" if base_all != "gold" else GOLD_LINE,
             lw=0.6, z=3)
    if highlight is not None:
        cube(ax, cen[highlight], chs, GOLD_RGB, alpha=0.95, ec=GOLD_LINE, lw=1.0, z=5)

# ---------------- figure scaffold ----------------
fig = plt.figure(figsize=(16.0, 4.5))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.5, 0.95, "The octree discretization of translation $\\mathbb{R}^3$: one equivolumetric partition by construction",
        ha="center", va="center", fontsize=17, fontweight="bold", color=INK)

CX = [0.095, 0.305, 0.515, 0.725, 0.925]
CY = 0.56; PW, PH = 0.150, 0.52
LY = CY - PH / 2 + 0.028

def panel(cx):
    ax = fig.add_axes([cx - PW / 2, CY - PH / 2, PW, PH])
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.15, 2.35); ax.set_aspect("equal"); ax.axis("off")
    return ax

def label(cx, title, sub):
    bg.text(cx, LY, title, ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
    bg.text(cx, LY - 0.075, sub, ha="center", va="top", fontsize=10.5, color=BLUE)

def arrow(x0, x1, top, bot):
    bg.annotate("", xy=(x1, CY), xytext=(x0, CY),
                arrowprops=dict(arrowstyle="-|>", color=GOLD_LINE, lw=2.6, mutation_scale=22))
    bg.text((x0 + x1) / 2, CY + 0.055, top, ha="center", va="bottom", fontsize=10.5,
            fontweight="bold", color=GOLD_LINE)
    if bot:
        bg.text((x0 + x1) / 2, CY - 0.055, bot, ha="center", va="top", fontsize=9.5, color=GREY)

# ---- panel 1: continuous translation in R^3 ----
ax = panel(CX[0])
wire(ax, np.zeros(3), 1.0, ec="#b7bec8", lw=1.0)
for L, d in [("$x$", np.array([1, 0, 0])), ("$y$", np.array([0, 1, 0])), ("$z$", np.array([0, 0, 1]))]:
    o = iso(np.array([-1.0, -1.0, -1.0])); e = iso(np.array([-1.0, -1.0, -1.0]) + d * 1.1)
    ax.annotate("", xy=e, xytext=o, arrowprops=dict(arrowstyle="-|>", color="#7b838d", lw=1.4))
    ax.text(*(e + 0.12 * (e - o)), L, ha="center", va="center", fontsize=10, color="#50575f")
gt = np.array([0.45, -0.35, 0.40])                      # true translation t
tc = iso(np.zeros(3)); tg = iso(gt)
ax.plot([tc[0], tg[0]], [tc[1], tg[1]], color=GOLD_LINE, lw=1.6, ls="--", zorder=4)
ax.scatter([tc[0]], [tc[1]], s=26, c=INK, zorder=6)
ax.text(tc[0] - 0.08, tc[1] - 0.14, "$\\hat t$", ha="right", va="center", fontsize=11, color=INK)
ax.scatter([tg[0]], [tg[1]], s=120, c=GOLD, edgecolors=GOLD_LINE, linewidths=1.5, zorder=7)
ax.text(tg[0] + 0.12, tg[1] + 0.04, "$t$", ha="left", va="center", fontsize=12, color=GOLD_LINE, fontweight="bold")
label(CX[0], "$t\\in\\mathbb{R}^3$", "camera-frame translation,\nanchor $\\hat t$, extent $A$")

# ---- panel 2: the bounded box = octree root cell ----
ax = panel(CX[1])
cube(ax, np.zeros(3), 1.0, BLUE2, alpha=0.9, ec="#3a3a3a", lw=1.0, z=3)
ax.scatter([iso(np.zeros(3))[0]], [iso(np.zeros(3))[1]], s=22, c="white", edgecolors=INK, linewidths=1.0, zorder=6)
label(CX[1], "root cell", "box $[\\hat t - A,\\ \\hat t + A]$,\nvolume $V_0 = 8\\,\\det A$")

# ---- panel 3: octree split x8 ----
ax = panel(CX[2])
octants(ax, 1.0, gap=0.085, highlight=None, dim=False)
label(CX[2], "octree split $\\times 8$", "8 octants $(\\pm,\\pm,\\pm)$,\neach $\\frac{1}{8}$ the volume")

# ---- panel 4: select one octant ----
ax = panel(CX[3])
CHO = 5                                                  # chosen octant (+,-,+)-ish, front and visible
octants(ax, 1.0, gap=0.085, highlight=CHO, dim=True)
label(CX[3], "chosen octant", "equivolumetric:\n$t = \\hat t + A\\,g,\\ \\ g\\in[-1,1]^3$")

# ---- panel 5: refine x8 (self-similar) ----
ax = panel(CX[4])
octants(ax, 1.0, gap=0.055, highlight=None, base_all="gold")   # the chosen octant subdivides into its own 8
label(CX[4], "refine $\\times 8$", "halve each side,\n$N(\\ell) = 8^{\\ell}$ cells")

# ---- chain arrows ----
r = PW / 2 + 0.004
arrow(CX[0] + r, CX[1] - r, "anchor + scale", "$t=\\hat t+Ag$")
arrow(CX[1] + r, CX[2] - r, "subdivide", "split each axis")
arrow(CX[2] + r, CX[3] - r, "select", "octant index $o$")
arrow(CX[3] + r, CX[4] - r, "recurse", "$A \\to A/2$")

for ext_ in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext_}", dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
