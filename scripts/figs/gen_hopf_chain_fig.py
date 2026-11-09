"""Single chain diagram of the Hopf-HEALPix discretization of SO(3).

A horizontal chain of shaded-globe panels (same style as the decode-tree figures):
  SO(3)  --Hopf-->  S^2 base x S^1 fibre  --HEALPix x uniform roll-->  equal-area bins
         --product-->  one equivolumetric cell  --nested refine x8-->  hierarchical tree.

Run:  cd /home/kavinder/SPACE-HOP && PYTHONPATH=src python -I scripts/figs/gen_hopf_chain_fig.py
Out:  logs/figs/hopf_chain.pdf/.png
"""
import os, sys, numpy as np, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
import _globe as G

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
INK = "#272727"; GOLD = "#FFD700"; GOLD_LINE = "#C99200"; BLUE = "#0F4D92"; GREY = "#6f7782"
HSV = mpl.colormaps["hsv"]
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/hopf_chain"

P0 = 4                                           # the nside=1 base pixel we follow down the chain
base = np.array(hp.pix2vec(1, P0, nest=True)); base = base / np.linalg.norm(base)
R = G.make_view(base, delta_deg=22.0)


def fibre_xyz(theta, phi, n=500):
    """Actual Hopf fibre over base point (theta,phi): great circle in S^3 stereographically projected to R^3."""
    t = np.linspace(0, 2 * np.pi, n)
    a = np.cos(theta / 2) * np.cos(t); b = np.cos(theta / 2) * np.sin(t)
    c = np.sin(theta / 2) * np.cos(t + phi); dd = np.sin(theta / 2) * np.sin(t + phi)
    s = 1.0 / (1.0 - dd)
    return np.stack([a * s, b * s, c * s], 1), t

fig = plt.figure(figsize=(16.0, 4.5))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.5, 0.95, "The Hopf--HEALPix discretization of $SO(3)$: one equivolumetric partition by construction",
        ha="center", va="center", fontsize=17, fontweight="bold", color=INK)

CX = [0.095, 0.305, 0.515, 0.725, 0.925]         # panel centres
CY = 0.58; GW, GH = 0.115, 0.46                  # globe box (square in display)
LY = CY - GH / 2 - 0.005                          # label baseline


def sphere(cx):
    ax = fig.add_axes([cx - GW / 2, CY - GH / 2, GW, GH])
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
    G.shaded_disk(ax); G.graticule(ax, R)
    return ax


def limb(ax):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#8f97a2", lw=1.1, zorder=8))


PASTEL = mpl.colormaps["Pastel1"]
def base_tiles(ax, fill=False):
    for p in range(12):
        poly = G._clamp_poly(R, hp.boundaries(1, p, step=16, nest=True).T)
        if poly is None:
            continue
        fc = "none"; ec = "#b9c0c9"; lw = 0.7
        if fill:                                    # distinct pastel per pixel -> the equal-area tiling reads
            sh = G._shade(hp.pix2vec(1, p, nest=True), R)
            fc = tuple(np.array(PASTEL((p % 9) / 8.0))[:3] * sh); ec = "#7f8893"; lw = 1.0
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec=ec, lw=lw, zorder=3))


def fill_pixel(ax, nside, pix, col, lw=2.4, ec=GOLD_LINE, alpha=0.95):
    poly = G._clamp_poly(R, hp.boundaries(nside, pix, step=14, nest=True).T)
    if poly is not None:
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec=ec, lw=lw, alpha=alpha, zorder=5))


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


# ---- panel 1: SO(3) ----
ax = sphere(CX[0]); limb(ax)
ax.annotate("", xy=(0.62, 0.66), xytext=(-0.62, 0.66),
            arrowprops=dict(arrowstyle="-|>", color=GOLD_LINE, lw=2.2,
                            connectionstyle="arc3,rad=-0.45"))
ax.scatter([0], [0], s=16, c=INK, zorder=9)
label(CX[0], "$SO(3)$", "rotations,\nHaar volume $\\pi^2$")

# ---- panel 2: the actual Hopf fibre drawn ON the same globe ----
ax = sphere(CX[1]); limb(ax)
th0 = float(np.arccos(np.clip(base[2], -1, 1)))                   # base point (theta, phi) of the followed pixel
ph0 = float(np.arctan2(-base[1], base[0]))
ctr = np.array([0.0, 0.03])                                      # draw it large and centred on the globe
Ps, ts = fibre_xyz(th0, ph0)                                      # actual Hopf fibre, projected with the globe view
fp = (Ps @ R.T)[:, :2]; fp = fp - fp.mean(0)
fp = fp * (0.58 / np.abs(fp).max()) + ctr
segs = np.stack([fp[:-1], fp[1:]], 1)
ax.add_collection(LineCollection(segs, colors=HSV((ts[:-1] / (2 * np.pi)) % 1.0), linewidths=5.2, zorder=6))
rb = 2; idx = int(np.argmin(np.abs((ts - (rb + 0.5) * 2 * np.pi / 6 + np.pi) % (2 * np.pi) - np.pi)))
ax.scatter([ctr[0]], [ctr[1]], s=32, c=INK, zorder=7)            # the base point
ax.scatter([fp[idx, 0]], [fp[idx, 1]], s=95, c=GOLD, edgecolors=GOLD_LINE, linewidths=1.3, zorder=8)
label(CX[1], "$S^2 \\times S^1$", "the actual Hopf fibre\n(roll $\\psi$ along it)")

# ---- panel 3: HEALPix x roll ----
ax = sphere(CX[2]); base_tiles(ax, fill=True); limb(ax)
# mini roll dial
dax = fig.add_axes([CX[2] + 0.030, CY - 0.175, 0.055, 0.055], projection="polar")
dax.set_xticks([]); dax.set_yticks([]); dax.set_ylim(0, 1); dax.set_frame_on(False)
th = (np.arange(6) + 0.5) * (2 * np.pi / 6)
dax.bar(th, np.full(6, 0.55), width=2 * np.pi / 6, bottom=0.40,
        color=[HSV(G.hue(k, 6)) for k in range(6)], edgecolor="white", linewidth=0.3)
label(CX[2], "HEALPix $\\times$ roll", "$N_{\\rm pix}{=}12N_{\\rm side}^2$ equal-area\n"
      "$\\times\\; N_\\psi{=}6N_{\\rm side}$ roll bins")

# ---- panel 4: one equivolumetric cell ----
ax = sphere(CX[3]); base_tiles(ax); fill_pixel(ax, 1, P0, tuple(np.array(HSV(G.hue(2, 6)))[:3])); limb(ax)
G.fibre_ring(ax, R, base, 6, 2, rad=0.36, accent=GOLD_LINE, dot=GOLD, s=26)
label(CX[3], "one cell", "equivolumetric:\n$\\mu(\\mathrm{cell})=\\pi^2/N(\\ell)$")

# ---- panel 5: nested refinement x8 ----
ax = sphere(CX[4]); base_tiles(ax)
for i, ch in enumerate(range(4 * P0, 4 * P0 + 4)):              # pixel splits x4 (NESTED)
    col = tuple(np.array(HSV(G.hue(2, 6)))[:3]) if ch == 4 * P0 else "none"
    fill_pixel(ax, 2, ch, col, lw=1.6, ec=(GOLD_LINE if ch == 4 * P0 else "#3a3a3a"),
               alpha=(0.95 if ch == 4 * P0 else 1.0))
limb(ax)
label(CX[4], "refine $\\times 8$", "pixel $\\times4$, roll $\\times2$\n$N(\\ell)=72\\cdot 8^{\\ell}$")

# ---- chain arrows ----
r = GW / 2 + 0.006
arrow(CX[0] + r, CX[1] - r, "Hopf map", "$h:\\,S^3\\!\\to\\!S^2$")
arrow(CX[1] + r, CX[2] - r, "discretize", "equal-area $\\times$ uniform")
arrow(CX[2] + r, CX[3] - r, "product", "pix $\\cdot n_{\\rm roll}+$ roll")
arrow(CX[3] + r, CX[4] - r, "subdivide", "$N_{\\rm side}\\!\\to\\!2N_{\\rm side}$")

for ext_ in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext_}", dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
