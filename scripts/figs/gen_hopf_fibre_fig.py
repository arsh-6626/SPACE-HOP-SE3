"""Demonstrate fibre selection with the ACTUAL Hopf fibres (not a roll dial).

Left: the base sphere S^2; choosing a HEALPix base point selects which fibre.
Right: the real Hopf fibres of SO(3) ~= S^3, stereographically projected to R^3. Each fibre is a great circle of
rotations sharing a base point (all in-plane rolls). The selected base point's fibre is drawn in full, coloured by
the roll angle psi (hue, matching the rest of the figures); the chosen roll bin is the gold point ON that circle.
A few neighbouring fibres (grey) show the linked-circle structure of the fibration.

Run:  cd /home/kavinder/SPACE-HOP && PYTHONPATH=src python -I scripts/figs/gen_hopf_fibre_fig.py
Out:  logs/figs/hopf_fibre.pdf/.png
"""
import os, sys, numpy as np, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
import _globe as G

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
INK = "#272727"; GOLD = "#FFD700"; GOLD_LINE = "#C99200"; BLUE = "#0F4D92"; GREY = "#8a9099"
HSV = mpl.colormaps["hsv"]
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/hopf_fibre"

N_ROLL = 6                                   # roll bins at the base resolution (n_roll = 6 N_side, N_side=1)
SEL = (np.radians(62.0), np.radians(35.0))   # selected base point (colat theta, lon phi)
SEL_ROLL = 1                                 # chosen roll bin on the selected fibre
CTX = [(np.radians(62.0), np.radians(a)) for a in (-75, -25, 95, 150, 200)] + \
      [(np.radians(112.0), np.radians(a)) for a in (10, 70, 160)]


def base_vec(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), np.cos(theta)])


def fibre_xyz(theta, phi, n=600):
    """Hopf fibre over base point (theta,phi): great circle in S^3, stereographically projected to R^3."""
    t = np.linspace(0, 2 * np.pi, n)
    a = np.cos(theta / 2) * np.cos(t); b = np.cos(theta / 2) * np.sin(t)
    c = np.sin(theta / 2) * np.cos(t + phi); dd = np.sin(theta / 2) * np.sin(t + phi)
    s = 1.0 / (1.0 - dd)
    return np.stack([a * s, b * s, c * s], 1), t


# ================================ figure ================================
fig = plt.figure(figsize=(15.0, 6.6))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.5, 0.955, "Fibre selection via the actual Hopf fibres of $SO(3)$",
        ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
bg.text(0.5, 0.905, "a base point on $S^2$ selects a fibre (a great circle of rotations in $SO(3)\\!\\cong\\!S^3$); "
        "the roll $\\psi$ is the position along that fibre", ha="center", va="center", fontsize=12, color=BLUE)

# ---- panel A: base sphere, selected base point ----
sb = base_vec(*SEL)
Rv = G.make_view(sb, delta_deg=16.0)
axA = fig.add_axes([0.02, 0.10, 0.34, 0.72])
axA.set_xlim(-1.15, 1.15); axA.set_ylim(-1.15, 1.15); axA.set_aspect("equal"); axA.axis("off")
G.shaded_disk(axA); G.graticule(axA, Rv)
for p in range(12):                                  # HEALPix base tiling for context
    poly = G._clamp_poly(Rv, hp.boundaries(1, p, step=16, nest=True).T)
    if poly is not None:
        axA.add_patch(mpl.patches.Polygon(poly, closed=True, fc="none", ec="#c3cad3", lw=0.7, zorder=3))
for th, ph in CTX:
    v = base_vec(th, ph); xy, dep = G.project(Rv, v[None])
    if dep[0] > 0:
        axA.scatter([xy[0, 0]], [xy[0, 1]], s=46, c="#6f7782", edgecolors="white", linewidths=0.8, zorder=6)
xy, _ = G.project(Rv, sb[None])
axA.scatter([xy[0, 0]], [xy[0, 1]], s=150, c=GOLD, edgecolors=GOLD_LINE, linewidths=1.6, zorder=8)
axA.add_patch(Circle((0, 0), 1.0, fill=False, ec="#8f97a2", lw=1.1, zorder=9))
bg.text(0.19, 0.075, "base point on $S^2$ (a HEALPix bin)\nselects ‘which fibre’", ha="center", va="top",
        fontsize=12, color=INK, fontweight="bold")

# ---- arrow A -> B ----
bg.annotate("", xy=(0.405, 0.46), xytext=(0.355, 0.46),
            arrowprops=dict(arrowstyle="-|>", color=GOLD_LINE, lw=2.6, mutation_scale=22))
bg.text(0.38, 0.50, "its fibre", ha="center", va="bottom", fontsize=11, fontweight="bold", color=GOLD_LINE)

# ---- panel B: Hopf fibres in R^3 (stereographic) ----
axB = fig.add_axes([0.40, 0.045, 0.58, 0.83], projection="3d")
for th, ph in CTX:                                   # neighbouring fibres: linked grey circles
    P, _ = fibre_xyz(th, ph)
    axB.plot(P[:, 0], P[:, 1], P[:, 2], color="#b4bcc6", lw=1.4, alpha=0.85, zorder=1)
P, t = fibre_xyz(*SEL)                                # selected fibre, coloured by roll psi
seg = np.stack([P[:-1], P[1:]], 1)
cols = HSV((t[:-1] / (2 * np.pi)) % 1.0)
lc = Line3DCollection(seg, colors=cols, linewidths=5.0, zorder=5); axB.add_collection3d(lc)
psi = (SEL_ROLL + 0.5) * 2 * np.pi / N_ROLL          # chosen roll bin -> point on the fibre
pa = np.cos(SEL[0] / 2) * np.cos(psi); pb = np.cos(SEL[0] / 2) * np.sin(psi)
pc = np.sin(SEL[0] / 2) * np.cos(psi + SEL[1]); pd = np.sin(SEL[0] / 2) * np.sin(psi + SEL[1])
ps = 1.0 / (1.0 - pd); pt = np.array([pa * ps, pb * ps, pc * ps])
axB.scatter([pt[0]], [pt[1]], [pt[2]], s=160, c=GOLD, edgecolors=GOLD_LINE, linewidths=1.6, zorder=10)
# tick marks for the N_ROLL bins along the selected fibre
for rb in range(N_ROLL):
    a_ = (rb + 0.5) * 2 * np.pi / N_ROLL
    qa = np.cos(SEL[0] / 2) * np.cos(a_); qb = np.cos(SEL[0] / 2) * np.sin(a_)
    qc = np.sin(SEL[0] / 2) * np.cos(a_ + SEL[1]); qd = np.sin(SEL[0] / 2) * np.sin(a_ + SEL[1])
    qs = 1.0 / (1.0 - qd); q = np.array([qa * qs, qb * qs, qc * qs])
    axB.scatter([q[0]], [q[1]], [q[2]], s=26, c=[HSV((a_ / (2 * np.pi)) % 1.0)],
                edgecolors="white", linewidths=0.6, zorder=8)

allp = np.concatenate([fibre_xyz(*c)[0] for c in CTX] + [P], 0)
lo, hi = np.percentile(allp, 2, 0), np.percentile(allp, 98, 0)
ctr = (lo + hi) / 2; rad = 0.60 * (hi - lo).max()
axB.set_xlim(ctr[0] - rad, ctr[0] + rad); axB.set_ylim(ctr[1] - rad, ctr[1] + rad)
axB.set_zlim(ctr[2] - rad, ctr[2] + rad)
axB.set_box_aspect((1, 1, 1)); axB.view_init(elev=20, azim=-62)
axB.set_xticks([]); axB.set_yticks([]); axB.set_zticks([])
axB.grid(False)
for pane in (axB.xaxis, axB.yaxis, axB.zaxis):
    pane.set_pane_color((1, 1, 1, 0)); pane.line.set_color((1, 1, 1, 0))
bg.text(0.69, 0.055, "the actual Hopf fibre $S^1\\subset SO(3)$ (stereographic)\nhue $=$ roll $\\psi$,   "
        "gold $=$ the chosen roll bin", ha="center", va="top", fontsize=12, color=INK, fontweight="bold")

# roll-hue colourbar
cax = fig.add_axes([0.945, 0.58, 0.011, 0.20])
sm = mpl.cm.ScalarMappable(cmap="hsv", norm=mpl.colors.Normalize(0, 2 * np.pi))
cb = fig.colorbar(sm, cax=cax, ticks=[0, np.pi, 2 * np.pi]); cb.ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
cb.set_label("roll $\\psi$ (hue)", fontsize=10)

for ext_ in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext_}", dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
