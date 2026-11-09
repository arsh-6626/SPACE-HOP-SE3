"""NeST SO(3) MULTI-BRANCH decode tree on a real SYMSOL image -- lifted-Mollweide nodes.

Layout follows the fan tree: the input image is the root node (top); each level fans the MAP node into all
8 children (4 HEALPix splits x 2 Hopf-roll halves), shown as a row of nodes; grey lines are the full branching,
the thick gold spine is the beam-MAP path; the output 3D orientation is the leaf node at the bottom. Every node is
a true equal-area MOLLWEIDE projection of the Hopf-HEALPix base sphere, tilted/lifted into 3D space (a perspective
card): filled HEALPix pixels = the S^2 binning, hue = the Hopf roll fibre, chosen cell = gold.

Data: logs/figs/_cache/so3_dump.npz (dump_so3.py, device 1).
Run:  cd /home/kavinder/SPACE-HOP && PYTHONPATH=src python -I scripts/figs/gen_so3_tree_fig.py
Out:  logs/figs/nest_so3_tree.pdf/.png
"""
import os, sys, numpy as np, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
import so3_nested as so3
import _moll3d as M

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300})
INK = "#272727"; GOLD = "#FFD700"; GOLD_LINE = "#C99200"; BLUE = "#0F4D92"; BLUE2 = "#3775BA"
GREY = "#8a9099"; RED = "#B64342"; GREEN = "#2E8B57"
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/nest_so3_tree"
HSV = mpl.colormaps["hsv"]
PRETTY = {"tet": "tetrahedron", "cube": "cube", "icosa": "icosahedron", "cone": "cone", "cyl": "cylinder"}

d = np.load("logs/figs/_cache/so3_dump.npz")
NLEV = 3                                                   # fan levels shown (root + 3), like nest_tree
raw = d["raw"]; p0 = d["p0"]; root_cell = int(d["root_cell"])
chosen = d["chosen"]; R_map = d["R_map"]; R_gt = d["R_gt"]; err = float(d["err"]); shape = str(d["shape"])
lv = [dict(cells=d[f"cells_{i}"], probs=d[f"probs_{i}"], ci=int(chosen[i])) for i in range(NLEV)]

# ================================ figure ================================
fig = plt.figure(figsize=(16.0, 11.4))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.54, 0.975, "NeST SO(3) decode tree on one real image: the beam-MAP path through the Hopf--HEALPix tree",
        ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
bg.text(0.54, 0.949, "every node is a true Mollweide projection lifted into 3D; filled HEALPix cells $=$ the "
        "$S^2$ binning, hue $=$ the Hopf roll fibre; each branch probability is the model's actual 8-way softmax",
        ha="center", va="center", fontsize=11.5, color=BLUE)
bg.text(0.54, 0.927, "gold $=$ the beam-MAP root$\\to$leaf path (branch width $\\propto$ probability); the chosen "
        "cell refines $\\times8$ (pixel $\\times4$, roll $\\times2$) each level", ha="center", va="center",
        fontsize=10.5, color=GOLD_LINE, fontweight="bold")

ROOT = (0.52, 0.862); ROOT_W, ROOT_H = 0.185, 0.130
row_y = [0.660, 0.470, 0.280]; col_x = np.linspace(0.105, 0.975, 8)
NW, NH = 0.112, 0.090
NSIDE = {1: 10, 2: 9, 4: 8, 8: 7, 16: 6, 32: 6}

def node_at(cx, cy, w, h, nside, nr, cells, prob, cell_sel, step):
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    M.node(ax, nside, nr, cells, prob, cell_sel, step=step)
    return ax

def zoom_node(ax, nside, nr, cell):
    """Cropped neighbourhood around the chosen nside cell: the fine HEALPix grid + the gold MAP cell."""
    from matplotlib.patches import Polygon
    pix = int(cell) // nr; roll = int(cell) % nr
    cvec = hp.pix2vec(nside, pix, nest=True)
    res = np.sqrt(4 * np.pi / (12 * nside ** 2))
    neigh = hp.query_disc(nside, cvec, radius=5 * res, nest=True, inclusive=True)
    for p in neigh:
        poly, seam = M._cell_poly2d(nside, int(p), step=6)
        if seam:
            continue
        if int(p) == pix:
            col = np.array(M.HSV(M.hue(roll, nr)))[:3] * M._shade(hp.pix2vec(nside, int(p), nest=True))
            ax.add_patch(Polygon(poly, closed=True, fc=col, ec=GOLD_LINE, lw=2.4, zorder=4))
        else:
            ax.add_patch(Polygon(poly, closed=True, fc="#eef1f5", ec="#b7bec8", lw=0.7, zorder=3))
    cpoly, _ = M._cell_poly2d(nside, pix, step=6)
    cx0, cy0 = cpoly[:, 0].mean(), cpoly[:, 1].mean()
    span = cpoly[:, 0].max() - cpoly[:, 0].min(); r = 3.2 * span
    ax.set_xlim(cx0 - r, cx0 + r); ax.set_ylim(cy0 - 0.72 * r, cy0 + 0.72 * r)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def roll_dial(cx, cy, r, nr, chosen_roll):
    ax = fig.add_axes([cx - r, cy - r, 2 * r, 2 * r], projection="polar")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 1.0); ax.set_frame_on(False)
    th = (np.arange(nr) + 0.5) * (2 * np.pi / nr)
    ax.bar(th, np.full(nr, 0.52), width=2 * np.pi / nr, bottom=0.42,
           color=[HSV(M.hue(k, nr)) for k in range(nr)], edgecolor="white", linewidth=0.2, zorder=2)
    cth = (int(chosen_roll) + 0.5) * (2 * np.pi / nr)
    ax.plot([cth, cth], [0, 1.02], color=GOLD_LINE, lw=2.0, zorder=5)
    ax.scatter([cth], [1.02], s=18, c=GOLD, edgecolors=GOLD_LINE, linewidths=0.8, zorder=6)

# ---- input image + encoder (top-left) ----
axim = fig.add_axes([0.012, 0.80, 0.11, 0.145]); axim.imshow(raw); axim.axis("off")
bg.text(0.068, 0.788, f"actual input\n({PRETTY[shape]} mask)", ha="center", va="top", fontsize=11, color=INK)
bg.add_patch(FancyBboxPatch((0.028, 0.715), 0.082, 0.045, boxstyle="round,pad=0.004", fc=BLUE2, ec=INK, lw=1.5, zorder=3))
bg.text(0.069, 0.7375, "ViT-B/16\nencoder", ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)
bg.annotate("", xy=(0.069, 0.762), xytext=(0.069, 0.80), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
bg.annotate("", xy=(ROOT[0] - ROOT_W / 2 + 0.01, ROOT[1] - 0.02), xytext=(0.112, 0.74),
            arrowprops=dict(arrowstyle="-|>", color=BLUE2, lw=1.3, alpha=0.5))
bg.text(0.30, 0.75, "tokens condition\nevery node", ha="center", va="center", fontsize=10.5, color=BLUE2, style="italic")

# roll-hue colorbar (top-right)
cax = fig.add_axes([0.945, 0.80, 0.012, 0.11])
sm = mpl.cm.ScalarMappable(cmap="hsv", norm=mpl.colors.Normalize(0, 2 * np.pi))
cb = fig.colorbar(sm, cax=cax, ticks=[0, np.pi, 2 * np.pi]); cb.ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
cb.set_label("in-plane roll (hue)", fontsize=10)

# ---- root node ----
node_at(ROOT[0], ROOT[1], ROOT_W, ROOT_H, 1, 6, np.arange(72), p0, root_cell, 10)
roll_dial(ROOT[0] + ROOT_W / 2 + 0.028, ROOT[1], 0.026, 6, root_cell % 6)
bg.text(ROOT[0] - ROOT_W / 2 - 0.012, ROOT[1] + 0.02, "root  $p_0(c_0\\mid x)$\n72-way $\\cdot$ full density",
        ha="right", va="center", fontsize=12.5, fontweight="bold", color=INK)

# ---- fan levels ----
prev = ROOT
for j, r in enumerate(lv):
    level = j + 1; yc = row_y[j]; nside, nr = so3.nside_of(level), so3.n_roll(level)
    bg.text(0.006, yc, f"$\\ell\\!=\\!{level}$", ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    bg.text(0.006, yc - 0.048, f"8-way\n$p(c_{{{level}}}\\!\\mid\\!\\cdot)$\nnside {nside}", ha="left", va="top",
            fontsize=9.5, color=GREY)
    px, py = prev
    for k in range(8):
        xc = col_x[k]; is_ch = (k == r["ci"]); p = float(r["probs"][k]); cell = int(r["cells"][k])
        bg.plot([px, xc], [py - (ROOT_H if j == 0 else NH) / 2 - 0.004, yc + NH / 2 + 0.004],
                color=(GOLD_LINE if is_ch else GREY), lw=(1.0 + 5.0 * p if is_ch else 0.5),
                alpha=(0.95 if is_ch else 0.3), zorder=(4 if is_ch else 0))
        node_at(xc, yc, NW, NH, nside, nr, np.array([cell]), np.array([p]), cell if is_ch else None,
                NSIDE.get(nside, 8))
        roll_dial(xc, yc - 0.070, 0.021, nr, cell % nr)
        bg.text(xc, yc - 0.098, f"$p={p:.2f}$", ha="center", va="top", fontsize=10,
                color=(GOLD_LINE if is_ch else GREY), fontweight=("bold" if is_ch else "normal"))
    prev = (col_x[r["ci"]], yc)

# ---- leaf node = output 3D orientation (cube wireframe ON the input mask, verified convention) ----
Vc = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
EDG = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
sil = raw.sum(-1) > 30
ys, xs = np.where(sil); u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
ext = max(u1 - u0, v1 - v0); cu = (u0 + u1) / 2; cv = (v0 + v1) / 2
def proj_cube(Rm, Mmode, sy):
    P = (Rm @ Vc.T).T if Mmode == 0 else (Rm.T @ Vc.T).T
    ux = P[:, 0]; uy = sy * P[:, 1]
    e = max(ux.max() - ux.min(), uy.max() - uy.min())
    s = 0.9 * ext / e
    return np.column_stack([cu + s * (ux - (ux.max() + ux.min()) / 2),
                            cv + s * (uy - (uy.max() + uy.min()) / 2)])
def hull_iou(uv):
    try:
        from scipy.spatial import ConvexHull
        poly = uv[ConvexHull(uv).vertices]
    except Exception:
        poly = uv
    gx, gy = np.meshgrid(np.linspace(u0, u1, 90), np.linspace(v0, v1, 90))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = Path(poly).contains_points(pts).reshape(90, 90)
    silc = sil[np.clip(gy.astype(int), 0, sil.shape[0]-1), np.clip(gx.astype(int), 0, sil.shape[1]-1)]
    inter = (inside & silc).sum(); union = (inside | silc).sum()
    return inter / max(union, 1)
best = max([(m, sy) for m in (0, 1) for sy in (1, -1)], key=lambda c: hull_iou(proj_cube(R_gt, *c)))
LY = 0.052; axo = fig.add_axes([0.5 - 0.072, LY, 0.144, 0.120]); axo.imshow(raw); axo.axis("off")
bg.add_patch(FancyBboxPatch((0.5 - 0.081, LY - 0.005), 0.162, 0.131, boxstyle="round,pad=0.003",
                            fc="none", ec=GOLD_LINE, lw=1.8, zorder=1))
def draw_cube(Rm, col, lw, ls="-"):
    uv = proj_cube(Rm, *best)
    for a, b in EDG:
        axo.plot(uv[[a, b], 0], uv[[a, b], 1], color=col, lw=lw, ls=ls, solid_capstyle="round", alpha=0.95)
draw_cube(R_gt, GREEN, 2.6, ls=(0, (4, 2))); draw_cube(R_map, RED, 2.2)
axo.plot([], [], color=GREEN, lw=2.6, ls=(0, (4, 2)), label="nearest sym. GT")
axo.plot([], [], color=RED, lw=2.2, label="NeST MAP")
axo.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=9.5,
           handlelength=1.6, columnspacing=1.2)
bg.annotate("", xy=(0.5, LY + 0.126), xytext=(col_x[lv[-1]["ci"]], row_y[-1] - 0.072),
            arrowprops=dict(arrowstyle="-|>", color=GOLD_LINE, lw=2.4, connectionstyle="arc3,rad=0.0"))
bg.text(0.5 + 0.092, LY + 0.065, f"leaf node =\noutput orientation $R$\ngeodesic err (min over\n{PRETTY[shape]} "
        f"symmetry) $= {err:.1f}^\\circ$", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")

# ---- zoom inset: magnify the tiny ell=3 MAP cell (legible at depth) ----
mci = lv[-1]["ci"]; mcell = int(lv[-1]["cells"][mci]); mp = float(lv[-1]["probs"][mci])
ns3, nr3 = so3.nside_of(NLEV), so3.n_roll(NLEV); mpix, mroll = mcell // nr3, mcell % nr3
snx, sny = col_x[mci], row_y[-1]                                  # the small ell=3 MAP node
bg.add_patch(mpl.patches.Rectangle((snx - NW / 2, sny - NH / 2), NW, NH, fc="none", ec=GOLD_LINE,
                                   lw=1.8, ls=(0, (3, 2)), zorder=6))
IX, IY, IW, IH = 0.818, 0.078, 0.188, 0.122
bg.annotate("", xy=(IX - IW / 2 + 0.012, IY + IH / 2), xytext=(snx + NW / 2 - 0.008, sny - NH / 2),
            arrowprops=dict(arrowstyle="-", color=GOLD_LINE, lw=1.1, ls=(0, (3, 2)), alpha=0.6,
                            connectionstyle="arc3,rad=-0.2"))
bg.add_patch(FancyBboxPatch((IX - IW / 2 - 0.008, IY - IH / 2 - 0.01), IW + 0.016, IH + 0.02,
                            boxstyle="round,pad=0.004", fc="#fbfcfe", ec=GOLD_LINE, lw=1.8, zorder=2))
axz = fig.add_axes([IX - IW / 2, IY - IH / 2, IW, IH]); axz.set_zorder(3)
zoom_node(axz, ns3, nr3, mcell)
bg.text(IX, IY + IH / 2 + 0.014, "zoom: the $\\ell\\!=\\!3$ MAP cell", ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=GOLD_LINE)
roll_dial(IX + IW / 2 + 0.030, IY + 0.020, 0.023, nr3, mroll)
bg.text(IX + IW / 2 + 0.030, IY - 0.028, f"nside {ns3}\npix {mpix}\nroll {mroll}/{nr3}\n$p={mp:.2f}$",
        ha="center", va="top", fontsize=9.5, color=INK)

for ext_ in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext_}", dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png  (cube convention M={best[0]} sy={best[1]})")
