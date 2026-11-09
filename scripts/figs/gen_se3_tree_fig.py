"""NeST SE(3) MULTI-BRANCH decode tree on a real YCB-V image -- shaded 3D GLOBE nodes.

Fan tree (like nest_tree.png): input image = root node (top); each level fans the MAP node's 8 ROTATION children
(4 HEALPix pixel splits x 2 Hopf-roll halves) as a row of shaded 3D globes of the Hopf-HEALPix S^2 base, with the
translation octree (the other factor of the 64-way joint child) drawn as an isometric octant cube on the gold MAP
node; the output 6D pose = leaf node (bottom), the predicted 3D box on the RGB image. Each globe shows the density
(alpha = probability, hue = Hopf roll fibre); the chosen rotation bin is gold-outlined with a Hopf fibre ring.

Data: logs/figs/_cache/se3_dump.npz (dump_se3.py, device 1).
Run:  cd /home/kavinder/SPACE-HOP && PYTHONPATH=src python -I scripts/figs/gen_se3_tree_fig.py
Out:  logs/figs/nest_se3_tree.pdf/.png
"""
import os, sys, numpy as np, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
import so3_nested as so3
from se3_head import TRANS_OCT_OFF
import _globe as G

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300})
INK = "#272727"; GOLD = "#FFD700"; GOLD_LINE = "#C99200"; BLUE = "#0F4D92"; BLUE2 = "#3775BA"
GREY = "#8a9099"; RED = "#B64342"; GREEN = "#2E8B57"
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/nest_se3_tree"
HSV = mpl.colormaps["hsv"]

d = np.load("logs/figs/_cache/se3_dump.npz")
NLEV = 3
p0 = d["p0"]; root_rc = int(d["root_rc"]); rc = d["rc"]; to = d["to"]; jointp = d["jointp"]
leaf_img = d["leaf_img"]; uvp = d["uvp"]; uvg = d["uvg"]
ang = float(d["ang"]); terr = float(d["terr"]); scene = int(d["scene"]); im = int(d["im"])
lv = [dict(chf=d[f"chf_{i}"], rotp=d[f"rotp_{i}"], transp=d[f"transp_{i}"], rc=int(rc[i]), to=int(to[i]))
      for i in range(NLEV)]

# one shared orthographic view, aimed ~26 deg off the MAP leaf rotation base point
nsL, nrL = so3.nside_of(NLEV), so3.n_roll(NLEV)
leaf_cell0 = int(lv[-1]["chf"][lv[-1]["rc"]])
map_base = np.array(hp.pix2vec(nsL, leaf_cell0 // nrL, nest=True))
R = G.make_view(map_base, delta_deg=26.0)

# ================================ figure ================================
fig = plt.figure(figsize=(16.0, 11.4))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.54, 0.975, "NeST SE(3) decode tree on one real YCB-V image: the beam-MAP path through the joint tree",
        ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
bg.text(0.54, 0.949, "each level is a 64-way child $=$ 8 rotations (shaded $S^2$ globes; filled HEALPix cells $=$ "
        "density, hue $=$ Hopf roll fibre) $\\times$ 8 translation octants (octree cube on the MAP node)",
        ha="center", va="center", fontsize=11.5, color=BLUE)
bg.text(0.54, 0.927, "gold $=$ the beam-MAP root$\\to$leaf path (branch width $\\propto$ joint probability); the "
        "bin+fibre refines rotation $\\times8$ and translation $\\times8$ each level", ha="center", va="center",
        fontsize=10.5, color=GOLD_LINE, fontweight="bold")

ROOT = (0.52, 0.836); ROOT_W, ROOT_H = 0.148, 0.148
row_y = [0.672, 0.486, 0.300]; col_x = np.linspace(0.105, 0.975, 8)
NW, NH = 0.108, 0.108


def node_at(cx, cy, w, h, nside, nr, cells, prob, gold_cell, focus_cell, draw_fibre):
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    base_c = np.array(hp.pix2vec(nside, int(focus_cell) // nr, nest=True))
    G.globe(ax, R, nside, nr, cells, prob, (gold_cell if gold_cell is not None else -1),
            base_c=base_c, chosen_roll=int(focus_cell) % nr, fibre=draw_fibre, frame=False)


def globe_zoom(ax, nside, nr, cell, prob):
    """Cropped globe on the MAP rotation cell: fine HEALPix neighbourhood (binning) + gold bin + Hopf fibre ring."""
    base_c = np.array(hp.pix2vec(nside, int(cell) // nr, nest=True))
    G.shaded_disk(ax); G.graticule(ax, R)
    res = np.sqrt(4 * np.pi / (12 * nside ** 2))
    for p in hp.query_disc(nside, base_c, radius=6 * res, nest=True, inclusive=True):
        poly = G._clamp_poly(R, hp.boundaries(nside, int(p), step=6, nest=True).T)
        if poly is not None:
            ax.add_patch(Polygon(poly, closed=True, fc="none", ec="#aeb6c1", lw=0.7, zorder=2))
    G.draw_cells(ax, R, nside, nr, np.array([cell]), np.array([prob]), cell, step=10, context=False)
    G.fibre_ring(ax, R, base_c, nr, int(cell) % nr, rad=0.14)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#8f97a2", lw=1.0, zorder=6))
    cxy, _ = G.project(R, base_c[None]); czx, czy = float(cxy[0, 0]), float(cxy[0, 1]); hw = 0.26
    ax.set_xlim(czx - hw, czx + hw); ax.set_ylim(czy - hw, czy + hw)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --- isometric translation octree cube ---
def iso(p): return np.array([(p[0] - p[1]) * 0.866, p[2] * 0.95 + (p[0] + p[1]) * 0.5])
SHADE = {"top": 1.0, "right": 0.80, "left": 0.63}
def _oct_faces(c, hs):
    def P(dx, dy, dz): return iso(c + np.array([dx, dy, dz]) * hs)
    return {"top": [P(-1,-1,1), P(1,-1,1), P(1,1,1), P(-1,1,1)],
            "right": [P(1,-1,-1), P(1,1,-1), P(1,1,1), P(1,-1,1)],
            "left": [P(-1,1,-1), P(1,1,-1), P(1,1,1), P(-1,1,1)]}
def cube_glyph(cx, cy, w, h, trans_p, chosen):
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h]); ax.axis("off")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.85, 2.0); ax.set_aspect("equal")
    hs = 0.5; smax = trans_p.max() if trans_p.max() > 0 else 1.0
    centers = {k: TRANS_OCT_OFF[k].astype(float) for k in range(8)}
    order = sorted([k for k in range(8) if k != chosen], key=lambda k: -iso(centers[k] + [0, 0, 1e-4])[1])
    def draw(k, gold):
        c = centers[k]; p = trans_p[k] / smax; col = GOLD if gold else BLUE2; ec = GOLD_LINE if gold else "#5f7fae"
        for fn, fv in _oct_faces(c, hs).items():
            a = (0.92 if gold else 0.15 + 0.6 * p) * SHADE[fn]
            ax.add_patch(Polygon(fv, closed=True, fc=col, ec=ec, lw=0.8 if gold else 0.5, alpha=a,
                                 zorder=3 if gold else 2))
    for k in order: draw(k, False)
    draw(chosen, True)


def roll_dial(cx, cy, r, nr, chosen_roll):
    ax = fig.add_axes([cx - r, cy - r, 2 * r, 2 * r], projection="polar")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 1.0); ax.set_frame_on(False)
    th = (np.arange(nr) + 0.5) * (2 * np.pi / nr)
    ax.bar(th, np.full(nr, 0.52), width=2 * np.pi / nr, bottom=0.42,
           color=[HSV(G.hue(k, nr)) for k in range(nr)], edgecolor="white", linewidth=0.2, zorder=2)
    cth = (int(chosen_roll) + 0.5) * (2 * np.pi / nr)
    ax.plot([cth, cth], [0, 1.02], color=GOLD_LINE, lw=2.0, zorder=5)
    ax.scatter([cth], [1.02], s=18, c=GOLD, edgecolors=GOLD_LINE, linewidths=0.8, zorder=6)


# ---- input image + encoder (top-left) ----
axim = fig.add_axes([0.012, 0.80, 0.11, 0.145]); axim.imshow(d["crop"].astype(np.uint8)); axim.axis("off")
bg.text(0.068, 0.788, "actual input\n(obj 3: sugar box)", ha="center", va="top", fontsize=11, color=INK)
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

# ---- root node: the full 72-way rotation density ----
node_at(ROOT[0], ROOT[1], ROOT_W, ROOT_H, 1, 6, np.arange(72), p0, root_rc, root_rc, True)
roll_dial(ROOT[0] + ROOT_W / 2 + 0.024, ROOT[1], 0.026, 6, root_rc % 6)
bg.text(ROOT[0] - ROOT_W / 2 - 0.010, ROOT[1] + 0.02, "root  $p_0(c_0\\mid x)$\n72-way rotation",
        ha="right", va="center", fontsize=12.5, fontweight="bold", color=INK)

# ---- fan levels ----
prev = ROOT
for j, r in enumerate(lv):
    level = j + 1; yc = row_y[j]; nside, nr = so3.nside_of(level), so3.n_roll(level)
    bg.text(0.006, yc, f"$\\ell\\!=\\!{level}$", ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    bg.text(0.006, yc - 0.058, f"64-way\n$8\\times8$\nnside {nside}", ha="left", va="top", fontsize=9.5, color=GREY)
    px, py = prev
    for k in range(8):
        xc = col_x[k]; is_ch = (k == r["rc"]); p = float(r["rotp"][k]); cell = int(r["chf"][k])
        bg.plot([px, xc], [py - (ROOT_H if j == 0 else NH) / 2 - 0.002, yc + NH / 2 + 0.002],
                color=(GOLD_LINE if is_ch else GREY), lw=(1.0 + 5.0 * float(jointp[j]) if is_ch else 0.5),
                alpha=(0.95 if is_ch else 0.3), zorder=(4 if is_ch else 0))
        node_at(xc, yc, NW, NH, nside, nr, np.array([cell]), np.array([p]),
                cell if is_ch else None, cell, is_ch)
        roll_dial(xc, yc - 0.066, 0.020, nr, cell % nr)
        bg.text(xc, yc - 0.092, f"rot $p={p:.2f}$", ha="center", va="top", fontsize=9.5,
                color=(GOLD_LINE if is_ch else GREY), fontweight=("bold" if is_ch else "normal"))
    # translation octree cube beside the MAP node (the other factor of the 64-way joint)
    mapx = col_x[r["rc"]]; cbx = mapx - 0.064 if mapx > 0.85 else mapx + 0.064
    cube_glyph(cbx, yc, 0.058, 0.058, r["transp"], r["to"])
    sgn = "".join("+" if v > 0 else "−" for v in TRANS_OCT_OFF[r["to"]])
    bg.text(cbx, yc + 0.038, f"trans octant $({sgn})$\n$p={r['transp'][r['to']]:.2f}$", ha="center",
            va="bottom", fontsize=8.0, color=GOLD_LINE, fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    prev = (mapx, yc)

# ---- leaf node = output 6D pose -> 3D box on the RGB image ----
EDGES = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
LY = 0.052; axL = fig.add_axes([0.5 - 0.072, LY, 0.144, 0.120]); axL.imshow(leaf_img); axL.axis("off")
bg.add_patch(FancyBboxPatch((0.5 - 0.081, LY - 0.005), 0.162, 0.131, boxstyle="round,pad=0.003",
                            fc="none", ec=GOLD_LINE, lw=1.8, zorder=1))
for uv, col, lab in [(uvg, GREEN, "ground truth"), (uvp, RED, "NeST MAP")]:
    for e in EDGES: axL.plot(uv[list(e), 0], uv[list(e), 1], color=col, lw=2.2)
    axL.plot([], [], color=col, lw=2.2, label=lab)
axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=9.5,
           handlelength=1.6, columnspacing=1.2)
bg.annotate("", xy=(0.5, LY + 0.126), xytext=(col_x[lv[-1]["rc"]], row_y[-1] - 0.070),
            arrowprops=dict(arrowstyle="-|>", color=GOLD_LINE, lw=2.4, connectionstyle="arc3,rad=0.0"))
bg.text(0.5 + 0.092, LY + 0.065, f"leaf node =\noutput 6D pose $\\to$ 3D box\nscene {scene}/{im}:  rot "
        f"${ang:.1f}^\\circ$,  trans {terr:.0f} mm", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")

# ---- zoom inset: the ell=3 MAP rotation bin + its Hopf fibre, on a cropped globe ----
mci = lv[-1]["rc"]; mcell = int(lv[-1]["chf"][mci]); mp = float(lv[-1]["rotp"][mci])
ns3, nr3 = so3.nside_of(NLEV), so3.n_roll(NLEV); mpix, mroll = mcell // nr3, mcell % nr3
snx, sny = col_x[mci], row_y[-1]
bg.add_patch(mpl.patches.Rectangle((snx - NW / 2, sny - NH / 2), NW, NH, fc="none", ec=GOLD_LINE,
                                   lw=1.8, ls=(0, (3, 2)), zorder=6))
IX, IY, IW, IH = 0.818, 0.080, 0.150, 0.150
bg.annotate("", xy=(IX - IW / 2 + 0.012, IY + IH / 2 - 0.01), xytext=(snx + NW / 2 - 0.012, sny - NH / 2 + 0.01),
            arrowprops=dict(arrowstyle="-", color=GOLD_LINE, lw=1.1, ls=(0, (3, 2)), alpha=0.6,
                            connectionstyle="arc3,rad=-0.2"))
bg.add_patch(FancyBboxPatch((IX - IW / 2 - 0.010, IY - IH / 2 - 0.012), IW + 0.020, IH + 0.024,
                            boxstyle="round,pad=0.004", fc="#fbfcfe", ec=GOLD_LINE, lw=1.8, zorder=2))
axz = fig.add_axes([IX - IW / 2, IY - IH / 2, IW, IH]); axz.set_zorder(3)
globe_zoom(axz, ns3, nr3, mcell, mp)
bg.text(IX, IY + IH / 2 + 0.016, "zoom: the $\\ell\\!=\\!3$ MAP bin + fibre", ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=GOLD_LINE)
bg.text(IX + IW / 2 + 0.030, IY, f"nside {ns3}\npix {mpix}\nroll {mroll}/{nr3}\nrot $p={mp:.2f}$",
        ha="left", va="center", fontsize=9.5, color=INK)

for ext_ in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext_}", dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
