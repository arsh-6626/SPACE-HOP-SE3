"""NeST unified figure: ONE top-down tree that is BOTH the architecture and the decode/selection process.
Reading top-down: a shared ViT-B/16 backbone (patch tokens X, a K/V rail) feeds, at every node, the network block
that computes that node's categorical (root: 72 learned queries -> cross-attn -> softmax; child: parent code (+)
offset -> Fourier -> MLP -> cross-attn -> softmax, shared weights reused at every level). Each node shows the REAL
softmax it produced and branches to its children; gold = the beam-MAP child, which becomes the parent code of the
next level (the recurrence). Real forward pass: YCB-V obj 3, scene 51/1629.

Run: cd /home/kavinder/SPACE-HOP && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -I scratchpad/gen_tree_arch_fig.py
Out: logs/figs/nest_tree_arch.pdf/.png  (print size ~7.4in wide; verify downscaled)
"""
import os, sys, json, numpy as np, torch, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
sys.path.insert(0, "src")
import se3_tless_data as td, se3_decode as dec, se3_bop_eval as be, se3_head as sh
import se3_tree as se3, so3_nested as so3
from se3_head import TRANS_OCT_OFF
from PIL import Image

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300})
BOP = "/media/kavinder/hdd2/liepose_data/bop_datasets/ycbv"; SPLIT = "test"; DEV = "cuda"; OBJ = 3
CKPT = f"checkpoints/se3_ycbv_aug_obj{OBJ}/best.pth"
IMN_M = np.array([0.485, 0.456, 0.406], np.float32); IMN_S = np.array([0.229, 0.224, 0.225], np.float32)
INK = "#232323"; ENC = "#0F4D92"; ENC_F = "#DCE7F4"; SCO = "#5B3B8C"; SCO_F = "#EADFF6"
SM = "#2E8B57"; SM_F = "#DCEFE4"; GOLD = "#FFD700"; GOLD_L = "#B8860B"; OR = "#C4661F"; OR_F = "#F6E6D7"
GREY = "#777d84"; RED = "#B64342"
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/nest_tree_arch"
HSV = mpl.colormaps["hsv"]

# ============================ real forward pass ============================
enc, head, L_rot, L_trans, crop_pad, out_px = be._load_ckpt(CKPT, DEV)
d_m = td.diameters_m(BOP, [OBJ])[OBJ]; mi = td.load_models_info(BOP)[OBJ]
dets = [x for x in json.load(open("data/detections/gdrnppdet_ycbv-test.json")) if x["category_id"] == OBJ]
det = next(x for x in dets if x["scene_id"] == 51 and x["image_id"] == 1629)
scene, im, bbox = det["scene_id"], det["image_id"], np.array(det["bbox"], np.float64)
sdir = f"{BOP}/{SPLIT}/{scene:06d}"
K = np.array(json.load(open(f"{sdir}/scene_camera.json"))[str(im)]["cam_K"], np.float64).reshape(3, 3)
img = np.asarray(Image.open(td.rgb_path(BOP, SPLIT, scene, im)).convert("RGB"), np.uint8)
t_hat = be._t_hat_from_bbox(bbox, K, d_m); side_px = d_m / t_hat[2] * K[0, 0] * crop_pad
cu, cv = td.project(K, t_hat); crop = td.crop_resize(img, cu, cv, side_px, out_px)
x = (crop.astype(np.float32) / 255.0 - IMN_M) / IMN_S
patches = enc(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV)).float()
P = patches.shape[1]

# beam-MAP leaf and its ancestry (the selection path)
rc_fin, tc_fin, _ = dec.beam_map(head, patches[0], L_rot, L_trans, beam=64)
def rot_anc(cf, Lf, L):
    nrf = so3.n_roll(Lf); pix, roll = cf // nrf, cf % nrf; k = Lf - L
    return (pix // 4 ** k) * so3.n_roll(L) + (roll // 2 ** k)
def trans_anc(cf, Lf, L):
    ix, iy, iz = (int(v[0]) for v in se3._trans_flat_to_coords(np.array([cf]), Lf)); k = Lf - L
    return int(se3._trans_coords_to_flat(ix >> k, iy >> k, iz >> k, L))

@torch.no_grad()
def level_J(l):
    rp = int(rot_anc(rc_fin, L_rot, l - 1)); tp = int(trans_anc(tc_fin, L_trans, l - 1))
    chf, par_flat, roff = dec._rot_child_geom(rp, l - 1); tch = dec._trans_children(tp, l - 1)
    tctr = se3.trans_cell_center(np.array([tp]), l - 1)[0].astype(np.float32)
    off6 = np.concatenate([np.repeat(roff[:, None, :], 8, 1),
                           np.broadcast_to(TRANS_OCT_OFF[None], (8, 8, 3))], -1).reshape(64, 6).astype(np.float32)
    par12 = np.concatenate([par_flat, tctr]).astype(np.float32)
    clp = head.child_logp(patches, torch.as_tensor(off6, device=DEV)[None],
                          torch.as_tensor(par12, device=DEV)[None], l - 1)[0].cpu().numpy()
    Jm = np.exp(clp - clp.max()).reshape(8, 8); Jm /= Jm.sum()
    rc = int(np.where(chf == rot_anc(rc_fin, L_rot, l))[0][0])
    oc = int(np.where(tch == trans_anc(tc_fin, L_trans, l))[0][0])
    return Jm, rc, oc, Jm.sum(1), float(Jm[rc, oc])
lp0 = head.logp0(patches)[0]; p0 = lp0.exp().cpu().numpy(); root_rc = int(lp0.argmax())
LV = {l: level_J(l) for l in range(1, 6)}
R_map = so3.cell_index_to_matrix(np.array([rc_fin]), L_rot)[0]
t_map = t_hat + td.box_A(d_m, t_hat) @ se3.trans_cell_center(np.array([tc_fin]), L_trans)[0]
print("root_rc", root_rc, "levels:", {l: (LV[l][1], LV[l][2], round(LV[l][4], 2)) for l in LV})

# ============================ helpers ============================
def _hue(roll, nr): return (0.5 + (roll + 0.5) / nr) % 1.0
def _pix_poly(nside, pix, step=10):
    v = hp.boundaries(nside, int(pix), step=step, nest=True).T
    th, ph = hp.vec2ang(v); lon = (ph + np.pi) % (2 * np.pi) - np.pi; lat = np.pi / 2 - th
    m = np.arctan2(np.sin(lon).mean(), np.cos(lon).mean()); lon = m + ((lon - m + np.pi) % (2 * np.pi) - np.pi)
    return np.column_stack([np.clip(lon, -np.pi + 1e-6, np.pi - 1e-6), lat]), bool((np.abs(lon) > np.pi - 2e-3).any())

def rbox(x, y, w, h, text, fc="white", ec=INK, fs=6.6, tc=INK, lw=1.0, bold=False, rs=0.010, z=3):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.002,rounding_size={rs}", fc=fc, ec=ec, lw=lw, zorder=z))
    bg.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=z + 1)

def arrow(x0, y0, x1, y1, color=INK, lw=1.3, rad=0.0, ls="-"):
    bg.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))

# ============================ figure ============================
fig = plt.figure(figsize=(7.6, 9.9))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.5, 0.984, "NeST: the network and the SE(3) decode are one top-down tree",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
bg.text(0.5, 0.966, "a shared ViT-B/16 feeds, at every node, the block that computes that node's categorical; "
        "gold = the beam-MAP child, which becomes the next level's parent code",
        ha="center", va="center", fontsize=7.2, color=ENC)

XS, XN = 0.255, 0.655           # scorer-block column centre, tree-node column centre
# ---------- shared backbone (top) ----------
axcr = fig.add_axes([0.055, 0.895, 0.085, 0.072]); axcr.imshow(crop.astype(np.uint8)); axcr.axis("off")
bg.text(0.0975, 0.888, "crop 224$^2$", ha="center", va="top", fontsize=6.2, color=INK)
arrow(0.142, 0.930, 0.168, 0.930, ENC)
rbox(0.168, 0.905, 0.150, 0.052, "ViT-B/16 (augreg2)\n12 transformer blocks\nfine-tuned (lr $10^{-5}$)", ENC_F, ENC, fs=6.6)
arrow(0.318, 0.930, 0.345, 0.930, ENC)
rbox(0.345, 0.912, 0.130, 0.040, "patch tokens\n$X\\in\\mathbb{R}^{196\\times768}$", "white", ENC, fs=6.8, bold=True)
# K/V rail down the left margin
RAIL = 0.040
bg.plot([0.410, RAIL], [0.912, 0.912], color=ENC, lw=1.5, zorder=1)
bg.plot([RAIL, RAIL], [0.912, 0.150], color=ENC, lw=1.5, zorder=1)
bg.text(RAIL + 0.004, 0.52, "patch tokens $X$  ($K,V$) condition every node", rotation=90, ha="left", va="center",
        fontsize=6.6, color=ENC, style="italic")

# ---------- per-node scorer block ----------
def scorer(y, kind, label):
    """Draw the network block that computes a node's categorical, as a compact top-down mini-flow. Returns top y."""
    w = 0.300; x0 = XS - w / 2
    if kind == "root":
        rows = ["72 learned queries  Emb$(72{\\times}768)$", "cross-attn $\\times2$  (Q; $K,V{=}X$)", "72 logits $\\to$ softmax"]
        cols = [SCO_F, SCO_F, SM_F]; ecs = [SCO, SCO, SM]
    else:
        rows = ["parent code $[R,\\hat g]$  $\\oplus$  child offset", "Fourier $\\pi2^k$  +  child_proj MLP",
                "cross-attn $\\times2$  (Q; $K,V{=}X$)", "64 logits $\\to$ softmax"]
        cols = [OR_F, OR_F, SCO_F, SM_F]; ecs = [OR, OR, SCO, SM]
    n = len(rows); hh = 0.019; gap = 0.006; H = n * hh + (n - 1) * gap
    ytop = y + H / 2
    bg.add_patch(FancyBboxPatch((x0 - 0.012, y - H / 2 - 0.012), w + 0.024, H + 0.024,
                 boxstyle="round,pad=0.002,rounding_size=0.010", fc="white", ec=SCO, lw=1.2, zorder=2))
    bg.text(XS, ytop + 0.020, label, ha="center", va="bottom", fontsize=6.6, color=SCO, fontweight="bold")
    yy = ytop
    attn_y = None
    for i, (r, fc, ec) in enumerate(zip(rows, cols, ecs)):
        rbox(x0, yy - hh, w, hh, r, fc, ec, fs=6.0, rs=0.006, z=4)
        if "cross-attn" in r: attn_y = yy - hh / 2
        if i < n - 1: arrow(XS, yy - hh, XS, yy - hh - gap, GREY, lw=0.9)
        yy -= hh + gap
    if attn_y is not None:                                   # K/V tap from the rail
        bg.plot([RAIL, x0 - 0.012], [attn_y, attn_y], color=ENC, lw=0.9, ls=(0, (3, 2)), zorder=1)
    return ytop, y - H / 2

# ---------- node panel (real softmax) + branch fan ----------
def root_node(y):
    axp = fig.add_axes([XN - 0.075, y - 0.050, 0.150, 0.100], projection="mollweide")
    axp.set_xticks([]); axp.set_yticks([]); axp.grid(True, alpha=0.1, lw=0.3)
    nr0 = so3.n_roll(0); pix0 = np.arange(72) // nr0; roll0 = np.arange(72) % nr0; pmax = p0.max()
    for pp in range(12):
        sel = pix0 == pp; pm = p0[sel].sum(); rb = int(roll0[sel][np.argmax(p0[sel])])
        poly, seam = _pix_poly(1, pp)
        if seam: continue
        axp.add_patch(Polygon(poly, closed=True, fc=HSV(_hue(rb, nr0)), ec="#555", lw=0.4,
                              alpha=float(np.clip(0.2 + 0.75 * pm / pmax, 0.12, 0.95)), zorder=2))
        if pp == root_rc // nr0:
            axp.add_patch(Polygon(poly, closed=True, fc="none", ec=GOLD_L, lw=1.8, zorder=5))
    for sp in axp.spines.values(): sp.set_edgecolor(GOLD_L); sp.set_linewidth(1.4)

def child_node(y, J, rc, oc):
    axp = fig.add_axes([XN - 0.050, y - 0.050, 0.100, 0.100])
    axp.imshow(J, cmap="magma", aspect="equal", origin="upper")
    axp.add_patch(Rectangle((oc - 0.5, rc - 0.5), 1, 1, fill=False, ec=GOLD, lw=1.8))
    axp.set_xticks([0, 7]); axp.set_yticks([0, 7]); axp.tick_params(labelsize=4.6, length=2)
    axp.set_xlabel("trans octant", fontsize=5.4, labelpad=0.5); axp.set_ylabel("rot child", fontsize=5.4, labelpad=0.5)

def fan(y_from, y_to, p_choice, label_mid=None):
    """Branch from a node (y_from) to the next (y_to): faded pruned siblings + a gold MAP edge down the spine."""
    for dx in (-0.075, -0.045, 0.045, 0.075, 0.10):
        bg.plot([XN, XN + dx], [y_from - 0.052, y_from - 0.085], color=GREY, lw=0.7, alpha=0.35, zorder=1)
    bg.annotate("", xy=(XN, y_to + 0.052), xytext=(XN, y_from - 0.052), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=GOLD_L, lw=1.2 + 3.0 * p_choice, alpha=0.95))
    if label_mid:
        bg.text(XN + 0.012, (y_from + y_to) / 2 - 0.016, label_mid, ha="left", va="center", fontsize=6.0,
                color=GOLD_L, fontweight="bold")

# ---------- lay out the tree (root + l1 + l2 + ... + leaf) ----------
rows_y = {"root": 0.800, "l1": 0.628, "l2": 0.456, "dots": 0.360, "leaf": 0.270}

# root
scorer(rows_y["root"], "root", "ROOT node — rotation ($\\ell{=}0$)")
root_node(rows_y["root"])
bg.text(XN + 0.082, rows_y["root"], "$p_0(c_0\\mid x)$\n72-way rotation\n(HEALPix S$^2$,\nhue = roll)",
        ha="left", va="center", fontsize=6.0, color=SM, fontweight="bold")
arrow(XS + 0.152, rows_y["root"], XN - 0.078, rows_y["root"], SCO, lw=1.1)   # scorer -> node

# children l1,l2
for lv, yk, prev_yk in [(1, "l1", "root"), (2, "l2", "l1")]:
    J, rc, oc, rp, jp = LV[lv]
    lab = "CHILD node ($\\ell{=}%d$) — child_attn" % lv + ("" if lv == 1 else "  (shared weights)")
    scorer(rows_y[yk], "child", lab)
    child_node(rows_y[yk], J, rc, oc)
    bg.text(XN + 0.058, rows_y[yk], "$p(c_{%d}\\mid\\cdot)$\n64-way\n($8\\,$rot$\\times8\\,$oct)" % lv,
            ha="left", va="center", fontsize=6.0, color=SM, fontweight="bold")
    arrow(XS + 0.152, rows_y[yk], XN - 0.053, rows_y[yk], SCO, lw=1.1)
    fan(rows_y[prev_yk], rows_y[yk], jp)
    ymid = (rows_y[prev_yk] + rows_y[yk]) / 2
    bg.text(XN - 0.016, ymid + 0.014, "MAP child  $p=%.2f$" % jp, ha="right", va="center",
            fontsize=6.0, color=GOLD_L, fontweight="bold")                  # path label, left of spine
    bg.text(XN - 0.016, ymid - 0.012, "chosen cell $\\to$ next parent code", ha="right", va="center",
            fontsize=5.6, color=OR, style="italic")

# collapsed deeper levels
J3, rc3, oc3, rp3, jp3 = LV[2]
fan(rows_y["l2"], rows_y["dots"], jp3)
bg.text(XN, rows_y["dots"], "$\\vdots$   child_attn reused, $\\ell{=}3,4,5$   (same block, deeper cells)",
        ha="center", va="center", fontsize=7.0, color=SCO, fontweight="bold")
# leaf
bg.annotate("", xy=(XN, rows_y["leaf"] + 0.030), xytext=(XN, rows_y["dots"] - 0.016), zorder=2,
            arrowprops=dict(arrowstyle="-|>", color=GOLD_L, lw=2.2))
rbox(XN - 0.145, rows_y["leaf"] - 0.004, 0.290, 0.038,
     "leaf cell $\\to$  $R=$ cell centre,   $t=\\hat t + A\\,g$", "white", INK, fs=6.8, bold=True)

# ---------- readout: MAP 6D box ----------
mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]]) / 1000.0
szb = np.array([mi["size_x"], mi["size_y"], mi["size_z"]]) / 1000.0
corners = np.array([[mn[0]+a*szb[0], mn[1]+b*szb[1], mn[2]+c*szb[2]] for a in (0,1) for b in (0,1) for c in (0,1)])
EDG = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
uv = (K @ (R_map @ corners.T + t_map[:, None])); uv = (uv[:2] / uv[2]).T
c0 = uv.min(0); c1 = uv.max(0); ctr = (c0 + c1) / 2; half = (c1 - c0).max() / 2 + 0.5 * (c1 - c0).max()
x0, y0 = max(0, int(ctr[0]-half)), max(0, int(ctr[1]-half))
x1, y1 = min(img.shape[1], int(ctr[0]+half)), min(img.shape[0], int(ctr[1]+half))
arrow(XN, rows_y["leaf"] - 0.004, XN, 0.205, INK)
axb = fig.add_axes([XN - 0.070, 0.110, 0.140, 0.095]); axb.imshow(img[y0:y1, x0:x1]); axb.axis("off")
for e in EDG:
    axb.plot(uv[list(e), 0] - x0, uv[list(e), 1] - y0, color=RED, lw=1.4)
bg.text(XN, 0.104, "MAP 6D pose $(R,t)\\to$ 3D box", ha="center", va="top", fontsize=6.6, color=INK, fontweight="bold")

# gold callout
bg.add_patch(FancyBboxPatch((0.085, 0.030), 0.830, 0.038, boxstyle="round,pad=0.004,rounding_size=0.01",
                            fc="#FFF8DC", ec=GOLD_L, lw=1.2, zorder=3))
bg.text(0.5, 0.049, "every node is a log-softmax of a shared-weight attention block  $\\Rightarrow$  the tree IS a "
        "normalized SE(3) density; zero-init output $\\Rightarrow$ starts at the uniform floor",
        ha="center", va="center", fontsize=6.8, color=GOLD_L, fontweight="bold", zorder=5)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
