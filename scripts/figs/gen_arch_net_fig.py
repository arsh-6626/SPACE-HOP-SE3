"""NeST architecture figure -- paper-ready, top-down, drawn from the ACTUAL network (se3_head.Se3DensityHead on a
ViT-B/16 encoder) and a REAL forward pass (YCB-V obj 3, scene 51/1629). Boxes carry real tensor shapes read from the
checkpoint; the two output panels and the two attention overlays are the network's genuine tensors (72-way root
softmax; 64-way child softmax = 8 rot x 8 oct; the chosen query's real cross-attention over the 14x14 patch grid).

Run:  cd /home/kavinder/SPACE-HOP && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -I scratchpad/gen_arch_net_fig.py
Out:  logs/figs/nest_arch_net.pdf/.png  (print size ~7.4in wide; verify downscaled)
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
SM = "#2E8B57"; SM_F = "#DCEFE4"; GOLD = "#FFD700"; GOLD_L = "#B8860B"; ORâ = "#C4661F"
OR = "#C4661F"; OR_F = "#F6E6D7"; GREY = "#6b7177"; RED = "#B64342"; GREEN = "#2E8B57"
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/nest_arch_net"

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
patches = enc(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV)).float()   # (1,196,768)
P = patches.shape[1]; G = int(round(P ** 0.5)); D = patches.shape[2]
A = td.box_A(d_m, t_hat)

@torch.no_grad()
def attn_of(module, q):                                                       # replay _CrossAttn, keep last-layer attn
    kv = module.kv_norm(patches); w = None
    for Ly in module.layers:
        o, w = Ly['attn'](Ly['n1'](q), kv, kv, need_weights=True, average_attn_weights=True)
        q = q + o; q = q + Ly['mlp'](Ly['n2'](q))
    return w                                                                  # (1,Nq,P)

lp0 = head.logp0(patches)[0]; p0 = lp0.exp().cpu().numpy(); root_rc = int(lp0.argmax())
w0 = attn_of(head.l0_attn, head.l0_query.weight.unsqueeze(0))
root_att = w0[0, root_rc].cpu().numpy().reshape(G, G)
# child level-1 at the MAP root node: real 64-way softmax + chosen-child attention
chf, par_flat, roff = dec._rot_child_geom(root_rc, 0); tch = dec._trans_children(0, 0)
tctr = se3.trans_cell_center(np.array([0]), 0)[0].astype(np.float32)
off6 = np.concatenate([np.repeat(roff[:, None, :], 8, 1),
                       np.broadcast_to(TRANS_OCT_OFF[None], (8, 8, 3))], -1).reshape(64, 6).astype(np.float32)
par12 = np.concatenate([par_flat, tctr]).astype(np.float32)
off_t = torch.as_tensor(off6, device=DEV)[None]; par_t = torch.as_tensor(par12, device=DEV)[None]
clp = head.child_logp(patches, off_t, par_t, 0)[0]; J = clp.exp().cpu().numpy().reshape(8, 8)
ci, oj = np.unravel_index(J.argmax(), J.shape)
qf = sh.fourier_encode(off_t, head.n_freq)
pf = sh.fourier_encode(par_t, head.n_freq).unsqueeze(1).expand(-1, 64, -1)
qchild = head.child_proj(torch.cat([qf, pf], -1)) + head.level_emb.weight[0]
wc = attn_of(head.child_attn, qchild); child_att = wc[0, ci * 8 + oj].cpu().numpy().reshape(G, G)
# MAP pose for the readout
rc_fin, tc_fin, _ = dec.beam_map(head, patches[0], L_rot, L_trans, beam=64)
R_map = so3.cell_index_to_matrix(np.array([rc_fin]), L_rot)[0]
t_map = t_hat + A @ se3.trans_cell_center(np.array([tc_fin]), L_trans)[0]
N_cells = 72 * 8 ** L_rot * 8 ** L_trans
print(f"P={P} G={G} D={D} root_rc={root_rc} child(rot,oct)=({ci},{oj}) p={J.max():.2f} N={N_cells:.3e}")

# ============================ helpers ============================
HSV = mpl.colormaps["hsv"]
def _hue(roll, nr): return (0.5 + (roll + 0.5) / nr) % 1.0
def _pix_poly(nside, pix, step=10):
    v = hp.boundaries(nside, int(pix), step=step, nest=True).T
    th, ph = hp.vec2ang(v); lon = (ph + np.pi) % (2 * np.pi) - np.pi; lat = np.pi / 2 - th
    m = np.arctan2(np.sin(lon).mean(), np.cos(lon).mean())
    lon = m + ((lon - m + np.pi) % (2 * np.pi) - np.pi)
    return np.column_stack([np.clip(lon, -np.pi + 1e-6, np.pi - 1e-6), lat]), bool((np.abs(lon) > np.pi - 2e-3).any())

def rbox(x, y, w, h, text, fc="white", ec=INK, fs=7.0, tc=INK, lw=1.1, bold=False, rs=0.012, z=3):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.002,rounding_size={rs}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    bg.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=z + 1)

def arrow(x0, y0, x1, y1, color=INK, lw=1.4, rad=0.0, ls="-"):
    bg.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))

def elabel(x, y, s, fs=6.2, color=GREY, ha="center", va="center"):
    bg.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, style="italic", zorder=5)

# ============================ figure ============================
fig = plt.figure(figsize=(7.4, 9.6))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.5, 0.982, "NeST architecture: an exactly-normalized SE(3) density by construction",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
bg.text(0.5, 0.963, "ViT-B/16 patch tokens condition a tree-structured softmax over the SO(3)$\\times$R$^3$ partition; "
        "every node is a log-softmax, so the product is a normalized density",
        ha="center", va="center", fontsize=7.4, color=ENC)

# ---------- BAND A: SE(3) front-end + encoder ----------
yA = 0.885
# detection -> t_hat -> crop
rbox(0.035, yA - 0.018, 0.145, 0.052, "detection bbox\n$+\\ K,\\ $diameter $d$", ENC_F, ENC, fs=6.6)
arrow(0.182, yA + 0.008, 0.212, yA + 0.008, ENC)
rbox(0.212, yA - 0.018, 0.085, 0.052, "$\\hat t$ back-\nprojection", "white", ENC, fs=6.6)
elabel(0.254, yA - 0.03, "$\\hat t,\\ A$ (box)")
arrow(0.297, yA + 0.008, 0.328, yA + 0.008, ENC)
axcrop = fig.add_axes([0.330, yA - 0.028, 0.072, 0.072]); axcrop.imshow(crop.astype(np.uint8)); axcrop.axis("off")
axcrop.set_title("crop 224$^2$", fontsize=6.4, color=INK, pad=1.5)
# patch grid glyph
arrow(0.404, yA + 0.008, 0.430, yA + 0.008, ENC)
axpg = fig.add_axes([0.432, yA - 0.024, 0.060, 0.060]); axpg.imshow(crop.astype(np.uint8)); axpg.axis("off")
for i in range(1, G):
    axpg.axhline(i * crop.shape[0] / G, color="w", lw=0.3, alpha=0.6)
    axpg.axvline(i * crop.shape[1] / G, color="w", lw=0.3, alpha=0.6)
axpg.set_title("patch-embed\n14$\\times$14", fontsize=6.0, color=INK, pad=1.0)
arrow(0.494, yA + 0.008, 0.520, yA + 0.008, ENC)
rbox(0.520, yA - 0.022, 0.175, 0.060,
     "ViT-B/16  (augreg2)\n12 transformer blocks\nfine-tuned  (lr $10^{-5}$)", ENC_F, ENC, fs=6.8, bold=False)
arrow(0.695, yA + 0.008, 0.724, yA + 0.008, ENC)
rbox(0.724, yA - 0.016, 0.150, 0.048, "patch tokens\n$X\\in\\mathbb{R}^{196\\times768}$", "white", ENC, fs=7.0, bold=True)

# K/V bus: a clear channel above the heads; taps enter each scorer from its left side (Q enters from the top)
ybus = 0.822
bg.plot([0.225, 0.799], [ybus, ybus], color=ENC, lw=1.6, zorder=1)
bg.plot([0.799, 0.799], [ybus, yA - 0.016], color=ENC, lw=1.6, zorder=1)
bg.text(0.225, ybus + 0.008, "$K,V$ bus: patch tokens $X$ condition every scorer", ha="left", va="bottom",
        fontsize=6.3, color=ENC, style="italic")
bg.plot([0.225, 0.225], [ybus, 0.722], color=ENC, lw=1.3, zorder=1)   # left tap -> l0_attn K/V
arrow(0.225, 0.722, 0.235, 0.722, ENC, lw=1.3)
bg.plot([0.555, 0.555], [ybus, 0.569], color=ENC, lw=1.3, zorder=1)   # right tap -> child_attn K/V
arrow(0.555, 0.569, 0.590, 0.569, ENC, lw=1.3)

# ---------- BAND B: two heads ----------
# ===== LEFT column: rotation root =====
XL = 0.33
rbox(XL - 0.085, 0.760, 0.170, 0.034, "72 learned queries   Emb$(72{\\times}768)$", "white", SCO, fs=6.8)
arrow(XL, 0.760, XL, 0.744, SCO)
rbox(XL - 0.095, 0.700, 0.190, 0.044, "l0_attn : Cross-Attn $\\times2$\n(same block, separate weights)",
     SCO_F, SCO, fs=6.6)
bg.text(XL, 0.695, "", ha="center")
arrow(XL, 0.700, XL, 0.686, SM)
rbox(XL - 0.082, 0.648, 0.164, 0.034, "72 logits $\\to$ log-softmax", SM_F, SM, fs=6.8)
bg.text(XL, 0.636, "$p_0(c_0\\mid x)$  : 72-way rotation root", ha="center", va="top", fontsize=6.8,
        color=SM, fontweight="bold")
# real attention overlay (chosen root query)
axra = fig.add_axes([XL - 0.145, 0.520, 0.095, 0.095]); axra.imshow(crop.astype(np.uint8)); axra.axis("off")
axra.imshow(np.kron(root_att, np.ones((16, 16))), cmap="inferno", alpha=0.55,
            extent=[0, crop.shape[1], crop.shape[0], 0])
axra.set_title("root query\nattention (real)", fontsize=6.0, color=INK, pad=1.0)
# real 72-way softmax on a HEALPix Mollweide (nside=1 base tiles)
axrm = fig.add_axes([XL - 0.030, 0.512, 0.150, 0.105], projection="mollweide")
axrm.set_xticks([]); axrm.set_yticks([]); axrm.grid(True, alpha=0.12, lw=0.3)
nr0 = so3.n_roll(0); pix0 = np.arange(72) // nr0; roll0 = np.arange(72) % nr0
pmax = p0.max()
for pp in range(12):
    sel = pix0 == pp; pm = p0[sel].sum(); rb = int(roll0[sel][np.argmax(p0[sel])])
    poly, seam = _pix_poly(1, pp, step=10)
    if seam: continue
    axrm.add_patch(Polygon(poly, closed=True, fc=HSV(_hue(rb, nr0)), ec="#555", lw=0.4,
                           alpha=float(np.clip(0.2 + 0.75 * pm / pmax, 0.12, 0.95)), zorder=2))
    if pp == root_rc // nr0:
        axrm.add_patch(Polygon(poly, closed=True, fc="none", ec=GOLD_L, lw=1.8, zorder=5))
for sp in axrm.spines.values(): sp.set_edgecolor(SM); sp.set_linewidth(1.4)
bg.text(XL + 0.045, 0.512, "HEALPix S$^2$ tiles,\nhue = Hopf roll", ha="center", va="top", fontsize=5.8, color=GREY)

# ===== RIGHT column: child scorer (all levels) =====
XR = 0.70
by = 0.763
rbox(XR - 0.135, by, 0.270, 0.030, "child offsets $[\\,$rot$(3),$ trans$(3)\\,]$   $\\oplus$   "
     "parent code $[\\,R(9),\\ \\hat g(3)\\,]$", "white", OR, fs=6.3)
arrow(XR, by, XR, by - 0.012, OR)
rbox(XR - 0.115, by - 0.044, 0.230, 0.028, "Fourier enc.  $\\pi\\cdot2^{k}$  ($n_{freq}{=}3$)", OR_F, OR, fs=6.4)
arrow(XR, by - 0.044, XR, by - 0.056, OR)
rbox(XR - 0.135, by - 0.094, 0.270, 0.034, "child_proj : Lin$(126{\\to}768)$, GELU, Lin$(768{\\to}768)$"
     "  $+$ level emb", OR_F, OR, fs=6.1)
arrow(XR, by - 0.094, XR, by - 0.106, OR)
rbox(XR - 0.085, by - 0.138, 0.170, 0.028, "$K$ child queries  (64)", "white", SCO, fs=6.6, bold=True)
arrow(XR, by - 0.138, XR, 0.618, SCO)

# detailed scorer glyph
sx, sy, sw, shh = XR - 0.11, 0.520, 0.22, 0.098
bg.add_patch(FancyBboxPatch((sx, sy), sw, shh, boxstyle="round,pad=0.003,rounding_size=0.012",
                            fc=SCO_F, ec=SCO, lw=1.4, zorder=3))
bg.text(sx + sw / 2, sy + shh - 0.008, "child_attn : Cross-Attn $\\times2$", ha="center", va="top",
        fontsize=6.6, color=SCO, fontweight="bold", zorder=5)
# internal sublayers (one layer shown, "x2")
iy = sy + shh - 0.030; ih = 0.014
for lab, dy in [("LN $\\to$ MHA$(Q;K,V)$  $\\oplus$", 0.0), ("LN $\\to$ MLP$(4\\times)$  $\\oplus$", -0.020)]:
    rbox(sx + 0.018, iy + dy - ih / 2, sw - 0.036, ih, lab, "white", SCO, fs=5.8, rs=0.006, z=4)
rbox(sx + 0.040, sy + 0.010, sw - 0.080, ih, "LN $\\to$ Linear (zero-init)", "white", GOLD_L, fs=5.8, rs=0.006, z=4)
bg.text(sx + sw - 0.004, sy + 0.028, "$\\times2$", ha="right", va="center", fontsize=6.2, color=SCO, fontweight="bold")
# K/V tap label into child scorer
elabel(0.558, 0.576, "$K,V$", fs=6.0, color=ENC, ha="left", va="bottom")

arrow(XR, 0.520, XR, 0.506, SM)
rbox(XR - 0.082, 0.470, 0.164, 0.032, "64 logits $\\to$ log-softmax", SM_F, SM, fs=6.6)
bg.text(XR + 0.090, 0.486, "$p(c_\\ell\\mid \\mathrm{path},x)$", ha="left", va="center", fontsize=6.8,
        color=SM, fontweight="bold")
# real 64-way softmax heatmap (8 rot x 8 oct)
axh = fig.add_axes([XR - 0.048, 0.356, 0.092, 0.082])
axh.imshow(J, cmap="magma", aspect="equal", origin="upper")
axh.add_patch(Rectangle((oj - 0.5, ci - 0.5), 1, 1, fill=False, ec=GOLD, lw=2.0))
axh.set_xticks([0, 7]); axh.set_yticks([0, 7]); axh.tick_params(labelsize=5.0, length=2)
axh.set_xlabel("trans octant", fontsize=5.8, labelpad=1); axh.set_ylabel("rot child", fontsize=5.8, labelpad=1)
axh.set_title("64-way child softmax\n$=8\\,$rot$\\times8\\,$oct (real)", fontsize=6.0, color=INK, pad=1.5)
# chosen-child attention overlay
axca = fig.add_axes([XR + 0.075, 0.352, 0.085, 0.085]); axca.imshow(crop.astype(np.uint8)); axca.axis("off")
axca.imshow(np.kron(child_att, np.ones((16, 16))), cmap="inferno", alpha=0.55,
            extent=[0, crop.shape[1], crop.shape[0], 0])
axca.set_title("chosen child\nattention (real)", fontsize=5.8, color=INK, pad=1.0)
# recurrence loop
arrow(XR + 0.135, 0.360, XR + 0.175, 0.45, OR, lw=1.3, rad=-0.4)
rbox(XR + 0.120, 0.600, 0.110, 0.030, "parent code\n$(R_{\\mathrm{sel}},\\,g_{\\mathrm{sel}})$", OR_F, OR, fs=5.9)
arrow(XR + 0.175, 0.47, XR + 0.175, 0.600, OR, lw=1.3)
arrow(XR + 0.175, 0.630, XR + 0.135, by + 0.010, OR, lw=1.3, rad=0.3)
bg.text(XR + 0.232, 0.52, "autoregressive\nover depth\n$\\ell=1..5$", ha="center", va="center",
        fontsize=6.0, color=OR, rotation=90, fontweight="bold")

# column captions (compact, clear of the bus)
bg.text(XL, 0.797, "ROOT head  (rotation)", ha="center", va="bottom", fontsize=6.8, color=SCO, fontweight="bold")
bg.text(XR, 0.797, "CHILD head  (shared, $\\ell{=}1..5$)", ha="center", va="bottom", fontsize=6.8,
        color=SCO, fontweight="bold")

# ---------- BAND C: tree-softmax readout ----------
bg.plot([0.5, 0.5], [0.44, 0.352], color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=1)
arrow(XL, 0.628, 0.47, 0.352, SM, lw=1.2, rad=0.12)
arrow(XR, 0.452, 0.53, 0.352, SM, lw=1.2, rad=-0.12)
rbox(0.150, 0.300, 0.700, 0.050,
     "tree-softmax  (path log-likelihood):   $\\log p(R,t)=\\log p_0(c_0\\mid x)+\\sum_{\\ell}\\log p(c_\\ell\\mid\\cdot)$",
     "white", SM, fs=7.2, bold=True)
bg.text(0.5, 0.287, "$\\Rightarrow\\ p(R,t)=P(\\text{leaf cell})\\cdot \\dfrac{N}{V_{SO(3)}\\,V_{\\text{box}}},"
        "\\quad N=72\\cdot8^{L_{rot}}\\cdot8^{L_{trans}}\\approx 7.7\\times10^{10}$",
        ha="center", va="top", fontsize=7.0, color=INK)
arrow(0.5, 0.282, 0.5, 0.258, SM)
rbox(0.175, 0.208, 0.330, 0.044, "leaf cell $\\to$  $R=$ cell centre (SO(3)),\n$t=\\hat t + A\\,g$",
     "white", INK, fs=6.8)
arrow(0.505, 0.230, 0.548, 0.230, INK)
# MAP pose 6D box on the crop
mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]]) / 1000.0
szb = np.array([mi["size_x"], mi["size_y"], mi["size_z"]]) / 1000.0
corners = np.array([[mn[0]+a*szb[0], mn[1]+b*szb[1], mn[2]+c*szb[2]] for a in (0,1) for b in (0,1) for c in (0,1)])
EDG = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
uv = (K @ (R_map @ corners.T + t_map[:, None])); uv = (uv[:2] / uv[2]).T
c0 = uv.min(0); c1 = uv.max(0); ctr = (c0 + c1) / 2; half = (c1 - c0).max() / 2 + 0.5 * (c1 - c0).max()
x0, y0 = max(0, int(ctr[0]-half)), max(0, int(ctr[1]-half))
x1, y1 = min(img.shape[1], int(ctr[0]+half)), min(img.shape[0], int(ctr[1]+half))
axbox = fig.add_axes([0.552, 0.170, 0.115, 0.090]); axbox.imshow(img[y0:y1, x0:x1]); axbox.axis("off")
for e in EDG:
    axbox.plot(uv[list(e), 0] - x0, uv[list(e), 1] - y0, color=RED, lw=1.4)
bg.text(0.610, 0.162, "MAP pose $(R,t)\\to$ 3D box", ha="center", va="top", fontsize=6.4, color=INK)

# gold callout
bg.add_patch(FancyBboxPatch((0.135, 0.095), 0.730, 0.040, boxstyle="round,pad=0.004,rounding_size=0.01",
                            fc="#FFF8DC", ec=GOLD_L, lw=1.3, zorder=3))
bg.text(0.5, 0.115, "every node is a log-softmax  $\\Rightarrow$  a normalized SE(3) density by construction;   "
        "output Linear zero-init  $\\Rightarrow$  starts exactly at the uniform floor $-\\ln(\\pi^2 V_{\\text{box}})$",
        ha="center", va="center", fontsize=6.8, color=GOLD_L, fontweight="bold", zorder=5)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
