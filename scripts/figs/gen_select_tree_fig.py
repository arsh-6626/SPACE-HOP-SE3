"""NeST decode as an elaborate MULTI-BRANCH tree: selection (a real beam search -- branch, score, prune) and
refinement (each kept cell subdivides, the HEALPix tile shrinks x4 and the octree octant x8 per level). Nodes are
the beam's kept SE(3) cells; each is a mini rotation-Mollweide (HEALPix tile on the Hopf base sphere, hue = roll)
plus a tiny translation octree cube (chosen octant). Edges are the child conditionals p(child|parent) (width),
gold = the beam-MAP path. Real forward pass: YCB-V obj 3, scene 51/1629; beam B=3.

Run: cd /home/kavinder/SPACE-HOP && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -I scratchpad/gen_select_tree_fig.py
Out: logs/figs/nest_select_tree.pdf/.png
"""
import os, sys, json, numpy as np, torch, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
sys.path.insert(0, "src")
import se3_tless_data as td, se3_decode as dec, se3_bop_eval as be
import se3_tree as se3, so3_nested as so3
from se3_head import TRANS_OCT_OFF
from PIL import Image

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300})
BOP = "/media/kavinder/hdd2/liepose_data/bop_datasets/ycbv"; SPLIT = "test"; DEV = "cuda"; OBJ = 3
IMN_M = np.array([0.485, 0.456, 0.406], np.float32); IMN_S = np.array([0.229, 0.224, 0.225], np.float32)
INK = "#232323"; BLUE = "#3775BA"; GOLD = "#FFD700"; GOLD_L = "#B8860B"; GREY = "#9aa0a8"; RED = "#B64342"
SM = "#2E8B57"; BLUE2 = "#5B8AC6"
OUTBASE = "/home/kavinder/SPACE-HOP/logs/figs/nest_select_tree"
HSV = mpl.colormaps["hsv"]; B = 3; L = 4

# ---------------- real forward pass + beam tree ----------------
enc, head, L_rot, L_trans, crop_pad, out_px = be._load_ckpt(f"checkpoints/se3_ycbv_aug_obj{OBJ}/best.pth", DEV)
d_m = td.diameters_m(BOP, [OBJ])[OBJ]; mi = td.load_models_info(BOP)[OBJ]
det = next(x for x in json.load(open("data/detections/gdrnppdet_ycbv-test.json"))
           if x["category_id"] == OBJ and x["scene_id"] == 51 and x["image_id"] == 1629)
scene, im, bbox = det["scene_id"], det["image_id"], np.array(det["bbox"], np.float64); sdir = f"{BOP}/{SPLIT}/{scene:06d}"
K = np.array(json.load(open(f"{sdir}/scene_camera.json"))[str(im)]["cam_K"], np.float64).reshape(3, 3)
img = np.asarray(Image.open(td.rgb_path(BOP, SPLIT, scene, im)).convert("RGB"), np.uint8)
t_hat = be._t_hat_from_bbox(bbox, K, d_m); side = d_m / t_hat[2] * K[0, 0] * crop_pad; cu, cv = td.project(K, t_hat)
crop = td.crop_resize(img, cu, cv, side, out_px); x = (crop.astype(np.float32) / 255.0 - IMN_M) / IMN_S
patches = enc(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV)).float()

@torch.no_grad()
def beam_tree():
    lp0 = head.logp0(patches)[0].cpu().numpy(); order = np.argsort(-lp0)[:B]
    cur = [dict(rot=int(c), trans=0, logp=float(lp0[c]), parent=-1, edgep=float(np.exp(lp0[c])), to=None) for c in order]
    tree = [cur]
    for l in range(1, L + 1):
        cand = []
        for bi, nd in enumerate(cur):
            chf, par_flat, roff = dec._rot_child_geom(nd['rot'], l - 1); tch = dec._trans_children(nd['trans'], l - 1)
            tctr = se3.trans_cell_center(np.array([nd['trans']]), l - 1)[0].astype(np.float32)
            off6 = np.concatenate([np.repeat(roff[:, None, :], 8, 1),
                                   np.broadcast_to(TRANS_OCT_OFF[None], (8, 8, 3))], -1).reshape(64, 6).astype(np.float32)
            par12 = np.concatenate([par_flat, tctr]).astype(np.float32)
            clp = head.child_logp(patches, torch.as_tensor(off6, device=DEV)[None],
                                  torch.as_tensor(par12, device=DEV)[None], l - 1)[0].cpu().numpy()
            for k in range(64):
                cand.append((nd['logp'] + clp[k], int(chf[k // 8]), int(tch[k % 8]), bi, float(np.exp(clp[k])), k % 8))
        cand.sort(key=lambda c: -c[0]); cand = cand[:B]
        cur = [dict(rot=r, trans=t, logp=cu_, parent=p, edgep=ep, to=to) for cu_, r, t, p, ep, to in cand]
        tree.append(cur)
    return tree
tree = beam_tree()
path = {L: 0}                                                      # beam-MAP leaf lineage
for l in range(L, 0, -1):
    path[l - 1] = tree[l][path[l]]['parent']
for l, lvl in enumerate(tree):
    print(f"lvl{l}: " + " | ".join(f"rot{n['rot']} par{n['parent']} ep{n['edgep']:.2f}" for n in lvl))

# ---------------- helpers ----------------
def _hue(roll, nr): return (0.5 + (roll + 0.5) / nr) % 1.0
def _pix_poly(nside, pix, step=10):
    v = hp.boundaries(nside, int(pix), step=step, nest=True).T
    th, ph = hp.vec2ang(v); lon = (ph + np.pi) % (2 * np.pi) - np.pi; lat = np.pi / 2 - th
    m = np.arctan2(np.sin(lon).mean(), np.cos(lon).mean()); lon = m + ((lon - m + np.pi) % (2 * np.pi) - np.pi)
    return np.column_stack([np.clip(lon, -np.pi + 1e-6, np.pi - 1e-6), lat]), bool((np.abs(lon) > np.pi - 2e-3).any())
def iso(p): return np.array([(p[0] - p[1]) * 0.86, p[2] * 0.95 + (p[0] + p[1]) * 0.5])

# ---------------- figure ----------------
fig = plt.figure(figsize=(8.2, 8.8))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.text(0.5, 0.982, "NeST decode as a multi-branch tree: selection (beam) and refinement",
        ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
bg.text(0.5, 0.960, "each node is a kept SE(3) cell (rotation HEALPix tile $\\times$ translation octree octant); "
        "edges are the child conditionals $p(c\\,|\\,\\mathrm{parent})$; gold = the beam-MAP path",
        ha="center", va="center", fontsize=8.0, color=BLUE)

rowy = np.linspace(0.845, 0.235, L + 1)
# ---- level labels (left): refinement bookkeeping ----
for l in range(L + 1):
    ns = so3.nside_of(l); nr = so3.n_roll(l); Ncells = 72 * 8 ** l
    bg.text(0.018, rowy[l] + 0.028, f"$\\ell{{=}}{l}$", ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    bg.text(0.018, rowy[l] - 0.004, f"nside {ns}\n$n_{{roll}}{{=}}{nr}$\n$N{{=}}72{{\\cdot}}8^{l}$", ha="left", va="top",
            fontsize=6.2, color=GREY)
bg.text(0.018, (rowy[0] + rowy[1]) / 2, "refine\n$\\times8$", ha="left", va="center", fontsize=6.8, color=SM,
        fontweight="bold", rotation=0)

# ---- node positions (grouped under parents to limit edge crossings) ----
POS = {}
for l, lvl in enumerate(tree):
    m = len(lvl)
    if l == 0:
        key = sorted(range(m), key=lambda j: -lvl[j]['logp'])
    else:
        key = sorted(range(m), key=lambda j: (POS[(l - 1, lvl[j]['parent'])][0], -lvl[j]['logp']))
    span = 0.0 if m == 1 else min(0.54, 0.17 * m)
    xs_sorted = np.linspace(0.55 - span / 2, 0.55 + span / 2, m)
    xs = [0.0] * m
    for rank, j in enumerate(key): xs[j] = xs_sorted[rank]
    for j in range(m): POS[(l, j)] = (xs[j], rowy[l])

# ---- edges (draw first, behind nodes) ----
for l in range(1, L + 1):
    for j, nd in enumerate(tree[l]):
        x1, y1 = POS[(l, j)]; x0, y0 = POS[(l - 1, nd['parent'])]
        on_map = (j == path[l] and nd['parent'] == path[l - 1])
        col = GOLD_L if on_map else BLUE2
        bg.annotate("", xy=(x1, y1 + 0.052), xytext=(x0, y0 - 0.050), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8 + 3.4 * nd['edgep'],
                                    alpha=0.95 if on_map else 0.5, connectionstyle="arc3,rad=0.0"))
        bg.text((x0 + x1) / 2 + 0.006, (y0 + y1) / 2, f"{nd['edgep']:.2f}", ha="left", va="center",
                fontsize=6.0, color=col, fontweight="bold" if on_map else "normal")
    # pruned-branch hint under each parent + beam note
    for j in range(len(tree[l - 1])):
        px, py = POS[(l - 1, j)]
        for dx in (-0.055, -0.03, 0.03, 0.055):
            bg.plot([px, px + dx], [py - 0.050, py - 0.072], color=GREY, lw=0.5, alpha=0.3, zorder=0)
bg.text(0.86, (rowy[0] + rowy[1]) / 2, "64-way split,\nkeep top-$%d$\n(beam)" % B, ha="center", va="center",
        fontsize=7.0, color=INK, style="italic")

# ---- node glyphs: rotation HEALPix tile (Mollweide) + translation octree cube ----
def node_glyph(l, j):
    nd = tree[l][j]; xc, yc = POS[(l, j)]; on_map = (j == path.get(l, -99))
    nside = so3.nside_of(l); nr = so3.n_roll(l); pix = nd['rot'] // nr; roll = nd['rot'] % nr
    # mini Mollweide
    axm = fig.add_axes([xc - 0.052, yc - 0.030, 0.082, 0.056], projection="mollweide")
    axm.set_xticks([]); axm.set_yticks([]); axm.grid(True, alpha=0.1, lw=0.25)
    for pp in range(12):
        poly, seam = _pix_poly(1, pp, step=8)
        if not seam: axm.add_patch(Polygon(poly, closed=True, fc="#eceef1", ec="#dfe2e6", lw=0.3, zorder=0))
    poly, seam = _pix_poly(nside, pix, step=8)
    if not seam:
        axm.add_patch(Polygon(poly, closed=True, fc=HSV(_hue(roll, nr)), ec="#4a4a4a", lw=0.5, alpha=0.95, zorder=2))
    for sp in axm.spines.values():
        sp.set_edgecolor(GOLD_L if on_map else BLUE2); sp.set_linewidth(2.0 if on_map else 1.1)
    # tiny translation octree cube
    axc = fig.add_axes([xc + 0.030, yc - 0.020, 0.036, 0.040]); axc.axis("off")
    axc.set_xlim(-1.9, 1.9); axc.set_ylim(-1.7, 1.9); axc.set_aspect("equal")
    hs = 0.5
    for s in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]:
        pass
    E = [((-1,-1,-1),(1,-1,-1)),((1,-1,-1),(1,1,-1)),((1,1,-1),(-1,1,-1)),((-1,1,-1),(-1,-1,-1)),
         ((-1,-1,1),(1,-1,1)),((1,-1,1),(1,1,1)),((1,1,1),(-1,1,1)),((-1,1,1),(-1,-1,1)),
         ((-1,-1,-1),(-1,-1,1)),((1,-1,-1),(1,-1,1)),((1,1,-1),(1,1,1)),((-1,1,-1),(-1,1,1))]
    for a, b in E:
        pa, pb = iso(np.array(a, float)), iso(np.array(b, float))
        axc.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#aab0b8", lw=0.6, zorder=1)
    if nd['to'] is not None:                                      # highlight chosen octant sub-cube
        c = TRANS_OCT_OFF[nd['to']].astype(float)
        def PT(dx, dy, dz): return iso(c + np.array([dx, dy, dz]) * hs)
        for fv in ([PT(-1,-1,1),PT(1,-1,1),PT(1,1,1),PT(-1,1,1)],
                   [PT(1,-1,-1),PT(1,1,-1),PT(1,1,1),PT(1,-1,1)],
                   [PT(-1,1,-1),PT(1,1,-1),PT(1,1,1),PT(-1,1,1)]):
            axc.add_patch(Polygon(fv, closed=True, fc=GOLD if on_map else BLUE2, ec=GOLD_L if on_map else "#4a6fa5",
                                  lw=0.5, alpha=0.9, zorder=3))
    else:
        axc.scatter([0], [0], s=10, c=GREY, zorder=3)
for l in range(L + 1):
    for j in range(len(tree[l])):
        node_glyph(l, j)
# root caption
bg.text(0.55, rowy[0] + 0.050, "root: 72-way rotation ($g{=}0$)", ha="center", va="bottom", fontsize=7.0, color=SM)

# ---- collapse to leaf + MAP box ----
xmap, ymap = POS[(L, path[L])]
bg.annotate("", xy=(xmap, 0.185), xytext=(xmap, rowy[L] - 0.050), zorder=1,
            arrowprops=dict(arrowstyle="-|>", color=GOLD_L, lw=2.4))
bg.text(xmap + 0.012, 0.205, "$\\vdots$ refine to $\\ell{=}5$ leaf", ha="left", va="center", fontsize=7.2,
        color=GOLD_L, fontweight="bold")
# MAP pose box
rc_fin, tc_fin, _ = dec.beam_map(head, patches[0], L_rot, L_trans, beam=64)
R_map = so3.cell_index_to_matrix(np.array([rc_fin]), L_rot)[0]
t_map = t_hat + td.box_A(d_m, t_hat) @ se3.trans_cell_center(np.array([tc_fin]), L_trans)[0]
mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]]) / 1000.0; szb = np.array([mi["size_x"], mi["size_y"], mi["size_z"]]) / 1000.0
corners = np.array([[mn[0]+a*szb[0], mn[1]+b*szb[1], mn[2]+c*szb[2]] for a in (0,1) for b in (0,1) for c in (0,1)])
EDG = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
uv = (K @ (R_map @ corners.T + t_map[:, None])); uv = (uv[:2] / uv[2]).T
c0 = uv.min(0); c1 = uv.max(0); ctr = (c0 + c1) / 2; half = (c1 - c0).max() / 2 + 0.5 * (c1 - c0).max()
x0, y0 = max(0, int(ctr[0]-half)), max(0, int(ctr[1]-half)); x1, y1 = min(img.shape[1], int(ctr[0]+half)), min(img.shape[0], int(ctr[1]+half))
axb = fig.add_axes([xmap - 0.072, 0.095, 0.144, 0.098]); axb.imshow(img[y0:y1, x0:x1]); axb.axis("off")
for e in EDG:
    axb.plot(uv[list(e), 0] - x0, uv[list(e), 1] - y0, color=RED, lw=1.4)
bg.text(xmap, 0.088, "beam-MAP leaf $\\to$ 6D pose $(R,t)$", ha="center", va="top", fontsize=7.2, color=INK, fontweight="bold")

# legend (all in one box, left)
bg.add_patch(FancyBboxPatch((0.030, 0.028), 0.345, 0.120, boxstyle="round,pad=0.004,rounding_size=0.008",
                            fc="white", ec=GREY, lw=1.0, zorder=3))
bg.text(0.044, 0.132, "node = HEALPix tile (rotation, hue=roll)", ha="left", va="center", fontsize=6.3, color=INK, fontweight="bold")
bg.text(0.044, 0.116, "         + octree octant (translation)", ha="left", va="center", fontsize=6.3, color=INK, fontweight="bold")
bg.text(0.044, 0.100, "cell refines $\\times4$ (tile), $\\times8$ (octant) / level", ha="left", va="center", fontsize=6.2, color=INK)
bg.plot([0.048, 0.076], [0.080, 0.080], color=GOLD_L, lw=2.6); bg.text(0.082, 0.080, "beam-MAP path", ha="left", va="center", fontsize=6.2, color=INK)
bg.plot([0.048, 0.076], [0.062, 0.062], color=BLUE2, lw=1.6); bg.text(0.082, 0.062, "other kept beam branches", ha="left", va="center", fontsize=6.2, color=INK)
bg.plot([0.050, 0.074], [0.044, 0.038], color=GREY, lw=0.6, alpha=0.5); bg.plot([0.050, 0.074], [0.044, 0.050], color=GREY, lw=0.6, alpha=0.5)
bg.text(0.082, 0.044, "faint stubs = 61 pruned children / node", ha="left", va="center", fontsize=6.2, color=INK)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUTBASE}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print(f"saved {OUTBASE}.pdf/.png")
