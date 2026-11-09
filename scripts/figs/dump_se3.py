"""Run ONE real YCB-V SE(3) forward pass (device 1) and dump everything the figure needs to an npz,
so the globe-tree figure can be iterated without re-touching the GPU."""
import os, sys, json, numpy as np, torch, healpy as hp
sys.path.insert(0, "src")
import se3_tless_data as td, se3_decode as dec, se3_head as sh
import se3_tree as se3, so3_nested as so3, se3_bop_eval as be
from se3_head import TRANS_OCT_OFF
from PIL import Image

BOP = "/media/kavinder/hdd2/liepose_data/bop_datasets/ycbv"; SPLIT = "test"
OBJ = 3; CKPT = f"checkpoints/se3_ycbv_aug_obj{OBJ}/best.pth"; DEV = "cuda"
IMN_M = np.array([0.485, 0.456, 0.406], np.float32); IMN_S = np.array([0.229, 0.224, 0.225], np.float32)
OUT = "logs/figs/_cache/se3_dump.npz"

enc, head, L_rot, L_trans, crop_pad, out_px = be._load_ckpt(CKPT, DEV)
d_m = td.diameters_m(BOP, [OBJ])[OBJ]; mi = td.load_models_info(BOP)[OBJ]
dets = [x for x in json.load(open("data/detections/gdrnppdet_ycbv-test.json")) if x["category_id"] == OBJ]
SEL_SCENE, SEL_IM = 51, 1629
det = next((x for x in dets if x["scene_id"] == SEL_SCENE and x["image_id"] == SEL_IM), None) \
      or max(dets, key=lambda x: x["score"])
scene, im, bbox = det["scene_id"], det["image_id"], np.array(det["bbox"], np.float64)
sdir = f"{BOP}/{SPLIT}/{scene:06d}"
K = np.array(json.load(open(f"{sdir}/scene_camera.json"))[str(im)]["cam_K"], np.float64).reshape(3, 3)
img = np.asarray(Image.open(td.rgb_path(BOP, SPLIT, scene, im)).convert("RGB"), np.uint8)
t_hat = be._t_hat_from_bbox(bbox, K, d_m); side_px = d_m / t_hat[2] * K[0, 0] * crop_pad
cu, cv = td.project(K, t_hat); crop = td.crop_resize(img, cu, cv, side_px, out_px)
x = (crop.astype(np.float32) / 255.0 - IMN_M) / IMN_S
patches = enc(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(DEV)).float()[0]
A = td.box_A(d_m, t_hat)

rc_fin, tc_fin, lp_fin = dec.beam_map(head, patches, L_rot, L_trans, beam=64)
R_map = so3.cell_index_to_matrix(np.array([rc_fin]), L_rot)[0]
g_leaf = se3.trans_cell_center(np.array([tc_fin]), L_trans)[0]
t_map = t_hat + A @ g_leaf
gt = json.load(open(f"{sdir}/scene_gt.json"))[str(im)]
gi = next(k for k, g in enumerate(gt) if g["obj_id"] == OBJ)
R_gt = np.array(gt[gi]["cam_R_m2c"], np.float64).reshape(3, 3)
t_gt = np.array(gt[gi]["cam_t_m2c"], np.float64) / 1000.0
ang = np.degrees(np.arccos(np.clip((np.trace(R_map.T @ R_gt) - 1) / 2, -1, 1)))
terr = np.linalg.norm(t_map - t_gt) * 1000

def rot_ancestor(cell_fin, Lf, L):
    nrf = so3.n_roll(Lf); pix = cell_fin // nrf; roll = cell_fin % nrf; k = Lf - L
    return (pix // (4 ** k)) * so3.n_roll(L) + (roll // (2 ** k))
def trans_ancestor(cell_fin, Lf, L):
    ix, iy, iz = (int(v[0]) for v in se3._trans_flat_to_coords(np.array([cell_fin]), Lf)); k = Lf - L
    return int(se3._trans_coords_to_flat(ix >> k, iy >> k, iz >> k, L))

@torch.no_grad()
def level_branch(l):
    rot_par = int(rot_ancestor(rc_fin, L_rot, l - 1)); trans_par = int(trans_ancestor(tc_fin, L_trans, l - 1))
    chf, par_flat, roff = dec._rot_child_geom(rot_par, l - 1)
    tch = dec._trans_children(trans_par, l - 1)
    tctr = se3.trans_cell_center(np.array([trans_par]), l - 1)[0].astype(np.float32)
    roff_rep = np.repeat(roff[:, None, :], 8, axis=1); toff_rep = np.broadcast_to(TRANS_OCT_OFF[None], (8, 8, 3))
    off6 = np.concatenate([roff_rep, toff_rep], -1).reshape(64, 6).astype(np.float32)
    par12 = np.concatenate([par_flat, tctr]).astype(np.float32)
    clp = head.child_logp(patches[None], torch.as_tensor(off6, device=DEV)[None],
                          torch.as_tensor(par12, device=DEV)[None], l - 1)[0].cpu().numpy()
    J = clp.reshape(8, 8); pj = np.exp(J - J.max()); pj /= pj.sum()
    rot_p = pj.sum(1); trans_p = pj.sum(0)
    rc_sel = int(np.where(chf == rot_ancestor(rc_fin, L_rot, l))[0][0])
    to_sel = int(np.where(tch == trans_ancestor(tc_fin, L_trans, l))[0][0])
    return dict(chf=chf.astype(np.int64), rot_p=rot_p, trans_p=trans_p, rc=rc_sel, to=to_sel,
                joint_p=float(pj[rc_sel, to_sel]))
levels = [level_branch(l) for l in range(1, L_rot + 1)]
lp0 = head.logp0(patches[None])[0].cpu().numpy()
p0 = np.exp(lp0 - lp0.max()); p0 /= p0.sum()
root_rc = int(rot_ancestor(rc_fin, L_rot, 0))

# ---- leaf box projection ----
mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]]) / 1000.0
szb = np.array([mi["size_x"], mi["size_y"], mi["size_z"]]) / 1000.0
corners = np.array([[mn[0]+a*szb[0], mn[1]+b*szb[1], mn[2]+c*szb[2]] for a in (0,1) for b in (0,1) for c in (0,1)])
def proj(R, t):
    uv = K @ (R @ corners.T + t[:, None]); return (uv[:2] / uv[2]).T
uvp = proj(R_map, t_map); uvg = proj(R_gt, t_gt)
allu = np.concatenate([uvp, uvg]); c0 = allu.min(0); c1 = allu.max(0)
pad = 0.55 * (c1 - c0).max(); ctr = (c0 + c1) / 2; half = (c1 - c0).max() / 2 + pad
x0, y0 = max(0, int(ctr[0]-half)), max(0, int(ctr[1]-half))
x1, y1 = min(img.shape[1], int(ctr[0]+half)), min(img.shape[0], int(ctr[1]+half))
leaf_img = img[y0:y1, x0:x1].copy()
uvp_c = uvp - np.array([x0, y0]); uvg_c = uvg - np.array([x0, y0])

np.savez(OUT,
         crop=crop.astype(np.uint8),
         p0=p0, root_rc=root_rc,
         L_rot=L_rot, L_trans=L_trans,
         base_map=(R_map @ np.array([0, 0, 1.0])),
         **{f"chf_{i}": lv["chf"] for i, lv in enumerate(levels)},
         **{f"rotp_{i}": lv["rot_p"] for i, lv in enumerate(levels)},
         **{f"transp_{i}": lv["trans_p"] for i, lv in enumerate(levels)},
         rc=np.array([lv["rc"] for lv in levels]),
         to=np.array([lv["to"] for lv in levels]),
         jointp=np.array([lv["joint_p"] for lv in levels]),
         leaf_img=leaf_img, uvp=uvp_c, uvg=uvg_c,
         ang=ang, terr=terr, scene=scene, im=im)
print(f"obj{OBJ} scene{scene}/{im} rot {ang:.1f}deg trans {terr:.0f}mm  saved {OUT}")
print("root_rc", root_rc, "base_map", (R_map @ np.array([0,0,1.0])).round(3))
for i, lv in enumerate(levels):
    print(f"  L{i+1} rc={lv['rc']} chosen_cell={lv['chf'][lv['rc']]} rotp={lv['rot_p'][lv['rc']]:.2f} jointp={lv['joint_p']:.2f}")
