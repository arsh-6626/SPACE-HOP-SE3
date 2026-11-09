"""BOP19 AR evaluation for the NeST SE(3) per-object models, using STANDARD detections (gdrnppdet-pbrreal),
so the T-LESS / YCB-V numbers are directly comparable to the BOP leaderboard. This is the paper number for the
real-data gates (training only produces the per-object density models; AR is what the baselines report).

Pipeline (predict mode): for every detection of a requested object,
  1. t_hat from the 2D bbox + K + object diameter d (standard back-projection):
       t_hat_z = d * fx / max(bbox_w, bbox_h)          (object ~ fills the detection box)
       (cu,cv)  = bbox centre
       t_hat_x  = (cu - cx) * t_hat_z / fx ,  t_hat_y = (cv - cy) * t_hat_z / fy
  2. affine box A = box_A(d, t_hat); crop about projected t_hat with the SAME side as training
       side = d / t_hat_z * fx * crop_pad               (se3_tless_data.Tless.__getitem__)
  3. patches = enc(crop); (R_map, t_map) = se3_decode.map_pose(head, patches, L_rot, L_trans, A, t_hat, beam)
  4. emit a BOP results row:  scene_id,im_id,obj_id,score,R(9 row-major),t(3 mm),time
Then run bop_toolkit/scripts/eval_bop19_pose.py on the CSV (done by the PBS, not here).

The per-object checkpoints are se3_tless_train.py outputs: keys enc/head/args + top-level L_rot/L_trans/d_m/obj.
Detections JSON = BOP COCO-style list of {scene_id,image_id,category_id,bbox[xywh],score}.

Usage:
  python src/se3_bop_eval.py predict --dataset tless --bop <bop_root> --det <gdrnppdet.json> \
         --ckpt-dir <checkpoints> --tag _aug --objs 1,2,... --out <results.csv> [--split test_primesense] \
         [--beam 64] [--crop_pad 1.4] [--out_px 224] [--device cuda] [--limit N]
"""
import os, sys, json, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image

import se3_tless_data as td           # project, crop_resize, box_A, diameters_m, rgb_path, load_models_info
import se3_decode as dec              # map_pose
from se3_tless_train import TimmEncoder
import se3_head as sh


def _load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    enc = TimmEncoder(a["backbone"], a["img_size"]).to(device).eval()
    enc.load_state_dict(ck["enc"])
    L_rot = int(ck.get("L_rot", a.get("L_rot", 5)))
    L_trans = int(ck.get("L_trans", a.get("L_trans", 5)))
    head = sh.Se3DensityHead(embed_dim=enc.embed_dim, L_rot=L_rot, L_trans=L_trans).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in enc.parameters(): p.requires_grad_(False)
    for p in head.parameters(): p.requires_grad_(False)
    crop_pad = float(a.get("crop_pad", 1.4))
    out_px = int(a.get("img_size", 224))
    return enc, head, L_rot, L_trans, crop_pad, out_px


def _t_hat_from_bbox(bbox, K, d_m):
    """bbox (x,y,w,h) px, K (3,3), d_m diameter (m) -> coarse t_hat (m) by back-projection."""
    x, y, w, h = (float(v) for v in bbox)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    s = max(w, h, 1.0)
    tz = d_m * fx / s
    cu, cv = x + w / 2.0, y + h / 2.0
    tx = (cu - cx) * tz / fx
    ty = (cv - cy) * tz / fy
    return np.array([tx, ty, tz], np.float64)


def predict(a):
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    objs = [int(o) for o in a.objs.replace("+", ",").split(",") if o != ""]
    diam = td.diameters_m(a.bop, objs)                       # obj -> diameter (m)

    with open(a.det) as f:
        dets_all = json.load(f)
    # group detections by object
    by_obj = {o: [] for o in objs}
    for d in dets_all:
        o = int(d.get("category_id", d.get("obj_id")))
        if o in by_obj:
            by_obj[o].append(d)

    rows = []
    for o in objs:
        _run = os.path.join(a.ckpt_dir, f"se3_{a.dataset}{a.tag}_obj{o}")
        ck_path = os.path.join(_run, f"{a.ckpt}.pth")
        if not os.path.exists(ck_path):   # fall back to the other tag if the requested one is absent
            ck_path = os.path.join(_run, ("last.pth" if a.ckpt == "best" else "best.pth"))
        if not os.path.exists(ck_path):
            print(f"[warn] no ckpt for obj {o} ({ck_path}); skipping its detections", flush=True)
            continue
        enc, head, L_rot, L_trans, crop_pad, out_px = _load_ckpt(ck_path, dev)
        d_m = diam[o]
        dl = by_obj[o]
        if a.limit:
            dl = dl[: a.limit]
        print(f"[obj {o}] {len(dl)} detections, ckpt {os.path.basename(os.path.dirname(ck_path))}, L=({L_rot},{L_trans}) d={d_m:.4f}m", flush=True)
        t0 = time.time()
        for j, d in enumerate(dl):
            scene = int(d.get("scene_id")); im = int(d.get("image_id", d.get("im_id")))
            score = float(d.get("score", 1.0))
            bbox = d["bbox"]
            # camera K for this image from scene_camera.json
            K = _scene_K(a.bop, a.split, scene, im)
            if K is None:
                continue
            th = d.get("t_hat")
            if th is not None and len(th) == 3 and float(th[2]) > 0:
                t_hat = np.asarray(th, np.float64)   # precomputed depth-map init (metres), see bop_precompute_that.py
            else:
                t_hat = _t_hat_from_bbox(bbox, K, d_m)  # fallback: crude bbox-diameter back-projection
            if t_hat[2] <= 0:
                continue
            A = td.box_A(d_m, t_hat)
            cu, cv = td.project(K, t_hat)
            side = d_m / float(t_hat[2]) * float(K[0, 0]) * crop_pad
            img = np.asarray(Image.open(td.rgb_path(a.bop, a.split, scene, im)).convert("RGB"))
            crop = td.crop_resize(img, cu, cv, max(8.0, side), out_px)
            xn = (crop.astype(np.float32) / 255.0 - td.IMAGENET_MEAN) / td.IMAGENET_STD   # match training norm
            x = torch.from_numpy(xn.transpose(2, 0, 1).copy()).unsqueeze(0).to(dev)
            tic = time.time()
            with torch.no_grad():
                patches = enc(x).float()[0]            # (1,P,D) -> (P,D): beam_map wants one image's patches
                R_map, t_map, _ = dec.map_pose(head, patches, L_rot, L_trans, A, t_hat, beam=a.beam)
            dt = time.time() - tic
            rows.append((scene, im, o, score, R_map.reshape(9), np.asarray(t_map) * 1000.0, dt))
        print(f"[obj {o}] done {len(dl)} in {time.time()-t0:.1f}s", flush=True)

    _write_bop_csv(a.out, rows)
    print(f"[done] wrote {len(rows)} rows -> {a.out}", flush=True)


_CAM_CACHE = {}
def _scene_K(bop, split, scene, im):
    key = (split, scene)
    if key not in _CAM_CACHE:
        p = os.path.join(bop, split, f"{scene:06d}", "scene_camera.json")
        try:
            with open(p) as f:
                _CAM_CACHE[key] = json.load(f)
        except FileNotFoundError:
            _CAM_CACHE[key] = {}
    cam = _CAM_CACHE[key].get(str(im))
    if cam is None:
        return None
    return np.array(cam["cam_K"], np.float64).reshape(3, 3)


def _write_bop_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for scene, im, o, score, R, t, dt in rows:
            w.writerow([scene, im, o, f"{score:.6f}",
                        " ".join(f"{v:.9f}" for v in R),
                        " ".join(f"{v:.6f}" for v in t),
                        f"{dt:.4f}"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--dataset", required=True, choices=["tless", "ycbv"])
    p.add_argument("--bop", required=True, help="BOP dataset root (has <split>/, models_eval/)")
    p.add_argument("--det", required=True, help="detections json (gdrnppdet-pbrreal ... .json)")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--tag", default="_aug")
    p.add_argument("--ckpt", default="best", choices=["best", "last"],
                   help="which checkpoint to load; use 'last' (step 40k) for the reported number since "
                        "'best' is selected on the real test split")
    p.add_argument("--objs", required=True, help="comma/plus list")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test_primesense")
    p.add_argument("--beam", type=int, default=64)
    p.add_argument("--out_px", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a)
