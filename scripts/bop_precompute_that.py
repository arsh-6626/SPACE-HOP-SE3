"""Precompute a depth-map-based translation init t_hat for each BOP detection, writing a new detections JSON
with a `t_hat` field (metres, [tx,ty,tz]). This is the PROPER depth init: t_hat_z is the object's metric depth
read from the sensor depth map over the detection's inner bbox (median), corrected by a per-object surface->center
offset k_o calibrated on train_pbr GT (see depth_offset_<ds>.json). t_hat_{x,y} back-project the bbox centre at
that depth. se3_bop_eval.predict() reads this field (fallback: crude bbox-diameter formula where depth is absent).

Modality note: the RGB density network is unchanged; only the translation INIT uses depth -> report as
"RGB density + depth translation init" (an RGB-D track entry), kept separate from the RGB-only number.

  python -I scripts/bop_precompute_that.py --dataset tless --bop <BOP_root> --split test_primesense \
         --det data/detections/tless_bop19.json --calib data/detections/depth_offset_tless.json \
         --out data/detections/tless_bop19_rgbd.json [--inner 0.6]
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bop", required=True)
    ap.add_argument("--split", default="test_primesense")
    ap.add_argument("--det", required=True)
    ap.add_argument("--calib", required=True, help="per-object k_o JSON (t_hat_z = surf_z*k_o)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=float, default=0.6, help="inner bbox fraction for the depth window")
    a = ap.parse_args()

    calib = {int(k): float(v) for k, v in json.load(open(a.calib)).items()}
    k_default = float(np.median(list(calib.values())))
    dets = json.load(open(a.det))

    cam_cache, dep_cache = {}, {}
    def cam_of(sc):
        if sc not in cam_cache:
            p = os.path.join(a.bop, a.split, f"{sc:06d}", "scene_camera.json")
            cam_cache[sc] = json.load(open(p)) if os.path.exists(p) else {}
        return cam_cache[sc]
    def dep_of(sc, im):
        key = (sc, im)
        if key not in dep_cache:
            p = os.path.join(a.bop, a.split, f"{sc:06d}", "depth", f"{im:06d}.png")
            dep_cache[key] = np.asarray(Image.open(p)).astype(np.float32) if os.path.exists(p) else None
        return dep_cache[key]

    n_depth, n_fallback = 0, 0
    for d in dets:
        o = int(d.get("category_id", d.get("obj_id", -1)))
        sc = int(d["scene_id"]); im = int(d.get("image_id", d.get("im_id")))
        cam = cam_of(sc).get(str(im))
        dep = dep_of(sc, im)
        if cam is None or dep is None:
            d["t_hat"] = None; n_fallback += 1; continue
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        ds = float(cam.get("depth_scale", 1.0))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H, W = dep.shape
        x, y, w, h = [float(v) for v in d["bbox"]]
        cu, cv = x + w / 2, y + h / 2
        x0, x1 = int(max(0, cu - a.inner * w / 2)), int(min(W, cu + a.inner * w / 2))
        y0, y1 = int(max(0, cv - a.inner * h / 2)), int(min(H, cv + a.inner * h / 2))
        patch = dep[y0:y1, x0:x1].ravel(); patch = patch[patch > 0]
        if patch.size < 10:
            d["t_hat"] = None; n_fallback += 1; continue
        surf_mm = float(np.median(patch)) * ds
        tz = surf_mm * calib.get(o, k_default) / 1000.0            # metres, center-corrected
        tx = (cu - cx) * tz / fx
        ty = (cv - cy) * tz / fy
        d["t_hat"] = [float(tx), float(ty), float(tz)]
        n_depth += 1

    json.dump(dets, open(a.out, "w"))
    print(f"[done] {a.out}: {n_depth} with depth t_hat, {n_fallback} fallback (no depth); total {len(dets)}", flush=True)


if __name__ == "__main__":
    main()
