"""Depth-override post-process: rescale each decoded pose's translation ALONG ITS RAY so the depth equals the
sensor depth init, keeping the density's projected centre and rotation.  t_final = t_map * (t_hat_z / t_map_z).
The density's absolute depth is unreliable (synth->real scale bias; it partially cancels any init change via the
crop), but its rotation and image-plane centre are good. Depth comes from the (validated ~1%) sensor init in the
RGB-D detections JSON. Detections lacking a depth t_hat keep their original t_map.

  python -I scripts/bop_override_depth.py --det data/detections/tless_bop19_rgbd.json \
         --in_dir results/tless_rgbd --out_dir results/tless_rgbdz
"""
import os, sys, csv, json, glob, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dets = json.load(open(a.det))
    by_obj = {}
    for d in dets:
        o = int(d.get("category_id", d.get("obj_id", -1)))
        by_obj.setdefault(o, []).append(d)

    for csvp in sorted(glob.glob(os.path.join(a.in_dir, "*.csv"))):
        base = os.path.basename(csvp)
        obj = int(base.replace("tless_obj", "").replace("ycbv_obj", "").replace(".csv", ""))
        rows = list(csv.reader(open(csvp)))
        header, data = rows[0], rows[1:]
        dl = by_obj.get(obj, [])
        n_over, n_keep = 0, 0
        # predict() emits one row per by_obj[o] entry in order; verify counts line up before zipping.
        if len(data) == len(dl):
            pair = zip(data, dl)
        else:
            print(f"[warn] obj{obj}: {len(data)} rows != {len(dl)} dets -> key-match (scene,im,score)", flush=True)
            idx = {}
            for d in dl:
                k = (int(d["scene_id"]), int(d.get("image_id", d.get("im_id"))), round(float(d.get("score", 1.0)), 6))
                idx.setdefault(k, []).append(d)
            pair = []
            for r in data:
                k = (int(r[0]), int(r[1]), round(float(r[3]), 6))
                d = idx.get(k, [None])
                pair.append((r, d.pop(0) if d and d[0] is not None else None))
        out = [header]
        for r, d in pair:
            t = np.array([float(v) for v in r[5].split()], float)   # mm
            th = d.get("t_hat") if d else None
            if th is not None and float(th[2]) > 0 and t[2] > 0:
                scale = (float(th[2]) * 1000.0) / t[2]              # t_hat_z(mm) / t_map_z(mm)
                t = t * scale
                n_over += 1
            else:
                n_keep += 1
            nr = list(r)
            nr[5] = " ".join(f"{v:.6f}" for v in t)
            out.append(nr)
        outp = os.path.join(a.out_dir, base)
        with open(outp, "w", newline="") as f:
            csv.writer(f).writerows(out)
        print(f"obj{obj}: {n_over} depth-override, {n_keep} kept -> {outp}", flush=True)


if __name__ == "__main__":
    main()
