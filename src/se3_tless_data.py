"""
T-LESS SE(3) dataset for the hierarchical SE(3) density head (se3_head.py), matching SpyroPose's protocol
(Haugaard et al. 2023, arXiv:2303.05308) so the distribution log-likelihood is comparable to their Table 5.

Translation box (their Eq.1-3), object-dependent, in METRES:
    A = [[d, 0, t_hat_x], [0, d, t_hat_y], [0, 0, t_hat_z]],   t = t_hat + A g,   g in [-1/2,1/2]^3,
    det(A) = d^2 * t_hat_z,   total SE(3) volume = V_SO3 * det(A) = pi^2 * d^2 * t_hat_z.
d is the object diameter (models_eval/models_info.json, mm->m); t_hat is a coarse position estimate (a detector at
test). The uniform SE(3) density is 1/(pi^2 det(A)), i.e. uniform LL = -ln(pi^2 d^2 t_hat_z); this reproduces
SpyroPose's per-object Uniform row (obj 1/14/25/27 = 3.5/3.3/2.4/1.7) to +-0.05 -- see docs/tless_se3_protocol.md.

The head consumes a rotation matrix R and the normalized-cube translation g = A^{-1}(t - t_hat). We pick t_hat as:
  - train ('jitter'): t_hat = t_GT + A(t_GT) e~, e~ ~ N(0, sigma I) truncated to ||e~||<=0.45, so g = A^{-1}(t-t_hat)
    ~ -e~ lands at a varying offset inside the box (if t_hat==t_GT then g==0 and translation would never train).
  - eval ('gt'): t_hat = t_GT (oracle detection) -> g = 0; the single-GT LL is scored with the per-instance det(A).
    This is the oracle-t_hat LL; a real detector (g!=0) gives the fully comparable real-image number (TODO: wire
    BOP default detections). t_hat only moves the crop/box centre; it does NOT change the normalization det(A).

LL is scored at the SINGLE annotated GT pose (no symmetry-equivalent expansion): the model learns the multimodal
density and we evaluate it at the one GT, exactly as SpyroPose reports. Train on PBR (train_pbr), test on real
(test_primesense, restricted to the BOP19 test targets). Images are cropped about the projected t_hat and resized.

    python src/se3_tless_data.py index --root <tless> --out <cache> --objs 1,14,25,27 --split train_pbr
    python src/se3_tless_data.py check --root <tless> --out <cache> --objs 1,14,25,27   # real-crop uniform floor
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- BOP readers ----

def load_models_info(root):
    """obj_id -> dict (has 'diameter' in mm). Prefer models_eval (used for BOP metrics)."""
    for sub in ("models_eval", "models", "models_cad", "models_reconst"):
        p = os.path.join(root, sub, "models_info.json")
        if os.path.exists(p):
            mi = json.load(open(p))
            return {int(k): v for k, v in mi.items()}
    raise FileNotFoundError(f"no models_info.json under {root}")


def diameters_m(root, objs):
    """object diameters in METRES (models_info is in mm)."""
    mi = load_models_info(root)
    return {o: mi[o]["diameter"] / 1000.0 for o in objs}


def box_A(d_m, t_hat_m):
    """SpyroPose affine translation box A for diameter d (m) and position estimate t_hat (m)."""
    tx, ty, tz = (float(v) for v in t_hat_m)
    return np.array([[d_m, 0.0, tx], [0.0, d_m, ty], [0.0, 0.0, tz]], np.float64)


# ---------------------------------------------------------------- indexing ----

def _scene_dirs(root, split):
    base = os.path.join(root, split)
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def index_split(root, out, objs, split, visib_thresh=0.3, bop19_targets=True):
    """Build a per-object instance index for `split` and save it to `out`/{split}_obj{o}.npz.
    Each row: scene, im, gt_idx, R(9), t(3, mm), K(9), bbox(4: x,y,w,h), visib_fract.
    For test splits restricted to the BOP19 targets (the evaluated (scene,im,obj) instances)."""
    os.makedirs(out, exist_ok=True)
    targets = None
    if bop19_targets and "test" in split:
        tp = os.path.join(root, "test_targets_bop19.json")
        if os.path.exists(tp):
            targets = set()
            for t in json.load(open(tp)):
                targets.add((t["scene_id"], t["im_id"], t["obj_id"]))

    rows = {o: [] for o in objs}
    for sc in _scene_dirs(root, split):
        scid = int(sc)
        sdir = os.path.join(root, split, sc)
        gt = json.load(open(os.path.join(sdir, "scene_gt.json")))
        gti = json.load(open(os.path.join(sdir, "scene_gt_info.json")))
        cam = json.load(open(os.path.join(sdir, "scene_camera.json")))
        for im_str, insts in gt.items():
            im = int(im_str)
            K = np.array(cam[im_str]["cam_K"], np.float64)
            infos = gti[im_str]
            for gi, inst in enumerate(insts):
                o = inst["obj_id"]
                if o not in objs:
                    continue
                if targets is not None and (scid, im, o) not in targets:
                    continue
                vf = float(infos[gi].get("visib_fract", 1.0))
                if targets is None and vf < visib_thresh:
                    continue
                bbox = infos[gi].get("bbox_visib") or infos[gi].get("bbox_obj") or [0, 0, 0, 0]
                rows[o].append((scid, im, gi,
                                np.array(inst["cam_R_m2c"], np.float64),
                                np.array(inst["cam_t_m2c"], np.float64),   # mm
                                K, np.array(bbox, np.float64), vf))
    for o in objs:
        r = rows[o]
        if not r:
            print(f"[index] {split} obj {o}: 0 instances", flush=True)
            continue
        np.savez(os.path.join(out, f"{split}_obj{o}.npz"),
                 scene=np.array([x[0] for x in r], np.int32),
                 im=np.array([x[1] for x in r], np.int32),
                 gt_idx=np.array([x[2] for x in r], np.int32),
                 R=np.stack([x[3] for x in r]).reshape(-1, 9),
                 t=np.stack([x[4] for x in r]),
                 K=np.stack([x[5] for x in r]).reshape(-1, 9),
                 bbox=np.stack([x[6] for x in r]),
                 visib=np.array([x[7] for x in r], np.float32),
                 split=split, obj=o)
        print(f"[index] {split} obj {o}: {len(r)} instances -> {split}_obj{o}.npz", flush=True)


# ---------------------------------------------------------------- crop ----

def rgb_path(root, split, scene, im):
    sdir = os.path.join(root, split, f"{scene:06d}")
    for ext in (".png", ".jpg"):
        p = os.path.join(sdir, "rgb", f"{im:06d}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no rgb for scene {scene} im {im} under {sdir}")


def project(K, t_m):
    """K (3,3), t in metres -> pixel (u,v)."""
    uvw = K @ (np.asarray(t_m, np.float64))
    return uvw[0] / uvw[2], uvw[1] / uvw[2]


def crop_resize(img, cu, cv, side_px, out_px):
    """Square crop of side `side_px` centred at (cu,cv) from HxWx3 uint8, padded if out of bounds, resized to out_px."""
    from PIL import Image
    H, W = img.shape[:2]
    half = side_px / 2.0
    x0, y0 = int(round(cu - half)), int(round(cv - half))
    x1, y1 = int(round(cu + half)), int(round(cv + half))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[ys0:ys1, xs0:xs1]
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    return np.asarray(Image.fromarray(crop).resize((out_px, out_px), Image.BILINEAR), np.uint8)


# ---------------------------------------------------------------- dataset ----

def make_dataset_class():
    import torch
    from torch.utils.data import Dataset

    class Tless(Dataset):
        """One object's SE(3) dataset. __getitem__ returns image + (R, g, det_A, logVbox) for build_geometry."""
        def __init__(self, root, cache, split, obj, L_rot, L_trans, d_m,
                     t_hat="jitter", sigma=0.25, crop_pad=1.4, out_px=224, seed=0):
            z = np.load(os.path.join(cache, f"{split}_obj{obj}.npz"))
            self.root, self.split, self.obj = root, split, obj
            self.L_rot, self.L_trans = L_rot, L_trans
            self.d_m = d_m
            self.t_hat_mode, self.sigma, self.crop_pad, self.out_px = t_hat, sigma, crop_pad, out_px
            self.scene, self.im, self.R = z["scene"], z["im"], z["R"].reshape(-1, 3, 3)
            self.t = z["t"]                                   # mm
            self.K = z["K"].reshape(-1, 3, 3)
            self.rng = np.random.RandomState(seed + obj)

        def __len__(self):
            return len(self.scene)

        def _t_hat(self, t_m, A):
            if self.t_hat_mode == "gt":
                return t_m.copy()
            # jitter: t_hat = t + A e~, e~ truncated normal ||e~||<=0.45  -> g = A^{-1}(t - t_hat) ~ -e~ in the box
            while True:
                e = self.rng.normal(0, self.sigma, 3)
                if np.linalg.norm(e) <= 0.45:
                    break
            return t_m + A @ e

        def __getitem__(self, i):
            t_m = self.t[i] / 1000.0                          # mm -> m
            A = box_A(self.d_m, t_m)                           # box uses t_hat; approx with t_GT for the jitter dir
            t_hat = self._t_hat(t_m, A)
            A = box_A(self.d_m, t_hat)                         # final box about the estimate
            g = np.linalg.solve(A, t_m - t_hat)                # normalized-cube coord of the GT translation
            g = np.clip(g, -0.49999, 0.49999)
            det_A = self.d_m ** 2 * float(t_hat[2])            # == |det A|

            # crop about projected t_hat; side ~ object diameter in px * pad
            img = np.asarray(_open_rgb(rgb_path(self.root, self.split, int(self.scene[i]), int(self.im[i]))), np.uint8)
            cu, cv = project(self.K[i], t_hat)
            side = self.d_m / float(t_hat[2]) * float(self.K[i][0, 0]) * self.crop_pad
            crop = crop_resize(img, cu, cv, max(8.0, side), self.out_px)
            x = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            return {"image": torch.from_numpy(x.transpose(2, 0, 1).copy()),
                    "R": torch.from_numpy(self.R[i].astype(np.float32)),
                    "g": torch.from_numpy(g.astype(np.float32)),
                    "det_A": float(det_A), "label": int(self.obj)}

    return Tless


def _open_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def collate(batch):
    """Build the head geometry for a batch via se3_head.build_geometry(R, g). Returns image + geometry + det_A."""
    import torch
    import se3_head as sh
    mats = np.stack([b["R"].numpy() for b in batch])
    gs = np.stack([b["g"].numpy() for b in batch])
    L_rot = batch[0].get("_L_rot"); L_trans = batch[0].get("_L_trans")
    geom = sh.build_geometry(mats, gs, collate.L_rot, collate.L_trans)
    out = {"image": torch.stack([b["image"] for b in batch]),
           "det_A": torch.tensor([b["det_A"] for b in batch], dtype=torch.float64),
           "label": torch.tensor([b["label"] for b in batch]),
           "cell0": torch.as_tensor(geom["cell0"]),
           "levels": [{"kind": lv["kind"],
                       "child_off6": torch.as_tensor(lv["child_off6"]),
                       "parent12": torch.as_tensor(lv["parent12"]),
                       "target": torch.as_tensor(lv["target"])} for lv in geom["levels"]]}
    return out


# ---------------------------------------------------------------- self-test: real-crop uniform floor ----

def check(root, cache, objs, L_rot=5, L_trans=5, n=24):
    """Load real crops and verify a zero-init head scores the per-instance uniform floor -ln(pi^2 det(A)).
    This validates the end-to-end g / det(A) wiring on real T-LESS data (the analogue of se3_head check A)."""
    import torch
    import se3_head as sh
    import se3_tree as se3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tless = make_dataset_class()
    dmap = diameters_m(root, objs)
    worst = 0.0
    for o in objs:
        ds = Tless(root, cache, "train_pbr", o, L_rot, L_trans, dmap[o], t_hat="jitter")
        idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
        items = [ds[int(i)] for i in idx]
        collate.L_rot, collate.L_trans = L_rot, L_trans
        b = collate(items)
        head = sh.Se3DensityHead(embed_dim=128, L_rot=L_rot, L_trans=L_trans, n_layers=1, heads=4).to(device).eval()
        patches = torch.randn(len(items), 16, 128, device=device)
        lv = [{"child_off6": x["child_off6"].to(device), "parent12": x["parent12"].to(device),
               "target": x["target"].to(device)} for x in b["levels"]]
        with torch.no_grad():
            lp_cell = head.path_loglik(patches, b["cell0"].to(device), lv).cpu().numpy()
        N = se3.n_joint_cells(L_rot, L_trans)
        det_A = b["det_A"].numpy()
        dens = lp_cell + np.log(N) - np.log(se3.V_SO3 * det_A)            # per-instance log density
        floor = -np.log(se3.V_SO3 * det_A)                                # per-instance uniform floor
        dev = float(np.max(np.abs(dens - floor)))
        worst = max(worst, dev)
        print(f"  obj {o:>2}: n={len(items):>2} d={dmap[o]*1000:6.1f}mm  mean uniform floor {floor.mean():.3f}  "
              f"NLL(cell)=lnN={np.log(N):.3f} (dev {abs(-lp_cell.mean()-np.log(N)):.1e})  density dev {dev:.1e}")
    assert worst < 1e-4, f"real-crop uniform floor off by {worst:.2e}"
    print(f"CHECK OK: real T-LESS crops score the per-instance uniform floor (max density dev {worst:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "check"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", "--cache", dest="out", required=True)
    ap.add_argument("--objs", default="1,14,25,27")
    ap.add_argument("--split", default="train_pbr")
    ap.add_argument("--visib_thresh", type=float, default=0.3)
    ap.add_argument("--L_rot", type=int, default=5)
    ap.add_argument("--L_trans", type=int, default=5)
    a = ap.parse_args()
    objs = [int(x) for x in a.objs.split(",")]
    if a.cmd == "index":
        index_split(a.root, a.out, objs, a.split, a.visib_thresh)
    else:
        check(a.root, a.out, objs, a.L_rot, a.L_trans)
