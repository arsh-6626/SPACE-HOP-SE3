"""
SYMSOL-T SE(3) dataset for the hierarchical SE(3) density head (se3_head.py).

Images + annotations come from liepose's synthesizer (plain `image`, the color_mode="none" render that the LiePose
SYMSOL-T experiment uses). Rotation is uniform on SO(3); translation is uniform in the box t in [-1,1]^3. We take the
affine translation box t = A g with A = diag(2,2,2) centred at 0 (t_hat = 0), so the normalized-cube coordinate is
g = t / 2 in [-1/2,1/2]^3 and V_box = |det A| = 8. Every GT pose is inside the box (coverage 1.0) by construction.

prepare() precomputes, per split, the SE(3) path geometry that the head consumes (se3_head.build_geometry) at a fixed
(L_rot, L_trans), saved as .npy:
  meta.json                         {L_rot, L_trans, box_half, kinds}
  {split}_cell0.npy  (R,)           rotation root cell
  {split}_childoff_l{l}.npy (R,K,6) {split}_parent_l{l}.npy (R,12)  {split}_target_l{l}.npy (R,)   per joint level l
  {split}_label.npy  (I,)           per-image shape id         {split}_imgpaths.npy (I,) str
  train: R == I (single GT per image)
  test:  R == sum_i K_i (the equivalent rotations, same translation per image);
         test_equiv_off.npy (I+1,)  row range per image,  test_equiv_R.npy (R,3,3),  test_t.npy (I,3)  for metrics

Train keeps the single-GT rule: one (R,t) per image, no equivalent set. The equivalent set is used only at test time
to average the log-likelihood and to score the min-geodesic rotation error (the SYMSOL / LiePose protocol).
"""
import gzip
import json
import os
import pickle

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

SHAPE_NAMES = ["tet", "cube", "icosa", "cone", "cyl"]


def load_annos(root):
    with gzip.open(os.path.join(root, "annotations.gz"), "rb") as f:
        return pickle.load(f)


def quat_wxyz_to_mat(q):
    """(...,4) wxyz quaternions -> (...,3,3) rotation matrices (via scipy, which is xyzw)."""
    from scipy.spatial.transform import Rotation as Rot
    q = np.asarray(q, np.float64).reshape(-1, 4)
    xyzw = np.roll(q, -1, axis=-1)
    return Rot.from_quat(xyzw).as_matrix()


# ---------------------------------------------------------------- prepare (needs healpy via so3_nested) ----

def rot_z(t):
    """Rotation by angle t (rad) about the optical axis (z), for the in-plane roll relabel.
    An image rolled CCW by theta corresponds to the pose relabel R -> rot_z(-theta) R, t -> rot_z(-theta) t
    (convention verified empirically in symsol_data for SYMSOL-I; re-verified for SYMSOL-T by the roll self-test)."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def prepare(root, out, L_rot, L_trans, box_half=1.0, splits=("train", "test")):
    """Precompute path geometry for the SYMSOL-T dataset at `root` into `out`, at depth (L_rot, L_trans)."""
    import se3_head as sh
    import se3_tree as se3
    os.makedirs(out, exist_ok=True)
    name_to_id = {n: i for i, n in enumerate(SHAPE_NAMES)}
    L = max(L_rot, L_trans)
    kinds = [se3.joint_level_kind(l, L_rot, L_trans) for l in range(1, L + 1)]

    for split in splits:
        annos = load_annos(root)
        I = len(annos)
        labels = np.array([int(a["label_shape"]) for a in annos], np.int64)
        imgpaths = np.array([a["image"] for a in annos], dtype=object)

        if split == "train":
            mats = quat_wxyz_to_mat(np.stack([a["rotation"] for a in annos]))        # (I,3,3)
            t = np.stack([np.asarray(a["translation"], np.float64) for a in annos])   # (I,3)
            gs = t / (2.0 * box_half)                                                 # normalized cube
            assert np.abs(gs).max() <= 0.5 + 1e-6, f"train translation outside box (max |g|={np.abs(gs).max():.3f})"
            geom = sh.build_geometry(mats, gs, L_rot, L_trans)
            np.save(f"{out}/train_cell0.npy", geom["cell0"])
            for l, lv in enumerate(geom["levels"]):
                np.save(f"{out}/train_childoff_l{l}.npy", lv["child_off6"])
                np.save(f"{out}/train_parent_l{l}.npy", lv["parent12"])
                np.save(f"{out}/train_target_l{l}.npy", lv["target"])
            np.save(f"{out}/train_label.npy", labels)
            np.save(f"{out}/train_imgpaths.npy", imgpaths)
            np.save(f"{out}/train_rot.npy", mats.astype(np.float32))   # raw single-GT R, for on-the-fly roll relabel
            np.save(f"{out}/train_g.npy", gs.astype(np.float32))       # raw normalized-cube translation g=t/(2h)
            print(f"[prepare] train: {I} images, single GT, L=({L_rot},{L_trans})", flush=True)
        else:
            # expand the equivalent rotations (same translation) per image
            rows_R, rows_g, equiv_off = [], [], [0]
            t_per_img = np.empty((I, 3), np.float64)
            equiv_R_concat = []
            for i, a in enumerate(annos):
                ne = int(a["num_equivalents"])
                eq = quat_wxyz_to_mat(np.asarray(a["rotations_equivalent"], np.float64).reshape(ne, 4))  # (K,3,3)
                ti = np.asarray(a["translation"], np.float64)
                gi = ti / (2.0 * box_half)
                rows_R.append(eq)
                rows_g.append(np.broadcast_to(gi, (ne, 3)))
                equiv_R_concat.append(eq)
                t_per_img[i] = ti
                equiv_off.append(equiv_off[-1] + ne)
            mats = np.concatenate(rows_R, 0)
            gs = np.concatenate(rows_g, 0)
            assert np.abs(gs).max() <= 0.5 + 1e-6, f"test translation outside box (max |g|={np.abs(gs).max():.3f})"
            geom = sh.build_geometry(mats, gs, L_rot, L_trans)
            np.save(f"{out}/test_cell0.npy", geom["cell0"])
            for l, lv in enumerate(geom["levels"]):
                np.save(f"{out}/test_childoff_l{l}.npy", lv["child_off6"])
                np.save(f"{out}/test_parent_l{l}.npy", lv["parent12"])
                np.save(f"{out}/test_target_l{l}.npy", lv["target"])
            np.save(f"{out}/test_label.npy", labels)
            np.save(f"{out}/test_imgpaths.npy", imgpaths)
            np.save(f"{out}/test_equiv_off.npy", np.array(equiv_off, np.int64))
            np.save(f"{out}/test_equiv_R.npy", np.concatenate(equiv_R_concat, 0).astype(np.float32))
            np.save(f"{out}/test_t.npy", t_per_img.astype(np.float32))
            print(f"[prepare] test: {I} images, {len(mats)} equivalents total, L=({L_rot},{L_trans})", flush=True)

    # Merge per-split image roots so preparing one split (e.g. test) never clobbers another's root (train and test
    # may live in different render dirs). img_root kept for back-compat; img_roots[split] is authoritative.
    meta_path = f"{out}/meta.json"
    img_roots = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            img_roots = json.load(f).get("img_roots", {})
    for split in splits:
        img_roots[split] = os.path.abspath(root)
    with open(meta_path, "w") as f:
        json.dump({"L_rot": L_rot, "L_trans": L_trans, "box_half": box_half, "kinds": kinds,
                   "img_root": os.path.abspath(root), "img_roots": img_roots}, f)
    print(f"[prepare] wrote meta.json (kinds={kinds}, img_roots={list(img_roots)})", flush=True)


# ---------------------------------------------------------------- dataset (torch) ----

def load_meta(out):
    with open(f"{out}/meta.json") as f:
        return json.load(f)


def make_dataset_class():
    import torch
    from torch.utils.data import Dataset

    class Symsolt(Dataset):
        """Train item = (image, single-GT path geometry). Test is accessed via item()/equiv geometry per image.
        Images are decoded from PNG on the fly (dataset is small and on local SSD)."""
        def __init__(self, out, split, augment=False):
            self.out, self.split = out, split
            self.meta = load_meta(out)
            self.L_rot, self.L_trans = self.meta["L_rot"], self.meta["L_trans"]
            self.L = max(self.L_rot, self.L_trans)
            self.img_root = self.meta.get("img_roots", {}).get(split, self.meta["img_root"])
            self.label = np.load(f"{out}/{split}_label.npy")
            self.imgpaths = np.load(f"{out}/{split}_imgpaths.npy", allow_pickle=True)
            self.cell0 = np.load(f"{out}/{split}_cell0.npy", mmap_mode="r")
            self.childoff = [np.load(f"{out}/{split}_childoff_l{l}.npy", mmap_mode="r") for l in range(self.L)]
            self.parent = [np.load(f"{out}/{split}_parent_l{l}.npy", mmap_mode="r") for l in range(self.L)]
            self.target = [np.load(f"{out}/{split}_target_l{l}.npy", mmap_mode="r") for l in range(self.L)]
            # exact in-plane roll augmentation (train only): roll the frame by theta and relabel the single GT pose
            # R -> rot_z(-theta) R, g -> rot_z(-theta) g. Keeps the single-GT rule (no equivalent set). rot_z preserves
            # the xy-radius, so theta is rejection-sampled to keep g in the cube box, with 90-degree multiples (which
            # always map the square box to itself, crop-free) as the guaranteed-valid fallback.
            self.augment = bool(augment) and split == "train"
            if self.augment:
                import se3_head as _sh
                self._build_geometry = _sh.build_geometry
                self.rot = np.load(f"{out}/train_rot.npy")   # (N,3,3)
                self.g = np.load(f"{out}/train_g.npy")       # (N,3)
            if split == "test":
                self.equiv_off = np.load(f"{out}/test_equiv_off.npy")
                self.equiv_R = np.load(f"{out}/test_equiv_R.npy", mmap_mode="r")
                self.t = np.load(f"{out}/test_t.npy")

        def __len__(self):
            return len(self.label)

        def _raw(self, path):
            from PIL import Image
            return np.asarray(Image.open(os.path.join(self.img_root, path)).convert("RGB"), np.uint8)

        def _norm(self, arr):
            x = arr.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(x.transpose(2, 0, 1).copy())

        def _img(self, path):
            return self._norm(self._raw(path))

        def _roll(self, arr, theta_deg):
            """Rotate the frame CCW by theta_deg about its centre, filling exposed corners with the (uniform)
            background median colour. Returns the rolled uint8 frame."""
            from PIL import Image
            corners = np.concatenate([arr[:8, :8].reshape(-1, 3), arr[:8, -8:].reshape(-1, 3),
                                      arr[-8:, :8].reshape(-1, 3), arr[-8:, -8:].reshape(-1, 3)])
            bg = tuple(int(v) for v in np.median(corners, 0))
            return np.asarray(Image.fromarray(arr).rotate(theta_deg, resample=Image.BILINEAR, fillcolor=bg), np.uint8)

        def _sample_theta(self, g):
            """Pick an in-plane roll theta (deg) whose relabel rot_z(-theta) g stays in the cube box [-0.5,0.5]^3.
            rot_z preserves the xy-radius, so small-radius samples accept any theta; otherwise rejection-sample a few
            times, then fall back to a random 90-degree multiple (always exact and in-box on a square frame)."""
            gx, gy = float(g[0]), float(g[1])
            r = (gx * gx + gy * gy) ** 0.5
            if r <= 0.5:                                  # any angle keeps |gx'|,|gy'| <= r <= 0.5
                return float(np.random.uniform(0.0, 360.0))
            for _ in range(8):
                th = float(np.random.uniform(0.0, 360.0))
                a = np.deg2rad(-th); c, s = np.cos(a), np.sin(a)
                if abs(c * gx - s * gy) <= 0.5 and abs(s * gx + c * gy) <= 0.5:
                    return th
            return float(np.random.choice([0.0, 90.0, 180.0, 270.0]))

        def _aug_item(self, i):
            theta = self._sample_theta(self.g[i])
            img = self._norm(self._roll(self._raw(self.imgpaths[i]), theta))
            Rz = rot_z(-np.deg2rad(theta))
            Rp = (Rz @ self.rot[i].astype(np.float64)).reshape(1, 3, 3)
            gp = (Rz @ self.g[i].astype(np.float64)).reshape(1, 3)
            geom = self._build_geometry(Rp, gp, self.L_rot, self.L_trans)
            lv = geom["levels"]
            out = {"image": img, "label": int(self.label[i]),
                   "cell0": torch.as_tensor(int(geom["cell0"][0])),
                   "child_off6": torch.as_tensor(np.stack([lv[l]["child_off6"][0] for l in range(self.L)])),
                   "parent12": torch.as_tensor(np.stack([lv[l]["parent12"][0] for l in range(self.L)])),
                   "target": torch.as_tensor(np.stack([int(lv[l]["target"][0]) for l in range(self.L)]))}
            if "refine" in geom:   # continuous within-cell GT residual targets for the refinement head
                r = geom["refine"]
                out["refine_code12"] = torch.as_tensor(r["code12"][0])
                out["refine_rot_resid"] = torch.as_tensor(r["rot_resid"][0])
                out["refine_g_resid"] = torch.as_tensor(r["g_resid"][0])
            return out

        def _levels_rows(self, rows):
            """Assemble the head's `levels` list for the given geometry row indices (int or slice/array)."""
            return [{"child_off6": torch.as_tensor(np.array(self.childoff[l][rows])),
                     "parent12": torch.as_tensor(np.array(self.parent[l][rows])),
                     "target": torch.as_tensor(np.array(self.target[l][rows]))} for l in range(self.L)]

        def __getitem__(self, i):
            # train: one geometry row per image (row index == image index)
            if self.augment:
                return self._aug_item(i)
            img = self._img(self.imgpaths[i])
            lv = self._levels_rows(i)
            return {"image": img, "label": int(self.label[i]),
                    "cell0": torch.as_tensor(int(self.cell0[i])),
                    "child_off6": torch.stack([x["child_off6"] for x in lv]),   # (L,K,6)
                    "parent12": torch.stack([x["parent12"] for x in lv]),       # (L,12)
                    "target": torch.stack([x["target"] for x in lv])}           # (L,)

        # ---- test helpers (equivalent set per image) ----
        def equiv_rows(self, i):
            return int(self.equiv_off[i]), int(self.equiv_off[i + 1])

        def test_item(self, i):
            a, b = self.equiv_rows(i)
            img = self._img(self.imgpaths[i])
            cell0 = torch.as_tensor(np.array(self.cell0[a:b]).astype(np.int64))         # (K,)
            levels = [{"child_off6": torch.as_tensor(np.array(self.childoff[l][a:b])),
                       "parent12": torch.as_tensor(np.array(self.parent[l][a:b])),
                       "target": torch.as_tensor(np.array(self.target[l][a:b]).astype(np.int64))}
                      for l in range(self.L)]
            return {"image": img, "label": int(self.label[i]), "cell0": cell0, "levels": levels,
                    "equiv_R": np.array(self.equiv_R[a:b], np.float32), "t": self.t[i]}

    return Symsolt


def collate_train(batch):
    """Stack train items into {image, label, cell0, levels:[{child_off6,parent12,target}]}."""
    import torch
    L = batch[0]["child_off6"].shape[0]
    out = {"image": torch.stack([b["image"] for b in batch]),
           "label": torch.tensor([b["label"] for b in batch]),
           "cell0": torch.stack([b["cell0"] for b in batch]),
           "levels": [{"child_off6": torch.stack([b["child_off6"][l] for b in batch]),
                       "parent12": torch.stack([b["parent12"][l] for b in batch]),
                       "target": torch.stack([b["target"][l] for b in batch])} for l in range(L)]}
    if "refine_code12" in batch[0]:
        out["refine"] = {"code12": torch.stack([b["refine_code12"] for b in batch]),
                         "rot_resid": torch.stack([b["refine_rot_resid"] for b in batch]),
                         "g_resid": torch.stack([b["refine_g_resid"] for b in batch])}
    return out


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("--root", required=True, help="rendered SYMSOL-T dir (has annotations.gz + <shape>/images)")
    ap.add_argument("--out", required=True, help="output dir for precomputed geometry")
    ap.add_argument("--L_rot", type=int, default=5)
    ap.add_argument("--L_trans", type=int, default=5)
    ap.add_argument("--box_half", type=float, default=1.0)
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()
    prepare(a.root, a.out, a.L_rot, a.L_trans, a.box_half, tuple(a.splits.split(",")))
