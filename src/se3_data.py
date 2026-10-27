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

    with open(f"{out}/meta.json", "w") as f:
        json.dump({"L_rot": L_rot, "L_trans": L_trans, "box_half": box_half, "kinds": kinds,
                   "img_root": os.path.abspath(root)}, f)
    print(f"[prepare] wrote meta.json (kinds={kinds})", flush=True)


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
        def __init__(self, out, split):
            self.out, self.split = out, split
            self.meta = load_meta(out)
            self.L = max(self.meta["L_rot"], self.meta["L_trans"])
            self.img_root = self.meta["img_root"]
            self.label = np.load(f"{out}/{split}_label.npy")
            self.imgpaths = np.load(f"{out}/{split}_imgpaths.npy", allow_pickle=True)
            self.cell0 = np.load(f"{out}/{split}_cell0.npy", mmap_mode="r")
            self.childoff = [np.load(f"{out}/{split}_childoff_l{l}.npy", mmap_mode="r") for l in range(self.L)]
            self.parent = [np.load(f"{out}/{split}_parent_l{l}.npy", mmap_mode="r") for l in range(self.L)]
            self.target = [np.load(f"{out}/{split}_target_l{l}.npy", mmap_mode="r") for l in range(self.L)]
            if split == "test":
                self.equiv_off = np.load(f"{out}/test_equiv_off.npy")
                self.equiv_R = np.load(f"{out}/test_equiv_R.npy", mmap_mode="r")
                self.t = np.load(f"{out}/test_t.npy")

        def __len__(self):
            return len(self.label)

        def _img(self, path):
            from PIL import Image
            arr = np.asarray(Image.open(os.path.join(self.img_root, path)).convert("RGB"), np.uint8)
            x = arr.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(x.transpose(2, 0, 1).copy())

        def _levels_rows(self, rows):
            """Assemble the head's `levels` list for the given geometry row indices (int or slice/array)."""
            return [{"child_off6": torch.as_tensor(np.array(self.childoff[l][rows])),
                     "parent12": torch.as_tensor(np.array(self.parent[l][rows])),
                     "target": torch.as_tensor(np.array(self.target[l][rows]))} for l in range(self.L)]

        def __getitem__(self, i):
            # train: one geometry row per image (row index == image index)
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
