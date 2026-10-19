"""
SYMSOL (symmetric_solids) data for the hierarchical SO(3) density head.

Two stages, so the GPU box never needs tensorflow:
  1. dump  -- run once where tfds is installed; downloads "symmetric_solids" and writes plain .npy under <out>.
             `python src/symsol_data.py dump --out /scratch/.../symsol`
  2. SymsolDataset -- a torch Dataset over those .npy files. Each item carries the GT path geometry the density head
             needs (cell0, child indices, per-level child offsets and parent rotation), precomputed here with
             so3_nested (healpy/numpy on CPU) so symsol_head.py stays pure torch.

Protocol (IPDF / Hsiao): train on the single rendered `rotation`; test carries the full `rotations_equivalent` set,
over which the log-likelihood is averaged. One joint model over all five shapes [tet, cube, icosa, cone, cyl].
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import so3_nested as g

SHAPE_NAMES = ["tet", "cube", "icosa", "cone", "cyl"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ path geometry (CPU, healpy)

def path_geometry(mats, level):
    """(M,3,3) rotations -> the per-path tensors the head consumes, all at a fixed `level`:
        cell0 (M,) int64, child_idx (M,level) int64, child_off (M,level,8,3) f32, parent_enc (M,level,9) f32.
    child_off[m,l] are the 8 children of sample m's level-l node, as rotation-vector offsets from that node,
    normalised by the level's cell size; parent_enc[m,l] is that node's rotation, flattened, for conditioning."""
    mats = np.asarray(mats, np.float64).reshape(-1, 3, 3)
    M = mats.shape[0]
    cells = g.rotations_to_cells(mats, level)
    node = np.empty((M, level + 1), np.int64)
    child_idx = np.empty((M, level), np.int64)
    for i, c in enumerate(cells):
        c0, ch = g.cell_to_path(c, level)
        node[i] = g.path_node_cells(c0, ch)
        child_idx[i] = ch
    child_off = np.empty((M, level, 8, 3), np.float32)
    parent_enc = np.empty((M, level, 9), np.float32)
    for l in range(level):
        par = node[:, l]
        chf = g.parent_to_children_cells(par, l)                       # (M,8)
        R_par = g.cell_index_to_matrix(par, l)                         # (M,3,3)
        R_ch = g.cell_index_to_matrix(chf, l + 1)                      # (M,8,3,3)
        rel = np.einsum('mcij,mkj->mcik', R_ch, R_par)                 # R_child @ R_parent^T
        rv = Rot.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(M, 8, 3)
        child_off[:, l] = (rv / g.cell_size_rad(l + 1)).astype(np.float32)
        parent_enc[:, l] = R_par.reshape(M, 9).astype(np.float32)
    return node[:, 0].copy(), child_idx, child_off, parent_enc


def node_children_features(node_cells, level):
    """For beam/MAP decoding: given (M,) flat cells at `level`, return the features to score their children ->
    child_off (M,8,3) f32 normalised offsets, parent_enc (M,9) f32, child_cells (M,8) int64."""
    node_cells = np.asarray(node_cells, np.int64)
    M = node_cells.shape[0]
    chf = g.parent_to_children_cells(node_cells, level)            # (M,8)
    R_par = g.cell_index_to_matrix(node_cells, level)              # (M,3,3)
    R_ch = g.cell_index_to_matrix(chf, level + 1)                  # (M,8,3,3)
    rel = np.einsum('mcij,mkj->mcik', R_ch, R_par)
    rv = Rot.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(M, 8, 3)
    return (rv / g.cell_size_rad(level + 1)).astype(np.float32), R_par.reshape(M, 9).astype(np.float32), chf


# ------------------------------------------------------------------ stage 1: dump tfds -> .npy

def dump(out, shapes=SHAPE_NAMES, splits=("train", "test")):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    os.makedirs(out, exist_ok=True)
    shape_ids = [SHAPE_NAMES.index(s) for s in shapes]
    for split in splits:
        ds = tfds.load("symmetric_solids", split=split)
        ds = ds.filter(lambda x: tf.reduce_any(tf.equal(x["label_shape"], shape_ids)))
        imgs, rots, labels, equiv = [], [], [], []
        for ex in tfds.as_numpy(ds):
            imgs.append(ex["image"])
            rots.append(ex["rotation"].astype(np.float32))
            labels.append(int(ex["label_shape"]))
            if split != "train":
                equiv.append(ex["rotations_equivalent"].astype(np.float32))   # (K_i,3,3), ragged
        np.save(f"{out}/{split}_images.npy", np.stack(imgs).astype(np.uint8))
        np.save(f"{out}/{split}_rot.npy", np.stack(rots))
        np.save(f"{out}/{split}_label.npy", np.array(labels, np.int64))
        if split != "train":
            np.save(f"{out}/{split}_equiv.npy", np.array(equiv, dtype=object), allow_pickle=True)
        print(f"[{split}] {len(imgs)} images -> {out}/{split}_*.npy "
              f"(equiv K: {np.mean([len(e) for e in equiv]):.0f} avg)" if equiv else
              f"[{split}] {len(imgs)} images -> {out}/{split}_*.npy")


# ------------------------------------------------------------------ stage 2: torch Dataset

def _import_torch():
    import torch
    from torch.utils.data import Dataset
    return torch, Dataset


def make_dataset_class():
    torch, Dataset = _import_torch()

    class SymsolDataset(Dataset):
        """Train: one item = (image, single-rotation path geometry). Test: image + the equivalent set's geometry
        (ragged K handled by returning lists; use the provided collate for the test loader)."""
        def __init__(self, root, split, level):
            self.split, self.level = split, level
            self.images = np.load(f"{root}/{split}_images.npy", mmap_mode="r")
            self.rot = np.load(f"{root}/{split}_rot.npy")
            self.label = np.load(f"{root}/{split}_label.npy")
            self.equiv = (np.load(f"{root}/{split}_equiv.npy", allow_pickle=True)
                          if split != "train" else None)

        def __len__(self):
            return len(self.rot)

        def _img(self, i):
            x = np.asarray(self.images[i], np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(x.transpose(2, 0, 1).copy())

        def __getitem__(self, i):
            img, label = self._img(i), int(self.label[i])
            if self.split == "train":
                cell0, child_idx, child_off, parent_enc = path_geometry(self.rot[i:i + 1], self.level)
                return {"image": img, "label": label,
                        "cell0": torch.from_numpy(cell0)[0],
                        "child_idx": torch.from_numpy(child_idx)[0],
                        "child_off": torch.from_numpy(child_off)[0],
                        "parent_enc": torch.from_numpy(parent_enc)[0]}
            equiv = self.equiv[i]                                           # (K,3,3)
            cell0, child_idx, child_off, parent_enc = path_geometry(equiv, self.level)
            return {"image": img, "label": label,
                    "cell0": torch.from_numpy(cell0),                       # (K,)
                    "child_idx": torch.from_numpy(child_idx),               # (K,level)
                    "child_off": torch.from_numpy(child_off),               # (K,level,8,3)
                    "parent_enc": torch.from_numpy(parent_enc)}             # (K,level,9)

    return SymsolDataset


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dump"); d.add_argument("--out", required=True)
    d.add_argument("--shapes", nargs="+", default=SHAPE_NAMES)
    d.add_argument("--splits", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.out, a.shapes, a.splits)
    else:
        ap.print_help()
