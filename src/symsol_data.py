"""
SYMSOL (symmetric_solids) data for the hierarchical SO(3) density head.

Staged so the GPU box never needs tensorflow, and the training dataloader never touches healpy/scipy:

  1. dump    -- run once where tfds is installed. Downloads "symmetric_solids" and writes, per split, the images as a
                single concatenated PNG blob (uint8) + int64 byte-offsets (mmap-able, no pickle), the labels, and the
                raw rotation matrices (train: the single `rotation`; test: the full `rotations_equivalent` set,
                concatenated with per-image offsets). No geometry, no healpy here.
                    <tfds-env>/python src/symsol_data.py dump --out <dir>
  2. prepare -- run once where healpy+scipy are installed. Reads those rotations and precomputes the per-path geometry
                the density head consumes (cell0, child indices, per-level child offsets and parent rotation), at a
                fixed `--level` (dump at the deepest level you may train/eval; shallower levels are exact prefixes).
                    <healpy-env>/python src/symsol_data.py prepare --out <dir> --level 6
  3. SymsolDataset -- a pure-torch/numpy Dataset over those .npy files (PNG decode with PIL, geometry sliced to the
                requested level). Train yields the single-rotation path; test yields the equivalent set's geometry.

Protocol (IPDF / Hsiao): train on the single rendered `rotation`; test carries the full `rotations_equivalent` set,
over which the log-likelihood is averaged. One joint model over all five shapes [tet, cube, icosa, cone, cyl].
"""
import os
import numpy as np

# scipy and so3_nested (healpy) are imported lazily inside the geometry functions below, so stage 1 (`dump`) runs in a
# tensorflow-only environment with no healpy/scipy, and the torch Dataset needs neither.

SHAPE_NAMES = ["tet", "cube", "icosa", "cone", "cyl"]            # SYMSOL I (the default pipeline)
ALL_SHAPES = ["tet", "cube", "icosa", "cone", "cyl", "tetX", "cylO", "sphereX"]  # tfds label_shape order (I + II)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ path geometry (CPU, healpy)

def rot_z(t):
    """Rotation by angle t (rad) about the optical axis (z), for the in-plane roll relabel."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def path_geometry(mats, level):
    """(M,3,3) rotations -> the per-path tensors the head consumes, all at a fixed `level`:
        cell0 (M,) int64, child_idx (M,level) int64, child_off (M,level,8,3) f32, parent_enc (M,level,9) f32.
    child_off[m,l] are the 8 children of sample m's level-l node, as rotation-vector offsets from that node,
    normalised by the level's cell size; parent_enc[m,l] is that node's rotation, flattened, for conditioning."""
    from scipy.spatial.transform import Rotation as Rot
    import so3_nested as g
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
    from scipy.spatial.transform import Rotation as Rot
    import so3_nested as g
    node_cells = np.asarray(node_cells, np.int64)
    M = node_cells.shape[0]
    chf = g.parent_to_children_cells(node_cells, level)            # (M,8)
    R_par = g.cell_index_to_matrix(node_cells, level)              # (M,3,3)
    R_ch = g.cell_index_to_matrix(chf, level + 1)                  # (M,8,3,3)
    rel = np.einsum('mcij,mkj->mcik', R_ch, R_par)
    rv = Rot.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(M, 8, 3)
    return (rv / g.cell_size_rad(level + 1)).astype(np.float32), R_par.reshape(M, 9).astype(np.float32), chf


# ------------------------------------------------------------------ stage 1: dump tfds -> PNG blob + rotations

def dump(out, shapes=SHAPE_NAMES, splits=("train", "test")):
    """tfds "symmetric_solids" -> per split: {split}_png.npy (uint8 concat), {split}_png_off.npy (int64 N+1),
    {split}_label.npy (int64 N), and rotations. Train: train_rot.npy (N,3,3). Test: test_equiv_mats.npy (T,3,3)
    concatenated over all images' equivalent sets + test_equiv_off.npy (int64 N+1). No geometry (see `prepare`)."""
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    os.makedirs(out, exist_ok=True)
    shape_ids = [ALL_SHAPES.index(s) for s in shapes]
    for split in splits:
        ds = tfds.load("symmetric_solids", split=split)
        ds = ds.filter(lambda x: tf.reduce_any(tf.equal(x["label_shape"], shape_ids)))
        blobs, off, labels = [], [0], []
        rots = []                                        # train: (3,3) each; test: list of (K,3,3)
        equiv_off = [0]
        for ex in tfds.as_numpy(ds):
            png = tf.io.encode_png(tf.convert_to_tensor(ex["image"], tf.uint8)).numpy()
            blobs.append(np.frombuffer(png, np.uint8))
            off.append(off[-1] + len(png))
            labels.append(int(ex["label_shape"]))
            if split == "train":
                rots.append(ex["rotation"].astype(np.float32).reshape(3, 3))
            else:
                e = ex["rotations_equivalent"].astype(np.float32).reshape(-1, 3, 3)   # (K_i,3,3)
                rots.append(e)
                equiv_off.append(equiv_off[-1] + e.shape[0])
        np.save(f"{out}/{split}_png.npy", np.concatenate(blobs))
        np.save(f"{out}/{split}_png_off.npy", np.array(off, np.int64))
        np.save(f"{out}/{split}_label.npy", np.array(labels, np.int64))
        if split == "train":
            np.save(f"{out}/train_rot.npy", np.stack(rots).astype(np.float32))
            print(f"[train] {len(labels)} images -> PNG blob {off[-1]/1e6:.0f} MB + train_rot {len(rots)}")
        else:
            np.save(f"{out}/test_equiv_mats.npy", np.concatenate(rots).astype(np.float32))
            np.save(f"{out}/test_equiv_off.npy", np.array(equiv_off, np.int64))
            print(f"[test] {len(labels)} images -> PNG blob {off[-1]/1e6:.0f} MB + equiv "
                  f"{equiv_off[-1]} rows (K avg {equiv_off[-1]/len(labels):.0f})")


# ------------------------------------------------------------------ stage 2: precompute path geometry

def prepare(out, level, splits=("train", "test")):
    """Read the rotations dumped by `dump` and write the per-path geometry at `level`:
    {split}_cell0.npy (int64 R), {split}_child_idx.npy (int64 R,level), {split}_child_off.npy (f32 R,level,8,3),
    {split}_parent_enc.npy (f32 R,level,9).  R = N for train, T (concat equiv) for test. Needs healpy+scipy."""
    for split in splits:
        mats = (np.load(f"{out}/train_rot.npy") if split == "train"
                else np.load(f"{out}/test_equiv_mats.npy"))
        cell0, child_idx, child_off, parent_enc = path_geometry(mats, level)
        np.save(f"{out}/{split}_cell0.npy", cell0)
        np.save(f"{out}/{split}_child_idx.npy", child_idx)
        np.save(f"{out}/{split}_child_off.npy", child_off)
        np.save(f"{out}/{split}_parent_enc.npy", parent_enc)
        print(f"[{split}] geometry L={level}: {len(cell0)} rows, "
              f"child_off {child_off.nbytes/1e6:.0f} MB, parent_enc {parent_enc.nbytes/1e6:.0f} MB")
    with open(f"{out}/meta.txt", "w") as f:
        f.write(f"dumped_level={level}\n")


# ------------------------------------------------------------------ stage 3: torch Dataset

def _import_torch():
    import torch
    from torch.utils.data import Dataset
    return torch, Dataset


def make_dataset_class():
    torch, Dataset = _import_torch()

    class SymsolDataset(Dataset):
        """Train: one item = (image, single-rotation path geometry). Test: one item = (image, the equivalent set's
        geometry, K ragged). Geometry is read precomputed (dumped at `meta.txt:dumped_level`) and sliced to `level`;
        a shallower level is an exact path prefix. Images are decoded from the PNG blob with PIL."""
        def __init__(self, root, split, level, augment=False):
            self.root, self.split, self.level = root, split, level
            # Exact in-plane roll augmentation (train only): rolling the frame by theta is an exact pose relabel
            # R -> Rz(-theta).R (optical-axis roll; convention verified empirically). The background is a uniform
            # solid and the object stays inside the inscribed circle, so a rotated frame is a genuine valid render.
            # Keeps the single-GT-rotation rule: one rotation per image, just relabelled -- no equivalent set is used.
            self.augment = bool(augment) and split == "train"
            try:
                dl = int(open(f"{root}/meta.txt").read().split("=")[1])
                assert level <= dl, f"requested level {level} > dumped level {dl}; re-run prepare"
            except FileNotFoundError:
                pass
            self.png = np.load(f"{root}/{split}_png.npy", mmap_mode="r")
            self.off = np.load(f"{root}/{split}_png_off.npy")
            self.label = np.load(f"{root}/{split}_label.npy")
            self.cell0 = np.load(f"{root}/{split}_cell0.npy", mmap_mode="r")
            self.child_idx = np.load(f"{root}/{split}_child_idx.npy", mmap_mode="r")
            self.child_off = np.load(f"{root}/{split}_child_off.npy", mmap_mode="r")
            self.parent_enc = np.load(f"{root}/{split}_parent_enc.npy", mmap_mode="r")
            if split == "train":
                self.rot = np.load(f"{root}/train_rot.npy")       # (N,3,3) single GT, for on-the-fly roll relabel
            else:
                self.equiv_off = np.load(f"{root}/test_equiv_off.npy")
                self.equiv_mats = np.load(f"{root}/test_equiv_mats.npy", mmap_mode="r")

        def __len__(self):
            return len(self.label)

        def _raw(self, i):
            import io
            from PIL import Image
            b = bytes(self.png[self.off[i]:self.off[i + 1]])
            return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.uint8)   # (H,W,3)

        def _norm(self, arr):
            x = arr.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(x.transpose(2, 0, 1).copy())

        def _img(self, i):
            return self._norm(self._raw(i))

        def _roll(self, arr, theta_deg):
            """Rotate the frame CCW by theta_deg about its centre, filling the exposed corners with the (uniform)
            background colour. Returns the rolled uint8 frame."""
            from PIL import Image
            corners = np.concatenate([arr[:8, :8].reshape(-1, 3), arr[:8, -8:].reshape(-1, 3),
                                      arr[-8:, :8].reshape(-1, 3), arr[-8:, -8:].reshape(-1, 3)])
            bg = tuple(int(v) for v in np.median(corners, 0))
            return np.asarray(Image.fromarray(arr).rotate(theta_deg, resample=Image.BILINEAR, fillcolor=bg), np.uint8)

        def equiv(self, i):
            """The i-th image's equivalent GT rotations (K,3,3) -- for the rotation-error metric."""
            a, b = int(self.equiv_off[i]), int(self.equiv_off[i + 1])
            return np.array(self.equiv_mats[a:b], np.float32)

        def __getitem__(self, i):
            L = self.level
            label = int(self.label[i])
            if self.split == "train":
                if self.augment:
                    theta = float(np.random.uniform(0.0, 360.0))
                    img = self._norm(self._roll(self._raw(i), theta))
                    Rp = rot_z(-np.deg2rad(theta)) @ self.rot[i].astype(np.float64)    # roll relabel: Rz(-theta).R
                    c0, ci, co, pe = path_geometry(Rp, L)
                    return {"image": img, "label": label,
                            "cell0": torch.as_tensor(int(c0[0])),
                            "child_idx": torch.from_numpy(ci[0]),
                            "child_off": torch.from_numpy(co[0]),
                            "parent_enc": torch.from_numpy(pe[0])}
                return {"image": self._img(i), "label": label,
                        "cell0": torch.as_tensor(int(self.cell0[i])),
                        "child_idx": torch.from_numpy(np.array(self.child_idx[i, :L])),
                        "child_off": torch.from_numpy(np.array(self.child_off[i, :L])),
                        "parent_enc": torch.from_numpy(np.array(self.parent_enc[i, :L]))}
            img = self._img(i)
            a, b = int(self.equiv_off[i]), int(self.equiv_off[i + 1])
            return {"image": img, "label": label,
                    "cell0": torch.from_numpy(np.array(self.cell0[a:b])),              # (K,)
                    "child_idx": torch.from_numpy(np.array(self.child_idx[a:b, :L])),  # (K,level)
                    "child_off": torch.from_numpy(np.array(self.child_off[a:b, :L])),  # (K,level,8,3)
                    "parent_enc": torch.from_numpy(np.array(self.parent_enc[a:b, :L]))}  # (K,level,9)

    return SymsolDataset


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dump"); d.add_argument("--out", required=True)
    d.add_argument("--shapes", nargs="+", default=SHAPE_NAMES)
    d.add_argument("--splits", nargs="+", default=["train", "test"])
    p = sub.add_parser("prepare"); p.add_argument("--out", required=True)
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.out, a.shapes, a.splits)
    elif a.cmd == "prepare":
        prepare(a.out, a.level, a.splits)
    else:
        ap.print_help()
