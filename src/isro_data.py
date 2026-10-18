"""
ISRO GSAT-12R dataset for SPACE-HOP.

Layout expected under `root`:
    images/train/*.jpg, images/test/*.jpg      1920x1200 renders
    pose_realcam.json                           [{filename, q_sat_wrt_cam (w,x,y,z), r_sat_wrt_cam}]
    cv_splits4/{train,val}_landmark_f{k}.json   4-fold split of images/train

Camera intrinsics and the 3D landmark model are not shipped with the dataset; they are
recovered by src/tools/calibrate_isro.py into configs/isro_camera.json.

Pose convention: X_cam = R @ X_body + t, R = quat_wxyz_to_matrix(q_sat_wrt_cam).

Crops are square, centred on the projected-landmark box, with side = crop_scale * box extent
(1.6 covers the dataset's annotated satellite bbox in >99.9% of frames). Regions outside the
image are zero-padded rather than clipped so crop pixels map exactly to image pixels.
crop_params = (x1, y1, side, side) in full-image pixels, the (bx, by, bw, bh) convention of
utils/geometry_utils.py.
"""
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from scipy.spatial.transform import Rotation as Rs
from torch.utils.data import Dataset

cv2.setNumThreads(0)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_camera(camera_file):
    with open(camera_file) as f:
        cam = json.load(f)
    K = np.array(cam['cameraMatrix'], dtype=np.float32)
    pts = np.array(cam['landmarks_3d'], dtype=np.float32)
    assert K.shape == (3, 3) and pts.ndim == 2 and pts.shape[1] == 3, camera_file
    return K, pts, tuple(cam.get('image_size', (1920, 1200)))


def split_filenames(root, split, fold=1):
    """train/val -> cv_splits4 fold k; test -> every image in images/test."""
    if split == 'test':
        return sorted(os.listdir(os.path.join(root, 'images', 'test')))
    path = os.path.join(root, 'cv_splits4', f'{split}_landmark_f{fold}.json')
    with open(path) as f:
        return sorted(e['image'] for e in json.load(f))


def project(points_3d, R, t, K):
    pc = points_3d @ R.T + t
    pi = pc @ K.T
    return pi[:, :2] / pi[:, 2:3]


def square_crop(img, x1, y1, side, out_size):
    """Crop img[y1:y1+side, x1:x1+side] (integer coords, zero-padded outside) and resize."""
    H, W = img.shape[:2]
    crop = np.zeros((side, side, 3), dtype=img.dtype)
    sx0, sy0 = max(x1, 0), max(y1, 0)
    sx1, sy1 = min(x1 + side, W), min(y1 + side, H)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y1:sy1 - y1, sx0 - x1:sx1 - x1] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if side > out_size else cv2.INTER_LINEAR
    return cv2.resize(crop, (out_size, out_size), interpolation=interp)


class IsroPoseDataset(Dataset):
    def __init__(self, root, split, camera_file, fold=1, train=False, crop_size=224,
                 crop_scale=1.6, scale_jitter=(0.9, 1.25), shift_jitter=0.1,
                 color_aug=True, return_keypoints=False, pixel_transform=None):
        self.root = root
        self.split = split
        self.train = train
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.scale_jitter = scale_jitter
        self.shift_jitter = shift_jitter
        self.return_keypoints = return_keypoints
        self.K, self.points_3d, self.full_size = load_camera(camera_file)

        img_dir = os.path.join(root, 'images', 'test' if split == 'test' else 'train')
        with open(os.path.join(root, 'pose_realcam.json')) as f:
            poses = {p['filename']: p for p in json.load(f)}
        self.items = []
        for fn in split_filenames(root, split, fold):
            p = poses[fn]
            R = Rs.from_quat(np.array(p['q_sat_wrt_cam'])[[1, 2, 3, 0]]).as_matrix().astype(np.float32)
            t = np.array(p['r_sat_wrt_cam'], dtype=np.float32)
            self.items.append((os.path.join(img_dir, fn), R, t))
        if not self.items:
            raise RuntimeError(f'no images for split={split} fold={fold} under {root}')

        if pixel_transform is not None:
            self.pixel_transform = pixel_transform
        else:
            ops = []
            if train and color_aug:
                ops += [T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8),
                        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2)]
            ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
            self.pixel_transform = T.Compose(ops)

    def __len__(self):
        return len(self.items)

    def crop_box(self, kpts):
        lo, hi = kpts.min(0), kpts.max(0)
        center = (lo + hi) / 2
        side = self.crop_scale * float((hi - lo).max())
        if self.train:
            side *= np.random.uniform(*self.scale_jitter)
            center = center + np.random.uniform(-self.shift_jitter, self.shift_jitter, 2) * side
        side = max(int(round(side)), 16)
        x1 = int(round(center[0] - side / 2))
        y1 = int(round(center[1] - side / 2))
        return x1, y1, side

    def __getitem__(self, idx):
        path, R, t = self.items[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'failed to read {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kpts = project(self.points_3d, R, t, self.K)
        x1, y1, side = self.crop_box(kpts)
        crop = square_crop(img, x1, y1, side, self.crop_size)
        pixels = self.pixel_transform(Image.fromarray(crop))

        sample = {
            'pixel_values': pixels,
            'R_gt_true': torch.from_numpy(R),
            'T_gt': torch.from_numpy(t),
            'crop_params': torch.tensor([x1, y1, side, side], dtype=torch.float32),
            'idx': idx,
        }
        if self.return_keypoints:
            sample['keypoints'] = torch.from_numpy(
                ((kpts - np.array([x1, y1])) * (self.crop_size / side)).astype(np.float32))
        return sample

    def filename(self, idx):
        return os.path.basename(self.items[idx][0])


def seed_worker(worker_id):
    """DataLoader worker_init_fn: give each worker its own numpy RNG stream (crop jitter)."""
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
