"""
Recover camera intrinsics and the 3D landmark model of the ISRO GSAT-12R dataset.

The dataset ships 2D landmarks (cv_splits4/*.json) and poses (pose_realcam.json) but
no camera matrix or 3D keypoints. Given poses, each landmark's 3D body-frame position
is a linear least-squares triangulation across views, so we fit (fx, fy, cx, cy) by
minimising the reprojection error of that triangulation.

Pose convention found this way: q_sat_wrt_cam is (w, x, y, z) and X_cam = R @ X_body + t.

Usage:
    python src/tools/calibrate_isro.py --root /path/to/ISROGSAT12RDataset --out configs/isro_camera.json
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rs


def triangulate(K, R, t, uv):
    Ki = np.linalg.inv(K)
    X_all, err_all = [], []
    for j in range(uv.shape[1]):
        rays = np.c_[uv[:, j], np.ones(len(uv))] @ Ki.T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        P = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]       # (N,3,3) ray-orthogonal projector
        A = np.einsum('nij,njk->nik', P, R).reshape(-1, 3)
        b = -np.einsum('nij,nj->ni', P, t).reshape(-1)
        X = np.linalg.lstsq(A, b, rcond=None)[0]
        pc = R @ X + t
        pr = pc @ K.T
        err_all.append(np.linalg.norm(pr[:, :2] / pr[:, 2:] - uv[:, j], axis=1))
        X_all.append(X)
    return np.array(X_all), np.concatenate(err_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=3000)
    args = ap.parse_args()

    poses = {p['filename']: p for p in json.load(open(os.path.join(args.root, 'pose_realcam.json')))}
    lms = json.load(open(os.path.join(args.root, 'cv_splits4', 'train_landmark_f1.json')))
    rng = np.random.default_rng(0)
    lms = [lms[i] for i in rng.choice(len(lms), min(args.n, len(lms)), replace=False)]

    q = np.array([poses[l['image']]['q_sat_wrt_cam'] for l in lms])
    t = np.array([poses[l['image']]['r_sat_wrt_cam'] for l in lms])
    uv = np.array([l['landmarks'] for l in lms], dtype=np.float64)
    R = Rs.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()

    def residual(p):
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
        return np.minimum(triangulate(K, R, t, uv)[1], 20.0)

    fit = least_squares(residual, [1000.0, 1000.0, 960.0, 600.0], diff_step=1e-4)
    fx, fy, cx, cy = fit.x
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    X, err = triangulate(K, R, t, uv)
    print(f'K = {K.round(4).tolist()}')
    print(f'reprojection error px: median {np.median(err):.4f}  p99 {np.percentile(err, 99):.4f}')
    print('landmarks (body frame, m):\n', X.round(4))

    json.dump({
        'cameraMatrix': K.tolist(),
        'image_size': [1920, 1200],
        'quat_order': 'wxyz',
        'landmarks_3d': X.tolist(),
        'reproj_err_median_px': float(np.median(err)),
    }, open(args.out, 'w'), indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
