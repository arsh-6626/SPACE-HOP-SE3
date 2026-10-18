"""
SE(3) Hopf-fibred iterative refinement for SPACE-HOP (CAD-free, PnP-free).

Camera-relative SE(3) is treated as a bundle  S^2 (viewpoint)  x  Sim(2) fibre (ray u,v, roll, range):

  * the fibre is removed exactly by warping, not learned: a virtual camera C = Rz(roll)^T Delta_R(T) is pointed
    along the estimated ray and rolled by the estimated in-plane twist, and the full image is resampled through
    the homography  p_src ~ K C^T K_v^{-1} p_dst  with focal f_v = out * range / OBJ_SPAN, so the satellite appears
    centred, upright and at a fixed size. In that canonical camera the estimate's rotation is a pure swing S
    (axis in the image plane): the 2-DOF base point of the fibration.
  * the base is handled by exemplars: the training image whose canonical swing is closest is canonicalised the
    same way around its own GT pose (twist 0 by construction), so query and reference differ only by the
    residual pose error.
  * a ViT on [q, q - r] predicts the residual: relative rotation (canonical camera frame, deg; x,y = swing/tilt,
    z = roll), the GT ray offset in canonical pixels, and 1000 * log(range_gt / range_est).

    python src/hopf_se3.py dump  --out checkpoints/hopf_se3      # cache SPACE-HOP level-0 predictions (val/test)
    python src/hopf_se3.py train --out checkpoints/hopf_se3
    python src/hopf_se3.py eval  --ckpt checkpoints/hopf_se3/last.pth --split val
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rs
from torch.utils.data import DataLoader, Dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC_DIR)
sys.path.append(os.path.join(SRC_DIR, 'utils'))
from isro_data import load_camera, seed_worker, split_filenames
from refine import Refiner, PoseBank, view_rotation, to_tensor, rot_deg, mean_rotation
from train import load_config

OBJ_SPAN = 2.8  # metres mapped onto the canonical image side


# ---------------------------------------------------------------- fibration geometry (numpy float64)

def Rz(g):
    c, s = np.cos(g), np.sin(g)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def twist_angle(A):
    """A = Rz(gamma) @ S with S a swing (axis in the xy-plane); returns gamma."""
    x, y, z, w = Rs.from_matrix(A).as_quat()
    return 2 * np.arctan2(z, w)


def canonical_camera(R, T):
    """C such that, in the virtual camera C, the object sits on the optical axis with zero twist.
    Returns C and the canonical (swing-only) rotation S = C @ R."""
    D = view_rotation(T)
    C = Rz(twist_angle(D @ R)).T @ D
    return C, C @ R


def f_virtual(rng_m, out_size):
    return out_size * rng_m / OBJ_SPAN


def warp_canonical(img, K, C, rng_m, out_size):
    """Resample the full image into the canonical view of camera C (exact homography)."""
    fv = f_virtual(rng_m, out_size)
    c = (out_size - 1) / 2
    Kv_inv = np.array([[1 / fv, 0, -c / fv], [0, 1 / fv, -c / fv], [0, 0, 1.0]])
    Ks = K.copy()
    # pre-filter with an image pyramid when the warp would downsample by more than 2x (avoids aliasing)
    native_span = K[0, 0] * OBJ_SPAN / rng_m
    while native_span / out_size > 2.0:
        img = cv2.pyrDown(img)
        Ks = np.diag([0.5, 0.5, 1.0]) @ Ks
        native_span /= 2
    H = Ks @ C.T @ Kv_inv
    return cv2.warpPerspective(img, H, (out_size, out_size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class CanonBank(PoseBank):
    """PoseBank plus each image's canonical swing, for base-point (S^2) exemplar retrieval."""

    def __init__(self, root, split, K, fold=1):
        super().__init__(root, split, K, fold)
        self.S = np.array([canonical_camera(r, t)[1] for r, t in zip(self.R, self.T)])
        q = Rs.from_matrix(self.S).as_quat()
        self.qS = q / np.linalg.norm(q, axis=1, keepdims=True)

    def nearest_swing(self, S_query, exclude=None, k=1):
        dots = np.abs(self.qS @ Rs.from_matrix(S_query).as_quat())
        if exclude is not None:
            dots[exclude] = -1
        return np.argsort(-dots)[:k]


def make_pair(q_img, R_gt, T_gt, R0, T0, bank, ref, K, out_size, ref_img=None):
    """Canonical query/reference pair, the conditioning rotation, the target and what is needed to update."""
    C, S0 = canonical_camera(R0, T0)
    rho0 = np.linalg.norm(T0)
    q = warp_canonical(q_img, K, C, rho0, out_size)
    if ref is None:  # no exemplar: targets relative to the identity swing, blank reference channels
        Sr, r = np.eye(3), np.zeros_like(q)
    else:
        Cr, Sr = canonical_camera(bank.R[ref], bank.T[ref])
        if ref_img is None:
            ref_img = read_rgb(bank.paths[ref])
        r = warp_canonical(ref_img, K, Cr, np.linalg.norm(bank.T[ref]), out_size)
    qt, rt = to_tensor(q), to_tensor(r)
    x = torch.cat([qt, rt], 0)
    cond = torch.tensor(np.degrees(Rs.from_matrix(S0 @ Sr.T).as_rotvec()), dtype=torch.float32)
    target = None
    if R_gt is not None:
        A = C @ R_gt                         # GT rotation seen by the canonical camera
        Tc = C @ T_gt
        fv = f_virtual(rho0, out_size)
        target = torch.tensor([*np.degrees(Rs.from_matrix(A @ Sr.T).as_rotvec()),
                               fv * Tc[0] / Tc[2], fv * Tc[1] / Tc[2],
                               1000 * np.log(np.linalg.norm(T_gt) / rho0)], dtype=torch.float32)
    return x, cond, target, (C, Sr, rho0)


def update_pose(pred, ctx, out_size):
    C, Sr, rho0 = ctx
    fv = f_virtual(rho0, out_size)
    rho = rho0 * np.exp(pred[5] / 1000)
    d = np.array([pred[3] / fv, pred[4] / fv, 1.0])
    T = C.T @ (rho * d / np.linalg.norm(d))
    R = C.T @ Rs.from_rotvec(np.radians(pred[:3])).as_matrix() @ Sr
    return R, T


def perturb(R, T, rot_deg_, lat, depth, rng):
    """GT + noise: rotation (deg, isotropic), lateral (fraction of the object span), log-range."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    R0 = Rs.from_rotvec(axis * np.radians(abs(rng.normal(0, rot_deg_)))).as_matrix() @ R
    rho = np.linalg.norm(T)
    lat_m = rng.normal(0, lat, 2) * OBJ_SPAN
    D = view_rotation(T)
    T0 = D.T @ np.array([lat_m[0], lat_m[1], rho])
    T0 = T0 / np.linalg.norm(T0) * rho * np.exp(rng.normal(0, depth))
    return R0, T0


# ---------------------------------------------------------------- data

class TrainSet(Dataset):
    def __init__(self, bank, K, out_size, noise):
        self.bank, self.K, self.out_size, self.noise = bank, K, out_size, noise

    def __len__(self):
        return len(self.bank.paths)

    def __getitem__(self, i):
        rng, n = np.random, self.noise
        s = np.exp(rng.uniform(np.log(n['min_scale']), 0.0))  # log-uniform: coarse init down to target precision
        R0, T0 = perturb(self.bank.R[i], self.bank.T[i], n['rot_deg'] * s, n['lat'] * s, n['depth'] * s, rng)
        if n.get('no_ref'):
            ref = None
        else:
            cand = self.bank.nearest_swing(canonical_camera(R0, T0)[1], exclude=i, k=n['ref_topk'])
            ref = cand[rng.randint(len(cand))]
        x, c, y, _ = make_pair(read_rgb(self.bank.paths[i]), self.bank.R[i], self.bank.T[i], R0, T0,
                               self.bank, ref, self.K, self.out_size)
        return x, c, y, torch.tensor(s, dtype=torch.float32)


# ---------------------------------------------------------------- inference

@torch.no_grad()
def refine_items(model, bank, items, K, out_size, iters, device, k_refs=1, bs=64, no_ref=False):
    """items: (path, R_init, T_init, R_gt, T_gt). Returns errors (iters+1, N, 2) [deg, m] and final estimates."""
    est = [(it[1], it[2]) for it in items]
    imgs = [read_rgb(it[0]) for it in items]
    err = lambda: [(rot_deg(R, it[3]), np.linalg.norm(T - it[4])) for (R, T), it in zip(est, items)]
    errs = [err()]
    for _ in range(iters):
        xs, cs, ctxs, owner = [], [], [], []
        for j, (R0, T0) in enumerate(est):
            for ref in ([None] if no_ref else bank.nearest_swing(canonical_camera(R0, T0)[1], k=k_refs)):
                x, c, _, ctx = make_pair(imgs[j], None, None, R0, T0, bank, ref, K, out_size)
                xs.append(x); cs.append(c); ctxs.append(ctx); owner.append(j)
        pred = []
        for s in range(0, len(xs), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred.append(model(torch.stack(xs[s:s + bs]).to(device), torch.stack(cs[s:s + bs]).to(device)).float().cpu().numpy())
        pred = np.concatenate(pred).astype(np.float64)
        new = []
        for j in range(len(est)):
            cand = [update_pose(pred[n], ctxs[n], out_size) for n in range(len(owner)) if owner[n] == j]
            new.append((mean_rotation([c[0] for c in cand]), np.mean([c[1] for c in cand], 0)))
        est = new
        errs.append(err())
    return np.array(errs), est


def report(E, tag):
    for k in range(E.shape[0]):
        print(f"  [{tag}] iter {k}: E_R median {np.median(E[k, :, 0]):.3f} deg (mean {E[k, :, 0].mean():.3f}) | "
              f"E_T median {1000 * np.median(E[k, :, 1]):.1f} mm (mean {1000 * E[k, :, 1].mean():.1f}) | "
              f"<=0.1deg {np.mean(E[k, :, 0] <= 0.1):.3f} <=10mm {np.mean(E[k, :, 1] <= 0.01):.3f}", flush=True)


def load_items(cfg, split, init, n_max, seed=0, init_dir=None):
    """init: 'spacehop' (cached level-0 predictions) or 'noise' (GT + SPACE-HOP-level noise)."""
    K = load_camera(cfg['data']['camera_file'])[0].astype(np.float64)
    ev = PoseBank(cfg['data']['root'], split, K, cfg['data'].get('fold', 1))
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(ev.paths), min(n_max, len(ev.paths)), replace=False))
    if init == 'spacehop':
        z = np.load(os.path.join(init_dir, f'spacehop_{split}.npz'))
        assert [os.path.basename(p) for p in z['paths']] == [os.path.basename(p) for p in ev.paths], \
            'cached SPACE-HOP predictions do not match the split'
        inits = [(z['R'][i], z['T'][i]) for i in idx]
    else:
        inits = [perturb(ev.R[i], ev.T[i], 2.0, 0.01, 0.01, rng) for i in idx]
    return [(ev.paths[i], r0, t0, ev.R[i], ev.T[i]) for i, (r0, t0) in zip(idx, inits)], idx


@torch.no_grad()
def probe(model, bank, val_items, K, args, device):
    """One step at the fixed training noise. noise removed = 1 - median|pred - y| / median|identity - y|
    per fibration component: swing (base tilt), roll, ray offset, range."""
    rng = np.random.RandomState(0)
    X, C, Y = [], [], []
    for path, _, _, Rg, Tg in val_items[:200]:
        R0, T0 = perturb(Rg, Tg, args.rot_deg * args.probe_scale, args.lat * args.probe_scale,
                         args.depth * args.probe_scale, rng)
        ref = None if args.no_ref else bank.nearest_swing(canonical_camera(R0, T0)[1], k=1)[0]
        x, c, y, _ = make_pair(read_rgb(path), Rg, Tg, R0, T0, bank, ref, K, args.img_size)
        X.append(x); C.append(c); Y.append(y)
    model.eval()
    P = []
    for s in range(0, len(X), 32):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            P.append(model(torch.stack(X[s:s + 32]).to(device), torch.stack(C[s:s + 32]).to(device)).float().cpu())
    P, C, Y = torch.cat(P).numpy(), torch.stack(C).numpy(), torch.stack(Y).numpy()
    base = np.concatenate([C, np.zeros_like(Y[:, 3:])], 1)
    out = {}
    for g, cols in {'swing / base tilt (x,y)': [0, 1], 'roll (z)': [2], 'ray offset uv': [3, 4], 'range': [5]}.items():
        e_p, e_i = np.median(np.abs(P[:, cols] - Y[:, cols])), np.median(np.abs(base[:, cols] - Y[:, cols]))
        out[g] = 1 - e_p / e_i
        print(f"  probe {g:24s}: identity {e_i:.4f} -> model {e_p:.4f} | noise removed {out[g]:+.2f}", flush=True)
    model.train()
    return out


def load_weights(model, path):
    """Load refiner weights; a sincos pos_embed for another image size is regenerated, not copied."""
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    own = model.state_dict()
    drop = [k for k in sd if k.endswith('pos_embed') and k in own and sd[k].shape != own[k].shape]
    for k in drop:
        sd.pop(k)
    msg = model.load_state_dict(sd, strict=False)
    assert sorted(msg.missing_keys) == sorted(drop) and not msg.unexpected_keys, msg
    return drop


# ---------------------------------------------------------------- level-0 cache

def dump_spacehop(cfg, args):
    from torch.utils.data import DataLoader as DL
    from model import FastPoseViT
    from train import build_dataset, predict_pose
    device = torch.device('cuda')
    model = FastPoseViT(jepa_path=None, img_size=cfg['data'].get('crop_size', 224), patch_size=16,
                        backbone=cfg['model'].get('backbone', 'jepa')).to(device)
    model.load_state_dict(torch.load(args.spacehop_ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    for split in ('val', 'test'):
        ds = build_dataset(cfg, split, train=False)
        K = torch.as_tensor(ds.K, dtype=torch.float32, device=device).unsqueeze(0)
        Rs_, Ts_ = [], []
        with torch.no_grad():
            for b in DL(ds, batch_size=64, num_workers=8):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(b['pixel_values'].to(device))
                out = {k: v.float() for k, v in out.items()}
                R, T = predict_pose(out, b['crop_params'].to(device), K.expand(len(b['idx']), 3, 3), tuple(ds.full_size))
                Rs_.append(R.cpu().double().numpy()); Ts_.append(T.cpu().double().numpy())
        R, T = np.concatenate(Rs_), np.concatenate(Ts_)
        U, _, Vt = np.linalg.svd(R)  # re-orthonormalise the fp32 predictions
        R = U @ Vt
        paths = [it[0] for it in ds.items]
        np.savez(os.path.join(args.out, f'spacehop_{split}.npz'), R=R, T=T, paths=np.array(paths))
        e_r = [rot_deg(a, it[1].astype(np.float64)) for a, it in zip(R, ds.items)]
        e_t = [np.linalg.norm(t - it[2]) for t, it in zip(T, ds.items)]
        print(f"{split}: SPACE-HOP E_R median {np.median(e_r):.3f} deg | E_T median {1000 * np.median(e_t):.1f} mm", flush=True)


def eval_cascade(cfg, args, bank, K, device):
    """Run refiners in sequence (e.g. coarse then fine), each for its own number of iterations, from the cached
    level-0 predictions in --init_dir. Reports every iteration of every stage."""
    items, _ = load_items(cfg, args.split, 'spacehop', args.n_eval, init_dir=args.init_dir)
    os.makedirs(args.out, exist_ok=True)
    stages = []
    for st in args.stages:
        f = st.split(':')
        # default image size: the one the refiner was trained at
        a = torch.load(f[0], map_location='cpu', weights_only=False)['args']
        size = int(f[2]) if len(f) > 2 else a['img_size']
        stages.append((f[0], int(f[1]), size, bool(a.get('no_ref', False))))
    res = {}
    for k in args.k_refs:
        cur, E_all = list(items), []
        for n, (ck, iters, size, no_ref) in enumerate(stages):
            model = Refiner(size, None, arch=args.arch, pool=args.pool).to(device)
            load_weights(model, ck)
            model.eval()
            E, est = [], []
            for s in range(0, len(cur), 64):
                e, x = refine_items(model, bank, cur[s:s + 64], K, size, iters, device, k_refs=k, no_ref=no_ref)
                E.append(e); est += x
            E = np.concatenate(E, 1)
            E_all.append(E if n == 0 else E[1:])
            cur = [(it[0], R, T, it[3], it[4]) for it, (R, T) in zip(cur, est)]
            del model; torch.cuda.empty_cache()
        E = np.concatenate(E_all, 0)
        if len(cur) == len(PoseBank(cfg['data']['root'], args.split, K, cfg['data'].get('fold', 1)).paths) and k == args.k_refs[-1]:
            # the whole split, in split order: save as a level-0 cache the next stage can start from
            np.savez(os.path.join(args.out, f'spacehop_{args.split}.npz'), R=np.array([it[1] for it in cur]),
                     T=np.array([it[2] for it in cur]), paths=np.array([it[0] for it in cur]))
        report(E, f'{args.split} k={k} cascade ' + ' -> '.join(f'{os.path.basename(os.path.dirname(c))}x{i}@{z}'
                                                                for c, i, z, _ in stages))
        res[f'k{k}'] = {'stages': args.stages,
                        'E_R_median_deg': [float(np.median(e)) for e in E[:, :, 0]],
                        'E_T_median_mm': [float(1000 * np.median(e)) for e in E[:, :, 1]],
                        'frac_le_0.1deg': [float(np.mean(e <= 0.1)) for e in E[:, :, 0]],
                        'frac_le_10mm': [float(np.mean(e <= 0.01)) for e in E[:, :, 1]]}
    out = args.out
    json.dump(res, open(os.path.join(out, f'cascade_{args.split}.json'), 'w'), indent=2)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dump', 'train', 'eval', 'selftest'])
    ap.add_argument('--config', default=os.path.join(REPO, 'configs', 'isro.yaml'))
    ap.add_argument('--out', default=os.path.join(REPO, 'checkpoints', 'hopf_se3'))
    ap.add_argument('--spacehop_ckpt', default=os.path.join(REPO, 'checkpoints', 'pose_isro', 'best_model.pth'))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--img_size', type=int, default=320)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--rot_deg', type=float, default=3.0)
    ap.add_argument('--lat', type=float, default=0.01, help='lateral noise, fraction of OBJ_SPAN')
    ap.add_argument('--depth', type=float, default=0.015)
    ap.add_argument('--min_scale', type=float, default=0.03)
    ap.add_argument('--probe_scale', type=float, default=0.5, help='probe noise as a fraction of the training noise')
    ap.add_argument('--probe_every', type=int, default=1000)
    ap.add_argument('--arch', default='diff', choices=['early', 'diff', 'siamese'])
    ap.add_argument('--pool', default='attn', choices=['mean', 'attn'])
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_eval', type=int, default=500)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--resume', default=None, help='checkpoint to continue training from')
    ap.add_argument('--no_ref', action='store_true', help='no retrieved exemplar: query crop + current estimate only')
    ap.add_argument('--init', default=None, help='checkpoint to take weights from (fresh optimiser and LR schedule)')
    ap.add_argument('--init_dir', default=None, help='directory with spacehop_{split}.npz (default: --out, or the ckpt dir)')
    ap.add_argument('--stages', nargs='+', default=None,
                    help='eval cascade, each ckpt:iters[:img_size], e.g. coarse/last.pth:3 fine/last.pth:3:448')
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--k_refs', type=int, nargs='+', default=[1, 4])
    args = ap.parse_args()
    cfg = load_config(args.config)
    root, fold = cfg['data']['root'], cfg['data'].get('fold', 1)
    K = load_camera(cfg['data']['camera_file'])[0].astype(np.float64)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.mode == 'dump':
        return dump_spacehop(cfg, args)

    bank = CanonBank(root, 'train', K, fold)

    if args.mode == 'selftest':
        # at the GT pose the canonical targets must be (cond, 0, 0, 0) and the update must return the GT
        val = PoseBank(root, 'val', K, fold)
        worst = 0
        for i in range(0, len(val.paths), 500):
            ref = bank.nearest_swing(canonical_camera(val.R[i], val.T[i])[1], k=1)[0]
            img = read_rgb(val.paths[i])
            _, c, y, ctx = make_pair(img, val.R[i], val.T[i], val.R[i], val.T[i], bank, ref, K, args.img_size)
            assert np.allclose(y[:3].numpy(), c.numpy(), atol=1e-3) and np.abs(y[3:].numpy()).max() < 1e-3, (c, y)
            R0, T0 = perturb(val.R[i], val.T[i], 2.0, 0.01, 0.01, np.random.RandomState(i))
            _, _, y, ctx = make_pair(img, val.R[i], val.T[i], R0, T0, bank, ref, K, args.img_size)
            R, T = update_pose(y.numpy().astype(np.float64), ctx, args.img_size)
            worst = max(worst, rot_deg(R, val.R[i]), 1000 * np.linalg.norm(T - val.T[i]))
        print(f"selftest ok: oracle update error (deg / mm, float32 targets) <= {worst:.4f}")
        return

    if args.mode == 'eval' and args.stages:
        return eval_cascade(cfg, args, bank, K, device)

    model = Refiner(args.img_size, None if args.mode == 'eval' or args.resume or args.init else cfg['model'].get('jepa_path'),
                    arch=args.arch, pool=args.pool).to(device)

    if args.mode == 'eval':
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=False)['model'])
        model.eval()
        items, _ = load_items(cfg, args.split, 'spacehop', args.n_eval,
                              init_dir=args.init_dir or os.path.dirname(os.path.abspath(args.ckpt)))
        res = {}
        for k in args.k_refs:
            E = np.concatenate([refine_items(model, bank, items[s:s + 64], K, args.img_size, args.iters, device, k_refs=k,
                                             no_ref=args.no_ref)[0]
                                for s in range(0, len(items), 64)], 1)
            report(E, f'{args.split} k={k}')
            res[f'k{k}'] = {'E_R_median_deg': [float(np.median(e)) for e in E[:, :, 0]],
                            'E_T_median_mm': [float(1000 * np.median(e)) for e in E[:, :, 1]]}
        json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f'eval_{args.split}.json'), 'w'), indent=2)
        return

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=2)
    noise = {'rot_deg': args.rot_deg, 'lat': args.lat, 'depth': args.depth, 'min_scale': args.min_scale, 'ref_topk': 3,
             'no_ref': args.no_ref}
    dl = DataLoader(TrainSet(bank, K, args.img_size, noise), batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers or cfg['data'].get('num_workers', 12), pin_memory=True, drop_last=True,
                    persistent_workers=True, worker_init_fn=seed_worker)
    val_items, _ = load_items(cfg, 'val', 'spacehop', 200, init_dir=args.init_dir or args.out)
    vit = [p for n, p in model.named_parameters() if n.startswith('vit.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('vit.')]
    opt = torch.optim.AdamW([{'params': vit, 'lr': args.lr}, {'params': rest, 'lr': args.lr * 10}], weight_decay=0.05)
    assert sum(p.numel() for g in opt.param_groups for p in g['params']) == sum(p.numel() for p in model.parameters())
    total = args.max_steps or args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10], total_steps=total, pct_start=0.05)
    unit = torch.tensor([args.rot_deg] * 3 + [args.lat * args.img_size] * 2 + [1000 * args.depth], device=device)
    step, t0, run = 0, time.time(), []
    if args.init:
        print(f"weights from {args.init} (regenerated: {load_weights(model, args.init)})", flush=True)
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        step = ck['step']
        if 'opt' in ck:
            opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        else:  # older checkpoints hold only weights: replay the (deterministic) LR schedule, Adam moments restart
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(step):
                    sched.step()
        print(f"resumed {args.resume} at step {step} (lr {opt.param_groups[0]['lr']:.2e})", flush=True)
    while step < total:
        model.train()
        for x, c, y, s in dl:
            x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(x, c)
            e = p.float() - y
            # Normalise each residual by its sample's noise level, so the small-noise samples of the log-uniform
            # schedule are not drowned out by (and do not collapse the model towards) the identity solution.
            loss = F.smooth_l1_loss(e / (unit * s.to(device)[:, None]), torch.zeros_like(e), beta=0.1)
            model.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); sched.step(); step += 1
            run.append(e.abs().detach().median(0).values.cpu().numpy())
            if step % 100 == 0:
                m = np.median(np.array(run[-100:]), 0)
                print(f"step {step}/{total} loss {loss.item():.4f} | med|err| rot {m[:3].round(3)} deg uv {m[3:5].round(2)} px "
                      f"range {m[5]:.2f} permille | gn {gn:.3f} | {time.time() - t0:.0f}s", flush=True)
            if step % args.probe_every == 0 or step == total:
                print(f"-- step {step}", flush=True)
                probe(model, bank, val_items, K, args, device)
                model.eval()
                report(refine_items(model, bank, val_items, K, args.img_size, 3, device, no_ref=args.no_ref)[0],
                       'val from init' + (' (no ref)' if args.no_ref else ' k=1'))
                model.train()
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                            'step': step, 'args': vars(args)}, os.path.join(args.out, 'last.pth'))
            if step >= total:
                break


if __name__ == '__main__':
    main()
