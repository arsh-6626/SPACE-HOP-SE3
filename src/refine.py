"""
CAD-free, PnP-free iterative pose refinement by exemplar comparison.

Given a current pose estimate (R_k, T_k):
  1. Crop the query image around the projection of T_k at near-native resolution.
     The crop side is f * OBJ_SPAN / Z_k, so the apparent object size encodes the depth error.
  2. Retrieve the training image whose apparent rotation is nearest to the estimate and crop
     it the same way around its own ground-truth pose. This "exemplar" replaces a CAD render.
  3. A ViT sees [query | reference] stacked channel-wise (early fusion, as in DeepIM) and predicts
       - the rotation of the query relative to the reference (apparent frame, axis-angle, degrees),
       - the offset of the query's projected centre from its crop centre (output px),
       - log(Z_gt / Z_k) (per-mille).
  4. Update the pose and repeat.

    python src/refine.py train --config configs/isro.yaml --out checkpoints/refine_isro
    python src/refine.py eval  --config configs/isro.yaml --ckpt checkpoints/refine_isro/last.pth --split val
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rs
from torch.utils.data import DataLoader, Dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SRC_DIR, 'utils'))
from isro_data import IMAGENET_MEAN, IMAGENET_STD, load_camera, seed_worker, split_filenames, square_crop
from model import vit_base
from train import load_config

OBJ_SPAN = 2.8  # metres; crop side = f * OBJ_SPAN / Z (landmark span is ~2.2 m at worst-case attitude)


# ---------------------------------------------------------------- geometry (numpy, single sample)

def view_rotation(T):
    """Delta_R(T): rotation taking the viewing ray through T onto the optical axis."""
    z = np.array([0.0, 0.0, 1.0])
    d = T / np.linalg.norm(T)
    axis = np.cross(d, z)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return Rs.from_rotvec(axis / s * np.arctan2(s, d @ z)).as_matrix()


def apparent(R, T):
    return view_rotation(T) @ R


def crop_for(T, K, out_size):
    """Square crop (x1, y1, side) centred on the projection of T, side = f * OBJ_SPAN / Z."""
    u = K @ T
    cx, cy = u[0] / u[2], u[1] / u[2]
    side = max(int(round(K[0, 0] * OBJ_SPAN / T[2])), 16)
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side, (cx, cy)


def rot_deg(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def perturb(R, T, rot_sigma_deg, lat_sigma, depth_sigma, rng):
    """GT + noise. lat_sigma is a fraction of the crop side, depth_sigma a log-ratio."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.radians(abs(rng.normal(0, rot_sigma_deg)))
    R0 = Rs.from_rotvec(axis * ang).as_matrix() @ R
    Z0 = T[2] * np.exp(rng.normal(0, depth_sigma))
    side = 1.0 * OBJ_SPAN / T[2]  # crop side in normalised image units (x / z)
    xn, yn = T[0] / T[2] + rng.normal(0, lat_sigma) * side, T[1] / T[2] + rng.normal(0, lat_sigma) * side
    return R0, np.array([xn * Z0, yn * Z0, Z0])


# ---------------------------------------------------------------- data

class PoseBank:
    """All images of a split with GT poses, plus the train-split exemplar index."""

    def __init__(self, root, split, K, fold=1):
        with open(os.path.join(root, 'pose_realcam.json')) as f:
            poses = {p['filename']: p for p in json.load(f)}
        img_dir = os.path.join(root, 'images', 'test' if split == 'test' else 'train')
        self.paths, self.R, self.T = [], [], []
        for fn in split_filenames(root, split, fold):
            p = poses[fn]
            self.paths.append(os.path.join(img_dir, fn))
            self.R.append(Rs.from_quat(np.array(p['q_sat_wrt_cam'])[[1, 2, 3, 0]]).as_matrix())
            self.T.append(np.array(p['r_sat_wrt_cam'], dtype=np.float64))
        self.R, self.T = np.array(self.R), np.array(self.T)
        self.Rapp = np.array([apparent(r, t) for r, t in zip(self.R, self.T)])
        q = Rs.from_matrix(self.Rapp).as_quat()
        self.qapp = q / np.linalg.norm(q, axis=1, keepdims=True)

    def nearest(self, Rapp_query, exclude=None, k=1):
        q = Rs.from_matrix(Rapp_query).as_quat()
        dots = np.abs(self.qapp @ q)
        if exclude is not None:
            dots[exclude] = -1
        return np.argsort(-dots)[:k]


def load_crop(path, x1, y1, side, out_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return cv2.cvtColor(square_crop(img, x1, y1, side, out_size), cv2.COLOR_BGR2RGB)


_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32) * 255
_STD = np.array(IMAGENET_STD, dtype=np.float32) * 255


def to_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1))


def make_pair(qpath, Rq_gt, Tq_gt, R0, T0, bank, ref_idx, K, out_size):
    """Inputs and targets for one (query, estimate, exemplar) triple."""
    x1, y1, side, (cx, cy) = crop_for(T0, K, out_size)
    q = load_crop(qpath, x1, y1, side, out_size)
    rx1, ry1, rside, _ = crop_for(bank.T[ref_idx], K, out_size)
    r = load_crop(bank.paths[ref_idx], rx1, ry1, rside, out_size)
    x = torch.cat([to_tensor(q), to_tensor(r)], 0)
    meta = np.array([x1, y1, side, cx, cy], dtype=np.float64)
    # where the current estimate sits relative to the reference (apparent frame, degrees)
    cond = torch.tensor(np.degrees(Rs.from_matrix(apparent(R0, T0) @ bank.Rapp[ref_idx].T).as_rotvec()),
                        dtype=torch.float32)
    target = None
    if Rq_gt is not None:
        R_rel = apparent(Rq_gt, Tq_gt) @ bank.Rapp[ref_idx].T
        u = K @ Tq_gt
        du = (u[0] / u[2] - cx) / side * out_size
        dv = (u[1] / u[2] - cy) / side * out_size
        target = torch.tensor([*np.degrees(Rs.from_matrix(R_rel).as_rotvec()), du, dv,
                               1000 * np.log(Tq_gt[2] / T0[2])], dtype=torch.float32)
    return x, cond, target, meta


class RefineTrainSet(Dataset):
    def __init__(self, bank, K, out_size, noise):
        self.bank, self.K, self.out_size, self.noise = bank, K, out_size, noise

    def __len__(self):
        return len(self.bank.paths)

    def __getitem__(self, i):
        rng = np.random
        n = self.noise
        # log-uniform noise scale: covers SPACE-HOP-level errors down to the target precision
        s = np.exp(rng.uniform(np.log(n['min_scale']), 0.0))
        R0, T0 = perturb(self.bank.R[i], self.bank.T[i], n['rot_deg'] * s, n['lat'] * s, n['depth'] * s, rng)
        cand = self.bank.nearest(apparent(R0, T0), exclude=i, k=n.get('ref_topk', 3))
        ref = cand[rng.randint(len(cand))]
        x, c, y, _ = make_pair(self.bank.paths[i], self.bank.R[i], self.bank.T[i], R0, T0, self.bank, ref,
                               self.K, self.out_size)
        return x, c, y


# ---------------------------------------------------------------- model

class AttnPool(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, tokens):
        return self.attn(self.q.expand(tokens.shape[0], -1, -1), tokens, tokens)[0][:, 0]


class Refiner(nn.Module):
    """arch: 'early'   [q, r] stacked as 6 channels, patch embed init 0.5*[W, W] (v1)
             'diff'    [q, q - r] stacked, patch embed init [W, W] so edge filters see the difference image
             'siamese' shared 3-channel ViT on q and r, per-token fusion [h_q, h_r, h_q - h_r] + transformer blocks
       pool: 'mean' (CLS + mean of patches) or 'attn' (CLS + attention pooling over patches)"""

    def __init__(self, img_size, jepa_path=None, arch='early', pool='mean', fuse_depth=2):
        super().__init__()
        self.arch, self.pool = arch, pool
        self.vit = vit_base(img_size=[img_size], patch_size=16)
        if jepa_path:
            ck = torch.load(jepa_path, map_location='cpu', weights_only=False)
            sd = {k.replace('module.', ''): v for k, v in ck['target_encoder'].items()}
            sd.pop('pos_embed', None)  # sincos, regenerated for img_size
            msg = self.vit.load_state_dict(sd, strict=False)
            assert msg.missing_keys == ['pos_embed'], msg.missing_keys
        d = self.vit.embed_dim
        if arch in ('early', 'diff'):
            conv = self.vit.patch_embed.proj
            new = nn.Conv2d(6, conv.out_channels, conv.kernel_size, conv.stride)
            with torch.no_grad():
                w = torch.cat([conv.weight, conv.weight], 1)
                new.weight.copy_(w * 0.5 if arch == 'early' else w)
                new.bias.copy_(conv.bias)
            self.vit.patch_embed.proj = new
        else:
            from src.models.vision_transformer import Block
            self.fuse_in = nn.Linear(3 * d, d)
            self.fuse = nn.ModuleList([Block(dim=d, num_heads=12, mlp_ratio=4.0, qkv_bias=True) for _ in range(fuse_depth)])
            self.fuse_norm = nn.LayerNorm(d)
        if pool == 'attn':
            self.attn_pool = AttnPool(d)
        self.cond_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.LayerNorm(2 * d + 128), nn.Linear(2 * d + 128, 512), nn.GELU(), nn.Linear(512, 6))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def tokens(self, x):
        q, r = x[:, :3], x[:, 3:]
        if self.arch == 'early':
            return self.vit(x)
        if self.arch == 'diff':
            return self.vit(torch.cat([q, q - r], 1))
        hq, hr = self.vit(torch.cat([q, r], 0)).chunk(2, 0)
        h = self.fuse_in(torch.cat([hq, hr, hq - hr], -1))
        for blk in self.fuse:
            h = blk(h)
        return self.fuse_norm(h)

    def forward(self, x, cond):
        """cond: rotation of the current estimate relative to the reference (deg). With a zero head the
        output reproduces the current estimate, so a refinement step starts as the identity."""
        h = self.tokens(x)
        pooled = self.attn_pool(h[:, 1:]) if self.pool == 'attn' else h[:, 1:].mean(1)
        out = self.head(torch.cat([h[:, 0], pooled, self.cond_embed(cond / 3.0)], -1))
        return torch.cat([out[:, :3] + cond, out[:, 3:]], -1)


LOSS_SCALE = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # deg, deg, deg, px, px, per-mille


# ---------------------------------------------------------------- inference step

def update_pose(pred, R_ref_app, meta, T0, K, out_size):
    x1, y1, side, cx, cy = meta
    u = cx + pred[3] * side / out_size
    v = cy + pred[4] * side / out_size
    Z = T0[2] * np.exp(pred[5] / 1000)
    T = np.array([(u - K[0, 2]) * Z / K[0, 0], (v - K[1, 2]) * Z / K[1, 1], Z])
    R_app = Rs.from_rotvec(np.radians(pred[:3])).as_matrix() @ R_ref_app
    return view_rotation(T).T @ R_app, T


def mean_rotation(Rs_list):
    q = Rs.from_matrix(np.array(Rs_list)).as_quat()
    q = q * np.sign(q @ q[0])[:, None]
    return Rs.from_quat(q.mean(0)).as_matrix()


@torch.no_grad()
def refine_batch(model, bank_ref, items, K, out_size, iters, device, amp=True, k_refs=1):
    """items: list of (path, R_init, T_init, R_gt, T_gt). Returns per-iteration errors."""
    est = [(it[1], it[2]) for it in items]
    errs = [[(rot_deg(R, it[3]), np.linalg.norm(T - it[4])) for (R, T), it in zip(est, items)]]
    for _ in range(iters):
        xs, cs, metas, refs, owner = [], [], [], [], []
        for j, ((R0, T0), it) in enumerate(zip(est, items)):
            for ref in bank_ref.nearest(apparent(R0, T0), k=k_refs):
                x, c, _, meta = make_pair(it[0], None, None, R0, T0, bank_ref, ref, K, out_size)
                xs.append(x); cs.append(c); metas.append(meta); refs.append(ref); owner.append(j)
        preds = []
        for s in range(0, len(xs), 64):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                preds.append(model(torch.stack(xs[s:s + 64]).to(device), torch.stack(cs[s:s + 64]).to(device)).float().cpu().numpy())
        pred = np.concatenate(preds)
        new_est = []
        for j, (R0, T0) in enumerate(est):
            cand = [update_pose(pred[n], bank_ref.Rapp[refs[n]], metas[n], T0, K, out_size)
                    for n in range(len(owner)) if owner[n] == j]
            new_est.append((mean_rotation([c[0] for c in cand]), np.mean([c[1] for c in cand], 0)))
        est = new_est
        errs.append([(rot_deg(R, it[3]), np.linalg.norm(T - it[4])) for (R, T), it in zip(est, items)])
    return np.array(errs), est  # (iters+1, N, 2)


def evaluate(model, cfg, args, device, split, n_max, init_noise, iters, seed=0, k_refs=1):
    K, _, _ = load_camera(cfg['data']['camera_file'])
    K = K.astype(np.float64)
    bank_train = getattr(evaluate, '_bank', None) or PoseBank(cfg['data']['root'], 'train', K, cfg['data'].get('fold', 1))
    evaluate._bank = bank_train
    bank_eval = PoseBank(cfg['data']['root'], split, K, cfg['data'].get('fold', 1))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(bank_eval.paths), min(n_max, len(bank_eval.paths)), replace=False)
    items = []
    for i in idx:
        if init_noise is None:
            R0, T0 = bank_eval.R[i], bank_eval.T[i]
        else:
            R0, T0 = perturb(bank_eval.R[i], bank_eval.T[i], *init_noise, rng)
        items.append((bank_eval.paths[i], R0, T0, bank_eval.R[i], bank_eval.T[i]))
    model.eval()
    all_errs = []
    for s in range(0, len(items), 32):
        e, _ = refine_batch(model, bank_train, items[s:s + 32], K, args.img_size, iters, device, k_refs=k_refs)
        all_errs.append(e)
    E = np.concatenate(all_errs, 1)
    for k in range(E.shape[0]):
        print(f"  [{split}] iter {k}: E_R median {np.median(E[k, :, 0]):.3f} deg (mean {E[k, :, 0].mean():.3f}) | "
              f"E_T median {1000 * np.median(E[k, :, 1]):.1f} mm (mean {1000 * E[k, :, 1].mean():.1f}) | "
              f"<=0.1deg {np.mean(E[k, :, 0] <= 0.1):.3f} <=10mm {np.mean(E[k, :, 1] <= 0.01):.3f} (k_refs={k_refs})", flush=True)
    return E


# ---------------------------------------------------------------- main

@torch.no_grad()
def probe(model, cfg, args, device, n=200, seed=0):
    """Single refinement step on val at the fixed training noise level. For each component reports
    noise removed = 1 - median|pred - target| / median|cond - target| (0 = identity, 1 = perfect)."""
    K, _, _ = load_camera(cfg['data']['camera_file']); K = K.astype(np.float64)
    bank = evaluate._bank
    val = PoseBank(cfg['data']['root'], 'val', K, cfg['data'].get('fold', 1))
    rng = np.random.RandomState(seed)
    X, C, Y = [], [], []
    for i in rng.choice(len(val.paths), n, replace=False):
        R0, T0 = perturb(val.R[i], val.T[i], args.rot_deg, args.lat, args.depth, rng)
        ref = bank.nearest(apparent(R0, T0), k=1)[0]
        x, c, y, _ = make_pair(val.paths[i], val.R[i], val.T[i], R0, T0, bank, ref, K, args.img_size)
        X.append(x); C.append(c); Y.append(y)
    model.eval()
    P = []
    for s in range(0, n, 32):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            P.append(model(torch.stack(X[s:s + 32]).to(device), torch.stack(C[s:s + 32]).to(device)).float().cpu())
    P, C, Y = torch.cat(P).numpy(), torch.stack(C).numpy(), torch.stack(Y).numpy()
    # identity baseline for translation is "no change" = 0 offset / 0 log-depth
    base = np.concatenate([C, np.zeros_like(Y[:, 3:])], 1)
    groups = {'out-of-plane rot (x,y)': [0, 1], 'in-plane rot (z)': [2], 'lateral uv': [3, 4], 'depth': [5]}
    res = {}
    for g, cols in groups.items():
        e_pred = np.median(np.abs(P[:, cols] - Y[:, cols])); e_id = np.median(np.abs(base[:, cols] - Y[:, cols]))
        res[g] = 1 - e_pred / e_id
        print(f"  probe {g:24s}: identity err {e_id:.4f} -> model err {e_pred:.4f} | noise removed {res[g]:+.2f}", flush=True)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'eval'])
    ap.add_argument('--config', default=os.path.join(os.path.dirname(SRC_DIR), 'configs', 'isro.yaml'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(SRC_DIR), 'checkpoints', 'refine_isro'))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--img_size', type=int, default=448)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_eval', type=int, default=500)
    ap.add_argument('--iters', type=int, default=4)
    # noise at scale 1 ~ SPACE-HOP error level: ~2 deg, lateral 1% of crop, depth 1%
    ap.add_argument('--rot_deg', type=float, default=3.0)
    ap.add_argument('--lat', type=float, default=0.015)
    ap.add_argument('--depth', type=float, default=0.015)
    ap.add_argument('--min_scale', type=float, default=0.01)
    ap.add_argument('--arch', default='early', choices=['early', 'diff', 'siamese'])
    ap.add_argument('--pool', default='mean', choices=['mean', 'attn'])
    ap.add_argument('--probe_every', type=int, default=0, help='run the noise-removed probe every N steps')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    init_noise = (2.0, 0.01, 0.01)  # gate: GT + SPACE-HOP-level noise

    if args.mode == 'eval':
        model = Refiner(args.img_size, arch=args.arch, pool=args.pool).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=False)['model'])
        print('init = GT (should stay ~0):'); evaluate(model, cfg, args, device, args.split, 200, None, 2)
        print('init = GT + SPACE-HOP-level noise:')
        for k in (1, 4):
            evaluate(model, cfg, args, device, args.split, args.n_eval, init_noise, args.iters, k_refs=k)
        return

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=2)
    K, _, _ = load_camera(cfg['data']['camera_file'])
    bank = PoseBank(cfg['data']['root'], 'train', K.astype(np.float64), cfg['data'].get('fold', 1))
    evaluate._bank = bank
    noise = {'rot_deg': args.rot_deg, 'lat': args.lat, 'depth': args.depth, 'min_scale': args.min_scale, 'ref_topk': 3}
    ds = RefineTrainSet(bank, K.astype(np.float64), args.img_size, noise)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=cfg['data'].get('num_workers', 12),
                    pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=seed_worker)
    model = Refiner(args.img_size, cfg['model'].get('jepa_path'), arch=args.arch, pool=args.pool).to(device)
    backbone = [p for n, p in model.named_parameters() if n.startswith('vit.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('vit.')]
    opt = torch.optim.AdamW([{'params': backbone, 'lr': args.lr}, {'params': rest, 'lr': args.lr * 10}], weight_decay=0.05)
    assert sum(p.numel() for g in opt.param_groups for p in g['params']) == sum(p.numel() for p in model.parameters())
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10], total_steps=total, pct_start=0.05)
    scale = LOSS_SCALE.to(device)
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        for x, c, y in dl:
            x, c, y = x.to(device, non_blocking=True), c.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(x, c)
            err = (p.float() - y) / scale
            loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=0.05)
            model.zero_grad(set_to_none=True); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); sched.step(); step += 1
            run.append(err.abs().detach().median(0).values.cpu().numpy())
            if step % 100 == 0:
                m = np.median(np.array(run[-100:]), 0)
                print(f"ep {ep+1} step {step}/{total} loss {loss.item():.4f} | med |err| rot {m[:3].round(3)} deg "
                      f"uv {m[3:5].round(2)} px dz {m[5]:.2f} permille | gnorm {gnorm:.3f} | {time.time()-t0:.0f}s", flush=True)
            if args.probe_every and step % args.probe_every == 0:
                print(f"-- probe at step {step}", flush=True); probe(model, cfg, args, device)
            if args.max_steps and step >= args.max_steps:
                break
        torch.save({'model': model.state_dict(), 'epoch': ep + 1, 'args': vars(args)}, os.path.join(args.out, 'last.pth'))
        print(f"== epoch {ep+1} done in {time.time()-t0:.0f}s", flush=True)
        evaluate(model, cfg, args, device, 'val', 200, init_noise, 3, k_refs=1)
        if (ep + 1) % 5 == 0:
            evaluate(model, cfg, args, device, 'val', 200, init_noise, 3, k_refs=4)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
