"""
Nested (coarse-to-fine) classification on the SE(3) Hopf fibration, without exemplar retrieval.

Each step re-canonicalises the image at the current estimate exactly as in hopf_se3.py: a virtual camera is pointed
along the estimated ray and rolled by the estimated twist, so the satellite appears centred, upright and at a fixed
size. A ViT sees only that query crop, the current canonical swing (the base point on S^2, 2 numbers) and the level.
It classifies the residual pose on local grids in the fibre coordinates, in units that do not depend on image size:

    tilt  (dx, dy)  2-D grid   residual rotation about the canonical x, y axes (deg)
    roll  (dz)      1-D grid   residual rotation about the line of sight (deg)
    ray   (u, v)    2-D grid   GT ray offset, per-mille of the canonical span (OBJ_SPAN)
    range (r)       1-D grid   1000 * log(range_gt / range_est)

Level l has spacing h_l = h_0 / 3^l over B = 15 bins (+-7 h_l), so every level's grid covers +-2.3 cells of its parent.
Labels are separable discretised Gaussians (sigma 0.5 bin) around the exact target; decoding takes the argmax and a
per-axis parabola through the log-probabilities, which is exact for a Gaussian-shaped prediction.

    python src/nested_se3.py selftest
    python src/nested_se3.py train --out checkpoints/nested224 --img_size 224 --levels 0 1 2 3 --init <refiner.pth>
    python src/nested_se3.py eval  --split val --init_dir checkpoints/hopf_se3_fixed \
        --schedule checkpoints/nested224/last.pth:0,0,1,2 checkpoints/nested448/last.pth:3,4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rs
from torch.utils.data import DataLoader, Dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from hopf_se3 import OBJ_SPAN, canonical_camera, warp_canonical, read_rgb, load_items, report
from isro_data import load_camera, seed_worker
from refine import Refiner, PoseBank, view_rotation, to_tensor, rot_deg
from train import load_config

B = 15                 # bins per axis
MID = B // 2
H0 = np.array([1.0, 1.0, 1.0, 7.0, 7.0, 4.0])  # level-0 spacing: deg, deg, deg, per-mille span x2, per-mille range
RATIO = 3.0
LABEL_SIGMA = 0.5      # bins
N_LEVELS = 5
GROUPS = {'tilt': [0, 1], 'roll': [2], 'ray': [3, 4], 'range': [5]}


def spacing(level):
    return H0 / RATIO ** np.asarray(level, dtype=np.float64)[..., None]


# ---------------------------------------------------------------- geometry (numpy float64)

def residual(R_gt, T_gt, R0, T0):
    """Fibre coordinates of the GT pose relative to the estimate (R0, T0), in the estimate's canonical camera."""
    C, S0 = canonical_camera(R0, T0)
    rho0 = np.linalg.norm(T0)
    d = np.degrees(Rs.from_matrix(C @ R_gt @ S0.T).as_rotvec())
    Tc = C @ T_gt
    uv = 1000 * Tc[:2] / Tc[2] * rho0 / OBJ_SPAN
    return np.array([*d, *uv, 1000 * np.log(np.linalg.norm(T_gt) / rho0)])


def apply_residual(y, R0, T0):
    """Inverse of residual(): the pose whose fibre coordinates relative to (R0, T0) are y."""
    C, S0 = canonical_camera(R0, T0)
    rho0 = np.linalg.norm(T0)
    R = C.T @ Rs.from_rotvec(np.radians(y[:3])).as_matrix() @ S0
    d = np.array([y[3] / 1000 * OBJ_SPAN / rho0, y[4] / 1000 * OBJ_SPAN / rho0, 1.0])
    T = C.T @ (rho0 * np.exp(y[5] / 1000) * d / np.linalg.norm(d))
    return R, T


def make_query(img, R0, T0, K, out_size):
    C, S0 = canonical_camera(R0, T0)
    q = to_tensor(warp_canonical(img, K, C, np.linalg.norm(T0), out_size))
    cond = torch.tensor(np.degrees(Rs.from_matrix(S0).as_rotvec()), dtype=torch.float32)  # swing: z ~ 0
    return q, cond


def perturb_level(R, T, level, rng):
    """GT + noise sized for a level: per axis N(0, 2 h) (80 %) or U(-7 h, 7 h) (20 %)."""
    h = spacing(level)
    e = np.where(rng.rand(6) < 0.8, rng.normal(0, 2 * h), rng.uniform(-MID * h, MID * h))
    # the noise is drawn in the fibre coordinates of the GT pose; residual() of the result is then close to -e
    return apply_residual(e, R, T)


# ---------------------------------------------------------------- bins (torch)

def soft_labels(t):
    """t: (N, k) targets in bin coordinates -> (N, k, B) normalised discretised Gaussians (edge-clamped)."""
    t = t.clamp(0, B - 1)
    i = torch.arange(B, device=t.device, dtype=t.dtype)
    g = torch.exp(-0.5 * ((i - t[..., None]) / LABEL_SIGMA) ** 2)
    return g / g.sum(-1, keepdim=True)


def split_logits(z):
    """z: (N, 2B^2 + 2B) -> dict of (N, B, B) / (N, B) logits."""
    a, b, c = B * B, B * B + B, 2 * B * B + B
    return {'tilt': z[:, :a].view(-1, B, B), 'roll': z[:, a:b], 'ray': z[:, b:c].view(-1, B, B), 'range': z[:, c:]}


def nested_loss(z, y, level):
    """CE against soft labels per group. y: (N, 6) physical targets, level: (N,) long. Returns total, per-group CE."""
    h = torch.as_tensor(H0, device=y.device, dtype=torch.float32)[None] / RATIO ** level.float()[:, None]
    t = y / h + MID
    L = soft_labels(t)                                  # (N, 6, B)
    lg = split_logits(z.float())
    ce = {}
    for g, cols in GROUPS.items():
        if len(cols) == 2:
            lab = L[:, cols[0], :, None] * L[:, cols[1], None, :]
            lp = F.log_softmax(lg[g].flatten(1), -1).view(-1, B, B)
            ce[g] = -(lab * lp).sum((1, 2))
        else:
            ce[g] = -(L[:, cols[0]] * F.log_softmax(lg[g], -1)).sum(-1)
    return sum(v.mean() for v in ce.values()), {g: v.detach() for g, v in ce.items()}


def _parabola(lm, l0, lp):
    den = lm - 2 * l0 + lp
    off = np.where(den < -1e-12, 0.5 * (lm - lp) / np.where(den < -1e-12, den, -1.0), 0.0)
    return np.clip(off, -0.5, 0.5)


def decode(z, level):
    """z: (N, D) logits (numpy float64), level: (N,) -> (N, 6) physical residuals (float64)."""
    lg = split_logits(torch.from_numpy(z))
    out = np.zeros((len(z), 6))
    for g, cols in GROUPS.items():
        lp = F.log_softmax(lg[g].flatten(1).double(), -1).numpy()
        if len(cols) == 2:
            lp = lp.reshape(-1, B, B)
            k = lp.reshape(len(z), -1).argmax(1)
            ix, iy = k // B, k % B
            for n in range(len(z)):
                a, b = ix[n], iy[n]
                ox = _parabola(lp[n, a - 1, b], lp[n, a, b], lp[n, a + 1, b]) if 0 < a < B - 1 else 0.0
                oy = _parabola(lp[n, a, b - 1], lp[n, a, b], lp[n, a, b + 1]) if 0 < b < B - 1 else 0.0
                out[n, cols[0]], out[n, cols[1]] = a + ox, b + oy
        else:
            k = lp.argmax(1)
            for n in range(len(z)):
                a = k[n]
                out[n, cols[0]] = a + (_parabola(lp[n, a - 1], lp[n, a], lp[n, a + 1]) if 0 < a < B - 1 else 0.0)
    return (out - MID) * spacing(level)


def labels_as_logits(y, level):
    """Log of the soft labels laid out like the network output (for the oracle self-test)."""
    yt = torch.as_tensor(y, dtype=torch.float64)
    t = yt / torch.as_tensor(spacing(level)) + MID
    L = soft_labels(t)
    parts = []
    for g, cols in GROUPS.items():
        lab = L[:, cols[0], :, None] * L[:, cols[1], None, :] if len(cols) == 2 else L[:, cols[0]]
        parts.append(torch.log(lab.flatten(1) + 1e-300))
    return torch.cat(parts, 1).numpy()


# ---------------------------------------------------------------- model

class NestedNet(nn.Module):
    """Refiner backbone (ViT-B/16, 'diff' patch embed fed [q, q]: no reference) + level/swing-conditioned bin head."""

    def __init__(self, img_size, jepa_path=None):
        super().__init__()
        self.body = Refiner(img_size, jepa_path, arch='diff', pool='attn')
        d = self.body.vit.embed_dim
        self.img_size = img_size
        # the blank reference exactly as hopf_se3 --no_ref feeds it: a black image after to_tensor normalisation
        self.register_buffer('blank', to_tensor(np.zeros((1, 1, 3), np.uint8))[None], persistent=False)
        self.cond = nn.Sequential(nn.Linear(3 + N_LEVELS, 128), nn.GELU(), nn.Linear(128, 128))
        # free bin logits (2B^2 + 2B), then a Gaussian anchor per group: 6 means (bins from the centre) + 4 log-widths
        self.head = nn.Sequential(nn.LayerNorm(2 * d + 128), nn.Linear(2 * d + 128, 1024), nn.GELU(),
                                  nn.Linear(1024, 2 * B * B + 2 * B + 10))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, q, swing, level):
        h = self.body.tokens(torch.cat([q, self.blank.to(q.dtype).expand_as(q)], 1))
        pooled = self.body.attn_pool(h[:, 1:])
        feat = torch.cat([h[:, 0], pooled], -1)
        # regression anchor: the Refiner's own head (trained with --no_ref when initialised from such a run), whose
        # raw output is the residual in (deg, deg, deg, canonical px, canonical px, per-mille); px -> per-mille span
        reg = self.body.head(torch.cat([feat, self.body.cond_embed(swing / 3.0)], -1)).float()
        # its rotation output is the absolute canonical rotation rotvec(swing) + reg[:3]; residual = log(exp(.) S0^T)
        sw = torch.deg2rad(swing.float())
        d = torch.rad2deg(so3_log(so3_exp(sw + torch.deg2rad(reg[:, :3])) @ so3_exp(sw).transpose(1, 2)))
        reg = torch.cat([d, reg[:, 3:5] * (1000.0 / self.img_size), reg[:, 5:]], 1)
        c = self.cond(torch.cat([swing / 30.0, F.one_hot(level, N_LEVELS).to(swing.dtype)], 1))
        return anchored_logits(self.head(torch.cat([feat, c], -1)).float(), level, reg)


def so3_exp(w):
    """(N, 3) rotation vectors (rad) -> (N, 3, 3), via the unit quaternion."""
    th = w.norm(dim=1, keepdim=True)
    half = th / 2
    k = torch.where(th > 1e-8, torch.sin(half) / th.clamp_min(1e-12), 0.5 - th ** 2 / 48)
    x, y, z = (w * k).unbind(1)
    c = torch.cos(half)[:, 0]
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - c * z), 2 * (x * z + c * y),
                        2 * (x * y + c * z), 1 - 2 * (x * x + z * z), 2 * (y * z - c * x),
                        2 * (x * z - c * y), 2 * (y * z + c * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)


def so3_log(R):
    """(N, 3, 3) -> (N, 3) rotation vectors (rad); angles here are small (residuals), so the axis form is stable."""
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1) / 2
    s = v.norm(dim=1, keepdim=True)
    c = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2)[:, None]
    th = torch.atan2(s, c)
    return v * torch.where(s > 1e-7, th / s.clamp_min(1e-12), 1 + th ** 2 / 6)


def anchored_logits(o, level, reg=None):
    """Gaussian-anchored bins: logits = -0.5 ((i - MID - mu) / sigma)^2 (+ the same per axis for 2-D groups) + free
    residual logits. The anchor gives the classifier a regression-strength gradient; the residual (zero-initialised)
    can still represent skewed or multi-modal (ambiguous) distributions. At init: a centred Gaussian of width 2 bins.
    The mean is one physical residual shared by all levels (in level-0 bins), so it reads mu * 3^level bins at level l;
    the widths are per level through the level conditioning."""
    D = 2 * B * B + 2 * B
    mu0 = o[:, D:D + 6] if reg is None else o[:, D:D + 6] + reg / torch.as_tensor(H0, device=o.device, dtype=o.dtype)
    free, mu, ls = o[:, :D], mu0 * RATIO ** level.float()[:, None], o[:, D + 6:]
    i = torch.arange(B, device=o.device, dtype=o.dtype)
    q = lambda col, g: -0.5 * ((i - MID - mu[:, col, None]) / (2 * torch.exp(ls[:, g, None]))) ** 2  # (N, B)
    base = torch.cat([(q(0, 0)[:, :, None] + q(1, 0)[:, None, :]).flatten(1), q(2, 1),
                      (q(3, 2)[:, :, None] + q(4, 2)[:, None, :]).flatten(1), q(5, 3)], 1)
    return base + free


def load_init(model, path):
    """Weights from a hopf_se3 Refiner checkpoint (into .body) or a NestedNet checkpoint; sincos pos_embed regenerated."""
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    if not any(k.startswith('body.') for k in sd):
        sd = {'body.' + k: v for k, v in sd.items()}
    own = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    msg = model.load_state_dict(sd, strict=False)
    return msg.missing_keys


# ---------------------------------------------------------------- data

class TrainSet(Dataset):
    def __init__(self, bank, K, out_size, levels):
        self.bank, self.K, self.out_size, self.levels = bank, K, out_size, levels

    def __len__(self):
        return len(self.bank.paths)

    def __getitem__(self, i):
        rng = np.random
        lv = int(self.levels[rng.randint(len(self.levels))])
        R, T = self.bank.R[i], self.bank.T[i]
        R0, T0 = perturb_level(R, T, lv, rng)
        q, c = make_query(read_rgb(self.bank.paths[i]), R0, T0, self.K, self.out_size)
        return q, c, torch.tensor(lv), torch.tensor(residual(R, T, R0, T0), dtype=torch.float32)


# ---------------------------------------------------------------- inference

def parse_schedule(spec):
    out = []
    for st in spec:
        ck, lv = st.rsplit(':', 1)
        out.append((ck, [int(x) for x in lv.split(',')]))
    return out


@torch.no_grad()
def run_steps(model, items, est, imgs, K, size, levels, device, bs=64):
    """One pass of `levels` (one re-canonicalised step each) over all items; returns per-step error lists."""
    errs = []
    for lv in levels:
        Z = []
        for s in range(0, len(est), bs):
            qs, cs = zip(*[make_query(imgs[j], *est[j], K, size) for j in range(s, min(s + bs, len(est)))])
            lvl = torch.full((len(qs),), lv, dtype=torch.long, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                Z.append(model(torch.stack(qs).to(device), torch.stack(cs).to(device), lvl).float().cpu().numpy())
        Y = decode(np.concatenate(Z).astype(np.float64), np.full(len(est), lv))
        est[:] = [apply_residual(y, R0, T0) for y, (R0, T0) in zip(Y, est)]
        errs.append([(rot_deg(R, it[3]), np.linalg.norm(T - it[4])) for (R, T), it in zip(est, items)])
    return errs


def residual_spread(items, est):
    """90th percentile |residual| per fibre coordinate (to compare with the next level's grid span 7 h)."""
    r = np.abs(np.array([residual(it[3], it[4], R, T) for (R, T), it in zip(est, items)]))
    return np.percentile(r, 90, axis=0)


def eval_schedule(cfg, args, K, device, schedule, items, log_spread=True, models=None):
    imgs = [read_rgb(it[0]) for it in items]
    est = [(it[1], it[2]) for it in items]
    errs = [[(rot_deg(R, it[3]), np.linalg.norm(T - it[4])) for (R, T), it in zip(est, items)]]
    tags = ['init']
    for n, (ck, levels) in enumerate(schedule):
        if models is not None:
            model, size = models[n]
        else:
            a = torch.load(ck, map_location='cpu', weights_only=False)['args']
            size = a['img_size']
            model = NestedNet(size).to(device)
            model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
        model.eval()
        for lv in levels:
            errs += run_steps(model, items, est, imgs, K, size, [lv], device)
            tags.append(f'L{lv}@{size}')
            if log_spread:
                sp = residual_spread(items, est)
                print(f"    after {tags[-1]}: p90 |residual| tilt {sp[0]:.3f} {sp[1]:.3f} roll {sp[2]:.3f} deg | "
                      f"ray {sp[3]:.2f} {sp[4]:.2f} range {sp[5]:.2f} permille", flush=True)
        if models is None:
            del model; torch.cuda.empty_cache()
    return np.array(errs), tags, est


@torch.no_grad()
def probe(model, val_items, K, args, device):
    """Per level, at that level's training noise: CE per group (vs uniform) and noise removed per coordinate."""
    rng = np.random.RandomState(0)
    imgs = {it[0]: read_rgb(it[0]) for it in val_items[:120]}
    model.eval()
    for lv in args.levels:
        Q, C, Y = [], [], []
        for path, _, _, Rg, Tg in val_items[:120]:
            R0, T0 = perturb_level(Rg, Tg, lv, rng)
            q, c = make_query(imgs[path], R0, T0, K, args.img_size)
            Q.append(q); C.append(c); Y.append(residual(Rg, Tg, R0, T0))
        Q, C = torch.stack(Q).to(device), torch.stack(C).to(device)
        lvl = torch.full((len(Q),), lv, dtype=torch.long, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = torch.cat([model(Q[s:s + 40], C[s:s + 40], lvl[s:s + 40]) for s in range(0, len(Q), 40)]).float()
        Yt = torch.tensor(np.array(Y), dtype=torch.float32, device=device)
        _, ce = nested_loss(z, Yt, lvl)
        P = decode(z.double().cpu().numpy(), np.full(len(Y), lv))
        Y = np.array(Y)
        rem = 1 - np.median(np.abs(P - Y), 0) / np.median(np.abs(Y), 0)
        print(f"  probe L{lv}: CE " + ' '.join(f"{g} {ce[g].mean().item():.2f}" for g in GROUPS) +
              f" (uniform 2D {np.log(B * B):.2f} 1D {np.log(B):.2f}) | noise removed tilt {rem[0]:+.2f} {rem[1]:+.2f} "
              f"roll {rem[2]:+.2f} ray {rem[3]:+.2f} {rem[4]:+.2f} range {rem[5]:+.2f}", flush=True)
    model.train()


# ---------------------------------------------------------------- main

def selftest(cfg, K):
    """Oracle round trips: residual/apply_residual, and soft label -> decode -> update recovers GT (incl. grid edges)."""
    val = PoseBank(cfg['data']['root'], 'val', K, cfg['data'].get('fold', 1))
    rng = np.random.RandomState(0)
    worst_geo, worst_dec = 0.0, 0.0
    for i in range(0, len(val.paths), 97):
        R, T = val.R[i], val.T[i]
        for lv in range(N_LEVELS):
            R0, T0 = perturb_level(R, T, lv, rng)
            y = residual(R, T, R0, T0)
            Rr, Tr = apply_residual(y, R0, T0)
            worst_geo = max(worst_geo, rot_deg(Rr, R), 1000 * np.linalg.norm(Tr - T))
            yc = np.clip(y, -(MID - 0.6) * spacing(lv), (MID - 0.6) * spacing(lv))  # stay off the clamped edge bin
            yd = decode(labels_as_logits(yc[None], np.array([lv])), np.array([lv]))[0]
            worst_dec = max(worst_dec, np.max(np.abs(yd - yc) / spacing(lv)))
    # edge: a target beyond the grid decodes to the edge bin centre
    yd = decode(labels_as_logits(np.array([[100.0] * 6]), np.array([0])), np.array([0]))[0]
    assert np.allclose(yd, MID * spacing(0)), yd
    print(f"selftest: residual/apply round trip <= {worst_geo:.2e} (deg / mm) | "
          f"label->decode error <= {worst_dec:.2e} bins | edge clamp ok")
    assert worst_geo < 1e-4 and worst_dec < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'eval', 'selftest'])
    ap.add_argument('--config', default=os.path.join(os.path.dirname(SRC_DIR), 'configs', 'isro.yaml'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--init', default=None, help='Refiner or NestedNet checkpoint to take weights from')
    ap.add_argument('--init_dir', default=None, help='directory with spacehop_{split}.npz')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--probe_every', type=int, default=1000)
    ap.add_argument('--clip', type=float, default=1.0, help='gradient-norm clip')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--monitor', default='0,0,1,2,3', help='levels run end-to-end on 200 val images at each probe')
    ap.add_argument('--schedule', nargs='+', default=None, help='eval: ckpt:levels ..., e.g. a.pth:0,0,1,2 b.pth:3,4')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_eval', type=int, default=5000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    K = load_camera(cfg['data']['camera_file'])[0].astype(np.float64)
    root, fold = cfg['data']['root'], cfg['data'].get('fold', 1)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.mode == 'selftest':
        return selftest(cfg, K)
    device = torch.device('cuda')

    if args.mode == 'eval':
        items, _ = load_items(cfg, args.split, 'spacehop', args.n_eval, init_dir=args.init_dir)
        E, tags, _ = eval_schedule(cfg, args, K, device, parse_schedule(args.schedule), items)
        name = ' -> '.join(f"{os.path.basename(os.path.dirname(c))}[{','.join(map(str, l))}]"
                           for c, l in parse_schedule(args.schedule))
        report(E, f'{args.split} {name}')
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            json.dump({'schedule': args.schedule, 'steps': tags,
                       'E_R_median_deg': [float(np.median(e)) for e in E[:, :, 0]],
                       'E_T_median_mm': [float(1000 * np.median(e)) for e in E[:, :, 1]],
                       'frac_le_0.1deg': [float(np.mean(e <= 0.1)) for e in E[:, :, 0]],
                       'frac_le_10mm': [float(np.mean(e <= 0.01)) for e in E[:, :, 1]]},
                      open(os.path.join(args.out, f'nested_{args.split}.json'), 'w'), indent=2)
        return

    # ---- train
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=2)
    model = NestedNet(args.img_size, None if args.init else cfg['model'].get('jepa_path')).to(device)
    if args.init:
        print(f"weights from {args.init}; newly initialised: {load_init(model, args.init)}", flush=True)
    bank = PoseBank(root, 'train', K, fold)
    dl = DataLoader(TrainSet(bank, K, args.img_size, args.levels), batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True,
                    worker_init_fn=seed_worker)
    val_items, _ = load_items(cfg, 'val', 'spacehop', 200, init_dir=args.init_dir)
    # everything in the (pretrained) Refiner body, including its regression head = the anchor, at the base LR;
    # only the new bin head and level conditioning at 10x
    old = [p for n, p in model.named_parameters() if n.startswith('body.')]
    new = [p for n, p in model.named_parameters() if not n.startswith('body.')]
    opt = torch.optim.AdamW([{'params': old, 'lr': args.lr}, {'params': new, 'lr': args.lr * 10}], weight_decay=0.05)
    total = args.max_steps or args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10], total_steps=total, pct_start=0.05)
    monitor = [int(x) for x in args.monitor.split(',')]
    step, t0, run = 0, time.time(), []
    while step < total:
        model.train()
        for q, c, lv, y in dl:
            q, c, lv, y = (t.to(device, non_blocking=True) for t in (q, c, lv, y))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(q, c, lv)
            loss, ce = nested_loss(z, y, lv)
            model.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            opt.step(); sched.step(); step += 1
            run.append([ce[g].mean().item() for g in GROUPS])
            if step % 100 == 0:
                m = np.mean(run[-100:], 0)
                print(f"step {step}/{total} loss {loss.item():.3f} | CE " +
                      ' '.join(f"{g} {v:.2f}" for g, v in zip(GROUPS, m)) + f" | gn {gn:.2f} | {time.time() - t0:.0f}s",
                      flush=True)
            if step % args.probe_every == 0 or step == total:
                print(f"-- step {step}", flush=True)
                probe(model, val_items, K, args, device)
                E, tags, _ = eval_schedule(cfg, args, K, device, [(None, monitor)], val_items, log_spread=False,
                                           models=[(model, args.img_size)])
                model.train()
                for k, tg in enumerate(tags):
                    print(f"  [val200 {tg}] E_R median {np.median(E[k, :, 0]):.3f} deg | "
                          f"E_T median {1000 * np.median(E[k, :, 1]):.1f} mm | <=0.1deg {np.mean(E[k, :, 0] <= 0.1):.3f} "
                          f"<=10mm {np.mean(E[k, :, 1] <= 0.01):.3f}", flush=True)
                torch.save({'model': model.state_dict(), 'step': step, 'args': vars(args)},
                           os.path.join(args.out, 'last.pth'))
            if step >= total:
                break


if __name__ == '__main__':
    main()
