"""
SPACE-HOP with the nested SE(3) Hopf refiner built in: one ViT, two kinds of pass.

    pass 0      landmark-box crop -> ViT -> global Hopf grid (S^2 base x S^1 roll anchors) + in-cell twist, U
    pass l>=1   canonical warp of the full image at the current estimate -> the same ViT -> local nested Hopf grid
                at level l (tilt 2-D, roll 1-D, ray 2-D, range 1-D; spacing h0 / 3^l), exact SE(3) update

Pass 0 is FastPoseViT unchanged. The nested head is NestedNet's (src/nested_se3.py) without its own ViT: Gaussian
bins anchored on a regression branch, conditioned on the canonical swing and the level. Both passes are trained
jointly, every step: the SPACE-HOP losses on a landmark crop and the nested CE on a canonical warp of the same image
at a level-sized perturbation.

Initialisation (--init):
    nested    ViT + nested head from a NestedNet checkpoint (its 6-channel [q, q - blank] patch embed folded exactly
              into 3 channels), pass-0 heads from a SPACE-HOP checkpoint; the refinement pass starts identical to the
              trained NestedNet, pass 0 has to re-adapt its heads
    spacehop  everything of pass 0 from a SPACE-HOP checkpoint; the nested head starts fresh

    python src/spacehop_nested.py check --nested_ckpt checkpoints/nested2_224/last.pth
    python src/spacehop_nested.py train --out checkpoints/shn_a --init nested --nested_ckpt ... --levels 0 1
    python src/spacehop_nested.py eval  --ckpt checkpoints/shn_a/last.pth --split test --schedule 0,0,1
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
from PIL import Image
from torch.utils.data import DataLoader, Subset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
from nested_se3 import (B, N_LEVELS, GROUPS, NestedNet, anchored_logits, eval_schedule, make_query, nested_loss,
                        perturb_level, residual, so3_exp, so3_log)
from hopf_se3 import report
from isro_data import IsroPoseDataset, project, seed_worker, square_crop
from model import FastPoseViT
from refine import AttnPool, to_tensor
from train import build_dataset, compute_hopf_losses, load_config, make_pose_targets, predict_pose, translation_loss

HOP_HEADS = ('pose_head', 'coarse_cross_attn', 'coarse_classifier', 'fine_head')


class NestedHead(nn.Module):
    """NestedNet minus its ViT: attention pooling, regression anchor (6 outputs), level/swing-conditioned bins."""

    def __init__(self, d, img_size):
        super().__init__()
        self.img_size = img_size
        self.attn_pool = AttnPool(d)
        self.cond_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        self.reg_head = nn.Sequential(nn.LayerNorm(2 * d + 128), nn.Linear(2 * d + 128, 512), nn.GELU(),
                                      nn.Linear(512, 6))
        nn.init.zeros_(self.reg_head[-1].weight); nn.init.zeros_(self.reg_head[-1].bias)
        self.cond = nn.Sequential(nn.Linear(3 + N_LEVELS, 128), nn.GELU(), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.LayerNorm(2 * d + 128), nn.Linear(2 * d + 128, 1024), nn.GELU(),
                                  nn.Linear(1024, 2 * B * B + 2 * B + 10))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, cls, patches, swing, level):
        feat = torch.cat([cls, self.attn_pool(patches)], -1)
        reg = self.reg_head(torch.cat([feat, self.cond_embed(swing / 3.0)], -1)).float()
        sw = torch.deg2rad(swing.float())
        d = torch.rad2deg(so3_log(so3_exp(sw + torch.deg2rad(reg[:, :3])) @ so3_exp(sw).transpose(1, 2)))
        reg = torch.cat([d, reg[:, 3:5] * (1000.0 / self.img_size), reg[:, 5:]], 1)
        c = self.cond(torch.cat([swing / 30.0, F.one_hot(level, N_LEVELS).to(swing.dtype)], 1))
        return anchored_logits(self.head(torch.cat([feat, c], -1)).float(), level, reg)


class SpaceHopNested(nn.Module):
    def __init__(self, img_size=224, jepa_path=None, backbone='jepa'):
        super().__init__()
        self.hop = FastPoseViT(jepa_path=jepa_path, img_size=img_size, backbone=backbone)
        self.nest = NestedHead(768, img_size)
        self.img_size = img_size

    def tokens(self, x):
        return self.hop.vit.forward_features(x) if self.hop.timm else self.hop.vit(x)

    def forward(self, x):
        """Pass 0 (SPACE-HOP)."""
        return self.hop_heads(self.tokens(x))

    def hop_heads(self, h):
        """FastPoseViT.forward after its ViT, on given tokens."""
        m = self.hop
        cls, patches = h[:, 0:1], h[:, m.n_prefix:]
        attn, _ = m.coarse_cross_attn(query=cls, key=patches, value=patches)
        return {'U': m.pose_head(cls[:, 0])[:, :3], 'logits': m.coarse_classifier(attn[:, 0]),
                'deltas': m.fine_head(cls[:, 0]).view(-1, m.K_bins, 3), 'anchors': m.anchors}

    def refine(self, q, swing, level, h=None):
        """Pass l >= 1: nested bin logits for a canonical warp q at the current estimate."""
        h = self.tokens(q) if h is None else h
        return self.nest(h[:, 0], h[:, self.hop.n_prefix:], swing, level)


class RefineView(nn.Module):
    """Presents model.refine as forward(q, swing, level), the interface nested_se3.eval_schedule expects."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, q, swing, level):
        return self.m.refine(q, swing, level)


# ---------------------------------------------------------------- initialisation

def fold_patch_embed(w6, b6):
    """[q, q - blank] 6-channel patch embed -> exact 3-channel equivalent on q (blank is a constant image)."""
    blank = to_tensor(np.zeros((1, 1, 3), np.uint8))[:, 0, 0].double()          # (3,) normalised black
    wa, wb = w6[:, :3].double(), w6[:, 3:].double()
    return (wa + wb).to(w6.dtype), (b6.double() - (wb.sum((2, 3)) * blank).sum(1)).to(b6.dtype)


def load_nested(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    out = {}
    for k, v in sd.items():
        if k.startswith('body.vit.'):
            out['hop.vit.' + k[9:]] = v
        elif k.startswith('body.attn_pool.') or k.startswith('body.cond_embed.'):
            out['nest.' + k[5:]] = v
        elif k.startswith('body.head.'):
            out['nest.reg_head.' + k[10:]] = v
        elif k.startswith('cond.') or k.startswith('head.'):
            out['nest.' + k] = v
    w, b = fold_patch_embed(out.pop('hop.vit.patch_embed.proj.weight'), out.pop('hop.vit.patch_embed.proj.bias'))
    out['hop.vit.patch_embed.proj.weight'], out['hop.vit.patch_embed.proj.bias'] = w, b
    return _load(model, out)


def load_spacehop(model, path, heads_only=False):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    sd = {'hop.' + k: v for k, v in sd.items() if not heads_only or k.split('.')[0] in HOP_HEADS}
    return _load(model, sd)


def _load(model, sd):
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape and k != 'hop.vit.pos_embed'}
    model.load_state_dict(keep, strict=False)
    return set(keep)


def build(cfg, args):
    model = SpaceHopNested(cfg['data'].get('crop_size', 224), None, cfg['model'].get('backbone', 'jepa'))
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=False)['model'])
        return model, set()
    if args.init == 'nested':
        trained = load_nested(model, args.nested_ckpt)
        load_spacehop(model, args.spacehop_ckpt, heads_only=True)   # pass-0 heads: loaded, but for another backbone
    else:
        trained = load_spacehop(model, args.spacehop_ckpt)
    return model, trained


# ---------------------------------------------------------------- data

class JointSet(IsroPoseDataset):
    """One image read per sample: the SPACE-HOP landmark crop (pass 0) and a canonical warp at a perturbed pose."""

    def __init__(self, *a, levels=(0, 1), **kw):
        super().__init__(*a, **kw)
        self.levels = levels
        self.K64 = self.K.astype(np.float64)

    def __getitem__(self, idx):
        path, R, t = self.items[idx]
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        kpts = project(self.points_3d, R, t, self.K)
        x1, y1, side = self.crop_box(kpts)
        pixels = self.pixel_transform(Image.fromarray(square_crop(img, x1, y1, side, self.crop_size)))
        lv = int(self.levels[np.random.randint(len(self.levels))])
        R64, T64 = R.astype(np.float64), t.astype(np.float64)
        R0, T0 = perturb_level(R64, T64, lv, np.random)
        q, c = make_query(img, R0, T0, self.K64, self.crop_size)
        return {'pixel_values': pixels, 'R_gt_true': torch.from_numpy(R), 'T_gt': torch.from_numpy(t),
                'crop_params': torch.tensor([x1, y1, side, side], dtype=torch.float32),
                'q': q, 'swing': c, 'level': torch.tensor(lv),
                'y': torch.tensor(residual(R64, T64, R0, T0), dtype=torch.float32)}


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def pass0(model, ds, device, bs=64, nw=6):
    """SPACE-HOP pass over a dataset (or Subset) -> items (path, R0, T0, R_gt, T_gt) for the nested passes."""
    base = ds.dataset if isinstance(ds, Subset) else ds
    K_t = torch.as_tensor(base.K, dtype=torch.float32, device=device)[None]
    model.eval()
    items = []
    for batch in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw):
        x = batch['pixel_values'].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x)
        out = {k: v.float() for k, v in out.items()}
        R, T = predict_pose(out, batch['crop_params'].to(device), K_t.expand(len(x), 3, 3), tuple(base.full_size))
        for j, i in enumerate(batch['idx'].tolist()):
            p, Rg, Tg = base.items[i]
            items.append((p, R[j].double().cpu().numpy(), T[j].double().cpu().numpy(),
                          Rg.astype(np.float64), Tg.astype(np.float64)))
    return items


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'train', 'eval'])
    ap.add_argument('--config', default=os.path.join(REPO, 'configs', 'isro.yaml'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--init', choices=['nested', 'spacehop'], default='nested')
    ap.add_argument('--nested_ckpt', default=os.path.join(REPO, 'checkpoints', 'nested2_224', 'last.pth'))
    ap.add_argument('--spacehop_ckpt', default=os.path.join(REPO, 'checkpoints', 'pose_isro', 'best_model_weights.pth'))
    ap.add_argument('--resume', default=None, help='SpaceHopNested checkpoint (weights only)')
    ap.add_argument('--ckpt', default=None, help='eval: SpaceHopNested checkpoint')
    ap.add_argument('--levels', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--schedule', default='0,0,1', help='nested levels run after pass 0')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--fast_mult', type=float, default=10.0, help='LR multiplier for parameters not yet trained')
    ap.add_argument('--w_nested', type=float, default=1.0)
    ap.add_argument('--pass0_grad', type=float, default=1.0, help='scale of the pass-0 gradient into the shared ViT')
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warm_steps', type=int, default=2000, help='untrained heads alone on a frozen backbone')
    ap.add_argument('--warm_lr', type=float, default=1e-3)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--probe_every', type=int, default=1000)
    ap.add_argument('--n_monitor', type=int, default=200)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_eval', type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    schedule = [int(x) for x in args.schedule.split(',')]
    K64 = build_dataset(cfg, 'val', train=False).K.astype(np.float64)

    def run_eval(model, ds, tag, log_spread=False):
        items = pass0(model, ds, device, nw=args.num_workers)
        E, tags, _ = eval_schedule(cfg, args, K64, device, [(None, schedule)], items, log_spread=log_spread,
                                   models=[(RefineView(model), model.img_size)])
        tags[0] = 'pass0'
        return E, tags

    if args.mode == 'check':
        # the folded model's refinement pass must reproduce the trained NestedNet exactly
        model, _ = build(cfg, argparse.Namespace(**{**vars(args), 'init': 'nested'}))
        model = model.to(device).eval()
        nn_ref = NestedNet(model.img_size).to(device).eval()
        nn_ref.load_state_dict(torch.load(args.nested_ckpt, map_location='cpu', weights_only=False)['model'])
        ds = build_dataset(cfg, 'val', train=False)
        rng = np.random.RandomState(0)
        Q, C, L = [], [], []
        for i in range(8):
            p, R, t = ds.items[i * 50]
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            lv = i % 2
            R0, T0 = perturb_level(R.astype(np.float64), t.astype(np.float64), lv, rng)
            q, c = make_query(img, R0, T0, K64, model.img_size)
            Q.append(q); C.append(c); L.append(lv)
        Q, C, L = torch.stack(Q).to(device), torch.stack(C).to(device), torch.tensor(L, device=device)
        with torch.no_grad():
            a, b = model.refine(Q, C, L), nn_ref(Q, C, L)
        print(f"refine pass vs NestedNet (fp32): max |logit diff| {(a - b).abs().max().item():.2e} "
              f"(logit scale {b.abs().max().item():.1f})")
        sub = Subset(ds, list(range(0, len(ds), max(1, len(ds) // 200))))
        E, tags = run_eval(model, sub, 'check')
        report(E, f'check val{len(sub)} folded, no training [{args.schedule}]')
        return

    if args.mode == 'eval':
        model = SpaceHopNested(cfg['data'].get('crop_size', 224), None, cfg['model'].get('backbone', 'jepa'))
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=False)['model'])
        model = model.to(device)
        ds = build_dataset(cfg, args.split, train=False)
        if args.n_eval:
            ds = Subset(ds, list(np.sort(np.random.RandomState(0).choice(len(ds), args.n_eval, replace=False))))
        E, tags = run_eval(model, ds, args.split, log_spread=True)
        report(E, f"{args.split} {os.path.basename(os.path.dirname(args.ckpt))} pass0 -> [{args.schedule}]")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            json.dump({'ckpt': args.ckpt, 'steps': tags,
                       'E_R_median_deg': [float(np.median(e)) for e in E[:, :, 0]],
                       'E_T_median_mm': [float(1000 * np.median(e)) for e in E[:, :, 1]],
                       'frac_le_0.1deg': [float(np.mean(e <= 0.1)) for e in E[:, :, 0]],
                       'frac_le_10mm': [float(np.mean(e <= 0.01)) for e in E[:, :, 1]]},
                      open(os.path.join(args.out, f'shn_{args.split}.json'), 'w'), indent=2)
        return

    # ---- train
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=2)
    model, trained = build(cfg, args)
    model = model.to(device)
    d = cfg['data']
    train_ds = JointSet(d['root'], 'train', d['camera_file'], fold=d.get('fold', 1), train=True,
                        crop_size=d.get('crop_size', 224), crop_scale=d.get('crop_scale', 1.6),
                        scale_jitter=tuple(d.get('scale_jitter', (0.9, 1.25))), shift_jitter=d.get('shift_jitter', 0.1),
                        color_aug=d.get('color_aug', True), levels=args.levels)
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, worker_init_fn=seed_worker)
    val_ds = build_dataset(cfg, 'val', train=False)
    mon = Subset(val_ds, list(np.sort(np.random.RandomState(0).choice(len(val_ds), args.n_monitor, replace=False))))
    K_t = torch.as_tensor(train_ds.K, dtype=torch.float32, device=device)[None].expand(args.bs, 3, 3)
    full = tuple(train_ds.full_size)
    # parameters already trained for the backbone they now sit on run at the base LR, the rest at fast_mult x
    names = dict(model.named_parameters())
    slow = [p for n, p in names.items() if n in trained]
    fast = [p for n, p in names.items() if n not in trained]
    print(f"init {args.resume or args.init}: {len(slow)} tensors at lr {args.lr}, {len(fast)} at "
          f"{args.lr * args.fast_mult} ({sum(p.numel() for p in fast) / 1e6:.1f}M params)", flush=True)
    opt = torch.optim.AdamW([{'params': slow, 'lr': args.lr}, {'params': fast, 'lr': args.lr * args.fast_mult}],
                            weight_decay=0.05)
    total = args.max_steps or args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * args.fast_mult], total_steps=total,
                                                pct_start=0.05)

    def losses(b, frozen):
        """frozen: the ViT runs without grad, so only the heads learn (warm-up of the untrained heads)."""
        R_app, U_gt = make_pose_targets(b['R_gt_true'], b['T_gt'], b['crop_params'], K_t, full)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.set_grad_enabled(not frozen):
                h0, hq = model.tokens(b['pixel_values']), model.tokens(b['q'])
            if not frozen and args.pass0_grad != 1.0:
                # same forward values; the pass-0 losses reach the shared ViT with pass0_grad x their gradient
                h0 = h0 * args.pass0_grad + h0.detach() * (1.0 - args.pass0_grad)
            out = model.hop_heads(h0)
            z = model.refine(None, b['swing'], b['level'], h=hq)
        l_t = translation_loss(out['U'].float(), U_gt)
        l_c, l_f, acc, _ = compute_hopf_losses(out['logits'].float(), out['deltas'].float(), out['anchors'], R_app)
        l_n, ce = nested_loss(z, b['y'], b['level'])
        return l_t + l_c + 10.0 * l_f + args.w_nested * l_n, [l_t.item(), l_c.item(), l_f.item(), acc.item()] + \
            [ce[g].mean().item() for g in GROUPS]

    def log(tag, step, n, run, gn):
        m = np.mean(run[-100:], 0)
        print(f"{tag} {step}/{n} | pass0 trans {m[0]:.4f} coarse {m[1]:.3f} fine {m[2]:.5f} acc {m[3]:.3f} "
              f"| nested CE " + ' '.join(f"{g} {v:.2f}" for g, v in zip(GROUPS, m[4:])) +
              f" | gn {gn:.2f} | {time.time() - t0:.0f}s", flush=True)

    def monitor(step):
        E, tags = run_eval(model, mon, 'monitor')
        for k, tg in enumerate(tags):
            print(f"  [val{args.n_monitor} {tg}] E_R median {np.median(E[k, :, 0]):.3f} deg | "
                  f"E_T median {1000 * np.median(E[k, :, 1]):.1f} mm | <=0.1deg {np.mean(E[k, :, 0] <= 0.1):.3f} "
                  f"<=10mm {np.mean(E[k, :, 1] <= 0.01):.3f}", flush=True)
        torch.save({'model': model.state_dict(), 'step': step, 'args': vars(args)}, os.path.join(args.out, 'last.pth'))
        model.train()

    t0, run = time.time(), []
    # 1) warm-up: the untrained heads alone, on a frozen backbone, so their first (large) gradients cannot
    #    disturb the features the trained part relies on
    if args.warm_steps and fast:
        wopt = torch.optim.AdamW(fast, lr=args.warm_lr, weight_decay=0.05)
        wsched = torch.optim.lr_scheduler.OneCycleLR(wopt, max_lr=args.warm_lr, total_steps=args.warm_steps,
                                                     pct_start=0.05)
        step = 0
        while step < args.warm_steps:
            model.train()
            for b in dl:
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                loss, r = losses(b, frozen=True)
                wopt.zero_grad(set_to_none=True); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(fast, args.clip).item()
                wopt.step(); wsched.step(); step += 1; run.append(r)
                if step % 100 == 0:
                    log('warm', step, args.warm_steps, run, gn)
                if step >= args.warm_steps:
                    break
        del wopt
        print(f"-- after warm-up", flush=True)
        monitor(0)
    # 2) joint training of everything
    step = 0
    while step < total:
        model.train()
        for b in dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            loss, r = losses(b, frozen=False)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            opt.step(); sched.step(); step += 1; run.append(r)
            if step % 100 == 0:
                log('step', step, total, run, gn)
            if step % args.probe_every == 0 or step == total:
                monitor(step)
            if step >= total:
                break

if __name__ == '__main__':
    main()
