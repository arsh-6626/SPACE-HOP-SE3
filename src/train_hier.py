"""
Fine-tune SPACE-HOP with the coarse-to-fine Hopf head (src/hier_head.py) on top of a trained level-0 model.

    python src/train_hier.py train --init checkpoints/pose_isro/best_model.pth --out checkpoints/hier_isro
    python src/train_hier.py eval  --ckpt checkpoints/hier_isro/best.pth --splits val test

Rotation errors are reported per level in the apparent frame (what the head predicts) and for the final true
rotation, both with the predicted translation (the real metric) and with the GT translation (isolates the error
that leaks in from the viewing-direction correction Delta_R(T)). All errors are computed in float64.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train import build_dataset, load_config, translation_loss, compute_hopf_losses
from model import FastPoseViT
from hier_head import HierHopfHead, so3_exp, geodesic_deg, delta_rotation
from isro_data import seed_worker
from geometry_utils import make_pose_targets, recover_translation_components, compute_delta_rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def level0_apparent(out):
    bins = out['logits'].argmax(-1)
    b = torch.arange(bins.shape[0], device=bins.device)
    return out['anchors'][bins].float() @ so3_exp(out['deltas'][b, bins].float())


def run_batch(model, head, batch, device, K, full, amp, train, freeze=False):
    images = batch['pixel_values'].to(device, non_blocking=True)
    R_gt = batch['R_gt_true'].to(device).float()
    T_gt = batch['T_gt'].to(device).float()
    crop = batch['crop_params'].to(device).float()
    Kb = K.expand(images.shape[0], 3, 3)
    R_app_gt, U_gt = make_pose_targets(R_gt, T_gt, crop, Kb, full)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp), torch.set_grad_enabled(train and not freeze):
        out = model(images)
    out = {k: v.float() for k, v in out.items()}
    R0 = level0_apparent(out)
    with torch.autocast('cuda', enabled=False):
        Rs, losses, inside = head(out['patches'], R0, R_app_gt if train else None)
    T_pred = recover_translation_components(out['U'], crop, full, None, Kb)
    return dict(out=out, Rs=Rs, losses=losses, inside=inside, R_app_gt=R_app_gt, U_gt=U_gt,
                R_gt=R_gt, T_gt=T_gt, T_pred=T_pred)


@torch.no_grad()
def evaluate(model, head, loader, device, K, full, amp):
    model.eval(); head.eval()
    lvl, r_true, r_true_gtT, t_err = None, [], [], []
    for batch in loader:
        r = run_batch(model, head, batch, device, K, full, amp, train=False)
        errs = [geodesic_deg(R, r['R_app_gt']) for R in r['Rs']]
        lvl = [e.cpu() for e in errs] if lvl is None else [torch.cat([a, e.cpu()]) for a, e in zip(lvl, errs)]
        R_fin = r['Rs'][-1].double()
        for T, acc in ((r['T_pred'], r_true), (r['T_gt'], r_true_gtT)):
            R_true = delta_rotation(T.double()).transpose(1, 2) @ R_fin
            acc.append(geodesic_deg(R_true, r['R_gt']).cpu())
        t_err.append((r['T_pred'].double() - r['T_gt'].double()).norm(dim=1).cpu())
    r_true, r_true_gtT, t_err = torch.cat(r_true), torch.cat(r_true_gtT), torch.cat(t_err)
    m = {f'app_L{i}_med_deg': e.median().item() for i, e in enumerate(lvl)}
    m.update({
        'E_R_median_deg': r_true.median().item(), 'E_R_mean_deg': r_true.mean().item(),
        'E_R_median_deg_gtT': r_true_gtT.median().item(),
        'E_T_median_mm': 1000 * t_err.median().item(), 'E_T_mean_mm': 1000 * t_err.mean().item(),
        'frac_R_le_0.1deg': (r_true <= 0.1).float().mean().item(),
        'frac_T_le_10mm': (t_err <= 0.01).float().mean().item(),
        'n': int(t_err.numel()),
    })
    return m, (r_true, t_err)


def fmt(m):
    lv = ' '.join(f"L{k[5]}={v:.3f}" for k, v in m.items() if k.startswith('app_L'))
    return (f"app med [{lv}] | true E_R med {m['E_R_median_deg']:.3f} (gtT {m['E_R_median_deg_gtT']:.3f}) "
            f"| E_T med {m['E_T_median_mm']:.1f} mm | <=0.1deg {m['frac_R_le_0.1deg']:.3f} <=10mm {m['frac_T_le_10mm']:.3f}")


def build(cfg, args, device):
    model = FastPoseViT(jepa_path=None, img_size=cfg['data'].get('crop_size', 224), patch_size=16).to(device)
    head = HierHopfHead(spacings_deg=args.spacings, half_width=args.half_width, sigma=args.sigma,
                        n_layers=args.layers).to(device)
    return model, head


def main_train(args):
    cfg = load_config(args.config)
    device = torch.device('cuda')
    torch.manual_seed(0); np.random.seed(0)
    model, head = build(cfg, args, device)
    init = torch.load(args.init, map_location='cpu', weights_only=False)
    model.load_state_dict(init['model_state_dict'])
    print(f"level-0 init from {args.init} (epoch {init['epoch'] + 1})")

    tr = build_dataset(cfg, 'train', train=True)
    va = build_dataset(cfg, 'val', train=False)
    K = torch.as_tensor(tr.K, dtype=torch.float32, device=device).unsqueeze(0)
    full = tuple(tr.full_size)
    nw = cfg['data'].get('num_workers', 8)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True,
                    persistent_workers=True, worker_init_fn=seed_worker)
    sub = Subset(va, np.random.RandomState(0).permutation(len(va))[:args.val_n].tolist())
    vl = DataLoader(sub, batch_size=args.bs * 2, shuffle=False, num_workers=nw, pin_memory=True)

    groups = [{'params': list(head.parameters()), 'lr': args.lr_head}]
    if args.freeze_base:
        model.requires_grad_(False)  # level 0 (backbone + original heads) stays exactly the trained model
    else:
        groups.append({'params': list(model.parameters()), 'lr': args.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = args.epochs * len(tl)
    base_lrs = [g['lr'] for g in groups]
    def set_lr(step):
        f = step / args.warmup if step < args.warmup else 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, total - args.warmup)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g['lr'] = lr * max(f, 0.01)

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'), indent=2)
    m, _ = evaluate(model, head, vl, device, K, full, args.amp)
    print(f"[val@init] {fmt(m)}", flush=True)
    best, step, t0 = float('inf'), 0, time.time()
    for ep in range(args.epochs):
        model.train(not args.freeze_base); head.train()
        for i, batch in enumerate(tl):
            set_lr(step)
            r = run_batch(model, head, batch, device, K, full, args.amp, train=True, freeze=args.freeze_base)
            o = r['out']
            l_c, l_f, acc, _ = compute_hopf_losses(o['logits'], o['deltas'], o['anchors'], r['R_app_gt'])
            l_h = sum(r['losses'])
            if args.freeze_base:
                loss = args.w_hier * l_h
            else:
                loss = translation_loss(o['U'], r['U_gt']) + l_c + 10.0 * l_f + args.w_hier * l_h
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                errs = [geodesic_deg(R, r['R_app_gt']).median().item() for R in r['Rs']]
                print(f"ep {ep + 1} step {step}/{total} loss {loss.item():.4f} | hier CE "
                      f"{[round(x.item(), 3) for x in r['losses']]} inside {[round(x.item(), 2) for x in r['inside']]} "
                      f"| train app med {[round(e, 3) for e in errs]} | acc0 {acc.item():.2f} gn {gn.item():.2f} "
                      f"| {time.time() - t0:.0f}s", flush=True)
        m, _ = evaluate(model, head, vl, device, K, full, args.amp)
        print(f"[val] ep {ep + 1} {fmt(m)}", flush=True)
        with open(os.path.join(args.out, 'history.jsonl'), 'a') as f:
            f.write(json.dumps({'epoch': ep + 1, 'step': step, **m}) + '\n')
        state = {'model': model.state_dict(), 'head': head.state_dict(), 'epoch': ep + 1, 'args': vars(args), 'val': m}
        torch.save(state, os.path.join(args.out, 'last.pth'))
        if m['E_R_median_deg'] < best:
            best = m['E_R_median_deg']
            torch.save(state, os.path.join(args.out, 'best.pth'))
            print(f"saved best (val E_R median {best:.3f})", flush=True)


def main_eval(args):
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = argparse.Namespace(**{**ck['args'], **{k: v for k, v in vars(args).items() if v is not None}})
    cfg = load_config(a.config)
    device = torch.device('cuda')
    model, head = build(cfg, a, device)
    model.load_state_dict(ck['model']); head.load_state_dict(ck['head'])
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), 'eval')
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for split in args.splits:
        ds = build_dataset(cfg, split, train=False)
        K = torch.as_tensor(ds.K, dtype=torch.float32, device=device).unsqueeze(0)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg['data'].get('num_workers', 8), pin_memory=True)
        m, (r_err, t_err) = evaluate(model, head, dl, device, K, tuple(ds.full_size), a.amp)
        res[split] = m
        print(f"[{split}] {fmt(m)}", flush=True)
        with open(os.path.join(out_dir, f'{split}_per_image.csv'), 'w') as f:
            f.write('filename,E_T_m,E_R_deg\n')
            for i, (te, re) in enumerate(zip(t_err.tolist(), r_err.tolist())):
                f.write(f'{ds.filename(i)},{te:.6f},{re:.6f}\n')
    json.dump({'ckpt': os.path.abspath(args.ckpt), 'epoch': ck['epoch'], **res},
              open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=2)
    print(f"wrote {out_dir}/metrics.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'eval'])
    ap.add_argument('--config', default=os.path.join(REPO, 'configs', 'isro.yaml'))
    ap.add_argument('--init', default=os.path.join(REPO, 'checkpoints', 'pose_isro', 'best_model.pth'))
    ap.add_argument('--out', default=os.path.join(REPO, 'checkpoints', 'hier_isro'))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--lr_head', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--w_hier', type=float, default=1.0)
    ap.add_argument('--spacings', type=float, nargs='+', default=[1.25, 0.4, 0.13])
    ap.add_argument('--half_width', type=int, default=3)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--val_n', type=int, default=1000)
    ap.add_argument('--freeze_base', action='store_true', help='train only the hierarchical head on frozen features')
    ap.add_argument('--no_amp', dest='amp', action='store_false')
    args = ap.parse_args()
    if args.mode == 'train':
        main_train(args)
    else:
        if args.ckpt is None:
            ap.error('eval needs --ckpt')
        # Only the CLI flags that eval actually uses override the saved training args.
        main_eval(argparse.Namespace(ckpt=args.ckpt, splits=args.splits, config=args.config))
