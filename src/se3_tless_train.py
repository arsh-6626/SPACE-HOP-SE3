"""
Train / evaluate the hierarchical SE(3) density head (se3_head.py) on T-LESS, per object, matching SpyroPose's
protocol so the distribution log-likelihood is comparable to their Table 5 (see docs/tless_se3_protocol.md).

Backbone: ImageNet ViT-B/16 patch tokens (same as SYMSOL-T). Head: Se3DensityHead at (L_rot, L_trans). Train = NLL
of the single annotated GT pose down its joint path on PBR crops (train_pbr), with train-time t_hat jitter so the
translation sub-tree trains. Eval = mean single-GT log-likelihood on the real test images (test_primesense, BOP19
targets), with the per-instance density constant  ln N - ln(V_SO3 * det(A)),  V_SO3 = pi^2,  det(A) = d^2 * t_hat_z.

The uniform predictor scores  -ln(pi^2 * det(A))  (per instance; mean ~2.7 over objects 1/14/25/27 on real test),
and a zero-init model starts exactly there (se3_tless_data.check), so any positive movement is real concentration.

    python src/se3_tless_data.py  index --root <tless> --out <cache> --objs 1,14,25,27 --split train_pbr
    python src/se3_tless_data.py  index --root <tless> --out <cache> --objs 1,14,25,27 --split test_primesense
    python src/se3_tless_train.py train --root <tless> --cache <cache> --obj 1 --out <ckpt_dir>
    python src/se3_tless_train.py eval  --root <tless> --cache <cache> --obj 1 --ckpt <ckpt_dir>/best.pth
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se3_tless_data as td
import se3_tree as se3
from se3_head import Se3DensityHead
from se3_train import TimmEncoder, _levels_to


def ll_offset(L_rot, L_trans, det_A):
    """Per-instance log-density offset added to log P(cell):  ln N - ln(V_SO3 * det(A))  (tensor in, tensor out)."""
    N = se3.n_joint_cells(L_rot, L_trans)
    return float(np.log(N)) - torch.log(se3.V_SO3 * det_A.double())


def build(L_rot, L_trans, backbone, img_size, device, grad_ckpt=False):
    enc = TimmEncoder(backbone, img_size, grad_ckpt)
    head = Se3DensityHead(embed_dim=enc.embed_dim, L_rot=L_rot, L_trans=L_trans)
    return enc.to(device), head.to(device)


def make_ds(a, split, t_hat, d_m, seed, augment=False, bbox_sj=0.0, bbox_cj=0.0):
    DS = td.make_dataset_class()
    return DS(a.root, a.cache, split, a.obj, a.L_rot, a.L_trans, d_m,
              t_hat=t_hat, sigma=a.sigma, crop_pad=a.crop_pad, out_px=a.img_size, seed=seed, augment=augment,
              bbox_sj=bbox_sj, bbox_cj=bbox_cj)


# ---------------------------------------------------------------- held-out val (single-GT LL, jittered t_hat) ----

@torch.no_grad()
def val_loglik(enc, head, val_ds, va_idx, device, L_rot, L_trans, bs, workers):
    """Mean single-GT LL on a held-out slice of train_pbr. Reseed the val dataset so its t_hat jitter is identical
    across calls (a stable, comparable proxy for the real-test metric)."""
    enc.eval(); head.eval()
    val_ds.rng = np.random.RandomState(12345 + val_ds.obj)                   # deterministic val jitter
    dl = DataLoader(Subset(val_ds, va_idx), batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True,
                    collate_fn=td.Collate(L_rot, L_trans))
    sll, cnt, floor = 0.0, 0, 0.0
    for b in dl:
        patches = enc(b["image"].to(device)).float()
        lp = head.path_loglik(patches, b["cell0"].to(device), _levels_to(b["levels"], device)).cpu().double()
        off = ll_offset(L_rot, L_trans, b["det_A"])
        sll += float((lp + off).sum()); cnt += len(lp)
        floor += float((-torch.log(se3.V_SO3 * b["det_A"].double())).sum())
    enc.train(); head.train()
    return sll / cnt, floor / cnt


# ---------------------------------------------------------------- test eval (real images, single GT) ----

@torch.no_grad()
def evaluate(enc, head, a, d_m, device, t_hat=None):
    enc.eval(); head.eval()
    t_hat = t_hat or a.test_that
    ds = make_ds(a, a.test_split, t_hat, d_m, seed=777)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True,
                    collate_fn=td.Collate(a.L_rot, a.L_trans))
    sll, cnt, floor = 0.0, 0, 0.0
    for b in dl:
        patches = enc(b["image"].to(device)).float()
        lp = head.path_loglik(patches, b["cell0"].to(device), _levels_to(b["levels"], device)).cpu().double()
        off = ll_offset(a.L_rot, a.L_trans, b["det_A"])
        sll += float((lp + off).sum()); cnt += len(lp)
        floor += float((-torch.log(se3.V_SO3 * b["det_A"].double())).sum())
    ll = sll / cnt
    print(f"  obj {a.obj}: test LL {ll:7.3f}  (n={cnt}, t_hat={t_hat}, uniform floor {floor/cnt:.3f})", flush=True)
    return ll


# ---------------------------------------------------------------- train ----

def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    d_m = td.diameters_m(a.root, [a.obj])[a.obj]
    enc, head = build(a.L_rot, a.L_trans, a.backbone, a.img_size, device, a.grad_ckpt)

    tht = getattr(a, "t_hat_train", "jitter")
    sj, cj = getattr(a, "bbox_sj", 0.12), getattr(a, "bbox_cj", 0.06)
    pbr_ds = make_ds(a, "train_pbr", tht, d_m, seed=a.seed, augment=a.augment, bbox_sj=sj, bbox_cj=cj)
    val_ds = make_ds(a, "train_pbr", tht, d_m, seed=a.seed, augment=False)   # val: clean anchor (no bbox jitter)
    n = len(pbr_ds)
    rng = np.random.RandomState(a.seed); perm = rng.permutation(n)
    k = max(1, int(round(a.val_frac * n)))
    va_idx, tr_idx = perm[:k], perm[k:]
    train_sub = Subset(pbr_ds, tr_idx)
    # optional PBR+real: concat the real split (train_real / train_primesense) so the depth readout adapts to the
    # real domain (synth->real gap: the net corrects depth on PBR but not on real). Real index must be in a.cache.
    real_split = getattr(a, "real_split", "") or ""
    if real_split:
        real_path = f"{a.cache}/{real_split}_obj{a.obj}.npz"
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"--real_split {real_split} but index missing: {real_path}")
        real_ds = make_ds(a, real_split, tht, d_m, seed=a.seed + 1, augment=a.augment, bbox_sj=sj, bbox_cj=cj)
        train_data = ConcatDataset([train_sub, real_ds])
        n_real = len(real_ds)
    else:
        train_data = train_sub; n_real = 0
    N = se3.n_joint_cells(a.L_rot, a.L_trans)
    print(f"obj {a.obj} d={d_m*1000:.1f}mm L=({a.L_rot},{a.L_trans}) N={N} | train_pbr {len(tr_idx)} + "
          f"{real_split or 'no-real'} {n_real} | held-out val {len(va_idx)} | t_hat={tht} | "
          f"uniform NLL(cell)=lnN={np.log(N):.3f}", flush=True)

    dl = DataLoader(train_data, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=True, persistent_workers=a.workers > 0, collate_fn=td.Collate(a.L_rot, a.L_trans))
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr_bb},
                             {"params": head.parameters(), "lr": a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_bb, a.lr], total_steps=a.steps, pct_start=0.05)
    step, t0, run, best = 0, time.time(), 0.0, -1e9
    if getattr(a, "init", None) and os.path.exists(a.init):
        ck = torch.load(a.init, map_location=device, weights_only=False)   # warm-start: weights only, fresh opt/step
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        print(f"warm-started weights from {a.init} (fresh optimizer/schedule)", flush=True)
    if getattr(a, "resume", None) and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        if "sched" in ck: sched.load_state_dict(ck["sched"])
        step = ck.get("step", 0); best = ck.get("best", -1e9)
        print(f"resumed from {a.resume} at step {step} (best val {best:.3f})", flush=True)

    def checkpoint(name):
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best": best, "args": vars(a), "step": step,
                    "L_rot": a.L_rot, "L_trans": a.L_trans, "d_m": d_m, "obj": a.obj}, f"{a.out}/{name}")

    def validate():
        nonlocal best
        ov, floor = val_loglik(enc, head, val_ds, va_idx, device, a.L_rot, a.L_trans, a.bs, a.workers)
        tag = "*" if ov > best else " "
        print(f"  [val step {step}] LL {ov:7.3f} {tag} (uniform floor {floor:.3f})", flush=True)
        if ov > best:
            best = ov; checkpoint("best.pth")

    while step < a.steps:
        for b in dl:
            batch = {"cell0": b["cell0"].to(device), "levels": _levels_to(b["levels"], device)}
            patches = enc(b["image"].to(device, non_blocking=True)).float()
            loss, logc = head(patches, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step(); sched.step()
            run += loss.item(); step += 1
            if step % a.log_every == 0:
                print(f"step {step}/{a.steps} | nll {run/a.log_every:.3f} | logcell {logc.mean().item():.2f} "
                      f"| {(time.time()-t0)/step:.2f}s/it", flush=True)
                run = 0.0
            if step % a.eval_every == 0:
                validate()
            if step % a.save_every == 0 or step == a.steps:
                checkpoint("last.pth")
            if step >= a.steps:
                break
    validate()
    if os.path.exists(f"{a.cache}/{a.test_split}_obj{a.obj}.npz"):
        print("=== final eval (real test, best.pth) ===", flush=True)
        ck = torch.load(f"{a.out}/best.pth", map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        evaluate(enc, head, a, d_m, device)
    else:
        print(f"=== test index missing ({a.cache}/{a.test_split}_obj{a.obj}.npz); run eval later ===", flush=True)


def load_eval(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    a.L_rot, a.L_trans = ck["L_rot"], ck["L_trans"]
    ta = ck["args"]; a.backbone = ta["backbone"]; a.img_size = ta["img_size"]
    enc, head = build(a.L_rot, a.L_trans, a.backbone, a.img_size, device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    evaluate(enc, head, a, ck["d_m"], device)


def _common(p):
    p.add_argument("--root", required=True, help="T-LESS BOP root")
    p.add_argument("--cache", required=True, help="index cache dir (se3_tless_data index --out)")
    p.add_argument("--obj", type=int, required=True)
    p.add_argument("--L_rot", type=int, default=5)
    p.add_argument("--L_trans", type=int, default=5)
    p.add_argument("--backbone", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--bs", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--crop_pad", type=float, default=1.4)
    p.add_argument("--test_that", default="gt", choices=["gt", "jitter", "bbox"],
                   help="test-time t_hat: 'gt'=oracle(g=0), 'jitter'=sim detector noise, 'bbox'=2D-box back-projection (real anchor)")
    p.add_argument("--test_split", default="test_primesense",
                   help="real-test split name (T-LESS: test_primesense; YCB-V: test)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    pt = sub.add_parser("train"); _common(pt)
    pt.add_argument("--out", required=True)
    pt.add_argument("--lr", type=float, default=1e-4); pt.add_argument("--lr_bb", type=float, default=1e-5)
    pt.add_argument("--steps", type=int, default=40000); pt.add_argument("--log_every", type=int, default=100)
    pt.add_argument("--save_every", type=int, default=5000); pt.add_argument("--eval_every", type=int, default=2000)
    pt.add_argument("--val_frac", type=float, default=0.04); pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--grad_ckpt", action="store_true")
    pt.add_argument("--augment", action="store_true", help="in-plane viewing-axis roll relabel on train_pbr (fixes memorization collapse)")
    pt.add_argument("--t_hat_train", default="jitter", choices=["jitter", "bbox"],
                    help="train anchor: 'jitter'=GT+noise (legacy), 'bbox'=2D-box back-projection (matches test, FastPose-style)")
    pt.add_argument("--real_split", default="",
                    help="optional real split to concat with train_pbr for domain adaptation (YCB-V: train_real; T-LESS: train_primesense)")
    pt.add_argument("--bbox_sj", type=float, default=0.12, help="bbox-anchor scale jitter (train only, bbox mode)")
    pt.add_argument("--bbox_cj", type=float, default=0.06, help="bbox-anchor centre jitter (train only, bbox mode)")
    pt.add_argument("--init", default=None, help="warm-start: load enc+head weights from this ckpt, fresh optimizer/step")
    pt.add_argument("--resume", default=None)
    pe = sub.add_parser("eval"); _common(pe)
    pe.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        load_eval(a)
    else:
        ap.print_help()
