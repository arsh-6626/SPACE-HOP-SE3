"""
Train / evaluate the hierarchical SE(3) density head (se3_head.py) on SYMSOL-T.

Backbone: ImageNet ViT-B/16 patch tokens (same as the SO(3) model). Head: Se3DensityHead at the (L_rot, L_trans) the
geometry was prepared at. Train = NLL of the single rendered (R,t) pose down its joint path. Eval reports, per shape
and overall, the average log-likelihood over the equivalent set (same translation, the symmetric rotations) --- the
SE(3) distribution metric, directly analogous to the SYMSOL SO(3) metric, with density constant
log(N / (V_SO3 * V_box)), V_SO3 = pi^2 and V_box = (2*box_half)^3 (= 8 for the SYMSOL-T box t in [-1,1]^3).

The uniform predictor scores -log(V_SO3 * V_box) (~ -4.37 nats for the SYMSOL-T box); a zero-init model starts exactly
there (se3_head check A), so any positive movement is real concentration.

    python src/se3_data.py  prepare --root data/symsolt/train-45k ... (see se3_data)
    python src/se3_train.py train --geom <prep_dir> --out <ckpt_dir>
    python src/se3_train.py eval  --geom <prep_dir> --ckpt <ckpt_dir>/best.pth
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import se3_data as sd
import se3_tree as se3
from hier_head import geodesic_deg
from se3_head import Se3DensityHead

SHAPE_NAMES = sd.SHAPE_NAMES


class TimmEncoder(nn.Module):
    """ImageNet ViT-B/16 -> patch tokens (B,P,D). Prefix (CLS/register) tokens dropped. Same as symsol_train."""
    def __init__(self, name="vit_base_patch16_224.augreg2_in21k_ft_in1k", img_size=224, grad_ckpt=False):
        super().__init__()
        import timm
        self.vit = timm.create_model(name, pretrained=True, img_size=img_size, num_classes=0)
        self.n_prefix = self.vit.num_prefix_tokens
        self.embed_dim = self.vit.embed_dim
        if grad_ckpt:
            self.vit.set_grad_checkpointing(True)

    def forward(self, x):
        return self.vit.forward_features(x)[:, self.n_prefix:]


def dens_const(L_rot, L_trans, box_half):
    V_box = (2.0 * box_half) ** 3
    return float(np.log(se3.n_joint_cells(L_rot, L_trans) / (se3.V_SO3 * V_box)))


def uniform_floor(box_half):
    V_box = (2.0 * box_half) ** 3
    return float(-np.log(se3.V_SO3 * V_box))


def build(meta, backbone, img_size, device, grad_ckpt=False):
    enc = TimmEncoder(backbone, img_size, grad_ckpt)
    head = Se3DensityHead(embed_dim=enc.embed_dim, L_rot=meta["L_rot"], L_trans=meta["L_trans"])
    return enc.to(device), head.to(device)


def _levels_to(levels, device):
    return [{"child_off6": lv["child_off6"].to(device), "parent12": lv["parent12"].to(device),
             "target": lv["target"].to(device)} for lv in levels]


# ---------------------------------------------------------------- held-out val (single-GT LL) ----

@torch.no_grad()
def val_loglik(enc, head, out_dir, device, val_idx, dens, bs=64, workers=4):
    """Mean single-GT log-likelihood on a held-out slice of TRAIN (unbiased proxy for the equiv-set test metric,
    since poses are uniform; keeps the single-GT rule). Per shape and overall."""
    enc.eval(); head.eval()
    DS = sd.make_dataset_class()
    sub = torch.utils.data.Subset(DS(out_dir, "train"), val_idx)
    dl = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True,
                    collate_fn=sd.collate_train)
    nS = len(SHAPE_NAMES); sll = np.zeros(nS); cnt = np.zeros(nS)
    for b in dl:
        patches = enc(b["image"].to(device)).float()
        lp = head.path_loglik(patches, b["cell0"].to(device), _levels_to(b["levels"], device))
        ll = lp.cpu().numpy() + dens; lab = b["label"].numpy()
        for s in range(nS):
            m = lab == s; n = int(m.sum())
            if n:
                sll[s] += ll[m].sum(); cnt[s] += n
    per = {SHAPE_NAMES[s]: sll[s] / cnt[s] for s in range(nS) if cnt[s]}
    enc.train(); head.train()
    return sll.sum() / cnt.sum(), per


def stratified_val_split(labels, val_frac, seed):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for s in np.unique(labels):
        idx = np.where(labels == s)[0]; rng.shuffle(idx)
        k = max(1, int(round(val_frac * len(idx))))
        va.append(idx[:k]); tr.append(idx[k:])
    return np.concatenate(tr), np.concatenate(va)


# ---------------------------------------------------------------- test eval (equivalent-set LL) ----

@torch.no_grad()
def evaluate(enc, head, out_dir, device, dens, n=None):
    enc.eval(); head.eval()
    DS = sd.make_dataset_class()
    ds = DS(out_dir, "test")
    idx = range(len(ds)) if n is None else range(min(n, len(ds)))
    per = {s: [] for s in SHAPE_NAMES}
    for i in idx:
        item = ds.test_item(i)
        K = item["cell0"].shape[0]
        patches = enc(item["image"][None].to(device)).float().expand(K, -1, -1)
        lp = head.path_loglik(patches, item["cell0"].to(device), _levels_to(item["levels"], device))
        ll = (lp + dens).mean().item()                                    # avg over equiv set
        per[SHAPE_NAMES[item["label"]]].append(ll)
    rows, allll = [], []
    for s in SHAPE_NAMES:
        if not per[s]:
            continue
        a = np.array(per[s]); allll += per[s]
        rows.append(f"  {s:5s} ll {a.mean():7.3f}  (n={len(a)})")
    allll = np.array(allll)
    rows.append(f"  ALL   ll {allll.mean():7.3f}  (uniform floor {uniform_floor_from_dens(dens, head):.3f})")
    print("\n".join(rows), flush=True)
    return allll.mean()


def uniform_floor_from_dens(dens, head):
    # dens = log(N/(V_SO3 V_box)); uniform log P(cell) = -log N; floor = dens - log N = -log(V_SO3 V_box)
    N = se3.n_joint_cells(head.L_rot, head.L_trans)
    return dens - float(np.log(N))


# ---------------------------------------------------------------- train ----

def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    meta = sd.load_meta(a.geom)
    dens = dens_const(meta["L_rot"], meta["L_trans"], meta["box_half"])
    enc, head = build(meta, a.backbone, a.img_size, device, a.grad_ckpt)
    DS = sd.make_dataset_class()
    full = DS(a.geom, "train")
    tr_idx, va_idx = stratified_val_split(full.label, a.val_frac, a.seed)
    print(f"L=({meta['L_rot']},{meta['L_trans']}) N={se3.n_joint_cells(meta['L_rot'],meta['L_trans'])} "
          f"dens={dens:.3f} uniform-floor={uniform_floor(meta['box_half']):.3f} | "
          f"train {len(tr_idx)} | held-out val {len(va_idx)}", flush=True)
    dl = DataLoader(torch.utils.data.Subset(full, tr_idx), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=a.workers > 0, collate_fn=sd.collate_train)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr_bb},
                             {"params": head.parameters(), "lr": a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_bb, a.lr], total_steps=a.steps, pct_start=0.05)
    step, t0, run, best = 0, time.time(), 0.0, -1e9
    if getattr(a, "resume", None) and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        if "sched" in ck: sched.load_state_dict(ck["sched"])
        step = ck.get("step", 0); best = ck.get("best", -1e9)
        print(f"resumed from {a.resume} at step {step} (best val {best:.3f})", flush=True)

    def checkpoint(name):
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best": best, "args": vars(a), "meta": meta, "step": step},
                   f"{a.out}/{name}")

    def validate():
        nonlocal best
        ov, perd = val_loglik(enc, head, a.geom, device, va_idx, dens, bs=a.bs, workers=a.workers)
        tag = "*" if ov > best else " "
        print(f"  [val step {step}] ll {ov:7.3f} {tag} | " + " ".join(f"{k} {v:6.2f}" for k, v in perd.items()),
              flush=True)
        if ov > best:
            best = ov; checkpoint("best.pth")

    while step < a.steps:
        for b in dl:
            img = b["image"].to(device, non_blocking=True)
            batch = {"cell0": b["cell0"].to(device), "levels": _levels_to(b["levels"], device)}
            patches = enc(img).float()
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
    if os.path.exists(f"{a.geom}/test_label.npy"):
        print("=== final eval (test, best.pth) ===", flush=True)
        ck = torch.load(f"{a.out}/best.pth", map_location=device, weights_only=False)
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        evaluate(enc, head, a.geom, device, dens, a.n_eval)
    else:
        print("=== test geometry not prepared yet; run se3_train.py eval --ckpt best.pth later ===", flush=True)


def load_eval(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    meta = ck["meta"]; ta = ck["args"]
    dens = dens_const(meta["L_rot"], meta["L_trans"], meta["box_half"])
    enc, head = build(meta, ta["backbone"], ta["img_size"], device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    evaluate(enc, head, a.geom, device, dens, a.n_eval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--geom", required=True, help="prepared geometry dir (se3_data prepare --out)")
        p.add_argument("--backbone", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
        p.add_argument("--img_size", type=int, default=224)
        p.add_argument("--n_eval", type=int, default=None)
        if name == "train":
            p.add_argument("--out", required=True); p.add_argument("--bs", type=int, default=48)
            p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--lr_bb", type=float, default=1e-5)
            p.add_argument("--steps", type=int, default=60000); p.add_argument("--workers", type=int, default=4)
            p.add_argument("--log_every", type=int, default=100); p.add_argument("--save_every", type=int, default=5000)
            p.add_argument("--eval_every", type=int, default=2500)
            p.add_argument("--val_frac", type=float, default=0.04); p.add_argument("--seed", type=int, default=0)
            p.add_argument("--grad_ckpt", action="store_true")
            p.add_argument("--resume", default=None, help="checkpoint to resume enc/head/opt/sched/step from")
        else:
            p.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        load_eval(a)
    else:
        ap.print_help()
