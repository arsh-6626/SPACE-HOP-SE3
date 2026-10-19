"""
Train / evaluate the hierarchical SO(3) density head (symsol_head.py) on SYMSOL.

Backbone: an ImageNet-pretrained timm ViT-B/16 (the standard SYMSOL init used by IPDF / Image2Sphere), giving patch
tokens. Head: SymsolDensityHead. Train = NLL of the single rendered rotation down its path. Eval reports, per shape
and overall:
  - avg log-likelihood over the equivalent set (the distribution metric; IPDF / Image2Sphere),
  - mean/median geodesic error and acc@2/5/10 deg of the beam-MAP rotation (the Hsiao / liepose metric),
with density constant log(N(L)/V_SO3) and V_SO3 = pi^2.

    python src/symsol_train.py train --data <npy_dir> --out <ckpt_dir> --level 5
    python src/symsol_train.py eval  --data <npy_dir> --ckpt <ckpt_dir>/last.pth --level 5
"""
import os, sys, time, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import so3_nested as g
import symsol_data as sd
from symsol_head import SymsolDensityHead
from hier_head import geodesic_deg

SHAPE_NAMES = sd.SHAPE_NAMES


class TimmEncoder(nn.Module):
    """ImageNet ViT-B/16 -> patch tokens (B, P, D). Register/CLS prefix tokens dropped."""
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


DENS = lambda L: float(np.log(g.n_cells(L) / g.V_SO3))     # log(N/V_SO3) density constant


def build(level, backbone, img_size, device, grad_ckpt=False):
    enc = TimmEncoder(backbone, img_size, grad_ckpt)
    head = SymsolDensityHead(embed_dim=enc.embed_dim, max_level=level)
    return enc.to(device), head.to(device)


# ---------------------------------------------------------------- beam-MAP decode (rotation metric)

@torch.no_grad()
def beam_map(enc, head, images, level, beam=8, device="cuda"):
    """Approx MAP rotation per image by top-down beam search over the tree. images (B,3,H,W) -> (B,3,3) np."""
    patches = enc(images.to(device)).float()
    B = patches.shape[0]
    logp0 = head.logp0(patches)                                   # (B,72)
    lp, nodes = torch.topk(logp0, min(beam, 72), dim=-1)          # (B,beam) logprob, (B,beam) level-0 cells
    nodes = nodes.cpu().numpy()
    for l in range(level):
        nb = nodes.shape[1]
        off_np, par_np, chf = sd.node_children_features(nodes.reshape(-1), l)       # (B*nb,8,3),(B*nb,9),(B*nb,8)
        off = torch.from_numpy(off_np).to(device).view(B, nb, 8, 3)
        par = torch.from_numpy(par_np).to(device).view(B, nb, 9)
        # score the 8 children of every beam node (flatten beam into batch)
        clp = torch.stack([head.child_logp(patches, off[:, j], par[:, j], l) for j in range(nb)], 1)  # (B,nb,8)
        total = lp[:, :, None] + clp                              # (B,nb,8)
        child_cells = chf.reshape(B, nb, 8)
        flat = total.reshape(B, -1)
        k = min(beam, flat.shape[1])
        lp, sel = torch.topk(flat, k, dim=-1)                     # (B,k)
        sel = sel.cpu().numpy()
        nodes = np.take_along_axis(child_cells.reshape(B, -1), sel, axis=1)         # (B,k)
    best = nodes[np.arange(B), lp.argmax(-1).cpu().numpy()]       # (B,) finest cell
    return g.cell_index_to_matrix(best, level)                    # (B,3,3)


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(enc, head, root, level, device, backbone, img_size, beam=8, n=None):
    enc.eval(); head.eval()
    DS = sd.make_dataset_class()
    ds = DS(root, "test", level)
    idx = range(len(ds)) if n is None else range(min(n, len(ds)))
    per = {s: {"ll": [], "err": []} for s in SHAPE_NAMES}
    dens = DENS(level)
    for i in idx:
        item = ds[i]
        img = item["image"][None].to(device)
        patches = enc(img).float()                                # (1,P,D)
        K = item["cell0"].shape[0]
        # batched log-lik over the K equivalent rotations (replicate the single image's patches)
        pr = patches.expand(K, -1, -1)
        lp = head.path_loglik(pr, item["cell0"].to(device), item["child_idx"].to(device),
                              item["child_off"].to(device), item["parent_enc"].to(device), level)
        ll = (lp + dens).mean().item()                            # avg over equiv set (IPDF metric)
        R_map = torch.from_numpy(beam_map(enc, head, item["image"][None], level, beam, device))  # (1,3,3)
        # rotation error = min geodesic to any equivalent GT (use the rendered equiv matrices directly)
        gt = torch.from_numpy(ds.equiv[i]).float()                # (K,3,3)
        err = geodesic_deg(R_map.expand(gt.shape[0], -1, -1).float(), gt).min().item()
        s = SHAPE_NAMES[item["label"]]
        per[s]["ll"].append(ll); per[s]["err"].append(err)
    rows, all_ll, all_err = [], [], []
    for s in SHAPE_NAMES:
        if not per[s]["ll"]:
            continue
        ll = np.array(per[s]["ll"]); er = np.array(per[s]["err"])
        all_ll += per[s]["ll"]; all_err += per[s]["err"]
        rows.append(f"  {s:5s} ll {ll.mean():6.2f} | errMed {np.median(er):6.2f} errMean {er.mean():6.2f} "
                    f"| acc@2 {(er<=2).mean():.3f} @5 {(er<=5).mean():.3f} @10 {(er<=10).mean():.3f}")
    all_ll = np.array(all_ll); all_err = np.array(all_err)
    rows.append(f"  ALL   ll {all_ll.mean():6.2f} | errMed {np.median(all_err):6.2f} errMean {all_err.mean():6.2f} "
                f"| acc@2 {(all_err<=2).mean():.3f} @5 {(all_err<=5).mean():.3f} @10 {(all_err<=10).mean():.3f}")
    print("\n".join(rows))
    return all_ll.mean()


# ---------------------------------------------------------------- train

def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    enc, head = build(a.level, a.backbone, a.img_size, device, a.grad_ckpt)
    DS = sd.make_dataset_class()
    dl = DataLoader(DS(a.data, "train", a.level), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr_bb},
                             {"params": head.parameters(), "lr": a.lr}], weight_decay=0.05)
    total = a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_bb, a.lr], total_steps=total, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=False)             # head needs fp32
    step, t0 = 0, time.time()
    run = 0.0
    while step < total:
        for b in dl:
            img = b["image"].to(device, non_blocking=True)
            batch = {k: b[k].to(device) for k in ("cell0", "child_idx", "child_off", "parent_enc")}
            patches = enc(img).float()
            loss, logc = head(patches, batch, a.level)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step(); sched.step()
            run += loss.item(); step += 1
            if step % a.log_every == 0:
                print(f"step {step}/{total} | nll {run / a.log_every:.3f} | logcell {logc.mean().item():.2f} "
                      f"| {(time.time() - t0) / step:.2f}s/it", flush=True)
                run = 0.0
            if step % a.save_every == 0 or step == total:
                torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "args": vars(a), "step": step},
                           f"{a.out}/last.pth")
            if step >= total:
                break
    print("=== final eval ==="); evaluate(enc, head, a.data, a.level, device, a.backbone, a.img_size, a.beam, a.n_eval)


def load_eval(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    ta = ck["args"]
    enc, head = build(a.level, ta["backbone"], ta["img_size"], device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    evaluate(enc, head, a.data, a.level, device, ta["backbone"], ta["img_size"], a.beam, a.n_eval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True); p.add_argument("--level", type=int, default=5)
        p.add_argument("--backbone", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
        p.add_argument("--img_size", type=int, default=224); p.add_argument("--beam", type=int, default=8)
        p.add_argument("--n_eval", type=int, default=None)
        if name == "train":
            p.add_argument("--out", required=True); p.add_argument("--bs", type=int, default=64)
            p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--lr_bb", type=float, default=1e-5)
            p.add_argument("--steps", type=int, default=100000); p.add_argument("--workers", type=int, default=6)
            p.add_argument("--log_every", type=int, default=100); p.add_argument("--save_every", type=int, default=5000)
            p.add_argument("--grad_ckpt", action="store_true")
        else:
            p.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        load_eval(a)
    else:
        ap.print_help()
