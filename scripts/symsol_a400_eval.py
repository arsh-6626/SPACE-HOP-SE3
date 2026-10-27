"""Eval-only, runs on the idle A400 (device 1) so it never touches the resident device-0 jobs.
 (1) beam-width sweep b in {1,4,8,16} on the headline L6 model -> acc@2/5/10 (fills the b-half of tab:abl-misc).
 (2) trained-at-L5 (L5_aug) vs L6_aug truncated->L5 : equiv-set LL (the trained-vs-transfer depth datapoint).
Balanced subset: NPS images/shape, same images across conditions. No timing (A400 timings are meaningless)."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/kavinder/SPACE-HOP/src")
import numpy as np, torch
import symsol_train as T
import symsol_data as sd
from hier_head import geodesic_deg

DATA = "/home/kavinder/SPACE-HOP/data/symsol_L6"
NPS = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
SH = T.SHAPE_NAMES


def load(ckpt, grid_L):
    ck = torch.load(ckpt, map_location=device, weights_only=False); ta = ck["args"]
    enc, head = T.build(grid_L, ta["backbone"], ta["img_size"], device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
    return enc, head


def balanced_idx(ds, nps):
    by = {s: [] for s in range(len(SH))}
    for i in range(len(ds)):
        by[int(ds.label[i])].append(i)
    return {s: by[s][:nps] for s in by}


print(f"=== device: {torch.cuda.get_device_name(0)} | {NPS} imgs/shape ===", flush=True)
ds6 = sd.make_dataset_class()(DATA, "test", 6)
idxs = balanced_idx(ds6, NPS)
e6, h6 = load("/home/kavinder/SPACE-HOP/checkpoints/symsol_L6_aug/best.pth", 6)

print("\n# (1) beam-width sweep (L6 model) -- tab:abl-misc b-column", flush=True)
BEAMS = [1, 4, 8, 16]
with torch.no_grad():
    # cache geodesic error per (beam): aggregate over all shapes
    agg = {b: [] for b in BEAMS}
    ico = {b: [] for b in BEAMS}                       # icosa alone (the beam-sensitive shape)
    for s in range(len(SH)):
        for i in idxs[s]:
            gt = torch.from_numpy(ds6.equiv(i)).float().to(device)      # (K,3,3)
            for b in BEAMS:
                Rm = torch.from_numpy(T.beam_map(e6, h6, ds6[i]["image"][None], 6, b, device)).float().to(device)
                err = geodesic_deg(Rm.expand(gt.shape[0], -1, -1), gt).min().item()
                agg[b].append(err);
                if SH[s] == "icosa": ico[b].append(err)
    print(f"{'b':>3} {'errMed':>7} {'acc@2':>7} {'acc@5':>7} {'acc@10':>7} {'ico_acc@10':>11}", flush=True)
    for b in BEAMS:
        e = np.array(agg[b]); ie = np.array(ico[b])
        print(f"{b:>3} {np.median(e):7.2f} {(e<=2).mean():7.3f} {(e<=5).mean():7.3f} "
              f"{(e<=10).mean():7.3f} {(ie<=10).mean():11.3f}", flush=True)

print("\n# (2) trained-at-L5 vs L6-truncated->L5 : equiv-set LL", flush=True)
ds5 = sd.make_dataset_class()(DATA, "test", 5)
idx5 = balanced_idx(ds5, NPS)


def eval_ll(enc, head, ds, read_L, idxmap):
    dens = T.DENS(read_L); per = []
    with torch.no_grad():
        for s in range(len(SH)):
            acc = []
            for i in idxmap[s]:
                it = ds[i]; K = it["cell0"].shape[0]
                p = enc(it["image"][None].to(device)).float().expand(K, -1, -1)
                lp = head.path_loglik(p, it["cell0"].to(device), it["child_idx"].to(device),
                                      it["child_off"].to(device), it["parent_enc"].to(device), read_L)
                acc.append((lp + dens).mean().item())
            per.append(float(np.mean(acc)))
    return per


hdr = f"{'model':>22} " + " ".join(f"{n:>6s}" for n in SH) + f" {'avg':>6s}"
print(hdr, flush=True)
per = eval_ll(e6, h6, ds5, 5, idx5)
print(f"{'L6_aug trunc->L5':>22} " + " ".join(f"{x:6.2f}" for x in per) + f" {np.mean(per):6.2f}", flush=True)
del e6, h6; torch.cuda.empty_cache()
e5, h5 = load("/home/kavinder/SPACE-HOP/checkpoints/symsol_L5_aug/best.pth", 5)
per = eval_ll(e5, h5, ds5, 5, idx5)
print(f"{'trained@L5 (L5_aug)':>22} " + " ".join(f"{x:6.2f}" for x in per) + f" {np.mean(per):6.2f}", flush=True)
print("DONE", flush=True)
