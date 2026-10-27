"""Reconcile the depth/efficiency tables honestly:
 (1) trained-at-L5 (symsol_L5_aug) vs L6-model-truncated-to-L5 -- the real 'trained at L vs resolution-transfer' signal
     the depth table was missing; equiv-set LL, 1000 imgs/shape, same images.
 (2) per-mode peak GPU memory (reset BETWEEN likelihood and MAP) -- fixes the single-reset artifact (1857 was max-over-both).
Eval only, bs=1, device 0 (shared). Prints a compact block ready to paste."""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/kavinder/SPACE-HOP/src")
import numpy as np, torch
import symsol_train as T
import symsol_data as sd

DATA = "/home/kavinder/SPACE-HOP/data/symsol_L6"
NSHOW = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt, grid_L):
    ck = torch.load(ckpt, map_location=device, weights_only=False); ta = ck["args"]
    enc, head = T.build(grid_L, ta["backbone"], ta["img_size"], device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
    return enc, head


def eval_LL(enc, head, ds, read_L):
    by = {s: [] for s in range(len(T.SHAPE_NAMES))}
    for i in range(len(ds)):
        by[int(ds.label[i])].append(i)
    dens = T.DENS(read_L); per = []
    with torch.no_grad():
        for s in range(len(T.SHAPE_NAMES)):
            idx = by[s][:NSHOW]; acc = []
            for i in idx:
                it = ds[i]; K = it["cell0"].shape[0]
                p = enc(it["image"][None].to(device)).float().expand(K, -1, -1)
                lp = head.path_loglik(p, it["cell0"].to(device), it["child_idx"].to(device),
                                      it["child_off"].to(device), it["parent_enc"].to(device), read_L)
                acc.append((lp + dens).mean().item())
            per.append(float(np.mean(acc)))
    return per


print("# (1) trained-at-L5 vs truncated-from-L6, read at level 5 -- %d imgs/shape" % NSHOW)
hdr = f"{'model':>22} " + " ".join(f"{n:>6s}" for n in T.SHAPE_NAMES) + f" {'avg':>6s}"
print(hdr)
ds5 = sd.make_dataset_class()(DATA, "test", 5)
e5, h5 = load("/home/kavinder/SPACE-HOP/checkpoints/symsol_L5_aug/best.pth", 5)
per = eval_LL(e5, h5, ds5, 5)
print(f"{'trained@L5 (L5_aug)':>22} " + " ".join(f"{x:6.2f}" for x in per) + f" {np.mean(per):6.2f}")
del e5, h5; torch.cuda.empty_cache()
e6, h6 = load("/home/kavinder/SPACE-HOP/checkpoints/symsol_L6_aug/best.pth", 6)
per = eval_LL(e6, h6, ds5, 5)
print(f"{'L6_aug truncated->L5':>22} " + " ".join(f"{x:6.2f}" for x in per) + f" {np.mean(per):6.2f}")

print("\n# (2) per-mode peak memory (reset between modes), L6_aug at L=6, bs=1")
ds6 = sd.make_dataset_class()(DATA, "test", 6)
imgs = [ds6[i]["image"] for i in range(200)]
it0 = ds6[0]
with torch.no_grad():
    _ = e6(imgs[0][None].to(device)).float(); torch.cuda.synchronize()
    # likelihood-only peak
    torch.cuda.reset_peak_memory_stats()
    t_ll = []
    for im in imgs:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        p = e6(im[None].to(device)).float()
        lp = h6.path_loglik(p, it0["cell0"][:1].to(device), it0["child_idx"][:1].to(device),
                            it0["child_off"][:1].to(device), it0["parent_enc"][:1].to(device), 6)
        torch.cuda.synchronize(); t_ll.append(time.perf_counter() - t0)
    mem_ll = torch.cuda.max_memory_allocated() / 1e6
    # MAP-only peak
    torch.cuda.reset_peak_memory_stats()
    t_map = []
    for im in imgs:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ = T.beam_map(e6, h6, im[None], 6, 8, device)
        torch.cuda.synchronize(); t_map.append(time.perf_counter() - t0)
    mem_map = torch.cuda.max_memory_allocated() / 1e6
npar = sum(p.numel() for p in list(e6.parameters()) + list(h6.parameters())) / 1e6
print(f"params {npar:.1f}M")
print(f"likelihood: mem {mem_ll:.0f} MB | median {np.median(t_ll)*1e3:.1f} ms  (NOTE: shared GPU)")
print(f"beam-MAP  : mem {mem_map:.0f} MB | median {np.median(t_map)*1e3:.1f} ms  (NOTE: shared GPU)")
print("DONE")
