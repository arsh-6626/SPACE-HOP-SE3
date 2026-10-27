"""Close out SYMSOL I on the idle GPU: (a) grid-depth sweep L3-6 (tab:abl-depth) by truncating the L6 model's
paths, equiv-set LL on a per-shape subset; (b) efficiency timing/memory (tab:efficiency) for the likelihood and the
beam-MAP, measured clean."""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/kavinder/SPACE-HOP/src")
import numpy as np, torch
import symsol_train as T
import symsol_data as sd

DATA = "/home/kavinder/SPACE-HOP/data/symsol_L6"
CKPT = "/home/kavinder/SPACE-HOP/checkpoints/symsol_L6_aug/best.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=device, weights_only=False); ta = ck["args"]
enc, head = T.build(6, ta["backbone"], ta["img_size"], device)
enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
NSHOW = 400

print("# (a) grid-depth sweep (equiv-set LL, subset)  -- tab:abl-depth")
print(f"{'L':>2} " + " ".join(f"{n:>6s}" for n in T.SHAPE_NAMES) + f" {'avg':>6s}")
with torch.no_grad():
    for L in [3, 4, 5, 6]:
        ds = sd.make_dataset_class()(DATA, "test", L)
        by = {s: [] for s in range(len(T.SHAPE_NAMES))}
        for i in range(len(ds)):
            by[int(ds.label[i])].append(i)
        dens = T.DENS(L); per = []
        for s in range(len(T.SHAPE_NAMES)):
            idx = by[s][:NSHOW]; acc = []
            for i in idx:
                it = ds[i]; K = it["cell0"].shape[0]
                p = enc(it["image"][None].to(device)).float().expand(K, -1, -1)
                lp = head.path_loglik(p, it["cell0"].to(device), it["child_idx"].to(device),
                                      it["child_off"].to(device), it["parent_enc"].to(device), L)
                acc.append((lp + dens).mean().item())
            per.append(np.mean(acc))
        print(f"{L:>2} " + " ".join(f"{x:6.2f}" for x in per) + f" {np.mean(per):6.2f}")

# (b) efficiency: time likelihood (one GT path) and beam-MAP per image, peak memory
print("\n# (b) efficiency (L=6, bs=1, median over 200 imgs)  -- tab:efficiency")
ds = sd.make_dataset_class()(DATA, "test", 6)
imgs = [ds[i]["image"] for i in range(200)]
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    # warmup
    _ = enc(imgs[0][None].to(device)).float()
    torch.cuda.synchronize()
    t_ll = []
    for im in imgs:
        it = ds[0]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        p = enc(im[None].to(device)).float()
        lp = head.path_loglik(p, it["cell0"][:1].to(device), it["child_idx"][:1].to(device),
                              it["child_off"][:1].to(device), it["parent_enc"][:1].to(device), 6)
        torch.cuda.synchronize(); t_ll.append(time.perf_counter() - t0)
    t_map = []
    for im in imgs:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ = T.beam_map(enc, head, im[None], 6, 8, device)
        torch.cuda.synchronize(); t_map.append(time.perf_counter() - t0)
peak = torch.cuda.max_memory_allocated() / 1e6
npar = sum(p.numel() for p in list(enc.parameters()) + list(head.parameters())) / 1e6
print(f"params {npar:.1f}M | peak mem {peak:.0f} MB")
print(f"likelihood (1 query): median {np.median(t_ll)*1e3:.1f} ms/img")
print(f"beam-MAP (b=8, L=6):  median {np.median(t_map)*1e3:.1f} ms/img")
