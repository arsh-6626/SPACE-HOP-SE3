"""Run ONE real SYMSOL SO(3) forward pass (device 1) and dump what the globe-tree figure needs."""
import os, sys, numpy as np, torch
sys.path.insert(0, "src")
import symsol_train as st, symsol_data as sd, so3_nested as g

CKPT = "checkpoints/symsol_L6_aug/best.pth"; DATA = "data/symsol"; DEV = "cuda"
L_VIZ = 5; OUT = "logs/figs/_cache/so3_dump.npz"
TARGET_SHAPE = "cube"

ck = torch.load(CKPT, map_location=DEV, weights_only=False); ta = ck["args"]
head_kw, proj_dim = st._cfg_from_args(ta); LEVEL = ta.get("level", 6)
enc, head = st.build(LEVEL, ta["backbone"], ta["img_size"], DEV, head_kw=head_kw, proj_dim=proj_dim)
enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
DS = sd.make_dataset_class(); ds = DS(DATA, "test", 5)
labels = np.load(f"{DATA}/test_label.npy")
em = np.load(f"{DATA}/test_equiv_mats.npy"); eo = np.load(f"{DATA}/test_equiv_off.npy")
sid = sd.SHAPE_NAMES.index(TARGET_SHAPE)

@torch.no_grad()
def feats(i):
    patches = enc(ds._img(i)[None].to(DEV)).float()
    return patches, head.logp0(patches)[0]

cands = [i for i in range(len(labels)) if int(labels[i]) == sid][:40]
best = max(cands, key=lambda i: float(feats(i)[1].max().exp()))
patches, lp0 = feats(best)
p0 = lp0.exp().cpu().numpy(); p0 /= p0.sum()
root_cell = int(lp0.argmax())
print(f"image #{best} shape={TARGET_SHAPE} root MAP p={p0.max():.3f}  n_equiv={eo[best+1]-eo[best]}")

@torch.no_grad()
def decode(nlevels):
    node = int(lp0.argmax()); rows = []
    for j in range(nlevels):
        off, par, chf = sd.node_children_features(np.array([node]), j)
        clp = head.child_logp(patches, torch.from_numpy(off).to(DEV), torch.from_numpy(par).to(DEV), j)[0]
        probs = clp.exp().cpu().numpy(); ci = int(clp.argmax())
        rows.append((chf[0].astype(np.int64), probs.astype(np.float64), ci))
        node = int(chf[0][ci])
    return node, rows
leaf_node, rows = decode(L_VIZ)
R_map = g.cell_index_to_matrix(np.array([leaf_node]), L_VIZ)[0]
for j, (cells, probs, ci) in enumerate(rows):
    print(f"  lvl {j+1}: chosen {ci} p={probs[ci]:.3f}")

# nearest symmetric GT + geodesic error
G = em[eo[best]:eo[best + 1]].astype(np.float64)
cos = (np.einsum("nij,ij->n", G, R_map) - 1) / 2          # tr(G^T R_map)=<G,R_map> since both SO(3)
ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
k = int(np.argmin(ang)); R_gt = G[k]; err = float(ang[k])
print(f"leaf rot err (min over {len(G)} symmetric GT) = {err:.1f} deg")

raw = ds._raw(int(best))
np.savez(OUT, raw=np.asarray(raw, np.uint8), p0=p0, root_cell=root_cell,
         base_map=(R_map @ np.array([0, 0, 1.0])), R_map=R_map, R_gt=R_gt, err=err, shape=TARGET_SHAPE,
         L_VIZ=L_VIZ,
         **{f"cells_{j}": rows[j][0] for j in range(L_VIZ)},
         **{f"probs_{j}": rows[j][1] for j in range(L_VIZ)},
         chosen=np.array([rows[j][2] for j in range(L_VIZ)]))
print("saved", OUT, "base_map", (R_map @ np.array([0,0,1.0])).round(3))
