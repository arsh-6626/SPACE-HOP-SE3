"""
Beam-MAP decoding for the hierarchical SE(3) density head (se3_head.py): find the most likely pose under the
tree-structured distribution by a top-B beam search down the joint rotation x translation tree, then read off the
finest cell's centre as the point estimate (R_map, t_map). This is the pose-error readout (geodesic rotation error,
translation error) that makes the SE(3) distribution comparable to point/diffusion baselines (e.g. LiePose on
SYMSOL-T, SpyroPose's BOP-metric line on YCB-V) in addition to the log-likelihood.

The expander builds each node's children exactly as build_geometry() does for the GT path, but for ALL children of
an arbitrary node: rotation child offsets from R_child @ R_parent^T / cell_size_rad, translation octant offsets
TRANS_OCT_OFF, and the 12-D parent code [parent rotation flat(9), parent translation-cell centre(3)]. Scoring each
node with head.child_logp reproduces the same log-softmax used in training, so a full-width beam equals exhaustive
argmax (verified in checks()).

    python src/se3_decode.py checks      # beam-MAP == exhaustive argmax on small grids; cell read-out round trip
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import so3_nested as so3
import se3_tree as se3
from se3_head import TRANS_OCT_OFF


# ---------------------------------------------------------------- node expanders (numpy) ----

def _rot_child_geom(par_cell, ri):
    """Parent rotation cell at level ri -> (child cells (8,), parent flat (9,), child offsets (8,3))."""
    from scipy.spatial.transform import Rotation as Rot
    chf = so3.parent_to_children_cells(np.array([par_cell]), ri)[0]            # (8,)
    R_par = so3.cell_index_to_matrix(np.array([par_cell]), ri)[0]              # (3,3)
    R_ch = so3.cell_index_to_matrix(chf, ri + 1)                              # (8,3,3)
    rel = np.einsum('cij,kj->cik', R_ch, R_par)                               # R_child @ R_parent^T
    off = (Rot.from_matrix(rel).as_rotvec() / so3.cell_size_rad(ri + 1)).astype(np.float32)
    return chf.astype(np.int64), R_par.reshape(9).astype(np.float32), off


def _trans_children(cell, level):
    """Translation cell at `level` -> 8 child flat cells at level+1, ordered by octant o = bx + 2by + 4bz."""
    ix, iy, iz = (int(v[0]) for v in se3._trans_flat_to_coords(np.array([cell]), level))
    out = np.empty(8, np.int64)
    for o in range(8):
        bx, by, bz = o & 1, (o >> 1) & 1, (o >> 2) & 1
        out[o] = se3._trans_coords_to_flat(2 * ix + bx, 2 * iy + by, 2 * iz + bz, level + 1)
    return out


# ---------------------------------------------------------------- beam-MAP ----

@torch.no_grad()
def beam_map(head, patches, L_rot, L_trans, beam=64):
    """patches (P,D) for ONE image -> (rot_cell_fin, trans_cell_fin, logp_cell) of the beam-MAP finest joint cell.
    beam is the number of nodes kept at every level; beam>=72 and kept-all children == exhaustive."""
    device = patches.device
    P, D = patches.shape
    L = max(L_rot, L_trans)
    # level 0: 72 rotation root cells (translation root is the single cell 0)
    lp0 = head.logp0(patches[None])[0].cpu().numpy()                          # (72,)
    order = np.argsort(-lp0)[:beam]
    rot_cell = order.astype(np.int64)                                         # (B,) at rot level 0
    trans_cell = np.zeros(len(order), np.int64)                               # (B,) at trans level 0
    logp = lp0[order].astype(np.float64)
    rl = tl = 0                                                               # current rotation / translation depth

    for l in range(1, L + 1):
        kind = se3.joint_level_kind(l, L_rot, L_trans)
        B = len(rot_cell)
        K = 64 if kind == 'joint64' else 8
        off6 = np.zeros((B, K, 6), np.float32)
        par12 = np.zeros((B, 12), np.float32)
        child_rot = np.empty((B, K), np.int64)                                # child rot cell per (node, child)
        child_trans = np.empty((B, K), np.int64)
        for b in range(B):
            if kind in ('joint64', 'rot8'):
                chf, par_flat, roff = _rot_child_geom(int(rot_cell[b]), rl)
            else:
                par_flat = so3.cell_index_to_matrix(np.array([rot_cell[b]]), rl)[0].reshape(9).astype(np.float32)
            if kind in ('joint64', 'trans8'):
                tch = _trans_children(int(trans_cell[b]), tl)
                tctr = se3.trans_cell_center(np.array([trans_cell[b]]), tl)[0].astype(np.float32)
            if kind == 'joint64':
                roff_rep = np.repeat(roff[:, None, :], 8, axis=1)             # (8rc,8to,3)
                toff_rep = np.broadcast_to(TRANS_OCT_OFF[None, :, :], (8, 8, 3))
                off6[b] = np.concatenate([roff_rep, toff_rep], -1).reshape(64, 6)
                par12[b] = np.concatenate([par_flat, tctr])
                child_rot[b] = np.repeat(chf, 8)                              # rc*8+to -> chf[rc]
                child_trans[b] = np.tile(tch, 8)                              # rc*8+to -> tch[to]
            elif kind == 'rot8':
                tctr_fin = se3.trans_cell_center(np.array([trans_cell[b]]), tl)[0].astype(np.float32)
                off6[b] = np.concatenate([roff, np.zeros((8, 3), np.float32)], -1)
                par12[b] = np.concatenate([par_flat, tctr_fin])
                child_rot[b] = chf; child_trans[b] = trans_cell[b]
            else:  # trans8
                off6[b] = np.concatenate([np.zeros((8, 3), np.float32),
                                          np.broadcast_to(TRANS_OCT_OFF, (8, 3))], -1)
                par12[b] = np.concatenate([par_flat, tctr])
                child_rot[b] = rot_cell[b]; child_trans[b] = tch
        # score all children of all beam nodes
        pts = patches[None].expand(B, P, D)
        clp = head.child_logp(pts, torch.as_tensor(off6, device=device),
                              torch.as_tensor(par12, device=device), l - 1).cpu().numpy()  # (B,K) log-softmax
        cum = (logp[:, None] + clp).reshape(-1)                               # (B*K,)
        keep = np.argsort(-cum)[:beam]
        logp = cum[keep]
        rot_cell = child_rot.reshape(-1)[keep]
        trans_cell = child_trans.reshape(-1)[keep]
        if kind in ('joint64', 'rot8'):
            rl += 1
        if kind in ('joint64', 'trans8'):
            tl += 1
    return int(rot_cell[0]), int(trans_cell[0]), float(logp[0])


def map_pose(head, patches, L_rot, L_trans, A, t_hat, beam=64):
    """Beam-MAP finest cell -> (R_map (3,3), t_map (3,)) in the metric frame, via t = t_hat + A g_centre."""
    rc, tc, lp = beam_map(head, patches, L_rot, L_trans, beam)
    R_map = so3.cell_index_to_matrix(np.array([rc]), L_rot)[0]
    g = se3.trans_cell_center(np.array([tc]), L_trans)[0]
    t_map = np.asarray(t_hat, np.float64) + np.asarray(A, np.float64) @ g
    return R_map, t_map, lp


def _geo_deg(R_map, R_set):
    """min geodesic angle (deg) between R_map (3,3) and any R in R_set (K,3,3)."""
    tr = np.einsum('ij,kij->k', R_map, R_set)                                # trace(R_map^T R_k) = sum R_map*R_k
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))).min())


# ---------------------------------------------------------------- SYMSOL-T pose-error eval ----

@torch.no_grad()
def eval_symsolt_pose(ckpt, geom_dir, n=None, beam=256, device=None):
    """Beam-MAP pose error on SYMSOL-T test: min-geodesic rotation error (deg, over the symmetric equivalent set)
    and translation L2 error (render units, box t in [-box_half,box_half]^3), per shape and overall."""
    import se3_data as sd
    from se3_train import TimmEncoder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    meta, ta = ck["meta"], ck["args"]
    L_rot, L_trans, bh = meta["L_rot"], meta["L_trans"], meta["box_half"]
    A = np.diag([2.0 * bh, 2.0 * bh, 2.0 * bh]); t_hat = np.zeros(3)
    enc = TimmEncoder(ta["backbone"], ta["img_size"]).to(device).eval()
    head = __import__("se3_head").Se3DensityHead(embed_dim=enc.embed_dim, L_rot=L_rot, L_trans=L_trans).to(device).eval()
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    ds = sd.make_dataset_class()(geom_dir, "test")
    idx = range(len(ds)) if n is None else range(min(n, len(ds)))
    per = {s: {"r": [], "t": []} for s in sd.SHAPE_NAMES}
    for i in idx:
        it = ds.test_item(i)
        patches = enc(it["image"][None].to(device)).float()[0]
        R_map, t_map, _ = map_pose(head, patches, L_rot, L_trans, A, t_hat, beam)
        per[sd.SHAPE_NAMES[it["label"]]]["r"].append(_geo_deg(R_map, it["equiv_R"]))
        per[sd.SHAPE_NAMES[it["label"]]]["t"].append(float(np.linalg.norm(t_map - it["t"])))
    rows, allr, allt = [], [], []
    for s in sd.SHAPE_NAMES:
        if not per[s]["r"]:
            continue
        r, t = np.array(per[s]["r"]), np.array(per[s]["t"])
        allr += per[s]["r"]; allt += per[s]["t"]
        rows.append(f"  {s:5s} rot {r.mean():6.2f}deg (med {np.median(r):5.2f})  trans {t.mean():.4f}  (n={len(r)})")
    allr, allt = np.array(allr), np.array(allt)
    rows.append(f"  ALL   rot {allr.mean():6.2f}deg (med {np.median(allr):5.2f})  trans {allt.mean():.4f}  "
                f"(n={len(allr)}, beam={beam}, L=({L_rot},{L_trans}))")
    print("\n".join(rows), flush=True)
    return allr.mean(), allt.mean()


# ---------------------------------------------------------------- checks: beam == independent exhaustive ----

@torch.no_grad()
def _indep_argmax(head, patches, L_rot, L_trans):
    """Independent exhaustive argmax: enumerate every finest joint cell, map it to its centre pose, score it with the
    training-time path_loglik (NOT the beam expander), and take the max. Small grids only. Returns (rc, tc, logp)."""
    import se3_head as sh
    rot_fin = np.arange(so3.n_cells(L_rot)); trans_fin = np.arange(se3.n_trans_cells(L_trans))
    Rmats = so3.cells_to_matrices(L_rot)                                      # (nrf,3,3)
    gctr = se3.trans_cell_center(trans_fin, L_trans)                          # (ntf,3)
    RC, TC = (a.reshape(-1) for a in np.meshgrid(rot_fin, trans_fin, indexing='ij'))
    geom = sh.to_torch(sh.build_geometry(Rmats[RC], gctr[TC], L_rot, L_trans), patches.device)
    pts = patches[None].expand(len(RC), *patches.shape)
    lp = head.path_loglik(pts, geom['cell0'], geom['levels']).cpu().numpy()
    k = int(np.argmax(lp))
    return int(RC[k]), int(TC[k]), float(lp[k])


def _cell_roundtrip():
    """A finest cell's centre pose maps back to that same cell (read-out used for the MAP point estimate)."""
    worst = 0
    for L_rot, L_trans in [(2, 2), (3, 3)]:
        rc = np.random.RandomState(0).randint(0, so3.n_cells(L_rot), 500)
        back = so3.rotations_to_cells(so3.cell_index_to_matrix(rc, L_rot), L_rot)
        worst = max(worst, int(np.max(np.abs(back - rc))))
        tc = np.random.RandomState(1).randint(0, se3.n_trans_cells(L_trans), 500)
        tback = se3.points_to_trans_cells(se3.trans_cell_center(tc, L_trans), L_trans)
        worst = max(worst, int(np.max(np.abs(tback - tc))))
    assert worst == 0, f"cell centre read-out round trip failed ({worst})"
    print(f"  cell read-out round trip OK (rot+trans, L up to (3,3))")


def checks():
    import torch.nn as nn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    from se3_head import Se3DensityHead
    _cell_roundtrip()
    for L_rot, L_trans in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        head = Se3DensityHead(embed_dim=64, L_rot=L_rot, L_trans=L_trans, n_layers=1, heads=4).to(device).eval()
        for m in head.modules():                                             # randomise zero-init outputs -> non-flat
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.normal_(m.weight, std=1.5); nn.init.normal_(m.bias, std=1.5)
        n_match, n_dlp = 0, 0.0
        for _ in range(6):
            patches = torch.randn(12, 64, device=device)
            erc, etc, elp = _indep_argmax(head, patches, L_rot, L_trans)     # ground-truth argmax
            brc, btc, blp = beam_map(head, patches, L_rot, L_trans, beam=10 ** 9)  # full-width == exhaustive
            n_match += int(brc == erc and btc == etc)
            n_dlp = max(n_dlp, abs(blp - elp))
        assert n_match == 6, f"beam-MAP != exhaustive argmax at L=({L_rot},{L_trans}): {n_match}/6"
        assert n_dlp < 1e-4, f"beam logp off by {n_dlp:.2e}"
        print(f"  L=({L_rot},{L_trans}): beam-MAP == exhaustive argmax 6/6  (max logp dev {n_dlp:.1e})")
    print("se3_decode checks OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "checks":
        checks()
    elif len(sys.argv) > 1 and sys.argv[1] == "posezeval":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("posezeval")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--geom", required=True)
        ap.add_argument("--n", type=int, default=None)
        ap.add_argument("--beam", type=int, default=256)
        a = ap.parse_args()
        eval_symsolt_pose(a.ckpt, a.geom, a.n, a.beam)
    else:
        print(__doc__)
