"""Continuous within-cell refinement head: training + refined-pose eval for the SE(3) density model.

The density head (se3_head.Se3DensityHead) gives an exactly-normalized distribution and a beam-MAP cell-centre
point estimate, but that estimate is capped by the finest-cell size (the readout-resolution ceiling). RefineHead
(se3_head.RefineHead) regresses a continuous residual (drotvec, dg) WITHIN the beam-MAP cell, lifting the estimate
off the centre. We fine-tune ONLY the RefineHead with the encoder + density head FROZEN ("isolate the effect":
LL untouched, cheap). Targets are the GT within-cell residuals from build_geometry (rot_resid, g_resid, finest-cell
units), trained on the roll-aug train stream (augment=True) so the geometry carries the refine targets per sample.

Works on all three benchmarks via `--bench`:
  - symsolt : se3_data geometry dir, fixed isotropic box A=diag(2*box_half), equiv-set geodesic eval
  - tless / ycbv : se3_tless_data (BOP index cache), per-instance detector box A=box_A(d, t_hat); the g-space refine
    TARGETS are box-independent so training is identical. (The leaderboard BOP-AR eval uses STANDARD BOP detections
    + bop_toolkit and lives in a separate pipeline; the eval here is a single-GT g-space/geodesic SANITY check.)

  python src/se3_refine.py oracle
  python src/se3_refine.py train --bench symsolt --init <best.pth> --geom <dir> --out <dir>
  python src/se3_refine.py train --bench tless  --init <best.pth> --out <dir> [--root R --cache C --obj O]
  python src/se3_refine.py eval  --ckpt <refine.pth> [--bench ...] [--geom ... | --root ... --cache ...]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import so3_nested as so3
import se3_tree as se3
import se3_head as sh
from se3_train import TimmEncoder
from se3_decode import map_pose_refined, beam_map, _geo_deg

ROT_THR = [2.0, 5.0, 10.0, 30.0]
TRN_THR = [0.02, 0.05, 0.1]


# ---------------------------------------------------------------- base-ckpt parsing (two formats) ----

def _detect_bench(ck):
    if "meta" in ck:                       # se3_train (SYMSOL-T): nested meta dict
        return "symsolt"
    if "d_m" in ck:                        # se3_tless_train (T-LESS / YCB-V): flat L_rot/L_trans/d_m/obj
        return "tless"
    raise ValueError("cannot detect bench from checkpoint keys: " + ",".join(sorted(ck.keys())))


def _params(ck, bench):
    """Normalize either checkpoint format into one param dict the refiner uses."""
    ta = ck["args"]
    if bench == "symsolt":
        m = ck["meta"]
        return dict(bench="symsolt", L_rot=m["L_rot"], L_trans=m["L_trans"],
                    backbone=ta["backbone"], img_size=ta["img_size"], box_half=m["box_half"])
    return dict(bench=bench, L_rot=ck["L_rot"], L_trans=ck["L_trans"],
                backbone=ta["backbone"], img_size=ta["img_size"],
                d_m=ck["d_m"], obj=ck["obj"], root=ta.get("root"), cache=ta.get("cache"),
                sigma=ta.get("sigma", 0.25), crop_pad=ta.get("crop_pad", 1.4))


def _build(p, device):
    enc = TimmEncoder(p["backbone"], p["img_size"]).to(device)
    head = sh.Se3DensityHead(embed_dim=enc.embed_dim, L_rot=p["L_rot"], L_trans=p["L_trans"]).to(device)
    refine = sh.RefineHead(embed_dim=enc.embed_dim).to(device)
    return enc, head, refine


def _train_loader(p, a):
    """augment=True stream so the collate carries the refine targets (both data paths emit them)."""
    if p["bench"] == "symsolt":
        import se3_data as sd
        full = sd.make_dataset_class()(a.geom, "train", augment=True)
        return DataLoader(full, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                          pin_memory=True, persistent_workers=a.workers > 0, collate_fn=sd.collate_train)
    import se3_tless_data as td
    root = a.root or p["root"]; cache = a.cache or p["cache"]; obj = a.obj or p["obj"]
    ds = td.make_dataset_class()(root, cache, "train_pbr", obj, p["L_rot"], p["L_trans"], p["d_m"],
                                 t_hat="jitter", sigma=p["sigma"], crop_pad=p["crop_pad"],
                                 out_px=p["img_size"], seed=a.seed, augment=True)
    return DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                      pin_memory=True, persistent_workers=a.workers > 0, collate_fn=td.Collate(p["L_rot"], p["L_trans"]))


# ---------------------------------------------------------------- training (frozen base) ----

def train_refine(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.init, map_location=device, weights_only=False)
    bench = a.bench or _detect_bench(ck)
    p = _params(ck, bench)
    enc, head, refine = _build(p, device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    enc.eval(); head.eval()
    for q in list(enc.parameters()) + list(head.parameters()):
        q.requires_grad_(False)
    cs = so3.cell_size_rad(p["L_rot"]) if p["L_rot"] > 0 else 1.0
    edge = (1.0 / (2 ** p["L_trans"])) if p["L_trans"] > 0 else 1.0
    print(f"refine fine-tune [{bench}]: init={a.init} L=({p['L_rot']},{p['L_trans']}) cs={cs:.4f}rad edge={edge:.5f} "
          f"lambda_rot={a.lambda_rot} | base FROZEN", flush=True)

    dl = _train_loader(p, a)
    opt = torch.optim.AdamW(refine.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    sm = nn.SmoothL1Loss(beta=0.1)

    step, t0, run = 0, time.time(), 0.0
    while step < a.steps:
        for b in dl:
            if "refine" not in b:
                raise KeyError("batch has no refine targets -- train must run with augment=True (roll-aug stream)")
            img = b["image"].to(device, non_blocking=True)
            code12 = b["refine"]["code12"].to(device)
            tgt = torch.cat([b["refine"]["rot_resid"], b["refine"]["g_resid"]], -1).to(device)   # (B,6)
            with torch.no_grad():
                patches = enc(img).float()
            pred = refine(patches, code12)
            loss = a.lambda_rot * sm(pred[:, :3], tgt[:, :3]) + sm(pred[:, 3:], tgt[:, 3:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(refine.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); step += 1
            if step % a.log_every == 0:
                print(f"step {step}/{a.steps} | smoothL1 {run/a.log_every:.4f} | {(time.time()-t0)/step:.2f}s/it",
                      flush=True)
                run = 0.0
            if step % a.save_every == 0 or step == a.steps:
                torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "refine": refine.state_dict(),
                            "bench": bench, "params": p, "args": vars(a), "step": step}, f"{a.out}/refine.pth")
            if step >= a.steps:
                break
    print("=== refine training done; running refined eval ===", flush=True)
    a.ckpt = f"{a.out}/refine.pth"
    eval_refined(a)


# ---------------------------------------------------------------- refined-pose eval ----

def _load_for_eval(a, device):
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    if "params" in ck:                                   # new refine.pth format
        p, bench = ck["params"], ck["bench"]
    else:                                                # legacy SYMSOL-T refine.pth (enc/head/refine + meta/init_args)
        bench = a.bench or ("symsolt" if "meta" in ck else "tless")
        p = _params({"meta": ck.get("meta"), "args": ck.get("init_args", ck.get("args")),
                     **{k: ck[k] for k in ("L_rot", "L_trans", "d_m", "obj") if k in ck}}, bench)
    enc, head, refine = _build(p, device)
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); refine.load_state_dict(ck["refine"])
    enc.eval(); head.eval(); refine.eval()
    return enc, head, refine, p, bench


@torch.no_grad()
def eval_refined(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, head, refine, p, bench = _load_for_eval(a, device)
    if bench == "symsolt":
        return _eval_symsolt(a, enc, head, refine, p, device)
    return _eval_tless_sanity(a, enc, head, refine, p, device)


def _accloop(R, T):
    ra = "  ".join(f"rot@{int(th)} {float((R <= th).mean()):.3f}" for th in ROT_THR)
    ta_ = "  ".join(f"trans@{th} {float((T <= th).mean()):.3f}" for th in TRN_THR)
    return ra, ta_


def _eval_symsolt(a, enc, head, refine, p, device):
    import se3_data as sd
    L_rot, L_trans, bh = p["L_rot"], p["L_trans"], p["box_half"]
    A = np.diag([2.0 * bh] * 3); t_hat = np.zeros(3)
    ds = sd.make_dataset_class()(a.geom, "test")
    labels_all = np.load(f"{a.geom}/test_label.npy")
    PER = max(1, a.n_eval // len(sd.SHAPE_NAMES)); idxs = []
    for s in range(len(sd.SHAPE_NAMES)):
        pool = np.where(labels_all == s)[0]
        if len(pool):
            idxs += pool[np.linspace(0, len(pool) - 1, min(PER, len(pool))).astype(int)].tolist()
    per = {s: {"r": [], "t": []} for s in range(len(sd.SHAPE_NAMES))}
    for i in idxs:
        it = ds.test_item(i)
        patches = enc(it["image"][None].to(device)).float()[0]
        R, t, _ = map_pose_refined(head, refine, patches, L_rot, L_trans, A, t_hat, a.beam)
        per[it["label"]]["r"].append(_geo_deg(R, np.asarray(it["equiv_R"], np.float64)))
        per[it["label"]]["t"].append(float(np.linalg.norm(t - np.asarray(it["t"], np.float64))))
    allr, allt = [], []
    for s in range(len(sd.SHAPE_NAMES)):
        if not per[s]["r"]:
            continue
        r, t = np.array(per[s]["r"]), np.array(per[s]["t"]); allr += per[s]["r"]; allt += per[s]["t"]
        ra, ta_ = _accloop(r, t)
        print(f"  {sd.SHAPE_NAMES[s]:5s} rot med {np.median(r):5.2f} | {ra} | {ta_}", flush=True)
    allr, allt = np.array(allr), np.array(allt); ra, ta_ = _accloop(allr, allt)
    print(f"  ALL   rot med {np.median(allr):5.2f} | {ra} | {ta_}  (n={len(allr)}, beam={a.beam})", flush=True)
    return allr, allt


def _eval_tless_sanity(a, enc, head, refine, p, device):
    """Single-GT g-space / geodesic SANITY (NOT BOP AR). Compares refined decode vs the raw cell-centre on the clean
    (augment=False) jitter val stream, so the within-cell improvement is visible. Full AR = standard BOP detections
    + bop_toolkit, separate pipeline. For symmetric objects geodesic-to-single-GT over-counts, but the refined-vs-
    centre DELTA is still meaningful (same single-GT basis)."""
    import se3_tless_data as td
    L_rot, L_trans = p["L_rot"], p["L_trans"]
    root = a.root or p["root"]; cache = a.cache or p["cache"]; obj = a.obj or p["obj"]
    ds = td.make_dataset_class()(root, cache, a.split, obj, L_rot, L_trans, p["d_m"],
                                 t_hat="jitter", sigma=p["sigma"], crop_pad=p["crop_pad"],
                                 out_px=p["img_size"], seed=777, augment=False)
    n = min(a.n_eval, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    cs = so3.cell_size_rad(L_rot); edge = 1.0 / (2 ** L_trans)
    rc_err, gc_err, rr_err, gr_err = [], [], [], []   # centre vs refined, rot(deg) & g-L2
    from scipy.spatial.transform import Rotation as Rot
    for i in idxs:
        it = ds[int(i)]
        R_gt = it["R"].numpy().astype(np.float64); g_gt = it["g"].numpy().astype(np.float64)
        patches = enc(it["image"][None].to(device)).float()[0]
        rcl, tcl, _ = beam_map(head, patches, L_rot, L_trans, a.beam)
        R_fin = so3.cell_index_to_matrix(np.array([rcl]), L_rot)[0]
        g_ctr = se3.trans_cell_center(np.array([tcl]), L_trans)[0]
        rc_err.append(_geo_deg(R_fin, R_gt[None])); gc_err.append(float(np.linalg.norm(g_ctr - g_gt)))
        code12 = torch.as_tensor(np.concatenate([R_fin.reshape(9), g_ctr]).astype(np.float32), device=device)[None]
        resid = refine(patches[None], code12)[0].double().cpu().numpy()
        R_ref = Rot.from_rotvec(resid[:3] * cs).as_matrix() @ R_fin
        g_ref = g_ctr + resid[3:] * edge
        rr_err.append(_geo_deg(R_ref, R_gt[None])); gr_err.append(float(np.linalg.norm(g_ref - g_gt)))
    rc, gc, rr, gr = map(np.array, (rc_err, gc_err, rr_err, gr_err))
    print(f"  [tless obj{obj} sanity, single-GT, n={len(rc)}, beam={a.beam}]", flush=True)
    print(f"    cell-centre : rot med {np.median(rc):6.2f}deg  g-L2 med {np.median(gc):.4f}", flush=True)
    print(f"    refined     : rot med {np.median(rr):6.2f}deg  g-L2 med {np.median(gr):.4f}", flush=True)
    print(f"    refine delta: rot {np.median(rr)-np.median(rc):+.2f}deg  g-L2 {np.median(gr)-np.median(gc):+.4f} "
          f"(negative = improvement)", flush=True)
    return rr, gr


# ---------------------------------------------------------------- oracle (decode math == target defn) ----

def oracle():
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(0); L_rot = L_trans = 5
    mats = Rot.random(64, random_state=0).as_matrix(); gs = rng.uniform(-0.5, 0.5, size=(64, 3))
    geom = sh.build_geometry(mats, gs, L_rot, L_trans); r = geom["refine"]
    cs, edge = float(r["cell_size_rad"]), float(r["trans_edge"])
    rerr, terr = [], []
    for i in range(64):
        R_fin = r["code12"][i, :9].reshape(3, 3).astype(np.float64); g_ctr = r["code12"][i, 9:].astype(np.float64)
        R = Rot.from_rotvec(r["rot_resid"][i].astype(np.float64) * cs).as_matrix() @ R_fin
        g = g_ctr + r["g_resid"][i].astype(np.float64) * edge
        tr = float(np.trace(R.T @ mats[i])); rerr.append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        terr.append(np.linalg.norm(g - gs[i]))
    print(f"ORACLE reconstruct GT from (cell code + GT residual): rot max {max(rerr):.3e} deg  g max {max(terr):.3e}",
          flush=True)
    assert max(rerr) < 0.05 and max(terr) < 1e-5, "decode math does not match target definition"
    print("ORACLE OK: refine decode math matches build_geometry target definition", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--bench", choices=["symsolt", "tless", "ycbv"], default=None, help="default: auto-detect from ckpt")
    pt.add_argument("--init", required=True); pt.add_argument("--out", required=True)
    pt.add_argument("--geom", default=None, help="symsolt geometry dir")
    pt.add_argument("--root", default=None); pt.add_argument("--cache", default=None); pt.add_argument("--obj", type=int, default=None)
    pt.add_argument("--steps", type=int, default=8000); pt.add_argument("--bs", type=int, default=24)
    pt.add_argument("--lr", type=float, default=3e-4); pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--lambda_rot", type=float, default=1.0); pt.add_argument("--log_every", type=int, default=250)
    pt.add_argument("--save_every", type=int, default=2000); pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--beam", type=int, default=64); pt.add_argument("--n_eval", type=int, default=600)
    pt.add_argument("--split", default="test", help="tless/ycbv sanity-eval split")
    pe = sub.add_parser("eval")
    pe.add_argument("--ckpt", required=True); pe.add_argument("--bench", choices=["symsolt", "tless", "ycbv"], default=None)
    pe.add_argument("--geom", default=None)
    pe.add_argument("--root", default=None); pe.add_argument("--cache", default=None); pe.add_argument("--obj", type=int, default=None)
    pe.add_argument("--beam", type=int, default=64); pe.add_argument("--n_eval", type=int, default=600)
    pe.add_argument("--split", default="test")
    sub.add_parser("oracle")
    a = ap.parse_args()
    if a.cmd == "oracle":
        oracle()
    elif a.cmd == "train":
        train_refine(a)
    else:
        eval_refined(a)
