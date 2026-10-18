"""
Evaluate a trained SPACE-HOP pose model on the ISRO GSAT-12R val fold and/or held-out test set.

    python src/val.py --config configs/isro.yaml --checkpoint checkpoints/pose_isro/best_model.pth --splits val test

Reports E_T (m), E_T/|t|, geodesic E_R (deg) and the SPEED-style score E_T/|t| + E_R[rad].
Crops are derived from ground-truth landmark projections (oracle detection), as in training.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from train import build_dataset, box_corners, load_config, validate
from model import FastPoseViT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out', default=None, help='output dir (default: next to the checkpoint)')
    ap.add_argument('--vis', action='store_true', help='save wireframe overlays for the first batch')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), 'eval')
    os.makedirs(out_dir, exist_ok=True)

    # Backbone weights come from the checkpoint, so the JEPA init is not needed here.
    model = FastPoseViT(jepa_path=None, img_size=cfg['data'].get('crop_size', 224), patch_size=16,
                        backbone=cfg['model'].get('backbone', 'jepa'), grad_ckpt=cfg['model'].get('grad_ckpt', False))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {epoch}")

    results = {}
    for split in args.splits:
        dataset = build_dataset(cfg, split, train=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg['data'].get('num_workers', 8), pin_memory=True)
        metrics, (idx, t_err, r_err) = validate(
            model, loader, device, dataset.K, tuple(dataset.full_size), epoch, name=split,
            vis_dir=os.path.join(out_dir, 'vis') if args.vis else None,
            corners_3d=box_corners(dataset.points_3d), amp=cfg['training'].get('amp_bf16', True),
            return_per_sample=True)
        results[split] = metrics
        with open(os.path.join(out_dir, f'{split}_per_image.csv'), 'w') as f:
            f.write('filename,E_T_m,E_R_deg\n')
            for i, te, re in zip(idx.tolist(), t_err.tolist(), r_err.tolist()):
                f.write(f'{dataset.filename(i)},{te:.6f},{re:.6f}\n')

    print("\n===== FINAL METRICS =====")
    for split, m in results.items():
        print(f"{split:5s} (n={m['n']}): E_T mean {m['E_T_mean_m']:.4f} m | median {m['E_T_median_m']:.4f} m | "
              f"E_T/|t| {m['E_T_norm_mean']:.4f} | E_R mean {m['E_R_mean_deg']:.3f} deg | "
              f"median {m['E_R_median_deg']:.3f} deg | <5deg {m['acc_R_5deg']:.3f} | score {m['score_mean']:.4f}")
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'checkpoint': os.path.abspath(args.checkpoint), 'epoch': epoch, **results}, f, indent=2)
    print(f"wrote {out_dir}/metrics.json")
