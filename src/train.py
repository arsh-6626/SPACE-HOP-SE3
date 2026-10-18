import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils 
from tqdm import tqdm
import numpy as np
import argparse
import json
import os
import sys
import time
import yaml
import cv2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SRC_DIR, 'utils'))  # after src/, so src/box.py wins over utils/box.py
from isro_data import IsroPoseDataset, seed_worker
from model import FastPoseViT
from box import draw_satellite_wireframe
from hopf_grid import get_closest_anchor
from geometry_utils import (
    recover_true_rotation, 
    recover_translation_components, 
    ortho6d_to_matrix,
    get_gt_deltas, 
    axis_angle_to_matrix,
    make_pose_targets,
    rotation_error_deg,
) 

class _NoWandb:
    """Stand-in when wandb logging is disabled."""
    def log(self, *a, **k): pass
    def init(self, *a, **k): pass
    def finish(self, *a, **k): pass
    Image = None

wandb = _NoWandb()

def translation_loss(u_pred, u_gt):
    return torch.sum((u_pred - u_gt) ** 2, dim=1).mean()

def rotation_loss(r6d_pred, r6d_gt):
    return torch.sum((r6d_pred - r6d_gt) ** 2, dim=1).mean()

def rotmat2quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert R.shape[-2:] == (3, 3)
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = r00 + r11 + r22
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)

    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    mask = (trace <= 0) & (r00 >= r11) & (r00 >= r22)
    s = torch.sqrt(1.0 + r00[mask] - r11[mask] - r22[mask]) * 2.0
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 1] = 0.25 * s
    q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s

    mask = (trace <= 0) & (r11 > r00) & (r11 >= r22)
    s = torch.sqrt(1.0 + r11[mask] - r00[mask] - r22[mask]) * 2.0
    q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    q[mask, 2] = 0.25 * s
    q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s

    mask = (trace <= 0) & (r22 > r00) & (r22 > r11)
    s = torch.sqrt(1.0 + r22[mask] - r00[mask] - r11[mask]) * 2.0
    q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
    q[mask, 3] = 0.25 * s
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    return q

def translation_error_euclidian(T_pred, T_gt):
    return torch.norm(T_pred - T_gt, dim=1).mean()

def rotation_error_geodesic(R_pred, R_gt):
    return rotation_error_deg(R_pred, R_gt).mean()

def save_debug_images(images, path="debug_input_batch.png"):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    denorm_imgs = images * std + mean
    denorm_imgs = torch.clamp(denorm_imgs, 0, 1)
    vutils.save_image(denorm_imgs, path, nrow=8)
    print(f"[DEBUG] Saved input batch grid to {path}")

def get_optimizer(model, config):
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # Catch all three of our new decoupled heads
        # In train.py inside get_optimizer()
        is_head = any(h in name for h in ["pose_head", "coarse_cross_attn", "coarse_classifier", "fine_head"])        
        # MUON CONSTRAINT: Only supports 2D parameters. 
        # Biases, 1D vectors, and 4D Conv filters MUST go to AdamW.
        if (p.ndim != 2 or 
            "bias" in name or 
            is_head or 
            "pos_embed" in name or 
            "cls_token" in name or
            "norm" in name):
            adamw_params.append(p)
        else:
            muon_params.append(p)

    # Heads, biases, norms, and the 4D PatchEmbed conv go to AdamW
    opt1 = torch.optim.AdamW(adamw_params, lr=config["lr"], weight_decay=0.01)
    
    # Core 2D transformer weight matrices go to Muon
    opt2 = torch.optim.Muon(muon_params, lr=config["lr"], weight_decay=0.0)
    
    return opt1, opt2

def compute_hopf_losses(logits, pred_deltas, anchors, R_apparent_gt):
    B = logits.shape[0]
    
    # 1. Coarse Target
    target_bins = get_closest_anchor(R_apparent_gt, anchors)
    loss_coarse = F.cross_entropy(logits, target_bins)

    # Coarse Accuracy (Diagnostic)
    winning_bins = torch.argmax(logits, dim=-1)
    coarse_acc = (winning_bins == target_bins).float().mean()
    
    # 2. Fine Target (Only supervise the bin that matches the GT)
    batch_indices = torch.arange(B, device=logits.device)
    active_pred_deltas = pred_deltas[batch_indices, target_bins, :]
    target_anchors = anchors[target_bins]
    
    with torch.no_grad():
        # Relative rotation from anchor to true apparent
        R_rel_true = torch.bmm(target_anchors.transpose(1, 2), R_apparent_gt)
        true_deltas = get_gt_deltas(R_rel_true)
        
    loss_fine = F.mse_loss(active_pred_deltas, true_deltas)
    # Fine Magnitude (Diagnostic)
    fine_mag = torch.norm(active_pred_deltas, dim=-1).mean()
    return loss_coarse, loss_fine, coarse_acc, fine_mag

def predict_pose(outputs, crop_params, K_batch, full_img_size):
    """Argmax anchor -> exp-map twist -> undo the viewing-direction rotation."""
    logits, pred_deltas, anchors = outputs['logits'], outputs['deltas'], outputs['anchors']
    winning_bins = torch.argmax(logits, dim=-1)
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    R_apparent_pred = torch.bmm(anchors[winning_bins], axis_angle_to_matrix(pred_deltas[batch_indices, winning_bins, :]))
    T_pred = recover_translation_components(outputs['U'], crop_params, full_img_size, None, K_batch)
    R_pred_true = recover_true_rotation(R_apparent_pred, T_pred)
    return R_pred_true, T_pred


def summarize_errors(t_err, t_norm, r_err):
    t_rel = t_err / t_norm
    score = t_rel + torch.deg2rad(r_err)
    return {
        'E_T_mean_m': t_err.mean().item(), 'E_T_median_m': t_err.median().item(),
        'E_T_norm_mean': t_rel.mean().item(),
        'E_R_mean_deg': r_err.mean().item(), 'E_R_median_deg': r_err.median().item(),
        'score_mean': score.mean().item(),  # SPEED-style: E_T/|t| + E_R[rad]
        'acc_R_5deg': (r_err < 5).float().mean().item(), 'acc_R_10deg': (r_err < 10).float().mean().item(),
        'n': int(t_err.numel()),
    }


def validate(model, loader, device, K, full_img_size, epoch, name="val", vis_dir=None, corners_3d=None,
             amp=True, return_per_sample=False):
    model.eval()
    val_loss = 0.0
    K_batch = torch.as_tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
    t_errs, t_norms, r_errs, idxs = [], [], [], []
    vis_done = vis_dir is None

    with torch.no_grad():
        for batch in tqdm(loader, desc=name, leave=False):
            images = batch['pixel_values'].to(device, non_blocking=True)
            R_gt_true = batch['R_gt_true'].to(device)
            T_gt = batch['T_gt'].to(device)
            crop_params = batch['crop_params'].to(device)
            K_b = K_batch.expand(images.shape[0], 3, 3)
            R_apparent_gt, u_gt = make_pose_targets(R_gt_true, T_gt, crop_params, K_b, full_img_size)

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                outputs = model(images)
            outputs = {k: v.float() for k, v in outputs.items()}

            l_trans = translation_loss(outputs['U'], u_gt)
            l_coarse, l_fine, _, _ = compute_hopf_losses(outputs['logits'], outputs['deltas'], outputs['anchors'], R_apparent_gt)
            val_loss += (l_trans + l_coarse + 10.0 * l_fine).item()

            R_pred_true, T_pred = predict_pose(outputs, crop_params, K_b, full_img_size)
            t_errs.append(torch.norm(T_pred - T_gt, dim=1).cpu())
            t_norms.append(torch.norm(T_gt, dim=1).cpu())
            r_errs.append(rotation_error_deg(R_pred_true, R_gt_true).cpu())
            idxs.append(batch['idx'])

            if not vis_done:
                os.makedirs(vis_dir, exist_ok=True)
                for i in range(min(8, images.shape[0])):
                    overlay = draw_satellite_wireframe(images[i].float(), R_gt_true[i], T_gt[i], R_pred_true[i], T_pred[i],
                                                       K_batch[0], bbox=crop_params[i], corners_3d=corners_3d)
                    cv2.imwrite(os.path.join(vis_dir, f'{name}_ep{epoch:03d}_{i}.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                vis_done = True

    t_err, t_norm, r_err = torch.cat(t_errs), torch.cat(t_norms), torch.cat(r_errs)
    metrics = summarize_errors(t_err, t_norm, r_err)
    metrics['loss'] = val_loss / len(loader)
    print(f"[{name}] ep {epoch} | loss {metrics['loss']:.4f} | E_T {metrics['E_T_mean_m']:.4f} m "
          f"(med {metrics['E_T_median_m']:.4f}) | E_R {metrics['E_R_mean_deg']:.3f} deg "
          f"(med {metrics['E_R_median_deg']:.3f}) | score {metrics['score_mean']:.4f}")
    wandb.log({**{f"{name}/{k}": v for k, v in metrics.items()}, "epoch": epoch})
    if return_per_sample:
        return metrics, (torch.cat(idxs), t_err, r_err)
    return metrics


def build_dataset(cfg, split, train):
    d = cfg['data']
    return IsroPoseDataset(
        root=d['root'], split=split, camera_file=d['camera_file'], fold=d.get('fold', 1), train=train,
        crop_size=d.get('crop_size', 224), crop_scale=d.get('crop_scale', 1.6),
        scale_jitter=tuple(d.get('scale_jitter', (0.9, 1.25))), shift_jitter=d.get('shift_jitter', 0.1),
        color_aug=d.get('color_aug', True))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    repo = os.path.dirname(SRC_DIR)
    for sec, key in [('data', 'camera_file'), ('model', 'jepa_path'), ('training', 'save_dir'), ('training', 'resume')]:
        v = cfg[sec].get(key)
        if v and not os.path.isabs(v):
            cfg[sec][key] = os.path.join(repo, v)
    return cfg


def box_corners(points_3d):
    lo, hi = points_3d.min(0), points_3d.max(0)
    return np.array([[x, y, z] for z in (lo[2], hi[2]) for (x, y) in
                     [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]], dtype=np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(os.path.dirname(SRC_DIR), 'configs', 'isro.yaml'))
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--max_train_samples', type=int, default=None, help='debug: subset of the train split')
    ap.add_argument('--max_steps', type=int, default=None, help='debug: stop each epoch after N steps')
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg['training']
    if args.epochs is not None:
        tcfg['epochs'] = args.epochs
    config = {'lr': float(tcfg['lr']), 'lr_min': float(tcfg['lr_min'])}
    save_dir = tcfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    if cfg.get('logging', {}).get('wandb', False):
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg['logging'].get('project', 'SPACE-HOP-ISRO'), config=cfg, dir=save_dir)

    torch.manual_seed(tcfg.get('seed', 0)); np.random.seed(tcfg.get('seed', 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    nw = cfg['data'].get('num_workers', 8)
    bs = int(tcfg['batch_size'])
    amp = bool(tcfg.get('amp_bf16', True))

    train_dataset = build_dataset(cfg, 'train', train=True)
    if args.max_train_samples:
        train_dataset.items = train_dataset.items[:args.max_train_samples]
    val_dataset = build_dataset(cfg, 'val', train=False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
                              drop_last=True, persistent_workers=True,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True,
                            persistent_workers=True)
    K = train_dataset.K
    full_img_size = tuple(train_dataset.full_size)
    corners_3d = box_corners(train_dataset.points_3d)
    print(f"train {len(train_dataset)} | val {len(val_dataset)} | K fx={K[0,0]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    model = FastPoseViT(jepa_path=cfg['model'].get('jepa_path'), img_size=cfg['data'].get('crop_size', 224), patch_size=16,
                        backbone=cfg['model'].get('backbone', 'jepa'), grad_ckpt=cfg['model'].get('grad_ckpt', False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    K_batch = torch.as_tensor(K, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, 3, 3)
    
    optimizer_adamw, optimizer_muon = get_optimizer(model, config)
    scheduler_adam = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_adamw, T_0=10, T_mult=2, eta_min=config['lr_min']
    )
    scheduler_muon = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_muon, T_0=10, T_mult=2, eta_min=config['lr_min']
    )
    start_epoch = 0
    best_score = float('inf')
    history_path = os.path.join(save_dir, 'history.jsonl')

    resume = tcfg.get('resume')
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"resume checkpoint not found: {resume}")
        print(f"Loading checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw_state_dict'])
        optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])
        scheduler_adam.load_state_dict(checkpoint['scheduler_adam_state_dict'])
        scheduler_muon.load_state_dict(checkpoint['scheduler_muon_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_score = checkpoint.get('best_score', best_score)
        print(f"Resumed from Epoch {start_epoch}. Previous best val score: {best_score:.4f}")

    def save(path, epoch, extra=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_adamw_state_dict': optimizer_adamw.state_dict(),
            'scheduler_adam_state_dict': scheduler_adam.state_dict(),
            'optimizer_muon_state_dict': optimizer_muon.state_dict(),
            'scheduler_muon_state_dict': scheduler_muon.state_dict(),
            'best_score': best_score,
            'config': cfg,
            **(extra or {}),
        }, path)

    print(f"Starting Training on {device}...")
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_rot_loss = 0.0
        epoch_trans_loss = 0.0
        epoch_acc = 0.0
        n_steps = 0
        t0 = time.time()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=30)
        
        for batch_idx, batch in enumerate(loop):
            if args.max_steps and batch_idx >= args.max_steps:
                break
            images = batch["pixel_values"].to(device, non_blocking=True)
            R_gt_true = batch["R_gt_true"].to(device, non_blocking=True)
            T_gt = batch["T_gt"].to(device, non_blocking=True)
            crop_params = batch["crop_params"].to(device, non_blocking=True)
            R_apparent_gt, U_gt = make_pose_targets(R_gt_true, T_gt, crop_params, K_batch, full_img_size)

            optimizer_adamw.zero_grad()
            optimizer_muon.zero_grad()
            
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                outputs = model(images)
            U_pred = outputs["U"].float()
            logits = outputs["logits"].float()
            pred_deltas = outputs["deltas"].float()
            anchors = outputs["anchors"]
            
            loss_trans = translation_loss(U_pred, U_gt)
            loss_coarse, loss_fine, coarse_acc, fine_mag = compute_hopf_losses(logits, pred_deltas, anchors, R_apparent_gt)
            
            loss_rot = loss_coarse + (10.0 * loss_fine) # Weight the fine loss
            loss = loss_rot + loss_trans
            
            loss.backward()
            optimizer_adamw.step()
            optimizer_muon.step()
            
            epoch_loss += loss.item()
            epoch_rot_loss += loss_rot.item()
            epoch_trans_loss += loss_trans.item()
            epoch_acc += coarse_acc.item()
            n_steps += 1
            
            loop.set_postfix(loss=loss.item())
            
            if batch_idx % 50 == 0:
                wandb.log({
                    "train/loss_total": loss.item(),
                    "train/loss_trans": loss_trans.item(),
                    "train/loss_coarse_rot": loss_coarse.item(),
                    "train/loss_fine_rot": loss_fine.item(),
                    "diagnostics/coarse_bin_accuracy": coarse_acc.item(),
                    "diagnostics/fine_twist_magnitude": fine_mag.item(),
                    "lr/adamw": optimizer_adamw.param_groups[0]['lr'],
                    "lr/muon": optimizer_muon.param_groups[0]['lr']
                })
        train_time = time.time() - t0
        vis_every = tcfg.get('vis_every', 10)
        vis_dir = os.path.join(save_dir, 'vis') if (epoch + 1) % vis_every == 0 or epoch == start_epoch else None
        val_metrics = validate(model, val_loader, device, K, full_img_size, epoch + 1, name="val",
                               vis_dir=vis_dir, corners_3d=corners_3d, amp=amp)
        scheduler_adam.step()
        scheduler_muon.step()
        avg_train_loss = epoch_loss / n_steps
        avg_rot_loss = epoch_rot_loss / n_steps
        avg_trans_loss = epoch_trans_loss / n_steps
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Rot: {avg_rot_loss:.4f}, Trans: {avg_trans_loss:.4f}) "
              f"| coarse acc {epoch_acc / n_steps:.3f} | {train_time:.0f}s")
        wandb.log({
            "train/epoch_total_loss": avg_train_loss,
            "train/epoch_rot_loss": avg_rot_loss,
            "train/epoch_trans_loss": avg_trans_loss,
            "epoch": epoch+1,
            "lr_adamw": optimizer_adamw.param_groups[0]['lr'],
            "lr_muon": optimizer_muon.param_groups[0]['lr']
        })
        with open(history_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'train_rot': avg_rot_loss,
                                'train_trans': avg_trans_loss, 'train_coarse_acc': epoch_acc / n_steps,
                                'lr': optimizer_adamw.param_groups[0]['lr'], 'train_time_s': train_time,
                                **{f'val_{k}': v for k, v in val_metrics.items()}}) + '\n')

        # Model selection on the val fold's SPEED-style pose score (lower is better)
        if val_metrics['score_mean'] < best_score:
            best_score = val_metrics['score_mean']
            save(os.path.join(save_dir, 'best_model.pth'), epoch, {'val_metrics': val_metrics})
            print(f"Saved best model (val score {best_score:.4f})")
        save(os.path.join(save_dir, 'last.pth'), epoch)
    wandb.finish()
