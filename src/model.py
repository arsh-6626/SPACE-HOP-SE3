import torch
import torch.nn as nn
import os
import sys

# Use the repo's ijepa package so the pose backbone is the exact ViT used for JEPA pretraining
jepa_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ijepa')
if jepa_root not in sys.path:
    sys.path.insert(0, jepa_root)

from src.models.vision_transformer import vit_base
from hopf_grid import generate_hopf_so3_grid

class FastPoseViT(nn.Module):
    def __init__(self, jepa_path=None, img_size=224, patch_size=16, backbone='jepa', grad_ckpt=False):
        super(FastPoseViT, self).__init__()
        
        # 1. BACKBONE: the repo's JEPA ViT-B/16, or a timm model ('timm:<name>', e.g. DINOv3 ViT-B/16)
        self.n_prefix = 1
        if backbone.startswith('timm:'):
            import timm
            self.vit = timm.create_model(backbone[5:], pretrained=True, img_size=img_size, num_classes=0)
            self.n_prefix = self.vit.num_prefix_tokens  # CLS + register tokens
            self.timm = True
            print(f"[*] timm backbone {backbone[5:]} at {img_size}px, {self.n_prefix} prefix tokens")
        else:
            self.vit = vit_base(img_size=[img_size], patch_size=patch_size)
            self.timm = False
        if grad_ckpt:
            assert self.timm, 'gradient checkpointing is only wired up for timm backbones'
            self.vit.set_grad_checkpointing(True)
        embed_dim = 768 
        
        # Load custom JEPA weights. A path that is given but missing is an error, not a silent random init.
        if self.timm:
            pass
        elif jepa_path:
            if not os.path.isfile(jepa_path):
                raise FileNotFoundError(f"JEPA checkpoint not found: {jepa_path}")
            print(f"[*] Loading custom JEPA backbone from: {jepa_path}")
            checkpoint = torch.load(jepa_path, map_location='cpu', weights_only=False)
            
            if 'target_encoder' in checkpoint: state_dict = checkpoint['target_encoder']
            elif 'encoder' in checkpoint: state_dict = checkpoint['encoder']
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            else: state_dict = checkpoint
                
            clean_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '').replace('target_encoder.', '').replace('encoder.', '')
                clean_dict[new_k] = v
                
            # pos_embed is a fixed 2D sincos table; the model regenerates it for its own img_size
            if 'pos_embed' in clean_dict and clean_dict['pos_embed'].shape != self.vit.pos_embed.shape:
                print(f"[*] pos_embed {tuple(clean_dict['pos_embed'].shape)} -> {tuple(self.vit.pos_embed.shape)}: using sincos for img_size={img_size}")
                clean_dict.pop('pos_embed')
            msg = self.vit.load_state_dict(clean_dict, strict=False)
            if [k for k in msg.missing_keys if k != 'pos_embed'] or ('pos_embed' in msg.missing_keys and 'pos_embed' in clean_dict):
                raise RuntimeError(f"JEPA checkpoint is missing backbone keys: {msg.missing_keys}")
            print(f"[*] JEPA Load Message: {msg}")
        else:
            print("[!] No JEPA checkpoint given: backbone is randomly initialised")
        
        # Generate the discrete Hopf anchors
        anchors = generate_hopf_so3_grid(num_points=256, num_rolls=12, device='cpu')
        self.register_buffer('anchors', anchors)
        self.K_bins = anchors.shape[0]

        # -------------------------------------------------------------------
        # 2. THE DITTO FASTPOSE HEAD (Predicts 9D)
        # -------------------------------------------------------------------
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, 9)
        )
        nn.init.xavier_uniform_(self.pose_head[0].weight)
        nn.init.zeros_(self.pose_head[0].bias)

        # -------------------------------------------------------------------
        # 3. COARSE: CROSS-ATTENTION WITH CLS
        # -------------------------------------------------------------------
        # CLS token queries the spatial patches to build a rotation-aware feature
        self.coarse_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.coarse_classifier = nn.Linear(embed_dim, self.K_bins)
        
        # -------------------------------------------------------------------
        # 4. FINE: BIN-CONSTRAINED TWIST
        # -------------------------------------------------------------------
        self.fine_head = nn.Linear(embed_dim, self.K_bins * 3)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)

    def forward(self, x):
        # Raw JEPA outputs: (B, 197, D)
        outputs = self.vit.forward_features(x) if self.timm else self.vit(x)
        
        # Split CLS from patches for Cross-Attention (register tokens, if any, are dropped)
        cls_token = outputs[:, 0:1, :]               # Shape: (B, 1, D)
        patch_tokens = outputs[:, self.n_prefix:, :] # Shape: (B, N_patches, D)
        
        # --- TRANSLATION (FastPose Ditto) ---
        # Predict all 9, slice out the first 3 for Translation $U$
        pose_9d = self.pose_head(cls_token.squeeze(1))
        U_pred = pose_9d[:, :3] 
        
        # --- COARSE (Cross-Attention) ---
        # CLS token acts as Query; Patches act as Key/Value
        attn_out, _ = self.coarse_cross_attn(query=cls_token, key=patch_tokens, value=patch_tokens)
        attn_out = attn_out.squeeze(1) # Shape back to (B, D)
        logits = self.coarse_classifier(attn_out)
        
        # --- FINE (Bin Constrained via Skip Connection) ---
        deltas = self.fine_head(cls_token.squeeze(1)).view(-1, self.K_bins, 3)
        
        return {
            'U': U_pred,
            'logits': logits,
            'deltas': deltas,
            'anchors': self.anchors,
            'patches': patch_tokens,
        }