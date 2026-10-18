"""
Coarse-to-fine Hopf head for SPACE-HOP.

Level 0 is the original head (3072 Hopf anchors + in-cell twist). Levels 1..L re-classify the apparent rotation
on a local cubic grid of camera-frame offsets around the previous estimate, R_c = exp(w_c) @ R_center, with the
grid spacing shrinking every level. Each candidate is a query token that cross-attends to the ViT patch tokens and
is scored; the per-level output is the probability-weighted mean offset (sub-grid precision).

The head runs in fp32: bf16 has ~0.4 deg resolution on rotation-matrix entries, which is coarser than the target.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z],
                       dim=-1).reshape(*w.shape[:-1], 3, 3)


def so3_exp(w):
    """Rodrigues with a Taylor branch, any batch shape (..., 3) -> (..., 3, 3)."""
    th2 = (w * w).sum(-1, keepdim=True)[..., None]
    th = th2.clamp_min(1e-30).sqrt()
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-30))
    K = hat(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + a * K + b * (K @ K)


def so3_log(R):
    """atan2-based log map, accurate for small angles (valid away from pi). (..., 3, 3) -> (..., 3)."""
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = v.norm(dim=-1, keepdim=True) / 2
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).unsqueeze(-1)
    th = torch.atan2(s, c)
    k = torch.where(s < 1e-7, 0.5 + th * th / 12, th / (2 * s.clamp_min(1e-30)))
    return v * k


def geodesic_deg(R1, R2):
    """Geodesic distance in degrees, computed in float64 so 0.01-deg errors are resolved."""
    return torch.rad2deg(so3_log(R1.double().transpose(-1, -2) @ R2.double()).norm(dim=-1))


class LocalLevelScorer(nn.Module):
    def __init__(self, embed_dim=768, n_levels=3, n_layers=2, heads=8):
        super().__init__()
        self.cand = nn.Sequential(nn.Linear(3 + 9, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.level_emb = nn.Embedding(n_levels, embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'n1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, heads, batch_first=True),
                'n2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim)),
            }))
        self.out = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)  # start at the prior (identity)

    def forward(self, patches, grid_unit, R_center, level):
        """patches (B,P,D); grid_unit (N,3) offsets in units of the level spacing; R_center (B,3,3) -> logits (B,N)."""
        B, N = R_center.shape[0], grid_unit.shape[0]
        feat = torch.cat([grid_unit.unsqueeze(0).expand(B, N, 3),
                          R_center.reshape(B, 1, 9).expand(B, N, 9)], -1)
        q = self.cand(feat) + self.level_emb.weight[level]
        kv = self.kv_norm(patches)
        for L in self.layers:
            h = L['n1'](q)
            q = q + L['attn'](h, kv, kv, need_weights=False)[0]
            q = q + L['mlp'](L['n2'](q))
        return self.out(q).squeeze(-1)


class HierHopfHead(nn.Module):
    def __init__(self, embed_dim=768, spacings_deg=(1.25, 0.4, 0.13), half_width=3, sigma=0.5, n_layers=2, prior_sigma=1.5):
        super().__init__()
        r = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
        g = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), -1).reshape(-1, 3)
        self.register_buffer('grid_unit', g)
        self.spacings = [float(s) for s in spacings_deg]
        self.half_width = half_width
        self.sigma = sigma  # soft-target width in grid units
        self.scorer = LocalLevelScorer(embed_dim, len(self.spacings), n_layers)
        # Fixed prior favouring the centre: with the zero-initialised scorer every level starts as the identity.
        self.register_buffer('prior', -(g ** 2).sum(-1) / (2 * prior_sigma ** 2))

    def level(self, patches, R_center, l):
        return self.scorer(patches, self.grid_unit, R_center, l) + self.prior

    def decode(self, logits, R_center, l):
        """Probability-weighted mean offset over the 3x3x3 neighbourhood of the argmax -> refined rotation."""
        p = logits.softmax(-1)
        best = self.grid_unit[logits.argmax(-1)]                                   # (B,3)
        near = ((self.grid_unit.unsqueeze(0) - best.unsqueeze(1)).abs() <= 1.0).all(-1)  # (B,N)
        w = p * near
        off = (w.unsqueeze(-1) * self.grid_unit).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        w_rad = torch.deg2rad(off * self.spacings[l])
        return so3_exp(w_rad) @ R_center

    def loss(self, logits, R_center, R_gt, l):
        """Soft cross-entropy against a Gaussian over grid points centred on the GT offset."""
        w_gt = torch.rad2deg(so3_log(R_gt @ R_center.transpose(-1, -2))) / self.spacings[l]   # grid units
        inside = (w_gt.abs() <= self.half_width + 0.5).all(-1)
        w_c = w_gt.clamp(-self.half_width, self.half_width)
        d2 = ((self.grid_unit.unsqueeze(0) - w_c.unsqueeze(1)) ** 2).sum(-1)
        target = F.softmax(-d2 / (2 * self.sigma ** 2), -1)
        ce = -(target * logits.log_softmax(-1)).sum(-1)
        return ce.mean(), inside.float().mean()

    def forward(self, patches, R0, R_gt=None, chain_prob=0.5):
        """Run all levels from the level-0 estimate R0. With R_gt (training) also return per-level losses.
        Training centres: level 1 starts from the (detached) level-0 prediction; deeper levels use the chained
        previous output with prob chain_prob, else GT perturbed by N(0, spacing^2) per axis."""
        patches = patches.float()
        R = R0.float().detach()
        outs, losses, inside = [R], [], []
        for l in range(len(self.spacings)):
            if R_gt is not None and l > 0:
                synth = so3_exp(torch.deg2rad(torch.randn_like(R[:, 0]) * self.spacings[l])) @ R_gt
                use_chain = (torch.rand(R.shape[0], device=R.device) < chain_prob)[:, None, None]
                R = torch.where(use_chain, R, synth)
            logits = self.level(patches, R, l)
            if R_gt is not None:
                lo, ins = self.loss(logits, R, R_gt, l)
                losses.append(lo); inside.append(ins)
            R = self.decode(logits, R, l).detach() if R_gt is not None else self.decode(logits, R, l)
            outs.append(R)
        return outs, losses, inside


def delta_rotation(T):
    """Rotation taking the viewing direction T/|T| onto +z (same convention as geometry_utils.compute_delta_rotation),
    dtype-preserving so it can run in float64."""
    t = F.normalize(T, dim=-1)
    ez = torch.zeros_like(t); ez[..., 2] = 1
    c = torch.cross(t, ez, dim=-1)
    s = c.norm(dim=-1, keepdim=True)
    ang = torch.atan2(s, (t * ez).sum(-1, keepdim=True))
    return so3_exp(c / s.clamp_min(1e-30) * ang)
