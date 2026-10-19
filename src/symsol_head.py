"""
Hierarchical SO(3) density head for the SYMSOL distribution benchmark.

The distribution is the strict-hierarchy Hopf-HEALPix tree of so3_nested.py: a categorical over the 72 level-0 cells
and, at every node, a categorical over its 8 children. The density at a rotation R is

    p(R) = P(finest cell of R) * N(L) / V_SO3,    log P(cell) = logp0[cell0] + sum_l logp_child[node_l -> child_l],

so the log-likelihood of any rotation is read off by scoring only the L+1 nodes on its path -- deep grids are cheap.

The network never touches healpy: the data pipeline (symsol_data.py, using so3_nested.py) precomputes, per sample,
the GT path (cell0, child indices) and the geometry each child scorer needs -- the 8 children's offsets from their
parent, normalised by the cell size, and the parent rotation for conditioning. The head is pure torch on those.

Training (IPDF / Hsiao protocol): NLL of the single rendered rotation down its path. The symmetry set is used only
at test time, to average the log-likelihood -- training on it would hand the model information the baselines lack.

Runs in fp32 (bf16 is too coarse for the finest cells).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encode(x, n_freq):
    """(..., D) -> (..., D + D*2*n_freq): x and sin/cos at frequencies pi * 2^k, k=0..n_freq-1 (IPDF positional enc)."""
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * torch.pi
    xf = x.unsqueeze(-1) * freqs                                   # (..., D, n_freq)
    enc = torch.cat([xf.sin(), xf.cos()], -1).flatten(-2)         # (..., D*2*n_freq)
    return torch.cat([x, enc], -1)


class _CrossAttn(nn.Module):
    """A stack of (query self-free) cross-attention blocks: queries attend to the frozen patch tokens, then an MLP.
    Mirrors hier_head.LocalLevelScorer; the final linear is zero-initialised so every node starts uniform."""
    def __init__(self, embed_dim, n_layers=2, heads=8):
        super().__init__()
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(nn.ModuleDict({
            'n1': nn.LayerNorm(embed_dim),
            'attn': nn.MultiheadAttention(embed_dim, heads, batch_first=True),
            'n2': nn.LayerNorm(embed_dim),
            'mlp': nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim)),
        }) for _ in range(n_layers))
        self.out = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, q, patches):
        kv = self.kv_norm(patches)
        for L in self.layers:
            q = q + L['attn'](L['n1'](q), kv, kv, need_weights=False)[0]
            q = q + L['mlp'](L['n2'](q))
        return self.out(q).squeeze(-1)                            # (B, Nq)


class SymsolDensityHead(nn.Module):
    """Level-0: 72 learned queries cross-attend to patches -> 72 logits. Children: one shared scorer whose 8 queries
    are built from each child's normalised offset from the parent, the Fourier-encoded parent rotation, and a level
    embedding -> 8 logits. All log-softmax, so the tree is a normalised distribution by construction."""
    def __init__(self, embed_dim=768, max_level=5, n_freq=3, n_layers=2, heads=8):
        super().__init__()
        self.max_level = max_level
        self.n_freq = n_freq
        self.l0_query = nn.Embedding(72, embed_dim)
        self.l0_attn = _CrossAttn(embed_dim, n_layers, heads)
        self.level_emb = nn.Embedding(max_level, embed_dim)
        off_dim = 3 + 6 * n_freq                                  # fourier_encode(offset[3])
        par_dim = 9 + 18 * n_freq                                 # fourier_encode(parent flat[9])
        self.child_proj = nn.Sequential(nn.Linear(off_dim + par_dim, embed_dim), nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim))
        self.child_attn = _CrossAttn(embed_dim, n_layers, heads)

    def logp0(self, patches):
        """patches (B,P,D) -> (B,72) log-softmax over the level-0 cells."""
        B = patches.shape[0]
        q = self.l0_query.weight.unsqueeze(0).expand(B, -1, -1)
        return self.l0_attn(q, patches).log_softmax(-1)

    def child_logp(self, patches, child_off, parent_enc, level):
        """Score the 8 children of one node per sample.
        patches (B,P,D); child_off (B,8,3) normalised offsets; parent_enc (B,9) parent flat; level int -> (B,8)."""
        B = patches.shape[0]
        off = fourier_encode(child_off, self.n_freq)                              # (B,8,off_dim)
        par = fourier_encode(parent_enc, self.n_freq).unsqueeze(1).expand(-1, 8, -1)
        q = self.child_proj(torch.cat([off, par], -1)) + self.level_emb.weight[level]
        return self.child_attn(q, patches).log_softmax(-1)                        # (B,8)

    def path_loglik(self, patches, cell0, child_idx, child_off, parent_enc, level):
        """log P(finest cell) for one path per sample (used for both NLL and eval, minus the density constant).
        cell0 (B,); child_idx (B,level); child_off (B,level,8,3); parent_enc (B,level,9) -> (B,) log-prob."""
        lp = self.logp0(patches).gather(1, cell0[:, None]).squeeze(1)
        for l in range(level):
            clp = self.child_logp(patches, child_off[:, l], parent_enc[:, l], l)  # (B,8)
            lp = lp + clp.gather(1, child_idx[:, l:l + 1]).squeeze(1)
        return lp

    def forward(self, patches, batch, level):
        """Training NLL of the single GT rotation. batch holds cell0/child_idx/child_off/parent_enc (see symsol_data).
        Returns (loss, logp_cell) with logp_cell detached for logging."""
        lp = self.path_loglik(patches, batch['cell0'], batch['child_idx'], batch['child_off'],
                              batch['parent_enc'], level)
        return (-lp).mean(), lp.detach()
