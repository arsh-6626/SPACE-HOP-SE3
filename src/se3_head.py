"""
Hierarchical SE(3) density head: the strict-hierarchy SO(3) rotation tree x R^3 translation octree of se3_tree.py.

The distribution is a tree-structured softmax over the joint grid of se3_tree.py: a categorical over the 72 level-0
rotation cells (the translation root is a single cell), and, at every joint node, a categorical over its children --
64-way (rot_child * 8 + trans_oct) while BOTH halves still subdivide, 8-way once one half has reached its depth.
The density at a pose (R,t) inside the affine translation box t = t_hat + A g (g in the unit cube) is

    p(R,t) = P(finest joint cell) * N / (V_SO3 * V_box),   N = 72 * 8^L_rot * 8^L_trans,   V_SO3 = pi^2,

so the log-likelihood of any pose is read off by scoring only the L+1 nodes on its joint path (L = max(L_rot,L_trans)).

This head generalises symsol_head.SymsolDensityHead. Level 0 is unchanged (72 learned queries cross-attend to the
frozen patch tokens). The one shared child scorer now builds its child queries from a 6-D offset
[rotation child offset (3), translation octant offset (3)] and a 12-D parent code [parent rotation flat (9),
parent translation-cell centre (3)], plus a level embedding; it emits one logit per child query (64, or 8 on a tail
level). Every categorical is a log-softmax, so the tree is a normalised SE(3) density by construction, and the output
linear is zero-initialised so the network starts exactly at the uniform floor -ln(pi^2 * V_box).

The geometry each scorer needs (child offsets, parent codes, GT child indices) is precomputed by build_geometry()
from se3_tree / so3_nested; the head is pure torch on those tensors and never touches healpy. Runs in fp32.

    python src/se3_head.py checks      # (A) uniform-floor forward pass, (B) overfit 32 samples
"""
import numpy as np
import torch
import torch.nn as nn

import so3_nested as so3
import se3_tree as se3


def fourier_encode(x, n_freq):
    """(..., D) -> (..., D + D*2*n_freq): x and sin/cos at frequencies pi * 2^k (IPDF positional enc)."""
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * torch.pi
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([xf.sin(), xf.cos()], -1).flatten(-2)
    return torch.cat([x, enc], -1)


# The 8 octants of any cube, o = bx + 2by + 4bz; child-centre offset from the parent centre, in units of the child
# cell size (so each component is +-0.5). Constant across the whole octree by self-similarity.
_OCT_BITS = np.array([[(o & 1), (o >> 1) & 1, (o >> 2) & 1] for o in range(8)], np.float32)
TRANS_OCT_OFF = _OCT_BITS - 0.5                                      # (8,3) in {-0.5,+0.5}


class _CrossAttn(nn.Module):
    """Query tokens cross-attend to the frozen patch tokens, then an MLP; final linear zero-init so nodes start
    uniform. Identical to symsol_head._CrossAttn."""
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
        return self.out(q).squeeze(-1)                              # (B, Nq)


class Se3DensityHead(nn.Module):
    """Level-0: 72 learned queries -> 72 logits (rotation root). Children: one shared scorer whose child queries are
    built from the 6-D [rot offset, trans offset] and the 12-D [parent rot flat, parent trans centre] codes plus a
    level embedding -> one logit per child. All log-softmax => a normalised SE(3) distribution by construction."""
    def __init__(self, embed_dim=768, L_rot=3, L_trans=3, n_freq=3, n_layers=2, heads=8):
        super().__init__()
        self.L_rot, self.L_trans = L_rot, L_trans
        self.max_level = max(L_rot, L_trans)
        self.n_freq = n_freq
        self.l0_query = nn.Embedding(72, embed_dim)
        self.l0_attn = _CrossAttn(embed_dim, n_layers, heads)
        self.level_emb = nn.Embedding(self.max_level, embed_dim)
        off_dim = 6 + 12 * n_freq                                   # fourier_encode([rot_off(3), trans_off(3)])
        par_dim = 12 + 24 * n_freq                                  # fourier_encode([parent_rot(9), parent_trans(3)])
        self.child_proj = nn.Sequential(nn.Linear(off_dim + par_dim, embed_dim), nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim))
        self.child_attn = _CrossAttn(embed_dim, n_layers, heads)

    def logp0(self, patches):
        """patches (B,P,D) -> (B,72) log-softmax over the level-0 rotation cells."""
        B = patches.shape[0]
        q = self.l0_query.weight.unsqueeze(0).expand(B, -1, -1)
        return self.l0_attn(q, patches).log_softmax(-1)

    def child_logp(self, patches, child_off6, parent12, level):
        """Score the K children of one joint node per sample.
        child_off6 (B,K,6); parent12 (B,12); level is the 0-based joint-level index -> (B,K) log-softmax."""
        off = fourier_encode(child_off6, self.n_freq)                               # (B,K,off_dim)
        par = fourier_encode(parent12, self.n_freq).unsqueeze(1).expand(-1, child_off6.shape[1], -1)
        q = self.child_proj(torch.cat([off, par], -1)) + self.level_emb.weight[level]
        return self.child_attn(q, patches).log_softmax(-1)                          # (B,K)

    def path_loglik(self, patches, cell0, levels):
        """log P(finest joint cell) for one path per sample. `levels` is a list over joint levels l=1..L, each a dict
        with tensors child_off6 (B,K,6), parent12 (B,12), target (B,) -- see build_geometry()."""
        lp = self.logp0(patches).gather(1, cell0[:, None]).squeeze(1)
        for l, lv in enumerate(levels):
            clp = self.child_logp(patches, lv['child_off6'], lv['parent12'], l)     # (B,K)
            lp = lp + clp.gather(1, lv['target'][:, None]).squeeze(1)
        return lp

    def forward(self, patches, batch):
        """Training NLL of the single GT pose. batch = {'cell0', 'levels'} from build_geometry().
        Returns (loss, logp_cell) with logp_cell detached for logging."""
        lp = self.path_loglik(patches, batch['cell0'], batch['levels'])
        return (-lp).mean(), lp.detach()


# ---------------------------------------------------------------- geometry builder (numpy; mirrors symsol_data) ----

def build_geometry(mats, gs, L_rot, L_trans):
    """(mats (M,3,3) rotations, gs (M,3) normalized-cube translations in [-1/2,1/2]^3) -> the per-path tensors the
    head consumes: cell0 (M,) and a list over joint levels l=1..L of dicts {kind, child_off6 (M,K,6), parent12 (M,12),
    target (M,)}, K=64 on 'joint64' levels and 8 on 'rot8'/'trans8' tails. All numpy; torchify with to_torch()."""
    from scipy.spatial.transform import Rotation as Rot
    mats = np.asarray(mats, np.float64).reshape(-1, 3, 3)
    gs = np.asarray(gs, np.float64).reshape(-1, 3)
    M = mats.shape[0]
    L = max(L_rot, L_trans)

    # ---- rotation path: node cells, child indices, child offsets, parent matrices (as in symsol_data.path_geometry)
    rot_cells = so3.rotations_to_cells(mats, L_rot) if L_rot > 0 else so3.rotations_to_cells(mats, 0)
    rot_node = np.empty((M, L_rot + 1), np.int64)
    rot_child_idx = np.empty((M, L_rot), np.int64)
    for i, c in enumerate(rot_cells):
        c0, ch = so3.cell_to_path(c, L_rot)
        rot_node[i] = so3.path_node_cells(c0, ch)
        rot_child_idx[i] = ch
    cell0 = rot_node[:, 0].copy()
    rot_child_off = np.zeros((M, L_rot, 8, 3), np.float32)
    rot_parent_flat = np.zeros((M, L_rot, 9), np.float32)
    for l in range(L_rot):
        par = rot_node[:, l]
        chf = so3.parent_to_children_cells(par, l)                              # (M,8)
        R_par = so3.cell_index_to_matrix(par, l)                                # (M,3,3)
        R_ch = so3.cell_index_to_matrix(chf, l + 1)                             # (M,8,3,3)
        rel = np.einsum('mcij,mkj->mcik', R_ch, R_par)                          # R_child @ R_parent^T
        rv = Rot.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(M, 8, 3)
        rot_child_off[:, l] = (rv / so3.cell_size_rad(l + 1)).astype(np.float32)
        rot_parent_flat[:, l] = R_par.reshape(M, 9).astype(np.float32)
    # finest rotation matrix (parent code for trans-only tail levels)
    R_fin_flat = so3.cell_index_to_matrix(rot_node[:, L_rot], L_rot).reshape(M, 9).astype(np.float32)

    # ---- translation path: octants, and the parent-cell centre at each translation level
    trans_octs = np.empty((M, L_trans), np.int64)
    trans_parent_ctr = np.zeros((M, L_trans, 3), np.float32)                     # centre of the level-m parent cell
    trans_fin_ctr = np.zeros((M, 3), np.float32)                                 # centre of the finest trans cell
    for i in range(M):
        octs = se3.points_to_trans_path(gs[i], L_trans)
        trans_octs[i] = octs
        for m in range(L_trans):
            pcell = se3._trans_path_to_cell(octs[:m], m)                         # parent cell at level m
            trans_parent_ctr[i, m] = se3.trans_cell_center(np.array([pcell]), m)[0]
        if L_trans > 0:
            fcell = se3._trans_path_to_cell(octs, L_trans)
            trans_fin_ctr[i] = se3.trans_cell_center(np.array([fcell]), L_trans)[0]

    oct_off = TRANS_OCT_OFF                                                      # (8,3) constant

    # ---- assemble per joint level
    levels = []
    for l in range(1, L + 1):
        kind = se3.joint_level_kind(l, L_rot, L_trans)
        if kind == 'joint64':
            ri = l - 1                                                           # rotation sub-level index
            ti = l - 1                                                           # translation sub-level index
            roff = np.repeat(rot_child_off[:, ri][:, :, None, :], 8, axis=2)     # (M,8rc,8to,3)
            toff = np.broadcast_to(oct_off[None, None, :, :], (M, 8, 8, 3))      # (M,8rc,8to,3)
            child_off6 = np.concatenate([roff, toff], -1).reshape(M, 64, 6).astype(np.float32)
            parent12 = np.concatenate([rot_parent_flat[:, ri], trans_parent_ctr[:, ti]], -1).astype(np.float32)
            target = (rot_child_idx[:, ri] * 8 + trans_octs[:, ti]).astype(np.int64)
        elif kind == 'rot8':
            ri = l - 1
            roff = rot_child_off[:, ri]                                          # (M,8,3)
            toff = np.zeros((M, 8, 3), np.float32)
            child_off6 = np.concatenate([roff, toff], -1).astype(np.float32)
            parent12 = np.concatenate([rot_parent_flat[:, ri], trans_fin_ctr], -1).astype(np.float32)
            target = rot_child_idx[:, ri].astype(np.int64)
        else:  # trans8
            ti = l - 1
            roff = np.zeros((M, 8, 3), np.float32)
            toff = np.broadcast_to(oct_off[None, :, :], (M, 8, 3))
            child_off6 = np.concatenate([roff, toff], -1).astype(np.float32)
            parent12 = np.concatenate([R_fin_flat, trans_parent_ctr[:, ti]], -1).astype(np.float32)
            target = trans_octs[:, ti].astype(np.int64)
        levels.append({'kind': kind, 'child_off6': child_off6, 'parent12': parent12, 'target': target})
    return {'cell0': cell0.astype(np.int64), 'levels': levels}


def to_torch(geom, device):
    out = {'cell0': torch.as_tensor(geom['cell0'], device=device),
           'levels': [{'kind': lv['kind'],
                       'child_off6': torch.as_tensor(lv['child_off6'], device=device),
                       'parent12': torch.as_tensor(lv['parent12'], device=device),
                       'target': torch.as_tensor(lv['target'], device=device)} for lv in geom['levels']]}
    return out


# ---------------------------------------------------------------- network checks (A: uniform floor, B: overfit) ----

def _check_uniform(device, A, n_freq=3, embed_dim=128):
    """Check A: a zero-init network on random patches scores the exact uniform floor at several (L_rot,L_trans).
    Tests the whole forward pass (not just the grid): density == -ln(pi^2 * V_box) and NLL == ln N."""
    Vb = se3.box_volume(A); floor = -np.log(se3.V_SO3 * Vb)
    rng = np.random.RandomState(0)
    worst = 0.0
    for L_rot, L_trans in [(0, 0), (1, 1), (2, 2), (3, 2), (2, 3), (3, 3)]:
        head = Se3DensityHead(embed_dim=embed_dim, L_rot=L_rot, L_trans=L_trans, n_freq=n_freq,
                              n_layers=1, heads=4).to(device).eval()
        M = 512
        from scipy.spatial.transform import Rotation as Rot
        mats = Rot.random(M, random_state=rng).as_matrix()
        gs = rng.uniform(-0.5, 0.5, size=(M, 3))
        geom = to_torch(build_geometry(mats, gs, L_rot, L_trans), device)
        patches = torch.randn(M, 16, embed_dim, device=device)
        with torch.no_grad():
            lp_cell = head.path_loglik(patches, geom['cell0'], geom['levels'])      # log P(cell)
        N = se3.n_joint_cells(L_rot, L_trans)
        dens = lp_cell.cpu().numpy() + np.log(N / (se3.V_SO3 * Vb))                  # log density
        nll = -lp_cell.cpu().numpy()
        d_dens = float(np.max(np.abs(dens - floor)))
        d_nll = float(np.max(np.abs(nll - np.log(N))))
        worst = max(worst, d_dens, d_nll)
        print(f"  (L_rot={L_rot},L_trans={L_trans}) N={N:>10d}  density dev {d_dens:.2e}  "
              f"NLL==lnN dev {d_nll:.2e}")
    assert worst < 1e-4, f"uniform-floor forward pass off by {worst:.2e}"
    print(f"  CHECK A OK: zero-init network scores uniform floor -ln(pi^2*V_box) = {floor:.4f} "
          f"(max dev {worst:.1e}, V_box={Vb:.3f})")


def _check_overfit(device, A, L_rot=3, L_trans=3, M=32, steps=400, n_freq=3, embed_dim=128):
    """Check B: overfit M samples with fixed random patches and random (R,t) labels. The NLL must fall well below the
    uniform floor ln N, proving gradients flow through the 64-way joint nodes and the 8-way tails."""
    Vb = se3.box_volume(A)
    rng = np.random.RandomState(1)
    from scipy.spatial.transform import Rotation as Rot
    mats = Rot.random(M, random_state=rng).as_matrix()
    gs = rng.uniform(-0.5, 0.5, size=(M, 3))
    geom = to_torch(build_geometry(mats, gs, L_rot, L_trans), device)
    patches = torch.randn(M, 16, embed_dim, device=device)                          # fixed "frozen ViT" tokens
    head = Se3DensityHead(embed_dim=embed_dim, L_rot=L_rot, L_trans=L_trans, n_freq=n_freq,
                          n_layers=2, heads=4).to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    N = se3.n_joint_cells(L_rot, L_trans)
    floor = float(np.log(N))
    first = None
    for s in range(steps):
        opt.zero_grad()
        loss, _ = head(patches, geom)
        loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    last = loss.item()
    # grid ceiling: the sharpest a piecewise-constant density can be is one cell -> NLL floor is 0 for log P(cell);
    # report NLL relative to the uniform ln N to show real concentration.
    print(f"  CHECK B OK: overfit {M} samples, L=({L_rot},{L_trans}), N={N}  "
          f"NLL(log P cell) {first:.3f} (uniform ln N={floor:.3f}) -> {last:.3f} after {steps} steps  "
          f"(drop {first - last:.3f} nats)")
    assert last < first - 1.0, f"overfit did not reduce NLL enough: {first:.3f} -> {last:.3f}"
    assert last < floor - 1.0, f"overfit NLL {last:.3f} not below uniform floor {floor:.3f}"


def checks():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    A = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])               # SYMSOL-T box: t in [-1,1]^3
    print(f"se3_head checks on {device} | box A=diag(2,2,2), V_box={se3.box_volume(A):.3f}")
    print("Check A -- zero-init forward pass hits the uniform floor:")
    _check_uniform(device, A)
    print("Check B -- overfit concentrates mass (gradients flow through 64-way + tails):")
    _check_overfit(device, A)
    print("se3_head checks OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "checks":
        checks()
    else:
        print(__doc__)
