"""
Strict-hierarchy nested distribution on SE(3) = SO(3) rotation tree  x  R^3 translation octree.

The rotational half is the Hopf-HEALPix tree of so3_nested.py (imported, not modified): N_rot(l)=72*8^l cells,
each node split into 8. The translational half is a bounded equivolumetric octree over a translation box: a unit
cube in normalized coordinates g in [-1/2,1/2]^3 mapped to the camera frame by an affine box  t = t_hat + A g
(as in SpyroPose's SE(3) pyramid, Haugaard et al. 2023); at level l the cube is the regular 2^l x 2^l x 2^l grid,
N_trans(l)=8^l cells, each node split into 8 octants. Because an affine map preserves equal volume, every cell has
the same volume det(A)/8^l, so the box volume is V_box = |det(A)|.

A JOINT cell factorizes into a rotation path and a translation path. The per-node categorical is the full
64-way product of the 8 rotation children and 8 translation octants (child = rot_child*8 + trans_oct) while both
halves still subdivide, and 8-way once one half has reached its depth -- so L_rot and L_trans are independent.
The density at a pose (R,t) inside the box is

    p(R,t) = P(finest joint cell containing (R,t)) * N / (V_SO3 * V_box),   N = 72 * 8^L_rot * 8^L_trans,

with V_SO3 = pi^2. A uniform predictor therefore has average log-likelihood  -ln(pi^2 * V_box)  at every level --
the self-test for the whole SE(3) log-likelihood path, before any data or training. Poses whose translation falls
outside the box have zero density (infinite NLL); callers must size the box from a detector and log the coverage.

    python src/se3_tree.py selftest
"""
import numpy as np
import so3_nested as so3
from scipy.spatial.transform import Rotation as R

V_SO3 = so3.V_SO3                       # pi^2


# ---------------------------------------------------------------- translation octree (unit cube [-1/2,1/2]^3)

def n_trans_cells(level):
    return 8 ** level                   # level 0 = 1 cell (the whole box)


def box_volume(A):
    """V_box = |det(A)| for the affine box t = t_hat + A g, g in the unit cube."""
    return float(abs(np.linalg.det(np.asarray(A, float))))


def _trans_flat_to_coords(cells, level):
    """flat index -> integer (ix,iy,iz) in [0,2^level), row-major flat = (ix*n+iy)*n+iz."""
    cells = np.asarray(cells, np.int64)
    n = 1 << level
    iz = cells % n; r = cells // n
    iy = r % n; ix = r // n
    return ix, iy, iz


def _trans_coords_to_flat(ix, iy, iz, level):
    n = 1 << level
    return (ix * n + iy) * n + iz


def trans_cell_center(cells, level):
    """flat cell(s) at `level` -> normalized-cube centre g in [-1/2,1/2]^3, shape (...,3)."""
    cells = np.asarray(cells, np.int64)
    ix, iy, iz = _trans_flat_to_coords(cells.reshape(-1), level)
    n = 1 << level
    g = (np.stack([ix, iy, iz], -1) + 0.5) / n - 0.5
    return g.reshape(*cells.shape, 3)


def points_to_trans_cells(g, level):
    """normalized-cube points g (...,3) in [-1/2,1/2]^3 -> flat octree cell at `level` (clipped to the box)."""
    g = np.asarray(g, float).reshape(-1, 3)
    n = 1 << level
    idx = np.floor((g + 0.5) * n).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    return _trans_coords_to_flat(idx[:, 0], idx[:, 1], idx[:, 2], level)


def trans_cell_to_path(cell, level):
    """flat octree cell at `level` -> list of `level` octants in 0..7 (MSB first), octant = bx + 2by + 4bz.
    The translation root (level 0) is a single cell, so there is no level-0 categorical (unlike rotation)."""
    ix, iy, iz = (int(v[0]) for v in _trans_flat_to_coords(np.array([cell]), level))
    octs = []
    for l in range(level):
        shift = level - 1 - l
        bx = (ix >> shift) & 1; by = (iy >> shift) & 1; bz = (iz >> shift) & 1
        octs.append(bx + 2 * by + 4 * bz)
    return octs


def _trans_path_to_cell(octs, level):
    ix = iy = iz = 0
    for o in octs:
        ix = (ix << 1) | (o & 1)
        iy = (iy << 1) | ((o >> 1) & 1)
        iz = (iz << 1) | ((o >> 2) & 1)
    return _trans_coords_to_flat(ix, iy, iz, level)


def points_to_trans_path(g, level):
    return trans_cell_to_path(int(points_to_trans_cells(g, level)[0]), level)


# ---------------------------------------------------------------- joint SE(3) path

def se3_paths(mats, g, L_rot, L_trans):
    """(R in (3,3), g in (3,)) -> (rot_cell0, rot_children[L_rot], trans_octs[L_trans]).
    `g` is the normalized-cube coordinate of the translation (already mapped through A^{-1}, t_hat)."""
    c0, rot_ch = so3.cell_to_path(int(so3.rotations_to_cells(mats[None], L_rot)[0]), L_rot)
    trans_octs = points_to_trans_path(g, L_trans)
    return c0, list(rot_ch), list(trans_octs)


def joint_level_kind(l, L_rot, L_trans):
    """Per joint level l (1..max): 'joint64' while both subdivide, else 'rot8' or 'trans8'."""
    r, t = l <= L_rot, l <= L_trans
    return "joint64" if (r and t) else ("rot8" if r else "trans8")


def se3_path_logprob(logp0, logp_nodes, rot_cell0, rot_children, trans_octs):
    """Exact log P(finest joint cell) summed along the GT path.
      logp0: (72,) log-softmax of the rotation root.
      logp_nodes: list over joint levels l=1..L of the node's child log-softmax
                  (length 64 for 'joint64', 8 for 'rot8'/'trans8').
    Child index for a joint64 node is rot_child*8 + trans_oct."""
    L_rot, L_trans = len(rot_children), len(trans_octs)
    L = max(L_rot, L_trans)
    lp = logp0[rot_cell0]
    for l in range(1, L + 1):
        kind = joint_level_kind(l, L_rot, L_trans)
        node = logp_nodes[l - 1]
        if kind == "joint64":
            lp = lp + node[rot_children[l - 1] * 8 + trans_octs[l - 1]]
        elif kind == "rot8":
            lp = lp + node[rot_children[l - 1]]
        else:
            lp = lp + node[trans_octs[l - 1]]
    return lp


def n_joint_cells(L_rot, L_trans):
    return 72 * (8 ** L_rot) * (8 ** L_trans)


def cell_logprob_to_density(lp_cell, L_rot, L_trans, A):
    """log P(joint cell) -> log p(R,t) = log P + log( N / (V_SO3 * V_box) )."""
    return lp_cell + np.log(n_joint_cells(L_rot, L_trans) / (V_SO3 * box_volume(A)))


# ---------------------------------------------------------------- self-test

def _uniform_nodes(L_rot, L_trans):
    nodes = []
    for l in range(1, max(L_rot, L_trans) + 1):
        k = joint_level_kind(l, L_rot, L_trans)
        size = 64 if k == "joint64" else 8
        nodes.append(np.full(size, -np.log(size)))
    return nodes


def selftest():
    rng = np.random.RandomState(0)

    # 1) translation cell round trip: centre -> cell -> centre, every level
    worst_rt = 0.0
    for level in range(5):
        cells = np.arange(n_trans_cells(level))
        g = trans_cell_center(cells, level)
        back = points_to_trans_cells(g, level)
        assert np.array_equal(back, cells), f"trans round trip failed at level {level}"
    # 2) translation path recompose
    for level in range(1, 6):
        cells = rng.randint(0, n_trans_cells(level), 3000)
        for c in cells:
            assert _trans_path_to_cell(trans_cell_to_path(c, level), level) == c, (c, level)

    # 3) uniform predictor -> avg log-lik == -ln(pi^2 * V_box) at every (L_rot, L_trans)
    A = np.array([[0.4, 0.0, 0.1], [0.0, 0.4, -0.05], [0.0, 0.0, 1.3]])   # a non-trivial affine box
    Vb = box_volume(A)
    worst = 0.0
    for L_rot, L_trans in [(0, 0), (1, 1), (2, 2), (3, 2), (2, 3), (4, 4)]:
        logp0 = np.full(72, -np.log(72.0))
        nodes = _uniform_nodes(L_rot, L_trans)
        mats = R.random(2000, random_state=rng).as_matrix()
        gs = rng.uniform(-0.5, 0.5, size=(2000, 3))
        lls = []
        for i in range(len(mats)):
            c0, rc, to = se3_paths(mats[i], gs[i], L_rot, L_trans)
            lls.append(cell_logprob_to_density(se3_path_logprob(logp0, nodes, c0, rc, to), L_rot, L_trans, A))
        lls = np.array(lls)
        worst = max(worst, abs(lls.mean() - (-np.log(V_SO3 * Vb))))
        assert np.allclose(lls, -np.log(V_SO3 * Vb)), f"uniform density wrong at ({L_rot},{L_trans})"

    # 4) MLE consistency (data-free): the tree-softmax equals the flat categorical, so fitting empirical
    #    cell frequencies and scoring via the path matches the direct flat-categorical log-lik + density const.
    L_rot = L_trans = 2
    N = n_joint_cells(L_rot, L_trans)
    logp0 = np.full(72, -np.log(72.0))
    # ground-truth distribution: uniform over K random finest joint cells; sample poses from it
    K = 40
    rot_cells = so3.cells_to_matrices(L_rot)
    supp_rot = rng.randint(0, so3.n_cells(L_rot), K)
    supp_tr = rng.randint(0, n_trans_cells(L_trans), K)
    samp = rng.randint(0, K, 6000)
    mats = rot_cells[supp_rot[samp]]                       # cell-centre rotations (inside their cell)
    gs = trans_cell_center(supp_tr[samp], L_trans)         # cell-centre translations
    # empirical conditionals -> tree-softmax nodes, scored on the same samples
    from collections import defaultdict
    cnt0 = np.zeros(72); node_cnt = [defaultdict(lambda: np.zeros(64)) for _ in range(L_rot)]
    paths = []
    for i in range(len(mats)):
        c0, rc, to = se3_paths(mats[i], gs[i], L_rot, L_trans)
        paths.append((c0, rc, to)); cnt0[c0] += 1
        key = (c0,)
        for l in range(L_rot):
            node_cnt[l][key][rc[l] * 8 + to[l]] += 1
            key = key + (rc[l] * 8 + to[l],)
    lp0 = np.log(cnt0 / cnt0.sum() + 1e-30)
    # flat empirical categorical over finest joint cells, for the exact equality check
    flat_cnt = defaultdict(int)
    for (c0, rc, to) in paths:
        flat_cnt[(c0, tuple(rc), tuple(to))] += 1
    ntot = len(paths)
    tree_lls, flat_lls = [], []
    for (c0, rc, to) in paths:
        lp = lp0[c0]; key = (c0,)
        for l in range(L_rot):
            v = node_cnt[l][key]; cond = np.log(v / v.sum() + 1e-30)
            idx = rc[l] * 8 + to[l]; lp = lp + cond[idx]; key = key + (idx,)
        tree_lls.append(cell_logprob_to_density(lp, L_rot, L_trans, A))
        flat_lp = np.log(flat_cnt[(c0, tuple(rc), tuple(to))] / ntot)
        flat_lls.append(cell_logprob_to_density(flat_lp, L_rot, L_trans, A))
    # (a) EXACT: tree-softmax of empirical conditionals == flat empirical joint categorical
    dev = float(np.max(np.abs(np.array(tree_lls) - np.array(flat_lls))))
    assert dev < 1e-9, f"tree != flat categorical: max dev {dev:.2e}"
    # (b) loose sanity: mean approaches the uniform-over-occupied-cells value within sampling noise
    n_occ = len(flat_cnt)
    analytic = np.log(N / (n_occ * V_SO3 * Vb))
    assert abs(np.mean(tree_lls) - analytic) < 0.05, f"mean {np.mean(tree_lls):.4f} vs uniform {analytic:.4f}"

    print(f"se3_tree selftest OK | trans round-trip + path recompose exact | "
          f"uniform avg log-lik = {-np.log(V_SO3 * Vb):.4f} nats (max dev {worst:.1e}) | "
          f"tree==flat categorical (max dev {dev:.1e}) | mean {np.mean(tree_lls):.3f} ~ uniform {analytic:.3f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        print(__doc__)
