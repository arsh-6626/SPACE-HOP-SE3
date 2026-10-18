"""
Strict-hierarchy nested distribution on SO(3) for the SYMSOL / SYMSOL-T distribution benchmarks.

The grid is the Hopf-HEALPix equivolumetric grid of Implicit-PDF (Murphy et al. 2021), built from Yershova et al.:
the S^2 base point is tiled by HEALPix in the NESTED scheme (12 nside^2 equal-area pixels) and the Hopf fibre (roll)
by 6 nside equal angles, giving  N(l) = 72 * 8^l  equal-volume cells at level l. Doubling nside splits every S^2
pixel into 4 (NESTED) and every roll bin into 2, so each cell has exactly 8 children -- a strict tree, unlike the
overlapping refinement grid of nested_se3.py.

A distribution is a categorical over the 72 level-0 cells and, at every expanded node, a categorical over its 8
children. The density at a rotation is  p(R) = P(finest cell containing R) * N(L) / V_SO3,  with V_SO3 = pi^2 (the
IPDF convention), so a uniform predictor has p = 1/pi^2 and average log-likelihood -ln(pi^2) ~ -2.2899 at every level.
That identity is the self-test for the whole log-likelihood path, before any data or training.

    python src/so3_nested.py selftest
"""
import numpy as np
import healpy as hp
from scipy.spatial.transform import Rotation as R

V_SO3 = np.pi ** 2            # IPDF volume convention; uniform density = 1 / V_SO3
BASE_NSIDE = 1                # level 0: nside 1 -> 12 S^2 pixels, 6 roll bins = 72 cells


def nside_of(level):
    return BASE_NSIDE << level          # 1, 2, 4, ...


def n_roll(level):
    return 6 * nside_of(level)          # 6, 12, 24, ...


def n_cells(level):
    return 72 * (8 ** level)


# ---------------------------------------------------------------- grid <-> rotation (Hopf map)

def cells_to_matrices(level):
    """All N(level) cell-centre rotations as (N, 3, 3). Cell c = pix * n_roll + roll (pix in NESTED HEALPix)."""
    nside, nr = nside_of(level), n_roll(level)
    npix = hp.nside2npix(nside)
    pix = np.repeat(np.arange(npix), nr)
    roll = np.tile(np.arange(nr), npix)
    theta, phi = hp.pix2ang(nside, pix, nest=True)      # colatitude, azimuth of the base point
    psi = (roll + 0.5) * (2 * np.pi / nr)
    # Hopf lift: base point = Rz(phi) Ry(theta) applied to z; fibre = Rz(psi). ZYZ Euler (phi, theta, psi).
    return R.from_euler('ZYZ', np.stack([phi, theta, psi], 1)).as_matrix()


def rotations_to_cells(mats, level):
    """(N, 3, 3) -> flat cell index at `level` (pix * n_roll + roll). Inverse of the Hopf lift above."""
    mats = np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3)
    nside, nr = nside_of(level), n_roll(level)
    base = mats @ np.array([0.0, 0.0, 1.0])             # R z  = base point on S^2
    theta, phi = hp.vec2ang(base)
    pix = hp.ang2pix(nside, theta, phi, nest=True)
    # remove the base point to read the fibre angle: Ry(-theta) Rz(-phi) R = Rz(psi)
    rem = (R.from_euler('Y', (-theta)[:, None]).as_matrix() @ R.from_euler('Z', (-phi)[:, None]).as_matrix() @ mats)
    psi = np.mod(np.arctan2(rem[:, 1, 0], rem[:, 0, 0]), 2 * np.pi)
    roll = np.minimum((psi / (2 * np.pi / nr)).astype(np.int64), nr - 1)
    return pix * nr + roll


def cell_to_path(cell, level):
    """Flat level-`level` cell index -> (level0 cell in 0..71, then child 0..7 for each of levels 1..level).

    NESTED HEALPix: pix children are pix*4 + {0,1,2,3}. Roll children are roll*2 + {0,1}. Child = s2_child*2 + roll_child.
    """
    cell = int(cell)
    nr = n_roll(level)
    pix, roll = divmod(cell, nr)
    s2_path, roll_path = [], []
    for l in range(level, 0, -1):
        pix, s2c = divmod(pix, 4)
        roll, rc = divmod(roll, 2)
        s2_path.append(s2c); roll_path.append(rc)
    # now pix in 0..11, roll in 0..5  -> level-0 flat index
    cell0 = pix * n_roll(0) + roll
    children = [s2_path[l] * 2 + roll_path[l] for l in reversed(range(level))]
    return cell0, children


def paths_batch(mats, level):
    """(N,3,3) -> (cell0 (N,), children (N, level) in 0..7). Vectorised wrapper over cell_to_path."""
    cells = rotations_to_cells(mats, level)
    cell0 = np.empty(len(cells), np.int64)
    children = np.empty((len(cells), level), np.int64)
    for i, c in enumerate(cells):
        c0, ch = cell_to_path(c, level)
        cell0[i] = c0
        children[i] = ch
    return cell0, children


# ---------------------------------------------------------------- log-likelihood from a distribution

def path_logprob(logp0, logp_child, cell0, children):
    """Exact log P(finest cell) summed over the GT path.
    logp0: (72,) log-softmax level-0. logp_child: (level, 8) log-softmax of each node on the path. Scalars in/out."""
    lp = logp0[cell0]
    for l, c in enumerate(children):
        lp = lp + logp_child[l, c]
    return lp


def cell_logprob_to_density(lp_cell, level):
    """log P(cell) -> log p(R) = log P(cell) + log(N(level) / V_SO3)."""
    return lp_cell + np.log(n_cells(level) / V_SO3)


# ---------------------------------------------------------------- self-test

def selftest():
    rng = np.random.RandomState(0)
    # 1) round trip: cell -> centre rotation -> cell, every level
    worst_rt = 0
    for level in range(4):
        mats = cells_to_matrices(level)
        back = rotations_to_cells(mats, level)
        worst_rt = max(worst_rt, int(np.max(np.abs(back - np.arange(n_cells(level))))))
    assert worst_rt == 0, f"cell round trip failed: {worst_rt}"

    # 2) path reconstruction: cell0 + children recomposes the same flat cell
    for level in range(1, 4):
        cells = rng.randint(0, n_cells(level), 2000)
        for c in cells:
            c0, ch = cell_to_path(c, level)
            # recompose
            pix0, roll0 = divmod(c0, n_roll(0))
            pix, roll = pix0, roll0
            for l in range(level):
                pix = pix * 4 + ch[l] // 2
                roll = roll * 2 + ch[l] % 2
            assert pix * n_roll(level) + roll == c, (c, level)

    # 3) uniform predictor -> average log-likelihood == -ln(pi^2) at every level
    worst = 0
    for level in range(5):
        logp0 = np.full(72, -np.log(72.0))
        logpc = np.full((level, 8), -np.log(8.0))
        mats = R.random(3000, random_state=rng).as_matrix()
        c0, ch = paths_batch(mats, level)
        lls = np.array([cell_logprob_to_density(path_logprob(logp0, logpc, c0[i], ch[i]), level)
                        for i in range(len(mats))])
        worst = max(worst, abs(lls.mean() - (-np.log(V_SO3))))
        assert np.allclose(lls, -np.log(V_SO3)), f"uniform density not 1/pi^2 at level {level}"
    print(f"selftest OK | cell round trip exact | path recompose exact (levels 1-3) | "
          f"uniform avg log-lik = {-np.log(V_SO3):.4f} nats at levels 0-4 (max dev {worst:.1e})")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        print(__doc__)
