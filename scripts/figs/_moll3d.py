"""Lifted-Mollweide node renderer: a true equal-area Mollweide projection of the Hopf--HEALPix base sphere,
tilted into 3D space (perspective 'card') so it reads as a 3D object instead of a flat ellipse.
Filled HEALPix pixels = the S^2 binning; hue = the Hopf roll fibre; chosen cell = gold.
"""
import numpy as np, healpy as hp, matplotlib as mpl
from matplotlib.patches import Polygon

HSV = mpl.colormaps["hsv"]
GOLD = "#FFD700"; GOLD_LINE = "#C99200"
SQ2 = np.sqrt(2.0)


def hue(roll, nr):
    return ((roll + 0.5) / nr) % 1.0


def _aux_theta(lat):
    th = np.asarray(lat, float).copy()
    for _ in range(8):
        den = 2 + 2 * np.cos(2 * th); den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        th = th - (2 * th + np.sin(2 * th) - np.pi * np.sin(lat)) / den
    return th


def mollweide(lon, lat):
    """lon in [-pi,pi], lat in [-pi/2,pi/2] -> (x,y) with x in [-1,1], y in [-0.5,0.5] (2:1 ellipse)."""
    th = _aux_theta(np.asarray(lat, float))
    x = (2 * SQ2 / np.pi) * np.asarray(lon, float) * np.cos(th)
    y = SQ2 * np.sin(th)
    return x / (2 * SQ2), y / (2 * SQ2)


def lift(x, y, tilt_deg=44.0, D=4.2, yshift=0.0):
    """Tilt the flat Mollweide card about its horizontal axis and apply weak perspective.
    Bottom edge (y<0) comes toward the viewer (larger); top edge recedes (smaller) -> a 3D card."""
    a = np.radians(tilt_deg)
    x = np.asarray(x, float); y = np.asarray(y, float)
    z = -y * np.sin(a)                      # bottom (y<0) -> z>0 toward camera
    yv = y * np.cos(a)
    s = D / (D - z)
    return x * s, yv * s + yshift


def _vec_lonlat(v):
    th, ph = hp.vec2ang(np.atleast_2d(np.asarray(v, float)))
    return ((ph + np.pi) % (2 * np.pi) - np.pi), (np.pi / 2 - th)


def _cell_poly2d(nside, pix, step=8):
    lon, lat = _vec_lonlat(hp.boundaries(nside, int(pix), step=step, nest=True).T)
    m = np.arctan2(np.sin(lon).mean(), np.cos(lon).mean())
    lon = m + ((lon - m + np.pi) % (2 * np.pi) - np.pi)
    seam = bool((np.abs(lon) > np.pi - 2e-3).any())
    mx, my = mollweide(np.clip(lon, -np.pi + 1e-6, np.pi - 1e-6), lat)
    return np.column_stack(lift(mx, my)), seam


def _boundary(step=240):
    lat = np.linspace(-np.pi / 2, np.pi / 2, step)
    r = np.concatenate([np.column_stack(mollweide(np.full_like(lat, np.pi - 1e-6), lat)),
                        np.column_stack(mollweide(np.full_like(lat, -np.pi + 1e-6), lat[::-1]))])
    return np.column_stack(lift(r[:, 0], r[:, 1]))


def _shade(cc):
    L = np.array([-0.3, 0.5, 0.8]); L /= np.linalg.norm(L)
    return float(np.clip(0.82 + 0.18 * np.dot(np.asarray(cc, float) / np.linalg.norm(cc), L), 0.72, 1.0))


def node(ax, nside, nr, cells, prob, chosen_cell, step=8, context=True, fibre=True, frame=True, base_scale=1.0):
    pix = cells // nr; roll = cells % nr
    chpix = None if chosen_cell is None else int(chosen_cell) // nr
    bnd = _boundary()
    # thickness slab (card underside) for a 3D object feel
    ax.add_patch(Polygon(bnd + [0, -0.028], closed=True, fc="#aeb6c2", ec="#8f97a2", lw=0.8, zorder=0))
    ax.add_patch(Polygon(bnd, closed=True, fc="#eef1f5", ec="#8f97a2", lw=1.0, zorder=1))
    # graticule
    for ph in np.linspace(-np.pi, np.pi, 7)[1:-1]:
        la = np.linspace(-np.pi / 2, np.pi / 2, 60); mx, my = mollweide(np.full_like(la, ph), la)
        sx, sy = lift(mx, my); ax.plot(sx, sy, color="#c3cad3", lw=0.4, zorder=2)
    for la in np.linspace(-np.pi / 2, np.pi / 2, 5)[1:-1]:
        lo = np.linspace(-np.pi, np.pi, 80); mx, my = mollweide(lo, np.full_like(lo, la))
        sx, sy = lift(mx, my); ax.plot(sx, sy, color="#c3cad3", lw=0.4, zorder=2)
    if context:
        for p in range(12):
            poly, seam = _cell_poly2d(1, p, step=12)
            if not seam:
                ax.add_patch(Polygon(poly, closed=True, fc="none", ec="#cfd4db", lw=0.4, zorder=3))
    pmax = prob.max() if prob.max() > 0 else 1.0
    for p in np.unique(pix):
        sel = pix == p; pm = prob[sel].sum(); rbest = int(roll[sel][np.argmax(prob[sel])])
        poly, seam = _cell_poly2d(nside, int(p), step=step)
        if seam:
            continue
        col = np.array(HSV(hue(rbest, nr)))[:3] * _shade(hp.pix2vec(nside, int(p), nest=True))
        a = float(np.clip(0.30 + 0.68 * pm / pmax, 0.20, 0.96))
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="#3a3a3a", lw=0.45, alpha=a, zorder=4))
        if chpix is not None and p == chpix:
            ax.add_patch(Polygon(poly, closed=True, fc="none", ec=GOLD_LINE, lw=2.4, zorder=6))
    ax.add_patch(Polygon(bnd, closed=True, fc="none", ec="#6f7782", lw=1.1, zorder=7))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.18 * base_scale, 1.18 * base_scale); ax.set_ylim(-1.02 * base_scale, 0.78 * base_scale)
    for s in ax.spines.values():
        s.set_visible(False)
