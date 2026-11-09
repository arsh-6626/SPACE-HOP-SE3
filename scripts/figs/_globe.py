"""Shared 3D-globe renderer for NeST decode-tree figures.

Draws the Hopf--HEALPix S^2 base as an orthographic shaded GLOBE (not a flat Mollweide):
  * Lambert-shaded disk background + front-hemisphere graticule + limb outline  -> reads as 3D
  * active HEALPix pixels as filled spherical polygons on the surface            -> the binning
  * a Hopf roll fibre ring drawn ON the globe at the chosen base point           -> the fibres
One fixed view per figure (aimed ~28 deg off the MAP base point) so the gold tile shrinks in place.
Convention: a NeST SO(3) cell = pix * n_roll + roll; base point = HEALPix pixel (NESTED).
"""
import numpy as np, healpy as hp, matplotlib as mpl
from matplotlib.patches import Polygon, Circle

HSV = mpl.colormaps["hsv"]
GOLD = "#FFD700"; GOLD_LINE = "#C99200"
LIGHT = np.array([-0.35, 0.42, 0.84]); LIGHT /= np.linalg.norm(LIGHT)   # screen-space light


def hue(roll, nr):
    return ((roll + 0.5) / nr) % 1.0           # matches psi=(roll+0.5)*2pi/nr on the hsv colorbar


def make_view(base, delta_deg=28.0, up=(0.0, 0.0, 1.0)):
    """Orthographic view whose rows are (e1, e2, a): screen_x=v.e1, screen_y=v.e2, depth=v.a.
    Aim a is `base` tilted by delta about a horizontal axis, so the MAP tile sits on visible curvature."""
    base = np.asarray(base, float); base = base / np.linalg.norm(base)
    up = np.asarray(up, float)
    axis = np.cross(base, up)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    d = np.radians(delta_deg)
    a = base * np.cos(d) + np.cross(axis, base) * np.sin(d) + axis * np.dot(axis, base) * (1 - np.cos(d))
    a = a / np.linalg.norm(a)
    e1 = np.cross(up, a)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(np.array([0.0, 1.0, 0.0]), a)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return np.stack([e1, e2, a])


def project(R, V):
    V = np.asarray(V, float)
    P = V @ R.T
    return P[..., :2], P[..., 2]


def _clamp_poly(R, V):
    """Project polygon verts; cull only if every vert is on the far hemisphere, else clamp back verts to the limb."""
    xy, d = project(R, V)
    if (d < 0).all():
        return None
    out = xy.copy(); bad = d < 0
    if bad.any():
        nrm = np.linalg.norm(xy, axis=1, keepdims=True); nrm[nrm < 1e-9] = 1.0
        out[bad] = (xy / nrm)[bad]
    return out


def _shade(cc, R):
    xy, d = project(R, np.asarray(cc, float)[None])
    n = np.array([xy[0, 0], xy[0, 1], d[0]])
    return float(np.clip(0.80 + 0.20 * np.dot(n, LIGHT), 0.70, 1.0))


def shaded_disk(ax, n=240, base_rgb=(0.82, 0.85, 0.90)):
    xs = np.linspace(-1, 1, n); X, Y = np.meshgrid(xs, xs)
    r2 = X ** 2 + Y ** 2; mask = r2 <= 1.0
    Z = np.sqrt(np.clip(1 - r2, 0, 1))
    sh = np.clip(X * LIGHT[0] + Y * LIGHT[1] + Z * LIGHT[2], 0, 1)
    sh = 0.42 + 0.58 * sh
    rgb = np.zeros((n, n, 4))
    for k in range(3):
        rgb[..., k] = base_rgb[k] * sh
    rgb[..., 3] = mask.astype(float)
    ax.imshow(rgb, extent=[-1, 1, -1, 1], origin="lower", zorder=0, interpolation="bilinear")


def _seg(ax, xy, m, color, lw, z=1, alpha=0.6):
    x = xy[:, 0].copy(); y = xy[:, 1].copy(); x[~m] = np.nan; y[~m] = np.nan
    ax.plot(x, y, color=color, lw=lw, zorder=z, alpha=alpha, solid_capstyle="round")


def graticule(ax, R, nlon=12, nlat=5, color="#aeb6c1", lw=0.45):
    t = np.linspace(0, np.pi, 120)
    for j in range(nlon):
        ph = 2 * np.pi * j / nlon
        v = np.stack([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)], 1)
        xy, d = project(R, v); _seg(ax, xy, d > 0, color, lw)
    ph = np.linspace(0, 2 * np.pi, 160)
    for j in range(1, nlat + 1):
        th = np.pi * j / (nlat + 1)
        v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.full_like(ph, np.cos(th))], 1)
        xy, d = project(R, v); _seg(ax, xy, d > 0, color, lw)


def draw_cells(ax, R, nside, nr, cells, prob, chosen_cell, step=8, context=True, lw=0.45, edge="#3a3a3a"):
    pix = cells // nr; roll = cells % nr; chpix = int(chosen_cell) // nr
    if context:
        for p in range(12):
            poly = _clamp_poly(R, hp.boundaries(1, int(p), step=14, nest=True).T)
            if poly is not None:
                ax.add_patch(Polygon(poly, closed=True, fc="none", ec="#ccd2da", lw=0.4, zorder=2))
    pmax = prob.max() if prob.max() > 0 else 1.0
    for p in np.unique(pix):
        sel = pix == p; pm = prob[sel].sum(); rbest = int(roll[sel][np.argmax(prob[sel])])
        poly = _clamp_poly(R, hp.boundaries(nside, int(p), step=step, nest=True).T)
        if poly is None:
            continue
        col = np.array(HSV(hue(rbest, nr)))[:3] * _shade(hp.pix2vec(nside, int(p), nest=True), R)
        a = float(np.clip(0.30 + 0.68 * pm / pmax, 0.20, 0.96))
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec=edge, lw=lw, alpha=a, zorder=3))
        if p == chpix:
            ax.add_patch(Polygon(poly, closed=True, fc="none", ec=GOLD_LINE, lw=2.4, zorder=5))


def fibre_ring(ax, R, base_c, nr, chosen_roll, rad=0.30):
    """Hopf roll S^1 drawn ON the globe at the chosen base point: a foreshortened hue ring, chosen psi in gold."""
    c = np.asarray(base_c, float); c = c / np.linalg.norm(c)
    t1 = np.cross(c, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(c, np.array([0.0, 1.0, 0.0]))
    t1 = t1 / np.linalg.norm(t1); t2 = np.cross(c, t1)
    npt = int(np.clip(nr, 48, 240))
    aa = (np.arange(npt) + 0.5) * 2 * np.pi / npt
    pts = c[None] + rad * (np.cos(aa)[:, None] * t1[None] + np.sin(aa)[:, None] * t2[None])
    xy, d = project(R, pts); vis = d > 0
    hues = [HSV(hue(int(a / (2 * np.pi) * nr) % nr, nr)) for a in aa]
    ax.scatter(xy[vis, 0], xy[vis, 1], s=8.0, c=np.array(hues)[vis], zorder=4, linewidths=0)
    psi = (int(chosen_roll) + 0.5) * 2 * np.pi / nr
    pc = c + rad * (np.cos(psi) * t1 + np.sin(psi) * t2)
    cxy, _ = project(R, c[None]); pxy, _ = project(R, pc[None])
    ax.plot([cxy[0, 0], pxy[0, 0]], [cxy[0, 1], pxy[0, 1]], color=GOLD_LINE, lw=1.8, zorder=5)
    ax.scatter([pxy[0, 0]], [pxy[0, 1]], s=34, c=GOLD, edgecolors=GOLD_LINE, linewidths=1.0, zorder=6)


def globe(ax, R, nside, nr, cells, prob, chosen_cell, base_c=None, chosen_roll=None,
          step=8, context=True, fibre=True, crop=None, frame=True):
    """Full globe node. crop=(cx,cy,half) zooms the axis limits (for deep-level insets)."""
    shaded_disk(ax)
    graticule(ax, R)
    draw_cells(ax, R, nside, nr, cells, prob, chosen_cell, step=step, context=context)
    if fibre and base_c is not None and chosen_roll is not None:
        fibre_ring(ax, R, base_c, nr, chosen_roll)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#8f97a2", lw=1.1, zorder=6))
    if crop is None:
        ax.set_xlim(-1.16, 1.16); ax.set_ylim(-1.16, 1.16)
    else:
        cx, cy, hw = crop
        ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        if frame:
            s.set_edgecolor(GOLD_LINE); s.set_linewidth(2.0)
