"""Render every decode-tree node as a SEPARATE shaded-globe tile, labelled by its cell (r_c).

For both the SO(3) and SE(3) trees, each node (root + all 8 fan children per level) is saved as its own PNG:
a shaded 3D globe of the Hopf-HEALPix S^2 base, aimed at that node's own cell, with the chosen HEALPix bin
(gold outline), the Hopf roll fibre ring, and a label giving level / child k / cell (r_c) / prob / roll.
MAP-path tiles use gold accents; the rest use dark accents.

Out:  logs/figs/tiles/so3/*.png  and  logs/figs/tiles/se3/*.png
Run:  cd /home/kavinder/SPACE-HOP && PYTHONPATH=src python -I scripts/figs/gen_node_tiles.py
"""
import os, sys, numpy as np, matplotlib, healpy as hp
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
import so3_nested as so3
import _globe as G

mpl.rcParams.update({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42, "savefig.dpi": 150})
GOLD = "#FFD700"; GOLD_LINE = "#C99200"; INK = "#272727"
TILES = "/home/kavinder/SPACE-HOP/logs/figs/tiles"
NLEV = 3


def tile(path, nside, nr, cells, prob, chosen_cell, focus_cell, is_map, title):
    base = np.array(hp.pix2vec(nside, int(focus_cell) // nr, nest=True))
    Rv = G.make_view(base, delta_deg=22.0)
    fig = plt.figure(figsize=(3.2, 3.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84])
    acc, dotc = (GOLD_LINE, GOLD) if is_map else ("#333333", "#333333")
    G.globe(ax, Rv, nside, nr, np.asarray(cells), np.asarray(prob),
            (chosen_cell if chosen_cell is not None else -1), base_c=base,
            chosen_roll=int(focus_cell) % nr, fibre=True, frame=False,
            fibre_accent=acc, fibre_dot=dotc, fibre_s=38.0, fibre_rad=0.44)
    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=11.5, fontweight="bold",
             color=(GOLD_LINE if is_map else INK))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def do_so3():
    out = f"{TILES}/so3"; os.makedirs(out, exist_ok=True)
    d = np.load("logs/figs/_cache/so3_dump.npz")
    p0 = d["p0"]; root_cell = int(d["root_cell"]); chosen = d["chosen"]
    tile(f"{out}/so3_root_rc{root_cell}.png", 1, 6, np.arange(72), p0, root_cell, root_cell, True,
         f"root · rc {root_cell} · roll {root_cell % 6}/6")
    n = 1
    for j in range(NLEV):
        level = j + 1; nside, nr = so3.nside_of(level), so3.n_roll(level)
        cells = d[f"cells_{j}"]; probs = d[f"probs_{j}"]; ci = int(chosen[j])
        for k in range(8):
            cell = int(cells[k]); p = float(probs[k]); is_map = (k == ci)
            tag = "_MAP" if is_map else ""
            title = f"ℓ{level} k{k} · rc {cell} · p={p:.2f} · roll {cell % nr}/{nr}"
            tile(f"{out}/so3_L{level}_k{k}_rc{cell}{tag}.png", nside, nr,
                 np.array([cell]), np.array([p]), cell, cell, is_map, title)
            n += 1
    print(f"so3: {n} tiles -> {out}")


def do_se3():
    out = f"{TILES}/se3"; os.makedirs(out, exist_ok=True)
    d = np.load("logs/figs/_cache/se3_dump.npz")
    p0 = d["p0"]; root_rc = int(d["root_rc"]); rc = d["rc"]
    tile(f"{out}/se3_root_rc{root_rc}.png", 1, 6, np.arange(72), p0, root_rc, root_rc, True,
         f"root · rc {root_rc} · roll {root_rc % 6}/6")
    n = 1
    for j in range(NLEV):
        level = j + 1; nside, nr = so3.nside_of(level), so3.n_roll(level)
        chf = d[f"chf_{j}"]; rotp = d[f"rotp_{j}"]; ci = int(rc[j])
        for k in range(8):
            cell = int(chf[k]); p = float(rotp[k]); is_map = (k == ci)
            tag = "_MAP" if is_map else ""
            title = f"ℓ{level} k{k} · rc {cell} · rot p={p:.2f} · roll {cell % nr}/{nr}"
            tile(f"{out}/se3_L{level}_k{k}_rc{cell}{tag}.png", nside, nr,
                 np.array([cell]), np.array([p]), cell, cell, is_map, title)
            n += 1
    print(f"se3: {n} tiles -> {out}")


if __name__ == "__main__":
    do_so3(); do_se3()
