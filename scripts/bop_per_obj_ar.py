"""Per-object BOP19 AR breakdown (VSD/MSSD/MSPD + overall) from a bop_toolkit eval dir.
AR_type(obj) = mean over that error type's (tau,threshold) score cells of obj_recalls[obj].
BOP-AR(obj)  = mean(AR_vsd, AR_mssd, AR_mspd)(obj)."""
import json, glob, os, sys, re
from collections import defaultdict

evaldir = sys.argv[1]                      # .../results/bop_eval/<method>_<ds>-<split>
objs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None

# type -> obj -> list of recalls (one per score cell)
acc = {"vsd": defaultdict(list), "mssd": defaultdict(list), "mspd": defaultdict(list)}
cells = defaultdict(int)
for f in glob.glob(os.path.join(evaldir, "error=*", "scores_*.json")):
    etype = re.search(r"error=([a-z]+)_", f).group(1)
    d = json.load(open(f))
    cells[etype] += 1
    for o, r in d["obj_recalls"].items():
        acc[etype][int(o)].append(r)

allobjs = objs or sorted({o for t in acc for o in acc[t]})
print(f"score cells per type: {dict(cells)}")
print(f"{'obj':>4} | {'AR_VSD':>7} {'AR_MSSD':>7} {'AR_MSPD':>7} | {'BOP-AR':>7}")
print("-"*45)
means = {"vsd": [], "mssd": [], "mspd": []}
for o in allobjs:
    vals = {}
    for t in ("vsd", "mssd", "mspd"):
        lst = acc[t].get(o, [])
        vals[t] = sum(lst)/len(lst) if lst else float("nan")
        if lst: means[t].append(vals[t])
    bop = sum(v for v in vals.values())/3.0
    print(f"{o:>4} | {vals['vsd']*100:7.2f} {vals['mssd']*100:7.2f} {vals['mspd']*100:7.2f} | {bop*100:7.2f}")
print("-"*45)
mv = {t: (sum(means[t])/len(means[t]) if means[t] else 0) for t in means}
print(f"{'mean':>4} | {mv['vsd']*100:7.2f} {mv['mssd']*100:7.2f} {mv['mspd']*100:7.2f} | {sum(mv.values())/3*100:7.2f}   (over listed objs)")
