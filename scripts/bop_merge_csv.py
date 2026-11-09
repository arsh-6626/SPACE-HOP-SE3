"""Merge per-object BOP result CSVs into one leaderboard CSV and make the `time` column satisfy
bop_toolkit's check (identical for every estimate of the same (scene,im)). We report the per-image
sum of inference times; a single value per image.

  python scripts/bop_merge_csv.py --out <METHOD_DATASET-SPLIT.csv> <obj1.csv obj2.csv ...>
"""
import csv, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("inputs", nargs="+")
a = ap.parse_args()

HEADER = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
rows = []
for f in a.inputs:
    if not os.path.exists(f):
        print(f"[warn] missing {f}"); continue
    with open(f) as fh:
        r = csv.DictReader(fh)
        for d in r:
            rows.append(d)

# per-image summed time
tsum = {}
for d in rows:
    k = (d["scene_id"], d["im_id"])
    tsum[k] = tsum.get(k, 0.0) + float(d.get("time", 0) or 0)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(HEADER)
    for d in rows:
        k = (d["scene_id"], d["im_id"])
        w.writerow([d["scene_id"], d["im_id"], d["obj_id"], d["score"],
                    d["R"], d["t"], f"{tsum[k]:.4f}"])
print(f"merged {len(a.inputs)} files -> {len(rows)} rows -> {a.out}")
