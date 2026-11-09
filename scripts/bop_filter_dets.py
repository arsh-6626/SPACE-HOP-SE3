"""Filter BOP default detections to the BOP19 test targets (scene,im,obj), so inference only
runs on evaluated images. n_top selection is left to bop_toolkit (n_top=-1 uses inst_count).

  python scripts/bop_filter_dets.py --det <dets.json> --targets <test_targets_bop19.json> --out <filtered.json>
"""
import json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--det", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dets = json.load(open(a.det))
tgts = json.load(open(a.targets))
keep_img = set()          # (scene, im) that are evaluated at all
keep_triple = set()       # (scene, im, obj) actually targeted
for t in tgts:
    keep_img.add((int(t["scene_id"]), int(t["im_id"])))
    keep_triple.add((int(t["scene_id"]), int(t["im_id"]), int(t["obj_id"])))

out = []
for d in dets:
    s = int(d["scene_id"]); im = int(d.get("image_id", d.get("im_id")))
    o = int(d.get("category_id", d.get("obj_id")))
    if (s, im, o) in keep_triple:
        out.append(d)

json.dump(out, open(a.out, "w"))
print(f"{a.det}: {len(dets)} dets -> {len(out)} on {len(keep_triple)} targeted (scene,im,obj) "
      f"across {len(keep_img)} images -> {a.out}")
