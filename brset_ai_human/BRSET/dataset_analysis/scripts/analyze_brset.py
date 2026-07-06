#!/usr/bin/env python3
"""Full-pass analysis of the BRSET dataset (labels CSV + fundus JPGs).

Writes per-distribution CSVs + a headline JSON to ../results/ and prints a
human-readable summary used to author /mnt/data1/BRSET/README.md.

Run:
  PY=/mnt/data3/hongyu/miniconda3/envs/biomni_e1/bin/python3
  $PY /mnt/data1/BRSET/dataset_analysis/scripts/analyze_brset.py
"""
import os, json
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

ROOT = "/mnt/data1/BRSET"
IMG_DIR = os.path.join(ROOT, "fundus_photos")
OUT = os.path.join(ROOT, "dataset_analysis", "results")
os.makedirs(OUT, exist_ok=True)
DATE = "2026-06-21"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

df = pd.read_csv(os.path.join(ROOT, "labels_brset.csv"))
S = {"computed": DATE, "n_label_rows": int(len(df)), "n_columns": int(df.shape[1]),
     "columns": list(df.columns)}

print("=== shape:", df.shape)
print("=== sample image_id values:", df["image_id"].head().tolist())
df.head(5).to_csv(os.path.join(OUT, "sample_rows.csv"), index=False)

# ---- column info (dtype, missingness, cardinality) ----
colinfo = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "pct_missing": df.isna().mean().mul(100).round(2),
    "n_unique": df.nunique(dropna=True),
})
colinfo.to_csv(os.path.join(OUT, "column_info.csv"))
print("\n=== COLUMN INFO ===\n", colinfo)

# ---- coded-value peek ----
disease_cols = ["diabetic_retinopathy", "macular_edema", "scar", "nevus", "amd",
                "vascular_occlusion", "hypertensive_retinopathy", "drusens", "hemorrhage",
                "retinal_detachment", "myopic_fundus", "increased_cup_disc", "other"]
coded = ["camera", "patient_sex", "exam_eye", "diabetes", "insuline", "nationality",
         "optic_disc", "vessels", "macula", "DR_SDRG", "DR_ICDR", "focus", "Illuminaton",
         "image_field", "artifacts", "quality"]
print("\n=== UNIQUE VALUES (coded cols) ===")
for c in coded + disease_cols:
    print(f"  {c:28s} -> {sorted(map(str, df[c].dropna().unique()))[:14]}")

# ---- patients ----
S["n_unique_patients"] = int(df["patient_id"].nunique())
ipp = df.groupby("patient_id").size()
S["images_per_patient"] = {"mean": round(float(ipp.mean()), 3), "median": float(ipp.median()),
                            "min": int(ipp.min()), "max": int(ipp.max())}
ipp.value_counts().sort_index().to_csv(os.path.join(OUT, "images_per_patient.csv"),
                                       header=["n_patients"])
print("\n=== patients:", S["n_unique_patients"], "images/patient:", S["images_per_patient"])

# ---- categorical distributions ----
def dist(col):
    vc = df[col].value_counts(dropna=False)
    out = pd.DataFrame({"count": vc, "pct": (vc / len(df) * 100).round(2)})
    out.to_csv(os.path.join(OUT, f"dist_{col}.csv"))
    return out

S["distributions"] = {}
for c in coded:
    d = dist(c)
    S["distributions"][c] = {str(k): [int(v["count"]), float(v["pct"])] for k, v in d.iterrows()}
    print(f"\n== {c} ==\n", d.head(12))

# ---- numeric summaries ----
def numsum(col):
    s = pd.to_numeric(df[col], errors="coerce")
    return {"n_nonnull": int(s.notna().sum()), "pct_missing": round(float(s.isna().mean() * 100), 2),
            "mean": round(float(s.mean()), 2), "median": float(s.median()),
            "p5": float(s.quantile(.05)), "p95": float(s.quantile(.95)),
            "min": float(s.min()), "max": float(s.max())}
S["patient_age"] = numsum("patient_age")
S["diabetes_time_y"] = numsum("diabetes_time_y")
print("\n=== patient_age:", S["patient_age"])
print("=== diabetes_time_y:", S["diabetes_time_y"])
age = pd.to_numeric(df["patient_age"], errors="coerce")
age_hist = pd.cut(age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 200]).value_counts().sort_index()
age_hist.to_csv(os.path.join(OUT, "age_bins.csv"), header=["count"])
print("\n=== age bins ===\n", age_hist)

# ---- multilabel disease prevalence ----
dmat = df[disease_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
lab = {}
for c in disease_cols:
    pos = int((dmat[c] == 1).sum())
    lab[c] = {"positive": pos, "pct": round(pos / len(df) * 100, 2)}
labdf = pd.DataFrame(lab).T.sort_values("positive", ascending=False)
labdf.to_csv(os.path.join(OUT, "label_prevalence.csv"))
S["label_prevalence"] = lab
print("\n=== LABEL PREVALENCE ===\n", labdf)

npos = (dmat == 1).sum(axis=1)
S["images_with_no_listed_disease"] = int((npos == 0).sum())
S["mean_labels_per_image"] = round(float(npos.mean()), 3)
S["max_labels_per_image"] = int(npos.max())
npos.value_counts().sort_index().to_csv(os.path.join(OUT, "labels_per_image.csv"),
                                        header=["n_images"])
print("\n=== images w/ 0 listed disease:", S["images_with_no_listed_disease"],
      " mean labels/img:", S["mean_labels_per_image"])

# ---- DR grade cross-tab ----
ct = pd.crosstab(df["DR_ICDR"], df["diabetic_retinopathy"], dropna=False)
ct.to_csv(os.path.join(OUT, "dr_icdr_vs_drflag.csv"))
print("\n=== DR_ICDR vs diabetic_retinopathy flag ===\n", ct)

# ---- IMAGE PASS (full) ----
files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
S["n_jpg_files"] = len(files)
stems = set(os.path.splitext(f)[0] for f in files)
ids_no_ext = set(os.path.splitext(str(i))[0] for i in df["image_id"])
S["n_ids_not_on_disk"] = len(ids_no_ext - stems)
S["ids_not_on_disk"] = sorted(ids_no_ext - stems)[:30]
S["n_files_not_in_labels"] = len(stems - ids_no_ext)
S["files_not_in_labels"] = sorted(stems - ids_no_ext)[:30]
S["n_duplicate_image_ids"] = int(df["image_id"].duplicated().sum())
print(f"\n=== files={len(files)} ids={len(ids_no_ext)} "
      f"ids_not_on_disk={S['n_ids_not_on_disk']} files_not_in_labels={S['n_files_not_in_labels']} "
      f"dup_ids={S['n_duplicate_image_ids']}")
print("    ids_not_on_disk sample:", S["ids_not_on_disk"][:10])

ws, hs, sizes = [], [], []
modes, res = Counter(), Counter()
errs = 0
for i, f in enumerate(files):
    p = os.path.join(IMG_DIR, f)
    try:
        with Image.open(p) as im:
            w, h = im.size
            modes[im.mode] += 1
            res[(w, h)] += 1
            ws.append(w); hs.append(h)
        sizes.append(os.path.getsize(p))
    except Exception as e:
        errs += 1
        if errs <= 5:
            print("  IMG ERR", f, e)
    if (i + 1) % 4000 == 0:
        print(f"  ...{i+1}/{len(files)} images scanned")
ws, hs, sizes = np.array(ws), np.array(hs), np.array(sizes)

def astats(a):
    return {"min": int(a.min()), "p5": int(np.percentile(a, 5)), "median": int(np.median(a)),
            "mean": round(float(a.mean()), 1), "p95": int(np.percentile(a, 95)), "max": int(a.max())}
S["image_errors"] = errs
S["img_width"] = astats(ws)
S["img_height"] = astats(hs)
S["img_filesize_kb"] = {k: round(v / 1024, 1) for k, v in astats(sizes).items()}
S["img_modes"] = dict(modes)
S["n_distinct_resolutions"] = len(res)
S["top_resolutions"] = [{"res": f"{w}x{h}", "count": c} for (w, h), c in res.most_common(15)]
pd.DataFrame([(f"{w}x{h}", c) for (w, h), c in res.most_common()],
             columns=["resolution", "count"]).to_csv(os.path.join(OUT, "image_resolutions.csv"), index=False)
print("\n=== IMAGE width:", S["img_width"])
print("=== IMAGE height:", S["img_height"])
print("=== filesize KB:", S["img_filesize_kb"])
print("=== modes:", dict(modes), " distinct resolutions:", len(res))
print("=== top resolutions:", S["top_resolutions"][:6])

json.dump(S, open(os.path.join(OUT, "brset_summary.json"), "w"), indent=2, default=str)
print("\nWROTE", os.path.join(OUT, "brset_summary.json"))
