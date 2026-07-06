# BRSET — A Brazilian Multilabel Ophthalmological Dataset (Retinal Fundus Photos, v1.0.1)

**BRSET** is a multi-labelled **color fundus photography** dataset of **16,266 retinal images from 8,524 Brazilian patients**, the first large public ophthalmological image dataset from South America. It was assembled from **three ophthalmological centers in São Paulo, Brazil**, with images acquired **2010–2020** on two retinal cameras (a **Nikon NF505** and a **Canon CR‑2**), explicitly to reduce the under-representation of low-/middle-income-country populations in retinal-imaging datasets. Every image is a single macula-centred (fovea-centred, 45°, optic-disc-included) posterior-pole fundus photograph, paired in a master CSV with **demographics** (age, sex, diabetes status), **three anatomical-structure assessments** (optic disc, vessels, macula), **four image-quality parameters** plus an overall quality flag, **two diabetic-retinopathy severity gradings** (ICDR and Scottish/SDRG, each 0–4), and **13 binary disease labels** (diabetic retinopathy, macular edema, AMD, drusen, increased cup/disc ratio, etc.). It supports diabetic-retinopathy grading, multi-label retinal-disease classification, image-quality assessment, and fairness / external-validation research. It was produced by Nakayama et al. and released on **PhysioNet (project `brazilian-ophthalmological`, v1.0.1, 2024)**.

> This dataset requires a **PhysioNet credentialed account + signed Data Use Agreement** (PhysioNet Credentialed Health Data License 1.5.0, see `LICENSE.txt`) and completion of CITI human-subjects training. Treat all images and metadata as **PHI-adjacent**: do not attempt re-identification, do not redistribute, do not share access.

> **Location note (organized 2026-06-21):** this dataset was moved out of the original PhysioNet download tree (`/mnt/data1/physionet.org/files/brazilian-ophthalmological/1.0.1/`) and now lives flat at **`/mnt/data1/BRSET/`**. The companion `dataset_analysis/` folder and this README were added on 2026-06-21.

---

## At-a-glance summary (full-pass, computed 2026-06-21)

| | |
|-|-|
| **Root path on disk** | `/mnt/data1/BRSET/` |
| **Total size on disk** | **≈ 16 GB** (fundus_photos ≈ 15.6 GB + 1.7 MB labels CSV + checksums) |
| **Modalities** | One: **color fundus photography (CFP)** — 8-bit RGB JPEG retinal images — plus one **tabular** label/metadata CSV. No OCT, no free text, no other modalities. |
| **Unit of the dataset** | One **image = one fundus photo of one eye**. Patients contribute 1–4 images (typically both eyes; mean 1.91). |
| **Images** | **16,266** rows in `labels_brset.csv`; **16,259** `.jpg` files on disk (**7 label rows have no image file** — see Data quality §7). |
| **Patients** | **8,524** unique `patient_id`. |
| **Label table** | `labels_brset.csv` — 16,266 rows × **34 columns** (1 image-level row each). |
| **Label families** | demographics (age/sex/diabetes), 3 anatomical assessments, 4 quality params + overall quality, 2 DR grading scales (0–4), 13 binary disease labels. |
| **Time span** | Acquisition **2010–2020** (no per-image acquisition date is included in the CSV). |
| **Cameras** | Canon CR‑2 (`Canon CR`) 65.1 % · Nikon NF505 (`NIKON NF5050`) 34.9 %. |
| **Country** | 100 % Brazil (`nationality` is single-valued). |
| **License / access** | PhysioNet **Credentialed** Health Data License 1.5.0 (DUA + CITI training required). De-identified (IRB-approved, consent waived). |
| **Predefined split** | **None.** Split per **patient** to avoid leakage — see §4. |
| **Companion artifacts** | `dataset_analysis/` — full-pass analysis script + ~25 per-distribution CSVs + `brset_summary.json`. |

**Headline numbers (full-pass, computed 2026-06-21):**

| Item | Value |
|-|-:|
| Fundus images (`.jpg` on disk) | **16,259** |
| Label rows (`labels_brset.csv`) | **16,266** |
| Label rows without an image on disk | **7** (`img04414`–`img04419`, `img06900`) |
| Unique patients | **8,524** |
| Images per patient | mean 1.91 · median 2 · max 4 |
| Female / Male | 61.8 % / 38.2 % (`patient_sex` 2 / 1) |
| Right / Left eye | 50.1 % / 49.9 % (`exam_eye` 1 / 2) |
| Diabetic patients (`diabetes = yes`) | 15.9 % |
| Any diabetic retinopathy (`DR_ICDR ≥ 1`) | 6.66 % (1,083 images) |
| Images with **none** of the 13 disease labels | **8,461 (52.0 %)** |
| Mean disease labels / image | 0.60 (max 3) |
| Distinct image resolutions | 6 (all RGB JPEG) |
| Overall quality "Adequate" | 87.8 % |

See `dataset_analysis/results/brset_summary.json` for every number below in one machine-readable file.

---

## 1. Provenance & collection

- **Producers:** Luís Filipe Nakayama and colleagues (UNIFESP / São Paulo Federal University, in collaboration with MIT Laboratory for Computational Physiology). Released on PhysioNet by the MIT-LCP.
- **Source sites:** **three ophthalmological centers in São Paulo, Brazil.**
- **Collection period:** **2010–2020.**
- **Acquisition devices:** two retinal fundus cameras — a **Nikon NF505** (Tokyo, Japan) and a **Canon CR‑2** (Canon Inc.). In the CSV these appear as `NIKON NF5050` and `Canon CR`.
- **Image protocol:** **macula/fovea-centred** posterior-pole color fundus photographs, **45° field**, optic disc included in frame. One paired (both-eye) exam per patient was targeted; some patients have repeats (up to 4 images).
- **Exclusions (by the producers):** fluorescein angiograms, non-retinal images, duplicates, and low-quality frames were removed before release.
- **Labelling:** demographic and clinical metadata plus image-level labels were assigned by ophthalmologists; the two DR grading scales and the AMD labels were **re-reviewed for v1.0.1** (see version history below).
- **Version:** **v1.0.1** (released 2024-08-14). Changes from v1.0.0 (2023-03-08): renamed `SAH` → `hypertension` inside the `comorbidities` text; fixed an `ilumination`→`illumination` spelling note (**but the CSV column is still spelled `Illuminaton`** — see §7); re-reviewed AMD labels; re-reviewed the `DR_ICDR` and `DR_SDRG` labels.

> Source paper: **Nakayama, L. F., Restrepo, D., Matos, J., Ribeiro, L. Z., Malerbi, F. K., Celi, L. A., & Regatieri, C. S. (2024). "BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos." *PLOS Digital Health* 3(7): e0000454.**

---

## 2. Directory / file layout

```
/mnt/data1/BRSET/                       ≈16 GB
├── fundus_photos/          15.6 GB   16,259 *.jpg  color fundus images (8-bit RGB) + 1 index.html artifact
├── labels_brset.csv         1.7 MB   16,266 rows × 34 cols — master label + metadata table (one row per image)
├── LICENSE.txt              2.5 KB   PhysioNet Credentialed Health Data License 1.5.0
├── SHA256SUMS.txt           1.5 MB   SHA-256 checksums for the payload files
├── dataset_analysis/          —      this README's evidence (added 2026-06-21)
│   ├── scripts/analyze_brset.py      full-pass analysis: labels CSV + every JPG header
│   └── results/                      ~25 per-distribution CSVs + brset_summary.json
└── README.md                         this file
```

- **File formats:** images are baseline **JPEG** (`.jpg`, 8-bit, 3-channel RGB); labels are a single **CSV** (UTF-8, comma-separated, header row).
- **Image naming:** flat directory, no patient/eye subfolders. Filename = `<image_id>.jpg`, where `<image_id>` is `imgNNNNN` (e.g. `img00001`) and is the join key in `labels_brset.csv`. (Verified: 0 duplicate `image_id`, 0 files missing from the labels, 7 label rows missing their file.)
- `index.html` inside `fundus_photos/` is a PhysioNet web-mirror artifact, not data.

---

## 3. Data contents & distribution — the core section

BRSET has **one image modality** (color fundus photography, §3.1) plus its **tabular label table** (§3.2). The label families are detailed in §3.3–§3.8. All counts are a **full pass** over 16,266 CSV rows and 16,259 JPGs, computed 2026-06-21; percentages are over all 16,266 label rows unless stated.

### 3.1 Modality: color fundus photography (CFP) — domain background

**What it is and how it is acquired.** A color fundus photograph is a **2-D color image of the interior posterior surface of the eye (the *fundus*)** — the retina, optic disc, macula, and retinal blood vessels — taken through the pupil with a specialized low-power microscope + flash camera (a *fundus camera*). It is **fast, non-invasive, and inexpensive**: no incision, no dye for a standard color photo (unlike fluorescein angiography), a few seconds per eye, often through a dilated pupil. The image is a roughly circular bright field on a black background; in BRSET each photo is **macula-centred at a 45° field**, so the dark-reddish **macula/fovea** sits near the center, the brighter round **optic disc** is to the side, and **arteries/veins** radiate across the orange-red retinal background.

**What it shows.** A normal fundus shows a sharp, pinkish-orange optic disc with a small central cup, smoothly tapering vessels, an even retinal background, and an unremarkable macula. Disease appears as deviations: **microaneurysms, dot/blot hemorrhages, hard exudates, cotton-wool spots and neovascularization** (diabetic retinopathy); **drusen and pigmentary/atrophic or neovascular changes at the macula** (age-related macular degeneration); a **large cup-to-disc ratio** (suspicious for glaucoma); **pale/atrophic scars** (e.g. healed toxoplasmosis); **tortuous vessels, flame hemorrhages, AV nicking** (hypertensive retinopathy); a **tessellated/“tigroid” pale fundus with peripapillary atrophy** (myopic fundus); or a **blocked vessel territory** (retinal vascular occlusion).

**How it is used clinically.** CFP is the **primary modality for screening and grading diabetic retinopathy and AMD**, for assessing the optic-nerve-head cup in **glaucoma**, and for documenting many other retinal conditions. Because diabetic retinopathy is a leading cause of preventable blindness and screening volumes are huge, **automated DR grading and multi-disease triage from fundus photos** (the central task this dataset targets) is one of the most active areas of medical AI.

**Known artifacts / quality issues** (and present here): **poor pupil dilation** (dark/peripheral shadowing), **uneven or over/under-illumination**, **media opacity** (cataract) blurring the image, **defocus blur**, **dust/lens artifacts and glare**, and **off-center field** (macula or disc cut off). BRSET grades exactly these via its `focus`, `Illuminaton`, `image_field`, `artifacts` and overall `quality` columns (§3.4).

#### 3.1.1 What kind of images are in this dataset (full pass over 16,259 JPGs)

- **Anatomy / subject:** human retinal fundus, posterior pole, macula-centred, 45° field, one eye per image.
- **Color:** **8-bit RGB**, 3 channels — **100 % `RGB` mode** (16,259 / 16,259).
- **Format / compression:** baseline **JPEG** (lossy). File size: median **1.04 MB**, p5 0.37 MB, p95 1.31 MB, max 2.49 MB.
- **Resolution:** images come in **only 6 distinct sizes** (essentially one native size per camera/setting), unlike the continuous spread seen in raw radiology data. Width median **2,672 px** (range 951–2,984); height median **2,056 px** (range 874–2,304).

| Resolution (W×H) | Images | Note |
|-|-:|-|
| 2672 × 2056 | 5,583 | most common |
| 2984 × 2304 | 4,927 | largest field |
| 2390 × 1880 | 3,650 | |
| 2420 × 1880 | 1,695 | |
| 951 × 874 | 403 | **small / downsized subset** |
| 990 × 874 | 1 | singleton |

> The 404 small (≈951×874) images are a distinct low-resolution subset — if you train at high resolution, decide explicitly whether to up-sample or drop them. Full table: `dataset_analysis/results/image_resolutions.csv`.

### 3.2 The label table `labels_brset.csv`

One row per image, **16,266 rows × 34 columns**, keyed by `image_id`. Columns by family:

| Family | Columns |
|-|-|
| Identifiers | `image_id`, `patient_id` |
| Acquisition | `camera` |
| Demographics / clinical | `patient_age`, `patient_sex`, `nationality`, `comorbidities`, `diabetes`, `diabetes_time_y`, `insuline` |
| Exam context | `exam_eye` |
| Anatomical (1=normal, 2=abnormal) | `optic_disc`, `vessels`, `macula` |
| Image quality (1=normal, 2=abnormal) | `focus`, `Illuminaton` *(sic)*, `image_field`, `artifacts`; overall `quality` (Adequate/Inadequate) |
| DR severity (0–4) | `DR_ICDR`, `DR_SDRG` |
| Binary disease labels (0/1) | `diabetic_retinopathy`, `macular_edema`, `scar`, `nevus`, `amd`, `vascular_occlusion`, `hypertensive_retinopathy`, `drusens`, `hemorrhage`, `retinal_detachment`, `myopic_fundus`, `increased_cup_disc`, `other` |

Per-column dtype, %-missing and cardinality are in `dataset_analysis/results/column_info.csv`.

### 3.3 Demographics & clinical metadata (full pass)

- **Sex** (`patient_sex`; **1 = male, 2 = female**): **female 61.8 %** (10,052), male 38.2 % (6,214). 0 % missing.
- **Eye** (`exam_eye`; **1 = right/OD, 2 = left/OS**): right 50.1 % (8,155), left 49.9 % (8,111) — well balanced.
- **Age** (`patient_age`, years): **33.5 % missing**. Among the 10,820 present: mean **57.7**, median **61**, p5 21, p95 83, range 5–97. Modal bands 60–70 (2,562) and 50–60 (2,083). 5-year histogram: `results/age_bins.csv`.
- **Diabetes** (`diabetes`, yes/No): **15.9 % yes** (2,579), 84.1 % No. *(Values are mixed-case `yes`/`No`.)*
- **Diabetes duration** (`diabetes_time_y`): **88.5 % missing**; among 1,864 present, median 10 y (mean 13.1, max 60).
- **Insulin use** (`insuline`, yes/no): **89.5 % missing**; yes 5.5 %, no 5.1 %.
- **Comorbidities** (`comorbidities`, free text): **50.6 % missing**, 213 distinct strings (e.g. hypertension, etc.) — uncurated free text, parse before use.
- **Nationality** (`nationality`): **100 % "Brazil"** — a constant column (no information; do not use as a feature).

### 3.4 Image-quality parameters (full pass)

Four sub-parameters (**1 = normal/adequate, 2 = abnormal/inadequate**) plus a derived overall `quality`:

| Column | abnormal (=2) | abnormal % | Note |
|-|-:|-:|-|
| `focus` | 541 | 3.33 % | + **2 rows coded `0`** (out of codebook, §7) |
| `Illuminaton` *(sic)* | 84 | 0.52 % | column spelling is `Illuminaton` |
| `image_field` | 1,401 | 8.61 % | most common quality problem |
| `artifacts` | 57 | 0.35 % | rarest |
| **`quality`** (overall) | 1,986 **Inadequate** | **12.21 %** | 87.79 % Adequate |

### 3.5 Anatomical-structure assessments (full pass)

Each structure is graded **1 = normal, 2 = abnormal**:

| Structure | abnormal (=2) | abnormal % | Note |
|-|-:|-:|-|
| `macula` | 4,677 | 28.75 % | |
| `optic_disc` | 3,279 | 20.16 % | + **1 row coded `bv`** (out of codebook, §7) |
| `vessels` | 807 | 4.96 % | |

### 3.6 Diabetic-retinopathy grading (two scales, full pass)

Two ordinal severity scales are provided **per image**, each 0–4. Both are heavily dominated by grade 0 (no DR).

**`DR_ICDR` — International Clinical Diabetic Retinopathy scale**

| Grade | Meaning | Images | % |
|-:|-|-:|-:|
| 0 | No retinopathy | 15,183 | 93.34 % |
| 1 | Mild non-proliferative (NPDR) | 158 | 0.97 % |
| 2 | Moderate NPDR | 451 | 2.77 % |
| 3 | Severe NPDR | 78 | 0.48 % |
| 4 | Proliferative DR / post-laser | 396 | 2.43 % |

**`DR_SDRG` — Scottish Diabetic Retinopathy Grading**

| Grade | Meaning | Images | % |
|-:|-|-:|-:|
| 0 | No retinopathy | 15,188 | 93.37 % |
| 1 | Mild background | 280 | 1.72 % |
| 2 | Moderate background | 133 | 0.82 % |
| 3 | Severe NPDR / pre-proliferative | 263 | 1.62 % |
| 4 | Proliferative DR / post-laser | 402 | 2.47 % |

> The binary `diabetic_retinopathy` label essentially equals `DR_ICDR ≥ 1`, but **~15 rows disagree** (1 image graded ICDR 0 yet flagged DR; 14 images graded ICDR ≥ 1 yet flagged 0) — see `results/dr_icdr_vs_drflag.csv` and §7. Decide which field is authoritative for your task.

### 3.7 Multi-label disease labels (binary, full pass)

Thirteen image-level **0/1** conditions. **Severely imbalanced** — only two exceed 10 % and the rarest has 7 positives:

| Label | Positives | % of images | Notes |
|-|-:|-:|-|
| `increased_cup_disc` | 3,205 | 19.70 % | large cup/disc ratio (glaucoma suspect) |
| `drusens` | 2,833 | 17.42 % | drusen deposits |
| `diabetic_retinopathy` | 1,070 | 6.58 % | ≈ `DR_ICDR ≥ 1` |
| `other` | 820 | 5.04 % | catch-all other findings |
| `macular_edema` | 401 | 2.47 % | |
| `amd` | 299 | 1.84 % | age-related macular degeneration |
| `scar` | 291 | 1.79 % | e.g. toxoplasmosis scar |
| `hypertensive_retinopathy` | 284 | 1.75 % | |
| `myopic_fundus` | 270 | 1.66 % | |
| `nevus` | 130 | 0.80 % | |
| `vascular_occlusion` | 101 | 0.62 % | |
| `hemorrhage` | 95 | 0.58 % | non-diabetic retinal hemorrhage |
| `retinal_detachment` | **7** | **0.04 %** | **too few for reliable train/eval** |

Full table: `results/label_prevalence.csv`.

### 3.8 Normal vs abnormal & multi-label structure

- **No-finding rate:** **8,461 images (52.0 %)** have **none** of the 13 binary disease labels positive. Mean positive labels per image **0.60**; **max 3** (most abnormal images carry a single label). Distribution: `results/labels_per_image.csv`.
- Treat classification as **multi-label**, not single-class. Because `increased_cup_disc` and `drusens` are common and the rest are rare, per-class metrics on the long tail will be statistically thin.

---

## 4. Data splits

**No train/validation/test split is predefined** in BRSET — `labels_brset.csv` is one flat table.

- **Split by patient, not by image.** A patient contributes **1–4 images** (typically both eyes; mean 1.91, max 4), so a naive per-image split leaks the same patient (and fellow eye) across folds. Group on **`patient_id`** (8,524 groups).
- **Stratify** on your target — DR grade and the rare disease labels (e.g. `retinal_detachment` n=7, `hemorrhage` n=95) — so that minority classes appear in every fold.
- The dataset's own paper and most downstream work construct their own patient-level splits; there is no "official" one to reproduce.

---

## 5. Intended uses / tasks

1. **Diabetic-retinopathy detection & grading.** Binary (`diabetic_retinopathy`, or `DR_ICDR ≥ 1`) or ordinal 5-class grading (`DR_ICDR` / `DR_SDRG`). Input → fundus image; output → DR presence/severity. The dataset's headline task.
2. **Multi-label retinal-disease classification.** Predict the 13 disease labels jointly (image → 13×{0,1}).
3. **Diabetic macular edema** detection (`macular_edema`), often jointly with DR.
4. **Glaucoma-suspect screening** via `increased_cup_disc` (large cup/disc ratio).
5. **Anatomical normal/abnormal classification** of optic disc, vessels, macula (`1` vs `2`).
6. **Automated image-quality assessment** — predict `quality` (Adequate/Inadequate) and the 4 sub-parameters; useful as a pre-filter in screening pipelines.
7. **Fairness, domain-shift & external validation.** As a **Brazilian / under-represented** cohort, BRSET is a strong out-of-distribution test set for DR models trained on EyePACS, Messidor-2, APTOS, etc., and for studying demographic bias (sex is complete; age 66.5 % complete).

**Limitations of use:**
- **Severe class imbalance / long tail** — many labels < 2 %; `retinal_detachment` has only 7 positives. Rare-class metrics are unreliable.
- **Image-level labels only** — no segmentation masks or lesion bounding boxes.
- **Single country (Brazil), 2010–2020, posterior-pole 45° only** — not representative of wide-field, OCT, pediatric/ROP, or non-Brazilian populations.
- **Missing metadata** — age 33.5 % missing; diabetes duration / insulin ~89 % missing; comorbidities 50.6 % missing and free-text.
- **Two cameras, 6 fixed resolutions** — watch for camera-specific shortcutting; consider camera as a confounder.
- **Not for clinical use.** Research / education only (per license).

---

## 6. How to load & use the data

**Environment.** A working interpreter on this machine: `/mnt/data3/hongyu/miniconda3/envs/biomni_e1/bin/python3` (has `pandas 2.3.1`, `numpy 1.26.4`, `Pillow 11.3.0`). JPEGs need only Pillow/OpenCV — no special codec (unlike the DICOMs in the sibling VinDr-CXR dataset).

**Load the labels and one image:**
```python
import pandas as pd
from PIL import Image

ROOT = "/mnt/data1/BRSET"
df = pd.read_csv(f"{ROOT}/labels_brset.csv")          # 16,266 rows × 34 cols
row = df.iloc[0]                                       # image_id == 'img00001'
img = Image.open(f"{ROOT}/fundus_photos/{row.image_id}.jpg")   # 8-bit RGB JPEG
print(img.size, img.mode)                             # e.g. (2672, 2056) RGB
```

**Build a clean DR-grading frame (drop the 7 rows whose file is missing):**
```python
import os
df["path"] = ROOT + "/fundus_photos/" + df["image_id"] + ".jpg"
df = df[df["path"].map(os.path.exists)].copy()        # 16,266 -> 16,259
y_binary = (df["DR_ICDR"] >= 1).astype(int)           # referable-ish DR target
y_grade  = df["DR_ICDR"]                               # 0..4 ordinal
groups   = df["patient_id"]                            # use for GroupKFold (no leakage)
```

**Keys & joins:**
- The single join key is **`image_id`** (`imgNNNNN`) = the JPEG filename stem. It links `labels_brset.csv` ↔ `fundus_photos/<image_id>.jpg` 1:1.
- **`patient_id`** groups a patient's 1–4 images (both eyes / repeats). Use it for splitting and for any per-patient analysis.
- **No cross-dataset / cross-modality link** exists (no EHR, no OCT, no reports).

**Coding schemes (interpret coded columns):**
- `patient_sex`: **1 = male, 2 = female**.
- `exam_eye`: **1 = right (OD), 2 = left (OS)**.
- `optic_disc`, `vessels`, `macula`, `focus`, `Illuminaton`, `image_field`, `artifacts`: **1 = normal, 2 = abnormal**.
- `quality`: text `Adequate` / `Inadequate`.
- `DR_ICDR`, `DR_SDRG`: **0–4** severity (tables in §3.6).
- 13 disease columns + `diabetes` / `insuline`: presence flags (disease columns are 0/1; `diabetes`/`insuline` are text `yes`/`no`).

**Verify integrity:** SHA-256 of every payload file is in `SHA256SUMS.txt` (run `sha256sum -c` from `/mnt/data1/BRSET/`).

---

## 7. Data quality, caveats & known issues (flagged during the full pass, 2026-06-21)

- **7 label rows have no image on disk.** `image_id` ∈ {`img04414`, `img04415`, `img04416`, `img04417`, `img04418`, `img04419`, `img06900`} appear in `labels_brset.csv` but have **no `.jpg`** in `fundus_photos/` (16,266 labels vs 16,259 files). Filter to existing files before training (snippet in §6). Conversely, **0** images lack a label and there are **0** duplicate `image_id`s.
- **`Illuminaton` is misspelled.** Despite the v1.0.1 note about fixing an `ilumination` typo, the actual CSV column header is **`Illuminaton`** (missing the second `i`). Reference it by that exact string.
- **`optic_disc` has 1 out-of-codebook value** `bv` (1 row) instead of `1`/`2`.
- **`focus` has 2 out-of-codebook values** `0` (2 rows) instead of `1`/`2`.
- **DR binary flag vs ICDR grade disagree on ~15 rows** (`diabetic_retinopathy` vs `DR_ICDR`): 1 row ICDR 0 but flagged, 14 rows ICDR ≥ 1 but not flagged. Pick the authoritative field per task (`results/dr_icdr_vs_drflag.csv`).
- **High missingness** in several metadata columns: `insuline` 89.5 %, `diabetes_time_y` 88.5 %, `comorbidities` 50.6 %, `patient_age` 33.5 %. Don't build analyses that assume these are populated.
- **`nationality` is constant** (`Brazil`, 100 %) — zero-variance column.
- **Mixed value casing:** `diabetes` uses `yes`/`No`; `insuline` uses `yes`/`no`. Normalize case before grouping.
- **Camera naming differs from the paper:** CSV `Canon CR` / `NIKON NF5050` correspond to the Canon CR‑2 / Nikon NF505 described in the publication.
- **Severe label imbalance** (§3.7) — 52 % of images carry no disease label and most positive classes are < 2 %.

**Known to be missing / not in this dataset:** per-image acquisition dates; OCT or wide-field images; lesion segmentation masks / bounding boxes; free-text reports; visual-acuity or other clinical outcomes; any non-Brazilian cohort; a predefined train/val/test split.

---

## 8. Ethics, de-identification, licensing & citation

- **De-identification / ethics:** approved by the **São Paulo Federal University (UNIFESP) IRB (CAAE 33842220.7.0000.5505)**; the requirement for individual consent was **waived**, and **all images were anonymized with identifiable patient information removed**. Treat as PHI-adjacent regardless; **do not attempt re-identification** (License clause 1).
- **License / access:** **PhysioNet Credentialed Health Data License 1.5.0** (`LICENSE.txt`). Requires a credentialed PhysioNet account, completion of **CITI "Data or Specimens Only Research"** training, and a signed Data Use Agreement. **You may not redistribute the data or share access** (clauses 3–4). Research/educational use only (clause 6).
- **Citation (required):**
  - Nakayama, L. F., Goncalves, M., Zago Ribeiro, L., Santos, H., Ferraz, D., Malerbi, F., Celi, L. A., & Regatieri, C. (2024). *A Brazilian Multilabel Ophthalmological Dataset (BRSET)* (version 1.0.1). **PhysioNet.** https://doi.org/10.13026/1pht-2b69
  - Nakayama, L. F., Restrepo, D., Matos, J., Ribeiro, L. Z., Malerbi, F. K., Celi, L. A., & Regatieri, C. S. (2024). *BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos.* **PLOS Digital Health** 3(7): e0000454.
  - Goldberger, A., et al. (2000). *PhysioNet: Components of a new research resource for complex physiologic signals.* **Circulation** 101(23): e215–e220.
- **Source-of-truth warning:** this is a **read-only source download**, relocated to `/mnt/data1/BRSET/` on 2026-06-21 (flattened from the original `/mnt/data1/physionet.org/files/brazilian-ophthalmological/1.0.1/`). **Do not modify, rename, or delete** any file under `/mnt/data1/BRSET/`. Write any derived data (resized images, splits, features) to your **own scratch directory** (e.g. `/mnt/data3/$USER/…`), not here.

---

## 9. Provenance of this README

- **Scope:** documents `/mnt/data1/BRSET/` (BRSET v1.0.1, PhysioNet `brazilian-ophthalmological`).
- **Statistics computed:** **2026-06-21**, by a **full pass** over the data on this disk:
  - All label distributions, demographics, DR-grade and disease-label counts, missingness, and the image↔label reconciliation are exact, computed from `labels_brset.csv` (16,266 rows) with pandas.
  - All image statistics (count, RGB mode, resolution distribution, file sizes) are a **full pass over all 16,259 JPGs** via Pillow header reads (no sampling).
- **Reproduce:** `/mnt/data3/hongyu/miniconda3/envs/biomni_e1/bin/python3 dataset_analysis/scripts/analyze_brset.py` → regenerates `dataset_analysis/results/*.csv` and `brset_summary.json`.
- **Domain background** (§3.1) is general color-fundus-photography knowledge plus the dataset's own methodology (Nakayama et al., *PLOS Digital Health* 2024); the **codebook** (sex/eye/anatomy/quality codings, DR scales) is from the official PhysioNet project documentation. All **quantitative** claims are computed from this copy of the data on 2026-06-21.
