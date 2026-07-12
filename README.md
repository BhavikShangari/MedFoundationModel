# BRSET Clinical Review UI

Flask application for the BRSET human review study with two modes:

- Human only
- Human + Dinomaly

The app shows each test scan, patient age and gender, optional Dinomaly retrieval evidence, anomaly maps, and saves doctor assessments with per-selected-disease certainty levels.

## Repository Contents

- `brset_review_app.py` - Flask app used by Render and local runs.
- `brset_ai_human/` - required dataset and retrieval artifacts:
  - `BRSET/fundus_photos/` original fundus images.
  - `normal_model/original_img/` test case images.
  - `normal_model/anomaly_scan/` anomaly maps.
  - `normal_model/brset_normal_test_name.pkl`, `normal_model/brset_normal_train_name.pkl`.
  - `normal_model/similarity_data.pkl`, `normal_model/indices_data.pkl`.
  - `normal_model/brset_normal_*_feats.pt`, `normal_model/brset_normal_*_labels.pt`.
  - `brset_dataset_distribution.csv` metadata and labels.
- `requirements.txt`, `Procfile`, `render.yaml` - deployment files.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python brset_review_app.py
```

Open:

```text
http://127.0.0.1:8502
```

## Push to GitHub

This repo contains large images and feature artifacts. Use Git LFS before the first commit.

```bash
git init
git lfs install
git add .gitattributes
git add .
git commit -m "Initial BRSET review app"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

The `brset_ai_human/` directory is about 16 GB. GitHub and Render deployments may require a paid Git LFS quota or external storage for this amount of data.

## Render Deploy

Render can use the included `render.yaml`, or you can create a Python web service manually.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn brset_review_app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

Environment variables:

```text
BRSET_REVIEW_SECRET_KEY=<random-secret>
```

## Runtime State

Responses are written to:

- `doctor_review_responses.csv`
- `doctor_review_session_log.csv`
- `doctor_review_sessions.json`
- `doctor_review_case_status.json`

On Render's normal filesystem, these files are not a durable database. For a production study, attach a persistent disk or move response saving to a database/object store.

These runtime files are intentionally ignored by Git so local test logs and doctor responses do not get pushed.
