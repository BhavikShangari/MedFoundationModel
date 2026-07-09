from __future__ import annotations

import csv
import io
import json
import os
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, send_file, session as browser_session, url_for
from PIL import Image, ImageOps


DISEASE_COLUMNS = [
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusens",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
]

NO_DISEASE_OPTION = "No Disease"
ASSESSMENT_OPTIONS = [NO_DISEASE_OPTION, *DISEASE_COLUMNS]
METHOD_MODES = {"combined"}
DINOMALY_RETRIEVAL_COUNT = 5
DINOMALY_PREDICTION_MIN_VOTES = 3
CERTAINTY_LEVELS = ("low", "medium", "high")


def assessment_field_name(label: str) -> str:
    return label.lower().replace(" ", "_")


def parse_assessment_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return [item for item in text.split(";") if item]


RESPONSE_FIELDS = [
    "timestamp",
    "reviewer_session_uid",
    "doctor_name",
    "designation",
    "department",
    "years_experience",
    "hospital_name",
    "posting_location",
    "registration_id",
    "contact",
    "contact_key",
    "mode",
    "case_number",
    "image_id",
    "patient_id",
    "doctor_selected_no_disease",
    "doctor_selected_diseases",
    "doctor_selected_count",
    "needs_recheck",
    "selected_disease_certainty_json",
    "review_time_seconds",
    "comments",
    "dinomaly_predicted_diseases",
    "dinomaly_vote_counts_json",
    "dinomaly_evidence_json",
    "retrieved_image_ids",
    "retrieved_similarities",
    "retrieved_diseases_json",
    "true_diseases_hidden_from_reviewer",
    "true_disease_count",
    "true_disease_category",
]

SESSION_LOG_FIELDS = [
    "timestamp",
    "event",
    "reviewer_session_uid",
    "doctor_name",
    "contact",
    "mode",
    "case_number",
    "image_id",
    "answered_human",
    "answered_dinomaly",
    "answered_combined",
    "total_cases",
    "details_json",
]

MODE_LABELS = {
    "human": "Human only",
    "combined": "Human + Dinomaly",
}

ROOT = Path(__file__).resolve().parent
METADATA_CSV = ROOT / "brset_ai_human/brset_dataset_distribution.csv"
TEST_NAMES_PKL = ROOT / "brset_ai_human/brset_normal_test_name.pkl"
TRAIN_NAMES_PKL = ROOT / "brset_ai_human/brset_normal_train_name.pkl"
SIMILARITIES_PKL = ROOT / "brset_ai_human/similarity_data.pkl"
INDICES_PKL = ROOT / "brset_ai_human/indices_data.pkl"
BRSET_DIR = ROOT / "brset_ai_human/BRSET"
TEST_IMAGE_DIR = ROOT / "brset_ai_human/normal_model/original_img"
NORMAL_MODEL_DIR = ROOT / "brset_ai_human/normal_model"
RETRIEVAL_IMAGE_DIR = ROOT / "brset_ai_human/BRSET/fundus_photos"
ANOMALY_DIR = ROOT / "brset_ai_human/normal_model/anomaly_scan"
RESPONSES_CSV = ROOT / "doctor_review_responses.csv"
SESSION_LOG_CSV = ROOT / "doctor_review_session_log.csv"
SESSIONS_JSON = ROOT / "doctor_review_sessions.json"
CASE_STATUS_JSON = ROOT / "doctor_review_case_status.json"

app = Flask(__name__)
app.secret_key = os.environ.get("BRSET_REVIEW_SECRET_KEY", "brset-review-local-secret")


class FlipBasedOnBrightness:
    def __init__(self, patch_size: int = 50) -> None:
        self.patch_size = patch_size

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        gray_np = np.array(gray)
        _, width = gray_np.shape
        patch_size = min(self.patch_size, width)

        max_brightness_sum = 0
        max_x = 0
        for x in range(0, width - patch_size + 1, patch_size):
            patch = gray_np[:, x : x + patch_size]
            patch_sum = patch.sum()
            if patch_sum > max_brightness_sum:
                max_brightness_sum = patch_sum
                max_x = x

        if max_x < width // 2:
            return ImageOps.mirror(image)
        return image


query_image_transform = FlipBasedOnBrightness()


def normalize_image_id(value: Any) -> str:
    image_id = str(value)
    if image_id.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        return Path(image_id).stem
    return image_id


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def metadata_df() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)
    df["image_id"] = df["image_id"].map(normalize_image_id)
    return df


@lru_cache(maxsize=1)
def test_names() -> list[str]:
    folder_names = sorted(
        normalize_image_id(path.name)
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file()
    )
    if folder_names:
        return folder_names
    return [normalize_image_id(name) for name in load_pickle(TEST_NAMES_PKL)]


@lru_cache(maxsize=1)
def train_names() -> list[str]:
    return [normalize_image_id(name) for name in load_pickle(TRAIN_NAMES_PKL)]


@lru_cache(maxsize=1)
def similarities() -> np.ndarray:
    return np.asarray(load_pickle(SIMILARITIES_PKL))


@lru_cache(maxsize=1)
def indices() -> np.ndarray:
    return np.asarray(load_pickle(INDICES_PKL))


def metadata_lookup(image_id: str) -> pd.Series | None:
    match = metadata_df()[metadata_df()["image_id"].eq(normalize_image_id(image_id))]
    if match.empty:
        return None
    return match.iloc[0]


def present_diseases(row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    return [column for column in DISEASE_COLUMNS if int(row.get(column, 0)) == 1]


def sex_label(value: Any) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "Unknown"
    if code == 1:
        return "Male"
    if code == 2:
        return "Female"
    return f"Code {code}"


def patient_demographics(row: pd.Series | None) -> dict[str, str]:
    if row is None:
        return {"age": "Unknown", "gender": "Unknown"}
    age = row.get("patient_age", "")
    if pd.isna(age) or age == "":
        age_text = "Unknown"
    else:
        try:
            age_float = float(age)
            age_text = str(int(age_float)) if age_float.is_integer() else f"{age_float:.1f}"
        except (TypeError, ValueError):
            age_text = str(age)
    return {"age": age_text, "gender": sex_label(row.get("patient_sex", ""))}


def resolve_image_path(
    image_id: str,
    search_dirs: list[Path],
    recursive_dirs: list[Path] | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> Path | None:
    image_id = normalize_image_id(image_id)
    for directory in search_dirs:
        raw = directory / image_id
        if raw.exists():
            return raw
        for extension in extensions:
            candidate = directory / f"{image_id}{extension}"
            if candidate.exists():
                return candidate
    for directory in recursive_dirs or []:
        if not directory.exists():
            continue
        for extension in extensions:
            matches = list(directory.rglob(f"{image_id}{extension}"))
            if matches:
                return matches[0]
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(preferred_fields)
    strict_schema = tuple(preferred_fields) in {tuple(RESPONSE_FIELDS), tuple(SESSION_LOG_FIELDS)}
    if not strict_schema:
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {key: str(row.get(key, "")) for key in fields}
                for row in rows
            ]
        )


def load_sessions() -> dict[str, dict[str, Any]]:
    if not SESSIONS_JSON.exists():
        return {}
    return json.loads(SESSIONS_JSON.read_text(encoding="utf-8"))


def save_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    SESSIONS_JSON.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def load_case_statuses() -> dict[str, Any]:
    if not CASE_STATUS_JSON.exists():
        return {}
    return json.loads(CASE_STATUS_JSON.read_text(encoding="utf-8"))


def save_case_statuses(statuses: dict[str, Any]) -> None:
    CASE_STATUS_JSON.write_text(json.dumps(statuses, indent=2), encoding="utf-8")


def status_bucket(statuses: dict[str, Any], session_id: str, mode: str) -> dict[str, Any]:
    return statuses.setdefault(session_id, {}).setdefault(mode, {})


def mark_case_status(
    session_id: str,
    mode: str,
    image_id: str,
    state: str,
    needs_recheck: bool | None = None,
) -> None:
    statuses = load_case_statuses()
    bucket = status_bucket(statuses, session_id, mode)
    image_id = normalize_image_id(image_id)
    current = bucket.get(image_id, {})
    current["state"] = state
    current["updated_at"] = now()
    if needs_recheck is not None:
        current["needs_recheck"] = bool(needs_recheck)
    bucket[image_id] = current
    save_case_statuses(statuses)


def case_status_for(
    session_id: str,
    mode: str,
    image_id: str,
    answered: set[str],
    statuses: dict[str, Any] | None = None,
) -> str:
    image_id = normalize_image_id(image_id)
    statuses = statuses if statuses is not None else load_case_statuses()
    stored = statuses.get(session_id, {}).get(mode, {}).get(image_id, {})
    if image_id in answered:
        return "recheck" if stored.get("needs_recheck") else "saved"
    if stored.get("state") == "opened":
        return "opened"
    return "new"


def needs_recheck_for(session_id: str, mode: str, image_id: str) -> bool:
    statuses = load_case_statuses()
    stored = statuses.get(session_id, {}).get(mode, {}).get(normalize_image_id(image_id), {})
    return bool(stored.get("needs_recheck"))


def normalize_contact(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def session_profile_from_form(form: Any, session_id: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    contact = form.get("contact", "")
    return {
        "doctor_name": form.get("doctor_name", ""),
        "designation": form.get("designation", ""),
        "department": form.get("department", ""),
        "years_experience": form.get("years_experience", "0"),
        "hospital_name": form.get("hospital_name", ""),
        "posting_location": form.get("posting_location", ""),
        "registration_id": form.get("registration_id", ""),
        "contact": contact,
        "contact_key": normalize_contact(contact),
        "session_id": session_id,
        "session_notes": form.get("session_notes", ""),
        "created_at": existing.get("created_at", existing.get("updated_at", now())),
        "updated_at": now(),
    }


def latest_matching_session(sessions: dict[str, dict[str, Any]], contact: str) -> tuple[str, dict[str, Any]] | None:
    contact_key = normalize_contact(contact)
    if not contact_key:
        return None
    matches = [
        (session_id, data)
        for session_id, data in sessions.items()
        if normalize_contact(data.get("contact_key") or data.get("contact")) == contact_key
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (str(item[1].get("updated_at", "")), item[0]), reverse=True)
    return matches[0]


def session_progress(session_id: str) -> dict[str, int]:
    return {mode: len(answered_ids(session_id, mode)) for mode in MODE_LABELS}


def active_session_id() -> str:
    return str(browser_session.get("review_session_id", ""))


def log_event(event: str, payload: dict[str, Any]) -> None:
    rows = read_csv_rows(SESSION_LOG_CSV)
    session_id = str(payload.get("session_id") or payload.get("reviewer_session_uid") or "")
    progress = session_progress(session_id) if session_id else {}
    detail_keys = {"deleted", "session_action"}
    if event in {"start_session", "resume_session"}:
        detail_keys |= {
            "designation",
            "department",
            "years_experience",
            "hospital_name",
            "posting_location",
            "registration_id",
            "session_notes",
            "created_at",
            "updated_at",
        }
    details = {key: payload[key] for key in detail_keys if key in payload and payload[key] != ""}
    row = {
        "timestamp": now(),
        "event": event,
        "reviewer_session_uid": session_id,
        "doctor_name": payload.get("doctor_name", ""),
        "contact": payload.get("contact", ""),
        "mode": payload.get("mode", ""),
        "case_number": payload.get("case_number", ""),
        "image_id": payload.get("image_id", ""),
        "answered_human": progress.get("human", payload.get("answered_human", "")),
        "answered_dinomaly": payload.get("answered_dinomaly", ""),
        "answered_combined": progress.get("combined", payload.get("answered_combined", "")),
        "total_cases": payload.get("total_cases", len(test_names())),
        "details_json": json.dumps(details, sort_keys=True),
    }
    rows.append(row)
    write_csv_rows(SESSION_LOG_CSV, rows, SESSION_LOG_FIELDS)


def response_key(row: dict[str, Any]) -> tuple[str, str, str]:
    session_id = row.get("reviewer_session_uid") or row.get("session_id") or ""
    return str(session_id), str(row.get("mode", "")), str(row.get("image_id", ""))


def answered_ids(session_id: str, mode: str) -> set[str]:
    return {
        normalize_image_id(row["image_id"])
        for row in read_csv_rows(RESPONSES_CSV)
        if (row.get("reviewer_session_uid") or row.get("session_id")) == session_id
        and row.get("mode") == mode
    }


def saved_response(session_id: str, mode: str, image_id: str) -> dict[str, str] | None:
    for row in reversed(read_csv_rows(RESPONSES_CSV)):
        if response_key(row) == (session_id, mode, normalize_image_id(image_id)):
            return row
    return None


def next_unanswered(answered: set[str], start_index: int = 0) -> int:
    names = test_names()
    for offset in range(len(names)):
        index = (start_index + offset) % len(names)
        if names[index] not in answered:
            return index
    return min(start_index, len(names) - 1)


def retrieval_rows(test_index: int, top_k: int) -> list[dict[str, Any]]:
    rows = []
    for rank, train_index in enumerate(indices()[test_index].tolist()[:top_k], start=1):
        image_id = train_names()[int(train_index)]
        row = metadata_lookup(image_id)
        rows.append(
            {
                "rank": rank,
                "image_id": image_id,
                "similarity": float(similarities()[test_index][rank - 1]),
                "diseases": present_diseases(row),
            }
        )
    return rows


def method_prediction(retrieved: list[dict[str, Any]], min_votes: int) -> dict[str, Any]:
    labels = [NO_DISEASE_OPTION, *DISEASE_COLUMNS]
    votes = {label: 0 for label in labels}
    similarity_sums = {label: 0.0 for label in labels}
    supporting_image_ids = {label: [] for label in labels}
    for item in retrieved:
        item_labels = item["diseases"] or [NO_DISEASE_OPTION]
        for label in item_labels:
            votes[label] += 1
            similarity_sums[label] += item["similarity"]
            supporting_image_ids[label].append(item["image_id"])
    predicted = [label for label in labels if votes[label] >= min_votes]
    evidence = [
        {
            "disease": label,
            "count": votes[label],
            "mean_similarity": round(similarity_sums[label] / votes[label], 4),
            "supporting_image_ids": supporting_image_ids[label],
        }
        for label in labels
        if votes[label] > 0
    ]
    evidence.sort(key=lambda x: (x["count"], x["mean_similarity"]), reverse=True)
    return {"predicted": predicted, "evidence": evidence}


def certainty_from_numeric(value: Any) -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return "medium"
    if numeric >= 75:
        return "high"
    if numeric <= 40:
        return "low"
    return "medium"


def disease_certainties_from_response(previous: dict[str, str]) -> dict[str, str]:
    defaults = {disease: "high" for disease in ASSESSMENT_OPTIONS}
    raw = previous.get("selected_disease_certainty_json", "")
    if raw:
        try:
            loaded = json.loads(raw)
            for disease, certainty in loaded.items():
                if disease in defaults and certainty in CERTAINTY_LEVELS:
                    defaults[disease] = certainty
            return defaults
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    legacy = previous.get("selected_disease_confidences_json") or previous.get("selected_disease_confidences", "")
    if legacy:
        try:
            loaded = json.loads(legacy)
            for disease, value in loaded.items():
                if disease in defaults:
                    defaults[disease] = certainty_from_numeric(value)
            return defaults
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    for disease in ASSESSMENT_OPTIONS:
        field = f"confidence_{assessment_field_name(disease)}"
        if previous.get(field):
            defaults[disease] = certainty_from_numeric(previous[field])
    return defaults


def upsert_response(response: dict[str, Any]) -> None:
    rows = read_csv_rows(RESPONSES_CSV)
    key = response_key(response)
    rows = [row for row in rows if response_key(row) != key]
    rows.append(response)
    write_csv_rows(RESPONSES_CSV, rows, RESPONSE_FIELDS)


def delete_response(session_id: str, mode: str, image_id: str) -> bool:
    rows = read_csv_rows(RESPONSES_CSV)
    key = (session_id, mode, normalize_image_id(image_id))
    kept = [row for row in rows if response_key(row) != key]
    deleted = len(kept) != len(rows)
    if rows:
        write_csv_rows(RESPONSES_CSV, kept, list(rows[0].keys()))
    return deleted


def missing_svg() -> Response:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
      <rect width="640" height="480" rx="12" fill="#f6f8fb"/>
      <rect x="24" y="24" width="592" height="432" rx="10" fill="none" stroke="#cbd5e1" stroke-dasharray="10 10" stroke-width="3"/>
      <text x="320" y="242" text-anchor="middle" font-family="Arial, sans-serif" font-size="26" fill="#64748b">Image unavailable</text>
    </svg>
    """
    return Response(svg, mimetype="image/svg+xml")


def send_resolved_image(path: Path | None) -> Response:
    if path is None:
        return missing_svg()
    return send_file(path)


def send_transformed_query_image(path: Path | None) -> Response:
    if path is None:
        return missing_svg()
    image = Image.open(path).convert("RGB")
    image = query_image_transform(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg")


CSS = """
:root {
  --bg: #f4f7fb;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #617081;
  --line: #d7dee8;
  --accent: #0f766e;
  --accent-soft: #dff4f0;
  --warn: #fff4db;
  --warn-line: #d49a22;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  letter-spacing: 0;
}
a { color: inherit; text-decoration: none; }
.shell { min-height: 100vh; display: grid; grid-template-columns: 292px 1fr; }
.sidebar {
  background: #ffffff;
  border-right: 1px solid var(--line);
  padding: 22px 18px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
}
.brand { font-size: 20px; font-weight: 760; margin-bottom: 4px; }
.brand-sub { color: var(--muted); font-size: 13px; margin-bottom: 22px; }
.main { padding: 24px 28px 36px; }
.topbar {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 18px;
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
}
.title { font-size: 22px; font-weight: 760; margin: 0; }
.subtitle { color: var(--muted); font-size: 13px; margin-top: 3px; }
.metrics { display: flex; gap: 10px; flex-wrap: wrap; }
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f9fbfd;
  padding: 8px 10px;
  min-width: 112px;
}
.metric-label { color: var(--muted); font-size: 12px; }
.metric-value { font-weight: 760; font-size: 16px; margin-top: 2px; }
.mode-list { display: grid; gap: 8px; margin: 10px 0 18px; }
.mode-link {
  display: block;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 11px;
  background: #f8fafc;
  color: #334155;
  font-size: 14px;
}
.mode-link.active {
  background: var(--accent-soft);
  border-color: #9fd7cf;
  color: #075e56;
  font-weight: 700;
}
.case-board {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 5px;
  margin: 12px 0 14px;
}
.case-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  aspect-ratio: 1 / 1;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  font-size: 11px;
  font-weight: 760;
}
.case-cell:hover { border-color: #0f766e; box-shadow: 0 0 0 2px var(--accent-soft); }
.case-cell.active { outline: 2px solid #111827; outline-offset: 1px; }
.case-cell.opened { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }
.case-cell.saved { background: #dcfce7; border-color: #86efac; color: #166534; }
.case-cell.recheck { background: #ede9fe; border-color: #c4b5fd; color: #5b21b6; }
.case-legend {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 7px;
  color: var(--muted);
  font-size: 11px;
  margin-bottom: 14px;
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-swatch {
  width: 13px;
  height: 13px;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: #fff;
}
.legend-swatch.opened { background: #fee2e2; border-color: #fca5a5; }
.legend-swatch.saved { background: #dcfce7; border-color: #86efac; }
.legend-swatch.recheck { background: #ede9fe; border-color: #c4b5fd; }
.field { margin-bottom: 13px; }
label { display: block; color: #334155; font-weight: 650; font-size: 13px; margin-bottom: 6px; }
.help-text { color: var(--muted); font-size: 12px; margin-top: 5px; }
input, select, textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 11px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}
textarea { min-height: 96px; resize: vertical; }
.btn {
  border: 0;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  font-weight: 740;
  padding: 10px 14px;
  cursor: pointer;
}
.btn.secondary {
  background: #ffffff;
  color: #334155;
  border: 1px solid var(--line);
}
.case-header {
  background: linear-gradient(90deg, #ffffff 0%, #f8fbfb 100%);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 15px 16px;
  margin-bottom: 18px;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
}
.case-title { font-size: 20px; font-weight: 760; }
.case-meta { color: var(--muted); font-size: 13px; margin-top: 4px; }
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  align-items: start;
}
.evidence-grid {
  display: grid;
  grid-template-columns: minmax(280px, 0.82fr) minmax(340px, 1.18fr);
  gap: 18px;
  align-items: start;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.panel-title {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  margin-bottom: 10px;
}
.panel-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.panel-title-row .panel-title { margin-bottom: 0; }
.panel-description {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
  margin: -4px 0 10px;
}
.zoom-link {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 9px;
  color: #334155;
  background: #fff;
  font-size: 12px;
  font-weight: 700;
}
.zoom-link:hover { border-color: #9fd7cf; color: #075e56; background: var(--accent-soft); }
.scan-img {
  display: block;
  width: 100%;
  max-height: 620px;
  object-fit: contain;
  border-radius: 8px;
  background: #0b0f14;
}
.retrieval-list { display: grid; gap: 12px; }
.retrieval-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 10px;
}
.retrieval-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
}
.retrieval-img {
  width: 100%;
  max-height: 240px;
  object-fit: contain;
  border-radius: 8px;
  background: #0b0f14;
}
.pill {
  display: inline-block;
  margin: 7px 5px 0 0;
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 12px;
  background: #fff;
}
.pill.strong {
  background: var(--accent-soft);
  border-color: #9fd7cf;
  color: #075e56;
  font-weight: 740;
}
.note {
  border-left: 4px solid var(--accent);
  background: var(--accent-soft);
  border-radius: 6px;
  padding: 10px 11px;
  margin: 10px 0;
  font-size: 13px;
}
.saved {
  border-left-color: var(--warn-line);
  background: var(--warn);
}
.timer-box {
  border: 1px solid #9fd7cf;
  border-radius: 8px;
  background: var(--accent-soft);
  padding: 8px 12px;
  min-width: 112px;
  text-align: right;
}
.timer-label {
  color: #075e56;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.timer-value {
  font-size: 20px;
  font-weight: 780;
  margin-top: 2px;
}
.checkbox-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 7px;
  margin-top: 6px;
}
.checkbox-item {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 8px;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 9px;
}
.checkbox-main { display: contents; }
.checkbox-item input { width: auto; }
.certainty-row {
  grid-column: 1 / -1;
  display: none;
  grid-template-columns: 1fr 72px;
  gap: 8px;
  align-items: center;
  color: var(--muted);
  font-size: 12px;
}
.checkbox-item.selected .certainty-row { display: block; }
.certainty-options {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-top: 4px;
}
.certainty-option {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 6px 4px;
  background: #f8fafc;
  color: #334155;
  font-size: 12px;
  font-weight: 700;
}
.certainty-option input { width: auto; }
.certainty-option:has(input:checked) {
  background: var(--accent-soft);
  border-color: #9fd7cf;
  color: #075e56;
}
.actions { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; align-items: center; }
.btn.danger {
  background: #fff;
  color: #9f1239;
  border: 1px solid #fecdd3;
}
.advanced { margin-top: 16px; color: var(--muted); font-size: 12px; }
.profile-wrap { max-width: 980px; margin: 38px auto; padding: 0 20px; }
.profile-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; }
.profile-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 18px; }
.span-2 { grid-column: span 2; }
.zoom-page {
  min-height: 100vh;
  padding: 20px;
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 14px;
}
.zoom-toolbar {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
}
.zoom-title { font-weight: 760; }
.zoom-subtitle { color: var(--muted); font-size: 13px; margin-top: 2px; }
.zoom-stage {
  background: #080b0f;
  border-radius: 8px;
  border: 1px solid #18202b;
  overflow: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 18px;
}
.zoom-img {
  max-width: none;
  width: 1400px;
  height: auto;
  object-fit: contain;
}
.zoom-compare {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  width: 100%;
}
.zoom-panel {
  min-width: 0;
}
.zoom-panel-title {
  color: #e2e8f0;
  font-weight: 760;
  margin-bottom: 10px;
}
.zoom-compare-img {
  width: 100%;
  height: auto;
  object-fit: contain;
  border-radius: 8px;
}
@media (max-width: 1100px) {
  .shell { grid-template-columns: 1fr; }
  .sidebar { position: relative; height: auto; }
  .case-header { align-items: flex-start; flex-wrap: wrap; }
  .timer-box { text-align: left; }
  .workspace, .evidence-grid, .profile-grid, .zoom-compare { grid-template-columns: 1fr; }
  .span-2 { grid-column: span 1; }
}
"""


BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BRSET Review</title>
  <style>{{ css }}</style>
</head>
<body>
{{ body|safe }}
<script>
function syncDiseaseSelection(checkbox) {
  const item = checkbox.closest(".checkbox-item");
  if (!item) return;
  item.classList.toggle("selected", checkbox.checked);
}
function setupReviewTimer() {
  const input = document.querySelector('input[name="review_time_seconds"]');
  const value = document.querySelector("[data-review-timer-value]");
  if (!input || !value) return;

  const startAt = performance.now();
  let elapsedSeconds = 0;

  function formatSeconds(totalSeconds) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  function updateTimer() {
    elapsedSeconds = Math.max(0, Math.floor((performance.now() - startAt) / 1000));
    input.value = String(elapsedSeconds);
    value.textContent = formatSeconds(elapsedSeconds);
  }

  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", updateTimer);
  });
  updateTimer();
  window.setInterval(updateTimer, 1000);
}
document.addEventListener("change", function (event) {
  const target = event.target;
  if (target.matches('input[type="checkbox"][name="diseases"]')) {
    syncDiseaseSelection(target);
  }
});
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll('input[type="checkbox"][name="diseases"]').forEach(syncDiseaseSelection);
  setupReviewTimer();
});
</script>
</body>
</html>
"""


PROFILE_BODY = """
<div class="profile-wrap">
  <div class="topbar">
    <div>
      <h1 class="title">BRSET Clinical Review</h1>
      <div class="subtitle">Reviewer profile and session setup</div>
    </div>
  </div>
  <form class="profile-card" method="post" action="{{ url_for('start') }}">
    <div class="panel-title">Reviewer details</div>
    <div class="profile-grid">
      <div class="field"><label>Doctor name</label><input name="doctor_name" required></div>
      <div class="field"><label>Designation</label><input name="designation" required></div>
      <div class="field"><label>Department / unit</label><input name="department"></div>
      <div class="field"><label>Years of experience</label><input name="years_experience" type="number" min="0" max="70" value="0"></div>
      <div class="field"><label>Hospital name</label><input name="hospital_name" required></div>
      <div class="field"><label>Posting / location</label><input name="posting_location"></div>
      <div class="field"><label>Medical registration ID</label><input name="registration_id"></div>
      <div class="field">
        <label>Email</label>
        <input name="contact" type="email" required>
        <div class="help-text">(Use the same email to continue sessions.)</div>
      </div>
      <div class="field"><label>Initial mode</label>
        <select name="mode">
          {% for key, label in mode_labels.items() %}
          <option value="{{ key }}">{{ label }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="field span-2"><label>Session notes</label><textarea name="session_notes"></textarea></div>
    </div>
    <button class="btn" type="submit">Start review</button>
  </form>
</div>
"""


RESUME_BODY = """
<div class="profile-wrap">
  <div class="topbar">
    <div>
      <h1 class="title">Continue Previous Review?</h1>
      <div class="subtitle">A saved review profile matches this email.</div>
    </div>
  </div>
  <form class="profile-card" method="post" action="{{ url_for('start') }}">
    <div class="panel-title">Matched reviewer</div>
    <div class="metrics" style="margin-bottom:16px;">
      <div class="metric"><div class="metric-label">Doctor</div><div class="metric-value">{{ existing.doctor_name }}</div></div>
      <div class="metric"><div class="metric-label">Hospital</div><div class="metric-value">{{ existing.hospital_name }}</div></div>
      <div class="metric"><div class="metric-label">Last updated</div><div class="metric-value">{{ existing.updated_at }}</div></div>
    </div>
    <div class="panel-title">Saved cases</div>
    <div class="metrics" style="margin-bottom:16px;">
      {% for key, label in mode_labels.items() %}
      <div class="metric"><div class="metric-label">{{ label }}</div><div class="metric-value">{{ progress[key] }} / {{ total_cases }}</div></div>
      {% endfor %}
    </div>

    {% for key, value in submitted.items() %}
    <input type="hidden" name="{{ key }}" value="{{ value }}">
    {% endfor %}

    <div class="actions">
      <button class="btn" name="session_action" value="resume" type="submit">Continue older session</button>
      <button class="btn secondary" name="session_action" value="new" type="submit">Start new review</button>
    </div>
  </form>
</div>
"""


REVIEW_BODY = """
<div class="shell">
  <aside class="sidebar">
    <div class="brand">BRSET Review</div>
    <div class="brand-sub">{{ doctor.doctor_name }} | {{ doctor.designation }}</div>
    <div class="mode-list">
      {% for key, label in mode_labels.items() %}
      <a class="mode-link {% if key == mode %}active{% endif %}"
         href="{{ url_for('review', mode=key) }}">{{ label }}</a>
      {% endfor %}
    </div>
    <div class="panel-title">Cases</div>
    <div class="case-board">
      {% for case in case_cells %}
      <a class="case-cell {{ case.status }} {% if case.index == index %}active{% endif %}"
         title="Case {{ case.index + 1 }} - {{ case.status_label }}"
         href="{{ url_for('review', mode=mode, index=case.index) }}">{{ case.index + 1 }}</a>
      {% endfor %}
    </div>
    <div class="case-legend">
      <div class="legend-item"><span class="legend-swatch"></span>Not opened</div>
      <div class="legend-item"><span class="legend-swatch opened"></span>Opened</div>
      <div class="legend-item"><span class="legend-swatch saved"></span>Saved</div>
      <div class="legend-item"><span class="legend-swatch recheck"></span>Recheck</div>
    </div>
    <div class="advanced">
      Responses are saved automatically by reviewer, mode, and case.
    </div>
  </aside>
  <main class="main">
    <div class="topbar">
      <div>
        <h1 class="title">Clinical Review Workspace</h1>
        <div class="subtitle">{{ doctor.hospital_name }}{% if doctor.posting_location %} | {{ doctor.posting_location }}{% endif %}</div>
      </div>
    </div>

    <div class="case-header">
      <div>
        <div class="case-title">Case {{ index + 1 }} of {{ total_cases }}</div>
        <div class="case-meta">{{ mode_label }} | {{ doctor.doctor_name }} | Age {{ patient_demo.age }} | {{ patient_demo.gender }}</div>
      </div>
      <div class="timer-box">
        <div class="timer-label">Review time</div>
        <div class="timer-value" data-review-timer-value>00:00</div>
      </div>
    </div>
    {% if already_saved %}
    <div class="note saved">A response is already saved for this case. Saving again will update it.</div>
    {% endif %}

    <div class="workspace">
      <section>
        {% if mode == "human" %}
        <div class="panel">
          <div class="panel-title-row">
            <div class="panel-title">Scan</div>
            <a class="zoom-link" target="_blank" href="{{ url_for('zoom_image', kind='scan', image_id=image_id) }}">Zoom</a>
          </div>
          <div class="panel-description">Test scan image. The doctor reviews this image and decides which disease or diseases are present.</div>
          <img class="scan-img" src="{{ url_for('scan_image', image_id=image_id) }}" alt="Scan">
        </div>
        {% elif mode == "combined" %}
        <div class="evidence-grid">
          <div class="panel">
            <div class="panel-title-row">
              <div class="panel-title">Scan</div>
              <a class="zoom-link" target="_blank" href="{{ url_for('zoom_image', kind='scan', image_id=image_id) }}">Zoom</a>
            </div>
            <div class="panel-description">Test scan image.</div>
            <img class="scan-img" src="{{ url_for('scan_image', image_id=image_id) }}" alt="Scan">
            <div class="panel-title-row" style="margin-top:14px;">
              <div class="panel-title">Anomaly map</div>
              <a class="zoom-link" target="_blank" href="{{ url_for('zoom_compare', image_id=image_id) }}">Zoom</a>
            </div>
            <div class="panel-description">The AI model highlights regions it thinks are anomalous. This model is trained only using healthy images.</div>
            <img class="scan-img" src="{{ url_for('anomaly_image', image_id=image_id) }}" alt="Anomaly map">
            <div class="panel-title" style="margin-top:14px;">Dinomaly prediction</div>
            <div class="case-meta" style="margin-bottom:8px;">Showing labels present in at least 3 of the 5 retrieved scans.</div>
            {{ prediction_pills|safe }}
            {{ evidence_html|safe }}
          </div>
          <div class="panel">
            <div class="panel-title">Retrieved scans with labels</div>
            <div class="panel-description">Similar retrieved cases which Dinomaly thinks are similar to the current test case, with the diseases present in those retrieved cases.</div>
            {{ retrieval_cards|safe }}
          </div>
        </div>
        {% endif %}
      </section>

      <aside class="panel">
        <div class="panel-title">Assessment</div>
        <form method="post" action="{{ url_for('save') }}">
          <input type="hidden" name="mode" value="{{ mode }}">
          <input type="hidden" name="index" value="{{ index }}">
          <input type="hidden" name="image_id" value="{{ image_id }}">
          <input type="hidden" name="top_k" value="{{ top_k }}">
          <input type="hidden" name="min_votes" value="{{ min_votes }}">
          <label>Doctor diagnosis</label>
          <div class="checkbox-grid">
            {% for disease in disease_columns %}
            <div class="checkbox-item">
              <label class="checkbox-main">
                <input type="checkbox" name="diseases" value="{{ disease }}" {% if disease in previous_diseases %}checked{% endif %}>
                <span>{{ disease }}</span>
              </label>
              <div class="certainty-row">
                <span>How sure?</span>
                <span class="certainty-options">
                  {% for certainty in certainty_levels %}
                  <label class="certainty-option">
                    <input type="radio" name="certainty_{{ disease_field_names[disease] }}" value="{{ certainty }}" {% if disease_certainties[disease] == certainty %}checked{% endif %}>
                    <span>{{ certainty|capitalize }}</span>
                  </label>
                  {% endfor %}
                </span>
              </div>
            </div>
            {% endfor %}
          </div>
          <input name="review_time_seconds" type="hidden" value="0">
          <div class="field"><label>Comments</label><textarea name="comments">{{ previous_comments }}</textarea></div>
          <div class="actions">
            <button class="btn" name="action" value="save_next" type="submit">Save and next</button>
            <button class="btn secondary" name="action" value="save_recheck_next" type="submit">Save for recheck</button>
            <a class="btn secondary" href="{{ next_url }}">Next</a>
            <button class="btn danger" name="action" value="clear_current" type="submit" formnovalidate>Clear current</button>
          </div>
        </form>
      </aside>
    </div>
  </main>
</div>
"""


ZOOM_BODY = """
<div class="zoom-page">
  <div class="zoom-toolbar">
    <div>
      <div class="zoom-title">{{ title }}</div>
      <div class="zoom-subtitle">Use browser zoom or scroll to inspect the image.</div>
    </div>
    <a class="btn secondary" href="javascript:window.close()">Close</a>
  </div>
  <div class="zoom-stage">
    <img class="zoom-img" src="{{ image_url }}" alt="{{ title }}">
  </div>
</div>
"""


COMPARE_ZOOM_BODY = """
<div class="zoom-page">
  <div class="zoom-toolbar">
    <div>
      <div class="zoom-title">Scan and anomaly map zoom</div>
      <div class="zoom-subtitle">Use browser zoom or scroll to inspect both images side by side.</div>
    </div>
    <a class="btn secondary" href="javascript:window.close()">Close</a>
  </div>
  <div class="zoom-stage">
    <div class="zoom-compare">
      <div class="zoom-panel">
        <div class="zoom-panel-title">Scan</div>
        <img class="zoom-compare-img" src="{{ scan_url }}" alt="Scan">
      </div>
      <div class="zoom-panel">
        <div class="zoom-panel-title">Anomaly map</div>
        <img class="zoom-compare-img" src="{{ anomaly_url }}" alt="Anomaly map">
      </div>
    </div>
  </div>
</div>
"""


def render_page(body: str, **context: Any) -> str:
    return render_template_string(BASE_HTML, css=CSS, body=render_template_string(body, **context))


def pill_html(values: list[str], strong: bool = False, empty: str = "None") -> str:
    if not values:
        return f"<span class='pill'>{empty}</span>"
    class_name = "pill strong" if strong else "pill"
    return "".join(f"<span class='{class_name}'>{value}</span>" for value in values)


def retrieval_cards_html(retrieved: list[dict[str, Any]], show_labels: bool) -> str:
    cards = ["<div class='retrieval-list'>"]
    for item in retrieved:
        cards.append(
            "<div class='retrieval-card'>"
            f"<div class='retrieval-head'><span>Retrieved scan {item['rank']}</span>"
            f"<span>Similarity {item['similarity']:.3f}</span></div>"
            f"<img class='retrieval-img' src='{url_for('retrieval_image', image_id=item['image_id'])}' alt='Retrieved scan'>"
            f"<div style='margin-top:8px;'><a class='zoom-link' target='_blank' "
            f"href='{url_for('zoom_image', kind='retrieval', image_id=item['image_id'])}'>Zoom</a></div>"
        )
        if show_labels:
            cards.append(pill_html(item["diseases"], empty=NO_DISEASE_OPTION))
        cards.append("</div>")
    cards.append("</div>")
    return "".join(cards)


def evidence_html(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return ""
    rows = ["<div class='note'>"]
    for item in evidence:
        rows.append(
            f"<div><strong>{item['disease']}</strong>: {item['count']} / {DINOMALY_RETRIEVAL_COUNT} retrieved scans, "
            f"mean similarity {item['mean_similarity']:.4f}</div>"
        )
    rows.append("</div>")
    return "".join(rows)


def case_cells_for(session_id: str, mode: str, current_index: int) -> list[dict[str, Any]]:
    statuses = load_case_statuses()
    answered = answered_ids(session_id, mode)
    labels = {
        "new": "not opened",
        "opened": "opened, not saved",
        "saved": "saved",
        "recheck": "saved, marked for recheck",
    }
    cells = []
    for idx, image_id in enumerate(test_names()):
        status = case_status_for(session_id, mode, image_id, answered, statuses)
        cells.append(
            {
                "index": idx,
                "image_id": image_id,
                "status": status,
                "status_label": labels[status],
                "active": idx == current_index,
            }
        )
    return cells


@app.get("/")
def profile() -> str:
    return render_page(
        PROFILE_BODY,
        mode_labels=MODE_LABELS,
    )


@app.post("/start")
def start() -> str | Response:
    form = request.form
    sessions = load_sessions()
    mode = form.get("mode", "human")
    if mode not in MODE_LABELS:
        mode = "human"
    action = form.get("session_action", "check")
    existing_match = latest_matching_session(sessions, form.get("contact", ""))

    if action == "check" and existing_match is not None:
        existing_session_id, existing = existing_match
        browser_session["pending_resume_session_id"] = existing_session_id
        submitted = {
            key: form.get(key, "")
            for key in [
                "doctor_name",
                "designation",
                "department",
                "years_experience",
                "hospital_name",
                "posting_location",
                "registration_id",
                "contact",
                "mode",
                "session_notes",
            ]
        }
        log_event(
            "resume_prompt",
            {
                **existing,
                "mode": mode,
                "answered_count": sum(session_progress(existing_session_id).values()),
                "total_cases": len(test_names()),
            },
        )
        return render_page(
            RESUME_BODY,
            existing=existing,
            submitted=submitted,
            progress=session_progress(existing_session_id),
            total_cases=len(test_names()),
            mode_labels=MODE_LABELS,
        )

    pending_resume_session_id = str(browser_session.get("pending_resume_session_id", ""))
    if action == "resume" and pending_resume_session_id in sessions:
        session_id = pending_resume_session_id
        event = "resume_session"
    else:
        browser_session.pop("pending_resume_session_id", None)
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        while session_id in sessions:
            session_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        event = "start_session"

    sessions[session_id] = session_profile_from_form(form, session_id, sessions.get(session_id))
    save_sessions(sessions)
    browser_session["review_session_id"] = session_id
    browser_session.pop("pending_resume_session_id", None)
    answered = answered_ids(session_id, mode)
    log_event(event, {**sessions[session_id], "mode": mode, "answered_count": len(answered), "total_cases": len(test_names())})
    return redirect(url_for("review", mode=mode, index=next_unanswered(answered)))


@app.get("/review")
def review() -> str | Response:
    sessions = load_sessions()
    session_id = active_session_id()
    legacy_session_id = request.args.get("session_id", "")
    if legacy_session_id in sessions:
        browser_session["review_session_id"] = legacy_session_id
        session_id = legacy_session_id
    if not session_id or session_id not in sessions:
        return redirect(url_for("profile"))

    mode = request.args.get("mode", "human")
    if mode not in MODE_LABELS:
        mode = "human"
    top_k = DINOMALY_RETRIEVAL_COUNT
    min_votes = DINOMALY_PREDICTION_MIN_VOTES
    answered = answered_ids(session_id, mode)
    index_arg = request.args.get("index")
    index = next_unanswered(answered) if index_arg is None else int(index_arg)
    index = max(0, min(len(test_names()) - 1, index))
    image_id = test_names()[index]
    row = metadata_lookup(image_id)
    if image_id not in answered:
        mark_case_status(session_id, mode, image_id, "opened", needs_recheck=False)
    retrieved = retrieval_rows(index, top_k)
    prediction_min_votes = DINOMALY_PREDICTION_MIN_VOTES
    prediction = method_prediction(retrieved, prediction_min_votes)
    if mode in METHOD_MODES:
        prediction["evidence"] = [
            item for item in prediction["evidence"] if item["count"] >= prediction_min_votes
        ]
    previous = saved_response(session_id, mode, image_id) or {}
    previous_diseases = [
        disease
        for disease in parse_assessment_list(previous.get("doctor_selected_diseases", ""))
        if disease in ASSESSMENT_OPTIONS
    ]
    if previous.get("doctor_selected_no_disease") in {"1", "True", "true"}:
        previous_diseases.insert(0, NO_DISEASE_OPTION)
    needs_recheck = previous.get("needs_recheck") in {"1", "True", "true"} or needs_recheck_for(session_id, mode, image_id)

    previous_url = url_for("review", mode=mode, top_k=top_k, min_votes=min_votes, index=max(0, index - 1))
    next_url = url_for("review", mode=mode, top_k=top_k, min_votes=min_votes, index=min(len(test_names()) - 1, index + 1))
    mode_short = {"human": "Human", "combined": "Combined"}[mode]

    return render_page(
        REVIEW_BODY,
        doctor=sessions[session_id],
        mode=mode,
        mode_label=MODE_LABELS[mode],
        mode_short=mode_short,
        mode_labels=MODE_LABELS,
        patient_demo=patient_demographics(row),
        index=index,
        image_id=image_id,
        total_cases=len(test_names()),
        answered_count=len(answered),
        already_saved=image_id in answered,
        case_cells=case_cells_for(session_id, mode, index),
        top_k=top_k,
        min_votes=min_votes,
        retrieved=retrieved,
        retrieval_cards=retrieval_cards_html(retrieved, show_labels=mode == "combined"),
        prediction_pills=pill_html(prediction["predicted"], strong=True, empty="No prediction"),
        evidence_html=evidence_html(prediction["evidence"]),
        disease_columns=ASSESSMENT_OPTIONS,
        disease_field_names={disease: assessment_field_name(disease) for disease in ASSESSMENT_OPTIONS},
        previous_diseases=previous_diseases,
        certainty_levels=CERTAINTY_LEVELS,
        disease_certainties=disease_certainties_from_response(previous),
        previous_comments=previous.get("comments", ""),
        needs_recheck=needs_recheck,
        previous_url=previous_url,
        next_url=next_url,
    )


@app.post("/save")
def save() -> Response:
    form = request.form
    session_id = active_session_id()
    if not session_id:
        return redirect(url_for("profile"))
    mode = form["mode"]
    index = int(form["index"])
    image_id = normalize_image_id(form["image_id"])
    top_k = DINOMALY_RETRIEVAL_COUNT
    min_votes = DINOMALY_PREDICTION_MIN_VOTES
    action = form.get("action", "save_next")
    sessions = load_sessions()
    if session_id not in sessions:
        return redirect(url_for("profile"))
    doctor = sessions.get(session_id, {})

    if action == "clear_current":
        deleted = delete_response(session_id, mode, image_id)
        mark_case_status(session_id, mode, image_id, "opened", needs_recheck=False)
        log_event(
            "clear_response",
            {
                **doctor,
                "mode": mode,
                "image_id": image_id,
                "case_number": index + 1,
                "deleted": deleted,
                "answered_count": len(answered_ids(session_id, mode)),
                "total_cases": len(test_names()),
            },
        )
        return redirect(
            url_for(
                "review",
                mode=mode,
                top_k=top_k,
                min_votes=min_votes,
                index=index,
            )
        )

    retrieved = retrieval_rows(index, top_k) if mode in METHOD_MODES else []
    prediction = {"predicted": [], "evidence": []}
    if mode in METHOD_MODES:
        prediction_min_votes = DINOMALY_PREDICTION_MIN_VOTES
        prediction = method_prediction(retrieved, prediction_min_votes)
        prediction["evidence"] = [
            item for item in prediction["evidence"] if item["count"] >= prediction_min_votes
        ]
    row = metadata_lookup(image_id)
    true_diseases = present_diseases(row)
    selected = request.form.getlist("diseases")
    needs_recheck = action == "save_recheck_next"
    selected_diseases = [
        disease for disease in selected if disease != NO_DISEASE_OPTION
    ]
    selected_disease_certainties = {}
    for disease in selected:
        certainty = form.get(f"certainty_{assessment_field_name(disease)}", "high")
        selected_disease_certainties[disease] = certainty if certainty in CERTAINTY_LEVELS else "high"
    try:
        review_time_seconds = max(0, int(float(form.get("review_time_seconds", "0"))))
    except (TypeError, ValueError):
        review_time_seconds = 0
    vote_counts = {
        item["disease"]: item["count"]
        for item in prediction["evidence"]
    }
    dinomaly_evidence = {
        item["disease"]: {
            "votes": item["count"],
            "mean_similarity": item["mean_similarity"],
            "supporting_image_ids": item["supporting_image_ids"],
        }
        for item in prediction["evidence"]
    }

    response = {
        "timestamp": now(),
        "reviewer_session_uid": session_id,
        **doctor,
        "contact_key": normalize_contact(doctor.get("contact", "")),
        "mode": mode,
        "case_number": index + 1,
        "image_id": image_id,
        "patient_id": "" if row is None else int(row["patient_id"]),
        "doctor_selected_no_disease": int(NO_DISEASE_OPTION in selected),
        "doctor_selected_diseases": json.dumps(selected_diseases, sort_keys=True),
        "doctor_selected_count": len(selected),
        "needs_recheck": int(needs_recheck),
        "selected_disease_certainty_json": json.dumps(selected_disease_certainties, sort_keys=True),
        "review_time_seconds": review_time_seconds,
        "comments": form.get("comments", ""),
        "dinomaly_predicted_diseases": ";".join(prediction["predicted"]),
        "dinomaly_vote_counts_json": json.dumps(vote_counts, sort_keys=True),
        "dinomaly_evidence_json": json.dumps(dinomaly_evidence, sort_keys=True),
        "retrieved_image_ids": ";".join(item["image_id"] for item in retrieved),
        "retrieved_similarities": ";".join(f"{item['similarity']:.6f}" for item in retrieved),
        "retrieved_diseases_json": json.dumps({item["image_id"]: item["diseases"] for item in retrieved}, sort_keys=True),
        "true_diseases_hidden_from_reviewer": ";".join(true_diseases),
        "true_disease_count": "" if row is None else int(row["disease_count"]),
        "true_disease_category": "" if row is None else str(row["disease_category"]),
    }
    upsert_response(response)
    mark_case_status(session_id, mode, image_id, "saved", needs_recheck=needs_recheck)
    updated_answered = answered_ids(session_id, mode)
    log_event("save_response", {**doctor, "mode": mode, "case_number": index + 1, "image_id": image_id, "answered_count": len(updated_answered), "total_cases": len(test_names())})
    if len(updated_answered) == len(test_names()):
        log_event("complete_session", {**doctor, "mode": mode, "answered_count": len(updated_answered), "total_cases": len(test_names())})
    return redirect(url_for("review", mode=mode, top_k=top_k, min_votes=min_votes, index=next_unanswered(updated_answered, index + 1)))


@app.get("/scan/<image_id>")
def scan_image(image_id: str) -> Response:
    return send_transformed_query_image(
        resolve_image_path(
            image_id,
            [RETRIEVAL_IMAGE_DIR],
            recursive_dirs=[BRSET_DIR],
        )
    )


@app.get("/anomaly/<image_id>")
def anomaly_image(image_id: str) -> Response:
    return send_resolved_image(
        resolve_image_path(image_id, [ANOMALY_DIR], recursive_dirs=[NORMAL_MODEL_DIR])
    )


@app.get("/retrieval/<image_id>")
def retrieval_image(image_id: str) -> Response:
    return send_resolved_image(
        resolve_image_path(
            image_id,
            [RETRIEVAL_IMAGE_DIR, TEST_IMAGE_DIR],
            recursive_dirs=[BRSET_DIR, NORMAL_MODEL_DIR],
        )
    )


@app.get("/zoom/<kind>/<image_id>")
def zoom_image(kind: str, image_id: str) -> str | Response:
    if kind == "scan":
        title = "Scan zoom"
        image_url = url_for("scan_image", image_id=image_id)
    elif kind == "anomaly":
        title = "Anomaly map zoom"
        image_url = url_for("anomaly_image", image_id=image_id)
    elif kind == "retrieval":
        title = "Retrieved scan zoom"
        image_url = url_for("retrieval_image", image_id=image_id)
    else:
        return Response("Unknown image type", status=404)
    return render_page(ZOOM_BODY, title=title, image_url=image_url)


@app.get("/zoom/compare/<image_id>")
def zoom_compare(image_id: str) -> str:
    return render_page(
        COMPARE_ZOOM_BODY,
        scan_url=url_for("scan_image", image_id=image_id),
        anomaly_url=url_for("anomaly_image", image_id=image_id),
    )


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status=204)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8502"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
