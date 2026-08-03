from __future__ import annotations

import csv
import io
import json
import os
import pickle
import random
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
DISEASE_DIAGNOSIS_OPTIONS = [
    "diabetic_retinopathy",
    "macular_edema",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "retinal_detachment",
    "myopic_fundus",
    "other",
]
SIGN_FINDING_OPTIONS = [
    "scar",
    "drusens",
    "hemorrhage",
    "increased_cup_disc",
]
ASSESSMENT_GROUPS = [
    {
        "title": "Diseases / diagnoses",
        "description": "Select No Disease or all diagnoses that apply.",
        "options": [NO_DISEASE_OPTION, *DISEASE_DIAGNOSIS_OPTIONS],
    },
    {
        "title": "Signs / findings",
        "description": "Select all observed clinical signs or retinal findings that apply.",
        "options": SIGN_FINDING_OPTIONS,
    },
]
ASSESSMENT_OPTIONS = [NO_DISEASE_OPTION, *DISEASE_DIAGNOSIS_OPTIONS, *SIGN_FINDING_OPTIONS]
METHOD_MODES = {"combined"}
MODEL_SCOPED_MODES = {"combined"}
DINOMALY_RETRIEVAL_COUNT = 5
DINOMALY_PREDICTION_MIN_VOTES = 3
CERTAINTY_LEVELS = ("low", "medium", "high")


def assessment_field_name(label: str) -> str:
    return label.lower().replace(" ", "_")


def disease_display_label(label: str) -> str:
    return "Other Disease" if label == "other" else label


def parse_assessment_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
    "dinomaly_model",
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
    "dinomaly_model",
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
DEFAULT_DINOMALY_MODEL = "dinomaly_h"
HUMAN_MODEL_KEY = "none"
DINOMALY_MODELS = {
    "dinomaly_h": {
        "label": "Dinomaly-H",
        "training": "Healthy-only pretraining",
        "description": "Dinomaly trained using healthy images only. The doctor sees the test scan, anomaly map, and retrieved cases selected by this model.",
        "folder": ROOT / "brset_ai_human/dinomaly_h",
        "test_names": "brset_normal_test_name.pkl",
        "train_names": "brset_normal_train_name.pkl",
    },
    "dinomaly_hd": {
        "label": "Dinomaly-HD",
        "training": "Healthy + diseased pretraining",
        "description": "Dinomaly pretrained using healthy and diseased images. The doctor sees the test scan, anomaly map, and retrieved cases selected by this model.",
        "folder": ROOT / "brset_ai_human/dinomaly_hd",
        "test_names": "brset_normal_disease_test_name.pkl",
        "train_names": "brset_normal_disease_train_name.pkl",
    },
}

ARM_OPTIONS = [
    {
        "key": "human",
        "model_key": HUMAN_MODEL_KEY,
        "mode": "human",
        "label": MODE_LABELS["human"],
        "model_label": MODE_LABELS["human"],
        "mode_label": "Clinical baseline",
        "training": "Unaided scan review",
        "description": "Doctor sees only the test scan and decides which disease or diagnosis is present.",
    }
] + [
    {
        "key": f"{model_key}:combined",
        "model_key": model_key,
        "mode": "combined",
        "label": f"{MODE_LABELS['combined']} - {config['label']}",
        "model_label": config["label"],
        "mode_label": MODE_LABELS["combined"],
        "training": config["training"],
        "description": config["description"],
    }
    for model_key, config in DINOMALY_MODELS.items()
]

METADATA_CSV = ROOT / "brset_ai_human/brset_dataset_distribution.csv"
BRSET_DIR = ROOT / "brset_ai_human/BRSET"
RETRIEVAL_IMAGE_DIR = ROOT / "brset_ai_human/BRSET/fundus_photos"
RESPONSES_CSV = ROOT / "doctor_review_responses.csv"
RESPONSES_JSONL = ROOT / "doctor_review_responses.jsonl"
SESSION_LOG_CSV = ROOT / "doctor_review_session_log.csv"
SESSIONS_JSON = ROOT / "doctor_review_sessions.json"
CASE_STATUS_JSON = ROOT / "doctor_review_case_status.json"

app = Flask(__name__)
app.secret_key = os.environ.get("BRSET_REVIEW_SECRET_KEY", "brset-review-local-secret")


class FlipBasedOnBrightness:
    def __init__(self, patch_size: int = 16, decision_size: int = 256) -> None:
        self.patch_size = patch_size
        self.decision_size = decision_size

    def __call__(self, image: Image.Image) -> Image.Image:
        decision_image = image.resize((self.decision_size, self.decision_size), resampling_filter())
        gray = decision_image.convert("L")
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


def resampling_filter() -> Any:
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


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


def dinomaly_model_key(value: Any = None) -> str:
    key = str(value or DEFAULT_DINOMALY_MODEL)
    return key if key in DINOMALY_MODELS else DEFAULT_DINOMALY_MODEL


def arm_key(model_key: Any, mode: Any) -> str:
    mode_key = str(mode or "human")
    if mode_key not in MODE_LABELS:
        mode_key = "human"
    if mode_key == "human":
        return "human"
    return f"{dinomaly_model_key(model_key)}:{mode_key}"


def parse_arm_key(value: Any) -> tuple[str, str]:
    text = str(value or "")
    if text in {"", "human", HUMAN_MODEL_KEY}:
        return HUMAN_MODEL_KEY, "human"
    if ":" in text:
        model_part, mode_part = text.split(":", 1)
    else:
        model_part, mode_part = DEFAULT_DINOMALY_MODEL, text
    mode = mode_part if mode_part in MODE_LABELS else "human"
    if mode == "human":
        return HUMAN_MODEL_KEY, "human"
    model_key = dinomaly_model_key(model_part)
    return model_key, mode


def arm_label(model_key: Any, mode: Any) -> str:
    mode = str(mode or "human")
    if mode not in MODE_LABELS:
        mode = "human"
    if mode == "human":
        return MODE_LABELS["human"]
    model_key = dinomaly_model_key(model_key)
    return f"{DINOMALY_MODELS[model_key]['label']} / {MODE_LABELS[mode]}"


def arm_option_for(model_key: Any, mode: Any) -> dict[str, Any]:
    key = arm_key(model_key, mode)
    for option in ARM_OPTIONS:
        if option["key"] == key:
            return option
    return ARM_OPTIONS[0]


def model_config(model_key: Any = None) -> dict[str, Any]:
    return DINOMALY_MODELS[dinomaly_model_key(model_key)]


def model_file(model_key: Any, filename_key: str) -> Path:
    config = model_config(model_key)
    return config["folder"] / config[filename_key]


def model_dir(model_key: Any = None) -> Path:
    return model_config(model_key)["folder"]


def model_test_image_dir(model_key: Any = None) -> Path:
    return model_dir(model_key) / "original_img"


def model_anomaly_dir(model_key: Any = None) -> Path:
    return model_dir(model_key) / "anomaly_scan"


def scoped_mode(mode: str, model_key: Any = None) -> str:
    if mode in MODEL_SCOPED_MODES:
        return f"{mode}:{dinomaly_model_key(model_key)}"
    return mode


@lru_cache(maxsize=1)
def metadata_df() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)
    df["image_id"] = df["image_id"].map(normalize_image_id)
    return df


@lru_cache(maxsize=None)
def test_names(model_key: str = DEFAULT_DINOMALY_MODEL) -> list[str]:
    model_key = dinomaly_model_key(model_key)
    test_image_dir = model_test_image_dir(model_key)
    folder_names = sorted(
        normalize_image_id(path.name)
        for path in test_image_dir.iterdir()
        if path.is_file()
    ) if test_image_dir.exists() else []
    if folder_names:
        return folder_names
    return [normalize_image_id(name) for name in load_pickle(model_file(model_key, "test_names"))]


@lru_cache(maxsize=None)
def train_names(model_key: str = DEFAULT_DINOMALY_MODEL) -> list[str]:
    model_key = dinomaly_model_key(model_key)
    return [normalize_image_id(name) for name in load_pickle(model_file(model_key, "train_names"))]


@lru_cache(maxsize=None)
def similarities(model_key: str = DEFAULT_DINOMALY_MODEL) -> np.ndarray:
    return np.asarray(load_pickle(model_dir(model_key) / "similarity_data.pkl"))


@lru_cache(maxsize=None)
def indices(model_key: str = DEFAULT_DINOMALY_MODEL) -> np.ndarray:
    return np.asarray(load_pickle(model_dir(model_key) / "indices_data.pkl"))


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
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(
            [
                {key: str(row.get(key, "")) for key in fields}
                for row in rows
            ]
        )


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_response_rows() -> list[dict[str, Any]]:
    if RESPONSES_JSONL.exists():
        return read_jsonl_rows(RESPONSES_JSONL)
    return read_csv_rows(RESPONSES_CSV)


def write_response_rows(rows: list[dict[str, Any]]) -> None:
    write_jsonl_rows(RESPONSES_JSONL, rows)


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


def session_profile_from_form(
    form: Any,
    session_id: str,
    existing: dict[str, Any] | None = None,
    assigned_model_key: str | None = None,
    assigned_mode: str | None = None,
) -> dict[str, Any]:
    existing = existing or {}
    contact = form.get("contact", "")
    if assigned_model_key is not None and assigned_mode is not None:
        mode = assigned_mode if assigned_mode in MODE_LABELS else "human"
        model_key = HUMAN_MODEL_KEY if mode == "human" else dinomaly_model_key(assigned_model_key)
    else:
        model_key, mode = parse_arm_key(form.get("review_arm") or arm_key(form.get("dinomaly_model"), form.get("mode")))
    total_cases = len(test_names(model_key))
    start_index = existing.get("start_index")
    if start_index in {"", None}:
        start_index = random.randrange(total_cases) if total_cases else 0
    else:
        try:
            start_index = int(start_index)
        except (TypeError, ValueError):
            start_index = 0
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
        "mode": mode,
        "dinomaly_model": model_key if mode in MODEL_SCOPED_MODES else "",
        "review_arm": arm_key(model_key, mode),
        "review_arm_label": arm_label(model_key, mode),
        "start_index": start_index,
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


def matching_sessions_for_contact(sessions: dict[str, dict[str, Any]], contact: str) -> list[tuple[str, dict[str, Any]]]:
    contact_key = normalize_contact(contact)
    if not contact_key:
        return []
    matches = [
        (session_id, data)
        for session_id, data in sessions.items()
        if normalize_contact(data.get("contact_key") or data.get("contact")) == contact_key
    ]
    matches.sort(key=lambda item: (str(item[1].get("updated_at", "")), item[0]), reverse=True)
    return matches


def session_arm(data: dict[str, Any]) -> tuple[str, str]:
    return parse_arm_key(data.get("review_arm") or arm_key(data.get("dinomaly_model"), data.get("mode")))


def latest_matching_arm_session(
    sessions: dict[str, dict[str, Any]],
    contact: str,
    model_key: str,
    mode: str,
) -> tuple[str, dict[str, Any]] | None:
    requested_arm = arm_key(model_key, mode)
    for session_id, data in matching_sessions_for_contact(sessions, contact):
        existing_model, existing_mode = session_arm(data)
        if arm_key(existing_model, existing_mode) == requested_arm:
            return session_id, data
    return None


def session_progress(session_id: str, model_key: str | None = None, mode: str | None = None) -> dict[str, int] | int:
    if model_key is not None and mode is not None:
        return len(answered_ids(session_id, mode, model_key))
    return {
        option["key"]: len(answered_ids(session_id, option["mode"], option["model_key"]))
        for option in ARM_OPTIONS
    }


def contact_arm_progress(sessions: dict[str, dict[str, Any]], contact: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for option in ARM_OPTIONS:
        match = latest_matching_arm_session(sessions, contact, option["model_key"], option["mode"])
        session_id = match[0] if match else ""
        answered = len(answered_ids(session_id, option["mode"], option["model_key"])) if session_id else 0
        total = len(test_names(option["model_key"]))
        if answered >= total and total > 0:
            status = "completed"
            status_label = "Completed"
            action_label = "Open completed arm"
        elif answered > 0:
            status = "partial"
            status_label = "Partial"
            action_label = "Continue this arm"
        else:
            status = "not-started"
            status_label = "Not started"
            action_label = "Start this arm"
        cards.append(
            {
                **option,
                "session_id": session_id,
                "answered": answered,
                "total": total,
                "status": status,
                "status_label": status_label,
                "action_label": action_label,
            }
        )
    return cards


def choose_start_arm(sessions: dict[str, dict[str, Any]], contact: str) -> dict[str, Any]:
    cards = contact_arm_progress(sessions, contact)
    partial = [card for card in cards if card["status"] == "partial"]
    if partial:
        partial.sort(key=lambda card: str(card.get("session_id", "")), reverse=True)
        return partial[0]
    not_started = [card for card in cards if card["status"] == "not-started"]
    if not_started:
        return random.choice(not_started)
    return random.choice(cards or ARM_OPTIONS)


def active_session_id() -> str:
    return str(browser_session.get("review_session_id", ""))


def log_event(event: str, payload: dict[str, Any]) -> None:
    rows = read_csv_rows(SESSION_LOG_CSV)
    session_id = str(payload.get("session_id") or payload.get("reviewer_session_uid") or "")
    mode = payload.get("mode", "")
    model_key = payload.get("dinomaly_model", DEFAULT_DINOMALY_MODEL)
    answered_for_arm = (
        session_progress(session_id, dinomaly_model_key(model_key), mode)
        if session_id and mode in MODE_LABELS
        else payload.get("answered_count", "")
    )
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
            "review_arm",
            "review_arm_label",
            "start_index",
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
        "mode": mode,
        "dinomaly_model": payload.get("dinomaly_model", ""),
        "case_number": payload.get("case_number", ""),
        "image_id": payload.get("image_id", ""),
        "answered_human": answered_for_arm if mode == "human" else payload.get("answered_human", ""),
        "answered_dinomaly": payload.get("answered_dinomaly", ""),
        "answered_combined": answered_for_arm if mode == "combined" else payload.get("answered_combined", ""),
        "total_cases": payload.get("total_cases", len(test_names())),
        "details_json": json.dumps(details, sort_keys=True),
    }
    rows.append(row)
    write_csv_rows(SESSION_LOG_CSV, rows, SESSION_LOG_FIELDS)


def response_model_key(row: dict[str, Any]) -> str:
    return dinomaly_model_key(row.get("dinomaly_model")) if row.get("mode") in MODEL_SCOPED_MODES else ""


def response_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    session_id = row.get("reviewer_session_uid") or row.get("session_id") or ""
    mode = str(row.get("mode", ""))
    return str(session_id), mode, response_model_key(row), normalize_image_id(row.get("image_id", ""))


def answered_ids(session_id: str, mode: str, model_key: str = DEFAULT_DINOMALY_MODEL) -> set[str]:
    expected_model = dinomaly_model_key(model_key) if mode in MODEL_SCOPED_MODES else ""
    return {
        normalize_image_id(row["image_id"])
        for row in read_response_rows()
        if (row.get("reviewer_session_uid") or row.get("session_id")) == session_id
        and row.get("mode") == mode
        and response_model_key(row) == expected_model
    }


def saved_response(session_id: str, mode: str, image_id: str, model_key: str = DEFAULT_DINOMALY_MODEL) -> dict[str, str] | None:
    key = (session_id, mode, dinomaly_model_key(model_key) if mode in MODEL_SCOPED_MODES else "", normalize_image_id(image_id))
    for row in reversed(read_response_rows()):
        if response_key(row) == key:
            return row
    return None


def next_unanswered(answered: set[str], start_index: int = 0, model_key: str = DEFAULT_DINOMALY_MODEL) -> int:
    names = test_names(model_key)
    for offset in range(len(names)):
        index = (start_index + offset) % len(names)
        if names[index] not in answered:
            return index
    return min(start_index, len(names) - 1)


def retrieval_rows(test_index: int, top_k: int, model_key: str = DEFAULT_DINOMALY_MODEL) -> list[dict[str, Any]]:
    model_key = dinomaly_model_key(model_key)
    rows = []
    for rank, train_index in enumerate(indices(model_key)[test_index].tolist()[:top_k], start=1):
        image_id = train_names(model_key)[int(train_index)]
        row = metadata_lookup(image_id)
        rows.append(
            {
                "rank": rank,
                "image_id": image_id,
                "similarity": float(similarities(model_key)[test_index][rank - 1]),
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
    if isinstance(raw, dict):
        for disease, certainty in raw.items():
            if disease in defaults and certainty in CERTAINTY_LEVELS:
                defaults[disease] = certainty
        return defaults
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
    rows = read_response_rows()
    key = response_key(response)
    rows = [row for row in rows if response_key(row) != key]
    rows.append(response)
    write_response_rows(rows)


def delete_response(session_id: str, mode: str, image_id: str, model_key: str = DEFAULT_DINOMALY_MODEL) -> bool:
    rows = read_response_rows()
    key = (session_id, mode, dinomaly_model_key(model_key) if mode in MODEL_SCOPED_MODES else "", normalize_image_id(image_id))
    kept = [row for row in rows if response_key(row) != key]
    deleted = len(kept) != len(rows)
    if rows:
        write_response_rows(kept)
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


def send_transformed_query_image(path: Path | None, image_id: str | None = None) -> Response:
    if path is None:
        return missing_svg()
    image = Image.open(path).convert("RGB")
    image = query_image_transform(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    response = send_file(buffer, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


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
.review-nav { display: grid; gap: 14px; margin: 12px 0 18px; }
.model-group {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  padding: 10px;
}
.model-group.active {
  border-color: #9fd7cf;
  box-shadow: 0 0 0 2px rgba(20, 143, 119, 0.08);
}
.model-heading {
  display: grid;
  gap: 3px;
  padding: 2px 2px 9px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 9px;
}
.model-heading span { font-size: 14px; font-weight: 800; color: #0f172a; }
.model-heading small { color: var(--muted); font-size: 12px; line-height: 1.25; }
.model-group.active .model-heading span { color: #075e56; }
.mode-list { display: grid; gap: 8px; margin: 0; }
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
.locked-arm {
  border: 1px solid #9fd7cf;
  border-radius: 8px;
  background: var(--accent-soft);
  padding: 12px;
  margin: 12px 0 18px;
}
.locked-label {
  color: #0f766e;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}
.locked-title { color: #075e56; font-size: 15px; font-weight: 820; }
.locked-subtitle { color: #0f766e; font-size: 12px; margin-top: 3px; line-height: 1.35; }
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
.assessment-section {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
  padding: 10px;
  margin-top: 10px;
}
.assessment-section-title {
  font-size: 12px;
  font-weight: 820;
  color: #0f172a;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.assessment-section-note {
  color: var(--muted);
  font-size: 12px;
  margin: 3px 0 8px;
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
.arm-status-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 12px 0 18px; }
.arm-status-card {
  border: 2px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  padding: 12px;
  text-align: left;
  color: inherit;
  font: inherit;
  cursor: pointer;
  width: 100%;
}
.arm-status-card:hover { box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.14); }
.arm-status-card.completed { border-color: #22c55e; background: #f0fdf4; }
.arm-status-card.partial { border-color: #f59e0b; background: #fffbeb; }
.arm-status-card.not-started { border-color: #ef4444; background: #fef2f2; }
.arm-status-title { font-size: 14px; font-weight: 800; color: #17202a; }
.arm-status-meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
.arm-status-value { margin-top: 9px; font-size: 13px; font-weight: 800; }
.arm-status-card.completed .arm-status-value { color: #15803d; }
.arm-status-card.partial .arm-status-value { color: #b45309; }
.arm-status-card.not-started .arm-status-value { color: #b91c1c; }
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
.logs-wrap { max-width: 1500px; margin: 28px auto; padding: 0 20px 34px; }
.logs-table-wrap {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  overflow: auto;
  max-height: 72vh;
}
.logs-table {
  width: 100%;
  min-width: 1400px;
  border-collapse: collapse;
  font-size: 12px;
}
.logs-table th,
.logs-table td {
  border-bottom: 1px solid var(--line);
  border-right: 1px solid var(--line);
  padding: 8px 9px;
  vertical-align: top;
  text-align: left;
}
.logs-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #f8fafc;
  color: #334155;
  font-size: 11px;
  text-transform: uppercase;
}
.logs-table td:last-child,
.logs-table th:last-child { border-right: 0; }
.log-cell-text {
  max-width: 260px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 0;
  font: inherit;
}
.empty-state {
  border: 1px dashed var(--line);
  border-radius: 8px;
  padding: 18px;
  color: var(--muted);
  background: #ffffff;
}
.logs-filter-grid {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr) auto auto;
  gap: 12px;
  align-items: end;
}
.logs-filter-grid .field { margin-bottom: 0; }
.logs-filter-actions { display: flex; gap: 10px; flex-wrap: wrap; }
@media (max-width: 1380px) {
  .workspace { grid-template-columns: 1fr; }
}
@media (max-width: 1100px) {
  .shell { display: flex; flex-direction: column; }
  .main { order: 1; }
  .sidebar {
    order: 2;
    position: relative;
    height: auto;
    border-right: 0;
    border-top: 1px solid var(--line);
  }
  .case-header { align-items: flex-start; flex-wrap: wrap; }
  .timer-box { text-align: left; }
  .workspace, .evidence-grid, .profile-grid, .arm-status-grid, .zoom-compare, .logs-filter-grid { grid-template-columns: 1fr; }
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
      <div class="field span-2"><label>Session notes</label><textarea name="session_notes"></textarea></div>
    </div>
    <button class="btn" type="submit">Continue</button>
  </form>
</div>
"""


ARM_SELECTION_BODY = """
<div class="profile-wrap">
  <div class="topbar">
    <div>
      <h1 class="title">Select Review Arm</h1>
      <div class="subtitle">Choose an arm after checking previous progress for this email.</div>
    </div>
  </div>
  <form class="profile-card" method="post" action="{{ url_for('start') }}">
    <div class="panel-title">Reviewer</div>
    <div class="metrics" style="margin-bottom:16px;">
      <div class="metric"><div class="metric-label">Doctor</div><div class="metric-value">{{ submitted.doctor_name }}</div></div>
      <div class="metric"><div class="metric-label">Hospital</div><div class="metric-value">{{ submitted.hospital_name }}</div></div>
      <div class="metric"><div class="metric-label">Email</div><div class="metric-value">{{ submitted.contact }}</div></div>
    </div>
    <div class="panel-title">Review arm status for this email</div>
    <div class="note">
      Suggested arm: <strong>{{ suggested_arm.label }}</strong>.
      <button class="btn secondary" style="margin-top:10px;" name="review_arm" value="{{ suggested_arm.key }}" type="submit">Use suggested arm</button>
    </div>
    <div class="arm-status-grid">
      {% for arm in arm_progress %}
      <button class="arm-status-card {{ arm.status }}" name="review_arm" value="{{ arm.key }}" type="submit">
        <div class="arm-status-title">{{ arm.label }}</div>
        <div class="arm-status-meta">{{ arm.training }}</div>
        <div class="help-text">{{ arm.description }}</div>
        <div class="arm-status-value">{{ arm.status_label }} - {{ arm.answered }} / {{ arm.total }}</div>
        <div class="help-text">{{ arm.action_label }}</div>
      </button>
      {% endfor %}
    </div>
    <div class="note">The selected arm will be locked for this review session and logged with the saved responses.</div>

    {% for key, value in submitted.items() %}
    {% if key != "review_arm" %}
    <input type="hidden" name="{{ key }}" value="{{ value }}">
    {% endif %}
    {% endfor %}
    <input type="hidden" name="session_action" value="select_arm">
  </form>
</div>
"""


REVIEW_BODY = """
<div class="shell">
  <aside class="sidebar">
    <div class="brand">BRSET Review</div>
    <div class="brand-sub">{{ doctor.doctor_name }} | {{ doctor.designation }}</div>
    <div class="locked-arm">
      <div class="locked-label">Locked review arm</div>
      <div class="locked-title">{{ model_label }}</div>
      <div class="locked-subtitle">{{ mode_label }}<br>{{ model_training }}</div>
    </div>
    <div class="panel-title">Cases</div>
    <div class="case-board">
      {% for case in case_cells %}
      <a class="case-cell {{ case.status }} {% if case.index == index %}active{% endif %}"
         title="Case {{ case.index + 1 }} - {{ case.status_label }}"
         href="{{ url_for('review', mode=mode, model=model_key, index=case.index) }}">{{ case.index + 1 }}</a>
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
        <div class="case-meta">{{ model_label }} | {{ mode_label }} | {{ doctor.doctor_name }} | Age {{ patient_demo.age }} | {{ patient_demo.gender }}</div>
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
              <a class="zoom-link" target="_blank" href="{{ url_for('zoom_compare', image_id=image_id, model=model_key) }}">Zoom</a>
            </div>
            <div class="panel-description">The AI model highlights regions it thinks are anomalous. {{ model_description }}</div>
            <img class="scan-img" src="{{ url_for('anomaly_image', image_id=image_id, model=model_key) }}" alt="Anomaly map">
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
          <input type="hidden" name="dinomaly_model" value="{{ model_key }}">
          <input type="hidden" name="index" value="{{ index }}">
          <input type="hidden" name="image_id" value="{{ image_id }}">
          <input type="hidden" name="top_k" value="{{ top_k }}">
          <input type="hidden" name="min_votes" value="{{ min_votes }}">
          <label>Doctor diagnosis</label>
          <div class="help-text">If there is more than one disease, diagnosis, sign, or finding, select all that apply.</div>
          {% for group in assessment_groups %}
          <div class="assessment-section">
            <div class="assessment-section-title">{{ group.title }}</div>
            <div class="assessment-section-note">{{ group.description }}</div>
            <div class="checkbox-grid">
              {% for disease in group.options %}
              <div class="checkbox-item">
                <label class="checkbox-main">
                  <input type="checkbox" name="diseases" value="{{ disease }}" {% if disease in previous_diseases %}checked{% endif %}>
                  <span>{{ disease_labels[disease] }}</span>
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
          </div>
          {% endfor %}
          <input name="review_time_seconds" type="hidden" value="0">
          <div class="field">
            <label>Session notes</label>
            <div class="help-text">If selecting Other Disease, enter the disease details here.</div>
            <textarea name="comments">{{ previous_comments }}</textarea>
          </div>
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


LOGS_BODY = """
<div class="logs-wrap">
  <div class="topbar">
    <div>
      <h1 class="title">Doctor Response Logs</h1>
      <div class="subtitle">Saved review responses. This page is only available by opening the /logs URL directly.</div>
    </div>
  </div>

  <form class="profile-card" style="margin-bottom:16px;" method="get" action="{{ url_for('logs') }}">
    <div class="panel-title">Filters</div>
    <div class="logs-filter-grid">
      <div class="field">
        <label>Doctor</label>
        <select name="doctor">
          <option value="all" {% if selected_doctor == "all" %}selected{% endif %}>All doctors</option>
          {% for doctor in doctor_options %}
          <option value="{{ doctor.key }}" {% if selected_doctor == doctor.key %}selected{% endif %}>{{ doctor.label }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="field">
        <label>Component</label>
        <select name="arm">
          <option value="all" {% if selected_arm == "all" %}selected{% endif %}>All components</option>
          {% for arm in arm_options %}
          <option value="{{ arm.key }}" {% if selected_arm == arm.key %}selected{% endif %}>{{ arm.label }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="logs-filter-actions">
        <button class="btn" type="submit">Apply</button>
        <a class="btn secondary" href="{{ url_for('logs') }}">Clear</a>
      </div>
      <div class="logs-filter-actions">
        <a class="btn secondary" href="{{ download_url }}">Download CSV</a>
      </div>
    </div>
  </form>

  <div class="profile-card" style="margin-bottom:16px;">
    <div class="panel-title">Summary</div>
    <div class="metrics">
      <div class="metric"><div class="metric-label">Filtered responses</div><div class="metric-value">{{ response_count }}</div></div>
      <div class="metric"><div class="metric-label">Total responses</div><div class="metric-value">{{ total_response_count }}</div></div>
      <div class="metric"><div class="metric-label">Reviewer sessions</div><div class="metric-value">{{ session_count }}</div></div>
      <div class="metric"><div class="metric-label">Session log rows</div><div class="metric-value">{{ session_log_count }}</div></div>
      <div class="metric"><div class="metric-label">Case status entries</div><div class="metric-value">{{ case_status_count }}</div></div>
    </div>
  </div>

  {% if arm_counts %}
  <div class="profile-card" style="margin-bottom:16px;">
    <div class="panel-title">Responses by arm</div>
    <div class="metrics">
      {% for item in arm_counts %}
      <div class="metric">
        <div class="metric-label">{{ item.label }}</div>
        <div class="metric-value">{{ item.count }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if rows %}
  <div class="logs-table-wrap">
    <table class="logs-table">
      <thead>
        <tr>
          {% for column in columns %}
          <th>{{ column }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in rows %}
        <tr>
          {% for column in columns %}
          <td><pre class="log-cell-text">{{ row[column] }}</pre></td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="empty-state">No doctor responses have been saved yet.</div>
  {% endif %}
</div>
"""


def render_page(body: str, **context: Any) -> str:
    return render_template_string(BASE_HTML, css=CSS, body=render_template_string(body, **context))


def pill_html(values: list[str], strong: bool = False, empty: str = "None") -> str:
    if not values:
        return f"<span class='pill'>{empty}</span>"
    class_name = "pill strong" if strong else "pill"
    return "".join(f"<span class='{class_name}'>{disease_display_label(value)}</span>" for value in values)


def retrieval_cards_html(retrieved: list[dict[str, Any]], show_labels: bool, model_key: str = DEFAULT_DINOMALY_MODEL) -> str:
    model_key = dinomaly_model_key(model_key)
    cards = ["<div class='retrieval-list'>"]
    for item in retrieved:
        cards.append(
            "<div class='retrieval-card'>"
            f"<div class='retrieval-head'><span>Retrieved scan {item['rank']}</span>"
            f"<span>Similarity {item['similarity']:.3f}</span></div>"
            f"<img class='retrieval-img' src='{url_for('retrieval_image', image_id=item['image_id'], model=model_key)}' alt='Retrieved scan'>"
            f"<div style='margin-top:8px;'><a class='zoom-link' target='_blank' "
            f"href='{url_for('zoom_image', kind='retrieval', image_id=item['image_id'], model=model_key)}'>Zoom</a></div>"
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
            f"<div><strong>{disease_display_label(item['disease'])}</strong>: {item['count']} / {DINOMALY_RETRIEVAL_COUNT} retrieved scans, "
            f"mean similarity {item['mean_similarity']:.4f}</div>"
        )
    rows.append("</div>")
    return "".join(rows)


def display_log_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def response_row_arm_key(row: dict[str, Any]) -> str:
    return arm_key(row.get("dinomaly_model"), row.get("mode"))


def response_row_doctor_key(row: dict[str, Any]) -> str:
    contact_key = normalize_contact(row.get("contact_key") or row.get("contact"))
    if contact_key:
        return contact_key
    return normalize_contact(row.get("doctor_name"))


def response_row_doctor_label(row: dict[str, Any]) -> str:
    name = str(row.get("doctor_name") or "Unknown doctor").strip() or "Unknown doctor"
    contact = str(row.get("contact") or row.get("contact_key") or "").strip()
    hospital = str(row.get("hospital_name") or "").strip()
    details = [value for value in [contact, hospital] if value]
    return f"{name} ({', '.join(details)})" if details else name


def doctor_log_options(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    options: dict[str, str] = {}
    for row in raw_rows:
        key = response_row_doctor_key(row)
        if key and key not in options:
            options[key] = response_row_doctor_label(row)
    return [
        {"key": key, "label": label}
        for key, label in sorted(options.items(), key=lambda item: item[1].lower())
    ]


def arm_log_options(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    present = {response_row_arm_key(row) for row in raw_rows}
    return [
        {"key": option["key"], "label": option["label"]}
        for option in ARM_OPTIONS
        if not present or option["key"] in present
    ]


def filter_response_rows(
    raw_rows: list[dict[str, Any]],
    doctor_key: str = "all",
    arm_filter: str = "all",
) -> list[dict[str, Any]]:
    filtered = raw_rows
    if doctor_key != "all":
        filtered = [row for row in filtered if response_row_doctor_key(row) == doctor_key]
    if arm_filter != "all":
        filtered = [row for row in filtered if response_row_arm_key(row) == arm_filter]
    return filtered


def response_log_rows(raw_rows: list[dict[str, Any]] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    raw_rows = read_response_rows() if raw_rows is None else raw_rows
    columns = list(RESPONSE_FIELDS)
    for row in raw_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    rows = [
        {column: display_log_value(row.get(column, "")) for column in columns}
        for row in raw_rows
    ]
    rows.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
    return columns, rows


def case_status_entry_count() -> int:
    statuses = load_case_statuses()
    total = 0
    for session_data in statuses.values():
        if isinstance(session_data, dict):
            for mode_data in session_data.values():
                if isinstance(mode_data, dict):
                    total += len(mode_data)
    return total


def response_arm_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {option["key"]: 0 for option in ARM_OPTIONS}
    for row in rows:
        key = response_row_arm_key(row)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"label": option["label"], "count": counts.get(option["key"], 0)}
        for option in ARM_OPTIONS
    ]


def csv_response(rows: list[dict[str, str]], columns: list[str], filename: str) -> Response:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def case_cells_for(session_id: str, mode: str, current_index: int, model_key: str = DEFAULT_DINOMALY_MODEL) -> list[dict[str, Any]]:
    statuses = load_case_statuses()
    status_mode = scoped_mode(mode, model_key)
    answered = answered_ids(session_id, mode, model_key)
    labels = {
        "new": "not opened",
        "opened": "opened, not saved",
        "saved": "saved",
        "recheck": "saved, marked for recheck",
    }
    cells = []
    for idx, image_id in enumerate(test_names(model_key)):
        status = case_status_for(session_id, status_mode, image_id, answered, statuses)
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
    )


@app.get("/logs")
def logs() -> str:
    raw_rows = read_response_rows()
    selected_doctor = request.args.get("doctor", "all")
    selected_arm = request.args.get("arm", "all")
    if selected_doctor not in {"all", *{option["key"] for option in doctor_log_options(raw_rows)}}:
        selected_doctor = "all"
    if selected_arm not in {"all", *{option["key"] for option in ARM_OPTIONS}}:
        selected_arm = "all"
    doctor_filtered_rows = filter_response_rows(raw_rows, selected_doctor, "all")
    filtered_raw_rows = filter_response_rows(raw_rows, selected_doctor, selected_arm)
    columns, rows = response_log_rows(filtered_raw_rows)
    return render_page(
        LOGS_BODY,
        columns=columns,
        rows=rows,
        response_count=len(rows),
        total_response_count=len(raw_rows),
        session_count=len(load_sessions()),
        session_log_count=len(read_csv_rows(SESSION_LOG_CSV)),
        case_status_count=case_status_entry_count(),
        arm_counts=response_arm_counts(doctor_filtered_rows),
        doctor_options=doctor_log_options(raw_rows),
        arm_options=arm_log_options(doctor_filtered_rows),
        selected_doctor=selected_doctor,
        selected_arm=selected_arm,
        download_url=url_for("download_logs", doctor=selected_doctor, arm=selected_arm),
    )


@app.get("/logs/download.csv")
def download_logs() -> Response:
    raw_rows = read_response_rows()
    selected_doctor = request.args.get("doctor", "all")
    selected_arm = request.args.get("arm", "all")
    filtered_raw_rows = filter_response_rows(raw_rows, selected_doctor, selected_arm)
    columns, rows = response_log_rows(filtered_raw_rows)
    filename_parts = ["doctor_responses"]
    if selected_doctor != "all":
        filename_parts.append(selected_doctor.replace("@", "_at_").replace(".", "_"))
    if selected_arm != "all":
        filename_parts.append(selected_arm.replace(":", "_"))
    filename = "_".join(filename_parts) + ".csv"
    return csv_response(rows, columns, filename)


@app.post("/start")
def start() -> str | Response:
    form = request.form
    sessions = load_sessions()
    action = form.get("session_action", "check")
    contact = form.get("contact", "")
    submitted_arm = form.get("review_arm", "")

    if action == "check" or not submitted_arm:
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
                "review_arm",
                "session_notes",
            ]
        }
        arm_progress = contact_arm_progress(sessions, contact)
        suggested_arm = choose_start_arm(sessions, contact)
        log_event(
            "arm_selection_prompt",
            {
                **submitted,
                "answered_count": 0,
                "total_cases": len(test_names()),
            },
        )
        return render_page(
            ARM_SELECTION_BODY,
            submitted=submitted,
            arm_progress=arm_progress,
            suggested_arm=suggested_arm,
        )

    model_key, mode = parse_arm_key(submitted_arm)
    selected_arm_match = latest_matching_arm_session(sessions, contact, model_key, mode)

    if selected_arm_match is not None:
        session_id = selected_arm_match[0]
        event = "resume_session"
    else:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        while session_id in sessions:
            session_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        event = "start_session"

    sessions[session_id] = session_profile_from_form(
        form,
        session_id,
        sessions.get(session_id),
        assigned_model_key=model_key,
        assigned_mode=mode,
    )
    save_sessions(sessions)
    browser_session["review_session_id"] = session_id
    answered = answered_ids(session_id, mode, model_key)
    log_event(event, {**sessions[session_id], "mode": mode, "dinomaly_model": model_key if mode in MODEL_SCOPED_MODES else "", "answered_count": len(answered), "total_cases": len(test_names(model_key))})
    start_index = int(sessions[session_id].get("start_index", 0) or 0)
    return redirect(url_for("review", index=next_unanswered(answered, start_index, model_key=model_key)))


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

    session_profile = sessions[session_id]
    session_model_key, session_mode = session_arm(session_profile)
    model_key = session_model_key
    mode = session_mode
    selected_arm = arm_option_for(model_key, mode)
    status_mode = scoped_mode(mode, model_key)
    names = test_names(model_key)
    total_cases = len(names)
    top_k = DINOMALY_RETRIEVAL_COUNT
    min_votes = DINOMALY_PREDICTION_MIN_VOTES
    answered = answered_ids(session_id, mode, model_key)
    index_arg = request.args.get("index")
    index = next_unanswered(answered, model_key=model_key) if index_arg is None else int(index_arg)
    index = max(0, min(total_cases - 1, index))
    image_id = names[index]
    row = metadata_lookup(image_id)
    if image_id not in answered:
        mark_case_status(session_id, status_mode, image_id, "opened", needs_recheck=False)
    retrieved = retrieval_rows(index, top_k, model_key) if mode in METHOD_MODES else []
    prediction_min_votes = DINOMALY_PREDICTION_MIN_VOTES
    prediction = method_prediction(retrieved, prediction_min_votes)
    if mode in METHOD_MODES:
        prediction["evidence"] = [
            item for item in prediction["evidence"] if item["count"] >= prediction_min_votes
        ]
    previous = saved_response(session_id, mode, image_id, model_key) or {}
    previous_diseases = [
        disease
        for disease in parse_assessment_list(previous.get("doctor_selected_diseases", ""))
        if disease in ASSESSMENT_OPTIONS
    ]
    if truthy(previous.get("doctor_selected_no_disease")):
        previous_diseases.insert(0, NO_DISEASE_OPTION)
    needs_recheck = truthy(previous.get("needs_recheck")) or needs_recheck_for(session_id, status_mode, image_id)

    previous_url = url_for("review", mode=mode, model=model_key, top_k=top_k, min_votes=min_votes, index=max(0, index - 1))
    next_url = url_for("review", mode=mode, model=model_key, top_k=top_k, min_votes=min_votes, index=min(total_cases - 1, index + 1))
    mode_short = {"human": "Human", "combined": "Combined"}[mode]

    return render_page(
        REVIEW_BODY,
        doctor=sessions[session_id],
        mode=mode,
        model_key=model_key,
        model_label=selected_arm["model_label"],
        model_training=selected_arm["training"],
        model_description=selected_arm["description"],
        mode_label=selected_arm["mode_label"],
        mode_short=mode_short,
        patient_demo=patient_demographics(row),
        index=index,
        image_id=image_id,
        total_cases=total_cases,
        answered_count=len(answered),
        already_saved=image_id in answered,
        case_cells=case_cells_for(session_id, mode, index, model_key),
        top_k=top_k,
        min_votes=min_votes,
        retrieved=retrieved,
        retrieval_cards=retrieval_cards_html(retrieved, show_labels=mode == "combined", model_key=model_key),
        prediction_pills=pill_html(prediction["predicted"], strong=True, empty="No prediction"),
        evidence_html=evidence_html(prediction["evidence"]),
        disease_columns=ASSESSMENT_OPTIONS,
        assessment_groups=ASSESSMENT_GROUPS,
        disease_labels={disease: disease_display_label(disease) for disease in ASSESSMENT_OPTIONS},
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
    sessions = load_sessions()
    if session_id not in sessions:
        return redirect(url_for("profile"))
    model_key, mode = session_arm(sessions[session_id])
    status_mode = scoped_mode(mode, model_key)
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
        deleted = delete_response(session_id, mode, image_id, model_key)
        mark_case_status(session_id, status_mode, image_id, "opened", needs_recheck=False)
        log_event(
            "clear_response",
            {
                **doctor,
                "mode": mode,
                "dinomaly_model": model_key if mode in MODEL_SCOPED_MODES else "",
                "image_id": image_id,
                "case_number": index + 1,
                "deleted": deleted,
                "answered_count": len(answered_ids(session_id, mode, model_key)),
                "total_cases": len(test_names(model_key)),
            },
        )
        return redirect(
            url_for(
                "review",
                mode=mode,
                model=model_key,
                top_k=top_k,
                min_votes=min_votes,
                index=index,
            )
        )

    retrieved = retrieval_rows(index, top_k, model_key) if mode in METHOD_MODES else []
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
        "dinomaly_model": model_key if mode in MODEL_SCOPED_MODES else "",
        "case_number": index + 1,
        "image_id": image_id,
        "patient_id": "" if row is None else int(row["patient_id"]),
        "doctor_selected_no_disease": NO_DISEASE_OPTION in selected,
        "doctor_selected_diseases": selected_diseases,
        "doctor_selected_count": len(selected),
        "needs_recheck": needs_recheck,
        "selected_disease_certainty_json": selected_disease_certainties,
        "review_time_seconds": review_time_seconds,
        "comments": form.get("comments", ""),
        "dinomaly_predicted_diseases": prediction["predicted"],
        "dinomaly_vote_counts_json": vote_counts,
        "dinomaly_evidence_json": dinomaly_evidence,
        "retrieved_image_ids": [item["image_id"] for item in retrieved],
        "retrieved_similarities": [round(float(item["similarity"]), 6) for item in retrieved],
        "retrieved_diseases_json": {item["image_id"]: item["diseases"] for item in retrieved},
        "true_diseases_hidden_from_reviewer": true_diseases,
        "true_disease_count": "" if row is None else int(row["disease_count"]),
        "true_disease_category": "" if row is None else str(row["disease_category"]),
    }
    upsert_response(response)
    mark_case_status(session_id, status_mode, image_id, "saved", needs_recheck=needs_recheck)
    updated_answered = answered_ids(session_id, mode, model_key)
    total_cases = len(test_names(model_key))
    log_event("save_response", {**doctor, "mode": mode, "dinomaly_model": model_key if mode in MODEL_SCOPED_MODES else "", "case_number": index + 1, "image_id": image_id, "answered_count": len(updated_answered), "total_cases": total_cases})
    if len(updated_answered) == total_cases:
        log_event("complete_session", {**doctor, "mode": mode, "dinomaly_model": model_key if mode in MODEL_SCOPED_MODES else "", "answered_count": len(updated_answered), "total_cases": total_cases})
    return redirect(url_for("review", mode=mode, model=model_key, top_k=top_k, min_votes=min_votes, index=next_unanswered(updated_answered, index + 1, model_key=model_key)))


@app.get("/scan/<image_id>")
def scan_image(image_id: str) -> Response:
    return send_transformed_query_image(
        resolve_image_path(
            image_id,
            [RETRIEVAL_IMAGE_DIR],
            recursive_dirs=[BRSET_DIR],
        ),
        image_id=image_id,
    )


@app.get("/anomaly/<image_id>")
def anomaly_image(image_id: str) -> Response:
    model_key = dinomaly_model_key(request.args.get("model"))
    return send_resolved_image(
        resolve_image_path(image_id, [model_anomaly_dir(model_key)], recursive_dirs=[model_dir(model_key)])
    )


@app.get("/retrieval/<image_id>")
def retrieval_image(image_id: str) -> Response:
    model_key = dinomaly_model_key(request.args.get("model"))
    return send_resolved_image(
        resolve_image_path(
            image_id,
            [RETRIEVAL_IMAGE_DIR, model_test_image_dir(model_key)],
            recursive_dirs=[BRSET_DIR, model_dir(model_key)],
        )
    )


@app.get("/zoom/<kind>/<image_id>")
def zoom_image(kind: str, image_id: str) -> str | Response:
    model_key = dinomaly_model_key(request.args.get("model"))
    if kind == "scan":
        title = "Scan zoom"
        image_url = url_for("scan_image", image_id=image_id)
    elif kind == "anomaly":
        title = "Anomaly map zoom"
        image_url = url_for("anomaly_image", image_id=image_id, model=model_key)
    elif kind == "retrieval":
        title = "Retrieved scan zoom"
        image_url = url_for("retrieval_image", image_id=image_id, model=model_key)
    else:
        return Response("Unknown image type", status=404)
    return render_page(ZOOM_BODY, title=title, image_url=image_url)


@app.get("/zoom/compare/<image_id>")
def zoom_compare(image_id: str) -> str:
    model_key = dinomaly_model_key(request.args.get("model"))
    return render_page(
        COMPARE_ZOOM_BODY,
        scan_url=url_for("scan_image", image_id=image_id),
        anomaly_url=url_for("anomaly_image", image_id=image_id, model=model_key),
    )


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status=204)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8502"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
