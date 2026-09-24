from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from bson.errors import InvalidId
import jwt
import datetime
import os
import json
import uuid
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
from dotenv import load_dotenv

from utils.career_scoring import marks_to_pslots_sorted, blend_career_probabilities
from utils.jobs_guidance_ai import call_jobs_openai
from utils.cv_maker_ai import call_cv_openai

load_dotenv()

app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
if "<db_password>" in MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["growsmart"]
users = db["users"]
career_questions_col = db["career_questions"]
career_scope_col = db["career_scope"]

SECRET_KEY = os.getenv("SECRET_KEY", "anas_secret_123")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Resend (forgot-password emails). Local test: no custom domain needed — use beth.t@example.com
RESEND_API_KEY = os.getenv("RESEND_API_KEY") or os.getenv("resend_api_key") or ""
# Without a verified domain, From MUST be Resend's onboarding address (not Gmail).
RESEND_FROM_EMAIL = os.getenv(
    "RESEND_FROM_EMAIL",
    "GrowSmart <onboarding@resend.dev>",
)
# Your Gmail — used as Reply-To so users can reply to you
RESEND_REPLY_TO = os.getenv("RESEND_REPLY_TO", "anas.dev200@gmail.com")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Kaggle career_data.xlsx skill columns are scored ~0–20 (not 1–3).
KAGGLE_SKILL_COLUMNS = [
    "Linguistic",
    "Musical",
    "Bodily",
    "Logical - Mathematical",
    "Spatial-Visualization",
    "Interpersonal",
    "Intrapersonal",
    "Naturalist",
]


def _skill_to_kaggle_scale(value):
    """Map UI / legacy values onto the Kaggle dataset 0–20 skill scale."""
    if isinstance(value, (int, float)):
        v = float(value)
        if 1 <= v <= 5:
            return int(round(4 * v))  # Likert 1–5 → 4,8,12,16,20
        if 0 <= v <= 20:
            return int(round(v))
        if 1 <= v <= 3:
            return {1: 7, 2: 12, 3: 17}[int(v)]
        return None
    if isinstance(value, str):
        return {"LOW": 7, "MEDIUM": 12, "HIGH": 17}.get(value.strip().upper())
    return None


def _kaggle_to_ternary(value):
    """Compress 0–20 skill into 1/2/3 for stream/skill blend heuristics."""
    v = int(value or 0)
    if v <= 8:
        return 1
    if v <= 14:
        return 2
    return 3


def _current_user():
    """Resolve logged-in user from Authorization: Bearer <jwt>."""
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        uid = payload.get("user_id")
        if not uid:
            return None
        return users.find_one({"_id": ObjectId(uid)})
    except (jwt.PyJWTError, InvalidId, PyMongoError, TypeError):
        return None


def _public_guidance_entry(entry):
    if not isinstance(entry, dict):
        return entry
    out = dict(entry)
    out.pop("_id", None)
    return out


def _load_json(path, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback


CAREER_QUESTIONS_FALLBACK = _load_json(
    os.path.join(DATASET_DIR, "career_questions.json"),
    [],
)
PAKISTAN_SCOPE_DATA = _load_json(
    os.path.join(DATASET_DIR, "pakistan_career_scope.json"),
    {"meta": {}, "categories": {}, "career_map": {}},
)


def seed_career_guidance_data():
    """Load career questions + Pakistan scope into MongoDB from dataset JSON."""
    try:
        if CAREER_QUESTIONS_FALLBACK:
            career_questions_col.delete_many({})
            docs = []
            for i, q in enumerate(CAREER_QUESTIONS_FALLBACK):
                docs.append({**q, "order": i, "active": True})
            if docs:
                career_questions_col.insert_many(docs)

        career_scope_col.delete_many({})
        career_scope_col.insert_one({
            "key": "pakistan_v1",
            "data": PAKISTAN_SCOPE_DATA,
            "updated": (PAKISTAN_SCOPE_DATA.get("meta") or {}).get("updated"),
        })
        print("Career guidance dataset seeded into MongoDB.")
    except PyMongoError as e:
        print(f"Warning: could not seed career guidance collections: {e}")


try:
    seed_career_guidance_data()
except Exception as e:
    print(f"Warning: career guidance seed skipped: {e}")

candidate_dirs = [
    os.path.join(BASE_DIR, "model"),
    os.path.join(BASE_DIR, "..", "model"),
]

model_path = None
encoder_path = None
model = None
encoder = None

for d in candidate_dirs:
    m = os.path.join(d, "model.pkl")
    e = os.path.join(d, "encoder.pkl")
    if os.path.exists(m) and os.path.exists(e):
        model_path = m
        encoder_path = e
        break

if model_path and encoder_path:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    MODEL_FEATURE_ORDER = list(getattr(model, "feature_names_in_", []))
else:
    MODEL_FEATURE_ORDER = []
    print(
        "Warning: model.pkl / encoder.pkl not found. "
        "Auth endpoints will work, but /predict-career will return an error "
        "until model files are added."
    )


@app.route("/")
def home():
    return "GrowSmart API Running"


@app.route("/career-questions", methods=["GET"])
def get_career_questions():
    """Career-aptitude questions from MongoDB (falls back to dataset JSON)."""
    try:
        docs = list(
            career_questions_col.find({"active": {"$ne": False}}, {"_id": 0})
            .sort("order", 1)
        )
        if docs:
            return jsonify({"success": True, "source": "database", "questions": docs})
    except PyMongoError:
        pass

    if CAREER_QUESTIONS_FALLBACK:
        return jsonify({
            "success": True,
            "source": "dataset",
            "questions": CAREER_QUESTIONS_FALLBACK,
        })
    return jsonify({"success": False, "message": "No career questions found", "questions": []}), 404


def _resolve_scope_for_career(career_name: str):
    payload = PAKISTAN_SCOPE_DATA
    try:
        doc = career_scope_col.find_one({"key": "pakistan_v1"})
        if doc and isinstance(doc.get("data"), dict):
            payload = doc["data"]
    except PyMongoError:
        pass

    meta = payload.get("meta") or {}
    categories = payload.get("categories") or {}
    career_map = payload.get("career_map") or {}
    name = (career_name or "").strip()
    lower = name.lower()

    cat_id = career_map.get(name)
    if not cat_id:
        for key, value in career_map.items():
            if key.lower() in lower or lower in key.lower():
                cat_id = value
                break
    if not cat_id:
        # keyword fallbacks
        if any(k in lower for k in ("computer", "program", "database", "software", "analyst")):
            cat_id = "it_software"
        elif any(k in lower for k in ("medical", "physician", "nurse", "pharma", "therap")):
            cat_id = "healthcare"
        elif any(k in lower for k in ("teacher", "professor", "librarian")):
            cat_id = "education"
        elif any(k in lower for k in ("bank", "account", "financ", "audit", "stock", "actuary")):
            cat_id = "finance"
        elif any(k in lower for k in ("market", "manager", "sales", "business", "consult")):
            cat_id = "business"
        elif any(k in lower for k in ("design", "artist", "broadcast", "journal", "editor", "fashion")):
            cat_id = "media_creative"
        elif any(k in lower for k in ("engineer", "pilot", "physic")):
            cat_id = "engineering"
        elif any(k in lower for k in ("police", "milit", "lawyer", "politic", "crime")):
            cat_id = "law_gov"
        else:
            cat_id = "business"

    category = categories.get(cat_id) or {
        "label": "General careers",
        "overall_scope": 60,
        "demand_note": "Mixed demand across Pakistan job portals.",
        "platforms": {"linkedin": 25, "rozee": 30, "mustakbil": 25, "other": 20},
    }

    return {
        "career": name,
        "category_id": cat_id,
        "category_label": category.get("label"),
        "overall_scope": category.get("overall_scope"),
        "demand_note": category.get("demand_note"),
        "platforms": category.get("platforms") or {},
        "portals": meta.get("portals") or [],
        "meta_note": meta.get("note"),
        "updated": meta.get("updated"),
        "source": "database",
    }


@app.route("/career-scope", methods=["GET", "POST"])
def career_scope():
    if request.method == "POST":
        body = request.json or {}
        career = str(body.get("career", "")).strip()
    else:
        career = str(request.args.get("career", "")).strip()

    if not career:
        return jsonify({"message": "career is required"}), 400

    return jsonify({"success": True, **_resolve_scope_for_career(career)})


# Auth routes live in routes/auth.py (register / login / forgot + reset via Resend)
from routes.auth import create_auth_blueprint
app.register_blueprint(
    create_auth_blueprint(
        users,
        SECRET_KEY,
        resend_api_key=RESEND_API_KEY,
        resend_from_email=RESEND_FROM_EMAIL,
        resend_reply_to=RESEND_REPLY_TO,
        frontend_url=FRONTEND_URL,
    )
)


def _save_guidance_for_user(user, entry):
    """Push one guidance entry into users.guidance (max 100)."""
    try:
        users.update_one(
            {"_id": user["_id"]},
            {
                "$push": {
                    "guidance": {
                        "$each": [entry],
                        "$position": 0,
                        "$slice": 100,
                    }
                }
            },
        )
        return True
    except PyMongoError:
        return False


def _build_career_guidance_entry(data, predicted_career, top_results, used_sorted_slots):
    return {
        "id": str(uuid.uuid4()),
        "type": "career",
        "createdAt": datetime.datetime.utcnow().isoformat() + "Z",
        "title": str(predicted_career),
        "career": str(predicted_career),
        "topCareers": top_results,
        "fullData": {
            "matric_marks": data.get("matric_marks") or data.get("matricMarks") or {},
            "intermediate_marks": data.get("intermediate_marks") or data.get("intermediateMarks") or {},
            "matric_stream": data.get("matric_stream") or data.get("matricStream"),
            "intermediate_stream": data.get("intermediate_stream") or data.get("intermediateStream"),
            "Linguistic": data.get("Linguistic"),
            "Musical": data.get("Musical"),
            "Bodily": data.get("Bodily"),
            "Logical": data.get("Logical") or data.get("Logical - Mathematical"),
            "Spatial": data.get("Spatial") or data.get("Spatial-Visualization"),
            "Interpersonal": data.get("Interpersonal"),
            "Intrapersonal": data.get("Intrapersonal"),
            "Naturalist": data.get("Naturalist"),
            "used_sorted_pslots": used_sorted_slots,
        },
        "skillsRaw": data.get("skillsRaw") or data.get("skills_raw") or {},
        "skillsQuestionMap": data.get("skillsQuestionMap") or {},
        "skillsConverted": {
            "Linguistic": data.get("Linguistic"),
            "Musical": data.get("Musical"),
            "Bodily": data.get("Bodily"),
            "Logical": data.get("Logical") or data.get("Logical - Mathematical"),
            "Spatial": data.get("Spatial") or data.get("Spatial-Visualization"),
            "Interpersonal": data.get("Interpersonal"),
            "Intrapersonal": data.get("Intrapersonal"),
            "Naturalist": data.get("Naturalist"),
        },
        "matric": {
            "stream": data.get("matric_stream") or data.get("matricStream"),
            "marks": data.get("matric_marks") or data.get("matricMarks") or {},
        },
        "intermediate": {
            "stream": data.get("intermediate_stream") or data.get("intermediateStream"),
            "marks": data.get("intermediate_marks") or data.get("intermediateMarks") or {},
        },
        "studyResult": None,
        "jobsResult": None,
        "cvResult": None,
        "payload": {},
    }


@app.route("/user/guidance", methods=["GET", "POST"])
def user_guidance():
    """Save/load guidance results inside the logged-in MongoDB user document."""
    user = _current_user()
    if not user:
        return jsonify({"message": "Login required"}), 401

    if request.method == "GET":
        guidance = user.get("guidance") or []
        if not isinstance(guidance, list):
            guidance = []
        guidance = [_public_guidance_entry(g) for g in guidance]
        guidance.sort(key=lambda g: g.get("createdAt") or "", reverse=True)
        return jsonify({"success": True, "guidance": guidance})

    body = request.json or {}
    guidance_type = str(body.get("type") or "career").strip().lower()
    if guidance_type not in ("career", "study", "jobs", "cv", "insights", "scope"):
        return jsonify({"message": "type must be career, study, jobs, cv, insights, or scope"}), 400

    entry = {
        "id": str(uuid.uuid4()),
        "type": guidance_type,
        "createdAt": datetime.datetime.utcnow().isoformat() + "Z",
        "title": str(body.get("title") or body.get("career") or guidance_type).strip(),
        "career": body.get("career"),
        "topCareers": body.get("topCareers") or body.get("top_careers") or [],
        "fullData": body.get("fullData") or body.get("full_data") or {},
        "skillsRaw": body.get("skillsRaw") or body.get("skills_raw") or {},
        "skillsQuestionMap": body.get("skillsQuestionMap") or {},
        "skillsConverted": body.get("skillsConverted") or {},
        "matric": body.get("matric") or {},
        "intermediate": body.get("intermediate") or {},
        "studyResult": body.get("studyResult") or body.get("study_result"),
        "jobsResult": body.get("jobsResult") or body.get("jobs_result"),
        "cvResult": body.get("cvResult") or body.get("cv_result"),
        "payload": body.get("payload") or {},
    }

    if not _save_guidance_for_user(user, entry):
        return jsonify({"message": "Database connection failed"}), 500

    return jsonify({"success": True, "entry": entry}), 201


@app.route("/user/guidance/<entry_id>", methods=["GET", "DELETE"])
def user_guidance_item(entry_id):
    user = _current_user()
    if not user:
        return jsonify({"message": "Login required"}), 401

    guidance = user.get("guidance") or []
    if not isinstance(guidance, list):
        guidance = []

    match = next((g for g in guidance if str(g.get("id")) == str(entry_id)), None)
    if not match:
        return jsonify({"message": "Guidance entry not found"}), 404

    if request.method == "GET":
        return jsonify({"success": True, "entry": _public_guidance_entry(match)})

    try:
        users.update_one(
            {"_id": user["_id"]},
            {"$pull": {"guidance": {"id": entry_id}}},
        )
    except PyMongoError:
        return jsonify({"message": "Database connection failed"}), 500

    return jsonify({"success": True, "message": "Deleted"})


@app.route("/predict-career", methods=["POST"])
def predict_career():
    if model is None or encoder is None:
        return jsonify({
            "message": "Model files not loaded. Train and save model.pkl and encoder.pkl first."
        }), 503

    data = request.json or {}

    performance_map = {"POOR": 0, "AVG": 1, "BEST": 2}

    def normalize_level(value, mapping):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return mapping.get(value.strip().upper())
        return None

    def pick_skill_value(payload, keys):
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    matric_marks = data.get("matric_marks") or data.get("matricMarks")
    inter_marks = data.get("intermediate_marks") or data.get("intermediateMarks")
    matric_stream = data.get("matric_stream") or data.get("matricStream")
    intermediate_stream = data.get("intermediate_stream") or data.get("intermediateStream")

    use_sorted_slots = False
    try:
        use_sorted_slots = (
            isinstance(matric_marks, dict)
            and isinstance(inter_marks, dict)
            and len(matric_marks) > 0
            and len(inter_marks) > 0
        )
        if use_sorted_slots:
            p_tuple = marks_to_pslots_sorted(matric_marks, inter_marks)
            P1, P2, P3, P4, P5, P6, P7, P8 = p_tuple
        else:
            P1 = normalize_level(data["P1"], performance_map)
            P2 = normalize_level(data["P2"], performance_map)
            P3 = normalize_level(data["P3"], performance_map)
            P4 = normalize_level(data["P4"], performance_map)
            P5 = normalize_level(data["P5"], performance_map)
            P6 = normalize_level(data["P6"], performance_map)
            P7 = normalize_level(data["P7"], performance_map)
            P8 = normalize_level(data["P8"], performance_map)

        # Skills must match Kaggle career_data.xlsx columns (≈0–20), not 1–3.
        Linguistic = _skill_to_kaggle_scale(data["Linguistic"])
        Musical = _skill_to_kaggle_scale(data["Musical"])
        Bodily = _skill_to_kaggle_scale(data["Bodily"])
        Logical = _skill_to_kaggle_scale(
            pick_skill_value(data, ["Logical", "Logical - Mathematical"])
        )
        Spatial = _skill_to_kaggle_scale(
            pick_skill_value(data, ["Spatial", "Spatial-Visualization"])
        )
        Interpersonal = _skill_to_kaggle_scale(data["Interpersonal"])
        Intrapersonal = _skill_to_kaggle_scale(data["Intrapersonal"])
        Naturalist = _skill_to_kaggle_scale(data["Naturalist"])
    except KeyError as e:
        return jsonify({"message": f"Invalid or missing field: {str(e)}"}), 400
    except TypeError as e:
        return jsonify({"message": f"Invalid marks payload: {str(e)}"}), 400

    normalized = {
        "P1": P1, "P2": P2, "P3": P3, "P4": P4,
        "P5": P5, "P6": P6, "P7": P7, "P8": P8,
        "Linguistic": Linguistic, "Musical": Musical, "Bodily": Bodily,
        "Logical": Logical, "Logical - Mathematical": Logical,
        "Spatial": Spatial, "Spatial-Visualization": Spatial,
        "Interpersonal": Interpersonal, "Intrapersonal": Intrapersonal, "Naturalist": Naturalist
    }

    if any(v is None for v in [P1, P2, P3, P4, P5, P6, P7, P8]):
        if use_sorted_slots:
            return jsonify({"message": "Could not derive P-slots from marks"}), 400
        return jsonify({"message": "Invalid P-slot levels. Use POOR/AVG/BEST or 0/1/2"}), 400
    if any(v is None for v in [Linguistic, Musical, Bodily, Logical, Spatial, Interpersonal, Intrapersonal, Naturalist]):
        return jsonify({
            "message": "Invalid skill levels. Use 1–5 Likert, 0–20 Kaggle scale, or LOW/MEDIUM/HIGH"
        }), 400

    if MODEL_FEATURE_ORDER:
        try:
            input_row = [normalized[col] for col in MODEL_FEATURE_ORDER]
        except KeyError as e:
            return jsonify({"message": f"Model feature missing in request mapping: {str(e)}"}), 500
    else:
        input_row = [
            P1, P2, P3, P4, P5, P6, P7, P8,
            Linguistic, Musical, Bodily, Logical,
            Spatial, Interpersonal, Intrapersonal, Naturalist
        ]

    input_data = np.array([input_row])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        classes = np.asarray(encoder.classes_)
        class_indices = list(range(len(probs)))
        class_labels = [str(c) for c in classes]

        # Blend heuristics still use ternary buckets; model itself gets 0–20.
        adjusted = blend_career_probabilities(
            probs=probs,
            class_indices=class_indices,
            career_labels=class_labels,
            matric_stream=matric_stream,
            intermediate_stream=intermediate_stream,
            skills={
                "Linguistic": _kaggle_to_ternary(Linguistic),
                "Musical": _kaggle_to_ternary(Musical),
                "Bodily": _kaggle_to_ternary(Bodily),
                "Logical": _kaggle_to_ternary(Logical),
                "Spatial": _kaggle_to_ternary(Spatial),
                "Interpersonal": _kaggle_to_ternary(Interpersonal),
                "Intrapersonal": _kaggle_to_ternary(Intrapersonal),
                "Naturalist": _kaggle_to_ternary(Naturalist),
            }
        )
        if not adjusted:
            return jsonify({
                "message": "No valid careers found after applying alignment rules."
            }), 422

        top_k = min(4, len(adjusted))
        top_rows = adjusted[:top_k]
        top_idx = [row["idx"] for row in top_rows]

        top_results = []
        for row in top_rows:
            idx = int(row["idx"])
            p = float(probs[idx])
            top_results.append({
                "career": str(classes[idx]),
                "confidence": round(p * 100, 2),
                "why": row.get("why", []),
                "fit_breakdown": {
                    "brain": round(float(row.get("brain_fit", 0.0)) * 100, 1),
                    "academic": round(float(row.get("academic_fit", 0.0)) * 100, 1),
                    "model": round(float(row.get("model_fit", 0.0)) * 100, 1),
                },
            })

        best = int(top_idx[0])
        response_payload = {
            "predicted_career": str(classes[best]),
            "top_careers": top_results,
            "used_sorted_pslots": use_sorted_slots,
            "skill_scale": "kaggle_0_20",
            "kaggle_skill_columns": KAGGLE_SKILL_COLUMNS,
            "saved_to_account": False,
            "saved_guidance_id": None,
        }

        # If logged in, also store this prediction under users.guidance
        user = _current_user()
        if user:
            entry = _build_career_guidance_entry(
                data, response_payload["predicted_career"], top_results, use_sorted_slots
            )
            if _save_guidance_for_user(user, entry):
                response_payload["saved_to_account"] = True
                response_payload["saved_guidance_id"] = entry["id"]

        return jsonify(response_payload)

    prediction = model.predict(input_data)
    career_name = encoder.inverse_transform(prediction)
    predicted = career_name[0]
    top_results = [{"career": predicted, "confidence": None}]
    response_payload = {
        "predicted_career": predicted,
        "top_careers": top_results,
        "saved_to_account": False,
        "saved_guidance_id": None,
    }
    user = _current_user()
    if user:
        entry = _build_career_guidance_entry(data, predicted, top_results, use_sorted_slots)
        if _save_guidance_for_user(user, entry):
            response_payload["saved_to_account"] = True
            response_payload["saved_guidance_id"] = entry["id"]
    return jsonify(response_payload)


def _merge_job_proficiency(degrees, raw_rows, legacy_related_fields):
    """Align percentages with canonical degree list (Pakistan job-market proxy %)."""
    rows = raw_rows or []
    normalized = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        deg = str(item.get("degree") or item.get("field") or "").strip()
        pct = item.get("percentage", item.get("pct"))
        try:
            pct = max(1, min(100, int(round(float(pct)))))
        except (TypeError, ValueError):
            continue
        if deg:
            normalized.append({"degree": deg.lower(), "percentage": pct})

    by_name = {r["degree"]: r["percentage"] for r in normalized}
    pcts = []
    for i, deg in enumerate(degrees):
        key = deg.lower()
        pct = None
        if i < len(rows) and isinstance(rows[i], dict):
            try:
                pct = max(1, min(100, int(round(float(rows[i].get("percentage", 0))))))
            except (TypeError, ValueError):
                pct = None
        if pct is None:
            pct = by_name.get(key)
        if pct is None:
            for cand, val in by_name.items():
                if cand in key or key in cand:
                    pct = val
                    break
        if pct is None and i < len(legacy_related_fields):
            try:
                pct = max(1, min(100, int(round(float(legacy_related_fields[i].get("percentage", 0))))))
            except (TypeError, ValueError, IndexError):
                pct = None
        if pct is None:
            pct = max(38, min(88, 76 - i * 7))
        pcts.append(int(pct))
    return pcts


def _finalize_insights_payload(parsed, career):
    degrees = [str(x).strip() for x in (parsed.get("degrees") or []) if str(x).strip()]
    institutes = [str(x).strip() for x in (parsed.get("institutes") or []) if str(x).strip()]
    legacy_related = parsed.get("related_fields") or []
    if not isinstance(legacy_related, list):
        legacy_related = []

    if len(degrees) < 4:
        if legacy_related and len(degrees) == 0:
            degrees = [
                str(x.get("field", "")).strip()
                for x in legacy_related
                if str(x.get("field", "")).strip()
            ]
        if len(degrees) < 4:
            return None

    jp_raw = parsed.get("job_proficiency") or parsed.get("job_proficiencies") or []
    pcts = _merge_job_proficiency(degrees, jp_raw, legacy_related)

    # Universities: prefer objects {name, url, programs}, accept plain strings
    uni_raw = parsed.get("universities") or parsed.get("top_universities") or []
    universities = []
    if isinstance(uni_raw, list):
        for item in uni_raw:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("university") or "").strip()
                url = str(item.get("url") or item.get("portal") or item.get("link") or "").strip()
                programs = item.get("programs") or item.get("offers") or []
                if isinstance(programs, str):
                    programs = [programs]
                programs = [str(p).strip() for p in programs if str(p).strip()][:4]
                note = str(item.get("note") or item.get("why") or "").strip()
                if name:
                    universities.append({
                        "name": name,
                        "url": url,
                        "programs": programs,
                        "note": note,
                    })
            else:
                name = str(item).strip()
                if name:
                    universities.append({
                        "name": name,
                        "url": "",
                        "programs": [],
                        "note": "",
                    })

    if len(universities) < 6:
        return None

    job_proficiency = [{"degree": d, "percentage": p} for d, p in zip(degrees, pcts)]
    return {
        "career": parsed.get("career") or career,
        "degrees": degrees,
        "universities": universities[:10],
        "top_universities": [u["name"] for u in universities[:10]],
        "job_proficiency": job_proficiency,
        "institutes": institutes[:6],
    }


@app.route("/career-insights", methods=["POST"])
def career_insights():
    payload = request.json or {}
    career = str(payload.get("career", "")).strip()
    if not career:
        return jsonify({"message": "career is required"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"message": "OPENAI_API_KEY is missing in .env (Cursor/backend env)"}), 503

    prompt = (
        "You are a Pakistan higher-education and labour-market advisor.\n"
        f'Career title: "{career}"\n'
        "Return strict JSON only (no markdown) with keys exactly:\n"
        "{\n"
        '  "career": string,\n'
        '  "degrees": string[],\n'
        '  "universities": [{"name": string, "url": string, "programs": string[], "note": string}],\n'
        '  "job_proficiency": [{"degree": string, "percentage": number}],\n'
        '  "institutes": string[]\n'
        "}\n"
        "Rules:\n"
        "- degrees: 6 to 8 items — concrete Pakistan-style qualifications "
        "(e.g. BS CS, BS SE, BS IT, ADP Computing) relevant to this career.\n"
        "- universities: exactly 8 to 10 distinct Pakistan universities. "
        "Each must include: official website url (https://...), 1–3 program names "
        "this university actually offers that lead toward this career, and a short note.\n"
        "Use real official portals when known (nust.edu.pk, lums.edu.pk, nu.edu.pk, "
        "comsats.edu.pk, uet.edu.pk, neduet.edu.pk, iba.edu.pk, pu.edu.pk, etc.).\n"
        "- job_proficiency: same length/order as degrees; Pakistan job-market alignment 1–100.\n"
        "- institutes: 4 to 6 vocational / skills bodies (NAVTTC, TEVTA, etc.).\n"
        "- Only Pakistan institutions. No invented fake .edu domains if unsure — "
        "use the best-known official homepage.\n"
    )

    req_body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    try:
        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(req_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=35) as resp:
            raw = resp.read().decode("utf-8")
            data_obj = json.loads(raw)
            content = data_obj["choices"][0]["message"]["content"]
            parsed = json.loads(content)
    except (HTTPError, URLError, KeyError, ValueError, TimeoutError) as e:
        return jsonify({"message": f"OpenAI request failed: {str(e)}"}), 502

    finalized = _finalize_insights_payload(parsed, career)
    if finalized is None:
        return jsonify({"message": "OpenAI returned invalid insights payload"}), 502
    return jsonify({**finalized, "source": "openai", "env": "OPENAI_API_KEY"})


@app.route("/jobs-guidance", methods=["POST"])
def jobs_guidance():
    data = request.json or {}
    ok, payload = call_jobs_openai(data, OPENAI_API_KEY, OPENAI_MODEL)
    if ok:
        return jsonify(payload)
    msg = str(payload.get("message", "")).lower()
    status = 400 if "education_level" in msg or "field or program" in msg else (503 if "missing" in msg else 502)
    return jsonify(payload), status


@app.route("/cv-maker", methods=["POST"])
def cv_maker():
    data = request.json or {}
    ok, payload = call_cv_openai(data, OPENAI_API_KEY, OPENAI_MODEL)
    if ok:
        return jsonify(payload)
    msg = str(payload.get("message", "")).lower()
    if "missing" in msg and "openai" in msg:
        status = 503
    elif "required" in msg:
        status = 400
    else:
        status = 502
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(debug=True)
