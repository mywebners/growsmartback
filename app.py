from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import bcrypt
import jwt
import datetime
import os
import json
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
from dotenv import load_dotenv

from utils.career_scoring import marks_to_pslots_sorted, blend_career_probabilities

load_dotenv()

app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
if "<db_password>" in MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["growsmart"]
users = db["users"]

SECRET_KEY = os.getenv("SECRET_KEY", "anas_secret_123")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}

    required_fields = ["name", "email", "password"]
    if not all(field in data and data[field] for field in required_fields):
        return jsonify({"message": "Missing required fields"}), 400

    try:
        if users.find_one({"email": data["email"]}):
            return jsonify({"message": "User already exists"}), 400
    except PyMongoError:
        return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500

    hashed_pw = bcrypt.hashpw(data["password"].encode("utf-8"), bcrypt.gensalt())

    try:
        users.insert_one({
            "name": data["name"],
            "email": data["email"],
            "password": hashed_pw
        })
    except PyMongoError:
        return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500

    return jsonify({"message": "User registered successfully"}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}

    required_fields = ["email", "password"]
    if not all(field in data and data[field] for field in required_fields):
        return jsonify({"message": "Missing required fields"}), 400

    try:
        user = users.find_one({"email": data["email"]})
    except PyMongoError:
        return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500
    if not user:
        return jsonify({"message": "User not found"}), 404

    stored_password = user.get("password")
    if isinstance(stored_password, str):
        stored_password = stored_password.encode("utf-8")

    if not stored_password or not bcrypt.checkpw(data["password"].encode("utf-8"), stored_password):
        return jsonify({"message": "Wrong password"}), 400

    token = jwt.encode(
        {
            "user_id": str(user["_id"]),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=5),
        },
        SECRET_KEY,
        algorithm="HS256",
    )

    return jsonify({
        "token": token,
        "name": user["name"]
    })


@app.route("/auth/forgot-password", methods=["POST"])
def forgot_password():
    data = request.json or {}

    if "email" not in data or not data["email"]:
        return jsonify({"message": "Email is required"}), 400

    user = users.find_one({"email": data["email"]})
    if not user:
        return jsonify({"message": "User not found"}), 404

    new_password = bcrypt.hashpw("123456".encode("utf-8"), bcrypt.gensalt())

    users.update_one(
        {"email": data["email"]},
        {"$set": {"password": new_password}}
    )

    return jsonify({"message": "Password reset to 123456"})


@app.route("/predict-career", methods=["POST"])
def predict_career():
    if model is None or encoder is None:
        return jsonify({
            "message": "Model files not loaded. Train and save model.pkl and encoder.pkl first."
        }), 503

    data = request.json or {}

    performance_map = {"POOR": 0, "AVG": 1, "BEST": 2}
    skill_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def normalize_level(value, mapping):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return mapping.get(value.strip().upper())
        return None

    def normalize_skill_input(value):
        """Map UI 1–5 or LOW/MEDIUM/HIGH to model scale 1–3 (same as frontend convertSkill)."""
        if isinstance(value, (int, float)):
            v = int(round(float(value)))
            if 1 <= v <= 5:
                if v <= 2:
                    return 1
                if v == 3:
                    return 2
                return 3
            if 1 <= v <= 3:
                return v
        if isinstance(value, str):
            return skill_map.get(value.strip().upper())
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

        Linguistic = normalize_skill_input(data["Linguistic"])
        Musical = normalize_skill_input(data["Musical"])
        Bodily = normalize_skill_input(data["Bodily"])
        Logical = normalize_skill_input(
            pick_skill_value(data, ["Logical", "Logical - Mathematical"])
        )
        Spatial = normalize_skill_input(
            pick_skill_value(data, ["Spatial", "Spatial-Visualization"])
        )
        Interpersonal = normalize_skill_input(data["Interpersonal"])
        Intrapersonal = normalize_skill_input(data["Intrapersonal"])
        Naturalist = normalize_skill_input(data["Naturalist"])
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
        return jsonify({"message": "Invalid skill levels. Use LOW/MEDIUM/HIGH or numeric scale"}), 400

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

        adjusted = blend_career_probabilities(
            probs=probs,
            class_indices=class_indices,
            career_labels=class_labels,
            matric_stream=matric_stream,
            intermediate_stream=intermediate_stream,
            skills={
                "Linguistic": Linguistic,
                "Musical": Musical,
                "Bodily": Bodily,
                "Logical": Logical,
                "Spatial": Spatial,
                "Interpersonal": Interpersonal,
                "Intrapersonal": Intrapersonal,
                "Naturalist": Naturalist,
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
        return jsonify({
            "predicted_career": str(classes[best]),
            "top_careers": top_results,
            "used_sorted_pslots": use_sorted_slots,
        })

    prediction = model.predict(input_data)
    career_name = encoder.inverse_transform(prediction)
    return jsonify({
        "predicted_career": career_name[0],
        "top_careers": [{"career": career_name[0], "confidence": None}]
    })


_DEFAULT_INSTITUTES = [
    "NAVTTC certified institutes",
    "TEVTA Punjab technical centers",
    "PITB and DigiSkills partner programs",
    "NUTECH professional diplomas",
    "Corvit and Aptech skill tracks",
    "NED academy extension programs",
]


def _pack_insights(career, degrees, universities, proficiency_pct, institutes=None):
    degrees = [str(d).strip() for d in degrees if str(d).strip()]
    universities = [str(u).strip() for u in universities if str(u).strip()]
    pcts = list(proficiency_pct or [])
    while len(pcts) < len(degrees):
        pcts.append(max(35, 72 - len(pcts) * 6))
    job_proficiency = []
    for i, deg in enumerate(degrees):
        try:
            pct = max(1, min(100, int(round(float(pcts[i])))))
        except (TypeError, ValueError, IndexError):
            pct = max(35, 72 - i * 6)
        job_proficiency.append({"degree": deg, "percentage": pct})
    return {
        "career": career,
        "degrees": degrees,
        "top_universities": universities[:10],
        "job_proficiency": job_proficiency,
        "institutes": institutes or _DEFAULT_INSTITUTES,
    }


def _fallback_career_insights(career):
    c = (career or "").lower()
    tech_kw = (
        "program", "software", "developer", "computer", "data", "ict",
        "it ", "network", "cyber", "web", "mobile", "engineer", "ai ",
        "machine learning", "database"
    )
    med_kw = ("doctor", "medical", "physician", "surgeon", "mbbs", "dent", "nurse", "pharma")
    biz_kw = ("business", "account", "finance", "marketing", "management", "commerce")

    if any(k in c for k in tech_kw):
        degrees = [
            "BS Computer Science (BSCS)",
            "BS Software Engineering (BSSE)",
            "BS Information Technology (BSIT)",
            "BS Data Science",
            "Associate Degree Program (ADP) / ADA in Computing",
            "Diploma of Associate Engineer (DAE) — Computer / IT",
        ]
        unis = [
            "FAST-NUCES (Islamabad, Lahore, Karachi)",
            "NUST — SEECS / SCEE",
            "COMSATS University",
            "ITU Lahore",
            "GIKI — Faculty of Computer Science",
            "UET Lahore — Computer Science & Engineering",
            "Bahria University — Computing",
            "Air University — Aerospace / Computing",
            "NED University — Computer & Info Systems",
            "University of the Punjab — IT / CS departments",
        ]
        pcts = [82, 79, 74, 71, 62, 55]
        return _pack_insights(career, degrees, unis, pcts)

    if any(k in c for k in med_kw):
        degrees = [
            "MBBS",
            "BDS (Dentistry)",
            "Doctor of Pharmacy (Pharm-D)",
            "BS Nursing (Generic)",
            "Doctor of Physical Therapy (DPT)",
            "BS Medical Laboratory Technology (BSMLT)",
        ]
        unis = [
            "King Edward Medical University",
            "Allama Iqbal Medical College",
            "Dow University of Health Sciences",
            "Aga Khan University",
            "Army Medical College (NUMS)",
            "Fatima Jinnah Medical University",
            "KEMU / affiliated teaching hospitals network",
            "LUMHS Jamshoro",
            "Khyber Medical University",
            "Islamabad Medical & Dental College",
        ]
        pcts = [84, 71, 76, 69, 63, 58]
        return _pack_insights(career, degrees, unis, pcts)

    if any(k in c for k in biz_kw):
        degrees = [
            "BBA / BS Business Administration",
            "BS Accounting & Finance",
            "BS Economics",
            "MBA / MS Management",
            "ACCA / CA pathway (with degree)",
            "Diploma / certificate in digital marketing / HR",
        ]
        unis = [
            "IBA Karachi",
            "LUMS",
            "University of the Punjab — Hailey College",
            "Institute of Management Sciences (IMS) Lahore",
            "SZABIST",
            "FAST School of Management",
            "NUST Business School",
            "University of Karachi — Commerce",
            "Bahria University — Management Sciences",
            "COMSATS — Management Sciences",
        ]
        pcts = [77, 73, 68, 72, 61, 54]
        return _pack_insights(career, degrees, unis, pcts)

    degrees = [
        f"BS / undergraduate degree aligned with {career}",
        "Associate Degree Program (ADP) in a related discipline",
        "Diploma / certificate from HEC-recognized institute",
        "Graduate / master's (MS / MPhil) in related field",
        "Professional certification or licensing pathway",
        "Short vocational train-the-trainer / NAVTTC skill course",
    ]
    unis = [
        "University of the Punjab",
        "University of Karachi",
        "University of Peshawar",
        "Bahauddin Zakariya University",
        "Government College University (GCU) Lahore",
        "COMSATS University",
        "NUML",
        "Virtual University of Pakistan",
        "Allama Iqbal Open University",
        "IBA / SZABIST / regional reputable departments",
    ]
    pcts = [70, 58, 52, 66, 48, 44]
    return _pack_insights(career, degrees, unis, pcts)


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
    universities = [str(x).strip() for x in (parsed.get("top_universities") or []) if str(x).strip()]
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

    packed = _pack_insights(
        parsed.get("career") or career,
        degrees,
        universities,
        pcts,
        institutes if len(institutes) >= 3 else None,
    )
    if len(packed["top_universities"]) < 6:
        return None
    return packed


@app.route("/career-insights", methods=["POST"])
def career_insights():
    payload = request.json or {}
    career = str(payload.get("career", "")).strip()
    if not career:
        return jsonify({"message": "career is required"}), 400

    if not OPENAI_API_KEY:
        return jsonify(_fallback_career_insights(career))

    prompt = (
        "You are a Pakistan higher-education and labour-market advisor.\n"
        f'Career title: "{career}"\n'
        "Return strict JSON only (no markdown) with keys exactly:\n"
        "{\n"
        '  "career": string,\n'
        '  "degrees": string[],\n'
        '  "top_universities": string[],\n'
        '  "job_proficiency": [{"degree": string, "percentage": number}],\n'
        '  "institutes": string[]\n'
        "}\n"
        "Rules:\n"
        "- degrees: 6 to 8 items — concrete Pakistan-style qualifications "
        "(e.g. BS/BSc names, MBBS, diplomas, ADP, DAE, MS/MPhil where relevant).\n"
        "- top_universities: exactly 10 distinct Pakistan universities or campuses "
        "that are especially strong for those degrees (not random rankings).\n"
        "- job_proficiency: same length and same order as degrees; each percentage "
        "is an approximate Pakistan job-market alignment score (1–100) for holders "
        "of that qualification toward this career.\n"
        "- institutes: 4 to 6 vocational / diploma / skills bodies (e.g. NAVTTC, "
        "TEVTA centres, sector skills councils) relevant to this career.\n"
        "- Use realistic Pakistani naming; avoid non-Pakistan institutions.\n"
    )

    req_body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.35,
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
        with urlrequest.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            data_obj = json.loads(raw)
            content = data_obj["choices"][0]["message"]["content"]
            parsed = json.loads(content)
    except (HTTPError, URLError, KeyError, ValueError, TimeoutError):
        return jsonify(_fallback_career_insights(career))

    finalized = _finalize_insights_payload(parsed, career)
    if finalized is None:
        return jsonify(_fallback_career_insights(career))
    return jsonify(finalized)


if __name__ == "__main__":
    app.run(debug=True)