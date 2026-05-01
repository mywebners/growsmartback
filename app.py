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
from dotenv import load_dotenv

from utils.career_scoring import marks_to_pslots_sorted, blend_career_probabilities

load_dotenv()

app = Flask(__name__)
CORS(app)

# ====================================
# MongoDB Connection
# ====================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# If .env contains an unfinished Atlas URI, fallback to local MongoDB.
if "<db_password>" in MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["growsmart"]
users = db["users"]

SECRET_KEY = os.getenv("SECRET_KEY", "anas_secret_123")

# ====================================
# Load ML Model (robust path resolution)
# ====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try both common locations:
# 1) growsmartback/model/
# 2) project-root/model/
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

# ====================================
# HOME
# ====================================
@app.route("/")
def home():
    return "GrowSmart API Running"


# ====================================
# REGISTER
# ====================================
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


# ====================================
# LOGIN
# ====================================
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


# ====================================
# FORGOT PASSWORD
# ====================================
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


# ====================================
# CAREER PREDICTION
# ====================================
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

        # Return 4 strong aligned options (student gets choice without irrelevant careers).
        top_k = min(4, len(adjusted))
        top_idx = [idx for idx, _ in adjusted[:top_k]]
        adjusted_sum = float(sum(score for _, score in adjusted[:top_k])) or 1e-9

        top_results = []
        for idx, adj_score in adjusted[:top_k]:
            idx = int(idx)
            p = float(probs[idx])
            top_results.append({
                "career": str(classes[idx]),
                "confidence": round(p * 100, 2),
                "blend_confidence": round(100.0 * float(adj_score) / adjusted_sum, 2),
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


# ====================================
# RUN SERVER
# ====================================
if __name__ == "__main__":
    app.run(debug=True)