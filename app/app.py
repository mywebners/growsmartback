from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# 🔥 NEW IMPORTS
from pymongo import MongoClient
import bcrypt
import jwt
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ====================================
# MongoDB Connection
# ====================================
client = MongoClient("mongodb://localhost:27017/")
db = client["growsmart"]
users = db["users"]

SECRET_KEY = "anas_secret_123"


# ====================================
# Load ML Model
# ====================================
model = joblib.load("../model/career_model.pkl")
label_encoder = joblib.load("../model/label_encoder.pkl")


# ====================================
# HOME
# ====================================
@app.route("/")
def home():
    return "GrowSmart API Running"


# ====================================
# 🔐 REGISTER
# ====================================
@app.route("/auth/register", methods=["POST"])
def register():

    data = request.json

    if users.find_one({"email": data["email"]}):
        return jsonify({"message": "User already exists"}), 400

    hashed_pw = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt())

    users.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hashed_pw
    })

    return jsonify({"message": "User registered successfully"})


# ====================================
# 🔐 LOGIN
# ====================================
@app.route("/auth/login", methods=["POST"])
def login():

    data = request.json

    user = users.find_one({"email": data["email"]})

    if not user:
        return jsonify({"message": "User not found"}), 404

    if not bcrypt.checkpw(data["password"].encode(), user["password"]):
        return jsonify({"message": "Wrong password"}), 400

    token = jwt.encode({
        "user_id": str(user["_id"]),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=5)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "token": token,
        "name": user["name"]
    })


# ====================================
# 🔐 FORGOT PASSWORD
# ====================================
@app.route("/auth/forgot-password", methods=["POST"])
def forgot_password():

    data = request.json

    user = users.find_one({"email": data["email"]})

    if not user:
        return jsonify({"message": "User not found"}), 404

    new_password = bcrypt.hashpw("123456".encode(), bcrypt.gensalt())

    users.update_one(
        {"email": data["email"]},
        {"$set": {"password": new_password}}
    )

    return jsonify({
        "message": "Password reset to 123456"
    })


# ====================================
# 🎯 CAREER PREDICTION (YOUR OLD CODE)
# ====================================
@app.route("/predict-career", methods=["POST"])
def predict_career():
    try:
        print("🔥 PREDICTION REQUEST RECEIVED")
        data = request.json
        print("📥 Input data keys:", list(data.keys()))

        if not all(key in data for key in ["P1","P2","P3","P4","P5","P6","P7","P8",
                                          "Linguistic","Musical","Bodily","Logical",
                                          "Spatial","Interpersonal","Intrapersonal","Naturalist"]):
            return jsonify({"error": "Missing required fields"}), 400

        performance_map = {"POOR": 0, "AVG": 1, "BEST": 2}
        skill_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

        P1 = performance_map[data["P1"]]
        P2 = performance_map[data["P2"]]
        P3 = performance_map[data["P3"]]
        P4 = performance_map[data["P4"]]
        P5 = performance_map[data["P5"]]
        P6 = performance_map[data["P6"]]
        P7 = performance_map[data["P7"]]
        P8 = performance_map[data["P8"]]

        Linguistic = skill_map[data["Linguistic"]]
        Musical = skill_map[data["Musical"]]
        Bodily = skill_map[data["Bodily"]]
        Logical = skill_map[data["Logical"]]
        Spatial = skill_map[data["Spatial"]]
        Interpersonal = skill_map[data["Interpersonal"]]
        Intrapersonal = skill_map[data["Intrapersonal"]]
        Naturalist = skill_map[data["Naturalist"]]

        input_data = np.array([[ 
            Linguistic, Musical, Bodily, Logical,
            Spatial, Interpersonal, Intrapersonal, Naturalist,
            P1, P2, P3, P4, P5, P6, P7, P8
        ]])

        print("🔬 Input features:", input_data.tolist()[0])

        prediction = model.predict(input_data)
        career_name = label_encoder.inverse_transform(prediction)
        
        print("🎯 PREDICTED CAREER:", career_name[0])

        return jsonify({
            "predicted_career": career_name[0],
            "success": True
        })
    
    except KeyError as e:
        print(f"❌ KeyError: {e}")
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        print(f"💥 Prediction Error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


# ====================================
# RUN SERVER
# ====================================
if __name__ == "__main__":
    app.run(debug=True)
