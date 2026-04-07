from flask import Blueprint, request, jsonify
from pymongo import MongoClient
import bcrypt
import jwt
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

auth_routes = Blueprint("auth", __name__)

client = MongoClient(os.getenv("MONGO_URI"))
db = client["growsmart"]
users = db["users"]

SECRET_KEY = os.getenv("SECRET_KEY")


# ================= REGISTER =================
@auth_routes.route("/register", methods=["POST"])
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


# ================= LOGIN =================
@auth_routes.route("/login", methods=["POST"])
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


# ================= FORGOT PASSWORD =================
@auth_routes.route("/forgot-password", methods=["POST"])
def forgot_password():

    data = request.json

    user = users.find_one({"email": data["email"]})

    if not user:
        return jsonify({"message": "User not found"}), 404

    # Simple reset (for now)
    new_password = bcrypt.hashpw("123456".encode(), bcrypt.gensalt())

    users.update_one(
        {"email": data["email"]},
        {"$set": {"password": new_password}}
    )

    return jsonify({
        "message": "Password reset to 123456"
    })