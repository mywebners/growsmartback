"""
Auth routes: register / login / me / profile / forgot-password / reset-password.
Forgot password sends a Resend email with a one-time link (no instant 123456 reset).
GET /auth/me is the first call after login — profile + full guidance history.
"""
from flask import Blueprint, request, jsonify
from pymongo.errors import PyMongoError
from bson.objectid import ObjectId
from bson.errors import InvalidId
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
import bcrypt
import jwt
import datetime
import secrets
import json
import hashlib


def create_auth_blueprint(
    users,
    secret_key,
    resend_api_key="",
    resend_from_email="GrowSmart <onboarding@resend.dev>",
    resend_reply_to="anas.dev200@gmail.com",
    frontend_url="http://localhost:3000",
):
    """
    users: pymongo collection
    secret_key: JWT signing key
    resend_api_key: Resend API key
    resend_from_email: MUST be a Resend address or verified domain
      (without domain use onboarding@resend.dev — Gmail cannot be From)
    resend_reply_to: your Gmail so users can reply to you
    """
    auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

    # Profile image: data-URL / base64 — keep under ~400KB for MongoDB comfort
    MAX_IMAGE_CHARS = 550_000

    def _make_token(user_id):
        return jwt.encode(
            {
                "user_id": str(user_id),
                "exp": datetime.datetime.utcnow() + datetime.timedelta(days=5),
            },
            secret_key,
            algorithm="HS256",
        )

    def _safe_password_bytes(stored):
        if isinstance(stored, str):
            return stored.encode("utf-8")
        return stored

    def _hash_reset_token(raw_token: str) -> str:
        """Store only a hash of the reset token in MongoDB."""
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def _bearer_user():
        """Resolve logged-in user from Authorization: Bearer <jwt>."""
        auth = request.headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            return None
        token = auth[7:].strip()
        if not token:
            return None
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            uid = payload.get("user_id")
            if not uid:
                return None
            return users.find_one({"_id": ObjectId(uid)})
        except (jwt.PyJWTError, InvalidId, PyMongoError, TypeError):
            return None

    def _public_guidance(entry):
        if not isinstance(entry, dict):
            return entry
        out = dict(entry)
        out.pop("_id", None)
        return out

    def _build_me_payload(user):
        guidance = user.get("guidance") or []
        if not isinstance(guidance, list):
            guidance = []
        guidance = [_public_guidance(g) for g in guidance]
        guidance.sort(key=lambda g: g.get("createdAt") or "", reverse=True)

        by_type = {
            "career": [],
            "study": [],
            "jobs": [],
            "cv": [],
            "insights": [],
            "scope": [],
            "other": [],
        }
        for g in guidance:
            t = str(g.get("type") or "other").strip().lower()
            if t in by_type:
                by_type[t].append(g)
            else:
                by_type["other"].append(g)

        return {
            "success": True,
            "profile": {
                "id": str(user["_id"]),
                "name": user.get("name") or "",
                "email": user.get("email") or "",
                "image": user.get("image") or "",
            },
            "guidance": guidance,
            "history": {
                "all": guidance,
                "career": by_type["career"],
                "study": by_type["study"],
                "jobs": by_type["jobs"],
                "cv": by_type["cv"],
                "insights": by_type["insights"],
                "scope": by_type["scope"],
                "other": by_type["other"],
            },
            "counts": {
                "total": len(guidance),
                "career": len(by_type["career"]),
                "study": len(by_type["study"]),
                "jobs": len(by_type["jobs"]),
                "cv": len(by_type["cv"]),
            },
        }

    def _send_resend_email(to_email: str, subject: str, html: str):
        if not resend_api_key:
            return False, "RESEND_API_KEY is missing in growsmartback/.env"

        from_addr = (resend_from_email or "GrowSmart <onboarding@resend.dev>").strip()
        # Gmail/Yahoo cannot be Resend "from" without owning that domain
        lower_from = from_addr.lower()
        if any(d in lower_from for d in ("@gmail.com", "@yahoo.com", "@outlook.com", "@hotmail.com")):
            return False, (
                "RESEND_FROM_EMAIL cannot be a Gmail/Yahoo address. "
                "Without a verified domain use: GrowSmart <onboarding@resend.dev> "
                "and set RESEND_REPLY_TO=anas.dev200@gmail.com so replies go to you."
            )

        payload = {
            "from": from_addr,
            "to": [to_email],
            "subject": subject,
            "html": html,
        }
        if resend_reply_to:
            payload["reply_to"] = resend_reply_to.strip()

        try:
            req = urlrequest.Request(
                "https://api.resend.com/emails",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {resend_api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 GrowSmart/1.0",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                json.loads(raw)
            return True, None
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            # Surface Resend JSON message clearly (502 body)
            try:
                err_obj = json.loads(detail)
                msg = err_obj.get("message") or detail
            except Exception:
                msg = detail
            return False, f"Resend error: {str(msg)[:400]}"
        except (URLError, TimeoutError, ValueError, KeyError) as e:
            return False, f"Resend request failed: {str(e)}"

    @auth_bp.route("/register", methods=["POST"])
    def register():
        data = request.json or {}

        required_fields = ["name", "email", "password"]
        if not all(field in data and data[field] for field in required_fields):
            return jsonify({"message": "Missing required fields"}), 400

        email = str(data["email"]).strip().lower()
        name = str(data["name"]).strip()
        password = data["password"]

        if len(str(password)) < 6:
            return jsonify({"message": "Password must be at least 6 characters"}), 400

        try:
            if users.find_one({"email": email}):
                return jsonify({"message": "User already exists"}), 400
        except PyMongoError:
            return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500

        hashed_pw = bcrypt.hashpw(str(password).encode("utf-8"), bcrypt.gensalt())

        try:
            result = users.insert_one({
                "name": name,
                "email": email,
                "password": hashed_pw,
                "image": "",
                "guidance": [],
            })
        except PyMongoError:
            return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500

        token = _make_token(result.inserted_id)

        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "name": name,
            "email": email,
            "image": "",
        }), 201

    @auth_bp.route("/login", methods=["POST"])
    def login():
        data = request.json or {}

        required_fields = ["email", "password"]
        if not all(field in data and data[field] for field in required_fields):
            return jsonify({"message": "Missing required fields"}), 400

        email = str(data["email"]).strip().lower()
        password = data["password"]

        try:
            user = users.find_one({"email": email})
        except PyMongoError:
            return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500

        if not user:
            return jsonify({"message": "User not found"}), 404

        stored_password = _safe_password_bytes(user.get("password"))

        if not stored_password or not bcrypt.checkpw(
            str(password).encode("utf-8"), stored_password
        ):
            return jsonify({"message": "Wrong password"}), 400

        token = _make_token(user["_id"])

        return jsonify({
            "token": token,
            "name": user["name"],
            "email": user.get("email"),
            "image": user.get("image") or "",
        })

    @auth_bp.route("/me", methods=["GET"])
    def me():
        """
        First API after login / on app load.
        Returns profile + full guidance history (career, jobs, cv, study, …).
        Header: Authorization: Bearer <jwt>
        """
        user = _bearer_user()
        if not user:
            return jsonify({"message": "Login required"}), 401
        try:
            return jsonify(_build_me_payload(user))
        except PyMongoError:
            return jsonify({"message": "Database connection failed"}), 500

    @auth_bp.route("/profile", methods=["PUT", "PATCH"])
    def update_profile():
        """
        Body (any subset):
          { name, image, newPassword, currentPassword }
        image: data URL string (e.g. data:image/jpeg;base64,...) or empty to clear.
        newPassword needs currentPassword when changing password.
        """
        user = _bearer_user()
        if not user:
            return jsonify({"message": "Login required"}), 401

        data = request.json or {}
        updates = {}

        if "name" in data:
            name = str(data.get("name") or "").strip()
            if not name:
                return jsonify({"message": "Name cannot be empty"}), 400
            if len(name) > 80:
                return jsonify({"message": "Name is too long"}), 400
            updates["name"] = name

        if "image" in data:
            image = data.get("image")
            if image is None or image == "":
                updates["image"] = ""
            else:
                image = str(image)
                if len(image) > MAX_IMAGE_CHARS:
                    return jsonify({
                        "message": "Image is too large. Use a smaller photo (under ~400KB)."
                    }), 400
                if not (
                    image.startswith("data:image/")
                    or image.startswith("http://")
                    or image.startswith("https://")
                ):
                    return jsonify({
                        "message": "Image must be a data URL or http(s) URL"
                    }), 400
                updates["image"] = image

        new_password = data.get("newPassword") or data.get("new_password")
        if new_password:
            if len(str(new_password)) < 6:
                return jsonify({"message": "Password must be at least 6 characters"}), 400
            current = data.get("currentPassword") or data.get("current_password") or ""
            stored = _safe_password_bytes(user.get("password"))
            if not stored or not bcrypt.checkpw(str(current).encode("utf-8"), stored):
                return jsonify({"message": "Current password is wrong"}), 400
            updates["password"] = bcrypt.hashpw(
                str(new_password).encode("utf-8"), bcrypt.gensalt()
            )

        if not updates:
            return jsonify({"message": "Nothing to update"}), 400

        try:
            users.update_one({"_id": user["_id"]}, {"$set": updates})
            fresh = users.find_one({"_id": user["_id"]})
        except PyMongoError:
            return jsonify({"message": "Database connection failed"}), 500

        payload = _build_me_payload(fresh)
        payload["message"] = "Profile updated"
        return jsonify(payload)

    @auth_bp.route("/forgot-password", methods=["POST"])
    def forgot_password():
        """
        Create a one-time reset token, email a link via Resend.
        Does NOT change the password yet.
        """
        data = request.json or {}

        if "email" not in data or not data["email"]:
            return jsonify({"message": "Email is required"}), 400

        email = str(data["email"]).strip().lower()

        try:
            user = users.find_one({"email": email})
        except PyMongoError:
            return jsonify({"message": "Database connection failed. Check MONGO_URI."}), 500

        # Same message whether or not user exists (avoid email enumeration),
        # but only send mail when the account is real.
        generic_ok = {
            "success": True,
            "message": "If that email is registered, a password reset link has been sent. Check your inbox.",
        }

        if not user:
            return jsonify(generic_ok)

        if not resend_api_key:
            return jsonify({
                "message": "RESEND_API_KEY missing in growsmartback/.env. Add it and restart the backend."
            }), 503

        raw_token = secrets.token_urlsafe(32)
        token_hash = _hash_reset_token(raw_token)
        expires = datetime.datetime.utcnow() + datetime.timedelta(hours=1)

        try:
            users.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "reset_token_hash": token_hash,
                        "reset_token_expires": expires,
                    }
                },
            )
        except PyMongoError:
            return jsonify({"message": "Database connection failed"}), 500

        base = (frontend_url or "http://localhost:3000").rstrip("/")
        reset_link = f"{base}/reset-password?token={raw_token}"

        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;padding:24px;color:#111;">
          <h2 style="color:#0056d2;">Reset your GrowSmart password</h2>
          <p>Hi {user.get('name') or 'there'},</p>
          <p>We received a request to reset your password. Click the button below
          (valid for <strong>1 hour</strong>):</p>
          <p style="margin:28px 0;">
            <a href="{reset_link}"
               style="background:#0056d2;color:#fff;padding:12px 20px;border-radius:10px;
                      text-decoration:none;font-weight:bold;display:inline-block;">
              Set new password
            </a>
          </p>
          <p style="font-size:13px;color:#555;">Or copy this link:<br/>
            <a href="{reset_link}">{reset_link}</a>
          </p>
          <p style="font-size:13px;color:#777;">If you did not ask for this, ignore this email.</p>
        </div>
        """

        ok, err = _send_resend_email(
            to_email=email,
            subject="GrowSmart — reset your password",
            html=html,
        )
        if not ok:
            return jsonify({"message": err or "Failed to send email"}), 502

        return jsonify(generic_ok)

    @auth_bp.route("/reset-password", methods=["POST"])
    def reset_password():
        """
        Body: { token, password, confirm_password }
        Validates token, then stores bcrypt hash for that user's email in MongoDB.
        """
        data = request.json or {}
        raw_token = str(data.get("token") or "").strip()
        password = data.get("password")
        confirm = data.get("confirm_password") or data.get("confirmPassword")

        if not raw_token:
            return jsonify({"message": "Reset token is required"}), 400
        if not password or not confirm:
            return jsonify({"message": "Password and confirm password are required"}), 400
        if str(password) != str(confirm):
            return jsonify({"message": "Passwords do not match"}), 400
        if len(str(password)) < 6:
            return jsonify({"message": "Password must be at least 6 characters"}), 400

        token_hash = _hash_reset_token(raw_token)
        now = datetime.datetime.utcnow()

        try:
            user = users.find_one({
                "reset_token_hash": token_hash,
                "reset_token_expires": {"$gt": now},
            })
        except PyMongoError:
            return jsonify({"message": "Database connection failed"}), 500

        if not user:
            return jsonify({
                "message": "Invalid or expired reset link. Please request a new one."
            }), 400

        hashed_pw = bcrypt.hashpw(str(password).encode("utf-8"), bcrypt.gensalt())

        try:
            users.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {"password": hashed_pw},
                    "$unset": {
                        "reset_token_hash": "",
                        "reset_token_expires": "",
                    },
                },
            )
        except PyMongoError:
            return jsonify({"message": "Database connection failed"}), 500

        return jsonify({
            "success": True,
            "message": "Password updated successfully. You can sign in now.",
        })

    return auth_bp
