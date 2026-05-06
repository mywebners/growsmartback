from flask import Blueprint, request, jsonify
import pickle
import numpy as np

from utils.preprocess import map_subjects_to_pslots

predict_bp = Blueprint("predict", __name__)

model = pickle.load(open("model/model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))


@predict_bp.route("/predict-career", methods=["POST"])
def predict_career():
    try:
        data = request.json

        marks = data.get("marks")
        skills = data.get("skills")

        p_slots = map_subjects_to_pslots(marks)

        skill_order = [
            "Linguistic",
            "Musical",
            "Bodily",
            "Logical - Mathematical",
            "Spatial-Visualization",
            "Interpersonal",
            "Intrapersonal",
            "Naturalist"
        ]

        input_data = []

        for i in range(1, 9):
            input_data.append(p_slots[f"P{i}"])

        for skill in skill_order:
            input_data.append(skills.get(skill, 0))

        input_array = np.array(input_data).reshape(1, -1)

        probs = model.predict_proba(input_array)[0]

        top_3_idx = probs.argsort()[-3:][::-1]
        top_3_careers = encoder.inverse_transform(top_3_idx)

        return jsonify({
            "success": True,
            "careers": top_3_careers.tolist()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })