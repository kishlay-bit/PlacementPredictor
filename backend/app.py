import json
import math
import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ML"
MODEL_DIR = ML_DIR / "03_models"
sys.path.insert(0, str(ML_DIR / "05_app_ready"))

from ml_utils import PlacementPredictor  # noqa: E402

predictor = PlacementPredictor(MODEL_DIR)
schema = predictor.schema
model_card = json.loads((MODEL_DIR / "model_card.json").read_text())
FEATURES = schema["input_features"]

app = Flask(__name__)


def fmt(x):
    return f"{x:g}"


def field_list():
    out = []
    for name in FEATURES:
        meta = schema["features"][name]
        item = {"name": name, "label": meta["label"], "type": meta["type"]}
        if meta["type"] == "categorical":
            item["options"] = meta["options"]
        else:
            item["min"], item["max"] = meta["min"], meta["max"]
        out.append(item)
    return out


def validate(payload):
    clean, errors = {}, {}
    for name in FEATURES:
        meta = schema["features"][name]
        raw = payload.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue
        if meta["type"] == "categorical":
            if raw not in meta["options"]:
                errors[name] = "Choose one of the listed options."
            else:
                clean[name] = raw
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a number."
            continue
        if not math.isfinite(value):
            errors[name] = "Enter a valid number."
        elif value < meta["min"] or value > meta["max"]:
            errors[name] = f"Must be between {fmt(meta['min'])} and {fmt(meta['max'])}."
        elif meta["type"] == "integer" and value != int(value):
            errors[name] = "Enter a whole number."
        else:
            clean[name] = int(value) if meta["type"] == "integer" else value
    return clean, errors


@app.get("/api/schema")
def get_schema():
    return jsonify({"fields": field_list()})


@app.get("/api/model-info")
def get_model_info():
    signal = model_card.get("placement_signal", {})
    return jsonify({
        "training_rows": model_card.get("n_rows"),
        "reliability_note": signal.get("reliability_note"),
    })


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    clean, errors = validate(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(predictor.predict(clean))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
