from flask import Flask, render_template, request
import os
import joblib
import pandas as pd

from features import extract_features

app = Flask(__name__)

MODELS_DIR = "models"
MODEL_NAME = "svm_model.pkl"   # or "gbc_model.pkl" if you prefer that model

# Load model once when the server starts
model_path = os.path.join(MODELS_DIR, MODEL_NAME)
model = joblib.load(model_path)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            # Extract features for this URL
            feats = extract_features(url)
            X = pd.DataFrame([feats])

            pred = model.predict(X)[0]
            prediction = int(pred)

            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(X)[0][prediction])

    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    # debug=True is fine for development
    app.run(debug=True)
